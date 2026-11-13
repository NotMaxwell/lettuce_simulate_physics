//! Spawns particles from balls into a ring pool. Pool entries use the ball buffer layout so the
//! ball shader can draw them (meta has flag_particle; the heat record holds r, g, b, a):
//!   pos / prev_pos: centre   body: radius, mass = ash start red, retention, friction
//!   state: vx, vy, age, gravity (angle fields are unused for particles)
//! `emitFire` and `emitAsh` share one layout; the host binds the fire or the ash pool.
//! Bindings: ro 0 pos, 1 body, 2 vel;
//!           rw 0 ball meta, 1 pool pos, 2 pool prev_pos, 3 pool body, 4 pool state, 5 pool heat,
//!           6 pool meta, 7 fx.
//! A full ring overwrites its oldest entries; the per-ball keep probability below makes that
//! rare by thinning emission when the steady-state demand would exceed the pool.
const c = @import("common.zig");
const L = c.L;
const cfg = c.cfg;

const pos = c.ro(L.Vec2, 0, "pos");
const body = c.ro(L.BallBody, 1, "body");
const vel = c.ro(L.Vec2, 2, "vel");
const meta = c.rw(L.BallMeta, 0, "meta");
const q_pos = c.rw(L.Vec2, 1, "q_pos");
const q_prev = c.rw(L.Vec2, 2, "q_prev");
const q_body = c.rw(L.BallBody, 3, "q_body");
const q_state = c.rw(L.BallState, 4, "q_state");
const q_heat = c.rw(L.BallHeat, 5, "q_heat");
const q_meta = c.rw(L.BallMeta, 6, "q_meta");
const fx = c.rw(u32, 7, "fx");

const particle_meta: u32 = c.flag_particle | L.flag_alive | 1;

/// effects.emitFire: one particle per burning ball per tick, anywhere inside the ball, drifting
/// up. With a 20-tick life the pool holds about 20 per burning ball, so beyond cap/20 burning
/// balls each ball emits with probability cap/(20 * burning).
export fn emitFire() callconv(c.kernel_cc) void {
    const P = c.params;
    const i = c.gid();
    if (i >= P.ball_count) return;
    const m = meta.v[i];
    if (m & (L.flag_alive | L.flag_burning) != (L.flag_alive | L.flag_burning)) return;
    const burning: f32 = @floatFromInt(fx.v[c.fx_prev_burning]);
    const keep = @min(1.0, @as(f32, @floatFromInt(P.pool_mask + 1)) / (c.fire_life * @max(burning, 1.0)));
    var rng = c.hash(i *% 0x2545f491 +% P.stamp *% 0x9e3779b1);
    if (c.next(&rng) >= keep) return;
    const p = pos.v[i];
    const r = body.v[i].radius;
    const size = 2.0 + @floor(c.next(&rng) * 9.0);
    const x = p.x + (c.next(&rng) * 2.0 - 1.0) * r + size * 0.5;
    const y = p.y + (c.next(&rng) * 2.0 - 1.0) * r + size * 0.5;
    const slot = c.atomicAdd(&fx.v[c.fx_fire_head], 1) & P.pool_mask;
    q_pos.v[slot] = .{ .x = x, .y = y };
    q_prev.v[slot] = .{ .x = x, .y = y };
    q_body.v[slot] = .{ .radius = size * 0.5, .mass = 0, .retention = 0, .friction = 0 };
    q_state.v[slot] = .{ .angle = @floor(c.next(&rng) * 3.0) - 1.0, .prev_angle = cfg.fire_particle_vy, .ang_vel = 0, .gravity = 0 };
    q_heat.v[slot] = .{ .temp = 1.0, .burn = c.next(&rng) * (140.0 / 255.0), ._pad0 = 0, ._pad1 = 1.0 };
    q_meta.v[slot] = particle_meta;
}

/// Ball.increase_burn's ash spawn for balls `burn` just retired: one ash per 8 px grid point
/// inside the ball, thinned when many balls die at once so the pool keeps ~180 ticks of them.
export fn emitAsh() callconv(c.kernel_cc) void {
    const P = c.params;
    const i = c.gid();
    if (i >= P.ball_count) return;
    const m = meta.v[i];
    if (m & c.flag_ashing == 0) return;
    meta.v[i] = m & ~c.flag_ashing;
    const deaths: f32 = @floatFromInt(fx.v[c.fx_prev_deaths]);
    const per_ball = 70.0;
    const keep = @min(1.0, @as(f32, @floatFromInt(P.pool_mask + 1)) / (c.ash_life * per_ball * @max(deaths, 1.0)));
    const p = pos.v[i];
    const v = vel.v[i];
    const radius = body.v[i].radius;
    const r2 = radius * radius;
    const step: f32 = cfg.ash_spawn_step;
    const n_axis: u32 = @as(u32, @intFromFloat(@floor(radius))) * 2 / cfg.ash_spawn_step + 1;
    var rng = c.hash(i *% 0x2545f491 +% P.stamp *% 0xc2b2ae35);
    var ix: u32 = 0;
    while (ix < n_axis) : (ix += 1) {
        var iy: u32 = 0;
        while (iy < n_axis) : (iy += 1) {
            const dx = @floor(radius) * -1.0 + @as(f32, @floatFromInt(ix)) * step;
            const dy = @floor(radius) * -1.0 + @as(f32, @floatFromInt(iy)) * step;
            if (dx * dx + dy * dy > r2) continue;
            if (keep < 1.0 and c.next(&rng) >= keep) continue;
            const slot = c.atomicAdd(&fx.v[c.fx_ash_head], 1) & P.pool_mask;
            const x = p.x + dx;
            const y = p.y + dy;
            q_pos.v[slot] = .{ .x = x, .y = y };
            q_prev.v[slot] = .{ .x = x, .y = y };
            const retention = 0.7 + c.next(&rng) * 0.25;
            const friction = 0.05 + c.next(&rng) * 0.25;
            const red = @floor(1.0 + c.next(&rng) * 254.0);
            q_body.v[slot] = .{ .radius = cfg.ash_radius, .mass = red, .retention = retention, .friction = friction };
            q_state.v[slot] = .{
                .angle = v.x + c.next(&rng) * 2.0 - 1.0,
                .prev_angle = v.y + c.next(&rng) - 1.0,
                .ang_vel = 0,
                .gravity = 0.1 + c.next(&rng) * 0.9,
            };
            q_heat.v[slot] = .{ .temp = red / 255.0, .burn = 0, ._pad0 = 0, ._pad1 = 1.0 };
            q_meta.v[slot] = particle_meta;
        }
    }
}
