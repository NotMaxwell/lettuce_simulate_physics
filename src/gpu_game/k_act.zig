//! Player actions on GPU balls: pick/grab/release, lasso kill, explosion knockback.
//! Bindings: ro 0 pos, 1 poly (lasso points), 2 blasts;
//!           rw 0 vel, 1 body, 2 meta, 3 pick, 4 out, 5 heat, 6 cand (gpu_sim event buffer).
const c = @import("common.zig");
const L = c.L;

const pos = c.ro(L.Vec2, 0, "pos");
const poly = c.ro(L.Vec2, 1, "poly");
const blasts = c.ro(c.Blast, 2, "blasts");
const vel = c.rw(L.Vec2, 0, "vel");
const body = c.rw(L.BallBody, 1, "body");
const meta = c.rw(L.BallMeta, 2, "meta");
const pick_buf = c.rw(u32, 3, "pick");
const out = c.rw(L.Event, 4, "out");
const heat = c.rw(L.BallHeat, 5, "heat");
const cand = c.rw(L.Event, 6, "cand");
const cfg = c.cfg;

export fn clearPick() callconv(c.kernel_cc) void {
    if (c.gid() == 0) pick_buf.v[0] = 0;
}

/// Ball under the mouse into pick[0] as (closeness << 20 | id) + 1 via atomic max, so the ball
/// whose centre is relatively nearest wins (physics.zig pickBall uses absolute distance).
export fn pick() callconv(c.kernel_cc) void {
    const i = c.gid();
    const P = c.params;
    if (i >= P.ball_count) return;
    if (meta.v[i] & L.flag_alive == 0) return;
    const p = pos.v[i];
    const r = body.v[i].radius;
    const dx = P.mouse_x - p.x;
    const dy = P.mouse_y - p.y;
    const q = (dx * dx + dy * dy) / (r * r);
    if (!(q <= 1.0)) return;
    const close: u32 = @intFromFloat((1.0 - q) * 2046.0);
    _ = c.atomicUMax(&pick_buf.v[0], ((close + 1) << c.id_bits) | i);
}

export fn grab() callconv(c.kernel_cc) void {
    if (c.gid() != 0) return;
    const k = pick_buf.v[0];
    if (k == 0) return;
    const i = k & c.id_mask;
    var m = meta.v[i] | L.flag_grabbed;
    // Ball.onClick -> update_temp: a click cools by 150 and anything below ignition resets.
    var t = heat.v[i].temp;
    if (t > cfg.room_temp) t = @max(cfg.room_temp, t - 150.0);
    if (t < cfg.ignition_temp) {
        t = cfg.room_temp;
        m &= ~L.flag_burning;
    }
    heat.v[i].temp = t;
    meta.v[i] = m;
    // No follow step yet: a release before the next integrate adds nothing.
    cand.v[0].b = 0;
    cand.v[0].value = 0;
}

/// physics.zig releaseGrabs: the throw also gets the last follow step's motion (integrate leaves
/// it in the candidate list's header), as Ball.releaseGrab adds it to the speed.
export fn release() callconv(c.kernel_cc) void {
    const i = c.gid();
    if (i >= c.params.ball_count) return;
    const m = meta.v[i];
    if (m & L.flag_grabbed == 0) return;
    const v = vel.v[i];
    vel.v[i] = .{ .x = v.x + @as(f32, @bitCast(cand.v[0].b)), .y = v.y + cand.v[0].value };
    meta.v[i] = (m & ~L.flag_grabbed) | L.flag_thrown;
}

/// physics.zig lassoExplode, kill half: every live ball whose centre is inside the polygon dies
/// and is reported (a = id, b = x bits, value = y) so the CPU spawns its explosion. The knockback
/// is applied by `explode` on a later tick.
export fn lasso() callconv(c.kernel_cc) void {
    const i = c.gid();
    const P = c.params;
    if (i >= P.ball_count) return;
    const m = meta.v[i];
    if (m & L.flag_alive == 0) return;
    const p = pos.v[i];
    if (p.x < P.lo_x or p.x > P.hi_x or p.y < P.lo_y or p.y > P.hi_y) return;
    var inside = false;
    var j = P.items - 1;
    var k: u32 = 0;
    while (k < P.items) : (k += 1) {
        const a = poly.v[k];
        const b = poly.v[j];
        if ((a.y > p.y) != (b.y > p.y)) {
            const xi = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (p.x < xi) inside = !inside;
        }
        j = k;
    }
    if (!inside) return;
    const slot = c.atomicAdd(&out.v[0].kind, 1);
    if (slot >= P.out_cap) {
        _ = c.atomicAdd(&out.v[0].a, 1);
        return;
    }
    meta.v[i] = m & ~(L.flag_alive | L.flag_grabbed);
    body.v[i].mass = 0;
    out.v[slot + 1] = .{ .kind = L.event_death, .a = i, .b = @bitCast(p.x), .value = p.y };
}

/// effects.knockback for up to `items` blasts at once: impulse, mass-scaled heat, and balls
/// within explosion_fire_frac of the blast radius are set to at least flame temperature (the
/// `burn` pass turns that into burning).
export fn explode() callconv(c.kernel_cc) void {
    const i = c.gid();
    const P = c.params;
    if (i >= P.ball_count) return;
    if (meta.v[i] & L.flag_alive == 0) return;
    const p = pos.v[i];
    const inv_m = 1.0 / body.v[i].mass;
    var v = vel.v[i];
    var t = heat.v[i].temp;
    const heat_k = cfg.base_explosion_heat * 100.0;
    var k: u32 = 0;
    while (k < P.items) : (k += 1) {
        const e = blasts.v[k];
        const dx = p.x - e.x;
        const dy = p.y - e.y;
        const d = @sqrt(dx * dx + dy * dy);
        if (d > 0 and d <= e.blast) {
            const s = e.force * inv_m * (1.0 - d / e.blast) / d;
            v.x += dx * s;
            v.y += dy * s;
            t += heat_k * inv_m * (1.0 - d / e.blast);
            if (d <= e.blast * cfg.explosion_fire_frac) t = @max(t, cfg.flame_temp);
        }
    }
    vel.v[i] = v;
    heat.v[i].temp = t;
}
