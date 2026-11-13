//! Advances one particle pool (layout in k_emit.zig): fire rises and expires, ash falls, bounces
//! off the play-area walls and fades. The host binds the fire or the ash pool.
//! Bindings: rw 0 pos, 1 prev_pos, 2 body, 3 state, 4 heat, 5 meta, 6 fx.
//! Deviation: ashes do not collide with each other (the CPU version sorts them into a grid);
//! they overlap where they pile up.
const c = @import("common.zig");
const L = c.L;
const cfg = c.cfg;

const pos = c.rw(L.Vec2, 0, "pos");
const prev = c.rw(L.Vec2, 1, "prev");
const body = c.rw(L.BallBody, 2, "body");
const state = c.rw(L.BallState, 3, "state");
const heat = c.rw(L.BallHeat, 4, "heat");
const meta = c.rw(L.BallMeta, 5, "meta");
const fx = c.rw(u32, 6, "fx");

/// Adds the pool's alive count to fx slot `items`.
inline fn count(alive: u32) void {
    const total = c.blockSum(alive);
    if (c.lid() == 0 and total != 0) _ = c.atomicAdd(&fx.v[c.params.items], total);
}

/// effects.updateParticles: move one step, die once age > life.
export fn fireUpdate() callconv(c.kernel_cc) void {
    const i = c.gid();
    var alive: u32 = 0;
    if (i <= c.params.pool_mask and meta.v[i] & L.flag_alive != 0) {
        var s = state.v[i];
        s.ang_vel += 1.0;
        if (s.ang_vel > c.fire_life) {
            meta.v[i] = 0;
        } else {
            const p = pos.v[i];
            prev.v[i] = p;
            pos.v[i] = .{ .x = p.x + s.angle, .y = p.y + s.prev_angle };
            state.v[i] = s;
            alive = 1;
        }
    }
    count(alive);
}

/// effects.updateAshes without ash-ash contacts: gravity, ground friction, walls, fade.
/// Fields: state = {vx, vy, age, gravity}, body = {radius, start red, retention, friction}.
export fn ashUpdate() callconv(c.kernel_cc) void {
    const P = c.params;
    const i = c.gid();
    var alive: u32 = 0;
    if (i <= P.pool_mask and meta.v[i] & L.flag_alive != 0) {
        var s = state.v[i];
        const b = body.v[i];
        var p = pos.v[i];
        prev.v[i] = p;
        const floor_y = P.hi_y - cfg.ash_radius - cfg.wall / 2;
        const left_x = P.lo_x + cfg.ash_radius + cfg.wall / 2;
        const right_x = P.hi_x - cfg.ash_radius - cfg.wall / 2;
        var vx = s.angle;
        var vy = s.prev_angle;
        if (p.y < floor_y) vy += s.gravity;
        if (vy == 0 and vx != 0) {
            vx += if (vx > 0) -b.friction else b.friction;
            if (@abs(vx) < b.friction) vx = 0;
        }
        p.x += vx;
        p.y += vy;
        s.ang_vel += 1.0;
        if (p.y > floor_y) {
            p.y = floor_y;
            const rebound = vy * b.retention - s.gravity;
            vy = if (rebound > cfg.bounce_stop) -rebound else 0;
        }
        if (p.x < left_x) {
            p.x = left_x;
            if (vx < 0) {
                vx *= -b.retention;
                if (@abs(vx) < cfg.bounce_stop) vx = 0;
            }
        } else if (p.x > right_x) {
            p.x = right_x;
            if (vx > 0) {
                vx *= -b.retention;
                if (@abs(vx) < cfg.bounce_stop) vx = 0;
            }
        }
        if (s.ang_vel >= c.ash_life) {
            meta.v[i] = 0;
        } else {
            const fade = 1.0 - @min(s.ang_vel / c.ash_life, 1.0);
            s.angle = vx;
            s.prev_angle = vy;
            state.v[i] = s;
            pos.v[i] = p;
            heat.v[i] = .{ .temp = @floor(b.mass * fade) / 255.0, .burn = 0, ._pad0 = 0, ._pad1 = 1.0 };
            alive = 1;
        }
    }
    count(alive);
}
