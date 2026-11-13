//! Advances one particle pool (layout in k_emit.zig): fire rises and expires, ash falls, bounces
//! off the play-area walls and fades. The host binds the fire or the ash pool.
//! Bindings: rw 0 pos, 1 prev_pos, 2 body, 3 state, 4 heat, 5 meta, 6 fx.
//! Ash-ash contacts: ashBin hashes every live ash into a spatial hash (a linked list per cell, so
//! no sort), ashCollide resolves each ash against its neighbours. Deviation from the CPU: the pass
//! is Jacobi (every ash applies only its own half of each pair, reading neighbours' positions as
//! they happen to be), and an ash looks at no more than `ash_scan_cap` neighbours, so a crush
//! costs O(n) and separates over a few ticks.
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
// Only the ash-contact kernels use these two, bound in place of prev_pos and body (bindings 1, 2).
const heads = c.rw(u32, 1, "heads");
const next = c.rw(u32, 2, "next");

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

const ash_cell: f32 = 2.0 * cfg.ash_radius;
const ash_scan_cap: u32 = 32;

/// Cell of a position, hashed into the table (items = table size - 1, a power of two minus one).
inline fn ashSlot(x: f32, y: f32) u32 {
    const cx: i32 = @intFromFloat(@floor((x - c.params.lo_x) / ash_cell));
    const cy: i32 = @intFromFloat(@floor((y - c.params.lo_y) / ash_cell));
    return ((@as(u32, @bitCast(cx)) *% 73856093) ^ (@as(u32, @bitCast(cy)) *% 19349663)) & c.params.items;
}

export fn ashClear() callconv(c.kernel_cc) void {
    const i = c.gid();
    if (i <= c.params.items) heads.v[i] = 0;
}

/// Pushes every live ash onto the list of its cell (entries are ash index + 1, 0 = empty).
export fn ashBin() callconv(c.kernel_cc) void {
    const i = c.gid();
    if (i > c.params.pool_mask or meta.v[i] & L.flag_alive == 0) return;
    const p = pos.v[i];
    next.v[i] = c.atomicExchange(&heads.v[ashSlot(p.x, p.y)], i + 1);
}

/// handle_ash_collisions for ash i: against every ash in the 3x3 cells around it, apply this
/// ash's half of the push-apart and of the velocity exchange (the neighbour does its own half).
export fn ashCollide() callconv(c.kernel_cc) void {
    const P = c.params;
    const i = c.gid();
    if (i > P.pool_mask or meta.v[i] & L.flag_alive == 0) return;
    const p0 = pos.v[i];
    var px = p0.x;
    var py = p0.y;
    var vx = state.v[i].angle;
    var vy = state.v[i].prev_angle;
    const min_d = 2.0 * cfg.ash_radius;
    var seen: u32 = 0;
    var oy: i32 = -1;
    while (oy <= 1) : (oy += 1) {
        var ox: i32 = -1;
        while (ox <= 1) : (ox += 1) {
            var j = heads.v[ashSlot(p0.x + @as(f32, @floatFromInt(ox)) * ash_cell, p0.y + @as(f32, @floatFromInt(oy)) * ash_cell)];
            while (j != 0 and seen < ash_scan_cap) : (j = next.v[j - 1]) {
                seen += 1;
                const q = j - 1;
                if (q == i) continue;
                const o = pos.v[q];
                const dx = o.x - px;
                const dy = o.y - py;
                const d2 = dx * dx + dy * dy;
                if (d2 >= min_d * min_d) continue;
                const d = @sqrt(d2);
                // The CPU skips d2 == 0 in the Python original; coincident ashes are pushed along an
                // index-dependent direction so a pile clamped into one spot still spreads.
                var nx: f32 = undefined;
                var ny: f32 = undefined;
                if (d2 > 0) {
                    nx = dx / d;
                    ny = dy / d;
                } else {
                    const a: f32 = @floatFromInt((i +% q) % 64);
                    nx = @cos(a);
                    ny = @sin(a);
                }
                const half = (min_d - d) * 0.5;
                px -= nx * half;
                py -= ny * half;
                const ov = state.v[q];
                const norm_v = (ov.angle - vx) * nx + (ov.prev_angle - vy) * ny;
                if (norm_v > 0) continue;
                const pw = -norm_v * 0.5;
                vx -= pw * nx;
                vy -= pw * ny;
            }
        }
    }
    if (px == p0.x and py == p0.y) return;
    // Back inside the walls: ashUpdate already clamped this tick, the push may have left them.
    pos.v[i] = .{
        .x = @min(@max(px, P.lo_x + cfg.ash_radius + cfg.wall / 2), P.hi_x - cfg.ash_radius - cfg.wall / 2),
        .y = @min(py, P.hi_y - cfg.ash_radius - cfg.wall / 2),
    };
    state.v[i].angle = vx;
    state.v[i].prev_angle = vy;
}
