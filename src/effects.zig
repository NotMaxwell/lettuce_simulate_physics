//! Fire particles, ash and explosions (fire_effect.py, particle.py, ash.py, explosion_effect.py).
const std = @import("std");
const cfg = @import("config.zig");
const world = @import("world.zig");
const World = world.World;

const V = std.simd.suggestVectorLength(f32) orelse 8;
const F = @Vector(V, f32);

/// Sorted grid for ash-ash contacts. Cell size equals the contact distance, so every
/// touching pair lies in the same or an adjacent cell (Python used cell_size=1 with
/// radius 2 and missed most contacts). Ashes are radix-sorted by cell and swept in that
/// order, which keeps every access sequential.
pub const AshGrid = struct {
    key: std.ArrayList(u32) = .empty,
    idx: std.ArrayList(u32) = .empty,
    key_tmp: std.ArrayList(u32) = .empty,
    idx_tmp: std.ArrayList(u32) = .empty,
    sorted: std.MultiArrayList(Body) = .empty,

    const Body = struct { x: f32, y: f32, vx: f32, vy: f32 };
    const cell_size = 2 * cfg.ash_radius;
    const radix_bits = 11;

    pub fn deinit(g: *AshGrid, gpa: std.mem.Allocator) void {
        inline for (.{ "key", "idx", "key_tmp", "idx_tmp", "sorted" }) |name| @field(g, name).deinit(gpa);
    }

    fn reserve(g: *AshGrid, gpa: std.mem.Allocator) !void {
        inline for (.{ "key", "idx", "key_tmp", "idx_tmp", "sorted" }) |name| try @field(g, name).ensureTotalCapacity(gpa, cfg.max_ashes);
    }
};

/// Pools are reserved to their caps once, so emitting never allocates mid-game.
pub fn reservePools(w: *World, grid: *AshGrid) !void {
    try w.particles.ensureTotalCapacity(w.gpa, cfg.max_particles);
    try w.ashes.ensureTotalCapacity(w.gpa, cfg.max_ashes);
    try grid.reserve(w.gpa);
}

fn randRange(r: std.Random, lo: f32, hi: f32) f32 {
    return lo + r.float(f32) * (hi - lo);
}

/// One particle per burning ball per tick; with a 20-tick life that is the same
/// ~20 particles per ball the Python FireEffect kept alive.
pub fn emitFire(w: *World) void {
    const r = w.prng.random();
    const s = w.balls.slice();
    for (s.items(.flags), s.items(.x), s.items(.y), s.items(.radius)) |f, x, y, rad| {
        if (!f.alive or !f.burning) continue;
        if (w.particles.len >= @min(w.particles.capacity, cfg.max_particles)) return;
        const ri: i32 = @intFromFloat(rad);
        const xi: i32 = @intFromFloat(x);
        const yi: i32 = @intFromFloat(y);
        w.particles.appendAssumeCapacity(.{
            .x = @floatFromInt(r.intRangeAtMost(i32, xi - ri, xi + ri)),
            .y = @floatFromInt(r.intRangeAtMost(i32, yi - ri, yi + ri)),
            .vx = @floatFromInt(r.intRangeAtMost(i8, -1, 1)),
            .vy = cfg.fire_particle_vy,
            .size = @floatFromInt(r.intRangeAtMost(u8, 2, 10)),
            .age = 0,
            .life = cfg.fire_particle_life,
            .r = 255,
            .g = r.intRangeAtMost(u8, 0, 140),
            .b = 0,
        });
    }
}

pub fn updateParticles(w: *World) void {
    const s = w.particles.slice();
    const xs = s.items(.x);
    const ys = s.items(.y);
    const vxs = s.items(.vx);
    const vys = s.items(.vy);
    const ages = s.items(.age);
    const lives = s.items(.life);
    for (xs, ys, vxs, vys, ages) |*x, *y, vx, vy, *age| {
        age.* += 1;
        x.* += vx;
        y.* += vy;
    }
    var i = w.particles.len;
    while (i > 0) {
        i -= 1;
        if (ages[i] > lives[i]) w.particles.swapRemove(i);
    }
}

/// Mirrors Ball.increase_burn's ash spawn: one ash per 8 px grid point inside the ball.
/// Near the pool cap each candidate is kept with probability room/count instead.
pub fn spawnAshes(w: *World, x: f32, y: f32, vx: f32, vy: f32, radius: f32) void {
    const r = w.prng.random();
    const ri: i32 = @intFromFloat(radius);
    const r2 = radius * radius;
    const per_axis: f32 = @floatFromInt(@divFloor(2 * ri, cfg.ash_spawn_step) + 1);
    const estimate = per_axis * per_axis * (std.math.pi / 4.0);
    const room: f32 = @floatFromInt(@min(w.ashes.capacity, cfg.max_ashes) -| w.ashes.len);
    const keep = @min(1.0, room / @max(estimate, 1));
    var dx: i32 = -ri;
    while (dx <= ri) : (dx += cfg.ash_spawn_step) {
        var dy: i32 = -ri;
        while (dy <= ri) : (dy += cfg.ash_spawn_step) {
            const fdx: f32 = @floatFromInt(dx);
            const fdy: f32 = @floatFromInt(dy);
            if (fdx * fdx + fdy * fdy > r2) continue;
            if (keep < 1 and r.float(f32) >= keep) continue;
            if (w.ashes.len >= @min(w.ashes.capacity, cfg.max_ashes)) return;
            w.ashes.appendAssumeCapacity(.{
                .x = x + fdx,
                .y = y + fdy,
                .prev_x = x + fdx,
                .prev_y = y + fdy,
                .vx = vx + randRange(r, -1, 1),
                .vy = vy + randRange(r, -1, 0),
                .gravity = randRange(r, 0.1, 1),
                .retention = randRange(r, 0.7, 0.95),
                .friction = randRange(r, 0.05, 0.3),
                .start_red = r.intRangeAtMost(u8, 1, 255),
            });
        }
    }
}

/// Ash.apply_gravity_and_velocity, update_position, increase_burn, handle_ash_collisions, bounds.
pub fn updateAshes(w: *World, grid: *AshGrid) void {
    const s = w.ashes.slice();
    const n = w.ashes.len;
    const xs = s.items(.x);
    const ys = s.items(.y);
    const vxs = s.items(.vx);
    const vys = s.items(.vy);
    const b = w.bounds;
    const floor_y = b.y + b.h - cfg.ash_radius - cfg.wall / 2;
    const left_x = b.x + cfg.ash_radius + cfg.wall / 2;
    const right_x = b.x + b.w - cfg.ash_radius - cfg.wall / 2;

    for (xs, ys, vxs, vys, s.items(.prev_x), s.items(.prev_y), s.items(.gravity), s.items(.friction), s.items(.age)) |*x, *y, *vx, *vy, *px, *py, g, fr, *age| {
        px.* = x.*;
        py.* = y.*;
        if (y.* < floor_y) vy.* += g;
        if (vy.* == 0 and vx.* != 0) {
            vx.* += if (vx.* > 0) -fr else fr;
            if (@abs(vx.*) < fr) vx.* = 0;
        }
        x.* += vx.*;
        y.* += vy.*;
        age.* += 1;
    }

    if (n > 1 and n <= grid.sorted.capacity) collideAshes(grid, b, xs[0..n], ys[0..n], vxs[0..n], vys[0..n]);

    for (xs, ys, vxs, vys, s.items(.gravity), s.items(.retention)) |*x, *y, *vx, *vy, g, ret| {
        if (y.* > floor_y) {
            y.* = floor_y;
            const rebound = vy.* * ret - g;
            vy.* = if (rebound > cfg.bounce_stop) -rebound else 0;
        }
        if (x.* < left_x) {
            x.* = left_x;
            if (vx.* < 0) {
                vx.* *= -ret;
                if (@abs(vx.*) < cfg.bounce_stop) vx.* = 0;
            }
        } else if (x.* > right_x) {
            x.* = right_x;
            if (vx.* > 0) {
                vx.* *= -ret;
                if (@abs(vx.*) < cfg.bounce_stop) vx.* = 0;
            }
        }
    }

    const ages = s.items(.age);
    var i = n;
    while (i > 0) {
        i -= 1;
        if (ages[i] >= cfg.ash_lifetime_ticks) w.ashes.swapRemove(i);
    }
}

/// Sweeps the cell-sorted ashes once. Each ash checks later ashes in its own cell and the
/// cell to its right, then the three cells below (found with a cursor that only moves
/// forward), so each pair is seen once. Both scans stop after `scan_cap` candidates, so
/// ashes crushed into one spot by a shrinking wall cost O(n) instead of O(n^2); they still
/// separate over a few ticks.
fn collideAshes(grid: *AshGrid, bounds: world.Rect, xs: []f32, ys: []f32, vxs: []f32, vys: []f32) void {
    const n: u32 = @intCast(xs.len);
    const margin = 4 * AshGrid.cell_size;
    const cols: u32 = @intFromFloat(@ceil((bounds.w + 2 * margin) / AshGrid.cell_size));
    const rows: u32 = @intFromFloat(@ceil((bounds.h + 2 * margin) / AshGrid.cell_size));
    const ox = bounds.x - margin;
    const oy = bounds.y - margin;
    const fcols: f32 = @floatFromInt(cols - 1);
    const frows: f32 = @floatFromInt(rows - 1);
    if (@as(u64, cols) * rows >= std.math.maxInt(u32) - 2 * cols) return;

    inline for (.{ &grid.key, &grid.idx, &grid.key_tmp, &grid.idx_tmp }) |list| list.items.len = n;
    var key = grid.key.items;
    var idx = grid.idx.items;
    var key_tmp = grid.key_tmp.items;
    var idx_tmp = grid.idx_tmp.items;
    for (key, idx, xs, ys, 0..) |*k, *ix, x, y, i| {
        const cx: u32 = @intFromFloat(std.math.clamp(@floor((x - ox) / AshGrid.cell_size), 0, fcols));
        const cy: u32 = @intFromFloat(std.math.clamp(@floor((y - oy) / AshGrid.cell_size), 0, frows));
        k.* = cy * cols + cx;
        ix.* = @intCast(i);
    }

    // LSD radix sort on as many 11-bit digits as the largest key needs.
    const max_key = cols * rows;
    var shift: u5 = 0;
    while (true) {
        var count: [1 << AshGrid.radix_bits]u32 = @splat(0);
        const mask = (1 << AshGrid.radix_bits) - 1;
        for (key) |k| count[(k >> shift) & mask] += 1;
        var sum: u32 = 0;
        for (&count) |*c| {
            const t = c.*;
            c.* = sum;
            sum += t;
        }
        for (key, idx) |k, ix| {
            const d = (k >> shift) & mask;
            key_tmp[count[d]] = k;
            idx_tmp[count[d]] = ix;
            count[d] += 1;
        }
        std.mem.swap([]u32, &key, &key_tmp);
        std.mem.swap([]u32, &idx, &idx_tmp);
        if (@as(u64, max_key) >> shift >> AshGrid.radix_bits == 0) break;
        shift += AshGrid.radix_bits;
    }

    std.debug.assert(n <= grid.sorted.capacity);
    grid.sorted.len = n;
    const s = grid.sorted.slice();
    const sx = s.items(.x);
    const sy = s.items(.y);
    const svx = s.items(.vx);
    const svy = s.items(.vy);
    for (idx, sx, sy, svx, svy) |ix, *x, *y, *vx, *vy| {
        x.* = xs[ix];
        y.* = ys[ix];
        vx.* = vxs[ix];
        vy.* = vys[ix];
    }

    const scan_cap = 4;
    var below: u32 = 0;
    for (0..n) |p| {
        const k = key[p];
        var q: u32 = @intCast(p + 1);
        while (q < n and key[q] <= k + 1 and q <= p + scan_cap) : (q += 1) ashPair(sx, sy, svx, svy, p, q);
        const lo = k + cols - 1;
        while (below < n and key[below] < lo) below += 1;
        q = below;
        while (q < n and key[q] <= k + cols + 1 and q < below + scan_cap) : (q += 1) ashPair(sx, sy, svx, svy, p, q);
    }

    for (idx, sx, sy, svx, svy) |ix, x, y, vx, vy| {
        xs[ix] = x;
        ys[ix] = y;
        vxs[ix] = vx;
        vys[ix] = vy;
    }
}

/// One pair of handle_ash_collisions.
inline fn ashPair(xs: []f32, ys: []f32, vxs: []f32, vys: []f32, i: usize, j: usize) void {
    const min_d = 2 * cfg.ash_radius;
    const dx = xs[j] - xs[i];
    const dy = ys[j] - ys[i];
    const d2 = dx * dx + dy * dy;
    if (d2 >= min_d * min_d) return;
    const d = @sqrt(d2);
    // Deviation: the Python skipped coincident ashes (d2 == 0), so ashes clamped into the
    // same wall corner stayed stacked for good.
    const nx, const ny = if (d2 > 0) .{ dx / d, dy / d } else blk: {
        const a: f32 = @floatFromInt((i +% j) % 64);
        break :blk .{ @cos(a), @sin(a) };
    };
    const half = (min_d - d) / 2;
    xs[i] -= nx * half;
    ys[i] -= ny * half;
    xs[j] += nx * half;
    ys[j] += ny * half;
    const norm_v = (vxs[j] - vxs[i]) * nx + (vys[j] - vys[i]) * ny;
    if (norm_v > 0) return;
    const p = -norm_v * 0.5;
    vxs[i] -= p * nx;
    vys[i] -= p * ny;
    vxs[j] += p * nx;
    vys[j] += p * ny;
}

pub fn updateExplosions(w: *World) void {
    var i = w.explosions.items.len;
    while (i > 0) {
        i -= 1;
        const e = &w.explosions.items[i];
        e.age += 1;
        if (e.age >= cfg.explosion_ticks) _ = w.explosions.swapRemove(i);
    }
}

/// ExplosionEffect + explosion_knockback. Does not count the event; callers do.
pub fn explode(w: *World, x: f32, y: f32, ball_size: f32) !void {
    const size = @floor(ball_size * cfg.explosion_size_mult);
    try w.explosions.append(w.gpa, .{ .x = x, .y = y, .size = size });
    knockback(w, x, y, size * cfg.blast_radius_mult);
}

/// Vectorised because a lasso around thousands of balls means thousands of blasts,
/// each touching every ball.
pub fn knockback(w: *World, ex: f32, ey: f32, blast: f32) void {
    if (blast <= 0) return;
    const s = w.balls.slice();
    const n = w.balls.len;
    const xs = s.items(.x);
    const ys = s.items(.y);
    const vxs = s.items(.vx);
    const vys = s.items(.vy);
    const temps = s.items(.temp);
    const masses = s.items(.mass);
    const flags: []const u8 = @ptrCast(s.items(.flags));
    const alive_bit: u8 = @bitCast(world.BallFlags{});
    const heat_k = cfg.base_explosion_heat * 100.0;
    const fire_r = blast * cfg.explosion_fire_frac;

    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const dx = @as(F, xs[i..][0..V].*) - @as(F, @splat(ex));
        const dy = @as(F, ys[i..][0..V].*) - @as(F, @splat(ey));
        const d = @sqrt(dx * dx + dy * dy);
        const fb: @Vector(V, u8) = flags[i..][0..V].*;
        const alive = (fb & @as(@Vector(V, u8), @splat(alive_bit))) != @as(@Vector(V, u8), @splat(0));
        const in_range = (d > @as(F, @splat(0))) & (d <= @as(F, @splat(blast)));
        const hit = alive & in_range;
        const inv_m = @as(F, @splat(1)) / @as(F, masses[i..][0..V].*);
        const fall = @as(F, @splat(1)) - d / @as(F, @splat(blast));
        const safe_d = @select(f32, hit, d, @as(F, @splat(1)));
        const k = @select(f32, hit, @as(F, @splat(cfg.base_explosion_force)) * inv_m * fall / safe_d, @as(F, @splat(0)));
        vxs[i..][0..V].* = @as(F, vxs[i..][0..V].*) + dx * k;
        vys[i..][0..V].* = @as(F, vys[i..][0..V].*) + dy * k;
        const heat = @select(f32, hit, @as(F, @splat(heat_k)) * inv_m * fall, @as(F, @splat(0)));
        const heated = @as(F, temps[i..][0..V].*) + heat;
        const ignite = hit & (d <= @as(F, @splat(fire_r)));
        temps[i..][0..V].* = @select(f32, ignite, @max(heated, @as(F, @splat(cfg.flame_temp))), heated);
    }
    while (i < n) : (i += 1) {
        if (flags[i] & alive_bit == 0) continue;
        const dx = xs[i] - ex;
        const dy = ys[i] - ey;
        const d = @sqrt(dx * dx + dy * dy);
        if (!(d > 0 and d <= blast)) continue;
        const fall = 1 - d / blast;
        const k = cfg.base_explosion_force / masses[i] * fall / d;
        vxs[i] += dx * k;
        vys[i] += dy * k;
        temps[i] += heat_k / masses[i] * fall;
        if (d <= fire_r) temps[i] = @max(temps[i], cfg.flame_temp);
    }
}

test "explosion sets nearby balls alight and only heats the outer ring" {
    var w = World.init(std.testing.allocator, 3);
    defer w.deinit();
    // Heavy balls so mass-scaled heat alone would not reach ignition.
    for ([_]f32{ 50, 200, 2000 }) |dx| {
        try w.spawnBall(500 + dx, 300);
        w.balls.items(.mass)[w.balls.len - 1] = 2000;
    }
    knockback(&w, 500, 300, 300);
    const temps = w.balls.items(.temp);
    try std.testing.expect(temps[0] >= cfg.flame_temp); // inside fire radius (180)
    try std.testing.expect(temps[1] > cfg.room_temp and temps[1] < cfg.ignition_temp); // outer ring
    try std.testing.expectEqual(cfg.room_temp, temps[2]); // outside the blast
}
