//! Mutual gravity on the GPU: a 2D fast multipole method on a uniform quadtree (compiled to
//! SPIR-V by the build; host side is gravity.zig).
//!
//! Force law (physics.zig mutualGravity): a_i = g * sum_j m_j * d / (|d|^2 + 0.001), d = x_j - x_i.
//! With complex z = x + iy that is a_i = -conj(phi(z_i)), phi(z) = sum_j m_j / (z - z_j), an
//! analytic function, so multipole/local expansions are plain complex power series (Greengard &
//! Rokhlin). Balls in the 3x3 leaf neighbourhood are summed exactly with the softened law; the
//! softening is negligible beyond one leaf cell, so the far field uses the bare 1/r kernel.
//!
//! Every entry point shares one binding layout so the host binds the same buffers for every pass:
//!   set 0 (read-only):  0 pos, 1 body
//!   set 1 (read-write): 0 vel, 1 cells, 2 keys, 3 sorted, 4 mpole, 5 local
//!   set 2 (uniform):    0 Params
//! The tree box is either Params.origin/root_size (the game) or, when Params.box_word is set,
//! computed on the GPU each pass sequence by boundsPartial/boundsFinal (the headless sim).
//! Expansions are stored per cell in dimensionless units (offsets divided by the cell side), so
//! coefficients stay O(mass) at every level and f32 is enough.
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");
const S = @import("gravity_shared.zig");
const order = S.order;
const scan_block = S.scan_block;
const min_level = S.min_level;
const Params = S.Params;
const Key = S.Key;
const Sorted = S.Sorted;
const countsOffset = S.countsOffset;
const scanOffset = S.scanOffset;
const blockOffset = S.blockOffset;
const levelOffset = S.levelOffset;

fn Rt(T: type) type {
    return extern struct { v: @SpirvType(.{ .runtime_array = T }) };
}
const pos = @extern(*addrspace(.storage_buffer) const Rt(L.BallPos), .{ .name = "pos", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const body = @extern(*addrspace(.storage_buffer) const Rt(L.BallBody), .{ .name = "body", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
const vel = @extern(*addrspace(.storage_buffer) Rt(L.BallVel), .{ .name = "vel", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });
const cells = @extern(*addrspace(.storage_buffer) Rt(u32), .{ .name = "cells", .decoration = .{ .descriptor = .{ .set = 1, .binding = 1 } } });
const keys = @extern(*addrspace(.storage_buffer) Rt(Key), .{ .name = "keys", .decoration = .{ .descriptor = .{ .set = 1, .binding = 2 } } });
const sorted = @extern(*addrspace(.storage_buffer) Rt(Sorted), .{ .name = "sorted", .decoration = .{ .descriptor = .{ .set = 1, .binding = 3 } } });
const mpole = @extern(*addrspace(.storage_buffer) Rt(L.Vec2), .{ .name = "mpole", .decoration = .{ .descriptor = .{ .set = 1, .binding = 4 } } });
const local = @extern(*addrspace(.storage_buffer) Rt(L.Vec2), .{ .name = "local", .decoration = .{ .descriptor = .{ .set = 1, .binding = 5 } } });
const params = @extern(*addrspace(.uniform) const Params, .{ .name = "params", .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });

const scan_sh = @extern(*addrspace(.shared) [L.group_size]u32, .{ .name = "scan_sh" });

const kernel_cc: std.builtin.CallingConvention = .{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } };

/// Zig's SPIR-V backend has no @atomicRmw yet (AIR atomic_rmw is a TODO), so emit the op directly.
inline fn atomicAdd(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
    return asm volatile (
        \\%ret = OpAtomicIAdd %u32 %p %scope %sem %v
        : [ret] "" (-> u32),
        : [u32] "t" (u32),
          [p] "" (p),
          [scope] "" (@as(u32, @backingInt(spirv.Scope.device))),
          [sem] "" (@as(u32, 0)),
          [v] "" (v),
    );
}

const C = struct {
    re: f32,
    im: f32,
    const zero: C = .{ .re = 0, .im = 0 };
    inline fn add(a: C, b: C) C {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }
    inline fn mul(a: C, b: C) C {
        return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
    }
    inline fn scale(a: C, s: f32) C {
        return .{ .re = a.re * s, .im = a.im * s };
    }
    /// a + b * c
    inline fn fma(a: C, b: C, c: C) C {
        return .{ .re = a.re + b.re * c.re - b.im * c.im, .im = a.im + b.re * c.im + b.im * c.re };
    }
    inline fn inv(a: C) C {
        const s = 1.0 / (a.re * a.re + a.im * a.im);
        return .{ .re = a.re * s, .im = -a.im * s };
    }
};

// Index-based, not pointer-taking: a helper parameter holding a StorageBuffer pointer can be
// spilled to a Function-storage variable, which needs VariablePointersStorageBuffer.
inline fn mAt(i: usize) C {
    return .{ .re = mpole.v[i].x, .im = mpole.v[i].y };
}
inline fn lAt(i: usize) C {
    return .{ .re = local.v[i].x, .im = local.v[i].y };
}
inline fn mPut(i: usize, a: C) void {
    mpole.v[i].x = a.re;
    mpole.v[i].y = a.im;
}
inline fn lPut(i: usize, a: C) void {
    local.v[i].x = a.re;
    local.v[i].y = a.im;
}

const binom = blk: {
    const n = 2 * order;
    var t: [n][n]f32 = @splat(@splat(0));
    for (0..n) |i| {
        t[i][0] = 1;
        for (1..i + 1) |k| t[i][k] = t[i - 1][k - 1] + (if (k < i) t[i - 1][k] else 0);
    }
    break :blk t;
};

inline fn gid() u32 {
    return spirv.global_invocation_id[0] + spirv.global_invocation_id[1] * (S.dispatch_x * L.group_size);
}

inline fn boxF(k: u32, fixed: f32) f32 {
    if (params.box_word == S.no_box) return fixed;
    return @bitCast(cells.v[params.box_word + k]);
}
inline fn originX() f32 {
    return boxF(0, params.origin_x);
}
inline fn originY() f32 {
    return boxF(1, params.origin_y);
}
inline fn rootSize() f32 {
    return boxF(2, params.root_size);
}

fn leafDim() u32 {
    return @as(u32, 1) << @intCast(params.leaf_level);
}

fn leafCellOf(x: f32, y: f32) u32 {
    const dim = leafDim();
    const inv_side = @as(f32, @floatFromInt(dim)) / rootSize();
    const fdim: f32 = @floatFromInt(dim - 1);
    // Balls outside the bounds are clamped into the edge cell; their own expansions then lose
    // accuracy, but the walls keep them inside in the game.
    const cx: u32 = @intFromFloat(@min(@max((x - originX()) * inv_side, 0.0), fdim));
    const cy: u32 = @intFromFloat(@min(@max((y - originY()) * inv_side, 0.0), fdim));
    return cy * dim + cx;
}

const dead_cell: u32 = 0xFFFF_FFFF;

fn cellStart(cell: u32) u32 {
    const nleaf = leafDim() * leafDim();
    return cells.v[scanOffset(nleaf) + cell] + cells.v[blockOffset(nleaf) + cell / scan_block];
}

fn cellCenter(level: u32, x: u32, y: u32) C {
    const side = rootSize() / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(level)));
    return .{
        .re = originX() + (@as(f32, @floatFromInt(x)) + 0.5) * side,
        .im = originY() + (@as(f32, @floatFromInt(y)) + 0.5) * side,
    };
}

export fn clearCounts() callconv(kernel_cc) void {
    const i = gid();
    if (i >= leafDim() * leafDim()) return;
    cells.v[i] = 0;
}

export fn binBalls() callconv(kernel_cc) void {
    const i = gid();
    if (i >= params.n) return;
    // Dead balls (mass 0) are left out entirely: they keep their last position, which after the
    // play area shrinks is outside the bounds, and clamping hundreds of thousands of them into the
    // edge cells made the exact near-field sums quadratic.
    if (body.v[i].mass == 0) {
        keys.v[i].cell = dead_cell;
        return;
    }
    const cell = leafCellOf(pos.v[i].x, pos.v[i].y);
    const rank = atomicAdd(&cells.v[countsOffset() + cell], 1);
    keys.v[i].cell = cell;
    keys.v[i].rank = rank;
}

/// Exclusive scan of 1024 counts per workgroup; the block total goes to the block region.
export fn scanBlocks() callconv(kernel_cc) void {
    const nleaf = leafDim() * leafDim();
    const l = spirv.local_invocation_id[0];
    const base = spirv.workgroup_id[0] * scan_block + l * 4;
    var v: [4]u32 = undefined;
    var sum: u32 = 0;
    inline for (0..4) |k| {
        v[k] = if (base + k < nleaf) cells.v[base + k] else 0;
        sum += v[k];
    }
    const total = workgroupScan(l, sum);
    var run = total.excl;
    inline for (0..4) |k| {
        if (base + k < nleaf) cells.v[scanOffset(nleaf) + base + k] = run;
        run += v[k];
    }
    if (l == L.group_size - 1) cells.v[blockOffset(nleaf) + spirv.workgroup_id[0]] = total.excl + sum;
}

/// One workgroup: exclusive scan of the block totals in place, 1024 at a time with a running carry.
export fn scanTotals() callconv(kernel_cc) void {
    const nleaf = leafDim() * leafDim();
    const nblocks = (nleaf + scan_block - 1) / scan_block;
    const off = blockOffset(nleaf);
    const l = spirv.local_invocation_id[0];
    var carry: u32 = 0;
    var base: u32 = 0;
    while (base < nblocks) : (base += scan_block) {
        var v: [4]u32 = undefined;
        var sum: u32 = 0;
        inline for (0..4) |k| {
            v[k] = if (base + l * 4 + k < nblocks) cells.v[off + base + l * 4 + k] else 0;
            sum += v[k];
        }
        const sc = workgroupScan(l, sum);
        var run = carry + sc.excl;
        inline for (0..4) |k| {
            if (base + l * 4 + k < nblocks) cells.v[off + base + l * 4 + k] = run;
            run += v[k];
        }
        carry += sc.total;
        spirv.workgroupBarrier();
    }
}

/// Hillis-Steele over the workgroup; returns the exclusive prefix of `x`.
fn workgroupScan(l: u32, x: u32) struct { excl: u32, total: u32 } {
    scan_sh[l] = x;
    spirv.workgroupBarrier();
    var off: u32 = 1;
    while (off < L.group_size) : (off <<= 1) {
        const add = if (l >= off) scan_sh[l - off] else 0;
        spirv.workgroupBarrier();
        scan_sh[l] += add;
        spirv.workgroupBarrier();
    }
    // l | (group_size - 1) is the last thread's slot for every l; a constant index into a shared
    // array miscompiles in the SPIR-V backend.
    return .{ .excl = scan_sh[l] - x, .total = scan_sh[l | (L.group_size - 1)] };
}

export fn scatterBalls() callconv(kernel_cc) void {
    const i = gid();
    if (i >= params.n) return;
    const kc = keys.v[i].cell;
    if (kc == dead_cell) return;
    const px = pos.v[i].x;
    const py = pos.v[i].y;
    // Field-wise: whole-struct loads/stores of buffer structs make Zig emit OpCopyLogical (SPIR-V 1.4) in a 1.0 module.
    const slot = cellStart(kc) + keys.v[i].rank;
    // Clamped into the tree box: a body far outside would otherwise give |w| >> 1 in leafMultipoles
    // and order-6 moments ~ m*w^5 that corrupt every far field. It acts as if at the box edge.
    const ox = originX();
    const oy = originY();
    const size = rootSize();
    sorted.v[slot].x = @min(@max(px, ox), ox + size);
    sorted.v[slot].y = @min(@max(py, oy), oy + size);
    sorted.v[slot].m = body.v[i].mass;
    sorted.v[slot].id = i;
}

/// Leaf multipoles: M_k = sum m * w^k, w = (z - center) / side.
export fn leafMultipoles() callconv(kernel_cc) void {
    const cell = gid();
    const dim = leafDim();
    if (cell >= dim * dim) return;
    const lvl = params.leaf_level;
    const center = cellCenter(lvl, cell % dim, cell / dim);
    const inv_side = @as(f32, @floatFromInt(dim)) / rootSize();
    var m: [order]C = @splat(C.zero);
    const s0 = cellStart(cell);
    const s1 = s0 + cells.v[countsOffset() + cell];
    var s = s0;
    while (s < s1) : (s += 1) {
        const b = sorted.v[s];
        const w: C = .{ .re = (b.x - center.re) * inv_side, .im = (b.y - center.im) * inv_side };
        var wk: C = .{ .re = b.m, .im = 0 };
        inline for (0..order) |k| {
            m[k] = m[k].add(wk);
            if (k + 1 < order) wk = wk.mul(w);
        }
    }
    const out = (levelOffset(lvl) + cell) * order;
    inline for (0..order) |k| mPut(out + k, m[k]);
}

/// Parent multipole from its four children: M_k = sum_c sum_{j<=k} C(k,j) 2^-j d^(k-j) M'_j,
/// d = (child center - parent center) / parent side = (+-1/4, +-1/4).
export fn upwardPass() callconv(kernel_cc) void {
    const lvl = params.level;
    const dim = @as(u32, 1) << @intCast(lvl);
    const cell = gid();
    if (cell >= dim * dim) return;
    const x = cell % dim;
    const y = cell / dim;
    var m: [order]C = @splat(C.zero);
    inline for (0..4) |c| {
        const cx = 2 * x + (c & 1);
        const cy = 2 * y + (c >> 1);
        const src = (levelOffset(lvl + 1) + cy * 2 * dim + cx) * order;
        var mc: [order]C = undefined;
        inline for (0..order) |j| mc[j] = mAt(src + j).scale(1.0 / @as(f32, @floatFromInt(@as(u32, 1) << j)));
        const d: C = .{ .re = if (c & 1 != 0) 0.25 else -0.25, .im = if (c & 2 != 0) 0.25 else -0.25 };
        var dp: [order]C = undefined;
        dp[0] = .{ .re = 1, .im = 0 };
        inline for (1..order) |k| dp[k] = dp[k - 1].mul(d);
        inline for (0..order) |k| {
            inline for (0..k + 1) |j| m[k] = m[k].add(mc[j].mul(dp[k - j]).scale(binom[k][j]));
        }
    }
    const out = (levelOffset(lvl) + cell) * order;
    inline for (0..order) |k| mPut(out + k, m[k]);
}

/// Local expansion of one cell: parent's local shifted down (L2L) plus the multipoles of its
/// interaction list (M2L): children of the parent's neighbours that are not its own neighbours.
export fn downwardPass() callconv(kernel_cc) void {
    const lvl = params.level;
    const dim = @as(u32, 1) << @intCast(lvl);
    const cell = gid();
    if (cell >= dim * dim) return;
    const off = levelOffset(lvl);
    // Empty cells have no balls below them, so nobody reads their local expansion.
    if (mpole.v[(off + cell) * order].x == 0) return;
    const x = cell % dim;
    const y = cell / dim;
    const side = rootSize() / @as(f32, @floatFromInt(dim));

    var acc: [order]C = @splat(C.zero);
    // M2L accumulates in units of 1/side; the parent's local is in absolute units.
    if (lvl > min_level) {
        const px = x >> 1;
        const py = y >> 1;
        const src = (levelOffset(lvl - 1) + py * (dim >> 1) + px) * order;
        var lp: [order]C = undefined;
        inline for (0..order) |k| lp[k] = lAt(src + k).scale(side);
        const d: C = .{ .re = if (x & 1 != 0) 0.25 else -0.25, .im = if (y & 1 != 0) 0.25 else -0.25 };
        var dp: [order]C = undefined;
        dp[0] = .{ .re = 1, .im = 0 };
        inline for (1..order) |k| dp[k] = dp[k - 1].mul(d);
        inline for (0..order) |j| {
            var s = C.zero;
            inline for (j..order) |k| s = s.add(lp[k].mul(dp[k - j]).scale(binom[k][j]));
            acc[j] = s.scale(1.0 / @as(f32, @floatFromInt(@as(u32, 1) << j)));
        }
    }

    // All unsigned: the backend mistypes signed ints (same-width OpSConvert for @intCast u32<->i32,
    // i32/u32 mismatched OpStore), so no i32 here.
    const px = x >> 1;
    const py = y >> 1;
    const x0 = if (px > 0) 2 * px - 2 else 0;
    const y0 = if (py > 0) 2 * py - 2 else 0;
    const x1 = @min(2 * px + 3, dim - 1);
    const y1 = @min(2 * py + 3, dim - 1);
    var sy = y0;
    while (sy <= y1) : (sy += 1) {
        var sx = x0;
        while (sx <= x1) : (sx += 1) {
            if (sx + 1 >= x and sx <= x + 1 and sy + 1 >= y and sy <= y + 1) continue;
            const src = (off + sy * dim + sx) * order;
            if (mpole.v[src].x == 0) continue;
            var ms: [order]C = undefined;
            inline for (0..order) |k| ms[k] = mAt(src + k);
            // u = (z - c_s)/side = v + D; (v + D)^-(k+1) = sum_j C(k+j,j) (-v)^j D^-(k+j+1).
            const dinv = C.inv(.{ .re = @as(f32, @floatFromInt(x)) - @as(f32, @floatFromInt(sx)), .im = @as(f32, @floatFromInt(y)) - @as(f32, @floatFromInt(sy)) });
            var ip: [2 * order - 1]C = undefined;
            ip[0] = dinv;
            inline for (1..2 * order - 1) |n| ip[n] = ip[n - 1].mul(dinv);
            inline for (0..order) |j| {
                var s = C.zero;
                inline for (0..order) |k| s = s.add(ms[k].mul(ip[k + j]).scale(binom[k + j][j]));
                acc[j] = if (j & 1 == 0) acc[j].add(s) else acc[j].add(s.scale(-1));
            }
        }
    }
    const out = (off + cell) * order;
    const inv_side = 1.0 / side;
    inline for (0..order) |k| lPut(out + k, acc[k].scale(inv_side));
}

/// Per ball (in cell-sorted order, so a warp shares its neighbour cells): far field from the
/// leaf's local expansion plus exact softened pairwise sum over the 3x3 leaf neighbourhood.
export fn evaluate() callconv(kernel_cc) void {
    const s = gid();
    if (s >= params.n) return;
    // Only live balls were sorted; slots past the last cell hold stale data.
    const last = leafDim() * leafDim() - 1;
    if (s >= cellStart(last) + cells.v[countsOffset() + last]) return;
    const b = sorted.v[s];
    const lvl = params.leaf_level;
    const dim = leafDim();
    const cell = leafCellOf(b.x, b.y);
    const cx = cell % dim;
    const cy = cell / dim;
    const center = cellCenter(lvl, cx, cy);
    const inv_side = @as(f32, @floatFromInt(dim)) / rootSize();
    const v: C = .{ .re = (b.x - center.re) * inv_side, .im = (b.y - center.im) * inv_side };
    const lsrc = (levelOffset(lvl) + cell) * order;
    var phi = lAt(lsrc + order - 1);
    inline for (1..order) |r| phi = lAt(lsrc + order - 1 - r).add(phi.mul(v));
    var ax = -phi.re;
    var ay = phi.im;

    const y0 = if (cy > 0) cy - 1 else 0;
    const y1 = @min(cy + 1, dim - 1);
    const x0 = if (cx > 0) cx - 1 else 0;
    const x1 = @min(cx + 1, dim - 1);
    var ny = y0;
    while (ny <= y1) : (ny += 1) {
        // A row of three cells is contiguous in sorted order.
        const row = ny * dim;
        const t0 = cellStart(row + x0);
        const t1 = cellStart(row + x1) + cells.v[countsOffset() + row + x1];
        var t = t0;
        while (t < t1) : (t += 1) {
            const o = sorted.v[t];
            const dx = o.x - b.x;
            const dy = o.y - b.y;
            const f = o.m / (dx * dx + dy * dy + params.softening);
            ax += f * dx;
            ay += f * dy;
        }
    }

    vel.v[b.id].x += params.g_scale * ax;
    vel.v[b.id].y += params.g_scale * ay;
}

const sh_minx = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh_minx" });
const sh_miny = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh_miny" });
const sh_maxx = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh_maxx" });
const sh_maxy = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh_maxy" });

/// Reduces the four shared arrays over the workgroup; the result is in index 0. (Constant-index
/// access to these arrays miscompiles in the SPIR-V backend, so callers index with l == 0.)
fn reduceBounds(l: u32) void {
    var off: u32 = L.group_size / 2;
    while (off > 0) : (off >>= 1) {
        spirv.workgroupBarrier();
        if (l < off) {
            sh_minx[l] = @min(sh_minx[l], sh_minx[l + off]);
            sh_miny[l] = @min(sh_miny[l], sh_miny[l + off]);
            sh_maxx[l] = @max(sh_maxx[l], sh_maxx[l + off]);
            sh_maxy[l] = @max(sh_maxy[l], sh_maxy[l + off]);
        }
    }
    spirv.workgroupBarrier();
}

fn boundsGroups() u32 {
    const threads = (params.n + S.bounds_per_thread - 1) / S.bounds_per_thread;
    return (threads + L.group_size - 1) / L.group_size;
}

/// Tree box, step 1: per-workgroup bounding box of the live bodies into `mpole` (scratch until
/// leafMultipoles overwrites it): group g writes min to slot 2g and max to slot 2g+1. A NaN
/// position fails both comparisons, so it never reaches the box.
export fn boundsPartial() callconv(kernel_cc) void {
    const l = spirv.local_invocation_id[0];
    const stride = boundsGroups() * L.group_size;
    var mnx: f32 = 3.0e38;
    var mny: f32 = 3.0e38;
    var mxx: f32 = -3.0e38;
    var mxy: f32 = -3.0e38;
    var i = gid();
    var k: u32 = 0;
    while (k < S.bounds_per_thread) : (k += 1) {
        if (i < params.n and body.v[i].mass != 0) {
            const x = pos.v[i].x;
            const y = pos.v[i].y;
            if (x < mnx) mnx = x;
            if (x > mxx) mxx = x;
            if (y < mny) mny = y;
            if (y > mxy) mxy = y;
        }
        i += stride;
    }
    sh_minx[l] = mnx;
    sh_miny[l] = mny;
    sh_maxx[l] = mxx;
    sh_maxy[l] = mxy;
    reduceBounds(l);
    if (l == 0) {
        const g = spirv.workgroup_id[0];
        mpole.v[2 * g].x = sh_minx[l];
        mpole.v[2 * g].y = sh_miny[l];
        mpole.v[2 * g + 1].x = sh_maxx[l];
        mpole.v[2 * g + 1].y = sh_maxy[l];
    }
}

/// Tree box, step 2 (one workgroup): the square box over all partials goes to cells[box_word..],
/// 0.1% larger than the bodies' extent so none sits exactly on the far edge.
export fn boundsFinal() callconv(kernel_cc) void {
    const l = spirv.local_invocation_id[0];
    const groups = boundsGroups();
    var mnx: f32 = 3.0e38;
    var mny: f32 = 3.0e38;
    var mxx: f32 = -3.0e38;
    var mxy: f32 = -3.0e38;
    var g = l;
    while (g < groups) : (g += L.group_size) {
        mnx = @min(mnx, mpole.v[2 * g].x);
        mny = @min(mny, mpole.v[2 * g].y);
        mxx = @max(mxx, mpole.v[2 * g + 1].x);
        mxy = @max(mxy, mpole.v[2 * g + 1].y);
    }
    sh_minx[l] = mnx;
    sh_miny[l] = mny;
    sh_maxx[l] = mxx;
    sh_maxy[l] = mxy;
    reduceBounds(l);
    if (l == 0) {
        const live = sh_minx[l] <= sh_maxx[l] and sh_miny[l] <= sh_maxy[l];
        const w = if (live) sh_maxx[l] - sh_minx[l] else 0.0;
        const h = if (live) sh_maxy[l] - sh_miny[l] else 0.0;
        const size = @max(@max(w, h) * 1.001, 1.0);
        const cx = if (live) 0.5 * (sh_minx[l] + sh_maxx[l]) else 0.0;
        const cy = if (live) 0.5 * (sh_miny[l] + sh_maxy[l]) else 0.0;
        cells.v[params.box_word] = @bitCast(cx - 0.5 * size);
        cells.v[params.box_word + 1] = @bitCast(cy - 0.5 * size);
        cells.v[params.box_word + 2] = @bitCast(size);
    }
}
