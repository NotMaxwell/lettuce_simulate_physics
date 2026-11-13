//! f64 CPU reference for the 3D Barnes-Hut pipeline (docs/sim3d.md): direct sum, Morton keys and
//! stable sort, leaf buckets, Karras radix tree, ropes, depth-ordered monopole+quadrupole moments and
//! the stackless walk. GPU packages 3-5 test against the arrays exposed by `Tree`.
//!
//! Everything is a pure function of (positions, masses, box, theta): same keys give the same tree.
//! Node numbering (what a GPU Karras build yields directly): internal node i is index i in
//! 0..M-2 (root is 0), leaf bucket j is index M-1+j, M = number of leaves, 2M-1 nodes in all.
const std = @import("std");
/// Test-only sampler, exported for ref3/measure.zig.
pub const plummer = @import("ref3/plummer.zig");

pub const Pos3 = extern struct { x: u32, y: u32, z: u32 };

/// Must stay identical (field order and size) to the canonical struct in gpu_layout.zig.
/// `info` = depth << 26 | count; count saturates at 2^26-1 for internal nodes (leaves are exact).
/// `open_r2` is (l/theta + |com - cell centre|)^2 in physical units: the node is accepted when
/// |target - com|^2 > open_r2, so theta is fixed at build time.
pub const Node = extern struct {
    com: Pos3,
    mass: f32,
    open_r2: f32,
    child: u32,
    skip: u32,
    info: u32,
};

/// Traceless quadrupole sum m (3 x x^T - |x|^2 I) about the node's com, physical units. Must stay
/// identical to the canonical struct in gpu_layout.zig.
pub const NodeQuad = extern struct { xx: f32, xy: f32, xz: f32, yy: f32, yz: f32, zz: f32 };

pub const SortRec = extern struct { key_hi: u32, key_lo: u32, idx: u32 };

comptime {
    std.debug.assert(@sizeOf(Node) == 32 and @alignOf(Node) == 4);
    std.debug.assert(@sizeOf(NodeQuad) == 24);
    std.debug.assert(@sizeOf(SortRec) == 12);
    std.debug.assert(@sizeOf(Pos3) == 12);
}

pub const leaf_flag: u32 = 0x8000_0000;
pub const end: u32 = 0xFFFF_FFFF;
pub const depth_shift = 5 + 21;
pub const count_mask: u32 = (1 << depth_shift) - 1;

pub fn nodeDepth(n: Node) u32 {
    return n.info >> depth_shift;
}
pub fn nodeCount(n: Node) u32 {
    return n.info & count_mask;
}
pub fn isLeaf(n: Node) bool {
    return n.child & leaf_flag != 0;
}

pub const Accel = struct { a: [3]f64 = .{ 0, 0, 0 }, pot: f64 = 0 };
pub const Stats = struct { visits: u64 = 0, cells: u64 = 0, pairs: u64 = 0 };

/// Physical coordinate of a u32 box fraction.
inline fn phys(v: u32, box: f64) f64 {
    return @as(f64, @floatFromInt(v)) * (box / 4294967296.0);
}

inline fn pair(acc: *Accel, d: [3]f64, m: f64, eps2: f64) void {
    const s = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + eps2;
    const inv = 1.0 / @sqrt(s);
    const f = m * inv * inv * inv;
    acc.a[0] += f * d[0];
    acc.a[1] += f * d[1];
    acc.a[2] += f * d[2];
    acc.pot -= m * inv;
}

/// Acceleration and potential at `target` from every body except index `self`, Plummer softened.
pub fn directAt(pos: []const Pos3, mass: []const f32, box: f64, target: Pos3, self: ?usize, g: f64, eps: f64) Accel {
    var acc: Accel = .{};
    const t = [3]f64{ phys(target.x, box), phys(target.y, box), phys(target.z, box) };
    for (pos, mass, 0..) |p, m, j| {
        if (j == self) continue;
        pair(&acc, .{ phys(p.x, box) - t[0], phys(p.y, box) - t[1], phys(p.z, box) - t[2] }, m, eps * eps);
    }
    return .{ .a = .{ acc.a[0] * g, acc.a[1] * g, acc.a[2] * g }, .pot = acc.pot * g };
}

/// O(N^2) oracle for all bodies. `out.len == pos.len`.
pub fn directSum(pos: []const Pos3, mass: []const f32, box: f64, g: f64, eps: f64, out: []Accel) void {
    for (pos, out, 0..) |p, *o, i| o.* = directAt(pos, mass, box, p, i, g, eps);
}

/// 63-bit Morton key: the top 21 bits of x, y, z interleaved with x most significant in each triple.
pub fn mortonKey(p: Pos3) u64 {
    return (spread(p.x >> 11) << 2) | (spread(p.y >> 11) << 1) | spread(p.z >> 11);
}

fn spread(v: u32) u64 {
    var x: u64 = v & 0x1FFFFF;
    x = (x | x << 32) & 0x001F00000000FFFF;
    x = (x | x << 16) & 0x001F0000FF0000FF;
    x = (x | x << 8) & 0x100F00F00F00F00F;
    x = (x | x << 4) & 0x10C30C30C30C30C3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

const KeyIdx = struct { key: u64, idx: u32 };

fn keyIdxLess(_: void, a: KeyIdx, b: KeyIdx) bool {
    return a.key < b.key or (a.key == b.key and a.idx < b.idx);
}

pub const Sorted = struct {
    keys: []u64,
    /// perm[sorted position] = original index.
    perm: []u32,

    pub fn deinit(s: Sorted, gpa: std.mem.Allocator) void {
        gpa.free(s.keys);
        gpa.free(s.perm);
    }
    /// The same data as the GPU's ping-pong record array.
    pub fn rec(s: Sorted, i: usize) SortRec {
        return .{ .key_hi = @intCast(s.keys[i] >> 32), .key_lo = @truncate(s.keys[i]), .idx = s.perm[i] };
    }
};

/// Stable sort by key (ties by original index), so equal keys always keep input order.
pub fn sortKeys(gpa: std.mem.Allocator, pos: []const Pos3) !Sorted {
    const recs = try gpa.alloc(KeyIdx, pos.len);
    defer gpa.free(recs);
    for (recs, pos, 0..) |*r, p, i| r.* = .{ .key = mortonKey(p), .idx = @intCast(i) };
    std.mem.sort(KeyIdx, recs, {}, keyIdxLess);
    const keys = try gpa.alloc(u64, pos.len);
    errdefer gpa.free(keys);
    const perm = try gpa.alloc(u32, pos.len);
    for (recs, keys, perm) |r, *k, *q| {
        k.* = r.key;
        q.* = r.idx;
    }
    return .{ .keys = keys, .perm = perm };
}

/// Number of leading key bits two adjacent sorted keys share (63 = equal).
fn commonBits(a: u64, b: u64) u32 {
    return if (a == b) 63 else @as(u32, @clz(a ^ b)) - 1;
}

/// The leaf bucket containing sorted body `i`: the largest prefix cell holding at most `cap` bodies,
/// found by looking at no more than `cap` neighbours on each side (the GPU build does the same per body).
/// Bodies sharing one full key cannot be split, so such a run is one leaf whatever its size.
pub fn leafRange(keys: []const u64, i: usize, cap: u32) struct { lo: usize, hi: usize } {
    const n = keys.len;
    var lo_l: u32 = 0;
    var hi_l: u32 = 63;
    // Smallest prefix length whose cell is feasible; feasibility is monotone in the length.
    while (lo_l < hi_l) {
        const mid = (lo_l + hi_l) / 2;
        if (windowCount(keys, i, mid, cap) <= cap) hi_l = mid else lo_l = mid + 1;
    }
    var lo = i;
    var hi = i + 1;
    if (windowCount(keys, i, lo_l, cap) <= cap) {
        while (lo > 0 and commonBits(keys[lo - 1], keys[lo]) >= lo_l) lo -= 1;
        while (hi < n and commonBits(keys[hi - 1], keys[hi]) >= lo_l) hi += 1;
    } else {
        while (lo > 0 and keys[lo - 1] == keys[i]) lo -= 1;
        while (hi < n and keys[hi] == keys[i]) hi += 1;
    }
    return .{ .lo = lo, .hi = hi };
}

fn windowCount(keys: []const u64, i: usize, prefix: u32, cap: u32) u32 {
    var cnt: u32 = 1;
    var j = i;
    while (cnt <= cap and j > 0 and commonBits(keys[j - 1], keys[j]) >= prefix) : (j -= 1) cnt += 1;
    j = i;
    while (cnt <= cap and j + 1 < keys.len and commonBits(keys[j], keys[j + 1]) >= prefix) : (j += 1) cnt += 1;
    return cnt;
}

pub const Params = struct {
    box: f64,
    theta: f64,
    leaf_cap: u32 = 32,
    /// Criterion geometry: the bounding box of the node's bodies (default, what the design table measured)
    /// or the smallest enclosing prefix cell. Only `open_r2` differs; the layout is the same.
    tight: bool = true,
};

pub const Tree = struct {
    box: f64,
    theta: f64,
    nodes: []Node,
    quads: []NodeQuad,
    /// perm[sorted position] = original index.
    perm: []u32,
    keys: []u64,
    /// Bodies in sorted order (what the leaves index).
    pos: []Pos3,
    mass: []f32,
    leaves: u32,

    pub fn deinit(t: Tree, gpa: std.mem.Allocator) void {
        gpa.free(t.nodes);
        gpa.free(t.quads);
        gpa.free(t.perm);
        gpa.free(t.keys);
        gpa.free(t.pos);
        gpa.free(t.mass);
    }

    pub fn leafNode(t: Tree, leaf: u32) u32 {
        return t.leaves - 1 + leaf;
    }

    /// Acceleration and potential at `target`; `self` is a sorted-order body index to exclude.
    pub fn walk(t: Tree, target: Pos3, self: ?u32, g: f64, eps: f64, stats: ?*Stats) Accel {
        var acc: Accel = .{};
        var st: Stats = .{};
        const tp = [3]f64{ phys(target.x, t.box), phys(target.y, t.box), phys(target.z, t.box) };
        const eps2 = eps * eps;
        var idx: u32 = 0;
        while (idx != end) {
            const n = t.nodes[idx];
            st.visits += 1;
            const r = [3]f64{ tp[0] - phys(n.com.x, t.box), tp[1] - phys(n.com.y, t.box), tp[2] - phys(n.com.z, t.box) };
            const r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            if (r2 > n.open_r2 and r2 > near_k * near_k * eps2) {
                st.cells += 1;
                cellAccel(&acc, r, r2, n.mass, t.quads[idx], eps2);
                idx = n.skip;
            } else if (isLeaf(n)) {
                const first = n.child & ~leaf_flag;
                const cnt = nodeCount(n);
                for (first..first + cnt) |j| {
                    if (self != null and j == self.?) continue;
                    st.pairs += 1;
                    const p = t.pos[j];
                    pair(&acc, .{ phys(p.x, t.box) - tp[0], phys(p.y, t.box) - tp[1], phys(p.z, t.box) - tp[2] }, t.mass[j], eps2);
                }
                idx = n.skip;
            } else idx = n.child;
        }
        if (stats) |s| {
            s.visits += st.visits;
            s.cells += st.cells;
            s.pairs += st.pairs;
        }
        return .{ .a = .{ acc.a[0] * g, acc.a[1] * g, acc.a[2] * g }, .pot = acc.pot * g };
    }
};

/// Nodes closer than near_k softening lengths are never accepted. Measured (2 x 400-body clumps of
/// size 0.2 eps, 0.5-40 eps apart): with no guard, cells inside a clump smaller than eps cost 4e-3
/// at every separation; the worst case over separations is 1.0e-3 with a guard of 1 eps, 4.4e-4
/// with 3 and 6e-5 with 4. A 1M Plummer cloud needs 948 pairs per body at eps 0.01 either way and
/// 1147 (guard 3) or 1902 (guard 4) at eps 0.05.
pub const near_k: f64 = 3;

/// Far field of one node; `r` is target minus com. Monopole and quadrupole are both the Plummer-
/// softened expansion, so a one-body node equals the direct pair term. Only the traceless part of
/// the softened quadrupole is kept (the trace, sum m |x|^2, is not stored); it is the eps^2/s share,
/// at most 10% of the quadrupole beyond near_k.
fn cellAccel(acc: *Accel, r: [3]f64, r2: f64, m: f32, q: NodeQuad, eps2: f64) void {
    const mm: f64 = m;
    const inv = 1.0 / @sqrt(r2 + eps2);
    const inv2 = inv * inv;
    const inv3 = inv2 * inv;
    const inv5 = inv3 * inv2;
    const qr = [3]f64{
        q.xx * r[0] + q.xy * r[1] + q.xz * r[2],
        q.xy * r[0] + q.yy * r[1] + q.yz * r[2],
        q.xz * r[0] + q.yz * r[1] + q.zz * r[2],
    };
    const rqr = r[0] * qr[0] + r[1] * qr[1] + r[2] * qr[2];
    for (0..3) |k| acc.a[k] += -mm * r[k] * inv3 + qr[k] * inv5 - 2.5 * rqr * r[k] * inv5 * inv2;
    acc.pot += -mm * inv - 0.5 * rqr * inv5;
}

const Moment = struct {
    m: f64 = 0,
    c: [3]f64 = .{ 0, 0, 0 },
    q: [6]f64 = @splat(0),
    lo: [3]f64 = @splat(std.math.inf(f64)),
    hi: [3]f64 = @splat(-std.math.inf(f64)),
};

fn addQuad(q: *[6]f64, m: f64, x: [3]f64) void {
    const r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    q[0] += m * (3 * x[0] * x[0] - r2);
    q[1] += m * 3 * x[0] * x[1];
    q[2] += m * 3 * x[0] * x[2];
    q[3] += m * (3 * x[1] * x[1] - r2);
    q[4] += m * 3 * x[1] * x[2];
    q[5] += m * (3 * x[2] * x[2] - r2);
}

fn leafMoment(pos: []const Pos3, mass: []const f32, box: f64) Moment {
    var mo: Moment = .{};
    for (pos, mass) |p, m| {
        const w: f64 = m;
        mo.m += w;
        mo.c[0] += w * phys(p.x, box);
        mo.c[1] += w * phys(p.y, box);
        mo.c[2] += w * phys(p.z, box);
        inline for (.{ p.x, p.y, p.z }, 0..) |v, k| {
            mo.lo[k] = @min(mo.lo[k], phys(v, box));
            mo.hi[k] = @max(mo.hi[k], phys(v, box));
        }
    }
    for (&mo.c) |*c| c.* /= mo.m;
    for (pos, mass) |p, m| addQuad(&mo.q, m, .{ phys(p.x, box) - mo.c[0], phys(p.y, box) - mo.c[1], phys(p.z, box) - mo.c[2] });
    return mo;
}

fn combine(a: Moment, b: Moment) Moment {
    var mo: Moment = .{ .m = a.m + b.m };
    for (0..3) |k| {
        mo.lo[k] = @min(a.lo[k], b.lo[k]);
        mo.hi[k] = @max(a.hi[k], b.hi[k]);
    }
    for (0..3) |k| mo.c[k] = (a.m * a.c[k] + b.m * b.c[k]) / mo.m;
    for (0..6) |k| mo.q[k] = a.q[k] + b.q[k];
    // Parallel-axis shift of each child's quadrupole to the combined com.
    for ([_]Moment{ a, b }) |ch| {
        addQuad(&mo.q, ch.m, .{ ch.c[0] - mo.c[0], ch.c[1] - mo.c[1], ch.c[2] - mo.c[2] });
    }
    return mo;
}

inline fn karrasDelta(keys: []const u64, i: i64, j: i64) i32 {
    if (j < 0 or j >= keys.len) return -1;
    return @clz(keys[@intCast(i)] ^ keys[@intCast(j)]);
}

/// Builds the tree. `pos` and `mass` are in input order; `Tree.perm` maps the sorted order back.
pub fn build(gpa: std.mem.Allocator, pos: []const Pos3, mass: []const f32, p: Params) !Tree {
    // Above 2/sqrt(3) a cell containing the target can pass the opening test.
    std.debug.assert(p.theta < 1.1547);
    std.debug.assert(pos.len == mass.len and pos.len > 0 and pos.len < leaf_flag);
    const n = pos.len;
    const sorted = try sortKeys(gpa, pos);
    errdefer sorted.deinit(gpa);
    const spos = try gpa.alloc(Pos3, n);
    errdefer gpa.free(spos);
    const smass = try gpa.alloc(f32, n);
    errdefer gpa.free(smass);
    for (sorted.perm, spos, smass) |o, *sp, *sm| {
        sp.* = pos[o];
        sm.* = mass[o];
    }
    const keys = sorted.keys;

    // Leaf buckets.
    var starts: std.ArrayList(u32) = .empty;
    defer starts.deinit(gpa);
    var i: usize = 0;
    while (i < n) {
        const r = leafRange(keys, i, p.leaf_cap);
        std.debug.assert(r.lo == i);
        try starts.append(gpa, @intCast(i));
        i = r.hi;
    }
    try starts.append(gpa, @intCast(n));
    const m: u32 = @intCast(starts.items.len - 1);
    const lstart = starts.items;
    const total: usize = 2 * m - 1;

    const nodes = try gpa.alloc(Node, total);
    errdefer gpa.free(nodes);
    const quads = try gpa.alloc(NodeQuad, total);
    errdefer gpa.free(quads);
    const lkeys = try gpa.alloc(u64, m);
    defer gpa.free(lkeys);
    for (lkeys, 0..) |*k, l| k.* = keys[lstart[l]];

    // Karras topology; lo/hi are the leaf range each internal node spans.
    const lo = try gpa.alloc(u32, total);
    defer gpa.free(lo);
    const hi = try gpa.alloc(u32, total);
    defer gpa.free(hi);
    const kids = try gpa.alloc([2]u32, total);
    defer gpa.free(kids);
    for (0..m - 1) |ii| {
        const i64_: i64 = @intCast(ii);
        const d: i64 = if (karrasDelta(lkeys, i64_, i64_ + 1) - karrasDelta(lkeys, i64_, i64_ - 1) > 0) 1 else -1;
        const dmin = karrasDelta(lkeys, i64_, i64_ - d);
        var lmax: i64 = 2;
        while (karrasDelta(lkeys, i64_, i64_ + lmax * d) > dmin) lmax *= 2;
        var l: i64 = 0;
        var t = lmax >> 1;
        while (t >= 1) : (t >>= 1) {
            if (karrasDelta(lkeys, i64_, i64_ + (l + t) * d) > dmin) l += t;
        }
        const j = i64_ + l * d;
        const dnode = karrasDelta(lkeys, i64_, j);
        var s: i64 = 0;
        t = l;
        while (true) {
            t = (t + 1) >> 1;
            if (karrasDelta(lkeys, i64_, i64_ + (s + t) * d) > dnode) s += t;
            if (t == 1) break;
        }
        const gamma = i64_ + s * d + @min(d, 0);
        const a = @min(i64_, j);
        const b = @max(i64_, j);
        lo[ii] = @intCast(a);
        hi[ii] = @intCast(b);
        const left: u32 = if (a == gamma) m - 1 + @as(u32, @intCast(gamma)) else @intCast(gamma);
        const right: u32 = if (b == gamma + 1) m - 1 + @as(u32, @intCast(gamma + 1)) else @intCast(gamma + 1);
        kids[ii] = .{ left, right };
    }
    for (0..m) |l| {
        lo[m - 1 + l] = @intCast(l);
        hi[m - 1 + l] = @intCast(l);
    }

    // Depth and ropes, top-down (the right child inherits its parent's rope).
    const depth = try gpa.alloc(u8, total);
    defer gpa.free(depth);
    const skip = try gpa.alloc(u32, total);
    defer gpa.free(skip);
    depth[0] = 0;
    skip[0] = end;
    {
        const stack = try gpa.alloc(u32, total);
        defer gpa.free(stack);
        var sp: usize = 0;
        if (m > 1) {
            stack[0] = 0;
            sp = 1;
        }
        while (sp > 0) {
            sp -= 1;
            const nd = stack[sp];
            const k = kids[nd];
            for (k) |c| depth[c] = depth[nd] + 1;
            skip[k[0]] = k[1];
            skip[k[1]] = skip[nd];
            for (k) |c| if (c < m - 1) {
                stack[sp] = c;
                sp += 1;
            };
        }
    }

    // Moments, deepest level first so both children are ready.
    var max_depth: usize = 0;
    for (depth) |dd| max_depth = @max(max_depth, dd);
    const order = try gpa.alloc(u32, total);
    defer gpa.free(order);
    {
        const counts = try gpa.alloc(u32, max_depth + 2);
        defer gpa.free(counts);
        @memset(counts, 0);
        for (depth) |dd| counts[dd + 1] += 1;
        for (1..counts.len) |k| counts[k] += counts[k - 1];
        for (depth, 0..) |dd, ni| {
            order[counts[dd]] = @intCast(ni);
            counts[dd] += 1;
        }
    }
    const mo = try gpa.alloc(Moment, total);
    defer gpa.free(mo);
    var oi = total;
    while (oi > 0) {
        oi -= 1;
        const nd = order[oi];
        const leaf = nd >= m - 1;
        const first: u32 = lstart[lo[nd]];
        const past: u32 = lstart[hi[nd] + 1];
        mo[nd] = if (leaf) leafMoment(spos[first..past], smass[first..past], p.box) else combine(mo[kids[nd][0]], mo[kids[nd][1]]);

        var centre: [3]f64 = undefined;
        var side_max: f64 = 0;
        const mt = mo[nd];
        if (p.tight) {
            for (0..3) |k| {
                centre[k] = 0.5 * (mt.lo[k] + mt.hi[k]);
                side_max = @max(side_max, mt.hi[k] - mt.lo[k]);
            }
        } else {
            // Smallest prefix cell holding the node's bodies: its centre and longest side.
            const x = keys[first] ^ keys[past - 1];
            const q: u32 = if (x == 0) 63 else @as(u32, @clz(x)) - 1;
            var xi: [3]u32 = .{ 0, 0, 0 };
            for (0..63) |bi| {
                const bit: u32 = @intCast((keys[first] >> @intCast(62 - bi)) & 1);
                xi[bi % 3] = (xi[bi % 3] << 1) | bit;
            }
            for (0..3) |k| {
                const nb: u5 = @intCast((q + 2 - @as(u32, @intCast(k))) / 3);
                const side = p.box / @as(f64, @floatFromInt(@as(u64, 1) << nb));
                centre[k] = (@as(f64, @floatFromInt(xi[k] >> (21 - nb))) + 0.5) * side;
                side_max = @max(side_max, side);
            }
        }
        var dl2: f64 = 0;
        for (0..3) |k| dl2 += (mt.c[k] - centre[k]) * (mt.c[k] - centre[k]);
        const reach = side_max / p.theta + @sqrt(dl2);

        const cnt = past - first;
        if (leaf and cnt > count_mask) return error.LeafTooLarge;
        const sc = 4294967296.0 / p.box;
        nodes[nd] = .{
            .com = .{ .x = fracU32(mt.c[0] * sc), .y = fracU32(mt.c[1] * sc), .z = fracU32(mt.c[2] * sc) },
            .mass = @floatCast(mt.m),
            .open_r2 = @floatCast(reach * reach),
            .child = if (leaf) leaf_flag | first else kids[nd][0],
            .skip = skip[nd],
            .info = @as(u32, depth[nd]) << depth_shift | @min(cnt, count_mask),
        };
        quads[nd] = .{
            .xx = @floatCast(mt.q[0]),
            .xy = @floatCast(mt.q[1]),
            .xz = @floatCast(mt.q[2]),
            .yy = @floatCast(mt.q[3]),
            .yz = @floatCast(mt.q[4]),
            .zz = @floatCast(mt.q[5]),
        };
    }
    return .{
        .box = p.box,
        .theta = p.theta,
        .nodes = nodes,
        .quads = quads,
        .perm = sorted.perm,
        .keys = keys,
        .pos = spos,
        .mass = smass,
        .leaves = m,
    };
}

fn fracU32(v: f64) u32 {
    return @intFromFloat(std.math.clamp(@round(v), 0, 4294967295.0));
}

// ---- tests ----

const testing = std.testing;
const box_half = 10.0;
const test_box = 2 * box_half;
const test_eps = 1e-4;

const Cloud = struct {
    pos: []Pos3,
    mass: []f32,
    fn deinit(c: Cloud, gpa: std.mem.Allocator) void {
        gpa.free(c.pos);
        gpa.free(c.mass);
    }
};

fn plummerCloud(gpa: std.mem.Allocator, n: usize, seed: u64) !Cloud {
    const raw = try plummer.positions(gpa, n, seed, box_half);
    defer gpa.free(raw);
    const pos = try gpa.alloc(Pos3, n);
    const mass = try gpa.alloc(f32, n);
    for (raw, pos, mass) |r, *q, *m| {
        q.* = .{ .x = plummer.quantise(r[0], box_half), .y = plummer.quantise(r[1], box_half), .z = plummer.quantise(r[2], box_half) };
        m.* = 1.0 / @as(f32, @floatFromInt(n));
    }
    return .{ .pos = pos, .mass = mass };
}

pub const ErrStats = struct { rms: f64, p99: f64, visits: f64, cells: f64, pairs: f64 };

/// Force error of the tree walk against direct sum on `targets` evenly spaced bodies.
pub fn forceError(gpa: std.mem.Allocator, t: Tree, cloud_pos: []const Pos3, cloud_mass: []const f32, targets: usize) !ErrStats {
    return forceErrorEps(gpa, t, cloud_pos, cloud_mass, targets, test_eps);
}

/// Like forceError at softening `eps`.
pub fn forceErrorEps(gpa: std.mem.Allocator, t: Tree, cloud_pos: []const Pos3, cloud_mass: []const f32, targets: usize, eps: f64) !ErrStats {
    const n = cloud_pos.len;
    const inv = try gpa.alloc(u32, n);
    defer gpa.free(inv);
    for (t.perm, 0..) |o, s| inv[o] = @intCast(s);
    const errs = try gpa.alloc(f64, targets);
    defer gpa.free(errs);
    var st: Stats = .{};
    var rng = std.Random.DefaultPrng.init(99);
    var sum2: f64 = 0;
    for (errs) |*e| {
        const i = rng.random().uintLessThan(usize, n);
        const ex = directAt(cloud_pos, cloud_mass, t.box, cloud_pos[i], i, 1, eps);
        const ap = t.walk(cloud_pos[i], inv[i], 1, eps, &st);
        var d2: f64 = 0;
        var e2: f64 = 0;
        for (0..3) |c| {
            d2 += (ap.a[c] - ex.a[c]) * (ap.a[c] - ex.a[c]);
            e2 += ex.a[c] * ex.a[c];
        }
        e.* = @sqrt(d2 / e2);
        sum2 += d2 / e2;
    }
    std.mem.sort(f64, errs, {}, std.sort.asc(f64));
    const f: f64 = @floatFromInt(targets);
    return .{
        .rms = @sqrt(sum2 / f),
        .p99 = errs[targets * 99 / 100],
        .visits = @as(f64, @floatFromInt(st.visits)) / f,
        .cells = @as(f64, @floatFromInt(st.cells)) / f,
        .pairs = @as(f64, @floatFromInt(st.pairs)) / f,
    };
}

test "ref3: morton keys interleave x,y,z with x most significant" {
    try testing.expectEqual(@as(u64, 0), mortonKey(.{ .x = 0, .y = 0, .z = 0 }));
    try testing.expectEqual(@as(u64, 1) << 62, mortonKey(.{ .x = 0x8000_0000, .y = 0, .z = 0 }));
    try testing.expectEqual(@as(u64, 1) << 61, mortonKey(.{ .x = 0, .y = 0x8000_0000, .z = 0 }));
    try testing.expectEqual(@as(u64, 1) << 60, mortonKey(.{ .x = 0, .y = 0, .z = 0x8000_0000 }));
    try testing.expectEqual((@as(u64, 1) << 63) - 1, mortonKey(.{ .x = ~@as(u32, 0), .y = ~@as(u32, 0), .z = ~@as(u32, 0) }));
}

test "ref3: sort is stable and a bijection" {
    const gpa = testing.allocator;
    var ps = [_]Pos3{ .{ .x = 5 << 11, .y = 0, .z = 0 }, .{ .x = 1 << 11, .y = 0, .z = 0 }, .{ .x = 5 << 11, .y = 0, .z = 0 }, .{ .x = 1 << 11, .y = 0, .z = 0 } };
    const s = try sortKeys(gpa, &ps);
    defer s.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 0, 2 }, s.perm);
    try testing.expectEqual(s.perm[2], s.rec(2).idx);
}

test "ref3: 10k Plummer force error, tree shape and invariants" {
    const gpa = testing.allocator;
    const n = 10_000;
    const cloud = try plummerCloud(gpa, n, 42);
    defer cloud.deinit(gpa);

    var total: f64 = 0;
    for (cloud.mass) |m| total += m;

    for ([_]struct { theta: f64, rms_max: f64 }{ .{ .theta = 0.6, .rms_max = 1e-3 }, .{ .theta = 0.4, .rms_max = 2.5e-4 } }) |c| {
        const t = try build(gpa, cloud.pos, cloud.mass, .{ .box = test_box, .theta = c.theta });
        defer t.deinit(gpa);
        const e = try forceError(gpa, t, cloud.pos, cloud.mass, 1000);
        std.debug.print("ref3 10k theta={d:.1}: rms {e:.2} p99 {e:.2} nodes/N {d:.3} visits {d:.0} cells {d:.0} pairs {d:.0}\n", .{
            c.theta, e.rms, e.p99, @as(f64, @floatFromInt(t.nodes.len)) / n, e.visits, e.cells, e.pairs,
        });
        try testing.expect(e.rms <= c.rms_max);
        if (c.theta == 0.6) try testing.expect(e.p99 <= 5e-3);
        // Softened nodes keep the error flat from eps far below the spacing to well above it.
        if (c.theta == 0.6) for ([_]f64{ 0.01, 0.05 }) |eps| {
            const es = try forceErrorEps(gpa, t, cloud.pos, cloud.mass, 1000, eps);
            std.debug.print("ref3 10k theta=0.6 eps={d}: rms {e:.2} p99 {e:.2} pairs {d:.0}\n", .{ eps, es.rms, es.p99, es.pairs });
            try testing.expect(es.rms <= 1e-3 and es.p99 <= 5e-3);
        };
        try testing.expect(@as(f64, @floatFromInt(t.nodes.len)) / n <= 0.12);
        try testing.expectApproxEqRel(total, t.nodes[0].mass, 1e-6);
        try testing.expectEqual(@as(u32, n), nodeCount(t.nodes[0]));

        // Every body is in exactly one leaf; every leaf is reached exactly once by a full rope walk.
        const seen = try gpa.alloc(u8, n);
        defer gpa.free(seen);
        @memset(seen, 0);
        const visited = try gpa.alloc(u8, t.nodes.len);
        defer gpa.free(visited);
        @memset(visited, 0);
        var idx: u32 = 0;
        while (idx != end) {
            const nd = t.nodes[idx];
            visited[idx] += 1;
            if (isLeaf(nd)) {
                const first = nd.child & ~leaf_flag;
                for (first..first + nodeCount(nd)) |j| seen[j] += 1;
                idx = nd.skip;
            } else idx = nd.child;
        }
        for (seen) |s| try testing.expectEqual(@as(u8, 1), s);
        for (visited) |v| try testing.expectEqual(@as(u8, 1), v);
        for (t.nodes) |nd| if (isLeaf(nd)) try testing.expect(nodeCount(nd) <= 32);
        for (0..n) |i| {
            // Per-body window agrees with the leaf partition.
            const r = leafRange(t.keys, i, 32);
            const owner = blk: {
                for (0..t.leaves) |l| {
                    const nd = t.nodes[t.leafNode(@intCast(l))];
                    const first = nd.child & ~leaf_flag;
                    if (i >= first and i < first + nodeCount(nd)) break :blk nd;
                }
                unreachable;
            };
            try testing.expectEqual(owner.child & ~leaf_flag, @as(u32, @intCast(r.lo)));
            try testing.expectEqual(nodeCount(owner), @as(u32, @intCast(r.hi - r.lo)));
            if (i >= 300) break;
        }
    }
}

test "ref3: tree is deterministic and independent of input order" {
    const gpa = testing.allocator;
    const n = 10_000;
    const cloud = try plummerCloud(gpa, n, 7);
    defer cloud.deinit(gpa);
    const prm: Params = .{ .box = test_box, .theta = 0.6 };
    const a = try build(gpa, cloud.pos, cloud.mass, prm);
    defer a.deinit(gpa);
    const b = try build(gpa, cloud.pos, cloud.mass, prm);
    defer b.deinit(gpa);

    var prng = std.Random.DefaultPrng.init(1);
    const sp = try gpa.dupe(Pos3, cloud.pos);
    defer gpa.free(sp);
    const idx = try gpa.alloc(u32, n);
    defer gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = @intCast(i);
    prng.random().shuffle(u32, idx);
    const sm = try gpa.alloc(f32, n);
    defer gpa.free(sm);
    for (idx, sp, sm) |o, *p, *m| {
        p.* = cloud.pos[o];
        m.* = cloud.mass[o];
    }
    const c = try build(gpa, sp, sm, prm);
    defer c.deinit(gpa);

    for ([_]Tree{ b, c }) |o| {
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a.nodes), std.mem.sliceAsBytes(o.nodes));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a.quads), std.mem.sliceAsBytes(o.quads));
        try testing.expectEqualSlices(u64, a.keys, o.keys);
    }
    try testing.expectEqualSlices(u32, a.perm, b.perm);
}

test "ref3: degenerate inputs (duplicates, one body, tiny N)" {
    const gpa = testing.allocator;
    var pos: [100]Pos3 = undefined;
    var mass: [100]f32 = undefined;
    for (&pos, &mass, 0..) |*p, *m, i| {
        p.* = if (i < 80) .{ .x = 1 << 30, .y = 1 << 30, .z = 1 << 30 } else .{ .x = @intCast(i * 1000003), .y = 5, .z = @intCast(i << 20) };
        m.* = 1;
    }
    for ([_]usize{ 1, 2, 33, 100 }) |k| {
        const t = try build(gpa, pos[0..k], mass[0..k], .{ .box = 1, .theta = 0.5 });
        defer t.deinit(gpa);
        try testing.expectApproxEqRel(@as(f32, @floatFromInt(k)), t.nodes[0].mass, 1e-6);
        try testing.expectEqual(@as(u32, @intCast(k)), nodeCount(t.nodes[0]));
        const a = t.walk(pos[0], null, 1, 1e-3, null);
        try testing.expect(std.math.isFinite(a.a[0]) and std.math.isFinite(a.pot));
    }
}

test "ref3: prefix-cell criterion also meets the error targets" {
    const gpa = testing.allocator;
    const cloud = try plummerCloud(gpa, 10_000, 42);
    defer cloud.deinit(gpa);
    const t = try build(gpa, cloud.pos, cloud.mass, .{ .box = test_box, .theta = 0.6, .tight = false });
    defer t.deinit(gpa);
    const e = try forceError(gpa, t, cloud.pos, cloud.mass, 1000);
    try testing.expect(e.rms <= 1e-3);
}

test "ref3: two tight clumps 1 to 40 softening lengths apart match direct sum" {
    for ([_]f64{ 1, 2, 3, 4, 6, 10, 40 }) |sep| try clumps(sep);
}

/// Two 400-body clumps of size 0.2 eps, `sep` eps apart: nodes accepted near and inside eps.
fn clumps(sep: f64) !void {
    const gpa = testing.allocator;
    const eps = 1e-4;
    const per = 400;
    var pos: [2 * per]Pos3 = undefined;
    var mass: [2 * per]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    for (&pos, &mass, 0..) |*p, *m, i| {
        const cx: f64 = if (i < per) 0.3 else 0.3 + sep * eps;
        const q = struct {
            fn f(c: f64, j: f64) u32 {
                return @intFromFloat((c + (j - 0.5) * 2e-5) * 4294967296.0);
            }
        }.f;
        p.* = .{ .x = q(cx, r.float(f64)), .y = q(0.5, r.float(f64)), .z = q(0.5, r.float(f64)) };
        m.* = 1;
    }
    const t = try build(gpa, &pos, &mass, .{ .box = 1, .theta = 0.6 });
    defer t.deinit(gpa);
    const inv = try gpa.alloc(u32, pos.len);
    defer gpa.free(inv);
    for (t.perm, 0..) |o, s| inv[o] = @intCast(s);
    for (pos, 0..) |p, i| {
        const ex = directAt(&pos, &mass, 1, p, i, 1, eps);
        const ap = t.walk(p, inv[i], 1, eps, null);
        var d2: f64 = 0;
        var e2: f64 = 0;
        for (0..3) |k| {
            d2 += (ap.a[k] - ex.a[k]) * (ap.a[k] - ex.a[k]);
            e2 += ex.a[k] * ex.a[k];
        }
        try testing.expect(@sqrt(d2 / e2) < 1e-3);
    }
}

test {
    _ = plummer;
}
