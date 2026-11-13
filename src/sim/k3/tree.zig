//! Tree build of the 3D sim (docs/sim3d.md, work package 4): leaf buckets, Karras radix tree,
//! ropes and depth-ordered moments over bodies already sorted by Morton key. src/sim/ref3.zig is
//! the spec; tree3.zig records the passes in this order:
//!   leafCount -> scanBlock/scanAdd (multi-level) -> leafWrite -> leafFinish -> karras ->
//!   depthLeaf -> moments (once per depth, deepest first).
//! No atomics: every output word has exactly one writer, so a rerun is bit-identical.
const C = @import("common.zig");
const L = C.L;

/// Uniform block; must stay identical to `KParams` in src/sim/tree3.zig. Scratch offsets are in
/// u32 words. Every index into a buffer goes through a uniform or a thread id: a comptime-folded
/// index miscompiles in the SPIR-V backend.
pub const Params = extern struct {
    n: u32,
    node_cap: u32,
    leaf_cap: u32,
    depth: u32,
    box_scale: f32,
    theta: f32,
    scan_count: u32,
    scan_data: u32,
    scan_sums: u32,
    off_hdr: u32,
    off_lstart: u32,
    off_lkey: u32,
    off_parent: u32,
    off_bbox: u32,
    off_sums0: u32,
    off_total: u32,
    /// 1.0, opaque to the driver compiler: see Kahan.
    one: f32,
    off_rem: u32,
    _pad0: u32,
    _pad1: u32,
};

const pos = C.ro(L.Pos3, 0, "pos");
const mass = C.ro(f32, 1, "mass");
const keys = C.ro(L.SortRec, 2, "keys");
const nodes = C.rw(L.Node, 0, "nodes");
const quads = C.rw(L.NodeQuad, 1, "quads");
const scr = C.rw(u32, 2, "scratch");
/// Indirect dispatch arguments and the host-visible status; only leafFinish binds it.
const args = C.rw(u32, 3, "args");
const P = C.uniform(Params, "params");

const gs = C.group_size;
pub const leaf_cap_bodies: u32 = 32;
pub const per_thread: u32 = 8;
/// Bodies per leafCount/leafWrite group and elements per scanBlock group.
pub const block: u32 = gs * per_thread;
const tile_n = block + 2 * leaf_cap_bodies;
const leaf_flag: u32 = 0x8000_0000;
const end: u32 = 0xFFFF_FFFF;
const depth_shift = 26;
const count_mask: u32 = (1 << depth_shift) - 1;

// ---- leaf buckets ----

const thi = C.Shared(u32, tile_n, "thi");
const tlo = C.Shared(u32, tile_n, "tlo");
const red = C.Shared(u32, gs, "red");

inline fn clz64(hi: u32, lo: u32) u32 {
    return if (hi != 0) C.clz(hi) else 32 + C.clz(lo);
}

/// ref3.commonBits: leading bits two 63-bit keys share, 63 when equal.
inline fn commonBits(ah: u32, al: u32, bh: u32, bl: u32) u32 {
    const xh = ah ^ bh;
    const xl = al ^ bl;
    if (xh == 0 and xl == 0) return 63;
    return clz64(xh, xl) - 1;
}

/// Keys [base - cap, base + block + cap) into the tile; out-of-range slots are never read.
fn loadTile(base: u32, l: u32) void {
    var t = l;
    while (t < tile_n) : (t += gs) {
        const g = (base +% t) -% leaf_cap_bodies;
        if (g < P.n) {
            const r = C.load(keys, g);
            thi[t] = r.key_hi;
            tlo[t] = r.key_lo;
        }
        // Wrapped g lands far above n for the halo before body 0.
    }
    C.barrier();
}

/// Whether sorted body i starts a leaf (ref3.leafRange(i).lo == i); t is its tile slot.
/// The leaf of i is the largest prefix cell of at most cap bodies, or the run of i's key when even
/// the full key holds more. Cells are contiguous in sorted order, so i starts its leaf exactly when
/// the cell of its left neighbour's common prefix c1 is too big, i.e. at least cap of the
/// +-cap neighbours share c1 bits with i (the same window ref3 scans). c1 = 63 (equal keys) never
/// starts a leaf: equal keys are never split.
fn leafStart(i: u32, t: u32) bool {
    if (i >= P.n) return false;
    if (i == 0) return true;
    const kh = thi[t];
    const kl = tlo[t];
    const c1 = commonBits(thi[t - 1], tlo[t - 1], kh, kl);
    if (c1 == 63) return false;
    var cnt: u32 = 1;
    var k: u32 = 2;
    while (k <= leaf_cap_bodies and k <= i and commonBits(thi[t - k], tlo[t - k], kh, kl) >= c1) : (k += 1) cnt += 1;
    k = 1;
    while (cnt < leaf_cap_bodies and k <= leaf_cap_bodies and i + k < P.n and commonBits(thi[t + k], tlo[t + k], kh, kl) >= c1) : (k += 1) cnt += 1;
    return cnt >= leaf_cap_bodies;
}

/// Leaf starts of this thread's 8 consecutive bodies as a bit mask.
fn startMask(base: u32, l: u32) u32 {
    var bits: u32 = 0;
    var q: u32 = 0;
    while (q < per_thread) : (q += 1) {
        const o = l * per_thread + q;
        if (leafStart(base + o, o + leaf_cap_bodies)) bits |= @as(u32, 1) << @as(u5, @truncate(q));
    }
    return bits;
}

inline fn popcount8(x: u32) u32 {
    var v = x;
    var c: u32 = 0;
    while (v != 0) : (v &= v - 1) c += 1;
    return c;
}

/// Leaf starts per block of 2048 bodies into sums0.
export fn leafCount() callconv(C.cc) void {
    const g = C.groupId();
    const base = g * block;
    if (base >= P.n) return;
    const l = C.lid();
    loadTile(base, l);
    red[l] = popcount8(startMask(base, l));
    var off: u32 = gs / 2;
    while (off > 0) : (off >>= 1) {
        C.barrier();
        if (l < off) red[l] += red[l + off];
    }
    C.barrier();
    if (l == 0) C.store(scr, P.off_sums0 + g, red[l]);
}

/// Inclusive Hillis-Steele scan of one value per thread; returns the exclusive prefix.
fn groupScan(l: u32, v: u32) u32 {
    red[l] = v;
    C.barrier();
    var off: u32 = 1;
    while (off < gs) : (off <<= 1) {
        const add = if (l >= off) red[l - off] else 0;
        C.barrier();
        red[l] += add;
        C.barrier();
    }
    return red[l] - v;
}

/// Writes lstart[leaf] = first body and the leaf's key, using the scanned block offsets.
export fn leafWrite() callconv(C.cc) void {
    const g = C.groupId();
    const base = g * block;
    if (base >= P.n) return;
    const l = C.lid();
    loadTile(base, l);
    const bits = startMask(base, l);
    var j = C.load(scr, P.off_sums0 + g) + groupScan(l, popcount8(bits));
    var q: u32 = 0;
    while (q < per_thread) : (q += 1) {
        if (bits & (@as(u32, 1) << @as(u5, @truncate(q))) != 0) {
            const o = l * per_thread + q;
            if (j < P.leaf_cap) {
                C.store(scr, P.off_lstart + j, base + o);
                C.store(scr, P.off_lkey + 2 * j, thi[o + leaf_cap_bodies]);
                C.store(scr, P.off_lkey + 2 * j + 1, tlo[o + leaf_cap_bodies]);
            }
            j += 1;
        }
    }
}

/// In-place exclusive scan of scratch[scan_data ..][0..scan_count] in blocks of 2048; each
/// block's total goes to scratch[scan_sums + block].
export fn scanBlock() callconv(C.cc) void {
    const g = C.groupId();
    if (g * block >= P.scan_count) return;
    const l = C.lid();
    const first = g * block + l * per_thread;
    var vals: [per_thread]u32 = undefined;
    var total: u32 = 0;
    inline for (0..per_thread) |k| {
        const idx = first + @as(u32, k);
        const x = if (idx < P.scan_count) C.load(scr, P.scan_data + idx) else 0;
        vals[k] = total;
        total += x;
    }
    const excl = groupScan(l, total);
    inline for (0..per_thread) |k| {
        const idx = first + @as(u32, k);
        if (idx < P.scan_count) C.store(scr, P.scan_data + idx, vals[k] + excl);
    }
    if (l == gs - 1) C.store(scr, P.scan_sums + g, excl + total);
}

/// scratch[scan_data + i] += scratch[scan_sums + i / 2048].
export fn scanAdd() callconv(C.cc) void {
    const g = C.groupId();
    if (g * gs >= P.scan_count) return;
    const i = g * gs + C.lid();
    if (i >= P.scan_count) return;
    C.store(scr, P.scan_data + i, C.load(scr, P.scan_data + i) + C.load(scr, P.scan_sums + i / block));
}

fn writeDispatch(o: u32, threads: u32) void {
    const groups = (threads + gs - 1) / gs;
    C.store(args, o, @min(groups, C.dispatch_x));
    C.store(args, o + 1, (groups + C.dispatch_x - 1) / C.dispatch_x);
    C.store(args, o + 2, 1);
}

/// One thread: leaf count, overflow check, end sentinel, indirect arguments and status.
/// args: [0..3] dispatch over leaves, [3..6] dispatch over all nodes, [6] leaves, [7] nodes,
/// [8] overflow. On overflow every later pass dispatches nothing.
export fn leafFinish() callconv(C.cc) void {
    const l = C.lid();
    if (l != 0 or C.groupId() != 0) return;
    const m = C.load(scr, P.off_total);
    const total = 2 * m - 1;
    const ok = m <= P.leaf_cap and total <= P.node_cap;
    C.store(scr, P.off_hdr, m);
    if (ok) C.store(scr, P.off_lstart + m, P.n);
    // l is 0 here; it keeps the indices runtime values.
    writeDispatch(l, if (ok) m else 0);
    writeDispatch(l + 3, if (ok) total else 0);
    C.store(args, l + 6, m);
    C.store(args, l + 7, total);
    C.store(args, l + 8, if (ok) 0 else 1);
}

// ---- topology ----

inline fn lkeyHi(j: u32) u32 {
    return C.load(scr, P.off_lkey + 2 * j);
}
inline fn lkeyLo(j: u32) u32 {
    return C.load(scr, P.off_lkey + 2 * j + 1);
}

/// Karras delta plus one (0 for j outside [0, m)), so comparisons stay unsigned. j may be a
/// wrapped "negative" index. Leaf keys are distinct, so i != j never gives equal keys.
fn kdelta(i: u32, j: u32, m: u32) u32 {
    if (j >= m) return 0;
    return 1 + clz64(lkeyHi(i) ^ lkeyHi(j), lkeyLo(i) ^ lkeyLo(j));
}

inline fn stepBy(i: u32, x: u32, up: bool) u32 {
    return if (up) i +% x else i -% x;
}

/// Rope of a node whose last leaf is s - 1: the topmost node starting at leaf s, which is the right
/// child of the split at s - 1. That child is internal node s when node s extends rightwards
/// (Karras: internal node s has s at one end of its range), otherwise leaf s. This equals ref3's
/// top-down skip[left] = right, skip[right] = skip[parent].
fn rope(s: u32, m: u32) u32 {
    if (s >= m) return end;
    if (s + 1 < m and kdelta(s, s + 1, m) > kdelta(s, s - 1, m)) return s;
    return m - 1 + s;
}

inline fn lstart(j: u32) u32 {
    return C.load(scr, P.off_lstart + j);
}

/// Thread j: leaf j (node m-1+j) and, for j < m-1, internal node j (ref3's Karras loop).
/// Moments are filled in later; parent links feed depthLeaf.
export fn karras() callconv(C.cc) void {
    const g = C.groupId();
    const m = C.load(scr, P.off_hdr);
    if (g * gs >= m) return;
    const j = g * gs + C.lid();
    if (j >= m) return;
    const first = lstart(j);
    const past = lstart(j + 1);
    C.store(nodes, m - 1 + j, .{
        .com = .{ .x = 0, .y = 0, .z = 0 },
        .mass = 0,
        .open_r2 = 0,
        .child = leaf_flag | first,
        .skip = rope(j + 1, m),
        .info = @min(past - first, count_mask),
    });
    if (j + 1 >= m) return;

    const i = j;
    const up = kdelta(i, i + 1, m) > kdelta(i, i -% 1, m);
    const dmin = kdelta(i, stepBy(i, 1, !up), m);
    var lmax: u32 = 2;
    while (kdelta(i, stepBy(i, lmax, up), m) > dmin) lmax *= 2;
    var len: u32 = 0;
    var t = lmax >> 1;
    while (t >= 1) : (t >>= 1) {
        if (kdelta(i, stepBy(i, len + t, up), m) > dmin) len += t;
    }
    const other = stepBy(i, len, up);
    const dnode = kdelta(i, other, m);
    var s: u32 = 0;
    t = len;
    while (true) {
        t = (t + 1) >> 1;
        if (kdelta(i, stepBy(i, s + t, up), m) > dnode) s += t;
        if (t == 1) break;
    }
    const gamma = if (up) i + s else i - s - 1;
    const a = @min(i, other);
    const b = @max(i, other);
    const left = if (a == gamma) m - 1 + gamma else gamma;
    const right = if (b == gamma + 1) m - 1 + gamma + 1 else gamma + 1;
    C.store(nodes, i, .{
        .com = .{ .x = 0, .y = 0, .z = 0 },
        .mass = 0,
        .open_r2 = 0,
        .child = left,
        .skip = rope(b + 1, m),
        .info = @min(lstart(b + 1) - lstart(a), count_mask),
    });
    C.store(scr, P.off_parent + left, i);
    C.store(scr, P.off_parent + right, i);
}

// ---- moments ----

/// a - b in quanta without wrapping: the tree lives in absolute coordinates (ref3.phys), so a
/// periodic minimum image would be wrong here.
inline fn sdelta(a: u32, b: u32) f32 {
    return if (a >= b) @floatFromInt(a - b) else -@as(f32, @floatFromInt(b - a));
}

/// base + round(off), clamped to [lo, hi].
fn offsetIn(base: u32, off: f32, lo: u32, hi: u32) u32 {
    // 4294967040 is the largest f32 below 2^32.
    const r = @round(@min(@abs(off), 4294967040.0));
    const d: u32 = @intFromFloat(r);
    const v = if (off >= 0) base + @min(d, ~base) else base - @min(d, base);
    return @min(@max(v, lo), hi);
}

/// Compensated sums for leaf moments: a leaf of one long equal-key run can hold thousands of
/// bodies, where plain f32 accumulation drifts by ~count * 2^-24. SPIR-V ops without NoContraction
/// are inexact to the driver, which folds (s + y) - s to y and drops the compensation; the
/// multiply by the uniform 1.0 hides that pattern.
fn Kahan(comptime n: usize) type {
    return struct {
        s: [n]f32 = @splat(0),
        c: [n]f32 = @splat(0),
        inline fn add(k: @This(), x: [n]f32) @This() {
            var r = k;
            inline for (0..n) |i| {
                const y = x[i] - k.c[i];
                const t = k.s[i] + y;
                r.c[i] = (t * P.one - k.s[i]) - y;
                r.s[i] = t;
            }
            return r;
        }
    };
}

const Quad = struct { xx: f32 = 0, xy: f32 = 0, xz: f32 = 0, yy: f32 = 0, yz: f32 = 0, zz: f32 = 0 };

inline fn addQuad(q: Quad, m: f32, x: f32, y: f32, z: f32) Quad {
    const r2 = x * x + y * y + z * z;
    return .{
        .xx = q.xx + m * (3 * x * x - r2),
        .xy = q.xy + m * 3 * x * y,
        .xz = q.xz + m * 3 * x * z,
        .yy = q.yy + m * (3 * y * y - r2),
        .yz = q.yz + m * 3 * y * z,
        .zz = q.zz + m * (3 * z * z - r2),
    };
}

/// (side/theta + |com - box centre|)^2 in physical units, from the tight integer bounding box.
fn openR2(com: L.Pos3, lo: L.Pos3, hi: L.Pos3) f32 {
    const sx: f32 = @floatFromInt(hi.x - lo.x);
    const sy: f32 = @floatFromInt(hi.y - lo.y);
    const sz: f32 = @floatFromInt(hi.z - lo.z);
    const dx = sdelta(com.x, lo.x) - 0.5 * sx;
    const dy = sdelta(com.y, lo.y) - 0.5 * sy;
    const dz = sdelta(com.z, lo.z) - 0.5 * sz;
    const reach = (@max(sx, @max(sy, sz)) / P.theta + @sqrt(dx * dx + dy * dy + dz * dz)) * P.box_scale;
    return reach * reach;
}

fn storeBox(x: u32, lo: L.Pos3, hi: L.Pos3) void {
    const o = P.off_bbox + 6 * x;
    C.store(scr, o, lo.x);
    C.store(scr, o + 1, lo.y);
    C.store(scr, o + 2, lo.z);
    C.store(scr, o + 3, hi.x);
    C.store(scr, o + 4, hi.y);
    C.store(scr, o + 5, hi.z);
}

inline fn boxLo(x: u32) L.Pos3 {
    const o = P.off_bbox + 6 * x;
    return .{ .x = C.load(scr, o), .y = C.load(scr, o + 1), .z = C.load(scr, o + 2) };
}
inline fn boxHi(x: u32) L.Pos3 {
    const o = P.off_bbox + 6 * x + 3;
    return .{ .x = C.load(scr, o), .y = C.load(scr, o + 1), .z = C.load(scr, o + 2) };
}

/// Exact centre of mass minus the stored (rounded) one, in quanta. Child quadrupoles are about
/// their exact centres, so the parallel-axis shift must use those, not the rounded ones (the
/// difference is a first-order error for nodes only a few thousand quanta across).
fn storeRem(x: u32, r: L.Vel3) void {
    const o = P.off_rem + 3 * x;
    C.store(scr, o, @bitCast(r.x));
    C.store(scr, o + 1, @bitCast(r.y));
    C.store(scr, o + 2, @bitCast(r.z));
}

inline fn loadRem(x: u32) L.Vel3 {
    const o = P.off_rem + 3 * x;
    return .{ .x = @bitCast(C.load(scr, o)), .y = @bitCast(C.load(scr, o + 1)), .z = @bitCast(C.load(scr, o + 2)) };
}

inline fn storeQuad(x: u32, q: Quad) void {
    C.store(quads, x, .{ .xx = q.xx, .xy = q.xy, .xz = q.xz, .yy = q.yy, .yz = q.yz, .zz = q.zz });
}

/// Thread x over all 2m-1 nodes: depth from the parent links, and leaf moments. A leaf's centre of
/// mass is an f32 offset from its first body, so it keeps full precision whatever the box.
export fn depthLeaf() callconv(C.cc) void {
    const g = C.groupId();
    const m = C.load(scr, P.off_hdr);
    const total = 2 * m - 1;
    if (g * gs >= total) return;
    const x = g * gs + C.lid();
    if (x >= total) return;
    var d: u32 = 0;
    var y = x;
    // Internal depth <= 62 and leaf depth <= 63 (distinct 63-bit leaf keys), so 64 only guards bugs.
    while (y != 0 and d < 64) : (d += 1) y = C.load(scr, P.off_parent + y);
    var nd = C.load(nodes, x);
    nd.info |= d << depth_shift;
    if (x >= m - 1) {
        const first = nd.child & ~leaf_flag;
        const cnt = nd.info & count_mask;
        const p0 = C.load(pos, first);
        var lo = p0;
        var hi = p0;
        var s1: Kahan(4) = .{};
        var k: u32 = 0;
        while (k < cnt) : (k += 1) {
            const p = C.load(pos, first + k);
            const w = C.load(mass, first + k);
            s1 = s1.add(.{ w, w * sdelta(p.x, p0.x), w * sdelta(p.y, p0.y), w * sdelta(p.z, p0.z) });
            lo = .{ .x = @min(lo.x, p.x), .y = @min(lo.y, p.y), .z = @min(lo.z, p.z) };
            hi = .{ .x = @max(hi.x, p.x), .y = @max(hi.y, p.y), .z = @max(hi.z, p.z) };
        }
        const msum = s1.s[0];
        // Massless leaves keep their first body as centre instead of 0/0.
        const inv: f32 = if (msum > 0) 1 / msum else 0;
        const o: L.Vel3 = .{ .x = s1.s[1] * inv, .y = s1.s[2] * inv, .z = s1.s[3] * inv };
        const com: L.Pos3 = .{ .x = offsetIn(p0.x, o.x, lo.x, hi.x), .y = offsetIn(p0.y, o.y, lo.y, hi.y), .z = offsetIn(p0.z, o.z, lo.z, hi.z) };
        const sc = P.box_scale;
        var s2: Kahan(6) = .{};
        k = 0;
        while (k < cnt) : (k += 1) {
            const p = C.load(pos, first + k);
            const t = addQuad(.{}, C.load(mass, first + k), (sdelta(p.x, p0.x) - o.x) * sc, (sdelta(p.y, p0.y) - o.y) * sc, (sdelta(p.z, p0.z) - o.z) * sc);
            s2 = s2.add(.{ t.xx, t.xy, t.xz, t.yy, t.yz, t.zz });
        }
        const q: Quad = .{ .xx = s2.s[0], .xy = s2.s[1], .xz = s2.s[2], .yy = s2.s[3], .yz = s2.s[4], .zz = s2.s[5] };
        nd.com = com;
        nd.mass = msum;
        nd.open_r2 = openR2(com, lo, hi);
        storeQuad(x, q);
        storeBox(x, lo, hi);
        storeRem(x, .{ .x = o.x - sdelta(com.x, p0.x), .y = o.y - sdelta(com.y, p0.y), .z = o.z - sdelta(com.z, p0.z) });
    }
    C.store(nodes, x, nd);
}

/// Internal nodes at depth P.depth from their two finished children (left = child, right =
/// skip(left)), in ref3.combine's order. Dispatched over leaves; thread i handles node i < m-1.
export fn moments() callconv(C.cc) void {
    const g = C.groupId();
    const m = C.load(scr, P.off_hdr);
    if (g * gs + 1 >= m) return;
    const i = g * gs + C.lid();
    if (i + 1 >= m) return;
    var nd = C.load(nodes, i);
    if (nd.info >> depth_shift != P.depth) return;
    const ia = nd.child;
    const a = C.load(nodes, ia);
    const ib = a.skip;
    const b = C.load(nodes, ib);
    const la = boxLo(ia);
    const ha = boxHi(ia);
    const lb = boxLo(ib);
    const hb = boxHi(ib);
    const lo: L.Pos3 = .{ .x = @min(la.x, lb.x), .y = @min(la.y, lb.y), .z = @min(la.z, lb.z) };
    const hi: L.Pos3 = .{ .x = @max(ha.x, hb.x), .y = @max(ha.y, hb.y), .z = @max(ha.z, hb.z) };
    const msum = a.mass + b.mass;
    const w: f32 = if (msum > 0) b.mass / msum else 0;
    // Exact centres relative to a's stored (rounded) one.
    const ea = loadRem(ia);
    const rb = loadRem(ib);
    const eb: L.Vel3 = .{ .x = sdelta(b.com.x, a.com.x) + rb.x, .y = sdelta(b.com.y, a.com.y) + rb.y, .z = sdelta(b.com.z, a.com.z) + rb.z };
    const o: L.Vel3 = .{ .x = ea.x + w * (eb.x - ea.x), .y = ea.y + w * (eb.y - ea.y), .z = ea.z + w * (eb.z - ea.z) };
    const com: L.Pos3 = .{ .x = offsetIn(a.com.x, o.x, lo.x, hi.x), .y = offsetIn(a.com.y, o.y, lo.y, hi.y), .z = offsetIn(a.com.z, o.z, lo.z, hi.z) };
    const qa = C.load(quads, ia);
    const qb = C.load(quads, ib);
    var q: Quad = .{ .xx = qa.xx + qb.xx, .xy = qa.xy + qb.xy, .xz = qa.xz + qb.xz, .yy = qa.yy + qb.yy, .yz = qa.yz + qb.yz, .zz = qa.zz + qb.zz };
    const sc = P.box_scale;
    q = addQuad(q, a.mass, (ea.x - o.x) * sc, (ea.y - o.y) * sc, (ea.z - o.z) * sc);
    q = addQuad(q, b.mass, (eb.x - o.x) * sc, (eb.y - o.y) * sc, (eb.z - o.z) * sc);
    nd.com = com;
    nd.mass = msum;
    nd.open_r2 = openR2(com, lo, hi);
    C.store(nodes, i, nd);
    storeQuad(i, q);
    storeBox(i, lo, hi);
    storeRem(i, .{ .x = o.x - sdelta(com.x, a.com.x), .y = o.y - sdelta(com.y, a.com.y), .z = o.z - sdelta(com.z, a.com.z) });
}
