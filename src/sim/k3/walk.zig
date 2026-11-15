//! Barnes-Hut tree kick (docs/sim3d.md, work package 5). Two walks over the radix tree's ropes, both
//! f32 ports of ref3 (the spec), with the same 3-eps guard, softened leaf pairs and Plummer-softened
//! monopole + traceless quadrupole for accepted nodes:
//!
//! - `walk<gs>` (ref3.Tree.walk): one independent walk per body.
//! - `walkGroup64` (ref3.Tree.walkGroups): one workgroup (one wave64 on RDNA3) per group of at most
//!   64 bodies, a subtree of the tree. The group shares one walk, and a node is accepted only if
//!   the group's whole bounding box would accept it, so every body gets the interactions its own
//!   walk would have had or finer ones. Every decision is the same for the whole workgroup and
//!   depends only on uniform data: the node fetches are scalar loads, one per group instead of one
//!   per body, the walk never diverges, and an opened leaf's bodies are fetched once into shared
//!   memory for the whole group.
//!
//! The interaction lists agree with ref3 except where f32 and f64 round a squared distance to
//! opposite sides of open_r2. Each body sums its terms in one fixed order and nothing is shared
//! between threads except loaded inputs, so a rerun is bit-identical.
const C = @import("common.zig");
const L = C.L;

const pos = C.ro(L.Pos3, 0, "pos");
const mass = C.ro(f32, 1, "mass");
const nodes = C.ro(L.Node, 2, "nodes");
const quads = C.ro(L.NodeQuad, 3, "quads");
/// Tree3's status words (tree3.zig status_*); only the group walk reads them.
const status = C.ro(u32, 4, "status");
const vel = C.rw(L.Vel3, 0, "vel");
const amag = C.rw(f32, 1, "amag");
const pot = C.rw(f32, 2, "pot");
const params = C.uniform(L.WalkParams3, "params");

const leaf_flag: u32 = 0x8000_0000;
const end: u32 = 0xFFFF_FFFF;
const count_mask: u32 = (1 << 26) - 1;
const status_leaves: u32 = 6;
const status_overflow: u32 = 8;

/// One tile of an opened leaf (walkGroup): positions, and masses over box_scale^2 (quantaPair).
const max_group: u32 = 64;
const sh_px = C.Shared(u32, max_group, "sh_px");
const sh_py = C.Shared(u32, max_group, "sh_py");
const sh_pz = C.Shared(u32, max_group, "sh_pz");
const sh_m = C.Shared(f32, max_group, "sh_m");

// A leaf above max_group bodies must be one run of equal keys (bigLeafBlock); that holds only if no leaf is capped above it.
comptime {
    if (C.leaf_cap_bodies > max_group) @compileError("tree leaf cap exceeds the walk's max_group");
}

fn cc(comptime x: u32) @import("std").builtin.CallingConvention {
    return C.ccFor(x);
}

/// Profiling switches (walk3_tests "profile"); the shipped entry points use the defaults.
const Variant = struct {
    /// vel = (visits, cells, pairs) instead of the kick.
    stats: bool = false,
    /// Leaf pairs without their pos/mass loads.
    no_leaf_mem: bool = false,
    /// Skip leaf pairs altogether.
    no_leaf: bool = false,
};

const Acc = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    phi: f32 = 0,
    /// Leaf-pair potential over bs (quantaPair).
    phi_q: f32 = 0,
    visits: u32 = 0,
    cells: u32 = 0,
    pairs: u32 = 0,
};

/// Far field of node `idx`; r is target minus com.
inline fn cellTerm(a: *Acc, q: L.NodeQuad, nmass: f32, rx: f32, ry: f32, rz: f32, r2: f32, eps2: f32) void {
    a.cells += 1;
    const inv = 1.0 / @sqrt(r2 + eps2);
    const inv2 = inv * inv;
    const inv3 = inv2 * inv;
    const inv5 = inv3 * inv2;
    const qx = q.xx * rx + q.xy * ry + q.xz * rz;
    const qy = q.xy * rx + q.yy * ry + q.yz * rz;
    const qz = q.xz * rx + q.yz * ry + q.zz * rz;
    const rqr = rx * qx + ry * qy + rz * qz;
    const mono = -nmass * inv3;
    const quad_r = -2.5 * rqr * inv5 * inv2;
    a.x += mono * rx + qx * inv5 + quad_r * rx;
    a.y += mono * ry + qy * inv5 + quad_r * ry;
    a.z += mono * rz + qz * inv5 + quad_r * rz;
    a.phi += -nmass * inv - 0.5 * rqr * inv5;
}

/// Softened pair with body p of mass m.
inline fn pairTerm(a: *Acc, me: L.Pos3, p: L.Pos3, m: f32, bs: f32, eps2: f32) void {
    a.pairs += 1;
    const dx = C.delta(p.x, me.x) * bs;
    const dy = C.delta(p.y, me.y) * bs;
    const dz = C.delta(p.z, me.z) * bs;
    const inv = 1.0 / @sqrt(dx * dx + dy * dy + dz * dz + eps2);
    const w = m * inv * inv * inv;
    a.x += w * dx;
    a.y += w * dy;
    a.z += w * dz;
    a.phi -= m * inv;
}

inline fn finish(i: u32, a: Acc, comptime vr: Variant) void {
    if (vr.stats) {
        C.store(vel, i, .{ .x = @floatFromInt(a.visits), .y = @floatFromInt(a.cells), .z = @floatFromInt(a.pairs) });
        return;
    }
    const g = params.g;
    const gx = a.x * g;
    const gy = a.y * g;
    const gz = a.z * g;
    if (params.flags & L.sim3_flag_store_accel != 0) {
        C.store(vel, i, .{ .x = gx, .y = gy, .z = gz });
    } else {
        const v = C.load(vel, i);
        const k = params.kick;
        C.store(vel, i, .{ .x = v.x + gx * k, .y = v.y + gy * k, .z = v.z + gz * k });
    }
    C.store(amag, i, @sqrt(gx * gx + gy * gy + gz * gz));
    if (params.flags & L.sim3_flag_potential != 0) {
        C.store(pot, i, g * (a.phi + a.phi_q * params.box_scale));
    }
}

inline fn walkBody(comptime gs: u32, comptime vr: Variant) void {
    const group = C.groupId();
    // Whole groups past this chunk (a 2D dispatch rounds up) leave before touching anything.
    if (group * gs >= params.count) return;
    const local = group * gs + C.lid();
    // The tail of the last group may belong to the next chunk.
    if (local >= params.count) return;
    const i = params.first + local;

    const me = C.load(pos, i);
    const bs = params.box_scale;
    const eps2 = params.eps2;
    // ref3.near_k: a node is never accepted inside 3 softening lengths.
    const near2 = 9 * eps2;
    var a: Acc = .{};
    var idx: u32 = 0;
    while (idx != end) {
        const nd = C.load(nodes, idx);
        a.visits += 1;
        // Target minus com, physical units.
        const rx = C.delta(me.x, nd.com.x) * bs;
        const ry = C.delta(me.y, nd.com.y) * bs;
        const rz = C.delta(me.z, nd.com.z) * bs;
        const r2 = rx * rx + ry * ry + rz * rz;
        if (r2 > nd.open_r2 and r2 > near2) {
            cellTerm(&a, C.load(quads, idx), nd.mass, rx, ry, rz, r2, eps2);
            idx = nd.skip;
        } else if (nd.child & leaf_flag != 0) {
            const first = nd.child & ~leaf_flag;
            const stop = if (vr.no_leaf) first else first + (nd.info & count_mask);
            var j = first;
            while (j < stop) : (j += 1) {
                // Self pair skipped by index, as in ref3: subtracting its m/eps afterwards cancels
                // a term that can be larger than the whole potential.
                if (j != i) {
                    const p = if (vr.no_leaf_mem) L.Pos3{ .x = me.x +% j, .y = me.y, .z = me.z } else C.load(pos, j);
                    const m = if (vr.no_leaf_mem) bs else C.load(mass, j);
                    pairTerm(&a, me, p, m, bs, eps2);
                }
            }
            idx = nd.skip;
        } else idx = nd.child;
    }
    finish(i, a, vr);
}

/// Two roles per workgroup w, both checked by every workgroup:
/// - Group nodes: the topmost nodes holding at most gs bodies. Workgroup 0 checks the root;
///   workgroup 1 + 2p + side checks child `side` of internal node p, which is a group node when p
///   holds more than gs bodies and the child does not.
/// - Big leaves: a leaf of more than gs bodies (one shared key) is walked in the blocks
///   [q gs, q gs + gs) of the body order it overlaps, block q by workgroup q, so its cost (each
///   of its bodies sums all of the others) is spread over workgroups instead of serialised.
/// The host dispatches max(1 + 2 * (a bound on the leaf count), n / gs) workgroups.
inline fn walkNodes(comptime gs: u32, comptime vr: Variant) void {
    const group = C.groupId();
    if (group >= params.count) return;
    const w = params.first + group;
    const lid = C.lid();
    // Runtime index base: a comptime index into a buffer array field miscompiles (k3/tree.zig).
    const s0 = params.count - params.count;
    // An overflowed build left an empty root: zero force for every body, as walkBody computes.
    if (C.load(status, s0 + status_overflow) != 0) {
        const i = w * gs + lid;
        if (i < params.n) finish(i, .{}, vr);
        return;
    }
    groupNode(gs, w, C.load(status, s0 + status_leaves), lid, vr);
    if (w < (params.n + gs - 1) / gs) bigLeafBlock(gs, w, lid, vr);
}

inline fn groupNode(comptime gs: u32, w: u32, m: u32, lid: u32, comptime vr: Variant) void {
    var c: u32 = 0;
    if (w != 0) {
        const p = (w - 1) / 2;
        if (p + 1 >= m) return;
        const pn = C.load(nodes, p);
        if (pn.info & count_mask <= gs) return;
        c = pn.child;
        if (w & 1 == 0) c = C.load(nodes, c).skip;
    }
    const cn = C.load(nodes, c);
    const count = cn.info & count_mask;
    // Internal nodes above gs are not groups; leaves above gs are bigLeafBlock's.
    if (count > gs) return;
    // First body: the leftmost leaf's.
    var dn = cn;
    while (dn.child & leaf_flag == 0) dn = C.load(nodes, dn.child);
    walkGroup(gs, dn.child & ~leaf_flag, count, lid, vr);
}

/// Leaves hold at most 32 bodies unless they share one 63-bit key (ref3.leafRange), so with
/// gs >= 32 a leaf of more than gs bodies is exactly a run of more than gs equal keys. Block q
/// meets at most two such runs: the one containing its first body and the one containing its last.
inline fn bigLeafBlock(comptime gs: u32, q: u32, lid: u32, comptime vr: Variant) void {
    const f = q * gs;
    const e = @min(f + gs, params.n) - 1;
    var after = f;
    if (inBigRun(gs, f)) {
        const hi = runEnd(f);
        walkGroup(gs, f, @min(hi, e) - f + 1, lid, vr);
        after = hi + 1;
    }
    if (e >= after and inBigRun(gs, e)) {
        const lo = runStart(e);
        walkGroup(gs, lo, e - lo + 1, lid, vr);
    }
}

inline fn sameKey(a: u32, b: u32) bool {
    const p = C.load(pos, a);
    const r = C.load(pos, b);
    return ((p.x ^ r.x) | (p.y ^ r.y) | (p.z ^ r.z)) >> 11 == 0;
}

/// Whether body x is in a run of more than gs equal keys. Such a run reaches 32 bodies to one side
/// of any of its members, so one or two compares reject almost every body before the searches.
inline fn inBigRun(comptime gs: u32, x: u32) bool {
    const n = params.n;
    const near = (x >= gs / 2 and sameKey(x, x - gs / 2)) or (x + gs / 2 < n and sameKey(x, x + gs / 2));
    return near and runEnd(x) - runStart(x) + 1 > gs;
}

/// First body of x's run of equal keys: doubling steps, then halving (all loads uniform).
inline fn runStart(x: u32) u32 {
    var lo = x;
    var step: u32 = 1;
    while (step <= lo and sameKey(lo - step, x)) {
        lo -= step;
        step *= 2;
    }
    // lo is in the run, lo - step is not (or does not exist).
    while (step > 1) {
        step /= 2;
        if (step <= lo and sameKey(lo - step, x)) lo -= step;
    }
    return lo;
}

/// Last body of x's run of equal keys.
inline fn runEnd(x: u32) u32 {
    const last = params.n - 1;
    var hi = x;
    var step: u32 = 1;
    while (step <= last - hi and sameKey(hi + step, x)) {
        hi += step;
        step *= 2;
    }
    while (step > 1) {
        step /= 2;
        if (step <= last - hi and sameKey(hi + step, x)) hi += step;
    }
    return hi;
}

/// The shared walk of bodies [g0, g0 + gn). Every thread runs it (barriers); threads past gn
/// shadow body g0 and store nothing.
inline fn walkGroup(comptime gs: u32, g0: u32, gn: u32, lid: u32, comptime vr: Variant) void {
    var lo = C.load(pos, g0);
    var hi = lo;
    var k: u32 = 1;
    while (k < gn) : (k += 1) {
        const p = C.load(pos, g0 + k);
        lo = .{ .x = @min(lo.x, p.x), .y = @min(lo.y, p.y), .z = @min(lo.z, p.z) };
        hi = .{ .x = @max(hi.x, p.x), .y = @max(hi.y, p.y), .z = @max(hi.z, p.z) };
    }
    const active = lid < gn;
    const i = if (active) g0 + lid else g0;
    const me = C.load(pos, i);
    const bs = params.box_scale;
    const eps2 = params.eps2;
    const near2 = 9 * eps2;
    // The target as an f32 offset from the box corner: a cell's r is then the uniform integer
    // corner-to-com offset plus mx, a conversion and a multiply-add per axis.
    const mx = C.delta(me.x, lo.x) * bs;
    const my = C.delta(me.y, lo.y) * bs;
    const mz = C.delta(me.z, lo.z) * bs;
    // Leaf pairs are summed in quanta: exact integer offsets, no scale per pair (quantaPair).
    const inv_bs2 = 1 / (bs * bs);
    const eps2_q = eps2 * inv_bs2;
    var a: Acc = .{};
    var idx: u32 = 0;
    while (idx != end) {
        const nd = C.load(nodes, idx);
        a.visits += 1;
        // Squared distance from the com to the nearest point of the box (ref3.boxDist2). The gaps
        // are uniform integers, so they stay on the scalar ALU; squared in quanta^2, scaled once.
        const gx: f32 = @floatFromInt(gap(lo.x, hi.x, nd.com.x));
        const gy: f32 = @floatFromInt(gap(lo.y, hi.y, nd.com.y));
        const gz: f32 = @floatFromInt(gap(lo.z, hi.z, nd.com.z));
        const d2 = (gx * gx + gy * gy + gz * gz) * (bs * bs);
        if (d2 > nd.open_r2 and d2 > near2) {
            const rx = C.delta(lo.x, nd.com.x) * bs + mx;
            const ry = C.delta(lo.y, nd.com.y) * bs + my;
            const rz = C.delta(lo.z, nd.com.z) * bs + mz;
            cellTerm(&a, C.load(quads, idx), nd.mass, rx, ry, rz, rx * rx + ry * ry + rz * rz, eps2);
            idx = nd.skip;
        } else if (nd.child & leaf_flag != 0) {
            const lfirst = nd.child & ~leaf_flag;
            const stop = if (vr.no_leaf) lfirst else lfirst + (nd.info & count_mask);
            var tb = lfirst;
            while (tb < stop) : (tb += gs) {
                const tn = @min(gs, stop - tb);
                // The previous tile is consumed before it is overwritten.
                C.barrier();
                if (lid < tn) {
                    const p = if (vr.no_leaf_mem) L.Pos3{ .x = lo.x +% lid, .y = lo.y, .z = lo.z } else C.load(pos, tb + lid);
                    sh_px[lid] = p.x;
                    sh_py[lid] = p.y;
                    sh_pz[lid] = p.z;
                    sh_m[lid] = (if (vr.no_leaf_mem) bs else C.load(mass, tb + lid)) * inv_bs2;
                }
                C.barrier();
                // Self pair by index, as in ref3; branch-free so the loop can be unrolled.
                var t: u32 = 0;
                while (t + 1 < tn) : (t += 2) {
                    quantaPair(&a, tb + t == i, me, t, eps2_q);
                    quantaPair(&a, tb + t + 1 == i, me, t + 1, eps2_q);
                }
                if (t < tn) quantaPair(&a, tb + t == i, me, t, eps2_q);
            }
            idx = nd.skip;
        } else idx = nd.child;
    }
    if (active) finish(i, a, vr);
}

/// Distance in quanta from v to the interval [l, h] (open domain: no wrap).
inline fn gap(l: u32, h: u32, v: u32) u32 {
    return if (v < l) l - v else if (v > h) v - h else 0;
}

/// Softened pair with tile body t, from exact integer offsets d in quanta: with the tile mass
/// prescaled by 1/bs^2 (and eps^2 by the same), m_q d / |d|^3 is the physical acceleration and
/// m_q / |d| is the potential over bs. Offsets from the group box as f32 would save the
/// conversions but round coincident bodies (escapers stuck at a wall) apart.
inline fn quantaPair(a: *Acc, self: bool, me: L.Pos3, t: u32, eps2_q: f32) void {
    if (!self) a.pairs += 1;
    const dx = C.delta(sh_px[t], me.x);
    const dy = C.delta(sh_py[t], me.y);
    const dz = C.delta(sh_pz[t], me.z);
    const m = if (self) 0 else sh_m[t];
    // 1 keeps the self term finite when eps is 0.
    const s = if (self) 1 else dx * dx + dy * dy + dz * dz + eps2_q;
    const inv = 1.0 / @sqrt(s);
    const w = m * inv * inv * inv;
    a.x += w * dx;
    a.y += w * dy;
    a.z += w * dz;
    a.phi_q -= m * inv;
}

// Per-body walk, one entry point per workgroup size; the host picks one (walk3.zig `group_sizes`).
export fn walk64() callconv(cc(64)) void {
    walkBody(64, .{});
}
export fn walk128() callconv(cc(128)) void {
    walkBody(128, .{});
}
export fn walk256() callconv(cc(256)) void {
    walkBody(256, .{});
}
export fn walk512() callconv(cc(512)) void {
    walkBody(512, .{});
}
export fn walk1024() callconv(cc(1024)) void {
    walkBody(1024, .{});
}
/// The group walk; walk3.zig group_bodies.
export fn walkGroup64() callconv(cc(64)) void {
    walkNodes(64, .{});
}

// Profiling entry points (walk3_tests "profile"); separate entry points, so the ones above pay
// nothing for them.
export fn walkStats() callconv(cc(1024)) void {
    walkBody(1024, .{ .stats = true });
}
export fn walkNoLeafMem() callconv(cc(1024)) void {
    walkBody(1024, .{ .no_leaf_mem = true });
}
export fn walkNoLeaf() callconv(cc(1024)) void {
    walkBody(1024, .{ .no_leaf = true });
}
export fn walkGroupStats64() callconv(cc(64)) void {
    walkNodes(64, .{ .stats = true });
}
export fn walkGroupNoLeafMem64() callconv(cc(64)) void {
    walkNodes(64, .{ .no_leaf_mem = true });
}
export fn walkGroupNoLeaf64() callconv(cc(64)) void {
    walkNodes(64, .{ .no_leaf = true });
}
