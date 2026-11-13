//! Morton keys, stable LSD radix sort of SortRec and the body permutation (docs/sim3d.md, WP3).
//!
//! One radix pass is three dispatches: `histogram` (per-block digit counts, digit-major so one
//! scan per digit row gives every block its output base), `scanRows` (one group per digit) and
//! `scatter`. Scatter ranks equal digits inside a block with eight 1-bit split passes in shared
//! memory, so the order of equal digits is the input order and no atomic decides a rank: reruns
//! are bit-identical. The only atomics are the shared-memory counts in `histogram`.
const C = @import("common.zig");
const L = C.L;

/// Host mirror in sort3.zig; kernel modules cannot import outside k3/.
pub const SortParams = extern struct {
    n: u32,
    /// Radix blocks of `block` records.
    blocks: u32,
    /// Bit offset of the digit in the 64-bit key; a multiple of 8, so a digit never straddles words.
    shift: u32,
    pad: u32 = 0,
};

pub const radix_bits = 8;
pub const radix: u32 = 1 << radix_bits;
pub const per_thread: u32 = 8;
pub const block: u32 = C.group_size * per_thread;

comptime {
    // blockScan and the per-digit loops use one thread per digit.
    if (radix != C.group_size) @compileError("radix must equal group_size");
}

const params = C.uniform(SortParams, "params");

/// Workgroup-scope OpAtomicIAdd on shared memory; the result is unused, so only counts depend on it.
inline fn sharedInc(p: *addrspace(.shared) u32) void {
    _ = asm volatile (
        \\%ret = OpAtomicIAdd %u32 %p %scope %sem %v
        : [ret] "" (-> u32),
        : [u32] "t" (u32),
          [p] "" (p),
          [scope] "" (@as(u32, @backingInt(@import("std").spirv.Scope.workgroup))),
          [sem] "" (@as(u32, 0)),
          [v] "" (@as(u32, 1)),
    );
}

const sc = C.Shared(u32, C.group_size, "sc");

const Scan = struct { excl: u32, total: u32 };

/// Exclusive scan of one value per thread across the group (Hillis-Steele). Every thread must
/// call it; it starts and ends with a barrier, so it also fences shared memory around it.
inline fn blockScan(l: u32, v: u32) Scan {
    C.barrier();
    sc[l] = v;
    C.barrier();
    var off: u32 = 1;
    while (off < C.group_size) : (off <<= 1) {
        // Masked index instead of a guarded load: l - off must not underflow even if hoisted.
        const t = sc[(l -% off) & (C.group_size - 1)];
        C.barrier();
        if (l >= off) sc[l] += t;
        C.barrier();
    }
    const incl = sc[l];
    const total = sc[l | (C.group_size - 1)];
    C.barrier();
    return .{ .excl = incl - v, .total = total };
}

inline fn digitOf(hi: u32, lo: u32) u32 {
    const s = params.shift;
    const w = if (s >= 32) hi else lo;
    return (w >> @as(u5, @truncate(s))) & (radix - 1);
}

inline fn groupsFor(threads: u32) u32 {
    return (threads + C.group_size - 1) / C.group_size;
}

/// Spreads the low 10 bits of v to every third bit.
inline fn spread10(v: u32) u32 {
    var x = v & 0x3ff;
    x = (x | x << 16) & 0x030000FF;
    x = (x | x << 8) & 0x0300F00F;
    x = (x | x << 4) & 0x030C30C3;
    x = (x | x << 2) & 0x09249249;
    return x;
}

const k_pos = C.ro(L.Pos3, 0, "k_pos");
const k_out = C.rw(L.SortRec, 0, "k_out");

/// ref3.mortonKey without 64-bit integers: key bits 0..29 are coordinate bits 0..9 (of the top
/// 21), bits 30..32 are bit 10 of z, y, x, and bits 33..62 are coordinate bits 11..20.
export fn keys() callconv(C.cc) void {
    if (C.groupId() >= groupsFor(params.n)) return;
    const i = C.gid();
    if (i >= params.n) return;
    const p = C.load(k_pos, i);
    const x = p.x >> 11;
    const y = p.y >> 11;
    const z = p.z >> 11;
    const lo = (spread10(x) << 2) | (spread10(y) << 1) | spread10(z) |
        (((z >> 10) & 1) << 30) | (((y >> 10) & 1) << 31);
    const hi = ((x >> 10) & 1) | (((spread10(x >> 11) << 2) | (spread10(y >> 11) << 1) | spread10(z >> 11)) << 1);
    C.store(k_out, i, .{ .key_hi = hi, .key_lo = lo, .idx = i });
}

const h_src = C.ro(L.SortRec, 0, "h_src");
const h_hist = C.rw(u32, 0, "h_hist");
const h_cnt = C.Shared(u32, radix, "h_cnt");

/// hist[digit * blocks + block] = count of `digit` in the block.
export fn histogram() callconv(C.cc) void {
    const b = C.groupId();
    if (b >= params.blocks) return;
    const l = C.lid();
    h_cnt[l] = 0;
    C.barrier();
    const base = b * block;
    inline for (0..per_thread) |k| {
        const i = base + @as(u32, k) * C.group_size + l;
        if (i < params.n) sharedInc(&h_cnt[digitOf(h_src.v[i].key_hi, h_src.v[i].key_lo)]);
    }
    C.barrier();
    C.store(h_hist, l * params.blocks + b, h_cnt[l]);
}

const s_hist = C.rw(u32, 0, "s_hist");
const scan_per_thread: u32 = 4;

/// One group per digit: exclusive scan of its row in place; the row total goes to
/// hist[radix * blocks + digit].
export fn scanRows() callconv(C.cc) void {
    const d = C.groupId();
    if (d >= radix) return;
    const l = C.lid();
    const nb = params.blocks;
    const row = d * nb;
    var carry: u32 = 0;
    var start: u32 = 0;
    while (start < nb) : (start += C.group_size * scan_per_thread) {
        const first = start + l * scan_per_thread;
        var v: [scan_per_thread]u32 = undefined;
        var sum: u32 = 0;
        inline for (0..scan_per_thread) |k| {
            const j = first + @as(u32, k);
            v[k] = if (j < nb) C.load(s_hist, row + j) else 0;
            sum += v[k];
        }
        const r = blockScan(l, sum);
        var run = carry + r.excl;
        inline for (0..scan_per_thread) |k| {
            const j = first + @as(u32, k);
            if (j < nb) C.store(s_hist, row + j, run);
            run += v[k];
        }
        carry += r.total;
    }
    if (l == 0) C.store(s_hist, radix * nb + d, carry);
}

const x_src = C.ro(L.SortRec, 0, "x_src");
const x_hist = C.ro(u32, 1, "x_hist");
const x_dst = C.rw(L.SortRec, 0, "x_dst");
/// (digit << 16) | position in the block.
const x_sv = C.Shared(u32, block, "x_sv");
const x_base = C.Shared(u32, radix, "x_base");
const x_start = C.Shared(u32, radix, "x_start");

export fn scatter() callconv(C.cc) void {
    const b = C.groupId();
    if (b >= params.blocks) return;
    const l = C.lid();
    const nb = params.blocks;
    const total = C.load(x_hist, radix * nb + l);
    const digit_base = blockScan(l, total).excl;
    x_base[l] = digit_base + C.load(x_hist, l * nb + b);

    const base = b * block;
    const count = @min(block, params.n - base);
    // Padding past n takes the largest digit and the largest positions, so it sorts after every
    // real record and never moves a real one.
    inline for (0..per_thread) |k| {
        const j = @as(u32, k) * C.group_size + l;
        const d = if (j < count) digitOf(x_src.v[base + j].key_hi, x_src.v[base + j].key_lo) else radix - 1;
        x_sv[j] = (d << 16) | j;
    }
    C.barrier();
    // Each thread owns `per_thread` consecutive slots, so scan order is block order: stable.
    var v: [per_thread]u32 = undefined;
    inline for (0..per_thread) |k| v[k] = x_sv[l * per_thread + @as(u32, k)];
    var bit: u32 = 16;
    while (bit < 16 + radix_bits) : (bit += 1) {
        var zeros: u32 = 0;
        inline for (0..per_thread) |k| zeros += ((v[k] >> @as(u5, @truncate(bit))) & 1) ^ 1;
        // The scan's barriers also order this pass's writes after the previous pass's reads.
        const r = blockScan(l, zeros);
        var zb: u32 = 0;
        inline for (0..per_thread) |k| {
            const j = l * per_thread + @as(u32, k);
            const z = ((v[k] >> @as(u5, @truncate(bit))) & 1) ^ 1;
            const dst = if (z == 1) r.excl + zb else r.total + (j - r.excl - zb);
            zb += z;
            x_sv[dst] = v[k];
        }
        C.barrier();
        inline for (0..per_thread) |k| v[k] = x_sv[l * per_thread + @as(u32, k)];
    }
    inline for (0..per_thread) |k| {
        const j = @as(u32, k) * C.group_size + l;
        const d = x_sv[j] >> 16;
        const prev = x_sv[(j -% 1) & (block - 1)] >> 16;
        if (j == 0 or d != prev) x_start[d] = j;
    }
    C.barrier();
    inline for (0..per_thread) |k| {
        const j = @as(u32, k) * C.group_size + l;
        const e = x_sv[j];
        const at = e & 0xffff;
        if (at < count) {
            const d = e >> 16;
            C.store(x_dst, x_base[d] + (j - x_start[d]), C.load(x_src, base + at));
        }
    }
}

const g_rec = C.ro(L.SortRec, 0, "g_rec");
const g_src12 = C.ro(L.Pos3, 1, "g_src12");
const g_dst12 = C.rw(L.Pos3, 0, "g_dst12");
const g_src4 = C.ro(u32, 1, "g_src4");
const g_dst4 = C.rw(u32, 0, "g_dst4");

/// dst[i] = src[rec[i].idx] for 12-byte bodies (Pos3, Vel3).
export fn gather12() callconv(C.cc) void {
    if (C.groupId() >= groupsFor(params.n)) return;
    const i = C.gid();
    if (i >= params.n) return;
    C.store(g_dst12, i, C.load(g_src12, g_rec.v[i].idx));
}

/// dst[i] = src[rec[i].idx] for 4-byte bodies (mass).
export fn gather4() callconv(C.cc) void {
    if (C.groupId() >= groupsFor(params.n)) return;
    const i = C.gid();
    if (i >= params.n) return;
    C.store(g_dst4, i, C.load(g_src4, g_rec.v[i].idx));
}
