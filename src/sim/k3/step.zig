//! Drift, diagnostics partials and the acceleration-rung histogram of the 3D sim.
const C = @import("common.zig");
const L = C.L;

const params = C.uniform(L.Params3, "params");

/// Open domain: the middle half of the u32 range, so wrapped differences are exact (see C.delta).
const lo: u32 = 1 << 30;
const hi: u32 = 3 * (1 << 30) - 1;
const max_step: f32 = 536870912.0;

/// Moves a position by d quanta, rounded to nearest. The resolution is box_size / 2^32 and no
/// sub-quantum remainder is kept: a body slower than half a quantum per tick stands still.
inline fn move(p: u32, d: f32, periodic: bool) u32 {
    const neg = d < 0;
    const a = if (neg) -d else d;
    // NaN fails both comparisons and moves by zero.
    const m: u32 = @intFromFloat(@floor((if (a < max_step) a else if (a >= max_step) max_step else 0) + 0.5));
    if (periodic) return if (neg) p -% m else p +% m;
    // Escapers stick to the domain wall; diag counts them.
    if (neg) return if (m > p - lo) lo else p - m;
    return if (m > hi - p) hi else p + m;
}

const d_pos = C.rw(L.Pos3, 0, "pos");
const d_vel = C.rw(L.Vel3, 1, "vel");

export fn drift() callconv(C.cc) void {
    const i = C.gid();
    if (i >= params.n) return;
    const periodic = params.flags & L.sim3_flag_periodic != 0;
    const p = C.load(d_pos, i);
    const v = C.load(d_vel, i);
    const q = params.drift_q;
    C.store(d_pos, i, .{ .x = move(p.x, v.x * q, periodic), .y = move(p.y, v.y * q, periodic), .z = move(p.z, v.z * q, periodic) });
}

pub const diag_per_thread: u32 = 16;
/// Floats per workgroup in the diagnostics partials; see `diag_fields`.
pub const diag_stride: u32 = 16;
/// ke, px, py, pz, lx, ly, lz, mass, pe, escapers (bodies on the domain wall).
const diag_fields = 10;

const g_pos = C.ro(L.Pos3, 0, "g_pos");
const g_vel = C.ro(L.Vel3, 1, "g_vel");
const g_mass = C.ro(f32, 2, "g_mass");
const g_pot = C.ro(f32, 3, "g_pot");
const partials = C.rw(f32, 0, "partials");
const sh = C.Shared(f32, diag_fields * C.group_size, "sh");

fn groups(n: u32) u32 {
    const threads = (n + diag_per_thread - 1) / diag_per_thread;
    return (threads + C.group_size - 1) / C.group_size;
}

/// Per workgroup sums. Strided so a warp reads consecutive bodies. L is about the box centre.
export fn diag() callconv(C.cc) void {
    // 2D dispatches round the group count up to a multiple of dispatch_x; the extra groups would
    // double-count and write past the partials. The return is uniform across the group, so it
    // precedes every barrier safely.
    if (C.groupId() >= groups(params.n)) return;
    const l = C.lid();
    const stride = groups(params.n) * C.group_size;
    var acc: [diag_fields]f32 = @splat(0);
    var i = C.gid();
    var k: u32 = 0;
    const centre: u32 = 1 << 31;
    while (k < diag_per_thread) : (k += 1) {
        if (i < params.n) {
            const m = C.load(g_mass, i);
            const v = C.load(g_vel, i);
            const p = C.load(g_pos, i);
            const x = C.delta(p.x, centre) * params.box_scale;
            const y = C.delta(p.y, centre) * params.box_scale;
            const z = C.delta(p.z, centre) * params.box_scale;
            acc[0] += 0.5 * m * (v.x * v.x + v.y * v.y + v.z * v.z);
            acc[1] += m * v.x;
            acc[2] += m * v.y;
            acc[3] += m * v.z;
            acc[4] += m * (y * v.z - z * v.y);
            acc[5] += m * (z * v.x - x * v.z);
            acc[6] += m * (x * v.y - y * v.x);
            acc[7] += m;
            acc[8] += 0.5 * m * C.load(g_pot, i);
            const wall = p.x == lo or p.x == hi or p.y == lo or p.y == hi or p.z == lo or p.z == hi;
            if (wall and params.flags & L.sim3_flag_periodic == 0) acc[9] += 1;
        }
        i += stride;
    }
    inline for (0..diag_fields) |f| sh[f * C.group_size + l] = acc[f];
    var off: u32 = C.group_size / 2;
    while (off > 0) : (off >>= 1) {
        C.barrier();
        if (l < off) {
            inline for (0..diag_fields) |f| sh[f * C.group_size + l] += sh[f * C.group_size + l + off];
        }
    }
    C.barrier();
    if (l == 0) {
        const o = C.groupId() * diag_stride;
        inline for (0..diag_fields) |f| C.store(partials, o + @as(u32, f), sh[f * C.group_size + l]);
    }
}

pub const rung_bins: u32 = 16;
const r_amag = C.ro(f32, 0, "r_amag");
const bins = C.rw(u32, 0, "bins");

export fn rungsClear() callconv(C.cc) void {
    const i = C.gid();
    if (i < rung_bins) C.store(bins, i, 0);
}

/// Rung r: the body wants dt/2^r or smaller, from the step sqrt(2 eta eps / |a|); r is capped
/// at rung_bins - 1.
export fn rungs() callconv(C.cc) void {
    const i = C.gid();
    if (i >= params.n) return;
    const a = C.load(r_amag, i);
    const ratio = params.dt * @sqrt(a / (2 * params.eta * params.eps));
    var r: u32 = 0;
    if (ratio > 1) {
        // ceil(log2 ratio) = bit length of ceil(ratio) - 1.
        const c: u32 = @intFromFloat(@ceil(@min(ratio, 1048576.0)));
        r = 32 - C.clz(c - 1);
    }
    _ = C.atomicAddAt(bins, @min(r, rung_bins - 1), 1);
}
