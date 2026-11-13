//! Narrow phase as parallel Gauss-Seidel over a 2x2 coloring of the grid. One dispatch handles every cell of one
//! color; each thread owns cell (x, y) and resolves, in order, every pair of a ball of its cell
//! with the same cell or the "forward" neighbours (+1,0), (0,+1), (+1,+1), plus the pairs
//! between cells (+1,0) and (0,+1) (the -1,+1 offset), updating both balls in place exactly
//! like physics.zig resolvePair. A thread touches only the 2x2 block x..x+1, y..y+1, so
//! same-colored threads (2 apart both ways) never share a cell, the four dispatches together
//! visit every neighbour pair once, and there are 4 dispatches instead of the 6 a 3-wide
//! (-1..+1, +1) stencil needs.
//!
//! Why not Jacobi: averaging simultaneous corrections cannot hold up a stack. A 40-ball column
//! still settled with ~20 px overlaps (of 60) after 32 Jacobi iterations per tick, and 10k-ball
//! piles crushed; sequential resolution, like the CPU, carries the floor up the pile.
//!
//! Pair order within a thread (one of many valid sequential orders; the CPU's i<j order cannot
//! be kept on the GPU anyway): the cell's balls are taken in chunks of `chunk`, held in
//! registers; each chunk resolves its internal pairs, then streams every later candidate j
//! once and resolves j against the whole chunk. Loading each candidate once per chunk instead
//! of once per ball cut this kernel's L2 traffic, its bottleneck in dense piles, by ~chunk x.
const c = @import("common.zig");
const L = c.L;

const body = c.ro(L.BallBody, 0, "body");
const starts = c.ro(u32, 1, "starts");
const sorted_id = c.ro(u32, 2, "sorted_id");
const meta = c.ro(L.BallMeta, 3, "meta");
const s_prm = c.rw(c.PosRad, 0, "s_prm");
const s_vel = c.rw(L.Vec2, 1, "s_vel");
const s_spin = c.rw(f32, 2, "s_spin");
const events = c.rw(L.Event, 3, "events");

const CP = c.uniform(c.ColorParams, 1, "cp");

const chunk = 4;
const batch = 2;

/// A ball held in registers while it is resolved. `spin` is the change accumulated this
/// sweep; `id`/`level` are filled only when merging is on; r <= 0 marks an empty/dead slot.
const Ball = struct { p: L.Vec2, v: L.Vec2, r: f32, m: f32, spin: f32, slot: u32, id: u32, level: u32, hit: bool };

inline fn load(k: u32, valid: bool, merging: bool) Ball {
    const r = s_prm.v[k].r;
    var x = Ball{
        .p = .{ .x = s_prm.v[k].x, .y = s_prm.v[k].y },
        .v = s_vel.v[k],
        .r = if (valid) r else -1,
        .m = s_prm.v[k].m,
        .spin = 0,
        .slot = k,
        .id = 0,
        .level = 0,
        .hit = false,
    };
    if (merging and valid) {
        x.id = sorted_id.v[k];
        x.level = c.level(meta.v[x.id]);
    }
    return x;
}

inline fn store(k: u32, x: Ball) void {
    s_prm.v[k].x = x.p.x;
    s_prm.v[k].y = x.p.y;
    s_vel.v[k] = x.v;
    if (x.spin != 0) s_spin.v[k] += x.spin;
}

/// Only the non-default inelastic mode needs retention, so it is not in the streamed record.
inline fn retention(x: *const Ball) f32 {
    return body.v[sorted_id.v[x.slot]].retention;
}
export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const nx = P.grid_cells_x;
    const ny = P.grid_cells_y;
    const per_row = (nx - CP.color_x + 1) / 2;
    const t = c.gid();
    const cx = CP.color_x + 2 * (t % per_row);
    const cy = CP.color_y + 2 * (t / per_row);
    if (cx >= nx or cy >= ny) return;
    const cell = cy * nx + cx;
    const merging = P.flags & c.flag_merging != 0;
    const first = CP.iteration == 0;
    const right = cx + 1 < nx;

    // Own cell vs itself, (+1,0) (contiguous with the rest of the own cell) and the next row's
    // (0,+1), (+1,+1) (contiguous with each other).
    const own_begin = starts.v[cell];
    const own_end = starts.v[cell + 1];
    const row0_end = starts.v[cell + (if (right) @as(u32, 2) else 1)];
    var below_begin: u32 = 0;
    var below_mid: u32 = 0;
    var below_end: u32 = 0;
    if (cy + 1 < ny) {
        below_begin = starts.v[cell + nx];
        below_mid = starts.v[cell + nx + 1];
        below_end = if (right) starts.v[cell + nx + 2] else below_mid;
    }
    pairs(own_begin, own_end, true, row0_end, below_begin, below_end, merging, first);
    // Anti-diagonal: (+1,0) vs (0,+1), the one neighbour offset no cell's forward set covers.
    if (right) pairs(own_end, row0_end, false, 0, below_begin, below_mid, merging, first);
}

/// Every pair of a ball in [a_begin, a_end) with a later ball of the same range (if `intra`;
/// the stream from the chunk's end to `row0_end` covers the rest) and with [r1_begin, r1_end).
inline fn pairs(a_begin: u32, a_end: u32, comptime intra: bool, row0_end: u32, r1_begin: u32, r1_end: u32, merging: bool, first: bool) void {
    var a0 = a_begin;
    while (a0 < a_end) : (a0 += chunk) {
        var A: [chunk]Ball = undefined;
        inline for (0..chunk) |kk| {
            const k: u32 = kk;
            A[kk] = load(@min(a0 + k, a_end - 1), a0 + k < a_end, merging);
        }
        if (intra) {
            inline for (0..chunk) |k1| {
                inline for (k1 + 1..chunk) |k2| resolve(&A[k1], &A[k2], first);
            }
            stream(&A, @min(a0 + chunk, a_end), row0_end, merging, first);
        }
        stream(&A, r1_begin, r1_end, merging, first);
        inline for (0..chunk) |kk| {
            const k: u32 = kk;
            if (a0 + k < a_end) store(a0 + k, A[kk]);
        }
    }
}

/// Resolve every candidate slot in [begin, end) against the whole chunk, `batch` candidates per
/// load batch. Exact: each candidate is read and written only here during this chunk.
inline fn stream(A: *[chunk]Ball, begin: u32, end: u32, merging: bool, first: bool) void {
    var j = begin;
    while (j < end) : (j += batch) {
        var J: [batch]Ball = undefined;
        inline for (0..batch) |kk| {
            const k: u32 = kk;
            J[kk] = load(@min(j + k, end - 1), j + k < end, merging);
        }
        inline for (0..batch) |b| {
            inline for (0..chunk) |k| resolve(&A[k], &J[b], first);
        }
        inline for (0..batch) |kk| {
            const k: u32 = kk;
            if (J[kk].hit) store(j + k, J[kk]);
        }
    }
}

/// physics.zig resolvePair between two register-held balls (heat transfer is not ported yet).
inline fn resolve(A: *Ball, B: *Ball, first: bool) void {
    const dx = B.p.x - A.p.x;
    const dy = B.p.y - A.p.y;
    const d2 = dx * dx + dy * dy;
    const rsum = A.r + B.r;
    if (!(d2 > 0 and d2 < rsum * rsum) or A.r <= 0 or B.r <= 0) return;
    const dist = @sqrt(d2);

    // Accretion merges every touching pair whatever its level; otherwise only equal levels.
    const accrete = c.collideMode(c.params.flags) == c.collide_accrete;
    if (c.params.flags & c.flag_merging != 0 and (accrete or (A.level == B.level and A.level < c.max_level))) {
        // Merging pairs get no contact response (as on the CPU); the merge kernel applies the
        // merge from this event in the same tick.
        if (first) {
            const slot = c.atomicAdd(&events.v[0].kind, 1);
            const cap = if (c.params.event_cap != 0) @min(c.params.event_cap, L.max_events - 1) else L.max_events - 1;
            if (slot < cap) {
                events.v[slot + 1] = .{ .kind = L.event_merge, .a = A.id, .b = B.id, .value = dist };
            } else {
                _ = c.atomicAdd(&events.v[0].a, 1);
            }
        }
        return;
    }

    const nx = dx / dist;
    const ny = dy / dist;
    const half = (rsum - dist) / 2;
    // Clamped onto the walls straight away (the CPU only applies walls after all pairs):
    // otherwise a ball pressed against the floor sinks into it by half of every push from
    // above, and deep piles crush from the bottom. k_finish still reflects the velocities.
    A.p = c.clampToWalls(.{ .x = A.p.x - nx * half, .y = A.p.y - ny * half }, A.r);
    B.p = c.clampToWalls(.{ .x = B.p.x + nx * half, .y = B.p.y + ny * half }, B.r);
    B.hit = true;

    const rvx = A.v.x - B.v.x;
    const rvy = A.v.y - B.v.y;
    const rel = rvx * nx + rvy * ny;
    if (first) {
        const tan = -rvx * ny + rvy * nx;
        A.spin += tan / A.r * 0.5;
        B.spin -= tan / B.r * 0.5;
    }
    if (rel > 0) {
        // Optional deviation (Sim.elastic = false): restitution is the pair's mean retention,
        // the bounce loss walls already apply, and approaches slower than rest_speed are fully
        // inelastic. The default matches the CPU's perfectly elastic 2 * rel_speed.
        const e: f32 = if (CP.elastic != 0) 1 else if (rel > CP.rest_speed) (retention(A) + retention(B)) * 0.5 else 0;
        const imp = (1 + e) * rel / (A.m + B.m);
        A.v.x -= imp * B.m * nx;
        A.v.y -= imp * B.m * ny;
        B.v.x += imp * A.m * nx;
        B.v.y += imp * A.m * ny;
    }
}
