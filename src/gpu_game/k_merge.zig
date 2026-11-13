//! Merge apply, hoop scoring, the live max-level reduction and the per-tick reset of the game event list.
//! Bindings: ro 0 candidates (gpu_sim event buffer), 1 prev_pos;
//!           rw 0 pos, 1 vel, 2 body, 3 meta, 4 claim, 5 out.
//! `out` uses the gpu_sim event layout: element 0 is the header (kind = count, a = dropped).
const c = @import("common.zig");
const L = c.L;

const cand = c.ro(L.Event, 0, "cand");
const prev_pos = c.ro(L.Vec2, 1, "prev_pos");
const pos = c.rw(L.Vec2, 0, "pos");
const vel = c.rw(L.Vec2, 1, "vel");
const body = c.rw(L.BallBody, 2, "body");
const meta = c.rw(L.BallMeta, 3, "meta");
const claim = c.rw(u32, 4, "claim");
const out = c.rw(L.Event, 5, "out");

export fn reset() callconv(c.kernel_cc) void {
    if (c.gid() != 0) return;
    out.v[0].kind = 0;
    out.v[0].a = 0;
    out.v[0].b = 0;
}

/// Highest level among live balls, as level + 1 into the header's `b` (0 = not computed this
/// tick). Runs after the merges, so the CPU's HUD level follows balls that merged away or died.
export fn maxLevel() callconv(c.kernel_cc) void {
    const i = c.gid();
    var lvl: u32 = 0;
    if (i < c.params.ball_count) {
        const m = meta.v[i];
        if (m & L.flag_alive != 0) lvl = (m & 0xff) + 1;
    }
    const top = c.blockMax(lvl);
    // Thread 0 always reports, so an empty world reads as level 0 rather than "not computed".
    const report = if (i == 0) @max(top, 1) else top;
    if (c.lid() == 0 and report != 0) _ = c.atomicUMax(&out.v[0].b, report);
}

/// Reserves an output record; null when full (the caller then skips its state change, so the
/// CPU's mirror of slots and radii never misses a change).
inline fn reserve() ?u32 {
    const slot = c.atomicAdd(&out.v[0].kind, 1);
    if (slot < c.params.out_cap) return slot + 1;
    _ = c.atomicAdd(&out.v[0].a, 1);
    return null;
}

/// One thread per merge candidate. Each ball merges at most once per tick: both balls are
/// claimed with an atomic exchange of the tick stamp; losing either claim skips the pair (a ball
/// claimed by a pair that then lost its partner simply waits a tick).
export fn merge() callconv(c.kernel_cc) void {
    const k = c.gid();
    const cap = if (c.params.event_cap != 0) @min(c.params.event_cap, L.max_events - 1) else L.max_events - 1;
    const count = @min(cand.v[0].kind, cap);
    if (k >= count) return;
    const e = cand.v[k + 1];
    if (e.kind != L.event_merge) return;
    const a = e.a;
    const b = e.b;
    const ma = meta.v[a];
    const mb = meta.v[b];
    if (ma & L.flag_alive == 0 or mb & L.flag_alive == 0) return;
    const accrete = c.params.collide == 1;
    if (!accrete and (ma & 0xff != mb & 0xff or ma & 0xff >= c.max_level)) return;
    const stamp = c.params.stamp;
    if (c.atomicExchange(&claim.v[a], stamp) == stamp) return;
    if (c.atomicExchange(&claim.v[b], stamp) == stamp) return;
    const slot = reserve() orelse return;

    const ra = body.v[a].radius;
    const rb = body.v[b].radius;
    const ba = body.v[a].mass;
    const bb = body.v[b].mass;
    // physics.zig resolvePair: the larger ball survives (j on ties); accretion keeps the heavier.
    const a_wins = if (accrete) ba > bb else ra > rb;
    const keep = if (a_wins) a else b;
    const gone = if (a_wins) b else a;
    const mk = body.v[keep].mass;
    const mg = body.v[gone].mass;
    const pk = pos.v[keep];
    const pg = pos.v[gone];
    const vk = vel.v[keep];
    const vg = vel.v[gone];
    // Deviation (requested): the CPU/Python keep the survivor's position and velocity; here it
    // moves to the mass-weighted centre with the pair's mass-weighted velocity. Mass stays
    // the survivor's in the merge game (as in the Python) and is summed in accrete mode below.
    const inv = 1.0 / @max(mk + mg, 1e-6);
    pos.v[keep] = .{ .x = (pk.x * mk + pg.x * mg) * inv, .y = (pk.y * mk + pg.y * mg) * inv };
    vel.v[keep] = .{ .x = (vk.x * mk + vg.x * mg) * inv, .y = (vk.y * mk + vg.y * mg) * inv };
    const r_keep = body.v[keep].radius;
    var r_new = @min(c.params.max_radius, r_keep + @log(1.0 + body.v[gone].radius));
    var lvl: u32 = @min((ma & 0xff) + 1, c.max_level);
    if (accrete) {
        // Area adds up (mass-proportional area), and the level is galaxy.zig levelFor(mass):
        // kernels cannot import galaxy.zig, so the formula is duplicated here.
        const m = mk + mg;
        r_new = @min(c.params.max_radius, @sqrt(r_keep * r_keep + body.v[gone].radius * body.v[gone].radius));
        const l = 1.0 + @log2(@max(m, 1.0));
        lvl = @intFromFloat(@min(@max(l, 1.0), @as(f32, @floatFromInt(c.max_level))));
        body.v[keep].mass = m;
    }
    body.v[keep].radius = r_new;
    meta.v[keep] = (meta.v[keep] & ~@as(u32, 0xff)) | lvl;
    // Dead balls keep their slot until the CPU reuses it; mass 0 removes them from gravity.
    meta.v[gone] = meta.v[gone] & ~(L.flag_alive | L.flag_grabbed);
    body.v[gone].mass = 0;
    out.v[slot] = .{ .kind = L.event_merge | (lvl << 8), .a = keep, .b = gone, .value = r_new };
}

/// physics.zig integrate hoop test, after the solver: a thrown, unscored ball whose centre
/// crossed the hoop line downwards between the hoop's ends scores.
export fn hoop() callconv(c.kernel_cc) void {
    const i = c.gid();
    const P = c.params;
    if (i >= P.ball_count) return;
    const m = meta.v[i];
    if (m & (L.flag_alive | L.flag_thrown | L.flag_scored) != (L.flag_alive | L.flag_thrown)) return;
    const o = prev_pos.v[i];
    const p = pos.v[i];
    if (!(o.y < P.hoop_y and P.hoop_y <= p.y)) return;
    const frac = (P.hoop_y - o.y) / (p.y - o.y);
    const x = o.x + frac * (p.x - o.x);
    if (x < P.hoop_left or x > P.hoop_right) return;
    meta.v[i] = m | L.flag_scored;
    const slot = reserve() orelse return;
    out.v[slot] = .{ .kind = L.event_score, .a = i, .b = 0, .value = 0 };
}
