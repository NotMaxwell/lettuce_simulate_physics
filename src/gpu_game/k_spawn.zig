//! Writes uploaded new balls into their slots (free slots or the tail).
//! Bindings: ro 0 recs; rw 0 pos, 1 prev_pos, 2 vel, 3 body, 4 state, 5 heat, 6 meta.
const c = @import("common.zig");
const L = c.L;

const recs = c.ro(c.SpawnRec, 0, "recs");
const pos = c.rw(L.Vec2, 0, "pos");
const prev_pos = c.rw(L.Vec2, 1, "prev_pos");
const vel = c.rw(L.Vec2, 2, "vel");
const body = c.rw(L.BallBody, 3, "body");
const state = c.rw(L.BallState, 4, "state");
const heat = c.rw(L.BallHeat, 5, "heat");
const meta = c.rw(L.BallMeta, 6, "meta");

export fn spawn() callconv(c.kernel_cc) void {
    const k = c.gid();
    if (k >= c.params.items) return;
    const r = recs.v[k];
    const i = r.id;
    pos.v[i] = .{ .x = r.x, .y = r.y };
    prev_pos.v[i] = .{ .x = r.x, .y = r.y };
    vel.v[i] = .{ .x = r.vx, .y = r.vy };
    body.v[i] = .{ .radius = r.radius, .mass = r.mass, .retention = r.retention, .friction = r.friction };
    state.v[i] = .{ .angle = 0, .prev_angle = 0, .ang_vel = 0, .gravity = r.gravity };
    heat.v[i] = .{ .temp = r.temp, .burn = 0 };
    meta.v[i] = r.meta;
}
