//! Per ball: save previous pose, mouse attraction, velocity -> position, grab, per-ball
//! gravity and floor friction (physics.zig attract + integrate). Fire/burn/hoop logic stays
//! on the CPU side for now. Thread 0 also resets the event append counter for this tick.
//! Also bins the ball into its grid cell (broad phase step 1), fused here to save a pass over
//! the positions: the atomic's return value is the ball's slot within the cell, so the
//! scatter pass needs no second atomic. Dead balls go to the extra cell `ncells`, which
//! neighbour queries never visit.
const c = @import("common.zig");
const L = c.L;

const body = c.ro(L.BallBody, 0, "body");
const meta = c.ro(L.BallMeta, 1, "meta");
const pos = c.rw(L.Vec2, 0, "pos");
const prev_pos = c.rw(L.Vec2, 1, "prev_pos");
const vel = c.rw(L.Vec2, 2, "vel");
const state = c.rw(L.BallState, 3, "state");
const events = c.rw(L.Event, 4, "events");
const counts = c.rw(u32, 5, "counts");
const ball_cell = c.rw(u32, 6, "ball_cell");
const ball_slot = c.rw(u32, 7, "ball_slot");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const i = c.gid();
    if (i == 0) {
        events.v[0].kind = 0;
        events.v[0].a = 0;
    }
    if (i >= P.ball_count) return;
    const m = meta.v[i];
    var cell = P.grid_cells_x * P.grid_cells_y;
    if (m & L.flag_alive != 0) cell = c.cellOf(integrate(i, m), P);
    ball_cell.v[i] = cell;
    ball_slot.v[i] = c.atomicAdd(&counts.v[cell], 1);
}

inline fn integrate(i: u32, m: u32) L.Vec2 {
    const P = c.params;
    var p = pos.v[i];
    var v = vel.v[i];
    var s = state.v[i];
    prev_pos.v[i] = p;
    s.prev_angle = s.angle;

    if (P.flags & c.flag_attract != 0) {
        v.x = (P.mouse_x - p.x) * P.attraction_strength;
        v.y = (P.mouse_y - p.y) * P.attraction_strength;
        s.ang_vel = 0;
    }

    p.x += v.x;
    p.y += v.y;
    s.angle += s.ang_vel;
    s.ang_vel *= 0.98;

    if (m & L.flag_grabbed != 0) {
        const dx = P.mouse_x - p.x;
        const dy = P.mouse_y - p.y;
        v.x += dx;
        v.y += dy;
        p = .{ .x = P.mouse_x, .y = P.mouse_y };
        // The last follow step, for gpu_game k_act release (Ball.releaseGrab adds it to the throw).
        // Only one ball is held at a time; the header's b/value are free (the solver only counts).
        events.v[0].b = @bitCast(dx);
        events.v[0].value = dy;
    }

    if (P.flags & c.flag_mutual_gravity == 0) {
        const r = body.v[i].radius;
        const floor_y = P.bounds_max_y - r - P.wall / 2;
        if (p.y < floor_y) v.y += s.gravity;
        if (v.y == 0 and v.x != 0) {
            const fr = body.v[i].friction;
            v.x += if (v.x > 0) -fr else fr;
            if (@abs(v.x) < fr) v.x = 0;
        }
    }

    pos.v[i] = p;
    vel.v[i] = v;
    state.v[i] = s;
    return p;
}
