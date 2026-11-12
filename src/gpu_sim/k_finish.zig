//! Gather solved positions/velocities back into ball-id order and apply the play-area walls
//! (physics.zig applyBounds, including its ceiling and floor-energy deviations from Python).
//! Runs in ball-id order (reading each ball's sorted slot, written by k_scatter) so the
//! id-order buffers are read and written linearly and only the sorted reads are random.
//! Contact tests are inclusive because k_collide already clamps positions onto the walls.
const c = @import("common.zig");
const L = c.L;

// Solved positions as floats: x, y at k * pos_stride (PosRad records or plain Vec2s).
const s_pos = c.ro(f32, 0, "s_pos");
const s_vel = c.ro(L.Vec2, 1, "s_vel");
const sorted_slot = c.ro(u32, 2, "sorted_slot");
const body = c.ro(L.BallBody, 3, "body");
// Spin the colored solvers accumulated per sorted slot (applied here so the solver never
// touches the ball-id-order state buffer, whose accesses from sorted order are random).
const s_spin = c.ro(f32, 4, "s_spin");
const meta = c.ro(L.BallMeta, 5, "meta");
const pos = c.rw(L.Vec2, 0, "pos");
const vel = c.rw(L.Vec2, 1, "vel");
const state = c.rw(L.BallState, 2, "state");

const SP = c.uniform(c.SortParams, 1, "sp");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const i = c.gid();
    if (i >= P.ball_count or meta.v[i] & L.flag_alive == 0) return;
    const k = sorted_slot.v[i];
    const b = body.v[i];
    const o = k * SP.pos_stride;
    var p = L.Vec2{ .x = s_pos.v[o], .y = s_pos.v[o + 1] };
    var v = s_vel.v[k];
    var s = state.v[i];
    s.ang_vel += s_spin.v[k];
    const r = b.radius;
    const half_wall = P.wall / 2;
    const ceiling_y = P.bounds_min_y + r + half_wall;
    const floor_y = P.bounds_max_y - r - half_wall;
    const left_x = P.bounds_min_x + r + half_wall;
    const right_x = P.bounds_max_x - r - half_wall;

    if (p.y <= ceiling_y) {
        p.y = ceiling_y;
        s.ang_vel -= v.x / r;
        const rebound = -v.y * b.retention;
        v.y = if (rebound > c.bounce_stop) rebound else 0;
    }
    if (p.y >= floor_y) {
        p.y = floor_y;
        s.ang_vel += v.x / r;
        const rebound = v.y * b.retention - s.gravity;
        v.y = if (rebound > c.bounce_stop) -rebound else 0;
    }
    if (p.x <= left_x) {
        p.x = left_x;
        s.ang_vel -= v.x / r;
        if (v.x < 0) {
            v.x *= -b.retention;
            if (@abs(v.x) < c.bounce_stop) v.x = 0;
        }
    } else if (p.x >= right_x) {
        p.x = right_x;
        s.ang_vel += v.x / r;
        if (v.x > 0) {
            v.x *= -b.retention;
            if (@abs(v.x) < c.bounce_stop) v.x = 0;
        }
    }
    pos.v[i] = p;
    vel.v[i] = v;
    state.v[i].ang_vel = s.ang_vel;
}
