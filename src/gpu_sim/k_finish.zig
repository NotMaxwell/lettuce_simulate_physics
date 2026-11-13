//! Gather solved positions/velocities back into ball-id order and apply the play-area walls
//! (physics.zig applyBounds, including its ceiling and floor-energy deviations from Python).
//! Runs in ball-id order (reading each ball's sorted slot, written by k_scatter) so the
//! id-order buffers are read and written linearly and only the sorted reads are random.
//! Contact tests are inclusive because the colored solver already clamps positions onto the walls.
const c = @import("common.zig");
const L = c.L;

const s_prm = c.ro(c.PosRad, 0, "s_prm");
const s_vel = c.ro(L.Vec2, 1, "s_vel");
const sorted_slot = c.ro(u32, 2, "sorted_slot");
const body = c.ro(L.BallBody, 3, "body");
// Spin the solver accumulated per sorted slot (applied here so the solver never
// touches the ball-id-order state buffer, whose accesses from sorted order are random).
const s_spin = c.ro(f32, 4, "s_spin");
const meta = c.ro(L.BallMeta, 5, "meta");
const pos = c.rw(L.Vec2, 0, "pos");
const vel = c.rw(L.Vec2, 1, "vel");
const state = c.rw(L.BallState, 2, "state");
// Only touched when SimParams.flags has flag_impact_heat (Sim.heat_buf / fx_buf; else dummies).
const heat = c.rw(L.BallHeat, 3, "heat");
const fx = c.rw(u32, 4, "fx");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const i = c.gid();
    if (i >= P.ball_count or meta.v[i] & L.flag_alive == 0) return;
    const k = sorted_slot.v[i];
    const b = body.v[i];
    const q = s_prm.v[k];
    var p = L.Vec2{ .x = q.x, .y = q.y };
    var v = s_vel.v[k];
    var s = state.v[i];
    s.ang_vel += s_spin.v[k];
    if (P.flags & c.flag_impact_heat != 0) {
        // Impact heat from the solver's change of velocity (`vel` still holds the pre-solve value).
        // physics.zig resolvePair heats a ball by closing speed * 1.5 * 10 / mass, and update_fire
        // resets anything below ignition (100) to room temperature on the next tick, so only a gain
        // that lights the ball at once counts. A contact changes the velocity by 2 * closing * m_other
        // / (m_self + m_other), so closing * 15 / m_self >= |dv| * 7.5 / m_self: the estimate never
        // lights a ball the CPU rule would not (it misses some when the other ball is lighter).
        // Tracking closing speed per contact in the solver instead measured +25% in a dense pile.
        const o = vel.v[i];
        const dx = v.x - o.x;
        const dy = v.y - o.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > c.impact_dv_min * c.impact_dv_min) {
            const gain = @sqrt(d2) * c.impact_heat_k / b.mass;
            const t = heat.v[i].temp;
            if (t + gain >= c.ignition_temp) {
                heat.v[i].temp = t + gain;
                fx.v[c.fx_impact] = P.tick +% 1;
            }
        }
    }
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
