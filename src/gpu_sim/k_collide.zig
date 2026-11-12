//! Narrow phase, one Jacobi iteration: each ball (thread = cell-sorted slot k) gathers the
//! separation and impulse from every overlapping neighbour in the 3x3 cells around the cell it
//! was binned in, and writes only its own slot of the destination buffers. The host ping-pongs
//! source/destination for 2-4 iterations.
//!
//! Deviations from the sequential CPU resolvePair, needed because all pairs resolve at once:
//! - Each pair's impulse is divided by max(contacts of i, contacts of j).
//!   Unscaled, a ball struck by two neighbours at once receives both full elastic impulses
//!   and the solver gains energy (measured +19% KE in 60 ticks with no gravity, and piles
//!   exploded). Using the same divisor on both sides keeps momentum exactly conserved.
//!   Isolated pairs (one contact each) stay perfectly elastic as on the CPU.
//! - Separation is mass-weighted (the heavier ball moves less) and averaged over the ball's own
//!   contacts with over-relaxation `pos_relax`.
//! - Spin from the tangential speed is applied on iteration 0 only, so it is not counted twice.
//! Contact counts come from the previous iteration (or k_contacts before iteration 0); this
//! iteration's counts are written for the next one.
const c = @import("common.zig");
const L = c.L;

const src_pos = c.ro(L.Vec2, 0, "src_pos");
const src_vel = c.ro(L.Vec2, 1, "src_vel");
const s_body = c.ro(L.BallBody, 2, "s_body");
const starts = c.ro(u32, 3, "starts");
const sorted_id = c.ro(u32, 4, "sorted_id");
const meta = c.ro(L.BallMeta, 5, "meta");
const src_cnt = c.ro(u32, 6, "src_cnt");
const dst_pos = c.rw(L.Vec2, 0, "dst_pos");
const dst_vel = c.rw(L.Vec2, 1, "dst_vel");
const state = c.rw(L.BallState, 2, "state");
const events = c.rw(L.Event, 3, "events");
const dst_cnt = c.rw(u32, 4, "dst_cnt");

const CP = c.uniform(c.CollideParams, 1, "cp");

/// The CPU/Python ball-ball impulse is perfectly elastic (2 * rel_speed); kept here.
const restitution: f32 = 1.0;

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const k = c.gid();
    if (k >= P.ball_count) return;
    const b = s_body.v[k];
    var p = src_pos.v[k];
    var v = src_vel.v[k];
    if (b.radius <= 0) {
        dst_pos.v[k] = p;
        dst_vel.v[k] = v;
        dst_cnt.v[k] = 0;
        return;
    }

    const first = CP.iteration == 0;
    const merging = P.flags & c.flag_merging != 0;
    const g = c.neighbourhood(@bitCast(b.friction), P);
    const my_cnt: f32 = @floatFromInt(src_cnt.v[k]);
    const pos_scale = CP.pos_relax / @max(1.0, my_cnt);

    var my_level: u32 = 0;
    var my_id: u32 = 0;
    if (merging) {
        my_id = sorted_id.v[k];
        my_level = c.level(meta.v[my_id]);
    }

    var dpx: f32 = 0;
    var dpy: f32 = 0;
    var dvx: f32 = 0;
    var dvy: f32 = 0;
    var spin: f32 = 0;
    var contacts: u32 = 0;

    var y = g.y0;
    while (y <= g.y1) : (y += 1) {
        // Cells of one row are contiguous in the sorted order, so a row is a single slot range.
        const row = y * P.grid_cells_x;
        const j_end = starts.v[row + g.x1 + 1];
        var j = starts.v[row + g.x0];
        while (j < j_end) : (j += 1) {
            if (j == k) continue;
            const q = src_pos.v[j];
            const dx = q.x - p.x;
            const dy = q.y - p.y;
            const d2 = dx * dx + dy * dy;
            const bj = s_body.v[j];
            const rsum = b.radius + bj.radius;
            if (!(d2 > 0 and d2 < rsum * rsum)) continue;

            if (merging) {
                const jid = sorted_id.v[j];
                const jl = c.level(meta.v[jid]);
                if (jl == my_level and my_level < c.max_level) {
                    // Merging pairs get no contact response (as on the CPU); the CPU applies
                    // the merge from this event one frame later.
                    if (first and k < j) {
                        const slot = c.atomicAdd(&events.v[0].kind, 1);
                        if (slot < L.max_events - 1) {
                            events.v[slot + 1] = .{ .kind = L.event_merge, .a = my_id, .b = jid, .value = @sqrt(d2) };
                        } else {
                            _ = c.atomicAdd(&events.v[0].a, 1);
                        }
                    }
                    continue;
                }
            }
            contacts += 1;

            const dist = @sqrt(d2);
            const ux = dx / dist;
            const uy = dy / dist;
            const share = 1.0 / @max(1.0, @max(my_cnt, @as(f32, @floatFromInt(src_cnt.v[j]))));
            const mass_frac = bj.mass / (b.mass + bj.mass);
            const wi = mass_frac * share;
            // Separation does not touch momentum, so it is averaged over this ball's own contacts
            // only (position-based-dynamics style, over-relaxed by pos_relax) and converges faster.
            const push = (rsum - dist) * mass_frac * pos_scale;
            dpx -= ux * push;
            dpy -= uy * push;

            const vj = src_vel.v[j];
            const rvx = v.x - vj.x;
            const rvy = v.y - vj.y;
            const rel = rvx * ux + rvy * uy;
            if (rel > CP.rest_speed) {
                const imp = (1 + restitution) * rel * wi;
                dvx -= imp * ux;
                dvy -= imp * uy;
            } else {
                // Resting contact: cancel the approach and turn the overlap into separating
                // speed (position-based-dynamics style). Without this a deep pile crushes:
                // a whole column falls at one speed, so relative speeds are ~0 and each Jacobi
                // iteration only passes the floor's stop one ball further up. Symmetric in i/j,
                // so momentum stays exact.
                const imp = (@max(rel, 0) + (rsum - dist) * CP.push_to_vel) * wi;
                dvx -= imp * ux;
                dvy -= imp * uy;
            }
            if (first) spin += (-rvx * uy + rvy * ux) / b.radius * 0.5;
        }
    }

    p.x += dpx;
    p.y += dpy;
    v.x += dvx;
    v.y += dvy;
    // Walls act inside every iteration; otherwise a ball pressed against the floor absorbs half of
    // each push from above by sinking into it, and stacks crush from the bottom. Velocities are
    // still reflected in k_finish, which treats a ball exactly on a wall as touching it.
    p = c.clampToWalls(p, b.radius);
    dst_pos.v[k] = p;
    dst_vel.v[k] = v;
    dst_cnt.v[k] = contacts;
    if (first and spin != 0) {
        const id = sorted_id.v[k];
        state.v[id].ang_vel += spin;
    }
}
