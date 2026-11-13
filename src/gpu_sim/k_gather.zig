//! Broad phase step 3b: copy the fields the solver reads into cell-sorted order so neighbour
//! loads are contiguous (linear writes, random reads by id). Also clears the per-slot spin
//! accumulator: one PosRad record per slot. Radius is -1 for dead
//! balls, which k_count put in the extra last cell, i.e. at slots >= starts[ncells]; the cell is
//! recomputed from the position (same inputs as k_count) rather than read back by id.
const c = @import("common.zig");
const L = c.L;

const pos = c.ro(L.Vec2, 0, "pos");
const vel = c.ro(L.Vec2, 1, "vel");
const body = c.ro(L.BallBody, 2, "body");
const starts = c.ro(u32, 3, "starts");
const sorted_id = c.ro(u32, 4, "sorted_id");
const s_prm = c.rw(c.PosRad, 0, "s_prm");
const s_vel = c.rw(L.Vec2, 1, "s_vel");
const s_spin = c.rw(f32, 2, "s_spin");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const k = c.gid();
    if (k >= P.ball_count) return;
    const i = sorted_id.v[k];
    const p = pos.v[i];
    const dead = k >= starts.v[P.grid_cells_x * P.grid_cells_y];
    s_vel.v[k] = vel.v[i];
    s_spin.v[k] = 0;
    const b = body.v[i];
    s_prm.v[k] = .{ .x = p.x, .y = p.y, .r = if (dead) -1 else b.radius, .m = b.mass };
}
