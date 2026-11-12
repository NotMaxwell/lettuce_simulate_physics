//! Broad phase step 3a: write each ball's id into its cell-sorted slot, and the slot back over
//! ball_slot (k_finish uses it to gather results in id order). Only the 4-byte id write is
//! random; k_gather then fills the sorted copies with linear writes and random reads, which
//! measured faster than scattering all the fields from here.
const c = @import("common.zig");
const L = c.L;

const ball_cell = c.ro(u32, 0, "ball_cell");
const starts = c.ro(u32, 1, "starts");
const sorted_id = c.rw(u32, 0, "sorted_id");
const ball_slot = c.rw(u32, 1, "ball_slot");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const P = c.params;
    const i = c.gid();
    if (i >= P.ball_count) return;
    const k = starts.v[ball_cell.v[i]] + ball_slot.v[i];
    ball_slot.v[i] = k;
    sorted_id.v[k] = i;
}
