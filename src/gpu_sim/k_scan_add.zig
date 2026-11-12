//! Add each block's scanned offset back onto its elements (second half of the multi-level scan).
const c = @import("common.zig");
const L = c.L;

const P = c.uniform(c.ScanParams, 0, "sp");
const sums = c.ro(u32, 0, "sums");
const data = c.rw(u32, 0, "data");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const i = c.gid();
    if (i < P.n) data.v[i] += sums.v[i / c.scan_block];
}
