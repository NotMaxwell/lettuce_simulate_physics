//! Zero the per-cell counters before the counting pass.
const c = @import("common.zig");
const L = c.L;

const P = c.uniform(c.ScanParams, 0, "sp");
const counts = c.rw(u32, 0, "counts");

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const i = c.gid();
    if (i < P.n) counts.v[i] = 0;
}
