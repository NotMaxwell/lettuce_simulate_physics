//! In-place exclusive scan of one block of 1024 elements per workgroup (4 per thread, then a
//! Hillis-Steele scan of the 256 thread totals in shared memory). Each block's total goes to
//! `sums`; the host scans `sums` with this same kernel and adds it back with k_scan_add.
const c = @import("common.zig");
const L = c.L;
const spirv = c.spirv;

const P = c.uniform(c.ScanParams, 0, "sp");
const data = c.rw(u32, 0, "data");
const sums = c.rw(u32, 1, "sums");
const tmp = @extern(*addrspace(.shared) [L.group_size]u32, .{ .name = "tmp" });

export fn main() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const n = P.n;
    const l = c.lid();
    const block = spirv.workgroup_id[0];
    const base = block * c.scan_block + l * c.scan_per_thread;

    var vals: [c.scan_per_thread]u32 = undefined;
    var total: u32 = 0;
    inline for (0..c.scan_per_thread) |k| {
        const idx = base + k;
        const x = if (idx < n) data.v[idx] else 0;
        vals[k] = total;
        total += x;
    }

    tmp[l] = total;
    spirv.workgroupBarrier();
    var off: u32 = 1;
    while (off < L.group_size) : (off <<= 1) {
        const add = if (l >= off) tmp[l - off] else 0;
        spirv.workgroupBarrier();
        tmp[l] += add;
        spirv.workgroupBarrier();
    }
    const excl = tmp[l] - total;

    inline for (0..c.scan_per_thread) |k| {
        const idx = base + k;
        if (idx < n) data.v[idx] = vals[k] + excl;
    }
    if (l == L.group_size - 1) sums.v[block] = tmp[l];
}
