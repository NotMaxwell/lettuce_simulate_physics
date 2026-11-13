//! GPU tree build of the 3D sim (docs/sim3d.md, work package 4); kernels in k3/tree.zig, spec in
//! ref3.zig. Input: bodies sorted by Morton key (Pos3, mass) and the sorted SortRec keys. Output:
//! `nodes` (L.Node) and `quads` (L.NodeQuad) for the walk, numbered like ref3 (internal node i at
//! i, root 0; leaf j at leaves-1+j). The whole build is recorded into one command buffer with no
//! readback. Status words stay on the GPU (`statusBuffer`, `readStatus`).
//!
//! Overflow (more nodes than the capacity): no pass after the leaf count runs and the root is
//! overwritten with an empty leaf (count 0, mass 0, rope = end), so a walk over the buffers gives
//! zero force rather than last build's tree. The overflow word tells the integrator to grow.
//!
//! Leaf counts in `info` saturate at 2^26-1 (ref3 raises LeafTooLarge instead). Only a leaf of
//! that many bodies all sharing one 63-bit key can reach it; its moments are still exact (summed
//! from the leaf starts), but a walk reading the count would skip its remaining bodies.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const sim = @import("sim.zig");
const Sort3 = @import("sort3.zig").Sort3;

const tree_spv = @embedFile("k3_tree.spv");

pub const Error = sim.Error || error{ScratchTooSmall};
const Buffer = *c.SDL_GPUBuffer;
const KParams = L.TreeParams3;

const block: u32 = 2048;
const dispatch_x: u32 = 32768;
/// Internal nodes sit at depth <= 62: leaf keys are distinct 63-bit values and each level splits
/// at a strictly longer common prefix. k3/tree.zig depth_passes.
const depth_passes: u32 = 63;
/// k3/tree.zig arg_moments: word offset of the per-depth indirect arguments.
const arg_moments: u32 = 10;
const args_words: u32 = arg_moments + 3 * depth_passes;
const max_levels = 4;
const hdr_words = 4;

/// Word offsets of the status in `statusBuffer()` (u32 each), for kernels that check the build
/// without a readback.
pub const status_leaves: u32 = 6;
pub const status_nodes: u32 = 7;
/// Nonzero when the tree did not fit; the root is then an empty leaf.
pub const status_overflow: u32 = 8;
pub const status_max_depth: u32 = 9;

const Kernel = enum { leafCount, scanBlock, scanAdd, leafWrite, leafFinish, karras, depthLeaf, bigLeaves, depthArgs, moments };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;

fn bindsArgs(k: Kernel) bool {
    return k == .leafFinish or k == .depthArgs;
}

pub const Inputs = struct {
    /// Sorted positions (L.Pos3), masses (f32) and SortRec keys, n each.
    pos: Buffer,
    mass: Buffer,
    keys: Buffer,
};

pub const Params = struct {
    theta: f32,
    /// Physical side of the root cube: position u32 v is at v * box_size / 2^32 (ref3.phys).
    box_size: f64,
};

pub const Status = struct { leaves: u32, nodes: u32, overflow: bool, max_depth: u32 };

/// N/4 nodes as in the design (measured need 0.09N), but never fewer than a complete tree of
/// small inputs needs, so tiny N cannot overflow. Only for an owned scratch buffer.
pub fn defaultNodeCapacity(bodies: u32) u32 {
    return @max(bodies / 4, @min(2 * bodies - 1, 1024));
}

pub const Options = struct {
    node_cap: ?u32 = null,
    /// Borrowed scratch (e.g. Sort3's idle SortRec copy) instead of an owned one; `init` fails
    /// with ScratchTooSmall when the layout for `node_cap` does not fit `scratch_bytes`.
    scratch: ?Buffer = null,
    scratch_bytes: u64 = 0,
};

const Layout = struct {
    off_lstart: u32,
    off_lkey: u32,
    off_parent: u32,
    off_bbox: u32,
    off_rem: u32,
    off_dmax: u32,
    /// Start of each scan level, sized for the body capacity; the last entry is the end.
    level_off: [max_levels + 2]u32,

    fn words(lay: Layout) u64 {
        return lay.level_off[max_levels + 1];
    }

    fn of(capacity: u32, node_cap: u32) Layout {
        const leaf_cap = (node_cap + 1) / 2;
        var lay: Layout = undefined;
        lay.off_lstart = hdr_words;
        lay.off_lkey = lay.off_lstart + leaf_cap + 1;
        lay.off_parent = lay.off_lkey + 2 * leaf_cap;
        lay.off_bbox = lay.off_parent + node_cap;
        lay.off_rem = lay.off_bbox + 6 * node_cap;
        lay.off_dmax = lay.off_rem + 3 * node_cap;
        var at: u32 = lay.off_dmax + ceilDiv(node_cap, L.group_size);
        var size = ceilDiv(capacity, block);
        for (&lay.level_off) |*o| {
            o.* = at;
            at += size;
            size = ceilDiv(size, block);
        }
        return lay;
    }
};

/// Scratch bytes the build needs for `capacity` bodies and `node_cap` nodes.
pub fn scratchBytes(capacity: u32, node_cap: u32) u64 {
    return 4 * Layout.of(capacity, node_cap).words();
}

pub const Tree3 = struct {
    dev: *c.SDL_GPUDevice,
    capacity: u32,
    node_cap: u32,
    leaf_cap: u32,
    nodes: Buffer,
    quads: Buffer,
    scratch: Buffer,
    owns_scratch: bool,
    /// Indirect dispatch arguments and status words (layout in k3/tree.zig).
    args: Buffer,
    pipes: [n_kernels]*c.SDL_GPUComputePipeline,
    lay: Layout,

    pub fn init(dev: *c.SDL_GPUDevice, capacity: u32) Error!Tree3 {
        return initWith(dev, capacity, .{});
    }

    /// Like `init` with an explicit node capacity (overflow tests, memory tuning).
    pub fn initNodes(dev: *c.SDL_GPUDevice, capacity: u32, node_cap: u32) Error!Tree3 {
        return initWith(dev, capacity, .{ .node_cap = node_cap });
    }

    /// Scratch aliased onto the sort's idle SortRec copy (12 B/body), node capacity N/4. The tree
    /// must be recorded after the sort in the same or a later command buffer, and the next sort
    /// overwrites the scratch (not the tree). The layout takes ~11.5 B/body at N/4 nodes plus a few
    /// dozen words, so this fails with ScratchTooSmall below N ~ 80; use `init` there.
    pub fn initOnSort(dev: *c.SDL_GPUDevice, sort: *const Sort3) Error!Tree3 {
        return initWith(dev, sort.capacity, .{
            .node_cap = sort.capacity / 4,
            .scratch = sort.scratch(),
            .scratch_bytes = @as(u64, sort.capacity) * @sizeOf(L.SortRec),
        });
    }

    pub fn initWith(dev: *c.SDL_GPUDevice, capacity: u32, o: Options) Error!Tree3 {
        const node_cap = o.node_cap orelse defaultNodeCapacity(capacity);
        if (capacity < 1 or capacity > 1 << 28 or node_cap < 1 or node_cap > 2 * capacity) return error.OutOfRange;
        var t: Tree3 = undefined;
        t.dev = dev;
        t.capacity = capacity;
        t.node_cap = node_cap;
        t.leaf_cap = (node_cap + 1) / 2;
        t.lay = Layout.of(capacity, node_cap);
        const need = 4 * t.lay.words();
        if (o.scratch != null and need > o.scratch_bytes) {
            std.log.warn("tree3: scratch needs {d} B for {d} nodes, borrowed buffer has {d} B", .{ need, node_cap, o.scratch_bytes });
            return error.ScratchTooSmall;
        }

        t.nodes = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.Node), 0);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.nodes);
        t.quads = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.NodeQuad), 0);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.quads);
        t.owns_scratch = o.scratch == null;
        t.scratch = o.scratch orelse try makeBuffer(dev, need, 0);
        errdefer if (t.owns_scratch) c.SDL_ReleaseGPUBuffer(dev, t.scratch);
        t.args = try makeBuffer(dev, args_words * 4, c.SDL_GPU_BUFFERUSAGE_INDIRECT);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.args);

        var infos: [n_kernels]c.SDL_GPUComputePipelineCreateInfo = undefined;
        inline for (@typeInfo(Kernel).@"enum".field_names, &infos, 0..) |name, *info, i| {
            info.* = .{
                .code_size = tree_spv.len,
                .code = tree_spv.ptr,
                .entrypoint = name,
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = 3,
                .num_readwrite_storage_buffers = if (bindsArgs(@fromBackingInt(@intCast(i)))) 4 else 3,
                .num_uniform_buffers = 1,
                .threadcount_x = L.group_size,
                .threadcount_y = 1,
                .threadcount_z = 1,
            };
        }
        var made: [n_kernels]?*c.SDL_GPUComputePipeline = undefined;
        if (!@import("gpu_sim").createComputePipelines(dev, &infos, &made)) return error.GpuPipeline;
        for (&t.pipes, made) |*p, m| p.* = m.?;
        return t;
    }

    pub fn deinit(t: *Tree3) void {
        for ([_]Buffer{ t.nodes, t.quads, t.args }) |b| c.SDL_ReleaseGPUBuffer(t.dev, b);
        if (t.owns_scratch) c.SDL_ReleaseGPUBuffer(t.dev, t.scratch);
        for (t.pipes) |p| c.SDL_ReleaseGPUComputePipeline(t.dev, p);
        t.* = undefined;
    }

    /// Status words (see `status_*`), readable as a storage buffer by later passes.
    pub fn statusBuffer(t: *const Tree3) Buffer {
        return t.args;
    }

    /// Device bytes per body this owns (nodes, quads, args, and the scratch only when owned).
    pub fn bytesPerBody(t: *const Tree3) f64 {
        var bytes: f64 = @floatFromInt(@as(u64, t.node_cap) * (@sizeOf(L.Node) + @sizeOf(L.NodeQuad)) + args_words * 4);
        if (t.owns_scratch) bytes += @floatFromInt(4 * t.lay.words());
        return bytes / @as(f64, @floatFromInt(t.capacity));
    }

    /// Records the build of `n` sorted bodies. Every dispatch is short (the longest is one pass
    /// over the bodies; a leaf of one huge equal-key run is summed by a whole workgroup), so the
    /// build fits one submit up to 2^28 bodies.
    pub fn record(t: *Tree3, cmd: *c.SDL_GPUCommandBuffer, in: Inputs, n: u32, p: Params) Error!void {
        if (n < 1 or n > t.capacity) return error.OutOfRange;
        if (!(p.theta > 0 and p.theta < 1.1547)) return error.OutOfRange;
        var kp: KParams = .{
            .n = n,
            .node_cap = t.node_cap,
            .leaf_cap = t.leaf_cap,
            .depth = 0,
            .box_scale = @floatCast(p.box_size / 0x1p32),
            .theta = p.theta,
            .off_hdr = 0,
            .off_lstart = t.lay.off_lstart,
            .off_lkey = t.lay.off_lkey,
            .off_parent = t.lay.off_parent,
            .off_bbox = t.lay.off_bbox,
            .off_rem = t.lay.off_rem,
            .off_dmax = t.lay.off_dmax,
            .off_sums0 = t.lay.level_off[0],
            .off_total = 0,
        };
        // Scan levels for this n: level k+1 holds the block totals of level k; the first level of
        // size 1 is the leaf count.
        var sizes: [max_levels + 1]u32 = undefined;
        sizes[0] = ceilDiv(n, block);
        var top: usize = 0;
        while (true) {
            sizes[top + 1] = ceilDiv(sizes[top], block);
            top += 1;
            if (sizes[top] == 1) break;
        }
        kp.off_total = t.lay.level_off[top];

        const groups0 = ceilDiv(n, block);
        t.pass(cmd, .leafCount, kp, in, .{ .direct = groups0 });
        for (0..top) |k| {
            var sp = kp;
            sp.scan_count = sizes[k];
            sp.scan_data = t.lay.level_off[k];
            sp.scan_sums = t.lay.level_off[k + 1];
            t.pass(cmd, .scanBlock, sp, in, .{ .direct = ceilDiv(sizes[k], block) });
        }
        var k = top - 1;
        while (k > 0) {
            k -= 1;
            var sp = kp;
            sp.scan_count = sizes[k];
            sp.scan_data = t.lay.level_off[k];
            sp.scan_sums = t.lay.level_off[k + 1];
            t.pass(cmd, .scanAdd, sp, in, .{ .direct = ceilDiv(sizes[k], L.group_size) });
        }
        t.pass(cmd, .leafWrite, kp, in, .{ .direct = groups0 });
        t.pass(cmd, .leafFinish, kp, in, .{ .direct = 1 });
        t.pass(cmd, .karras, kp, in, .{ .indirect = 0 });
        t.pass(cmd, .depthLeaf, kp, in, .{ .indirect = 12 });
        t.pass(cmd, .bigLeaves, kp, in, .{ .indirect = 0 });
        t.pass(cmd, .depthArgs, kp, in, .{ .direct = 1 });
        // One pass per depth so both children are final before their parent reads them; depths
        // below the deepest internal node get zero groups from depthArgs.
        var d: u32 = depth_passes;
        while (d > 0) {
            d -= 1;
            kp.depth = d;
            t.pass(cmd, .moments, kp, in, .{ .indirect = 4 * (arg_moments + 3 * d) });
        }
    }

    const Dispatch = union(enum) { direct: u32, indirect: u32 };

    fn pass(t: *Tree3, cmd: *c.SDL_GPUCommandBuffer, k: Kernel, kp: KParams, in: Inputs, how: Dispatch) void {
        c.SDL_PushGPUComputeUniformData(cmd, 0, &kp, @sizeOf(KParams));
        const rw = [4]c.SDL_GPUStorageBufferReadWriteBinding{ .{ .buffer = t.nodes }, .{ .buffer = t.quads }, .{ .buffer = t.scratch }, .{ .buffer = t.args } };
        const n_rw: u32 = if (bindsArgs(k)) 4 else 3;
        const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &rw, n_rw);
        c.SDL_BindGPUComputePipeline(cp, t.pipes[@backingInt(k)]);
        const ro = [3]Buffer{ in.pos, in.mass, in.keys };
        c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(&ro), 3);
        switch (how) {
            .direct => |groups| c.SDL_DispatchGPUCompute(cp, @min(groups, dispatch_x), ceilDiv(groups, dispatch_x), 1),
            .indirect => |off| c.SDL_DispatchGPUComputeIndirect(cp, t.args, off),
        }
        c.SDL_EndGPUComputePass(cp);
    }

    /// Leaf and node counts of the last build and whether they exceeded the capacity (blocking).
    pub fn readStatus(t: *Tree3) Error!Status {
        var w: [4]u32 = undefined;
        try download(t.dev, t.args, 4 * status_leaves, std.mem.sliceAsBytes(&w));
        return .{ .leaves = w[0], .nodes = w[1], .overflow = w[2] != 0, .max_depth = w[3] };
    }
};

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// Blocking read of `out.len` bytes at `offset` of `buf` (status, tests).
pub fn download(dev: *c.SDL_GPUDevice, buf: Buffer, offset: u32, out: []u8) Error!void {
    const size: u32 = @intCast(out.len);
    const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = size }) orelse return error.GpuBuffer;
    defer c.SDL_ReleaseGPUTransferBuffer(dev, tb);
    const cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
    const cp = c.SDL_BeginGPUCopyPass(cmd);
    c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = buf, .offset = offset, .size = size }, &.{ .transfer_buffer = tb, .offset = 0 });
    c.SDL_EndGPUCopyPass(cp);
    try submitWait(dev, cmd);
    const m: [*]const u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(dev, tb, false) orelse return error.GpuBuffer);
    defer c.SDL_UnmapGPUTransferBuffer(dev, tb);
    @memcpy(out, m[0..out.len]);
}

pub fn submitWait(dev: *c.SDL_GPUDevice, cmd: *c.SDL_GPUCommandBuffer) Error!void {
    var f = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSubmit;
    _ = c.SDL_WaitForGPUFences(dev, true, &f, 1);
    c.SDL_ReleaseGPUFence(dev, f);
}

pub fn makeBuffer(dev: *c.SDL_GPUDevice, size: u64, extra_usage: u32) Error!Buffer {
    if (size > std.math.maxInt(u32)) return error.OutOfRange;
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE | extra_usage,
        .size = @intCast(@max(size, 16)),
    }) orelse error.GpuBuffer;
}
