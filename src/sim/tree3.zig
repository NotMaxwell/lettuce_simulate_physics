//! GPU tree build of the 3D sim (docs/sim3d.md, work package 4); kernels in k3/tree.zig, spec in
//! ref3.zig. Input: bodies sorted by Morton key (Pos3, mass) and the sorted SortRec keys. Output:
//! `nodes` (L.Node) and `quads` (L.NodeQuad) for the walk, numbered like ref3 (internal node i at
//! i, root 0; leaf j at leaves-1+j). The whole build is recorded into one command buffer with no
//! readback; the leaf count and the overflow flag stay on the GPU (`readStatus` fetches them).
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const sim = @import("sim.zig");

const tree_spv = @embedFile("k3_tree.spv");

pub const Error = sim.Error;
const Buffer = *c.SDL_GPUBuffer;

/// Must stay identical to `Params` in k3/tree.zig.
const KParams = extern struct {
    n: u32,
    node_cap: u32,
    leaf_cap: u32,
    depth: u32,
    box_scale: f32,
    theta: f32,
    scan_count: u32 = 0,
    scan_data: u32 = 0,
    scan_sums: u32 = 0,
    off_hdr: u32,
    off_lstart: u32,
    off_lkey: u32,
    off_parent: u32,
    off_bbox: u32,
    off_sums0: u32,
    off_total: u32,
    one: f32 = 1,
    off_rem: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(KParams) == 80);
}

const block: u32 = 2048;
const dispatch_x: u32 = 32768;
/// Internal nodes sit at depth <= 62: leaf keys are distinct 63-bit values and each level splits
/// at a strictly longer common prefix.
const max_internal_depth: u32 = 62;
const max_levels = 4;
const hdr_words = 16;

const Kernel = enum { leafCount, scanBlock, scanAdd, leafWrite, leafFinish, karras, depthLeaf, moments };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;

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

pub const Status = struct { leaves: u32, nodes: u32, overflow: bool };

/// N/4 nodes as in the design (measured need 0.09N), but never fewer than a complete tree of
/// small inputs needs, so tiny N cannot overflow.
pub fn defaultNodeCapacity(bodies: u32) u32 {
    return @max(bodies / 4, @min(2 * bodies - 1, 1024));
}

pub const Tree3 = struct {
    dev: *c.SDL_GPUDevice,
    capacity: u32,
    node_cap: u32,
    leaf_cap: u32,
    nodes: Buffer,
    quads: Buffer,
    scratch: Buffer,
    /// Indirect dispatch arguments and status words (see leafFinish in k3/tree.zig).
    args: Buffer,
    pipes: [n_kernels]*c.SDL_GPUComputePipeline,
    off_lstart: u32,
    off_lkey: u32,
    off_parent: u32,
    off_bbox: u32,
    off_rem: u32,
    /// Start of each scan level, sized for `capacity`.
    level_off: [max_levels + 1]u32,

    pub fn init(dev: *c.SDL_GPUDevice, capacity: u32) Error!Tree3 {
        return initNodes(dev, capacity, defaultNodeCapacity(capacity));
    }

    /// Like `init` with an explicit node capacity (overflow tests, memory tuning).
    pub fn initNodes(dev: *c.SDL_GPUDevice, capacity: u32, node_cap: u32) Error!Tree3 {
        if (capacity < 1 or capacity > 1 << 28 or node_cap < 1) return error.OutOfRange;
        var t: Tree3 = undefined;
        t.dev = dev;
        t.capacity = capacity;
        t.node_cap = node_cap;
        t.leaf_cap = (node_cap + 1) / 2;
        t.off_lstart = hdr_words;
        t.off_lkey = t.off_lstart + t.leaf_cap + 1;
        t.off_parent = t.off_lkey + 2 * t.leaf_cap;
        t.off_bbox = t.off_parent + node_cap;
        t.off_rem = t.off_bbox + 6 * node_cap;
        var at: u64 = @as(u64, t.off_rem) + 3 * @as(u64, node_cap);
        var size = ceilDiv(capacity, block);
        for (&t.level_off) |*o| {
            o.* = @intCast(at);
            at += size;
            size = ceilDiv(size, block);
        }
        const words = at;

        t.nodes = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.Node), 0);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.nodes);
        t.quads = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.NodeQuad), 0);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.quads);
        t.scratch = try makeBuffer(dev, words * 4, 0);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.scratch);
        t.args = try makeBuffer(dev, 16 * 4, c.SDL_GPU_BUFFERUSAGE_INDIRECT);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.args);

        var infos: [n_kernels]c.SDL_GPUComputePipelineCreateInfo = undefined;
        inline for (@typeInfo(Kernel).@"enum".field_names, &infos) |name, *info| {
            info.* = .{
                .code_size = tree_spv.len,
                .code = tree_spv.ptr,
                .entrypoint = name,
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = 3,
                .num_readwrite_storage_buffers = if (comptime std.mem.eql(u8, name, "leafFinish")) 4 else 3,
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
        for ([_]Buffer{ t.nodes, t.quads, t.scratch, t.args }) |b| c.SDL_ReleaseGPUBuffer(t.dev, b);
        for (t.pipes) |p| c.SDL_ReleaseGPUComputePipeline(t.dev, p);
        t.* = undefined;
    }

    /// Scratch and output bytes per body at the default node capacity (for memory tables).
    pub fn bytesPerBody(t: *const Tree3) f64 {
        const words: f64 = @floatFromInt(t.level_off[max_levels]);
        const out: f64 = @floatFromInt(@as(u64, t.node_cap) * (@sizeOf(L.Node) + @sizeOf(L.NodeQuad)));
        return (words * 4 + out) / @as(f64, @floatFromInt(t.capacity));
    }

    /// Records the build of `n` sorted bodies. Every dispatch is short (the longest is one pass
    /// over the bodies), so the whole build fits one submit up to 2^28 bodies; the exception is a
    /// single leaf of millions of bodies sharing one 63-bit key, which one thread sums alone.
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
            .off_lstart = t.off_lstart,
            .off_lkey = t.off_lkey,
            .off_parent = t.off_parent,
            .off_bbox = t.off_bbox,
            .off_rem = t.off_rem,
            .off_sums0 = t.level_off[0],
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
        kp.off_total = t.level_off[top];

        const groups0 = ceilDiv(n, block);
        t.pass(cmd, .leafCount, kp, in, .{ .direct = groups0 });
        for (0..top) |k| {
            var sp = kp;
            sp.scan_count = sizes[k];
            sp.scan_data = t.level_off[k];
            sp.scan_sums = t.level_off[k + 1];
            t.pass(cmd, .scanBlock, sp, in, .{ .direct = ceilDiv(sizes[k], block) });
        }
        var k = top - 1;
        while (k > 0) {
            k -= 1;
            var sp = kp;
            sp.scan_count = sizes[k];
            sp.scan_data = t.level_off[k];
            sp.scan_sums = t.level_off[k + 1];
            t.pass(cmd, .scanAdd, sp, in, .{ .direct = ceilDiv(sizes[k], L.group_size) });
        }
        t.pass(cmd, .leafWrite, kp, in, .{ .direct = groups0 });
        t.pass(cmd, .leafFinish, kp, in, .{ .direct = 1 });
        t.pass(cmd, .karras, kp, in, .{ .indirect = 0 });
        t.pass(cmd, .depthLeaf, kp, in, .{ .indirect = 12 });
        // One pass per depth so both children are final before their parent reads them.
        var d: u32 = max_internal_depth + 1;
        while (d > 0) {
            d -= 1;
            kp.depth = d;
            t.pass(cmd, .moments, kp, in, .{ .indirect = 0 });
        }
    }

    const Dispatch = union(enum) { direct: u32, indirect: u32 };

    fn pass(t: *Tree3, cmd: *c.SDL_GPUCommandBuffer, k: Kernel, kp: KParams, in: Inputs, how: Dispatch) void {
        c.SDL_PushGPUComputeUniformData(cmd, 0, &kp, @sizeOf(KParams));
        const rw = [4]c.SDL_GPUStorageBufferReadWriteBinding{ .{ .buffer = t.nodes }, .{ .buffer = t.quads }, .{ .buffer = t.scratch }, .{ .buffer = t.args } };
        const n_rw: u32 = if (k == .leafFinish) 4 else 3;
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
        var w: [3]u32 = undefined;
        try download(t.dev, t.args, 24, std.mem.sliceAsBytes(&w));
        return .{ .leaves = w[0], .nodes = w[1], .overflow = w[2] != 0 };
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
