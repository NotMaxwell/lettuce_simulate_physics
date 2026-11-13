//! Host side of the 3D tree kick (k3/walk.zig): pipelines, chunked dispatch, and a GPU copy of a
//! tree built on the CPU by ref3.build (the GPU build, work package 4, must produce the same bytes).
//!
//! Bindings, in order. Read-only: pos (sorted), mass (sorted), nodes, quads. Read-write: vel, amag,
//! pot (all three per sorted body). Flags are gpu_layout's sim3_flag_*: store_accel writes the
//! acceleration into vel instead of kicking, potential fills pot.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const ref3 = @import("ref3.zig");
const sim = @import("sim.zig");

const walk_spv = @embedFile("k3_walk.spv");

pub const Error = sim.Error;
pub const Buffer = *c.SDL_GPUBuffer;
const dispatch_x: u32 = 32768;

/// Must match k3/walk.zig WalkParams3 field for field.
pub const WalkParams3 = extern struct {
    n: u32,
    first: u32,
    count: u32,
    flags: u32,
    g: f32,
    eps: f32,
    eps2: f32,
    kick: f32,
    box_scale: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(WalkParams3) == 48);
    // ref3's arrays are uploaded byte for byte.
    std.debug.assert(@sizeOf(ref3.Node) == @sizeOf(L.Node) and @sizeOf(ref3.NodeQuad) == @sizeOf(L.NodeQuad));
    for (@typeInfo(L.Node).@"struct".field_names) |f| std.debug.assert(@offsetOf(L.Node, f) == @offsetOf(ref3.Node, f));
    for (@typeInfo(L.NodeQuad).@"struct".field_names) |f| std.debug.assert(@offsetOf(L.NodeQuad, f) == @offsetOf(ref3.NodeQuad, f));
}

/// Workgroup sizes with an entry point in k3/walk.zig (`walk<size>`).
pub const group_sizes = [_]u32{ 64, 128, 256, 512, 1024 };

/// Measured on an RX 7800 XT (walk3_tests timing table): 1024 is 12% faster than 256 from 1M bodies
/// up, presumably because a workgroup's Morton-adjacent bodies share one CU's cache; at 100k the
/// big groups leave CUs idle and 256 wins.
pub fn autoGroupSize(n: u32) u32 {
    return if (n >= 1 << 19) 1024 else 256;
}

/// Bodies per submitted command buffer. At 16M Plummer bodies one body costs about 55 ns of GPU
/// time, so a chunk is ~60 ms: far under the driver's job watchdog even on a slower or shared GPU.
pub const chunk_bodies: u32 = 1 << 20;

pub const Kick = struct {
    n: u32,
    g: f32,
    eps: f32,
    /// vel += kick * a (ignored with sim3_flag_store_accel).
    kick: f32,
    /// Length per position quantum: box_size / 2^32.
    box_scale: f32,
    flags: u32 = 0,
};

pub const Bindings = struct { pos: Buffer, mass: Buffer, nodes: Buffer, quads: Buffer, vel: Buffer, amag: Buffer, pot: Buffer };

pub const Walk3 = struct {
    dev: *c.SDL_GPUDevice,
    pipes: [group_sizes.len]*c.SDL_GPUComputePipeline,
    /// 0 picks autoGroupSize(n).
    group_size: u32 = 0,
    chunk: u32 = chunk_bodies,

    pub fn init(dev: *c.SDL_GPUDevice) Error!Walk3 {
        var infos: [group_sizes.len]c.SDL_GPUComputePipelineCreateInfo = undefined;
        inline for (group_sizes, &infos) |gs, *info| {
            info.* = .{
                .code_size = walk_spv.len,
                .code = walk_spv.ptr,
                .entrypoint = std.fmt.comptimePrint("walk{d}", .{gs}),
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = 4,
                .num_readwrite_storage_buffers = 3,
                .num_uniform_buffers = 1,
                .threadcount_x = gs,
                .threadcount_y = 1,
                .threadcount_z = 1,
            };
        }
        var made: [group_sizes.len]?*c.SDL_GPUComputePipeline = undefined;
        if (!@import("gpu_sim").createComputePipelines(dev, &infos, &made)) return error.GpuPipeline;
        var w: Walk3 = .{ .dev = dev, .pipes = undefined };
        for (&w.pipes, made) |*p, m| p.* = m.?;
        return w;
    }

    pub fn deinit(w: *Walk3) void {
        for (w.pipes) |p| c.SDL_ReleaseGPUComputePipeline(w.dev, p);
        w.* = undefined;
    }

    /// Records the kick of bodies [0, k.n) into `cmd0`, one chunk of `w.chunk` bodies per command
    /// buffer. Every chunk but the last is submitted here without waiting (chunks write disjoint
    /// ranges and later passes are ordered after them by SDL_GPU's pass barriers); the returned
    /// command buffer holds the last chunk and is the caller's to submit.
    pub fn record(w: *Walk3, cmd0: *c.SDL_GPUCommandBuffer, b: Bindings, k: Kick) Error!*c.SDL_GPUCommandBuffer {
        const gs = if (w.group_size == 0) autoGroupSize(k.n) else w.group_size;
        const pi = for (group_sizes, 0..) |s, i| {
            if (s == gs) break i;
        } else return error.OutOfRange;
        var cmd = cmd0;
        var first: u32 = 0;
        while (true) {
            const count = @min(w.chunk, k.n - first);
            const p: WalkParams3 = .{
                .n = k.n,
                .first = first,
                .count = count,
                .flags = k.flags,
                .g = k.g,
                .eps = k.eps,
                .eps2 = k.eps * k.eps,
                .kick = k.kick,
                .box_scale = k.box_scale,
            };
            c.SDL_PushGPUComputeUniformData(cmd, 0, &p, @sizeOf(WalkParams3));
            const rw = [3]c.SDL_GPUStorageBufferReadWriteBinding{ .{ .buffer = b.vel }, .{ .buffer = b.amag }, .{ .buffer = b.pot } };
            const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &rw, rw.len);
            c.SDL_BindGPUComputePipeline(cp, w.pipes[pi]);
            const ro = [4]?*c.SDL_GPUBuffer{ b.pos, b.mass, b.nodes, b.quads };
            c.SDL_BindGPUComputeStorageBuffers(cp, 0, &ro, ro.len);
            const groups = (count + gs - 1) / gs;
            c.SDL_DispatchGPUCompute(cp, @min(groups, dispatch_x), (groups + dispatch_x - 1) / dispatch_x, 1);
            c.SDL_EndGPUComputePass(cp);
            first += count;
            if (first >= k.n) return cmd;
            if (!c.SDL_SubmitGPUCommandBuffer(cmd)) return error.GpuSubmit;
            cmd = c.SDL_AcquireGPUCommandBuffer(w.dev) orelse return error.GpuSubmit;
        }
    }
};

/// A tree and its sorted bodies on the GPU, uploaded from ref3.build's output.
pub const GpuTree = struct {
    dev: *c.SDL_GPUDevice,
    pos: Buffer,
    mass: Buffer,
    nodes: Buffer,
    quads: Buffer,
    n: u32,
    node_cap: u32,
    node_count: u32 = 0,

    pub fn init(dev: *c.SDL_GPUDevice, n: u32, node_cap: u32) Error!GpuTree {
        var t: GpuTree = .{ .dev = dev, .pos = undefined, .mass = undefined, .nodes = undefined, .quads = undefined, .n = n, .node_cap = node_cap };
        t.pos = try makeBuffer(dev, @as(u64, n) * @sizeOf(L.Pos3));
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.pos);
        t.mass = try makeBuffer(dev, @as(u64, n) * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.mass);
        t.nodes = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.Node));
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.nodes);
        t.quads = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.NodeQuad));
        return t;
    }

    pub fn deinit(t: *GpuTree) void {
        for ([_]Buffer{ t.pos, t.mass, t.nodes, t.quads }) |b| c.SDL_ReleaseGPUBuffer(t.dev, b);
        t.* = undefined;
    }

    /// Blocking upload; node buffers grow when the tree has more nodes than they hold.
    pub fn upload(t: *GpuTree, tree: ref3.Tree) Error!void {
        if (tree.pos.len != t.n) return error.OutOfRange;
        if (tree.nodes.len > t.node_cap) {
            const cap: u32 = @intCast(tree.nodes.len + tree.nodes.len / 4);
            const nodes = try makeBuffer(t.dev, @as(u64, cap) * @sizeOf(L.Node));
            errdefer c.SDL_ReleaseGPUBuffer(t.dev, nodes);
            const quads = try makeBuffer(t.dev, @as(u64, cap) * @sizeOf(L.NodeQuad));
            c.SDL_ReleaseGPUBuffer(t.dev, t.nodes);
            c.SDL_ReleaseGPUBuffer(t.dev, t.quads);
            t.nodes = nodes;
            t.quads = quads;
            t.node_cap = cap;
        }
        try uploadBytes(t.dev, t.pos, std.mem.sliceAsBytes(tree.pos));
        try uploadBytes(t.dev, t.mass, std.mem.sliceAsBytes(tree.mass));
        try uploadBytes(t.dev, t.nodes, std.mem.sliceAsBytes(tree.nodes));
        try uploadBytes(t.dev, t.quads, std.mem.sliceAsBytes(tree.quads));
        t.node_count = @intCast(tree.nodes.len);
    }

    pub fn bindings(t: *const GpuTree, vel: Buffer, amag: Buffer, pot: Buffer) Bindings {
        return .{ .pos = t.pos, .mass = t.mass, .nodes = t.nodes, .quads = t.quads, .vel = vel, .amag = amag, .pot = pot };
    }
};

pub fn makeBuffer(dev: *c.SDL_GPUDevice, size: u64) Error!Buffer {
    if (size > std.math.maxInt(u32)) return error.OutOfRange;
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = @intCast(@max(size, 16)),
    }) orelse error.GpuBuffer;
}

const staging_bytes: usize = 64 << 20;

pub fn submitWait(dev: *c.SDL_GPUDevice, cmd: *c.SDL_GPUCommandBuffer) Error!void {
    var f = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSubmit;
    _ = c.SDL_WaitForGPUFences(dev, true, &f, 1);
    c.SDL_ReleaseGPUFence(dev, f);
}

/// Blocking copy of `bytes` to the start of `buf`.
pub fn uploadBytes(dev: *c.SDL_GPUDevice, buf: Buffer, bytes: []const u8) Error!void {
    if (bytes.len == 0) return;
    const stage = @min(bytes.len, staging_bytes);
    const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = @intCast(stage) }) orelse return error.GpuBuffer;
    defer c.SDL_ReleaseGPUTransferBuffer(dev, tb);
    var at: usize = 0;
    while (at < bytes.len) : (at += stage) {
        const part = bytes[at..@min(at + stage, bytes.len)];
        const m: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(dev, tb, false) orelse return error.GpuBuffer);
        @memcpy(m[0..part.len], part);
        c.SDL_UnmapGPUTransferBuffer(dev, tb);
        const cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = tb }, &.{ .buffer = buf, .offset = @intCast(at), .size = @intCast(part.len) }, false);
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(dev, cmd);
    }
}

/// Blocking copy of the start of `buf` into `out`.
pub fn downloadBytes(dev: *c.SDL_GPUDevice, buf: Buffer, out: []u8) Error!void {
    if (out.len == 0) return;
    const stage = @min(out.len, staging_bytes);
    const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = @intCast(stage) }) orelse return error.GpuBuffer;
    defer c.SDL_ReleaseGPUTransferBuffer(dev, tb);
    var at: usize = 0;
    while (at < out.len) : (at += stage) {
        const part = out[at..@min(at + stage, out.len)];
        const cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = buf, .offset = @intCast(at), .size = @intCast(part.len) }, &.{ .transfer_buffer = tb, .offset = 0 });
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(dev, cmd);
        const m: [*]const u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(dev, tb, false) orelse return error.GpuBuffer);
        @memcpy(part, m[0..part.len]);
        c.SDL_UnmapGPUTransferBuffer(dev, tb);
    }
}
