//! Host side of the 3D tree kick (k3/walk.zig): pipelines, chunked dispatch, and a GPU copy of a
//! tree built on the CPU by ref3.build (the GPU build, work package 4, must produce the same bytes).
//!
//! Bindings, in order. Read-only: pos (sorted), mass (sorted), nodes, quads, tree3 status words.
//! Read-write: vel, amag, pot (all three per sorted body). Flags are gpu_layout's sim3_flag_*: store_accel writes the
//! acceleration into vel instead of kicking, potential fills pot.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const ref3 = @import("ref3.zig");
const sim = @import("sim.zig");
const tree3 = @import("tree3.zig");

const walk_spv = @embedFile("k3_walk.spv");

pub const Error = sim.Error;
pub const Buffer = *c.SDL_GPUBuffer;
const dispatch_x: u32 = 32768;

comptime {
    // ref3's arrays are uploaded byte for byte.
    std.debug.assert(@sizeOf(ref3.Node) == @sizeOf(L.Node) and @sizeOf(ref3.NodeQuad) == @sizeOf(L.NodeQuad));
    for (@typeInfo(L.Node).@"struct".field_names) |f| std.debug.assert(@offsetOf(L.Node, f) == @offsetOf(ref3.Node, f));
    for (@typeInfo(L.NodeQuad).@"struct".field_names) |f| std.debug.assert(@offsetOf(L.NodeQuad, f) == @offsetOf(ref3.NodeQuad, f));
}

/// Workgroup sizes with an entry point in k3/walk.zig (`walk<size>`), ascending.
pub const group_sizes = [_]u32{ 64, 128, 256, 512, 1024 };

/// Measured on an RX 7800 XT (walk3_tests timing table): 1024 is 12% faster than 256 from 1M bodies
/// up, presumably because a workgroup's Morton-adjacent bodies share one CU's cache; at 100k the
/// big groups leave CUs idle and 256 wins.
pub fn preferredGroupSize(n: u32) u32 {
    return if (n >= 1 << 19) 1024 else 256;
}

/// Bodies per submitted command buffer. At 16M Plummer bodies one body costs about 55 ns of GPU
/// time, so a chunk is ~60 ms: far under the driver's job watchdog even on a slower or shared GPU.
pub const chunk_bodies: u32 = 1 << 20;

/// Bodies per group of the group walk (k3/walk.zig walkGroup64, ref3.Tree.walkGroups); it is
/// also the workgroup size, and the results depend on it.
pub const group_bodies: u32 = 64;
/// Group mode submits w.chunk / chunk_group_div workgroups per command buffer. Workgroup w checks
/// one child of internal node (w - 1) / 2, and internal nodes are spread over the bodies like the
/// leaves (Plummer: one per 22 bodies), so a chunk is about 1.4M bodies.
const chunk_group_div: u32 = 8;

pub const Mode = enum {
    /// k3/walk.zig walk<gs>, ref3.Tree.walk.
    body,
    /// k3/walk.zig walkGroup64, ref3.Tree.walkGroups.
    group,
};

pub const Kick = struct {
    n: u32,
    g: f32,
    eps: f32,
    /// vel += kick * a (ignored with sim3_flag_store_accel).
    kick: f32,
    /// Length per position quantum: box_size / 2^32.
    box_scale: f32,
    flags: u32 = 0,
    /// Group mode: an upper bound on the tree's leaf count (0: n).
    leaves_max: u32 = 0,
};

/// `status` holds tree3's status words (tree3.statusBuffer); the per-body walk ignores it.
pub const Bindings = struct { pos: Buffer, mass: Buffer, nodes: Buffer, quads: Buffer, status: Buffer, vel: Buffer, amag: Buffer, pot: Buffer };

fn makePipeline(dev: *c.SDL_GPUDevice, entry: [*:0]const u8, gs: u32) ?*c.SDL_GPUComputePipeline {
    return c.SDL_CreateGPUComputePipeline(dev, &.{
        .code_size = walk_spv.len,
        .code = walk_spv.ptr,
        .entrypoint = entry,
        .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
        .num_readonly_storage_buffers = 5,
        .num_readwrite_storage_buffers = 3,
        .num_uniform_buffers = 1,
        .threadcount_x = gs,
        .threadcount_y = 1,
        .threadcount_z = 1,
    });
}

pub const Walk3 = struct {
    dev: *c.SDL_GPUDevice,
    /// Per-body walk pipelines; null where the device refused that workgroup size.
    pipes: [group_sizes.len]?*c.SDL_GPUComputePipeline,
    group_pipe: ?*c.SDL_GPUComputePipeline = null,
    mode: Mode = .group,
    /// Body mode workgroup size; 0 picks autoGroupSize(n).
    group_size: u32 = 0,
    chunk: u32 = chunk_bodies,
    /// Profiling (initEntry): replaces the pipeline of `mode`.
    extra: ?*c.SDL_GPUComputePipeline = null,
    extra_gs: u32 = 0,

    /// SDL_GPU exposes no device limits (maxComputeWorkGroupSize/Invocations), so each size is
    /// created on its own and the ones the driver accepts are kept. Vulkan guarantees 128
    /// invocations; larger sizes rely on the driver rejecting what it cannot run.
    pub fn init(dev: *c.SDL_GPUDevice) Error!Walk3 {
        var w: Walk3 = .{ .dev = dev, .pipes = @splat(null) };
        w.group_pipe = makePipeline(dev, "walkGroup64", group_bodies) orelse return error.GpuPipeline;
        errdefer c.SDL_ReleaseGPUComputePipeline(dev, w.group_pipe.?);
        var any = false;
        inline for (group_sizes, &w.pipes) |gs, *p| {
            p.* = makePipeline(dev, std.fmt.comptimePrint("walk{d}", .{gs}), gs);
            if (p.* == null) std.log.warn("walk3: workgroup size {d} unavailable: {s}", .{ gs, c.SDL_GetError() }) else any = true;
        }
        if (!any) return error.GpuPipeline;
        return w;
    }

    /// Profiling only: a Walk3 that runs entry point `entry` (workgroups of `gs`) as a `mode` walk.
    pub fn initEntry(dev: *c.SDL_GPUDevice, entry: [*:0]const u8, gs: u32, mode: Mode) Error!Walk3 {
        return .{ .dev = dev, .pipes = @splat(null), .mode = mode, .extra_gs = gs, .extra = makePipeline(dev, entry, gs) orelse return error.GpuPipeline };
    }

    pub fn deinit(w: *Walk3) void {
        for (w.pipes) |p| if (p) |q| c.SDL_ReleaseGPUComputePipeline(w.dev, q);
        if (w.group_pipe) |q| c.SDL_ReleaseGPUComputePipeline(w.dev, q);
        if (w.extra) |q| c.SDL_ReleaseGPUComputePipeline(w.dev, q);
        w.* = undefined;
    }

    pub fn available(w: *const Walk3, gs: u32) bool {
        for (group_sizes, w.pipes) |s, p| if (s == gs) return p != null;
        return false;
    }

    /// preferredGroupSize(n), or the largest available size below it, or the smallest available.
    pub fn autoGroupSize(w: *const Walk3, n: u32) u32 {
        const want = preferredGroupSize(n);
        var best: u32 = 0;
        for (group_sizes, w.pipes) |s, p| {
            if (p == null) continue;
            if (best == 0 or s <= want) best = s;
        }
        return best;
    }

    /// Records the kick of bodies [0, k.n) into `cmd0`, one chunk per command buffer (w.chunk
    /// bodies, or in group mode w.chunk / chunk_group_div workgroups). Every chunk but the last is
    /// submitted here without waiting (chunks write disjoint ranges and later passes are ordered
    /// after them by SDL_GPU's pass barriers); the returned command buffer holds the last chunk and
    /// is the caller's to submit.
    pub fn record(w: *Walk3, cmd0: *c.SDL_GPUCommandBuffer, b: Bindings, k: Kick) Error!*c.SDL_GPUCommandBuffer {
        const grouped = w.mode == .group;
        const gs = if (w.extra != null) w.extra_gs else if (grouped) group_bodies else if (w.group_size == 0) w.autoGroupSize(k.n) else w.group_size;
        const pipe = if (w.extra) |e| e else if (grouped) w.group_pipe orelse return error.OutOfRange else for (group_sizes, w.pipes) |s, p| {
            if (s == gs) break p orelse return error.OutOfRange;
        } else return error.OutOfRange;
        // Group mode: workgroups for the root and both children of every internal node, and at
        // least n / gs of them so that an overflowed tree (zero force for every body) is covered.
        const lmax: u64 = if (k.leaves_max == 0) k.n else @min(k.leaves_max, k.n);
        const units: u32 = if (grouped) @intCast(@max(@min(1 + 2 * lmax, std.math.maxInt(u32)), (k.n + gs - 1) / gs)) else k.n;
        const chunk = if (grouped) @max(w.chunk / chunk_group_div, 1) else w.chunk;
        var cmd = cmd0;
        var first: u32 = 0;
        while (true) {
            const count = @min(chunk, units - first);
            const p: L.WalkParams3 = .{
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
            c.SDL_PushGPUComputeUniformData(cmd, 0, &p, @sizeOf(L.WalkParams3));
            const rw = [3]c.SDL_GPUStorageBufferReadWriteBinding{ .{ .buffer = b.vel }, .{ .buffer = b.amag }, .{ .buffer = b.pot } };
            const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &rw, rw.len);
            c.SDL_BindGPUComputePipeline(cp, pipe);
            const ro = [5]?*c.SDL_GPUBuffer{ b.pos, b.mass, b.nodes, b.quads, b.status };
            c.SDL_BindGPUComputeStorageBuffers(cp, 0, &ro, ro.len);
            const groups = if (grouped) count else (count + gs - 1) / gs;
            c.SDL_DispatchGPUCompute(cp, @min(groups, dispatch_x), (groups + dispatch_x - 1) / dispatch_x, 1);
            c.SDL_EndGPUComputePass(cp);
            first += count;
            if (first >= units) return cmd;
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
    /// tree3's status words for this tree (leaf count, node count, no overflow).
    status: Buffer,
    n: u32,
    node_cap: u32,
    node_count: u32 = 0,
    leaves: u32 = 0,

    pub fn init(dev: *c.SDL_GPUDevice, n: u32, node_cap: u32) Error!GpuTree {
        var t: GpuTree = .{ .dev = dev, .pos = undefined, .mass = undefined, .nodes = undefined, .quads = undefined, .status = undefined, .n = n, .node_cap = node_cap };
        t.pos = try makeBuffer(dev, @as(u64, n) * @sizeOf(L.Pos3));
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.pos);
        t.mass = try makeBuffer(dev, @as(u64, n) * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.mass);
        t.nodes = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.Node));
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.nodes);
        t.quads = try makeBuffer(dev, @as(u64, node_cap) * @sizeOf(L.NodeQuad));
        errdefer c.SDL_ReleaseGPUBuffer(dev, t.quads);
        t.status = try makeBuffer(dev, 4 * (tree3.status_max_depth + 1));
        return t;
    }

    pub fn deinit(t: *GpuTree) void {
        for ([_]Buffer{ t.pos, t.mass, t.nodes, t.quads, t.status }) |b| c.SDL_ReleaseGPUBuffer(t.dev, b);
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
        t.leaves = tree.leaves;
        var st: [tree3.status_max_depth + 1]u32 = @splat(0);
        st[tree3.status_leaves] = tree.leaves;
        st[tree3.status_nodes] = t.node_count;
        try uploadBytes(t.dev, t.status, std.mem.sliceAsBytes(&st));
    }

    pub fn bindings(t: *const GpuTree, vel: Buffer, amag: Buffer, pot: Buffer) Bindings {
        return .{ .pos = t.pos, .mass = t.mass, .nodes = t.nodes, .quads = t.quads, .status = t.status, .vel = vel, .amag = amag, .pot = pot };
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
