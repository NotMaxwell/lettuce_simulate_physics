//! Host side of GPU mutual gravity (kernels and method: gravity_kernels.zig).
//! Needs named imports "c", "gpu_layout", "config", and "gravity.spv" (the compiled kernels).
//!
//! Per tick: bin balls into leaf cells (atomics), counting-sort them, build the multipole
//! pyramid bottom-up, local expansions top-down, then one evaluate pass adds g*a to `vel`.
//! Every step depends on the previous one, so each is its own compute pass (SDL inserts the
//! barrier between passes).
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const K = @import("gravity_shared.zig");

const spv = @embedFile("gravity.spv");

pub const Error = error{ GpuPipeline, GpuBuffer };

/// The shared ball buffers; pos/body need COMPUTE_STORAGE_READ, vel needs READ|WRITE.
pub const Balls = struct {
    pos: *c.SDL_GPUBuffer,
    body: *c.SDL_GPUBuffer,
    vel: *c.SDL_GPUBuffer,
};

const Kernel = enum { boundsPartial, boundsFinal, clearCounts, binBalls, scanBlocks, scanTotals, scatterBalls, leafMultipoles, upwardPass, downwardPass, evaluate };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;

/// Average balls per occupied leaf cell the leaf level aims for. Lower means more M2L work,
/// higher more pairwise work; 6-12 measured fastest in the harness at 10k-1M balls.
pub const target_per_leaf: f32 = 8;

/// Default softening eps^2 (world units^2) for `apply`; the game uses it outside galaxy mode.
pub const default_softening: f32 = K.softening;

/// Largest body count: the `sorted` buffer (16 B per body) must stay under SDL_GPU's 4 GiB buffer size.
pub const max_capacity: u32 = 0x0FFF_FFFF;

/// Solver scratch per body (keys + cell-sorted copy), on top of the caller's pos/body/vel.
pub const scratch_bytes_per_body: u32 = @sizeOf(K.Key) + @sizeOf(K.Sorted);

pub const Gravity = struct {
    device: *c.SDL_GPUDevice,
    capacity: u32,
    /// Word index of the GPU-computed tree box in `cells` (the last three words, plus one spare).
    box_word: u32,
    max_leaf_level: u32,
    pipes: [n_kernels]*c.SDL_GPUComputePipeline,
    cells: *c.SDL_GPUBuffer,
    keys: *c.SDL_GPUBuffer,
    sorted: *c.SDL_GPUBuffer,
    mpole: *c.SDL_GPUBuffer,
    local: *c.SDL_GPUBuffer,
    /// Leaf level chosen by the last apply (for the perf overlay).
    last_leaf_level: u32 = 0,

    pub fn init(device: *c.SDL_GPUDevice, capacity: u32) Error!Gravity {
        var g: Gravity = undefined;
        g.device = device;
        g.capacity = @min(capacity, max_capacity);
        g.max_leaf_level = @min(K.max_level, leafLevelFor(g.capacity, 1.0, K.max_level) + 1);
        g.last_leaf_level = 0;
        const names = @typeInfo(Kernel).@"enum".field_names;
        var infos: [names.len]c.SDL_GPUComputePipelineCreateInfo = undefined;
        inline for (names, &infos) |name, *info| {
            info.* = .{
                .code_size = spv.len,
                .code = spv.ptr,
                .entrypoint = name,
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = 2,
                .num_readwrite_storage_buffers = 6,
                .num_uniform_buffers = 1,
                .threadcount_x = L.group_size,
                .threadcount_y = 1,
                .threadcount_z = 1,
            };
        }
        var made: [names.len]?*c.SDL_GPUComputePipeline = undefined;
        if (!@import("gpu_sim").createComputePipelines(device, &infos, &made)) return error.GpuPipeline;
        errdefer for (g.pipes) |p| c.SDL_ReleaseGPUComputePipeline(device, p);
        for (&g.pipes, made) |*p, m| p.* = m.?;
        const leaf_cells = @as(u32, 1) << @intCast(2 * g.max_leaf_level);
        const pyramid = K.levelOffset(g.max_leaf_level + 1);
        g.box_word = K.cellsWords(leaf_cells);
        g.cells = try makeBuffer(device, (g.box_word + 4) * 4);
        errdefer c.SDL_ReleaseGPUBuffer(device, g.cells);
        g.keys = try makeBuffer(device, g.capacity * @sizeOf(K.Key));
        errdefer c.SDL_ReleaseGPUBuffer(device, g.keys);
        g.sorted = try makeBuffer(device, g.capacity * @sizeOf(K.Sorted));
        errdefer c.SDL_ReleaseGPUBuffer(device, g.sorted);
        g.mpole = try makeBuffer(device, pyramid * K.order * @sizeOf(L.Vec2));
        errdefer c.SDL_ReleaseGPUBuffer(device, g.mpole);
        g.local = try makeBuffer(device, pyramid * K.order * @sizeOf(L.Vec2));
        return g;
    }

    pub fn deinit(g: *Gravity) void {
        for (g.pipes) |p| c.SDL_ReleaseGPUComputePipeline(g.device, p);
        for ([_]*c.SDL_GPUBuffer{ g.cells, g.keys, g.sorted, g.mpole, g.local }) |b| c.SDL_ReleaseGPUBuffer(g.device, b);
    }

    /// Records the gravity passes into `cmd`: vel += g_scale * a for the first params.ball_count
    /// balls. `softening` is eps^2 in world units^2 (see gravity_shared.softening for the default).
    pub fn apply(g: *Gravity, cmd: *c.SDL_GPUCommandBuffer, params: L.SimParams, balls: Balls, g_scale: f32, softening: f32) void {
        const w = params.bounds_max_x - params.bounds_min_x;
        const h = params.bounds_max_y - params.bounds_min_y;
        const size = @max(w, h, 1.0);
        // The tree is square over the bounds; only the w*h part of it holds balls.
        const fill = @max(w, 1.0) * @max(h, 1.0) / (size * size);
        g.record(cmd, params.ball_count, fill, .{ params.bounds_min_x, params.bounds_min_y, size }, K.no_box, balls, g_scale, softening, false);
    }

    /// Like `apply`, but the tree box is recomputed on the GPU from the bodies' bounding box (two
    /// small reduction passes, no readback), so bodies may roam anywhere. Assumes the bodies fill
    /// the box (fill = 1) when picking the leaf level.
    pub fn applyAutoBox(g: *Gravity, cmd: *c.SDL_GPUCommandBuffer, n: u32, balls: Balls, g_scale: f32, softening: f32) void {
        g.record(cmd, n, 1.0, .{ 0, 0, 1 }, g.box_word, balls, g_scale, softening, false);
    }

    /// Records only the box reduction of `applyAutoBox` (for timing it).
    pub fn recordBoxOnly(g: *Gravity, cmd: *c.SDL_GPUCommandBuffer, n: u32, balls: Balls) void {
        g.record(cmd, n, 1.0, .{ 0, 0, 1 }, g.box_word, balls, 0, 0, true);
    }

    fn record(g: *Gravity, cmd: *c.SDL_GPUCommandBuffer, count: u32, fill: f32, box: [3]f32, box_word: u32, balls: Balls, g_scale: f32, softening: f32, box_only: bool) void {
        const n = @min(count, g.capacity);
        if (n < 2) return;
        const leaf = leafLevelFor(n, fill, g.max_leaf_level);
        g.last_leaf_level = leaf;
        var kp: K.Params = .{
            .n = n,
            .leaf_level = leaf,
            .level = 0,
            .origin_x = box[0],
            .origin_y = box[1],
            .root_size = box[2],
            .g_scale = g_scale,
            .softening = softening,
            .box_word = box_word,
        };
        const leaf_cells = @as(u32, 1) << @intCast(2 * leaf);
        const ro = [_]*c.SDL_GPUBuffer{ balls.pos, balls.body };
        const rw = [_]c.SDL_GPUStorageBufferReadWriteBinding{
            .{ .buffer = balls.vel },
            .{ .buffer = g.cells },
            .{ .buffer = g.keys },
            .{ .buffer = g.sorted },
            .{ .buffer = g.mpole },
            .{ .buffer = g.local },
        };
        const pass = struct {
            fn run(gg: *Gravity, cb: *c.SDL_GPUCommandBuffer, k: Kernel, p: *const K.Params, r: []const *c.SDL_GPUBuffer, b: []const c.SDL_GPUStorageBufferReadWriteBinding, threads: u32) void {
                c.SDL_PushGPUComputeUniformData(cb, 0, p, @sizeOf(K.Params));
                const cp = c.SDL_BeginGPUComputePass(cb, null, 0, b.ptr, @intCast(b.len));
                c.SDL_BindGPUComputePipeline(cp, gg.pipes[@backingInt(k)]);
                c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(r.ptr), @intCast(r.len));
                const groups = (threads + L.group_size - 1) / L.group_size;
                c.SDL_DispatchGPUCompute(cp, @min(groups, K.dispatch_x), (groups + K.dispatch_x - 1) / K.dispatch_x, 1);
                c.SDL_EndGPUComputePass(cp);
            }
        }.run;
        if (box_word != K.no_box) {
            pass(g, cmd, .boundsPartial, &kp, &ro, &rw, (n + K.bounds_per_thread - 1) / K.bounds_per_thread);
            pass(g, cmd, .boundsFinal, &kp, &ro, &rw, L.group_size);
        }
        if (box_only) return;
        pass(g, cmd, .clearCounts, &kp, &ro, &rw, leaf_cells);
        pass(g, cmd, .binBalls, &kp, &ro, &rw, n);
        pass(g, cmd, .scanBlocks, &kp, &ro, &rw, (leaf_cells + K.scan_block - 1) / K.scan_block * L.group_size);
        pass(g, cmd, .scanTotals, &kp, &ro, &rw, L.group_size);
        pass(g, cmd, .scatterBalls, &kp, &ro, &rw, n);
        pass(g, cmd, .leafMultipoles, &kp, &ro, &rw, leaf_cells);
        var lvl = leaf;
        while (lvl > K.min_level) {
            lvl -= 1;
            kp.level = lvl;
            pass(g, cmd, .upwardPass, &kp, &ro, &rw, @as(u32, 1) << @intCast(2 * lvl));
        }
        lvl = K.min_level;
        while (lvl <= leaf) : (lvl += 1) {
            kp.level = lvl;
            pass(g, cmd, .downwardPass, &kp, &ro, &rw, @as(u32, 1) << @intCast(2 * lvl));
        }
        pass(g, cmd, .evaluate, &kp, &ro, &rw, n);
    }
};

fn leafLevelFor(n: u32, fill: f32, max: u32) u32 {
    // 4^level * fill * target_per_leaf ~= n
    const want = @as(f32, @floatFromInt(n)) / (fill * target_per_leaf);
    const lvl: u32 = @intFromFloat(@max(@round(@log2(@max(want, 1.0)) / 2.0), 0.0));
    return std.math.clamp(lvl, K.min_level, max);
}

fn makeBuffer(device: *c.SDL_GPUDevice, size: u32) Error!*c.SDL_GPUBuffer {
    return c.SDL_CreateGPUBuffer(device, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = @max(size, 16),
    }) orelse error.GpuBuffer;
}
