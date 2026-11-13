//! Offline snapshot renderer: streams a snapshot to the GPU in chunks, splats body mass into a u32
//! grid with atomics, then tone-maps it to a PNG (see png.zig). Used by `zig build render-snap`.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const gs = @import("gpu_sim");
const sh = @import("shared.zig");
const snap = @import("snapshot.zig");
const png = @import("png.zig");

const spv = @embedFile("render_kernels.spv");

pub const Error = error{ GpuPipeline, GpuBuffer, GpuSubmit, FileIo, OutOfMemory, BadSize } || snap.Error;

pub const Options = struct {
    width: u32 = 1920,
    height: u32 = 1080,
    /// World point at the image centre; null = mean position of the bodies.
    center: ?[2]f32 = null,
    /// Half the shorter image side, in world units; null = 3x the rms radius about the centre.
    extent: ?f32 = null,
    log: bool = false,
};

pub const Result = struct {
    n: u64,
    cx: f32,
    cy: f32,
    extent: f32,
    render_ms: f64,
};

const Kernel = enum { clear, splat, findMax, tonemap };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;
/// Bodies per upload chunk.
const chunk_bodies: usize = 4 << 20;
/// Fixed-point mass weight of the average body: sums stay far below 2^32 even for 10^7 bodies in a pixel.
const mean_weight: f32 = 64;

pub const Renderer = struct {
    dev: *c.SDL_GPUDevice,
    width: u32,
    height: u32,
    pipes: [n_kernels]*c.SDL_GPUComputePipeline,
    grid: *c.SDL_GPUBuffer,
    maxv: *c.SDL_GPUBuffer,
    img: *c.SDL_GPUBuffer,
    pos: *c.SDL_GPUBuffer,
    mass: *c.SDL_GPUBuffer,
    up_pos: [2]*c.SDL_GPUTransferBuffer,
    up_mass: [2]*c.SDL_GPUTransferBuffer,
    down: *c.SDL_GPUTransferBuffer,

    pub fn init(dev: *c.SDL_GPUDevice, width: u32, height: u32) Error!Renderer {
        if (width == 0 or height == 0 or @as(u64, width) * height > (1 << 28)) return error.BadSize;
        var r: Renderer = undefined;
        r.dev = dev;
        r.width = width;
        r.height = height;
        const px: u32 = width * height * 4;
        r.grid = try buffer(dev, px);
        r.maxv = try buffer(dev, 16);
        r.img = try buffer(dev, px);
        r.pos = try buffer(dev, chunk_bodies * @sizeOf(L.Vec2));
        r.mass = try buffer(dev, chunk_bodies * 4);
        for (0..2) |i| {
            r.up_pos[i] = try transfer(dev, c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, chunk_bodies * @sizeOf(L.Vec2));
            r.up_mass[i] = try transfer(dev, c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, chunk_bodies * 4);
        }
        r.down = try transfer(dev, c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, px);
        const Counts = struct { ro: u32, rw: u32 };
        const counts = [n_kernels]Counts{ .{ .ro = 0, .rw = 2 }, .{ .ro = 2, .rw = 1 }, .{ .ro = 0, .rw = 2 }, .{ .ro = 0, .rw = 3 } };
        var infos: [n_kernels]c.SDL_GPUComputePipelineCreateInfo = undefined;
        inline for (@typeInfo(Kernel).@"enum".field_names, &infos, counts) |name, *info, cnt| {
            info.* = .{
                .code_size = spv.len,
                .code = spv.ptr,
                .entrypoint = name,
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = cnt.ro,
                .num_readwrite_storage_buffers = cnt.rw,
                .num_uniform_buffers = 1,
                .threadcount_x = L.group_size,
                .threadcount_y = 1,
                .threadcount_z = 1,
            };
        }
        var made: [n_kernels]?*c.SDL_GPUComputePipeline = undefined;
        if (!gs.createComputePipelines(dev, &infos, &made)) return error.GpuPipeline;
        for (&r.pipes, made) |*p, m| p.* = m.?;
        return r;
    }

    pub fn deinit(r: *Renderer) void {
        for (r.pipes) |p| c.SDL_ReleaseGPUComputePipeline(r.dev, p);
        for (r.up_pos ++ r.up_mass ++ .{r.down}) |t| c.SDL_ReleaseGPUTransferBuffer(r.dev, t);
        for ([_]*c.SDL_GPUBuffer{ r.grid, r.maxv, r.img, r.pos, r.mass }) |b| c.SDL_ReleaseGPUBuffer(r.dev, b);
        r.* = undefined;
    }

    fn dispatch(r: *Renderer, cmd: *c.SDL_GPUCommandBuffer, k: Kernel, p: sh.RenderParams, threads: u32) void {
        c.SDL_PushGPUComputeUniformData(cmd, 0, &p, @sizeOf(sh.RenderParams));
        const rw_all = [_]c.SDL_GPUStorageBufferReadWriteBinding{ .{ .buffer = r.grid }, .{ .buffer = r.maxv }, .{ .buffer = r.img } };
        const rw_n: u32 = switch (k) {
            .splat => 1,
            .tonemap => 3,
            else => 2,
        };
        const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &rw_all, rw_n);
        c.SDL_BindGPUComputePipeline(cp, r.pipes[@backingInt(k)]);
        if (k == .splat) {
            const ro = [_]*c.SDL_GPUBuffer{ r.pos, r.mass };
            c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(&ro), 2);
        }
        const groups = (threads + L.group_size - 1) / L.group_size;
        c.SDL_DispatchGPUCompute(cp, @min(groups, sh.dispatch_x), (groups + sh.dispatch_x - 1) / sh.dispatch_x, 1);
        c.SDL_EndGPUComputePass(cp);
    }

    /// Renders `snapshot` to `png_path`.
    pub fn render(r: *Renderer, gpa: std.mem.Allocator, snapshot: [:0]const u8, png_path: [:0]const u8, o: Options) Error!Result {
        const io = try snap.open(snapshot);
        defer _ = c.SDL_CloseIO(io);
        const h = try snap.readHeader(io);
        const t0 = c.SDL_GetPerformanceCounter();

        const scratch = try gpa.alloc(f32, chunk_bodies * 3);
        defer gpa.free(scratch);
        const xs = scratch[0..chunk_bodies];
        const ys = scratch[chunk_bodies..][0..chunk_bodies];
        const ms = scratch[2 * chunk_bodies ..][0..chunk_bodies];

        var sum_m: f64 = 0;
        var sx: f64 = 0;
        var sy: f64 = 0;
        var sxx: f64 = 0;
        var at: u64 = 0;
        const need_pos = o.center == null or o.extent == null;
        while (at < h.n) : (at += chunk_bodies) {
            const m: usize = @intCast(@min(chunk_bodies, h.n - at));
            try snap.readField(io, &h, .mass, at, ms[0..m]);
            for (ms[0..m]) |v| sum_m += v;
            if (need_pos) {
                try snap.readField(io, &h, .pos_x, at, xs[0..m]);
                try snap.readField(io, &h, .pos_y, at, ys[0..m]);
                for (xs[0..m], ys[0..m]) |x, y| {
                    sx += x;
                    sy += y;
                    sxx += @as(f64, x) * x + @as(f64, y) * y;
                }
            }
        }
        const nf: f64 = @floatFromInt(h.n);
        const mean_x: f32 = @floatCast(sx / nf);
        const mean_y: f32 = @floatCast(sy / nf);
        const cx = if (o.center) |ctr| ctr[0] else mean_x;
        const cy = if (o.center) |ctr| ctr[1] else mean_y;
        const rms = @sqrt(@max(sxx / nf - (sx / nf) * (sx / nf) - (sy / nf) * (sy / nf), 0));
        const extent: f32 = o.extent orelse @floatCast(@max(3 * rms, 1e-3));
        const mean_mass = sum_m / nf;
        const base: sh.RenderParams = .{
            .count = 0,
            .width = r.width,
            .height = r.height,
            .log_scale = @intFromBool(o.log),
            .cx = cx,
            .cy = cy,
            .scale = @as(f32, @floatFromInt(@min(r.width, r.height))) / (2 * extent),
            .weight_scale = if (mean_mass > 0) mean_weight / @as(f32, @floatCast(mean_mass)) else 1,
        };

        var cmd = c.SDL_AcquireGPUCommandBuffer(r.dev) orelse return error.GpuSubmit;
        r.dispatch(cmd, .clear, base, r.width * r.height);
        var fences: [2]?*c.SDL_GPUFence = .{ null, null };
        var k: usize = 0;
        at = 0;
        while (at < h.n) : (at += chunk_bodies) {
            const m: usize = @intCast(@min(chunk_bodies, h.n - at));
            const set = k % 2;
            if (fences[set]) |f| waitRelease(r.dev, f);
            fences[set] = null;
            try snap.readField(io, &h, .pos_x, at, xs[0..m]);
            try snap.readField(io, &h, .pos_y, at, ys[0..m]);
            try snap.readField(io, &h, .mass, at, ms[0..m]);
            const pm: [*]L.Vec2 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(r.dev, r.up_pos[set], false) orelse return error.GpuBuffer));
            for (pm[0..m], xs[0..m], ys[0..m]) |*o_, x, y| o_.* = .{ .x = x, .y = y };
            c.SDL_UnmapGPUTransferBuffer(r.dev, r.up_pos[set]);
            const mm: [*]f32 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(r.dev, r.up_mass[set], false) orelse return error.GpuBuffer));
            @memcpy(mm[0..m], ms[0..m]);
            c.SDL_UnmapGPUTransferBuffer(r.dev, r.up_mass[set]);
            if (k > 0) cmd = c.SDL_AcquireGPUCommandBuffer(r.dev) orelse return error.GpuSubmit;
            const cp = c.SDL_BeginGPUCopyPass(cmd);
            c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = r.up_pos[set] }, &.{ .buffer = r.pos, .offset = 0, .size = @intCast(m * @sizeOf(L.Vec2)) }, false);
            c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = r.up_mass[set] }, &.{ .buffer = r.mass, .offset = 0, .size = @intCast(m * 4) }, false);
            c.SDL_EndGPUCopyPass(cp);
            var p = base;
            p.count = @intCast(m);
            r.dispatch(cmd, .splat, p, @intCast(m));
            fences[set] = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSubmit;
            k += 1;
        }
        for (&fences) |*f| if (f.*) |ff| waitRelease(r.dev, ff);

        cmd = c.SDL_AcquireGPUCommandBuffer(r.dev) orelse return error.GpuSubmit;
        r.dispatch(cmd, .findMax, base, r.width * r.height);
        r.dispatch(cmd, .tonemap, base, r.width * r.height);
        const bytes: u32 = r.width * r.height * 4;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = r.img, .size = bytes }, &.{ .transfer_buffer = r.down, .offset = 0 });
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(r.dev, cmd);
        const px: [*]const u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(r.dev, r.down, false) orelse return error.GpuBuffer);
        defer c.SDL_UnmapGPUTransferBuffer(r.dev, r.down);
        const render_ms = msSince(t0);
        const file = try png.encodeRgba(gpa, r.width, r.height, px[0..bytes]);
        defer gpa.free(file);
        if (!c.SDL_SaveFile(png_path.ptr, file.ptr, file.len)) return error.FileIo;
        return .{ .n = h.n, .cx = cx, .cy = cy, .extent = extent, .render_ms = render_ms };
    }
};

fn waitRelease(dev: *c.SDL_GPUDevice, fence: *c.SDL_GPUFence) void {
    var f = fence;
    _ = c.SDL_WaitForGPUFences(dev, true, &f, 1);
    c.SDL_ReleaseGPUFence(dev, f);
}

fn submitWait(dev: *c.SDL_GPUDevice, cmd: *c.SDL_GPUCommandBuffer) Error!void {
    const f = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSubmit;
    waitRelease(dev, f);
}

fn msSince(t0: u64) f64 {
    return @as(f64, @floatFromInt(c.SDL_GetPerformanceCounter() - t0)) * 1000.0 / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency()));
}

fn buffer(dev: *c.SDL_GPUDevice, size: usize) Error!*c.SDL_GPUBuffer {
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = @intCast(@max(size, 16)),
    }) orelse error.GpuBuffer;
}

fn transfer(dev: *c.SDL_GPUDevice, usage: c.SDL_GPUTransferBufferUsage, size: usize) Error!*c.SDL_GPUTransferBuffer {
    return c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = usage, .size = @intCast(size) }) orelse error.GpuBuffer;
}
