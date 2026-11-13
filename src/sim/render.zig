//! Offline snapshot renderer: streams a snapshot to the GPU in chunks, splats body mass into a u32
//! grid with atomics, then tone-maps it to a PNG (see png.zig). Used by `zig build render-snap`.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const gs = @import("gpu_sim");
const sh = @import("render_params.zig");
const shared = @import("shared.zig");
const snap = @import("snapshot.zig");
const png = @import("png.zig");

/// Three 32-bit words per body in the GPU position buffer (2D uses two).
const pos_bytes = 12;

const spv = @embedFile("render_kernels.spv");

pub const Error = error{ GpuPipeline, GpuBuffer, GpuSubmit, FileIo, OutOfMemory, BadSize } || snap.Error;

/// Orthographic view of a 3D snapshot (2D snapshots ignore it). Screen x is `right`, screen y is `up`,
/// depth is right x up. `.angles` starts from the xy view, turns it by yaw around the y axis, then
/// pitches it around the turned x axis; (0, 0) is xy.
pub const View = union(enum) {
    xy,
    xz,
    yz,
    angles: struct { yaw: f32, pitch: f32 },

    pub const Basis = struct { right: [3]f32, up: [3]f32, depth: [3]f32 };

    pub fn parse(text: []const u8) error{BadView}!View {
        if (std.mem.eql(u8, text, "xy")) return .xy;
        if (std.mem.eql(u8, text, "xz")) return .xz;
        if (std.mem.eql(u8, text, "yz")) return .yz;
        const comma = std.mem.indexOfScalar(u8, text, ',') orelse return error.BadView;
        const yaw = std.fmt.parseFloat(f32, text[0..comma]) catch return error.BadView;
        const pitch = std.fmt.parseFloat(f32, text[comma + 1 ..]) catch return error.BadView;
        return .{ .angles = .{ .yaw = yaw, .pitch = pitch } };
    }

    pub fn basis(v: View) Basis {
        switch (v) {
            .xy => return .{ .right = .{ 1, 0, 0 }, .up = .{ 0, 1, 0 }, .depth = .{ 0, 0, 1 } },
            .xz => return .{ .right = .{ 1, 0, 0 }, .up = .{ 0, 0, 1 }, .depth = .{ 0, -1, 0 } },
            .yz => return .{ .right = .{ 0, 1, 0 }, .up = .{ 0, 0, 1 }, .depth = .{ 1, 0, 0 } },
            .angles => |a| {
                const yaw = std.math.degreesToRadians(a.yaw);
                const pitch = std.math.degreesToRadians(a.pitch);
                const sy, const cy = .{ @sin(yaw), @cos(yaw) };
                const sp, const cp = .{ @sin(pitch), @cos(pitch) };
                // yawed frame: right (cy, 0, -sy), up (0, 1, 0), depth (sy, 0, cy)
                return .{
                    .right = .{ cy, 0, -sy },
                    .up = .{ sp * sy, cp, sp * cy },
                    .depth = .{ cp * sy, -sp, cp * cy },
                };
            },
        }
    }
};

pub const Options = struct {
    width: u32 = 1920,
    height: u32 = 1080,
    /// World point at the image centre; null = mean position of the bodies. A 2D snapshot uses x, y.
    center: ?[3]f32 = null,
    /// Half the shorter image side, in world units; null = 1.05x the 99th percentile of the projected, sampled distance from the centre.
    extent: ?f32 = null,
    log: bool = false,
    view: View = .xy,
    /// Keep only bodies within +-slab/2 of the view plane through the centre (3D snapshots).
    slab: ?f32 = null,
};

pub const Result = struct {
    n: u64,
    cx: f32,
    cy: f32,
    cz: f32,
    extent: f32,
    render_ms: f64,
};

const Kernel = enum { clear, splat, findMax, tonemap };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;
/// Bodies per upload chunk.
const chunk_bodies: usize = 4 << 20;
/// Fixed-point mass weight of the average body: sums stay far below 2^32 even for 10^7 bodies in a pixel.
const mean_weight: f32 = 64;
/// Bodies sampled to pick the automatic extent.
const max_samples: u64 = 1 << 20;

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
        r.pos = try buffer(dev, chunk_bodies * pos_bytes);
        r.mass = try buffer(dev, chunk_bodies * 4);
        for (0..2) |i| {
            r.up_pos[i] = try transfer(dev, c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, chunk_bodies * pos_bytes);
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
        c.SDL_DispatchGPUCompute(cp, @min(groups, shared.dispatch_x), (groups + shared.dispatch_x - 1) / shared.dispatch_x, 1);
        c.SDL_EndGPUComputePass(cp);
    }

    /// Renders `snapshot` to `png_path`.
    pub fn render(r: *Renderer, gpa: std.mem.Allocator, snapshot: [:0]const u8, png_path: [:0]const u8, o: Options) Error!Result {
        const io = try snap.open(snapshot);
        defer _ = c.SDL_CloseIO(io);
        const h = try snap.readHeader(io);
        const t0 = c.SDL_GetPerformanceCounter();
        const is3 = h.dim == 3;
        const mass_id: u32 = if (is3) @backingInt(snap.Field3.mass) else @backingInt(snap.Field.mass);
        const uniform = is3 and h.hasUniformMass();
        const unit: f64 = if (is3) h.box_size / 4294967296.0 else 0;
        const view: View = if (is3) o.view else .xy;
        const basis = view.basis();

        const scratch = try gpa.alloc(u32, chunk_bodies * 4);
        defer gpa.free(scratch);
        const ws: [3][]u32 = .{ scratch[0..chunk_bodies], scratch[chunk_bodies..][0..chunk_bodies], scratch[2 * chunk_bodies ..][0..chunk_bodies] };
        const ms: []f32 = @ptrCast(scratch[3 * chunk_bodies ..][0..chunk_bodies]);

        var sum_m: f64 = 0;
        var sum: [3]f64 = @splat(0);
        const stride: u64 = @max(1, h.n / max_samples);
        var samples: std.ArrayList([3]f32) = .empty;
        defer samples.deinit(gpa);
        var at: u64 = 0;
        const need_pos = o.center == null or o.extent == null;
        const dims: usize = if (is3) 3 else 2;
        while (at < h.n) : (at += chunk_bodies) {
            const m: usize = @intCast(@min(chunk_bodies, h.n - at));
            if (!uniform) {
                try snap.readWords(io, &h, mass_id, at, @ptrCast(ms[0..m]));
                for (ms[0..m]) |v| sum_m += v;
            }
            if (need_pos) {
                for (0..dims) |d| try snap.readWords(io, &h, @intCast(d), at, ws[d][0..m]);
                for (0..m) |j| {
                    for (0..dims) |d| sum[d] += worldCoord(is3, unit, ws[d][j]);
                }
                var j: usize = @intCast((stride - at % stride) % stride);
                while (j < m) : (j += stride) {
                    var p: [3]f32 = .{ 0, 0, 0 };
                    for (0..dims) |d| p[d] = @floatCast(worldCoord(is3, unit, ws[d][j]));
                    try samples.append(gpa, p);
                }
            }
        }
        const nf: f64 = @floatFromInt(h.n);
        var ctr: [3]f32 = .{ 0, 0, 0 };
        for (0..3) |d| ctr[d] = if (o.center) |v| v[d] else @floatCast(sum[d] / nf);
        const slab_half: f32 = if (is3) if (o.slab) |t| 0.5 * @max(t, 0) else 0 else 0;
        const extent: f32 = o.extent orelse try percentileRadius(gpa, samples.items, ctr, basis, slab_half);
        const mean_mass: f64 = if (uniform) h.uniform_mass else sum_m / nf;
        const base: sh.RenderParams = .{
            .count = 0,
            .width = r.width,
            .height = r.height,
            .log_scale = @intFromBool(o.log),
            .mode = if (is3) sh.mode_xyz_u32 else sh.mode_xy_f32,
            .cx = ctr[0],
            .cy = ctr[1],
            .cz = ctr[2],
            .scale = @as(f32, @floatFromInt(@min(r.width, r.height))) / (2 * extent),
            .weight_scale = if (mean_mass > 0) mean_weight / @as(f32, @floatCast(mean_mass)) else 1,
            .unit = @floatCast(unit),
            .uniform_mass = if (uniform) h.uniform_mass else 0,
            .slab_half = slab_half,
            .rx = basis.right[0],
            .ry = basis.right[1],
            .rz = basis.right[2],
            .ux = basis.up[0],
            .uy = basis.up[1],
            .uz = basis.up[2],
            .fx = basis.depth[0],
            .fy = basis.depth[1],
            .fz = basis.depth[2],
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
            for (0..dims) |d| try snap.readWords(io, &h, @intCast(d), at, ws[d][0..m]);
            if (!uniform) try snap.readWords(io, &h, mass_id, at, @ptrCast(ms[0..m]));
            const pm: [*]u32 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(r.dev, r.up_pos[set], false) orelse return error.GpuBuffer));
            for (0..m) |j| {
                for (0..dims) |d| pm[j * dims + d] = ws[d][j];
            }
            c.SDL_UnmapGPUTransferBuffer(r.dev, r.up_pos[set]);
            const mm: [*]f32 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(r.dev, r.up_mass[set], false) orelse return error.GpuBuffer));
            if (!uniform) @memcpy(mm[0..m], ms[0..m]);
            c.SDL_UnmapGPUTransferBuffer(r.dev, r.up_mass[set]);
            if (k > 0) cmd = c.SDL_AcquireGPUCommandBuffer(r.dev) orelse return error.GpuSubmit;
            const cp = c.SDL_BeginGPUCopyPass(cmd);
            c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = r.up_pos[set] }, &.{ .buffer = r.pos, .offset = 0, .size = @intCast(m * dims * 4) }, false);
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
        return .{ .n = h.n, .cx = ctr[0], .cy = ctr[1], .cz = ctr[2], .extent = extent, .render_ms = render_ms };
    }
};

fn worldCoord(is3: bool, unit: f64, word: u32) f64 {
    if (!is3) return @as(f32, @bitCast(word));
    return (@as(f64, @floatFromInt(word)) - 2147483648.0) * unit;
}

/// 99th percentile of the distance from the centre in the image plane, over the bodies the slab keeps.
fn percentileRadius(gpa: std.mem.Allocator, pts: []const [3]f32, ctr: [3]f32, b: View.Basis, slab_half: f32) Error!f32 {
    const r = try gpa.alloc(f32, pts.len);
    defer gpa.free(r);
    var n: usize = 0;
    for (pts) |p| {
        const d: [3]f32 = .{ p[0] - ctr[0], p[1] - ctr[1], p[2] - ctr[2] };
        if (slab_half > 0 and @abs(dot(b.depth, d)) > slab_half) continue;
        r[n] = std.math.hypot(dot(b.right, d), dot(b.up, d));
        n += 1;
    }
    if (n == 0) return 1;
    std.mem.sort(f32, r[0..n], {}, std.sort.asc(f32));
    const p99 = r[(n - 1) * 99 / 100];
    return @max(1.05 * p99, 1e-3);
}

fn dot(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

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
