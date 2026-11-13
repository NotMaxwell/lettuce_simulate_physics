//! Headless matter simulation on the GPU: open-space gravity (the shared 2D FMM) with a
//! kick-drift-kick leapfrog, plus energy / momentum / angular-momentum diagnostics by reduction.
//! No windows, walls, collisions or merging. See main.zig for the CLI.
//!
//! Time stepping. One tick is kick(dt/2) drift(dt) kick(dt/2), but the closing kick of tick k and the
//! opening kick of tick k+1 use the same accelerations, so ticks run as one full kick plus a drift:
//! between ticks `vel` is the half-step velocity v(k-1/2) and `half_pending` is false. `measure`
//! gives synchronous velocities by applying the missing half kick (half_pending = true), which the
//! next tick then skips. A fresh simulation starts that way, with synchronous initial velocities.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const galaxy = @import("galaxy");
const gs = @import("gpu_sim");
const gravity = @import("gravity");
const sh = @import("shared.zig");

const spv = @embedFile("sim_kernels.spv");

pub const Error = error{ GpuDevice, GpuPipeline, GpuBuffer, GpuSubmit, OutOfRange };

const Kernel = enum { drift, diagPartial, potential };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;
const Buffer = *c.SDL_GPUBuffer;

pub const Config = struct {
    n: u32,
    /// Gravity strength: acceleration = g * sum m_j d / (d^2 + softening).
    g: f32,
    /// Plummer eps^2 (world units^2).
    softening: f32 = 4,
    dt: f32 = 1,
    /// Bodies whose potential is summed against all others (0 = automatic, see `autoSamples`).
    pe_samples: u32 = 0,
};

pub const Diag = struct {
    ke: f64,
    pe: f64,
    px: f64,
    py: f64,
    lz: f64,
    mass: f64,
    /// Potential-energy samples used (== n when exact).
    samples: u32,

    pub fn e(d: Diag) f64 {
        return d.ke + d.pe;
    }
    pub fn p(d: Diag) f64 {
        return @sqrt(d.px * d.px + d.py * d.py);
    }
};

/// Mean ms per pass from blocking samples (each includes one submit + wait round trip).
pub const PassTimes = struct { gravity_fixed_box: f64, box: f64, gravity: f64, drift: f64, diag: f64, potential: f64 };

pub const Sim = struct {
    dev: *c.SDL_GPUDevice,
    cfg: Config,
    pos: Buffer,
    vel: Buffer,
    body: Buffer,
    grav: gravity.Gravity,
    pipes: [n_kernels]*c.SDL_GPUComputePipeline,
    diag_buf: Buffer,
    pot_buf: Buffer,
    download: *c.SDL_GPUTransferBuffer,
    diag_groups: u32,
    pot_groups: u32,
    samples: u32,
    chunks: u32,
    chunk_len: u32,
    stride: u32,
    tick_no: u64 = 0,
    half_pending: bool = true,
    /// Fixed tree box for the bench comparison only: centre 0, half side `box_half`.
    box_half: f32 = 0,

    pub fn init(dev: *c.SDL_GPUDevice, cfg: Config) Error!Sim {
        if (cfg.n < 2 or cfg.n > gravity.max_capacity) return error.OutOfRange;
        var s: Sim = undefined;
        s.dev = dev;
        s.cfg = cfg;
        s.tick_no = 0;
        s.half_pending = true;
        s.box_half = 0;
        const n = cfg.n;
        s.pos = try makeBuffer(dev, @as(u64, n) * @sizeOf(L.Vec2));
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.pos);
        s.vel = try makeBuffer(dev, @as(u64, n) * @sizeOf(L.Vec2));
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.vel);
        s.body = try makeBuffer(dev, @as(u64, n) * @sizeOf(L.BallBody));
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.body);
        s.grav = gravity.Gravity.init(dev, n) catch return error.GpuPipeline;
        errdefer s.grav.deinit();

        const Counts = struct { ro: u32, rw: u32 };
        const counts = [n_kernels]Counts{ .{ .ro = 0, .rw = 2 }, .{ .ro = 2, .rw = 2 }, .{ .ro = 2, .rw = 1 } };
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
        for (&s.pipes, made) |*p, m| p.* = m.?;
        errdefer for (s.pipes) |p| c.SDL_ReleaseGPUComputePipeline(dev, p);

        s.diag_groups = ((n + sh.diag_per_thread - 1) / sh.diag_per_thread + L.group_size - 1) / L.group_size;
        s.samples = if (cfg.pe_samples == 0) autoSamples(n) else @min(cfg.pe_samples, n);
        s.stride = n / s.samples;
        s.chunks = @min(@max(1, sh.potential_threads / s.samples), n);
        s.chunk_len = (n + s.chunks - 1) / s.chunks;
        s.chunks = (n + s.chunk_len - 1) / s.chunk_len;
        s.pot_groups = (s.samples * s.chunks + L.group_size - 1) / L.group_size;
        s.diag_buf = try makeBuffer(dev, @as(u64, s.diag_groups) * sh.diag_stride * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.diag_buf);
        s.pot_buf = try makeBuffer(dev, @as(u64, s.pot_groups) * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.pot_buf);
        s.download = c.SDL_CreateGPUTransferBuffer(dev, &.{
            .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
            .size = s.diag_groups * sh.diag_stride * 4 + s.pot_groups * 4,
        }) orelse return error.GpuBuffer;
        return s;
    }

    pub fn deinit(s: *Sim) void {
        c.SDL_ReleaseGPUTransferBuffer(s.dev, s.download);
        for ([_]Buffer{ s.pos, s.vel, s.body, s.diag_buf, s.pot_buf }) |b| c.SDL_ReleaseGPUBuffer(s.dev, b);
        for (s.pipes) |p| c.SDL_ReleaseGPUComputePipeline(s.dev, p);
        s.grav.deinit();
        s.* = undefined;
    }

    /// Resident bytes per body: state buffers plus the gravity solver's per-body scratch.
    pub fn bytesPerBody(_: *const Sim) u64 {
        return sh.bytes_per_body_state + gravity.scratch_bytes_per_body;
    }

    /// Blocking upload of the initial conditions. Synchronous velocities; resets the clock.
    pub fn upload(s: *Sim, bodies: []const galaxy.Body) Error!void {
        if (bodies.len != s.cfg.n) return error.OutOfRange;
        const chunk: usize = 1 << 20;
        const tb = c.SDL_CreateGPUTransferBuffer(s.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = @intCast(chunk * @sizeOf(L.BallBody)) }) orelse return error.GpuBuffer;
        defer c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);
        var at: usize = 0;
        while (at < bodies.len) : (at += chunk) {
            const part = bodies[at..@min(at + chunk, bodies.len)];
            const targets = [_]struct { buf: Buffer, size: usize }{
                .{ .buf = s.pos, .size = @sizeOf(L.Vec2) },
                .{ .buf = s.vel, .size = @sizeOf(L.Vec2) },
                .{ .buf = s.body, .size = @sizeOf(L.BallBody) },
            };
            for (targets, 0..) |t, which| {
                const m: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuBuffer);
                switch (which) {
                    0 => for (@as([*]L.Vec2, @ptrCast(@alignCast(m)))[0..part.len], part) |*o, b| {
                        o.* = .{ .x = b.x, .y = b.y };
                    },
                    1 => for (@as([*]L.Vec2, @ptrCast(@alignCast(m)))[0..part.len], part) |*o, b| {
                        o.* = .{ .x = b.vx, .y = b.vy };
                    },
                    else => for (@as([*]L.BallBody, @ptrCast(@alignCast(m)))[0..part.len], part) |*o, b| {
                        o.* = .{ .radius = 0, .mass = b.mass, .retention = 0, .friction = 0 };
                    },
                }
                c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
                const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                const cp = c.SDL_BeginGPUCopyPass(cmd);
                c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = tb }, &.{ .buffer = t.buf, .offset = @intCast(at * t.size), .size = @intCast(part.len * t.size) }, false);
                c.SDL_EndGPUCopyPass(cp);
                try submitWait(s.dev, cmd);
            }
        }
        s.tick_no = 0;
        s.half_pending = true;
    }

    /// Sets the box half side `recordGravityFixedBox` uses (bench comparison only).
    pub fn setFixedBox(s: *Sim, half: f32) void {
        s.box_half = half;
    }

    fn kick(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, factor: f32) void {
        const scale = s.cfg.g * s.cfg.dt * factor;
        s.grav.applyAutoBox(cmd, s.cfg.n, .{ .pos = s.pos, .body = s.body, .vel = s.vel }, scale, s.cfg.softening);
    }

    fn recordDrift(s: *Sim, cmd: *c.SDL_GPUCommandBuffer) void {
        const rw = [_]c.SDL_GPUStorageBufferReadWriteBinding{ .{ .buffer = s.pos }, .{ .buffer = s.vel } };
        s.pass(cmd, .drift, &.{}, &rw, s.cfg.n);
    }

    /// Records one tick: the kick that takes velocities to the next half step, then the drift.
    pub fn recordTick(s: *Sim, cmd: *c.SDL_GPUCommandBuffer) void {
        s.kick(cmd, if (s.half_pending) 0.5 else 1.0);
        s.half_pending = false;
        s.recordDrift(cmd);
        s.tick_no += 1;
    }

    /// Runs `ticks` ticks, one command buffer each, keeping a few in flight; returns when done.
    pub fn run(s: *Sim, ticks: u64) Error!void {
        // A tick of 100M+ bodies takes seconds; queueing several risks the driver's ~10 s job watchdog.
        const depth: usize = if (s.cfg.n >= 8_000_000) 1 else 4;
        var fences: [4]?*c.SDL_GPUFence = @splat(null);
        for (0..ticks) |k| {
            const slot = k % depth;
            if (fences[slot]) |f| waitRelease(s.dev, f);
            const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            s.recordTick(cmd);
            fences[slot] = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSubmit;
        }
        for (&fences) |*f| if (f.*) |ff| {
            waitRelease(s.dev, ff);
            f.* = null;
        };
    }

    fn recordDiag(s: *Sim, cmd: *c.SDL_GPUCommandBuffer) void {
        const n = s.cfg.n;
        const ro = [_]Buffer{ s.pos, s.body };
        s.pass(cmd, .diagPartial, &ro, &.{ .{ .buffer = s.vel }, .{ .buffer = s.diag_buf } }, (n + sh.diag_per_thread - 1) / sh.diag_per_thread);
        s.pass(cmd, .potential, &ro, &.{.{ .buffer = s.pot_buf }}, s.samples * s.chunks);
    }

    /// Diagnostics at synchronous velocities (blocking). Applies the pending half kick if needed.
    pub fn measure(s: *Sim) Error!Diag {
        const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
        if (!s.half_pending) {
            s.kick(cmd, 0.5);
            s.half_pending = true;
        }
        s.recordDiag(cmd);
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        const diag_bytes = s.diag_groups * sh.diag_stride * 4;
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.diag_buf, .size = diag_bytes }, &.{ .transfer_buffer = s.download, .offset = 0 });
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.pot_buf, .size = s.pot_groups * 4 }, &.{ .transfer_buffer = s.download, .offset = diag_bytes });
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(s.dev, cmd);
        const m: [*]const f32 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(s.dev, s.download, false) orelse return error.GpuBuffer));
        defer c.SDL_UnmapGPUTransferBuffer(s.dev, s.download);
        var d: Diag = .{ .ke = 0, .pe = 0, .px = 0, .py = 0, .lz = 0, .mass = 0, .samples = s.samples };
        for (0..s.diag_groups) |gi| {
            const row = m[gi * sh.diag_stride ..][0..5];
            d.ke += row[0];
            d.px += row[1];
            d.py += row[2];
            d.lz += row[3];
            d.mass += row[4];
        }
        var pot: f64 = 0;
        for (m[s.diag_groups * sh.diag_stride ..][0..s.pot_groups]) |v| pot += v;
        const weight = @as(f64, @floatFromInt(s.cfg.n)) / @as(f64, @floatFromInt(s.samples));
        d.pe = 0.5 * @as(f64, s.cfg.g) * weight * pot;
        return d;
    }

    /// Blocking per-pass timing, mean of `reps` samples after one warm-up.
    pub fn timePasses(s: *Sim, reps: u32) Error!PassTimes {
        var t: PassTimes = undefined;
        const Which = enum { fixed, box, gravity, drift, diag, potential };
        inline for (@typeInfo(Which).@"enum".field_names, 0..) |name, wi| {
            var total: f64 = 0;
            for (0..reps + 1) |k| {
                const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                const t0 = c.SDL_GetPerformanceCounter();
                const balls: gravity.Balls = .{ .pos = s.pos, .body = s.body, .vel = s.vel };
                const scale = s.cfg.g * s.cfg.dt;
                switch (@as(Which, @fromBackingInt(@intCast(wi)))) {
                    .fixed => {
                        var p = std.mem.zeroes(L.SimParams);
                        p.ball_count = s.cfg.n;
                        p.bounds_min_x = -s.box_half;
                        p.bounds_min_y = -s.box_half;
                        p.bounds_max_x = s.box_half;
                        p.bounds_max_y = s.box_half;
                        s.grav.apply(cmd, p, balls, scale, s.cfg.softening);
                    },
                    .box => s.grav.recordBoxOnly(cmd, s.cfg.n, balls),
                    .gravity => s.grav.applyAutoBox(cmd, s.cfg.n, balls, scale, s.cfg.softening),
                    .drift => s.recordDrift(cmd),
                    .diag => {
                        const ro = [_]Buffer{ s.pos, s.body };
                        s.pass(cmd, .diagPartial, &ro, &.{ .{ .buffer = s.vel }, .{ .buffer = s.diag_buf } }, (s.cfg.n + sh.diag_per_thread - 1) / sh.diag_per_thread);
                    },
                    .potential => {
                        const ro = [_]Buffer{ s.pos, s.body };
                        s.pass(cmd, .potential, &ro, &.{.{ .buffer = s.pot_buf }}, s.samples * s.chunks);
                    },
                }
                try submitWait(s.dev, cmd);
                if (k > 0) total += msSince(t0);
            }
            @field(t, if (wi == 0) "gravity_fixed_box" else name) = total / @as(f64, @floatFromInt(reps));
        }
        return t;
    }

    fn pass(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, k: Kernel, ro: []const Buffer, rw: []const c.SDL_GPUStorageBufferReadWriteBinding, threads: u32) void {
        const p: sh.Params = .{
            .n = s.cfg.n,
            .samples = s.samples,
            .stride = s.stride,
            .offset = s.stride / 2,
            .chunks = s.chunks,
            .chunk_len = s.chunk_len,
            .dt = s.cfg.dt,
            .softening = s.cfg.softening,
        };
        c.SDL_PushGPUComputeUniformData(cmd, 0, &p, @sizeOf(sh.Params));
        const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, rw.ptr, @intCast(rw.len));
        c.SDL_BindGPUComputePipeline(cp, s.pipes[@backingInt(k)]);
        if (ro.len > 0) c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(ro.ptr), @intCast(ro.len));
        const groups = (threads + L.group_size - 1) / L.group_size;
        c.SDL_DispatchGPUCompute(cp, @min(groups, sh.dispatch_x), (groups + sh.dispatch_x - 1) / sh.dispatch_x, 1);
        c.SDL_EndGPUComputePass(cp);
    }
};

/// Potential-energy samples when none are requested: all bodies up to ~46k, then a budget of 2^31
/// pair evaluations (a few hundred ms on the dev GPU), at least 64.
pub fn autoSamples(n: u32) u32 {
    return @min(n, @max(64, @as(u32, @intCast((@as(u64, 1) << 31) / n))));
}

/// SDL_GPU device without a window: Vulkan 1.2 for the SPIR-V 1.5 kernels, no swapchain, no audio.
pub fn createDevice() Error!*c.SDL_GPUDevice {
    var vk_opts = std.mem.zeroes(c.SDL_GPUVulkanOptions);
    vk_opts.vulkan_api_version = (1 << 22) | (2 << 12);
    const props = c.SDL_CreateProperties();
    defer c.SDL_DestroyProperties(props);
    _ = c.SDL_SetBooleanProperty(props, c.SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, true);
    _ = c.SDL_SetPointerProperty(props, c.SDL_PROP_GPU_DEVICE_CREATE_VULKAN_OPTIONS_POINTER, &vk_opts);
    return c.SDL_CreateGPUDeviceWithProperties(props) orelse return error.GpuDevice;
}

pub fn msSince(t0: u64) f64 {
    const dt: f64 = @floatFromInt(c.SDL_GetPerformanceCounter() - t0);
    return dt * 1000.0 / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency()));
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

fn makeBuffer(dev: *c.SDL_GPUDevice, size: u64) Error!Buffer {
    if (size > std.math.maxInt(u32)) return error.OutOfRange;
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = @intCast(@max(size, 16)),
    }) orelse error.GpuBuffer;
}
