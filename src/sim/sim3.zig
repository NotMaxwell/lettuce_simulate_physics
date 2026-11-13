//! 3D matter simulation on the GPU: Newtonian 1/r^2 gravity (Plummer softened) with a global
//! kick-drift-kick leapfrog. Force is direct summation for now (k3/direct.zig); the tree comes later
//! (docs/sim3d.md). Positions are u32 box fractions, so the resolution is box_size / 2^32 everywhere.
//!
//! Time stepping is the 2D sim's (sim.zig): between ticks `vel` is the half-step velocity and
//! `half_pending` is false; `measure` applies the missing half kick to get synchronous velocities.
//!
//! Boundaries. Open: the domain is the middle half of the u32 range, [2^30, 3*2^30), so a wrapped
//! u32 difference never overflows i32 and the force kernel needs no boundary case; bodies that
//! reach the wall are clamped there and counted. Periodic: the whole range, wrap on overflow.
//! Drift rounds to the nearest quantum and keeps no remainder; a body moving less than half a
//! quantum (box_size / 2^33) per tick stands still. For a box 100x the system size that is ~1e-9 of
//! its extent, below f32 position resolution, so it is not worth 12 B/body of remainder.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const ic3 = @import("ic3");
const sim = @import("sim.zig");

const direct_spv = @embedFile("k3_direct.spv");
const step_spv = @embedFile("k3_step.spv");

pub const Error = sim.Error;
const Buffer = *c.SDL_GPUBuffer;

pub const Boundary = enum { open, periodic };

pub const Config = struct {
    n: u32,
    g: f32 = 1,
    /// Plummer softening length.
    eps: f32,
    dt: f32,
    box: ic3.Box,
    boundary: Boundary = .open,
    /// Time-step accuracy parameter for the rung histogram.
    eta: f32 = 0.02,
};

pub const rung_bins = 16;
const diag_per_thread = 16;
const diag_stride = 16;
const dispatch_x: u32 = 32768;
/// Bodies per kick dispatch when n is large: a dispatch of 65536 x 1M pairs is about half a
/// second, comfortably under the driver's job watchdog, while still filling the GPU.
const chunk_bodies: u32 = 65536;
const whole_kick_max: u32 = 131072;

pub const Diag3 = struct {
    ke: f64 = 0,
    pe: f64 = 0,
    px: f64 = 0,
    py: f64 = 0,
    pz: f64 = 0,
    lx: f64 = 0,
    ly: f64 = 0,
    lz: f64 = 0,
    mass: f64 = 0,
    escaped: u64 = 0,
    rungs: [rung_bins]u32 = @splat(0),

    pub fn e(d: Diag3) f64 {
        return d.ke + d.pe;
    }
    pub fn p(d: Diag3) f64 {
        return @sqrt(d.px * d.px + d.py * d.py + d.pz * d.pz);
    }
    pub fn l(d: Diag3) f64 {
        return @sqrt(d.lx * d.lx + d.ly * d.ly + d.lz * d.lz);
    }
    pub fn virial(d: Diag3) f64 {
        return 2 * d.ke / @abs(d.pe);
    }
};

const Kernel = enum { kick, drift, diag, rungsClear, rungs };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;

pub const Sim3 = struct {
    dev: *c.SDL_GPUDevice,
    cfg: Config,
    pos: Buffer,
    vel: Buffer,
    mass: Buffer,
    amag: Buffer,
    pot: Buffer,
    diag_buf: Buffer,
    bins: Buffer,
    download: *c.SDL_GPUTransferBuffer,
    pipes: [n_kernels]*c.SDL_GPUComputePipeline,
    diag_groups: u32,
    tick_no: u64 = 0,
    half_pending: bool = true,

    pub fn init(dev: *c.SDL_GPUDevice, cfg: Config) Error!Sim3 {
        if (cfg.n < 1 or cfg.n > 1 << 28) return error.OutOfRange;
        var s: Sim3 = undefined;
        s.dev = dev;
        s.cfg = cfg;
        s.tick_no = 0;
        s.half_pending = true;
        const n: u64 = cfg.n;
        s.pos = try makeBuffer(dev, n * @sizeOf(L.Pos3));
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.pos);
        s.vel = try makeBuffer(dev, n * @sizeOf(L.Vel3));
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.vel);
        s.mass = try makeBuffer(dev, n * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.mass);
        s.amag = try makeBuffer(dev, n * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.amag);
        s.pot = try makeBuffer(dev, n * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.pot);
        s.diag_groups = ((cfg.n + diag_per_thread - 1) / diag_per_thread + L.group_size - 1) / L.group_size;
        s.diag_buf = try makeBuffer(dev, @as(u64, s.diag_groups) * diag_stride * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.diag_buf);
        s.bins = try makeBuffer(dev, rung_bins * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.bins);
        s.download = c.SDL_CreateGPUTransferBuffer(dev, &.{
            .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
            .size = s.diag_groups * diag_stride * 4 + rung_bins * 4,
        }) orelse return error.GpuBuffer;
        errdefer c.SDL_ReleaseGPUTransferBuffer(dev, s.download);

        const Spec = struct { code: []const u8, ro: u32, rw: u32 };
        const specs = [n_kernels]Spec{
            .{ .code = direct_spv, .ro = 2, .rw = 3 },
            .{ .code = step_spv, .ro = 0, .rw = 2 },
            .{ .code = step_spv, .ro = 4, .rw = 1 },
            .{ .code = step_spv, .ro = 0, .rw = 1 },
            .{ .code = step_spv, .ro = 1, .rw = 1 },
        };
        var infos: [n_kernels]c.SDL_GPUComputePipelineCreateInfo = undefined;
        inline for (@typeInfo(Kernel).@"enum".field_names, &infos, specs) |name, *info, sp| {
            info.* = .{
                .code_size = sp.code.len,
                .code = sp.code.ptr,
                .entrypoint = name,
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = sp.ro,
                .num_readwrite_storage_buffers = sp.rw,
                .num_uniform_buffers = 1,
                .threadcount_x = L.group_size,
                .threadcount_y = 1,
                .threadcount_z = 1,
            };
        }
        var made: [n_kernels]?*c.SDL_GPUComputePipeline = undefined;
        if (!@import("gpu_sim").createComputePipelines(dev, &infos, &made)) return error.GpuPipeline;
        for (&s.pipes, made) |*p, m| p.* = m.?;
        return s;
    }

    pub fn deinit(s: *Sim3) void {
        c.SDL_ReleaseGPUTransferBuffer(s.dev, s.download);
        for ([_]Buffer{ s.pos, s.vel, s.mass, s.amag, s.pot, s.diag_buf, s.bins }) |b| c.SDL_ReleaseGPUBuffer(s.dev, b);
        for (s.pipes) |p| c.SDL_ReleaseGPUComputePipeline(s.dev, p);
        s.* = undefined;
    }

    pub fn bytesPerBody() u64 {
        return @sizeOf(L.Pos3) + @sizeOf(L.Vel3) + 3 * 4;
    }

    /// Quantum count of a world coordinate; clamped into the open domain, wrapped when periodic.
    pub fn toQuanta(s: *const Sim3, x: f64, axis: usize) u32 {
        const f = (x - s.cfg.box.centre[axis]) / s.cfg.box.size;
        const q = std.math.ldexp(f, 32) + 0x1p31;
        if (s.cfg.boundary == .periodic) {
            const w = @mod(@round(q), 0x1p32);
            return @intFromFloat(w);
        }
        return @intFromFloat(std.math.clamp(@round(q), 0x1p30, 3 * 0x1p30 - 1));
    }

    pub fn toWorld(s: *const Sim3, q: u32, axis: usize) f64 {
        const signed: i32 = @bitCast(q -% (1 << 31));
        return s.cfg.box.centre[axis] + @as(f64, @floatFromInt(signed)) / 0x1p32 * s.cfg.box.size;
    }

    /// Blocking upload of the initial conditions (synchronous velocities); resets the clock.
    pub fn upload(s: *Sim3, bodies: []const ic3.Body3) Error!void {
        if (bodies.len != s.cfg.n) return error.OutOfRange;
        const chunk: usize = 1 << 20;
        const tb = c.SDL_CreateGPUTransferBuffer(s.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = @intCast(chunk * 12) }) orelse return error.GpuBuffer;
        defer c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);
        var at: usize = 0;
        while (at < bodies.len) : (at += chunk) {
            const part = bodies[at..@min(at + chunk, bodies.len)];
            for (0..3) |which| {
                const m: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuBuffer);
                const size: usize = if (which == 2) 4 else 12;
                switch (which) {
                    0 => for (@as([*]L.Pos3, @ptrCast(@alignCast(m)))[0..part.len], part) |*o, b| {
                        o.* = .{ .x = s.toQuanta(b.x, 0), .y = s.toQuanta(b.y, 1), .z = s.toQuanta(b.z, 2) };
                    },
                    1 => for (@as([*]L.Vel3, @ptrCast(@alignCast(m)))[0..part.len], part) |*o, b| {
                        o.* = .{ .x = @floatCast(b.vx), .y = @floatCast(b.vy), .z = @floatCast(b.vz) };
                    },
                    else => for (@as([*]f32, @ptrCast(@alignCast(m)))[0..part.len], part) |*o, b| {
                        o.* = @floatCast(b.m);
                    },
                }
                c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
                const dsts = [3]Buffer{ s.pos, s.vel, s.mass };
                const dst = dsts[which];
                const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                const cp = c.SDL_BeginGPUCopyPass(cmd);
                c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = tb }, &.{ .buffer = dst, .offset = @intCast(at * size), .size = @intCast(part.len * size) }, false);
                c.SDL_EndGPUCopyPass(cp);
                try submitWait(s.dev, cmd);
            }
        }
        s.tick_no = 0;
        s.half_pending = true;
    }

    fn params(s: *const Sim3, first: u32, kick: f32, flags: u32) L.Params3 {
        const size = s.cfg.box.size;
        return .{
            .n = s.cfg.n,
            .first = first,
            .flags = flags | if (s.cfg.boundary == .periodic) L.sim3_flag_periodic else 0,
            .g = s.cfg.g,
            .eps = s.cfg.eps,
            .eps2 = s.cfg.eps * s.cfg.eps,
            .kick = kick,
            .drift_q = @floatCast(@as(f64, s.cfg.dt) * 0x1p32 / size),
            .box_scale = @floatCast(size / 0x1p32),
            .dt = s.cfg.dt,
            .eta = s.cfg.eta,
        };
    }

    fn pass(s: *Sim3, cmd: *c.SDL_GPUCommandBuffer, k: Kernel, p: L.Params3, ro: []const Buffer, rw: []const Buffer, threads: u32) void {
        c.SDL_PushGPUComputeUniformData(cmd, 0, &p, @sizeOf(L.Params3));
        var rwb: [4]c.SDL_GPUStorageBufferReadWriteBinding = undefined;
        for (rw, 0..) |b, i| rwb[i] = .{ .buffer = b };
        const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &rwb, @intCast(rw.len));
        c.SDL_BindGPUComputePipeline(cp, s.pipes[@intFromEnum(k)]);
        if (ro.len > 0) c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(ro.ptr), @intCast(ro.len));
        const groups = (threads + L.group_size - 1) / L.group_size;
        c.SDL_DispatchGPUCompute(cp, @min(groups, dispatch_x), (groups + dispatch_x - 1) / dispatch_x, 1);
        c.SDL_EndGPUComputePass(cp);
    }

    /// Records vel += factor * dt * a(pos) (plus potentials when asked). Large n is cut into
    /// separately submitted chunks, so the returned command buffer may not be `cmd0`.
    fn recordKick(s: *Sim3, cmd0: *c.SDL_GPUCommandBuffer, factor: f32, flags: u32) Error!*c.SDL_GPUCommandBuffer {
        const n = s.cfg.n;
        const kick = s.cfg.dt * factor;
        const ro = [_]Buffer{ s.pos, s.mass };
        const rw = [_]Buffer{ s.vel, s.amag, s.pot };
        if (n <= whole_kick_max) {
            s.pass(cmd0, .kick, s.params(0, kick, flags), &ro, &rw, n);
            return cmd0;
        }
        var cmd = cmd0;
        var first: u32 = 0;
        while (first < n) : (first += chunk_bodies) {
            s.pass(cmd, .kick, s.params(first, kick, flags), &ro, &rw, @min(chunk_bodies, n - first));
            try submitWait(s.dev, cmd);
            cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
        }
        return cmd;
    }

    fn recordDrift(s: *Sim3, cmd: *c.SDL_GPUCommandBuffer) void {
        s.pass(cmd, .drift, s.params(0, 0, 0), &.{}, &.{ s.pos, s.vel }, s.cfg.n);
    }

    pub fn recordTick(s: *Sim3, cmd0: *c.SDL_GPUCommandBuffer) Error!*c.SDL_GPUCommandBuffer {
        const cmd = try s.recordKick(cmd0, if (s.half_pending) 0.5 else 1.0, 0);
        s.half_pending = false;
        s.recordDrift(cmd);
        s.tick_no += 1;
        return cmd;
    }

    /// Runs `ticks` ticks, a few in flight when each is short; returns when done.
    pub fn run(s: *Sim3, ticks: u64) Error!void {
        const depth: usize = if (s.cfg.n > whole_kick_max) 1 else 4;
        var fences: [4]?*c.SDL_GPUFence = @splat(null);
        for (0..ticks) |k| {
            const slot = k % depth;
            if (fences[slot]) |f| waitRelease(s.dev, f);
            const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            const last = try s.recordTick(cmd);
            fences[slot] = c.SDL_SubmitGPUCommandBufferAndAcquireFence(last) orelse return error.GpuSubmit;
        }
        for (&fences) |*f| if (f.*) |ff| {
            waitRelease(s.dev, ff);
            f.* = null;
        };
    }

    /// Diagnostics at synchronous velocities (blocking). The force pass that completes the pending
    /// half kick also yields the potentials and |a|; with nothing pending it runs with a zero kick.
    pub fn measure(s: *Sim3) Error!Diag3 {
        var cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
        cmd = try s.recordKick(cmd, if (s.half_pending) 0 else 0.5, L.sim3_flag_potential);
        s.half_pending = true;
        const p = s.params(0, 0, 0);
        s.pass(cmd, .rungsClear, p, &.{}, &.{s.bins}, rung_bins);
        s.pass(cmd, .rungs, p, &.{s.amag}, &.{s.bins}, s.cfg.n);
        s.pass(cmd, .diag, p, &.{ s.pos, s.vel, s.mass, s.pot }, &.{s.diag_buf}, (s.cfg.n + diag_per_thread - 1) / diag_per_thread);
        const diag_bytes = s.diag_groups * diag_stride * 4;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.diag_buf, .size = diag_bytes }, &.{ .transfer_buffer = s.download, .offset = 0 });
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.bins, .size = rung_bins * 4 }, &.{ .transfer_buffer = s.download, .offset = diag_bytes });
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(s.dev, cmd);
        const m: [*]const u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(s.dev, s.download, false) orelse return error.GpuBuffer);
        defer c.SDL_UnmapGPUTransferBuffer(s.dev, s.download);
        const f: [*]const f32 = @ptrCast(@alignCast(m));
        var d: Diag3 = .{};
        for (0..s.diag_groups) |gi| {
            const row = f[gi * diag_stride ..][0..10];
            d.ke += row[0];
            d.px += row[1];
            d.py += row[2];
            d.pz += row[3];
            d.lx += row[4];
            d.ly += row[5];
            d.lz += row[6];
            d.mass += row[7];
            d.pe += row[8];
            d.escaped += @intFromFloat(row[9]);
        }
        const r: [*]const u32 = @ptrCast(@alignCast(m + diag_bytes));
        d.rungs = r[0..rung_bins].*;
        return d;
    }

    /// Reads `out.len` elements of `buf` back (blocking; tests and small runs).
    pub fn read(s: *Sim3, comptime T: type, buf: Buffer, out: []T) Error!void {
        const size: u32 = @intCast(out.len * @sizeOf(T));
        const tb = c.SDL_CreateGPUTransferBuffer(s.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = size }) orelse return error.GpuBuffer;
        defer c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);
        const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = buf, .size = size }, &.{ .transfer_buffer = tb, .offset = 0 });
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(s.dev, cmd);
        const m: [*]const T = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuBuffer));
        defer c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
        @memcpy(out, m[0..out.len]);
    }

    /// Accelerations of the current positions (G applied) into `out`; overwrites vel, so only
    /// use it before the first tick or for force tests.
    pub fn accelerations(s: *Sim3, out: []L.Vel3) Error!void {
        const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
        const last = try s.recordKick(cmd, 0, L.sim3_flag_store_accel);
        try submitWait(s.dev, last);
        try s.read(L.Vel3, s.vel, out);
    }

    pub const PassTimes = struct { kick: f64, drift: f64, diag: f64 };

    /// Blocking per-pass times in ms: the minimum of `reps` samples after one warm-up (the GPU is shared).
    pub fn timePasses(s: *Sim3, reps: u32) Error!PassTimes {
        var best = PassTimes{ .kick = std.math.inf(f64), .drift = std.math.inf(f64), .diag = std.math.inf(f64) };
        for (0..reps + 1) |k| {
            var cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            var t0 = c.SDL_GetPerformanceCounter();
            cmd = try s.recordKick(cmd, 1, 0);
            try submitWait(s.dev, cmd);
            const tk = sim.msSince(t0);
            cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            t0 = c.SDL_GetPerformanceCounter();
            s.recordDrift(cmd);
            try submitWait(s.dev, cmd);
            const td = sim.msSince(t0);
            if (k > 0) {
                best.kick = @min(best.kick, tk);
                best.drift = @min(best.drift, td);
            }
        }
        return best;
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

fn makeBuffer(dev: *c.SDL_GPUDevice, size: u64) Error!Buffer {
    if (size > std.math.maxInt(u32)) return error.OutOfRange;
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = @intCast(@max(size, 16)),
    }) orelse error.GpuBuffer;
}

pub const Options = struct {
    ic: ic3.Model = .plummer,
    n: u32 = 10_000,
    ticks: u64 = 1000,
    dt: f32 = 0.01,
    every: u64 = 100,
    seed: u64 = 1,
    eps: f32 = 0.01,
    ecc: f64 = 0,
    margin: f64 = 4,
    boundary: Boundary = .open,
    bench: bool = false,
    /// Accepted for the tree solver; direct summation ignores it.
    theta: f32 = 0.6,
};

const usage =
    \\usage: sim --dim 3 [--ic plummer|kepler|uniform] [--n N] [--ticks K] [--dt F] [--every K] [--seed U64]
    \\               [--eps LEN] [--ecc E] [--margin M] [--boundary open|periodic] [--method direct]
    \\               [--theta F (ignored)] [--bench]
    \\
;

fn parse(args: []const [:0]const u8) !Options {
    var o: Options = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--bench")) {
            o.bench = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        const v = args[i];
        if (std.mem.eql(u8, a, "--dim")) {
            if (!std.mem.eql(u8, v, "3")) return error.BadDim;
        } else if (std.mem.eql(u8, a, "--ic")) {
            o.ic = std.meta.stringToEnum(ic3.Model, v) orelse return error.BadIc;
        } else if (std.mem.eql(u8, a, "--n")) {
            o.n = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--ticks")) {
            o.ticks = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, a, "--dt")) {
            o.dt = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, a, "--every")) {
            o.every = try std.fmt.parseInt(u64, v, 10);
            if (o.every == 0) return error.ZeroEvery;
        } else if (std.mem.eql(u8, a, "--seed")) {
            o.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, a, "--eps")) {
            o.eps = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, a, "--ecc")) {
            o.ecc = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, a, "--margin")) {
            o.margin = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, a, "--boundary")) {
            o.boundary = std.meta.stringToEnum(Boundary, v) orelse return error.BadBoundary;
        } else if (std.mem.eql(u8, a, "--method")) {
            if (!std.mem.eql(u8, v, "direct")) return error.BadMethod;
        } else if (std.mem.eql(u8, a, "--theta")) {
            o.theta = try std.fmt.parseFloat(f32, v);
        } else return error.UnknownOption;
    }
    return o;
}

pub fn main(gpa: std.mem.Allocator, args: []const [:0]const u8) !void {
    const o = parse(args) catch |err| {
        std.debug.print("sim: {s}\n{s}", .{ @errorName(err), usage });
        return err;
    };
    if (o.ticks == 0) return error.BadTicks;
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();
    const dev = try sim.createDevice();
    defer c.SDL_DestroyGPUDevice(dev);

    var prng = std.Random.DefaultPrng.init(o.seed);
    var dt = o.dt;
    const bodies: []ic3.Body3 = switch (o.ic) {
        .plummer => try ic3.plummer(gpa, prng.random(), o.n, .{}),
        .uniform => try ic3.uniform(gpa, prng.random(), o.n, .{ .sigma = 0.3 }),
        .kepler => blk: {
            const k: ic3.Kepler = .{ .e = o.ecc };
            dt = @floatCast(ic3.keplerPeriod(k) / 2000.0);
            const pair = ic3.kepler(k);
            break :blk try gpa.dupe(ic3.Body3, &pair);
        },
    };
    defer gpa.free(bodies);
    const n: u32 = @intCast(bodies.len);
    const box = ic3.fitBox(bodies, o.margin);
    var s = Sim3.init(dev, .{ .n = n, .eps = o.eps, .dt = dt, .box = box, .boundary = o.boundary }) catch |err| {
        std.debug.print("sim3 init failed for n={d}: {s} ({s})\n", .{ n, @errorName(err), c.SDL_GetError() });
        return err;
    };
    defer s.deinit();
    try s.upload(bodies);
    std.debug.print("{s}: {d} bodies, box {d:.4} (quantum {e:.2}), eps {d}, dt {d}, {d} B/body\n", .{ @tagName(o.ic), n, box.size, box.size / 0x1p32, o.eps, dt, Sim3.bytesPerBody() });
    if (o.bench) return bench(&s, o);

    const d0 = try s.measure();
    std.debug.print("{s:>8} {s:>14} {s:>10} {s:>8} {s:>10} {s:>10} {s:>10} {s:>9}\n", .{ "tick", "E", "dE/E0", "2K/|W|", "|P|", "|L|", "dL/L0", "ms/tick" });
    report(0, d0, d0, 0);
    const end = o.ticks;
    var ms_acc: f64 = 0;
    var ticks_acc: u64 = 0;
    while (s.tick_no < end) {
        const step = @min((s.tick_no / o.every + 1) * o.every, end) - s.tick_no;
        const t = c.SDL_GetPerformanceCounter();
        try s.run(step);
        ms_acc += sim.msSince(t);
        ticks_acc += step;
        if (s.tick_no % o.every == 0 or s.tick_no == end) {
            report(s.tick_no, try s.measure(), d0, ms_acc / @as(f64, @floatFromInt(ticks_acc)));
            ms_acc = 0;
            ticks_acc = 0;
        }
    }
}

fn report(tick: u64, d: Diag3, d0: Diag3, ms: f64) void {
    std.debug.print("{d:>8} {e:>14.6} {e:>10.2} {d:>8.4} {e:>10.3} {e:>10.3} {e:>10.2} {d:>9.3}\n", .{
        tick, d.e(), (d.e() - d0.e()) / @abs(d0.e()), d.virial(), d.p(), d.l(), (d.l() - d0.l()) / d0.l(), ms,
    });
    std.debug.print("         rungs (r: bodies wanting dt/2^r):", .{});
    for (d.rungs, 0..) |cnt, r| if (cnt > 0) std.debug.print(" {d}:{d}", .{ r, cnt });
    std.debug.print("\n", .{});
    if (d.escaped > 0) std.debug.print("         escapers clamped at the wall: {d}\n", .{d.escaped});
}

fn bench(s: *Sim3, o: Options) !void {
    try s.run(3);
    const ticks = @min(o.ticks, 100);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const t = c.SDL_GetPerformanceCounter();
        try s.run(ticks);
        best = @min(best, sim.msSince(t) / @as(f64, @floatFromInt(ticks)));
    }
    const pt = try s.timePasses(3);
    std.debug.print("bench n={d}: pipelined {d:.3} ms/tick (min of 3 runs of {d} ticks); blocking kick {d:.3}, drift {d:.3}\n", .{ s.cfg.n, best, ticks, pt.kick, pt.drift });
}
