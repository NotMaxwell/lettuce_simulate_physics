//! 3D matter simulation on the GPU: Newtonian 1/r^2 gravity (Plummer softened) with a global
//! kick-drift-kick leapfrog. Force is direct summation (k3/direct.zig) or Barnes-Hut (Sort3, Tree3,
//! Walk3; docs/sim3d.md). Positions are u32 box fractions, so the resolution is box_size / 2^32 everywhere.
//!
//! Tree method: every force evaluation first sorts pos/vel/mass into Morton order in place, so the
//! bodies live in sorted order and snapshots store them that way (the stable sort makes restart
//! bit-identical, ties included). |a| and the potential only matter right after a walk, so they live
//! in the sort's SortRec buffers, which are idle by then (8 B/body saved).
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
const snap = @import("snapshot.zig");
const snapper = @import("snapper.zig");
const Sort3 = @import("sort3.zig").Sort3;
const tree3 = @import("tree3.zig");
const Tree3 = tree3.Tree3;
const Walk3 = @import("walk3.zig").Walk3;

const direct_spv = @embedFile("k3_direct.spv");
const step_spv = @embedFile("k3_step.spv");

pub const Error = sim.Error || error{ GpuAcquire, ScratchTooSmall };
const Buffer = *c.SDL_GPUBuffer;
const Cmd = *c.SDL_GPUCommandBuffer;

pub const Boundary = enum { open, periodic };
pub const Method = enum { direct, tree };

/// `--method auto` takes direct summation up to this many bodies.
pub const auto_tree_max_direct: u32 = 50_000;

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
    method: Method = .direct,
    /// Opening angle of the tree method.
    theta: f32 = 0.6,
    /// Tree node capacity is n / node_div (the design's N/4; the measured need is 0.09N). A larger
    /// divisor exists to exercise the overflow path.
    node_div: u32 = 4,
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
    /// Tree method only.
    sort: ?Sort3 = null,
    tree: ?Tree3 = null,
    walk: ?Walk3 = null,
    /// Per in-flight tick, the tree status words (leaves, nodes, overflow, depth) copied at the end
    /// of its force evaluation; read once that tick's fence has passed.
    status_tb: ?*c.SDL_GPUTransferBuffer = null,
    status_slot: u32 = 0,
    overflow_pending: bool = false,
    /// Ticks whose tree overflowed (they had zero force) and how many times the tree was regrown.
    overflow_ticks: u64 = 0,
    regrows: u32 = 0,

    const status_slots = 4;
    const status_words = 4;

    pub fn init(dev: *c.SDL_GPUDevice, cfg: Config) Error!Sim3 {
        if (cfg.n < 1 or cfg.n > 1 << 28 or !(cfg.eps > 0) or !(cfg.box.size > 0)) return error.OutOfRange;
        // The walk assumes an open box (no minimum-image wrap).
        if (cfg.method == .tree and (cfg.boundary == .periodic or !(cfg.theta > 0) or cfg.node_div == 0)) return error.OutOfRange;
        var s: Sim3 = undefined;
        s.dev = dev;
        s.cfg = cfg;
        s.tick_no = 0;
        s.half_pending = true;
        s.sort = null;
        s.tree = null;
        s.walk = null;
        s.status_tb = null;
        s.status_slot = 0;
        s.overflow_pending = false;
        s.overflow_ticks = 0;
        s.regrows = 0;
        const n: u64 = cfg.n;
        s.pos = try makeBuffer(dev, n * @sizeOf(L.Pos3));
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.pos);
        s.vel = try makeBuffer(dev, n * @sizeOf(L.Vel3));
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.vel);
        s.mass = try makeBuffer(dev, n * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.mass);
        if (cfg.method == .tree) {
            s.sort = try Sort3.init(dev, cfg.n);
            errdefer s.sort.?.deinit();
            s.amag = s.sort.?.recs[0];
            s.pot = s.sort.?.recs[1];
            s.tree = try s.buildTree(s.cfg.n / cfg.node_div);
            errdefer s.tree.?.deinit();
            s.walk = try Walk3.init(dev);
            errdefer s.walk.?.deinit();
            s.status_tb = c.SDL_CreateGPUTransferBuffer(dev, &.{
                .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
                .size = status_slots * status_words * 4,
            }) orelse return error.GpuBuffer;
        } else {
            s.amag = try makeBuffer(dev, n * 4);
            errdefer c.SDL_ReleaseGPUBuffer(dev, s.amag);
            s.pot = try makeBuffer(dev, n * 4);
            errdefer c.SDL_ReleaseGPUBuffer(dev, s.pot);
        }
        errdefer if (s.status_tb) |tb| c.SDL_ReleaseGPUTransferBuffer(dev, tb);
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
        for ([_]Buffer{ s.pos, s.vel, s.mass, s.diag_buf, s.bins }) |b| c.SDL_ReleaseGPUBuffer(s.dev, b);
        if (s.sort) |*x| x.deinit() else for ([_]Buffer{ s.amag, s.pot }) |b| c.SDL_ReleaseGPUBuffer(s.dev, b);
        if (s.tree) |*x| x.deinit();
        if (s.walk) |*x| x.deinit();
        if (s.status_tb) |tb| c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);
        for (s.pipes) |p| c.SDL_ReleaseGPUComputePipeline(s.dev, p);
        s.* = undefined;
    }

    /// Device bytes per body of the solver state (the small diagnostics buffers excluded).
    pub fn bytesPerBody(s: *const Sim3) f64 {
        var b: f64 = @sizeOf(L.Pos3) + @sizeOf(L.Vel3) + 4;
        if (s.sort != null) {
            b += Sort3.bytesPerBody() + s.tree.?.bytesPerBody();
        } else b += 8;
        return b;
    }

    /// Tree on the sort's idle SortRec copy, or on its own scratch when that is too small (tiny n,
    /// or a regrow past what the borrowed scratch holds).
    fn buildTree(s: *Sim3, node_cap: u32) Error!Tree3 {
        const sort = &s.sort.?;
        const cap = std.math.clamp(node_cap, 1, 2 * s.cfg.n);
        return Tree3.initWith(s.dev, s.cfg.n, .{
            .node_cap = cap,
            .scratch = sort.scratch(),
            .scratch_bytes = @as(u64, s.cfg.n) * @sizeOf(L.SortRec),
        }) catch |err| switch (err) {
            error.ScratchTooSmall => Tree3.initWith(s.dev, s.cfg.n, .{ .node_cap = cap }),
            else => err,
        };
    }

    /// Quantum count of a world coordinate; clamped into the open domain, wrapped when periodic.
    pub fn toQuanta(s: *const Sim3, x: f64) u32 {
        const f = x / s.cfg.box.size;
        const q = std.math.ldexp(f, 32) + 0x1p31;
        if (s.cfg.boundary == .periodic) {
            const w = @mod(@round(q), 0x1p32);
            return @intFromFloat(w);
        }
        return @intFromFloat(std.math.clamp(@round(q), 0x1p30, 3 * 0x1p30 - 1));
    }

    pub fn toWorld(s: *const Sim3, q: u32) f64 {
        const signed: i32 = @bitCast(q -% (1 << 31));
        return @as(f64, @floatFromInt(signed)) / 0x1p32 * s.cfg.box.size;
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
                        o.* = .{ .x = s.toQuanta(b.x), .y = s.toQuanta(b.y), .z = s.toQuanta(b.z) };
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

    /// Snapshotter over this sim (snapshot v2: pos xyz as u32 words, vel xyz, mass).
    pub fn snapshotter(s: *Sim3, gpa: std.mem.Allocator) snapper.Error!*snapper.Snapper {
        const sources = [_]snapper.Source{
            .{ .buf = s.pos, .stride = @sizeOf(L.Pos3) },
            .{ .buf = s.vel, .stride = @sizeOf(L.Vel3) },
            .{ .buf = s.mass, .stride = 4 },
        };
        const fields = [_]snapper.FieldSpec{
            .{ .source = 0, .word = 0 }, .{ .source = 0, .word = 1 }, .{ .source = 0, .word = 2 },
            .{ .source = 1, .word = 0 }, .{ .source = 1, .word = 1 }, .{ .source = 1, .word = 2 },
            .{ .source = 2, .word = 0 },
        };
        return snapper.Snapper.createGeneric(gpa, s.dev, s.cfg.n, &sources, &fields, .{ .ctx = s, .recordSync = syncHook, .makeHeader = headerHook });
    }

    /// Loads a 3D snapshot (n and box must match) and sets the clock.
    pub fn restore(s: *Sim3, gpa: std.mem.Allocator, io: *c.SDL_IOStream, h: *const snap.Header) (Error || snap.Error || error{OutOfMemory})!void {
        if (h.version != snap.version_3d or h.dim != 3 or h.n != s.cfg.n or h.box_size != s.cfg.box.size or h.hasUniformMass()) return error.BadHeader;
        const chunk: usize = 1 << 18;
        const tb = c.SDL_CreateGPUTransferBuffer(s.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = @intCast(chunk * 12) }) orelse return error.GpuBuffer;
        defer c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);
        const words = try gpa.alloc(u32, chunk * snap.field_count3);
        defer gpa.free(words);
        var at: usize = 0;
        while (at < s.cfg.n) : (at += chunk) {
            const m = @min(chunk, s.cfg.n - at);
            for (0..snap.field_count3) |k| try snap.readWords(io, h, @intCast(k), at, words[k * chunk ..][0..m]);
            const dsts = [3]Buffer{ s.pos, s.vel, s.mass };
            for (dsts, [3]usize{ 3, 3, 1 }, [3]usize{ 0, 3, 6 }) |dst, per, first| {
                const out: [*]u32 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuBuffer));
                for (0..m) |j| for (0..per) |d| {
                    out[j * per + d] = words[(first + d) * chunk + j];
                };
                c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
                const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                const cp = c.SDL_BeginGPUCopyPass(cmd);
                c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = tb }, &.{ .buffer = dst, .offset = @intCast(at * per * 4), .size = @intCast(m * per * 4) }, false);
                c.SDL_EndGPUCopyPass(cp);
                try submitWait(s.dev, cmd);
            }
        }
        s.tick_no = h.tick;
        s.half_pending = h.half_pending != 0;
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
        c.SDL_BindGPUComputePipeline(cp, s.pipes[@backingInt(k)]);
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

    /// Sort + tree build for the current positions (tree method).
    fn recordSortTree(s: *Sim3, cmd0: Cmd) Error!Cmd {
        const sort = &s.sort.?;
        const cmd = try sort.record(cmd0, .{ .pos = s.pos, .vel = s.vel, .mass = s.mass }, s.cfg.n);
        try s.recordTreeBuild(cmd);
        return cmd;
    }

    fn recordTreeBuild(s: *Sim3, cmd: Cmd) Error!void {
        try s.tree.?.record(cmd, .{ .pos = s.pos, .mass = s.mass, .keys = s.sort.?.sorted() }, s.cfg.n, .{ .theta = s.cfg.theta, .box_size = s.cfg.box.size });
    }

    fn recordWalk(s: *Sim3, cmd: Cmd, factor: f32, flags: u32) Error!Cmd {
        const tree = &s.tree.?;
        return s.walk.?.record(cmd, .{
            .pos = s.pos,
            .mass = s.mass,
            .nodes = tree.nodes,
            .quads = tree.quads,
            .vel = s.vel,
            .amag = s.amag,
            .pot = s.pot,
        }, .{
            .n = s.cfg.n,
            .g = s.cfg.g,
            .eps = s.cfg.eps,
            .kick = s.cfg.dt * factor,
            .box_scale = @floatCast(s.cfg.box.size / 0x1p32),
            .flags = flags,
        });
    }

    fn recordStatusCopy(s: *Sim3, cmd: Cmd) void {
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{
            .buffer = s.tree.?.statusBuffer(),
            .offset = 4 * tree3.status_leaves,
            .size = status_words * 4,
        }, &.{ .transfer_buffer = s.status_tb.?, .offset = s.status_slot * status_words * 4 });
        c.SDL_EndGPUCopyPass(cp);
    }

    /// Records vel += factor * dt * a(pos) by the configured method. The tree method reorders the
    /// bodies first. The returned command buffer may not be `cmd0` (see recordKick, Sort3.record).
    fn recordForce(s: *Sim3, cmd0: Cmd, factor: f32, flags: u32) Error!Cmd {
        if (s.cfg.method == .direct) return s.recordKick(cmd0, factor, flags);
        var cmd = try s.recordSortTree(cmd0);
        cmd = try s.recordWalk(cmd, factor, flags);
        s.recordStatusCopy(cmd);
        return cmd;
    }

    /// Reads the status of the tick that used `slot` (its fence has passed) and flags an overflow.
    fn noteStatus(s: *Sim3, slot: usize) void {
        const tb = s.status_tb orelse return;
        const w: [*]const u32 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return));
        defer c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
        if (w[slot * status_words + 2] != 0) {
            s.overflow_pending = true;
            s.overflow_ticks += 1;
        }
    }

    /// Replaces the tree by one with more nodes, once every command buffer using it has finished.
    /// Ticks whose build overflowed have had zero force; there is no way to redo them.
    fn regrow(s: *Sim3) Error!void {
        s.overflow_pending = false;
        const st = try s.tree.?.readStatus();
        const old = s.tree.?.node_cap;
        const want: u32 = @intCast(@min(@as(u64, s.cfg.n) * 2, @max(@as(u64, old) * 2, @as(u64, st.nodes) + st.nodes / 8)));
        std.debug.print("sim3: tree overflow at tick {d}: {d} nodes needed, capacity {d}; regrowing to {d} ({d} ticks so far had zero force)\n", .{ s.tick_no, st.nodes, old, want, s.overflow_ticks });
        s.tree.?.deinit();
        s.tree = null;
        s.tree = try s.buildTree(want);
        s.regrows += 1;
    }

    pub fn recordTick(s: *Sim3, cmd0: Cmd) Error!Cmd {
        const cmd = try s.recordForce(cmd0, if (s.half_pending) 0.5 else 1.0, 0);
        s.half_pending = false;
        s.recordDrift(cmd);
        s.tick_no += 1;
        return cmd;
    }

    /// Runs `ticks` ticks, a few in flight when each is short; returns when done.
    pub fn run(s: *Sim3, ticks: u64) Error!void {
        const depth: usize = if (s.cfg.n > whole_kick_max) 1 else status_slots;
        var fences: [status_slots]?*c.SDL_GPUFence = @splat(null);
        errdefer s.drain(&fences);
        for (0..ticks) |k| {
            const slot = k % depth;
            if (fences[slot]) |f| {
                waitRelease(s.dev, f);
                fences[slot] = null;
                s.noteStatus(slot);
            }
            if (s.overflow_pending) {
                s.drain(&fences);
                try s.regrow();
            }
            s.status_slot = @intCast(slot);
            const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            const last = try s.recordTick(cmd);
            fences[slot] = c.SDL_SubmitGPUCommandBufferAndAcquireFence(last) orelse return error.GpuSubmit;
        }
        s.drain(&fences);
        if (s.overflow_pending) try s.regrow();
    }

    fn drain(s: *Sim3, fences: *[status_slots]?*c.SDL_GPUFence) void {
        for (fences, 0..) |*f, slot| if (f.*) |ff| {
            waitRelease(s.dev, ff);
            f.* = null;
            s.noteStatus(slot);
        };
    }

    fn recordDiag(s: *Sim3, cmd: Cmd) void {
        const p = s.params(0, 0, 0);
        s.pass(cmd, .rungsClear, p, &.{}, &.{s.bins}, rung_bins);
        s.pass(cmd, .rungs, p, &.{s.amag}, &.{s.bins}, s.cfg.n);
        s.pass(cmd, .diag, p, &.{ s.pos, s.vel, s.mass, s.pot }, &.{s.diag_buf}, (s.cfg.n + diag_per_thread - 1) / diag_per_thread);
        const diag_bytes = s.diag_groups * diag_stride * 4;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.diag_buf, .size = diag_bytes }, &.{ .transfer_buffer = s.download, .offset = 0 });
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.bins, .size = rung_bins * 4 }, &.{ .transfer_buffer = s.download, .offset = diag_bytes });
        c.SDL_EndGPUCopyPass(cp);
    }

    /// Diagnostics at synchronous velocities (blocking). The force pass that completes the pending
    /// half kick also yields the potentials and |a|; with nothing pending it runs with a zero kick.
    pub fn measure(s: *Sim3) Error!Diag3 {
        const factor: f32 = if (s.half_pending) 0 else 0.5;
        s.half_pending = true;
        // An overflowed build gives zero force, hence no kick and a wrong potential. The state is
        // unchanged by it, so the evaluation is repeated after the regrow (a node capacity of 2n
        // cannot overflow, which bounds the loop).
        for (0..4) |_| {
            var cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            s.status_slot = 0;
            cmd = try s.recordForce(cmd, factor, L.sim3_flag_potential);
            s.recordDiag(cmd);
            try submitWait(s.dev, cmd);
            s.noteStatus(0);
            if (!s.overflow_pending) break;
            try s.regrow();
        }
        const diag_bytes = s.diag_groups * diag_stride * 4;
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
        const last = try s.recordForce(cmd, 0, L.sim3_flag_store_accel);
        try submitWait(s.dev, last);
        try s.read(L.Vel3, s.vel, out);
    }

    /// ms of each pass (kick = the whole force evaluation of the direct method; walk_pot is a walk
    /// that also writes potentials, as on diagnostic ticks).
    pub const PassTimes = struct { sort: f64 = 0, tree: f64 = 0, walk: f64 = 0, walk_pot: f64 = 0, kick: f64 = 0, drift: f64 = 0, diag: f64 = 0 };

    /// Blocking per-pass times in ms: the minimum of `reps` samples after one warm-up (the GPU is shared).
    pub fn timePasses(s: *Sim3, reps: u32) Error!PassTimes {
        var best = PassTimes{};
        inline for (@typeInfo(PassTimes).@"struct".field_names) |f| @field(best, f) = std.math.inf(f64);
        for (0..reps + 1) |k| {
            var t: PassTimes = .{};
            if (s.cfg.method == .tree) {
                var cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                var t0 = c.SDL_GetPerformanceCounter();
                cmd = try s.sort.?.record(cmd, .{ .pos = s.pos, .vel = s.vel, .mass = s.mass }, s.cfg.n);
                try submitWait(s.dev, cmd);
                t.sort = sim.msSince(t0);
                cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                t0 = c.SDL_GetPerformanceCounter();
                try s.recordTreeBuild(cmd);
                try submitWait(s.dev, cmd);
                t.tree = sim.msSince(t0);
                for ([2]u32{ 0, L.sim3_flag_potential }) |flags| {
                    cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                    t0 = c.SDL_GetPerformanceCounter();
                    cmd = try s.recordWalk(cmd, 1, flags);
                    try submitWait(s.dev, cmd);
                    if (flags == 0) t.walk = sim.msSince(t0) else t.walk_pot = sim.msSince(t0);
                }
            } else {
                var cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
                const t0 = c.SDL_GetPerformanceCounter();
                cmd = try s.recordKick(cmd, 1, 0);
                try submitWait(s.dev, cmd);
                t.kick = sim.msSince(t0);
            }
            var cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            var t0 = c.SDL_GetPerformanceCounter();
            s.recordDrift(cmd);
            try submitWait(s.dev, cmd);
            t.drift = sim.msSince(t0);
            cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
            t0 = c.SDL_GetPerformanceCounter();
            s.recordDiag(cmd);
            try submitWait(s.dev, cmd);
            t.diag = sim.msSince(t0);
            if (k > 0) inline for (@typeInfo(PassTimes).@"struct".field_names) |f| {
                @field(best, f) = @min(@field(best, f), @field(t, f));
            };
        }
        return best;
    }
};

fn syncHook(ctx: *anyopaque, cmd: *c.SDL_GPUCommandBuffer) *c.SDL_GPUCommandBuffer {
    const s: *Sim3 = @ptrCast(@alignCast(ctx));
    if (s.half_pending) return cmd;
    s.half_pending = true;
    return s.recordForce(cmd, 0.5, 0) catch @panic("sim3: half kick before snapshot failed");
}

fn headerHook(ctx: *anyopaque, seed: u64, scenario: u32) snap.Header {
    const s: *Sim3 = @ptrCast(@alignCast(ctx));
    return snap.Header.init3(.{
        .n = s.cfg.n,
        .tick = s.tick_no,
        .dt = s.cfg.dt,
        .g = s.cfg.g,
        .softening = s.cfg.eps,
        .half_pending = s.half_pending,
        .seed = seed,
        .scenario = scenario,
        .box_size = s.cfg.box.size,
        .boundary = @fromBackingInt(@intCast(@backingInt(s.cfg.boundary))),
    });
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
    method: enum { auto, direct, tree } = .auto,
    theta: f32 = 0.6,
    out: ?[:0]const u8 = null,
    every_snap: u64 = 0,
    restart: ?[:0]const u8 = null,
};

const usage =
    \\usage: sim --dim 3 [--ic plummer|kepler|uniform] [--n N] [--ticks K] [--dt F] [--every K] [--seed U64]
    \\               [--eps LEN] [--ecc E] [--margin M] [--boundary open|periodic] [--method auto|direct|tree]
    \\               [--theta F] [--bench]
    \\               [--out DIR --every-snap K] [--restart FILE]   (restart: n, g, eps, dt, box, boundary come from the file; --ticks = further ticks)
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
            if (!(o.eps > 0)) return error.BadEps;
        } else if (std.mem.eql(u8, a, "--ecc")) {
            o.ecc = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, a, "--margin")) {
            o.margin = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, a, "--boundary")) {
            o.boundary = std.meta.stringToEnum(Boundary, v) orelse return error.BadBoundary;
        } else if (std.mem.eql(u8, a, "--method")) {
            o.method = std.meta.stringToEnum(@TypeOf(o.method), v) orelse return error.BadMethod;
        } else if (std.mem.eql(u8, a, "--out")) {
            o.out = v;
        } else if (std.mem.eql(u8, a, "--every-snap")) {
            o.every_snap = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, a, "--restart")) {
            o.restart = v;
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

    if (o.every_snap > 0 and o.out == null) return error.MissingValue;

    var restart_io: ?*c.SDL_IOStream = null;
    defer if (restart_io) |io| {
        _ = c.SDL_CloseIO(io);
    };
    var header: snap.Header = undefined;
    var bodies: []ic3.Body3 = &.{};
    defer gpa.free(bodies);
    var cfg: Config = undefined;
    var dt = o.dt;
    if (o.restart) |path| {
        const io = try snap.open(path);
        restart_io = io;
        header = try snap.readHeader(io);
        if (header.dim != 3) return error.BadHeader;
        const b: Boundary = @fromBackingInt(@intCast(@backingInt(@as(snap.Boundary, @fromBackingInt(@intCast(header.boundary))))));
        cfg = .{ .n = @intCast(header.n), .g = header.g, .eps = header.softening, .dt = header.dt, .box = .{ .size = header.box_size }, .boundary = b, .theta = o.theta };
        std.debug.print("restart from {s}: {d} bodies at tick {d}\n", .{ path, header.n, header.tick });
    } else {
        var prng = std.Random.DefaultPrng.init(o.seed);
        bodies = switch (o.ic) {
            .plummer => try ic3.plummer(gpa, prng.random(), o.n, .{}),
            .uniform => try ic3.uniform(gpa, prng.random(), o.n, .{ .sigma = 0.3 }),
            .kepler => blk: {
                const k: ic3.Kepler = .{ .e = o.ecc };
                dt = @floatCast(ic3.keplerPeriod(k) / 2000.0);
                const pair = ic3.kepler(k);
                break :blk try gpa.dupe(ic3.Body3, &pair);
            },
        };
        cfg = .{ .n = @intCast(bodies.len), .eps = o.eps, .dt = dt, .box = ic3.fitBox(bodies, o.margin), .boundary = o.boundary, .theta = o.theta };
    }
    const n = cfg.n;
    cfg.method = switch (o.method) {
        .direct => .direct,
        .tree => .tree,
        .auto => if (n > auto_tree_max_direct and cfg.boundary == .open) Method.tree else Method.direct,
    };
    var s = Sim3.init(dev, cfg) catch |err| {
        std.debug.print("sim3 init failed for n={d}: {s} ({s})\n", .{ n, @errorName(err), c.SDL_GetError() });
        return err;
    };
    defer s.deinit();
    if (restart_io) |io| try s.restore(gpa, io, &header) else try s.upload(bodies);
    gpa.free(bodies);
    bodies = &.{};
    std.debug.print("{s}: {d} bodies, {s} (theta {d}), box {d:.4} (quantum {e:.2}), eps {d}, dt {d}, {d:.1} B/body\n", .{ @tagName(o.ic), n, @tagName(cfg.method), cfg.theta, cfg.box.size, cfg.box.size / 0x1p32, cfg.eps, cfg.dt, s.bytesPerBody() });
    if (o.bench) return bench(&s, o);

    var sn: ?*snapper.Snapper = null;
    if (o.every_snap > 0) {
        _ = c.SDL_CreateDirectory(o.out.?.ptr);
        sn = try s.snapshotter(gpa);
    }
    defer if (sn) |p| p.destroy(gpa);
    const d0 = try s.measure();
    std.debug.print("{s:>8} {s:>14} {s:>10} {s:>8} {s:>10} {s:>10} {s:>10} {s:>9}\n", .{ "tick", "E", "dE/E0", "2K/|W|", "|P|", "|L|", "dL/L0", "ms/tick" });
    report(s.tick_no, d0, d0, 0);
    const end = s.tick_no + o.ticks;
    var ms_acc: f64 = 0;
    var ticks_acc: u64 = 0;
    while (s.tick_no < end) {
        const next_snap = if (o.every_snap == 0) std.math.maxInt(u64) else (s.tick_no / o.every_snap + 1) * o.every_snap;
        const step = @min(@min((s.tick_no / o.every + 1) * o.every, next_snap), end) - s.tick_no;
        const t = c.SDL_GetPerformanceCounter();
        try s.run(step);
        ms_acc += sim.msSince(t);
        ticks_acc += step;
        if (s.tick_no % o.every == 0 or s.tick_no == end) {
            report(s.tick_no, try s.measure(), d0, ms_acc / @as(f64, @floatFromInt(ticks_acc)));
            ms_acc = 0;
            ticks_acc = 0;
        }
        if (sn) |p| if (s.tick_no % o.every_snap == 0) {
            var buf: [1024]u8 = undefined;
            try p.write(snap.pathZ(&buf, "{s}/snap_{d:0>8}.snap", .{ o.out.?, s.tick_no }), o.seed, @backingInt(o.ic));
            if (p.writeFailed()) return error.SnapshotWrite;
        };
    }
    if (sn) |p| {
        p.flush();
        if (p.writeFailed()) return error.SnapshotWrite;
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
    std.debug.print("bench n={d} {s}: pipelined {d:.3} ms/tick (min of 3 runs of {d} ticks), {d:.1} B/body\n", .{ s.cfg.n, @tagName(s.cfg.method), best, ticks, s.bytesPerBody() });
    if (s.cfg.method == .tree) {
        std.debug.print("  blocking ms (min of 3): sort {d:.3}  tree {d:.3}  walk {d:.3}  drift {d:.3}  diag {d:.3}  (walk with potential {d:.3}; sum of tick passes {d:.3})\n", .{ pt.sort, pt.tree, pt.walk, pt.drift, pt.diag, pt.walk_pot, pt.sort + pt.tree + pt.walk + pt.drift });
        const st = try s.tree.?.readStatus();
        std.debug.print("  tree: {d} leaves, {d} nodes ({d:.3} per body), capacity {d}, depth {d}, regrows {d}\n", .{ st.leaves, st.nodes, @as(f64, @floatFromInt(st.nodes)) / @as(f64, @floatFromInt(s.cfg.n)), s.tree.?.node_cap, st.max_depth, s.regrows });
    } else {
        std.debug.print("  blocking ms (min of 3): kick {d:.3}  drift {d:.3}  diag {d:.3}\n", .{ pt.kick, pt.drift, pt.diag });
    }
}
