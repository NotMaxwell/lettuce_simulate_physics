//! Host side of the GPU ball solver: owns the ball buffers (gpu_layout SoA, ball-id order),
//! the broad-phase grid buffers and the compute pipelines, and records one simulation tick
//! into a caller-supplied command buffer.
//!
//! Per tick: clear cells -> integrate + bin -> scan (3 levels + 2 adds) -> scatter ids ->
//! gather sorted copies -> collide x iterations (colored Gauss-Seidel in place) -> finish
//! (gather back in id order + walls).
//! Every kernel is its own compute pass because SDL only places barriers between passes.

const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");

pub const Error = error{GpuSetupFailed};

pub const Kernel = enum { integrate, clear, scan, scan_add, scatter, gather, colored, finish };

/// Stages of one tick, in order; `integrate` includes clearing and binning, `scan` is the
/// multi-level prefix sum, `scatter` includes the gather.
pub const Part = enum { integrate, scan, scatter, collide, finish };

const spv = std.enums.EnumArray(Kernel, []const u8).init(.{
    .integrate = @embedFile("gpu_sim_integrate.spv"),
    .clear = @embedFile("gpu_sim_clear.spv"),
    .scan = @embedFile("gpu_sim_scan.spv"),
    .scan_add = @embedFile("gpu_sim_scan_add.spv"),
    .scatter = @embedFile("gpu_sim_scatter.spv"),
    .gather = @embedFile("gpu_sim_gather.spv"),
    .colored = @embedFile("gpu_sim_colored.spv"),
    .finish = @embedFile("gpu_sim_finish.spv"),
});

/// Binding counts per kernel: read-only buffers, read-write buffers, uniform slots.
const Layout = struct { ro: u32, rw: u32, uni: u32 };
const layouts = std.enums.EnumArray(Kernel, Layout).init(.{
    .integrate = .{ .ro = 2, .rw = 8, .uni = 1 },
    .clear = .{ .ro = 0, .rw = 1, .uni = 1 },
    .scan = .{ .ro = 0, .rw = 2, .uni = 1 },
    .scan_add = .{ .ro = 1, .rw = 1, .uni = 1 },
    .scatter = .{ .ro = 2, .rw = 2, .uni = 1 },
    .gather = .{ .ro = 5, .rw = 3, .uni = 1 },
    .colored = .{ .ro = 4, .rw = 4, .uni = 2 },
    .finish = .{ .ro = 6, .rw = 3, .uni = 1 },
});

const scan_block: u32 = L.group_size * 4; // matches common.scan_block
const ScanParams = extern struct { n: u32, _pad0: u32 = 0, _pad1: u32 = 0, _pad2: u32 = 0 };
const ColorParams = extern struct { color_x: u32, color_y: u32, iteration: u32, rest_speed: f32, elastic: u32, _pad0: u32 = 0, _pad1: u32 = 0, _pad2: u32 = 0 };

const Buffer = *c.SDL_GPUBuffer;

pub const Sim = struct {
    dev: *c.SDL_GPUDevice,
    capacity: u32,
    /// Most grid cells the cell buffers hold; `prepare` grows the cell size to stay within it.
    cell_capacity: u32,
    /// true: ball-ball impulses perfectly elastic like the CPU/Python; false: restitution is the
    /// pair's mean retention.
    elastic: bool = true,
    /// Solver sweeps per tick (1 matches the CPU).
    iterations: u32 = 1,
    /// Approach speeds (px/tick) at or below this resolve as resting contacts (inelastic, only
    /// when elastic = false).
    rest_speed: f32 = 2,
    /// Test hook, see SimParams.event_cap.
    event_cap: u32 = 0,

    // Ball buffers, ball-id order (gpu_layout contract). Renderers may bind these read-only.
    pos: Buffer,
    prev_pos: Buffer,
    vel: Buffer,
    body: Buffer,
    state: Buffer,
    meta: Buffer,
    /// Element 0 is the header (kind = count, a = dropped), then up to max_events-1 records.
    events: Buffer,

    // Broad phase and solver scratch.
    counts: Buffer, // cell counts, scanned in place into cell starts
    sums1: Buffer,
    sums2: Buffer,
    sums3: Buffer,
    ball_cell: Buffer,
    ball_slot: Buffer,
    sorted_id: Buffer,
    s_vel: Buffer,
    s_prm: Buffer, // c.PosRad per sorted slot
    s_spin: Buffer, // spin accumulated per sorted slot by the solver, applied in finish

    pipes: std.enums.EnumArray(Kernel, *c.SDL_GPUComputePipeline),

    pub fn init(dev: *c.SDL_GPUDevice, capacity: u32, cell_capacity: u32) Error!Sim {
        std.debug.assert(capacity <= L.max_balls);
        var s: Sim = undefined;
        s.dev = dev;
        s.capacity = capacity;
        s.cell_capacity = cell_capacity;
        s.elastic = true;
        s.iterations = 1;
        s.rest_speed = 2;
        s.event_cap = 0;
        const n = capacity;
        // +2: the dead-ball cell and the end sentinel of the last real cell's row range.
        const ncount = cell_capacity + 2;
        const nsum1 = divCeil(ncount, scan_block);
        const nsum2 = divCeil(nsum1, scan_block);
        std.debug.assert(nsum2 <= scan_block);
        s.pos = try buffer(dev, n * @sizeOf(L.BallPos));
        s.prev_pos = try buffer(dev, n * @sizeOf(L.BallPrevPos));
        s.vel = try buffer(dev, n * @sizeOf(L.BallVel));
        s.body = try buffer(dev, n * @sizeOf(L.BallBody));
        s.state = try buffer(dev, n * @sizeOf(L.BallState));
        s.meta = try buffer(dev, n * @sizeOf(L.BallMeta));
        s.events = try buffer(dev, L.max_events * @sizeOf(L.Event));
        s.counts = try buffer(dev, ncount * 4);
        s.sums1 = try buffer(dev, nsum1 * 4);
        s.sums2 = try buffer(dev, nsum2 * 4);
        s.sums3 = try buffer(dev, 16);
        s.ball_cell = try buffer(dev, n * 4);
        s.ball_slot = try buffer(dev, n * 4);
        s.sorted_id = try buffer(dev, n * 4);
        s.s_vel = try buffer(dev, n * @sizeOf(L.Vec2));
        s.s_spin = try buffer(dev, n * 4);
        s.s_prm = try buffer(dev, n * 16);

        const kernels = comptime std.enums.values(Kernel);
        var infos: [kernels.len]c.SDL_GPUComputePipelineCreateInfo = undefined;
        for (kernels, &infos) |k, *info| {
            const code = spv.get(k);
            const lay = layouts.get(k);
            info.* = .{
                .code_size = code.len,
                .code = code.ptr,
                .entrypoint = "main",
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = lay.ro,
                .num_readwrite_storage_buffers = lay.rw,
                .num_uniform_buffers = lay.uni,
                .threadcount_x = L.group_size,
                .threadcount_y = 1,
                .threadcount_z = 1,
            };
        }
        var made: [kernels.len]?*c.SDL_GPUComputePipeline = undefined;
        if (!createComputePipelines(dev, &infos, &made)) return error.GpuSetupFailed;
        for (kernels, made) |k, p| s.pipes.set(k, p.?);
        return s;
    }

    pub fn deinit(s: *Sim) void {
        for (std.enums.values(Kernel)) |k| c.SDL_ReleaseGPUComputePipeline(s.dev, s.pipes.get(k));
        inline for (.{ "pos", "prev_pos", "vel", "body", "state", "meta", "events", "counts", "sums1", "sums2", "sums3", "ball_cell", "ball_slot", "sorted_id", "s_vel", "s_spin", "s_prm" }) |f|
            c.SDL_ReleaseGPUBuffer(s.dev, @field(s, f));
        s.* = undefined;
    }

    /// Fill in grid dimensions for `p.bounds`, growing `p.cell_size` if the grid would exceed
    /// `cell_capacity` (a coarser grid is still correct, just more candidates per query).
    /// `p.cell_size` must already be >= 2 * the largest radius.
    pub fn prepare(s: *const Sim, p: *L.SimParams) void {
        const w = @max(1.0, p.bounds_max_x - p.bounds_min_x);
        const h = @max(1.0, p.bounds_max_y - p.bounds_min_y);
        var cell = p.cell_size;
        while (true) {
            const nx: u32 = @max(1, @as(u32, @intFromFloat(@ceil(w / cell))));
            const ny: u32 = @max(1, @as(u32, @intFromFloat(@ceil(h / cell))));
            if (@as(u64, nx) * ny <= s.cell_capacity) {
                p.grid_cells_x = nx;
                p.grid_cells_y = ny;
                p.cell_size = cell;
                return;
            }
            cell *= 1.05;
        }
    }

    /// Read-only view of the broad phase, for other GPU passes that need ball neighbours.
    /// Valid after `step` (in the same or a later command buffer) until the next `step`'s first
    /// pass rewrites it; describes the grid of the SimParams that `step` got (grid_cells_x/y,
    /// cell_size, bounds_min_x/y as filled by `prepare`). Cell (cx, cy) has index
    /// cy * grid_cells_x + cx, with cx = floor((x - bounds_min_x) / cell_size) clamped into the
    /// grid; let ncells = grid_cells_x * grid_cells_y.
    /// - `cell_start`: u32[ncells + 2]. Balls of cell c occupy sorted slots
    ///   [cell_start[c], cell_start[c + 1]). Cells of one row are consecutive, so a run of cells
    ///   in a row is the single range [cell_start[c0], cell_start[c1 + 1]). Element ncells starts
    ///   the pseudo-cell holding dead balls (never a neighbour); element ncells + 1 = ball_count.
    /// - `sorted_id`: u32[ball_count], the ball id at each sorted slot.
    /// - `ball_slot`: u32[ball_count], the sorted slot of each ball id (inverse of sorted_id).
    /// Balls are binned at their post-integrate, pre-collision positions, so a ball may sit up
    /// to its solver push outside its cell at the end of the tick; with cell_size >= 2 * max
    /// radius, a 3x3 cell query still finds every pair overlapping at binning time.
    /// Bind them as read-only storage buffers; per-ball data is in the id-order buffers.
    /// Record one tick. `p` must have gone through `prepare`.
    pub fn step(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, p: L.SimParams) void {
        if (p.ball_count == 0) return;
        s.record(cmd, p, .integrate);
        s.record(cmd, p, .scan);
        s.record(cmd, p, .scatter);
        // Ghost mode: no contact response. Binning (integrate/scan/scatter) still ran, so the
        // grid other passes read is valid, and `finish` copies the unchanged sorted state back.
        const ghost = (p.flags & L.sim_collide_mask) >> L.sim_collide_shift == @intFromEnum(L.CollideMode.ghost);
        var it: u32 = 0;
        while (!ghost and it < s.iterations) : (it += 1) s.recordCollide(cmd, p, it);
        s.record(cmd, p, .finish);
    }

    /// Record one stage of the tick. `step` is the normal entry point; this is public so a
    /// profiler can repeat a single stage (SDL_GPU has no timestamp queries). `.collide`
    /// records iteration 0 only (one full sweep for the colored solver).
    pub fn record(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, p: L.SimParams, part: Part) void {
        const groups = divCeil(p.ball_count, L.group_size);
        const ncount = p.grid_cells_x * p.grid_cells_y + 2;
        switch (part) {
            .integrate => {
                s.pushScan(cmd, ncount);
                s.pass(cmd, .clear, &.{}, &.{s.counts}, divCeil(ncount, L.group_size));
                s.push(cmd, 0, &p);
                s.pass(cmd, .integrate, &.{ s.body, s.meta }, &.{ s.pos, s.prev_pos, s.vel, s.state, s.events, s.counts, s.ball_cell, s.ball_slot }, groups);
            },
            .scan => {
                const n1 = divCeil(ncount, scan_block);
                const n2 = divCeil(n1, scan_block);
                s.pushScan(cmd, ncount);
                s.pass(cmd, .scan, &.{}, &.{ s.counts, s.sums1 }, n1);
                if (n1 > 1) {
                    s.pushScan(cmd, n1);
                    s.pass(cmd, .scan, &.{}, &.{ s.sums1, s.sums2 }, n2);
                    if (n2 > 1) {
                        s.pushScan(cmd, n2);
                        s.pass(cmd, .scan, &.{}, &.{ s.sums2, s.sums3 }, 1);
                        s.pushScan(cmd, n1);
                        s.pass(cmd, .scan_add, &.{s.sums2}, &.{s.sums1}, divCeil(n1, L.group_size));
                    }
                    s.pushScan(cmd, ncount);
                    s.pass(cmd, .scan_add, &.{s.sums1}, &.{s.counts}, divCeil(ncount, L.group_size));
                }
            },
            .scatter => {
                s.push(cmd, 0, &p);
                s.pass(cmd, .scatter, &.{ s.ball_cell, s.counts }, &.{ s.sorted_id, s.ball_slot }, groups);
                s.pass(cmd, .gather, &.{ s.pos, s.vel, s.body, s.counts, s.sorted_id }, &.{ s.s_prm, s.s_vel, s.s_spin }, groups);
            },
            .collide => s.recordCollide(cmd, p, 0),
            .finish => {
                s.push(cmd, 0, &p);
                s.pass(cmd, .finish, &.{ s.s_prm, s.s_vel, s.ball_slot, s.body, s.s_spin, s.meta }, &.{ s.pos, s.vel, s.state }, groups);
            },
        }
    }

    fn recordCollide(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, p: L.SimParams, it: u32) void {
        for (0..2) |cy| for (0..2) |cx| {
            const per_row = (p.grid_cells_x -| @as(u32, @intCast(cx)) + 1) / 2;
            const per_col = (p.grid_cells_y -| @as(u32, @intCast(cy)) + 1) / 2;
            if (per_row == 0 or per_col == 0) continue;
            s.push(cmd, 0, &p);
            s.push(cmd, 1, &ColorParams{ .color_x = @intCast(cx), .color_y = @intCast(cy), .iteration = it, .rest_speed = s.rest_speed, .elastic = @intFromBool(s.elastic) });
            s.pass(cmd, .colored, &.{ s.body, s.counts, s.sorted_id, s.meta }, &.{ s.s_prm, s.s_vel, s.s_spin, s.events }, divCeil(per_row * per_col, L.group_size));
        };
    }

    fn push(_: *Sim, cmd: *c.SDL_GPUCommandBuffer, slot: u32, data: anytype) void {
        c.SDL_PushGPUComputeUniformData(cmd, slot, data, @sizeOf(@TypeOf(data.*)));
    }

    fn pushScan(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, n: u32) void {
        s.push(cmd, 0, &ScanParams{ .n = n });
    }

    fn pass(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, k: Kernel, ro: []const Buffer, rw: []const Buffer, groups: u32) void {
        std.debug.assert(ro.len == layouts.get(k).ro and rw.len == layouts.get(k).rw);
        var binds: [8]c.SDL_GPUStorageBufferReadWriteBinding = undefined;
        for (rw, 0..) |b, i| binds[i] = .{ .buffer = b, .cycle = false };
        const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &binds, @intCast(rw.len));
        c.SDL_BindGPUComputePipeline(cp, s.pipes.get(k));
        if (ro.len > 0) c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(ro.ptr), @intCast(ro.len));
        c.SDL_DispatchGPUCompute(cp, groups, 1, 1);
        c.SDL_EndGPUComputePass(cp);
    }

    /// CPU-side ball data for upload/download; any null field is skipped.
    pub const Fields = struct {
        pos: ?[]L.BallPos = null,
        vel: ?[]L.BallVel = null,
        body: ?[]L.BallBody = null,
        state: ?[]L.BallState = null,
        meta: ?[]L.BallMeta = null,
    };

    /// Blocking upload of balls [first, first + len) (len taken from the first non-null field).
    /// Also sets prev_pos = pos for uploaded balls.
    pub fn upload(s: *Sim, first: u32, f: Fields) Error!void {
        try s.transfer(first, f, true);
    }

    /// Blocking readback into the given slices; for tests and tooling, not the frame loop.
    pub fn download(s: *Sim, first: u32, f: Fields) Error!void {
        try s.transfer(first, f, false);
    }

    /// Blocking readback of the start of any buffer (debugging internal state).
    pub fn readBuffer(s: *Sim, buf: Buffer, out: []u8) Error!void {
        const tb = c.SDL_CreateGPUTransferBuffer(s.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = @intCast(out.len) }) orelse return error.GpuSetupFailed;
        defer c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);
        const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSetupFailed;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = buf, .offset = 0, .size = @intCast(out.len) }, &.{ .transfer_buffer = tb, .offset = 0 });
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(s.dev, cmd);
        const m: [*]const u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuSetupFailed);
        @memcpy(out, m[0..out.len]);
        c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
    }

    fn transfer(s: *Sim, first: u32, f: Fields, up: bool) Error!void {
        const Item = struct { buf: Buffer, bytes: []u8 };
        var items: [6]Item = undefined;
        var count: usize = 0;
        inline for (.{ "pos", "vel", "body", "state", "meta" }) |name| {
            if (@field(f, name)) |sl| {
                items[count] = .{ .buf = @field(s, name), .bytes = std.mem.sliceAsBytes(sl) };
                count += 1;
                if (up and std.mem.eql(u8, name, "pos")) {
                    items[count] = .{ .buf = s.prev_pos, .bytes = std.mem.sliceAsBytes(sl) };
                    count += 1;
                }
            }
        }
        if (count == 0) return;
        var total: u32 = 0;
        for (items[0..count]) |it| total += @intCast(it.bytes.len);
        const tb = c.SDL_CreateGPUTransferBuffer(s.dev, &.{
            .usage = if (up) c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD else c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
            .size = total,
        }) orelse return error.GpuSetupFailed;
        defer c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);

        if (up) {
            const m: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuSetupFailed);
            var off: usize = 0;
            for (items[0..count]) |it| {
                @memcpy(m[off..][0..it.bytes.len], it.bytes);
                off += it.bytes.len;
            }
            c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
        }
        const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSetupFailed;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        var off: u32 = 0;
        for (items[0..count]) |it| {
            const elem: u32 = if (it.buf == s.body or it.buf == s.state) 16 else if (it.buf == s.meta) 4 else 8;
            const loc = c.SDL_GPUTransferBufferLocation{ .transfer_buffer = tb, .offset = off };
            const reg = c.SDL_GPUBufferRegion{ .buffer = it.buf, .offset = first * elem, .size = @intCast(it.bytes.len) };
            if (up) c.SDL_UploadToGPUBuffer(cp, &loc, &reg, false) else c.SDL_DownloadFromGPUBuffer(cp, &reg, &loc);
            off += @intCast(it.bytes.len);
        }
        c.SDL_EndGPUCopyPass(cp);
        try submitWait(s.dev, cmd);
        if (!up) {
            const m: [*]const u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuSetupFailed);
            var o: usize = 0;
            for (items[0..count]) |it| {
                @memcpy(it.bytes, m[o..][0..it.bytes.len]);
                o += it.bytes.len;
            }
            c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
        }
    }

    /// Record a copy of the event buffer (header + records) into `dst` (a download transfer
    /// buffer of at least max_events * 16 bytes). Must come after `step` in the same command
    /// buffer, before the next tick's integrate pass resets the counter.
    pub fn copyEvents(s: *Sim, cmd: *c.SDL_GPUCommandBuffer, dst: *c.SDL_GPUTransferBuffer) void {
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.events, .offset = 0, .size = L.max_events * @sizeOf(L.Event) }, &.{ .transfer_buffer = dst, .offset = 0 });
        c.SDL_EndGPUCopyPass(cp);
    }
};

/// Creates the pipelines one after another. The thread-per-pipeline version (6410365) was
/// dropped while chasing a RADV startup crash that turned out to be invalid SPIR-V (see
/// gpu_game/common.zig blockSum); serial startup on RADV measured 370 ms warm / 560 ms cold
/// cache, so the threads were not brought back.
/// Returns false (after releasing any that were made) if one failed.
pub fn createComputePipelines(
    dev: *c.SDL_GPUDevice,
    infos: []const c.SDL_GPUComputePipelineCreateInfo,
    out: []?*c.SDL_GPUComputePipeline,
) bool {
    std.debug.assert(out.len == infos.len);
    @memset(out, null);
    for (infos, out) |*info, *o| {
        o.* = c.SDL_CreateGPUComputePipeline(dev, info);
        if (o.* == null) {
            std.log.err("compute pipeline {s}: {s}", .{ info.entrypoint, c.SDL_GetError() });
            for (out) |q| if (q) |pp| c.SDL_ReleaseGPUComputePipeline(dev, pp);
            return false;
        }
    }
    return true;
}

pub fn submitWait(dev: *c.SDL_GPUDevice, cmd: *c.SDL_GPUCommandBuffer) Error!void {
    var f = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSetupFailed;
    _ = c.SDL_WaitForGPUFences(dev, true, &f, 1);
    c.SDL_ReleaseGPUFence(dev, f);
}

fn buffer(dev: *c.SDL_GPUDevice, size: u32) Error!Buffer {
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE | c.SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = size,
    }) orelse {
        std.log.err("gpu_sim: buffer of {d} bytes: {s}", .{ size, c.SDL_GetError() });
        return error.GpuSetupFailed;
    };
}

fn divCeil(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}
