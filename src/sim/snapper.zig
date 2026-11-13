//! Asynchronous snapshots of a running `Sim`. The GPU state is downloaded in chunks through a small
//! ring of transfer buffers; a writer thread de-interleaves each chunk into the planar file layout
//! (snapshot.zig) while the main thread downloads the next one, and the last chunks are still being
//! written while the following ticks run. Snapshot n+1 first waits for snapshot n to be on disk.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const galaxy = @import("galaxy");
const sim = @import("sim.zig");
const snap = @import("snapshot.zig");

pub const max_sources = 4;

/// One GPU buffer holding `stride` bytes per body (a multiple of 4), in body order.
pub const Source = struct { buf: *c.SDL_GPUBuffer, stride: u32 };

/// One file array: 4-byte word `word` of every element of source `source`; the file array index
/// is the position in the field list (it must equal the header's field id).
pub const FieldSpec = struct { source: u32, word: u32 };

/// What the Snapper needs from its owner: submit anything pending (returns the command buffer to
/// continue with), then describe the state. The header is made after the sync so tick/half-step are final.
pub const Hooks = struct {
    ctx: *anyopaque,
    recordSync: *const fn (ctx: *anyopaque, cmd: *c.SDL_GPUCommandBuffer) *c.SDL_GPUCommandBuffer,
    makeHeader: *const fn (ctx: *anyopaque, seed: u64, scenario: u32) snap.Header,
};
/// Largest transfer buffer; larger snapshots download in several chunks.
pub const max_transfer_bytes: u64 = 256 << 20;
const slot_count = 3;
/// Bodies the writer thread de-interleaves per file write.
const block = 1 << 18;

pub const Error = sim.Error || snap.Error || error{ OutOfMemory, ThreadSpawn, OutOfRange };

const Job = struct {
    io: *c.SDL_IOStream = undefined,
    header: snap.Header = undefined,
    start: u64 = 0,
    count: u32 = 0,
    slot: u32 = 0,
    last: bool = false,
    quit: bool = false,
};

const Slot = struct {
    tb: *c.SDL_GPUTransferBuffer,
    mapped: ?[*]const u8 = null,
};

pub const Stats = struct {
    bytes: u64 = 0,
    /// ms the simulation was held up (sync kick + downloads + waiting for the previous snapshot).
    stall_ms: f64 = 0,
    /// ms from the start of the snapshot until its last byte was written.
    total_ms: f64 = 0,
};

pub const Snapper = struct {
    dev: *c.SDL_GPUDevice,
    n: u64,
    hooks: Hooks,
    sources: [max_sources]Source,
    source_count: u32,
    fields: [snap.max_fields]FieldSpec,
    field_count: u32,
    bytes_per_body: u32,
    chunk_bodies: u32,
    slots: [slot_count]Slot,
    slots_made: u32,
    jobs: [slot_count]Job = @splat(.{}),
    head: u64 = 0,
    tail: u64 = 0,
    ready: *c.SDL_Semaphore,
    free: *c.SDL_Semaphore,
    thread: std.Thread,
    failed: std.atomic.Value(bool) = .init(false),
    t_start: u64 = 0,
    t_done: std.atomic.Value(u64) = .init(0),
    stats: Stats = .{},
    in_flight: bool = false,
    tmp_path: [1100]u8 = undefined,
    final_path: [1100]u8 = undefined,

    /// 2D sim: positions, velocities and the mass inside `L.BallBody`.
    pub fn create(gpa: std.mem.Allocator, s: *sim.Sim) Error!*Snapper {
        const mass_word = @offsetOf(L.BallBody, "mass") / 4;
        const sources = [_]Source{
            .{ .buf = s.pos, .stride = @sizeOf(L.Vec2) },
            .{ .buf = s.vel, .stride = @sizeOf(L.Vec2) },
            .{ .buf = s.body, .stride = @sizeOf(L.BallBody) },
        };
        const fields = [_]FieldSpec{ .{ .source = 0, .word = 0 }, .{ .source = 0, .word = 1 }, .{ .source = 1, .word = 0 }, .{ .source = 1, .word = 1 }, .{ .source = 2, .word = mass_word } };
        const hooks: Hooks = .{ .ctx = s, .recordSync = syncSim, .makeHeader = headerSim };
        return createGeneric(gpa, s.dev, s.cfg.n, &sources, &fields, hooks);
    }

    /// Generic form: `sources` are the GPU buffers to download, `fields` the file arrays in header
    /// field order (e.g. 3D: pos xyz from a 12-byte source, vel xyz, then mass unless uniform).
    /// Heap-allocated because the writer thread holds a pointer to it.
    pub fn createGeneric(gpa: std.mem.Allocator, dev: *c.SDL_GPUDevice, n: u64, sources: []const Source, fields: []const FieldSpec, hooks: Hooks) Error!*Snapper {
        if (sources.len == 0 or sources.len > max_sources or fields.len == 0 or fields.len > snap.max_fields) return error.BadHeader;
        var bpb: u32 = 0;
        for (sources, 0..) |src, i| {
            if (src.stride == 0 or src.stride % 4 != 0) return error.BadHeader;
            // SDL buffer regions use u32 offsets.
            if (n * src.stride > std.math.maxInt(u32)) {
                std.debug.print("snapshot: source {d} is {d} B x {d} bodies = {d} B, over SDL_GPU's 4 GiB buffer-offset limit; cannot snapshot this many bodies\n", .{ i, src.stride, n, n * src.stride });
                return error.OutOfRange;
            }
            bpb += src.stride;
        }
        for (fields) |f| if (f.source >= sources.len or (f.word + 1) * 4 > sources[f.source].stride) return error.BadHeader;
        const self = try gpa.create(Snapper);
        errdefer gpa.destroy(self);
        self.* = .{
            .dev = dev,
            .n = n,
            .hooks = hooks,
            .sources = undefined,
            .source_count = @intCast(sources.len),
            .fields = undefined,
            .field_count = @intCast(fields.len),
            .bytes_per_body = bpb,
            .chunk_bodies = @intCast(@min(n, max_transfer_bytes / bpb)),
            .slots = undefined,
            .slots_made = 0,
            .ready = c.SDL_CreateSemaphore(0) orelse return error.GpuBuffer,
            .free = c.SDL_CreateSemaphore(slot_count) orelse return error.GpuBuffer,
            .thread = undefined,
        };
        @memcpy(self.sources[0..sources.len], sources);
        @memcpy(self.fields[0..fields.len], fields);
        const chunks = (n + self.chunk_bodies - 1) / self.chunk_bodies;
        const need = @min(chunks, slot_count);
        errdefer for (self.slots[0..self.slots_made]) |sl| c.SDL_ReleaseGPUTransferBuffer(dev, sl.tb);
        while (self.slots_made < need) : (self.slots_made += 1) {
            const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{
                .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
                .size = @intCast(@as(u64, self.chunk_bodies) * bpb),
            }) orelse return error.GpuBuffer;
            self.slots[self.slots_made] = .{ .tb = tb };
        }
        self.thread = std.Thread.spawn(.{}, writerMain, .{self}) catch return error.ThreadSpawn;
        return self;
    }

    pub fn destroy(self: *Snapper, gpa: std.mem.Allocator) void {
        self.flush();
        self.jobs[self.head % slot_count] = .{ .quit = true };
        self.head += 1;
        c.SDL_SignalSemaphore(self.ready);
        self.thread.join();
        for (self.slots[0..self.slots_made]) |*sl| {
            if (sl.mapped != null) c.SDL_UnmapGPUTransferBuffer(self.dev, sl.tb);
            c.SDL_ReleaseGPUTransferBuffer(self.dev, sl.tb);
        }
        c.SDL_DestroySemaphore(self.ready);
        c.SDL_DestroySemaphore(self.free);
        gpa.destroy(self);
    }

    /// Blocks until the previous snapshot is completely written.
    pub fn flush(self: *Snapper) void {
        if (!self.in_flight) return;
        for (0..slot_count) |_| c.SDL_WaitSemaphore(self.free);
        for (0..slot_count) |_| c.SDL_SignalSemaphore(self.free);
        self.in_flight = false;
        const done = self.t_done.load(.acquire);
        self.stats.total_ms = ticksToMs(done -% self.t_start);
    }

    /// Statistics of the last snapshot; total_ms is valid once `flush` has returned.
    pub fn lastStats(self: *Snapper) Stats {
        return self.stats;
    }

    pub fn writeFailed(self: *const Snapper) bool {
        return self.failed.load(.acquire);
    }

    /// Takes a snapshot now (applies the pending half kick first) and returns once all data is
    /// downloaded; file writing continues in the background. The data goes to `<path>.tmp` and is
    /// renamed to `path` only after the footer is written, so `path` never holds a partial file.
    pub fn write(self: *Snapper, path: [:0]const u8, seed: u64, scenario: u32) Error!void {
        const t0 = c.SDL_GetPerformanceCounter();
        self.flush();
        if (self.failed.load(.acquire)) return error.FileIo;
        self.t_start = t0;
        var cmd = c.SDL_AcquireGPUCommandBuffer(self.dev) orelse return error.GpuSubmit;
        // The synchronised state must be on the GPU before the first download is recorded; it is
        // the same command buffer when the solver does not split submits.
        cmd = self.hooks.recordSync(self.hooks.ctx, cmd);
        const header = self.hooks.makeHeader(self.hooks.ctx, seed, scenario);
        if (header.n != self.n or header.fields_used != self.field_count) return error.BadHeader;
        const tmp = std.fmt.bufPrintSentinel(&self.tmp_path, "{s}.tmp", .{path}, 0) catch return error.FileOpen;
        _ = std.fmt.bufPrintSentinel(&self.final_path, "{s}", .{path}, 0) catch return error.FileOpen;
        const io = try snap.create(tmp, &header);
        self.in_flight = true;
        var held = false;
        self.writeChunks(io, header, &cmd, &held) catch |err| {
            if (held) c.SDL_SignalSemaphore(self.free);
            self.abortWrite(io);
            return err;
        };
        self.stats = .{ .bytes = header.completeSize(), .stall_ms = ticksToMs(c.SDL_GetPerformanceCounter() - t0) };
    }

    fn writeChunks(self: *Snapper, io: *c.SDL_IOStream, header: snap.Header, cmd: *(*c.SDL_GPUCommandBuffer), held: *bool) Error!void {
        const dev = self.dev;
        const n = self.n;
        var at: u64 = 0;
        var k: u64 = 0;
        var first = true;
        while (at < n) : (k += 1) {
            const count: u32 = @intCast(@min(n - at, self.chunk_bodies));
            const slot_i: u32 = @intCast(k % slot_count);
            const slot = &self.slots[slot_i];
            c.SDL_WaitSemaphore(self.free);
            held.* = true;
            if (slot.mapped != null) {
                c.SDL_UnmapGPUTransferBuffer(dev, slot.tb);
                slot.mapped = null;
            }
            if (!first) cmd.* = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
            first = false;
            const cp = c.SDL_BeginGPUCopyPass(cmd.*);
            var off: u32 = 0;
            for (self.sources[0..self.source_count]) |p| {
                const bytes: u32 = count * p.stride;
                // createGeneric bounds n * stride, so every offset fits SDL's u32.
                c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = p.buf, .offset = @intCast(at * p.stride), .size = bytes }, &.{ .transfer_buffer = slot.tb, .offset = off });
                off += bytes;
            }
            c.SDL_EndGPUCopyPass(cp);
            const f = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd.*) orelse return error.GpuSubmit;
            var ff = f;
            _ = c.SDL_WaitForGPUFences(dev, true, &ff, 1);
            c.SDL_ReleaseGPUFence(dev, f);
            slot.mapped = @ptrCast(c.SDL_MapGPUTransferBuffer(dev, slot.tb, false) orelse return error.GpuBuffer);
            self.jobs[self.head % slot_count] = .{ .io = io, .header = header, .start = at, .count = count, .slot = slot_i, .last = at + count == n };
            self.head += 1;
            held.* = false;
            c.SDL_SignalSemaphore(self.ready);
            at += count;
        }
    }

    /// Error path: lets queued chunks finish, then drops the incomplete temp file.
    fn abortWrite(self: *Snapper, io: *c.SDL_IOStream) void {
        for (0..slot_count) |_| c.SDL_WaitSemaphore(self.free);
        for (0..slot_count) |_| c.SDL_SignalSemaphore(self.free);
        _ = c.SDL_CloseIO(io);
        _ = c.SDL_RemovePath(@ptrCast(&self.tmp_path));
        self.in_flight = false;
    }
};

fn ticksToMs(t: u64) f64 {
    return @as(f64, @floatFromInt(t)) * 1000.0 / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency()));
}

fn writerMain(self: *Snapper) void {
    var arrays: [snap.max_fields][]u32 = undefined;
    const gpa = std.heap.smp_allocator;
    const mem = gpa.alloc(u32, block * self.field_count) catch {
        self.failed.store(true, .release);
        return;
    };
    defer gpa.free(mem);
    for (arrays[0..self.field_count], 0..) |*a, i| a.* = mem[i * block ..][0..block];
    while (true) {
        c.SDL_WaitSemaphore(self.ready);
        const job = self.jobs[self.tail % slot_count];
        self.tail += 1;
        if (job.quit) return;
        writeChunk(self, job, &arrays) catch self.failed.store(true, .release);
        if (job.last) {
            finish(self, job);
            self.t_done.store(c.SDL_GetPerformanceCounter(), .release);
        }
        c.SDL_SignalSemaphore(self.free);
    }
}

/// Footer, flush, close, then the atomic rename that publishes the file.
fn finish(self: *Snapper, job: Job) void {
    var ok = true;
    snap.writeFooter(job.io, &job.header) catch {
        ok = false;
    };
    if (!c.SDL_FlushIO(job.io)) ok = false;
    if (!c.SDL_CloseIO(job.io)) ok = false;
    if (ok and !c.SDL_RenamePath(@ptrCast(&self.tmp_path), @ptrCast(&self.final_path))) ok = false;
    if (!ok) {
        _ = c.SDL_RemovePath(@ptrCast(&self.tmp_path));
        self.failed.store(true, .release);
    }
}

fn writeChunk(self: *Snapper, job: Job, arrays: *[snap.max_fields][]u32) snap.Error!void {
    const base = self.slots[job.slot].mapped.?;
    const count: usize = job.count;
    var src_base: [max_sources][*]const u32 = undefined;
    var off: usize = 0;
    for (self.sources[0..self.source_count], 0..) |p, i| {
        src_base[i] = @ptrCast(@alignCast(base + off));
        off += count * p.stride;
    }
    var i: usize = 0;
    while (i < count) : (i += block) {
        const m = @min(block, count - i);
        for (self.fields[0..self.field_count], 0..) |f, k| {
            const words = self.sources[f.source].stride / 4;
            const src = src_base[f.source];
            for (arrays[k][0..m], 0..) |*o, j| o.* = src[(i + j) * words + f.word];
            try snap.writeWords(job.io, &job.header, @intCast(k), job.start + i, arrays[k][0..m]);
        }
    }
}

fn syncSim(ctx: *anyopaque, cmd: *c.SDL_GPUCommandBuffer) *c.SDL_GPUCommandBuffer {
    const s: *sim.Sim = @ptrCast(@alignCast(ctx));
    return s.recordSync(cmd);
}

fn headerSim(ctx: *anyopaque, seed: u64, scenario: u32) snap.Header {
    const s: *sim.Sim = @ptrCast(@alignCast(ctx));
    return snap.Header.init(s.cfg.n, s.tick_no, s.cfg.dt, s.cfg.g, s.cfg.softening, s.half_pending, seed, scenario);
}

/// Loads a snapshot into `s` (n and physics constants must already match the header) and sets the clock.
pub fn restore(gpa: std.mem.Allocator, s: *sim.Sim, io: *c.SDL_IOStream, h: *const snap.Header) Error!void {
    if (h.version != snap.version_2d or h.dim != 2 or h.n != s.cfg.n) return error.BadHeader;
    const chunk: usize = 1 << 20;
    const u = try s.uploaderInit(chunk);
    defer s.uploaderDeinit(u);
    const bodies = try gpa.alloc(galaxy.Body, chunk);
    defer gpa.free(bodies);
    const f = try gpa.alloc(f32, chunk * snap.field_count);
    defer gpa.free(f);
    var at: usize = 0;
    while (at < s.cfg.n) : (at += chunk) {
        const m = @min(chunk, s.cfg.n - at);
        for (0..snap.field_count) |k| try snap.readField(io, h, @fromBackingInt(@intCast(k)), at, f[k * chunk ..][0..m]);
        for (bodies[0..m], 0..) |*b, j| b.* = .{
            .x = f[j],
            .y = f[chunk + j],
            .vx = f[2 * chunk + j],
            .vy = f[3 * chunk + j],
            .mass = f[4 * chunk + j],
            .radius = 0,
        };
        try s.uploadPart(u, at, bodies[0..m]);
    }
    s.setClock(h.tick, h.half_pending != 0);
}
