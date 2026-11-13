//! GPU Morton sort of the 3D bodies (docs/sim3d.md, WP3): keys from Pos3, a stable LSD radix
//! sort of SortRec ping-ponged between two buffers, then pos/vel/mass gathered into sorted order
//! through the idle SortRec buffer. Kernels in k3/sort.zig. The host side only records passes
//! (buffers, dispatch sizes, push constants), so a CUDA backend can mirror it call for call.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");

const sort_spv = @embedFile("k3_sort.spv");

/// GpuSubmit and GpuAcquire come from the internal submit of a split `record`: by then the caller's
/// command buffer has been submitted, so it must not be used again.
pub const Error = error{ GpuPipeline, GpuBuffer, GpuSubmit, GpuAcquire, OutOfRange };
const Buffer = *c.SDL_GPUBuffer;
const Cmd = *c.SDL_GPUCommandBuffer;

/// Mirror of k3/sort.zig SortParams.
const SortParams = extern struct { n: u32, blocks: u32, shift: u32, pad: u32 = 0 };
const radix_bits = 8;
const radix: u32 = 1 << radix_bits;
/// Records per radix block; k3/sort.zig `block`.
const block: u32 = L.group_size * 8;
/// Digit passes over the 63-bit key; the last covers bits 56..62. Even, so the sorted records end
/// in recs[0] and recs[1] is the idle copy.
const passes = 8;
const dispatch_x: u32 = 32768;
/// Above this many bodies each radix pass is its own submit, so no job nears the driver watchdog
/// even on a slow device (one pass at 16M is a few ms on the RX 7800 XT).
const split_n: u32 = 1 << 24;

const Kernel = enum { keys, histogram, scanRows, scatter, gather12, gather4 };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;

/// The body arrays to put into sorted order. `vel` and `mass` are optional (uniform-mass runs).
pub const Bufs = struct { pos: Buffer, vel: ?Buffer = null, mass: ?Buffer = null };

pub const Sort3 = struct {
    dev: *c.SDL_GPUDevice,
    capacity: u32,
    /// recs[0] holds the sorted records after `record`; recs[1] is scratch.
    recs: [2]Buffer,
    /// radix * blocks digit-major block counts, then radix row totals.
    hist: Buffer,
    pipes: [n_kernels]*c.SDL_GPUComputePipeline,

    pub fn init(dev: *c.SDL_GPUDevice, capacity: u32) Error!Sort3 {
        if (capacity < 1 or capacity > maxCapacity()) return error.OutOfRange;
        var s: Sort3 = undefined;
        s.dev = dev;
        s.capacity = capacity;
        const rec_bytes = @as(u64, capacity) * @sizeOf(L.SortRec);
        s.recs[0] = try makeBuffer(dev, rec_bytes);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.recs[0]);
        s.recs[1] = try makeBuffer(dev, rec_bytes);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.recs[1]);
        s.hist = try makeBuffer(dev, (@as(u64, radix) * blocksFor(capacity) + radix) * 4);
        errdefer c.SDL_ReleaseGPUBuffer(dev, s.hist);

        const Spec = struct { ro: u32, rw: u32 };
        const specs = [n_kernels]Spec{
            .{ .ro = 1, .rw = 1 },
            .{ .ro = 1, .rw = 1 },
            .{ .ro = 0, .rw = 1 },
            .{ .ro = 2, .rw = 1 },
            .{ .ro = 2, .rw = 1 },
            .{ .ro = 2, .rw = 1 },
        };
        var made: [n_kernels]?*c.SDL_GPUComputePipeline = @splat(null);
        errdefer for (made) |m| if (m) |p| c.SDL_ReleaseGPUComputePipeline(dev, p);
        inline for (@typeInfo(Kernel).@"enum".field_names, specs, 0..) |name, sp, i| {
            made[i] = c.SDL_CreateGPUComputePipeline(dev, &.{
                .code_size = sort_spv.len,
                .code = sort_spv.ptr,
                .entrypoint = name,
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = sp.ro,
                .num_readwrite_storage_buffers = sp.rw,
                .num_uniform_buffers = 1,
                .threadcount_x = L.group_size,
                .threadcount_y = 1,
                .threadcount_z = 1,
            });
            if (made[i] == null) {
                std.log.err("sort3 pipeline {s}: {s}", .{ name, c.SDL_GetError() });
                return error.GpuPipeline;
            }
        }
        for (&s.pipes, made) |*p, m| p.* = m.?;
        return s;
    }

    pub fn deinit(s: *Sort3) void {
        for (s.pipes) |p| c.SDL_ReleaseGPUComputePipeline(s.dev, p);
        for ([_]Buffer{ s.recs[0], s.recs[1], s.hist }) |b| c.SDL_ReleaseGPUBuffer(s.dev, b);
        s.* = undefined;
    }

    /// Largest capacity whose SortRec buffer fits one 4 GiB binding.
    pub fn maxCapacity() u32 {
        return @intCast((@as(u64, 1) << 32) / @sizeOf(L.SortRec) - 1);
    }

    /// Device bytes per body this owns: two SortRec copies plus the block histogram.
    pub fn bytesPerBody() f64 {
        return 2 * @sizeOf(L.SortRec) + @as(f64, radix * 4) / block;
    }

    /// Sorted records (key, original index) after `record`.
    pub fn sorted(s: *const Sort3) Buffer {
        return s.recs[0];
    }

    /// The idle SortRec copy; free for other passes (tree build) once `record`'s work is done.
    pub fn scratch(s: *const Sort3) Buffer {
        return s.recs[1];
    }

    /// Records keys, sort and permutation of the first `n` bodies. Large n is split into several
    /// submits (see `split_n`), so the returned command buffer, which the caller submits, may not
    /// be `cmd0`.
    /// On error.GpuSubmit or error.GpuAcquire `cmd0` is already submitted and must not be touched;
    /// on error.OutOfRange nothing was recorded.
    pub fn record(s: *Sort3, cmd0: Cmd, bufs: Bufs, n: u32) Error!Cmd {
        if (n < 1 or n > s.capacity) return error.OutOfRange;
        var cmd = cmd0;
        s.recordKeys(cmd, bufs.pos, n);
        cmd = try s.recordRadix(cmd, n);
        s.recordPermute(cmd, bufs, n);
        return cmd;
    }

    /// recs[0] = (mortonKey(pos[i]), i).
    pub fn recordKeys(s: *Sort3, cmd: Cmd, pos: Buffer, n: u32) void {
        s.pass(cmd, .keys, .{ .n = n, .blocks = 0, .shift = 0 }, &.{pos}, &.{s.recs[0]}, groupsFor(n));
    }

    /// Stable sort of recs[0] by key; the result is back in recs[0].
    pub fn recordRadix(s: *Sort3, cmd0: Cmd, n: u32) Error!Cmd {
        var cmd = cmd0;
        const nb = blocksFor(n);
        for (0..passes) |p| {
            const prm: SortParams = .{ .n = n, .blocks = nb, .shift = @intCast(p * radix_bits) };
            const src = s.recs[p & 1];
            const dst = s.recs[(p & 1) ^ 1];
            s.pass(cmd, .histogram, prm, &.{src}, &.{s.hist}, nb);
            s.pass(cmd, .scanRows, prm, &.{}, &.{s.hist}, radix);
            s.pass(cmd, .scatter, prm, &.{ src, s.hist }, &.{dst}, nb);
            if (n > split_n and p + 1 < passes) {
                if (!c.SDL_SubmitGPUCommandBuffer(cmd)) return error.GpuSubmit;
                cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuAcquire;
            }
        }
        return cmd;
    }

    /// Gathers each body array into recs[1] in sorted order and copies it back.
    pub fn recordPermute(s: *Sort3, cmd: Cmd, bufs: Bufs, n: u32) void {
        const prm: SortParams = .{ .n = n, .blocks = 0, .shift = 0 };
        const tmp = s.recs[1];
        const arrays = [_]struct { buf: ?Buffer, k: Kernel, size: u32 }{
            .{ .buf = bufs.pos, .k = .gather12, .size = 12 },
            .{ .buf = bufs.vel, .k = .gather12, .size = 12 },
            .{ .buf = bufs.mass, .k = .gather4, .size = 4 },
        };
        for (arrays) |a| if (a.buf) |buf| {
            s.pass(cmd, a.k, prm, &.{ s.recs[0], buf }, &.{tmp}, groupsFor(n));
            const cp = c.SDL_BeginGPUCopyPass(cmd);
            // n * 12 < 2^32 because n <= capacity <= maxCapacity() = 2^32 / 12 - 1.
            c.SDL_CopyGPUBufferToBuffer(cp, &.{ .buffer = tmp, .offset = 0 }, &.{ .buffer = buf, .offset = 0 }, n * a.size, false);
            c.SDL_EndGPUCopyPass(cp);
        };
    }

    fn pass(s: *Sort3, cmd: Cmd, k: Kernel, p: SortParams, ro: []const Buffer, rw: []const Buffer, groups: u32) void {
        c.SDL_PushGPUComputeUniformData(cmd, 0, &p, @sizeOf(SortParams));
        var rwb: [1]c.SDL_GPUStorageBufferReadWriteBinding = undefined;
        for (rw, 0..) |b, i| rwb[i] = .{ .buffer = b };
        const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &rwb, @intCast(rw.len));
        c.SDL_BindGPUComputePipeline(cp, s.pipes[@backingInt(k)]);
        if (ro.len > 0) c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(ro.ptr), @intCast(ro.len));
        c.SDL_DispatchGPUCompute(cp, @min(groups, dispatch_x), (groups + dispatch_x - 1) / dispatch_x, 1);
        c.SDL_EndGPUComputePass(cp);
    }
};

fn blocksFor(n: u32) u32 {
    return @intCast((@as(u64, n) + block - 1) / block);
}

fn groupsFor(n: u32) u32 {
    return @intCast((@as(u64, n) + L.group_size - 1) / L.group_size);
}

fn makeBuffer(dev: *c.SDL_GPUDevice, size: u64) Error!Buffer {
    if (size > std.math.maxInt(u32)) return error.OutOfRange;
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = @intCast(@max(size, 16)),
    }) orelse error.GpuBuffer;
}

comptime {
    std.debug.assert(@sizeOf(SortParams) == 16);
    std.debug.assert(radix == L.group_size);
}
