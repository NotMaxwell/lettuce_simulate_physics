//! Acceptance tests of the GPU Morton sort (docs/sim3d.md, WP3) and its benchmark, which runs only
//! with SORT3_BENCH set (`zig build sort3-bench`).
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const sim = @import("sim.zig");
const ref3 = @import("ref3.zig");
const sort3 = @import("sort3.zig");

const Buffer = *c.SDL_GPUBuffer;

fn device() !*c.SDL_GPUDevice {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    return sim.createDevice();
}

fn makeBuffer(dev: *c.SDL_GPUDevice, size: u64) !Buffer {
    return c.SDL_CreateGPUBuffer(dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = @intCast(@max(size, 16)),
    }) orelse error.GpuBuffer;
}

fn submitWait(dev: *c.SDL_GPUDevice, cmd: *c.SDL_GPUCommandBuffer) !void {
    var f = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSubmit;
    _ = c.SDL_WaitForGPUFences(dev, true, &f, 1);
    c.SDL_ReleaseGPUFence(dev, f);
}

const xfer_chunk: usize = 64 << 20;

fn upload(dev: *c.SDL_GPUDevice, buf: Buffer, bytes: []const u8) !void {
    const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = @intCast(@min(bytes.len, xfer_chunk)) }) orelse return error.GpuBuffer;
    defer c.SDL_ReleaseGPUTransferBuffer(dev, tb);
    var at: usize = 0;
    while (at < bytes.len) : (at += xfer_chunk) {
        const part = bytes[at..@min(at + xfer_chunk, bytes.len)];
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

fn download(dev: *c.SDL_GPUDevice, buf: Buffer, out: []u8) !void {
    const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = @intCast(@min(out.len, xfer_chunk)) }) orelse return error.GpuBuffer;
    defer c.SDL_ReleaseGPUTransferBuffer(dev, tb);
    var at: usize = 0;
    while (at < out.len) : (at += xfer_chunk) {
        const part = out[at..@min(at + xfer_chunk, out.len)];
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

/// Body arrays on the device plus the sorter, sized for `n`.
const Rig = struct {
    dev: *c.SDL_GPUDevice,
    pos: Buffer,
    vel: Buffer,
    mass: Buffer,
    sorter: sort3.Sort3,

    fn init(dev: *c.SDL_GPUDevice, n: u32) !Rig {
        var r: Rig = undefined;
        r.dev = dev;
        r.pos = try makeBuffer(dev, @as(u64, n) * 12);
        r.vel = try makeBuffer(dev, @as(u64, n) * 12);
        r.mass = try makeBuffer(dev, @as(u64, n) * 4);
        r.sorter = try sort3.Sort3.init(dev, n);
        return r;
    }

    fn deinit(r: *Rig) void {
        r.sorter.deinit();
        for ([_]Buffer{ r.pos, r.vel, r.mass }) |b| c.SDL_ReleaseGPUBuffer(r.dev, b);
    }

    fn sort(r: *Rig, n: u32) !void {
        const cmd = c.SDL_AcquireGPUCommandBuffer(r.dev) orelse return error.GpuSubmit;
        try submitWait(r.dev, try r.sorter.record(cmd, .{ .pos = r.pos, .vel = r.vel, .mass = r.mass }, n));
    }
};

const Case = struct { pos: []L.Pos3, vel: []L.Vel3, mass: []f32 };

/// Velocities and masses carry the original index, so the permutation can be checked through them.
fn fillBodies(c_: Case) void {
    for (c_.vel, c_.mass, 0..) |*v, *m, i| {
        const f: f32 = @floatFromInt(i);
        v.* = .{ .x = f, .y = -f, .z = f + 0.5 };
        m.* = f * 0.25;
    }
}

/// Runs the GPU sort on `pos` and checks everything against ref3: keys, records (tie order
/// included), the permutation, and the permuted body arrays.
fn checkCase(gpa: std.mem.Allocator, dev: *c.SDL_GPUDevice, pos: []L.Pos3) !void {
    const n: u32 = @intCast(pos.len);
    const vel = try gpa.alloc(L.Vel3, n);
    defer gpa.free(vel);
    const mass = try gpa.alloc(f32, n);
    defer gpa.free(mass);
    fillBodies(.{ .pos = pos, .vel = vel, .mass = mass });
    var rig = try Rig.init(dev, n);
    defer rig.deinit();
    try upload(dev, rig.pos, std.mem.sliceAsBytes(pos));
    try upload(dev, rig.vel, std.mem.sliceAsBytes(vel));
    try upload(dev, rig.mass, std.mem.sliceAsBytes(mass));
    try rig.sort(n);

    const recs = try gpa.alloc(L.SortRec, n);
    defer gpa.free(recs);
    const pos_out = try gpa.alloc(L.Pos3, n);
    defer gpa.free(pos_out);
    const vel_out = try gpa.alloc(L.Vel3, n);
    defer gpa.free(vel_out);
    const mass_out = try gpa.alloc(f32, n);
    defer gpa.free(mass_out);
    try download(dev, rig.sorter.sorted(), std.mem.sliceAsBytes(recs));
    try download(dev, rig.pos, std.mem.sliceAsBytes(pos_out));
    try download(dev, rig.vel, std.mem.sliceAsBytes(vel_out));
    try download(dev, rig.mass, std.mem.sliceAsBytes(mass_out));

    const ref_pos: []const ref3.Pos3 = @ptrCast(pos);
    const want = try ref3.sortKeys(gpa, ref_pos);
    defer want.deinit(gpa);
    const seen = try gpa.alloc(bool, n);
    defer gpa.free(seen);
    @memset(seen, false);
    for (recs, 0..) |r, i| {
        const w = want.rec(i);
        if (r.key_hi != w.key_hi or r.key_lo != w.key_lo or r.idx != w.idx) {
            std.debug.print("n={d} rec {d}: got {x:0>8}{x:0>8}/{d}, want {x:0>8}{x:0>8}/{d}\n", .{ n, i, r.key_hi, r.key_lo, r.idx, w.key_hi, w.key_lo, w.idx });
            return error.TestUnexpectedResult;
        }
        try std.testing.expect(r.idx < n and !seen[r.idx]);
        seen[r.idx] = true;
        try std.testing.expectEqual(ref3.mortonKey(ref_pos[r.idx]), (@as(u64, r.key_hi) << 32) | r.key_lo);
        try std.testing.expectEqual(pos[r.idx], pos_out[i]);
        try std.testing.expectEqual(vel[r.idx], vel_out[i]);
        try std.testing.expectEqual(mass[r.idx], mass_out[i]);
    }
}

fn randomPos(gpa: std.mem.Allocator, n: usize, seed: u64, distinct: ?u32) ![]L.Pos3 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const pos = try gpa.alloc(L.Pos3, n);
    for (pos, 0..) |*p, i| {
        if (distinct) |d| {
            // Few distinct positions: long runs of equal keys test the tie order.
            const k = r.uintLessThan(u32, d);
            p.* = .{ .x = k *% 0x9E3779B9, .y = k *% 0x85EBCA6B, .z = k *% 0xC2B2AE35 };
        } else if (i > 0 and r.uintLessThan(u32, 8) == 0) {
            p.* = pos[r.uintLessThan(usize, i)];
        } else {
            p.* = .{ .x = r.int(u32), .y = r.int(u32), .z = r.int(u32) };
        }
    }
    return pos;
}

test "sort3: keys, sorted records, tie order and permuted bodies match ref3" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;

    // Block edges (2048 records per radix block), a single body, and heavy ties.
    for ([_]u32{ 1, 2, 255, 2047, 2048, 2049, 5000 }, 0..) |n, k| {
        const pos = try randomPos(gpa, n, 10 + k, if (k % 2 == 0) 7 else null);
        defer gpa.free(pos);
        try checkCase(gpa, dev, pos);
    }
    {
        const pos = try randomPos(gpa, 1_000_003, 1, null);
        defer gpa.free(pos);
        try checkCase(gpa, dev, pos);
    }
    {
        const pos = try randomPos(gpa, 300_000, 2, 1000);
        defer gpa.free(pos);
        try checkCase(gpa, dev, pos);
    }
    {
        const cut = 20.0;
        const pp = try ref3.plummer.positions(gpa, 1 << 20, 4, cut);
        defer gpa.free(pp);
        const pos = try gpa.alloc(L.Pos3, pp.len);
        defer gpa.free(pos);
        for (pos, pp) |*o, p| o.* = .{ .x = ref3.plummer.quantise(p[0], cut), .y = ref3.plummer.quantise(p[1], cut), .z = ref3.plummer.quantise(p[2], cut) };
        try checkCase(gpa, dev, pos);
    }
}

test "sort3: two runs give bit-identical records and bodies" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    const n: u32 = 500_000;
    const pos = try randomPos(gpa, n, 7, 5000);
    defer gpa.free(pos);
    var rig = try Rig.init(dev, n);
    defer rig.deinit();
    var outs: [2][]u8 = undefined;
    for (&outs) |*o| {
        try upload(dev, rig.pos, std.mem.sliceAsBytes(pos));
        try rig.sort(n);
        o.* = try gpa.alloc(u8, n * 24);
        try download(dev, rig.sorter.sorted(), o.*[0 .. n * 12]);
        try download(dev, rig.pos, o.*[n * 12 ..]);
    }
    defer for (outs) |o| gpa.free(o);
    try std.testing.expectEqualSlices(u8, outs[0], outs[1]);
}

/// Exact check without a CPU sort: a stable sort's output is the unique sequence whose (key, idx)
/// pairs are strictly increasing, whose keys are right and whose indices form a bijection.
fn verifyLarge(gpa: std.mem.Allocator, rig: *Rig, pos: []const L.Pos3) !void {
    const n = pos.len;
    const recs = try gpa.alloc(L.SortRec, n);
    defer gpa.free(recs);
    const pos_out = try gpa.alloc(L.Pos3, n);
    defer gpa.free(pos_out);
    try download(rig.dev, rig.sorter.sorted(), std.mem.sliceAsBytes(recs));
    try download(rig.dev, rig.pos, std.mem.sliceAsBytes(pos_out));
    const seen = try gpa.alloc(bool, n);
    defer gpa.free(seen);
    @memset(seen, false);
    var prev_key: u64 = 0;
    var prev_idx: u32 = 0;
    for (recs, pos_out, 0..) |r, po, i| {
        const key = (@as(u64, r.key_hi) << 32) | r.key_lo;
        if (r.idx >= n or seen[r.idx]) return error.NotABijection;
        seen[r.idx] = true;
        const p = pos[r.idx];
        if (key != ref3.mortonKey(.{ .x = p.x, .y = p.y, .z = p.z })) return error.WrongKey;
        if (i > 0 and !(key > prev_key or (key == prev_key and r.idx > prev_idx))) return error.NotSorted;
        if (!std.meta.eql(po, pos[r.idx])) return error.WrongPermutation;
        prev_key = key;
        prev_idx = r.idx;
    }
}

test "sort3 bench" {
    if (std.testing.environ.getPosix("SORT3_BENCH") == null) return error.SkipZigTest;
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.heap.smp_allocator;
    const reps = 5;
    // 70M is checked, not timed: it is the size that puts histogram/scatter on a 2D dispatch.
    const sizes = [_]u32{ 1 << 20, 1 << 22, 1 << 24, 1 << 26, 70_000_000 };
    std.debug.print("\nsort3 bench, uniform random positions, ms (min of {d} after 1 warm-up)\n", .{reps});
    std.debug.print("{s:>10} {s:>12} {s:>12} {s:>14}\n", .{ "n", "keys+sort", "+permute", "presorted+perm" });
    for (sizes) |n| {
        const pos = try randomPos(gpa, n, 99, null);
        defer gpa.free(pos);
        var rig = Rig.init(dev, n) catch |err| {
            std.debug.print("{d:>10} skipped: {s} ({s})\n", .{ n, @errorName(err), c.SDL_GetError() });
            continue;
        };
        defer rig.deinit();
        if (n == 70_000_000) {
            try upload(dev, rig.pos, std.mem.sliceAsBytes(pos));
            try rig.sort(n);
            try verifyLarge(gpa, &rig, pos);
            std.debug.print("{d:>10} verified (2D histogram/scatter dispatch)\n", .{n});
            continue;
        }
        var best_sort: f64 = std.math.inf(f64);
        var best_all: f64 = std.math.inf(f64);
        var best_pre: f64 = std.math.inf(f64);
        for (0..reps + 1) |k| {
            try upload(dev, rig.pos, std.mem.sliceAsBytes(pos));
            var cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
            var t0 = c.SDL_GetPerformanceCounter();
            rig.sorter.recordKeys(cmd, rig.pos, n);
            try submitWait(dev, try rig.sorter.recordRadix(cmd, n));
            const ts = sim.msSince(t0);

            cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
            t0 = c.SDL_GetPerformanceCounter();
            try submitWait(dev, try rig.sorter.record(cmd, .{ .pos = rig.pos, .vel = rig.vel, .mass = rig.mass }, n));
            const ta = sim.msSince(t0);
            if (k == 0) try verifyLarge(gpa, &rig, pos);

            // Bodies are now in Morton order: the steady state of a simulation that sorts every tick.
            cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
            t0 = c.SDL_GetPerformanceCounter();
            try submitWait(dev, try rig.sorter.record(cmd, .{ .pos = rig.pos, .vel = rig.vel, .mass = rig.mass }, n));
            const tp = sim.msSince(t0);
            if (k > 0) {
                best_sort = @min(best_sort, ts);
                best_all = @min(best_all, ta);
                best_pre = @min(best_pre, tp);
            }
        }
        std.debug.print("{d:>10} {d:>12.3} {d:>12.3} {d:>14.3}\n", .{ n, best_sort, best_all, best_pre });
    }
}
