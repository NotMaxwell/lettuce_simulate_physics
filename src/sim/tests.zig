const std = @import("std");
const c = @import("c");
const galaxy = @import("galaxy");
const sim = @import("sim.zig");

fn device() !*c.SDL_GPUDevice {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    return sim.createDevice();
}

/// Reads positions back (tests only; the sim itself never reads bodies back).
fn readPos(s: *sim.Sim, out: []galaxy.Body) !void {
    const size: u32 = @intCast(out.len * 8);
    const tb = c.SDL_CreateGPUTransferBuffer(s.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = size }) orelse return error.GpuBuffer;
    defer c.SDL_ReleaseGPUTransferBuffer(s.dev, tb);
    const cmd = c.SDL_AcquireGPUCommandBuffer(s.dev) orelse return error.GpuSubmit;
    const cp = c.SDL_BeginGPUCopyPass(cmd);
    c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = s.pos, .size = size }, &.{ .transfer_buffer = tb, .offset = 0 });
    c.SDL_EndGPUCopyPass(cp);
    var f = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSubmit;
    _ = c.SDL_WaitForGPUFences(s.dev, true, &f, 1);
    c.SDL_ReleaseGPUFence(s.dev, f);
    const m: [*]const f32 = @ptrCast(@alignCast(c.SDL_MapGPUTransferBuffer(s.dev, tb, false) orelse return error.GpuBuffer));
    defer c.SDL_UnmapGPUTransferBuffer(s.dev, tb);
    for (out, 0..) |*b, i| {
        b.x = m[2 * i];
        b.y = m[2 * i + 1];
    }
}

test "sim: two equal masses on a circular orbit return to their start after one period" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    // Pull g m / r from a mass m at distance 2a gives each body g m / (2a), so v = sqrt(g m / 2)
    // at any radius (2D log potential) and the period is 2 pi a / v.
    const g: f32 = 1;
    const m: f32 = 1;
    const a: f32 = 100;
    const v = @sqrt(g * m / 2);
    const period = 2 * std.math.pi * a / v;
    const ticks: u32 = 2000;
    var bodies = [_]galaxy.Body{
        .{ .x = -a, .y = 0, .vx = 0, .vy = -v, .mass = m, .radius = 1 },
        .{ .x = a, .y = 0, .vx = 0, .vy = v, .mass = m, .radius = 1 },
    };
    var s = try sim.Sim.init(dev, .{ .n = 2, .g = g, .softening = 1e-4, .dt = period / ticks });
    defer s.deinit();
    try s.upload(&bodies);
    const d0 = try s.measure();
    try s.run(ticks);
    // Measured position error after the period: ~2e-3 (a = 100); the bound below is 20x that.
    var end = bodies;
    try readPos(&s, &end);
    for (bodies, end) |b, e| {
        try std.testing.expect(@abs(e.x - b.x) < 0.05);
        try std.testing.expect(@abs(e.y - b.y) < 0.05);
    }
    const d1 = try s.measure();
    try std.testing.expectApproxEqRel(d0.e(), d1.e(), 1e-3);
}

test "sim: a 2000-body disk keeps energy and angular momentum over 2000 ticks" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    var built = try galaxy.build(gpa, prng.random(), .disk, 2000, .{ .x = 0, .y = 0 });
    defer built.deinit(gpa);
    const n: u32 = @intCast(built.bodies.len);
    var s = try sim.Sim.init(dev, .{ .n = n, .g = built.g, .pe_samples = n });
    defer s.deinit();
    try s.upload(built.bodies);
    const d0 = try s.measure();
    try std.testing.expectEqual(n, d0.samples);
    try s.run(2000);
    const d1 = try s.measure();
    // Measured over repeated runs: dE/K0 about 1.1e-4, dL/L0 about 5.5e-5. The bounds are ~10x that,
    // loose enough for the run-to-run noise of the atomic binning, tight enough to catch a broken
    // kick, drift or tree box (those drift by percents within a few hundred ticks).
    try std.testing.expect(@abs(d1.e() - d0.e()) / d0.ke < 1e-3);
    try std.testing.expect(@abs(d1.lz - d0.lz) / d0.lz < 5e-4);
}
