const std = @import("std");
const c = @import("c");
const galaxy = @import("galaxy");
const sim = @import("sim.zig");
const snap = @import("snapshot.zig");
const snapper = @import("snapper.zig");
const render = @import("render.zig");

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

fn takeSnapshot(gpa: std.mem.Allocator, s: *sim.Sim, path: [:0]const u8) !void {
    const sn = try snapper.Snapper.create(gpa, s);
    defer sn.destroy(gpa);
    try sn.write(path, 7, 0);
    sn.flush();
    try std.testing.expect(!sn.writeFailed());
}

fn loadFile(path: [:0]const u8) ![]const u8 {
    var size: usize = 0;
    const p: [*]const u8 = @ptrCast(c.SDL_LoadFile(path.ptr, &size) orelse return error.FileOpen);
    return p[0..size];
}

fn freeFile(data: []const u8) void {
    c.SDL_free(@constCast(data.ptr));
}

fn diskSim(dev: *c.SDL_GPUDevice, gpa: std.mem.Allocator, n: u32) !struct { s: sim.Sim, built: galaxy.Built } {
    var prng = std.Random.DefaultPrng.init(11);
    var built = try galaxy.build(gpa, prng.random(), .disk, n, .{ .x = 0, .y = 0 });
    errdefer built.deinit(gpa);
    var s = try sim.Sim.init(dev, .{ .n = @intCast(built.bodies.len), .g = built.g, .pe_samples = 64 });
    errdefer s.deinit();
    try s.upload(built.bodies);
    return .{ .s = s, .built = built };
}

test "snapshot: header and arrays round-trip, and match the uploaded bodies at tick 0" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    var made = try diskSim(dev, gpa, 3000);
    defer made.built.deinit(gpa);
    defer made.s.deinit();
    const path = ".scratch/test_roundtrip.snap";
    try takeSnapshot(gpa, &made.s, path);

    const io = try snap.open(path);
    defer _ = c.SDL_CloseIO(io);
    const h = try snap.readHeader(io);
    const bodies = made.built.bodies;
    try std.testing.expectEqual(@as(u64, bodies.len), h.n);
    try std.testing.expectEqual(@as(u64, 0), h.tick);
    try std.testing.expectEqual(@as(u32, 1), h.half_pending);
    try std.testing.expectEqual(made.built.g, h.g);
    try std.testing.expectEqual(made.s.cfg.dt, h.dt);
    try std.testing.expectEqual(@as(u64, 7), h.seed);
    const arr = try gpa.alloc(f32, bodies.len);
    defer gpa.free(arr);
    inline for (.{ .pos_x, .pos_y, .vel_x, .vel_y, .mass }, .{ "x", "y", "vx", "vy", "mass" }) |f, name| {
        try snap.readField(io, &h, f, 0, arr);
        for (arr, bodies) |v, b| try std.testing.expectEqual(@field(b, name), v);
    }

    // Restoring and writing again reproduces the file exactly.
    var s2 = try sim.Sim.init(dev, .{ .n = @intCast(h.n), .g = h.g, .softening = h.softening, .dt = h.dt, .pe_samples = 64 });
    defer s2.deinit();
    try snapper.restore(gpa, &s2, io, &h);
    try std.testing.expectEqual(h.tick, s2.tick_no);
    try takeSnapshot(gpa, &s2, ".scratch/test_roundtrip2.snap");
    const a = try loadFile(path);
    defer freeFile(a);
    const b = try loadFile(".scratch/test_roundtrip2.snap");
    defer freeFile(b);
    try std.testing.expectEqualSlices(u8, a, b);
}

test "snapshot: a restarted run matches the continuous run to solver noise" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    var made = try diskSim(dev, gpa, 3000);
    defer made.built.deinit(gpa);
    defer made.s.deinit();
    try made.s.run(100);
    try takeSnapshot(gpa, &made.s, ".scratch/test_restart_mid.snap");
    try made.s.run(100);
    try takeSnapshot(gpa, &made.s, ".scratch/test_restart_a.snap");

    const io = try snap.open(".scratch/test_restart_mid.snap");
    defer _ = c.SDL_CloseIO(io);
    const h = try snap.readHeader(io);
    try std.testing.expectEqual(@as(u64, 100), h.tick);
    var s2 = try sim.Sim.init(dev, .{ .n = @intCast(h.n), .g = h.g, .softening = h.softening, .dt = h.dt, .pe_samples = 64 });
    defer s2.deinit();
    try snapper.restore(gpa, &s2, io, &h);
    try s2.run(100);
    try takeSnapshot(gpa, &s2, ".scratch/test_restart_b.snap");
    const a = try loadFile(".scratch/test_restart_a.snap");
    defer freeFile(a);
    const b = try loadFile(".scratch/test_restart_b.snap");
    defer freeFile(b);
    try expectSameRun(a, b);
}

test "render-snap: a small snapshot renders to a PNG of the right size with lit pixels" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    var made = try diskSim(dev, gpa, 3000);
    defer made.built.deinit(gpa);
    defer made.s.deinit();
    try takeSnapshot(gpa, &made.s, ".scratch/test_render.snap");
    var r = try render.Renderer.init(dev, 320, 200);
    defer r.deinit();
    _ = try r.render(gpa, ".scratch/test_render.snap", ".scratch/test_render.png", .{ .log = true });
    const file = try loadFile(".scratch/test_render.png");
    defer freeFile(file);
    var w: c_int = 0;
    var hgt: c_int = 0;
    var comp: c_int = 0;
    const px = c.stbi_load_from_memory(file.ptr, @intCast(file.len), &w, &hgt, &comp, 3) orelse return error.PngDecode;
    defer c.stbi_image_free(px);
    try std.testing.expectEqual(@as(c_int, 320), w);
    try std.testing.expectEqual(@as(c_int, 200), hgt);
    var lit: usize = 0;
    for (px[0 .. 320 * 200 * 3]) |v| lit += @intFromBool(v != 0);
    try std.testing.expect(lit > 300);
}

/// Largest position and velocity difference between two snapshot files; the headers must be identical.
const Diffs = struct { pos: f32, vel: f32 };

fn maxDiff(a: []const u8, b: []const u8) !Diffs {
    try std.testing.expectEqual(a.len, b.len);
    try std.testing.expectEqualSlices(u8, a[0..snap.header_bytes], b[0..snap.header_bytes]);
    var h: snap.Header = undefined;
    @memcpy(std.mem.asBytes(&h), a[0..snap.header_bytes]);
    const n: usize = @intCast(h.n);
    var out: Diffs = .{ .pos = 0, .vel = 0 };
    for (0..4) |f| {
        const off: usize = @intCast(h.fields[f].offset);
        const fa = std.mem.bytesAsSlice(f32, a[off..][0 .. n * 4]);
        const fb = std.mem.bytesAsSlice(f32, b[off..][0 .. n * 4]);
        for (fa, fb) |x, y| {
            const d = @abs(x - y);
            if (f < 2) out.pos = @max(out.pos, d) else out.vel = @max(out.vel, d);
        }
    }
    return out;
}

/// The gravity solver is not deterministic from run to run (float atomics in the tree build), so
/// two runs of the same state only agree to rounding noise; see the TODO note on restarts.
fn expectSameRun(a: []const u8, b: []const u8) !void {
    const d = try maxDiff(a, b);
    std.debug.print("restart vs continuous run: max position diff {e:.2}, max velocity diff {e:.2}\n", .{ d.pos, d.vel });
    try std.testing.expect(d.pos < 2e-3);
    try std.testing.expect(d.vel < 1e-4);
}

test "snapshot: truncated or unfinished files are rejected, and no temp file is left behind" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    var made = try diskSim(dev, gpa, 1000);
    defer made.built.deinit(gpa);
    defer made.s.deinit();
    try takeSnapshot(gpa, &made.s, ".scratch/test_trunc.snap");
    try std.testing.expect(!c.SDL_GetPathInfo(".scratch/test_trunc.snap.tmp", null));
    const full = try loadFile(".scratch/test_trunc.snap");
    defer freeFile(full);
    {
        const io = try snap.open(".scratch/test_trunc.snap");
        defer _ = c.SDL_CloseIO(io);
        _ = try snap.readHeader(io);
    }
    for ([_]usize{ full.len - 1, full.len - snap.footer_bytes, snap.header_bytes }) |len| {
        try std.testing.expect(c.SDL_SaveFile(".scratch/test_trunc_cut.snap", full.ptr, len));
        const io = try snap.open(".scratch/test_trunc_cut.snap");
        defer _ = c.SDL_CloseIO(io);
        try std.testing.expectError(error.Truncated, snap.readHeader(io));
    }
}
test { _ = @import("tests3.zig"); }
