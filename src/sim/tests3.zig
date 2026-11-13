//! Acceptance tests of the 3D sim with direct summation (docs/sim3d.md, work packages 0+2).
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const ic3 = @import("ic3");
const sim = @import("sim.zig");
const sim3 = @import("sim3.zig");

fn device() !*c.SDL_GPUDevice {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    return sim.createDevice();
}

test "sim3: GPU forces on 10k Plummer bodies match an f64 direct sum" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const n = 10_000;
    const bodies = try ic3.plummer(gpa, prng.random(), n, .{});
    defer gpa.free(bodies);
    const eps: f32 = 0.01;
    var s = try sim3.Sim3.init(dev, .{ .n = n, .eps = eps, .dt = 0.01, .box = ic3.fitBox(bodies, 4) });
    defer s.deinit();
    try s.upload(bodies);

    const pos = try gpa.alloc(L.Pos3, n);
    defer gpa.free(pos);
    try s.read(L.Pos3, s.pos, pos);
    // The reference sees the same quantised positions and f32 masses the GPU does.
    const x = try gpa.alloc([3]f64, n);
    defer gpa.free(x);
    const m = try gpa.alloc(f64, n);
    defer gpa.free(m);
    for (x, m, pos, bodies) |*xi, *mi, p, b| {
        xi.* = .{ s.toWorld(p.x, 0), s.toWorld(p.y, 1), s.toWorld(p.z, 2) };
        mi.* = @as(f32, @floatCast(b.m));
    }
    const acc = try gpa.alloc(L.Vel3, n);
    defer gpa.free(acc);
    try s.accelerations(acc);

    var sum_sq: f64 = 0;
    for (0..n) |i| {
        var a: [3]f64 = @splat(0);
        for (0..n) |j| {
            if (i == j) continue;
            const d = [3]f64{ x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2] };
            const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + @as(f64, eps) * eps;
            const w = m[j] / (r2 * @sqrt(r2));
            for (0..3) |k| a[k] += w * d[k];
        }
        const g = [3]f64{ acc[i].x, acc[i].y, acc[i].z };
        var err2: f64 = 0;
        var ref2: f64 = 0;
        for (0..3) |k| {
            err2 += (g[k] - a[k]) * (g[k] - a[k]);
            ref2 += a[k] * a[k];
        }
        sum_sq += err2 / ref2;
    }
    const rms = @sqrt(sum_sq / n);
    std.debug.print("sim3 force rms relative error {e:.3}\n", .{rms});
    try std.testing.expect(rms <= 1e-5);
}

fn keplerRun(dev: *c.SDL_GPUDevice, o: ic3.Kepler, periods: u32, steps_per_period: u32) !struct { s: sim3.Sim3, d0: sim3.Diag3, bodies: [2]ic3.Body3 } {
    const bodies = ic3.kepler(o);
    const dt: f32 = @floatCast(ic3.keplerPeriod(o) / @as(f64, @floatFromInt(steps_per_period)));
    _ = periods;
    var s = try sim3.Sim3.init(dev, .{ .n = 2, .eps = 1e-4, .dt = dt, .box = ic3.fitBox(&bodies, 4) });
    errdefer s.deinit();
    try s.upload(&bodies);
    return .{ .s = s, .d0 = try s.measure(), .bodies = bodies };
}

test "sim3: circular Kepler orbit at 2000 steps per period returns within 1e-3 a" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    var r = try keplerRun(dev, .{ .e = 0 }, 1, 2000);
    defer r.s.deinit();
    try r.s.run(2000);
    var pos: [2]L.Pos3 = undefined;
    try r.s.read(L.Pos3, r.s.pos, &pos);
    var worst: f64 = 0;
    for (pos, r.bodies) |p, b| {
        const d = [3]f64{ r.s.toWorld(p.x, 0) - b.x, r.s.toWorld(p.y, 1) - b.y, r.s.toWorld(p.z, 2) - b.z };
        worst = @max(worst, @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
    }
    std.debug.print("sim3 circular Kepler return error {e:.3} a\n", .{worst});
    try std.testing.expect(worst <= 1e-3);
}

test "sim3: eccentric Kepler orbit (e = 0.5) conserves energy over 10 periods" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    var r = try keplerRun(dev, .{ .e = 0.5 }, 10, 2000);
    defer r.s.deinit();
    var worst: f64 = 0;
    for (0..10) |_| {
        try r.s.run(2000);
        const d = try r.s.measure();
        worst = @max(worst, @abs((d.e() - r.d0.e()) / r.d0.e()));
    }
    std.debug.print("sim3 e=0.5 max |dE/E| over 10 periods {e:.3}\n", .{worst});
    try std.testing.expect(worst <= 1e-3);
}

test "sim3: 10k Plummer sphere stays virialised over 10 crossing times" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const n = 10_000;
    const bodies = try ic3.plummer(gpa, prng.random(), n, .{});
    defer gpa.free(bodies);
    // Henon units: crossing time 2 sqrt(2).
    const dt: f32 = 0.01;
    const ticks: u64 = @intFromFloat(10 * 2 * @sqrt(2.0) / dt);
    var s = try sim3.Sim3.init(dev, .{ .n = n, .eps = 0.01, .dt = dt, .box = ic3.fitBox(bodies, 4) });
    defer s.deinit();
    try s.upload(bodies);
    const d0 = try s.measure();
    try s.run(ticks);
    const d1 = try s.measure();
    std.debug.print("sim3 Plummer 2K/|W| {d:.4} -> {d:.4}, dE/E {e:.2}, escapers {d}\n", .{ d0.virial(), d1.virial(), (d1.e() - d0.e()) / @abs(d0.e()), d1.escaped });
    try std.testing.expect(@abs(d1.virial() - 1) <= 0.05);
}
