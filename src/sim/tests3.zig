//! Acceptance tests of the 3D sim with direct summation (docs/sim3d.md, work packages 0+2).
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const ic3 = @import("ic3");
const sim = @import("sim.zig");
const sim3 = @import("sim3.zig");
const snap = @import("snapshot.zig");
const render = @import("render.zig");

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
        xi.* = .{ s.toWorld(p.x), s.toWorld(p.y), s.toWorld(p.z) };
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
    var bodies = ic3.kepler(o);
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
        const d = [3]f64{ r.s.toWorld(p.x) - b.x, r.s.toWorld(p.y) - b.y, r.s.toWorld(p.z) - b.z };
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

test "sim3: softening must be positive" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    try std.testing.expectError(error.OutOfRange, sim3.Sim3.init(dev, .{ .n = 2, .eps = 0, .dt = 0.01, .box = .{ .size = 1 } }));
}

fn plummerSim(dev: *c.SDL_GPUDevice, gpa: std.mem.Allocator, n: u32) !sim3.Sim3 {
    var prng = std.Random.DefaultPrng.init(9);
    const bodies = try ic3.plummer(gpa, prng.random(), n, .{});
    defer gpa.free(bodies);
    var s = try sim3.Sim3.init(dev, .{ .n = n, .eps = 0.02, .dt = 0.01, .box = ic3.fitBox(bodies, 4) });
    errdefer s.deinit();
    try s.upload(bodies);
    return s;
}

fn takeSnapshot(gpa: std.mem.Allocator, s: *sim3.Sim3, path: [:0]const u8) !void {
    const sn = try s.snapshotter(gpa);
    defer sn.destroy(gpa);
    try sn.write(path, 9, 0);
    sn.flush();
    try std.testing.expect(!sn.writeFailed());
}

test "sim3: a restarted run is bit-identical to the continuous run" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    const n = 3000;
    const path = ".scratch/test3_restart.snap";

    var a = try plummerSim(dev, gpa, n);
    defer a.deinit();
    try a.run(60);
    try takeSnapshot(gpa, &a, path);
    try a.run(60);

    const io = try snap.open(path);
    defer _ = c.SDL_CloseIO(io);
    const h = try snap.readHeader(io);
    try std.testing.expectEqual(@as(u64, 60), h.tick);
    var b = try sim3.Sim3.init(dev, .{ .n = n, .g = h.g, .eps = h.softening, .dt = h.dt, .box = .{ .size = h.box_size } });
    defer b.deinit();
    try b.restore(gpa, io, &h);
    try b.run(60);

    try std.testing.expectEqual(a.tick_no, b.tick_no);
    const pa = try gpa.alloc(L.Pos3, n);
    defer gpa.free(pa);
    const pb = try gpa.alloc(L.Pos3, n);
    defer gpa.free(pb);
    const va = try gpa.alloc(L.Vel3, n);
    defer gpa.free(va);
    const vb = try gpa.alloc(L.Vel3, n);
    defer gpa.free(vb);
    try a.read(L.Pos3, a.pos, pa);
    try b.read(L.Pos3, b.pos, pb);
    try a.read(L.Vel3, a.vel, va);
    try b.read(L.Vel3, b.vel, vb);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(pa), std.mem.sliceAsBytes(pb));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(va), std.mem.sliceAsBytes(vb));
}

test "sim3: render-snap of a 3D snapshot gives a PNG with lit pixels" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    var s = try plummerSim(dev, gpa, 5000);
    defer s.deinit();
    try s.run(20);
    try takeSnapshot(gpa, &s, ".scratch/test3_render.snap");
    var r = try render.Renderer.init(dev, 320, 200);
    defer r.deinit();
    _ = try r.render(gpa, ".scratch/test3_render.snap", ".scratch/test3_render.png", .{ .log = true });
    var size: usize = 0;
    const file: [*]const u8 = @ptrCast(c.SDL_LoadFile(".scratch/test3_render.png", &size) orelse return error.FileOpen);
    defer c.SDL_free(@constCast(file));
    var w: c_int = 0;
    var hgt: c_int = 0;
    var comp: c_int = 0;
    const px = c.stbi_load_from_memory(file, @intCast(size), &w, &hgt, &comp, 3) orelse return error.PngDecode;
    defer c.stbi_image_free(px);
    try std.testing.expectEqual(@as(c_int, 320), w);
    var lit: usize = 0;
    for (px[0..@intCast(w * hgt * 3)]) |v| lit += @intFromBool(v != 0);
    try std.testing.expect(lit > 300);
}

fn treeSim(dev: *c.SDL_GPUDevice, gpa: std.mem.Allocator, n: u32, method: sim3.Method, node_div: u32) !sim3.Sim3 {
    var prng = std.Random.DefaultPrng.init(9);
    const bodies = try ic3.plummer(gpa, prng.random(), n, .{});
    defer gpa.free(bodies);
    var s = try sim3.Sim3.init(dev, .{ .n = n, .eps = 0.02, .dt = 0.01, .box = ic3.fitBox(bodies, 4), .method = method, .node_div = node_div });
    errdefer s.deinit();
    try s.upload(bodies);
    return s;
}

fn readState(gpa: std.mem.Allocator, s: *sim3.Sim3) !struct { pos: []L.Pos3, vel: []L.Vel3 } {
    const pos = try gpa.alloc(L.Pos3, s.cfg.n);
    errdefer gpa.free(pos);
    const vel = try gpa.alloc(L.Vel3, s.cfg.n);
    try s.read(L.Pos3, s.pos, pos);
    try s.read(L.Vel3, s.vel, vel);
    return .{ .pos = pos, .vel = vel };
}

test "sim3 tree: first-tick acceleration of 10k Plummer matches direct at theta 0.6" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    const n = 10_000;
    var d = try treeSim(dev, gpa, n, .direct, 4);
    defer d.deinit();
    var t = try treeSim(dev, gpa, n, .tree, 4);
    defer t.deinit();
    const ad = try gpa.alloc(L.Vel3, n);
    defer gpa.free(ad);
    const at = try gpa.alloc(L.Vel3, n);
    defer gpa.free(at);
    const pd = try gpa.alloc(L.Pos3, n);
    defer gpa.free(pd);
    const pt = try gpa.alloc(L.Pos3, n);
    defer gpa.free(pt);
    try d.read(L.Pos3, d.pos, pd);
    try d.accelerations(ad);
    try t.accelerations(at);
    // The tree sim reorders its bodies; match them up by exact quantised position.
    try t.read(L.Pos3, t.pos, pt);
    var index: std.AutoHashMap(L.Pos3, u32) = .init(gpa);
    defer index.deinit();
    for (pd, 0..) |p, i| try index.put(p, @intCast(i));
    var sum_sq: f64 = 0;
    for (pt, at) |p, a| {
        const b = ad[index.get(p).?];
        const dx = [3]f64{ a.x - b.x, a.y - b.y, a.z - b.z };
        const r = [3]f64{ b.x, b.y, b.z };
        sum_sq += (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]) / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    }
    const rms = @sqrt(sum_sq / n);
    std.debug.print("sim3 tree vs direct first-tick rms {e:.3}\n", .{rms});
    try std.testing.expect(rms <= 1.2e-3);
}

test "sim3 tree: 20k Plummer reruns are bit-identical over 50 ticks" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    var a = try treeSim(dev, gpa, 20_000, .tree, 4);
    defer a.deinit();
    var b = try treeSim(dev, gpa, 20_000, .tree, 4);
    defer b.deinit();
    try a.run(50);
    try b.run(50);
    const sa = try readState(gpa, &a);
    defer gpa.free(sa.pos);
    defer gpa.free(sa.vel);
    const sb = try readState(gpa, &b);
    defer gpa.free(sb.pos);
    defer gpa.free(sb.vel);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(sa.pos), std.mem.sliceAsBytes(sb.pos));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(sa.vel), std.mem.sliceAsBytes(sb.vel));
}

test "sim3 tree: a restarted run is bit-identical to the continuous run" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    const n = 5000;
    const path = ".scratch/test3_tree_restart.snap";
    var a = try treeSim(dev, gpa, n, .tree, 4);
    defer a.deinit();
    try a.run(40);
    try takeSnapshot(gpa, &a, path);
    try a.run(40);

    const io = try snap.open(path);
    defer _ = c.SDL_CloseIO(io);
    const h = try snap.readHeader(io);
    var b = try sim3.Sim3.init(dev, .{ .n = n, .g = h.g, .eps = h.softening, .dt = h.dt, .box = .{ .size = h.box_size }, .method = .tree });
    defer b.deinit();
    try b.restore(gpa, io, &h);
    try b.run(40);
    try std.testing.expectEqual(a.tick_no, b.tick_no);
    const sa = try readState(gpa, &a);
    defer gpa.free(sa.pos);
    defer gpa.free(sa.vel);
    const sb = try readState(gpa, &b);
    defer gpa.free(sb.pos);
    defer gpa.free(sb.vel);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(sa.pos), std.mem.sliceAsBytes(sb.pos));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(sa.vel), std.mem.sliceAsBytes(sb.vel));
}

test "sim3 tree: node overflow regrows the tree and the run continues with real forces" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    var s = try treeSim(dev, gpa, 20_000, .tree, 200);
    defer s.deinit();
    try s.run(10);
    try std.testing.expect(s.regrows >= 1);
    try std.testing.expect(s.overflow_ticks >= 1);
    const before = s.overflow_ticks;
    try s.run(10);
    try std.testing.expectEqual(before, s.overflow_ticks);
    const d = try s.measure();
    try std.testing.expect(@abs(d.virial() - 1) <= 0.1);
}

test "sim3 tree: a measurement whose tree overflows is repeated after the regrow" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    var s = try treeSim(dev, gpa, 20_000, .tree, 200);
    defer s.deinit();
    const d = try s.measure();
    try std.testing.expectEqual(@as(u32, 1), s.regrows);
    try std.testing.expect(@abs(d.virial() - 1) <= 0.1);
}
