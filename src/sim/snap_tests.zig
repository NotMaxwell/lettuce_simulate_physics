//! Snapshot v2 (3D) format, v1 compatibility and the projecting renderer. Imported from tests.zig.
const std = @import("std");
const c = @import("c");
const sim = @import("sim.zig");
const snap = @import("snapshot.zig");
const render = @import("render.zig");

fn device() !*c.SDL_GPUDevice {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    return sim.createDevice();
}

const Body3 = struct { pos: [3]f64, vel: [3]f32, mass: f32 };

fn toWord(frac: f64) u32 {
    return @intFromFloat(@min(@max(frac, 0), 1) * 4294967295.0);
}

/// Positions are world coordinates relative to the cube centre; mass is dropped when `uniform_mass` is set.
fn writeSynthetic(gpa: std.mem.Allocator, path: [:0]const u8, bodies: []const Body3, box: f64, uniform_mass: ?f32) !snap.Header {
    const h = snap.Header.init3(.{ .n = bodies.len, .tick = 12, .dt = 0.01, .g = 1, .softening = 0.1, .half_pending = true, .seed = 5, .box_size = box, .boundary = .periodic, .uniform_mass = uniform_mass });
    const io = try snap.create(path, &h);
    defer _ = c.SDL_CloseIO(io);
    const words = try gpa.alloc(u32, bodies.len);
    defer gpa.free(words);
    for (0..h.fields_used) |id| {
        for (words, bodies) |*w, b| w.* = switch (id) {
            0...2 => toWord(b.pos[id] / box + 0.5),
            3...5 => @bitCast(b.vel[id - 3]),
            else => @bitCast(b.mass),
        };
        try snap.writeWords(io, &h, @intCast(id), 0, words);
    }
    try snap.writeFooter(io, &h);
    return h;
}

fn loadFile(path: [:0]const u8) ![]const u8 {
    var size: usize = 0;
    const p: [*]const u8 = @ptrCast(c.SDL_LoadFile(path.ptr, &size) orelse return error.FileOpen);
    return p[0..size];
}

fn roundTrip(uniform: ?f32) !void {
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const bodies = try gpa.alloc(Body3, 5000);
    defer gpa.free(bodies);
    for (bodies) |*b| b.* = .{
        .pos = .{ (rnd.float(f64) - 0.5) * 80, (rnd.float(f64) - 0.5) * 80, (rnd.float(f64) - 0.5) * 80 },
        .vel = .{ rnd.floatNorm(f32), rnd.floatNorm(f32), rnd.floatNorm(f32) },
        .mass = uniform orelse 0.5 + rnd.float(f32),
    };
    const path = if (uniform == null) ".scratch/test_v2_mass.snap" else ".scratch/test_v2_uniform.snap";
    const written = try writeSynthetic(gpa, path, bodies, 100, uniform);
    const io = try snap.open(path);
    defer _ = c.SDL_CloseIO(io);
    const h = try snap.readHeader(io);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&written), std.mem.asBytes(&h));
    try std.testing.expectEqual(@as(u32, 3), h.dim);
    try std.testing.expectEqual(@as(f64, 100), h.box_size);
    try std.testing.expectEqual(@backingInt(snap.Boundary.periodic), h.boundary);
    try std.testing.expectEqual(uniform != null, h.hasUniformMass());
    try std.testing.expectEqual(@as(u32, if (uniform != null) 6 else 7), h.fields_used);
    const words = try gpa.alloc(u32, bodies.len);
    defer gpa.free(words);
    for (0..h.fields_used) |id| {
        try snap.readWords(io, &h, @intCast(id), 0, words);
        for (words, bodies) |w, b| try std.testing.expectEqual(switch (id) {
            0...2 => toWord(b.pos[id] / 100 + 0.5),
            3...5 => @as(u32, @bitCast(b.vel[id - 3])),
            else => @as(u32, @bitCast(b.mass)),
        }, w);
    }
    if (uniform) |m| try std.testing.expectEqual(m, h.uniform_mass);
}

test "snapshot v2: header and all seven fields round-trip" {
    try roundTrip(null);
}

test "snapshot v2: uniform mass is stored in the header, not as an array" {
    try roundTrip(0.25);
}

test "snapshot v1: a 2D file still loads with the v2 reader" {
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    const n = 300;
    const h = snap.Header.init(n, 9, 0.01, 1, 0.1, false, 4, 0);
    const path = ".scratch/test_v1.snap";
    {
        const io = try snap.create(path, &h);
        defer _ = c.SDL_CloseIO(io);
        const a = try gpa.alloc(f32, n);
        defer gpa.free(a);
        for (0..snap.field_count) |f| {
            for (a, 0..) |*v, i| v.* = @floatFromInt(i * 10 + f);
            try snap.writeField(io, &h, @fromBackingInt(@intCast(f)), 0, a);
        }
        try snap.writeFooter(io, &h);
    }
    const io = try snap.open(path);
    defer _ = c.SDL_CloseIO(io);
    const r = try snap.readHeader(io);
    try std.testing.expectEqual(snap.version_2d, r.version);
    try std.testing.expectEqual(@as(u32, 2), r.dim);
    try std.testing.expectEqual(@as(u64, 9), r.tick);
    try std.testing.expectEqual(@as(f64, 0), r.box_size);
    var out: [n]f32 = undefined;
    try snap.readField(io, &r, .mass, 0, &out);
    try std.testing.expectEqual(@as(f32, 2 * 10 + 4), out[2]);
}

test "snapshot v2: truncated or footerless files are rejected" {
    const gpa = std.testing.allocator;
    var bodies: [100]Body3 = @splat(.{ .pos = .{ 1, 2, 3 }, .vel = .{ 0, 0, 0 }, .mass = 1 });
    _ = try writeSynthetic(gpa, ".scratch/test_v2_trunc_src.snap", &bodies, 10, null);
    const full = try loadFile(".scratch/test_v2_trunc_src.snap");
    defer c.SDL_free(@constCast(full.ptr));
    for ([_]usize{ full.len - 1, full.len - snap.footer_bytes, snap.header_bytes, 100 }) |len| {
        try std.testing.expect(c.SDL_SaveFile(".scratch/test_v2_trunc.snap", full.ptr, len));
        const io = try snap.open(".scratch/test_v2_trunc.snap");
        defer _ = c.SDL_CloseIO(io);
        if (len < snap.header_bytes) {
            try std.testing.expectError(error.FileIo, snap.readHeader(io));
        } else try std.testing.expectError(error.Truncated, snap.readHeader(io));
    }
}

const Image = struct {
    px: [*]u8,
    w: usize,
    h: usize,

    fn lit(self: Image, x: usize, y: usize) bool {
        const p = self.px[(y * self.w + x) * 3 ..][0..3];
        return p[0] != 0 or p[1] != 0 or p[2] != 0;
    }

    fn deinit(self: Image) void {
        c.stbi_image_free(self.px);
    }
};

fn renderTo(r: *render.Renderer, dev: *c.SDL_GPUDevice, snapshot: [:0]const u8, png_path: [:0]const u8, o: render.Options) !Image {
    _ = dev;
    const gpa = std.testing.allocator;
    _ = try r.render(gpa, snapshot, png_path, o);
    const file = try loadFile(png_path);
    defer c.SDL_free(@constCast(file.ptr));
    var w: c_int = 0;
    var hgt: c_int = 0;
    var comp: c_int = 0;
    const px = c.stbi_load_from_memory(file.ptr, @intCast(file.len), &w, &hgt, &comp, 3) orelse return error.PngDecode;
    return .{ .px = px, .w = @intCast(w), .h = @intCast(hgt) };
}

test "render: a ring in the xz plane is a line in the xy view and a ring in the xz view" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    const n = 20000;
    const bodies = try gpa.alloc(Body3, n);
    defer gpa.free(bodies);
    for (bodies, 0..) |*b, i| {
        const a = std.math.tau * @as(f64, @floatFromInt(i)) / n;
        b.* = .{ .pos = .{ 30 * @cos(a), 0, 30 * @sin(a) }, .vel = .{ 0, 0, 0 }, .mass = 1 };
    }
    _ = try writeSynthetic(gpa, ".scratch/test_ring.snap", bodies, 100, 1);
    // 200x200 px over [-40, 40]: 2.5 px per unit, ring radius 75 px.
    var r = try render.Renderer.init(dev, 200, 200);
    defer r.deinit();
    const base: render.Options = .{ .center = .{ 0, 0, 0 }, .extent = 40, .log = true };

    var o = base;
    o.view = .xy;
    const xy = try renderTo(&r, dev, ".scratch/test_ring.snap", ".scratch/test_ring_xy.png", o);
    defer xy.deinit();
    var rows_lit: usize = 0;
    var min_x: usize = 200;
    var max_x: usize = 0;
    for (0..200) |y| {
        var any = false;
        for (0..200) |x| if (xy.lit(x, y)) {
            any = true;
            min_x = @min(min_x, x);
            max_x = @max(max_x, x);
        };
        rows_lit += @intFromBool(any);
    }
    try std.testing.expect(rows_lit >= 1 and rows_lit <= 2);
    try std.testing.expect(min_x <= 26 and max_x >= 173);

    o.view = .xz;
    const xz = try renderTo(&r, dev, ".scratch/test_ring.snap", ".scratch/test_ring_xz.png", o);
    defer xz.deinit();
    var quadrants: [4]usize = @splat(0);
    var lit: usize = 0;
    for (0..200) |y| for (0..200) |x| if (xz.lit(x, y)) {
        const dx = @as(f32, @floatFromInt(x)) + 0.5 - 100;
        const dy = @as(f32, @floatFromInt(y)) + 0.5 - 100;
        const rad = std.math.hypot(dx, dy);
        try std.testing.expect(rad > 72 and rad < 78);
        quadrants[(@as(usize, @intFromBool(dx > 0)) << 1) | @intFromBool(dy > 0)] += 1;
        lit += 1;
    };
    try std.testing.expect(lit > 300);
    for (quadrants) |q| try std.testing.expect(q > lit / 8);
}

test "render: slab keeps only bodies near the view plane" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    var bodies: [2000]Body3 = undefined;
    for (&bodies, 0..) |*b, i| {
        const a = std.math.tau * @as(f64, @floatFromInt(i)) / bodies.len;
        b.* = .{ .pos = .{ 30 * @cos(a), if (i % 2 == 0) 10 else -10, 30 * @sin(a) }, .vel = .{ 0, 0, 0 }, .mass = 1 };
    }
    _ = try writeSynthetic(gpa, ".scratch/test_slab.snap", &bodies, 100, 1);
    var r = try render.Renderer.init(dev, 100, 100);
    defer r.deinit();
    var o: render.Options = .{ .center = .{ 0, 0, 0 }, .extent = 40, .log = true, .view = .xz, .slab = 5 };
    const thin = try renderTo(&r, dev, ".scratch/test_slab.snap", ".scratch/test_slab.png", o);
    defer thin.deinit();
    for (0..100) |y| for (0..100) |x| try std.testing.expect(!thin.lit(x, y));
    o.slab = 30;
    const thick = try renderTo(&r, dev, ".scratch/test_slab.snap", ".scratch/test_slab.png", o);
    defer thick.deinit();
    var lit: usize = 0;
    for (0..100) |y| for (0..100) |x| {
        lit += @intFromBool(thick.lit(x, y));
    };
    try std.testing.expect(lit > 100);
}

test "render: synthetic Plummer sphere plus disc, views xy and 30,20 (look at .scratch/plummer_*.png)" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.testing.allocator;
    _ = c.SDL_CreateDirectory(".scratch");
    const n = 200_000;
    const bodies = try gpa.alloc(Body3, n);
    defer gpa.free(bodies);
    var prng = std.Random.DefaultPrng.init(8);
    const rnd = prng.random();
    for (bodies, 0..) |*b, i| {
        var p: [3]f64 = undefined;
        if (i % 10 < 4) {
            const u = @max(rnd.float(f64) * 0.97, 1e-6);
            const rad = 4.0 / @sqrt(std.math.pow(f64, u, -2.0 / 3.0) - 1);
            const z = 2 * rnd.float(f64) - 1;
            const phi = std.math.tau * rnd.float(f64);
            const s = @sqrt(1 - z * z);
            p = .{ rad * s * @cos(phi), rad * s * @sin(phi), rad * z };
        } else {
            const rad = -9.0 * @log(1 - rnd.float(f64) * 0.97);
            const phi = std.math.tau * rnd.float(f64) + rad * 0.3;
            p = .{ rad * @cos(phi), rad * @sin(phi), 0.3 * rnd.floatNorm(f64) };
        }
        b.* = .{ .pos = p, .vel = .{ 0, 0, 0 }, .mass = 1 };
    }
    _ = try writeSynthetic(gpa, ".scratch/plummer.snap", bodies, 100, 1);
    var r = try render.Renderer.init(dev, 800, 800);
    defer r.deinit();
    const views = [_]struct { v: render.View, out: [:0]const u8 }{
        .{ .v = .xy, .out = ".scratch/plummer_xy.png" },
        .{ .v = .{ .angles = .{ .yaw = 30, .pitch = 20 } }, .out = ".scratch/plummer_30_20.png" },
    };
    for (views) |v| {
        const img = try renderTo(&r, dev, ".scratch/plummer.snap", v.out, .{ .view = v.v, .log = true });
        img.deinit();
    }
}
