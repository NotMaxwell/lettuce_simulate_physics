//! `zig build render-snap -- <snapshot>... --out <dir> [--size WxH] [--center x,y[,z] --extent e] [--log] [--view xy|xz|yz|yaw,pitch] [--slab t]`
const std = @import("std");
const c = @import("c");
const render = @import("render.zig");
const sim = @import("sim.zig");

const usage =
    \\usage: render-snap <snapshot>... --out <dir> [--size 1920x1080] [--center x,y[,z] --extent e | auto] [--log]
    \\       [--view xy|xz|yz|yaw,pitch] [--slab thickness]
    \\  extent = half of the shorter image side in world units; without --center/--extent both are
    \\  chosen from each snapshot (mean position, 1.05x the 99th percentile radius in the image plane).
    \\  3D snapshots only: --view picks the orthographic projection (xy default; yaw,pitch in degrees,
    \\  0,0 = xy, yaw turns around y, pitch around the turned x); --slab keeps bodies within +-t/2 of
    \\  the view plane through the centre. 2D snapshots ignore both.
    \\
;

pub fn run(gpa: std.mem.Allocator, args: []const [:0]const u8) !void {
    var o: render.Options = .{};
    var out_dir: []const u8 = ".";
    var files: std.ArrayList([:0]const u8) = .empty;
    defer files.deinit(gpa);
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--log")) {
            o.log = true;
        } else if (std.mem.eql(u8, a, "auto")) {
            o.center = null;
            o.extent = null;
        } else if (std.mem.startsWith(u8, a, "--")) {
            i += 1;
            if (i >= args.len) return fail("missing value");
            const v = args[i];
            if (std.mem.eql(u8, a, "--out")) {
                out_dir = v;
            } else if (std.mem.eql(u8, a, "--size")) {
                const x = std.mem.indexOfScalar(u8, v, 'x') orelse return fail("--size WxH");
                o.width = try std.fmt.parseInt(u32, v[0..x], 10);
                o.height = try std.fmt.parseInt(u32, v[x + 1 ..], 10);
            } else if (std.mem.eql(u8, a, "--center")) {
                var ctr: [3]f32 = .{ 0, 0, 0 };
                var parts = std.mem.splitScalar(u8, v, ',');
                var k: usize = 0;
                while (parts.next()) |part| : (k += 1) {
                    if (k >= 3) return fail("--center x,y[,z]");
                    ctr[k] = try std.fmt.parseFloat(f32, part);
                }
                if (k < 2) return fail("--center x,y[,z]");
                o.center = ctr;
            } else if (std.mem.eql(u8, a, "--view")) {
                o.view = render.View.parse(v) catch return fail("--view xy|xz|yz|yaw,pitch");
            } else if (std.mem.eql(u8, a, "--slab")) {
                o.slab = try std.fmt.parseFloat(f32, v);
            } else if (std.mem.eql(u8, a, "--extent")) {
                o.extent = try std.fmt.parseFloat(f32, v);
            } else return fail("unknown option");
        } else try files.append(gpa, a);
    }
    if (files.items.len == 0) return fail("no snapshots given");

    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();
    const dev = try sim.createDevice();
    defer c.SDL_DestroyGPUDevice(dev);
    var r = try render.Renderer.init(dev, o.width, o.height);
    defer r.deinit();
    var dir_buf: [1024]u8 = undefined;
    _ = c.SDL_CreateDirectory(std.fmt.bufPrintSentinel(&dir_buf, "{s}", .{out_dir}, 0) catch return fail("--out too long"));
    for (files.items) |f| {
        var path_buf: [1024]u8 = undefined;
        const stem = std.fs.path.stem(f);
        const out = std.fmt.bufPrintSentinel(&path_buf, "{s}/{s}.png", .{ out_dir, stem }, 0) catch return fail("path too long");
        const res = try r.render(gpa, f, out, o);
        std.debug.print("{s}: {d} bodies, centre ({d:.3}, {d:.3}, {d:.3}), extent {d:.3}, {d:.0} ms -> {s}\n", .{ f, res.n, res.cx, res.cy, res.cz, res.extent, res.render_ms, out });
    }
}

fn fail(msg: []const u8) error{BadArgs} {
    std.debug.print("render-snap: {s}\n{s}", .{ msg, usage });
    return error.BadArgs;
}
