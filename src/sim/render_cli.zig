//! `zig build render-snap -- <snapshot>... --out <dir> [--size WxH] [--center x,y --extent e] [--log]`
const std = @import("std");
const c = @import("c");
const render = @import("render.zig");
const sim = @import("sim.zig");

const usage =
    \\usage: render-snap <snapshot>... --out <dir> [--size 1920x1080] [--center x,y --extent e | auto] [--log]
    \\  extent = half of the shorter image side in world units; without --center/--extent both are
    \\  chosen from each snapshot (mean position, 3x rms radius).
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
                const x = std.mem.indexOfScalar(u8, v, ',') orelse return fail("--center x,y");
                o.center = .{ try std.fmt.parseFloat(f32, v[0..x]), try std.fmt.parseFloat(f32, v[x + 1 ..]) };
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
        std.debug.print("{s}: {d} bodies, centre ({d:.1}, {d:.1}), extent {d:.1}, {d:.0} ms -> {s}\n", .{ f, res.n, res.cx, res.cy, res.extent, res.render_ms, out });
    }
}

fn fail(msg: []const u8) error{BadArgs} {
    std.debug.print("render-snap: {s}\n{s}", .{ msg, usage });
    return error.BadArgs;
}
