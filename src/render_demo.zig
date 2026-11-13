//! Renders a frame of the real GPU world to .scratch/frame.bmp and times Renderer.draw (CPU side).
const std = @import("std");
const c = @import("c");
const world = @import("world.zig");
const cfg = @import("config.zig");
const render = @import("render.zig");
const gfx = @import("gfx.zig");
const gpu_world = @import("gpu_world.zig");

fn poolView(p: *const gpu_world.Pool, count: u32) render.GpuPool {
    return .{ .pos = p.pos, .prev_pos = p.prev_pos, .body = p.body, .state = p.state, .heat = p.heat, .meta = p.meta, .count = count };
}

fn gpuBalls(gw: *const gpu_world.GpuWorld, max_level: u8) render.GpuBalls {
    const b = gw.ballBuffers();
    const draw = gw.poolDrawCounts();
    return .{
        .pos = b.pos,
        .prev_pos = b.prev_pos,
        .body = b.body,
        .state = b.state,
        .heat = b.heat,
        .meta = b.meta,
        .count = gw.count,
        .alive = gw.alive,
        .queued = gw.pendingSpawns(),
        .max_level = max_level,
        .fire = poolView(&gw.fire, draw.fire),
        .ash = poolView(&gw.ash, draw.ash),
        .pool_tiles = gw.pool_tiles,
        .burning = gw.stats.burning,
        .particles = gw.stats.particles,
        .ashes = gw.stats.ashes,
    };
}

pub fn main() !void {
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();
    const driver: []const u8 = if (c.SDL_GetCurrentVideoDriver()) |d| std.mem.span(d) else "";
    const headless = std.mem.eql(u8, driver, "offscreen");
    const window: ?*c.SDL_Window = if (headless) null else c.SDL_CreateWindow("render-demo", 1029, 720, 0);
    defer if (window) |w| c.SDL_DestroyWindow(w);
    var g = try gfx.Gfx.init(.{ .window = window, .width = 1029, .height = 720, .vsync = false });
    defer g.deinit();
    const gpa = std.heap.smp_allocator;

    var gw = try gpu_world.GpuWorld.init(gpa, g.dev);
    defer gw.deinit();
    var w = world.World.init(gpa, 7);
    defer w.deinit();
    w.current_level = 20;

    for (0..1500) |_| try gw.queueBall(&w, 0, 0);
    gw.snapBounds(&w);
    const rnd = w.prng.random();
    const b = w.bounds;
    for (gw.pending_pos.items, gw.pending.items) |*p, nb| {
        const m = nb.radius + cfg.wall;
        p.* = .{ .x = b.x + m + rnd.float(f32) * (b.w - 2 * m), .y = b.y + m + rnd.float(f32) * (b.h - 2 * m) };
    }
    // Let the balls settle and the fire pools fill before the snapshot.
    gw.ignite_every = 20;
    for (0..240) |_| gw.tick(&w, .{ .merging = false, .random_explosions = false });
    gw.drain(&w);

    var lasso: [40]world.Vec2 = undefined;
    for (&lasso, 0..) |*p, i| {
        const a = @as(f32, @floatFromInt(i)) / 39 * 6.0;
        p.* = .{ .x = 1100 + 250 * @cos(a), .y = 800 + 200 * @sin(a) };
    }

    var r = try render.Renderer.init(gpa, &g);
    defer r.deinit();

    const hud: render.Hud = .{
        .balls_per_click = 7,
        .countdown = 87,
        .attraction_energy = 50,
        .lasso = &lasso,
        .lasso_alpha = 0.7,
        .perf = .{ .fps = 60, .tick_ms = 3.1, .render_ms = 0.9 },
        .gpu = gpuBalls(&gw, w.current_level),
    };

    r.draw(&w, 1.0, &hud);
    r.present(".scratch/frame.bmp");

    const iters = 300;
    var total: u64 = 0;
    for (0..iters) |i| {
        const t0 = c.SDL_GetPerformanceCounter();
        r.draw(&w, @as(f32, @floatFromInt(i % 10)) / 10.0, &hud);
        total += c.SDL_GetPerformanceCounter() - t0;
        r.present(null);
    }
    const ms = @as(f64, @floatFromInt(total)) * 1000.0 / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency())) / iters;
    std.debug.print("{d} balls: mean {d:.3} ms per draw() over {d} calls (CPU recording only, present excluded)\n", .{ gw.alive, ms, iters });
}
