//! Renders one synthetic frame to .scratch/frame.bmp and times Renderer.draw.
const std = @import("std");
const c = @import("c");
const world = @import("world.zig");
const cfg = @import("config.zig");
const render = @import("render.zig");

pub fn main() !void {
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();
    var window: ?*c.SDL_Window = null;
    var sdl: ?*c.SDL_Renderer = null;
    if (!c.SDL_CreateWindowAndRenderer("render-demo", 1029, 720, 0, &window, &sdl)) return error.SdlWindow;
    const gpa = std.heap.smp_allocator;

    var w = world.World.init(gpa, 7);
    defer w.deinit();
    const scale: f32 = 2.2;
    w.bounds = .{ .x = 0, .y = 0, .w = cfg.base_w * scale, .h = cfg.base_h * scale };
    w.prev_bounds = w.bounds;
    w.current_level = 20;

    const rnd = w.prng.random();
    const n_balls: usize = 1500;
    for (0..n_balls) |_| {
        const x = 30 + rnd.float(f32) * (w.bounds.w - 60);
        const y = 30 + rnd.float(f32) * (w.bounds.h - 60);
        try w.spawnBall(x, y);
    }
    const sl = w.balls.slice();
    for (0..sl.len) |i| {
        const b = &sl.items(.trail)[i];
        const vx = sl.items(.vx)[i];
        const vy = sl.items(.vy)[i];
        const x = sl.items(.x)[i];
        const y = sl.items(.y)[i];
        for (0..cfg.max_trail) |k| {
            const back: f32 = @floatFromInt(cfg.max_trail - 1 - k);
            b[k] = .{ .x = x - vx * back * 2, .y = y - vy * back * 2 };
        }
        sl.items(.trail_len)[i] = cfg.max_trail;
        sl.items(.trail_head)[i] = 0;
        sl.items(.prev_x)[i] = x - vx;
        sl.items(.prev_y)[i] = y - vy;
        sl.items(.angle)[i] = rnd.float(f32) * 360;
        sl.items(.prev_angle)[i] = sl.items(.angle)[i] - 3;
        sl.items(.level)[i] = rnd.intRangeAtMost(u8, 1, 38);
        if (i % 7 == 0) sl.items(.temp)[i] = 21 + rnd.float(f32) * 100;
        if (i % 11 == 0) sl.items(.burn)[i] = rnd.float(f32) * 100;
        if (i % 50 == 0) sl.items(.flags)[i].scored = true;
    }

    for (0..300) |_| {
        try w.particles.append(gpa, .{
            .x = 200 + rnd.float(f32) * 1500,
            .y = 200 + rnd.float(f32) * 1000,
            .vx = 0,
            .vy = 0,
            .size = 3 + rnd.float(f32) * 5,
            .age = 0,
            .life = 30,
            .r = 255,
            .g = rnd.intRangeAtMost(u8, 60, 220),
            .b = 0,
        });
    }
    for (0..400) |_| {
        const x = rnd.float(f32) * w.bounds.w;
        const y = rnd.float(f32) * w.bounds.h;
        try w.ashes.append(gpa, .{
            .x = x,
            .y = y,
            .prev_x = x,
            .prev_y = y,
            .vx = 0,
            .vy = 0,
            .gravity = 0,
            .retention = 0,
            .friction = 0,
            .age = rnd.intRangeAtMost(u16, 0, 170),
            .start_red = 255,
        });
    }
    try w.explosions.append(gpa, .{ .x = 700, .y = 500, .size = 300, .age = 20 });

    var lasso: [40]world.Vec2 = undefined;
    for (&lasso, 0..) |*p, i| {
        const a = @as(f32, @floatFromInt(i)) / 39 * 6.0;
        p.* = .{ .x = 1100 + 250 * @cos(a), .y = 800 + 200 * @sin(a) };
    }

    var r = try render.Renderer.init(gpa, sdl.?);
    defer r.deinit();

    const hud: render.Hud = .{
        .balls_per_click = 7,
        .countdown = 87,
        .attraction_energy = 50,
        .lasso = &lasso,
        .lasso_alpha = 0.7,
        .perf = .{ .fps = 60, .tick_ms = 3.1, .render_ms = 0.9 },
    };

    r.draw(&w, 1.0, &hud);
    const surf = c.SDL_RenderReadPixels(sdl.?, null) orelse return error.ReadPixels;
    defer c.SDL_DestroySurface(surf);
    if (!c.SDL_SaveBMP(surf, ".scratch/frame.bmp")) return error.SaveBmp;

    const iters = 300;
    const t0 = c.SDL_GetPerformanceCounter();
    for (0..iters) |i| r.draw(&w, @as(f32, @floatFromInt(i % 10)) / 10.0, &hud);
    const t1 = c.SDL_GetPerformanceCounter();
    const ms = @as(f64, @floatFromInt(t1 - t0)) * 1000.0 / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency())) / iters;
    std.debug.print("{d} balls: mean {d:.3} ms per draw() over {d} calls (CPU, no present/GPU sync)\n", .{ w.balls.len, ms, iters });
}
