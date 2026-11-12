const std = @import("std");
const c = @import("c");
const cfg = @import("config.zig");
const world = @import("world.zig");
const physics = @import("physics.zig");
const render = @import("render.zig");
const audio = @import("audio.zig");

const Vec2 = world.Vec2;
const tick_seconds: f64 = 1.0 / @as(f64, cfg.tick_hz);
/// Catch-up cap: after a long stall, drop simulation time instead of spiralling.
const max_ticks_per_frame = 4;
const stress_count = 1000;

const Options = struct {
    /// Exit after this many frames (0 = run until closed). For headless verification.
    frames: u32 = 0,
    stress: u32 = 0,
    screenshot: ?[:0]const u8 = null,
};

const Game = struct {
    gpa: std.mem.Allocator,
    w: world.World,
    renderer: render.Renderer,
    sound: audio.Audio,
    sdl: *c.SDL_Renderer,

    balls_per_click: u32 = 1,
    /// A click's balls are spread over several ticks, as in the original.
    spawn_ticks_left: u32 = 0,
    spawn_per_tick: u32 = 0,
    spawn_remainder: u32 = 0,

    mutual_gravity: bool = true,
    merging: bool = true,
    muted: bool = false,
    show_perf: bool = true,
    won: bool = false,
    countdown: i32 = cfg.countdown_seconds,
    attraction_energy: f32 = cfg.attraction_energy_max,
    second_accum: f64 = 0,

    lasso_active: bool = false,
    lasso: std.ArrayList(Vec2) = .empty,
    lasso_fade_left_ms: f32 = 0,

    mouse: Vec2 = .{},
    perf: render.Perf = .{},

    fn attracting(g: *const Game) bool {
        var mx: f32 = 0;
        var my: f32 = 0;
        const buttons = c.SDL_GetMouseState(&mx, &my);
        const keys = c.SDL_GetKeyboardState(null);
        return (buttons & c.SDL_BUTTON_LMASK) != 0 and keys[c.SDL_SCANCODE_G] and g.attraction_energy > 0;
    }

    fn startSpawn(g: *Game) void {
        if (g.spawn_ticks_left > 0) return;
        const n = g.balls_per_click;
        const ticks: u32 = @max(1, @as(u32, @intFromFloat(std.math.cbrt(@as(f32, @floatFromInt(n))))));
        g.spawn_ticks_left = ticks;
        g.spawn_per_tick = n / ticks;
        g.spawn_remainder = n - g.spawn_per_tick * ticks;
    }

    fn onKey(g: *Game, key: c.SDL_Keycode, mods: c.SDL_Keymod) void {
        const step: u32 = if (mods & c.SDL_KMOD_CTRL != 0) 25 else 1;
        switch (key) {
            c.SDLK_M => {
                g.muted = !g.muted;
                g.sound.setMusicMuted(g.muted);
            },
            c.SDLK_F => g.mutual_gravity = !g.mutual_gravity,
            c.SDLK_N => g.merging = !g.merging,
            c.SDLK_P => g.show_perf = !g.show_perf,
            c.SDLK_T => g.stress(stress_count),
            c.SDLK_PLUS, c.SDLK_EQUALS, c.SDLK_KP_PLUS => g.balls_per_click += step,
            c.SDLK_MINUS, c.SDLK_KP_MINUS => g.balls_per_click = @max(1, g.balls_per_click -| step),
            else => {},
        }
    }

    /// Showcase mode: merging off so the population is sustained instead of collapsing.
    fn stress(g: *Game, count: u32) void {
        g.merging = false;
        const b = g.w.bounds;
        physics.spawnBurst(&g.w, b.x + b.w / 2, b.y + b.h / 2, count) catch {};
    }

    fn onMouseDown(g: *Game, button: u8, p: Vec2) void {
        const shift = c.SDL_GetModState() & c.SDL_KMOD_SHIFT != 0;
        if (button == c.SDL_BUTTON_LEFT) {
            if (shift) {
                g.lasso_active = true;
                g.lasso_fade_left_ms = 0;
                g.lasso.clearRetainingCapacity();
                g.lasso.append(g.gpa, p) catch {};
            } else if (physics.pickBall(&g.w, p)) |i| {
                physics.clickBall(&g.w, i);
            } else {
                g.startSpawn();
            }
        } else if (button == c.SDL_BUTTON_RIGHT and !g.lasso_active) {
            if (physics.pickBall(&g.w, p)) |i| physics.ignite(&g.w, i);
        }
    }

    fn onMouseUp(g: *Game, button: u8) void {
        if (button != c.SDL_BUTTON_LEFT) return;
        physics.releaseGrabs(&g.w);
        if (!g.lasso_active) return;
        g.lasso_active = false;
        g.lasso_fade_left_ms = cfg.lasso_fade_ms;
        if (g.lasso.items.len >= 3) physics.lassoExplode(&g.w, g.lasso.items) catch {};
    }

    fn everySecond(g: *Game) void {
        if (!g.won and g.countdown > 0) g.countdown -= 1;
        if (g.attracting()) {
            g.attraction_energy = @max(0, g.attraction_energy - 25);
        } else if (g.attraction_energy < cfg.attraction_energy_max) {
            g.attraction_energy = @min(cfg.attraction_energy_max, g.attraction_energy + 10);
        }
    }

    fn tick(g: *Game) void {
        if (g.spawn_ticks_left > 0) {
            const extra = if (g.spawn_ticks_left == 1) g.spawn_remainder else 0;
            physics.spawnBurst(&g.w, g.mouse.x, g.mouse.y, g.spawn_per_tick + extra) catch {};
            g.spawn_ticks_left -= 1;
        }

        physics.tick(&g.w, .{
            .mouse = g.mouse,
            .attract = g.attracting(),
            .mutual_gravity = g.mutual_gravity,
            .merging = g.merging,
        });

        const ev = g.w.events;
        if (ev.merges > 0) g.sound.play(.hit);
        if (ev.deaths > 0) g.sound.play(.death);
        if (ev.explosions > 0) g.sound.play(.explosion);
        if (ev.any_burning) g.sound.play(.damage);
        if (ev.won) g.won = true;

        g.second_accum += tick_seconds;
        if (g.second_accum >= 1.0) {
            g.second_accum -= 1.0;
            g.everySecond();
        }
    }
};

fn seconds(counter: u64) f64 {
    return @as(f64, @floatFromInt(counter)) / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency()));
}

fn smooth(old: f32, new: f32) f32 {
    return if (old == 0) new else old + (new - old) * 0.1;
}

fn parseOptions(args: []const [:0]const u8) Options {
    var o: Options = .{};
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const flag = args[i];
        const value = args[i + 1];
        if (std.mem.eql(u8, flag, "--frames")) {
            o.frames = std.fmt.parseInt(u32, value, 10) catch 0;
        } else if (std.mem.eql(u8, flag, "--stress")) {
            o.stress = std.fmt.parseInt(u32, value, 10) catch 0;
        } else if (std.mem.eql(u8, flag, "--screenshot")) {
            o.screenshot = value;
        }
    }
    return o;
}

fn saveScreenshot(sdl: *c.SDL_Renderer, path: [:0]const u8) void {
    const surface = c.SDL_RenderReadPixels(sdl, null) orelse return;
    defer c.SDL_DestroySurface(surface);
    _ = c.SDL_SaveBMP(surface, path.ptr);
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const opts = parseOptions(try init.minimal.args.toSlice(init.arena.allocator()));

    // Audio is not initialised here: Audio.init brings it up itself and degrades to silence, so a
    // machine with no sound backend (e.g. WSL without libpulse) still runs the game.
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) {
        std.debug.print("SDL_Init failed: {s}\n", .{c.SDL_GetError()});
        return error.SdlInit;
    }
    defer c.SDL_Quit();

    var window: ?*c.SDL_Window = null;
    var sdl_opt: ?*c.SDL_Renderer = null;
    if (!c.SDL_CreateWindowAndRenderer("Lettuce Simulate Physics", cfg.base_w, cfg.base_h, c.SDL_WINDOW_RESIZABLE, &window, &sdl_opt)) {
        std.debug.print("SDL window/renderer failed: {s}\n", .{c.SDL_GetError()});
        return error.SdlWindow;
    }
    const sdl = sdl_opt.?;
    defer c.SDL_DestroyRenderer(sdl);
    defer c.SDL_DestroyWindow(window);
    _ = c.SDL_SetRenderVSync(sdl, 1);

    var g: Game = .{
        .gpa = gpa,
        .w = .init(gpa, c.SDL_GetPerformanceCounter()),
        .renderer = try .init(gpa, sdl),
        .sound = .init(gpa),
        .sdl = sdl,
    };
    if (!g.sound.isActive()) std.debug.print("audio unavailable: no SDL audio backend could open a device (Linux needs libpulse or libasound); running silent\n", .{});
    defer g.lasso.deinit(gpa);
    defer g.sound.deinit();
    defer g.renderer.deinit();
    defer g.w.deinit();

    if (opts.stress > 0) g.stress(opts.stress);

    var last = c.SDL_GetPerformanceCounter();
    var accumulator: f64 = 0;
    var frame: u32 = 0;
    var running = true;

    while (running) : (frame += 1) {
        var event: c.SDL_Event = undefined;
        while (c.SDL_PollEvent(&event)) {
            _ = c.SDL_ConvertEventToRenderCoordinates(sdl, &event);
            switch (event.type) {
                c.SDL_EVENT_QUIT => running = false,
                c.SDL_EVENT_KEY_DOWN => g.onKey(event.key.key, event.key.mod),
                c.SDL_EVENT_MOUSE_BUTTON_DOWN => g.onMouseDown(event.button.button, g.renderer.screenToWorld(&g.w, event.button.x, event.button.y)),
                c.SDL_EVENT_MOUSE_BUTTON_UP => g.onMouseUp(event.button.button),
                c.SDL_EVENT_MOUSE_MOTION => if (g.lasso_active) {
                    g.lasso.append(gpa, g.renderer.screenToWorld(&g.w, event.motion.x, event.motion.y)) catch {};
                },
                else => {},
            }
        }

        var wx: f32 = 0;
        var wy: f32 = 0;
        _ = c.SDL_GetMouseState(&wx, &wy);
        var rx: f32 = 0;
        var ry: f32 = 0;
        _ = c.SDL_RenderCoordinatesFromWindow(sdl, wx, wy, &rx, &ry);
        g.mouse = g.renderer.screenToWorld(&g.w, rx, ry);

        const now = c.SDL_GetPerformanceCounter();
        const frame_seconds = seconds(now - last);
        last = now;
        g.perf.fps = smooth(g.perf.fps, @floatCast(1.0 / @max(frame_seconds, 1e-6)));
        // Headless verification runs advance exactly one tick per frame so results do not depend on wall time.
        accumulator += if (opts.frames > 0) tick_seconds else @min(frame_seconds, tick_seconds * max_ticks_per_frame);

        while (accumulator >= tick_seconds) : (accumulator -= tick_seconds) {
            const t0 = c.SDL_GetPerformanceCounter();
            g.tick();
            g.perf.tick_ms = smooth(g.perf.tick_ms, @floatCast(seconds(c.SDL_GetPerformanceCounter() - t0) * 1000));
        }

        if (g.lasso_fade_left_ms > 0) {
            g.lasso_fade_left_ms -= @floatCast(frame_seconds * 1000);
            if (g.lasso_fade_left_ms <= 0) g.lasso.clearRetainingCapacity();
        }

        const hud: render.Hud = .{
            .balls_per_click = g.balls_per_click,
            .countdown = g.countdown,
            .won = g.won,
            .attraction_energy = g.attraction_energy,
            .mutual_gravity = g.mutual_gravity,
            .merging = g.merging,
            .muted = g.muted,
            .lasso = g.lasso.items,
            .lasso_alpha = if (g.lasso_active) 1 else std.math.clamp(g.lasso_fade_left_ms / cfg.lasso_fade_ms, 0, 1),
            .show_perf = g.show_perf,
            .perf = g.perf,
        };
        const r0 = c.SDL_GetPerformanceCounter();
        g.renderer.draw(&g.w, @floatCast(accumulator / tick_seconds), &hud);
        g.perf.render_ms = smooth(g.perf.render_ms, @floatCast(seconds(c.SDL_GetPerformanceCounter() - r0) * 1000));

        if (opts.frames > 0 and frame + 1 >= opts.frames) {
            if (opts.screenshot) |path| saveScreenshot(sdl, path);
            std.debug.print("frames={d} balls={d} tick_ms={d:.3} render_ms={d:.3}\n", .{ frame + 1, g.w.balls.len, g.perf.tick_ms, g.perf.render_ms });
            running = false;
        }
        _ = c.SDL_RenderPresent(sdl);
    }
}
