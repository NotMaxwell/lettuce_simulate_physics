const std = @import("std");
const c = @import("c");
const cfg = @import("config.zig");
const world = @import("world.zig");
const physics = @import("physics.zig");
const render = @import("render.zig");
const audio = @import("audio.zig");
const gfx = @import("gfx.zig");
const gpu_world = @import("gpu_world.zig");

const Vec2 = world.Vec2;
const tick_seconds: f64 = 1.0 / @as(f64, cfg.tick_hz);
/// Catch-up cap: after a long stall, drop simulation time instead of spiralling.
const max_ticks_per_frame = 4;
const stress_count = 100_000;

const Options = struct {
    /// Exit after this many frames (0 = run until closed). For headless verification.
    frames: u32 = 0,
    stress: u32 = 0,
    screenshot: ?[:0]const u8 = null,
    /// Keep merging on after --stress (stress turns it off by default).
    merge: bool = false,
    /// Start every Nth spawned ball alight (0 = none), for verifying fire headlessly.
    ignite: u32 = 0,
    fire: bool = true,
    /// Random clicks, grabs, ignites, lassos and key presses every few frames, for hunting
    /// crashes in interactive paths headlessly.
    autoplay: bool = false,
};

const Game = struct {
    gpa: std.mem.Allocator,
    w: world.World,
    gw: *gpu_world.GpuWorld,
    renderer: render.Renderer,
    sound: audio.Audio,
    window: *c.SDL_Window,

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
    pick_pending: bool = false,
    merges_total: u64 = 0,
    perf: render.Perf = .{},

    /// SDL mouse coordinates are in window points; the renderer works in window pixels.
    fn windowToWorld(g: *const Game, x: f32, y: f32) Vec2 {
        var pw: c_int = 0;
        var ph: c_int = 0;
        var ww: c_int = 0;
        var wh: c_int = 0;
        _ = c.SDL_GetWindowSizeInPixels(g.window, &pw, &ph);
        _ = c.SDL_GetWindowSize(g.window, &ww, &wh);
        const sx: f32 = if (ww > 0) @as(f32, @floatFromInt(pw)) / @as(f32, @floatFromInt(ww)) else 1;
        const sy: f32 = if (wh > 0) @as(f32, @floatFromInt(ph)) / @as(f32, @floatFromInt(wh)) else 1;
        return g.renderer.screenToWorld(&g.w, x * sx, y * sy);
    }

    fn mouseWorld(g: *const Game) Vec2 {
        var x: f32 = 0;
        var y: f32 = 0;
        _ = c.SDL_GetMouseState(&x, &y);
        return g.windowToWorld(x, y);
    }

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
        const b = physics.targetBounds(@floatCast(g.gw.total_area));
        g.gw.queueBurst(&g.w, b.x + b.w / 2, b.y + b.h / 2, count) catch {};
        // Jump straight to the size the new balls need instead of easing over many ticks, so a
        // large burst is not crushed against walls that are still growing.
        g.gw.snapBounds(&g.w);
    }

    fn onMouseDown(g: *Game, button: u8, p: Vec2) void {
        g.mouse = p;
        const shift = c.SDL_GetModState() & c.SDL_KMOD_SHIFT != 0;
        if (button == c.SDL_BUTTON_LEFT) {
            if (shift) {
                g.lasso_active = true;
                g.lasso_fade_left_ms = 0;
                g.lasso.clearRetainingCapacity();
                g.lasso.append(g.gpa, p) catch {};
            } else {
                // The pick runs on the GPU and its result arrives a few ticks later: a hit has
                // already been grabbed there, a miss starts the spawn (see `tick`).
                g.gw.requestPick();
                g.pick_pending = true;
            }
        }
        if (button == c.SDL_BUTTON_RIGHT) g.gw.requestIgnite();
    }

    fn onMouseUp(g: *Game, button: u8) void {
        if (button != c.SDL_BUTTON_LEFT) return;
        g.gw.requestRelease();
        if (!g.lasso_active) return;
        g.lasso_active = false;
        g.lasso_fade_left_ms = cfg.lasso_fade_ms;
        if (g.lasso.items.len >= 3) g.gw.requestLasso(g.lasso.items) catch {};
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
            g.gw.queueBurst(&g.w, g.mouse.x, g.mouse.y, g.spawn_per_tick + extra) catch {};
            g.spawn_ticks_left -= 1;
        }

        // Stage timing stalls one tick, so it only runs (every 2 s) while the overlay shows it.
        g.gw.profile_every = if (g.show_perf) 120 else 0;
        g.gw.tick(&g.w, .{
            .mouse = g.mouse,
            .attract = g.attracting(),
            .mutual_gravity = g.mutual_gravity,
            .merging = g.merging,
        });

        if (g.pick_pending) if (g.gw.pick_result) |hit| {
            g.pick_pending = false;
            if (!hit) g.startSpawn();
        };

        const ev = g.w.events;
        g.merges_total += ev.merges;
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

fn poolView(p: *const gpu_world.Pool, count: u32) render.GpuPool {
    return .{ .pos = p.pos, .prev_pos = p.prev_pos, .body = p.body, .state = p.state, .heat = p.heat, .meta = p.meta, .count = count };
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
        } else if (std.mem.eql(u8, flag, "--ignite")) {
            o.ignite = std.fmt.parseInt(u32, value, 10) catch 0;
        } else if (std.mem.eql(u8, flag, "--fire")) {
            o.fire = !std.mem.eql(u8, value, "off");
        } else if (std.mem.eql(u8, flag, "--merge")) {
            o.merge = std.mem.eql(u8, value, "on");
        } else if (std.mem.eql(u8, flag, "--autoplay")) {
            o.autoplay = std.mem.eql(u8, value, "on");
        }
    }
    return o;
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

    const window = c.SDL_CreateWindow("Lettuce Simulate Physics", cfg.base_w, cfg.base_h, c.SDL_WINDOW_RESIZABLE) orelse {
        std.debug.print("SDL_CreateWindow failed: {s}\n", .{c.SDL_GetError()});
        return error.SdlWindow;
    };
    defer c.SDL_DestroyWindow(window);

    // The offscreen video driver cannot present, so headless runs render to the GPU without a swapchain.
    const driver: []const u8 = if (c.SDL_GetCurrentVideoDriver()) |d| std.mem.span(d) else "";
    const headless = std.mem.eql(u8, driver, "offscreen");
    var gpu = try gfx.Gfx.init(.{
        .window = if (headless) null else window,
        .width = cfg.base_w,
        .height = cfg.base_h,
        .vsync = opts.frames == 0,
    });
    defer gpu.deinit();

    var gw = try gpu_world.GpuWorld.init(gpa, gpu.dev);
    defer gw.deinit();

    var g: Game = .{
        .gpa = gpa,
        .w = .init(gpa, c.SDL_GetPerformanceCounter()),
        .gw = &gw,
        .renderer = try .init(gpa, &gpu),
        .sound = .init(gpa),
        .window = window,
    };
    defer g.lasso.deinit(gpa);
    defer g.sound.deinit();
    defer g.renderer.deinit();
    defer g.w.deinit();

    gw.ignite_every = opts.ignite;
    gw.fire_enabled = opts.fire;
    if (opts.stress > 0) g.stress(opts.stress);
    if (opts.merge) g.merging = true;

    var last = c.SDL_GetPerformanceCounter();
    var accumulator: f64 = 0;
    var frame: u32 = 0;
    var running = true;

    while (running) : (frame += 1) {
        var event: c.SDL_Event = undefined;
        while (c.SDL_PollEvent(&event)) {
            switch (event.type) {
                c.SDL_EVENT_QUIT => running = false,
                c.SDL_EVENT_KEY_DOWN => g.onKey(event.key.key, event.key.mod),
                c.SDL_EVENT_MOUSE_BUTTON_DOWN => g.onMouseDown(event.button.button, g.windowToWorld(event.button.x, event.button.y)),
                c.SDL_EVENT_MOUSE_BUTTON_UP => g.onMouseUp(event.button.button),
                c.SDL_EVENT_MOUSE_MOTION => if (g.lasso_active) {
                    g.lasso.append(gpa, g.windowToWorld(event.motion.x, event.motion.y)) catch {};
                },
                else => {},
            }
        }

        if (opts.autoplay) autoplay(&g, frame) else g.mouse = g.mouseWorld();

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
            const prof = g.gw.profile;
            g.perf.gpu_gravity_ms = prof.gravity_ms;
            g.perf.gpu_collide_ms = prof.collide_ms;
            g.perf.gpu_game_ms = prof.game_ms;
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
            .gpu = gpuBalls(&gw, g.w.current_level),
        };
        const r0 = c.SDL_GetPerformanceCounter();
        g.renderer.draw(&g.w, @floatCast(accumulator / tick_seconds), &hud);
        g.perf.render_ms = smooth(g.perf.render_ms, @floatCast(seconds(c.SDL_GetPerformanceCounter() - r0) * 1000));

        const done = opts.frames > 0 and frame + 1 >= opts.frames;
        g.renderer.present(if (done) if (opts.screenshot) |path| path.ptr else null else null);
        if (done) {
            std.debug.print("frames={d} balls={d} ids={d} queued={d} max_level={d} merges={d} fps={d:.1} tick_ms={d:.3} render_ms={d:.3} gpu_gravity_ms={d:.3} gpu_collide_ms={d:.3} gpu_game_ms={d:.3} (samples {d}) burning={d} particles={d} ashes={d}\n", .{ frame + 1, gw.alive, gw.count, gw.pendingSpawns(), g.w.current_level, g.merges_total, g.perf.fps, g.perf.tick_ms, g.perf.render_ms, g.perf.gpu_gravity_ms, g.perf.gpu_collide_ms, g.perf.gpu_game_ms, gw.profile.samples, gw.stats.burning, gw.stats.particles, gw.stats.ashes });
            running = false;
        }
    }
}

test {
    _ = gpu_world;
    _ = world;
}

fn autoplay(g: *Game, frame: u32) void {
    if (frame % 6 != 0) return;
    const r = g.w.prng.random();
    const b = g.w.bounds;
    g.mouse = .{ .x = b.x + r.float(f32) * b.w, .y = b.y + r.float(f32) * b.h };
    switch (r.uintLessThan(u8, 12)) {
        0...3 => {
            g.onMouseDown(c.SDL_BUTTON_LEFT, g.mouse);
            g.onMouseUp(c.SDL_BUTTON_LEFT);
        },
        4, 5 => g.onMouseDown(c.SDL_BUTTON_LEFT, g.mouse), // held: grab follows the mouse until a later release
        6 => g.onMouseUp(c.SDL_BUTTON_LEFT),
        7 => g.onMouseDown(c.SDL_BUTTON_RIGHT, g.mouse),
        8 => {
            g.lasso_active = true;
            g.lasso.clearRetainingCapacity();
            const rad = 50 + r.float(f32) * 300;
            for (0..12) |k| {
                const a = @as(f32, @floatFromInt(k)) * std.math.tau / 12.0;
                g.lasso.append(g.gpa, .{ .x = g.mouse.x + @cos(a) * rad, .y = g.mouse.y + @sin(a) * rad }) catch {};
            }
            g.onMouseUp(c.SDL_BUTTON_LEFT);
        },
        9 => g.onKey(c.SDLK_EQUALS, c.SDL_KMOD_CTRL),
        10 => g.onKey(c.SDLK_F, 0),
        else => {},
    }
}
