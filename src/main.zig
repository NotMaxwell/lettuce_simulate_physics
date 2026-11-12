const std = @import("std");
const c = @import("c");
const world = @import("world.zig");

pub fn main() !void {
    if (!c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_AUDIO)) return error.SdlInit;
    defer c.SDL_Quit();
    var window: ?*c.SDL_Window = null;
    var renderer: ?*c.SDL_Renderer = null;
    if (!c.SDL_CreateWindowAndRenderer("lettuce", 1029, 720, c.SDL_WINDOW_RESIZABLE, &window, &renderer)) return error.SdlWindow;
    const gpa = std.heap.smp_allocator;
    var w = world.World.init(gpa, 1);
    defer w.deinit();
    try w.spawnBall(1, 2);
    const v: @Vector(8, f32) = @splat(2.0);
    const t0 = c.SDL_GetPerformanceCounter();
    _ = c.SDL_SetRenderDrawColor(renderer, 10, 20, 30, 255);
    _ = c.SDL_RenderClear(renderer);
    _ = c.SDL_RenderDebugText(renderer, 10, 10, "hello");
    _ = c.SDL_RenderPresent(renderer);
    var prng: std.Random.DefaultPrng = .init(42);
    std.debug.print("ok renderer={s} sum={d} rnd={d} ticks={d} stbi={any}\n", .{ c.SDL_GetRendererName(renderer), @reduce(.Add, v), prng.random().float(f32), c.SDL_GetPerformanceCounter() - t0, @TypeOf(c.stbi_load_from_memory) });
}
