//! Headless physics benchmark: `zig build bench -- <balls> <ticks> [gpu]`.
const std = @import("std");
const c = @import("c");
const cfg = @import("config.zig");
const world = @import("world.zig");
const physics = @import("physics.zig");
const gfx = @import("gfx.zig");
const gpu_world = @import("gpu_world.zig");

test {
    _ = physics;
    _ = @import("effects.zig");
}

/// Wraps the real allocator to prove steady-state ticks do not allocate.
const CountingAllocator = struct {
    child: std.mem.Allocator,
    allocs: usize = 0,

    fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(p: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(p));
        self.allocs += 1;
        return self.child.rawAlloc(len, a, ra);
    }
    fn resize(p: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(p));
        if (n > m.len) self.allocs += 1;
        return self.child.rawResize(m, a, n, ra);
    }
    fn remap(p: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(p));
        if (n > m.len) self.allocs += 1;
        return self.child.rawRemap(m, a, n, ra);
    }
    fn free(p: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(p));
        self.child.rawFree(m, a, ra);
    }
};

const Scenario = struct {
    name: []const u8,
    input: world.TickInput,
    ignite_all: bool = false,
};

pub fn main(init: std.process.Init.Minimal) !void {
    var it = init.args.iterate();
    _ = it.next();
    const n = if (it.next()) |a| try std.fmt.parseInt(u32, a, 10) else 1000;
    const ticks = if (it.next()) |a| try std.fmt.parseInt(u32, a, 10) else 600;
    if (it.next()) |mode| if (std.mem.eql(u8, mode, "gpu")) return gpuBench(n, ticks);

    const base: world.TickInput = .{ .merging = false, .random_explosions = false };
    var on = base;
    on.mutual_gravity = true;
    var off = base;
    off.mutual_gravity = false;
    const scenarios = [_]Scenario{
        .{ .name = "mutual gravity ON ", .input = on },
        .{ .name = "mutual gravity OFF", .input = off },
        .{ .name = "gravity OFF, all burning (fire+ash)", .input = off, .ignite_all = true },
    };
    std.debug.print("bench: {d} balls, {d} ticks, merging off, random explosions off, SIMD width {d}\n", .{ n, ticks, std.simd.suggestVectorLength(f32) orelse 0 });
    for (scenarios) |s| try run(s, n, ticks);
}

fn run(s: Scenario, n: u32, ticks: u32) !void {
    var counter: CountingAllocator = .{ .child = std.heap.smp_allocator };
    const gpa = counter.allocator();
    var w = world.World.init(gpa, 42);
    defer w.deinit();

    for (0..n) |_| try w.spawnBall(0, 0);
    physics.snapBounds(&w);
    const r = w.prng.random();
    const b = w.bounds;
    const sl = w.balls.slice();
    for (sl.items(.x), sl.items(.y), sl.items(.prev_x), sl.items(.prev_y), sl.items(.radius)) |*x, *y, *px, *py, rad| {
        const m = rad + cfg.wall;
        x.* = b.x + m + r.float(f32) * (b.w - 2 * m);
        y.* = b.y + m + r.float(f32) * (b.h - 2 * m);
        px.* = x.*;
        py.* = y.*;
    }
    if (s.ignite_all) for (0..w.balls.len) |i| physics.ignite(&w, i);

    // The first tick sizes the scratch buffers and pools; it is not measured.
    physics.tick(&w, s.input);
    counter.allocs = 0;

    const times = try std.heap.smp_allocator.alloc(f64, ticks);
    defer std.heap.smp_allocator.free(times);
    var grav: f64 = 0;
    var coll: f64 = 0;
    var fx: f64 = 0;
    var total_prof: f64 = 0;
    var max_tests: u32 = 0;
    var max_ashes: usize = 0;
    var threads: u32 = 1;
    const freq: f64 = @floatFromInt(c.SDL_GetPerformanceFrequency());
    for (times) |*t| {
        const t0 = c.SDL_GetPerformanceCounter();
        physics.tick(&w, s.input);
        const t1 = c.SDL_GetPerformanceCounter();
        t.* = @as(f64, @floatFromInt(t1 - t0)) * 1000.0 / freq;
        const p = w.profile;
        grav += ms(p.gravity_ns);
        coll += ms(p.collide_ns);
        fx += ms(p.effects_ns);
        total_prof += ms(p.total_ns);
        max_tests = @max(max_tests, p.pair_tests);
        max_ashes = @max(max_ashes, w.ashes.len);
        threads = @max(threads, p.gravity_threads);
    }
    const allocs = counter.allocs;

    var sum: f64 = 0;
    for (times) |t| sum += t;
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    const nt: f64 = @floatFromInt(ticks);
    const p99 = times[@min(times.len - 1, (times.len * 99) / 100)];
    std.debug.print(
        \\[{s}]
        \\  tick ms: mean {d:.3}  p99 {d:.3}  max {d:.3}
        \\  breakdown (mean ms): gravity {d:.3} ({d} threads)  collisions {d:.3}  effects {d:.3}  rest {d:.3}
        \\  max pair tests/tick {d}  balls at end {d}  peak ashes {d}  particles at end {d}  heap allocs during measured ticks: {d}
        \\
    , .{
        s.name,                               sum / nt,  p99,         times[times.len - 1],
        grav / nt,                            threads,   coll / nt,   fx / nt,
        (total_prof - grav - coll - fx) / nt, max_tests, w.balls.len, max_ashes,
        w.particles.len,                      allocs,
    });
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// GPU world: balls uniformly random over the snapped play area, merging and random explosions
/// off. Reports pipelined throughput (ticks submitted back to back, wall time / ticks, which is
/// the GPU cost per tick once the queue is full) and a per-stage breakdown from blocking samples.
fn gpuBench(n: u32, ticks: u32) !void {
    // SDL_GPU needs the video subsystem; the bench never opens a window.
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();
    var gpu = try gfx.Gfx.init(.{ .window = null });
    defer gpu.deinit();
    const gpa = std.heap.smp_allocator;
    var gw = try gpu_world.GpuWorld.init(gpa, gpu.dev);
    defer gw.deinit();
    var w = world.World.init(gpa, 42);
    defer w.deinit();

    for (0..n) |_| try gw.queueBall(&w, 0, 0);
    gw.snapBounds(&w);
    const r = w.prng.random();
    const b = w.bounds;
    for (gw.pending_pos.items, gw.pending.items) |*p, nb| {
        const m = nb.radius + cfg.wall;
        p.* = .{ .x = b.x + m + r.float(f32) * (b.w - 2 * m), .y = b.y + m + r.float(f32) * (b.h - 2 * m) };
    }
    std.debug.print("gpu bench: {d} balls, {d} ticks, merging off, random explosions off\n", .{ n, ticks });

    const freq: f64 = @floatFromInt(c.SDL_GetPerformanceFrequency());
    for ([_]bool{ true, false }) |grav| {
        const in: gpu_world.Input = .{ .mutual_gravity = grav, .merging = false, .random_explosions = false };
        gw.profile_every = 0;
        // Upload every ball and warm up pipelines; not measured.
        while (gw.pendingSpawns() > 0) gw.tick(&w, in);
        for (0..10) |_| gw.tick(&w, in);
        gw.drain(&w);

        const t0 = c.SDL_GetPerformanceCounter();
        for (0..ticks) |_| gw.tick(&w, in);
        gw.drain(&w);
        const wall_ms = @as(f64, @floatFromInt(c.SDL_GetPerformanceCounter() - t0)) * 1000.0 / freq;

        const samples: u32 = @min(ticks, 30);
        gw.profile_every = 1;
        var gsum: f64 = 0;
        var csum: f64 = 0;
        var osum: f64 = 0;
        for (0..samples) |_| {
            gw.tick(&w, in);
            gsum += gw.profile.gravity_ms;
            csum += gw.profile.collide_ms;
            osum += gw.profile.game_ms;
        }
        gw.drain(&w);
        const ns: f64 = @floatFromInt(samples);
        std.debug.print(
            \\[GPU mutual gravity {s}]
            \\  pipelined: {d:.3} ms/tick ({d} ticks)
            \\  blocking samples (mean of {d}, include submit+wait latency): gravity {d:.3}  collide {d:.3}  merge/hoop/readback {d:.3} ms
            \\  balls alive {d}
            \\
        , .{ if (grav) "ON " else "OFF", wall_ms / @as(f64, @floatFromInt(ticks)), ticks, samples, gsum / ns, csum / ns, osum / ns, gw.alive });
    }
}
