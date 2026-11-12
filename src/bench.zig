//! Headless physics benchmark: `zig build bench -- <balls> <ticks>`.
const std = @import("std");
const c = @import("c");
const cfg = @import("config.zig");
const world = @import("world.zig");
const physics = @import("physics.zig");

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
