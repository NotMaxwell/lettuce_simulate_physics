//! Shared data contract. Simulation mutates it, rendering only reads it.
const std = @import("std");
const cfg = @import("config.zig");

pub const Vec2 = struct { x: f32 = 0, y: f32 = 0 };

pub const Rect = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
};

pub const BallFlags = packed struct(u8) {
    alive: bool = true,
    burning: bool = false,
    scored: bool = false,
    grabbed: bool = false,
    /// Set once a ball has been thrown; only thrown balls can score in the hoop.
    thrown: bool = false,
    _pad: u3 = 0,
};

pub const Ball = struct {
    x: f32,
    y: f32,
    /// Position/angle at the previous tick, for render interpolation.
    prev_x: f32,
    prev_y: f32,
    vx: f32,
    vy: f32,
    /// Degrees, counter-clockwise positive (pygame.transform.rotate convention).
    angle: f32 = 0,
    prev_angle: f32 = 0,
    ang_vel: f32 = 0,
    radius: f32,
    mass: f32,
    density: f32,
    retention: f32,
    friction: f32,
    /// Per-ball downward gravity, used only when mutual gravity is off.
    gravity: f32,
    temp: f32 = cfg.room_temp,
    burn: f32 = 0,
    level: u8,
    flags: BallFlags = .{},
    /// Ring buffer of past positions; `trail_head` is the next write slot.
    trail: [cfg.max_trail]Vec2 = undefined,
    trail_len: u8 = 0,
    trail_head: u8 = 0,
    /// Position before the last grab-follow step; the throw impulse is measured from it.
    grab_prev: Vec2 = .{},
};

pub const Particle = struct { x: f32, y: f32, vx: f32, vy: f32, size: f32, age: u16, life: u16, r: u8, g: u8, b: u8 };

pub const Ash = struct {
    x: f32,
    y: f32,
    prev_x: f32,
    prev_y: f32,
    vx: f32,
    vy: f32,
    gravity: f32,
    retention: f32,
    friction: f32,
    age: u16 = 0,
    start_red: u8,
};

pub const Explosion = struct { x: f32, y: f32, size: f32, age: u16 = 0 };

/// Things that happened during the last tick; audio and HUD react to them.
pub const TickEvents = struct {
    merges: u32 = 0,
    deaths: u32 = 0,
    explosions: u32 = 0,
    any_burning: bool = false,
    won: bool = false,
    scores: u32 = 0,
};

/// Wall-clock cost of the last tick, filled by physics for the perf overlay and bench.
pub const TickProfile = struct {
    total_ns: u64 = 0,
    gravity_ns: u64 = 0,
    collide_ns: u64 = 0,
    effects_ns: u64 = 0,
    /// Narrow-phase distance tests done by the ball broad-phase.
    pair_tests: u32 = 0,
    /// Threads that took part in mutual gravity (1 = single-threaded).
    gravity_threads: u32 = 1,
};

pub const TickInput = struct {
    /// Mouse position in world coordinates.
    mouse: Vec2 = .{},
    attract: bool = false,
    mutual_gravity: bool = true,
    merging: bool = true,
    random_explosions: bool = true,
};

pub const World = struct {
    gpa: std.mem.Allocator,
    balls: std.MultiArrayList(Ball) = .empty,
    particles: std.MultiArrayList(Particle) = .empty,
    ashes: std.MultiArrayList(Ash) = .empty,
    explosions: std.ArrayList(Explosion) = .empty,
    /// Current play area in world units. Starts as the base window and grows
    /// (keeping the aspect ratio) so the balls always fit; the camera zooms out to match.
    bounds: Rect = .{ .x = 0, .y = 0, .w = cfg.base_w, .h = cfg.base_h },
    prev_bounds: Rect = .{ .x = 0, .y = 0, .w = cfg.base_w, .h = cfg.base_h },
    /// Highest level reached so far (min start_level); new balls spawn at 1..current_level.
    current_level: u8 = cfg.start_level,
    events: TickEvents = .{},
    /// Events from actions taken between ticks (lasso, explodeAt); folded into the next tick's events.
    pending_events: TickEvents = .{},
    profile: TickProfile = .{},
    tick: u64 = 0,
    prng: std.Random.DefaultPrng,
    /// Buffers owned by physics.zig, kept alive between ticks so steady-state ticks never allocate.
    scratch: ?*anyopaque = null,
    scratch_free: ?*const fn (std.mem.Allocator, *anyopaque) void = null,

    pub fn init(gpa: std.mem.Allocator, seed: u64) World {
        return .{ .gpa = gpa, .prng = .init(seed) };
    }

    pub fn deinit(w: *World) void {
        if (w.scratch) |s| w.scratch_free.?(w.gpa, s);
        w.balls.deinit(w.gpa);
        w.particles.deinit(w.gpa);
        w.ashes.deinit(w.gpa);
        w.explosions.deinit(w.gpa);
    }

    pub fn spawnBall(w: *World, x: f32, y: f32) !void {
        const nb = randomBall(w.prng.random(), w.current_level);
        try w.balls.append(w.gpa, .{
            .x = x,
            .y = y,
            .prev_x = x,
            .prev_y = y,
            .vx = nb.vx,
            .vy = nb.vy,
            .radius = nb.radius,
            .mass = nb.mass,
            .density = nb.density,
            .retention = nb.retention,
            .friction = nb.friction,
            .gravity = nb.gravity,
            .level = nb.level,
        });
    }
};

/// Random properties of a new ball, shared by the CPU world and gpu_world.
pub const NewBall = struct {
    vx: f32,
    vy: f32,
    radius: f32,
    mass: f32,
    density: f32,
    retention: f32,
    friction: f32,
    gravity: f32,
    level: u8,
};

/// Mirrors Ball.__init__ in the Python original. Draw order is kept from the original spawnBall
/// so seeded CPU runs are unchanged.
pub fn randomBall(r: std.Random, current_level: u8) NewBall {
    const radius: f32 = @floatFromInt(r.intRangeAtMost(u8, cfg.spawn_radius_min, cfg.spawn_radius_max));
    const density = 0.5 + r.float(f32) * 1.5;
    const vx: f32 = @floatFromInt(r.intRangeAtMost(i8, -5, 5));
    const vy: f32 = @floatFromInt(r.intRangeAtMost(i8, -5, 5));
    const retention = 0.7 + r.float(f32) * 0.25;
    const friction = 0.05 + r.float(f32) * 0.25;
    const gravity = 0.6 + 0.3 * @as(f32, @floatFromInt(r.intRangeAtMost(u8, 0, 5)));
    return .{
        .vx = vx,
        .vy = vy,
        .radius = radius,
        .mass = @floor(20 + radius * 4 * density),
        .density = density,
        .retention = retention,
        .friction = friction,
        .gravity = gravity,
        .level = r.intRangeAtMost(u8, 1, current_level),
    };
}

test "spawn" {
    var w = World.init(std.testing.allocator, 1);
    defer w.deinit();
    try w.spawnBall(10, 20);
    try std.testing.expectEqual(@as(usize, 1), w.balls.len);
}
