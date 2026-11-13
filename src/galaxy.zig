//! Galaxy-mode scenarios: initial conditions for orbiting systems under the mutual gravity of
//! gpu_sim/gravity.zig. Pure CPU math; gpu_world queues the bodies like any other spawn.
//!
//! That gravity is two-dimensional (log potential): mass M pulls with g*M/r, not g*M/r^2. So a
//! circular orbit has speed sqrt(g * M_enclosed) whatever its radius, and (2D shell theorem) a
//! ring of mass acts on anything outside it as if all of it sat at the centre and not at all on
//! anything inside it. Every scenario builds its velocities from those two facts.
const std = @import("std");
const cfg = @import("config.zig");
const world = @import("world.zig");

/// Orbital speed (world units per tick) a scenario is normalised to, whatever its mass or size,
/// so a 1k and a 1M body galaxy turn at the same rate on screen.
pub const v0: f32 = 2.5;

pub const Scenario = enum {
    /// Rotating disk of stars around a heavy core.
    disk,
    /// Two disks on a collision course, one counter-rotating.
    collision,
    /// Cold, slowly spinning cloud with no core; it fragments into clumps and little systems.
    cloud,
    /// A rotating cluster of star systems (planets with moons).
    systems,

    pub fn defaultCount(s: Scenario) u32 {
        return switch (s) {
            .disk => 3000,
            .collision => 4000,
            .cloud => 2500,
            .systems => 600,
        };
    }
};

pub const Body = struct {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    mass: f32,
    radius: f32,
};

pub const Built = struct {
    bodies: []Body,
    /// Gravity strength (gpu gravity g_scale) the velocities were computed for.
    g: f32,
    /// Distance from the centre that holds every body.
    extent: f32,

    pub fn deinit(b: *Built, gpa: std.mem.Allocator) void {
        gpa.free(b.bodies);
        b.* = undefined;
    }
};

/// Builds `n` bodies (roughly; `systems` rounds to whole systems) of scenario `s` around `c`.
pub fn build(gpa: std.mem.Allocator, r: std.Random, s: Scenario, n: u32, c: world.Vec2) !Built {
    var list: std.ArrayList(Body) = .empty;
    errdefer list.deinit(gpa);
    const n_f: f32 = @floatFromInt(@max(n, 2));
    // Disk radius grows with sqrt(n) so the stars cover about the same fraction of it at any count.
    const radius = 450 * @sqrt(n_f / 3000);
    var g: f32 = undefined;
    switch (s) {
        .disk => g = try disk(gpa, &list, r, n, radius, c, .{}, 1, null),
        .collision => {
            const half = n / 2;
            const sep = 2.6 * radius;
            // Bulk approach speed as a fraction of v0: low enough that the pair falls together
            // and tears into tidal tails instead of passing straight through.
            const approach = 0.35 * v0;
            const total = 2 * (diskMass(half) * (1 + DiskOpts.core_frac_default));
            g = v0 * v0 / total;
            _ = try disk(gpa, &list, r, half, radius * 0.8, .{ .x = c.x - sep / 2, .y = c.y - radius * 0.5 }, .{ .x = approach, .y = 0 }, 1, g);
            _ = try disk(gpa, &list, r, n - half, radius * 0.8, .{ .x = c.x + sep / 2, .y = c.y + radius * 0.5 }, .{ .x = -approach, .y = 0 }, -1, g);
        },
        .cloud => g = try cloud(gpa, &list, r, n, radius, c),
        .systems => g = try cluster(gpa, &list, r, n, radius * 1.4, c),
    }
    var extent: f32 = 0;
    for (list.items) |b| extent = @max(extent, @sqrt((b.x - c.x) * (b.x - c.x) + (b.y - c.y) * (b.y - c.y)) + b.radius);
    return .{ .bodies = try list.toOwnedSlice(gpa), .g = g, .extent = extent };
}

/// One star system (star, planets, moons) centred on `c` with bulk velocity `bulk`, for strength
/// `g`. Appends to `list`; returns the system's mass.
pub fn system(gpa: std.mem.Allocator, list: *std.ArrayList(Body), r: std.Random, c: world.Vec2, bulk: world.Vec2, g: f32) !f32 {
    const star_m: f32 = 160 + r.float(f32) * 120;
    try list.append(gpa, body(c.x, c.y, bulk.x, bulk.y, star_m));
    var inner = star_m;
    const planets = r.intRangeAtMost(u32, 3, 6);
    const star_r = radiusFor(star_m);
    var a: f32 = star_r + 25;
    for (0..planets) |_| {
        a += 30 + r.float(f32) * 35;
        const pm: f32 = 8 + r.float(f32) * 40;
        const th = r.float(f32) * std.math.tau;
        const ex = @cos(th);
        const ey = @sin(th);
        const v = @sqrt(g * inner);
        const px = c.x + a * ex;
        const py = c.y + a * ey;
        const pvx = bulk.x - ey * v;
        const pvy = bulk.y + ex * v;
        try list.append(gpa, body(px, py, pvx, pvy, pm));
        // A moon stays bound while the planet's pull g*m/d beats the star's tidal pull
        // g*M*d/a^2, i.e. d << a*sqrt(m/M); 0.4 of that bound keeps most moons for many orbits.
        const pr = radiusFor(pm);
        const d_max = 0.4 * a * @sqrt(pm / inner);
        var d = pr + 5;
        var moons: u32 = 0;
        while (moons < 3 and d + 3 < d_max) : (moons += 1) {
            const mm: f32 = 0.5 + r.float(f32) * 1.5;
            const mth = r.float(f32) * std.math.tau;
            const mv = @sqrt(g * pm);
            try list.append(gpa, body(px + d * @cos(mth), py + d * @sin(mth), pvx - @sin(mth) * mv, pvy + @cos(mth) * mv, mm));
            d += radiusFor(mm) * 2 + 3;
        }
        inner += pm;
    }
    return inner;
}

/// Gravity strength that makes a lone `system` orbit at about v0.
pub fn systemG() f32 {
    return v0 * v0 / 220;
}

/// Ball radius for a body of mass `m` (world units). Area grows with mass, as for accretion.
pub fn radiusFor(m: f32) f32 {
    return std.math.clamp(1.2 + 1.1 * @sqrt(m), 1.5, cfg.max_radius * 0.4);
}

/// Merge-game level (ball texture) for a body of mass `m`: heavier bodies get later textures.
pub fn levelFor(m: f32) u8 {
    const l = 1 + @log2(@max(m, 1));
    return @intFromFloat(std.math.clamp(l, 1, @as(f32, @floatFromInt(cfg.max_level))));
}

/// The spawn record for a body. No floor gravity or floor friction: galaxy mode runs with mutual
/// gravity on, which already ignores both.
pub fn toNewBall(b: Body) world.NewBall {
    return .{
        .vx = b.vx,
        .vy = b.vy,
        .radius = b.radius,
        .mass = b.mass,
        .density = 1,
        .retention = 0.9,
        .friction = 0,
        .gravity = 0,
        .level = levelFor(b.mass),
    };
}

/// A square play area around `c` big enough that a scenario's bodies rarely reach its walls.
pub fn boundsFor(c: world.Vec2, extent: f32) world.Rect {
    const half = @max(4 * extent, 1200);
    return .{ .x = c.x - half, .y = c.y - half, .w = 2 * half, .h = 2 * half };
}

fn body(x: f32, y: f32, vx: f32, vy: f32, m: f32) Body {
    return .{ .x = x, .y = y, .vx = vx, .vy = vy, .mass = m, .radius = radiusFor(m) };
}

/// Star masses 1..21, most of them small (u^4 skew).
fn starMass(r: std.Random) f32 {
    const u = r.float(f32);
    return 1 + @floor(u * u * u * u * 21);
}

/// Mean of starMass is 1 + 21/5 (minus the floor's ~0.5).
fn diskMass(n: u32) f32 {
    return @as(f32, @floatFromInt(n)) * 4.7;
}

const DiskOpts = struct {
    const core_frac_default: f32 = 0.25;
};

/// Exponential disk of `n` stars (scale length radius/3, cut at `radius`) around a core of a
/// quarter of the disk mass, on circular orbits plus 5% velocity dispersion. `spin` is +1 or -1.
/// With `g_fixed` null the strength is chosen so the edge turns at v0; returns the strength used.
fn disk(gpa: std.mem.Allocator, list: *std.ArrayList(Body), r: std.Random, n: u32, radius: f32, c: world.Vec2, bulk: world.Vec2, spin: f32, g_fixed: ?f32) !f32 {
    const first = list.items.len;
    try list.ensureUnusedCapacity(gpa, n + 1);
    var m_disk: f32 = 0;
    const h = radius / 3;
    const cut = 1 - @exp(-radius / h);
    const stars = list.addManyAsSliceAssumeCapacity(n);
    for (stars) |*b| {
        const m = starMass(r);
        m_disk += m;
        b.* = body(0, 0, 0, 0, m);
        // Distance from the centre stored in x until velocities are known.
        b.x = -h * @log(1 - r.float(f32) * cut);
    }
    const core_m = DiskOpts.core_frac_default * m_disk;
    const g = g_fixed orelse v0 * v0 / (core_m + m_disk);
    const core_r = radiusFor(core_m);
    // Circular speed needs the mass inside each orbit: sort by radius and accumulate.
    std.mem.sort(Body, stars, {}, struct {
        fn lt(_: void, a: Body, b: Body) bool {
            return a.x < b.x;
        }
    }.lt);
    var enclosed = core_m;
    for (stars) |*b| {
        const d = b.x + core_r + 4;
        enclosed += b.mass;
        const th = r.float(f32) * std.math.tau;
        const ex = @cos(th);
        const ey = @sin(th);
        const v = @sqrt(g * enclosed);
        const disp = 0.05 * v;
        b.x = c.x + d * ex;
        b.y = c.y + d * ey;
        b.vx = bulk.x - spin * ey * v + disp * r.floatNorm(f32);
        b.vy = bulk.y + spin * ex * v + disp * r.floatNorm(f32);
    }
    list.appendAssumeCapacity(body(c.x, c.y, bulk.x, bulk.y, core_m));
    std.debug.assert(list.items.len == first + n + 1);
    return g;
}

/// Uniform disk of `n` bodies with a third of the circular speed and some random motion: it
/// collapses and fragments. Returns the strength used.
fn cloud(gpa: std.mem.Allocator, list: *std.ArrayList(Body), r: std.Random, n: u32, radius: f32, c: world.Vec2) !f32 {
    const stars = try list.addManyAsSlice(gpa, n);
    var total: f32 = 0;
    for (stars) |*b| {
        const m = starMass(r);
        total += m;
        b.* = body(0, 0, 0, 0, m);
        b.x = radius * @sqrt(r.float(f32));
    }
    const g = v0 * v0 / total;
    std.mem.sort(Body, stars, {}, struct {
        fn lt(_: void, a: Body, b: Body) bool {
            return a.x < b.x;
        }
    }.lt);
    var enclosed: f32 = 0;
    for (stars) |*b| {
        enclosed += b.mass;
        const d = b.x;
        const th = r.float(f32) * std.math.tau;
        const v = 0.35 * @sqrt(g * enclosed);
        const noise = 0.15 * v0;
        b.x = c.x + d * @cos(th);
        b.y = c.y + d * @sin(th);
        b.vx = -@sin(th) * v + noise * r.floatNorm(f32);
        b.vy = @cos(th) * v + noise * r.floatNorm(f32);
    }
    return g;
}

/// Star systems scattered over a disk, each moving on a circular orbit about the cluster centre.
/// Returns the strength used, which makes planets orbit their stars at about v0.
fn cluster(gpa: std.mem.Allocator, list: *std.ArrayList(Body), r: std.Random, n: u32, radius: f32, c: world.Vec2) !f32 {
    // About 18 bodies per system; at least two systems.
    const count = @max(n / 18, 2);
    const g = systemG();
    const Site = struct { d: f32, th: f32 };
    const sites = try gpa.alloc(Site, count);
    defer gpa.free(sites);
    for (sites) |*s| s.* = .{ .d = radius * @sqrt(0.05 + 0.95 * r.float(f32)), .th = r.float(f32) * std.math.tau };
    std.mem.sort(Site, sites, {}, struct {
        fn lt(_: void, a: Site, b: Site) bool {
            return a.d < b.d;
        }
    }.lt);
    // Each system's orbit uses the systems inside it, which (sorted by distance) were already
    // built. That mass is estimated from the average system so the build stays a single pass.
    const sys_mass: f32 = 220 + 4.5 * 28;
    for (sites, 0..) |s, k| {
        const enclosed = sys_mass * @as(f32, @floatFromInt(k + 1));
        // Half the circular speed: a full-speed cluster flies apart once it starts merging.
        const v = 0.5 * @sqrt(g * enclosed);
        const ex = @cos(s.th);
        const ey = @sin(s.th);
        _ = try system(gpa, list, r, .{ .x = c.x + s.d * ex, .y = c.y + s.d * ey }, .{ .x = -ey * v, .y = ex * v }, g);
    }
    return g;
}

test "disk orbits are circular for its own gravity" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);
    var b = try build(gpa, prng.random(), .disk, 2000, .{ .x = 100, .y = -50 });
    defer b.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2001), b.bodies.len);
    var total: f32 = 0;
    for (b.bodies) |x| total += x.mass;
    // The outermost star moves at v0 within the 5% dispersion (3 sigma).
    var far: Body = b.bodies[0];
    for (b.bodies) |x| {
        if (@abs(x.x - 100) + @abs(x.y + 50) > @abs(far.x - 100) + @abs(far.y + 50)) far = x;
    }
    const speed = @sqrt(far.vx * far.vx + far.vy * far.vy);
    try std.testing.expectApproxEqRel(v0, speed, 0.2);
    try std.testing.expectApproxEqRel(v0 * v0 / total, b.g, 1e-4);
    // Everything fits the reported extent, and the bounds hold the extent.
    const bounds = boundsFor(.{ .x = 100, .y = -50 }, b.extent);
    for (b.bodies) |x| {
        try std.testing.expect(x.x > bounds.x and x.x < bounds.x + bounds.w);
        try std.testing.expect(x.y > bounds.y and x.y < bounds.y + bounds.h);
    }
}

test "every scenario builds sane bodies" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    inline for (@typeInfo(Scenario).@"enum".field_names) |name| {
        const s = @field(Scenario, name);
        var b = try build(gpa, prng.random(), s, s.defaultCount(), .{});
        defer b.deinit(gpa);
        try std.testing.expect(b.bodies.len > 20);
        try std.testing.expect(b.g > 0 and std.math.isFinite(b.g));
        for (b.bodies) |x| {
            try std.testing.expect(std.math.isFinite(x.x) and std.math.isFinite(x.vy));
            try std.testing.expect(x.radius >= 1.5 and x.mass > 0);
            const nb = toNewBall(x);
            try std.testing.expect(nb.level >= 1 and nb.level <= cfg.max_level);
        }
    }
}

test "moons start bound to their planet" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    var list: std.ArrayList(Body) = .empty;
    defer list.deinit(gpa);
    const g = systemG();
    _ = try system(gpa, &list, prng.random(), .{}, .{}, g);
    // In 2D gravity nothing escapes to infinity, so "bound" here means: relative to its planet a
    // moon moves no faster than the planet's circular speed at that distance (sqrt(g*m)) + 1%.
    var planet: ?Body = null;
    for (list.items[1..]) |x| {
        if (x.mass >= 8) {
            planet = x;
            continue;
        }
        const p = planet.?;
        const rel = @sqrt((x.vx - p.vx) * (x.vx - p.vx) + (x.vy - p.vy) * (x.vy - p.vy));
        try std.testing.expect(rel <= @sqrt(g * p.mass) * 1.01);
    }
}
