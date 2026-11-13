//! Initial conditions for the 3D sim, in f64 and physical units. Positions become u32 box
//! fractions in sim3.zig, once the box is known (`fitBox`).
const std = @import("std");

pub const Body3 = struct { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, m: f64 };

pub const Model = enum { plummer, kepler, uniform };

pub const Plummer = struct {
    /// Henon units: G = M = 1, E = -1/4, which fixes the scale radius at 3 pi / 16.
    henon: bool = true,
    /// Scale radius and total mass, used when `henon` is false (G = 1).
    a: f64 = 1,
    mass: f64 = 1,
    /// Radius cut in scale radii; the mass beyond r = 20a is 3.7e-3.
    cut: f64 = 20,
};

/// Plummer sphere sampled from its distribution function (Aarseth, Henon and Wielen 1974), so
/// velocities are in virial equilibrium on average. Centre of mass and momentum are removed.
pub fn plummer(gpa: std.mem.Allocator, rng: std.Random, n: usize, o: Plummer) ![]Body3 {
    const a = if (o.henon) 3.0 * std.math.pi / 16.0 else o.a;
    const total = if (o.henon) 1.0 else o.mass;
    const out = try gpa.alloc(Body3, n);
    errdefer gpa.free(out);
    for (out) |*b| {
        var r: f64 = undefined;
        while (true) {
            const x = rng.float(f64);
            if (x < 1e-12) continue;
            r = a / @sqrt(std.math.pow(f64, x, -2.0 / 3.0) - 1);
            if (r <= o.cut * a) break;
        }
        // q = v / v_esc from g(q) = q^2 (1 - q^2)^(7/2), maximum below 0.1.
        var q: f64 = undefined;
        while (true) {
            q = rng.float(f64);
            if (0.1 * rng.float(f64) < q * q * std.math.pow(f64, 1 - q * q, 3.5)) break;
        }
        const v = q * @sqrt(2.0 * total / a) * std.math.pow(f64, 1 + r * r / (a * a), -0.25);
        const p = unitVector(rng);
        const w = unitVector(rng);
        b.* = .{ .x = r * p[0], .y = r * p[1], .z = r * p[2], .vx = v * w[0], .vy = v * w[1], .vz = v * w[2], .m = total / @as(f64, @floatFromInt(n)) };
    }
    recentre(out);
    return out;
}

fn unitVector(rng: std.Random) [3]f64 {
    const z = 2 * rng.float(f64) - 1;
    const phi = 2 * std.math.pi * rng.float(f64);
    const s = @sqrt(1 - z * z);
    return .{ s * @cos(phi), s * @sin(phi), z };
}

fn recentre(bodies: []Body3) void {
    var c: [6]f64 = @splat(0);
    var m: f64 = 0;
    for (bodies) |b| {
        c[0] += b.m * b.x;
        c[1] += b.m * b.y;
        c[2] += b.m * b.z;
        c[3] += b.m * b.vx;
        c[4] += b.m * b.vy;
        c[5] += b.m * b.vz;
        m += b.m;
    }
    for (&c) |*v| v.* /= m;
    for (bodies) |*b| {
        b.x -= c[0];
        b.y -= c[1];
        b.z -= c[2];
        b.vx -= c[3];
        b.vy -= c[4];
        b.vz -= c[5];
    }
}

pub const Kepler = struct {
    m1: f64 = 1,
    m2: f64 = 1,
    /// Semi-major axis of the relative orbit.
    a: f64 = 1,
    e: f64 = 0,
    g: f64 = 1,
};

/// Two bodies in the orbital plane z = 0, centre-of-mass frame, starting at apocentre on the x axis.
pub fn kepler(o: Kepler) [2]Body3 {
    const mt = o.m1 + o.m2;
    const r = o.a * (1 + o.e);
    const v = @sqrt(o.g * mt * (1 - o.e) / (o.a * (1 + o.e)));
    return .{
        .{ .x = -o.m2 / mt * r, .y = 0, .z = 0, .vx = 0, .vy = -o.m2 / mt * v, .vz = 0, .m = o.m1 },
        .{ .x = o.m1 / mt * r, .y = 0, .z = 0, .vx = 0, .vy = o.m1 / mt * v, .vz = 0, .m = o.m2 },
    };
}

pub fn keplerPeriod(o: Kepler) f64 {
    return 2 * std.math.pi * @sqrt(o.a * o.a * o.a / (o.g * (o.m1 + o.m2)));
}

pub const Uniform = struct {
    /// Cube side; the cube is centred on the origin.
    side: f64 = 1,
    mass: f64 = 1,
    /// Per-component Gaussian velocity dispersion; 0 is cold.
    sigma: f64 = 0,
};

pub fn uniform(gpa: std.mem.Allocator, rng: std.Random, n: usize, o: Uniform) ![]Body3 {
    const out = try gpa.alloc(Body3, n);
    for (out) |*b| b.* = .{
        .x = (rng.float(f64) - 0.5) * o.side,
        .y = (rng.float(f64) - 0.5) * o.side,
        .z = (rng.float(f64) - 0.5) * o.side,
        .vx = o.sigma * rng.floatNorm(f64),
        .vy = o.sigma * rng.floatNorm(f64),
        .vz = o.sigma * rng.floatNorm(f64),
        .m = o.mass / @as(f64, @floatFromInt(n)),
    };
    return out;
}

/// The u32 position frame: world = (u / 2^32 - 0.5) * size, centred on the origin (the renderer's
/// convention). An open domain is the middle half of that cube, so `size` is twice the domain side;
/// see sim3.zig.
pub const Box = struct { size: f64 };

/// Open-boundary box around the bodies. Shifts them in place so the bounding cube is centred on
/// the origin, then grows it by `margin` (>= 1) so bodies can spread before reaching the wall.
pub fn fitBox(bodies: []Body3, margin: f64) Box {
    var lo: [3]f64 = @splat(std.math.inf(f64));
    var hi: [3]f64 = @splat(-std.math.inf(f64));
    for (bodies) |b| {
        const p = [3]f64{ b.x, b.y, b.z };
        for (0..3) |k| {
            lo[k] = @min(lo[k], p[k]);
            hi[k] = @max(hi[k], p[k]);
        }
    }
    var side: f64 = 0;
    for (0..3) |k| side = @max(side, hi[k] - lo[k]);
    const mid = [3]f64{ 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]) };
    for (bodies) |*b| {
        b.x -= mid[0];
        b.y -= mid[1];
        b.z -= mid[2];
    }
    return .{ .size = 2 * side * margin };
}
