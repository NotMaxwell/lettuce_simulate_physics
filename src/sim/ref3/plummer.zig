//! Plummer sphere sampler for the ref3 tests and the standalone measurement. Private to ref3: the
//! game-facing generator is src/ic3.zig.
const std = @import("std");

pub const Vec = [3]f64;

/// Equal-mass Plummer positions (scale radius 1), rejected beyond `cut` so they fit a cube of half-width `cut`.
pub fn positions(gpa: std.mem.Allocator, n: usize, seed: u64, cut: f64) ![]Vec {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try gpa.alloc(Vec, n);
    for (out) |*p| while (true) {
        const u = r.float(f64);
        const rad = 1.0 / @sqrt(std.math.pow(f64, @max(u, 1e-12), -2.0 / 3.0) - 1.0);
        if (rad >= cut * 0.999) continue;
        const cz = 2 * r.float(f64) - 1;
        const ph = 2 * std.math.pi * r.float(f64);
        const sz = @sqrt(1 - cz * cz);
        p.* = .{ rad * sz * @cos(ph), rad * sz * @sin(ph), rad * cz };
        break;
    };
    return out;
}

/// Maps a coordinate in [-cut, cut) to the u32 box fraction.
pub fn quantise(x: f64, cut: f64) u32 {
    const f = (x + cut) / (2 * cut) * 4294967296.0;
    return @intFromFloat(std.math.clamp(f, 0, 4294967295.0));
}

pub const Sample = struct {
    pos: []Vec,
    vel: []Vec,
    pub fn deinit(s: Sample, gpa: std.mem.Allocator) void {
        gpa.free(s.pos);
        gpa.free(s.vel);
    }
};

/// Equal-mass Plummer sphere in Henon units (G = M = 1, scale radius 3 pi / 16, virial radius 1) with
/// Aarseth-Henon-Wielen velocities, so 2K/|W| = 1 in expectation. Positions beyond `cut` are rejected.
pub fn withVelocities(gpa: std.mem.Allocator, n: usize, seed: u64, cut: f64) !Sample {
    const a = 3.0 * std.math.pi / 16.0;
    const pos = try positions(gpa, n, seed, cut / a);
    errdefer gpa.free(pos);
    const vel = try gpa.alloc(Vec, n);
    var prng = std.Random.DefaultPrng.init(seed ^ 0x9e3779b97f4a7c15);
    const r = prng.random();
    for (pos, vel) |*p, *v| {
        for (p) |*c| c.* *= a;
        const rad = @sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        const ve = @sqrt(2.0) / std.math.pow(f64, rad * rad + a * a, 0.25);
        const q = while (true) {
            const q = r.float(f64);
            if (0.1 * r.float(f64) < q * q * std.math.pow(f64, 1 - q * q, 3.5)) break q;
        };
        const cz = 2 * r.float(f64) - 1;
        const ph = 2 * std.math.pi * r.float(f64);
        const sz = @sqrt(1 - cz * cz);
        const s = q * ve;
        v.* = .{ s * sz * @cos(ph), s * sz * @sin(ph), s * cz };
    }
    return .{ .pos = pos, .vel = vel };
}

test "plummer: Henon-unit sample is in virial equilibrium" {
    const gpa = std.testing.allocator;
    const n = 10_000;
    const s = try withVelocities(gpa, n, 3, 20);
    defer s.deinit(gpa);
    var k: f64 = 0;
    for (s.vel) |v| k += 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / n;
    var w: f64 = 0;
    const eps2 = 1e-8;
    for (s.pos, 0..) |p, i| for (s.pos[i + 1 ..]) |q| {
        const d = [3]f64{ p[0] - q[0], p[1] - q[1], p[2] - q[2] };
        w -= 1.0 / (@as(f64, n) * n * @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + eps2));
    };
    try std.testing.expectApproxEqAbs(@as(f64, 1), 2 * k / @abs(w), 0.05);
}
