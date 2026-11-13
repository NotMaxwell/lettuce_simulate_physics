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
