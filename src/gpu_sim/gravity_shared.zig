//! Constants and layouts shared by the gravity kernels (gravity_kernels.zig) and host (gravity.zig).
const L = @import("gpu_layout");

/// Expansion order (terms). Measured in the harness: 6 gives ~2e-5 median / <1e-3 p99 relative
/// force error; 8 is ~4x more accurate and ~30% slower at 1M balls; 5 is ~4x less accurate.
pub const order = 6;
pub const scan_block: u32 = 1024; // cells per workgroup in scanBlocks (4 per thread)
pub const softening: f32 = 0.001; // physics.zig GravityJob eps
pub const min_level: u32 = 2;
pub const max_level: u32 = 10;

pub const Params = extern struct {
    n: u32,
    leaf_level: u32,
    /// Level the current pass works on (m2m / m2l).
    level: u32,
    _pad: u32 = 0,
    // Scalars, not a Vec2 struct: std140 rounds a struct member up to 16 bytes, so a Vec2 here
    // would overlap the next field in the uniform block.
    origin_x: f32,
    origin_y: f32,
    root_size: f32,
    g_scale: f32,
};

pub const Key = extern struct { cell: u32, rank: u32 };
pub const Sorted = extern struct { x: f32, y: f32, m: f32, id: u32 };

/// Cells region layout in the `cells` u32 buffer.
pub fn countsOffset() u32 {
    return 0;
}
pub fn scanOffset(leaf_cells: u32) u32 {
    return leaf_cells;
}
pub fn blockOffset(leaf_cells: u32) u32 {
    return 2 * leaf_cells;
}
pub fn cellsWords(leaf_cells: u32) u32 {
    return 2 * leaf_cells + (leaf_cells + scan_block - 1) / scan_block;
}
/// First cell of `level` in the pyramid buffers (levels stored root-first).
pub fn levelOffset(level: u32) u32 {
    return ((@as(u32, 1) << @intCast(2 * level)) - 1) / 3;
}
