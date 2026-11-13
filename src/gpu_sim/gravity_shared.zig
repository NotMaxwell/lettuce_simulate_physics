//! Constants and layouts shared by the gravity kernels (gravity_kernels.zig) and host (gravity.zig).
const L = @import("gpu_layout");

/// Expansion order (terms). Measured in the harness: 6 gives ~2e-5 median / <1e-3 p99 relative
/// force error; 8 is ~4x more accurate and ~30% slower at 1M balls; 5 is ~4x less accurate.
pub const order = 6;
pub const scan_block: u32 = 1024; // cells per workgroup in scanBlocks (4 per thread)
/// Default eps^2 (physics.zig GravityJob eps); the caller passes the real value in Params.softening.
pub const softening: f32 = 0.001;
pub const min_level: u32 = 2;
pub const max_level: u32 = 12;

pub const Params = extern struct {
    n: u32,
    leaf_level: u32,
    /// Level the current pass works on (m2m / m2l).
    level: u32,
    /// Plummer eps^2 (world units^2) of the near-field pair force.
    softening: f32,
    // Scalars, not a Vec2 struct: std140 rounds a struct member up to 16 bytes, so a Vec2 here
    // would overlap the next field in the uniform block.
    origin_x: f32,
    origin_y: f32,
    root_size: f32,
    g_scale: f32,
    /// Word index in `cells` of a GPU-computed tree box [origin_x, origin_y, root_size] (f32 bits),
    /// or no_box to use origin_x/origin_y/root_size above.
    box_word: u32 = no_box,
    /// 1: boundsPartial clamps positions into the window [origin, origin + root_size]^2 before
    /// taking the box, so the GPU box is tight but never larger than the window.
    clamp_window: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
    /// Potential pass: sampled bodies (== n when exact), each pair sum split into `chunks` slices
    /// of `chunk_len` bodies, sample choice hashed with `seed`.
    samples: u32 = 0,
    chunks: u32 = 0,
    chunk_len: u32 = 0,
    seed: u32 = 0,
};

/// Bodies one thread covers in diagPartial.
pub const diag_per_thread: u32 = 16;
/// Threads the potential pass aims for.
pub const potential_threads: u32 = 131072;
/// Words of the `cells` buffer after the tree box where diagFinal leaves its result:
/// kinetic energy, angular momentum, mass, potential sum (f32 bits).
pub const diag_result_words: u32 = 4;
pub const diag_result_offset: u32 = 4;

pub const no_box: u32 = 0xFFFF_FFFF;
/// Bodies one thread covers in boundsPartial.
pub const bounds_per_thread: u32 = 16;
/// Workgroups per row of a 2D dispatch: x is capped (65535 on some drivers), so passes over more
/// than dispatch_x * group_size threads spill into y. Kernels recover the flat id from both.
pub const dispatch_x: u32 = 32768;

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
