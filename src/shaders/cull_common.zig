//! Layouts shared by the ball cull kernels (ball_cull.zig), the ball vertex shaders (balls.zig)
//! and gfx.zig. Data only, so the host can import it.

/// Below this on-screen radius (pixels) a ball is drawn as a one-pixel point instead of a sprite.
/// The sprite path draws such a ball as a fixed 1.5 px square whose texture window has shrunk to
/// the central quarter of the tile (balls.zig shrink_px), so the point's centre texel matches its
/// colour; between here and 1.5 px the window is wider and the sprite path is kept.
pub const point_px: f32 = 0.375;

/// Compute uniform of the cull kernels (std140, scalars only).
pub const CullParams = extern struct {
    /// Ball ids to classify.
    n: u32,
    /// Workgroups of cullCount / cullWrite (ceil(n / group_size)).
    groups: u32,
    /// Last index of the id list: points are written downwards from here, sprites upwards from 0,
    /// so both fit in one buffer of n entries.
    point_top: u32,
    _pad0: u32 = 0,
    view_w: f32,
    view_h: f32,
    alpha: f32,
    scale: f32,
    ox: f32,
    oy: f32,
    bx: f32,
    by: f32,
};

/// The indirect buffer the scan writes: two SDL_GPUIndirectDrawCommand records
/// {num_vertices, num_instances, first_vertex, first_instance}: sprites (6 vertices per instance)
/// then points (one vertex each).
pub const args_words: u32 = 8;

comptime {
    @import("std").debug.assert(@sizeOf(CullParams) == 48);
}
