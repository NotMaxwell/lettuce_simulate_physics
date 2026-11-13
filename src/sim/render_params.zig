//! Uniform block of the offline renderer (render_kernels.zig), shared with render.zig.
//! Kept apart from shared.zig so the renderer owns its own layout.

pub const mode_xy_f32: u32 = 0;
pub const mode_xyz_u32: u32 = 1;

pub const RenderParams = extern struct {
    count: u32,
    width: u32,
    height: u32,
    log_scale: u32,
    /// `mode_xy_f32`: positions are f32 pairs (2D snapshots). `mode_xyz_u32`: u32 triples, fractions of the root cube.
    mode: u32,
    /// World point projected to the image centre.
    cx: f32,
    cy: f32,
    cz: f32,
    /// Pixels per world unit.
    scale: f32,
    /// Fixed-point weight of a body: u32(mass * weight_scale).
    weight_scale: f32,
    /// World units per 2^32 position steps (box_size / 2^32); u32 mode only.
    unit: f32,
    /// > 0: every body has this mass and the mass buffer is ignored.
    uniform_mass: f32,
    /// > 0: keep only bodies with |depth| <= slab_half.
    slab_half: f32,
    /// Orthonormal view rows: screen x, screen y (up) and depth, as coefficients of (world - centre).
    rx: f32,
    ry: f32,
    rz: f32,
    ux: f32,
    uy: f32,
    uz: f32,
    fx: f32,
    fy: f32,
    fz: f32,
    pad0: u32 = 0,
    pad1: u32 = 0,
};
