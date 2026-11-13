//! Constants and uniform layouts shared by the trail kernels (trail_rec.zig, trails.zig) and gfx.zig.

/// Ball ids [0, trail_cap) keep a history; higher ids draw no trail.
pub const trail_cap: u32 = 32768;
/// Samples per ball kept in the ring.
pub const trail_len: u32 = 48;
/// A sample is recorded every `record_stride` ticks, so 48 samples cover 96 ticks (1.6 s).
pub const record_stride: u32 = 2;
/// A segment longer than this in world units is a spawn, merge or slot reuse, not motion.
pub const teleport: f32 = 160;
/// x of a sample taken from a dead ball; segments touching it are dropped.
pub const invalid: f32 = 1e30;

/// Compute uniform (std140, scalars only). Ring layout: sample `slot` of ball `i` is `slot * trail_cap + i`.
pub const RecParams = extern struct {
    count: u32,
    slot: u32,
    clear: u32,
    _pad0: u32 = 0,
};

/// Vertex uniform (std140); camera as in gfx.BallCamera.
pub const DrawParams = extern struct {
    view_w: f32,
    view_h: f32,
    alpha: f32,
    scale: f32,
    ox: f32,
    oy: f32,
    bx: f32,
    by: f32,
    /// Ring slot of the newest sample.
    head: u32,
    max_level: f32,
    _pad0: f32 = 0,
    _pad1: f32 = 0,
};
