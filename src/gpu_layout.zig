//! Data contract shared by host code (SDL_GPU) and Zig→SPIR-V shaders.
//! Everything here is `extern` with std430-compatible layout: vec2 = [2]f32 (8-byte aligned),
//! vec4 = [4]f32 (16-byte aligned), no implicit padding. Shader and host import this same file,
//! so a layout change cannot drift between the two.
//!
//! Balls live on the GPU as struct-of-arrays storage buffers (one buffer per field group below),
//! sized for `max_balls` and indexed by ball id. Passes that only need positions touch only `pos`.

pub const max_balls: u32 = 1 << 20;

/// Workgroup size for every 1D compute kernel.
pub const group_size: u32 = 256;

pub const Vec2 = extern struct { x: f32, y: f32 };

/// Per-ball field groups; each is its own storage buffer of `max_balls` elements.
pub const BallPos = Vec2; // current position (world units)
pub const BallPrevPos = Vec2; // position at previous tick, for render interpolation
pub const BallVel = Vec2; // velocity, world units per tick

pub const BallBody = extern struct {
    radius: f32,
    mass: f32,
    /// Bounce energy kept on wall/ball impact (0..1).
    retention: f32,
    friction: f32,
};

pub const BallState = extern struct {
    /// Degrees, counter-clockwise positive (pygame convention, matches world.zig).
    angle: f32,
    prev_angle: f32,
    ang_vel: f32,
    /// Per-ball downward gravity when mutual gravity is off.
    gravity: f32,
};

pub const BallHeat = extern struct {
    temp: f32,
    burn: f32,
    _pad0: f32 = 0,
    _pad1: f32 = 0,
};

/// Packed: bits 0..7 level, bits 8..15 flags (world.BallFlags bit order: alive, burning,
/// scored, grabbed, thrown), bits 16..31 reserved.
pub const BallMeta = u32;

pub const flag_alive: u32 = 1 << 8;
pub const flag_burning: u32 = 1 << 9;
pub const flag_scored: u32 = 1 << 10;
pub const flag_grabbed: u32 = 1 << 11;
pub const flag_thrown: u32 = 1 << 12;

/// Scalar fields, not Vec2: std140 aligns struct members inside uniform blocks to 16 bytes.
/// Uniform block pushed every simulation tick (SDL_PushGPUComputeUniformData slot 0).
pub const SimParams = extern struct {
    ball_count: u32,
    tick: u32,
    /// Spatial grid: cell size in world units (>= 2 * largest radius) and dimensions.
    grid_cells_x: u32,
    grid_cells_y: u32,
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    cell_size: f32,
    mutual_g: f32,
    wall: f32,
    flags: u32, // sim_flag_* below
    mouse_x: f32,
    mouse_y: f32,
    attraction_strength: f32,
    /// Merge candidate list capacity (header excluded); 0 = the full max_events - 1. A test hook:
    /// a small value forces the overflow path with few balls.
    event_cap: u32 = 0,
};

/// SimParams.flags bits. gpu_sim/common.zig re-exports the first three under its old names.
pub const sim_flag_mutual_gravity: u32 = 1 << 0;
pub const sim_flag_merging: u32 = 1 << 1;
pub const sim_flag_attract: u32 = 1 << 2;
/// Reserved: set in galaxy mode but read by no kernel yet. Keep the bit.
pub const sim_flag_galaxy: u32 = 1 << 3;
/// Collision response, two bits: solid (default solver), accrete (any touching pair merges,
/// conserving mass and momentum), ghost (no contact forces; gravity softening only).
pub const sim_collide_shift: u5 = 4;
/// Solid contacts heat the balls involved (physics.zig resolvePair impact heat); needs a heat and an
/// fx buffer attached to the solver (gpu_sim Sim.heat_buf / fx_buf). Off in galaxy mode.
pub const sim_flag_impact_heat: u32 = 1 << 6;
pub const sim_collide_mask: u32 = 3 << sim_collide_shift;
pub const CollideMode = enum(u32) { solid = 0, accrete = 1, ghost = 2 };

/// GPU→CPU event record appended by kernels (merge, ignite, explode, sound...).
/// Read back one frame late; the CPU applies game logic and audio from these.
pub const Event = extern struct {
    kind: u32,
    a: u32,
    b: u32,
    value: f32,
};

pub const event_merge: u32 = 1;
pub const event_death: u32 = 2;
pub const event_explosion: u32 = 3;
pub const event_score: u32 = 4;

/// Merge candidates one tick can hold (plus the header). A candidate that does not fit is not
/// lost: merging pairs get no contact response, so they still overlap and are found again next tick.
pub const max_events: u32 = 1 << 15;

/// Generic textured, rotatable quad for the sprite pipeline (backgrounds, explosions, HUD,
/// glyphs, particles while still CPU-side). Balls are drawn by a dedicated pipeline that reads
/// the ball buffers directly instead of these.
pub const Sprite = extern struct {
    center: Vec2, // screen pixels
    half: Vec2, // half width/height in pixels
    uv_min: Vec2,
    uv_max: Vec2,
    color: [4]f32, // multiplied with the texel; rgba 0..1
    /// Radians, counter-clockwise on screen.
    angle: f32,
    _pad0: f32 = 0,
    _pad1: f32 = 0,
    _pad2: f32 = 0,
};

/// Untextured/textured triangle-list vertex for trails, rounded rects, lasso.
pub const Vertex = extern struct {
    pos: Vec2, // screen pixels
    uv: Vec2,
    color: [4]f32,
};

comptime {
    const std = @import("std");
    std.debug.assert(@sizeOf(Sprite) == 64);
    std.debug.assert(@sizeOf(Vertex) == 32);
    std.debug.assert(@sizeOf(SimParams) == 64);
    std.debug.assert(@sizeOf(Event) == 16);
}

// 3D matter sim (src/sim/sim3.zig, kernels in src/sim/k3/); layouts follow docs/sim3d.md.
// Scalar members only: these cross the SDL_GPU storage-buffer and uniform boundaries unpadded.

/// Position as per-axis fractions of the root cube (value / 2^32). Wrapped subtraction is the periodic minimum image.
pub const Pos3 = extern struct { x: u32, y: u32, z: u32 };
pub const Vel3 = extern struct { x: f32, y: f32, z: f32 };
/// Sorted-key record (Morton key split in two words, original index).
pub const SortRec = extern struct { key_hi: u32, key_lo: u32, idx: u32 };
/// Binary radix tree node. child: left child, or bit31 | first body of a leaf; skip: rope; info: count | depth.
pub const Node = extern struct { com: Pos3, mass: f32, open_r2: f32, child: u32, skip: u32, info: u32 };
pub const NodeQuad = extern struct { xx: f32, xy: f32, xz: f32, yy: f32, yz: f32, zz: f32 };

pub const sim3_flag_periodic: u32 = 1 << 0;
/// Kick kernel: write per-body potential (diagnostic ticks).
pub const sim3_flag_potential: u32 = 1 << 1;
/// Kick kernel: store the acceleration into vel instead of adding kick * a (force tests).
pub const sim3_flag_store_accel: u32 = 1 << 2;

/// Uniform block of the 3D sim kernels (slot 0).
pub const Params3 = extern struct {
    n: u32,
    /// First body of this dispatch (long kicks are split into chunks for the driver watchdog).
    first: u32,
    flags: u32,
    _pad: u32 = 0,
    g: f32,
    eps: f32,
    eps2: f32,
    /// vel += kick * a, with kick = dt * (1 or 1/2).
    kick: f32,
    /// Quanta per unit velocity: dt * 2^32 / box_size.
    drift_q: f32,
    /// Length per position quantum: box_size / 2^32.
    box_scale: f32,
    dt: f32,
    /// Time-step accuracy parameter for the rung histogram.
    eta: f32,
};

comptime {
    const std = @import("std");
    std.debug.assert(@sizeOf(Pos3) == 12);
    std.debug.assert(@sizeOf(Vel3) == 12);
    std.debug.assert(@sizeOf(SortRec) == 12);
    std.debug.assert(@sizeOf(Node) == 32);
    std.debug.assert(@sizeOf(NodeQuad) == 24);
    std.debug.assert(@sizeOf(Params3) == 48);
}
