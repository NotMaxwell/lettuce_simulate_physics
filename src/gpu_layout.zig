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
    flags: u32, // bit0 mutual gravity, bit1 merging, bit2 attract
    mouse_x: f32,
    mouse_y: f32,
    attraction_strength: f32,
    _pad: f32 = 0,
};

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

pub const max_events: u32 = 1 << 14;

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
