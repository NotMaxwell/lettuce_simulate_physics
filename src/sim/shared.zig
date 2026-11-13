//! Constants and layouts shared by the sim kernels (kernels.zig) and host (sim.zig).
//! Kernel modules see only their own directory, so this file sits next to them.
const L = @import("gpu_layout");

/// Bodies one thread covers in diagPartial.
pub const diag_per_thread: u32 = 16;
/// Floats per workgroup in the diagnostics partials: kinetic energy, px, py, angular momentum, mass, 3 spare.
pub const diag_stride: u32 = 8;
/// Workgroups per row of a 2D dispatch (see gpu_sim/gravity_shared.zig dispatch_x).
pub const dispatch_x: u32 = 32768;
/// Threads the potential pass aims for, so a few thousand sampled bodies still fill the GPU.
pub const potential_threads: u32 = 131072;

pub const Params = extern struct {
    n: u32,
    /// Potential pass: sampled body s is s when samples == n, else a hash of (s, seed) mod n, so a
    /// generator that orders bodies systematically cannot alias with the sample.
    samples: u32,
    seed: u32,
    _pad: u32 = 0,
    /// Potential pass: each sampled body's pair sum over all n is split into `chunks` slices of this length.
    chunks: u32,
    chunk_len: u32,
    dt: f32,
    softening: f32,
};

/// Lean body layout the sim needs: position, velocity, mass (in `L.BallBody.mass` for now, because
/// the shared gravity kernels read the mass from there).
pub const bytes_per_body_state = @sizeOf(L.Vec2) * 2 + @sizeOf(L.BallBody);
pub const RenderParams = extern struct {
    count: u32,
    width: u32,
    height: u32,
    log_scale: u32,
    cx: f32,
    cy: f32,
    /// Pixels per world unit.
    scale: f32,
    /// Fixed-point weight of a body: u32(mass * weight_scale).
    weight_scale: f32,
};
