//! Shared helpers for the game-logic kernels (merge apply, spawn, pick/grab, lasso, explosions).
//! Same conventions as src/gpu_sim/common.zig, duplicated because a kernel module cannot import
//! files outside its own directory.
const std = @import("std");
pub const spirv = std.spirv;
pub const L = @import("gpu_layout");
pub const cfg = @import("config");
pub const Vec2 = L.Vec2;

pub fn Arr(comptime T: type) type {
    return extern struct { v: @SpirvType(.{ .runtime_array = T }) };
}

pub fn ro(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 0, .binding = binding } } });
}

pub fn rw(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 1, .binding = binding } } });
}

/// Mirrored by gpu_world.zig GameParams. Scalars only (std140).
pub const GameParams = extern struct {
    /// Ball ids in use (high-water mark).
    ball_count: u32,
    /// Per-tick claim stamp for merges (tick + 1, never 0 so a zeroed claim buffer is free).
    stamp: u32,
    /// Items in the per-dispatch input buffer (spawn records, polygon points, blasts).
    items: u32,
    out_cap: u32,
    mouse_x: f32,
    mouse_y: f32,
    max_radius: f32,
    hoop_y: f32,
    hoop_left: f32,
    hoop_right: f32,
    lo_x: f32,
    lo_y: f32,
    hi_x: f32,
    hi_y: f32,
    /// Spatial grid of the last solver step (cell starts / sorted ids in gpu_sim's buffers).
    grid_cells_x: u32,
    grid_cells_y: u32,
    /// Particle pool capacity - 1 (a power of two), for the pool being emitted into or updated.
    pool_mask: u32,
    /// bit0: the conduction pass ran this tick (burn reads its scratch temperatures).
    fx_flags: u32,
    /// 1 = accrete: the merge kernel merges any touching pair, summing mass (L.CollideMode).
    collide: u32 = 0,
    /// gpu_layout SimParams.event_cap (0 = full), so merge reads the candidate list as it was written.
    event_cap: u32 = 0,
    _pad2: u32 = 0,
    _pad3: u32 = 0,
    _pad4: u32 = 0,
    _pad5: u32 = 0,
};

pub const params = @extern(*addrspace(.uniform) GameParams, .{ .name = "gparams", .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });

/// New ball written by the spawn kernel into slot `id`. Mirrored by gpu_world.zig.
pub const SpawnRec = extern struct {
    id: u32,
    meta: u32,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    radius: f32,
    mass: f32,
    retention: f32,
    friction: f32,
    gravity: f32,
    temp: f32,
};

/// Explosion knockback request: centre and blast radius.
pub const Blast = extern struct { x: f32, y: f32, blast: f32, force: f32 };

pub const kernel_cc: std.builtin.CallingConvention = .{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } };

pub inline fn gid() u32 {
    return spirv.global_invocation_id[0];
}

// Zig's SPIR-V backend has no @atomicRmw/@cmpxchg lowering yet, so these are inline asm.
// Device scope, relaxed: results are consumed by later passes that SDL separates with barriers.

pub inline fn atomicAdd(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
    return asm volatile (
        \\%ret = OpAtomicIAdd %u32 %p %scope %sem %v
        : [ret] "" (-> u32),
        : [u32] "t" (u32),
          [p] "" (p),
          [scope] "" (@as(u32, @intFromEnum(spirv.Scope.device))),
          [sem] "" (@as(u32, 0)),
          [v] "" (v),
    );
}

pub inline fn atomicExchange(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
    return asm volatile (
        \\%ret = OpAtomicExchange %u32 %p %scope %sem %v
        : [ret] "" (-> u32),
        : [u32] "t" (u32),
          [p] "" (p),
          [scope] "" (@as(u32, @intFromEnum(spirv.Scope.device))),
          [sem] "" (@as(u32, 0)),
          [v] "" (v),
    );
}

pub inline fn atomicUMax(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
    return asm volatile (
        \\%ret = OpAtomicUMax %u32 %p %scope %sem %v
        : [ret] "" (-> u32),
        : [u32] "t" (u32),
          [p] "" (p),
          [scope] "" (@as(u32, @intFromEnum(spirv.Scope.device))),
          [sem] "" (@as(u32, 0)),
          [v] "" (v),
    );
}

pub const max_level: u32 = cfg.max_level;
pub const id_bits: u32 = 20;
pub const id_mask: u32 = (1 << id_bits) - 1;

pub const event_burnout: u32 = 5;
/// Set by `burn` on a ball that just burnt out; `emitAsh` clears it in the same tick.
pub const flag_ashing: u32 = 1 << 13;
/// Meta bit of pool entries: the ball shader draws them as flat coloured squares whose colour is
/// the four floats of their heat record.
pub const flag_particle: u32 = 1 << 16;

/// fx counter slots (u32 each). Written by the fire kernels, read back with each tick's events.
pub const fx_burning = 0;
pub const fx_warm = 1;
pub const fx_fire_alive = 2;
pub const fx_ash_alive = 3;
pub const fx_fire_head = 4;
pub const fx_ash_head = 5;
pub const fx_prev_burning = 6;
pub const fx_prev_warm = 7;
pub const fx_deaths = 8;
pub const fx_prev_deaths = 9;
/// Written by the solver's finish pass (gpu_sim/common.zig fx_impact): tick + 1 of the last tick
/// in which impact heat warmed a ball, so the host wakes the fire passes.
pub const fx_impact = 10;
pub const fx_slots = 16;

pub inline fn lid() u32 {
    return spirv.local_invocation_id[0];
}

const red = @extern(*addrspace(.shared) [L.group_size]u32, .{ .name = "red" });

/// Sum of `v` over the workgroup, on every thread. All threads must call it (it has barriers).
pub inline fn blockSum(v: u32) u32 {
    const l = lid();
    red[l] = v;
    spirv.workgroupBarrier();
    var s: u32 = L.group_size / 2;
    while (s > 0) : (s >>= 1) {
        if (l < s) red[l] += red[l + s];
        spirv.workgroupBarrier();
    }
    // Not `red[0]`: Zig's SPIR-V backend emits a comptime index 0 into a shared array as an
    // access chain typed as the whole array (invalid SPIR-V), which segfaults RADV's
    // vkCreateComputePipelines. A runtime index (always 0 here) emits a correct chain.
    const total = red[l - l];
    spirv.workgroupBarrier();
    return total;
}

/// Maximum of `v` over the workgroup, on every thread (same hazards and calling rule as blockSum).
pub inline fn blockMax(v: u32) u32 {
    const l = lid();
    red[l] = v;
    spirv.workgroupBarrier();
    var s: u32 = L.group_size / 2;
    while (s > 0) : (s >>= 1) {
        if (l < s) red[l] = @max(red[l], red[l + s]);
        spirv.workgroupBarrier();
    }
    const top = red[l - l];
    spirv.workgroupBarrier();
    return top;
}

pub inline fn hash(seed: u32) u32 {
    var x = seed;
    x ^= x >> 16;
    x *%= 0x7feb352d;
    x ^= x >> 15;
    x *%= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

/// Uniform in [0, 1).
pub inline fn unit(h: u32) f32 {
    return @as(f32, @floatFromInt(h >> 8)) * (1.0 / 16777216.0);
}

/// Next value of a per-thread random stream.
pub inline fn next(state: *u32) f32 {
    state.* = hash(state.* +% 0x9e3779b9);
    return unit(state.*);
}

pub inline fn density(mass: f32, radius: f32) f32 {
    return @max(0.0, (mass - 20.0) / (4.0 * radius));
}

pub const fire_life: f32 = cfg.fire_particle_life;
pub const ash_life: f32 = cfg.ash_lifetime_ticks;
