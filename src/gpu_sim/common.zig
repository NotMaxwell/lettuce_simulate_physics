//! Shared helpers for the gpu_sim compute kernels (compiled to SPIR-V, one module per kernel
//! because SDL_GPU binds resources per pipeline and each kernel has its own binding layout).

const std = @import("std");
pub const spirv = std.spirv;
pub const L = @import("gpu_layout");
pub const Vec2 = L.Vec2;

pub fn Arr(comptime T: type) type {
    return extern struct { v: @SpirvType(.{ .runtime_array = T }) };
}


/// Read-only storage buffer: SDL puts these in descriptor set 0, in binding order.
pub fn ro(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 0, .binding = binding } } });
}

/// Read-write storage buffer: descriptor set 1.
pub fn rw(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 1, .binding = binding } } });
}

/// Uniform block in slot `binding` (descriptor set 2).
pub fn uniform(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.uniform) T {
    return @extern(*addrspace(.uniform) T, .{ .name = name, .decoration = .{ .descriptor = .{ .set = 2, .binding = binding } } });
}

pub const Params = L.SimParams;

pub const params = uniform(Params, 0, "params");

pub inline fn gid() u32 {
    return spirv.global_invocation_id[0];
}

pub inline fn lid() u32 {
    return spirv.local_invocation_id[0];
}

/// Zig's SPIR-V backend does not lower @atomicRmw yet ("TODO (SPIR-V): implement AIR tag
/// atomic_rmw"), so atomics are inline OpAtomicIAdd. Device scope, relaxed semantics: the
/// results are only consumed in later passes, which SDL separates with barriers.
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

/// Grid cell of a position (clamped into the grid, so out-of-area balls still get a cell).
pub inline fn cellOf(p: Vec2, P: *addrspace(.uniform) Params) u32 {
    const inv = 1.0 / P.cell_size;
    const fx = @floor((p.x - P.bounds_min_x) * inv);
    const fy = @floor((p.y - P.bounds_min_y) * inv);
    const mx: f32 = @floatFromInt(P.grid_cells_x - 1);
    const my: f32 = @floatFromInt(P.grid_cells_y - 1);
    // NaN-safe clamp: a NaN position lands in cell 0 instead of an out-of-range index.
    const cx: u32 = @intFromFloat(if (fx > 0) @min(fx, mx) else 0);
    const cy: u32 = @intFromFloat(if (fy > 0) @min(fy, my) else 0);
    return cy * P.grid_cells_x + cx;
}

pub const flag_mutual_gravity = L.sim_flag_mutual_gravity;
pub const flag_merging = L.sim_flag_merging;
pub const flag_attract = L.sim_flag_attract;
pub const flag_impact_heat = L.sim_flag_impact_heat;

/// Temperature gained per unit of speed change over mass: config.zig impact_heat_mult (1.5) times the
/// `* 10` of resolvePair's heat transfer, halved because a contact's velocity change is up to
/// twice the closing speed (see k_finish: the estimate is a lower bound).
pub const impact_heat_k: f32 = 7.5;
/// Velocity changes of one solve below this are not considered for impact heat.
pub const impact_dv_min: f32 = 30.0;
/// Must equal config.zig ignition_temp.
pub const ignition_temp: f32 = 100;
/// gpu_game/common.zig fx_impact: finish stamps tick + 1 here when it lit a ball.
pub const fx_impact = 10;

/// The collision response chosen in SimParams.flags (L.CollideMode as a u32).
pub inline fn collideMode(flags: u32) u32 {
    return (flags & L.sim_collide_mask) >> L.sim_collide_shift;
}
pub const collide_accrete: u32 = @intFromEnum(L.CollideMode.accrete);
pub const collide_ghost: u32 = @intFromEnum(L.CollideMode.ghost);

pub inline fn level(meta: u32) u32 {
    return meta & 0xff;
}

/// Uniform for the scan kernels.
pub const ScanParams = extern struct { n: u32, _pad0: u32 = 0, _pad1: u32 = 0, _pad2: u32 = 0 };
/// Elements one scan workgroup covers: 256 threads x 4 elements each.
pub const scan_per_thread: u32 = 4;
pub const scan_block: u32 = L.group_size * scan_per_thread;

/// Uniform for the collide kernel (on top of SimParams in slot 0).

/// Event buffer layout: element 0 is a header (kind = append count, a = records dropped for
/// lack of space), records follow from element 1.

// Mirrors of config.zig values the kernels need (the shader modules cannot import config.zig,
// which lives outside their module root).
pub const bounce_stop: f32 = 0.3;
// Must equal config.zig max_level. gpu_game/common.zig reads it from a config module that
// build.zig provides only to those kernels; build_kernels.zig gives the gpu_sim kernels none.
pub const max_level: u32 = 38;

/// Inclusive cell range of the 3x3 block around `cell`, clipped to the grid.
pub const Neighbourhood = struct { x0: u32, x1: u32, y0: u32, y1: u32 };

pub inline fn neighbourhood(cell: u32, P: *addrspace(.uniform) Params) Neighbourhood {
    const nx = P.grid_cells_x;
    const cx = cell % nx;
    const cy = cell / nx;
    return .{
        .x0 = if (cx > 0) cx - 1 else 0,
        .x1 = if (cx + 1 < nx) cx + 1 else cx,
        .y0 = if (cy > 0) cy - 1 else 0,
        .y1 = if (cy + 1 < P.grid_cells_y) cy + 1 else cy,
    };
}

/// Uniform for the colored Gauss-Seidel kernel (on top of SimParams in slot 0).
pub const ColorParams = extern struct { color_x: u32, color_y: u32, iteration: u32, rest_speed: f32, elastic: u32, _pad0: u32 = 0, _pad1: u32 = 0, _pad2: u32 = 0 };

/// Keep a ball centre inside the play-area walls (positions only).
pub inline fn clampToWalls(p: Vec2, r: f32) Vec2 {
    const h = params.wall / 2 + r;
    return .{
        .x = @min(@max(p.x, params.bounds_min_x + h), params.bounds_max_x - h),
        .y = @min(@max(p.y, params.bounds_min_y + h), params.bounds_max_y - h),
    };
}

/// Cell-sorted solver record for the colored solver: position, radius (<= 0: dead or empty)
/// and mass in one 16-byte load.
pub const PosRad = extern struct { x: f32, y: f32, r: f32, m: f32 };

