//! Offline renderer kernels: splat bodies into a u32 mass grid, find its maximum, tone-map to RGBA8.
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");
const S = @import("shared.zig");


fn Rt(T: type) type {
    return extern struct { v: @SpirvType(.{ .runtime_array = T }) };
}
const pos = @extern(*addrspace(.storage_buffer) const Rt(L.Vec2), .{ .name = "pos", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const mass = @extern(*addrspace(.storage_buffer) const Rt(f32), .{ .name = "mass", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
const grid = @extern(*addrspace(.storage_buffer) Rt(u32), .{ .name = "grid", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });
const maxv = @extern(*addrspace(.storage_buffer) Rt(u32), .{ .name = "maxv", .decoration = .{ .descriptor = .{ .set = 1, .binding = 1 } } });
const img = @extern(*addrspace(.storage_buffer) Rt(u32), .{ .name = "img", .decoration = .{ .descriptor = .{ .set = 1, .binding = 2 } } });
const params = @extern(*addrspace(.uniform) const S.RenderParams, .{ .name = "params", .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });

const cc: std.builtin.CallingConvention = .{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } };

inline fn gid() u32 {
    return spirv.global_invocation_id[0] + spirv.global_invocation_id[1] * (S.dispatch_x * L.group_size);
}

inline fn atomicAdd(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
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

inline fn atomicUMax(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
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

export fn clear() callconv(cc) void {
    const i = gid();
    if (i < params.width * params.height) grid.v[i] = 0;
    if (i == 0) maxv.v[0] = 0;
}

export fn splat() callconv(cc) void {
    const i = gid();
    if (i >= params.count) return;
    const w: f32 = @floatFromInt(params.width);
    const h: f32 = @floatFromInt(params.height);
    const fx = @floor((pos.v[i].x - params.cx) * params.scale + 0.5 * w);
    const fy = @floor(0.5 * h - (pos.v[i].y - params.cy) * params.scale);
    // Written so NaN fails the test and skips the body.
    if (!(fx >= 0 and fx < w and fy >= 0 and fy < h)) return;
    const wt = @min(mass.v[i] * params.weight_scale + 0.5, 4.0e6);
    _ = atomicAdd(&grid.v[@as(u32, @intFromFloat(fy)) * params.width + @as(u32, @intFromFloat(fx))], @intFromFloat(wt));
}

export fn findMax() callconv(cc) void {
    const n = params.width * params.height;
    const stride = S.dispatch_x * L.group_size;
    var i = gid();
    var m: u32 = 0;
    while (i < n) : (i += stride) m = @max(m, grid.v[i]);
    if (m > 0) _ = atomicUMax(&maxv.v[0], m);
}

inline fn sat(x: f32) f32 {
    return @min(@max(x, 0.0), 1.0);
}

export fn tonemap() callconv(cc) void {
    const i = gid();
    if (i >= params.width * params.height) return;
    const v: f32 = @floatFromInt(grid.v[i]);
    const top: f32 = @floatFromInt(@max(maxv.v[0], 1));
    const t = if (params.log_scale != 0) @log(1.0 + v) / @log(1.0 + top) else v / top;
    // Dark blue, blue, red, yellow, white.
    const lo = sat(3.0 * t);
    const r = sat(3.0 * t - 1.0);
    const g = sat(3.0 * t - 2.0);
    const b = lo * (1.0 - r) * 0.9 + g;
    const R: u32 = @intFromFloat(sat(r) * 255.0 + 0.5);
    const G: u32 = @intFromFloat(sat(g) * 255.0 + 0.5);
    const B: u32 = @intFromFloat(sat(b) * 255.0 + 0.5);
    img.v[i] = R | (G << 8) | (B << 16) | (255 << 24);
}
