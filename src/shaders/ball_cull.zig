//! Ball culling and level of detail on the GPU, before the ball draw. Each live ball is
//! off-screen (dropped), a point (on-screen radius below cull_common.point_px) or a sprite, and
//! its id goes into one compacted list in id order, so overlapping balls stack exactly as in a
//! draw over every id: sprites from index 0 up, points from `point_top` down.
//!
//! Three passes: cullCount (per-workgroup counts), cullScan (one workgroup: offsets per group and
//! the indirect draw arguments), cullWrite (classify again and write ids). The classification is
//! recomputed rather than stored, which costs a second read of four small fields but no per-ball
//! buffer.
//!
//! Bindings (one layout for all three): ro 0 meta, 1 prev_pos, 2 pos, 3 body; rw 0 group counts,
//! 1 group offsets, 2 indirect args, 3 id list; uniform set 2 = cull_common.CullParams.
//! Compiled into the trail_rec.spv module (see trail_rec.zig).
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");
const C = @import("cull_common.zig");

fn Arr(comptime T: type) type {
    return extern struct { v: @SpirvType(.{ .runtime_array = T }) };
}

fn ro(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 0, .binding = binding } } });
}

fn rw(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 1, .binding = binding } } });
}

const meta = ro(L.BallMeta, 0, "c_meta");
const prev_pos = ro(L.Vec2, 1, "c_prev_pos");
const pos = ro(L.Vec2, 2, "c_pos");
const body = ro(L.BallBody, 3, "c_body");
/// Per workgroup: sprites | points << 16 (each at most group_size).
const grp = rw(u32, 0, "c_grp");
/// Per workgroup: first sprite slot, first point slot (from point_top down).
const offs = rw(u32, 1, "c_offs");
const args = rw(u32, 2, "c_args");
const list = rw(u32, 3, "c_list");
const P = @extern(*addrspace(.uniform) C.CullParams, .{ .name = "cp", .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });

const sh = @extern(*addrspace(.shared) [L.group_size]u32, .{ .name = "c_sh" });

const kernel_cc: std.builtin.CallingConvention = .{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } };

const is_sprite: u32 = 1;
const is_point: u32 = 1 << 16;

inline fn lid() u32 {
    return spirv.local_invocation_id[0];
}

/// Flat workgroup index of a dispatch that may be 2D (gpu_layout dispatch_x).
inline fn groupId() u32 {
    return spirv.workgroup_id[0] + spirv.workgroup_id[1] * L.dispatch_x;
}

/// Inclusive prefix sum of `v` over the workgroup. Every thread must call it (barriers).
inline fn scanIncl(v: u32) u32 {
    const l = lid();
    sh[l] = v;
    spirv.workgroupBarrier();
    var off: u32 = 1;
    while (off < L.group_size) : (off <<= 1) {
        const add = if (l >= off) sh[l - off] else 0;
        spirv.workgroupBarrier();
        sh[l] += add;
        spirv.workgroupBarrier();
    }
    const r = sh[l];
    spirv.workgroupBarrier();
    return r;
}

/// is_sprite, is_point or 0 (dead or off-screen), with balls.zig's camera math. The sprite test
/// is conservative: a rotated quad reaches half * sqrt(2) from its centre.
inline fn classify(i: u32) u32 {
    if (i >= P.n) return 0;
    if (meta.v[i] & L.flag_alive == 0) return 0;
    const q = prev_pos.v[i];
    const s = pos.v[i];
    const a = P.alpha;
    const sx = P.ox + (q.x + (s.x - q.x) * a - P.bx) * P.scale;
    const sy = P.oy + (q.y + (s.y - q.y) * a - P.by) * P.scale;
    const r_px = body.v[i].radius * P.scale;
    if (r_px < C.point_px) {
        return if (sx >= 0 and sx < P.view_w and sy >= 0 and sy < P.view_h) is_point else 0;
    }
    const e = r_px * 1.4143 + 1.0;
    return if (sx + e >= 0 and sx - e <= P.view_w and sy + e >= 0 and sy - e <= P.view_h) is_sprite else 0;
}

export fn cullCount() callconv(kernel_cc) void {
    const g = groupId();
    if (g >= P.groups) return;
    const l = lid();
    const total = scanIncl(classify(g * L.group_size + l));
    if (l == L.group_size - 1) grp.v[g] = total;
}

/// One workgroup: each thread owns a contiguous run of workgroups' counts.
export fn cullScan() callconv(kernel_cc) void {
    const l = lid();
    const n = P.groups;
    const chunk = (n + L.group_size - 1) / L.group_size;
    const lo = @min(l * chunk, n);
    const hi = @min(lo + chunk, n);
    var s: u32 = 0;
    var p: u32 = 0;
    var k = lo;
    while (k < hi) : (k += 1) {
        const v = grp.v[k];
        s += v & 0xffff;
        p += v >> 16;
    }
    const s_incl = scanIncl(s);
    const p_incl = scanIncl(p);
    var rs = s_incl - s;
    var rp = p_incl - p;
    k = lo;
    while (k < hi) : (k += 1) {
        const v = grp.v[k];
        offs.v[2 * k] = rs;
        offs.v[2 * k + 1] = rp;
        rs += v & 0xffff;
        rp += v >> 16;
    }
    if (l == L.group_size - 1) {
        // Runtime base index: a comptime index 0 into a buffer array is an invalid access chain
        // with Zig's SPIR-V backend (see gpu_game/common.zig blockSum).
        const b = l - l;
        args.v[b] = 6;
        args.v[b + 1] = s_incl;
        args.v[b + 2] = 0;
        args.v[b + 3] = 0;
        args.v[b + 4] = p_incl;
        args.v[b + 5] = 1;
        args.v[b + 6] = 0;
        args.v[b + 7] = 0;
    }
}

export fn cullWrite() callconv(kernel_cc) void {
    const g = groupId();
    if (g >= P.groups) return;
    const l = lid();
    const i = g * L.group_size + l;
    const v = classify(i);
    const ex = scanIncl(v) - v;
    if (v == is_sprite) {
        list.v[offs.v[2 * g] + (ex & 0xffff)] = i;
    } else if (v == is_point) {
        list.v[P.point_top - (offs.v[2 * g + 1] + (ex >> 16))] = i;
    }
}
