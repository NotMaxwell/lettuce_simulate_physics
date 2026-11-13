//! Trail vertex shader: instance = ball id * trail_len + segment, six vertices per quad.
//! Segment t joins the sample of age t to the one of age t-1; age -1 is the interpolated ball
//! position, so the newest segment ends on the ball. Vertex storage set 0: 0 ring, 1 pos,
//! 2 prev_pos, 3 body, 4 meta; uniform set 1. Fragment stage is `spriteFs` (additive pipeline).
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");
const T = @import("trail_common.zig");

const Vec2 = @Vector(2, f32);
const Vec4 = @Vector(4, f32);

const V2Buf = extern struct { items: @SpirvType(.{ .runtime_array = L.Vec2 }) };
const BodyBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.BallBody }) };
const MetaBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.BallMeta }) };

fn sb(comptime Ty: type, comptime name: [:0]const u8, comptime binding: u32) *addrspace(.storage_buffer) const Ty {
    return @extern(*addrspace(.storage_buffer) const Ty, .{ .name = name, .decoration = .{ .descriptor = .{ .set = 0, .binding = binding } } });
}
const ring = sb(V2Buf, "ring", 0);
const pos = sb(V2Buf, "pos", 1);
const prev_pos = sb(V2Buf, "prev_pos", 2);
const body = sb(BodyBuf, "body", 3);
const meta = sb(MetaBuf, "meta", 4);
const p = @extern(*addrspace(.uniform) const T.DrawParams, .{ .name = "p", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });

const out_uv = @extern(*addrspace(.output) Vec2, .{ .name = "out_uv", .decoration = .{ .location = 0 } });
const out_color = @extern(*addrspace(.output) Vec4, .{ .name = "out_color", .decoration = .{ .location = 1 } });

/// Peak opacity of the newest end of a trail.
const peak_alpha: f32 = 0.55;

fn sample(id: u32, age: u32) L.Vec2 {
    const slot = (p.head + T.trail_len - age) % T.trail_len;
    return ring.items[slot * T.trail_cap + id];
}

fn hide() void {
    spirv.position_out.* = .{ 0, 0, 0, 1 };
    out_uv.* = .{ 0, 0 };
    out_color.* = .{ 0, 0, 0, 0 };
}

export fn trailVs() callconv(.spirv_vertex) void {
    const inst = spirv.instance_index;
    const id = inst / T.trail_len;
    const t = inst % T.trail_len;
    const m = meta.items[id];
    if (m & L.flag_alive == 0) return hide();

    const older = sample(id, t);
    var newer: L.Vec2 = undefined;
    if (t == 0) {
        newer = .{
            .x = prev_pos.items[id].x + (pos.items[id].x - prev_pos.items[id].x) * p.alpha,
            .y = prev_pos.items[id].y + (pos.items[id].y - prev_pos.items[id].y) * p.alpha,
        };
    } else {
        newer = sample(id, t - 1);
    }
    if (older.x > 1e20 or newer.x > 1e20) return hide();
    const dx = newer.x - older.x;
    const dy = newer.y - older.y;
    if (dx * dx + dy * dy > T.teleport * T.teleport) return hide();

    const ax = p.ox + (older.x - p.bx) * p.scale;
    const ay = p.oy + (older.y - p.by) * p.scale;
    const bx = p.ox + (newer.x - p.bx) * p.scale;
    const by = p.oy + (newer.y - p.by) * p.scale;
    const sx = bx - ax;
    const sy = by - ay;
    const slen = @sqrt(sx * sx + sy * sy);
    const dirx = if (slen > 1e-4) sx / slen else 1.0;
    const diry = if (slen > 1e-4) sy / slen else 0.0;
    const hw = @max(0.3 * body.items[id].radius * p.scale, 0.5);

    const v = spirv.vertex_index;
    const corner: u32 = (@as(u32, 0x312210) >> @intCast(v * 4)) & 3;
    const at_new = corner & 1 != 0;
    const side: f32 = if (corner & 2 != 0) 1.0 else -1.0;
    const px = (if (at_new) bx else ax) - diry * hw * side;
    const py = (if (at_new) by else ay) + dirx * hw * side;
    spirv.position_out.* = .{ px / p.view_w * 2.0 - 1.0, 1.0 - py / p.view_h * 2.0, 0.0, 1.0 };

    const age_v: f32 = if (at_new) @as(f32, @floatFromInt(t)) - 1.0 else @floatFromInt(t);
    const fade = 1.0 - (age_v + 1.0) / @as(f32, @floatFromInt(T.trail_len));

    // Golden-ratio hue steps keep neighbouring levels apart; cosine palette avoids a lookup table.
    const level: f32 = @floatFromInt(@min(@max(m & 0xff, 1), 255));
    const h = level * 0.618034 * 6.2831853;
    out_color.* = .{
        0.55 + 0.45 * @cos(h),
        0.55 + 0.45 * @cos(h + 2.0943951),
        0.55 + 0.45 * @cos(h + 4.1887902),
        fade * peak_alpha,
    };
    out_uv.* = .{ 0.5, 0.5 };
}
