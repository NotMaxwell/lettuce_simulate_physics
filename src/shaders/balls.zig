//! Ball vertex shader: one instance per ball id, reads the physics storage buffers directly.
//! Resource sets (SDL_GPU SPIR-V rules): vertex set 0 = storage buffers, set 1 = uniforms.
//! The fragment stage is `spriteFs` from sprite.zig (same out_uv/out_color interface).
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");

const Vec4 = @Vector(4, f32);

/// std140 uniform; scalars only. Mirrored by `gfx.BallUniforms`.
pub const Params = extern struct {
    view_w: f32,
    view_h: f32,
    alpha: f32,
    /// World-to-screen: screen = o + (world - b) * scale (render.zig Camera).
    scale: f32,
    ox: f32,
    oy: f32,
    bx: f32,
    by: f32,
    room_temp: f32,
    ignition_temp: f32,
    max_burn: f32,
    /// Highest ball level; tile table index `tile_count - 1` is the basketball.
    max_level: f32,
};

const V2Buf = extern struct { items: @SpirvType(.{ .runtime_array = L.Vec2 }) };
const BodyBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.BallBody }) };
const StateBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.BallState }) };
const HeatBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.BallHeat }) };
const MetaBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.BallMeta }) };
/// uv_min.x, uv_min.y, uv_max.x, uv_max.y per tile.
const TileBuf = extern struct { items: @SpirvType(.{ .runtime_array = [4]f32 }) };

fn sb(comptime T: type, comptime name: [:0]const u8, comptime binding: u32) *addrspace(.storage_buffer) const T {
    return @extern(*addrspace(.storage_buffer) const T, .{ .name = name, .decoration = .{ .descriptor = .{ .set = 0, .binding = binding } } });
}
const pos = sb(V2Buf, "pos", 0);
const prev_pos = sb(V2Buf, "prev_pos", 1);
const body = sb(BodyBuf, "body", 2);
const state = sb(StateBuf, "state", 3);
const heat_buf = sb(HeatBuf, "heat", 4);
const meta = sb(MetaBuf, "meta", 5);
const tiles = sb(TileBuf, "tiles", 6);
const p = @extern(*addrspace(.uniform) const Params, .{ .name = "p", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });

const out_uv = @extern(*addrspace(.output) @Vector(2, f32), .{ .name = "out_uv", .decoration = .{ .location = 0 } });
const out_color = @extern(*addrspace(.output) Vec4, .{ .name = "out_color", .decoration = .{ .location = 1 } });

/// Smallest drawn half-size in pixels (a 1.5 px square).
const min_half_px: f32 = 0.75;
/// Below this on-screen radius the texture window shrinks towards the tile centre.
const shrink_px: f32 = 1.5;
/// ball meta bit of particle pool entries (gpu_game/common.zig flag_particle): drawn as flat
/// squares coloured by their heat record's four floats.
const flag_particle: u32 = 1 << 16;

inline fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

export fn ballVs() callconv(.spirv_vertex) void {
    const id = spirv.instance_index;
    const m = meta.items[id];
    if (m & L.flag_alive == 0) {
        spirv.position_out.* = .{ 0, 0, 0, 1 };
        out_uv.* = .{ 0, 0 };
        out_color.* = .{ 0, 0, 0, 0 };
        return;
    }

    const v = spirv.vertex_index;
    // Two triangles (0,1,2) (2,1,3) over corners 0=(-,-) 1=(+,-) 2=(-,+) 3=(+,+).
    const corner: u32 = (@as(u32, 0x312210) >> @intCast(v * 4)) & 3;
    const cx: f32 = if (corner & 1 != 0) 1.0 else -1.0;
    const cy: f32 = if (corner & 2 != 0) 1.0 else -1.0;

    const particle = m & flag_particle != 0;
    const a = p.alpha;
    const wx = lerp(prev_pos.items[id].x, pos.items[id].x, a);
    const wy = lerp(prev_pos.items[id].y, pos.items[id].y, a);
    const r_world = body.items[id].radius * p.scale;
    // Particles keep angle fields for their own state, so they never rotate.
    const ang = if (particle) 0.0 else lerp(state.items[id].prev_angle, state.items[id].angle, a) * (std.math.pi / 180.0);
    const cs = @cos(ang);
    const sn = @sin(ang);
    // A ball smaller than min_half_px still covers a pixel, so a zoomed-out swarm stays solid;
    // below shrink_px the texture window narrows to the tile centre (fully inside the circle
    // mask) so the covered pixel gets the ball's colour at full opacity rather than a mip
    // average that fades with the mask's edge.
    const r = @max(r_world, min_half_px);
    const lx = cx * r;
    const ly = cy * r;
    // Counter-clockwise on screen with y down means the y term is negated.
    const px = p.ox + (wx - p.bx) * p.scale + lx * cs + ly * sn;
    const py = p.oy + (wy - p.by) * p.scale - lx * sn + ly * cs;
    spirv.position_out.* = .{ px / p.view_w * 2.0 - 1.0, 1.0 - py / p.view_h * 2.0, 0.0, 1.0 };

    var level: f32 = @floatFromInt(m & 0xff);
    level = @min(@max(level, 1.0), p.max_level);
    const tile: u32 = if (m & L.flag_scored != 0) @as(u32, @intFromFloat(p.max_level)) else @as(u32, @intFromFloat(level)) - 1;
    const uv = tiles.items[tile];
    var uv0 = uv[0];
    var uv1 = uv[1];
    var uv2 = uv[2];
    var uv3 = uv[3];
    if (!particle and r_world < shrink_px) {
        const k = @max(r_world / shrink_px, 0.15);
        const mu = (uv[0] + uv[2]) * 0.5;
        const mv = (uv[1] + uv[3]) * 0.5;
        uv0 = mu + (uv0 - mu) * k;
        uv2 = mu + (uv2 - mu) * k;
        uv1 = mv + (uv1 - mv) * k;
        uv3 = mv + (uv3 - mv) * k;
    }
    out_uv.* = .{
        if (corner & 1 != 0) uv2 else uv0,
        if (corner & 2 != 0) uv3 else uv1,
    };

    const h = heat_buf.items[id];
    if (particle) {
        out_color.* = .{ h.temp, h.burn, h._pad0, h._pad1 };
        return;
    }
    const heat = @min(@max((h.temp - p.room_temp) / (p.ignition_temp - p.room_temp), 0.0), 1.0);
    const char = 1.0 - @min(@max(h.burn / p.max_burn, 0.0), 1.0);
    const gb = (1.0 - heat * 0.5) * char;
    out_color.* = .{ char, gb, gb, 1.0 };
}
