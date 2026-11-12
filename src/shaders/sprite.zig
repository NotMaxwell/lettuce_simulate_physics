//! Render shaders for the generic 2D pipelines (compiled to SPIR-V by build.zig).
//! Resource sets follow SDL_GPU's SPIR-V rules: vertex stage set 0 = storage buffers,
//! set 1 = uniforms; fragment stage set 2 = sampled textures.
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");

const Vec4 = @Vector(4, f32);

/// Uniform blocks follow std140: keep members scalar or vec4-aligned.
pub const View = extern struct { width: f32, height: f32, _pad0: f32 = 0, _pad1: f32 = 0 };

const SpriteBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.Sprite }) };
const VertexBuf = extern struct { items: @SpirvType(.{ .runtime_array = L.Vertex }) };

const sprites = @extern(*addrspace(.storage_buffer) const SpriteBuf, .{ .name = "sprites", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const verts = @extern(*addrspace(.storage_buffer) const VertexBuf, .{ .name = "verts", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const view = @extern(*addrspace(.uniform) const View, .{ .name = "view", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });

const out_uv = @extern(*addrspace(.output) @Vector(2, f32), .{ .name = "out_uv", .decoration = .{ .location = 0 } });
const out_color = @extern(*addrspace(.output) Vec4, .{ .name = "out_color", .decoration = .{ .location = 1 } });

fn toClip(px: f32, py: f32) Vec4 {
    // Screen pixels (y down) to NDC (y up); SDL_GPU's Vulkan backend flips the viewport so NDC y is up.
    return .{ px / view.width * 2.0 - 1.0, 1.0 - py / view.height * 2.0, 0.0, 1.0 };
}

/// Instanced quads: 6 vertices per instance, no vertex buffer.
export fn spriteVs() callconv(.spirv_vertex) void {
    const s = sprites.items[spirv.instance_index];
    const v = spirv.vertex_index;
    // Two triangles (0,1,2) (2,1,3) over corners 0=(-,-) 1=(+,-) 2=(-,+) 3=(+,+).
    const corner: u32 = (@as(u32, 0x312210) >> @intCast(v * 4)) & 3;
    const cx: f32 = if (corner & 1 != 0) 1.0 else -1.0;
    const cy: f32 = if (corner & 2 != 0) 1.0 else -1.0;
    const lx = cx * s.half.x;
    const ly = cy * s.half.y;
    const cs = @cos(s.angle);
    const sn = @sin(s.angle);
    // Counter-clockwise on screen with y down means the y term is negated.
    const px = s.center.x + lx * cs + ly * sn;
    const py = s.center.y - lx * sn + ly * cs;
    spirv.position_out.* = toClip(px, py);
    out_uv.* = .{
        if (corner & 1 != 0) s.uv_max.x else s.uv_min.x,
        if (corner & 2 != 0) s.uv_max.y else s.uv_min.y,
    };
    out_color.* = .{ s.color[0], s.color[1], s.color[2], s.color[3] };
}

/// Plain triangle list read from a storage buffer.
export fn triVs() callconv(.spirv_vertex) void {
    const p = verts.items[spirv.vertex_index];
    spirv.position_out.* = toClip(p.pos.x, p.pos.y);
    out_uv.* = .{ p.uv.x, p.uv.y };
    out_color.* = .{ p.color[0], p.color[1], p.color[2], p.color[3] };
}

const Texture = @SpirvType(.{ .sampled_image = @SpirvType(.{ .image = .{
    .usage = .{ .sampled = f32 },
    .format = .unknown,
    .dim = .@"2d",
    .depth = .not_depth,
    .access = .unknown,
    .arrayed = false,
    .multisampled = false,
} }) });
const tex = @extern(*addrspace(.constant) const Texture, .{ .name = "tex", .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });

const in_uv = @extern(*addrspace(.input) const @Vector(2, f32), .{ .name = "in_uv", .decoration = .{ .location = 0 } });
const in_color = @extern(*addrspace(.input) const Vec4, .{ .name = "in_color", .decoration = .{ .location = 1 } });
const frag_out = @extern(*addrspace(.output) Vec4, .{ .name = "frag_out", .decoration = .{ .location = 0 } });

export fn spriteFs() callconv(.{ .spirv_fragment = .{} }) void {
    frag_out.* = spirv.imageSampleImplicitLod(tex, in_uv.*) * in_color.*;
}
