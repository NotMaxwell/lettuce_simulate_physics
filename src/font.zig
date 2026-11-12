//! Glyph atlas built from SDL's own debug font, plus text -> sprite layout for the GPU renderer.
const std = @import("std");
const c = @import("c");
const gpu = @import("gpu_layout");

pub const glyph_px: u32 = 8; // SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE
pub const first_char: u8 = 32;
pub const last_char: u8 = 126;
const glyph_count: u32 = last_char - first_char + 1;
const cols: u32 = 16;
const rows: u32 = (glyph_count + cols - 1) / cols;
/// 1 px gutter around each glyph so linear filtering never bleeds a neighbour in.
const cell: u32 = glyph_px + 2;

pub const Atlas = struct {
    pixels: []u8, // RGBA8, white with alpha coverage
    w: u32,
    h: u32,

    pub fn init(gpa: std.mem.Allocator) !Atlas {
        const w = cols * cell;
        const h = rows * cell;
        const surf = c.SDL_CreateSurface(@intCast(w), @intCast(h), c.SDL_PIXELFORMAT_RGBA32) orelse return error.SdlSurface;
        defer c.SDL_DestroySurface(surf);
        const r = c.SDL_CreateSoftwareRenderer(surf) orelse return error.SdlRenderer;
        defer c.SDL_DestroyRenderer(r);

        _ = c.SDL_SetRenderDrawColor(r, 0, 0, 0, 0);
        _ = c.SDL_RenderClear(r);
        _ = c.SDL_SetRenderDrawColor(r, 255, 255, 255, 255);
        var i: u32 = 0;
        while (i < glyph_count) : (i += 1) {
            const s = [2:0]u8{ @intCast(first_char + i), 0 };
            const ox: f32 = @floatFromInt((i % cols) * cell + 1);
            const oy: f32 = @floatFromInt((i / cols) * cell + 1);
            _ = c.SDL_RenderDebugText(r, ox, oy, &s);
        }
        _ = c.SDL_FlushRenderer(r);

        const pixels = try gpa.alloc(u8, w * h * 4);
        errdefer gpa.free(pixels);
        const pitch: usize = @intCast(surf.*.pitch);
        const src: [*]const u8 = @ptrCast(surf.*.pixels.?);
        for (0..h) |y| @memcpy(pixels[y * w * 4 ..][0 .. w * 4], src[y * pitch ..][0 .. w * 4]);
        return .{ .pixels = pixels, .w = w, .h = h };
    }

    pub fn deinit(self: *Atlas, gpa: std.mem.Allocator) void {
        gpa.free(self.pixels);
        self.* = undefined;
    }
};

pub fn textWidth(len: usize, scale: f32) f32 {
    return @as(f32, @floatFromInt(len)) * @as(f32, @floatFromInt(glyph_px)) * scale;
}

/// One sprite per non-space glyph; (x, y) is the top-left of the first glyph. Characters outside
/// 32..126 are drawn as '?'. Stops silently when `out` is full. UVs span exact texel boundaries,
/// which is crisp under nearest sampling and stays inside the gutter under linear.
pub fn layout(atlas: *const Atlas, text: []const u8, x: f32, y: f32, scale: f32, color: [4]f32, out: []gpu.Sprite) usize {
    const gw: f32 = @floatFromInt(glyph_px);
    const half = gw * scale * 0.5;
    const aw: f32 = @floatFromInt(atlas.w);
    const ah: f32 = @floatFromInt(atlas.h);
    var n: usize = 0;
    for (text, 0..) |ch0, k| {
        if (ch0 == ' ') continue;
        if (n == out.len) break;
        const ch = if (ch0 > first_char and ch0 <= last_char) ch0 else '?';
        const idx: u32 = ch - first_char;
        const tu: f32 = @floatFromInt((idx % cols) * cell + 1);
        const tv: f32 = @floatFromInt((idx / cols) * cell + 1);
        out[n] = .{
            .center = .{ .x = x + (@as(f32, @floatFromInt(k)) * gw * scale) + half, .y = y + half },
            .half = .{ .x = half, .y = half },
            .uv_min = .{ .x = tu / aw, .y = tv / ah },
            .uv_max = .{ .x = (tu + gw) / aw, .y = (tv + gw) / ah },
            .color = color,
            .angle = 0,
        };
        n += 1;
    }
    return n;
}

test "layout skips spaces and positions glyphs" {
    const atlas = Atlas{ .pixels = &.{}, .w = cols * cell, .h = rows * cell };
    var out: [8]gpu.Sprite = undefined;
    const n = layout(&atlas, "A B", 10, 20, 2, .{ 1, 0, 0, 1 }, &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(f32, 18), out[0].center.x);
    try std.testing.expectEqual(@as(f32, 28), out[0].center.y);
    try std.testing.expectEqual(@as(f32, 8), out[0].half.x);
    try std.testing.expectEqual(@as(f32, 10 + 16 * 2 + 8), out[1].center.x);
    try std.testing.expectEqual(@as(f32, 1), out[1].color[0]);
    try std.testing.expectEqual(@as(f32, 0), out[0].angle);
}

test "layout uvs are inside the atlas and 8 texels wide" {
    const atlas = Atlas{ .pixels = &.{}, .w = cols * cell, .h = rows * cell };
    var out: [96]gpu.Sprite = undefined;
    var s: [95]u8 = undefined;
    for (&s, 0..) |*ch, i| ch.* = @intCast(33 + i % 94);
    const n = layout(&atlas, s[0..94], 0, 0, 1, .{ 1, 1, 1, 1 }, &out);
    try std.testing.expectEqual(@as(usize, 94), n);
    const aw: f32 = @floatFromInt(atlas.w);
    for (out[0..n]) |sp| {
        try std.testing.expect(sp.uv_min.x >= 0 and sp.uv_max.x <= 1 and sp.uv_min.y >= 0 and sp.uv_max.y <= 1);
        try std.testing.expectApproxEqAbs(@as(f32, 8), (sp.uv_max.x - sp.uv_min.x) * aw, 1e-3);
    }
}

test "layout truncates to out and textWidth matches" {
    const atlas = Atlas{ .pixels = &.{}, .w = cols * cell, .h = rows * cell };
    var out: [3]gpu.Sprite = undefined;
    try std.testing.expectEqual(@as(usize, 3), layout(&atlas, "abcdef", 0, 0, 1, .{ 1, 1, 1, 1 }, &out));
    try std.testing.expectEqual(@as(f32, 24), textWidth(3, 1));
}
