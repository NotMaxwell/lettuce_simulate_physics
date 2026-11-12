//! Batched GPU renderer: every visual layer is one SDL_RenderGeometry call
//! (balls, trails, particles, HUD...), so cost scales with vertex fill, not draw calls.
const std = @import("std");
const c = @import("c");
const world = @import("world.zig");
const cfg = @import("config.zig");

pub const Perf = struct { fps: f32 = 0, tick_ms: f32 = 0, render_ms: f32 = 0 };

pub const Hud = struct {
    balls_per_click: u32 = 1,
    countdown: i32 = 120,
    won: bool = false,
    attraction_energy: f32 = 75,
    mutual_gravity: bool = true,
    merging: bool = true,
    muted: bool = false,
    lasso: []const world.Vec2 = &.{},
    lasso_alpha: f32 = 1,
    show_perf: bool = true,
    perf: Perf = .{},
};

const Vertex = c.SDL_Vertex;
const tile_px = 256;
const gutter = 1;
const cell_px = tile_px + 2 * gutter;
const atlas_cols = 8;
const atlas_rows = 5;
const atlas_w = atlas_cols * cell_px;
const atlas_h = atlas_rows * cell_px;
/// Levels 1..max_level, then the basketball.
const tile_count = cfg.max_level + 1;
const basketball_tile = cfg.max_level;
const expl_frames = 12;
const expl_frame_ticks = 6;
const ash_life_ticks = 180;
const hud_w: f32 = cfg.base_w;
const hud_h: f32 = cfg.base_h;

const Color = struct { r: f32, g: f32, b: f32, a: f32 = 1 };

fn rgb8(r: u8, g: u8, b: u8) Color {
    return .{ .r = @as(f32, @floatFromInt(r)) / 255, .g = @as(f32, @floatFromInt(g)) / 255, .b = @as(f32, @floatFromInt(b)) / 255 };
}

fn fcolor(col: Color) c.SDL_FColor {
    return .{ .r = col.r, .g = col.g, .b = col.b, .a = col.a };
}

/// Growable vertex/index arrays that keep their capacity between frames.
const Batch = struct {
    v: []Vertex = &.{},
    ix: []c_int = &.{},
    nv: usize = 0,
    ni: usize = 0,

    fn deinit(b: *Batch, gpa: std.mem.Allocator) void {
        gpa.free(b.v);
        gpa.free(b.ix);
    }

    fn reserve(b: *Batch, gpa: std.mem.Allocator, add_v: usize, add_i: usize) !void {
        if (b.nv + add_v > b.v.len) b.v = try gpa.realloc(b.v, @max(b.nv + add_v, b.v.len * 2));
        if (b.ni + add_i > b.ix.len) b.ix = try gpa.realloc(b.ix, @max(b.ni + add_i, b.ix.len * 2));
    }

    inline fn vert(b: *Batch, x: f32, y: f32, col: c.SDL_FColor, u: f32, vv: f32) void {
        b.v[b.nv] = .{ .position = .{ .x = x, .y = y }, .color = col, .tex_coord = .{ .x = u, .y = vv } };
        b.nv += 1;
    }

    /// Triangles (0,1,2) and (0,2,3) over the last four vertices written.
    inline fn quadIndices(b: *Batch) void {
        const base: c_int = @intCast(b.nv - 4);
        const ids = [6]c_int{ 0, 1, 2, 0, 2, 3 };
        inline for (ids, 0..) |k, j| b.ix[b.ni + j] = base + k;
        b.ni += 6;
    }

    fn rect(b: *Batch, x: f32, y: f32, w: f32, h: f32, col: Color) void {
        const fc = fcolor(col);
        b.vert(x, y, fc, 0, 0);
        b.vert(x + w, y, fc, 0, 0);
        b.vert(x + w, y + h, fc, 0, 0);
        b.vert(x, y + h, fc, 0, 0);
        b.quadIndices();
    }

    /// Axis-aligned textured quad; u/v are the tile or frame rectangle.
    fn texRect(b: *Batch, x: f32, y: f32, w: f32, h: f32, uv: [4]f32, col: c.SDL_FColor) void {
        b.vert(x, y, col, uv[0], uv[1]);
        b.vert(x + w, y, col, uv[2], uv[1]);
        b.vert(x + w, y + h, col, uv[2], uv[3]);
        b.vert(x, y + h, col, uv[0], uv[3]);
        b.quadIndices();
    }

    const corner_steps = 6;
    const round_rect_verts = 1 + 4 * (corner_steps + 1);
    const round_rect_idx = 3 * 4 * (corner_steps + 1);

    fn roundRect(b: *Batch, x: f32, y: f32, w: f32, h: f32, radius: f32, col: Color) void {
        const r = @min(radius, w / 2, h / 2);
        const fc = fcolor(col);
        const centre: c_int = @intCast(b.nv);
        b.vert(x + w / 2, y + h / 2, fc, 0, 0);
        const corners = [4][3]f32{
            .{ x + w - r, y + r, -std.math.pi / 2.0 },
            .{ x + w - r, y + h - r, 0 },
            .{ x + r, y + h - r, std.math.pi / 2.0 },
            .{ x + r, y + r, std.math.pi },
        };
        for (corners) |cn| {
            for (0..corner_steps + 1) |k| {
                const a = cn[2] + (std.math.pi / 2.0) * @as(f32, @floatFromInt(k)) / corner_steps;
                b.vert(cn[0] + r * @cos(a), cn[1] + r * @sin(a), fc, 0, 0);
            }
        }
        const n: c_int = 4 * (corner_steps + 1);
        var i: c_int = 0;
        while (i < n) : (i += 1) {
            b.ix[b.ni] = centre;
            b.ix[b.ni + 1] = centre + 1 + i;
            b.ix[b.ni + 2] = centre + 1 + @mod(i + 1, n);
            b.ni += 3;
        }
    }

    fn flush(b: *Batch, r: *c.SDL_Renderer, tex: ?*c.SDL_Texture) void {
        if (b.ni > 0) _ = c.SDL_RenderGeometry(r, tex, b.v.ptr, @intCast(b.nv), b.ix.ptr, @intCast(b.ni));
        b.nv = 0;
        b.ni = 0;
    }
};

const Camera = struct {
    s: f32 = 1,
    ox: f32 = 0,
    oy: f32 = 0,
    bx: f32 = 0,
    by: f32 = 0,

    fn fit(b: world.Rect, out_w: f32, out_h: f32) Camera {
        const s = @min(out_w / b.w, out_h / b.h);
        return .{ .s = s, .ox = (out_w - b.w * s) / 2, .oy = (out_h - b.h * s) / 2, .bx = b.x, .by = b.y };
    }
    inline fn sx(cam: Camera, x: f32) f32 {
        return cam.ox + (x - cam.bx) * cam.s;
    }
    inline fn sy(cam: Camera, y: f32) f32 {
        return cam.oy + (y - cam.by) * cam.s;
    }
};

inline fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

const Image = struct {
    px: []u8,
    w: usize,
    h: usize,

    fn load(path: [:0]const u8) !Image {
        var size: usize = 0;
        const data = c.SDL_LoadFile(path.ptr, &size) orelse return error.AssetMissing;
        defer c.SDL_free(data);
        var w: c_int = 0;
        var h: c_int = 0;
        var ch: c_int = 0;
        const px = c.stbi_load_from_memory(@ptrCast(data), @intCast(size), &w, &h, &ch, 4) orelse return error.AssetDecode;
        const n: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 4;
        return .{ .px = px[0..n], .w = @intCast(w), .h = @intCast(h) };
    }

    fn free(img: Image) void {
        c.stbi_image_free(img.px.ptr);
    }
};

/// Area-average resize of an RGBA image to a square `dst` x `dst` (aspect is squashed on purpose,
/// matching pygame.transform.scale in the original).
fn boxResize(gpa: std.mem.Allocator, img: Image, dst: usize, out: []u8) !void {
    const tmp = try gpa.alloc(f32, dst * img.h * 4);
    defer gpa.free(tmp);
    const sw: f32 = @floatFromInt(img.w);
    const sh: f32 = @floatFromInt(img.h);
    const d: f32 = @floatFromInt(dst);
    for (0..img.h) |y| {
        for (0..dst) |i| {
            const lo = @as(f32, @floatFromInt(i)) * sw / d;
            const hi = @as(f32, @floatFromInt(i + 1)) * sw / d;
            var acc = [4]f32{ 0, 0, 0, 0 };
            var px: usize = @intFromFloat(@floor(lo));
            while (@as(f32, @floatFromInt(px)) < hi and px < img.w) : (px += 1) {
                const pf: f32 = @floatFromInt(px);
                const wgt = @min(pf + 1, hi) - @max(pf, lo);
                for (0..4) |k| acc[k] += wgt * @as(f32, @floatFromInt(img.px[(y * img.w + px) * 4 + k]));
            }
            for (0..4) |k| tmp[(y * dst + i) * 4 + k] = acc[k] / (hi - lo);
        }
    }
    for (0..dst) |j| {
        const lo = @as(f32, @floatFromInt(j)) * sh / d;
        const hi = @as(f32, @floatFromInt(j + 1)) * sh / d;
        for (0..dst) |i| {
            var acc = [4]f32{ 0, 0, 0, 0 };
            var py: usize = @intFromFloat(@floor(lo));
            while (@as(f32, @floatFromInt(py)) < hi and py < img.h) : (py += 1) {
                const pf: f32 = @floatFromInt(py);
                const wgt = @min(pf + 1, hi) - @max(pf, lo);
                for (0..4) |k| acc[k] += wgt * tmp[(py * dst + i) * 4 + k];
            }
            for (0..4) |k| out[(j * dst + i) * 4 + k] = @intFromFloat(std.math.clamp(acc[k] / (hi - lo) + 0.5, 0, 255));
        }
    }
}

fn maskCircle(tile: []u8) void {
    const half: f32 = tile_px / 2.0;
    for (0..tile_px) |y| {
        for (0..tile_px) |x| {
            const dx = @as(f32, @floatFromInt(x)) + 0.5 - half;
            const dy = @as(f32, @floatFromInt(y)) + 0.5 - half;
            const cover = std.math.clamp(half - @sqrt(dx * dx + dy * dy) + 0.5, 0, 1);
            const a = &tile[(y * tile_px + x) * 4 + 3];
            a.* = @intFromFloat(@as(f32, @floatFromInt(a.*)) * cover + 0.5);
        }
    }
}

fn texFromPixels(sdl: *c.SDL_Renderer, px: []const u8, w: usize, h: usize) !*c.SDL_Texture {
    const tex = c.SDL_CreateTexture(sdl, c.SDL_PIXELFORMAT_RGBA32, c.SDL_TEXTUREACCESS_STATIC, @intCast(w), @intCast(h)) orelse return error.Texture;
    errdefer c.SDL_DestroyTexture(tex);
    if (!c.SDL_UpdateTexture(tex, null, px.ptr, @intCast(w * 4))) return error.Texture;
    _ = c.SDL_SetTextureScaleMode(tex, c.SDL_SCALEMODE_LINEAR);
    _ = c.SDL_SetTextureBlendMode(tex, c.SDL_BLENDMODE_BLEND);
    return tex;
}

pub const Renderer = struct {
    gpa: std.mem.Allocator,
    sdl: *c.SDL_Renderer,
    atlas: *c.SDL_Texture,
    bg: *c.SDL_Texture,
    bg_w: f32,
    bg_h: f32,
    expl: *c.SDL_Texture,
    tile_uv: [tile_count][4]f32,
    tris: Batch = .{},
    quads: Batch = .{},
    cam: Camera = .{},
    out_w: f32 = hud_w,
    out_h: f32 = hud_h,
    rng: u64 = 0x9E3779B97F4A7C15,

    pub fn init(gpa: std.mem.Allocator, sdl: *c.SDL_Renderer) !Renderer {
        _ = c.SDL_SetRenderDrawBlendMode(sdl, c.SDL_BLENDMODE_BLEND);

        const atlas_px = try gpa.alloc(u8, atlas_w * atlas_h * 4);
        defer gpa.free(atlas_px);
        @memset(atlas_px, 0);
        const tile = try gpa.alloc(u8, tile_px * tile_px * 4);
        defer gpa.free(tile);

        var uv: [tile_count][4]f32 = undefined;
        for (0..tile_count) |k| {
            var buf: [64]u8 = undefined;
            const path = if (k == basketball_tile)
                try std.mem.printSentinel(&buf, "assets/balls/basketball.png", .{}, 0)
            else
                try std.mem.printSentinel(&buf, "assets/balls/{d}.png", .{k + 1}, 0);
            const img = try Image.load(path);
            defer img.free();
            try boxResize(gpa, img, tile_px, tile);
            maskCircle(tile);

            const cx0 = (k % atlas_cols) * cell_px;
            const cy0 = (k / atlas_cols) * cell_px;
            for (0..cell_px) |y| {
                const sy = std.math.clamp(@as(isize, @intCast(y)) - gutter, 0, tile_px - 1);
                for (0..cell_px) |x| {
                    const sx = std.math.clamp(@as(isize, @intCast(x)) - gutter, 0, tile_px - 1);
                    const src = (@as(usize, @intCast(sy)) * tile_px + @as(usize, @intCast(sx))) * 4;
                    const dst = ((cy0 + y) * atlas_w + cx0 + x) * 4;
                    @memcpy(atlas_px[dst..][0..4], tile[src..][0..4]);
                }
            }
            const aw: f32 = atlas_w;
            const ah: f32 = atlas_h;
            uv[k] = .{
                @as(f32, @floatFromInt(cx0 + gutter)) / aw,
                @as(f32, @floatFromInt(cy0 + gutter)) / ah,
                @as(f32, @floatFromInt(cx0 + gutter + tile_px)) / aw,
                @as(f32, @floatFromInt(cy0 + gutter + tile_px)) / ah,
            };
        }
        const atlas = try texFromPixels(sdl, atlas_px, atlas_w, atlas_h);
        errdefer c.SDL_DestroyTexture(atlas);

        const bg_img = try Image.load("assets/background/background.png");
        defer bg_img.free();
        const bg = try texFromPixels(sdl, bg_img.px, bg_img.w, bg_img.h);
        errdefer c.SDL_DestroyTexture(bg);

        const ex_img = try Image.load("assets/explosion/explosion.png");
        defer ex_img.free();
        const expl = try texFromPixels(sdl, ex_img.px, ex_img.w, ex_img.h);

        return .{
            .gpa = gpa,
            .sdl = sdl,
            .atlas = atlas,
            .bg = bg,
            .bg_w = @floatFromInt(bg_img.w),
            .bg_h = @floatFromInt(bg_img.h),
            .expl = expl,
            .tile_uv = uv,
        };
    }

    pub fn deinit(self: *Renderer) void {
        c.SDL_DestroyTexture(self.atlas);
        c.SDL_DestroyTexture(self.bg);
        c.SDL_DestroyTexture(self.expl);
        self.tris.deinit(self.gpa);
        self.quads.deinit(self.gpa);
    }

    pub fn screenToWorld(self: *const Renderer, w: *const world.World, sx: f32, sy: f32) world.Vec2 {
        _ = w;
        const cam = self.cam;
        return .{ .x = (sx - cam.ox) / cam.s + cam.bx, .y = (sy - cam.oy) / cam.s + cam.by };
    }

    fn nextSeed(self: *Renderer) u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        return self.rng | 1;
    }

    pub fn draw(self: *Renderer, w: *const world.World, alpha: f32, hud: *const Hud) void {
        self.drawInner(w, alpha, hud) catch {};
    }

    fn drawInner(self: *Renderer, w: *const world.World, alpha: f32, hud: *const Hud) !void {
        const sdl = self.sdl;
        var iw: c_int = 0;
        var ih: c_int = 0;
        _ = c.SDL_GetCurrentRenderOutputSize(sdl, &iw, &ih);
        self.out_w = @floatFromInt(@max(iw, 1));
        self.out_h = @floatFromInt(@max(ih, 1));

        const pb = w.prev_bounds;
        const nb = w.bounds;
        const bounds: world.Rect = .{
            .x = lerp(pb.x, nb.x, alpha),
            .y = lerp(pb.y, nb.y, alpha),
            .w = lerp(pb.w, nb.w, alpha),
            .h = lerp(pb.h, nb.h, alpha),
        };
        const cam = Camera.fit(bounds, self.out_w, self.out_h);
        self.cam = cam;

        _ = c.SDL_SetRenderScale(sdl, 1, 1);
        _ = c.SDL_SetRenderDrawColor(sdl, 12, 12, 16, 255);
        _ = c.SDL_RenderClear(sdl);

        try self.drawBackground(bounds, cam);
        try self.drawTrails(w, cam);
        try self.drawBalls(w, alpha, cam);
        try self.drawParticles(w, alpha, cam);
        try self.drawHud(w, hud);
        try self.drawExplosions(w, cam);
        self.drawText(w, hud);
    }

    fn drawBackground(self: *Renderer, b: world.Rect, cam: Camera) !void {
        // The original window showed the image's top-left base_w x base_h crop; that crop is
        // stretched over the whole play area so the hoop stays where physics expects it.
        const q = &self.quads;
        try q.reserve(self.gpa, 4, 6);
        q.texRect(cam.sx(b.x), cam.sy(b.y), b.w * cam.s, b.h * cam.s, .{ 0, 0, cfg.base_w / self.bg_w, cfg.base_h / self.bg_h }, fcolor(.{ .r = 1, .g = 1, .b = 1 }));
        q.flush(self.sdl, self.bg);
        const t = &self.tris;
        try t.reserve(self.gpa, 16, 24);
        const wc = rgb8(70, 72, 84);
        const l = cam.sx(b.x);
        const tp = cam.sy(b.y);
        const bw = b.w * cam.s;
        const bh = b.h * cam.s;
        const th = @max(cfg.wall * cam.s, 1);
        t.rect(l, tp, bw, th, wc);
        t.rect(l, tp + bh - th, bw, th, wc);
        t.rect(l, tp + th, th, bh - 2 * th, wc);
        t.rect(l + bw - th, tp + th, th, bh - 2 * th, wc);
        t.flush(self.sdl, null);
    }

    const trail_threads = 4;
    const trail_vpb = cfg.max_trail * 2;
    const trail_ipb = (cfg.max_trail - 1) * 6;

    const TrailJob = struct {
        sl: std.MultiArrayList(world.Ball).Slice,
        lo: usize,
        hi: usize,
        cam: Camera,
        stride: usize,
        rng: u64,
        verts: []Vertex,
        idx: []c_int,
        nv: usize = 0,
        ni: usize = 0,

        fn rand01(job: *@This()) f32 {
            job.rng ^= job.rng << 13;
            job.rng ^= job.rng >> 7;
            job.rng ^= job.rng << 17;
            return @as(f32, @floatFromInt(@as(u32, @truncate(job.rng >> 32)) >> 8)) / 16777216.0;
        }

        /// Writes ribbons for balls lo..hi; indices are relative to this job's own vertex slice.
        fn run(job: *@This()) void {
            const trails = job.sl.items(.trail);
            const lens = job.sl.items(.trail_len);
            const heads = job.sl.items(.trail_head);
            const flags = job.sl.items(.flags);
            const cam = job.cam;
            for (job.lo..job.hi) |i| {
                if (!flags[i].alive) continue;
                const len: usize = lens[i];
                if (len < 2) continue;
                const start: usize = (@as(usize, heads[i]) + cfg.max_trail - len) % cfg.max_trail;
                const ring = &trails[i];
                const first = ring[start];
                const last = ring[(start + len - 1) % cfg.max_trail];
                const ddx = last.x - first.x;
                const ddy = last.y - first.y;
                if (ddx * ddx + ddy * ddy < 0.01) continue;

                const col = fcolor(.{ .r = job.rand01(), .g = job.rand01(), .b = job.rand01() });
                // Oldest-to-newest screen-space points, taking every `stride`-th one when zoomed out
                // (segments are then sub-pixel anyway) but always keeping the newest.
                var sxs: [cfg.max_trail]f32 = undefined;
                var sys: [cfg.max_trail]f32 = undefined;
                var m: usize = 0;
                var k: usize = len - 1;
                while (true) {
                    const p = ring[(start + k) % cfg.max_trail];
                    sxs[m] = cam.sx(p.x);
                    sys[m] = cam.sy(p.y);
                    m += 1;
                    if (k < job.stride) break;
                    k -= job.stride;
                }
                if (m < 2) continue;
                std.mem.reverse(f32, sxs[0..m]);
                std.mem.reverse(f32, sys[0..m]);

                const base: c_int = @intCast(job.nv);
                const vp = job.verts[job.nv..][0 .. 2 * m];
                const ip = job.idx[job.ni..][0 .. 6 * (m - 1)];
                var nx: f32 = 0;
                var ny: f32 = 1;
                const inv_m1 = 1.0 / @as(f32, @floatFromInt(m - 1));
                for (0..m) |q| {
                    const dx = sxs[@min(q + 1, m - 1)] - sxs[q -| 1];
                    const dy = sys[@min(q + 1, m - 1)] - sys[q -| 1];
                    const l2 = dx * dx + dy * dy;
                    if (l2 > 1e-8) {
                        const inv = 1.0 / @sqrt(l2);
                        nx = -dy * inv;
                        ny = dx * inv;
                    }
                    const wpx = @floor(2 + @as(f32, @floatFromInt(q)) * inv_m1 * 4);
                    const hw = @max(wpx * cam.s, 1) * 0.5;
                    vp[2 * q] = .{ .position = .{ .x = sxs[q] + nx * hw, .y = sys[q] + ny * hw }, .color = col, .tex_coord = .{} };
                    vp[2 * q + 1] = .{ .position = .{ .x = sxs[q] - nx * hw, .y = sys[q] - ny * hw }, .color = col, .tex_coord = .{} };
                }
                job.nv += 2 * m;
                for (0..m - 1) |sg| {
                    const a: c_int = base + @as(c_int, @intCast(2 * sg));
                    ip[6 * sg ..][0..6].* = .{ a, a + 1, a + 2, a + 1, a + 3, a + 2 };
                }
                job.ni += 6 * (m - 1);
            }
        }
    };

    /// Ribbon generation is the heaviest CPU stage, so balls are split over a few threads;
    /// each writes its own slice of the buffers and gets its own draw call.
    fn drawTrails(self: *Renderer, w: *const world.World, cam: Camera) !void {
        const sl = w.balls.slice();
        const n_balls = sl.len;
        const t = &self.tris;
        try t.reserve(self.gpa, n_balls * trail_vpb, n_balls * trail_ipb);
        const stride: usize = if (cam.s < 0.6) 2 else 1;
        const n_jobs: usize = if (n_balls >= 600) trail_threads else 1;
        const per = (n_balls + n_jobs - 1) / n_jobs;
        var jobs: [trail_threads]TrailJob = undefined;
        var threads: [trail_threads]?std.Thread = @splat(null);
        for (0..n_jobs) |j| {
            const lo = @min(j * per, n_balls);
            const hi = @min(lo + per, n_balls);
            jobs[j] = .{
                .sl = sl,
                .lo = lo,
                .hi = hi,
                .cam = cam,
                .stride = stride,
                .rng = self.nextSeed(),
                .verts = t.v[lo * trail_vpb .. hi * trail_vpb],
                .idx = t.ix[lo * trail_ipb .. hi * trail_ipb],
            };
        }
        for (1..n_jobs) |j| threads[j] = std.Thread.spawn(.{}, TrailJob.run, .{&jobs[j]}) catch null;
        jobs[0].run();
        for (1..n_jobs) |j| {
            if (threads[j]) |th| th.join() else jobs[j].run();
        }
        for (jobs[0..n_jobs]) |*job| {
            if (job.ni > 0) _ = c.SDL_RenderGeometry(self.sdl, null, job.verts.ptr, @intCast(job.nv), job.idx.ptr, @intCast(job.ni));
        }
    }

    fn drawBalls(self: *Renderer, w: *const world.World, alpha: f32, cam: Camera) !void {
        const sl = w.balls.slice();
        const xs = sl.items(.x);
        const ys = sl.items(.y);
        const pxs = sl.items(.prev_x);
        const pys = sl.items(.prev_y);
        const angs = sl.items(.angle);
        const pangs = sl.items(.prev_angle);
        const rads = sl.items(.radius);
        const temps = sl.items(.temp);
        const burns = sl.items(.burn);
        const levels = sl.items(.level);
        const flags = sl.items(.flags);
        const q = &self.quads;
        try q.reserve(self.gpa, sl.len * 4, sl.len * 6);
        const rad_per_deg = std.math.pi / 180.0;
        for (0..sl.len) |i| {
            const f = flags[i];
            if (!f.alive) continue;
            const heat = std.math.clamp((temps[i] - cfg.room_temp) / (cfg.ignition_temp - cfg.room_temp), 0, 1);
            const char = 1.0 - std.math.clamp(burns[i] / cfg.max_burn, 0, 1);
            const gb = (1.0 - heat * 0.5) * char;
            const col: c.SDL_FColor = .{ .r = char, .g = gb, .b = gb, .a = 1 };

            const tile = if (f.scored) basketball_tile else std.math.clamp(@as(usize, levels[i]), 1, cfg.max_level) - 1;
            const uv = self.tile_uv[tile];
            const cx = cam.sx(lerp(pxs[i], xs[i], alpha));
            const cy = cam.sy(lerp(pys[i], ys[i], alpha));
            const a = lerp(pangs[i], angs[i], alpha) * rad_per_deg;
            const r = rads[i] * cam.s;
            const cs = @cos(a) * r;
            const sn = @sin(a) * r;
            // Corners (-,-) (+,-) (+,+) (-,+) rotated counter-clockwise on screen, like pygame.transform.rotate.
            q.vert(cx - cs - sn, cy + sn - cs, col, uv[0], uv[1]);
            q.vert(cx + cs - sn, cy - sn - cs, col, uv[2], uv[1]);
            q.vert(cx + cs + sn, cy - sn + cs, col, uv[2], uv[3]);
            q.vert(cx - cs + sn, cy + sn + cs, col, uv[0], uv[3]);
            q.quadIndices();
        }
        q.flush(self.sdl, self.atlas);
    }

    fn drawParticles(self: *Renderer, w: *const world.World, alpha: f32, cam: Camera) !void {
        const t = &self.tris;
        const ps = w.particles.slice();
        const ash = w.ashes.slice();
        try t.reserve(self.gpa, (ps.len + ash.len) * 4, (ps.len + ash.len) * 6);
        const px = ps.items(.x);
        const py = ps.items(.y);
        const psz = ps.items(.size);
        const pr = ps.items(.r);
        const pg = ps.items(.g);
        const pb = ps.items(.b);
        for (0..ps.len) |i| {
            const sz = @max(psz[i] * cam.s, 1);
            t.rect(cam.sx(px[i]), cam.sy(py[i]), sz, sz, rgb8(pr[i], pg[i], pb[i]));
        }
        const ax = ash.items(.x);
        const ay = ash.items(.y);
        const apx = ash.items(.prev_x);
        const apy = ash.items(.prev_y);
        const age = ash.items(.age);
        const red = ash.items(.start_red);
        const d = @max(4 * cam.s, 1.5);
        for (0..ash.len) |i| {
            const fade = 1.0 - @min(@as(f32, @floatFromInt(age[i])) / ash_life_ticks, 1.0);
            const rr: u8 = @intFromFloat(@as(f32, @floatFromInt(red[i])) * fade);
            t.rect(cam.sx(lerp(apx[i], ax[i], alpha)) - d / 2, cam.sy(lerp(apy[i], ay[i], alpha)) - d / 2, d, d, rgb8(rr, 0, 0));
        }
        t.flush(self.sdl, null);
    }

    fn drawExplosions(self: *Renderer, w: *const world.World, cam: Camera) !void {
        const q = &self.quads;
        try q.reserve(self.gpa, w.explosions.items.len * 4, w.explosions.items.len * 6);
        const white = fcolor(.{ .r = 1, .g = 1, .b = 1 });
        for (w.explosions.items) |e| {
            const frame = e.age / expl_frame_ticks;
            if (frame >= expl_frames) continue;
            const f: f32 = @floatFromInt(frame);
            const sz = e.size * cam.s;
            q.texRect(cam.sx(e.x) - sz / 2, cam.sy(e.y - 0.3 * e.size) - sz / 2, sz, sz, .{ f / expl_frames, 0, (f + 1) / expl_frames, 1 }, white);
        }
        q.flush(self.sdl, self.expl);
    }

    fn hudScale(self: *const Renderer) f32 {
        return @max(@min(self.out_w / hud_w, self.out_h / hud_h), 0.5);
    }

    /// Panels, top-right ball picker, energy bar and lasso (no text).
    fn drawHud(self: *Renderer, w: *const world.World, hud: *const Hud) !void {
        const hs = self.hudScale();
        const t = &self.tris;
        const q = &self.quads;
        const panel_x = (self.out_w / hs - 149) * hs;
        const panel_y = 15 * hs;
        try t.reserve(self.gpa, 6 * Batch.round_rect_verts + 8, 6 * Batch.round_rect_idx + 12);
        t.roundRect(panel_x, panel_y, 135 * hs, 105 * hs, 12 * hs, rgb8(100, 100, 100));
        t.roundRect(panel_x + 2 * hs, panel_y + 2 * hs, 131 * hs, 101 * hs, 10 * hs, rgb8(20, 20, 20));

        const bar_y = 130 * hs;
        t.roundRect(panel_x, bar_y, 135 * hs, 20 * hs, 6 * hs, rgb8(100, 100, 100));
        t.roundRect(panel_x + 2 * hs, bar_y + 2 * hs, 131 * hs, 16 * hs, 4 * hs, rgb8(20, 20, 20));
        const frac = std.math.clamp(hud.attraction_energy / cfg.attraction_energy_max, 0, 1);
        const fill = 131 * hs * frac;
        if (fill >= 1) t.roundRect(panel_x + 2 * hs, bar_y + 2 * hs, fill, 16 * hs, 4 * hs, rgb8(40, 120, 255));

        if (w.balls.len > 0) {
            const sl = w.balls.slice();
            const levels = sl.items(.level);
            const flags = sl.items(.flags);
            var best: ?usize = null;
            for (0..sl.len) |i| {
                if (!flags[i].alive) continue;
                if (best == null or levels[i] > levels[best.?]) best = i;
            }
            if (best) |bi| {
                const tile = if (flags[bi].scored) basketball_tile else std.math.clamp(@as(usize, levels[bi]), 1, cfg.max_level) - 1;
                try q.reserve(self.gpa, 4, 6);
                q.texRect(panel_x + 10 * hs, panel_y + 12 * hs, 80 * hs, 80 * hs, self.tile_uv[tile], fcolor(.{ .r = 1, .g = 1, .b = 1 }));
            }
        }

        if (hud.lasso.len >= 2) {
            const n = hud.lasso.len - 1;
            try t.reserve(self.gpa, n * 4, n * 6);
            const col: Color = .{ .r = 1, .g = 0, .b = 0, .a = std.math.clamp(hud.lasso_alpha, 0, 1) };
            const hw = 2.5 * hs;
            for (0..n) |i| {
                const a = hud.lasso[i];
                const b = hud.lasso[i + 1];
                const x0 = self.cam.sx(a.x);
                const y0 = self.cam.sy(a.y);
                const x1 = self.cam.sx(b.x);
                const y1 = self.cam.sy(b.y);
                const dx = x1 - x0;
                const dy = y1 - y0;
                const l = @sqrt(dx * dx + dy * dy);
                if (l < 1e-4) continue;
                const nx = -dy / l * hw;
                const ny = dx / l * hw;
                const fc = fcolor(col);
                t.vert(x0 + nx, y0 + ny, fc, 0, 0);
                t.vert(x1 + nx, y1 + ny, fc, 0, 0);
                t.vert(x1 - nx, y1 - ny, fc, 0, 0);
                t.vert(x0 - nx, y0 - ny, fc, 0, 0);
                t.quadIndices();
            }
        }
        t.flush(self.sdl, null);
        q.flush(self.sdl, self.atlas);
    }

    fn text(self: *Renderer, x: f32, y: f32, scale: f32, col: Color, comptime fmt: []const u8, args: anytype) void {
        var buf: [160]u8 = undefined;
        const s = std.mem.printSentinel(&buf, fmt, args, 0) catch return;
        const hs = self.hudScale() * scale;
        _ = c.SDL_SetRenderScale(self.sdl, hs, hs);
        _ = c.SDL_SetRenderDrawColor(self.sdl, @intFromFloat(col.r * 255), @intFromFloat(col.g * 255), @intFromFloat(col.b * 255), 255);
        _ = c.SDL_RenderDebugText(self.sdl, x / hs, y / hs, s.ptr);
    }

    fn textWidth(self: *const Renderer, scale: f32, len: usize) f32 {
        return @as(f32, @floatFromInt(len)) * c.SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE * scale * self.hudScale();
    }

    fn drawText(self: *Renderer, w: *const world.World, hud: *const Hud) void {
        const hs = self.hudScale();
        const white = rgb8(255, 255, 255);
        const lh = 17 * hs;
        const x = (cfg.wall + 5) * hs;
        var y = (cfg.wall + 5) * hs;
        const help = [_][:0]const u8{
            "Press [LEFT CLICK] to spawn ball.",
            "Press [RIGHT CLICK] to ignite a ball.",
            "Hold [SHIFT] while dragging mouse to lasso. Lasso around ball to make boom boom.",
            "Press [M] to mute background music.",
            "Hold [G + LEFT CLICK] and drag to attract balls to your mouse.",
            "Press [F] to toggle gravity.",
            "[N] toggle merging",
            "[P] perf overlay",
            "[T] stress test: +1000 balls, merging off",
        };
        for (help) |line| {
            self.text(x, y, 1.5, white, "{s}", .{line});
            y += lh;
        }
        if (hud.muted) {
            self.text(x, y, 1.5, rgb8(255, 200, 80), "(music muted)", .{});
            y += lh;
        }

        if (hud.show_perf) {
            y += lh * 0.5;
            const p = hud.perf;
            const grey = rgb8(190, 255, 190);
            self.text(x, y, 1.5, grey, "fps {d:.0}  tick {d:.2} ms  render {d:.2} ms", .{ p.fps, p.tick_ms, p.render_ms });
            y += lh;
            self.text(x, y, 1.5, grey, "balls {d}  particles {d}  ashes {d}", .{ w.balls.len, w.particles.len, w.ashes.len });
            y += lh;
            self.text(x, y, 1.5, grey, "gravity: {s}  merging: {s}", .{ if (hud.mutual_gravity) "mutual" else "constant", if (hud.merging) "on" else "off" });
        }

        const panel_x = self.out_w - 149 * hs;
        if (w.balls.len > 0) {
            const sl = w.balls.slice();
            const levels = sl.items(.level);
            const flags = sl.items(.flags);
            var best: u8 = 0;
            for (0..sl.len) |i| if (flags[i].alive) {
                best = @max(best, levels[i]);
            };
            if (best > 0) self.text(panel_x + 96 * hs, 15 * hs + 44 * hs, 2, white, "{d}", .{best});
        }

        var pb: [32]u8 = undefined;
        const bpc = std.fmt.bufPrint(&pb, "balls per click: {d}", .{hud.balls_per_click}) catch "";
        const bpc_y = self.out_h - 50 * hs;
        self.text(x, bpc_y, 4, rgb8(0, 255, 0), "{s}", .{bpc});
        self.text(x + self.textWidth(4, bpc.len) + 20 * hs, bpc_y + 16 * hs, 2, rgb8(0, 255, 0), "(use + or -)", .{});

        if (hud.won) {
            const s = "You win!";
            self.text((self.out_w - self.textWidth(6, s.len)) / 2, 100 * hs - 24 * hs, 6, white, s, .{});
        } else if (hud.countdown <= 0) {
            const s = "You Lose!!";
            self.text((self.out_w - self.textWidth(6, s.len)) / 2, 100 * hs - 24 * hs, 6, rgb8(255, 0, 0), s, .{});
        } else {
            var cb: [8]u8 = undefined;
            const s = std.fmt.bufPrint(&cb, "{d}", .{hud.countdown}) catch "";
            self.text((self.out_w - self.textWidth(6, s.len)) / 2, 100 * hs - 24 * hs, 6, rgb8(255, 0, 0), "{s}", .{s});
        }
        _ = c.SDL_SetRenderScale(self.sdl, 1, 1);
    }
};
