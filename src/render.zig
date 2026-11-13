//! Batched GPU renderer on SDL_GPU: balls and text are instanced sprites filled in place in
//! mapped GPU memory, everything else is a plain triangle list, so cost scales with fill and
//! vertex count rather than draw calls.
const std = @import("std");
const c = @import("c");
const world = @import("world.zig");
const cfg = @import("config.zig");
const gfx = @import("gfx.zig");
const font = @import("font.zig");

pub const Perf = struct {
    fps: f32 = 0,
    tick_ms: f32 = 0,
    render_ms: f32 = 0,
    /// GPU stage times from gpu_world's blocking samples (0 = no GPU world).
    gpu_gravity_ms: f32 = 0,
    gpu_collide_ms: f32 = 0,
    gpu_game_ms: f32 = 0,
};

/// A particle ring pool on the GPU (gpu_world.Pool), drawn as flat squares by the ball shader.
pub const GpuPool = struct {
    pos: *c.SDL_GPUBuffer,
    prev_pos: *c.SDL_GPUBuffer,
    body: *c.SDL_GPUBuffer,
    state: *c.SDL_GPUBuffer,
    heat: *c.SDL_GPUBuffer,
    meta: *c.SDL_GPUBuffer,
    /// Ring slots to draw (dead entries are skipped on the GPU).
    count: u32,
};

/// Balls living on the GPU (gpu_world.zig); drawn with gfx.ballPass instead of `World.balls`.
pub const GpuBalls = struct {
    pos: *c.SDL_GPUBuffer,
    prev_pos: *c.SDL_GPUBuffer,
    body: *c.SDL_GPUBuffer,
    state: *c.SDL_GPUBuffer,
    heat: *c.SDL_GPUBuffer,
    meta: *c.SDL_GPUBuffer,
    /// Ball ids to draw (dead ids are skipped on the GPU).
    count: u32,
    alive: u32,
    queued: usize = 0,
    max_level: u8,
    fire: ?GpuPool = null,
    ash: ?GpuPool = null,
    /// One-entry tile table for the pools (uv 0..1 over the white texture).
    pool_tiles: ?*c.SDL_GPUBuffer = null,
    /// Counters from the GPU fire system (about two ticks old).
    burning: u32 = 0,
    particles: u32 = 0,
    ashes: u32 = 0,
};

/// Galaxy-mode readout; null in the normal game.
pub const GalaxyHud = struct {
    scenario: []const u8,
    bodies: u32,
    collide: []const u8,
    /// Gravity relative to the scenario's own (1 = as built).
    g_mult: f32,
};

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
    gpu: ?GpuBalls = null,
    galaxy: ?GalaxyHud = null,
};

const Vertex = gfx.Vertex;
const Sprite = gfx.Sprite;
const Rgba = [4]f32;
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

fn fcolor(col: Color) Rgba {
    return .{ col.r, col.g, col.b, col.a };
}

inline fn vtx(x: f32, y: f32, col: Rgba, u: f32, v: f32) Vertex {
    return .{ .pos = .{ .x = x, .y = y }, .uv = .{ .x = u, .y = v }, .color = col };
}

/// Growable triangle-list vertex array that keeps its capacity between frames.
const Batch = struct {
    v: []Vertex = &.{},
    nv: usize = 0,

    fn deinit(b: *Batch, gpa: std.mem.Allocator) void {
        gpa.free(b.v);
    }

    fn reserve(b: *Batch, gpa: std.mem.Allocator, add: usize) !void {
        if (b.nv + add > b.v.len) b.v = try gpa.realloc(b.v, @max(b.nv + add, b.v.len * 2));
    }

    /// Triangles (0,1,2) and (0,2,3).
    inline fn quad(b: *Batch, p0: Vertex, p1: Vertex, p2: Vertex, p3: Vertex) void {
        b.v[b.nv..][0..6].* = .{ p0, p1, p2, p0, p2, p3 };
        b.nv += 6;
    }

    fn rect(b: *Batch, x: f32, y: f32, w: f32, h: f32, col: Color) void {
        const fc = fcolor(col);
        b.quad(vtx(x, y, fc, 0, 0), vtx(x + w, y, fc, 0, 0), vtx(x + w, y + h, fc, 0, 0), vtx(x, y + h, fc, 0, 0));
    }

    /// Axis-aligned textured quad; uv is {u0, v0, u1, v1}.
    fn texRect(b: *Batch, x: f32, y: f32, w: f32, h: f32, uv: [4]f32, col: Rgba) void {
        b.quad(vtx(x, y, col, uv[0], uv[1]), vtx(x + w, y, col, uv[2], uv[1]), vtx(x + w, y + h, col, uv[2], uv[3]), vtx(x, y + h, col, uv[0], uv[3]));
    }

    const corner_steps = 6;
    const ring_pts = 4 * (corner_steps + 1);
    const round_rect_verts = 3 * ring_pts;

    fn roundRect(b: *Batch, x: f32, y: f32, w: f32, h: f32, radius: f32, col: Color) void {
        const r = @min(radius, w / 2, h / 2);
        const fc = fcolor(col);
        const centre = vtx(x + w / 2, y + h / 2, fc, 0, 0);
        const corners = [4][3]f32{
            .{ x + w - r, y + r, -std.math.pi / 2.0 },
            .{ x + w - r, y + h - r, 0 },
            .{ x + r, y + h - r, std.math.pi / 2.0 },
            .{ x + r, y + r, std.math.pi },
        };
        var ring: [ring_pts]Vertex = undefined;
        var n: usize = 0;
        for (corners) |cn| {
            for (0..corner_steps + 1) |k| {
                const a = cn[2] + (std.math.pi / 2.0) * @as(f32, @floatFromInt(k)) / corner_steps;
                ring[n] = vtx(cn[0] + r * @cos(a), cn[1] + r * @sin(a), fc, 0, 0);
                n += 1;
            }
        }
        for (0..ring_pts) |i| {
            b.v[b.nv..][0..3].* = .{ centre, ring[i], ring[(i + 1) % ring_pts] };
            b.nv += 3;
        }
    }

    fn flush(b: *Batch, g: *gfx.Gfx, tex: ?gfx.Texture) !void {
        defer b.nv = 0;
        try g.triangles(tex, b.v[0..b.nv]);
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

fn buildTileJob(gpa: std.mem.Allocator, k: usize, atlas_px: []u8, result: *anyerror!void) void {
    result.* = buildTile(gpa, k, atlas_px);
}

/// Loads ball image `k`, resizes and circle-masks it, and copies it with a clamped gutter into
/// its atlas cell.
fn buildTile(gpa: std.mem.Allocator, k: usize, atlas_px: []u8) !void {
    const tile = try gpa.alloc(u8, tile_px * tile_px * 4);
    defer gpa.free(tile);
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


fn texFromPixels(g: *gfx.Gfx, px: []const u8, w: usize, h: usize) !gfx.Texture {
    return g.uploadTexture(px, @intCast(w), @intCast(h), false);
}

pub const Renderer = struct {
    gpa: std.mem.Allocator,
    gfx: *gfx.Gfx,
    atlas: gfx.Texture,
    bg: gfx.Texture,
    bg_w: f32,
    bg_h: f32,
    expl: gfx.Texture,
    font_tex: gfx.Texture,
    font_atlas: font.Atlas,
    tile_uv: [tile_count][4]f32,
    /// tile_uv as a GPU storage buffer, the tile table of gfx.ballPass.
    tiles: *c.SDL_GPUBuffer,
    tris: Batch = .{},
    trail_buf: []Vertex = &.{},
    /// GPU-ball trails (shown before the balls); toggled by the caller.
    trails: bool = false,
    trails_was_on: bool = false,
    trail_last_tick: ?u64 = null,
    cam: Camera = .{},
    /// User view on top of the auto-fit camera: zoom 1 and zero pan is exactly the fit.
    zoom: f32 = 1,
    /// World-space offset of the view focus from the base focus (follow point or play-area centre).
    pan: world.Vec2 = .{ .x = 0, .y = 0 },
    follow_pt: ?world.Vec2 = null,
    /// Focus and fit scale actually used by the last drawn frame, for zooming about the cursor.
    view_focus: world.Vec2 = .{ .x = 0, .y = 0 },
    fit_scale: f32 = 1,
    out_w: f32 = hud_w,
    out_h: f32 = hud_h,
    rng: u64 = 0x9E3779B97F4A7C15,

    pub fn init(gpa: std.mem.Allocator, g: *gfx.Gfx) !Renderer {
        const atlas_px = try gpa.alloc(u8, atlas_w * atlas_h * 4);
        defer gpa.free(atlas_px);
        @memset(atlas_px, 0);
        // Each tile is decoded and resized on its own thread (they write disjoint atlas cells);
        // done one after another this was ~0.4 s of startup.
        var results: [tile_count]anyerror!void = undefined;
        var threads: [tile_count]?std.Thread = @splat(null);
        for (0..tile_count) |k| {
            threads[k] = std.Thread.spawn(.{}, buildTileJob, .{ gpa, k, atlas_px, &results[k] }) catch null;
            if (threads[k] == null) buildTileJob(gpa, k, atlas_px, &results[k]);
        }
        for (threads) |t| if (t) |th| th.join();
        for (results) |r| try r;

        var uv: [tile_count][4]f32 = undefined;
        for (0..tile_count) |k| {
            const cx0 = (k % atlas_cols) * cell_px;
            const cy0 = (k / atlas_cols) * cell_px;
            const aw: f32 = atlas_w;
            const ah: f32 = atlas_h;
            uv[k] = .{
                @as(f32, @floatFromInt(cx0 + gutter)) / aw,
                @as(f32, @floatFromInt(cy0 + gutter)) / ah,
                @as(f32, @floatFromInt(cx0 + gutter + tile_px)) / aw,
                @as(f32, @floatFromInt(cy0 + gutter + tile_px)) / ah,
            };
        }
        const atlas = try texFromPixels(g, atlas_px, atlas_w, atlas_h);
        errdefer g.releaseTexture(atlas);

        const bg_img = try Image.load("assets/background/background.png");
        defer bg_img.free();
        const bg = try texFromPixels(g, bg_img.px, bg_img.w, bg_img.h);
        errdefer g.releaseTexture(bg);

        const ex_img = try Image.load("assets/explosion/explosion.png");
        defer ex_img.free();
        const expl = try texFromPixels(g, ex_img.px, ex_img.w, ex_img.h);
        errdefer g.releaseTexture(expl);

        var fa = try font.Atlas.init(gpa);
        errdefer fa.deinit(gpa);
        const font_tex = try texFromPixels(g, fa.pixels, fa.w, fa.h);

        return .{
            .gpa = gpa,
            .gfx = g,
            .atlas = atlas,
            .bg = bg,
            .bg_w = @floatFromInt(bg_img.w),
            .bg_h = @floatFromInt(bg_img.h),
            .expl = expl,
            .font_tex = font_tex,
            .font_atlas = fa,
            .tile_uv = uv,
            .tiles = try tileBuffer(g, &uv),
        };
    }

    pub fn deinit(self: *Renderer) void {
        _ = c.SDL_WaitForGPUIdle(self.gfx.dev);
        self.gfx.releaseTexture(self.atlas);
        self.gfx.releaseTexture(self.bg);
        self.gfx.releaseTexture(self.expl);
        self.gfx.releaseTexture(self.font_tex);
        c.SDL_ReleaseGPUBuffer(self.gfx.dev, self.tiles);
        self.font_atlas.deinit(self.gpa);
        self.tris.deinit(self.gpa);
        self.gpa.free(self.trail_buf);
    }

    /// Takes window pixel coordinates (not window points on high-DPI displays).
    pub fn screenToWorld(self: *const Renderer, w: *const world.World, sx: f32, sy: f32) world.Vec2 {
        _ = w;
        const cam = self.cam;
        return .{ .x = (sx - cam.ox) / cam.s + cam.bx, .y = (sy - cam.oy) / cam.s + cam.by };
    }

    pub const zoom_min: f32 = 0.05;
    pub const zoom_max: f32 = 20;

    /// Makes `p` the base of the view focus (call every frame to track it). Pan is kept as an offset.
    pub fn follow(self: *Renderer, p: world.Vec2) void {
        self.follow_pt = p;
    }

    pub fn unfollow(self: *Renderer) void {
        self.follow_pt = null;
    }

    /// Back to the auto-fit view.
    pub fn resetView(self: *Renderer) void {
        self.zoom = 1;
        self.pan = .{ .x = 0, .y = 0 };
        self.follow_pt = null;
    }

    /// Multiplies the zoom by `factor`, keeping the world point under window pixel (sx, sy) fixed.
    pub fn zoomAt(self: *Renderer, sx: f32, sy: f32, factor: f32) void {
        const nz = std.math.clamp(self.zoom * factor, zoom_min, zoom_max);
        if (nz == self.zoom) return;
        const old_s = self.fit_scale * self.zoom;
        const new_s = self.fit_scale * nz;
        const dx = sx - self.out_w / 2;
        const dy = sy - self.out_h / 2;
        // p = f + d/old_s must equal f' + d/new_s.
        self.pan.x += dx / old_s - dx / new_s;
        self.pan.y += dy / old_s - dy / new_s;
        self.zoom = nz;
    }

    /// Drags the view by a window-pixel delta (the world follows the cursor).
    pub fn panByPixels(self: *Renderer, dx: f32, dy: f32) void {
        const s = self.fit_scale * self.zoom;
        self.pan.x -= dx / s;
        self.pan.y -= dy / s;
    }

    /// Drops the GPU trail history, e.g. when a new scene loads.
    pub fn clearTrails(self: *Renderer) void {
        self.gfx.clearTrails();
        self.trail_last_tick = null;
    }

    fn nextSeed(self: *Renderer) u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        return self.rng | 1;
    }

    /// Records the frame; `present` submits it.
    pub fn draw(self: *Renderer, w: *const world.World, alpha: f32, hud: *const Hud) void {
        self.drawInner(w, alpha, hud) catch |e| std.log.err("draw failed: {s}", .{@errorName(e)});
    }

    pub fn present(self: *Renderer, screenshot: ?[*:0]const u8) void {
        self.gfx.end(.{ .screenshot = screenshot }) catch |e| std.log.err("present failed: {s}", .{@errorName(e)});
    }

    fn drawInner(self: *Renderer, w: *const world.World, alpha: f32, hud: *const Hud) !void {
        self.gfx.begin(.{ 12.0 / 255.0, 12.0 / 255.0, 16.0 / 255.0, 1 });
        self.out_w = @floatFromInt(self.gfx.width);
        self.out_h = @floatFromInt(self.gfx.height);

        const pb = w.prev_bounds;
        const nb = w.bounds;
        const bounds: world.Rect = .{
            .x = lerp(pb.x, nb.x, alpha),
            .y = lerp(pb.y, nb.y, alpha),
            .w = lerp(pb.w, nb.w, alpha),
            .h = lerp(pb.h, nb.h, alpha),
        };
        const fit = Camera.fit(bounds, self.out_w, self.out_h);
        const base: world.Vec2 = self.follow_pt orelse .{ .x = bounds.x + bounds.w / 2, .y = bounds.y + bounds.h / 2 };
        const focus: world.Vec2 = .{ .x = base.x + self.pan.x, .y = base.y + self.pan.y };
        self.fit_scale = fit.s;
        self.view_focus = focus;
        const identity = self.zoom == 1 and self.pan.x == 0 and self.pan.y == 0 and self.follow_pt == null;
        const cam: Camera = if (identity) fit else .{ .s = fit.s * self.zoom, .ox = self.out_w / 2, .oy = self.out_h / 2, .bx = focus.x, .by = focus.y };
        self.cam = cam;

        try self.drawBackground(bounds, cam);
        if (hud.gpu) |gb| {
            if (self.trails) {
                if (!self.trails_was_on) self.clearTrails();
                const stride = gfx.trail_stride;
                const record = self.trail_last_tick == null or self.trail_last_tick.? / stride != w.tick / stride;
                self.trail_last_tick = w.tick;
                try self.gfx.trailPass(.{
                    .pos = gb.pos,
                    .prev_pos = gb.prev_pos,
                    .body = gb.body,
                    .meta = gb.meta,
                    .count = gb.count,
                    .alpha = alpha,
                    .cam = .{ .scale = cam.s, .ox = cam.ox, .oy = cam.oy, .bx = cam.bx, .by = cam.by },
                    .record = record,
                });
            }
            self.trails_was_on = self.trails;
            try self.gfx.ballPass(.{
                .bufs = .{ .pos = gb.pos, .prev_pos = gb.prev_pos, .body = gb.body, .state = gb.state, .heat = gb.heat, .meta = gb.meta, .tiles = self.tiles },
                .atlas = self.atlas,
                .count = gb.count,
                .alpha = alpha,
                .cam = .{ .scale = cam.s, .ox = cam.ox, .oy = cam.oy, .bx = cam.bx, .by = cam.by },
            });
        } else {
            try self.drawTrails(w, cam);
            try self.drawBalls(w, alpha, cam);
        }
        if (hud.gpu) |gb| {
            if (gb.pool_tiles) |tiles| for ([_]?GpuPool{ gb.fire, gb.ash }) |maybe| if (maybe) |pool| {
                try self.gfx.ballPass(.{
                    .bufs = .{ .pos = pool.pos, .prev_pos = pool.prev_pos, .body = pool.body, .state = pool.state, .heat = pool.heat, .meta = pool.meta, .tiles = tiles },
                    .atlas = self.gfx.white,
                    .count = pool.count,
                    .alpha = alpha,
                    .cam = .{ .scale = cam.s, .ox = cam.ox, .oy = cam.oy, .bx = cam.bx, .by = cam.by },
                });
            };
        } else try self.drawParticles(w, alpha, cam);
        try self.drawHud(w, hud);
        try self.drawExplosions(w, cam);
        self.drawText(w, hud);
    }

    fn drawBackground(self: *Renderer, b: world.Rect, cam: Camera) !void {
        // The original window showed the image's top-left base_w x base_h crop; that crop is
        // stretched over the whole play area so the hoop stays where physics expects it.
        const t = &self.tris;
        try t.reserve(self.gpa, 6);
        t.texRect(cam.sx(b.x), cam.sy(b.y), b.w * cam.s, b.h * cam.s, .{ 0, 0, cfg.base_w / self.bg_w, cfg.base_h / self.bg_h }, .{ 1, 1, 1, 1 });
        try t.flush(self.gfx, self.bg);
        try t.reserve(self.gpa, 24);
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
        try t.flush(self.gfx, null);
    }

    const trail_threads = 4;
    const trail_vpb = (cfg.max_trail - 1) * 6;

    const TrailJob = struct {
        sl: std.MultiArrayList(world.Ball).Slice,
        lo: usize,
        hi: usize,
        cam: Camera,
        stride: usize,
        rng: u64,
        verts: []Vertex,
        nv: usize = 0,

        fn rand01(job: *@This()) f32 {
            job.rng ^= job.rng << 13;
            job.rng ^= job.rng >> 7;
            job.rng ^= job.rng << 17;
            return @as(f32, @floatFromInt(@as(u32, @truncate(job.rng >> 32)) >> 8)) / 16777216.0;
        }

        /// Writes ribbons for balls lo..hi into this job's own vertex slice.
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

                var pts: [2 * cfg.max_trail]Vertex = undefined;
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
                    pts[2 * q] = vtx(sxs[q] + nx * hw, sys[q] + ny * hw, col, 0, 0);
                    pts[2 * q + 1] = vtx(sxs[q] - nx * hw, sys[q] - ny * hw, col, 0, 0);
                }
                const out = job.verts[job.nv..][0 .. 6 * (m - 1)];
                for (0..m - 1) |sg| {
                    const a = 2 * sg;
                    out[6 * sg ..][0..6].* = .{ pts[a], pts[a + 1], pts[a + 2], pts[a + 1], pts[a + 3], pts[a + 2] };
                }
                job.nv += 6 * (m - 1);
            }
        }
    };

    /// Ribbon generation is the heaviest CPU stage, so balls are split over a few threads;
    /// each writes its own slice of the scratch buffer.
    fn drawTrails(self: *Renderer, w: *const world.World, cam: Camera) !void {
        const sl = w.balls.slice();
        const n_balls = sl.len;
        if (self.trail_buf.len < n_balls * trail_vpb) {
            self.gpa.free(self.trail_buf);
            self.trail_buf = &.{};
            self.trail_buf = try self.gpa.alloc(Vertex, n_balls * trail_vpb);
        }
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
                .verts = self.trail_buf[lo * trail_vpb .. hi * trail_vpb],
            };
        }
        for (1..n_jobs) |j| threads[j] = std.Thread.spawn(.{}, TrailJob.run, .{&jobs[j]}) catch null;
        jobs[0].run();
        for (1..n_jobs) |j| {
            if (threads[j]) |th| th.join() else jobs[j].run();
        }
        for (jobs[0..n_jobs]) |*job| try self.gfx.triangles(null, job.verts[0..job.nv]);
    }

    const BallJob = struct {
        sl: std.MultiArrayList(world.Ball).Slice,
        out: []Sprite,
        lo: usize,
        hi: usize,
        alpha: f32,
        cam: Camera,
        tile_uv: *const [tile_count][4]f32,

        /// Dead balls become zero-size sprites so slots stay aligned with ball indices.
        fn run(job: *@This()) void {
            const sl = job.sl;
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
            const cam = job.cam;
            const alpha = job.alpha;
            const rad_per_deg = std.math.pi / 180.0;
            for (job.lo..job.hi) |i| {
                const f = flags[i];
                const o = &job.out[i];
                if (!f.alive) {
                    o.* = std.mem.zeroes(Sprite);
                    continue;
                }
                const heat = std.math.clamp((temps[i] - cfg.room_temp) / (cfg.ignition_temp - cfg.room_temp), 0, 1);
                const char = 1.0 - std.math.clamp(burns[i] / cfg.max_burn, 0, 1);
                const gb = (1.0 - heat * 0.5) * char;
                const tile = if (f.scored) basketball_tile else std.math.clamp(@as(usize, levels[i]), 1, cfg.max_level) - 1;
                const uv = job.tile_uv[tile];
                const r = rads[i] * cam.s;
                o.* = .{
                    .center = .{ .x = cam.sx(lerp(pxs[i], xs[i], alpha)), .y = cam.sy(lerp(pys[i], ys[i], alpha)) },
                    .half = .{ .x = r, .y = r },
                    .uv_min = .{ .x = uv[0], .y = uv[1] },
                    .uv_max = .{ .x = uv[2], .y = uv[3] },
                    .color = .{ char, gb, gb, 1 },
                    .angle = lerp(pangs[i], angs[i], alpha) * rad_per_deg,
                };
            }
        }
    };

    fn drawBalls(self: *Renderer, w: *const world.World, alpha: f32, cam: Camera) !void {
        const sl = w.balls.slice();
        if (sl.len == 0) return;
        const out = try self.gfx.allocSprites(self.atlas, @intCast(sl.len));
        const n_jobs: usize = if (sl.len >= 4096) 4 else 1;
        const per = (sl.len + n_jobs - 1) / n_jobs;
        var jobs: [4]BallJob = undefined;
        var threads: [4]?std.Thread = @splat(null);
        for (0..n_jobs) |j| {
            const lo = @min(j * per, sl.len);
            jobs[j] = .{ .sl = sl, .out = out, .lo = lo, .hi = @min(lo + per, sl.len), .alpha = alpha, .cam = cam, .tile_uv = &self.tile_uv };
        }
        for (1..n_jobs) |j| threads[j] = std.Thread.spawn(.{}, BallJob.run, .{&jobs[j]}) catch null;
        jobs[0].run();
        for (1..n_jobs) |j| {
            if (threads[j]) |th| th.join() else jobs[j].run();
        }
    }

    fn drawParticles(self: *Renderer, w: *const world.World, alpha: f32, cam: Camera) !void {
        const t = &self.tris;
        const ps = w.particles.slice();
        const ash = w.ashes.slice();
        try t.reserve(self.gpa, (ps.len + ash.len) * 6);
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
        try t.flush(self.gfx, null);
    }

    fn drawExplosions(self: *Renderer, w: *const world.World, cam: Camera) !void {
        const t = &self.tris;
        try t.reserve(self.gpa, w.explosions.items.len * 6);
        for (w.explosions.items) |e| {
            const frame = e.age / expl_frame_ticks;
            if (frame >= expl_frames) continue;
            const f: f32 = @floatFromInt(frame);
            const sz = e.size * cam.s;
            t.texRect(cam.sx(e.x) - sz / 2, cam.sy(e.y - 0.3 * e.size) - sz / 2, sz, sz, .{ f / expl_frames, 0, (f + 1) / expl_frames, 1 }, .{ 1, 1, 1, 1 });
        }
        try t.flush(self.gfx, self.expl);
    }

    fn hudScale(self: *const Renderer) f32 {
        return @max(@min(self.out_w / hud_w, self.out_h / hud_h), 0.5);
    }

    /// Panels, top-right ball picker, energy bar and lasso (no text).
    fn drawHud(self: *Renderer, w: *const world.World, hud: *const Hud) !void {
        const hs = self.hudScale();
        const t = &self.tris;
        const panel_x = (self.out_w / hs - 149) * hs;
        const panel_y = 15 * hs;
        try t.reserve(self.gpa, 6 * Batch.round_rect_verts);
        t.roundRect(panel_x, panel_y, 135 * hs, 105 * hs, 12 * hs, rgb8(100, 100, 100));
        t.roundRect(panel_x + 2 * hs, panel_y + 2 * hs, 131 * hs, 101 * hs, 10 * hs, rgb8(20, 20, 20));

        const bar_y = 130 * hs;
        t.roundRect(panel_x, bar_y, 135 * hs, 20 * hs, 6 * hs, rgb8(100, 100, 100));
        t.roundRect(panel_x + 2 * hs, bar_y + 2 * hs, 131 * hs, 16 * hs, 4 * hs, rgb8(20, 20, 20));
        const frac = std.math.clamp(hud.attraction_energy / cfg.attraction_energy_max, 0, 1);
        const fill = 131 * hs * frac;
        if (fill >= 1) t.roundRect(panel_x + 2 * hs, bar_y + 2 * hs, fill, 16 * hs, 4 * hs, rgb8(40, 120, 255));

        if (hud.lasso.len >= 2) {
            const n = hud.lasso.len - 1;
            try t.reserve(self.gpa, n * 6);
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
                t.quad(vtx(x0 + nx, y0 + ny, fc, 0, 0), vtx(x1 + nx, y1 + ny, fc, 0, 0), vtx(x1 - nx, y1 - ny, fc, 0, 0), vtx(x0 - nx, y0 - ny, fc, 0, 0));
            }
        }
        try t.flush(self.gfx, null);

        if (bestBall(w, hud)) |bb| {
            const tile = if (bb.scored) basketball_tile else std.math.clamp(@as(usize, bb.level), 1, cfg.max_level) - 1;
            try t.reserve(self.gpa, 6);
            t.texRect(panel_x + 10 * hs, panel_y + 12 * hs, 80 * hs, 80 * hs, self.tile_uv[tile], .{ 1, 1, 1, 1 });
            try t.flush(self.gfx, self.atlas);
        }
    }

    fn text(self: *Renderer, x: f32, y: f32, scale: f32, col: Color, comptime fmt: []const u8, args: anytype) void {
        var buf: [160]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
        var out: [buf.len]Sprite = undefined;
        const n = font.layout(&self.font_atlas, s, x, y, self.hudScale() * scale, fcolor(col), &out);
        self.gfx.sprites(self.font_tex, out[0..n]) catch {};
    }

    fn textWidth(self: *const Renderer, scale: f32, len: usize) f32 {
        return font.textWidth(len, scale * self.hudScale());
    }

    fn drawText(self: *Renderer, w: *const world.World, hud: *const Hud) void {
        const hs = self.hudScale();
        const white = rgb8(255, 255, 255);
        const lh = 17 * hs;
        const x = (cfg.wall + 5) * hs;
        var y = (cfg.wall + 5) * hs;
        const galaxy_help = [_][:0]const u8{
            "[LEFT CLICK] spawn a star system. [RIGHT CLICK] ignite.",
            "[1] disk  [2] collision  [3] cloud  [4] systems",
            "[X] collision mode: solid / accrete / ghost",
            "[ and ] weaken / strengthen gravity",
            "[O] back to the normal game   [G + LEFT DRAG] attract   [C] follow   [HOME] reset view",
            "[P] perf overlay   [M] mute",
        };
        if (hud.galaxy) |gh| {
            self.text(x, y, 2, rgb8(160, 200, 255), "GALAXY: {s}  bodies {d}  collide {s}  g x{d:.2}", .{ gh.scenario, gh.bodies, gh.collide, gh.g_mult });
            y += lh * 1.4;
        }
        const help = [_][:0]const u8{
            "Press [LEFT CLICK] to spawn ball.",
            "Press [O] for galaxy mode (orbiting bodies).",
            "Press [RIGHT CLICK] to ignite a ball.",
            "Hold [SHIFT] while dragging mouse to lasso. Lasso around ball to make boom boom.",
            "Press [M] to mute background music.",
            "Hold [G + LEFT CLICK] and drag to attract balls to your mouse.",
            "Press [F] to toggle gravity.",
            "[N] toggle merging",
            "[P] perf overlay",
            "[T] stress test: +100000 balls, merging off",
        };
        const lines: []const [:0]const u8 = if (hud.galaxy != null) &galaxy_help else &help;
        for (lines) |line| {
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
            if (hud.gpu) |gb| {
                self.text(x, y, 1.5, grey, "GPU balls {d} (ids {d}, queued {d})  gravity {d:.2} ms  collide {d:.2} ms  merge+io {d:.2} ms", .{ gb.alive, gb.count, gb.queued, p.gpu_gravity_ms, p.gpu_collide_ms, p.gpu_game_ms });
            } else {
                self.text(x, y, 1.5, grey, "balls {d}  particles {d}  ashes {d}", .{ w.balls.len, w.particles.len, w.ashes.len });
            }
            y += lh;
            if (hud.gpu) |gb| {
                self.text(x, y, 1.5, grey, "burning {d}  particles {d}  ashes {d}", .{ gb.burning, gb.particles, gb.ashes });
                y += lh;
            }
            self.text(x, y, 1.5, grey, "gravity: {s}  merging: {s}", .{ if (hud.mutual_gravity) "mutual" else "constant", if (hud.merging) "on" else "off" });
        }

        const panel_x = self.out_w - 149 * hs;
        if (bestBall(w, hud)) |bb| self.text(panel_x + 96 * hs, 15 * hs + 44 * hs, 2, white, "{d}", .{bb.level});

        if (hud.galaxy != null) return;
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
    }
};

fn tileBuffer(g: *gfx.Gfx, uv: *const [tile_count][4]f32) !*c.SDL_GPUBuffer {
    const size: u32 = @sizeOf(@TypeOf(uv.*));
    const buf = c.SDL_CreateGPUBuffer(g.dev, &.{ .usage = c.SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ, .size = size }) orelse return error.GpuAlloc;
    errdefer c.SDL_ReleaseGPUBuffer(g.dev, buf);
    const tb = c.SDL_CreateGPUTransferBuffer(g.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = size }) orelse return error.GpuAlloc;
    defer c.SDL_ReleaseGPUTransferBuffer(g.dev, tb);
    const m: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(g.dev, tb, false) orelse return error.GpuAlloc);
    @memcpy(m[0..size], std.mem.asBytes(uv));
    c.SDL_UnmapGPUTransferBuffer(g.dev, tb);
    const cmd = c.SDL_AcquireGPUCommandBuffer(g.dev) orelse return error.GpuAlloc;
    const cp = c.SDL_BeginGPUCopyPass(cmd);
    c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = tb }, &.{ .buffer = buf, .size = size }, false);
    c.SDL_EndGPUCopyPass(cp);
    if (!c.SDL_SubmitGPUCommandBuffer(cmd)) return error.GpuAlloc;
    return buf;
}

const BestBall = struct { level: u8, scored: bool };

/// Highest-level live ball for the HUD; GPU balls report the max level from merge events.
fn bestBall(w: *const world.World, hud: *const Hud) ?BestBall {
    if (hud.gpu) |gb| return if (gb.alive > 0) .{ .level = gb.max_level, .scored = false } else null;
    const sl = w.balls.slice();
    const levels = sl.items(.level);
    const flags = sl.items(.flags);
    var best: ?usize = null;
    for (0..sl.len) |i| {
        if (!flags[i].alive) continue;
        if (best == null or levels[i] > levels[best.?]) best = i;
    }
    const bi = best orelse return null;
    return .{ .level = levels[bi], .scored = flags[bi].scored };
}
