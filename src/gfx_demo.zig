//! SDL_GPU smoke test and timing: textured rotated sprites + triangles, then 1M sprites.
//! Saves .scratch/gfx.bmp and .scratch/gfx_1m.bmp. Headless when SDL_VIDEODRIVER=offscreen.
const std = @import("std");
const c = @import("c");
const gfx = @import("gfx.zig");
const L = @import("gpu_layout");
const cfg = @import("config.zig");

const W = 1280;
const H = 720;
const ball_textures = 8;

fn loadTexture(g: *gfx.Gfx, path: [:0]const u8) !gfx.Texture {
    var size: usize = 0;
    const data = c.SDL_LoadFile(path.ptr, &size) orelse return error.AssetMissing;
    defer c.SDL_free(data);
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const px = c.stbi_load_from_memory(@ptrCast(data), @intCast(size), &w, &h, &ch, 4) orelse return error.AssetDecode;
    defer c.stbi_image_free(px);
    const n: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 4;
    return g.uploadTexture(px[0..n], @intCast(w), @intCast(h), true);
}

fn now() u64 {
    return c.SDL_GetPerformanceCounter();
}

fn ms(t0: u64, t1: u64) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) * 1000.0 / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency()));
}

fn fillSprites(out: []gfx.Sprite, rng: std.Random, t: f32, min_r: f32, max_r: f32) void {
    for (out, 0..) |*s, i| {
        const r = min_r + rng.float(f32) * (max_r - min_r);
        const fi: f32 = @floatFromInt(i);
        s.* = .{
            .center = .{ .x = rng.float(f32) * W, .y = rng.float(f32) * H },
            .half = .{ .x = r, .y = r },
            .uv_min = .{ .x = 0, .y = 0 },
            .uv_max = .{ .x = 1, .y = 1 },
            .color = .{ 1, 1, 1, 1 },
            .angle = t + fi * 0.37,
        };
    }
}

fn star(g: *gfx.Gfx, cx: f32, cy: f32, r: f32, color: [4]f32) !void {
    var v: [10 * 3]gfx.Vertex = undefined;
    for (0..10) |k| {
        const a0 = @as(f32, @floatFromInt(k)) * std.math.pi / 5.0;
        const a1 = a0 + std.math.pi / 5.0;
        const r0: f32 = if (k % 2 == 0) r else r * 0.45;
        const r1: f32 = if (k % 2 == 0) r * 0.45 else r;
        const tip = [4]f32{ color[0], color[1], color[2], 0.9 };
        v[k * 3 + 0] = .{ .pos = .{ .x = cx, .y = cy }, .uv = .{ .x = 0, .y = 0 }, .color = .{ 1, 1, 1, 1 } };
        v[k * 3 + 1] = .{ .pos = .{ .x = cx + r0 * @cos(a0), .y = cy - r0 * @sin(a0) }, .uv = .{ .x = 0, .y = 0 }, .color = tip };
        v[k * 3 + 2] = .{ .pos = .{ .x = cx + r1 * @cos(a1), .y = cy - r1 * @sin(a1) }, .uv = .{ .x = 0, .y = 0 }, .color = tip };
    }
    try g.triangles(null, &v);
}

pub fn main() !void {
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();

    const driver: []const u8 = if (c.SDL_GetCurrentVideoDriver()) |d| std.mem.span(d) else "none";
    const headless = std.mem.eql(u8, driver, "offscreen");
    const window: ?*c.SDL_Window = if (headless) null else c.SDL_CreateWindow("gfx demo", W, H, 0);
    defer if (window) |w| c.SDL_DestroyWindow(w);

    var g = try gfx.Gfx.init(.{ .window = window, .width = W, .height = H, .vsync = false });
    defer g.deinit();
    std.debug.print("video: {s}, gpu driver: {s}, device: {s}\n", .{
        driver,
        c.SDL_GetGPUDeviceDriver(g.dev),
        c.SDL_GetStringProperty(c.SDL_GetGPUDeviceProperties(g.dev), c.SDL_PROP_GPU_DEVICE_NAME_STRING, "?"),
    });

    var tex: [ball_textures]gfx.Texture = undefined;
    for (&tex, 1..) |*t, i| {
        var buf: [64]u8 = undefined;
        t.* = try loadTexture(&g, try std.fmt.bufPrintSentinel(&buf, "assets/balls/{d}.png", .{i}, 0));
    }
    defer for (tex) |t| g.releaseTexture(t);

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Scene 1: a few thousand rotated textured sprites over a tinted untextured backdrop, plus triangles.
    const per_tex = 500;
    var frame_ms: [10]f64 = undefined;
    for (&frame_ms, 0..) |*fm, f| {
        const t0 = now();
        g.begin(.{ 0.08, 0.09, 0.12, 1 });
        try g.sprites(null, &.{.{
            .center = .{ .x = W / 2, .y = H / 2 },
            .half = .{ .x = W / 2 - 20, .y = H / 2 - 20 },
            .uv_min = .{ .x = 0, .y = 0 },
            .uv_max = .{ .x = 1, .y = 1 },
            .color = .{ 0.15, 0.2, 0.3, 1 },
            .angle = 0,
        }});
        prng = std.Random.DefaultPrng.init(42);
        for (tex) |t| fillSprites(try g.allocSprites(t, per_tex), rng, @as(f32, @floatFromInt(f)) * 0.1, 6, 30);
        try star(&g, 200, 200, 120, .{ 1, 0.8, 0.1, 1 });
        try star(&g, 1080, 520, 150, .{ 0.2, 0.9, 1, 1 });
        // Rotated, half-transparent big sprite to check rotation direction and alpha.
        try g.sprites(tex[0], &.{.{
            .center = .{ .x = W / 2, .y = H / 2 },
            .half = .{ .x = 200, .y = 100 },
            .uv_min = .{ .x = 0, .y = 0 },
            .uv_max = .{ .x = 1, .y = 1 },
            .color = .{ 1, 1, 1, 0.8 },
            .angle = std.math.pi / 8.0,
        }});
        try g.end(.{ .wait = true, .screenshot = if (f == frame_ms.len - 1) ".scratch/gfx.bmp" else null });
        fm.* = ms(t0, now());
    }
    std.debug.print("scene 1 ({d} sprites + 60 tris): frame ms (incl. GPU wait) {d:.3} first, {d:.3} median-ish (last screenshot frame {d:.3})\n", .{
        per_tex * ball_textures + 2, frame_ms[0], frame_ms[5], frame_ms[9],
    });

    // Scene 2: 1M small sprites, one texture, one draw.
    const big: u32 = 1_000_000;
    const frames = 20;
    var fill_total: f64 = 0;
    var end_total: f64 = 0;
    var best: f64 = std.math.inf(f64);
    for (0..frames + 1) |f| {
        const t0 = now();
        g.begin(.{ 0, 0, 0, 1 });
        fillSprites(try g.allocSprites(tex[2], big), rng, @as(f32, @floatFromInt(f)) * 0.1, 1.5, 4);
        const t1 = now();
        try g.end(.{ .wait = true, .screenshot = if (f == frames) ".scratch/gfx_1m.bmp" else null });
        const t2 = now();
        if (f == 0 or f == frames) continue; // first frame grows buffers; last one includes the screenshot
        fill_total += ms(t0, t1);
        end_total += ms(t1, t2);
        best = @min(best, ms(t0, t2));
    }
    std.debug.print("scene 2 (1M sprites): avg CPU fill {d:.2} ms, avg upload+draw+GPU wait {d:.2} ms, best frame {d:.2} ms\n", .{
        fill_total / (frames - 1), end_total / (frames - 1), best,
    });

    try ballScenes(&g);
}

const tile_px = 64;
const atlas_cols = 8;
const tile_count: usize = cfg.max_level + 1;

/// Atlas of tile_px tiles (box-downsampled assets), plus the {u0,v0,u1,v1} table.
fn buildAtlas(g: *gfx.Gfx, uv: *[tile_count][4]f32) !gfx.Texture {
    const rows = (tile_count + atlas_cols - 1) / atlas_cols;
    const aw = atlas_cols * tile_px;
    const ah = rows * tile_px;
    const px = try std.heap.smp_allocator.alloc(u8, aw * ah * 4);
    defer std.heap.smp_allocator.free(px);
    @memset(px, 0);
    for (0..tile_count) |k| {
        var buf: [64]u8 = undefined;
        const path = if (k == cfg.max_level)
            try std.fmt.bufPrintSentinel(&buf, "assets/balls/basketball.png", .{}, 0)
        else
            try std.fmt.bufPrintSentinel(&buf, "assets/balls/{d}.png", .{k + 1}, 0);
        var size: usize = 0;
        const data = c.SDL_LoadFile(path.ptr, &size) orelse return error.AssetMissing;
        defer c.SDL_free(data);
        var w: c_int = 0;
        var h: c_int = 0;
        var ch: c_int = 0;
        const src = c.stbi_load_from_memory(@ptrCast(data), @intCast(size), &w, &h, &ch, 4) orelse return error.AssetDecode;
        defer c.stbi_image_free(src);
        const uw: usize = @intCast(w);
        const uh: usize = @intCast(h);
        const cx0 = (k % atlas_cols) * tile_px;
        const cy0 = (k / atlas_cols) * tile_px;
        for (0..tile_px) |y| for (0..tile_px) |x| {
            // Premultiplied box filter over the source footprint.
            const x0 = x * uw / tile_px;
            const x1 = @max((x + 1) * uw / tile_px, x0 + 1);
            const y0 = y * uh / tile_px;
            const y1 = @max((y + 1) * uh / tile_px, y0 + 1);
            var acc = [4]u32{ 0, 0, 0, 0 };
            for (y0..y1) |sy| for (x0..x1) |sx| {
                const p = src[(sy * uw + sx) * 4 ..][0..4];
                acc[0] += @as(u32, p[0]) * p[3];
                acc[1] += @as(u32, p[1]) * p[3];
                acc[2] += @as(u32, p[2]) * p[3];
                acc[3] += p[3];
            };
            const n: u32 = @intCast((x1 - x0) * (y1 - y0));
            const dst = px[((cy0 + y) * aw + cx0 + x) * 4 ..][0..4];
            if (acc[3] != 0) for (0..3) |i| {
                dst[i] = @intCast(acc[i] / acc[3]);
            };
            dst[3] = @intCast(acc[3] / n);
        };
        uv[k] = .{
            @as(f32, @floatFromInt(cx0)) / aw,
            @as(f32, @floatFromInt(cy0)) / ah,
            @as(f32, @floatFromInt(cx0 + tile_px)) / aw,
            @as(f32, @floatFromInt(cy0 + tile_px)) / ah,
        };
    }
    return g.uploadTexture(px, aw, ah, true);
}

fn makeBuf(g: *gfx.Gfx, comptime T: type, items: []const T) !*c.SDL_GPUBuffer {
    const size: u32 = @intCast(items.len * @sizeOf(T));
    const buf = c.SDL_CreateGPUBuffer(g.dev, &.{
        .usage = c.SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = size,
    }) orelse return error.GpuAlloc;
    const tb = c.SDL_CreateGPUTransferBuffer(g.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = size }) orelse return error.GpuAlloc;
    defer c.SDL_ReleaseGPUTransferBuffer(g.dev, tb);
    const p: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(g.dev, tb, false).?);
    @memcpy(p[0..size], std.mem.sliceAsBytes(items));
    c.SDL_UnmapGPUTransferBuffer(g.dev, tb);
    const cmd = c.SDL_AcquireGPUCommandBuffer(g.dev) orelse return error.Gpu;
    const copy = c.SDL_BeginGPUCopyPass(cmd);
    c.SDL_UploadToGPUBuffer(copy, &.{ .transfer_buffer = tb }, &.{ .buffer = buf, .size = size }, false);
    c.SDL_EndGPUCopyPass(copy);
    _ = c.SDL_SubmitGPUCommandBuffer(cmd);
    return buf;
}

const Scene = struct {
    pos: []L.BallPos,
    prev: []L.BallPrevPos,
    body: []L.BallBody,
    state: []L.BallState,
    heat: []L.BallHeat,
    meta: []L.BallMeta,

    fn init(n: usize) !Scene {
        const a = std.heap.smp_allocator;
        return .{
            .pos = try a.alloc(L.BallPos, n),
            .prev = try a.alloc(L.BallPrevPos, n),
            .body = try a.alloc(L.BallBody, n),
            .state = try a.alloc(L.BallState, n),
            .heat = try a.alloc(L.BallHeat, n),
            .meta = try a.alloc(L.BallMeta, n),
        };
    }

    fn deinit(s: Scene) void {
        const a = std.heap.smp_allocator;
        a.free(s.pos);
        a.free(s.prev);
        a.free(s.body);
        a.free(s.state);
        a.free(s.heat);
        a.free(s.meta);
    }

    fn set(s: Scene, i: usize, x: f32, y: f32, r: f32, level: u32, scored: bool, temp: f32, burn: f32, ang: f32, rng: std.Random) void {
        s.pos[i] = .{ .x = x, .y = y };
        s.prev[i] = .{ .x = x - 1 + rng.float(f32) * 2, .y = y - 1 + rng.float(f32) * 2 };
        s.body[i] = .{ .radius = r, .mass = 1, .retention = 0.5, .friction = 0.1 };
        s.state[i] = .{ .angle = ang, .prev_angle = ang - 30, .ang_vel = 0, .gravity = 0 };
        s.heat[i] = .{ .temp = temp, .burn = burn };
        s.meta[i] = level | L.flag_alive | (if (scored) L.flag_scored else 0);
    }

    fn upload(s: Scene, g: *gfx.Gfx, tile_uv: *const [tile_count][4]f32, atlas: gfx.Texture) !gfx.BallPass {
        return .{
            .bufs = .{
                .pos = try makeBuf(g, L.BallPos, s.pos),
                .prev_pos = try makeBuf(g, L.BallPrevPos, s.prev),
                .body = try makeBuf(g, L.BallBody, s.body),
                .state = try makeBuf(g, L.BallState, s.state),
                .heat = try makeBuf(g, L.BallHeat, s.heat),
                .meta = try makeBuf(g, L.BallMeta, s.meta),
                .tiles = try makeBuf(g, [4]f32, tile_uv),
            },
            .atlas = atlas,
            .count = @intCast(s.pos.len),
        };
    }
};

fn releaseBalls(g: *gfx.Gfx, bp: gfx.BallPass) void {
    const b = bp.bufs;
    for ([_]*c.SDL_GPUBuffer{ b.pos, b.prev_pos, b.body, b.state, b.heat, b.meta, b.tiles }) |x| c.SDL_ReleaseGPUBuffer(g.dev, x);
}

fn ballScenes(g: *gfx.Gfx) !void {
    var tile_uv: [tile_count][4]f32 = undefined;
    const atlas = try buildAtlas(g, &tile_uv);
    defer g.releaseTexture(atlas);
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();

    // Zoomed scene: 10k small balls, then a tile gallery on top (all 38 levels, basketball, heat/char ramps).
    {
        const n = 10_000;
        const s = try Scene.init(n);
        defer s.deinit();
        const scale: f32 = 6;
        for (0..n - 52) |i| {
            const lvl: u32 = 1 + rng.uintLessThan(u32, cfg.max_level);
            const hot = rng.float(f32) < 0.3;
            s.set(i, rng.float(f32) * W / scale, rng.float(f32) * H / scale, 1.5 + rng.float(f32) * 1.5, lvl, false, if (hot) 21 + rng.float(f32) * 100 else 21, if (rng.float(f32) < 0.1) rng.float(f32) * 100 else 0, rng.float(f32) * 360, rng);
            if (i % 97 == 0) s.meta[i] &= ~L.flag_alive;
        }
        for (n - 52..n) |i| {
            const k = i - (n - 52);
            const col: f32 = @floatFromInt(k % 13);
            const row: f32 = @floatFromInt(k / 13);
            const lvl: u32 = @intCast(@min(k, cfg.max_level - 1) + 1);
            // Rows 0..2: levels 1..38 + basketball; row 3: heat ramp on level 10; char ramp on level 20 below.
            const t: f32 = @floatFromInt(k % 13);
            const heat_row = k >= 39;
            s.set(i, 10 + col * 15, 10 + row * 15, 6.5, if (heat_row) 10 else lvl, k == 38, if (heat_row and k < 52) 21 + t / 12 * 90 else 21, 0, @as(f32, @floatFromInt(k)) * 20, rng);
            s.prev[i] = s.pos[i];
            s.state[i].prev_angle = s.state[i].angle;
            if (k == 39) s.heat[i].burn = 0;
        }
        var bp = try s.upload(g, &tile_uv, atlas);
        defer releaseBalls(g, bp);
        bp.cam = .{ .scale = scale };
        bp.alpha = 0.5;
        g.begin(.{ 0.1, 0.2, 0.15, 1 });
        try g.ballPass(bp);
        try g.end(.{ .wait = true, .screenshot = ".scratch/gfx_balls10k.bmp" });
        var best: f64 = std.math.inf(f64);
        for (0..30) |f| {
            bp.alpha = @as(f32, @floatFromInt(f % 10)) / 10;
            const t0 = now();
            g.begin(.{ 0, 0, 0, 1 });
            try g.ballPass(bp);
            try g.end(.{ .wait = true });
            best = @min(best, ms(t0, now()));
        }
        std.debug.print("balls 10k zoomed: best frame {d:.2} ms\n", .{best});
    }

    inline for (.{ 100_000, 1_000_000 }) |n| {
        const s = try Scene.init(n);
        defer s.deinit();
        for (0..n) |i| {
            const lvl: u32 = 1 + rng.uintLessThan(u32, cfg.max_level);
            const hot = rng.float(f32) < 0.2;
            s.set(i, rng.float(f32) * W, rng.float(f32) * H, 1 + rng.float(f32) * 2, lvl, rng.float(f32) < 0.02, if (hot) 21 + rng.float(f32) * 100 else 21, if (rng.float(f32) < 0.05) rng.float(f32) * 100 else 0, rng.float(f32) * 360, rng);
            if (rng.float(f32) < 0.05) s.meta[i] &= ~L.flag_alive;
        }
        var bp = try s.upload(g, &tile_uv, atlas);
        defer releaseBalls(g, bp);
        var best: f64 = std.math.inf(f64);
        var sum: f64 = 0;
        const frames = 20;
        for (0..frames + 1) |f| {
            bp.alpha = @as(f32, @floatFromInt(f % 10)) / 10;
            const t0 = now();
            g.begin(.{ 0, 0, 0, 1 });
            try g.ballPass(bp);
            try g.end(.{ .wait = true, .screenshot = if (f == frames and n == 1_000_000) ".scratch/gfx_balls.bmp" else null });
            const t = ms(t0, now());
            if (f == 0 or f == frames) continue;
            best = @min(best, t);
            sum += t;
        }
        std.debug.print("balls {d}: avg frame {d:.2} ms, best {d:.2} ms (CPU+GPU wait, no per-ball CPU work)\n", .{ n, sum / (frames - 1), best });
    }
}
