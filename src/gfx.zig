//! Small SDL_GPU host layer: device + window, textures, and batched 2D drawing.
//!
//! Frame flow: `begin` → any number of `sprites`/`triangles` (or `allocSprites` to write in
//! place) → `end`. Draw data is written straight into mapped transfer buffers, uploaded in one
//! copy pass, then replayed in one render pass in call order. Everything renders into an
//! offscreen canvas that is blitted to the swapchain, so screenshots and headless runs share
//! one path (swapchain images are not transfer sources in SDL's Vulkan backend).
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const cfg = @import("config.zig");

pub const Sprite = L.Sprite;
pub const Vertex = L.Vertex;
pub const Texture = *c.SDL_GPUTexture;

const spv = @embedFile("sprite.spv");
const balls_spv = @embedFile("balls.spv");
const trails_spv = @embedFile("trails.spv");
const trail_rec_spv = @embedFile("trail_rec.spv");
const TC = @import("shaders/trail_common.zig");
pub const trail_cap = TC.trail_cap;
pub const trail_len = TC.trail_len;
pub const trail_stride = TC.record_stride;

/// World-to-screen camera: screen = o + (world - b) * scale (render.zig Camera).
pub const BallCamera = struct { scale: f32 = 1, ox: f32 = 0, oy: f32 = 0, bx: f32 = 0, by: f32 = 0 };

/// Must match `Params` in shaders/balls.zig (std140 uniform).
const BallUniforms = extern struct {
    view_w: f32,
    view_h: f32,
    alpha: f32,
    scale: f32,
    ox: f32,
    oy: f32,
    bx: f32,
    by: f32,
    room_temp: f32,
    ignition_temp: f32,
    max_burn: f32,
    max_level: f32,
};

/// Caller-owned GPU buffers read by the ball pass. Create each with usage
/// SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ | COMPUTE_STORAGE_READ | COMPUTE_STORAGE_WRITE
/// (graphics read is what the draw needs; the compute flags let the physics kernels own them).
/// Element types come from gpu_layout.zig, indexed by ball id.
pub const BallBuffers = struct {
    pos: *c.SDL_GPUBuffer, // L.BallPos
    prev_pos: *c.SDL_GPUBuffer, // L.BallPrevPos
    body: *c.SDL_GPUBuffer, // L.BallBody
    state: *c.SDL_GPUBuffer, // L.BallState
    heat: *c.SDL_GPUBuffer, // L.BallHeat
    meta: *c.SDL_GPUBuffer, // L.BallMeta
    /// [cfg.max_level + 1][4]f32 {u0, v0, u1, v1}: levels 1..max_level, then the basketball.
    /// Same usage flags as above (graphics storage read at minimum).
    tiles: *c.SDL_GPUBuffer,
};

/// Ball trails drawn from a GPU ring of past positions (see shaders/trail_common.zig). Call once
/// per frame before the ball pass; `record` appends this frame's positions as a new sample.
pub const TrailPass = struct {
    pos: *c.SDL_GPUBuffer,
    prev_pos: *c.SDL_GPUBuffer,
    body: *c.SDL_GPUBuffer,
    meta: *c.SDL_GPUBuffer,
    count: u32,
    alpha: f32 = 1,
    cam: BallCamera = .{},
    record: bool = false,
};

pub const BallPass = struct {
    bufs: BallBuffers,
    /// Atlas texture the tile table indexes.
    atlas: Texture,
    /// Number of ball ids to draw (alive flag is checked per ball on the GPU).
    count: u32,
    /// Render interpolation between previous and current tick, 0..1.
    alpha: f32 = 1,
    cam: BallCamera = .{},
};

/// Must match `View` in shaders/sprite.zig (std140 uniform).
const View = extern struct { width: f32, height: f32, _pad0: f32 = 0, _pad1: f32 = 0 };

const Kind = enum { sprites, triangles, balls, trails };
/// For `.balls`, `first` indexes `Gfx.ball_passes`.
const Draw = struct { kind: Kind, tex: ?Texture, first: u32, count: u32 };

/// A host-visible staging buffer that stays mapped for the frame and grows on demand.
const Staging = struct {
    tb: ?*c.SDL_GPUTransferBuffer = null,
    buf: ?*c.SDL_GPUBuffer = null,
    cap: u32 = 0, // elements, both buffers
    buf_cap: u32 = 0,
    ptr: [*]u8 = undefined,
    len: u32 = 0, // elements written this frame

    fn map(s: *Staging, dev: *c.SDL_GPUDevice) void {
        s.len = 0;
        if (s.tb) |tb| s.ptr = @ptrCast(c.SDL_MapGPUTransferBuffer(dev, tb, true).?);
    }

    fn alloc(s: *Staging, dev: *c.SDL_GPUDevice, comptime T: type, n: u32) ![]T {
        const need = s.len + n;
        if (need > s.cap) {
            const new_cap = @max(need, s.cap * 2, 1024);
            const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = new_cap * @sizeOf(T) }) orelse return error.GpuAlloc;
            const p: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(dev, tb, false).?);
            if (s.tb) |old| {
                @memcpy(p[0 .. s.len * @sizeOf(T)], s.ptr[0 .. s.len * @sizeOf(T)]);
                c.SDL_UnmapGPUTransferBuffer(dev, old);
                c.SDL_ReleaseGPUTransferBuffer(dev, old);
            }
            s.tb = tb;
            s.ptr = p;
            s.cap = new_cap;
        }
        const items: [*]T = @ptrCast(@alignCast(s.ptr));
        defer s.len = need;
        return items[s.len..need];
    }

    /// Unmaps and records the upload; grows the GPU buffer to fit.
    fn upload(s: *Staging, dev: *c.SDL_GPUDevice, copy: *c.SDL_GPUCopyPass, comptime T: type) !void {
        const tb = s.tb orelse return;
        c.SDL_UnmapGPUTransferBuffer(dev, tb);
        if (s.len == 0) return;
        if (s.buf_cap < s.len) {
            if (s.buf) |b| c.SDL_ReleaseGPUBuffer(dev, b);
            s.buf_cap = s.cap;
            s.buf = c.SDL_CreateGPUBuffer(dev, &.{ .usage = c.SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ, .size = s.buf_cap * @sizeOf(T) }) orelse return error.GpuAlloc;
        }
        c.SDL_UploadToGPUBuffer(copy, &.{ .transfer_buffer = tb, .offset = 0 }, &.{ .buffer = s.buf, .offset = 0, .size = s.len * @sizeOf(T) }, true);
    }

    fn deinit(s: *Staging, dev: *c.SDL_GPUDevice) void {
        if (s.tb) |tb| c.SDL_ReleaseGPUTransferBuffer(dev, tb);
        if (s.buf) |b| c.SDL_ReleaseGPUBuffer(dev, b);
    }
};

pub const Options = struct {
    /// Window to present to, or null for headless (offscreen only).
    window: ?*c.SDL_Window = null,
    /// Canvas size when headless; with a window the canvas follows the swapchain size.
    width: u32 = 1280,
    height: u32 = 720,
    /// false asks for MAILBOX/IMMEDIATE so frame times are not capped by the display.
    vsync: bool = true,
};

pub const Gfx = struct {
    dev: *c.SDL_GPUDevice,
    window: ?*c.SDL_Window,
    format: c.SDL_GPUTextureFormat,
    sprite_pipe: *c.SDL_GPUGraphicsPipeline,
    tri_pipe: *c.SDL_GPUGraphicsPipeline,
    ball_pipe: *c.SDL_GPUGraphicsPipeline,
    sampler: *c.SDL_GPUSampler,
    white: Texture,
    canvas: ?Texture = null,
    canvas_w: u32 = 0,
    canvas_h: u32 = 0,
    width: u32,
    height: u32,
    sprite_stage: Staging = .{},
    vert_stage: Staging = .{},
    draws: std.ArrayList(Draw) = .empty,
    ball_passes: std.ArrayList(BallPass) = .empty,
    clear: [4]f32 = .{ 0, 0, 0, 1 },
    trail_pipe: ?*c.SDL_GPUGraphicsPipeline = null,
    trail_rec_pipe: ?*c.SDL_GPUComputePipeline = null,
    trail_ring: ?*c.SDL_GPUBuffer = null,
    /// Samples recorded since the last clear.
    trail_n: u32 = 0,
    trail_clear: bool = true,
    trail_pass: TrailPass = undefined,
    trail_slot: u32 = 0,
    trail_do_clear: bool = false,
    trail_do_record: bool = false,
    trail_active: bool = false,

    pub fn init(opts: Options) !Gfx {
        // SPIR-V 1.5 shaders (see build.zig) need a Vulkan 1.2 instance; SDL defaults to 1.0.
        var vk_opts = std.mem.zeroes(c.SDL_GPUVulkanOptions);
        vk_opts.vulkan_api_version = (1 << 22) | (2 << 12);
        const props = c.SDL_CreateProperties();
        defer c.SDL_DestroyProperties(props);
        _ = c.SDL_SetBooleanProperty(props, c.SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, true);
        _ = c.SDL_SetPointerProperty(props, c.SDL_PROP_GPU_DEVICE_CREATE_VULKAN_OPTIONS_POINTER, &vk_opts);
        const dev = c.SDL_CreateGPUDeviceWithProperties(props) orelse return fail("SDL_CreateGPUDevice");
        errdefer c.SDL_DestroyGPUDevice(dev);

        var format: c.SDL_GPUTextureFormat = c.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        if (opts.window) |w| {
            if (!c.SDL_ClaimWindowForGPUDevice(dev, w)) return fail("SDL_ClaimWindowForGPUDevice");
            if (!opts.vsync) {
                const mode: c.SDL_GPUPresentMode = if (c.SDL_WindowSupportsGPUPresentMode(dev, w, c.SDL_GPU_PRESENTMODE_MAILBOX))
                    c.SDL_GPU_PRESENTMODE_MAILBOX
                else if (c.SDL_WindowSupportsGPUPresentMode(dev, w, c.SDL_GPU_PRESENTMODE_IMMEDIATE))
                    c.SDL_GPU_PRESENTMODE_IMMEDIATE
                else
                    c.SDL_GPU_PRESENTMODE_VSYNC;
                _ = c.SDL_SetGPUSwapchainParameters(dev, w, c.SDL_GPU_SWAPCHAINCOMPOSITION_SDR, mode);
            }
            format = c.SDL_GetGPUSwapchainTextureFormat(dev, w);
        }

        const vs_sprite = try shader(dev, "spriteVs", c.SDL_GPU_SHADERSTAGE_VERTEX, 0, 1, 1);
        defer c.SDL_ReleaseGPUShader(dev, vs_sprite);
        const vs_tri = try shader(dev, "triVs", c.SDL_GPU_SHADERSTAGE_VERTEX, 0, 1, 1);
        defer c.SDL_ReleaseGPUShader(dev, vs_tri);
        const vs_ball = try shaderFrom(dev, balls_spv, "ballVs", c.SDL_GPU_SHADERSTAGE_VERTEX, 0, 7, 1);
        defer c.SDL_ReleaseGPUShader(dev, vs_ball);
        const fs = try shader(dev, "spriteFs", c.SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 0, 0);
        defer c.SDL_ReleaseGPUShader(dev, fs);

        const sampler = c.SDL_CreateGPUSampler(dev, &.{
            .min_filter = c.SDL_GPU_FILTER_LINEAR,
            .mag_filter = c.SDL_GPU_FILTER_LINEAR,
            .mipmap_mode = c.SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
            .address_mode_u = c.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .address_mode_v = c.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .address_mode_w = c.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .max_lod = 1000,
        }) orelse return fail("SDL_CreateGPUSampler");

        var g = Gfx{
            .dev = dev,
            .window = opts.window,
            .format = format,
            .sprite_pipe = try pipeline(dev, vs_sprite, fs, format),
            .tri_pipe = try pipeline(dev, vs_tri, fs, format),
            .ball_pipe = try pipeline(dev, vs_ball, fs, format),
            .sampler = sampler,
            .white = undefined,
            .width = opts.width,
            .height = opts.height,
        };
        g.white = try g.uploadTexture(&.{ 255, 255, 255, 255 }, 1, 1, false);
        return g;
    }

    pub fn deinit(g: *Gfx) void {
        _ = c.SDL_WaitForGPUIdle(g.dev);
        g.sprite_stage.deinit(g.dev);
        g.vert_stage.deinit(g.dev);
        if (g.trail_pipe) |pp| c.SDL_ReleaseGPUGraphicsPipeline(g.dev, pp);
        if (g.trail_rec_pipe) |pp| c.SDL_ReleaseGPUComputePipeline(g.dev, pp);
        if (g.trail_ring) |b| c.SDL_ReleaseGPUBuffer(g.dev, b);
        g.draws.deinit(std.heap.smp_allocator);
        g.ball_passes.deinit(std.heap.smp_allocator);
        if (g.canvas) |t| c.SDL_ReleaseGPUTexture(g.dev, t);
        c.SDL_ReleaseGPUTexture(g.dev, g.white);
        c.SDL_ReleaseGPUSampler(g.dev, g.sampler);
        c.SDL_ReleaseGPUGraphicsPipeline(g.dev, g.sprite_pipe);
        c.SDL_ReleaseGPUGraphicsPipeline(g.dev, g.tri_pipe);
        c.SDL_ReleaseGPUGraphicsPipeline(g.dev, g.ball_pipe);
        if (g.window) |w| c.SDL_ReleaseWindowFromGPUDevice(g.dev, w);
        c.SDL_DestroyGPUDevice(g.dev);
    }

    /// Uploads tightly packed RGBA8 pixels (straight alpha). Optional GPU-generated mipmaps keep
    /// heavily minified sprites from shimmering.
    pub fn uploadTexture(g: *Gfx, rgba: []const u8, w: u32, h: u32, mipmaps: bool) !Texture {
        std.debug.assert(rgba.len == @as(usize, w) * h * 4);
        const levels: u32 = if (mipmaps) std.math.log2_int(u32, @max(w, h)) + 1 else 1;
        var usage: c.SDL_GPUTextureUsageFlags = c.SDL_GPU_TEXTUREUSAGE_SAMPLER;
        if (levels > 1) usage |= c.SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
        const tex = c.SDL_CreateGPUTexture(g.dev, &.{
            .type = c.SDL_GPU_TEXTURETYPE_2D,
            .format = c.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
            .usage = usage,
            .width = w,
            .height = h,
            .layer_count_or_depth = 1,
            .num_levels = levels,
        }) orelse return fail("SDL_CreateGPUTexture");
        errdefer c.SDL_ReleaseGPUTexture(g.dev, tex);

        const tb = c.SDL_CreateGPUTransferBuffer(g.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = @intCast(rgba.len) }) orelse return fail("SDL_CreateGPUTransferBuffer");
        defer c.SDL_ReleaseGPUTransferBuffer(g.dev, tb);
        const p: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(g.dev, tb, false) orelse return fail("SDL_MapGPUTransferBuffer"));
        @memcpy(p[0..rgba.len], rgba);
        c.SDL_UnmapGPUTransferBuffer(g.dev, tb);

        const cmd = c.SDL_AcquireGPUCommandBuffer(g.dev) orelse return fail("SDL_AcquireGPUCommandBuffer");
        const copy = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_UploadToGPUTexture(copy, &.{ .transfer_buffer = tb }, &.{ .texture = tex, .w = w, .h = h, .d = 1 }, false);
        c.SDL_EndGPUCopyPass(copy);
        if (levels > 1) c.SDL_GenerateMipmapsForGPUTexture(cmd, tex);
        if (!c.SDL_SubmitGPUCommandBuffer(cmd)) return fail("SDL_SubmitGPUCommandBuffer");
        return tex;
    }

    pub fn releaseTexture(g: *Gfx, tex: Texture) void {
        c.SDL_ReleaseGPUTexture(g.dev, tex);
    }

    /// Starts a frame. With a window, the canvas follows the window's pixel size.
    pub fn begin(g: *Gfx, clear: [4]f32) void {
        g.clear = clear;
        g.draws.clearRetainingCapacity();
        g.ball_passes.clearRetainingCapacity();
        g.trail_active = false;
        if (g.window) |w| {
            var pw: c_int = 0;
            var ph: c_int = 0;
            _ = c.SDL_GetWindowSizeInPixels(w, &pw, &ph);
            g.width = @intCast(@max(pw, 1));
            g.height = @intCast(@max(ph, 1));
        }
        g.sprite_stage.map(g.dev);
        g.vert_stage.map(g.dev);
    }

    /// Reserves `n` sprites drawn with `tex` (null = untextured) and returns them to fill in
    /// place; the slice points into mapped GPU-visible memory and is valid until `end`.
    pub fn allocSprites(g: *Gfx, tex: ?Texture, n: u32) ![]Sprite {
        const first = g.sprite_stage.len;
        const out = try g.sprite_stage.alloc(g.dev, Sprite, n);
        try g.push(.sprites, tex, first, n);
        return out;
    }

    pub fn sprites(g: *Gfx, tex: ?Texture, items: []const Sprite) !void {
        @memcpy(try g.allocSprites(tex, @intCast(items.len)), items);
    }

    /// Triangle list in screen pixels; `items.len` must be a multiple of 3.
    pub fn triangles(g: *Gfx, tex: ?Texture, items: []const Vertex) !void {
        std.debug.assert(items.len % 3 == 0);
        const first = g.vert_stage.len;
        @memcpy(try g.vert_stage.alloc(g.dev, Vertex, @intCast(items.len)), items);
        try g.push(.triangles, tex, first, @intCast(items.len));
    }

    /// Draws `pass.count` balls straight from the caller's storage buffers (no per-ball CPU work),
    /// in call order with the other draws. The buffers must stay valid until `end` returns, and any
    /// compute work writing them must be submitted before `end` (same command stream, earlier).
    pub fn ballPass(g: *Gfx, pass: BallPass) !void {
        if (pass.count == 0) return;
        try g.ball_passes.append(std.heap.smp_allocator, pass);
        try g.draws.append(std.heap.smp_allocator, .{ .kind = .balls, .tex = pass.atlas, .first = @intCast(g.ball_passes.items.len - 1), .count = pass.count });
    }

    /// Forgets all trail history (a new scene); applied by the next frame that draws trails.
    pub fn clearTrails(g: *Gfx) void {
        g.trail_clear = true;
    }

    /// Builds the trail pipelines and ring on first use, so the game without trails never
    /// loads the trail shaders.
    fn ensureTrails(g: *Gfx) !void {
        if (g.trail_ring != null) return;
        const dev = g.dev;
        const vs = try shaderFrom(dev, trails_spv, "trailVs", c.SDL_GPU_SHADERSTAGE_VERTEX, 0, 5, 1);
        defer c.SDL_ReleaseGPUShader(dev, vs);
        const fs = try shader(dev, "spriteFs", c.SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 0, 0);
        defer c.SDL_ReleaseGPUShader(dev, fs);
        g.trail_pipe = try pipelineBlend(dev, vs, fs, g.format, true);
        g.trail_rec_pipe = c.SDL_CreateGPUComputePipeline(dev, &.{
            .code_size = trail_rec_spv.len,
            .code = trail_rec_spv.ptr,
            .entrypoint = "record",
            .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
            .num_readonly_storage_buffers = 2,
            .num_readwrite_storage_buffers = 1,
            .num_uniform_buffers = 1,
            .threadcount_x = L.group_size,
            .threadcount_y = 1,
            .threadcount_z = 1,
        }) orelse return fail("SDL_CreateGPUComputePipeline(trail_rec)");
        g.trail_ring = c.SDL_CreateGPUBuffer(dev, &.{
            .usage = c.SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
            .size = TC.trail_cap * TC.trail_len * @sizeOf(L.Vec2),
        }) orelse return fail("SDL_CreateGPUBuffer(trail ring)");
        g.trail_clear = true;
    }

    /// Draws trails for the first `min(count, trail_cap)` ball ids; at most one per frame.
    pub fn trailPass(g: *Gfx, pass: TrailPass) !void {
        if (pass.count == 0) return;
        try g.ensureTrails();
        var p = pass;
        p.count = @min(p.count, TC.trail_cap);
        g.trail_do_clear = g.trail_clear;
        if (g.trail_clear) g.trail_n = 0;
        g.trail_clear = false;
        g.trail_do_record = p.record;
        g.trail_slot = g.trail_n % TC.trail_len;
        if (p.record) g.trail_n += 1;
        g.trail_pass = p;
        g.trail_active = true;
        try g.draws.append(std.heap.smp_allocator, .{ .kind = .trails, .tex = g.white, .first = 0, .count = p.count });
    }

    fn recordTrails(g: *Gfx, cmd: *c.SDL_GPUCommandBuffer) void {
        const p = g.trail_pass;
        const ring = g.trail_ring.?;
        const binds = [1]c.SDL_GPUStorageBufferReadWriteBinding{.{ .buffer = ring, .cycle = false }};
        const ro = [2]*c.SDL_GPUBuffer{ p.pos, p.meta };
        const steps = [2]TC.RecParams{
            .{ .count = TC.trail_cap, .slot = 0, .clear = 1 },
            .{ .count = p.count, .slot = g.trail_slot, .clear = 0 },
        };
        for (steps, 0..) |rp, k| {
            if (k == 0 and !g.trail_do_clear) continue;
            if (k == 1 and !g.trail_do_record) continue;
            c.SDL_PushGPUComputeUniformData(cmd, 0, &rp, @sizeOf(TC.RecParams));
            const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &binds, 1);
            c.SDL_BindGPUComputePipeline(cp, g.trail_rec_pipe.?);
            c.SDL_BindGPUComputeStorageBuffers(cp, 0, &ro, 2);
            c.SDL_DispatchGPUCompute(cp, (rp.count + L.group_size - 1) / L.group_size, 1, 1);
            c.SDL_EndGPUComputePass(cp);
        }
    }

    fn push(g: *Gfx, kind: Kind, tex: ?Texture, first: u32, count: u32) !void {
        if (count == 0) return;
        const draws = g.draws.items;
        if (draws.len > 0) {
            const last = &draws[draws.len - 1];
            if (last.kind == kind and last.tex == tex and last.first + last.count == first) {
                last.count += count;
                return;
            }
        }
        try g.draws.append(std.heap.smp_allocator, .{ .kind = kind, .tex = tex, .first = first, .count = count });
    }

    pub const EndOptions = struct {
        /// Save the frame to this BMP path (waits for the GPU).
        screenshot: ?[*:0]const u8 = null,
        /// Block until the GPU has finished the frame (for timing).
        wait: bool = false,
    };

    /// Uploads, draws, presents (if windowed) and submits the frame.
    pub fn end(g: *Gfx, opts: EndOptions) !void {
        const dev = g.dev;
        const cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return fail("SDL_AcquireGPUCommandBuffer");

        var swap: ?*c.SDL_GPUTexture = null;
        var sw: u32 = 0;
        var sh: u32 = 0;
        if (g.window) |w| {
            if (!c.SDL_WaitAndAcquireGPUSwapchainTexture(cmd, w, &swap, &sw, &sh)) return fail("SDL_WaitAndAcquireGPUSwapchainTexture");
            if (swap != null) {
                g.width = sw;
                g.height = sh;
            }
        }
        try g.ensureCanvas();

        const copy = c.SDL_BeginGPUCopyPass(cmd);
        try g.sprite_stage.upload(dev, copy.?, Sprite);
        try g.vert_stage.upload(dev, copy.?, Vertex);
        c.SDL_EndGPUCopyPass(copy);
        if (g.trail_ring != null and g.trail_active) g.recordTrails(cmd);

        const view = View{ .width = @floatFromInt(g.width), .height = @floatFromInt(g.height) };
        c.SDL_PushGPUVertexUniformData(cmd, 0, &view, @sizeOf(View));
        const target = c.SDL_GPUColorTargetInfo{
            .texture = g.canvas,
            .clear_color = .{ .r = g.clear[0], .g = g.clear[1], .b = g.clear[2], .a = g.clear[3] },
            .load_op = c.SDL_GPU_LOADOP_CLEAR,
            .store_op = c.SDL_GPU_STOREOP_STORE,
            .cycle = true,
        };
        const pass = c.SDL_BeginGPURenderPass(cmd, &target, 1, null) orelse return fail("SDL_BeginGPURenderPass");
        var bound: ?Kind = null;
        for (g.draws.items) |d| {
            if (d.kind == .trails) {
                const tp = g.trail_pass;
                const u = TC.DrawParams{
                    .view_w = @floatFromInt(g.width),
                    .view_h = @floatFromInt(g.height),
                    .alpha = tp.alpha,
                    .scale = tp.cam.scale,
                    .ox = tp.cam.ox,
                    .oy = tp.cam.oy,
                    .bx = tp.cam.bx,
                    .by = tp.cam.by,
                    .head = (g.trail_n +% TC.trail_len -% 1) % TC.trail_len,
                    .max_level = cfg.max_level,
                };
                bound = .trails;
                c.SDL_BindGPUGraphicsPipeline(pass, g.trail_pipe.?);
                const sbufs = [5]*c.SDL_GPUBuffer{ g.trail_ring.?, tp.pos, tp.prev_pos, tp.body, tp.meta };
                c.SDL_BindGPUVertexStorageBuffers(pass, 0, &sbufs, 5);
                c.SDL_PushGPUVertexUniformData(cmd, 0, &u, @sizeOf(TC.DrawParams));
                const bind = c.SDL_GPUTextureSamplerBinding{ .texture = g.white, .sampler = g.sampler };
                c.SDL_BindGPUFragmentSamplers(pass, 0, &bind, 1);
                c.SDL_DrawGPUPrimitives(pass, 6, d.count * TC.trail_len, 0, 0);
                continue;
            }
            if (d.kind == .balls) {
                const bp = g.ball_passes.items[d.first];
                const u = BallUniforms{
                    .view_w = @floatFromInt(g.width),
                    .view_h = @floatFromInt(g.height),
                    .alpha = bp.alpha,
                    .scale = bp.cam.scale,
                    .ox = bp.cam.ox,
                    .oy = bp.cam.oy,
                    .bx = bp.cam.bx,
                    .by = bp.cam.by,
                    .room_temp = cfg.room_temp,
                    .ignition_temp = cfg.ignition_temp,
                    .max_burn = cfg.max_burn,
                    .max_level = cfg.max_level,
                };
                bound = .balls;
                c.SDL_BindGPUGraphicsPipeline(pass, g.ball_pipe);
                const b = bp.bufs;
                const sbufs = [7]*c.SDL_GPUBuffer{ b.pos, b.prev_pos, b.body, b.state, b.heat, b.meta, b.tiles };
                c.SDL_BindGPUVertexStorageBuffers(pass, 0, &sbufs, 7);
                c.SDL_PushGPUVertexUniformData(cmd, 0, &u, @sizeOf(BallUniforms));
                const bind = c.SDL_GPUTextureSamplerBinding{ .texture = bp.atlas, .sampler = g.sampler };
                c.SDL_BindGPUFragmentSamplers(pass, 0, &bind, 1);
                c.SDL_DrawGPUPrimitives(pass, 6, bp.count, 0, 0);
                continue;
            }
            if (bound != d.kind) {
                if (bound == .balls or bound == .trails) c.SDL_PushGPUVertexUniformData(cmd, 0, &view, @sizeOf(View));
                bound = d.kind;
                switch (d.kind) {
                    .balls, .trails => unreachable,
                    .sprites => {
                        c.SDL_BindGPUGraphicsPipeline(pass, g.sprite_pipe);
                        c.SDL_BindGPUVertexStorageBuffers(pass, 0, &g.sprite_stage.buf, 1);
                    },
                    .triangles => {
                        c.SDL_BindGPUGraphicsPipeline(pass, g.tri_pipe);
                        c.SDL_BindGPUVertexStorageBuffers(pass, 0, &g.vert_stage.buf, 1);
                    },
                }
            }
            const bind = c.SDL_GPUTextureSamplerBinding{ .texture = d.tex orelse g.white, .sampler = g.sampler };
            c.SDL_BindGPUFragmentSamplers(pass, 0, &bind, 1);
            // gl_InstanceIndex / gl_VertexIndex include the first-instance / first-vertex offsets,
            // so each draw indexes its own slice of the shared storage buffer.
            switch (d.kind) {
                .sprites => c.SDL_DrawGPUPrimitives(pass, 6, d.count, 0, d.first),
                .triangles => c.SDL_DrawGPUPrimitives(pass, d.count, 1, d.first, 0),
                .balls, .trails => unreachable,
            }
        }
        c.SDL_EndGPURenderPass(pass);

        if (swap) |s| {
            c.SDL_BlitGPUTexture(cmd, &.{
                .source = .{ .texture = g.canvas, .w = g.width, .h = g.height },
                .destination = .{ .texture = s, .w = sw, .h = sh },
                .load_op = c.SDL_GPU_LOADOP_DONT_CARE,
                .filter = c.SDL_GPU_FILTER_NEAREST,
            });
        }

        var download: ?*c.SDL_GPUTransferBuffer = null;
        defer if (download) |d| c.SDL_ReleaseGPUTransferBuffer(dev, d);
        if (opts.screenshot != null) {
            download = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = g.width * g.height * 4 }) orelse return fail("SDL_CreateGPUTransferBuffer");
            const dl = c.SDL_BeginGPUCopyPass(cmd);
            c.SDL_DownloadFromGPUTexture(dl, &.{ .texture = g.canvas, .w = g.width, .h = g.height, .d = 1 }, &.{ .transfer_buffer = download });
            c.SDL_EndGPUCopyPass(dl);
        }

        if (opts.wait or download != null) {
            var fence = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return fail("SDL_SubmitGPUCommandBuffer");
            _ = c.SDL_WaitForGPUFences(dev, true, &fence, 1);
            c.SDL_ReleaseGPUFence(dev, fence);
        } else if (!c.SDL_SubmitGPUCommandBuffer(cmd)) return fail("SDL_SubmitGPUCommandBuffer");

        if (download) |d| try g.saveBmp(d, opts.screenshot.?);
    }

    fn ensureCanvas(g: *Gfx) !void {
        if (g.canvas != null and g.canvas_w == g.width and g.canvas_h == g.height) return;
        if (g.canvas) |t| c.SDL_ReleaseGPUTexture(g.dev, t);
        g.canvas = c.SDL_CreateGPUTexture(g.dev, &.{
            .type = c.SDL_GPU_TEXTURETYPE_2D,
            .format = g.format,
            .usage = c.SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | c.SDL_GPU_TEXTUREUSAGE_SAMPLER,
            .width = g.width,
            .height = g.height,
            .layer_count_or_depth = 1,
            .num_levels = 1,
        }) orelse return fail("SDL_CreateGPUTexture(canvas)");
        g.canvas_w = g.width;
        g.canvas_h = g.height;
    }

    fn saveBmp(g: *Gfx, tb: *c.SDL_GPUTransferBuffer, path: [*:0]const u8) !void {
        const px = c.SDL_MapGPUTransferBuffer(g.dev, tb, false) orelse return fail("SDL_MapGPUTransferBuffer");
        defer c.SDL_UnmapGPUTransferBuffer(g.dev, tb);
        // SDL's packed-format names are word order; on little-endian ABGR8888 is bytes R,G,B,A.
        const pf: c.SDL_PixelFormat = switch (g.format) {
            c.SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM, c.SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB => c.SDL_PIXELFORMAT_ARGB8888,
            else => c.SDL_PIXELFORMAT_ABGR8888,
        };
        const surf = c.SDL_CreateSurfaceFrom(@intCast(g.width), @intCast(g.height), pf, px, @intCast(g.width * 4)) orelse return fail("SDL_CreateSurfaceFrom");
        defer c.SDL_DestroySurface(surf);
        if (!c.SDL_SaveBMP(surf, path)) return fail("SDL_SaveBMP");
    }
};

fn shader(dev: *c.SDL_GPUDevice, entry: [*:0]const u8, stage: c.SDL_GPUShaderStage, samplers: u32, storage_buffers: u32, uniforms: u32) !*c.SDL_GPUShader {
    return shaderFrom(dev, spv, entry, stage, samplers, storage_buffers, uniforms);
}

fn shaderFrom(dev: *c.SDL_GPUDevice, code: []const u8, entry: [*:0]const u8, stage: c.SDL_GPUShaderStage, samplers: u32, storage_buffers: u32, uniforms: u32) !*c.SDL_GPUShader {
    return c.SDL_CreateGPUShader(dev, &.{
        .code_size = code.len,
        .code = code.ptr,
        .entrypoint = entry,
        .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
        .stage = stage,
        .num_samplers = samplers,
        .num_storage_buffers = storage_buffers,
        .num_uniform_buffers = uniforms,
    }) orelse fail("SDL_CreateGPUShader");
}

fn pipeline(dev: *c.SDL_GPUDevice, vs: *c.SDL_GPUShader, fs: *c.SDL_GPUShader, format: c.SDL_GPUTextureFormat) !*c.SDL_GPUGraphicsPipeline {
    return pipelineBlend(dev, vs, fs, format, false);
}

fn pipelineBlend(dev: *c.SDL_GPUDevice, vs: *c.SDL_GPUShader, fs: *c.SDL_GPUShader, format: c.SDL_GPUTextureFormat, additive: bool) !*c.SDL_GPUGraphicsPipeline {
    const color = c.SDL_GPUColorTargetDescription{
        .format = format,
        .blend_state = .{
            .enable_blend = true,
            .src_color_blendfactor = c.SDL_GPU_BLENDFACTOR_SRC_ALPHA,
            .dst_color_blendfactor = if (additive) c.SDL_GPU_BLENDFACTOR_ONE else c.SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
            .color_blend_op = c.SDL_GPU_BLENDOP_ADD,
            .src_alpha_blendfactor = c.SDL_GPU_BLENDFACTOR_ONE,
            .dst_alpha_blendfactor = if (additive) c.SDL_GPU_BLENDFACTOR_ONE else c.SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
            .alpha_blend_op = c.SDL_GPU_BLENDOP_ADD,
        },
    };
    return c.SDL_CreateGPUGraphicsPipeline(dev, &.{
        .vertex_shader = vs,
        .fragment_shader = fs,
        .primitive_type = c.SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = .{ .cull_mode = c.SDL_GPU_CULLMODE_NONE },
        .target_info = .{ .color_target_descriptions = &color, .num_color_targets = 1 },
    }) orelse fail("SDL_CreateGPUGraphicsPipeline");
}

fn fail(what: []const u8) error{Gpu} {
    std.log.err("{s} failed: {s}", .{ what, c.SDL_GetError() });
    return error.Gpu;
}
