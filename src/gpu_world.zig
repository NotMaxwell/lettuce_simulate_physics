//! Balls living on the GPU: owns the solver (gpu_sim Sim), mutual gravity (Gravity, bound to
//! the Sim's pos/body/vel), the game-logic kernels in src/gpu_game/, and the CPU-side
//! bookkeeping the game needs (slot allocation, radius mirror for play-area growth, max level).
//!
//! Each tick records one command buffer (uploads, spawn, actions, gravity, solver, merge apply,
//! hoop, event download) and submits it with a fence, without waiting. Results come back through
//! a ring of download buffers and are read once their fence has signalled, normally two ticks
//! later, so merge/death/score events reach the CPU late by design (accepted by the user).
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const gs = @import("gpu_sim");
const Gravity = @import("gravity").Gravity;
const cfg = @import("config.zig");
const world = @import("world.zig");
const physics = @import("physics.zig");
const effects = @import("effects.zig");
const galaxy = @import("galaxy.zig");

const Buffer = *c.SDL_GPUBuffer;
const Vec2 = world.Vec2;

pub const Error = error{ GpuSetupFailed, OutOfMemory };

/// Mirrors gpu_game/common.zig GameParams.
const GameParams = extern struct {
    ball_count: u32,
    stamp: u32,
    items: u32,
    out_cap: u32,
    mouse_x: f32 = 0,
    mouse_y: f32 = 0,
    max_radius: f32 = cfg.max_radius,
    hoop_y: f32 = 0,
    hoop_left: f32 = 0,
    hoop_right: f32 = 0,
    lo_x: f32 = 0,
    lo_y: f32 = 0,
    hi_x: f32 = 0,
    hi_y: f32 = 0,
    grid_cells_x: u32 = 0,
    grid_cells_y: u32 = 0,
    pool_mask: u32 = 0,
    fx_flags: u32 = 0,
    /// 1 = accrete (gpu_game/common.zig GameParams.collide).
    collide: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
    _pad3: u32 = 0,
    _pad4: u32 = 0,
    _pad5: u32 = 0,
};

/// Mirrors gpu_game/common.zig SpawnRec.
const SpawnRec = extern struct {
    id: u32,
    meta: u32,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    radius: f32,
    mass: f32,
    retention: f32,
    friction: f32,
    gravity: f32,
    temp: f32,
};

/// Ball-format buffers of one particle ring pool (see gpu_game/k_emit.zig).
pub const Pool = struct {
    pos: Buffer,
    prev_pos: Buffer,
    body: Buffer,
    state: Buffer,
    heat: Buffer,
    meta: Buffer,
    cap: u32,

    fn init(dev: *c.SDL_GPUDevice, cap: u32) Error!Pool {
        return .{
            .pos = try buffer(dev, cap * @sizeOf(L.Vec2), true),
            .prev_pos = try buffer(dev, cap * @sizeOf(L.Vec2), true),
            .body = try buffer(dev, cap * @sizeOf(L.BallBody), true),
            .state = try buffer(dev, cap * @sizeOf(L.BallState), true),
            .heat = try buffer(dev, cap * @sizeOf(L.BallHeat), true),
            .meta = try buffer(dev, cap * 4, true),
            .cap = cap,
        };
    }

    fn all(p: *const Pool) [6]Buffer {
        return .{ p.pos, p.prev_pos, p.body, p.state, p.heat, p.meta };
    }
};

/// Fire particle and ash ring sizes (config.max_particles / max_ashes are the CPU caps). Emission
/// is thinned so a ring holds a full lifetime of particles; see k_emit.zig.
pub const fire_pool_cap: u32 = 1 << 20;
pub const ash_pool_cap: u32 = 1 << 19;

/// Latest fire counters read back from the GPU (about two ticks old).
pub const FxStats = struct {
    burning: u32 = 0,
    warm: u32 = 0,
    particles: u32 = 0,
    ashes: u32 = 0,
    fire_head: u32 = 0,
    ash_head: u32 = 0,
};

const event_burnout: u32 = 5;
/// gpu_game/common.zig fx_fire_alive / fx_ash_alive.
const c_fx_fire_alive: u32 = 2;
const c_fx_ash_alive: u32 = 3;
const fx_slots: u32 = 16;

const Blast = extern struct { x: f32, y: f32, blast: f32, force: f32 };

comptime {
    std.debug.assert(@sizeOf(GameParams) == 96);
    std.debug.assert(@sizeOf(SpawnRec) == 48);
}

/// New balls uploaded per tick; a 1M stress fill takes 16 ticks.
pub const spawn_batch: u32 = 1 << 16;
const max_poly: u32 = 4096;
const max_blasts: u32 = 256;
/// Game events per tick. Merges are bounded by gpu_sim's candidate list (max_events - 1); a
/// state change that finds the list full is skipped, never lost, so the CPU mirror stays exact.
const out_cap: u32 = 1 << 15;
const ring = 3;
const out_bytes: u32 = (out_cap + 1) * @sizeOf(L.Event);
const fx_bytes: u32 = fx_slots * 4;
const download_bytes: u32 = out_bytes + 16 + fx_bytes;

const Kernel = enum { spawn, reset, merge, hoop, clearPick, pick, grab, release, lasso, explode, conduct, fxReset, igniteGrab, burn, emitFire, emitAsh, fireUpdate, ashUpdate };
const Module = enum { spawn, merge, act, conduct, burn, emit, particles };
const module_of = std.enums.EnumArray(Kernel, Module).init(.{
    .spawn = .spawn,
    .reset = .merge,
    .merge = .merge,
    .hoop = .merge,
    .clearPick = .act,
    .pick = .act,
    .grab = .act,
    .release = .act,
    .lasso = .act,
    .explode = .act,
    .conduct = .conduct,
    .fxReset = .burn,
    .igniteGrab = .burn,
    .burn = .burn,
    .emitFire = .emit,
    .emitAsh = .emit,
    .fireUpdate = .particles,
    .ashUpdate = .particles,
});
const module_spv = std.enums.EnumArray(Module, []const u8).init(.{
    .spawn = @embedFile("gpu_game_spawn.spv"),
    .merge = @embedFile("gpu_game_merge.spv"),
    .act = @embedFile("gpu_game_act.spv"),
    .conduct = @embedFile("gpu_game_conduct.spv"),
    .burn = @embedFile("gpu_game_burn.spv"),
    .emit = @embedFile("gpu_game_emit.spv"),
    .particles = @embedFile("gpu_game_particles.spv"),
});
const Layout = struct { ro: u32, rw: u32 };
const module_layout = std.enums.EnumArray(Module, Layout).init(.{
    .spawn = .{ .ro = 1, .rw = 7 },
    .merge = .{ .ro = 2, .rw = 6 },
    .act = .{ .ro = 3, .rw = 6 },
    .conduct = .{ .ro = 8, .rw = 1 },
    .burn = .{ .ro = 1, .rw = 6 },
    .emit = .{ .ro = 3, .rw = 8 },
    .particles = .{ .ro = 0, .rw = 7 },
});

/// Gravity softening eps^2 (world units^2) in galaxy mode: bodies are a few units wide and
/// orbit at speed ~2.5, so 2 units of softening tames close passes without blurring orbits.
pub const galaxy_softening: f32 = 4;

/// Half-side of the gravity tree box, in multiples of the scenario's extent.
const galaxy_tree_reach: f32 = 1.75;

pub const Input = struct {
    mouse: Vec2 = .{},
    attract: bool = false,
    mutual_gravity: bool = true,
    merging: bool = true,
    random_explosions: bool = true,
    /// Contact response (L.CollideMode). In galaxy mode the world also forces mutual gravity on,
    /// random explosions off, and merging on only for accrete.
    collide: L.CollideMode = .solid,
};

/// GPU time of the last sampled tick (blocking sample, see `profile_every`).
pub const Profile = struct {
    gravity_ms: f32 = 0,
    collide_ms: f32 = 0,
    game_ms: f32 = 0,
    samples: u32 = 0,
};

const Slot = struct {
    tb: *c.SDL_GPUTransferBuffer,
    fence: ?*c.SDL_GPUFence = null,
    picked: bool = false,
};

pub const GpuWorld = struct {
    gpa: std.mem.Allocator,
    dev: *c.SDL_GPUDevice,
    sim: gs.Sim,
    grav: Gravity,
    heat: Buffer,
    /// Temperatures after conduction (scratch for the Jacobi pass).
    tmp: Buffer,
    /// u32 counters shared by the fire kernels (gpu_game/common.zig fx_*).
    fx: Buffer,
    fire: Pool,
    ash: Pool,
    /// One tile {0, 0, 1, 1}: the tile table for drawing pools with a white texture.
    pool_tiles: Buffer,
    claim: Buffer,
    out: Buffer,
    pick_buf: Buffer,
    spawn_buf: Buffer,
    poly_buf: Buffer,
    blast_buf: Buffer,
    upload_tb: *c.SDL_GPUTransferBuffer,
    pipes: std.enums.EnumArray(Kernel, *c.SDL_GPUComputePipeline),
    slots: [ring]Slot,
    /// Index of the oldest submitted slot whose results are not yet processed.
    read_slot: u32 = 0,
    write_slot: u32 = 0,
    in_flight: u32 = 0,

    /// Ball ids in use: every id below this is either alive or in `free`.
    count: u32 = 0,
    alive: u32 = 0,
    free: std.ArrayList(u32) = .empty,
    /// CPU mirror of every slot's radius (0 = dead), kept exact from spawns and events.
    radius: []f32,
    total_area: f64 = 0,
    max_radius: f32 = cfg.spawn_radius_max,
    pending: std.ArrayList(world.NewBall) = .empty,
    pending_pos: std.ArrayList(Vec2) = .empty,
    pending_head: usize = 0,
    blasts: std.ArrayList(Blast) = .empty,
    lasso_poly: std.ArrayList(Vec2) = .empty,
    lasso_req: bool = false,
    pick_req: bool = false,
    release_req: bool = false,
    ignite_req: bool = false,
    /// Test/showcase hook: every Nth spawned ball (by slot id) starts at `ignite_temp`.
    ignite_every: u32 = 0,
    ignite_temp: f32 = cfg.flame_temp,
    stats: FxStats = .{},
    /// false skips every fire pass (to measure their cost).
    fire_enabled: bool = true,
    /// Ticks left in which something may have been heated that the (late) counters do not show
    /// yet: set by ignition, blasts and hot spawns, so idle fire passes can be skipped safely.
    fire_wake: u32 = 0,
    /// Outcome of the last pick once it has been read back: true = grabbed a ball.
    pick_result: ?bool = null,
    tick_no: u32 = 0,
    profile: Profile = .{},
    /// Every this many ticks one tick is split into separately fenced command buffers and waited
    /// on, to time gravity and the solver (SDL_GPU has no timestamp queries). 0 = never.
    profile_every: u32 = 120,

    /// Galaxy mode state (see loadGalaxy): fixed bounds, forced mutual gravity, own g and softening.
    galaxy_on: bool = false,
    galaxy_scenario: galaxy.Scenario = .disk,
    /// Gravity strength the loaded scenario was built for, and the user's multiplier on it.
    galaxy_g_base: f32 = 0,
    galaxy_g_mult: f32 = 1,
    galaxy_extent: f32 = 0,
    galaxy_centre: Vec2 = .{},
    /// Bumped whenever the scene is replaced; renderers drop per-ball history (trails) on a change.
    scene_epoch: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, dev: *c.SDL_GPUDevice) Error!GpuWorld {
        const cap = L.max_balls;
        var sim = try gs.Sim.init(dev, cap, 2 * cap);
        errdefer sim.deinit();
        var grav = Gravity.init(dev, cap) catch return error.GpuSetupFailed;
        errdefer grav.deinit();
        var gw: GpuWorld = .{
            .gpa = gpa,
            .dev = dev,
            .sim = sim,
            .grav = grav,
            .heat = try buffer(dev, cap * @sizeOf(L.BallHeat), true),
            .tmp = try buffer(dev, cap * 4, false),
            .fx = try buffer(dev, fx_bytes, false),
            .fire = try Pool.init(dev, fire_pool_cap),
            .ash = try Pool.init(dev, ash_pool_cap),
            .pool_tiles = try buffer(dev, 16, true),
            .claim = try buffer(dev, cap * 4, false),
            .out = try buffer(dev, out_bytes, false),
            .pick_buf = try buffer(dev, 16, false),
            .spawn_buf = try buffer(dev, spawn_batch * @sizeOf(SpawnRec), false),
            .poly_buf = try buffer(dev, max_poly * @sizeOf(L.Vec2), false),
            .blast_buf = try buffer(dev, max_blasts * @sizeOf(Blast), false),
            .upload_tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = uploadBytes() }) orelse return error.GpuSetupFailed,
            .pipes = undefined,
            .slots = undefined,
            .radius = try gpa.alloc(f32, cap),
        };
        @memset(gw.radius, 0);
        for (&gw.slots) |*s| s.* = .{ .tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, .size = download_bytes }) orelse return error.GpuSetupFailed };
        const kernels = comptime std.enums.values(Kernel);
        var infos: [kernels.len]c.SDL_GPUComputePipelineCreateInfo = undefined;
        for (kernels, &infos) |k, *info| {
            const m = module_of.get(k);
            const code = module_spv.get(m);
            const lay = module_layout.get(m);
            info.* = .{
                .code_size = code.len,
                .code = code.ptr,
                .entrypoint = @tagName(k),
                .format = c.SDL_GPU_SHADERFORMAT_SPIRV,
                .num_readonly_storage_buffers = lay.ro,
                .num_readwrite_storage_buffers = lay.rw,
                .num_uniform_buffers = 1,
                .threadcount_x = L.group_size,
                .threadcount_y = 1,
                .threadcount_z = 1,
            };
        }
        var made: [kernels.len]?*c.SDL_GPUComputePipeline = undefined;
        if (!gs.createComputePipelines(dev, &infos, &made)) return error.GpuSetupFailed;
        for (kernels, made) |k, p| gw.pipes.set(k, p.?);
        // Claims compare against a stamp that is never 0, so zeroed memory means "unclaimed".
        try gw.zeroBuffer(gw.claim, cap * 4);
        try gw.zeroBuffer(gw.fx, fx_bytes);
        try gw.zeroBuffer(gw.fire.meta, fire_pool_cap * 4);
        try gw.zeroBuffer(gw.ash.meta, ash_pool_cap * 4);
        try gw.upload(gw.pool_tiles, std.mem.sliceAsBytes(&[4]f32{ 0, 0, 1, 1 }));
        return gw;
    }

    pub fn deinit(gw: *GpuWorld) void {
        _ = c.SDL_WaitForGPUIdle(gw.dev);
        for (&gw.slots) |*s| {
            if (s.fence) |f| c.SDL_ReleaseGPUFence(gw.dev, f);
            c.SDL_ReleaseGPUTransferBuffer(gw.dev, s.tb);
        }
        for (std.enums.values(Kernel)) |k| c.SDL_ReleaseGPUComputePipeline(gw.dev, gw.pipes.get(k));
        for ([_]Buffer{ gw.heat, gw.tmp, gw.fx, gw.pool_tiles, gw.claim, gw.out, gw.pick_buf, gw.spawn_buf, gw.poly_buf, gw.blast_buf }) |b| c.SDL_ReleaseGPUBuffer(gw.dev, b);
        for ([_]*const Pool{ &gw.fire, &gw.ash }) |p| for (p.all()) |b| c.SDL_ReleaseGPUBuffer(gw.dev, b);
        c.SDL_ReleaseGPUTransferBuffer(gw.dev, gw.upload_tb);
        gw.grav.deinit();
        gw.sim.deinit();
        gw.gpa.free(gw.radius);
        gw.free.deinit(gw.gpa);
        gw.pending.deinit(gw.gpa);
        gw.pending_pos.deinit(gw.gpa);
        gw.blasts.deinit(gw.gpa);
        gw.lasso_poly.deinit(gw.gpa);
    }

    fn uploadBytes() u32 {
        return spawn_batch * @sizeOf(SpawnRec) + max_poly * @sizeOf(L.Vec2) + max_blasts * @sizeOf(Blast);
    }

    fn zeroBuffer(gw: *GpuWorld, buf: Buffer, size: u32) Error!void {
        return gw.fillBuffer(buf, size, null);
    }

    fn upload(gw: *GpuWorld, buf: Buffer, bytes: []const u8) Error!void {
        return gw.fillBuffer(buf, @intCast(bytes.len), bytes);
    }

    /// Blocking upload of `bytes` (or zeros) into the start of `buf`; setup only.
    fn fillBuffer(gw: *GpuWorld, buf: Buffer, size: u32, bytes: ?[]const u8) Error!void {
        const tb = c.SDL_CreateGPUTransferBuffer(gw.dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = size }) orelse return error.GpuSetupFailed;
        defer c.SDL_ReleaseGPUTransferBuffer(gw.dev, tb);
        const m: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(gw.dev, tb, false) orelse return error.GpuSetupFailed);
        if (bytes) |b| @memcpy(m[0..size], b) else @memset(m[0..size], 0);
        c.SDL_UnmapGPUTransferBuffer(gw.dev, tb);
        const cmd = c.SDL_AcquireGPUCommandBuffer(gw.dev) orelse return error.GpuSetupFailed;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = tb }, &.{ .buffer = buf, .size = size }, false);
        c.SDL_EndGPUCopyPass(cp);
        gs.submitWait(gw.dev, cmd) catch return error.GpuSetupFailed;
    }

    /// Buffers for gfx.ballPass (the renderer adds its tile table).
    pub fn ballBuffers(gw: *const GpuWorld) struct { pos: Buffer, prev_pos: Buffer, body: Buffer, state: Buffer, heat: Buffer, meta: Buffer } {
        return .{ .pos = gw.sim.pos, .prev_pos = gw.sim.prev_pos, .body = gw.sim.body, .state = gw.sim.state, .heat = gw.heat, .meta = gw.sim.meta };
    }

    /// Ring slots worth drawing for each pool: the emitted high-water mark (counters are about two
    /// ticks old, so a margin covers what has been emitted since).
    pub fn poolDrawCounts(gw: *const GpuWorld) struct { fire: u32, ash: u32 } {
        const st = gw.stats;
        const fire: u32 = if (st.particles == 0 and st.burning == 0) 0 else @intCast(@min(gw.fire.cap, @as(u64, st.fire_head) + 3 * @as(u64, st.burning) + 4096));
        const ash: u32 = if (st.ashes == 0) 0 else @intCast(@min(gw.ash.cap, @as(u64, st.ash_head) + 65536));
        return .{ .fire = fire, .ash = ash };
    }

    pub fn pendingSpawns(gw: *const GpuWorld) usize {
        return gw.pending.items.len - gw.pending_head;
    }

    /// Queues one ball with world.spawnBall's random properties; it reaches the GPU on a later
    /// tick (at most spawn_batch per tick). Its area counts towards the play area immediately.
    pub fn queueBall(gw: *GpuWorld, w: *world.World, x: f32, y: f32) !void {
        const nb = world.randomBall(w.prng.random(), w.current_level);
        try gw.pending.append(gw.gpa, nb);
        try gw.pending_pos.append(gw.gpa, .{ .x = x, .y = y });
        gw.total_area += std.math.pi * nb.radius * nb.radius;
    }

    /// physics.spawnBurst for GPU balls: a sunflower spiral one spawn diameter apart.
    pub fn queueBurst(gw: *GpuWorld, w: *world.World, x: f32, y: f32, count: u32) !void {
        if (count == 0) return;
        try gw.pending.ensureUnusedCapacity(gw.gpa, count);
        try gw.pending_pos.ensureUnusedCapacity(gw.gpa, count);
        if (count == 1) return gw.queueBall(w, x, y);
        const n: f32 = @floatFromInt(count);
        const spacing = 2 * cfg.spawn_radius_max;
        const disc_r = spacing * @sqrt(n / std.math.pi);
        const golden_angle = std.math.pi * (3 - @sqrt(5.0));
        for (0..count) |k| {
            const fk: f32 = @floatFromInt(k);
            const rr = disc_r * @sqrt((fk + 0.5) / n);
            const th = fk * golden_angle;
            try gw.queueBall(w, x + rr * @cos(th), y + rr * @sin(th));
        }
    }

    /// Queues one ball with explicit properties (galaxy bodies, systems).
    pub fn queueBody(gw: *GpuWorld, _: *world.World, x: f32, y: f32, nb: world.NewBall) !void {
        try gw.pending.append(gw.gpa, nb);
        errdefer _ = gw.pending.pop();
        try gw.pending_pos.append(gw.gpa, .{ .x = x, .y = y });
        gw.total_area += std.math.pi * nb.radius * nb.radius;
        // The solver's cell size follows this; merges raise it later, spawns must too.
        gw.max_radius = @max(gw.max_radius, nb.radius);
    }

    /// Removes every ball and queued spawn and returns the CPU mirror to its initial state. The
    /// GPU buffers are not wiped: kernels only look at ids below `count`, and a spawn rewrites
    /// every field of its slot.
    fn clearBalls(gw: *GpuWorld, w: *world.World) void {
        gw.scene_epoch +%= 1;
        gw.drain(w);
        @memset(gw.radius[0..gw.count], 0);
        gw.count = 0;
        gw.alive = 0;
        gw.free.clearRetainingCapacity();
        gw.pending.clearRetainingCapacity();
        gw.pending_pos.clearRetainingCapacity();
        gw.pending_head = 0;
        gw.blasts.clearRetainingCapacity();
        gw.lasso_poly.clearRetainingCapacity();
        gw.lasso_req = false;
        gw.pick_req = false;
        gw.release_req = false;
        gw.ignite_req = false;
        gw.pick_result = null;
        gw.total_area = 0;
        gw.max_radius = cfg.spawn_radius_max;
        w.explosions.clearRetainingCapacity();
        w.current_level = cfg.start_level;
        w.events = .{};
        w.pending_events = .{};
    }

    /// Replaces the game with galaxy scenario `s` of about `n` bodies around the current play
    /// area's centre, in a large fixed play area. Bodies reach the GPU over the next ticks.
    pub fn loadGalaxy(gw: *GpuWorld, w: *world.World, s: galaxy.Scenario, n: u32) !void {
        const centre: Vec2 = .{ .x = w.bounds.x + w.bounds.w / 2, .y = w.bounds.y + w.bounds.h / 2 };
        var built = try galaxy.build(gw.gpa, w.prng.random(), s, n, centre);
        defer built.deinit(gw.gpa);
        // Everything that can fail runs before the old world is cleared. clearBalls keeps capacity.
        try gw.pending.ensureUnusedCapacity(gw.gpa, built.bodies.len);
        try gw.pending_pos.ensureUnusedCapacity(gw.gpa, built.bodies.len);
        gw.clearBalls(w);
        for (built.bodies) |b| try gw.queueBody(w, b.x, b.y, galaxy.toNewBall(b));
        w.bounds = galaxy.boundsFor(centre, built.extent);
        w.prev_bounds = w.bounds;
        gw.galaxy_on = true;
        gw.galaxy_scenario = s;
        gw.galaxy_g_base = built.g;
        gw.galaxy_g_mult = 1;
        gw.galaxy_extent = built.extent;
        gw.galaxy_centre = centre;
    }

    /// Current gravity strength in galaxy mode.
    pub fn galaxyG(gw: *const GpuWorld) f32 {
        return gw.galaxy_g_base * gw.galaxy_g_mult;
    }

    /// Raises `galaxy_extent` (and the play area) when bodies lie beyond the gravity tree box, so
    /// they keep their place in the tree instead of being clamped to its edge.
    fn growToFit(gw: *GpuWorld, w: *world.World, bodies: []const galaxy.Body) void {
        var far: f32 = 0;
        for (bodies) |b| far = @max(far, @max(@abs(b.x - gw.galaxy_centre.x), @abs(b.y - gw.galaxy_centre.y)));
        if (far <= galaxy_tree_reach * gw.galaxy_extent) return;
        gw.galaxy_extent = far / galaxy_tree_reach * 1.1;
        w.bounds = galaxy.boundsFor(gw.galaxy_centre, gw.galaxy_extent);
        w.prev_bounds = w.bounds;
    }

    /// One star system at (x, y), at rest, for the current gravity strength.
    pub fn spawnSystemAt(gw: *GpuWorld, w: *world.World, x: f32, y: f32) !void {
        var list: std.ArrayList(galaxy.Body) = .empty;
        defer list.deinit(gw.gpa);
        _ = try galaxy.system(gw.gpa, &list, w.prng.random(), .{ .x = x, .y = y }, .{}, gw.galaxyG());
        gw.growToFit(w, list.items);
        for (list.items) |b| try gw.queueBody(w, b.x, b.y, galaxy.toNewBall(b));
    }

    /// Leaves galaxy mode for a fresh normal game.
    pub fn exitGalaxy(gw: *GpuWorld, w: *world.World) void {
        gw.clearBalls(w);
        gw.galaxy_on = false;
        w.bounds = .{ .x = 0, .y = 0, .w = cfg.base_w, .h = cfg.base_h };
        w.prev_bounds = w.bounds;
    }

    /// Jumps the play area to fit every live and queued ball (stress fills).
    pub fn snapBounds(gw: *const GpuWorld, w: *world.World) void {
        w.bounds = physics.targetBounds(@floatCast(gw.total_area));
        w.prev_bounds = w.bounds;
    }

    pub fn requestPick(gw: *GpuWorld) void {
        gw.pick_req = true;
        gw.pick_result = null;
    }

    /// Right-click: set the ball under the mouse alight (main.onRightClick).
    pub fn requestIgnite(gw: *GpuWorld) void {
        gw.ignite_req = true;
    }

    pub fn requestRelease(gw: *GpuWorld) void {
        gw.release_req = true;
    }

    pub fn requestLasso(gw: *GpuWorld, poly: []const Vec2) !void {
        if (poly.len < 3) return;
        gw.lasso_poly.clearRetainingCapacity();
        // Long lassos are thinned to fit the kernel's polygon buffer.
        const step = (poly.len + max_poly - 1) / max_poly;
        var i: usize = 0;
        while (i < poly.len) : (i += step) try gw.lasso_poly.append(gw.gpa, poly[i]);
        gw.lasso_req = gw.lasso_poly.items.len >= 3;
    }

    /// Queues an explosion sprite plus its GPU knockback (effects.explode for GPU balls).
    fn explode(gw: *GpuWorld, w: *world.World, x: f32, y: f32, ball_size: f32) void {
        effects.explode(w, x, y, ball_size) catch {};
        const size = @floor(ball_size * cfg.explosion_size_mult);
        gw.blasts.append(gw.gpa, .{ .x = x, .y = y, .blast = size * cfg.blast_radius_mult, .force = cfg.base_explosion_force }) catch {};
    }

    /// One simulation tick. Fills `w.events` (from results read back this tick), eases
    /// `w.bounds`, and ages explosions. `w.balls` is not used.
    pub fn tick(gw: *GpuWorld, w: *world.World, in_req: Input) void {
        var in = in_req;
        // Collision modes are a galaxy feature; the merge game is always solid.
        if (!gw.galaxy_on) in.collide = .solid;
        const accrete = in.collide == .accrete;
        if (gw.galaxy_on) {
            in.mutual_gravity = true;
            in.random_explosions = false;
            in.merging = accrete;
        }
        w.events = w.pending_events;
        w.pending_events = .{};
        w.prev_bounds = w.bounds;

        gw.collect(w, false);
        if (gw.in_flight == ring) gw.collect(w, true);

        if (in.random_explosions) {
            const r = w.prng.random();
            if (r.float(f32) < cfg.random_explosion_chance) {
                const b = w.bounds;
                const m = cfg.wall + 10;
                const x = b.x + m + r.float(f32) * @max(0, b.w - 2 * m);
                const y = b.y + m + r.float(f32) * @max(0, b.h - 2 * m);
                const size: f32 = @floatFromInt(r.intRangeAtMost(u8, cfg.random_explosion_size_min, cfg.random_explosion_size_max));
                gw.explode(w, x, y, size);
                w.events.explosions += 1;
            }
        }
        effects.updateExplosions(w);

        const sample = gw.profile_every > 0 and gw.tick_no % gw.profile_every == 0;
        gw.record(w, in, sample) catch |e| std.log.err("gpu tick failed: {s}", .{@errorName(e)});

        w.current_level = @max(w.current_level, cfg.start_level);
        w.events.won = w.current_level >= cfg.max_level;
        if (!gw.galaxy_on) w.bounds = physics.easeBounds(w.bounds, @floatCast(gw.total_area));
        w.tick += 1;
        gw.tick_no += 1;
    }

    /// Processes finished readbacks in submission order; with `wait`, blocks for the oldest.
    fn collect(gw: *GpuWorld, w: *world.World, wait: bool) void {
        var must_wait = wait;
        while (gw.in_flight > 0) {
            const s = &gw.slots[gw.read_slot];
            const f = s.fence.?;
            if (must_wait) {
                _ = c.SDL_WaitForGPUFences(gw.dev, true, &s.fence.?, 1);
                must_wait = false;
            } else if (!c.SDL_QueryGPUFence(gw.dev, f)) return;
            c.SDL_ReleaseGPUFence(gw.dev, f);
            s.fence = null;
            gw.process(w, s);
            gw.read_slot = (gw.read_slot + 1) % ring;
            gw.in_flight -= 1;
        }
    }

    /// Blocks until every submitted tick is done and processed (bench, shutdown, tests).
    pub fn drain(gw: *GpuWorld, w: *world.World) void {
        while (gw.in_flight > 0) gw.collect(w, true);
    }

    fn process(gw: *GpuWorld, w: *world.World, s: *Slot) void {
        const m: [*]const u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(gw.dev, s.tb, false) orelse return);
        defer c.SDL_UnmapGPUTransferBuffer(gw.dev, s.tb);
        const ev: [*]const L.Event = @ptrCast(@alignCast(m));
        const n = @min(ev[0].kind, out_cap);
        for (ev[1 .. n + 1]) |e| switch (e.kind & 0xff) {
            L.event_merge => {
                const lvl: u8 = @intCast(e.kind >> 8);
                const rk = gw.radius[e.a];
                const rg = gw.radius[e.b];
                gw.total_area += std.math.pi * (@as(f64, e.value) * e.value - @as(f64, rk) * rk - @as(f64, rg) * rg);
                gw.radius[e.a] = e.value;
                gw.kill(e.b);
                gw.max_radius = @max(gw.max_radius, e.value);
                w.current_level = @max(w.current_level, lvl);
                w.events.merges += 1;
            },
            L.event_death => {
                const r = gw.radius[e.a];
                gw.kill(e.a);
                gw.explode(w, @bitCast(e.b), e.value, r);
                w.events.explosions += 1;
            },
            L.event_score => w.events.scores += 1,
            event_burnout => {
                gw.kill(e.a);
                w.events.deaths += 1;
            },
            else => {},
        };
        const fxw: [*]const u32 = @ptrCast(@alignCast(m + out_bytes + 16));
        gw.stats = .{
            .burning = fxw[0],
            .warm = fxw[1],
            .particles = fxw[2],
            .ashes = fxw[3],
            .fire_head = fxw[4],
            .ash_head = fxw[5],
        };
        if (gw.stats.burning > 0) w.events.any_burning = true;
        if (s.picked) {
            const pick_word = std.mem.readInt(u32, m[out_bytes..][0..4], .little);
            gw.pick_result = pick_word != 0;
            s.picked = false;
        }
    }

    fn kill(gw: *GpuWorld, id: u32) void {
        const r = gw.radius[id];
        if (r == 0) return;
        gw.total_area = @max(0, gw.total_area - std.math.pi * @as(f64, r) * r);
        gw.radius[id] = 0;
        gw.alive -= 1;
        gw.free.append(gw.gpa, id) catch {};
    }

    /// Mutual gravity strength (vel += g_scale * a). Normal play: physics.zig's crowd formula,
    /// weaker the more balls there are; galaxy mode: the scenario's own g.
    fn gravityScale(gw: *const GpuWorld, sp: L.SimParams) f32 {
        if (gw.galaxy_on) return gw.galaxyG();
        const n = @min(sp.ball_count, gw.grav.capacity);
        const crowd = @min(1.0, cfg.gravity_full_count / @as(f32, @floatFromInt(n)));
        return 0.5 * sp.mutual_g * crowd;
    }

    /// The region the gravity tree covers. Normally the play area; in galaxy mode the play area is
    /// much larger than the bodies (so walls are never met), and a tree over all of it would leave
    /// most leaf cells empty and the occupied ones crowded (1M disk: ~48 ms vs ~5 ms measured).
    /// Walls sit at 4x the extent or more, so bodies can leave the box. The gravity kernels clamp
    /// every body's position into the box (scatterBalls), so an escapee acts as if at the box edge:
    /// it gets an approximate force and cannot corrupt the multipoles. Spawns that land outside grow
    /// the extent (growToFit).
    fn gravityBox(gw: *const GpuWorld, sp: L.SimParams) L.SimParams {
        if (!gw.galaxy_on) return sp;
        var box = sp;
        const r = galaxy_tree_reach * gw.galaxy_extent;
        box.bounds_min_x = gw.galaxy_centre.x - r;
        box.bounds_max_x = gw.galaxy_centre.x + r;
        box.bounds_min_y = gw.galaxy_centre.y - r;
        box.bounds_max_y = gw.galaxy_centre.y + r;
        return box;
    }

    fn simParams(gw: *const GpuWorld, w: *const world.World, in: Input) L.SimParams {
        const b = w.bounds;
        var p: L.SimParams = .{
            .ball_count = gw.count,
            .tick = gw.tick_no,
            .grid_cells_x = 0,
            .grid_cells_y = 0,
            .bounds_min_x = b.x,
            .bounds_min_y = b.y,
            .bounds_max_x = b.x + b.w,
            .bounds_max_y = b.y + b.h,
            // Pairs are only searched in neighbouring cells, so a cell must span the largest diameter.
            .cell_size = 2 * gw.max_radius,
            .mutual_g = cfg.mutual_g,
            .wall = cfg.wall,
            .flags = (if (in.mutual_gravity) L.sim_flag_mutual_gravity else 0) |
                (if (in.merging) L.sim_flag_merging else 0) |
                (if (in.attract) L.sim_flag_attract else 0) |
                (if (gw.galaxy_on) L.sim_flag_galaxy else 0) |
                @as(u32, @backingInt(in.collide)) << L.sim_collide_shift,
            .mouse_x = in.mouse.x,
            .mouse_y = in.mouse.y,
            .attraction_strength = cfg.attraction_strength,
        };
        gw.sim.prepare(&p);
        return p;
    }

    fn record(gw: *GpuWorld, w: *world.World, in: Input, sample: bool) !void {
        const slot = &gw.slots[gw.write_slot];
        std.debug.assert(slot.fence == null);

        // Stage uploads: spawns (into free slots first, then the tail), lasso polygon, blasts.
        const up: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(gw.dev, gw.upload_tb, true) orelse return error.GpuSetupFailed);
        const recs: [*]SpawnRec = @ptrCast(@alignCast(up));
        var n_spawn: u32 = 0;
        while (n_spawn < spawn_batch and gw.pending_head < gw.pending.items.len) {
            const id = gw.free.pop() orelse blk: {
                if (gw.count == L.max_balls) break;
                gw.count += 1;
                break :blk gw.count - 1;
            };
            const nb = gw.pending.items[gw.pending_head];
            const p = gw.pending_pos.items[gw.pending_head];
            gw.pending_head += 1;
            recs[n_spawn] = .{
                .id = id,
                .meta = @as(u32, nb.level) | L.flag_alive,
                .x = p.x,
                .y = p.y,
                .vx = nb.vx,
                .vy = nb.vy,
                .radius = nb.radius,
                .mass = nb.mass,
                .retention = nb.retention,
                .friction = nb.friction,
                .gravity = nb.gravity,
                .temp = if (gw.ignite_every > 0 and id % gw.ignite_every == 0) gw.ignite_temp else cfg.room_temp,
            };
            gw.radius[id] = nb.radius;
            gw.alive += 1;
            n_spawn += 1;
        }
        if (gw.pending_head == gw.pending.items.len) {
            gw.pending.clearRetainingCapacity();
            gw.pending_pos.clearRetainingCapacity();
            gw.pending_head = 0;
        }
        // Balls that found no slot (capacity full) are dropped; their area is given back.
        if (gw.count == L.max_balls and gw.free.items.len == 0 and gw.pendingSpawns() > 0) {
            for (gw.pending.items[gw.pending_head..]) |nb| gw.total_area -= std.math.pi * nb.radius * nb.radius;
            gw.pending.clearRetainingCapacity();
            gw.pending_pos.clearRetainingCapacity();
            gw.pending_head = 0;
        }
        const poly_off = spawn_batch * @sizeOf(SpawnRec);
        const blast_off = poly_off + max_poly * @sizeOf(L.Vec2);
        const n_poly: u32 = if (gw.lasso_req) @intCast(gw.lasso_poly.items.len) else 0;
        if (n_poly > 0) @memcpy(@as([*]Vec2, @ptrCast(@alignCast(up + poly_off)))[0..n_poly], gw.lasso_poly.items);
        const n_blast: u32 = @intCast(@min(gw.blasts.items.len, max_blasts));
        const blast_dst: [*]Blast = @ptrCast(@alignCast(up + blast_off));
        @memcpy(blast_dst[0..n_blast], gw.blasts.items[0..n_blast]);
        gw.blasts.replaceRangeAssumeCapacity(0, n_blast, &.{});
        c.SDL_UnmapGPUTransferBuffer(gw.dev, gw.upload_tb);

        var timer = Timer{ .on = sample, .dev = gw.dev };
        if (sample) _ = c.SDL_WaitForGPUIdle(gw.dev);
        var cmd = c.SDL_AcquireGPUCommandBuffer(gw.dev) orelse return error.GpuSetupFailed;
        timer.start();

        if (n_spawn > 0 or n_poly > 0 or n_blast > 0) {
            const cp = c.SDL_BeginGPUCopyPass(cmd);
            if (n_spawn > 0) c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = gw.upload_tb, .offset = 0 }, &.{ .buffer = gw.spawn_buf, .size = n_spawn * @sizeOf(SpawnRec) }, true);
            if (n_poly > 0) c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = gw.upload_tb, .offset = poly_off }, &.{ .buffer = gw.poly_buf, .size = n_poly * @sizeOf(L.Vec2) }, true);
            if (n_blast > 0) c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = gw.upload_tb, .offset = blast_off }, &.{ .buffer = gw.blast_buf, .size = n_blast * @sizeOf(Blast) }, true);
            c.SDL_EndGPUCopyPass(cp);
        }

        const b = w.bounds;
        const hoop_scale = b.w / cfg.base_w;
        var gp: GameParams = .{
            .ball_count = gw.count,
            .stamp = gw.tick_no + 1,
            .items = 0,
            .out_cap = out_cap,
            .mouse_x = in.mouse.x,
            .mouse_y = in.mouse.y,
            .hoop_y = b.y + cfg.hoop_y * hoop_scale,
            .hoop_left = b.x + cfg.hoop_left * hoop_scale,
            .hoop_right = b.x + cfg.hoop_right * hoop_scale,
            .collide = @intFromBool(in.collide == .accrete),
        };
        const groups = divCeil(@max(gw.count, 1), L.group_size);

        if (n_spawn > 0) {
            gp.items = n_spawn;
            gw.dispatch(cmd, .spawn, &gp, divCeil(n_spawn, L.group_size));
        }
        gw.dispatch(cmd, .reset, &gp, 1);
        if (n_poly > 0) {
            var lo: Vec2 = .{ .x = std.math.inf(f32), .y = std.math.inf(f32) };
            var hi: Vec2 = .{ .x = -std.math.inf(f32), .y = -std.math.inf(f32) };
            for (gw.lasso_poly.items) |p| {
                lo = .{ .x = @min(lo.x, p.x), .y = @min(lo.y, p.y) };
                hi = .{ .x = @max(hi.x, p.x), .y = @max(hi.y, p.y) };
            }
            gp.items = n_poly;
            gp.lo_x = lo.x;
            gp.lo_y = lo.y;
            gp.hi_x = hi.x;
            gp.hi_y = hi.y;
            gw.dispatch(cmd, .lasso, &gp, groups);
            gw.lasso_req = false;
        }
        if (n_blast > 0) {
            gp.items = n_blast;
            gw.dispatch(cmd, .explode, &gp, groups);
        }
        if (gw.pick_req) {
            gw.dispatch(cmd, .clearPick, &gp, 1);
            gw.dispatch(cmd, .pick, &gp, groups);
            gw.dispatch(cmd, .grab, &gp, 1);
            gw.pick_req = false;
            slot.picked = true;
        }
        var ignite_now = false;
        if (gw.ignite_req and !gw.pick_req) {
            gw.dispatch(cmd, .clearPick, &gp, 1);
            gw.dispatch(cmd, .pick, &gp, groups);
            gw.dispatch(cmd, .igniteGrab, &gp, 1);
            gw.ignite_req = false;
            ignite_now = true;
        }
        if (gw.release_req) {
            gw.dispatch(cmd, .release, &gp, groups);
            gw.release_req = false;
        }

        const sp = gw.simParams(w, in);
        if (in.mutual_gravity) gw.grav.apply(cmd, gw.gravityBox(sp), .{ .pos = gw.sim.pos, .body = gw.sim.body, .vel = gw.sim.vel }, gw.gravityScale(sp), if (gw.galaxy_on) galaxy_softening else @import("gravity").default_softening);
        cmd = try timer.lap(cmd, &gw.profile.gravity_ms);
        gw.sim.step(cmd, sp);
        cmd = try timer.lap(cmd, &gw.profile.collide_ms);
        if (in.merging) gw.dispatch(cmd, .merge, &gp, divCeil(L.max_events - 1, L.group_size));
        if (!gw.galaxy_on) gw.dispatch(cmd, .hoop, &gp, groups);

        // Fire: roll the counters, conduct heat (Jacobi), update burning/burn-out, emit and move
        // particles. Uses this tick's broad-phase grid from the solver step above.
        if (gw.fire_enabled) {
            if (n_blast > 0 or ignite_now or (n_spawn > 0 and gw.ignite_every > 0)) gw.fire_wake = 4;
            gw.recordFire(cmd, &gp, b, sp, groups);
            if (gw.fire_wake > 0) gw.fire_wake -= 1;
        }
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = gw.out, .size = out_bytes }, &.{ .transfer_buffer = slot.tb, .offset = 0 });
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = gw.pick_buf, .size = 16 }, &.{ .transfer_buffer = slot.tb, .offset = out_bytes });
        c.SDL_DownloadFromGPUBuffer(cp, &.{ .buffer = gw.fx, .size = fx_bytes }, &.{ .transfer_buffer = slot.tb, .offset = out_bytes + 16 });
        c.SDL_EndGPUCopyPass(cp);
        slot.fence = c.SDL_SubmitGPUCommandBufferAndAcquireFence(cmd) orelse return error.GpuSetupFailed;
        if (sample) {
            _ = c.SDL_WaitForGPUFences(gw.dev, true, &slot.fence.?, 1);
            gw.profile.game_ms = timer.since();
            gw.profile.samples += 1;
        }
        gw.write_slot = (gw.write_slot + 1) % ring;
        gw.in_flight += 1;
    }

    fn recordFire(gw: *GpuWorld, cmd: *c.SDL_GPUCommandBuffer, gp_in: *const GameParams, b: world.Rect, sp: L.SimParams, groups: u32) void {
        var gp = gp_in.*;
        const st = gw.stats;
        const woke = gw.fire_wake > 0;
        const hot = woke or st.warm > 0;
        const burning = woke or st.burning > 0;
        // Everything below early-outs on the GPU anyway; skipping the dispatches saves the
        // ~1.4 ms that 1M idle threads per pass cost.
        if (!(hot or burning or st.particles > 0 or st.ashes > 0)) return;
        gp.grid_cells_x = sp.grid_cells_x;
        gp.grid_cells_y = sp.grid_cells_y;
        gp.fx_flags = @intFromBool(hot);
        gw.dispatch(cmd, .fxReset, &gp, 1);
        if (hot) gw.dispatch(cmd, .conduct, &gp, groups);
        if (hot or burning) gw.dispatch(cmd, .burn, &gp, groups);
        gp.lo_x = b.x;
        gp.lo_y = b.y;
        gp.hi_x = b.x + b.w;
        gp.hi_y = b.y + b.h;
        if (burning) {
            gp.pool_mask = fire_pool_cap - 1;
            gw.dispatch(cmd, .emitFire, &gp, groups);
            gp.pool_mask = ash_pool_cap - 1;
            gw.dispatch(cmd, .emitAsh, &gp, groups);
        }
        if (burning or st.particles > 0) {
            gp.pool_mask = fire_pool_cap - 1;
            gp.items = c_fx_fire_alive;
            gw.dispatch(cmd, .fireUpdate, &gp, divCeil(fire_pool_cap, L.group_size));
        }
        if (burning or st.ashes > 0) {
            gp.pool_mask = ash_pool_cap - 1;
            gp.items = c_fx_ash_alive;
            gw.dispatch(cmd, .ashUpdate, &gp, divCeil(ash_pool_cap, L.group_size));
        }
    }

    fn dispatch(gw: *GpuWorld, cmd: *c.SDL_GPUCommandBuffer, k: Kernel, gp: *const GameParams, groups: u32) void {
        const pool = if (k == .emitAsh or k == .ashUpdate) &gw.ash else &gw.fire;
        const pb = pool.all();
        if (groups == 0) return;
        c.SDL_PushGPUComputeUniformData(cmd, 0, gp, @sizeOf(GameParams));
        const s = &gw.sim;
        var ro_buf: [8]Buffer = undefined;
        var rw_buf: [8]Buffer = undefined;
        const ro: []const Buffer, const rw: []const Buffer = switch (module_of.get(k)) {
            .spawn => .{ blk: {
                ro_buf[0] = gw.spawn_buf;
                break :blk ro_buf[0..1];
            }, blk: {
                rw_buf[0..7].* = .{ s.pos, s.prev_pos, s.vel, s.body, s.state, gw.heat, s.meta };
                break :blk rw_buf[0..7];
            } },
            .merge => .{ blk: {
                ro_buf[0..2].* = .{ s.events, s.prev_pos };
                break :blk ro_buf[0..2];
            }, blk: {
                rw_buf[0..6].* = .{ s.pos, s.vel, s.body, s.meta, gw.claim, gw.out };
                break :blk rw_buf[0..6];
            } },
            .act => .{ blk: {
                ro_buf[0..3].* = .{ s.pos, gw.poly_buf, gw.blast_buf };
                break :blk ro_buf[0..3];
            }, blk: {
                rw_buf[0..6].* = .{ s.vel, s.body, s.meta, gw.pick_buf, gw.out, gw.heat };
                break :blk rw_buf[0..6];
            } },
            .conduct => .{ blk: {
                ro_buf[0..8].* = .{ s.pos, s.body, s.ball_cell, s.counts, s.sorted_id, s.meta, gw.heat, gw.fx };
                break :blk ro_buf[0..8];
            }, blk: {
                rw_buf[0] = gw.tmp;
                break :blk rw_buf[0..1];
            } },
            .burn => .{ blk: {
                ro_buf[0] = gw.tmp;
                break :blk ro_buf[0..1];
            }, blk: {
                rw_buf[0..6].* = .{ gw.heat, s.meta, s.body, gw.out, gw.fx, gw.pick_buf };
                break :blk rw_buf[0..6];
            } },
            .emit => .{ blk: {
                ro_buf[0..3].* = .{ s.pos, s.body, s.vel };
                break :blk ro_buf[0..3];
            }, blk: {
                rw_buf[0] = s.meta;
                rw_buf[1..7].* = pb;
                rw_buf[7] = gw.fx;
                break :blk rw_buf[0..8];
            } },
            .particles => .{ ro_buf[0..0], blk: {
                rw_buf[0..6].* = pb;
                rw_buf[6] = gw.fx;
                break :blk rw_buf[0..7];
            } },
        };
        var binds: [8]c.SDL_GPUStorageBufferReadWriteBinding = undefined;
        for (rw, 0..) |bf, i| binds[i] = .{ .buffer = bf, .cycle = false };
        const cp = c.SDL_BeginGPUComputePass(cmd, null, 0, &binds, @intCast(rw.len));
        c.SDL_BindGPUComputePipeline(cp, gw.pipes.get(k));
        c.SDL_BindGPUComputeStorageBuffers(cp, 0, @ptrCast(ro.ptr), @intCast(ro.len));
        c.SDL_DispatchGPUCompute(cp, groups, 1, 1);
        c.SDL_EndGPUComputePass(cp);
    }
};

/// Splits a sampled tick into separately waited command buffers to time each stage.
const Timer = struct {
    on: bool,
    dev: *c.SDL_GPUDevice,
    t0: u64 = 0,
    t: u64 = 0,

    fn start(tm: *Timer) void {
        tm.t0 = c.SDL_GetPerformanceCounter();
        tm.t = tm.t0;
    }

    fn lap(tm: *Timer, cmd: *c.SDL_GPUCommandBuffer, out: *f32) !*c.SDL_GPUCommandBuffer {
        if (!tm.on) return cmd;
        gs.submitWait(tm.dev, cmd) catch return error.GpuSetupFailed;
        const now = c.SDL_GetPerformanceCounter();
        out.* = msBetween(tm.t, now);
        tm.t = now;
        return c.SDL_AcquireGPUCommandBuffer(tm.dev) orelse error.GpuSetupFailed;
    }

    fn since(tm: *Timer) f32 {
        return msBetween(tm.t, c.SDL_GetPerformanceCounter());
    }
};

fn msBetween(a: u64, b: u64) f32 {
    return @floatCast(@as(f64, @floatFromInt(b - a)) * 1000.0 / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency())));
}

fn buffer(dev: *c.SDL_GPUDevice, size: u32, graphics: bool) Error!Buffer {
    var usage: u32 = c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | c.SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE;
    if (graphics) usage |= c.SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ;
    return c.SDL_CreateGPUBuffer(dev, &.{ .usage = usage, .size = size }) orelse {
        std.log.err("gpu_world: buffer of {d} bytes: {s}", .{ size, c.SDL_GetError() });
        return error.GpuSetupFailed;
    };
}

fn divCeil(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

test "gpu world: spawn, merge, pick/grab/release, lasso, explosion" {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SkipZigTest;
    defer c.SDL_Quit();
    const gfx = @import("gfx.zig");
    var gpu = gfx.Gfx.init(.{ .window = null }) catch return error.SkipZigTest;
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    var gw = try GpuWorld.init(gpa, gpu.dev);
    defer gw.deinit();
    gw.profile_every = 0;
    var w = world.World.init(gpa, 7);
    defer w.deinit();
    const in: Input = .{ .mutual_gravity = false, .merging = true, .random_explosions = false };

    // Two overlapping balls of one level merge; a third far away of another level does not.
    for ([_]f32{ 400, 430, 700 }) |x| try gw.queueBall(&w, x, 300);
    gw.pending.items[0].level = 2;
    gw.pending.items[1].level = 2;
    gw.pending.items[2].level = 3;
    for (gw.pending.items) |*nb| {
        nb.vx = 0;
        nb.vy = 0;
        nb.gravity = 0;
    }
    for (0..3) |_| gw.tick(&w, in);
    gw.drain(&w);
    try std.testing.expectEqual(@as(u32, 2), gw.alive);
    try std.testing.expectEqual(@as(u8, cfg.start_level), w.current_level);
    var meta: [3]u32 = undefined;
    var pos: [3]L.Vec2 = undefined;
    try gw.sim.download(0, .{ .meta = &meta, .pos = &pos });
    const keep: usize = if (meta[0] & L.flag_alive != 0) 0 else 1;
    try std.testing.expectEqual(@as(u32, 3), meta[keep] & 0xff);
    try std.testing.expectEqual(@as(usize, 1), gw.free.items.len);

    // Pick the far ball and grab it, then release: it ends up thrown.
    gw.requestPick();
    gw.tick(&w, .{ .mouse = .{ .x = pos[2].x, .y = pos[2].y }, .mutual_gravity = false, .random_explosions = false });
    gw.drain(&w);
    try std.testing.expectEqual(@as(?bool, true), gw.pick_result);
    try gw.sim.download(0, .{ .meta = &meta });
    try std.testing.expect(meta[2] & L.flag_grabbed != 0);
    gw.requestRelease();
    gw.tick(&w, .{ .mouse = .{ .x = pos[2].x, .y = pos[2].y }, .mutual_gravity = false, .random_explosions = false });
    gw.drain(&w);
    try gw.sim.download(0, .{ .meta = &meta });
    try std.testing.expect(meta[2] & L.flag_grabbed == 0 and meta[2] & L.flag_thrown != 0);

    // A miss reports false.
    gw.requestPick();
    gw.tick(&w, .{ .mouse = .{ .x = -5000, .y = -5000 }, .mutual_gravity = false, .random_explosions = false });
    gw.drain(&w);
    try std.testing.expectEqual(@as(?bool, false), gw.pick_result);

    // Lasso around the merged ball kills it, spawns an explosion and knocks the other ball.
    try gw.sim.download(0, .{ .pos = &pos });
    const p = pos[keep];
    const poly = [_]Vec2{ .{ .x = p.x - 20, .y = p.y - 20 }, .{ .x = p.x + 20, .y = p.y - 20 }, .{ .x = p.x + 20, .y = p.y + 20 }, .{ .x = p.x - 20, .y = p.y + 20 } };
    try gw.requestLasso(&poly);
    gw.tick(&w, in);
    gw.drain(&w);
    try std.testing.expectEqual(@as(u32, 1), gw.alive);
    try std.testing.expectEqual(@as(usize, 1), w.explosions.items.len);
    var vel: [3]L.Vec2 = undefined;
    try gw.sim.download(0, .{ .vel = &vel });
    const v0 = vel[2];
    gw.tick(&w, in);
    gw.drain(&w);
    try gw.sim.download(0, .{ .vel = &vel });
    try std.testing.expect(vel[2].x > v0.x);

    // Freed slots are reused before the tail grows.
    try gw.queueBall(&w, 200, 200);
    gw.tick(&w, in);
    gw.drain(&w);
    try std.testing.expectEqual(@as(u32, 3), gw.count);
    try std.testing.expectEqual(@as(u32, 2), gw.alive);
}

test "gpu world: conduction ignites a neighbour, burning balls emit particles and burn out" {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SkipZigTest;
    defer c.SDL_Quit();
    const gfx = @import("gfx.zig");
    var gpu = gfx.Gfx.init(.{ .window = null }) catch return error.SkipZigTest;
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    var gw = try GpuWorld.init(gpa, gpu.dev);
    defer gw.deinit();
    gw.profile_every = 0;
    var w = world.World.init(gpa, 9);
    defer w.deinit();
    const in: Input = .{ .mutual_gravity = false, .merging = false, .random_explosions = false };

    // Ball 0 starts far above ignition and touches ball 1; ball 2 is far away. Ball 0's heat
    // transfer to ball 1 alone is 107 degrees per tick, enough to cross the ignition point.
    gw.ignite_every = 3;
    gw.ignite_temp = 100_000;
    for ([_]f32{ 400, 460, 800 }) |x| try gw.queueBall(&w, x, 300);
    for (gw.pending.items, 0..) |*nb, k| {
        nb.vx = 0;
        nb.vy = 0;
        nb.gravity = 0;
        nb.radius = 30;
        nb.density = 1;
        nb.mass = 140;
        nb.level = @intCast(k + 1);
    }
    for (0..8) |_| gw.tick(&w, in);
    gw.drain(&w);
    var meta: [3]u32 = undefined;
    try gw.sim.download(0, .{ .meta = &meta });
    try std.testing.expect(meta[0] & L.flag_burning != 0);
    try std.testing.expect(meta[1] & L.flag_burning != 0);
    try std.testing.expect(meta[2] & L.flag_burning == 0);
    try std.testing.expect(gw.stats.burning == 2);
    try std.testing.expect(gw.stats.particles > 0);
    try std.testing.expect(w.events.any_burning or gw.stats.burning > 0);

    // Burn rises 0.1..0.3 per tick against max_burn 100, so by 1100 ticks both have burnt out.
    var deaths: u32 = 0;
    var peak_ashes: u32 = 0;
    for (0..1100) |_| {
        gw.tick(&w, in);
        deaths += w.events.deaths;
        peak_ashes = @max(peak_ashes, gw.stats.ashes);
    }
    gw.drain(&w);
    deaths += w.events.deaths;
    try std.testing.expectEqual(@as(u32, 2), deaths);
    try std.testing.expectEqual(@as(u32, 1), gw.alive);
    try std.testing.expect(gw.stats.ash_head > 0);
    try std.testing.expect(peak_ashes > 0);
    try std.testing.expectEqual(@as(u32, 0), gw.stats.burning);
}

const GalaxyDump = struct {
    pos: []L.Vec2,
    vel: []L.Vec2,
    body: []L.BallBody,
    meta: []L.BallMeta,

    fn read(gpa: std.mem.Allocator, gw: *GpuWorld) !GalaxyDump {
        const n = gw.count;
        const d: GalaxyDump = .{
            .pos = try gpa.alloc(L.Vec2, n),
            .vel = try gpa.alloc(L.Vec2, n),
            .body = try gpa.alloc(L.BallBody, n),
            .meta = try gpa.alloc(L.BallMeta, n),
        };
        try gw.sim.download(0, .{ .pos = d.pos, .vel = d.vel, .body = d.body, .meta = d.meta });
        return d;
    }

    fn free(d: GalaxyDump, gpa: std.mem.Allocator) void {
        gpa.free(d.pos);
        gpa.free(d.vel);
        gpa.free(d.body);
        gpa.free(d.meta);
    }

    fn mass(d: GalaxyDump) f64 {
        var m: f64 = 0;
        for (d.body, d.meta) |b, mt| {
            if (mt & L.flag_alive != 0) m += b.mass;
        }
        return m;
    }

    /// Mass-weighted centre and angular momentum about it over the live bodies.
    fn centreAndSpin(d: GalaxyDump) struct { cx: f64, cy: f64, spin: f64 } {
        var m: f64 = 0;
        var cx: f64 = 0;
        var cy: f64 = 0;
        for (d.pos, d.body, d.meta) |p, b, mt| {
            if (mt & L.flag_alive == 0) continue;
            m += b.mass;
            cx += b.mass * p.x;
            cy += b.mass * p.y;
        }
        cx /= m;
        cy /= m;
        var spin: f64 = 0;
        for (d.pos, d.vel, d.body, d.meta) |p, v, b, mt| {
            if (mt & L.flag_alive == 0) continue;
            spin += b.mass * ((p.x - cx) * v.y - (p.y - cy) * v.x);
        }
        return .{ .cx = cx, .cy = cy, .spin = spin };
    }
};

test "gpu world: galaxy disk in ghost mode keeps its shape and angular momentum" {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SkipZigTest;
    defer c.SDL_Quit();
    const gfx = @import("gfx.zig");
    var gpu = gfx.Gfx.init(.{ .window = null }) catch return error.SkipZigTest;
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    var gw = try GpuWorld.init(gpa, gpu.dev);
    defer gw.deinit();
    gw.profile_every = 0;
    var w = world.World.init(gpa, 11);
    defer w.deinit();

    try gw.loadGalaxy(&w, .disk, 2000);
    try std.testing.expect(gw.galaxy_on);
    const extent = gw.galaxy_extent;
    for (0..8) |_| gw.tick(&w, .{ .collide = .ghost });
    gw.drain(&w);
    try std.testing.expectEqual(@as(u32, 2001), gw.alive);
    const d0 = try GalaxyDump.read(gpa, &gw);
    defer d0.free(gpa);
    const s0 = d0.centreAndSpin();
    for (0..600) |_| gw.tick(&w, .{ .collide = .ghost });
    gw.drain(&w);
    const d1 = try GalaxyDump.read(gpa, &gw);
    defer d1.free(gpa);
    const s1 = d1.centreAndSpin();

    var inside: usize = 0;
    for (d1.pos, d1.vel) |p, v| {
        try std.testing.expect(std.math.isFinite(p.x) and std.math.isFinite(p.y) and std.math.isFinite(v.x) and std.math.isFinite(v.y));
        const dx = p.x - s0.cx;
        const dy = p.y - s0.cy;
        if (@sqrt(dx * dx + dy * dy) < 1.5 * extent) inside += 1;
    }
    const frac = @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(d1.pos.len));
    const drift = @abs(s1.spin - s0.spin) / @abs(s0.spin);
    std.debug.print("galaxy ghost: inside 1.5x extent {d:.1}%, angular momentum drift {d:.2}%\n", .{ frac * 100, drift * 100 });
    try std.testing.expect(frac >= 0.9);
    try std.testing.expect(drift < 0.05);
    // Ghost never merges.
    try std.testing.expectEqual(@as(u32, 2001), gw.alive);
}

test "gpu world: galaxy accretion merges bodies and conserves mass" {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SkipZigTest;
    defer c.SDL_Quit();
    const gfx = @import("gfx.zig");
    var gpu = gfx.Gfx.init(.{ .window = null }) catch return error.SkipZigTest;
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    var gw = try GpuWorld.init(gpa, gpu.dev);
    defer gw.deinit();
    gw.profile_every = 0;
    var w = world.World.init(gpa, 12);
    defer w.deinit();

    try gw.loadGalaxy(&w, .disk, 2000);
    for (0..4) |_| gw.tick(&w, .{ .collide = .accrete });
    gw.drain(&w);
    const d0 = try GalaxyDump.read(gpa, &gw);
    defer d0.free(gpa);
    const m0 = d0.mass();
    const n0 = gw.alive;
    for (0..200) |_| gw.tick(&w, .{ .collide = .accrete });
    gw.drain(&w);
    const d1 = try GalaxyDump.read(gpa, &gw);
    defer d1.free(gpa);
    var live: u32 = 0;
    for (d1.meta) |mt| live += @intFromBool(mt & L.flag_alive != 0);
    std.debug.print("galaxy accrete: bodies {d} -> {d}, mass {d:.1} -> {d:.1}\n", .{ n0, live, m0, d1.mass() });
    try std.testing.expect(live < n0);
    // The CPU mirror follows the GPU's merges.
    try std.testing.expectEqual(live, gw.alive);
    try std.testing.expectApproxEqRel(m0, d1.mass(), 1e-4);
    for (d1.pos, d1.meta) |p, mt| if (mt & L.flag_alive != 0) {
        try std.testing.expect(std.math.isFinite(p.x) and std.math.isFinite(p.y));
    };
}

const Escapee = union(enum) { none, far: f32, corner: f32 };

/// Runs a 2000-body disk for a few ticks with an optional extra body of the given mass: `far` is
/// outside the gravity tree box, `corner` sits on the box corner that a far body is clamped to.
/// Returns every body's velocity; the extra body, if any, is last.
fn escapeeRun(gpa: std.mem.Allocator, dev: *c.SDL_GPUDevice, esc: Escapee) ![]L.Vec2 {
    var gw = try GpuWorld.init(gpa, dev);
    defer gw.deinit();
    gw.profile_every = 0;
    var w = world.World.init(gpa, 21);
    defer w.deinit();
    try gw.loadGalaxy(&w, .disk, 2000);
    const reach = galaxy_tree_reach * gw.galaxy_extent;
    const centre = gw.galaxy_centre;
    switch (esc) {
        .none => {},
        .far => |m| {
            const far = 3.5 * gw.galaxy_extent;
            try std.testing.expect(far > reach and far < w.bounds.w / 2);
            try gw.queueBody(&w, centre.x + far, centre.y + far, galaxy.toNewBall(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = m, .radius = 1.5 }));
        },
        .corner => |m| try gw.queueBody(&w, centre.x + reach, centre.y + reach, galaxy.toNewBall(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = m, .radius = 1.5 })),
    }
    for (0..8) |_| gw.tick(&w, .{ .collide = .ghost });
    gw.drain(&w);
    const d = try GalaxyDump.read(gpa, &gw);
    defer d.free(gpa);
    try std.testing.expectEqual(@as(u32, if (esc == .none) 2001 else 2002), gw.alive);
    return gpa.dupe(L.Vec2, d.vel);
}

/// Largest velocity difference over the first `a.len` bodies; fails on non-finite values.
fn worstDiff(a: []const L.Vec2, b: []const L.Vec2) !f32 {
    var worst: f32 = 0;
    for (a, b[0..a.len]) |p, q| {
        try std.testing.expect(std.math.isFinite(q.x) and std.math.isFinite(q.y));
        worst = @max(worst, @max(@abs(p.x - q.x), @abs(p.y - q.y)));
    }
    return worst;
}

test "gpu world: a body outside the gravity box does not corrupt the others" {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SkipZigTest;
    defer c.SDL_Quit();
    const gfx = @import("gfx.zig");
    var gpu = gfx.Gfx.init(.{ .window = null }) catch return error.SkipZigTest;
    defer gpu.deinit();
    const gpa = std.testing.allocator;
    const none = try escapeeRun(gpa, gpu.dev, .none);
    defer gpa.free(none);
    const light = try escapeeRun(gpa, gpu.dev, .{ .far = 0.01 });
    defer gpa.free(light);
    const heavy_far = try escapeeRun(gpa, gpu.dev, .{ .far = 5 });
    defer gpa.free(heavy_far);
    const heavy_corner = try escapeeRun(gpa, gpu.dev, .{ .corner = 5 });
    defer gpa.free(heavy_corner);

    // A light escapee barely pulls on anyone (disk speeds are ~2.5).
    const light_diff = try worstDiff(none, light);
    // A heavy one acts as if at the box corner, so the disk must see the same as with a body there.
    const heavy_diff = try worstDiff(none, heavy_far);
    const clamp_diff = try worstDiff(heavy_corner[0..2001], heavy_far);
    const e = light[none.len];
    std.debug.print("escapee: light vs none {d:.6}, heavy far vs corner {d:.6} (heavy far vs none {d:.6}), escapee v ({d:.4}, {d:.4})\n", .{ light_diff, clamp_diff, heavy_diff, e.x, e.y });
    try std.testing.expect(light_diff < 0.01);
    try std.testing.expect(clamp_diff < 1e-4);
    try std.testing.expect(std.math.isFinite(e.x) and std.math.isFinite(e.y));
    try std.testing.expect(e.x < 0 and e.y < 0);
}
