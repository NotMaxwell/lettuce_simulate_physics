//! Fixed-step simulation. One tick equals one frame of the Python original (main.py loop).
const std = @import("std");
const cfg = @import("config.zig");
const world = @import("world.zig");
const effects = @import("effects.zig");

const World = world.World;
const Vec2 = world.Vec2;
const Rect = world.Rect;
const TickInput = world.TickInput;

const V = std.simd.suggestVectorLength(f32) orelse 8;
const F = @Vector(V, f32);

/// Collision grid cell: one spawn-size diameter plus one merge of growth, so while no ball
/// has grown past spawn size every query spans at most one cell sideways. Larger merged
/// balls are found by widening the query by the current max radius, not by growing cells.
const cell_size: f32 = 2 * (cfg.spawn_radius_max + 5);
/// Below these ball counts waking the worker pool costs more than it saves.
const gravity_thread_min_balls = 1500;
const collide_thread_min_balls = 1500;

/// The per-ball fields the collision pass touches, gathered in grid-cell order so each
/// neighbour query walks contiguous memory.
const HotBall = struct {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    r: f32,
    m: f32,
    density: f32,
    temp: f32,
    ang_vel: f32,
    level: u8,
    alive: bool,
    cell: u32,
    idx: u32,
};

const Scratch = struct {
    gx: std.ArrayList(f32) = .empty,
    gy: std.ArrayList(f32) = .empty,
    gm: std.ArrayList(f32) = .empty,
    ax: std.ArrayList(f32) = .empty,
    ay: std.ArrayList(f32) = .empty,
    ball_cell: std.ArrayList(u32) = .empty,
    cell_start: std.ArrayList(u32) = .empty,
    hot: std.MultiArrayList(HotBall) = .empty,
    ash: effects.AshGrid = .{},

    fn free(gpa: std.mem.Allocator, p: *anyopaque) void {
        const s: *Scratch = @ptrCast(@alignCast(p));
        inline for (.{ "gx", "gy", "gm", "ax", "ay", "ball_cell", "cell_start", "hot" }) |name| @field(s, name).deinit(gpa);
        s.ash.deinit(gpa);
        gpa.destroy(s);
    }

    /// Grows buffers to fit `n` balls; a no-op once capacity is reached.
    fn reserve(s: *Scratch, gpa: std.mem.Allocator, n: usize, ncells: usize) !void {
        const np = padded(n);
        inline for (.{ "gx", "gy", "gm", "ax", "ay" }) |name| try @field(s, name).ensureTotalCapacity(gpa, np);
        try s.ball_cell.ensureTotalCapacity(gpa, n);
        try s.hot.ensureTotalCapacity(gpa, n);
        try s.cell_start.ensureTotalCapacity(gpa, ncells + 1);
    }
};

fn padded(n: usize) usize {
    return std.mem.alignForward(usize, n, V);
}

fn scratchOf(w: *World) !*Scratch {
    if (w.scratch) |p| return @ptrCast(@alignCast(p));
    const s = try w.gpa.create(Scratch);
    s.* = .{};
    w.scratch = s;
    w.scratch_free = &Scratch.free;
    try effects.reservePools(w, &s.ash);
    return s;
}

const GridDims = struct {
    ox: f32,
    oy: f32,
    nx: u32,
    ny: u32,

    fn of(b: Rect) GridDims {
        const margin = cfg.max_radius;
        const nx: u32 = @intFromFloat(@ceil((b.w + 2 * margin) / cell_size));
        const ny: u32 = @intFromFloat(@ceil((b.h + 2 * margin) / cell_size));
        return .{ .ox = b.x - margin, .oy = b.y - margin, .nx = @max(nx, 1), .ny = @max(ny, 1) };
    }

    fn cx(g: GridDims, x: f32) u32 {
        return clampCell((x - g.ox) / cell_size, g.nx);
    }

    fn cy(g: GridDims, y: f32) u32 {
        return clampCell((y - g.oy) / cell_size, g.ny);
    }

    /// Monotone clamp, so two balls within reach always land in overlapping cell ranges
    /// even when they sit outside the grid (e.g. a grabbed ball dragged off-screen).
    fn clampCell(f: f32, n: u32) u32 {
        const c = std.math.clamp(@floor(f), 0, @as(f32, @floatFromInt(n - 1)));
        return @intFromFloat(c);
    }
};

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds;
}

fn since(t: i96) u64 {
    return @intCast(@max(0, nowNs() - t));
}

pub fn tick(w: *World, in: TickInput) void {
    const t_start = nowNs();
    w.events = w.pending_events;
    w.pending_events = .{};
    w.profile = .{};
    const sc = scratchOf(w) catch return;
    const dims = GridDims.of(w.bounds);
    sc.reserve(w.gpa, w.balls.len, dims.nx * dims.ny) catch return;

    w.prev_bounds = w.bounds;
    {
        const s = w.balls.slice();
        @memcpy(s.items(.prev_x), s.items(.x));
        @memcpy(s.items(.prev_y), s.items(.y));
        @memcpy(s.items(.prev_angle), s.items(.angle));
    }

    if (in.attract) attract(w, in.mouse);

    const t_grav = nowNs();
    if (in.mutual_gravity and w.balls.len > 1) mutualGravity(w, sc);
    w.profile.gravity_ns = since(t_grav);

    integrate(w, in);

    const t_col = nowNs();
    collide(w, sc, dims, in.merging);
    applyBounds(w);
    w.profile.collide_ns = since(t_col);

    const t_fx = nowNs();
    effects.emitFire(w);
    effects.updateParticles(w);
    effects.updateAshes(w, &sc.ash);
    if (in.random_explosions) randomExplosion(w);
    effects.updateExplosions(w);
    w.profile.effects_ns = since(t_fx);

    removeDead(w);
    pushTrails(w);
    const fill_area = updateLevelAndEvents(w);
    w.bounds = easeBounds(w.bounds, fill_area);
    w.tick += 1;
    w.profile.total_ns = since(t_start);
}

fn attract(w: *World, mouse: Vec2) void {
    const s = w.balls.slice();
    for (s.items(.x), s.items(.y), s.items(.vx), s.items(.vy), s.items(.ang_vel)) |x, y, *vx, *vy, *av| {
        vx.* = (mouse.x - x) * cfg.attraction_strength;
        vy.* = (mouse.y - y) * cfg.attraction_strength;
        av.* = 0;
    }
}

// ---------------------------------------------------------------- mutual gravity

/// Python: a_i = sum_j G*m_j*d/(|d|^2 + 0.001), applied at half strength.
/// The self term is 0 because d = 0, and padding lanes carry m = 0, so neither needs masking.
fn mutualGravity(w: *World, sc: *Scratch) void {
    const n = w.balls.len;
    const np = padded(n);
    const s = w.balls.slice();
    sc.gx.items.len = np;
    sc.gy.items.len = np;
    sc.gm.items.len = np;
    sc.ax.items.len = np;
    sc.ay.items.len = np;
    @memcpy(sc.gx.items[0..n], s.items(.x));
    @memcpy(sc.gy.items[0..n], s.items(.y));
    // The pull on a ball grows with total mass, so past a normal game's population the
    // Python constant makes balls outrun their own diameter per tick and tunnel. Capping the
    // total pull at `gravity_full_count` balls' worth leaves small games identical.
    const crowd: f32 = @min(1.0, cfg.gravity_full_count / @as(f32, @floatFromInt(n)));
    const g_scale = 0.5 * cfg.mutual_g * crowd;
    for (sc.gm.items[0..n], s.items(.mass)) |*g, m| g.* = g_scale * m;
    @memset(sc.gx.items[n..], 0);
    @memset(sc.gy.items[n..], 0);
    @memset(sc.gm.items[n..], 0);

    var job: GravityJob = .{
        .gx = sc.gx.items,
        .gy = sc.gy.items,
        .gm = sc.gm.items,
        .ax = sc.ax.items,
        .ay = sc.ay.items,
        .n = n,
    };
    w.profile.gravity_threads = 1;
    const p = if (n >= gravity_thread_min_balls) pool.get() else null;
    if (p) |pl| {
        const threads = pl.workers + 1;
        job.rows_per_block = @max(16, std.mem.alignForward(usize, n / (threads * 4), 4));
        const blocks: u32 = @intCast(std.math.divCeil(usize, n, job.rows_per_block) catch unreachable);
        w.profile.gravity_threads = pl.run(.{ .ctx = &job, .func = GravityJob.runBlock, .blocks = blocks });
    } else job.rows(0, n);

    for (s.items(.vx), s.items(.vy), sc.ax.items[0..n], sc.ay.items[0..n]) |*vx, *vy, ax, ay| {
        vx.* += ax;
        vy.* += ay;
    }
}

const GravityJob = struct {
    gx: []const f32,
    gy: []const f32,
    gm: []const f32,
    ax: []f32,
    ay: []f32,
    n: usize,
    rows_per_block: usize = 0,

    fn runBlock(ctx: *anyopaque, b: u32) void {
        const j: *const GravityJob = @ptrCast(@alignCast(ctx));
        const lo = b * j.rows_per_block;
        j.rows(lo, @min(j.n, lo + j.rows_per_block));
    }

    /// Rows are done four at a time so each loaded column vector feeds four rows.
    fn rows(j: GravityJob, lo: usize, hi: usize) void {
        var i = lo;
        while (i + 4 <= hi) : (i += 4) j.block(4, i);
        while (i < hi) : (i += 1) j.block(1, i);
    }

    inline fn block(j: GravityJob, comptime R: usize, row0: usize) void {
        @setFloatMode(.optimized);
        var xi: [R]F = undefined;
        var yi: [R]F = undefined;
        var accx: [R]F = @splat(@splat(0));
        var accy: [R]F = @splat(@splat(0));
        inline for (0..R) |r| {
            xi[r] = @splat(j.gx[row0 + r]);
            yi[r] = @splat(j.gy[row0 + r]);
        }
        const eps: F = @splat(0.001);
        var k: usize = 0;
        while (k < j.gx.len) : (k += V) {
            const px: F = j.gx[k..][0..V].*;
            const py: F = j.gy[k..][0..V].*;
            const pm: F = j.gm[k..][0..V].*;
            inline for (0..R) |r| {
                const dx = px - xi[r];
                const dy = py - yi[r];
                const f = pm / (dx * dx + dy * dy + eps);
                accx[r] += f * dx;
                accy[r] += f * dy;
            }
        }
        inline for (0..R) |r| {
            j.ax[row0 + r] = @reduce(.Add, accx[r]);
            j.ay[row0 + r] = @reduce(.Add, accy[r]);
        }
    }
};

// ---------------------------------------------------------------- worker pool

const pool = struct {
    var instance: Pool = .{};
    var state: enum { untried, ready, failed } = .untried;

    fn get() ?*Pool {
        switch (state) {
            .ready => return &instance,
            .failed => return null,
            .untried => {
                state = if (instance.start()) .ready else .failed;
                return get();
            },
        }
    }
};

const Task = struct {
    ctx: *anyopaque,
    func: *const fn (*anyopaque, u32) void,
    blocks: u32,
};

/// Persistent workers that split one task into blocks. Blocks are claimed through a single
/// atomic word holding (generation, block count, next index), so the main thread never
/// waits for a worker that has not woken up yet: it takes the blocks itself and only waits
/// for blocks already claimed. That keeps a descheduled worker from stalling a tick.
const Pool = struct {
    workers: u32 = 0,
    /// Futex word bumped once per task.
    gen: std.atomic.Value(u32) = .init(0),
    claim: std.atomic.Value(u64) = .init(0),
    done_blocks: std.atomic.Value(u32) = .init(0),
    task: Task = undefined,

    const max_workers = 15;
    /// Tens of microseconds of pause instructions before a worker sleeps on the futex,
    /// so the several tasks of one tick (and back-to-back catch-up ticks) skip the wake-up.
    const spin_before_sleep = 4000;
    const gen_bits = 24;
    const count_bits = 20;
    const count_mask = (1 << count_bits) - 1;

    fn io() std.Io {
        return std.Io.Threaded.global_single_threaded.io();
    }

    fn pack(g: u32, blocks: u32, idx: u32) u64 {
        return (@as(u64, g & ((1 << gen_bits) - 1)) << (2 * count_bits)) | (@as(u64, blocks) << count_bits) | idx;
    }

    fn start(p: *Pool) bool {
        const cpus = std.Thread.getCpuCount() catch return false;
        // Leave SMT siblings and a core for the render/audio threads.
        const want: u32 = @intCast(@min(max_workers, cpus / 2 -| 1));
        for (0..want) |_| {
            // Default stack: glibc rejects small stacks once static TLS is counted.
            const t = std.Thread.spawn(.{}, workerMain, .{p}) catch break;
            t.detach();
            p.workers += 1;
        }
        return p.workers > 0;
    }

    /// Runs `task` to completion on the pool plus the calling thread.
    /// Returns how many threads ran at least one block (approximate: assumes the blocks
    /// the caller did not run went to distinct workers).
    fn run(p: *Pool, task: Task) u32 {
        std.debug.assert(task.blocks <= count_mask);
        p.task = task;
        p.done_blocks.store(0, .monotonic);
        const g = p.gen.load(.monotonic) +% 1;
        p.claim.store(pack(g, task.blocks, 0), .release);
        p.gen.store(g, .release);
        io().futexWake(u32, &p.gen.raw, std.math.maxInt(u32));

        var mine: u32 = 0;
        while (p.work(g)) mine += 1;
        var spins: u32 = 0;
        while (p.done_blocks.load(.acquire) != task.blocks) {
            std.atomic.spinLoopHint();
            spins += 1;
            if (spins % 4096 == 0) std.Thread.yield() catch {};
        }
        return @min(p.workers + 1, 1 + task.blocks - mine);
    }

    /// Claims and runs one block of generation `g`; false when none are left.
    fn work(p: *Pool, g: u32) bool {
        var cur = p.claim.load(.acquire);
        while (true) {
            if (cur >> (2 * count_bits) != g & ((1 << gen_bits) - 1)) return false;
            const idx: u32 = @intCast(cur & count_mask);
            const blocks: u32 = @intCast((cur >> count_bits) & count_mask);
            if (idx >= blocks) return false;
            cur = p.claim.cmpxchgWeak(cur, cur + 1, .acq_rel, .acquire) orelse {
                // A successful claim under generation `g` means the task for `g` is
                // published and cannot be replaced until this block is reported done.
                p.task.func(p.task.ctx, idx);
                _ = p.done_blocks.fetchAdd(1, .release);
                return true;
            };
        }
    }

    fn workerMain(p: *Pool) void {
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            var g = p.gen.load(.acquire);
            while (g == seen) : (g = p.gen.load(.acquire)) {
                if (spins < spin_before_sleep) {
                    std.atomic.spinLoopHint();
                    spins += 1;
                } else io().futexWaitUncancelable(u32, &p.gen.raw, seen);
            }
            seen = g;
            while (p.work(g)) {}
        }
    }
};

// ---------------------------------------------------------------- per-ball update

fn integrate(w: *World, in: TickInput) void {
    const s = w.balls.slice();
    const r = w.prng.random();
    const b = w.bounds;
    const xs = s.items(.x);
    const ys = s.items(.y);
    const vxs = s.items(.vx);
    const vys = s.items(.vy);
    const temps = s.items(.temp);
    const flags = s.items(.flags);
    const radii = s.items(.radius);
    // The background stretches to fill the play area, so the hoop drawn in it moves and
    // scales with the bounds; config gives its position in base-window coordinates.
    const hoop_scale = b.w / cfg.base_w;
    const hoop_y = b.y + cfg.hoop_y * hoop_scale;
    const hoop_left = b.x + cfg.hoop_left * hoop_scale;
    const hoop_right = b.x + cfg.hoop_right * hoop_scale;
    for (0..w.balls.len) |i| {
        const f = &flags[i];
        // Ball.update_fire
        if (temps[i] > cfg.room_temp) temps[i] = @max(cfg.room_temp, temps[i] - 0.1);
        if (temps[i] >= cfg.ignition_temp) {
            f.burning = true;
        } else {
            temps[i] = cfg.room_temp;
            f.burning = false;
        }

        const old_x = xs[i];
        const old_y = ys[i];
        xs[i] += vxs[i];
        ys[i] += vys[i];
        s.items(.angle)[i] += s.items(.ang_vel)[i];
        s.items(.ang_vel)[i] *= 0.98;

        if (f.grabbed) {
            // Ball.handleGrab
            s.items(.grab_prev)[i] = .{ .x = xs[i], .y = ys[i] };
            vxs[i] += in.mouse.x - xs[i];
            vys[i] += in.mouse.y - ys[i];
            xs[i] = in.mouse.x;
            ys[i] = in.mouse.y;
        }

        // Ball.increase_burn
        if (f.burning) s.items(.burn)[i] += 0.1 + r.float(f32) * 0.2;
        if (s.items(.burn)[i] >= cfg.max_burn) {
            effects.spawnAshes(w, xs[i], ys[i], vxs[i], vys[i], radii[i]);
            f.alive = false;
            w.events.deaths += 1;
            continue;
        }

        if (f.thrown and !f.scored and old_y < hoop_y and hoop_y <= ys[i]) {
            const frac = (hoop_y - old_y) / (ys[i] - old_y);
            const crossing = old_x + frac * (xs[i] - old_x);
            if (hoop_left <= crossing and crossing <= hoop_right) {
                f.scored = true;
                w.events.scores += 1;
            }
        }

        if (!in.mutual_gravity) {
            // Ball.apply_gravity_and_velocity
            const floor_y = b.y + b.h - radii[i] - cfg.wall / 2;
            if (ys[i] < floor_y) vys[i] += s.items(.gravity)[i];
            if (vys[i] == 0 and vxs[i] != 0) {
                const fr = s.items(.friction)[i];
                vxs[i] += if (vxs[i] > 0) -fr else fr;
                if (@abs(vxs[i]) < fr) vxs[i] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------- ball collisions

const Balls = struct {
    x: []f32,
    y: []f32,
    vx: []f32,
    vy: []f32,
    r: []f32,
    m: []f32,
    density: []f32,
    temp: []f32,
    ang_vel: []f32,
    level: []u8,
    alive: []bool,

    fn of(hot: *std.MultiArrayList(HotBall)) Balls {
        const s = hot.slice();
        return .{
            .x = s.items(.x),
            .y = s.items(.y),
            .vx = s.items(.vx),
            .vy = s.items(.vy),
            .r = s.items(.r),
            .m = s.items(.m),
            .density = s.items(.density),
            .temp = s.items(.temp),
            .ang_vel = s.items(.ang_vel),
            .level = s.items(.level),
            .alive = s.items(.alive),
        };
    }
};

/// Uniform grid over the cell-sorted hot copy: cell c holds slots [starts[c], starts[c+1]).
const Grid = struct {
    dims: GridDims,
    starts: []const u32,
    cell: []const u32,
    /// Largest radius at build time plus the most one merge can add, so a ball that grows
    /// mid-pass is still covered.
    rmax: f32,

    /// Most cells a query reaches sideways; strips this wide never touch each other's
    /// neighbours' neighbours.
    fn maxReachCells(g: Grid) u32 {
        return @intFromFloat(@ceil(2 * g.rmax / cell_size));
    }

    /// Visits every slot k > si whose cell is within reach of slot si. Slots are in
    /// row-major cell order, so rows above si hold only k < si and are skipped: each
    /// pair is seen once, from its lower slot. Candidates arrive as contiguous slot ranges
    /// through `visitor.range(lo, hi)`; it returns false to stop.
    inline fn forward(g: Grid, bl: Balls, si: usize, visitor: anytype) void {
        const nx = g.dims.nx;
        const reach = bl.r[si] + g.rmax;
        const R: u32 = @intFromFloat(@ceil(reach / cell_size));
        const c = g.cell[si];
        const cx = c % nx;
        const cy = c / nx;
        // Position-based range, clamped to the membership-based one so the cells touched
        // stay within R columns of the ball's own cell (the strip scheduler relies on it).
        const x_lo = @max(cx -| R, g.dims.cx(bl.x[si] - reach));
        const x_hi = @min(cx + R, g.dims.cx(bl.x[si] + reach));
        const y_hi = @min(cy + R, g.dims.cy(bl.y[si] + reach));
        if (x_hi >= cx) {
            const end = g.starts[cy * nx + x_hi + 1];
            if (si + 1 < end and !visitor.range(si + 1, end)) return;
        }
        if (x_lo > x_hi) return;
        var row = cy + 1;
        while (row <= y_hi) : (row += 1) {
            const base = row * nx;
            if (!visitor.range(g.starts[base + x_lo], g.starts[base + x_hi + 1])) return;
        }
    }
};

/// Counting-sorts alive balls by grid cell into `sc.hot`. Returns null if there is nothing to do.
fn buildHot(w: *World, sc: *Scratch, dims: GridDims, merging: bool) ?Grid {
    const n = w.balls.len;
    const s = w.balls.slice();
    const flags = s.items(.flags);
    const xs = s.items(.x);
    const ys = s.items(.y);
    const radii = s.items(.radius);
    const ncells = dims.nx * dims.ny;
    sc.ball_cell.items.len = n;
    sc.cell_start.items.len = ncells + 1;
    const cells = sc.ball_cell.items;
    const starts = sc.cell_start.items;
    @memset(starts, 0);
    var rmax: f32 = 0;
    var alive: u32 = 0;
    for (0..n) |i| {
        if (!flags[i].alive) {
            cells[i] = std.math.maxInt(u32);
            continue;
        }
        alive += 1;
        rmax = @max(rmax, radii[i]);
        const c = dims.cy(ys[i]) * dims.nx + dims.cx(xs[i]);
        cells[i] = c;
        starts[c + 1] += 1;
    }
    if (alive < 2) return null;
    for (1..ncells + 1) |k| starts[k] += starts[k - 1];

    sc.hot.resize(w.gpa, alive) catch return null; // capacity reserved in tick(), never allocates
    var hot = sc.hot.slice();
    // Fill from the back so starts[c+1] ends up as the first slot of cell c.
    var i = n;
    while (i > 0) {
        i -= 1;
        const c = cells[i];
        if (c == std.math.maxInt(u32)) continue;
        starts[c + 1] -= 1;
        hot.set(starts[c + 1], .{
            .x = xs[i],
            .y = ys[i],
            .vx = s.items(.vx)[i],
            .vy = s.items(.vy)[i],
            .r = radii[i],
            .m = s.items(.mass)[i],
            .density = s.items(.density)[i],
            .temp = s.items(.temp)[i],
            .ang_vel = s.items(.ang_vel)[i],
            .level = s.items(.level)[i],
            .alive = true,
            .cell = c,
            .idx = @intCast(i),
        });
    }
    std.mem.copyForwards(u32, starts[0..ncells], starts[1 .. ncells + 1]);
    starts[ncells] = alive;
    const merge_growth = if (merging) std.math.log1p(cfg.max_radius) else 0;
    return .{ .dims = dims, .starts = starts, .cell = hot.items(.cell), .rmax = rmax + merge_growth };
}

fn scatterHot(w: *World, sc: *Scratch) void {
    const s = w.balls.slice();
    const hot = sc.hot.slice();
    for (0..sc.hot.len) |si| {
        const h = hot.get(si);
        const i = h.idx;
        s.items(.x)[i] = h.x;
        s.items(.y)[i] = h.y;
        s.items(.vx)[i] = h.vx;
        s.items(.vy)[i] = h.vy;
        s.items(.radius)[i] = h.r;
        s.items(.temp)[i] = h.temp;
        s.items(.ang_vel)[i] = h.ang_vel;
        s.items(.level)[i] = h.level;
        s.items(.flags)[i].alive = h.alive;
    }
}

const Counts = struct { tests: u32 = 0, merges: u32 = 0 };

const PairVisitor = struct {
    bl: Balls,
    si: usize,
    merging: bool,
    counts: *Counts,

    /// Filters a slot range V lanes at a time, then resolves the overlapping ones in order.
    /// The filter uses ball si's position from the start of the chunk; resolvePair re-tests
    /// with current positions, so a push can only make a pair wait until the next tick.
    inline fn range(v: PairVisitor, lo: usize, hi: usize) bool {
        const bl = v.bl;
        v.counts.tests += @intCast(hi - lo);
        var k = lo;
        while (k + V <= hi) : (k += V) {
            const dx = @as(F, bl.x[k..][0..V].*) - @as(F, @splat(bl.x[v.si]));
            const dy = @as(F, bl.y[k..][0..V].*) - @as(F, @splat(bl.y[v.si]));
            const rs = @as(F, bl.r[k..][0..V].*) + @as(F, @splat(bl.r[v.si]));
            const d2 = dx * dx + dy * dy;
            const alive: @Vector(V, bool) = bl.alive[k..][0..V].*;
            const hit = alive & (d2 < rs * rs) & (d2 > @as(F, @splat(0)));
            var bits: @Int(.unsigned, V) = @bitCast(hit);
            while (bits != 0) : (bits &= bits - 1) {
                if (!v.pair(k + @ctz(bits))) return false;
            }
        }
        while (k < hi) : (k += 1) {
            if (bl.alive[k] and !v.pair(k)) return false;
        }
        return true;
    }

    inline fn pair(v: PairVisitor, k: usize) bool {
        switch (resolvePair(v.bl, v.si, k, v.merging)) {
            .none => return true,
            .merged_into_i => {
                v.counts.merges += 1;
                return true;
            },
            .merged_into_j => {
                v.counts.merges += 1;
                return false;
            },
        }
    }
};

fn collideSlot(g: Grid, bl: Balls, si: usize, merging: bool, counts: *Counts) void {
    if (!bl.alive[si]) return;
    g.forward(bl, si, PairVisitor{ .bl = bl, .si = si, .merging = merging, .counts = counts });
}

/// Pairs are resolved sequentially (each push is seen by the next test, as in the Python).
/// For big counts the grid is cut into vertical strips at least one query reach wide and
/// run in three phases: strips s and s+3 touch disjoint balls, so a phase's strips can run
/// on different threads with no locks and the result does not depend on scheduling.
fn collide(w: *World, sc: *Scratch, dims: GridDims, merging: bool) void {
    const g = buildHot(w, sc, dims, merging) orelse return;
    const bl = Balls.of(&sc.hot);
    var counts: Counts = .{};

    const strip_w = @max(1, g.maxReachCells());
    const nstrips = std.math.divCeil(u32, dims.nx, strip_w) catch unreachable;
    const p = if (sc.hot.len >= collide_thread_min_balls and nstrips >= 6) pool.get() else null;
    if (p) |pl| {
        var job: CollideJob = .{ .grid = g, .bl = bl, .merging = merging, .strip_w = strip_w, .nstrips = nstrips };
        for (0..3) |phase| {
            job.phase = @intCast(phase);
            _ = pl.run(.{ .ctx = &job, .func = CollideJob.runBlock, .blocks = (nstrips - job.phase + 2) / 3 });
        }
        counts = .{ .tests = job.tests.load(.monotonic), .merges = job.merges.load(.monotonic) };
    } else {
        for (0..sc.hot.len) |si| collideSlot(g, bl, si, merging, &counts);
    }

    scatterHot(w, sc);
    w.events.merges += counts.merges;
    w.profile.pair_tests = counts.tests;
}

const CollideJob = struct {
    grid: Grid,
    bl: Balls,
    merging: bool,
    strip_w: u32,
    nstrips: u32,
    phase: u32 = 0,
    tests: std.atomic.Value(u32) = .init(0),
    merges: std.atomic.Value(u32) = .init(0),

    fn runBlock(ctx: *anyopaque, b: u32) void {
        const j: *CollideJob = @ptrCast(@alignCast(ctx));
        const g = j.grid;
        const strip = j.phase + 3 * b;
        const x0 = strip * j.strip_w;
        const x1 = @min(g.dims.nx, x0 + j.strip_w);
        var counts: Counts = .{};
        for (0..g.dims.ny) |cy| {
            const lo = g.starts[cy * g.dims.nx + x0];
            const hi = g.starts[cy * g.dims.nx + x1];
            for (lo..hi) |si| collideSlot(g, j.bl, si, j.merging, &counts);
        }
        _ = j.tests.fetchAdd(counts.tests, .monotonic);
        _ = j.merges.fetchAdd(counts.merges, .monotonic);
    }
};

const PairResult = enum { none, merged_into_i, merged_into_j };

/// utils.handle_ball_collisions for one pair, ball1 = i, ball2 = j.
fn resolvePair(bl: Balls, i: usize, j: usize, merging: bool) PairResult {
    const dx = bl.x[j] - bl.x[i];
    const dy = bl.y[j] - bl.y[i];
    const rsum = bl.r[i] + bl.r[j];
    const d2 = dx * dx + dy * dy;
    if (!(d2 > 0 and d2 < rsum * rsum)) return .none;
    const dist = @sqrt(d2);

    if (merging and bl.level[i] == bl.level[j] and bl.level[i] < cfg.max_level) {
        const keep, const gone, const result: PairResult = if (bl.r[i] > bl.r[j])
            .{ i, j, .merged_into_i }
        else
            .{ j, i, .merged_into_j };
        bl.alive[gone] = false;
        bl.level[keep] = @min(bl.level[keep] + 1, cfg.max_level);
        bl.temp[keep] = (bl.temp[keep] + bl.temp[gone]) * 0.75;
        // Mass is left unchanged, as in the Python.
        bl.r[keep] = @min(cfg.max_radius, bl.r[keep] + std.math.log1p(bl.r[gone]));
        return result;
    }

    const nx = dx / dist;
    const ny = dy / dist;
    const half = (rsum - dist) / 2;
    bl.x[i] -= nx * half;
    bl.y[i] -= ny * half;
    bl.x[j] += nx * half;
    bl.y[j] += ny * half;

    const rvx = bl.vx[i] - bl.vx[j];
    const rvy = bl.vy[i] - bl.vy[j];
    const rel_speed = rvx * nx + rvy * ny;
    const tan_speed = -rvx * ny + rvy * nx;
    bl.ang_vel[i] += tan_speed / bl.r[i] * 0.5;
    bl.ang_vel[j] -= tan_speed / bl.r[j] * 0.5;

    const mi = bl.m[i];
    const mj = bl.m[j];
    if (rel_speed > 0) {
        const impulse = 2 * rel_speed / (mi + mj);
        bl.vx[i] -= impulse * mj * nx;
        bl.vy[i] -= impulse * mj * ny;
        bl.vx[j] += impulse * mi * nx;
        bl.vy[j] += impulse * mi * ny;
    }

    const temp_diff = bl.temp[i] - bl.temp[j];
    const conductivity = (bl.density[i] + bl.density[j]) / 2 * rsum / 2 * cfg.base_conduct_mult;
    const impact_heat = @max(0, rel_speed) * cfg.impact_heat_mult;
    if (temp_diff > 0) {
        bl.temp[j] += (temp_diff * conductivity + impact_heat) / mj * 10;
    } else if (temp_diff < 0) {
        bl.temp[i] += (-temp_diff * conductivity + impact_heat) / mi * 10;
    } else {
        bl.temp[i] += impact_heat / mi * 10;
        bl.temp[j] += impact_heat / mj * 10;
    }
    return .none;
}

/// Ball.bounds against the current play area.
fn applyBounds(w: *World) void {
    const s = w.balls.slice();
    const b = w.bounds;
    const half_wall = cfg.wall / 2;
    for (s.items(.x), s.items(.y), s.items(.vx), s.items(.vy), s.items(.ang_vel), s.items(.radius), s.items(.retention), s.items(.gravity), s.items(.flags)) |*x, *y, *vx, *vy, *av, r, ret, g, f| {
        if (!f.alive) continue;
        const ceiling_y = b.y + r + half_wall;
        const floor_y = b.y + b.h - r - half_wall;
        const left_x = b.x + r + half_wall;
        const right_x = b.x + b.w - r - half_wall;

        if (y.* < ceiling_y) {
            y.* = ceiling_y;
            av.* -= vx.* / r;
            // Deviation: the Python used the floor formula here (vy*ret - g with vy < 0),
            // which is always negative, so the ceiling silently ate all upward speed.
            const rebound = -vy.* * ret;
            vy.* = if (rebound > cfg.bounce_stop) rebound else 0;
        }
        if (y.* > floor_y) {
            y.* = floor_y;
            av.* += vx.* / r;
            // Minus one tick of gravity, else the floor snap-up feeds energy and balls bounce forever.
            const rebound = vy.* * ret - g;
            vy.* = if (rebound > cfg.bounce_stop) -rebound else 0;
        }
        if (x.* < left_x) {
            x.* = left_x;
            av.* -= vx.* / r;
            if (vx.* < 0) {
                vx.* *= -ret;
                if (@abs(vx.*) < cfg.bounce_stop) vx.* = 0;
            }
        } else if (x.* > right_x) {
            x.* = right_x;
            av.* += vx.* / r;
            if (vx.* > 0) {
                vx.* *= -ret;
                if (@abs(vx.*) < cfg.bounce_stop) vx.* = 0;
            }
        }
    }
}

// ---------------------------------------------------------------- bookkeeping

fn randomExplosion(w: *World) void {
    const r = w.prng.random();
    if (r.float(f32) >= cfg.random_explosion_chance) return;
    const b = w.bounds;
    const m = cfg.wall + 10;
    const x = b.x + m + r.float(f32) * @max(0, b.w - 2 * m);
    const y = b.y + m + r.float(f32) * @max(0, b.h - 2 * m);
    const size: f32 = @floatFromInt(r.intRangeAtMost(u8, cfg.random_explosion_size_min, cfg.random_explosion_size_max));
    effects.explode(w, x, y, size) catch return;
    w.events.explosions += 1;
}

fn removeDead(w: *World) void {
    const flags = w.balls.items(.flags);
    var i = w.balls.len;
    while (i > 0) {
        i -= 1;
        if (!flags[i].alive) w.balls.swapRemove(i);
    }
}

fn pushTrails(w: *World) void {
    const s = w.balls.slice();
    for (s.items(.trail), s.items(.trail_head), s.items(.trail_len), s.items(.x), s.items(.y)) |*trail, *head, *len, x, y| {
        trail[head.*] = .{ .x = x, .y = y };
        head.* = @intCast((@as(usize, head.*) + 1) % cfg.max_trail);
        if (len.* < cfg.max_trail) len.* += 1;
    }
}

/// Returns the summed ball area for sizing the play area.
fn updateLevelAndEvents(w: *World) f32 {
    const s = w.balls.slice();
    var max_level: u8 = 0;
    var area: f32 = 0;
    for (s.items(.level), s.items(.flags), s.items(.radius)) |lvl, f, r| {
        max_level = @max(max_level, lvl);
        if (f.burning) w.events.any_burning = true;
        area += std.math.pi * r * r;
    }
    if (w.balls.len > 0) w.current_level = @max(cfg.start_level, max_level);
    w.events.won = max_level >= cfg.max_level;
    return area;
}

/// Play area that keeps `fill_area` at or below cfg.max_fill of the total, never smaller
/// than the base window, with the base window's aspect ratio and centre.
pub fn targetBounds(fill_area: f32) Rect {
    const area = @max(cfg.base_w * cfg.base_h, fill_area / cfg.max_fill);
    const aspect = cfg.base_w / cfg.base_h;
    const tw = @sqrt(area * aspect);
    const th = tw / aspect;
    return .{ .x = (cfg.base_w - tw) / 2, .y = (cfg.base_h - th) / 2, .w = tw, .h = th };
}

pub fn easeBounds(cur: Rect, fill_area: f32) Rect {
    const t = targetBounds(fill_area);
    const rate = if (t.w > cur.w) cfg.area_grow_rate else cfg.area_shrink_rate;
    var w = cur.w + (t.w - cur.w) * rate;
    if (@abs(t.w - w) < 0.01) w = t.w;
    const h = w * (cfg.base_h / cfg.base_w);
    return .{ .x = (cfg.base_w - w) / 2, .y = (cfg.base_h - h) / 2, .w = w, .h = h };
}

/// Jumps the play area straight to its target size (no easing), e.g. before a benchmark.
pub fn snapBounds(w: *World) void {
    var area: f32 = 0;
    for (w.balls.items(.radius)) |r| area += std.math.pi * r * r;
    w.bounds = targetBounds(area);
    w.prev_bounds = w.bounds;
}

// ---------------------------------------------------------------- input-driven actions

/// Spawns `count` balls on a sunflower spiral around (x, y), one spawn diameter apart,
/// so big bursts start mostly separated instead of all on one point.
pub fn spawnBurst(w: *World, x: f32, y: f32, count: u32) !void {
    if (count == 0) return;
    try w.balls.ensureUnusedCapacity(w.gpa, count);
    if (count == 1) return w.spawnBall(x, y);
    const n: f32 = @floatFromInt(count);
    const spacing = 2 * cfg.spawn_radius_max;
    const disc_r = spacing * @sqrt(n / std.math.pi);
    const golden_angle = std.math.pi * (3 - @sqrt(5.0));
    for (0..count) |k| {
        const fk: f32 = @floatFromInt(k);
        const rr = disc_r * @sqrt((fk + 0.5) / n);
        const th = fk * golden_angle;
        try w.spawnBall(x + rr * @cos(th), y + rr * @sin(th));
    }
}

/// The ball whose disc contains `p`; the nearest centre wins when several do.
pub fn pickBall(w: *const World, p: Vec2) ?usize {
    const s = w.balls.slice();
    var best: ?usize = null;
    var best_d2: f32 = std.math.inf(f32);
    for (s.items(.x), s.items(.y), s.items(.radius), s.items(.flags), 0..) |x, y, r, f, i| {
        if (!f.alive) continue;
        const d2 = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
        if (d2 <= r * r and d2 < best_d2) {
            best = i;
            best_d2 = d2;
        }
    }
    return best;
}

/// Ball.update_temp: a click cools by 150 and anything below ignition snaps to room temp.
fn updateTemp(temp: *f32, flags: *world.BallFlags) void {
    if (temp.* > cfg.room_temp) temp.* = @max(cfg.room_temp, temp.* - 150);
    if (temp.* >= cfg.ignition_temp) {
        flags.burning = true;
    } else {
        temp.* = cfg.room_temp;
        flags.burning = false;
    }
}

/// Ball.onClick: cool down, then grab.
pub fn clickBall(w: *World, i: usize) void {
    const s = w.balls.slice();
    const f = &s.items(.flags)[i];
    updateTemp(&s.items(.temp)[i], f);
    f.grabbed = true;
    s.items(.grab_prev)[i] = .{ .x = s.items(.x)[i], .y = s.items(.y)[i] };
}

/// Ball.releaseGrab for every held ball: throw with the last follow step's motion.
pub fn releaseGrabs(w: *World) void {
    const s = w.balls.slice();
    for (s.items(.flags), s.items(.x), s.items(.y), s.items(.vx), s.items(.vy), s.items(.grab_prev)) |*f, x, y, *vx, *vy, gp| {
        if (!f.grabbed) continue;
        vx.* += x - gp.x;
        vy.* += y - gp.y;
        f.grabbed = false;
        f.thrown = true;
    }
}

/// main.onRightClick: heat to flame temperature, then update_temp (which leaves it burning at 250).
pub fn ignite(w: *World, i: usize) void {
    const s = w.balls.slice();
    s.items(.temp)[i] = cfg.flame_temp;
    updateTemp(&s.items(.temp)[i], &s.items(.flags)[i]);
}

pub fn explodeAt(w: *World, x: f32, y: f32, ball_size: f32) !void {
    try effects.explode(w, x, y, ball_size);
    w.pending_events.explosions += 1;
}

pub fn pointInPolygon(p: Vec2, poly: []const Vec2) bool {
    var inside = false;
    var j = poly.len - 1;
    for (poly, 0..) |a, i| {
        const b = poly[j];
        if ((a.y > p.y) != (b.y > p.y)) {
            const xi = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (p.x < xi) inside = !inside;
        }
        j = i;
    }
    return inside;
}

/// Every ball whose centre is inside the polygon explodes and is removed immediately.
/// All lassoed balls are marked dead before any blast is applied; the Python applied
/// blasts in list order, which only differed for balls that were about to die anyway.
pub fn lassoExplode(w: *World, polygon: []const Vec2) !void {
    if (polygon.len < 3) return;
    var lo: Vec2 = .{ .x = std.math.inf(f32), .y = std.math.inf(f32) };
    var hi: Vec2 = .{ .x = -std.math.inf(f32), .y = -std.math.inf(f32) };
    for (polygon) |p| {
        lo = .{ .x = @min(lo.x, p.x), .y = @min(lo.y, p.y) };
        hi = .{ .x = @max(hi.x, p.x), .y = @max(hi.y, p.y) };
    }
    const s = w.balls.slice();
    const xs = s.items(.x);
    const ys = s.items(.y);
    const flags = s.items(.flags);
    var hit: usize = 0;
    for (xs, ys, flags) |x, y, *f| {
        if (!f.alive or x < lo.x or x > hi.x or y < lo.y or y > hi.y) continue;
        if (pointInPolygon(.{ .x = x, .y = y }, polygon)) {
            f.alive = false;
            hit += 1;
        }
    }
    if (hit == 0) return;
    try w.explosions.ensureUnusedCapacity(w.gpa, hit);
    for (xs, ys, s.items(.radius), flags) |x, y, r, f| {
        if (f.alive) continue;
        effects.explode(w, x, y, r) catch unreachable;
    }
    w.pending_events.explosions += @intCast(hit);
    removeDead(w);
}

// ---------------------------------------------------------------- tests

const testing = std.testing;

fn testWorld() World {
    return World.init(testing.allocator, 1234);
}

fn addBall(w: *World, x: f32, y: f32, r: f32, level: u8) !void {
    try w.spawnBall(x, y);
    const i = w.balls.len - 1;
    const s = w.balls.slice();
    s.items(.radius)[i] = r;
    s.items(.level)[i] = level;
    s.items(.vx)[i] = 0;
    s.items(.vy)[i] = 0;
}

const Pair = struct { a: u32, b: u32 };

fn lessPair(_: void, p: Pair, q: Pair) bool {
    return p.a < q.a or (p.a == q.a and p.b < q.b);
}

const PairRecorder = struct {
    bl: Balls,
    idx: []const u32,
    si: usize,
    out: *std.ArrayList(Pair),

    fn range(v: PairRecorder, lo: usize, hi: usize) bool {
        for (lo..hi) |k| {
            const dx = v.bl.x[k] - v.bl.x[v.si];
            const dy = v.bl.y[k] - v.bl.y[v.si];
            const rs = v.bl.r[k] + v.bl.r[v.si];
            if (dx * dx + dy * dy < rs * rs) {
                const a = v.idx[v.si];
                const b = v.idx[k];
                v.out.append(testing.allocator, .{ .a = @min(a, b), .b = @max(a, b) }) catch unreachable;
            }
        }
        return true;
    }
};

test "grid broad-phase finds the same overlapping pairs as brute force" {
    var w = testWorld();
    defer w.deinit();
    const r = w.prng.random();
    // Dense clump plus a few max-size balls so the widened query is exercised.
    for (0..800) |k| {
        const rad: f32 = if (k % 50 == 0) cfg.max_radius else 25 + r.float(f32) * 20;
        try addBall(&w, 300 + r.float(f32) * 700, 100 + r.float(f32) * 500, rad, 1);
    }
    // Some outside the play area, as a grabbed ball dragged off-screen would be.
    try addBall(&w, -500, 300, 40, 1);
    try addBall(&w, -480, 320, 40, 1);
    const sc = try scratchOf(&w);
    const dims = GridDims.of(w.bounds);
    try sc.reserve(w.gpa, w.balls.len, dims.nx * dims.ny);
    const g = buildHot(&w, sc, dims, false).?;
    const bl = Balls.of(&sc.hot);

    var brute: std.ArrayList(Pair) = .empty;
    defer brute.deinit(testing.allocator);
    var grid: std.ArrayList(Pair) = .empty;
    defer grid.deinit(testing.allocator);
    const s = w.balls.slice();
    const n = w.balls.len;
    for (0..n) |i| for (i + 1..n) |j| {
        const dx = s.items(.x)[j] - s.items(.x)[i];
        const dy = s.items(.y)[j] - s.items(.y)[i];
        const rs = s.items(.radius)[i] + s.items(.radius)[j];
        if (dx * dx + dy * dy < rs * rs) try brute.append(testing.allocator, .{ .a = @intCast(i), .b = @intCast(j) });
    };
    for (0..sc.hot.len) |si| g.forward(bl, si, PairRecorder{ .bl = bl, .idx = sc.hot.items(.idx), .si = si, .out = &grid });
    std.mem.sort(Pair, grid.items, {}, lessPair);
    try testing.expect(brute.items.len > 100);
    try testing.expectEqualSlices(Pair, brute.items, grid.items);
}

test "parallel strip collisions match the sequential pass" {
    var results: [2][]f32 = undefined;
    for (0..2) |run| {
        var w = testWorld();
        defer w.deinit();
        const r = w.prng.random();
        for (0..3000) |_| try addBall(&w, -800 + r.float(f32) * 2600, -400 + r.float(f32) * 1500, 25 + r.float(f32) * 20, 1);
        snapBounds(&w);
        const sc = try scratchOf(&w);
        const dims = GridDims.of(w.bounds);
        try sc.reserve(w.gpa, w.balls.len, dims.nx * dims.ny);
        if (run == 0) {
            const g = buildHot(&w, sc, dims, false).?;
            const bl = Balls.of(&sc.hot);
            // Same per-strip, per-phase order the threaded path uses.
            const strip_w = @max(1, g.maxReachCells());
            const nstrips = std.math.divCeil(u32, dims.nx, strip_w) catch unreachable;
            var job: CollideJob = .{ .grid = g, .bl = bl, .merging = false, .strip_w = strip_w, .nstrips = nstrips };
            for (0..3) |phase| {
                job.phase = @intCast(phase);
                for (0..(nstrips - job.phase + 2) / 3) |b| CollideJob.runBlock(&job, @intCast(b));
            }
            scatterHot(&w, sc);
        } else {
            collide(&w, sc, dims, false);
        }
        results[run] = try testing.allocator.dupe(f32, w.balls.items(.x));
    }
    defer for (results) |x| testing.allocator.free(x);
    try testing.expectEqualSlices(f32, results[0], results[1]);
}

test "two equal-level touching balls merge into one of level+1" {
    var w = testWorld();
    defer w.deinit();
    try addBall(&w, 500, 300, 30, 5);
    try addBall(&w, 540, 300, 35, 5);
    tick(&w, .{ .mutual_gravity = false, .random_explosions = false });
    try testing.expectEqual(@as(usize, 1), w.balls.len);
    try testing.expectEqual(@as(u8, 6), w.balls.items(.level)[0]);
    try testing.expectApproxEqAbs(35 + std.math.log1p(@as(f32, 30)), w.balls.items(.radius)[0], 1e-4);
    try testing.expectEqual(@as(u32, 1), w.events.merges);
}

test "a ball stays inside bounds" {
    var w = testWorld();
    defer w.deinit();
    try addBall(&w, 500, 300, 40, 1);
    w.balls.items(.vx)[0] = 900;
    w.balls.items(.vy)[0] = -700;
    for (0..200) |_| {
        tick(&w, .{ .mutual_gravity = false, .random_explosions = false });
        const b = w.bounds;
        const x = w.balls.items(.x)[0];
        const y = w.balls.items(.y)[0];
        try testing.expect(x - 40 >= b.x and x + 40 <= b.x + b.w);
        try testing.expect(y - 40 >= b.y and y + 40 <= b.y + b.h);
    }
}

test "lasso explodes exactly the balls inside the polygon" {
    var w = testWorld();
    defer w.deinit();
    try addBall(&w, 100, 100, 30, 1);
    try addBall(&w, 300, 300, 30, 2); // inside
    try addBall(&w, 600, 600, 30, 3);
    const tri = [_]Vec2{ .{ .x = 200, .y = 200 }, .{ .x = 450, .y = 250 }, .{ .x = 250, .y = 450 } };
    try testing.expect(pointInPolygon(.{ .x = 300, .y = 300 }, &tri));
    try testing.expect(!pointInPolygon(.{ .x = 100, .y = 100 }, &tri));
    try lassoExplode(&w, &tri);
    try testing.expectEqual(@as(usize, 2), w.balls.len);
    for (w.balls.items(.level)) |lvl| try testing.expect(lvl != 2);
    try testing.expectEqual(@as(usize, 1), w.explosions.items.len);
    try testing.expectEqual(@as(f32, 300), w.explosions.items[0].size);
    try testing.expectEqual(@as(u32, 1), w.pending_events.explosions);
}

test "gravity kernel matches scalar reference" {
    const n = 37;
    var gx: [padded(n)]f32 = @splat(0);
    var gy: [padded(n)]f32 = @splat(0);
    var gm: [padded(n)]f32 = @splat(0);
    var ax: [padded(n)]f32 = undefined;
    var ay: [padded(n)]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    for (0..n) |i| {
        gx[i] = r.float(f32) * 1000;
        gy[i] = r.float(f32) * 700;
        gm[i] = 0.3 * (50 + r.float(f32) * 300);
    }
    const job: GravityJob = .{ .gx = &gx, .gy = &gy, .gm = &gm, .ax = &ax, .ay = &ay, .n = n };
    job.rows(0, n);
    for (0..n) |i| {
        var sx: f64 = 0;
        var sy: f64 = 0;
        for (0..n) |j| {
            if (i == j) continue;
            const dx: f64 = gx[j] - gx[i];
            const dy: f64 = gy[j] - gy[i];
            const f = gm[j] / (dx * dx + dy * dy + 0.001);
            sx += f * dx;
            sy += f * dy;
        }
        try testing.expectApproxEqRel(@as(f32, @floatCast(sx)), ax[i], 1e-3);
        try testing.expectApproxEqRel(@as(f32, @floatCast(sy)), ay[i], 1e-3);
    }
}

test "smoke: every feature on for a few seconds stays finite and in bounds" {
    var w = testWorld();
    defer w.deinit();
    try spawnBurst(&w, 500, 300, 400);
    try spawnBurst(&w, 200, 200, 1);
    try testing.expectEqual(@as(usize, 401), w.balls.len);
    ignite(&w, 0);
    clickBall(&w, 5);
    var in: TickInput = .{ .mouse = .{ .x = 300, .y = 300 } };
    for (0..900) |t| {
        if (t == 100) releaseGrabs(&w);
        if (t == 200) try explodeAt(&w, 500, 300, 40);
        if (t == 300) in.mutual_gravity = false;
        if (t == 400) in.attract = true;
        if (t == 420) in.attract = false;
        if (t == 500) {
            const b = w.bounds;
            try lassoExplode(&w, &.{ .{ .x = b.x, .y = b.y }, .{ .x = b.x + b.w / 2, .y = b.y }, .{ .x = b.x + b.w / 2, .y = b.y + b.h }, .{ .x = b.x, .y = b.y + b.h } });
        }
        if (t % 50 == 0 and w.balls.len > 0) ignite(&w, t % w.balls.len);
        tick(&w, in);
        const b = w.bounds;
        for (w.balls.items(.x), w.balls.items(.y), w.balls.items(.radius), w.balls.items(.flags)) |x, y, r, f| {
            try testing.expect(f.alive);
            try testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
            if (!f.grabbed) try testing.expect(x - r >= b.x - 1e-3 and x + r <= b.x + b.w + 1e-3);
        }
    }
    try testing.expect(w.tick == 900);
    try testing.expect(w.particles.len <= cfg.max_particles and w.ashes.len <= cfg.max_ashes);
}
