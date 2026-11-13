//! Tests of the 3D tree kick (k3/walk.zig) on trees built by ref3.build, plus two measurements
//! that only run when asked, because they take minutes:
//!   WALK3_BENCH=100000,1000000,4000000,16000000  walk times per size and workgroup size
//!   WALK3_ENERGY=1                               100k Plummer, 200 ticks, tree rebuilt every tick
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const ic3 = @import("ic3");
const sim = @import("sim.zig");
const ref3 = @import("ref3.zig");
const walk3 = @import("walk3.zig");

const testing = std.testing;

fn device() !*c.SDL_GPUDevice {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    return sim.createDevice();
}

const Cloud = struct {
    pos: []ref3.Pos3,
    mass: []f32,
    box: f64,
    fn deinit(cl: Cloud, gpa: std.mem.Allocator) void {
        gpa.free(cl.pos);
        gpa.free(cl.mass);
    }
};

/// The ref3 test cloud (equal-mass Plummer, scale radius 1, cut at 10), but in a box of side 40:
/// the kernel takes wrapped u32 differences, so an open domain must stay in the middle half.
fn plummerCloud(gpa: std.mem.Allocator, n: usize, seed: u64) !Cloud {
    const half = 20.0;
    const raw = try ref3.plummer.positions(gpa, n, seed, 10);
    defer gpa.free(raw);
    const pos = try gpa.alloc(ref3.Pos3, n);
    errdefer gpa.free(pos);
    const mass = try gpa.alloc(f32, n);
    for (raw, pos, mass) |r, *q, *m| {
        q.* = .{ .x = ref3.plummer.quantise(r[0], half), .y = ref3.plummer.quantise(r[1], half), .z = ref3.plummer.quantise(r[2], half) };
        m.* = 1.0 / @as(f32, @floatFromInt(n));
    }
    return .{ .pos = pos, .mass = mass, .box = 2 * half };
}

/// A walk pipeline plus one tree's worth of GPU buffers.
const Rig = struct {
    dev: *c.SDL_GPUDevice,
    w: walk3.Walk3,
    t: walk3.GpuTree,
    vel: walk3.Buffer,
    amag: walk3.Buffer,
    pot: walk3.Buffer,
    n: u32,
    box_scale: f32,

    fn init(dev: *c.SDL_GPUDevice, tree: ref3.Tree) !Rig {
        const n: u32 = @intCast(tree.pos.len);
        var r: Rig = .{ .dev = dev, .w = try walk3.Walk3.init(dev), .t = undefined, .vel = undefined, .amag = undefined, .pot = undefined, .n = n, .box_scale = @floatCast(tree.box / 0x1p32) };
        r.t = try walk3.GpuTree.init(dev, n, @intCast(tree.nodes.len));
        r.vel = try walk3.makeBuffer(dev, @as(u64, n) * 12);
        r.amag = try walk3.makeBuffer(dev, @as(u64, n) * 4);
        r.pot = try walk3.makeBuffer(dev, @as(u64, n) * 4);
        try r.t.upload(tree);
        return r;
    }

    fn deinit(r: *Rig) void {
        for ([_]walk3.Buffer{ r.vel, r.amag, r.pot }) |b| c.SDL_ReleaseGPUBuffer(r.dev, b);
        r.t.deinit();
        r.w.deinit();
    }

    fn run(r: *Rig, eps: f32, kick: f32, flags: u32) !void {
        const cmd = c.SDL_AcquireGPUCommandBuffer(r.dev) orelse return error.GpuSubmit;
        const last = try r.w.record(cmd, r.t.bindings(r.vel, r.amag, r.pot), .{ .n = r.n, .g = 1, .eps = eps, .kick = kick, .box_scale = r.box_scale, .flags = flags });
        try walk3.submitWait(r.dev, last);
    }

    const Out = struct {
        acc: []L.Vel3,
        amag: []f32,
        pot: []f32,
        fn deinit(o: Out, gpa: std.mem.Allocator) void {
            gpa.free(o.acc);
            gpa.free(o.amag);
            gpa.free(o.pot);
        }
    };

    /// Accelerations, |a| and potentials of the sorted bodies.
    fn forces(r: *Rig, gpa: std.mem.Allocator, eps: f32) !Out {
        try r.run(eps, 0, L.sim3_flag_store_accel | L.sim3_flag_potential);
        const o: Out = .{ .acc = try gpa.alloc(L.Vel3, r.n), .amag = try gpa.alloc(f32, r.n), .pot = try gpa.alloc(f32, r.n) };
        try walk3.downloadBytes(r.dev, r.vel, std.mem.sliceAsBytes(o.acc));
        try walk3.downloadBytes(r.dev, r.amag, std.mem.sliceAsBytes(o.amag));
        try walk3.downloadBytes(r.dev, r.pot, std.mem.sliceAsBytes(o.pot));
        return o;
    }
};

fn relErr(g: L.Vel3, ref: [3]f64) f64 {
    const d = [3]f64{ g.x - ref[0], g.y - ref[1], g.z - ref[2] };
    return @sqrt((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (ref[0] * ref[0] + ref[1] * ref[1] + ref[2] * ref[2]));
}

const ErrSummary = struct { rms: f64, p99: f64, max: f64 };

fn summarise(errs: []f64) ErrSummary {
    var sum2: f64 = 0;
    for (errs) |e| sum2 += e * e;
    std.mem.sort(f64, errs, {}, std.sort.asc(f64));
    return .{ .rms = @sqrt(sum2 / @as(f64, @floatFromInt(errs.len))), .p99 = errs[errs.len * 99 / 100], .max = errs[errs.len - 1] };
}

test "walk3: 10k Plummer at theta 0.6 matches direct sum and the ref3 walk, reruns are bit-identical" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = testing.allocator;
    const n = 10_000;
    const eps: f32 = 1e-4;
    const cloud = try plummerCloud(gpa, n, 42);
    defer cloud.deinit(gpa);
    const tree = try ref3.build(gpa, cloud.pos, cloud.mass, .{ .box = cloud.box, .theta = 0.6 });
    defer tree.deinit(gpa);
    var rig = try Rig.init(dev, tree);
    defer rig.deinit();

    const out = try rig.forces(gpa, eps);
    defer out.deinit(gpa);

    // Against f64 direct summation (original order; sorted body s is original body perm[s]).
    const direct = try gpa.alloc(ref3.Accel, n);
    defer gpa.free(direct);
    ref3.directSum(cloud.pos, cloud.mass, cloud.box, 1, eps, direct);
    const errs = try gpa.alloc(f64, n);
    defer gpa.free(errs);
    for (errs, out.acc, tree.perm) |*e, a, o| e.* = relErr(a, direct[o].a);
    const vs_direct = summarise(errs);

    // Against the f64 ref3 walk of the same tree: only f32 arithmetic separates the two.
    var pot_err: f64 = 0;
    var amag_err: f64 = 0;
    for (errs, out.acc, out.amag, out.pot, 0..) |*e, a, am, p, s| {
        const ref = tree.walk(tree.pos[s], @intCast(s), 1, eps, null);
        e.* = relErr(a, ref.a);
        pot_err = @max(pot_err, @abs(p - ref.pot) / @abs(ref.pot));
        const m = @sqrt(ref.a[0] * ref.a[0] + ref.a[1] * ref.a[1] + ref.a[2] * ref.a[2]);
        amag_err = @max(amag_err, @abs(am - m) / m);
    }
    const vs_ref = summarise(errs);
    std.debug.print("walk3 10k theta 0.6: vs direct rms {e:.3} p99 {e:.3} max {e:.3}; vs ref3 walk rms {e:.3} p99 {e:.3} max {e:.3}, pot max {e:.3}, |a| max {e:.3}\n", .{
        vs_direct.rms, vs_direct.p99, vs_direct.max, vs_ref.rms, vs_ref.p99, vs_ref.max, pot_err, amag_err,
    });
    try testing.expect(vs_direct.rms <= 1.2e-3);
    try testing.expect(vs_direct.p99 <= 5e-3);
    try testing.expect(vs_ref.rms <= 1e-5);
    try testing.expect(vs_ref.p99 <= 2e-5);
    try testing.expect(pot_err <= 1e-5);
    try testing.expect(amag_err <= 1e-4);

    // Reruns, other workgroup sizes and other chunkings give the same bits.
    for (walk3.group_sizes) |gs| for ([_]u32{ walk3.chunk_bodies, 999 }) |chunk| {
        rig.w.group_size = gs;
        rig.w.chunk = chunk;
        const again = try rig.forces(gpa, eps);
        defer again.deinit(gpa);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(out.acc), std.mem.sliceAsBytes(again.acc));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(out.amag), std.mem.sliceAsBytes(again.amag));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(out.pot), std.mem.sliceAsBytes(again.pot));
    };

    // Kick mode adds kick * a to the stored velocity.
    const v0 = try gpa.alloc(L.Vel3, n);
    defer gpa.free(v0);
    for (v0, 0..) |*v, i| {
        const f: f32 = @floatFromInt(i);
        v.* = .{ .x = f, .y = -f, .z = 0.5 };
    }
    try walk3.uploadBytes(dev, rig.vel, std.mem.sliceAsBytes(v0));
    const kick: f32 = 0.25;
    try rig.run(eps, kick, 0);
    const v1 = try gpa.alloc(L.Vel3, n);
    defer gpa.free(v1);
    try walk3.downloadBytes(dev, rig.vel, std.mem.sliceAsBytes(v1));
    for (v0, v1, out.acc) |a, b, g| {
        try testing.expectApproxEqAbs(a.x + kick * g.x, b.x, 1e-6 * (@abs(a.x) + 1));
        try testing.expectApproxEqAbs(a.y + kick * g.y, b.y, 1e-6 * (@abs(a.y) + 1));
        try testing.expectApproxEqAbs(a.z + kick * g.z, b.z, 1e-6 * (@abs(a.z) + 1));
    }
}

fn envSet(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

/// Best-of-`reps` wall time (ms) of one full kick at the rig's group size, after a warm-up.
fn timeWalk(rig: *Rig, eps: f32, reps: u32) !f64 {
    var best = std.math.inf(f64);
    for (0..reps + 1) |k| {
        const t0 = c.SDL_GetPerformanceCounter();
        try rig.run(eps, 0, 0);
        const ms = sim.msSince(t0);
        if (k > 0) best = @min(best, ms);
    }
    return best;
}

test "walk3: timing table (WALK3_BENCH=sizes)" {
    const list = envSet("WALK3_BENCH") orelse return error.SkipZigTest;
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.heap.smp_allocator;
    const eps: f32 = 1e-4;
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |tok| {
        const n = try std.fmt.parseInt(u32, tok, 10);
        const cloud = try plummerCloud(gpa, n, 42);
        defer cloud.deinit(gpa);
        const tb = c.SDL_GetPerformanceCounter();
        const tree = try ref3.build(gpa, cloud.pos, cloud.mass, .{ .box = cloud.box, .theta = 0.6 });
        defer tree.deinit(gpa);
        const build_ms = sim.msSince(tb);
        var rig = try Rig.init(dev, tree);
        defer rig.deinit();

        // Per-body work from the f64 walk on a sample (its decisions match the GPU's).
        var st: ref3.Stats = .{};
        const samples = 2000;
        var prng = std.Random.DefaultPrng.init(5);
        var worst: f64 = 0;
        const out = try rig.forces(gpa, eps);
        defer out.deinit(gpa);
        for (0..samples) |_| {
            const s = prng.random().uintLessThan(u32, n);
            const ref = tree.walk(tree.pos[s], s, 1, eps, &st);
            worst = @max(worst, relErr(out.acc[s], ref.a));
        }
        const fs: f64 = samples;
        const visits = @as(f64, @floatFromInt(st.visits)) / fs;
        const inter = @as(f64, @floatFromInt(st.cells + st.pairs)) / fs;
        std.debug.print("n={d}: CPU build {d:.0} ms, nodes/N {d:.3}; per body visits {d:.0} cells {d:.0} pairs {d:.0}; max rel diff vs ref3 walk (sample) {e:.2}\n", .{
            n, build_ms, @as(f64, @floatFromInt(tree.nodes.len)) / @as(f64, @floatFromInt(n)), visits, @as(f64, @floatFromInt(st.cells)) / fs, @as(f64, @floatFromInt(st.pairs)) / fs, worst,
        });
        for (walk3.group_sizes) |gs| {
            rig.w.group_size = gs;
            const ms = try timeWalk(&rig, eps, 5);
            const nf: f64 = @floatFromInt(n);
            std.debug.print("  group {d:>3}: {d:>9.2} ms (min of 5)  {d:.2} G node visits/s  {d:.2} G interactions/s\n", .{
                gs, ms, nf * visits / ms * 1e-6, nf * inter / ms * 1e-6,
            });
        }
    }
}

test "walk3: 100k Plummer energy over 200 ticks with a CPU tree per tick (WALK3_ENERGY=1)" {
    _ = envSet("WALK3_ENERGY") orelse return error.SkipZigTest;
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const gpa = std.heap.smp_allocator;
    const n: u32 = if (envSet("WALK3_ENERGY_N")) |s| try std.fmt.parseInt(u32, s, 10) else 100_000;
    const ticks: u32 = if (envSet("WALK3_ENERGY_TICKS")) |s| try std.fmt.parseInt(u32, s, 10) else 200;
    const eps: f32 = if (envSet("WALK3_ENERGY_EPS")) |s| try std.fmt.parseFloat(f32, s) else 0.01;
    const dt: f32 = 0.01;
    const theta = 0.6;
    // The sim.s seed and eps, so `zig build sim -- --dim 3 --n 100000 --ticks 200` is a direct-sum
    // baseline for the same bodies. The box is computed here (an origin-centred cube, 4x the extent)
    // rather than taken from ic3, whose Box is changing on gpu-physics.
    var prng = std.Random.DefaultPrng.init(1);
    const bodies = try ic3.plummer(gpa, prng.random(), n, .{});
    defer gpa.free(bodies);
    var extent: f64 = 0;
    for (bodies) |b| extent = @max(extent, @abs(b.x), @abs(b.y), @abs(b.z));
    const box_size = 8 * extent;
    const pos = try gpa.alloc(ref3.Pos3, n);
    defer gpa.free(pos);
    const vel = try gpa.alloc(L.Vel3, n);
    defer gpa.free(vel);
    const mass = try gpa.alloc(f32, n);
    defer gpa.free(mass);
    const quanta = struct {
        fn f(x: f64, size: f64) u32 {
            const q = std.math.ldexp(x / size + 0.5, 32);
            return @intFromFloat(std.math.clamp(@round(q), 0x1p30, 3 * 0x1p30 - 1));
        }
    }.f;
    for (bodies, pos, vel, mass) |b, *p, *v, *m| {
        p.* = .{ .x = quanta(b.x, box_size), .y = quanta(b.y, box_size), .z = quanta(b.z, box_size) };
        v.* = .{ .x = @floatCast(b.vx), .y = @floatCast(b.vy), .z = @floatCast(b.vz) };
        m.* = @floatCast(b.m);
    }
    const sorted_vel = try gpa.alloc(L.Vel3, n);
    defer gpa.free(sorted_vel);
    const sorted_pot = try gpa.alloc(f32, n);
    defer gpa.free(sorted_pot);

    var w = try walk3.Walk3.init(dev);
    defer w.deinit();
    var gt = try walk3.GpuTree.init(dev, n, n / 4);
    defer gt.deinit();
    const vbuf = try walk3.makeBuffer(dev, @as(u64, n) * 12);
    defer c.SDL_ReleaseGPUBuffer(dev, vbuf);
    const abuf = try walk3.makeBuffer(dev, @as(u64, n) * 4);
    defer c.SDL_ReleaseGPUBuffer(dev, abuf);
    const pbuf = try walk3.makeBuffer(dev, @as(u64, n) * 4);
    defer c.SDL_ReleaseGPUBuffer(dev, pbuf);
    const box_scale: f32 = @floatCast(box_size / 0x1p32);
    const drift_q: f32 = @floatCast(@as(f64, dt) * 0x1p32 / box_size);

    const Energy = struct { k: f64, w: f64 };
    var e0: Energy = undefined;
    var walk_ms: f64 = 0;
    var build_ms: f64 = 0;
    // Kick-drift-kick; tick 0 is the opening half kick, the last tick ends with a half kick.
    for (0..ticks + 1) |t| {
        if (t > 0) for (pos, vel) |*p, v| {
            p.* = .{ .x = drift(p.x, v.x * drift_q), .y = drift(p.y, v.y * drift_q), .z = drift(p.z, v.z * drift_q) };
        };
        const tb = c.SDL_GetPerformanceCounter();
        const tree = try ref3.build(gpa, pos, mass, .{ .box = box_size, .theta = theta });
        defer tree.deinit(gpa);
        build_ms += sim.msSince(tb);
        try gt.upload(tree);
        for (sorted_vel, tree.perm) |*sv, o| sv.* = vel[o];
        try walk3.uploadBytes(dev, vbuf, std.mem.sliceAsBytes(sorted_vel));
        const edge = t == 0 or t == ticks;
        const k0: f64 = if (t == 0) kinetic(vel, mass) else 0;
        const cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
        const tw = c.SDL_GetPerformanceCounter();
        const last = try w.record(cmd, gt.bindings(vbuf, abuf, pbuf), .{
            .n = n,
            .g = 1,
            .eps = eps,
            .kick = if (edge) 0.5 * dt else dt,
            .box_scale = box_scale,
            .flags = if (edge) L.sim3_flag_potential else 0,
        });
        try walk3.submitWait(dev, last);
        walk_ms += sim.msSince(tw);
        try walk3.downloadBytes(dev, vbuf, std.mem.sliceAsBytes(sorted_vel));
        for (sorted_vel, tree.perm) |sv, o| vel[o] = sv;
        if (edge) {
            try walk3.downloadBytes(dev, pbuf, std.mem.sliceAsBytes(sorted_pot));
            var pe: f64 = 0;
            for (sorted_pot, tree.mass) |p, m| pe += 0.5 * @as(f64, m) * p;
            const k = if (t == 0) k0 else kinetic(vel, mass);
            if (t == 0) e0 = .{ .k = k, .w = pe } else {
                const e = k + pe;
                const ez = e0.k + e0.w;
                std.debug.print("walk3 energy: n={d} eps={d} dt={d} theta {d:.1}, {d} ticks: E0 {d:.6} E {d:.6} dE/E0 {e:.3}; 2K/|W| {d:.4} -> {d:.4}; CPU build {d:.1} ms/tick, GPU walk {d:.2} ms/tick\n", .{
                    n, eps, dt, theta, ticks, ez, e, (e - ez) / @abs(ez), 2 * e0.k / @abs(e0.w), 2 * k / @abs(pe), build_ms / @as(f64, @floatFromInt(ticks + 1)), walk_ms / @as(f64, @floatFromInt(ticks + 1)),
                });
            }
        }
    }
}

fn kinetic(vel: []const L.Vel3, mass: []const f32) f64 {
    var k: f64 = 0;
    for (vel, mass) |v, m| k += 0.5 * @as(f64, m) * (@as(f64, v.x) * v.x + @as(f64, v.y) * v.y + @as(f64, v.z) * v.z);
    return k;
}

/// The open-domain drift of k3/step.zig: round to the nearest quantum, stick to the wall.
fn drift(p: u32, d: f32) u32 {
    const lo: u32 = 1 << 30;
    const hi: u32 = 3 * (1 << 30) - 1;
    const a = @min(@abs(d), 536870912.0);
    const m: u32 = @intFromFloat(@floor(a + 0.5));
    if (d < 0) return if (m > p - lo) lo else p - m;
    return if (m > hi - p) hi else p + m;
}
