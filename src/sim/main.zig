//! `zig build sim -- --scenario disk --n 1000000 --ticks 1000 --every 100` : headless matter
//! simulation (see sim.zig). `--bench` prints per-pass GPU times instead of running diagnostics.
const std = @import("std");
const c = @import("c");
const galaxy = @import("galaxy");
const sim = @import("sim.zig");
const snapshot = @import("snapshot.zig");
const snapper = @import("snapper.zig");

test {
    _ = @import("tests.zig");
}

const Options = struct {
    scenario: galaxy.Scenario = .disk,
    n: u32 = 100_000,
    ticks: u64 = 1000,
    dt: f32 = 1,
    every: u64 = 100,
    seed: u64 = 1,
    soft: f32 = 4,
    pe_samples: u32 = 0,
    bench: bool = false,
    out: ?[:0]const u8 = null,
    every_snap: u64 = 0,
    restart: ?[:0]const u8 = null,
};

const usage =
    \\usage: sim [--scenario disk|collision|cloud|systems] [--n N] [--ticks K] [--dt F] [--every K]
    \\           [--seed U64] [--soft EPS2] [--pe-samples K] [--bench]
    \\           [--out DIR --every-snap K] [--restart FILE]   (restart: n, g, softening, dt come from the file; --ticks = further ticks)
    \\       sim render-snap ...  (see `zig build render-snap`)
    \\
;

fn parse(args: []const [:0]const u8) !Options {
    var o: Options = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--bench")) {
            o.bench = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        const v = args[i];
        if (std.mem.eql(u8, a, "--scenario")) {
            o.scenario = std.meta.stringToEnum(galaxy.Scenario, v) orelse return error.BadScenario;
        } else if (std.mem.eql(u8, a, "--n")) {
            o.n = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--ticks")) {
            o.ticks = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, a, "--dt")) {
            o.dt = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, a, "--every")) {
            o.every = @max(1, try std.fmt.parseInt(u64, v, 10));
        } else if (std.mem.eql(u8, a, "--seed")) {
            o.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, a, "--soft")) {
            o.soft = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, a, "--out")) {
            o.out = v;
        } else if (std.mem.eql(u8, a, "--every-snap")) {
            o.every_snap = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, a, "--restart")) {
            o.restart = v;
        } else if (std.mem.eql(u8, a, "--pe-samples")) {
            o.pe_samples = try std.fmt.parseInt(u32, v, 10);
        } else return error.UnknownOption;
    }
    return o;
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 1 and std.mem.eql(u8, args[1], "render-snap")) return @import("render_cli.zig").run(gpa, args[2..]);
    const o = parse(args) catch |err| {
        std.debug.print("sim: {s}\n{s}", .{ @errorName(err), usage });
        return err;
    };
    if (o.every_snap > 0 and o.out == null) {
        std.debug.print("sim: --every-snap needs --out\n{s}", .{usage});
        return error.MissingValue;
    }

    // SDL_GPU needs the video subsystem initialised; the offscreen driver never opens a window.
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();
    const dev = try sim.createDevice();
    defer c.SDL_DestroyGPUDevice(dev);

    var t0 = c.SDL_GetPerformanceCounter();
    var cfg: sim.Config = undefined;
    var scenario = o.scenario;
    var seed = o.seed;
    var built: ?galaxy.Built = null;
    defer if (built) |*b| b.deinit(gpa);
    var restart_io: ?*c.SDL_IOStream = null;
    defer if (restart_io) |io| {
        _ = c.SDL_CloseIO(io);
    };
    var restart_header: snapshot.Header = undefined;
    if (o.restart) |path| {
        const io = try snapshot.open(path);
        restart_io = io;
        restart_header = try snapshot.readHeader(io);
        const h = &restart_header;
        scenario = std.enums.fromInt(galaxy.Scenario, h.scenario) orelse o.scenario;
        seed = h.seed;
        cfg = .{ .n = @intCast(h.n), .g = h.g, .softening = h.softening, .dt = h.dt, .pe_samples = o.pe_samples };
        std.debug.print("restart from {s}: {d} bodies at tick {d}, half_pending {d}\n", .{ path, h.n, h.tick, h.half_pending });
    } else {
        var prng = std.Random.DefaultPrng.init(o.seed);
        built = try galaxy.build(gpa, prng.random(), o.scenario, o.n, .{ .x = 0, .y = 0 });
        const b = &built.?;
        cfg = .{ .n = @intCast(b.bodies.len), .g = b.g, .softening = o.soft, .dt = o.dt, .pe_samples = o.pe_samples };
        std.debug.print("scenario {s}: {d} bodies built in {d:.0} ms (g {e:.3}, extent {d:.0})\n", .{ @tagName(o.scenario), cfg.n, sim.msSince(t0), b.g, b.extent });
    }
    const n = cfg.n;

    var s = sim.Sim.init(dev, cfg) catch |err| {
        std.debug.print("sim init failed for n={d}: {s} ({s})\n", .{ n, @errorName(err), c.SDL_GetError() });
        return err;
    };
    defer s.deinit();
    t0 = c.SDL_GetPerformanceCounter();
    if (built) |b| {
        try s.upload(b.bodies);
        s.setFixedBox(b.extent * 1.1);
    } else try snapper.restore(gpa, &s, restart_io.?, &restart_header);
    std.debug.print("uploaded in {d:.0} ms; {d} B/body resident ({d} state + {d} solver scratch) = {d:.2} GB\n", .{
        sim.msSince(t0),
        s.bytesPerBody(),
        @import("shared.zig").bytes_per_body_state,
        @import("gravity").scratch_bytes_per_body,
        @as(f64, @floatFromInt(s.bytesPerBody() * n)) / 1e9,
    });

    if (o.bench) return bench(&s, o);
    var sn: ?*snapper.Snapper = null;
    if (o.every_snap > 0) {
        _ = c.SDL_CreateDirectory(o.out.?.ptr);
        sn = try snapper.Snapper.create(gpa, &s);
    }
    defer if (sn) |p| p.destroy(gpa);
    try runDiagnostics(&s, o, sn, seed, @intFromEnum(scenario));
}

fn nextEvent(now: u64, every: u64) u64 {
    return if (every == 0) std.math.maxInt(u64) else (now / every + 1) * every;
}

fn runDiagnostics(s: *sim.Sim, o: Options, sn: ?*snapper.Snapper, seed: u64, scenario: u32) !void {
    const d0 = try s.measure();
    const e0 = d0.e();
    const k0 = d0.ke;
    const l0 = d0.lz;
    std.debug.print("potential energy from {d} sampled bodies of {d}{s}\n", .{ d0.samples, s.cfg.n, if (d0.samples == s.cfg.n) " (exact)" else " (estimate from a fixed sample: absolute E is approximate, its drift tracks the sampled terms)" });
    std.debug.print("{s:>8} {s:>14} {s:>10} {s:>10} {s:>10} {s:>14} {s:>10} {s:>9}\n", .{ "tick", "E", "dE/E0", "dE/K0", "|P|", "L", "dL/L0", "ms/tick" });
    report(s.tick_no, d0, e0, k0, l0, 0);
    const end = s.tick_no + o.ticks;
    const wall0 = c.SDL_GetPerformanceCounter();
    var ms_acc: f64 = 0;
    var ticks_acc: u64 = 0;
    while (s.tick_no < end) {
        const step = @min(@min(nextEvent(s.tick_no, o.every), nextEvent(s.tick_no, o.every_snap)), end) - s.tick_no;
        const t = c.SDL_GetPerformanceCounter();
        try s.run(step);
        ms_acc += sim.msSince(t);
        ticks_acc += step;
        const tick = s.tick_no;
        if (tick % o.every == 0 or tick == end) {
            report(tick, try s.measure(), e0, k0, l0, ms_acc / @as(f64, @floatFromInt(ticks_acc)));
            ms_acc = 0;
            ticks_acc = 0;
        }
        if (sn) |p| if (o.every_snap > 0 and tick % o.every_snap == 0) {
            var buf: [1024]u8 = undefined;
            try p.write(snapshot.pathZ(&buf, "{s}/snap_{d:0>8}.snap", .{ o.out.?, tick }), seed, scenario);
            if (p.writeFailed()) return error.SnapshotWrite;
            const st = p.lastStats();
            std.debug.print("  snapshot tick {d}: {d:.1} MB, simulation stalled {d:.0} ms\n", .{ tick, @as(f64, @floatFromInt(st.bytes)) / 1e6, st.stall_ms });
        };
    }
    std.debug.print("wall {d:.3} ms/tick over {d} ticks (diagnostics and snapshots included)\n", .{ sim.msSince(wall0) / @as(f64, @floatFromInt(o.ticks)), o.ticks });
    if (sn) |p| {
        p.flush();
        if (p.writeFailed()) return error.SnapshotWrite;
        const st = p.lastStats();
        std.debug.print("last snapshot: {d:.1} MB in {d:.0} ms ({d:.0} MB/s start to last byte written)\n", .{ @as(f64, @floatFromInt(st.bytes)) / 1e6, st.total_ms, @as(f64, @floatFromInt(st.bytes)) / 1e6 / (st.total_ms / 1000) });
    }
}

fn report(tick: u64, d: sim.Diag, e0: f64, k0: f64, l0: f64, ms: f64) void {
    std.debug.print("{d:>8} {e:>14.6} {e:>10.2} {e:>10.2} {e:>10.3} {e:>14.6} {e:>10.2} {d:>9.3}\n", .{
        tick, d.e(), (d.e() - e0) / e0, (d.e() - e0) / k0, d.p(), d.lz, (d.lz - l0) / l0, ms,
    });
}

fn bench(s: *sim.Sim, o: Options) !void {
    try s.run(5);
    const ticks = @min(o.ticks, 200);
    const t = c.SDL_GetPerformanceCounter();
    try s.run(ticks);
    const pipelined = sim.msSince(t) / @as(f64, @floatFromInt(ticks));
    const pt = try s.timePasses(10);
    std.debug.print(
        \\bench n={d} leaf level {d}: pipelined {d:.3} ms/tick over {d} ticks
        \\  blocking per-pass ms: gravity {d:.3} (of which box reduction {d:.3}; fixed box {d:.3})  drift {d:.3}  diagnostics {d:.3}  potential {d:.3} ({d} samples)
        \\
    , .{ s.cfg.n, s.grav.last_leaf_level, pipelined, ticks, pt.gravity, pt.box, pt.gravity_fixed_box, pt.drift, pt.diag, pt.potential, s.samples });
}
