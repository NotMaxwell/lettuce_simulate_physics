//! `zig build sim -- --scenario disk --n 1000000 --ticks 1000 --every 100` : headless matter
//! simulation (see sim.zig). `--bench` prints per-pass GPU times instead of running diagnostics.
const std = @import("std");
const c = @import("c");
const galaxy = @import("galaxy");
const sim = @import("sim.zig");

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
};

const usage =
    \\usage: sim [--scenario disk|collision|cloud|systems] [--n N] [--ticks K] [--dt F] [--every K]
    \\           [--seed U64] [--soft EPS2] [--pe-samples K] [--bench]
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
        } else if (std.mem.eql(u8, a, "--pe-samples")) {
            o.pe_samples = try std.fmt.parseInt(u32, v, 10);
        } else return error.UnknownOption;
    }
    return o;
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const o = parse(try init.minimal.args.toSlice(init.arena.allocator())) catch |err| {
        std.debug.print("sim: {s}\n{s}", .{ @errorName(err), usage });
        return err;
    };

    // SDL_GPU needs the video subsystem initialised; the offscreen driver never opens a window.
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    defer c.SDL_Quit();
    const dev = try sim.createDevice();
    defer c.SDL_DestroyGPUDevice(dev);

    var t0 = c.SDL_GetPerformanceCounter();
    var prng = std.Random.DefaultPrng.init(o.seed);
    var built = try galaxy.build(gpa, prng.random(), o.scenario, o.n, .{ .x = 0, .y = 0 });
    defer built.deinit(gpa);
    const n: u32 = @intCast(built.bodies.len);
    std.debug.print("scenario {s}: {d} bodies built in {d:.0} ms (g {e:.3}, extent {d:.0})\n", .{ @tagName(o.scenario), n, sim.msSince(t0), built.g, built.extent });

    var s = sim.Sim.init(dev, .{ .n = n, .g = built.g, .softening = o.soft, .dt = o.dt, .pe_samples = o.pe_samples }) catch |err| {
        std.debug.print("sim init failed for n={d}: {s} ({s})\n", .{ n, @errorName(err), c.SDL_GetError() });
        return err;
    };
    defer s.deinit();
    t0 = c.SDL_GetPerformanceCounter();
    try s.upload(built.bodies);
    s.setFixedBox(built.extent * 1.1);
    std.debug.print("uploaded in {d:.0} ms; {d} B/body resident ({d} state + {d} solver scratch) = {d:.2} GB\n", .{
        sim.msSince(t0),
        s.bytesPerBody(),
        @import("shared.zig").bytes_per_body_state,
        @import("gravity").scratch_bytes_per_body,
        @as(f64, @floatFromInt(s.bytesPerBody() * n)) / 1e9,
    });

    if (o.bench) return bench(&s, o);
    try runDiagnostics(&s, o);
}

fn runDiagnostics(s: *sim.Sim, o: Options) !void {
    const d0 = try s.measure();
    const e0 = d0.e();
    const k0 = d0.ke;
    const l0 = d0.lz;
    std.debug.print("potential energy from {d} sampled bodies of {d}{s}\n", .{ d0.samples, s.cfg.n, if (d0.samples == s.cfg.n) " (exact)" else " (estimate from a fixed sample: absolute E is approximate, its drift tracks the sampled terms)" });
    std.debug.print("{s:>8} {s:>14} {s:>10} {s:>10} {s:>10} {s:>14} {s:>10} {s:>9}\n", .{ "tick", "E", "dE/E0", "dE/K0", "|P|", "L", "dL/L0", "ms/tick" });
    report(0, d0, e0, k0, l0, 0);
    var done: u64 = 0;
    while (done < o.ticks) {
        const step = @min(o.every, o.ticks - done);
        const t = c.SDL_GetPerformanceCounter();
        try s.run(step);
        const ms = sim.msSince(t) / @as(f64, @floatFromInt(step));
        done += step;
        report(done, try s.measure(), e0, k0, l0, ms);
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
