//! Standalone 1M Plummer measurement for ref3. Build and run from the repo root:
//!   zig build-exe -O ReleaseFast -femit-bin=.scratch/ref3_measure --dep ref3 -Mroot=src/sim/ref3/measure.zig -Mref3=src/sim/ref3.zig
//!   .scratch/ref3_measure [n]
const std = @import("std");
const ref3 = @import("ref3");
const plummer = ref3.plummer;

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const n: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 1_000_000;
    const io = init.io;
    const half = 10.0;

    const raw = try plummer.positions(gpa, n, 42, half);
    const pos = try gpa.alloc(ref3.Pos3, n);
    const mass = try gpa.alloc(f32, n);
    for (raw, pos, mass) |r, *q, *m| {
        q.* = .{ .x = plummer.quantise(r[0], half), .y = plummer.quantise(r[1], half), .z = plummer.quantise(r[2], half) };
        m.* = 1.0 / @as(f32, @floatFromInt(n));
    }
    for ([_]f64{ 0.6, 0.5, 0.4 }) |theta| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        const t = try ref3.build(gpa, pos, mass, .{ .box = 2 * half, .theta = theta });
        const t1 = std.Io.Timestamp.now(io, .awake);
        const e = try ref3.forceError(gpa, t, pos, mass, 2000);
        const t2 = std.Io.Timestamp.now(io, .awake);
        std.debug.print("N={d} theta={d:.1} nodes/N={d:.3} rms={e:.2} p99={e:.2} per body: visits {d:.0} cells {d:.0} pairs {d:.0} | build {d} ms, direct+walk of 2000 targets {d} ms\n", .{
            n,                                                                 theta,
            @as(f64, @floatFromInt(t.nodes.len)) / @as(f64, @floatFromInt(n)), e.rms,
            e.p99,                                                             e.visits,
            e.cells,                                                           e.pairs,
            @divTrunc(t0.durationTo(t1).nanoseconds, 1_000_000),               @divTrunc(t1.durationTo(t2).nanoseconds, 1_000_000),
        });
        t.deinit(gpa);
    }
}
