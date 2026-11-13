//! GPU tree build (tree3.zig) against the f64 reference (ref3.zig). Inputs are sorted on the CPU
//! with ref3 and uploaded, as the GPU sort is a separate package.
//! `TREE3_BENCH=1 zig build test-tree3` also times the build at 1M, 4M and 16M bodies.
const std = @import("std");
const c = @import("c");
const L = @import("gpu_layout");
const sim = @import("sim.zig");
const ref3 = @import("ref3.zig");
const tree3 = @import("tree3.zig");
const Sort3 = @import("sort3.zig").Sort3;

const gpa = std.testing.allocator;
const Buffer = *c.SDL_GPUBuffer;

fn device() !*c.SDL_GPUDevice {
    _ = c.SDL_SetHint(c.SDL_HINT_VIDEO_DRIVER, "offscreen");
    if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.SdlInit;
    return sim.createDevice();
}

fn upload(dev: *c.SDL_GPUDevice, buf: Buffer, bytes: []const u8) !void {
    const chunk: usize = 64 << 20;
    const tb = c.SDL_CreateGPUTransferBuffer(dev, &.{ .usage = c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = @intCast(@min(chunk, bytes.len)) }) orelse return error.GpuBuffer;
    defer c.SDL_ReleaseGPUTransferBuffer(dev, tb);
    var at: usize = 0;
    while (at < bytes.len) : (at += chunk) {
        const part = bytes[at..@min(at + chunk, bytes.len)];
        const m: [*]u8 = @ptrCast(c.SDL_MapGPUTransferBuffer(dev, tb, false) orelse return error.GpuBuffer);
        @memcpy(m[0..part.len], part);
        c.SDL_UnmapGPUTransferBuffer(dev, tb);
        const cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
        const cp = c.SDL_BeginGPUCopyPass(cmd);
        c.SDL_UploadToGPUBuffer(cp, &.{ .transfer_buffer = tb }, &.{ .buffer = buf, .offset = @intCast(at), .size = @intCast(part.len) }, false);
        c.SDL_EndGPUCopyPass(cp);
        try tree3.submitWait(dev, cmd);
    }
}

/// Sorted inputs on the GPU.
const Gpu = struct {
    dev: *c.SDL_GPUDevice,
    in: tree3.Inputs,

    fn init(dev: *c.SDL_GPUDevice, cap: u32) !Gpu {
        return .{ .dev = dev, .in = .{
            .pos = try tree3.makeBuffer(dev, @as(u64, cap) * 12, 0),
            .mass = try tree3.makeBuffer(dev, @as(u64, cap) * 4, 0),
            .keys = try tree3.makeBuffer(dev, @as(u64, cap) * 12, 0),
        } };
    }
    fn deinit(g: Gpu) void {
        for ([_]Buffer{ g.in.pos, g.in.mass, g.in.keys }) |b| c.SDL_ReleaseGPUBuffer(g.dev, b);
    }
    fn set(g: Gpu, pos: []const ref3.Pos3, mass: []const f32, keys: []const u64, perm: []const u32) !void {
        const recs = try gpa.alloc(L.SortRec, keys.len);
        defer gpa.free(recs);
        for (recs, keys, perm) |*r, k, p| r.* = .{ .key_hi = @intCast(k >> 32), .key_lo = @truncate(k), .idx = p };
        try upload(g.dev, g.in.pos, std.mem.sliceAsBytes(pos));
        try upload(g.dev, g.in.mass, std.mem.sliceAsBytes(mass));
        try upload(g.dev, g.in.keys, std.mem.sliceAsBytes(recs));
    }
};

fn buildGpu(t: *tree3.Tree3, g: Gpu, n: u32, p: tree3.Params) !tree3.Status {
    const cmd = c.SDL_AcquireGPUCommandBuffer(g.dev) orelse return error.GpuSubmit;
    try t.record(cmd, g.in, n, p);
    try tree3.submitWait(g.dev, cmd);
    return t.readStatus();
}

const Result = struct { nodes: []L.Node, quads: []L.NodeQuad };

fn readTree(t: *tree3.Tree3, count: u32) !Result {
    const nodes = try gpa.alloc(L.Node, count);
    errdefer gpa.free(nodes);
    const quads = try gpa.alloc(L.NodeQuad, count);
    errdefer gpa.free(quads);
    try tree3.download(t.dev, t.nodes, 0, std.mem.sliceAsBytes(nodes));
    try tree3.download(t.dev, t.quads, 0, std.mem.sliceAsBytes(quads));
    return .{ .nodes = nodes, .quads = quads };
}

/// Stated tolerances against the f64 reference, per node. com in quanta: absolute plus relative to
/// the node's extent (its longest bounding-box side); reach = sqrt(open_r2) likewise; quadrupole
/// components relative to mass * extent^2; mass relative.
const tol = struct {
    const mass_rel = 5e-7;
    const com_quanta = 2.0;
    const com_rel = 1e-6;
    const reach_rel = 1e-6;
    const quad_rel = 1e-6;
};

const Worst = struct { mass: f64 = 0, com: f64 = 0, reach: f64 = 0, quad: f64 = 0 };

/// Topology bit-exact, floats within `tol`; returns the worst error seen in units of each bound.
fn compare(r: ref3.Tree, res: Result) !Worst {
    var w: Worst = .{};
    const q2p = r.box / 0x1p32;
    for (r.nodes, res.nodes, r.quads, res.quads, 0..) |rn, gn, rq, gq, x| {
        if (rn.child != gn.child or rn.skip != gn.skip or rn.info != gn.info) {
            std.debug.print("tree3 node {d}: ref child {x} skip {x} info {x}, gpu child {x} skip {x} info {x}\n", .{ x, rn.child, rn.skip, rn.info, gn.child, gn.skip, gn.info });
            return error.TopologyMismatch;
        }
        // The node's bodies: from its leftmost leaf, count from info (exact below 2^26).
        var y: usize = x;
        while (!ref3.isLeaf(r.nodes[y])) y = r.nodes[y].child;
        const first = r.nodes[y].child & ~ref3.leaf_flag;
        var lo: [3]u32 = @splat(std.math.maxInt(u32));
        var hi: [3]u32 = @splat(0);
        for (r.pos[first..][0..ref3.nodeCount(rn)]) |p| for ([3]u32{ p.x, p.y, p.z }, 0..) |v, k| {
            lo[k] = @min(lo[k], v);
            hi[k] = @max(hi[k], v);
        };
        var ext: f64 = 0;
        for (0..3) |k| ext = @max(ext, @as(f64, @floatFromInt(hi[k] - lo[k])));

        w.mass = @max(w.mass, @abs(@as(f64, gn.mass) - rn.mass) / rn.mass / tol.mass_rel);
        const rc = [3]u32{ rn.com.x, rn.com.y, rn.com.z };
        const gc = [3]u32{ gn.com.x, gn.com.y, gn.com.z };
        for (rc, gc) |a, b| {
            const d: f64 = @floatFromInt(@as(i64, a) - @as(i64, b));
            w.com = @max(w.com, @abs(d) / (tol.com_quanta + tol.com_rel * ext));
        }
        const rr = @sqrt(@as(f64, rn.open_r2)) / q2p;
        const gr = @sqrt(@as(f64, gn.open_r2)) / q2p;
        w.reach = @max(w.reach, @abs(rr - gr) / (tol.com_quanta + tol.reach_rel * rr));
        const scale = @as(f64, rn.mass) * (ext * q2p) * (ext * q2p);
        inline for (.{ "xx", "xy", "xz", "yy", "yz", "zz" }) |f| {
            const d = @abs(@as(f64, @field(gq, f)) - @field(rq, f));
            // Single-body leaves (scale 0) must be exactly zero in both.
            w.quad = @max(w.quad, if (scale > 0) d / scale / tol.quad_rel else if (d == 0) 0 else std.math.inf(f64));
        }
    }
    return w;
}

fn checkCase(name: []const u8, t: *tree3.Tree3, g: Gpu, pos: []const ref3.Pos3, mass: []const f32, box: f64, theta: f32) !void {
    const r = try ref3.build(gpa, pos, mass, .{ .box = box, .theta = theta });
    defer r.deinit(gpa);
    try g.set(r.pos, r.mass, r.keys, r.perm);
    const n: u32 = @intCast(pos.len);
    const st = try buildGpu(t, g, n, .{ .theta = theta, .box_size = box });
    try std.testing.expect(!st.overflow);
    try std.testing.expectEqual(r.leaves, st.leaves);
    try std.testing.expectEqual(@as(u32, @intCast(r.nodes.len)), st.nodes);
    const res = try readTree(t, st.nodes);
    defer gpa.free(res.nodes);
    defer gpa.free(res.quads);
    const w = try compare(r, res);
    std.debug.print("tree3 {s}: n {d} leaves {d} nodes/N {d:.4}; worst error / tolerance: mass {d:.3} com {d:.3} reach {d:.3} quad {d:.3}\n", .{
        name, n, st.leaves, @as(f64, @floatFromInt(st.nodes)) / @as(f64, @floatFromInt(n)), w.mass, w.com, w.reach, w.quad,
    });
    try std.testing.expect(w.mass <= 1 and w.com <= 1 and w.reach <= 1 and w.quad <= 1);
    if (n >= 10_000) try std.testing.expect(@as(f64, @floatFromInt(st.nodes)) / @as(f64, @floatFromInt(n)) <= 0.12);
}

const Cloud = struct {
    pos: []ref3.Pos3,
    mass: []f32,
    fn deinit(cl: Cloud) void {
        gpa.free(cl.pos);
        gpa.free(cl.mass);
    }
};

fn plummer(n: usize, seed: u64, cut: f64) !Cloud {
    const raw = try ref3.plummer.positions(gpa, n, seed, cut);
    defer gpa.free(raw);
    const pos = try gpa.alloc(ref3.Pos3, n);
    const mass = try gpa.alloc(f32, n);
    for (raw, pos, mass) |p, *q, *m| {
        q.* = .{ .x = ref3.plummer.quantise(p[0], cut), .y = ref3.plummer.quantise(p[1], cut), .z = ref3.plummer.quantise(p[2], cut) };
        m.* = 1.0 / @as(f32, @floatFromInt(n));
    }
    return .{ .pos = pos, .mass = mass };
}

/// Uniform over `[lo, lo + span)` per axis with masses in [0.5, 1.5).
fn uniform(n: usize, seed: u64, lo: u32, span: u32) !Cloud {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const pos = try gpa.alloc(ref3.Pos3, n);
    const mass = try gpa.alloc(f32, n);
    for (pos, mass) |*q, *m| {
        q.* = .{ .x = lo + r.uintLessThan(u32, span), .y = lo + r.uintLessThan(u32, span), .z = lo + r.uintLessThan(u32, span) };
        m.* = 0.5 + r.float(f32);
    }
    return .{ .pos = pos, .mass = mass };
}

/// Bodies on `sites` points, jittered only below the 21-bit key resolution, so each site is one
/// run of equal keys far longer than the leaf cap.
fn sites(n: usize, seed: u64, count: u32) !Cloud {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const centres = try gpa.alloc(ref3.Pos3, count);
    defer gpa.free(centres);
    for (centres) |*s| s.* = .{ .x = r.int(u32) & ~@as(u32, 0x7FF), .y = r.int(u32) & ~@as(u32, 0x7FF), .z = r.int(u32) & ~@as(u32, 0x7FF) };
    const pos = try gpa.alloc(ref3.Pos3, n);
    const mass = try gpa.alloc(f32, n);
    for (pos, mass, 0..) |*q, *m, i| {
        // Every fourth body is a lone uniform one, so leaves of ordinary cells mix in.
        if (i % 4 == 3) {
            q.* = .{ .x = r.int(u32), .y = r.int(u32), .z = r.int(u32) };
        } else {
            const s = centres[r.uintLessThan(u32, count)];
            q.* = .{ .x = s.x | r.uintLessThan(u32, 0x800), .y = s.y | r.uintLessThan(u32, 0x800), .z = s.z | r.uintLessThan(u32, 0x800) };
        }
        m.* = 1;
    }
    return .{ .pos = pos, .mass = mass };
}

test "tree3: random, Plummer and equal-key inputs match ref3" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const cap = 200_000;
    var t = try tree3.Tree3.init(dev, cap);
    defer t.deinit();
    const g = try Gpu.init(dev, cap);
    defer g.deinit();

    const pl = try plummer(cap, 42, 10);
    defer pl.deinit();
    try checkCase("plummer 200k theta 0.6", &t, g, pl.pos, pl.mass, 20, 0.6);
    try checkCase("plummer 10k theta 0.4", &t, g, pl.pos[0..10_000], pl.mass[0..10_000], 20, 0.4);
    const un = try uniform(100_000, 7, 0, std.math.maxInt(u32));
    defer un.deinit();
    try checkCase("uniform 100k full range", &t, g, un.pos, un.mass, 1, 0.6);
    const small = try uniform(30_000, 8, 1 << 30, 1 << 12);
    defer small.deinit();
    try checkCase("uniform 30k in a 2^12-quanta cube", &t, g, small.pos, small.mass, 3, 0.5);
    const dup = try sites(20_000, 9, 60);
    defer dup.deinit();
    try checkCase("equal-key runs 20k on 60 sites", &t, g, dup.pos, dup.mass, 1, 0.6);
}

test "tree3: degenerate inputs (N = 1, 2, 33, 100 with 80 coincident bodies)" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    var t = try tree3.Tree3.init(dev, 100);
    defer t.deinit();
    const g = try Gpu.init(dev, 100);
    defer g.deinit();
    // ref3's degenerate set.
    var pos: [100]ref3.Pos3 = undefined;
    var mass: [100]f32 = undefined;
    for (&pos, &mass, 0..) |*p, *m, i| {
        p.* = if (i < 80) .{ .x = 1 << 30, .y = 1 << 30, .z = 1 << 30 } else .{ .x = @intCast(i * 1000003), .y = 5, .z = @intCast(i << 20) };
        m.* = 1;
    }
    for ([_]usize{ 1, 2, 33, 100 }) |k| {
        var buf: [16]u8 = undefined;
        try checkCase(try std.fmt.bufPrint(&buf, "N={d}", .{k}), &t, g, pos[0..k], mass[0..k], 1, 0.5);
    }
    // 33 distinct bodies: just over one leaf.
    const u = try uniform(33, 3, 0, std.math.maxInt(u32));
    defer u.deinit();
    try checkCase("N=33 distinct", &t, g, u.pos, u.mass, 1, 0.6);
}

test "tree3: reruns are bit-identical, also after building other inputs in between" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const n = 100_000;
    var t = try tree3.Tree3.init(dev, n);
    defer t.deinit();
    const g = try Gpu.init(dev, n);
    defer g.deinit();
    const a = try plummer(n, 5, 10);
    defer a.deinit();
    const b = try uniform(n / 2, 6, 0, std.math.maxInt(u32));
    defer b.deinit();
    const sa = try ref3.sortKeys(gpa, a.pos);
    defer sa.deinit(gpa);
    const sb = try ref3.sortKeys(gpa, b.pos);
    defer sb.deinit(gpa);
    const pa = try permuted(a, sa.perm);
    defer pa.deinit();
    const pb = try permuted(b, sb.perm);
    defer pb.deinit();
    const prm: tree3.Params = .{ .theta = 0.6, .box_size = 20 };

    var first: ?Result = null;
    defer if (first) |f| {
        gpa.free(f.nodes);
        gpa.free(f.quads);
    };
    for (0..3) |rep| {
        try g.set(pa.pos, pa.mass, sa.keys, sa.perm);
        const st = try buildGpu(&t, g, n, prm);
        const res = try readTree(&t, st.nodes);
        if (first) |f| {
            defer gpa.free(res.nodes);
            defer gpa.free(res.quads);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(f.nodes), std.mem.sliceAsBytes(res.nodes));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(f.quads), std.mem.sliceAsBytes(res.quads));
        } else first = res;
        if (rep == 1) {
            try g.set(pb.pos, pb.mass, sb.keys, sb.perm);
            _ = try buildGpu(&t, g, n / 2, prm);
        }
    }
}

fn permuted(cl: Cloud, perm: []const u32) !Cloud {
    const pos = try gpa.alloc(ref3.Pos3, perm.len);
    const mass = try gpa.alloc(f32, perm.len);
    for (perm, pos, mass) |o, *p, *m| {
        p.* = cl.pos[o];
        m.* = cl.mass[o];
    }
    return .{ .pos = pos, .mass = mass };
}

test "tree3: overflow sets the flag on the GPU and replaces the root with an empty leaf" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const n = 5000;
    var t = try tree3.Tree3.initNodes(dev, n, 64);
    defer t.deinit();
    const g = try Gpu.init(dev, n);
    defer g.deinit();
    const prm: tree3.Params = .{ .theta = 0.6, .box_size = 1 };

    // A tree that fits first, so the overflow build has a stale tree to hide.
    const few = try uniform(100, 12, 0, std.math.maxInt(u32));
    defer few.deinit();
    const sf = try ref3.sortKeys(gpa, few.pos);
    defer sf.deinit(gpa);
    const pf = try permuted(few, sf.perm);
    defer pf.deinit();
    try g.set(pf.pos, pf.mass, sf.keys, sf.perm);
    const ok = try buildGpu(&t, g, 100, prm);
    try std.testing.expect(!ok.overflow);
    var root: [1]L.Node = undefined;
    try tree3.download(dev, t.nodes, 0, std.mem.sliceAsBytes(&root));
    try std.testing.expect(root[0].mass > 0);

    const u = try uniform(n, 11, 0, std.math.maxInt(u32));
    defer u.deinit();
    const s = try ref3.sortKeys(gpa, u.pos);
    defer s.deinit(gpa);
    const p = try permuted(u, s.perm);
    defer p.deinit();
    try g.set(p.pos, p.mass, s.keys, s.perm);
    const st = try buildGpu(&t, g, n, prm);
    try std.testing.expect(st.overflow);
    try std.testing.expect(st.nodes > 64);
    var word: [1]u32 = undefined;
    try tree3.download(dev, t.statusBuffer(), 4 * tree3.status_overflow, std.mem.sliceAsBytes(&word));
    try std.testing.expectEqual(@as(u32, 1), word[0]);
    try tree3.download(dev, t.nodes, 0, std.mem.sliceAsBytes(&root));
    try std.testing.expectEqual(L.Node{ .com = .{ .x = 0, .y = 0, .z = 0 }, .mass = 0, .open_r2 = 0, .child = ref3.leaf_flag, .skip = ref3.end, .info = 0 }, root[0]);
    var q: [1]L.NodeQuad = undefined;
    try tree3.download(dev, t.quads, 0, std.mem.sliceAsBytes(&q));
    try std.testing.expectEqual(L.NodeQuad{ .xx = 0, .xy = 0, .xz = 0, .yy = 0, .yz = 0, .zz = 0 }, q[0]);
}

test "tree3: one leaf of a 100k-body equal-key run is summed by a workgroup and matches ref3" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const n = 102_000;
    var t = try tree3.Tree3.init(dev, n);
    defer t.deinit();
    const g = try Gpu.init(dev, n);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(21);
    const r = prng.random();
    const cl = try uniform(n, 22, 0, std.math.maxInt(u32));
    defer cl.deinit();
    // 100k bodies inside one 2^11-quanta key cell, the rest uniform; also a 300-body run, just
    // above the single-thread limit.
    for (cl.pos[0..100_000]) |*p| p.* = .{ .x = 0x4000_0000 | r.uintLessThan(u32, 0x800), .y = 0x2000_0000 | r.uintLessThan(u32, 0x800), .z = 0x6000_0000 | r.uintLessThan(u32, 0x800) };
    for (cl.pos[100_000..100_300]) |*p| p.* = .{ .x = 0x9000_0000 | r.uintLessThan(u32, 0x800), .y = 0x1000_0000 | r.uintLessThan(u32, 0x800), .z = 0x3000_0000 | r.uintLessThan(u32, 0x800) };
    try checkCase("100k equal-key run + 300-body run + 1700 uniform", &t, g, cl.pos, cl.mass, 1, 0.6);
}

test "tree3: built on Sort3 output with the scratch aliased onto its idle records" {
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const n = 200_000;
    var sort = try Sort3.init(dev, n);
    defer sort.deinit();
    var t = try tree3.Tree3.initOnSort(dev, &sort);
    defer t.deinit();
    const g = try Gpu.init(dev, n);
    defer g.deinit();
    const cl = try plummer(n, 31, 10);
    defer cl.deinit();
    try upload(dev, g.in.pos, std.mem.sliceAsBytes(cl.pos));
    try upload(dev, g.in.mass, std.mem.sliceAsBytes(cl.mass));
    const prm: tree3.Params = .{ .theta = 0.6, .box_size = 20 };
    var cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
    cmd = try sort.record(cmd, .{ .pos = g.in.pos, .mass = g.in.mass }, n);
    try t.record(cmd, .{ .pos = g.in.pos, .mass = g.in.mass, .keys = sort.sorted() }, n, prm);
    try tree3.submitWait(dev, cmd);
    const st = try t.readStatus();
    try std.testing.expect(!st.overflow);

    const r = try ref3.build(gpa, cl.pos, cl.mass, .{ .box = 20, .theta = 0.6 });
    defer r.deinit(gpa);
    try std.testing.expectEqual(@as(u32, @intCast(r.nodes.len)), st.nodes);
    const res = try readTree(&t, st.nodes);
    defer gpa.free(res.nodes);
    defer gpa.free(res.quads);
    const w = try compare(r, res);
    std.debug.print("tree3 on sort3, aliased scratch: nodes/N {d:.4}, {d:.1} B/body own; worst error / tolerance: mass {d:.3} com {d:.3} reach {d:.3} quad {d:.3}\n", .{
        @as(f64, @floatFromInt(st.nodes)) / n, t.bytesPerBody(), w.mass, w.com, w.reach, w.quad,
    });
    try std.testing.expect(w.mass <= 1 and w.com <= 1 and w.reach <= 1 and w.quad <= 1);

    // The size check: more nodes than the borrowed 12 B/body holds must fail at init.
    try std.testing.expectError(error.ScratchTooSmall, tree3.Tree3.initWith(dev, n, .{ .node_cap = n / 2, .scratch = sort.scratch(), .scratch_bytes = @as(u64, n) * 12 }));
    var tiny = try Sort3.init(dev, 20);
    defer tiny.deinit();
    try std.testing.expectError(error.ScratchTooSmall, tree3.Tree3.initOnSort(dev, &tiny));
}

test "tree3: build time at 1M, 4M and 16M Plummer bodies (TREE3_BENCH=1)" {
    if (c.SDL_getenv("TREE3_BENCH") == null) return error.SkipZigTest;
    const dev = try device();
    defer c.SDL_DestroyGPUDevice(dev);
    const max_n = 16 << 20;
    var t = try tree3.Tree3.init(dev, max_n);
    defer t.deinit();
    const g = try Gpu.init(dev, max_n);
    defer g.deinit();
    for ([_]u32{ 1 << 20, 4 << 20, 16 << 20 }) |n| {
        const cl = try plummer(n, 1, 10);
        defer cl.deinit();
        const s = try ref3.sortKeys(gpa, cl.pos);
        defer s.deinit(gpa);
        const p = try permuted(cl, s.perm);
        defer p.deinit();
        try g.set(p.pos, p.mass, s.keys, s.perm);
        const prm: tree3.Params = .{ .theta = 0.6, .box_size = 20 };
        _ = try buildGpu(&t, g, n, prm);
        var best: f64 = std.math.inf(f64);
        for (0..5) |_| {
            const cmd = c.SDL_AcquireGPUCommandBuffer(dev) orelse return error.GpuSubmit;
            const t0 = c.SDL_GetPerformanceCounter();
            try t.record(cmd, g.in, n, prm);
            try tree3.submitWait(dev, cmd);
            best = @min(best, sim.msSince(t0));
        }
        const st = try t.readStatus();
        std.debug.print("tree3 bench n={d}: build {d:.3} ms (min of 5, submit to fence), nodes/N {d:.4}, deepest internal node {d}, overflow {}, {d:.1} B/body tree+scratch at capacity\n", .{
            n, best, @as(f64, @floatFromInt(st.nodes)) / @as(f64, @floatFromInt(n)), st.max_depth, st.overflow, t.bytesPerBody(),
        });
        try std.testing.expect(!st.overflow);
    }
}
