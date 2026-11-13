//! Barnes-Hut tree kick (docs/sim3d.md, work package 5): one thread per body walks the radix tree
//! along its ropes. A port of ref3.Tree.walk (the spec) to f32: same acceptance test, same 40-eps
//! guard, same softened leaf pairs and traceless-quadrupole far field, so the interaction lists
//! agree except where f32 and f64 round r^2 to opposite sides of open_r2.
//!
//! Bodies are in Morton order (the leaves index them), so neighbouring threads walk nearly the same
//! nodes. Each body sums its terms in one fixed order and nothing is shared between threads, so a
//! rerun is bit-identical.
const C = @import("common.zig");
const L = C.L;

/// Uniform block. Mirrored in src/sim/walk3.zig (the host cannot import a kernel file); proposed
/// for gpu_layout.zig.
pub const WalkParams3 = extern struct {
    n: u32,
    /// First body of this dispatch and the number of bodies in it (chunked for the driver watchdog).
    first: u32,
    count: u32,
    flags: u32,
    g: f32,
    eps: f32,
    eps2: f32,
    kick: f32,
    box_scale: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

const pos = C.ro(L.Pos3, 0, "pos");
const mass = C.ro(f32, 1, "mass");
const nodes = C.ro(L.Node, 2, "nodes");
const quads = C.ro(L.NodeQuad, 3, "quads");
const vel = C.rw(L.Vel3, 0, "vel");
const amag = C.rw(f32, 1, "amag");
const pot = C.rw(f32, 2, "pot");
const params = C.uniform(WalkParams3, "params");

const leaf_flag: u32 = 0x8000_0000;
const end: u32 = 0xFFFF_FFFF;
const count_mask: u32 = (1 << 26) - 1;

fn cc(comptime x: u32) @import("std").builtin.CallingConvention {
    return .{ .spirv_kernel = .{ .x = x, .y = 1, .z = 1 } };
}

inline fn walkBody(comptime gs: u32) void {
    const group = C.groupId();
    // Whole groups past this chunk (a 2D dispatch rounds up) leave before touching anything.
    if (group * gs >= params.count) return;
    const local = group * gs + C.lid();
    // The tail of the last group may belong to the next chunk.
    if (local >= params.count) return;
    const i = params.first + local;

    const me = C.load(pos, i);
    const bs = params.box_scale;
    const eps2 = params.eps2;
    // ref3: a node is never accepted inside 40 softening lengths.
    const near2 = 1600 * eps2;
    var ax: f32 = 0;
    var ay: f32 = 0;
    var az: f32 = 0;
    var phi: f32 = 0;
    var idx: u32 = 0;
    while (idx != end) {
        const nd = C.load(nodes, idx);
        // Target minus com, physical units.
        const rx = C.delta(me.x, nd.com.x) * bs;
        const ry = C.delta(me.y, nd.com.y) * bs;
        const rz = C.delta(me.z, nd.com.z) * bs;
        const r2 = rx * rx + ry * ry + rz * rz;
        if (r2 > nd.open_r2 and r2 > near2) {
            const q = C.load(quads, idx);
            const ir2 = 1.0 / r2;
            const ir = @sqrt(ir2);
            const ir3 = ir2 * ir;
            const ir5 = ir3 * ir2;
            const qx = q.xx * rx + q.xy * ry + q.xz * rz;
            const qy = q.xy * rx + q.yy * ry + q.yz * rz;
            const qz = q.xz * rx + q.yz * ry + q.zz * rz;
            const rqr = rx * qx + ry * qy + rz * qz;
            const mono = -nd.mass * ir3;
            const quad_r = -2.5 * rqr * ir5 * ir2;
            ax += mono * rx + qx * ir5 + quad_r * rx;
            ay += mono * ry + qy * ir5 + quad_r * ry;
            az += mono * rz + qz * ir5 + quad_r * rz;
            phi += -nd.mass * ir - 0.5 * rqr * ir5;
            idx = nd.skip;
        } else if (nd.child & leaf_flag != 0) {
            const first = nd.child & ~leaf_flag;
            const stop = first + (nd.info & count_mask);
            var j = first;
            while (j < stop) : (j += 1) {
                // Self pair skipped by index, as in ref3: subtracting its m/eps afterwards cancels
                // a term that can be larger than the whole potential.
                if (j != i) {
                    const p = C.load(pos, j);
                    const m = C.load(mass, j);
                    const dx = C.delta(p.x, me.x) * bs;
                    const dy = C.delta(p.y, me.y) * bs;
                    const dz = C.delta(p.z, me.z) * bs;
                    const inv = 1.0 / @sqrt(dx * dx + dy * dy + dz * dz + eps2);
                    const w = m * inv * inv * inv;
                    ax += w * dx;
                    ay += w * dy;
                    az += w * dz;
                    phi -= m * inv;
                }
            }
            idx = nd.skip;
        } else idx = nd.child;
    }

    const g = params.g;
    const gx = ax * g;
    const gy = ay * g;
    const gz = az * g;
    if (params.flags & L.sim3_flag_store_accel != 0) {
        C.store(vel, i, .{ .x = gx, .y = gy, .z = gz });
    } else {
        const v = C.load(vel, i);
        const k = params.kick;
        C.store(vel, i, .{ .x = v.x + gx * k, .y = v.y + gy * k, .z = v.z + gz * k });
    }
    C.store(amag, i, @sqrt(gx * gx + gy * gy + gz * gz));
    if (params.flags & L.sim3_flag_potential != 0) {
        C.store(pot, i, g * phi);
    }
}

// One entry point per workgroup size; the host picks one (walk3.zig `group_sizes`).
export fn walk64() callconv(cc(64)) void {
    walkBody(64);
}
export fn walk128() callconv(cc(128)) void {
    walkBody(128);
}
export fn walk256() callconv(cc(256)) void {
    walkBody(256);
}
