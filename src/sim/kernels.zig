//! Sim kernels (compiled to SPIR-V by the build): leapfrog drift and the diagnostics reductions.
//! Each entry point declares its own bindings; the host binds them in the same order.
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");
const S = @import("shared.zig");

fn Rt(T: type) type {
    return extern struct { v: @SpirvType(.{ .runtime_array = T }) };
}
const pos_ro = @extern(*addrspace(.storage_buffer) const Rt(L.BallPos), .{ .name = "pos_ro", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const body = @extern(*addrspace(.storage_buffer) const Rt(L.BallBody), .{ .name = "body", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
// drift: set 1 = pos, vel. diagPartial: set 1 = vel, partials. potential: set 1 = partials.
const pos_rw = @extern(*addrspace(.storage_buffer) Rt(L.BallPos), .{ .name = "pos_rw", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });
const vel_rw = @extern(*addrspace(.storage_buffer) Rt(L.BallVel), .{ .name = "vel_rw", .decoration = .{ .descriptor = .{ .set = 1, .binding = 1 } } });
const vel_ro = @extern(*addrspace(.storage_buffer) Rt(L.BallVel), .{ .name = "vel_ro", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });
const partials_d = @extern(*addrspace(.storage_buffer) Rt(f32), .{ .name = "partials_d", .decoration = .{ .descriptor = .{ .set = 1, .binding = 1 } } });
const partials_p = @extern(*addrspace(.storage_buffer) Rt(f32), .{ .name = "partials_p", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });
const params = @extern(*addrspace(.uniform) const S.Params, .{ .name = "params", .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });

const cc: std.builtin.CallingConvention = .{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } };

inline fn gid() u32 {
    return spirv.global_invocation_id[0] + spirv.global_invocation_id[1] * (S.dispatch_x * L.group_size);
}

const sh0 = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh0" });
const sh1 = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh1" });
const sh2 = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh2" });
const sh3 = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh3" });
const sh4 = @extern(*addrspace(.shared) [L.group_size]f32, .{ .name = "sh4" });

/// Position += velocity * dt. Velocity already holds the (half-)kicks from the gravity pass.
export fn drift() callconv(cc) void {
    const i = gid();
    if (i >= params.n) return;
    pos_rw.v[i].x += vel_rw.v[i].x * params.dt;
    pos_rw.v[i].y += vel_rw.v[i].y * params.dt;
}

fn groups(n: u32) u32 {
    const threads = (n + S.diag_per_thread - 1) / S.diag_per_thread;
    return (threads + L.group_size - 1) / L.group_size;
}

/// Per workgroup: sum of 1/2 m v^2, m vx, m vy, m (x vy - y vx), m over the bodies it covers.
/// Strided so a warp reads consecutive bodies.
export fn diagPartial() callconv(cc) void {
    const l = spirv.local_invocation_id[0];
    const stride = groups(params.n) * L.group_size;
    var ke: f32 = 0;
    var px: f32 = 0;
    var py: f32 = 0;
    var lz: f32 = 0;
    var ms: f32 = 0;
    var i = gid();
    var k: u32 = 0;
    while (k < S.diag_per_thread) : (k += 1) {
        if (i < params.n) {
            const m = body.v[i].mass;
            const vx = vel_ro.v[i].x;
            const vy = vel_ro.v[i].y;
            ke += 0.5 * m * (vx * vx + vy * vy);
            px += m * vx;
            py += m * vy;
            lz += m * (pos_ro.v[i].x * vy - pos_ro.v[i].y * vx);
            ms += m;
        }
        i += stride;
    }
    sh0[l] = ke;
    sh1[l] = px;
    sh2[l] = py;
    sh3[l] = lz;
    sh4[l] = ms;
    var off: u32 = L.group_size / 2;
    while (off > 0) : (off >>= 1) {
        spirv.workgroupBarrier();
        if (l < off) {
            sh0[l] += sh0[l + off];
            sh1[l] += sh1[l + off];
            sh2[l] += sh2[l + off];
            sh3[l] += sh3[l + off];
            sh4[l] += sh4[l + off];
        }
    }
    spirv.workgroupBarrier();
    // Constant-index access to shared arrays miscompiles in the SPIR-V backend; l == 0 here.
    if (l == 0) {
        const o = (spirv.workgroup_id[0] + spirv.workgroup_id[1] * S.dispatch_x) * S.diag_stride;
        partials_d.v[o] = sh0[l];
        partials_d.v[o + 1] = sh1[l];
        partials_d.v[o + 2] = sh2[l];
        partials_d.v[o + 3] = sh3[l];
        partials_d.v[o + 4] = sh4[l];
    }
}

fn hash(x0: u32) u32 {
    var x = x0;
    x ^= x >> 16;
    x *%= 0x7feb352d;
    x ^= x >> 15;
    x *%= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

/// Pair potential sum for sampled bodies: thread t takes sampled body (t % samples) and slice
/// (t / samples) of the other bodies; adds m_i * sum_j m_j * ln(r^2 + eps) / 2 per workgroup.
/// (The softened pull g m d / (d^2 + eps) is the gradient of g m ln(d^2 + eps) / 2.)
export fn potential() callconv(cc) void {
    const l = spirv.local_invocation_id[0];
    const t = gid();
    var acc: f32 = 0;
    if (t < params.samples * params.chunks) {
        const si = t % params.samples;
        const i = if (params.samples == params.n) si else hash(si ^ params.seed) % params.n;
        const j0 = (t / params.samples) * params.chunk_len;
        const j1 = @min(j0 + params.chunk_len, params.n);
        const xi = pos_ro.v[i].x;
        const yi = pos_ro.v[i].y;
        // Two-level sum: slices of 64 terms, then the slices, to keep f32 rounding error small
        // over the ~50k terms a thread can see.
        var sum: f32 = 0;
        var j = j0;
        while (j < j1) {
            var part: f32 = 0;
            const end = @min(j + 64, j1);
            while (j < end) : (j += 1) {
                if (j == i) continue;
                const dx = pos_ro.v[j].x - xi;
                const dy = pos_ro.v[j].y - yi;
                part += body.v[j].mass * @log(dx * dx + dy * dy + params.softening);
            }
            sum += part;
        }
        acc = 0.5 * body.v[i].mass * sum;
    }
    sh0[l] = acc;
    var off: u32 = L.group_size / 2;
    while (off > 0) : (off >>= 1) {
        spirv.workgroupBarrier();
        if (l < off) sh0[l] += sh0[l + off];
    }
    spirv.workgroupBarrier();
    if (l == 0) partials_p.v[spirv.workgroup_id[0] + spirv.workgroup_id[1] * S.dispatch_x] = sh0[l];
}
