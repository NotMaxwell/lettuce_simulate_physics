//! Copies each ball's position into the current trail ring slot (or fills the whole ring with
//! the invalid marker when `clear` is set). Bindings: ro 0 pos, 1 meta; rw 0 ring; uniform set 2.
const std = @import("std");
const spirv = std.spirv;
const L = @import("gpu_layout");
const T = @import("trail_common.zig");

const Pos = extern struct { v: @SpirvType(.{ .runtime_array = L.Vec2 }) };
const Meta = extern struct { v: @SpirvType(.{ .runtime_array = L.BallMeta }) };

const pos = @extern(*addrspace(.storage_buffer) Pos, .{ .name = "pos", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const meta = @extern(*addrspace(.storage_buffer) Meta, .{ .name = "meta", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
const ring = @extern(*addrspace(.storage_buffer) Pos, .{ .name = "ring", .decoration = .{ .descriptor = .{ .set = 1, .binding = 0 } } });
const p = @extern(*addrspace(.uniform) T.RecParams, .{ .name = "rp", .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });

export fn record() callconv(.{ .spirv_kernel = .{ .x = L.group_size, .y = 1, .z = 1 } }) void {
    const i = spirv.global_invocation_id[0];
    if (i >= p.count) return;
    const bad: L.Vec2 = .{ .x = T.invalid, .y = T.invalid };
    if (p.clear != 0) {
        var s: u32 = 0;
        while (s < T.trail_len) : (s += 1) ring.v[s * T.trail_cap + i] = bad;
        return;
    }
    ring.v[p.slot * T.trail_cap + i] = if (meta.v[i] & L.flag_alive != 0) pos.v[i] else bad;
}
