//! Backend shim for the k3 kernels. Kernel bodies use only what is exported here, so a PTX
//! backend (nvptx64-cuda, see docs/sim3d.md "Backends") replaces this file and nothing else.
//! Only the SPIR-V backend exists today.
const std = @import("std");
const spirv = std.spirv;
pub const L = @import("gpu_layout");

pub const group_size: u32 = L.group_size;
/// Workgroups per row of a 2D dispatch: Vulkan caps one dimension at 65535.
pub const dispatch_x: u32 = 32768;

pub const cc: std.builtin.CallingConvention = .{ .spirv_kernel = .{ .x = group_size, .y = 1, .z = 1 } };

fn Arr(comptime T: type) type {
    return extern struct { v: @SpirvType(.{ .runtime_array = T }) };
}

/// Read-only storage buffer. SDL_GPU puts these in descriptor set 0, in binding order.
pub fn ro(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 0, .binding = binding } } });
}

/// Read-write storage buffer: descriptor set 1.
pub fn rw(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.storage_buffer) Arr(T) {
    return @extern(*addrspace(.storage_buffer) Arr(T), .{ .name = name, .decoration = .{ .descriptor = .{ .set = 1, .binding = binding } } });
}

/// Uniform block in slot 0 (descriptor set 2).
pub fn uniform(comptime T: type, comptime name: []const u8) *addrspace(.uniform) const T {
    return @extern(*addrspace(.uniform) const T, .{ .name = name, .decoration = .{ .descriptor = .{ .set = 2, .binding = 0 } } });
}

pub inline fn load(comptime buf: anytype, i: u32) @TypeOf(buf.v[0]) {
    return buf.v[i];
}

pub inline fn store(comptime buf: anytype, i: u32, value: @TypeOf(buf.v[0])) void {
    buf.v[i] = value;
}

/// Workgroup-shared array. Index it with a runtime value: a comptime-folded index (x[l - l])
/// miscompiles in the SPIR-V backend.
pub fn Shared(comptime T: type, comptime n: u32, comptime name: []const u8) *addrspace(.shared) [n]T {
    return @extern(*addrspace(.shared) [n]T, .{ .name = name });
}

/// Flat thread index; safe for the 2D dispatches that cover more than dispatch_x groups.
pub inline fn gid() u32 {
    return spirv.global_invocation_id[0] + spirv.global_invocation_id[1] * (dispatch_x * group_size);
}

pub inline fn lid() u32 {
    return spirv.local_invocation_id[0];
}

pub inline fn groupId() u32 {
    return spirv.workgroup_id[0] + spirv.workgroup_id[1] * dispatch_x;
}

pub inline fn barrier() void {
    spirv.workgroupBarrier();
}

/// The backend has no @atomicRmw ("TODO (SPIR-V): implement AIR tag atomic_rmw"), so this is
/// OpAtomicIAdd at device scope with relaxed semantics; results are consumed by later passes.
pub inline fn atomicAdd(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
    return asm volatile (
        \\%ret = OpAtomicIAdd %u32 %p %scope %sem %v
        : [ret] "" (-> u32),
        : [u32] "t" (u32),
          [p] "" (p),
          [scope] "" (@as(u32, @backingInt(spirv.Scope.device))),
          [sem] "" (@as(u32, 0)),
          [v] "" (v),
    );
}

/// Shift-based because @clz is not implemented in the SPIR-V backend ("TODO (SPIR-V): implement
/// unary operation 'clz'").
pub inline fn clz(x0: u32) u32 {
    var x = x0;
    var n: u32 = 0;
    if (x <= 0x0000ffff) {
        n += 16;
        x <<= 16;
    }
    if (x <= 0x00ffffff) {
        n += 8;
        x <<= 8;
    }
    if (x <= 0x0fffffff) {
        n += 4;
        x <<= 4;
    }
    if (x <= 0x3fffffff) {
        n += 2;
        x <<= 2;
    }
    if (x <= 0x7fffffff) n += 1;
    return if (x0 == 0) 32 else n;
}

/// Signed distance between two positions in quanta, as f32: the wrapped difference reinterpreted
/// as i32, so the periodic minimum image comes for free and an open domain (the middle half of
/// the range) never overflows it.
pub inline fn delta(a: u32, b: u32) f32 {
    return @floatFromInt(@as(i32, @bitCast(a -% b)));
}
