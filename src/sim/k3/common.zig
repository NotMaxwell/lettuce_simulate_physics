//! Backend shim for the k3 kernels. Kernel bodies use only what is exported here; the backend is
//! picked at comptime from the target: spirv64 gives SPIR-V (Vulkan/SDL_GPU), nvptx64 gives PTX
//! (CUDA driver API, `zig build ptx`, docs/ptx.md).
//!
//! PTX host contract. A CUDA kernel normally takes pointers as parameters, but these are
//! `fn () callconv(.kernel)` that name their buffers through module-level constants, so each
//! `ro`/`rw` buffer becomes one exported `.global` variable called `name`, holding a device
//! pointer that the kernel loads. The host writes it before launch (cuModuleGetGlobal, then
//! cuMemcpyHtoD of 8 bytes). `uniform(T, name)` is an exported `.global` variable `name` of type T
//! (cuModuleGetGlobal, cuMemcpyHtoD of @sizeOf(T) bytes). Binding numbers are ignored: the symbol
//! name is the whole interface. Launch with block = (group size, 1, 1) and grid =
//! (min(g, dispatch_x), ceil(g / dispatch_x), 1) for g groups, the same 2D dispatch as SPIR-V:
//! `groupId` is ctaid.x + ctaid.y * dispatch_x and `gid` is groupId * ntid.x + tid.x.
const std = @import("std");
const builtin = @import("builtin");
pub const L = @import("gpu_layout");

const is_ptx = builtin.cpu.arch == .nvptx64;

pub const group_size: u32 = L.group_size;
/// Workgroups per row of a 2D dispatch: Vulkan caps one dimension at 65535.
pub const dispatch_x: u32 = 32768;

pub const cc: std.builtin.CallingConvention = if (is_ptx) .kernel else .{ .spirv_kernel = .{ .x = group_size, .y = 1, .z = 1 } };

/// Calling convention for a kernel with its own group size x (SPIR-V local size; PTX takes it from the launch block).
pub fn ccFor(comptime x: u32) std.builtin.CallingConvention {
    return if (is_ptx) .kernel else .{ .spirv_kernel = .{ .x = x, .y = 1, .z = 1 } };
}

const be = if (is_ptx) ptx else spv;

pub const ro = be.ro;
pub const rw = be.rw;
pub const uniform = be.uniform;
pub const Shared = be.Shared;
pub const gid = be.gid;
pub const lid = be.lid;
pub const groupId = be.groupId;
pub const barrier = be.barrier;
pub const atomicAdd = be.atomicAdd;
pub const atomicIncShared = be.atomicIncShared;
pub const clz = be.clz;

pub inline fn load(comptime buf: anytype, i: u32) @TypeOf(buf.v[0]) {
    return buf.v[i];
}

pub inline fn store(comptime buf: anytype, i: u32, value: @TypeOf(buf.v[0])) void {
    buf.v[i] = value;
}

/// atomicAdd on element `i` of a u32 storage buffer; keeps buffer fields out of kernel bodies.
pub inline fn atomicAddAt(comptime buf: anytype, i: u32, v: u32) u32 {
    return atomicAdd(&buf.v[i], v);
}

const spv = struct {
    const spirv = std.spirv;

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

    /// Workgroup-scope OpAtomicIAdd of 1 on shared memory; the result is unused, so only counts depend on it.
    pub inline fn atomicIncShared(p: *addrspace(.shared) u32) void {
        _ = asm volatile (
            \\%ret = OpAtomicIAdd %u32 %p %scope %sem %v
            : [ret] "" (-> u32),
            : [u32] "t" (u32),
              [p] "" (p),
              [scope] "" (@as(u32, @backingInt(spirv.Scope.workgroup))),
              [sem] "" (@as(u32, 0)),
              [v] "" (@as(u32, 1)),
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
};

const ptx = struct {
    fn Arr(comptime T: type) type {
        return extern struct { v: [*]addrspace(.global) T };
    }

    /// The returned pointer is to an exported global holding the device pointer, so `buf.v[i]`
    /// in load/store reads it from the symbol `name` (host-written, see the header).
    fn buffer(comptime T: type, comptime name: []const u8) *addrspace(.global) Arr(T) {
        const S = struct {
            const key = name; // captured, so each name gets its own variable
            var slot: Arr(T) addrspace(.global) = undefined;
        };
        @export(&S.slot, .{ .name = S.key });
        return &S.slot;
    }

    pub fn ro(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.global) Arr(T) {
        _ = binding;
        return buffer(T, name);
    }

    pub fn rw(comptime T: type, comptime binding: u32, comptime name: []const u8) *addrspace(.global) Arr(T) {
        _ = binding;
        return buffer(T, name);
    }

    pub fn uniform(comptime T: type, comptime name: []const u8) *addrspace(.global) const T {
        const S = struct {
            const key = name;
            var slot: T addrspace(.global) = undefined;
        };
        @export(&S.slot, .{ .name = S.key });
        return &S.slot;
    }

    pub fn Shared(comptime T: type, comptime n: u32, comptime name: []const u8) *addrspace(.shared) [n]T {
        const S = struct {
            const key = name; // makes the type per-name, or equal-shaped arrays would alias
            var arr: [n]T addrspace(.shared) = undefined;
        };
        _ = S.key;
        return &S.arr;
    }

    pub inline fn lid() u32 {
        return @workItemId(0);
    }

    pub inline fn groupId() u32 {
        return @workGroupId(0) + @workGroupId(1) * dispatch_x;
    }

    pub inline fn gid() u32 {
        return ptx.groupId() * @workGroupSize(0) + @workItemId(0);
    }

    pub inline fn barrier() void {
        asm volatile ("bar.sync 0;" ::: .{ .memory = true });
    }

    pub inline fn atomicAdd(p: *addrspace(.global) u32, v: u32) u32 {
        return @atomicRmw(u32, p, .Add, v, .monotonic);
    }

    pub inline fn atomicIncShared(p: *addrspace(.shared) u32) void {
        _ = @atomicRmw(u32, p, .Add, 1, .monotonic);
    }

    pub inline fn clz(x: u32) u32 {
        return @clz(x);
    }
};

/// Signed distance between two positions in quanta, as f32: the wrapped difference reinterpreted
/// as i32, so the periodic minimum image comes for free and an open domain (the middle half of
/// the range) never overflows it.
pub inline fn delta(a: u32, b: u32) f32 {
    return @floatFromInt(@as(i32, @bitCast(a -% b)));
}

/// Max bodies per tree leaf. Lives here, not in tree.zig, so walk.zig can check it without
/// importing tree.zig: on PTX that import would drag tree's kernels and its "params" symbol in.
pub const leaf_cap_bodies: u32 = 32;
