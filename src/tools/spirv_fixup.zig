//! Build step: `spirv_fixup <in.spv> <out.spv>`.
//!
//! Zig's self-hosted SPIR-V backend emits, next to each Block-decorated storage-buffer struct,
//! an undecorated duplicate of it (a struct whose last member is a runtime array). Vulkan
//! requires Block or BufferBlock on any such struct (VUID-StandaloneSpirv-OpTypeRuntimeArray-04680);
//! RADV from Mesa 25.0 segfaults creating the sprite pipeline because of it (Mesa 26 tolerates it).
//! This adds `OpDecorate %s Block` for each one, after the module's existing decorations.
const std = @import("std");

const op_decorate: u32 = 71;
const op_member_decorate: u32 = 72;
const op_decorate_id: u32 = 332;
const op_decorate_string: u32 = 5632;
const op_member_decorate_string: u32 = 5633;
const op_type_runtime_array: u32 = 29;
const op_type_struct: u32 = 30;
const decoration_block: u32 = 2;
const decoration_buffer_block: u32 = 3;
const magic: u32 = 0x07230203;
const header_words = 5;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.log.err("usage: spirv_fixup <in.spv> <out.spv>", .{});
        return error.InvalidArguments;
    }
    const cwd = std.Io.Dir.cwd();
    const bytes = try cwd.readFileAlloc(init.io, args[1], init.gpa, .limited(64 << 20));
    defer init.gpa.free(bytes);
    const out = try fixup(init.gpa, bytes);
    defer init.gpa.free(out);
    try cwd.writeFile(init.io, .{ .sub_path = args[2], .data = out });
}

/// Returns the module with the missing Block decorations added (a copy even if none were missing).
pub fn fixup(gpa: std.mem.Allocator, bytes: []const u8) ![]align(4) u8 {
    if (bytes.len % 4 != 0 or bytes.len < header_words * 4) return error.NotSpirv;
    const words = try gpa.alloc(u32, bytes.len / 4);
    defer gpa.free(words);
    @memcpy(std.mem.sliceAsBytes(words), bytes);
    if (words[0] != magic) return error.NotSpirv;

    var runtime_arrays: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer runtime_arrays.deinit(gpa);
    var blocks: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer blocks.deinit(gpa);
    var need: std.ArrayList(u32) = .empty;
    defer need.deinit(gpa);
    // Index (in words) just past the last decoration; new decorations go there.
    var insert_at: ?usize = null;

    var i: usize = header_words;
    while (i < words.len) {
        const count = words[i] >> 16;
        const op = words[i] & 0xffff;
        if (count == 0 or i + count > words.len) return error.NotSpirv;
        const inst = words[i .. i + count];
        switch (op) {
            op_decorate => {
                if (count >= 3 and (inst[2] == decoration_block or inst[2] == decoration_buffer_block)) try blocks.put(gpa, inst[1], {});
                insert_at = i + count;
            },
            op_member_decorate, op_decorate_id, op_decorate_string, op_member_decorate_string => insert_at = i + count,
            op_type_runtime_array => try runtime_arrays.put(gpa, inst[1], {}),
            // Types are declared before use, so the member's runtime array is already known.
            op_type_struct => if (count >= 3 and runtime_arrays.contains(inst[count - 1])) try need.append(gpa, inst[1]),
            else => {},
        }
        i += count;
    }

    var missing: std.ArrayList(u32) = .empty;
    defer missing.deinit(gpa);
    for (need.items) |id| if (!blocks.contains(id)) try missing.append(gpa, id);
    const at = insert_at orelse return error.NoDecorations;

    const out_bytes = try gpa.alignedAlloc(u8, .of(u32), 4 * (words.len + 3 * missing.items.len));
    const out = std.mem.bytesAsSlice(u32, out_bytes);
    @memcpy(out[0..at], words[0..at]);
    for (missing.items, 0..) |id, k| {
        out[at + 3 * k ..][0..3].* = .{ (3 << 16) | op_decorate, id, decoration_block };
    }
    @memcpy(out[at + 3 * missing.items.len ..], words[at..]);
    return out_bytes;
}

test "adds Block to an undecorated runtime-array struct only" {
    const gpa = std.testing.allocator;
    // header; OpDecorate %5 Block; %3 = OpTypeInt 32 0; %4 = OpTypeRuntimeArray %3;
    // %5 = OpTypeStruct %4; %6 = OpTypeStruct %4; %7 = OpTypeStruct %3
    const in = [_]u32{
        magic,                    0x00010500, 0, 8, 0,
        (3 << 16) | op_decorate,  5,          decoration_block,
        (4 << 16) | 21,           3,          32, 0,
        (3 << 16) | op_type_runtime_array, 4, 3,
        (3 << 16) | op_type_struct, 5, 4,
        (3 << 16) | op_type_struct, 6, 4,
        (3 << 16) | op_type_struct, 7, 3,
    };
    const out = try fixup(gpa, std.mem.sliceAsBytes(&in));
    defer gpa.free(out);
    const w = std.mem.bytesAsSlice(u32, out);
    try std.testing.expectEqual(in.len + 3, w.len);
    try std.testing.expectEqualSlices(u32, &.{ (3 << 16) | op_decorate, 6, decoration_block }, w[8..11]);
    try std.testing.expectEqualSlices(u32, in[8..], w[11..]);
}
