//! Snapshot file format: a fixed 256-byte header, then one planar f32 array per field
//! (struct-of-arrays), each starting on a 4096-byte boundary so chunks can be written and read
//! independently. Little endian. Velocities are synchronous (v at `tick`), or, when
//! `half_pending` is 0, the raw half-step velocities v(tick - 1/2) of the integrator.
const std = @import("std");
const c = @import("c");

comptime {
    if (@import("builtin").cpu.arch.endian() != .little) @compileError("snapshot files are little endian");
}

pub const magic: [8]u8 = "LTCSNAP\x00".*;
pub const version: u32 = 1;
pub const header_bytes = 256;
pub const data_align = 4096;

pub const Field = enum(u32) { pos_x, pos_y, vel_x, vel_y, mass };
pub const field_count = @typeInfo(Field).@"enum".field_names.len;
pub const max_fields = 8;

pub const FieldEntry = extern struct { id: u32, elem_bytes: u32, offset: u64 };

pub const Header = extern struct {
    magic: [8]u8 = magic,
    version: u32 = version,
    dim: u32 = 2,
    n: u64,
    tick: u64,
    time: f64,
    dt: f32,
    g: f32,
    softening: f32,
    /// 1: `vel` is synchronous with `pos`. 0: it lags by half a step and the next tick kicks a full step.
    half_pending: u32,
    seed: u64,
    /// `galaxy.Scenario` as an integer.
    scenario: u32,
    fields_used: u32 = field_count,
    fields: [max_fields]FieldEntry = @splat(.{ .id = 0, .elem_bytes = 0, .offset = 0 }),
    _reserved: [header_bytes - 72 - max_fields * 16]u8 = @splat(0),

    pub fn init(n: u64, tick: u64, dt: f32, g: f32, softening: f32, half_pending: bool, seed: u64, scenario: u32) Header {
        var h: Header = .{
            .n = n,
            .tick = tick,
            .time = @as(f64, @floatFromInt(tick)) * dt,
            .dt = dt,
            .g = g,
            .softening = softening,
            .half_pending = @intFromBool(half_pending),
            .seed = seed,
            .scenario = scenario,
        };
        for (0..field_count) |i| h.fields[i] = .{ .id = @intCast(i), .elem_bytes = 4, .offset = fieldOffset(n, i) };
        return h;
    }

    pub fn offset(h: *const Header, f: Field) u64 {
        return h.fields[@intFromEnum(f)].offset;
    }

    pub fn fileSize(h: *const Header) u64 {
        return fieldOffset(h.n, field_count);
    }

    /// Payload bytes (arrays only, no padding).
    pub fn payloadBytes(h: *const Header) u64 {
        return h.n * 4 * field_count;
    }
};

comptime {
    std.debug.assert(@sizeOf(Header) == header_bytes);
}

fn fieldOffset(n: u64, i: usize) u64 {
    const stride = std.mem.alignForward(u64, n * 4, data_align);
    return data_align + stride * i;
}

pub const Error = error{ FileOpen, FileIo, BadMagic, BadVersion, BadHeader };

pub fn create(path: [:0]const u8, h: *const Header) Error!*c.SDL_IOStream {
    const io = c.SDL_IOFromFile(path.ptr, "wb") orelse return error.FileOpen;
    errdefer _ = c.SDL_CloseIO(io);
    try writeAt(io, 0, std.mem.asBytes(h));
    return io;
}

pub fn writeAt(io: *c.SDL_IOStream, offset: u64, bytes: []const u8) Error!void {
    if (c.SDL_SeekIO(io, @intCast(offset), c.SDL_IO_SEEK_SET) < 0) return error.FileIo;
    if (c.SDL_WriteIO(io, bytes.ptr, bytes.len) != bytes.len) return error.FileIo;
}

pub fn open(path: [:0]const u8) Error!*c.SDL_IOStream {
    return c.SDL_IOFromFile(path.ptr, "rb") orelse error.FileOpen;
}

pub fn readHeader(io: *c.SDL_IOStream) Error!Header {
    var h: Header = undefined;
    try readAt(io, 0, std.mem.asBytes(&h));
    if (!std.mem.eql(u8, &h.magic, &magic)) return error.BadMagic;
    if (h.version != version) return error.BadVersion;
    if (h.dim != 2 or h.fields_used != field_count) return error.BadHeader;
    return h;
}

pub fn readAt(io: *c.SDL_IOStream, offset: u64, bytes: []u8) Error!void {
    if (c.SDL_SeekIO(io, @intCast(offset), c.SDL_IO_SEEK_SET) < 0) return error.FileIo;
    if (c.SDL_ReadIO(io, bytes.ptr, bytes.len) != bytes.len) return error.FileIo;
}

/// Reads `out.len` values of `field` starting at body `start`.
pub fn readField(io: *c.SDL_IOStream, h: *const Header, f: Field, start: u64, out: []f32) Error!void {
    try readAt(io, h.offset(f) + start * 4, std.mem.sliceAsBytes(out));
}

pub fn writeField(io: *c.SDL_IOStream, h: *const Header, f: Field, start: u64, data: []const f32) Error!void {
    try writeAt(io, h.offset(f) + start * 4, std.mem.sliceAsBytes(data));
}

pub fn pathZ(buf: []u8, comptime fmt: []const u8, args: anytype) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, fmt, args, 0) catch unreachable;
}
