//! Snapshot file format: a fixed 256-byte header, then one planar 4-byte array per field
//! (struct-of-arrays; f32, or u32 for 3D positions), each starting on a 4096-byte boundary so chunks can be written and read
//! independently. Little endian. Velocities are synchronous (v at `tick`), or, when
//! `half_pending` is 0, the raw half-step velocities v(tick - 1/2) of the integrator.
const std = @import("std");
const c = @import("c");

comptime {
    if (@import("builtin").cpu.arch.endian() != .little) @compileError("snapshot files are little endian");
}

pub const magic: [8]u8 = "LTCSNAP\x00".*;
/// Version written by the 2D sim; version 2 adds dim=3 and the box/boundary/cosmology block. Both load.
pub const version_2d: u32 = 1;
pub const version_3d: u32 = 2;
pub const version = version_2d;
pub const header_bytes = 256;
pub const data_align = 4096;

/// 2D fields (version 1): ids are also the index into `Header.fields`.
pub const Field = enum(u32) { pos_x, pos_y, vel_x, vel_y, mass };
pub const field_count = @typeInfo(Field).@"enum".field_names.len;
/// 3D fields (version 2). Positions are u32 fractions of the root cube (value / 2^32 of `box_size`,
/// world origin at the cube centre); velocities and mass are f32. `mass` is absent when `flag_uniform_mass`.
pub const Field3 = enum(u32) { pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass };
pub const field_count3 = @typeInfo(Field3).@"enum".field_names.len;
pub const max_fields = 8;

pub const Boundary = enum(u32) { open, periodic };
pub const flag_uniform_mass: u32 = 1;

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
    // Version 2 block, inside what version 1 reserved (zero there). Offsets 200..236.
    box_size: f64 = 0,
    boundary: u32 = 0,
    /// `flag_uniform_mass`: the mass field is not stored, every body weighs `uniform_mass`.
    flags: u32 = 0,
    uniform_mass: f32 = 0,
    /// Cosmology placeholder (unused until comoving integration exists): scale factor, Omega_m, Omega_lambda, h.
    cosmo_a: f32 = 0,
    cosmo_omega_m: f32 = 0,
    cosmo_omega_l: f32 = 0,
    cosmo_h: f32 = 0,
    _reserved: [header_bytes - 236]u8 = @splat(0),

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
        h.setFields(field_count);
        return h;
    }

    pub const Params3 = struct {
        n: u64,
        tick: u64,
        dt: f32,
        g: f32,
        softening: f32,
        half_pending: bool,
        seed: u64,
        scenario: u32 = 0,
        box_size: f64,
        boundary: Boundary = .open,
        /// Non-null: every body has this mass and the mass field is omitted.
        uniform_mass: ?f32 = null,
    };

    pub fn init3(p: Params3) Header {
        var h = init(p.n, p.tick, p.dt, p.g, p.softening, p.half_pending, p.seed, p.scenario);
        h.version = version_3d;
        h.dim = 3;
        h.box_size = p.box_size;
        h.boundary = @backingInt(p.boundary);
        if (p.uniform_mass) |m| {
            h.flags |= flag_uniform_mass;
            h.uniform_mass = m;
        }
        h.setFields(if (p.uniform_mass != null) field_count3 - 1 else field_count3);
        return h;
    }

    /// Lays out `count` planar 4-byte arrays, field i having id i.
    fn setFields(h: *Header, count: usize) void {
        h.fields_used = @intCast(count);
        h.fields = @splat(.{ .id = 0, .elem_bytes = 0, .offset = 0 });
        for (0..count) |i| h.fields[i] = .{ .id = @intCast(i), .elem_bytes = 4, .offset = fieldOffset(h.n, i) };
    }

    pub fn hasUniformMass(h: *const Header) bool {
        return h.flags & flag_uniform_mass != 0;
    }

    pub fn offset(h: *const Header, f: Field) u64 {
        return h.fields[@backingInt(f)].offset;
    }

    pub fn offset3(h: *const Header, f: Field3) u64 {
        return h.fields[@backingInt(f)].offset;
    }

    pub fn fileSize(h: *const Header) u64 {
        return fieldOffset(h.n, h.fields_used);
    }

    /// Size of a complete file: arrays plus footer.
    pub fn completeSize(h: *const Header) u64 {
        return h.fileSize() + footer_bytes;
    }

    /// Payload bytes (arrays only, no padding).
    pub fn payloadBytes(h: *const Header) u64 {
        return h.n * 4 * h.fields_used;
    }
};

comptime {
    std.debug.assert(@sizeOf(Header) == header_bytes);
    std.debug.assert(@offsetOf(Header, "box_size") == 200);
}

fn fieldOffset(n: u64, i: usize) u64 {
    const stride = std.mem.alignForward(u64, n * 4, data_align);
    return data_align + stride * i;
}

pub const Error = error{ FileOpen, FileIo, BadMagic, BadVersion, BadHeader, Truncated };

/// Written after the last array, once everything else is on disk: `footer_magic` then the total file size.
pub const footer_magic: [8]u8 = "LTCEND\x00\x00".*;
pub const footer_bytes = 16;

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
    if (h.version != version_2d and h.version != version_3d) return error.BadVersion;
    try validate(&h);
    const size = c.SDL_GetIOSize(io);
    if (size < 0 or @as(u64, @intCast(size)) != h.completeSize()) return error.Truncated;
    var footer: [footer_bytes]u8 = undefined;
    try readAt(io, h.fileSize(), &footer);
    if (!std.mem.eql(u8, footer[0..8], &footer_magic) or std.mem.readInt(u64, footer[8..16], .little) != h.completeSize()) return error.Truncated;
    return h;
}

fn validate(h: *const Header) Error!void {
    if (h.n >= (1 << 40)) return error.BadHeader;
    var expect = h.*;
    if (h.version == version_2d) {
        if (h.dim != 2) return error.BadHeader;
        expect.setFields(field_count);
    } else {
        const uniform = h.hasUniformMass();
        if (h.dim != 3 or h.boundary > @backingInt(Boundary.periodic)) return error.BadHeader;
        if (!(h.box_size > 0) or !std.math.isFinite(h.box_size)) return error.BadHeader;
        if (uniform and !(h.uniform_mass > 0)) return error.BadHeader;
        expect.setFields(if (uniform) field_count3 - 1 else field_count3);
    }
    if (h.fields_used != expect.fields_used) return error.BadHeader;
    for (h.fields, expect.fields) |a, b| if (a.id != b.id or a.elem_bytes != b.elem_bytes or a.offset != b.offset) return error.BadHeader;
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

/// 3D variants: the raw 32-bit words of field `id` (u32 positions, or f32 bits).
pub fn readWords(io: *c.SDL_IOStream, h: *const Header, id: u32, start: u64, out: []u32) Error!void {
    if (id >= h.fields_used) return error.BadHeader;
    try readAt(io, h.fields[id].offset + start * 4, std.mem.sliceAsBytes(out));
}

pub fn writeWords(io: *c.SDL_IOStream, h: *const Header, id: u32, start: u64, data: []const u32) Error!void {
    if (id >= h.fields_used) return error.BadHeader;
    try writeAt(io, h.fields[id].offset + start * 4, std.mem.sliceAsBytes(data));
}

pub fn writeFooter(io: *c.SDL_IOStream, h: *const Header) Error!void {
    var footer: [footer_bytes]u8 = undefined;
    footer[0..8].* = footer_magic;
    std.mem.writeInt(u64, footer[8..16], h.completeSize(), .little);
    try writeAt(io, h.fileSize(), &footer);
}

pub fn pathZ(buf: []u8, comptime fmt: []const u8, args: anytype) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, fmt, args, 0) catch unreachable;
}
