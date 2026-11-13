//! Minimal PNG encoder: 8-bit RGB, no filtering, zlib "stored" (uncompressed) deflate blocks.
//! Files are large but any decoder reads them, and no C encoder is vendored.
const std = @import("std");

const max_stored = 65535;

fn chunk(out: *std.ArrayList(u8), gpa: std.mem.Allocator, kind: *const [4]u8, data: []const u8) !void {
    var len: [4]u8 = undefined;
    std.mem.writeInt(u32, &len, @intCast(data.len), .big);
    try out.appendSlice(gpa, &len);
    try out.appendSlice(gpa, kind);
    try out.appendSlice(gpa, data);
    var crc = std.hash.Crc32.init();
    crc.update(kind);
    crc.update(data);
    var tail: [4]u8 = undefined;
    std.mem.writeInt(u32, &tail, crc.final(), .big);
    try out.appendSlice(gpa, &tail);
}

/// `rgba` is width*height pixels of 4 bytes (alpha ignored). Caller frees the result.
pub fn encodeRgba(gpa: std.mem.Allocator, width: u32, height: u32, rgba: []const u8) ![]u8 {
    std.debug.assert(rgba.len == @as(usize, width) * height * 4);
    const row_bytes = @as(usize, width) * 3 + 1;
    const raw_len = row_bytes * height;
    const blocks = (raw_len + max_stored - 1) / max_stored;
    var z: std.ArrayList(u8) = .empty;
    defer z.deinit(gpa);
    try z.ensureTotalCapacity(gpa, raw_len + blocks * 5 + 6);
    z.appendSliceAssumeCapacity(&.{ 0x78, 0x01 });
    var adler: std.hash.Adler32 = .{};
    // Build the raw scanline stream, then cut it into stored blocks.
    var raw = try gpa.alloc(u8, raw_len);
    defer gpa.free(raw);
    for (0..height) |y| {
        const dst = raw[y * row_bytes ..][0..row_bytes];
        dst[0] = 0;
        for (0..width) |x| {
            const s = rgba[(y * width + x) * 4 ..][0..3];
            dst[1 + x * 3 ..][0..3].* = s.*;
        }
    }
    adler.update(raw);
    var at: usize = 0;
    while (at < raw_len) {
        const m = @min(max_stored, raw_len - at);
        const final: u8 = @intFromBool(at + m == raw_len);
        var hdr: [5]u8 = undefined;
        hdr[0] = final;
        std.mem.writeInt(u16, hdr[1..3], @intCast(m), .little);
        std.mem.writeInt(u16, hdr[3..5], @intCast(~@as(u16, @intCast(m))), .little);
        z.appendSliceAssumeCapacity(&hdr);
        z.appendSliceAssumeCapacity(raw[at..][0..m]);
        at += m;
    }
    var ad: [4]u8 = undefined;
    std.mem.writeInt(u32, &ad, adler.adler, .big);
    z.appendSliceAssumeCapacity(&ad);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "\x89PNG\r\n\x1a\n");
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], width, .big);
    std.mem.writeInt(u32, ihdr[4..8], height, .big);
    ihdr[8..13].* = .{ 8, 2, 0, 0, 0 };
    try chunk(&out, gpa, "IHDR", &ihdr);
    try chunk(&out, gpa, "IDAT", z.items);
    try chunk(&out, gpa, "IEND", &.{});
    return out.toOwnedSlice(gpa);
}
