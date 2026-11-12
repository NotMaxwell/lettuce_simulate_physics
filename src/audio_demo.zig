const std = @import("std");
const c = @import("c");
const audio = @import("audio.zig");

fn report(name: []const u8, info: ?audio.Info) void {
    if (info) |i| {
        std.debug.print("{s}: {d:.2} s, {d} Hz, {d} ch\n", .{ name, i.seconds, i.sample_rate, i.channels });
    } else {
        std.debug.print("{s}: not loaded\n", .{name});
    }
}

pub fn main() void {
    var a = audio.Audio.init(std.heap.smp_allocator);
    defer a.deinit();
    std.debug.print("audio active: {}\n", .{a.isActive()});

    for (std.enums.values(audio.Sound)) |s| {
        report(@tagName(s), a.soundInfo(s));
        a.play(s);
    }

    var waited: u32 = 0;
    while (a.isActive() and a.musicInfo() == null and waited < 10000) : (waited += 50) c.SDL_Delay(50);
    report("music", a.musicInfo());
    c.SDL_Delay(1000);

    a.setMusicMuted(true);
    std.debug.print("music muted\n", .{});
    c.SDL_Delay(200);
    a.setMusicMuted(false);
    std.debug.print("music unmuted\n", .{});
}
