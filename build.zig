const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Speed is the point of this project, so plain `zig build` is ReleaseFast; pass -Doptimize=debug for safety checks.
    const optimize = b.option(std.builtin.Optimize, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_c.addIncludePath(b.path("vendor"));
    const c_mod = translate_c.createModule();

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "c", .module = c_mod }},
    });
    exe_mod.addIncludePath(b.path("vendor"));
    exe_mod.addCSourceFile(.{ .file = b.path("vendor/impl.c"), .flags = &.{"-O2"} });
    exe_mod.linkSystemLibrary("SDL3", .{});

    const exe = b.addExecutable(.{ .name = "lettuce", .root_module = exe_mod });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.setCwd(b.path("."));
    run_cmd.addPassthruArgs();
    b.step("run", "Run the game").dependOn(&run_cmd.step);

    const tests = b.addTest(.{ .root_module = exe_mod });
    b.step("test", "Run tests").dependOn(&b.addRunArtifact(tests).step);

    // Each tool has its own root so modules can be built and verified independently.
    const Tool = struct { step: []const u8, root: []const u8, desc: []const u8 };
    const tools = [_]Tool{
        .{ .step = "bench", .root = "src/bench.zig", .desc = "Headless physics benchmark" },
        .{ .step = "render-demo", .root = "src/render_demo.zig", .desc = "Render one frame to .scratch/frame.bmp" },
        .{ .step = "audio-demo", .root = "src/audio_demo.zig", .desc = "Decode and play each sound" },
    };
    for (tools) |tool| {
        const mod = b.createModule(.{
            .root_source_file = b.path(tool.root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "c", .module = c_mod }},
        });
        mod.addIncludePath(b.path("vendor"));
        mod.addCSourceFile(.{ .file = b.path("vendor/impl.c"), .flags = &.{"-O2"} });
        mod.linkSystemLibrary("SDL3", .{});
        const tool_exe = b.addExecutable(.{ .name = tool.step, .root_module = mod });
        const tool_run = b.addRunArtifact(tool_exe);
        tool_run.setCwd(b.path("."));
        tool_run.addPassthruArgs();
        b.step(tool.step, tool.desc).dependOn(&tool_run.step);
        const tool_tests = b.addTest(.{ .root_module = mod });
        b.step(b.fmt("test-{s}", .{tool.step}), b.fmt("Tests reachable from {s}", .{tool.root})).dependOn(&b.addRunArtifact(tool_tests).step);
    }
}
