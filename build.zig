const std = @import("std");
const gpu_sim_build = @import("src/gpu_sim/build_kernels.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Speed is the point of this project, so plain `zig build` is ReleaseFast; pass -Doptimize=debug for safety checks.
    const optimize = b.option(std.builtin.Optimize, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;

    // SDL3 is built from source through the Zig package manager (static, X11/Wayland/audio are
    // dlopened at runtime), so no system SDL3 or C toolchain is needed. Always ReleaseFast: our
    // debug builds only need safety checks in our own code.
    const sdl_lib = b.dependency("sdl", .{ .target = target, .optimize = .ReleaseFast }).artifact("SDL3");

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_c.addIncludePath(b.path("vendor"));
    translate_c.addIncludePath(sdl_lib.getEmittedIncludeTree());
    const c_mod = translate_c.createModule();

    const layout_mod = b.createModule(.{ .root_source_file = b.path("src/gpu_layout.zig") });
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "c", .module = c_mod }, .{ .name = "gpu_layout", .module = layout_mod } },
    });
    exe_mod.addIncludePath(b.path("vendor"));
    exe_mod.addCSourceFile(.{ .file = b.path("vendor/impl.c"), .flags = &.{"-O2"} });
    exe_mod.linkLibrary(sdl_lib);
    addShaders(b, exe_mod, layout_mod);
    const gpu = GpuModules.create(b, target, optimize, c_mod, layout_mod);
    gpu.attach(exe_mod);

    const exe = b.addExecutable(.{ .name = "lettuce", .root_module = exe_mod });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.setCwd(b.path("."));
    run_cmd.addPassthruArgs();
    setGpuEnv(b, run_cmd);
    b.step("run", "Run the game").dependOn(&run_cmd.step);

    const tests = b.addTest(.{ .root_module = exe_mod });
    const test_run = b.addRunArtifact(tests);
    setGpuEnv(b, test_run); // gpu_world.zig has a test that needs the GPU
    b.step("test", "Run tests").dependOn(&test_run.step);

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
            .imports = &.{ .{ .name = "c", .module = c_mod }, .{ .name = "gpu_layout", .module = layout_mod } },
        });
        mod.addIncludePath(b.path("vendor"));
        mod.addCSourceFile(.{ .file = b.path("vendor/impl.c"), .flags = &.{"-O2"} });
        mod.linkLibrary(sdl_lib);
        if (std.mem.eql(u8, tool.step, "render-demo") or std.mem.eql(u8, tool.step, "bench")) addShaders(b, mod, layout_mod);
        if (std.mem.eql(u8, tool.step, "bench")) gpu.attach(mod);
        const tool_exe = b.addExecutable(.{ .name = tool.step, .root_module = mod });
        const tool_run = b.addRunArtifact(tool_exe);
        tool_run.setCwd(b.path("."));
        tool_run.addPassthruArgs();
        setGpuEnv(b, tool_run);
        b.step(tool.step, tool.desc).dependOn(&tool_run.step);
        const tool_tests = b.addTest(.{ .root_module = mod });
        b.step(b.fmt("test-{s}", .{tool.step}), b.fmt("Tests reachable from {s}", .{tool.root})).dependOn(&b.addRunArtifact(tool_tests).step);
    }

    // SDL_GPU demo: shaders are Zig compiled to SPIR-V and embedded with @embedFile.
    const demo_mod = b.createModule(.{
        .root_source_file = b.path("src/gfx_demo.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "c", .module = c_mod },
            .{ .name = "gpu_layout", .module = layout_mod },
        },
    });
    demo_mod.addIncludePath(b.path("vendor"));
    demo_mod.addCSourceFile(.{ .file = b.path("vendor/impl.c"), .flags = &.{"-O2"} });
    demo_mod.linkLibrary(sdl_lib);
    addShaders(b, demo_mod, layout_mod);
    const demo_exe = b.addExecutable(.{ .name = "gfx-demo", .root_module = demo_mod });
    const demo_run = b.addRunArtifact(demo_exe);
    demo_run.setCwd(b.path("."));
    demo_run.addPassthruArgs();
    setGpuEnv(b, demo_run);
    b.step("gfx-demo", "SDL_GPU sprite demo; saves .scratch/gfx.bmp").dependOn(&demo_run.step);
}

/// Selects the dozen (Vulkan-on-D3D12) driver on WSL when it is installed under $HOME/.local/mesa-dzn.
fn setGpuEnv(b: *std.Build, run: *std.Build.Step.Run) void {
    const home = b.graph.environ_map.get("HOME") orelse return;
    const icd = b.fmt("{s}/.local/mesa-dzn/share/vulkan/icd.d/dzn_icd.x86_64.json", .{home});
    std.Io.Dir.accessAbsolute(b.graph.io, icd, .{}) catch return;
    run.setEnvironmentVariable("VK_DRIVER_FILES", icd);
    run.setEnvironmentVariable("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA");
    // dozen prints a non-conformance warning to stderr, which makes `zig build test` fail.
    run.setEnvironmentVariable("MESA_VK_IGNORE_CONFORMANCE_WARNING", "1");
}

/// Render shaders used by gfx.zig. Any module that imports gfx.zig needs these.
pub fn addShaders(b: *std.Build, mod: *std.Build.Module, layout_mod: *std.Build.Module) void {
    addShader(b, mod, layout_mod, "src/shaders/sprite.zig", "sprite.spv");
    addShader(b, mod, layout_mod, "src/shaders/balls.zig", "balls.spv");
}

/// Compiles a Zig shader to SPIR-V (self-hosted backend; LLVM has no SPIR-V shader support)
/// and makes it available to `mod` as `@embedFile(import_name)`. The shader can
/// `@import("gpu_layout")` so host and shader share struct layouts.
pub fn addShader(b: *std.Build, mod: *std.Build.Module, layout_mod: *std.Build.Module, path: []const u8, import_name: []const u8) void {
    const obj = b.addObject(.{
        .name = std.fs.path.stem(path),
        .use_llvm = false,
        .root_module = b.createModule(.{
            .root_source_file = b.path(path),
            // The baseline target stamps SPIR-V 1.0 yet emits 1.4+ opcodes (OpCopyLogical) and the
            // StorageBuffer class without its extension; 1.5 is valid and is core in Vulkan 1.2,
            // which gfx.zig requests. spirv32 keeps the module free of the Int64 and
            // PhysicalStorageBufferAddresses capabilities that spirv64 pulls in.
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .spirv32,
                .os_tag = .vulkan,
                .ofmt = .spirv,
                .cpu_features_add = std.Target.spirv.featureSet(&.{.v1_5}),
            }),
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "gpu_layout", .module = layout_mod }},
        }),
    });
    mod.addAnonymousImport(import_name, .{ .root_source_file = obj.getEmittedBin() });
}

/// GPU simulation modules shared by the game and the bench: the gpu_sim solver, mutual
/// gravity, and the game-logic kernels in src/gpu_game/ (embedded as "gpu_game_<name>.spv").
const GpuModules = struct {
    sim: *std.Build.Module,
    gravity: *std.Build.Module,
    game_spv: [game_kernels.len]std.Build.LazyPath,

    const game_kernels = [_][]const u8{ "spawn", "merge", "act", "conduct", "burn", "emit", "particles" };

    fn create(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, c_mod: *std.Build.Module, layout_mod: *std.Build.Module) GpuModules {
        const sim = gpu_sim_build.addModule(b, .{ .target = target, .optimize = optimize, .c = c_mod, .layout = layout_mod, .dir = b.path("src/gpu_sim") });
        const shader_layout = b.createModule(.{ .root_source_file = b.path("src/gpu_layout.zig") });
        // gravity.zig imports "config"; a copy of config.zig under another path, because one file
        // cannot belong to two modules and the game already imports config.zig by path.
        const wf = b.addWriteFiles();
        const config_mod = b.createModule(.{ .root_source_file = wf.addCopyFile(b.path("src/config.zig"), "config.zig") });
        const grav_spv = spirvObject(b, "gravity_kernels", b.path("src/gpu_sim/gravity_kernels.zig"), shader_layout, null);
        const gravity = b.createModule(.{
            .root_source_file = b.path("src/gpu_sim/gravity.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "c", .module = c_mod },
                .{ .name = "gpu_layout", .module = layout_mod },
                .{ .name = "config", .module = config_mod },
                .{ .name = "gpu_sim", .module = sim },
            },
        });
        gravity.addAnonymousImport("gravity.spv", .{ .root_source_file = grav_spv });
        var m: GpuModules = .{ .sim = sim, .gravity = gravity, .game_spv = undefined };
        inline for (game_kernels, 0..) |name, i| {
            m.game_spv[i] = spirvObject(b, "gpu_game_" ++ name, b.path("src/gpu_game/k_" ++ name ++ ".zig"), shader_layout, config_mod);
        }
        return m;
    }

    fn attach(m: GpuModules, mod: *std.Build.Module) void {
        mod.addImport("gpu_sim", m.sim);
        mod.addImport("gravity", m.gravity);
        inline for (game_kernels, 0..) |name, i| {
            mod.addAnonymousImport("gpu_game_" ++ name ++ ".spv", .{ .root_source_file = m.game_spv[i] });
        }
    }
};

/// Compute kernels: SPIR-V 1.5 via the vulkan_v1_2 CPU model, like src/gpu_sim/build_kernels.zig.
fn spirvObject(b: *std.Build, name: []const u8, root: std.Build.LazyPath, layout: *std.Build.Module, config: ?*std.Build.Module) std.Build.LazyPath {
    const obj = b.addObject(.{
        .name = name,
        .use_llvm = false,
        .root_module = b.createModule(.{
            .root_source_file = root,
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .spirv64,
                .os_tag = .vulkan,
                .ofmt = .spirv,
                .cpu_model = .{ .explicit = &std.Target.spirv.cpu.vulkan_v1_2 },
            }),
            .optimize = .ReleaseFast,
            .imports = if (config) |cm|
                &.{ .{ .name = "gpu_layout", .module = layout }, .{ .name = "config", .module = cm } }
            else
                &.{.{ .name = "gpu_layout", .module = layout }},
        }),
    });
    return obj.getEmittedBin();
}
