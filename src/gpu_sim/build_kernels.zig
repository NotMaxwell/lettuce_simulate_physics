//! Build helper: compiles the gpu_sim kernels to SPIR-V and returns the host module
//! (`collide.zig`) with each kernel embedded as "gpu_sim_<name>.spv".
//! Usage from build.zig: `const gpu_sim = @import("src/gpu_sim/build_kernels.zig");`
//! `const sim_mod = gpu_sim.addModule(b, .{ ... });` then `exe_mod.addImport("gpu_sim", sim_mod)`.

const std = @import("std");

pub const kernels = [_][]const u8{ "integrate", "clear", "scan", "scan_add", "scatter", "colored", "gather", "finish" };

pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Optimize,
    /// The translate-C SDL module (imported as "c").
    c: *std.Build.Module,
    /// Host-side module for src/gpu_layout.zig (imported as "gpu_layout"); pass the same module
    /// the rest of the game imports so the file lives in only one module.
    layout: *std.Build.Module,
    /// Directory holding collide.zig and the k_*.zig kernels.
    dir: std.Build.LazyPath,
};

pub fn addModule(b: *std.Build, o: Options) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = o.dir.path(b, "collide.zig"),
        .target = o.target,
        .optimize = o.optimize,
        .imports = &.{
            .{ .name = "c", .module = o.c },
            .{ .name = "gpu_layout", .module = o.layout },
        },
    });
    // vulkan_v1_2 emits SPIR-V 1.5, which validates cleanly (`spirv-val --target-env vulkan1.2`);
    // the generic model emits 1.0 with StorageBuffer pointers that need an undeclared extension.
    const spv_target = b.resolveTargetQuery(.{
        .cpu_arch = .spirv64,
        .os_tag = .vulkan,
        .ofmt = .spirv,
        .cpu_model = .{ .explicit = &std.Target.spirv.cpu.vulkan_v1_2 },
    });
    const shader_layout = b.createModule(.{ .root_source_file = o.layout.root_source_file.? });
    inline for (kernels) |name| {
        const obj = b.addObject(.{
            .name = "gpu_sim_" ++ name,
            .use_llvm = false,
            .root_module = b.createModule(.{
                .root_source_file = o.dir.path(b, "k_" ++ name ++ ".zig"),
                .target = spv_target,
                .optimize = .ReleaseFast,
                .imports = &.{.{ .name = "gpu_layout", .module = shader_layout }},
            }),
        });
        mod.addAnonymousImport("gpu_sim_" ++ name ++ ".spv", .{ .root_source_file = obj.getEmittedBin() });
    }
    return mod;
}
