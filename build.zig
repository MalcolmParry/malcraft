const std = @import("std");
const Build = std.Build;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mwengine = b.dependency("mwengine", .{
        .target = target,
        .optimize = optimize,
    });

    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "malcraft",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{
                    .name = "mwengine",
                    .module = mwengine.module("mwengine"),
                },
                .{
                    .name = "zigimg",
                    .module = zigimg.module("zigimg"),
                },
            },
        }),
    });

    const znoise = b.dependency("znoise", .{});
    exe.root_module.addImport("znoise", znoise.module("root"));
    exe.linkLibrary(znoise.artifact("FastNoiseLite"));

    const default_render_radius: u32 = if (optimize == .ReleaseFast or optimize == .ReleaseSafe) 64 else 3;
    const default_vrender_radius: u32 = if (optimize == .ReleaseFast or optimize == .ReleaseSafe) 16 else 1;

    const options = b.addOptions();
    options.addOption(bool, "gpu_validation", b.option(bool, "gpu-validation", "") orelse (optimize != .ReleaseFast));
    options.addOption(u32, "render_radius", b.option(u32, "render-radius", "") orelse default_render_radius);
    options.addOption(u32, "vrender_radius", b.option(u32, "render-height", "") orelse default_vrender_radius);
    options.addOption(bool, "render_borders_with_nonexistant_chunks", b.option(bool, "borders", "Should render borders with nonexistant chunks (kind of broken now)") orelse true);
    exe.root_module.addOptions("options", options);

    const res_install = b.addInstallDirectory(.{
        .source_dir = b.path("res"),
        .install_dir = .prefix,
        .install_subdir = "res",
    });

    const exe_install = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&exe_install.step);
    b.getInstallStep().dependOn(&res_install.step);
    try buildShaders(b, b.getInstallStep());

    const run_step = b.step("run", "");
    const run = b.addRunArtifact(exe);
    run.setCwd(.{ .cwd_relative = b.install_prefix });
    if (b.option(bool, "renderdoc", "Enable render doc capture") orelse false)
        run.setEnvironmentVariable("ENABLE_VULKAN_RENDERDOC_CAPTURE", "1");

    run.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run.step);
}

fn buildShaders(b: *Build, build_step: *Build.Step) !void {
    const shaders = @import("src/assets.zon").shaders;

    inline for (std.meta.fields(@TypeOf(shaders))) |field| {
        const data = @field(shaders, field.name);
        const src = b.path(data.src);

        const compile = b.addSystemCommand(&.{"glslangValidator"});
        inline for (std.meta.fields(@TypeOf(data.compile_opts))) |field2| {
            compile.addArg(@field(data.compile_opts, field2.name));
        }

        compile.addArg("-V");
        compile.addFileInput(src);
        compile.addFileArg(src);
        compile.addArg("-o");
        const comp_out = compile.addOutputFileArg(b.fmt("{s}.no-opt", .{data.bin}));

        const opt = b.addSystemCommand(&.{ "spirv-opt", "-O" });
        opt.addFileInput(comp_out);
        opt.addFileArg(comp_out);
        opt.addArg("-o");
        const opt_out = opt.addOutputFileArg(data.bin);
        opt.step.dependOn(&compile.step);

        const install = b.addInstallFile(opt_out, data.bin);
        install.step.dependOn(&opt.step);
        build_step.dependOn(&install.step);
    }
}
