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

    const znoise = b.dependency("znoise", .{
        .target = target,
        .optimize = optimize,
    });
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

const ShaderLanguage = enum {
    glsl,
    slang,
};

const ShaderStage = enum {
    vertex,
    pixel,
};

fn buildShaders(b: *Build, build_step: *Build.Step) !void {
    const shaders = @import("src/shader_list.zon");
    const base_bin_path = "res/shaders/";

    inline for (std.meta.fields(@TypeOf(shaders))) |field| {
        const entry = @field(shaders, field.name);
        const Entry = @TypeOf(entry);

        const src = b.path(entry.src);

        const compile_opts_raw = if (@hasField(Entry, "compile_opts")) entry.compile_opts else .{};
        const compile_opts: [std.meta.fields(@TypeOf(compile_opts_raw)).len][]const u8 = compile_opts_raw;

        const language: ShaderLanguage = if (@hasField(Entry, "language")) entry.language else .slang;
        const stage: ShaderStage = entry.stage;

        const compile, const comp_out = blk: switch (language) {
            .glsl => {
                const compile = b.addSystemCommand(&.{"glslangValidator"});
                compile.addArgs(&compile_opts);
                compile.addArg("-V");
                compile.addFileInput(src);
                compile.addFileArg(src);
                compile.addArg("-o");
                const comp_out = compile.addOutputFileArg(b.fmt("{s}.no-opt", .{field.name}));

                break :blk .{ compile, comp_out };
            },
            .slang => {
                const compile = b.addSystemCommand(&.{"slangc"});
                compile.addFileInput(src);
                compile.addFileArg(src);

                compile.addArgs(&compile_opts);
                compile.addArg("-O3");
                compile.addArgs(&.{ "-target", "spirv" });
                compile.addArgs(&.{ "-profile", "spirv_1_3" });
                compile.addArgs(&.{ "-entry", entry.entry });
                compile.addArgs(&.{ "-stage", switch (stage) {
                    .vertex => "vertex",
                    .pixel => "fragment",
                } });

                compile.addArg("-o");
                const comp_out = compile.addOutputFileArg(b.fmt("{s}.no-opt", .{field.name}));

                break :blk .{ compile, comp_out };
            },
        };

        const opt = b.addSystemCommand(&.{ "spirv-opt", "-O" });
        opt.addFileInput(comp_out);
        opt.addFileArg(comp_out);
        opt.addArg("-o");
        const opt_out = opt.addOutputFileArg(field.name);
        opt.step.dependOn(&compile.step);

        const install = b.addInstallFile(opt_out, b.fmt("{s}/{s}.spv", .{ base_bin_path, field.name }));
        install.step.dependOn(&opt.step);
        build_step.dependOn(&install.step);
    }
}
