const std = @import("std");
const Build = std.Build;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const client_step = b.step("client", "build the client");
    var client_dep_steps: std.ArrayList(*Build.Step) = .empty;
    const client = try buildClient(b, target, optimize, &client_dep_steps);
    for (client_dep_steps.items) |step| client_step.dependOn(step);
    b.getInstallStep().dependOn(client_step);

    const server_step = b.step("server", "build the server");
    var server_dep_steps: std.ArrayList(*Build.Step) = .empty;
    const server = try buildServer(b, target, optimize, &server_dep_steps);
    for (server_dep_steps.items) |step| server_step.dependOn(step);

    const client_run_step = b.step("client-run", "build and run the client");
    const client_run = b.addRunArtifact(client);
    client_run.step.dependOn(client_step);
    client_run_step.dependOn(&client_run.step);
    client_run.setCwd(.{ .cwd_relative = b.install_prefix });
    if (b.option(bool, "renderdoc", "enable render doc capture") orelse false)
        client_run.setEnvironmentVariable("ENABLE_VULKAN_RENDERDOC_CAPTURE", "1");

    const server_run_step = b.step("server-run", "build and run the server");
    const server_run = b.addRunArtifact(server);
    server_run.step.dependOn(server_step);
    server_run_step.dependOn(&server_run.step);
}

fn buildClient(b: *Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, step_list: *std.ArrayList(*Build.Step)) !*Build.Step.Compile {
    const mwengine = b.dependency("mwengine", .{
        .target = target,
        .optimize = optimize,
    });

    const znet = b.dependency("znet", .{
        .target = target,
        .optimize = optimize,
    });

    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    const client = b.addExecutable(.{
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
                    .name = "znet",
                    .module = znet.module("znet"),
                },
                .{
                    .name = "zigimg",
                    .module = zigimg.module("zigimg"),
                },
            },
        }),
    });

    const zstd = b.dependency("zstd", .{
        .target = target,
        .optimize = optimize,
    });
    client.root_module.linkLibrary(zstd.artifact("zstd"));
    client.root_module.addIncludePath(zstd.path("lib/"));

    const default_render_radius: u32 = if (optimize == .ReleaseFast or optimize == .ReleaseSafe) 64 else 3;
    const default_render_height: u32 = if (optimize == .ReleaseFast or optimize == .ReleaseSafe) 16 else 8;

    const options = b.addOptions();
    options.addOption(bool, "gpu_validation", b.option(bool, "gpu-validation", "") orelse (optimize != .ReleaseFast));
    options.addOption(u32, "render_radius", default_render_radius);
    options.addOption(u32, "render_height", default_render_height);
    options.addOption(bool, "render_borders_with_nonexistant_chunks", b.option(bool, "borders", "Should render borders with nonexistant chunks (kind of broken now)") orelse true);
    client.root_module.addOptions("options", options);

    const exe_install = b.addInstallArtifact(client, .{});
    const res_install = b.addInstallDirectory(.{
        .source_dir = b.path("res"),
        .install_dir = .prefix,
        .install_subdir = "res",
    });

    try buildShaders(b, step_list);
    try step_list.append(b.allocator, &exe_install.step);
    try step_list.append(b.allocator, &res_install.step);

    return client;
}

fn buildServer(b: *Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, step_list: *std.ArrayList(*Build.Step)) !*Build.Step.Compile {
    const mwengine = b.dependency("mwengine", .{
        .target = target,
        .optimize = optimize,
        .gpu = false,
        .renderer = false,
        .windowing = false,
    });

    const znet = b.dependency("znet", .{
        .target = target,
        .optimize = optimize,
    });

    const server = b.addExecutable(.{
        .name = "server",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/server_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{
                    .name = "mwengine",
                    .module = mwengine.module("mwengine"),
                },
                .{
                    .name = "znet",
                    .module = znet.module("znet"),
                },
            },
        }),
    });

    const zstd = b.dependency("zstd", .{
        .target = target,
        .optimize = optimize,
    });
    server.root_module.linkLibrary(zstd.artifact("zstd"));
    server.root_module.addIncludePath(zstd.path("lib/"));

    const znoise = b.dependency("znoise", .{
        .target = target,
        .optimize = optimize,
    });
    server.root_module.addImport("znoise", znoise.module("root"));
    server.root_module.linkLibrary(znoise.artifact("FastNoiseLite"));

    const default_render_radius: u32 = if (optimize == .ReleaseFast or optimize == .ReleaseSafe) 64 else 3;
    const default_render_height: u32 = if (optimize == .ReleaseFast or optimize == .ReleaseSafe) 16 else 8;

    const options = b.addOptions();
    options.addOption(u32, "render_radius", default_render_radius);
    options.addOption(u32, "render_height", default_render_height);
    server.root_module.addOptions("options", options);

    const exe_install = b.addInstallArtifact(server, .{});
    try step_list.append(b.allocator, &exe_install.step);

    return server;
}

const ShaderStage = enum {
    vertex,
    pixel,
};

fn buildShaders(b: *Build, step_list: *std.ArrayList(*Build.Step)) !void {
    const shaders = @import("src/client/shader_list.zon");
    const base_bin_path = "res/shaders/";

    inline for (std.meta.fields(@TypeOf(shaders))) |field| {
        const entry = @field(shaders, field.name);
        const Entry = @TypeOf(entry);

        const src = b.path(entry.src);

        const compile_opts_raw = if (@hasField(Entry, "compile_opts")) entry.compile_opts else .{};
        const compile_opts: [std.meta.fields(@TypeOf(compile_opts_raw)).len][]const u8 = compile_opts_raw;

        const stage: ShaderStage = entry.stage;

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

        const opt = b.addSystemCommand(&.{ "spirv-opt", "-O" });
        opt.addFileInput(comp_out);
        opt.addFileArg(comp_out);
        opt.addArg("-o");
        const opt_out = opt.addOutputFileArg(field.name);
        opt.step.dependOn(&compile.step);

        const install = b.addInstallFile(opt_out, b.fmt("{s}/{s}.spv", .{ base_bin_path, field.name }));
        install.step.dependOn(&opt.step);
        try step_list.append(b.allocator, &install.step);
    }
}
