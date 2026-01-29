const std = @import("std");
const Build = std.Build;

const shader_path = "src/shaders/";
const shader_output = "res/shaders/";

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mwengine = b.dependency("mwengine", .{
        .target = target,
        .optimize = optimize,
    }).module("mwengine");

    const exe = b.addExecutable(.{
        .name = "malcraft",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("mwengine", mwengine);

    const exe_install = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&exe_install.step);
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
    const dir = try b.build_root.handle.openDir(shader_path, .{ .iterate = true });
    var iter = try dir.walk(b.allocator);
    defer iter.deinit();

    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".glsl")) continue;

        try buildShader(b, build_step, entry, "vert", &.{"-D_VERTEX"});
        try buildShader(b, build_step, entry, "frag", &.{"-D_PIXEL"});
    }
}

fn buildShader(b: *Build, build_step: *Build.Step, entry: std.fs.Dir.Walker.Entry, t: []const u8, defines: []const []const u8) !void {
    const src = b.path(b.fmt("{s}/{s}", .{ shader_path, entry.path }));
    var name_iter = std.mem.splitScalar(u8, entry.basename, '.');
    const name = name_iter.next() orelse return error.Failed;
    const out_name = b.fmt("{s}.{s}.spv", .{ name, t });

    const command = b.addSystemCommand(&.{ "glslangValidator", "-S", t });
    command.addArgs(defines);
    command.addArg("-V");
    command.addFileInput(src);
    command.addFileArg(src);
    command.addArg("-o");

    const out = command.addOutputFileArg(out_name);
    const install = b.addInstallFile(out, b.fmt("{s}/{s}", .{ shader_output, out_name }));
    install.step.dependOn(&command.step);
    build_step.dependOn(&install.step);
}
