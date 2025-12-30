const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "malcraft",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.getInstallStep().dependOn(&exe.step);

    const run_step = b.step("run", "");
    const run = b.addRunArtifact(exe);
    run_step.dependOn(&run.step);
}
