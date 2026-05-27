const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const shader_list = @import("shader_list.zon");

pub const ShaderManager = @This();

shaders: [shader_count]gpu.Shader,

pub fn init(device: gpu.Device, alloc: std.mem.Allocator, io: std.Io) !ShaderManager {
    var shaders: [shader_count]gpu.Shader = undefined;
    for (&shaders, 0..) |*shader, i| {
        const shader_file = try std.Io.Dir.cwd().openFile(io, shader_bin_paths[i], .{});
        defer shader_file.close(io);
        var reader = shader_file.reader(io, &.{});

        const shader_code = try reader.interface.allocRemaining(alloc, .limited(1024 * 1024));
        defer alloc.free(shader_code);

        shader.* = try gpu.Shader.fromSpirv(device, shader_stages[i], @ptrCast(@alignCast(shader_code)), alloc);
    }

    return .{
        .shaders = shaders,
    };
}

pub fn deinit(man: *ShaderManager, device: gpu.Device, alloc: std.mem.Allocator) void {
    for (&man.shaders) |shader| shader.deinit(device, alloc);
}

pub fn getShader(man: *ShaderManager, id: ShaderId) gpu.Shader {
    return man.shaders[@intFromEnum(id)];
}

const shader_count = @typeInfo(@TypeOf(shader_list)).@"struct".fields.len;
pub const ShaderId = blk: {
    const Tag = u32;

    var names: [shader_count][]const u8 = undefined;
    var values: [shader_count]Tag = undefined;
    for (std.meta.fields(@TypeOf(shader_list)), 0..) |field, i| {
        names[i] = field.name;
        values[i] = i;
    }

    break :blk @Enum(Tag, .exhaustive, &names, &values);
};

const shader_bin_paths = blk: {
    var result: [shader_count][]const u8 = undefined;

    for (std.meta.fields(@TypeOf(shader_list)), 0..) |field, i| {
        result[i] = "res/shaders/" ++ field.name ++ ".spv";
    }

    break :blk result;
};

const shader_stages = blk: {
    var result: [shader_count]gpu.Shader.Stage = undefined;

    for (std.meta.fields(@TypeOf(shader_list)), 0..) |field, i| {
        result[i] = @field(shader_list, field.name).stage;
    }

    break :blk result;
};
