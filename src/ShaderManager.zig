const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const shader_list = @import("shader_list.zon");

pub const ShaderManager = @This();

shaders: [shader_count]gpu.Shader,

pub fn init(device: gpu.Device, alloc: std.mem.Allocator) !ShaderManager {
    var shaders: [shader_count]gpu.Shader = undefined;
    for (&shaders, 0..) |*shader, i| {
        const shader_file = try std.fs.cwd().openFile(shader_bin_paths[i], .{});
        defer shader_file.close();
        const shader_code = try shader_file.readToEndAlloc(alloc, 1024 * 1024);
        defer alloc.free(shader_code);

        shader.* = try gpu.Shader.fromSpirv(device, shader_kinds[i], @ptrCast(@alignCast(shader_code)), alloc);
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
    var fields: [shader_count]std.builtin.Type.EnumField = undefined;
    for (std.meta.fields(@TypeOf(shader_list)), 0..) |field, i| {
        fields[i] = .{
            .name = field.name,
            .value = i,
        };
    }

    const info: std.builtin.Type = .{
        .@"enum" = .{
            .tag_type = u32,
            .is_exhaustive = true,
            .decls = &.{},
            .fields = &fields,
        },
    };

    break :blk @Type(info);
};

const shader_bin_paths = blk: {
    var result: [shader_count][]const u8 = undefined;

    for (std.meta.fields(@TypeOf(shader_list)), 0..) |field, i| {
        result[i] = @field(shader_list, field.name).bin;
    }

    break :blk result;
};

const shader_kinds = blk: {
    var result: [shader_count]gpu.Shader.Stage = undefined;

    for (std.meta.fields(@TypeOf(shader_list)), 0..) |field, i| {
        result[i] = @field(shader_list, field.name).kind;
    }

    break :blk result;
};
