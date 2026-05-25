const std = @import("std");
const zigimg = @import("zigimg");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const block = @import("../common/block.zig");
const TextureManager = @This();

image: gpu.Image,
view: gpu.Image.View,
sampler: gpu.Sampler,

pub const Id = enum(u3) {
    stone,
    grass,
    water,
    sand,
    missing,

    pub fn getFilePath(id: Id) []const u8 {
        return switch (id) {
            .stone => "res/textures/stone.png",
            .grass => "res/textures/grass.png",
            .water => "res/textures/water.png",
            .sand => "res/textures/sand.png",
            .missing => "res/textures/missing.png",
        };
    }

    pub fn fromBlockId(id: block.Kind) Id {
        return switch (id) {
            .air => .missing,
            .stone => .stone,
            .grass => .grass,
            .water => .water,
            .sand => .sand,
            _ => .missing,
        };
    }
};

pub fn init(alloc: std.mem.Allocator, device: gpu.Device, stage_man: *gpu.StagingManager) !TextureManager {
    const Pixel = [4]u8;

    const mip_levels = 5;
    const size: gpu.Image.Size2D = .{ 16, 16 };
    const pixel_count = @reduce(.Mul, size);
    const layer_size = pixel_count * @sizeOf(Pixel);
    const layer_count: u32 = @intCast(std.enums.values(Id).len);

    const image = try device.initImage(.{
        .alloc = alloc,
        .format = .rgba8_srgb,
        .usage = .{
            .src = true,
            .dst = true,
            .sampled = true,
        },
        .loc = .device,
        .layer_count = layer_count,
        .mip_count = mip_levels,
        .size = size,
    });
    errdefer image.deinit(device, alloc);

    const sampler = try device.initSampler(.{
        .alloc = alloc,
        .min_filter = .nearest,
        .mag_filter = .nearest,
        .address_mode_u = .repeat,
        .address_mode_v = .repeat,
        .address_mode_w = .repeat,
    });
    errdefer sampler.deinit(device, alloc);

    const view = try device.initImageView(.{
        .alloc = alloc,
        .kind = .array_2d,
        .image = image,
        .subresource_range = .{
            .aspect = .{ .color = true },
        },
    });
    errdefer view.deinit(device, alloc);

    const staging = try stage_man.allocateBytesAligned(layer_size * layer_count, .@"4");
    defer stage_man.reset();

    for (std.enums.values(Id), 0..) |id, i| {
        const path = id.getFilePath();

        var read_buffer: [zigimg.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
        var image_cpu = try zigimg.Image.fromFilePath(alloc, path, &read_buffer);
        defer image_cpu.deinit(alloc);

        var cropped = try image_cpu.crop(alloc, .{ .width = size[0], .height = size[1] });
        defer cropped.deinit(alloc);
        try cropped.convert(alloc, .rgba32);

        @memcpy(staging.slice[layer_size * i .. layer_size * (i + 1)], cropped.rawBytes());
    }

    const cmd_encoder = try device.initCommandEncoder();
    defer cmd_encoder.deinit(device);

    try cmd_encoder.begin();

    cmd_encoder.cmdMemoryBarrier(.{
        .image_barriers = &.{.{
            .image = image,
            .subresource_range = .{
                .aspect = .{ .color = true },
            },
            .old_layout = .undefined,
            .new_layout = .transfer_dst,
            .src_stage = .{ .pipeline_start = true },
            .dst_stage = .{ .transfer = true },
            .src_access = .{},
            .dst_access = .{ .transfer_write = true },
        }},
    });

    cmd_encoder.cmdCopyBufferToImage(.{
        .region = .{
            .size = .{ size[0], size[1], 1 },
        },
        .src = staging.region,
        .dst = image,
        .layout = .transfer_dst,
        .subresource = .{
            .aspect = .{ .color = true },
            .layer_count = layer_count,
        },
    });

    var mip_size: gpu.Image.Size2D = size;
    for (1..mip_levels) |level| {
        const old_mip_size = mip_size;
        if (mip_size[0] > 1) mip_size[0] /= 2;
        if (mip_size[1] > 1) mip_size[1] /= 2;

        cmd_encoder.cmdMemoryBarrier(.{
            .image_barriers = &.{.{
                .image = image,
                .subresource_range = .{
                    .aspect = .{ .color = true },
                    .mip_offset = @intCast(level - 1),
                    .mip_count = 1,
                },
                .old_layout = .transfer_dst,
                .new_layout = .transfer_src,
                .src_stage = .{ .transfer = true },
                .dst_stage = .{ .transfer = true },
                .src_access = .{ .transfer_write = true },
                .dst_access = .{ .transfer_read = true },
            }},
        });

        cmd_encoder.cmdCopyImageWithScaling(.{
            .filter = .linear,
            .src = image,
            .src_layout = .transfer_src,
            .src_subresource = .{
                .aspect = .{ .color = true },
                .mip_level = @intCast(level - 1),
                .layer_count = layer_count,
            },
            .src_rect = .{ .size = old_mip_size },
            .dst = image,
            .dst_layout = .transfer_dst,
            .dst_subresource = .{
                .aspect = .{ .color = true },
                .mip_level = @intCast(level),
                .layer_count = layer_count,
            },
            .dst_rect = .{ .size = mip_size },
        });

        cmd_encoder.cmdMemoryBarrier(.{
            .image_barriers = &.{.{
                .image = image,
                .subresource_range = .{
                    .aspect = .{ .color = true },
                    .mip_offset = @intCast(level - 1),
                    .mip_count = 1,
                },
                .old_layout = .transfer_src,
                .new_layout = .shader_read_only,
                .src_stage = .{ .transfer = true },
                .dst_stage = .{ .pixel_shader = true },
                .src_access = .{ .transfer_read = true },
                .dst_access = .{ .shader_read = true },
            }},
        });
    }

    cmd_encoder.cmdMemoryBarrier(.{
        .image_barriers = &.{.{
            .image = image,
            .subresource_range = .{
                .aspect = .{ .color = true },
                .mip_offset = mip_levels - 1,
                .mip_count = 1,
            },
            .old_layout = .transfer_dst,
            .new_layout = .shader_read_only,
            .src_stage = .{ .transfer = true },
            .dst_stage = .{ .pixel_shader = true },
            .src_access = .{ .transfer_write = true },
            .dst_access = .{ .shader_read = true },
        }},
    });

    try cmd_encoder.end();

    const timeline = try device.initTimeline(0);
    defer timeline.deinit(device);

    try device.submitCommands(.{
        .encoder = cmd_encoder,
        .signals = &.{.{
            .timeline = timeline,
            .value = 1,
            .stages = .{ .all_commands = true },
        }},
    });

    try timeline.wait(device, 1, std.time.ns_per_s);

    return .{
        .image = image,
        .view = view,
        .sampler = sampler,
    };
}

pub fn deinit(man: *TextureManager, alloc: std.mem.Allocator, device: gpu.Device) void {
    man.sampler.deinit(device, alloc);
    man.view.deinit(device, alloc);
    man.image.deinit(device, alloc);
}
