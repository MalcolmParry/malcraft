const std = @import("std");
const options = @import("options");
const zigimg = @import("zigimg");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const Camera = @import("Camera.zig");
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const ChunkMeshAllocator = @import("ChunkMeshAllocator.zig");
const ShaderManager = @import("ShaderManager.zig");
const Renderer = @import("Renderer.zig");
const UIRenderer = @This();

const dt_hist_size = 256;

immediate: mw.ImmediateRenderer,
font_face: mw.text.Face,
glyph_cache: mw.text.GlyphCache,

dt_hist: [dt_hist_size]u64,
dt_hist_scratch: [dt_hist_size]u64,

pub const InitInfo = struct {
    alloc: std.mem.Allocator,
    device: gpu.Device,
    stage_man: *mw.gpu.StagingManager,
    shader_man: *ShaderManager,
    color_format: gpu.Image.Format,
    frames_in_flight: u32,
};

pub fn init(ui: *UIRenderer, info: InitInfo) !void {
    const alloc = info.alloc;

    // const font_path = "res/fonts/Roboto_Condensed-Regular.ttf";
    // const font_path = "res/fonts/NFPixels-Regular.ttf";
    const font_path = "res/fonts/press-start-2p/PressStart2P-vaV7.ttf";
    var face: mw.text.Face = try .init(font_path);
    errdefer face.deinit();

    var cache: mw.text.GlyphCache = .{
        .alloc = alloc,
        .stage_man = info.stage_man,
        .atlas_size = @splat(512),
    };
    errdefer cache.deinit(info.device);

    ui.* = .{
        .immediate = undefined,
        .font_face = face,
        .glyph_cache = cache,
        .dt_hist = undefined,
        .dt_hist_scratch = undefined,
    };

    @memset(&ui.dt_hist, 0);

    ui.immediate = try .init(.{
        .alloc = alloc,
        .device = info.device,
        .frames_in_flight = info.frames_in_flight,
        .streaming_buffer_size_pf = 128 * 1024,
        .color_format = info.color_format,
        .box_info = .{
            .shaders = &.{
                info.shader_man.getShader(.immediate_box_vertex),
                info.shader_man.getShader(.immediate_box_pixel),
            },
        },
        .image_info = .{
            .shaders = &.{
                info.shader_man.getShader(.immediate_image_vertex),
                info.shader_man.getShader(.immediate_image_pixel),
            },
        },
        .text_info = .{
            .glyph_cache = &ui.glyph_cache,
            .shaders = &.{
                info.shader_man.getShader(.immediate_text_vertex),
                info.shader_man.getShader(.immediate_text_pixel),
            },
        },
    });
    errdefer ui.immediate.deinit(info.device);
}

pub fn deinit(ui: *UIRenderer, device: gpu.Device) void {
    ui.immediate.deinit(device);
    ui.glyph_cache.deinit(device);
    ui.font_face.deinit();
}

pub const RenderInfo = struct {
    alloc: std.mem.Allocator,
    device: gpu.Device,

    cmd_encoder: gpu.CommandEncoder,
    image_view: gpu.Image.View,
    viewport: gpu.Image.Size2D,
    frame_count: usize,

    show_crosshair: bool,
    camera: Camera,
    dt_ns: u64,
    chunk_mesh_buffer_bytes_used: usize,
    chunk_mesh_buffer_bytes_total: usize,
    loaded_mesh_count: usize,
    overwritten_meshes: u64,
};

pub fn render(ui: *UIRenderer, info: RenderInfo) !void {
    const viewport_f: math.Vec2 = @floatFromInt(info.viewport);
    const frame_count = info.frame_count;

    try ui.immediate.begin(@as(@Vector(2, u16), @intCast(info.viewport)));

    if (info.show_crosshair) {
        const scale = viewport_f[1] * 0.00035;
        const length_f = @max(11, scale * 100);
        const width_f = @max(3, scale * 7);
        const outline_f = @max(1, scale * 2);

        const outline_color: [4]u8 = .{ 150, 150, 150, 255 };
        const color: [4]u8 = .{ 0, 0, 0, 255 };

        const length = @as(i16, @intFromFloat(length_f)) | 1;
        const width = @as(i16, @intFromFloat(width_f)) | 1;
        const outline: i16 = @intFromFloat(outline_f);
        const outline_2 = outline * 2;

        try ui.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ length + outline_2, width + outline_2 },
                },
                .pivot = @splat(0.5),
            },
            .color = outline_color,
        });
        try ui.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ width + outline_2, length + outline_2 },
                },
                .pivot = @splat(0.5),
            },
            .color = outline_color,
        });

        try ui.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ length, width },
                },
                .pivot = @splat(0.5),
            },
            .color = color,
        });
        try ui.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ width, length },
                },
                .pivot = @splat(0.5),
            },
            .color = color,
        });
    }

    {
        const hist_slot = frame_count % dt_hist_size;
        ui.dt_hist[hist_slot] = info.dt_ns;

        const dt_hist_count = @min(frame_count + 1, dt_hist_size);
        const dt_hist_sorted = ui.dt_hist_scratch[0..dt_hist_count];
        @memcpy(dt_hist_sorted, ui.dt_hist[0..dt_hist_count]);

        std.mem.sort(u64, dt_hist_sorted, {}, struct {
            fn less(_: void, left: u64, right: u64) bool {
                return left < right;
            }
        }.less);

        const median = dt_hist_sorted[dt_hist_count / 2];
        const median_s = math.i2f(f32, median) / std.time.ns_per_s;
        const median_fps = 1 / median_s;

        const low = dt_hist_sorted[dt_hist_count / 10 * 9];
        const low_s = math.i2f(f32, low) / std.time.ns_per_s;
        const low_fps = 1 / low_s;

        const cam_euler = @mod(info.camera.euler, @as(math.Vec3, @splat(math.pi * 2)));

        const text = try std.fmt.allocPrint(info.alloc,
            \\FPS: {d: >4.0}, {d: >5.2}ms
            \\Low: {d: >4.0}, {d: >5.2}ms  (10% Low)
            \\
            \\X:   {d: >8.2}
            \\Y:   {d: >8.2}
            \\Z:   {d: >8.2}
            \\
            \\Yaw:   {d: >6.2}
            \\Pitch: {d: >6.2}
            \\
            \\Mesh Buffer: {Bi:.2} / {Bi:.2}
            \\Meshed Chunks: {}
            \\Overwritten Meshes: {}
        , .{
            median_fps,
            median_s * std.time.ms_per_s,
            low_fps,
            low_s * std.time.ms_per_s,
            info.camera.pos[0],
            info.camera.pos[1],
            info.camera.pos[2],
            math.deg(cam_euler[2]),
            math.deg(cam_euler[1]),
            info.chunk_mesh_buffer_bytes_used,
            info.chunk_mesh_buffer_bytes_total,
            info.loaded_mesh_count,
            info.overwritten_meshes,
        });
        defer info.alloc.free(text);

        try ui.immediate.drawText(.{
            .font_face = &ui.font_face,
            .height_px = 24,
            .height_in_atlas_px = 24,
            .text = text,
            .pos = .{
                .norm = .{ 0, 1 },
                .offset = .{ 8, -8 },
            },
            .color = .{ 0, 0, 0, 255 },
            .outline_width = 1,
            .outline_color = @splat(255),
        });
    }

    try ui.glyph_cache.upload(info.device, info.cmd_encoder);
    try ui.immediate.render(info.device, info.cmd_encoder, info.image_view);
}
