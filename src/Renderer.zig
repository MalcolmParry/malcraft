const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const Chunk = @import("Chunk.zig");

const Renderer = @This();

event_queue: mw.EventQueue,
window: mw.Window,
instance: gpu.Instance,
device: gpu.Device,
display: gpu.Display,
frame_index: usize,
per_frame_in_flight: []PerFrameInFlight,
chunk: Chunk,
mesh: ChunkMesh,
chunk_shader_vertex: gpu.Shader,
chunk_shader_pixel: gpu.Shader,
chunk_shader_set: gpu.Shader.Set,
chunk_pipeline: gpu.GraphicsPipeline,

pub fn init(this: *@This(), alloc: std.mem.Allocator) !void {
    this.event_queue = try .init(alloc);
    errdefer this.event_queue.deinit();

    this.window = try .init(alloc, "malcraft", .{ 100, 100 }, &this.event_queue);
    errdefer this.window.deinit();

    this.instance = try .init(true, alloc);
    errdefer this.instance.deinit(alloc);

    const phys_device = try this.instance.bestPhysicalDevice();
    this.device = try .init(this.instance, phys_device, alloc);
    errdefer this.device.deinit(alloc);

    this.display = try .init(this.device, &this.window, alloc);
    errdefer this.display.deinit(alloc);

    try this.initFramesInFlight(alloc);
    errdefer this.deinitFramesInFlight(alloc);

    this.chunk.init(.{ 0, 0, 0 });
    try this.mesh.build(&this.chunk, alloc);

    this.chunk_shader_vertex = try makeShader(this, alloc, "res/shaders/chunk_opaque.vert.spv", .vertex);
    errdefer this.chunk_shader_vertex.deinit(this.device, alloc);

    this.chunk_shader_pixel = try makeShader(this, alloc, "res/shaders/chunk_opaque.frag.spv", .pixel);
    errdefer this.chunk_shader_pixel.deinit(this.device, alloc);

    this.chunk_shader_set = try .init(this.device, this.chunk_shader_vertex, this.chunk_shader_pixel, &.{}, alloc);
    errdefer this.chunk_shader_set.deinit(this.device, alloc);

    this.chunk_pipeline = try .init(this.device, .{
        .alloc = alloc,
        .render_target_desc = .{
            .color_format = this.display.imageFormat(),
        },
        .shader_set = this.chunk_shader_set,
        .resource_layouts = &.{},
    });
    errdefer this.chunk_pipeline.deinit(this.device, alloc);
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.mesh.deinit(alloc);

    this.device.waitUntilIdle();
    this.chunk_pipeline.deinit(this.device, alloc);
    this.chunk_shader_set.deinit(this.device, alloc);
    this.chunk_shader_pixel.deinit(this.device, alloc);
    this.chunk_shader_vertex.deinit(this.device, alloc);
    this.deinitFramesInFlight(alloc);
    this.display.deinit(alloc);
    this.device.deinit(alloc);
    this.instance.deinit(alloc);
    this.window.deinit();
    this.event_queue.deinit();
}

pub fn render(this: *@This(), alloc: std.mem.Allocator) !void {
    _ = alloc;
    const per_frame = this.per_frame_in_flight[this.frame_index];
    try per_frame.presented_fence.wait(this.device, std.time.ns_per_s);
    try per_frame.presented_fence.reset(this.device);

    const image_index = switch (try this.display.acquireImageIndex(per_frame.image_available_semaphore, null, std.time.ns_per_s)) {
        .success => |x| x,
        .suboptimal => |x| x,
        .out_of_date => return error.Failed,
    };

    try per_frame.cmd_encoder.begin(this.device);
    per_frame.cmd_encoder.cmdMemoryBarrier(this.device, &.{.{
        .image = .{
            .image = this.display.image(image_index),
            .old_layout = .undefined,
            .new_layout = .color_attachment,
            .src_stage = .{ .pipeline_start = true },
            .dst_stage = .{ .color_attachment_output = true },
            .src_access = .{},
            .dst_access = .{ .color_attachment_write = true },
        },
    }});

    const render_pass = per_frame.cmd_encoder.cmdBeginRenderPass(.{
        .device = this.device,
        .target = .{
            .color_image_view = this.display.imageView(image_index),
            .color_clear_value = .{ 0, 0, 0, 1 },
        },
        .image_size = this.display.imageSize(),
    });

    render_pass.cmdBindPipeline(this.device, this.chunk_pipeline, this.display.imageSize());
    render_pass.cmdDraw(.{
        .device = this.device,
        .vertex_count = 3,
        .indexed = false,
    });

    render_pass.cmdEnd(this.device);

    per_frame.cmd_encoder.cmdMemoryBarrier(this.device, &.{
        .{ .image = .{
            .image = this.display.image(image_index),
            .old_layout = .color_attachment,
            .new_layout = .present_src,
            .src_stage = .{ .color_attachment_output = true },
            .dst_stage = .{ .pipeline_end = true },
            .src_access = .{ .color_attachment_write = true },
            .dst_access = .{},
        } },
    });

    try per_frame.cmd_encoder.end(this.device);
    try per_frame.cmd_encoder.submit(this.device, &.{per_frame.image_available_semaphore}, &.{per_frame.render_finished_semaphore}, null);

    _ = try this.display.presentImage(image_index, &.{per_frame.render_finished_semaphore}, per_frame.presented_fence);
    this.frame_index = (this.frame_index + 1) % this.display.imageCount();
}

fn initFramesInFlight(this: *Renderer, alloc: std.mem.Allocator) !void {
    this.frame_index = 0;
    this.per_frame_in_flight = try alloc.alloc(PerFrameInFlight, this.display.imageCount());
    errdefer alloc.free(this.per_frame_in_flight);

    for (this.per_frame_in_flight) |*x| {
        try x.init(this, alloc);
    }
}

fn deinitFramesInFlight(this: *Renderer, alloc: std.mem.Allocator) void {
    for (this.per_frame_in_flight) |*x| {
        x.deinit(this, alloc);
    }

    alloc.free(this.per_frame_in_flight);
}

const PerFrameInFlight = struct {
    image_available_semaphore: gpu.Semaphore,
    render_finished_semaphore: gpu.Semaphore,
    presented_fence: gpu.Fence,
    cmd_encoder: gpu.CommandEncoder,

    pub fn init(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) !void {
        _ = alloc;

        this.image_available_semaphore = try .init(renderer.device);
        errdefer this.image_available_semaphore.deinit(renderer.device);

        this.render_finished_semaphore = try .init(renderer.device);
        errdefer this.render_finished_semaphore.deinit(renderer.device);

        this.presented_fence = try .init(renderer.device, true);
        errdefer this.presented_fence.deinit(renderer.device);

        this.cmd_encoder = try .init(renderer.device);
        errdefer this.cmd_encoder.deinit(renderer.device);
    }

    pub fn deinit(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) void {
        _ = alloc;

        this.cmd_encoder.deinit(renderer.device);
        this.presented_fence.deinit(renderer.device);
        this.render_finished_semaphore.deinit(renderer.device);
        this.image_available_semaphore.deinit(renderer.device);
    }
};

const ChunkMesh = struct {
    const PerVertex = struct {
        pos: [3]f32,
    };

    per_vertex: []PerVertex,

    const quat_table: [6]math.Quat = .{
        math.quat_identity,
        math.quatFromAxisAngle(math.dir_up, math.rad(180.0)),
        math.quatFromAxisAngle(math.dir_up, math.rad(90.0)),
        math.quatFromAxisAngle(math.dir_up, math.rad(-90.0)),
        math.quatFromAxisAngle(math.dir_right, math.rad(90.0)),
        math.quatFromAxisAngle(math.dir_right, math.rad(-90.0)),
    };

    const offset_table = blk: {
        var result: [6]Chunk.BlockPos = undefined;

        for (quat_table, 0..) |quat, i| {
            result[i] = @intFromFloat(@round(math.quatMulVec(quat, math.dir_forward)));
        }

        break :blk result;
    };

    const forward_face: [6]@Vector(3, f32) = .{
        .{ 1, 0, 0 },
        .{ 1, 1, 0 },
        .{ 1, 1, 1 },

        .{ 1, 0, 0 },
        .{ 1, 1, 1 },
        .{ 1, 0, 1 },
    };

    const face_table = blk: {
        var result: [6][6]@Vector(3, f32) = undefined;

        for (quat_table, 0..) |q, i| {
            for (0..6) |ii| {
                result[i][ii] = math.quatMulVec(q, forward_face[ii]);
            }
        }

        break :blk result;
    };

    fn build(mesh: *ChunkMesh, chunk: *Chunk, alloc: std.mem.Allocator) !void {
        // test other starting values and maybe do prepass to find correct vertex count
        var vertices = try std.ArrayList(PerVertex).initCapacity(alloc, 256);
        errdefer vertices.deinit(alloc);

        var iter: Chunk.Iterator = .{};
        while (iter.next()) |pos| {
            const ipos: Chunk.Pos = pos;

            for (offset_table, 0..) |offset, i| {
                const adjacent = ipos + offset;
                if (chunk.isOpaqueSafe(adjacent)) continue;

                const face_vectors = face_table[i];
                var face: [6]PerVertex = undefined;
                for (face_vectors, 0..) |face_v, ii| {
                    face[ii].pos = math.toArray(face_v + @as(math.Vec3, @floatFromInt(pos)));
                }

                try vertices.appendSlice(alloc, &face);
            }
        }

        mesh.per_vertex = try vertices.toOwnedSlice(alloc);
    }

    fn deinit(mesh: *ChunkMesh, alloc: std.mem.Allocator) void {
        alloc.free(mesh.per_vertex);
    }
};

fn makeShader(this: *Renderer, alloc: std.mem.Allocator, file_name: []const u8, t: gpu.Shader.Stage) !gpu.Shader {
    const shader_file = try std.fs.cwd().openFile(file_name, .{});
    defer shader_file.close();
    const shader_code = try shader_file.readToEndAlloc(alloc, 1024 * 1024);
    defer alloc.free(shader_code);

    var shader = try gpu.Shader.fromSpirv(this.device, t, @ptrCast(@alignCast(shader_code)), alloc);
    errdefer shader.deinit(this.device, alloc);

    return shader;
}
