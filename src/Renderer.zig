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
frame_timer: std.time.Timer,
last_cursor: @Vector(2, f32),

chunk: Chunk,
mesh: ChunkMesh,
mesh_on_gpu: ChunkMesh.OnGpu,

chunk_shader_vertex: gpu.Shader,
chunk_shader_pixel: gpu.Shader,
chunk_shader_set: gpu.Shader.Set,
chunk_pipeline: gpu.GraphicsPipeline,
camera: struct {
    pos: math.Vec3,
    euler: math.Vec3,
},

pub fn init(this: *@This(), alloc: std.mem.Allocator) !void {
    this.event_queue = try .init(alloc);
    errdefer this.event_queue.deinit();

    this.window = try .init(alloc, "malcraft", .{ 100, 100 }, &this.event_queue);
    errdefer this.window.deinit();
    try this.window.setCursorMode(.disabled);

    this.instance = try .init(true, alloc);
    errdefer this.instance.deinit(alloc);

    const phys_device = try this.instance.bestPhysicalDevice();
    this.device = try .init(this.instance, phys_device, alloc);
    errdefer this.device.deinit(alloc);

    this.display = try .init(this.device, &this.window, alloc);
    errdefer this.display.deinit(alloc);

    try this.initFramesInFlight(alloc);
    errdefer this.deinitFramesInFlight(alloc);

    // transition depth
    {
        const transitons = try alloc.alloc(gpu.CommandEncoder.MemoryBarrier, this.per_frame_in_flight.len);
        defer alloc.free(transitons);

        for (transitons, this.per_frame_in_flight) |*transition, *per_frame| {
            transition.* = .{ .image = .{
                .image = per_frame.depth_image,
                .aspect = .{ .depth = true },
                .old_layout = .undefined,
                .new_layout = .depth_stencil,
                .src_stage = .{ .pipeline_start = true },
                .dst_stage = .{ .early_depth_tests = true },
                .src_access = .{},
                .dst_access = .{
                    .depth_stencil_read = true,
                    .depth_stencil_write = true,
                },
            } };
        }

        var cmd_encoder = try this.device.initCommandEncoder();
        defer cmd_encoder.deinit(this.device);

        var fence = try this.device.initFence(false);
        defer fence.deinit(this.device);

        try cmd_encoder.begin(this.device);
        cmd_encoder.cmdMemoryBarrier(this.device, transitons);
        try cmd_encoder.end(this.device);
        try cmd_encoder.submit(this.device, &.{}, &.{}, fence);
        try fence.wait(this.device, std.time.ns_per_s);
    }

    this.chunk.init(.{ 0, 0, 0 });
    try this.mesh.build(&this.chunk, alloc);
    try this.mesh_on_gpu.init(this.device, &this.mesh, alloc);
    errdefer this.mesh_on_gpu.deinit(this.device, alloc);

    this.chunk_shader_vertex = try makeShader(this, alloc, "res/shaders/chunk_opaque.vert.spv", .vertex);
    errdefer this.chunk_shader_vertex.deinit(this.device, alloc);

    this.chunk_shader_pixel = try makeShader(this, alloc, "res/shaders/chunk_opaque.frag.spv", .pixel);
    errdefer this.chunk_shader_pixel.deinit(this.device, alloc);

    this.chunk_shader_set = try .init(this.device, this.chunk_shader_vertex, this.chunk_shader_pixel, &.{
        .float32x3,
        .float32x3,
        .float32x3,
    }, alloc);
    errdefer this.chunk_shader_set.deinit(this.device, alloc);

    this.chunk_pipeline = try .init(this.device, .{
        .alloc = alloc,
        .render_target_desc = .{
            .color_format = this.display.imageFormat(),
            .depth_format = .d32_sfloat,
        },
        .shader_set = this.chunk_shader_set,
        .push_constant_ranges = &.{.{
            .stages = .{ .vertex = true },
            .offset = 0,
            .size = 64,
        }},
        .cull_mode = .front,
        .depth_mode = .{
            .testing = true,
            .writing = true,
            .compare_op = .less,
        },
    });
    errdefer this.chunk_pipeline.deinit(this.device, alloc);

    this.frame_timer = try .start();
    this.camera = .{
        .pos = .{ 0, 0, 23 },
        .euler = .{ 0, 0, 0 },
    };
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.mesh.deinit(alloc);

    this.device.waitUntilIdle();
    this.mesh_on_gpu.deinit(this.device, alloc);
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

    const viewport = this.window.getFramebufferSize();
    const viewport_f: math.Vec2 = @floatFromInt(viewport);
    const aspect_ratio = viewport_f[0] / viewport_f[1];
    const dt_ns = this.frame_timer.lap();
    const dt = @as(f32, @floatFromInt(dt_ns)) / std.time.ns_per_s;

    {
        var move_vector: math.Vec3 = @splat(0);

        if (this.window.isKeyDown(.w))
            move_vector += math.dir_forward;
        if (this.window.isKeyDown(.s))
            move_vector -= math.dir_forward;
        if (this.window.isKeyDown(.a))
            move_vector -= math.dir_right;
        if (this.window.isKeyDown(.d))
            move_vector += math.dir_right;
        if (this.window.isKeyDown(.e))
            move_vector += math.dir_up;
        if (this.window.isKeyDown(.q))
            move_vector -= math.dir_up;

        if (!math.eql(move_vector, @as(math.Vec3, @splat(0)))) {
            const q = math.quatFromEuler(this.camera.euler);
            move_vector = math.quatMulVec(q, move_vector);
            move_vector = math.normalize(move_vector);
            move_vector *= @splat(dt * 15);
            this.camera.pos += move_vector;
        }
    }

    {
        const cursor = this.window.getCursorPos();
        var moved = cursor - this.last_cursor;
        this.last_cursor = cursor;

        if (!math.eql(moved, @as(math.Vec2, @splat(0)))) {
            moved *= @splat(math.rad(0.18));
            this.camera.euler += math.Vec3{ 0, moved[1], moved[0] };
        }
    }

    const vp_mat = math.matMulMany(.{
        math.perspective(aspect_ratio, math.rad(90.0), 0.1, 1000),
        math.rotateEuler(this.camera.euler),
        math.translate(-this.camera.pos),
    });

    const push_constants = math.toArray(vp_mat);

    const image_index = switch (try this.display.acquireImageIndex(per_frame.image_available_semaphore, null, std.time.ns_per_s)) {
        .success => |x| x,
        .suboptimal => |x| x,
        .out_of_date => return error.Failed,
    };

    try per_frame.cmd_encoder.begin(this.device);
    per_frame.cmd_encoder.cmdMemoryBarrier(this.device, &.{
        .{ .image = .{
            .image = this.display.image(image_index),
            .aspect = .{ .color = true },
            .old_layout = .undefined,
            .new_layout = .color_attachment,
            .src_stage = .{ .pipeline_start = true },
            .dst_stage = .{ .color_attachment_output = true },
            .src_access = .{},
            .dst_access = .{ .color_attachment_write = true },
        } },
    });

    const render_pass = per_frame.cmd_encoder.cmdBeginRenderPass(.{
        .device = this.device,
        .target = .{
            .color_image_view = this.display.imageView(image_index),
            .color_clear_value = @as(math.Vec4, .{ 66.0, 130.0, 250.0, 255.0 }) / @as(math.Vec4, @splat(255.0)),
            .depth_image_view = per_frame.depth_image_view,
        },
        .image_size = this.display.imageSize(),
    });

    render_pass.cmdBindPipeline(this.device, this.chunk_pipeline, this.display.imageSize());
    render_pass.cmdPushConstants(this.device, this.chunk_pipeline, .{
        .stages = .{ .vertex = true },
        .offset = 0,
        .size = 64,
    }, @ptrCast(&push_constants));
    render_pass.cmdBindVertexBuffer(this.device, this.mesh_on_gpu.vertex_buffer.region());

    render_pass.cmdDraw(.{
        .device = this.device,
        .vertex_count = @intCast(this.mesh.per_vertex.len),
        .indexed = false,
    });

    render_pass.cmdEnd(this.device);

    per_frame.cmd_encoder.cmdMemoryBarrier(this.device, &.{
        .{ .image = .{
            .image = this.display.image(image_index),
            .aspect = .{ .color = true },
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
    depth_image: gpu.Image,
    depth_image_view: gpu.Image.View,

    pub fn init(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) !void {
        this.image_available_semaphore = try .init(renderer.device);
        errdefer this.image_available_semaphore.deinit(renderer.device);

        this.render_finished_semaphore = try .init(renderer.device);
        errdefer this.render_finished_semaphore.deinit(renderer.device);

        this.presented_fence = try .init(renderer.device, true);
        errdefer this.presented_fence.deinit(renderer.device);

        this.cmd_encoder = try .init(renderer.device);
        errdefer this.cmd_encoder.deinit(renderer.device);

        this.depth_image = try .init(renderer.device, .{
            .alloc = alloc,
            .format = .d32_sfloat,
            .usage = .{
                .depth_stencil_attachment = true,
            },
            .loc = .device,
            .size = renderer.display.imageSize(),
        });
        errdefer this.depth_image.deinit(renderer.device, alloc);

        this.depth_image_view = try .init(
            renderer.device,
            this.depth_image,
            .{ .depth = true },
            alloc,
        );
        errdefer this.depth_image_view.deinit(renderer.device, alloc);
    }

    pub fn deinit(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) void {
        this.depth_image_view.deinit(renderer.device, alloc);
        this.depth_image.deinit(renderer.device, alloc);
        this.cmd_encoder.deinit(renderer.device);
        this.presented_fence.deinit(renderer.device);
        this.render_finished_semaphore.deinit(renderer.device);
        this.image_available_semaphore.deinit(renderer.device);
    }
};

const ChunkMesh = struct {
    const OnGpu = struct {
        vertex_buffer: gpu.Buffer,

        fn init(this: *OnGpu, device: gpu.Device, mesh: *ChunkMesh, alloc: std.mem.Allocator) !void {
            this.vertex_buffer = try .init(device, .{
                .alloc = alloc,
                .loc = .device,
                .usage = .{
                    .vertex = true,
                    .dst = true,
                },
                .size = mesh.per_vertex.len * @sizeOf(PerVertex),
            });
            errdefer this.vertex_buffer.deinit(device, alloc);

            try device.setBufferRegions(
                &.{this.vertex_buffer.region()},
                &.{std.mem.sliceAsBytes(mesh.per_vertex)},
            );
        }

        fn deinit(this: *OnGpu, device: gpu.Device, alloc: std.mem.Allocator) void {
            this.vertex_buffer.deinit(device, alloc);
        }
    };

    const PerVertex = struct {
        pos: [3]f32,
        color: [3]f32,
        normal: [3]f32,
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

    const normal_face: [6]@Vector(3, f32) = .{
        .{ 0, -0.5, -0.5 },
        .{ 0, 0.5, -0.5 },
        .{ 0, 0.5, 0.5 },

        .{ 0, -0.5, -0.5 },
        .{ 0, 0.5, 0.5 },
        .{ 0, -0.5, 0.5 },
    };

    const face_table = blk: {
        var result: [6][6]@Vector(3, f32) = undefined;

        for (quat_table, 0..) |q, i| {
            for (0..6) |ii| {
                const f_offset: math.Vec3 = @floatFromInt(offset_table[i]);
                const half_offset = f_offset / @as(math.Vec3, @splat(2.0));

                var vert = normal_face[ii];
                vert = math.quatMulVec(q, vert);
                vert += half_offset;
                result[i][ii] = vert;
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
                const block = chunk.getBlock(pos);
                if (block == .air) continue;
                if (chunk.isOpaqueSafe(adjacent)) continue;

                const face_vectors = face_table[i];
                var face: [6]PerVertex = undefined;
                for (face_vectors, 0..) |face_v, ii| {
                    face[ii] = .{
                        .pos = math.toArray(face_v + @as(math.Vec3, @floatFromInt(pos))),
                        .color = math.toArray(block.color()),
                        .normal = math.toArray(@as(math.Vec3, @floatFromInt(offset))),
                    };
                }

                try vertices.appendSlice(alloc, &face);
            }
        }

        mesh.per_vertex = try vertices.toOwnedSlice(alloc);
        std.log.info("vertices {}", .{mesh.per_vertex.len});
        std.log.info("faces {}", .{mesh.per_vertex.len / 6});
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
