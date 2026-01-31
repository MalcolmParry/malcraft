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

chunks: std.AutoHashMap(Chunk.ChunkPos, *Chunk),
chunks_on_gpu: std.AutoArrayHashMap(Chunk.ChunkPos, ChunkMesh.OnGpu),

chunk_face_lookup: gpu.Buffer,
chunk_resource_layout: gpu.ResourceSet.Layout,
chunk_resource_set: gpu.ResourceSet,
chunk_shader_vertex: gpu.Shader,
chunk_shader_pixel: gpu.Shader,
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
    std.log.info("display size: {}", .{this.display.imageSize()});

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

    this.chunks = .init(alloc);
    this.chunks_on_gpu = .init(alloc);
    const render_radius = 1;
    {
        const diameter = render_radius * 2 + 1;
        const chunks = diameter * diameter;
        var gen_time_ns: usize = 0;
        var mesh_time_ns: usize = 0;
        var timer = try std.time.Timer.start();

        var x: i32 = -render_radius;
        while (x <= render_radius) : (x += 1) {
            var y: i32 = -render_radius;
            while (y <= render_radius) : (y += 1) {
                const pos: Chunk.ChunkPos = .{ x, y, 0 };
                timer.reset();
                const chunk = try alloc.create(Chunk);
                gen_time_ns += timer.read();
                chunk.init(pos);

                try this.chunks.put(pos, chunk);

                var mesh: ChunkMesh = undefined;
                timer.reset();
                try mesh.build(chunk, alloc);
                mesh_time_ns += timer.read();
                defer mesh.deinit(alloc);

                var on_gpu: ChunkMesh.OnGpu = undefined;
                try on_gpu.init(this.device, &mesh, alloc);
                errdefer on_gpu.deinit(this.device, alloc);
                try this.chunks_on_gpu.put(pos, on_gpu);
            }
        }

        std.log.info("average chunk gen time: {} ns", .{@as(f64, @floatFromInt(gen_time_ns)) / @as(f64, @floatFromInt(chunks))});
        std.log.info("average chunk mesh time: {} ns", .{@as(f64, @floatFromInt(mesh_time_ns)) / @as(f64, @floatFromInt(chunks))});
    }

    this.chunk_face_lookup = try this.device.initBuffer(.{
        .alloc = alloc,
        .loc = .device,
        .usage = .{
            .uniform = true,
            .dst = true,
        },
        .size = @sizeOf(@TypeOf(ChunkMesh.face_table)),
    });
    errdefer this.chunk_face_lookup.deinit(this.device, alloc);

    try this.device.setBufferRegions(
        &.{this.chunk_face_lookup.region()},
        &.{std.mem.sliceAsBytes(&ChunkMesh.face_table)},
    );

    this.chunk_resource_layout = try .init(this.device, .{
        .alloc = alloc,
        .descriptors = &.{
            .{
                .t = .uniform,
                .stages = .{ .vertex = true },
                .binding = 0,
                .count = 1,
            },
        },
    });
    errdefer this.chunk_resource_layout.deinit(this.device, alloc);

    this.chunk_resource_set = try .init(this.device, this.chunk_resource_layout, alloc);
    errdefer this.chunk_resource_set.deinit(this.device, alloc);

    try this.chunk_resource_set.update(this.device, &.{.{
        .binding = 0,
        .data = .{ .uniform = &.{this.chunk_face_lookup.region()} },
    }}, alloc);

    this.chunk_shader_vertex = try makeShader(this, alloc, "res/shaders/chunk_opaque.vert.spv", .vertex);
    errdefer this.chunk_shader_vertex.deinit(this.device, alloc);

    this.chunk_shader_pixel = try makeShader(this, alloc, "res/shaders/chunk_opaque.frag.spv", .pixel);
    errdefer this.chunk_shader_pixel.deinit(this.device, alloc);

    this.chunk_pipeline = try .init(this.device, .{
        .alloc = alloc,
        .render_target_desc = .{
            .color_format = this.display.imageFormat(),
            .depth_format = .d32_sfloat,
        },
        .push_constant_ranges = &.{
            .{
                .stages = .{ .vertex = true },
                .offset = 0,
                .size = 64 + 12,
            },
        },
        .resource_layouts = &.{this.chunk_resource_layout},
        .shaders = &.{
            this.chunk_shader_vertex,
            this.chunk_shader_pixel,
        },
        .vertex_input_bindings = &.{.{
            .binding = 0,
            .rate = .per_instance,
            .fields = &.{
                .{ .type = .uint32 },
            },
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
    this.device.waitUntilIdle();

    var chunk_on_gpu_iter = this.chunks_on_gpu.iterator();
    while (chunk_on_gpu_iter.next()) |on_gpu| {
        on_gpu.value_ptr.deinit(this.device, alloc);
    }
    this.chunks_on_gpu.deinit();

    var chunk_iter = this.chunks.iterator();
    while (chunk_iter.next()) |chunk| {
        alloc.destroy(chunk.value_ptr.*);
    }
    this.chunks.deinit();

    this.chunk_face_lookup.deinit(this.device, alloc);
    this.chunk_pipeline.deinit(this.device, alloc);
    this.chunk_shader_pixel.deinit(this.device, alloc);
    this.chunk_shader_vertex.deinit(this.device, alloc);
    this.chunk_resource_set.deinit(this.device, alloc);
    this.chunk_resource_layout.deinit(this.device, alloc);
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
        math.perspective(aspect_ratio, math.rad(90.0), 0.1, 100),
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
    render_pass.cmdBindResourceSets(this.device, this.chunk_pipeline, &.{this.chunk_resource_set}, 0);
    render_pass.cmdPushConstants(this.device, this.chunk_pipeline, .{
        .stages = .{ .vertex = true },
        .offset = 0,
        .size = 64,
    }, @ptrCast(&push_constants));

    var chunk_mesh_iter = this.chunks_on_gpu.iterator();
    while (chunk_mesh_iter.next()) |kv| {
        this.drawChunk(render_pass, kv.key_ptr.*, kv.value_ptr.*);
    }

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

fn drawChunk(this: *Renderer, render_pass: gpu.RenderPassEncoder, pos: Chunk.ChunkPos, on_gpu: ChunkMesh.OnGpu) void {
    const packed_pos: [3]i32 = math.toArray(pos);

    render_pass.cmdPushConstants(
        this.device,
        this.chunk_pipeline,
        .{
            .stages = .{ .vertex = true },
            .offset = 64,
            .size = 12,
        },
        @ptrCast(std.mem.sliceAsBytes(&packed_pos)),
    );

    render_pass.cmdBindVertexBuffer(this.device, 0, on_gpu.vertex_buffer.region());
    render_pass.cmdDraw(.{
        .device = this.device,
        .vertex_count = 6,
        .instance_count = @intCast(on_gpu.face_count),
        .indexed = false,
    });
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

const Face = enum(u3) {
    north,
    south,
    east,
    west,
    up,
    down,

    pub inline fn quat(face: Face) math.Quat {
        return quat_table[@intFromEnum(face)];
    }

    pub inline fn dir(face: Face) Chunk.BlockPos {
        return direction_table[@intFromEnum(face)];
    }

    const quat_table: [6]math.Quat = .{
        math.quat_identity,
        math.quatFromAxisAngle(math.dir_up, math.rad(180.0)),
        math.quatFromAxisAngle(math.dir_up, math.rad(90.0)),
        math.quatFromAxisAngle(math.dir_up, math.rad(-90.0)),
        math.quatFromAxisAngle(math.dir_right, math.rad(-90.0)),
        math.quatFromAxisAngle(math.dir_right, math.rad(90.0)),
    };

    const direction_table: [6]Chunk.BlockPos = .{
        .{ 1, 0, 0 },
        .{ -1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, -1, 0 },
        .{ 0, 0, 1 },
        .{ 0, 0, -1 },
    };
};

const ChunkMesh = struct {
    const OnGpu = struct {
        vertex_buffer: gpu.Buffer,
        face_count: usize,

        fn init(this: *OnGpu, device: gpu.Device, mesh: *ChunkMesh, alloc: std.mem.Allocator) !void {
            this.vertex_buffer = try .init(device, .{
                .alloc = alloc,
                .loc = .device,
                .usage = .{
                    .vertex = true,
                    .dst = true,
                },
                .size = mesh.per_face.len * @sizeOf(PerFace),
            });
            errdefer this.vertex_buffer.deinit(device, alloc);
            this.face_count = mesh.per_face.len;

            try device.setBufferRegions(
                &.{this.vertex_buffer.region()},
                &.{std.mem.sliceAsBytes(mesh.per_face)},
            );
        }

        fn deinit(this: *OnGpu, device: gpu.Device, alloc: std.mem.Allocator) void {
            this.vertex_buffer.deinit(device, alloc);
        }
    };

    const PerFace = packed struct(u32) {
        x: u5,
        y: u5,
        z: u5,
        face: Face,
        block_id: Chunk.BlockId,
        padding: u12 = undefined,
    };

    per_face: []PerFace,

    const normal_face: [6]@Vector(3, f32) = .{
        .{ 0, -0.5, -0.5 },
        .{ 0, 0.5, -0.5 },
        .{ 0, 0.5, 0.5 },

        .{ 0, -0.5, -0.5 },
        .{ 0, 0.5, 0.5 },
        .{ 0, -0.5, 0.5 },
    };

    const face_table = blk: {
        var result: [6][6][4]f32 = undefined;

        for (Face.quat_table, 0..) |q, i| {
            for (0..6) |ii| {
                const f_offset: math.Vec3 = @floatFromInt(Face.direction_table[i]);
                const half_offset = f_offset / @as(math.Vec3, @splat(2.0));

                var vert = normal_face[ii];
                vert = math.quatMulVec(q, vert);
                vert += half_offset;
                const vec4 = math.changeSize(4, vert);
                result[i][ii] = math.toArray(vec4);
            }
        }

        break :blk result;
    };

    fn build(mesh: *ChunkMesh, chunk: *Chunk, alloc: std.mem.Allocator) !void {
        // test other starting values and maybe do prepass to find correct vertex count
        var faces = try std.ArrayList(PerFace).initCapacity(alloc, 5000);
        errdefer faces.deinit(alloc);

        var iter: Chunk.Iterator = .{};
        while (iter.next()) |pos| {
            const block = chunk.getBlock(pos);
            if (block == .air) continue;
            const ipos: Chunk.Pos = pos;

            for (Face.direction_table, 0..) |offset, i| {
                const adjacent = ipos + offset;
                if (chunk.isOpaqueSafe(adjacent)) continue;
                const face: PerFace = .{
                    .x = pos[0],
                    .y = pos[1],
                    .z = pos[2],
                    .face = @enumFromInt(i),
                    .block_id = block,
                };
                try faces.append(alloc, face);
            }
        }

        mesh.per_face = try faces.toOwnedSlice(alloc);
        std.log.info("vertices {}", .{mesh.per_face.len * 6});
        std.log.info("faces {}", .{mesh.per_face.len});
    }

    fn deinit(mesh: *ChunkMesh, alloc: std.mem.Allocator) void {
        alloc.free(mesh.per_face);
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
