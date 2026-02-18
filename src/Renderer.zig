const std = @import("std");
const options = @import("options");
const zigimg = @import("zigimg");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const Chunk = @import("Chunk.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const ChunkMeshAllocator = @import("ChunkMeshAllocator.zig");
const WorldGenerator = @import("WorldGenerator.zig");

const Renderer = @This();

destruct_queue: std.ArrayList(gpu.AnyObject),
event_queue: mw.EventQueue,
window: mw.Window,
instance: gpu.Instance,
device: gpu.Device,
display: gpu.Display,
images_initialized: []bool,
frame_index: usize,
per_frame_in_flight: []PerFrameInFlight,
frame_timer: std.time.Timer,
total_timer: std.time.Timer,
frame_count: usize,
last_cursor: @Vector(2, f32),
dirty_swapchain: bool,
wireframe: bool,

chunks: Chunk.Map,

world_gen: WorldGenerator,
chunk_mesh_alloc: ChunkMeshAllocator,
chunk_mesher: ChunkMesher,
chunk_resource_layout: gpu.ResourceSet.Layout,
chunk_resource_set: gpu.ResourceSet,
chunk_shader_vertex: gpu.Shader,
chunk_shader_pixel: gpu.Shader,
chunk_pipeline: gpu.GraphicsPipeline,
camera: Camera,

texture_image: gpu.Image,
texture_view: gpu.Image.View,
texture_sampler: gpu.Sampler,

pub const Input = struct {
    wireframe: bool = false,
};

pub fn init(this: *@This(), alloc: std.mem.Allocator) !void {
    this.event_queue = try .init(alloc);
    errdefer this.event_queue.deinit();

    this.window = try .init(alloc, "malcraft", .{ 100, 100 }, &this.event_queue);
    errdefer this.window.deinit();
    try this.window.setCursorMode(.disabled);

    this.instance = try .init(options.gpu_validation, alloc);
    errdefer this.instance.deinit(alloc);

    const phys_device = try this.instance.bestPhysicalDevice();
    this.device = try .init(this.instance, phys_device, alloc);
    errdefer this.device.deinit(alloc);

    this.display = try .init(this.device, &this.window, alloc);
    errdefer this.display.deinit(alloc);
    std.log.info("display size: {}", .{this.display.imageSize()});

    this.images_initialized = try alloc.alloc(bool, this.display.imageCount());
    errdefer alloc.free(this.images_initialized);
    @memset(this.images_initialized, false);

    this.destruct_queue = try .initCapacity(alloc, 32);
    errdefer this.destruct_queue.deinit(alloc);
    errdefer gpu.AnyObject.deinitAllReversed(this.destruct_queue.items, this.device, alloc);

    this.frame_index = 0;
    try this.initFramesInFlight(alloc);
    errdefer this.deinitFramesInFlight(alloc);

    // transition depth
    {
        const transitions = try alloc.alloc(gpu.CommandEncoder.MemoryBarrier, this.per_frame_in_flight.len);
        defer alloc.free(transitions);

        for (transitions, this.per_frame_in_flight) |*transition, *per_frame| {
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
        try cmd_encoder.cmdMemoryBarrier(this.device, transitions, alloc);
        try cmd_encoder.end(this.device);
        try this.device.submitCommands(.{
            .encoder = cmd_encoder,
            .wait_semaphores = &.{},
            .wait_dst_stages = &.{},
            .signal_semaphores = &.{},
            .signal_fence = fence,
        });
        try fence.wait(this.device, std.time.ns_per_s);
    }

    this.chunks = .empty;
    errdefer this.chunks.deinit(alloc);

    try this.world_gen.init(alloc);
    errdefer this.world_gen.deinit();

    try this.chunk_mesh_alloc.init(.{
        .device = this.device,
        .alloc = alloc,
    });
    errdefer this.chunk_mesh_alloc.deinit();

    try this.chunk_mesher.init(.{
        .mesh_alloc = &this.chunk_mesh_alloc,
        .alloc = alloc,
        .chunks = &this.chunks,
    });
    errdefer this.chunk_mesher.deinit();

    try this.loadChunks(alloc);

    this.chunk_resource_layout = try .init(this.device, .{
        .alloc = alloc,
        .descriptors = &.{.{
            .t = .image,
            .stages = .{ .pixel = true },
            .binding = 0,
            .count = 1,
        }},
    });
    try this.destruct_queue.append(alloc, .{ .resource_layout = this.chunk_resource_layout });

    this.chunk_resource_set = try .init(this.device, this.chunk_resource_layout, alloc);
    try this.destruct_queue.append(alloc, .{ .resource_set = this.chunk_resource_set });

    this.chunk_shader_vertex = try makeShader(this, alloc, "res/shaders/chunk_opaque.vert.spv", .vertex);
    try this.destruct_queue.append(alloc, .{ .shader = this.chunk_shader_vertex });

    this.chunk_shader_pixel = try makeShader(this, alloc, "res/shaders/chunk_opaque.frag.spv", .pixel);
    try this.destruct_queue.append(alloc, .{ .shader = this.chunk_shader_pixel });

    this.wireframe = false;
    try this.initChunkPipeline(alloc);
    errdefer this.chunk_pipeline.deinit(this.device, alloc);

    this.total_timer = try .start();
    this.frame_count = 0;
    this.last_cursor = .{ 0, 0 };
    this.dirty_swapchain = false;
    this.camera = .{
        .pos = .{ 0, 0, 35 },
        .euler = .{ 0, 0, 0 },
        .v_fov = math.rad(90.0),
        .near = 0.1,
        .far = 10_000,
    };

    {
        const image_paths: [2][]const u8 = .{
            "res/textures/grass.png",
            "res/textures/stone.png",
        };
        const Pixel = [4]u8;

        const size: @Vector(2, u32) = .{ 8, 8 };
        const pixel_count = @reduce(.Mul, size);
        const image_stride = pixel_count * @sizeOf(Pixel);

        this.texture_image = try .init(this.device, .{
            .alloc = alloc,
            .format = .rgba8_srgb,
            .usage = .{
                .dst = true,
                .sampled = true,
            },
            .loc = .device,
            .layer_count = image_paths.len,
            .size = size,
        });
        errdefer this.texture_image.deinit(this.device, alloc);
        this.texture_sampler = try .init(this.device, .{
            .alloc = alloc,
            .min_filter = .nearest,
            .mag_filter = .nearest,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
        });
        errdefer this.texture_sampler.deinit(this.device, alloc);

        this.texture_view = try .init(this.device, .{
            .alloc = alloc,
            .image = this.texture_image,
            .aspect = .{ .color = true },
            .layer_count = image_paths.len,
        });
        errdefer this.texture_view.deinit(this.device, alloc);

        const staging = try this.device.initBuffer(.{
            .alloc = alloc,
            .loc = .host,
            .usage = .{ .src = true },
            .size = image_stride * image_paths.len,
        });
        defer staging.deinit(this.device, alloc);
        const mapped_bytes = try staging.map(this.device);

        for (image_paths, 0..) |image_path, i| {
            var read_buffer: [zigimg.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
            var image = try zigimg.Image.fromFilePath(alloc, image_path, &read_buffer);
            defer image.deinit(alloc);
            var cropped = try image.crop(alloc, .{ .width = size[0], .height = size[1] });
            try cropped.convert(alloc, .rgba32);
            defer cropped.deinit(alloc);
            @memcpy(mapped_bytes[image_stride * i .. image_stride * (i + 1)], cropped.rawBytes());
        }

        const fence = try this.device.initFence(false);
        defer fence.deinit(this.device);
        const cmd_encoder = try this.device.initCommandEncoder();
        defer cmd_encoder.deinit(this.device);

        try cmd_encoder.begin(this.device);

        try cmd_encoder.cmdMemoryBarrier(this.device, &.{
            .{
                .image = .{
                    .image = this.texture_image,
                    .aspect = .{ .color = true },
                    .old_layout = .undefined,
                    .new_layout = .transfer_dst,
                    .src_stage = .{ .pipeline_start = true },
                    .dst_stage = .{ .transfer = true },
                    .src_access = .{},
                    .dst_access = .{ .transfer_write = true },
                },
            },
        }, alloc);

        cmd_encoder.cmdCopyBufferToImage(.{
            .device = this.device,
            .src = staging.region(),
            .dst = this.texture_image,
            .layout = .transfer_dst,
            .aspect = .{ .color = true },
            .image_offset = @splat(0),
            .image_size = .{ size[0], size[1], 1 },
            .layer_count = image_paths.len,
        });

        try cmd_encoder.cmdMemoryBarrier(this.device, &.{
            .{
                .image = .{
                    .image = this.texture_image,
                    .aspect = .{ .color = true },
                    .old_layout = .transfer_dst,
                    .new_layout = .shader_read_only,
                    .src_stage = .{ .transfer = true },
                    .dst_stage = .{ .pixel_shader = true },
                    .src_access = .{ .transfer_write = true },
                    .dst_access = .{ .shader_read = true },
                },
            },
        }, alloc);

        try cmd_encoder.end(this.device);
        try this.device.submitCommands(.{
            .encoder = cmd_encoder,
            .signal_fence = fence,
            .signal_semaphores = &.{},
            .wait_dst_stages = &.{},
            .wait_semaphores = &.{},
        });
        try fence.wait(this.device, std.time.ns_per_s);
    }

    try this.chunk_resource_set.update(this.device, &.{
        .{
            .binding = 0,
            .data = .{ .image = &.{.{
                .layout = .shader_read_only,
                .view = this.texture_view,
                .sampler = this.texture_sampler,
            }} },
        },
    }, alloc);
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.device.waitUntilIdle();

    this.texture_sampler.deinit(this.device, alloc);
    this.texture_view.deinit(this.device, alloc);
    this.texture_image.deinit(this.device, alloc);
    this.chunk_mesher.deinit();
    this.chunk_mesh_alloc.deinit();
    this.world_gen.deinit();

    var chunk_iter = this.chunks.iterator();
    while (chunk_iter.next()) |kv| {
        kv.value_ptr.deinit(alloc);
    }
    this.chunks.deinit(alloc);

    this.deinitFramesInFlight(alloc);
    this.chunk_pipeline.deinit(this.device, alloc);
    gpu.AnyObject.deinitAllReversed(this.destruct_queue.items, this.device, alloc);
    this.destruct_queue.deinit(alloc);
    alloc.free(this.images_initialized);
    this.display.deinit(alloc);
    this.device.deinit(alloc);
    this.instance.deinit(alloc);
    this.window.deinit();
    this.event_queue.deinit();

    const total_time_s: f64 = @as(f64, @floatFromInt(this.total_timer.read())) / std.time.ns_per_s;
    const fps: f64 = @as(f64, @floatFromInt(this.frame_count)) / total_time_s;
    std.log.info("mean fps {}", .{fps});
}

fn loadChunks(this: *@This(), alloc: std.mem.Allocator) !void {
    const render_radius: i32 = @intCast(options.render_radius);
    const vertical_render_radius: i32 = @intCast(options.vrender_radius);
    const chunk_count = (render_radius * 2 + 1) * (render_radius * 2 + 1) * (vertical_render_radius * 2 + 1);

    try this.world_gen.queue.ensureUnusedCapacity(alloc, chunk_count);
    try this.chunk_mesher.thread_info.queue.ensureUnusedCapacity(alloc, chunk_count);
    var z: i32 = vertical_render_radius;
    while (z >= -vertical_render_radius) : (z -= 1) {
        var y: i32 = -render_radius;
        while (y <= render_radius) : (y += 1) {
            var x: i32 = -render_radius;
            while (x <= render_radius) : (x += 1) {
                const pos: Chunk.ChunkPos = .{ x, y, z };
                this.world_gen.queue.pushBackAssumeCapacity(pos);
            }
        }
    }

    const queue = this.world_gen.queue.buffer[0..this.world_gen.queue.len];
    std.mem.sort(Chunk.ChunkPos, queue, {}, struct {
        fn lessThanFn(_: void, left: Chunk.ChunkPos, right: Chunk.ChunkPos) bool {
            const f_left: math.Vec3 = @floatFromInt(left);
            const f_right: math.Vec3 = @floatFromInt(right);
            return math.lengthSqr(f_left) < math.lengthSqr(f_right);
        }
    }.lessThanFn);

    var timer: std.time.Timer = try .start();
    try this.world_gen.genMany(&this.chunks, &this.chunk_mesher, alloc);
    std.log.info("world gen time: {} ns", .{timer.read()});
    std.log.info("chunk count {}", .{chunk_count});
}

pub fn render(this: *@This(), input: Input, alloc: std.mem.Allocator) !void {
    if (this.frame_count == 0)
        this.frame_timer = try .start();

    if (this.dirty_swapchain) {
        std.log.info("rebuilding swapchain", .{});
        const fences = try alloc.alloc(gpu.Fence, this.per_frame_in_flight.len);
        defer alloc.free(fences);
        for (fences, 0..) |*fence, i| fence.* = this.per_frame_in_flight[i].presented_fence;
        try gpu.Fence.waitMany(fences, this.device, .all, std.time.ns_per_s);

        const new_viewport = this.window.getFramebufferSize();
        try this.display.rebuild(new_viewport, alloc);
        for (this.per_frame_in_flight) |*x| {
            x.deinitViewportDependants(this, alloc);
            try x.initViewportDependants(this, alloc);
        }

        @memset(this.images_initialized, false);
        this.dirty_swapchain = false;
        return;
    }

    const per_frame = &this.per_frame_in_flight[this.frame_index];
    try per_frame.presented_fence.wait(this.device, std.time.ns_per_s);
    gpu.AnyObject.deinitAllReversed(per_frame.trash.items, this.device, alloc);
    per_frame.trash.clearRetainingCapacity();

    try this.chunk_mesher.meshMany();

    if (input.wireframe) {
        try per_frame.trash.append(alloc, .{ .graphics_pipeline = this.chunk_pipeline });
        this.wireframe = !this.wireframe;
        try this.initChunkPipeline(alloc);
    }

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

        var speed: f32 = 40;
        if (this.window.isKeyDown(.left_shift))
            speed = 200;
        if (this.window.isKeyDown(.left_control))
            speed = 1000;

        if (!math.eql(move_vector, @as(math.Vec3, @splat(0)))) {
            const q = math.quatFromEuler(this.camera.euler);
            move_vector = math.quatMulVec(q, move_vector);
            move_vector = math.normalize(move_vector);
            move_vector *= @splat(dt * speed);
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

    const image_index = blk: switch (try this.display.acquireImageIndex(per_frame.image_available_semaphore, null, std.time.ns_per_s)) {
        .success => |x| x,
        .suboptimal => |x| {
            this.dirty_swapchain = true;
            break :blk x;
        },
        .out_of_date => {
            this.dirty_swapchain = true;
            return;
        },
    };

    try per_frame.cmd_encoder.begin(this.device);
    try this.chunk_mesh_alloc.upload(this.device, per_frame.cmd_encoder);

    try per_frame.cmd_encoder.cmdMemoryBarrier(this.device, &.{
        .{ .image = .{
            .image = this.display.image(image_index),
            .aspect = .{ .color = true },
            .old_layout = if (this.images_initialized[image_index]) .present_src else .undefined,
            .new_layout = .color_attachment,
            .src_stage = .{ .pipeline_start = true },
            .dst_stage = .{ .color_attachment_output = true },
            .src_access = .{},
            .dst_access = .{ .color_attachment_write = true },
        } },
    }, alloc);

    const render_pass = per_frame.cmd_encoder.cmdBeginRenderPass(.{
        .device = this.device,
        .target = .{
            .color_image_view = this.display.imageView(image_index),
            .color_clear_value = @as(math.Vec4, .{ 66.0, 130.0, 250.0, 255.0 }) / @as(math.Vec4, @splat(255.0)),
            .depth_image_view = per_frame.depth_image_view,
        },
        .image_size = this.display.imageSize(),
    });

    const push_constants: PerFramePushConstants = .{
        .vp = math.toArray(this.camera.vp(aspect_ratio)),
    };

    this.drawChunks(render_pass, push_constants, aspect_ratio);

    render_pass.cmdEnd(this.device);

    try per_frame.cmd_encoder.cmdMemoryBarrier(this.device, &.{
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
    }, alloc);

    try per_frame.cmd_encoder.end(this.device);
    try this.device.submitCommands(.{
        .encoder = per_frame.cmd_encoder,
        .wait_semaphores = &.{per_frame.image_available_semaphore},
        .wait_dst_stages = &.{.{
            .color_attachment_output = true,
            .early_depth_tests = true,
        }},
        .signal_semaphores = &.{per_frame.render_finished_semaphore},
        .signal_fence = null,
    });

    try per_frame.presented_fence.reset(this.device);
    _ = try this.display.presentImage(image_index, &.{per_frame.render_finished_semaphore}, per_frame.presented_fence);
    this.frame_count += 1;
    this.frame_index = (this.frame_index + 1) % this.per_frame_in_flight.len;
}

fn drawChunks(this: *Renderer, render_pass: gpu.RenderPassEncoder, push_constants: PerFramePushConstants, aspect_ratio: f32) void {
    render_pass.cmdBindPipeline(this.device, this.chunk_pipeline, this.display.imageSize());
    render_pass.cmdBindResourceSets(this.device, this.chunk_pipeline, &.{this.chunk_resource_set}, 0);
    render_pass.cmdBindVertexBuffer(this.device, 0, this.chunk_mesh_alloc.buffer.region());
    render_pass.cmdPushConstants(this.device, this.chunk_pipeline, .{
        .stages = .{ .vertex = true },
        .offset = 0,
        .size = @sizeOf(PerFramePushConstants),
    }, @ptrCast(&push_constants));

    // frustum culling stuff
    const frustum_planes = frustumPlanes(this.camera, aspect_ratio);
    const q = math.quatFromEuler(this.camera.euler);
    const forward = math.quatMulVec(q, math.dir_forward);
    const right = math.quatMulVec(q, math.dir_right);
    const up = math.quatMulVec(q, math.dir_up);
    const af = @abs(forward);
    const ar = @abs(right);
    const au = @abs(up);

    var chunk_mesh_iter = this.chunk_mesh_alloc.loaded_meshes.iterator();
    while (chunk_mesh_iter.next()) |kv| {
        const pos: math.Vec3 = @floatFromInt(kv.key_ptr.* * Chunk.size);
        const chunk_size_f = math.i2f(math.Vec3, Chunk.size);
        // extent
        const e_ws = chunk_size_f / math.splat3(f32, 2);
        // center
        const c_ws = pos + e_ws;
        const c_vs = pointToViewSpace(c_ws, this.camera.pos, forward, right, up);
        const e_vs = pointToViewSpace(e_ws, @splat(0), af, ar, au);
        const min = c_vs - e_vs;
        const max = c_vs + e_vs;

        const culled = blk: for (frustum_planes) |p| {
            if (!aabbInPlane(min, max, p)) {
                break :blk true;
            }
        } else false;

        if (!culled)
            this.drawChunk(render_pass, kv.key_ptr.*, kv.value_ptr.*);
    }
}

fn frustumPlanes(cam: Camera, aspect_ratio: f32) [6]Plane {
    const hy = cam.v_fov / 2;
    const ty = @tan(hy);
    const tx = ty * aspect_ratio;

    return .{
        .{ .n = .{ 1, 0, 0 }, .d = -cam.near }, // near
        .{ .n = .{ -1, 0, 0 }, .d = cam.far }, // far
        .{ .n = .{ tx, 1, 0 }, .d = 0 }, // left
        .{ .n = .{ tx, -1, 0 }, .d = 0 }, // right
        .{ .n = .{ ty, 0, 1 }, .d = 0 }, // bottom
        .{ .n = .{ ty, 0, -1 }, .d = 0 }, // top
    };
}

fn aabbInPlane(min: math.Vec3, max: math.Vec3, p: Plane) bool {
    const c = (min + max) / math.splat3(f32, 2);
    const e = (max - min) / math.splat3(f32, 2);

    const n = p.n;
    const s = math.dot(n, c) + p.d;
    const r = math.dot(@abs(n), e);

    if (s + r < 0) return false;
    return true;
}

fn pointToViewSpace(
    point: math.Vec3,
    cam_pos: math.Vec3,
    forward: math.Vec3,
    right: math.Vec3,
    up: math.Vec3,
) math.Vec3 {
    return .{
        math.dot(point - cam_pos, forward),
        math.dot(point - cam_pos, right),
        math.dot(point - cam_pos, up),
    };
}

const Plane = struct {
    n: math.Vec3,
    d: f32,
};

fn drawChunk(this: *Renderer, render_pass: gpu.RenderPassEncoder, pos: Chunk.ChunkPos, loaded_mesh: ChunkMesher.GpuLoaded) void {
    const push_constants: ChunkPushConstants = .{
        .pos = math.toArray(pos),
    };

    render_pass.cmdPushConstants(
        this.device,
        this.chunk_pipeline,
        .{
            .stages = .{ .vertex = true },
            .offset = @sizeOf(PerFramePushConstants),
            .size = @sizeOf(ChunkPushConstants),
        },
        @ptrCast(std.mem.asBytes(&push_constants)),
    );

    render_pass.cmdDraw(.{
        .device = this.device,
        .vertex_count = 6,
        .instance_count = @intCast(loaded_mesh.face_count),
        .first_instance = @intCast(loaded_mesh.face_offset),
        .indexed = false,
    });
}

fn initFramesInFlight(this: *Renderer, alloc: std.mem.Allocator) !void {
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

fn initChunkPipeline(this: *Renderer, alloc: std.mem.Allocator) !void {
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
                .size = @sizeOf(PerFramePushConstants) + @sizeOf(ChunkPushConstants),
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
                .{ .type = .uint32x2 },
            },
        }},
        .polygon_mode = if (this.wireframe) .line else .fill,
        .cull_mode = .back,
        .depth_mode = .{
            .testing = true,
            .writing = true,
            .compare_op = .less,
        },
    });
}

const PerFrameInFlight = struct {
    presented_fence: gpu.Fence,
    render_finished_semaphore: gpu.Semaphore,
    image_available_semaphore: gpu.Semaphore,
    cmd_encoder: gpu.CommandEncoder,
    depth_image: gpu.Image,
    depth_image_view: gpu.Image.View,
    trash: std.ArrayList(gpu.AnyObject),

    pub fn init(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) !void {
        this.presented_fence = try .init(renderer.device, true);
        errdefer this.presented_fence.deinit(renderer.device);

        this.render_finished_semaphore = try .init(renderer.device);
        errdefer this.render_finished_semaphore.deinit(renderer.device);

        this.image_available_semaphore = try .init(renderer.device);
        errdefer this.image_available_semaphore.deinit(renderer.device);

        this.cmd_encoder = try .init(renderer.device);
        errdefer this.cmd_encoder.deinit(renderer.device);

        try this.initViewportDependants(renderer, alloc);
        errdefer this.deinitViewportDependants(renderer, alloc);

        this.trash = .empty;
    }

    pub fn deinit(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) void {
        gpu.AnyObject.deinitAllReversed(this.trash.items, renderer.device, alloc);
        this.trash.deinit(alloc);
        this.deinitViewportDependants(renderer, alloc);
        this.cmd_encoder.deinit(renderer.device);
        this.image_available_semaphore.deinit(renderer.device);
        this.render_finished_semaphore.deinit(renderer.device);
        this.presented_fence.deinit(renderer.device);
    }

    pub fn initViewportDependants(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) !void {
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

        this.depth_image_view = try .init(renderer.device, .{
            .alloc = alloc,
            .image = this.depth_image,
            .aspect = .{ .depth = true },
        });
        errdefer this.depth_image_view.deinit(renderer.device, alloc);
    }

    pub fn deinitViewportDependants(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) void {
        this.depth_image_view.deinit(renderer.device, alloc);
        this.depth_image.deinit(renderer.device, alloc);
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

const ChunkPushConstants = struct {
    pos: [3]i32,
};

const PerFramePushConstants = struct {
    vp: [16]f32,
};

const Camera = struct {
    pos: math.Vec3,
    euler: math.Vec3,
    v_fov: f32,
    near: f32,
    far: f32,

    fn view(this: Camera) math.Mat4 {
        return math.matMulMany(.{
            math.rotateEuler(this.euler),
            math.translate(-this.pos),
        });
    }

    fn proj(this: Camera, aspect_ratio: f32) math.Mat4 {
        return math.perspective(aspect_ratio, this.v_fov, this.near, this.far);
    }

    fn vp(this: Camera, aspect_ratio: f32) math.Mat4 {
        return math.matMul(this.proj(aspect_ratio), this.view());
    }
};
