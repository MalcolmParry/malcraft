const std = @import("std");
const options = @import("options");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const Camera = @import("Camera.zig");
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const ChunkMeshAllocator = @import("ChunkMeshAllocator.zig");
const ShaderManager = @import("ShaderManager.zig");
const TextureManager = @import("TextureManager.zig");
const UIRenderer = @import("UIRenderer.zig");
const Renderer = @This();

info: Info,
stage_man: gpu.StagingManager,
upload_man: gpu.UploadManager,
shader_man: ShaderManager,
destruct_queue: std.ArrayList(gpu.AnyObject),
instance: gpu.Instance,
device: gpu.Device,
display: gpu.Display,
timeline: gpu.Timeline,
timeline_value: gpu.Timeline.Value,
images_initialized: []bool,
per_frame_in_flight: []PerFrameInFlight,
dirty_swapchain: bool,
wireframe: bool,

ui: UIRenderer,
texture_man: TextureManager,
chunk_mesh_alloc: ChunkMeshAllocator,
chunk_resource_layout: gpu.ResourceSet.Layout,
chunk_resource_set: gpu.ResourceSet,
chunk_pipeline: gpu.GraphicsPipeline,

pub const FrameData = struct {
    pub const Input = struct {
        wireframe: bool = false,
    };

    dt_ns: u64,
    camera: Camera,
    viewport: mw.gpu.Image.Size2D,
    input: Input = .{},

    show_crosshair: bool,
    generating_chunks: bool,
};

pub const InitInfo = struct {
    alloc: std.mem.Allocator,
    window: *mw.Window,
};

pub fn init(this: *@This(), info: InitInfo) !void {
    const alloc = info.alloc;

    this.instance = try .init(options.gpu_validation, alloc);
    errdefer this.instance.deinit(alloc);

    const phys_device = try this.instance.bestPhysicalDevice();
    this.device = try .init(this.instance, phys_device, alloc);
    errdefer this.device.deinit(alloc);

    this.display = try .init(this.device, info.window, alloc);
    errdefer this.display.deinit(alloc);
    std.log.info("display size: {}", .{this.display.imageSize()});

    this.timeline_value = 0;
    this.timeline = try this.device.initTimeline(0);
    errdefer this.timeline.deinit(this.device);

    this.info = .{
        .frames_in_flight = @min(this.display.imageCount(), 3),
        .frame_count = 0,
    };

    this.stage_man = try .init(.{
        .alloc = alloc,
        .device = this.device,
        .buffer_size = 1024 * 1024 * 2,
        .frames_in_flight = this.info.frames_in_flight,
    });
    errdefer this.stage_man.deinit(this.device, alloc);

    this.upload_man = .{
        .alloc = alloc,
        .stage_man = &this.stage_man,
    };
    errdefer this.upload_man.deinit();

    this.shader_man = try .init(this.device, alloc);
    errdefer this.shader_man.deinit(this.device, alloc);

    this.images_initialized = try alloc.alloc(bool, this.info.frames_in_flight);
    errdefer alloc.free(this.images_initialized);
    @memset(this.images_initialized, false);

    this.destruct_queue = try .initCapacity(alloc, 32);
    errdefer this.destruct_queue.deinit(alloc);
    errdefer gpu.AnyObject.deinitAllReversed(this.destruct_queue.items, this.device, alloc);

    try this.initFramesInFlight(alloc);
    errdefer this.deinitFramesInFlight(alloc);

    try this.chunk_mesh_alloc.init(.{
        .device = this.device,
        .alloc = alloc,
        .upload_man = &this.upload_man,
        .renderer_info = &this.info,
    });
    errdefer this.chunk_mesh_alloc.deinit();

    this.chunk_resource_layout = try .init(this.device, .{
        .alloc = alloc,
        .descriptors = &.{.{
            .t = .image,
            .stages = .{ .pixel = true },
            .flags = .{},
            .binding = 0,
            .count = 1,
        }},
    });
    try this.destruct_queue.append(alloc, .{ .resource_layout = this.chunk_resource_layout });

    this.chunk_resource_set = try .init(this.device, this.chunk_resource_layout, alloc);
    try this.destruct_queue.append(alloc, .{ .resource_set = this.chunk_resource_set });

    this.wireframe = false;
    try this.initChunkPipeline(alloc);
    errdefer this.chunk_pipeline.deinit(this.device, alloc);

    this.dirty_swapchain = false;

    try this.ui.init(.{
        .alloc = alloc,
        .device = this.device,
        .stage_man = &this.stage_man,
        .shader_man = &this.shader_man,
        .color_format = this.display.imageFormat(),
        .frames_in_flight = this.info.frames_in_flight,
    });
    errdefer this.ui.deinit(this.device);

    this.texture_man = try .init(alloc, this.device, &this.stage_man);
    errdefer this.texture_man.deinit(alloc, this.device);

    try this.chunk_resource_set.update(this.device, &.{
        .{
            .binding = 0,
            .data = .{ .image = &.{.{
                .layout = .shader_read_only,
                .view = this.texture_man.view,
                .sampler = this.texture_man.sampler,
            }} },
        },
    }, alloc);
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.device.waitUntilIdle() catch @panic("failed to wait for device in deinit");

    this.chunk_mesh_alloc.deinit();
    this.texture_man.deinit(alloc, this.device);

    this.ui.deinit(this.device);
    this.deinitFramesInFlight(alloc);
    this.chunk_pipeline.deinit(this.device, alloc);
    gpu.AnyObject.deinitAllReversed(this.destruct_queue.items, this.device, alloc);
    this.destruct_queue.deinit(alloc);
    alloc.free(this.images_initialized);
    this.shader_man.deinit(this.device, alloc);
    this.upload_man.deinit();
    this.stage_man.deinit(this.device, alloc);
    this.timeline.deinit(this.device);
    this.display.deinit(alloc);
    this.device.deinit(alloc);
    this.instance.deinit(alloc);
}

pub fn render(this: *@This(), data: FrameData, alloc: std.mem.Allocator) !void {
    const input = data.input;

    if (this.timeline_value >= this.per_frame_in_flight.len)
        try this.timeline.wait(this.device, this.timeline_value - this.per_frame_in_flight.len + 1, std.time.ns_per_s);

    const frame_slot = (this.info.frame_count + 1) % this.info.frames_in_flight;
    const per_frame = &this.per_frame_in_flight[frame_slot];
    this.info.frame_count += 1;

    gpu.AnyObject.deinitAllReversed(per_frame.trash.items, this.device, alloc);
    per_frame.trash.clearRetainingCapacity();
    try this.chunk_mesh_alloc.freeQueued();
    this.stage_man.nextFrame();

    if (this.dirty_swapchain) {
        try this.device.waitUntilIdle();

        std.log.info("rebuilding swapchain {}", .{data.viewport});
        try this.display.rebuild(data.viewport, alloc);
        for (this.per_frame_in_flight) |*x| {
            x.deinitViewportDependants(this, alloc);
            try x.initViewportDependants(this, alloc);
        }

        @memset(this.images_initialized, false);
        this.dirty_swapchain = false;
        return;
    }

    if (input.wireframe) {
        try per_frame.trash.append(alloc, .{ .graphics_pipeline = this.chunk_pipeline });
        this.wireframe = !this.wireframe;
        try this.initChunkPipeline(alloc);
    }

    const viewport_f: math.Vec2 = @floatFromInt(data.viewport);
    const aspect_ratio = viewport_f[0] / viewport_f[1];

    const acquired_image = blk: {
        const result = this.display.acquireImage(std.time.ns_per_s) catch |err| switch (err) {
            error.OutOfDate => {
                this.dirty_swapchain = true;
                return;
            },
            else => return err,
        };

        if (!result.optimal) this.dirty_swapchain = true;
        break :blk result.image;
    };

    try per_frame.cmd_encoder.begin();
    try this.upload_man.upload(per_frame.cmd_encoder);

    per_frame.cmd_encoder.cmdMemoryBarrier(.{
        .image_barriers = &.{.{
            .image = acquired_image.image(this.display),
            .subresource_range = .{
                .aspect = .{ .color = true },
            },
            .old_layout = acquired_image.initialLayout(this.display),
            .new_layout = .color_attachment,
            .src_stage = .{ .pipeline_start = true },
            .dst_stage = .{ .color_attachment_output = true },
            .src_access = .{},
            .dst_access = .{ .color_attachment_write = true },
        }},
    });

    const render_pass = per_frame.cmd_encoder.cmdBeginRenderPass(.{
        .target = .{
            .color_attachment = .{
                .image_view = acquired_image.imageView(this.display),
                .load = .{
                    .clear = .{
                        .color = @as(math.Vec4, .{ 66.0, 130.0, 250.0, 255.0 }) / @as(math.Vec4, @splat(255.0)),
                        // .color = .{ 0, 0, 0, 1 },
                    },
                },
                .store = .store,
            },
            .depth_attachment = .{
                .image_view = per_frame.depth_image_view,
                .load = .{ .clear = .{ .depth = 1 } },
                .store = .store,
            },
        },
        .image_size = this.display.imageSize(),
    });

    const push_constants: PerFramePushConstants = .{
        .vp = math.matrixToArray(data.camera.vp(aspect_ratio), .row_major),
    };

    this.drawChunks(render_pass, push_constants, aspect_ratio, data.camera);

    render_pass.cmdEnd();

    try this.ui.render(.{
        .alloc = alloc,
        .device = this.device,
        .image_view = acquired_image.imageView(this.display),
        .cmd_encoder = per_frame.cmd_encoder,
        .viewport = data.viewport,
        .frame_count = this.info.frame_count,
        .dt_ns = data.dt_ns,
        .show_crosshair = data.show_crosshair,
        .camera = data.camera,
        .chunk_mesh_buffer_bytes_used = ChunkMeshAllocator.buffer_size - this.chunk_mesh_alloc.queryBytesFree(),
        .chunk_mesh_buffer_bytes_total = ChunkMeshAllocator.buffer_size,
        .chunk_mesh_buffer_largest_free_block = this.chunk_mesh_alloc.queryLargestFreeBlock(),
        .loaded_mesh_count = this.chunk_mesh_alloc.loaded_meshes.count(),
        .overwritten_meshes = this.chunk_mesh_alloc.overwritten_meshes,
        .generating_chunks = data.generating_chunks,
    });

    per_frame.cmd_encoder.cmdMemoryBarrier(.{
        .image_barriers = &.{.{
            .image = acquired_image.image(this.display),
            .subresource_range = .{
                .aspect = .{ .color = true },
            },
            .old_layout = .color_attachment,
            .new_layout = .present_src,
            .src_stage = .{ .color_attachment_output = true },
            .dst_stage = .{ .pipeline_end = true },
            .src_access = .{ .color_attachment_write = true },
            .dst_access = .{},
        }},
    });

    try per_frame.cmd_encoder.end();

    this.timeline_value += 1;
    try this.device.submitCommands(.{
        .encoder = per_frame.cmd_encoder,
        .signals = &.{.{
            .timeline = this.timeline,
            .value = this.timeline_value,
            .stages = .{ .all_commands = true },
        }},
        .display_acquire_waits = &.{.{
            .display = this.display,
            .image = acquired_image,
            .stages = .{
                .color_attachment_output = true,
            },
        }},
        .display_present_signals = &.{.{
            .display = this.display,
            .image = acquired_image,
            .stages = .{
                .color_attachment_output = true,
            },
        }},
    });

    const present_result = acquired_image.present(this.display) catch |err| switch (err) {
        error.OutOfDate => {
            this.dirty_swapchain = true;
            return;
        },
        else => return err,
    };

    switch (present_result) {
        .success => {},
        .suboptimal => this.dirty_swapchain = true,
    }
}

fn drawChunks(this: *Renderer, render_pass: gpu.RenderPassEncoder, push_constants: PerFramePushConstants, aspect_ratio: f32, camera: Camera) void {
    render_pass.cmdBindPipeline(this.chunk_pipeline);
    render_pass.cmdBindResourceSets(this.chunk_pipeline, &.{this.chunk_resource_set}, 0);
    render_pass.cmdBindVertexBuffer(0, this.chunk_mesh_alloc.buffer.region());
    render_pass.cmdPushConstants(this.chunk_pipeline, .{
        .stages = .{ .vertex = true },
        .offset = 0,
        .size = @sizeOf(PerFramePushConstants),
    }, @ptrCast(&push_constants));

    // frustum culling stuff
    const frustum_planes = frustumPlanes(camera, aspect_ratio);
    const q = math.quatFromEuler(camera.euler);
    const forward = math.quatMulVec(q, math.dir_forward);
    const right = math.quatMulVec(q, math.dir_right);
    const up = math.quatMulVec(q, math.dir_up);
    const af = @abs(forward);
    const ar = @abs(right);
    const au = @abs(up);

    var chunk_mesh_iter = this.chunk_mesh_alloc.loaded_meshes.iterator();
    while (chunk_mesh_iter.next()) |kv| {
        const pos: math.Vec3 = @floatFromInt(kv.key_ptr.*.vec() * Chunk.size);
        const chunk_size_f = math.i2f(math.Vec3, Chunk.size);
        // extent
        const e_ws = chunk_size_f / math.splat3(f32, 2);
        // center
        const c_ws = pos + e_ws;
        const c_vs = pointToViewSpace(c_ws, camera.pos, forward, right, up);
        const e_vs = pointToViewSpace(e_ws, @splat(0), af, ar, au);
        const min = c_vs - e_vs;
        const max = c_vs + e_vs;

        const culled = blk: for (frustum_planes) |p| {
            if (!aabbInPlane(min, max, p)) {
                break :blk true;
            }
        } else false;

        if (!culled)
            this.drawChunk(render_pass, kv.key_ptr.*.vec(), kv.value_ptr.*);
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
    const s = math.dot(math.Vec3, n, c) + p.d;
    const r = math.dot(math.Vec3, @abs(n), e);

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
        math.dot(math.Vec3, point - cam_pos, forward),
        math.dot(math.Vec3, point - cam_pos, right),
        math.dot(math.Vec3, point - cam_pos, up),
    };
}

const Plane = struct {
    n: math.Vec3,
    d: f32,
};

fn drawChunk(this: *Renderer, render_pass: gpu.RenderPassEncoder, pos: Chunk.Pos, loaded_mesh: ChunkMesher.GpuLoaded) void {
    const push_constants: ChunkPushConstants = .{
        .pos = pos,
    };

    render_pass.cmdPushConstants(
        this.chunk_pipeline,
        .{
            .stages = .{ .vertex = true },
            .offset = @sizeOf(PerFramePushConstants),
            .size = @sizeOf(ChunkPushConstants),
        },
        @ptrCast(std.mem.asBytes(&push_constants)),
    );

    render_pass.cmdDraw(.{
        .vertex_count = 6,
        .instance_count = @intCast(loaded_mesh.face_count),
        .first_instance = @intCast(loaded_mesh.face_offset),
        .indexed = false,
    });
}

fn initFramesInFlight(this: *Renderer, alloc: std.mem.Allocator) !void {
    this.per_frame_in_flight = try alloc.alloc(PerFrameInFlight, this.info.frames_in_flight);
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
            this.shader_man.getShader(.chunk_opaque_vertex),
            this.shader_man.getShader(.chunk_opaque_pixel),
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
    cmd_encoder: gpu.CommandEncoder,
    depth_image: gpu.Image,
    depth_image_view: gpu.Image.View,
    trash: std.ArrayList(gpu.AnyObject),

    pub fn init(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) !void {
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
            .kind = .@"2d",
            .image = this.depth_image,
            .subresource_range = .{
                .aspect = .{ .depth = true },
            },
        });
        errdefer this.depth_image_view.deinit(renderer.device, alloc);

        try renderer.upload_man.post_copy_image_barriers.append(alloc, .{
            .image = this.depth_image,
            .subresource_range = .{
                .aspect = .{ .depth = true },
            },
            .old_layout = .undefined,
            .new_layout = .depth_stencil,
            .src_stage = .{ .pipeline_start = true },
            .dst_stage = .{ .early_depth_tests = true },
            .src_access = .{},
            .dst_access = .{
                .depth_stencil_read = true,
                .depth_stencil_write = true,
            },
        });
    }

    pub fn deinitViewportDependants(this: *PerFrameInFlight, renderer: *Renderer, alloc: std.mem.Allocator) void {
        this.depth_image_view.deinit(renderer.device, alloc);
        this.depth_image.deinit(renderer.device, alloc);
    }
};

const ChunkPushConstants = struct {
    pos: [3]i32,
};

const PerFramePushConstants = struct {
    vp: [16]f32,
};

pub const Info = struct {
    frame_count: u64,
    frames_in_flight: u32,

    pub inline fn frame_slot(info: *const Info) u32 {
        return @intCast(info.frame_count % info.frames_in_flight);
    }
};
