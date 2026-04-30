const std = @import("std");
const options = @import("options");
const zigimg = @import("zigimg");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const block = @import("block.zig");
const Chunk = @import("Chunk.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const ChunkMeshAllocator = @import("ChunkMeshAllocator.zig");
const WorldGenerator = @import("WorldGenerator.zig");
const World = @import("World.zig");
const ShaderManager = @import("ShaderManager.zig");

const Renderer = @This();
const dt_hist_size = 256;
const cpu_time_hist_size = 256;

info: Info,
stage_man: gpu.StagingManager,
upload_man: gpu.UploadManager,
shader_man: ShaderManager,
immediate: mw.ImmediateRenderer,
destruct_queue: std.ArrayList(gpu.AnyObject),
event_queue: mw.EventQueue,
window: mw.Window,
instance: gpu.Instance,
device: gpu.Device,
display: gpu.Display,
last_image_index: ?u32,
timeline: gpu.Timeline,
timeline_value: gpu.Timeline.Value,
images_initialized: []bool,
per_frame_in_flight: []PerFrameInFlight,
frame_timer: std.time.Timer,
last_cursor: @Vector(2, f32),
dirty_swapchain: bool,
wireframe: bool,
mouse_lock: bool,

world: World,
world_gen: WorldGenerator,
chunk_mesh_alloc: ChunkMeshAllocator,
chunk_mesher: ChunkMesher,
chunk_resource_layout: gpu.ResourceSet.Layout,
chunk_resource_set: gpu.ResourceSet,
chunk_pipeline: gpu.GraphicsPipeline,
camera: Camera,

texture_image: gpu.Image,
texture_view: gpu.Image.View,
texture_sampler: gpu.Sampler,

font_face: mw.text.Face,
glyph_cache: mw.text.GlyphCache,

dt_hist: [dt_hist_size]u64,
cpu_time_hist: [cpu_time_hist_size]u64,

pub const Input = struct {
    wireframe: bool = false,
    break_block: bool = false,
    place_block: bool = false,
    cam_reset: bool = false,
    mouse_lock_toggle: bool = false,
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

    this.last_image_index = null;
    this.display = try .init(this.device, &this.window, alloc);
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
        .buffer_size = 1024 * 1024 * 8,
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

    this.world = .{};
    errdefer this.world.deinit(alloc);

    try this.world_gen.init(alloc);
    errdefer this.world_gen.deinit();

    try this.chunk_mesh_alloc.init(.{
        .device = this.device,
        .alloc = alloc,
        .upload_man = &this.upload_man,
        .renderer_info = &this.info,
    });
    errdefer this.chunk_mesh_alloc.deinit();

    try this.chunk_mesher.init(.{
        .mesh_alloc = &this.chunk_mesh_alloc,
        .alloc = alloc,
        .world = &this.world,
    });
    errdefer this.chunk_mesher.deinit();

    try this.loadChunks(alloc);

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

    this.last_cursor = .{ 0, 0 };
    this.dirty_swapchain = false;
    this.mouse_lock = true;
    this.camera = .{};
    this.cpu_time_hist[0] = 0;

    this.immediate = try .init(.{
        .alloc = alloc,
        .device = this.device,
        .frames_in_flight = this.info.frames_in_flight,
        .streaming_buffer_size_pf = 128 * 1024,
        .color_format = this.display.imageFormat(),
        .box_info = .{
            .shaders = &.{
                this.shader_man.getShader(.immediate_box_vertex),
                this.shader_man.getShader(.immediate_box_pixel),
            },
        },
        .image_info = .{
            .shaders = &.{
                this.shader_man.getShader(.immediate_image_vertex),
                this.shader_man.getShader(.immediate_image_pixel),
            },
        },
        .text_info = .{
            .glyph_cache = &this.glyph_cache,
            .shaders = &.{
                this.shader_man.getShader(.immediate_text_vertex),
                this.shader_man.getShader(.immediate_text_pixel),
            },
        },
    });
    errdefer this.immediate.deinit(this.device);

    {
        // const file = "res/fonts/Roboto_Condensed-Regular.ttf";
        // const file = "res/fonts/NFPixels-Regular.ttf";
        const file = "res/fonts/press-start-2p/PressStart2P-vaV7.ttf";
        var face: mw.text.Face = try .init(file);
        errdefer face.deinit();

        var cache: mw.text.GlyphCache = .{
            .alloc = alloc,
            .stage_man = &this.stage_man,
            .atlas_size = @splat(512),
        };
        errdefer cache.deinit(this.device);

        this.font_face = face;
        this.glyph_cache = cache;
    }

    {
        const image_paths: [2][]const u8 = .{
            "res/textures/grass.png",
            "res/textures/stone.png",
        };
        const Pixel = [4]u8;

        const mip_levels = 5;
        const size: gpu.Image.Size2D = .{ 16, 16 };
        const pixel_count = @reduce(.Mul, size);
        const layer_size = pixel_count * @sizeOf(Pixel);

        this.texture_image = try .init(this.device, .{
            .alloc = alloc,
            .format = .rgba8_srgb,
            .usage = .{
                .src = true,
                .dst = true,
                .sampled = true,
            },
            .loc = .device,
            .layer_count = image_paths.len,
            .mip_count = mip_levels,
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
            .kind = .array_2d,
            .image = this.texture_image,
            .subresource_range = .{
                .aspect = .{ .color = true },
            },
        });
        errdefer this.texture_view.deinit(this.device, alloc);

        const staging = try this.stage_man.allocateBytesAligned(layer_size * image_paths.len, .@"4");
        defer this.stage_man.reset();

        for (image_paths, 0..) |image_path, i| {
            var read_buffer: [zigimg.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
            var image = try zigimg.Image.fromFilePath(alloc, image_path, &read_buffer);
            defer image.deinit(alloc);
            var cropped = try image.crop(alloc, .{ .width = size[0], .height = size[1] });
            try cropped.convert(alloc, .rgba32);
            defer cropped.deinit(alloc);
            @memcpy(staging.slice[layer_size * i .. layer_size * (i + 1)], cropped.rawBytes());
        }

        const cmd_encoder = try this.device.initCommandEncoder();
        defer cmd_encoder.deinit(this.device);

        try cmd_encoder.begin();

        cmd_encoder.cmdMemoryBarrier(.{
            .image_barriers = &.{.{
                .image = this.texture_image,
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
            .dst = this.texture_image,
            .layout = .transfer_dst,
            .subresource = .{
                .aspect = .{ .color = true },
                .layer_count = image_paths.len,
            },
        });

        var mip_size: gpu.Image.Size2D = size;
        for (1..mip_levels) |level| {
            const old_mip_size = mip_size;
            if (mip_size[0] > 1) mip_size[0] /= 2;
            if (mip_size[1] > 1) mip_size[1] /= 2;

            cmd_encoder.cmdMemoryBarrier(.{
                .image_barriers = &.{.{
                    .image = this.texture_image,
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
                .src = this.texture_image,
                .src_layout = .transfer_src,
                .src_subresource = .{
                    .aspect = .{ .color = true },
                    .mip_level = @intCast(level - 1),
                    .layer_count = image_paths.len,
                },
                .src_rect = .{ .size = old_mip_size },
                .dst = this.texture_image,
                .dst_layout = .transfer_dst,
                .dst_subresource = .{
                    .aspect = .{ .color = true },
                    .mip_level = @intCast(level),
                    .layer_count = image_paths.len,
                },
                .dst_rect = .{ .size = mip_size },
            });

            cmd_encoder.cmdMemoryBarrier(.{
                .image_barriers = &.{.{
                    .image = this.texture_image,
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
                .image = this.texture_image,
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
        this.timeline_value += 1;
        try this.device.submitCommands(.{
            .encoder = cmd_encoder,
            .signals = &.{.{
                .timeline = this.timeline,
                .value = this.timeline_value,
                .stages = .{ .all_commands = true },
            }},
        });

        try this.timeline.wait(this.device, this.timeline_value, std.time.ns_per_s);
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
    this.device.waitUntilIdle() catch @panic("failed to wait for device in deinit");

    this.glyph_cache.deinit(this.device);
    this.font_face.deinit();

    this.texture_sampler.deinit(this.device, alloc);
    this.texture_view.deinit(this.device, alloc);
    this.texture_image.deinit(this.device, alloc);
    this.chunk_mesher.deinit();
    this.chunk_mesh_alloc.deinit();
    this.world_gen.deinit();
    this.world.deinit(alloc);

    this.deinitFramesInFlight(alloc);
    this.chunk_pipeline.deinit(this.device, alloc);
    gpu.AnyObject.deinitAllReversed(this.destruct_queue.items, this.device, alloc);
    this.destruct_queue.deinit(alloc);
    alloc.free(this.images_initialized);
    this.immediate.deinit(this.device);
    this.shader_man.deinit(this.device, alloc);
    this.upload_man.deinit();
    this.stage_man.deinit(this.device, alloc);
    this.timeline.deinit(this.device);
    this.display.deinit(alloc);
    this.device.deinit(alloc);
    this.instance.deinit(alloc);
    this.window.deinit();
    this.event_queue.deinit();
}

fn loadChunks(this: *@This(), alloc: std.mem.Allocator) !void {
    const render_radius: i32 = @intCast(options.render_radius);
    const vertical_render_radius: i32 = @intCast(options.vrender_radius);
    const chunk_count = (render_radius * 2 + 1) * (render_radius * 2 + 1) * (vertical_render_radius * 2 + 1);

    try this.world_gen.queue.ensureUnusedCapacity(alloc, chunk_count);

    var z: i32 = 0;
    while (z <= vertical_render_radius) : (z += 1) {
        var y: i32 = -render_radius;
        while (y <= render_radius) : (y += 1) {
            var x: i32 = -render_radius;
            while (x <= render_radius) : (x += 1) {
                const pos: Chunk.Pos = .{ x, y, z };
                this.world_gen.queue.pushBackAssumeCapacity(pos);
            }
        }
    }

    const queue = this.world_gen.queue.buffer[0..this.world_gen.queue.len];
    std.mem.sort(Chunk.Pos, queue, {}, struct {
        fn lessThanFn(_: void, left: Chunk.Pos, right: Chunk.Pos) bool {
            const f_left: math.Vec3 = @floatFromInt(left);
            const f_right: math.Vec3 = @floatFromInt(right);
            return math.lengthSqr(f_left) < math.lengthSqr(f_right);
        }
    }.lessThanFn);
}

pub fn render(this: *@This(), input: Input, alloc: std.mem.Allocator) !void {
    if (this.info.frame_count == 0)
        this.frame_timer = try .start();

    if (this.timeline_value >= this.per_frame_in_flight.len)
        try this.timeline.wait(this.device, this.timeline_value - this.per_frame_in_flight.len + 1, std.time.ns_per_s);

    const frame_slot = (this.info.frame_count + 1) % this.info.frames_in_flight;
    const per_frame = &this.per_frame_in_flight[frame_slot];
    this.info.frame_count += 1;

    var cpu_work_timer: std.time.Timer = try .start();
    defer {
        const ns = cpu_work_timer.read();
        const slot = this.info.frame_count % cpu_time_hist_size;
        this.cpu_time_hist[slot] = ns;
    }

    gpu.AnyObject.deinitAllReversed(per_frame.trash.items, this.device, alloc);
    per_frame.trash.clearRetainingCapacity();
    try this.chunk_mesh_alloc.freeQueued();
    this.stage_man.nextFrame();

    if (this.dirty_swapchain) {
        try this.device.waitUntilIdle();

        const new_viewport = this.window.getFramebufferSize();
        std.log.info("rebuilding swapchain {}", .{new_viewport});
        try this.display.rebuild(new_viewport, alloc);
        for (this.per_frame_in_flight) |*x| {
            x.deinitViewportDependants(this, alloc);
            try x.initViewportDependants(this, alloc);
        }

        @memset(this.images_initialized, false);
        this.dirty_swapchain = false;
        return;
    }

    if (input.break_block or this.window.isMouseDown(.five)) blk: {
        const origin = this.camera.pos;
        const q = math.quatFromEuler(this.camera.euler);
        const dir = math.normalize(math.quatMulVec(q, math.dir_forward));

        const ray_cast = this.world.rayCast(origin, dir);
        const pos: block.Pos = switch (ray_cast) {
            .no_hit => break :blk,
            .inside => @intFromFloat(@floor(origin)),
            .hit => |x| x.pos,
        };

        try this.world.setBlock(alloc, pos, .air);
        try this.chunk_mesher.addRequestWithCollateral(pos);
    }

    if (input.place_block) blk: {
        const origin = this.camera.pos;
        const q = math.quatFromEuler(this.camera.euler);
        const dir = math.normalize(math.quatMulVec(q, math.dir_forward));

        const ray_cast = this.world.rayCast(origin, dir);
        const pos: block.Pos = switch (ray_cast) {
            .no_hit => break :blk,
            .inside => @intFromFloat(@floor(origin)),
            .hit => |x| x.pos + x.face.dir(),
        };

        this.world.setBlock(alloc, pos, .stone) catch |err| switch (err) {
            error.ChunkNotPresent => break :blk,
            else => return err,
        };
        try this.chunk_mesher.addRequestWithCollateral(pos);
    }

    if (input.mouse_lock_toggle) {
        this.mouse_lock = !this.mouse_lock;

        try this.window.setCursorMode(if (this.mouse_lock) .disabled else .normal);
    }

    if (input.cam_reset) this.camera = .{};

    try this.world_gen.genMany(&this.world, &this.chunk_mesher, alloc);
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

        if (!math.eql(move_vector, math.splat3(f32, 0))) {
            const q = math.quatFromEuler(this.camera.euler);
            move_vector = math.quatMulVec(q, move_vector);
            move_vector = math.normalize(move_vector);
            move_vector *= @splat(dt * speed);
            this.camera.pos += move_vector;
        }
    }

    if (this.mouse_lock) {
        const cursor = this.window.getCursorPos();
        var moved = cursor - this.last_cursor;
        this.last_cursor = cursor;

        if (!math.eql(moved, math.splat2(f32, 0))) {
            moved *= @splat(math.rad(0.18));
            this.camera.euler += math.shuffle(moved, &.{ .zero, .y, .x });
        }
    }

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
        .vp = math.matrixToArray(this.camera.vp(aspect_ratio), .row_major),
    };

    this.drawChunks(render_pass, push_constants, aspect_ratio);

    render_pass.cmdEnd();

    try this.immediate.begin(@as(@Vector(2, u16), @intCast(viewport)));

    // crosshair
    if (this.mouse_lock) {
        const outline_times_2 = 2;
        const length: i16 = @intFromFloat(viewport_f[1] * 0.025);
        const width = @divTrunc(length, 10);
        const outline_color: [4]u8 = .{ 128, 128, 128, 255 };
        const color: [4]u8 = .{ 0, 0, 0, 255 };

        try this.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ length, width },
                },
                .pivot = @splat(0.5),
            },
            .color = outline_color,
        });
        try this.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ width, length },
                },
                .pivot = @splat(0.5),
            },
            .color = outline_color,
        });

        try this.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ length - outline_times_2, width - outline_times_2 },
                },
                .pivot = @splat(0.5),
            },
            .color = color,
        });
        try this.immediate.drawRect(.{
            .transform = .{
                .pos = .{ .norm = @splat(0.5) },
                .size = .{
                    .offset = .{ width - outline_times_2, length - outline_times_2 },
                },
                .pivot = @splat(0.5),
            },
            .color = color,
        });
    }

    {
        const hist_slot = this.info.frame_count % dt_hist_size;
        this.dt_hist[hist_slot] = dt_ns;

        const hist_count = @min(this.info.frame_count + 1, dt_hist_size);
        const hist = this.dt_hist[0..hist_count];

        const sorted = try alloc.dupe(u64, hist);
        defer alloc.free(sorted);

        std.mem.sort(u64, sorted, {}, struct {
            fn less(_: void, left: u64, right: u64) bool {
                return left < right;
            }
        }.less);

        const median = sorted[hist_count / 2];
        const median_s = math.i2f(f32, median) / std.time.ns_per_s;
        const median_fps = 1 / median_s;

        const low = sorted[hist_count / 10 * 9];
        const low_s = math.i2f(f32, low) / std.time.ns_per_s;
        const low_fps = 1 / low_s;

        const cam_euler = @mod(this.camera.euler, @as(math.Vec3, @splat(math.pi * 2)));

        const cpu_hist_count = @min(this.info.frame_count + 1, cpu_time_hist_size);
        const cpu_hist = this.cpu_time_hist[0..cpu_hist_count];

        const cpu_time_sorted = try alloc.dupe(u64, cpu_hist);
        defer alloc.free(cpu_time_sorted);

        std.mem.sort(u64, cpu_time_sorted, {}, struct {
            fn less(_: void, left: u64, right: u64) bool {
                return left < right;
            }
        }.less);

        const text = try std.fmt.allocPrint(alloc,
            \\FPS: {d: >4.0}, {d: >5.2}ms
            \\Low: {d: >4.0}, {d: >5.2}ms
            \\
            \\X:   {d: >8.2}
            \\Y:   {d: >8.2}
            \\Z:   {d: >8.2}
            \\
            \\Yaw:   {d: >6.2}
            \\Pitch: {d: >6.2}
            \\
            \\CPU Time: {d: >5}μs
            \\Mesh Buffer: {} / {}kb
            \\Meshed Chunks: {}
            \\Quad Count: {}
        , .{
            median_fps,
            median_s * std.time.ms_per_s,
            low_fps,
            low_s * std.time.ms_per_s,
            this.camera.pos[0],
            this.camera.pos[1],
            this.camera.pos[2],
            math.deg(cam_euler[2]),
            math.deg(cam_euler[1]),
            cpu_time_sorted[cpu_hist_count / 2] / 1000,
            (ChunkMeshAllocator.buffer_size - this.chunk_mesh_alloc.queryBytesFree()) / 1024,
            ChunkMeshAllocator.buffer_size / 1024,
            this.chunk_mesh_alloc.loaded_meshes.count(),
            this.chunk_mesher.face_count,
        });
        defer alloc.free(text);

        try this.immediate.drawText(.{
            .font_face = &this.font_face,
            .height_px = 24,
            .text = text,
            .pos = .{
                .norm = .{ 0, 1 },
                .offset = .{ 8, -8 },
            },
            .color = .{ 0, 0, 0, 255 },
        });
    }

    try this.glyph_cache.upload(this.device, per_frame.cmd_encoder);
    try this.immediate.render(this.device, per_frame.cmd_encoder, acquired_image.imageView(this.display));

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

fn drawChunks(this: *Renderer, render_pass: gpu.RenderPassEncoder, push_constants: PerFramePushConstants, aspect_ratio: f32) void {
    render_pass.cmdBindPipeline(this.chunk_pipeline);
    render_pass.cmdBindResourceSets(this.chunk_pipeline, &.{this.chunk_resource_set}, 0);
    render_pass.cmdBindVertexBuffer(0, this.chunk_mesh_alloc.buffer.region());
    render_pass.cmdPushConstants(this.chunk_pipeline, .{
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

const Camera = struct {
    pos: math.Vec3 = .{ 0, 0, 210 },
    euler: math.Vec3 = .{ 0, 0, 0 },
    v_fov: f32 = math.rad(90.0),
    near: f32 = 0.1,
    far: f32 = 10_000,

    fn view(this: Camera) math.Mat4 {
        return math.matMulMany(math.Mat4, .{
            math.rotateEuler(this.euler),
            math.translate(-this.pos),
        });
    }

    fn proj(this: Camera, aspect_ratio: f32) math.Mat4 {
        return math.matMul(
            math.Mat4,
            math.perspective(aspect_ratio, this.v_fov, this.near, this.far),
            math.to_vulkan,
        );
    }

    fn vp(this: Camera, aspect_ratio: f32) math.Mat4 {
        return math.matMul(math.Mat4, this.proj(aspect_ratio), this.view());
    }
};

pub const Info = struct {
    frame_count: u64,
    frames_in_flight: u32,

    pub inline fn frame_slot(info: *const Info) u32 {
        return @intCast(info.frame_count % info.frames_in_flight);
    }
};
