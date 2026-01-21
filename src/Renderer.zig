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
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.device.waitUntilIdle();
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
