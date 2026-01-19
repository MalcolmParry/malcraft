const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;

event_queue: mw.EventQueue,
window: mw.Window,
instance: gpu.Instance,
device: gpu.Device,
display: gpu.Display,
image_available_semaphore: gpu.Semaphore,
render_finished_semaphore: gpu.Semaphore,
presented_fence: gpu.Fence,
cmd_encoder: gpu.CommandEncoder,

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

    this.image_available_semaphore = try .init(this.device);
    errdefer this.image_available_semaphore.deinit(this.device);

    this.render_finished_semaphore = try .init(this.device);
    errdefer this.render_finished_semaphore.deinit(this.device);

    this.presented_fence = try .init(this.device, true);
    errdefer this.presented_fence.deinit(this.device);

    this.cmd_encoder = try .init(this.device);
    errdefer this.cmd_encoder.deinit(this.device);
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.device.waitUntilIdle();
    this.cmd_encoder.deinit(this.device);
    this.presented_fence.deinit(this.device);
    this.render_finished_semaphore.deinit(this.device);
    this.image_available_semaphore.deinit(this.device);
    this.display.deinit(alloc);
    this.device.deinit(alloc);
    this.instance.deinit(alloc);
    this.window.deinit();
    this.event_queue.deinit();
}

pub fn render(this: *@This(), alloc: std.mem.Allocator) !void {
    _ = alloc;
    try this.presented_fence.wait(this.device, std.time.ns_per_s);
    try this.presented_fence.reset(this.device);

    const image_index = switch (try this.display.acquireImageIndex(this.image_available_semaphore, null, std.time.ns_per_s)) {
        .success => |x| x,
        .suboptimal => |x| x,
        .out_of_date => return error.Failed,
    };

    try this.cmd_encoder.begin(this.device);
    this.cmd_encoder.cmdMemoryBarrier(this.device, &.{.{
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

    this.cmd_encoder.cmdMemoryBarrier(this.device, &.{
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

    try this.cmd_encoder.end(this.device);
    try this.cmd_encoder.submit(this.device, &.{this.image_available_semaphore}, &.{this.render_finished_semaphore}, null);

    _ = try this.display.presentImage(image_index, &.{this.render_finished_semaphore}, this.presented_fence);
}
