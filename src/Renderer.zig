const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;

event_queue: mw.EventQueue,
window: mw.Window,

pub fn init(this: *@This(), alloc: std.mem.Allocator) !void {
    this.event_queue = try .init(alloc);
    errdefer this.event_queue.deinit();

    this.window = try .init(alloc, "malcraft", .{ 100, 100 }, &this.event_queue);
    errdefer this.window.deinit();
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.window.deinit();
    this.event_queue.deinit();
    _ = alloc;
}

pub fn render(this: *@This(), alloc: std.mem.Allocator) !bool {
    _ = alloc;
    this.window.update();
    while (this.event_queue.pending()) {
        switch (this.event_queue.pop()) {
            else => {},
        }
    }

    return !this.window.shouldClose();
}
