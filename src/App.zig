const std = @import("std");
const Renderer = @import("Renderer.zig");

renderer: Renderer,

pub fn init(this: *@This(), alloc: std.mem.Allocator) !void {
    try this.renderer.init(alloc);
    errdefer this.renderer.deinit(alloc);
}

pub fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
    this.renderer.deinit(alloc);
}
