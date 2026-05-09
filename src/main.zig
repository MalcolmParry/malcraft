const std = @import("std");
const App = @import("App.zig");
const Renderer = @import("Renderer.zig");

pub fn main() !void {
    var alloc_obj = std.heap.DebugAllocator(.{}).init;
    defer _ = alloc_obj.deinit();
    const alloc = alloc_obj.allocator();

    var app: App = undefined;
    try app.init(alloc);
    defer app.deinit(alloc);

    while (!app.should_close) {
        try app.tick(alloc);
    }
}
