const std = @import("std");
const App = @import("App.zig");

pub fn main() !void {
    const alloc = std.heap.smp_allocator;

    var app: App = undefined;
    try app.init(alloc);
    defer app.deinit(alloc);
    while (try app.renderer.render(alloc)) {}
}
