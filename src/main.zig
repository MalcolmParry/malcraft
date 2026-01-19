const std = @import("std");
const App = @import("App.zig");

pub fn main() !void {
    const alloc = std.heap.smp_allocator;

    var app: App = undefined;
    try app.init(alloc);
    defer app.deinit(alloc);
    const renderer = &app.renderer;
    while (true) {
        try renderer.render(alloc);

        renderer.window.update();
        while (renderer.event_queue.pending()) {
            switch (renderer.event_queue.pop()) {
                else => {},
            }
        }

        if (renderer.window.shouldClose()) break;
    }
}
