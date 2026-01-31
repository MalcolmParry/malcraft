const std = @import("std");
const App = @import("App.zig");

pub fn main() !void {
    var alloc_obj = std.heap.DebugAllocator(.{}).init;
    defer _ = alloc_obj.deinit();
    const alloc = alloc_obj.allocator();

    var should_close: bool = false;
    var app: App = undefined;
    try app.init(alloc);
    defer app.deinit(alloc);
    const renderer = &app.renderer;
    while (!(should_close or renderer.window.shouldClose())) {
        renderer.window.update();
        while (renderer.event_queue.pending()) {
            switch (renderer.event_queue.pop()) {
                .resize => |_| renderer.dirty_swapchain = true,
                .key_down => |key| {
                    switch (key) {
                        .escape => should_close = true,
                        else => {},
                    }
                },
                else => {},
            }
        }

        try renderer.render(alloc);
    }
}
