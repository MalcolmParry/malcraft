const std = @import("std");
const App = @import("App.zig");

pub fn main() !void {
    const alloc = std.heap.smp_allocator;

    var should_close: bool = false;
    var app: App = undefined;
    try app.init(alloc);
    defer app.deinit(alloc);
    const renderer = &app.renderer;
    while (!(should_close or renderer.window.shouldClose())) {
        renderer.window.update();
        while (renderer.event_queue.pending()) {
            switch (renderer.event_queue.pop()) {
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
