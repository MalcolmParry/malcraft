const std = @import("std");
const App = @import("client/App.zig");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var app: App = undefined;
    try app.init(alloc, io);
    defer app.deinit();

    while (!app.should_close) {
        try app.tick();
    }
}
