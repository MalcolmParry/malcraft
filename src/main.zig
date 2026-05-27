const std = @import("std");
const options = @import("options");
const builtin = @import("builtin");
const App = @import("client/App.zig");
const Server = if (options.build_server) @import("server/Server.zig") else void;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    if (comptime options.build_server) {
        var server: Server = undefined;
        try server.init(alloc, io);
        defer server.deinit();

        while (try server.tick()) {}
    } else {
        var app: App = undefined;
        try app.init(alloc, io);
        defer app.deinit();

        while (!app.should_close) {
            try app.tick();
        }
    }
}
