const std = @import("std");
const options = @import("options");
const builtin = @import("builtin");
const App = @import("client/App.zig");
const Server = @import("server/Server.zig");
const GPA = @import("utils/GPA.zig");

pub fn main() !void {
    var gpa_obj: GPA = .init();
    defer gpa_obj.deinit();
    const alloc = gpa_obj.allocator();

    if (comptime options.build_server) {
        var server: Server = undefined;
        try server.init(alloc);
        defer server.deinit();

        while (try server.tick()) {}
    } else {
        var app: App = undefined;
        try app.init(alloc);
        defer app.deinit(alloc);

        while (!app.should_close) {
            try app.tick(alloc);
        }
    }
}
