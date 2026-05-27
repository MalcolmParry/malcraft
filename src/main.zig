const std = @import("std");
const options = @import("options");
const builtin = @import("builtin");
const App = @import("client/App.zig");
const Server = if (options.build_server) @import("server/Server.zig") else void;

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

const GPA = struct {
    const debug = switch (builtin.mode) {
        .Debug, .ReleaseSafe => true,
        else => false,
    };

    debug_alloc: if (debug) std.heap.DebugAllocator(.{}) else void,

    pub fn init() GPA {
        return if (debug)
            .{ .debug_alloc = .init }
        else
            .{ .debug_alloc = {} };
    }

    pub fn deinit(gpa: *GPA) void {
        if (debug)
            _ = gpa.debug_alloc.deinit();
    }

    pub fn allocator(gpa: *GPA) std.mem.Allocator {
        return if (debug)
            gpa.debug_alloc.allocator()
        else
            std.heap.smp_allocator;
    }
};
