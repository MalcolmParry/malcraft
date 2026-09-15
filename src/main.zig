const std = @import("std");
const builtin = @import("builtin");
const App = @import("client/App.zig");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var opts: App.Options = .{
        .render_radius = switch (builtin.mode) {
            .ReleaseFast, .ReleaseSafe => 64,
            else => 3,
        },
        .render_height = switch (builtin.mode) {
            .ReleaseFast, .ReleaseSafe => 16,
            else => 8,
        },
        .ip = "localhost",
        .port = 5000,
    };

    var iter = try init.minimal.args.iterateAllocator(alloc);
    defer iter.deinit();
    _ = iter.next();

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-rr")) {
            const next = iter.next() orelse return error.BadArgs;
            opts.render_radius = try std.fmt.parseInt(u32, next, 10);
        } else if (std.mem.eql(u8, arg, "-rh")) {
            const next = iter.next() orelse return error.BadArgs;
            opts.render_height = try std.fmt.parseInt(u32, next, 10);
        } else if (std.mem.eql(u8, arg, "-ip")) {
            const next = iter.next() orelse return error.BadArgs;
            opts.ip = next;
        } else if (std.mem.eql(u8, arg, "-port")) {
            const next = iter.next() orelse return error.BadArgs;
            opts.port = try std.fmt.parseInt(u16, next, 10);
        } else return error.BadArgs;
    }

    var app: App = undefined;
    try app.init(alloc, io, opts);
    defer app.deinit();

    while (!app.should_close) {
        try app.tick();
    }
}
