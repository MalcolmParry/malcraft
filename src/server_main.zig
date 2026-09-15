const std = @import("std");
const Server = @import("server/Server.zig");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var server: Server = undefined;
    try server.init(alloc, io);
    defer server.deinit();

    while (try server.tick()) {}
}
