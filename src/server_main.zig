const std = @import("std");
const Server = @import("server/Server.zig");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const opts: Server.Options = .{
        .chunk_streaming_radius = 256,
        .chunk_streaming_height = 64,
        .ip = "0.0.0.0",
        .port = 5000,
    };

    var server: Server = undefined;
    try server.init(alloc, io, opts);
    defer server.deinit();

    while (try server.tick()) {}
}
