const std = @import("std");
const znet = @import("znet");

pub fn main() !void {
    var alloc_obj = std.heap.DebugAllocator(.{}).init;
    defer _ = alloc_obj.deinit();
    const alloc = alloc_obj.allocator();
    _ = alloc;

    try znet.init();
    defer znet.deinit();

    const host = try znet.Host.init(.{
        .addr = try .init(.{
            .ip = .any,
            .port = .{ .uint = 5000 },
        }),
        .peer_limit = 32,
        .channel_limit = .{ .count = 1 },
        .incoming_bandwidth = .unlimited,
        .outgoing_bandwidth = .unlimited,
    });
    defer shutdown(host);

    std.log.info("server started", .{});

    while (true) {
        const anyevent = try host.service(100) orelse continue;

        switch (anyevent) {
            .connect => |data| {
                const address: znet.Address = .{ .inner = data.peer.ptr.address };
                std.log.info("connection from {x} on {}", .{ address.inner.host, address.inner.port });
            },
            .disconnect => |data| {
                _ = data;
            },
            .receive => |data| {
                defer data.packet.deinit();
            },
        }
    }

    std.log.info("server stopped", .{});
}

fn shutdown(host: znet.Host) void {
    defer host.deinit();

    var iter = host.iterPeers();
    while (iter.next()) |peer| {
        peer.disconnect(0);
    }

    var timer = std.time.Timer.start() catch return;
    while (timer.read() < std.time.ns_per_s * 3) {
        while (host.service(100) catch return) |event| switch (event) {
            .connect => |data| {
                data.peer.disconnect(0);
            },
            .disconnect => {},
            .receive => |data| {
                data.packet.deinit();
            },
        };

        var remaining: bool = false;
        iter = host.iterPeers();
        while (iter.next()) |peer| {
            if (peer.state() != .disconnected) {
                remaining = true;
            }
        }

        if (!remaining) return;
    }
}
