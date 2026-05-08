const std = @import("std");
const znet = @import("znet");
const net = @import("net.zig");
const Renderer = @import("Renderer.zig");
const App = @This();

host: znet.Host,
server: znet.Peer,

renderer: Renderer,

pub fn init(app: *App, alloc: std.mem.Allocator) !void {
    try znet.init();
    errdefer znet.deinit();

    const host = try znet.Host.init(.{
        .addr = null,
        .peer_limit = 1,
        .channel_limit = .{ .count = 1 },
        .incoming_bandwidth = .unlimited,
        .outgoing_bandwidth = .unlimited,
    });
    errdefer host.deinit();

    const peer = try host.connect(.{
        .addr = try .init(.{
            .ip = .{ .ipv4 = "localhost" },
            .port = .{ .uint = 5000 },
        }),
        .channel_limit = .{ .count = 1 },
        .data = 0,
    });

    app.* = .{
        .host = host,
        .server = peer,
        .renderer = undefined,
    };

    try app.renderer.init(alloc);
    errdefer app.renderer.deinit(alloc);
}

pub fn deinit(app: *App, alloc: std.mem.Allocator) void {
    app.renderer.deinit(alloc);

    app.host.deinit();
    znet.deinit();
}

pub fn tick(app: *App, alloc: std.mem.Allocator) !void {
    while (try app.host.service(0)) |any_event| {
        try app.handleNetworkEvent(alloc, any_event);
    }
}

fn handleNetworkEvent(app: *App, alloc: std.mem.Allocator, any_event: znet.Event) !void {
    _ = alloc;

    switch (any_event) {
        .connect => |event| {
            const addr: znet.Address = .{ .inner = event.peer.ptr.address };
            std.log.info("connected to server at {x} on {}", .{ addr.inner.host, addr.inner.port });
        },
        .disconnect => |_| {
            const addr: znet.Address = .{ .inner = app.server.ptr.address };
            std.log.info("disconnected from server at {x} on {}", .{ addr.inner.host, addr.inner.port });

            return error.ServerClosed;
        },
        .receive => |event| {
            defer event.packet.deinit();
            var reader = event.packet.reader();
            const kind = try net.server_message.Kind.decode(&reader);

            switch (kind) {
                .init => {
                    const msg = try net.server_message.Init.decode(&reader);

                    std.log.info("playerid: {}", .{msg.player_id});
                },
            }
        },
    }
}
