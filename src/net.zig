const std = @import("std");
const znet = @import("znet");

pub const server_message = struct {
    pub const Kind = enum(u16) {
        init,

        pub fn decode(reader: *std.Io.Reader) !Kind {
            const int = try reader.takeInt(u16, .little);
            return std.meta.intToEnum(Kind, int) catch return error.BadMessage;
        }
    };

    pub const Init = struct {
        player_id: u32,

        pub fn encode(msg: Init, writer: *std.Io.Writer) !void {
            try writer.writeInt(u16, @intFromEnum(Kind.init), .little);

            try writer.writeInt(u32, msg.player_id, .little);
        }

        pub fn decode(reader: *std.Io.Reader) !Init {
            const player_id = try reader.takeInt(u32, .little);

            return .{
                .player_id = player_id,
            };
        }
    };
};

pub fn shutdown(host: znet.Host) void {
    defer host.deinit();

    var iter = host.iterPeers();
    while (iter.next()) |peer| {
        peer.disconnect(0);
    }

    var timer = std.time.Timer.start() catch return;
    while (timer.read() < std.time.ns_per_s * 3) {
        while (host.service(100) catch return) |event| switch (event) {
            .connect => |data| {
                data.peer.reset();
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
