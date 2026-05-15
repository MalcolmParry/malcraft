const std = @import("std");
const znet = @import("znet");
const Chunk = @import("Chunk.zig");
const block = @import("block.zig");

pub const max_packet_size = 1024 * 64;
pub const chunk_batch_target_size = 1024 * 32;

/// server -> client message
/// everything is little endian
/// every message starts with:
///     id: u8,
pub const ServerMsgId = enum(u8) {
    /// player_id: u16,
    init,

    /// chunk_count: u16,
    /// entries: [chunk_count],
    ///     pos: Chunk.PackedPos,
    ///     kind: block.Kind,
    uniform_chunk_batch,

    /// chunk_count: u16,
    /// entries: [chunk_count],
    ///     pos: Chunk.PackedPos,
    ///     compressed_size: u16,
    ///     compressed_bytes: [compressed_size]u8,
    compressed_chunk_batch,

    pub fn encode(id: ServerMsgId, writer: *std.Io.Writer) !void {
        try writer.writeInt(u8, @intFromEnum(id), .little);
    }

    pub fn decode(reader: *std.Io.Reader) !ServerMsgId {
        const int = try reader.takeInt(u8, .little);
        return std.enums.fromInt(ServerMsgId, int) orelse error.BadMessage;
    }
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
