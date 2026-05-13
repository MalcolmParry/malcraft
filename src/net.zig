const std = @import("std");
const znet = @import("znet");
const Chunk = @import("Chunk.zig");
const block = @import("block.zig");

pub const max_packet_size = 1024 * 64;

pub const server_message = struct {
    pub const Kind = enum(u16) {
        init,
        one_to_one_chunk_data,

        /// u16 for chunk count
        /// each entry:
        ///     Chunk.PackedPos
        ///     block.Kind
        uniform_chunk_batch,

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

    pub const OneToOneChunkData = struct {
        pos: Chunk.PackedPos,
        chunk: *Chunk.OneToOne,

        pub fn encode(msg: OneToOneChunkData, writer: *std.Io.Writer) !void {
            try writer.writeInt(u16, @intFromEnum(Kind.one_to_one_chunk_data), .little);
            try writer.writeStruct(msg.pos, .little);

            for (&msg.chunk.blocks) |*plane| {
                for (plane) |*col| {
                    try writer.writeSliceEndian(block.Kind, col, .little);
                }
            }
        }

        pub fn decode(alloc: std.mem.Allocator, reader: *std.Io.Reader) !OneToOneChunkData {
            const pos = try reader.takeStruct(Chunk.PackedPos, .little);

            const one_to_one = try alloc.create(Chunk.OneToOne);
            errdefer alloc.destroy(one_to_one);

            for (&one_to_one.blocks) |*plane| {
                for (plane) |*col| {
                    try reader.readSliceEndian(block.Kind, col, .little);
                }
            }

            return .{
                .pos = pos,
                .chunk = one_to_one,
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
