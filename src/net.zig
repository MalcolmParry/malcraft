const std = @import("std");
const znet = @import("znet");
const Chunk = @import("Chunk.zig");
const block = @import("block.zig");

pub const server_message = struct {
    pub const Kind = enum(u16) {
        init,
        chunk_data,

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

    pub const ChunkData = struct {
        pos: Chunk.Pos,
        chunk: Chunk,

        const StorageType = enum(u8) {
            single,
            one_to_one,
        };

        pub fn encode(msg: ChunkData, writer: *std.Io.Writer) !void {
            try writer.writeInt(u16, @intFromEnum(Kind.chunk_data), .little);

            try writer.writeInt(i32, msg.pos[0], .little);
            try writer.writeInt(i32, msg.pos[1], .little);
            try writer.writeInt(i32, msg.pos[2], .little);

            switch (msg.chunk.data) {
                .single => |kind| {
                    try writer.writeInt(u8, @intFromEnum(StorageType.single), .little);
                    try writer.writeInt(u8, @intFromEnum(kind), .little);
                },
                .one_to_one => |one_to_one| {
                    try writer.writeInt(u8, @intFromEnum(StorageType.one_to_one), .little);

                    for (&one_to_one.blocks) |*plane| {
                        for (plane) |*col| {
                            try writer.writeSliceEndian(block.Kind, col, .little);
                        }
                    }
                },
            }
        }

        pub fn decode(alloc: std.mem.Allocator, reader: *std.Io.Reader) !ChunkData {
            const x = try reader.takeInt(i32, .little);
            const y = try reader.takeInt(i32, .little);
            const z = try reader.takeInt(i32, .little);
            const pos: Chunk.Pos = .{ x, y, z };

            const storage_type_i = try reader.takeInt(u8, .little);
            const storage_type = std.enums.fromInt(StorageType, storage_type_i) orelse return error.BadMessage;

            const chunk: Chunk = switch (storage_type) {
                .single => .{ .data = .{
                    .single = std.enums.fromInt(
                        block.Kind,
                        try reader.takeInt(u8, .little),
                    ) orelse return error.BadMessage,
                } },
                .one_to_one => blk: {
                    const one_to_one = try alloc.create(Chunk.OneToOne);
                    errdefer alloc.destroy(one_to_one);

                    for (&one_to_one.blocks) |*plane| {
                        for (plane) |*col| {
                            try reader.readSliceEndian(block.Kind, col, .little);
                        }
                    }

                    break :blk .{ .data = .{
                        .one_to_one = one_to_one,
                    } };
                },
            };

            return .{
                .pos = pos,
                .chunk = chunk,
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
