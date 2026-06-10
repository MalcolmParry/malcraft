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
    ///     storage_type: ChunkStorageType,
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

/// server -> client message
/// everything is little endian
/// every message starts with:
///     id: u8,
pub const ClientMsgId = enum(u8) {
    /// chunk_pos: Chunk.PackedPos,
    update_chunk_cursor,

    pub fn encode(id: ClientMsgId, writer: *std.Io.Writer) !void {
        try writer.writeInt(u8, @intFromEnum(id), .little);
    }

    pub fn decode(reader: *std.Io.Reader) !ClientMsgId {
        const int = try reader.takeInt(u8, .little);
        return std.enums.fromInt(ClientMsgId, int) orelse error.BadMessage;
    }
};

pub const ChunkStorageType = enum(u8) {
    u2_palette,
    u4,
};

pub const Channel = enum(u8) {
    control,
    chunk_transfer,

    pub fn toInt(channel: Channel) u8 {
        return @intFromEnum(channel);
    }

    pub fn getFlags(channel: Channel) znet.Flags {
        return switch (channel) {
            .control,
            .chunk_transfer,
            => .reliable,
        };
    }
};
