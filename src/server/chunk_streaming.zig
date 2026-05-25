const std = @import("std");
const znet = @import("znet");
const options = @import("options");
const mw = @import("mwengine");
const math = mw.math;
const Deque = @import("../utils/deque.zig").Deque;
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const World = @import("../common/World.zig");
const protocol = @import("../common/protocol.zig");
const ServerMsgId = protocol.ServerMsgId;
const NetworkManager = @import("../common/NetworkManager.zig");

const zstd = @cImport({
    @cInclude("zstd.h");
});

const chunk_transfer_limit = 256 * 1024;
const region_len = 4;

pub const Cursor = struct {
    /// region is 4x4x4 chunks
    regions_to_send: Deque(Chunk.PackedPos) = .empty,

    pub fn init(cursor: *Cursor, alloc: std.mem.Allocator) !void {
        const radius_regions: i32 = @intCast(options.render_radius / region_len);
        const height_regions: i32 = @intCast(options.render_height / region_len);
        const region_count = (radius_regions * 2 + 1) * (radius_regions * 2 + 1) * (height_regions * 2 + 1);
        try cursor.regions_to_send.ensureUnusedCapacity(alloc, region_count);

        var z: i20 = 0;
        while (z <= height_regions) : (z += 1) {
            var y: i22 = -radius_regions;
            while (y <= radius_regions) : (y += 1) {
                var x: i22 = -radius_regions;
                while (x <= radius_regions) : (x += 1) {
                    const pos: Chunk.PackedPos = .{ .x = x, .y = y, .z = z };
                    cursor.regions_to_send.pushBackAssumeCapacity(pos);
                }
            }
        }

        std.mem.sort(Chunk.PackedPos, cursor.regions_to_send.buffer[0..cursor.regions_to_send.len], {}, struct {
            fn lessThanFn(_: void, left: Chunk.PackedPos, right: Chunk.PackedPos) bool {
                const i64x3 = @Vector(3, i64);
                const left_64: i64x3 = left.vec();
                const right_64: i64x3 = right.vec();
                return math.lengthSqr(left_64) < math.lengthSqr(right_64);
            }
        }.lessThanFn);
    }

    pub fn deinit(cursor: *Cursor, alloc: std.mem.Allocator) void {
        cursor.regions_to_send.deinit(alloc);
    }
};

pub fn sendChunks(net_man: *NetworkManager, world: *const World, peer: NetworkManager.PeerRef, cursor: *Cursor) !void {
    if (cursor.regions_to_send.len == 0) return;

    var timer: std.time.Timer = try .start();

    var uniform_buffer: [protocol.max_packet_size]u8 = undefined;
    var uniform_writer = std.Io.Writer.fixed(&uniform_buffer);

    var compressed_buffer: [protocol.max_packet_size]u8 = undefined;
    var compressed_writer = std.Io.Writer.fixed(&compressed_buffer);

    var send_state: SendState = .{
        .net_man = net_man,
        .peer = peer,
        .uniform_writer = &uniform_writer,
        .compressed_writer = &compressed_writer,
    };
    try send_state.init();

    var regions_sent: usize = 0;
    while (cursor.regions_to_send.popFront()) |packed_region_pos| {
        const region_pos = packed_region_pos.vec();

        for (0..region_len) |ux| {
            for (0..region_len) |uy| {
                for (0..region_len) |uz| {
                    const chunk_pos = (region_pos * @as(Chunk.Pos, @splat(region_len))) + @as(Chunk.Pos, .{
                        @intCast(ux),
                        @intCast(uy),
                        @intCast(uz),
                    });

                    const packed_chunk_pos: Chunk.PackedPos = .pack(chunk_pos);
                    const chunk = world.getChunk(packed_chunk_pos).?;
                    try send_state.send(packed_chunk_pos, chunk);
                }
            }
        }

        regions_sent += 1;
        if (send_state.totalBytesToSend() >= chunk_transfer_limit)
            break;
    }

    try send_state.flush();
    std.log.info("{}μs to send {} regions ({} chunks)", .{ timer.read() / 1000, regions_sent, regions_sent * 4 * 4 * 4 });
}

const SendState = struct {
    net_man: *NetworkManager,
    peer: NetworkManager.PeerRef,
    bytes_sent: usize = 0,

    uniform_writer: *std.Io.Writer,
    uniform_chunk_count_pos: usize = 0,
    uniform_chunk_count: u16 = 0,

    compressed_writer: *std.Io.Writer,
    compressed_chunk_count_pos: usize = 0,
    compressed_chunk_count: u16 = 0,

    pub fn init(state: *SendState) !void {
        try state.initUniform();
        try state.initCompressed();
    }

    pub fn flush(state: *SendState) !void {
        try state.flushUniform();
        try state.flushCompressed();
    }

    pub fn send(state: *SendState, pos: Chunk.PackedPos, chunk: Chunk) !void {
        switch (chunk.data) {
            .uniform => |kind| try state.sendUniform(pos, kind),
            .u2_palette,
            .one_to_one,
            => try state.sendCompressed(pos, chunk),
        }
    }

    pub fn totalBytesToSend(state: *const SendState) usize {
        return state.bytes_sent + state.uniform_writer.end + state.compressed_writer.end;
    }

    pub fn initUniform(state: *SendState) !void {
        const writer = state.uniform_writer;
        writer.end = 0;

        try ServerMsgId.uniform_chunk_batch.encode(state.uniform_writer);
        state.uniform_chunk_count_pos = writer.end;
        try writer.writeInt(u16, 0, .little);
        state.uniform_chunk_count = 0;
    }

    pub fn flushUniform(state: *SendState) !void {
        if (state.uniform_chunk_count == 0) return;

        const writer = state.uniform_writer;
        const end = writer.end;
        writer.end = state.uniform_chunk_count_pos;
        try writer.writeInt(u16, state.uniform_chunk_count, .little);
        writer.end = end;

        const channel = protocol.Channel.chunk_transfer;
        const packet = try znet.Packet.init(writer.buffered(), channel.toInt(), channel.getFlags());
        state.bytes_sent += packet.dataSlice().len;
        try state.net_man.send(state.peer, packet);
    }

    pub fn sendUniform(state: *SendState, pos: Chunk.PackedPos, kind: block.Kind) !void {
        const writer = state.uniform_writer;
        if (writer.end >= protocol.chunk_batch_target_size) {
            try state.flushUniform();
            try state.initUniform();
        }

        try writer.writeStruct(pos, .little);
        try writer.writeInt(u8, @intFromEnum(kind), .little);
        state.uniform_chunk_count += 1;
    }

    pub fn initCompressed(state: *SendState) !void {
        const writer = state.compressed_writer;
        writer.end = 0;

        try ServerMsgId.compressed_chunk_batch.encode(writer);
        state.compressed_chunk_count_pos = writer.end;
        try writer.writeInt(u16, 0, .little);
        state.compressed_chunk_count = 0;
    }

    pub fn flushCompressed(state: *SendState) !void {
        if (state.compressed_chunk_count == 0) return;

        const writer = state.compressed_writer;
        const end = writer.end;
        writer.end = state.compressed_chunk_count_pos;
        try writer.writeInt(u16, state.compressed_chunk_count, .little);
        writer.end = end;

        const channel = protocol.Channel.chunk_transfer;
        const packet = try znet.Packet.init(writer.buffered(), channel.toInt(), channel.getFlags());
        state.bytes_sent += packet.dataSlice().len;
        try state.net_man.send(state.peer, packet);
    }

    pub fn sendCompressed(state: *SendState, pos: Chunk.PackedPos, chunk: Chunk) !void {
        const writer = state.compressed_writer;
        if (writer.end >= protocol.chunk_batch_target_size) {
            try state.flushCompressed();
            try state.initCompressed();
        }

        try writer.writeStruct(pos, .little);
        switch (chunk.data) {
            .uniform => unreachable,
            .u2_palette => |data| {
                try writer.writeInt(u8, @intFromEnum(protocol.ChunkStorageType.u2_palette), .little);
                const compressed_size_pos = writer.end;
                try writer.writeInt(u16, 0, .little);

                const compress_buffer = writer.unusedCapacitySlice();
                const compressed_size = zstd.ZSTD_compress(compress_buffer.ptr, compress_buffer.len, data, @sizeOf(Chunk.U2Palette), 1);
                if (zstd.ZSTD_isError(compressed_size) != 0) return error.ZstdCompressFailed;

                const end = writer.end + compressed_size;
                writer.end = compressed_size_pos;
                try writer.writeInt(u16, @intCast(compressed_size), .little);
                writer.end = end;
                state.compressed_chunk_count += 1;
            },
            .one_to_one => |data| {
                try writer.writeInt(u8, @intFromEnum(protocol.ChunkStorageType.u4), .little);
                const compressed_size_pos = writer.end;
                try writer.writeInt(u16, 0, .little);

                const compress_buffer = writer.unusedCapacitySlice();
                const compressed_size = zstd.ZSTD_compress(compress_buffer.ptr, compress_buffer.len, data, @sizeOf(Chunk.OneToOne), 1);
                if (zstd.ZSTD_isError(compressed_size) != 0) return error.ZstdCompressFailed;

                const end = writer.end + compressed_size;
                writer.end = compressed_size_pos;
                try writer.writeInt(u16, @intCast(compressed_size), .little);
                writer.end = end;
                state.compressed_chunk_count += 1;
            },
        }
    }
};
