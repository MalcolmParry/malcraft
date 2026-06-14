const std = @import("std");
const znet = @import("znet");
const options = @import("options");
const mw = @import("mwengine");
const math = mw.math;
const Deque = @import("../utils/deque.zig").Deque;
const Aabb = @import("../utils/Aabb.zig");
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const region = @import("../common/region.zig");
const World = @import("../common/World.zig");
const protocol = @import("../common/protocol.zig");
const ServerMsgId = protocol.ServerMsgId;
const NetworkManager = @import("../common/NetworkManager.zig");

const zstd = @cImport({
    @cInclude("zstd.h");
});

const chunk_transfer_limit = 256 * 1024;

pub const Cursor = struct {
    /// region is 4x4x4 chunks
    regions_to_send: Deque(region.PackedPos) = .empty,
    regions_to_gen: Deque(region.PackedPos) = .empty,
    chunks_to_send: Deque(Chunk.PackedPos) = .empty,
    chunks_to_gen: Deque(Chunk.PackedPos) = .empty,

    pos: Chunk.PackedPos = .pack(@splat(0)),
    render_radius: u32 = @max(options.render_radius / region.len, 1),
    render_height: u32 = @max(options.render_height / region.len, 1),

    pub fn init(cursor: *Cursor, alloc: std.mem.Allocator) !void {
        try cursor.queueSendAabb(alloc, cursor.loadedAabb());

        const SortContext = struct {
            queue: *Deque(Chunk.PackedPos),

            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const i64x3 = @Vector(3, i64);
                const left_64: i64x3 = ctx.queue.at(a).vec();
                const right_64: i64x3 = ctx.queue.at(b).vec();
                return math.lengthSqr(left_64) < math.lengthSqr(right_64);
            }

            pub fn swap(ctx: @This(), a: usize, b: usize) void {
                return std.mem.swap(region.PackedPos, ctx.queue.atPtr(a), ctx.queue.atPtr(b));
            }
        };

        std.sort.heapContext(0, cursor.regions_to_send.len, SortContext{
            .queue = &cursor.regions_to_send,
        });
    }

    pub fn deinit(cursor: *Cursor, alloc: std.mem.Allocator) void {
        cursor.regions_to_send.deinit(alloc);
        cursor.regions_to_gen.deinit(alloc);
        cursor.chunks_to_send.deinit(alloc);
        cursor.chunks_to_gen.deinit(alloc);
    }

    pub fn updatePos(cursor: *Cursor, alloc: std.mem.Allocator, new: Chunk.Pos) !void {
        if (@reduce(.And, cursor.pos.vec() == new)) return;

        const old_box = cursor.loadedAabb();
        cursor.pos = .pack(new);
        const new_box = cursor.loadedAabb();

        var buffer: [6]Aabb = undefined;
        const to_load_regions = new_box.subtract(old_box, &buffer);

        for (to_load_regions) |x| {
            try cursor.queueSendAabb(alloc, x);
        }
    }

    pub fn chunkInRange(cursor: *const Cursor, pos: Chunk.Pos) bool {
        const region_pos = pos / region.size;
        return cursor.regionInRange(region_pos);
    }

    pub fn regionInRange(cursor: *const Cursor, pos: Chunk.Pos) bool {
        const rel = pos - cursor.pos.vec();
        if (@abs(rel[0]) > cursor.render_radius) return false;
        if (@abs(rel[1]) > cursor.render_radius) return false;
        if (@abs(rel[2]) > cursor.render_height) return false;
        return true;
    }

    pub fn loadedAabb(cursor: *const Cursor) Aabb {
        const bounds: Chunk.Pos = .{ @intCast(cursor.render_radius), @intCast(cursor.render_radius), @intCast(cursor.render_height) };

        return .{
            .min = cursor.pos.vec() - bounds,
            .max = cursor.pos.vec() + bounds,
        };
    }

    pub fn queueSendAabb(cursor: *Cursor, alloc: std.mem.Allocator, aabb: Aabb) !void {
        try cursor.regions_to_send.ensureUnusedCapacity(alloc, aabb.volume());

        var x = aabb.min[0];
        while (x < aabb.max[0]) : (x += 1) {
            var y = aabb.min[1];
            while (y < aabb.max[1]) : (y += 1) {
                var z = aabb.min[2];
                while (z < aabb.max[2]) : (z += 1) {
                    const pos: Chunk.Pos = .{ x, y, z };
                    cursor.regions_to_send.pushBackAssumeCapacity(.pack(pos));
                }
            }
        }
    }
};

pub fn sendChunks(alloc: std.mem.Allocator, net_man: *NetworkManager, world: *const World, peer: NetworkManager.PeerRef, cursor: *Cursor) !void {
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

    var chunks_sent: usize = 0;
    while (send_state.totalBytesToSend() < chunk_transfer_limit) {
        const pos_p = cursor.chunks_to_send.popFront() orelse break;
        if (!cursor.chunkInRange(pos_p.vec())) continue;

        const chunk = world.getChunk(pos_p) orelse {
            try cursor.chunks_to_gen.pushBack(alloc, pos_p);
            continue;
        };

        try send_state.send(pos_p, chunk);
        chunks_sent += 1;
    }

    while (send_state.totalBytesToSend() < chunk_transfer_limit) {
        const region_pos_p = cursor.regions_to_send.popFront() orelse break;
        const region_pos = region_pos_p.vec();
        if (!cursor.regionInRange(region_pos)) continue;

        var exists: bool = false;
        for (0..region.len) |ux| blk: {
            for (0..region.len) |uy| {
                for (0..region.len) |uz| {
                    const chunk_pos = (region_pos * region.size) + @as(Chunk.Pos, .{
                        @intCast(ux),
                        @intCast(uy),
                        @intCast(uz),
                    });

                    if (world.containsChunk(.pack(chunk_pos))) {
                        exists = true;
                        break :blk;
                    }
                }
            }
        }

        if (!exists) {
            try cursor.regions_to_gen.pushBack(alloc, region_pos_p);
            continue;
        }

        for (0..region.len) |ux| {
            for (0..region.len) |uy| {
                for (0..region.len) |uz| {
                    const chunk_pos = (region_pos * region.size) + @as(Chunk.Pos, .{
                        @intCast(ux),
                        @intCast(uy),
                        @intCast(uz),
                    });

                    const packed_chunk_pos: Chunk.PackedPos = .pack(chunk_pos);
                    const chunk = world.getChunk(packed_chunk_pos) orelse {
                        try cursor.chunks_to_gen.pushBack(alloc, packed_chunk_pos);
                        continue;
                    };

                    try send_state.send(packed_chunk_pos, chunk);
                    chunks_sent += 1;
                }
            }
        }
    }

    try send_state.flush();
    std.log.info("{}μs to send {} chunks in {Bi:.0}", .{ timer.read() / 1000, chunks_sent, send_state.totalBytesToSend() });
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
