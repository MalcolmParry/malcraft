const std = @import("std");
const builtin = @import("builtin");
const znet = @import("znet");
const SparseSet = @import("../utils/sparse_set.zig").SparseSet;
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const World = @import("../common/World.zig");
const WorldGenerator = @import("../server/WorldGenerator.zig");
const protocol = @import("../common/protocol.zig");
const ServerMsgId = protocol.ServerMsgId;
const NetworkManager = @import("../common/NetworkManager.zig");
const Server = @This();

const zstd = @cImport({
    @cInclude("zstd.h");
});

const tps = 20;
const ms_per_tick = 50;
const ns_per_tick = ms_per_tick * std.time.ns_per_ms;

alloc: std.mem.Allocator,
stdin_thread: std.Thread,
console_input: ConsoleInput,
net_man: NetworkManager,

world: World = .{},
world_gen: WorldGenerator,

tick_timer: std.time.Timer,
tick_count: u64 = 0,
total_work_ns: u64 = 0,
total_bytes_sent: usize = 0,
players: SparseSet(Player) = .empty,

pub fn init(server: *Server, alloc: std.mem.Allocator) !void {
    server.* = .{
        .alloc = alloc,
        .stdin_thread = undefined,
        .console_input = .{},
        .net_man = undefined,

        .world_gen = undefined,
        .tick_timer = try .start(),
    };
    errdefer server.world.deinit(alloc);

    server.stdin_thread = try .spawn(.{}, ConsoleInput.worker, .{&server.console_input});
    errdefer server.stdin_thread.detach();
    server.stdin_thread.setName("stdin") catch |err|
        std.log.warn("failed to set thread name: {}", .{err});

    try znet.init();
    errdefer znet.deinit();

    try server.net_man.init(alloc, .{
        .addr = try .init(.{
            .ip = .any,
            .port = .{ .uint = 5000 },
        }),
        .channel_limit = .{ .count = std.enums.values(protocol.Channel).len },
        .peer_limit = 32,
        .incoming_bandwidth = .unlimited,
        .outgoing_bandwidth = .{ .bps = 64 * 1024 * 1024 },
    });
    errdefer server.net_man.deinit();
    std.log.info("server started", .{});

    server.world_gen = try .init(alloc);
    errdefer server.world_gen.deinit();
    try server.world_gen.queueChunks();
    while (server.world_gen.queue.len != 0)
        try server.world_gen.genMany(alloc, &server.world);

    std.log.info("world gen done", .{});
}

pub fn deinit(server: *Server) void {
    const alloc = server.alloc;

    std.log.info("server stopped", .{});
    std.log.info("tick count: {}", .{server.tick_count});
    std.log.info("mean work time per tick: {}ns", .{std.math.divFloor(u64, server.total_work_ns, server.tick_count) catch 0});
    std.log.info("total bytes sent: {Bi:.2}", .{server.total_bytes_sent});

    server.world_gen.deinit();
    server.world.deinit(alloc);
    server.players.deinit(alloc);
    server.net_man.deinit();
    znet.deinit();
    server.stdin_thread.detach();
}

pub fn tick(server: *Server) !bool {
    _ = server.tick_timer.lap();

    no_cmd: {
        server.console_input.mutex.lock();
        defer server.console_input.mutex.unlock();

        if (server.console_input.len == 0) break :no_cmd;
        const cmd = server.console_input.buffer[0..server.console_input.len];
        server.console_input.len = 0;

        if (std.ascii.eqlIgnoreCase(cmd, "quit")) {
            return false;
        } else {
            std.log.info("unknown command: {s}", .{cmd});
        }
    }

    while (try server.net_man.popEvent()) |event| try server.processNetEvent(event);

    const work_ns = server.tick_timer.read();
    server.tick_count += 1;
    server.total_work_ns += work_ns;

    if (work_ns > ns_per_tick) return true;
    const remaining_ns = ns_per_tick - work_ns;
    std.Thread.sleep(remaining_ns);
    return true;
}

pub fn processNetEvent(server: *Server, event: NetworkManager.Event) !void {
    switch (event) {
        .connect => |peer| {
            std.log.info("connection from {f}", .{peer.address});

            try server.players.insertAtRef(server.alloc, .{
                .slot = peer.ref.slot,
                .gen = peer.ref.gen,
            }, .{
                .peer = peer.ref,
            });

            var timer: std.time.Timer = try .start();
            defer std.log.info("time taken to send world: {}μs", .{timer.read() / 1000});

            var small_buffer: [64]u8 = undefined;
            var small_writer = std.Io.Writer.fixed(&small_buffer);

            try ServerMsgId.init.encode(&small_writer);
            try small_writer.writeInt(u16, @intCast(peer.ref.slot), .little);

            const init_channel = protocol.Channel.control;
            const init_packet = try znet.Packet.init(small_writer.buffered(), init_channel.toInt(), init_channel.getFlags());
            server.total_bytes_sent += init_packet.dataSlice().len;
            try server.net_man.send(peer.ref, init_packet);

            var uniform_buffer: [protocol.max_packet_size]u8 = undefined;
            var uniform_writer = std.Io.Writer.fixed(&uniform_buffer);

            var compressed_buffer: [protocol.max_packet_size]u8 = undefined;
            var compressed_writer = std.Io.Writer.fixed(&compressed_buffer);

            var send_state: ChunkSendState = .{
                .net_man = &server.net_man,
                .peer = peer.ref,
                .uniform_writer = &uniform_writer,
                .compressed_writer = &compressed_writer,
            };
            try send_state.init();

            var uniform_iter = server.world.uniform_chunks.iterator();
            while (uniform_iter.next()) |kv| try send_state.sendUniform(kv.key_ptr.*, kv.value_ptr.*);

            var u2_palette_iter = server.world.u2_palette_chunks.iterator();
            while (u2_palette_iter.next()) |kv| try send_state.send(kv.key_ptr.*, .{ .data = .{ .u2_palette = kv.value_ptr.* } });

            var one_to_one_iter = server.world.one_to_one_chunks.iterator();
            while (one_to_one_iter.next()) |kv| try send_state.send(kv.key_ptr.*, .{ .data = .{ .one_to_one = kv.value_ptr.* } });
            try send_state.flush();
            server.total_bytes_sent += send_state.bytes_sent;

            small_writer.end = 0;
            try ServerMsgId.done.encode(&small_writer);
            const packet = try znet.Packet.init(small_writer.buffered(), 0, .reliable);
            server.total_bytes_sent += packet.dataSlice().len;
            try server.net_man.send(peer.ref, packet);
        },
        .disconnect => |peer| {
            std.log.info("disconnected {f}", .{peer.address});

            try server.players.swapRemove(.{
                .slot = peer.ref.slot,
                .gen = peer.ref.gen,
            });
        },
        .receive => |packet_peer| {
            defer packet_peer.packet.deinit();
        },
    }
}

const Player = struct {
    peer: NetworkManager.PeerRef,
    /// region is 4x4x4 chunks
    regions_to_send: std.ArrayList(Chunk.PackedPos) = .empty,

    const Ref = SparseSet(Player).Ref;
};

const ChunkSendState = struct {
    net_man: *NetworkManager,
    peer: NetworkManager.PeerRef,
    bytes_sent: usize = 0,

    uniform_writer: *std.Io.Writer,
    uniform_chunk_count_pos: usize = 0,
    uniform_chunk_count: u16 = 0,

    compressed_writer: *std.Io.Writer,
    compressed_chunk_count_pos: usize = 0,
    compressed_chunk_count: u16 = 0,

    pub fn init(state: *ChunkSendState) !void {
        try state.initUniform();
        try state.initCompressed();
    }

    pub fn flush(state: *ChunkSendState) !void {
        try state.flushUniform();
        try state.flushCompressed();
    }

    pub fn send(state: *ChunkSendState, pos: Chunk.PackedPos, chunk: Chunk) !void {
        switch (chunk.data) {
            .uniform => |kind| try state.sendUniform(pos, kind),
            .u2_palette,
            .one_to_one,
            => try state.sendCompressed(pos, chunk),
        }
    }

    pub fn totalBytesToSend(state: *const ChunkSendState) usize {
        return state.bytes_sent + state.uniform_writer.end + state.compressed_writer.end;
    }

    pub fn initUniform(state: *ChunkSendState) !void {
        const writer = state.uniform_writer;
        writer.end = 0;

        try ServerMsgId.uniform_chunk_batch.encode(state.uniform_writer);
        state.uniform_chunk_count_pos = writer.end;
        try writer.writeInt(u16, 0, .little);
        state.uniform_chunk_count = 0;
    }

    pub fn flushUniform(state: *ChunkSendState) !void {
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

    pub fn sendUniform(state: *ChunkSendState, pos: Chunk.PackedPos, kind: block.Kind) !void {
        const writer = state.uniform_writer;
        if (writer.end >= protocol.chunk_batch_target_size) {
            try state.flushUniform();
            try state.initUniform();
        }

        try writer.writeStruct(pos, .little);
        try writer.writeInt(u8, @intFromEnum(kind), .little);
        state.uniform_chunk_count += 1;
    }

    pub fn initCompressed(state: *ChunkSendState) !void {
        const writer = state.compressed_writer;
        writer.end = 0;

        try ServerMsgId.compressed_chunk_batch.encode(writer);
        state.compressed_chunk_count_pos = writer.end;
        try writer.writeInt(u16, 0, .little);
        state.compressed_chunk_count = 0;
    }

    pub fn flushCompressed(state: *ChunkSendState) !void {
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

    pub fn sendCompressed(state: *ChunkSendState, pos: Chunk.PackedPos, chunk: Chunk) !void {
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

const ConsoleInput = struct {
    mutex: std.Thread.Mutex = .{},
    buffer: [1024]u8 = undefined,
    len: usize = 0,

    fn worker(state: *ConsoleInput) !void {
        var stdin_buffer: [1024]u8 = undefined;
        const stdin_file = std.fs.File.stdin();
        var stdin_reader = stdin_file.reader(&stdin_buffer);
        const stdin = &stdin_reader.interface;

        while (true) {
            const line = stdin.takeDelimiterExclusive('\n') catch |err| switch (err) {
                error.EndOfStream => return,
                else => return err,
            };

            stdin.toss(1);

            state.mutex.lock();
            defer state.mutex.unlock();

            const n = @min(line.len, state.buffer.len);
            @memcpy(state.buffer[0..n], line[0..n]);
            state.len = n;
        }
    }
};
