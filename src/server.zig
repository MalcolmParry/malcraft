const std = @import("std");
const builtin = @import("builtin");
const znet = @import("znet");
const Chunk = @import("common/Chunk.zig");
const World = @import("common/World.zig");
const WorldGenerator = @import("server/WorldGenerator.zig");
const GPA = @import("utils/GPA.zig");
const protocol = @import("common/protocol.zig");
const ServerMsgId = protocol.ServerMsgId;
const NetworkManager = @import("common/NetworkManager.zig");
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
next_player_id: u16 = 0,

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

    while (try server.net_man.popEvent()) |event| switch (event) {
        .connect => |peer| {
            std.log.info("connection from {f}", .{peer.address});

            var timer: std.time.Timer = try .start();
            defer std.log.info("time taken to send world: {}μs", .{timer.read() / 1000});

            var buffer: [protocol.max_packet_size]u8 = undefined;
            var writer = std.Io.Writer.fixed(&buffer);

            try ServerMsgId.init.encode(&writer);
            try writer.writeInt(u16, server.next_player_id, .little);
            server.next_player_id += 1;

            const init_channel = protocol.Channel.control;
            const init_packet = try znet.Packet.init(writer.buffered(), init_channel.toInt(), init_channel.getFlags());
            server.total_bytes_sent += init_packet.dataSlice().len;
            try server.net_man.send(peer.ref, init_packet);

            const target_uniform_entries = (protocol.chunk_batch_target_size - 4) / 9;
            var uniform_iter = server.world.uniform_chunks.iterator();
            var remaining_entries = server.world.uniform_chunks.count();

            while (remaining_entries > 0) {
                _ = writer.consumeAll();
                try ServerMsgId.uniform_chunk_batch.encode(&writer);
                const entry_count = @min(remaining_entries, target_uniform_entries);
                try writer.writeInt(u16, @intCast(entry_count), .little);

                for (0..entry_count) |_| {
                    const entry = uniform_iter.next() orelse unreachable;
                    const pos = entry.key_ptr.*;

                    try writer.writeStruct(pos, .little);
                    try writer.writeInt(u8, @intFromEnum(entry.value_ptr.*), .little);
                }

                remaining_entries -= entry_count;
                const channel = protocol.Channel.chunk_transfer;
                const packet = try znet.Packet.init(writer.buffered(), channel.toInt(), channel.getFlags());
                server.total_bytes_sent += packet.dataSlice().len;
                try server.net_man.send(peer.ref, packet);
            }

            {
                _ = writer.consumeAll();
                try ServerMsgId.compressed_chunk_batch.encode(&writer);
                const chunk_count_pos = writer.end;
                try writer.writeInt(u16, 0, .little);
                var chunk_count: u16 = 0;

                var iter = server.world.one_to_one_chunks.iterator();
                while (true) {
                    const maybe_entry = iter.next();

                    if ((maybe_entry == null and chunk_count > 0) or writer.end >= protocol.chunk_batch_target_size) {
                        const end = writer.end;
                        writer.end = chunk_count_pos;
                        try writer.writeInt(u16, chunk_count, .little);
                        writer.end = end;

                        const channel = protocol.Channel.chunk_transfer;
                        const packet = try znet.Packet.init(writer.buffered(), channel.toInt(), channel.getFlags());
                        try server.net_man.send(peer.ref, packet);

                        server.total_bytes_sent += packet.dataSlice().len;
                        chunk_count = 0;

                        _ = writer.consumeAll();
                        try ServerMsgId.compressed_chunk_batch.encode(&writer);
                        try writer.writeInt(u16, 0, .little);
                    }

                    const entry = maybe_entry orelse break;
                    const pos = entry.key_ptr.*;

                    try writer.writeStruct(pos, .little);
                    const compressed_size_pos = writer.end;
                    try writer.writeInt(u16, 0, .little);

                    const compress_buffer = writer.unusedCapacitySlice();
                    const compressed_size = zstd.ZSTD_compress(compress_buffer.ptr, compress_buffer.len, entry.value_ptr.*, @sizeOf(Chunk.OneToOne), 1);
                    if (zstd.ZSTD_isError(compressed_size) != 0) return error.ZstdCompressFailed;

                    const end = writer.end + compressed_size;
                    writer.end = compressed_size_pos;
                    try writer.writeInt(u16, @intCast(compressed_size), .little);
                    writer.end = end;
                    chunk_count += 1;
                }
            }

            _ = writer.consumeAll();
            try ServerMsgId.done.encode(&writer);
            const packet = try znet.Packet.init(writer.buffered(), 0, .reliable);
            try server.net_man.send(peer.ref, packet);
        },
        .disconnect => |peer| {
            std.log.info("disconnected {f}", .{peer.address});
        },
        .receive => |packet_peer| {
            defer packet_peer.packet.deinit();
        },
    };

    const work_ns = server.tick_timer.read();
    server.tick_count += 1;
    server.total_work_ns += work_ns;

    if (work_ns > ns_per_tick) return true;
    const remaining_ns = ns_per_tick - work_ns;
    std.Thread.sleep(remaining_ns);
    return true;
}

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

pub fn main() !void {
    var alloc_obj: GPA = .init();
    defer alloc_obj.deinit();
    const alloc = alloc_obj.allocator();

    var server: Server = undefined;
    try server.init(alloc);
    defer server.deinit();

    while (try server.tick()) {}
}
