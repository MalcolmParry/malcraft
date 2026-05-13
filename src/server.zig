const std = @import("std");
const znet = @import("znet");
const net = @import("net.zig");
const Chunk = @import("Chunk.zig");
const World = @import("World.zig");
const WorldGenerator = @import("WorldGenerator.zig");

const tps = 20;
const ms_per_tick = 50;
const ns_per_tick = ms_per_tick * std.time.ns_per_ms;

pub fn main() !void {
    var alloc_obj = std.heap.DebugAllocator(.{}).init;
    defer _ = alloc_obj.deinit();
    const alloc = alloc_obj.allocator();

    var console_input: ConsoleInput = .{};
    const stdin_thread = try std.Thread.spawn(.{}, ConsoleInput.worker, .{&console_input});
    defer stdin_thread.detach();
    stdin_thread.setName("stdin") catch |err|
        std.log.warn("failed to set thread name: {}", .{err});

    try znet.init();
    defer znet.deinit();

    const host = try znet.Host.init(.{
        .addr = try .init(.{
            .ip = .any,
            .port = .{ .uint = 5000 },
        }),
        .peer_limit = 32,
        .channel_limit = .{ .count = 1 },
        .incoming_bandwidth = .unlimited,
        .outgoing_bandwidth = .unlimited,
    });
    defer net.shutdown(host);

    std.log.info("server started", .{});

    var world: World = .{};
    defer world.deinit(alloc);

    var world_gen: WorldGenerator = try .init(alloc);
    defer world_gen.deinit();
    try world_gen.queueChunks();

    while (world_gen.queue.len != 0)
        try world_gen.genMany(alloc, &world);

    std.log.info("world gen done", .{});

    var next_player_id: u32 = 0;
    var tick_timer = try std.time.Timer.start();
    var tick_count: u64 = 0;
    var total_work_ns: u64 = 0;
    var total_bytes_sent: usize = 0;
    while (true) {
        _ = tick_timer.lap();

        no_cmd: {
            console_input.mutex.lock();
            defer console_input.mutex.unlock();

            if (console_input.len == 0) break :no_cmd;
            const cmd = console_input.buffer[0..console_input.len];
            console_input.len = 0;

            if (std.ascii.eqlIgnoreCase(cmd, "quit")) {
                break;
            } else {
                std.log.info("unknown command: {s}", .{cmd});
            }
        }

        while (try host.service(0)) |anyevent| switch (anyevent) {
            .connect => |data| {
                std.log.info("connection from {f}", .{data.peer.address()});

                const init_message: net.server_message.Init = .{
                    .player_id = next_player_id,
                };
                next_player_id += 1;

                var buffer: [net.max_packet_size]u8 = undefined;
                var writer = std.Io.Writer.fixed(&buffer);
                try init_message.encode(&writer);

                const init_packet = try znet.Packet.init(writer.buffered(), 0, .reliable);

                total_bytes_sent += init_packet.dataSlice().len;
                try data.peer.send(init_packet);

                const max_uniform_entries = (net.max_packet_size - 4) / 9;
                var uniform_iter = world.uniform_chunks.iterator();
                var remaining_entries = world.uniform_chunks.count();

                while (remaining_entries > 0) {
                    _ = writer.consumeAll();
                    try writer.writeInt(u16, @intFromEnum(net.server_message.Kind.uniform_chunk_batch), .little);
                    const entry_count = @min(remaining_entries, max_uniform_entries);
                    try writer.writeInt(u16, @intCast(entry_count), .little);

                    for (0..entry_count) |_| {
                        const entry = uniform_iter.next() orelse unreachable;
                        const pos = entry.key_ptr.*;

                        try writer.writeStruct(pos, .little);
                        try writer.writeInt(u8, @intFromEnum(entry.value_ptr.*), .little);
                    }

                    remaining_entries -= entry_count;
                    const packet = try znet.Packet.init(writer.buffered(), 0, .reliable);
                    total_bytes_sent += packet.dataSlice().len;
                    try data.peer.send(packet);
                }

                var one_to_one_iter = world.one_to_one_chunks.iterator();
                while (one_to_one_iter.next()) |entry| {
                    _ = writer.consumeAll();
                    const pos = entry.key_ptr.*;

                    const message: net.server_message.OneToOneChunkData = .{
                        .pos = pos,
                        .chunk = entry.value_ptr.*,
                    };

                    try message.encode(&writer);
                    const packet = try znet.Packet.init(writer.buffered(), 0, .reliable);

                    total_bytes_sent += packet.dataSlice().len;
                    try data.peer.send(packet);
                }
            },
            .disconnect => |data| {
                std.log.info("disconnected {f}", .{data.peer.address()});
            },
            .receive => |data| {
                defer data.packet.deinit();
            },
        };

        const work_ns = tick_timer.read();
        tick_count += 1;
        total_work_ns += tick_timer.read();

        if (work_ns > ns_per_tick) continue;
        const remaining_ns = ns_per_tick - work_ns;
        std.Thread.sleep(remaining_ns);
    }

    std.log.info("server stopped", .{});
    std.log.info("tick count: {}", .{tick_count});
    std.log.info("mean work time per tick: {}ns", .{std.math.divFloor(u64, total_work_ns, tick_count) catch 0});
    std.log.info("total bytes sent: {Bi:.2}", .{total_bytes_sent});
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
