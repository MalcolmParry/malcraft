const std = @import("std");
const builtin = @import("builtin");
const znet = @import("znet");
const GenerationalSparseSet = @import("../utils/generational_sparse_set.zig").GenerationalSparseSet;
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const World = @import("../common/World.zig");
const WorldGenerator = @import("../server/WorldGenerator.zig");
const protocol = @import("../common/protocol.zig");
const ServerMsgId = protocol.ServerMsgId;
const NetworkManager = @import("../common/NetworkManager.zig");
const chunk_streaming = @import("chunk_streaming.zig");
const Player = @import("Player.zig");
const Server = @This();

const tps = 20;
const ns_per_tick: comptime_int = (1.0 / @as(comptime_float, tps)) * std.time.ns_per_s;

alloc: std.mem.Allocator,
io: std.Io,
stdin_thread: std.Thread,
console_input: ConsoleInput,
net_man: NetworkManager,

world: World = .{},
world_gen: WorldGenerator,

tick_count: u64 = 0,
total_work_ns: u64 = 0,
players: Player.Set = .empty,

pub fn init(server: *Server, alloc: std.mem.Allocator, io: std.Io) !void {
    server.* = .{
        .alloc = alloc,
        .io = io,
        .stdin_thread = undefined,
        .console_input = .{},
        .net_man = undefined,

        .world_gen = undefined,
    };
    errdefer server.world.deinit(alloc);

    server.stdin_thread = try .spawn(.{}, ConsoleInput.worker, .{ io, &server.console_input });
    errdefer server.stdin_thread.detach();
    server.stdin_thread.setName(io, "stdin") catch |err|
        std.log.warn("failed to set thread name: {}", .{err});

    try znet.init();
    errdefer znet.deinit();

    try server.net_man.init(alloc, io, .{
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
}

pub fn deinit(server: *Server) void {
    const alloc = server.alloc;

    std.log.info("server stopped", .{});
    std.log.info("tick count: {}", .{server.tick_count});
    std.log.info("mean work time per tick: {}ns", .{std.math.divFloor(u64, server.total_work_ns, server.tick_count) catch 0});
    std.log.info("total bytes sent: {Bi:.2}", .{server.net_man.bytes_sent});

    server.world_gen.deinit();
    server.world.deinit(alloc);

    for (server.players.dense.items) |*player| player.deinit(alloc);
    server.players.deinit(alloc);
    server.net_man.deinit();
    znet.deinit();
    server.stdin_thread.detach();
}

pub fn tick(server: *Server) !bool {
    const io = server.io;
    const tick_start: std.Io.Timestamp = .now(io, .awake);

    no_cmd: {
        server.console_input.mutex.lockUncancelable(io);
        defer server.console_input.mutex.unlock(io);

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

    for (server.players.dense.items) |*player| {
        try chunk_streaming.sendChunks(server.alloc, io, &server.net_man, &server.world, player.peer, &player.chunk_cursor);
    }

    blk: {
        const headroom_ns = 500_000;
        const prev_work_ns: u64 = @intCast(tick_start.untilNow(io, .awake).toNanoseconds());
        if (prev_work_ns + headroom_ns >= ns_per_tick) break :blk;

        const budget_ns = ns_per_tick - prev_work_ns - headroom_ns;
        try server.world_gen.genMany(server.alloc, io, &server.world, server.players.dense.items, budget_ns);
    }

    const work_ns: u64 = @intCast(tick_start.untilNow(io, .awake).toNanoseconds());
    server.tick_count += 1;
    server.total_work_ns += work_ns;

    if (work_ns > ns_per_tick) return true;
    const remaining_ns = ns_per_tick - work_ns;

    try io.sleep(.fromNanoseconds(remaining_ns), .awake);
    return true;
}

pub fn processNetEvent(server: *Server, event: NetworkManager.Event) !void {
    switch (event) {
        .connect => |peer| {
            std.log.info("connection from {f}", .{peer.address});

            const player_ref: Player.Ref = .{
                .slot = peer.ref.slot,
                .gen = peer.ref.gen,
            };

            try server.players.insertAtRef(server.alloc, player_ref, .{
                .peer = peer.ref,
            });

            var small_buffer: [64]u8 = undefined;
            var small_writer = std.Io.Writer.fixed(&small_buffer);

            try ServerMsgId.init.encode(&small_writer);
            try small_writer.writeInt(u16, @intCast(peer.ref.slot), .little);

            const init_channel = protocol.Channel.control;
            const init_packet = try znet.Packet.init(small_writer.buffered(), init_channel.toInt(), init_channel.getFlags());
            try server.net_man.send(peer.ref, init_packet);

            const player = server.players.getPtr(player_ref).?;
            try player.chunk_cursor.init(server.alloc);
        },
        .disconnect => |peer| {
            std.log.info("disconnected {f}", .{peer.address});

            const ref: Player.Ref = .{
                .slot = peer.ref.slot,
                .gen = peer.ref.gen,
            };

            server.players.getPtr(ref).?.deinit(server.alloc);
            try server.players.swapRemove(ref);
        },
        .receive => |packet_peer| {
            defer packet_peer.packet.deinit();
        },
    }
}

const ConsoleInput = struct {
    mutex: std.Io.Mutex = .init,
    buffer: [1024]u8 = undefined,
    len: usize = 0,

    fn worker(io: std.Io, state: *ConsoleInput) !void {
        var stdin_buffer: [1024]u8 = undefined;
        const stdin_file = std.Io.File.stdin();
        var stdin_reader = stdin_file.reader(io, &stdin_buffer);
        const stdin = &stdin_reader.interface;

        while (true) {
            const line = stdin.takeDelimiterExclusive('\n') catch |err| switch (err) {
                error.EndOfStream => return,
                else => return err,
            };

            stdin.toss(1);

            state.mutex.lockUncancelable(io);
            defer state.mutex.unlock(io);

            const n = @min(line.len, state.buffer.len);
            @memcpy(state.buffer[0..n], line[0..n]);
            state.len = n;
        }
    }
};
