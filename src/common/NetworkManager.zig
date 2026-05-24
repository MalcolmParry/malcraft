const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const znet = @import("znet");
const protocol = @import("protocol.zig");
const Deque = @import("../utils/deque.zig").Deque;
const GenerationalSparseSet = @import("../utils/generational_sparse_set.zig").GenerationalSparseSet;
const ServerMsgId = protocol.ServerMsgId;
const NetworkManager = @This();
const cl = std.atomic.cache_line;

alloc: std.mem.Allocator,
thread: std.Thread,
failed: std.atomic.Value(bool) = .init(false),
running: std.atomic.Value(bool) = .init(true),
// TODO: replace this with atomic queue
mutex: std.Thread.Mutex = .{},
outgoing: Deque(Command),
incoming: Deque(Event),
bytes_sent: usize = 0,

pub fn init(man: *NetworkManager, alloc: std.mem.Allocator, host_config: znet.HostConfig) !void {
    man.* = .{
        .alloc = alloc,
        .thread = undefined,
        .outgoing = undefined,
        .incoming = undefined,
    };

    man.outgoing = try .initCapacity(alloc, 64);
    errdefer man.outgoing.deinit(alloc);

    man.incoming = try .initCapacity(alloc, 64);
    errdefer man.incoming.deinit(alloc);

    man.thread = try .spawn(.{}, worker, .{ man, host_config });
}

pub fn deinit(man: *NetworkManager) void {
    man.running.store(false, .monotonic);
    man.thread.join();
    man.outgoing.deinit(man.alloc);
    man.incoming.deinit(man.alloc);
}

pub fn send(man: *NetworkManager, peer: PeerRef, packet: znet.Packet) !void {
    man.bytes_sent += packet.dataSlice().len;

    try man.pushCommand(.{ .send = .{
        .peer = peer,
        .packet = packet,
    } });
}

pub fn pushCommand(man: *NetworkManager, cmd: Command) !void {
    if (man.failed.load(.monotonic)) return error.NetworkFailed;

    man.mutex.lock();
    defer man.mutex.unlock();

    try man.outgoing.pushBack(man.alloc, cmd);
}

pub fn popEvent(man: *NetworkManager) !?Event {
    if (man.failed.load(.monotonic)) return error.NetworkFailed;

    man.mutex.lock();
    defer man.mutex.unlock();

    return man.incoming.popFront();
}

fn worker(man: *NetworkManager, host_config: znet.HostConfig) !void {
    errdefer man.failed.store(true, .monotonic);

    const host = try znet.Host.init(host_config);
    defer shutdown(host);

    var peers: PeerSet = .empty;
    defer peers.deinit(man.alloc);

    while (man.running.load(.monotonic)) {
        {
            man.mutex.lock();
            defer man.mutex.unlock();

            while (man.outgoing.popFront()) |cmd| switch (cmd) {
                .connect => |data| {
                    const peer = try host.connect(data.config);
                    const ref = try peers.append(man.alloc, peer);
                    peer.ptr.data = &peers.sparse_to_dense.items[ref.slot];
                    data.peer.* = .{
                        .ref = ref,
                        .address = peer.address(),
                    };
                    data.semaphore.post();
                },
                .disconnect => |ref| blk: {
                    const peer = peers.get(ref) orelse break :blk;
                    peer.disconnect(0);
                },
                .send => |data| {
                    if (peers.get(data.peer)) |peer| {
                        errdefer data.packet.deinit();
                        try peer.send(data.packet);
                    } else {
                        data.packet.deinit();
                    }
                },
            };
        }

        const event = try host.service(1) orelse continue;
        man.mutex.lock();
        defer man.mutex.unlock();

        switch (event) {
            .connect => |data| {
                try man.incoming.pushBack(man.alloc, .{
                    .connect = .{
                        .ref = try refFromPeer(man.alloc, &peers, data.peer),
                        .address = data.peer.address(),
                    },
                });
            },
            .disconnect => |data| {
                const ref = try refFromPeer(man.alloc, &peers, data.peer);
                try man.incoming.pushBack(man.alloc, .{
                    .disconnect = .{
                        .ref = ref,
                        .address = data.peer.address(),
                    },
                });

                data.peer.ptr.data = null;
                peers.swapRemove(ref) catch unreachable;
            },
            .receive => |data| {
                errdefer data.packet.deinit();

                try man.incoming.pushBack(man.alloc, .{
                    .receive = .{
                        .peer = try refFromPeer(man.alloc, &peers, data.peer),
                        .packet = data.packet,
                    },
                });
            },
        }
    }
}

pub fn refFromPeer(alloc: std.mem.Allocator, peers: *PeerSet, peer: znet.Peer) !PeerRef {
    if (peer.ptr.data) |usr_data| {
        const sparse: *const PeerSet.Sparse = @ptrCast(@alignCast(usr_data));

        return .{
            .slot = peers.dense_to_sparse.items[sparse.dense_slot],
            .gen = sparse.gen,
        };
    } else {
        const ref = try peers.append(alloc, peer);
        peer.ptr.data = &peers.sparse_to_dense.items[ref.slot];
        return ref;
    }
}

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

pub const PacketWithPeer = struct {
    packet: znet.Packet,
    peer: PeerRef,
};

pub const Event = union(enum) {
    connect: PeerData,
    disconnect: PeerData,
    receive: PacketWithPeer,
};

pub const Command = union(enum) {
    pub const Connect = struct {
        config: znet.ConnectConfig,
        peer: *PeerData,
        semaphore: *std.Thread.Semaphore,
    };

    connect: Connect,
    disconnect: PeerRef,
    send: PacketWithPeer,
};

pub const PeerSet = GenerationalSparseSet(znet.Peer);
pub const PeerRef = PeerSet.Ref;
pub const PeerData = struct {
    ref: PeerRef,
    address: znet.Address,
};
