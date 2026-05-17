const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const znet = @import("znet");
const protocol = @import("protocol.zig");
const Deque = @import("utils/deque.zig").Deque;
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

const PeerSlot = struct {
    peer: znet.Peer,
    gen: u16,
    used: bool,
};

fn worker(man: *NetworkManager, host_config: znet.HostConfig) !void {
    errdefer man.failed.store(true, .monotonic);

    const host = try znet.Host.init(host_config);
    defer shutdown(host);

    var peer_slots: std.ArrayList(PeerSlot) = .empty;
    defer peer_slots.deinit(man.alloc);

    while (man.running.load(.monotonic)) {
        {
            man.mutex.lock();
            defer man.mutex.unlock();

            while (man.outgoing.popFront()) |cmd| switch (cmd) {
                .connect => |data| {
                    const peer = try host.connect(data.config);
                    data.peer.* = try allocPeer(man.alloc, &peer_slots, peer);
                    data.semaphore.post();
                },
                .disconnect => |ref| if (ref.gen == peer_slots.items[ref.slot].gen) {
                    peer_slots.items[ref.slot].peer.disconnect(0);
                },
                .send => |data| if (data.peer.gen == peer_slots.items[data.peer.slot].gen) {
                    try peer_slots.items[data.peer.slot].peer.send(data.packet);
                },
            };
        }

        const event = try host.service(1) orelse continue;
        man.mutex.lock();
        defer man.mutex.unlock();

        switch (event) {
            .connect => |data| {
                const x: PeerData = if (data.peer.ptr.data == null)
                    try allocPeer(man.alloc, &peer_slots, data.peer)
                else
                    .{
                        .ref = refFromPeer(peer_slots.items, data.peer),
                        .address = data.peer.address(),
                    };

                try man.incoming.pushBack(man.alloc, .{
                    .connect = x,
                });
            },
            .disconnect => |data| {
                const ref = refFromPeer(peer_slots.items, data.peer);
                try man.incoming.pushBack(man.alloc, .{
                    .disconnect = .{
                        .ref = ref,
                        .address = data.peer.address(),
                    },
                });

                peer_slots.items[ref.slot].used = false;
                data.peer.ptr.data = null;
            },
            .receive => |data| {
                errdefer data.packet.deinit();

                try man.incoming.pushBack(man.alloc, .{
                    .receive = .{
                        .peer = refFromPeer(peer_slots.items, data.peer),
                        .packet = data.packet,
                    },
                });
            },
        }
    }
}

fn allocPeer(alloc: std.mem.Allocator, slots: *std.ArrayList(PeerSlot), peer: znet.Peer) !PeerData {
    var maybe_id: ?u16 = null;
    for (slots.items, 0..) |slot, i| {
        if (!slot.used) {
            maybe_id = @intCast(i);
            break;
        }
    }

    if (maybe_id == null) {
        maybe_id = @intCast(slots.items.len);
        try slots.append(alloc, .{
            .peer = undefined,
            .used = false,
            .gen = 0,
        });
    }

    const id = maybe_id.?;
    peer.ptr.data = @ptrFromInt(id + 1);
    slots.items[id] = .{
        .peer = peer,
        .gen = slots.items[id].gen +% 1,
        .used = true,
    };

    return .{
        .ref = .{
            .slot = id,
            .gen = slots.items[id].gen,
        },
        .address = peer.address(),
    };
}

fn refFromPeer(slots: []const PeerSlot, peer: znet.Peer) PeerRef {
    const id: u16 = @intCast(@intFromPtr(peer.ptr.data) - 1);

    return .{
        .slot = id,
        .gen = slots[id].gen,
    };
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

pub const PeerRef = struct {
    slot: u16,
    gen: u16,
};

pub const PeerData = struct {
    ref: PeerRef,
    address: znet.Address,
};
