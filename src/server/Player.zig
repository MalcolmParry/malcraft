const std = @import("std");
const NetworkManager = @import("../common/NetworkManager.zig");
const ChunkStreamer = @import("ChunkStreamer.zig");
const GenerationalSparseSet = @import("../utils/generational_sparse_set.zig").GenerationalSparseSet;
const Player = @This();

pub const Set = GenerationalSparseSet(Player);
pub const Ref = Set.Ref;

peer: NetworkManager.PeerRef,
chunk_streamer: ChunkStreamer = .{},

pub fn deinit(player: *Player, alloc: std.mem.Allocator) void {
    player.chunk_streamer.deinit(alloc);
}
