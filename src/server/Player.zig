const std = @import("std");
const NetworkManager = @import("../common/NetworkManager.zig");
const ChunkStreamer = @import("ChunkStreamer.zig");
const GenerationalSparseSet = @import("../utils/generational_sparse_set.zig").GenerationalSparseSet;
const Player = @This();

pub const Set = GenerationalSparseSet(Player);
pub const Ref = Set.Ref;
pub const State = enum {
    pre_init,
    normal,
};

state: State,
peer: NetworkManager.PeerRef,
chunk_streamer: ChunkStreamer,

pub fn deinit(player: *Player, alloc: std.mem.Allocator) void {
    if (player.state == .normal) {
        player.chunk_streamer.deinit(alloc);
    }
}
