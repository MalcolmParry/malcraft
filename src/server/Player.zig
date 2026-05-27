const std = @import("std");
const NetworkManager = @import("../common/NetworkManager.zig");
const chunk_streaming = @import("chunk_streaming.zig");
const GenerationalSparseSet = @import("../utils/generational_sparse_set.zig").GenerationalSparseSet;
const Player = @This();

pub const Set = GenerationalSparseSet(Player);
pub const Ref = Set.Ref;

peer: NetworkManager.PeerRef,
chunk_cursor: chunk_streaming.Cursor = .{},

pub fn deinit(player: *Player, alloc: std.mem.Allocator) void {
    player.chunk_cursor.deinit(alloc);
}
