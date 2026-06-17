const std = @import("std");
const Chunk = @import("Chunk.zig");
const Region = @This();

pub const Pos = Chunk.Pos;
pub const PackedPos = Chunk.PackedPos;
pub const len = 4;
pub const size: Pos = .{ 4, 4, 4 };
pub const chunk_count = 4 * 4 * 4;

chunks: [chunk_count]?Chunk,

pub fn deinit(region: *Region, alloc: std.mem.Allocator) void {
    for (&region.chunks) |*maybe_chunk| {
        if (maybe_chunk.*) |*chunk| {
            chunk.deinit(alloc);
        }
    }
}

pub fn index(pos: Pos) usize {
    const upos: @Vector(3, usize) = @intCast(pos);
    return upos[0] * 4 * 4 + upos[1] * 4 + upos[2];
}
