const std = @import("std");
const Chunk = @import("Chunk.zig");

pub const Pos = Chunk.Pos;
pub const PackedPos = Chunk.PackedPos;
pub const len = 4;
pub const size: Pos = .{ 4, 4, 4 };
pub const chunk_count = 4 * 4 * 4;
