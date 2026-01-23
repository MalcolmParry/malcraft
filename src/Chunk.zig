const std = @import("std");
const mw = @import("mw");
const math = mw.math;

const Chunk = @This();
pub const BlockPos = @Vector(3, i32);
pub const ChunkPos = BlockPos;
pub const T = u5;
pub const Pos = @Vector(3, u5);
pub const len = 32;

/// indexed with blocks[z][y][x]
/// dont use, use getters and setters instead
blocks: [len][len][len]BlockId,

pub const BlockId = enum {
    air,
    grass,
    stone,

    pub fn isOpaque(this: BlockId) bool {
        return switch (this) {
            .air => false,
            else => true,
        };
    }
};

pub fn init(this: *Chunk, chunk_pos: ChunkPos) void {
    const pos = chunk_pos * @as(BlockPos, @splat(len));

    const grass_height = 16;
    var iter: Iterator = .{};
    while (iter.next()) |chunk_rel| {
        const block_pos = pos + @as(BlockPos, @intCast(chunk_rel));

        this.setBlock(chunk_rel, switch (std.math.order(block_pos[2], grass_height)) {
            .gt => .air,
            .eq => .grass,
            .lt => .stone,
        });
    }
}

pub inline fn isOpaqueSafe(this: *Chunk, pos: BlockPos) bool {
    if (@reduce(.Or, pos < @as(Pos, @splat(0))))
        return false;

    if (@reduce(.Or, pos > @as(Pos, @splat(len - 1))))
        return false;

    return this.getBlock(@intCast(pos)).isOpaque();
}

pub inline fn getBlock(this: *Chunk, pos: Pos) BlockId {
    return this.blocks[pos[2]][pos[1]][pos[0]];
}

pub inline fn setBlock(this: *Chunk, pos: Pos, val: BlockId) void {
    this.blocks[pos[2]][pos[1]][pos[0]] = val;
}

pub const Iterator = struct {
    pos: @Vector(3, u8) = @splat(0),

    pub fn next(this: *Iterator) ?Pos {
        const result = this.pos;

        this.pos[0] += 1;
        if (this.pos[0] == len) {
            this.pos[0] = 0;
            this.pos[1] += 1;
            if (this.pos[1] == len) {
                this.pos[1] = 0;
                this.pos[2] += 1;
            }
        }

        return if (result[2] == len) null else @intCast(result);
    }
};
