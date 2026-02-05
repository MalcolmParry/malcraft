const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;

const Chunk = @This();
pub const BlockPos = @Vector(3, i32);
pub const ChunkPos = BlockPos;
pub const T = u5;
pub const Pos = @Vector(3, u5);
pub const len = 32;
pub const size: ChunkPos = @splat(len);

/// indexed with blocks[z][y][x]
/// dont use, use getters and setters instead
blocks: [len][len][len]BlockId,

pub const BlockId = enum(u2) {
    air,
    grass,
    stone,

    pub fn isOpaque(this: BlockId) bool {
        return switch (this) {
            .air => false,
            else => true,
        };
    }

    pub fn color(this: BlockId) math.Vec3 {
        return switch (this) {
            .air => unreachable,
            .grass => .{ 0, 1, 0 },
            .stone => .{ 0.5, 0.5, 0.5 },
        };
    }
};

pub inline fn getBlock(this: *const Chunk, pos: Pos) BlockId {
    return this.blocks[pos[2]][pos[1]][pos[0]];
}

pub inline fn setBlock(this: *Chunk, pos: Pos, val: BlockId) void {
    this.blocks[pos[2]][pos[1]][pos[0]] = val;
}

pub const Iterator = struct {
    pos: @Vector(3, u8) = @splat(0),

    pub inline fn next(this: *Iterator) ?Pos {
        const result = this.pos;
        if (result[2] == len) return null;

        this.pos[0] += 1;
        if (this.pos[0] == len) {
            this.pos[0] = 0;
            this.pos[1] += 1;
            if (this.pos[1] == len) {
                this.pos[1] = 0;
                this.pos[2] += 1;
            }
        }

        return @intCast(result);
    }
};
