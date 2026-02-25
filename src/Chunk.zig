const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;

const Chunk = @This();
pub const Map = std.AutoHashMapUnmanaged(ChunkPos, Chunk);
pub const BlockPos = @Vector(3, i32);
pub const ChunkPos = BlockPos;
pub const T = u5;
pub const Pos = @Vector(3, u5);
pub const len = 32;
pub const size: ChunkPos = @splat(len);
pub const block_count = len * len * len;

pub const OneToOne = struct {
    blocks: [len][len][len]BlockId,

    pub inline fn setBlock(one_to_one: *OneToOne, pos: Pos, val: BlockId) void {
        const x, const y, const z = pos;
        one_to_one.blocks[z][y][x] = val;
    }
};

data: union(enum) {
    single: BlockId,
    /// use oneToOneIndex
    one_to_one: *OneToOne,
},

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

pub fn deinit(chunk: *Chunk, alloc: std.mem.Allocator) void {
    switch (chunk.data) {
        .single => {},
        .one_to_one => |data| alloc.destroy(data),
    }
}

pub inline fn getBlock(chunk: *const Chunk, pos: Pos) BlockId {
    return switch (chunk.data) {
        .single => |single| single,
        .one_to_one => |one_to_one| one_to_one.blocks[pos[2]][pos[1]][pos[0]],
    };
}

pub fn setBlock(chunk: *Chunk, alloc: std.mem.Allocator, pos: Pos, val: BlockId) !void {
    switch (chunk.data) {
        .single => |old| {
            if (old == val) return;

            const new = try alloc.create(OneToOne);
            const flat: *[block_count]BlockId = @ptrCast(&new.blocks);
            @memset(flat, old);

            const x, const y, const z = pos;
            new.blocks[z][y][x] = val;

            chunk.* = .{ .data = .{
                .one_to_one = new,
            } };
        },
        .one_to_one => |one_to_one| one_to_one.setBlock(pos, val),
    }
}

/// can give false negatives
pub inline fn allAirFast(chunk: *const Chunk) bool {
    return switch (chunk.data) {
        .single => |block| block == .air,
        .one_to_one => false,
    };
}

/// can give false negatives
pub inline fn allOpaqueFast(chunk: *const Chunk) bool {
    return switch (chunk.data) {
        .single => |block| block.isOpaque(),
        else => false,
    };
}

pub const Iterator = struct {
    pos: @Vector(3, u8) = @splat(0),

    pub inline fn next(iter: *Iterator) ?Pos {
        const result = iter.pos;
        if (result[2] == len) return null;

        iter.pos[0] += 1;
        if (iter.pos[0] == len) {
            iter.pos[0] = 0;
            iter.pos[1] += 1;
            if (iter.pos[1] == len) {
                iter.pos[1] = 0;
                iter.pos[2] += 1;
            }
        }

        return @intCast(result);
    }
};
