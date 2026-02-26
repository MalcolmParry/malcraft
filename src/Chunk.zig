const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("block.zig");

const Chunk = @This();
pub const Map = std.AutoHashMapUnmanaged(Pos, Chunk);
pub const Pos = @Vector(3, i32);
pub const RelPos = @Vector(3, u5);
pub const len = 32;
pub const size: Pos = @splat(len);
pub const block_count = len * len * len;

pub const OneToOne = struct {
    blocks: [len][len][len]block.Kind,

    pub inline fn setBlock(one_to_one: *OneToOne, pos: RelPos, new: block.Kind) void {
        const x, const y, const z = pos;
        one_to_one.blocks[z][y][x] = new;
    }
};

data: union(enum) {
    single: block.Kind,
    /// use oneToOneIndex
    one_to_one: *OneToOne,
},

pub fn deinit(chunk: *Chunk, alloc: std.mem.Allocator) void {
    switch (chunk.data) {
        .single => {},
        .one_to_one => |data| alloc.destroy(data),
    }
}

pub inline fn getBlock(chunk: *const Chunk, pos: RelPos) block.Kind {
    return switch (chunk.data) {
        .single => |single| single,
        .one_to_one => |one_to_one| one_to_one.blocks[pos[2]][pos[1]][pos[0]],
    };
}

pub fn setBlock(chunk: *Chunk, alloc: std.mem.Allocator, pos: RelPos, new: block.Kind) !void {
    switch (chunk.data) {
        .single => |old| {
            if (old == new) return;

            const one_to_one = try alloc.create(OneToOne);
            const flat: *[block_count]block.Kind = @ptrCast(&one_to_one.blocks);
            @memset(flat, old);

            const x, const y, const z = pos;
            one_to_one.blocks[z][y][x] = new;

            chunk.* = .{ .data = .{
                .one_to_one = one_to_one,
            } };
        },
        .one_to_one => |one_to_one| one_to_one.setBlock(pos, new),
    }
}

/// can give false negatives
pub inline fn allAirFast(chunk: *const Chunk) bool {
    return switch (chunk.data) {
        .single => |single| single == .air,
        .one_to_one => false,
    };
}

/// can give false negatives
pub inline fn allOpaqueFast(chunk: *const Chunk) bool {
    return switch (chunk.data) {
        .single => |single| single.isOpaque(),
        else => false,
    };
}

pub const Iterator = struct {
    pos: @Vector(3, u8) = @splat(0),

    pub inline fn next(iter: *Iterator) ?RelPos {
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
