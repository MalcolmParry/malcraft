const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("block.zig");

const Chunk = @This();
pub const len = 32;
pub const size: Pos = @splat(len);
pub const block_count = len * len * len;

pub const Pos = @Vector(3, i32);
pub const RelPos = @Vector(3, u5);
pub const PackedPos = block.PackedPos;

pub const OneToOne = struct {
    blocks: [block_count / 4]u8,

    pub const Col = [len / 4]u8;
    const plane_stride = len * len / 4;
    const col_stride = len / 4;

    comptime {
        std.debug.assert(@bitSizeOf(block.Kind) <= 2);
        std.debug.assert(len % 4 == 0);
    }

    pub inline fn index(pos: RelPos) usize {
        return @as(usize, pos[0]) * len * len + @as(usize, pos[1]) * len + @as(usize, pos[2]);
    }

    pub inline fn getBlock(one_to_one: *const OneToOne, pos: RelPos) block.Kind {
        const i = index(pos);
        const byte = i / 4;
        const bit: u3 = @intCast((i % 4) * 2);
        const kind_i: u2 = @truncate(one_to_one.blocks[byte] >> bit);
        return @enumFromInt(kind_i);
    }

    pub inline fn setBlockAtIndex(block_data: []u8, i: usize, new: block.Kind) void {
        const byte = i / 4;
        const bit: u3 = @intCast((i % 4) * 2);

        const clear_mask = ~(@as(u8, 0b11) << bit);
        const new_i: u2 = @intFromEnum(new);
        const set_mask = @as(u8, new_i) << bit;

        var data = block_data[byte];
        data &= clear_mask;
        data |= set_mask;
        block_data[byte] = data;
    }

    pub inline fn setBlock(one_to_one: *OneToOne, pos: RelPos, new: block.Kind) void {
        setBlockAtIndex(one_to_one.blocks[0..], index(pos), new);
    }

    pub inline fn getCol(one_to_one: *OneToOne, x: u5, y: u5) *Col {
        return one_to_one.blocks[@as(usize, x) * plane_stride + @as(usize, y) * col_stride ..][0..col_stride];
    }

    pub inline fn fillCol(col: *Col, start: usize, h: usize, value: block.Kind) void {
        std.debug.assert(start + h <= len);

        const value_u8: u8 = @intFromEnum(value);
        const fill_byte = value_u8 * 0b0101_0101; // repeat 4x

        const end = start + h;
        const start_aligned = std.mem.alignForward(usize, start, 4);
        const end_aligned = std.mem.alignBackward(usize, end, 4);

        if (start_aligned <= end_aligned) {
            for (start..start_aligned) |i| {
                setBlockAtIndex(col, i, value);
            }

            @memset(col[start_aligned / 4 .. end_aligned / 4], fill_byte);

            for (end_aligned..end) |i| {
                setBlockAtIndex(col, i, value);
            }
        } else {
            for (start..end) |i| {
                setBlockAtIndex(col, i, value);
            }
        }
    }

    pub inline fn fill(one_to_one: *OneToOne, value: block.Kind) void {
        const value_u8: u8 = @intFromEnum(value);
        const fill_byte = value_u8 * 0b0101_0101; // repeat 4x

        @memset(one_to_one.blocks[0..], fill_byte);
    }
};

pub const StorageType = std.meta.Tag(Data);
pub const Data = union(enum) {
    uniform: block.Kind,
    one_to_one: *OneToOne,
};

data: Data,

pub fn deinit(chunk: *Chunk, alloc: std.mem.Allocator) void {
    switch (chunk.data) {
        .uniform => {},
        .one_to_one => |data| alloc.destroy(data),
    }
}

pub inline fn getBlock(chunk: *const Chunk, pos: RelPos) block.Kind {
    return switch (chunk.data) {
        .uniform => |kind| kind,
        .one_to_one => |one_to_one| one_to_one.getBlock(pos),
    };
}

pub fn setBlock(chunk: *Chunk, alloc: std.mem.Allocator, pos: RelPos, new: block.Kind) !void {
    switch (chunk.data) {
        .uniform => |old| {
            if (old == new) return;

            const one_to_one = try alloc.create(OneToOne);
            one_to_one.fill(old);
            one_to_one.setBlock(pos, new);

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
        .uniform => |kind| kind == .air,
        .one_to_one => false,
    };
}

/// can give false negatives
pub inline fn allOpaqueFast(chunk: *const Chunk) bool {
    return switch (chunk.data) {
        .uniform => |kind| kind.isOpaque(),
        else => false,
    };
}
