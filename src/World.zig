const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("block.zig");
const Chunk = @import("Chunk.zig");
const World = @This();

chunks: Chunk.Map = .empty,

pub fn deinit(world: *World, alloc: std.mem.Allocator) void {
    var iter = world.chunks.iterator();
    while (iter.next()) |kv| {
        kv.value_ptr.deinit(alloc);
    }

    world.chunks.deinit(alloc);
}

pub fn chunkPosFromBlockPos(pos: block.Pos) Chunk.Pos {
    return @divFloor(pos, Chunk.size);
}

pub fn chunkRelFromBlockPos(pos: block.Pos) Chunk.RelPos {
    return @intCast(@abs(@mod(pos, Chunk.size)));
}

pub fn getBlock(world: *const World, pos: block.Pos) ?block.Kind {
    const chunk_pos = chunkPosFromBlockPos(pos);
    const rel_pos = chunkRelFromBlockPos(pos);

    return if (world.chunks.get(chunk_pos)) |chunk|
        chunk.getBlock(@intCast(rel_pos))
    else
        null;
}

pub fn setBlock(world: *World, alloc: std.mem.Allocator, pos: block.Pos, new: block.Kind) !void {
    const chunk_pos = chunkPosFromBlockPos(pos);
    const chunk = world.chunks.getPtr(chunk_pos) orelse return error.ChunkNotPresent;
    const rel_pos = chunkRelFromBlockPos(pos);
    try chunk.setBlock(alloc, @intCast(rel_pos), new);
}

pub fn rayCast(world: *const World, origin: math.Vec3, dir: math.Vec3) ?block.Pos {
    const origin_floor = @floor(origin);
    const origin_floor_i: block.Pos = @intFromFloat(origin_floor);

    const dir_sign = std.math.sign(dir);
    const dir_sign_i: block.Pos = @intFromFloat(dir_sign);
    const dir_inv = @as(math.Vec3, @splat(1)) / dir;

    const dist = @abs(dir_inv);
    const half_vec: math.Vec3 = @splat(0.5);
    var side_dist = (dir_sign * (origin_floor - origin) + (dir_sign * half_vec) + half_vec) * dist;

    var block_pos = origin_floor_i;
    const max_iterations = 64;
    for (0..max_iterations) |_| {
        if (world.getBlock(block_pos)) |x| {
            if (x.isOpaque()) return block_pos;
        }

        const side_dist_yzx: math.Vec3 = .{
            side_dist[1],
            side_dist[2],
            side_dist[0],
        };

        const side_dist_zxy: math.Vec3 = .{
            side_dist[2],
            side_dist[0],
            side_dist[1],
        };

        const mask = side_dist <= @min(side_dist_yzx, side_dist_zxy);
        const int_mask: block.Pos = @intFromBool(mask);
        const float_mask: math.Vec3 = @floatFromInt(int_mask);
        side_dist += float_mask * dist;
        block_pos += int_mask * dir_sign_i;
    }

    return null;
}
