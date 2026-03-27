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

pub const RayCastResult = union(enum) {
    no_hit,
    inside,
    hit: struct {
        pos: block.Pos,
        face: block.Face,
    },
};

pub fn rayCast(world: *const World, origin: math.Vec3, dir: math.Vec3) RayCastResult {
    if (@reduce(.And, dir == @as(math.Vec3, @splat(0)))) return .no_hit;

    const pos_f = @floor(origin);
    var pos: block.Pos = @intFromFloat(pos_f);

    const step_f = std.math.sign(dir);
    const step: block.Pos = @intFromFloat(step_f);

    const inv_dir = @as(math.Vec3, @splat(1.0)) / dir;
    const delta = @abs(inv_dir);

    const half: math.Vec3 = @splat(0.5);
    var side_dist = ((step_f * (pos_f - origin)) + (step_f * half) + half) * delta;

    if (world.getBlock(pos)) |b| {
        if (b.isOpaque()) return .inside;
    }

    const max_iterations = 64;
    for (0..max_iterations) |_| {
        const axis: u8 = if (side_dist[0] < side_dist[1])
            if (side_dist[0] < side_dist[2])
                0
            else
                2
        else if (side_dist[1] < side_dist[2])
            1
        else
            2;

        side_dist[axis] += delta[axis];
        pos[axis] += step[axis];

        if (world.getBlock(pos)) |b| {
            if (b.isOpaque()) {
                const pos_face = block.Face.posDirFromAxis(axis);
                const face = if (step[axis] > 0) pos_face.opposite() else pos_face;

                return .{ .hit = .{
                    .pos = pos,
                    .face = face,
                } };
            }
        }
    }

    return .no_hit;
}
