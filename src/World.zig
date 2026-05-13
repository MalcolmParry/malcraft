const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("block.zig");
const Chunk = @import("Chunk.zig");
const World = @This();

uniform_chunks: std.AutoHashMapUnmanaged(Chunk.PackedPos, block.Kind) = .empty,
one_to_one_chunks: std.AutoHashMapUnmanaged(Chunk.PackedPos, *Chunk.OneToOne) = .empty,

pub fn deinit(world: *World, alloc: std.mem.Allocator) void {
    var one_to_one_iter = world.one_to_one_chunks.iterator();
    while (one_to_one_iter.next()) |kv| {
        alloc.destroy(kv.value_ptr.*);
    }

    world.uniform_chunks.deinit(alloc);
    world.one_to_one_chunks.deinit(alloc);
}

pub fn chunkPosFromBlockPos(pos: block.Pos) Chunk.Pos {
    return @divFloor(pos, Chunk.size);
}

pub fn chunkRelFromBlockPos(pos: block.Pos) Chunk.RelPos {
    return @intCast(@mod(pos, Chunk.size));
}

pub fn getChunk(world: *const World, pos: Chunk.PackedPos) ?Chunk {
    if (world.uniform_chunks.get(pos)) |kind| return .{ .data = .{ .uniform = kind } };
    if (world.one_to_one_chunks.get(pos)) |one_to_one| return .{ .data = .{ .one_to_one = one_to_one } };
    return null;
}

/// not safe if chunk in that position already exists
pub fn placeChunk(world: *World, alloc: std.mem.Allocator, pos: Chunk.PackedPos, chunk: Chunk) !void {
    switch (chunk.data) {
        .uniform => |kind| try world.uniform_chunks.put(alloc, pos, kind),
        .one_to_one => |one_to_one| try world.one_to_one_chunks.put(alloc, pos, one_to_one),
    }
}

pub fn removeChunk(world: *World, alloc: std.mem.Allocator, pos: Chunk.PackedPos) void {
    _ = world.uniform_chunks.remove(pos);

    const one_to_one_kv = world.one_to_one_chunks.fetchRemove(pos);
    if (one_to_one_kv) |kv| {
        alloc.destroy(kv.value);
    }
}

/// caller takes ownership
pub fn fetchRemoveChunk(world: *World, pos: Chunk.PackedPos) ?Chunk {
    if (world.uniform_chunks.fetchRemove(pos)) |kv| return .{ .data = .{ .uniform = kv.value } };
    if (world.one_to_one_chunks.fetchRemove(pos)) |kv| return .{ .data = .{ .one_to_one = kv.value } };
    return null;
}

pub fn getBlock(world: *const World, pos: block.Pos) ?block.Kind {
    const chunk_pos = chunkPosFromBlockPos(pos);
    const rel_pos = chunkRelFromBlockPos(pos);

    return if (world.getChunk(.pack(chunk_pos))) |chunk|
        chunk.getBlock(rel_pos)
    else
        null;
}

pub fn setBlock(world: *World, alloc: std.mem.Allocator, pos: block.Pos, new: block.Kind) !void {
    const chunk_pos = chunkPosFromBlockPos(pos);
    const packed_pos: Chunk.PackedPos = .pack(chunk_pos);
    var chunk = world.fetchRemoveChunk(packed_pos) orelse return error.ChunkNotPresent;
    const rel_pos = chunkRelFromBlockPos(pos);
    try chunk.setBlock(alloc, rel_pos, new);
    try world.placeChunk(alloc, packed_pos, chunk);
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
