const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("block.zig");
const Chunk = @import("Chunk.zig");
const Region = @import("Region.zig");
const World = @This();

regions: std.AutoHashMapUnmanaged(Region.PackedPos, *Region) = .empty,

pub fn deinit(world: *World, alloc: std.mem.Allocator) void {
    var region_iter = world.regions.iterator();
    while (region_iter.next()) |kv| {
        kv.value_ptr.*.deinit(alloc);
        alloc.destroy(kv.value_ptr.*);
    }

    world.regions.deinit(alloc);
}

pub fn chunkPosFromBlockPos(pos: block.Pos) Chunk.Pos {
    return @divFloor(pos, Chunk.size);
}

pub fn chunkRelFromBlockPos(pos: block.Pos) Chunk.RelPos {
    return @intCast(@mod(pos, Chunk.size));
}

pub fn containsChunk(world: *const World, pos: Chunk.Pos) bool {
    return world.getChunk(pos) != null;
}

pub fn getChunkPtr(world: *const World, pos: Chunk.Pos) ?*?Chunk {
    const region_pos = @divFloor(pos, Region.size);
    const region = world.regions.get(.pack(region_pos)) orelse return null;
    const local_chunk_pos = @mod(pos, Region.size);
    return &region.chunks[Region.index(local_chunk_pos)];
}

pub fn getChunk(world: *const World, pos: Chunk.Pos) ?Chunk {
    return (world.getChunkPtr(pos) orelse return null).*;
}

/// unsafe if chunk already exists
pub fn placeChunk(world: *World, alloc: std.mem.Allocator, pos: Chunk.Pos, chunk: ?Chunk) !void {
    const region_pos = @divFloor(pos, Region.size);
    const kv = try world.regions.getOrPut(alloc, .pack(region_pos));

    if (!kv.found_existing) {
        kv.value_ptr.* = try alloc.create(Region);
        @memset(kv.value_ptr.*.chunks[0..], null);
    }
    const region = kv.value_ptr.*;

    const local_chunk_pos = @mod(pos, Region.size);
    region.chunks[Region.index(local_chunk_pos)] = chunk;
}

pub fn removeChunk(world: *World, alloc: std.mem.Allocator, pos: Chunk.Pos) void {
    const chunk = world.getChunkPtr(pos) orelse return;
    if (chunk.* == null) return;

    chunk.*.?.deinit(alloc);
    chunk.* = null;
}

pub fn getBlock(world: *const World, pos: block.Pos) ?block.Kind {
    const chunk_pos = chunkPosFromBlockPos(pos);
    const rel_pos = chunkRelFromBlockPos(pos);

    return if (world.getChunk(chunk_pos)) |chunk|
        chunk.getBlock(rel_pos)
    else
        null;
}

pub fn setBlock(world: *World, alloc: std.mem.Allocator, pos: block.Pos, new: block.Kind) !void {
    const chunk_pos = chunkPosFromBlockPos(pos);
    const maybe_chunk_ptr = world.getChunkPtr(chunk_pos) orelse return error.ChunkNotPresent;
    if (maybe_chunk_ptr.* == null) return error.ChunkNotPresent;
    const chunk_ptr = &maybe_chunk_ptr.*.?;
    const rel_pos = chunkRelFromBlockPos(pos);
    try chunk_ptr.setBlock(alloc, rel_pos, new);
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
