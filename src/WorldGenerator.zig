const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const Chunk = @import("Chunk.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const Deque = @import("deque.zig").Deque;

const WorldGenerator = @This();
const i32x2 = @Vector(2, i32);
const ChunkHeightMap = [Chunk.len][Chunk.len]i32;
height_map: std.AutoHashMapUnmanaged(i32x2, *ChunkHeightMap),
queue: Deque(Chunk.ChunkPos),
alloc: std.mem.Allocator,

pub fn init(gen: *WorldGenerator, alloc: std.mem.Allocator) !void {
    gen.* = .{
        .height_map = .empty,
        .queue = .empty,
        .alloc = alloc,
    };
}

pub fn deinit(gen: *WorldGenerator) void {
    var iter = gen.height_map.iterator();
    while (iter.next()) |kv| gen.alloc.destroy(kv.value_ptr.*);
    gen.height_map.deinit(gen.alloc);
    gen.queue.deinit(gen.alloc);
}

pub fn genMany(
    gen: *WorldGenerator,
    chunks: *Chunk.Map,
    mesher: *ChunkMesher,
    alloc: std.mem.Allocator,
) !void {
    try mesher.thread_info.queue.ensureUnusedCapacity(gen.alloc, gen.queue.len);

    var iter = gen.queue.iterator();
    while (iter.next()) |pos| {
        const chunk = try gen.generate(pos);
        try chunks.put(alloc, pos, chunk);
        if (chunk.allAir()) continue;
        mesher.thread_info.queue.pushBackAssumeCapacity(pos);
    }
}

pub fn generate(gen: *WorldGenerator, chunk_pos: Chunk.ChunkPos) !Chunk {
    const map = try gen.getOrCreateHeightMap(.{ chunk_pos[0], chunk_pos[1] });
    if (isAllOneBlock(map, chunk_pos[2])) |only_block|
        return .{ .data = .{ .single = only_block } };

    const pos = chunk_pos * Chunk.size;
    var chunk: Chunk = .{ .data = .{
        .one_to_one = try gen.alloc.create(Chunk.OneToOne),
    } };

    var iter: Chunk.Iterator = .{};
    while (iter.next()) |chunk_rel| {
        const block_pos = pos + @as(Chunk.BlockPos, @intCast(chunk_rel));
        const grass_height = map[chunk_rel[1]][chunk_rel[0]];

        if (block_pos[2] == grass_height - 1 and chunk_rel[0] % 6 == 0) {
            chunk.setBlock(chunk_rel, .air);
            continue;
        }

        chunk.setBlock(chunk_rel, switch (std.math.order(block_pos[2], grass_height)) {
            .gt => .air,
            .eq => .grass,
            .lt => .stone,
        });
    }

    return chunk;
}

fn isAllOneBlock(map: *ChunkHeightMap, cs_z: i32) ?Chunk.BlockId {
    const bs_z = cs_z * Chunk.len;
    const below = map.*[0][0] > bs_z;

    for (0..Chunk.len) |y| {
        for (0..Chunk.len) |x| {
            if (below) {
                if (map.*[y][x] <= bs_z + Chunk.len)
                    return null;
            } else {
                if (map.*[y][x] >= bs_z)
                    return null;
            }
        }
    }

    return if (below) .stone else .air;
}

fn getOrCreateHeightMap(gen: *WorldGenerator, pos: i32x2) !*ChunkHeightMap {
    const map = try gen.height_map.getOrPut(gen.alloc, pos);

    if (!map.found_existing) {
        map.value_ptr.* = try gen.alloc.create(ChunkHeightMap);

        for (0..Chunk.len) |yr| {
            for (0..Chunk.len) |xr| {
                const bs_chunk_pos = pos * @as(i32x2, @splat(Chunk.len));
                const block_pos = bs_chunk_pos + @as(i32x2, .{ @intCast(xr), @intCast(yr) });
                const f_block_pos: math.Vec2 = @floatFromInt(block_pos);
                const x, const y = f_block_pos;
                const nx = x / 50;
                const ny = y / 50;

                const m = @sin(x / 50 + (y / 70) * @sin(x / 100));

                const height_diff: f32 = 8 * (@sin((nx + 3 * ny) + 3 * @sin((3 * nx - ny) + @sin(nx + ny / 2))) + @sin(9 * nx) / 3) + 20 * (m * m * m * m * m);

                const grass_height: i32 = @as(i32, @intFromFloat(height_diff)) + 16;

                map.value_ptr.*.*[yr][xr] = grass_height;
            }
        }
    }

    return map.value_ptr.*;
}
