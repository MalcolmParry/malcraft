const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("block.zig");
const Chunk = @import("Chunk.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const Deque = @import("deque.zig").Deque;
const World = @import("World.zig");

const WorldGenerator = @This();
const i32x2 = @Vector(2, i32);
height_map: std.AutoHashMapUnmanaged(i32x2, *HeightMap),
queue: Deque(Chunk.Pos),
alloc: std.mem.Allocator,

const HeightMap = struct {
    const Map = [Chunk.len][Chunk.len]i32;

    lowest: i32,
    highest: i32,
    map: Map,
};

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

const target_gen_time_ns = 8_000_000;
pub fn genMany(
    gen: *WorldGenerator,
    world: *World,
    mesher: *ChunkMesher,
    alloc: std.mem.Allocator,
) !void {
    var timer: std.time.Timer = try .start();

    while (true) {
        if (timer.read() >= target_gen_time_ns) break;

        const pos = gen.queue.popFront() orelse break;
        const chunk = try gen.generate(pos);
        try world.chunks.put(alloc, pos, chunk);

        try mesher.addRequest(pos);
        try mesher.addRequest(pos + @as(Chunk.Pos, .{ 1, 0, 0 }));
        try mesher.addRequest(pos + @as(Chunk.Pos, .{ -1, 0, 0 }));
        try mesher.addRequest(pos + @as(Chunk.Pos, .{ 0, 1, 0 }));
        try mesher.addRequest(pos + @as(Chunk.Pos, .{ 0, -1, 0 }));
        try mesher.addRequest(pos + @as(Chunk.Pos, .{ 0, 0, 1 }));
        try mesher.addRequest(pos + @as(Chunk.Pos, .{ 0, 0, -1 }));
    }
}

pub fn generate(gen: *WorldGenerator, chunk_pos: Chunk.Pos) !Chunk {
    const map = try gen.getOrCreateHeightMap(.{ chunk_pos[0], chunk_pos[1] });
    if (isAllOneBlock(map, chunk_pos[2])) |only_block|
        return .{ .data = .{ .single = only_block } };

    const pos = chunk_pos * Chunk.size;
    const one_to_one = try gen.alloc.create(Chunk.OneToOne);

    var iter: Chunk.Iterator = .{};
    while (iter.next()) |chunk_rel| {
        const block_pos = pos + @as(block.Pos, @intCast(chunk_rel));
        const grass_height = map.map[chunk_rel[1]][chunk_rel[0]];

        const kind: block.Kind = switch (std.math.order(block_pos[2], grass_height)) {
            .gt => .air,
            .eq => .grass,
            .lt => .stone,
        };

        one_to_one.setBlock(chunk_rel, kind);
    }

    return .{ .data = .{ .one_to_one = one_to_one } };
}

fn isAllOneBlock(map: *HeightMap, cs_z: i32) ?block.Kind {
    const bs_z = cs_z * Chunk.len;

    if (bs_z > map.highest) return .air;
    if (bs_z + Chunk.len - 1 < map.lowest) return .stone;
    return null;
}

fn getOrCreateHeightMap(gen: *WorldGenerator, pos: i32x2) !*HeightMap {
    const map = try gen.height_map.getOrPut(gen.alloc, pos);

    if (!map.found_existing) {
        map.value_ptr.* = try gen.alloc.create(HeightMap);
        const values = &map.value_ptr.*.*.map;

        var lowest: i32 = std.math.maxInt(i32);
        var highest: i32 = std.math.minInt(i32);
        for (0..Chunk.len) |yr| {
            for (0..Chunk.len) |xr| {
                const bs_chunk_pos = pos * @as(i32x2, @splat(Chunk.len));
                const block_pos = bs_chunk_pos + @as(i32x2, .{ @intCast(xr), @intCast(yr) });
                const f_block_pos: math.Vec2 = @floatFromInt(block_pos);
                const x, const y = f_block_pos;
                const nx = x / 50;
                const ny = y / 50;

                const m = @sin(x / 50 + (y / 70) * @sin(x / 100));

                const height_diff: f32 = 8 * (@sin((nx + 3 * ny) + 3 * @sin((3 * nx - ny) + @sin(nx + ny / 2))) + @sin(9 * nx) / 3) + 20 * math.powComptime(m, 5);

                const grass_height: i32 = @as(i32, @intFromFloat(height_diff));

                lowest = @min(lowest, grass_height);
                highest = @max(highest, grass_height);
                values.*[yr][xr] = grass_height;
            }
        }

        map.value_ptr.*.*.lowest = lowest;
        map.value_ptr.*.*.highest = highest;
    }

    return map.value_ptr.*;
}
