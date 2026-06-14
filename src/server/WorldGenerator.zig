const std = @import("std");
const options = @import("options");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const region = @import("../common/region.zig");
const Deque = @import("../utils/deque.zig").Deque;
const World = @import("../common/World.zig");
const Player = @import("Player.zig");
const chunk_streaming = @import("chunk_streaming.zig");
const znoise = @import("znoise");

const WorldGenerator = @This();
const i32x2 = @Vector(2, i32);

height_map: std.AutoHashMapUnmanaged([2]i32, *HeightMap),
alloc: std.mem.Allocator,
total_time: u64 = 0,

const HeightMap = struct {
    const Map = [Chunk.len][Chunk.len]i16;

    lowest: i16,
    highest: i16,
    map: Map,
};

pub fn init(alloc: std.mem.Allocator) !WorldGenerator {
    return .{
        .height_map = .empty,
        .alloc = alloc,
    };
}

pub fn deinit(gen: *WorldGenerator) void {
    var iter = gen.height_map.iterator();
    while (iter.next()) |kv| gen.alloc.destroy(kv.value_ptr.*);
    gen.height_map.deinit(gen.alloc);
    std.log.info("world gen time: {}μs", .{gen.total_time / std.time.ns_per_us});
}

pub fn genMany(
    gen: *WorldGenerator,
    alloc: std.mem.Allocator,
    world: *World,
    players: []Player,
    target_time_ns: u64,
) !void {
    if (players.len == 0) return;
    var timer: std.time.Timer = try .start();
    var player_index: usize = 0;
    var empty_queue_count: usize = 0;

    while (true) {
        if (timer.read() >= target_time_ns) break;
        if (empty_queue_count == players.len) break;
        if (player_index == 0) empty_queue_count = 0;

        const player = &players[player_index];
        if (player.chunk_cursor.chunks_to_gen.popFront()) |pos| {
            if (!player.chunk_cursor.chunkInRange(pos.vec())) continue;
            if (world.containsChunk(pos)) continue;

            const chunk = try gen.generate(pos.vec());
            try world.placeChunk(alloc, pos, chunk);

            try player.chunk_cursor.chunks_to_send.pushBack(alloc, pos);
        } else if (player.chunk_cursor.regions_to_gen.popFront()) |region_pos| {
            if (!player.chunk_cursor.regionInRange(region_pos.vec())) continue;
            const base_chunk_pos = region_pos.vec() * region.size;

            for (0..region.len) |x| {
                for (0..region.len) |y| {
                    for (0..region.len) |z| {
                        const rel: Chunk.Pos = .{
                            @intCast(x),
                            @intCast(y),
                            @intCast(z),
                        };

                        const chunk_pos = base_chunk_pos + rel;
                        if (!player.chunk_cursor.chunkInRange(chunk_pos)) continue;
                        if (world.containsChunk(.pack(chunk_pos))) continue;
                        const chunk = try gen.generate(chunk_pos);
                        try world.placeChunk(alloc, .pack(chunk_pos), chunk);
                    }
                }
            }

            try player.chunk_cursor.regions_to_send.pushBack(alloc, region_pos);
        } else {
            empty_queue_count += 1;
        }

        player_index = (player_index + 1) % players.len;
    }

    gen.total_time += timer.read();
}

const water_height = 80;
const sand_height = 3;

pub fn generate(gen: *WorldGenerator, chunk_pos: Chunk.Pos) !Chunk {
    const map = try gen.getOrCreateHeightMap(.{ chunk_pos[0], chunk_pos[1] });
    if (isAllOneBlock(map, chunk_pos[2])) |only_block|
        return .{ .data = .{ .uniform = only_block } };

    const pos = chunk_pos * Chunk.size;
    const one_to_one = try gen.alloc.create(Chunk.OneToOne);
    var block_exists: [block.Kind.named_count]bool = @splat(false);

    for (0..Chunk.len) |ux| {
        for (0..Chunk.len) |uy| {
            const gh = map.map[ux][uy];

            for (0..Chunk.len) |uz| {
                const h = pos[2] + @as(i32, @intCast(uz));
                const block_id: block.Kind = switch (std.math.order(h, gh)) {
                    .gt => if (h <= water_height) .water else .air,
                    .eq => if (h > water_height) .grass else .sand,
                    .lt => if (gh <= water_height and h + sand_height > gh) .sand else .stone,
                };

                const upos: @Vector(3, usize) = .{ ux, uy, uz };
                const rel: Chunk.RelPos = @intCast(upos);
                one_to_one.setBlock(rel, block_id);
                block_exists[@intFromEnum(block_id)] = true;
            }
        }
    }

    var unique_block_count: u16 = 0;
    for (block_exists) |exists| {
        unique_block_count += @intFromBool(exists);
    }

    switch (unique_block_count) {
        0 => unreachable,
        1 => {
            @branchHint(.cold);
            std.log.warn("slow path taken for uniform chunk gen at: {}, lowest: {}, highest: {}", .{ pos, map.lowest, map.highest });
            defer gen.alloc.destroy(one_to_one);

            return .{ .data = .{ .uniform = one_to_one.getBlock(@splat(0)) } };
        },
        2...4 => {
            defer gen.alloc.destroy(one_to_one);

            var palette_to_kind: [4]block.Kind = undefined;
            var kind_to_palette: [block.Kind.named_count]u2 = undefined;
            var palette_index: u8 = 0;
            var palette_bitmask: std.StaticBitSet(4) = .initEmpty();
            for (block_exists, 0..) |exists, kind_i| {
                if (exists) {
                    palette_to_kind[palette_index] = @enumFromInt(kind_i);
                    kind_to_palette[kind_i] = @intCast(palette_index);
                    palette_bitmask.set(palette_index);
                    palette_index += 1;
                }
            }

            const palette = try gen.alloc.create(Chunk.U2Palette);
            errdefer gen.alloc.destroy(palette);
            palette.palette_bitmask = palette_bitmask;
            palette.palette = palette_to_kind;

            for (0..Chunk.block_count) |i| {
                const kind = Chunk.OneToOne.getBlockAtIndex(one_to_one.blocks[0..], i);
                const palette_val = kind_to_palette[@intFromEnum(kind)];
                Chunk.U2Palette.setBlockAtIndex(palette.blocks[0..], i, palette_val);
            }

            return .{ .data = .{ .u2_palette = palette } };
        },
        else => {
            return .{ .data = .{ .one_to_one = one_to_one } };
        },
    }
}

fn isAllOneBlock(map: *HeightMap, cs_z: i32) ?block.Kind {
    const bottom = cs_z * Chunk.len;
    const top = bottom + Chunk.len - 1;

    if (bottom > map.highest) {
        if (bottom > water_height) return .air;
        if (top <= water_height) return .water;
    }

    if (map.lowest > water_height and top < map.lowest) return .stone;
    if (top <= map.lowest - sand_height) return .stone;
    return null;
}

fn getOrCreateHeightMap(gen: *WorldGenerator, pos: i32x2) !*HeightMap {
    const map = try gen.height_map.getOrPut(gen.alloc, pos);

    if (!map.found_existing) {
        map.value_ptr.* = try gen.alloc.create(HeightMap);
        const values = &map.value_ptr.*.*.map;

        const seed = 69422;
        var lowest: i16 = std.math.maxInt(i16);
        var highest: i16 = std.math.minInt(i16);
        for (0..Chunk.len) |xr| {
            for (0..Chunk.len) |yr| {
                const bs_chunk_pos = pos * @as(i32x2, @splat(Chunk.len));
                const block_pos = bs_chunk_pos + @as(i32x2, .{ @intCast(xr), @intCast(yr) });
                const f_block_pos: math.Vec2 = @floatFromInt(block_pos);
                const x, const y = f_block_pos;

                const warp = znoise.FnlGenerator{
                    .seed = seed,
                    .frequency = 0.002,
                    .octaves = 3,
                };

                const wx = warp.noise2(x, y) * 100;
                const wy = warp.noise2(x + 1000, y + 1000) * 100;

                const nx = x + wx;
                const ny = y + wy;

                var height: f32 = 0;

                const iterations = 6;
                var ridge: f32 = 0;
                inline for (0..iterations) |i| {
                    const p: f32 = comptime math.powComptime(2, i);

                    const noise_state: znoise.FnlGenerator = .{
                        .seed = seed,
                        .frequency = 0.0005,
                    };

                    const h = noise_state.noise2(nx * p, ny * p);

                    ridge += 1 - (@abs(h) / p);
                }
                height += math.powComptime(ridge / iterations, 4) * 200;

                const height_i: i16 = @intFromFloat(height);

                lowest = @min(lowest, height_i);
                highest = @max(highest, height_i);
                values.*[xr][yr] = height_i;
            }
        }

        map.value_ptr.*.*.lowest = lowest;
        map.value_ptr.*.*.highest = highest;
    }

    return map.value_ptr.*;
}

fn zeroToOne(x: f32) f32 {
    return (x + 1) / 2;
}
