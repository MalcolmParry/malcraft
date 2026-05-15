const std = @import("std");
const options = @import("options");
const mw = @import("mwengine");
const math = mw.math;
const block = @import("block.zig");
const Chunk = @import("Chunk.zig");
const Deque = @import("deque.zig").Deque;
const World = @import("World.zig");
const znoise = @import("znoise");

const WorldGenerator = @This();
const i32x2 = @Vector(2, i32);
height_map: std.AutoHashMapUnmanaged([2]i32, *HeightMap),
queue: Deque(Chunk.PackedPos),
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
        .queue = .empty,
        .alloc = alloc,
    };
}

pub fn deinit(gen: *WorldGenerator) void {
    var iter = gen.height_map.iterator();
    while (iter.next()) |kv| gen.alloc.destroy(kv.value_ptr.*);
    gen.height_map.deinit(gen.alloc);
    gen.queue.deinit(gen.alloc);
    std.log.info("world gen time: {}μs", .{gen.total_time / std.time.ns_per_us});
}

const target_gen_time_ns = 8_000_000;
pub fn genMany(
    gen: *WorldGenerator,
    alloc: std.mem.Allocator,
    world: *World,
) !void {
    var timer: std.time.Timer = try .start();

    while (true) {
        if (timer.read() >= target_gen_time_ns) break;

        const pos = gen.queue.popFront() orelse break;
        const chunk = try gen.generate(pos.vec());
        try world.placeChunk(alloc, pos, chunk);
    }

    gen.total_time += timer.read();
}

const water_height = 80;
pub fn generate(gen: *WorldGenerator, chunk_pos: Chunk.Pos) !Chunk {
    const map = try gen.getOrCreateHeightMap(.{ chunk_pos[0], chunk_pos[1] });
    if (isAllOneBlock(map, chunk_pos[2])) |only_block|
        return .{ .data = .{ .uniform = only_block } };

    const pos = chunk_pos * Chunk.size;
    const one_to_one = try gen.alloc.create(Chunk.OneToOne);
    one_to_one.fill(.air);

    for (0..Chunk.len) |ux| {
        for (0..Chunk.len) |uy| {
            // const col = one_to_one.getCol(@intCast(ux), @intCast(uy));
            const gh = map.map[uy][ux];
            // const hr: i32 = h - pos[2];
            // const wr: i32 = water_height - pos[2];

            for (0..Chunk.len) |uz| {
                const h = pos[2] + @as(i32, @intCast(uz));
                const block_id: block.Kind = switch (std.math.order(h, gh)) {
                    .gt => if (h <= water_height) .water else .air,
                    .eq => switch (std.math.order(h, water_height)) {
                        .lt => .stone,
                        .eq => .grass,
                        .gt => .grass,
                    },
                    .lt => .stone,
                };

                const upos: @Vector(3, usize) = .{ ux, uy, uz };
                const rel: Chunk.RelPos = @intCast(upos);
                one_to_one.setBlock(rel, block_id);
            }

            // if (hr <= 0) {
            //     Chunk.OneToOne.fillCol(col, 0, Chunk.len, .air);
            //
            //     if (hr == 0)
            //         Chunk.OneToOne.setBlockAtIndex(col, 0, .grass);
            // } else if (hr >= Chunk.len) {
            //     Chunk.OneToOne.fillCol(col, 0, Chunk.len, .stone);
            // } else {
            //     const hru: usize = @intCast(hr);
            //
            //     Chunk.OneToOne.setBlockAtIndex(col, hru, .grass);
            //     Chunk.OneToOne.fillCol(col, 0, hru, .stone);
            //     Chunk.OneToOne.fillCol(col, hru + 1, Chunk.len - hru - 1, .air);
            // }
        }
    }

    return .{ .data = .{ .one_to_one = one_to_one } };
}

fn isAllOneBlock(map: *HeightMap, cs_z: i32) ?block.Kind {
    const bs_z = cs_z * Chunk.len;

    if (bs_z > map.highest and bs_z > water_height) return .air;
    if (bs_z + Chunk.len - 1 < map.lowest) return .stone;
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
        for (0..Chunk.len) |yr| {
            for (0..Chunk.len) |xr| {
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
                values.*[yr][xr] = height_i;
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

pub fn queueChunks(gen: *WorldGenerator) !void {
    const alloc = gen.alloc;
    const render_radius: i32 = @intCast(options.render_radius);
    const vertical_render_radius: i32 = @intCast(options.vrender_radius);
    const chunk_count = (render_radius * 2 + 1) * (render_radius * 2 + 1) * (vertical_render_radius * 2 + 1);

    try gen.queue.ensureUnusedCapacity(alloc, chunk_count);

    var z: i20 = 0;
    while (z <= vertical_render_radius) : (z += 1) {
        var y: i22 = -render_radius;
        while (y <= render_radius) : (y += 1) {
            var x: i22 = -render_radius;
            while (x <= render_radius) : (x += 1) {
                const pos: Chunk.PackedPos = .{ .x = x, .y = y, .z = z };
                gen.queue.pushBackAssumeCapacity(pos);
            }
        }
    }

    const queue = gen.queue.buffer[0..gen.queue.len];
    std.mem.sort(Chunk.PackedPos, queue, {}, struct {
        fn lessThanFn(_: void, left: Chunk.PackedPos, right: Chunk.PackedPos) bool {
            const f_left: math.Vec3 = @floatFromInt(left.vec());
            const f_right: math.Vec3 = @floatFromInt(right.vec());
            return math.lengthSqr(f_left) < math.lengthSqr(f_right);
        }
    }.lessThanFn);
}
