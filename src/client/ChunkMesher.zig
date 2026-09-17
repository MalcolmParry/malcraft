const std = @import("std");
const mw = @import("mwengine");
const options = @import("options");
const math = mw.math;
const gpu = mw.gpu;
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const World = @import("../common/World.zig");
const ChunkMeshAllocator = @import("ChunkMeshAllocator.zig");
const TextureManager = @import("TextureManager.zig");

const ChunkMesher = @This();
pub const max_faces = (Chunk.block_count / 2) * 6;

pub const GpuLoaded = struct {
    face_count: u32,
    face_offset: u32,
};

pub const GreedyQuad = packed struct(u64) {
    x: packed struct(u32) {
        face: block.Face,
        x: u5,
        y: u5,
        z: u5,
        /// width  - 1 so range is 1-32
        w: u5,
        /// height - 1 so range is 1-32
        h: u5,
        flip: u1,
        tex_id: TextureManager.Id,
    },
    y: packed struct(u32) {
        ao_corners: AoCorners,
        unused2: u24 = undefined,
    },
};

comptime {
    std.debug.assert(@bitSizeOf(block.Face) == 3);
    std.debug.assert(@bitSizeOf(TextureManager.Id) == 3);
    std.debug.assert(@bitSizeOf(AoCorners) == 8);
}

const AoCorners = packed struct(u8) {
    bl: u2,
    br: u2,
    tl: u2,
    tr: u2,
};

alloc: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
mesh_alloc: *ChunkMeshAllocator,
thread_info: MeshThreadInfo,
threads: []std.Thread,
queue: std.AutoArrayHashMapUnmanaged(Chunk.PackedPos, void),

meshing_time_ns: u64,

pub const InitInfo = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    mesh_alloc: *ChunkMeshAllocator,
    world: *const World,
};

pub fn init(this: *ChunkMesher, info: InitInfo) !void {
    this.alloc = info.alloc;
    this.arena = .init(info.alloc);
    this.mesh_alloc = info.mesh_alloc;
    this.meshing_time_ns = 0;
    this.queue = .empty;

    const thread_count: u8 = @min(255, @max(1, std.Thread.getCpuCount() catch 1));
    this.threads = try info.alloc.alloc(std.Thread, thread_count);
    errdefer info.alloc.free(this.threads);

    this.thread_info = .{
        .thread_count = thread_count,
        .io = info.io,
        .world = info.world,
        .jobs = &.{},
    };

    for (this.threads) |*thread| {
        thread.* = try .spawn(.{}, worker, .{
            &this.thread_info,
        });
    }

    this.thread_info.waitUntilDone();
}

pub fn deinit(this: *ChunkMesher) void {
    std.log.info("total chunk mesh time {} ns", .{this.meshing_time_ns});
    std.log.info("mesh time per chunk {} ns", .{std.math.divTrunc(u64, this.meshing_time_ns, this.mesh_alloc.loaded_meshes.count()) catch 0});

    if (mesh_normal_path_count.load(.monotonic) != 0) {
        std.log.info("mean mesh time {} ns", .{mesh_time_ns.load(.monotonic) / mesh_normal_path_count.load(.monotonic)});
        std.log.info("mean get opaque time {} ns", .{get_opaque_ns.load(.monotonic) / mesh_normal_path_count.load(.monotonic)});
        std.log.info("mean edge detect time {} ns", .{edge_detect_ns.load(.monotonic) / mesh_normal_path_count.load(.monotonic)});
        std.log.info("map build time {} ns", .{map_build_ns.load(.monotonic) / mesh_normal_path_count.load(.monotonic)});
        std.log.info("greedy mesh time {} ns", .{greedy_mesh_ns.load(.monotonic) / mesh_normal_path_count.load(.monotonic)});
    }

    this.thread_info.shutdown();

    for (this.threads) |thread| {
        thread.join();
    }

    this.queue.deinit(this.alloc);
    this.alloc.free(this.threads);
    this.arena.deinit();
}

pub fn addRequest(mesher: *ChunkMesher, pos: Chunk.PackedPos) !void {
    try mesher.queue.put(mesher.alloc, pos, {});
}

/// takes in a block pos
/// adds requests for the chunk and adjacent ones if on the boundary
pub fn addRequestWithCollateral(mesher: *ChunkMesher, pos: block.Pos) !void {
    const chunk_pos = World.chunkPosFromBlockPos(pos);
    const rel = World.chunkRelFromBlockPos(pos);
    try mesher.addRequest(.pack(chunk_pos));

    const zero_mask_vec = rel == @as(Chunk.Pos, @splat(0));
    const max_mask_vec = rel == @as(Chunk.Pos, @splat(Chunk.len - 1));

    const zero_mask: [3]bool = zero_mask_vec;
    const max_mask: [3]bool = max_mask_vec;

    if (@reduce(.Or, zero_mask_vec)) {
        for (0..3) |axis| {
            if (zero_mask[axis]) {
                var new: [3]i32 = chunk_pos;
                new[axis] -= 1;
                try mesher.addRequest(.pack(new));
            }
        }
    }

    if (@reduce(.Or, max_mask_vec)) {
        for (0..3) |axis| {
            if (max_mask[axis]) {
                var new: [3]i32 = chunk_pos;
                new[axis] += 1;
                try mesher.addRequest(.pack(new));
            }
        }
    }
}

pub fn addRequestWithFullCollateral(mesher: *ChunkMesher, pos: Chunk.Pos) !void {
    try mesher.addRequest(.pack(pos));
    try mesher.addRequest(.pack(pos + Chunk.Pos{ 1, 0, 0 }));
    try mesher.addRequest(.pack(pos + Chunk.Pos{ -1, 0, 0 }));
    try mesher.addRequest(.pack(pos + Chunk.Pos{ 0, 1, 0 }));
    try mesher.addRequest(.pack(pos + Chunk.Pos{ 0, -1, 0 }));
    try mesher.addRequest(.pack(pos + Chunk.Pos{ 0, 0, 1 }));
    try mesher.addRequest(.pack(pos + Chunk.Pos{ 0, 0, -1 }));
}

const target_mesh_time_ns = 4_000_000;
const max_chunks_meshed = 1024;
pub fn meshMany(this: *ChunkMesher) !void {
    const io = this.thread_info.io;
    const start: std.Io.Timestamp = .now(io, .awake);
    defer this.meshing_time_ns += @intCast(start.untilNow(io, .awake).toNanoseconds());

    _ = this.arena.reset(.retain_capacity);
    const arena = this.arena.allocator();

    const job_count = @min(this.queue.count(), max_chunks_meshed);
    if (job_count == 0) return;

    this.thread_info.jobs = try arena.alloc(Job, job_count);
    var iter = this.queue.iterator();
    var i: usize = 0;
    while (iter.next()) |kv| : (i += 1) {
        if (i >= job_count) break;

        this.thread_info.jobs[i] = .{
            .pos = kv.key_ptr.*,
        };
    }

    this.thread_info.index.store(0, .monotonic);
    this.thread_info.run();
    this.thread_info.waitUntilDone();
    const completed = this.thread_info.completed.load(.monotonic);
    const completed_jobs = this.thread_info.jobs[0..completed];

    std.debug.assert(completed <= this.thread_info.jobs.len);
    try this.mesh_alloc.ensureCapacity(completed);
    for (completed_jobs) |job| {
        _ = this.queue.swapRemove(job.pos);

        if (job.faces.len == 0) {
            _ = this.mesh_alloc.loaded_meshes.swapRemove(job.pos);
            continue;
        }

        try this.mesh_alloc.writeChunkAssumeCapacity(
            job.faces,
            job.pos,
        );
    }
}

const Job = struct {
    pos: Chunk.PackedPos,
    // result
    faces: []GreedyQuad = &.{},
};

const MeshThreadInfo = struct {
    const cl = std.atomic.cache_line;

    phase: std.atomic.Value(u32) align(cl) = .init(0),
    done: std.atomic.Value(u32) align(cl) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),

    index: std.atomic.Value(u32) align(cl) = .init(0),
    completed: std.atomic.Value(u32) align(cl) = .init(0),
    thread_count: u32 align(cl),
    io: std.Io,
    world: *const World,
    jobs: []Job,

    fn waitUntilDone(this: *MeshThreadInfo) void {
        while (true) {
            const cur = this.done.load(.acquire);
            if (cur >= this.thread_count) break;
            this.io.futexWaitUncancelable(u32, &this.done.raw, cur);
        }
    }

    fn run(this: *MeshThreadInfo) void {
        this.done.store(0, .monotonic);
        this.completed.store(0, .monotonic);
        _ = this.phase.fetchAdd(1, .release);
        this.io.futexWake(u32, &this.phase.raw, this.thread_count);
    }

    fn shutdown(this: *MeshThreadInfo) void {
        this.stop.store(true, .monotonic);
        this.io.futexWake(u32, &this.phase.raw, this.thread_count);
    }
};

const MeshingState = struct {
    quads: std.ArrayList(GreedyQuad),
    maps: [6]std.AutoArrayHashMapUnmanaged(QuadData, MaskCube),

    fn init(state: *MeshingState, alloc: std.mem.Allocator) !void {
        state.quads = try .initCapacity(alloc, max_faces);
        for (&state.maps) |*map| map.* = .empty;
    }

    fn deinit(state: *MeshingState, alloc: std.mem.Allocator) void {
        state.quads.deinit(alloc);
        for (&state.maps) |*map| map.deinit(alloc);
    }
};

fn worker(info: *MeshThreadInfo) void {
    const mesher: *ChunkMesher = @fieldParentPtr("thread_info", info);
    const alloc = mesher.alloc;
    const io = info.io;

    var arena_obj: std.heap.ArenaAllocator = .init(alloc);
    defer arena_obj.deinit();
    const arena = arena_obj.allocator();

    var state: MeshingState = undefined;
    state.init(alloc) catch @panic("");
    defer state.deinit(alloc);

    var seen_phase = info.phase.load(.acquire);
    while (true) {
        _ = info.done.fetchAdd(1, .release);
        io.futexWake(u32, &info.done.raw, 1);

        while (true) {
            if (info.stop.load(.monotonic)) return;

            const cur = info.phase.load(.acquire);
            if (cur != seen_phase) {
                seen_phase = cur;
                break;
            }

            io.futexWaitUncancelable(u32, &info.phase.raw, seen_phase);
        }

        var start: std.Io.Timestamp = .now(io, .awake);
        var completed: u32 = 0;
        _ = arena_obj.reset(.retain_capacity);
        while (true) {
            if (start.untilNow(io, .awake).toNanoseconds() > target_mesh_time_ns) break;
            const i = info.index.fetchAdd(1, .monotonic);
            if (i >= info.jobs.len) break;
            const job = &info.jobs[i];

            state.quads.clearRetainingCapacity();
            for (&state.maps) |*map| map.clearRetainingCapacity();

            greedyMeshWithFastExits(alloc, &state, info.world, job.pos.vec());

            job.faces = arena.dupe(GreedyQuad, state.quads.items) catch @panic("");
            completed += 1;
        }

        _ = info.completed.fetchAdd(completed, .monotonic);
    }
}

fn greedyMeshWithFastExits(alloc: std.mem.Allocator, state: *MeshingState, world: *const World, pos: Chunk.Pos) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const start: std.Io.Timestamp = .now(io, .awake);

    const chunk = world.getChunk(pos) orelse return;
    if (chunk.allAirFast()) return;

    var refs: ChunkRefs = .{
        .this = chunk,
        .adjacent = undefined,
    };

    for (0..6) |face_i| {
        const face: block.Face = @enumFromInt(face_i);
        refs.adjacent[face_i] = world.getChunk(pos + face.dir());
    }

    if (chunk.allOpaqueFast()) {
        const adjacent_all_opaque = blk: for (&refs.adjacent) |maybe_chunk| {
            const is_opaque = if (maybe_chunk) |x|
                x.allOpaqueFast()
            else
                !options.render_borders_with_nonexistant_chunks;

            if (!is_opaque) break :blk false;
        } else true;

        if (adjacent_all_opaque)
            return;
    }

    _ = mesh_normal_path_count.fetchAdd(1, .monotonic);
    defer _ = mesh_time_ns.fetchAdd(@intCast(start.untilNow(io, .awake).toNanoseconds()), .monotonic);

    greedyMesh(alloc, state, refs);
}

var mesh_time_ns: std.atomic.Value(u64) = .init(0);

var mesh_normal_path_count: std.atomic.Value(u64) = .init(0);
var get_opaque_ns: std.atomic.Value(u64) = .init(0);
var edge_detect_ns: std.atomic.Value(u64) = .init(0);
var map_build_ns: std.atomic.Value(u64) = .init(0);
var greedy_mesh_ns: std.atomic.Value(u64) = .init(0);

// meshing algorithm from
// https://youtu.be/qnGoGq7DWMc
// https://github.com/TanTanDev/binary_greedy_mesher_demo
const chunk_len_p = Chunk.len + 2;
const MaskP = u64;
const MaskPlaneP = [chunk_len_p]MaskP;
const MaskCubeP = [chunk_len_p]MaskPlaneP;
const Mask = u32;
const MaskPlane = [Chunk.len]Mask;
const MaskCube = [Chunk.len]MaskPlane;
fn greedyMesh(alloc: std.mem.Allocator, state: *MeshingState, refs: ChunkRefs) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var start: std.Io.Timestamp = .now(io, .awake);
    std.debug.assert(Chunk.len == 32);

    // first index is axis, second is how far along plane normal
    var cols: [3]MaskCubeP = @splat(@splat(@splat(0)));

    switch (refs.this.data) {
        .one_to_one => |data| {
            for (0..Chunk.len) |x| {
                for (0..Chunk.len) |y| {
                    for (0..Chunk.len) |z| {
                        const is_opaque = data.getBlock(.{
                            @intCast(x), @intCast(y), @intCast(z),
                        }).isOpaque();
                        const bit: MaskP = if (is_opaque) 1 else 0;

                        const px: u6 = @intCast(x + 1);
                        const py: u6 = @intCast(y + 1);
                        const pz: u6 = @intCast(z + 1);

                        // z,y - x axis
                        cols[0][py][pz] |= bit << px;
                        // x,z - y axis
                        cols[1][pz][px] |= bit << py;
                        // x,y - z axis
                        cols[2][py][px] |= bit << pz;
                    }
                }
            }
        },
        .u2_palette => |data| {
            var opaque_bits: std.StaticBitSet(4) = undefined;
            for (0..4) |i|
                opaque_bits.setValue(i, data.palette[i].isOpaque());

            for (0..Chunk.len) |x| {
                for (0..Chunk.len) |y| {
                    for (0..Chunk.len) |z| {
                        const palette_index = data.getBlock(.{
                            @intCast(x), @intCast(y), @intCast(z),
                        });

                        const bit: MaskP = @intFromBool(opaque_bits.isSet(palette_index));
                        const px: u6 = @intCast(x + 1);
                        const py: u6 = @intCast(y + 1);
                        const pz: u6 = @intCast(z + 1);

                        // z,y - x axis
                        cols[0][py][pz] |= bit << px;
                        // x,z - y axis
                        cols[1][pz][px] |= bit << py;
                        // x,y - z axis
                        cols[2][py][px] |= bit << pz;
                    }
                }
            }
        },
        .uniform => |kind| blk: {
            if (!kind.isOpaque()) break :blk;
            const mask = ((@as(MaskP, 1) << Chunk.len) - 1) << 1;

            for (1..Chunk.len + 1) |px| {
                for (1..Chunk.len + 1) |py| {
                    cols[0][px][py] = mask;
                    cols[1][px][py] = mask;
                    cols[2][px][py] = mask;
                }
            }
        },
    }

    inline for (0..6) |face_i| {
        const face: block.Face = @enumFromInt(face_i);
        if (refs.adjacent[face_i]) |chunk| {
            for (0..Chunk.len) |uu| {
                const u: u5 = @intCast(uu);
                for (0..Chunk.len) |uv| {
                    const v: u5 = @intCast(uv);
                    const l = Chunk.len - 1;
                    const pos: Chunk.RelPos = switch (face) {
                        .north => .{ 0, u, v },
                        .south => .{ l, u, v },
                        .east => .{ u, 0, v },
                        .west => .{ u, l, v },
                        .up => .{ u, v, 0 },
                        .down => .{ u, v, l },
                    };

                    if (!chunk.getBlock(pos).isOpaque()) continue;
                    const px: u6 = switch (face) {
                        .north => Chunk.len + 1,
                        .south => 0,
                        else => @as(u6, u) + 1,
                    };

                    const py: u6 = switch (face) {
                        .east => Chunk.len + 1,
                        .west => 0,
                        .north, .south => @as(u6, u) + 1,
                        .up, .down => @as(u6, v) + 1,
                    };

                    const pz: u6 = switch (face) {
                        .up => Chunk.len + 1,
                        .down => 0,
                        else => @as(u6, v) + 1,
                    };

                    // z,y - x axis
                    cols[0][py][pz] |= @as(MaskP, 1) << px;
                    // x,z - y axis
                    cols[1][pz][px] |= @as(MaskP, 1) << py;
                    // x,y - z axis
                    cols[2][py][px] |= @as(MaskP, 1) << pz;
                }
            }
        }
    }

    _ = get_opaque_ns.fetchAdd(@intCast(start.untilNow(io, .awake).toNanoseconds()), .monotonic);
    start = .now(io, .awake);

    var masks: [6]MaskCube = undefined;
    for (0..3) |axis| {
        for (0..Chunk.len) |z| {
            const vec_size = std.simd.suggestVectorLength(u64) orelse 0;
            const Vec = @Vector(vec_size, u64);
            const Vec32 = @Vector(vec_size, u32);
            const vecs_in_plane = if (vec_size != 0) 32 / vec_size else 0;

            for (0..vecs_in_plane) |i| {
                const plane: Vec = cols[axis][z + 1][1 + i * vec_size ..][0..vec_size].*;
                const one: Vec = @splat(1);
                masks[axis * 2 + 0][z][i * vec_size ..][0..vec_size].* = @as(Vec32, @truncate((plane & ~(plane >> one)) >> one));
                masks[axis * 2 + 1][z][i * vec_size ..][0..vec_size].* = @as(Vec32, @truncate((plane & ~(plane << one)) >> one));
            }

            if (vec_size == 0 or 32 % vec_size != 0) {
                for (vecs_in_plane * vec_size..32) |x| {
                    const col = cols[axis][z + 1][x + 1];
                    masks[axis * 2 + 0][z][x] = @truncate((col & ~(col >> 1)) >> 1);
                    masks[axis * 2 + 1][z][x] = @truncate((col & ~(col << 1)) >> 1);
                }
            }
        }
    }

    _ = edge_detect_ns.fetchAdd(@intCast(start.untilNow(io, .awake).toNanoseconds()), .monotonic);
    start = .now(io, .awake);

    inline for (0..6) |face_int| {
        const face: block.Face = @enumFromInt(face_int);
        var last_quad_data: ?QuadData = null;
        var last_map: *MaskCube = undefined;

        for (0..Chunk.len) |uz| {
            const z: u5 = @intCast(uz);
            for (0..Chunk.len) |ux| {
                const x: u5 = @intCast(ux);
                var col = masks[face_int][z][x];

                while (col != 0) {
                    const y: u5 = @intCast(@ctz(col));
                    col &= col - 1;

                    const pos: Chunk.RelPos = switch (face) {
                        .north, .south => .{ y, z, x },
                        .east, .west => .{ x, y, z },
                        .up, .down => .{ x, z, y },
                    };

                    const ao_sample_dirs: [8][2]i8 = .{
                        .{ -1, -1 },
                        .{ 1, -1 },
                        .{ -1, 1 },
                        .{ 1, 1 },
                        .{ 0, 1 },
                        .{ 0, -1 },
                        .{ -1, 0 },
                        .{ 1, 0 },
                    };

                    var ao_bits: u8 = 0;
                    for (ao_sample_dirs, 0..) |sdir, i| {
                        const sample_offset: block.Pos = switch (face) {
                            .north => .{ 1, -sdir[0], sdir[1] },
                            .south => .{ -1, sdir[0], sdir[1] },
                            .east => .{ sdir[0], 1, sdir[1] },
                            .west => .{ -sdir[0], -1, sdir[1] },
                            .up => .{ sdir[1], sdir[0], 1 },
                            .down => .{ sdir[1], -sdir[0], -1 },
                        };

                        const spos = pos + sample_offset;
                        const px: u6 = @intCast(spos[0] + 1);
                        const py: u6 = @intCast(spos[1] + 1);
                        const pz: u6 = @intCast(spos[2] + 1);

                        if (cols[0][py][pz] & @as(MaskP, 1) << px == 0) continue;
                        ao_bits |= @as(u8, 1) << @intCast(i);
                    }

                    const corner_bl = ao_bits & 1 > 0;
                    const corner_br = ao_bits & 2 > 0;
                    const corner_tl = ao_bits & 4 > 0;
                    const corner_tr = ao_bits & 8 > 0;

                    const side_b = ao_bits & 32 > 0;
                    const side_t = ao_bits & 16 > 0;
                    const side_l = ao_bits & 64 > 0;
                    const side_r = ao_bits & 128 > 0;

                    const ao: AoCorners = .{
                        .bl = aoCorner(corner_bl, side_b, side_l),
                        .br = aoCorner(corner_br, side_b, side_r),
                        .tl = aoCorner(corner_tl, side_t, side_l),
                        .tr = aoCorner(corner_tr, side_t, side_r),
                    };

                    const block_id = refs.this.getBlock(pos);
                    const quad_data: QuadData = .{
                        .tex_id = .fromBlockId(block_id),
                        .ao = ao,
                        .flip = shouldFlip(ao),
                    };

                    const map = if (std.meta.eql(last_quad_data, quad_data))
                        last_map
                    else blk: {
                        const res = state.maps[face_int].getOrPut(alloc, quad_data) catch @panic("oom");
                        if (!res.found_existing) res.value_ptr.* = @splat(@splat(0));
                        last_quad_data = quad_data;
                        last_map = res.value_ptr;
                        break :blk res.value_ptr;
                    };

                    map.*[y][x] |= @as(Mask, 1) << @intCast(z);
                }
            }
        }
    }

    _ = map_build_ns.fetchAdd(@intCast(start.untilNow(io, .awake).toNanoseconds()), .monotonic);
    start = .now(io, .awake);
    defer _ = greedy_mesh_ns.fetchAdd(@intCast(start.untilNow(io, .awake).toNanoseconds()), .monotonic);

    inline for (&state.maps, 0..) |*map, face_int| {
        const face: block.Face = @enumFromInt(face_int);

        var iter = map.iterator();
        while (iter.next()) |kv| {
            const data = kv.key_ptr.*;
            const cube_ptr = kv.value_ptr;

            for (0..Chunk.len) |z| {
                greedyMeshBinaryPlane(&state.quads, cube_ptr.*[z], data, face, @intCast(z));
            }
        }
    }
}

fn aoCorner(corner: bool, side1: bool, side2: bool) u2 {
    if (side1 and side2) return 3;

    const a: u8 = @intFromBool(corner);
    const b: u8 = @intFromBool(side1);
    const c: u8 = @intFromBool(side2);

    return @intCast(a + b + c);
}

fn shouldFlip(ao: AoCorners) bool {
    const bl: u8 = ao.bl;
    const br: u8 = ao.br;
    const tl: u8 = ao.tl;
    const tr: u8 = ao.tr;

    return (bl + tr) > (br + tl);
}

const QuadData = packed struct(u12) {
    ao: AoCorners,
    tex_id: TextureManager.Id,
    flip: bool,
};

inline fn greedyMeshBinaryPlane(quads: *std.ArrayList(GreedyQuad), plane: MaskPlane, data: QuadData, face: block.Face, z: u5) void {
    var new = plane;

    for (0..Chunk.len) |x_usize| {
        const x: u5 = @intCast(x_usize);
        const col = new[x];

        var y_usize: usize = 0;
        while (y_usize < Chunk.len) {
            y_usize += @ctz(col >> @intCast(y_usize));
            if (y_usize >= Chunk.len) continue;
            const y: u5 = @intCast(y_usize);
            const h = @ctz(~(col >> y));

            const h_mask: Mask = @truncate((@as(u64, 1) << h) - 1);
            const mask = h_mask << y;

            var w: usize = 1;
            while (x + w < Chunk.len) {
                const next_h = (new[x + w] >> y) & h_mask;
                if (next_h != h_mask) break;

                new[x + w] &= ~mask;
                w += 1;
            }

            var pos: Chunk.RelPos = switch (face) {
                .north, .south => .{ z, y, x },
                .west, .east => .{ x, z, y },
                .up, .down => .{ x, y, z },
            };

            switch (face) {
                .north, .down => pos[1] += @intCast(h - 1),
                .west => pos[0] += @intCast(w - 1),
                else => {},
            }

            const swapped_w = switch (face) {
                .north, .south, .up, .down => h,
                else => w,
            };

            const swapped_h = switch (face) {
                .north, .south, .up, .down => w,
                else => h,
            };

            quads.appendAssumeCapacity(.{
                .x = .{
                    .face = face,
                    .x = pos[0],
                    .y = pos[1],
                    .z = pos[2],
                    .w = @intCast(swapped_w - 1),
                    .h = @intCast(swapped_h - 1),
                    .flip = @intFromBool(data.flip),
                    .tex_id = data.tex_id,
                },
                .y = .{
                    .ao_corners = data.ao,
                },
            });

            y_usize += h;
        }
    }
}

const ChunkRefs = struct {
    this: Chunk,
    adjacent: [6]?Chunk,

    fn refFromFaceDir(refs: ChunkRefs, face_dir: block.Face) ?Chunk {
        return refs.adjacent[@intFromEnum(face_dir)];
    }

    fn isOpaqueSafe(refs: ChunkRefs, pos: block.Pos, default: bool) bool {
        const lesser_masks = pos < @as(block.Pos, @splat(0));
        const greater_masks = pos > @as(block.Pos, @splat(Chunk.len - 1));
        const lesser_num_vec: @Vector(3, u8) = @intFromBool(lesser_masks);
        const greater_num_vec: @Vector(3, u8) = @intFromBool(greater_masks);
        const axis_out_of_bounds = @reduce(.Add, lesser_num_vec + greater_num_vec);

        return switch (axis_out_of_bounds) {
            0 => refs.this.getBlock(@intCast(pos)).isOpaque(),
            1 => inline for (0..3) |axis| {
                const pos_dir = block.Face.posDirFromAxis(axis);
                const neg_dir = pos_dir.opposite();

                if (lesser_masks[axis]) {
                    return if (refs.refFromFaceDir(neg_dir)) |next|
                        next.getBlock(@intCast(pos + pos_dir.dir() * Chunk.size)).isOpaque()
                    else
                        default;
                }

                if (greater_masks[axis]) {
                    return if (refs.refFromFaceDir(pos_dir)) |next|
                        next.getBlock(@intCast(pos + neg_dir.dir() * Chunk.size)).isOpaque()
                    else
                        default;
                }
            } else unreachable,
            else => default,
        };
    }
};
