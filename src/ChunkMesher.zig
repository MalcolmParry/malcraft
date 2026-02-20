const std = @import("std");
const mw = @import("mwengine");
const Deque = @import("deque.zig").Deque;
const math = mw.math;
const gpu = mw.gpu;
const Chunk = @import("Chunk.zig");
const ChunkMeshAllocator = @import("ChunkMeshAllocator.zig");
const World = @import("World.zig");

const ChunkMesher = @This();
pub const max_faces = (32 * 32 * 32 / 2) * 6;

pub const GpuLoaded = struct {
    face_count: u32,
    face_offset: u32,
    version: u32,
};

const TexId = u1;
pub const GreedyQuad = packed struct(u64) {
    face_dir: FaceDir,
    x: u5,
    y: u5,
    z: u5,
    /// width  - 1 so range is 1-32
    w: u5,
    /// height - 1 so range is 1-32
    h: u5,
    flip: u1,
    tex_id: TexId,
    unused: u2 = undefined,
    // 32 bit boundary
    ao_corners: AoCorners,
    unused2: u24 = undefined,
};

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
queue: std.AutoArrayHashMapUnmanaged(Chunk.ChunkPos, Chunk.Version),

meshing_time_ns: u64,
total_chunks_meshed: u64,
face_count: u64,

pub const InitInfo = struct {
    alloc: std.mem.Allocator,
    mesh_alloc: *ChunkMeshAllocator,
    world: *const World,
};

pub fn init(this: *ChunkMesher, info: InitInfo) !void {
    this.alloc = info.alloc;
    this.arena = .init(info.alloc);
    this.mesh_alloc = info.mesh_alloc;
    this.meshing_time_ns = 0;
    this.total_chunks_meshed = 0;
    this.face_count = 0;
    this.queue = .empty;

    const thread_count: u8 = @min(255, @max(1, std.Thread.getCpuCount() catch 1));
    this.threads = try info.alloc.alloc(std.Thread, thread_count);
    errdefer info.alloc.free(this.threads);

    this.thread_info = .{
        .thread_count = thread_count,
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
    std.log.info("mesh time per chunk {} ns", .{this.meshing_time_ns / this.total_chunks_meshed});
    std.log.info("total chunks meshed {}", .{this.total_chunks_meshed});
    std.log.info("total chunk quads {}", .{this.face_count});
    this.thread_info.shutdown();

    for (this.threads) |thread| {
        thread.join();
    }

    this.queue.deinit(this.alloc);
    this.alloc.free(this.threads);
    this.arena.deinit();
}

pub fn addRequest(mesher: *ChunkMesher, pos: Chunk.ChunkPos) !void {
    const new_version = if (mesher.mesh_alloc.loaded_meshes.get(pos)) |old|
        old.version + 1
    else
        0;

    const entry = try mesher.queue.getOrPut(mesher.alloc, pos);
    entry.value_ptr.* = new_version;
}

/// takes in a block pos
/// adds requests for the chunk and adjacent ones if on the boundary
pub fn addRequestWithCollateral(mesher: *ChunkMesher, pos: Chunk.BlockPos) !void {
    const chunk_pos = World.chunkPosFromBlockPos(pos);
    const rel = World.chunkRelFromBlockPos(pos);
    try mesher.addRequest(chunk_pos);

    const zero_mask = rel == @as(Chunk.ChunkPos, @splat(0));
    const max_mask = rel == @as(Chunk.ChunkPos, @splat(Chunk.len - 1));

    if (@reduce(.Or, zero_mask)) {
        for (0..3) |axis| {
            if (zero_mask[axis]) {
                var new = chunk_pos;
                new[axis] -= 1;
                try mesher.addRequest(new);
            }
        }
    }

    if (@reduce(.Or, max_mask)) {
        for (0..3) |axis| {
            if (max_mask[axis]) {
                var new = chunk_pos;
                new[axis] += 1;
                try mesher.addRequest(new);
            }
        }
    }
}

const target_mesh_time_ns = 4_000_000;
const max_chunks_meshed = 1000;
pub fn meshMany(this: *ChunkMesher) !void {
    var timer: std.time.Timer = try .start();
    defer this.meshing_time_ns += timer.read();

    _ = this.arena.reset(.retain_capacity);
    const arena = this.arena.allocator();

    const SortContext = struct {
        map: *const @TypeOf(this.queue),

        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            const a_pos = ctx.map.keys()[a];
            const b_pos = ctx.map.keys()[b];

            return math.lengthSqr(a_pos) < math.lengthSqr(b_pos);
        }
    };

    this.queue.sort(SortContext{ .map = &this.queue });

    const job_count = @min(this.queue.count(), max_chunks_meshed);
    if (job_count == 0) return;

    this.thread_info.jobs = try arena.alloc(Job, job_count);
    var iter = this.queue.iterator();
    var i: usize = 0;
    while (iter.next()) |kv| : (i += 1) {
        if (i >= job_count) break;

        this.thread_info.jobs[i] = .{
            .pos = kv.key_ptr.*,
            .version = kv.value_ptr.*,
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
            job.version,
        );

        this.face_count += job.faces.len;
    }

    this.total_chunks_meshed += completed;
}

const Job = struct {
    pos: Chunk.ChunkPos,
    version: Chunk.Version,
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
    world: *const World,
    jobs: []Job,

    fn waitUntilDone(this: *MeshThreadInfo) void {
        while (true) {
            const cur = this.done.load(.acquire);
            if (cur >= this.thread_count) break;
            std.Thread.Futex.wait(&this.done, cur);
        }
    }

    fn run(this: *MeshThreadInfo) void {
        this.done.store(0, .monotonic);
        this.completed.store(0, .monotonic);
        _ = this.phase.fetchAdd(1, .release);
        std.Thread.Futex.wake(&this.phase, this.thread_count);
    }

    fn shutdown(this: *MeshThreadInfo) void {
        this.stop.store(true, .monotonic);
        std.Thread.Futex.wake(&this.phase, this.thread_count);
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

    var arena_obj: std.heap.ArenaAllocator = .init(alloc);
    defer arena_obj.deinit();
    const arena = arena_obj.allocator();

    var state: MeshingState = undefined;
    state.init(alloc) catch @panic("");
    defer state.deinit(alloc);

    var timer = std.time.Timer.start() catch @panic("");
    var seen_phase = info.phase.load(.acquire);
    while (true) {
        _ = info.done.fetchAdd(1, .release);
        std.Thread.Futex.wake(&info.done, 1);

        while (true) {
            if (info.stop.load(.monotonic)) return;

            const cur = info.phase.load(.acquire);
            if (cur != seen_phase) {
                seen_phase = cur;
                break;
            }

            std.Thread.Futex.wait(&info.phase, seen_phase);
        }

        timer.reset();
        var completed: u32 = 0;
        _ = arena_obj.reset(.retain_capacity);
        while (true) {
            if (timer.read() > target_mesh_time_ns) break;
            const i = info.index.fetchAdd(1, .monotonic);
            if (i >= info.jobs.len) break;
            const job = &info.jobs[i];

            state.quads.clearRetainingCapacity();
            for (&state.maps) |*map| map.clearRetainingCapacity();

            greedyMeshWithFastExits(alloc, &state, info.world, job.pos);

            job.faces = arena.dupe(GreedyQuad, state.quads.items) catch @panic("");
            completed += 1;
        }

        _ = info.completed.fetchAdd(completed, .monotonic);
    }
}

fn greedyMeshWithFastExits(alloc: std.mem.Allocator, state: *MeshingState, world: *const World, pos: Chunk.ChunkPos) void {
    const chunk = world.chunks.get(pos) orelse return;
    if (chunk.allAir()) return;

    const refs: ChunkRefs = .{
        .this = chunk,
        .north = world.chunks.get(pos + @as(Chunk.BlockPos, .{ 1, 0, 0 })),
        .south = world.chunks.get(pos + @as(Chunk.BlockPos, .{ -1, 0, 0 })),
        .east = world.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 1, 0 })),
        .west = world.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, -1, 0 })),
        .up = world.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 0, 1 })),
        .down = world.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 0, -1 })),
    };

    if (chunk.allOpaque()) {
        const adjacent_all_opaque = blk: for (refs.adjacent()) |maybe_chunk| {
            if (maybe_chunk) |adjacent| {
                if (!adjacent.allOpaque())
                    break :blk false;
            } else break :blk false;
        } else true;

        if (adjacent_all_opaque)
            return;
    }

    greedyMesh(alloc, state, refs);
}

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
    // first index is axis, second is how far along plane normal
    var cols: [3]MaskCubeP = @splat(@splat(@splat(0)));

    for (0..chunk_len_p) |z| {
        for (0..chunk_len_p) |y| {
            for (0..chunk_len_p) |x| {
                const pos_p: Chunk.BlockPos = .{ @intCast(x), @intCast(y), @intCast(z) };
                const pos = pos_p - @as(Chunk.BlockPos, @splat(1));

                if (refs.isOpaqueSafe(pos)) {
                    // z,y - x axis
                    cols[0][y][z] |= @as(MaskP, 1) << @intCast(x);
                    // x,z - y axis
                    cols[1][z][x] |= @as(MaskP, 1) << @intCast(y);
                    // x,y - z axis
                    cols[2][y][x] |= @as(MaskP, 1) << @intCast(z);
                }
            }
        }
    }

    var masks: [6]MaskCubeP = undefined;
    for (0..3) |axis| {
        for (0..chunk_len_p) |z| {
            for (0..chunk_len_p) |x| {
                const col = cols[axis][z][x];
                masks[axis * 2 + 0][z][x] = col & ~(col >> 1);
                masks[axis * 2 + 1][z][x] = col & ~(col << 1);
            }
        }
    }

    for (0..6) |face_int| {
        const face: FaceDir = @enumFromInt(face_int);

        for (0..Chunk.len) |z| {
            for (0..Chunk.len) |x| {
                var col = (masks[face_int][z + 1][x + 1] >> 1) & std.math.maxInt(Mask);

                while (col != 0) {
                    const y = @ctz(col);
                    col &= col - 1;

                    const pos_usize: @Vector(3, usize) = switch (face) {
                        .north, .south => .{ y, z, x },
                        .east, .west => .{ x, y, z },
                        .up, .down => .{ x, z, y },
                    };
                    const pos: Chunk.Pos = @intCast(pos_usize);

                    const ao_sample_dirs: [8]@Vector(2, i32) = .{
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
                        const sample_offset: Chunk.BlockPos = switch (face) {
                            .north => .{ 1, -sdir[0], sdir[1] },
                            .south => .{ -1, sdir[0], sdir[1] },
                            .east => .{ sdir[0], 1, sdir[1] },
                            .west => .{ -sdir[0], -1, sdir[1] },
                            .up => .{ sdir[1], sdir[0], 1 },
                            .down => .{ sdir[1], -sdir[0], -1 },
                        };

                        const spos = pos + sample_offset;
                        if (refs.isOpaqueSafe(spos)) {
                            ao_bits |= @as(u8, 1) << @intCast(i);
                        }
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
                        .tex_id = switch (block_id) {
                            .air => 0,
                            .grass => 0,
                            .stone => 1,
                        },
                        .ao = ao,
                        .flip = shouldFlip(ao),
                    };

                    const res = state.maps[face_int].getOrPut(alloc, quad_data) catch @panic("");
                    if (!res.found_existing) res.value_ptr.* = @splat(@splat(0));
                    res.value_ptr.*[y][x] |= @as(Mask, 1) << @intCast(z);
                }
            }
        }
    }

    for (&state.maps, 0..) |*map, face_int| {
        const face: FaceDir = @enumFromInt(face_int);

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

const QuadData = struct {
    ao: AoCorners,
    flip: bool,
    tex_id: TexId,
};

fn greedyMeshBinaryPlane(quads: *std.ArrayList(GreedyQuad), plane: MaskPlane, data: QuadData, face: FaceDir, z: u5) void {
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

            var pos: Chunk.Pos = switch (face) {
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
                .face_dir = face,
                .x = pos[0],
                .y = pos[1],
                .z = pos[2],
                .w = @intCast(swapped_w - 1),
                .h = @intCast(swapped_h - 1),
                .flip = @intFromBool(data.flip),
                .ao_corners = data.ao,
                .tex_id = data.tex_id,
            });

            y_usize += h;
        }
    }
}

const ChunkRefs = struct {
    this: Chunk,
    north: ?Chunk,
    south: ?Chunk,
    east: ?Chunk,
    west: ?Chunk,
    up: ?Chunk,
    down: ?Chunk,

    fn adjacent(refs: ChunkRefs) [6]?Chunk {
        return .{
            refs.north,
            refs.south,
            refs.east,
            refs.west,
            refs.up,
            refs.down,
        };
    }

    fn refFromFaceDir(refs: ChunkRefs, face_dir: FaceDir) ?Chunk {
        return switch (face_dir) {
            .north => refs.north,
            .south => refs.south,
            .east => refs.east,
            .west => refs.west,
            .up => refs.up,
            .down => refs.down,
        };
    }

    fn isOpaqueSafe(refs: ChunkRefs, pos: Chunk.BlockPos) bool {
        var axis_out_of_bounds: usize = 0;
        for (0..3) |axis| {
            if (pos[axis] < 0 or pos[axis] >= Chunk.len)
                axis_out_of_bounds += 1;
        }

        return switch (axis_out_of_bounds) {
            0 => refs.this.getBlock(@intCast(pos)).isOpaque(),
            1 => inline for (0..3) |axis| {
                const pos_dir: FaceDir = FaceDir.posDirFromAxis(axis);
                const neg_dir = pos_dir.opposite();

                if (pos[axis] < 0) {
                    return if (refs.refFromFaceDir(neg_dir)) |next|
                        next.getBlock(@intCast(pos + pos_dir.dir() * Chunk.size)).isOpaque()
                    else
                        false;
                }

                if (pos[axis] >= Chunk.len) {
                    return if (refs.refFromFaceDir(pos_dir)) |next|
                        next.getBlock(@intCast(pos + neg_dir.dir() * Chunk.size)).isOpaque()
                    else
                        false;
                }
            } else unreachable,
            else => false,
        };
    }
};

pub const FaceDir = enum(u3) {
    north,
    south,
    east,
    west,
    up,
    down,

    pub inline fn posDirFromAxis(axis: u8) FaceDir {
        return @enumFromInt(axis * 2);
    }

    pub inline fn opposite(face: FaceDir) FaceDir {
        return switch (face) {
            .north => .south,
            .south => .north,
            .east => .west,
            .west => .east,
            .up => .down,
            .down => .up,
        };
    }

    pub inline fn quat(face: FaceDir) math.Quat {
        return quat_table[@intFromEnum(face)];
    }

    pub inline fn dir(face: FaceDir) Chunk.BlockPos {
        return direction_table[@intFromEnum(face)];
    }

    const quat_table: [6]math.Quat = .{
        math.quat_identity,
        math.quatFromAxisAngle(math.dir_up, math.rad(180.0)),
        math.quatFromAxisAngle(math.dir_up, math.rad(90.0)),
        math.quatFromAxisAngle(math.dir_up, math.rad(-90.0)),
        math.quatFromAxisAngle(math.dir_right, math.rad(-90.0)),
        math.quatFromAxisAngle(math.dir_right, math.rad(90.0)),
    };

    const direction_table: [6]Chunk.BlockPos = .{
        .{ 1, 0, 0 },
        .{ -1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, -1, 0 },
        .{ 0, 0, 1 },
        .{ 0, 0, -1 },
    };
};
