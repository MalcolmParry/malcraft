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

io: std.Io,
alloc: std.mem.Allocator,
mesh_alloc: *ChunkMeshAllocator,
queue: std.AutoArrayHashMapUnmanaged(Chunk.PackedPos, void) = .empty,
world: *const World,
thread_states: []ThreadState,

mesh_batch_count: u64 = 0,
meshing_time_ns: u64 = 0,
meshed_chunk_count: u64 = 0,

pub const InitInfo = struct {
    io: std.Io,
    alloc: std.mem.Allocator,
    mesh_alloc: *ChunkMeshAllocator,
    world: *const World,
};

pub fn init(mesher: *ChunkMesher, info: InitInfo) !void {
    const alloc = info.alloc;

    const thread_count = @max(1, std.Thread.getCpuCount() catch 1);
    const thread_states = try alloc.alloc(ThreadState, thread_count);
    errdefer alloc.free(thread_states);

    for (thread_states) |*state| try state.init(alloc);

    mesher.* = .{
        .io = info.io,
        .alloc = alloc,
        .mesh_alloc = info.mesh_alloc,
        .world = info.world,
        .thread_states = thread_states,
    };
}

pub fn deinit(mesher: *ChunkMesher) void {
    const alloc = mesher.alloc;

    std.log.info("total chunk mesh time {} ns in {} batches", .{ mesher.meshing_time_ns, mesher.mesh_batch_count });
    if (mesher.meshed_chunk_count != 0) {
        std.log.info("mean mesh time per batch {} ns", .{mesher.meshing_time_ns / mesher.mesh_batch_count});
        std.log.info("mesh time per chunk {} ns", .{mesher.meshing_time_ns / mesher.meshed_chunk_count});
    }

    for (mesher.thread_states) |*state| state.deinit(alloc);
    alloc.free(mesher.thread_states);
    mesher.queue.deinit(mesher.alloc);
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

    const zero: Chunk.Pos = @splat(0);
    var zero_mask: u3 = @bitCast(rel == zero);
    while (zero_mask != 0) {
        const axis = @ctz(zero_mask);
        zero_mask &= zero_mask - 1;

        var new: [3]i32 = chunk_pos;
        new[axis] -= 1;
        try mesher.addRequest(.pack(new));
    }

    const max: Chunk.Pos = @splat(Chunk.len - 1);
    var max_mask: u3 = @bitCast(rel == max);
    while (max_mask != 0) {
        const axis = @ctz(max_mask);
        max_mask &= max_mask - 1;

        var new: [3]i32 = chunk_pos;
        new[axis] += 1;
        try mesher.addRequest(.pack(new));
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

pub fn meshMany(mesher: *ChunkMesher, deadline: std.Io.Timestamp) !void {
    const io = mesher.io;

    const thread_count = mesher.thread_states.len;
    const jobs = mesher.queue.keys();
    if (jobs.len == 0) return;

    const start: std.Io.Timestamp = .now(io, .boot);
    mesher.mesh_batch_count += 1;
    defer mesher.meshing_time_ns += @intCast(start.untilNow(io, .boot).toNanoseconds());

    const jobs_per_thread = jobs.len / thread_count;
    for (mesher.thread_states, 0..) |*state, i| {
        const owned_job_count = if (i + 1 == thread_count) jobs.len - (thread_count * jobs_per_thread) else jobs_per_thread;

        state.completed.clearRetainingCapacity();
        state.job_index = .init(0);
        state.jobs = jobs[i * jobs_per_thread ..][0..owned_job_count];
    }

    var group: std.Io.Group = .init;
    for (0..mesher.thread_states.len) |thread_i| {
        try group.concurrent(io, worker, .{ mesher, thread_i, deadline });
    }
    try group.await(io);

    for (mesher.thread_states) |state| {
        try mesher.mesh_alloc.ensureCapacity(state.completed.items.len);
        mesher.meshed_chunk_count += state.completed.items.len;

        for (state.completed.items) |job| {
            _ = mesher.queue.swapRemove(job.pos);

            if (job.faces.len == 0) {
                if (mesher.mesh_alloc.loaded_meshes.fetchSwapRemove(job.pos)) |kv|
                    try mesher.mesh_alloc.queueFree(kv.value);

                continue;
            }

            try mesher.mesh_alloc.writeChunkAssumeCapacity(job.faces, job.pos);
        }
    }
}

const CompletedJob = struct {
    pos: Chunk.PackedPos,
    faces: []GreedyQuad,
};

const ThreadState = struct {
    _: void align(std.atomic.cache_line) = {},
    arena: std.heap.ArenaAllocator,
    quads: std.ArrayList(GreedyQuad),
    maps: [6]std.AutoArrayHashMapUnmanaged(QuadData, MaskCube),

    job_index: std.atomic.Value(u64),
    jobs: []Chunk.PackedPos,
    completed: std.ArrayList(CompletedJob),

    fn init(state: *ThreadState, alloc: std.mem.Allocator) !void {
        state.arena = .init(alloc);
        state.quads = try .initCapacity(alloc, max_faces);
        for (&state.maps) |*map| map.* = .empty;
        state.completed = .empty;
    }

    fn deinit(state: *ThreadState, alloc: std.mem.Allocator) void {
        state.arena.deinit();
        state.quads.deinit(alloc);
        for (&state.maps) |*map| map.deinit(alloc);
        state.completed.deinit(alloc);
    }
};

fn worker(mesher: *ChunkMesher, thread_i: usize, deadline: std.Io.Timestamp) void {
    const io = mesher.io;
    const alloc = mesher.alloc;
    const state = &mesher.thread_states[thread_i];

    const arena = state.arena.allocator();
    _ = state.arena.reset(.retain_capacity);

    var queue_index = thread_i;
    while (true) top: {
        if (deadline.untilNow(io, .boot).toNanoseconds() > 0) break;

        const pos = job: while (true) {
            const queue = &mesher.thread_states[queue_index];
            const i = queue.job_index.fetchAdd(1, .monotonic);

            if (i >= queue.jobs.len) {
                queue_index = (queue_index + 1) % mesher.thread_states.len;
                if (queue_index == thread_i) break :top;
                continue;
            }

            break :job queue.jobs[i];
        };

        state.quads.clearRetainingCapacity();
        for (&state.maps) |*map| map.clearRetainingCapacity();

        greedyMeshWithFastExits(alloc, state, mesher.world, pos.vec());

        state.completed.append(alloc, .{
            .pos = pos,
            .faces = arena.dupe(GreedyQuad, state.quads.items) catch @panic("oom"),
        }) catch @panic("oom");
    }
}

fn greedyMeshWithFastExits(alloc: std.mem.Allocator, state: *ThreadState, world: *const World, pos: Chunk.Pos) void {
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
fn greedyMesh(alloc: std.mem.Allocator, state: *ThreadState, refs: ChunkRefs) void {
    std.debug.assert(Chunk.len == 32);

    // first index is axis, second is how far along plane normal
    var opaque_cols: [3]MaskCubeP = @splat(@splat(@splat(0)));
    var water_cols: [3]MaskCubeP = @splat(@splat(@splat(0)));

    switch (refs.this.data) {
        .one_to_one => |data| {
            const bytes_per_col = Chunk.len / 2;
            const bytes_per_plane = bytes_per_col * Chunk.len;
            const Vec = @Vector(bytes_per_col, u8);
            const FullVec = @Vector(Chunk.len, u8);

            for (0..Chunk.len) |x| {
                const px: u6 = @intCast(x + 1);
                const plane = data.blocks[x * bytes_per_plane ..][0..bytes_per_plane];
                for (0..Chunk.len) |y| {
                    const py: u6 = @intCast(y + 1);
                    const col_slice: *align(16) const [bytes_per_col]u8 = @alignCast(plane[y * bytes_per_col ..][0..bytes_per_col]);
                    comptime std.debug.assert(@alignOf(Vec) <= 16);
                    const vec: Vec = col_slice.*;

                    const mask: Vec = @splat(15);
                    const v0 = vec & mask;
                    const v1 = vec >> @splat(4);
                    const v: FullVec = std.simd.interlace(.{ v0, v1 });

                    var opaque_col: Mask = @bitCast(@call(.always_inline, block.Kind.isOpaqueVec, .{v}));
                    var water_col: Mask = @bitCast(v == @as(FullVec, @splat(@intFromEnum(block.Kind.water))));

                    opaque_cols[2][py][px] = @as(MaskP, opaque_col) << 1;
                    while (opaque_col != 0) {
                        const z = @ctz(opaque_col);
                        opaque_col &= opaque_col - 1;
                        const pz: u6 = @intCast(z + 1);

                        opaque_cols[0][py][pz] |= @as(MaskP, 1) << px;
                        opaque_cols[1][pz][px] |= @as(MaskP, 1) << py;
                    }

                    water_cols[2][py][px] = @as(MaskP, water_col) << 1;
                    while (water_col != 0) {
                        const z = @ctz(water_col);
                        water_col &= water_col - 1;
                        const pz: u6 = @intCast(z + 1);

                        water_cols[0][py][pz] |= @as(MaskP, 1) << px;
                        water_cols[1][pz][px] |= @as(MaskP, 1) << py;
                    }
                }
            }
        },
        .u2_palette => |data| {
            var opaque_bits: std.StaticBitSet(4) = undefined;
            var water_index: u8 = 255;
            for (0..4) |i| {
                const kind = data.palette[i];
                if (kind == .water) water_index = @intCast(i);
                opaque_bits.setValue(i, kind.isOpaque());
            }

            const bytes_per_col = Chunk.len / 4;
            const bytes_per_plane = bytes_per_col * Chunk.len;
            const Vec = @Vector(bytes_per_col, u8);
            const FullVec = @Vector(Chunk.len, u8);

            for (0..Chunk.len) |x| {
                const px: u6 = @intCast(x + 1);
                const plane = data.blocks[x * bytes_per_plane ..][0..bytes_per_plane];
                for (0..Chunk.len) |y| {
                    const py: u6 = @intCast(y + 1);
                    const col_slice: *align(8) const [bytes_per_col]u8 = @alignCast(plane[y * bytes_per_col ..][0..bytes_per_col]);
                    const vec: Vec = col_slice.*;

                    const mask: Vec = @splat(3);
                    const v0 = vec & mask;
                    const v1 = (vec >> @splat(2)) & mask;
                    const v2 = (vec >> @splat(4)) & mask;
                    const v3 = vec >> @splat(6);

                    const v: FullVec = std.simd.interlace(.{ v0, v1, v2, v3 });
                    var water_col: Mask = @bitCast(v == @as(FullVec, @splat(water_index)));
                    var opaque_col: Mask = 0;
                    inline for (0..4) |i| {
                        if (opaque_bits.isSet(i)) {
                            const b = v == @as(FullVec, @splat(@intCast(i)));
                            opaque_col |= @bitCast(b);
                        }
                    }

                    opaque_cols[2][py][px] = @as(MaskP, opaque_col) << 1;
                    while (opaque_col != 0) {
                        const z = @ctz(opaque_col);
                        opaque_col &= opaque_col - 1;
                        const pz: u6 = @intCast(z + 1);

                        opaque_cols[0][py][pz] |= @as(MaskP, 1) << px;
                        opaque_cols[1][pz][px] |= @as(MaskP, 1) << py;
                    }

                    water_cols[2][py][px] = @as(MaskP, water_col) << 1;
                    while (water_col != 0) {
                        const z = @ctz(water_col);
                        water_col &= water_col - 1;
                        const pz: u6 = @intCast(z + 1);

                        water_cols[0][py][pz] |= @as(MaskP, 1) << px;
                        water_cols[1][pz][px] |= @as(MaskP, 1) << py;
                    }
                }
            }
        },
        .uniform => |kind| blk: {
            if (!kind.isOpaque() and kind != .water) break :blk;
            const cols = if (kind == .water) &water_cols else &opaque_cols;
            const mask = ((1 << Chunk.len) - 1) << 1;

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

                    const kind = chunk.getBlock(pos);
                    if (!kind.isOpaque() and kind != .water) continue;
                    const cols = if (kind == .water) &water_cols else &opaque_cols;

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

    var masks: [6]MaskCube = undefined;
    for (0..3) |axis| {
        for (0..Chunk.len) |z| {
            const vec_size = std.simd.suggestVectorLength(MaskP) orelse 0;
            const VecP = @Vector(vec_size, MaskP);
            const Vec = @Vector(vec_size, Mask);
            const vecs_in_plane = if (vec_size != 0) Chunk.len / vec_size else 0;

            for (0..vecs_in_plane) |i| {
                const opaque_plane: VecP = opaque_cols[axis][z + 1][1 + i * vec_size ..][0..vec_size].*;
                const water_plane: VecP = water_cols[axis][z + 1][1 + i * vec_size ..][0..vec_size].*;
                const one: VecP = @splat(1);

                const opaque_pos_edges: Vec = @truncate((opaque_plane & ~(opaque_plane >> one)) >> one);
                const opaque_neg_edges: Vec = @truncate((opaque_plane & ~(opaque_plane << one)) >> one);

                const water_pos_edges: Vec = @truncate((water_plane & ~((water_plane | opaque_plane) >> one)) >> one);
                const water_neg_edges: Vec = @truncate((water_plane & ~((water_plane | opaque_plane) << one)) >> one);

                masks[axis * 2 + 0][z][i * vec_size ..][0..vec_size].* = opaque_pos_edges | water_pos_edges;
                masks[axis * 2 + 1][z][i * vec_size ..][0..vec_size].* = opaque_neg_edges | water_neg_edges;
            }

            for (vecs_in_plane * vec_size..Chunk.len) |x| {
                const opaque_col = opaque_cols[axis][z + 1][x + 1];
                const water_col = water_cols[axis][z + 1][x + 1];

                const opaque_pos_edges: Mask = @truncate((opaque_col & ~(opaque_col >> 1)) >> 1);
                const opaque_neg_edges: Mask = @truncate((opaque_col & ~(opaque_col << 1)) >> 1);

                const water_pos_edges: Mask = @truncate((water_col & ~((water_col | opaque_col) >> 1)) >> 1);
                const water_neg_edges: Mask = @truncate((water_col & ~((water_col | opaque_col) << 1)) >> 1);

                masks[axis * 2 + 0][z][x] = opaque_pos_edges | water_pos_edges;
                masks[axis * 2 + 1][z][x] = opaque_neg_edges | water_neg_edges;
            }
        }
    }

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

                        if (opaque_cols[0][py][pz] & @as(MaskP, 1) << px == 0) continue;
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
};
