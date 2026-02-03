const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const gpu = mw.gpu;
const Chunk = @import("Chunk.zig");
const ChunkMeshAllocator = @import("ChunkMeshAllocator.zig");

const ChunkMesher = @This();
pub const max_faces = (32 * 32 * 32 / 2) * 6;

pub const GpuLoaded = struct {
    face_count: u32,
    face_offset: u32,
};

pub const PerFace = packed struct(u32) {
    x: u5,
    y: u5,
    z: u5,
    face: Face,
    block_id: Chunk.BlockId,
    padding: u12 = undefined,
};

alloc: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
mesh_alloc: *ChunkMeshAllocator,

pub fn init(this: *ChunkMesher, mesh_alloc: *ChunkMeshAllocator, alloc: std.mem.Allocator) !void {
    this.alloc = alloc;
    this.arena = .init(alloc);
    this.mesh_alloc = mesh_alloc;
}

pub fn deinit(this: *ChunkMesher) void {
    _ = this;
}

pub fn meshMany(this: *ChunkMesher, chunks: *const std.AutoHashMap(Chunk.ChunkPos, *Chunk), meshes_on_gpu: *std.AutoArrayHashMap(Chunk.ChunkPos, GpuLoaded)) !void {
    _ = this.arena.reset(.retain_capacity);
    const aalloc = this.arena.allocator();

    std.log.info("meshing started", .{});
    const chunk_count: u32 = @min(chunks.count(), 500);
    const jobs: []MeshJob = try aalloc.alloc(MeshJob, chunk_count);
    const all_faces = try aalloc.alloc(PerFace, max_faces * chunk_count);

    var chunk_iter = chunks.iterator();
    for (jobs, 0..chunk_count) |*job, i| {
        const kv = chunk_iter.next().?;
        const pos = kv.key_ptr.*;

        job.* = .{
            .faces = .initBuffer(all_faces[i * max_faces .. (i + 1) * max_faces]),
            .chunk_pos = pos,
        };
    }

    var info: MeshThreadInfo = .{
        .chunks = chunks,
        .jobs = jobs,
    };

    const cpus: u32 = @intCast(std.Thread.getCpuCount() catch 1);
    const thread_count = @min(chunk_count, @max(1, cpus));
    const threads = try aalloc.alloc(std.Thread, thread_count);
    for (threads) |*thread| {
        thread.* = try .spawn(.{}, worker, .{&info});
    }

    while (true) {
        const cur = info.done.load(.acquire);
        if (cur >= thread_count) break;
        std.Thread.Futex.wait(&info.done, cur);
    }

    info.done.store(0, .monotonic);
    _ = info.phase.fetchAdd(1, .release);
    std.Thread.Futex.wake(&info.phase, cpus);

    while (true) {
        const cur = info.done.load(.acquire);
        if (cur >= thread_count) break;
        std.Thread.Futex.wait(&info.done, cur);
    }

    info.stop.store(1, .monotonic);
    std.Thread.Futex.wake(&info.phase, cpus);

    for (threads) |thread| {
        thread.join();
    }

    const data = try aalloc.alloc([]const u8, chunk_count);
    var total_faces: usize = 0;
    for (jobs, data) |job, *faces| {
        faces.* = std.mem.sliceAsBytes(job.faces.items);
        total_faces += job.faces.items.len;
    }

    const regions = try aalloc.alloc(gpu.Buffer.Region, chunk_count);

    var offset: usize = 0;
    try meshes_on_gpu.ensureUnusedCapacity(chunk_count);
    for (regions, jobs) |*region, job| {
        const result = try this.mesh_alloc.allocate(job.faces.items.len);

        region.* = result.buffer_region;
        meshes_on_gpu.putAssumeCapacity(job.chunk_pos, result.gpu_loaded_mesh);
        offset += job.faces.items.len;
    }

    try this.mesh_alloc.writeBuffers(regions, data);
}

const MeshThreadInfo = struct {
    phase: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(u32) = .init(0),

    index: std.atomic.Value(u32) = .init(0),
    chunks: *const std.AutoHashMap(Chunk.ChunkPos, *Chunk),
    jobs: []MeshJob,
};

const MeshJob = struct {
    faces: std.ArrayList(PerFace),
    chunk_pos: Chunk.ChunkPos,
};

fn worker(info: *MeshThreadInfo) void {
    var seen_phase = info.phase.load(.acquire);
    _ = info.done.fetchAdd(1, .release);
    std.Thread.Futex.wake(&info.done, 1);

    while (true) {
        while (true) {
            if (info.stop.load(.monotonic) != 0) return;

            const cur = info.phase.load(.acquire);
            if (cur != seen_phase) {
                seen_phase = cur;
                break;
            }

            std.Thread.Futex.wait(&info.phase, seen_phase);
        }

        while (true) {
            const i = info.index.fetchAdd(1, .monotonic);
            if (i >= info.jobs.len) break;

            const job = &info.jobs[i];
            const chunk = info.chunks.get(job.chunk_pos).?;
            const adjacent_chunks: [6]?*const Chunk = .{
                info.chunks.get(job.chunk_pos + @as(Chunk.BlockPos, .{ 1, 0, 0 })),
                info.chunks.get(job.chunk_pos + @as(Chunk.BlockPos, .{ -1, 0, 0 })),
                info.chunks.get(job.chunk_pos + @as(Chunk.BlockPos, .{ 0, 1, 0 })),
                info.chunks.get(job.chunk_pos + @as(Chunk.BlockPos, .{ 0, -1, 0 })),
                info.chunks.get(job.chunk_pos + @as(Chunk.BlockPos, .{ 0, 0, 1 })),
                info.chunks.get(job.chunk_pos + @as(Chunk.BlockPos, .{ 0, 0, -1 })),
            };

            mesh(&job.faces, chunk, &adjacent_chunks);
        }

        _ = info.done.fetchAdd(1, .release);
        std.Thread.Futex.wake(&info.done, 1);
    }
}

pub fn mesh(faces: *std.ArrayList(PerFace), chunk: *const Chunk, adjacent_chunks: *const [6]?*const Chunk) void {
    var iter: Chunk.Iterator = .{};
    while (iter.next()) |pos| {
        const block = chunk.getBlock(pos);
        if (block == .air) continue;
        const ipos: Chunk.Pos = pos;

        for (Face.direction_table, 0..) |offset, i| {
            const adjacent = ipos + offset;
            if (isOpaqueSafe(chunk, adjacent_chunks, adjacent)) continue;

            const face: PerFace = .{
                .x = pos[0],
                .y = pos[1],
                .z = pos[2],
                .face = @enumFromInt(i),
                .block_id = block,
            };
            faces.appendAssumeCapacity(face);
        }
    }
}

pub inline fn isOpaqueSafe(chunk: *const Chunk, adjacent_chunks: *const [6]?*const Chunk, pos: Chunk.BlockPos) bool {
    @setRuntimeSafety(false);

    inline for (0..3) |axis| {
        if (pos[axis] < 0) {
            if (adjacent_chunks[axis * 2 + 1]) |adjacent| {
                const new_pos = pos + Face.direction_table[axis * 2] * Chunk.size;
                return adjacent.getBlock(@intCast(new_pos)).isOpaque();
            }

            return false;
        }

        if (pos[axis] >= Chunk.len) {
            if (adjacent_chunks[axis * 2]) |adjacent| {
                const new_pos = pos - Face.direction_table[axis * 2] * Chunk.size;
                return adjacent.getBlock(@intCast(new_pos)).isOpaque();
            }

            return false;
        }
    }

    return chunk.getBlock(@intCast(pos)).isOpaque();
}

pub const Face = enum(u3) {
    north,
    south,
    east,
    west,
    up,
    down,

    pub inline fn quat(face: Face) math.Quat {
        return quat_table[@intFromEnum(face)];
    }

    pub inline fn dir(face: Face) Chunk.BlockPos {
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

const normal_face: [6]@Vector(3, f32) = .{
    .{ 0, -0.5, -0.5 },
    .{ 0, 0.5, -0.5 },
    .{ 0, 0.5, 0.5 },

    .{ 0, -0.5, -0.5 },
    .{ 0, 0.5, 0.5 },
    .{ 0, -0.5, 0.5 },
};

pub const face_table = blk: {
    var result: [6][6][4]f32 = undefined;

    for (Face.quat_table, 0..) |q, i| {
        for (0..6) |ii| {
            const f_offset: math.Vec3 = @floatFromInt(Face.direction_table[i]);
            const half_offset = f_offset / @as(math.Vec3, @splat(2.0));

            var vert = normal_face[ii];
            vert = math.quatMulVec(q, vert);
            vert += half_offset;
            const vec4 = math.changeSize(4, vert);
            result[i][ii] = math.toArray(vec4);
        }
    }

    break :blk result;
};
