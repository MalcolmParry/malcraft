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
thread_info: MeshThreadInfo,
threads: []std.Thread,
loaded_meshes: *std.AutoArrayHashMap(Chunk.ChunkPos, GpuLoaded),
meshing_buffer: []PerFace,
per_thread: []PerThread,
queue: []MeshJob,
queue_start: usize,

pub const InitInfo = struct {
    alloc: std.mem.Allocator,
    mesh_alloc: *ChunkMeshAllocator,
    queue: []Chunk.ChunkPos,
    chunks: *const std.AutoHashMap(Chunk.ChunkPos, *Chunk),
    loaded_meshes: *std.AutoArrayHashMap(Chunk.ChunkPos, GpuLoaded),
};

pub fn init(this: *ChunkMesher, info: InitInfo) !void {
    this.alloc = info.alloc;
    this.arena = .init(info.alloc);
    this.mesh_alloc = info.mesh_alloc;
    this.loaded_meshes = info.loaded_meshes;

    const thread_count: u8 = @min(255, @max(1, std.Thread.getCpuCount() catch 1));
    this.threads = try info.alloc.alloc(std.Thread, thread_count);
    errdefer info.alloc.free(this.threads);

    this.meshing_buffer = try info.alloc.alloc(PerFace, @as(usize, max_faces) * thread_count);
    errdefer info.alloc.free(this.meshing_buffer);

    this.per_thread = try info.alloc.alloc(PerThread, thread_count);
    errdefer info.alloc.free(this.per_thread);
    for (this.per_thread, 0..) |*x, i| {
        x.* = .{
            .meshing_buffer = @ptrCast(&this.meshing_buffer[@as(usize, max_faces) * i]),
            .arena = .init(std.heap.page_allocator),
        };
    }

    this.queue_start = 0;
    this.queue = try info.alloc.alloc(MeshJob, info.queue.len);
    errdefer info.alloc.free(this.queue);
    for (this.queue, info.queue) |*job, pos| {
        job.* = .{
            .faces = &.{},
            .chunk_pos = pos,
        };
    }

    this.thread_info = .{
        .thread_count = thread_count,
        .chunks = info.chunks,
        .jobs = undefined,
    };

    for (this.threads, 0..) |*thread, i| {
        thread.* = try .spawn(.{}, worker, .{
            &this.thread_info,
            &this.per_thread[i],
        });
    }

    this.thread_info.waitUntilDone();
}

pub fn deinit(this: *ChunkMesher) void {
    this.thread_info.shutdown();

    for (this.threads) |thread| {
        thread.join();
    }

    for (this.per_thread) |x| x.arena.deinit();
    this.alloc.free(this.per_thread);
    this.alloc.free(this.meshing_buffer);
    this.alloc.free(this.queue);
    this.alloc.free(this.threads);
    this.arena.deinit();
}

const target_mesh_time_ns = 4_000_000;
pub fn meshMany(this: *ChunkMesher) !void {
    _ = this.arena.reset(.retain_capacity);
    const arena = this.arena.allocator();

    const jobs = this.queue[this.queue_start..];
    if (jobs.len == 0) return;

    this.thread_info.index.store(0, .monotonic);
    this.thread_info.jobs = jobs;
    this.thread_info.run();
    this.thread_info.waitUntilDone();
    const completed = this.thread_info.completed.load(.monotonic);
    const completed_jobs = jobs[0..completed];

    const faces = try arena.alloc([]const PerFace, completed);
    var total_faces: usize = 0;
    var index: usize = 0;
    for (completed_jobs) |job| {
        if (job.faces.len == 0) continue;

        faces[index] = job.faces;
        total_faces += job.faces.len;
        index += 1;
    }

    const loaded_meshes = try arena.alloc(GpuLoaded, index);

    var offset: usize = 0;
    try this.loaded_meshes.ensureUnusedCapacity(index);
    index = 0;
    for (completed_jobs) |job| {
        if (job.faces.len == 0) continue;

        const loaded_mesh = try this.mesh_alloc.allocate(job.faces.len);
        this.loaded_meshes.putAssumeCapacity(job.chunk_pos, loaded_mesh);
        loaded_meshes[index] = loaded_mesh;
        offset += job.faces.len;
        index += 1;
    }

    try this.mesh_alloc.writeChunks(loaded_meshes, faces[0..index]);
    for (this.per_thread) |*x| _ = x.arena.reset(.retain_capacity);
    this.queue_start += completed;
}

const MeshThreadInfo = struct {
    const cl = std.atomic.cache_line;

    phase: std.atomic.Value(u32) align(cl) = .init(0),
    done: std.atomic.Value(u32) align(cl) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),

    index: std.atomic.Value(u32) align(cl) = .init(0),
    completed: std.atomic.Value(u32) align(cl) = .init(0),
    thread_count: u32 align(cl),
    chunks: *const std.AutoHashMap(Chunk.ChunkPos, *Chunk),
    jobs: []MeshJob,

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

const PerThread = struct {
    meshing_buffer: *[max_faces]PerFace,
    arena: std.heap.ArenaAllocator,
};

const MeshJob = struct {
    chunk_pos: Chunk.ChunkPos,
    /// allocated by per-thread arena
    faces: []PerFace,
};

fn worker(info: *MeshThreadInfo, per_thread: *PerThread) void {
    var timer = std.time.Timer.start() catch @panic("");
    var faces: std.ArrayList(PerFace) = .initBuffer(per_thread.meshing_buffer);
    const arena = per_thread.arena.allocator();

    var seen_phase = info.phase.load(.acquire);
    _ = info.done.fetchAdd(1, .release);
    std.Thread.Futex.wake(&info.done, 1);

    while (true) {
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
        while (true) {
            if (timer.read() > target_mesh_time_ns) break;
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

            faces.clearRetainingCapacity();
            mesh(&faces, chunk, &adjacent_chunks);
            job.faces = arena.dupe(PerFace, faces.items) catch @panic("");
            _ = info.completed.fetchAdd(1, .monotonic);
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
