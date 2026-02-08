const std = @import("std");
const mw = @import("mwengine");
const Deque = @import("deque.zig").Deque;
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

pub const GreedyQuad = packed struct(u32) {
    block_id: Chunk.BlockId,
    face_dir: FaceDir,
    x: u5,
    y: u5,
    z: u5,
    // width and height currently ignored by shader
    w: u6,
    h: u6,
    unused: u0 = undefined,
};

alloc: std.mem.Allocator,
arena: std.heap.ArenaAllocator,
mesh_alloc: *ChunkMeshAllocator,
thread_info: MeshThreadInfo,
threads: []std.Thread,
meshing_buffer: []GreedyQuad,
per_thread: []PerThread,

meshing_time_ns: u64,
total_chunks_meshed: u64,

pub const InitInfo = struct {
    alloc: std.mem.Allocator,
    mesh_alloc: *ChunkMeshAllocator,
    chunks: *const std.AutoHashMap(Chunk.ChunkPos, Chunk),
};

pub fn init(this: *ChunkMesher, info: InitInfo) !void {
    this.alloc = info.alloc;
    this.arena = .init(info.alloc);
    this.mesh_alloc = info.mesh_alloc;
    this.meshing_time_ns = 0;
    this.total_chunks_meshed = 0;

    const thread_count: u8 = @min(255, @max(1, std.Thread.getCpuCount() catch 1));
    this.threads = try info.alloc.alloc(std.Thread, thread_count);
    errdefer info.alloc.free(this.threads);

    this.meshing_buffer = try info.alloc.alloc(GreedyQuad, @as(usize, max_faces) * thread_count);
    errdefer info.alloc.free(this.meshing_buffer);

    this.per_thread = try info.alloc.alloc(PerThread, thread_count);
    errdefer info.alloc.free(this.per_thread);
    for (this.per_thread, 0..) |*x, i| {
        x.* = .{
            .meshing_buffer = @ptrCast(&this.meshing_buffer[@as(usize, max_faces) * i]),
            .arena = .init(std.heap.page_allocator),
        };
    }

    this.thread_info = .{
        .thread_count = thread_count,
        .chunks = info.chunks,
        .queue = .empty,
        .faces = undefined,
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
    std.log.info("total chunk mesh time {} ns", .{this.meshing_time_ns});
    std.log.info("mesh time per chunk {} ns", .{this.meshing_time_ns / this.total_chunks_meshed});
    std.log.info("total chunks meshed {}", .{this.total_chunks_meshed});
    this.thread_info.shutdown();

    for (this.threads) |thread| {
        thread.join();
    }

    this.thread_info.queue.deinit(this.alloc);
    for (this.per_thread) |x| x.arena.deinit();
    this.alloc.free(this.per_thread);
    this.alloc.free(this.meshing_buffer);
    this.alloc.free(this.threads);
    this.arena.deinit();
}

const target_mesh_time_ns = 4_000_000;
const max_chunks_meshed = 1000;
pub fn meshMany(this: *ChunkMesher) !void {
    var timer: std.time.Timer = try .start();
    defer this.meshing_time_ns += timer.read();

    _ = this.arena.reset(.retain_capacity);
    const arena = this.arena.allocator();

    const job_count = @min(this.thread_info.queue.len, max_chunks_meshed);
    if (job_count == 0) return;

    this.thread_info.index.store(0, .monotonic);
    this.thread_info.faces = try arena.alloc([]GreedyQuad, job_count);
    this.thread_info.run();
    this.thread_info.waitUntilDone();
    const completed = this.thread_info.completed.load(.monotonic);
    const all_faces = this.thread_info.faces[0..completed];

    try this.mesh_alloc.ensureCapacity(completed);
    var index: usize = 0;
    for (all_faces, 0..) |faces, full_i| {
        if (faces.len == 0) continue;

        const pos = this.thread_info.queue.at(full_i);
        try this.mesh_alloc.writeChunkAssumeCapacity(faces, pos);
        index += 1;
    }

    std.debug.assert(index <= completed);
    std.debug.assert(completed <= this.thread_info.queue.len);
    for (this.per_thread) |*x| _ = x.arena.reset(.retain_capacity);
    this.thread_info.queue.head += completed;
    this.thread_info.queue.head %= this.thread_info.queue.buffer.len;
    this.thread_info.queue.len -= completed;
    this.total_chunks_meshed += completed;
}

const MeshThreadInfo = struct {
    const cl = std.atomic.cache_line;

    phase: std.atomic.Value(u32) align(cl) = .init(0),
    done: std.atomic.Value(u32) align(cl) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),

    index: std.atomic.Value(u32) align(cl) = .init(0),
    completed: std.atomic.Value(u32) align(cl) = .init(0),
    thread_count: u32 align(cl),
    chunks: *const std.AutoHashMap(Chunk.ChunkPos, Chunk),
    queue: Deque(Chunk.ChunkPos),
    faces: [][]GreedyQuad,

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
    meshing_buffer: *[max_faces]GreedyQuad,
    arena: std.heap.ArenaAllocator,
};

pub const AdjacentChunks = [6]?Chunk;

fn worker(info: *MeshThreadInfo, per_thread: *PerThread) void {
    var timer = std.time.Timer.start() catch @panic("");
    var faces: std.ArrayList(GreedyQuad) = .initBuffer(per_thread.meshing_buffer);
    const arena = per_thread.arena.allocator();

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
        while (true) {
            if (timer.read() > target_mesh_time_ns) break;
            const i = info.index.fetchAdd(1, .monotonic);
            if (i >= info.faces.len) break;

            const pos = info.queue.at(i);
            const chunk = info.chunks.get(pos).?;

            const adjacent_chunks: AdjacentChunks = .{
                info.chunks.get(pos + @as(Chunk.BlockPos, .{ 1, 0, 0 })),
                info.chunks.get(pos + @as(Chunk.BlockPos, .{ -1, 0, 0 })),
                info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 1, 0 })),
                info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, -1, 0 })),
                info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 0, 1 })),
                info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 0, -1 })),
            };

            faces.clearRetainingCapacity();
            mesh(&faces, chunk, &adjacent_chunks);
            info.faces[i] = arena.dupe(GreedyQuad, faces.items) catch @panic("");
            completed += 1;
        }

        _ = info.completed.fetchAdd(completed, .monotonic);
    }
}

pub fn mesh(faces: *std.ArrayList(GreedyQuad), chunk: Chunk, adjacent_chunks: *const AdjacentChunks) void {
    if (chunk.allAir()) return;
    if (chunk.allOpaque()) {
        const adjacent_all_opaque = blk: for (adjacent_chunks) |maybe_chunk| {
            if (maybe_chunk) |adjacent| {
                if (!adjacent.allOpaque())
                    break :blk false;
            } else break :blk false;
        } else true;

        if (adjacent_all_opaque)
            return;
    }

    var iter: Chunk.Iterator = .{};
    while (iter.next()) |pos| {
        const block = chunk.getBlock(pos);
        if (block == .air) continue;
        const ipos: Chunk.Pos = pos;

        for (FaceDir.direction_table, 0..) |offset, i| {
            const face_dir: FaceDir = @enumFromInt(i);
            const adjacent = ipos + offset;
            if (isOpaqueSafe(chunk, adjacent_chunks, adjacent)) continue;
            switch (face_dir) {
                .north, .east => {},
                else => continue,
            }

            const face: GreedyQuad = .{
                .block_id = block,
                .face_dir = face_dir,
                .x = pos[0],
                .y = pos[1],
                .z = pos[2],
                .w = 1,
                .h = 1,
            };
            faces.appendAssumeCapacity(face);
        }
    }
}

pub inline fn isOpaqueSafe(chunk: Chunk, adjacent_chunks: *const AdjacentChunks, pos: Chunk.BlockPos) bool {
    inline for (0..3) |axis| {
        if (pos[axis] < 0) {
            if (adjacent_chunks[axis * 2 + 1]) |adjacent| {
                const new_pos = pos + FaceDir.direction_table[axis * 2] * Chunk.size;
                return adjacent.getBlock(@intCast(new_pos)).isOpaque();
            }

            return false;
        }

        if (pos[axis] >= Chunk.len) {
            if (adjacent_chunks[axis * 2]) |adjacent| {
                const new_pos = pos - FaceDir.direction_table[axis * 2] * Chunk.size;
                return adjacent.getBlock(@intCast(new_pos)).isOpaque();
            }

            return false;
        }
    }

    return chunk.getBlock(@intCast(pos)).isOpaque();
}

pub const FaceDir = enum(u3) {
    north,
    south,
    east,
    west,
    up,
    down,

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

    for (FaceDir.quat_table, 0..) |q, i| {
        for (0..6) |ii| {
            const f_offset: math.Vec3 = @floatFromInt(FaceDir.direction_table[i]);
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
