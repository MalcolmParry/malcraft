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
    /// width and height currently ignored by shader
    /// width  - 1 so range is 1-32
    w: u5,
    /// height - 1 so range is 1-32
    h: u5,
    unused: u2 = undefined,
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
    chunks: *const Chunk.Map,
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
    chunks: *const Chunk.Map,
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

            const refs: ChunkRefs = .{
                .this = info.chunks.get(pos).?,
                .north = info.chunks.get(pos + @as(Chunk.BlockPos, .{ 1, 0, 0 })),
                .south = info.chunks.get(pos + @as(Chunk.BlockPos, .{ -1, 0, 0 })),
                .east = info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 1, 0 })),
                .west = info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, -1, 0 })),
                .up = info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 0, 1 })),
                .down = info.chunks.get(pos + @as(Chunk.BlockPos, .{ 0, 0, -1 })),
            };

            faces.clearRetainingCapacity();
            mesh(&faces, refs);
            info.faces[i] = arena.dupe(GreedyQuad, faces.items) catch @panic("");
            completed += 1;
        }

        _ = info.completed.fetchAdd(completed, .monotonic);
    }
}

pub fn mesh(faces: *std.ArrayList(GreedyQuad), refs: ChunkRefs) void {
    const chunk = refs.this;

    if (chunk.allAir()) return;
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

    var iter: Chunk.Iterator = .{};
    while (iter.next()) |pos| {
        const block = chunk.getBlock(pos);
        if (block == .air) continue;
        const ipos: Chunk.Pos = pos;

        for (FaceDir.direction_table, 0..) |offset, i| {
            const face_dir: FaceDir = @enumFromInt(i);
            const adjacent = ipos + offset;
            if (refs.isOpaqueSafe(adjacent)) continue;

            const face: GreedyQuad = .{
                .block_id = block,
                .face_dir = face_dir,
                .x = pos[0],
                .y = pos[1],
                .z = pos[2],
                .w = 1 - 1,
                .h = 1 - 1,
            };
            faces.appendAssumeCapacity(face);
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
        inline for (0..3) |axis| {
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
        }

        return refs.this.getBlock(@intCast(pos)).isOpaque();
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
