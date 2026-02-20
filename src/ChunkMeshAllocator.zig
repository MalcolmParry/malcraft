const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const ChunkMesher = @import("ChunkMesher.zig");
const Chunk = @import("Chunk.zig");
const RendererInfo = @import("Renderer.zig").Info;

const ChunkMeshAllocator = @This();
pub const buffer_size = 1024 * 1024 * 512;
pub const staging_size = 1024 * 1024 * 32;

comptime {
    std.debug.assert(buffer_size / @sizeOf(ChunkMesher.GreedyQuad) < std.math.maxInt(u32));
}

renderer_info: *const RendererInfo,
free_queues: []std.ArrayList(ChunkMesher.GpuLoaded),
staging: gpu.Buffer,
staging_face_offset: gpu.Size,
mapping: []ChunkMesher.GreedyQuad,
buffer: gpu.Buffer,
free_list: std.DoublyLinkedList,
alloc: std.mem.Allocator,
device: gpu.Device,
loaded_meshes: std.AutoArrayHashMapUnmanaged(Chunk.ChunkPos, ChunkMesher.GpuLoaded),
buffer_copy_src: std.ArrayList(gpu.Buffer.Region),
buffer_copy_dst: std.ArrayList(gpu.Buffer.Region),
memory_barriers: std.ArrayList(gpu.CommandEncoder.MemoryBarrier),

const FreeRegion = struct {
    offset: gpu.Size,
    size: gpu.Size,
    node: std.DoublyLinkedList.Node = .{},
};

const InitInfo = struct {
    device: gpu.Device,
    alloc: std.mem.Allocator,
    renderer_info: *const RendererInfo,
};

pub fn init(this: *ChunkMeshAllocator, info: InitInfo) !void {
    this.renderer_info = info.renderer_info;

    this.free_queues = try info.alloc.alloc(std.ArrayList(ChunkMesher.GpuLoaded), info.renderer_info.frames_in_flight);
    @memset(this.free_queues, .empty);

    this.staging = try .init(info.device, .{
        .alloc = info.alloc,
        .loc = .host,
        .size = staging_size,
        .usage = .{
            .src = true,
        },
    });
    errdefer this.staging.deinit(info.device, info.alloc);

    const byte_mapping = try this.staging.map(info.device);
    const face_mapping: [*]ChunkMesher.GreedyQuad = @ptrCast(@alignCast(byte_mapping));
    this.mapping = face_mapping[0 .. staging_size / @sizeOf(ChunkMesher.GreedyQuad)];

    this.buffer = try .init(info.device, .{
        .alloc = info.alloc,
        .loc = .device,
        .size = buffer_size,
        .usage = .{
            .vertex = true,
            .dst = true,
        },
    });
    errdefer this.buffer.deinit(info.device, info.alloc);

    const free_region = try info.alloc.create(FreeRegion);
    errdefer info.alloc.destroy(free_region);
    free_region.* = .{
        .offset = 0,
        .size = buffer_size,
    };

    this.free_list = .{};
    this.free_list.append(&free_region.node);

    this.alloc = info.alloc;
    this.device = info.device;
    this.buffer_copy_src = .empty;
    this.buffer_copy_dst = .empty;
    this.memory_barriers = .empty;
    this.loaded_meshes = .empty;
    this.staging_face_offset = 0;
}

pub fn deinit(this: *ChunkMeshAllocator) void {
    std.log.info("{} bytes used in chunk mesh buffer", .{buffer_size - this.queryBytesFree()});

    this.memory_barriers.deinit(this.alloc);
    this.buffer_copy_dst.deinit(this.alloc);
    this.buffer_copy_src.deinit(this.alloc);

    var maybe_node = this.free_list.first;
    while (maybe_node) |node| {
        const next = node.next;
        const free_region: *FreeRegion = @fieldParentPtr("node", node);
        this.alloc.destroy(free_region);
        maybe_node = next;
    }

    for (this.free_queues) |*queue| {
        queue.deinit(this.alloc);
    }
    this.alloc.free(this.free_queues);

    this.buffer.deinit(this.device, this.alloc);
    this.staging.unmap(this.device);
    this.staging.deinit(this.device, this.alloc);
    this.loaded_meshes.deinit(this.alloc);
}

pub fn ensureCapacity(mesh_alloc: *ChunkMeshAllocator, count: usize) !void {
    try mesh_alloc.buffer_copy_src.ensureUnusedCapacity(mesh_alloc.alloc, count);
    try mesh_alloc.buffer_copy_dst.ensureUnusedCapacity(mesh_alloc.alloc, count);
    try mesh_alloc.memory_barriers.ensureUnusedCapacity(mesh_alloc.alloc, count);
    try mesh_alloc.loaded_meshes.ensureUnusedCapacity(mesh_alloc.alloc, count);
}

pub fn writeChunkAssumeCapacity(this: *ChunkMeshAllocator, on_cpu: []const ChunkMesher.GreedyQuad, pos: Chunk.ChunkPos, version: u32) !void {
    const on_gpu = try this.allocate(on_cpu.len, version);

    const staging_offset_bytes = this.staging_face_offset * @sizeOf(ChunkMesher.GreedyQuad);
    const size_bytes = on_gpu.face_count * @sizeOf(ChunkMesher.GreedyQuad);

    if (staging_offset_bytes + size_bytes >= staging_size) @panic("staging buffer overflow");
    @memcpy(this.mapping[this.staging_face_offset .. this.staging_face_offset + on_cpu.len], on_cpu);

    this.buffer_copy_src.appendAssumeCapacity(.{
        .buffer = this.staging,
        .offset = staging_offset_bytes,
        .size_or_whole = .{ .size = size_bytes },
    });

    const dst: gpu.Buffer.Region = .{
        .buffer = this.buffer,
        .offset = on_gpu.face_offset * @sizeOf(ChunkMesher.GreedyQuad),
        .size_or_whole = .{ .size = size_bytes },
    };
    this.buffer_copy_dst.appendAssumeCapacity(dst);

    this.memory_barriers.appendAssumeCapacity(.{ .buffer = .{
        .region = dst,
        .src_stage = .{ .transfer = true },
        .dst_stage = .{ .vertex_input = true },
        .src_access = .{ .transfer_write = true },
        .dst_access = .{ .vertex_read = true },
    } });

    const entry = this.loaded_meshes.getOrPutAssumeCapacity(pos);
    if (entry.found_existing) try this.queueFree(entry.value_ptr.*);
    entry.value_ptr.* = on_gpu;

    this.staging_face_offset += on_cpu.len;
}

pub fn upload(this: *ChunkMeshAllocator, device: gpu.Device, cmd_encoder: gpu.CommandEncoder) !void {
    std.debug.assert(this.buffer_copy_src.items.len == this.buffer_copy_dst.items.len);
    std.debug.assert(this.buffer_copy_src.items.len == this.memory_barriers.items.len);

    if (this.buffer_copy_src.items.len != 0) {
        for (this.buffer_copy_src.items, this.buffer_copy_dst.items) |src, dst| {
            cmd_encoder.cmdCopyBuffer(device, src, dst);
        }

        try cmd_encoder.cmdMemoryBarrier(this.device, this.memory_barriers.items, this.alloc);
        this.buffer_copy_src.clearRetainingCapacity();
        this.buffer_copy_dst.clearRetainingCapacity();
        this.memory_barriers.clearRetainingCapacity();
        this.staging_face_offset = 0;
    }
}

pub fn allocate(this: *ChunkMeshAllocator, quad_count: usize, version: u32) !ChunkMesher.GpuLoaded {
    const byte_count = quad_count * @sizeOf(ChunkMesher.GreedyQuad);

    var maybe_node = this.free_list.first;
    while (maybe_node) |node| : (maybe_node = node.next) {
        const free_region: *FreeRegion = @fieldParentPtr("node", node);
        if (free_region.size < byte_count) continue;

        const offset = free_region.offset;
        const quad_offset = offset / @sizeOf(ChunkMesher.GreedyQuad);

        if (free_region.size == byte_count) {
            this.free_list.remove(node);
            this.alloc.destroy(free_region);
        } else {
            free_region.offset += byte_count;
            free_region.size -= byte_count;
        }

        return .{
            .face_offset = @intCast(quad_offset),
            .face_count = @intCast(quad_count),
            .version = version,
        };
    }

    return error.ChunkMeshBufferFull;
}

pub fn free(mesh_alloc: *ChunkMeshAllocator, chunk: ChunkMesher.GpuLoaded) !void {
    const offset_bytes = chunk.face_offset * @sizeOf(ChunkMesher.GreedyQuad);
    const size_bytes = chunk.face_count * @sizeOf(ChunkMesher.GreedyQuad);

    var maybe_node = mesh_alloc.free_list.first;
    const maybe_closest_after: ?*std.DoublyLinkedList.Node = blk: while (maybe_node) |node| : (maybe_node = node.next) {
        const free_region: *FreeRegion = @fieldParentPtr("node", node);

        if (free_region.offset > offset_bytes) break :blk node;
    } else null;

    if (maybe_closest_after) |after| {
        if (after.prev) |before| {
            const before_region: *FreeRegion = @fieldParentPtr("node", before);

            if (before_region.offset + before_region.size == offset_bytes) {
                before_region.size += size_bytes;
                return;
            }
        }
    }

    const new = try mesh_alloc.alloc.create(FreeRegion);
    new.* = .{
        .offset = offset_bytes,
        .size = size_bytes,
    };

    if (maybe_closest_after) |closest| {
        mesh_alloc.free_list.insertBefore(closest, &new.node);
    } else {
        mesh_alloc.free_list.prepend(&new.node);
    }
}

pub fn queueFree(mesh_alloc: *ChunkMeshAllocator, chunk: ChunkMesher.GpuLoaded) !void {
    try mesh_alloc.free_queues[mesh_alloc.renderer_info.frame_slot()].append(mesh_alloc.alloc, chunk);
}

pub fn freeQueued(mesh_alloc: *ChunkMeshAllocator) !void {
    const queue = &mesh_alloc.free_queues[mesh_alloc.renderer_info.frame_slot()];

    for (queue.items) |mesh| {
        try mesh_alloc.free(mesh);
    }

    queue.clearRetainingCapacity();
}

pub fn queryBytesFree(mesh_alloc: *ChunkMeshAllocator) usize {
    var bytes_free: usize = 0;
    var maybe_node = mesh_alloc.free_list.first;
    while (maybe_node) |node| : (maybe_node = node.next) {
        const free_region: *FreeRegion = @fieldParentPtr("node", node);
        bytes_free += @intCast(free_region.size);
    }

    return bytes_free;
}
