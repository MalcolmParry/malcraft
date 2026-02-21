const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const ChunkMesher = @import("ChunkMesher.zig");
const Chunk = @import("Chunk.zig");
const RendererInfo = @import("Renderer.zig").Info;

const ChunkMeshAllocator = @This();
pub const buffer_size = 1024 * 1024 * 512;
pub const staging_size = 1024 * 1024 * 8;

comptime {
    std.debug.assert(buffer_size / @sizeOf(ChunkMesher.GreedyQuad) < std.math.maxInt(u32));
}

renderer_info: *const RendererInfo,
per_frame_in_flight: []PerFrameInFlight,
buffer: gpu.Buffer,
free_list: std.DoublyLinkedList,
alloc: std.mem.Allocator,
device: gpu.Device,
loaded_meshes: std.AutoArrayHashMapUnmanaged(Chunk.ChunkPos, ChunkMesher.GpuLoaded),
buffer_copy_src: std.ArrayList(gpu.Buffer.Region),
buffer_copy_dst: std.ArrayList(gpu.Buffer.Region),
write_barriers: std.ArrayList(gpu.CommandEncoder.MemoryBarrier),

const PerFrameInFlight = struct {
    staging: gpu.Buffer,
    mapping: []ChunkMesher.GreedyQuad,
    offset_bytes: gpu.Size,
    free_queue: std.ArrayList(ChunkMesher.GpuLoaded),
};

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

    this.per_frame_in_flight = try info.alloc.alloc(PerFrameInFlight, info.renderer_info.frames_in_flight);
    errdefer info.alloc.free(this.per_frame_in_flight);
    for (this.per_frame_in_flight) |*per_frame| {
        per_frame.staging = try .init(info.device, .{
            .alloc = info.alloc,
            .loc = .host,
            .size = staging_size,
            .usage = .{
                .src = true,
            },
        });

        const byte_mapping = try per_frame.staging.map(info.device);
        const face_mapping: [*]ChunkMesher.GreedyQuad = @ptrCast(@alignCast(byte_mapping));
        per_frame.mapping = face_mapping[0 .. staging_size / @sizeOf(ChunkMesher.GreedyQuad)];

        per_frame.offset_bytes = 0;
        per_frame.free_queue = .empty;
    }

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
    this.write_barriers = .empty;
    this.loaded_meshes = .empty;
}

pub fn deinit(this: *ChunkMeshAllocator) void {
    std.log.info("{} bytes free in chunk mesh buffer", .{this.queryBytesFree()});
    std.log.info("{} bytes used in chunk mesh buffer", .{buffer_size - this.queryBytesFree()});
    std.log.info("chunk mesh count on deinit {}", .{this.loaded_meshes.count()});

    this.write_barriers.deinit(this.alloc);
    this.buffer_copy_dst.deinit(this.alloc);
    this.buffer_copy_src.deinit(this.alloc);

    var maybe_node = this.free_list.first;
    while (maybe_node) |node| {
        const next = node.next;
        const free_region: *FreeRegion = @fieldParentPtr("node", node);
        this.alloc.destroy(free_region);
        maybe_node = next;
    }

    for (this.per_frame_in_flight) |*per_frame| {
        per_frame.free_queue.deinit(this.alloc);
        per_frame.staging.unmap(this.device);
        per_frame.staging.deinit(this.device, this.alloc);
    }
    this.alloc.free(this.per_frame_in_flight);

    this.buffer.deinit(this.device, this.alloc);
    this.loaded_meshes.deinit(this.alloc);
}

pub fn ensureCapacity(mesh_alloc: *ChunkMeshAllocator, count: usize) !void {
    try mesh_alloc.buffer_copy_src.ensureUnusedCapacity(mesh_alloc.alloc, count);
    try mesh_alloc.buffer_copy_dst.ensureUnusedCapacity(mesh_alloc.alloc, count);
    try mesh_alloc.write_barriers.ensureUnusedCapacity(mesh_alloc.alloc, count);
    try mesh_alloc.loaded_meshes.ensureUnusedCapacity(mesh_alloc.alloc, count);
}

pub fn writeChunkAssumeCapacity(this: *ChunkMeshAllocator, on_cpu: []const ChunkMesher.GreedyQuad, pos: Chunk.ChunkPos) !void {
    const on_gpu = try this.allocate(on_cpu.len);

    const per_frame = &this.per_frame_in_flight[this.renderer_info.frame_slot()];
    const offset_bytes = per_frame.offset_bytes;
    const offset_faces = offset_bytes / @sizeOf(ChunkMesher.GreedyQuad);
    const size_bytes = on_gpu.face_count * @sizeOf(ChunkMesher.GreedyQuad);

    if (offset_bytes + size_bytes > staging_size) @panic("staging buffer overflow");
    @memcpy(per_frame.mapping[offset_faces .. offset_faces + on_cpu.len], on_cpu);

    this.buffer_copy_src.appendAssumeCapacity(.{
        .buffer = per_frame.staging,
        .offset = offset_bytes,
        .size_or_whole = .{ .size = size_bytes },
    });

    const dst: gpu.Buffer.Region = .{
        .buffer = this.buffer,
        .offset = on_gpu.face_offset * @sizeOf(ChunkMesher.GreedyQuad),
        .size_or_whole = .{ .size = size_bytes },
    };
    this.buffer_copy_dst.appendAssumeCapacity(dst);

    this.write_barriers.appendAssumeCapacity(.{ .buffer = .{
        .region = dst,
        .src_stage = .{ .transfer = true },
        .dst_stage = .{ .vertex_input = true },
        .src_access = .{ .transfer_write = true },
        .dst_access = .{ .vertex_read = true },
    } });

    const entry = this.loaded_meshes.getOrPutAssumeCapacity(pos);
    if (entry.found_existing) try this.queueFree(entry.value_ptr.*);
    entry.value_ptr.* = on_gpu;

    per_frame.offset_bytes += on_cpu.len * @sizeOf(ChunkMesher.GreedyQuad);
}

pub fn upload(this: *ChunkMeshAllocator, cmd_encoder: gpu.CommandEncoder) !void {
    std.debug.assert(this.buffer_copy_src.items.len == this.buffer_copy_dst.items.len);
    std.debug.assert(this.buffer_copy_src.items.len == this.write_barriers.items.len);

    if (this.buffer_copy_src.items.len != 0) {
        for (this.buffer_copy_src.items, this.buffer_copy_dst.items) |src, dst| {
            cmd_encoder.cmdCopyBuffer(this.device, src, dst);
        }

        try cmd_encoder.cmdMemoryBarrier(this.device, this.write_barriers.items, this.alloc);
        this.buffer_copy_src.clearRetainingCapacity();
        this.buffer_copy_dst.clearRetainingCapacity();
        this.write_barriers.clearRetainingCapacity();

        const per_frame = &this.per_frame_in_flight[this.renderer_info.frame_slot()];
        per_frame.offset_bytes = 0;
    }
}

pub fn allocate(this: *ChunkMeshAllocator, quad_count: usize) !ChunkMesher.GpuLoaded {
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
        };
    }

    return error.ChunkMeshBufferFull;
}

pub fn free(mesh_alloc: *ChunkMeshAllocator, chunk: ChunkMesher.GpuLoaded) !void {
    const offset_bytes: gpu.Size = chunk.face_offset * @sizeOf(ChunkMesher.GreedyQuad);
    const size_bytes: gpu.Size = chunk.face_count * @sizeOf(ChunkMesher.GreedyQuad);

    var iter = mesh_alloc.free_list.first;
    const maybe_closest_after: ?*std.DoublyLinkedList.Node = blk: while (iter) |node| : (iter = node.next) {
        const free_region: *FreeRegion = @fieldParentPtr("node", node);

        if (free_region.offset > offset_bytes) break :blk node;
    } else null;

    var maybe_region: ?*FreeRegion = null;
    var new_offset = offset_bytes;
    var new_size = size_bytes;

    const prev_node = if (maybe_closest_after) |x| x.prev else mesh_alloc.free_list.last;
    if (prev_node) |before| {
        const before_region: *FreeRegion = @fieldParentPtr("node", before);

        if (before_region.offset + before_region.size == offset_bytes) {
            maybe_region = before_region;
            new_offset = before_region.offset;
            new_size = size_bytes + before_region.size;
        }
    }

    if (maybe_closest_after) |after| {
        const after_region: *FreeRegion = @fieldParentPtr("node", after);
        if (after_region.offset == new_offset + new_size) {
            if (maybe_region) |x| {
                mesh_alloc.free_list.remove(&x.node);
                mesh_alloc.alloc.destroy(x);
            }

            maybe_region = after_region;
            new_size += after_region.size;
        }
    }

    if (maybe_region) |region| {
        region.offset = new_offset;
        region.size = new_size;
        return;
    }

    const new = try mesh_alloc.alloc.create(FreeRegion);
    new.* = .{
        .offset = new_offset,
        .size = new_size,
    };

    if (maybe_closest_after) |closest| {
        mesh_alloc.free_list.insertBefore(closest, &new.node);
    } else {
        mesh_alloc.free_list.append(&new.node);
    }
}

pub fn queueFree(mesh_alloc: *ChunkMeshAllocator, chunk: ChunkMesher.GpuLoaded) !void {
    const queue = &mesh_alloc.per_frame_in_flight[mesh_alloc.renderer_info.frame_slot()].free_queue;
    try queue.append(mesh_alloc.alloc, chunk);
}

pub fn freeQueued(mesh_alloc: *ChunkMeshAllocator) !void {
    const queue = &mesh_alloc.per_frame_in_flight[mesh_alloc.renderer_info.frame_slot()].free_queue;

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
