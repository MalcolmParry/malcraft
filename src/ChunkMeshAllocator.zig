const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const ChunkMesher = @import("ChunkMesher.zig");
const Chunk = @import("Chunk.zig");

const ChunkMeshAllocator = @This();
pub const buffer_size = 1024 * 1024 * 512;
pub const staging_size = 1024 * 1024 * 32;

comptime {
    std.debug.assert(buffer_size / @sizeOf(ChunkMesher.GreedyQuad) < std.math.maxInt(u32));
}

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
};

pub fn init(this: *ChunkMeshAllocator, info: InitInfo) !void {
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
    this.memory_barriers.deinit(this.alloc);
    this.buffer_copy_dst.deinit(this.alloc);
    this.buffer_copy_src.deinit(this.alloc);

    var maybe_node = this.free_list.first;
    while (maybe_node) |node| : (maybe_node = node.next) {
        const free_region: *FreeRegion = @fieldParentPtr("node", node);
        this.alloc.destroy(free_region);
    }

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

pub fn writeChunkAssumeCapacity(this: *ChunkMeshAllocator, on_cpu: []const ChunkMesher.GreedyQuad, pos: Chunk.ChunkPos) !void {
    const on_gpu = try this.allocate(on_cpu.len);

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
    this.loaded_meshes.putAssumeCapacityNoClobber(pos, on_gpu);

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
