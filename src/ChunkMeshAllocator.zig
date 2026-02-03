const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const ChunkMesher = @import("ChunkMesher.zig");

const ChunkMeshAllocator = @This();
const buffer_size = 1024 * 1024 * 512;
const staging_size = 1024 * 1024;

staging: gpu.Buffer,
mapping: []ChunkMesher.PerFace,
buffer: gpu.Buffer,
free_list: std.DoublyLinkedList,
alloc: std.mem.Allocator,
device: gpu.Device,
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
    const face_mapping: [*]ChunkMesher.PerFace = @ptrCast(@alignCast(byte_mapping));
    this.mapping = face_mapping[0 .. staging_size / @sizeOf(ChunkMesher.PerFace)];

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
}

pub fn writeChunks(this: *ChunkMeshAllocator, on_gpu: []ChunkMesher.GpuLoaded, on_cpu: []const []const ChunkMesher.PerFace) !void {
    std.debug.assert(on_gpu.len == on_cpu.len);

    try this.buffer_copy_src.ensureUnusedCapacity(this.alloc, on_gpu.len);
    try this.buffer_copy_dst.ensureUnusedCapacity(this.alloc, on_gpu.len);
    try this.memory_barriers.ensureUnusedCapacity(this.alloc, on_gpu.len);

    var face: usize = 0;
    for (on_cpu, on_gpu) |x, y| {
        @memcpy(this.mapping[face .. face + x.len], x);
        const size_bytes = y.face_count * @sizeOf(ChunkMesher.PerFace);

        this.buffer_copy_src.appendAssumeCapacity(.{
            .buffer = this.staging,
            .offset = face * @sizeOf(ChunkMesher.PerFace),
            .size_or_whole = .{ .size = size_bytes },
        });

        const dst: gpu.Buffer.Region = .{
            .buffer = this.buffer,
            .offset = y.face_offset * @sizeOf(ChunkMesher.PerFace),
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

        face += x.len;
    }
}

pub fn upload(this: *ChunkMeshAllocator, device: gpu.Device, cmd_encoder: gpu.CommandEncoder) !void {
    std.debug.assert(this.buffer_copy_src.items.len == this.buffer_copy_dst.items.len);

    for (this.buffer_copy_src.items, this.buffer_copy_dst.items) |src, dst| {
        cmd_encoder.cmdCopyBuffer(device, src, dst);
    }

    if (this.buffer_copy_src.items.len != 0) {
        try cmd_encoder.cmdMemoryBarrier(this.device, this.memory_barriers.items, this.alloc);
    }

    this.buffer_copy_src.clearRetainingCapacity();
    this.buffer_copy_dst.clearRetainingCapacity();
    this.memory_barriers.clearRetainingCapacity();
}

pub fn allocate(this: *ChunkMeshAllocator, face_count: usize) !ChunkMesher.GpuLoaded {
    const byte_count = face_count * @sizeOf(ChunkMesher.PerFace);

    var maybe_node = this.free_list.first;
    while (maybe_node) |node| : (maybe_node = node.next) {
        const free_region: *FreeRegion = @fieldParentPtr("node", node);
        if (free_region.size < byte_count) continue;

        const offset = free_region.offset;
        const face_offset = offset / @sizeOf(ChunkMesher.PerFace);

        if (free_region.size == byte_count) {
            this.free_list.remove(node);
        } else {
            free_region.offset += byte_count;
            free_region.size -= byte_count;
        }

        return .{
            .face_offset = @intCast(face_offset),
            .face_count = @intCast(face_count),
        };
    }

    return error.ChunkMeshBufferFull;
}
