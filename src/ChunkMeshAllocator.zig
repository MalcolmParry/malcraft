const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const ChunkMesher = @import("ChunkMesher.zig");

const ChunkMeshAllocator = @This();
const buffer_size = 1024 * 1024 * 512;
const staging_size = 1024 * 1024;

staging: gpu.Buffer,
buffer: gpu.Buffer,
free_list: std.DoublyLinkedList,
alloc: std.mem.Allocator,
device: gpu.Device,

const FreeRegion = struct {
    offset: gpu.Size,
    size: gpu.Size,
    node: std.DoublyLinkedList.Node = .{},
};

pub fn init(this: *ChunkMeshAllocator, device: gpu.Device, alloc: std.mem.Allocator) !void {
    this.staging = try .init(device, .{
        .alloc = alloc,
        .loc = .host,
        .size = staging_size,
        .usage = .{
            .src = true,
        },
    });
    errdefer this.staging.deinit(device, alloc);

    this.buffer = try .init(device, .{
        .alloc = alloc,
        .loc = .device,
        .size = buffer_size,
        .usage = .{
            .vertex = true,
            .dst = true,
        },
    });
    errdefer this.buffer.deinit(device, alloc);

    const free_region = try alloc.create(FreeRegion);
    errdefer alloc.destroy(free_region);
    free_region.* = .{
        .offset = 0,
        .size = buffer_size,
    };

    this.free_list = .{};
    this.free_list.append(&free_region.node);

    this.alloc = alloc;
    this.device = device;
}

pub fn deinit(this: *ChunkMeshAllocator) void {
    var maybe_node = this.free_list.first;
    while (maybe_node) |node| : (maybe_node = node.next) {
        const free_region: *FreeRegion = @fieldParentPtr("node", node);
        this.alloc.destroy(free_region);
    }

    this.buffer.deinit(this.device, this.alloc);
    this.staging.deinit(this.device, this.alloc);
}

pub const AllocateResult = struct {
    buffer_region: gpu.Buffer.Region,
    gpu_loaded_mesh: ChunkMesher.GpuLoaded,
};

pub fn writeBuffers(this: *ChunkMeshAllocator, regions: []gpu.Buffer.Region, data: []const []const u8) !void {
    // temporary
    try this.device.setBufferRegions(regions, data);
}

pub fn allocate(this: *ChunkMeshAllocator, face_count: usize) !AllocateResult {
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
            .buffer_region = .{
                .buffer = this.buffer,
                .offset = offset,
                .size_or_whole = .{ .size = byte_count },
            },
            .gpu_loaded_mesh = .{
                .face_offset = @intCast(face_offset),
                .face_count = @intCast(face_count),
            },
        };
    }

    return error.ChunkMeshBufferFull;
}
