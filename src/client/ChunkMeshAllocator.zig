const std = @import("std");
const mw = @import("mwengine");
const gpu = mw.gpu;
const math = mw.math;
const ChunkMesher = @import("ChunkMesher.zig");
const Chunk = @import("../common/Chunk.zig");
const RendererInfo = @import("Renderer.zig").Info;

const ChunkMeshAllocator = @This();
pub const buffer_size = 1024 * 1024 * 512;

comptime {
    std.debug.assert(buffer_size / @sizeOf(ChunkMesher.GreedyQuad) < std.math.maxInt(u32));
}

renderer_info: *const RendererInfo,
free_list_alloc: gpu.FreeListAllocator(.{
    .super_slab_size = buffer_size,
    .allow_multiple_super_slabs = false,
    .stats = true,
}),
upload_man: *gpu.UploadManager,
free_queues: []std.ArrayList(ChunkMesher.GpuLoaded),
alloc: std.mem.Allocator,
device: gpu.Device,
loaded_meshes: std.AutoArrayHashMapUnmanaged(Chunk.PackedPos, ChunkMesher.GpuLoaded),

overwritten_meshes: u64,

const FreeRegion = struct {
    offset: gpu.Size,
    size: gpu.Size,
    node: std.DoublyLinkedList.Node = .{},
};

const InitInfo = struct {
    device: gpu.Device,
    alloc: std.mem.Allocator,
    renderer_info: *const RendererInfo,
    upload_man: *gpu.UploadManager,
};

pub fn init(this: *ChunkMeshAllocator, info: InitInfo) !void {
    this.renderer_info = info.renderer_info;

    this.free_queues = try info.alloc.alloc(std.ArrayList(ChunkMesher.GpuLoaded), info.renderer_info.frames_in_flight);
    errdefer info.alloc.free(this.free_queues);
    @memset(this.free_queues, .empty);

    this.free_list_alloc = .{
        .gpa = info.alloc,
        .buffer_loc = .device,
        .buffer_usage = .{
            .dst = true,
            .device_address = true,
        },
    };
    errdefer this.free_list_alloc.deinit(info.device);

    this.alloc = info.alloc;
    this.device = info.device;
    this.loaded_meshes = .empty;
    this.upload_man = info.upload_man;
    this.overwritten_meshes = 0;
}

pub fn deinit(this: *ChunkMeshAllocator) void {
    for (this.free_queues) |*queue| {
        queue.deinit(this.alloc);
    }
    this.alloc.free(this.free_queues);

    this.loaded_meshes.deinit(this.alloc);
    this.free_list_alloc.deinit(this.device);
}

pub fn ensureCapacity(mesh_alloc: *ChunkMeshAllocator, count: usize) !void {
    try mesh_alloc.loaded_meshes.ensureUnusedCapacity(mesh_alloc.alloc, count);
}

pub fn writeChunkAssumeCapacity(this: *ChunkMeshAllocator, opaque_quads: []const ChunkMesher.GreedyQuad, water_faces: []const ChunkMesher.WaterFace, pos: Chunk.PackedPos) !void {
    const opaque_byte_count: u32 = @intCast(opaque_quads.len * @sizeOf(ChunkMesher.GreedyQuad));
    const water_byte_count: u32 = @intCast(water_faces.len * @sizeOf(ChunkMesher.WaterFace));
    const byte_count: u32 = opaque_byte_count + water_byte_count;

    const allocation = try this.free_list_alloc.alloc(this.device, byte_count, .max(.of(ChunkMesher.GreedyQuad), .of(ChunkMesher.WaterFace)));
    std.debug.assert(allocation.super_slab == 0);

    if (opaque_quads.len != 0) {
        const opaque_dst: gpu.Buffer.Region = .{
            .buffer = this.free_list_alloc.super_descs.items[0].buffer,
            .offset = allocation.offset,
            .size = opaque_byte_count,
        };

        try this.upload_man.submit(ChunkMesher.GreedyQuad, .{
            .data = opaque_quads,
            .region = opaque_dst,
            .post_copy_barrier = .{
                .region = opaque_dst,
                .src_stage = .{ .transfer = true },
                .dst_stage = .{ .vertex_shader = true },
                .src_access = .{ .transfer_write = true },
                .dst_access = .{ .shader_storage_read = true },
            },
        });
    }

    if (water_faces.len != 0) {
        const water_dst: gpu.Buffer.Region = .{
            .buffer = this.free_list_alloc.super_descs.items[0].buffer,
            .offset = allocation.offset + opaque_byte_count,
            .size = water_byte_count,
        };

        try this.upload_man.submit(ChunkMesher.WaterFace, .{
            .data = water_faces,
            .region = water_dst,
            .post_copy_barrier = .{
                .region = water_dst,
                .src_stage = .{ .transfer = true },
                .dst_stage = .{ .vertex_shader = true },
                .src_access = .{ .transfer_write = true },
                .dst_access = .{ .shader_storage_read = true },
            },
        });
    }

    const entry = this.loaded_meshes.getOrPutAssumeCapacity(pos);
    if (entry.found_existing) {
        try this.queueFree(entry.value_ptr.*);
        this.overwritten_meshes += 1;
    }

    entry.value_ptr.* = .{
        .buffer_offset = allocation.offset,
        .opaque_count = @intCast(opaque_quads.len),
        .water_count = @intCast(water_faces.len),
    };
}

pub fn free(mesh_alloc: *ChunkMeshAllocator, chunk: ChunkMesher.GpuLoaded) !void {
    mesh_alloc.free_list_alloc.free(.{
        .super_slab = 0,
        .offset = chunk.buffer_offset,
        .size = chunk.opaque_count * @sizeOf(ChunkMesher.GreedyQuad) + chunk.water_count * @sizeOf(ChunkMesher.WaterFace),
        .alignment = .max(.of(ChunkMesher.GreedyQuad), .of(ChunkMesher.WaterFace)),
    });
}

pub fn queueFree(mesh_alloc: *ChunkMeshAllocator, chunk: ChunkMesher.GpuLoaded) !void {
    const queue = &mesh_alloc.free_queues[mesh_alloc.renderer_info.frame_slot()];
    try queue.append(mesh_alloc.alloc, chunk);
}

pub fn freeQueued(mesh_alloc: *ChunkMeshAllocator) !void {
    const queue = &mesh_alloc.free_queues[mesh_alloc.renderer_info.frame_slot()];

    for (queue.items) |mesh| {
        try mesh_alloc.free(mesh);
    }

    queue.clearRetainingCapacity();
}
