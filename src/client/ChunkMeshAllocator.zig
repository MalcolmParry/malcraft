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
            .vertex = true,
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

pub fn writeChunkAssumeCapacity(this: *ChunkMeshAllocator, on_cpu: []const ChunkMesher.GreedyQuad, pos: Chunk.PackedPos) !void {
    const on_gpu = try this.allocate(on_cpu.len);

    const size_bytes = on_gpu.face_count * @sizeOf(ChunkMesher.GreedyQuad);
    const dst: gpu.Buffer.Region = .{
        .buffer = this.free_list_alloc.super_descs.items[0].buffer,
        .offset = on_gpu.face_offset * @sizeOf(ChunkMesher.GreedyQuad),
        .size = size_bytes,
    };

    try this.upload_man.submit(ChunkMesher.GreedyQuad, .{
        .data = on_cpu,
        .region = dst,
        .post_copy_barrier = .{
            .region = dst,
            .src_stage = .{ .transfer = true },
            .dst_stage = .{ .vertex_input = true },
            .src_access = .{ .transfer_write = true },
            .dst_access = .{ .vertex_read = true },
        },
    });

    const entry = this.loaded_meshes.getOrPutAssumeCapacity(pos);
    if (entry.found_existing) {
        try this.queueFree(entry.value_ptr.*);
        this.overwritten_meshes += 1;
    }

    entry.value_ptr.* = on_gpu;
}

pub fn allocate(this: *ChunkMeshAllocator, quad_count: usize) !ChunkMesher.GpuLoaded {
    const allocation = try this.free_list_alloc.alloc(this.device, @intCast(quad_count * @sizeOf(ChunkMesher.GreedyQuad)));
    std.debug.assert(allocation.super_slab == 0);

    return .{
        .face_offset = @divExact(allocation.offset, @sizeOf(ChunkMesher.GreedyQuad)),
        .face_count = @intCast(quad_count),
    };
}

pub fn free(mesh_alloc: *ChunkMeshAllocator, chunk: ChunkMesher.GpuLoaded) !void {
    mesh_alloc.free_list_alloc.free(.{
        .super_slab = 0,
        .offset = chunk.face_offset * @sizeOf(ChunkMesher.GreedyQuad),
        .size = chunk.face_count * @sizeOf(ChunkMesher.GreedyQuad),
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
