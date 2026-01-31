const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const gpu = mw.gpu;
const Chunk = @import("Chunk.zig");

const ChunkMesh = @This();

pub const OnGpu = struct {
    vertex_buffer: gpu.Buffer,
    face_count: usize,

    pub fn init(this: *OnGpu, device: gpu.Device, mesh: *ChunkMesh, alloc: std.mem.Allocator) !void {
        this.vertex_buffer = try .init(device, .{
            .alloc = alloc,
            .loc = .device,
            .usage = .{
                .vertex = true,
                .dst = true,
            },
            .size = mesh.per_face.len * @sizeOf(PerFace),
        });
        errdefer this.vertex_buffer.deinit(device, alloc);
        this.face_count = mesh.per_face.len;

        try device.setBufferRegions(
            &.{this.vertex_buffer.region()},
            &.{std.mem.sliceAsBytes(mesh.per_face)},
        );
    }

    pub fn deinit(this: *OnGpu, device: gpu.Device, alloc: std.mem.Allocator) void {
        this.vertex_buffer.deinit(device, alloc);
    }
};

pub const PerFace = packed struct(u32) {
    x: u5,
    y: u5,
    z: u5,
    face: Face,
    block_id: Chunk.BlockId,
    padding: u12 = undefined,
};

per_face: []PerFace,

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

pub fn build(mesh: *ChunkMesh, chunk: *Chunk, alloc: std.mem.Allocator) !void {
    var faces = try std.ArrayList(PerFace).initCapacity(alloc, (32 * 32 * 32 / 2) * 6);
    errdefer faces.deinit(alloc);

    var iter: Chunk.Iterator = .{};
    while (iter.next()) |pos| {
        const block = chunk.getBlock(pos);
        if (block == .air) continue;
        const ipos: Chunk.Pos = pos;

        for (Face.direction_table, 0..) |offset, i| {
            const adjacent = ipos + offset;
            if (chunk.isOpaqueSafe(adjacent)) continue;
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

    mesh.per_face = try faces.toOwnedSlice(alloc);
    std.log.info("faces {}", .{mesh.per_face.len});
}

pub fn deinit(mesh: *ChunkMesh, alloc: std.mem.Allocator) void {
    alloc.free(mesh.per_face);
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
