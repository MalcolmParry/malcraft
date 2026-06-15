const std = @import("std");
const options = @import("options");
const mw = @import("mwengine");
const math = mw.math;
const znet = @import("znet");
const NetworkManager = @import("../common/NetworkManager.zig");
const protocol = @import("../common/protocol.zig");
const ServerMsgId = protocol.ServerMsgId;
const Renderer = @import("Renderer.zig");
const Camera = @import("Camera.zig");
const block = @import("../common/block.zig");
const Chunk = @import("../common/Chunk.zig");
const region = @import("../common/region.zig");
const World = @import("../common/World.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const Aabb = @import("../utils/Aabb.zig");
const App = @This();

const zstd = @cImport({
    @cInclude("zstd.h");
});

net_man: NetworkManager,
server: NetworkManager.PeerData,

should_close: bool = false,
window: *mw.Window,
renderer: Renderer,

frame_timer: std.time.Timer,
camera: Camera = .default,
last_region_pos: region.Pos = @splat(0),
mouse_lock: bool = true,
generate_chunks: bool = true,
last_cursor: math.Vec2,

world: World,
chunk_mesher: ChunkMesher,

pub fn init(app: *App, alloc: std.mem.Allocator) !void {
    try znet.init();
    errdefer znet.deinit();

    const window: *mw.Window = try .init(alloc, "malcraft", .{ 100, 100 });
    errdefer window.deinit();
    try window.setCursorMode(.disabled);

    app.* = .{
        .net_man = undefined,
        .server = undefined,

        .window = window,
        .renderer = undefined,

        .frame_timer = try .start(),
        .last_cursor = window.getCursorPos(),

        .world = .{},
        .chunk_mesher = undefined,
    };

    try app.net_man.init(alloc, .{
        .addr = try .init(.{
            .ip = .any,
            .port = .any,
        }),
        .channel_limit = .{ .count = 1 },
        .peer_limit = 1,
        .outgoing_bandwidth = .unlimited,
        .incoming_bandwidth = .unlimited,
    });
    errdefer app.net_man.deinit();

    var semaphore: std.Thread.Semaphore = .{};
    try app.net_man.pushCommand(.{ .connect = .{
        .config = .{
            .addr = try .init(.{
                .ip = .{ .ipv4 = "localhost" },
                .port = .{ .uint = 5000 },
            }),
            .channel_limit = .{ .count = std.enums.values(protocol.Channel).len },
            .data = 0,
        },
        .peer = &app.server,
        .semaphore = &semaphore,
    } });
    semaphore.wait();

    try app.renderer.init(.{
        .alloc = alloc,
        .window = app.window,
    });
    errdefer app.renderer.deinit(alloc);

    try app.chunk_mesher.init(.{
        .alloc = alloc,
        .world = &app.world,
        .mesh_alloc = &app.renderer.chunk_mesh_alloc,
    });
    errdefer app.chunk_mesher.deinit();
}

pub fn deinit(app: *App, alloc: std.mem.Allocator) void {
    app.renderer.deinit(alloc);
    app.window.deinit();

    app.chunk_mesher.deinit();
    app.world.deinit(alloc);

    app.net_man.deinit();
    znet.deinit();
}

pub fn tick(app: *App, alloc: std.mem.Allocator) !void {
    const dt_ns = app.frame_timer.lap();
    const dt = @as(f32, @floatFromInt(dt_ns)) / std.time.ns_per_s;

    try app.chunk_mesher.meshMany();

    app.window.update();
    if (app.window.shouldClose()) app.should_close = true;

    const renderer_input = try app.handleInput(alloc, dt);

    const region_pos = @as(Chunk.Pos, @intFromFloat(app.camera.pos)) / Chunk.size / region.size;
    if (app.generate_chunks)
        try app.maybeUpdateChunkCursor(alloc, region_pos);

    while (try app.net_man.popEvent()) |event| {
        try app.handleNetworkEvent(alloc, event);
    }

    try app.renderer.render(.{
        .dt_ns = dt_ns,
        .camera = app.camera,
        .input = renderer_input,
        .viewport = app.window.getFramebufferSize(),
        .show_crosshair = app.mouse_lock,
        .generating_chunks = app.generate_chunks,
    }, alloc);
}

pub fn chunkInRange(app: *const App, pos: Chunk.Pos) bool {
    const region_pos = pos / region.size;
    return app.regionInRange(region_pos);
}

pub fn regionInRange(app: *const App, pos: Chunk.Pos) bool {
    const render_radius = @max(options.render_radius / region.len, 1);
    const render_height = @max(options.render_height / region.len, 1);

    const rel = pos - app.last_region_pos;
    if (@abs(rel[0]) > render_radius) return false;
    if (@abs(rel[1]) > render_radius) return false;
    if (@abs(rel[2]) > render_height) return false;
    return true;
}

fn maybeUpdateChunkCursor(app: *App, alloc: std.mem.Allocator, region_pos: region.Pos) !void {
    if (@reduce(.And, app.last_region_pos == region_pos)) return;

    var buffer: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try writer.writeInt(u8, @intFromEnum(protocol.ClientMsgId.update_chunk_cursor), .little);
    try writer.writeStruct(region.PackedPos.pack(region_pos), .little);

    const channel: protocol.Channel = .control;
    const packet = try znet.Packet.init(writer.buffered(), channel.toInt(), channel.getFlags());
    try app.net_man.send(app.server.ref, packet);

    const old_aabb = app.loadedChunkAabb();
    app.last_region_pos = region_pos;
    const new_aabb = app.loadedChunkAabb();

    var aabb_buffer: [6]Aabb = undefined;
    const old_regions = old_aabb.subtract(new_aabb, &aabb_buffer);

    for (old_regions) |aabb| {
        const min = aabb.min * region.size;
        const max = aabb.max * region.size;

        var x = min[0];
        while (x < max[0]) : (x += 1) {
            var y = min[1];
            while (y < max[1]) : (y += 1) {
                var z = min[2];
                while (z < max[2]) : (z += 1) {
                    const pos: Chunk.Pos = .{ x, y, z };
                    const packed_pos: Chunk.PackedPos = .pack(pos);

                    app.world.removeChunk(alloc, packed_pos);
                    if (app.renderer.chunk_mesh_alloc.loaded_meshes.fetchSwapRemove(packed_pos)) |kv| {
                        try app.renderer.chunk_mesh_alloc.queueFree(kv.value);
                    }
                }
            }
        }
    }
}

fn loadedChunkAabb(app: *const App) Aabb {
    const bounds: Chunk.Pos = .{
        @max(options.render_radius / region.len, 1),
        @max(options.render_radius / region.len, 1),
        @max(options.render_height / region.len, 1),
    };

    return .{
        .min = app.last_region_pos - bounds,
        .max = app.last_region_pos + bounds,
    };
}

fn handleInput(app: *App, alloc: std.mem.Allocator, dt: f32) !Renderer.FrameData.Input {
    var break_block: bool = false;
    var renderer_input: Renderer.FrameData.Input = .{};
    while (app.window.popEvent()) |event| switch (event) {
        .resize => |_| app.renderer.dirty_swapchain = true,
        .key_down => |key| {
            switch (key) {
                .escape => app.should_close = true,
                .f => renderer_input.wireframe = true,
                .g => app.generate_chunks = !app.generate_chunks,
                .o => app.camera = .default,
                .left_alt => {
                    app.mouse_lock = !app.mouse_lock;
                    try app.window.setCursorMode(if (app.mouse_lock) .disabled else .normal);
                    app.last_cursor = app.window.getCursorPos();
                },
                else => {},
            }
        },
        .mouse_down => |button| {
            switch (button) {
                .left => break_block = true,
                .right => blk: {
                    const origin = app.camera.pos;
                    const q = math.quatFromEuler(app.camera.euler);
                    const dir = math.normalize(math.quatMulVec(q, math.dir_forward));

                    const ray_cast = app.world.rayCast(origin, dir);
                    const pos: block.Pos = switch (ray_cast) {
                        .no_hit => break :blk,
                        .inside => @intFromFloat(@floor(origin)),
                        .hit => |x| x.pos + x.face.dir(),
                    };

                    app.world.setBlock(alloc, pos, .grass) catch |err| switch (err) {
                        error.ChunkNotPresent => break :blk,
                        else => return err,
                    };
                    try app.chunk_mesher.addRequestWithCollateral(pos);
                },
                else => {},
            }
        },
        else => {},
    };

    if (break_block or app.window.isMouseDown(.five)) blk: {
        const origin = app.camera.pos;
        const q = math.quatFromEuler(app.camera.euler);
        const dir = math.normalize(math.quatMulVec(q, math.dir_forward));

        const ray_cast = app.world.rayCast(origin, dir);
        const pos: block.Pos = switch (ray_cast) {
            .no_hit => break :blk,
            .inside => @intFromFloat(@floor(origin)),
            .hit => |x| x.pos,
        };

        try app.world.setBlock(alloc, pos, .air);
        try app.chunk_mesher.addRequestWithCollateral(pos);
    }

    {
        var move_vector: math.Vec3 = @splat(0);

        if (app.window.isKeyDown(.w))
            move_vector += math.dir_forward;
        if (app.window.isKeyDown(.s))
            move_vector -= math.dir_forward;
        if (app.window.isKeyDown(.a))
            move_vector -= math.dir_right;
        if (app.window.isKeyDown(.d))
            move_vector += math.dir_right;
        if (app.window.isKeyDown(.e))
            move_vector += math.dir_up;
        if (app.window.isKeyDown(.q))
            move_vector -= math.dir_up;

        var speed: f32 = 40;
        if (app.window.isKeyDown(.left_shift))
            speed = 200;
        if (app.window.isKeyDown(.left_control))
            speed = 1000;

        if (!math.eql(move_vector, math.splat3(f32, 0))) {
            const q = math.quatFromEuler(app.camera.euler);
            move_vector = math.quatMulVec(q, move_vector);
            move_vector = math.normalize(move_vector);
            move_vector *= @splat(dt * speed);
            app.camera.pos += move_vector;
        }
    }

    if (app.mouse_lock) {
        const cursor = app.window.getCursorPos();
        var moved = cursor - app.last_cursor;
        app.last_cursor = cursor;

        if (!math.eql(moved, math.splat2(f32, 0))) {
            moved *= @splat(math.rad(0.18));
            app.camera.euler += math.shuffle(moved, &.{ .zero, .y, .x });
            app.camera.euler[1] = std.math.clamp(app.camera.euler[1], math.rad(-90.0), math.rad(90.0));
        }
    }

    return renderer_input;
}

fn handleNetworkEvent(app: *App, alloc: std.mem.Allocator, any_event: NetworkManager.Event) !void {
    switch (any_event) {
        .connect => |peer| {
            std.log.info("connected to server at {f}", .{peer.address});

            const region_pos = @as(Chunk.Pos, @intFromFloat(app.camera.pos)) / Chunk.size / region.size;
            try app.maybeUpdateChunkCursor(alloc, region_pos);
        },
        .disconnect => |_| {
            std.log.info("disconnected from server at {f}", .{app.server.address});

            return error.ServerClosed;
        },
        .receive => |event| {
            defer event.packet.deinit();
            var reader = event.packet.reader();
            const msg_id = try ServerMsgId.decode(&reader);

            switch (msg_id) {
                .init => {
                    const player_id = try reader.takeInt(u16, .little);

                    std.log.info("player_id: {}", .{player_id});
                },
                .uniform_chunk_batch => {
                    var timer: std.time.Timer = try .start();
                    const count = try reader.takeInt(u16, .little);
                    try app.world.uniform_chunks.ensureUnusedCapacity(alloc, count);
                    try app.chunk_mesher.queue.ensureUnusedCapacity(alloc, count * 2);

                    for (0..count) |_| {
                        const pos = try reader.takeStruct(Chunk.PackedPos, .little);
                        const kind_i = try reader.takeInt(u8, .little);
                        const kind = std.enums.fromInt(block.Kind, kind_i) orelse return error.BadMessage;

                        app.world.uniform_chunks.putAssumeCapacity(pos, kind);
                        try app.chunk_mesher.addRequestWithFullCollateral(pos.vec());
                    }

                    std.log.info("processed packet: {d: >4} uniform chunks: {d: >4}μs", .{ count, timer.read() / 1000 });
                },
                .compressed_chunk_batch => {
                    const count = try reader.takeInt(u16, .little);
                    try app.world.one_to_one_chunks.ensureUnusedCapacity(alloc, count);
                    try app.world.u2_palette_chunks.ensureUnusedCapacity(alloc, count);
                    try app.chunk_mesher.queue.ensureUnusedCapacity(alloc, count * 2);

                    for (0..count) |_| {
                        const pos = try reader.takeStruct(Chunk.PackedPos, .little);
                        const storage_type_i = try reader.takeInt(u8, .little);
                        const storage_type = std.enums.fromInt(protocol.ChunkStorageType, storage_type_i) orelse return error.BadMessage;

                        const compressed_size: usize = try reader.takeInt(u16, .little);
                        if (reader.bufferedLen() < compressed_size) return error.BadMessage;

                        const compressed_bytes = reader.buffered()[0..compressed_size];
                        reader.toss(compressed_size);

                        if (!app.chunkInRange(pos.vec()))
                            continue;

                        switch (storage_type) {
                            .u2_palette => {
                                const chunk = try alloc.create(Chunk.U2Palette);
                                errdefer alloc.destroy(chunk);

                                const result = zstd.ZSTD_decompress(std.mem.asBytes(chunk), @sizeOf(Chunk.U2Palette), compressed_bytes.ptr, compressed_size);
                                if (zstd.ZSTD_isError(result) != 0) return error.ZstdDecompressFailed;
                                if (result != @sizeOf(Chunk.U2Palette)) return error.BadMessage;

                                app.world.removeChunk(alloc, pos);
                                app.world.u2_palette_chunks.putAssumeCapacity(pos, chunk);
                            },
                            .u4 => {
                                const one_to_one = try alloc.create(Chunk.OneToOne);
                                errdefer alloc.destroy(one_to_one);

                                const result = zstd.ZSTD_decompress(std.mem.asBytes(one_to_one), @sizeOf(Chunk.OneToOne), compressed_bytes.ptr, compressed_size);
                                if (zstd.ZSTD_isError(result) != 0) return error.ZstdDecompressFailed;
                                if (result != @sizeOf(Chunk.OneToOne)) return error.BadMessage;

                                app.world.removeChunk(alloc, pos);
                                app.world.one_to_one_chunks.putAssumeCapacity(pos, one_to_one);
                            },
                        }

                        try app.chunk_mesher.addRequestWithFullCollateral(pos.vec());
                    }

                    std.log.info("processed packet: {d: >3} compressed chunks: {Bi:.2}", .{ count, event.packet.dataSlice().len });
                },
            }
        },
    }
}
