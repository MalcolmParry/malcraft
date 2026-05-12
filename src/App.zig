const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const znet = @import("znet");
const net = @import("net.zig");
const Renderer = @import("Renderer.zig");
const Camera = @import("Camera.zig");
const block = @import("block.zig");
const Chunk = @import("Chunk.zig");
const World = @import("World.zig");
const ChunkMesher = @import("ChunkMesher.zig");
const App = @This();

host: znet.Host,
server: znet.Peer,

should_close: bool = false,
window: *mw.Window,
renderer: Renderer,

frame_timer: std.time.Timer,
camera: Camera = .default,
mouse_lock: bool = true,
last_cursor: math.Vec2,

world: World,
chunk_mesher: ChunkMesher,

pub fn init(app: *App, alloc: std.mem.Allocator) !void {
    try znet.init();
    errdefer znet.deinit();

    const host = try znet.Host.init(.{
        .addr = null,
        .peer_limit = 1,
        .channel_limit = .{ .count = 1 },
        .incoming_bandwidth = .unlimited,
        .outgoing_bandwidth = .unlimited,
    });
    errdefer host.deinit();

    const peer = try host.connect(.{
        .addr = try .init(.{
            .ip = .{ .ipv4 = "localhost" },
            .port = .{ .uint = 5000 },
        }),
        .channel_limit = .{ .count = 1 },
        .data = 0,
    });

    const window: *mw.Window = try .init(alloc, "malcraft", .{ 100, 100 });
    errdefer window.deinit();
    try window.setCursorMode(.disabled);

    app.* = .{
        .host = host,
        .server = peer,

        .window = window,
        .renderer = undefined,

        .frame_timer = try .start(),
        .last_cursor = window.getCursorPos(),

        .world = .{},
        .chunk_mesher = undefined,
    };

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

    app.server.disconnect(0);
    app.host.flush();
    app.host.deinit();
    znet.deinit();
}

pub fn tick(app: *App, alloc: std.mem.Allocator) !void {
    const dt_ns = app.frame_timer.lap();
    const dt = @as(f32, @floatFromInt(dt_ns)) / std.time.ns_per_s;

    try app.chunk_mesher.meshMany();

    app.window.update();
    if (app.window.shouldClose()) app.should_close = true;

    const renderer_input = try app.handleInput(alloc, dt);

    while (try app.host.service(0)) |any_event| {
        try app.handleNetworkEvent(alloc, any_event);
    }

    try app.renderer.render(.{
        .dt_ns = dt_ns,
        .camera = app.camera,
        .input = renderer_input,
        .viewport = app.window.getFramebufferSize(),
        .show_crosshair = app.mouse_lock,
    }, alloc);
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

                    app.world.setBlock(alloc, pos, .stone) catch |err| switch (err) {
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
        }
    }

    return renderer_input;
}

fn handleNetworkEvent(app: *App, alloc: std.mem.Allocator, any_event: znet.Event) !void {
    switch (any_event) {
        .connect => |event| {
            std.log.info("connected to server at {f}", .{event.peer.address()});
        },
        .disconnect => |_| {
            std.log.info("disconnected from server at {f}", .{app.server.address()});

            return error.ServerClosed;
        },
        .receive => |event| {
            defer event.packet.deinit();
            var reader = event.packet.reader();
            const kind = try net.server_message.Kind.decode(&reader);

            switch (kind) {
                .init => {
                    const msg = try net.server_message.Init.decode(&reader);

                    std.log.info("playerid: {}", .{msg.player_id});
                },
                .chunk_data => {
                    const msg = try net.server_message.ChunkData.decode(alloc, &reader);

                    app.world.removeChunk(alloc, msg.pos);
                    try app.world.placeChunk(alloc, msg.pos, msg.chunk);

                    const pos = msg.pos.vec();
                    try app.chunk_mesher.addRequest(.pack(pos));
                    try app.chunk_mesher.addRequest(.pack(pos + Chunk.Pos{ 1, 0, 0 }));
                    try app.chunk_mesher.addRequest(.pack(pos + Chunk.Pos{ -1, 0, 0 }));
                    try app.chunk_mesher.addRequest(.pack(pos + Chunk.Pos{ 0, 1, 0 }));
                    try app.chunk_mesher.addRequest(.pack(pos + Chunk.Pos{ 0, -1, 0 }));
                    try app.chunk_mesher.addRequest(.pack(pos + Chunk.Pos{ 0, 0, 1 }));
                    try app.chunk_mesher.addRequest(.pack(pos + Chunk.Pos{ 0, 0, -1 }));
                },
            }
        },
    }
}
