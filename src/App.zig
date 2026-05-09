const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;
const znet = @import("znet");
const net = @import("net.zig");
const Renderer = @import("Renderer.zig");
const Camera = @import("Camera.zig");
const App = @This();

host: znet.Host,
server: znet.Peer,

should_close: bool = false,
event_queue: mw.EventQueue,
window: mw.Window,
renderer: Renderer,

frame_timer: std.time.Timer,
camera: Camera = .default,
mouse_lock: bool = true,
last_cursor: math.Vec2,

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

    var event_queue: mw.EventQueue = try .init(alloc);
    errdefer event_queue.deinit();

    var window: mw.Window = try .init(alloc, "malcraft", .{ 100, 100 }, &app.event_queue);
    errdefer window.deinit();
    try window.setCursorMode(.disabled);

    app.* = .{
        .host = host,
        .server = peer,

        .event_queue = event_queue,
        .window = window,
        .renderer = undefined,

        .frame_timer = try .start(),
        .last_cursor = window.getCursorPos(),
    };

    try app.renderer.init(.{
        .alloc = alloc,
        .window = &app.window,
    });
    errdefer app.renderer.deinit(alloc);
}

pub fn deinit(app: *App, alloc: std.mem.Allocator) void {
    app.renderer.deinit(alloc);
    app.window.deinit();
    app.event_queue.deinit();

    app.server.disconnect(0);
    app.host.flush();
    app.host.deinit();
    znet.deinit();
}

pub fn tick(app: *App, alloc: std.mem.Allocator) !void {
    const dt_ns = app.frame_timer.lap();
    const dt = @as(f32, @floatFromInt(dt_ns)) / std.time.ns_per_s;

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
    }, alloc);
}

fn handleInput(app: *App, alloc: std.mem.Allocator, dt: f32) !Renderer.FrameData.Input {
    _ = alloc;

    var renderer_input: Renderer.FrameData.Input = .{};
    while (app.event_queue.pending()) {
        switch (app.event_queue.pop()) {
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
                    .left => renderer_input.break_block = true,
                    .right => renderer_input.place_block = true,
                    else => {},
                }
            },
            else => {},
        }
    }

    if (app.window.isKeyDown(.five)) renderer_input.break_block = true;

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
    _ = alloc;

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
            }
        },
    }
}
