const std = @import("std");
const builtin = @import("builtin");
const App = @import("client/App.zig");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var opts: Options = .{};
    var iter = try init.minimal.args.iterateAllocator(alloc);
    defer iter.deinit();
    _ = iter.next();

    var specified_args: std.StaticBitSet(std.meta.fields(@TypeOf(args)).len) = .initEmpty();
    while (iter.next()) |full_arg| {
        const arg, const maybe_str_val = if (std.mem.findScalar(u8, full_arg, '=')) |eql_pos|
            .{ full_arg[0..eql_pos], full_arg[eql_pos + 1 ..] }
        else
            .{ full_arg, null };

        const arg_enum = arg_map.get(arg) orelse {
            std.log.err("unknown arg: '{s}'", .{arg});
            std.process.exit(1);
        };

        if (specified_args.isSet(@intFromEnum(arg_enum))) {
            std.log.err("{s} specified twice", .{arg});
            std.process.exit(1);
        }
        specified_args.set(@intFromEnum(arg_enum));

        inline for (std.enums.values(ArgEnum)) |other| {
            if (other == arg_enum) {
                const field = std.meta.fields(@TypeOf(args))[@intFromEnum(other)];
                const desc = field.defaultValue().?;

                const val: desc.t = val: switch (@typeInfo(desc.t)) {
                    .int => if (maybe_str_val) |str_val|
                        std.fmt.parseInt(desc.t, str_val, 10) catch {
                            std.log.err("bad int value: '{s}'", .{str_val});
                            std.process.exit(1);
                        }
                    else
                        argRequiresValue(arg),
                    .bool => if (maybe_str_val) |str_val| {
                        if (std.mem.eql(u8, str_val, "true")) break :val true;
                        if (std.mem.eql(u8, str_val, "false")) break :val false;
                        std.log.err("bad bool value: '{s}'", .{str_val});
                        std.process.exit(1);
                    } else true,
                    .pointer => switch (desc.t) {
                        []const u8 => if (maybe_str_val) |str_val| str_val else argRequiresValue(arg),
                        else => @compileError("unsupported type"),
                    },
                    else => @compileError("unsupported type"),
                };

                @field(opts, field.name) = val;
            }
        }
    }

    var app: App = undefined;
    try app.init(alloc, io, opts);
    defer app.deinit();

    while (!app.should_close) {
        try app.tick();
    }
}

fn argRequiresValue(arg: []const u8) noreturn {
    std.log.err("{s} requires value", .{arg});
    std.process.exit(1);
}

pub const Options = blk: {
    const args_fields = std.meta.fields(@TypeOf(args));
    var field_names: [args_fields.len][]const u8 = undefined;
    var field_types: [args_fields.len]type = undefined;
    var field_attribs: [args_fields.len]std.builtin.Type.StructField.Attributes = @splat(.{
        .@"comptime" = false,
        .@"align" = null,
        .default_value_ptr = null,
    });

    for (args_fields, &field_names, &field_types, &field_attribs) |arg_field, *name, *t, *attribs| {
        const desc = arg_field.defaultValue().?;
        const has_default = @hasField(arg_field.type, "default");
        const default: desc.t = if (has_default) desc.default else undefined;

        name.* = arg_field.name;
        t.* = desc.t;
        attribs.* = .{
            .@"comptime" = false,
            .@"align" = null,
            .default_value_ptr = if (has_default) @ptrCast(&default) else null,
        };
    }

    break :blk @Struct(.auto, null, &field_names, &field_types, &field_attribs);
};

const ArgEnum = blk: {
    const TagInt = u32;
    const args_fields = std.meta.fields(@TypeOf(args));
    var names: [args_fields.len][]const u8 = undefined;
    var values: [args_fields.len]TagInt = undefined;

    for (args_fields, &names, &values, 0..) |arg_field, *name, *value, i| {
        name.* = arg_field.name;
        value.* = @intCast(i);
    }

    break :blk @Enum(TagInt, .exhaustive, &names, &values);
};

const arg_map = blk: {
    const KV = struct { []const u8, ArgEnum };
    const args_fields = std.meta.fields(@TypeOf(args));
    var kv_pairs: []const KV = &.{};

    for (args_fields, 0..) |arg_field, i| {
        const name = convertName(arg_field.name);
        const new: KV = .{ &name, @enumFromInt(i) };
        kv_pairs = kv_pairs ++ @as([1]KV, .{new});

        const desc = arg_field.defaultValue().?;
        if (@hasField(arg_field.type, "aliases")) {
            for (std.meta.fields(@TypeOf(desc.aliases))) |alias_field| {
                const raw_alias = alias_field.defaultValue().?;
                const alias = convertName(raw_alias);
                const new2: KV = .{ &alias, @enumFromInt(i) };
                kv_pairs = kv_pairs ++ @as([1]KV, .{new2});
            }
        }
    }

    break :blk std.StaticStringMap(ArgEnum).initComptime(kv_pairs);
};

fn convertName(comptime name: []const u8) [2 + name.len]u8 {
    var result: [2 + name.len]u8 = undefined;
    @memset(result[0..2], '-');
    @memcpy(result[2..], name);
    std.mem.replaceScalar(u8, result[2..], '_', '-');
    return result;
}

const args = .{
    .render_radius = .{
        .aliases = .{"rr"},
        .t = u32,
        .default = switch (builtin.mode) {
            .ReleaseFast, .ReleaseSafe => 64,
            else => 4,
        },
    },
    .render_height = .{
        .aliases = .{"rh"},
        .t = u32,
        .default = switch (builtin.mode) {
            .ReleaseFast, .ReleaseSafe => 16,
            else => 8,
        },
    },
    .ip = .{
        .t = []const u8,
        .default = "localhost",
    },
    .port = .{
        .t = u16,
        .default = 5000,
    },
    .log_chunk_packets = .{
        .t = bool,
        .default = false,
    },
};
