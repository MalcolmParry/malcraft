const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;

pub const Pos = @Vector(3, i32);
pub const PackedPos = packed struct(u64) {
    x: i22,
    y: i22,
    z: i20,

    pub fn pack(pos: Pos) PackedPos {
        return .{
            .x = @intCast(pos[0]),
            .y = @intCast(pos[1]),
            .z = @intCast(pos[2]),
        };
    }

    pub fn vec(pos: PackedPos) Pos {
        return .{ pos.x, pos.y, pos.z };
    }
};

pub const Kind = enum(u4) {
    air,
    grass,
    stone,
    water,
    sand,

    pub const count = std.enums.values(Kind).len;
    pub fn isOpaque(this: Kind) bool {
        return switch (this) {
            .air => false,
            else => true,
        };
    }
};

pub const Face = enum(u3) {
    north,
    south,
    east,
    west,
    up,
    down,

    pub inline fn posDirFromAxis(axis: u8) Face {
        return @enumFromInt(axis * 2);
    }

    pub inline fn opposite(face: Face) Face {
        return switch (face) {
            .north => .south,
            .south => .north,
            .east => .west,
            .west => .east,
            .up => .down,
            .down => .up,
        };
    }

    pub inline fn quat(face: Face) math.Quat {
        return switch (face) {
            .north => math.quat_identity,
            .south => comptime math.quatFromAxisAngle(math.dir_up, math.rad(180.0)),
            .east => comptime math.quatFromAxisAngle(math.dir_up, math.rad(90.0)),
            .west => comptime math.quatFromAxisAngle(math.dir_up, math.rad(-90.0)),
            .up => comptime math.quatFromAxisAngle(math.dir_right, math.rad(-90.0)),
            .down => comptime math.quatFromAxisAngle(math.dir_right, math.rad(90.0)),
        };
    }

    pub inline fn dir(face: Face) Pos {
        return switch (face) {
            .north => .{ 1, 0, 0 },
            .south => .{ -1, 0, 0 },
            .east => .{ 0, 1, 0 },
            .west => .{ 0, -1, 0 },
            .up => .{ 0, 0, 1 },
            .down => .{ 0, 0, -1 },
        };
    }
};
