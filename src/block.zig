const std = @import("std");
const mw = @import("mwengine");
const math = mw.math;

pub const Pos = @Vector(3, i32);

pub const Kind = enum(u2) {
    air,
    grass,
    stone,

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
