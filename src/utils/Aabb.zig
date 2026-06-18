const std = @import("std");
const Pos = @import("../common/Chunk.zig").Pos;
const Aabb = @This();

/// inclusive bound
min: [3]i32,
/// exclusive bound
max: [3]i32,

pub fn isValid(a: Aabb) bool {
    return @reduce(.And, @as(Pos, a.max) > @as(Pos, a.min));
}

pub fn subtract(a: Aabb, b: Aabb, buffer: *[6]Aabb) []Aabb {
    const i = intersection(a, b) orelse {
        buffer.*[0] = a;
        return buffer.*[0..1];
    };

    const boxes: [6]Aabb = .{
        .{ .min = .{ a.min[0], a.min[1], a.min[2] }, .max = .{ i.min[0], a.max[1], a.max[2] } },
        .{ .min = .{ i.max[0], a.min[1], a.min[2] }, .max = .{ a.max[0], a.max[1], a.max[2] } },

        .{ .min = .{ i.min[0], a.min[1], a.min[2] }, .max = .{ i.max[0], i.min[1], a.max[2] } },
        .{ .min = .{ i.min[0], i.max[1], a.min[2] }, .max = .{ i.max[0], a.max[1], a.max[2] } },

        .{ .min = .{ i.min[0], i.min[1], a.min[2] }, .max = .{ i.max[0], i.max[1], i.min[2] } },
        .{ .min = .{ i.min[0], i.min[1], i.max[2] }, .max = .{ i.max[0], i.max[1], a.max[2] } },
    };

    var n: usize = 0;
    for (boxes) |box| {
        if (box.isValid()) {
            buffer.*[n] = box;
            n += 1;
        }
    }

    return buffer.*[0..n];
}

pub fn volume(a: Aabb) u32 {
    std.debug.assert(a.isValid());

    const min: Pos = a.min;
    const max: Pos = a.max;
    const size = max - min;
    return @intCast(@reduce(.Mul, size));
}

pub fn containsPoint(aabb: Aabb, pos: Pos) bool {
    const min: Pos = aabb.min;
    const max: Pos = aabb.max;

    return @reduce(.And, pos >= min) and @reduce(.And, pos < max);
}

pub fn intersection(a: Aabb, b: Aabb) ?Aabb {
    const amin: Pos = a.min;
    const bmin: Pos = b.min;
    const amax: Pos = a.max;
    const bmax: Pos = b.max;

    const result: Aabb = .{
        .min = @max(amin, bmin),
        .max = @min(amax, bmax),
    };

    return if (result.isValid()) result else null;
}
