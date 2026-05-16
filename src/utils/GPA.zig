const std = @import("std");
const builtin = @import("builtin");
const GPA = @This();

const debug = switch (builtin.mode) {
    .Debug, .ReleaseSafe => true,
    else => false,
};

debug_alloc: if (debug) std.heap.DebugAllocator(.{}) else void,

pub fn init() GPA {
    return if (debug)
        .{ .debug_alloc = .init }
    else
        .{ .debug_alloc = {} };
}

pub fn deinit(gpa: *GPA) void {
    if (debug)
        _ = gpa.debug_alloc.deinit();
}

pub fn allocator(gpa: *GPA) std.mem.Allocator {
    return if (debug)
        gpa.debug_alloc.allocator()
    else
        std.heap.smp_allocator;
}
