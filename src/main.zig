const std = @import("std");
const builtin = @import("builtin");
const App = @import("client/App.zig");
const GPA = @import("utils/GPA.zig");

pub fn main() !void {
    var gpa_obj: GPA = .init();
    defer gpa_obj.deinit();
    const alloc = gpa_obj.allocator();

    var app: App = undefined;
    try app.init(alloc);
    defer app.deinit(alloc);

    while (!app.should_close) {
        try app.tick(alloc);
    }
}
