const std = @import("std");
const znet = @import("znet");

pub const server_message = struct {
    pub const Kind = enum(u16) {
        init,

        pub fn decode(reader: *std.Io.Reader) !Kind {
            const int = try reader.takeInt(u16, .little);
            return std.meta.intToEnum(Kind, int) catch return error.BadMessage;
        }
    };

    pub const Init = struct {
        player_id: u32,

        pub fn encode(msg: Init, writer: *std.Io.Writer) !void {
            try writer.writeInt(u16, @intFromEnum(Kind.init), .little);

            try writer.writeInt(u32, msg.player_id, .little);
        }

        pub fn decode(reader: *std.Io.Reader) !Init {
            const player_id = try reader.takeInt(u32, .little);

            return .{
                .player_id = player_id,
            };
        }
    };
};
