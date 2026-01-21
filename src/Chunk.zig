const mw = @import("mw");
const math = mw.math;

const Chunk = @This();
pub const Pos = @Vector(3, u32);
pub const size: Pos = .{ 32, 32, 32 };

blocks: [size[2]][size[1]][size[0]]BlockId,

pub const BlockId = enum {
    air,
    grass,
    stone,
};

pub fn init(this: *Chunk, chunk_pos: Pos) void {
    const pos = chunk_pos * size;

    const grass_height = 16;
    for (0..size[2]) |z| {
        for (0..size[1]) |y| {
            for (0..size[0]) |x| {
                const offset: Pos = .{ @intCast(x), @intCast(y), @intCast(z) };
                const block_pos = pos + offset;

                if (block_pos[2] > grass_height) {
                    this.blocks[z][y][x] = .air;
                } else if (block_pos[2] == grass_height) {
                    this.blocks[z][y][x] = .grass;
                } else {
                    this.blocks[z][y][x] = .stone;
                }
            }
        }
    }
}
