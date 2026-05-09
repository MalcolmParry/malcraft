const mw = @import("mwengine");
const math = mw.math;
const Camera = @This();

pos: math.Vec3,
euler: math.Vec3,
v_fov: f32,
near: f32,
far: f32,

pub const default: Camera = .{
    .pos = .{ 0, 0, 210 },
    .euler = .{ 0, 0, 0 },
    .v_fov = math.rad(90.0),
    .near = 0.1,
    .far = 10_000,
};

pub fn view(this: Camera) math.Mat4 {
    return math.matMulMany(math.Mat4, .{
        math.rotateEuler(this.euler),
        math.translate(-this.pos),
    });
}

pub fn proj(this: Camera, aspect_ratio: f32) math.Mat4 {
    return math.matMul(
        math.Mat4,
        math.perspective(aspect_ratio, this.v_fov, this.near, this.far),
        math.to_vulkan,
    );
}

pub fn vp(this: Camera, aspect_ratio: f32) math.Mat4 {
    return math.matMul(math.Mat4, this.proj(aspect_ratio), this.view());
}
