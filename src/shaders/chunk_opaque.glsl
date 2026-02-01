#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "core.hglsl"

layout(location = 0) toPixel flat uint pPacked;

#ifdef _VERTEX

layout(location=0) in uint iPacked;
layout(set=0,binding=0) uniform UniformBufferObject {
    vec3 face_table[6 * 6];
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 vp;
    // 21 bits per component
    uint64_t packedChunkPos;
} constants;

int unpackI21(uint64_t packed, uint shift) {
    uint raw = uint(packed >> shift);

    return int(raw << uint(32 - 21)) >> (32 - 21);
}

void main() {
    uvec3 rel_pos = uvec3(
        (iPacked >> 00) & 0x1f,
        (iPacked >> 05) & 0x1f,
        (iPacked >> 10) & 0x1f
    );
    uint face =  (iPacked >> 15) & 0x07;
    uint block = (iPacked >> 18) & 0x03;

    ivec3 chunk_pos = ivec3(
        unpackI21(constants.packedChunkPos,  0),
        unpackI21(constants.packedChunkPos, 21),
        unpackI21(constants.packedChunkPos, 42)
    );

    ivec3 block_pos = ivec3(rel_pos) + chunk_pos * 32;
    vec3 face_pos = ubo.face_table[face * 6 + gl_VertexIndex];

    gl_Position = constants.vp * vec4(face_pos + vec3(block_pos), 1);
    pPacked = face | (block << 3);
}

#endif

#ifdef _PIXEL

layout(location = 0) out vec4 oColor;

vec3 sunDir = normalize(vec3(-1, -2, 5));

vec3 color_lookup[3] = {
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.5, 0.5, 0.5),
};

// vec3 normal_lookup[6] = {
//     vec3( 1,  0,  0),
//     vec3(-1,  0,  0),
//     vec3( 0,  1,  0),
//     vec3( 0, -1,  0),
//     vec3( 0,  0,  1),
//     vec3( 0,  0, -1),
// };

float light_lookup[6] = {
    max(0, dot(vec3( 1,  0,  0), sunDir)) * 0.7 + 0.3,
    max(0, dot(vec3(-1,  0,  0), sunDir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0,  1,  0), sunDir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0, -1,  0), sunDir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0,  0,  1), sunDir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0,  0, -1), sunDir)) * 0.7 + 0.3,
};

void main() {
    uint face = pPacked & 0x7;
    uint block = pPacked >> 3;

    // vec3 normal = normal_lookup[face];

    // float diffuse = max(0, dot(normal, sunDir));
    // float light = diffuse * 0.7 + 0.3;
    float light = light_lookup[face];

    oColor = vec4(color_lookup[block], 1) * light;
}

#endif
