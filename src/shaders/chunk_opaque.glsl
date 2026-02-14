#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "core.hglsl"

#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3
#define UP    4
#define DOWN  5

layout(location = 0) toPixel flat uint pPacked;

#ifdef _VERTEX

layout(location=0) in uint iPacked;

layout(push_constant) uniform PushConstants {
    mat4 vp;
    // 21 bits per component
    uint64_t packedChunkPos;
} constants;

const ivec3 face_table[6 * 6] = {
    // north
    ivec3(1, 1, 0),
    ivec3(1, 0, 0),
    ivec3(1, 0, 1),
    ivec3(1, 1, 0),
    ivec3(1, 0, 1),
    ivec3(1, 1, 1),
    // south
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 1, 1),
    ivec3(0, 0, 0),
    ivec3(0, 1, 1),
    ivec3(0, 0, 1),
    //  east
    ivec3(0, 1, 0),
    ivec3(1, 1, 0),
    ivec3(1, 1, 1),
    ivec3(0, 1, 0),
    ivec3(1, 1, 1),
    ivec3(0, 1, 1),
    // west
    ivec3(0, 0, 0),
    ivec3(1, 0, 1),
    ivec3(1, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(1, 0, 1),
    // up
    ivec3(0, 0, 1),
    ivec3(0, 1, 1),
    ivec3(1, 1, 1),
    ivec3(0, 0, 1),
    ivec3(1, 1, 1),
    ivec3(1, 0, 1),
    // down
    ivec3(0, 0, 0),
    ivec3(1, 1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 1, 0),
};

const ivec3 width_offset_table[6 * 6] = {
    // north 
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 1),
    // south
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 1),
    // east
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 0, 0),
    // west
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    // up
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 0, 0),
    // down
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 0, 0),
};

const ivec3 height_offset_table[6 * 6] = {
    // north 
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    // south
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    // east
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 1),
    // west
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 0, 1),
    // up
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    // down
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 0),
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
};

int unpackI21(uint64_t packed, uint shift) {
    uint raw = uint(packed >> shift);

    return int(raw << uint(32 - 21)) >> (32 - 21);
}

void main() {
    uint block = (iPacked >> 00) & 0x03;
    uint face =  (iPacked >> 02) & 0x07;
    uvec3 rel_pos = uvec3(
        (iPacked >> 05) & 0x1f,
        (iPacked >> 10) & 0x1f,
        (iPacked >> 15) & 0x1f
    );
    uint w = ((iPacked >> 20) & 0x1f);
    uint h = ((iPacked >> 25) & 0x1f);

    ivec3 chunk_pos = ivec3(
        unpackI21(constants.packedChunkPos,  0),
        unpackI21(constants.packedChunkPos, 21),
        unpackI21(constants.packedChunkPos, 42)
    );

    ivec3 block_pos = ivec3(rel_pos) + chunk_pos * 32;
    uint i = face * 6 + gl_VertexIndex;
    ivec3 width_offset = int(w) * width_offset_table[i];
    ivec3 height_offset = int(h) * height_offset_table[i];
    vec3 face_pos = face_table[i] + width_offset + height_offset;

    gl_Position = constants.vp * vec4(face_pos + vec3(block_pos), 1);
    pPacked = face | (block << 3);
}

#endif

#ifdef _PIXEL

layout(location = 0) out vec4 oColor;

const vec3 sunDir = normalize(vec3(-1, -2, 5));

const vec3 color_lookup[3] = {
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.5, 0.5, 0.5),
};

// const vec3 normal_lookup[6] = {
//     vec3( 1,  0,  0),
//     vec3(-1,  0,  0),
//     vec3( 0,  1,  0),
//     vec3( 0, -1,  0),
//     vec3( 0,  0,  1),
//     vec3( 0,  0, -1),
// };

const float light_lookup[6] = {
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
