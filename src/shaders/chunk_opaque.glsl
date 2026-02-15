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
layout(location = 1) toPixel float pAo;

#ifdef _VERTEX

layout(location=0) in uvec2 iPacked;

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

const uint ao_index_table[6 * 6] = {
    // north
    2, 0, 4, 2, 4, 6,
    // south
    0, 2, 6, 0, 6, 4,
    // east
    0, 2, 6, 0, 6, 4,
    // west
    0, 6, 2, 0, 4, 6,
    // up
    0, 4, 6, 0, 6, 2,
    // down
    0, 2, 1, 0, 3, 2,
};

const float ao_table[4] = {
    1.00,
    0.75,
    0.55,
    0.35,
};

const uint vindex_table[6 * 6 * 2] = {
    // not flipped
    // north
    0, 1, 2, 3, 4, 5,
    // south
    0, 1, 2, 3, 4, 5,
    // east
    0, 1, 2, 3, 4, 5,
    // west
    0, 1, 2, 3, 4, 5,
    // up
    0, 1, 2, 3, 4, 5,
    // down
    0, 1, 2, 3, 4, 5,
    // flipped
    // north
    0, 1, 5, 1, 2, 5,
    // south
    0, 1, 5, 1, 2, 5,
    // east
    0, 1, 5, 1, 2, 5,
    // west
    0, 4, 2, 4, 1, 2,
    // up
    0, 1, 5, 1, 2, 5,
    // down
    0, 4, 2, 4, 1, 2,
};

int unpackI21(uint64_t packed, uint shift) {
    uint raw = uint(packed >> shift);

    return int(raw << uint(32 - 21)) >> (32 - 21);
}

void main() {
    uint lower = iPacked.x;
    uint upper = iPacked.y;

    uint block = (lower >> 28) & 0x03;
    uint face =  (lower >>  0) & 0x07;
    uvec3 rel_pos = uvec3(
        (lower >>  3) & 0x1f,
        (lower >>  8) & 0x1f,
        (lower >> 13) & 0x1f
    );
    uint w = ((lower >> 18) & 0x1f);
    uint h = ((lower >> 23) & 0x1f);

    ivec3 chunk_pos = ivec3(
        unpackI21(constants.packedChunkPos,  0),
        unpackI21(constants.packedChunkPos, 21),
        unpackI21(constants.packedChunkPos, 42)
    );
    uint flipped = 0;

    uint vindex = vindex_table[gl_VertexIndex + flipped * 36 + face * 6];
    uint i = face * 6 + vindex;

    ivec3 block_pos = ivec3(rel_pos) + chunk_pos * 32;
    ivec3 width_offset = int(w) * width_offset_table[i];
    ivec3 height_offset = int(h) * height_offset_table[i];
    vec3 face_pos = face_table[i] + width_offset + height_offset;

    gl_Position = constants.vp * vec4(face_pos + vec3(block_pos), 1);
    pPacked = face | (block << 3);

    pAo = ao_table[(upper >> ao_index_table[i]) & 3];
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
    float light = light_lookup[face] * pAo;

    oColor = vec4(color_lookup[block], 1) * light;
}

#endif
