#version 450
#extension GL_GOOGLE_include_directive : require

#include "core.hglsl"

layout(location = 0) toPixel flat uint pPacked;
layout(location = 1) toPixel float pAo;
layout(location = 2) toPixel vec2 pUvs;

#ifdef _VERTEX

layout(location = 0) in uvec2 iPacked;

layout(push_constant) uniform PushConstants {
    mat4 vp;
    ivec3 chunk_pos;
} constants;

const ivec3 face_origin_table[6] = {
    ivec3(1, 1, 0),
    ivec3(0, 0, 0),
    ivec3(0, 1, 0),
    ivec3(1, 0, 0),
    ivec3(0, 0, 1),
    ivec3(0, 1, 0),
};

const ivec3 face_right_table[6] = {
    ivec3(0, -1, 0),
    ivec3(0, 1, 0),
    ivec3(1, 0, 0),
    ivec3(-1, 0, 0),
    ivec3(0, 1, 0),
    ivec3(0, -1, 0),
};

const ivec3 face_up_table[6] = {
    ivec3(0, 0, 1),
    ivec3(0, 0, 1),
    ivec3(0, 0, 1),
    ivec3(0, 0, 1),
    ivec3(1, 0, 0),
    ivec3(1, 0, 0),
};

const int right_table[6] = {
    0, 1, 1, 0, 1, 0
};

const int up_table[6] = {
    0, 0, 1, 0, 1, 1
};

const uint ao_index_table[6] = {
    0, 2, 6, 0, 6, 4,
};

const float ao_table[4] = {
    1.00,
    0.40,
    0.20,
    0.10,
};

const vec2 uv_table[6] = {
    vec2(0, 0),
    vec2(1, 0),
    vec2(1, 1),
    vec2(0, 0),
    vec2(1, 1),
    vec2(0, 1),
};

const uint vindex_table[6 * 2] = {
    // not flipped
    0, 1, 2, 3, 4, 5,
    // flipped
    0, 1, 5, 1, 2, 5,
};

void main() {
    uint lower = iPacked.x;
    uint upper = iPacked.y;

    int w = int(((lower >> 18) & 0x1f) + 1);
    int h = int(((lower >> 23) & 0x1f) + 1);
    uint tex_id = (lower >> 29) & 0x1;
    uint flipped = (lower >> 28) & 0x1;
    uint face =  (lower >>  0) & 0x07;
    ivec3 rel_pos = ivec3(
        (lower >>  3) & 0x1f,
        (lower >>  8) & 0x1f,
        (lower >> 13) & 0x1f
    );

    uint vindex = vindex_table[gl_VertexIndex + flipped * 6];
    ivec3 block_pos = rel_pos + constants.chunk_pos * 32;
    ivec3 face_origin = face_origin_table[face];
    ivec3 right_offset = w * right_table[vindex] * face_right_table[face];
    ivec3 up_offset = h * up_table[vindex] * face_up_table[face];
    ivec3 face_pos = face_origin + right_offset + up_offset;

    gl_Position = constants.vp * vec4(face_pos + block_pos, 1);
    pPacked = face | (tex_id << 3);

    pAo = ao_table[(upper >> ao_index_table[vindex]) & 3];

    vec2 base_uv = uv_table[vindex];
    vec2 repeat = ivec2(w, h);
    pUvs = base_uv * repeat;
}

#endif

#ifdef _PIXEL

layout(binding = 0) uniform sampler2DArray uSampler;
layout(location = 0) out vec4 oColor;

const vec3 sun_dir = normalize(vec3(-1, -2, 5));

const float light_lookup[6] = {
    max(0, dot(vec3( 1,  0,  0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(vec3(-1,  0,  0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0,  1,  0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0, -1,  0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0,  0,  1), sun_dir)) * 0.7 + 0.3,
    max(0, dot(vec3( 0,  0, -1), sun_dir)) * 0.7 + 0.3,
};

void main() {
    uint face = pPacked & 0x7;
    uint tex_id = pPacked >> 3;

    float light = light_lookup[face] * pAo;

    vec4 albedo = texture(uSampler, vec3(pUvs, tex_id));
    oColor = albedo * light;
}

#endif
