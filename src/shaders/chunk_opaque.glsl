#version 450
#extension GL_GOOGLE_include_directive : require

#include "core.hglsl"

layout(location = 0) toPixel vec3 pColor;
layout(location = 1) toPixel vec3 pNormal;
layout(push_constant) uniform PushConstants {
    mat4 vp;
    ivec3 chunkPos;
} constants;

#ifdef _VERTEX

layout(location=0) in vec3 iPos;
layout(location=1) in vec3 iColor;
layout(location=2) in vec3 iNormal;

void main() {
    gl_Position = constants.vp * vec4(iPos + constants.chunkPos * 32, 1);
    pColor = iColor;
    pNormal = iNormal;
}

#endif

#ifdef _PIXEL

layout(location = 0) out vec4 oColor;

vec3 sunDir = normalize(vec3(-1, -2, 5));

void main() {
    float diffuse = max(0, dot(pNormal, sunDir));
    float light = diffuse * 0.7 + 0.3;

     oColor = vec4(pColor, 1) * light;
}

#endif
