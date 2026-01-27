#version 450
#extension GL_GOOGLE_include_directive : require

#include "core.hglsl"

layout(location = 0) toPixel vec3 pColor;
layout(push_constant) uniform PushConstants {
    mat4 vp;
} constants;

#ifdef _VERTEX

vec4 vert_table[3] = {
	{ 0, 1, 0, 1 },
	{ 0, -1, 0, 1 },
	{ 0, 0, 1, 1 },
};

void main() {
    gl_Position = constants.vp * vert_table[gl_VertexIndex];
    pColor = vec3(1,1,1);
}

#endif

#ifdef _PIXEL

layout(location = 0) out vec4 oColor;

void main() {
     oColor = vec4(pColor, 1);
}

#endif
