struct PushConstants {
    float4x4 vp;
    int3 chunk_pos;
};

struct VSInput {
    [[vk::location(0)]] uint packed;
    uint vert_id : SV_VulkanVertexID;
};

struct VSOutput {
    float4 position : SV_Position;
    float2 uvs;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const int3 face_origin_table[6] = {
    { 1, 1, 0 },
    { 0, 0, 0 },
    { 0, 1, 0 },
    { 1, 0, 0 },
    { 0, 0, 1 },
    { 0, 1, 0 },
};

static const int3 face_right_table[6] = {
    { 0,-1, 0 },
    { 0, 1, 0 },
    { 1, 0, 0 },
    {-1, 0, 0 },
    { 0, 1, 0 },
    { 0,-1, 0 },
};

static const int3 face_up_table[6] = {
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 1, 0, 0 },
    { 1, 0, 0 },
};

static const uint right_table[4] = {
    0, 1, 0, 1
};

static const uint up_table[4] = {
    0, 0, 1, 1
};

static const float2 uv_table[4] = {
    { 0, 0 },
    { 1, 0 },
    { 1, 1 },
    { 0, 1 },
};

[[shader("vertex")]]
VSOutput vsMain(VSInput input) {
    VSOutput o;

    uint packed = input.packed;

    uint face = packed & 7;
    bool has_water_above = packed >> 18 > 0;
    int3 rel_pos = {
        (packed >>  3) & 0x1f,
        (packed >>  8) & 0x1f,
        (packed >> 13) & 0x1f,
    };

    uint vid = input.vert_id;
    int3 block_pos = rel_pos + pc.chunk_pos * 32;
    int3 face_origin = face_origin_table[face];
    int3 right_offset = right_table[vid] * face_right_table[face];
    int3 up_offset = up_table[vid] * face_up_table[face];
    int3 face_pos = face_origin + right_offset + up_offset;
    int3 vertex_pos = face_pos + block_pos;
    float3 fvertex_pos = vertex_pos;

    bool top = !has_water_above && (face == 4 || (face != 5 && up_table[vid] == 1));
    fvertex_pos.z -= uint(top) * 0.125;

    o.position = mul(float4(fvertex_pos, 1), pc.vp);
    o.uvs = uv_table[vid];

    return o;
}

[[vk::binding(0, 0)]]
Sampler2DArray sampler;

[[shader("pixel")]]
float4 psMain(VSOutput input) : SV_Target {
    uint tex_id = 2;

    float4 albedo = sampler.Sample(float3(input.uvs, tex_id));
    albedo.a = 0.95;
    return albedo;
}
