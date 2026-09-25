struct PushConstants {
    float4x4 vp;
    ChunkData *chunk_data;
};

struct ChunkData {
    int3 chunk_pos;
    uint pad;
    Quad *quads;
};

struct Quad {
    uint packed;
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
    { 0, 1 },
    { 1, 1 },
};

[[shader("vertex")]]
VSOutput vsMain(uint vid : SV_VulkanVertexID, uint instance_id : SV_InstanceId, uint draw_id : SV_DrawIndex) {
    VSOutput o;

    ChunkData chunk_data = pc.chunk_data[draw_id];
    int3 chunk_pos = chunk_data.chunk_pos;
    uint packed = chunk_data.quads[instance_id].packed;

    uint face = packed & 7;
    bool has_water_above = packed >> 18 > 0;
    int3 rel_pos = {
        (packed >>  3) & 0x1f,
        (packed >>  8) & 0x1f,
        (packed >> 13) & 0x1f,
    };

    int3 block_pos = rel_pos + chunk_pos * 32;
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
