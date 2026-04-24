struct PushConstants {
    float4x4 vp;
    int3 chunk_pos;
};

struct VSInput {
    [[vk::location(0)]] uint2 packed;
    uint vert_id : SV_VulkanVertexID;
};

struct VSOutput {
    float4 position : SV_Position;
    nointerpolation uint packed;
    float ao;
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

static const uint right_table[6] = {
    0, 1, 1, 0, 1, 0
};

static const uint up_table[6] = {
    0, 0, 1, 0, 1, 1
};

static const uint ao_index_table[6] = {
    0, 2, 6, 0, 6, 4
};

static const float ao_table[4] = {
    1.00,
    0.80,
    0.60,
    0.40,
};

static const float2 uv_table[6] = {
    { 0, 0 },
    { 1, 0 },
    { 1, 1 },
    { 0, 0 },
    { 1, 1 },
    { 0, 1 },
};

static uint vid_flip_table[12] = {
    // not flipped
    0, 1, 2, 3, 4, 5,
    // flipped
    0, 1, 5, 1, 2, 5,
};

[[shader("vertex")]]
VSOutput vsMain(VSInput input) {
    VSOutput o;

    uint lower = input.packed.x;
    uint upper = input.packed.y;

    uint w = ((lower >> 18) & 0x1f) + 1;
    uint h = ((lower >> 23) & 0x1f) + 1;
    uint tex_id = (lower >> 29) & 0x01;
    uint flipped = (lower >> 28) & 0x01;
    uint face = (lower >> 0) & 0x07;
    int3 rel_pos = {
        (lower >>  3) & 0x1f,
        (lower >>  8) & 0x1f,
        (lower >> 13) & 0x1f,
    };

    uint vid = vid_flip_table[input.vert_id + flipped * 6];
    int3 block_pos = rel_pos + pc.chunk_pos * 32;
    int3 face_origin = face_origin_table[face];
    int3 right_offset = w * right_table[vid] * face_right_table[face];
    int3 up_offset = h * up_table[vid] * face_up_table[face];
    int3 face_pos = face_origin + right_offset + up_offset;
    int3 vertex_pos = face_pos + block_pos;

    o.position = mul(float4(vertex_pos, 1), pc.vp);
    o.packed = face | (tex_id << 3);
    o.ao = ao_table[(upper >> ao_index_table[vid]) & 3];

    float2 base_uv = uv_table[vid];
    float2 repeat = { w, h };
    o.uvs = base_uv * repeat;

    return o;
}

[[vk::binding(0, 0)]]
Sampler2DArray sampler;

static const float3 sun_dir = normalize(float3(-1, -2, 5));

static const float light_lookup[6] = {
    max(0, dot(float3( 1, 0, 0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(float3(-1, 0, 0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(float3( 0, 1, 0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(float3( 0,-1, 0), sun_dir)) * 0.7 + 0.3,
    max(0, dot(float3( 0, 0, 1), sun_dir)) * 0.7 + 0.3,
    max(0, dot(float3( 0, 0,-1), sun_dir)) * 0.7 + 0.3,
};

[[shader("pixel")]]
float4 psMain(VSOutput input) : SV_Target {
    uint face = input.packed & 0x7;
    uint tex_id = input.packed >> 3;

    float light = light_lookup[face] * (input.ao * input.ao);

    float4 albedo = sampler.Sample(float3(input.uvs, tex_id));
    return albedo * light;
}
