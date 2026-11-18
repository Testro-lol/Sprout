#version 460
// The look field (README, "Look", "Carried by the particles"): the particles of liquids with
// platelets, flakes, a marble or a film (listed by look.comp), each as one triangle round a
// kernel, into a target laid out as the sharp field (the same uv). Per particle it lays down
// the sheen of its platelets (how squarely they face the half vector between the light and
// the viewer, H.A.H), its carried scalar phi, which way the platelets lie in the plane, and
// its weight; the composite divides by the weight.

struct Particle {
    vec2 pos;
    vec2 vel;
    vec4 c;
    float s;
    uint kind;
    uint id;
    float t;
};

layout(std430, set = 0, binding = 0) readonly buffer ParticleBuf { Particle particles[]; };
layout(std430, set = 0, binding = 1) readonly buffer LookBuf { uvec4 plook[]; };
layout(std430, set = 0, binding = 2) readonly buffer LookListBuf { uint lctl[]; };

layout(push_constant) uniform PC {
    vec2 scale;     // world (cells) -> clip
    vec2 offset;
    float radius;   // kernel radius, cells
    float weight;   // peak weight
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out vec4 v_look;

const vec2 CORNER[3] = vec2[3](vec2(0.0, 2.0), vec2(-1.7320508, -1.0), vec2(1.7320508, -1.0));
// composite.frag: L3, the key light in 3D; the half vector towards a viewer down +z.
const vec3 H = normalize(vec3(-0.3375, 0.67, 0.66) + vec3(0.0, 0.0, 1.0));


void main() {
    uint slot = lctl[8u + uint(gl_VertexIndex) / 3u];
    Particle p = particles[slot];
    uvec4 st = plook[slot];
    vec2 q = CORNER[gl_VertexIndex % 3];
    gl_Position = vec4((p.pos + q * pc.radius) * pc.scale + pc.offset, 0.0, 1.0);
    v_uv = q;
    // The platelets' sheen: <(n.H)^2> = H.A.H over their distribution A (look.comp); a third
    // where they lie every which way, more where they face the half vector, less on edge.
    vec3 a = vec3(float(st.z & 1023u), float((st.z >> 10) & 1023u), float((st.z >> 20) & 1023u)) / 1023.0 - vec3(0.0, 0.0, 0.5);
    float sheen = H.x * H.x * a.x + H.y * H.y * a.y + 2.0 * H.x * H.y * a.z + H.z * H.z * (1.0 - a.x - a.y);
    float phi = float(st.w & 0xffffu) / 65535.0;
    // Which way the platelets lie in the plane: A_xx - A_yy, 0.5 for no preference.
    float lie = clamp(0.5 + (a.x - a.y), 0.0, 1.0);
    v_look = vec4(sheen, phi, lie, 1.0) * pc.weight;
}
