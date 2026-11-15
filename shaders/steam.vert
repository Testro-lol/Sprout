#version 460
// The vapour field (README, "Steam"), splatted additively like the sharp field and into a
// target of the same layout (render.rs: Targets::vapour): steam still in the liquid (table
// row 19, material.rs: Steam), each bubble a small kernel (r) and a thin ring (b); where they
// crowd, the kernels merge into one pocket of vapour with a smooth outline (composite.frag
// draws its rim and glint). Steam that escaped is not drawn from here but from the smoke
// field (smoke.comp). Everything else in the particle buffer is skipped.

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

layout(push_constant) uniform PC {
    vec2 scale;     // world (cells) -> clip
    vec2 offset;
    float bubble;   // kernel radius of a bubble, cells
    float bubble_w; // its peak weight (below 1 when the kernel had to be widened)
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out float v_w;

const vec2 CORNER[3] = vec2[3](vec2(0.0, 2.0), vec2(-1.7320508, -1.0), vec2(1.7320508, -1.0));
const uint KIND_STEAM = 20u;  // table row 19 + 1

void main() {
    Particle p = particles[gl_VertexIndex / 3];
    if ((p.kind & 255u) != KIND_STEAM) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        v_uv = vec2(0.0);
        v_w = 0.0;
        return;
    }
    vec2 q = CORNER[gl_VertexIndex % 3];
    v_w = pc.bubble_w;
    gl_Position = vec4((p.pos + q * pc.bubble) * pc.scale + pc.offset, 0.0, 1.0);
    v_uv = q;
}
