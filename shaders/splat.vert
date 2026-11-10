#version 460

// Draws every live particle as a small quad, straight from the simulation buffer.

struct Particle {
    vec2 pos;
    vec2 vel;
    vec4 c;
    float s;
    uint kind;
    float pad0;
    float pad1;
};

layout(std430, set = 0, binding = 0) readonly buffer ParticleBuf { Particle particles[]; };

layout(push_constant) uniform PC {
    vec2 scale;   // world (cells) -> clip
    vec2 offset;
    vec4 water;
    vec4 syrup;
    vec4 oil;
    float radius; // splat radius, cells
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_color;

const vec2 CORNER[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    Particle p = particles[gl_VertexIndex / 6];
    if (p.kind == 0u) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        v_uv = vec2(0.0);
        v_color = vec3(0.0);
        return;
    }
    vec2 q = CORNER[gl_VertexIndex % 6];
    gl_Position = vec4((p.pos + q * pc.radius) * pc.scale + pc.offset, 0.0, 1.0);
    v_uv = q;
    // A little syrup tints a lot of water, like a dye.
    v_color = p.kind == 2u ? pc.oil.rgb : mix(pc.water.rgb, pc.syrup.rgb, pow(p.s, 0.7));
}
