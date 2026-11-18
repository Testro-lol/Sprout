#version 460
#extension GL_GOOGLE_include_directive : enable
#include "optics.glsl"
// Draws every live particle as one triangle around its kernel, straight from the simulation
// buffer (half the vertices of a quad, for a third more fill). A moving particle is drawn
// stretched back along its path, so spray reads as streaks. It deposits its kernel weight in
// the channel of its phase group: the sharp field the surfaces and interfaces are drawn from.
// The same shader draws the tint field (tint.frag, at half the resolution): how the particles
// of the aqueous liquids absorb and scatter light, the colour of those liquids at the
// particles' resolution (composite.frag; the rest of the look comes from the grid,
// cell_optics.comp).

struct Particle {
    vec2 pos;
    vec2 vel;
    vec4 c;
    float s;
    uint kind;
    uint id;
    float t;
};

// The simulation's material table (common.glsl: Mat): the phase group of each material.
struct Mat {
    ivec4 ia; ivec4 ib; vec4 a; vec4 b; vec4 rheo; vec4 temp; vec4 th; vec4 th2; vec4 el; vec4 misc; ivec4 ic; vec4 rx; vec4 gr;
};

layout(std430, set = 0, binding = 0) readonly buffer ParticleBuf { Particle particles[]; };
layout(std430, set = 0, binding = 1) readonly buffer MatBuf { Mat mats[]; };
layout(std430, set = 0, binding = 2) readonly buffer OptBuf { Optics optics[]; };

layout(push_constant) uniform PC {
    vec2 scale;       // world (cells) -> clip
    vec2 offset;
    float radius;     // kernel radius, cells
    float weight;     // peak weight of one particle (below 1 when the kernel had to be widened)
    float exposure;   // a particle is drawn over the path of this many seconds
    float inv_speed;  // 1 / the speed that counts as fast, s / cell
    float tint;       // 1: drawing the tint field (steam does not draw)
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out vec4 v_d0;
// For the tint field: the absorption (rgb) of its concentrate part and its scattering (all of
// it for a liquid of another phase group); its concentrate share, 1 if it is aqueous, 1 if
// not. The diluent's own absorption is smooth and comes from the cells (cell_optics.comp
// layer 15), fading by the share at the pixel as in the cells. The composite uses each
// side's colour only where that side has the pixel nearly to itself (mixed, every speck of
// chocolate in water tinted a ring around itself).
layout(location = 2) flat out vec4 v_conc;
layout(location = 3) flat out vec3 v_share;
// common.glsl: ROW_STEAM. Steam still in the liquid has no look of its own (cell_optics.comp
// leaves it out): it lays down nothing here.
const uint ROW_STEAM = 57u;

// The triangle circumscribed about the unit disc.
const vec2 CORNER[3] = vec2[3](vec2(0.0, 2.0), vec2(-1.7320508, -1.0), vec2(1.7320508, -1.0));
// Longest streak, in kernel radii.
const float STREAK_MAX = 1.5;

void main() {
    Particle p = particles[gl_VertexIndex / 3];
    // Dead slots and puffs of escaped steam (low byte 0: steam.vert draws them). Steam still
    // in the liquid (row 57, material.rs: Steam) is part of the body here, in its group
    // (aqueous): the liquid stays whole around a bubble, and the vapour field (steam.vert,
    // composite.frag) draws the bubble inside it. (Drawn as the holes they leave, a boiling
    // pool showed the dark back wall between its bubbles.)
    uint low = p.kind & 255u;
    if (low == 0u || (pc.tint > 0.0 && low - 1u == ROW_STEAM)) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        v_uv = vec2(0.0);
        v_d0 = vec4(0.0);
        v_conc = vec4(0.0);
        v_share = vec3(0.0);
        return;
    }
    vec2 q = CORNER[gl_VertexIndex % 3];
    float speed = length(p.vel);
    vec2 dir = speed > 1e-3 ? p.vel / speed : vec2(1.0, 0.0);
    float streak = min(speed * pc.exposure, STREAK_MAX * pc.radius);
    // The kernel becomes an ellipse whose front stays on the particle.
    float along = pc.radius + 0.5 * streak;
    vec2 pos = p.pos - dir * (0.5 * streak) + dir * (q.x * along) + vec2(-dir.y, dir.x) * (q.y * pc.radius);
    gl_Position = vec4(pos * pc.scale + pc.offset, 0.0, 1.0);
    v_uv = q;

    uint row = (p.kind & 255u) - 1u;
    vec4 group = vec4(0.0);
    group[clamp(mats[row].ia.x, 0, 3)] = pc.weight;
    v_d0 = group;
    Optics o = optics[row];
    float s = o.idx.x >= 0 ? clamp(p.s, 0.0, 1.0) : 0.0;
    vec4 kc = o.absorb, kp = o.idx.x >= 0 ? optics[o.idx.x].absorb : o.absorb;
    float w = row == ROW_STEAM ? 0.0 : pc.weight;
    bool aq = o.idx.y == 0;
    v_conc = aq ? vec4(s * kc.rgb, s * kc.a + (1.0 - s) * kp.a) * w : kc * w;
    v_share = aq ? vec3(s, 1.0, 0.0) * w : vec3(0.0, 0.0, w);
}
