#version 460
// Draws every live particle as one triangle around its kernel, straight from the simulation
// buffer (half the vertices of a quad, for a third more fill). A moving particle is drawn
// stretched back along its path, so spray reads as streaks. It deposits its kernel weight in
// the channel of its phase group: the sharp field the surfaces and interfaces are drawn from.
// (What the liquid looks like comes from the grid: cell_optics.comp.)

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
    ivec4 ia; ivec4 ib; vec4 a; vec4 b; vec4 rheo; vec4 temp; vec4 th; vec4 th2; vec4 el; vec4 misc;
};

layout(std430, set = 0, binding = 0) readonly buffer ParticleBuf { Particle particles[]; };
layout(std430, set = 0, binding = 1) readonly buffer MatBuf { Mat mats[]; };

layout(push_constant) uniform PC {
    vec2 scale;       // world (cells) -> clip
    vec2 offset;
    float radius;     // kernel radius, cells
    float weight;     // peak weight of one particle (below 1 when the kernel had to be widened)
    float exposure;   // a particle is drawn over the path of this many seconds
    float inv_speed;  // 1 / the speed that counts as fast, s / cell
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out vec4 v_d0;

// The triangle circumscribed about the unit disc.
const vec2 CORNER[3] = vec2[3](vec2(0.0, 2.0), vec2(-1.7320508, -1.0), vec2(1.7320508, -1.0));
// Longest streak, in kernel radii.
const float STREAK_MAX = 1.5;

void main() {
    Particle p = particles[gl_VertexIndex / 3];
    if (p.kind == 0u) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        v_uv = vec2(0.0);
        v_d0 = vec4(0.0);
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
}
