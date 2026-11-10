// Shared declarations for every simulation compute shader.
//
// Units: lengths are in grid cells (dx = 1), time in seconds, densities relative to water.
// Grid layout (MAC / staggered), for cell (i, j) spanning [i, i+1] x [j, j+1]:
//   u(i, j)  x-velocity on the LEFT face,   at (i,       j + 0.5)
//   v(i, j)  y-velocity on the BOTTOM face, at (i + 0.5, j)
//   scalars (pressure, composition, occupancy) at the centre (i + 0.5, j + 0.5)
// +y is up; gravity points to -y.

struct Particle {
    vec2 pos;   // cells
    vec2 vel;   // cells / s
    vec4 c;     // APIC affine matrix (du/dx, du/dy, dv/dx, dv/dy), 1 / s
    float s;    // syrup volume fraction of an aqueous particle (0 = water, 1 = syrup)
    uint kind;  // 0 = dead slot, 1 = aqueous (water <-> syrup), 2 = oil
    float pad0;
    float pad1;
};

const uint KIND_DEAD = 0u;
const uint KIND_AQUEOUS = 1u;
const uint KIND_OIL = 2u;

layout(std140, set = 0, binding = 0) uniform Params {
    ivec2 grid;           // W, H in cells
    uint n_slots;         // particle slots in use (high-water mark)
    uint spawn_count;     // entries waiting in the spawn buffer
    float dt;             // substep, s
    float gravity;        // cells / s^2
    float inv_ppc;        // 1 / (particles per cell at rest)
    float fluid_thr;      // occupancy above which a cell is liquid
    vec4 rho;             // relative density: water, syrup, oil
    vec4 nu;              // kinematic viscosity (cells^2 / s): water, syrup, oil
    float sep_tau;        // oil <-> aqueous slip relaxation time, s
    float sep_max;        // cap on slip speed, cells / s
    float diffuse;        // water <-> syrup mixing rate, 1 / s
    float density_k;      // share of a cell's density error corrected per step
    float omega;          // SOR relaxation factor
    float rest_phi;       // rest occupancy
    float vmax;           // particle speed cap, cells / s
    float wall_margin;    // particles are kept this far outside solids, cells
    vec2 tool_pos;        // cells
    vec2 tool_vel;        // cells / s
    float tool_radius;    // cells
    int tool_mode;        // 0 none, 1 grab / stir, 2 erase
    float tool_strength;
    uint frame;
} P;

layout(std430, set = 0, binding = 1) buffer ParticleBuf { Particle particles[]; };
// Fixed-point scatter accumulators, ACC_STRIDE ints per cell:
//   0 u-face mass   1 u-face momentum   2 v-face mass   3 v-face momentum
//   4 cell weight   5 cell oil weight   6 cell syrup weight   7 unused
// (the periodic particle sort borrows slots 6 and 7 between substeps)
layout(std430, set = 0, binding = 2) buffer AccBuf { int acc[]; };
// Face velocities u, v plus, in zw, the density-correction velocity of each face. Faces that
// carry no liquid store FACE_EMPTY there and faces closed by a solid store FACE_CLOSED.
layout(std430, set = 0, binding = 3) buffer VelBuf { vec4 vel[]; };
layout(std430, set = 0, binding = 4) buffer Vel0Buf { vec2 vel0[]; };   // pre-viscosity velocities
layout(std430, set = 0, binding = 5) buffer CellBuf { vec4 cell[]; };   // liquid occupancy, density, dynamic viscosity, level
layout(std430, set = 0, binding = 6) buffer CompBuf { vec4 comp[]; };   // density, oil fraction, syrup fraction, total occupancy
layout(std430, set = 0, binding = 7) buffer CoefBuf { vec4 coef[]; };   // g_u, g_v (1 / (rho * theta)), rho_u, rho_v
layout(std430, set = 0, binding = 8) buffer PwBuf { vec4 pw[]; };       // Poisson weights: left, right, bottom, top
layout(std430, set = 0, binding = 9) buffer RhsBuf { vec4 prhs[]; };    // rhs (pressure), rhs (correction), 1 / diagonal
// Two potentials solved side by side with the same matrix (both warm-started):
//   x: pressure * dt, which makes the velocity field divergence-free
//   y: density-correction potential, whose gradient only ever moves particle positions
layout(std430, set = 0, binding = 10) buffer PresBuf { vec2 pres[]; };
layout(std430, set = 0, binding = 11) readonly buffer SolidBuf { vec4 solid[]; }; // open fraction of left face, bottom face, smoothed solid occupancy, open volume
layout(std430, set = 0, binding = 12) readonly buffer SdfBuf { float sdf[]; };    // (W+1) x (H+1) node-centred signed distance
layout(std430, set = 0, binding = 13) readonly buffer SpawnBuf { Particle spawn[]; };
layout(std430, set = 0, binding = 14) buffer CounterBuf { uint counter[]; };
layout(std430, set = 0, binding = 15) buffer SortedBuf { Particle sorted[]; };  // target of the periodic sort

const float MASS_SCALE = 1048576.0;  // 2^20
const float MOM_SCALE = 4194304.0;   // 2^22, momentum is stored in cells / step
const int ACC_STRIDE = 8;
const int MIN_FACE_MASS = 64;
const float THETA_MIN = 0.1;
const float FACE_EMPTY = 1e30;
const float FACE_CLOSED = -1e30;

bool face_live(float state) { return abs(state) < 1e29; }

int cidx(ivec2 c) { return c.y * P.grid.x + c.x; }

float aqueous_density(float s) { return mix(P.rho.x, P.rho.y, s); }

// Dynamic viscosity of a water/syrup blend. Viscosity of sugar solutions is strongly
// non-linear in concentration, so blend the logarithms (Arrhenius mixing rule).
float aqueous_viscosity(float s) {
    float nu = exp(mix(log(max(P.nu.x, 1e-4)), log(max(P.nu.y, 1e-4)), s));
    return aqueous_density(s) * nu;
}

float particle_density(Particle p) {
    return p.kind == KIND_OIL ? P.rho.z : aqueous_density(p.s);
}

// Quadratic B-spline weights for a point at fractional offset f in [0.5, 1.5) from its base node.
void bspline(vec2 f, out vec2 w[3]) {
    w[0] = 0.5 * (1.5 - f) * (1.5 - f);
    w[1] = 0.75 - (f - 1.0) * (f - 1.0);
    w[2] = 0.5 * (f - 0.5) * (f - 0.5);
}
