// Shared declarations for every simulation compute shader.
//
// Units: lengths are in grid cells (dx = 1), time in seconds, densities relative to water.
// Grid layout (MAC / staggered), for cell (i, j) spanning [i, i+1] x [j, j+1]:
//   u(i, j)  x-velocity on the LEFT face,   at (i,       j + 0.5)
//   v(i, j)  y-velocity on the BOTTOM face, at (i + 0.5, j)
//   scalars (pressure, composition, occupancy) at the centre (i + 0.5, j + 0.5)
//   node (i, j) is the bottom-left corner (i, j) of cell (i, j)
// +y is up; gravity points to -y.

struct Particle {
    vec2 pos;   // cells
    vec2 vel;   // cells / s
    vec4 c;     // APIC affine matrix (du/dx, du/dy, dv/dx, dv/dy), 1 / s
    float s;    // syrup volume fraction of an aqueous particle (0 = water, 1 = syrup)
    uint kind;  // 0 = dead slot, 1 = aqueous (water <-> syrup), 2 = oil
    uint id;    // permanent identity (survives sorting); seeds the per-particle random numbers
    float pad1;
};

const uint KIND_DEAD = 0u;
const uint KIND_AQUEOUS = 1u;
const uint KIND_OIL = 2u;

layout(std140, set = 0, binding = 0) uniform Params {
    ivec2 grid;           // W, H in cells
    uint n_slots;         // particle slots in use (high-water mark)
    uint spawn_count;     // entries waiting in the spawn buffer
    float dt_fixed;       // > 0: fixed substep, s; 0: CFL-limited substep
    float gravity;        // cells / s^2
    float inv_ppc;        // 1 / (particles per cell at rest)
    float fluid_thr;      // occupancy above which a cell is liquid
    vec4 rho;             // relative density: water, syrup, oil
    vec4 nu;              // kinematic viscosity (cells^2 / s): water, syrup, oil
    vec4 sigma;           // tension / rho_water (cells^3 / s^2): water-air, syrup-air, oil-air, oil-aqueous
    vec4 contact_aq;      // cos, sin of the contact angle: water, syrup
    vec4 contact_oil;     // cos, sin of the contact angle: oil; kappa_max (1 / cells); unused
    float sep_tau;        // diameter of unresolved oil / aqueous droplets, cells
    float cfl;            // most cells a particle may travel per substep
    float diffuse;        // water <-> syrup mixing rate, 1 / s
    float density_k;      // share of a cell's density error corrected per step
    float dt_max;         // longest substep, s (includes the capillary limit)
    float rest_phi;       // rest occupancy
    float ext_speed;      // fastest speed the tool or the pour will impose, cells / s
    float wall_margin;    // particles are kept this far outside solids, cells
    vec2 tool_pos;        // cells
    vec2 tool_vel;        // cells / s
    float tool_radius;    // cells
    int tool_mode;        // 0 none, 1 grab / stir, 2 erase
    float tool_strength;
    uint frame;
    vec4 tol;             // relative tolerance: pressure, density correction, viscosity; density-correction time constant (s)
    vec4 ext;             // particle speed cap (cells / s, huge = off); advection order (1 Euler, 2 midpoint); CFL multiple allowed to spray; unused
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
layout(std430, set = 0, binding = 4) buffer Vel0Buf { vec2 vel0[]; };   // velocities before viscosity
layout(std430, set = 0, binding = 5) buffer CellBuf { vec4 cell[]; };   // liquid occupancy, density, dynamic viscosity, level
layout(std430, set = 0, binding = 6) buffer CompBuf { vec4 comp[]; };   // density, oil fraction, syrup fraction, total occupancy
layout(std430, set = 0, binding = 7) buffer CoefBuf { vec4 coef[]; };   // g_u, g_v (1 / (rho * theta)), rho_u, rho_v
// Multigrid matrices of every level back to back (level 0 = the pressure matrix itself):
// couplings to the left, right, bottom, top neighbour. Diagonals are in MgD.
layout(std430, set = 0, binding = 8) buffer MgABuf { vec4 mga[]; };
layout(std430, set = 0, binding = 9) buffer RhsBuf { vec4 prhs[]; };    // rhs (pressure), rhs (correction)
// Two potentials solved side by side with the same matrix (both warm-started):
//   x: pressure * dt, which makes the velocity field divergence-free
//   y: density-correction potential, whose gradient only ever moves particle positions
layout(std430, set = 0, binding = 10) buffer PresBuf { vec2 pres[]; };
layout(std430, set = 0, binding = 11) readonly buffer SolidBuf { vec4 solid[]; }; // open fraction of left face, bottom face, smoothed solid occupancy, open volume
layout(std430, set = 0, binding = 12) readonly buffer SdfBuf { float sdf[]; };    // (W+1) x (H+1) node-centred signed distance
layout(std430, set = 0, binding = 13) readonly buffer SpawnBuf { Particle spawn[]; };
layout(std430, set = 0, binding = 14) buffer CounterBuf { uint counter[]; };
layout(std430, set = 0, binding = 15) buffer SortedBuf { Particle sorted[]; };  // target of the periodic sort

// Time step state. The CPU writes the first four words before each batch of substeps;
// `step.comp` picks every substep's length on the GPU from the velocities of the last one.
layout(std430, set = 0, binding = 16) buffer StepBuf {
    float frame_left;   // simulated time still to cover in this batch, s
    float advanced;     // simulated time since the last reset, s (compensation in pad0)
    uint batch_steps;   // substeps recorded in this batch
    uint batch_done;    // substeps of this batch already run
    float dt;           // this substep, s
    float dt_prev;      // the one before, s (to rescale warm starts)
    uint umax_grid;     // float bits: fastest face velocity after the last projection
    uint umax_part;     // float bits: fastest particle displacement speed in the last advection
    float dt_cfl;       // what the CFL condition allowed for this substep, s
    uint substeps;      // substeps run since the last reset
    float umax_seen;    // speed the last step was sized for, cells / s
    uint pad0;          // float bits: Kahan compensation of `advanced`
} S;

// Krylov solver state, written only by the shaders.
layout(std430, set = 0, binding = 17) buffer SolveBuf {
    vec2 p_rz; vec2 p_alpha;
    vec2 p_beta; vec2 p_bb;
    vec2 p_rr; uvec2 p_done;
    uvec2 p_iters; float v_rz; float v_alpha;
    float v_beta; float v_r0; float v_rr; uint v_done;
    uint v_iters; uint p_it_max; uint v_it_max; uint p_unconv;   // stats from here on are
    uint v_unconv; uint solves; float p_res_max; float v_res_max; // cleared by the CPU
} K;

// Indirect dispatch arguments, 3 uints each: the live set first, then the nominal set it is
// restored from. A converged solve zeroes its live entries so its remaining iterations cost
// only an empty dispatch.
layout(std430, set = 0, binding = 18) buffer IndBuf { uint ind[]; };
layout(std430, set = 0, binding = 19) buffer PartBuf { vec4 part[]; };   // per-workgroup partial sums
layout(std430, set = 0, binding = 20) buffer MgDBuf { float mgd[]; };    // multigrid diagonals
layout(std430, set = 0, binding = 21) buffer MgXBuf { vec2 mgx[]; };     // multigrid: z after pre-smoothing
layout(std430, set = 0, binding = 36) buffer MgZBuf { vec2 mgz[]; };     // multigrid: finished z of each level
layout(std430, set = 0, binding = 22) buffer MgRBuf { vec2 mgr[]; };     // multigrid right-hand side (r)
// Search direction of the pressure solve: the previous one and the one being built. The two
// buffers swap bindings every iteration.
layout(std430, set = 0, binding = 23) buffer CgPBuf { vec2 cgp[]; };
layout(std430, set = 0, binding = 34) buffer CgP2Buf { vec2 cgp2[]; };
layout(std430, set = 0, binding = 24) buffer CgQBuf { vec2 cgq[]; };
layout(std430, set = 0, binding = 25) buffer VisRBuf { vec2 vr[]; };     // viscosity residual
layout(std430, set = 0, binding = 26) buffer VisPBuf { vec2 vp[]; };     // viscosity search direction
layout(std430, set = 0, binding = 27) buffer VisQBuf { vec2 vq[]; };     // A * search direction
// Viscosity weights: 2 dt mu (cell sample), 4 dt mu V (node sample), diagonal of A for u, v.
layout(std430, set = 0, binding = 28) buffer VisWBuf { vec4 vw[]; };
layout(std430, set = 0, binding = 29) buffer VisSBuf { vec4 vs[]; };     // weighted strain of a vector: xx, yy, xy
// Surface: smoothed liquid fraction, smoothed oil fraction, curvature of each (1 / cells).
layout(std430, set = 0, binding = 30) buffer SurfBuf { vec4 surf[]; };
// Ghost pressure (* dt) at the liquid-air interface on the u and v face of each cell.
layout(std430, set = 0, binding = 31) buffer JumpBuf { vec2 pjump[]; };
// One word per 16 x 16 tile: non-zero where the implicit viscosity solve has work to do.
layout(std430, set = 0, binding = 32) buffer TileBuf { uint tiles[]; };

// Multigrid level table: offset into the level buffers, width, height, and the offset of the
// level's tile flags in PTileBuf.
layout(std430, set = 0, binding = 33) readonly buffer LevBuf { ivec4 lev[]; };
// One word per 16 x 16 tile of every dispatched multigrid level: non-zero where the tile
// holds pressure unknowns. Kernels skip the other tiles outright (air, mostly).
layout(std430, set = 0, binding = 35) buffer PTileBuf { uint ptile[]; };
// Particles per cell in a window of the grid (count.comp), for the brush tools.
layout(std430, set = 0, binding = 37) buffer CountBuf { uint cell_count[]; };

const float MASS_SCALE = 1048576.0;  // 2^20
const float MOM_SCALE = 4194304.0;   // 2^22, momentum is stored in cells / step
const int ACC_STRIDE = 8;
const int MIN_FACE_MASS = 64;
const float THETA_MIN = 0.1;
const float FACE_EMPTY = 1e30;
const float FACE_CLOSED = -1e30;

bool face_live(float state) { return abs(state) < 1e29; }
bool face_closed(float state) { return state < -1e29; }

int cidx(ivec2 c) { return c.y * P.grid.x + c.x; }

float aqueous_density(float s) { return mix(P.rho.x, P.rho.y, s); }

// Dynamic viscosity of a water/syrup blend. Viscosity of sugar solutions is strongly
// non-linear in concentration, so blend the logarithms (Arrhenius mixing rule).
float aqueous_viscosity(float s) {
    float nu = exp(mix(log(max(P.nu.x, 1e-8)), log(max(P.nu.y, 1e-8)), s));
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
