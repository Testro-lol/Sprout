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
    float s;    // concentration: share of the particle that is its own material (rest: its diluent)
    uint kind;  // 0 = dead slot, otherwise the material's table row + 1 (low byte)
    uint id;    // permanent identity (survives sorting); seeds the per-particle random numbers
    float t;    // enthalpy temperature, deg C (heat.comp)
};

const uint KIND_DEAD = 0u;
// A puff of steam that escaped at the free surface (params.rs: KIND_PUFF): low byte 0, `s`
// its age. Drawn only; particles.comp moves it, nothing else sees it.
const uint KIND_PUFF = 0x40000u;
// Of steam (row ROW_STEAM): the table row + 1 of the liquid it boiled from, in bits 8 - 15
// (params.rs: KIND_SRC); it condenses back into that liquid.
const uint KIND_SRC = 0xff00u;
const uint ROW_WATER = 0u;   // material.rs: Material::Water
const uint ROW_STEAM = 19u;  // material.rs: Material::Steam
const int GROUPS = 4;        // material.rs: GROUPS (0 aqueous, 1 oil-like, 2 liquid metal, 3 silicate melt)
const int MAX_PRESENT = 12;  // material.rs: MAX_PRESENT

const int MAX_CURRENTS = 8;  // params.rs: MAX_CURRENTS
const int MAX_MAGNETS = 8;   // params.rs: MAX_MAGNETS

layout(std140, set = 0, binding = 0) uniform Params {
    ivec2 grid;           // W, H in cells
    uint n_slots;         // particle slots in use (high-water mark)
    uint spawn_count;     // entries waiting in the spawn buffer
    float dt_fixed;       // > 0: fixed substep, s; 0: CFL-limited substep
    float gravity;        // cells / s^2
    float inv_ppc;        // 1 / (particles per cell at rest)
    float apic_clamp;     // bound on APIC matrix entries, units of 1 / dt (0 = off)
    vec4 pair[GROUPS];    // interfacial tension between phase groups / rho_water (cells^3 / s^2)
    vec4 misc;            // curvature limit (1 / cells, 0 = none); sealed-region vent (1 on); cell size (m); rigid bodies: friction on solids (< 0: no contact handling)
    int n_present;        // materials in the tank
    int acc_stride;       // ints per cell in the scatter accumulators
    int n_fam;            // concentration families in the tank
    uint group_mask;      // bit g: phase group g is in the tank
    uint caps;            // CAP_* bits: what the materials in the tank need
    int primary;          // row of the family that shares the first row's weight (-1 none); see grid_prep.comp
    uint group_heat;      // byte g: channel + 1 of phase group g's temperature sum (0: none; sim/mod.rs: Tank::chan_g)
    uint pad_c;
    ivec4 present[MAX_PRESENT / 4];  // table rows of the materials in the tank
    ivec4 fams[MAX_PRESENT];         // per concentration family: its row, its diluent's row
    vec4 heat;            // ambient temperature (deg C); heat rate (x real time, 0 = no conduction); air film coefficient (cells / s); radiation coefficient (cells / s / K^3)
    vec4 heat2;           // ln kinematic viscosity of frozen liquid (cells^2 / s); heat / cool tool rate (K / s for water); channel of the temperature sum; heat transfer coefficient between different liquids (cells / s, like the air film)
    float sep_tau;        // diameter of unresolved oil / aqueous droplets, cells
    float cfl;            // most cells a particle may travel per substep
    float tool_crowd_max; // occupancy (x rest) above which Push and Grab stop driving a particle (0 = off)
    float density_k;      // share of a cell's density error corrected per step
    float dt_max;         // longest substep, s (includes the capillary limit)
    float crowd_jitter;   // amplitude of the shake that separates stacked particles, cells (0 = off)
    float ext_speed;      // fastest speed the tool or the pour will impose, cells / s
    float wall_margin;    // particles are kept this far outside solids, cells
    vec2 tool_pos;        // cells
    vec2 tool_vel;        // cells / s
    float tool_radius;    // cells
    int tool_mode;        // 0 none, 1 grab / stir, 2 erase, 3 push, 4 stream nozzle
    float tool_strength;
    uint frame;
    vec4 tol;             // relative tolerance: pressure, density correction, viscosity; density-correction time constant (s)
    vec4 ext;             // particle speed cap (cells / s, huge = off); advection order (1 Euler, 2 midpoint); CFL multiple allowed to spray; number of currents
    vec4 corr;            // density correction: largest step (cells, 0 = none), dead band; oil slip cap (1 on); explicit-viscosity diffusion number
    vec4 current_zone[MAX_CURRENTS];  // position (cells), radius, rate (1 / s)
    vec4 current_vel[MAX_CURRENTS];   // velocity the liquid is driven towards (cells / s); unused
    vec4 mag;             // magnets; largest susceptibility in the tank; its saturation (T); Pa -> grid units
    vec4 magnet[MAX_MAGNETS];      // position (cells), radius (cells), field at the radius (T)
    vec4 magnet_dir[MAX_MAGNETS];  // pole direction
    vec4 elastic;         // accumulator channel of the stress (3 channels); largest shear modulus in the tank (grid units); most stress, x modulus; most deformation per substep
    vec4 steam;           // vapour quality below which steam condenses; puff rise speed (cells / s); puff lifetime (s); quality at which boiling water becomes steam
    vec4 steam2;          // rate of the exchange inside a cell that holds steam (1 / s); 1: boiling aqueous liquid turns into steam (0: it leaves at once); eddy speed of the air puffs ride (cells / s); eddy size (cells)
    vec4 heat3;           // Smagorinsky constant of the sub-grid heat mixing (0 off); evaporation: L_vap / c_p,air (K, 0 off); unused; unused
} P;

layout(std430, set = 0, binding = 1) buffer ParticleBuf { Particle particles[]; };
// Fixed-point scatter accumulators, ACC_STRIDE ints per cell:
//   0 u-face mass   1 u-face momentum   2 v-face mass   3 v-face momentum
//   4 cell weight
// then the channels the tank's materials were given (Mat.ia.zw): the weight of each
// material but the first (whose weight is the remainder) and, for a material with a
// diluent, its weight times concentration. P.acc_stride (at least 8) ints per cell.
// (the periodic particle sort borrows slots 6 and 7 between substeps)
layout(std430, set = 0, binding = 2) buffer AccBuf { int acc[]; };
// Face velocities u, v plus, in zw, the density-correction velocity of each face. Faces that
// carry no liquid store FACE_EMPTY there and faces closed by a solid store FACE_CLOSED.
layout(std430, set = 0, binding = 3) buffer VelBuf { vec4 vel[]; };
layout(std430, set = 0, binding = 4) buffer Vel0Buf { vec2 vel0[]; };   // velocities before viscosity
layout(std430, set = 0, binding = 5) buffer CellBuf { vec4 cell[]; };   // liquid occupancy, density, dynamic viscosity, level
layout(std430, set = 0, binding = 6) buffer CompBuf { vec4 comp[]; };   // density, share of group 1, mean concentration of family 0 (-1: none here), total occupancy
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
// Surface: smoothed liquid fraction, smoothed share of group 1, curvature of each (1 / cells).
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

// Regions of open cells connected through open faces (scene.rs: `regions`). `region_air`
// collects, each substep, one bit per region that holds a cell without liquid; `region` has
// per cell the number of its region (bits 0-4) and whether it is the region's vent (bit 8).
layout(std430, set = 0, binding = 38) buffer RegionBuf { uint region_air; uint region[]; };
const uint REGION_MASK = 31u;
const uint REGION_VENT = 256u;

// One row of the material table (material.rs: GpuMat), in grid units.
struct Mat {
    ivec4 ia;    // phase group; diluent row (-1 none); channel of its weight (-1: the remainder); channel of weight * concentration (-1 none)
    ivec4 ib;    // family index (-1 none); rheology model; temperature model; residue row (-1 none)
    vec4 a;      // relative density; ln kinematic viscosity (cells^2 / s) at the reference temperature; tension vs air / rho_water; mixing rate (1 / s)
    vec4 b;      // cos, sin of the contact angle; grain diameter (cells); relative density when frozen
    vec4 rheo;   // yield stress / rho_water; flow index; shear time (s); viscosity ratio
    vec4 temp;   // temperature law B (K), C (K), reference temperature (K); skin yield / rho_water
    vec4 th;     // volumetric heat capacity / water's; conductivity; solidus, liquidus (deg C)
    vec4 th2;    // latent heat as degrees; emissivity; boiling point (deg C); latent heat of vaporisation as degrees
    vec4 el;     // shear modulus / rho_water; relaxation time (s); susceptibility; residue share
    vec4 misc;   // freeze shift per unit concentration (K); load; viscosity blending exponent against the diluent; thermal expansion (1/K)
};
layout(std430, set = 0, binding = 39) readonly buffer MatBuf { Mat mats[]; };
// Per cell, from grid_prep.comp:
//   aux:   kinematic viscosity and density of the cell's majority phase group, shares of groups 2 and 3
//   sprop: tension against air, cos and sin of the contact angle (not normalised), the majority group
//   mixb:  mean concentration of families 1.. (P.n_fam - 1 floats per cell; -1: none here)
//   surf2: smoothed shares of groups 2 and 3, curvature of each
layout(std430, set = 0, binding = 40) buffer AuxBuf { vec4 aux[]; };
// Per cell, where phase groups meet (grid_prep.comp): share x density of each group, so that
// the drift (particles.comp) can take each group's local mean density.
layout(std430, set = 0, binding = 59) buffer GroupRhoBuf { vec4 grho[]; };
// Per cell, where a phase group other than the aqueous one holds two or more materials
// (`group_sorts`): share x base density (no expansion, not frozen) of groups 1 - 3, so that
// the drift within a group (particles.comp) can take the group's local mean.
// xyz: share x base density of groups 1 - 3 (the drift within a group); w: share of steam.
layout(std430, set = 0, binding = 65) buffer GroupBaseBuf { vec4 gbase[]; };
layout(std430, set = 0, binding = 41) buffer SPropBuf { vec4 sprop[]; };
layout(std430, set = 0, binding = 42) buffer MixBuf { float mixb[]; };
layout(std430, set = 0, binding = 43) buffer Surf2Buf { vec4 surf2[]; };

// Rigid bodies of frozen liquid (rigid.comp). rl: per cell its body's root cell (-1: not
// rigid) and, at a root, the body's slot in rbody. rbody[0]: bodies claimed; body b at
// 16 + BODY_WORDS b: 64-bit fixed-point sums (lo, hi words) of mass, mass x, mass y, mass u,
// mass v, mass (x v - y u), mass (x^2 + y^2) + cell inertia; then of its cells that touch a
// solid (contacts): count, normal x, normal y, x, y, x^2, y^2, x y. Positions are relative
// to the root cell.
layout(std430, set = 0, binding = 48) buffer LabelBuf { ivec2 rl[]; };
layout(std430, set = 0, binding = 49) buffer BodyBuf { uint rbody[]; };
const int MAX_BODIES = 4096;   // mod.rs: RIGID_BODIES
const uint BODY_WORDS = 32u;   // mod.rs: BODY_WORDS
const float FIX = 65536.0;

float get64(uint w) {
    return (float(int(rbody[w + 1u])) * 4294967296.0 + float(rbody[w])) / FIX;
}

// Contact of a rigid body with the solids it touches (params.rs: rigid_contact,
// rigid_friction). The liquid solve sees a body as liquid, and a body that is not one solid
// block (a heap of frozen lumps, water in its pores) slumps in it like liquid: the momentum
// average of that slump moved the whole body into the floor, the particle collision pushed
// it back out, and it shook, and the liquid around it, every substep. Here the contacts are
// treated as one support: a line through their centroid cc, along the mean contact normal
// n, as wide as their spread. The support stops motion into it (inelastic) and lets the body
// turn only as far as the support's ends stay out of the solid, or, where the centre of mass
// overhangs the support, about its edge (the body tips over). The speed it stops along n
// bounds a Coulomb friction along the support. Everything is relative to the root cell.
void body_contact(uint o, float M, vec2 cm, inout vec2 U, inout float w) {
    float mu = P.misc.w;
    float nc = get64(o + 14u);
    if (mu < 0.0 || nc < 0.5) return;
    vec2 ns = vec2(get64(o + 16u), get64(o + 18u));
    if (length(ns) < 0.25 * nc) return;  // contacts on opposite sides: wedged, leave it to the solve
    vec2 n = normalize(ns);
    vec2 t = vec2(n.y, -n.x);
    vec2 cc = vec2(get64(o + 20u), get64(o + 22u)) / nc;
    vec3 sq = vec3(get64(o + 24u), get64(o + 26u), get64(o + 28u)) / nc - vec3(cc.x * cc.x, cc.y * cc.y, cc.x * cc.y);
    // Half width of the support along t (a uniform spread of contact cells), plus half a cell.
    float ext = sqrt(3.0 * max(t.x * t.x * sq.x + t.y * t.y * sq.y + 2.0 * t.x * t.y * sq.z, 0.0)) + 0.5;
    vec2 rc = cc - cm;
    vec2 uc = U + w * vec2(-rc.y, rc.x);  // body velocity at the support's centre
    float vn = dot(uc, n);
    float s = dot(cm - cc, t);            // where the centre of mass is along the support
    // The velocity of a support point cc + a t along n is vn + w a.
    float vn1 = vn, w1 = w;
    if (abs(s) <= ext) {
        // Resting on it: neither end may go in.
        vn1 = max(vn, 0.0);
        w1 = clamp(w, -vn1 / ext, vn1 / ext);
    } else {
        // Overhanging: it may tip about the edge on the side of its centre of mass.
        float edge = sign(s) * ext;
        vn1 = max(vn + w * edge, 0.0) - w * edge;
    }
    if (vn1 == vn && w1 == w) return;
    // Friction along the support, bounded by the normal speed the contact took away.
    float vt = dot(uc, t);
    float vt1 = sign(vt) * max(abs(vt) - mu * (vn1 - vn), 0.0);
    uc += (vn1 - vn) * n + (vt1 - vt) * t;
    w = w1;
    U = uc - w * vec2(-rc.y, rc.x);
}

// Rigid motion of body slot b (root cell r) at x (cells): velocity and angular velocity.
// False if the body is too light to have one.
bool body_motion(uint b, int r, vec2 x, out vec3 u) {
    uint o = 16u + BODY_WORDS * b;
    float M = get64(o);
    if (M <= 1e-3) return false;
    vec2 cm = vec2(get64(o + 2u), get64(o + 4u)) / M;
    vec2 U = vec2(get64(o + 6u), get64(o + 8u)) / M;
    float L = get64(o + 10u) - M * (cm.x * U.y - cm.y * U.x);
    float I = get64(o + 12u) - M * dot(cm, cm);
    float w = I > 1e-3 ? L / I : 0.0;
    body_contact(o, M, cm, U, w);
    vec2 d = x - (vec2(r % P.grid.x, r / P.grid.x) + 0.5) - cm;
    u = vec3(U + w * vec2(-d.y, d.x), w);
    return true;
}

const uint CAP_FAMILIES = 1u;   // params.rs: CAP_*
const uint CAP_THERMAL = 2u;    // the tank is not isothermal: temperatures are carried and conducted
const uint CAP_RIGID = 4u;      // liquid can turn rigid (frozen, too viscous, unyielded, jammed): rigid.comp runs
const uint CAP_RHEO = 8u;       // a non-Newtonian liquid (or one with a skin) is in the tank: rheo.comp runs
const uint CAP_MAGNET = 16u;    // a magnet and a susceptible liquid: magnet.comp runs
const uint CAP_ELASTIC = 32u;   // a viscoelastic liquid: particles carry a stress, elastic.comp runs
const uint CAP_SETTLE = 64u;    // a suspension whose grains settle: settle.comp runs
const uint CAP_STEAM = 128u;    // boiling aqueous liquid turns into steam (README, "Steam")

// Specialisation constant 0 (sim/mod.rs: CapsPass): the CAP_* bits the pipeline was built for.
// has_cap() of a bit outside them is constant false, so the driver drops the code behind it
// and the registers it would hold (a pass compiled for every capability ran the particle pass
// of a plain water-oil-syrup tank 12 % slower). An unspecialised pipeline keeps every bit.
layout(constant_id = 0) const uint SPEC_CAPS = 0xffffffffu;
bool has_cap(uint c) { return (P.caps & SPEC_CAPS & c) != 0u; }

// Per cell and concentration family (P.n_fam entries per cell): the family's occupancy of
// the cell (grid_prep.comp), and what settling moves this substep (settle.comp): the
// concentrate gained as a share of the room the cell's family has for it, and the
// concentrate lost as a share of what it holds.
layout(std430, set = 0, binding = 55) buffer FamOccBuf { float famocc[]; };
layout(std430, set = 0, binding = 56) buffer SettleBuf { vec2 dsettle[]; };

// Terminal velocity (cells / s) of a grain or droplet of diameter d (cells) whose density
// differs by drho from a fluid of density rho_c and kinematic viscosity nu_c (relative
// densities), with the drag law C_d = 24/Re + 0.44 (Stokes at low Reynolds number, Newton
// at high): (4/3) |drho| g d = 24 nu_c rho_c v / d + 0.44 rho_c v^2.
float terminal(float drho, float rho_c, float nu_c, float d) {
    float qa = 0.44 * rho_c, qb = 24.0 * nu_c * rho_c / d, qc = 4.0 / 3.0 * abs(drho) * P.gravity * d;
    // (No buoyancy, no drift: also keeps 0 / 0 out when gravity and viscosity are both zero.)
    return qc > 0.0 ? 2.0 * qc / (qb + sqrt(qb * qb + 4.0 * qa * qc)) : 0.0;
}

// Viscoelastic stress (xx, yy, xy; grid units) per particle slot, side buffers that hold
// real memory only while an elastic material is in the tank (mod.rs: `set_elastic`), and
// per cell (elastic.comp). STRESS_FIX: fixed-point scale of stress / P.elastic.y.
layout(std430, set = 0, binding = 52) buffer PStressBuf { vec4 pstress[]; };
layout(std430, set = 0, binding = 53) buffer PStress2Buf { vec4 pstress2[]; };
layout(std430, set = 0, binding = 54) buffer EStrBuf { vec4 estr[]; };
const float STRESS_FIX = 65536.0;

// Per cell (when CAP_MAGNET): susceptibility of the cell's liquid (share-weighted), and the
// magnetic energy density of pure magnetic liquid there (grid units; magnet.comp).
layout(std430, set = 0, binding = 51) buffer MagBuf { vec2 magc[]; };

// Per cell, from grid_prep.comp (when CAP_RIGID): share of the cell's non-Newtonian
// material, its table row, its concentration, and 1 where the cell is too stiff to flow
// (rheo.comp adds unyielded and jammed cells).
layout(std430, set = 0, binding = 50) buffer RheoBuf { vec4 rh[]; };

// ln of the factor by which the viscosity of material `r` at T (deg C) differs from its
// value at the reference temperature (material.rs: TEMP_*; mirrored in `MatProps::temp_factor_ln`).
float temp_law(int r, float T) {
    int model = mats[r].ib.z;
    if (model == 0) return 0.0;
    vec4 tp = mats[r].temp;  // B, C, reference temperature (K)
    float Tk = max(T + 273.15, 1.0);
    float x = model == 1 ? tp.x * (1.0 / Tk - 1.0 / tp.z) : tp.x / max(Tk - tp.y, 5.0) - tp.x / max(tp.z - tp.y, 5.0);
    return clamp(x, -30.0, 30.0);
}

// Heat (heat.comp). therm: per cell temperature (deg C), heat capacity (sum of particle
// weights x volumetric heat capacity relative to water), conductivity, solid fraction.
// dheat: what the particles take next substep (heat.comp): x the change of enthalpy
// temperature from conduction (gathered with the scatter weights); for the particles whose
// position lies in the cell, y lambda, the share of its difference from z, the heat-capacity-
// weighted mean of (T + 273.15) / T_SCALE of exactly those particles, that each gives up to the others
// (exchange inside the cell).
// hcell: per cell, sums over the particles whose position lies in it (not spread over
// neighbours): heat capacity x MASS_SCALE, heat capacity x (T + 273.15) / T_SCALE x MASS_SCALE.
// solid_t: temperature a solid is held at (deg C), NO_TEMP where solids insulate.
layout(std430, set = 0, binding = 44) buffer ThermBuf { vec4 therm[]; };
layout(std430, set = 0, binding = 45) buffer DHeatBuf { vec4 dheat[]; };
layout(std430, set = 0, binding = 61) buffer HCellBuf { ivec2 hcell[]; };
layout(std430, set = 0, binding = 46) readonly buffer SolidTBuf { float solid_t[]; };
// Ledger of what crossed the boundary of the simulation, never cleared by the shaders:
//   [2 e], [2 e + 1]  energy entry e as a 64-bit fixed-point sum (LEDGER_SCALE per K of one
//                     water-filled cell, rounded to nearest per addition): 0 gained from air, 1 lost to air, 2 gained from
//                     solids, 3 lost to solids, 4 added by the heat tool, 5 removed by the
//                     cool tool, 6 carried off by evaporated liquid
//   [16 + m]          particles of row m that boiled away
//   [80 + m]          particles of row m left behind as residue
//   [144 + m]         particles of row m that turned into steam (README, "Steam")
layout(std430, set = 0, binding = 47) buffer LedgerBuf { uint ledger[]; };
const float NO_TEMP = -1e9;
const float LEDGER_SCALE = 65536.0;
const float T_SCALE = 4096.0;   // the temperature channel holds weight * C * (T + 273.15) / T_SCALE

void ledger_add(int entry, float v) {
    uint x = uint(v * LEDGER_SCALE + 0.5);
    if (x == 0u) return;
    uint old = atomicAdd(ledger[2 * entry], x);
    if (old + x < old) atomicAdd(ledger[2 * entry + 1], 1u);
}

const float MASS_SCALE = 1048576.0;  // 2^20
const float MOM_SCALE = 4194304.0;   // 2^22, momentum is stored in cells / step
const int MIN_FACE_MASS = 64;
const float THETA_MIN = 0.1;
// params.rs: REST_PHI, LIQUID_THR (definitions, not settings).
const float REST_PHI = 1.0;    // rest occupancy of a cell
const float LIQUID_THR = 0.5;  // occupancy above which a cell is liquid
const float FACE_EMPTY = 1e30;
const float FACE_CLOSED = -1e30;

bool face_live(float state) { return abs(state) < 1e29; }
bool face_closed(float state) { return state < -1e29; }

int cidx(ivec2 c) { return c.y * P.grid.x + c.x; }

int present_row(int j) { return P.present[j >> 2][j & 3]; }

// Density and ln(kinematic viscosity) of material `m` at concentration s: linear by volume
// and logarithmic (the Arrhenius mixing rule: viscosity of solutions is strongly non-linear
// in concentration) between its diluent and itself.
float row_density(uint row, float s) {
    int d = mats[row].ia.y;
    float rho = mats[row].a.x;
    return d >= 0 ? mix(mats[d].a.x, rho, s) : rho;
}

// Thermal expansion: the factor on the density of liquid `row` at concentration s and
// temperature T (deg C), linear about the reference temperature (misc.w: coefficient, 1/K;
// temp.z: reference, K). Frozen liquid takes the solid's density instead. A liquid is never
// hotter than its boiling point (a cell's mean temperature can be, with lava in it).
float expansion(uint row, float s, float T) {
    int d = mats[row].ia.y;
    float b = mats[row].misc.w, tr = mats[row].temp.z - 273.15, tb = mats[row].th2.z;
    if (d >= 0) {
        b = mix(mats[d].misc.w, b, s);
        tr = mix(mats[d].temp.z - 273.15, tr, s);
        tb = mix(mats[d].th2.z, tb, s);
    }
    return clamp(1.0 - b * (min(T, tb) - tr), 0.7, 1.3);
}

// Thermal data of material `row` at concentration s: solidus, liquidus (deg C), latent heat
// (degrees: L / c) and volumetric heat capacity (relative to water). A concentrate freezes
// like its diluent, shifted by `freeze shift` per unit concentration (brine), with the
// latent heat blended by volume.
vec4 thermal_of(uint row, float s) {
    vec4 th = mats[row].th;
    float ld = mats[row].th2.x;
    int d = mats[row].ia.y;
    if (d < 0) return vec4(th.z, th.w, ld, th.x);
    vec4 dth = mats[d].th;
    float shift = mats[row].misc.x * s;
    float C = mix(dth.x, th.x, s);
    return vec4(dth.z + shift, dth.w + shift, mix(dth.x * mats[d].th2.x, th.x * ld, s) / C, C);
}

// Temperature (deg C) and solid fraction from the enthalpy temperature `te`: heat content
// over heat capacity, counted so that it equals the temperature in the liquid. Above the
// liquidus m.y they are the same; across the melting range the latent heat `m.z` (degrees)
// is given up; below the solidus m.x, te runs behind the temperature by it.
vec2 phase_of(float te, vec4 m) {
    if (te >= m.y) return vec2(te, 0.0);
    float lo = m.x - m.z;
    if (te <= lo) return vec2(te + m.z, 1.0);
    float fr = (te - lo) / (m.y - lo);
    return vec2(m.x + fr * (m.y - m.x), 1.0 - fr);
}

// Shear rate sqrt(2 D:D) at the centre of cell i from the face velocities, live faces only.
// `pre`: the velocities before the solve, which hold this substep's gravity increment on v
// faces; it is taken back out (a uniform pull is not a deformation, but it would read as
// shear against closed faces).
float shear_rate(int i, bool pre) {
    int W = P.grid.x;
    float gv = pre ? P.gravity * S.dt : 0.0;
    #define UF(k) (face_live(vel[k].z) ? vel[k].x : 0.0)
    #define VF(k) (face_live(vel[k].w) ? vel[k].y + gv : 0.0)
    float dxx = UF(i + 1) - UF(i);
    float dyy = VF(i + W) - VF(i);
    float duy = 0.25 * (UF(i + W) + UF(i + 1 + W) - UF(i - W) - UF(i + 1 - W));
    float dvx = 0.25 * (VF(i + 1) + VF(i + 1 + W) - VF(i - 1) - VF(i - 1 + W));
    #undef UF
    #undef VF
    float dxy = 0.5 * (duy + dvx);
    return sqrt(2.0 * (dxx * dxx + dyy * dyy + 2.0 * dxy * dxy));
}

// Non-Newtonian response of cell i (rh: share f, row, concentration) at shear rate g, given
// the viscosity mu the cell would have if Newtonian: the factor F on mu (README,
// "Rheology") and the yield stress (the material's, times its concentration, plus a skin
// at the free surface), both for the share f.
void nn_eval(int i, vec4 q, float g, out float F, out float tau) {
    int r = int(q.y);
    vec4 rp = mats[r].rheo;  // yield stress, flow index n, shear time lambda, ratio
    int model = mats[r].ib.y;
    F = 1.0;
    tau = 0.0;
    if (model == 1) {
        F = rp.w + (1.0 - rp.w) * pow(1.0 + (rp.z * g) * (rp.z * g), 0.5 * (rp.y - 1.0));
    } else if (model == 2) {
        F = pow(max(g, 1e-6), rp.y - 1.0);
        tau = rp.x * q.z;
    } else if (model == 3) {
        F = min(pow(1.0 + (rp.z * g) * (rp.z * g), 0.5 * (rp.y - 1.0)), rp.w);
    }
    int W = P.grid.x;
    float skin = mats[r].temp.w;
    if (skin > 0.0 && (cell[i - 1].w <= 0.0 || cell[i + 1].w <= 0.0 || cell[i - W].w <= 0.0 || cell[i + W].w <= 0.0)) tau += skin;
    F = 1.0 + q.x * (F - 1.0);
    tau *= q.x;
}

// Share of a concentrate in the logarithmic viscosity blend with its diluent: s itself
// unless a mixing override (material.rs: MIX_OVERRIDES) gives an exponent. (pow(s, 1) is
// not exactly s on the GPU.)
float blend_share(float s, float e) { return e == 1.0 ? s : pow(s, e); }

// ln kinematic viscosity (cells^2 / s, zero shear) of material `row` at concentration s and
// temperature T (deg C; ignored in an isothermal tank, whose table has it folded in).
// Accumulator channel of the temperature sum of phase group g's particles alone (-1: none;
// the first group in the tank takes what the others leave of the total, heat2.z).
int group_heat_chan(int g) {
    return int((P.group_heat >> (8u * uint(g))) & 255u) - 1;
}

// Whether the materials of phase group g sort themselves by density below the grid's scale
// (the drift within a group, particles.comp): g is not the aqueous group, holds two or more
// materials in this tank, and none of them is a concentrate or a diluent (families mix
// instead: their exchange in particles.comp evens out the concentration).
bool group_sorts(int g) {
    if (g <= 0) return false;
    int n = 0;
    for (int j = 0; j < P.n_present; j++) {
        int r = present_row(j);
        ivec4 ia = mats[r].ia;
        if (ia.y >= 0 && mats[ia.y].ia.x == g) return false;
        if (ia.x != g) continue;
        if (ia.y >= 0) return false;
        n++;
    }
    return n >= 2;
}

// The temperature of each phase group present in the cell at accumulator offset k (cw: its
// particle weight), from its own heat channel and heat capacity Cg (share-weighted, as the
// cell's total Csh); the first present group without a channel takes the rest of the total.
// Liquids of different groups meet at a sharp interface and keep their own temperatures, so
// neither the solver (grid_prep.comp) nor the look (cell_optics.comp) may evaluate lava that
// touches water at the mean of the two. T comes in as the fallback and leaves as the mean.
void group_temperatures(int k, int cw, float Csh, float Cg[GROUPS], inout float T, out float Tg[GROUPS]) {
    for (int g = 0; g < GROUPS; g++) Tg[g] = T;
    if (Csh <= 0.0) return;
    int Htot = acc[k + int(P.heat2.z)];
    T = float(Htot) * T_SCALE / (Csh * float(cw)) - 273.15;
    for (int g = 0; g < GROUPS; g++) Tg[g] = T;
    int rest = Htot;
    int first = -1;
    for (int g = 0; g < GROUPS; g++) {
        int ch = group_heat_chan(g);
        if (ch >= 0) {
            int h = acc[k + ch];
            rest -= h;
            if (Cg[g] > 0.0) Tg[g] = float(h) * T_SCALE / (Cg[g] * float(cw)) - 273.15;
        } else if (first < 0 && (P.group_mask & (1u << uint(g))) != 0u) {
            first = g;
        }
    }
    if (first >= 0 && Cg[first] > 0.0) Tg[first] = float(rest) * T_SCALE / (Cg[first] * float(cw)) - 273.15;
}

float row_lnu(uint row, float s, float T, bool thermal) {
    float l = mats[row].a.y + (thermal ? temp_law(int(row), T) : 0.0);
    int d = mats[row].ia.y;
    if (d >= 0) l = mix(mats[d].a.y + (thermal ? temp_law(d, T) : 0.0), l, blend_share(s, mats[row].misc.z));
    return l;
}

// Quadratic B-spline weights for a point at fractional offset f in [0.5, 1.5) from its base node.
void bspline(vec2 f, out vec2 w[3]) {
    w[0] = 0.5 * (1.5 - f) * (1.5 - f);
    w[1] = 0.75 - (f - 1.0) * (f - 1.0);
    w[2] = 0.5 * (f - 0.5) * (f - 0.5);
}
