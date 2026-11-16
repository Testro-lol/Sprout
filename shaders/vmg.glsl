// Shared by the kernels of the viscosity multigrid preconditioner (`viscosity_precond` 1):
// vmg_build.comp, vmg_down.comp, vmg_up.comp.
//
// The unknowns of the viscosity solve are the u and v faces of each cell. The matrix of
// level 0 is A = M + G^T W G itself (visc_diag.comp writes it out per cell); the coarser
// levels aggregate 2 x 2 cells per component (u with u, v with v), with piecewise-constant
// prolongation P, restriction P^T and the Galerkin product P^T A P. The stencil keeps its
// shape under that coarsening: u couples to u left, right, below and above, v likewise, and
// the u face of a cell to the v faces of the cells at (0, 0), (-1, 0), (0, 1) and (-1, 1).
// Each coupling is stored once (VmABuf, VmBBuf), so every level's matrix is exactly
// symmetric.
//
// The smoother is red-black Gauss-Seidel over the cells, the u and v faces of a cell
// updated together (between the two, and between the diagonal neighbours of one colour that
// the u-v coupling links, it is a Jacobi step). Pre-smoothing is red then black, from zero;
// post-smoothing black then red, the mirror image, so the V-cycle is a symmetric positive
// definite preconditioner, as conjugate gradients requires (the u-v coupling within a
// colour stays far below the diagonal). The coarsest level of the cycle gets the one
// symmetric sweep of its two visits and no exact solve: the mass term makes coarse levels
// well conditioned, and two or three levels below the grid do as well as a full hierarchy
// for diffusion numbers up to ~100 (README, Performance).
//
// Each visit of a level smooths NU times (specialisation constant: a red and a black
// half-sweep each). The down and up kernels hold the tile and 2 NU cells around it in
// shared memory: every half-sweep after the first works on a ring one cell narrower, down to
// the tile itself for the residual (down) or the last half-sweep (up). The cells around the
// tile are computed again by each tile that needs them, so more sweeps cost arithmetic in
// shared memory, not dispatches.
//
// Conjugate gradients with this preconditioner (mod.rs: record_viscosity_solve) takes four
// dispatches an iteration with one coarse level: visc_mgu.comp (the update, then the way down
// on level 0), the coarse level down and up, and visc_mgz.comp (the way up on level 0, which
// also forms A z and the dot products for both step lengths).

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform VmPC {
    uint level;
    uint mode;   // bit 0: coarsest level of the cycle; bits 4..7 (visc_mgz): cg_reduce mode
} VM;

ivec4 LV;   // lev[VM.level]: offset, width, height, tile-flag offset

int at_l(ivec4 L, ivec2 c) { return L.x + c.y * L.y + c.x; }
bool inside_l(ivec4 L, ivec2 c) { return all(greaterThanEqual(c, ivec2(0))) && c.x < L.y && c.y < L.z; }
bool coarsest() { return (VM.mode & 1u) != 0u; }

// Tile flags of level l, in its own 16 x 16 tiles: level 0 is the viscosity tile buffer.
uint tile_flag(uint l, uint slot) { return l == 0u ? tiles[slot] : vtile[uint(lev[l].w) + slot]; }

layout(constant_id = 3) const uint NU = 1u;
const int HALO = 2 * int(NU);
// One more ring of cells for a kernel that applies A to the smoothed values (visc_mgz.comp).
layout(constant_id = 4) const uint EXTRA = 0u;
const int RH = HALO + int(EXTRA);   // cells around the tile in the region
const int RW = 16 + 2 * RH;
const int PER = (RW * RW + 255) / 256;   // region cells per invocation
shared vec4 sa[RW * RW];
shared vec4 sb[RW * RW];
shared vec2 sd[RW * RW];
shared vec2 sx[RW * RW];
shared vec2 sr[RW * RW];

ivec2 O;   // first cell of the tile

ivec2 region_cell(int k) { return O - RH + ivec2(k % RW, k / RW); }
bool red_cell(ivec2 c) { return ((c.x + c.y) & 1) == 0; }

// Off-diagonal part of A x at region slot k (x in sx); needs the slots around k.
vec2 offd(int k) {
    vec4 a = sa[k], b = sb[k];
    vec2 xl = sx[k - 1], xr = sx[k + 1], xd = sx[k - RW], xu = sx[k + RW];
    float u = a.x * xl.x + sa[k + 1].x * xr.x + a.y * xd.x + sa[k + RW].y * xu.x
            + b.x * sx[k].y + b.y * xl.y + b.z * xu.y + b.w * sx[k + RW - 1].y;
    float v = a.z * xl.y + sa[k + 1].z * xr.y + a.w * xd.y + sa[k + RW].w * xu.y
            + b.x * sx[k].x + sb[k + 1].y * xr.x + sb[k - RW].z * xd.x + sb[k + 1 - RW].w * sx[k + 1 - RW].x;
    return vec2(u, v);
}

// One Gauss-Seidel update of slot k from the current neighbours: (r - offd) / diag, zero
// where a face is no unknown.
vec2 relax(int k) {
    vec2 d = sd[k];
    vec2 s = sr[k] - offd(k);
    return vec2(d.x > 0.0 ? s.x / d.x : 0.0, d.y > 0.0 ? s.y / d.y : 0.0);
}

// Loads the matrix of the region (zero outside the level) and its right-hand side: vr on
// level 0, vmr on the others; RHS_OF may define another (visc_mgu.comp, visc_mgz.comp).
void load_region(uint t) {
    for (int k = int(t); k < RW * RW; k += 256) {
        ivec2 c = region_cell(k);
        vec4 a = vec4(0.0), b = vec4(0.0);
        vec2 d = vec2(0.0), r = vec2(0.0);
        if (inside_l(LV, c)) {
            int i = at_l(LV, c);
            a = vma[i];
            b = vmb[i];
            d = vmd[i];
#ifdef RHS_OF
            r = RHS_OF(i);
#else
            r = VM.level == 0u ? vr[i] : vmr[i];
#endif
        }
        sa[k] = a;
        sb[k] = b;
        sd[k] = d;
        sr[k] = r;
    }
}

// Region slots [lo, RW - lo) in each direction: 0 the whole region, RH the tile.
bool in_ring(int k, int lo) {
    ivec2 l = ivec2(k % RW, k / RW);
    return all(greaterThanEqual(l, ivec2(lo))) && all(lessThan(l, ivec2(RW - lo)));
}

// A half-sweep over the cells of one colour within [lo, RW - lo): every update reads the
// values from before the half-sweep, then all are written.
void half_sweep(uint t, bool red, int lo) {
    vec2 nv[PER];
    for (int j = 0; j < PER; j++) {
        int k = int(t) + 256 * j;
        nv[j] = vec2(0.0);
        if (k < RW * RW && in_ring(k, lo) && red_cell(region_cell(k)) == red) nv[j] = relax(k);
    }
    barrier();
    for (int j = 0; j < PER; j++) {
        int k = int(t) + 256 * j;
        if (k < RW * RW && in_ring(k, lo) && red_cell(region_cell(k)) == red) sx[k] = nv[j];
    }
    barrier();
}

int tile_slot(uvec2 lc) { return (int(lc.y) + RH) * RW + int(lc.x) + RH; }

vec2 unknowns(vec2 d) { return vec2(d.x > 0.0 ? 1.0 : 0.0, d.y > 0.0 ? 1.0 : 0.0); }

#ifdef VMG_DOWN
// The way down on the loaded region (EXTRA 0): pre-smoothing from zero, NU sweeps red then
// black, the tile's values to vmx and its residual r - A x, summed over each 2 x 2 aggregate,
// to the right-hand side of the next level (none on the coarsest level of the cycle).
shared vec2 sres[256];
void way_down(uint t) {
    for (int k = int(t); k < RW * RW; k += 256) {
        vec2 d = sd[k], r = sr[k];
        vec2 x = vec2(d.x > 0.0 ? r.x / d.x : 0.0, d.y > 0.0 ? r.y / d.y : 0.0);
        sx[k] = red_cell(region_cell(k)) ? x : vec2(0.0);
    }
    barrier();
    for (int h = 1; h < HALO; h++) half_sweep(t, (h & 1) == 0, h);   // black first

    uvec2 lc = gl_LocalInvocationID.xy;
    ivec2 c = O + ivec2(lc);
    int k = tile_slot(lc);
    vec2 x = sx[k];
    vec2 d = sd[k];
    vec2 res = (sr[k] - d * x - offd(k)) * unknowns(d);
    if (inside_l(LV, c)) vmx[at_l(LV, c)] = x;
    if (!coarsest()) {   // (the same for the whole work group)
        sres[t] = res;
        barrier();
        ivec4 LC = lev[VM.level + 1u];
        ivec2 C = c >> 1;
        if (((lc.x | lc.y) & 1u) == 0u && inside_l(LC, C)) vmr[at_l(LC, C)] = sres[t] + sres[t + 1u] + sres[t + 16u] + sres[t + 17u];
    }
}
#endif
