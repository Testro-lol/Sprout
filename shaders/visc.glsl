// Shared by the viscosity kernels.
//
// Implicit viscosity, backward Euler, in the variational form of Batty & Bridson (2008):
// the new face velocities u minimise
//     sum_faces  rho_f / 2 (u - u*)^2  +  dt * sum  mu (D:D)
// with D = (grad u + grad u^T) / 2 the strain rate. That is the full stress tensor
// div(mu (grad u + grad u^T)), not the Laplacian mu lap(u): it conserves angular momentum
// (a rigid rotation is not damped), gives traction-free free surfaces and is consistent
// across jumps in viscosity, such as between water and honey.
//
// Discretisation: D_xx, D_yy at cell centres (weight 2 dt mu), D_xy at cell corners (weight
// 4 dt mu V, V the liquid share of the four cells around). Samples touching air are dropped
// (zero traction there). At a solid the wall-normal face is closed (velocity 0) and a
// tangential face inside the solid mirrors its fluid partner (-u), which puts the no-slip
// wall half a cell from the first velocity sample, where the wall is. The system
// A = M + G^T W G is symmetric positive definite and solved with conjugate gradients,
// preconditioned by a multigrid V-cycle (vmg.glsl, `viscosity_precond` 1) or the diagonal
// (Jacobi, 0); every step of it can only remove kinetic energy.

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform VPC {
    uint mode;
    uint half_off;   // offset of the second half of the partial buffer
} V;

bool live_u(int i) { return face_live(vel[i].z); }
bool live_v(int i) { return face_live(vel[i].w); }
bool closed_u(int i) { return face_closed(vel[i].z); }
bool closed_v(int i) { return face_closed(vel[i].w); }
bool in_grid(ivec2 c) { return all(greaterThanEqual(c, ivec2(0))) && all(lessThan(c, P.grid)); }
bool interior(ivec2 c) { return all(greaterThanEqual(c, ivec2(1))) && all(lessThan(c, P.grid - 1)); }

// Tiles are the 16 x 16 work groups of the grid kernels.
uint tile_of(ivec2 c) { return uint((c.y >> 4) * ((P.grid.x + 15) >> 4) + (c.x >> 4)); }
bool tile_active() { return tiles[gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x] != 0u; }

// Below the viscous diffusion number nu dt / dx^2 in P.corr.w (params.rs:
// implicit_viscosity_min, default 1e-3) a cell needs no implicit solve: the explicit step
// with the same operator is accurate there (its stability limit is ~0.12). Pure water on
// 1 cm cells sits near 1e-5.

// Weighted strain W G u of the current velocities at cell i, with its weights w (vw[i].xy):
// D_xx, D_yy at the cell centre (x, y), D_xy at its lower-left node (z).
float vel_u(int i) { return live_u(i) ? vel[i].x : 0.0; }
float vel_v(int i) { return live_v(i) ? vel[i].y : 0.0; }
vec4 strain_of_vel(int i, vec2 w) {
    int W = P.grid.x;
    vec4 o = vec4(0.0);
    if (w.x > 0.0) {
        o.x = w.x * (vel_u(i + 1) - vel_u(i));
        o.y = w.x * (vel_v(i + W) - vel_v(i));
    }
    if (w.y > 0.0) {
        float ua = vel_u(i), ub = vel_u(i - W), va = vel_v(i), vb = vel_v(i - 1);
        if (closed_u(i) && live_u(i - W)) ua = -ub;
        if (closed_u(i - W) && live_u(i)) ub = -ua;
        if (closed_v(i) && live_v(i - 1)) va = -vb;
        if (closed_v(i - 1) && live_v(i)) vb = -va;
        o.z = w.y * 0.5 * ((ua - ub) + (va - vb));
    }
    return o;
}
