// Shared by the multigrid and Krylov kernels of the pressure solve.
//
// Every level is a W_l x H_l grid stored back to back in mga / mgd / mgx / mgr at offset
// off_l (level 0 is the pressure grid itself). A cell is an unknown when its diagonal is
// positive; air, solid and empty coarse cells have mgd = 0.

layout(push_constant) uniform MgPC {
    int off_f;     // offset of the (fine) level this dispatch works on
    int off_c;     // offset of the next coarser level
    ivec2 dim_f;
    ivec2 dim_c;
    uint mode;     // kernel-specific flag
    float omega;   // coarse-correction weight
} M;

int fidx(ivec2 c) { return M.off_f + c.y * M.dim_f.x + c.x; }
int cidx_c(ivec2 c) { return M.off_c + c.y * M.dim_c.x + c.x; }
bool inside_f(ivec2 c) { return all(greaterThanEqual(c, ivec2(0))) && all(lessThan(c, M.dim_f)); }
bool unknown_f(ivec2 c) { return inside_f(c) && mgd[fidx(c)] > 0.0; }

const ivec2 NB[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));

float wcomp(vec4 w, int k) { return k == 0 ? w.x : k == 1 ? w.y : k == 2 ? w.z : w.w; }

#include "reduce.glsl"
