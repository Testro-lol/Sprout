#version 460

// The wet map: one texel per cell, remembering which liquid last lay against a solid there
// and when. The composite pass draws solids wet from it, drying with time (each material's
// own drying time, from the optics table). This pass only writes the cells that are wet
// now; every other texel keeps what it had.

layout(std430, set = 0, binding = 0) readonly buffer CellBuf { vec4 cell[]; };  // x: liquid occupancy
layout(std430, set = 0, binding = 1) readonly buffer CompBuf { vec4 comp[]; };  // z: concentration of family 0
layout(std430, set = 0, binding = 2) readonly buffer SdfBuf { float sdf[]; };
layout(std430, set = 0, binding = 3) readonly buffer SPropBuf { vec4 sprop[]; }; // w: group + (row + 1) / 128

layout(push_constant) uniform PC {
    ivec2 grid;
    float now;   // simulated time, s
} pc;

// The material (table row + 1) with the largest share, family 0's concentration (-1: none),
// and the time. The composite pass reads the texels unfiltered.
layout(location = 0) out vec4 f_wet;

// A cell this close to a solid (cells, at its centre) lies against it.
const float REACH = 1.75;
// Occupancy (1 = rest) above which a cell wets what it touches.
const float WETTING = 0.3;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    int i = c.y * pc.grid.x + c.x;
    int w1 = pc.grid.x + 1;
    int k = c.y * w1 + c.x;
    float d = 0.25 * (sdf[k] + sdf[k + 1] + sdf[k + w1] + sdf[k + w1 + 1]);
    if (d > REACH || cell[i].x < WETTING) discard;
    float code = floor(fract(sprop[i].w) * 128.0 + 0.5);
    if (code < 1.0) discard;
    f_wet = vec4(code, comp[i].z, pc.now, 1.0);
}
