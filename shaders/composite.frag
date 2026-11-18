#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : enable
#include "optics.glsl"

// Shades the liquid from the splat accumulation and its blurred copy, and draws the solids
// and the backdrop. (The brush cursor is an egui overlay, see ui.rs.)
//
// The liquid is drawn from accumulated optical properties (render.rs: ACCUM_TARGETS), with
// no code for any particular material:
//   accum  the weight of each phase group, with a kernel of about a cell. The level sets of
//          the total are the surfaces; where the leading group changes is an interface, the
//          only other place a sharp line is drawn (miscible liquids blend softly).
//   wide   the same blurred over WIDE_SIGMA cells. Its total says how much liquid is around
//          a point (0.5 at a flat surface, 1 deep inside, little in a droplet or a film).
//   colour, optics, emission, style: body colour (OKLab with chroma), opacity, scattering,
//          metal, gloss, glow, speed, body, foam, drying time.
// Everything below is a continuous function of these, so nothing can flicker or pop that
// the fields themselves do not.

layout(set = 0, binding = 0) uniform sampler2D accum;
layout(std430, set = 0, binding = 1) readonly buffer SdfBuf { float sdf[]; };
layout(set = 0, binding = 2) uniform sampler2D wide;
// One texel per cell: which liquid last lay against a solid there, and when (wet.frag).
layout(set = 0, binding = 3) uniform sampler2D wet_map;
// Per cell, what the liquid there looks like (cell_optics.comp): one texel per cell in each of
// five layers, each times the liquid's occupancy: aqueous colour, other colour, (opacity,
// scattering, metal, gloss), (emission, speed), (body, foam, aqueous share, total).
layout(set = 0, binding = 4) uniform sampler2DArray cellopt;
layout(std430, set = 0, binding = 8) readonly buffer OptBuf { Optics optics[]; };
// The glow field (glow.comp), half the grid's resolution: emission (linear) and heat.
layout(set = 0, binding = 5) uniform sampler2D glow_map;
// The film map (film.comp; README, "Stains and runs"), FILM_RES texels per cell, two copies
// (layers; pc.film_layer holds it now): colour (sRGB), and thickness (0.1 mm), wetness,
// drying time, mobility.
layout(set = 0, binding = 6) uniform sampler2DArray film_c;
layout(set = 0, binding = 10) uniform sampler2DArray film_h;
// The vapour field (steam.vert), laid out as `accum`: bubbles of steam in the liquid (r),
// their froth walls (b).
layout(set = 0, binding = 7) uniform sampler2D vapour_map;
// The smoke field (smoke.comp), two texels per cell: the vapour of escaped steam (r).
layout(set = 0, binding = 9) uniform sampler2D smoke_map;
// The flow map (flow_map.comp), one texel per cell: how far the liquid has moved since its
// patterns restarted, two copies (xy, zw) for each of two periods; layer 2 pc.flow_layer +
// 0 the short period, + 1 the long.
layout(set = 0, binding = 11) uniform sampler2DArray flow_map;
// The tint field (splat.vert, tint.frag): how the particles absorb and scatter, at half the
// resolution of `accum` and laid out as it (the same uv): the (aqueous: concentrate part's)
// absorption and the scattering; the aqueous concentrate share, the aqueous weight and the
// others', times the kernel (optics.glsl: particle_absorb).
layout(set = 0, binding = 12) uniform sampler2D tint_c;
layout(set = 0, binding = 13) uniform sampler2D tint_s;
// The look field (look.vert), laid out as `accum`: what the particles of liquids with a
// carried look lay down, times the kernel: their platelets' sheen, their carried scalar phi
// (look.comp), which way their platelets lie in the plane, and their weight.
layout(set = 0, binding = 14) uniform sampler2D look_map;

layout(push_constant) uniform PC {
    vec2 cam;         // world position at the centre of the view, cells
    vec2 half_res;    // half the target size, pixels
    ivec2 grid;
    float inv_zoom;   // cells per pixel
    float threshold;  // accumulated weight that counts as liquid
    vec2 wide_scale;  // pixel -> uv of the wide field
    vec2 wide_max;    // last used uv of the wide field
    float rest;       // accumulated weight deep inside liquid at rest
    float mist;       // 0..1: how far the kernel is below a pixel (droplets then draw as haze)
    vec2 accum_scale; // pixel -> uv of the accumulation target
    vec2 accum_max;   // last used uv of the accumulation target
    float accum_div;  // pixels per texel of the accumulation target
    float now;        // simulated time, s
    float wet_width;  // how far the wet film reaches into a solid, cells (Display settings)
    float wet_tint;   // 0..1: how strongly a wet solid takes the liquid's colour
    float glow_k;     // Display: glow (bloom and the light hot liquid throws), 0 off
    float stain_k;    // Display: how strongly stains show, 0 off
    float vapour;     // 1 when the tank has steam (the vapour field is drawn), else 0
    float film_layer; // the copy of the film map that holds it now (0 or 1)
    float flow_layer; // the copy of the flow map that holds it now (0 or 1)
    float tint;       // 1 when the tint field is drawn (two aqueous liquids or more), else 0
    float look;       // 1 when the look field is drawn (a liquid with a carried look), else 0
    float sss;        // 1 when the light under the surface is solved (sss.comp: cell optics layer 20)
} pc;

layout(location = 0) out vec4 f_color;

// 0 compiles the vapour field out (render.rs: composite variants, for a tank without steam).
layout(constant_id = 0) const uint HAS_VAPOUR = 1u;
// What else the tank can show (render.rs: `Features`): bit 0 the carried patterns (lava's
// crackle, oil's iridescence),
// bit 1 stains (the film map), bit 2 ferrofluid spikes (a magnet and a magnetisable
// liquid; cell optics layer 13). A variant
// without one compiles its code out: the code alone cost about 0.1 ms a 1600 x 900 frame
// where it never ran. Leaving a bit out is exact where no liquid in the tank has the effect:
// its weight is then zero in every cell, and the film map is empty until a staining liquid
// has been in the tank.
layout(constant_id = 1) const uint FX = 3u;
const bool FX_PATTERNS = (FX & 1u) != 0u;
const bool FX_FILM = (FX & 2u) != 0u;
const bool FX_SPIKES = (FX & 4u) != 0u;

// Light comes from the upper left; world space, +y up.
const vec2 LIGHT = vec2(-0.45, 0.893);
// render.rs: WIDE_SIGMA
const float WIDE_SIGMA = 2.0;
// Share of the other (non-aqueous) groups above which a point is drawn as that liquid.
const float OIL_CUT = 0.36;
// Most an aqueous liquid is tinted towards the other one by droplets too small to draw.
const float OIL_TINT = 0.18;
// Share of the other groups in the sharp field above which a dispersed one is drawn as a
// droplet: above the share one particle among aqueous ones reaches, so only clusters show.
const float DROP_CUT = 0.55;
// Share of one side in the sharp field above which a small droplet of it is drawn inside a
// body of the other side (lava in water, water in oil): below the peak share of a single
// particle, so that a droplet of one or two particles shows, with its own outline.
const float SPECK_CUT = 0.10;
// ...and at least this many times its share in the blurred field (what is around it).
const float SPECK_CONTRAST = 2.5;
// Accumulated weight at the surface of a body of liquid: thin liquids, thick ones.
const float BODY_CUT = 2.8;
const float BODY_CUT_THICK = 1.9;
// Vapour field weight at the wall of a bubble (a lone one peaks at 1; render.rs:
// BUBBLE_RADIUS), and the share of its glow field's spread (glow.comp: 3 half-texels, 6
// cells) over which the glow falls off beyond the grid.
const float BUBBLE_CUT = 0.45;
const float GLOW_FALL = 6.0;

// ---- light in the liquid (README, "Look") -------------------------------------------------
// Every liquid is drawn from how it absorbs and scatters (optics.glsl), lit by:
//   KEY        the light from the upper left, which reaches a point in the liquid through the
//              liquid above it (the light depth, cell_optics.comp) and is coloured on the way:
//              water turns from pale aqua under its surface to deep blue at the floor, honey
//              from gold to dark amber, a pool of blood is near black under a scarlet skin;
//   AMB_FRONT  soft light on the face of the liquid towards the viewer: what a scattering
//              liquid (milk, mud, blood, chocolate) shows its own colour by;
//   IN_SCAT    the share of the key light a clear liquid sends towards the viewer (its slight
//              turbidity, and the lit back of the tank it stands in front of).
// Mixtures add their coefficients, so they take the colour real ones do.
const float KEY = 0.95;
const float AMB_FRONT = 0.66;
const float IN_SCAT = 0.6;
const float AMB_DEEP = 0.18;  // light from the front that a clear liquid sends back however deep
const float TURBID = 0.06;    // per cell: how much a clear liquid scatters (a film or a droplet of
                              // it is nearly transparent; a body fills with its colour)
const float LV_BODY = 16.0;   // cells: thickness along the view of a body of liquid
const float EDGE_ROUND = 3.0; // cells: over which a body's thickness along the view rounds off at its outline
const float SURFACE_PATH = 0.6; // light through the nearest surface, per cell in from it
const float RIND = 1.6;       // cells: width of the quenched skin of a melt as drawn
const vec3 L3 = vec3(-0.3375, 0.67, 0.66);  // the key light in 3D (towards the viewer, z)

float sample_sdf(vec2 x) {
    vec2 g = clamp(x, vec2(0.0), vec2(pc.grid) - 0.001);
    ivec2 i = ivec2(g);
    vec2 f = g - vec2(i);
    int w1 = pc.grid.x + 1;
    int k = i.y * w1 + i.x;
    float d = mix(mix(sdf[k], sdf[k + 1], f.x), mix(sdf[k + w1], sdf[k + w1 + 1], f.x), f.y);
    // Outside the grid, fall back to the distance to it so the walls extend outwards.
    vec2 o = max(max(-x, x - vec2(pc.grid)), vec2(0.0));
    return d - length(o);
}

// The sharp field at a pixel position.
vec4 field(vec2 p) { return textureLod(accum, min(p * pc.accum_scale, pc.accum_max), 0.0); }
// The vapour field at a pixel position.
vec4 vapour(vec2 p) { return textureLod(vapour_map, min(p * pc.accum_scale, pc.accum_max), 0.0); }

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 lighten(vec3 c, float a) { return mix(c, vec3(1.0), a); }

// Darker and a little more saturated, as a deeper layer of the same liquid.
vec3 deepen(vec3 c, float a) {
    vec3 d = c * (1.0 - a);
    return max(mix(vec3(luma(d)), d, 1.0 + 0.5 * a), 0.0);
}

// The same colour, brighter and more saturated: light that has come through a thin edge
// of a coloured liquid, or the warm rim of a thick one.
vec3 glowup(vec3 c) {
    vec3 s = clamp(mix(vec3(luma(c)), c, 1.45), 0.0, 1.0);
    return lighten(s, 0.28);
}

// The cell optics at a world position, bilinear between cell centres (the sampler's filter;
// clamped to the edge cells outside the grid).
// (ot: the oil-like group; ds: the dense groups, metals and melts; dn: the dense groups'
// share and frozen share.)
struct CellOptics { vec4 aq, ot, opt, em, sty, fz, fx, l9, ds; vec2 dn; };
CellOptics cell_optics(vec2 x) {
    vec2 uv = x / vec2(pc.grid);
    return CellOptics(textureLod(cellopt, vec3(uv, 0.0), 0.0), textureLod(cellopt, vec3(uv, 1.0), 0.0),
                      textureLod(cellopt, vec3(uv, 2.0), 0.0), textureLod(cellopt, vec3(uv, 3.0), 0.0),
                      textureLod(cellopt, vec3(uv, 4.0), 0.0), textureLod(cellopt, vec3(uv, 7.0), 0.0),
                      textureLod(cellopt, vec3(uv, 8.0), 0.0), textureLod(cellopt, vec3(uv, 9.0), 0.0),
                      textureLod(cellopt, vec3(uv, 11.0), 0.0),
                      vec2(textureLod(cellopt, vec3(uv, 5.0), 0.0).a, textureLod(cellopt, vec3(uv, 6.0), 0.0).a));
}
CellOptics mix_optics(CellOptics a, CellOptics b, float t) {
    return CellOptics(mix(a.aq, b.aq, t), mix(a.ot, b.ot, t), mix(a.opt, b.opt, t), mix(a.em, b.em, t),
                      mix(a.sty, b.sty, t), mix(a.fz, b.fz, t), mix(a.fx, b.fx, t), mix(a.l9, b.l9, t),
                      mix(a.ds, b.ds, t), mix(a.dn, b.dn, t));
}

// The tint field read with a cubic B-spline (four bilinear taps) instead of bilinearly: its
// texels are 2 - 4 pixels, its kernels only one and a half texels wide when zoomed out, and
// the sharp cut on the share below (own_aq, own_ot) followed bilinear's texel grid: blood in
// honey had an edge in pixel steps until zoomed in.
vec4 tint_read(sampler2D t, vec2 uv) {
    vec2 ts = vec2(textureSize(t, 0));
    vec2 p = uv * ts - 0.5;
    vec2 i = floor(p), f = p - i;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    vec2 w3 = f3 / 6.0;
    vec2 g0 = w0 + w1, g1 = w2 + w3;
    vec2 h0 = (i - 0.5 + w1 / g0) / ts, h1 = (i + 1.5 + w3 / g1) / ts;
    return g0.y * (g0.x * textureLod(t, h0, 0.0) + g1.x * textureLod(t, vec2(h1.x, h0.y), 0.0))
         + g1.y * (g0.x * textureLod(t, vec2(h0.x, h1.y), 0.0) + g1.x * textureLod(t, h1, 0.0));
}

// ---- which liquid where: the look key (README, "Look") ------------------------------------
// The cell optics hold a cell's liquids blended, smoothed over 3 x 3 cells: a strand of blood
// in water or honey, ferrofluid in oil or glass in lava (one phase group, so the sharp field
// sees one liquid) was drawn in the cells' shapes, a cell or two wider than the particles,
// in lumps and blocks. Each particle also lays down the key of its liquid (splat.vert), at
// the particles' resolution. Where the key under a pixel differs from the cells' there, the
// optics are read a little way off instead: along the direction in which the cells' key
// moves towards the particles', at the point of that path whose key comes closest to it
// (up to SEEK cells away). Inside a blob of blood the read moves inwards to where the cells
// hold blood, in the water beside it outwards to clear water; the edge between them is the
// particles'. The colours are still the cells' own, smoothed: only where they apply follows
// the particles. (Reading the cells before smoothing, weighted towards those whose key
// matched, drew their cell-to-cell noise as dark blocks along every edge.)
const float KEY_LO = 0.03, KEY_HI = 0.10;   // key mismatch over which the moved read takes over
const float SEEK = 1.6;                     // cells: farthest the read moves
// The particles' colour gives way to the cells' as the other side's absorption, taken out of
// it with the cells' coefficients, goes from the first to the second fraction of its own.
const vec2 TINT_TRUST = vec2(0.1, 0.4);

vec3 key_cell(vec2 x) {
    vec4 k = textureLod(cellopt, vec3(x / vec2(pc.grid), 12.0), 0.0);
    return k.rgb / max(k.a, 1e-4);
}

// Where to read the cell optics for particles of key `ks` at x, whose cells' key is `k0`.
vec2 seek_key(vec2 x, vec3 ks, vec3 k0) {
    // The direction in which the cells' key moves towards ks (each channel's gradient,
    // weighted by how far it has to go).
    vec3 dx = key_cell(x + vec2(0.5, 0.0)) - key_cell(x - vec2(0.5, 0.0));
    vec3 dy = key_cell(x + vec2(0.0, 0.5)) - key_cell(x - vec2(0.0, 0.5));
    vec3 want = ks - k0;
    vec2 dir = vec2(dot(dx, want), dot(dy, want));
    float dl = length(dir);
    if (dl < 1e-6) return x;
    dir /= dl;
    // Along x -> x + dir SEEK, the point whose key is closest to ks (the path through two
    // samples, projected segment by segment).
    vec3 k1 = key_cell(x + dir * (0.5 * SEEK)), k2 = key_cell(x + dir * SEEK);
    float best = 1e9, at = 0.0;
    vec3 ka[3] = vec3[3](k0, k1, k2);
    for (int j = 0; j < 2; j++) {
        vec3 seg = ka[j + 1] - ka[j];
        float tau = clamp(dot(ks - ka[j], seg) / max(dot(seg, seg), 1e-8), 0.0, 1.0);
        vec3 r = ka[j] + tau * seg - ks;
        float dd = dot(r, r);
        if (dd < best - 1e-6) { best = dd; at = (float(j) + tau) * 0.5 * SEEK; }
    }
    return x + dir * at;
}

// The colour of liquid of coefficients `k` (absorption rgb, scattering; per cell), linear RGB,
// where `il` of the key light reaches it, `thick` cells thick along the view where the view
// passes (`body` cells where its body is, for how much it scatters into the view), its face
// shaded by `shade`, with `back` (linear) behind it.
// The spectral shape of the scattering at this pixel (cell optics layer 16 and the means): fine
// particles scatter blue more than red (Rayleigh), so a thin cloud of them looks blue against
// the dark and passes orange light. Set in main() before any shading.
vec3 g_sr = vec3(1.0);

vec3 liquid_lin(vec4 k, vec3 il, float thick, float body, float shade, vec3 back) {
    vec3 sv = k.a * g_sr;
    vec3 ext = k.rgb + sv;
    vec3 tv = exp(-ext * thick);
    float hide = 1.0 - exp(-k.a * thick);
    // (With the light under the surface solved, sss.comp, more of it comes from the light in
    // the liquid: a body is lit where light reaches it and spreads, darker where it does not,
    // tinted as the light is.)
    vec3 front = pc.sss > 0.0 ? vec3(AMB_FRONT * 0.75 * shade) + 0.55 * KEY * il : vec3(AMB_FRONT * shade) + 0.35 * KEY * il;
    vec3 scat = deep_reflectance3(k.rgb, sv) * (1.0 - tv) * front;
    // (The key light is scattered towards the viewer half way through the layer; the soft
    // light from the front crosses all of it, so deep in a pool, where only that is left, a
    // clear liquid shows its colour at its fullest.)
    vec3 through = IN_SCAT * (KEY * il * exp(-k.rgb * 0.5 * thick) + AMB_DEEP * exp(-k.rgb * thick)) * (1.0 - hide) * (1.0 - exp(-TURBID * body));
    return scat + through + back * tv;
}

// The same in sRGB, with what lies behind it (sRGB) showing through by its transmission. (The
// back wall is added after the conversion: it is dark, and in linear light it vanished
// behind even a thin layer, so that clear liquids looked like paint.)
vec3 liquid_srgb(vec4 k, vec3 il, float thick, float body, float shade, vec3 back) {
    vec3 tv = exp(-(k.rgb + k.a * g_sr) * thick);
    return to_srgb(liquid_lin(k, il, thick, body, shade, vec3(0.0))) + back * tv * 0.85;
}

// Ice: the frozen colour (layer 5 aqueous, 6 the others) where the smoothed frozen share `f` of
// those liquids passes a half. A solid has an edge, not a fade: the step is a few pixels wide
// at any zoom, along a smooth curve (the share is smoothed over 3 x 3 cells first).
vec3 iced(vec3 c, vec2 x, float layer, float w, float f, float zoom) {
    if (f <= 0.02 || w <= 1e-4) return c;
    vec4 ice = textureLod(cellopt, vec3(x / vec2(pc.grid), layer), 0.0);
    float edge = clamp(0.5 / zoom, 0.04, 0.25);  // half width of the step, in frozen share (about 2 px)
    return mix(c, to_srgb(ice.rgb / w), smoothstep(0.5 - edge, 0.5 + edge, f));
}

float total(vec4 g) { return g.x + g.y + g.z + g.w; }
float hot_in(vec3 e) { return max(e.r, max(e.g, e.b)); }

// The leading group's share less the next one's, for the groups a, b chosen at this pixel:
// positive on a's side of an interface, zero on it.
float lead(vec4 g, int a, int b) { return (g[a] - g[b]) / max(total(g), 1e-4); }

// Colour and drying time of a wet-map texel (wet.frag): material row + 1, concentration.
vec4 film_of(vec4 m) {
    int row = int(m.x + 0.5) - 1;
    Optics o = optics[max(row, 0)];
    int d = o.idx.x;
    vec4 k = o.absorb;
    float dry = o.sty.z;
    if (d >= 0) {
        float s = m.y >= 0.0 ? clamp(m.y, 0.0, 1.0) : 1.0;
        k = mix(optics[d].absorb, k, s);
        dry = mix(optics[d].sty.z, dry, s);
    }
    return vec4(film_colour(k, 0.35), dry);
}

// The colour of the liquid in the cells at `x`, aqueous and others by their share there;
// `fallback` where those cells hold none. Solids read it just outside their surface: the
// cells inside a solid hold no liquid, and reading the colour there drew wet edges black in
// cell-sized steps.
vec3 liquid_at(vec2 x, vec3 fallback) {
    vec2 uv = x / vec2(pc.grid);
    vec4 sty = textureLod(cellopt, vec3(uv, 4.0), 0.0);
    if (sty.w < 1e-3) return fallback;
    vec3 a = film_colour(textureLod(cellopt, vec3(uv, 0.0), 0.0) / max(sty.z, 1e-4), 0.35);
    vec3 o = film_colour((textureLod(cellopt, vec3(uv, 1.0), 0.0) + textureLod(cellopt, vec3(uv, 11.0), 0.0)) / max(sty.w - sty.z, 1e-4), 0.35);
    vec3 c = mix(a, o, smoothstep(0.2, 0.6, (sty.w - sty.z) / sty.w));
    return mix(fallback, c, smoothstep(1e-3, 0.05, sty.w));
}

// Coverage of a band lo..hi (cells) of a distance d (cells), antialiased at `zoom` px/cell.
float band(float d, float lo, float hi, float zoom) {
    return clamp((d - lo) * zoom + 0.5, 0.0, 1.0) * clamp((hi - d) * zoom + 0.5, 0.0, 1.0);
}

// ---- texture: noise anchored to the world, or carried by the liquid ---------------------
// (Hoskins' hashes; gradient noise with a quintic fade.) Every pattern below is a smooth
// function of world position and simulated time, so it moves only as the liquid or the
// air does, and is faded out where its features would be smaller than about two pixels.

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Gradient noise, about 0..1, mean 0.5.
float gnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float va = dot(hash22(i) * 2.0 - 1.0, f);
    float vb = dot(hash22(i + vec2(1.0, 0.0)) * 2.0 - 1.0, f - vec2(1.0, 0.0));
    float vc = dot(hash22(i + vec2(0.0, 1.0)) * 2.0 - 1.0, f - vec2(0.0, 1.0));
    float vd = dot(hash22(i + 1.0) * 2.0 - 1.0, f - 1.0);
    return 0.5 + 0.9 * mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
}

// How far `freq` (features per cell) is from aliasing at this zoom: 1 when a feature
// spans 4 px or more, 0 below 2 px.
float detail(float freq, float zoom) { return smoothstep(2.0, 4.0, zoom / freq); }

// Gradient noise in three dimensions, about 0..1, mean 0.5: a 2D pattern that evolves in
// time (the third coordinate), rather than sliding.
// (An integer hash, PCG3D (Jarzynski & Olano 2020): a float hash of the corner's position
// gave two cells sharing a corner slightly different values, which the compiler rounds
// differently in each, and now and then one wrapped round and drew a seam.)
uvec3 pcg3d(ivec3 q) {
    uvec3 v = uvec3(q) * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
vec3 hash33(ivec3 q) { return vec3(pcg3d(q)) * (1.0 / 4294967295.0); }

// Two independent evolving noises from one evaluation: one hash's 96 bits are two
// gradients of 16 bits a component (lava's glow and its veins; half the cost of two).
vec2 gnoise3x2(vec3 p) {
    vec3 fl = floor(p), f = p - fl;
    ivec3 i = ivec3(fl);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 n[8];
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, k >> 2);
        uvec3 h = pcg3d(i + o);
        vec3 d = f - vec3(o);
        vec3 g1 = vec3(h & 0xffffu) * (2.0 / 65535.0) - 1.0;
        vec3 g2 = vec3(h >> 16u) * (2.0 / 65535.0) - 1.0;
        n[k] = vec2(dot(g1, d), dot(g2, d));
    }
    vec2 r = mix(mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y),
                 mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y), u.z);
    return 0.5 + 0.8 * r;
}
float gnoise3(vec3 p) {
    vec3 fl = floor(p), f = p - fl;
    ivec3 i = ivec3(fl);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    #define G(a, b, c) dot(hash33(i + ivec3(a, b, c)) * 2.0 - 1.0, f - vec3(a, b, c))
    float n = mix(mix(mix(G(0, 0, 0), G(1, 0, 0), u.x), mix(G(0, 1, 0), G(1, 1, 0), u.x), u.y),
                  mix(mix(G(0, 0, 1), G(1, 0, 1), u.x), mix(G(0, 1, 1), G(1, 1, 1), u.x), u.y), u.z);
    #undef G
    return 0.5 + 0.8 * n;
}

// A hue wheel: h in turns.
vec3 hue(float h) { return 0.5 + 0.5 * cos(6.2832 * (h + vec3(0.0, 0.33, 0.67))); }

// Seams of a cell pattern (Voronoi F2 - F1, in pattern cells): small on the seams.
float seams(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int b = -1; b <= 1; b++) {
        for (int a = -1; a <= 1; a++) {
            vec2 g = vec2(a, b);
            vec2 r = g + 0.15 + 0.7 * hash22(i + g) - f;
            float dd = dot(r, r);
            if (dd < d1) { d2 = d1; d1 = dd; } else if (dd < d2) d2 = dd;
        }
    }
    return sqrt(d2) - sqrt(d1);
}

// ---- patterns carried by the liquid (flow_map.comp) ----
// Each pattern is drawn at x - d, d how far the liquid there has moved since the pattern
// restarted, so it moves exactly as the liquid does. Two copies half a period apart, each
// faded out as it restarts (a flow map, Vlachos 2010). A copy that has moved shows a new
// part of the pattern from its restart on (it would have been sheared into streaks); one
// that has not moved shows the pattern at rest, the same in both copies, so liquid at rest
// keeps its pattern instead of morphing every few seconds. sim/mod.rs: FLOW_PERIOD_*.
const float PERIOD_SHORT = 4.0;
const float PERIOD_LONG = 12.0;

struct Carry {
    vec2 dA, dB;     // displacement of each copy, cells
    float w;         // weight of copy A (B: 1 - w)
    vec2 gen;        // generation of each copy (restarts so far)
    float stretch;   // areal stretch of the liquid since the restarts (div d, blended)
    float strain;    // how far it has been sheared or stretched since (|sym grad d|, blended)
};

vec4 flow_at(vec2 x, float k) { return textureLod(flow_map, vec3(x / vec2(pc.grid), 2.0 * pc.flow_layer + k), 0.0); }

Carry carry(vec2 x, bool long_period, bool strain) {
    float k = long_period ? 1.0 : 0.0;
    float period = long_period ? PERIOD_LONG : PERIOD_SHORT;
    vec4 d = flow_at(x, k);
    float ph = pc.now / period;
    Carry c;
    c.dA = d.xy;
    c.dB = d.zw;
    c.w = 1.0 - abs(2.0 * fract(ph) - 1.0);
    c.gen = vec2(floor(ph), floor(ph + 0.5));
    c.stretch = 0.0;
    c.strain = 0.0;
    if (strain) {
        vec4 gx = 0.5 * (flow_at(x + vec2(1.0, 0.0), k) - flow_at(x - vec2(1.0, 0.0), k));
        vec4 gy = 0.5 * (flow_at(x + vec2(0.0, 1.0), k) - flow_at(x - vec2(0.0, 1.0), k));
        c.stretch = mix(gx.z + gy.w, gx.x + gy.y, c.w);
        float sA = sqrt(gx.x * gx.x + gy.y * gy.y + 0.5 * (gx.y + gy.x) * (gx.y + gy.x));
        float sB = sqrt(gx.z * gx.z + gy.w * gy.w + 0.5 * (gx.w + gy.z) * (gx.w + gy.z));
        c.strain = mix(sB, sA, c.w);
    }
    return c;
}

// How far a copy has moved, in features of the pattern: 0 the pattern at rest, 1 a new one.
float moved(vec2 d, float freq) { return smoothstep(0.3, 1.2, length(d) * freq); }

// The pattern at rest is offset by `seed`; a moved copy's by its generation.
vec2 rest_off(float seed) { return vec2(13.7, 7.3) * seed; }
vec2 gen_off(float gen, float seed) { return hash22(vec2(gen, seed)) * 97.0 + 31.0; }

float pattern(vec2 q, float evolve) { return evolve > 0.0 ? gnoise3(vec3(q, pc.now * evolve)) : gnoise(q); }

// The (up to) four patterns behind a carried pattern: each copy's pattern at rest and its
// moved one. Their weights (summing to 1) and the variance of their blend come first; where
// each is read is worked out in the loop that reads it (kept as arrays, the four places held
// so many registers that the whole shader ran slower: 0.27 ms a frame at 1080p in a tank
// without a single pattern on screen). Where the two copies have moved alike, to a
// twentieth of a feature, their patterns at rest are one: read once, at the weighted
// displacement (most of a resting pool).
struct Slots { vec4 c; float var; vec2 dA; };
Slots slots(Carry k, float freq) {
    float mA = moved(k.dA, freq), mB = moved(k.dB, freq);
    Slots s;
    s.c = vec4(k.w * (1.0 - mA), k.w * mA, (1.0 - k.w) * (1.0 - mB), (1.0 - k.w) * mB);
    // (A blend of independent patterns is flatter than either; the two copies of the
    // pattern at rest are the same pattern where they have moved alike.)
    float same = 1.0 - smoothstep(0.25, 1.0, length(k.dA - k.dB) * freq);
    s.var = dot(s.c, s.c) + 2.0 * same * s.c.x * s.c.z;
    s.dA = k.dA;
    if (length(k.dA - k.dB) * freq < 0.05 && s.c.x + s.c.z > 0.0) {
        s.dA = mix(k.dB, k.dA, s.c.x / (s.c.x + s.c.z));
        s.c.x += s.c.z;
        s.c.z = 0.0;
    }
    return s;
}
// Where slot j is read.
vec2 slot_q(vec2 x, Carry k, Slots s, int j, float freq, float seed) {
    vec2 d = j < 2 ? s.dA : k.dB;
    vec2 off = (j & 1) == 0 ? rest_off(seed) : gen_off(j < 2 ? k.gen.x : k.gen.y + 0.5, seed);
    return (x - d) * freq + off;
}

// Gradient noise carried by the liquid (about 0..1, mean 0.5): `freq` features per cell,
// evolving at `evolve` features per second even where the liquid rests (0: still).
float carried_noise(vec2 x, Carry k, float freq, float seed, float evolve) {
    Slots s = slots(k, freq);
    float n = 0.0;
    [[dont_unroll]] for (int j = 0; j < 4; j++)
        if (s.c[j] > 0.0) n += s.c[j] * (pattern(slot_q(x, k, s, j, freq, seed), evolve) - 0.5);
    return clamp(0.5 + n / sqrt(max(s.var, 1e-4)), 0.0, 1.0);
}

// Two evolving noises carried by the liquid at once (as carried_noise, `evolve` > 0).
vec2 carried_noise2(vec2 x, Carry k, float freq, float seed, float evolve) {
    Slots s = slots(k, freq);
    float t = pc.now * evolve;
    vec2 n = vec2(0.0);
    [[dont_unroll]] for (int j = 0; j < 4; j++)
        if (s.c[j] > 0.0) n += s.c[j] * (gnoise3x2(vec3(slot_q(x, k, s, j, freq, seed), t)) - 0.5);
    return clamp(0.5 + n / sqrt(max(s.var, 1e-4)), 0.0, 1.0);
}

// The seams of a cell pattern carried by the liquid: 0 on a seam, 1 well inside a plate.
// `width` in pattern cells, but never under about a pixel and a half: thinner seams winked
// on and off as they drifted across pixels.
float carried_seams(vec2 x, Carry k, float freq, float width, float zoom, float seed) {
    width = max(width, 1.5 * freq / zoom);
    Slots s = slots(k, freq);
    float r = 0.0;
    [[dont_unroll]] for (int j = 0; j < 4; j++)
        if (s.c[j] > 0.0) r += s.c[j] * smoothstep(0.0, width, seams(slot_q(x, k, s, j, freq, seed)));
    return r;
}

// The glow field at a world position, filtered with a cubic B-spline (four bilinear taps):
// the field is at half the grid's resolution, and plain bilinear filtering of a bright,
// smooth glow drew the texels as a faint lattice of diamonds and creases. Where the
// bilinear sample says there is no glow at all, that one tap is enough.
// The field ends at the grid. Beyond it the glow falls off as its blur would have carried
// on (the field's last texel, clamped, once drew a band of light down from a melt on the
// floor to the bottom of the screen).
vec4 glow_at(vec2 x) {
    vec2 o = max(max(-x, x - vec2(pc.grid)), vec2(0.0));
    float od2 = dot(o, o);
    if (od2 > 9.0 * GLOW_FALL * GLOW_FALL) return vec4(0.0);
    float fall = exp(-0.5 * od2 / (GLOW_FALL * GLOW_FALL));
    x = clamp(x, vec2(0.0), vec2(pc.grid));
    vec2 size = vec2(textureSize(glow_map, 0));
    vec2 uv = x / (2.0 * size);
    vec4 near = textureLod(glow_map, uv, 0.0) * fall;
    if (max(max(near.r, near.g), max(near.b, near.a)) < 1e-4) return near;
    vec2 p = uv * size - 0.5, i = floor(p), f = p - i;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
    vec2 g0 = w0 + w1, g1 = w2 + w3;
    vec2 h0 = (i - 0.5 + w1 / g0) / size, h1 = (i + 1.5 + w3 / g1) / size;
    return fall * (g0.y * (g0.x * textureLod(glow_map, vec2(h0.x, h0.y), 0.0) + g1.x * textureLod(glow_map, vec2(h1.x, h0.y), 0.0))
                 + g1.y * (g0.x * textureLod(glow_map, vec2(h0.x, h1.y), 0.0) + g1.x * textureLod(glow_map, vec2(h1.x, h1.y), 0.0)));
}

// The smoke field at a world position (cells), cubic B-spline filtered (four bilinear taps),
// so a wisp's edge stays smooth close up.
// (Until 2026-11-14 0.7 and a cover of up to 0.75: a boiling pot's plume hid the tank
// behind it like smoke. Steam is a thin mist: it veils, it does not hide.)
const float STEAM_OPACITY = 0.32; // per unit of vapour along the view
const float STEAM_COVER = 0.55;   // most light it scatters (it hides 0.3 of that of the backdrop)
const float STEAM_SHADOW = 0.16;  // per unit of vapour per cell towards the light
float smoke_at(vec2 x) {
    vec2 size = vec2(textureSize(smoke_map, 0));
    vec2 t = x * size / vec2(pc.grid) - 0.5;
    vec2 i = floor(t);
    vec2 f = t - i;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    vec2 w3 = f3 / 6.0;
    vec2 g0 = w0 + w1, g1 = w2 + w3;
    vec2 h0 = (i - 0.5 + w1 / g0) / size, h1 = (i + 1.5 + w3 / g1) / size;
    return g0.y * (g0.x * textureLod(smoke_map, vec2(h0.x, h0.y), 0.0).r + g1.x * textureLod(smoke_map, vec2(h1.x, h0.y), 0.0).r)
         + g1.y * (g0.x * textureLod(smoke_map, vec2(h0.x, h1.y), 0.0).r + g1.x * textureLod(smoke_map, vec2(h1.x, h1.y), 0.0).r);
}

// What liquids left on the back wall or a solid's face here (the film map, film.comp; README,
// "Films, stains and drips"), drawn over `under`. The film is what it absorbs and scatters,
// integrated through it, and how much of it is still liquid:
//   - a layer of absorbance A and scattering S over the surface: light goes through it to the
//     surface and back (exp(-2 (A + S))) and it sends back what it scatters (two-stream), so a
//     thin water coat is clear, a blood stain scarlet where thin and near black where thick,
//     a mud splat brown, a brine crust white;
//   - while wet, it darkens the surface under it (light trapped in the liquid layer by total
//     internal reflection, Lekner & Dorf 1988) and its slope catches the light (the bead at a
//     run's head, a drip's ridge glisten);
//   - dried, rigid grains sit in air instead of liquid and scatter several times as much (the
//     refractive index step to air is larger): dried mud and salt lighten and go matte; soft
//     particles (blood cells) coalesce into a glassy layer that scatters less: dried blood
//     is darker (material.rs: dry_scatter).
const float WET_DARK = 0.45;       // most a wet film darkens the surface under it
const float FILM_LIGHT = 0.75;     // light that reaches the back wall and the solids' faces (linear)
// The film map at a world position, filtered with a cubic B-spline (four bilinear taps): its
// texels are a few pixels wide close up, and bilinear filtering drew drops as diamonds.
vec4 film_sample(sampler2DArray m, vec2 x) {
    vec2 size = vec2(textureSize(m, 0).xy);
    vec2 t = x * size / vec2(pc.grid) - 0.5;
    vec2 i = floor(t), f = t - i;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
    vec2 g0 = w0 + w1, g1 = w2 + w3;
    vec2 h0 = (i - 0.5 + w1 / g0) / size, h1 = (i + 1.5 + w3 / g1) / size;
    float L = pc.film_layer;
    return g0.y * (g0.x * textureLod(m, vec3(h0.x, h0.y, L), 0.0) + g1.x * textureLod(m, vec3(h1.x, h0.y, L), 0.0))
         + g1.y * (g0.x * textureLod(m, vec3(h0.x, h1.y, L), 0.0) + g1.x * textureLod(m, vec3(h1.x, h1.y, L), 0.0));
}

vec3 draw_film(vec3 under, vec2 x, float zoom) {
    // (The map ends at the grid: beyond it the sampler would repeat its edge texels, and a
    // run on the tank's floor was drawn as a curtain of streaks down to the bottom of the
    // window.)
    if (!FX_FILM || pc.stain_k <= 0.0 || any(lessThan(x, vec2(0.0))) || any(greaterThan(x, vec2(pc.grid)))) return under;
    // (One bilinear tap first: most of the screen has no film within reach.)
    vec2 uv = x / vec2(pc.grid);
    vec2 ts = 1.0 / vec2(textureSize(film_h, 0).xy);
    float any_h = textureLod(film_h, vec3(uv, pc.film_layer), 0.0).x + textureLod(film_h, vec3(uv + 1.5 * ts, pc.film_layer), 0.0).x
                + textureLod(film_h, vec3(uv - 1.5 * ts, pc.film_layer), 0.0).x;
    if (any_h < 0.003) return under;
    vec4 fh = film_sample(film_h, x);
    if (fh.x < 0.003) return under;
    vec4 fa = max(film_sample(film_c, x), vec4(0.0));
    // How liquid it still is (film.comp: liquid): its volatile share against its liquid's
    // as laid; a film of a liquid that never dries is always wet.
    int frow = int(fh.z + 0.5) - 1;
    float v0 = frow >= 0 ? optics[min(frow, 95)].fx.x : 1.0;
    float h = fh.x, v = v0 > 0.0 ? clamp(fh.y / v0, 0.0, 1.0) : 1.0;
    vec2 tc = vec2(pc.grid) * ts;   // cells per texel
    // (Its slope from bilinear taps a texel apart, and only where texels span pixels.)
    float sharp = clamp(zoom * tc.x / 3.0, 0.0, 1.0);
    vec2 gh = vec2(0.0);
    if (sharp > 0.05) {
        float L = pc.film_layer;
        gh = 0.5 * vec2(textureLod(film_h, vec3(uv + vec2(ts.x, 0.0), L), 0.0).x - textureLod(film_h, vec3(uv - vec2(ts.x, 0.0), L), 0.0).x,
                        textureLod(film_h, vec3(uv + vec2(0.0, ts.y), L), 0.0).x - textureLod(film_h, vec3(uv - vec2(0.0, ts.y), L), 0.0).x);
    }
    float k = pc.stain_k;
    // The surface under it, darker where the film is wet.
    float wet = v * (1.0 - exp(-h * v / 0.15));
    vec3 ul = to_linear(under) * (1.0 - WET_DARK * wet * k);
    // The layer.
    vec3 A = fa.rgb * k;
    float dry_k = frow >= 0 ? float(optics[min(frow, 95)].idx.z) / 1000.0 : 1.0;
    vec3 S = vec3(fa.a * mix(dry_k, 1.0, v)) * k;
    vec3 tv = exp(-(A + S));
    vec3 refl = deep_reflectance3(A + 1e-4, S) * (1.0 - tv);
    vec3 o = to_srgb(refl * FILM_LIGHT + ul * tv * tv);
    // Wet: a faint sheen and the glint of its slope (texels are a fraction of a pixel far
    // out, where the slope says nothing: faded there).
    vec3 N = normalize(vec3(-gh * 0.5 * sharp, 1.0));
    vec3 H = normalize(L3 + vec3(0.0, 0.0, 1.0));
    float cover = 1.0 - exp(-h / 0.4);
    float spec = pow(max(dot(N, H), 0.0), 60.0) * v * cover;
    o += vec3(1.0, 0.94, 0.9) * (0.6 * spec + 0.03 * wet) * k;
    return o;
}

// The back wall: a quiet gradient in large tiles (each a touch lighter or darker, with a
// faint grain and lighter joints), what has stained it, the shadow of the liquid in front
// of it, and the light hot liquid throws on it. `p` is the pixel position (for the shadow,
// from the wide field).
vec3 backdrop(vec2 x, vec2 p, float zoom, vec3 glow, bool grain) {
    float h = clamp(x.y / float(pc.grid.y), 0.0, 1.0);
    vec3 col = mix(vec3(0.058, 0.066, 0.094), vec3(0.098, 0.116, 0.162), h * h * (3.0 - 2.0 * h));
    vec2 tile = floor(x / 16.0);
    col *= 0.955 + 0.09 * hash12(tile + 0.5);
    if (grain) col *= mix(1.0, 0.95 + 0.1 * gnoise(x * 0.5), detail(0.5, zoom));
    vec2 gl = abs(fract(x / 16.0 + 0.5) - 0.5) * 16.0;
    col += 0.016 * (1.0 - smoothstep(0.0, 1.5 / zoom, min(gl.x, gl.y))) * clamp(zoom * 0.5, 0.0, 1.0);
    col = draw_film(col, x, zoom);
    // Shadow: the liquid up and to the left (towards the light), from the wide field.
    vec2 ps = p - vec2(0.9, 1.3) * zoom;
    float Ts = total(textureLod(wide, min(ps * pc.wide_scale, pc.wide_max), 0.0)) / pc.rest;
    col *= 1.0 - 0.28 * smoothstep(0.04, 0.5, Ts);
    col += 0.8 * glow * (col * 3.0 + 0.04);
    return col;
}

// ---- ferrofluid spikes (README, "Magnetism") ---------------------------------------------
// Where a magnet's field passes the Rosensweig threshold at the surface of a magnetisable
// liquid, the surface stands up in a row of spikes along the field (cell_optics.comp works
// out their height per cell from the field, layer 13). They are drawn into the sharp field:
// from a pixel, the surface below it along the field is found (marching towards the liquid),
// and the spike there is a cone with slightly hollow flanks, centred on a lattice across the
// field (fixed to the field, as real spikes stand over a magnet while the liquid moves under
// them). The spike adds liquid of the base's own phase groups where it reaches past the
// field, so the rest of the shading draws it as that liquid.
const int SPIKE_STEPS = 6;
const float SPIKE_SHARP = 1.6;   // flank exponent (1: straight cones)
const float SPIKE_G = 2.0;       // sharp-field weight per cell inside a spike's outline
const float SPIKE_PHASE_DIV = 4.0;   // cell_optics.comp: the lattice's phase is stored at this fraction

vec2 px_of(vec2 x) { return pc.half_res + (x - pc.cam) / pc.inv_zoom * vec2(1.0, -1.0); }
float wide_total(vec2 x) { return total(textureLod(wide, min(px_of(x) * pc.wide_scale, pc.wide_max), 0.0)); }

struct Spike { bool on; vec2 x0, ax, tg; float s0, gap, hs, ph; vec4 comp; vec2 inside; };

Spike find_spike(vec2 world, float W, float zoom) {
    Spike sp = Spike(false, world, vec2(0.0), vec2(0.0), 0.0, 1.0, 0.0, 0.0, vec4(0.0), world);
    if (W >= BODY_CUT) return sp;
    vec4 m = textureLod(cellopt, vec3(world / vec2(pc.grid), 13.0), 0.0);
    float gap = m.w;
    // (Spikes narrower than a few pixels would shimmer: faded out.)
    float fade = smoothstep(4.0, 8.0, gap * zoom);
    if (gap <= 0.0 || fade <= 0.0 || length(m.xy) < 1e-6) return sp;
    vec2 bh = m.xy / length(m.xy);
    // How tall a spike can be (cell_optics.comp: spike_k x the spacing); only where some
    // magnetisable liquid lies within that along the field.
    float reach = 0.9 * gap;
    vec2 g = pc.grid;
    if (max(textureLod(cellopt, vec3(world / g, 14.0), 0.0).z, max(textureLod(cellopt, vec3((world - bh * 0.6 * reach) / g, 14.0), 0.0).z,
                     textureLod(cellopt, vec3((world + bh * 0.6 * reach) / g, 14.0), 0.0).z)) < 0.01) return sp;
    float step = reach / float(SPIKE_STEPS);
    bool found = false;
    vec2 dir = vec2(0.0);
    float s0 = 0.0;
    for (int side = 0; side < 2 && !found; side++) {
        dir = side == 0 ? -bh : bh;   // towards the liquid
        float prev = W;
        for (int k = 1; k <= SPIKE_STEPS; k++) {
            float wk = total(field(px_of(world + dir * (step * float(k)))));
            if (wk >= BODY_CUT) {
                // Narrowed by bisection, then between the last two (a step's linear guess
                // drew terraces down the flanks).
                float lo = step * float(k - 1), hi = step * float(k);
                for (int it = 0; it < 3; it++) {
                    float mid = 0.5 * (lo + hi);
                    float wm = total(field(px_of(world + dir * mid)));
                    if (wm >= BODY_CUT) { hi = mid; wk = wm; } else { lo = mid; prev = wm; }
                }
                s0 = mix(lo, hi, clamp((BODY_CUT - prev) / max(wk - prev, 1e-4), 0.0, 1.0));
                found = true;
                break;
            }
            prev = wk;
        }
    }
    if (!found) return sp;
    vec2 base = world + dir * s0;
    vec2 xb = base + dir * 0.8;
    // The height is the field's at the base (it falls off towards the tips) ...
    vec2 ub = xb / vec2(pc.grid);
    vec4 mb = textureLod(cellopt, vec3(ub, 13.0), 0.0);
    float hs = min(length(mb.xy), reach) * fade;
    // ... only from a magnetisable liquid ...
    vec4 mp = textureLod(cellopt, vec3(ub, 14.0), 0.0);
    float share = mp.z / max(textureLod(cellopt, vec3(ub, 4.0), 0.0).w, 1e-3);
    // ... whose surface faces the field (from the wide field: along it, the field holds a
    // surface flat).
    vec2 tg = vec2(-dir.y, dir.x);
    float along = wide_total(base + dir * 1.5) - wide_total(base - dir * 1.5);
    float across = wide_total(base + tg * 1.5) - wide_total(base - tg * 1.5);
    float facing = along / max(length(vec2(along, across)), 1e-4);
    hs *= smoothstep(0.3, 0.7, share) * smoothstep(0.45, 0.85, facing);
    if (hs < 0.2) return sp;
    vec4 ab = field(px_of(xb));
    // The lattice's phase at the base, and which way along tg it grows.
    float ph = SPIKE_PHASE_DIV * atan(mp.y, mp.x) / 6.2831853;
    vec2 q = textureLod(cellopt, vec3((xb + tg) / g, 14.0), 0.0).xy;
    float dph = atan(mp.x * q.y - mp.y * q.x, dot(mp.xy, q));
    sp.on = true;
    sp.ax = -dir;
    sp.tg = dph < 0.0 ? -tg : tg;
    sp.ph = ph;
    sp.s0 = s0;
    sp.gap = gap;
    sp.hs = hs;
    sp.comp = ab / max(total(ab), 1e-4);
    sp.inside = xb;
    return sp;
}

// The sharp field's weight of the spike at `dp` (cells) from the pixel: `cut` at its outline,
// rising SPIKE_G a cell inwards; 0 below its base (the liquid itself).
float spike_weight(Spike sp, vec2 dp, float cut) {
    float s = sp.s0 + dot(dp, sp.ax);
    if (s < -0.5) return 0.0;
    float v = sp.ph + dot(dp, sp.tg) / sp.gap;
    float u = abs(fract(v + 0.5) * 2.0 - 1.0);
    float h = sp.hs * pow(1.0 - u, SPIKE_SHARP);
    float slope = sp.hs * SPIKE_SHARP * pow(max(1.0 - u, 1e-4), SPIKE_SHARP - 1.0) * 2.0 / sp.gap;
    return max(cut + SPIKE_G * (h - s) / sqrt(1.0 + slope * slope), 0.0);
}
vec4 with_spike(vec4 f, Spike sp, vec2 dp, float cut) {
    return f + sp.comp * max(spike_weight(sp, dp, cut) - total(f), 0.0);
}

void main() {
    vec2 px = gl_FragCoord.xy;
    float zoom = 1.0 / pc.inv_zoom;
    vec2 world = pc.cam + (px - pc.half_res) * vec2(pc.inv_zoom, -pc.inv_zoom);

    // ---- the back wall, and a vignette --------------------------------------------------
    vec2 vg = (px - pc.half_res) / pc.half_res;
    float vignette = 1.0 - 0.22 * smoothstep(0.5, 1.5, length(vg));
    // The glow field here (light on the back wall, liquids and solids; bloom).
    vec4 glow_here = pc.glow_k > 0.0 ? glow_at(world) * pc.glow_k : vec4(0.0);
    vec3 col = backdrop(world, px, zoom, glow_here.rgb, true) * vignette;

    // ---- the fields ------------------------------------------------------------------
    // Differences are taken over a fixed share of the kernel, so that they stay well above
    // the precision of the target however far the view is zoomed in.
    float e = max(pc.accum_div, 0.3 * zoom);
    vec4 a = field(px);
    vec4 vap = (HAS_VAPOUR != 0u && pc.vapour > 0.0) ? vapour(px) : vec4(0.0);
    vec4 ax0 = field(px - vec2(e, 0.0)), ax1 = field(px + vec2(e, 0.0));
    vec4 ay0 = field(px + vec2(0.0, e)), ay1 = field(px - vec2(0.0, e));

    vec2 wuv = min(px * pc.wide_scale, pc.wide_max);
    vec4 b = textureLod(wide, wuv, 0.0);
    float Wb = total(b);
    // How much liquid is around, 0..1. Next to a solid, part of the blur lies inside it and
    // can hold none: count only the open part (for a flat wall at distance sd, the share of
    // a Gaussian beyond it), or every wall and floor would read as a surface.
    float sd = sample_sdf(world);
    float open = 0.5 + 0.5 * tanh(1.2 * max(sd, -0.5) / (WIDE_SIGMA * 1.4142));
    float T = Wb / (pc.rest * open);
    // The surface's cut (below) is taken from how much liquid is around before the spikes,
    // which the wide field does not hold.
    float Tc = T;
    // Where the cell optics are read: here, or for a spike the liquid at its base.
    vec2 xo = world;
    bool spiked = false;
    if (FX_SPIKES) {
        Spike sp = find_spike(world, total(a), zoom);
        if (sp.on) {
            // (The cut a liquid without body takes, as below with `thick` 0.)
            float cut = mix(pc.threshold, BODY_CUT, smoothstep(0.12, 0.38, Tc)) * (1.0 - 0.75 * smoothstep(0.62, 0.86, Tc));
            float W0 = total(a);
            float c = e * pc.inv_zoom;
            a = with_spike(a, sp, vec2(0.0), cut);
            ax0 = with_spike(ax0, sp, vec2(-c, 0.0), cut);
            ax1 = with_spike(ax1, sp, vec2(c, 0.0), cut);
            ay0 = with_spike(ay0, sp, vec2(0.0, -c), cut);
            ay1 = with_spike(ay1, sp, vec2(0.0, c), cut);
            // A spike shades as the surface of its liquid, not as spray.
            float sw = smoothstep(0.0, 0.5, total(a) - W0);
            if (sw > 0.0) {
                xo = sp.inside;
                spiked = true;
                T = mix(T, max(T, 0.45), sw);
            }
        }
    }
    float W = total(a);
    // Gradients per cell, +y up.
    float ge = 0.5 * zoom / e;
    vec2 gW = vec2(total(ax1) - total(ax0), total(ay1) - total(ay0)) * ge;

    // Aqueous liquids against the others (oil-like, metal, melt): the share of the others.
    float oil = (a.y + a.z + a.w) / max(W, 1e-4);
    float oil_w = (b.y + b.z + b.w) / max(Wb, 1e-4);
    // Among the others, the leading group and the next: where they swap (oil on mercury) is
    // an interface too.
    int A = 1;
    for (int g = 2; g < 4; g++) if (a[g] > a[A]) A = g;
    int B = A == 1 ? 2 : 1;
    for (int g = 1; g < 4; g++) if (g != A && a[g] > a[B]) B = g;
    float mixed_ot = smoothstep(0.02, 0.2, a[B] / max(a.y + a.z + a.w, 1e-4));
    vec2 gx = vec2(lead(ax1, A, B) - lead(ax0, A, B), lead(ay1, A, B) - lead(ay0, A, B)) * ge;
    float gxn = length(gx) + 1e-5;
    float d_if = lead(a, A, B) / gxn;                  // cells from that interface, into A
    vec2 n_if = gx / gxn;

    // The optics, per cell (soft, as blends are); immiscible edges are kept by the split
    // into the aqueous liquids and the others. (Empty air with no liquid nearby needs none of
    // it: most of the screen.)
    bool near = W > 1e-4 || T > 0.003;
    CellOptics co = near ? cell_optics(xo) : CellOptics(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec2(0.0));
    // The liquids' colour from their particles (splat.vert), each side's (aqueous, others)
    // where it has the pixel nearly to itself, and the rest of the look from the cells that
    // match (above). (Not in a spike: no particles there.)
    vec4 kaq_px = vec4(0.0), kot_px = vec4(0.0);
    float own_aq = 0.0, own_ot = 0.0;
    bool from_px = pc.tint > 0.0 && near && W > 0.05 && !spiked;
    if (from_px) {
        vec2 auv = min(px * pc.accum_scale, pc.accum_max);
        vec3 sw = tint_read(tint_s, auv).xyz;
        vec4 tc = tint_read(tint_c, auv);
        float tw = sw.y + sw.z;
        own_aq = smoothstep(0.95, 0.995, sw.y / max(tw, 1e-4)) * smoothstep(0.02, 0.1, sw.y);
        own_ot = smoothstep(0.95, 0.995, sw.z / max(tw, 1e-4)) * smoothstep(0.02, 0.1, sw.z);
        // (What the few percent of the other side lay down is taken out with its coefficients
        // from the cells: a liquid that absorbs as strongly as chocolate still drew a brown
        // line round every speck where its share passed through the cut.)
        // Those coefficients are the cells' blend of that side, not those of the particles
        // here: ferrofluid stirred into a pool that also holds oil was taken out as mostly
        // oil (a black ring round each speck) or the reverse (a clear one). Where what is
        // taken out is more than a fraction of what stays, the cells' colour is kept
        // (TINT_TRUST).
        vec4 kc_ot = (co.ot + co.ds) / max(co.sty.w - co.sty.z, 1e-4);
        vec4 kc_aq = co.aq / max(co.sty.z, 1e-4);
        float k_ot = sw.z * dot(kc_ot.rgb, vec3(1.0)), k_aq = sw.y * dot(kc_aq.rgb, vec3(1.0));
        own_aq *= 1.0 - smoothstep(TINT_TRUST.x, TINT_TRUST.y, k_ot / max(k_aq, 1e-4));
        own_ot *= 1.0 - smoothstep(TINT_TRUST.x, TINT_TRUST.y, k_aq / max(k_ot, 1e-4));
        if (own_aq > 0.0) {
            vec4 pl = textureLod(cellopt, vec3(xo / vec2(pc.grid), 15.0), 0.0);
            kaq_px = particle_absorb(max(tc - sw.z * kc_ot, 0.0), sw.xy, pl.rgb / max(pl.a, 1e-4));
        }
        if (own_ot > 0.0) kot_px = max(tc - sw.y * kc_aq, 0.0) / max(sw.z, 1e-4);
        from_px = own_aq + own_ot > 0.0;
    }
    if (near && W > 0.05 && !spiked) {
        vec4 kc = textureLod(cellopt, vec3(xo / vec2(pc.grid), 12.0), 0.0);
        if (kc.a > 1e-3) {
            // (The particles' key where they are aqueous; elsewhere the cells' smoothed key, a
            // little way along their gradient, stands in: the read then moves only where the
            // cells change.)
            vec3 k0 = kc.rgb / kc.a;
            vec3 ks = own_aq > 0.5 ? look_key(kaq_px) : own_ot > 0.5 ? look_key(kot_px) : k0;
            float guide = smoothstep(KEY_LO, KEY_HI, length(k0 - ks));
            if (guide > 0.0) co = mix_optics(co, cell_optics(seek_key(xo, ks, k0)), guide);
        }
    }
    // A pocket of steam is part of the body (splat.vert), but the cell optics hold only the
    // liquid: where the cells here are mostly vapour, the look is taken from the liquid
    // around them (rings of eight points three and seven cells out), in whatever liquid it
    // boiled from.
    float hollow = vap.r > 0.04 ? 1.0 - smoothstep(0.08, 0.6, co.sty.w) : 0.0;
    if (near && hollow > 0.0) {
        CellOptics ring = CellOptics(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec2(0.0));
        for (int k = 0; k < 16; k++) {
            float ang = 0.785398 * float(k) + (k < 8 ? 0.39 : 0.0);
            CellOptics q = cell_optics(world + (k < 8 ? 3.0 : 7.0) * vec2(cos(ang), sin(ang)));
            ring.aq += q.aq; ring.ot += q.ot; ring.opt += q.opt; ring.em += q.em;
            ring.sty += q.sty; ring.fz += q.fz; ring.fx += q.fx; ring.l9 += q.l9; ring.ds += q.ds; ring.dn += q.dn;
        }
        float f = 0.0625 * hollow;
        co.aq += f * ring.aq; co.ot += f * ring.ot; co.opt += f * ring.opt; co.em += f * ring.em;
        co.sty += f * ring.sty; co.fz += f * ring.fz; co.fx += f * ring.fx; co.l9 += f * ring.l9; co.ds += f * ring.ds; co.dn += f * ring.dn;
    }
    vec4 sty = co.sty;
    // The specks' share of the colour (layer 18: those too many to draw, where they lie) and
    // the colour of the scattering: red and blue (layer 16) against the mean.
    vec4 kdust = vec4(0.0);
    if (near) {
        // (Layer 16 holds the red and blue against the mean as ratios, which interpolate
        // between cells without the fringes that sums read at another point than the
        // colour gave; green is what the mean leaves. The specks scatter grey.)
        vec2 cuv = xo / vec2(pc.grid);
        kdust = textureLod(cellopt, vec3(cuv, 18.0), 0.0) / max(sty.w, 1e-4);
        vec2 rb = textureLod(cellopt, vec3(cuv, 16.0), 0.0).zw;
        float sm = (co.aq.a + co.ot.a + co.ds.a) / max(sty.w, 1e-4);
        vec3 r3 = vec3(rb.x, max(3.0 - rb.x - rb.y, 0.0), rb.y);
        g_sr = clamp((r3 * sm + kdust.a) / max(sm + kdust.a, 1e-5), 0.15, 3.0);
        if (sm + kdust.a <= 1e-5) g_sr = vec3(1.0);
    }
    // Coefficients of the aqueous liquids here, and their colour as a few cells of it in full
    // light (for rims, droplets and the wet tint).
    vec4 kaq = co.aq / max(sty.z, 1e-4);
    // The colour is the particles' own where one side (aqueous or not) fills the pixel: a
    // trail of a few particles of blood in honey is a trail of red drops. From the cells, which
    // count those few as spread through each cell they pass and then smooth over 3 x 3, a
    // liquid that absorbs as strongly as blood coloured a band two or three cells wide in
    // their shapes (README, "Look").
    kaq = mix(kaq, kaq_px, own_aq) + kdust;
    vec3 base = near ? to_srgb(liquid_lin(kaq, light_in(kaq, 1.5), 3.0, LV_BODY, 1.0, vec3(0.0))) : vec3(0.0);
    if (near) base = iced(base, world, 5.0, co.fz.x, co.fz.x / max(sty.z, 1e-4), zoom);
    // Whether any of the other liquids is within reach of the blur. Where none is, every use
    // of their colour below is weighted by a share that is zero, and skipping it (with the
    // shading of the others) keeps aqueous-only pixels from paying for it.
    bool others = oil > 0.0 || oil_w > 0.0 || mixed_ot > 0.0;
    // Next to an interface between two of the others the colour is read a little way into
    // this side, so that each keeps its own colour right up to the line.
    vec2 off = n_if * clamp(1.2 - d_if, 0.0, 1.2) * mixed_ot * zoom;
    vec2 opx = px + vec2(off.x, -off.y);
    vec2 owld = pc.cam + (opx - pc.half_res) * vec2(pc.inv_zoom, -pc.inv_zoom);
    CellOptics oo = (mixed_ot > 0.0) ? cell_optics(owld) : co;
    // The others: oil-like and dense (metal, melt) liquids each by their own coefficients,
    // chosen between by the particles' own field, so a drop of lava in oil is drawn at the
    // shape of its particles (from the cells, which hold both, it was a dark square a few
    // cells across).
    float den_o = oo.dn.x, oil_o = max(oo.sty.w - oo.sty.z - den_o, 0.0);
    float wd_sharp = (a.z + a.w) / max(a.y + a.z + a.w, 1e-4);
    float wd = oil_o < 1e-3 ? 1.0 : den_o < 1e-3 ? 0.0 : wd_sharp;
    vec4 kot = mix(oo.ot / max(oil_o, 1e-4), oo.ds / max(den_o, 1e-4), wd);
    kot = mix(kot, kot_px, own_ot * (1.0 - mixed_ot)) + kdust;
    float fz_ot = mix((oo.fz.y - oo.dn.y) / max(oil_o, 1e-4), oo.dn.y / max(den_o, 1e-4), wd);
    vec3 ob = others ? to_srgb(liquid_lin(kot, light_in(kot, 1.5), 3.0, LV_BODY, 1.0, vec3(0.0))) : vec3(0.0);
    if (others) ob = iced(ob, owld, 6.0, oo.fz.y, fz_ot, zoom);
    // Everything else as blended here.
    float wo = max(sty.w, 1e-4);
    vec4 opt = co.opt / wo;
    vec4 em = co.em / wo;
    float opac = opt.x, plate = opt.y, metal = opt.z, gloss = opt.w;
    float body = sty.x / wo;
    // The effects (optics table `fx`), and how far the liquid has carried their patterns.
    vec4 fxv = co.fx / wo;
    float irid = FX_PATTERNS ? fxv.z : 0.0, crackle = FX_PATTERNS ? fxv.w : 0.0;
    // Cell optics layer 17: the liquid's own light.
    vec3 lumc = near ? textureLod(cellopt, vec3(xo / vec2(pc.grid), 17.0), 0.0).rgb / wo : vec3(0.0);
    Carry ks = carry(world, false, false), kl = carry(world, true, false);
    // The key light that reaches the liquid here through the liquid above it, and the
    // quenched skin of a melt.
    vec4 l9 = co.l9 / wo;
    vec3 il = clamp(l9.rgb, 0.0, 1.0);
    // With the light under the surface solved (sss.comp), the light at the liquid is that:
    // it came in through every face that sees the sky and spread, each colour by how the
    // liquid absorbs and scatters it.
    if (pc.sss > 0.0 && near) il = clamp(textureLod(cellopt, vec3(xo / vec2(pc.grid), 20.0), 0.0).rgb, 0.0, 1.0);
    // The refractive index (layer 17 a).
    float ior = near && wo > 1e-3 ? textureLod(cellopt, vec3(xo / vec2(pc.grid), 17.0), 0.0).a / wo : 1.33;
    ior = clamp(ior, 1.0, 2.5);
    // The look field: the platelets' sheen, unused, their lie (per unit of liquid with platelets there).
    vec4 lkf = vec4(0.0);
    if (pc.look > 0.0 && near) {
        vec4 lr = tint_read(look_map, min(px * pc.accum_scale, pc.accum_max));
        if (lr.a > 1e-3) lkf = vec4(lr.rgb / lr.a, smoothstep(0.02, 0.25, lr.a));
    }
    float skin = clamp(l9.a, 0.0, 1.0);
    vec4 glf = glow_here;

    if (W > 1e-3) {
        // Thick liquid is drawn fuller: honey holds together in fat, rounded shapes.
        float thick = smoothstep(0.3, 0.9, body) * (1.0 - oil);
        // A lone particle is a droplet (its weight peaks at 1). Against a body of liquid the
        // surface is taken deeper in the field, where several particles overlap: it is then
        // smooth instead of following each particle of the outermost layer.
        // Well inside, the cut drops again: only a real cavity opens a hole, not a spot where
        // the particles happen to sit a little apart (as they do where two liquids meet).
        float t = mix(pc.threshold, mix(BODY_CUT, BODY_CUT_THICK, thick), smoothstep(0.12, 0.38, Tc));
        t *= 1.0 - 0.75 * smoothstep(0.62, 0.86, Tc);
        float gn = length(gW) + 1e-5;
        float d = (W - t) / gn;                          // cells inside the surface
        vec2 n = -gW / gn;                               // out of the liquid
        float alpha = clamp(d * zoom + 0.5, 0.0, 1.0);
        // Kernels smaller than a pixel cannot make a surface: far out, spray is a haze.
        alpha = max(alpha, pc.mist * clamp(W / t, 0.0, 1.0));
        // For the lighting below, the depth the wide field gives (about 0.19 of its range per
        // cell under a flat surface) bounds the distance from below: deep in a body the sharp
        // field ripples with the layers of particles, and its distance dipped towards zero
        // there, which lit rims and streaks inside a liquid metal in vertical stripes. (The
        // coverage above still comes from the sharp field alone: a real cavity opens.)
        d = max(d, 4.5 * (T - 0.58));

        if (alpha > 0.0) {
            float lit = dot(n, LIGHT);
            float up = smoothstep(-0.25, 0.65, lit);
            float thin = 1.0 - smoothstep(0.30, 0.62, T);    // droplets, films, sheets, spray
            float deep = smoothstep(0.55, 0.97, T);
            float px_cells = pc.inv_zoom;
            vec4 bx = textureLod(wide, min((px + vec2(2.0 * e, 0.0)) * pc.wide_scale, pc.wide_max), 0.0);
            vec4 by = textureLod(wide, min((px - vec2(0.0, 2.0 * e)) * pc.wide_scale, pc.wide_max), 0.0);
            float gwo = 0.5 * zoom / e;                      // forward differences, per cell
            // The body's gradient, central over 1.5 of the wide field's own texels each way:
            // its samples are bilinear, and a difference over less than a texel is constant
            // across each texel, which drew the normal in stripes and squares (most plainly
            // in a liquid metal's reflection).
            float hw = max(2.0 * e, 1.5 / (pc.wide_scale.x * float(textureSize(wide, 0).x)));
            vec2 gT = vec2(total(textureLod(wide, min((px + vec2(hw, 0.0)) * pc.wide_scale, pc.wide_max), 0.0))
                         - total(textureLod(wide, min((px - vec2(hw, 0.0)) * pc.wide_scale, pc.wide_max), 0.0)),
                           total(textureLod(wide, min((px - vec2(0.0, hw)) * pc.wide_scale, pc.wide_max), 0.0))
                         - total(textureLod(wide, min((px + vec2(0.0, hw)) * pc.wide_scale, pc.wide_max), 0.0)));
            float gw = 0.5 * zoom / hw;                      // per cell
            vec2 n_w = -gT / (length(gT) + 1e-5);            // the body's outward normal, smooth
            float lit_w = dot(n_w, LIGHT);
            // Only where the wide field says a surface is near: deep in a body the sharp field
            // still ripples with the layers of particles (in a dense metal most of all).
            float near_surface = 1.0 - smoothstep(0.72, 0.92, T);

            // -- the surface as a height field over the liquid, lit from the upper left --
            // It rounds over within `rad` cells of its edge (fuller for thick liquids); further
            // in, the body's broad shape comes from the wide field.
            float rad = mix(0.8, 1.9, max(body, thick));
            float edge_k = 1.0 - smoothstep(0.0, rad, d);
            vec2 slope = n * (3.0 * edge_k * edge_k) * mix(1.0, near_surface, metal)
                       + n_w * min(length(gT) * gw * 1.5, 0.8) * near_surface;
            vec3 N = normalize(vec3(slope, 1.0));
            // The face towards the viewer, shaded by how it turns to the light near the edge.
            float shade = mix(1.0, mix(0.70, 1.15, smoothstep(-0.6, 0.8, dot(normalize(slope + vec2(0.0, 1e-4)), LIGHT))), clamp(length(slope), 0.0, 1.0));

            // What lies behind a clear liquid: the back wall, bent by the surface. (At most
            // 0.7 of a cell: near the edge, where the slope is steep, a larger shift made the
            // wall's joints swim and flicker as the surface moved.)
            vec2 roff = slope * 0.9 * (ior - 1.0) / 0.33;
            roff *= min(1.0, 0.7 / max(length(roff), 1e-4));
            vec3 bd = backdrop(world + roff, px + vec2(roff.x, -roff.y) * zoom, zoom, glf.rgb, false) * vignette;
            // Thickness along the view: a body is a few cells thick, a droplet, film or sheet
            // far less.
            float Lb = mix(0.7, LV_BODY, smoothstep(0.15, 0.95, T));
            // A body is round in the depth of the tank too: towards its outline its thickness
            // along the view falls to nothing (a cylinder's chord, sqrt(d (2R - d))), so its
            // edge shows the liquid thin: honey gold, blood scarlet, water clear.
            float chord = sqrt(clamp(d / EDGE_ROUND, 0.0, 1.0) * (2.0 - clamp(d / EDGE_ROUND, 0.0, 1.0)));
            float Lv = Lb * mix(1.0, max(chord, 0.08), near_surface);
            // Light reaches a point from above (light.comp) and through the nearest surface,
            // whichever path takes less of it: light comes in all around the outline of a drop
            // of honey, not only through its top.
            // (Only near the surface: deeper in, the sharp field's distance ripples with the
            // layers of particles, and light through a "surface" there drew a maze over the
            // whole body.)
            float d_s = mix(d, 1e3, 1.0 - near_surface);
            vec3 il_aq = max(il, light_in(kaq, SURFACE_PATH * d_s));
            vec3 il_ot = max(il, light_in(kot, SURFACE_PATH * d_s));

            // -- the aqueous liquids --
            vec3 aq = liquid_srgb(kaq, il_aq, Lv, Lb, shade, bd);
            // Ice: the frozen colour, with a crisp edge (`iced`).
            aq = iced(aq, world, 5.0, co.fz.x, co.fz.x / max(sty.z, 1e-4), zoom);
            // Fast, thin liquid that foams is white water: splashes, spray, wave lips.

            // -- the others: oil, metals, melts, each by its own coefficients --
            float iw = max(1.1 * px_cells, 0.17);
            vec3 oc = vec3(0.0);
            if (others) {
                oc = liquid_srgb(kot, il_ot, Lv, Lb, shade, bd);
                oc = iced(oc, owld, 6.0, oo.fz.y, fz_ot, zoom);
                // Metal: a mirror of a soft studio sky, by the surface normal near the surface
                // and by the shape of the body further in.
                if (metal > 0.0) {
                    // (Only right at the surface: a little way in, the sharp field's gradient
                    // follows the layers of particles and drew the body in vertical stripes.)
                    float at_surf = clamp(1.0 - d, 0.0, 1.0);
                    at_surf *= at_surf * (1.0 - smoothstep(0.45, 0.7, T));
                    vec2 nm = normalize(mix(n_w, n, at_surf) + vec2(0.0, 1e-4));
                    float sky = smoothstep(-0.35, 0.95, dot(nm, LIGHT));
                    // Where the body is flat (deep inside) its normal faces the viewer: an even
                    // mid reflection, not the direction of a vanishing gradient.
                    float shaped = max(clamp(length(gT) * gw / 0.12, 0.0, 1.0), at_surf);
                    sky = mix(0.5, sky, shaped);
                    // A studio around the tank: a dark floor up to a horizon, a graded sky
                    // above it, a soft box towards the light and a cool strip light behind.
                    float e_l = dot(nm, LIGHT);
                    vec3 ground = mix(vec3(0.045, 0.05, 0.065), vec3(0.24, 0.23, 0.24), smoothstep(-1.0, -0.04, nm.y));
                    vec3 skyc = mix(vec3(0.36, 0.42, 0.56), vec3(0.95, 0.97, 1.0), sky * sky);
                    vec3 env = mix(ground, skyc, smoothstep(-0.07, 0.05, nm.y));
                    env = mix(vec3(0.40, 0.42, 0.47), env, shaped);
                    env += vec3(0.55) * smoothstep(0.90, 0.985, e_l) * shaped;
                    env += vec3(0.20, 0.24, 0.32) * smoothstep(0.72, 0.92, -e_l) * shaped;
                    vec3 mc = to_srgb(deep_reflectance(kot.rgb, kot.a));
                    oc = mix(oc, mc * env * 1.15, metal);
                }
                // Between two of the others (oil on mercury): a fine dark line on the later
                // group's side.
                float edge_in = 1.0 - clamp((d_if - iw) * zoom + 0.5, 0.0, 1.0);
                oc = mix(oc, oc * 0.55, 0.6 * edge_in * mixed_ot);
            }

            // -- the others against the aqueous ones --
            // Where the two are finely dispersed, the other liquid is drawn as droplets of its
            // own colour suspended in the clear one (from the sharp field, so each one keeps
            // its own outline), and the aqueous liquid around them takes a faint tint from
            // those too small to draw. A pale "milk" blend here made every stirred tank look
            // cloudy white. Where the interface is resolved it is a sharp line.
            float oil_bx = (bx.y + bx.z + bx.w) / max(total(bx), 1e-4), oil_by = (by.y + by.z + by.w) / max(total(by), 1e-4);
            vec2 gOw = vec2(oil_bx - oil_w, oil_by - oil_w) * gwo;
            float resolved = smoothstep(0.03, 0.10, length(gOw));
            float os = mix(oil, oil_w, 0.85);
            vec3 warm = mix(aq, deepen(mix(ob, base, 0.25), 0.10), OIL_TINT * smoothstep(0.0, 0.5, oil_w));
            // Partly from the blurred field, so a droplet's edge does not shimmer as its
            // particles shuffle.
            float drop = smoothstep(DROP_CUT - 0.14, DROP_CUT + 0.14, mix(oil, oil_w, 0.35));
            vec3 droplets = mix(warm, oc, drop);
            // A slightly darker edge so a droplet reads as a bead, not a smudge.
            droplets = mix(droplets, deepen(ob, 0.30), 0.22 * drop * (1.0 - drop) * 4.0);
            vec3 emulsion = mix(droplets, oc, smoothstep(0.55, 0.9, os));

            vec2 gO = vec2((ax1.y + ax1.z + ax1.w) / max(total(ax1), 1e-4) - (ax0.y + ax0.z + ax0.w) / max(total(ax0), 1e-4),
                           (ay1.y + ay1.z + ay1.w) / max(total(ay1), 1e-4) - (ay0.y + ay0.z + ay0.w) / max(total(ay0), 1e-4)) * ge;
            float gon = length(gO) + 1e-5;
            float od = (oil - OIL_CUT) / gon;                // cells inside the other liquid
            float om = clamp(od * zoom + 0.5, 0.0, 1.0);
            // Stray droplets below the cut tint the aqueous liquid near them a little.
            vec3 fc = mix(mix(aq, deepen(mix(ob, base, 0.25), 0.10), OIL_TINT * smoothstep(0.0, OIL_CUT, oil)), oc, om);
            // The interface: a fine darker line on the other liquid's side, and the light the
            // boundary between the two reflects on the aqueous side.
            float mixed = clamp(smoothstep(0.02, 0.2, oil) - smoothstep(0.8, 0.98, oil), 0.0, 1.0);
            fc = mix(fc, fc * 0.62, 0.55 * om * (1.0 - clamp((od - iw) * zoom + 0.5, 0.0, 1.0)) * mixed);
            fc = mix(fc, lighten(fc, 0.35), 0.4 * band(-od, 0.0, max(1.0 * px_cells, 0.12), zoom) * mixed);
            fc = mix(emulsion, fc, resolved);
            float oily = mix(smoothstep(0.42, 0.9, os), om, resolved);

            // -- small droplets inside a body of the other side --
            // A drop of one or a few particles never reaches the cuts above: the blurred field
            // hardly sees it and the sharp one peaks below OIL_CUT. Inside a body of the other
            // side (from the wide field) it is drawn from the sharp field alone, as a bead of
            // its own colour with a darker rim, antialiased along the field's gradient, so
            // that it moves smoothly with its particles instead of blinking.
            float sp_o = 0.0, sp_a = 0.0;
            if (others) {
                // Only a drop that stands out from what is around it (its sharp share well above
                // the blurred one), in liquid that is clearly the other side: a finely stirred
                // mixture keeps its soft look (droplets above, tint).
                float in_aq = (1.0 - smoothstep(0.12, 0.25, oil_w)) * (1.0 - resolved);
                float in_ot = smoothstep(0.75, 0.88, oil_w) * (1.0 - resolved);
                float d_o = (oil - max(SPECK_CUT, SPECK_CONTRAST * oil_w)) / gon;
                float d_a = (1.0 - oil - max(SPECK_CUT, SPECK_CONTRAST * (1.0 - oil_w))) / gon;
                sp_o = clamp(d_o * zoom + 0.5, 0.0, 1.0) * in_aq;
                sp_a = clamp(d_a * zoom + 0.5, 0.0, 1.0) * in_ot;
                float rim_o = 1.0 - clamp((d_o - max(0.9 * px_cells, 0.12)) * zoom + 0.5, 0.0, 1.0);
                float rim_a = 1.0 - clamp((d_a - max(0.9 * px_cells, 0.12)) * zoom + 0.5, 0.0, 1.0);
                fc = mix(fc, mix(oc, deepen(ob, 0.35), 0.40 * rim_o), sp_o);
                vec3 bead = mix(lighten(base, 0.10), aq, 0.5);
                fc = mix(fc, mix(bead, deepen(base, 0.45), 0.40 * rim_a), sp_a);
                oily = clamp(oily + sp_o - sp_a, 0.0, 1.0);
            }
            vec3 tint_c = mix(base, ob, oily);
            // How cloudy it is here: what its particles scatter within a few cells.
            float cloudy = 1.0 - exp(-0.35 * mix(kaq, kot, oily).a);
            float clear = (1.0 - smoothstep(0.02, 0.5, max(opac, cloudy))) * (1.0 - metal);
            float translucent = (1.0 - smoothstep(0.75, 0.97, max(opac, cloudy))) * (1.0 - smoothstep(0.3, 0.7, cloudy)) * (1.0 - metal);

            // -- platelets: lined up by the flow (look.comp, the look field) --
            // Each particle carries the orientation of its platelets (cloud_shape < 1), so the
            // pattern is the liquid's: it moves and folds as the liquid does, a still body keeps
            // it until Brownian rotation undoes it. Platelets turned square to the half vector
            // reflect what they scatter towards the viewer (a silky sheen); on edge they let
            // the body show, a little darker. `plate` is how much of the liquid's look is
            // their scattering (material.rs).
            if (lkf.a > 0.0 && plate > 0.02) {
                // (H.A.H: a third where the platelets lie every which way; sh 0.5 there.)
                float sh = clamp((lkf.x - 0.333) * 2.2 + 0.5, 0.0, 1.0);
                float pk = clamp(plate, 0.0, 1.0) * lkf.a;
                fc *= 1.0 - 0.40 * pk * (1.0 - sh);
                fc = mix(fc, lighten(fc, 0.55), 0.9 * pk * sh * sh * mix(0.6, 1.0, il.g));
            }

            // -- the free surface --
            // A clear liquid seen from the side shows its surface as a thin bright line (light
            // reflected inside it, from below); every liquid reflects more at a grazing angle
            // (Fresnel, below), which rounds the rest of its edge.
            float glossy = max(oily, smoothstep(0.85, 1.0, gloss));
            float rim_w = max(1.1 * px_cells, 0.14);
            float rim_line = 1.0 - clamp((d - rim_w) * zoom + 0.5, 0.0, 1.0);
            float topside = smoothstep(0.15, 0.6, n.y);
            fc = mix(fc, lighten(tint_c, 0.72), 0.55 * rim_line * topside * clear * (1.0 - thin) * near_surface);
            // Light that has just come in through the top of a clear liquid: its first cells
            // are paler (sunlit shallows), over a fine darker line where the surface's own
            // reflection ends. An opaque liquid has a lit edge and a shaded one instead.
            float body_k = (1.0 - thin) * near_surface * (1.0 - metal);
            fc = mix(fc, lighten(tint_c, 0.42), 0.20 * clear * topside * (1.0 - smoothstep(0.0, 2.6, d)) * body_k);
            fc *= 1.0 - 0.16 * band(d, rim_w, rim_w + max(1.3 * px_cells, 0.16), zoom) * body_k * mix(0.5, 1.0, clear);
            fc = mix(fc, lighten(fc, 0.30), 0.55 * rim_line * up * (1.0 - clear) * body_k);
            fc *= 1.0 - 0.25 * rim_line * (1.0 - up) * (1.0 - clear) * body_k;
            // Hot liquid glows from its own temperature: the glow replaces the body's colour
            // as it brightens (a white-hot melt does not show its colour underneath).
            // A droplet glows with its own heat, not the cell's average with the liquid
            // around it (the aqueous liquids are never hot enough to glow): a drop of lava in
            // water glows, a drop of water in lava does not.
            vec3 glow = max(mix(em.rgb, co.em.rgb / max(sty.w - sty.z, 1e-4), sp_o) * (1.0 - sp_a) - lumc, 0.0);
            // Where a melt shares its cells with an oil-like liquid, its glow is where its
            // particles are: the cell's emission over the melt's share of the cell, times its
            // share here.
            if (co.dn.x > 1e-4 && co.sty.w - co.sty.z - co.dn.x > 0.02 * co.sty.w)
                glow *= clamp(((a.z + a.w) / max(W, 1e-4)) / max(co.dn.x / max(co.sty.w, 1e-4), 1e-3), 0.0, 1.5);

            // Thin-film colours (oil): a film a few hundred nanometres thick reflects some
            // wavelengths and cancels others (oil on water: phase flip at the top face only,
            // R ~ sin^2(2 pi n t / lambda), n = 1.47), averaged over a little of the spectrum
            // around each primary so that thick films go pastel as real ones do: Newton's
            // sequence from silver through gold, magenta and blue to green. Its thickness drifts
            // with the flow in broad swirls and thins towards the edges; it shows where the
            // surface reflects (at its top and in thin layers), not through the body.
            if (irid > 0.01 && oily > 0.01) {
                // (Broad swirls that the flow draws out, drifting slowly even on still oil.)
                // (From the scalar the oil's particles carry, look.comp: its swirls are folded by
                // the flow and never slide over the oil.)
                float swirl = carried_noise(world, kl, 0.11, 3.0, 0.03) * 0.7 + carried_noise(world, ks, 0.19, 8.0, 0.05) * 0.3;
                float tnm = 180.0 + 720.0 * swirl + 160.0 * clamp(d, 0.0, 2.0);
                vec3 film = vec3(0.0);
                for (int k = -1; k <= 1; k++) {
                    vec3 lam = vec3(640.0, 540.0, 460.0) + 22.0 * float(k);
                    vec3 sn = sin(6.2832 * 1.47 * tnm / lam);
                    film += sn * sn;
                }
                film /= 3.0;
                float where = max(1.0 - smoothstep(0.0, 2.2, d), thin) * oily * irid * detail(0.11, zoom);
                // (A sheen over the body's own colour: the reflected light adds to it.)
                fc += (to_srgb(film) - 0.35) * 0.42 * where;
            }

            // Light through a translucent liquid (honey, slime, oil): where its surface faces
            // away from the light, the light that came through the body leaves it there, coloured
            // by the way through: the warm glow on the far side of a drop of honey.
            if (translucent > 0.02) {
                vec4 kk = mix(kaq, kot, oily);
                float away = smoothstep(0.05, 0.75, -dot(n_w, LIGHT));
                float at_edge = 1.0 - smoothstep(0.2, 2.6, d);
                vec3 through = to_srgb(KEY * mix(il_aq, il_ot, oily) * exp(-kk.rgb * 0.8) * 0.9);
                fc = mix(fc, max(fc, through), 0.75 * translucent * away * at_edge * (1.0 - thin * 0.5) * mix(1.0, 0.6, oily * irid));
            }

            // Reflection (Fresnel): edges facing up mirror the lit top of the tank, those
            // facing down the dark floor; nearby glow too. Then the light's own highlight.
            float shine = gloss;
            // Fresnel (Schlick) from the refractive index: water reflects 2 % head on, glass 4 %.
            float f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
            float fres = f0 + (1.0 - f0) * pow(1.0 - N.z, 5.0);
            fres *= 1.0 + 0.5 * (ior - 1.33);
            float sky_up = smoothstep(-0.6, 0.9, N.y / max(length(N.xy), 1e-3));
            vec3 env = mix(vec3(0.04, 0.05, 0.07), vec3(0.62, 0.68, 0.78), sky_up) + 1.2 * glf.rgb;
            fc = mix(fc, env, 0.75 * fres * (1.0 - metal) * (0.35 + 0.65 * shine));
            vec3 Hh = normalize(L3 + vec3(0.0, 0.0, 1.0));
            // (Sharp highlights only where they span a few pixels: a thinner one sparkles
            // from frame to frame.)
            float zk = clamp(zoom / 14.0, 0.2, 1.0);
            float shin = mix(30.0, 420.0, shine) * zk;
            float nh = max(dot(N, Hh), 0.0);
            float spec = pow(nh, shin) * mix(0.2, 1.15, shine) * mix(0.6, 1.0, zk) * (1.0 - 0.4 * metal);
            fc += vec3(1.0, 0.97, 0.92) * spec;
            // The soft box's broad reflection around the highlight, on a glossy liquid.
            fc += vec3(0.95, 0.97, 1.0) * 0.10 * shine * pow(nh, 7.0) * near_surface * (1.0 - metal) * (1.0 - thin);
            // (Glitter is drawn as sprites over the shaded liquid: sprite.vert.)
            // Liquid metal mirrors the glow around it.
            fc += metal * ob * glf.rgb * 1.5;

            // (Bubbles of dissolved gas, foam and spray are particles of their own, drawn as
            // sprites over the shaded liquid: spray.comp, sprite.vert.)

            // -- ice: fine veins of trapped air and cracks through it, a frosted edge --
            float ice_k = smoothstep(0.5, 0.62, mix(co.fz.x / max(sty.z, 1e-4), fz_ot, oily)) * (1.0 - metal) * (1.0 - smoothstep(0.02, 0.3, hot_in(em.rgb)));
            if (ice_k > 0.01) {
                float vein = 1.0 - carried_seams(world, kl, 0.17, 0.05, zoom, 71.0);
                fc = mix(fc, lighten(fc, 0.55), 0.30 * vein * ice_k * detail(0.17, zoom));
                fc = mix(fc, vec3(0.96, 0.98, 1.0), 0.40 * rim_line * ice_k);
            }

            // -- molten rock: the glow churns, and as it cools a crust forms in plates --
            // The crust: where the melt has cooled into its freezing range (a glow of about
            // 1.2 at lava's liquidus, 0.85 at its solidus), and its quenched skin where it meets
            // water or air (cell_optics.comp) from the moment it does.
            float glow_lvl = max(glow.r, max(glow.g, glow.b));
            float cgf = detail(0.34, zoom);
            // (Both the skin and the crackle are shares of the cell's liquid: the skin per unit
            // of melt is their ratio.)
            float skin_k = clamp(skin / max(crackle, 0.05), 0.0, 1.0) * smoothstep(0.0, 0.25, crackle) * smoothstep(0.2, 0.6, oily);
            if (crackle > 0.01 && (glow_lvl > 0.0 || skin_k > 0.01) && cgf > 0.0) {
                // Molten: where the melt rests, its glow swells and fades slowly (it convects
                // inside); where it flows, the flow draws hotter and cooler melt out into
                // bright streaks along it, carried and stretched with the liquid: the more it
                // has been sheared (the flow map's strain), the more of them.
                Carry km = carry(world, false, true);
                float flowing = smoothstep(0.08, 0.7, km.strain);
                vec2 cv = carried_noise2(world, km, 0.25, 11.0, 0.08);
                float churn = cv.x;
                float vein = 1.0 - abs(2.0 * cv.y - 1.0);
                vein = pow(vein, 6.0) * smoothstep(0.35, 0.75, churn) * detail(0.3, zoom) * mix(0.15, 1.0, flowing);
                glow *= mix(1.0, 0.93 + mix(0.28, 0.55, flowing) * (churn - 0.5) * 2.0 * 0.5 + 0.4 * vein, crackle * cgf);
                glow.g *= 1.0 + 0.18 * vein * crackle;
                // (Not where it is mixed with another liquid from its own cooling: there a
                // cell's temperature jumps with the share of each from frame to frame, and
                // the plates and seams flickered; the skin is steady there.)
                float crust = crackle * cgf * (1.0 - smoothstep(0.7, 1.22, glow_lvl)) * smoothstep(0.85, 0.97, os);
                if (crust > 0.0) {
                    // Plates of crust carried by the melt, which keep their size: where the melt
                    // spreads (the stretch of the flow map) the crust tears and its seams open
                    // onto the glowing melt; where it is pushed together they close.
                    Carry kc = carry(world, true, true);
                    float tear = clamp(2.5 * kc.stretch, -0.5, 1.0);
                    float open = (1.0 + 1.6 * max(tear, 0.0)) * (1.0 + 0.6 * min(tear, 0.0));
                    float plate = carried_seams(world, kc, 0.34, 0.22 * open, zoom, 7.0);
                    plate = min(plate, mix(1.0, carried_seams(world, kc, 1.0, 0.14 * open, zoom, 13.0), 0.6 * detail(1.0, zoom)));
                    float seam = 1.0 - plate;
                    vec3 rock = ob * (0.7 + 0.6 * mix(0.5, carried_noise(world, kc, 1.6, 21.0, 0.0), detail(1.6, zoom)));
                    fc = mix(fc, rock, crust * plate);
                    glow *= mix(1.0, mix(0.08, 1.0, seam) * (1.0 + 0.7 * seam + 0.8 * max(tear, 0.0) * seam), crust);
                    glow.g *= 1.0 + 0.25 * seam * crust;
                }
                // The quenched skin (cell_optics.comp): a rind of dark glassy rock RIND cells
                // wide along the melt's outline against air (d) or water (od), from the moment
                // it meets them, with a few dim cracks across it and the melt glowing hot just
                // inside it. (The skin field is per cell and smoothed, reaching about two cells
                // in from the outline: only its presence decides, RIND the width.)
                float to_edge = min(d, od > -0.5 ? max(od, 0.0) : 1e3);
                float skin_on = smoothstep(0.02, 0.12, skin_k) * cgf;
                float rind = skin_on * (1.0 - clamp((to_edge - RIND) * zoom + 0.5, 0.0, 1.0));
                if (rind > 0.0) {
                    float crack = 1.0 - carried_seams(world, kl, 0.22, 0.10, zoom, 41.0);
                    vec3 glass = ob * (0.55 + 0.35 * mix(0.5, carried_noise(world, kl, 2.3, 5.0, 0.0), detail(2.3, zoom)));
                    fc = mix(fc, glass, rind * (1.0 - 0.8 * crack));
                    glow *= mix(1.0, 0.04 + 0.5 * crack, rind);
                }
                // The hot line just inside the rind.
                float under = skin_on * band(to_edge, RIND, RIND + max(1.2 * px_cells, 0.25), zoom);
                glow *= 1.0 + 0.6 * under;
            }
            float hot = clamp(max(glow.r, max(glow.g, glow.b)), 0.0, 1.0);
            fc = fc * (1.0 - 0.85 * hot) + glow * (1.0 - 0.3 * thin);
            // The liquid's own light (`luminous`): from within, fuller where the body is
            // deep, so its surface, bubbles and edges still read against it.
            float lum_l = max(lumc.r, max(lumc.g, lumc.b));
            if (lum_l > 1e-3) fc = fc * (1.0 - 0.3 * min(lum_l, 1.0)) + to_srgb(lumc) * (0.30 + 0.38 * smoothstep(0.0, 4.0, d)) * (1.0 - 0.3 * thin);
            // Bubbles of steam (README, "Steam"). The bubbles are part of the body in the sharp
            // field (splat.vert), so the liquid stays whole around them; here the vapour field
            // draws them: crowded bubbles merge into one pocket with a smooth wall. Inside,
            // vapour is clearer and paler than the liquid; its wall is a thin bright line,
            // brightest towards the light, with a glint just inside it there and a faint
            // second rim opposite, where light comes through the bubble.
            if (vap.r > 0.04) {
                vec2 gV = vec2(vapour(px + vec2(e, 0.0)).r - vapour(px - vec2(e, 0.0)).r,
                               vapour(px - vec2(0.0, e)).r - vapour(px + vec2(0.0, e)).r) * ge;
                float gvn = length(gV) + 1e-5;
                float db = (vap.r - BUBBLE_CUT) / gvn;      // cells inside the bubble's wall
                vec2 nb = -gV / gvn;                        // out of the bubble
                float in_b = clamp(db * zoom + 0.5, 0.0, 1.0);
                float lit_b = dot(nb, LIGHT);
                float rim_wb = max(1.1 * px_cells, 0.13);
                float wall = 1.0 - clamp((db - rim_wb) * zoom + 0.5, 0.0, 1.0);
                // Paler the more vapour there is (a lone bubble just over the cut, a pocket of
                // pure steam at about 4): a froth shows its thin and dense parts.
                float rich = smoothstep(BUBBLE_CUT, 3.2, vap.r);
                vec3 bc = mix(fc, lighten(base, mix(0.16, 0.42, rich)), mix(0.45, 0.75, rich));
                // The froth inside a pocket: the walls between its bubbles (the rings each
                // lays, steam.frag), faint, and only where they are wider than a pixel.
                float web = smoothstep(0.35, 0.9, vap.b) * (1.0 - smoothstep(1.3, 2.2, vap.b)) * smoothstep(0.0, 0.25, db) * detail(0.7, zoom);
                bc = mix(bc, lighten(base, 0.6), 0.45 * web);
                bc = mix(bc, lighten(base, 0.82), wall * mix(0.30, 0.95, smoothstep(-0.3, 0.6, lit_b)));
                float bglint = band(db, rim_wb + 0.08, rim_wb + 0.30, zoom) * smoothstep(0.55, 0.9, lit_b);
                bc = mix(bc, vec3(1.0), 0.75 * bglint * detail(0.5, zoom));
                float back = band(db, rim_wb, rim_wb + 0.22, zoom) * smoothstep(0.4, 0.9, -lit_b);
                bc = mix(bc, lighten(glowup(base), 0.45), 0.35 * back * detail(0.5, zoom));
                fc = mix(fc, bc, in_b);
            }

            // Lit by hot liquid nearby.
            fc += 0.5 * glf.rgb * (0.25 + fc) * (1.0 - hot);
            col = mix(col, fc, alpha);
        }
    }
    // Hot liquid throws a glow on the air just around it. (The cell fields reach a little
    // past the surface.)
    col += 0.35 * em.rgb * min(sty.w, 1.0) * clamp(1.0 - W / pc.threshold, 0.0, 1.0);

    // ---- solids: dark slate with a lit bevel; wet where liquid touches --------------------
    // A soft shadow grounds them on the backdrop.
    col *= 1.0 - 0.20 * exp(-max(sd, 0.0) / 2.5) * clamp(1.0 - W / pc.threshold, 0.0, 1.0);
    if (sd < 1.5 * pc.inv_zoom) {
        float inside = clamp(0.5 - sd * zoom, 0.0, 1.0);
        float hs = 0.5;
        vec2 gs = vec2(sample_sdf(world + vec2(hs, 0.0)) - sample_sdf(world - vec2(hs, 0.0)),
                       sample_sdf(world + vec2(0.0, hs)) - sample_sdf(world - vec2(0.0, hs)));
        float facing = dot(gs / (length(gs) + 1e-5), LIGHT);
        vec3 body = mix(vec3(0.205, 0.225, 0.285), vec3(0.165, 0.180, 0.230), smoothstep(0.0, 6.0, -sd));
        // Stone: a broad mottle and a fine grain, fixed to the solid.
        body *= 0.92 + 0.16 * gnoise(world * 0.23 + 3.7);
        body *= mix(1.0, 0.93 + 0.14 * gnoise(world * 0.6), detail(0.6, zoom));
        // Beyond the tank's own wall there is nothing: it stands on a dark ground.
        vec2 out_of = max(max(-world, world - vec2(pc.grid)), vec2(0.0));
        body = mix(body, vec3(0.040, 0.045, 0.062), smoothstep(0.0, 3.0, max(out_of.x, out_of.y)));
        float bevel_w = max(1.5 * pc.inv_zoom, 0.9);
        float bevel = 1.0 - clamp((-sd - bevel_w) * zoom + 0.5, 0.0, 1.0);
        vec3 edge = mix(vec3(0.150, 0.165, 0.215), vec3(0.470, 0.520, 0.640), smoothstep(-0.3, 0.8, facing));
        vec3 sc = mix(body, edge, bevel * 0.9);
        // Wet: the bevel takes the colour of the liquid lying against it now (read just
        // outside the surface, see liquid_at) ...
        vec2 n_s = gs / (length(gs) + 1e-5);
        vec2 against = world + n_s * (0.9 - sd);
        vec3 tint = liquid_at(against, mix(base, ob, smoothstep(0.2, 0.6, oil_w)));
        float wet = smoothstep(0.10, 0.42, T);
        // ... and of the liquid that has run off it, until it dries: each liquid in its own
        // time (water in a moment, oil slowly, honey hardly at all). The map is read just
        // outside the surface, bilinearly over the texels that have been wet.
        vec2 mp = against - 0.5;
        ivec2 m0 = ivec2(floor(mp));
        vec2 mf = mp - vec2(m0);
        vec3 film = vec3(0.0);
        float left = 0.0, mw = 0.0;
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                vec4 m = texelFetch(wet_map, clamp(m0 + ivec2(i, j), ivec2(0), pc.grid - 1), 0);
                if (m.w <= 0.0) continue;
                float bw = (i == 0 ? 1.0 - mf.x : mf.x) * (j == 0 ? 1.0 - mf.y : mf.y);
                vec4 f = film_of(m);
                film += bw * f.rgb;
                left += bw * exp(-max(pc.now - m.z, 0.0) / max(f.w, 0.1));
                mw += bw;
            }
        }
        if (mw > 1e-3) {
            film /= mw;
            float stays = smoothstep(0.0, 0.5, mw) * left / mw;
            tint = mix(film, tint, wet);
            wet = max(wet, stays * 0.8);
        }
        wet *= pc.wet_tint;
        float film_w = max(1.5 * pc.inv_zoom, pc.wet_width);
        float film_band = 1.0 - smoothstep(0.3 * film_w, film_w, -sd);
        // Wet stone darkens (water fills its pores), and takes a little of the liquid's tint.
        sc = mix(sc, mix(sc * 0.62, tint * 0.8, 0.45), wet * film_band);
        // What has stained its face: the film map, laid along its edges and run down it.
        sc = draw_film(sc, world, zoom);
        // A wet edge catches the light.
        sc += 0.20 * wet * (1.0 - clamp((-sd - 0.35 * bevel_w) * zoom + 0.5, 0.0, 1.0)) * smoothstep(0.1, 0.8, facing);
        // Lit by hot liquid nearby, most along the edge facing it.
        if (pc.glow_k > 0.0) sc += 0.8 * pc.glow_k * glow_at(against).rgb * (sc * 2.5 + 0.03) * mix(0.25 * exp(min(sd, 0.0) / 3.0), 1.0, bevel);
        col = mix(col, sc, inside);
    }

    // ---- steam that escaped (README, "Steam"): the smoke field ------------------------------
    // A thin scattering mist: how much of the backdrop it hides goes with the vapour along
    // the view (Beer-Lambert); how bright it is, with how much vapour lies between it and the
    // light (upper left), so thick billows have a lit side and a shaded side and thin wisps
    // are lit all through. Hot liquid below lights it from underneath.
    if (HAS_VAPOUR != 0u && pc.vapour > 0.0) {
        // Only in the open: not over a solid or past the tank's walls (the field is filtered
        // across the solid's outline and clamps at the grid's edge).
        float D = smoke_at(world) * clamp(sample_sdf(world) * zoom + 0.5, 0.0, 1.0);
        if (D > 0.002) {
            float shade = 0.0;
            for (int k = 1; k <= 4; k++) shade += smoke_at(world + LIGHT * (1.6 * float(k)));
            float lit = exp(-STEAM_SHADOW * 1.6 * shade);
            vec3 cloud = mix(vec3(0.62, 0.66, 0.72), vec3(0.97, 0.98, 1.0), lit);
            cloud += 1.4 * glow_here.rgb;
            // Steam absorbs next to nothing: it adds the light it scatters and dims what is
            // behind it only a little (mixed over the backdrop as a cover it read as grey
            // smoke).
            float cover = STEAM_COVER * (1.0 - exp(-STEAM_OPACITY * D));
            col = col * (1.0 - 0.3 * cover) + cloud * cover * 0.8;
        }
    }

    // ---- bloom: the glow of hot liquid spreads over everything ---------------------------
    col += 0.5 * glow_here.rgb;
    // A soft shoulder above 0.8, so that bright glow and highlights roll off into white
    // instead of clipping each channel (everything below is untouched).
    vec3 over = max(col - 0.8, 0.0);
    col = min(col, vec3(0.8)) + 0.2 * (1.0 - exp(-over / 0.2));

    // Break up banding in the dark gradients (a fixed pattern: it does not move).
    float dither = fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715)))) - 0.5;
    f_color = vec4(col + dither / 255.0, 1.0);
}
