#version 460

#extension GL_GOOGLE_include_directive : enable
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
// The stain map (stain.comp), one texel per cell: colour x cover, cover.
layout(set = 0, binding = 6) uniform sampler2D stain_map;

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
} pc;

layout(location = 0) out vec4 f_color;

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
struct CellOptics { vec4 aq, ot, opt, em, sty, fz, fx; };
CellOptics cell_optics(vec2 x) {
    vec2 uv = x / vec2(pc.grid);
    return CellOptics(textureLod(cellopt, vec3(uv, 0.0), 0.0), textureLod(cellopt, vec3(uv, 1.0), 0.0),
                      textureLod(cellopt, vec3(uv, 2.0), 0.0), textureLod(cellopt, vec3(uv, 3.0), 0.0),
                      textureLod(cellopt, vec3(uv, 4.0), 0.0), textureLod(cellopt, vec3(uv, 7.0), 0.0),
                      textureLod(cellopt, vec3(uv, 8.0), 0.0));
}

// Ice: the frozen colour (layer 5 aqueous, 6 the others) where the smoothed frozen share `f` of
// those liquids passes a half. A solid has an edge, not a fade: the step is a few pixels wide
// at any zoom, along a smooth curve (the share is smoothed over 3 x 3 cells first).
vec3 iced(vec3 c, vec2 x, float layer, float w, float f, float zoom) {
    if (f <= 0.02 || w <= 1e-4) return c;
    vec4 ice = textureLod(cellopt, vec3(x / vec2(pc.grid), layer), 0.0);
    float edge = clamp(0.5 / zoom, 0.04, 0.25);  // half width of the step, in frozen share (about 2 px)
    return mix(c, resolve_colour(ice / w), smoothstep(0.5 - edge, 0.5 + edge, f));
}

float total(vec4 g) { return g.x + g.y + g.z + g.w; }

// The leading group's share less the next one's, for the groups a, b chosen at this pixel:
// positive on a's side of an interface, zero on it.
float lead(vec4 g, int a, int b) { return (g[a] - g[b]) / max(total(g), 1e-4); }

// Colour and drying time of a wet-map texel (wet.frag): material row + 1, concentration.
vec4 film_of(vec4 m) {
    int row = int(m.x + 0.5) - 1;
    Optics o = optics[max(row, 0)];
    int d = o.idx.x;
    vec3 c = oklab_to_srgb(o.lab.xyz);
    float dry = o.sty.z;
    if (d >= 0) {
        float s = m.y >= 0.0 ? clamp(m.y, 0.0, 1.0) : 1.0;
        c = resolve_colour(blend_colour(optics[d], o, optics[d].lab, o.lab, s));
        dry = mix(optics[d].sty.z, dry, s);
    }
    return vec4(c, dry);
}

// The colour of the liquid in the cells at `x`, aqueous and others by their share there;
// `fallback` where those cells hold none. Solids read it just outside their surface: the
// cells inside a solid hold no liquid, and reading the colour there drew wet edges black in
// cell-sized steps.
vec3 liquid_at(vec2 x, vec3 fallback) {
    vec2 uv = x / vec2(pc.grid);
    vec4 sty = textureLod(cellopt, vec3(uv, 4.0), 0.0);
    if (sty.w < 1e-3) return fallback;
    vec3 a = resolve_colour(textureLod(cellopt, vec3(uv, 0.0), 0.0) / max(sty.z, 1e-4));
    vec3 o = resolve_colour(textureLod(cellopt, vec3(uv, 1.0), 0.0) / max(sty.w - sty.z, 1e-4));
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

// Noise carried by the flow `u` (cells / s), restarted every `period` seconds: two copies
// half a period apart, each faded out when it restarts (a flow map, Vlachos 2010), each
// restart at a new place in the pattern so that it never repeats. The pattern drifts at
// ADVECT times the flow: carried all the way, shear drew it out into streaks within a
// period.
const float ADVECT = 0.35;
float flow_noise(vec2 x, vec2 u, float freq, float period, float seed) {
    float ph = pc.now / period + seed;
    float t0 = fract(ph), t1 = fract(ph + 0.5);
    float w0 = 1.0 - abs(2.0 * t0 - 1.0);
    vec2 o0 = hash22(vec2(floor(ph), seed)) * 97.0, o1 = hash22(vec2(floor(ph + 0.5), seed + 0.5)) * 97.0;
    float n0 = gnoise((x - u * (ADVECT * t0 * period)) * freq + o0);
    float n1 = gnoise((x - u * (ADVECT * t1 * period)) * freq + o1);
    // (The blend of two is flatter than either: stretch it back.)
    return clamp(0.5 + (mix(n1, n0, w0) - 0.5) / sqrt(w0 * w0 + (1.0 - w0) * (1.0 - w0)), 0.0, 1.0);
}

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

// The same seams carried by the flow (as flow_noise): 0 on a seam, 1 well inside a plate.
// `width` in pattern cells, but never under about a pixel and a half: thinner seams winked
// on and off as they drifted across pixels.
float flow_seams(vec2 x, vec2 u, float freq, float period, float width, float zoom) {
    width = max(width, 1.5 * freq / zoom);
    float ph = pc.now / period;
    float t0 = fract(ph), t1 = fract(ph + 0.5);
    float w0 = 1.0 - abs(2.0 * t0 - 1.0);
    vec2 o0 = hash22(vec2(floor(ph), 7.0)) * 61.0, o1 = hash22(vec2(floor(ph + 0.5), 7.5)) * 61.0;
    float s0 = smoothstep(0.0, width, seams((x - u * (ADVECT * t0 * period)) * freq + o0));
    float s1 = smoothstep(0.0, width, seams((x - u * (ADVECT * t1 * period)) * freq + o1));
    return mix(s1, s0, w0);
}

// The glow field at a world position, filtered with a cubic B-spline (four bilinear taps):
// the field is at half the grid's resolution, and plain bilinear filtering of a bright,
// smooth glow drew the texels as a faint lattice of diamonds and creases. Where the
// bilinear sample says there is no glow at all, that one tap is enough.
vec4 glow_at(vec2 x) {
    vec2 size = vec2(textureSize(glow_map, 0));
    vec2 uv = x / (2.0 * size);
    vec4 near = textureLod(glow_map, uv, 0.0);
    if (max(max(near.r, near.g), max(near.b, near.a)) < 1e-4) return near;
    vec2 p = uv * size - 0.5, i = floor(p), f = p - i;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
    vec2 g0 = w0 + w1, g1 = w2 + w3;
    vec2 h0 = (i - 0.5 + w1 / g0) / size, h1 = (i + 1.5 + w3 / g1) / size;
    return g0.y * (g0.x * textureLod(glow_map, vec2(h0.x, h0.y), 0.0) + g1.x * textureLod(glow_map, vec2(h1.x, h0.y), 0.0))
         + g1.y * (g0.x * textureLod(glow_map, vec2(h0.x, h1.y), 0.0) + g1.x * textureLod(glow_map, vec2(h1.x, h1.y), 0.0));
}

// Stain cover and colour (premultiplied) at a world position, with the runs a stain makes
// down a wall: in thin streaks, each column of its own length, a stain reaches a few cells
// below where it was laid. `cx` places the columns: the point's own x on the back wall; on
// a solid, where `x` is the point just outside its surface, the pixel's x, so that the
// runs streak down the face of a wall.
vec4 stain_at(vec2 x, float cx, float zoom) {
    vec2 ig = 1.0 / vec2(pc.grid);
    vec4 s = textureLod(stain_map, x * ig, 0.0);
    float col = floor(cx * 1.25);
    float len = 0.8 + 9.0 * pow(hash12(vec2(col, 3.7)), 3.0);
    float across = 1.0 - smoothstep(0.12, 0.36, abs(fract(cx * 1.25) - 0.5));
    across *= detail(2.5, zoom);
    // (Only where a streak is drawn: most pixels skip the three reads.)
    for (int k = 1; k <= 3 && across > 0.0; k++) {
        float f = float(k) / 3.0;
        vec4 a = textureLod(stain_map, (x + vec2(0.0, len * f)) * ig, 0.0);
        s = max(s, a * (1.0 - 0.75 * f) * across);
    }
    return s;
}

// How a stain shows: patchy at its fringe (where little was laid) and solid where much
// was, in the colour it dries to (darker, a little duller). The patches are broad (a few
// cells) and soft: a fine second noise (two features a cell) with a hard threshold drew
// the fringe as dirt-like speckle, and on a small solid (a peg) as a dark asterisk.
vec4 stain_look(vec4 s, vec2 x, float zoom) {
    if (s.w < 0.01) return vec4(0.0);
    vec3 c = s.rgb / s.w;
    float blotch = mix(0.5, 0.78 * gnoise(x * 0.32) + 0.22 * gnoise(x * 0.85 + 7.3), detail(0.85, zoom));
    float vis = smoothstep(0.0, 0.45, 1.3 * s.w - 0.7 * blotch);
    vec3 dried = mix(c, vec3(luma(c)), 0.2) * 0.62;
    return vec4(dried, vis);
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
    if (pc.stain_k > 0.0) {
        vec4 st = stain_look(stain_at(x, x.x, zoom), x, zoom);
        col = mix(col, st.rgb, 0.8 * st.w * pc.stain_k);
    }
    // Shadow: the liquid up and to the left (towards the light), from the wide field.
    vec2 ps = p - vec2(0.9, 1.3) * zoom;
    float Ts = total(textureLod(wide, min(ps * pc.wide_scale, pc.wide_max), 0.0)) / pc.rest;
    col *= 1.0 - 0.28 * smoothstep(0.04, 0.5, Ts);
    col += 0.8 * glow * (col * 3.0 + 0.04);
    return col;
}

void main() {
    vec2 px0 = gl_FragCoord.xy;
    float zoom = 1.0 / pc.inv_zoom;
    vec2 world0 = pc.cam + (px0 - pc.half_res) * vec2(pc.inv_zoom, -pc.inv_zoom);

    // (Until 2026-11-14 a heat haze displaced the picture above hot liquid with rising
    // noise, made the bloom waver with it and drew embers. It read as low resolution: the
    // displacement followed the half-resolution glow field, the wavering bloom went negative
    // in places and drew a dark, streaky band over every melt. The glow is now steady.)
    vec2 px = px0;
    vec2 world = world0;

    // ---- the back wall, and a vignette --------------------------------------------------
    vec2 vg = (px0 - pc.half_res) / pc.half_res;
    float vignette = 1.0 - 0.22 * smoothstep(0.5, 1.5, length(vg));
    // The glow field here (light on the back wall, liquids and solids; bloom).
    vec4 glow_here = pc.glow_k > 0.0 ? glow_at(world) * pc.glow_k : vec4(0.0);
    vec3 col = backdrop(world, px, zoom, glow_here.rgb, true) * vignette;
    float h = clamp(world.y / float(pc.grid.y), 0.0, 1.0);

    // ---- the fields ------------------------------------------------------------------
    // Differences are taken over a fixed share of the kernel, so that they stay well above
    // the precision of the target however far the view is zoomed in.
    float e = max(pc.accum_div, 0.3 * zoom);
    vec4 a = field(px);
    vec4 ax0 = field(px - vec2(e, 0.0)), ax1 = field(px + vec2(e, 0.0));
    vec4 ay0 = field(px + vec2(0.0, e)), ay1 = field(px - vec2(0.0, e));
    float W = total(a);
    // Gradients per cell, +y up.
    float ge = 0.5 * zoom / e;
    vec2 gW = vec2(total(ax1) - total(ax0), total(ay1) - total(ay0)) * ge;

    vec2 wuv = min(px * pc.wide_scale, pc.wide_max);
    vec4 b = textureLod(wide, wuv, 0.0);
    float Wb = total(b);
    // How much liquid is around, 0..1. Next to a solid, part of the blur lies inside it and
    // can hold none: count only the open part (for a flat wall at distance sd, the share of
    // a Gaussian beyond it), or every wall and floor would read as a surface.
    float sd = sample_sdf(world);
    float open = 0.5 + 0.5 * tanh(1.2 * max(sd, -0.5) / (WIDE_SIGMA * 1.4142));
    float T = Wb / (pc.rest * open);

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

    // Body colours, each from its own liquids only. Next to an interface between two of
    // the others the colour is read a little way into this side, so that each keeps its
    // own colour right up to the line.
    // (The optics come per cell, soft as blends are; immiscible edges are kept by the split
    // into two colours.)
    // (Empty air with no liquid nearby needs none of it: most of the screen.)
    bool near = W > 1e-4 || T > 0.003;
    CellOptics co = near ? cell_optics(world) : CellOptics(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    vec4 sty = co.sty;
    // (Zero optics resolve to black.)
    vec3 base = near ? resolve_colour(co.aq / max(sty.z, 1e-4)) : vec3(0.0);
    if (near) base = iced(base, world, 5.0, co.fz.x, co.fz.x / max(sty.z, 1e-4), zoom);
    // Whether any of the other liquids is within reach of the blur. Where none is, every use
    // of their colour below is weighted by a share that is zero, and skipping it (with the
    // shading of the others) keeps aqueous-only pixels from paying for it.
    bool others = oil > 0.0 || oil_w > 0.0 || mixed_ot > 0.0;
    vec2 off = n_if * clamp(1.2 - d_if, 0.0, 1.2) * mixed_ot * zoom;
    vec2 opx = px + vec2(off.x, -off.y);
    vec2 owld = pc.cam + (opx - pc.half_res) * vec2(pc.inv_zoom, -pc.inv_zoom);
    CellOptics oo = (mixed_ot > 0.0) ? cell_optics(owld) : co;
    vec3 ob = others ? resolve_colour(oo.ot / max(oo.sty.w - oo.sty.z, 1e-4)) : vec3(0.0);
    if (others) ob = iced(ob, owld, 6.0, oo.fz.y, oo.fz.y / max(oo.sty.w - oo.sty.z, 1e-4), zoom);
    // Everything else as blended here.
    float wo = max(sty.w, 1e-4);
    vec4 opt = co.opt / wo;
    vec4 em = co.em / wo;
    float opac = opt.x, scat = opt.y, metal = opt.z, gloss = opt.w;
    float body = sty.x / wo, foam_k = sty.y / wo;
    float fast = em.w;
    // The effects (optics table `fx`), and the flow that carries their patterns.
    vec4 fxv = co.fx / wo;
    float glint = fxv.y, irid = fxv.z, crackle = fxv.w;
    vec2 uflow = co.fz.zw / wo;
    uflow *= min(1.0, 10.0 / max(length(uflow), 1e-3));
    vec4 glf = glow_here;

    if (W > 1e-3) {
        // Thick liquid is drawn fuller: syrup holds together in fat, rounded shapes.
        float thick = smoothstep(0.3, 0.9, body) * (1.0 - oil);
        // A lone particle is a droplet (its weight peaks at 1). Against a body of liquid the
        // surface is taken deeper in the field, where several particles overlap: it is then
        // smooth instead of following each particle of the outermost layer.
        // Well inside, the cut drops again: only a real cavity opens a hole, not a spot where
        // the particles happen to sit a little apart (as they do where two liquids meet).
        float t = mix(pc.threshold, mix(BODY_CUT, BODY_CUT_THICK, thick), smoothstep(0.12, 0.38, T));
        t *= 1.0 - 0.75 * smoothstep(0.62, 0.86, T);
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

            // -- the aqueous liquids --
            // Clear ones (water) are pale in the shallows under the surface and where there is
            // little of them, darker in the depths; opaque or thick ones (syrup, mud) are
            // nearly uniform in the body and only their edges let light through.
            float watery = 1.0 - smoothstep(0.03, 0.30, max(opac, body));
            float dense = smoothstep(0.55, 0.95, max(opac, body));
            vec3 aq = mix(lighten(base, 0.16 * watery), deepen(base, mix(0.10, 0.30, max(watery, dense))), deep);
            aq = mix(aq, mix(mix(base, glowup(base), 0.5 * dense), lighten(base, 0.42), watery), thin);
            // Fast clear liquid in bulk runs paler: a jet or a current shows as a ribbon of light.
            float rush = smoothstep(0.05, 0.36, fast) * watery * (1.0 - thin);
            aq = mix(aq, mix(base, vec3(0.50, 0.84, 1.0), 0.75), 0.5 * rush);
            // Fast and thin is white water (for liquids that foam): splashes, spray, wave lips.
            float foam = smoothstep(0.16, 0.5, fast) * (1.0 - smoothstep(0.34, 0.72, T)) * foam_k;
            aq = mix(aq, vec3(0.94, 0.98, 1.0), 0.9 * foam);
            // A thick liquid's sheen: a broad soft light well inside the body, on the lit side.
            float sheen = smoothstep(0.58, 0.68, T) * (1.0 - smoothstep(0.78, 0.90, T)) * smoothstep(0.2, 0.75, lit_w);
            aq = mix(aq, glowup(base), 0.42 * sheen * dense);
            // Scattering (milk, mud): light comes back from inside, the body keeps its colour
            // and depth hardly darkens it.
            aq = mix(aq, mix(lighten(base, 0.10), deepen(base, 0.06), deep), scat * (1.0 - oil));

            // -- the others: bright and glossy (oil), mirror-like (metal), glowing (melt) --
            // (Only where one is near, see `others`.)
            float iw = max(1.1 * px_cells, 0.17);
            vec3 oc = vec3(0.0);
            if (others) {
                oc = mix(lighten(ob, 0.08), mix(ob, deepen(glowup(ob), 0.25), 0.42), deep);
                oc = mix(oc, lighten(ob, 0.40), thin);
                // Metal: a mirror of a soft studio sky, by the surface normal near the surface
                // and by the shape of the body further in.
                // (Inside, the sharp field's gradient follows the particle layers: not a surface.)
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
                    vec3 env = mix(vec3(0.10, 0.11, 0.14), vec3(0.93, 0.95, 0.98), sky * sky);
                    env += vec3(0.5) * smoothstep(0.93, 0.99, dot(nm, LIGHT));
                    oc = mix(oc, ob * env * 1.15, metal);
                }
                oc = mix(oc, mix(lighten(ob, 0.10), deepen(ob, 0.06), deep), scat * oil);
                // Between two of the others (oil on mercury): an ink line on the later group's side.
                float edge_in = 1.0 - clamp((d_if - iw) * zoom + 0.5, 0.0, 1.0);
                oc = mix(oc, deepen(ob, 0.40), 0.80 * edge_in * mixed_ot);
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
            // A thin darker edge so a droplet reads as a bead, not a smudge.
            droplets = mix(droplets, deepen(ob, 0.35), 0.30 * drop * (1.0 - drop) * 4.0);
            vec3 emulsion = mix(droplets, oc, smoothstep(0.55, 0.9, os));

            vec2 gO = vec2((ax1.y + ax1.z + ax1.w) / max(total(ax1), 1e-4) - (ax0.y + ax0.z + ax0.w) / max(total(ax0), 1e-4),
                           (ay1.y + ay1.z + ay1.w) / max(total(ay1), 1e-4) - (ay0.y + ay0.z + ay0.w) / max(total(ay0), 1e-4)) * ge;
            float gon = length(gO) + 1e-5;
            float od = (oil - OIL_CUT) / gon;                // cells inside the other liquid
            float om = clamp(od * zoom + 0.5, 0.0, 1.0);
            // Stray droplets below the cut tint the aqueous liquid near them a little.
            vec3 fc = mix(mix(aq, deepen(mix(ob, base, 0.25), 0.10), OIL_TINT * smoothstep(0.0, OIL_CUT, oil)), oc, om);
            vec3 ink = deepen(mix(ob, glowup(ob), 0.5), 0.30);
            // The interface: an ink line on the other liquid's side, a pale meniscus on the
            // aqueous side.
            float mixed = clamp(smoothstep(0.02, 0.2, oil) - smoothstep(0.8, 0.98, oil), 0.0, 1.0);
            fc = mix(fc, ink, 0.6 * om * (1.0 - clamp((od - iw) * zoom + 0.5, 0.0, 1.0)) * mixed);
            fc = mix(fc, lighten(fc, 0.5), 0.5 * band(-od, 0.0, max(1.0 * px_cells, 0.12), zoom) * mixed);
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
                fc = mix(fc, mix(oc, deepen(ob, 0.35), 0.45 * rim_o), sp_o);
                vec3 bead = mix(lighten(base, 0.10), aq, 0.5);
                fc = mix(fc, mix(bead, deepen(base, 0.45), 0.45 * rim_a), sp_a);
                oily = clamp(oily + sp_o - sp_a, 0.0, 1.0);
            }

            // -- the free surface: a rim of light, in each liquid's own way --
            // Clear: a thin bright line, strongest where it faces the light. The others: a pale
            // line and a hard streak of gloss just inside it. Thick: a broad warm rim, where
            // light gets through its edge.
            float glossy = max(oily, smoothstep(0.85, 1.0, gloss) * dense);
            float rim_w = max(1.2 * px_cells, mix(mix(0.45, 0.16, watery), 0.20, glossy));
            float rim_hard = 1.0 - clamp((d - rim_w) * zoom + 0.5, 0.0, 1.0);
            float rim_soft = 1.0 - smoothstep(0.3 * rim_w, max(rim_w, 0.3 * rim_w + px_cells), d);
            float rim = mix(rim_hard, rim_soft, (1.0 - watery) * (1.0 - glossy));
            vec3 rim_aq = mix(mix(glowup(base), lighten(glowup(base), 0.45), up), lighten(base, 0.78), watery);
            vec3 rim_col = mix(rim_aq, lighten(glowup(mix(base, ob, oily)), 0.6), glossy);
            float rim_a = mix(mix(0.80, mix(0.35, 0.95, up), watery), mix(0.5, 0.9, up), glossy) * (1.0 - 0.5 * metal);
            fc = mix(fc, rim_col, 0.6 * rim * rim_a * mix(1.0, near_surface, metal));
            float streak = band(d, 0.50, 0.92, zoom) * smoothstep(0.45, 0.6, lit) * max(glossy * gloss, metal) * near_surface;
            fc = mix(fc, lighten(glowup(mix(base, ob, oily)), 0.75), 0.45 * streak);
            // Hot liquid glows from its own temperature: the glow replaces the body's colour
            // as it brightens (a white-hot melt does not show its colour underneath).
            // A droplet glows with its own heat, not the cell's average with the liquid
            // around it (the aqueous liquids are never hot enough to glow): a drop of lava in
            // water glows, a drop of water in lava does not.
            vec3 glow = mix(em.rgb, co.em.rgb / max(sty.w - sty.z, 1e-4), sp_o) * (1.0 - sp_a);

            // -- light: a height field over the liquid, lit from the upper left --
            // The surface rounds over within `rad` cells of its edge (fuller for thick
            // liquids); further in, the body's broad shape comes from the wide field. A
            // liquid that glints is rippled with noise carried by its flow, which breaks its
            // highlights into moving glints.
            float rad = mix(0.8, 1.9, max(body, thick));
            float edge_k = 1.0 - smoothstep(0.0, rad, d);
            vec2 slope = n * (3.0 * edge_k * edge_k) * mix(1.0, near_surface, metal)
                       + n_w * min(length(gT) * gw * 1.5, 0.8) * near_surface;
            vec2 rip = vec2(0.0);
            // (Mostly along the surface: deep in the body there is no surface to glint.)
            float gfine = detail(0.8, zoom);
            if (glint > 0.01 && gfine > 0.0 && d < 2.5)
                rip = (vec2(flow_noise(world, uflow, 0.8, 1.4, 1.0), flow_noise(world, uflow, 0.8, 1.4, 2.0)) - 0.5)
                    * (1.0 * glint * gfine * (1.0 - smoothstep(0.0, 2.5, d)));
            vec3 N = normalize(vec3(slope + rip, 1.0));
            vec3 tint_c = mix(base, ob, oily);

            // Clear liquid: the back wall shows through, bent by the surface and tinted by
            // the liquid, with the light its surface focuses (caustics) playing over it.
            float clear = (1.0 - smoothstep(0.02, 0.5, opac)) * (1.0 - metal);
            if (clear > 0.01) {
                // (At most 0.7 of a cell: near the edge, where the slope is steep, a larger
                // shift made the wall's joints swim and flicker as the surface moved.)
                vec2 roff = slope * 0.9 + rip * 1.5;
                roff *= min(1.0, 0.7 / max(length(roff), 1e-4));
                vec3 bd = backdrop(world + roff, px + vec2(roff.x, -roff.y) * zoom, zoom, glf.rgb, false) * vignette;
                float tr = clear * 0.5 * (1.0 - 0.4 * deep) * (1.0 - 0.6 * thin);
                fc = mix(fc, fc * 0.72 + bd * mix(vec3(1.0), tint_c, 0.6) * 1.6, tr);
                float cf = detail(0.3, zoom);
                if (cf > 0.0) {
                    float c1 = flow_noise(world * vec2(1.0, 0.8), uflow, 0.3, 2.2, 4.0);
                    float c2 = flow_noise(world * vec2(1.0, 0.8) + vec2(5.2, 1.3), uflow, 0.3, 2.7, 5.0);
                    float caus = pow(clamp(1.0 - 2.6 * abs(c1 - c2), 0.0, 1.0), 5.0);
                    fc += mix(vec3(0.75, 0.9, 1.0), tint_c, 0.3) * caus * 0.12 * clear * cf * (1.0 - 0.75 * deep) * (1.0 - thin);
                }
            }
            // Thin-film colours (oil): bands that follow the surface and drift with the flow.
            if (irid > 0.01) {
                float film = 1.3 * d + 2.4 * flow_noise(world, uflow, 0.3, 4.0, 3.0);
                vec3 rainbow = 0.5 + 0.5 * cos(6.2832 * (film + vec3(0.0, 0.33, 0.67)));
                fc += (rainbow - 0.3) * 0.32 * irid * detail(0.3, zoom) * (0.12 + 0.6 * edge_k + 0.3 * thin);
            }
            // Light through a thick liquid that is not opaque (honey): just inside its lit
            // surface it glows its own colour, brighter and more golden.
            float translucent = dense * (1.0 - smoothstep(0.6, 0.95, opac)) * (1.0 - oily);
            fc = mix(fc, lighten(glowup(base), 0.1), 0.55 * translucent * (1.0 - smoothstep(0.0, 3.5, d)) * smoothstep(-0.3, 0.7, lit_w));
            // A slow shimmer through a glinting body (honey): light moving inside it.
            if (glint * (0.2 + 0.8 * dense) > 0.08) {
                float sh = smoothstep(0.5, 0.9, flow_noise(world, uflow, 0.14, 3.5, 6.0));
                fc = mix(fc, glowup(fc), 0.3 * glint * sh * (0.2 + 0.8 * dense) * detail(0.14, zoom));
            }
            // Reflection (Fresnel): edges facing up mirror the lit top of the tank, those
            // facing down the dark floor; nearby glow too. Then the light's own highlight.
            float shine = max(gloss, glint);
            float fres = 0.04 + 0.96 * pow(1.0 - N.z, 5.0);
            float sky_up = smoothstep(-0.6, 0.9, N.y / max(length(N.xy), 1e-3));
            vec3 env = mix(vec3(0.05, 0.06, 0.08), vec3(0.60, 0.66, 0.76), sky_up) + 1.2 * glf.rgb;
            fc = mix(fc, env, 0.7 * fres * (1.0 - metal) * (0.35 + 0.65 * shine));
            const vec3 L3 = vec3(-0.3375, 0.67, 0.66);
            vec3 Hh = normalize(L3 + vec3(0.0, 0.0, 1.0));
            // (Sharp highlights only where they span a few pixels: a thinner one sparkles
            // from frame to frame.)
            float zk = clamp(zoom / 14.0, 0.2, 1.0);
            float shin = mix(30.0, 320.0, shine) * zk;
            fc += vec3(1.0, 0.97, 0.92) * pow(max(dot(N, Hh), 0.0), shin) * mix(0.2, 1.0, shine) * mix(0.6, 1.0, zk) * (1.0 - 0.4 * metal);
            // Liquid metal mirrors the glow around it.
            fc += metal * ob * glf.rgb * 1.5;

            // -- molten rock: the glow churns, and as it cools a crust forms in plates --
            float glow_lvl = max(glow.r, max(glow.g, glow.b));
            float cgf = detail(0.34, zoom);
            if (crackle > 0.01 && glow_lvl > 0.0 && cgf > 0.0) {
                // Molten: the glow churns with the flow, in broad swells and thin brighter veins.
                float churn = flow_noise(world, uflow, 0.22, 3.0, 11.0);
                float vein = 1.0 - abs(2.0 * flow_noise(world, uflow, 0.3, 2.5, 12.0) - 1.0);
                vein = pow(vein, 6.0) * smoothstep(0.35, 0.75, churn) * detail(0.3, zoom);
                glow *= mix(1.0, 0.68 + 0.5 * churn + 0.4 * vein, crackle * cgf);
                glow.g *= 1.0 + 0.18 * vein * crackle;
                // Plates where it has cooled into its freezing range (a glow of about 1.2 at
                // the liquidus of lava, 0.85 at the solidus); the seams between them are
                // hotter than the plates were, and glow brighter and yellower.
                // (Not where it is mixed with another liquid: there a cell's temperature jumps
                // with the share of each from frame to frame, and the plates and seams
                // flickered; the quenched skin is drawn there as frozen, see `iced`.)
                float crust = crackle * cgf * (1.0 - smoothstep(0.7, 1.22, glow_lvl)) * smoothstep(0.85, 0.97, os);
                if (crust > 0.0) {
                    float plate = flow_seams(world, uflow * 0.8, 0.34, 6.0, 0.22, zoom);
                    plate = min(plate, mix(1.0, flow_seams(world + 13.1, uflow * 0.8, 1.0, 6.0, 0.14, zoom), 0.6 * detail(1.0, zoom)));
                    float seam = 1.0 - plate;
                    vec3 rock = ob * (0.7 + 0.6 * mix(0.5, gnoise(world * 1.6), detail(1.6, zoom)));
                    fc = mix(fc, rock, crust * plate);
                    glow *= mix(1.0, mix(0.1, 1.0, seam) * (1.0 + 0.7 * seam), crust);
                    glow.g *= 1.0 + 0.25 * seam * crust;
                }
            }
            float hot = clamp(max(glow.r, max(glow.g, glow.b)), 0.0, 1.0);
            fc = fc * (1.0 - 0.85 * hot) + glow * (1.0 - 0.3 * thin);
            // Lit by hot liquid nearby.
            fc += 0.5 * glf.rgb * (0.25 + fc) * (1.0 - hot);

            // The depth of the tank: a touch darker towards the floor.
            fc *= mix(0.88, 1.04, h);
            col = mix(col, fc, alpha);
        }
    }
    // Liquid lights the air around it a little, and hot liquid throws a glow.
    {
        float lit_air = clamp(T, 0.0, 0.6) * clamp(1.0 - W / pc.threshold, 0.0, 1.0);
        vec3 tint = mix(base, ob, smoothstep(0.2, 0.6, oil_w));
        col += tint * 0.16 * lit_air * smoothstep(0.0, 0.05, W);
        // (The cell fields reach a little past the surface.)
        col += 0.35 * em.rgb * min(sty.w, 1.0) * clamp(1.0 - W / pc.threshold, 0.0, 1.0);
    }

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
        // time (water in a moment, oil slowly, syrup hardly at all). The map is read just
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
        sc = mix(sc, mix(sc * 0.7, tint, 0.6) * 0.92, wet * film_band);
        // What has stained it: on its face along the edge (about as deep as the wet film,
        // with a gently wavering inner edge), and in runs down a wall. (It was a band 2.4
        // times the film deep with a ragged edge of fine noise: speckle reaching well into
        // every floor and peg.)
        if (pc.stain_k > 0.0) {
            vec4 st = stain_look(stain_at(against, world.x, zoom), world, zoom);
            float sw = 1.3 * film_w;
            float sband = 1.0 - smoothstep(0.4 * sw, sw, -sd + 0.5 * sw * (gnoise(world * 0.3) - 0.5) * detail(0.3, zoom));
            sc = mix(sc, st.rgb, 0.95 * st.w * sband * pc.stain_k);
        }
        // A wet edge catches the light.
        sc += 0.20 * wet * (1.0 - clamp((-sd - 0.35 * bevel_w) * zoom + 0.5, 0.0, 1.0)) * smoothstep(0.1, 0.8, facing);
        // Lit by hot liquid nearby, most along the edge facing it.
        if (pc.glow_k > 0.0) sc += 0.8 * pc.glow_k * glow_at(against).rgb * (sc * 2.5 + 0.03) * mix(0.25 * exp(min(sd, 0.0) / 3.0), 1.0, bevel);
        col = mix(col, sc, inside);
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
