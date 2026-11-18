#version 460
#extension GL_GOOGLE_include_directive : enable
#include "optics.glsl"
// Sprites drawn over the shaded liquid in the final pass (README, "Look", "Bubbles and foam",
// "Particles in the liquid"): the bubbles, foam, spray and specks of spray.comp. Each is a
// quad round its disc; how it looks is worked out here from physics, not set by hand:
//
//   bubble  a ball of gas in the liquid. Light meeting its wall from the liquid side beyond
//           the critical angle, sin = 1 / n, is totally reflected: the ring from 1 / n of its
//           radius outwards mirrors the light in the liquid around it (wider in a liquid of
//           higher index: water 0.25, oil 0.32). Inside it, the view crosses gas instead of
//           liquid: it is a window, as bright as what the liquid would have taken out of it.
//           How much of it shows is how much the liquid in front hides (a bubble deep in milk
//           is gone, one in water shows to the back of the tank).
//   foam    a parcel of bubbles. It is white by scattering at its films, and coloured by
//           what the liquid in them absorbs: its liquid fraction eps drains with age
//           (eps0 / (1 + 8 age)^2, wet to dry foam) and its bubbles coarsen
//           (r sqrt(1 + 3 age)); a layer of it reflects as a medium absorbing eps x the
//           liquid's absorption and scattering ~ sqrt(eps) / (3.5 r) (transport mean free
//           path of foam, Durian et al.): beer foam is cream, cola foam tan, blood foam pink,
//           and all of them whiten as they drain. Its films thin by drainage,
//           h = H0 / sqrt(1 + t / tau), tau from the liquid's viscosity (material.rs): a film
//           of a few hundred nanometres shows interference colours (soap), one of a few tens
//           goes black before it bursts; a short-lived foam bursts while its films are thick
//           and white. Only bubbles a few pixels across show their walls' colours.
//   spray   a droplet of its liquid, as a thin film of it lit from the front.
//   speck   a coarse particle of the liquid: a grain (a disc) or a flake (a hexagon
//           foreshortened along its normal), at its real size. Diffusely it shows its own
//           colour in the light that reaches it; a flake's face is also a mirror of
//           reflectance `dust_reflect`, flashing where it is the half vector to the key light.
//           It lies at a depth in the layer of the liquid the eye sees into, and the liquid
//           in front of it tints and hides it by its absorption and scattering.

struct Diffuse { vec2 pos; vec2 vel; float life; float r; uint info; float z; };

layout(std430, set = 0, binding = 3) readonly buffer DiffuseBuf { Diffuse dpart[]; };
layout(std430, set = 0, binding = 4) readonly buffer DiffuseCtl { uint dctl[]; };
layout(std430, set = 0, binding = 5) readonly buffer OptBuf { Optics optics[]; };
layout(set = 0, binding = 6) uniform sampler2DArray cellopt;

layout(push_constant) uniform PC {
    vec2 scale;      // world (cells) -> clip
    vec2 offset;
    vec2 half_res;   // half the target, pixels
    ivec2 grid;
    float zoom;      // pixels per cell
    float now;
    uint mode;       // 0 (the only one)
    uint cap;
    float cell_size; // m
    float sss;       // 1: the light inside the liquid is sss.comp's (layer 20), else layer 9
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out vec4 v_a;   // kind (1 bubble, 2 foam, 3 spray, 4 flake, 5 grain), radius (px), seed or minor axis, bubble radius (px) or coverage
layout(location = 2) flat out vec3 v_col; // bubble: the light its ring mirrors; foam: its colour; spray, speck: its colour as lit
layout(location = 3) flat out vec3 v_t;   // how much of it shows through the liquid in front
layout(location = 4) flat out vec3 v_light;
layout(location = 5) flat out vec2 v_axis; // flake: its major axis on screen
layout(location = 6) flat out vec4 v_film; // bubble: inner radius of its ring (1 / n), window; foam: its films' colour, how much they show

const vec2 QUAD[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
const vec3 L3 = vec3(-0.3375, 0.67, 0.66);   // composite.frag
const vec3 H = normalize(L3 + vec3(0.0, 0.0, 1.0));
const float LV_BODY = 16.0;                  // composite.frag: thickness of a body along the view, cells
const float FOAM_PARCEL = 0.55;              // radius of a parcel of foam, cells (spray.comp: FOAM_CROWD)
const float FOAM_EPS0 = 0.25;                // liquid fraction of freshly made (wet) foam
const float FILM_H0 = 2.0e-6;                // thickness of a fresh film between bubbles, m
const float SEEN_DEPTH = 1.5;                // specks are drawn in the first optical depths of the liquid

uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float rnd(uint h) { return float(h >> 8) / 16777216.0; }

vec3 unpack_n(uint b) {
    vec2 q = vec2(float(b & 1023u), float((b >> 10) & 1023u)) / 1023.0 * 2.0 - 1.0;
    vec2 p = vec2(q.x + q.y, q.x - q.y) * 0.5;
    return normalize(vec3(p, max(1.0 - abs(p.x) - abs(p.y), 0.0)));
}

// The liquid at x: its coefficients (absorption rgb, scattering; the specks' share included),
// its liquid weight, refractive index, the light inside it (linear).
void liquid(vec2 x, out vec4 k, out float w, out float ior, out vec3 light) {
    vec2 uv = x / vec2(pc.grid);
    w = textureLod(cellopt, vec3(uv, 4.0), 0.0).w;
    float iw = 1.0 / max(w, 1e-3);
    k = (textureLod(cellopt, vec3(uv, 0.0), 0.0) + textureLod(cellopt, vec3(uv, 1.0), 0.0) + textureLod(cellopt, vec3(uv, 11.0), 0.0)
       + textureLod(cellopt, vec3(uv, 18.0), 0.0)) * iw;
    ior = w > 0.02 ? clamp(textureLod(cellopt, vec3(uv, 17.0), 0.0).a * iw, 1.0, 2.5) : 1.33;
    light = pc.sss > 0.0 ? textureLod(cellopt, vec3(uv, 20.0), 0.0).rgb : textureLod(cellopt, vec3(uv, 9.0), 0.0).rgb * iw;
    light = w > 0.02 ? clamp(light, 0.0, 1.0) : vec3(1.0);
}

// Reflectance of a soap film `h` m thick of index n at normal incidence, per channel,
// averaged over a little of the spectrum and of the film's spread in thickness (two
// surfaces of reflectance R1, a phase flip at one: R = 4 R1 sin^2(2 pi n h / lambda)).
vec3 film_reflect(float h, float n) {
    float r1 = pow((n - 1.0) / (n + 1.0), 2.0);
    vec3 acc = vec3(0.0);
    for (int a = -1; a <= 1; a++) {
        for (int b = -1; b <= 1; b++) {
            vec3 lam = vec3(610.0e-9, 550.0e-9, 465.0e-9) + 22.0e-9 * float(a);
            vec3 sn = sin(6.2831853 * n * h * (1.0 + 0.12 * float(b)) / lam);
            acc += sn * sn;
        }
    }
    return 4.0 * r1 * acc / 9.0;
}

void main() {
    uint e = dctl[8u + uint(gl_VertexIndex) / 6u];
    vec2 q = QUAD[gl_VertexIndex % 6];
    Diffuse d = dpart[e];
    uint st = d.info & 3u;
    bool speck = (d.info & 8u) != 0u;
    int row = int((d.info >> 8) & 127u) - 1;
    Optics o = optics[max(row, 0)];
    vec2 pos = d.pos;
    float rp;
    v_axis = vec2(1.0, 0.0);
    v_film = vec4(0.0);
    vec4 kq;
    float wq, ior;
    vec3 light;
    liquid(pos, kq, wq, ior, light);
    // Its own liquid's coefficients (as it came out of it; the cells' where it lies in some).
    vec4 krow = row >= 0 ? o.absorb : vec4(0.02, 0.01, 0.005, 0.0);
    vec4 kown = wq > 0.05 ? kq : krow;
    v_light = light;
    if (speck) {
        float ext = dot(kq.rgb, vec3(0.3, 0.45, 0.25)) + kq.a;
        bool wet = wq > 0.05;
        // Where in the depth it lies: through the layer of liquid the eye sees into.
        float depth = wet ? d.z * min(LV_BODY, SEEN_DEPTH / max(ext, 1e-3)) : 0.0;
        v_t = exp(-(kq.rgb + kq.a) * depth);
        // (The light at that depth: the liquid's light falls off below its face as it does
        // with depth into it.)
        vec3 lf = wet ? mix(vec3(1.0), light, clamp(depth / 3.0, 0.0, 1.0)) * exp(-attenuation(kq) * depth * 0.5) : vec3(1.0);
        float rs = d.r * pc.zoom;                       // its radius, px
        rp = max(rs, 1.0);
        float aspect = o.dust2.z, r0 = o.dust2.w;
        vec3 dc = o.dcol.rgb;
        vec3 n = unpack_n(floatBitsToUint(d.life));
        if (aspect < 0.6667) {
            // A flake: diffuse by how it faces the light, a mirror of the studio by the
            // direction its normal sends the view (a dark floor, a graded sky), and a flash
            // where it is the half vector to the key light (a soft box, so a share of the
            // flakes catch it at any moment).
            vec3 R = 2.0 * n.z * n - vec3(0.0, 0.0, 1.0);
            float sky = smoothstep(-0.5, 0.9, R.y);
            float env = mix(0.10, 0.85, sky) * mix(0.75, 1.0, max(R.z, 0.0)) + 0.45 * smoothstep(0.6, 0.95, dot(R, normalize(L3)));
            float nh = abs(dot(n, H));
            float flash = pow(nh, 160.0) * 4.0 + pow(nh, 30.0) * 0.35;
            vec3 spec_tint = mix(vec3(1.0), dc, 0.6);
            vec3 diffuse = dc * (0.35 + 0.65 * abs(dot(n, normalize(L3))));
            v_col = lf * ((1.0 - r0) * diffuse * 0.9 + r0 * spec_tint * (env + flash * 1.5));
            vec2 nxy = n.xy;
            float l = length(nxy);
            v_axis = l > 1e-4 ? vec2(-nxy.y, nxy.x) / l : vec2(1.0, 0.0);
            // (Its narrow side never under a pixel: a sliver thinner than that would wink.)
            v_a = vec4(4.0, rp, max(abs(n.z), 1.0 / rp), clamp(rs * rs * max(abs(n.z), aspect) / (rp * rp * max(abs(n.z), 1.0 / rp)), 0.05, 1.0));
        } else {
            // A grain: its colour lit from the upper left, a small highlight of its mirror part.
            v_col = lf * (dc * 0.85 + vec3(r0) * 0.6);
            v_a = vec4(5.0, rp, 0.0, clamp(rs * rs / (rp * rp), 0.05, 1.0));
        }
    } else if (st == 2u) {
        // Foam: a parcel FOAM_PARCEL cells across, drawn with its own bubbles.
        rp = max(FOAM_PARCEL * pc.zoom, 2.0);
        float life0 = max(d.z, 1e-3);
        float age = clamp(1.0 - d.life / life0, 0.0, 1.0);
        float t = max(life0 - d.life, 0.0);              // s since it formed
        float eps = FOAM_EPS0 / pow(1.0 + 8.0 * age, 2.0);
        float rb = d.r * sqrt(1.0 + 3.0 * age);          // its bubbles, cells
        float sf = sqrt(eps) / (3.5 * max(rb, 1e-3));
        vec3 sc = vec3(sf) + eps * krow.a * (row >= 0 ? o.scat.rgb : vec3(1.0));
        vec3 R = deep_reflectance3(eps * krow.rgb, sc);
        v_col = to_srgb(R * 1.05);
        // Its films: drained from FILM_H0, at the index of its liquid.
        float tau = row >= 0 ? max(o.fx2.w, 1e-4) : 0.02;
        float hf = FILM_H0 / sqrt(1.0 + t / tau);
        vec3 fr = film_reflect(hf, row >= 0 ? o.opt2.x : 1.33);
        float mean = max(dot(fr, vec3(0.333)), 1e-4);
        // (How much they show: a wall a few pixels long, and a film thin enough to colour.)
        float px = rb * pc.zoom;
        float show = smoothstep(2.0, 6.0, px) * (1.0 - smoothstep(1.0e-6, 1.6e-6, hf));
        v_film = vec4(clamp(fr / mean, 0.0, 2.5), show * clamp(mean / 0.08, 0.15, 1.0));
        v_t = vec3(1.0);   // (foam lies at the front)
        v_a = vec4(2.0, rp, rnd(d.info), rb * pc.zoom);
    } else if (st == 1u) {
        // A bubble at its depth in the body.
        rp = max(d.r * pc.zoom, 1.4);
        float depth = d.z * LV_BODY;
        v_t = exp(-(kq.rgb + kq.a) * depth);
        // The light in the liquid around it, which its ring mirrors (tinted by the liquid
        // between it and the wall it came from), and its window: the liquid it replaces.
        vec3 tint = film_colour(kown, 0.6);
        v_col = light * mix(vec3(1.0), to_linear(tint), 0.35);
        vec3 window = 1.0 - exp(-(kown.rgb + kown.a) * 2.0 * d.r);
        v_film = vec4(1.0 / max(ior, 1.0), dot(window, vec3(0.333)), 0.0, 0.0);
        v_a = vec4(1.0, rp, rnd(d.info), pow(d.r * pc.zoom / rp, 2.0));
    } else {
        // Spray: a droplet of its liquid.
        rp = max(d.r * pc.zoom, 1.0);
        v_t = vec3(1.0);
        v_col = to_linear(film_colour(krow, max(d.r, 0.05) * 2.0));
        v_a = vec4(3.0, rp, rnd(d.info), pow(d.r * pc.zoom / rp, 2.0));
    }
    float half_c = (rp + 1.0) / pc.zoom;
    gl_Position = vec4((pos + q * half_c) * pc.scale + pc.offset, 0.0, 1.0);
    v_uv = q * (rp + 1.0) / rp;
}
