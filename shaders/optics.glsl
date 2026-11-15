// Shared by the render shaders: the optics table (material.rs: GpuOptics) and the colour of
// liquid from how it absorbs and scatters light (README, "Look").

struct Optics {
    vec4 absorb;     // absorption coefficient (linear RGB) and scattering coefficient, 1 / cell
    vec4 solid;      // colour when frozen (linear RGB); tint depth (cells)
    vec4 opt;        // opacity, scattering, metal, gloss
    vec4 sty;        // body, foam, drying time (s), emissivity
    ivec4 idx;       // diluent row (-1 none), phase group
    vec4 fx;         // stain, glint, iridescence, crackle
};

// Linear to sRGB (gamma) and back.
vec3 to_srgb(vec3 lin) {
    lin = clamp(lin, 0.0, 1.0);
    return mix(lin * 12.92, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin));
}
vec3 to_linear(vec3 c) {
    return mix(c / 12.92, pow((max(c, 0.0) + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

// Diffuse reflectance of a deep layer of a medium that absorbs `a` and scatters `s` (per
// channel; two-stream / Hapke: albedo w, q = sqrt(1 - w), R = (1 - q) / (1 + q)).
vec3 deep_reflectance(vec3 a, float s) {
    vec3 w = s / max(a + s, vec3(1e-6));
    vec3 q = sqrt(max(1.0 - w, 0.0));
    return (1.0 - q) / (1.0 + q);
}

// How fast light that has entered a liquid of coefficients `k` (absorption rgb, scattering)
// falls off with depth, per cell: in a clear liquid by its absorption a; in a strongly
// scattering one it diffuses, at sqrt(3 a s) (the diffusion rate sqrt(3 a (a + s)) for s >> a);
// sqrt(a (a + 3 s)) has both limits.
vec3 attenuation(vec4 k) {
    return sqrt(max(k.rgb * (k.rgb + 3.0 * k.a), 0.0));
}

// Light that has come `depth` cells into a liquid of coefficients `k`, as a share of what fell
// on it.
vec3 light_in(vec4 k, float depth) {
    return exp(-attenuation(k) * depth);
}

// What a layer of liquid `thick` cells along the view looks like, linear RGB, lit by `lit`
// (the light that reached it, linear) from the front and `glow_in` (light coming through it):
// the light its scattering sends back, plus light that came through it, tinted by the way.
// `back` (linear) is what lies behind it; it shows through in proportion to the transmission.
vec3 layer_colour(vec4 k, float thick, vec3 lit, vec3 back) {
    vec3 ext = k.rgb + k.a;
    vec3 tv = exp(-ext * thick);
    vec3 refl = deep_reflectance(k.rgb, k.a) * (1.0 - tv);
    return lit * refl + back * tv;
}

// The colour a film of `film` cells of liquid lays over a pale surface (wet solids, stains),
// sRGB: a scattering liquid its own colour, a clear one the tint of light through the film
// twice (down and back).
vec3 film_colour(vec4 k, float film) {
    vec3 ext = k.rgb + k.a;
    vec3 tv = exp(-2.0 * ext * film);
    vec3 refl = deep_reflectance(k.rgb, k.a) * (1.0 - tv);
    return to_srgb(refl + 0.75 * tv * exp(-2.0 * k.rgb * film));
}
