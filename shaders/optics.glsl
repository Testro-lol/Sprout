// Shared by the render shaders: the optics table (material.rs: GpuOptics) and the colour of
// liquid from how it absorbs and scatters light (README, "Look").

struct Optics {
    vec4 absorb;     // absorption coefficient (linear RGB: dissolved colour and particles), mean reduced scattering, 1 / cell
    vec4 solid;      // colour when frozen (linear RGB); tint depth (cells)
    vec4 opt;        // opacity, platelet sheen, metal, gloss
    vec4 sty;        // fullness, foam, drying time (s), emissivity
    ivec4 idx;       // diluent row (-1 none), phase group
    vec4 fx;         // stain, unused, iridescence, crackle
    vec4 fx2;        // froth (s), fizz, specks drawn per cell, drainage time of its foam's films (s)
    vec4 lum;        // its own light (linear RGB), the share of it that needs stirring
    vec4 hot;        // absorption and scattering at the top of its thermochromic range
    vec4 thermo;     // thermochromic range lo, hi (deg C; hi <= lo: none), unused, unused
    vec4 scat;       // reduced scattering in red, green, blue over its mean (sum 3); asymmetry g
    vec4 opt2;       // refractive index, platelets' rotational diffusion (rad^2 / s), their Bretherton constant, unused
    vec4 dust;       // what one drawn speck stands for in the liquid's colour: absorption (rgb), scattering, 1 / cell
    vec4 dust2;      // speck radius (m), density (kg / m^3), thickness over diameter, mirror reflectance
    vec4 dcol;       // speck colour (linear rgb); drawn specks per cell where they lie packed
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

// The same with the scattering per channel (fine particles scatter blue more than red).
vec3 deep_reflectance3(vec3 a, vec3 s) {
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

// How fast a diluent's own tint gives way to its concentrate's: exp(-DILUENT_FADE c), c the
// concentrate share of the aqueous liquid (cell_optics.comp). Water's blue is the look of a
// body of clear water, many times stronger than its real absorption; kept in proportion it
// turned a little honey in it green, as a blue and a yellow filter do.
const float DILUENT_FADE = 24.0;

// The look key of liquid of coefficients `k` (cell_optics.comp layer 12, composite.frag):
// the colour of a thin film of it, sRGB. Two liquids that look alike have close keys.
vec3 look_key(vec4 k) {
    return film_colour(k, 0.35);
}

// The coefficients of the aqueous liquid the particles lay down at a pixel (the tint field,
// splat.vert: `c` the concentrate part's absorption and the scattering, `sw` the concentrate
// share and the weight), `plain` the absorption of the plain and diluent liquid per unit of
// it (from the cells), its tint fading by the concentrate share as in the cells
// (cell_optics.comp).
vec4 particle_absorb(vec4 c, vec2 sw, vec3 plain) {
    float iw = 1.0 / max(sw.y, 1e-4);
    float cs = clamp(sw.x * iw, 0.0, 1.0);
    return vec4(plain * (1.0 - cs) * exp(-DILUENT_FADE * cs) + c.rgb * iw, c.a * iw);
}
