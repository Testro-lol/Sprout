// Shared by the render shaders: the optics table (render.rs: GpuOptics) and colour helpers.

struct Optics {
    vec4 lab;        // body colour: OKLab L, a, b, chroma
    vec4 lab_solid;  // the same when frozen
    vec4 lab_mid;    // colour at half concentration against the diluent (w < 0: none)
    vec4 opt;        // opacity, scattering, metal, gloss
    vec4 sty;        // body, foam, drying time (s), emissivity
    ivec4 idx;       // diluent row (-1 none), phase group
    vec4 fx;         // stain, glint, iridescence, crackle
};

// Colour of a concentrate at concentration s in its diluent: each weighted by how opaque
// it is (a pigment colours clear water far more than its share), or, if the material gives
// a mid-blend colour, a quadratic Bezier through it.
vec4 blend_colour(Optics d, Optics o, vec4 dl, vec4 ol, float s) {
    if (o.lab_mid.w >= 0.0) return (1.0 - s) * (1.0 - s) * dl + 2.0 * s * (1.0 - s) * o.lab_mid + s * s * ol;
    float wd = (1.0 - s) * (d.opt.x + 0.15), wc = s * (o.opt.x + 0.15);
    return (wd * dl + wc * ol) / max(wd + wc, 1e-6);
}

// OKLab to sRGB (gamma), Ottosson 2020.
vec3 oklab_to_srgb(vec3 c) {
    float l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    float m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    float s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
    l = l * l * l;
    m = m * m * m;
    s = s * s * s;
    vec3 lin = vec3(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
    lin = clamp(lin, 0.0, 1.0);
    return mix(lin * 12.92, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin));
}

// An accumulated OKLab colour (L, a, b, chroma, all times the weight) to sRGB: the a-b
// direction of the average with the average chroma, so that a blend keeps the
// saturation of what was blended (blue and amber meet in violet and rose, not grey).
vec3 resolve_colour(vec4 acc) {
    vec2 ab = acc.yz;
    float n = length(ab);
    vec2 abc = n > 1e-5 ? ab / n * acc.w : ab;
    return oklab_to_srgb(vec3(acc.x, abc));
}
