#version 460

// Resolves the splat accumulation into a liquid surface and draws solids and the
// backdrop. (The brush cursor is an egui overlay, see ui.rs.)

layout(set = 0, binding = 0) uniform sampler2D accum;
layout(std430, set = 0, binding = 1) readonly buffer SdfBuf { float sdf[]; };

layout(push_constant) uniform PC {
    vec2 cam;         // world position at the centre of the view, cells
    vec2 half_res;    // half the target size, pixels
    ivec2 grid;
    float inv_zoom;   // cells per pixel
    float threshold;  // accumulated weight that counts as liquid
} pc;

layout(location = 0) out vec4 f_color;

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

void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 world = pc.cam + (px - pc.half_res) * vec2(pc.inv_zoom, -pc.inv_zoom);
    float aa = pc.inv_zoom;

    // Backdrop: a quiet vertical gradient with a faint 16-cell grid for scale.
    float h = clamp(world.y / float(pc.grid.y), 0.0, 1.0);
    vec3 col = mix(vec3(0.075, 0.085, 0.11), vec3(0.11, 0.125, 0.16), h);
    vec2 gl = abs(fract(world / 16.0 + 0.5) - 0.5) * 16.0;
    col += 0.018 * (1.0 - smoothstep(0.0, 1.5 * aa, min(gl.x, gl.y)));

    // Liquid.
    vec4 a = texture(accum, px / (2.0 * pc.half_res));
    if (a.a > 1e-3) {
        vec3 fc = a.rgb / a.a;
        float t = pc.threshold;
        float alpha = smoothstep(t * 0.75, t * 1.25, a.a);
        float rim = 1.0 - smoothstep(t, t * 3.0, a.a);
        fc = mix(fc, fc * 1.18 + 0.06, rim * 0.75);
        col = mix(col, fc, alpha);
    }

    // Solids.
    float d = sample_sdf(world);
    float inside = 1.0 - smoothstep(-aa, aa, d);
    vec3 sc = mix(vec3(0.30, 0.33, 0.39), vec3(0.62, 0.66, 0.72), 1.0 - smoothstep(-1.4, -0.6, d));
    sc = mix(sc, vec3(0.36, 0.39, 0.45), smoothstep(-0.6, -1.6, d) * 0.0);
    col = mix(col, sc, inside);

    f_color = vec4(col, 1.0);
}
