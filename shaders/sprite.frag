#version 460
// sprite.vert: bubbles, foam, spray and specks, premultiplied over the shaded liquid.
layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in vec4 v_a;
layout(location = 2) flat in vec3 v_col;
layout(location = 3) flat in vec3 v_t;
layout(location = 4) flat in vec3 v_light;
layout(location = 5) flat in vec2 v_axis;
layout(location = 6) flat in vec4 v_film;
layout(location = 0) out vec4 f_color;

layout(push_constant) uniform PC {
    vec2 scale; vec2 offset; vec2 half_res; ivec2 grid; float zoom; float now; uint mode; uint cap; float cell_size; float sss;
} pc;

const vec2 LIGHT = vec2(-0.45, 0.893);   // composite.frag

vec3 to_srgb(vec3 lin) {
    lin = clamp(lin, 0.0, 1.0);
    return mix(lin * 12.92, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin));
}

void main() {
    float rp = v_a.y;
    vec2 uv = v_uv;
    float kind = v_a.x;
    // The vignette of composite.frag.
    vec2 vg = (gl_FragCoord.xy - pc.half_res) / pc.half_res;
    float vignette = 1.0 - 0.22 * smoothstep(0.5, 1.5, length(vg));
    float tl = dot(v_t, vec3(0.333));
    if (kind > 3.5) {
        float dist;
        if (kind < 4.5) {
            // A flake: a hexagon foreshortened along its normal.
            vec2 a = vec2(dot(uv, v_axis), dot(uv, vec2(-v_axis.y, v_axis.x)) / v_a.z);
            vec2 h = abs(a);
            dist = max(h.x * 0.866 + h.y * 0.5, h.y);
        } else {
            dist = length(uv);
        }
        float cov = clamp((1.0 - dist) * rp + 0.5, 0.0, 1.0) * v_a.w;
        if (cov <= 0.0) discard;
        // Opaque: it hides what is behind it, as far as the liquid in front lets it show. A
        // speck smaller than its pixel adds the light it sends over that pixel: a flake's
        // flash (many times as bright as white) lights the pixel even when it covers a
        // tenth of it, as real glitter twinkles far below the eye's resolution.
        vec3 lin = v_col * v_t;
        if (kind > 4.5) lin *= 0.85 + 0.25 * smoothstep(0.2, -0.8, dot(uv, vec2(LIGHT.x, -LIGHT.y)));
        f_color = vec4(to_srgb(lin * cov) * vignette, cov * tl);
        return;
    }
    float d = length(uv);
    float cov = clamp((1.0 - d) * rp + 0.5, 0.0, 1.0);
    if (cov <= 0.0) discard;
    vec2 nrm = uv / max(d, 1e-3);
    float lit = dot(nrm, LIGHT);
    vec3 light = to_srgb(v_light);
    vec3 rgb;
    float alpha;
    if (kind < 1.5) {
        // A bubble: the ring beyond 1 / n of its radius totally reflects the light around it
        // (brightest towards the light, whose side it faces); a glint just inside it there;
        // inside, a clear window, a little brighter than the liquid it replaces.
        float small = 1.0 - smoothstep(1.6, 3.0, rp);
        float r_in = v_film.x;
        float aa = 1.5 / max(rp, 1.0);
        float ring = smoothstep(r_in - aa, r_in + aa, d);
        float lit_k = 0.45 + 0.55 * smoothstep(-0.4, 0.8, lit);
        float glint = smoothstep(0.30, 0.05, length(uv - LIGHT * mix(0.42, r_in * 0.9, 0.5))) * (1.0 - small);
        vec3 wall = to_srgb(v_col);
        rgb = (wall * ring * lit_k * 0.85 + vec3(1.0) * glint * 0.9 + light * v_film.y * 0.5 * (1.0 - ring)) * v_t;
        alpha = (0.06 + 0.10 * v_film.y + 0.60 * ring * lit_k + 0.6 * glint) * tl;
        rgb = mix(rgb, wall * 0.40 * v_t, small);
        alpha = mix(alpha, 0.30 * tl, small);
        // Smaller than the pixel and a half it is drawn at, it covers less of it.
        float area = clamp(v_a.w, 0.12, 1.0);
        rgb *= area;
        alpha *= area;
    } else if (kind < 2.5) {
        // Foam: a parcel of small bubbles, soft at its edge so that parcels merge into one
        // mass; its bubbles (a cell pattern in the parcel's own frame, so they travel with
        // it) show where they are a few pixels across, each wall a fine line, bright towards
        // the light and in its films' colours, with a glint in the bubbles on the lit side.
        // Paler and lit towards the top of the parcel, as the top of a head is.
        float soft = 1.0 - smoothstep(0.2, 1.0, d);
        float rb = max(v_a.w, 1.0);                   // bubble radius, px
        vec2 q = uv * rp / (1.6 * rb) + v_a.z * vec2(37.1, 17.3);
        vec2 qi = floor(q), qf = q - qi;
        float d1 = 8.0, d2 = 8.0;
        vec2 near_c = vec2(0.0);
        for (int b = -1; b <= 1; b++) {
            for (int a = -1; a <= 1; a++) {
                vec2 g = vec2(a, b);
                vec2 o = g + 0.5 + 0.38 * sin(6.2831 * fract(sin(dot(qi + g, vec2(127.1, 311.7))) * vec2(43758.5, 22578.1)));
                float dd = length(o - qf);
                if (dd < d1) { d2 = d1; d1 = dd; near_c = o - qf; } else if (dd < d2) d2 = dd;
            }
        }
        float detail = smoothstep(2.0, 4.5, rb);
        float wall = (1.0 - smoothstep(0.0, max(0.08, 1.2 / (1.6 * rb)), d2 - d1)) * detail;
        float glint = smoothstep(0.32, 0.12, length(-near_c - vec2(-0.45, 0.893) * 0.25)) * detail;
        float shade = 0.90 + 0.10 * smoothstep(-0.9, 0.7, uv.y);
        vec3 body = v_col * light;
        // The films' interference colour on the walls and in the glints (white where thick).
        vec3 fc = mix(vec3(1.0), v_film.rgb, v_film.a);
        rgb = body * shade * (1.0 - 0.22 * wall) + wall * 0.18 * (fc - 0.6) * v_film.a + vec3(0.30) * glint * fc;
        alpha = 0.88 * soft;
        rgb *= alpha;
        cov = 1.0;
    } else {
        // Spray: a droplet of its liquid.
        rgb = to_srgb(v_col) * light * (0.85 + 0.25 * smoothstep(-0.6, 0.8, lit));
        alpha = 0.75 * clamp(v_a.w, 0.2, 1.0);
        rgb *= alpha;
    }
    f_color = vec4(rgb * cov * vignette, alpha * cov);
}
