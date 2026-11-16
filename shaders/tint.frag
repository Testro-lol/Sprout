#version 460

// The tint field (render.rs; splat.vert): how the particles absorb and scatter, times the
// kernel: the (aqueous: concentrate part's) absorption (rgb) and the scattering; the aqueous
// concentrate share, the aqueous weight and the others' (optics.glsl: particle_absorb).
layout(location = 0) in vec2 v_uv;
layout(location = 2) flat in vec4 v_conc;
layout(location = 3) flat in vec3 v_share;
layout(location = 0) out vec4 f0;
layout(location = 1) out vec4 f1;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 >= 1.0) discard;
    float w = (1.0 - r2) * (1.0 - r2);
    f0 = v_conc * w;
    f1 = vec4(v_share * w, 0.0);
}
