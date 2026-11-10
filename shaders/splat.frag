#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_color;
layout(location = 0) out vec4 f_accum;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 >= 1.0) discard;
    float w = (1.0 - r2) * (1.0 - r2);
    f_accum = vec4(v_color * w, w);  // additive: colour-weighted sum + weight
}
