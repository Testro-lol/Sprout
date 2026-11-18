#version 460
// The look field (look.vert): sheen, carried scalar, grain and weight, times the kernel.
layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in vec4 v_look;
layout(location = 0) out vec4 f0;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 >= 1.0) discard;
    float w = (1.0 - r2) * (1.0 - r2);
    f0 = v_look * w;
}
