#version 460
// The vapour field (steam.vert): each bubble's weight times the kernel (r), and a thin ring
// at about two thirds of its kernel (b): where bubbles crowd into one pocket, the rings are
// the walls of the froth inside it.
layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in float v_w;
layout(location = 0) out vec4 f;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 >= 1.0) discard;
    float k = (1.0 - r2) * (1.0 - r2);
    float r = sqrt(r2);
    float ring = exp(-0.5 * (r - 0.62) * (r - 0.62) / (0.09 * 0.09));
    f = vec4(v_w * k, 0.0, v_w * ring, 0.0);
}
