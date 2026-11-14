#version 460

// One direction of a 13-tap Gaussian over the used corner of the wide target.

layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 texel;   // 1 / size of the target, texels
    vec2 step;    // half a standard deviation along the blur direction, in uv
    vec2 uv_max;  // centre of the last used texel, in uv
} pc;

layout(location = 0) out vec4 f_color;

// exp(-(i / 2)^2 / 2), i = 0..6
const float G[7] = float[7](1.0, 0.8825, 0.6065, 0.3247, 0.1353, 0.0439, 0.0111);

void main() {
    vec2 uv = gl_FragCoord.xy * pc.texel;
    vec2 lo = 0.5 * pc.texel;
    vec4 sum = G[0] * textureLod(src, uv, 0.0);
    float norm = G[0];
    for (int i = 1; i < 7; i++) {
        // Past the edge of the view the liquid is taken to carry on as it is at the edge.
        sum += G[i] * (textureLod(src, clamp(uv + float(i) * pc.step, lo, pc.uv_max), 0.0) + textureLod(src, clamp(uv - float(i) * pc.step, lo, pc.uv_max), 0.0));
        norm += 2.0 * G[i];
    }
    f_color = sum / norm;
}
