#version 460

// Box-filters the accumulation target down by `factor` in each direction: the first step of
// the wide (blurred) copy the composite pass shades depth, thin liquid and flow from.

layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 src_texel;  // 1 / size of the source, texels
    vec2 src_max;    // centre of the last used texel of the source
    float factor;    // source texels per pixel of this pass
    int taps;        // bilinear taps per axis; each covers two source texels
} pc;

layout(location = 0) out vec4 f_color;

void main() {
    vec2 c = gl_FragCoord.xy * pc.factor;
    vec4 sum = vec4(0.0);
    for (int j = 0; j < pc.taps; j++) {
        for (int i = 0; i < pc.taps; i++) {
            vec2 o = ((vec2(i, j) + 0.5) / float(pc.taps) - 0.5) * pc.factor;
            sum += textureLod(src, min(c + o, pc.src_max) * pc.src_texel, 0.0);
        }
    }
    f_color = sum / float(pc.taps * pc.taps);
}
