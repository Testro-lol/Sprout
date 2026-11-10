#version 460

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coords;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_tex_coords;

layout(push_constant) uniform PC {
    vec2 screen_size;  // in egui points
} pc;

void main() {
    gl_Position = vec4(2.0 * position / pc.screen_size - 1.0, 0.0, 1.0);
    v_color = color;
    v_tex_coords = tex_coords;
}
