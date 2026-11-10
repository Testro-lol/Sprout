// Workgroup reduction into the per-workgroup partial buffer (256 invocations per group).

// Workgroup sum of a vec4 into part[slot]; every invocation of the group must call it.
shared vec4 red_buf[256];
void group_sum(vec4 v, uint slot) {
    uint t = gl_LocalInvocationIndex;
    red_buf[t] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) red_buf[t] += red_buf[t + s];
        barrier();
    }
    if (t == 0u) part[slot] = red_buf[0];
}
uint group_slot() { return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x; }
