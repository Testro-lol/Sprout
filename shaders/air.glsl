// Shared by particles.comp and smoke.comp (include after common.glsl): sampling the grid's
// fields at a point, and the air that escaped steam rides (README, "Steam").

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float sample_sdf(vec2 x, out vec2 grad) {
    vec2 g = clamp(x, vec2(0.0), vec2(P.grid) - 0.001);
    ivec2 i = ivec2(g);
    vec2 f = g - vec2(i);
    int w1 = P.grid.x + 1;
    int k = i.y * w1 + i.x;
    float d00 = sdf[k], d10 = sdf[k + 1], d01 = sdf[k + w1], d11 = sdf[k + w1 + 1];
    grad = vec2(mix(d10 - d00, d11 - d01, f.y), mix(d01 - d00, d11 - d10, f.x));
    return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

// Velocity the mouse tool drives a particle towards; `d` points from the particle to the
// cursor. Grab (1) carries liquid along with the cursor and pulls it in; push (3) drives it
// radially outwards at the speed in tool_vel.x, with a smooth centre where the direction is
// undefined; a stream's nozzle (4) holds it at the launch velocity.
// Whirl (7) turns it around the cursor: like a rigid body inside 0.4 of the radius, at the
// speed in tool_vel.x outside, with a gentle pull inwards that keeps the liquid in the
// whirl; hold (8) brings it to rest.
vec2 tool_target(vec2 d) {
    if (P.tool_mode == 3) return -d / max(length(d), 0.15 * P.tool_radius) * P.tool_vel.x;
    if (P.tool_mode == 4) return P.tool_vel;
    if (P.tool_mode == 7) return vec2(d.y, -d.x) / max(length(d), 0.4 * P.tool_radius) * P.tool_vel.x + d * 2.0;
    if (P.tool_mode == 8) return vec2(0.0);
    return P.tool_vel + d * 4.0;
}

// Grid velocity at x, bilinear over the live faces around it; components with no live face
// nearby (spray in free flight) keep `fallback`.
vec2 sample_vel(vec2 x, vec2 fallback) {
    vec2 o = fallback;
    {
        vec2 gp = x - vec2(0.0, 0.5);
        ivec2 base = ivec2(floor(gp));
        vec2 f = gp - vec2(base);
        float sw = 0.0, su = 0.0;
        for (int b = 0; b < 2; b++) {
            for (int a = 0; a < 2; a++) {
                vec4 q = vel[cidx(base + ivec2(a, b))];
                if (face_live(q.z)) {
                    float wt = (a == 0 ? 1.0 - f.x : f.x) * (b == 0 ? 1.0 - f.y : f.y);
                    sw += wt;
                    su += wt * q.x;
                }
            }
        }
        if (sw > 1e-3) o.x = su / sw;
    }
    {
        vec2 gp = x - vec2(0.5, 0.0);
        ivec2 base = ivec2(floor(gp));
        vec2 f = gp - vec2(base);
        float sw = 0.0, su = 0.0;
        for (int b = 0; b < 2; b++) {
            for (int a = 0; a < 2; a++) {
                vec4 q = vel[cidx(base + ivec2(a, b))];
                if (face_live(q.w)) {
                    float wt = (a == 0 ? 1.0 - f.x : f.x) * (b == 0 ? 1.0 - f.y : f.y);
                    sw += wt;
                    su += wt * q.y;
                }
            }
        }
        if (sw > 1e-3) o.y = su / sw;
    }
    return o;
}


// Liquid occupancy at x, bilinear over the cell centres.
float sample_occ(vec2 x) {
    vec2 gc = clamp(x - 0.5, vec2(0.0), vec2(P.grid) - 1.001);
    ivec2 c0 = ivec2(gc);
    vec2 fc = gc - vec2(c0);
    return mix(mix(cell[cidx(c0)].x, cell[cidx(c0 + ivec2(1, 0))].x, fc.x),
               mix(cell[cidx(c0 + ivec2(0, 1))].x, cell[cidx(c0 + ivec2(1, 1))].x, fc.x), fc.y);
}

// ---- the air above the liquid ---------------------------------------------------------
// Air is not simulated: escaped steam rides a procedural flow, a buoyant rise (P.steam.y)
// plus eddies from a stream function of drifting noise (P.steam2.z: their speed, P.steam2.w:
// their size, cells). A stream function's curl is divergence-free, so the eddies stretch the
// steam into wisps and curl it without bunching it into spots; ramped to zero towards solids
// (Bridson et al. 2007, "Curl-noise for procedural fluid flow"), it flows along walls and
// around obstacles, never into them. The rise loses its component into a solid and turns
// along it, so a plume under a shelf spreads along it and spills around the edge. Near
// liquid the air moves with it (a splash or a sinking body drags the steam), and the mouse
// stirs, pushes and blows it as it does the liquid.

const float PUFF_FOLLOW = 0.08;  // s: how quickly a puff takes up the air's velocity
const float PUFF_EMIT = 0.3;     // s: time constant of the vapour a puff lays down
const float PUFF_VAPOUR = 40.0;  // vapour a puff lays down in all (smoke field units)

// Value noise (-0.5 - 0.5) and its gradient: (n, dn/dx, dn/dy, dn/dz).
vec4 vnoise3(vec3 x) {
    vec3 i = floor(x);
    vec3 f = x - i;
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec3 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    uvec3 c = uvec3(ivec3(i) + 4096);
    float r[8];
    for (int k = 0; k < 8; k++) {
        uvec3 q = c + uvec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        r[k] = float(hash(q.x * 73856093u ^ q.y * 19349663u ^ q.z * 83492791u) >> 8) / 16777216.0;
    }
    float k0 = r[0], k1 = r[1] - r[0], k2 = r[2] - r[0], k3 = r[4] - r[0];
    float k4 = r[0] - r[1] - r[2] + r[3], k5 = r[0] - r[2] - r[4] + r[6], k6 = r[0] - r[1] - r[4] + r[5];
    float k7 = -r[0] + r[1] + r[2] - r[3] + r[4] - r[5] - r[6] + r[7];
    float n = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    return vec4(n - 0.5,
                du.x * (k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z),
                du.y * (k2 + k4 * u.x + k5 * u.z + k7 * u.x * u.z),
                du.z * (k3 + k5 * u.y + k6 * u.x + k7 * u.x * u.y));
}

// Stream function of the eddies, cells^2 / s, and its gradient: two octaves that evolve, and
// drift up with the plume so that eddies are carried along rather than standing still.
vec3 air_psi(vec2 x, float t) {
    float L = max(P.steam2.w, 1.0);
    float turn = 0.7 * t * P.steam2.z / L;
    vec2 a = (x - vec2(0.0, 0.5 * P.steam.y * t)) / L;
    vec4 n1 = vnoise3(vec3(a, turn));
    vec4 n2 = vnoise3(vec3(a * 2.17 + 17.3, 2.2 * turn));
    float amp = 2.2 * P.steam2.z * L;
    return amp * vec3(n1.x + 0.45 * n2.x, (n1.yz + 0.45 * 2.17 * n2.yz) / L);
}

// Each puff leans a little to one side as it rises (-0.15 - 0.15).
float puff_lean(uint pid) {
    return (float(hash(pid ^ 0x5bd1e995u) >> 8) / 16777216.0 - 0.5) * 0.3;
}

// Velocity of the air at x, cells / s; `occ`: the liquid occupancy there.
vec2 air_velocity(vec2 x, float lean, out float occ) {
    float t = S.advanced;
    vec2 n;
    float d = sample_sdf(x, n);
    float nl = length(n);
    n = nl > 1e-5 ? n / nl : vec2(0.0, 1.0);
    float Lb = max(P.steam2.w, 1.0);
    float r = clamp(d / Lb, 0.0, 1.0);
    float ramp = r * (1.875 - r * r * (1.25 - 0.375 * r * r));
    float dramp = d < Lb && d > 0.0 ? (1.875 - r * r * (3.75 - 1.875 * r * r)) / Lb : 0.0;
    vec2 v = vec2(0.0);
    if (P.steam2.z > 0.0) {
        vec3 psi = air_psi(x, t);
        v = ramp * vec2(psi.z, -psi.y) + psi.x * dramp * vec2(n.y, -n.x);
    }
    vec2 rise = vec2(lean, 1.0) * P.steam.y;
    float into = dot(rise, n);
    if (into < 0.0) {
        vec2 tg = vec2(n.y, -n.x);
        float side = dot(rise, tg) + lean * 1e-3 >= 0.0 ? 1.0 : -1.0;
        rise -= (n + side * tg * 0.7) * into * (1.0 - ramp);  // into < 0
    }
    v += rise;
    // Liquid: the air moves with it, and the steam still rises through it.
    occ = sample_occ(x);
    vec2 vl = sample_vel(x, v - vec2(0.0, P.steam.y));
    v = mix(v, vl + vec2(0.0, P.steam.y), smoothstep(0.0, 0.4, occ));
    // The mouse tool (Erase is handled by the callers).
    if (P.tool_mode == 1 || P.tool_mode == 3 || P.tool_mode == 4 || P.tool_mode == 7) {
        vec2 td = P.tool_pos - x;
        float r2 = dot(td, td) / (P.tool_radius * P.tool_radius);
        if (r2 < 1.0) v = mix(v, tool_target(td), 1.0 - r2);
    }
    return v;
}

// ---- the smoke field (smoke.comp) ------------------------------------------------------
// The vapour of escaped steam, at smoke.comp's SMOKE_RES texels per cell. Puffs lay theirs into `smoke_in`,
// one word per cell in fixed point (bilinear); smoke.comp spreads each cell's over a
// Gaussian of SMOKE_PUFF cells: boiled water expands ~1700 times as it turns to vapour, so
// what a particle releases is a billow a few cells across, not a point.
const float SMOKE_SCALE = 65536.0;
const float SMOKE_PUFF = 1.4;  // cells: standard deviation of the billow
layout(std430, set = 0, binding = 69) buffer SmokeInBuf { uint smoke_in[]; };

void smoke_inject(vec2 x, float amount) {
    if (amount <= 0.0) return;
    vec2 g = clamp(x - 0.5, vec2(0.0), vec2(P.grid) - 1.001);
    ivec2 i = ivec2(g);
    vec2 f = g - vec2(i);
    float q = amount * SMOKE_SCALE;
    atomicAdd(smoke_in[cidx(i)], uint(q * (1.0 - f.x) * (1.0 - f.y) + 0.5));
    atomicAdd(smoke_in[cidx(i + ivec2(1, 0))], uint(q * f.x * (1.0 - f.y) + 0.5));
    atomicAdd(smoke_in[cidx(i + ivec2(0, 1))], uint(q * (1.0 - f.x) * f.y + 0.5));
    atomicAdd(smoke_in[cidx(i + ivec2(1, 1))], uint(q * f.x * f.y + 0.5));
}
