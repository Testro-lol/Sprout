// The scalar step of the conjugate-gradient solves: sum the per-workgroup partials and
// advance the solve's scalars. Used by cg_reduce.comp (the closing statistics) and, through
// finish_group, by the kernels that produce the partials. Modes:
//   mode 0  pressure start   rz, rr from part[0..n), bb from part[n..2n).xy
//   mode 1  pressure alpha   pq from part[n..2n).zw
//   mode 2  pressure beta    rz, rr from part[0..n)
//   mode 3  pressure end     statistics
//   mode 4  viscosity start  rz, rr from part[0..n).xy (the residual of the initial guess)
//   mode 5  viscosity alpha  pq from part[n..2n).z
//   mode 6  viscosity beta   rz, rr from part[0..n).xy
//   mode 7  viscosity end    statistics
//   mode 8  pressure start of the second solve, after the viscosity (rhs_visc.comp): as
//           mode 0, but the density-correction potential counts as solved already
//   mode 9  pressure end of the second solve    statistics (its own iteration count)
//   mode 10 viscosity after the warm start (visc_iter.comp, WARM): rz, rr from part[0..n).xy
// With the multigrid preconditioner (`viscosity_precond` 1) z comes from the V-cycle after the
// update (vmg.glsl), which also forms A z, so both step lengths come from one reduction
// (p.q = z.Az + 2 beta z.q + beta^2 p_old.q_old, A symmetric):
//   mode 11 viscosity start, preconditioned (visc_mgz.comp): r.z, z.Az from part[0..n).xy
//   mode 12 viscosity update (visc_mgu.comp): r.r, p.q from part[0..n).yz; convergence
//   mode 13 viscosity iteration (visc_mgz.comp): r.z, z.Az, z.q from part[0..n).xyz
// When a solve has converged, its entries in the indirect argument buffer are zeroed so
// the iterations recorded after that point do no work.

shared vec4 sbuf[256];
shared uint s_live;

vec4 total(uint base, uint n) {
    uint t = gl_LocalInvocationIndex;
    vec4 s = vec4(0.0);
    for (uint k = t; k < n; k += 256u) s += part[base + k];
    sbuf[t] = s;
    barrier();
    for (uint h = 128u; h > 0u; h >>= 1) {
        if (t < h) sbuf[t] += sbuf[t + h];
        barrier();
    }
    vec4 v = sbuf[0];
    barrier();
    return v;
}

// Called by thread 0 only after it updated K; spreads the decision through shared memory.
void publish(bool on, uint n_entries, uint first_entry, uint last_entry) {
    if (gl_LocalInvocationIndex == 0u) s_live = on ? 1u : 0u;
    barrier();
    bool live = s_live != 0u;
    uint t = gl_LocalInvocationIndex;
    on = live;
    for (uint e = first_entry + t; e < last_entry; e += 256u) {
        for (uint k = 0u; k < 3u; k++) ind[3u * e + k] = on ? ind[3u * (n_entries + e) + k] : 0u;
    }
}

void publish_pressure(uint n_entries, uint first_entry, uint last_entry) {
    publish(gl_LocalInvocationIndex == 0u ? !(K.p_done.x != 0u && K.p_done.y != 0u) : false, n_entries, first_entry, last_entry);
}

void reduce_scalars(uint mode, uint n, uint n_entries, uint first_entry, uint last_entry) {
    uint t = gl_LocalInvocationIndex;
    if (mode == 0u || mode == 8u) {
        vec4 a = total(0u, n);
        vec4 b = total(n, n);
        bool all_done = false;
        if (t == 0u) {
            K.p_bb = b.xy;
            K.p_rz = a.xy;
            K.p_rr = a.zw;
            K.p_beta = vec2(0.0);
            K.p_alpha = vec2(0.0);
            K.p_iters = uvec2(0u);
            vec2 tol = P.tol.xy;
            bvec2 done = lessThanEqual(a.zw, tol * tol * b.xy);
            if (mode == 8u) done.y = true;
            K.p_done = uvec2(done);
            K.solves += 1u;
        }
        publish_pressure(n_entries, first_entry, last_entry);
    } else if (mode == 1u) {
        vec4 b = total(n, n);
        if (t == 0u) {
            vec2 pq = b.zw;
            vec2 a = vec2(0.0);
            if (K.p_done.x == 0u && pq.x > 0.0) a.x = K.p_rz.x / pq.x;
            if (K.p_done.y == 0u && pq.y > 0.0) a.y = K.p_rz.y / pq.y;
            K.p_alpha = a;
        }
    } else if (mode == 2u) {
        vec4 a = total(0u, n);
        if (t == 0u) {
            uvec2 was = K.p_done;
            K.p_iters += uvec2(1u) - was;
            vec2 tol = P.tol.xy;
            bvec2 now = lessThanEqual(a.zw, tol * tol * K.p_bb);
            uvec2 done = max(was, uvec2(now));
            vec2 beta = vec2(0.0);
            if (done.x == 0u && K.p_rz.x > 0.0) beta.x = a.x / K.p_rz.x;
            if (done.y == 0u && K.p_rz.y > 0.0) beta.y = a.y / K.p_rz.y;
            K.p_beta = beta;
            K.p_rz = a.xy;
            K.p_rr = mix(a.zw, K.p_rr, vec2(was));
            K.p_done = done;
        }
        publish_pressure(n_entries, first_entry, last_entry);
    } else if (mode == 3u || mode == 9u) {
        if (t == 0u) {
            if (mode == 3u) {
                K.p_it_max = max(K.p_it_max, max(K.p_iters.x, K.p_iters.y));
                if (K.p_done.x == 0u) K.p_unconv += 1u;
            } else {
                K.p2_it_max = max(K.p2_it_max, K.p_iters.x);
                if (K.p_done.x == 0u) K.p2_unconv += 1u;
            }
            float rel = K.p_bb.x > 0.0 ? sqrt(K.p_rr.x / K.p_bb.x) : 0.0;
            K.p_res_max = max(K.p_res_max, rel);
        }
    } else if (mode == 4u) {
        vec4 a = total(0u, n);
        if (t == 0u) {
            K.v_rz = a.x;
            K.v_r0 = a.y;
            K.v_rr = a.y;
            K.v_beta = 0.0;
            K.v_alpha = 0.0;
            K.v_iters = 0u;
            K.v_done = (a.y <= 1e-30 || a.x <= 0.0) ? 1u : 0u;
        }
        publish(t == 0u ? K.v_done == 0u : false, n_entries, first_entry, last_entry);
    } else if (mode == 5u) {
        vec4 b = total(n, n);
        if (t == 0u) K.v_alpha = (K.v_done == 0u && b.z > 0.0) ? K.v_rz / b.z : 0.0;
    } else if (mode == 6u) {
        vec4 a = total(0u, n);
        if (t == 0u) {
            uint was = K.v_done;
            K.v_iters += 1u - was;
            bool now = a.y <= P.tol.z * P.tol.z * K.v_r0;
            uint done = max(was, uint(now));
            K.v_beta = (done == 0u && K.v_rz > 0.0) ? a.x / K.v_rz : 0.0;
            K.v_rz = a.x;
            if (was == 0u) K.v_rr = a.y;
            K.v_done = done;
        }
        publish(t == 0u ? K.v_done == 0u : false, n_entries, first_entry, last_entry);
    } else if (mode == 10u) {
        // Viscosity, after the warm start (visc_iter.comp, WARM): rz, rr of its residual.
        vec4 a = total(0u, n);
        if (t == 0u) {
            K.v_rz = a.x;
            K.v_rr = a.y;
            K.v_beta = 0.0;
            bool done = a.y <= P.tol.z * P.tol.z * K.v_r0 || a.x <= 0.0;
            K.v_done = done ? 1u : 0u;
        }
        publish(t == 0u ? K.v_done == 0u : false, n_entries, first_entry, last_entry);
    } else if (mode == 11u) {
        vec4 a = total(0u, n);
        if (t == 0u) {
            K.v_rz = a.x;
            K.v_beta = 0.0;
            K.v_pq = a.y;
            K.v_alpha = a.y > 0.0 ? a.x / a.y : 0.0;
            if (a.x <= 0.0 || a.y <= 0.0) K.v_done = 1u;
        }
        publish(t == 0u ? K.v_done == 0u : false, n_entries, first_entry, last_entry);
    } else if (mode == 12u) {
        vec4 a = total(0u, n);
        if (t == 0u) {
            uint was = K.v_done;
            K.v_iters += 1u - was;
            bool now = a.y <= P.tol.z * P.tol.z * K.v_r0;
            if (was == 0u) K.v_rr = a.y;
            K.v_pq = a.z;
            K.v_done = max(was, uint(now));
        }
        publish(t == 0u ? K.v_done == 0u : false, n_entries, first_entry, last_entry);
    } else if (mode == 13u) {
        vec4 a = total(0u, n);
        if (t == 0u) {
            float beta = (K.v_done == 0u && K.v_rz > 0.0) ? a.x / K.v_rz : 0.0;
            float pq = a.y + 2.0 * beta * a.z + beta * beta * K.v_pq;
            K.v_beta = beta;
            K.v_alpha = (K.v_done == 0u && pq > 0.0) ? a.x / pq : 0.0;
            K.v_rz = a.x;
        }
    } else if (mode == 7u) {
        if (t == 0u) {
            K.v_it_max = max(K.v_it_max, K.v_iters);
            if (K.v_done == 0u) K.v_unconv += 1u;
            float rel = K.v_r0 > 0.0 ? sqrt(K.v_rr / K.v_r0) : 0.0;
            K.v_res_max = max(K.v_res_max, rel);
        }
    }
}

// The indirect entries a solve owns: pressure [0, v), viscosity [v, n), v = P.solve.y (the
// viscosity grid, its reduction and its coarse multigrid levels).
void reduce_entries(uint mode, uint n_entries, out uint first_entry, out uint last_entry) {
    bool pressure = !((mode >= 4u && mode < 8u) || mode >= 10u);
    first_entry = pressure ? 0u : P.solve.y;
    last_entry = pressure ? P.solve.y : n_entries;
}

// Called by every invocation of a 256-invocation work group of a kernel that has just
// written its partial sum (group_sum, reduce.glsl): the last work group of the dispatch to
// get here sums all of them and advances the solve as cg_reduce `mode` would, in the same
// order (the same invocations add the same partials), so the result is the same to the
// bit. That saves the dispatch and barrier of a separate reduction, two per iteration.
shared uint s_last;
void finish_group(uint mode) {
    if (gl_LocalInvocationIndex == 0u) {
        memoryBarrierBuffer();
        uint prev = atomicAdd(K.arrive, 1u);
        s_last = prev + 1u == gl_NumWorkGroups.x * gl_NumWorkGroups.y ? 1u : 0u;
    }
    barrier();
    if (s_last == 0u) return;
    memoryBarrierBuffer();
    uint first_entry, last_entry;
    reduce_entries(mode, P.solve.x, first_entry, last_entry);
    reduce_scalars(mode, gl_NumWorkGroups.x * gl_NumWorkGroups.y, P.solve.x, first_entry, last_entry);
    if (gl_LocalInvocationIndex == 0u) K.arrive = 0u;
}
