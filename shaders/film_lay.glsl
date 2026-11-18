// Laying liquid onto the film map (README, "Films, stains and drips"): a drop that flies past
// the back wall or strikes a solid (particles.comp), a falling drip or a spray droplet that
// lands (spray.comp). The includer declares film_in (binding 72), film_col (73), ftile (74).
//
// film_in, per texel: thickness laid since the last update, FILM_UNITs x FILM_FIX.
// film_col, per cell, 8 words: what was laid there, each weighted by thickness (x FILM_FIX):
//   absorbance and scattering per FILM_UNIT of thickness (x FILM_A_FIX), the thickness, its
//   volatile share (x 256), the row + 1 of the liquid last laid. film.comp spreads them over
//   the cell's texels in proportion to film_in, then clears them.
const float FILM_FIX = 256.0;
const float FILM_A_FIX = 64.0;

// A mark of radius `r` texels (a dot, or a disc for a big drop), `amount` FILM_UNITs thick in
// its middle, of a liquid that absorbs and scatters `k` per FILM_UNIT, `vol` of it volatile.
void film_lay(vec2 x, vec4 k, float vol, int row, float amount, float r) {
    int R = int(P.heat3.z);
    if (R <= 0) return;
    ivec2 dimF = P.grid * R;
    vec2 g = x * float(R);
    int ri = int(ceil(r));
    uint tx = uint(dimF.x + 15) / 16u, nt = tx * (uint(dimF.y + 15) / 16u);
    for (int j = -ri; j <= ri; j++) {
        for (int i = -ri; i <= ri; i++) {
            ivec2 t = ivec2(floor(g)) + ivec2(i, j);
            if (any(lessThan(t, ivec2(0))) || any(greaterThanEqual(t, dimF))) continue;
            vec2 o = vec2(t) + 0.5 - g;
            float w = clamp(1.0 - dot(o, o) / max(r * r, 0.36), 0.0, 1.0);
            if (w <= 0.0) continue;
            uint q = uint(amount * w * FILM_FIX + 0.5);
            if (q == 0u) continue;
            atomicAdd(film_in[t.y * dimF.x + t.x], q);
            uint ci = uint(cidx(t / R)) * 8u;
            atomicAdd(film_col[ci], uint(min(k.r, 4000.0) * FILM_A_FIX * float(q) / FILM_FIX + 0.5));
            atomicAdd(film_col[ci + 1u], uint(min(k.g, 4000.0) * FILM_A_FIX * float(q) / FILM_FIX + 0.5));
            atomicAdd(film_col[ci + 2u], uint(min(k.b, 4000.0) * FILM_A_FIX * float(q) / FILM_FIX + 0.5));
            atomicAdd(film_col[ci + 3u], uint(min(k.a, 4000.0) * FILM_A_FIX * float(q) / FILM_FIX + 0.5));
            atomicAdd(film_col[ci + 4u], q);
            atomicAdd(film_col[ci + 5u], uint(clamp(vol, 0.0, 1.0) * 256.0 * float(q) / FILM_FIX + 0.5));
            film_col[ci + 6u] = uint(row + 1);
            // The tile's intake flag (film.comp: after the two copies' flags).
            ftile[2u * nt + uint(t.y / 16) * tx + uint(t.x / 16)] = 1u;
        }
    }
}
