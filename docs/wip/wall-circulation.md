# WIP: resting viscous pools circulate along the walls

Branch `solver/wall-circulation`. Parked 2026-11-15, unfinished: it fixes the wall flow
but fails 3 validate checks (below). Not merged into `main`.

## Cause (measured)

`grid_prep.comp` adds gravity `g dt` to the v faces *before* the implicit viscosity
(`visc_*.comp`), and the pressure solve (`rhs.comp`, mg/cg, `project.comp`) runs after
it. For a pool at rest, the viscosity solve sees the whole pull `g dt` as motion, and the
no-slip walls hold the liquid next to every wall back against it. The pressure then pushes
that wall layer up. A wall-to-wall layer of lava or syrup at rest rises along each wall at
about 6 cells/s (11.6 cells in 2 s next to the wall), at any viscosity; mean particle speed
0.41 cells/s. Water doesn't show it: it takes the explicit viscous step, which moves a wall
face by only `nu dt / dx²` of `g dt`.

Reproduce (prints how far particles by the left wall moved in 2 s; `LAB_MAP=1` prints a
block map, `LAB_JITTER_MAT=syrup|water` changes the liquid, `LAB_SETTLE=24` runs longer):

```bash
LAB_LAVA_JITTER=1 LAB_FULL=1 LAB_BARE=1 LAB_NOHEAT=1 LAB_WALL=1 ./target/release/sprout lab
```

## Approaches tried

All but the last were tried and reverted in an earlier session; the last is on this
branch.

| Approach | Rest pool | What broke |
|---|---|---|
| Gravity added only after the viscosity (in rhs/project) | 0.006 cells/s | Sludge slump never stopped; Bingham slot 95 % off (g dt slip at walls in gravity-driven wall flows) |
| Last substep's pressure gradient taken off before the viscosity (stored in `vel0`), given back in rhs/project | — | Rigid ice drifted at 157 cells/s, Stefan front +188 % |
| Same, clamped to [0, g dt] on v faces only | — | Slump ran out to 81 cells |
| Viscosity moved after the single projection | — | Slump 65 vs 47 ± 20 %; with `rheo.comp` moved too: 87 cells, unconverged solves |
| **This branch:** pressure solve, then viscosity, then a second pressure-only solve (Batty & Bridson order) | 0.04 – 0.07 cells/s; wall particles 0.02 cells in 2 s (was 11.6) | 3 validate checks (below) |

## Current change (this branch)

- `src/sim/mod.rs` `record_substep`: when the viscosity is implicit and
  `viscous_projection` is 1, the order is rhs → pressure → project → viscosity →
  `rhs_visc` → second pressure solve → `project_visc`. Otherwise the old order.
  `record_pressure(.., second)`: the second solve skips `mg_build`/`cg_scale`. It reuses the
  matrix and multigrid hierarchy and runs `cg_init` + V-cycle + `reduce(8)` + `record_cg`.
  The parameter is in the substep cache key (bit 31 of the caps word).
- `shaders/rhs_visc.comp` (new): `b = A·P1 − div(u_visc)`, stores P1 in `prhs.z` and keeps
  `prhs.y`. The solve runs for the total pressure P, starting from P1, so its first
  residual is just the divergence the viscosity added, and `pres` keeps the total pressure
  for the next warm start and for the `pressed` test.
- `shaders/project_visc.comp` (new): subtracts the gradient of `P − P1` (0 in air: no
  interface jumps) and leaves `vel.zw` (the density-correction gradient) alone. Reports
  umax for the CFL condition.
- `shaders/cg_reduce.comp` mode 8: like mode 0, but marks the density-correction potential
  as done, so it isn't solved twice.
- `src/sim/params.rs`: `viscous_projection` (Accuracy, default 1; 0 restores the old order
  for A/B in one binary, e.g. `--set viscous_projection=0`).
- README: Physics step list and Viscosity paragraph updated, and the Limitations entry
  rewritten (still to be revised to match the final state). Validation and Performance
  aren't updated yet.

## Measured so far

- Lava rest lab: mean particle speed 0.415 → 0.066 cells/s (0.041 in perturbed runs;
  results fall into two groups, independent of the tolerances). Wall particles +11.6 →
  +0.02 cells in 2 s. Syrup the same (0.407 → 0.063). Water is unchanged (0.634 in both
  builds; that is the separate density-correction pump).
- The rest is all density correction: with `--set correction_max=0` the new build is at
  0.001 cells/s, and the old build still at 0.446 with its wall flow. The `LAB_MAP=1` block map
  shows it only in the top band (free surface); the bulk moves < 0.01 cells/s. That part
  belongs to the density-correction session (`solver/` sibling branch).
- `cargo test --release` 42/42 and `cargo clippy --release` are clean. `--selftest` hasn't
  been run.
- Full `validate` on the new build was interrupted at 216 checks: 213 pass, 3 fail. On the
  old build (same tree, without this change) all 136 checks it reached passed.

### Failing checks

1. **layers** (`validate layers`): "water/syrup left above the oil layer's floor" is
   11.43 % (needs < 3 %). Trend at 8/16/24/32/40 s: new 81/71/47/27/11 %, old
   68.5/41/13.6/0.26/0.34 %. Perturbed (`viscosity_oil` × 1.001 / × 0.999): new 0.07 % /
   4.91 %, old 0.32 % / 0.35 %. **Likely cause: the old wall circulation was seeding the
   overturn.** `sprout shot X.png --map 2 --at 1.5` (Upside Down): the old build already
   has syrup sliding down both side walls, while the new interface is still flat and the
   Rayleigh-Taylor instability has to grow from particle noise. Either the check relied on
   the spurious flow, or the onset is slower for another reason; check that before
   changing the test.
2. **steam** "…2 s after the brush is lifted the steam is gone": 2 particles left (old: 0).
3. **steam** "bubbles condense before the surface": 103 of 1928 escaped, 5.3 % (old 87,
   4.5 %; limit 5 %). In a thermal tank water counts as implicit (it can freeze, so
   `viscosity_mode` gives it infinite nu), so these water-only tests take the new path too.

## Next steps

1. Rebuild, then finish the full comparison (about 12 min each; run both, then diff the
   pass lists):
   ```bash
   export PATH=$HOME/.cargo/bin:$PATH; cargo build --release
   ./target/release/sprout validate > /tmp/v_new.txt 2>&1
   ./target/release/sprout validate --set viscous_projection=0 > /tmp/v_old.txt 2>&1
   ```
2. **layers**: find out whether the slower onset only comes from the missing wall seed.
   Run `sprout validate layers` with a small initial interface perturbation in both
   orders, or compare the growth rate with linear RT theory. If the check relied on the
   spurious flow, change the test setup to a seeded interface (standard for RT) and
   document it. That's not a fudge factor, but the README must say why.
3. **steam**: check whether the 2 extra particles / 0.8 % more escapes are noise. Run
   `validate` steam with `--set steam_mixing=1.001` etc. under both `viscous_projection`
   values. If the gap is systematic, look at whether the second solve should also run in
   thermal water-only tanks (the explicit-in-tile path), or skip it when no tile was
   implicit.
4. Cost: in thermal tanks the second solve now runs for water too. Measure with
   `sprout profile --grid 1024x576 --map 0` (per-pass) and the CLAUDE.md benchmark
   (`--no-vsync --frames 1500`, `SPROUT_FRAMES_CSV`, interleaved, Playground / Lava Meets
   Water / Kitchen, check `nvidia-smi` first). A cheaper option: give the second solve its
   own adaptive iteration count (it usually converges in 0 – 2 iterations, but replays the
   first solve's recorded count as empty indirect dispatches).
5. Then `--selftest` and README Validation / Performance / Limitations.
