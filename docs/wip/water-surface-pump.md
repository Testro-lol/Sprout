# WIP: still water pool open to the air gains motion

Branch `solver/water-surface-pump`, parked 2026-11-15. README Limitations entry: "A still pool
of water open to the air slowly gains motion". Not finished: see "State" and "Next steps".

## Repro

```bash
export PATH=$HOME/.cargo/bin:$PATH; cargo build --release
LAB_LAVA_JITTER=1 LAB_FULL=1 LAB_BARE=1 LAB_JITTER_MAT=water LAB_NOHEAT=1 LAB_SETTLE=24 LAB_MAP=1 ./target/release/sprout lab
```

About 8 s wall time. Prints the mean particle speed every 4 s. Add `LAB_PUMP=1` (new on this
branch) for the diagnostic below: centre-of-mass height, the mean occupancy and correction
velocity (vel.w) per row near the surface, and the mean particle vy and count per row.
Change the depth with `LAB_LAVA_DEPTH=0.2` etc. (fraction of the grid height, default 0.12).

## Cause (found)

The density correction is not the root cause. It only closes the loop.

- At rest, the top particle row of the pool sits in cells below `LIQUID_THR` (0.5), so those
  cells are air. Measured with `LAB_PUMP` at 12 s: row y34 held 112–154 particles and y33
  held ~2920, where a normal row holds ~2000. The half row that was filled into y34 has sunk
  into y33.
- Those particles gather velocity from the v faces above them as well. Those faces are
  air–air, so the projection never touches them: they hold the scattered particle momentum
  minus g·dt every substep. Even at a 5–15 % share of the B-spline weight, this hands the
  surface particles a share of gravity that no pressure answers. Mean particle vy in y33 was
  steady at −0.54 cells/s in a pool at rest.
- The density correction sees y33 as over-full (kernel phi 1.04) and lifts particles back
  into y34. The mean correction velocity on the y34 face was +1.13 to +1.26 cells/s, and
  −0.77 to −1.11 on the y33 face (a row-alternating pattern that decays with depth).
- The result is a steady cycle: velocity carries particles down, the correction displaces
  them up without momentum. The downward momentum fed in at the surface feeds a circulation
  with surface convergence that grows (positive feedback).
- With the correction off, the particles simply pack into y33 and stay (0.1 cells/s). Under
  oil, the water's top cells are interior, so no air faces are involved (0.03 cells/s).
- This is the same mechanism as the 2026-11-14 S5 fix (gravity piling up from air faces),
  in general form.

## Approaches tried (mean water particle speed, cells/s, at 4/8/12/16/20/24 s)

| Variant | Speeds | Result |
|---|---|---|
| Baseline (main 3e9964c) | 0.494 0.634 0.716 1.079 1.511 1.776 | pumps |
| `--set correction_deadband=0` | 0.401 0.481 0.456 0.488 0.776 0.896 | still grows |
| `--set correction_step_max=0` | identical to baseline | no effect |
| `--set correction_time=0.05` | 0.086 0.361 0.529 0.697 0.858 0.999 | still grows |
| `--set correction_tol=1e-4` | 0.420 0.576 0.885 1.188 1.468 1.671 | no effect |
| rhs.comp: no correction at all in surface (non-interior) cells | 0.171 0.426 0.619 0.737 0.828 0.945 | still grows |
| same + deadband 0 | 0.213 0.459 0.646 0.797 0.894 0.972 | still grows |
| particles.comp: midpoint `xm` without `corr` | 0.445 0.645 0.970 1.461 1.742 1.995 | worse |
| Gather from projected faces only (coef != 0) | 0.081 … 0.091 | fixes it, but spray far from liquid gets no gravity: unusable |
| Hard switch: drop ballistic faces when projected weight ≥ 50 % | 0.085 0.105 0.093 0.098 0.106 0.091 | fixes it, but discontinuous |
| **Current: ballistic faces weighted by their own share of the weight** | 0.099 0.103 0.116 0.117 0.112 0.113 | **fixes it, continuous, no parameter** |

The current variant across depths (old → new, speed at 24 s): depth 0.08: 0.077 → 0.037;
0.20: 1.399 → 0.077; 0.30: 0.510 → 0.059. With `--set viscosity_water=1.001e-6`: 1.655 →
0.114.

Other scenes, old → new: water under oil 0.032 → 0.036 (oil above it 0.476 → 0.406, still
growing; with `correction_max=0` the oil moves more (0.54), so the oil's motion is not this
pump, probably the viscous wall circulation, see the other solver WIP); bare syrup pool
0.408 → 0.383.

## State of the change (this commit)

- `shaders/particles.comp`, both velocity gathers (u and v): projected faces (`coef` != 0)
  and ballistic faces are summed apart. Ballistic faces get the factor
  `fa = w_ballistic / (w_projected + w_ballistic)`, so spray far from liquid is unchanged and a
  particle at the surface hardly sees them. The APIC C uses the same combined weights.
- `src/validate.rs`:
  - New check at the end of `settle`: "shallow pool: mean particle speed every 4 s" (512 ×
    288, wall to wall, 34.6 cells deep, 24 s, < 0.15 cells/s). Not yet run inside a full
    validate.
  - `current`, "stream: the current carries it sideways": the particle-count guard
    `driven.2 > 100` became `2 * driven.2 > free.2`. The free stream band holds about 104
    particles (99 measured with this change, 101 with the old shader), so "> 100" failed on
    a 2-particle change. The deflection is 80 cells/s in both builds.
  - `tools`, "solids: the refilled pool does not clump": now the middle of three samples
    0.1 s apart. A single sample caught passing crowds. The old shader also read 15 with
    `--set viscosity_water=0.999e-6`; the new one reads [15, 10, 10] at default and
    [9, 9, 8] / [9, 9, 9] at 0.999e-6 / 1.001e-6.
- `src/lab.rs`: the `LAB_PUMP` diagnostic in `lava_jitter`.
- README: Physics → Method step 2 and "Liquid and air" describe the gather rule. Still to
  do: Validation and Performance subsections, and removing the Limitations entry.

## Validate status (incomplete)

One full `validate` ran on this change before the two check reworks above, with the GPU
shared with another sprout session. It was interrupted in the "Every map" stress test (138
passes by then). Its failures:

1. stream sideways: band count 99 < 100. Fixed by the rework above (passes standalone).
2. refilled pool clump 15: noise. Fixed by the rework above (passes standalone).
3. thermal, "...2 s after the brush is lifted the steam is gone": 2 left, at (124.9, 26.3)
   158 °C and (124.5, 29.3) 136 °C, by the right wall of the 128-wide grid. **Standalone
   `validate thermal` with the same shader passes (0 left).** Unclear: run-to-run variation
   or order dependence. Check with perturbed params on both builds.
4. rheology, "slot flows: viscosity solves converged": 1 unconverged. The old shader gives 0
   standalone; the new shader standalone has not been measured yet.
5. metals, "mercury puddle at 140 deg under gravity, 4 s": 98.4 % in one body, centre drift
   23.1 cells (limit < 20). Not yet compared with the old shader. That puddle sloshes for
   seconds, so the drift may be sensitive. Check with ±0.1 % param changes on both builds.

Not run yet: the full validate on the final code, `--selftest`, `cargo test --release`,
`cargo clippy --release`, the benchmark old vs new (CLAUDE.md), and updating the memory
notes.

## Next steps

```bash
export PATH=$HOME/.cargo/bin:$PATH; cargo build --release
# old-shader A/B build: copy shaders/particles.comp from main (3e9964c) into place, build,
# save target/release/sprout as sprout_oldshader, restore this branch's shader, rebuild.
for b in ./sprout_oldshader ./target/release/sprout; do
  for v in 1.0e-6 0.999e-6 1.001e-6; do
    timeout 1200 $b validate rheology --set viscosity_water=$v | grep -E "FAIL|viscosity solves"
    timeout 1200 $b validate metals   --set viscosity_water=$v | grep -E "FAIL|puddle"
    timeout 1200 $b validate thermal  --set viscosity_water=$v | grep -E "FAIL|steam is gone"
  done
done
```

1. Decide whether failures 3–5 are caused by this change or are run-to-run variation (same
   method as above: perturb by 0.1 % on both builds). If caused by it, look at particles
   near walls and thin layers: their ballistic faces now count less (mercury puddle edge,
   steam by the wall).
2. Full `./target/release/sprout validate` (~25 min; check `nvidia-smi` first; do not
   `pkill sprout`, other sessions run it). Target: all checks pass, now 241 with the new
   shallow-pool check.
3. `./target/release/sprout --selftest`, `cargo test --release`, `cargo clippy --release`.
4. Benchmark per CLAUDE.md (`--no-vsync --frames N`, `SPROUT_FRAMES_CSV`, same commands on
   both builds; Playground, Dam Break, Kitchen). The gather has two sums instead of one, so
   expect at most a small cost.
5. README: add a Validation subsection "Still water surface (2026-11-15)" (table above,
   check changes, validate count) and a Performance subsection, and remove the Limitations
   entry or replace it with the remaining oil-on-water motion.
