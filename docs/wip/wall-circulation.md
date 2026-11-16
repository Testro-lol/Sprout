# Wall circulation of resting viscous pools: finished

Branch `solver/wall-circulation`, finished 2026-11-15; not merged into `main`.

Fixed: the implicit viscosity now runs between two pressure solves (`viscous_projection` 1),
and weakly viscous (explicit) tiles take their step before the first solve. The README
records it: Physics (step list, Viscosity), Validation "Wall circulation (2026-11-15)",
Performance "Wall circulation (2026-11-15)", and Limitations.

- Rest lab: wall particles rise 11.6 → 0.02 cells in 2 s; mean speed 0.415 → 0.066 cells/s,
  and what is left is the density correction at the free surface (0.001 with
  `correction_max=0`).
- `validate` 240/240, `--selftest`, `cargo test` (42) and clippy pass.
- Test change: `validate layers` starts from a one-cell sine bump on both interfaces and
  is judged at 48 s (agreed with the user). The old flat start passed only because the
  spurious wall flow seeded the overturn.
- Cost: +6 % (Kitchen) to +36 % (Lava Meets Water) GPU time per substep; the second
  solve needs 6 – 7 iterations against the first one's 8 – 9.

Follow-up done (2026-11-15, branch perf/speed-pass, README Performance "Performance pass"):
the cost is won back. Both pressure solves and the viscosity solve start warm, and the
solver as a whole needs fewer dispatches. Still open:
- A coupled pressure–viscosity solve, or a looser tolerance for the second solve measured
  against the density correction's volume budget.
- The remaining free-surface motion (0.04 – 0.07 cells/s) belongs to the density-correction
  issue (README Limitations, the water pool entry).
