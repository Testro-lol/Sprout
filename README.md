# Sprout

A fast, interactive 2D multi-fluid sandbox in Rust and Vulkan (via Vulkano). Pour water,
oil and syrup, stir them with the mouse, drop obstacles in, and watch them splash, layer
and blend. The whole simulation runs in compute shaders on the GPU.

![Playground map with the control panel](docs/playground.png)

The fluid solver, its tools and maps; see [Physics](#physics) and [Validation](#validation) for what changed in the second round. The code is laid out so
that more simulations can join the same world later (see [Code layout](#code-layout)).

## Build and run

You need:

- Rust 1.88 or newer (the code uses edition 2024 and let-chains; built here with 1.99).
- A GPU and driver with Vulkan 1.1.
- `glslc` on the `PATH` at build time. Fedora: `sudo dnf install glslc`. Debian/Ubuntu:
  `sudo apt install glslc`. Arch: `sudo pacman -S shaderc`. The Vulkan SDK also ships it.
  Set the `GLSLC` environment variable to use a compiler elsewhere.

```bash
cargo run --release
```

A clean build takes under a minute. The window opens on the Playground map; the panel in
the corner has everything, and every control also has a key.

Useful options (run `sprout --help` for all of them):

| Option | Effect |
|---|---|
| `--map <N or name>` | Starting map, e.g. `--map 2` or `--map dam` |
| `--quality low\|medium\|high\|ultra` | Grid of 384x216, 512x288 (default), 768x432 or 1024x576 cells |
| `--no-vsync` | Uncapped frame rate |
| `--set name=value` | Override a physics parameter (SI units), e.g. `--set nu_syrup=0.09` or `--set cell_size=0.001` |
| `--frames N` | Run N frames, print frame timings, exit |

## Controls

| Input | Action |
|---|---|
| Left mouse | Use the selected tool |
| Right mouse | Grab / stir, whatever tool is selected |
| Middle drag, wheel | Pan, zoom about the cursor |
| Shift + wheel, `[` `]` | Brush size |
| `-` `=` | Pour rate |
| `1` `2` `3` | Water, oil, syrup |
| `Q` `W` `E` | Pour, grab / stir, erase |
| `A` `S` `D` `X` | Place a circle, box, ramp; remove a solid |
| `T` `G` `F` | Rotate the box or ramp, flip the ramp |
| `Space` `N` `R` | Pause, single step, reset the map |
| `PgUp` `PgDn` | Previous / next map |
| `Home` `H` `F11` `Esc` | Reset view, hide panel, fullscreen, quit |

The panel also has sliders for the fluid properties (viscosities, surface tensions, contact
angles, densities, cell size), the solver (CFL number, tolerances; it also shows the current
substep and iteration counts), the grid resolution and slow motion.

Pouring lays down a solid stream as wide as the brush. A low pour rate narrows it. Nothing
pours while paused, because the stream is laid out for the time the simulation advances.

## Maps

| # | Name | What it shows |
|---|---|---|
| 0 | Playground | A pool, a few obstacles, and oil and syrup about to fall in |
| 1 | Dam Break | A tall column of water collapses across the floor and over a step |
| 2 | Upside Down | Syrup over water over oil: the layers overturn into their stable order |
| 3 | Three Columns | Oil, water and syrup side by side: syrup slides under, oil rides over |
| 4 | Cascade | Three reservoirs drain down a zig-zag of ramps into a shared basin |
| 5 | Empty Tank | Nothing but walls |

Maps are written in resolution-independent units in [src/maps.rs](src/maps.rs); a new one
is a dozen lines.

Upside Down six seconds in, and settled after forty:

![Upside Down at 6 s](docs/upside-down_0.png)
![Upside Down at 40 s](docs/upside-down_1.png)

## Other commands

```bash
cargo run --release -- validate          # physics and tool checks, off-screen (about 40 s)
cargo run --release -- --selftest        # drives the real window through every tool and key
cargo run --release -- bench             # throughput table (about a minute)
cargo run --release -- shot out.png --map 2 --at 3,10
cargo test --release                     # CPU-side unit tests
```

`validate` takes a test name (`hydrostatic`, `dam`, `ritter`, `layers`, `viscosity`,
`viscous`, `buoyancy`, `surface`, `settle`, `convergence`, `tools`) and `--shots DIR` to save a
picture of each scenario. Two debug commands are not in `--help`: `sprout probe [DEPTH]`
prints an energy, step and iteration trace (`--map N` for a map, `--at T` for the duration),
and `sprout profile` splits a frame into CPU recording and GPU time and times every pass of
one substep with GPU timestamps. `shot` renders
any map at the given simulated times, with `--ui` to include the panel.

## Physics

**Units and scale.** The solver works in grid units (cells, seconds, densities relative to
water). Everything you can edit is in SI and converted in one place
([src/sim/params.rs](src/sim/params.rs), `SimParams::gpu_units`), together with the physical
size of a cell, `cell_size` (default 1 cm, so gravity is 981 cells/s²). Changing the cell size
changes the physical scale of the whole tank: at 1 mm per cell surface tension matters, at
1 cm it barely does (see below).

**Method.** Particles carry the fluid and a staggered grid solves the forces (APIC, a
low-dissipation particle-in-cell scheme). Four particles per cell at rest. Each substep:

1. The GPU picks the substep length (`step.comp`) from the CFL condition,
   `dt = cfl / (u_max + sqrt(5 g dx))`, with `u_max` the fastest face velocity after the last
   projection, the fastest particle displacement of the last advection (velocity, density
   correction and oil slip) and what the mouse or a pour will impose; `cfl` = 2 cells. It is
   also capped at 1/60 s and at the capillary-wave limit of explicit surface tension. The
   CPU only asks for a span of simulated time per frame; it never caps velocities.
2. One pass over the particles gathers velocity from the grid, moves them, collides them
   with solids, and scatters mass and momentum back to the grid; gravity is added on the grid.
3. Surface tension: curvature of smoothed liquid and oil indicators, turned into pressure
   jumps at liquid–air and oil–aqueous interfaces (below).
4. Viscosity, full stress tensor (below).
5. Pressure: the velocity field is made divergence-free with a variable-density, cut-cell,
   ghost-fluid Poisson solve, by conjugate gradients preconditioned with an aggregation
   multigrid V-cycle (red–black Gauss–Seidel smoothing, Galerkin coarse operators,
   coarse-correction weight 1.8). It stops at a relative residual of 1e-4, so its accuracy
   does not depend on the resolution: a resting pool needs about 3 iterations, the dam
   break at most 10 (512 × 288) to 12 (1600 × 900), with no unconverged solve in either.
6. A second solve with the same matrix corrects density errors by nudging particle
   positions only; it never adds momentum.

How many Krylov iterations each substep records adapts to what recent solves needed (read
back a frame later); iterations recorded after convergence are dispatched indirectly with
zero work. A solve that runs out of recorded iterations is counted and reported (`probe`,
`validate`, the panel). Every substep is recorded once into a raw secondary command buffer
with explicit barriers and replayed ([src/sim/raw.rs](src/sim/raw.rs)); vulkano's
automatic synchronisation cost several milliseconds of CPU per substep at this dispatch count.

**Viscosity** is implicit (backward Euler) in the variational form of Batty & Bridson (2008):
the new velocities minimise kinetic energy change plus `dt * mu * D:D`, D the strain rate.
That is `div(mu (grad u + grad u^T))`, not `mu lap(u)`: it is consistent across viscosity
jumps (water next to syrup), gives traction-free free surfaces and does not slow a rigid
rotation. Solid walls are no-slip, with the wall where it is (half a cell from the first
tangential sample). The system is symmetric positive definite, solved by Jacobi-
preconditioned conjugate gradients to a relative residual of 1e-3, so no viscosity is too
large for the time step. Where the viscous diffusion number `nu dt / dx²` is below 1e-3
(water: about 1e-5 on 1 cm cells) the same operator is applied explicitly instead, which is
stable to about 0.12; in a water-only tank that is the whole tank, in mixed tanks the
implicit solve runs only on 16 × 16 tiles near more viscous fluid.

**Surface tension** uses the ghost-fluid method (Kang, Fedkiw & Liu 2000) for both kinds of
interface, so the jump is as sharp as the free surface and is balanced exactly by the
pressure gradient:

- *Liquid–air:* the air side of a surface face takes the Laplace pressure `sigma * kappa` at
  the interface's sub-cell position. `sigma` is that of the liquid there (water, syrup or oil
  against air, blended by composition).
- *Oil–aqueous:* where the oil fraction crosses 1/2 between two cells, the pressure jumps by
  `sigma_oil * kappa` across that face.
- *Water–syrup:* miscible, one aqueous phase, no interfacial tension (the weak transient
  Korteweg stress of a fresh miscible interface is neglected).

Curvature is the divergence of the unit normal of a 5 × 5 binomially smoothed occupancy
field. **Wetting** is the minimal static-contact-angle model (Brackbill, Kothe & Zemach 1992):
within 1.5 cells of a solid the normal is replaced by the one the material's contact angle
prescribes. There is no contact-angle hysteresis and no contact-line slip model; a contact
line moves as fast as the no-slip viscous wall lets it.

**The three fluids** (all editable; `--set nu_oil=8e-5`, `--set sigma_water=0.07`,
`--set contact_water=40`, ...):

| | Density | Kinematic viscosity | Tension vs air | Contact angle |
|---|---|---|---|---|
| Water | 998 kg/m³ | 1.0e-6 m²/s | 0.072 N/m | 90° |
| Syrup | 1.4 × water | 0.06 m²/s | 0.076 N/m | 90° |
| Oil | 0.9 × water | 6.0e-5 m²/s | 0.032 N/m | 90° |

Oil against water or syrup: 0.025 N/m. Water's values are the standard 20 °C ones; oil's are
typical of olive or sunflower oil (50–90 mPa·s, 0.03 N/m against air, 0.02–0.03 N/m against
water). Syrup is a choice, not a measured product: a thick golden syrup or honey. The
contact angles are a neutral default because the scene does not say what the solids are.
The previous version tuned oil to 50 × and water to zero viscosity "for feel"; water now has
its real viscosity, and it changes almost nothing at this scale (see Validation).

- **Water and syrup are miscible.** Each aqueous particle carries a syrup fraction.
  Neighbours exchange it at a set rate, conserving total syrup exactly. Density blends
  linearly with the fraction and viscosity logarithmically.
- **Oil is a separate phase.** Where oil and water share a cell, below what the grid
  resolves, they would otherwise move as one fluid. Sub-cell droplets therefore drift at
  their terminal velocity: a droplet one cell across (`droplet`, in cells) in whichever
  phase is the majority, with drag `C_d = 24/Re + 0.44`. This follows the cell size and the
  viscosity of the fluids (a droplet barely drifts through syrup). The previous model used a
  fixed relaxation time, which made the drift ten times faster in cells per second at 1 mm
  cells and ignored viscosity; at 1 mm it broke up oil blobs.

**Liquid and air.** Air is not simulated. A cell is liquid when it holds at least half its
rest density; sparser spray flies ballistically until it lands.

**Numerical stabilisation.** Every device below exists for a stated reason; none limits
speed or damps motion globally.

| Device | Purpose | Measured effect |
|---|---|---|
| Density correction (position-only potential, 1 − exp(−dt/6 ms) of the error per substep, at most 0.5) | Removes the volume drift that particle advection accumulates | Volume −0.29 % after the 40 s overturn; lifts a resting 1 M-particle pool by 0.016 cells (an energy change of 1.5e-4 of its potential energy). A time constant rather than a per-step share, so short substeps do not over-correct; per-step it caused visible jitter |
| Correction displacement ≤ 0.5 cell per substep | Keeps one bad cell from throwing particles | Inactive in every validation scene except at impacts |
| Crowding jitter (cells over 1.5 × rest density) | Lets particles stacked on one spot separate | Unchanged from before |
| APIC matrix clamp ‖C‖ ≤ 1/dt | Bounds the affine velocity of a particle in thin spray | Rarely active at CFL 2 |
| Oil slip ≤ cfl cells per substep | The step size already accounts for the slip; this only acts in the substep it appears | — |
| Ghost-fluid fraction θ ≥ 0.1 | Bounds the pressure coefficient of a nearly empty surface cell | Standard; unchanged |
| Curvature only where the smoothed indicator's gradient is resolved; ‖κ‖ ≤ 0.5 / cell | Radii under two cells are not representable; particle noise there would become huge pressure jumps | Spurious currents around a resting drop 1.2 % of the capillary velocity |
| Implicit-viscosity tiles (diffusion number ≥ 1e-3) | Skip Krylov work where it changes nothing above tolerance; those tiles get the explicit step | Viscosity checks unchanged; viscosity cost 5.8 → 2.3 ms per substep on Playground 1024 |

## Validation

`sprout validate` runs these on the real solver, mostly at 512 × 288, and compares against a
known answer. All checks pass on the machine below (one measurement is reported without a
pass/fail, see Limitations). Measured values from the last run, with the previous version
where it was measured under identical conditions:

| Test | Now | Before | Expected |
|---|---|---|---|
| Resting pool 1024 × 576, 250 cells deep, 1 M particles, 6 s: peak KE per particle | 0.0011 | 162 (25 cells/s sloshing) | 0 |
| Same: mean height | constant to 0.016 cells | swung by 1.7 cells | constant |
| Same: unconverged pressure solves | 0 of 367 (≤ 10 iterations) | — (fixed 30 sweeps) | 0 |
| Resting pool 512 × 288, 10 s | level −0.57 %, rms 0.00018 √(gh) | −0.51 %, 0.00007 | level, still |
| Syrup slump vs Huppert (1982) at 16 s | 1.02, exponent 0.204 | 1.13, 0.192 | 1.0, 0.2 |
| Shear decay, free slab (traction-free surfaces) | −4.4 % of ν(π/H)² | not tested | exact rate |
| Shear decay on a no-slip floor | −1.7 % of ν(π/2H)² | not tested | exact rate |
| Rigid rotation, ν = 0.01 m²/s, angular momentum after 1 s | 1.011 | 0.919 (Laplacian form, converged) | 1 |
| Resting drop, 1 mm cells: pressure / (σ/R) | 1.002 | no tension | 1 |
| Spurious currents around it | 1.2 % of √(σ/ρR) | — | small |
| Oil disc in water: pressure jump / (σ_oa/R) | 1.002 | — | 1 |
| Water drop n = 2 oscillation vs Rayleigh | +0.6 % | — | 0 |
| Syrup blob in water (miscible): shape change | 0.000 | — | none |
| Contact angle 60° / 90° / 120° | 73° / 86° / 132° | none | as set |
| Oil square in water, 0.5 s: R_max / R_equal-area | 0.99 with tension, 1.19 without | — | 1 = disc |
| Dam break 2:1, front Z at T = 3.33, a = 30 / 60 / 120 cells | 4.495 / 4.530 / 4.546 | — | converging |
| Dam break front speed dZ/dT | 1.47 | 1.47 | experiments 1.14–1.74 |
| Dam break front vs Martin & Moyce table | +8 … +15 % | +8 … +16 % | see below |
| Shallow dam break vs Ritter | depth 0.459, speed 0.638 | 0.463, 0.637 | 0.444, 0.667 |
| Overturn of three layers, 40 s | oil / water / syrup at 149 / 83 / 45, 0.6 % trapped, energy never rises | 151 / 79 / 48 at 30 s | sorted |
| Stir then 10 s of rest: energy above rest | 2066 → 1072 | 2072 → 1087 | falling |
| Same: particle-scale jitter | 9.3 → 3.0 cells/s (0.4 % of the energy) | 9.1 → 3.1 | decaying |
| Tools: pour, erase, recycle, grab, solids | all pass | all pass | |

What the numbers say about the reported problems:

- **"Rarely settles at high particle counts"** was the pressure solve: thirty SOR sweeps do
  not converge a 576-cell-tall grid, so the pool compressed under gravity, the density
  correction lifted it back, and the cycle pumped energy in. With 300 sweeps the old solver
  settled too; the new solve stops on its residual and needs 3–10 iterations.
- **"Syrup front 13 % fast"** was the viscosity solve: six Gauss–Seidel sweeps leave the
  long-wavelength viscous modes largely unsolved, so syrup acted thinner than set. With 60
  sweeps the old code gave 1.04; the new solve gives 1.02 and decays shear at the analytical
  rate.
- **"Dam break 8–16 % fast"** is not a solver error. The front does not move when the pressure
  sweeps, the substep rate, the density correction or the speed cap change, it converges
  under grid refinement (0.4 % between the two finest grids), and its speed (1.47) sits
  inside the spread of published experiments (1.14–1.74 for v/√(gH) after t* = 1, Lobovský et
  al. 2014, table 1; Martin & Moyce's own 2:1 run is 1.69 there). The table the check
  compares positions against (1.32) is one data set typed from memory, and real experiments
  lose time to the gate and to floor friction that a frictionless instant release does not
  have.
- **"Water sloshes excessively"** after a stir is physical: waves in a 5 m tank decay over
  minutes. The same stir on the old and new solvers loses the same energy at the same rate,
  and the old one had no injection there. Water's real viscosity (1e-6 m²/s) is now in;
  on 1 cm cells it changes a velocity by about 1e-4 of its neighbour difference per substep,
  which is why it is not visible. Raising it would be tuning, not physics.
- **"Oil does not bead"**: oil now has interfacial tension and rounds off (0.99 vs 1.19). At
  the default 1 cm cells the Bond number of any drop the grid can resolve is large
  (a 5 cm oil drop in water: Δρ g R² / σ ≈ 100), so gravity wins and blobs flatten, as real
  5 cm blobs would. Set `--set cell_size=0.001` to see capillary-scale beading.

`sprout --selftest` drives the real window through every tool and key (19 checks, all pass).
`cargo test` covers the CPU side: 18 tests, all pass. `cargo clippy` is clean.

## Performance

Hardware: AMD Ryzen 5 3600, NVIDIA RTX 3060 Ti 8 GB (driver 615.78), Fedora 45, Wayland.
Both versions measured the same day with the same command, a 1600 × 900 window, uncapped
(`--no-vsync --frames 1800`, the first 360 frames skipped), nothing else on the GPU. Frame
times in ms; "sim speed" is simulated seconds per wall-clock second.

| Scene | Version | mean | p5 | p25 | p50 | p75 | p95 | p99 | max | sim speed |
|---|---|---|---|---|---|---|---|---|---|---|
| Dam Break 1600 × 900, 1.04 M water | before | 13.4 | 9.9 | 10.1 | 10.4 | 18.8 | 19.1 | 19.4 | 23.4 | 0.42 |
| | now | 16.1 | 11.3 | 11.9 | 12.7 | 22.2 | 24.1 | 25.2 | 28.7 | 0.07 |
| | now, fixed 240 substeps/s | 17.5 | 12.5 | 13.1 | 13.7 | 24.3 | 26.1 | 27.1 | 29.2 | 0.32 |
| Dam Break 768 × 432, 237 k water | before | 0.9 | 0.4 | 0.4 | 0.4 | 0.7 | 2.8 | 2.9 | 13.6 | 0.96 |
| | now | 11.3 | 0.5 | 4.4 | 13.2 | 16.7 | 18.9 | 21.4 | 24.9 | 0.57 |
| Playground 1024 × 576, 732 k mixed | before | 17.2 | 12.2 | 17.7 | 17.9 | 18.0 | 18.2 | 20.0 | 20.5 | 0.70 |
| | now | 14.5 | 9.3 | 9.9 | 17.0 | 18.2 | 19.3 | 20.1 | 22.0 | 0.14 |
| | now, fixed 240 substeps/s | 14.7 | 10.3 | 10.9 | 11.5 | 20.4 | 22.0 | 22.6 | 24.9 | 0.39 |
| Upside Down 512 × 288, 369 k mixed | before | 1.5 | 0.6 | 0.6 | 0.9 | 2.7 | 3.2 | 3.2 | 4.0 | 0.98 |
| | now | 8.7 | 0.6 | 0.8 | 7.2 | 13.6 | 18.9 | 20.4 | 25.3 | 0.87 |

Resting 1 M-particle pool, 1024 × 576, 6 simulated seconds (`sprout probe 250 --grid
1024x576 --at 6`, same command on both): 10.2 s wall before, 3.7 s now, and it ends at rest
(before: still moving at up to 2.5 cells/s).

Reading these honestly:

- **Simulated time per second is lower in fast scenes, and that is mostly the step count.**
  The previous version took 240 substeps per simulated second whatever happened, and clipped
  every particle to 600 cells/s to make that look stable. In the 1600 × 900 dam break the
  impact jets reach 2000–3000 cells/s (0.1 % of the particles above 2000; the old code
  produces the same jets once its clamp is lifted, and total energy never rises), so a step
  that moves nothing more than 2 cells needs about 1600 substeps per simulated second. At
  the old fixed rate the new solver costs 1.3 × (dam) to 1.8 × (playground) more per
  simulated second; the rest of the gap is the steps the physics requires.
- **Calm scenes are faster** (2.7 × on the resting pool): the CFL step there is 1/60 s.
- **The frame rate holds near 60 fps** in heavy scenes: the app gives each frame fewer
  substeps and slows simulated time instead, as before; the panel shows it.
- **Small scenes now run at about 60–120 fps uncapped instead of 700–1100**: each frame
  simulates its share in one or two long substeps rather than a fraction of a short one,
  and a substep with converged solves costs more than an SOR sweep set.

Where one substep goes (Dam Break 1600 × 900, GPU timestamps from `sprout profile`, ms):
particles 2.7, pressure solve 5.6 (multigrid levels 3.9, Krylov vectors 1.7), viscosity
(explicit, water) 0.6, surface tension 0.7, other grid passes 1.1; total 10.7. The pressure
solve is bandwidth-bound on the finest level (roughly 370 bytes per cell per iteration).
Recording a frame costs 0.1 ms of CPU.

## Limitations

- **Fast flows at high resolution are expensive.** See Performance: the CFL step follows the
  fastest jets. A larger CFL number trades accuracy (at CFL 4 the dam front moves by 2 % and
  energy rises by 0.1 % between samples); particle sub-stepping would allow longer grid
  steps but is not implemented.
- **Oil–water capillary oscillation is off.** The static jump across an oil–water interface
  is exact (1.002), but an oil drop in water oscillates 27 % slower than the inviscid theory
  with the real oil viscosity, and 30 % faster with both fluids made inviscid, where it
  should match. The dynamic error is not understood yet; `validate surface` reports it
  without a pass/fail.
- **Contact angles are biased** about 12° towards non-wetting (60° → 73°, 120° → 132°); there
  is no hysteresis and no contact-line slip.
- **Free surfaces sit about half a cell off** for viscosity (the free-slab shear decays 4 %
  slow, as for a slab one cell thicker).
- **Surface tension needs small cells to matter**: see the beading note above.
- **Runs are not bit-reproducible**: the periodic particle sort orders particles within a
  cell by atomics, and the crowding jitter depends on that order, so chaotic scenes
  (the overturn) end slightly differently every run (oil-layer trapping 0.45–1 % at 40 s).
  This was already true before.
- **Air is nothing.** Trapped pockets have no pressure and collapse; there are no bubbles.
  A tank filled to the lid has no pressure reference and is not supported.
- **Solids are static**, and a tank is limited to four particles per cell of its area.
  Solids thinner than about three cells can leak (particles move up to 2 cells per substep).
- **Tested on one machine.** NVIDIA on Linux only; the solver needs subgroup arithmetic
  (Vulkan 1.1).
- No save or load, and no undo.

## Code layout

| Path | Role |
|---|---|
| [src/world.rs](src/world.rs) | The sandbox world: solids plus the simulations living in it. New simulations plug in here. |
| [src/scene.rs](src/scene.rs) | Shapes and their rasterisation into the fields the solver and renderer use |
| [src/sim/](src/sim/mod.rs) | The fluid solver: buffers, pipelines, the substep; [params.rs](src/sim/params.rs) holds every material property; [raw.rs](src/sim/raw.rs) the hand-synchronised substep command buffers |
| [shaders/](shaders/) | All GPU code. `common.glsl` documents the grid layout and every buffer. |
| [src/render.rs](src/render.rs), [src/gui.rs](src/gui.rs) | Particle splatting and compositing; a small egui painter |
| [src/app.rs](src/app.rs), [src/ui.rs](src/ui.rs), [src/pacing.rs](src/pacing.rs) | Window, input, panel, substep pacing |
| [src/maps.rs](src/maps.rs) | Built-in maps |
| [src/validate.rs](src/validate.rs), [src/selftest.rs](src/selftest.rs), [src/headless.rs](src/headless.rs) | Checks, benchmarks, screenshots |

Shaders are compiled by [build.rs](build.rs) with the system `glslc` rather than
`vulkano-shaders`, which builds the shaderc C++ toolchain from source and fails with
current CMake.
