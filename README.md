# Sprout

A fast, interactive 2D multi-fluid sandbox in Rust and Vulkan (via Vulkano). Pour water,
oil and syrup, stir them with the mouse, drop obstacles in, and watch them splash, layer
and blend. The whole simulation runs in compute shaders on the GPU.

![Playground map with the control panel](docs/playground.png)

This is the first version: the fluid solver, its tools and maps. The code is laid out so
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
| `--set name=value` | Override a physics parameter, e.g. `--set nu_syrup=900` |
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

The panel also has sliders for the fluid properties (viscosities, densities, how fast oil
separates and syrup blends), the solver (substeps per second, sweep counts), the grid
resolution and slow motion.

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

`validate` takes a test name (`hydrostatic`, `dam`, `ritter`, `layers`, `viscous`,
`buoyancy`, `tools`) and `--shots DIR` to save a picture of each scenario. `shot` renders
any map at the given simulated times, with `--ui` to include the panel.

## Physics

Units: one grid cell is one length unit, nominally 1 cm, so gravity is 981 cells/s².
Densities are relative to water.

**Method.** Particles carry the fluid and a staggered grid solves the forces (APIC, a
low-dissipation particle-in-cell scheme). Four particles per cell at rest. Each substep
(240 per simulated second by default):

1. One pass over the particles gathers velocity from the grid, moves them, collides them
   with solids, and scatters mass and momentum back to the grid.
2. Viscosity is applied implicitly, so syrup can be hundreds of times thicker than an
   explicit step would allow. Solid surfaces are no-slip for viscous fluid.
3. A pressure solve makes the velocity field divergence-free. It accounts for the density
   of every cell, which is what produces buoyancy: light fluid is pushed up through heavy
   fluid by the same pressure that holds a pool at rest. The free surface sits at its
   sub-cell position, and solids cut cells and faces by their true open fraction.
4. A second solve with the same matrix corrects density errors by nudging particle
   positions only. It never adds momentum, so it cannot pump energy into the flow.

**The three fluids.**

| | Density | Kinematic viscosity (cells²/s) |
|---|---|---|
| Water | 1.0 | 0 |
| Oil | 0.9 | 30 |
| Syrup | 1.4 | 600 |

The viscosities are chosen for feel at this scale, not taken from material tables: syrup
is thicker than real honey, oil thicker than real cooking oil.

- **Water and syrup are miscible.** Each aqueous particle carries a syrup fraction.
  Neighbours exchange it at a set rate, conserving total syrup exactly. Density blends
  linearly with the fraction and viscosity logarithmically, so a little water thins syrup
  a lot, as with real sugar solutions.
- **Oil is a separate phase.** Oil particles stay oil. Where oil and water share a cell,
  below what the grid can resolve, they would otherwise move as one fluid and never
  separate. A slip model handles that scale: oil rises through the mixture and the
  aqueous phase sinks, each in proportion to the other's share, so pure regions do not
  slip and the two fluxes cancel.

**Liquid and air.** Air is not simulated. A cell is liquid when it holds at least half
its rest density; sparser spray flies ballistically until it lands. Liquid being stretched
(a falling stream) narrows instead of tearing, because one-cell gaps inside it stay
connected.

**Details that matter for stability**, each found by a failing test: faces leading into
cells that are almost entirely solid are sealed, because the pressure solve would treat
such a cell as a free surface and drain fluid into it; particles caught under a newly
placed solid are mirrored out through its surface so they keep their area; and badly
over-full spots are jittered apart so that stacked particles can separate.

## Validation

`sprout validate` runs these on the real solver at 512x288 and compares against a known
answer. All 35 checks pass on the machine below. Measured values from the last run:

| Test | Measured | Expected |
|---|---|---|
| Resting pool, 10 s | level −0.51 %, residual speed 0.00007 √(gh) | stays level and still |
| Shallow dam break vs Ritter's exact solution | depth 0.463 h₀, speed 0.637 √(gh₀) at the dam | 0.444, 0.667 |
| Dam break, column 2:1, vs Martin & Moyce (1952) | front 8–16 % ahead, front speed 1.47 | experiment 1.32 |
| Three layers stacked upside down, 30 s | oil / water / syrup end at mean heights 151 / 79 / 48 cells; 1 % of water left above the oil; volume −0.45 % | sorted by density |
| Same run, energy | never rises between samples; 0.3 % of the released energy left as motion | only falls |
| Same run, conservation | syrup volume +0.0000 %, oil particles unchanged | conserved |
| Syrup slumping vs Huppert's (1982) viscous current | exponent 0.192, front 13 % ahead at 16 s | exponent 0.2 |
| Water vs syrup column after 1 s | fronts at 431 vs 166 cells | water far ahead |
| Oil blob at the bottom, syrup blob at the top of a pool | oil reaches the surface, syrup the floor | rise and sink |
| Pour, erase, slot reuse, grab, place and remove solids | counts exact, pool rises by the displaced volume (36.00 vs 35.90 cells mean height), no clumps | as stated |

`sprout --selftest` opens the real window and feeds a script through the same handlers
that mouse and keyboard events reach: pouring each fluid, brush keys, placing and removing
solids, both stir buttons, erase, pause and single step, zoom, pan, map reset, next map,
and a change of resolution. All 19 checks pass. It does not replace a hand on the mouse:
how the operating system delivers real input events is the one part not exercised.

`cargo test` covers the CPU side (shape distances, rasterised solids, map building,
option parsing, frame pacing): 17 tests, all passing.

Two caveats on the references. The Martin & Moyce numbers in
[src/validate.rs](src/validate.rs) are the table as commonly reproduced in the dam-break
literature, typed from memory and not checked against the paper; the Ritter check beside
it needs no data. And a frictionless solver released instantly is expected to run ahead of
that experiment, whose gate takes time to open, so the 20 % tolerance there is loose.

## Performance

Hardware: AMD Ryzen 5 3600, NVIDIA RTX 3060 Ti 8 GB (driver 615.78), Fedora 45, Wayland,
165 Hz display.

`sprout bench`, off-screen at 1920x1080, default settings (240 substeps/s, 30 pressure
sweeps, 6 viscosity sweeps), each scene timed after 6 s of simulated mixing. Times are
wall-clock for submitted-and-finished work, CPU recording included.

| Map | Grid | Particles | ms per substep | Render ms | Real-time load | Frame rate left |
|---|---|---|---|---|---|---|
| Dam Break | 384x216 | 57,846 | 0.59 | 0.27 | 14 % | 3,194 fps |
| Playground | 384x216 | 99,600 | 0.85 | 0.30 | 20 % | 2,662 fps |
| Upside Down | 384x216 | 205,632 | 1.29 | 0.33 | 31 % | 2,124 fps |
| Dam Break | 512x288 | 104,000 | 0.85 | 0.33 | 20 % | 2,389 fps |
| Playground | 512x288 | 179,372 | 1.54 | 0.34 | 37 % | 1,857 fps |
| Upside Down | 512x288 | 369,380 | 2.46 | 0.43 | 59 % | 955 fps |
| Dam Break | 768x432 | 237,384 | 2.25 | 0.41 | 54 % | 1,120 fps |
| Playground | 768x432 | 408,036 | 3.40 | 0.49 | 81 % | 378 fps |
| Upside Down | 768x432 | 839,724 | 5.40 | 0.69 | 130 % | 0.77x speed |
| Dam Break | 1024x576 | 424,534 | 3.75 | 0.66 | 90 % | 149 fps |
| Playground | 1024x576 | 732,432 | 5.82 | 0.80 | 140 % | 0.72x speed |
| Upside Down | 1024x576 | 1,498,496 | 9.34 | 1.15 | 224 % | 0.45x speed |

"Real-time load" is the share of each second needed to run one simulated second of
substeps. "Frame rate left" is what the rest of the second renders at. Over 100 %, the
scene cannot run at full speed on this GPU.

In the actual window (1600x900, `--frames`):

| Scene | Present mode | Frame rate | Frame time p99 | Simulation speed |
|---|---|---|---|---|
| Playground, 512x288, 179k particles | VSync | 165 fps | 7.4 ms | 1.00x |
| Playground, 512x288, 179k particles | uncapped | 1,670 fps | 2.0 ms | 0.98x |
| Upside Down, 512x288, 369k particles | uncapped | 679 fps | 3.2 ms | 0.99x |
| Playground, 768x432, 408k particles | uncapped | 358 fps | 5.3 ms | 0.99x |
| Upside Down, 768x432, 840k particles | uncapped | 60 fps | 18.8 ms | 0.74x |

When a scene is too heavy, the app gives each frame fewer substeps: simulated time slows
down and the frame rate holds near 60 fps, as in the last row. The panel says so when it
happens.

Where the time goes at 512x288 on Playground: about 40 % in the per-particle pass and the
fixed grid passes, about 60 % in the pressure and viscosity sweeps (their share of command
recording on the CPU included). Particles are re-sorted by cell every 128 substeps;
without that a substep of Upside Down slows by a third over twenty seconds, as the fluid
mixes and neighbours drift apart in memory.

## Limitations

- **No surface tension.** Oil and water have no interfacial tension, so there are no
  round droplets or beading, and a flat layer of water resting on oil starts seeping
  through at once instead of waiting for fingers to form.
- **Water never quite stops.** It has zero viscosity, so swirls and sloshing fade slowly:
  ten seconds after a hard stir, half the energy is still there.
- **The pressure solve is not run to convergence.** Thirty sweeps, warm-started, are
  enough for the checks above, but violent large-scale motion leaves small errors that
  the density correction then cleans up. Volume drifts by about half a percent.
- **Fronts run ahead of the references**: 8–16 % in the dam break, 13 % in the viscous
  current.
- **Speed cap.** Particles move at most 2.5 cells per substep (600 cells/s). A fall from
  the top of the default tank would reach about 740, so the fastest streams are clipped.
  Solids thinner than about three cells can leak at such speeds.
- **Fewer than 240 substeps per second is rougher.** At 120 the falling stream shows
  occasional gaps.
- **Air is nothing.** Trapped pockets have no pressure and collapse; there are no bubbles.
- **Solids are static**, and a tank is limited to four particles per cell of its area.
  When it is full, pouring stops until something is erased.
- **Tested on one machine.** NVIDIA on Linux only. Nothing is vendor-specific, but AMD,
  Intel, Windows and macOS (through MoltenVK) are untried.
- No save or load, and no undo.

## Code layout

| Path | Role |
|---|---|
| [src/world.rs](src/world.rs) | The sandbox world: solids plus the simulations living in it. New simulations plug in here. |
| [src/scene.rs](src/scene.rs) | Shapes and their rasterisation into the fields the solver and renderer use |
| [src/sim/](src/sim/mod.rs) | The fluid solver: buffers, pipelines, the substep |
| [shaders/](shaders/) | All GPU code. `common.glsl` documents the grid layout and every buffer. |
| [src/render.rs](src/render.rs), [src/gui.rs](src/gui.rs) | Particle splatting and compositing; a small egui painter |
| [src/app.rs](src/app.rs), [src/ui.rs](src/ui.rs), [src/pacing.rs](src/pacing.rs) | Window, input, panel, substep pacing |
| [src/maps.rs](src/maps.rs) | Built-in maps |
| [src/validate.rs](src/validate.rs), [src/selftest.rs](src/selftest.rs), [src/headless.rs](src/headless.rs) | Checks, benchmarks, screenshots |

Shaders are compiled by [build.rs](build.rs) with the system `glslc` rather than
`vulkano-shaders`, which builds the shaderc C++ toolchain from source and fails with
current CMake.
