//! The sandbox world: static scene geometry plus the simulations that live in it.
//!
//! Today that is one fluid solver. The world owns the CPU-side truth about solids and
//! pushes changes to whichever simulations need them, which is where future systems
//! (rigid bodies, heat, ...) will plug in.

use std::sync::Arc;

use crate::{
    brush::{self, Admission},
    gpu::Gpu,
    maps::{self, MAPS},
    scene::{Scene, Shape},
    sim::{FluidSim, Material, PPC, Particle, SPAWN_MAX, SimParams, ToolState},
};

/// What a held brush emits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// A faucet: a steady stream straight down, as wide as the brush (or narrower at low
    /// flow), at a speed set by the flow.
    Pour,
    /// Sparse droplets thrown outwards from the brush rim.
    Spray,
    /// A stream launched at `brush::JET_SPEED`, aimed `angle` degrees clockwise from down.
    Jet { angle: f32 },
}

impl Source {
    /// The force a held brush applies besides emitting: the nozzle of a stream (see
    /// `ToolState::nozzle`), nothing for spray.
    pub fn nozzle(self, at: [f32; 2], radius: f32, rate: f32) -> ToolState {
        let (direction, speed) = match self {
            Source::Pour => ([0.0, -1.0], brush::faucet_speed(rate, radius)),
            Source::Jet { angle } => (brush::jet_direction(angle), brush::JET_SPEED),
            Source::Spray => return ToolState::default(),
        };
        ToolState::nozzle(at, [direction[0] * speed, direction[1] * speed], radius)
    }
}

/// State of the brush between frames while it is held.
#[derive(Clone, Copy, Debug, Default)]
struct Emitter {
    /// Where the brush was at the last frame: each frame paints the stroke from there.
    last: Option<[f32; 2]>,
    /// Particles (spray) or stream rows (pour, jet) owed from earlier frames, below one.
    owed: f32,
}

pub struct World {
    pub scene: Scene,
    pub sim: FluidSim,
    pub map: usize,
    rng: u32,
    emitter: Emitter,
    /// Signed distance to the solids at the grid nodes, as rasterised for the solver.
    sdf: Vec<f32>,
    /// `--set cell_size=...` was given: maps do not change the physical scale.
    pub scale_locked: bool,
}

impl World {
    pub fn new(gpu: Arc<Gpu>, grid: [u32; 2], map: usize) -> World {
        let sim = FluidSim::new(gpu, grid, grid[0] * grid[1] * PPC);
        let mut world = World::assemble(Scene::new(grid), sim);
        world.load_map(map);
        world
    }

    /// A world with hand-built contents, for tests and benchmarks.
    pub fn custom(gpu: Arc<Gpu>, scene: Scene, particles: Vec<Particle>) -> World {
        let capacity = scene.grid[0] * scene.grid[1] * PPC;
        World::with_capacity(gpu, scene, particles, capacity)
    }

    /// Like `custom`, with an explicit limit on the number of particles.
    pub fn with_capacity(gpu: Arc<Gpu>, scene: Scene, particles: Vec<Particle>, capacity: u32) -> World {
        let mut sim = FluidSim::new(gpu, scene.grid, capacity);
        sim.reset(particles);
        let mut world = World::assemble(scene, sim);
        world.refresh_solids();
        world
    }

    fn assemble(scene: Scene, sim: FluidSim) -> World {
        World { scene, sim, map: 0, rng: 0x1234_5678, emitter: Emitter::default(), sdf: Vec::new(), scale_locked: false }
    }

    /// Rasterises the scene's solids for the solver (and for the brushes).
    fn refresh_solids(&mut self) {
        let field = self.scene.rasterize();
        self.sdf.clone_from(&field.sdf);
        self.sim.set_solids(field);
    }

    /// Distance to the nearest solid at `p` (cells), interpolated from the rasterised field.
    fn solid_distance(&self, p: [f32; 2]) -> f32 {
        let nw = self.sim.grid[0] as usize + 1;
        let x = p[0].clamp(0.0, self.sim.grid[0] as f32 - 1e-3);
        let y = p[1].clamp(0.0, self.sim.grid[1] as f32 - 1e-3);
        let (i, j) = (x as usize, y as usize);
        let (fx, fy) = (x - i as f32, y - j as f32);
        let d = |i: usize, j: usize| self.sdf[j * nw + i];
        let bottom = d(i, j) * (1.0 - fx) + d(i + 1, j) * fx;
        let top = d(i, j + 1) * (1.0 - fx) + d(i + 1, j + 1) * fx;
        bottom * (1.0 - fy) + top * fy
    }

    /// Loads (or reloads) a built-in map, replacing all solids and fluid.
    pub fn load_map(&mut self, map: usize) {
        self.map = map.min(MAPS.len() - 1);
        self.apply_map_scale();
        let (scene, particles) = maps::build(&MAPS[self.map], self.sim.grid);
        self.scene = scene;
        self.refresh_solids();
        self.sim.reset(particles);
        self.end_stroke();
    }

    /// Sets the cell size the current map is built for (see `MapDef::width_m`), unless the
    /// scale was fixed from the command line.
    pub fn apply_map_scale(&mut self) {
        if !self.scale_locked {
            self.sim.params.cell_size =
                MAPS[self.map].cell_size(self.sim.grid[0]).unwrap_or(SimParams::default().cell_size);
        }
    }

    pub fn add_shape(&mut self, shape: Shape) {
        self.scene.shapes.push(shape);
        self.refresh_solids();
    }

    /// Removes the topmost shape under `pos`. Returns whether anything was removed.
    pub fn remove_shape_at(&mut self, pos: [f32; 2]) -> bool {
        match self.scene.pick(pos) {
            Some(i) => {
                self.scene.shapes.remove(i);
                self.refresh_solids();
                true
            }
            None => false,
        }
    }

    /// Runs a held brush for `seconds` of simulated time: the stroke goes from where the
    /// brush was at the last call to `at`. Returns the number of particles emitted. With
    /// `seconds` = 0 (paused) the brush only moves. `rate` is in particles per second.
    pub fn emit(&mut self, source: Source, material: Material, at: [f32; 2], radius: f32, rate: f32, seconds: f32) -> usize {
        let from = self.emitter.last.replace(at).unwrap_or(at);
        if seconds <= 0.0 {
            return 0;
        }
        match source {
            Source::Pour => {
                let speed = brush::faucet_speed(rate, radius);
                self.stream(material, from, at, radius, rate, [0.0, -1.0], speed, seconds)
            }
            Source::Jet { angle } => {
                self.stream(material, from, at, radius, rate, brush::jet_direction(angle), brush::JET_SPEED, seconds)
            }
            Source::Spray => {
                self.emitter.owed += rate * seconds * brush::SPRAY_SHARE;
                // At most one brush-full per frame; what does not fit is dropped rather than
                // released as a backlog after a slow frame or a blocked stroke.
                let full = (std::f32::consts::PI * radius * radius * PPC as f32) as usize;
                let count = (self.emitter.owed as usize).min(full);
                self.emitter.owed = self.emitter.owed.fract();
                self.paint(material, from, at, radius, count, true)
            }
        }
    }

    /// Lets go of the brush: the next `emit` starts a new stroke.
    pub fn end_stroke(&mut self) {
        self.emitter = Emitter::default();
    }

    /// Fills the brush disk at `at` with liquid at rest, wherever there is none yet.
    pub fn stamp(&mut self, material: Material, at: [f32; 2], radius: f32) -> usize {
        self.paint(material, at, at, radius, usize::MAX, false)
    }

    /// Places up to `count` particles on the rest-density lattice (or the sparse spray
    /// lattice) inside the stroke from `from` to `to`, in empty cells only.
    fn paint(&mut self, material: Material, from: [f32; 2], to: [f32; 2], radius: f32, count: usize, spray: bool) -> usize {
        let count = count.min(SPAWN_MAX);
        if count == 0 {
            return 0;
        }
        let spacing = if spray { brush::SPRAY_SPACING } else { 1.0 / (PPC as f32).sqrt() };
        let mut candidates = brush::stroke(from, to, radius, self.sim.grid, spacing);
        // A shuffled lattice: a low rate still covers the whole footprint, and no two
        // particles land closer than the lattice spacing.
        for i in (1..candidates.len()).rev() {
            let j = ((self.rand() * (i + 1) as f32) as usize).min(i);
            candidates.swap(i, j);
        }
        let mut admission = self.admission(from, to, radius + 1.0, false);
        let mut particles = Vec::with_capacity(count.min(candidates.len()));
        for pos in candidates {
            if self.solid_distance(pos) <= 0.4 {
                continue;
            }
            let vel = if spray {
                match brush::spray_velocity(pos, from, to, radius) {
                    Some(v) => v,
                    None => continue,
                }
            } else {
                [0.0; 2]
            };
            if !admission.admit(pos) {
                continue;
            }
            particles.push(Particle::new(material, pos, vel));
            if particles.len() == count {
                break;
            }
        }
        self.sim.spawn(&particles)
    }

    /// A stream (the faucet, the jet): rows of rest-density liquid across the nozzle,
    /// launched along `direction` at `speed`. Each row is placed as far along the stream as
    /// it has travelled since it was due, and the nozzle sweeps from `from` to `to` over the
    /// frame, so the stream comes out continuous whatever the frame rate and however the
    /// brush moves. `rate` (particles/s) sets the width, up to the brush's diameter.
    #[allow(clippy::too_many_arguments)]
    fn stream(&mut self, material: Material, from: [f32; 2], to: [f32; 2], radius: f32, rate: f32, direction: [f32; 2], speed: f32, seconds: f32) -> usize {
        let spacing = 1.0 / (PPC as f32).sqrt();
        let width = (rate / (PPC as f32 * speed)).min(2.0 * radius);
        let cols = (width / spacing).round().max(1.0) as usize;
        let travel = speed * seconds / spacing;
        self.emitter.owed += travel;
        let rows = (self.emitter.owed as usize).min(SPAWN_MAX / cols);
        self.emitter.owed = self.emitter.owed.fract();
        let side = [-direction[1], direction[0]];
        let velocity = [direction[0] * speed, direction[1] * speed];
        let reach = speed * seconds + radius + 1.0;
        let mut admission = self.admission(from, to, reach, true);
        let mut particles = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let age = (rows - 1 - row) as f32 + self.emitter.owed;
            // Older rows left the nozzle earlier in the frame, from earlier along the sweep.
            let t = (1.0 - age / travel.max(1e-6)).clamp(0.0, 1.0);
            let centre = [
                from[0] + (to[0] - from[0]) * t + direction[0] * age * spacing,
                from[1] + (to[1] - from[1]) * t + direction[1] * age * spacing,
            ];
            for col in 0..cols {
                // A little jitter keeps the lattice from showing as stripes in the stream.
                let offset = (col as f32 + 0.5 - cols as f32 * 0.5 + 0.3 * (self.rand() - 0.5)) * spacing;
                let along = 0.3 * (self.rand() - 0.5) * spacing;
                let pos = [
                    centre[0] + side[0] * offset + direction[0] * along,
                    centre[1] + side[1] * offset + direction[1] * along,
                ];
                if self.solid_distance(pos) > 0.4 && admission.admit(pos) {
                    particles.push(Particle::new(material, pos, velocity));
                }
            }
        }
        self.sim.spawn(&particles)
    }

    /// Which cells within `margin` of the stroke can take new liquid (see `Admission`);
    /// `top_up` lets partly filled cells fill to rest density.
    fn admission(&self, from: [f32; 2], to: [f32; 2], margin: f32, top_up: bool) -> Admission {
        let (lo, hi) = Admission::window(from, to, margin);
        let lo = [lo[0].max(0), lo[1].max(0)];
        let hi = [hi[0].clamp(lo[0], self.sim.grid[0] as i32), hi[1].clamp(lo[1], self.sim.grid[1] as i32)];
        Admission::new(lo, [hi[0] - lo[0], hi[1] - lo[1]], &self.sim.cell_counts(lo, hi), PPC, top_up)
    }

    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }
}
