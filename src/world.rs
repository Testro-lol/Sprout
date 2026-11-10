//! The sandbox world: static scene geometry plus the simulations that live in it.
//!
//! Today that is one fluid solver. The world owns the CPU-side truth about solids and
//! pushes changes to whichever simulations need them, which is where future systems
//! (rigid bodies, heat, ...) will plug in.

use std::sync::Arc;

use crate::{
    gpu::Gpu,
    maps::{self, MAPS},
    scene::{Scene, Shape},
    sim::{FluidSim, Material, PPC, Particle, SPAWN_MAX},
};

/// A pouring nozzle: `2 * radius` cells wide, emitting `rate` particles per second straight
/// down plus half of its own motion `drag`, while it moves from `from` to `to`.
#[derive(Clone, Copy, Debug)]
pub struct Nozzle {
    pub material: Material,
    pub from: [f32; 2],
    pub to: [f32; 2],
    pub radius: f32,
    pub rate: f32,
    pub drag: [f32; 2],
}

impl Nozzle {
    /// A nozzle that stays where it is.
    pub fn fixed(material: Material, pos: [f32; 2], radius: f32, rate: f32) -> Nozzle {
        Nozzle { material, from: pos, to: pos, radius, rate, drag: [0.0, 0.0] }
    }
}

pub struct World {
    pub scene: Scene,
    pub sim: FluidSim,
    pub map: usize,
    rng: u32,
    /// Fraction of a row of particles the nozzle still owes from earlier frames.
    pour_rows: f32,
}

impl World {
    pub fn new(gpu: Arc<Gpu>, grid: [u32; 2], map: usize) -> World {
        let sim = FluidSim::new(gpu, grid, grid[0] * grid[1] * PPC);
        let mut world = World { scene: Scene::new(grid), sim, map: 0, rng: 0x1234_5678, pour_rows: 0.0 };
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
        sim.set_solids(scene.rasterize());
        sim.reset(particles);
        World { scene, sim, map: 0, rng: 0x1234_5678, pour_rows: 0.0 }
    }

    /// Loads (or reloads) a built-in map, replacing all solids and fluid.
    pub fn load_map(&mut self, map: usize) {
        self.map = map.min(MAPS.len() - 1);
        let (scene, particles) = maps::build(&MAPS[self.map], self.sim.grid);
        self.scene = scene;
        self.sim.set_solids(self.scene.rasterize());
        self.sim.reset(particles);
    }

    pub fn add_shape(&mut self, shape: Shape) {
        self.scene.shapes.push(shape);
        self.sim.set_solids(self.scene.rasterize());
    }

    /// Removes the topmost shape under `pos`. Returns whether anything was removed.
    pub fn remove_shape_at(&mut self, pos: [f32; 2]) -> bool {
        match self.scene.pick(pos) {
            Some(i) => {
                self.scene.shapes.remove(i);
                self.sim.set_solids(self.scene.rasterize());
                true
            }
            None => false,
        }
    }

    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Runs a pouring nozzle for `seconds` of simulated time.
    ///
    /// The stream leaves as a slab of rest-density liquid: row after row of particles, each
    /// placed as far below the nozzle as it has fallen since its turn came. A low rate
    /// slows the stream down to a minimum speed and then narrows it. Returns the number of
    /// particles emitted.
    pub fn pour(&mut self, nozzle: &Nozzle, seconds: f32) -> usize {
        let Nozzle { material, from, to, radius, rate, drag } = *nozzle;
        // Slower than this and gravity stretches the stream to many times its length on
        // the way down, which tears it into ribbons.
        const MIN_SPEED: f32 = 150.0;
        const MAX_SPEED: f32 = 400.0;
        let spacing = 1.0 / (PPC as f32).sqrt();
        let per_area = PPC as f32;
        let speed = (rate / (per_area * 2.0 * radius)).clamp(MIN_SPEED, MAX_SPEED);
        let cols = ((rate / (per_area * speed)).min(2.0 * radius) / spacing).round().max(1.0) as usize;
        self.pour_rows += speed * seconds / spacing;
        let rows = (self.pour_rows as usize).min(SPAWN_MAX / cols);
        self.pour_rows = self.pour_rows.fract();

        let vel = [drag[0] * 0.5, drag[1] * 0.5 - speed];
        let mut out = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            // Rows are emitted at even intervals; the first of this batch is the oldest.
            let age = (rows - 1 - row) as f32 + self.pour_rows;
            let t = 1.0 - age / (rows as f32 + self.pour_rows).max(1.0);
            let centre = [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t - age * spacing];
            for col in 0..cols {
                let x = (col as f32 + 0.5 - cols as f32 * 0.5) * spacing;
                let jitter = [(self.rand() - 0.5) * 0.3 * spacing, (self.rand() - 0.5) * 0.3 * spacing];
                let pos = [centre[0] + x + jitter[0], centre[1] + jitter[1]];
                if self.scene.sdf(pos) > 0.4 {
                    out.push(Particle::new(material, pos, vel));
                }
            }
        }
        self.sim.spawn(&out)
    }
}
