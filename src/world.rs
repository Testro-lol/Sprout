//! The sandbox world: static scene geometry plus the simulations that live in it.
//!
//! Today that is one fluid solver. The world owns the CPU-side truth about solids and
//! pushes changes to whichever simulations need them, which is where future systems
//! (rigid bodies, heat, ...) will plug in.

use std::sync::Arc;

use crate::{
    brush::{self, Admission, Spout, Stream},
    gpu::Gpu,
    maps::{self, MAPS},
    scene::{Scene, Shape},
    sim::{Current, FluidSim, MAX_CURRENTS, MAX_MAGNETS, Magnet, Material, PPC, Particle, SPAWN_MAX, SimParams, ToolState},
};

/// Most faucets in the world at once (each reads back the cells around its spout every
/// frame, about 0.3 ms).
pub const MAX_FAUCETS: usize = 6;
/// Steps of undo kept.
const UNDO_DEPTH: usize = 64;

/// A faucet: a placed source that keeps pouring its liquid while the simulation runs.
#[derive(Clone, Copy, Debug)]
pub struct Faucet {
    pub pos: [f32; 2],
    pub radius: f32,
    /// Heading, degrees clockwise from straight down.
    pub angle: f32,
    /// Particles per second.
    pub rate: f32,
    pub material: Material,
    stream: Stream,
}

impl Faucet {
    pub fn new(pos: [f32; 2], radius: f32, angle: f32, rate: f32, material: Material) -> Faucet {
        Faucet { pos, radius, angle, rate, material, stream: Stream::default() }
    }

    pub fn spout(&self) -> Spout {
        Spout::faucet(self.angle, self.rate, self.radius)
    }
}

/// Something placed in the world that the Move and Remove tools can pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Shape(usize),
    Magnet(usize),
    Fan(usize),
    Drain(usize),
    Faucet(usize),
}

/// Everything the build tools place: what undo and redo step through. The liquid is not
/// part of it.
#[derive(Clone, Debug, Default)]
struct Layout {
    shapes: Vec<Shape>,
    temps: Vec<Option<f32>>,
    magnets: Vec<Magnet>,
    fans: Vec<Current>,
    drains: Vec<Current>,
    faucets: Vec<Faucet>,
}

/// What the Pipette finds under the cursor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Row of the material table.
    pub row: usize,
    /// °C.
    pub temp: f32,
    pub frozen: bool,
    /// cells/s.
    pub speed: f32,
}

/// What a held brush emits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// A stream straight down out of the round brush, at a speed set by the flow.
    Pour,
    /// Sparse droplets thrown outwards from the brush rim.
    Spray,
    /// A stream launched at `brush::JET_SPEED`, aimed `angle` degrees clockwise from down.
    Jet { angle: f32 },
    /// Sparse drops falling from a wide cloud around the brush.
    Rain,
}

impl Source {
    /// The outlet of the streams (pour and jet); spray has none.
    pub fn spout(self, radius: f32, rate: f32) -> Option<Spout> {
        match self {
            Source::Pour => Some(Spout::pour(rate, radius)),
            Source::Jet { angle } => Some(Spout::jet(angle, rate, radius)),
            Source::Spray | Source::Rain => None,
        }
    }

    /// The force a held brush applies besides emitting: a stream's nozzle (see
    /// `ToolState::nozzle`), nothing for spray.
    pub fn nozzle(self, at: [f32; 2], radius: f32, rate: f32) -> ToolState {
        match self.spout(radius, rate) {
            Some(spout) => ToolState::nozzle(at, spout.velocity(), 0.5 * spout.width + 1.0),
            None => ToolState::default(),
        }
    }
}

/// State of the brush between frames while it is held.
#[derive(Clone, Copy, Debug, Default)]
struct Emitter {
    /// Where the brush was at the last frame: each frame paints the stroke from there.
    last: Option<[f32; 2]>,
    /// Spray particles owed from earlier frames, below one.
    owed: f32,
    /// The stream of a pour or a jet.
    stream: Stream,
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
    /// Custom liquids (their rows of the material table follow the built-ins).
    pub library: crate::library::Library,
    /// Ids of the custom liquids as last written to (or read from) the library file.
    pub saved_ids: Vec<String>,
    /// Placed currents; with the drains and the held current they fill the sim's zones.
    fans: Vec<Current>,
    drains: Vec<Current>,
    faucets: Vec<Faucet>,
    /// The Wind tool's current while its button is down.
    held: Option<Current>,
    undo: Vec<Layout>,
    redo: Vec<Layout>,
    /// The scene has changed since it was last rasterised (`edit_shape`, `flush_solids`),
    /// from this shape on (for the overlay: `unrasterised`).
    stale_from: Option<usize>,
    /// How long the last rasterisation took, s.
    raster_cost: f32,
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
        World {
            scene,
            sim,
            map: 0,
            rng: 0x1234_5678,
            emitter: Emitter::default(),
            sdf: Vec::new(),
            scale_locked: false,
            library: Default::default(),
            saved_ids: Vec::new(),
            fans: Vec::new(),
            drains: Vec::new(),
            faucets: Vec::new(),
            held: None,
            undo: Vec::new(),
            redo: Vec::new(),
            stale_from: None,
            raster_cost: 0.0,
        }
    }

    /// Rasterises the scene's solids for the solver (and for the brushes).
    fn refresh_solids(&mut self) {
        let start = std::time::Instant::now();
        let field = self.scene.rasterize();
        self.raster_cost = start.elapsed().as_secs_f32();
        self.stale_from = None;
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
        let (scene, particles, extras) = maps::build_full(&MAPS[self.map], self.sim.grid);
        self.sim.params.ambient_temp = extras.ambient.unwrap_or(crate::sim::T_REF);
        self.sim.set_magnets(&extras.magnets);
        self.scene = scene;
        self.refresh_solids();
        self.sim.reset(particles);
        self.fans.clear();
        self.drains.clear();
        self.faucets.clear();
        self.held = None;
        self.sync_zones();
        self.undo.clear();
        self.redo.clear();
        self.end_stroke();
    }

    /// Removes all liquid; solids and devices stay.
    pub fn clear_liquid(&mut self) {
        self.sim.reset(Vec::new());
        self.end_stroke();
    }

    // ---- undo ------------------------------------------------------------------------

    fn layout(&self) -> Layout {
        Layout {
            shapes: self.scene.shapes.clone(),
            temps: self.scene.temps.clone(),
            magnets: self.sim.magnets().to_vec(),
            fans: self.fans.clone(),
            drains: self.drains.clone(),
            faucets: self.faucets.clone(),
        }
    }

    fn restore(&mut self, l: Layout) {
        self.scene.shapes = l.shapes;
        self.scene.temps = l.temps;
        self.sim.set_magnets(&l.magnets);
        self.fans = l.fans;
        self.drains = l.drains;
        self.faucets = l.faucets;
        self.sync_zones();
        self.refresh_solids();
    }

    /// Remembers the solids and devices as they are now: call before an edit, so that
    /// `undo` comes back here. A new edit drops what was undone.
    pub fn checkpoint(&mut self) {
        self.undo.push(self.layout());
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Forgets the last checkpoint (the edit it was taken for did not happen).
    pub fn drop_checkpoint(&mut self) {
        self.undo.pop();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Takes back the last edit of the solids and devices (never the liquid).
    pub fn undo(&mut self) -> bool {
        let Some(l) = self.undo.pop() else { return false };
        self.redo.push(self.layout());
        self.restore(l);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(l) = self.redo.pop() else { return false };
        self.undo.push(self.layout());
        self.restore(l);
        true
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
        self.add_shape_at(shape, None);
    }

    /// Adds a solid held at `temp` °C (a hot plate, a block of ice), or an insulating one.
    pub fn add_shape_at(&mut self, shape: Shape, temp: Option<f32>) {
        self.scene.temps.resize(self.scene.shapes.len(), None);
        self.scene.shapes.push(shape);
        self.scene.temps.push(temp);
        self.refresh_solids();
    }

    /// Adds a solid to the scene without rasterising it yet: for the strokes of the Draw
    /// tool, which add many in a row. The simulation sees it at the next `flush_solids`.
    pub fn add_shape_later(&mut self, shape: Shape, temp: Option<f32>) {
        self.scene.temps.resize(self.scene.shapes.len(), None);
        self.stale_from = Some(self.stale_from.unwrap_or(self.scene.shapes.len()));
        self.scene.shapes.push(shape);
        self.scene.temps.push(temp);
    }

    /// Rasterises the scene if it has changed since the last time. Returns whether it had.
    pub fn flush_solids(&mut self) -> bool {
        let stale = self.stale_from.is_some();
        if stale {
            self.refresh_solids();
        }
        stale
    }

    /// Solids added but not rasterised yet (the overlay draws them until they are).
    pub fn unrasterised(&self) -> &[Shape] {
        match self.stale_from {
            Some(k) => &self.scene.shapes[k.min(self.scene.shapes.len())..],
            None => &[],
        }
    }

    /// How long the last rasterisation of the solids took, s (0.01 at 512 x 288, 0.05 at
    /// 1024 x 576): what sets how often a moving solid can be handed to the simulation.
    pub fn raster_cost(&self) -> f32 {
        self.raster_cost
    }

    // ---- devices ---------------------------------------------------------------------

    /// Hands the sim its zones: the fans, the drains and the held current.
    fn sync_zones(&mut self) {
        let zones: Vec<Current> = self.fans.iter().chain(&self.drains).chain(&self.held).copied().collect();
        self.sim.set_currents(&zones);
    }

    pub fn fans(&self) -> &[Current] {
        &self.fans
    }

    pub fn drains(&self) -> &[Current] {
        &self.drains
    }

    pub fn faucets(&self) -> &[Faucet] {
        &self.faucets
    }

    pub fn held_current(&self) -> Option<Current> {
        self.held
    }

    /// Places a drain, unless the zones are used up (one is kept for the held current).
    pub fn add_drain(&mut self, drain: Current) -> bool {
        if self.fans.len() + self.drains.len() + 1 >= MAX_CURRENTS {
            return false;
        }
        self.drains.push(drain);
        self.sync_zones();
        true
    }

    /// Places a faucet, unless the world already holds `MAX_FAUCETS`.
    pub fn add_faucet(&mut self, faucet: Faucet) -> bool {
        if self.faucets.len() >= MAX_FAUCETS {
            return false;
        }
        self.faucets.push(faucet);
        true
    }

    /// Runs the faucets for `seconds` of simulated time. Returns the particles emitted.
    pub fn run_faucets(&mut self, seconds: f32) -> usize {
        if seconds <= 0.0 {
            return 0;
        }
        let mut emitted = 0;
        for k in 0..self.faucets.len() {
            let f = self.faucets[k];
            let mut stream = f.stream;
            emitted += self.stream(&mut stream, f.material, &f.spout(), f.pos, f.pos, seconds);
            self.faucets[k].stream = stream;
        }
        emitted
    }

    /// The topmost thing under `pos`: devices lie over solids.
    pub fn pick(&self, pos: [f32; 2]) -> Option<Pick> {
        let within = |c: [f32; 2], r: f32| (c[0] - pos[0]).hypot(c[1] - pos[1]) < r.max(3.0);
        if let Some(i) = self.faucets.iter().rposition(|f| within(f.pos, f.radius)) {
            return Some(Pick::Faucet(i));
        }
        if let Some(i) = self.drains.iter().rposition(|d| within(d.pos, d.radius)) {
            return Some(Pick::Drain(i));
        }
        if let Some(i) = self.sim.magnets().iter().rposition(|m| within(m.pos, m.radius)) {
            return Some(Pick::Magnet(i));
        }
        if let Some(i) = self.fans.iter().rposition(|c| within(c.pos, c.radius)) {
            return Some(Pick::Fan(i));
        }
        self.scene.pick(pos).map(Pick::Shape)
    }

    /// Where the picked thing is (the point it is moved by).
    pub fn pick_pos(&self, pick: Pick) -> Option<[f32; 2]> {
        match pick {
            Pick::Shape(i) => self.scene.shapes.get(i).map(|s| s.center()),
            Pick::Magnet(i) => self.sim.magnets().get(i).map(|m| m.pos),
            Pick::Fan(i) => self.fans.get(i).map(|c| c.pos),
            Pick::Drain(i) => self.drains.get(i).map(|c| c.pos),
            Pick::Faucet(i) => self.faucets.get(i).map(|f| f.pos),
        }
    }

    /// Moves the picked thing by `d` cells.
    pub fn move_pick(&mut self, pick: Pick, d: [f32; 2]) {
        let shift = |p: &mut [f32; 2]| *p = [p[0] + d[0], p[1] + d[1]];
        match pick {
            // Rasterised at the next `flush_solids`: the caller paces that.
            Pick::Shape(i) => {
                if let Some(s) = self.scene.shapes.get_mut(i) {
                    s.translate(d);
                    self.stale_from = Some(self.stale_from.unwrap_or(self.scene.shapes.len()));
                }
            }
            Pick::Magnet(i) => {
                let mut magnets = self.sim.magnets().to_vec();
                if let Some(m) = magnets.get_mut(i) {
                    shift(&mut m.pos);
                    self.sim.set_magnets(&magnets);
                }
            }
            Pick::Fan(i) => {
                if let Some(c) = self.fans.get_mut(i) {
                    shift(&mut c.pos);
                }
                self.sync_zones();
            }
            Pick::Drain(i) => {
                if let Some(c) = self.drains.get_mut(i) {
                    shift(&mut c.pos);
                }
                self.sync_zones();
            }
            Pick::Faucet(i) => {
                if let Some(f) = self.faucets.get_mut(i) {
                    shift(&mut f.pos);
                }
            }
        }
    }

    /// Removes the picked thing.
    pub fn remove_pick(&mut self, pick: Pick) {
        match pick {
            Pick::Shape(i) => {
                if i < self.scene.shapes.len() {
                    self.scene.temps.resize(self.scene.shapes.len(), None);
                    self.scene.shapes.remove(i);
                    self.scene.temps.remove(i);
                    self.refresh_solids();
                }
            }
            Pick::Magnet(i) => {
                let mut magnets = self.sim.magnets().to_vec();
                if i < magnets.len() {
                    magnets.remove(i);
                    self.sim.set_magnets(&magnets);
                }
            }
            Pick::Fan(i) => {
                if i < self.fans.len() {
                    self.fans.remove(i);
                }
                self.sync_zones();
            }
            Pick::Drain(i) => {
                if i < self.drains.len() {
                    self.drains.remove(i);
                }
                self.sync_zones();
            }
            Pick::Faucet(i) => {
                if i < self.faucets.len() {
                    self.faucets.remove(i);
                }
            }
        }
    }

    /// The liquid nearest `pos`, within `radius` cells. Reads every particle back: for a
    /// click, not for every frame.
    pub fn sample(&self, pos: [f32; 2], radius: f32) -> Option<Sample> {
        let ps = self.sim.download_particles();
        let nearest = ps
            .iter()
            .filter(|p| p.material().is_some())
            .map(|p| ((p.pos[0] - pos[0]).powi(2) + (p.pos[1] - pos[1]).powi(2), p))
            .filter(|(d2, _)| *d2 <= radius * radius)
            .min_by(|a, b| a.0.total_cmp(&b.0))?
            .1;
        let (temp, frozen) = self.sim.params.temperature(nearest);
        Some(Sample { row: nearest.material()?.index(), temp, frozen: frozen > 0.5, speed: nearest.vel[0].hypot(nearest.vel[1]) })
    }

    /// The currents placed in the world (see `Current`).
    pub fn currents(&self) -> &[Current] {
        self.sim.currents()
    }

    /// Places a current, unless the world already holds `MAX_CURRENTS`.
    /// Uses this library of custom liquids: their definitions go into the material table.
    pub fn set_library(&mut self, library: crate::library::Library) {
        library.install(&mut self.sim.params);
        self.saved_ids = library.custom.iter().map(|c| c.id.clone()).collect();
        self.library = library;
    }

    /// Places a magnet, unless the world already holds `MAX_MAGNETS`.
    pub fn add_magnet(&mut self, magnet: Magnet) -> bool {
        let mut magnets = self.sim.magnets().to_vec();
        if magnets.len() >= MAX_MAGNETS {
            return false;
        }
        magnets.push(magnet);
        self.sim.set_magnets(&magnets);
        true
    }

    /// Places a fan (a current that stays), unless the zones are used up (one is kept for
    /// the held current).
    pub fn add_current(&mut self, current: Current) -> bool {
        if self.fans.len() + self.drains.len() + 1 >= MAX_CURRENTS {
            return false;
        }
        self.fans.push(current);
        self.sync_zones();
        true
    }

    /// The Wind tool held at the cursor: a current of its own while held, none otherwise.
    pub fn hold_current(&mut self, current: Option<Current>) {
        if self.held != current {
            self.held = current;
            self.sync_zones();
        }
    }

    /// Removes the topmost fan under `pos`. Returns whether anything was removed.
    pub fn remove_current_at(&mut self, pos: [f32; 2]) -> bool {
        match self.fans.iter().rposition(|c| (pos[0] - c.pos[0]).hypot(pos[1] - c.pos[1]) <= c.radius) {
            Some(i) => {
                self.fans.remove(i);
                self.sync_zones();
                true
            }
            None => false,
        }
    }

    /// Removes the topmost shape under `pos`. Returns whether anything was removed.
    pub fn remove_shape_at(&mut self, pos: [f32; 2]) -> bool {
        match self.scene.pick(pos) {
            Some(i) => {
                self.remove_pick(Pick::Shape(i));
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
        if let Some(spout) = source.spout(radius, rate) {
            let mut stream = self.emitter.stream;
            let n = self.stream(&mut stream, material, &spout, from, at, seconds);
            self.emitter.stream = stream;
            return n;
        }
        if source == Source::Rain {
            return self.rain(material, at, radius, rate, seconds);
        }
        self.emitter.owed += rate * seconds * brush::SPRAY_SHARE;
        // At most one brush-full per frame; what does not fit is dropped rather than
        // released as a backlog after a slow frame or a blocked stroke.
        let full = (std::f32::consts::PI * radius * radius * PPC as f32) as usize;
        let count = (self.emitter.owed as usize).min(full);
        self.emitter.owed = self.emitter.owed.fract();
        self.paint(material, from, at, radius, count, true)
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
        for mut pos in candidates {
            if spray {
                // Off the lattice a little, so that the droplets do not leave in rows.
                pos = [pos[0] + (self.rand() - 0.5) * 0.6 * spacing, pos[1] + (self.rand() - 0.5) * 0.6 * spacing];
            }
            if self.solid_distance(pos) <= 0.4 {
                continue;
            }
            let vel = if spray {
                let jitter = [self.rand(), self.rand()];
                match brush::spray_velocity(pos, from, to, radius, jitter) {
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

    /// Rain: sparse drops falling at `brush::RAIN_SPEED` from a cloud `brush::RAIN_SPAN`
    /// brush radii to either side of `at`, at `brush::RAIN_SHARE` of the flow.
    fn rain(&mut self, material: Material, at: [f32; 2], radius: f32, rate: f32, seconds: f32) -> usize {
        self.emitter.owed += rate * seconds * brush::RAIN_SHARE;
        let half = radius * brush::RAIN_SPAN;
        // No more than a sparse sheet per frame; a backlog is dropped, as for spray.
        let count = (self.emitter.owed as usize).min((2.0 * half * 2.0) as usize).min(SPAWN_MAX);
        self.emitter.owed = self.emitter.owed.fract();
        if count == 0 {
            return 0;
        }
        let (from, to) = ([at[0] - half, at[1]], [at[0] + half, at[1]]);
        let mut admission = self.admission(from, to, 3.0, false);
        let mut particles = Vec::with_capacity(count);
        for _ in 0..count {
            let pos = [at[0] + (self.rand() * 2.0 - 1.0) * half, at[1] + (self.rand() * 2.0 - 1.0) * 2.0];
            if self.solid_distance(pos) <= 0.4 || !admission.admit(pos) {
                continue;
            }
            let speed = brush::RAIN_SPEED * (0.8 + 0.4 * self.rand());
            particles.push(Particle::new(material, pos, [(self.rand() - 0.5) * 12.0, -speed]));
        }
        self.sim.spawn(&particles)
    }

    /// A stream (the pour, the jet): this frame's particles of `brush::Stream`, launched at
    /// the spout's velocity, except where a solid or liquid that cannot get away blocks them.
    fn stream(&mut self, stream: &mut Stream, material: Material, spout: &Spout, from: [f32; 2], to: [f32; 2], seconds: f32) -> usize {
        let positions = stream.advance(spout, from, to, seconds, SPAWN_MAX);
        let reach = spout.speed * seconds + spout.width + 1.0;
        let mut admission = self.admission(from, to, reach, true);
        let velocity = spout.velocity();
        let particles: Vec<Particle> = positions
            .into_iter()
            .filter(|&pos| self.solid_distance(pos) > 0.4 && admission.admit(pos))
            .map(|pos| Particle::new(material, pos, velocity))
            .collect();
        self.sim.spawn(&particles)
    }

    /// Which cells within `margin` of the stroke can take new liquid, for painted liquid or
    /// for a `stream` (see `Admission`).
    fn admission(&self, from: [f32; 2], to: [f32; 2], margin: f32, stream: bool) -> Admission {
        let (lo, hi) = Admission::window(from, to, margin);
        // (A stroke wholly past the edge of the grid has an empty window, not an inverted one.)
        let (gw, gh) = (self.sim.grid[0] as i32, self.sim.grid[1] as i32);
        let lo = [lo[0].clamp(0, gw), lo[1].clamp(0, gh)];
        let hi = [hi[0].clamp(lo[0], gw), hi[1].clamp(lo[1], gh)];
        Admission::new(lo, [hi[0] - lo[0], hi[1] - lo[1]], &self.sim.cell_counts(lo, hi), PPC, stream)
    }

    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }
}
