//! Robustness scenarios (`sprout validate maps | liquids | abuse | extremes | render`): no
//! known answer, but bounds that any healthy run keeps. Every built-in map and every liquid
//! is run for a few seconds, the tools are driven into walls, corners and the edge of the
//! grid, parameters are pushed to the ends of their ranges, and the renderer is checked for
//! black specks, flicker and light that hot liquid takes away instead of adding.
//!
//! What is checked everywhere ("health"):
//! - every particle position, velocity and temperature is finite;
//! - nothing sits more than a cell inside a solid or outside the grid;
//! - no cell holds a clump (`CROWD_MAX` particles; rest is 4, the fixed-point sums wrap
//!   near 2000);
//! - particles are conserved, apart from liquid that boiled away and the residue it left;
//! - no particle is faster than the speed cap allows;
//! - the pressure solve converges (at most `UNCONVERGED_MAX` of the substeps stop at their
//!   recorded iteration limit).

use std::sync::Arc;

use crate::{
    gpu::Gpu,
    headless::Offscreen,
    maps::{MAPS, fill_rect},
    render::View,
    scene::{Scene, Shape, WALL},
    sim::{BUILTINS, Current, Magnet, Material, SimParams, ToolState, pickable},
    validate::{Report, rms_speed},
    world::{Source, World},
};

const GRID: [u32; 2] = [512, 288];
/// Most particles any cell may hold (rest: 4). The refilled-pool check allows 12; a stream
/// hitting the floor packs 20 to 50 into the thin sheet it spreads into, for a moment. A
/// clump that keeps growing ends in the thousands (the fixed-point sums wrap near 2000).
const CROWD_MAX: u32 = 64;
/// Share of substeps whose pressure solve may stop at its recorded iteration limit.
const UNCONVERGED_MAX: f64 = 0.03;

/// What a run looked like at one moment.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Health {
    pub particles: usize,
    pub non_finite: usize,
    pub in_solid: usize,
    pub outside: usize,
    pub crowd: u32,
    /// The fullest cell.
    pub crowd_at: [u32; 2],
    pub fastest: f32,
    pub boiled: i64,
    pub unconverged: u64,
    pub substeps: u64,
}

impl Health {
    pub(crate) fn of(w: &World) -> Health {
        let ps = w.sim.download_particles();
        let grid = w.sim.grid;
        let (gw, gh) = (grid[0] as usize, grid[1] as usize);
        let mut count = vec![0u32; gw * gh];
        let mut h = Health { particles: ps.len(), ..Default::default() };
        for p in &ps {
            let finite = p.pos.iter().chain(&p.vel).all(|v| v.is_finite()) && p.t.is_finite() && p.s.is_finite();
            if !finite {
                h.non_finite += 1;
                continue;
            }
            if p.pos[0] < 0.0 || p.pos[1] < 0.0 || p.pos[0] > grid[0] as f32 || p.pos[1] > grid[1] as f32 {
                h.outside += 1;
                continue;
            }
            if w.scene.sdf(p.pos) < -1.0 {
                h.in_solid += 1;
            }
            let (i, j) = ((p.pos[0] as usize).min(gw - 1), (p.pos[1] as usize).min(gh - 1));
            count[j * gw + i] += 1;
            h.fastest = h.fastest.max(p.vel[0].hypot(p.vel[1]));
        }
        if let Some((k, &n)) = count.iter().enumerate().max_by_key(|(_, n)| **n) {
            h.crowd = n;
            h.crowd_at = [(k % gw) as u32, (k / gw) as u32];
        }
        let ledger = w.sim.download_ledger();
        h.boiled = ledger.boiled_count.iter().map(|&n| n as i64).sum::<i64>() - ledger.residue_count.iter().map(|&n| n as i64).sum::<i64>();
        h.unconverged = w.sim.solver.pressure_unconverged;
        h.substeps = w.sim.solver.substeps;
        h
    }

    /// Particles conserved since `start` (given what boiled away).
    pub(crate) fn conserved(&self, start: &Health, added: i64) -> bool {
        self.particles as i64 + (self.boiled - start.boiled) == start.particles as i64 + added
    }

    pub(crate) fn unconverged_share(&self, start: &Health) -> f64 {
        (self.unconverged - start.unconverged) as f64 / (self.substeps - start.substeps).max(1) as f64
    }
}

/// Checks everything in `Health` for one scenario and prints one line per failed bound
/// (one summary line when all hold). Returns whether all held.
pub(crate) fn check_health(r: &mut Report, what: &str, w: &World, start: &Health, end: &Health, added: i64) -> bool {
    let cap = w.sim.params.speed_cap_cells(w.sim.grid[1]) * 1.02 + 1.0;
    let conserved = end.conserved(start, added);
    let unconv = end.unconverged_share(start);
    let fails: Vec<String> = [
        (end.non_finite > 0, format!("{} non-finite particles", end.non_finite)),
        (end.in_solid > 0, format!("{} particles inside solids", end.in_solid)),
        (end.outside > 0, format!("{} particles outside the grid", end.outside)),
        (end.crowd > CROWD_MAX, format!("cell {:?} holds {} particles", end.crowd_at, end.crowd)),
        (
            !conserved,
            format!("particles {} -> {} ({:+} added, {} boiled)", start.particles, end.particles, added, end.boiled - start.boiled),
        ),
        (end.fastest > cap, format!("fastest particle {:.0} cells/s above the cap {:.0}", end.fastest, cap)),
        (unconv > UNCONVERGED_MAX, format!("{:.1} % unconverged pressure solves", unconv * 100.0)),
    ]
    .into_iter()
    .filter_map(|(bad, msg)| bad.then_some(msg))
    .collect();
    let measured = format!(
        "{} particles, crowd {}, {:.0} cells/s, {:.1} % unconv.",
        end.particles,
        end.crowd,
        end.fastest,
        unconv * 100.0
    );
    r.check(what, if fails.is_empty() { measured } else { fails.join("; ") }, "healthy", fails.is_empty());
    fails.is_empty()
}

fn cup_world(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), m: Material) -> World {
    let mut scene = Scene::new(GRID);
    let wall = |x: f32| Shape::Box { c: [x, 40.0], half: [4.0, 38.0], angle: 0.0 };
    scene.shapes.extend([wall(176.0), wall(336.0), Shape::Circle { c: [236.0, 20.0], r: 8.0 }]);
    let mut ps = Vec::new();
    fill_rect(&scene, [226.0, 40.0, 296.0, 90.0], m, &mut ps);
    let mut w = World::custom(gpu.clone(), scene, ps);
    apply(&mut w.sim.params);
    w.sim.step_blocking(0, ToolState::default());
    w
}

fn pool_world(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), fills: &[([f32; 4], Material)], shapes: &[Shape]) -> World {
    let mut scene = Scene::new(GRID);
    scene.shapes.extend_from_slice(shapes);
    let mut ps = Vec::new();
    for &(rect, m) in fills {
        fill_rect(&scene, rect, m, &mut ps);
    }
    let mut w = World::custom(gpu.clone(), scene, ps);
    apply(&mut w.sim.params);
    w.sim.step_blocking(0, ToolState::default());
    w
}

/// Drives `tool` for `frames` app frames of 1/60 s.
fn drive(w: &mut World, frames: usize, tool: impl Fn(usize) -> ToolState) {
    for k in 0..frames {
        w.sim.advance_blocking(1.0 / 60.0, tool(k));
    }
}

// ---- every map --------------------------------------------------------------------------

/// Every built-in map, 4 s, checked every second.
pub fn maps(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nEvery map (4 s each at 512 x 288, checked every second)");
    for (i, m) in MAPS.iter().enumerate() {
        let mut w = World::new(gpu.clone(), GRID, i);
        apply(&mut w.sim.params);
        w.sim.step_blocking(0, ToolState::default());
        let start = Health::of(&w);
        let mut worst = start;
        let mut ok = true;
        for _ in 0..4 {
            w.sim.advance_blocking(1.0, ToolState::default());
            let h = Health::of(&w);
            worst.crowd = worst.crowd.max(h.crowd);
            if h.non_finite > 0 || h.in_solid > 0 || h.outside > 0 || h.crowd > CROWD_MAX || !h.conserved(&start, 0) {
                ok = check_health(r, &format!("map {i} {}", m.name), &w, &start, &h, 0);
                break;
            }
            worst = Health { crowd: worst.crowd, ..h };
        }
        if ok {
            check_health(r, &format!("map {i} {}", m.name), &w, &start, &worst, 0);
        }
        r.shot(&w, &format!("stress_map_{i}"));
    }
}

// ---- every liquid -----------------------------------------------------------------------

/// Every liquid dropped into a cup with a peg: healthy, and it comes to rest.
pub fn liquids(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nEvery liquid (a block dropped into a cup with a peg, 8 s)");
    for (row, b) in BUILTINS.iter().enumerate() {
        if !pickable(row) {
            continue;
        }
        let m = Material(row as u8);
        let mut w = cup_world(gpu, apply, m);
        let start = Health::of(&w);
        w.sim.advance_blocking(4.0, ToolState::default());
        let mid = Health::of(&w);
        let v4 = rms_speed(&w.sim.download_particles());
        w.sim.advance_blocking(4.0, ToolState::default());
        let end = Health::of(&w);
        let ps = w.sim.download_particles();
        let v8 = rms_speed(&ps);
        let worst = Health { crowd: mid.crowd.max(end.crowd), ..end };
        check_health(r, &format!("{}: healthy", b.name), &w, &start, &worst, 0);
        // At rest: what still moves after 8 s is slow against the free-fall speed over the
        // drop (sqrt(2 g h), h = 50 cells) and not speeding up.
        let vref = (2.0 * w.sim.params.g_cells() * 50.0).sqrt();
        // Inviscid liquids slosh for a long time: they only have to keep losing energy.
        let rest = v8 < 0.02 * vref || v8 < 0.95 * v4;
        // Nothing left racing: a particle stuck at the speed cap on a resting surface used to
        // be drawn as white water and launch itself when nudged.
        let cap = w.sim.params.speed_cap_cells(w.sim.grid[1]);
        r.check(&format!("{}: nothing races at 8 s", b.name), format!("fastest {:.0} cells/s", end.fastest), &format!("< {:.0} (half the cap)", 0.5 * cap), end.fastest < 0.5 * cap);
        r.check(&format!("{}: settles", b.name), format!("rms {:.1} -> {:.1} cells/s", v4, v8), &format!("< {:.1} or falling", 0.02 * vref), rest);
        r.shot(&w, &format!("stress_liquid_{}", b.key));
    }
}

// ---- tools pushed to their limits -----------------------------------------------------

pub fn abuse(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nTools at their limits (corners, walls, the edge of the grid, full strength)");
    let pool = |gpu: &Arc<Gpu>| pool_world(gpu, apply, &[([0.0, 0.0, 512.0, 80.0], Material::Water)], &[]);

    // Push held at full strength in a corner of the pool, then released.
    let mut w = pool(gpu);
    let start = Health::of(&w);
    drive(&mut w, 120, |_| {
        let mut t = ToolState::push([WALL + 4.0, WALL + 4.0], 20.0);
        t.strength *= 4.0;
        t
    });
    let held = Health::of(&w);
    w.sim.advance_blocking(2.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "push at 400 % in a corner, 2 s", &w, &start, &Health { crowd: held.crowd.max(end.crowd), ..end }, 0);

    // Grab dragged fast to and fro across the tank and into the walls.
    let mut w = pool(gpu);
    let start = Health::of(&w);
    drive(&mut w, 180, |k| {
        let x = 256.0 + 250.0 * (k as f32 * 0.12).sin();
        let vx = 250.0 * 0.12 * 60.0 * (k as f32 * 0.12).cos();
        let mut t = ToolState::grab([x, 40.0], [vx, 0.0], 25.0);
        t.strength *= 4.0;
        t
    });
    w.sim.advance_blocking(1.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "grab swept wall to wall at 400 %, 3 s", &w, &start, &end, 0);

    // A box placed over part of the pool, then a ramp through it; then both removed.
    let mut w = pool(gpu);
    let start = Health::of(&w);
    w.add_shape(Shape::Box { c: [256.0, 40.0], half: [40.0, 30.0], angle: 0.3 });
    w.sim.advance_blocking(1.0, ToolState::default());
    let boxed = Health::of(&w);
    w.add_shape(Shape::Ramp { c: [150.0, 30.0], half: [30.0, 20.0], angle: 0.0, flip: false });
    w.sim.advance_blocking(1.0, ToolState::default());
    let ramped = Health::of(&w);
    w.remove_shape_at([256.0, 40.0]);
    w.remove_shape_at([150.0, 30.0]);
    w.sim.advance_blocking(1.0, ToolState::default());
    let end = Health::of(&w);
    let worst = Health { crowd: boxed.crowd.max(ramped.crowd).max(end.crowd), in_solid: boxed.in_solid.max(ramped.in_solid), ..end };
    check_health(r, "solids placed over liquid and removed", &w, &start, &worst, 0);

    // A box laid over the pool two cells beside another one: the liquid it covers is pushed
    // out, and must not be thrown across the gap into its neighbour.
    let mut w = pool_world(gpu, apply, &[([0.0, 0.0, 512.0, 80.0], Material::Water)], &[Shape::Box { c: [300.0, 40.0], half: [20.0, 30.0], angle: 0.0 }]);
    let start = Health::of(&w);
    w.add_shape(Shape::Box { c: [258.0, 40.0], half: [20.0, 30.0], angle: 0.0 });
    // (The first substep after a reset only builds the grid.)
    w.sim.step_blocking(3, ToolState::default());
    let first = Health::of(&w);
    // (The liquid squeezed into the gap packs its top, 50 - 90 particles a cell, until the
    // gap drains after about 1.5 s; in the build before this round too.)
    w.sim.advance_blocking(2.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "a box laid over liquid 2 cells from another, 2 s", &w, &start, &Health { in_solid: first.in_solid.max(end.in_solid), ..end }, 0);
    if first.crowd > CROWD_MAX {
        println!("         (while it was being emptied: {} particles in cell {:?})", first.crowd, first.crowd_at);
    }

    // A solid closing the pool in: a lid over all of it (sealed, full region).
    let mut w = pool(gpu);
    let start = Health::of(&w);
    w.add_shape(Shape::Box { c: [256.0, 84.0], half: [260.0, 6.0], angle: 0.0 });
    w.sim.advance_blocking(2.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "a lid laid over a full pool", &w, &start, &end, 0);
    r.check("same: the pool stays still", format!("rms {:.2} cells/s", rms_speed(&w.sim.download_particles())), "< 5", rms_speed(&w.sim.download_particles()) < 5.0);

    // Pour, spray, jet and drop at and beyond the edge of the grid.
    let mut w = pool(gpu);
    let start = Health::of(&w);
    let mut added = 0i64;
    for (at, source) in [
        ([-5.0, 200.0], Source::Pour),
        ([520.0, 100.0], Source::Jet { angle: 90.0 }),
        ([256.0, 300.0], Source::Spray),
        ([2.0, 2.0], Source::Pour),
        ([509.0, 285.0], Source::Jet { angle: -135.0 }),
    ] {
        for _ in 0..30 {
            added += w.emit(source, Material::Oil, at, 12.0, 30000.0, 1.0 / 60.0) as i64;
            w.sim.advance_blocking(1.0 / 60.0, source.nozzle(at, 12.0, 30000.0));
        }
        w.end_stroke();
    }
    for at in [[0.0, 0.0], [512.0, 288.0], [-40.0, 140.0], [256.0, 290.0]] {
        added += w.stamp(Material::Honey, at, 15.0) as i64;
    }
    w.sim.advance_blocking(2.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "liquid tools at and past the edge of the grid", &w, &start, &end, added);

    // The same pour into a tank that never held anything, for comparison.
    let mut w = pool_world(gpu, apply, &[], &[]);
    let fresh = Health::of(&w);
    let mut added = 0i64;
    for _ in 0..60 {
        added += w.emit(Source::Pour, Material::Water, [256.0, 200.0], 9.0, 12000.0, 1.0 / 60.0) as i64;
        w.sim.advance_blocking(1.0 / 60.0, Source::Pour.nozzle([256.0, 200.0], 9.0, 12000.0));
    }
    w.end_stroke();
    w.sim.advance_blocking(1.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "pour into a fresh empty tank", &w, &fresh, &end, added);

    // Erase everything, then pour again into the empty tank.
    let mut w = pool(gpu);
    for k in 0..60 {
        let x = 20.0 + 472.0 * k as f32 / 59.0;
        w.sim.advance_blocking(1.0 / 60.0, ToolState::erase([x, 40.0], 60.0));
    }
    let gone = Health::of(&w);
    let mut added = 0i64;
    for _ in 0..60 {
        added += w.emit(Source::Pour, Material::Water, [256.0, 200.0], 9.0, 12000.0, 1.0 / 60.0) as i64;
        w.sim.advance_blocking(1.0 / 60.0, Source::Pour.nozzle([256.0, 200.0], 9.0, 12000.0));
    }
    w.end_stroke();
    w.sim.advance_blocking(1.0, ToolState::default());
    let end = Health::of(&w);
    r.check("erase: a whole pool can be erased", format!("{} left", gone.particles), "0", gone.particles == 0);
    check_health(r, "pour into the emptied tank", &w, &gone, &end, added);

    // Heat held on a pool until it boils; Cool held on lava until it sets.
    let mut w = pool(gpu);
    let start = Health::of(&w);
    drive(&mut w, 600, |_| ToolState::heat([256.0, 30.0], 30.0, false));
    let end = Health::of(&w);
    check_health(r, "heat brush on water, 10 s", &w, &start, &end, 0);
    r.check("same: water boils away", format!("{} particles boiled", end.boiled - start.boiled), "> 0", end.boiled > start.boiled);
    let mut w = pool_world(gpu, apply, &[([0.0, 0.0, 512.0, 30.0], Material::Water), ([200.0, 30.0, 312.0, 70.0], Material::Lava)], &[]);
    let start = Health::of(&w);
    drive(&mut w, 600, |k| ToolState::heat([256.0 + 50.0 * (k as f32 * 0.05).sin(), 45.0], 25.0, true));
    w.sim.advance_blocking(3.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "cool brush swept over lava on water, 10 s", &w, &start, &end, 0);

    // The strongest magnet against a wall, in ferrofluid.
    let mut w = pool_world(gpu, apply, &[([0.0, 0.0, 512.0, 40.0], Material::Ferrofluid)], &[]);
    w.add_magnet(Magnet::new([WALL + 2.0, 20.0], 10.0, 90.0, crate::sim::MAGNET_STRENGTH[1]));
    w.add_magnet(Magnet::new([256.0, 2.0], 10.0, 0.0, crate::sim::MAGNET_STRENGTH[1]));
    let start = Health::of(&w);
    w.sim.advance_blocking(3.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "two strongest magnets at the walls, ferrofluid", &w, &start, &end, 0);

    // Currents at full speed aimed into the floor and a corner.
    let mut w = pool(gpu);
    let start = Health::of(&w);
    w.add_current(Current::new([256.0, 20.0], 30.0, 0.0, crate::sim::CURRENT_STRENGTH[1]));
    w.add_current(Current::new([480.0, 40.0], 25.0, -45.0, crate::sim::CURRENT_STRENGTH[1]));
    w.sim.advance_blocking(3.0, ToolState::default());
    // (While they run, splashes at the speed cap pack 20 - 80 particles into a cell for a
    // moment; what matters is that nothing stays packed.)
    w.hold_current(None);
    w.sim.advance_blocking(1.5, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "two fastest currents into the floor and a corner, 1.5 s after", &w, &start, &end, 0);

    // A tank cluttered with solids, every liquid poured over it.
    let mut w = pool_world(gpu, apply, &[], &[]);
    let mut seed = 0x9e37_79b9u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    for _ in 0..30 {
        let c = [20.0 + 472.0 * rnd(), 10.0 + 180.0 * rnd()];
        let s = match (rnd() * 3.0) as u32 {
            0 => Shape::Circle { c, r: 3.0 + 12.0 * rnd() },
            1 => Shape::Box { c, half: [3.0 + 25.0 * rnd(), 1.0 + 6.0 * rnd()], angle: 3.0 * rnd() },
            _ => Shape::Ramp { c, half: [5.0 + 20.0 * rnd(), 3.0 + 8.0 * rnd()], angle: 3.0 * rnd(), flip: rnd() > 0.5 },
        };
        w.add_shape(s);
    }
    let start = Health::of(&w);
    let mut added = 0i64;
    for (k, (row, _)) in BUILTINS.iter().enumerate().filter(|(row, _)| pickable(*row)).enumerate() {
        let at = [30.0 + 452.0 * (k % 18) as f32 / 17.0, 250.0];
        for _ in 0..12 {
            added += w.emit(Source::Pour, Material(row as u8), at, 6.0, 8000.0, 1.0 / 60.0) as i64;
            w.sim.advance_blocking(1.0 / 60.0, Source::Pour.nozzle(at, 6.0, 8000.0));
        }
        w.end_stroke();
    }
    w.sim.advance_blocking(4.0, ToolState::default());
    let end = Health::of(&w);
    check_health(r, "every liquid poured over 30 random solids", &w, &start, &end, added);
    r.shot(&w, "stress_clutter");

    // Maps reloaded over and over keep the same particle budget and start from scratch.
    let mut w = World::new(gpu.clone(), GRID, 0);
    apply(&mut w.sim.params);
    let mut counts = Vec::new();
    for k in 0..6 {
        w.load_map(k % 3);
        w.sim.step_blocking(0, ToolState::default());
        w.sim.advance_blocking(0.2, ToolState::default());
        counts.push(Health::of(&w).particles);
    }
    let repeat = counts[0] == counts[3] && counts[1] == counts[4] && counts[2] == counts[5];
    r.check("reloading maps: the same contents each time", format!("{:?}", counts), "repeating", repeat);
}

// ---- parameters at the ends of their ranges ---------------------------------------------

/// Playground for 1 s with each listed parameter at its minimum, its maximum and its off
/// value (README, "How the ranges were checked").
pub fn extremes(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nParameters at the ends of their ranges (Playground, 1 s each)");
    let keys = [
        "gravity", "cfl", "dt_max", "speed_cap", "correction_max", "crowd_jitter", "apic_clamp", "viscosity_water",
        "viscosity_honey", "tension_water", "tension_oil_water", "contact_angle_water", "droplet_cells", "mix_rate",
        "density_oil", "spray_cfl", "curvature_max", "pressure_tol", "viscosity_tol", "heat_rate",
    ];
    for key in keys {
        let Some(def) = crate::sim::Param::find(key) else {
            r.check(&format!("{key}: exists"), "missing".into(), "a parameter", false);
            continue;
        };
        let mut values = vec![("min", def.min), ("max", def.max)];
        if let Some(off) = def.off {
            values.push(("off", off));
        }
        let mut fails = Vec::new();
        let mut line = Vec::new();
        for (label, v) in values {
            let mut w = World::new(gpu.clone(), GRID, 0);
            apply(&mut w.sim.params);
            w.sim.params.set(key, v);
            w.sim.step_blocking(0, ToolState::default());
            let start = Health::of(&w);
            w.sim.advance_blocking(1.0, ToolState::default());
            let end = Health::of(&w);
            let bad = end.non_finite > 0 || end.in_solid > 0 || end.outside > 0 || end.crowd > CROWD_MAX || !end.conserved(&start, 0);
            line.push(format!("{label} crowd {}", end.crowd));
            if bad {
                fails.push(format!(
                    "{label}: {} non-finite, {} in solids, {} outside, crowd {}, {} -> {}",
                    end.non_finite, end.in_solid, end.outside, end.crowd, start.particles, end.particles
                ));
            }
        }
        r.check(&format!("{key} at min / max / off"), if fails.is_empty() { line.join(", ") } else { fails.join("; ") }, "healthy", fails.is_empty());
    }
}

// ---- the renderer ------------------------------------------------------------------------

/// Pixels that are pure black (a NaN or a hole in the shading: nothing in the picture is
/// darker than the back wall's 0.04).
fn black_pixels(px: &[u8]) -> usize {
    px.as_chunks::<4>().0.iter().filter(|c| c[0] <= 2 && c[1] <= 2 && c[2] <= 2).count()
}

/// Share of pixels that change by more than 40/255 between consecutive frames and are back
/// the frame after (the flicker metric of README "Look").
pub(crate) fn flicker(frames: &[Vec<u8>]) -> f64 {
    let (mut n, mut total) = (0usize, 0usize);
    for f in frames.windows(3) {
        for ((a, b), c) in f[0].as_chunks::<4>().0.iter().zip(f[1].as_chunks::<4>().0).zip(f[2].as_chunks::<4>().0) {
            total += 1;
            let jump = (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
            let back = (0..3).map(|k| (a[k] as i32 - c[k] as i32).abs()).max().unwrap_or(0);
            if jump > 40 && back < 20 {
                n += 1;
            }
        }
    }
    n as f64 / total.max(1) as f64
}

pub fn render(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nRenderer (every map at 2 s; hot liquid's light; flicker over six frames)");
    let size = [960, 540];
    let mut off = Offscreen::new(gpu.clone(), size);
    let fit = View::fit(GRID, size);
    let mut worst = (0usize, String::new());
    for (i, m) in MAPS.iter().enumerate() {
        let mut w = World::new(gpu.clone(), GRID, i);
        apply(&mut w.sim.params);
        off.renderer.invalidate();
        w.sim.step_blocking(0, ToolState::default());
        for _ in 0..120 {
            w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
            off.track(&w.sim);
        }
        for (label, view) in [("fit", fit), ("4x", View { center: [256.0, 40.0], zoom: fit.zoom * 4.0 })] {
            let px = off.capture(&w.sim, view);
            let n = black_pixels(&px);
            if n > worst.0 {
                worst = (n, format!("{} ({label})", m.name));
            }
        }
    }
    r.check("no black specks on any map (fit and 4 x)", format!("worst {} px {}", worst.0, worst.1), "0", worst.0 == 0);

    // Hot liquid only adds light: the picture with its glow on is nowhere darker than with
    // the glow off, above the melt and beside it.
    let wall = |x: f32| Shape::Box { c: [x, 50.0], half: [4.0, 48.0], angle: 0.0 };
    let mut w = pool_world(gpu, apply, &[([200.0, WALL, 312.0, 60.0], Material::Lava)], &[wall(196.0), wall(316.0)]);
    off.renderer.invalidate();
    for _ in 0..120 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
        off.track(&w.sim);
    }
    let view = View { center: [256.0, 70.0], zoom: fit.zoom * 3.0 };
    let lit = off.capture(&w.sim, view);
    let look = off.renderer.look;
    off.renderer.look.glow = 0.0;
    let dark = off.capture(&w.sim, view);
    off.renderer.look = look;
    let darker = lit.as_chunks::<4>().0.iter().zip(dark.as_chunks::<4>().0).filter(|(a, b)| (0..3).any(|k| (a[k] as i32) < b[k] as i32 - 6)).count();
    let share = darker as f64 / (size[0] * size[1]) as f64;
    r.check("lava: its glow only adds light", format!("{:.2} % of pixels darker with it", share * 100.0), "< 0.05 %", share < 5e-4);

    // Flicker of the hot surface and the air above it, over six consecutive frames.
    let mut frames = Vec::new();
    for _ in 0..6 {
        frames.push(off.capture(&w.sim, view));
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
        off.track(&w.sim);
    }
    let f = flicker(&frames);
    r.check("lava cup: flicker over six frames", format!("{:.3} % of pixels", f * 100.0), "< 0.05 %", f < 5e-4);
    if let Some(dir) = r.shots_dir() {
        off.write_png(&lit, &dir.join("stress_render_lava.png"));
    }
}
