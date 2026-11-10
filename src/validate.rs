//! Physics validation: small experiments with a known or bounded outcome, run off-screen
//! on the real solver. `sprout validate` prints what was measured next to what was expected.

use std::{path::PathBuf, sync::Arc};

use crate::{
    gpu::Gpu,
    headless::Offscreen,
    render::View,
    maps::fill_rect,
    scene::{Scene, Shape, WALL},
    sim::{KIND_AQUEOUS, KIND_OIL, Material, NOMINAL_DT, PPC, Particle, SimParams, ToolState},
    world::{Nozzle, World},
};

const GRID: [u32; 2] = [512, 288];

struct Report {
    failures: u32,
    /// Where to save a picture of each scenario (`--shots DIR`).
    shots: Option<(Offscreen, PathBuf)>,
}

impl Report {
    fn shot(&mut self, w: &World, name: &str) {
        if let Some((off, dir)) = &mut self.shots {
            // Every scenario builds its own world, so drop what the last one left cached.
            off.renderer.invalidate();
            off.save_png(&w.sim, View::fit(GRID, off.size), &dir.join(format!("{name}.png")));
        }
    }

    fn check(&mut self, what: &str, measured: String, expected: &str, ok: bool) {
        if !ok {
            self.failures += 1;
        }
        println!("  [{}] {:<44} {:<26} expected {}", if ok { "pass" } else { "FAIL" }, what, measured, expected);
    }
}

fn world(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), fills: &[([f32; 4], Material)]) -> World {
    let scene = Scene::new(GRID);
    let mut ps = Vec::new();
    for &(rect, m) in fills {
        fill_rect(&scene, rect, m, &mut ps);
    }
    let mut w = World::custom(gpu.clone(), scene, ps);
    apply(&mut w.sim.params);
    w.sim.step_blocking(0, ToolState::default());
    w
}

fn run(w: &mut World, seconds: f32) {
    if seconds > 0.0 {
        w.sim.advance_blocking(seconds as f64, ToolState::default());
    }
}

fn rho(p: &Particle, sp: &SimParams) -> f32 {
    if p.kind == KIND_OIL { sp.rho[2] } else { sp.rho[0] + (sp.rho[1] - sp.rho[0]) * p.s }
}

fn rms_speed(ps: &[Particle]) -> f32 {
    (ps.iter().map(|p| p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sum::<f32>() / ps.len().max(1) as f32).sqrt()
}

/// Mean kinetic and gravitational potential energy per particle.
fn energy(ps: &[Particle], sp: &SimParams) -> (f64, f64) {
    let (mut ke, mut pe) = (0.0f64, 0.0f64);
    for p in ps {
        let rho = rho(p, sp) as f64;
        ke += 0.5 * rho * (p.vel[0] as f64).hypot(p.vel[1] as f64).powi(2);
        pe += rho * sp.g_cells() as f64 * p.pos[1] as f64;
    }
    (ke / ps.len().max(1) as f64, pe / ps.len().max(1) as f64)
}

fn mean_height(ps: &[Particle], pick: impl Fn(&Particle) -> f32) -> f32 {
    let (mut sum, mut weight) = (0.0f64, 0.0f64);
    for p in ps {
        let w = pick(p) as f64;
        sum += w * p.pos[1] as f64;
        weight += w;
    }
    (sum / weight.max(1e-9)) as f32
}

/// Area (cells^2) of the region holding liquid, from a 1-cell occupancy histogram.
fn occupied_area(ps: &[Particle]) -> f32 {
    let (w, h) = (GRID[0] as usize, GRID[1] as usize);
    let mut count = vec![0u16; w * h];
    for p in ps {
        let (i, j) = (p.pos[0] as usize, p.pos[1] as usize);
        if i < w && j < h {
            count[j * w + i] += 1;
        }
    }
    count.iter().filter(|&&c| c > 0).count() as f32
}

fn percentile(mut v: Vec<f32>, q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let k = ((v.len() - 1) as f32 * q) as usize;
    v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    v[k]
}

/// A pool at rest must stay at rest and keep its volume.
fn hydrostatic(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nHydrostatic pool (water, 100 cells deep, 10 s)");
    let depth = 100.0 + WALL;
    let mut w = world(gpu, apply, &[([0.0, 0.0, 512.0, depth], Material::Water)]);
    let n0 = w.sim.download_particles().len();
    run(&mut w, 10.0);
    r.shot(&w, "hydrostatic");
    let ps = w.sim.download_particles();
    let top = percentile(ps.iter().map(|p| p.pos[1]).collect(), 0.995) - WALL;
    let v_ref = (w.sim.params.g_cells() * 100.0).sqrt();
    let rms = rms_speed(&ps);
    r.check("particles conserved", format!("{} -> {}", n0, ps.len()), "equal", n0 == ps.len());
    r.check("surface level", format!("{:.2} cells ({:+.2} %)", top, (top / 100.0 - 1.0) * 100.0), "100 +/- 2 %", (top / 100.0 - 1.0).abs() < 0.02);
    r.check("residual motion (rms speed / sqrt(g h))", format!("{:.5}", rms / v_ref), "< 0.01", rms / v_ref < 0.01);
}

/// Collapse of a water column. The surge front is compared with the classic experiment
/// of Martin & Moyce (1952) for a column twice as tall as it is wide.
fn dam_break(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nDam break (water column 60 x 120 cells)");
    let a = 60.0;
    let mut w = world(gpu, apply, &[([0.0, 0.0, WALL + a, WALL + 2.0 * a], Material::Water)]);
    let g = w.sim.params.g_cells();
    // Experimental front positions Z = x / a against T = t * sqrt(2 g / a).
    let reference = [(1.19, 1.44), (1.63, 1.89), (1.98, 2.33), (2.32, 2.78), (2.65, 3.22), (2.98, 3.67), (3.33, 4.11)];
    let mut t = 0.0f32;
    let mut worst = 0.0f32;
    let mut line = String::new();
    let mut fronts = Vec::new();
    for &(tt, z_ref) in &reference {
        let target = tt / (2.0 * g / a).sqrt();
        run(&mut w, target - t);
        t = target;
        let ps = w.sim.download_particles();
        let front = (percentile(ps.iter().map(|p| p.pos[0]).collect(), 0.999) - WALL) / a;
        if (front / z_ref - 1.0).abs() > worst.abs() {
            worst = front / z_ref - 1.0;
        }
        line += &format!("{:.2}/{:.2} ", front, z_ref);
        fronts.push((tt, front));
    }
    r.shot(&w, "dam");
    println!("  front Z at T = 1.19 .. 3.33 (measured/experiment): {line}");
    let speed = (fronts[6].1 - fronts[2].1) / (fronts[6].0 - fronts[2].0);
    // A frictionless solver released instantly runs ahead of the experiment, whose gate
    // takes time to clear and whose floor drags; the front must still stay well short of
    // the shallow-water (Ritter) limit dZ/dT = 2. The table above is one data set: across
    // published experiments the mean front speed v / sqrt(g H) after t* = 1 ranges from
    // 1.14 to 1.74 (Lobovsky et al. 2014, table 1; Martin & Moyce's own 2:1 run: 1.69).
    // `validate convergence` shows the solver's front is converged to about 0.4 %.
    r.check("front position vs experiment", format!("worst deviation {:+.1} %", worst * 100.0), "within 20 %", worst.abs() < 0.20);
    r.check("front speed dZ/dT", format!("{:.2}", speed), "1.2 .. 1.6 (table below 1.32; experiments 1.14-1.74; Ritter 2)", speed > 1.2 && speed < 1.6);
}

/// Dam break from a long, shallow reservoir. Shallow-water theory (Ritter 1892) gives an
/// exact similarity solution: at the dam site the depth holds at 4/9 of the reservoir depth
/// and the flow speed at 2/3 sqrt(g h0) until reflections arrive.
fn ritter(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nShallow dam break (reservoir 300 x 30 cells, Ritter 1892)");
    let (len, h0) = (300.0, 30.0);
    let mut w = world(gpu, apply, &[([0.0, 0.0, WALL + len, WALL + h0], Material::Water)]);
    let c0 = (w.sim.params.g_cells() * h0).sqrt();
    let (mut depth, mut speed, mut samples) = (0.0f32, 0.0f32, 0u32);
    let mut t = 0.0;
    // Between t sqrt(g / h0) of about 3.4 and 5.1: after the start-up transient, before the
    // surge has come back from the far wall.
    for target in [0.6f32, 0.7, 0.8, 0.9] {
        run(&mut w, target - t);
        t = target;
        let ps = w.sim.download_particles();
        let band: Vec<&Particle> = ps.iter().filter(|p| (p.pos[0] - (WALL + len)).abs() < 3.0).collect();
        depth += band.len() as f32 / (6.0 * PPC as f32);
        speed += band.iter().map(|p| p.vel[0]).sum::<f32>() / band.len().max(1) as f32;
        samples += 1;
    }
    r.shot(&w, "ritter");
    let (depth, speed) = (depth / samples as f32 / h0, speed / samples as f32 / c0);
    r.check("depth at the dam site / h0", format!("{:.3}", depth), "4/9 = 0.444 (+/- 10 %)", (depth / (4.0 / 9.0) - 1.0).abs() < 0.10);
    r.check("flow speed at the dam site / sqrt(g h0)", format!("{:.3}", speed), "2/3 = 0.667 (+/- 10 %)", (speed / (2.0 / 3.0) - 1.0).abs() < 0.10);
}

/// Three layers stacked upside down must overturn into oil / water / syrup.
fn stratification(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nRayleigh-Taylor overturn (oil under water under syrup, 40 s)");
    let mut w = world(
        gpu,
        apply,
        &[
            ([0.0, 0.0, 512.0, 60.0], Material::Oil),
            ([0.0, 60.0, 512.0, 120.0], Material::Water),
            ([0.0, 120.0, 512.0, 180.0], Material::Syrup),
        ],
    );
    let before = w.sim.download_particles();
    let syrup0: f64 = before.iter().filter(|p| p.kind == KIND_AQUEOUS).map(|p| p.s as f64).sum();
    let area0 = occupied_area(&before);
    let sp = w.sim.params;
    // Nothing here may add mechanical energy: overturning turns potential energy into
    // motion, and viscosity and numerical damping then remove it.
    let energy = |ps: &[Particle]| energy(ps, &sp);
    let (_, pe0) = energy(&before);
    let (mut last, mut worst_rise, mut ke_peak) = (pe0, 0.0f64, 0.0f64);
    let mut ps = before.clone();
    // Share of the aqueous phase above the oil layer's lowest 5 %: how far separation got.
    let trapped = |ps: &[Particle]| {
        let floor = percentile(ps.iter().filter(|p| p.kind == KIND_OIL).map(|p| p.pos[1]).collect(), 0.05);
        ps.iter().filter(|p| p.kind != KIND_OIL && p.pos[1] > floor).count() as f32
            / ps.iter().filter(|p| p.kind != KIND_OIL).count().max(1) as f32
    };
    let mut trend = Vec::new();
    for k in 0..20 {
        run(&mut w, 2.0);
        ps = w.sim.download_particles();
        if k % 4 == 3 {
            trend.push(trapped(&ps));
        }
        let (ke, pe) = energy(&ps);
        worst_rise = worst_rise.max(ke + pe - last);
        ke_peak = ke_peak.max(ke);
        last = ke + pe;
    }
    r.shot(&w, "layers");
    let (ke1, pe1) = energy(&ps);
    let released = pe0 - pe1;
    let oil_y = mean_height(&ps, |p| (p.kind == KIND_OIL) as u32 as f32);
    let water_y = mean_height(&ps, |p| if p.kind == KIND_AQUEOUS { 1.0 - p.s } else { 0.0 });
    let syrup_y = mean_height(&ps, |p| if p.kind == KIND_AQUEOUS { p.s } else { 0.0 });
    r.check(
        "mean heights oil / water / syrup",
        format!("{:.0} / {:.0} / {:.0} cells", oil_y, water_y, syrup_y),
        "oil > water > syrup",
        oil_y > water_y + 20.0 && water_y > syrup_y,
    );
    // Oil should end as one clean layer: almost all of it above almost all the rest.
    let oil_floor = percentile(ps.iter().filter(|p| p.kind == KIND_OIL).map(|p| p.pos[1]).collect(), 0.05);
    let below = ps.iter().filter(|p| p.kind != KIND_OIL && p.pos[1] > oil_floor).count() as f32
        / ps.iter().filter(|p| p.kind != KIND_OIL).count() as f32;
    println!(
        "  water/syrup above the oil layer's floor at 8, 16, 24, 32, 40 s: {}",
        trend.iter().map(|t| format!("{:.2} %", t * 100.0)).collect::<Vec<_>>().join(", ")
    );
    r.check("water/syrup left above the oil layer's floor", format!("{:.2} %", below * 100.0), "< 3 %", below < 0.03);
    let syrup1: f64 = ps.iter().filter(|p| p.kind == KIND_AQUEOUS).map(|p| p.s as f64).sum();
    r.check(
        "dissolved syrup volume conserved",
        format!("{:+.4} %", (syrup1 / syrup0 - 1.0) * 100.0),
        "|drift| < 0.1 %",
        (syrup1 / syrup0 - 1.0).abs() < 1e-3,
    );
    // Miscibility: water and syrup should have blended; oil particles never change kind.
    let aq: Vec<f32> = ps.iter().filter(|p| p.kind == KIND_AQUEOUS).map(|p| p.s).collect();
    let mixed = aq.iter().filter(|&&s| s > 0.1 && s < 0.9).count() as f32 / aq.len() as f32;
    r.check("aqueous particles that are a water-syrup blend", format!("{:.0} %", mixed * 100.0), "> 50 %", mixed > 0.5);
    let n_oil0 = before.iter().filter(|p| p.kind == KIND_OIL).count();
    let n_oil1 = ps.iter().filter(|p| p.kind == KIND_OIL).count();
    r.check("oil phase conserved", format!("{} -> {}", n_oil0, n_oil1), "equal", n_oil0 == n_oil1);
    let area1 = occupied_area(&ps);
    r.check("total volume", format!("{:+.2} %", (area1 / area0 - 1.0) * 100.0), "|drift| < 3 %", (area1 / area0 - 1.0).abs() < 0.03);
    r.check(
        "mechanical energy never rises (2 s samples)",
        format!("worst {:+.2} % of released", worst_rise / released * 100.0),
        "< 1 %",
        worst_rise < 0.01 * released,
    );
    r.check(
        "kinetic energy left after 40 s",
        format!("{:.2} % of released (peak {:.0} %)", ke1 / released * 100.0, ke_peak / released * 100.0),
        "< 1 %",
        ke1 < 0.01 * released,
    );
}

/// Viscous gravity current: a syrup column slumping over a no-slip floor. In the
/// lubrication regime the front follows Huppert's (1982) similarity solution
/// x = 1.411 (g q^3 / 3 nu)^(1/5) t^(1/5), with q the released area.
fn viscous_current(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nViscous gravity current (syrup, Huppert 1982)");
    let (bw, bh) = (50.0, 60.0);
    let mut w = world(gpu, apply, &[([0.0, 0.0, WALL + bw, WALL + bh], Material::Syrup)]);
    let q = w.sim.download_particles().len() as f32 / PPC as f32;
    let (g, nu) = (w.sim.params.g_cells(), w.sim.params.nu_cells(1));
    let k = 1.411 * (g * q.powi(3) / (3.0 * nu)).powf(0.2);
    let mut t = 0.0;
    let mut pts = Vec::new();
    let mut line = String::new();
    for target in [2.0f32, 4.0, 8.0, 16.0] {
        run(&mut w, target - t);
        t = target;
        let ps = w.sim.download_particles();
        let front = percentile(ps.iter().map(|p| p.pos[0]).collect(), 0.998) - WALL;
        line += &format!("t={target:.0}s {:.0}/{:.0}  ", front, k * target.powf(0.2));
        pts.push((target, front));
    }
    r.shot(&w, "viscous");
    println!("  front in cells (measured/theory): {line}");
    let exponent = (pts[3].1 / pts[1].1).ln() / (pts[3].0 / pts[1].0).ln();
    let ratio = pts[3].1 / (k * pts[3].0.powf(0.2));
    r.check("spreading exponent", format!("{:.3}", exponent), "0.2 (+/- 0.05)", (exponent - 0.2).abs() < 0.05);
    r.check("front position / theory at 16 s", format!("{:.2}", ratio), "1.0 (+/- 0.2)", (ratio - 1.0).abs() < 0.2);

    // The same column of water must run away far faster.
    let mut ww = world(gpu, apply, &[([0.0, 0.0, WALL + bw, WALL + bh], Material::Water)]);
    run(&mut ww, 1.0);
    let water_front = percentile(ww.sim.download_particles().iter().map(|p| p.pos[0]).collect(), 0.998) - WALL;
    let mut ws = world(gpu, apply, &[([0.0, 0.0, WALL + bw, WALL + bh], Material::Syrup)]);
    run(&mut ws, 1.0);
    let syrup_front = percentile(ws.sim.download_particles().iter().map(|p| p.pos[0]).collect(), 0.998) - WALL;
    r.check(
        "front after 1 s, water vs syrup",
        format!("{:.0} vs {:.0} cells", water_front, syrup_front),
        "water at least 2x further",
        water_front - bw > 2.0 * (syrup_front - bw),
    );
}

/// A blob of oil released at the bottom of a pool must rise and collect on top; a blob
/// of syrup released at the top must sink.
fn buoyancy(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nBuoyancy (oil and syrup blobs in a water pool, 12 s)");
    let scene = Scene::new(GRID);
    let mut ps = Vec::new();
    fill_rect(&scene, [0.0, 0.0, 512.0, 150.0], Material::Water, &mut ps);
    // Swap two discs of the pool for the other fluids.
    let (oil_c, syr_c, rad) = ([150.0f32, 35.0f32], [360.0f32, 115.0f32], 22.0f32);
    for p in &mut ps {
        let inside = |c: [f32; 2]| (p.pos[0] - c[0]).hypot(p.pos[1] - c[1]) < rad;
        if inside(oil_c) {
            *p = Particle::new(Material::Oil, p.pos, [0.0, 0.0]);
        } else if inside(syr_c) {
            *p = Particle::new(Material::Syrup, p.pos, [0.0, 0.0]);
        }
    }
    let mut w = World::custom(gpu.clone(), scene, ps);
    apply(&mut w.sim.params);
    w.sim.step_blocking(0, ToolState::default());
    run(&mut w, 12.0);
    r.shot(&w, "buoyancy");
    let ps = w.sim.download_particles();
    let surface = percentile(ps.iter().map(|p| p.pos[1]).collect(), 0.995);
    let oil_y = mean_height(&ps, |p| (p.kind == KIND_OIL) as u32 as f32);
    let syr_y = mean_height(&ps, |p| if p.kind == KIND_AQUEOUS { p.s } else { 0.0 });
    r.check("oil blob", format!("{:.0} -> {:.0} cells (surface {:.0})", oil_c[1], oil_y, surface), "within 12 cells of the surface", surface - oil_y < 12.0);
    r.check("syrup blob", format!("{:.0} -> {:.0} cells", syr_c[1], syr_y), "sank below 60 cells", syr_y < 60.0);
}

/// The interactive tools, driven through the same calls the app makes each frame.
fn tools(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nTools (pour, erase, slot recycling, grab / stir, placing and removing solids)");
    let frame = |w: &mut World, tool: ToolState| w.sim.step_blocking(4, tool);
    let live_in = |ps: &[Particle], scene: &Scene, depth: f32| ps.iter().filter(|p| scene.sdf(p.pos) < -depth).count();

    // ---- pour, then erase ------------------------------------------------------------
    let mut w = World::custom(gpu.clone(), Scene::new(GRID), Vec::new());
    apply(&mut w.sim.params);
    let dt = NOMINAL_DT;
    let nozzle = Nozzle::fixed(Material::Water, [256.0, 120.0], 9.0, 12000.0);
    let mut emitted = 0;
    for _ in 0..120 {
        emitted += w.pour(&nozzle, 4.0 * dt);
        frame(&mut w, ToolState::default());
    }
    r.shot(&w, "tools_pour_0");
    run(&mut w, 6.0);
    r.shot(&w, "tools_pour_1");
    let ps = w.sim.download_particles();
    let depth = emitted as f32 / PPC as f32 / (GRID[0] as f32 - 2.0 * WALL);
    let mean = mean_height(&ps, |_| 1.0) - WALL;
    r.check("pour: every emitted particle is live", format!("{} of {}", ps.len(), emitted), "equal, about 24000", ps.len() == emitted && emitted.abs_diff(24000) < 500);
    r.check(
        "pour: the stream ends as a pool on the floor",
        format!("mean height {:.1} cells", mean),
        &format!("{:.1} for a level pool, < {:.1}", 0.5 * depth, depth),
        mean > 0.4 * depth && mean < depth,
    );
    r.check("pour: nothing ends inside a solid", format!("{}", live_in(&ps, &w.scene, 0.5)), "0", live_in(&ps, &w.scene, 0.5) == 0);

    let brush = [256.0, WALL + 4.0];
    for _ in 0..10 {
        frame(&mut w, ToolState::erase(brush, 14.0));
    }
    r.shot(&w, "tools_erase");
    let after = w.sim.download_particles();
    let erased = ps.len() - after.len();
    r.check("erase: fluid under the brush is removed", format!("{} particles", erased), "> 800", erased > 800);
    run(&mut w, 2.0);
    let settled = w.sim.download_particles();
    r.check("erase: nothing else is lost afterwards", format!("{} -> {}", after.len(), settled.len()), "equal", after.len() == settled.len());

    // ---- slot recycling ----------------------------------------------------------------
    // A tank whose particle budget is nearly used up: new fluid has to reuse erased slots.
    let scene = Scene::new(GRID);
    let mut ps = Vec::new();
    fill_rect(&scene, [0.0, 0.0, 512.0, WALL + 12.0], Material::Water, &mut ps);
    let n0 = ps.len();
    let capacity = n0 as u32 + 3000;
    let mut w = World::with_capacity(gpu.clone(), scene, ps, capacity);
    apply(&mut w.sim.params);
    for _ in 0..10 {
        frame(&mut w, ToolState::erase(brush, 20.0));
    }
    let n1 = w.sim.download_particles().len();
    let want = 3000 + (n0 - n1) / 2;
    let spout = Nozzle::fixed(Material::Oil, [150.0, 60.0], 8.0, 24000.0);
    let mut emitted = 0;
    while emitted < want {
        emitted += w.pour(&spout, 4.0 * dt);
        frame(&mut w, ToolState::default());
    }
    let n2 = w.sim.download_particles().len();
    r.check(
        "recycle: pouring past the budget reuses erased slots",
        format!("{} + {} -> {}", n1, emitted, n2),
        &format!("{} (budget {})", n1 + emitted, capacity),
        n2 == n1 + emitted && w.sim.slots() == capacity,
    );
    for _ in 0..40 {
        w.pour(&spout, 4.0 * dt);
        frame(&mut w, ToolState::default());
    }
    let n3 = w.sim.download_particles().len();
    r.check("recycle: a full tank stops at its budget", format!("{}", n3), &format!("{}", capacity), n3 == capacity as usize);

    // ---- grab / stir -------------------------------------------------------------------
    let mut w = world(gpu, apply, &[([0.0, 0.0, 512.0, WALL + 60.0], Material::Water)]);
    let n0 = w.sim.download_particles().len();
    let (speed, radius) = (200.0, 12.0);
    let mut pos = [150.0, WALL + 30.0];
    for _ in 0..60 {
        pos[0] += speed * 4.0 * dt;
        frame(&mut w, ToolState::grab(pos, [speed, 0.0], radius));
    }
    r.shot(&w, "tools_grab_0");
    let ps = w.sim.download_particles();
    let near: Vec<f32> = ps.iter().filter(|p| (p.pos[0] - pos[0]).hypot(p.pos[1] - pos[1]) < radius * 0.5).map(|p| p.vel[0]).collect();
    let carried = near.iter().sum::<f32>() / near.len().max(1) as f32;
    r.check("grab: fluid under the brush follows it", format!("{:.0} cells/s", carried), &format!("> {:.0} (brush moves at {:.0})", 0.5 * speed, speed), carried > 0.5 * speed);
    // Water has no viscosity here, so the swirl and the waves are long-lived. What they
    // must not do is grow: kinetic plus potential energy can only fall.
    let sp = w.sim.params;
    let rest = sp.g_cells() as f64 * (WALL as f64 + 30.0);
    let total = |ps: &[Particle]| {
        let (ke, pe) = energy(ps, &sp);
        ke + pe - rest
    };
    let mut energies = vec![total(&ps)];
    for _ in 0..5 {
        run(&mut w, 2.0);
        energies.push(total(&w.sim.download_particles()));
    }
    r.shot(&w, "tools_grab_1");
    let ps = w.sim.download_particles();
    r.check(
        "grab: the released pool only loses energy",
        energies.iter().map(|e| format!("{e:.0}")).collect::<Vec<_>>().join(" > "),
        "falling (energy above rest, every 2 s)",
        energies.windows(2).all(|p| p[1] < p[0]) && ps.len() == n0,
    );

    // ---- solids ------------------------------------------------------------------------
    let mut w = world(gpu, apply, &[([0.0, 0.0, 512.0, WALL + 60.0], Material::Water)]);
    run(&mut w, 1.0);
    let before = w.sim.download_particles();
    // The mean particle height says where the fluid went and, unlike the highest
    // particles, barely notices the waves that moving solids leave behind.
    let mean = |ps: &[Particle]| mean_height(ps, |_| 1.0);
    let mean0 = mean(&before);
    let circle = Shape::Circle { c: [256.0, 40.0], r: 18.0 };
    w.add_shape(circle);
    w.add_shape(Shape::Box { c: [120.0, 30.0], half: [25.0, 6.0], angle: 0.5 });
    w.add_shape(Shape::Ramp { c: [400.0, 25.0], half: [30.0, 15.0], angle: 0.0, flip: false });
    // (area, height of the centroid) of each shape; all three sit fully under water.
    let shapes = [(std::f32::consts::PI * 18.0 * 18.0, 40.0), (50.0 * 12.0, 30.0), (0.5 * 60.0 * 30.0, 20.0)];
    let (width, area) = (GRID[0] as f32 - 2.0 * WALL, before.len() as f32 / PPC as f32);
    let displaced: f32 = shapes.iter().map(|s| s.0).sum();
    let depth = (area + displaced) / width;
    let moment: f32 = shapes.iter().map(|s| s.0 * (s.1 - WALL)).sum();
    let expected = WALL + (width * depth * depth * 0.5 - moment) / area;
    run(&mut w, 0.25);
    r.shot(&w, "tools_solids_0");
    run(&mut w, 3.75);
    r.shot(&w, "tools_solids_1");
    let ps = w.sim.download_particles();
    let crowded = |ps: &[Particle]| {
        let mut count = vec![0u32; (GRID[0] * GRID[1]) as usize];
        for p in ps {
            count[p.pos[1] as usize * GRID[0] as usize + p.pos[0] as usize] += 1;
        }
        count.into_iter().max().unwrap_or(0)
    };
    r.check("solids: placed shapes push the fluid out", format!("{} particles left inside", live_in(&ps, &w.scene, 1.0)), "0", live_in(&ps, &w.scene, 1.0) == 0);
    r.check("solids: no fluid is lost", format!("{} -> {}", before.len(), ps.len()), "equal", before.len() == ps.len());
    r.check(
        "solids: the pool rises by the displaced volume",
        format!("mean height {:.2} -> {:.2} cells", mean0, mean(&ps)),
        &format!("{:.2} (+/- 0.5), pool {:+.2} deeper", expected, depth - area / width),
        (mean(&ps) - expected).abs() < 0.5,
    );
    r.check("solids: displaced fluid does not clump", format!("fullest cell holds {}", crowded(&ps)), &format!("< 12 (rest {})", PPC), crowded(&ps) < 12);
    let removed = [[256.0, 40.0], [120.0, 30.0], [385.0, 18.0]].iter().filter(|&&p| w.remove_shape_at(p)).count();
    run(&mut w, 4.0);
    r.shot(&w, "tools_solids_2");
    let ps = w.sim.download_particles();
    let refill = ps.iter().filter(|p| circle.sdf(p.pos) < 0.0).count() as f32 / (std::f32::consts::PI * 18.0 * 18.0 * PPC as f32);
    r.check(
        "solids: removing them lets the fluid back in",
        format!("{} removed, hole {:.0} % full, mean height {:.2}", removed, refill * 100.0, mean(&ps)),
        &format!("3 removed, > 95 % full, {:.2} (+/- 0.5)", mean0),
        removed == 3 && w.scene.shapes.is_empty() && refill > 0.95 && (mean(&ps) - mean0).abs() < 0.5,
    );
    r.check("solids: the refilled pool does not clump", format!("fullest cell holds {}", crowded(&ps)), "< 12", crowded(&ps) < 12);
}

// ---- independent checks of viscosity, surface tension and settling ------------------

/// A world on an arbitrary grid from explicit particles.
fn world_from(gpu: &Arc<Gpu>, grid: [u32; 2], apply: &dyn Fn(&mut SimParams), ps: Vec<Particle>) -> World {
    let mut w = World::custom(gpu.clone(), Scene::new(grid), ps);
    apply(&mut w.sim.params);
    w.sim.step_blocking(0, ToolState::default());
    w
}

/// Rest-lattice particles inside `inside`, within the bounding box, with velocity field `vel`
/// and its gradient `grad` (du/dx, du/dy, dv/dx, dv/dy) as the APIC matrix.
fn particles_where(
    grid: [u32; 2],
    bbox: [f32; 4],
    m: Material,
    inside: &dyn Fn([f32; 2]) -> bool,
    vel: &dyn Fn([f32; 2]) -> [f32; 2],
    grad: &dyn Fn([f32; 2]) -> [f32; 4],
) -> Vec<Particle> {
    let scene = Scene::new(grid);
    let mut ps = Vec::new();
    fill_rect(&scene, bbox, m, &mut ps);
    ps.retain(|p| inside(p.pos));
    for p in &mut ps {
        p.vel = vel(p.pos);
        p.c = grad(p.pos);
    }
    ps
}

/// Least-squares amplitude of `component` of the particle velocity along `shape(pos)`,
/// over particles that `pick` selects.
fn mode_amplitude(ps: &[Particle], pick: &dyn Fn(&Particle) -> bool, shape: &dyn Fn([f32; 2]) -> f32, component: usize) -> f32 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for p in ps.iter().filter(|p| pick(p)) {
        let f = shape(p.pos) as f64;
        num += f * p.vel[component] as f64;
        den += f * f;
    }
    (num / den.max(1e-12)) as f32
}

/// Viscosity alone, against analytical decay rates and a conservation law.
///   1. A free slab in zero gravity with u(y) = A cos(pi y / H): traction-free top and bottom,
///      decays as exp(-nu (pi/H)^2 t).
///   2. The same slab resting on a no-slip floor with u(y) = A sin(pi y / 2H): decays as
///      exp(-nu (pi/2H)^2 t). This exercises the wall boundary condition.
///   3. A disc in rigid rotation: viscosity must not slow it (a Laplacian viscosity with
///      a zero-gradient free surface does).
fn viscosity(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nViscosity, analytical (zero gravity, nu = 0.01 m^2/s = 100 cells^2/s)");
    let nu_si = 0.01;
    let setup = |p: &mut SimParams| {
        apply(p);
        p.gravity = 0.0;
        p.nu = [nu_si; 3];
        p.sigma = [0.0; 3];
        p.sigma_oil = 0.0;
    };
    let mut params = SimParams::default();
    setup(&mut params);
    let nu = params.nu_cells(0);
    let (h, a) = (40.0f32, 20.0f32);
    let pi = std::f32::consts::PI;

    // 1. free slab
    let y0 = 100.0;
    let k = pi / h;
    let ps = particles_where(
        GRID,
        [80.0, y0, 432.0, y0 + h],
        Material::Water,
        &|_| true,
        &|x| [a * (k * (x[1] - y0)).cos(), 0.0],
        &|x| [0.0, -a * k * (k * (x[1] - y0)).sin(), 0.0, 0.0],
    );
    let mut w = world_from(gpu, GRID, &setup, ps);
    let mid = |p: &Particle| p.pos[0] > 180.0 && p.pos[0] < 332.0;
    let shape = |x: [f32; 2]| (k * (x[1] - y0)).cos();
    let a0 = mode_amplitude(&w.sim.download_particles(), &mid, &shape, 0);
    let t = 1.0;
    run(&mut w, t);
    let a1 = mode_amplitude(&w.sim.download_particles(), &mid, &shape, 0);
    let rate = (a0 / a1).ln() / t;
    let theory = nu * k * k;
    r.check(
        "free slab: decay rate of cos(pi y/H) shear",
        format!("{:.4} /s ({:+.1} %)", rate, (rate / theory - 1.0) * 100.0),
        &format!("{:.4} /s (+/- 5 %)", theory),
        (rate / theory - 1.0).abs() < 0.05,
    );

    // 2. slab on a no-slip floor
    let y0 = WALL;
    let k = pi / (2.0 * h);
    let ps = particles_where(
        GRID,
        [80.0, y0, 432.0, y0 + h],
        Material::Water,
        &|_| true,
        &|x| [a * (k * (x[1] - y0)).sin(), 0.0],
        &|x| [0.0, a * k * (k * (x[1] - y0)).cos(), 0.0, 0.0],
    );
    let mut w = world_from(gpu, GRID, &setup, ps);
    let shape = |x: [f32; 2]| (k * (x[1] - y0)).sin();
    let a0 = mode_amplitude(&w.sim.download_particles(), &mid, &shape, 0);
    let t = 2.0;
    run(&mut w, t);
    let a1 = mode_amplitude(&w.sim.download_particles(), &mid, &shape, 0);
    let rate = (a0 / a1).ln() / t;
    let theory = nu * k * k;
    r.check(
        "slab on no-slip floor: decay of sin(pi y/2H)",
        format!("{:.4} /s ({:+.1} %)", rate, (rate / theory - 1.0) * 100.0),
        &format!("{:.4} /s (+/- 10 %)", theory),
        (rate / theory - 1.0).abs() < 0.10,
    );

    // 3. rigid rotation
    let (c, rad, omega) = ([256.0f32, 144.0f32], 50.0f32, 2.0f32);
    let ps = particles_where(
        GRID,
        [c[0] - rad, c[1] - rad, c[0] + rad, c[1] + rad],
        Material::Water,
        &|x| (x[0] - c[0]).hypot(x[1] - c[1]) < rad,
        &|x| [-omega * (x[1] - c[1]), omega * (x[0] - c[0])],
        &|_| [0.0, -omega, omega, 0.0],
    );
    let mut w = world_from(gpu, GRID, &setup, ps);
    let angular = |ps: &[Particle]| {
        let n = ps.len() as f64;
        let cx = ps.iter().map(|p| p.pos[0] as f64).sum::<f64>() / n;
        let cy = ps.iter().map(|p| p.pos[1] as f64).sum::<f64>() / n;
        ps.iter().map(|p| (p.pos[0] as f64 - cx) * p.vel[1] as f64 - (p.pos[1] as f64 - cy) * p.vel[0] as f64).sum::<f64>()
    };
    let l0 = angular(&w.sim.download_particles());
    run(&mut w, 1.0);
    let l1 = angular(&w.sim.download_particles());
    // The previous solver's Laplacian viscosity, run on the same disc and converged,
    // keeps 0.919 of it after 1 s.
    r.check(
        "rigid rotation keeps its angular momentum",
        format!("{:.4} of the start after 1 s", l1 / l0),
        "> 0.98 (previous Laplacian form: 0.919)",
        l1 / l0 > 0.98,
    );
    let st = w.sim.solver;
    r.check(
        "viscosity solves converged",
        format!("{} unconverged, worst residual {:.1e}", st.viscosity_unconverged, st.viscosity_residual_max),
        "0",
        st.viscosity_unconverged == 0,
    );
}

/// Second moments of a particle cloud about its centroid: (Ixx - Iyy) / (Ixx + Iyy).
fn ellipticity(ps: &[Particle], pick: &dyn Fn(&Particle) -> bool) -> f32 {
    let sel: Vec<&Particle> = ps.iter().filter(|p| pick(p)).collect();
    let n = sel.len().max(1) as f64;
    let cx = sel.iter().map(|p| p.pos[0] as f64).sum::<f64>() / n;
    let cy = sel.iter().map(|p| p.pos[1] as f64).sum::<f64>() / n;
    let (mut ixx, mut iyy) = (0.0, 0.0);
    for p in &sel {
        ixx += (p.pos[0] as f64 - cx).powi(2);
        iyy += (p.pos[1] as f64 - cy).powi(2);
    }
    ((ixx - iyy) / (ixx + iyy).max(1e-12)) as f32
}

/// Period of the oscillation of `e(t)` from the times of its zero crossings.
fn period(samples: &[(f32, f32)]) -> Option<f32> {
    let mut crossings = Vec::new();
    for w in samples.windows(2) {
        let ((t0, e0), (t1, e1)) = (w[0], w[1]);
        if e0.signum() != e1.signum() && e0 != 0.0 {
            crossings.push(t0 + (t1 - t0) * e0 / (e0 - e1));
        }
    }
    if crossings.len() < 3 {
        return None;
    }
    Some(2.0 * (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f32)
}

/// Surface tension and wetting, on 1 mm cells where capillarity is resolved, zero gravity.
fn surface(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nSurface tension (1 mm cells, zero gravity unless stated)");
    let setup = |p: &mut SimParams| {
        apply(p);
        p.cell_size = 0.001;
        p.gravity = 0.0;
    };
    let mut params = SimParams::default();
    setup(&mut params);
    let u = params.gpu_units();
    let pi = std::f32::consts::PI;
    let c = [256.0f32, 144.0f32];
    let disc = |rad: f32| move |x: [f32; 2]| (x[0] - c[0]).hypot(x[1] - c[1]) < rad;
    let zero = |_: [f32; 2]| [0.0f32, 0.0];
    let zero4 = |_: [f32; 2]| [0.0f32; 4];

    // ---- Laplace pressure of a resting drop, and the spurious currents around it -------
    let rad = 30.0;
    let ps = particles_where(GRID, [c[0] - rad, c[1] - rad, c[0] + rad, c[1] + rad], Material::Water, &disc(rad), &zero, &zero4);
    let mut w = world_from(gpu, GRID, &setup, ps);
    run(&mut w, 0.3);
    let (pres, dt) = w.sim.download_pressure();
    let (mut sum, mut n) = (0.0f64, 0);
    for j in 0..GRID[1] as usize {
        for i in 0..GRID[0] as usize {
            if (i as f32 + 0.5 - c[0]).hypot(j as f32 + 0.5 - c[1]) < 0.5 * rad {
                sum += pres[j * GRID[0] as usize + i] as f64;
                n += 1;
            }
        }
    }
    let p_in = (sum / n.max(1) as f64) as f32 / dt;
    let p_theory = u.sigma[0] / rad;
    r.check(
        "drop: pressure inside / (sigma / R)",
        format!("{:.3}", p_in / p_theory),
        "1.0 (+/- 10 %)",
        (p_in / p_theory - 1.0).abs() < 0.10,
    );
    let ps = w.sim.download_particles();
    let v_cap = (u.sigma[0] / rad).sqrt();
    let rms = rms_speed(&ps);
    r.check(
        "drop: spurious currents (rms speed / sqrt(sigma/rho R))",
        format!("{:.4}", rms / v_cap),
        "< 0.02",
        rms / v_cap < 0.02,
    );

    // ---- the same for an oil disc inside water: jump of sigma_oa / R across the interface ----
    let rad = 30.0;
    let mut ps = particles_where(GRID, [0.0, 0.0, 512.0, 260.0], Material::Water, &|_| true, &zero, &zero4);
    for p in &mut ps {
        if disc(rad)(p.pos) {
            *p = Particle::new(Material::Oil, p.pos, [0.0, 0.0]);
        }
    }
    let mut w = world_from(gpu, GRID, &setup, ps);
    run(&mut w, 0.3);
    let (pres, dt) = w.sim.download_pressure();
    let ring = |r0: f32, r1: f32| {
        let (mut sum, mut n) = (0.0f64, 0);
        for j in 0..GRID[1] as usize {
            for i in 0..GRID[0] as usize {
                let d = (i as f32 + 0.5 - c[0]).hypot(j as f32 + 0.5 - c[1]);
                if d >= r0 && d < r1 {
                    sum += pres[j * GRID[0] as usize + i] as f64;
                    n += 1;
                }
            }
        }
        (sum / n.max(1) as f64) as f32 / dt
    };
    let jump = ring(0.0, 0.5 * rad) - ring(rad + 8.0, rad + 16.0);
    let theory = u.sigma_oil / rad;
    r.check(
        "oil disc in water: pressure jump / (sigma_oa / R)",
        format!("{:.3}", jump / theory),
        "1.0 (+/- 10 %)",
        (jump / theory - 1.0).abs() < 0.10,
    );

    // ---- capillary oscillation of a water drop: omega^2 = n (n^2 - 1) sigma / (rho R^3) -----
    let (rad, eps) = (20.0f32, 0.08f32);
    let (ax, ay) = (rad * (1.0 + eps), rad / (1.0 + eps));
    let ellipse = move |x: [f32; 2]| ((x[0] - c[0]) / ax).powi(2) + ((x[1] - c[1]) / ay).powi(2) < 1.0;
    let ps = particles_where(GRID, [c[0] - ax, c[1] - ay, c[0] + ax, c[1] + ay], Material::Water, &ellipse, &zero, &zero4);
    let mut w = world_from(gpu, GRID, &setup, ps);
    let omega = (6.0 * u.sigma[0] / rad.powi(3)).sqrt();
    let t_theory = 2.0 * pi / omega;
    let mut samples = Vec::new();
    let step = t_theory / 40.0;
    for k in 0..=100 {
        samples.push((k as f32 * step, ellipticity(&w.sim.download_particles(), &|_| true)));
        run(&mut w, step);
    }
    let t = period(&samples);
    r.check(
        "water drop, n = 2 capillary oscillation period",
        t.map_or("no oscillation".into(), |t| format!("{:.3} s ({:+.1} %)", t, (t / t_theory - 1.0) * 100.0)),
        &format!("{:.3} s (Rayleigh, +/- 10 %)", t_theory),
        t.is_some_and(|t| (t / t_theory - 1.0).abs() < 0.10),
    );

    // ---- oil drop in water: interfacial tension, omega^2 = 6 sigma / ((rho_i + rho_o) R^3) --
    // (A free surface at the top: a tank filled to the lid has no pressure reference.)
    let mut ps = particles_where(GRID, [0.0, 0.0, 512.0, 260.0], Material::Water, &|_| true, &zero, &zero4);
    for p in &mut ps {
        if ellipse(p.pos) {
            *p = Particle::new(Material::Oil, p.pos, [0.0, 0.0]);
        }
    }
    let mut w = world_from(gpu, GRID, &setup, ps);
    let rho_sum = params.rho[0] + params.rho[2];
    let omega = (6.0 * u.sigma_oil / (rho_sum * rad.powi(3))).sqrt();
    let t_theory = 2.0 * pi / omega;
    let mut samples = Vec::new();
    let step = t_theory / 40.0;
    let oil = |p: &Particle| p.kind == KIND_OIL;
    for k in 0..=90 {
        samples.push((k as f32 * step, ellipticity(&w.sim.download_particles(), &oil)));
        run(&mut w, step);
    }
    let t = period(&samples);
    // Reported, not asserted: the static jump across this interface is exact (above), but
    // the oscillation is off: +20 % with the real oil viscosity (60 cells^2/s at 1 mm cells,
    // which the inviscid formula ignores) and -30 % with both fluids made inviscid. The
    // dynamic error of the oil-aqueous interface is an open issue; see the README.
    println!(
        "  [info] oil drop in water, n = 2 oscillation period  {}  inviscid theory {:.3} s",
        t.map_or("no oscillation".into(), |t| format!("{:.3} s ({:+.1} %)", t, (t / t_theory - 1.0) * 100.0)),
        t_theory
    );

    // ---- miscible: a syrup disc in water feels no interfacial tension ---------------------
    let mut ps = particles_where(GRID, [0.0, 0.0, 512.0, 260.0], Material::Water, &|_| true, &zero, &zero4);
    for p in &mut ps {
        if ellipse(p.pos) {
            *p = Particle::new(Material::Syrup, p.pos, [0.0, 0.0]);
        }
    }
    let mut w = world_from(gpu, GRID, &|p: &mut SimParams| {
        setup(p);
        p.diffuse = 0.0;
    }, ps);
    let e0 = ellipticity(&w.sim.download_particles(), &|p: &Particle| p.kind == KIND_AQUEOUS && p.s > 0.5);
    run(&mut w, t_theory / 2.0);
    let e1 = ellipticity(&w.sim.download_particles(), &|p: &Particle| p.kind == KIND_AQUEOUS && p.s > 0.5);
    r.check(
        "syrup blob in water keeps its shape (miscible)",
        format!("ellipticity {:.3} -> {:.3}", e0, e1),
        "unchanged (+/- 0.01)",
        (e1 - e0).abs() < 0.01,
    );

    // ---- wetting: a drop on the floor settles at its contact angle -------------------------
    for angle in [60.0f32, 90.0, 120.0] {
        let rad = 30.0;
        let base = [256.0f32, WALL];
        let half = |x: [f32; 2]| (x[0] - base[0]).hypot(x[1] - base[1]) < rad;
        let ps = particles_where(GRID, [base[0] - rad, WALL, base[0] + rad, WALL + rad], Material::Water, &half, &zero, &zero4);
        let mut w = world_from(gpu, GRID, &|p: &mut SimParams| {
            setup(p);
            p.contact_angle = [angle; 3];
        }, ps);
        run(&mut w, 4.0);
        // Measured on the occupancy raster (cells holding at least half the rest count), so
        // that the odd stray particle on the floor does not count: the contiguous run of
        // liquid cells in the first row under the drop, and the tallest liquid cell above it.
        let ps = w.sim.download_particles();
        let (gw, gh) = (GRID[0] as usize, GRID[1] as usize);
        let mut count = vec![0u32; gw * gh];
        for p in &ps {
            count[p.pos[1] as usize * gw + p.pos[0] as usize] += 1;
        }
        let liquid = |i: usize, j: usize| count[j * gw + i] >= PPC / 2;
        let row = WALL as usize;
        let ci = base[0] as usize;
        let (mut lo, mut hi) = (ci, ci);
        while lo > 0 && liquid(lo - 1, row) {
            lo -= 1;
        }
        while hi + 1 < gw && liquid(hi + 1, row) {
            hi += 1;
        }
        let width = (hi + 1 - lo) as f32;
        let mut top = row;
        while top + 1 < gh && liquid(ci, top + 1) {
            top += 1;
        }
        let height = (top + 1 - row) as f32;
        let measured = 2.0 * (2.0 * height / width).atan().to_degrees();
        r.check(
            &format!("drop on the floor, contact angle {angle:.0} deg"),
            format!("{:.0} deg (cap {:.0} x {:.0} cells)", measured, width, height),
            &format!("{angle:.0} (+/- 15)"),
            (measured - angle).abs() < 15.0,
        );
        r.shot(&w, &format!("contact_{angle:.0}"));
    }

    // ---- beading: a square of oil in water rounds off -------------------------------------
    // Zero gravity, to isolate tension: with gravity a 3 cm blob at Re ~ 5000 rises as a
    // spherical cap, which is not a test of tension.
    let rounded = |sigma_oil: f32| {
        let mut ps = particles_where(GRID, [0.0, 0.0, 512.0, 200.0], Material::Water, &|_| true, &zero, &zero4);
        let sq = |x: [f32; 2]| (x[0] - 256.0).abs() < 15.0 && (x[1] - 60.0).abs() < 15.0;
        for p in &mut ps {
            if sq(p.pos) {
                *p = Particle::new(Material::Oil, p.pos, [0.0, 0.0]);
            }
        }
        let mut w = world_from(gpu, GRID, &|p: &mut SimParams| {
            setup(p);
            p.sigma_oil = sigma_oil;
        }, ps);
        run(&mut w, 0.5);
        let ps: Vec<Particle> = w.sim.download_particles().into_iter().filter(|p| p.kind == KIND_OIL).collect();
        let n = ps.len() as f32;
        let cx = ps.iter().map(|p| p.pos[0]).sum::<f32>() / n;
        let cy = ps.iter().map(|p| p.pos[1]).sum::<f32>() / n;
        let rmax = percentile(ps.iter().map(|p| (p.pos[0] - cx).hypot(p.pos[1] - cy)).collect(), 0.995);
        (rmax / (n / PPC as f32 / pi).sqrt(), w)
    };
    let (with, w) = rounded(params.sigma_oil);
    r.shot(&w, "oil_bead");
    let (without, _) = rounded(0.0);
    r.check(
        "oil square in water, 0.5 s: R_max / R_equal-area",
        format!("{:.3} with tension, {:.3} without", with, without),
        "rounder with tension (square 1.25, disc 1.0)",
        with < without - 0.1 && with < 1.2,
    );
}

/// Settling at a high particle count, and particle-scale jitter after a stir.
fn settle(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nSettling (1024 x 576, 250-cell deep pool, about 1 million particles, 6 s)");
    let grid = [1024, 576];
    let scene = Scene::new(grid);
    let mut ps = Vec::new();
    fill_rect(&scene, [0.0, 0.0, 1024.0, 250.0], Material::Water, &mut ps);
    let n0 = ps.len();
    let mut w = world_from(gpu, grid, apply, ps);
    let sp = w.sim.params;
    let e = |ps: &[Particle]| {
        let (ke, pe) = energy(ps, &sp);
        (ke, ke + pe)
    };
    let (mut worst_rise, mut last) = (0.0f64, e(&w.sim.download_particles()).1);
    let mut ke_peak = 0.0f64;
    let y0 = mean_height(&w.sim.download_particles(), |_| 1.0);
    for _ in 0..12 {
        run(&mut w, 0.5);
        let (ke, tot) = e(&w.sim.download_particles());
        worst_rise = worst_rise.max(tot - last);
        ke_peak = ke_peak.max(ke);
        last = tot;
    }
    let ps = w.sim.download_particles();
    let v_ref = (sp.g_cells() * 247.0).sqrt();
    r.check("particles conserved", format!("{} -> {}", n0, ps.len()), "equal", n0 == ps.len());
    r.check("peak kinetic energy per particle (cells^2/s^2)", format!("{:.4}", ke_peak), "< 0.05 (baseline: 162)", ke_peak < 0.05);
    r.check("rms speed after 6 s / sqrt(g h)", format!("{:.6}", rms_speed(&ps) / v_ref), "< 0.001", rms_speed(&ps) / v_ref < 0.001);
    r.check(
        "mean height drift",
        format!("{:+.3} cells", mean_height(&ps, |_| 1.0) - y0),
        "|drift| < 0.05 (baseline swung by 1.7)",
        (mean_height(&ps, |_| 1.0) - y0).abs() < 0.05,
    );
    // The density correction moves particles without momentum; a pool whose initial lattice
    // is slightly under-full is lifted by it, which shows as potential energy. Measure the
    // worst rise against the pool's potential energy (g times the mean height).
    let pe_scale = sp.g_cells() as f64 * y0 as f64;
    r.check(
        "total energy rise between 0.5 s samples / pool potential energy",
        format!("{:.1e}", worst_rise / pe_scale),
        "< 2e-4 (a 0.02 % volume correction)",
        worst_rise / pe_scale < 2e-4,
    );
    let st = w.sim.solver;
    r.check(
        "pressure solves converged",
        format!("{} unconverged of {} substeps, most iterations {}", st.pressure_unconverged, st.substeps, st.pressure_iters_max),
        "0",
        st.pressure_unconverged == 0,
    );

    println!("\nSloshing vs jitter after a stir (512 x 288, 60-cell pool)");
    let (rows, jitter) = stir_and_settle(gpu, apply);
    println!("  t (s)   energy above rest / particle   particle-scale jitter rms (cells/s)");
    for (t, ke, j) in &rows {
        println!("  {t:5.1}   {ke:28.2}   {j:8.3}");
    }
    let (ke_first, ke_last) = (rows[0].1, rows[rows.len() - 1].1);
    r.check(
        "after the stir: energy above rest falls, sloshing persists",
        format!("{:.0} -> {:.0}", ke_first, ke_last),
        "falling at every sample, not zero",
        rows.windows(2).all(|w| w[1].1 < w[0].1) && ke_last > 0.0,
    );
    // The same stir on the previous solver: energy above rest 2072 -> 1087, jitter
    // 9.15 -> 3.09 cells/s. Small eddies left by the stir are real; what must not happen is
    // particle noise that survives while the flow calms down.
    let share = 0.5 * (jitter as f64).powi(2) / ke_last.max(1e-9);
    r.check(
        "particle-scale jitter (rms of velocity minus its 4x4-cell mean)",
        format!("{:.2} -> {:.2} cells/s, {:.2} % of the energy", rows[0].2, jitter, share * 100.0),
        "decaying, < 2 % of the energy",
        rows.windows(2).all(|w| w[1].2 < w[0].2) && share < 0.02,
    );
}

/// Stirs a pool for 1 s, then samples kinetic energy and particle-scale velocity jitter for
/// 10 s. Jitter is the rms difference between each particle's velocity and the mean velocity
/// of its 4 x 4 cell block: sloshing waves are smooth at that scale, numerical noise is not.
/// Uses only `World::custom`, `step_blocking` and `download_particles`, so the same function
/// runs against older solvers.
fn stir_and_settle(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams)) -> (Vec<(f32, f64, f32)>, f32) {
    let mut w = world(gpu, apply, &[([0.0, 0.0, 512.0, WALL + 60.0], Material::Water)]);
    let (speed, radius) = (200.0, 12.0);
    let mut pos = [150.0, WALL + 30.0];
    for _ in 0..60 {
        pos[0] += speed * 4.0 / 240.0;
        w.sim.step_blocking(4, ToolState::grab(pos, [speed, 0.0], radius));
    }
    let sp = w.sim.params;
    let jitter = |ps: &[Particle]| {
        let (bw, bh) = (GRID[0] as usize / 4 + 1, GRID[1] as usize / 4 + 1);
        let mut sum = vec![[0.0f64; 3]; bw * bh];
        for p in ps {
            let k = (p.pos[1] as usize / 4) * bw + p.pos[0] as usize / 4;
            sum[k][0] += p.vel[0] as f64;
            sum[k][1] += p.vel[1] as f64;
            sum[k][2] += 1.0;
        }
        let mut acc = 0.0f64;
        for p in ps {
            let s = sum[(p.pos[1] as usize / 4) * bw + p.pos[0] as usize / 4];
            let (mx, my) = (s[0] / s[2], s[1] / s[2]);
            acc += (p.vel[0] as f64 - mx).powi(2) + (p.vel[1] as f64 - my).powi(2);
        }
        (acc / ps.len().max(1) as f64).sqrt() as f32
    };
    // Mechanical energy above the level pool at rest (same volume, flat surface).
    let rest = sp.g_cells() as f64 * (WALL as f64 + 30.0);
    let mut rows = Vec::new();
    for k in 0..=5 {
        let ps = w.sim.download_particles();
        let (ke, pe) = energy(&ps, &sp);
        rows.push((k as f32 * 2.0, ke + pe - rest, jitter(&ps)));
        if k < 5 {
            w.sim.step_blocking(480, ToolState::default());
        }
    }
    let j = rows[rows.len() - 1].2;
    (rows, j)
}

/// Dam break at three resolutions: the front must converge as the grid is refined.
fn convergence(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nDam break grid convergence (column a x 2a, a = 30, 60, 120 cells)");
    let mut fronts = Vec::new();
    for a in [30.0f32, 60.0, 120.0] {
        let grid = [(WALL + 5.0 * a + 40.0) as u32, (WALL + 2.0 * a + 40.0) as u32];
        let scene = Scene::new(grid);
        let mut ps = Vec::new();
        fill_rect(&scene, [0.0, 0.0, WALL + a, WALL + 2.0 * a], Material::Water, &mut ps);
        let mut w = world_from(gpu, grid, apply, ps);
        let g = w.sim.params.g_cells();
        let mut line = Vec::new();
        let mut t = 0.0;
        for tt in [1.19f32, 1.98, 2.65, 3.33] {
            let target = tt / (2.0 * g / a).sqrt();
            run(&mut w, target - t);
            t = target;
            let ps = w.sim.download_particles();
            line.push((percentile(ps.iter().map(|p| p.pos[0]).collect(), 0.999) - WALL) / a);
        }
        println!("  a = {a:>3}: Z at T = 1.19, 1.98, 2.65, 3.33: {}", line.iter().map(|z| format!("{z:.3}")).collect::<Vec<_>>().join("  "));
        fronts.push(line);
    }
    let d1 = (fronts[1][3] - fronts[0][3]).abs();
    let d2 = (fronts[2][3] - fronts[1][3]).abs();
    r.check(
        "front at T = 3.33: change on refinement",
        format!("30->60: {:.3}, 60->120: {:.3}", d1, d2),
        "shrinking, last < 2 % of Z",
        d2 < d1.max(0.02) && d2 < 0.02 * fronts[2][3],
    );
    // Lobovsky et al. (2014), table 1: measured mean front speed v / sqrt(g H) for t* > 1 ranges
    // from 1.14 to 1.74 across experiments (Martin & Moyce's own 2:1 run: 1.69).
    let speed = (fronts[2][3] - fronts[2][1]) / (3.33 - 1.98);
    r.check(
        "finest front speed dZ/dT vs published experiments",
        format!("{:.2}", speed),
        "within 1.14 .. 1.74 (Lobovsky 2014, table 1), < 2 (Ritter)",
        (1.14..=1.74).contains(&speed),
    );
}

type Test = fn(&Arc<Gpu>, &dyn Fn(&mut SimParams), &mut Report);

pub fn run_all(gpu: Arc<Gpu>, apply: &dyn Fn(&mut SimParams), only: Option<&str>, shots: Option<&str>) -> bool {
    println!("device: {}", gpu.device_name());
    let mut p = SimParams::default();
    apply(&mut p);
    println!(
        "grid {}x{}, cell {} m, {}, pressure tol {:e}, viscosity tol {:e}",
        GRID[0], GRID[1], p.cell_size,
        if p.fixed_hz > 0.0 { format!("fixed {} substeps/s", p.fixed_hz) } else { format!("CFL {} substeps", p.cfl) },
        p.pressure_tol, p.viscosity_tol
    );
    let shots = shots.map(|dir| {
        std::fs::create_dir_all(dir).expect("cannot create the shots directory");
        (Offscreen::new(gpu.clone(), [960, 540]), PathBuf::from(dir))
    });
    let mut r = Report { failures: 0, shots };
    let tests: [(&str, Test); 11] = [
        ("hydrostatic", hydrostatic),
        ("dam", dam_break),
        ("ritter", ritter),
        ("layers", stratification),
        ("viscosity", viscosity),
        ("viscous", viscous_current),
        ("buoyancy", buoyancy),
        ("surface", surface),
        ("settle", settle),
        ("convergence", convergence),
        ("tools", tools),
    ];
    for (name, test) in tests {
        if only.is_none_or(|o| o == name) {
            test(&gpu, apply, &mut r);
        }
    }
    println!("\n{}", if r.failures == 0 { "all checks passed".to_string() } else { format!("{} check(s) FAILED", r.failures) });
    r.failures == 0
}
