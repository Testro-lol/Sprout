//! Validation of the material system's physics (heat, phase change, rheology, suspensions,
//! magnetism, the 18 liquids): experiments with known answers, run on the real solver.
//! Registered in `validate::run_all` next to the original checks.

use std::sync::Arc;

use crate::{
    gpu::Gpu,
    maps::fill_rect,
    scene::{Scene, Shape, WALL},
    sim::{KIND_SRC, Material, PPC, Particle, SimParams, ToolState, heat_content},
    validate::{Report, percentile, run},
    world::World,
};

/// A world on `grid` with these solids (each held at a temperature, or insulating) and
/// particles, the parameters applied before anything runs.
fn build(gpu: &Arc<Gpu>, grid: [u32; 2], apply: &dyn Fn(&mut SimParams), shapes: Vec<(Shape, Option<f32>)>, ps: Vec<Particle>, tweak: &dyn Fn(&mut SimParams)) -> World {
    let mut scene = Scene::new(grid);
    for (s, t) in shapes {
        scene.shapes.push(s);
        scene.temps.push(t);
    }
    let mut w = World::custom(gpu.clone(), scene, Vec::new());
    apply(&mut w.sim.params);
    tweak(&mut w.sim.params);
    w.sim.reset(ps);
    w.sim.step_blocking(0, ToolState::default());
    w
}

/// Root of lambda e^(lambda^2) erf(lambda) = St / sqrt(pi) (one-phase Stefan problem).
fn stefan_lambda(st: f64) -> f64 {
    let erf = |x: f64| {
        // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t * (-x * x).exp();
        if x >= 0.0 { y } else { -y }
    };
    let f = |l: f64| l * (l * l).exp() * erf(l) - st / std::f64::consts::PI.sqrt();
    let (mut a, mut b) = (1e-6, 3.0);
    for _ in 0..100 {
        let m = 0.5 * (a + b);
        if f(m) > 0.0 { b = m } else { a = m }
    }
    0.5 * (a + b)
}

/// Heat conduction, melting and freezing: a Stefan melting front against its similarity
/// solution, the energy budget through the phase change, and a block of ice that floats as a
/// rigid body and melts in warm water.
pub fn thermal(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nHeat and phase change (Stefan melting front, energy budget, a floating ice block)");
    // ---- Stefan: ice at its melting point on a hot plate, 1 mm cells ----------------------
    let grid = [48, 160];
    let (plate_top, ice_top) = (WALL + 4.0, 130.0);
    let t_wall = 40.0f32;
    let rate = 100.0f32;
    let mut ice = Vec::new();
    fill_rect(&Scene::new(grid), [0.0, plate_top, grid[0] as f32, ice_top], Material::Water, &mut ice);
    let ice: Vec<Particle> = ice.into_iter().map(|p| p.at_temp(-0.5)).collect();
    let plate = Shape::Box { c: [grid[0] as f32 * 0.5, plate_top * 0.5], half: [grid[0] as f32, plate_top * 0.5], angle: 0.0 };
    let tweak = |p: &mut SimParams| {
        p.cell_size = 0.001;
        p.heat_rate = rate;
        // No exchange with the air above: the plate is the only source of heat.
        p.air_cooling = 0.0;
        p.mat[Material::Water.index()].emissivity = 0.0;
        // Ice as dense as water, so that the melt does not lift it off the plate.
        p.mat[Material::Water.index()].density_solid = p.mat[Material::Water.index()].density;
    };
    let mut w = build(gpu, grid, apply, vec![(plate, Some(t_wall))], ice, &tweak);
    let sp = w.sim.params;
    let e0 = heat_content(&w.sim.download_particles(), &sp);
    let water = sp.mat[Material::Water.index()];
    let t_melt = 0.5 * (water.melt_lo + water.melt_hi);
    let st = (water.heat_capacity * (t_wall - t_melt) / water.latent) as f64;
    let lambda = stefan_lambda(st);
    let alpha = (water.conductivity / water.heat_capacity_vol()) as f64;
    let width = grid[0] as f32 - 2.0 * WALL;
    let mut line = String::new();
    let mut worst = 0.0f64;
    let mut moved = 0.0f32;
    let mut elapsed = 0.0;
    for t in [2.5f64, 5.0, 10.0] {
        run(&mut w, (t - elapsed) as f32);
        elapsed = t;
        let ps = w.sim.download_particles();
        // Melted volume (each particle's melted share), not the particles past half melted:
        // particles of one cell share their heat, so a front cell melts its particles together.
        let melted = ps.iter().map(|q| 1.0 - sp.temperature(q).1 as f64).sum::<f64>();
        let front = melted / (PPC as f64 * width as f64);
        let theory = 2.0 * lambda * (alpha * rate as f64 * t).sqrt() / sp.cell_size as f64;
        worst = worst.max((front / theory - 1.0).abs());
        line += &format!("t={t}s {front:.2}/{theory:.2}  ");
        moved = moved.max(ps.iter().filter(|q| sp.temperature(q).1 > 0.5).map(|q| q.vel[0].hypot(q.vel[1])).fold(0.0, f32::max));
    }
    r.shot(&w, "stefan");
    println!("  melted depth in cells (measured/similarity solution, St {st:.3}, lambda {lambda:.4}): {line}");
    r.check("Stefan front vs similarity solution", format!("worst {:+.1} %", worst * 100.0), "within 10 %", worst < 0.10);
    let ps = w.sim.download_particles();
    let l = w.sim.download_ledger();
    let e1 = heat_content(&ps, &sp);
    let gained = l.net_in();
    let miss = ((e1 - e0) - gained) / gained.abs().max(1e-9);
    r.check(
        "energy: content change vs heat from the plate",
        format!("{:.4e} vs {:.4e} J/m ({:+.2} %)", e1 - e0, gained, miss * 100.0),
        "equal within 1 %",
        miss.abs() < 0.01,
    );
    let latent = ps.iter().filter(|q| sp.temperature(q).1 < 0.5).count() as f64 / PPC as f64 * (sp.cell_size as f64).powi(2) * (water.density * water.latent) as f64;
    r.check("energy: latent share of the heat taken up", format!("{:.0} %", latent / gained * 100.0), "most of it (St = 0.5: 1/(1 + St/2) ~ 80 %)", latent / gained > 0.6 && latent / gained < 1.0);
    r.check("rigid ice stays put", format!("fastest frozen particle {moved:.2} cells/s"), "< 0.5", moved < 0.5);

    // ---- a block of ice dropped into water at 20 deg C, 1 cm cells -------------------------
    let grid = [256, 144];
    let scene = Scene::new(grid);
    let mut ps = Vec::new();
    fill_rect(&scene, [0.0, 0.0, 256.0, 60.0], Material::Water, &mut ps);
    let mut block = Vec::new();
    // Two cells above where it floats, so that it settles with a small bob.
    let y0 = 44.0;
    fill_rect(&scene, [108.0, y0, 148.0, y0 + 20.0], Material::Water, &mut block);
    ps.retain(|p| !(p.pos[0] > 108.0 && p.pos[0] < 148.0 && p.pos[1] > y0));
    let n_block = block.len();
    ps.extend(block.into_iter().map(|p| p.at_temp(-10.0)));
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| p.heat_rate = 1.0);
    // Freeboard (height above the water far from the block), averaged over 5 s after 1 s.
    let mut freeboard = 0.0;
    for k in 0..60 {
        run(&mut w, 0.1);
        if k >= 10 {
            let sp = w.sim.params;
            let ps = w.sim.download_particles();
            let top = percentile(ps.iter().filter(|q| sp.temperature(q).1 > 0.5).map(|q| q.pos[1]).collect(), 0.99);
            let surface = percentile(ps.iter().filter(|q| sp.temperature(q).1 < 0.5 && (q.pos[0] < 70.0 || q.pos[0] > 186.0)).map(|q| q.pos[1]).collect(), 0.99);
            freeboard += (top - surface) / 50.0;
        }
    }
    r.shot(&w, "ice_block");
    let sp = w.sim.params;
    let ps = w.sim.download_particles();
    let frozen: Vec<&Particle> = ps.iter().filter(|q| sp.temperature(q).1 > 0.5).collect();
    let xs: Vec<f32> = frozen.iter().map(|q| q.pos[0]).collect();
    let ys: Vec<f32> = frozen.iter().map(|q| q.pos[1]).collect();
    let (wid, hei) = (percentile(xs.clone(), 0.99) - percentile(xs, 0.01), percentile(ys.clone(), 0.99) - percentile(ys.clone(), 0.01));
    let above = freeboard / 20.0;
    let water = sp.mat[Material::Water.index()];
    let expect = 1.0 - water.density_solid / water.density;
    r.check("ice block keeps its shape", format!("{wid:.1} x {hei:.1} cells (was 40 x 20), {} of {n_block} still frozen", frozen.len()), "within 2 cells", (wid - 40.0).abs() < 2.0 && (hei - 20.0).abs() < 2.0);
    r.check("ice block floats", format!("{:.1} % of its height above the water", above * 100.0), &format!("{:.1} % (1 - rho_ice / rho_water) +/- 4", expect * 100.0), (above - expect).abs() < 0.04);

    // ...and melts under the heat brush.
    let held = frozen.len();
    let centre = [128.0, mean_y(&frozen)];
    for _ in 0..120 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::heat(centre, 30.0, false));
    }
    r.shot(&w, "ice_melts");
    let left = w.sim.download_particles().iter().filter(|q| sp.temperature(q).1 > 0.5).count();
    r.check("ice melts under the heat brush (2 s)", format!("{held} -> {left} frozen particles"), "fewer than a third left", left * 3 < held);

    // ---- steam (README, "Steam") -----------------------------------------------------------
    // A pool 30 cells deep heated from below by the brush: at 99 deg C bubbles rise through it
    // and escape at the surface; at 20 deg C they condense on the cold water around the brush.
    let grid = [128, 96];
    let brush = ToolState::heat([64.0, WALL + 5.0], 12.0, false);
    let pool = |t: f32| self::block(grid, [0.0, WALL, 128.0, WALL + 30.0], Material::Water).into_iter().map(|p| p.at_temp(t)).collect::<Vec<_>>();
    let steam = |w: &World| w.sim.download_particles().into_iter().filter(|p| p.is(Material::Steam)).collect::<Vec<_>>();
    let mut w = build(gpu, grid, apply, Vec::new(), pool(99.0), &|_| {});
    let sp = w.sim.params;
    let e0 = heat_content(&w.sim.download_particles(), &sp);
    let mut top = 0.0f32;
    for k in 0..120 {
        w.sim.advance_blocking(1.0 / 60.0, brush);
        if k % 10 == 9 {
            top = steam(&w).iter().map(|p| p.pos[1]).fold(top, f32::max);
        }
    }
    r.shot(&w, "steam_boils");
    let l = w.sim.download_ledger();
    let (turned, escaped) = (l.steam_count[Material::Water.index()], l.boiled_count[Material::Steam.index()]);
    r.check(
        "water at 99 deg C over the brush boils: bubbles rise, steam escapes",
        format!("{turned} particles turned to steam, {escaped} escaped; highest bubble {:.0} cells above the brush", top - WALL - 5.0),
        "some escape; bubbles reach the surface (25 cells up)",
        escaped > 0 && top > WALL + 28.0,
    );
    for _ in 0..120 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
    }
    let rest = steam(&w);
    let left = rest.len();
    // (Where any is left, and at what temperature: a diagnosis when it fails.)
    let at: Vec<String> = rest.iter().take(3).map(|p| format!("({:.1}, {:.1}) {:.1} C", p.pos[0], p.pos[1], sp.temperature(p).0)).collect();
    r.check("...2 s after the brush is lifted the steam is gone", format!("{left} steam particles left {}", at.join(" ")), "none (escaped or condensed)", left == 0);
    let l = w.sim.download_ledger();
    let e1 = heat_content(&w.sim.download_particles(), &sp);
    let miss = ((e1 - e0) - l.net_in()) / l.heated.max(1e-9);
    r.check(
        "...energy: content change vs brush in, air and steam out",
        format!("{:+.4e} vs {:+.4e} J/m ({:+.3} % of the brush's heat)", e1 - e0, l.net_in(), miss * 100.0),
        "equal within 0.5 %",
        miss.abs() < 0.005,
    );
    let mut w = build(gpu, grid, apply, Vec::new(), pool(20.0), &|_| {});
    // How far bubbles rise before they condense: the mean height above the brush of the
    // steam present, sampled every frame of the last second.
    let (mut rise, mut nrise) = (0.0f64, 0usize);
    for k in 0..90 {
        w.sim.advance_blocking(1.0 / 60.0, brush);
        if k >= 30 {
            for p in steam(&w) {
                rise += (p.pos[1] - WALL - 5.0).max(0.0) as f64;
                nrise += 1;
            }
        }
    }
    println!("  bubbles in 20 deg C water: mean height above the brush {:.1} cells", rise / nrise.max(1) as f64);
    let l = w.sim.download_ledger();
    let (turned, escaped) = (l.steam_count[Material::Water.index()], l.boiled_count[Material::Steam.index()]);
    r.check(
        "water at 20 deg C over the brush: bubbles condense before the surface",
        format!("{turned} turned to steam, {escaped} escaped"),
        "some turn, under 5 % of them escape",
        turned > 0 && (escaped as f32) < 0.05 * turned as f32,
    );
}

fn mean_y(ps: &[&Particle]) -> f32 {
    ps.iter().map(|q| q.pos[1]).sum::<f32>() / ps.len().max(1) as f32
}

/// Particles of `m` filling `rect` (cells), with the material's emitted state.
fn block(grid: [u32; 2], rect: [f32; 4], m: Material) -> Vec<Particle> {
    let mut ps = Vec::new();
    fill_rect(&Scene::new(grid), rect, m, &mut ps);
    ps
}

/// Mean vertical velocity of particles in `band` (y range), binned by x over `[x0, x1)`.
fn profile(ps: &[Particle], band: [f32; 2], x0: f32, x1: f32, bins: usize) -> Vec<f32> {
    let mut sum = vec![0.0f64; bins];
    let mut n = vec![0u32; bins];
    for p in ps.iter().filter(|p| p.pos[1] >= band[0] && p.pos[1] < band[1] && p.pos[0] >= x0 && p.pos[0] < x1) {
        let k = (((p.pos[0] - x0) / (x1 - x0)) * bins as f32) as usize;
        sum[k.min(bins - 1)] += p.vel[1] as f64;
        n[k.min(bins - 1)] += 1;
    }
    sum.iter().zip(&n).map(|(s, &n)| if n > 0 { (*s / n as f64) as f32 } else { 0.0 }).collect()
}

/// Relative L2 difference between a measured and a predicted profile.
fn l2(measured: &[f32], theory: &[f32]) -> f32 {
    let num: f32 = measured.iter().zip(theory).map(|(m, t)| (m - t) * (m - t)).sum();
    let den: f32 = theory.iter().map(|t| t * t).sum();
    (num / den.max(1e-12)).sqrt()
}

/// Yield stress and shear-rate dependent viscosity: a Herschel-Bulkley slump against the
/// static shape yield-stress theory predicts, and gravity-driven flow down a vertical slot
/// against the Bingham and Carreau velocity profiles.
pub fn rheology(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nRheology (clay slump vs yield-stress theory; Bingham and Carreau slot flow)");
    // ---- slump: a 20 x 20 cm column of clay on a 1 cm grid ------------------------------
    let grid = [256, 144];
    let (side, x_mid) = (20.0f32, 128.0f32);
    let ps = block(grid, [x_mid - side * 0.5, 0.0, x_mid + side * 0.5, WALL + side], Material::Clay);
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|_| {});
    run(&mut w, 8.0);
    r.shot(&w, "slump");
    let sp = w.sim.params;
    let ps = w.sim.download_particles();
    let clay = sp.mat[Material::Clay.index()];
    // Static shape of a yield-stress layer (Liu & Mei 1989; Coussot 1997):
    // tau_y = rho g h |dh/dx|, h(x) = sqrt(2 tau_y (L - x) / (rho g)); its area fixes L.
    let lc = 2.0 * clay.yield_stress / (clay.density * sp.gravity) / sp.cell_size;
    let half_area = side * side * 0.5;
    let run_out = (1.5 * half_area / lc.sqrt()).powf(2.0 / 3.0);
    let peak = (lc * run_out).sqrt();
    let xs: Vec<f32> = ps.iter().map(|p| p.pos[0]).collect();
    let measured_run = 0.5 * (percentile(xs.clone(), 0.998) - percentile(xs, 0.002));
    let measured_peak = percentile(ps.iter().filter(|p| (p.pos[0] - x_mid).abs() < 3.0).map(|p| p.pos[1]).collect(), 0.99) + 0.25 - WALL;
    let speed = (ps.iter().map(|p| p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sum::<f32>() / ps.len() as f32).sqrt();
    r.check("slump: run-out (half width)", format!("{measured_run:.1} cells"), &format!("{run_out:.1} (yield-stress theory, +/- 20 %)"), (measured_run / run_out - 1.0).abs() < 0.2);
    r.check("slump: height at the centre", format!("{measured_peak:.1} cells"), &format!("{peak:.1} (+/- 25 %)"), (measured_peak / peak - 1.0).abs() < 0.25);
    r.check("slump: it stops (rms speed after 8 s)", format!("{speed:.3} cells/s"), "< 0.5", speed < 0.5);

    // ---- slot flow, 1 mm cells: a column falling between two vertical walls -----------------
    let grid = [64, 320];
    let (a, x_c) = (20.0f32, 32.0f32);
    let unconverged = std::cell::Cell::new(0u64);
    let slot = |gpu: &Arc<Gpu>, m: Material, a: f32, t: f32, tweak: &dyn Fn(&mut SimParams)| -> (Vec<f32>, SimParams) {
        let walls = vec![
            (Shape::Box { c: [x_c - a - 5.0, 190.0], half: [5.0, 130.0], angle: 0.0 }, None),
            (Shape::Box { c: [x_c + a + 5.0, 190.0], half: [5.0, 130.0], angle: 0.0 }, None),
        ];
        // Air above the column: liquid up to the lid would hang there, held by the vacuum.
        let ps = block(grid, [x_c - a, 80.0, x_c + a, 290.0], m);
        let mut w = build(gpu, grid, apply, walls, ps, &|p: &mut SimParams| {
            p.cell_size = 0.001;
            for mm in p.mat.iter_mut() {
                mm.tension = 0.0;
            }
            tweak(p);
        });
        run(&mut w, t);
        unconverged.set(unconverged.get() + w.sim.solver.viscosity_unconverged);
        let ps = w.sim.download_particles();
        (profile(&ps, [150.0, 220.0], x_c - a, x_c + a, 20), w.sim.params)
    };
    // Bingham: plug of half-width y_p = tau_y / (rho g) in the middle, parabola outside it.
    let (bing, sp) = slot(gpu, Material::Clay, a, 0.15, &|p: &mut SimParams| {
        let m = &mut p.mat[Material::Clay.index()];
        m.flow_index = 1.0;
        m.yield_stress = 20.0;
        m.viscosity = 0.02;
    });
    let m = sp.mat[Material::Clay.index()];
    let g = sp.gravity / sp.cell_size;
    let nu = m.viscosity / (sp.cell_size * sp.cell_size);
    let yp = m.yield_stress / (m.density * sp.gravity) / sp.cell_size;
    let bingham = |xi: f32| -g / (2.0 * nu) * ((a - yp).powi(2) - (xi.max(yp) - yp).powi(2));
    let theory: Vec<f32> = (0..20).map(|k| bingham((x_c - a + (k as f32 + 0.5) * 2.0 - x_c).abs())).collect();
    let err = l2(&bing, &theory);
    println!("  Bingham slot, v (cells/s) measured/theory by x: {}", bing.iter().zip(&theory).step_by(2).map(|(m, t)| format!("{m:.1}/{t:.1}")).collect::<Vec<_>>().join(" "));
    r.check("Bingham slot flow vs plug + parabola", format!("L2 error {:.1} %, plug {:.1} cells wide", err * 100.0, 2.0 * yp), "< 10 %", err < 0.10);
    // Carreau: stress tau = rho g xi across the slot; solve mu(g) g = tau for the shear rate.
    // A narrower slot and a thinner liquid than the Bingham case, so that the viscosity
    // solve converges within its default iterations (its cost grows with nu dt / dx^2).
    let a = 10.0f32;
    let (car, sp) = slot(gpu, Material::Blood, a, 0.25, &|p: &mut SimParams| {
        let m = &mut p.mat[Material::Blood.index()];
        m.viscosity = 0.005;
        m.shear_time = 0.5;
        m.flow_index = 0.5;
        m.visc_ratio = 0.1;
    });
    let m = sp.mat[Material::Blood.index()];
    let nu0 = m.viscosity / (sp.cell_size * sp.cell_size);
    let carreau = |gd: f32| nu0 * (m.visc_ratio + (1.0 - m.visc_ratio) * (1.0 + (m.shear_time * gd).powi(2)).powf(0.5 * (m.flow_index - 1.0))) * gd;
    let rate = |xi: f32| {
        let (mut lo, mut hi) = (0.0f32, 1e5f32);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if carreau(mid) < g * xi { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    };
    let steps = 400;
    let speed_at = |xi: f32| -> f32 { -(0..steps).map(|k| xi + (a - xi) * (k as f32 + 0.5) / steps as f32).map(rate).sum::<f32>() * (a - xi) / steps as f32 };
    let theory: Vec<f32> = (0..20).map(|k| speed_at((x_c - a + (k as f32 + 0.5) * (2.0 * a / 20.0) - x_c).abs())).collect();
    let err = l2(&car, &theory);
    println!("  Carreau slot, v (cells/s) measured/theory by x: {}", car.iter().zip(&theory).step_by(2).map(|(m, t)| format!("{m:.1}/{t:.1}")).collect::<Vec<_>>().join(" "));
    r.check("Carreau slot flow vs its profile", format!("L2 error {:.1} %", err * 100.0), "< 10 %", err < 0.10);
    r.check("slot flows: viscosity solves converged", format!("{} unconverged", unconverged.get()), "0", unconverged.get() == 0);
}

/// Suspensions and dissolved content: grains settling against Stokes, brine's lower
/// freezing point, and the salt evaporating brine leaves behind.
pub fn suspension(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nSuspensions and solutions (settling vs Stokes, brine freezing point, evaporation)");
    let grid = [128, 128];
    let ps = block(grid, [0.0, 0.0, 128.0, 100.0], Material::Mud);
    let load = 0.1;
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| {
        p.cell_size = 0.001;
        p.mat[Material::Mud.index()].load = load;
    });
    // Centre of the concentrate (each particle weighted by its concentration).
    let mean = |ps: &[Particle]| ps.iter().map(|p| p.s as f64 * p.pos[1] as f64).sum::<f64>() / ps.iter().map(|p| p.s as f64).sum::<f64>().max(1e-9);
    let y0 = mean(&w.sim.download_particles());
    let t = 4.0;
    run(&mut w, t);
    r.shot(&w, "settling");
    let ps = w.sim.download_particles();
    let sp = w.sim.params;
    let mud = sp.mat[Material::Mud.index()];
    let water = sp.mat[Material::Water.index()];
    let drho = (mud.density - water.density) as f64;
    let stokes = drho * sp.gravity as f64 * (mud.grain as f64).powi(2) / (18.0 * water.viscosity as f64 * water.density as f64);
    let conc = load as f64;
    let expect = stokes * (1.0 - conc) / sp.cell_size as f64;
    let v = (y0 - mean(&ps)) / t as f64;
    r.check(
        "grains settle at the hindered Stokes speed",
        format!("{v:.2} cells/s ({:+.1} %)", (v / expect - 1.0) * 100.0),
        &format!("{expect:.2} = v_Stokes (1 - {conc:.2}) (+/- 15 %)"),
        (v / expect - 1.0).abs() < 0.15,
    );
    let top: Vec<f32> = ps.iter().filter(|p| p.pos[1] > 97.0).map(|p| p.s).collect();
    let clear = top.iter().sum::<f32>() / top.len().max(1) as f32;
    let total: f64 = ps.iter().map(|p| p.s as f64).sum();
    let start = load as f64 * ps.len() as f64;
    r.check("a clear layer forms at the top", format!("concentration {clear:.3} in the top 3 cells"), &format!("< {:.3} (it was {load})", 0.3 * load), clear < 0.3 * load);
    r.check("concentrate conserved", format!("{:+.4} %", (total / start - 1.0) * 100.0), "|drift| < 0.1 %", (total / start - 1.0).abs() < 1e-3);

    // ---- brine stays liquid where water freezes ---------------------------------------------
    // Two 6 mm layers (1 mm cells) side by side, under air at -15 deg C: the ice grows about
    // 16 mm in the 20 minutes of physical time this covers.
    let grid = [128, 72];
    let mut ps = block(grid, [0.0, 0.0, 62.0, WALL + 6.0], Material::Water);
    ps.extend(block(grid, [66.0, 0.0, 128.0, WALL + 6.0], Material::Brine));
    let wall = Shape::Box { c: [64.0, 20.0], half: [2.0, 20.0], angle: 0.0 };
    let mut w = build(gpu, grid, apply, vec![(wall, None)], ps, &|p: &mut SimParams| {
        p.cell_size = 0.001;
        p.ambient_temp = -15.0;
        p.heat_rate = 200.0;
        p.air_cooling = 300.0;
    });
    run(&mut w, 6.0);
    r.shot(&w, "brine_freezing");
    let sp = w.sim.params;
    let ps = w.sim.download_particles();
    let frozen = |m: Material| {
        let all: Vec<&Particle> = ps.iter().filter(|p| p.is(m)).collect();
        all.iter().filter(|p| sp.temperature(p).1 > 0.5).count() as f32 / all.len().max(1) as f32
    };
    let t_brine = ps.iter().filter(|p| p.is(Material::Brine)).map(|p| sp.temperature(p).0).sum::<f32>() / ps.iter().filter(|p| p.is(Material::Brine)).count().max(1) as f32;
    r.check("air at -15 deg C: water freezes", format!("{:.0} % frozen", frozen(Material::Water) * 100.0), "> 50 %", frozen(Material::Water) > 0.5);
    r.check("...brine (freezes at -21 deg C) does not", format!("{:.0} % frozen, mean {t_brine:.1} deg C", frozen(Material::Brine) * 100.0), "0 %, below 0 deg C", frozen(Material::Brine) == 0.0 && t_brine < 0.0);

    // ---- evaporating brine leaves salt ---------------------------------------------------------
    let ps = block(grid, [20.0, 0.0, 108.0, 20.0], Material::Brine);
    // Dissolved salt in the liquid, as volume of brine at concentration 1: in the brine, and
    // in the bubbles of steam it boiled into (they condense back into brine).
    let from_brine = (Material::Brine.index() as u32 + 1) << 8;
    let dissolved = |ps: &[Particle]| {
        ps.iter().filter(|p| p.is(Material::Brine) || (p.is(Material::Steam) && p.kind & KIND_SRC == from_brine)).map(|p| p.s as f64).sum::<f64>()
    };
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|_| {});
    let s0 = dissolved(&w.sim.download_particles());
    for _ in 0..240 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::heat([64.0, 8.0], 30.0, false));
    }
    let l = w.sim.download_ledger();
    // What boiled leaves its salt behind, one salt particle per `residue_share` of dissolved
    // salt; the water escapes as steam (README, "Steam") or condenses and rejoins the brine.
    let escaped = l.boiled_count[Material::Steam.index()] + l.boiled_count[Material::Brine.index()] - l.residue_count[Material::Salt.index()];
    let salt = l.residue_count[Material::Salt.index()];
    let ps = w.sim.download_particles();
    let live_salt = ps.iter().filter(|p| p.is(Material::Salt)).count();
    let share = w.sim.params.mat[Material::Brine.index()].residue_share as f64;
    let lost = s0 - dissolved(&ps);
    let ratio = salt as f64 / (share * lost).max(1e-9);
    r.shot(&w, "brine_boils");
    r.check(
        "boiling brine: water lost, salt left",
        format!("{escaped} particles escaped as steam; {lost:.0} of {s0:.0} parts of dissolved salt boiled out, {salt} salt particles ({live_salt} in the tank): {:.2} x expected", ratio),
        &format!("some escaped; salt = residue share ({:.1} %) x what boiled out, +/- 15 %", share * 100.0),
        escaped > 0 && (ratio - 1.0).abs() < 0.15 && live_salt as u32 == salt,
    );
}

/// Viscoelasticity: the fundamental shear mode of a slime layer on a no-slip floor rings
/// and relaxes as a Maxwell liquid's should.
pub fn elastic(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nViscoelasticity (slime: Maxwell shear mode of a free layer, zero gravity)");
    let grid = [512, 144];
    let (h, amp) = (30.0f32, 20.0f32);
    // A free layer (no walls): the stress vanishes at both free surfaces by itself. (On a
    // no-slip wall the particles' velocity gradient misses the wall shear, so the elastic
    // stress is not anchored there and only the viscosity holds slime to walls: README.)
    let free = true;
    let k = std::f32::consts::PI / if free { h } else { 2.0 * h };
    let y0 = if free { 50.0 } else { WALL };
    let ps = crate::validate::particles_where(
        grid,
        [60.0, y0, 452.0, y0 + h],
        Material::Slime,
        &|_| true,
        &|x| if free { [amp * (k * (x[1] - y0)).cos(), 0.0] } else { [amp * (k * (x[1] - y0)).sin(), 0.0] },
        &|x| if free { [0.0, -amp * k * (k * (x[1] - y0)).sin(), 0.0, 0.0] } else { [0.0, amp * k * (k * (x[1] - y0)).cos(), 0.0, 0.0] },
    );
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| {
        p.gravity = 0.0;
        p.mat[Material::Slime.index()].tension = 0.0;
    });
    let mid = |p: &Particle| p.pos[0] > 160.0 && p.pos[0] < 352.0;
    let shape = |x: [f32; 2]| if free { (k * (x[1] - y0)).cos() } else { (k * (x[1] - y0)).sin() };
    let mut samples = Vec::new();
    for i in 0..=100 {
        let a = crate::validate::mode_amplitude(&w.sim.download_particles(), &mid, &shape, 0);
        samples.push((i as f32 * 0.08, a));
        run(&mut w, 0.08);
    }
    r.shot(&w, "slime_mode");
    let sp = w.sim.params;
    let m = sp.mat[Material::Slime.index()];
    let kk = (k / sp.cell_size).powi(2);
    let gamma = 0.5 * (1.0 / m.relax_time + m.viscosity * kk);
    let w0 = m.shear_modulus / m.density * kk + m.viscosity * kk / m.relax_time;
    let omega = (w0 - gamma * gamma).max(0.0).sqrt();
    let t_theory = 2.0 * std::f32::consts::PI / omega;
    let measured = crate::validate::period(&samples);
    // Decay: ratio of successive extremes of |a|.
    let peaks: Vec<(f32, f32)> = samples.windows(3).filter(|w| w[1].1.abs() > w[0].1.abs() && w[1].1.abs() >= w[2].1.abs() && w[1].1.abs() > 0.05 * amp).map(|w| (w[1].0, w[1].1.abs())).collect();
    let decay = if peaks.len() >= 2 { (peaks[0].1 / peaks[peaks.len() - 1].1).ln() / (peaks[peaks.len() - 1].0 - peaks[0].0) } else { 0.0 };
    match measured {
        Some(t) => r.check("period of the shear mode", format!("{t:.2} s ({:+.1} %)", (t / t_theory - 1.0) * 100.0), &format!("{t_theory:.2} s (Maxwell: G = {} Pa, lambda = {} s; +/- 10 %)", m.shear_modulus, m.relax_time), (t / t_theory - 1.0).abs() < 0.10),
        None => r.check("period of the shear mode", "no oscillation".into(), &format!("{t_theory:.2} s"), false),
    }
    r.check("decay rate (relaxation)", format!("{decay:.3} /s ({:+.1} %)", (decay / gamma - 1.0) * 100.0), &format!("{gamma:.3} /s = (1/lambda + nu k^2) / 2 (+/- 20 %)"), (decay / gamma - 1.0).abs() < 0.2);

    // ---- a block of slime dropped into a cup with a peg (gallery `slime`) ---------------------
    // It splashes, but holds together: with a 1 Pa s solvent it flung its impact back out as
    // sheets that tore into hundreds of droplets (README, "Viscoelasticity").
    let grid = [512, 288];
    let wall = |x: f32| (Shape::Box { c: [x, 40.0], half: [4.0, 38.0], angle: 0.0 }, None);
    let peg = (Shape::Circle { c: [236.0, 20.0], r: 8.0 }, None);
    let ps = block(grid, [226.0, 40.0, 296.0, 90.0], Material::Slime);
    let n0 = ps.len();
    let mut w = build(gpu, grid, apply, vec![wall(176.0), wall(336.0), peg], ps, &|_| {});
    let mut worst = 0;
    for _ in 0..8 {
        run(&mut w, 0.4);
        worst = worst.max(crate::headless::blob_stats(&w.sim.download_particles(), grid).2);
    }
    r.shot(&w, "slime_drop");
    let (pieces, _, dropped) = crate::headless::blob_stats(&w.sim.download_particles(), grid);
    r.check(
        "slime dropped into a cup holds together",
        format!("after 3.2 s {pieces} pieces, {dropped} particles in droplets (most at any time {worst}) of {n0}"),
        "droplets under 0.5 % after 3.2 s, under 2 % at any time",
        (dropped as f32) < 0.005 * n0 as f32 && (worst as f32) < 0.02 * n0 as f32,
    );
}

/// Liquid metals and magnetism: layering by density with mercury at 13.5 times water's
/// density, a mercury puddle's height and contact angle, a ferrofluid drawn to a magnet, and
/// gallium's oxide skin holding a shape.
pub fn metals(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nDensity extremes, liquid metals and magnetism");
    // ---- layering: the lightest at the bottom, mercury on top ---------------------------------
    let grid = [256, 144];
    let mut ps = block(grid, [0.0, 0.0, 256.0, WALL + 12.0], Material::Wax);
    ps.extend(block(grid, [0.0, WALL + 12.0, 256.0, WALL + 24.0], Material::Oil));
    ps.extend(block(grid, [0.0, WALL + 24.0, 256.0, WALL + 36.0], Material::Water));
    ps.extend(block(grid, [0.0, WALL + 36.0, 256.0, WALL + 40.0], Material::Mercury));
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|_| {});
    run(&mut w, 20.0);
    r.shot(&w, "layers_metal");
    let ps = w.sim.download_particles();
    let h = |m: Material| crate::validate::mean_height(&ps, |p| p.is(m) as u32 as f32);
    let (hm, hw, ho, hx) = (h(Material::Mercury), h(Material::Water), h(Material::Oil), h(Material::Wax));
    let st = w.sim.solver;
    r.check("layers: mercury / water / oil / molten wax", format!("{hm:.0} / {hw:.0} / {ho:.0} / {hx:.0} cells"), "rising in that order", hm < hw && hw < ho && ho < hx);
    r.check("layers: pressure solves with a 13.5 density ratio", format!("{} unconverged of {}, most iterations {}", st.pressure_unconverged, st.substeps, st.pressure_iters_max), "0", st.pressure_unconverged == 0);

    // ---- one phase group, two densities, stirred finer than a cell: molten wax and oil, and
    // gallium and mercury, on a checkerboard of single cells, each pair at one temperature at
    // which both are liquid (one group shares one temperature per cell: wax at 75 °C beside oil
    // at 20 °C would meet at ~47 °C and set). With no interfacial tension between them nothing
    // at the grid's scale tells them apart; the drift within a group (particles.comp) has to
    // sort them. ------------------------------------------------------------------------------
    let grid = [256, 144];
    let checker = |rect: [f32; 4], a: Material, b: Material, t: f32| -> Vec<Particle> {
        let odd = |p: &Particle| (p.pos[0].floor() as i32 + p.pos[1].floor() as i32) & 1 == 1;
        let mut ps: Vec<Particle> = block(grid, rect, a).into_iter().filter(|p| !odd(p)).collect();
        ps.extend(block(grid, rect, b).into_iter().filter(odd));
        ps.into_iter().map(|p| p.at_temp(t)).collect()
    };
    let mut ps = checker([0.0, 0.0, 124.0, WALL + 30.0], Material::Wax, Material::Oil, 75.0);
    ps.extend(checker([132.0, 0.0, 256.0, WALL + 10.0], Material::Gallium, Material::Mercury, 40.0));
    let wall = Shape::Box { c: [128.0, 30.0], half: [4.0, 30.0], angle: 0.0 };
    let mut w = build(gpu, grid, apply, vec![(wall, None)], ps, &|p: &mut SimParams| p.heat_rate = 0.0);
    run(&mut w, 10.0);
    r.shot(&w, "sort_in_group");
    let ps = w.sim.download_particles();
    // Share of the light material among the upper half of the pair's particles: 1 when
    // fully sorted, 0.5 when still stirred.
    let sorted = |light: Material, heavy: Material| -> f64 {
        let mut hs: Vec<(f32, bool)> = ps.iter().filter(|p| p.is(light) || p.is(heavy)).map(|p| (p.pos[1], p.is(light))).collect();
        hs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let n = hs.iter().filter(|h| h.1).count();
        hs[..n].iter().filter(|h| h.1).count() as f64 / n.max(1) as f64
    };
    let (sw, sg) = (sorted(Material::Wax, Material::Oil), sorted(Material::Gallium, Material::Mercury));
    r.check("one group, stirred finer than a cell: molten wax rises out of oil", format!("{:.0} % of the wax on top after 10 s", sw * 100.0), "> 85 %", sw > 0.85);
    r.check("one group, stirred finer than a cell: gallium rises out of mercury", format!("{:.0} % of the gallium on top after 10 s", sg * 100.0), "> 85 %", sg > 0.85);

    // ---- a mercury bead on the floor, zero gravity (as the contact-angle checks of
    // `validate surface`), 1 mm cells: the angle from the cap's area and apex height --------
    let grid = [512, 288];
    let (rad, base) = (30.0f32, [256.0f32, WALL]);
    let zero = |_: [f32; 2]| [0.0f32; 2];
    let zero4 = |_: [f32; 2]| [0.0f32; 4];
    let ps = crate::validate::particles_where(grid, [base[0] - rad, WALL, base[0] + rad, WALL + rad], Material::Mercury, &|x| (x[0] - base[0]).hypot(x[1] - base[1]) < rad, &zero, &zero4);
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| {
        p.cell_size = 0.001;
        p.gravity = 0.0;
    });
    run(&mut w, 2.0);
    r.shot(&w, "mercury_bead");
    let ps = w.sim.download_particles();
    let body: Vec<&Particle> = ps.iter().filter(|p| (p.pos[0] - base[0]).hypot(p.pos[1] - base[1]) < 2.0 * rad).collect();
    let area = body.len() as f64 / PPC as f64;
    let ci = base[0] as usize;
    let apex = (ci - 2..=ci + 2).map(|i| body.iter().filter(|p| p.pos[0] as usize == i).count()).sum::<usize>() as f64 / (5.0 * PPC as f64);
    let cap = |t: f64| (t - t.sin() * t.cos()) / (1.0 - t.cos()).powi(2);
    let (mut lo, mut hi) = (0.05f64, 3.1f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if cap(mid) > area / (apex * apex) { lo = mid } else { hi = mid }
    }
    let theta = (0.5 * (lo + hi)).to_degrees() as f32;
    let set = w.sim.params.mat[Material::Mercury.index()].contact_angle;
    r.check(
        "mercury bead contact angle (zero gravity)",
        format!("{theta:.0} deg (cap {area:.0} cells^2, apex {apex:.1})"),
        &format!("{set} deg; the model's known non-wetting bias is about +12 (accepted {:.0} - {:.0})", set - 5.0, set + 20.0),
        theta > set - 5.0 && theta < set + 20.0,
    );
    let round = crate::validate::ellipticity(&ps, &|p: &Particle| p.is(Material::Mercury));
    println!("  [info] bead ellipticity at zero gravity {round:.3}");

    // ---- liquid resting on a wall must stay in one piece and in place (README, Wetting) ----
    // Both are symmetric about x = 256, so the centre of the body has no reason to move, and
    // nothing should leave it. The contact-angle normal used to be imposed over a band 1.5
    // cells deep, which left the cells along the wall without curvature (the drop's own
    // pressure pushed liquid out along the wall as a sub-cell film), and on the top of thin
    // layers, where its direction along the wall is noise: non-wetting puddles walked
    // sideways and fell apart (surf_curv.comp).
    // A half disc of water at 90 deg, zero gravity, 1 mm cells: already its equilibrium shape.
    {
        let ps = crate::validate::particles_where(grid, [base[0] - rad, WALL, base[0] + rad, WALL + rad], Material::Water, &|x| (x[0] - base[0]).hypot(x[1] - base[1]) < rad, &zero, &zero4);
        let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| {
            p.cell_size = 0.001;
            p.gravity = 0.0;
            p.mat[Material::Water.index()].contact_angle = 90.0;
        });
        run(&mut w, 4.0);
        let (share, cx, _) = puddle_body(&w.sim.download_particles(), grid);
        r.check(
            "sessile drop at its equilibrium (90 deg, zero gravity, 4 s)",
            format!("{:.1} % in one body, centre moved {:+.1} cells", share * 100.0, cx - base[0]),
            "> 99 % (before the fix: 96.3 %), < 2 cells",
            share > 0.99 && (cx - base[0]).abs() < 2.0,
        );
    }
    // Non-wetting puddles under gravity, 0.3 mm cells (capillary length 6 cells for mercury,
    // 9 for water): slabs of their equilibrium height (2 l_c sin(theta / 2)), 100 cells wide.
    // Their square edges round off and the puddles slosh for seconds (mercury is nearly
    // inviscid, and there is no contact-angle hysteresis), so the centre is allowed to wander
    // a little.
    for (m, angle, h) in [(Material::Mercury, None, 12.0f32), (Material::Water, Some(140.0f32), 17.0)] {
        let ps = crate::validate::particles_where(grid, [base[0] - 50.0, WALL, base[0] + 50.0, WALL + h], m, &|_| true, &zero, &zero4);
        let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| {
            p.cell_size = 0.0003;
            if let Some(a) = angle {
                p.mat[m.index()].contact_angle = a;
            }
        });
        let (mut drift, mut share) = (0.0f32, 1.0f64);
        for _ in 0..8 {
            run(&mut w, 0.5);
            let (s, cx, _) = puddle_body(&w.sim.download_particles(), grid);
            drift = drift.max((cx - base[0]).abs());
            share = share.min(s);
        }
        r.shot(&w, &format!("puddle_{}", m.name().to_lowercase()));
        let set = w.sim.params.mat[m.index()].contact_angle;
        r.check(
            &format!("{} puddle at {set:.0} deg under gravity, 4 s", m.name().to_lowercase()),
            format!("at least {:.1} % in one body, centre within {:.1} cells", share * 100.0, drift),
            "> 95 %, < 20 cells (before the fix: mercury 31 %, 239 cells; water 78 %, 35 cells)",
            share > 0.95 && drift < 20.0,
        );
    }

    // ---- ferrofluid drawn to a magnet, zero gravity, 1 mm cells ----------------------------------
    // Its mean acceleration at the start against the force the model predicts there (the
    // same Kelvin-force energy as magnet.comp, with the bulk demagnetisation factor 1/3).
    let grid = [192, 108];
    let magnet = crate::sim::Magnet::new([130.0, 56.0], 6.0, 90.0, 0.4);
    let mut drop = |gpu: &Arc<Gpu>, with: bool| -> (f32, [f32; 2]) {
        let ps: Vec<Particle> = block(grid, [80.0, 40.0, 112.0, 72.0], Material::Ferrofluid).into_iter().filter(|p| (p.pos[0] - 96.0).hypot(p.pos[1] - 56.0) < 14.0).collect();
        let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| {
            p.cell_size = 0.001;
            p.gravity = 0.0;
        });
        if with {
            w.sim.set_magnets(&[magnet]);
        }
        let ps0 = w.sim.download_particles();
        let sp = w.sim.params;
        let ff = sp.mat[Material::Ferrofluid.index()];
        let mu0 = 1.256_637e-6f32;
        let psi = |x: [f32; 2], n: f32| -> f32 {
            let d = [x[0] - magnet.pos[0], x[1] - magnet.pos[1]];
            let r = d[0].hypot(d[1]).max(magnet.radius);
            let e = [d[0] / d[0].hypot(d[1]), d[1] / d[0].hypot(d[1])];
            let md = magnet.dir[0] * e[0] + magnet.dir[1] * e[1];
            let f = magnet.strength * (magnet.radius / r).powi(3) * 0.5;
            let b = [f * (3.0 * md * e[0] - magnet.dir[0]), f * (3.0 * md * e[1] - magnet.dir[1])];
            let h = b[0].hypot(b[1]) / mu0;
            let ce = ff.susceptibility / (1.0 + n * ff.susceptibility);
            let ms = ff.saturation / mu0;
            let xs = ce * h / ms;
            -mu0 * ms * ms / ce * (xs - (1.0 + xs).ln()) / (SimParams::WATER_DENSITY * sp.cell_size * sp.cell_size)
        };
        let rho = ff.density / SimParams::WATER_DENSITY;
        let accel = |n: f32| ps0.iter().map(|p| -(psi([p.pos[0] + 0.5, p.pos[1]], n) - psi([p.pos[0] - 0.5, p.pos[1]], n)) / rho).sum::<f32>() / ps0.len() as f32;
        let predicted = [accel(1.0), accel(0.0)];
        let x0 = ps0.iter().map(|p| p.pos[0]).sum::<f32>() / ps0.len() as f32;
        let t = 0.15;
        run(&mut w, t);
        let ps = w.sim.download_particles();
        let x1 = ps.iter().map(|p| p.pos[0]).sum::<f32>() / ps.len() as f32;
        run(&mut w, 0.5);
        if with {
            r.shot(&w, "ferrofluid_magnet");
        }
        (2.0 * (x1 - x0) / (t * t), predicted)
    };
    let (pulled, predicted) = drop(gpu, true);
    let (still, _) = drop(gpu, false);
    r.check(
        "ferrofluid drop accelerates towards a magnet",
        format!("{pulled:.0} cells/s^2 (no magnet: {still:.2})"),
        &format!("between {:.0} and {:.0} (Kelvin force of the analytic field, demagnetisation N = 1 .. 0), ~0 without", predicted[0], predicted[1]),
        pulled > predicted[0] && pulled < predicted[1] && still.abs() < 0.05 * predicted[0],
    );

    // ---- a dish of ferrofluid with a magnet under it, and one hanging in it, comes to rest -------
    // 1 cm cells, the Magnet tool's default magnet. The liquid heaps up over the first (or
    // climbs round the second) and should then lie still: a force that is the gradient of an
    // energy cannot keep it moving. The rms speed over the 2 s after the first 6 s.
    let grid = [192, 108];
    let mut settle = |gpu: &Arc<Gpu>, at: [f32; 2], degrees: f32| -> f32 {
        let cup = |x: f32| (Shape::Box { c: [x, 20.0], half: [3.0, 18.0], angle: 0.0 }, None);
        let ps = block(grid, [40.0, WALL, 152.0, WALL + 12.0], Material::Ferrofluid);
        let mut w = build(gpu, grid, apply, vec![cup(36.0), cup(156.0)], ps, &|p: &mut SimParams| p.cell_size = 0.01);
        w.add_magnet(crate::sim::Magnet::new(at, 6.0, degrees, crate::sim::MAGNET_STRENGTH_DEFAULT));
        run(&mut w, 6.0);
        let mut ms = 0.0f64;
        for _ in 0..8 {
            run(&mut w, 0.25);
            let ps = w.sim.download_particles();
            let live: Vec<&Particle> = ps.iter().filter(|p| p.kind & 255 != 0).collect();
            ms += live.iter().map(|p| (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]) as f64).sum::<f64>() / live.len().max(1) as f64 / 8.0;
        }
        if degrees == 0.0 {
            r.shot(&w, "ferrofluid_dish_magnet");
        }
        ms.sqrt() as f32
    };
    let under = settle(gpu, [96.0, WALL + 3.0], 0.0);
    let inside = settle(gpu, [96.0, WALL + 22.0], 180.0);
    r.check(
        "ferrofluid dish over a magnet comes to rest, 1 cm cells",
        format!("rms speed {under:.2} cells/s 6 - 8 s after the magnet was placed"),
        "< 8 (before the 2026-11-15 fixes: 25; after: 3.6, a slow surface current over the heap)",
        under < 8.0,
    );
    r.check(
        "ferrofluid round a magnet held in it settles, 1 cm cells",
        format!("rms speed {inside:.2} cells/s 6 - 8 s after the magnet was placed"),
        "< 22 (before the 2026-11-15 fixes: 34; after: 15, at the neck where it meets the dish)",
        inside < 22.0,
    );

    // ---- gallium: the oxide skin holds a stretched shape; without it tension pulls it round -----
    let grid = [192, 108];
    let mut aspect = |gpu: &Arc<Gpu>, skin: bool| -> f32 {
        // A 50 x 12 cell stadium (round ends of radius 6), zero gravity, 1 mm cells.
        let ps: Vec<Particle> = block(grid, [65.0, 48.0, 127.0, 60.0], Material::Gallium)
            .into_iter()
            .filter(|p| {
                let x = p.pos[0].clamp(71.0, 121.0);
                (p.pos[0] - x).hypot(p.pos[1] - 54.0) < 6.0
            })
            .collect();
        let mut w = build(gpu, grid, apply, Vec::new(), ps, &|p: &mut SimParams| {
            p.cell_size = 0.001;
            p.gravity = 0.0;
            if !skin {
                p.mat[Material::Gallium.index()].skin = 0.0;
            }
        });
        run(&mut w, 0.1);
        run(&mut w, 0.1);
        run(&mut w, 0.3);
        if skin {
            r.shot(&w, "gallium_skin");
        }
        let ps = w.sim.download_particles();
        let xs: Vec<f32> = ps.iter().map(|p| p.pos[0]).collect();
        let ys: Vec<f32> = ps.iter().map(|p| p.pos[1]).collect();
        (percentile(xs.clone(), 0.995) - percentile(xs, 0.005)) / (percentile(ys.clone(), 0.995) - percentile(ys, 0.005))
    };
    let with = aspect(gpu, true);
    let without = aspect(gpu, false);
    r.check("gallium: the oxide skin holds a stretched blob", format!("aspect ratio after 0.5 s: {with:.2} with skin, {without:.2} without (start 5.0)"), "with > 4, without < 3", with > 4.0 && without < 3.0);
}

/// Mixing conserves every component: ink, mud, blood, milk and honey stirred into water.
pub fn mixing(gpu: &Arc<Gpu>, apply: &dyn Fn(&mut SimParams), r: &mut Report) {
    println!("\nMixing (ink, mud, blood, milk, honey and brine stirred into water, 10 s)");
    let grid = [256, 144];
    let mut ps = block(grid, [0.0, 0.0, 256.0, 60.0], Material::Water);
    let blobs = [(Material::Ink, 30.0), (Material::Mud, 70.0), (Material::Blood, 110.0), (Material::Milk, 150.0), (Material::Honey, 190.0), (Material::Brine, 230.0)];
    for (m, x) in blobs {
        ps.extend(block(grid, [x - 10.0, 62.0, x + 10.0, 82.0], m));
    }
    let mut w = build(gpu, grid, apply, Vec::new(), ps, &|_| {});
    let volumes = |ps: &[Particle]| -> Vec<f64> {
        let mut v = vec![0.0f64; blobs.len() + 1];
        for p in ps {
            let m = p.material().unwrap();
            match blobs.iter().position(|b| b.0 == m) {
                Some(k) => {
                    v[k] += p.s as f64;
                    v[blobs.len()] += 1.0 - p.s as f64;
                }
                None => v[blobs.len()] += 1.0,
            }
        }
        v
    };
    let ps0 = w.sim.download_particles();
    let v0 = volumes(&ps0);
    for k in 0..120 {
        let x = 20.0 + 216.0 * (k as f32 / 119.0);
        w.sim.advance_blocking(1.0 / 60.0, ToolState::grab([x, 40.0], [200.0, 0.0], 20.0));
    }
    run(&mut w, 8.0);
    r.shot(&w, "mixing");
    let ps = w.sim.download_particles();
    let v1 = volumes(&ps);
    let mut worst = 0.0f64;
    let mut line = String::new();
    for (k, (a, b)) in v0.iter().zip(&v1).enumerate() {
        let d = (b - a) / a.max(1.0);
        worst = worst.max(d.abs());
        let name = if k < blobs.len() { blobs[k].0.name() } else { "water" };
        line += &format!("{name}: {:+.4} %  ", d * 100.0);
    }
    println!("  volume change per component: {line}");
    r.check("particles conserved", format!("{} -> {}", ps0.len(), ps.len()), "equal", ps0.len() == ps.len());
    r.check("every component's volume conserved", format!("worst {:+.4} %", worst * 100.0), "|drift| < 0.1 %", worst < 1e-3);
    let blended = ps.iter().filter(|p| p.s > 0.02 && p.s < 0.98 && p.material().is_some_and(|m| m != Material::Mud)).count();
    r.check("the dissolved ones blended", format!("{blended} particles at intermediate concentration"), "many", blended > 1000);
}

/// The largest 8-connected group of cells holding at least half the particles a full cell
/// holds: its share of all particles, its mean x (cells) and the rms speed of its particles
/// (cells / s).
fn puddle_body(ps: &[Particle], grid: [u32; 2]) -> (f64, f32, f64) {
    let (gw, gh) = (grid[0] as usize, grid[1] as usize);
    let at = |p: &Particle| (p.pos[1] as usize).min(gh - 1) * gw + (p.pos[0] as usize).min(gw - 1);
    let mut count = vec![0u32; gw * gh];
    for p in ps {
        count[at(p)] += 1;
    }
    let mut label = vec![0u32; gw * gh];
    let (mut best, mut best_n, mut next) = (0u32, 0u32, 1u32);
    for start in 0..gw * gh {
        if label[start] != 0 || count[start] < PPC / 2 {
            continue;
        }
        let (mut stack, mut n) = (vec![start], 0u32);
        label[start] = next;
        while let Some(i) = stack.pop() {
            n += count[i];
            let (x, y) = ((i % gw) as i64, (i / gw) as i64);
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let (u, v) = (x + dx, y + dy);
                if u < 0 || v < 0 || u >= gw as i64 || v >= gh as i64 {
                    continue;
                }
                let j = v as usize * gw + u as usize;
                if label[j] == 0 && count[j] >= PPC / 2 {
                    label[j] = next;
                    stack.push(j);
                }
            }
        }
        if n > best_n {
            (best, best_n) = (next, n);
        }
        next += 1;
    }
    let body: Vec<&Particle> = ps.iter().filter(|p| label[at(p)] == best).collect();
    let n = body.len().max(1) as f64;
    let cx = body.iter().map(|p| p.pos[0] as f64).sum::<f64>() / n;
    let v2 = body.iter().map(|p| (p.vel[0] as f64).powi(2) + (p.vel[1] as f64).powi(2)).sum::<f64>() / n;
    (body.len() as f64 / ps.len().max(1) as f64, cx as f32, v2.sqrt())
}
