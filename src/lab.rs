//! `sprout lab`: a scripted lava-and-water scene for chasing heat and stability problems
//! (developer aid, not in `--help`). Lava is dropped into a pool of water and left to settle,
//! then the Cool brush is held over it; every sample prints temperatures, phase shares,
//! particle speeds, crowding and solver effort, and pictures are saved along the way.

use std::{collections::HashMap, path::Path, sync::Arc};

use crate::{
    gpu::Gpu,
    headless::Offscreen,
    maps::fill_rect,
    render::View,
    scene::Scene,
    sim::{Material, Particle, SimParams, ToolState, heat_content},
    world::World,
};

struct Stats {
    mean_t: f32,
    max_t: f32,
    frozen: f32,
    rms: f32,
    p95: f32,
    n: usize,
}

fn stats(ps: &[&Particle], sp: &SimParams) -> Stats {
    if ps.is_empty() {
        return Stats { mean_t: 0.0, max_t: 0.0, frozen: 0.0, rms: 0.0, p95: 0.0, n: 0 };
    }
    let mut sum_t = 0.0f64;
    let mut max_t = f32::MIN;
    let mut frozen = 0usize;
    let mut v2 = 0.0f64;
    let mut speeds = Vec::with_capacity(ps.len());
    for p in ps {
        let (t, f) = sp.temperature(p);
        sum_t += t as f64;
        max_t = max_t.max(t);
        if f > 0.5 {
            frozen += 1;
        }
        let s = p.vel[0].hypot(p.vel[1]);
        v2 += (s * s) as f64;
        speeds.push(s);
    }
    speeds.sort_by(f32::total_cmp);
    Stats {
        mean_t: (sum_t / ps.len() as f64) as f32,
        max_t,
        frozen: frozen as f32 / ps.len() as f32,
        rms: (v2 / ps.len() as f64).sqrt() as f32,
        p95: speeds[(speeds.len() * 95 / 100).min(speeds.len() - 1)],
        n: ps.len(),
    }
}

/// Velocity noise: rms deviation of each particle's velocity from the mean of its 3 x 3-cell
/// block, cells/s, for the particles `pick` selects. Smooth flow (sloshing, a current) has
/// almost none; particles jittering against each other have a lot.
fn noise(ps: &[Particle], pick: &dyn Fn(&Particle) -> bool) -> f32 {
    let mut blocks: HashMap<(i32, i32), ([f64; 2], f64)> = HashMap::new();
    let key = |p: &Particle| ((p.pos[0] / 3.0) as i32, (p.pos[1] / 3.0) as i32);
    for p in ps {
        let e = blocks.entry(key(p)).or_insert(([0.0; 2], 0.0));
        e.0[0] += p.vel[0] as f64;
        e.0[1] += p.vel[1] as f64;
        e.1 += 1.0;
    }
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for p in ps.iter().filter(|p| pick(p)) {
        let b = blocks[&key(p)];
        let (dx, dy) = (p.vel[0] as f64 - b.0[0] / b.1, p.vel[1] as f64 - b.0[1] / b.1);
        s += dx * dx + dy * dy;
        n += 1.0;
    }
    if n > 0.0 { (s / n).sqrt() as f32 } else { 0.0 }
}

/// Frozen particles of material `m` whose 3 x 3-cell block is mostly not frozen: specks.
fn stray_frozen(ps: &[Particle], sp: &SimParams, m: Material) -> usize {
    let mut blocks: HashMap<(i32, i32), (u32, u32)> = HashMap::new();
    let key = |p: &Particle| ((p.pos[0] / 3.0) as i32, (p.pos[1] / 3.0) as i32);
    let fr: Vec<bool> = ps.iter().map(|p| sp.temperature(p).1 > 0.5).collect();
    for (p, f) in ps.iter().zip(&fr).filter(|(p, _)| p.material() == Some(m)) {
        let e = blocks.entry(key(p)).or_insert((0, 0));
        e.0 += *f as u32;
        e.1 += 1;
    }
    ps.iter().zip(&fr).filter(|(p, f)| **f && p.material() == Some(m) && { let b = blocks[&key(p)]; b.0 * 2 < b.1 }).count()
}

fn crowded(ps: &[Particle], grid: [u32; 2]) -> u32 {
    let mut n = vec![0u32; (grid[0] * grid[1]) as usize];
    for p in ps {
        let (x, y) = (p.pos[0] as u32, p.pos[1] as u32);
        if x < grid[0] && y < grid[1] {
            n[(y * grid[0] + x) as usize] += 1;
        }
    }
    n.into_iter().max().unwrap_or(0)
}

/// Spread of temperature among particles sharing a cell: the mean over cells holding both
/// lava and water of (hottest - coldest), °C.
fn cell_spread(ps: &[Particle], sp: &SimParams) -> f32 {
    let mut cells: HashMap<(i32, i32), (f32, f32, bool, bool)> = HashMap::new();
    for p in ps {
        let k = (p.pos[0] as i32, p.pos[1] as i32);
        let t = sp.temperature(p).0;
        let lava = p.material() == Some(Material::Lava);
        let e = cells.entry(k).or_insert((f32::MAX, f32::MIN, false, false));
        e.0 = e.0.min(t);
        e.1 = e.1.max(t);
        if lava { e.2 = true } else { e.3 = true }
    }
    let mixed: Vec<f32> = cells.values().filter(|c| c.2 && c.3).map(|c| c.1 - c.0).collect();
    if mixed.is_empty() { 0.0 } else { mixed.iter().sum::<f32>() / mixed.len() as f32 }
}

pub fn run(gpu: Arc<Gpu>, apply: &dyn Fn(&mut SimParams), grid: [u32; 2], settle: f32, cool: f32, radius: f32, out: Option<&str>) {
    let scene = Scene::new(grid);
    let (gw, gh) = (grid[0] as f32, grid[1] as f32);
    let mut ps = Vec::new();
    fill_rect(&scene, [0.0, 0.0, gw, 0.30 * gh], Material::Water, &mut ps);
    let mut lava = Vec::new();
    fill_rect(&scene, [0.30 * gw, 0.45 * gh, 0.70 * gw, 0.62 * gh], Material::Lava, &mut lava);
    if std::env::var("LAB_NOLAVA").is_ok() {
        for p in &mut lava {
            *p = Particle::new(Material::Water, p.pos, p.vel);
        }
    }
    if std::env::var("LAB_STEFAN").is_ok() {
        // Closed box at 1 mm: ice below, warm water above, no exchange with anything.
        let grid = [48u32, 160];
        let scene = Scene::new(grid);
        let mut ps = Vec::new();
        fill_rect(&scene, [0.0, 0.0, 48.0, 60.0], Material::Water, &mut ps);
        let mut ps: Vec<Particle> = ps.into_iter().map(|p| p.at_temp(-0.5)).collect();
        let mut warm = Vec::new();
        fill_rect(&scene, [0.0, 60.0, 48.0, 120.0], Material::Water, &mut warm);
        ps.extend(warm.into_iter().map(|p| p.at_temp(40.0)));
        let mut w = World::custom(gpu.clone(), scene, ps);
        apply(&mut w.sim.params);
        w.sim.params.cell_size = 0.001;
        w.sim.params.heat_rate = 100.0;
        w.sim.params.air_cooling = 0.0;
        w.sim.params.mat[Material::Water.index()].emissivity = 0.0;
        w.sim.params.mat[Material::Water.index()].density_solid = w.sim.params.mat[Material::Water.index()].density;
        w.sim.step_blocking(0, ToolState::default());
        let e0 = heat_content(&w.sim.download_particles(), &w.sim.params);
        for k in 0..10 {
            w.sim.advance_blocking(0.5, ToolState::default());
            let ps = w.sim.download_particles();
            let e = heat_content(&ps, &w.sim.params);
            let frozen = ps.iter().filter(|q| w.sim.params.temperature(q).1 > 0.5).count();
            println!("t {:4.1}  content change {:+.4e} ({:+.4} %)  net in {:+.3e}  frozen {frozen}", 0.5 * (k + 1) as f32, e - e0, 100.0 * (e - e0) / e0, w.sim.download_ledger().net_in());
        }
        return;
    }
    if let Ok(t) = std::env::var("LAB_WATER_T") {
        // A pool of water at T (deg C) left in the air: how it cools (README, Heat,
        // "Evaporation"). Prints the mean temperature and the heat lost to the air.
        let t: f32 = t.parse().unwrap_or(80.0);
        let ps: Vec<Particle> = ps.iter().map(|p| p.at_temp(t)).collect();
        let mut w = World::with_capacity(gpu.clone(), scene, ps, 400_000);
        apply(&mut w.sim.params);
        w.sim.enable_heat();
        w.sim.step_blocking(0, ToolState::default());
        for k in 0..=(settle / 2.0) as usize {
            let ps = w.sim.download_particles();
            let water: Vec<&Particle> = ps.iter().filter(|p| p.material() == Some(Material::Water)).collect();
            let st = stats(&water, &w.sim.params);
            let l = w.sim.download_ledger();
            println!("t {:5.1} s  water mean {:6.2} C  max {:6.2} C  to air {:.3e} J/m", 2.0 * k as f32, st.mean_t, st.max_t, l.to_air);
            w.sim.advance_blocking(2.0, ToolState::default());
        }
        return;
    }
    if std::env::var("LAB_BUBBLES").is_ok() {
        // Bubbles through cool water: a pool at 20 deg C (or LAB_BUBBLES_T) boiled at its
        // floor by the Heat brush for 0.5 s; then, frame by frame for 1.25 s, the highest
        // steam bubble above the brush and how many there are.
        let t0: f32 = std::env::var("LAB_BUBBLES_T").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0);
        let ps: Vec<Particle> = ps.iter().map(|p| p.at_temp(t0)).collect();
        let mut w = World::with_capacity(gpu.clone(), scene, ps, 400_000);
        apply(&mut w.sim.params);
        w.sim.enable_heat();
        w.sim.step_blocking(0, ToolState::default());
        let at = [0.5 * gw, crate::scene::WALL + 3.0];
        let (mut top, mut made) = (0.0f32, 0usize);
        for f in 0..105 {
            let tool = if f < 30 { ToolState::heat(at, 8.0, false) } else { ToolState::default() };
            w.sim.advance_blocking(1.0 / 60.0, tool);
            let steam: Vec<f32> = w.sim.download_particles().iter().filter(|p| p.is(Material::Steam)).map(|p| p.pos[1] - at[1]).collect();
            made = made.max(steam.len());
            top = steam.iter().copied().fold(top, f32::max);
            if f % 15 == 14 {
                println!("t {:4.2} s  steam {:5}  highest so far {:5.1} cells above the brush", (f + 1) as f32 / 60.0, steam.len(), top);
            }
        }
        println!("most steam at once {made}, highest bubble {top:.1} cells");
        return;
    }
    if let Ok(m) = std::env::var("LAB_DROP") {
        drop_test(gpu, apply, grid, &m);
        return;
    }
    let pour = std::env::var("LAB_POUR").is_ok();
    if !pour {
        ps.extend(lava);
    }
    let map = std::env::var("LAB_MAP").ok().and_then(|v| v.parse::<usize>().ok());
    let mut w = match map {
        Some(m) => World::new(gpu.clone(), grid, m),
        None => World::with_capacity(gpu.clone(), scene, ps, 400_000),
    };
    apply(&mut w.sim.params);
    w.sim.step_blocking(0, ToolState::default());
    if map.is_some() {
        // As in the reported scene: lava, then water, then lava again, poured from a
        // sweeping brush into the map (12 s), then everything left to run.
        let sweep = |f: usize, y: f32| {
            let ph = (f as f32 / 60.0) * 0.35;
            [gw * (0.5 + 0.38 * (ph * std::f32::consts::TAU).sin()), y * gh]
        };
        let rounds = [(Material::Lava, 300usize), (Material::Water, 180), (Material::Lava, 240)];
        let mut f0 = 0;
        for (m, frames) in rounds {
            for f in f0..f0 + frames {
                let at = sweep(f, 0.8);
                w.emit(crate::world::Source::Pour, m, at, 12.0, 9000.0, 1.0 / 60.0);
                w.sim.advance_blocking(1.0 / 60.0, crate::world::Source::Pour.nozzle(at, 12.0, 9000.0));
            }
            w.end_stroke();
            f0 += frames;
        }
    }
    if pour {
        // Lava poured from a sweeping jet into the pool, as a user would (4 s).
        let at = |f: usize| {
            let ph = (f as f32 / 60.0) * 0.8;
            [gw * (0.5 + 0.3 * (ph * std::f32::consts::TAU).sin()), 0.6 * gh]
        };
        for f in 0..240 {
            w.emit(crate::world::Source::Pour, Material::Lava, at(f), 8.0, 6000.0, 1.0 / 60.0);
            w.sim.advance_blocking(1.0 / 60.0, crate::world::Source::Pour.nozzle(at(f), 8.0, 6000.0));
        }
        w.end_stroke();
    }
    // Let the last emitted particles enter the tank before the heat budget starts.
    for _ in 0..30 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
    }
    let mut off = out.map(|_| Offscreen::new(gpu.clone(), [1280, 720]));
    let e0 = heat_content(&w.sim.download_particles(), &w.sim.params);
    let n0 = w.sim.download_ledger().net_in();
    println!(
        "{:>6} {:>6} {:>4} {:>4} | {:>7} {:>7} {:>5} {:>6} {:>6} | {:>6} {:>6} {:>5} {:>6} {:>6} | {:>7} {:>5} {:>6} {:>6} {:>6} {:>5} {:>6}",
        "t", "dt ms", "p it", "v it", "lava T", "max", "frz", "rms", "p95", "wat T", "max", "frz", "rms", "p95", "spread", "crowd", "boiled", "noiseL", "noiseW", "iceW", "crustL"
    );
    let release: f32 = std::env::var("LAB_RELEASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let total = settle + cool + release;
    let dt_sample = 0.5f32;
    let mut t = 0.0f32;
    let mut shot = 0;
    let mut centre = [0.5 * gw, 0.15 * gh];
    while t <= total + 1e-3 {
        let sp = w.sim.params;
        let ps = w.sim.download_particles();
        let lava: Vec<&Particle> = ps.iter().filter(|p| p.material() == Some(Material::Lava)).collect();
        let water: Vec<&Particle> = ps.iter().filter(|p| p.material() == Some(Material::Water)).collect();
        if !lava.is_empty() && t <= settle {
            let n = lava.len() as f32;
            centre = [lava.iter().map(|p| p.pos[0]).sum::<f32>() / n, lava.iter().map(|p| p.pos[1]).sum::<f32>() / n];
        }
        let (a, b) = (stats(&lava, &sp), stats(&water, &sp));
        let l = w.sim.download_ledger();
        if std::env::var("LAB_BUDGET").is_ok() {
            let e = heat_content(&ps, &sp);
            println!("    budget: unaccounted {:+.3e} J/m (brush out {:.3e}, air out {:.3e}, boiled {:.3e})", (e - e0) - (l.net_in() - n0), l.cooled, l.to_air, l.boiled);
        }
        let st = w.sim.solver;
        println!(
            "{:6.2} {:6.3} {:4} {:4} | {:7.1} {:7.1} {:5.2} {:6.2} {:6.2} | {:6.1} {:6.1} {:5.2} {:6.2} {:6.2} | {:7.1} {:5} {:6} {:6.2} {:6.2} {:5} {:6}",
            w.sim.sim_time(), st.dt * 1e3, st.pressure_iters_max, st.viscosity_iters_max,
            a.mean_t, a.max_t, a.frozen, a.rms, a.p95, b.mean_t, b.max_t, b.frozen, b.rms, b.p95,
            cell_spread(&ps, &sp), crowded(&ps, grid), l.boiled_count.iter().sum::<u32>(),
            noise(&ps, &|p| p.material() == Some(Material::Lava)), noise(&ps, &|p| p.material() == Some(Material::Water)),
            stray_frozen(&ps, &sp, Material::Water), stray_frozen(&ps, &sp, Material::Lava)
        );
        let _ = (a.n, b.n);
        w.sim.solver.reset_extremes();
        if let (Some(dir), Some(off)) = (out, off.as_mut())
            && (shot as f32 * 2.0 <= t + 1e-3)
        {
            std::fs::create_dir_all(dir).ok();
            off.save_png(&w.sim, View::fit(grid, off.size), &Path::new(dir).join(format!("lab_{:03}.png", (t * 10.0).round() as u32)));
            let z = View { center: centre, zoom: off.size[0] as f32 / 160.0 };
            off.save_png(&w.sim, z, &Path::new(dir).join(format!("zoom_{:03}.png", (t * 10.0).round() as u32)));
            shot += 1;
        }
        if let (Some(dir), Some(off)) = (out, off.as_mut())
            && ((t - settle).abs() < 1e-3 || (t - total).abs() < 1e-3)
        {
            // Six consecutive frames, for the flicker count (tools/flicker.py).
            let tool = if t >= settle { ToolState::heat(centre, radius, true) } else { ToolState::default() };
            let z = View { center: centre, zoom: off.size[0] as f32 / 200.0 };
            for k in 0..6 {
                off.save_png(&w.sim, z, &Path::new(dir).join(format!("motion_{:03}_{k}.png", (t * 10.0).round() as u32)));
                w.sim.advance_blocking(1.0 / 60.0, tool);
            }
        }
        let tool = if t >= settle && t < settle + cool - 1e-3 { ToolState::heat(centre, radius, true) } else { ToolState::default() };
        let frames = (dt_sample * 60.0).round() as usize;
        for _ in 0..frames {
            w.sim.advance_blocking(1.0 / 60.0, tool);
        }
        t += dt_sample;
    }
    let ps = w.sim.download_particles();
    let l = w.sim.download_ledger();
    let e1 = heat_content(&ps, &w.sim.params);
    println!(
        "heat (J/m): change {:+.4e}, net in {:+.4e} (air in {:.3e} out {:.3e}, brush out {:.3e}, boiled {:.3e}); unaccounted {:+.2e} ({:+.3} % of content)",
        e1 - e0, l.net_in() - n0, l.from_air, l.to_air, l.cooled, l.boiled, (e1 - e0) - (l.net_in() - n0), 100.0 * ((e1 - e0) - (l.net_in() - n0)) / e0
    );
    let st = w.sim.solver;
    println!("substeps {}  unconverged pressure {} viscosity {}", st.substeps, st.pressure_unconverged, st.viscosity_unconverged);
}

/// `LAB_DROP=<material> sprout lab`: a pool of water left to settle (6 s), then a small
/// stream of the material poured onto one side (1 s), then 12 s of watching. Prints, per
/// 0.5 s window, the rms speed and velocity noise of the water far from the poured
/// material (more than 24 cells from any of it): a calm pool stays near zero, a pool that
/// shakes as a whole does not.
fn drop_test(gpu: Arc<Gpu>, apply: &dyn Fn(&mut SimParams), grid: [u32; 2], m: &str) {
    let scene = Scene::new(grid);
    let (gw, gh) = (grid[0] as f32, grid[1] as f32);
    let mut ps = Vec::new();
    fill_rect(&scene, [0.0, 0.0, gw, 0.35 * gh], Material::Water, &mut ps);
    let mut w = World::with_capacity(gpu, scene, ps, 400_000);
    apply(&mut w.sim.params);
    w.sim.step_blocking(0, ToolState::default());
    for _ in 0..360 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
    }
    let mat = (0..19u8).map(Material).find(|q| q.name().eq_ignore_ascii_case(m)).unwrap_or(Material::Lava);
    let at = [0.25 * gw, 0.6 * gh];
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    for _ in 0..(env("LAB_POUR_S", 1.0) * 60.0) as usize {
        w.emit(crate::world::Source::Pour, mat, at, 6.0, 3000.0, 1.0 / 60.0);
        w.sim.advance_blocking(1.0 / 60.0, crate::world::Source::Pour.nozzle(at, 6.0, 3000.0));
    }
    w.end_stroke();
    for _ in 0..(env("LAB_WAIT", 0.0) * 60.0) as usize {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
    }
    if std::env::var("LAB_BODIES").is_ok() {
        // One substep at a time: the bodies, largest first (mass in cells x density / 1000).
        for k in 0..24 {
            let dt = w.sim.dt_estimate();
            w.sim.frame_blocking(1, dt, ToolState::default());
            let mut b = w.sim.download_bodies();
            b.sort_by(|a, c| c[0].total_cmp(&a[0]));
            let big: Vec<String> = b.iter().take(4).map(|q| {
                let m = q[0].max(1e-9);
                let (cx, cy) = (q[1] / m, q[2] / m);
                let (ux, uy) = (q[3] / m, q[4] / m);
                let l = q[5] - m * (cx * uy - cy * ux);
                let i = q[6] - m * (cx * cx + cy * cy);
                format!("M {:7.0} U ({:+6.2},{:+6.2}) w {:+.4}", m / 1000.0, ux, uy, if i > 1e-3 { l / i } else { 0.0 })
            }).collect();
            println!("sub {k:2} dt {:.2} ms bodies {:4}: {}", dt * 1e3, b.len(), big.join(" | "));
            // The biggest body's cells: mean face velocity (after the rigid pass) by row
            // band, and, with the rigid pass off for one substep, the projected field.
            let crate::sim::GridDump { vel, labels: rl, cell } = w.sim.download_grid();
            let gw = grid[0] as usize;
            let mut counts: HashMap<i32, usize> = HashMap::new();
            for l in &rl { if l[0] >= 0 { *counts.entry(l[0]).or_default() += 1; } }
            let Some((&root, &nc)) = counts.iter().max_by_key(|e| *e.1) else { continue };
            let (mut sv, mut n, mut ymin, mut ymax) = (0.0f64, 0usize, usize::MAX, 0usize);
            let mut low = 0.0f64;
            let mut nlow = 0;
            for (i, l) in rl.iter().enumerate() {
                if l[0] != root { continue; }
                let (x, y) = (i % gw, i / gw);
                let _ = x;
                ymin = ymin.min(y);
                ymax = ymax.max(y);
                let vc = 0.5 * (vel[i][1] + vel[i + gw][1]);
                sv += vc as f64;
                n += 1;
                if cell[i][0] < 0.8 { low += 1.0; nlow += 1; }
            }
            let _ = low;
            println!("        biggest body: {nc} cells rows {ymin}..{ymax}, mean cell vy {:+.2}, under-full cells {nlow}", sv / n.max(1) as f64);
        }
        return;
    }
    if std::env::var("LAB_FROZEN").is_ok() {
        // Per frame after `LAB_WAIT`: mean and rms velocity of the frozen poured particles,
        // and the rms about the mean (rotation or deformation), cells/s.
        let sp = w.sim.params;
        for f in 0..40 {
            w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
            let ps = w.sim.download_particles();
            let fr: Vec<&Particle> = ps.iter().filter(|p| p.material() == Some(mat) && sp.temperature(p).1 > 0.5).collect();
            let n = fr.len().max(1) as f32;
            let m = [fr.iter().map(|p| p.vel[0]).sum::<f32>() / n, fr.iter().map(|p| p.vel[1]).sum::<f32>() / n];
            let rms = (fr.iter().map(|p| p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sum::<f32>() / n).sqrt();
            let dev = (fr.iter().map(|p| (p.vel[0] - m[0]).powi(2) + (p.vel[1] - m[1]).powi(2)).sum::<f32>() / n).sqrt();
            let mx = fr.iter().map(|p| p.vel[0].hypot(p.vel[1])).fold(0.0f32, f32::max);
            println!("frame {f:3}: frozen {:5}  mean ({:+.3}, {:+.3})  rms {:.3}  about mean {:.3}  max {:.2}", fr.len(), m[0], m[1], rms, dev, mx);
        }
        return;
    }
    if std::env::var("LAB_SHAKE").is_ok() {
        // Shake: rms second difference of particle positions over consecutive frames
        // (x(t+1) - 2 x(t) + x(t-1), cells), in groups. Smooth motion of any speed has
        // almost none; a particle jolted back and forth has a lot.
        let sp = w.sim.params;
        println!("{:>6} {:>9} {:>9} {:>9} {:>9} {:>9}  (rms 2nd difference, 1e-3 cells; counts in last col)", "t", "far wat", "near wat", "liq pour", "frz pour", "all wat");
        let mut t = 0.0f32;
        for at in [0.5f32, 2.0, 5.0, 10.0, 20.0, 30.0] {
            while t < at {
                w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
                t += 1.0 / 60.0;
            }
            let mut acc = [(0.0f64, 0usize); 5];
            let mut hist: Vec<HashMap<u32, [f32; 2]>> = Vec::new();
            let mut last = Vec::new();
            for _ in 0..12 {
                let ps = w.sim.download_particles();
                hist.push(ps.iter().map(|p| (p.id, p.pos)).collect());
                last = ps;
                w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
                t += 1.0 / 60.0;
            }
            let (cw, ch) = (grid[0] as usize, grid[1] as usize);
            let mut near = vec![false; cw * ch];
            for p in last.iter().filter(|p| p.material() == Some(mat)) {
                let (x, y) = (p.pos[0] as i32, p.pos[1] as i32);
                for yy in (y - 24).max(0)..(y + 25).min(ch as i32) {
                    for xx in (x - 24).max(0)..(x + 25).min(cw as i32) {
                        near[yy as usize * cw + xx as usize] = true;
                    }
                }
            }
            for p in &last {
                let (x, y) = ((p.pos[0] as usize).min(cw - 1), (p.pos[1] as usize).min(ch - 1));
                let poured = p.material() == Some(mat);
                let g = if poured { if sp.temperature(p).1 > 0.5 { 3 } else { 2 } } else if near[y * cw + x] { 1 } else { 0 };
                for k in 1..hist.len() - 1 {
                    let (Some(a), Some(b), Some(c)) = (hist[k - 1].get(&p.id), hist[k].get(&p.id), hist[k + 1].get(&p.id)) else { continue };
                    let d2 = ((c[0] - 2.0 * b[0] + a[0]) as f64).powi(2) + ((c[1] - 2.0 * b[1] + a[1]) as f64).powi(2);
                    acc[g].0 += d2;
                    acc[g].1 += 1;
                    if !poured {
                        acc[4].0 += d2;
                        acc[4].1 += 1;
                    }
                }
            }
            let f = |a: (f64, usize)| if a.1 > 0 { 1e3 * (a.0 / a.1 as f64).sqrt() } else { 0.0 };
            println!("{:6.1} {:9.2} {:9.2} {:9.2} {:9.2} {:9.2}  n {:?}", at, f(acc[0]), f(acc[1]), f(acc[2]), f(acc[3]), f(acc[4]), acc.iter().map(|a| a.1 / 10).collect::<Vec<_>>());
        }
        return;
    }
    if std::env::var("LAB_PACE").is_ok() {
        // The app's frame pacing at 60 Hz for 20 s: simulated time each frame covered.
        let mut pacer = crate::pacing::Pacer::new();
        let cost = env("LAB_COST", 0.0015);
        let frame = 1.0 / 60.0;
        let mut per = Vec::new();
        let mut short = 0;
        let mut subs = Vec::new();
        let mut win = Vec::new();
        for f in 0..1200 {
            let plan = pacer.plan(frame, w.sim.dt_estimate(), false, 1.0, cost, 0.8 / 60.0);
            let got = w.sim.frame_blocking(plan.substeps, plan.target, ToolState::default()) as f32;
            per.push(got);
            subs.push(plan.substeps);
            win.push(got);
            if got < 0.5 * frame {
                short += 1;
            }
            if f % 120 == 119 {
                let m = win.iter().sum::<f32>() / win.len() as f32;
                let sd = (win.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / win.len() as f32).sqrt();
                let lo = win.iter().cloned().fold(f32::MAX, f32::min);
                println!("frames {:4}-{:4}: sim per frame mean {:.2} ms sd {:.2} min {:.2}  dt now {:.2} ms", f - 119, f, m * 1e3, sd * 1e3, lo * 1e3, w.sim.dt_estimate() * 1e3);
                win.clear();
            }
        }
        let m = per.iter().sum::<f32>() / per.len() as f32;
        let sd = (per.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / per.len() as f32).sqrt();
        // Judder: rms change of the simulated time from one frame to the next, relative.
        let jd = (per.windows(2).map(|p| ((p[1] - p[0]) / frame).powi(2)).sum::<f32>() / (per.len() - 1) as f32).sqrt();
        println!("pace: mean {:.2} ms/frame (sd {:.2}), frames under half a frame {short}/{}, judder {:.3}, substeps {:?}", m * 1e3, sd * 1e3, per.len(), jd, &subs[..24]);
        return;
    }
    if let Ok(dir) = std::env::var("LAB_DIR") {
        // Bursts of 8 consecutive frames at 1, 5, 15 and 30 s after the pour.
        let mut off = Offscreen::new(w.sim.gpu(), [1280, 720]);
        let mut t = 0.0;
        for (b, at) in [1.0f32, 5.0, 15.0, 30.0].into_iter().enumerate() {
            while t < at {
                w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
                t += 1.0 / 60.0;
            }
            let d = Path::new(&dir).join(format!("b{b}"));
            std::fs::create_dir_all(&d).ok();
            for k in 0..8 {
                off.save_png(&w.sim, View::fit(grid, off.size), &d.join(format!("f_{k}.png")));
                w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
                t += 1.0 / 60.0;
            }
        }
        return;
    }
    println!("{:>6} {:>8} {:>8} {:>8} {:>7} {:>6} {:>5} {:>6} {:>7}", "t", "far rms", "far max", "far nse", "frozen", "boiled", "p it", "v it", "dt ms");
    let (cw, ch) = (grid[0] as usize, grid[1] as usize);
    let (mut sum_rms, mut nwin) = (0.0f64, 0);
    for win in 0..24 {
        let (mut s2, mut n, mut mx, mut nse) = (0.0f64, 0usize, 0.0f32, 0.0f32);
        let mut frozen = 0.0;
        for _k in 0..5 {
            for _ in 0..6 {
                w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
            }
            let ps = w.sim.download_particles();
            let sp = w.sim.params;
            // Cells within 24 of any particle of the poured material.
            let mut near = vec![false; cw * ch];
            let r = 24i32;
            let mut seen = vec![false; cw * ch];
            let mut cnt = 0usize;
            let mut fr = 0usize;
            for p in ps.iter().filter(|p| p.material() == Some(mat)) {
                cnt += 1;
                if sp.temperature(p).1 > 0.5 {
                    fr += 1;
                }
                let (x, y) = (p.pos[0] as i32, p.pos[1] as i32);
                if x < 0 || y < 0 || x >= cw as i32 || y >= ch as i32 || seen[y as usize * cw + x as usize] {
                    continue;
                }
                seen[y as usize * cw + x as usize] = true;
                for yy in (y - r).max(0)..(y + r + 1).min(ch as i32) {
                    for xx in (x - r).max(0)..(x + r + 1).min(cw as i32) {
                        near[yy as usize * cw + xx as usize] = true;
                    }
                }
            }
            frozen = if cnt > 0 { fr as f32 / cnt as f32 } else { 0.0 };
            let far = |p: &Particle| {
                let (x, y) = (p.pos[0] as i32, p.pos[1] as i32);
                p.material() == Some(Material::Water) && x >= 0 && y >= 0 && x < cw as i32 && y < ch as i32 && !near[y as usize * cw + x as usize]
            };
            for p in ps.iter().filter(|p| far(p)) {
                let s = p.vel[0].hypot(p.vel[1]);
                s2 += (s * s) as f64;
                n += 1;
                mx = mx.max(s);
            }
            nse = nse.max(noise(&ps, &far));
        }
        let rms = if n > 0 { (s2 / n as f64).sqrt() as f32 } else { 0.0 };
        if win >= 8 {
            sum_rms += rms as f64;
            nwin += 1;
        }
        let st = w.sim.solver;
        let l = w.sim.download_ledger();
        println!(
            "{:6.2} {:8.3} {:8.2} {:8.3} {:7.2} {:6} {:5} {:6} {:7.3}",
            w.sim.sim_time(), rms, mx, nse, frozen, l.boiled_count.iter().sum::<u32>(), st.pressure_iters_max, st.viscosity_iters_max, st.dt * 1e3
        );
        w.sim.solver.reset_extremes();
    }
    println!("mean far rms over the last 8 s: {:.3} cells/s", sum_rms / nwin.max(1) as f64);
    // Per frame for 2 s: the mean velocity of the water in a box at the far end, and the
    // rms of velocity changes from one frame to the next there (jitter, cells/s).
    let in_box = |p: &Particle| p.material() == Some(Material::Water) && p.pos[0] > 0.75 * gw && p.pos[0] < 0.95 * gw && p.pos[1] > 0.05 * gh && p.pos[1] < 0.28 * gh;
    let mut prev: Option<Vec<[f32; 2]>> = None;
    let (mut jit, mut nj) = (0.0f64, 0usize);
    let mut means = Vec::new();
    for _ in 0..120 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
        let ps = w.sim.download_particles();
        let v: Vec<[f32; 2]> = ps.iter().map(|p| if in_box(p) { p.vel } else { [f32::NAN; 2] }).collect();
        let (mut mx, mut my, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for q in v.iter().filter(|q| !q[0].is_nan()) {
            mx += q[0] as f64;
            my += q[1] as f64;
            n += 1.0;
        }
        means.push(((mx / n.max(1.0)) as f32, (my / n.max(1.0)) as f32));
        if let Some(pv) = &prev {
            for (a, b) in pv.iter().zip(&v) {
                if !a[0].is_nan() && !b[0].is_nan() {
                    let (dx, dy) = ((b[0] - a[0]) as f64, (b[1] - a[1]) as f64);
                    jit += dx * dx + dy * dy;
                    nj += 1;
                }
            }
        }
        prev = Some(v);
    }
    // Surface height of the far columns, per frame for 3 s: its frame-to-frame change
    // after removing the slow motion (a 9-frame running mean), rms, in cells.
    let cols: Vec<f32> = (0..8).map(|k| (0.70 + 0.03 * k as f32) * gw).collect();
    let mut hs: Vec<Vec<f32>> = vec![Vec::new(); cols.len()];
    for _ in 0..180 {
        w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
        let ps = w.sim.download_particles();
        for (k, &x) in cols.iter().enumerate() {
            let mut ys: Vec<f32> = ps.iter().filter(|p| p.material() == Some(Material::Water) && (p.pos[0] - x).abs() < 1.5).map(|p| p.pos[1]).collect();
            ys.sort_by(f32::total_cmp);
            let top = &ys[ys.len().saturating_sub(6)..];
            hs[k].push(top.iter().sum::<f32>() / top.len().max(1) as f32);
        }
    }
    let (mut hf, mut nh) = (0.0f64, 0usize);
    for h in &hs {
        for i in 4..h.len() - 4 {
            let m = h[i - 4..=i + 4].iter().sum::<f32>() / 9.0;
            hf += ((h[i] - m) as f64).powi(2);
            nh += 1;
        }
    }
    println!("far surface: high-frequency height rms {:.4} cells; column 0 heights every 6 frames: {}", (hf / nh.max(1) as f64).sqrt(),
        hs[0].iter().step_by(6).map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "));
    let s: Vec<String> = means.iter().step_by(4).map(|m| format!("{:+.1}", m.1)).collect();
    println!("far box mean vy every 4 frames: {}", s.join(" "));
    println!("far box frame-to-frame jitter: {:.3} cells/s", (jit / nj.max(1) as f64).sqrt());
}
