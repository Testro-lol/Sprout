//! Off-screen tools: screenshots, benchmarks and physics validation. They drive the same
//! `World` and `Renderer` as the interactive app, without a window.

use std::{cell::RefCell, fs::File, io::BufWriter, path::Path, sync::Arc, time::Instant};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::CopyImageToBufferInfo,
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{
    gpu::{Builder, Gpu},
    gui::GuiPainter,
    render::{Renderer, View},
    ui::{self, UiState},
    sim::{FluidSim, ToolState, submit_and_wait},
    world::World,
};

/// An off-screen render target with CPU readback.
pub struct Offscreen {
    gpu: Arc<Gpu>,
    pub size: [u32; 2],
    pub view: Arc<ImageView>,
    host: Subbuffer<[u8]>,
    pub renderer: Renderer,
}

impl Offscreen {
    pub fn new(gpu: Arc<Gpu>, size: [u32; 2]) -> Offscreen {
        let format = Format::R8G8B8A8_UNORM;
        let image = Image::new(
            gpu.memory.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [size[0], size[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let host = Buffer::new_slice::<u8>(
            gpu.memory.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (size[0] * size[1] * 4) as u64,
        )
        .unwrap();
        Offscreen {
            renderer: Renderer::new(gpu.clone(), format),
            view: ImageView::new_default(image).unwrap(),
            host,
            size,
            gpu,
        }
    }

    /// Renders the simulation and blocks until the GPU is done. Does not read pixels.
    pub fn draw(&mut self, sim: &FluidSim, view: View) {
        let mut b = self.gpu.builder();
        self.renderer.render(&mut b, self.view.clone(), sim, view, |_| {});
        submit_and_wait(&self.gpu, b);
    }

    /// Notes which solids are wet at this moment (see `Renderer::track`).
    pub fn track(&mut self, sim: &FluidSim) {
        let mut b = self.gpu.builder();
        self.renderer.track(&mut b, sim);
        submit_and_wait(&self.gpu, b);
    }

    /// Renders the simulation and returns RGBA8 pixels.
    pub fn capture(&mut self, sim: &FluidSim, view: View) -> Vec<u8> {
        self.capture_with(sim, view, |_| {}, |_| {})
    }

    /// Like `capture`, with hooks that record before the render passes and inside the
    /// final one (used to composite the UI).
    pub fn capture_with(
        &mut self,
        sim: &FluidSim,
        view: View,
        before: impl FnOnce(&mut Builder),
        overlay: impl FnOnce(&mut Builder),
    ) -> Vec<u8> {
        let mut b = self.gpu.builder();
        before(&mut b);
        self.renderer.render(&mut b, self.view.clone(), sim, view, overlay);
        b.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(self.view.image().clone(), self.host.clone()))
            .unwrap();
        submit_and_wait(&self.gpu, b);
        self.host.read().unwrap().to_vec()
    }

    pub fn save_png(&mut self, sim: &FluidSim, view: View, path: &Path) {
        let pixels = self.capture(sim, view);
        self.write_png(&pixels, path);
    }

    pub fn write_png(&self, pixels: &[u8], path: &Path) {
        let file = BufWriter::new(File::create(path).expect("cannot create image file"));
        let mut encoder = png::Encoder::new(file, self.size[0], self.size[1]);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header().unwrap().write_image_data(pixels).unwrap();
    }
}

pub struct ShotOptions {
    pub map: usize,
    pub grid: [u32; 2],
    pub size: [u32; 2],
    /// Simulated seconds at which to save a frame.
    pub times: Vec<f32>,
    pub out: String,
    /// Draw the control panel too, exactly as the app would.
    pub with_ui: bool,
}

/// Runs a map and saves a frame at each requested time.
pub fn shot(gpu: Arc<Gpu>, o: ShotOptions, apply: &dyn Fn(&mut crate::sim::SimParams)) {
    let mut world = World::new(gpu.clone(), o.grid, o.map);
    apply(&mut world.sim.params);
    let mut off = Offscreen::new(gpu.clone(), o.size);
    let view = View::fit(o.grid, o.size);
    let start = Instant::now();
    world.sim.step_blocking(0, ToolState::default());
    for (k, &t) in o.times.iter().enumerate() {
        // In pieces of about 1/30 s, noting what is wet after each, as the app does every
        // frame. (A fixed number of pieces: `advance_blocking` stops within a sliver of its
        // goal, so "until the time is reached" would never end.)
        let from = world.sim.sim_time();
        let pieces = ((t as f64 - from) * 30.0).ceil().max(0.0) as u32;
        for k in 1..=pieces {
            let goal = from + (t as f64 - from) * k as f64 / pieces as f64;
            world.sim.advance_blocking((goal - world.sim.sim_time()).max(0.0), ToolState::default());
            off.track(&world.sim);
        }
        let path = if o.times.len() == 1 { o.out.clone() } else { o.out.replace(".png", &format!("_{k}.png")) };
        if o.with_ui {
            // As the app frames it: the tank above the dock.
            let view = View::fit_above(world.sim.grid, off.size, ui::DOCK_INSET);
            ui_frame(&gpu, &mut off, &mut world, view, Path::new(&path));
        } else {
            off.save_png(&world.sim, view, Path::new(&path));
        }
        println!("t = {t:6.2} s  ->  {path}   ({:.1} s wall)", start.elapsed().as_secs_f32());
    }
}

/// Renders one frame with the egui panel on top, without a window: the panel is laid out
/// against a synthetic screen and painted by the same `GuiPainter` the app uses.
fn ui_frame(gpu: &Arc<Gpu>, off: &mut Offscreen, world: &mut World, view: View, path: &Path) {
    let ctx = egui::Context::default();
    ui::style(&ctx);
    let mut state = UiState { grid: world.sim.grid, ..Default::default() };
    let painter = GuiPainter::new(gpu.clone(), off.renderer.final_subpass());
    let size = off.size;
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0] as f32, size[1] as f32))),
        ..Default::default()
    };
    world.sim.step_blocking(0, ToolState::default());
    let mut b = gpu.builder();
    world.sim.request_stats(&mut b);
    submit_and_wait(gpu, b);
    world.sim.poll_stats();

    // egui sizes windows over two passes and fades them in; run enough frames for the
    // panel to settle, keeping every texture update along the way.
    // The canvas overlay as the app draws it: the brush cursor and the map's toast.
    let mut overlay = ui::Overlay::default();
    let cursor = [world.sim.grid[0] as f32 * 0.62, world.sim.grid[1] as f32 * 0.7];
    let mut deltas = Vec::new();
    let mut shapes = Vec::new();
    let mut swatches = crate::editor::Swatches::new(gpu.clone());
    // `SPROUT_UI_LIQUIDS=1`: the picker and the editor open too (for the README pictures).
    if std::env::var_os("SPROUT_UI_LIQUIDS").is_some() {
        state.editor.picker_open = true;
        state.editor.duplicate(world, Material::Chocolate.index());
        state.editor.draft.viscosity *= 3.0;
    }
    // `SPROUT_UI_TOOL=name`: that tool selected; `SPROUT_UI_PAGE=world|display|solver|keys|
    // maps|stats`: that window open; `SPROUT_UI_HOVER=x,y`: the pointer there (points).
    if let Ok(name) = std::env::var("SPROUT_UI_TOOL")
        && let Some(tool) = ui::Tool::all().find(|t| t.name().eq_ignore_ascii_case(&name))
    {
        state.select(tool);
    }
    match std::env::var("SPROUT_UI_PAGE").as_deref() {
        Ok("world") => state.page = Some(ui::Page::World),
        Ok("display") => state.page = Some(ui::Page::Display),
        Ok("solver") => state.page = Some(ui::Page::Solver),
        Ok("keys") => state.page = Some(ui::Page::Keys),
        Ok("maps") => state.maps_open = true,
        Ok("stats") => state.stats_open = true,
        _ => {}
    }
    let hover: Option<egui::Pos2> = std::env::var("SPROUT_UI_HOVER").ok().and_then(|v| {
        let (x, y) = v.split_once(',')?;
        Some(egui::pos2(x.parse().ok()?, y.parse().ok()?))
    });
    for _ in 0..30 {
        state.frame_times.push(1.0 / 144.0);
    }
    let ghost = ui::placement(&state, None, cursor, 1.0, false).map(|p| (p, false));
    for frame in 0..60 {
        let mut input = egui::RawInput { time: Some(1.0 + frame as f64 / 60.0), ..input.clone() };
        if let Some(pos) = hover {
            input.events.push(egui::Event::PointerMoved(pos));
        }
        let out = ctx.run(input, |ctx| {
            ui::build(ctx, &mut state, world, &mut swatches);
            if frame == 0 && hover.is_none() && std::env::var_os("SPROUT_UI_TOOL").is_none() {
                overlay.toast_map(ctx, world.map);
            }
            let canvas = ui::Canvas { view, size, cursor: Some(cursor), pressed: false, grabbing: false, ghost, target: None, world };
            overlay.draw(ctx, &state, &canvas);
        });
        deltas.push(out.textures_delta);
        shapes = out.shapes;
    }
    let primitives = ctx.tessellate(shapes, 1.0);
    // The two hooks need the painter one after the other, never at once.
    let painter = RefCell::new(painter);
    let pixels = off.capture_with(
        &world.sim,
        view,
        |b| {
            for d in &deltas {
                painter.borrow_mut().update_textures(b, d);
            }
        },
        |b| painter.borrow_mut().paint(b, &primitives, 1.0, size),
    );
    off.write_png(&pixels, path);
}

// ---- diagnostics ---------------------------------------------------------------------

use crate::{
    maps::fill_rect,
    scene::Scene,
    sim::{Material, Particle, SimParams},
};

/// Bulk measurements of a particle snapshot.
#[derive(Clone, Copy, Debug, Default)]
pub struct Diag {
    /// Mean kinetic energy per particle, (cells/s)^2 / 2, density-weighted.
    pub ke: f32,
    /// Mean gravitational potential energy per particle, g * rho * y.
    pub pe: f32,
    pub max_speed: f32,
    pub mean_y: f32,
    /// 99th percentile of height: a robust "surface level".
    pub top_y: f32,
    pub max_x: f32,
}

pub fn diagnose(ps: &[Particle], params: &SimParams) -> Diag {
    if ps.is_empty() {
        return Diag::default();
    }
    let mut d = Diag::default();
    let mut ys: Vec<f32> = Vec::with_capacity(ps.len());
    let mut pe = 0.0f64;
    for p in ps {
        let rho = p.density(params);
        let v2 = p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1];
        d.ke += 0.5 * rho * v2;
        pe += (params.g_cells() * rho * p.pos[1]) as f64;
        d.max_speed = d.max_speed.max(v2.sqrt());
        d.mean_y += p.pos[1];
        d.max_x = d.max_x.max(p.pos[0]);
        ys.push(p.pos[1]);
    }
    d.ke /= ps.len() as f32;
    d.pe = (pe / ps.len() as f64) as f32;
    d.mean_y /= ps.len() as f32;
    let k = ((ys.len() as f32 * 0.99) as usize).min(ys.len() - 1);
    ys.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    d.top_y = ys[k];
    d
}

/// Debug probe: a resting pool (or, with `map`, a built-in map) sampled over time.
pub fn probe(gpu: Arc<Gpu>, grid: [u32; 2], apply: &dyn Fn(&mut SimParams), map: Option<usize>, depth: f32, seconds: f32, out: Option<&str>) {
    let mut world = match map {
        Some(m) => World::new(gpu.clone(), grid, m),
        None => {
            let scene = Scene::new(grid);
            let mut ps = Vec::new();
            fill_rect(&scene, [0.0, 0.0, grid[0] as f32, depth], Material::Water, &mut ps);
            World::custom(gpu.clone(), scene, ps)
        }
    };
    apply(&mut world.sim.params);
    world.sim.step_blocking(0, ToolState::default());
    let samples = 20;
    println!("probe: {} particles, {seconds} s", world.sim.slots());
    println!(
        "{:>7} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>8} {:>6} {:>6} {:>8} {:>8}",
        "t", "ke", "pe", "ke+pe", "max_speed", "mean_y", "top_y", "dt ms", "p it", "v it", "u grid", "u part"
    );
    let mut worst = (0.0f32, 0.0f32);
    let e0 = crate::sim::heat_content(&world.sim.download_particles(), &world.sim.params);
    for s in 0..=samples {
        let d = diagnose(&world.sim.download_particles(), &world.sim.params);
        let _ = world.sim.download_pressure();
        let st = world.sim.solver;
        println!(
            "{:7.3} {:10.3} {:10.1} {:10.1} {:10.2} {:9.3} {:9.3} {:8.3} {:6} {:6} {:8.1} {:8.1}",
            world.sim.sim_time(), d.ke, d.pe, d.ke + d.pe, d.max_speed, d.mean_y, d.top_y, st.dt * 1e3,
            st.pressure_iters_max, st.viscosity_iters_max, st.umax_grid, st.umax_part
        );
        worst = (worst.0.max(st.pressure_residual_max), worst.1.max(st.viscosity_residual_max));
        world.sim.solver.reset_extremes();
        if s < samples {
            world.sim.advance_blocking(seconds as f64 / samples as f64, ToolState::default());
        }
    }
    if world.sim.thermal() {
        let ps = world.sim.download_particles();
        let l = world.sim.download_ledger();
        let e1 = crate::sim::heat_content(&ps, &world.sim.params);
        let lost: u32 = l.boiled_count.iter().sum();
        println!(
            "heat (J/m): content {:.4e} -> {:.4e}, change {:+.4e}; in from air {:.3e} solids {:.3e} brush {:.3e}; out to air {:.3e} solids {:.3e} brush {:.3e} boiled {:.3e} ({} particles lost); unaccounted {:+.3e}",
            e0, e1, e1 - e0, l.from_air, l.from_solids, l.heated, l.to_air, l.to_solids, l.cooled, l.boiled, lost, (e1 - e0) - l.net_in()
        );
    }
    let st = world.sim.solver;
    println!(
        "substeps {}  unconverged pressure {} viscosity {}  worst residual pressure {:.1e} viscosity {:.1e}  recording {:?}",
        st.substeps, st.pressure_unconverged, st.viscosity_unconverged, worst.0, worst.1, world.sim.recorded_iters()
    );
    if let Some(path) = out {
        let size = [1280, 720];
        Offscreen::new(gpu, size).save_png(&world.sim, View::fit(grid, size), Path::new(path));
    }
}

// ---- benchmark -----------------------------------------------------------------------

/// Simulated time each benchmark frame asks for, s.
const FRAME: f32 = 1.0 / 60.0;

/// Runs `frames` frames of `FRAME` simulated seconds each, the way the app would record
/// them (enough substeps at the last known step length), and returns the wall time of each
/// frame in ms. Every frame is submitted and waited on.
fn time_frames(world: &mut World, frames: u32) -> Vec<f64> {
    let gpu = world.sim.gpu();
    let mut out = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        let start = Instant::now();
        let dt = world.sim.dt_estimate();
        let n = if world.sim.params.fixed_hz > 0.0 { (FRAME / dt).round().max(1.0) as u32 } else { (FRAME / dt).ceil().max(1.0) as u32 };
        let mut b = gpu.builder();
        world.sim.record(&mut b, n, FRAME, ToolState::default());
        submit_and_wait(&gpu, b);
        world.sim.poll_readbacks();
        out.push(start.elapsed().as_secs_f64() * 1e3);
    }
    out
}

fn pct(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q) as usize]
}

/// Measures simulation and rendering throughput on a few workloads and prints a table.
///
/// Each scene runs for `warmup` simulated seconds, then 120 frames of 1/60 s simulated time
/// are timed (wall clock, submitted and waited on, CPU recording included). What matters is
/// simulated time per wall-clock second: a solver that takes fewer, longer steps is only
/// faster if it still covers the same simulated time.
pub fn bench(gpu: Arc<Gpu>, apply: &dyn Fn(&mut SimParams), grids: &[[u32; 2]], maps: &[usize], size: [u32; 2], warmup: f32) {
    println!("device: {}", gpu.device_name());
    println!("render target: {}x{} (off-screen), each map measured after {warmup} s of simulated time\n", size[0], size[1]);
    println!(
        "{:<14} {:>9} {:>10} | {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} | {:>8} | {:>9}",
        "map", "grid", "particles", "frame", "p50", "p95", "max", "substeps", "p iters", "render", "sim speed"
    );
    println!(
        "{:<14} {:>9} {:>10} | {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} | {:>8} | {:>9}",
        "", "", "", "mean ms", "ms", "ms", "ms", "/ sim s", "max", "ms", "x real"
    );
    for &grid in grids {
        let mut off = Offscreen::new(gpu.clone(), size);
        let view = View::fit(grid, size);
        for &map in maps {
            let mut world = World::new(gpu.clone(), grid, map);
            apply(&mut world.sim.params);
            world.sim.step_blocking(0, ToolState::default());
            // Let the scene get going: a moving, mixed fluid is the workload, not the
            // tidy lattice a map starts as.
            time_frames(&mut world, ((warmup / FRAME) as u32).max(1));

            world.sim.solver.reset_extremes();
            let (t0, s0) = (world.sim.sim_time(), world.sim.solver.substeps);
            let frames = time_frames(&mut world, 120);
            let wall: f64 = frames.iter().sum::<f64>() / 1e3;
            let simulated = world.sim.sim_time() - t0;
            let substeps = (world.sim.solver.substeps - s0) as f64 / simulated.max(1e-9);
            let iters = world.sim.solver.pressure_iters_max;

            let start = Instant::now();
            let n = 60;
            for _ in 0..n {
                off.draw(&world.sim, view);
            }
            let render = start.elapsed().as_secs_f64() * 1e3 / n as f64;
            let mean = frames.iter().sum::<f64>() / frames.len() as f64;
            println!(
                "{:<14} {:>9} {:>10} | {:>8.2} {:>8.2} {:>8.2} {:>8.2} | {:>8.0} {:>8} | {:>8.3} | {:>9.2}",
                crate::maps::MAPS[map].name,
                format!("{}x{}", grid[0], grid[1]),
                world.sim.slots(),
                mean,
                pct(&frames, 0.5),
                pct(&frames, 0.95),
                pct(&frames, 1.0),
                substeps,
                iters,
                render,
                simulated / wall
            );
        }
    }
    println!(
        "\nframe: wall time to simulate 1/60 s (no rendering).  substeps: per simulated second (CFL-sized).\n\
         p iters: most pressure iterations one substep needed.  sim speed: simulated seconds per wall second\n\
         with the simulation alone; above 1 the scene can run in real time with time left to render."
    );
}

/// Splits the cost of simulating 1/60 s frames into CPU recording and GPU execution.
pub fn profile(gpu: Arc<Gpu>, apply: &dyn Fn(&mut SimParams), grid: [u32; 2], map: usize, warmup: f32) {
    let mut world = World::new(gpu.clone(), grid, map);
    apply(&mut world.sim.params);
    world.sim.step_blocking(0, ToolState::default());
    time_frames(&mut world, ((warmup / FRAME) as u32).max(1));
    let (mut rec, mut exec) = (0.0f64, 0.0f64);
    let n = 100;
    let s0 = world.sim.solver.substeps;
    for _ in 0..n {
        let t0 = Instant::now();
        let dt = world.sim.dt_estimate();
        let steps = (FRAME / dt).ceil().max(1.0) as u32;
        let mut b = gpu.builder();
        world.sim.record(&mut b, steps, FRAME, ToolState::default());
        let cb = b.build().unwrap();
        let t1 = Instant::now();
        use vulkano::{command_buffer::PrimaryCommandBufferAbstract, sync::GpuFuture};
        cb.execute(gpu.queue.clone()).unwrap().then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        let t2 = Instant::now();
        world.sim.poll_readbacks();
        rec += (t1 - t0).as_secs_f64();
        exec += (t2 - t1).as_secs_f64();
    }
    let substeps = (world.sim.solver.substeps - s0) as f64;
    println!("recorded iterations (pressure, second pressure, viscosity): {:?}", world.sim.recorded_iters());
    for (rebind, fine) in [(true, false), (false, false), (true, true), (false, true)] {
        let ms = world.sim.dispatch_floor(2000, rebind, fine);
        println!("dispatch floor: {:.2} us per {} dispatch{}", ms * 1e3, if fine { "full-grid cg_scale" } else { "single-group" }, if rebind { " (rebinding each time)" } else { "" });
    }
    for _ in 0..2 {
        println!("per-frame optics passes: {:.3} ms per frame", world.sim.profile_optics(300));
    }
    for _ in 0..2 {
        println!("pressure V-cycle: {:.1} us each", world.sim.vcycle_floor(500) * 1e3);
    }
    // Rendering: `n` frames recorded into one submission (GPU time per frame), and the CPU
    // time to record one. Window-sized target (`--size`, default 1600 x 900).
    {
        use vulkano::{command_buffer::PrimaryCommandBufferAbstract, sync::GpuFuture};
        let size = std::env::var("SPROUT_PROFILE_SIZE").ok().and_then(|s| crate::parse_pair(&s)).unwrap_or([1600, 900]);
        let mut off = Offscreen::new(gpu.clone(), size);
        let view = View::fit(grid, size);
        off.draw(&world.sim, view);
        for _ in 0..2 {
            let n = 100;
            let t0 = Instant::now();
            let mut b = gpu.builder();
            for _ in 0..n {
                off.renderer.render(&mut b, off.view.clone(), &world.sim, view, |_| {});
            }
            let cb = b.build().unwrap();
            let t1 = Instant::now();
            cb.execute(gpu.queue.clone()).unwrap().then_signal_fence_and_flush().unwrap().wait(None).unwrap();
            let t2 = Instant::now();
            println!(
                "render {}x{}: GPU {:.3} ms per frame, recording {:.3} ms per frame",
                size[0], size[1], (t2 - t1).as_secs_f64() * 1e3 / n as f64, (t1 - t0).as_secs_f64() * 1e3 / n as f64
            );
        }
        // One frame with a timestamp after each pass.
        use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
        let pool = QueryPool::new(gpu.device.clone(), QueryPoolCreateInfo { query_count: 6, ..QueryPoolCreateInfo::query_type(QueryType::Timestamp) }).unwrap();
        let mut best = [f64::MAX; 5];
        let mut full_best = f64::MAX;
        for rep in 0..40 {
            let full = rep % 2 == 1;
            let mut b = gpu.builder();
            unsafe { b.reset_query_pool(pool.clone(), 0..6) }.unwrap();
            off.renderer.timing = Some(pool.clone());
            off.renderer.full_composite = full;
            off.renderer.render(&mut b, off.view.clone(), &world.sim, view, |_| {});
            off.renderer.full_composite = false;
            off.renderer.timing = None;
            crate::sim::submit_and_wait(&gpu, b);
            let mut ticks = [0u64; 6];
            pool.get_results(0..6, &mut ticks, QueryResultFlags::WAIT).unwrap();
            let period = gpu.device.physical_device().properties().timestamp_period as f64 * 1e-6;
            if full {
                full_best = full_best.min(ticks[5].saturating_sub(ticks[4]) as f64 * period);
                continue;
            }
            for k in 0..5 {
                best[k] = best[k].min(ticks[k + 1].saturating_sub(ticks[k]) as f64 * period);
            }
        }
        println!("composite with every effect compiled in: {full_best:.3} ms");
        println!("render passes, ms (best of 20, the variant this tank gets): optics {:.3}, splat {:.3}, vapour {:.3}, wide field {:.3}, composite {:.3}", best[0], best[1], best[2], best[3], best[4]);
        // The composite variant this tank gets against the full shader, on the same state.
        let mine = off.capture(&world.sim, view);
        off.renderer.full_composite = true;
        let full = off.capture(&world.sim, view);
        off.renderer.full_composite = false;
        let (mut differ, mut most) = (0usize, 0u8);
        for (a, b) in mine.iter().zip(&full) {
            let d = a.abs_diff(*b);
            if d > 0 {
                differ += 1;
                most = most.max(d);
            }
        }
        println!("composite variant {:?} against the full shader: {differ} channel values differ, by at most {most}", world.sim.render_features());
    }
    // Best of several runs per pass: one timed substep alone is at the mercy of the clocks.
    let reps = std::env::var("SPROUT_PROFILE_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1u32).max(1);
    let mut passes = world.sim.profile_passes();
    for _ in 1..reps {
        for (best, now) in passes.iter_mut().zip(world.sim.profile_passes()) {
            best.1 = best.1.min(now.1);
        }
    }
    let line: Vec<String> = passes.iter().map(|(n, ms)| format!("{n} {ms:.3}")).collect();
    println!("one substep, GPU ms by pass: {}", line.join(", "));
    println!(
        "{} {}x{} {} particles: {:.1} substeps per frame; per frame: record {:.3} ms, submit+execute {:.3} ms; per substep: record {:.3} ms, execute {:.3} ms",
        crate::maps::MAPS[map].name, grid[0], grid[1], world.sim.slots(), substeps / n as f64,
        rec * 1e3 / n as f64, exec * 1e3 / n as f64, rec * 1e3 / substeps, exec * 1e3 / substeps
    );
}

// ---- gallery -------------------------------------------------------------------------

/// Renders a fixed set of pictures that covers every liquid, every way two of them meet and
/// Connected pieces of liquid (8-connected occupied cells): how many, how many of them are
/// small (under 40 particles: droplets), and how many particles those hold.
pub fn blob_stats(ps: &[Particle], grid: [u32; 2]) -> (usize, usize, usize) {
    let (w, h) = (grid[0] as usize, grid[1] as usize);
    let mut count = vec![0u32; w * h];
    for p in ps {
        let (x, y) = (p.pos[0] as usize, p.pos[1] as usize);
        if x < w && y < h {
            count[y * w + x] += 1;
        }
    }
    let mut label = vec![usize::MAX; w * h];
    let (mut blobs, mut drops, mut dropped) = (0, 0, 0);
    for start in 0..w * h {
        if count[start] == 0 || label[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        label[start] = start;
        let mut n = 0;
        while let Some(c) = stack.pop() {
            n += count[c] as usize;
            let (x, y) = ((c % w) as i64, (c / w) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let k = ny as usize * w + nx as usize;
                    if count[k] > 0 && label[k] == usize::MAX {
                        label[k] = start;
                        stack.push(k);
                    }
                }
            }
        }
        blobs += 1;
        if n < 40 {
            drops += 1;
            dropped += n;
        }
    }
    (blobs, drops, dropped)
}

/// every tool, at several zoom levels and resolutions (`sprout gallery DIR`). The scenes are
/// deterministic, so two builds can be compared picture by picture.
pub fn gallery(gpu: Arc<Gpu>, dir: &str, only: Option<&str>, apply: &dyn Fn(&mut SimParams)) {
    use crate::{scene::WALL, sim::Current, world::Source};
    const SIZE: [u32; 2] = [1280, 720];
    const G: [u32; 2] = [512, 288];
    std::fs::create_dir_all(dir).expect("cannot create the gallery directory");
    let off = RefCell::new(Offscreen::new(gpu.clone(), SIZE));
    let start = Instant::now();
    let wants = |name: &str| only.is_none_or(|o| name.starts_with(o));
    let save = |world: &World, name: &str, view: View| {
        off.borrow_mut().save_png(&world.sim, view, &Path::new(dir).join(format!("{name}.png")));
        println!("{name:<22} t = {:5.2} s   ({:.1} s wall)", world.sim.sim_time(), start.elapsed().as_secs_f32());
        if std::env::var_os("SPROUT_DEBUG_LOOK").is_some() {
            println!("    look/flakes/diffuse draw commands {:?}, slots {}", world.sim.debug_look_counts(), world.sim.slots());
        }
    };
    let map = |name: &str, grid: [u32; 2], t: f64| {
        let index = crate::maps::MAPS.iter().position(|m| m.name == name).expect("built-in map");
        let mut world = World::new(gpu.clone(), grid, index);
        apply(&mut world.sim.params);
        // Every scene builds its own world, so drop what the last one left cached.
        off.borrow_mut().renderer.invalidate();
        world.sim.step_blocking(0, ToolState::default());
        world.sim.advance_blocking(t, ToolState::default());
        world
    };
    let pool = |fills: &[([f32; 4], Material)], shapes: &[crate::scene::Shape]| {
        let mut scene = Scene::new(G);
        scene.shapes.extend_from_slice(shapes);
        let mut ps = Vec::new();
        for &(r, m) in fills {
            fill_rect(&scene, r, m, &mut ps);
        }
        let mut world = World::custom(gpu.clone(), scene, ps);
        apply(&mut world.sim.params);
        off.borrow_mut().renderer.invalidate();
        world.sim.step_blocking(0, ToolState::default());
        world
    };
    // A brush held still for `seconds`, in 1/60 s frames as the app drives it.
    let hold = |w: &mut World, source: Source, material: Material, at: [f32; 2], radius: f32, rate: f32, seconds: f32| {
        for _ in 0..(seconds * 60.0).round() as usize {
            w.emit(source, material, at, radius, rate, 1.0 / 60.0);
            w.sim.advance_blocking(1.0 / 60.0, source.nozzle(at, radius, rate));
            off.borrow_mut().track(&w.sim);
        }
        w.end_stroke();
    };
    // Nothing held, for `seconds`, in the same frames.
    let wait = |w: &mut World, seconds: f32| {
        for _ in 0..(seconds * 60.0).round() as usize {
            w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
            off.borrow_mut().track(&w.sim);
        }
    };
    let fit = View::fit(G, SIZE);
    let zoom = |center: [f32; 2], factor: f32| View { center, zoom: fit.zoom * factor };

    // Each liquid on its own, moving.
    if wants("water") {
        let w = map("Dam Break", G, 1.15);
        save(&w, "water_dam", fit);
        save(&w, "water_dam_zoom", zoom([300.0, 60.0], 3.0));
    }
    if wants("honey") {
        let mut w = pool(&[([150.0, 0.0, 250.0, 150.0], Material::Honey)], &[crate::scene::Shape::Circle { c: [330.0, 30.0], r: 22.0 }]);
        w.sim.advance_blocking(2.5, ToolState::default());
        save(&w, "honey_slump", fit);
        save(&w, "honey_slump_zoom", zoom([300.0, 45.0], 3.0));
    }
    if wants("oil") {
        let mut w = pool(&[([150.0, 0.0, 250.0, 150.0], Material::Oil)], &[crate::scene::Shape::Circle { c: [330.0, 30.0], r: 22.0 }]);
        w.sim.advance_blocking(0.9, ToolState::default());
        save(&w, "oil_collapse", fit);
        save(&w, "oil_collapse_zoom", zoom([300.0, 45.0], 3.0));
    }
    // Liquids meeting.
    if wants("playground") {
        let w = map("Playground", G, 1.3);
        save(&w, "playground_impact", fit);
        save(&w, "playground_impact_zoom", zoom([130.0, 45.0], 3.0));
        save(&w, "playground_honey_zoom", zoom([330.0, 60.0], 3.0));
    }
    if wants("overturn") {
        let mut w = map("Upside Down", G, 5.0);
        save(&w, "overturn_5s", fit);
        save(&w, "overturn_5s_zoom", zoom([256.0, 90.0], 3.0));
        w.sim.advance_blocking(20.0, ToolState::default());
        save(&w, "overturn_25s", fit);
        save(&w, "overturn_25s_zoom", zoom([256.0, 110.0], 4.0));
    }
    if wants("columns") {
        let w = map("Three Columns", G, 2.0);
        save(&w, "columns_2s", fit);
    }
    if wants("mix") {
        let mut w = pool(&[([0.0, 0.0, 512.0, 110.0], Material::Water)], &[]);
        hold(&mut w, Source::Pour, Material::Honey, [256.0, 200.0], 9.0, 12000.0, 2.0);
        save(&w, "mix_honey_pour", fit);
        save(&w, "mix_honey_pour_zoom", zoom([256.0, 75.0], 3.0));
    }
    // Blood stirred into water: a dispersing, half-mixed plume (per-cell concentration).
    if wants("mix") {
        let mut w = pool(&[([0.0, 0.0, 512.0, 110.0], Material::Water)], &[]);
        hold(&mut w, Source::Pour, Material::Blood, [256.0, 200.0], 9.0, 12000.0, 1.5);
        wait(&mut w, 3.0);
        save(&w, "mix_blood", fit);
        save(&w, "mix_blood_zoom", zoom([256.0, 60.0], 3.0));
        save(&w, "mix_blood_close", zoom([236.0, 40.0], 9.0));
    }
    // Blood poured onto honey over water (two liquids of one phase group that a concentration
    // alone does not tell apart), and ferrofluid stirred into oil.
    if wants("mix") {
        let mut w = pool(&[([0.0, 0.0, 512.0, 50.0], Material::Water), ([0.0, 50.0, 512.0, 100.0], Material::Honey)], &[]);
        hold(&mut w, Source::Pour, Material::Blood, [256.0, 200.0], 9.0, 12000.0, 1.5);
        wait(&mut w, 2.0);
        save(&w, "mix_blood_honey", zoom([256.0, 70.0], 3.0));
        save(&w, "mix_blood_honey_close", zoom([256.0, 80.0], 8.0));
        save(&w, "mix_blood_honey_far", fit);
        save(&w, "mix_blood_honey_mid", zoom([256.0, 70.0], 1.5));
        let mut w = pool(&[([0.0, 0.0, 512.0, 80.0], Material::Oil)], &[]);
        hold(&mut w, Source::Pour, Material::Ferrofluid, [256.0, 200.0], 6.0, 8000.0, 1.0);
        wait(&mut w, 1.5);
        save(&w, "mix_ferro_oil", zoom([256.0, 50.0], 3.0));
    }
    // Ferrofluid poured into the Playground's pool after the honey has fallen in (the user's
    // screenshot of 2026-11-16: dark outlines round the ferrofluid in the honey and water).
    if wants("ferro_pool") {
        let mut w = map("Playground", G, 4.0);
        hold(&mut w, Source::Pour, Material::Ferrofluid, [300.0, 150.0], 9.0, 8000.0, 3.0);
        wait(&mut w, 6.0);
        save(&w, "ferro_pool", fit);
        save(&w, "ferro_pool_zoom", zoom([300.0, 40.0], 3.0));
        save(&w, "ferro_pool_close", zoom([300.0, 40.0], 8.0));
    }
    // A magnet under and over a pool of ferrofluid on 1 cm cells, as the Magnet tool places
    // them (README, "Magnetism"): the dish heaps up and spikes, and comes to rest. And six
    // consecutive frames of the first, for the flicker count. Kinetic energy is printed.
    if wants("magnet") {
        let cup = |x: f32| crate::scene::Shape::Box { c: [x, 30.0], half: [4.0, 28.0], angle: 0.0 };
        for (name, at, deg) in [("magnet_under", [256.0, WALL + 3.0], 0.0f32), ("magnet_over", [256.0, 28.0], 180.0)] {
            let mut w = pool(&[([152.0, WALL, 360.0, WALL + 14.0], Material::Ferrofluid)], &[cup(146.0), cup(366.0)]);
            w.add_magnet(crate::sim::Magnet::new(at, 6.0, deg, crate::sim::MAGNET_STRENGTH_DEFAULT));
            for k in 0..8 {
                wait(&mut w, 1.0);
                let d = diagnose(&w.sim.download_particles(), &w.sim.params);
                println!("{name} t = {} s: kinetic energy {:.1}, fastest {:.1} cells/s", k + 1, d.ke, d.max_speed);
            }
            save(&w, name, fit);
            save(&w, &format!("{name}_zoom"), zoom([256.0, 25.0], 4.0));
            if name == "magnet_under" {
                for k in 0..6 {
                    save(&w, &format!("motion_magnet_{k}"), zoom([256.0, 25.0], 4.0));
                    wait(&mut w, 1.0 / 60.0);
                }
            }
        }
    }
    // Small droplets inside another liquid: clusters of 1, 2, 4, 9 and 16 particles of lava and
    // of oil in water, and of water in lava and in oil, shortly after they were placed (and six
    // consecutive frames of the first, for the flicker count).
    if wants("droplets") {
        let seed = |host: Material, guests: [Material; 2]| {
            let scene = Scene::new(G);
            let mut ps = Vec::new();
            fill_rect(&scene, [0.0, 0.0, 512.0, 120.0], host, &mut ps);
            for (row, &m) in guests.iter().enumerate() {
                for (k, n) in [1usize, 2, 4, 9, 16].into_iter().enumerate() {
                    let c = [176.0 + 40.0 * k as f32, 50.0 + 40.0 * row as f32];
                    let mut idx: Vec<usize> = (0..ps.len()).collect();
                    let d2 = |p: &Particle| (p.pos[0] - c[0]).powi(2) + (p.pos[1] - c[1]).powi(2);
                    idx.sort_by(|&a, &b| d2(&ps[a]).total_cmp(&d2(&ps[b])));
                    for &i in &idx[..n] {
                        ps[i] = Particle::new(m, ps[i].pos, [0.0, 0.0]);
                    }
                }
            }
            let mut w = World::custom(gpu.clone(), scene, ps);
            apply(&mut w.sim.params);
            off.borrow_mut().renderer.invalidate();
            w.sim.step_blocking(0, ToolState::default());
            w
        };
        for (name, host, guests) in [
            ("droplets_in_water", Material::Water, [Material::Lava, Material::Oil]),
            ("droplets_in_lava", Material::Lava, [Material::Water, Material::Oil]),
            ("droplets_in_oil", Material::Oil, [Material::Water, Material::Lava]),
        ] {
            let mut w = seed(host, guests);
            wait(&mut w, 0.15);
            save(&w, name, zoom([256.0, 70.0], 2.4));
            if host == Material::Water {
                for k in 0..6 {
                    save(&w, &format!("motion_droplets_{k}"), zoom([256.0, 70.0], 2.4));
                    wait(&mut w, 1.0 / 60.0);
                }
            }
        }
    }
    // The effects (README, "Look"): glow and crust of lava, stains, glints, thin films,
    // a metal mirroring a melt.
    if wants("fx") {
        let wall = |x: f32| crate::scene::Shape::Box { c: [x, 50.0], half: [4.0, 48.0], angle: 0.0 };
        // A cup of lava, freshly poured, then cooling.
        let mut w = pool(&[([200.0, WALL, 312.0, 60.0], Material::Lava)], &[wall(196.0), wall(316.0)]);
        wait(&mut w, 1.0);
        save(&w, "fx_lava", zoom([256.0, 60.0], 3.0));
        wait(&mut w, 14.0);
        save(&w, "fx_lava_cooling", zoom([256.0, 60.0], 3.0));
        // Blood sprayed at a wall and left to drain.
        let mut w = pool(&[], &[wall(330.0)]);
        hold(&mut w, Source::Spray, Material::Blood, [270.0, 60.0], 8.0, 20000.0, 1.0);
        save(&w, "fx_blood_spray", zoom([300.0, 50.0], 3.0));
        wait(&mut w, 4.0);
        save(&w, "fx_blood_stains", zoom([300.0, 50.0], 3.0));
        save(&w, "fx_blood_stains_zoom8", zoom([325.0, 40.0], 8.0));
        // Honey over a ball, close.
        let mut w = pool(&[], &[crate::scene::Shape::Circle { c: [256.0, 80.0], r: 30.0 }]);
        hold(&mut w, Source::Pour, Material::Honey, [250.0, 200.0], 7.0, 8000.0, 2.5);
        save(&w, "fx_honey", zoom([256.0, 100.0], 4.0));
        save(&w, "fx_honey_zoom8", zoom([240.0, 112.0], 8.0));
        // A thin layer of oil on water.
        let mut w = pool(&[([0.0, 0.0, 512.0, 60.0], Material::Water), ([0.0, 60.0, 512.0, 64.0], Material::Oil)], &[]);
        hold(&mut w, Source::Pour, Material::Oil, [256.0, 160.0], 5.0, 4000.0, 1.0);
        wait(&mut w, 1.5);
        save(&w, "fx_oil_film", zoom([256.0, 62.0], 5.0));
        // Mercury beside a lava pool: it mirrors the glow.
        let mut w = pool(&[([150.0, WALL, 250.0, 40.0], Material::Mercury), ([262.0, WALL, 362.0, 40.0], Material::Lava)], &[wall(256.0)]);
        wait(&mut w, 1.0);
        save(&w, "fx_mercury_lava", zoom([256.0, 35.0], 3.5));
        // Lava poured onto the floor, spreading: the flow draws its glow out into streaks.
        let mut w = pool(&[], &[]);
        hold(&mut w, Source::Pour, Material::Lava, [256.0, 120.0], 8.0, 6000.0, 2.0);
        save(&w, "fx_lava_pour", zoom([256.0, 30.0], 3.0));
        // Lava poured into the Playground's pool and left for a minute, partly frozen around
        // its molten core (the user's screenshot of 2026-11-15); 30 frames 1/60 s apart.
        if std::env::var("SPROUT_LAVA_POOL").is_ok() {
            let mut w = map("Playground", G, 1.0);
            w.sim.enable_heat();
            hold(&mut w, Source::Pour, Material::Lava, [250.0, 150.0], 9.0, 6000.0, 4.0);
            wait(&mut w, 60.0);
            let view = zoom([250.0, 40.0], 3.0);
            let mut frames = Vec::new();
            for k in 0..30 {
                frames.push(off.borrow_mut().capture(&w.sim, view));
                if k % 10 == 0 {
                    save(&w, &format!("fx_lava_pool_{k}"), view);
                }
                wait(&mut w, 1.0 / 60.0);
            }
            // Inside the lava only: rows 560 - 630 of the 720 (below its surface, above the floor).
            let band = |f: &Vec<u8>| f[560 * 1280 * 4..630 * 1280 * 4].to_vec();
            let frames: Vec<Vec<u8>> = frames.iter().map(band).collect();
            let diffs: Vec<f64> = frames
                .windows(2)
                .map(|f| f[0].iter().zip(&f[1]).map(|(a, b)| (*a as i32 - *b as i32).abs() as f64).sum::<f64>() / f[0].len() as f64)
                .collect();
            let far: f64 = frames[0].iter().zip(&frames[29]).map(|(a, b)| (*a as i32 - *b as i32).abs() as f64).sum::<f64>() / frames[0].len() as f64;
            println!(
                "fx_lava_pool: mean change frame to frame {:.3} levels (largest {:.3}); frame 0 to 29 {:.3}",
                diffs.iter().sum::<f64>() / diffs.len() as f64,
                diffs.iter().cloned().fold(0.0, f64::max),
                far
            );
        }
        // Lava resting under oil until it crusts over (the user's report of 2026-11-15: the
        // crust's pattern stuttered back and forth); frames 1/60 s apart and 1 s apart, and
        // how many pixels change from frame to frame (README, "Look": the flicker metric).
        let mut w = pool(&[([100.0, WALL, 412.0, 40.0], Material::Lava), ([WALL, 40.0, 509.0, 90.0], Material::Oil), ([WALL, WALL, 100.0, 40.0], Material::Oil), ([412.0, WALL, 509.0, 40.0], Material::Oil)], &[]);
        w.sim.enable_heat();
        wait(&mut w, 30.0);
        let mut frames = Vec::new();
        for k in 0..6 {
            frames.push(off.borrow_mut().capture(&w.sim, zoom([256.0, 30.0], 3.0)));
            if k == 0 {
                save(&w, "fx_lava_crust", zoom([256.0, 30.0], 3.0));
            }
            wait(&mut w, 1.0 / 60.0);
        }
        // (And the mean change of a pixel channel, and the share of pixels whose largest
        // channel change is over 4 levels: the slow sliding the flicker count passes over.)
        let pairs: Vec<(f64, f64)> = frames
            .windows(2)
            .map(|f| {
                let px = f[0].as_chunks::<4>().0.iter().zip(f[1].as_chunks::<4>().0);
                let (mut sum, mut moved, mut n) = (0.0, 0.0, 0.0);
                for (a, b) in px {
                    let d = (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
                    sum += (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs() as f64).sum::<f64>() / 3.0;
                    moved += (d > 4) as u32 as f64;
                    n += 1.0;
                }
                (sum / n, moved / n)
            })
            .collect();
        let m = pairs.len() as f64;
        println!(
            "fx_lava_crust: {:.3} % of pixels flicker; mean change {:.3} levels, {:.2} % of pixels change by more than 4",
            100.0 * crate::stress::flicker(&frames),
            pairs.iter().map(|p| p.0).sum::<f64>() / m,
            100.0 * pairs.iter().map(|p| p.1).sum::<f64>() / m
        );
        wait(&mut w, 1.0);
        save(&w, "fx_lava_crust_1s", zoom([256.0, 30.0], 3.0));
        // Six frames, to look for flicker in the glow and crust.
        let mut w = pool(&[([200.0, WALL, 312.0, 60.0], Material::Lava)], &[wall(196.0), wall(316.0)]);
        wait(&mut w, 6.0);
        for k in 0..6 {
            save(&w, &format!("motion_fx_lava_{k}"), zoom([256.0, 70.0], 3.0));
            wait(&mut w, 1.0 / 60.0);
        }
    }
    // Steam (README, "Steam"): a cup of water held over the Heat brush until it boils, close,
    // and six consecutive frames of it for flicker; then lava poured into a shallow pan.
    if wants("steam") {
        let mut w = pool(&[([200.0, WALL, 312.0, WALL + 30.0], Material::Water)], &[crate::scene::Shape::Box { c: [196.0, 50.0], half: [4.0, 48.0], angle: 0.0 }, crate::scene::Shape::Box { c: [316.0, 50.0], half: [4.0, 48.0], angle: 0.0 }]);
        w.sim.enable_heat();
        let brush = ToolState::heat([256.0, WALL + 5.0], 14.0, false);
        for _ in 0..(8 * 60) {
            w.sim.advance_blocking(1.0 / 60.0, brush);
            off.borrow_mut().track(&w.sim);
        }
        save(&w, "steam_boil", fit);
        save(&w, "steam_boil_zoom", zoom([256.0, 45.0], 3.0));
        for k in 0..6 {
            save(&w, &format!("motion_steam_{k}"), zoom([256.0, 45.0], 3.0));
            w.sim.advance_blocking(1.0 / 60.0, brush);
            off.borrow_mut().track(&w.sim);
        }
        // Steam meeting an object: a cup boiling under a tilted shelf, which the plume has
        // to spread along and spill around.
        let mut w = pool(
            &[([200.0, WALL, 312.0, WALL + 30.0], Material::Water)],
            &[
                crate::scene::Shape::Box { c: [196.0, 50.0], half: [4.0, 48.0], angle: 0.0 },
                crate::scene::Shape::Box { c: [316.0, 50.0], half: [4.0, 48.0], angle: 0.0 },
                crate::scene::Shape::Box { c: [250.0, 150.0], half: [45.0, 3.0], angle: 0.25 },
            ],
        );
        w.sim.enable_heat();
        for _ in 0..(8 * 60) {
            w.sim.advance_blocking(1.0 / 60.0, brush);
            off.borrow_mut().track(&w.sim);
        }
        save(&w, "steam_shelf", zoom([256.0, 110.0], 2.2));
        // Lava poured into a shallow pan of water.
        let mut w = pool(&[([100.0, WALL, 412.0, WALL + 12.0], Material::Water)], &[]);
        w.sim.enable_heat();
        hold(&mut w, Source::Pour, Material::Lava, [256.0, 120.0], 6.0, 6000.0, 1.5);
        wait(&mut w, 2.5);
        save(&w, "steam_lava_pan", zoom([256.0, 40.0], 3.0));
    }
    // Tools.
    if wants("spray") {
        let mut w = pool(&[([0.0, 0.0, 512.0, 50.0], Material::Water)], &[]);
        hold(&mut w, Source::Spray, Material::Water, [180.0, 170.0], 12.0, 30000.0, 0.7);
        hold(&mut w, Source::Spray, Material::Oil, [330.0, 150.0], 12.0, 30000.0, 0.5);
        save(&w, "spray", fit);
        save(&w, "spray_zoom", zoom([300.0, 110.0], 2.5));
    }
    if wants("jet") {
        let mut w = pool(&[([0.0, 0.0, 512.0, 90.0], Material::Water), ([0.0, 90.0, 512.0, 115.0], Material::Oil)], &[]);
        w.sim.advance_blocking(0.5, ToolState::default());
        hold(&mut w, Source::Jet { angle: -70.0 }, Material::Water, [120.0, 60.0], 9.0, 20000.0, 1.2);
        save(&w, "jet_in_pool", fit);
        save(&w, "jet_in_pool_zoom", zoom([230.0, 80.0], 2.5));
        let mut w = pool(&[([0.0, 0.0, 512.0, 60.0], Material::Water)], &[]);
        hold(&mut w, Source::Jet { angle: 60.0 }, Material::Water, [380.0, 200.0], 9.0, 20000.0, 0.9);
        save(&w, "jet_impact", fit);
        save(&w, "jet_impact_zoom", zoom([200.0, 60.0], 2.5));
    }
    if wants("current") {
        let mut w = pool(&[([0.0, 0.0, 512.0, 80.0], Material::Water), ([0.0, 80.0, 512.0, 105.0], Material::Oil), ([0.0, 0.0, 512.0, WALL + 18.0], Material::Honey)], &[]);
        w.add_current(Current::new([256.0, 55.0], 16.0, 90.0, 1.0));
        w.sim.advance_blocking(3.0, ToolState::default());
        save(&w, "current_in_pool", fit);
        save(&w, "current_in_pool_zoom", zoom([300.0, 65.0], 2.5));
    }
    if wants("pour") {
        let mut w = pool(&[], &[crate::scene::Shape::Circle { c: [256.0, 80.0], r: 30.0 }]);
        hold(&mut w, Source::Pour, Material::Honey, [250.0, 200.0], 7.0, 8000.0, 2.5);
        save(&w, "pour_honey_on_ball", fit);
        save(&w, "pour_honey_on_ball_zoom", zoom([256.0, 90.0], 3.0));
        let mut w = pool(&[], &[crate::scene::Shape::Circle { c: [256.0, 80.0], r: 30.0 }]);
        hold(&mut w, Source::Pour, Material::Water, [250.0, 200.0], 7.0, 8000.0, 1.2);
        save(&w, "pour_water_on_ball", fit);
        save(&w, "pour_water_on_ball_zoom", zoom([256.0, 90.0], 3.0));
    }
    // Cooling a stirred pool: ice and the half-mixed liquid around it (per-cell fields; they
    // looked pixelated before the optics blur, README "Look").
    if wants("freeze") {
        let mut w = map("Playground", G, 3.0);
        for k in 0..240 {
            let ph = k as f32 / 240.0 * std::f32::consts::TAU;
            let at = [200.0 + 40.0 * ph.sin(), 40.0 + 12.0 * (2.0 * ph).sin()];
            w.sim.advance_blocking(1.0 / 60.0, ToolState::heat(at, 13.0, true));
            off.borrow_mut().track(&w.sim);
        }
        wait(&mut w, 1.0);
        save(&w, "freeze_pool", fit);
        save(&w, "freeze_pool_zoom", zoom([200.0, 40.0], 3.0));
        save(&w, "freeze_pool_zoom8", zoom([200.0, 40.0], 8.0));
    }
    // Liquid on solids.
    if wants("wet") {
        // Each liquid poured over a ball and left to run off: what it leaves behind.
        let ball = |x: f32| crate::scene::Shape::Circle { c: [x, 90.0], r: 24.0 };
        let mut w = pool(&[], &[ball(110.0), ball(256.0), ball(402.0)]);
        hold(&mut w, Source::Pour, Material::Honey, [398.0, 190.0], 6.0, 6000.0, 1.0);
        hold(&mut w, Source::Pour, Material::Oil, [252.0, 190.0], 6.0, 6000.0, 1.0);
        hold(&mut w, Source::Pour, Material::Water, [106.0, 190.0], 6.0, 6000.0, 1.0);
        wait(&mut w, 0.25);
        save(&w, "wet_running_off", fit);
        wait(&mut w, 1.25);
        save(&w, "wet_after", fit);
        save(&w, "wet_after_zoom", zoom([256.0, 95.0], 2.4));
    }
    if wants("plank") {
        // Blood poured over the end of the Playground plank: the wet edge of a slanted solid,
        // close up.
        let mut w = map("Playground", G, 0.0);
        hold(&mut w, Source::Pour, Material::Blood, [60.0, 230.0], 8.0, 4000.0, 1.5);
        wait(&mut w, 1.0);
        save(&w, "plank_wet", zoom([80.0, 180.0], 3.0));
        save(&w, "plank_wet_zoom8", zoom([70.0, 182.0], 8.0));
        save(&w, "plank_wet_zoom20", zoom([60.0, 186.0], 20.0));
        // The same plank deep under blood, water and oil.
        let plank = crate::scene::Shape::Box { c: [134.4, 166.4], half: [93.8, 3.2], angle: (-51.2f32).atan2(179.2) };
        let mut w = pool(
            &[([0.0, 0.0, 512.0, 150.0], Material::Blood), ([0.0, 150.0, 300.0, 270.0], Material::Water), ([300.0, 150.0, 512.0, 270.0], Material::Oil)],
            &[plank],
        );
        wait(&mut w, 2.0);
        save(&w, "plank_deep", zoom([80.0, 180.0], 3.0));
        save(&w, "plank_deep_zoom8", zoom([70.0, 182.0], 8.0));
        save(&w, "plank_deep_zoom20", zoom([60.0, 186.0], 20.0));
    }
    if wants("cascade") {
        let w = map("Cascade", G, 2.2);
        save(&w, "cascade", fit);
        save(&w, "cascade_zoom", zoom([120.0, 200.0], 3.0));
    }
    if wants("drops") {
        let w = map("Oil Drops", G, 0.5);
        save(&w, "drops", fit);
        save(&w, "drops_zoom", zoom([400.0, 165.0], 4.0));
    }
    if wants("pinball") {
        let w = map("Pinball Garden", G, 1.4);
        save(&w, "pinball", fit);
    }
    // Consecutive frames, to look for flicker (see README, "Look").
    if wants("motion") {
        for (name, t, view) in [
            ("Playground", 1.3, zoom([130.0, 45.0], 3.0)),
            ("Upside Down", 5.0, zoom([256.0, 90.0], 3.0)),
            ("Dam Break", 1.15, fit),
            ("Lava Meets Water", 6.0, zoom([230.0, 60.0], 2.5)),
            ("Ferrofluid", 3.0, zoom([256.0, 30.0], 3.0)),
            ("Kitchen", 4.0, zoom([380.0, 70.0], 2.0)),
        ] {
            let mut w = map(name, G, t);
            for k in 0..6 {
                save(&w, &format!("motion_{}_{k}", name.to_lowercase().replace(' ', "_")), view);
                wait(&mut w, 1.0 / 60.0);
            }
        }
    }
    // Slime (README, "Viscoelasticity"): the cup drop over time.
    if wants("slime") {
        let wall = |x: f32| crate::scene::Shape::Box { c: [x, 40.0], half: [4.0, 38.0], angle: 0.0 };
        let peg = crate::scene::Shape::Circle { c: [236.0, 20.0], r: 8.0 };
        let mut w = pool(&[([226.0, 40.0, 296.0, 90.0], Material::Slime)], &[wall(176.0), wall(336.0), peg]);
        let mut t = 0.0;
        for k in [0.3f32, 0.6, 0.9, 1.2, 1.6, 2.4, 3.2] {
            wait(&mut w, k - t);
            t = k;
            save(&w, &format!("slime_drop_{:02}", (k * 10.0).round() as u32), zoom([256.0, 55.0], 2.2));
        }
        // A block set down on the floor, and a rope poured from above.
        let mut w = pool(&[([206.0, WALL, 306.0, WALL + 40.0], Material::Slime)], &[wall(176.0), wall(336.0)]);
        wait(&mut w, 3.0);
        save(&w, "slime_rest", zoom([256.0, 55.0], 2.2));
        let mut w = pool(&[], &[wall(176.0), wall(336.0)]);
        hold(&mut w, Source::Pour, Material::Slime, [256.0, 120.0], 3.0, 3000.0, 2.0);
        wait(&mut w, 0.5);
        save(&w, "slime_pour", zoom([256.0, 70.0], 2.2));
    }
    // The look carried by the particles and the bubbles and foam (README, "Look", "Carried by
    // the particles", "Bubbles and foam"): stirred, then left, and side by side in the light.
    if wants("look") {
        use crate::scene::Shape;
        let wall = |x: f32| Shape::Box { c: [x, 50.0], half: [4.0, 48.0], angle: 0.0 };
        let cup = [wall(176.0), wall(336.0)];
        let stir = |w: &mut World, at: [f32; 2], seconds: f32| {
            let tool = ToolState::whirl(at, 24.0, false);
            for _ in 0..(seconds * 60.0).round() as usize {
                w.sim.advance_blocking(1.0 / 60.0, tool);
                off.borrow_mut().track(&w.sim);
            }
        };
        for (m, name) in [(Material::Soap, "look_pearl"), (Material::Snowy, "look_snow"), (Material::Polish, "look_glitter"), (Material::Honey, "look_honey")] {
            let mut w = pool(&[([180.0, WALL, 332.0, 72.0], m)], &cup);
            wait(&mut w, 0.5);
            save(&w, &format!("{name}_still"), zoom([256.0, 45.0], 3.0));
            stir(&mut w, [256.0, 40.0], 2.0);
            save(&w, &format!("{name}_stirred"), zoom([256.0, 45.0], 3.0));
            wait(&mut w, 2.0);
            save(&w, &format!("{name}_after"), zoom([256.0, 45.0], 3.0));
        }
        // Water poured into water: white water and bubbles; beer into a glass: a head.
        let mut w = pool(&[([180.0, WALL, 332.0, 40.0], Material::Water)], &cup);
        hold(&mut w, Source::Pour, Material::Water, [256.0, 140.0], 5.0, 6000.0, 1.5);
        save(&w, "look_splash", zoom([256.0, 45.0], 3.0));
        save(&w, "look_splash_zoom", zoom([256.0, 40.0], 8.0));
        wait(&mut w, 1.5);
        save(&w, "look_splash_after", zoom([256.0, 45.0], 3.0));
        let mut w = pool(&[], &cup);
        hold(&mut w, Source::Pour, Material::Beer, [256.0, 130.0], 5.0, 6000.0, 2.5);
        wait(&mut w, 3.0);
        save(&w, "look_beer", zoom([256.0, 45.0], 3.0));
        save(&w, "look_beer_zoom", zoom([256.0, 60.0], 8.0));
        // Light under the surface: milk, blood, honey, mud and water side by side.
        let mut w = pool(&[
            ([40.0, WALL, 120.0, 60.0], Material::Milk),
            ([136.0, WALL, 216.0, 60.0], Material::Blood),
            ([232.0, WALL, 312.0, 60.0], Material::Honey),
            ([328.0, WALL, 408.0, 60.0], Material::Mud),
            ([424.0, WALL, 504.0, 60.0], Material::Water),
        ], &[wall(128.0), wall(224.0), wall(320.0), wall(416.0)]);
        wait(&mut w, 1.0);
        save(&w, "look_light", zoom([256.0, 40.0], 1.6));
    }
    // Debug: how soon after a reset and a pour the bubbles, foam and specks start
    // (SPROUT_DEBUG_LOOK=1 prints the draw counts each frame).
    if only == Some("delay") {
        let mut w = map("Playground", G, 0.0);
        for round in 0..2 {
            for f in 0..90 {
                let src = if f < 60 { Some(Source::Pour) } else { None };
                if let Some(sr) = src {
                    w.emit(sr, Material::Mud, [256.0, 200.0], 5.0, 4000.0, 1.0 / 60.0);
                    w.sim.advance_blocking(1.0 / 60.0, sr.nozzle([256.0, 200.0], 5.0, 4000.0));
                } else {
                    w.sim.advance_blocking(1.0 / 60.0, ToolState::default());
                }
                off.borrow_mut().track(&w.sim);
                if f % 5 == 0 {
                    let c = w.sim.debug_look_counts();
                    println!("round {round} frame {f:3} t {:.2}: look verts {} sprite verts {} handed bubbles {} specks {}", w.sim.sim_time(), c[0], c[12], c[8], c[9]);
                }
            }
            w.end_stroke();
            w.load_map(0);
        }
        return;
    }
    // Films, stains and drips (README, "Films, stains and drips"): liquids poured over a ledge
    // run, drip off its edge and dry; sprayed at the wall they spatter; lava cooling.
    if wants("films") {
        use crate::scene::Shape;
        // A funnel: two walls on a floor with a gap in the middle, so the liquid drains out
        // and leaves its coat on the walls and the back wall as its level falls; a ledge
        // across the gap's outflow catches the stream, which runs and drips off its edges.
        let funnel = [
            Shape::Box { c: [180.0, 130.0], half: [4.0, 60.0], angle: 0.0 },
            Shape::Box { c: [332.0, 130.0], half: [4.0, 60.0], angle: 0.0 },
            Shape::Box { c: [212.0, 72.0], half: [40.0, 3.0], angle: -0.25 },
            Shape::Box { c: [300.0, 72.0], half: [40.0, 3.0], angle: 0.25 },
            Shape::Box { c: [256.0, 30.0], half: [30.0, 3.0], angle: 0.0 },
        ];
        for (m, name) in [(Material::Blood, "film_blood"), (Material::Honey, "film_honey"), (Material::Mud, "film_mud"), (Material::Water, "film_water"), (Material::Oil, "film_oil"), (Material::Brine, "film_brine"), (Material::Milk, "film_milk"), (Material::Ink, "film_ink")] {
            let mut w = pool(&[([186.0, 90.0, 326.0, 170.0], m)], &funnel);
            wait(&mut w, 1.5);
            save(&w, &format!("{name}_1"), zoom([256.0, 100.0], 2.2));
            hold(&mut w, Source::Spray, m, [256.0, 175.0], 14.0, 3000.0, 1.0);
            wait(&mut w, 5.0);
            save(&w, &format!("{name}_2"), zoom([256.0, 100.0], 2.2));
            save(&w, &format!("{name}_zoom"), zoom([300.0, 70.0], 7.0));
            wait(&mut w, 25.0);
            save(&w, &format!("{name}_3"), zoom([256.0, 100.0], 2.2));
        }
        // Lava poured out cools from white through yellow and orange to a dull red under its crust.
        let mut w = pool(&[], &[]);
        hold(&mut w, Source::Pour, Material::Lava, [256.0, 150.0], 8.0, 5000.0, 2.0);
        for k in 0..3 {
            wait(&mut w, 4.0);
            save(&w, &format!("film_lava_{k}"), zoom([256.0, 30.0], 3.0));
        }
    }
    // Particles in the liquid (README, "Particles in the liquid"): cloudiness by size, platelets
    // in the flow, specks settling and stirred up, foam coloured by its liquid.
    if wants("particles") {
        use crate::scene::Shape;
        let wall = |x: f32| Shape::Box { c: [x, 50.0], half: [4.0, 48.0], angle: 0.0 };
        let cup = [wall(176.0), wall(336.0)];
        let stir = |w: &mut World, at: [f32; 2], seconds: f32| {
            let tool = ToolState::whirl(at, 24.0, false);
            for _ in 0..(seconds * 60.0).round() as usize {
                w.sim.advance_blocking(1.0 / 60.0, tool);
                off.borrow_mut().track(&w.sim);
            }
        };
        // Milk, blood, mud and ink each poured into water and stirred: dilute clouds.
        for (m, name) in [(Material::Milk, "par_milk_in_water"), (Material::Blood, "par_blood_in_water"), (Material::Mud, "par_mud_in_water"), (Material::Ink, "par_ink_in_water")] {
            let mut w = pool(&[([180.0, WALL, 332.0, 60.0], Material::Water)], &cup);
            hold(&mut w, Source::Pour, m, [256.0, 110.0], 3.0, 3000.0, 0.6);
            stir(&mut w, [256.0, 30.0], 1.5);
            wait(&mut w, 1.0);
            save(&w, name, zoom([256.0, 40.0], 3.0));
            if m == Material::Mud {
                wait(&mut w, 8.0);
                save(&w, "par_mud_settled", zoom([256.0, 40.0], 3.0));
            }
        }
        // Platelets, glitter, snow: stirred, then left.
        for (m, name) in [(Material::Pearly, "par_pearly"), (Material::Glitter, "par_glitter"), (Material::Snowy, "par_snow"), (Material::Polish, "par_polish")] {
            let mut w = pool(&[([180.0, WALL, 332.0, 72.0], m)], &cup);
            wait(&mut w, 1.0);
            save(&w, &format!("{name}_still"), zoom([256.0, 45.0], 3.0));
            stir(&mut w, [256.0, 40.0], 2.0);
            save(&w, &format!("{name}_stirred"), zoom([256.0, 45.0], 3.0));
            save(&w, &format!("{name}_zoom"), zoom([256.0, 40.0], 8.0));
            wait(&mut w, 6.0);
            save(&w, &format!("{name}_after"), zoom([256.0, 45.0], 3.0));
        }
        // Foam of beer, blood, milk and soap: poured in, then aged.
        for (m, name) in [(Material::Beer, "par_foam_beer"), (Material::Blood, "par_foam_blood"), (Material::Milk, "par_foam_milk"), (Material::Soap, "par_foam_soap")] {
            let mut w = pool(&[([180.0, WALL, 332.0, 30.0], m)], &cup);
            hold(&mut w, Source::Pour, m, [256.0, 130.0], 5.0, 6000.0, 2.0);
            wait(&mut w, 0.5);
            save(&w, name, zoom([256.0, 50.0], 4.0));
            save(&w, &format!("{name}_zoom"), zoom([256.0, 45.0], 10.0));
            wait(&mut w, 6.0);
            save(&w, &format!("{name}_aged"), zoom([256.0, 45.0], 10.0));
        }
    }
    // The properties of 2026-11-16 (README, "New properties"), each in a scene that shows it.
    if wants("new") {
        use crate::scene::Shape;
        let wall = |x: f32| Shape::Box { c: [x, 50.0], half: [4.0, 48.0], angle: 0.0 };
        let cup = [wall(196.0), wall(316.0)];
        let ball = [Shape::Circle { c: [256.0, 60.0], r: 26.0 }];
        // Beer poured into a cup: streams of bubbles, and a head that stays.
        let mut w = pool(&[], &cup);
        hold(&mut w, Source::Pour, Material::Beer, [256.0, 130.0], 5.0, 6000.0, 2.5);
        wait(&mut w, 4.0);
        save(&w, "new_beer", zoom([256.0, 50.0], 4.0));
        save(&w, "new_beer_zoom8", zoom([256.0, 28.0], 8.0));
        // Liquid soap poured into water: pearl, and lather where it splashes.
        let mut w = pool(&[([200.0, WALL, 312.0, 30.0], Material::Water)], &cup);
        hold(&mut w, Source::Pour, Material::Soap, [256.0, 130.0], 4.0, 4000.0, 2.0);
        wait(&mut w, 1.5);
        save(&w, "new_soap", zoom([256.0, 50.0], 4.0));
        // Glitter polish, the snow globe and the rheoscopic liquid over a ball.
        for (m, name) in [(Material::Polish, "new_glitter"), (Material::Snowy, "new_snow"), (Material::Pearly, "new_pearl")] {
            let mut w = pool(&[], &ball);
            hold(&mut w, Source::Pour, m, [250.0, 170.0], 6.0, 6000.0, 2.5);
            save(&w, name, zoom([256.0, 60.0], 3.5));
        }
        // Glow-stick fluid poured into water.
        let mut w = pool(&[([0.0, 0.0, 512.0, 50.0], Material::Water)], &[]);
        hold(&mut w, Source::Pour, Material::GlowFluid, [256.0, 150.0], 5.0, 5000.0, 1.5);
        wait(&mut w, 1.0);
        save(&w, "new_glow", zoom([256.0, 50.0], 2.5));
        // Lye poured into battery acid: salt water, and heat.
        let mut w = pool(&[([200.0, WALL, 312.0, 40.0], Material::Acid)], &cup);
        hold(&mut w, Source::Pour, Material::Lye, [256.0, 130.0], 4.0, 4000.0, 2.0);
        wait(&mut w, 2.0);
        save(&w, "new_reaction", zoom([256.0, 50.0], 4.0));
        // Liquid nitrogen poured onto water.
        let mut w = pool(&[([150.0, WALL, 362.0, 40.0], Material::Water)], &[wall(146.0), wall(366.0)]);
        hold(&mut w, Source::Pour, Material::Nitrogen, [256.0, 120.0], 5.0, 5000.0, 2.0);
        wait(&mut w, 1.0);
        save(&w, "new_nitrogen", zoom([256.0, 60.0], 3.0));
        // A granular liquid (Grainy) poured into a pile, and into a pool.
        let mut w = pool(&[([330.0, WALL, 508.0, 50.0], Material::Water)], &[wall(326.0)]);
        hold(&mut w, Source::Pour, Material::Grainy, [200.0, 120.0], 5.0, 6000.0, 3.0);
        wait(&mut w, 2.0);
        hold(&mut w, Source::Pour, Material::Grainy, [420.0, 150.0], 4.0, 4000.0, 1.5);
        wait(&mut w, 2.0);
        save(&w, "new_grainy", zoom([320.0, 60.0], 2.2));
        // The two parts of the resin.
        let mut w = pool(&[([200.0, WALL, 312.0, 30.0], Material::PartA)], &cup);
        hold(&mut w, Source::Pour, Material::PartB, [256.0, 120.0], 4.0, 4000.0, 1.5);
        wait(&mut w, 3.0);
        save(&w, "new_resin", zoom([256.0, 50.0], 4.0));
        // A volatile liquid, three seconds on.
        let mut w = pool(&[([200.0, WALL, 312.0, 40.0], Material::Vanishing)], &cup);
        wait(&mut w, 3.0);
        save(&w, "new_vanishing", zoom([256.0, 60.0], 3.0));
    }
    // Every liquid on its own: a block dropped into a cup with a peg, 1.6 s later.
    for (row, b) in crate::sim::BUILTINS.iter().enumerate() {
        let name = format!("liquid_{}", b.key);
        if !crate::sim::pickable(row) || !wants(&name) {
            continue;
        }
        let m = Material(row as u8);
        // A cup with a peg in it.
        let wall = |x: f32| crate::scene::Shape::Box { c: [x, 40.0], half: [4.0, 38.0], angle: 0.0 };
        let peg = crate::scene::Shape::Circle { c: [236.0, 20.0], r: 8.0 };
        let mut w = pool(&[([226.0, 40.0, 296.0, 90.0], m)], &[wall(176.0), wall(336.0), peg]);
        wait(&mut w, 1.6);
        save(&w, &name, zoom([256.0, 55.0], 2.2));
    }
    // The new maps, and the interactions they are about.
    for (name, file, t, view) in [
        ("Lava Meets Water", "lava_water", 8.0, zoom([200.0, 60.0], 2.0)),
        ("Density Tower", "density_tower", 12.0, zoom([256.0, 60.0], 2.2)),
        ("Glass and Slime", "glass_slime", 4.0, fit),
        ("Mud Flats", "mud_flats", 6.0, fit),
        ("Ferrofluid", "ferrofluid_spikes", 4.0, zoom([256.0, 30.0], 3.0)),
        ("Kitchen", "kitchen", 8.0, fit),
        ("Cold Snap", "cold_snap", 30.0, fit),
        ("Gallium", "gallium", 3.0, fit),
    ] {
        if !wants(file) {
            continue;
        }
        let mut w = map(name, G, 0.0);
        // In frames, so that the wet map and the drying films follow.
        wait(&mut w, t as f32);
        save(&w, file, fit);
        save(&w, &format!("{file}_zoom"), view);
        if file == "ferrofluid_spikes" {
            save(&w, "ferrofluid_spikes_close", zoom([256.0, 22.0], 10.0));
        }
    }

    // Zoom and resolution: far out, very close, and the coarsest and finest presets.
    if wants("zoom") {
        let w = map("Playground", G, 3.0);
        save(&w, "zoom_out", zoom([256.0, 144.0], 0.6));
        save(&w, "zoom_in_8x", zoom([330.0, 68.0], 8.0));
        save(&w, "zoom_in_20x", zoom([330.0, 70.0], 20.0));
    }
    if wants("quality") {
        for (label, grid) in [("low", [384, 216]), ("ultra", [1024, 576])] {
            let w = map("Playground", grid, 1.3);
            let fit = View::fit(grid, SIZE);
            save(&w, &format!("quality_{label}"), fit);
            let s = grid[0] as f32 / 512.0;
            save(&w, &format!("quality_{label}_zoom"), View { center: [130.0 * s, 45.0 * s], zoom: fit.zoom * 3.0 });
        }
    }
}
