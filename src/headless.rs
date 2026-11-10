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
    render::{Cursor, Renderer, View},
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
        self.renderer.render(&mut b, self.view.clone(), sim, view, Cursor::default(), |_| {});
        submit_and_wait(&self.gpu, b);
    }

    /// Renders the simulation and returns RGBA8 pixels.
    pub fn capture(&mut self, sim: &FluidSim, view: View) -> Vec<u8> {
        self.capture_with(sim, view, Cursor::default(), |_| {}, |_| {})
    }

    /// Like `capture`, with hooks that record before the render passes and inside the
    /// final one (used to composite the UI).
    pub fn capture_with(
        &mut self,
        sim: &FluidSim,
        view: View,
        cursor: Cursor,
        before: impl FnOnce(&mut Builder),
        overlay: impl FnOnce(&mut Builder),
    ) -> Vec<u8> {
        let mut b = self.gpu.builder();
        before(&mut b);
        self.renderer.render(&mut b, self.view.clone(), sim, view, cursor, overlay);
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
        let left = t as f64 - world.sim.sim_time();
        if left > 0.0 {
            world.sim.advance_blocking(left, ToolState::default());
        }
        let path = if o.times.len() == 1 { o.out.clone() } else { o.out.replace(".png", &format!("_{k}.png")) };
        if o.with_ui {
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
    let mut deltas = Vec::new();
    let mut shapes = Vec::new();
    for frame in 0..30 {
        let input = egui::RawInput { time: Some(frame as f64 / 30.0), ..input.clone() };
        let out = ctx.run(input, |ctx| {
            ui::build(ctx, &mut state, world);
        });
        deltas.push(out.textures_delta);
        shapes = out.shapes;
    }
    let primitives = ctx.tessellate(shapes, 1.0);
    let cursor = Cursor { pos: [world.sim.grid[0] as f32 * 0.62, world.sim.grid[1] as f32 * 0.7], radius: state.brush, alpha: 0.55 };
    // The two hooks need the painter one after the other, never at once.
    let painter = RefCell::new(painter);
    let pixels = off.capture_with(
        &world.sim,
        view,
        cursor,
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
    sim::{KIND_OIL, Material, Particle, SimParams},
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
        let rho = if p.kind == KIND_OIL { params.rho[2] } else { params.rho[0] + (params.rho[1] - params.rho[0]) * p.s };
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
        "{:>7} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>8} {:>6} {:>6}",
        "t", "ke", "pe", "ke+pe", "max_speed", "mean_y", "top_y", "dt ms", "p it", "v it"
    );
    for s in 0..=samples {
        let d = diagnose(&world.sim.download_particles(), &world.sim.params);
        let st = world.sim.solver;
        println!(
            "{:7.3} {:10.3} {:10.1} {:10.1} {:10.2} {:9.3} {:9.3} {:8.3} {:6} {:6}",
            world.sim.sim_time(), d.ke, d.pe, d.ke + d.pe, d.max_speed, d.mean_y, d.top_y, st.dt * 1e3,
            st.pressure_iters_max, st.viscosity_iters_max
        );
        world.sim.solver.reset_extremes();
        if s < samples {
            world.sim.advance_blocking(seconds as f64 / samples as f64, ToolState::default());
        }
    }
    let st = world.sim.solver;
    println!(
        "substeps {}  unconverged pressure {} viscosity {}",
        st.substeps, st.pressure_unconverged, st.viscosity_unconverged
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
    let passes = world.sim.profile_passes();
    let line: Vec<String> = passes.iter().map(|(n, ms)| format!("{n} {ms:.3}")).collect();
    println!("one substep, GPU ms by pass: {}", line.join(", "));
    println!(
        "{} {}x{} {} particles: {:.1} substeps per frame; per frame: record {:.3} ms, submit+execute {:.3} ms; per substep: record {:.3} ms, execute {:.3} ms",
        crate::maps::MAPS[map].name, grid[0], grid[1], world.sim.slots(), substeps / n as f64,
        rec * 1e3 / n as f64, exec * 1e3 / n as f64, rec * 1e3 / substeps, exec * 1e3 / substeps
    );
}
