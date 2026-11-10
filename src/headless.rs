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
    let hz = world.sim.params.substep_hz;
    let start = Instant::now();
    for (k, &t) in o.times.iter().enumerate() {
        let target = (t * hz).round() as u64;
        while world.sim.substeps_done() < target {
            let n = (target - world.sim.substeps_done()).min(8) as u32;
            world.sim.step_blocking(n, ToolState::default());
        }
        if world.sim.substeps_done() == 0 {
            world.sim.step_blocking(0, ToolState::default());
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
        pe += (params.gravity * rho * p.pos[1]) as f64;
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
    let hz = world.sim.params.substep_hz;
    let steps = (seconds * hz) as u32;
    println!("probe: {} particles, {steps} substeps", world.sim.slots());
    println!("{:>7} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}", "t", "ke", "pe", "ke+pe", "max_speed", "mean_y", "top_y");
    let every = (steps / 20).max(1);
    for s in 0..=steps {
        if s % every == 0 {
            let d = diagnose(&world.sim.download_particles(), &world.sim.params);
            println!("{:7.3} {:10.3} {:10.1} {:10.1} {:10.2} {:9.3} {:9.3}", s as f32 / hz, d.ke, d.pe, d.ke + d.pe, d.max_speed, d.mean_y, d.top_y);
        }
        world.sim.step_blocking(1, ToolState::default());
    }
    if let Some(path) = out {
        let size = [1280, 720];
        Offscreen::new(gpu, size).save_png(&world.sim, View::fit(grid, size), Path::new(path));
    }
}

// ---- benchmark -----------------------------------------------------------------------

fn time_substeps(world: &mut World, batches: u32, per_batch: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..batches {
        world.sim.step_blocking(per_batch, ToolState::default());
    }
    start.elapsed().as_secs_f64() * 1e3 / (batches * per_batch) as f64
}

/// Measures simulation and rendering throughput on a few workloads and prints a table.
///
/// Every number is wall-clock time for work submitted and waited on, so it includes
/// command recording on the CPU as well as GPU execution. (The app records the next frame
/// while the GPU is still busy with the last one, so it does slightly better.)
pub fn bench(gpu: Arc<Gpu>, apply: &dyn Fn(&mut SimParams), grids: &[[u32; 2]], maps: &[usize], size: [u32; 2], warmup: f32) {
    println!("device: {}", gpu.device_name());
    println!("render target: {}x{} (off-screen), each map measured after {warmup} s of simulated time\n", size[0], size[1]);
    println!(
        "{:<14} {:>9} {:>10} | {:>9} {:>9} {:>9} | {:>8} | {:>9} {:>9}",
        "map", "grid", "particles", "substep", "particles", "solve", "render", "real-time", "fps left"
    );
    println!("{:<14} {:>9} {:>10} | {:>9} {:>9} {:>9} | {:>8} | {:>9} {:>9}", "", "", "", "ms", "+ grid ms", "ms", "ms", "load", "");
    for &grid in grids {
        let mut off = Offscreen::new(gpu.clone(), size);
        let view = View::fit(grid, size);
        for &map in maps {
            let mut world = World::new(gpu.clone(), grid, map);
            apply(&mut world.sim.params);
            let params = world.sim.params;
            let hz = params.substep_hz;
            let per_frame = (hz / 60.0).round().max(1.0) as u32;

            // Let the scene get going: a moving, mixed fluid is the workload, not the
            // tidy lattice a map starts as.
            time_substeps(&mut world, ((hz * warmup) as u32 / per_frame).max(1), per_frame);

            let full = time_substeps(&mut world, 60, per_frame);
            // Running every sweep twice costs exactly one more set of sweeps, which prices
            // the iterative solves without disturbing the flow being measured.
            world.sim.params.pressure_iters *= 2;
            world.sim.params.viscosity_iters *= 2;
            let solve = (time_substeps(&mut world, 30, per_frame) - full).max(0.0);
            world.sim.params = params;
            time_substeps(&mut world, 10, per_frame);

            let start = Instant::now();
            let n = 60;
            for _ in 0..n {
                off.draw(&world.sim, view);
            }
            let render = start.elapsed().as_secs_f64() * 1e3 / n as f64;

            // Share of every second that simulating in real time takes, and the frame
            // rate the rest of the second can render at.
            let load = full * hz as f64 / 1e3;
            let fps = if load < 1.0 { format!("{:.0}", (1.0 - load) * 1e3 / render) } else { format!("{:.2}x speed", 1.0 / load) };
            println!(
                "{:<14} {:>9} {:>10} | {:>9.3} {:>9.3} {:>9.3} | {:>8.3} | {:>8.0}% {:>9}",
                crate::maps::MAPS[map].name,
                format!("{}x{}", grid[0], grid[1]),
                world.sim.slots(),
                full,
                full - solve,
                solve,
                render,
                load * 100.0,
                fps
            );
        }
    }
    println!(
        "\nsubstep: one full simulation step.  particles + grid: the per-particle pass and the fixed grid passes.\n\
         solve: the pressure and viscosity sweeps.  real-time load: share of each second needed to run the\n\
         substeps of one simulated second.  fps left: frame rate the remaining time renders at; where the load\n\
         is over 100 % the app slows simulated time down instead, to the speed shown."
    );
}

/// Splits the cost of a batch of substeps into CPU recording and GPU execution.
pub fn profile(gpu: Arc<Gpu>, apply: &dyn Fn(&mut SimParams), grid: [u32; 2], map: usize, warmup: f32) {
    let mut world = World::new(gpu.clone(), grid, map);
    apply(&mut world.sim.params);
    let per_frame = 4;
    let batches = (warmup * world.sim.params.substep_hz) as u32 / per_frame;
    time_substeps(&mut world, batches.max(1), per_frame);
    let (mut rec, mut exec) = (0.0f64, 0.0f64);
    let n = 100;
    for _ in 0..n {
        let t0 = Instant::now();
        let mut b = gpu.builder();
        world.sim.record(&mut b, per_frame, ToolState::default());
        let cb = b.build().unwrap();
        let t1 = Instant::now();
        use vulkano::{command_buffer::PrimaryCommandBufferAbstract, sync::GpuFuture};
        cb.execute(gpu.queue.clone()).unwrap().then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        let t2 = Instant::now();
        rec += (t1 - t0).as_secs_f64();
        exec += (t2 - t1).as_secs_f64();
    }
    let k = 1e3 / (n * per_frame) as f64;
    println!(
        "{} {}x{} {} particles: record {:.3} ms/substep, submit+execute {:.3} ms/substep",
        crate::maps::MAPS[map].name, grid[0], grid[1], world.sim.slots(), rec * k, exec * k
    );
}
