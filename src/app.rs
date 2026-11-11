//! The interactive sandbox: window, input, tools and frame loop.

use std::{sync::Arc, time::Instant};

use vulkano::{
    Validated, VulkanError,
    format::Format,
    image::{ImageUsage, view::ImageView},
    swapchain::{
        PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture, future::FenceSignalFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

use crate::{
    brush,
    camera::Camera,
    gpu::Gpu,
    gui::GuiPainter,
    maps::MAPS,
    pacing::{Pacer, Plan},
    render::Renderer,
    scene::Shape,
    selftest::{self, Act, Snapshot},
    sim::{KIND_OIL, Material, ToolState},
    ui::{self, Overlay, Tool, UiActions, UiState},
    world::{Source, World},
};

pub struct AppOptions {
    pub map: usize,
    pub grid: [u32; 2],
    pub size: [u32; 2],
    pub vsync: bool,
    pub overrides: Vec<(String, f32)>,
    /// Exit after this many frames and print frame timings (for benchmarking).
    pub frames: Option<u64>,
    /// Run the scripted session of `selftest` instead of waiting for input.
    pub selftest: bool,
}

type FrameFuture = FenceSignalFuture<Box<dyn GpuFuture>>;

/// Everything tied to the OS window.
struct Win {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    views: Vec<Arc<ImageView>>,
    renderer: Renderer,
    painter: GuiPainter,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    present_modes: Vec<PresentMode>,
    recreate: bool,
    /// The frame the GPU is currently working on, if any.
    in_flight: Option<FrameFuture>,
}

#[derive(Default)]
struct Input {
    cursor_px: [f32; 2],
    cursor_world: [f32; 2],
    prev_world: [f32; 2],
    /// Smoothed cursor velocity in cells/s.
    velocity: [f32; 2],
    left: bool,
    right: bool,
    middle: bool,
    modifiers: ModifiersState,
    inside: bool,
}

struct App {
    gpu: Arc<Gpu>,
    opts: AppOptions,
    win: Option<Win>,
    world: World,
    ui: UiState,
    input: Input,
    camera: Camera,
    overlay: Overlay,
    /// Tool and liquid of the last frame, to announce changes.
    shown: (Tool, Material),
    last_frame: Instant,
    pacer: Pacer,
    step_once: bool,
    frame_index: u64,
    /// Frame times of the whole run, kept only for `--frames`.
    bench: Vec<f32>,
    selftest: Option<SelfTest>,
}

/// Progress through the `--selftest` script.
struct SelfTest {
    script: Vec<(u64, Act)>,
    next: usize,
    mark: Snapshot,
    failures: u32,
}

/// Runs the app until its window closes. Returns false if a `--selftest` check failed.
pub fn run(event_loop: EventLoop<()>, gpu: Arc<Gpu>, opts: AppOptions) -> bool {
    let mut world = World::new(gpu.clone(), opts.grid, opts.map);
    for (k, v) in &opts.overrides {
        world.sim.params.set(k, *v);
        world.scale_locked |= k == "cell_size";
    }
    let ui = UiState { vsync: opts.vsync, grid: opts.grid, ..Default::default() };
    let mut app = App {
        camera: Camera::new(opts.grid, opts.size),
        overlay: Overlay::default(),
        shown: (ui.tool, ui.material),
        gpu,
        opts,
        win: None,
        world,
        ui,
        input: Input::default(),
        last_frame: Instant::now(),
        pacer: Pacer::new(),
        step_once: false,
        frame_index: 0,
        bench: Vec::new(),
        selftest: None,
    };
    if app.opts.selftest {
        app.selftest = Some(SelfTest { script: selftest::script(), next: 0, mark: Snapshot::default(), failures: 0 });
    }
    event_loop.run_app(&mut app).expect("event loop failed");
    app.selftest.is_none_or(|t| t.failures == 0)
}

fn pick_present_mode(available: &[PresentMode], vsync: bool) -> PresentMode {
    let prefer: &[PresentMode] =
        if vsync { &[PresentMode::Fifo] } else { &[PresentMode::Mailbox, PresentMode::Immediate, PresentMode::Fifo] };
    prefer.iter().copied().find(|m| available.contains(m)).unwrap_or(PresentMode::Fifo)
}

/// The tool a key selects (`Tool::key` names it in the UI).
fn tool_for_key(code: KeyCode) -> Option<Tool> {
    Some(match code {
        KeyCode::KeyQ => Tool::Pour,
        KeyCode::KeyB => Tool::Spray,
        KeyCode::KeyJ => Tool::Jet,
        KeyCode::KeyC => Tool::Drop,
        KeyCode::KeyW => Tool::Grab,
        KeyCode::KeyV => Tool::Push,
        KeyCode::KeyE => Tool::Erase,
        KeyCode::KeyA => Tool::Circle,
        KeyCode::KeyS => Tool::Box,
        KeyCode::KeyD => Tool::Ramp,
        KeyCode::KeyX => Tool::Remove,
        _ => return None,
    })
}

/// The solid a shape tool would place at `c` right now.
fn pending_shape(ui: &UiState, c: [f32; 2]) -> Option<Shape> {
    let r = ui.brush;
    let k = ui.shape_aspect.sqrt();
    let angle = ui.shape_angle.to_radians();
    match ui.tool {
        Tool::Circle => Some(Shape::Circle { c, r }),
        Tool::Box => Some(Shape::Box { c, half: [r * k, r / k], angle }),
        Tool::Ramp => Some(Shape::Ramp { c, half: [r * k, r / k], angle, flip: ui.shape_flip }),
        _ => None,
    }
}

impl Win {
    fn new(event_loop: &ActiveEventLoop, gpu: &Arc<Gpu>, opts: &AppOptions) -> Win {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Sprout")
                        .with_inner_size(PhysicalSize::new(opts.size[0], opts.size[1])),
                )
                .expect("failed to create a window"),
        );
        let surface = Surface::from_window(gpu.instance.clone(), window.clone()).expect("failed to create a surface");
        let physical = gpu.device.physical_device();
        let caps = physical.surface_capabilities(&surface, Default::default()).unwrap();
        let formats = physical.surface_formats(&surface, Default::default()).unwrap();
        // The renderer writes display-referred colours, so it wants a UNORM target.
        let format = [Format::B8G8R8A8_UNORM, Format::R8G8B8A8_UNORM, Format::A8B8G8R8_UNORM_PACK32]
            .into_iter()
            .find(|f| formats.iter().any(|(g, _)| g == f))
            .unwrap_or(formats[0].0);
        let present_modes: Vec<PresentMode> =
            physical.surface_present_modes(&surface, SurfaceInfo::default()).unwrap().into_iter().collect();

        let size = window.inner_size();
        let mut min_image_count = caps.min_image_count.max(3);
        if let Some(max) = caps.max_image_count {
            min_image_count = min_image_count.min(max);
        }
        let (swapchain, images) = Swapchain::new(
            gpu.device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count,
                image_format: format,
                image_extent: [size.width.max(1), size.height.max(1)],
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: caps.supported_composite_alpha.into_iter().next().unwrap(),
                present_mode: pick_present_mode(&present_modes, opts.vsync),
                ..Default::default()
            },
        )
        .expect("failed to create a swapchain");
        let views = images.into_iter().map(|i| ImageView::new_default(i).unwrap()).collect();

        let renderer = Renderer::new(gpu.clone(), format);
        let painter = GuiPainter::new(gpu.clone(), renderer.final_subpass());
        let egui_ctx = egui::Context::default();
        ui::style(&egui_ctx);
        let max_texture_side = physical.properties().max_image_dimension2_d as usize;
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            event_loop,
            Some(window.scale_factor() as f32),
            None,
            Some(max_texture_side),
        );

        Win {
            window,
            swapchain,
            views,
            renderer,
            painter,
            egui_ctx,
            egui_state,
            present_modes,
            recreate: false,
            in_flight: None,
        }
    }

    /// GPU time a frame may spend on the simulation, s: four fifths of the refresh interval
    /// when presenting with vsync (the rest is for rendering), of 1/60 s otherwise.
    fn sim_budget(&self, vsync: bool) -> f32 {
        let refresh = self.window.current_monitor().and_then(|m| m.refresh_rate_millihertz());
        let interval = match refresh {
            Some(mhz) if vsync => 1000.0 / mhz.max(1) as f32,
            _ => 1.0 / 60.0,
        };
        0.8 * interval
    }

    fn size(&self) -> [u32; 2] {
        let s = self.window.inner_size();
        [s.width, s.height]
    }
}

impl App {
    fn reset_view(&mut self, animate: bool) {
        if let Some(win) = &self.win {
            self.camera.fit(self.world.sim.grid, win.size(), animate);
        }
    }

    fn egui(&self) -> Option<egui::Context> {
        self.win.as_ref().map(|w| w.egui_ctx.clone())
    }

    fn load_map(&mut self, map: usize) {
        let changed = map != self.world.map;
        self.world.load_map(map);
        self.pacer.reset();
        if let Some(ctx) = self.egui() {
            if changed {
                self.overlay.toast_map(&ctx, self.world.map);
            } else {
                self.overlay.toast(&ctx, "Reset", MAPS[map].name, egui::Color32::from_gray(235));
            }
        }
    }

    fn change_grid(&mut self, grid: [u32; 2]) {
        let params = self.world.sim.params;
        let map = self.world.map;
        // Drop the old world first so two full sets of buffers never coexist.
        if let Some(win) = &mut self.win {
            if let Some(f) = win.in_flight.take() {
                let _ = f.wait(None);
            }
            win.renderer.invalidate();
        }
        self.ui.brush *= grid[0] as f32 / self.world.sim.grid[0] as f32;
        let locked = self.world.scale_locked;
        self.world = World::new(self.gpu.clone(), grid, map);
        self.world.sim.params = params;
        self.world.scale_locked = locked;
        self.world.apply_map_scale();
        self.ui.grid = grid;
        self.reset_view(false);
        if let Some(ctx) = self.egui() {
            let name = ui::GRIDS.iter().find(|g| g.1 == grid).map_or("Custom", |g| g.0);
            self.overlay.toast(&ctx, "Quality", name, egui::Color32::from_gray(235));
        }
    }

    fn apply(&mut self, actions: UiActions, event_loop: Option<&ActiveEventLoop>) {
        if actions.reset {
            self.load_map(self.world.map);
        }
        if let Some(map) = actions.load_map {
            self.load_map(map);
        }
        if actions.step {
            self.step_once = true;
            self.ui.paused = true;
        }
        if let Some(grid) = actions.grid {
            self.change_grid(grid);
        }
        if actions.reset_view {
            self.reset_view(true);
        }
        if actions.vsync_changed && let Some(win) = &mut self.win {
            win.recreate = true;
        }
        if actions.quit && let Some(el) = event_loop {
            el.exit();
        }
    }

    fn key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        let mut a = UiActions::default();
        let n_maps = MAPS.len();
        if let Some(tool) = tool_for_key(code) {
            self.ui.tool = tool;
        }
        match code {
            KeyCode::Space => self.ui.paused = !self.ui.paused,
            KeyCode::KeyN | KeyCode::Period => a.step = true,
            KeyCode::KeyR => a.reset = true,
            KeyCode::Digit1 => self.ui.pick_material(Material::Water),
            KeyCode::Digit2 => self.ui.pick_material(Material::Oil),
            KeyCode::Digit3 => self.ui.pick_material(Material::Syrup),
            KeyCode::BracketLeft => self.ui.scale_brush(1.0 / 1.15),
            KeyCode::BracketRight => self.ui.scale_brush(1.15),
            KeyCode::Minus => self.ui.scale_rate(1.0 / 1.25),
            KeyCode::Equal => self.ui.scale_rate(1.25),
            KeyCode::KeyT => self.ui.rotate(15.0),
            KeyCode::KeyG => self.ui.rotate(-15.0),
            KeyCode::KeyF => self.ui.shape_flip = !self.ui.shape_flip,
            KeyCode::KeyH => self.ui.show_panel = !self.ui.show_panel,
            KeyCode::Home | KeyCode::Digit0 => a.reset_view = true,
            KeyCode::PageDown => a.load_map = Some((self.world.map + 1) % n_maps),
            KeyCode::PageUp => a.load_map = Some((self.world.map + n_maps - 1) % n_maps),
            KeyCode::F11 => {
                if let Some(win) = &self.win {
                    let fs = win.window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                    win.window.set_fullscreen(fs);
                }
            }
            KeyCode::Escape => a.quit = true,
            _ => {}
        }
        self.apply(a, Some(event_loop));
    }

    fn pointer_moved(&mut self, px: [f32; 2]) {
        if self.input.middle {
            self.camera.pan([px[0] - self.input.cursor_px[0], px[1] - self.input.cursor_px[1]]);
        }
        self.input.cursor_px = px;
        if let Some(win) = &self.win {
            self.input.cursor_world = self.camera.view.screen_to_world(px, win.size());
        }
        self.input.inside = true;
    }

    fn button(&mut self, button: MouseButton, down: bool) {
        match button {
            MouseButton::Left => {
                self.input.left = down;
                if down {
                    self.click();
                }
            }
            MouseButton::Right => self.input.right = down,
            MouseButton::Middle => self.input.middle = down,
            _ => {}
        }
    }

    fn wheel(&mut self, lines: f32) {
        if self.input.modifiers.shift_key() || self.input.modifiers.control_key() {
            self.ui.scale_brush(1.12f32.powf(lines));
        } else if self.ui.tool == Tool::Jet && !self.input.modifiers.alt_key() {
            self.ui.rotate_jet(lines * 15.0);
        } else {
            self.zoom_at_cursor(1.15f32.powf(lines));
        }
    }

    /// What the self-test can observe. Reads the particles back, so it waits for the GPU.
    fn snapshot(&mut self) -> Snapshot {
        if let Some(f) = self.win.as_mut().and_then(|w| w.in_flight.take()) {
            let _ = f.wait(None);
        }
        let ps = self.world.sim.download_particles();
        let speed2: f32 = ps.iter().map(|p| p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sum();
        let direction = brush::jet_direction(self.ui.jet_angle);
        let mut axial = 0.0;
        let mut outlet_count = 0;
        let mut radial = 0.0;
        for p in &ps {
            let d = [p.pos[0] - self.input.cursor_world[0], p.pos[1] - self.input.cursor_world[1]];
            let distance = d[0].hypot(d[1]);
            if distance < self.ui.brush * 0.7 {
                axial += p.vel[0] * direction[0] + p.vel[1] * direction[1];
                outlet_count += 1;
            }
            if distance > 1e-4 { radial += (p.vel[0] * d[0] + p.vel[1] * d[1]) / distance; }
        }
        Snapshot {
            jet_speed: axial / outlet_count.max(1) as f32,
            radial_speed: radial / ps.len().max(1) as f32,
            particles: ps.len(),
            oil: ps.iter().filter(|p| p.kind == KIND_OIL).count(),
            syrup: ps.iter().filter(|p| p.kind != KIND_OIL).map(|p| p.s).sum(),
            rms_speed: (speed2 / ps.len().max(1) as f32).sqrt(),
            shapes: self.world.scene.shapes.len(),
            substeps: self.world.sim.substeps_done(),
            grid: self.world.sim.grid,
            map: self.world.map,
            zoom: self.camera.target().zoom,
            center: self.camera.target().center,
            brush: self.ui.brush,
            jet_angle: self.ui.jet_angle,
            panel: self.ui.show_panel,
        }
    }

    /// Feeds this frame's share of the self-test script to the input handlers.
    fn run_selftest(&mut self, event_loop: &ActiveEventLoop) {
        let Some(mut test) = self.selftest.take() else { return };
        while let Some((frame, act)) = test.script.get(test.next) {
            if *frame > self.frame_index {
                break;
            }
            test.next += 1;
            match *act {
                Act::Key(code) => self.key(code, event_loop),
                Act::Move(cells) => {
                    let size = self.win.as_ref().map_or(self.opts.size, |w| w.size());
                    self.pointer_moved(self.camera.view.world_to_screen(cells, size));
                }
                Act::Press(button) => self.button(button, true),
                Act::Release(button) => self.button(button, false),
                Act::Wheel(lines) => self.wheel(lines),
                Act::Grid(grid) => self.apply(UiActions { grid: Some(grid), ..Default::default() }, Some(event_loop)),
                Act::Mark => test.mark = self.snapshot(),
                Act::Check(what, check) => {
                    let now = self.snapshot();
                    let ok = check(&test.mark, &now);
                    test.failures += !ok as u32;
                    println!(
                        "  [{}] {:<38} {} particles ({} oil, {:.0} syrup), {} solids, rms {:.0} cells/s",
                        if ok { "pass" } else { "FAIL" }, what, now.particles, now.oil, now.syrup.abs(), now.shapes, now.rms_speed.abs()
                    );
                }
            }
        }
        if test.next == test.script.len() {
            println!("\n{}", if test.failures == 0 { "all checks passed".to_string() } else { format!("{} check(s) FAILED", test.failures) });
            event_loop.exit();
        }
        self.selftest = Some(test);
    }

    /// A press of the left button: the tools that act once per click.
    fn click(&mut self) {
        let at = self.input.cursor_world;
        let (tool, material, radius) = (self.ui.tool, self.ui.material, self.ui.brush);
        let done = match tool {
            Tool::Drop => self.world.stamp(material, at, radius) > 0,
            Tool::Remove => self.world.remove_shape_at(at),
            _ => match pending_shape(&self.ui, at) {
                Some(shape) => {
                    self.world.add_shape(shape);
                    true
                }
                None => false,
            },
        };
        if done && let Some(ctx) = self.egui() {
            self.overlay.ripple_for(&ctx, at, if tool == Tool::Remove { 2.0 } else { radius }, tool, material);
        }
    }

    /// Prints frame-time statistics for a `--frames` run, skipping the warm-up.
    fn report(&mut self) {
        let Some(win) = &mut self.win else { return };
        if let Some(f) = win.in_flight.take() {
            let _ = f.wait(None);
        }
        let skip = (self.bench.len() / 5).max(1).min(self.bench.len());
        let mut t: Vec<f32> = self.bench[skip..].to_vec();
        if t.is_empty() {
            return;
        }
        // Raw frame times (seconds, warm-up skipped) for offline comparison of runs.
        if let Ok(path) = std::env::var("SPROUT_FRAMES_CSV") {
            let body: String = t.iter().map(|x| format!("{x}\n")).collect();
            let _ = std::fs::write(path, body);
        }
        t.sort_by(|a, b| a.total_cmp(b));
        let mean = t.iter().sum::<f32>() / t.len() as f32;
        let pct = |q: f32| t[((t.len() - 1) as f32 * q) as usize] * 1e3;
        println!(
            "frame time percentiles (ms): p5 {:.2}  p25 {:.2}  p50 {:.2}  p75 {:.2}  p90 {:.2}  p95 {:.2}  p99 {:.2}",
            pct(0.05), pct(0.25), pct(0.5), pct(0.75), pct(0.9), pct(0.95), pct(0.99)
        );
        let size = win.size();
        println!(
            "window {}x{}  grid {}x{}  {} particle slots  present {:?}",
            size[0], size[1], self.world.sim.grid[0], self.world.sim.grid[1], self.world.sim.slots(), win.swapchain.present_mode()
        );
        println!(
            "{} frames: mean {:.2} ms ({:.0} fps)  median {:.2}  p99 {:.2}  max {:.2} ms",
            t.len(), mean * 1e3, 1.0 / mean, pct(0.5), pct(0.99), pct(1.0)
        );
        // Below 1.0 the simulation could not keep up and the app slowed time down instead.
        let wall: f32 = self.bench.iter().sum();
        let simulated = self.world.sim.sim_time() as f32;
        println!("simulated {:.2} s in {:.2} s: {:.2}x real time", simulated, wall, simulated / wall);
    }

    fn zoom_at_cursor(&mut self, factor: f32) {
        let Some(win) = &self.win else { return };
        self.camera.zoom_at(self.input.cursor_px, factor, self.world.sim.grid, win.size());
    }

    fn redraw(&mut self) {
        let Some(win) = &mut self.win else { return };
        let size = win.size();
        if size[0] == 0 || size[1] == 0 {
            return;
        }

        let wanted_mode = pick_present_mode(&win.present_modes, self.ui.vsync);
        if win.recreate || wanted_mode != win.swapchain.present_mode() {
            let created = win.swapchain.recreate(SwapchainCreateInfo {
                image_extent: size,
                present_mode: wanted_mode,
                ..win.swapchain.create_info()
            });
            match created {
                Ok((swapchain, images)) => {
                    win.swapchain = swapchain;
                    win.views = images.into_iter().map(|i| ImageView::new_default(i).unwrap()).collect();
                    win.renderer.invalidate_targets();
                    win.recreate = false;
                    self.camera.resized(self.world.sim.grid, size);
                }
                // The window is mid-resize; try again next frame.
                Err(_) => return,
            }
        }

        let (image_index, suboptimal, acquire) =
            match acquire_next_image(win.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    win.recreate = true;
                    return;
                }
                Err(e) => panic!("failed to acquire a swapchain image: {e}"),
            };
        win.recreate |= suboptimal;

        // ---- time ------------------------------------------------------------------
        let now = Instant::now();
        let frame_dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.ui.frame_times.push(frame_dt);
        let dt = self.world.sim.dt_estimate();
        let fixed = self.world.sim.params.fixed_hz > 0.0;
        let mut plan = Plan { substeps: 0, target: 0.0, expected: 0.0 };
        if !self.ui.paused {
            let cost = self.world.sim.substep_cost();
            plan = self.pacer.plan(frame_dt, dt, fixed, self.ui.time_scale, cost, win.sim_budget(self.ui.vsync));
        } else if self.step_once {
            plan = Plan { substeps: 1, target: dt, expected: dt };
        }
        self.step_once = false;
        let substeps = plan.substeps;
        self.ui.substeps = substeps;
        self.ui.speed = if self.ui.paused { 1.0 } else { self.pacer.speed() };

        // ---- input -> world --------------------------------------------------------
        self.camera.update(frame_dt, size);
        let view = self.camera.view;
        let input = &mut self.input;
        input.cursor_world = view.screen_to_world(input.cursor_px, size);
        let raw_vel = [
            (input.cursor_world[0] - input.prev_world[0]) / frame_dt.max(1e-4),
            (input.cursor_world[1] - input.prev_world[1]) / frame_dt.max(1e-4),
        ];
        let blend = (frame_dt * 30.0).min(1.0);
        input.velocity[0] += (raw_vel[0] - input.velocity[0]) * blend;
        input.velocity[1] += (raw_vel[1] - input.velocity[1]) * blend;
        input.prev_world = input.cursor_world;

        let ppp = win.egui_ctx.pixels_per_point();
        let over_panel = win.egui_ctx.layer_id_at(egui::pos2(input.cursor_px[0] / ppp, input.cursor_px[1] / ppp))
            .is_some_and(|layer| layer.order != egui::Order::Background);
        let on_canvas = input.inside && !over_panel && !input.middle;
        let held = on_canvas && input.left && !input.right;
        let (at, radius, material) = (input.cursor_world, self.ui.brush, self.ui.material);
        let source = match self.ui.tool {
            Tool::Pour => Some(Source::Pour),
            Tool::Spray => Some(Source::Spray),
            Tool::Jet => Some(Source::Jet { angle: self.ui.jet_angle }),
            _ => None,
        };
        let mut tool = ToolState::default();
        if on_canvas && (input.right || (input.left && self.ui.tool == Tool::Grab)) {
            tool = ToolState::grab(at, input.velocity, radius);
            tool.strength *= self.ui.force_strength;
        } else if held && self.ui.tool == Tool::Push {
            tool = ToolState::push(at, radius);
            tool.strength *= self.ui.force_strength;
        } else if held && self.ui.tool == Tool::Erase {
            tool = ToolState::erase(at, radius);
        } else if held && let Some(source) = source {
            // The stroke is laid out for the time the simulation is about to advance, so
            // nothing is emitted while paused (single steps still emit).
            self.world.emit(source, material, at, radius, self.ui.pour_rate, plan.expected);
            tool = source.nozzle(at, radius, self.ui.pour_rate);
        }
        if !(held && source.is_some()) {
            self.world.end_stroke();
        }

        // ---- UI --------------------------------------------------------------------
        let raw_input = win.egui_state.take_egui_input(&win.window);
        let shape = if self.ui.tool == Tool::Remove {
            self.world.scene.pick(at).map(|i| (self.world.scene.shapes[i], true))
        } else {
            pending_shape(&self.ui, at).map(|s| (s, false))
        };
        let mut actions = UiActions::default();
        let (pressed, grabbing) = (on_canvas && (self.input.left || self.input.right), on_canvas && self.input.right);
        let full = win.egui_ctx.run(raw_input, |ctx| {
            actions = ui::build(ctx, &mut self.ui, &mut self.world);
            let now = (self.ui.tool, self.ui.material);
            if now.0 != self.shown.0 || (now.1 != self.shown.1 && self.ui.tool.uses_material()) {
                self.overlay.toast_tool(ctx, &self.ui);
            }
            self.shown = now;
            let hover = self.input.inside && !ctx.is_pointer_over_area();
            self.overlay.draw(ctx, &self.ui, view, size, hover.then_some(at), pressed, grabbing, shape.filter(|_| hover));
        });
        win.egui_state.handle_platform_output(&win.window, full.platform_output);
        let primitives = win.egui_ctx.tessellate(full.shapes, full.pixels_per_point);

        // ---- record ----------------------------------------------------------------
        let mut b = self.gpu.builder();
        self.world.sim.record(&mut b, substeps, plan.target, tool);
        self.world.sim.poll_stats();
        if self.frame_index.is_multiple_of(15) {
            self.world.sim.request_stats(&mut b);
        }
        win.painter.update_textures(&mut b, &full.textures_delta);
        let painter = &mut win.painter;
        win.renderer.render(&mut b, win.views[image_index as usize].clone(), &self.world.sim, view, |b| {
            painter.paint(b, &primitives, full.pixels_per_point, size)
        });
        win.painter.free_textures(&full.textures_delta);
        let commands = b.build().unwrap();

        // ---- submit ----------------------------------------------------------------
        // This frame was recorded while the GPU was still busy with the previous one. Let
        // that one finish before submitting: it keeps the GPU exactly one frame deep, and
        // releases its hold on the simulation buffers this frame is about to write.
        if let Some(f) = win.in_flight.take() {
            let _ = f.wait(None);
        }
        let submitted = sync::now(self.gpu.device.clone())
            .boxed()
            .join(acquire)
            .then_execute(self.gpu.queue.clone(), commands)
            .unwrap()
            .then_swapchain_present(
                self.gpu.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(win.swapchain.clone(), image_index),
            )
            .boxed()
            .then_signal_fence_and_flush();
        match submitted.map_err(Validated::unwrap) {
            Ok(f) => win.in_flight = Some(f),
            Err(VulkanError::OutOfDate) => win.recreate = true,
            Err(e) => eprintln!("frame submission failed: {e}"),
        }
        self.frame_index += 1;
        self.bench.push(frame_dt);

        // Actions that replace the world run last, once this frame no longer borrows it.
        self.apply(actions, None);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.win.is_none() {
            let win = Win::new(event_loop, &self.gpu, &self.opts);
            self.camera = Camera::new(self.world.sim.grid, win.size());
            self.win = Some(win);
            self.last_frame = Instant::now();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(win) = &mut self.win else { return };
        let consumed = win.egui_state.on_window_event(&win.window, &event).consumed;
        let wants_keys = win.egui_ctx.wants_keyboard_input();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => win.recreate = true,
            WindowEvent::RedrawRequested => {
                self.redraw();
                if self.opts.frames.is_some_and(|n| self.frame_index >= n) {
                    self.report();
                    event_loop.exit();
                }
            }
            WindowEvent::ModifiersChanged(m) => self.input.modifiers = m.state(),
            WindowEvent::CursorEntered { .. } => self.input.inside = true,
            WindowEvent::CursorLeft { .. } => {
                self.input.inside = false;
                self.input.left = false;
                self.input.right = false;
                self.input.middle = false;
            }
            WindowEvent::Focused(false) => {
                self.input.left = false;
                self.input.right = false;
                self.input.middle = false;
                self.input.velocity = [0.0; 2];
                self.input.modifiers = ModifiersState::empty();
            }
            WindowEvent::CursorMoved { position, .. } => self.pointer_moved([position.x as f32, position.y as f32]),
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                // A press that lands on the panel belongs to the panel; releases always count.
                if !(down && consumed) {
                    self.button(button, down);
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !consumed => self.wheel(match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
            }),
            WindowEvent::KeyboardInput { event, .. } if !wants_keys => {
                if let (PhysicalKey::Code(code), ElementState::Pressed) = (event.physical_key, event.state) {
                    let repeatable = matches!(
                        code,
                        KeyCode::BracketLeft | KeyCode::BracketRight | KeyCode::Minus | KeyCode::Equal | KeyCode::KeyN | KeyCode::Period
                    );
                    if !event.repeat || repeatable {
                        self.key(code, event_loop);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.win.is_some() {
            self.run_selftest(event_loop);
        }
        if let Some(win) = &self.win {
            win.window.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_the_key_its_label_shows() {
        for tool in Tool::LIQUID.into_iter().chain(Tool::SOLID) {
            let code = [KeyCode::KeyQ, KeyCode::KeyB, KeyCode::KeyJ, KeyCode::KeyC, KeyCode::KeyW, KeyCode::KeyV, KeyCode::KeyE,
                KeyCode::KeyA, KeyCode::KeyS, KeyCode::KeyD, KeyCode::KeyX]
                .into_iter()
                .find(|&c| format!("{c:?}") == format!("Key{}", tool.key()))
                .unwrap_or_else(|| panic!("no key for {}", tool.name()));
            assert_eq!(tool_for_key(code), Some(tool));
        }
    }
}
