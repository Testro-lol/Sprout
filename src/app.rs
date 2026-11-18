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
    sim::{Current, Material, ToolState},
    ui::{self, Canvas, Group, Overlay, Page, Placement, Tool, UiActions, UiState},
    world::{Pick, Source, World},
};

/// What a held left button is in the middle of, for the tools that are more than a brush.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gesture {
    None,
    /// Dragging out what a placing tool places, from where the button went down (cells).
    Place { anchor: [f32; 2] },
    /// Drawing a freehand line: the end of the last piece.
    Draw { last: [f32; 2] },
    /// Moving a solid or a device: where the cursor was at the last frame, and whether the
    /// thing has moved at all.
    Move { pick: Pick, last: [f32; 2], moved: bool },
}

/// How far the cursor must travel from where the button went down before a press counts
/// as dragging something out rather than a click, pixels.
const DRAG_MIN_PX: f32 = 6.0;

pub struct AppOptions {
    pub map: usize,
    pub grid: [u32; 2],
    pub size: [u32; 2],
    pub vsync: bool,
    /// Lowest frame rate kept at real time (`UiState::min_fps`); `None` keeps the default.
    pub min_fps: Option<f32>,
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
    /// Where the cursor was on screen at the last frame, pixels.
    prev_px: [f32; 2],
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
    gesture: Gesture,
    /// When the solids were last rasterised during a gesture (Draw, Move).
    last_raster: Instant,
    /// Tool and liquid of the last frame, to announce changes.
    shown: (Tool, Material),
    last_frame: Instant,
    pacer: Pacer,
    step_once: bool,
    frame_index: u64,
    /// Frame times of the whole run, kept only for `--frames`.
    bench: Vec<f32>,
    /// Simulated time at each benchmark frame, s.
    bench_sim: Vec<f64>,
    selftest: Option<SelfTest>,
    /// Liquid swatches, drawn by the renderer (made with the window).
    swatches: Option<crate::editor::Swatches>,
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
    let mut ui = UiState { vsync: opts.vsync, grid: opts.grid, ..Default::default() };
    if let Some(f) = opts.min_fps {
        ui.min_fps = f;
    }
    let mut app = App {
        camera: Camera::new(opts.grid, opts.size),
        overlay: Overlay::default(),
        gesture: Gesture::None,
        last_raster: Instant::now(),
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
        bench_sim: Vec::new(),
        selftest: None,
        swatches: None,
    };
    // Custom liquids. The self-test uses a library of its own, never the user's.
    let path = if app.opts.selftest {
        std::env::temp_dir().join(format!("sprout-selftest-{}", std::process::id())).join("liquids.json")
    } else {
        crate::library::Library::default_path()
    };
    let library = crate::library::Library::load(path);
    for w in &library.warnings {
        eprintln!("liquids file: {w}");
    }
    if !library.warnings.is_empty() {
        app.ui.editor.message = Some((format!("{} warning(s) loading the liquids file (see the terminal)", library.warnings.len()), false));
    }
    app.world.set_library(library);
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

/// The tool a key selects directly (`Tool::key` names it in the UI).
fn tool_for_key(code: KeyCode) -> Option<Tool> {
    Some(match code {
        KeyCode::KeyJ => Tool::Jet,
        KeyCode::KeyB => Tool::Spray,
        KeyCode::KeyC => Tool::Drop,
        KeyCode::KeyV => Tool::Push,
        KeyCode::KeyZ => Tool::Current,
        KeyCode::KeyX => Tool::Remove,
        KeyCode::KeyI => Tool::Pipette,
        KeyCode::KeyM => Tool::Magnet,
        KeyCode::KeyY => Tool::Heat,
        KeyCode::KeyU => Tool::Cool,
        _ => return None,
    })
}

/// The group a key opens (`Group::key`): the left hand's two rows, in the dock's order.
fn group_for_key(code: KeyCode) -> Option<Group> {
    Some(match code {
        KeyCode::KeyQ => Group::Pour,
        KeyCode::KeyW => Group::Move,
        KeyCode::KeyE => Group::Edit,
        KeyCode::KeyA => Group::Build,
        KeyCode::KeyS => Group::Devices,
        KeyCode::KeyD => Group::Heat,
        _ => return None,
    })
}

/// The palette slot of a number key.
fn slot_for_key(code: KeyCode) -> Option<usize> {
    [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4, KeyCode::Digit5, KeyCode::Digit6, KeyCode::Digit7, KeyCode::Digit8]
        .iter()
        .position(|c| *c == code)
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

    /// GPU time a frame may spend on the simulation, s: four fifths (the rest is for
    /// rendering) of a frame at `min_fps`, the lowest frame rate the simulation may push the
    /// app down to before simulated time slows down instead.
    fn sim_budget(min_fps: f32) -> f32 {
        0.8 / min_fps.max(1.0)
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
        self.gesture = Gesture::None;
        let changed = map != self.world.map;
        self.world.load_map(map);
        self.pacer.reset();
        if let Some(ctx) = self.egui() {
            if changed {
                self.overlay.toast_map(&ctx, self.world.map);
            } else {
                self.overlay.toast(&ctx, "Reset", MAPS[map].name, ui::theme::LEAF);
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
            self.overlay.toast(&ctx, "Quality", name, ui::theme::LEAF);
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
        if actions.clean_stains {
            self.world.sim.clean_stains();
        }
        if actions.clear_liquid {
            self.world.clear_liquid();
            self.pacer.reset();
            self.toast("Cleared", "The liquid is gone; solids and devices stay.", ui::theme::WATER);
        }
        if actions.undo {
            self.cancel_gesture();
            if self.world.undo() {
                self.toast("Undo", "", ui::theme::CLAY);
            }
        }
        if actions.redo && self.world.redo() {
            self.toast("Redo", "", ui::theme::CLAY);
        }
        if actions.vsync_changed && let Some(win) = &mut self.win {
            win.recreate = true;
        }
        if actions.quit && let Some(el) = event_loop {
            el.exit();
        }
    }

    fn toast(&mut self, text: &str, detail: &str, color: egui::Color32) {
        if let Some(ctx) = self.egui() {
            self.overlay.toast(&ctx, text, detail, color);
        }
    }

    fn key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        let mut a = UiActions::default();
        let n_maps = MAPS.len();
        let mods = self.input.modifiers;
        if mods.control_key() {
            // Commands, not tools.
            match code {
                KeyCode::KeyZ if mods.shift_key() => a.redo = true,
                KeyCode::KeyZ => a.undo = true,
                KeyCode::KeyY => a.redo = true,
                KeyCode::KeyQ => a.quit = true,
                _ => {}
            }
            self.apply(a, Some(event_loop));
            return;
        }
        if let Some(tool) = tool_for_key(code) {
            self.cancel_gesture();
            self.ui.select(tool);
        } else if let Some(group) = group_for_key(code) {
            self.cancel_gesture();
            self.ui.select_group(group, mods.shift_key());
        } else if let Some(slot) = slot_for_key(code) {
            self.ui.pick_material(Material(self.ui.palette[slot] as u8));
        }
        match code {
            KeyCode::Space => self.ui.paused = !self.ui.paused,
            KeyCode::KeyN | KeyCode::Period => a.step = true,
            KeyCode::KeyR if mods.shift_key() => a.clear_liquid = true,
            KeyCode::KeyR => a.reset = true,
            KeyCode::Tab => {
                self.cancel_gesture();
                self.ui.swap_tool();
            }
            KeyCode::KeyL => self.ui.editor.picker_open = !self.ui.editor.picker_open,
            KeyCode::BracketLeft => self.ui.scale_brush(1.0 / 1.15),
            KeyCode::BracketRight => self.ui.scale_brush(1.15),
            KeyCode::Minus => self.ui.scale_rate(1.0 / 1.25),
            KeyCode::Equal => self.ui.scale_rate(1.25),
            KeyCode::KeyT => {
                self.ui.rotate(15.0);
            }
            KeyCode::KeyG => {
                self.ui.rotate(-15.0);
            }
            KeyCode::KeyF => self.ui.flip(),
            KeyCode::KeyH => self.ui.show_panel = !self.ui.show_panel,
            KeyCode::F1 => self.ui.page = if self.ui.page == Some(Page::Keys) { None } else { Some(Page::Keys) },
            KeyCode::Home | KeyCode::Digit0 => a.reset_view = true,
            KeyCode::PageDown => a.load_map = Some((self.world.map + 1) % n_maps),
            KeyCode::PageUp => a.load_map = Some((self.world.map + n_maps - 1) % n_maps),
            KeyCode::Delete | KeyCode::Backspace => self.remove_at(self.input.cursor_world),
            KeyCode::F11 => {
                if let Some(win) = &self.win {
                    let fs = win.window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                    win.window.set_fullscreen(fs);
                }
            }
            KeyCode::Escape => self.escape(),
            _ => {}
        }
        self.apply(a, Some(event_loop));
    }

    /// Esc: backs out of whatever is in progress or open, one thing at a time.
    fn escape(&mut self) {
        let ui = &mut self.ui;
        if self.gesture != Gesture::None {
            self.cancel_gesture();
        } else if ui.maps_open {
            ui.maps_open = false;
        } else if ui.stats_open {
            ui.stats_open = false;
        } else if ui.page.is_some() {
            ui.page = None;
        } else if ui.editor.open {
            ui.editor.open = false;
        } else if ui.editor.picker_open {
            ui.editor.picker_open = false;
        } else if !ui.show_panel {
            ui.show_panel = true;
        }
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
                    self.press();
                } else {
                    self.release();
                }
            }
            // The right button backs out of a drag; otherwise it grabs.
            MouseButton::Right if down && self.gesture != Gesture::None => self.cancel_gesture(),
            MouseButton::Right => self.input.right = down,
            MouseButton::Middle => self.input.middle = down,
            _ => {}
        }
    }

    /// The wheel zooms; with Shift it sizes the brush, with Ctrl it turns what the tool aims
    /// or places, with Alt it sets the flow or strength.
    fn wheel(&mut self, lines: f32) {
        let mods = self.input.modifiers;
        if mods.shift_key() {
            self.ui.scale_brush(1.12f32.powf(lines));
        } else if mods.control_key() {
            self.ui.rotate(lines * 15.0);
        } else if mods.alt_key() {
            self.ui.scale_rate(1.12f32.powf(lines));
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
        let ed = &self.ui.editor;
        let file = crate::library::Library::load(self.world.library.path.clone()).custom;
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
            oil: ps.iter().filter(|p| p.is(Material::Oil)).count(),
            honey: ps.iter().map(|p| p.share(Material::Honey, &self.world.sim.params)).sum(),
            rms_speed: (speed2 / ps.len().max(1) as f32).sqrt(),
            shapes: self.world.scene.shapes.len(),
            substeps: self.world.sim.substeps_done(),
            grid: self.world.sim.grid,
            map: self.world.map,
            zoom: self.camera.target().zoom,
            center: self.camera.target().center,
            brush: self.ui.brush,
            currents: self.world.currents().len(),
            drift: ps.iter().map(|p| p.vel[0]).sum::<f32>() / ps.len().max(1) as f32,
            jet_angle: self.ui.jet_angle,
            panel: self.ui.show_panel,
            custom: self.world.library.custom.iter().filter(|c| !c.deleted).count(),
            file_ids: file.len(),
            edit_row: ed.row,
            edit_particles: ps.iter().filter(|p| p.material().is_some_and(|m| m.index() == ed.row)).count(),
            live_visc: self.world.sim.params.mat[ed.row].viscosity,
            draft_visc: ed.draft.viscosity,
            file_matches: ed.row >= crate::sim::BUILTINS.len()
                && file.iter().zip(&self.world.library.custom).any(|(f, c)| f.id == c.id && f.props == self.world.sim.params.mat[ed.row] && crate::library::custom_row(self.world.library.custom.iter().position(|x| x.id == c.id).unwrap()) == ed.row),
            water: ps.iter().filter(|p| p.is(Material::Water)).count(),
            magnets: self.world.sim.magnets().len(),
            tool: self.ui.tool.name(),
            faucets: self.world.faucets().len(),
            drains: self.world.drains().len(),
            pumps: self.world.fans().len(),
            can_undo: self.world.can_undo(),
            material: self.ui.material.index(),
            spin: {
                let near: Vec<f32> = ps
                    .iter()
                    .map(|p| ([p.pos[0] - self.input.cursor_world[0], p.pos[1] - self.input.cursor_world[1]], p.vel))
                    .filter(|(d, _)| d[0].hypot(d[1]) < self.ui.brush)
                    .map(|(d, v)| d[0] * v[1] - d[1] * v[0])
                    .collect();
                near.iter().sum::<f32>() / near.len().max(1) as f32
            },
            mean_y: ps.iter().map(|p| p.pos[1]).sum::<f32>() / ps.len().max(1) as f32,
            mean_t: ps.iter().map(|p| self.world.sim.params.temperature(p).0).sum::<f32>() / ps.len().max(1) as f32,
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
                Act::Tool(tool) => {
                    self.cancel_gesture();
                    self.ui.select(tool);
                }
                Act::Mods(mods) => self.input.modifiers = mods,
                Act::Tweak(f) => f(&mut self.ui),
                Act::Grid(grid) => self.apply(UiActions { grid: Some(grid), ..Default::default() }, Some(event_loop)),
                Act::Mark => test.mark = self.snapshot(),
                Act::Liquid(act) => {
                    use selftest::LiquidAct;
                    let ed = &mut self.ui.editor;
                    match act {
                        LiquidAct::Duplicate(row) => {
                            ed.duplicate(&mut self.world, row);
                        }
                        LiquidAct::Edit(f) => f(&mut ed.draft),
                        LiquidAct::Apply => {
                            ed.apply(&mut self.world);
                        }
                        LiquidAct::Save => {
                            ed.save(&mut self.world);
                        }
                        LiquidAct::Delete => {
                            ed.delete(&mut self.world);
                        }
                        LiquidAct::Pick => {
                            let row = ed.row;
                            self.ui.pick_material(Material(row as u8));
                        }
                        LiquidAct::Select(row) => ed.select(&self.world, row),
                    }
                }
                Act::Check(what, check) => {
                    let now = self.snapshot();
                    let ok = check(&test.mark, &now);
                    test.failures += !ok as u32;
                    println!(
                        "  [{}] {:<38} {} particles ({} oil, {:.0} honey), {} solids, rms {:.0} cells/s",
                        if ok { "pass" } else { "FAIL" }, what, now.particles, now.oil, now.honey.abs(), now.shapes, now.rms_speed.abs()
                    );
                    if !ok {
                        println!("         at the mark: {:?}\n         now:         {:?}", test.mark, now);
                    }
                }
            }
        }
        if test.next == test.script.len() {
            println!("\n{}", if test.failures == 0 { "all checks passed".to_string() } else { format!("{} check(s) FAILED", test.failures) });
            event_loop.exit();
        }
        self.selftest = Some(test);
    }

    fn liquid_color(&self) -> egui::Color32 {
        ui::swatch_of(&self.world.sim.params, self.ui.material.index())
    }

    fn ripple(&mut self, at: [f32; 2], radius: f32, color: egui::Color32) {
        if let Some(ctx) = self.egui() {
            self.overlay.ripple(&ctx, at, radius, color);
        }
    }

    /// What the selected tool would place now: at the cursor, or dragged out from `anchor`.
    /// The flag says whether the drag has gone far enough to count.
    fn placement(&self, anchor: Option<[f32; 2]>) -> Option<(Placement, bool)> {
        let at = self.input.cursor_world;
        let drag_min = DRAG_MIN_PX / self.camera.view.zoom;
        let dragged = anchor.is_some_and(|a| (at[0] - a[0]).hypot(at[1] - a[1]) > drag_min);
        ui::placement(&self.ui, anchor, at, drag_min, self.input.modifiers.shift_key()).map(|p| (p, dragged))
    }

    /// A press of the left button on the canvas: the tools that act once per click, and the
    /// start of the ones that are dragged.
    fn press(&mut self) {
        let at = self.input.cursor_world;
        let (tool, material, radius) = (self.ui.tool, self.ui.material, self.ui.brush);
        if self.input.modifiers.alt_key() || tool == Tool::Pipette {
            self.sample(at);
            return;
        }
        match tool {
            Tool::Drop => {
                if self.world.stamp(material, at, radius) > 0 {
                    self.ripple(at, radius, tool.color(self.liquid_color()));
                }
            }
            Tool::Remove => self.remove_at(at),
            Tool::Move => {
                if let Some(pick) = self.world.pick(at) {
                    self.world.checkpoint();
                    self.gesture = Gesture::Move { pick, last: at, moved: false };
                }
            }
            Tool::Draw => {
                self.world.checkpoint();
                self.world.add_shape_later(Shape::Capsule { a: at, b: at, r: self.ui.thickness }, self.ui.solid_temp());
                self.gesture = Gesture::Draw { last: at };
            }
            t if t.places() => self.gesture = Gesture::Place { anchor: at },
            _ => {}
        }
    }

    /// The left button came up: what was being dragged out is placed.
    fn release(&mut self) {
        match std::mem::replace(&mut self.gesture, Gesture::None) {
            Gesture::Place { anchor } => {
                if let Some((p, dragged)) = self.placement(Some(anchor)) {
                    self.place(p, dragged);
                }
            }
            Gesture::Move { moved: false, .. } => self.world.drop_checkpoint(),
            _ => {}
        }
        self.world.flush_solids();
    }

    /// Backs out of a drag: nothing is placed, a line being drawn is taken back, a thing
    /// being moved returns to where it was.
    fn cancel_gesture(&mut self) {
        match std::mem::replace(&mut self.gesture, Gesture::None) {
            Gesture::Draw { .. } | Gesture::Move { moved: true, .. } => {
                self.world.undo();
            }
            Gesture::Move { moved: false, .. } => self.world.drop_checkpoint(),
            _ => {}
        }
        self.world.flush_solids();
    }

    fn place(&mut self, p: Placement, dragged: bool) {
        self.world.checkpoint();
        let color = self.ui.tool.color(self.liquid_color());
        let (ok, at, radius, full) = match p {
            Placement::Solid(s) => {
                self.world.add_shape_at(s, self.ui.solid_temp());
                (true, s.center(), s.bound().1.min(24.0), "")
            }
            Placement::Faucet(f) => (self.world.add_faucet(f), f.pos, f.radius, "The tank holds six faucets"),
            Placement::Pump(c) => (self.world.add_current(c), c.pos, c.radius, "The tank holds fifteen pumps and drains"),
            Placement::Drain(c) => (self.world.add_drain(c), c.pos, c.radius, "The tank holds fifteen pumps and drains"),
            Placement::Magnet(m) => (self.world.add_magnet(m), m.pos, m.radius, "The tank holds eight magnets"),
        };
        if !ok {
            self.world.drop_checkpoint();
            self.toast(full, "Remove one first (the Remove tool, or Delete over it).", ui::theme::WARN);
            return;
        }
        // A heading set by dragging stays for the next one.
        if dragged {
            match p {
                Placement::Faucet(f) => self.ui.faucet_angle = f.angle,
                Placement::Pump(c) => self.ui.pump_angle = c.vel[0].atan2(-c.vel[1]).to_degrees(),
                Placement::Magnet(m) => self.ui.magnet_angle = m.dir[0].atan2(m.dir[1]).to_degrees(),
                _ => {}
            }
        }
        self.ripple(at, radius, color);
    }

    /// Removes the solid or device under `at` (the Remove tool, or Delete with any tool).
    fn remove_at(&mut self, at: [f32; 2]) {
        if let Some(pick) = self.world.pick(at) {
            self.cancel_gesture();
            self.world.checkpoint();
            self.world.remove_pick(pick);
            self.ripple(at, 2.0, ui::theme::DANGER);
        }
    }

    /// The pipette: the liquid under `at` becomes the one to pour.
    fn sample(&mut self, at: [f32; 2]) {
        if let Some(f) = self.win.as_mut().and_then(|w| w.in_flight.take()) {
            let _ = f.wait(None);
        }
        match self.world.sample(at, self.ui.brush.max(3.0)) {
            Some(s) => {
                let name = self.world.library.name(s.row);
                self.ui.material = Material(s.row as u8);
                let color = ui::theme::lifted(self.liquid_color());
                let dx = self.world.sim.params.cell_size;
                let detail = format!("{:.0} °C{}  ·  {:.2} m/s", s.temp, if s.frozen { ", frozen" } else { "" }, s.speed * dx);
                self.toast(&name, &detail, color);
                self.ripple(at, 2.0, color);
            }
            None => self.toast("No liquid here", "", ui::theme::TEXT_WEAK),
        }
    }

    /// Carries on what the held button is in the middle of, once per frame.
    fn continue_gesture(&mut self) {
        let at = self.input.cursor_world;
        match self.gesture {
            Gesture::Draw { last } => {
                // A new piece once the cursor is a thickness away: short enough to follow a
                // curve, long enough not to stack pieces on a slow hand.
                let r = self.ui.thickness;
                if (at[0] - last[0]).hypot(at[1] - last[1]) >= r.max(1.5) {
                    self.world.add_shape_later(Shape::Capsule { a: last, b: at, r }, self.ui.solid_temp());
                    self.gesture = Gesture::Draw { last: at };
                }
            }
            Gesture::Move { pick, last, moved } => {
                let d = [at[0] - last[0], at[1] - last[1]];
                if d != [0.0, 0.0] {
                    self.world.move_pick(pick, d);
                    self.gesture = Gesture::Move { pick, last: at, moved: true };
                } else if !moved {
                    self.gesture = Gesture::Move { pick, last, moved };
                }
            }
            _ => {}
        }
        // The overlay shows the line or the moved solid at once; the simulation gets it as
        // often as rasterising allows: at most 60 times a second, and for no more than half
        // the time (a solid that jumps further than its own thickness between two
        // rasterisations leaves the liquid it was carrying behind).
        let pace = (2.0 * self.world.raster_cost()).max(1.0 / 60.0);
        if self.last_raster.elapsed().as_secs_f32() >= pace && self.world.flush_solids() {
            self.last_raster = Instant::now();
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
        // The same past the warm-up (a faster build reaches the end of its frames sooner, so
        // the start weighs more in the whole-run figure).
        let wall_late: f32 = self.bench[skip..].iter().sum();
        let sim_late = self.bench_sim.last().copied().unwrap_or(0.0) - self.bench_sim[skip - 1];
        println!("after warm-up: simulated {:.2} s in {:.2} s: {:.2}x real time", sim_late, wall_late, sim_late as f32 / wall_late);
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
        // The self-test script counts frames of 1/60 s. Vsync alone does not give it those:
        // on a faster display (144 Hz, say) each scripted wait covered less than half the
        // simulated time it was written for, and liquid had not got where a check expected it.
        if self.selftest.is_some() {
            let due = self.last_frame + std::time::Duration::from_secs_f64(selftest::FRAME);
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
        }
        let now = Instant::now();
        let frame_dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.ui.frame_times.push(frame_dt);
        let dt = self.world.sim.dt_estimate();
        let fixed = self.world.sim.params.fixed_hz > 0.0;
        let mut plan = Plan { substeps: 0, target: 0.0, expected: 0.0 };
        if !self.ui.paused {
            let cost = self.world.sim.substep_cost();
            plan = self.pacer.plan(frame_dt, dt, fixed, self.ui.time_scale, cost, Win::sim_budget(self.ui.min_fps));
        } else if self.step_once {
            plan = Plan { substeps: 1, target: dt, expected: dt };
        }
        self.step_once = false;
        let substeps = plan.substeps;
        self.ui.substeps = substeps;
        self.ui.speed = if self.ui.paused { 1.0 } else { self.pacer.speed() };

        // ---- input -> world --------------------------------------------------------
        // The tank sits above the dock while the interface is shown, and fills the window
        // when it is hidden.
        let below = if self.ui.show_panel { ui::DOCK_INSET * win.egui_ctx.pixels_per_point() } else { 0.0 };
        self.camera.set_below(below.min(size[1] as f32 * 0.4), self.world.sim.grid, size);
        self.camera.update(frame_dt, size);
        let view = self.camera.view;
        let input = &mut self.input;
        input.cursor_world = view.screen_to_world(input.cursor_px, size);
        // The cursor's own motion, both ends seen through this frame's view: while the camera
        // eases (a zoom, Home) the point under a still cursor moves through the world, and
        // stirring then flung the liquid along with the view.
        let before = view.screen_to_world(input.prev_px, size);
        let raw_vel = [
            (input.cursor_world[0] - before[0]) / frame_dt.max(1e-4),
            (input.cursor_world[1] - before[1]) / frame_dt.max(1e-4),
        ];
        let blend = (frame_dt * 30.0).min(1.0);
        input.velocity[0] += (raw_vel[0] - input.velocity[0]) * blend;
        input.velocity[1] += (raw_vel[1] - input.velocity[1]) * blend;
        input.prev_px = input.cursor_px;

        let ppp = win.egui_ctx.pixels_per_point();
        let over_panel = win.egui_ctx.layer_id_at(egui::pos2(input.cursor_px[0] / ppp, input.cursor_px[1] / ppp))
            .is_some_and(|layer| layer.order != egui::Order::Background);
        let on_canvas = input.inside && !over_panel && !input.middle;
        let held = on_canvas && input.left && !input.right;
        let (at, radius, material) = (input.cursor_world, self.ui.brush, self.ui.material);
        // Alt + click samples (the pipette) with any tool.
        let held = held && !input.modifiers.alt_key();
        let (left, right) = (input.left, input.right);
        self.continue_gesture();
        let source = match self.ui.tool {
            Tool::Pour => Some(Source::Pour),
            Tool::Spray => Some(Source::Spray),
            Tool::Rain => Some(Source::Rain),
            Tool::Jet => Some(Source::Jet { angle: self.ui.jet_angle }),
            _ => None,
        };
        let mut tool = ToolState::default();
        let force = |mut t: ToolState, k: f32| {
            t.strength *= k;
            t
        };
        let strength = self.ui.force_strength;
        if on_canvas && self.gesture == Gesture::None && (right || (left && self.ui.tool == Tool::Grab)) {
            tool = force(ToolState::grab(at, self.input.velocity, radius), strength);
        } else if held {
            match self.ui.tool {
                Tool::Push => tool = force(ToolState::push(at, radius), strength),
                Tool::Pull => tool = force(ToolState::pull(at, radius), strength),
                Tool::Whirl => tool = force(ToolState::whirl(at, radius, self.ui.whirl_clockwise), strength),
                Tool::Hold => tool = force(ToolState::hold(at, radius), strength),
                Tool::Erase => tool = ToolState::erase(at, radius),
                Tool::Heat | Tool::Cool => tool = ToolState::heat(at, radius, self.ui.tool == Tool::Cool),
                _ => {}
            }
            if let Some(source) = source {
                // The stroke is laid out for the time the simulation is about to advance, so
                // nothing is emitted while paused (single steps still emit).
                self.world.emit(source, material, at, radius, self.ui.pour_rate, plan.expected);
                tool = source.nozzle(at, radius, self.ui.pour_rate);
            }
        }
        if !(held && source.is_some()) {
            self.world.end_stroke();
        }
        self.world.run_faucets(plan.expected);
        // The Current tool drives the liquid under the cursor while the button is held.
        let current = (held && self.ui.tool == Tool::Current)
            .then(|| Current::new(at, radius, self.ui.current_angle, self.ui.current_strength));
        self.world.hold_current(current);

        // ---- UI --------------------------------------------------------------------
        // What a click would place, remove or move, for the overlay to show.
        let placing = match self.gesture {
            Gesture::Place { anchor } => Some(anchor),
            _ => None,
        };
        let ghost = self.placement(placing);
        let target = match (self.gesture, self.ui.tool) {
            (Gesture::Move { pick, .. }, _) => Some((pick, false)),
            (_, Tool::Remove) => self.world.pick(at).map(|p| (p, true)),
            (_, Tool::Move) => self.world.pick(at).map(|p| (p, false)),
            _ => None,
        };
        let busy = self.gesture != Gesture::None;
        let Some(win) = &mut self.win else { return };
        let raw_input = win.egui_state.take_egui_input(&win.window);
        let mut actions = UiActions::default();
        let (pressed, grabbing) = (on_canvas && (self.input.left || self.input.right), on_canvas && self.input.right && !busy);
        let full = win.egui_ctx.run(raw_input, |ctx| {
            let swatches = self.swatches.get_or_insert_with(|| crate::editor::Swatches::new(self.gpu.clone()));
            actions = ui::build(ctx, &mut self.ui, &mut self.world, swatches);
            let now = (self.ui.tool, self.ui.material);
            // With the interface hidden a change of tool or liquid is announced instead.
            if !self.ui.show_panel && (now.0 != self.shown.0 || (now.1 != self.shown.1 && self.ui.tool.uses_material())) {
                self.overlay.toast_tool(ctx, &self.ui, &self.world);
            }
            self.shown = now;
            let hover = self.input.inside && !ctx.is_pointer_over_area();
            let canvas = Canvas {
                view,
                size,
                cursor: (hover || busy).then_some(at),
                pressed,
                grabbing,
                ghost: ghost.filter(|_| hover || busy),
                target: target.filter(|_| hover || busy),
                world: &self.world,
            };
            self.overlay.draw(ctx, &self.ui, &canvas);
        });
        win.egui_state.handle_platform_output(&win.window, full.platform_output);
        let primitives = win.egui_ctx.tessellate(full.shapes, full.pixels_per_point);

        // ---- record ----------------------------------------------------------------
        // Read-backs are picked up before this frame records new ones. (Polled after `record`,
        // the read-back it had just asked for was not submitted yet, so nothing held its host
        // copy and reading it succeeded: the slot was taken as done and its data lost, and
        // the next request into it read what an earlier copy had left there. After a reset that
        // stale copy was the old tank's time: simulated time jumped ahead by the time
        // since the last reset and then stood still until the new tank caught up, and the
        // bubbles, foam and specks, which run on it, were neither made nor moved meanwhile.)
        self.world.sim.poll_stats();
        let mut b = self.gpu.builder();
        self.world.sim.record(&mut b, substeps, plan.target, tool);
        if self.frame_index.is_multiple_of(15) {
            self.world.sim.request_stats(&mut b);
        }
        win.painter.update_textures(&mut b, &full.textures_delta);
        let painter = &mut win.painter;
        win.renderer.look = self.ui.look;
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
        self.bench_sim.push(self.world.sim.sim_time());

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
                self.cancel_gesture();
            }
            WindowEvent::Focused(false) => {
                self.cancel_gesture();
                self.input.left = false;
                self.input.right = false;
                self.input.middle = false;
                self.input.velocity = [0.0; 2];
                self.input.modifiers = ModifiersState::empty();
            }
            WindowEvent::CursorMoved { position, .. } => self.pointer_moved([position.x as f32, position.y as f32]),
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                // A press that lands on a panel belongs to the panel; releases always count.
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
                        KeyCode::BracketLeft | KeyCode::BracketRight | KeyCode::Minus | KeyCode::Equal | KeyCode::KeyN | KeyCode::Period | KeyCode::KeyT | KeyCode::KeyG
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

    const KEYS: [KeyCode; 26] = [
        KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC, KeyCode::KeyD, KeyCode::KeyE, KeyCode::KeyF, KeyCode::KeyG, KeyCode::KeyH, KeyCode::KeyI,
        KeyCode::KeyJ, KeyCode::KeyK, KeyCode::KeyL, KeyCode::KeyM, KeyCode::KeyN, KeyCode::KeyO, KeyCode::KeyP, KeyCode::KeyQ, KeyCode::KeyR,
        KeyCode::KeyS, KeyCode::KeyT, KeyCode::KeyU, KeyCode::KeyV, KeyCode::KeyW, KeyCode::KeyX, KeyCode::KeyY, KeyCode::KeyZ,
    ];

    fn code(letter: &str) -> KeyCode {
        KEYS.into_iter().find(|c| format!("{c:?}") == format!("Key{letter}")).unwrap_or_else(|| panic!("no key {letter}"))
    }

    #[test]
    fn every_tool_and_group_has_the_key_its_label_shows() {
        for tool in Tool::all() {
            if let Some(k) = tool.key() {
                assert_eq!(tool_for_key(code(k)), Some(tool), "{}", tool.name());
            }
        }
        for group in Group::ALL {
            assert_eq!(group_for_key(code(group.key())), Some(group));
        }
        // No key means two things, and none of them is taken by a command.
        for c in KEYS {
            assert!(tool_for_key(c).is_none() || group_for_key(c).is_none(), "{c:?}");
            let command = matches!(c, KeyCode::KeyN | KeyCode::KeyR | KeyCode::KeyL | KeyCode::KeyT | KeyCode::KeyG | KeyCode::KeyF | KeyCode::KeyH);
            assert!(!(command && (tool_for_key(c).is_some() || group_for_key(c).is_some())), "{c:?}");
        }
    }
}
