//! The control panel and on-canvas overlays, built with egui.

use std::collections::VecDeque;

use egui::{Color32, RichText, Stroke};

use crate::{
    maps::MAPS,
    render::View,
    scene::Shape,
    sim::{Material, SimParams, SolverStats},
    world::World,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Pour,
    Grab,
    Erase,
    Circle,
    Box,
    Ramp,
    Remove,
}

impl Tool {
    const FLUID: [(Tool, &'static str, &'static str); 3] = [
        (Tool::Pour, "Pour", "Q"),
        (Tool::Grab, "Grab / stir", "W"),
        (Tool::Erase, "Erase", "E"),
    ];
    const SOLID: [(Tool, &'static str, &'static str); 4] = [
        (Tool::Circle, "Circle", "A"),
        (Tool::Box, "Box", "S"),
        (Tool::Ramp, "Ramp", "D"),
        (Tool::Remove, "Remove", "X"),
    ];
}

pub const GRIDS: [(&str, [u32; 2]); 4] = [
    ("Low  384 x 216", [384, 216]),
    ("Medium  512 x 288", [512, 288]),
    ("High  768 x 432", [768, 432]),
    ("Ultra  1024 x 576", [1024, 576]),
];

/// Rolling window of frame times.
#[derive(Default)]
pub struct FrameTimes {
    samples: VecDeque<f32>,
}

impl FrameTimes {
    pub fn push(&mut self, dt: f32) {
        self.samples.push_back(dt);
        while self.samples.len() > 120 {
            self.samples.pop_front();
        }
    }

    /// (mean, worst) frame time in milliseconds.
    pub fn summary(&self) -> (f32, f32) {
        if self.samples.is_empty() {
            return (0.0, 0.0);
        }
        let mean = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
        let worst = self.samples.iter().copied().fold(0.0, f32::max);
        (mean * 1e3, worst * 1e3)
    }
}

pub struct UiState {
    pub tool: Tool,
    pub material: Material,
    /// Brush radius in cells.
    pub brush: f32,
    /// Pour rate in particles per second.
    pub pour_rate: f32,
    /// Rotation of placed boxes and ramps, degrees.
    pub shape_angle: f32,
    /// Width / height ratio of placed boxes and ramps.
    pub shape_aspect: f32,
    pub shape_flip: bool,
    pub paused: bool,
    pub time_scale: f32,
    pub vsync: bool,
    pub show_panel: bool,
    pub grid: [u32; 2],
    /// Substeps taken in the last frame.
    pub substeps: u32,
    /// Simulated time delivered relative to real time; below 1 the scene is too heavy.
    pub speed: f32,
    pub frame_times: FrameTimes,
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            tool: Tool::Pour,
            material: Material::Water,
            brush: 9.0,
            pour_rate: 12000.0,
            shape_angle: 0.0,
            shape_aspect: 4.0,
            shape_flip: false,
            paused: false,
            time_scale: 1.0,
            vsync: true,
            show_panel: true,
            grid: [512, 288],
            substeps: 0,
            speed: 1.0,
            frame_times: FrameTimes::default(),
        }
    }
}

impl UiState {
    pub fn scale_brush(&mut self, factor: f32) {
        self.brush = (self.brush * factor).clamp(2.0, 60.0);
    }

    pub fn scale_rate(&mut self, factor: f32) {
        self.pour_rate = (self.pour_rate * factor).clamp(1000.0, 200_000.0);
    }

    pub fn rotate_shape(&mut self, degrees: f32) {
        self.shape_angle = (self.shape_angle + degrees + 90.0).rem_euclid(180.0) - 90.0;
    }
}

/// Things the panel asked for that the app has to carry out.
#[derive(Default)]
pub struct UiActions {
    pub reset: bool,
    pub load_map: Option<usize>,
    pub step: bool,
    pub grid: Option<[u32; 2]>,
    pub reset_view: bool,
    pub vsync_changed: bool,
    pub quit: bool,
}

pub fn style(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::dark());
    ctx.style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(6.0, 5.0);
        s.spacing.slider_width = 130.0;
        s.visuals.window_shadow = egui::Shadow::NONE;
    });
}

fn swatch(m: Material) -> Color32 {
    let c = m.color();
    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

fn tool_row(ui: &mut egui::Ui, state: &mut UiState, tools: &[(Tool, &str, &str)]) {
    ui.horizontal_wrapped(|ui| {
        for &(tool, name, key) in tools {
            ui.selectable_value(&mut state.tool, tool, name).on_hover_text(format!("Shortcut: {key}"));
        }
    });
}

fn physics(ui: &mut egui::Ui, p: &mut SimParams, solver: &SolverStats) {
    let d = SimParams::default();
    // Material properties are edited in physical units; the solver converts them.
    egui::Grid::new("physics").num_columns(2).show(ui, |ui| {
        ui.label("Gravity");
        ui.add(egui::Slider::new(&mut p.gravity, 0.0..=25.0).suffix(" m/s²"));
        ui.end_row();
        ui.label("Cell size");
        ui.add(egui::Slider::new(&mut p.cell_size, 0.0005..=0.05).logarithmic(true).suffix(" m"))
            .on_hover_text("Physical size of one grid cell. Smaller cells make surface tension matter more.");
        ui.end_row();
        for (i, name) in ["Water", "Syrup", "Oil"].iter().enumerate() {
            ui.label(format!("{name} viscosity"));
            ui.add(egui::Slider::new(&mut p.nu[i], 1e-7..=0.5).logarithmic(true).suffix(" m²/s"));
            ui.end_row();
        }
        for (i, name) in ["Water", "Syrup", "Oil"].iter().enumerate() {
            ui.label(format!("{name}-air tension"));
            ui.add(egui::Slider::new(&mut p.sigma[i], 0.0..=0.2).suffix(" N/m"));
            ui.end_row();
        }
        ui.label("Oil-water tension");
        ui.add(egui::Slider::new(&mut p.sigma_oil, 0.0..=0.1).suffix(" N/m"));
        ui.end_row();
        for (i, name) in ["Water", "Syrup", "Oil"].iter().enumerate() {
            ui.label(format!("{name} contact angle"));
            ui.add(egui::Slider::new(&mut p.contact_angle[i], 0.0..=180.0).suffix("°"));
            ui.end_row();
        }
        ui.label("Syrup density");
        ui.add(egui::Slider::new(&mut p.rho[1], 1.0..=2.5).suffix(" × water"));
        ui.end_row();
        ui.label("Oil density");
        ui.add(egui::Slider::new(&mut p.rho[2], 0.4..=1.0).suffix(" × water"));
        ui.end_row();
        ui.label("Oil droplet size");
        ui.add(egui::Slider::new(&mut p.droplet, 0.1..=4.0).suffix(" cells"))
            .on_hover_text("Size of oil droplets too small for the grid; they rise out of water at their terminal velocity.");
        ui.end_row();
        ui.label("Syrup mixing");
        ui.add(egui::Slider::new(&mut p.diffuse, 0.0..=20.0).suffix(" /s"))
            .on_hover_text("How fast touching water and syrup blend.");
        ui.end_row();
    });
    ui.separator();
    egui::Grid::new("solver").num_columns(2).show(ui, |ui| {
        ui.label("CFL number");
        ui.add(egui::Slider::new(&mut p.cfl, 0.2..=2.0)).on_hover_text("Most cells a particle may move per substep.");
        ui.end_row();
        ui.label("Pressure tolerance");
        ui.add(egui::Slider::new(&mut p.pressure_tol, 1e-6..=1e-2).logarithmic(true));
        ui.end_row();
        ui.label("Viscosity tolerance");
        ui.add(egui::Slider::new(&mut p.viscosity_tol, 1e-5..=1e-1).logarithmic(true));
        ui.end_row();
        ui.label("Substep");
        ui.label(format!("{:.2} ms (CFL {:.2} ms)", solver.dt * 1e3, solver.dt_cfl * 1e3));
        ui.end_row();
        ui.label("Iterations");
        ui.label(format!("pressure {}, viscosity {}", solver.pressure_iters_max, solver.viscosity_iters_max));
        ui.end_row();
    });
    if ui.button("Restore defaults").clicked() {
        *p = d;
    }
}

/// Builds the panel for this frame.
pub fn build(ctx: &egui::Context, state: &mut UiState, world: &mut World) -> UiActions {
    let mut a = UiActions::default();
    if !state.show_panel {
        return a;
    }
    egui::Window::new("Sprout")
        .default_pos([12.0, 12.0])
        .default_width(270.0)
        .resizable(false)
        .show(ctx, |ui| {
            // ---- transport ---------------------------------------------------------
            ui.horizontal(|ui| {
                let label = if state.paused { "▶ Resume" } else { "⏸ Pause" };
                if ui.button(label).on_hover_text("Space").clicked() {
                    state.paused = !state.paused;
                }
                if ui.button("Step").on_hover_text("N: advance one substep").clicked() {
                    a.step = true;
                }
                if ui.button("Reset").on_hover_text("R: reload the map").clicked() {
                    a.reset = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Map");
                let mut selected = world.map;
                egui::ComboBox::from_id_salt("map").width(170.0).selected_text(MAPS[world.map].name).show_ui(
                    ui,
                    |ui| {
                        for (i, m) in MAPS.iter().enumerate() {
                            ui.selectable_value(&mut selected, i, m.name).on_hover_text(m.blurb);
                        }
                    },
                );
                if selected != world.map {
                    a.load_map = Some(selected);
                }
            });
            ui.label(RichText::new(MAPS[world.map].blurb).weak().small());

            // ---- materials and tools -----------------------------------------------
            ui.separator();
            ui.horizontal(|ui| {
                for (i, m) in Material::ALL.into_iter().enumerate() {
                    let selected = state.material == m;
                    let text = RichText::new(format!("⏺ {}", m.name())).color(swatch(m)).strong();
                    if ui.selectable_label(selected, text).on_hover_text(format!("Shortcut: {}", i + 1)).clicked() {
                        state.material = m;
                        if !matches!(state.tool, Tool::Pour) {
                            state.tool = Tool::Pour;
                        }
                    }
                }
            });
            tool_row(ui, state, &Tool::FLUID);
            tool_row(ui, state, &Tool::SOLID);

            egui::Grid::new("brush").num_columns(2).show(ui, |ui| {
                ui.label("Brush size");
                ui.add(egui::Slider::new(&mut state.brush, 2.0..=60.0).logarithmic(true).suffix(" cells"))
                    .on_hover_text("[ and ], or Shift + wheel");
                ui.end_row();
                ui.label("Pour rate");
                let mut k = state.pour_rate / 1000.0;
                ui.add(egui::Slider::new(&mut k, 1.0..=200.0).logarithmic(true).suffix("k /s"))
                    .on_hover_text("Particles per second. - and = adjust it.\nA small brush caps how much it can deliver.");
                state.pour_rate = k * 1000.0;
                ui.end_row();
                if matches!(state.tool, Tool::Box | Tool::Ramp) {
                    ui.label("Shape angle");
                    ui.add(egui::Slider::new(&mut state.shape_angle, -90.0..=90.0).suffix("°"))
                        .on_hover_text("T and G rotate");
                    ui.end_row();
                    ui.label("Shape aspect");
                    ui.add(egui::Slider::new(&mut state.shape_aspect, 0.2..=12.0).logarithmic(true));
                    ui.end_row();
                }
            });
            if state.tool == Tool::Ramp {
                ui.checkbox(&mut state.shape_flip, "Flip ramp (F)");
            }

            // ---- status ------------------------------------------------------------
            ui.separator();
            let (mean, worst) = state.frame_times.summary();
            let stats = world.sim.stats;
            ui.label(
                RichText::new(format!(
                    "{:5.0} fps   {:4.1} ms  (worst {:4.1})\n{} substeps/frame   t = {:.1} s",
                    if mean > 0.0 { 1000.0 / mean } else { 0.0 },
                    mean,
                    worst,
                    state.substeps,
                    world.sim.sim_time(),
                ))
                .monospace(),
            );
            if state.speed < 0.95 {
                ui.label(RichText::new(format!("running at {:.0} % speed to keep the frame rate", state.speed * 100.0)).small().color(Color32::from_rgb(240, 180, 90)))
                    .on_hover_text("The scene is too heavy to simulate in real time. A lower quality or fewer substeps brings it back.");
            }
            ui.label(
                RichText::new(format!(
                    "{} particles\nwater {:.0}  oil {}  syrup {:.0}",
                    stats.live,
                    stats.water().max(0.0),
                    stats.oil,
                    stats.syrup
                ))
                .monospace(),
            );

            // ---- settings ----------------------------------------------------------
            ui.collapsing("Physics", |ui| {
                let solver = world.sim.solver;
                physics(ui, &mut world.sim.params, &solver);
            });
            ui.collapsing("Display", |ui| {
                let mut selected = state.grid;
                let current = GRIDS
                    .iter()
                    .find(|g| g.1 == state.grid)
                    .map(|g| g.0.to_string())
                    .unwrap_or(format!("{} x {}", state.grid[0], state.grid[1]));
                ui.horizontal(|ui| {
                    ui.label("Grid");
                    egui::ComboBox::from_id_salt("grid").width(150.0).selected_text(current).show_ui(ui, |ui| {
                        for (name, g) in GRIDS {
                            ui.selectable_value(&mut selected, g, name);
                        }
                    });
                });
                if selected != state.grid {
                    a.grid = Some(selected);
                }
                ui.horizontal(|ui| {
                    ui.label("Speed");
                    ui.add(egui::Slider::new(&mut state.time_scale, 0.1..=2.0).suffix("x"));
                });
                if ui.checkbox(&mut state.vsync, "VSync").changed() {
                    a.vsync_changed = true;
                }
                if ui.button("Reset view").on_hover_text("Home").clicked() {
                    a.reset_view = true;
                }
            });
            ui.collapsing("Controls", |ui| {
                ui.label(
                    RichText::new(
                        "Left mouse    use the selected tool\n\
                         Right mouse   grab / stir\n\
                         Middle drag   pan\n\
                         Wheel         zoom\n\
                         Shift+wheel   brush size  ([ ])\n\
                         - / =         pour rate\n\
                         1 2 3         water, oil, syrup\n\
                         Q W E         pour, grab, erase\n\
                         A S D X       circle, box, ramp, remove\n\
                         T G F         rotate / flip shape\n\
                         Space N R     pause, step, reset\n\
                         PgUp PgDn     previous / next map\n\
                         Home H F11    reset view, hide panel, fullscreen",
                    )
                    .monospace()
                    .small(),
                );
            });
        });
    a
}

/// Draws the outline of a solid on the canvas: a placement preview, or (with `danger`) the
/// shape that a click would remove.
pub fn outline(ctx: &egui::Context, shape: &Shape, view: View, size: [u32; 2], danger: bool) {
    let ppp = ctx.pixels_per_point();
    let to_screen = |p: [f32; 2]| {
        egui::pos2(
            (size[0] as f32 * 0.5 + (p[0] - view.center[0]) * view.zoom) / ppp,
            (size[1] as f32 * 0.5 - (p[1] - view.center[1]) * view.zoom) / ppp,
        )
    };
    let color = if danger { Color32::from_rgb(255, 110, 90) } else { Color32::from_white_alpha(200) };
    let stroke = Stroke::new(1.5_f32, color);
    let painter = ctx.layer_painter(egui::LayerId::background());
    let local = |c: [f32; 2], angle: f32, q: [f32; 2]| {
        let (s, co) = angle.sin_cos();
        to_screen([c[0] + co * q[0] - s * q[1], c[1] + s * q[0] + co * q[1]])
    };
    match *shape {
        Shape::Circle { c, r } => {
            painter.circle_stroke(to_screen(c), r * view.zoom / ppp, stroke);
        }
        Shape::Box { c, half, angle } => {
            let pts = [[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]]
                .map(|q| local(c, angle, q))
                .to_vec();
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        Shape::Ramp { c, half, angle, flip } => {
            let top = if flip { half[0] } else { -half[0] };
            let pts = [[-half[0], -half[1]], [half[0], -half[1]], [top, half[1]]].map(|q| local(c, angle, q)).to_vec();
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
    }
}
