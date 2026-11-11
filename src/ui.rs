//! The control panel and the on-canvas overlays (brush cursor, shape previews, toasts and
//! ripples), built with egui.
//!
//! Every overlay is drawn in screen space each frame from the state it shows; the only
//! state kept here is what animates: `Overlay` eases the brush cursor towards the current
//! brush and fades toasts and ripples in and out.

use std::collections::VecDeque;

use egui::{Color32, CornerRadius, Pos2, RichText, Shadow, Stroke, Vec2, emath::easing};

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
    Spray,
    Jet,
    Drop,
    Grab,
    Push,
    Erase,
    Circle,
    Box,
    Ramp,
    Remove,
}

impl Tool {
    pub const LIQUID: [Tool; 7] = [Tool::Pour, Tool::Spray, Tool::Jet, Tool::Drop, Tool::Grab, Tool::Push, Tool::Erase];
    pub const SOLID: [Tool; 4] = [Tool::Circle, Tool::Box, Tool::Ramp, Tool::Remove];

    pub fn name(self) -> &'static str {
        match self {
            Tool::Pour => "Pour",
            Tool::Spray => "Spray",
            Tool::Jet => "Jet",
            Tool::Drop => "Drop",
            Tool::Grab => "Grab",
            Tool::Push => "Push",
            Tool::Erase => "Erase",
            Tool::Circle => "Circle",
            Tool::Box => "Box",
            Tool::Ramp => "Ramp",
            Tool::Remove => "Remove",
        }
    }

    /// Keyboard shortcut (see `app::tool_for_key`).
    pub fn key(self) -> &'static str {
        match self {
            Tool::Pour => "Q",
            Tool::Spray => "B",
            Tool::Jet => "J",
            Tool::Drop => "C",
            Tool::Grab => "W",
            Tool::Push => "V",
            Tool::Erase => "E",
            Tool::Circle => "A",
            Tool::Box => "S",
            Tool::Ramp => "D",
            Tool::Remove => "X",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tool::Pour => "Hold for a steady stream, like a faucet. Flow sets how fast it runs.",
            Tool::Spray => "Hold to throw droplets outwards from the rim of the brush.",
            Tool::Jet => "Hold for a powered stream. T / G or the wheel aims it (Alt + wheel zooms).",
            Tool::Drop => "Click to drop a round blob. Works while paused.",
            Tool::Grab => "Drag to carry and stir liquid. The right button grabs with any tool.",
            Tool::Push => "Hold to drive liquid outwards from the cursor.",
            Tool::Erase => "Hold to remove liquid under the brush.",
            Tool::Circle | Tool::Box | Tool::Ramp => "Click to place. T / G rotate, F flips a ramp.",
            Tool::Remove => "Click a highlighted solid to remove it.",
        }
    }

    /// Tools that add the selected liquid.
    pub fn uses_material(self) -> bool {
        matches!(self, Tool::Pour | Tool::Spray | Tool::Jet | Tool::Drop)
    }

    /// Tools whose strength is a force rather than a flow rate.
    pub fn is_force(self) -> bool {
        matches!(self, Tool::Grab | Tool::Push)
    }

    pub fn places_shape(self) -> bool {
        matches!(self, Tool::Circle | Tool::Box | Tool::Ramp)
    }

    /// Accent colour of the brush cursor; liquid tools take the liquid's colour.
    fn color(self, material: Material) -> Color32 {
        match self {
            Tool::Grab => Color32::from_rgb(120, 228, 200),
            Tool::Push => Color32::from_rgb(196, 160, 255),
            Tool::Erase | Tool::Remove => DANGER,
            Tool::Circle | Tool::Box | Tool::Ramp => Color32::from_gray(225),
            _ => swatch(material),
        }
    }
}

pub const GRIDS: [(&str, [u32; 2]); 4] = [
    ("Low  384 x 216", [384, 216]),
    ("Medium  512 x 288", [512, 288]),
    ("High  768 x 432", [768, 432]),
    ("Ultra  1024 x 576", [1024, 576]),
];

const ACCENT: Color32 = Color32::from_rgb(110, 190, 255);
const DANGER: Color32 = Color32::from_rgb(255, 110, 96);
const WARN: Color32 = Color32::from_rgb(240, 180, 90);

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
    /// Flow of the liquid tools, particles per second.
    pub pour_rate: f32,
    /// Strength of grab and push relative to their default.
    pub force_strength: f32,
    /// Jet heading in degrees clockwise from straight down.
    pub jet_angle: f32,
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
            force_strength: 1.0,
            jet_angle: 0.0,
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

    /// `-` and `=`: the flow of liquid tools, or the strength of force tools.
    pub fn scale_rate(&mut self, factor: f32) {
        if self.tool.is_force() {
            self.force_strength = (self.force_strength * factor).clamp(0.1, 4.0);
        } else {
            self.pour_rate = (self.pour_rate * factor).clamp(1000.0, 200_000.0);
        }
    }

    /// T / G: aims the jet, or turns the shape about to be placed.
    pub fn rotate(&mut self, degrees: f32) {
        if self.tool == Tool::Jet {
            self.rotate_jet(degrees);
        } else {
            self.rotate_shape(degrees);
        }
    }

    pub fn rotate_jet(&mut self, degrees: f32) {
        self.jet_angle = (self.jet_angle + degrees + 180.0).rem_euclid(360.0) - 180.0;
    }

    pub fn rotate_shape(&mut self, degrees: f32) {
        self.shape_angle = (self.shape_angle + degrees + 90.0).rem_euclid(180.0) - 90.0;
    }

    /// Picks a liquid; a tool that does not add liquid gives way to Pour.
    pub fn pick_material(&mut self, material: Material) {
        self.material = material;
        if !self.tool.uses_material() {
            self.tool = Tool::Pour;
        }
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
    let mut v = egui::Visuals::dark();
    let radius = CornerRadius::same(6);
    v.window_corner_radius = CornerRadius::same(12);
    v.menu_corner_radius = CornerRadius::same(8);
    v.window_fill = Color32::from_rgb(20, 23, 29);
    v.panel_fill = v.window_fill;
    v.window_stroke = Stroke::new(1.0_f32, Color32::from_white_alpha(16));
    v.window_shadow = Shadow { offset: [0, 8], blur: 24, spread: 0, color: Color32::from_black_alpha(110) };
    v.popup_shadow = Shadow { offset: [0, 4], blur: 12, spread: 0, color: Color32::from_black_alpha(90) };
    v.selection.bg_fill = Color32::from_rgb(38, 72, 104);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.hyperlink_color = ACCENT;
    v.slider_trailing_fill = true;
    // Opaque fills, a step lighter than the panel per state.
    for (w, fill) in [
        (&mut v.widgets.noninteractive, None),
        (&mut v.widgets.inactive, Some(Color32::from_rgb(34, 38, 46))),
        (&mut v.widgets.hovered, Some(Color32::from_rgb(46, 51, 61))),
        (&mut v.widgets.active, Some(Color32::from_rgb(56, 62, 74))),
        (&mut v.widgets.open, Some(Color32::from_rgb(40, 44, 53))),
    ] {
        w.corner_radius = radius;
        if let Some(fill) = fill {
            w.weak_bg_fill = fill;
            w.bg_fill = fill;
            w.bg_stroke = Stroke::NONE;
        }
    }
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(40, 44, 53));
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(70, 77, 90));
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.8));
    v.extreme_bg_color = Color32::from_rgb(14, 16, 20);
    v.faint_bg_color = Color32::from_rgb(26, 29, 36);
    ctx.set_visuals(v);
    ctx.style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(6.0, 6.0);
        s.spacing.button_padding = egui::vec2(8.0, 4.0);
        s.spacing.window_margin = egui::Margin::same(12);
        s.spacing.slider_width = 130.0;
        s.spacing.interact_size.y = 22.0;
        s.animation_time = 0.12;
    });
}

fn swatch(m: Material) -> Color32 {
    let c = m.color();
    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

/// A material colour lifted enough to read as text on the dark panel.
fn swatch_text(m: Material) -> Color32 {
    swatch(m).lerp_to_gamma(Color32::WHITE, 0.25)
}

/// A small filled circle in line with text.
fn dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, color);
}

/// A row of equally sized toggle buttons, one per tool.
fn tool_grid(ui: &mut egui::Ui, state: &mut UiState, tools: &[Tool]) {
    const COLUMNS: usize = 4;
    let gap = ui.spacing().item_spacing.x;
    let width = (ui.available_width() - gap * (COLUMNS - 1) as f32) / COLUMNS as f32;
    for row in tools.chunks(COLUMNS) {
        ui.horizontal(|ui| {
            for &tool in row {
                let selected = state.tool == tool;
                let text = RichText::new(tool.name()).color(if selected { Color32::WHITE } else { ui.visuals().text_color() });
                let response = ui
                    .add_sized([width, 26.0], egui::Button::new(text).selected(selected))
                    .on_hover_text(format!("{}  ({})\n{}", tool.name(), tool.key(), tool.hint()));
                if response.clicked() {
                    state.tool = tool;
                }
            }
        });
    }
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title).small().strong().color(Color32::from_gray(150)));
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
        ui.add(egui::Slider::new(&mut p.cfl, 0.5..=6.0)).on_hover_text("Most cells a particle may move per substep.");
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

/// Builds the panel for this frame. It fades in and out with `show_panel`.
pub fn build(ctx: &egui::Context, state: &mut UiState, world: &mut World) -> UiActions {
    let mut a = UiActions::default();
    // Fades and slides in and out with H.
    let shown = ctx.animate_bool_with_time_and_easing(egui::Id::new("panel_shown"), state.show_panel, 0.16, easing::cubic_out);
    if shown == 0.0 {
        return a;
    }
    egui::Window::new("Sprout")
        .title_bar(false)
        .fixed_pos([14.0 - 18.0 * (1.0 - shown), 14.0])
        .min_width(292.0)
        .max_width(292.0)
        .auto_sized()
        .interactable(state.show_panel)
        .show(ctx, |ui| {
            ui.multiply_opacity(shown);
            // Scrolls only when the window is too short for the panel.
            let max = (ctx.screen_rect().height() - 52.0).max(120.0);
            egui::ScrollArea::vertical().max_height(max).auto_shrink([false, true]).show(ui, |ui| panel(ui, state, world, &mut a));
        });
    a
}

fn panel(ui: &mut egui::Ui, state: &mut UiState, world: &mut World, a: &mut UiActions) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Sprout").size(17.0).strong().color(Color32::from_gray(235)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(egui::Button::new(RichText::new("H  hide").small()).frame(false)).clicked() {
                state.show_panel = false;
            }
        });
    });
    // ---- transport ---------------------------------------------------------
    ui.horizontal(|ui| {
        let (label, tip) = if state.paused { ("▶  Resume", "Space") } else { ("⏸  Pause", "Space") };
        if ui.add(egui::Button::new(label).min_size(egui::vec2(88.0, 26.0))).on_hover_text(tip).clicked() {
            state.paused = !state.paused;
        }
        if ui.add(egui::Button::new("Step").min_size(egui::vec2(0.0, 26.0))).on_hover_text("N: advance one substep").clicked() {
            a.step = true;
        }
        if ui.add(egui::Button::new("↺  Reset").min_size(egui::vec2(0.0, 26.0))).on_hover_text("R: reload the map").clicked() {
            a.reset = true;
        }
    });
    ui.horizontal(|ui| {
        let mut selected = world.map;
        egui::ComboBox::from_id_salt("map")
            .width(ui.available_width())
            .height(420.0)
            .selected_text(RichText::new(MAPS[world.map].name).strong())
            .show_ui(ui, |ui| {
                for (i, m) in MAPS.iter().enumerate() {
                    ui.selectable_value(&mut selected, i, m.name).on_hover_text(m.blurb);
                }
            })
            .response
            .on_hover_text("PgUp / PgDn");
        if selected != world.map {
            a.load_map = Some(selected);
        }
    });
    ui.label(RichText::new(MAPS[world.map].blurb).weak().small());

    // ---- liquids and tools -------------------------------------------------
    section(ui, "LIQUID");
    ui.columns(3, |cols| {
        for (i, (ui, m)) in cols.iter_mut().zip(Material::ALL).enumerate() {
            let selected = state.material == m;
            let text = RichText::new(m.name()).color(swatch_text(m)).strong();
            let mut button = egui::Button::new(text).selected(selected).min_size(egui::vec2(ui.available_width(), 26.0));
            if selected {
                button = button.fill(swatch(m).gamma_multiply(0.28)).stroke(Stroke::new(1.0_f32, swatch_text(m)));
            }
            if ui.add(button).on_hover_text(format!("Shortcut: {}", i + 1)).clicked() {
                state.pick_material(m);
            }
        }
    });
    section(ui, "TOOLS");
    tool_grid(ui, state, &Tool::LIQUID);
    tool_grid(ui, state, &Tool::SOLID);
    ui.label(RichText::new(state.tool.hint()).small().weak());

    egui::Grid::new("brush").num_columns(2).show(ui, |ui| {
        ui.label("Brush");
        ui.add(egui::Slider::new(&mut state.brush, 2.0..=60.0).logarithmic(true).suffix(" cells"))
            .on_hover_text("[ and ], or Shift + wheel");
        ui.end_row();
        if state.tool.is_force() {
            ui.label("Strength");
            let mut percent = state.force_strength * 100.0;
            ui.add(egui::Slider::new(&mut percent, 10.0..=400.0).logarithmic(true).suffix(" %"))
                .on_hover_text("How hard Grab and Push move liquid. - and = adjust it.");
            state.force_strength = percent / 100.0;
            ui.end_row();
        } else if state.tool.uses_material() && state.tool != Tool::Drop {
            ui.label("Flow");
            let mut k = state.pour_rate / 1000.0;
            ui.add(egui::Slider::new(&mut k, 1.0..=200.0).logarithmic(true).suffix("k /s"))
                .on_hover_text("Particles per second. - and = adjust it.\nA small brush caps how much it can deliver.");
            state.pour_rate = k * 1000.0;
            ui.end_row();
        }
        if state.tool == Tool::Jet {
            ui.label("Aim");
            ui.add(egui::Slider::new(&mut state.jet_angle, -180.0..=180.0).suffix("°"))
                .on_hover_text("0° points down, 90° right. T / G or the wheel turn it by 15°.");
            ui.end_row();
        }
        if matches!(state.tool, Tool::Box | Tool::Ramp) {
            ui.label("Angle");
            ui.add(egui::Slider::new(&mut state.shape_angle, -90.0..=90.0).suffix("°"))
                .on_hover_text("T and G rotate");
            ui.end_row();
            ui.label("Aspect");
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
        .monospace()
        .small(),
    );
    if state.speed < 0.95 {
        ui.label(RichText::new(format!("running at {:.0} % speed to keep the frame rate", state.speed * 100.0)).small().color(WARN))
            .on_hover_text("The scene is too heavy to simulate in real time. A lower quality brings it back.");
    }
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{} particles", stats.live)).monospace().small());
        for (m, n) in [(Material::Water, stats.water().max(0.0)), (Material::Oil, stats.oil as f32), (Material::Syrup, stats.syrup)] {
            dot(ui, swatch(m));
            ui.label(RichText::new(format!("{n:.0}")).monospace().small().color(swatch_text(m)));
        }
    });

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
            ui.label("Quality");
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
                 Wheel         zoom (aims the jet)\n\
                 Alt+wheel     zoom with the jet\n\
                 Shift+wheel   brush size  ([ ])\n\
                 - / =         flow / strength\n\
                 1 2 3         water, oil, syrup\n\
                 Q B J C       pour, spray, jet, drop\n\
                 W V E         grab, push, erase\n\
                 A S D X       circle, box, ramp, remove\n\
                 T G F         rotate (jet, shape), flip\n\
                 Space N R     pause, step, reset\n\
                 PgUp PgDn     previous / next map\n\
                 Home H F11    reset view, panel, fullscreen",
            )
            .monospace()
            .small(),
        );
    });
}

// ---- canvas overlays ---------------------------------------------------------------------

/// Where a point of the grid lands in egui's coordinates (points, not pixels).
fn to_screen(ctx: &egui::Context, view: View, size: [u32; 2], p: [f32; 2]) -> Pos2 {
    let ppp = ctx.pixels_per_point();
    let px = view.world_to_screen(p, size);
    egui::pos2(px[0] / ppp, px[1] / ppp)
}

/// A ring that expands and fades where something happened on the canvas.
struct Ripple {
    pos: [f32; 2],
    radius: f32,
    color: Color32,
    start: f64,
}

/// A short message near the bottom of the screen.
struct Toast {
    text: String,
    detail: String,
    color: Color32,
    start: f64,
}

/// Animated canvas feedback. Times are egui's clock, s.
#[derive(Default)]
pub struct Overlay {
    ripples: Vec<Ripple>,
    toast: Option<Toast>,
    /// Brush radius and jet angle as drawn, easing towards the real ones.
    shown_brush: f32,
    shown_jet: f32,
}

const RIPPLE_TIME: f64 = 0.45;
const TOAST_TIME: f64 = 1.6;

impl Overlay {
    /// Rings out from `pos` (grid cells), e.g. where a solid was placed.
    pub fn ripple(&mut self, ctx: &egui::Context, pos: [f32; 2], radius: f32, color: Color32) {
        let start = ctx.input(|i| i.time);
        self.ripples.push(Ripple { pos, radius, color, start });
    }

    /// Ripple in a tool's accent colour.
    pub fn ripple_for(&mut self, ctx: &egui::Context, pos: [f32; 2], radius: f32, tool: Tool, material: Material) {
        self.ripple(ctx, pos, radius, tool.color(material));
    }

    pub fn toast(&mut self, ctx: &egui::Context, text: impl Into<String>, detail: impl Into<String>, color: Color32) {
        let start = ctx.input(|i| i.time);
        self.toast = Some(Toast { text: text.into(), detail: detail.into(), color, start });
    }

    pub fn toast_tool(&mut self, ctx: &egui::Context, state: &UiState) {
        let tool = state.tool;
        let text = if tool.uses_material() { format!("{}  ·  {}", tool.name(), state.material.name()) } else { tool.name().to_string() };
        self.toast(ctx, text, tool.key(), tool.color(state.material));
    }

    pub fn toast_map(&mut self, ctx: &egui::Context, map: usize) {
        self.toast(ctx, MAPS[map].name, MAPS[map].blurb, Color32::from_gray(235));
    }

    /// Draws everything for this frame: the brush cursor (when the pointer is on the
    /// canvas), the shape preview, ripples, the pause badge and the toast.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        ctx: &egui::Context,
        state: &UiState,
        view: View,
        size: [u32; 2],
        cursor: Option<[f32; 2]>,
        pressed: bool,
        grabbing: bool,
        shape: Option<(Shape, bool)>,
    ) {
        let now = ctx.input(|i| i.time);
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        let painter = ctx.layer_painter(egui::LayerId::background());
        let zoom = view.zoom / ctx.pixels_per_point();

        // Ease the cursor towards the brush size and jet aim (about 60 ms).
        let k = 1.0 - (-dt * 18.0).exp();
        if self.shown_brush == 0.0 {
            self.shown_brush = state.brush;
            self.shown_jet = state.jet_angle;
        }
        self.shown_brush += (state.brush - self.shown_brush) * k;
        let turn = (state.jet_angle - self.shown_jet + 540.0).rem_euclid(360.0) - 180.0;
        self.shown_jet += turn * k;

        if let Some((shape, remove)) = shape {
            outline(&painter, ctx, &shape, view, size, remove);
        }
        if let Some(pos) = cursor {
            self.cursor(ctx, &painter, state, to_screen(ctx, view, size, pos), self.shown_brush * zoom, pressed, grabbing);
        }

        self.ripples.retain(|r| now - r.start < RIPPLE_TIME);
        for r in &self.ripples {
            let t = ((now - r.start) / RIPPLE_TIME) as f32;
            let e = easing::cubic_out(t);
            let c = to_screen(ctx, view, size, r.pos);
            let radius = r.radius * zoom + 4.0 + 18.0 * e;
            painter.circle_stroke(c, radius, Stroke::new(2.0 * (1.0 - t) + 0.5, r.color.gamma_multiply((1.0 - t) * 0.9)));
        }

        let screen = ctx.screen_rect();
        let paused = ctx.animate_bool_with_time(egui::Id::new("paused_badge"), state.paused, 0.15);
        if paused > 0.0 {
            let rect = egui::Rect::from_center_size(egui::pos2(screen.center().x, screen.top() + 22.0 + 6.0 * (1.0 - paused)), egui::vec2(132.0, 28.0));
            painter.rect_filled(rect, CornerRadius::same(14), Color32::from_rgba_unmultiplied(20, 23, 29, (220.0 * paused) as u8));
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "⏸  Paused  ·  N steps", egui::FontId::proportional(13.0), Color32::from_gray(230).gamma_multiply(paused));
        }

        if let Some(toast) = &self.toast {
            let age = now - toast.start;
            if age > TOAST_TIME {
                self.toast = None;
            } else {
                // Rise and fade in over 0.15 s, fade out over the last 0.35 s.
                let appear = easing::cubic_out((age / 0.15).min(1.0) as f32);
                let alpha = appear * ((TOAST_TIME - age) / 0.35).min(1.0) as f32;
                let galley = ctx.fonts(|f| f.layout_no_wrap(toast.text.clone(), egui::FontId::proportional(15.0), Color32::WHITE));
                let detail = ctx.fonts(|f| f.layout(toast.detail.clone(), egui::FontId::proportional(12.0), Color32::WHITE, 420.0));
                let width = galley.size().x.max(detail.size().x) + 36.0;
                let height = galley.size().y + if toast.detail.is_empty() { 0.0 } else { detail.size().y + 4.0 } + 18.0;
                let centre = egui::pos2(screen.center().x, screen.bottom() - 48.0 - height * 0.5 + 10.0 * (1.0 - appear));
                let rect = egui::Rect::from_center_size(centre, egui::vec2(width, height));
                painter.rect_filled(rect, CornerRadius::same(12), Color32::from_rgba_unmultiplied(20, 23, 29, (232.0 * alpha) as u8));
                painter.rect_filled(
                    egui::Rect::from_min_size(rect.left_top() + egui::vec2(12.0, 9.0), egui::vec2(3.0, height - 18.0)),
                    CornerRadius::same(2),
                    toast.color.gamma_multiply(alpha),
                );
                let mut y = rect.top() + 9.0;
                painter.galley(egui::pos2(rect.left() + 22.0, y), galley.clone(), Color32::from_gray(240).gamma_multiply(alpha));
                y += galley.size().y + 4.0;
                if !toast.detail.is_empty() {
                    painter.galley(egui::pos2(rect.left() + 22.0, y), detail, Color32::from_gray(160).gamma_multiply(alpha));
                }
            }
        }
    }

    /// The brush footprint: a ring in the tool's colour with a light fill that firms up
    /// and tightens slightly while the button is held, plus a glyph for the tool.
    #[allow(clippy::too_many_arguments)]
    fn cursor(&self, ctx: &egui::Context, painter: &egui::Painter, state: &UiState, c: Pos2, r: f32, pressed: bool, grabbing: bool) {
        let tool = if grabbing { Tool::Grab } else { state.tool };
        if tool.places_shape() || tool == Tool::Remove {
            painter.circle_filled(c, 2.5, Color32::from_gray(230));
            return;
        }
        let color = tool.color(state.material);
        let held = ctx.animate_bool_with_time_and_easing(egui::Id::new("brush_held"), pressed, 0.12, easing::cubic_out);
        let r = r * (1.0 - 0.06 * held);
        painter.circle_filled(c, r, color.gamma_multiply(0.05 + 0.10 * held));
        painter.circle_stroke(c, r, Stroke::new(1.25 + 0.75 * held, color.gamma_multiply(0.75 + 0.25 * held)));
        painter.circle_filled(c, 2.0, color);
        let glyph = Stroke::new(1.5_f32, color.gamma_multiply(0.85));
        match tool {
            Tool::Jet => {
                let a = self.shown_jet.to_radians();
                let dir = egui::vec2(a.sin(), a.cos());
                painter.arrow(c, dir * (r + 14.0), glyph);
            }
            Tool::Push => {
                for i in 0..8 {
                    let d = Vec2::angled(i as f32 * std::f32::consts::TAU / 8.0);
                    painter.line_segment([c + d * r * 0.45, c + d * r * 0.8], glyph);
                }
            }
            Tool::Spray => {
                for i in 0..12 {
                    let d = Vec2::angled(i as f32 * std::f32::consts::TAU / 12.0);
                    painter.circle_filled(c + d * r * 0.85, 1.5, color);
                }
            }
            Tool::Erase => {
                let s = (r * 0.3).min(6.0);
                painter.line_segment([c + egui::vec2(-s, -s), c + egui::vec2(s, s)], glyph);
                painter.line_segment([c + egui::vec2(-s, s), c + egui::vec2(s, -s)], glyph);
            }
            _ => {}
        }
    }
}

/// Draws a solid on the canvas: a placement preview, or (with `remove`) the shape that a
/// click would remove.
fn outline(painter: &egui::Painter, ctx: &egui::Context, shape: &Shape, view: View, size: [u32; 2], remove: bool) {
    let color = if remove { DANGER } else { Color32::from_gray(225) };
    let stroke = Stroke::new(1.5_f32, color);
    let fill = color.gamma_multiply(if remove { 0.18 } else { 0.08 });
    let local = |c: [f32; 2], angle: f32, q: [f32; 2]| {
        let (s, co) = angle.sin_cos();
        to_screen(ctx, view, size, [c[0] + co * q[0] - s * q[1], c[1] + s * q[0] + co * q[1]])
    };
    let polygon = |pts: Vec<Pos2>| egui::Shape::convex_polygon(pts, fill, stroke);
    match *shape {
        Shape::Circle { c, r } => {
            painter.circle(to_screen(ctx, view, size, c), r * view.zoom / ctx.pixels_per_point(), fill, stroke);
        }
        Shape::Box { c, half, angle } => {
            let pts = [[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]];
            painter.add(polygon(pts.map(|q| local(c, angle, q)).to_vec()));
        }
        Shape::Ramp { c, half, angle, flip } => {
            let top = if flip { half[0] } else { -half[0] };
            let pts = [[-half[0], -half[1]], [half[0], -half[1]], [top, half[1]]];
            painter.add(polygon(pts.map(|q| local(c, angle, q)).to_vec()));
        }
    }
}
