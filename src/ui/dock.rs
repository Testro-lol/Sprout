//! The tool dock at the bottom of the window: a row of groups, the tools of the open group
//! under it, and above both a strip with the options of the selected tool.

use egui::{Align2, Color32, CornerRadius, Id, Order, Rect, Sense, Ui, Vec2, emath::easing, pos2, vec2};

use super::{
    BRUSH, FLOW, Group, SOLID_COLD, SOLID_HOT, SolidKind, THICKNESS, Tool, UiState, icons, length_unit, swatch_of,
    theme::{self, fade, mix},
    widgets::{self, Scale, glide, glide_bool, spring},
};
use crate::{
    sim::{CURRENT_SPEED, CURRENT_STRENGTH, CURRENT_STRENGTH_DEFAULT, MAGNET_STRENGTH, MAGNET_STRENGTH_DEFAULT, PPC, Param, SimParams},
    world::World,
};

/// Height of the dock's content: the group row, a gap, the tool row.
const GROUP_H: f32 = 30.0;
const TILE_H: f32 = 56.0;
const GAP: f32 = 6.0;

/// Sizes that depend on how wide the window is.
struct Metrics {
    group_w: f32,
    tile_w: f32,
    /// Whether group names fit.
    names: bool,
}

fn metrics(ctx: &egui::Context) -> Metrics {
    let w = ctx.screen_rect().width();
    if w >= 760.0 {
        Metrics { group_w: 88.0, tile_w: 64.0, names: true }
    } else {
        // A narrow window: icons only for the groups, tighter tiles.
        let s = ((w - 40.0) / 6.0).clamp(40.0, 60.0);
        Metrics { group_w: s, tile_w: s, names: false }
    }
}

pub fn dock(ctx: &egui::Context, state: &mut UiState, world: &mut World, shown: f32) {
    let m = metrics(ctx);
    let slide = (1.0 - shown) * 150.0;
    let dock = egui::Area::new(Id::new("dock"))
        .order(Order::Middle)
        .anchor(Align2::CENTER_BOTTOM, vec2(0.0, -12.0 + slide))
        .interactable(state.show_panel)
        .show(ctx, |ui| {
            ui.multiply_opacity(shown);
            theme::panel().inner_margin(egui::Margin::same(8)).show(ui, |ui| tools(ui, state, &m));
        });
    let top = dock.response.rect.height() + 12.0 + 8.0;
    egui::Area::new(Id::new("options"))
        .order(Order::Middle)
        .anchor(Align2::CENTER_BOTTOM, vec2(0.0, -top + slide))
        .interactable(state.show_panel)
        .show(ctx, |ui| {
            ui.multiply_opacity(shown);
            theme::panel().corner_radius(CornerRadius::same(15)).inner_margin(egui::Margin::symmetric(12, 7)).show(ui, |ui| options(ui, state, world));
        });
}

// ---- groups and tools ----------------------------------------------------------------------

fn tools(ui: &mut Ui, state: &mut UiState, m: &Metrics) {
    let ctx = ui.ctx().clone();
    let now = ctx.input(|i| i.time);
    let width = m.group_w * Group::ALL.len() as f32;
    let (rect, _) = ui.allocate_exact_size(vec2(width, GROUP_H + GAP + TILE_H), Sense::hover());
    let p = ui.painter().clone();
    let group = state.tool.group();
    let accent = group.color();

    // ---- the group row: a leaf slides under the open group --------------------------------
    let row = Rect::from_min_size(rect.min, vec2(width, GROUP_H));
    p.rect_filled(row, CornerRadius::same(11), theme::RAISED);
    let x = spring(&ctx, Id::new("group_x"), group.index() as f32 * m.group_w);
    // The leaf takes the colour of the group it is moving to.
    let tint = {
        let c = [accent.r() as f32, accent.g() as f32, accent.b() as f32];
        let k = [Id::new("group_r"), Id::new("group_g"), Id::new("group_b")];
        Color32::from_rgb(glide(&ctx, k[0], c[0], 16.0) as u8, glide(&ctx, k[1], c[1], 16.0) as u8, glide(&ctx, k[2], c[2], 16.0) as u8)
    };
    p.rect_filled(Rect::from_min_size(row.min + vec2(x, 0.0), vec2(m.group_w, GROUP_H)).shrink(2.0), theme::leaf(11), tint);
    for g in Group::ALL {
        let cell = Rect::from_min_size(row.min + vec2(g.index() as f32 * m.group_w, 0.0), vec2(m.group_w, GROUP_H));
        let r = ui.interact(cell, Id::new(("group", g.index())), Sense::click());
        let open = g == group;
        let hover = glide_bool(&ctx, r.id.with("h"), r.hovered(), 20.0);
        let on = glide_bool(&ctx, r.id.with("on"), open, 18.0);
        let ink = mix(mix(theme::TEXT_WEAK, theme::TEXT, hover), theme::INK, on);
        if m.names {
            let galley = p.layout_no_wrap(g.name().to_string(), theme::display(12.5), ink);
            let total = 16.0 + 5.0 + galley.size().x;
            let left = cell.center().x - total * 0.5;
            icons::paint(&p, g.icon(), Rect::from_min_size(pos2(left, cell.center().y - 8.0), Vec2::splat(16.0)), ink);
            p.galley(pos2(left + 21.0, cell.center().y - galley.size().y * 0.5 + 0.5), galley, ink);
        } else {
            icons::paint(&p, g.icon(), Rect::from_center_size(cell.center(), Vec2::splat(18.0)), ink);
        }
        if r.clicked() {
            state.select_group(g, false);
        }
        r.on_hover_ui(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(g.name()).font(theme::display(13.0)).color(g.color()));
                key_cap(ui, g.key());
            });
            ui.label(egui::RichText::new(g.blurb()).color(theme::TEXT_WEAK));
            ui.label(egui::RichText::new(format!("{} again steps through its tools", g.key())).small().color(theme::TEXT_FAINT));
        });
    }

    // ---- the tool row -----------------------------------------------------------------------
    // When the group changes its tools rise into place one after the other.
    let (shown_group, since) = ctx.data_mut(|d| {
        let s = d.get_temp_mut_or(Id::new("dock_group"), (group.index(), now - 10.0));
        if s.0 != group.index() {
            *s = (group.index(), now);
        }
        *s
    });
    let _ = shown_group;
    let list = group.tools();
    let tiles_w = list.len() as f32 * m.tile_w;
    let origin = pos2(rect.center().x - tiles_w * 0.5, rect.top() + GROUP_H + GAP);
    let selected = list.iter().position(|t| *t == state.tool).unwrap_or(0);
    let sx = spring(&ctx, Id::new(("tile_x", group.index())), selected as f32 * m.tile_w);
    let enter0 = easing::cubic_out(((now - since) / 0.16).clamp(0.0, 1.0) as f32);
    p.rect_filled(Rect::from_min_size(origin + vec2(sx, 0.0), vec2(m.tile_w, TILE_H)).shrink(2.5), theme::leaf(16), fade(accent, enter0));
    for (k, &tool) in list.iter().enumerate() {
        let enter = easing::cubic_out(((now - since - 0.022 * k as f64) / 0.2).clamp(0.0, 1.0) as f32);
        let cell = Rect::from_min_size(origin + vec2(k as f32 * m.tile_w, 0.0), vec2(m.tile_w, TILE_H));
        let r = ui.interact(cell, Id::new(("tool", tool.name())), Sense::click());
        let is = tool == state.tool;
        let hover = glide_bool(&ctx, r.id.with("h"), r.hovered(), 22.0);
        let down = glide_bool(&ctx, r.id.with("d"), r.is_pointer_button_down_on(), 30.0);
        let on = glide_bool(&ctx, r.id.with("on"), is, 20.0);
        // A freshly picked tool's icon pops.
        let pop = spring(&ctx, r.id.with("pop"), 1.0);
        if r.clicked() {
            state.select(tool);
            widgets::kick(&ctx, r.id.with("pop"), 1.35);
        }
        let lift = vec2(0.0, (1.0 - enter) * 10.0 - 1.5 * hover * (1.0 - on) + down);
        if !is {
            p.rect_filled(cell.shrink(2.5).translate(lift), CornerRadius::same(12), fade(theme::TEXT, 0.07 * hover * enter));
        }
        let ink = fade(mix(mix(theme::TEXT_WEAK, theme::TEXT, hover), theme::INK, on), enter);
        let size = 23.0 * pop * (1.0 - 0.06 * down);
        icons::paint(&p, tool.icon(), Rect::from_center_size(cell.center() + vec2(0.0, -8.0) + lift, Vec2::splat(size)), ink);
        p.text(cell.center() + vec2(0.0, 16.0) + lift, Align2::CENTER_CENTER, tool.name(), theme::display(10.5), ink);
        r.on_hover_ui(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(tool.name()).font(theme::display(13.0)).color(accent));
                key_cap(ui, &tool.key_hint());
            });
            ui.label(egui::RichText::new(tool.hint()).color(theme::TEXT_WEAK));
        });
    }
}

/// A key, drawn as a key.
pub fn key_cap(ui: &mut Ui, key: &str) {
    let galley = ui.fonts(|f| f.layout_no_wrap(key.to_string(), theme::mono(10.5), theme::TEXT_WEAK));
    let (rect, _) = ui.allocate_exact_size(vec2(galley.size().x + 10.0, 17.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(5), theme::RAISED);
    ui.painter().rect_stroke(rect, CornerRadius::same(5), theme::stroke(1.0, fade(theme::TEXT, 0.14)), egui::StrokeKind::Inside);
    ui.painter().galley(rect.center() - galley.size() * 0.5, galley, theme::TEXT_WEAK);
}

// ---- the options of the selected tool ------------------------------------------------------

fn options(ui: &mut Ui, state: &mut UiState, world: &mut World) {
    let tool = state.tool;
    let liquid = swatch_of(&world.sim.params, state.material.index());
    let accent = tool.color(liquid);
    let dx = world.sim.params.cell_size;
    ui.spacing_mut().item_spacing = vec2(12.0, 4.0);
    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            // What is selected: the tool, and for liquid tools the liquid.
            let (rect, _) = ui.allocate_exact_size(vec2(22.0, 34.0), Sense::hover());
            icons::paint(ui.painter(), tool.icon(), Rect::from_center_size(rect.center(), Vec2::splat(20.0)), accent);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.add_space(2.0);
                ui.label(egui::RichText::new(tool.name()).font(theme::display(13.5)).color(theme::TEXT));
                let sub = if tool.uses_material() { world.library.name(state.material.index()) } else { tool.group().name().to_string() };
                ui.label(egui::RichText::new(sub).font(theme::body(11.0)).color(if tool.uses_material() { accent } else { theme::TEXT_FAINT }));
            });
            widgets::rule(ui);
            controls(ui, state, world, accent, dx);
        });
        ui.label(egui::RichText::new(tool.hint()).font(theme::body(11.0)).color(theme::TEXT_FAINT));
    });
}

/// Sliders over a parameter of the world (`PARAMS`), in its own unit.
fn world_slider(ui: &mut Ui, p: &mut SimParams, key: &str, label: &str, unit: &str, color: Color32, tip: &str) {
    let def = Param::find(key).expect("parameter listed in PARAMS");
    let mut v = def.read(p);
    let scale = if def.logarithmic() { Scale::log(def.min..=def.max) } else { Scale::lin(def.min..=def.max) }.reset_to(def.read(&SimParams::default()));
    let text = format!("{v:.0} {unit}");
    if widgets::option_slider(ui, label, &mut v, scale, text, color, tip).changed() {
        def.write(p, def.clamp(v));
    }
}

fn controls(ui: &mut Ui, state: &mut UiState, world: &mut World, accent: Color32, dx: f32) {
    let tool = state.tool;
    let (unit_m, unit) = length_unit(30.0 * dx);
    let k = dx / unit_m;
    let length = |cells: f32| format!("{:.1} {unit}", cells * k);

    // ---- size ------------------------------------------------------------------------------
    if tool.uses_thickness() || tool == Tool::Bowl {
        if tool == Tool::Bowl {
            let text = length(state.brush);
            widgets::option_slider(ui, "Size", &mut state.brush, Scale::log(BRUSH[0]..=BRUSH[1]).reset_to(9.0), text, accent, "Radius of the bowl; or drag it out on the canvas.");
        }
        let text = length(2.0 * state.thickness);
        widgets::option_slider(ui, "Thickness", &mut state.thickness, Scale::log(THICKNESS[0]..=THICKNESS[1]).reset_to(2.5), text, accent, "How thick the wall is. [ ] or Shift + wheel.");
    } else if tool.has_brush() || tool.places_shape() {
        let tip = format!("Radius: {:.0} cells. [ ] or Shift + wheel.", state.brush);
        let text = length(state.brush);
        widgets::option_slider(ui, "Size", &mut state.brush, Scale::log(BRUSH[0]..=BRUSH[1]).reset_to(9.0), text, accent, &tip);
    }

    // ---- how much ----------------------------------------------------------------------------
    if tool.has_flow() {
        // Particles per second -> area of liquid per second (2D: per metre of depth).
        let (kf, unit) = if dx < 3e-3 { (dx * dx / PPC as f32 * 1e4, "cm²/s") } else { (dx * dx / PPC as f32, "m²/s") };
        let text = format!("{:.2} {unit}", state.pour_rate * kf);
        let tip = format!("Area of liquid added per second ({:.0}k particles/s). - = or Alt + wheel.\nA small brush caps how much it can deliver.", state.pour_rate / 1000.0);
        widgets::option_slider(ui, "Flow", &mut state.pour_rate, Scale::log(FLOW[0]..=FLOW[1]).reset_to(12000.0), text, accent, &tip);
    } else if matches!(tool, Tool::Current | Tool::Pump) {
        let text = format!("{:.1} m/s", state.current_strength * CURRENT_SPEED * dx);
        let scale = Scale::log(CURRENT_STRENGTH[0]..=CURRENT_STRENGTH[1]).reset_to(CURRENT_STRENGTH_DEFAULT);
        widgets::option_slider(ui, "Speed", &mut state.current_strength, scale, text, accent, "Speed the current drives liquid towards. - = or Alt + wheel.");
    } else if tool == Tool::Magnet {
        let text = format!("{:.2} T", state.magnet_strength);
        let scale = Scale::log(MAGNET_STRENGTH[0]..=MAGNET_STRENGTH[1]).reset_to(MAGNET_STRENGTH_DEFAULT);
        widgets::option_slider(ui, "Field", &mut state.magnet_strength, scale, text, accent, "Field at the magnet's rim. A strong neodymium magnet: 0.3 - 0.5 T. Only magnetic liquids feel it.");
    } else if tool == Tool::Drain {
        let text = format!("{:.0} %", state.drain_strength * 100.0);
        widgets::option_slider(ui, "Rate", &mut state.drain_strength, Scale::log(0.1..=4.0).reset_to(1.0), text, accent, "How fast the drain takes the liquid that reaches it.");
    } else if tool.is_force() {
        let text = format!("{:.0} %", state.force_strength * 100.0);
        widgets::option_slider(ui, "Strength", &mut state.force_strength, Scale::log(0.1..=4.0).reset_to(1.0), text, accent, "How hard the tool moves liquid (also right-button grabbing). - = or Alt + wheel.");
    } else if matches!(tool, Tool::Heat | Tool::Cool) {
        world_slider(ui, &mut world.sim.params, "heat_power", "Power", "K/s", accent, "How fast the brush heats or cools water at its centre.");
    }

    // ---- which way ---------------------------------------------------------------------------
    match tool {
        Tool::Jet | Tool::Current | Tool::Faucet | Tool::Pump => {
            let angle = state.angle_mut().unwrap();
            widgets::aim_dial(ui, "Aim", angle, widgets::Dial::Down, accent, "Where it points. Ctrl + wheel or T / G turn it by 15°; placed tools can be dragged to aim.");
        }
        Tool::Magnet => {
            widgets::aim_dial(ui, "North", &mut state.magnet_angle, widgets::Dial::Up, accent, "Where the north pole points. Ctrl + wheel, T / G, or drag when placing.");
        }
        Tool::Box | Tool::Ramp => {
            let text = format!("{:.1} : 1", state.shape_aspect);
            widgets::option_slider(ui, "Shape", &mut state.shape_aspect, Scale::log(0.2..=12.0).reset_to(4.0), text, accent, "Width divided by height.");
            widgets::aim_dial(ui, "Turn", &mut state.shape_angle, widgets::Dial::Ccw, accent, "Ctrl + wheel or T / G turn it by 15°; or drag it out on the canvas.");
        }
        Tool::Bowl => {
            widgets::aim_dial(ui, "Tip", &mut state.shape_angle, widgets::Dial::Ccw, accent, "Tips the bowl; half a turn makes a dome. Ctrl + wheel or T / G.");
        }
        Tool::Whirl => {
            if let Some(k) = widgets::chips(ui, Id::new("whirl_dir"), &["↺", "↻"], state.whirl_clockwise as usize, accent) {
                state.whirl_clockwise = k == 1;
            }
        }
        _ => {}
    }
    if tool == Tool::Ramp
        && let Some(k) = widgets::chips(ui, Id::new("ramp_flip"), &["Tall left", "Tall right"], state.shape_flip as usize, accent)
    {
        state.shape_flip = k == 1;
    }

    // ---- what solids are made of -------------------------------------------------------------
    if tool.places_shape() {
        widgets::rule(ui);
        let kinds = [SolidKind::Plain, SolidKind::Hot, SolidKind::Cold];
        let chosen = kinds.iter().position(|s| *s == state.solid_kind).unwrap();
        let tint = match state.solid_kind {
            SolidKind::Plain => accent,
            SolidKind::Hot => theme::EMBER,
            SolidKind::Cold => theme::FROST,
        };
        if let Some(k) = widgets::chips(ui, Id::new("solid_kind"), &["Plain", "Hot", "Cold"], chosen, tint) {
            state.solid_kind = kinds[k];
        }
        match state.solid_kind {
            SolidKind::Plain => {}
            SolidKind::Hot => {
                let text = format!("{:.0} °C", state.solid_hot);
                widgets::option_slider(ui, "Held at", &mut state.solid_hot, Scale::log(SOLID_HOT[0]..=SOLID_HOT[1]).reset_to(300.0), text, tint, "A hot plate: it heats the liquid touching it and never cools.");
            }
            SolidKind::Cold => {
                let text = format!("{:.0} °C", state.solid_cold);
                widgets::option_slider(ui, "Held at", &mut state.solid_cold, Scale::lin(SOLID_COLD[0]..=SOLID_COLD[1]).reset_to(-30.0), text, tint, "A cold block: it chills (and freezes) the liquid touching it and never warms.");
            }
        }
    }
}
