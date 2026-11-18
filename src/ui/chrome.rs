//! Everything around the dock: the bar at the top left (map and time), the one at the top
//! right (how it is running, settings), the liquid palette down the left edge, and the
//! windows they open (maps, statistics, the settings drawer).

use egui::{Align2, Color32, CornerRadius, Id, Order, Rect, RichText, Sense, Ui, Vec2, emath::easing, pos2, vec2};

use super::{
    GRIDS, Group, MIN_FPS_RANGE, PALETTE, Page, SPEEDS, Tool, UiActions, UiState,
    dock::key_cap,
    icons::{self, Icon},
    length_unit, swatch_of,
    theme::{self, fade, mix},
    widgets::{self, glide_bool, spring},
};
use crate::{
    maps::MAPS,
    sim::{Group as ParamGroup, Material, Param, SimParams, SolverStats},
    world::World,
};

/// Distance of the bars from the window's edge.
const EDGE: f32 = 12.0;
/// Where the windows under the bars start.
const BELOW_BARS: f32 = 64.0;

// ---- top left: map and time ------------------------------------------------------------------

pub fn top_left(ctx: &egui::Context, state: &mut UiState, world: &mut World, a: &mut UiActions, shown: f32) {
    egui::Area::new(Id::new("top_left"))
        .order(Order::Middle)
        .anchor(Align2::LEFT_TOP, vec2(EDGE, EDGE - (1.0 - shown) * 80.0))
        .interactable(state.show_panel)
        .show(ctx, |ui| {
            ui.multiply_opacity(shown);
            theme::panel().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    let narrow = ctx.screen_rect().width() < 900.0;
                    logo(ui, !narrow);
                    widgets::rule(ui);
                    map_button(ui, state, world);
                    widgets::rule(ui);
                    let (icon, tip, on) = if state.paused { (Icon::Play, "Resume  (Space)", Some(theme::WARN)) } else { (Icon::Pause, "Pause  (Space)", None) };
                    if widgets::icon_button(ui, icon, tip, on).clicked() {
                        state.paused = !state.paused;
                    }
                    if widgets::icon_button(ui, Icon::Step, "Advance one substep  (N)", None).clicked() {
                        a.step = true;
                    }
                    speed_chip(ui, state);
                    if widgets::icon_button(ui, Icon::Reset, "Reload the map  (R)", None).clicked() {
                        a.reset = true;
                    }
                    if !narrow {
                        widgets::rule(ui);
                        ui.add_enabled_ui(world.can_undo(), |ui| {
                            if widgets::icon_button(ui, Icon::Undo, "Undo the last solid or device  (Ctrl + Z)", None).clicked() {
                                a.undo = true;
                            }
                        });
                        ui.add_enabled_ui(world.can_redo(), |ui| {
                            if widgets::icon_button(ui, Icon::Redo, "Redo  (Ctrl + Shift + Z)", None).clicked() {
                                a.redo = true;
                            }
                        });
                    }
                });
            });
        });
}

/// The sprout and the wordmark. The sprout unfolds when the app starts and sways after.
fn logo(ui: &mut Ui, wordmark: bool) {
    let time = ui.ctx().input(|i| i.time);
    let grow = easing::back_out((time / 0.9).clamp(0.0, 1.0) as f32);
    let (rect, _) = ui.allocate_exact_size(vec2(26.0, 30.0), Sense::hover());
    icons::sprout(ui.painter(), Rect::from_center_size(rect.center() + vec2(0.0, -1.0), Vec2::splat(26.0)), grow, time, theme::LEAF, theme::LEAF_DEEP);
    if wordmark {
        ui.add_space(2.0);
        ui.label(RichText::new("Sprout").font(theme::display(17.0)).color(theme::TEXT));
        ui.add_space(4.0);
    }
}

fn map_button(ui: &mut Ui, state: &mut UiState, world: &World) {
    let galley = ui.fonts(|f| f.layout_no_wrap(MAPS[world.map].name.to_string(), theme::display(13.0), theme::TEXT));
    let (rect, r) = ui.allocate_exact_size(vec2(galley.size().x + 36.0, 30.0), Sense::click());
    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered(), 20.0);
    let open = glide_bool(ui.ctx(), r.id.with("o"), state.maps_open, 18.0);
    let p = ui.painter();
    p.rect_filled(rect.shrink(1.0), theme::leaf(11), mix(fade(theme::TEXT, 0.05 + 0.06 * hover), fade(theme::LEAF, 0.22), open));
    p.galley_with_override_text_color(pos2(rect.left() + 11.0, rect.center().y - galley.size().y * 0.5 + 0.5), galley, mix(theme::TEXT, theme::LEAF, open));
    icons::paint(p, Icon::Chevron, Rect::from_center_size(pos2(rect.right() - 14.0, rect.center().y + 1.0 - 2.0 * open), Vec2::splat(13.0)), theme::TEXT_WEAK);
    if r.clicked() {
        state.maps_open = !state.maps_open;
    }
    r.on_hover_text("Choose a map  (PgUp / PgDn step through them)");
}

/// The time scale: click for the next speed, right-click for the one before.
fn speed_chip(ui: &mut Ui, state: &mut UiState) {
    let text = if state.time_scale >= 1.0 { format!("{:.0}×", state.time_scale) } else { format!("{:.2}×", state.time_scale).replace("0.", ".") };
    let (rect, r) = ui.allocate_exact_size(vec2(40.0, 30.0), Sense::click());
    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered(), 20.0);
    let off = state.time_scale != 1.0;
    let p = ui.painter();
    p.rect_filled(rect.shrink2(vec2(2.0, 4.0)), CornerRadius::same(9), if off { fade(theme::WATER, 0.22 + 0.1 * hover) } else { fade(theme::TEXT, 0.05 + 0.06 * hover) });
    p.text(rect.center(), Align2::CENTER_CENTER, text, theme::display(11.5), if off { theme::WATER } else { mix(theme::TEXT_WEAK, theme::TEXT, hover) });
    let at = SPEEDS.iter().position(|s| (*s - state.time_scale).abs() < 1e-3);
    let n = SPEEDS.len();
    if r.clicked() {
        state.time_scale = SPEEDS[at.map_or(2, |i| (i + 1) % n)];
    }
    if r.secondary_clicked() {
        state.time_scale = SPEEDS[at.map_or(2, |i| (i + n - 1) % n)];
    }
    r.on_hover_text("Time scale: simulated seconds per real second.\nClick for the next speed, right-click for the one before; Display has a slider.");
}

// ---- top right: how it runs, settings ------------------------------------------------------

pub fn top_right(ctx: &egui::Context, state: &mut UiState, world: &mut World, a: &mut UiActions, shown: f32) {
    egui::Area::new(Id::new("top_right"))
        .order(Order::Middle)
        .anchor(Align2::RIGHT_TOP, vec2(-EDGE, EDGE - (1.0 - shown) * 80.0))
        .interactable(state.show_panel)
        .show(ctx, |ui| {
            ui.multiply_opacity(shown);
            theme::panel().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    pulse(ui, state, world);
                    widgets::rule(ui);
                    if widgets::icon_button(ui, Icon::Wipe, "Clear the liquid; solids and devices stay  (Shift + R)", None).clicked() {
                        a.clear_liquid = true;
                    }
                    if widgets::icon_button(ui, Icon::Home, "Fit the tank to the window  (Home)", None).clicked() {
                        a.reset_view = true;
                    }
                    let settings = matches!(state.page, Some(p) if p != Page::Keys);
                    if widgets::icon_button(ui, Icon::Gear, "Settings: world, display, solver", settings.then_some(theme::LEAF)).clicked() {
                        state.page = if settings { None } else { Some(Page::World) };
                        state.stats_open = false;
                    }
                    let keys = state.page == Some(Page::Keys);
                    if widgets::icon_button(ui, Icon::Keys, "Controls  (F1)", keys.then_some(theme::LEAF)).clicked() {
                        state.page = if keys { None } else { Some(Page::Keys) };
                        state.stats_open = false;
                    }
                    if widgets::icon_button(ui, Icon::Hide, "Hide the interface  (H brings it back)", None).clicked() {
                        state.show_panel = false;
                    }
                });
            });
        });
}

/// The pulse of the simulation: the last two seconds of frame times as a line, the frame
/// rate and how much liquid there is. Click for the details.
fn pulse(ui: &mut Ui, state: &mut UiState, world: &World) {
    let (mean, _) = state.frame_times.summary();
    let fps = if mean > 0.0 { 1000.0 / mean } else { 0.0 };
    let slow = state.speed < 0.95 && !state.paused;
    let color = if slow { theme::WARN } else { theme::LEAF };
    let live = world.sim.stats.live as f32;
    let count = if live >= 1e6 { format!("{:.2}M", live / 1e6) } else if live >= 1e3 { format!("{:.0}k", live / 1e3) } else { format!("{live:.0}") };
    let (rect, r) = ui.allocate_exact_size(vec2(158.0, 30.0), Sense::click());
    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered(), 20.0);
    let open = glide_bool(ui.ctx(), r.id.with("o"), state.stats_open, 18.0);
    let p = ui.painter();
    p.rect_filled(rect.shrink(1.0), theme::leaf(11), mix(fade(theme::TEXT, 0.04 + 0.06 * hover), fade(color, 0.2), open));
    // The line: frame times against twice their mean.
    let graph = Rect::from_min_size(rect.min + vec2(9.0, 7.0), vec2(46.0, 16.0));
    let n = state.frame_times.samples().len();
    if n > 1 && mean > 0.0 {
        let top = 2.0 * mean * 1e-3;
        let pts: Vec<egui::Pos2> = state
            .frame_times
            .samples()
            .enumerate()
            .map(|(k, dt)| pos2(graph.left() + graph.width() * k as f32 / (n - 1) as f32, graph.bottom() - graph.height() * (dt / top).clamp(0.0, 1.0)))
            .collect();
        p.add(egui::Shape::line(pts, theme::stroke(1.3, color)));
    }
    p.text(pos2(rect.left() + 62.0, rect.center().y), Align2::LEFT_CENTER, format!("{fps:.0} fps"), theme::display(11.5), theme::TEXT);
    let text = if slow { format!("{:.0} %", state.speed * 100.0) } else { count };
    p.text(pos2(rect.right() - 9.0, rect.center().y), Align2::RIGHT_CENTER, text, theme::body(11.5), if slow { theme::WARN } else { theme::TEXT_WEAK });
    if r.clicked() {
        state.stats_open = !state.stats_open;
        if state.stats_open {
            state.page = None;
        }
    }
    r.on_hover_text(if slow {
        "The scene is too heavy to simulate in real time: it runs in slow motion to keep the frame rate. A lower quality brings it back.\nClick for the details."
    } else {
        "Frame rate and particles of liquid. Click for the details."
    });
}

pub fn stats(ctx: &egui::Context, state: &mut UiState, world: &World) {
    if !state.stats_open {
        return;
    }
    egui::Window::new("Statistics")
        .title_bar(false)
        .resizable(false)
        .anchor(Align2::RIGHT_TOP, vec2(-EDGE, BELOW_BARS))
        .min_width(250.0)
        .max_width(250.0)
        .show(ctx, |ui| {
            title(ui, "Pulse", &mut state.stats_open);
            let (mean, worst) = state.frame_times.summary();
            let stats = world.sim.stats;
            let solver = world.sim.solver;
            let row = |ui: &mut Ui, k: &str, v: String| {
                ui.label(RichText::new(k).color(theme::TEXT_WEAK));
                ui.label(RichText::new(v).monospace());
                ui.end_row();
            };
            egui::Grid::new("stats").num_columns(2).spacing(vec2(18.0, 4.0)).show(ui, |ui| {
                row(ui, "Frame", format!("{mean:.1} ms  (worst {worst:.1})"));
                row(ui, "Simulated", format!("{:.1} s", world.sim.sim_time()));
                row(ui, "Speed", format!("{:.0} % of real time", state.speed * state.time_scale * 100.0));
                row(ui, "Substeps", format!("{} per frame, {:.2} ms each", state.substeps, solver.dt * 1e3));
                row(ui, "Iterations", format!("pressure {}, viscosity {}", solver.pressure_iters_max, solver.viscosity_iters_max));
                row(ui, "Grid", format!("{} × {}", state.grid[0], state.grid[1]));
                row(ui, "Particles", format!("{}", stats.live));
            });
            let mut rows: Vec<usize> = (0..world.sim.params.mat_count).filter(|&r| stats.count[r] > 0).collect();
            if !rows.is_empty() {
                widgets::section(ui, "In the tank");
                rows.sort_by_key(|&r| std::cmp::Reverse(stats.count[r]));
                let total: f32 = rows.iter().map(|&r| stats.volume(Material(r as u8), &world.sim.params)).sum::<f32>().max(1.0);
                for &r in rows.iter().take(8) {
                    let share = stats.volume(Material(r as u8), &world.sim.params) / total;
                    let color = theme::lifted(swatch_of(&world.sim.params, r));
                    ui.horizontal(|ui| {
                        widgets::dot(ui, color);
                        ui.label(world.library.name(r));
                        // A bar for its share of the liquid.
                        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width() - 44.0, 6.0), Sense::hover());
                        ui.painter().rect_filled(rect, CornerRadius::same(3), theme::RAISED);
                        ui.painter().rect_filled(Rect::from_min_size(rect.min, vec2(rect.width() * share, 6.0)), CornerRadius::same(3), color);
                        ui.label(RichText::new(format!("{:.0} %", share * 100.0)).small().color(theme::TEXT_WEAK));
                    });
                }
            }
        });
}

/// The heading of a window, with its close button.
fn title(ui: &mut Ui, text: &str, open: &mut bool) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(text).font(theme::display(15.0)).color(theme::TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widgets::icon_button(ui, Icon::Close, "Close  (Esc)", None).clicked() {
                *open = false;
            }
        });
    });
}

// ---- maps ----------------------------------------------------------------------------------

pub fn maps(ctx: &egui::Context, state: &mut UiState, world: &World, a: &mut UiActions) {
    if !state.maps_open {
        return;
    }
    let height = (ctx.screen_rect().height() - BELOW_BARS - 190.0).clamp(160.0, 520.0);
    egui::Window::new("Maps")
        .title_bar(false)
        .resizable(false)
        .anchor(Align2::LEFT_TOP, vec2(EDGE + 56.0, BELOW_BARS))
        .min_width(310.0)
        .max_width(310.0)
        .default_height(height + 80.0)
        .show(ctx, |ui| {
            title(ui, "Maps", &mut state.maps_open);
            egui::ScrollArea::vertical().max_height(height).auto_shrink([false, true]).show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                for (i, m) in MAPS.iter().enumerate() {
                    let current = i == world.map;
                    let width = ui.available_width();
                    let name = ui.fonts(|f| f.layout_no_wrap(m.name.to_string(), theme::display(12.5), theme::TEXT));
                    let blurb = ui.fonts(|f| f.layout(m.blurb.to_string(), theme::body(11.5), theme::TEXT_WEAK, width - 30.0));
                    let h = name.size().y + blurb.size().y + 13.0;
                    let (rect, r) = ui.allocate_exact_size(vec2(width, h), Sense::click());
                    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered(), 22.0);
                    let p = ui.painter();
                    let fill = if current { fade(theme::LEAF, 0.16 + 0.05 * hover) } else { fade(theme::TEXT, 0.07 * hover) };
                    p.rect_filled(rect, theme::leaf(12), fill);
                    if current {
                        p.rect_filled(Rect::from_min_size(rect.min + vec2(5.0, 8.0), vec2(3.0, h - 16.0)), CornerRadius::same(2), theme::LEAF);
                    }
                    p.galley_with_override_text_color(rect.min + vec2(16.0, 5.0), name.clone(), if current { theme::LEAF } else { theme::TEXT });
                    p.galley(rect.min + vec2(16.0, 7.0 + name.size().y), blurb, theme::TEXT_WEAK);
                    if r.clicked() {
                        a.load_map = Some(i);
                        state.maps_open = false;
                    }
                }
            });
        });
}

// ---- the liquid palette --------------------------------------------------------------------

pub fn palette(ctx: &egui::Context, state: &mut UiState, world: &mut World, swatches: &mut crate::editor::Swatches, shown: f32) {
    // Between the top bar and the dock: smaller portholes in a short window.
    let room = ctx.screen_rect().height() - BELOW_BARS - super::DOCK_INSET;
    let size = ((room - 100.0) / PALETTE as f32 - 5.0).clamp(20.0, 38.0);
    let height = PALETTE as f32 * (size + 5.0) + 96.0;
    egui::Area::new(Id::new("palette"))
        .order(Order::Middle)
        .anchor(Align2::LEFT_TOP, vec2(EDGE - (1.0 - shown) * 90.0, BELOW_BARS + ((room - height) * 0.5).max(0.0)))
        .interactable(state.show_panel)
        .show(ctx, |ui| {
            ui.multiply_opacity(shown);
            theme::panel().inner_margin(egui::Margin::symmetric(7, 9)).show(ui, |ui| {
                ui.spacing_mut().item_spacing = vec2(0.0, 5.0);
                ui.vertical_centered(|ui| {
                    ui.set_width(size + 6.0);
                    let current = state.material.index();
                    // A liquid picked from the library that is not on the palette shows above it.
                    if !state.palette.contains(&current) {
                        swatch(ui, state, world, swatches, current, None, size);
                        let (rect, _) = ui.allocate_exact_size(vec2(size * 0.6, 3.0), Sense::hover());
                        ui.painter().hline(rect.x_range(), rect.center().y, theme::stroke(1.0, fade(theme::TEXT, 0.12)));
                    }
                    for slot in 0..PALETTE {
                        swatch(ui, state, world, swatches, state.palette[slot], Some(slot), size);
                    }
                    let (rect, _) = ui.allocate_exact_size(vec2(size * 0.6, 3.0), Sense::hover());
                    ui.painter().hline(rect.x_range(), rect.center().y, theme::stroke(1.0, fade(theme::TEXT, 0.12)));
                    let open = state.editor.picker_open;
                    if widgets::icon_button(ui, Icon::Library, "Every liquid, by kind; make your own  (L)", open.then_some(theme::LEAF)).clicked() {
                        state.editor.picker_open = !open;
                    }
                    if widgets::icon_button(ui, Icon::Pencil, "Edit the selected liquid (built-ins are duplicated first)", state.editor.open.then_some(theme::LEAF)).clicked() {
                        edit(state, world, current);
                    }
                });
            });
        });
}

fn edit(state: &mut UiState, world: &mut World, row: usize) {
    if row < crate::sim::BUILTINS.len() {
        state.editor.duplicate(world, row);
    } else {
        state.editor.select(world, row);
    }
}

/// One liquid of the palette: a porthole onto a drop of it, drawn by the renderer.
fn swatch(ui: &mut Ui, state: &mut UiState, world: &mut World, swatches: &mut crate::editor::Swatches, row: usize, slot: Option<usize>, size: f32) {
    let ctx = ui.ctx().clone();
    let (rect, r) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    let selected = state.material.index() == row;
    let pouring = state.tool.uses_material();
    let hover = glide_bool(&ctx, r.id.with("h"), r.hovered(), 22.0);
    let on = spring(&ctx, r.id.with("on"), selected as u8 as f32);
    let down = glide_bool(&ctx, r.id.with("d"), r.is_pointer_button_down_on(), 30.0);
    let lit = glide_bool(&ctx, r.id.with("lit"), selected || r.hovered() || pouring, 12.0);
    let radius = size * 0.5 * (0.80 + 0.06 * hover + 0.10 * on - 0.04 * down);
    let tex = swatches.live(&ctx, &world.sim.params, row);
    // The middle of the drop, with its surface in view (the swatch is 72 x 48: a square of it).
    let uv = Rect::from_min_max(pos2(0.34, 0.40), pos2(0.66, 0.88));
    let tint = Color32::from_gray((150.0 + 105.0 * lit) as u8);
    let body = Rect::from_center_size(rect.center(), Vec2::splat(2.0 * radius));
    egui::Image::new((tex, body.size())).uv(uv).corner_radius(CornerRadius::same(radius.round() as u8)).tint(tint).paint_at(ui, body);
    let color = theme::lifted(swatch_of(&world.sim.params, row));
    let p = ui.painter();
    p.circle_stroke(rect.center(), radius, theme::stroke(1.0, fade(theme::TEXT, 0.18)));
    if on > 0.01 {
        p.circle_stroke(rect.center(), radius + 2.5, theme::stroke(2.0, fade(color, on.clamp(0.0, 1.0))));
    }
    if let Some(slot) = slot {
        // The key that picks it.
        let at = rect.right_bottom() - vec2(5.0, 5.0);
        p.circle_filled(at, 6.5, theme::PANEL);
        p.text(at, Align2::CENTER_CENTER, format!("{}", slot + 1), theme::mono(9.0), mix(theme::TEXT_FAINT, theme::TEXT, on.clamp(0.0, 1.0)));
    }
    if r.clicked() {
        state.pick_material(Material(row as u8));
    }
    let name = world.library.name(row);
    let current = world.library.name(state.material.index());
    let r = r.on_hover_ui(|ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&name).font(theme::display(13.0)).color(color));
            if let Some(slot) = slot {
                key_cap(ui, &format!("{}", slot + 1));
            }
        });
        ui.label(RichText::new("Right-click to change this slot or edit the liquid").small().color(theme::TEXT_FAINT));
    });
    r.context_menu(|ui| {
        if let Some(slot) = slot
            && !selected
            && ui.button(format!("Put {current} here")).clicked()
        {
            state.palette[slot] = state.material.index();
            ui.close_menu();
        }
        if ui.button(format!("Edit {name}…")).clicked() {
            edit(state, world, row);
            ui.close_menu();
        }
    });
}

// ---- the settings drawer -------------------------------------------------------------------

pub fn drawer(ctx: &egui::Context, state: &mut UiState, world: &mut World, a: &mut UiActions) {
    let Some(page) = state.page else { return };
    let height = (ctx.screen_rect().height() - BELOW_BARS - 150.0).max(140.0);
    let mut open = true;
    egui::Window::new("Settings")
        .title_bar(false)
        .resizable(false)
        .anchor(Align2::RIGHT_TOP, vec2(-EDGE, BELOW_BARS))
        .min_width(316.0)
        .max_width(316.0)
        .default_height(height + 120.0)
        .show(ctx, |ui| {
            title(ui, if page == Page::Keys { "Controls" } else { "Settings" }, &mut open);
            if page != Page::Keys {
                let pages = [Page::World, Page::Display, Page::Solver];
                let chosen = pages.iter().position(|p| *p == page).unwrap_or(0);
                if let Some(k) = widgets::chips(ui, Id::new("pages"), &["World", "Display", "Solver"], chosen, theme::LEAF) {
                    state.page = Some(pages[k]);
                }
            }
            egui::ScrollArea::vertical().max_height(height).auto_shrink([false, true]).show(ui, |ui| match page {
                Page::World => physics(ui, world),
                Page::Display => display(ui, state, a),
                Page::Solver => {
                    let stats = world.sim.solver;
                    solver(ui, &mut world.sim.params, &stats);
                }
                Page::Keys => keys(ui),
            });
        });
    if !open {
        state.page = None;
    }
}

/// A grid row for parameter `key` of `PARAMS`: a slider over its safe range, shown as
/// `scale` times the parameter's own unit (in `unit`), plus an on/off box for a
/// stabilisation device that can be switched off. Writes only when the slider moves, so a
/// value is never rewritten by the round trip through the display unit.
fn param_row(ui: &mut Ui, p: &mut SimParams, key: &str, label: &str, scale: f32, unit: &str) -> egui::Response {
    let def = Param::find(key).expect("parameter listed in PARAMS");
    ui.label(label);
    let value = def.read(p);
    let mut on = def.off != Some(value);
    let response = ui
        .horizontal(|ui| {
            if let Some(off) = def.off
                && ui.checkbox(&mut on, "").on_hover_text("On / off").changed()
            {
                let d = def.read(&SimParams::default());
                def.write(p, if !on { off } else if def.off == Some(d) { def.min } else { d });
            }
            let mut shown = if on { value * scale } else { def.read(&SimParams::default()) * scale };
            let mut slider = egui::Slider::new(&mut shown, def.min * scale..=def.max * scale).logarithmic(def.logarithmic());
            if !unit.is_empty() {
                slider = slider.suffix(format!(" {unit}"));
            }
            if def.integer {
                slider = slider.integer();
            } else if def.logarithmic() && def.min * scale < 1e-2 {
                // Tolerances and viscosities span decades: show them in scientific notation.
                slider = slider.custom_formatter(|v, _| if v != 0.0 && v.abs() < 1e-2 { format!("{v:.1e}") } else { format!("{v:.3}") });
            }
            let r = ui.add_enabled(on, slider);
            if r.changed() {
                def.write(p, def.clamp(shown / scale));
            }
            r
        })
        .inner;
    ui.end_row();
    response.on_hover_text(def.help)
}

/// Player-facing physics: the world and what happens between liquids, in physical units.
fn physics(ui: &mut Ui, world: &mut World) {
    ui.spacing_mut().slider_width = 104.0;
    let grid = world.sim.grid;
    let p = &mut world.sim.params;
    widgets::section(ui, "World");
    egui::Grid::new("world").num_columns(2).show(ui, |ui| {
        param_row(ui, p, "gravity", "Gravity", 1.0, "m/s²");
        let (w, h) = (p.cell_size * grid[0] as f32, p.cell_size * grid[1] as f32);
        let (k, unit) = length_unit(w);
        param_row(ui, p, "cell_size", "Cell size", 1e3, "mm").on_hover_text(format!(
            "Physical size of one grid cell: the tank is {:.2} × {:.2} {unit}. Small cells make surface tension matter more.\n\
             Each map sets its own scale; reloading the map restores it.",
            w / k,
            h / k
        ));
        ui.label("");
        ui.label(RichText::new(format!("tank {:.2} × {:.2} {unit}", w / k, h / k)).small().color(theme::TEXT_FAINT));
        ui.end_row();
    });

    widgets::section(ui, "Heat");
    egui::Grid::new("heat").num_columns(2).show(ui, |ui| {
        param_row(ui, p, "ambient_temp", "Air", 1.0, "°C");
        param_row(ui, p, "heat_rate", "Heat rate", 1.0, "×");
        param_row(ui, p, "air_cooling", "Air cooling", 1.0, "W/m²K");
        param_row(ui, p, "contact_heat", "Liquid contact", 1.0, "W/m²K");
        param_row(ui, p, "eddy_heat", "Eddy mixing", 1.0, "");
        param_row(ui, p, "evaporation", "Evaporation", 1.0, "");
        param_row(ui, p, "heat_power", "Heat brush", 1.0, "K/s");
        param_row(ui, p, "rigid_viscosity", "Sets above", 1.0, "m²/s");
        param_row(ui, p, "steam", "Steam", 1.0, "");
    });
    widgets::section(ui, "Between liquids");
    egui::Grid::new("between").num_columns(2).show(ui, |ui| {
        param_row(ui, p, "tension_oil_water", "Oil–water", 1.0, "N/m");
        param_row(ui, p, "tension_metal_water", "Metal–water", 1.0, "N/m");
        param_row(ui, p, "tension_metal_oil", "Metal–oil", 1.0, "N/m");
        param_row(ui, p, "tension_melt_water", "Melt–water", 1.0, "N/m");
        param_row(ui, p, "tension_melt_oil", "Melt–oil", 1.0, "N/m");
        param_row(ui, p, "tension_melt_metal", "Melt–metal", 1.0, "N/m");
        let mm = p.droplet_cells * p.cell_size * 1e3;
        param_row(ui, p, "droplet_cells", "Droplet size", 1.0, "cells").on_hover_text(format!(
            "Size of the droplets, too small for the grid, that separate out where two immiscible liquids share a cell: \
             {mm:.2} mm at this cell size. Bigger droplets separate faster."
        ));
    });
    ui.add_space(4.0);
    ui.label(RichText::new("Each liquid's own properties are in the liquid editor (the pencil under the palette).").small().color(theme::TEXT_FAINT));
    ui.add_space(2.0);
    if widgets::button(ui, "Restore defaults", None).on_hover_text("World and liquids; the map's own scale; solver settings are kept").clicked() {
        let cell = p.cell_size;
        // The material table belongs to the liquids (and the editor), not to this page.
        let mat = p.mat;
        let count = p.mat_count;
        p.reset_groups(&[ParamGroup::World, ParamGroup::Liquid]);
        p.mat = mat;
        p.mat_count = count;
        p.cell_size = cell;
        world.apply_map_scale();
    }
}

/// Solver accuracy and cost, plus what the solver is doing. Everything else is `--set`.
fn solver(ui: &mut Ui, p: &mut SimParams, stats: &SolverStats) {
    ui.spacing_mut().slider_width = 104.0;
    ui.add_space(4.0);
    ui.label(RichText::new("Accuracy and cost, not the liquids. Every setting: sprout params").small().color(theme::TEXT_FAINT));
    egui::Grid::new("solver").num_columns(2).show(ui, |ui| {
        param_row(ui, p, "cfl", "CFL number", 1.0, "cells");
        param_row(ui, p, "speed_cap", "Speed cap", 1.0, "×")
            .on_hover_text("Fastest a particle may move, as a multiple of free fall over the tank height, √(2gH). Off: no cap.");
        param_row(ui, p, "pressure_tol", "Pressure tol.", 1.0, "");
        param_row(ui, p, "viscosity_tol", "Viscosity tol.", 1.0, "");
        ui.label("Substep");
        ui.label(format!("{:.2} ms (CFL {:.2} ms)", stats.dt * 1e3, stats.dt_cfl * 1e3));
        ui.end_row();
        ui.label("Iterations");
        ui.label(format!("pressure {}, viscosity {}", stats.pressure_iters_max, stats.viscosity_iters_max));
        ui.end_row();
    });
    let groups = [ParamGroup::Accuracy, ParamGroup::Stabilisation];
    ui.add_enabled_ui(!p.is_default(&groups), |ui| {
        if widgets::button(ui, "Restore solver defaults", None).clicked() {
            p.reset_groups(&groups);
        }
    });
}

fn display(ui: &mut Ui, state: &mut UiState, a: &mut UiActions) {
    ui.spacing_mut().slider_width = 104.0;
    widgets::section(ui, "Quality");
    let chosen = GRIDS.iter().position(|g| g.1 == state.grid);
    if let Some(k) = widgets::chips(ui, Id::new("quality"), &["Low", "Medium", "High", "Ultra"], chosen.unwrap_or(1), theme::LEAF) {
        a.grid = Some(GRIDS[k].1);
    }
    ui.label(RichText::new(format!("{} × {} cells. More cells: finer liquid, heavier frames.", state.grid[0], state.grid[1])).small().color(theme::TEXT_FAINT));
    widgets::section(ui, "Time");
    egui::Grid::new("time").num_columns(2).show(ui, |ui| {
        ui.label("Time scale");
        ui.add(egui::Slider::new(&mut state.time_scale, 0.1..=2.0).suffix(" ×"))
            .on_hover_text("Simulated seconds per real second, for slow motion. A scene too heavy for real time runs slower still.");
        ui.end_row();
        ui.label("Real time down to");
        ui.add(egui::Slider::new(&mut state.min_fps, MIN_FPS_RANGE).step_by(5.0).suffix(" fps")).on_hover_text(
            "How far a heavy scene may lower the frame rate before it goes into slow motion instead. \
             Lower keeps big scenes at real speed for longer but makes them choppier; set it to \
             the display's refresh rate to never give up frames.",
        );
        ui.end_row();
        ui.label("VSync");
        if ui.checkbox(&mut state.vsync, "").changed() {
            a.vsync_changed = true;
        }
        ui.end_row();
    });
    widgets::section(ui, "Look");
    egui::Grid::new("look").num_columns(2).show(ui, |ui| {
        ui.label("Wet edge");
        ui.add(egui::Slider::new(&mut state.look.wet_width, crate::render::Look::WET_WIDTH).step_by(0.1).suffix(" cells"))
            .on_hover_text("How far a liquid's wet film reaches into a solid it touches or has run off.");
        ui.end_row();
        ui.label("Wet tint");
        ui.add(egui::Slider::new(&mut state.look.wet_tint, 0.0..=1.0)).on_hover_text("How strongly wet solids take the liquid's colour. 0 draws solids dry.");
        ui.end_row();
        ui.label("Glow");
        ui.add(egui::Slider::new(&mut state.look.glow, 0.0..=2.0))
            .on_hover_text("The glow of hot liquid: bloom, and the light it throws on walls and other liquids. 0 turns it off.");
        ui.end_row();
        ui.label("Stains");
        ui.add(egui::Slider::new(&mut state.look.stains, 0.0..=1.0)).on_hover_text(
            "How strongly the stains liquids leave on walls and the back of the tank show (blood, ink, mud, scorch marks of lava). Other liquids slowly wash them off.",
        );
        ui.end_row();
    });
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        if widgets::button(ui, "Clean stains", None).on_hover_text("Wipe every stain off").clicked() {
            a.clean_stains = true;
        }
        if widgets::button(ui, "Fit view", None).on_hover_text("Home").clicked() {
            a.reset_view = true;
        }
    });
}

/// The controls, as keys.
fn keys(ui: &mut Ui) {
    let table = |ui: &mut Ui, id: &str, rows: &[(&[&str], &str)]| {
        egui::Grid::new(id).num_columns(2).min_col_width(96.0).spacing(vec2(12.0, 5.0)).show(ui, |ui| {
            for (keys, what) in rows {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for k in *keys {
                        key_cap(ui, k);
                    }
                });
                ui.label(RichText::new(*what).color(theme::TEXT_WEAK));
                ui.end_row();
            }
        });
    };
    widgets::section(ui, "Mouse");
    table(
        ui,
        "mouse",
        &[
            (&["Left"], "Use the selected tool"),
            (&["Right"], "Grab and stir, with any tool"),
            (&["Middle"], "Pan"),
            (&["Wheel"], "Zoom about the cursor"),
            (&["Shift", "Wheel"], "Brush size"),
            (&["Ctrl", "Wheel"], "Aim or turn"),
            (&["Alt", "Wheel"], "Flow or strength"),
            (&["Alt", "Left"], "Sample the liquid there"),
        ],
    );
    widgets::section(ui, "Tools");
    let groups: Vec<(String, String)> = Group::ALL
        .iter()
        .map(|g| (g.key().to_string(), format!("{}: {}", g.name(), g.tools().iter().map(|t| t.name()).collect::<Vec<_>>().join(", "))))
        .collect();
    egui::Grid::new("groups").num_columns(2).min_col_width(96.0).spacing(vec2(12.0, 5.0)).show(ui, |ui| {
        for (k, what) in &groups {
            key_cap(ui, k);
            ui.add(egui::Label::new(RichText::new(what).color(theme::TEXT_WEAK)).wrap());
            ui.end_row();
        }
    });
    ui.label(RichText::new("A group's key again steps through its tools (Shift: backwards). These go straight to a tool:").small().color(theme::TEXT_FAINT));
    egui::Grid::new("direct").num_columns(4).min_col_width(60.0).spacing(vec2(8.0, 5.0)).show(ui, |ui| {
        for (n, tool) in Tool::all().filter(|t| t.key().is_some()).enumerate() {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                key_cap(ui, tool.key().unwrap());
                ui.label(RichText::new(tool.name()).color(theme::TEXT_WEAK));
            });
            if n % 4 == 3 {
                ui.end_row();
            }
        }
    });
    table(
        ui,
        "toolkeys",
        &[
            (&["Tab"], "Back to the tool before"),
            (&["1", "…", "8"], "Liquids of the palette"),
            (&["L"], "The liquid library"),
            (&["[", "]"], "Brush size"),
            (&["-", "="], "Flow or strength"),
            (&["T", "G"], "Aim or turn by 15°"),
            (&["F"], "Flip the ramp, reverse the whirl"),
            (&["Shift"], "While dragging out: snap to 15°"),
            (&["Delete"], "Remove what is under the cursor"),
            (&["Ctrl", "Z"], "Undo a solid or device"),
            (&["Ctrl", "Shift", "Z"], "Redo"),
            (&["Esc"], "Cancel a drag, close a window"),
        ],
    );
    widgets::section(ui, "Time and view");
    table(
        ui,
        "time",
        &[
            (&["Space"], "Pause"),
            (&["N"], "One substep"),
            (&["R"], "Reload the map"),
            (&["Shift", "R"], "Clear the liquid"),
            (&["PgUp", "PgDn"], "Previous / next map"),
            (&["Home"], "Fit the view"),
            (&["H"], "Hide the interface"),
            (&["F1"], "This page"),
            (&["F11"], "Fullscreen"),
            (&["Ctrl", "Q"], "Quit"),
        ],
    );
}
