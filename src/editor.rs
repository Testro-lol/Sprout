//! Choosing and making liquids (README, "Liquids"): the picker, the editor, and the
//! swatches both show.
//!
//! The workflow is choose a preset -> duplicate (or create) -> edit -> preview -> save ->
//! spawn, over three distinct states of a liquid, which the editor always names:
//! - the **draft**: what the editor holds; editing changes only this;
//! - the **live** row of the material table: what the tank uses. *Apply* copies the
//!   draft there, and every particle of that liquid in the tank follows at once;
//! - the **saved** definition in the library file. *Save* applies and writes the file.
//!
//! Built-ins are read-only: editing one makes a duplicate with a new id (and row), so the
//! liquid already in the tank keeps its material. Rows are never reused within a session
//! (a deleted liquid's row stays as it was until the tank is emptied), for the same reason.
//!
//! The operations are plain functions on `World` and `LiquidEditor`, so that `--selftest`
//! drives exactly what the buttons do.

use std::{collections::HashMap, sync::Arc};

use egui::{Color32, RichText};

use crate::{
    gpu::Gpu,
    headless::Offscreen,
    library::{CustomLiquid, custom_row, MAX_CUSTOM},
    render::View,
    scene::Scene,
    sim::{BUILTINS, FluidSim, MAT_PARAMS, MatProps, Material, Particle, SimParams, ToolState, pickable},
    world::World,
};

/// The editor's state: which liquid it shows, and its draft.
#[derive(Clone, Debug, Default)]
pub struct LiquidEditor {
    pub open: bool,
    pub picker_open: bool,
    /// Live table row being edited.
    pub row: usize,
    pub draft: MatProps,
    pub name: String,
    pub search: String,
    /// Last result to show (saved, applied, an error, load warnings).
    pub message: Option<(String, bool)>,
}

/// Where a liquid's draft stands against the live table and the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status {
    pub builtin: bool,
    /// The draft differs from the live row (Apply would change the tank).
    pub unapplied: bool,
    /// The live row (or the name) differs from the saved file.
    pub unsaved: bool,
    pub deleted: bool,
}

fn custom_index(row: usize) -> Option<usize> {
    row.checked_sub(BUILTINS.len())
}

impl LiquidEditor {
    /// Opens the editor on a table row: the draft starts as the live definition.
    pub fn select(&mut self, world: &World, row: usize) {
        self.row = row;
        self.draft = world.sim.params.mat[row];
        self.name = world.library.name(row);
        self.open = true;
    }

    pub fn status(&self, world: &World) -> Status {
        let live = world.sim.params.mat[self.row];
        let saved = custom_index(self.row).and_then(|k| world.library.custom.get(k));
        Status {
            builtin: self.row < BUILTINS.len(),
            unapplied: self.draft != live,
            unsaved: saved.is_some_and(|c| c.props != live || c.name != self.name || !world.saved_ids.contains(&c.id)),
            deleted: saved.is_some_and(|c| c.deleted),
        }
    }

    /// A new custom liquid from the live definition of `row` (a preset or another custom
    /// liquid), with a new id and its own row; the editor moves to it. None if the table is
    /// full.
    pub fn duplicate(&mut self, world: &mut World, row: usize) -> Option<usize> {
        if world.library.custom.len() >= MAX_CUSTOM {
            self.message = Some(("The liquid table is full: delete a custom liquid first.".into(), false));
            return None;
        }
        let base = if row < BUILTINS.len() {
            BUILTINS[row].key.to_string()
        } else {
            world.library.custom.get(row - BUILTINS.len()).map_or("water".to_string(), |c| c.base.clone())
        };
        let props = if row == self.row && self.open { self.draft } else { world.sim.params.mat[row] };
        let name = format!("{} copy", if row == self.row && self.open { self.name.clone() } else { world.library.name(row) });
        let id = world.library.new_id();
        world.library.custom.push(CustomLiquid { id, name, base, props, deleted: false });
        let new_row = custom_row(world.library.custom.len() - 1);
        world.library.install(&mut world.sim.params);
        self.select(world, new_row);
        self.message = Some((format!("New liquid \"{}\": a draft until you save it.", self.name), true));
        Some(new_row)
    }

    /// Draft -> live: every particle of this liquid in the tank follows. Built-ins are
    /// read-only (duplicate first).
    pub fn apply(&mut self, world: &mut World) -> bool {
        let Some(k) = custom_index(self.row) else {
            self.message = Some(("Built-in liquids are read-only: duplicate it to edit.".into(), false));
            return false;
        };
        world.sim.params.mat[self.row] = self.draft;
        if let Some(c) = world.library.custom.get_mut(k) {
            c.name = self.name.clone();
        }
        self.message = Some(("Applied: the liquid in the tank follows the draft.".into(), true));
        true
    }

    /// Applies, then writes the library file.
    pub fn save(&mut self, world: &mut World) -> bool {
        if !self.apply(world) {
            return false;
        }
        let k = self.row - BUILTINS.len();
        world.library.custom[k].props = self.draft;
        world.library.custom[k].name = self.name.clone();
        match world.library.save() {
            Ok(()) => {
                world.saved_ids = world.library.custom.iter().filter(|c| !c.deleted).map(|c| c.id.clone()).collect();
                self.message = Some((format!("Saved to {}", world.library.path.display()), true));
                true
            }
            Err(e) => {
                self.message = Some((e, false));
                false
            }
        }
    }

    /// Draft <- the saved definition (or, for a built-in, its preset values).
    pub fn revert(&mut self, world: &World) {
        self.draft = match custom_index(self.row).and_then(|k| world.library.custom.get(k)) {
            Some(c) => c.props,
            None => (BUILTINS[self.row].props)(),
        };
        self.message = Some(("Draft reset to the saved definition.".into(), true));
    }

    /// Marks a custom liquid deleted: gone from the picker and from the file at the next
    /// save; its row stays as it is for the liquid already in the tank.
    pub fn delete(&mut self, world: &mut World) -> bool {
        let Some(c) = custom_index(self.row).and_then(|k| world.library.custom.get_mut(k)) else {
            return false;
        };
        c.deleted = true;
        let id = c.id.clone();
        // Take it out of the file as it is on disk: other liquids' unsaved drafts stay unsaved.
        let r = world.library.remove_from_file(&id);
        world.saved_ids.retain(|s| *s != id);
        self.message = Some((r.map_or_else(|e| e, |_| "Deleted.".into()), true));
        self.open = false;
        true
    }
}

// ---- swatches ------------------------------------------------------------------------------

/// Swatches drawn by the real renderer: a small drop of each liquid in a tiny tank,
/// rendered off-screen and read back once whenever its look changes (never per frame).
pub struct Swatches {
    off: Offscreen,
    sim: FluidSim,
    cache: HashMap<usize, (MatProps, egui::TextureHandle)>,
    draft: Option<(MatProps, egui::TextureHandle)>,
    draft_time: f64,
}

const SWATCH_GRID: [u32; 2] = [36, 24];
const SWATCH_SIZE: [u32; 2] = [72, 48];

impl Swatches {
    pub fn new(gpu: Arc<Gpu>) -> Swatches {
        let mut sim = FluidSim::new(gpu.clone(), SWATCH_GRID, SWATCH_GRID[0] * SWATCH_GRID[1] * 4);
        sim.set_solids(Scene::new(SWATCH_GRID).rasterize());
        Swatches { off: Offscreen::new(gpu, SWATCH_SIZE), sim, cache: HashMap::new(), draft: None, draft_time: -1.0 }
    }

    fn render(&mut self, ctx: &egui::Context, params: &SimParams, row: usize, name: &str) -> egui::TextureHandle {
        self.sim.params = *params;
        self.sim.params.cell_size = 0.004;
        let mut ps = Vec::new();
        for j in 0..40 {
            for i in 0..48 {
                let p = [6.0 + i as f32 * 0.5, 2.3 + j as f32 * 0.5];
                let d = ((p[0] - 18.0) / 12.0).powi(2) + ((p[1] - 2.0) / 11.0).powi(2);
                if d < 1.0 {
                    ps.push(Particle::new(Material(row as u8), p, [0.0; 2]));
                }
            }
        }
        self.sim.reset(ps);
        // One real substep: the renderer colours the liquid from the grid (cell_optics.comp),
        // which only a substep fills; with none the drop draws as an uncoloured shape.
        self.sim.step_blocking(1, ToolState::default());
        self.off.renderer.invalidate();
        let pixels = self.off.capture(&self.sim, View::fit(SWATCH_GRID, SWATCH_SIZE));
        let image = egui::ColorImage::from_rgba_unmultiplied([SWATCH_SIZE[0] as usize, SWATCH_SIZE[1] as usize], &pixels);
        ctx.load_texture(format!("swatch-{name}"), image, egui::TextureOptions::LINEAR)
    }

    /// The swatch of a live row, rendered when first asked for or when the row changed.
    pub fn live(&mut self, ctx: &egui::Context, params: &SimParams, row: usize) -> egui::TextureId {
        if let Some((p, t)) = self.cache.get(&row)
            && *p == params.mat[row]
        {
            return t.id();
        }
        let t = self.render(ctx, params, row, &row.to_string());
        let id = t.id();
        self.cache.insert(row, (params.mat[row], t));
        id
    }

    /// The draft's swatch (the editor's preview), re-rendered at most four times a second.
    pub fn draft(&mut self, ctx: &egui::Context, params: &SimParams, row: usize, draft: &MatProps, now: f64) -> egui::TextureId {
        let stale = self.draft.as_ref().is_none_or(|(p, _)| p != draft);
        if stale && (self.draft.is_none() || now - self.draft_time > 0.25) {
            let mut p = *params;
            p.mat[row] = *draft;
            let t = self.render(ctx, &p, row, "draft");
            self.draft = Some((*draft, t));
            self.draft_time = now;
        }
        self.draft.as_ref().unwrap().1.id()
    }
}

// ---- the windows -------------------------------------------------------------------------

/// Rows the picker offers, by category, filtered by the search text.
pub fn picker_rows(world: &World, search: &str) -> Vec<(String, Vec<usize>)> {
    let needle = search.trim().to_lowercase();
    let matches = |row: usize| needle.is_empty() || world.library.name(row).to_lowercase().contains(&needle);
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    for (row, b) in BUILTINS.iter().enumerate() {
        if !pickable(row) || !matches(row) {
            continue;
        }
        match out.iter_mut().find(|(c, _)| c == b.category) {
            Some((_, rows)) => rows.push(row),
            None => out.push((b.category.to_string(), vec![row])),
        }
    }
    let mine: Vec<usize> = (0..world.library.custom.len()).filter(|&k| !world.library.custom[k].deleted).map(custom_row).filter(|&r| matches(r)).collect();
    if !mine.is_empty() {
        out.insert(0, ("My liquids".to_string(), mine));
    }
    out
}

/// Size of a picker tile's swatch, points.
const TILE: egui::Vec2 = egui::vec2(60.0, 30.0);
/// Tiles per picker row.
const TILE_COLUMNS: usize = 4;

/// One picker tile: the swatch (cropped to the drop) over the name. Click pours, the context
/// menu edits. Returns (clicked, edit asked for).
fn tile(ui: &mut egui::Ui, tex: egui::TextureId, name: &str, selected: bool) -> (bool, bool) {
    let label_h = 14.0;
    let (rect, r) = ui.allocate_exact_size(egui::vec2(TILE.x, TILE.y + label_h), egui::Sense::click());
    let img = egui::Rect::from_min_size(rect.min, TILE);
    let p = ui.painter();
    let v = ui.style().interact_selectable(&r, selected);
    p.rect_filled(img.expand(2.0), 5.0, v.weak_bg_fill);
    // The swatch is a small tank with the drop on its floor: show the drop, not the walls.
    let uv = egui::Rect::from_min_max(egui::pos2(0.1, 0.36), egui::pos2(0.9, 0.95));
    p.image(tex, img, uv, Color32::WHITE);
    if selected || r.hovered() {
        p.rect_stroke(img.expand(2.0), 5.0, v.bg_stroke, egui::StrokeKind::Outside);
    }
    let font = egui::FontId::proportional(11.0);
    let colour = if selected { ui.visuals().strong_text_color() } else { ui.visuals().text_color() };
    // One line, elided with "…" when the name is longer than the tile.
    let mut job = egui::text::LayoutJob::simple_singleline(name.to_string(), font, colour);
    job.wrap = egui::text::TextWrapping::truncate_at_width(TILE.x + 4.0);
    let galley = ui.fonts(|f| f.layout_job(job));
    let at = egui::pos2(rect.center().x - galley.size().x * 0.5, img.bottom() + 2.0);
    p.galley(at, galley, colour);
    let mut edit = false;
    let r = r.on_hover_text(format!("{name}: click to pour, right-click to edit"));
    r.context_menu(|ui| {
        if ui.button("Edit…").clicked() {
            edit = true;
            ui.close_menu();
        }
    });
    (r.clicked(), edit)
}

/// The picker window. `current`: the row being poured. Returns the row picked, if any.
pub fn picker(ctx: &egui::Context, ed: &mut LiquidEditor, sw: &mut Swatches, world: &mut World, current: usize) -> Option<usize> {
    if !ed.picker_open {
        return None;
    }
    let mut picked = None;
    let mut edit = None;
    let width = TILE_COLUMNS as f32 * (TILE.x + 6.0) - 6.0;
    egui::Window::new("Liquids")
        .title_bar(false)
        .default_pos([340.0, 14.0])
        .resizable(false)
        .min_width(width)
        .max_width(width)
        .frame(egui::Frame::window(&ctx.style()).inner_margin(10))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Liquids").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").on_hover_text("Close").clicked() {
                        ed.picker_open = false;
                    }
                    if ui.small_button("+ New").on_hover_text("A new liquid, starting from water").clicked() {
                        ed.duplicate(world, Material::Water.index());
                    }
                });
            });
            ui.add(egui::TextEdit::singleline(&mut ed.search).hint_text("Search").desired_width(f32::INFINITY));
            egui::ScrollArea::vertical().max_height(300.0).auto_shrink([false, true]).show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                let cats = picker_rows(world, &ed.search);
                if cats.is_empty() {
                    ui.label(RichText::new("No liquid matches.").weak().small());
                }
                for (category, rows) in cats {
                    ui.add_space(4.0);
                    ui.label(RichText::new(category.to_uppercase()).small().weak());
                    ui.add_space(1.0);
                    for chunk in rows.chunks(TILE_COLUMNS) {
                        ui.horizontal(|ui| {
                            for &row in chunk {
                                let tex = sw.live(ctx, &world.sim.params, row);
                                let (clicked, e) = tile(ui, tex, &world.library.name(row), row == current);
                                if clicked {
                                    picked = Some(row);
                                }
                                if e {
                                    edit = Some(row);
                                }
                            }
                        });
                    }
                }
            });
        });
    if let Some(row) = edit {
        if row < BUILTINS.len() {
            ed.duplicate(world, row);
        } else {
            ed.select(world, row);
        }
    }
    picked
}

fn slider(ui: &mut egui::Ui, scratch: &mut SimParams, row: usize, key: &str, label: &str) {
    let Some(p) = MAT_PARAMS.iter().find(|p| p.key == key) else { return };
    let mut v = (p.get)(scratch, row);
    ui.label(label);
    let r = if p.integer {
        ui.add(egui::Slider::new(&mut v, p.min..=p.max).step_by(1.0))
    } else {
        ui.add(egui::Slider::new(&mut v, p.min..=p.max).logarithmic(p.logarithmic()).suffix(format!(" {}", p.unit)))
    };
    r.on_hover_text(p.help);
    (p.put)(scratch, row, p.clamp(v));
    ui.end_row();
}

fn colour_row(ui: &mut egui::Ui, label: &str, c: &mut [f32; 3]) {
    ui.label(label);
    ui.color_edit_button_rgb(c);
    ui.end_row();
}

/// The editor window.
pub fn editor(ctx: &egui::Context, ed: &mut LiquidEditor, sw: &mut Swatches, world: &mut World) {
    if !ed.open {
        return;
    }
    let mut open = ed.open;
    let now = ctx.input(|i| i.time);
    let frame = egui::Frame::window(&ctx.style()).inner_margin(10);
    egui::Window::new("Liquid editor").title_bar(false).frame(frame).default_pos([630.0, 14.0]).default_width(320.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Liquid editor").strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("×").on_hover_text("Close").clicked() {
                    open = false;
                }
            });
        });
        let st = ed.status(world);
        ui.horizontal(|ui| {
            let tex = sw.draft(ctx, &world.sim.params, ed.row, &ed.draft, now);
            ui.add(egui::Image::new((tex, egui::vec2(108.0, 72.0)))).on_hover_text("Preview: the draft, drawn by the renderer");
            ui.vertical(|ui| {
                if st.builtin {
                    ui.label(RichText::new(&ed.name).strong());
                    ui.label(RichText::new("Built-in: read-only").small().color(Color32::from_rgb(240, 180, 90)));
                } else {
                    ui.text_edit_singleline(&mut ed.name);
                }
                // Which state is being changed, said plainly.
                let (text, colour) = match (st.unapplied, st.unsaved) {
                    (true, _) => ("Editing the DRAFT: not applied to the tank yet", Color32::from_rgb(240, 180, 90)),
                    (false, true) => ("LIVE in the tank, not saved to the file", Color32::from_rgb(110, 190, 255)),
                    (false, false) => ("Live and saved", Color32::from_rgb(120, 228, 160)),
                };
                ui.label(RichText::new(text).small().color(colour));
            });
        });
        let mut scratch = world.sim.params;
        scratch.mat[ed.row] = ed.draft;
        let row = ed.row;
        ui.add_enabled_ui(!st.builtin, |ui| {
            egui::Grid::new("common").num_columns(2).show(ui, |ui| {
                colour_row(ui, "Colour", &mut scratch.mat[row].color);
                slider(ui, &mut scratch, row, "viscosity", "Thickness");
                slider(ui, &mut scratch, row, "density", "Density");
                slider(ui, &mut scratch, row, "spawn_temp", "Temperature");
                slider(ui, &mut scratch, row, "opacity", "Opacity");
            });
            let groups: [(&str, &[&str]); 7] = [
                ("Flow (rheology)", &["rheology", "yield_stress", "flow_index", "shear_time", "visc_ratio", "temp_model", "temp_b", "temp_c", "temp_ref"]),
                ("Surface", &["tension", "contact_angle", "skin"]),
                ("Heat and freezing", &["heat_capacity", "conductivity", "melt_lo", "melt_hi", "latent", "density_solid", "emissivity", "boil", "latent_vap"]),
                ("Mixing and suspension", &["mix_rate", "grain", "load", "freeze_shift", "residue_share"]),
                ("Elastic and magnetic", &["shear_modulus", "relax_time", "susceptibility", "saturation"]),
                ("Look", &["scatter", "metal", "gloss", "body", "foam", "dry_time", "stain", "glint", "iridescence", "crackle"]),
                ("Phase", &[]),
            ];
            for (title, keys) in groups {
                egui::CollapsingHeader::new(title).id_salt(title).default_open(false).show(ui, |ui| {
                    egui::Grid::new(title).num_columns(2).show(ui, |ui| {
                        for k in keys {
                            slider(ui, &mut scratch, row, k, k);
                        }
                        if title == "Look" {
                            colour_row(ui, "Frozen colour", &mut scratch.mat[row].color_solid);
                        }
                        if title == "Phase" {
                            ui.label("Phase group");
                            let names = ["Aqueous", "Oil-like", "Liquid metal", "Silicate melt"];
                            let g = &mut scratch.mat[row].group;
                            egui::ComboBox::from_id_salt("group").selected_text(names[(*g as usize).min(3)]).show_ui(ui, |ui| {
                                for (i, n) in names.iter().enumerate() {
                                    ui.selectable_value(g, i as u32, *n);
                                }
                            });
                            ui.end_row();
                            ui.label("Dissolves in water");
                            let mut dil = scratch.mat[row].diluent == Material::Water.0 as i32;
                            if ui.checkbox(&mut dil, "").on_hover_text("A concentrate of water: blends with it and dilutes").changed() {
                                scratch.mat[row].diluent = if dil { Material::Water.0 as i32 } else { -1 };
                            }
                            ui.end_row();
                        }
                    });
                });
            }
        });
        ed.draft = scratch.mat[row];
        ui.separator();
        ui.horizontal(|ui| {
            if st.builtin {
                if ui.button("Duplicate to edit").clicked() {
                    ed.duplicate(world, row);
                }
            } else {
                if ui.add_enabled(st.unapplied, egui::Button::new("Apply")).on_hover_text("Draft -> the liquid in the tank").clicked() {
                    ed.apply(world);
                }
                if ui.button("Save").on_hover_text("Apply and write the liquids file").clicked() {
                    ed.save(world);
                }
                if ui.button("Revert").on_hover_text("Draft <- the saved definition").clicked() {
                    ed.revert(world);
                }
                if ui.button("Duplicate").clicked() {
                    ed.duplicate(world, row);
                }
                if ui.button(RichText::new("Delete").color(Color32::from_rgb(255, 110, 96))).clicked() {
                    ed.delete(world);
                }
            }
        });
        if let Some((m, ok)) = &ed.message {
            ui.label(RichText::new(m).small().color(if *ok { Color32::from_gray(190) } else { Color32::from_rgb(255, 110, 96) }));
        }
    });
    ed.open = ed.open && open;
}
