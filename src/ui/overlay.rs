//! What is drawn on the canvas itself: the brush cursor, previews of what a click would
//! place or remove, the devices in the world, ripples and toasts.
//!
//! Everything is drawn in screen space each frame from the state it shows; the only state
//! kept here is what animates: `Overlay` eases the cursor towards the current brush and
//! fades toasts and ripples in and out.

use std::f32::consts::{PI, TAU};

use egui::{Color32, CornerRadius, Painter, Pos2, Stroke, Vec2, emath::easing, pos2, vec2};

use super::{
    Placement, Tool, UiState, length_unit, swatch_of,
    theme::{self, fade},
};
use crate::{
    brush::RAIN_SPAN,
    maps::MAPS,
    render::View,
    scene::Shape,
    sim::{CURRENT_SPEED, Magnet},
    world::{Faucet, Pick, World},
};

/// What the overlay needs to know about the canvas this frame.
pub struct Canvas<'a> {
    pub view: View,
    /// Window size, pixels.
    pub size: [u32; 2],
    /// Where the pointer is (grid cells), when it is over the canvas.
    pub cursor: Option<[f32; 2]>,
    /// A button is held on the canvas.
    pub pressed: bool,
    /// The right button is grabbing.
    pub grabbing: bool,
    /// What the placing tool would place, and whether it is being dragged out.
    pub ghost: Option<(Placement, bool)>,
    /// What Remove, Move or Delete would act on (`true`: remove).
    pub target: Option<(Pick, bool)>,
    pub world: &'a World,
}

impl Canvas<'_> {
    /// Where a point of the grid lands in egui's coordinates (points, not pixels).
    fn to_screen(&self, ctx: &egui::Context, p: [f32; 2]) -> Pos2 {
        let ppp = ctx.pixels_per_point();
        let px = self.view.world_to_screen(p, self.size);
        pos2(px[0] / ppp, px[1] / ppp)
    }
}

/// A ring that expands and fades where something happened on the canvas.
struct Ripple {
    pos: [f32; 2],
    radius: f32,
    color: Color32,
    start: f64,
}

/// A short message under the top bars.
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
    /// Brush radius and aim as drawn, easing towards the real ones.
    shown_brush: f32,
    shown_aim: f32,
}

const RIPPLE_TIME: f64 = 0.5;
const TOAST_TIME: f64 = 1.7;

impl Overlay {
    /// Rings out from `pos` (grid cells), e.g. where a solid was placed.
    pub fn ripple(&mut self, ctx: &egui::Context, pos: [f32; 2], radius: f32, color: Color32) {
        let start = ctx.input(|i| i.time);
        self.ripples.push(Ripple { pos, radius, color, start });
    }

    pub fn toast(&mut self, ctx: &egui::Context, text: impl Into<String>, detail: impl Into<String>, color: Color32) {
        let start = ctx.input(|i| i.time);
        self.toast = Some(Toast { text: text.into(), detail: detail.into(), color, start });
    }

    pub fn toast_tool(&mut self, ctx: &egui::Context, state: &UiState, world: &World) {
        let tool = state.tool;
        let liquid = swatch_of(&world.sim.params, state.material.index());
        let text = if tool.uses_material() { format!("{}  ·  {}", tool.name(), world.library.name(state.material.index())) } else { tool.name().to_string() };
        self.toast(ctx, text, tool.hint(), tool.color(liquid));
    }

    pub fn toast_map(&mut self, ctx: &egui::Context, map: usize) {
        self.toast(ctx, MAPS[map].name, MAPS[map].blurb, theme::LEAF);
    }

    /// Draws everything for this frame.
    pub fn draw(&mut self, ctx: &egui::Context, state: &UiState, c: &Canvas) {
        let now = ctx.input(|i| i.time);
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        let painter = ctx.layer_painter(egui::LayerId::background());
        let zoom = c.view.zoom / ctx.pixels_per_point();
        let world = c.world;
        let liquid = swatch_of(&world.sim.params, state.material.index());

        // Ease the cursor towards the brush size and the aim (about 60 ms).
        let k = 1.0 - (-dt * 18.0).exp();
        let brush = if state.tool.uses_thickness() { state.thickness } else { state.brush };
        let aim = match state.tool {
            Tool::Jet => state.jet_angle,
            Tool::Current => state.current_angle,
            _ => 0.0,
        };
        if self.shown_brush == 0.0 {
            self.shown_brush = brush;
            self.shown_aim = aim;
        }
        self.shown_brush += (brush - self.shown_brush) * k;
        self.shown_aim += ((aim - self.shown_aim + 540.0).rem_euclid(360.0) - 180.0) * k;

        // ---- what is in the world -----------------------------------------------------------
        for (shape, temp) in world.scene.shapes.iter().zip(&world.scene.temps) {
            if let Some(t) = temp {
                // A solid held at a temperature glows at its edge: warm or cold.
                let color = if *t > world.sim.params.ambient_temp { theme::EMBER } else { theme::FROST };
                let pulse = 0.55 + 0.15 * (now * 1.6).sin() as f32;
                draw_shape(&painter, ctx, c, shape, theme::stroke(1.6, fade(color, pulse)), Color32::TRANSPARENT);
            }
        }
        // A line being drawn, ahead of the simulation taking it up.
        for shape in world.unrasterised() {
            draw_shape(&painter, ctx, c, shape, theme::stroke(1.2, fade(theme::CLAY, 0.8)), fade(theme::CLAY, 0.35));
        }
        for fan in world.fans() {
            draw_current(&painter, c.to_screen(ctx, fan.pos), fan.radius * zoom, fan.vel, theme::BRASS, now, 1.0);
        }
        if let Some(held) = world.held_current() {
            draw_current(&painter, c.to_screen(ctx, held.pos), held.radius * zoom, held.vel, theme::WIND, now, 1.0);
        }
        for d in world.drains() {
            draw_drain(&painter, c.to_screen(ctx, d.pos), d.radius * zoom, now, 1.0);
        }
        for f in world.faucets() {
            let color = theme::lifted(swatch_of(&world.sim.params, f.material.index()));
            draw_faucet(&painter, c.to_screen(ctx, f.pos), zoom, f, color, 1.0);
        }
        for m in world.sim.magnets() {
            draw_magnet(&painter, c.to_screen(ctx, m.pos), zoom, m, 1.0);
        }

        // ---- what a click would do ----------------------------------------------------------
        if let Some((pick, remove)) = c.target {
            let color = if remove { theme::DANGER } else { theme::WIND };
            match pick {
                Pick::Shape(i) => {
                    if let Some(shape) = world.scene.shapes.get(i) {
                        draw_shape(&painter, ctx, c, shape, theme::stroke(2.0, color), fade(color, 0.16));
                    }
                }
                _ => {
                    if let Some(pos) = world.pick_pos(pick) {
                        let r = match pick {
                            Pick::Magnet(i) => world.sim.magnets()[i].radius * 1.6,
                            Pick::Fan(i) => world.fans()[i].radius,
                            Pick::Drain(i) => world.drains()[i].radius,
                            Pick::Faucet(i) => world.faucets()[i].radius,
                            Pick::Shape(_) => 0.0,
                        };
                        painter.circle(c.to_screen(ctx, pos), (r * zoom).max(10.0) + 3.0, fade(color, 0.12), theme::stroke(2.0, color));
                    }
                }
            }
        }
        if let Some((placement, dragging)) = &c.ghost {
            let strength = if *dragging { 1.0 } else { 0.6 };
            match placement {
                Placement::Solid(shape) => {
                    let color = match state.solid_kind {
                        super::SolidKind::Plain => theme::CLAY,
                        super::SolidKind::Hot => theme::EMBER,
                        super::SolidKind::Cold => theme::FROST,
                    };
                    draw_shape(&painter, ctx, c, shape, theme::stroke(1.6, fade(color, 0.5 + 0.5 * strength)), fade(color, 0.07 + 0.09 * strength));
                    if *dragging {
                        let at = c.cursor.map(|p| c.to_screen(ctx, p)).unwrap_or_default();
                        label(&painter, ctx, at + vec2(16.0, -18.0), &measure(shape, world.sim.params.cell_size), color);
                    }
                }
                Placement::Faucet(f) => draw_faucet(&painter, c.to_screen(ctx, f.pos), zoom, f, theme::lifted(liquid), strength),
                Placement::Pump(p) => draw_current(&painter, c.to_screen(ctx, p.pos), p.radius * zoom, p.vel, theme::BRASS, now, strength),
                Placement::Drain(d) => draw_drain(&painter, c.to_screen(ctx, d.pos), d.radius * zoom, now, strength),
                Placement::Magnet(m) => draw_magnet(&painter, c.to_screen(ctx, m.pos), zoom, m, strength),
            }
        }
        if let Some(pos) = c.cursor {
            self.cursor(ctx, &painter, state, c.to_screen(ctx, pos), zoom, c.pressed, c.grabbing, liquid, now);
        }

        self.ripples.retain(|r| now - r.start < RIPPLE_TIME);
        for r in &self.ripples {
            let t = ((now - r.start) / RIPPLE_TIME) as f32;
            let e = easing::cubic_out(t);
            let centre = c.to_screen(ctx, r.pos);
            let radius = r.radius * zoom + 4.0 + 22.0 * e;
            painter.circle_stroke(centre, radius, theme::stroke(2.5 * (1.0 - t) + 0.5, r.color.gamma_multiply((1.0 - t) * 0.9)));
            painter.circle_stroke(centre, radius * 0.72, theme::stroke(1.2 * (1.0 - t), r.color.gamma_multiply((1.0 - t) * 0.45)));
        }

        // ---- paused: an amber edge around the tank, and a badge ---------------------------------
        let screen = ctx.screen_rect();
        let paused = ctx.animate_bool_with_time(egui::Id::new("paused_badge"), state.paused, 0.18);
        if paused > 0.0 {
            let breathe = 0.75 + 0.25 * (now * 2.2).sin() as f32;
            painter.rect_stroke(screen.shrink(1.5), CornerRadius::ZERO, theme::stroke(3.0, fade(theme::WARN, 0.55 * paused * breathe)), egui::StrokeKind::Inside);
            if screen.width() > 1150.0 || !state.show_panel {
                let rect = egui::Rect::from_center_size(pos2(screen.center().x, screen.top() + 34.0 - 8.0 * (1.0 - paused)), vec2(150.0, 28.0));
                painter.rect_filled(rect, theme::leaf(14), fade(theme::PANEL, paused));
                painter.rect_stroke(rect, theme::leaf(14), theme::stroke(1.0, fade(theme::WARN, 0.6 * paused)), egui::StrokeKind::Inside);
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Paused  ·  N steps", theme::display(12.0), fade(theme::WARN, paused));
            }
        }

        if let Some(toast) = &self.toast {
            let age = now - toast.start;
            if age > TOAST_TIME {
                self.toast = None;
            } else {
                // Drops in over 0.18 s, fades out over the last 0.35 s.
                let appear = easing::back_out((age / 0.18).min(1.0) as f32);
                let alpha = (age / 0.12).min(1.0) as f32 * ((TOAST_TIME - age) / 0.35).min(1.0) as f32;
                let galley = ctx.fonts(|f| f.layout_no_wrap(toast.text.clone(), theme::display(14.0), Color32::WHITE));
                let detail = ctx.fonts(|f| f.layout(toast.detail.clone(), theme::body(12.0), Color32::WHITE, 440.0));
                let width = galley.size().x.max(detail.size().x) + 40.0;
                let height = galley.size().y + if toast.detail.is_empty() { 0.0 } else { detail.size().y + 4.0 } + 18.0;
                let centre = pos2(screen.center().x, screen.top() + 78.0 + height * 0.5 - 12.0 * (1.0 - appear));
                let rect = egui::Rect::from_center_size(centre, vec2(width, height));
                painter.rect_filled(rect.translate(vec2(0.0, 4.0)).expand(2.0), theme::leaf(16), Color32::from_black_alpha((70.0 * alpha) as u8));
                painter.rect_filled(rect, theme::leaf(16), fade(theme::PANEL, alpha));
                painter.rect_stroke(rect, theme::leaf(16), theme::stroke(1.0, fade(toast.color, 0.45 * alpha)), egui::StrokeKind::Inside);
                painter.circle_filled(pos2(rect.left() + 15.0, rect.top() + 9.0 + galley.size().y * 0.5), 4.0, fade(toast.color, alpha));
                let mut y = rect.top() + 9.0;
                painter.galley_with_override_text_color(pos2(rect.left() + 26.0, y), galley.clone(), fade(theme::TEXT, alpha));
                y += galley.size().y + 4.0;
                if !toast.detail.is_empty() {
                    painter.galley_with_override_text_color(pos2(rect.left() + 26.0, y), detail, fade(theme::TEXT_WEAK, alpha));
                }
            }
        }
    }

    /// The cursor: for brush tools the brush's footprint, a ring in the tool's colour with a
    /// light fill that firms up and tightens slightly while the button is held, plus a
    /// glyph for what the tool does; for the others a small mark.
    #[allow(clippy::too_many_arguments)]
    fn cursor(&self, ctx: &egui::Context, painter: &Painter, state: &UiState, c: Pos2, zoom: f32, pressed: bool, grabbing: bool, liquid: Color32, now: f64) {
        let tool = if grabbing { Tool::Grab } else { state.tool };
        let color = tool.color(liquid);
        let held = ctx.animate_bool_with_time_and_easing(egui::Id::new("brush_held"), pressed, 0.12, easing::cubic_out);
        let glyph = theme::stroke(1.5, color.gamma_multiply(0.85));
        let t = now as f32;
        match tool {
            Tool::Circle | Tool::Box | Tool::Ramp | Tool::Bowl | Tool::Wall | Tool::Faucet | Tool::Drain | Tool::Pump | Tool::Magnet => {
                painter.circle_filled(c, 2.5, color);
                return;
            }
            Tool::Draw => {
                let r = (self.shown_brush * zoom).max(2.0);
                painter.circle(c, r, fade(color, 0.25 + 0.3 * held), theme::stroke(1.5, color));
                return;
            }
            Tool::Remove | Tool::Move => {
                let s = 5.0;
                if tool == Tool::Remove {
                    painter.line_segment([c + vec2(-s, -s), c + vec2(s, s)], glyph);
                    painter.line_segment([c + vec2(-s, s), c + vec2(s, -s)], glyph);
                } else {
                    for k in 0..4 {
                        let d = Vec2::angled(k as f32 * TAU / 4.0);
                        painter.arrow(c + d * 3.0, d * (7.0 + 2.0 * held), glyph);
                    }
                }
                return;
            }
            Tool::Pipette => {
                painter.circle_stroke(c, 7.0 - held, theme::stroke(1.5, color));
                for k in 0..4 {
                    let d = Vec2::angled(k as f32 * TAU / 4.0);
                    painter.line_segment([c + d * 9.0, c + d * 13.0], glyph);
                }
                painter.circle_filled(c, 1.5, color);
                return;
            }
            _ => {}
        }
        let r = self.shown_brush * zoom * (1.0 - 0.06 * held);
        painter.circle_filled(c, r, color.gamma_multiply(0.05 + 0.10 * held));
        painter.circle_stroke(c, r, theme::stroke(1.25 + 0.75 * held, color.gamma_multiply(0.75 + 0.25 * held)));
        painter.circle_filled(c, 2.0, color);
        match tool {
            Tool::Jet => {
                let a = self.shown_aim.to_radians();
                painter.arrow(c, vec2(a.sin(), a.cos()) * (r + 14.0), glyph);
            }
            Tool::Current => {
                let a = self.shown_aim.to_radians();
                let dir = vec2(a.sin(), a.cos());
                painter.arrow(c - dir * r * 0.6, dir * r * 1.2, glyph);
            }
            Tool::Push | Tool::Pull => {
                // Ticks that drift outwards (push) or inwards (pull).
                let phase = (t * 1.6).fract();
                for i in 0..8 {
                    let d = Vec2::angled(i as f32 * TAU / 8.0);
                    let f = if tool == Tool::Push { 0.35 + 0.45 * phase } else { 0.8 - 0.45 * phase };
                    let fade_k = (PI * phase).sin();
                    painter.line_segment([c + d * r * f, c + d * r * (f + 0.14)], theme::stroke(1.5, color.gamma_multiply(0.3 + 0.6 * fade_k)));
                }
            }
            Tool::Whirl => {
                // Three arcs going round the way the liquid will.
                let sign = if state.whirl_clockwise { 1.0 } else { -1.0 };
                let turn = t * (1.6 + 2.4 * held) * sign;
                for i in 0..3 {
                    let a0 = turn + i as f32 * TAU / 3.0;
                    let pts: Vec<Pos2> = (0..=10).map(|k| c + Vec2::angled(a0 + sign * k as f32 * 0.11) * r * 0.62).collect();
                    let tip = *pts.last().unwrap();
                    let dir = Vec2::angled(a0 + sign * 1.1 + sign * PI / 2.0);
                    painter.add(egui::Shape::line(pts, glyph));
                    painter.line_segment([tip, tip - Vec2::angled(dir.angle() + 0.6) * 5.0], glyph);
                    painter.line_segment([tip, tip - Vec2::angled(dir.angle() - 0.6) * 5.0], glyph);
                }
            }
            Tool::Hold => {
                let s = (r * 0.5).clamp(5.0, 16.0);
                for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let corner = c + vec2(sx * s, sy * s);
                    painter.line_segment([corner, corner - vec2(sx * s * 0.5, 0.0)], glyph);
                    painter.line_segment([corner, corner - vec2(0.0, sy * s * 0.5)], glyph);
                }
            }
            Tool::Spray => {
                for i in 0..12 {
                    let d = Vec2::angled(i as f32 * TAU / 12.0);
                    painter.circle_filled(c + d * r * 0.85, 1.5, color);
                }
            }
            Tool::Rain => {
                // The cloud: a bar three brushes to either side, with drops leaving it.
                let half = r * RAIN_SPAN;
                painter.line_segment([c - vec2(half, 0.0), c + vec2(half, 0.0)], theme::stroke(2.0, color.gamma_multiply(0.7)));
                let n = ((half / 14.0) as usize).clamp(4, 24);
                for i in 0..n {
                    let x = -half + 2.0 * half * (i as f32 + 0.5) / n as f32;
                    let phase = (t * 1.8 + i as f32 * 0.37).fract();
                    let y = 4.0 + 16.0 * phase;
                    painter.line_segment([c + vec2(x, y), c + vec2(x, y + 5.0)], theme::stroke(1.3, color.gamma_multiply(0.8 * (1.0 - phase))));
                }
            }
            Tool::Erase => {
                let s = (r * 0.3).min(6.0);
                painter.line_segment([c + vec2(-s, -s), c + vec2(s, s)], glyph);
                painter.line_segment([c + vec2(-s, s), c + vec2(s, -s)], glyph);
            }
            Tool::Heat => {
                // Rising wisps.
                for i in -1..=1 {
                    let x = i as f32 * r * 0.35;
                    let pts: Vec<Pos2> = (0..=8).map(|k| k as f32 / 8.0).map(|u| c + vec2(x + 2.5 * (u * TAU + t * 3.0 + i as f32).sin(), r * (0.3 - 0.6 * u))).collect();
                    painter.add(egui::Shape::line(pts, theme::stroke(1.3, color.gamma_multiply(0.7))));
                }
            }
            Tool::Cool => {
                let s = (r * 0.4).clamp(5.0, 14.0);
                for i in 0..3 {
                    let d = Vec2::angled(i as f32 * PI / 3.0 + PI / 2.0);
                    painter.line_segment([c - d * s, c + d * s], glyph);
                }
            }
            _ => {}
        }
    }
}

/// A small label on the canvas (the size of what is being dragged out).
fn label(painter: &Painter, ctx: &egui::Context, at: Pos2, text: &str, color: Color32) {
    let galley = ctx.fonts(|f| f.layout_no_wrap(text.to_string(), theme::body(11.5), theme::TEXT));
    let rect = egui::Rect::from_min_size(at, galley.size() + vec2(14.0, 8.0));
    painter.rect_filled(rect, theme::leaf(9), theme::PANEL);
    painter.rect_stroke(rect, theme::leaf(9), theme::stroke(1.0, fade(color, 0.5)), egui::StrokeKind::Inside);
    painter.galley(rect.min + vec2(7.0, 4.0), galley, theme::TEXT);
}

/// The size of a solid in physical units, for the label.
fn measure(shape: &Shape, dx: f32) -> String {
    let (unit_m, unit) = length_unit(30.0 * dx);
    let k = dx / unit_m;
    match *shape {
        Shape::Circle { r, .. } => format!("⌀ {:.1} {unit}", 2.0 * r * k),
        Shape::Box { half, angle, .. } | Shape::Ramp { half, angle, .. } => format!("{:.1} × {:.1} {unit}  ·  {:.0}°", 2.0 * half[0] * k, 2.0 * half[1] * k, angle.to_degrees()),
        Shape::Capsule { a, b, .. } => {
            let d = [b[0] - a[0], b[1] - a[1]];
            format!("{:.1} {unit}  ·  {:.0}°", d[0].hypot(d[1]) * k, d[1].atan2(d[0]).to_degrees())
        }
        Shape::Bowl { r, .. } => format!("⌀ {:.1} {unit}", 2.0 * r * k),
    }
}

/// Draws a current: its zone, and chevrons drifting along it at a pace that follows
/// its speed. `vel` is in world axes (+y up), `c` and `r` in screen points.
fn draw_current(painter: &Painter, c: Pos2, r: f32, vel: [f32; 2], color: Color32, now: f64, strength: f32) {
    let color = fade(color, strength);
    let speed = vel[0].hypot(vel[1]).max(1e-3);
    let dir = vec2(vel[0], -vel[1]) / speed;
    let side = vec2(-dir.y, dir.x);
    painter.circle_filled(c, r, color.gamma_multiply(0.04));
    painter.circle_stroke(c, r, theme::stroke(1.0, color.gamma_multiply(0.45)));
    const CHEVRONS: usize = 3;
    let phase = (now * (speed as f64 / CURRENT_SPEED as f64) * 0.6).fract() as f32;
    for i in 0..CHEVRONS {
        // Each chevron crosses the zone from tail to head, fading in and out at the ends.
        let t = (i as f32 + phase) / CHEVRONS as f32;
        let tip = c + dir * r * (1.5 * t - 0.6);
        let wing = (r * 0.28).clamp(3.0, 14.0);
        let stroke = theme::stroke(1.5, color.gamma_multiply(0.9 * (PI * t).sin()));
        painter.line_segment([tip, tip - dir * wing + side * wing], stroke);
        painter.line_segment([tip, tip - dir * wing - side * wing], stroke);
    }
}

/// A drain: a dark hole with a grate, and marks circling in.
fn draw_drain(painter: &Painter, c: Pos2, r: f32, now: f64, strength: f32) {
    let r = r.max(5.0);
    let color = fade(theme::DANGER, strength);
    painter.circle_filled(c, r, Color32::from_black_alpha((110.0 * strength) as u8));
    painter.circle_stroke(c, r, theme::stroke(1.5, color.gamma_multiply(0.8)));
    painter.circle_stroke(c, r * 0.55, theme::stroke(1.0, color.gamma_multiply(0.4)));
    let turn = now as f32 * 1.2;
    for i in 0..3 {
        let a = turn + i as f32 * TAU / 3.0;
        let pts: Vec<Pos2> = (0..=8).map(|k| k as f32 / 8.0).map(|u| c + Vec2::angled(a + u * 1.3) * r * (0.92 - 0.55 * u)).collect();
        painter.add(egui::Shape::line(pts, theme::stroke(1.3, color.gamma_multiply(0.75))));
    }
}

/// A faucet: a brass tap with a bead of its liquid, its mouth as wide as its stream.
fn draw_faucet(painter: &Painter, c: Pos2, zoom: f32, f: &Faucet, liquid: Color32, strength: f32) {
    let spout = f.spout();
    let dir = vec2(spout.direction[0], -spout.direction[1]);
    let side = vec2(-dir.y, dir.x);
    let half = (0.5 * spout.width * zoom).max(4.0);
    let brass = fade(theme::BRASS, strength);
    // The mouth and the pipe behind it.
    painter.line_segment([c - side * half, c + side * half], theme::stroke(3.0, brass));
    painter.line_segment([c - dir * (half * 0.6 + 8.0), c], theme::stroke(3.0, brass.gamma_multiply(0.75)));
    painter.circle_filled(c - dir * (half * 0.6 + 8.0), 4.5, brass);
    painter.circle_filled(c - dir * (half * 0.6 + 8.0), 2.4, fade(liquid, strength));
    // Where it will run.
    painter.arrow(c + dir * 5.0, dir * (half + 10.0), theme::stroke(1.5, fade(liquid, 0.8 * strength)));
}

/// A bar magnet: berry-red north towards its pole direction, blue south.
fn draw_magnet(painter: &Painter, c: Pos2, zoom: f32, m: &Magnet, strength: f32) {
    let r = (m.radius * zoom).max(4.0);
    let dir = vec2(m.dir[0], -m.dir[1]);
    let side = vec2(-dir.y, dir.x) * r * 0.45;
    let north = [c + side, c + side + dir * r, c - side + dir * r, c - side];
    let south = [c + side, c + side - dir * r, c - side - dir * r, c - side];
    painter.add(egui::Shape::convex_polygon(north.to_vec(), Color32::from_rgb(222, 84, 104).gamma_multiply(0.85 * strength), Stroke::NONE));
    painter.add(egui::Shape::convex_polygon(south.to_vec(), Color32::from_rgb(86, 130, 226).gamma_multiply(0.85 * strength), Stroke::NONE));
    painter.circle_stroke(c, r * 1.6, theme::stroke(1.0, theme::TEXT.gamma_multiply(0.22 * strength)));
}

/// Draws a solid on the canvas: a placement preview, or the outline of one in the world.
fn draw_shape(painter: &Painter, ctx: &egui::Context, c: &Canvas, shape: &Shape, stroke: Stroke, fill: Color32) {
    let zoom = c.view.zoom / ctx.pixels_per_point();
    let local = |centre: [f32; 2], angle: f32, q: [f32; 2]| {
        let (s, co) = angle.sin_cos();
        c.to_screen(ctx, [centre[0] + co * q[0] - s * q[1], centre[1] + s * q[0] + co * q[1]])
    };
    let polygon = |pts: Vec<Pos2>| egui::Shape::convex_polygon(pts, fill, stroke);
    match *shape {
        Shape::Circle { c: centre, r } => {
            painter.circle(c.to_screen(ctx, centre), r * zoom, fill, stroke);
        }
        Shape::Box { c: centre, half, angle } => {
            let pts = [[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]];
            painter.add(polygon(pts.map(|q| local(centre, angle, q)).to_vec()));
        }
        Shape::Ramp { c: centre, half, angle, flip } => {
            let top = if flip { half[0] } else { -half[0] };
            let pts = [[-half[0], -half[1]], [half[0], -half[1]], [top, half[1]]];
            painter.add(polygon(pts.map(|q| local(centre, angle, q)).to_vec()));
        }
        Shape::Capsule { a, b, r } => {
            let d = [b[0] - a[0], b[1] - a[1]];
            let angle = if d[0].hypot(d[1]) > 1e-4 { d[1].atan2(d[0]) } else { 0.0 };
            let n = 10;
            let mut pts = Vec::with_capacity(2 * n + 2);
            // Round the far end, then the near one.
            for (end, turn) in [(b, -PI / 2.0), (a, PI / 2.0)] {
                for k in 0..=n {
                    let t = angle + turn + PI * k as f32 / n as f32;
                    pts.push(c.to_screen(ctx, [end[0] + r * t.cos(), end[1] + r * t.sin()]));
                }
            }
            painter.add(polygon(pts));
        }
        Shape::Bowl { c: centre, r, t, angle } => {
            // Not convex: the fill is a thick arc, the outline a closed line around it.
            let n = 28;
            let arc = |rho: f32, k: usize| {
                let th = PI + PI * k as f32 / n as f32;
                local(centre, angle, [rho * th.cos(), rho * th.sin()])
            };
            let cap = |x: f32, k: usize| {
                let ph = PI * k as f32 / 8.0;
                local(centre, angle, [x + t * ph.cos(), t * ph.sin()])
            };
            if fill != Color32::TRANSPARENT {
                painter.add(egui::Shape::line((0..=n).map(|k| arc(r, k)).collect(), theme::stroke(2.0 * t * zoom, fill)));
            }
            let mut pts: Vec<Pos2> = (0..=n).map(|k| arc(r + t, k)).collect();
            pts.extend((1..8).map(|k| cap(r, k)));
            pts.extend((0..=n).rev().map(|k| arc((r - t).max(0.0), k)));
            pts.extend((1..8).map(|k| cap(-r, k)));
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
    }
}
