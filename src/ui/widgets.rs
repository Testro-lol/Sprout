//! The controls the panels are built from, and the motion they share.
//!
//! Motion: nothing snaps. Positions and sizes `glide` (an exponential ease) or `spring`
//! (with a little overshoot, for things the user just chose); both keep their state in
//! egui's memory under the widget's id and advance by the frame time, so they run the same
//! at any frame rate.

use std::ops::RangeInclusive;

use egui::{Color32, CornerRadius, Id, Rect, Response, Sense, Ui, Vec2, pos2, vec2};

use super::{
    icons::{self, Icon},
    theme::{self, fade, mix},
};

// ---- motion --------------------------------------------------------------------------------

fn frame_dt(ctx: &egui::Context) -> f32 {
    ctx.input(|i| i.stable_dt).clamp(0.0, 0.05)
}

/// Eases towards `target`, closing the gap at `rate` per second. Starts at the target.
pub fn glide(ctx: &egui::Context, id: Id, target: f32, rate: f32) -> f32 {
    let dt = frame_dt(ctx);
    ctx.data_mut(|d| {
        let v = d.get_temp_mut_or(id, target);
        *v += (target - *v) * (1.0 - (-rate * dt).exp());
        if (target - *v).abs() < 1e-3 {
            *v = target;
        }
        *v
    })
}

/// 0..1 following `on`, eased.
pub fn glide_bool(ctx: &egui::Context, id: Id, on: bool, rate: f32) -> f32 {
    glide(ctx, id, on as u8 as f32, rate)
}

/// A damped spring towards `target`: arrives in about a fifth of a second with a small
/// overshoot. Starts at the target.
pub fn spring(ctx: &egui::Context, id: Id, target: f32) -> f32 {
    const STIFFNESS: f32 = 520.0;
    const DAMPING: f32 = 30.0;
    let dt = frame_dt(ctx);
    ctx.data_mut(|d| {
        let s = d.get_temp_mut_or(id, [target, 0.0f32]);
        // Small steps: the spring is stiff next to a long frame.
        let n = (dt / 0.004).ceil().max(1.0) as usize;
        let h = dt / n as f32;
        for _ in 0..n {
            s[1] += (STIFFNESS * (target - s[0]) - DAMPING * s[1]) * h;
            s[0] += s[1] * h;
        }
        if (target - s[0]).abs() < 1e-3 && s[1].abs() < 1e-2 {
            *s = [target, 0.0];
        }
        s[0]
    })
}

/// Moves a spring's value without changing where it is heading (a kick: the pop of a
/// thing just picked).
pub fn kick(ctx: &egui::Context, id: Id, value: f32) {
    ctx.data_mut(|d| {
        let s = d.get_temp_mut_or(id, [value, 0.0f32]);
        s[0] = value;
    });
}

// ---- small pieces --------------------------------------------------------------------------

/// A round icon button for the bars. `on` tints it as active.
pub fn icon_button(ui: &mut Ui, icon: Icon, tip: &str, on: Option<Color32>) -> Response {
    let size = 30.0;
    let (rect, r) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered() && ui.is_enabled(), 20.0);
    let down = glide_bool(ui.ctx(), r.id.with("d"), r.is_pointer_button_down_on(), 30.0);
    let p = ui.painter();
    let bg = match on {
        Some(c) => fade(c, 0.22 + 0.12 * hover),
        None => fade(theme::TEXT, 0.09 * hover),
    };
    p.rect_filled(rect.shrink(1.0 + down), CornerRadius::same(10), bg);
    let color = if !ui.is_enabled() {
        theme::TEXT_FAINT.gamma_multiply(0.6)
    } else {
        match on {
            Some(c) => c,
            None => mix(theme::TEXT_WEAK, theme::TEXT, hover),
        }
    };
    icons::paint(p, icon, Rect::from_center_size(rect.center(), Vec2::splat(18.0 - down)), color);
    if tip.is_empty() { r } else { r.on_hover_text(tip) }
}

/// A thin vertical rule between groups of a bar.
pub fn rule(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(9.0, 22.0), Sense::hover());
    ui.painter().vline(rect.center().x, rect.y_range(), theme::stroke(1.0, fade(theme::TEXT, 0.10)));
}

/// A small filled circle in line with text.
pub fn dot(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(9.0, 9.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
}

/// A section title inside a panel.
pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).font(theme::display(11.5)).color(theme::LEAF));
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 10.0), Sense::hover());
        ui.painter().hline(rect.x_range(), rect.center().y, theme::stroke(1.0, fade(theme::LEAF, 0.18)));
    });
}

/// A segmented choice: one chip per option, the chosen one a leaf in `color`. Returns the
/// option clicked, if any.
pub fn chips(ui: &mut Ui, id: Id, options: &[&str], chosen: usize, color: Color32) -> Option<usize> {
    let font = theme::body(12.0);
    let widths: Vec<f32> = options.iter().map(|o| ui.fonts(|f| f.layout_no_wrap(o.to_string(), font.clone(), theme::TEXT).size().x) + 18.0).collect();
    let total: f32 = widths.iter().sum();
    let (rect, _) = ui.allocate_exact_size(vec2(total, 24.0), Sense::hover());
    let p = ui.painter().clone();
    p.rect_filled(rect, CornerRadius::same(9), theme::RAISED);
    // The leaf slides to the chosen option.
    let x0: f32 = widths[..chosen.min(options.len() - 1)].iter().sum();
    let x = glide(ui.ctx(), id.with("x"), x0, 22.0);
    let w = glide(ui.ctx(), id.with("w"), widths[chosen.min(options.len() - 1)], 22.0);
    p.rect_filled(Rect::from_min_size(rect.min + vec2(x, 0.0), vec2(w, 24.0)).shrink(2.0), theme::leaf(9), color);
    let mut clicked = None;
    let mut left = rect.left();
    for (k, o) in options.iter().enumerate() {
        let cell = Rect::from_min_size(pos2(left, rect.top()), vec2(widths[k], 24.0));
        left += widths[k];
        let r = ui.interact(cell, id.with(k), Sense::click());
        let color = if k == chosen { theme::INK } else if r.hovered() { theme::TEXT } else { theme::TEXT_WEAK };
        p.text(cell.center(), egui::Align2::CENTER_CENTER, o, font.clone(), color);
        if r.clicked() && k != chosen {
            clicked = Some(k);
        }
    }
    clicked
}

// ---- the slider ----------------------------------------------------------------------------

/// How a slider maps its track to values.
#[derive(Clone, Copy)]
pub struct Scale {
    pub range: [f32; 2],
    pub log: bool,
    /// Double-click returns here.
    pub default: Option<f32>,
}

impl Scale {
    pub fn lin(range: RangeInclusive<f32>) -> Scale {
        Scale { range: [*range.start(), *range.end()], log: false, default: None }
    }

    pub fn log(range: RangeInclusive<f32>) -> Scale {
        Scale { range: [*range.start(), *range.end()], log: true, default: None }
    }

    pub fn reset_to(self, default: f32) -> Scale {
        Scale { default: Some(default), ..self }
    }

    fn to_t(self, v: f32) -> f32 {
        let [lo, hi] = self.range;
        let t = if self.log { (v / lo).max(1e-9).ln() / (hi / lo).ln() } else { (v - lo) / (hi - lo) };
        t.clamp(0.0, 1.0)
    }

    fn value_at(self, t: f32) -> f32 {
        let [lo, hi] = self.range;
        if self.log { lo * (hi / lo).powf(t) } else { lo + (hi - lo) * t }
    }
}

/// A labelled slider for the tool options: the name above a rail that fills in `color`, the
/// value to its right. The knob swells under the pointer and the fill follows the value
/// with a short glide. Drag anywhere on it; double-click resets (when the scale has a
/// default).
pub fn option_slider(ui: &mut Ui, label: &str, value: &mut f32, scale: Scale, text: String, color: Color32, tip: &str) -> Response {
    let width = 138.0;
    let (rect, mut r) = ui.allocate_exact_size(vec2(width, 34.0), Sense::click_and_drag());
    if let Some(pos) = r.interact_pointer_pos().filter(|_| r.dragged() || r.is_pointer_button_down_on()) {
        let t = ((pos.x - rect.left() - 7.0) / (rect.width() - 14.0)).clamp(0.0, 1.0);
        let v = scale.value_at(t);
        if v != *value {
            *value = v;
            r.mark_changed();
        }
    }
    if r.double_clicked()
        && let Some(d) = scale.default
    {
        *value = d;
        r.mark_changed();
    }
    let active = r.dragged() || r.is_pointer_button_down_on();
    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered() || active, 20.0);
    let grab = glide_bool(ui.ctx(), r.id.with("g"), active, 26.0);
    let t = glide(ui.ctx(), r.id.with("t"), scale.to_t(*value), 30.0);
    let p = ui.painter();
    p.text(rect.left_top() + vec2(1.0, 0.0), egui::Align2::LEFT_TOP, label, theme::body(11.0), mix(theme::TEXT_FAINT, theme::TEXT_WEAK, hover));
    p.text(rect.right_top(), egui::Align2::RIGHT_TOP, text, theme::body(11.0), mix(theme::TEXT_WEAK, theme::TEXT, hover));
    let y = rect.bottom() - 9.0;
    let (x0, x1) = (rect.left() + 7.0, rect.right() - 7.0);
    let rail = 4.0 + hover;
    p.rect_filled(Rect::from_min_max(pos2(x0 - 3.0, y - rail * 0.5), pos2(x1 + 3.0, y + rail * 0.5)), CornerRadius::same(4), theme::RAISED);
    let x = x0 + (x1 - x0) * t;
    p.rect_filled(Rect::from_min_max(pos2(x0 - 3.0, y - rail * 0.5), pos2(x, y + rail * 0.5)), CornerRadius::same(4), fade(color, 0.85));
    let knob = 5.0 + 1.5 * hover + 1.0 * grab;
    p.circle_filled(pos2(x, y), knob + 2.0, fade(color, 0.25 * hover));
    p.circle_filled(pos2(x, y), knob, theme::TEXT);
    p.circle_filled(pos2(x, y), knob - 2.4, color);
    if tip.is_empty() { r } else { r.on_hover_text(tip) }
}

/// What a dial's angle is measured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dial {
    /// Clockwise from straight down: the jet, currents, faucets.
    Down,
    /// Clockwise from straight up: a magnet's north pole.
    Up,
    /// Anticlockwise from the right: the turn of a solid.
    Ccw,
}

impl Dial {
    /// The direction on screen (y down) of `degrees`.
    fn dir(self, degrees: f32) -> Vec2 {
        let (s, c) = degrees.to_radians().sin_cos();
        match self {
            Dial::Down => vec2(s, c),
            Dial::Up => vec2(s, -c),
            Dial::Ccw => vec2(c, -s),
        }
    }

    fn degrees(self, d: Vec2) -> f32 {
        match self {
            Dial::Down => d.x.atan2(d.y),
            Dial::Up => d.x.atan2(-d.y),
            Dial::Ccw => (-d.y).atan2(d.x),
        }
        .to_degrees()
    }
}

/// A dial for a heading in degrees. Drag around it; it snaps to 5 degrees.
pub fn aim_dial(ui: &mut Ui, label: &str, degrees: &mut f32, kind: Dial, color: Color32, tip: &str) -> Response {
    let (rect, mut r) = ui.allocate_exact_size(vec2(70.0, 34.0), Sense::click_and_drag());
    let centre = pos2(rect.left() + 17.0, rect.center().y);
    if let Some(pos) = r.interact_pointer_pos().filter(|_| r.dragged() || r.is_pointer_button_down_on()) {
        let d = pos - centre;
        if d.length() > 3.0 {
            let a = (kind.degrees(d) / 5.0).round() * 5.0;
            if a != *degrees {
                *degrees = a;
                r.mark_changed();
            }
        }
    }
    let active = r.dragged() || r.is_pointer_button_down_on();
    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered() || active, 20.0);
    // The needle turns the short way round.
    // (Read before `data_mut`: the context's lock is not re-entrant.)
    let dt = frame_dt(ui.ctx());
    let shown = ui.ctx().data_mut(|d| {
        let s = d.get_temp_mut_or(r.id.with("a"), *degrees);
        let gap = (*degrees - *s + 540.0).rem_euclid(360.0) - 180.0;
        *s += gap * (1.0 - (-26.0 * dt).exp());
        *s
    });
    let dir = kind.dir(shown);
    let p = ui.painter();
    let radius = 13.0 + hover;
    p.circle_filled(centre, radius, theme::RAISED);
    p.circle_stroke(centre, radius, theme::stroke(1.0, fade(color, 0.35 + 0.4 * hover)));
    for k in 0..8 {
        let v = Vec2::angled(k as f32 * std::f32::consts::TAU / 8.0);
        p.line_segment([centre + v * (radius - 2.5), centre + v * (radius - 0.5)], theme::stroke(1.0, fade(theme::TEXT, 0.25)));
    }
    p.line_segment([centre, centre + dir * (radius - 3.0)], theme::stroke(2.0, color));
    p.circle_filled(centre + dir * (radius - 3.0), 2.6, color);
    p.circle_filled(centre, 2.0, theme::TEXT);
    p.text(pos2(rect.left() + 36.0, rect.top()), egui::Align2::LEFT_TOP, label, theme::body(11.0), mix(theme::TEXT_FAINT, theme::TEXT_WEAK, hover));
    p.text(pos2(rect.left() + 36.0, rect.bottom() - 2.0), egui::Align2::LEFT_BOTTOM, format!("{:.0}°", *degrees), theme::body(12.0), mix(theme::TEXT_WEAK, theme::TEXT, hover));
    if tip.is_empty() { r } else { r.on_hover_text(tip) }
}

/// A text button in the theme's shape. `color`: `Some` makes it a filled leaf (the main
/// action of a panel).
pub fn button(ui: &mut Ui, text: &str, color: Option<Color32>) -> Response {
    let font = theme::body(13.0);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font, theme::TEXT));
    let (rect, r) = ui.allocate_exact_size(vec2(galley.size().x + 22.0, 26.0), Sense::click());
    let hover = glide_bool(ui.ctx(), r.id.with("h"), r.hovered() && ui.is_enabled(), 20.0);
    let down = glide_bool(ui.ctx(), r.id.with("d"), r.is_pointer_button_down_on(), 30.0);
    let p = ui.painter();
    let body = rect.shrink(down);
    let enabled = ui.is_enabled();
    let (fill, ink) = match color {
        Some(c) if enabled => (mix(c, Color32::WHITE, 0.15 * hover), theme::INK),
        _ if enabled => (mix(theme::RAISED, theme::HOVER, hover), mix(theme::TEXT_WEAK, theme::TEXT, 0.6 + 0.4 * hover)),
        _ => (fade(theme::RAISED, 0.6), theme::TEXT_FAINT),
    };
    p.rect_filled(body, theme::leaf(10), fill);
    p.galley_with_override_text_color(body.center() - galley.size() * 0.5, galley, ink);
    r
}
