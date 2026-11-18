//! Sprout's look: a greenhouse at dusk. Deep moss panels, cream text, a fresh leaf green
//! for what is alive (selected, running), and one colour per tool group taken from what the
//! group handles: water, wind, berry, clay, brass, ember.
//!
//! Everything that is a colour, a radius or a font size in the UI comes from here.

use egui::{Color32, CornerRadius, FontFamily, FontId, Shadow, Stroke};

// ---- palette -------------------------------------------------------------------------------

/// Panel background: nearly opaque, so the tank behind never fights the controls.
pub const PANEL: Color32 = Color32::from_rgba_premultiplied(17, 27, 21, 242);
/// A step up from the panel: wells, tracks, unselected chips.
pub const RAISED: Color32 = Color32::from_rgb(30, 44, 35);
pub const HOVER: Color32 = Color32::from_rgb(40, 58, 46);
pub const ACTIVE: Color32 = Color32::from_rgb(52, 75, 59);
/// Text fields and the bottom of wells.
pub const SUNK: Color32 = Color32::from_rgb(9, 14, 11);
/// The hairline around panels: light catching a leaf's edge.
pub const EDGE: Color32 = Color32::from_rgba_premultiplied(26, 32, 26, 34);

pub const TEXT: Color32 = Color32::from_rgb(232, 238, 224);
pub const TEXT_WEAK: Color32 = Color32::from_rgb(160, 176, 158);
pub const TEXT_FAINT: Color32 = Color32::from_rgb(108, 124, 110);
/// Text and icons on top of an accent fill.
pub const INK: Color32 = Color32::from_rgb(12, 20, 14);

/// Leaf green: selection, running, "yes".
pub const LEAF: Color32 = Color32::from_rgb(155, 229, 100);
pub const LEAF_DEEP: Color32 = Color32::from_rgb(78, 159, 61);
pub const DANGER: Color32 = Color32::from_rgb(255, 112, 98);
pub const WARN: Color32 = Color32::from_rgb(246, 186, 92);

pub const WATER: Color32 = Color32::from_rgb(98, 196, 240);
pub const WIND: Color32 = Color32::from_rgb(112, 228, 196);
pub const BERRY: Color32 = Color32::from_rgb(242, 116, 146);
pub const CLAY: Color32 = Color32::from_rgb(224, 184, 142);
pub const BRASS: Color32 = Color32::from_rgb(244, 196, 84);
pub const EMBER: Color32 = Color32::from_rgb(255, 142, 82);
pub const FROST: Color32 = Color32::from_rgb(150, 208, 255);

// ---- shapes --------------------------------------------------------------------------------

/// Corner radius of floating panels.
pub const PANEL_RADIUS: u8 = 18;

/// A leaf: two opposite corners round, the other two nearly sharp. The shape of everything
/// selected.
pub const fn leaf(r: u8) -> CornerRadius {
    CornerRadius { nw: r, ne: r / 4, sw: r / 4, se: r }
}

/// The frame of a floating panel.
pub fn panel() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(stroke(1.0, EDGE))
        .corner_radius(CornerRadius::same(PANEL_RADIUS))
        .shadow(Shadow { offset: [0, 10], blur: 28, spread: 0, color: Color32::from_black_alpha(120) })
        .inner_margin(egui::Margin::symmetric(10, 8))
}

/// A stroke (with the width as an `f32`, which `Stroke::new` leaves to inference).
pub fn stroke(width: f32, color: Color32) -> Stroke {
    Stroke { width, color }
}

// ---- type ----------------------------------------------------------------------------------

/// The rounded display face (Comfortaa): the wordmark, titles, tool and group names.
pub fn display(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("display".into()))
}

pub fn body(size: f32) -> FontId {
    FontId::proportional(size)
}

pub fn mono(size: f32) -> FontId {
    FontId::monospace(size)
}

// ---- colour helpers ------------------------------------------------------------------------

pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_premultiplied(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()), f(a.a(), b.a()))
}

pub fn fade(c: Color32, alpha: f32) -> Color32 {
    c.gamma_multiply(alpha.clamp(0.0, 1.0))
}

/// A liquid's colour lifted enough to read as an accent on the dark panel.
pub fn lifted(c: Color32) -> Color32 {
    let lum = 0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32;
    if lum < 110.0 { mix(c, TEXT, (110.0 - lum) / 150.0) } else { c }
}

// ---- egui style ----------------------------------------------------------------------------

/// Installs the fonts and the visuals. Call once per context.
pub fn apply(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "comfortaa".into(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!("../../assets/fonts/Comfortaa-Bold.otf"))),
    );
    // Falls back to the body face for anything Comfortaa lacks (arrows, symbols).
    let mut display_family = vec!["comfortaa".to_string()];
    display_family.extend(fonts.families[&FontFamily::Proportional].iter().cloned());
    fonts.families.insert(FontFamily::Name("display".into()), display_family);
    ctx.set_fonts(fonts);

    let mut v = egui::Visuals::dark();
    let radius = CornerRadius::same(8);
    v.window_corner_radius = CornerRadius::same(PANEL_RADIUS);
    v.menu_corner_radius = CornerRadius::same(12);
    v.window_fill = PANEL;
    v.panel_fill = PANEL;
    v.window_stroke = stroke(1.0, EDGE);
    v.window_shadow = Shadow { offset: [0, 10], blur: 28, spread: 0, color: Color32::from_black_alpha(120) };
    v.popup_shadow = Shadow { offset: [0, 6], blur: 16, spread: 0, color: Color32::from_black_alpha(110) };
    v.selection.bg_fill = fade(LEAF_DEEP, 0.55);
    v.selection.stroke = stroke(1.0, LEAF);
    v.hyperlink_color = LEAF;
    v.slider_trailing_fill = true;
    v.handle_shape = egui::style::HandleShape::Circle;
    v.override_text_color = None;
    v.widgets.noninteractive.fg_stroke = stroke(1.0, TEXT_WEAK);
    v.widgets.inactive.fg_stroke = stroke(1.0, TEXT);
    v.widgets.hovered.fg_stroke = stroke(1.5, Color32::WHITE);
    v.widgets.active.fg_stroke = stroke(1.5, Color32::WHITE);
    v.widgets.open.fg_stroke = stroke(1.0, TEXT);
    for (w, fill) in [
        (&mut v.widgets.noninteractive, None),
        (&mut v.widgets.inactive, Some(RAISED)),
        (&mut v.widgets.hovered, Some(HOVER)),
        (&mut v.widgets.active, Some(ACTIVE)),
        (&mut v.widgets.open, Some(HOVER)),
    ] {
        w.corner_radius = radius;
        if let Some(fill) = fill {
            w.weak_bg_fill = fill;
            w.bg_fill = fill;
            w.bg_stroke = Stroke::NONE;
        }
    }
    v.widgets.noninteractive.bg_stroke = stroke(1.0, fade(TEXT, 0.09));
    v.widgets.hovered.bg_stroke = stroke(1.0, fade(LEAF, 0.35));
    v.widgets.active.bg_stroke = stroke(1.0, fade(LEAF, 0.8));
    v.extreme_bg_color = SUNK;
    v.faint_bg_color = RAISED;
    v.text_cursor.stroke = stroke(2.0, LEAF);
    ctx.set_visuals(v);
    ctx.style_mut(|s| {
        use egui::TextStyle::*;
        s.text_styles.insert(Body, body(13.5));
        s.text_styles.insert(Button, body(13.5));
        s.text_styles.insert(Small, body(11.5));
        s.text_styles.insert(Monospace, mono(12.0));
        s.text_styles.insert(Heading, display(17.0));
        s.spacing.item_spacing = egui::vec2(6.0, 6.0);
        s.spacing.button_padding = egui::vec2(9.0, 4.0);
        s.spacing.window_margin = egui::Margin::same(12);
        s.spacing.menu_margin = egui::Margin::same(8);
        s.spacing.slider_width = 130.0;
        s.spacing.slider_rail_height = 5.0;
        s.spacing.interact_size.y = 22.0;
        s.spacing.scroll.bar_width = 6.0;
        s.spacing.scroll.floating = true;
        s.animation_time = 0.14;
        s.interaction.tooltip_delay = 0.35;
    });
}
