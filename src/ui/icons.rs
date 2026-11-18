//! Icons, drawn as vectors each frame: crisp at any size and scale, and tinted to whatever
//! they sit on. Each is designed in a 24 x 24 box.

use std::f32::consts::{PI, TAU};

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, Vec2, vec2};

use super::theme;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    // Tools.
    Pour,
    Jet,
    Spray,
    Drop,
    Rain,
    Grab,
    Push,
    Pull,
    Whirl,
    Current,
    Hold,
    Heat,
    Cool,
    Circle,
    Box,
    Ramp,
    Bowl,
    Wall,
    Draw,
    Faucet,
    Drain,
    Pump,
    Magnet,
    Erase,
    Remove,
    Move,
    Pipette,
    // Groups without a tool of the same picture.
    Blocks,
    // Chrome.
    Play,
    Pause,
    Step,
    Reset,
    Undo,
    Redo,
    Gear,
    Keys,
    Hide,
    Library,
    Pencil,
    Close,
    Chevron,
    Wipe,
    Home,
}

/// The pen an icon is drawn with: maps the 24-box onto `rect`.
struct Pen<'a> {
    p: &'a Painter,
    rect: Rect,
    stroke: Stroke,
    color: Color32,
}

impl Pen<'_> {
    fn at(&self, x: f32, y: f32) -> Pos2 {
        self.rect.min + vec2(x, y) * (self.rect.width() / 24.0)
    }

    fn s(&self, v: f32) -> f32 {
        v * self.rect.width() / 24.0
    }

    fn line(&self, pts: &[(f32, f32)]) {
        self.p.add(Shape::line(pts.iter().map(|&(x, y)| self.at(x, y)).collect(), self.stroke));
    }

    fn closed(&self, pts: &[(f32, f32)]) {
        self.p.add(Shape::closed_line(pts.iter().map(|&(x, y)| self.at(x, y)).collect(), self.stroke));
    }

    fn fill(&self, pts: &[(f32, f32)], alpha: f32) {
        self.p.add(Shape::convex_polygon(pts.iter().map(|&(x, y)| self.at(x, y)).collect(), self.color.gamma_multiply(alpha), Stroke::NONE));
    }

    fn ring(&self, x: f32, y: f32, r: f32) {
        self.p.circle_stroke(self.at(x, y), self.s(r), self.stroke);
    }

    fn dot(&self, x: f32, y: f32, r: f32) {
        self.p.circle_filled(self.at(x, y), self.s(r), self.color);
    }

    /// An arc around (x, y) from angle `a0` to `a1` (radians, clockwise from +x on screen).
    fn arc(&self, x: f32, y: f32, r: f32, a0: f32, a1: f32) {
        let n = ((a1 - a0).abs() / 0.25).ceil().max(2.0) as usize;
        let pts = (0..=n).map(|k| self.at(x, y) + Vec2::angled(a0 + (a1 - a0) * k as f32 / n as f32) * self.s(r)).collect();
        self.p.add(Shape::line(pts, self.stroke));
    }

    /// An arrowhead with its tip at (x, y), pointing along `angle`.
    fn head(&self, x: f32, y: f32, angle: f32, size: f32) {
        let tip = self.at(x, y);
        for side in [-1.0, 1.0] {
            self.p.line_segment([tip, tip - Vec2::angled(angle + side * 0.6) * self.s(size)], self.stroke);
        }
    }

    /// A teardrop: tip up at (x, y - 1.7 r), round below around (x, y).
    fn drop(&self, x: f32, y: f32, r: f32, filled: bool) {
        let a = (1.0f32 / 1.7).acos();
        let mut pts = vec![self.at(x, y - 1.7 * r)];
        let n = 14;
        for k in 0..=n {
            // From the right tangent point, round the bottom, to the left one.
            let t = -PI / 2.0 + a + (TAU - 2.0 * a) * k as f32 / n as f32;
            pts.push(self.at(x, y) + Vec2::angled(t) * self.s(r));
        }
        if filled {
            self.p.add(Shape::convex_polygon(pts, self.color, Stroke::NONE));
        } else {
            self.p.add(Shape::closed_line(pts, self.stroke));
        }
    }
}

/// Draws `icon` into `rect` (square) in `color`.
pub fn paint(painter: &Painter, icon: Icon, rect: Rect, color: Color32) {
    let width = (rect.width() / 24.0 * 1.7).max(1.2);
    let d = Pen { p: painter, rect, stroke: theme::stroke(width, color), color };
    match icon {
        Icon::Pour => {
            // A tipped cup and the stream leaving its lip.
            let cup = [(3.5, 6.5), (12.5, 3.5), (15.0, 10.5), (7.5, 13.0)];
            d.fill(&cup, 0.25);
            d.closed(&cup);
            let n = 8;
            let pts: Vec<(f32, f32)> = (0..=n).map(|k| k as f32 / n as f32).map(|t| (15.0 + 3.0 * t.sqrt(), 10.5 + 6.0 * t * t)).collect();
            d.line(&pts);
            d.drop(18.0, 20.2, 1.9, true);
        }
        Icon::Jet => {
            d.closed(&[(3.0, 9.0), (8.0, 9.0), (10.5, 12.0), (8.0, 15.0), (3.0, 15.0)]);
            d.line(&[(12.5, 12.0), (21.0, 12.0)]);
            d.line(&[(13.0, 8.5), (18.0, 7.0)]);
            d.line(&[(13.0, 15.5), (18.0, 17.0)]);
        }
        Icon::Spray => {
            d.ring(12.0, 12.0, 2.6);
            for k in 0..8 {
                let v = Vec2::angled(k as f32 * TAU / 8.0);
                d.dot(12.0 + v.x * 7.2, 12.0 + v.y * 7.2, 1.15);
            }
        }
        Icon::Drop => d.drop(12.0, 14.5, 5.6, true),
        Icon::Rain => {
            d.arc(8.5, 9.5, 3.5, PI * 0.5, PI * 1.5);
            d.arc(12.5, 8.0, 4.5, PI, TAU);
            d.arc(16.5, 10.0, 3.0, -PI * 0.5, PI * 0.5);
            d.line(&[(8.5, 13.0), (16.5, 13.0)]);
            for x in [8.5, 12.5, 16.5] {
                d.line(&[(x, 16.0), (x - 1.5, 20.0)]);
            }
        }
        Icon::Grab => {
            // A stirring swirl ending in a bead.
            d.arc(11.0, 12.0, 7.0, PI * 0.15, PI * 1.25);
            d.arc(13.0, 12.0, 3.4, PI * 1.1, PI * 2.1);
            d.dot(17.6, 14.2, 2.3);
        }
        Icon::Push | Icon::Pull => {
            let out = icon == Icon::Push;
            d.dot(12.0, 12.0, 1.8);
            for k in 0..4 {
                let a = k as f32 * TAU / 4.0 + PI / 4.0;
                let v = Vec2::angled(a);
                d.line(&[(12.0 + v.x * 4.5, 12.0 + v.y * 4.5), (12.0 + v.x * 10.0, 12.0 + v.y * 10.0)]);
                if out {
                    d.head(12.0 + v.x * 10.0, 12.0 + v.y * 10.0, a, 3.4);
                } else {
                    d.head(12.0 + v.x * 4.5, 12.0 + v.y * 4.5, a + PI, 3.4);
                }
            }
        }
        Icon::Whirl => {
            d.arc(12.0, 12.0, 8.0, -PI * 0.4, PI * 0.55);
            d.head(12.0 + 8.0 * (PI * 0.55).cos(), 12.0 + 8.0 * (PI * 0.55).sin(), PI * 1.05 + 0.2, 3.6);
            d.arc(12.0, 12.0, 8.0, PI * 0.6, PI * 1.55);
            d.head(12.0 + 8.0 * (PI * 1.55).cos(), 12.0 + 8.0 * (PI * 1.55).sin(), PI * 0.05 + 0.2, 3.6);
            d.dot(12.0, 12.0, 1.8);
        }
        Icon::Current => {
            for x in [4.0, 10.0, 16.0] {
                d.line(&[(x, 6.5), (x + 5.0, 12.0), (x, 17.5)]);
            }
        }
        Icon::Hold => {
            // Corner brackets around a bead: pinned in place.
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                d.line(&[(12.0 + sx * 8.0, 12.0 + sy * 4.0), (12.0 + sx * 8.0, 12.0 + sy * 8.0), (12.0 + sx * 4.0, 12.0 + sy * 8.0)]);
            }
            d.dot(12.0, 12.0, 2.6);
        }
        Icon::Heat => {
            d.drop(12.0, 14.5, 6.0, false);
            d.drop(12.0, 16.5, 2.4, true);
        }
        Icon::Cool => {
            for k in 0..3 {
                let v = Vec2::angled(k as f32 * PI / 3.0 + PI / 2.0);
                d.line(&[(12.0 - v.x * 9.0, 12.0 - v.y * 9.0), (12.0 + v.x * 9.0, 12.0 + v.y * 9.0)]);
                for end in [-1.0, 1.0] {
                    let (ex, ey) = (12.0 + v.x * 9.0 * end, 12.0 + v.y * 9.0 * end);
                    let a = v.y.atan2(v.x) + if end > 0.0 { PI } else { 0.0 };
                    for side in [-0.7f32, 0.7] {
                        let w = Vec2::angled(a + side);
                        d.line(&[(ex + w.x * 0.8, ey + w.y * 0.8), (ex + w.x * 3.6, ey + w.y * 3.6)]);
                    }
                }
            }
        }
        Icon::Circle => {
            d.p.circle_filled(d.at(12.0, 12.0), d.s(7.5), color.gamma_multiply(0.25));
            d.ring(12.0, 12.0, 7.5);
        }
        Icon::Box => {
            let pts = [(4.0, 9.5), (18.0, 6.0), (20.0, 14.5), (6.0, 18.0)];
            d.fill(&pts, 0.25);
            d.closed(&pts);
        }
        Icon::Ramp => {
            let pts = [(4.0, 18.5), (20.0, 18.5), (4.0, 6.5)];
            d.fill(&pts, 0.25);
            d.closed(&pts);
        }
        Icon::Bowl => {
            let thick = Pen { stroke: theme::stroke(width * 2.0, color), ..d };
            thick.arc(12.0, 9.0, 8.0, 0.0, PI);
            d.dot(12.0, 12.5, 1.6);
        }
        Icon::Wall => {
            let thick = Pen { stroke: theme::stroke(width * 2.2, color), ..d };
            thick.line(&[(5.0, 18.0), (19.0, 6.0)]);
            d.dot(5.0, 18.0, 2.0);
            d.dot(19.0, 6.0, 2.0);
        }
        Icon::Draw => {
            let n = 16;
            let pts: Vec<(f32, f32)> = (0..=n).map(|k| k as f32 / n as f32).map(|t| (3.5 + 12.0 * t, 16.5 + 3.0 * (t * TAU * 1.25).sin())).collect();
            d.line(&pts);
            d.line(&[(15.0, 13.0), (20.5, 4.5)]);
            d.dot(15.0, 13.0, 1.5);
        }
        Icon::Faucet => {
            let thick = Pen { stroke: theme::stroke(width * 1.8, color), ..d };
            thick.line(&[(3.5, 9.0), (15.5, 9.0), (15.5, 13.0)]);
            d.line(&[(9.0, 9.0), (9.0, 5.0)]);
            d.line(&[(6.0, 4.5), (12.0, 4.5)]);
            d.drop(15.5, 19.0, 2.2, true);
        }
        Icon::Drain => {
            d.ring(12.0, 12.0, 8.0);
            for x in [8.5, 12.0, 15.5] {
                let h = (64.0f32 - (x - 12.0) * (x - 12.0)).sqrt() - 2.2;
                d.line(&[(x, 12.0 - h), (x, 12.0 + h)]);
            }
        }
        Icon::Pump => {
            d.ring(12.0, 12.0, 8.0);
            d.fill(&[(9.0, 7.5), (16.5, 12.0), (9.0, 16.5)], 1.0);
        }
        Icon::Magnet => {
            let thick = Pen { stroke: theme::stroke(width * 2.2, color), ..d };
            thick.arc(12.0, 11.0, 6.0, PI, TAU);
            thick.line(&[(6.0, 11.0), (6.0, 15.0)]);
            thick.line(&[(18.0, 11.0), (18.0, 15.0)]);
            d.line(&[(3.8, 18.5), (8.2, 18.5)]);
            d.line(&[(15.8, 18.5), (20.2, 18.5)]);
        }
        Icon::Erase => {
            let pts = [(10.0, 4.5), (19.5, 14.0), (14.0, 19.5), (4.5, 10.0)];
            d.closed(&pts);
            d.fill(&[(7.25, 7.25), (16.75, 16.75), (14.0, 19.5), (4.5, 10.0)], 0.35);
            d.line(&[(7.25, 7.25), (16.75, 16.75)]);
        }
        Icon::Remove => {
            d.line(&[(4.5, 7.0), (19.5, 7.0)]);
            d.line(&[(9.5, 7.0), (9.5, 4.5), (14.5, 4.5), (14.5, 7.0)]);
            d.line(&[(6.5, 7.0), (7.5, 20.0), (16.5, 20.0), (17.5, 7.0)]);
            d.line(&[(10.3, 10.5), (10.6, 16.5)]);
            d.line(&[(13.7, 10.5), (13.4, 16.5)]);
        }
        Icon::Move => {
            d.line(&[(12.0, 3.5), (12.0, 20.5)]);
            d.line(&[(3.5, 12.0), (20.5, 12.0)]);
            for (x, y, a) in [(12.0, 3.5, -PI / 2.0), (12.0, 20.5, PI / 2.0), (3.5, 12.0, PI), (20.5, 12.0, 0.0)] {
                d.head(x, y, a, 3.4);
            }
        }
        Icon::Pipette => {
            let thick = Pen { stroke: theme::stroke(width * 1.9, color), ..d };
            thick.line(&[(14.5, 9.5), (18.5, 5.5)]);
            d.line(&[(12.0, 7.5), (16.5, 12.0)]);
            d.line(&[(13.0, 9.5), (6.0, 16.5), (5.5, 18.5), (7.5, 18.0), (14.5, 11.0)]);
            d.dot(4.6, 20.2, 1.3);
        }
        Icon::Blocks => {
            let b = [(3.5, 12.0), (12.0, 12.0), (12.0, 20.0), (3.5, 20.0)];
            d.fill(&b, 0.25);
            d.closed(&b);
            let t = [(14.5, 20.0), (21.0, 20.0), (14.5, 11.0)];
            d.fill(&t, 0.25);
            d.closed(&t);
            d.ring(9.0, 6.5, 3.2);
        }
        Icon::Play => d.fill(&[(7.5, 5.0), (19.0, 12.0), (7.5, 19.0)], 1.0),
        Icon::Pause => {
            d.fill(&[(6.5, 5.5), (10.0, 5.5), (10.0, 18.5), (6.5, 18.5)], 1.0);
            d.fill(&[(14.0, 5.5), (17.5, 5.5), (17.5, 18.5), (14.0, 18.5)], 1.0);
        }
        Icon::Step => {
            d.fill(&[(5.5, 6.0), (14.5, 12.0), (5.5, 18.0)], 1.0);
            d.fill(&[(16.0, 6.0), (18.8, 6.0), (18.8, 18.0), (16.0, 18.0)], 1.0);
        }
        Icon::Reset => {
            d.arc(12.0, 12.5, 7.0, -PI * 0.35, PI * 1.35);
            d.head(12.0 + 7.0 * (-PI * 0.35).cos(), 12.5 + 7.0 * (-PI * 0.35).sin(), -PI * 0.35 - PI / 2.0 + 0.5, 4.2);
        }
        Icon::Undo | Icon::Redo => {
            // Mirrored for redo.
            let m = |x: f32| if icon == Icon::Undo { x } else { 24.0 - x };
            let n = 12;
            let pts: Vec<(f32, f32)> = (0..=n)
                .map(|k| -PI * 0.5 + PI * 0.95 * k as f32 / n as f32)
                .map(|a| (m(13.0 + 6.0 * a.cos()), 13.5 + 6.0 * a.sin()))
                .collect();
            d.line(&pts);
            d.line(&[(m(13.0), 7.5), (m(5.0), 7.5)]);
            d.line(&[(m(8.5), 4.0), (m(5.0), 7.5), (m(8.5), 11.0)]);
        }
        Icon::Gear => {
            d.ring(12.0, 12.0, 3.0);
            d.ring(12.0, 12.0, 6.6);
            for k in 0..8 {
                let v = Vec2::angled(k as f32 * TAU / 8.0);
                d.line(&[(12.0 + v.x * 6.6, 12.0 + v.y * 6.6), (12.0 + v.x * 9.6, 12.0 + v.y * 9.6)]);
            }
        }
        Icon::Keys => {
            d.closed(&[(3.0, 7.0), (21.0, 7.0), (21.0, 17.5), (3.0, 17.5)]);
            for x in [7.0, 12.0, 17.0] {
                d.dot(x, 10.6, 1.0);
            }
            d.line(&[(8.0, 14.3), (16.0, 14.3)]);
        }
        Icon::Hide => {
            // An eye.
            let n = 12;
            let lid = |sign: f32| -> Vec<(f32, f32)> { (0..=n).map(|k| k as f32 / n as f32).map(|t| (3.5 + 17.0 * t, 12.0 + sign * 5.5 * (t * PI).sin())).collect() };
            d.line(&lid(-1.0));
            d.line(&lid(1.0));
            d.dot(12.0, 12.0, 2.6);
        }
        Icon::Library => {
            // A shelf of bottles.
            for (x, h) in [(6.0, 9.0), (12.0, 12.0), (18.0, 7.0)] {
                d.closed(&[(x - 2.2, 19.0), (x - 2.2, 19.0 - h), (x + 2.2, 19.0 - h), (x + 2.2, 19.0)]);
                d.line(&[(x, 19.0 - h), (x, 16.5 - h)]);
            }
            d.line(&[(2.5, 20.5), (21.5, 20.5)]);
        }
        Icon::Pencil => {
            d.closed(&[(15.5, 4.5), (19.5, 8.5), (8.5, 19.5), (4.0, 20.0), (4.5, 15.5)]);
            d.line(&[(13.0, 7.0), (17.0, 11.0)]);
        }
        Icon::Close => {
            d.line(&[(6.5, 6.5), (17.5, 17.5)]);
            d.line(&[(17.5, 6.5), (6.5, 17.5)]);
        }
        Icon::Chevron => d.line(&[(7.0, 9.5), (12.0, 14.5), (17.0, 9.5)]),
        Icon::Wipe => {
            // A droplet struck through: clear the liquid.
            d.drop(12.0, 14.5, 5.2, false);
            d.line(&[(4.5, 19.5), (19.5, 4.5)]);
        }
        Icon::Home => {
            // Frame corners: fit the view.
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                d.line(&[(12.0 + sx * 8.0, 12.0 + sy * 3.5), (12.0 + sx * 8.0, 12.0 + sy * 8.0), (12.0 + sx * 3.5, 12.0 + sy * 8.0)]);
            }
            d.ring(12.0, 12.0, 2.2);
        }
    }
}

/// The sprout: a stem and two leaves that sway. `grow` (0..1) unfolds it, `time` in seconds.
pub fn sprout(painter: &Painter, rect: Rect, grow: f32, time: f64, leaf: Color32, stem: Color32) {
    let s = rect.width() / 24.0;
    let base = rect.min + vec2(12.0, 22.0) * s;
    let sway = (time * 1.4).sin() as f32 * 0.07;
    let g = grow.clamp(0.0, 1.0);
    // The stem: a gentle curve up from the soil.
    let top = base + vec2(sway * 10.0, -11.0 * g) * s;
    let n = 8;
    let stem_pts: Vec<Pos2> = (0..=n)
        .map(|k| k as f32 / n as f32)
        .map(|t| {
            let mid = base + vec2(1.6, -6.0 * g) * s;
            // Quadratic Bezier base -> mid -> top.
            let a = base.lerp(mid, t);
            let b = mid.lerp(top, t);
            a.lerp(b, t)
        })
        .collect();
    painter.add(Shape::line(stem_pts, theme::stroke(1.9 * s, stem)));
    // Each leaf: a lens between two arcs, from the top of the stem out along `dir`.
    let leaf_at = |angle: f32, length: f32, tint: Color32| {
        let dir = Vec2::angled(angle + sway);
        let side = vec2(-dir.y, dir.x);
        let len = length * s * g;
        let n = 10;
        let mut pts = Vec::with_capacity(2 * n);
        for k in 0..=n {
            let t = k as f32 / n as f32;
            pts.push(top + dir * len * t + side * len * 0.34 * (t * PI).sin());
        }
        for k in (1..n).rev() {
            let t = k as f32 / n as f32;
            pts.push(top + dir * len * t - side * len * 0.22 * (t * PI).sin());
        }
        painter.add(Shape::convex_polygon(pts, tint, Stroke::NONE));
    };
    leaf_at(-PI * 0.80, 10.5, leaf);
    leaf_at(-PI * 0.22, 8.5, leaf.gamma_multiply(0.82));
}
