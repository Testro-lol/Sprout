//! The interactive camera: a `View` that eases towards where it was asked to go.
//!
//! Zooming animates (about 0.15 s) and keeps the point of the grid under the cursor fixed
//! all the way, so a zoom never slides away from what the user is looking at. Panning
//! follows the pointer one to one, since anything else feels like lag.

use crate::render::View;

/// How fast the view closes the gap to its target, per second.
const RATE: f32 = 22.0;

pub struct Camera {
    /// The view to draw this frame.
    pub view: View,
    /// Where the view is heading.
    target: View,
    /// While a zoom animates: the grid point to keep under this pixel.
    anchor: Option<([f32; 2], [f32; 2])>,
    /// Whether the target is "the whole grid", so that it follows window resizes.
    fit: bool,
    /// Pixels at the bottom of the window that the interface covers: the grid is fitted
    /// into what is above them.
    below: f32,
}

impl Camera {
    pub fn new(grid: [u32; 2], size: [u32; 2]) -> Camera {
        let view = View::fit(grid, size);
        Camera { view, target: view, anchor: None, fit: true, below: 0.0 }
    }

    /// Where the camera is heading (what the user asked for), as opposed to `view`.
    pub fn target(&self) -> View {
        self.target
    }

    /// Shows the whole grid: eased, or at once (on a resize or a new grid).
    pub fn fit(&mut self, grid: [u32; 2], size: [u32; 2], animate: bool) {
        self.target = View::fit_above(grid, size, self.below);
        self.anchor = None;
        self.fit = true;
        if !animate {
            self.view = self.target;
        }
    }

    /// The window changed size: a fitted view stays fitted.
    pub fn resized(&mut self, grid: [u32; 2], size: [u32; 2]) {
        if self.fit {
            self.fit(grid, size, false);
        }
    }

    /// Reserves the bottom `below` pixels of the window for the interface (0 when it is
    /// hidden). A fitted view glides to its new place; any other view stays where it is.
    pub fn set_below(&mut self, below: f32, grid: [u32; 2], size: [u32; 2]) {
        if below != self.below {
            self.below = below;
            if self.fit {
                self.fit(grid, size, true);
            }
        }
    }

    /// Zooms by `factor` about the pixel `px`, between half and 24 times the fitted zoom.
    pub fn zoom_at(&mut self, px: [f32; 2], factor: f32, grid: [u32; 2], size: [u32; 2]) {
        let fit = View::fit_above(grid, size, self.below).zoom;
        let anchor = self.anchor.map_or_else(|| self.view.screen_to_world(px, size), |(w, p)| {
            // A new notch mid-animation keeps the original anchor if the pointer has not moved.
            if p == px { w } else { self.view.screen_to_world(px, size) }
        });
        self.target.zoom = (self.target.zoom * factor).clamp(fit * 0.5, fit * 24.0);
        self.anchor = Some((anchor, px));
        self.fit = false;
    }

    /// Moves the view by `delta_px` pixels, at once.
    pub fn pan(&mut self, delta_px: [f32; 2]) {
        let d = [-delta_px[0] / self.view.zoom, delta_px[1] / self.view.zoom];
        for v in [&mut self.view, &mut self.target] {
            v.center[0] += d[0];
            v.center[1] += d[1];
        }
        if let Some((w, _)) = &mut self.anchor {
            w[0] += d[0];
            w[1] += d[1];
        }
        self.fit = false;
    }

    /// Advances the animation by `dt` seconds.
    pub fn update(&mut self, dt: f32, size: [u32; 2]) {
        let k = 1.0 - (-RATE * dt.clamp(0.0, 0.1)).exp();
        let (z, tz) = (self.view.zoom.ln(), self.target.zoom.ln());
        let done = (tz - z).abs() < 1e-3;
        self.view.zoom = if done { self.target.zoom } else { (z + (tz - z) * k).exp() };
        match self.anchor {
            Some((world, px)) => {
                // Solve screen_to_world(px) == world for the centre.
                self.view.center = [
                    world[0] - (px[0] - size[0] as f32 * 0.5) / self.view.zoom,
                    world[1] + (px[1] - size[1] as f32 * 0.5) / self.view.zoom,
                ];
                self.target.center = self.view.center;
                if done {
                    self.anchor = None;
                }
            }
            None => {
                for i in 0..2 {
                    let gap = self.target.center[i] - self.view.center[i];
                    self.view.center[i] = if gap.abs() * self.view.zoom < 0.05 { self.target.center[i] } else { self.view.center[i] + gap * k };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID: [u32; 2] = [512, 288];
    const SIZE: [u32; 2] = [1600, 900];

    #[test]
    fn zoom_keeps_the_point_under_the_cursor_and_settles() {
        let mut cam = Camera::new(GRID, SIZE);
        let px = [1200.0, 300.0];
        let under = cam.view.screen_to_world(px, SIZE);
        cam.zoom_at(px, 2.0, GRID, SIZE);
        for frame in 0..60 {
            cam.update(1.0 / 60.0, SIZE);
            let now = cam.view.screen_to_world(px, SIZE);
            assert!((now[0] - under[0]).abs() < 1e-3 && (now[1] - under[1]).abs() < 1e-3, "frame {frame}: {now:?}");
        }
        assert_eq!(cam.view.zoom, cam.target().zoom);
        assert!((cam.view.zoom / View::fit(GRID, SIZE).zoom - 2.0).abs() < 1e-4);
    }

    #[test]
    fn a_fitted_view_makes_room_for_the_dock_and_takes_it_back() {
        let mut cam = Camera::new(GRID, SIZE);
        let full = cam.view;
        cam.set_below(200.0, GRID, SIZE);
        for _ in 0..120 {
            cam.update(1.0 / 60.0, SIZE);
        }
        // The whole grid is above the dock: its floor at 700 px, its top at 0, centred.
        let floor = cam.view.world_to_screen([256.0, 0.0], SIZE);
        let top = cam.view.world_to_screen([256.0, 288.0], SIZE);
        assert!((floor[1] - 700.0).abs() < 0.5 && top[1].abs() < 0.5 && (floor[0] - 800.0).abs() < 0.5, "{floor:?} {top:?}");
        cam.set_below(0.0, GRID, SIZE);
        for _ in 0..120 {
            cam.update(1.0 / 60.0, SIZE);
        }
        assert_eq!(cam.view, full);
        // A view the user has moved is left alone.
        cam.pan([40.0, 0.0]);
        let moved = cam.view;
        cam.set_below(200.0, GRID, SIZE);
        cam.update(1.0 / 60.0, SIZE);
        assert_eq!(cam.view, moved);
    }

    #[test]
    fn panning_is_immediate_and_fit_animates_back() {
        let mut cam = Camera::new(GRID, SIZE);
        let start = cam.view;
        cam.pan([100.0, -50.0]);
        assert!((cam.view.center[0] - (start.center[0] - 100.0 / start.zoom)).abs() < 1e-4);
        assert!((cam.view.center[1] - (start.center[1] - 50.0 / start.zoom)).abs() < 1e-4);
        cam.fit(GRID, SIZE, true);
        cam.update(1.0 / 60.0, SIZE);
        assert!(cam.view.center != start.center, "eased, not instant");
        for _ in 0..120 {
            cam.update(1.0 / 60.0, SIZE);
        }
        assert_eq!(cam.view.center, start.center);
    }
}
