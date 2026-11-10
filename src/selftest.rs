//! A scripted session for the real window: `sprout --selftest`.
//!
//! The physics checks in `validate` run off-screen. This drives the interactive app
//! instead: a script feeds the same handlers that mouse and keyboard events reach, frame
//! by frame, while the window renders, and checks what each tool did to the world.

use winit::{event::MouseButton, keyboard::KeyCode};

use crate::scene::WALL;

/// What the script can see of the app.
#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub particles: usize,
    pub oil: usize,
    /// Syrup content, in particles.
    pub syrup: f32,
    pub rms_speed: f32,
    pub shapes: usize,
    pub substeps: u64,
    pub grid: [u32; 2],
    pub map: usize,
    pub zoom: f32,
    pub center: [f32; 2],
    pub brush: f32,
    pub panel: bool,
}

/// A check compares the state at the last `Mark` with the state now.
pub type Check = fn(&Snapshot, &Snapshot) -> bool;

pub enum Act {
    Key(KeyCode),
    /// Moves the cursor to a position given in grid cells.
    Move([f32; 2]),
    Press(MouseButton),
    Release(MouseButton),
    /// Scrolls the wheel by this many lines.
    Wheel(f32),
    /// Picks a grid resolution, as the quality menu does.
    Grid([u32; 2]),
    Mark,
    Check(&'static str, Check),
}

/// Builds the list of (frame, action) pairs.
struct Script {
    frame: u64,
    acts: Vec<(u64, Act)>,
}

impl Script {
    fn push(&mut self, act: Act) -> &mut Self {
        self.acts.push((self.frame, act));
        self
    }

    fn wait(&mut self, frames: u64) -> &mut Self {
        self.frame += frames;
        self
    }

    fn key(&mut self, code: KeyCode) -> &mut Self {
        self.push(Act::Key(code)).wait(1)
    }

    fn click(&mut self, at: [f32; 2]) -> &mut Self {
        self.push(Act::Move(at)).wait(2).push(Act::Press(MouseButton::Left)).wait(2).push(Act::Release(MouseButton::Left)).wait(2)
    }

    /// Holds a button while the cursor travels from `from` to `to` over `frames` frames.
    fn drag(&mut self, button: MouseButton, from: [f32; 2], to: [f32; 2], frames: u64) -> &mut Self {
        self.push(Act::Move(from)).wait(2).push(Act::Press(button));
        for k in 1..=frames {
            let t = k as f32 / frames as f32;
            self.wait(1).push(Act::Move([from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t]));
        }
        self.wait(1).push(Act::Release(button)).wait(2)
    }

    fn mark(&mut self) -> &mut Self {
        self.push(Act::Mark)
    }

    fn check(&mut self, what: &'static str, check: Check) -> &mut Self {
        self.push(Act::Check(what, check)).wait(1)
    }
}

/// The session: starts on the Empty Tank map at 512 x 288.
pub fn script() -> Vec<(u64, Act)> {
    use KeyCode::*;
    let mut s = Script { frame: 20, acts: Vec::new() };
    let floor = WALL + 6.0;

    // Pour each fluid (Q = pour tool; 1, 2, 3 = water, oil, syrup).
    s.mark().key(KeyQ).key(Digit1).drag(MouseButton::Left, [200.0, 150.0], [300.0, 150.0], 150);
    s.check("pour water (Q, 1, drag)", |m, s| s.particles > m.particles + 2000 && s.oil == 0 && s.syrup == 0.0);
    s.mark().key(Digit2).drag(MouseButton::Left, [120.0, 120.0], [180.0, 120.0], 90);
    s.check("pour oil (2)", |m, s| s.oil > 1000 && s.particles >= m.particles + s.oil);
    s.mark().key(Digit3).drag(MouseButton::Left, [340.0, 120.0], [400.0, 120.0], 90);
    s.check("pour syrup (3)", |m, s| s.syrup > 1000.0 && s.oil == m.oil);

    // Brush size and pour rate keys change what the nozzle emits.
    s.mark().key(BracketRight).key(BracketRight).check("brush size (])", |m, s| s.brush > m.brush * 1.2);
    s.key(BracketLeft).key(BracketLeft);

    // Place a circle, a rotated box and a flipped ramp (A, S, D; T rotates, F flips).
    s.wait(60).mark();
    s.key(KeyA).click([256.0, 40.0]).key(KeyS).key(KeyT).click([120.0, 60.0]).key(KeyD).key(KeyF).click([400.0, 60.0]);
    s.wait(30).check("place circle, box and ramp (A, S, D)", |m, s| s.shapes == m.shapes + 3 && s.particles == m.particles);

    // Stir with the grab tool and with the right mouse button.
    s.wait(240).mark().key(KeyW).drag(MouseButton::Left, [60.0, floor], [200.0, floor], 30);
    s.check("grab / stir (W, drag)", |m, s| s.rms_speed > m.rms_speed && s.particles == m.particles);
    s.mark().key(KeyQ).drag(MouseButton::Right, [200.0, floor], [60.0, floor], 30);
    s.check("stir with the right button", |m, s| s.particles == m.particles);

    // Erase, then remove a solid.
    s.mark().key(KeyE).drag(MouseButton::Left, [300.0, floor], [360.0, floor], 30);
    s.check("erase (E, drag)", |m, s| s.particles + 100 < m.particles);
    s.mark().key(KeyX).click([256.0, 40.0]);
    s.check("remove a solid (X, click)", |m, s| s.shapes + 1 == m.shapes && s.particles == m.particles);

    // Pause, single-step, resume.
    s.key(Space).wait(10).mark().wait(20).check("pause (Space)", |m, s| s.substeps == m.substeps);
    s.mark().key(KeyN).wait(3).check("single step (N)", |m, s| s.substeps == m.substeps + 1);
    s.mark().key(Space).wait(30).check("resume (Space)", |m, s| s.substeps > m.substeps + 5);

    // Camera: wheel zooms about the cursor, middle button pans, Home resets.
    s.mark().push(Act::Move([256.0, 60.0])).wait(1).push(Act::Wheel(4.0)).wait(2);
    s.check("zoom (wheel)", |m, s| s.zoom > m.zoom * 1.5);
    s.drag(MouseButton::Middle, [256.0, 60.0], [226.0, 60.0], 10);
    s.check("pan (middle drag)", |m, s| (s.center[0] - m.center[0]).abs() > 5.0);
    s.key(Home).check("reset view (Home)", |m, s| s.zoom == m.zoom && s.center == m.center);
    s.mark().key(KeyH).wait(2).check("hide the panel (H)", |m, s| s.panel != m.panel);
    s.key(KeyH);

    // Reset, next map, and a change of resolution with fluid on screen.
    s.mark().key(KeyR).wait(5).check("reset the map (R)", |m, s| s.particles == 0 && s.shapes == 0 && s.map == m.map);
    s.mark().key(PageDown).wait(30).check("next map (PageDown)", |m, s| s.map != m.map && s.particles > 10_000);
    s.mark().push(Act::Grid([384, 216])).wait(60);
    s.check("change quality", |m, s| s.grid == [384, 216] && s.map == m.map && s.particles > 10_000 && s.substeps > 5);
    s.wait(30);
    s.acts
}
