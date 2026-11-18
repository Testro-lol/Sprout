//! A scripted session for the real window: `sprout --selftest`.
//!
//! The physics checks in `validate` run off-screen. This drives the interactive app
//! instead: a script feeds the same handlers that mouse and keyboard events reach, frame
//! by frame, while the window renders, and checks what each tool did to the world.

use winit::{
    event::MouseButton,
    keyboard::{KeyCode, ModifiersState},
};

use crate::{scene::WALL, ui::Tool};

/// Length of the frames the script is written in, s. The app holds its frame rate to this
/// while the script runs, whatever the display's refresh rate.
pub const FRAME: f64 = 1.0 / 60.0;

/// What the script can see of the app.
#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub particles: usize,
    pub oil: usize,
    /// Honey content, in particles.
    pub honey: f32,
    pub rms_speed: f32,
    pub shapes: usize,
    pub substeps: u64,
    pub grid: [u32; 2],
    pub map: usize,
    pub zoom: f32,
    pub center: [f32; 2],
    pub brush: f32,
    pub currents: usize,
    /// Mean horizontal velocity of the liquid, cells/s.
    pub drift: f32,
    pub jet_angle: f32,
    pub jet_speed: f32,
    pub radial_speed: f32,
    pub panel: bool,
    /// Custom liquids in the library (not deleted), and ids in the saved file.
    pub custom: usize,
    pub file_ids: usize,
    /// The editor's liquid: its row, its particles in the tank, its live and draft viscosity,
    /// whether the saved file holds exactly the live definition.
    pub edit_row: usize,
    pub edit_particles: usize,
    pub live_visc: f32,
    pub draft_visc: f32,
    pub file_matches: bool,
    /// Particles of water, magnets placed, mean temperature of the liquid (deg C).
    pub water: usize,
    pub magnets: usize,
    pub mean_t: f32,
    /// Name of the selected tool, and the row of the selected liquid.
    pub tool: &'static str,
    pub material: usize,
    /// Devices placed, and whether there is an edit to undo.
    pub faucets: usize,
    pub drains: usize,
    pub pumps: usize,
    pub can_undo: bool,
    /// Mean angular momentum of the liquid about the cursor (anticlockwise positive), and
    /// its mean height, cells.
    pub spin: f32,
    pub mean_y: f32,
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
    /// Selects a tool, as clicking it in the dock does.
    Tool(Tool),
    /// Holds these modifier keys from now on.
    Mods(ModifiersState),
    /// Sets tool options, as the sliders of the options strip do.
    Tweak(fn(&mut crate::ui::UiState)),
    /// Picks a grid resolution, as the quality menu does.
    Grid([u32; 2]),
    Mark,
    Check(&'static str, Check),
    /// What the liquid picker and editor buttons do (editor.rs).
    Liquid(LiquidAct),
}

#[derive(Clone, Copy)]
pub enum LiquidAct {
    /// Edit (a built-in is duplicated first), or Duplicate on a custom liquid.
    Duplicate(usize),
    /// Changes the draft, as the editor's controls do.
    Edit(fn(&mut crate::sim::MatProps)),
    Apply,
    Save,
    Delete,
    /// Picks the editor's liquid to pour, as clicking it in the picker does.
    Pick,
    /// Opens the editor on a row without duplicating it (for a built-in: read-only).
    Select(usize),
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

    fn tool(&mut self, tool: Tool) -> &mut Self {
        self.push(Act::Tool(tool)).wait(1)
    }

    fn mods(&mut self, mods: ModifiersState) -> &mut Self {
        self.push(Act::Mods(mods)).wait(1)
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
    use MouseButton::{Left, Middle, Right};
    let mut s = Script { frame: 20, acts: Vec::new() };
    let floor = WALL + 6.0;
    let none = ModifiersState::empty();

    // Pour each fluid (1, 2, 3 = water, oil, honey).
    s.mark().tool(Tool::Pour).key(Digit1).drag(Left, [200.0, 150.0], [300.0, 150.0], 150);
    s.check("pour water (1, drag)", |m, s| s.tool == "Pour" && s.particles > m.particles + 2000 && s.oil == 0 && s.honey == 0.0);
    s.mark().key(Digit2).drag(Left, [120.0, 120.0], [180.0, 120.0], 90);
    s.check("pour oil (2)", |m, s| s.oil > 1000 && s.particles >= m.particles + s.oil);
    s.mark().key(Digit3).drag(Left, [340.0, 120.0], [400.0, 120.0], 90);
    s.check("pour honey (3)", |m, s| s.honey > 1000.0 && s.oil == m.oil);

    // Keys: a group's key steps through its tools and comes back where it was; Tab swaps.
    s.key(KeyW).check("W opens Move on Grab", |_, s| s.tool == "Grab");
    s.key(KeyW).key(KeyW).check("W again steps through the group", |_, s| s.tool == "Pull");
    s.mods(ModifiersState::SHIFT).key(KeyW).mods(none).check("Shift + W steps back", |_, s| s.tool == "Push");
    s.key(KeyA).check("A opens Build on Circle", |_, s| s.tool == "Circle");
    s.key(KeyW).check("a group comes back on the tool it was left on", |_, s| s.tool == "Push");
    s.key(Tab).check("Tab goes back to the tool before", |_, s| s.tool == "Circle");
    s.key(KeyE).check("E opens Edit on Erase", |_, s| s.tool == "Erase");
    s.key(KeyQ).check("Q opens Pour", |_, s| s.tool == "Pour");
    s.key(KeyQ).check("Q again goes on to Jet", |_, s| s.tool == "Jet");
    s.tool(Tool::Erase);
    s.key(Digit1).check("picking a liquid goes back to the pouring tool used last", |_, s| s.tool == "Jet" && s.material == crate::sim::Material::Water.index());

    s.tool(Tool::Pour);
    // Brush size keys change what the nozzle emits.
    s.mark().key(BracketRight).key(BracketRight).check("brush size (])", |m, s| s.brush > m.brush * 1.2);
    s.key(BracketLeft).key(BracketLeft);
    s.mark().mods(ModifiersState::SHIFT).push(Act::Wheel(2.0)).wait(1).mods(none).check("Shift + wheel sizes the brush, without zoom", |m, s| s.brush > m.brush * 1.2 && s.zoom == m.zoom);
    s.mods(ModifiersState::SHIFT).push(Act::Wheel(-2.0)).wait(1).mods(none);

    // Place a circle, a rotated box and a flipped ramp (the Build group; T rotates, F flips).
    s.wait(60).mark();
    s.tool(Tool::Circle).click([256.0, 40.0]).tool(Tool::Box).key(KeyT).click([120.0, 60.0]).tool(Tool::Ramp).key(KeyF).click([400.0, 60.0]);
    s.wait(30).check("place circle, box and ramp", |m, s| s.shapes == m.shapes + 3 && s.particles == m.particles && s.can_undo);

    // Stir with the grab tool and with the right mouse button.
    s.wait(240).mark().tool(Tool::Grab).drag(Left, [60.0, floor], [200.0, floor], 30);
    s.check("grab / stir (drag)", |m, s| s.rms_speed > m.rms_speed && s.particles == m.particles);
    s.mark().tool(Tool::Pour).drag(Right, [200.0, floor], [60.0, floor], 30);
    s.check("stir with the right button", |m, s| s.particles == m.particles);

    // Erase, then remove a solid; undo brings it back, redo takes it away again.
    s.mark().tool(Tool::Erase).drag(Left, [300.0, floor], [360.0, floor], 30);
    s.check("erase (drag)", |m, s| s.particles + 100 < m.particles);
    s.mark().key(KeyX).click([256.0, 40.0]);
    s.check("remove a solid (X, click)", |m, s| s.tool == "Remove" && s.shapes + 1 == m.shapes && s.particles == m.particles);
    s.mark().mods(ModifiersState::CONTROL).key(KeyZ).mods(none).wait(2);
    s.check("undo brings it back (Ctrl + Z)", |m, s| s.shapes == m.shapes + 1 && s.particles == m.particles && s.tool == "Remove");
    s.mark().mods(ModifiersState::CONTROL | ModifiersState::SHIFT).key(KeyZ).mods(none).wait(2);
    s.check("redo removes it again (Ctrl + Shift + Z)", |m, s| s.shapes + 1 == m.shapes);
    s.mark().tool(Tool::Pour).push(Act::Move([120.0, 60.0])).wait(2).key(Delete).wait(2);
    s.check("Delete removes what is under the cursor, with any tool", |m, s| s.shapes + 1 == m.shapes && s.tool == "Pour");

    // Pause, single-step, resume.
    s.key(Space).wait(10).mark().wait(20).check("pause (Space)", |m, s| s.substeps == m.substeps);
    s.mark().key(KeyN).wait(3).check("single step (N)", |m, s| s.substeps == m.substeps + 1);
    s.mark().key(Space).wait(30).check("resume (Space)", |m, s| s.substeps > m.substeps + 5);

    // Camera: wheel zooms about the cursor, middle button pans, Home resets.
    s.mark().push(Act::Move([256.0, 60.0])).wait(1).push(Act::Wheel(4.0)).wait(2);
    s.check("zoom (wheel)", |m, s| s.zoom > m.zoom * 1.5);
    s.drag(Middle, [256.0, 60.0], [226.0, 60.0], 10);
    s.check("pan (middle drag)", |m, s| (s.center[0] - m.center[0]).abs() > 5.0);
    s.key(Home).check("reset view (Home)", |m, s| s.zoom == m.zoom && s.center == m.center);
    // The view eases back; let it settle before clicking at positions in the grid again.
    s.wait(40);
    s.mark().key(KeyH).wait(40).check("hide the interface (H): the tank fills the window", |m, s| s.panel != m.panel && s.zoom > m.zoom * 1.1);
    // The view glides back above the dock; let it settle, as after Home.
    s.mark().key(KeyH).wait(40).check("and makes room for the dock again", |m, s| s.zoom < m.zoom / 1.1);

    // Emitters: a single stamp while paused, and continuous spray / jet / rain.
    s.key(Space).key(KeyC).key(Digit1).mark().click([360.0, 180.0]);
    s.check("drop stamps while paused (C)", |m, s| s.tool == "Drop" && s.particles > m.particles + 500 && s.substeps == m.substeps);
    s.mark().wait(10).check("drop does not repeat", |m, s| s.particles == m.particles);
    // (Round-off in the scripted pointer may bring an empty rim cell or two into the disk.)
    s.mark().click([360.0, 180.0]).check("repeat drop does not overfill", |m, s| s.particles < m.particles + 20);
    s.key(Space).key(KeyB).mark().drag(Left, [340.0, 180.0], [400.0, 180.0], 20);
    s.check("spray emits (B)", |m, s| s.particles > m.particles);
    s.key(KeyJ).mark().push(Act::Wheel(2.0)).wait(2).check("the wheel zooms with every tool", |m, s| s.jet_angle == m.jet_angle && s.zoom > m.zoom);
    s.key(Home).wait(40);
    s.mark().mods(ModifiersState::CONTROL).push(Act::Wheel(6.0)).wait(2).mods(none);
    s.check("Ctrl + wheel aims the jet without zoom", |m, s| (s.jet_angle - m.jet_angle - 90.0).abs() < 0.01 && s.zoom == m.zoom);
    s.mark().key(KeyG).check("G aims the jet back (J)", |m, s| (s.jet_angle - m.jet_angle + 15.0).abs() < 0.01);
    s.mark().drag(Left, [340.0, 180.0], [400.0, 180.0], 20);
    s.check("jet emits (J)", |m, s| s.particles > m.particles);
    s.tool(Tool::Rain).mark().drag(Left, [200.0, 220.0], [300.0, 220.0], 40);
    s.check("rain emits", |m, s| s.particles > m.particles + 50);

    // Liquids: choose a preset, duplicate it, edit the draft, preview, apply, save, pour.
    s.key(KeyR).wait(5).tool(Tool::Pour).key(Digit1).drag(Left, [100.0, 150.0], [160.0, 150.0], 40).wait(30);
    s.mark().push(Act::Liquid(LiquidAct::Duplicate(crate::sim::Material::Honey.index()))).wait(3);
    s.check("duplicate a preset: a new liquid, a draft not saved", |m, s| s.custom == m.custom + 1 && s.edit_row >= crate::sim::BUILTINS.len() && s.file_ids == m.file_ids);
    s.mark().push(Act::Liquid(LiquidAct::Edit(|p| {
        p.viscosity = 0.3;
        p.color = [0.2, 0.8, 0.3];
    })))
    .wait(10);
    s.check("editing changes the draft only", |m, s| s.draft_visc == 0.3 && s.live_visc == m.live_visc);
    s.mark().push(Act::Liquid(LiquidAct::Apply)).wait(3).check("apply: the live table takes the draft", |_, s| s.live_visc == 0.3 && !s.file_matches);
    s.mark().push(Act::Liquid(LiquidAct::Save)).wait(3).check("save: written to the file, read back the same", |m, s| s.file_ids == m.file_ids + 1 && s.file_matches);
    s.mark().push(Act::Liquid(LiquidAct::Pick)).wait(2).drag(Left, [300.0, 150.0], [360.0, 150.0], 60).wait(10);
    // (A honey copy dissolves in water: water around it joins it as it mixes.)
    s.check("pour the new liquid", |m, s| s.edit_particles > 500 && s.particles > m.particles + 500);
    s.mark().push(Act::Liquid(LiquidAct::Edit(|p| p.viscosity = 0.01))).push(Act::Liquid(LiquidAct::Apply)).wait(5);
    s.check("apply again: the liquid in the tank follows", |m, s| s.live_visc == 0.01 && s.edit_particles > 0 && s.particles == m.particles);
    s.mark().push(Act::Liquid(LiquidAct::Duplicate(crate::sim::Material::Water.index()))).wait(3);
    s.check("editing a built-in duplicates it; the water in the tank keeps its material", |m, s| s.edit_row != m.edit_row && s.edit_particles == 0 && s.water > 0 && s.particles == m.particles);
    s.mark().push(Act::Liquid(LiquidAct::Select(crate::sim::Material::Water.index()))).push(Act::Liquid(LiquidAct::Edit(|p| p.viscosity = 0.2))).push(Act::Liquid(LiquidAct::Apply)).wait(3);
    s.check("built-ins are read-only", |_, s| s.live_visc == 1.0e-6);
    s.mark().push(Act::Liquid(LiquidAct::Duplicate(crate::sim::Material::Oil.index()))).push(Act::Liquid(LiquidAct::Delete)).wait(3);
    s.check("delete a custom liquid (and only it leaves the file)", |m, s| s.custom == m.custom && s.file_ids == m.file_ids);

    // The pipette: water is selected, the oil under the cursor becomes the liquid to pour.
    s.key(KeyR).wait(5).tool(Tool::Pour).key(Digit2).drag(Left, [240.0, 60.0], [270.0, 60.0], 60).wait(90).key(Digit1);
    s.mark().key(KeyI).click([256.0, floor - 3.0]);
    s.check("the pipette picks the liquid under it (I)", |_, s| s.tool == "Pipette" && s.material == crate::sim::Material::Oil.index());
    s.key(Digit1).mark().mods(ModifiersState::ALT).click([256.0, floor - 3.0]).mods(none);
    s.check("Alt + click samples with any tool", |m, s| s.tool == "Pour" && s.material == crate::sim::Material::Oil.index() && s.particles == m.particles);

    // Heat brush and magnets.
    s.key(KeyR).wait(5).tool(Tool::Pour).key(Digit1).drag(Left, [200.0, 100.0], [300.0, 100.0], 60).wait(150);
    s.mark().key(KeyY).drag(Left, [230.0, 6.0], [280.0, 6.0], 90).wait(5);
    s.check("heat brush warms the liquid (Y)", |m, s| s.mean_t > m.mean_t + 1.0);
    s.mark().key(KeyM).click([256.0, 200.0]).check("place a magnet (M)", |m, s| s.magnets == m.magnets + 1);
    s.mark().key(KeyX).click([256.0, 200.0]).check("remove it (X)", |m, s| s.magnets + 1 == m.magnets);

    // Reset, next map, and a change of resolution with fluid on screen.
    s.mark().key(KeyR).wait(5).check("reset the map (R)", |m, s| s.particles == 0 && s.shapes == 0 && s.map == m.map && !s.can_undo);
    // Isolate repulsion from gravity and other liquid: a fresh round blob.
    s.key(Space).key(KeyC).click([360.0, 180.0]).key(KeyV).mark();
    s.key(Space).push(Act::Press(Left)).wait(8).push(Act::Release(Left));
    s.check("push repels without adding liquid (V)", |m, s| s.particles == m.particles && s.radial_speed > m.radial_speed + 10.0);
    // The same blob, pulled: it moves inwards.
    s.key(KeyR).wait(5).key(Space).key(KeyC).push(Act::Tweak(|u| u.brush = 14.0)).click([360.0, 180.0]).tool(Tool::Pull).push(Act::Tweak(|u| u.brush = 24.0)).mark();
    s.key(Space).push(Act::Press(Left)).wait(6).push(Act::Release(Left));
    s.check("pull draws liquid in", |m, s| s.particles == m.particles && s.radial_speed < m.radial_speed - 10.0);
    // Whirl turns the liquid under the brush anticlockwise, and clockwise after F: in a
    // pool deep enough to hold the whole brush.
    s.key(KeyR).wait(5).push(Act::Tweak(|u| u.brush = 9.0)).tool(Tool::Pour).drag(Left, [200.0, 100.0], [300.0, 100.0], 330).wait(150);
    s.tool(Tool::Whirl).push(Act::Tweak(|u| u.brush = 12.0)).push(Act::Move([256.0, WALL + 16.0])).wait(2).mark().push(Act::Press(Left)).wait(40);
    s.check("whirl spins the liquid", |m, s| s.spin > m.spin + 300.0 && s.particles == m.particles);
    s.push(Act::Release(Left)).wait(90).key(KeyF).mark().push(Act::Press(Left)).wait(40);
    s.check("F reverses the whirl", |m, s| s.spin < m.spin - 300.0 && s.spin < -300.0);
    s.push(Act::Release(Left)).wait(2);
    // Hold: a blob dropped into a held brush stays up; let go and it falls.
    s.key(KeyR).wait(5).push(Act::Tweak(|u| u.brush = 12.0)).key(KeyC).click([256.0, 150.0]).tool(Tool::Hold).push(Act::Tweak(|u| u.brush = 22.0));
    s.push(Act::Move([256.0, 150.0])).wait(1).push(Act::Press(Left)).wait(10).mark().wait(40);
    s.check("hold keeps liquid where it is", |m, s| (s.mean_y - m.mean_y).abs() < 4.0 && s.particles == m.particles);
    s.push(Act::Release(Left)).mark().wait(50).check("let go, it falls", |m, s| s.mean_y < m.mean_y - 30.0);
    s.push(Act::Tweak(|u| u.brush = 9.0));
    // A blob dropped through the Current tool held pointing right (Z) is carried right;
    // releasing the button ends the current.
    s.key(KeyR).wait(5).mark().key(KeyC).click([256.0, 150.0]).key(KeyZ).push(Act::Move([256.0, 110.0]));
    s.push(Act::Press(Left)).wait(30);
    s.check("a held current carries liquid along (Z)", |m, s| s.currents == 1 && s.particles > m.particles && s.drift > 5.0);
    s.mark().push(Act::Release(Left)).wait(3);
    s.check("releasing it ends the current", |m, s| s.currents == 0 && s.particles == m.particles);

    // Devices: a faucet keeps pouring on its own, a drain takes the liquid away, a pump stays.
    s.key(KeyR).wait(5).key(Digit1).mark().tool(Tool::Faucet).click([150.0, 200.0]);
    s.check("place a faucet", |m, s| s.tool == "Faucet" && s.faucets == m.faucets + 1);
    s.mark().wait(90).check("the faucet pours with no button held", |m, s| s.particles > m.particles + 2000 && s.water == s.particles);
    s.key(Space).wait(3).mark().wait(20).check("and stops while paused", |m, s| s.particles == m.particles);
    s.key(Space).tool(Tool::Drain).push(Act::Tweak(|u| u.brush = 16.0)).mark().click([150.0, floor]);
    s.check("place a drain", |m, s| s.drains == m.drains + 1 && s.currents == 1);
    s.tool(Tool::Remove).mark().click([150.0, 200.0]).wait(120);
    s.check("remove the faucet; the drain empties the tank", |m, s| s.faucets + 1 == m.faucets && s.particles * 2 < m.particles);
    s.push(Act::Tweak(|u| u.brush = 9.0)).tool(Tool::Pump).mark().drag(Left, [300.0, 60.0], [300.0, 100.0], 6);
    s.check("place a pump, dragged to aim it up", |m, s| s.pumps == m.pumps + 1 && s.currents == m.currents + 1);
    // Walls, drawn lines and bowls; moving a solid; backing out of a drag.
    s.key(KeyR).wait(5).mark().tool(Tool::Wall).drag(Left, [100.0, 40.0], [180.0, 80.0], 8);
    s.check("drag a wall from end to end", |m, s| s.shapes == m.shapes + 1);
    s.mark().tool(Tool::Draw).drag(Left, [220.0, 40.0], [300.0, 60.0], 20);
    s.check("draw a line freehand", |m, s| s.shapes > m.shapes + 5);
    s.mark().mods(ModifiersState::CONTROL).key(KeyZ).mods(none).wait(2);
    s.check("one undo takes the whole line back", |m, s| s.shapes == 1 && s.shapes < m.shapes);
    s.mark().tool(Tool::Bowl).drag(Left, [380.0, 80.0], [410.0, 80.0], 6);
    s.check("drag out a bowl", |m, s| s.shapes == m.shapes + 1);
    s.tool(Tool::Drop).push(Act::Tweak(|u| u.brush = 12.0)).click([380.0, 78.0]).wait(120).mark();
    s.check("the bowl holds what is dropped into it", |_, s| s.particles > 300 && s.mean_y > 45.0);
    // (Slowly: a solid has no velocity, it is rasterised anew where it is now, up to 60
    // times a second. Moved by more than half its thickness at a time it passes through
    // the liquid instead of lifting it.)
    s.tool(Tool::Move).mark().drag(Left, [380.0, 50.5], [380.0, 120.5], 100).wait(60);
    s.check("move the bowl slowly: the liquid in it comes along", |m, s| s.shapes == m.shapes && s.mean_y > m.mean_y + 40.0);
    s.tool(Tool::Box).mark().push(Act::Move([200.0, 150.0])).wait(2).push(Act::Press(Left)).wait(2).push(Act::Move([240.0, 150.0])).wait(2);
    s.push(Act::Press(Right)).push(Act::Release(Right)).wait(1).push(Act::Release(Left)).wait(2);
    s.check("the right button backs out of a drag", |m, s| s.shapes == m.shapes);
    s.push(Act::Tweak(|u| u.brush = 9.0));

    s.key(KeyR).wait(5).key(KeyJ).push(Act::Move([256.0, 140.0])).wait(2).push(Act::Press(Left));
    s.wait(120).mark().wait(360);
    s.check("held jet keeps outlet power after 3s", |m, s| s.jet_speed > 100.0 && s.jet_speed > m.jet_speed * 0.7 && s.particles > m.particles);
    s.push(Act::Release(Left)).key(KeyR).wait(5);
    s.mark().key(PageDown).wait(30).check("next map (PageDown)", |m, s| s.map != m.map && s.particles > 10_000);
    s.mark().mods(ModifiersState::SHIFT).key(KeyR).mods(none).wait(5);
    s.check("clear the liquid, keep the solids (Shift + R)", |m, s| s.particles == 0 && s.shapes == m.shapes && s.shapes > 0 && s.map == m.map);
    s.key(KeyR).wait(30);
    s.mark().push(Act::Grid([384, 216])).wait(60);
    s.check("change quality", |m, s| s.grid == [384, 216] && s.map == m.map && s.particles > 10_000 && s.substeps > 5);
    s.wait(30);
    s.acts
}
