//! The interface: floating panels over the tank, built with egui each frame.
//!
//! - `theme`: colours, shapes, fonts.
//! - `icons`: every icon, drawn as vectors.
//! - `widgets`: the controls and the motion they share.
//! - `dock`: the tool dock at the bottom (groups, tools, the options of the selected tool).
//! - `chrome`: the bars at the top, the liquid palette on the left, the settings drawer.
//! - `overlay`: what is drawn on the canvas itself (cursor, previews, devices, toasts).
//!
//! This file holds what they share: the tools and their groups, the state the interface
//! edits (`UiState`) and what it asks the app to do (`UiActions`).

mod chrome;
mod dock;
pub mod icons;
mod overlay;
pub mod theme;
pub mod widgets;

use std::collections::VecDeque;
use std::ops::RangeInclusive;

use egui::Color32;

pub use overlay::{Canvas, Overlay};

use crate::{
    scene::Shape,
    sim::{CURRENT_STRENGTH, CURRENT_STRENGTH_DEFAULT, Current, Magnet, Material},
    world::{Faucet, World},
};
use icons::Icon;

/// Default and range of `UiState::min_fps`.
pub const MIN_FPS_DEFAULT: f32 = 60.0;
pub const MIN_FPS_RANGE: RangeInclusive<f32> = 20.0..=240.0;

/// Range of the brush radius, cells.
pub const BRUSH: [f32; 2] = [2.0, 60.0];
/// Range of the half-thickness of walls, drawn lines and bowls, cells.
pub const THICKNESS: [f32; 2] = [2.0, 12.0];
/// Range of the flow of the liquid tools, particles/s.
pub const FLOW: [f32; 2] = [1000.0, 200_000.0];
/// Liquids on the palette (keys 1 to 8).
pub const PALETTE: usize = 8;
/// Height of the window's bottom that the dock and the options strip cover, points. The
/// camera fits the tank into what is above it.
pub const DOCK_INSET: f32 = 204.0;

// ---- tools ---------------------------------------------------------------------------------

/// What a group of tools does. The dock shows one group at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// Adds liquid.
    Pour,
    /// Moves the liquid that is there.
    Move,
    /// Takes things away, moves them, looks at them.
    Edit,
    /// Places solids.
    Build,
    /// Places things that keep working on their own.
    Devices,
    /// Changes temperature.
    Heat,
}

impl Group {
    pub const ALL: [Group; 6] = [Group::Pour, Group::Move, Group::Edit, Group::Build, Group::Devices, Group::Heat];

    pub fn index(self) -> usize {
        Group::ALL.iter().position(|g| *g == self).unwrap()
    }

    pub fn name(self) -> &'static str {
        match self {
            Group::Pour => "Pour",
            Group::Move => "Move",
            Group::Edit => "Edit",
            Group::Build => "Build",
            Group::Devices => "Devices",
            Group::Heat => "Heat",
        }
    }

    /// The key that selects the group (and, pressed again, steps through its tools).
    pub fn key(self) -> &'static str {
        match self {
            Group::Pour => "Q",
            Group::Move => "W",
            Group::Edit => "E",
            Group::Build => "A",
            Group::Devices => "S",
            Group::Heat => "D",
        }
    }

    pub fn blurb(self) -> &'static str {
        match self {
            Group::Pour => "Add liquid",
            Group::Move => "Stir and steer the liquid",
            Group::Edit => "Erase, remove, move, sample",
            Group::Build => "Place solids",
            Group::Devices => "Things that keep working",
            Group::Heat => "Warm it up, cool it down",
        }
    }

    pub fn icon(self) -> Icon {
        match self {
            Group::Pour => Icon::Drop,
            Group::Move => Icon::Grab,
            Group::Edit => Icon::Erase,
            Group::Build => Icon::Blocks,
            Group::Devices => Icon::Faucet,
            Group::Heat => Icon::Heat,
        }
    }

    pub fn color(self) -> Color32 {
        match self {
            Group::Pour => theme::WATER,
            Group::Move => theme::WIND,
            Group::Edit => theme::BERRY,
            Group::Build => theme::CLAY,
            Group::Devices => theme::BRASS,
            Group::Heat => theme::EMBER,
        }
    }

    pub fn tools(self) -> &'static [Tool] {
        match self {
            Group::Pour => &[Tool::Pour, Tool::Jet, Tool::Spray, Tool::Drop, Tool::Rain],
            Group::Move => &[Tool::Grab, Tool::Push, Tool::Pull, Tool::Whirl, Tool::Current, Tool::Hold],
            Group::Edit => &[Tool::Erase, Tool::Remove, Tool::Move, Tool::Pipette],
            Group::Build => &[Tool::Circle, Tool::Box, Tool::Ramp, Tool::Bowl, Tool::Wall, Tool::Draw],
            Group::Devices => &[Tool::Faucet, Tool::Drain, Tool::Pump, Tool::Magnet],
            Group::Heat => &[Tool::Heat, Tool::Cool],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
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
    Erase,
    Remove,
    Move,
    Pipette,
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
    Heat,
    Cool,
}

impl Tool {
    pub fn all() -> impl Iterator<Item = Tool> {
        Group::ALL.into_iter().flat_map(|g| g.tools().iter().copied())
    }

    pub fn group(self) -> Group {
        Group::ALL.into_iter().find(|g| g.tools().contains(&self)).unwrap()
    }

    pub fn name(self) -> &'static str {
        match self {
            Tool::Pour => "Pour",
            Tool::Jet => "Jet",
            Tool::Spray => "Spray",
            Tool::Drop => "Drop",
            Tool::Rain => "Rain",
            Tool::Grab => "Grab",
            Tool::Push => "Push",
            Tool::Pull => "Pull",
            Tool::Whirl => "Whirl",
            Tool::Current => "Current",
            Tool::Hold => "Hold",
            Tool::Erase => "Erase",
            Tool::Remove => "Remove",
            Tool::Move => "Move",
            Tool::Pipette => "Pipette",
            Tool::Circle => "Circle",
            Tool::Box => "Box",
            Tool::Ramp => "Ramp",
            Tool::Bowl => "Bowl",
            Tool::Wall => "Wall",
            Tool::Draw => "Draw",
            Tool::Faucet => "Faucet",
            Tool::Drain => "Drain",
            Tool::Pump => "Pump",
            Tool::Magnet => "Magnet",
            Tool::Heat => "Heat",
            Tool::Cool => "Cool",
        }
    }

    pub fn icon(self) -> Icon {
        match self {
            Tool::Pour => Icon::Pour,
            Tool::Jet => Icon::Jet,
            Tool::Spray => Icon::Spray,
            Tool::Drop => Icon::Drop,
            Tool::Rain => Icon::Rain,
            Tool::Grab => Icon::Grab,
            Tool::Push => Icon::Push,
            Tool::Pull => Icon::Pull,
            Tool::Whirl => Icon::Whirl,
            Tool::Current => Icon::Current,
            Tool::Hold => Icon::Hold,
            Tool::Erase => Icon::Erase,
            Tool::Remove => Icon::Remove,
            Tool::Move => Icon::Move,
            Tool::Pipette => Icon::Pipette,
            Tool::Circle => Icon::Circle,
            Tool::Box => Icon::Box,
            Tool::Ramp => Icon::Ramp,
            Tool::Bowl => Icon::Bowl,
            Tool::Wall => Icon::Wall,
            Tool::Draw => Icon::Draw,
            Tool::Faucet => Icon::Faucet,
            Tool::Drain => Icon::Drain,
            Tool::Pump => Icon::Pump,
            Tool::Magnet => Icon::Magnet,
            Tool::Heat => Icon::Heat,
            Tool::Cool => Icon::Cool,
        }
    }

    /// A key of the tool's own, for the ones used most (see `app::tool_for_key`); every
    /// tool is also reached by pressing its group's key until it comes up.
    pub fn key(self) -> Option<&'static str> {
        Some(match self {
            Tool::Jet => "J",
            Tool::Spray => "B",
            Tool::Drop => "C",
            Tool::Push => "V",
            Tool::Current => "Z",
            Tool::Remove => "X",
            Tool::Pipette => "I",
            Tool::Magnet => "M",
            Tool::Heat => "Y",
            Tool::Cool => "U",
            _ => return None,
        })
    }

    /// How to reach the tool from the keyboard, for tooltips: its own key, or its group's
    /// key and how many presses.
    pub fn key_hint(self) -> String {
        let g = self.group();
        let n = g.tools().iter().position(|t| *t == self).unwrap() + 1;
        match self.key() {
            Some(k) => k.to_string(),
            None if n == 1 => g.key().to_string(),
            None => format!("{} ×{n}", g.key()),
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tool::Pour => "Hold to pour a stream as wide as the brush.",
            Tool::Jet => "Hold for a powered stream. Ctrl + wheel aims it.",
            Tool::Spray => "Hold to throw droplets outwards from the rim of the brush.",
            Tool::Drop => "Click to drop a round blob. Works while paused.",
            Tool::Rain => "Hold to rain from a cloud three brushes wide.",
            Tool::Grab => "Drag to carry and stir. The right button grabs with any tool.",
            Tool::Push => "Hold to drive liquid away from the cursor.",
            Tool::Pull => "Hold to draw liquid in towards the cursor.",
            Tool::Whirl => "Hold to spin the liquid around the cursor. F reverses it.",
            Tool::Current => "Hold to drive liquid along the arrow. Ctrl + wheel aims it.",
            Tool::Hold => "Hold to bring liquid to rest and keep it there.",
            Tool::Erase => "Hold to remove liquid under the brush.",
            Tool::Remove => "Click a solid or a device to remove it. Delete does it with any tool.",
            Tool::Move => "Drag a solid or a device to move it.",
            Tool::Pipette => "Click liquid to pour more of it. Alt + click does it with any tool.",
            Tool::Circle => "Click to place, or drag out from the centre for a size.",
            Tool::Box => "Click to place, or drag from one corner to the opposite one (Shift: a square).",
            Tool::Ramp => "Click to place, or drag out from the centre for a size and angle.",
            Tool::Bowl => "Click to place, or drag out from the centre for a size. Ctrl + wheel tips it.",
            Tool::Wall => "Drag from one end to the other. Shift snaps to 15°.",
            Tool::Draw => "Drag to draw a solid line freehand.",
            Tool::Faucet => "Click to place a source of the selected liquid; drag to aim it.",
            Tool::Drain => "Click to place a drain: liquid that reaches it leaves the tank.",
            Tool::Pump => "Click to place a current that stays; drag to aim it.",
            Tool::Magnet => "Click to place a magnet; drag to point its north pole.",
            Tool::Heat => "Hold to heat the liquid under the brush (melts, boils away).",
            Tool::Cool => "Hold to cool the liquid under the brush (freezes, sets).",
        }
    }

    /// Tools that add the selected liquid.
    pub fn uses_material(self) -> bool {
        matches!(self, Tool::Pour | Tool::Spray | Tool::Jet | Tool::Drop | Tool::Rain | Tool::Faucet)
    }

    /// Tools with a flow rate.
    pub fn has_flow(self) -> bool {
        matches!(self, Tool::Pour | Tool::Spray | Tool::Jet | Tool::Rain | Tool::Faucet)
    }

    /// Tools whose strength is a force rather than a flow rate.
    pub fn is_force(self) -> bool {
        matches!(self, Tool::Grab | Tool::Push | Tool::Pull | Tool::Whirl | Tool::Hold)
    }

    /// Tools that place a solid.
    pub fn places_shape(self) -> bool {
        matches!(self, Tool::Circle | Tool::Box | Tool::Ramp | Tool::Bowl | Tool::Wall | Tool::Draw)
    }

    /// Tools that place something with a press, a drag and a release (`placement`).
    pub fn places(self) -> bool {
        matches!(self, Tool::Circle | Tool::Box | Tool::Ramp | Tool::Bowl | Tool::Wall | Tool::Faucet | Tool::Drain | Tool::Pump | Tool::Magnet)
    }

    /// Tools sized by the thickness instead of the brush.
    pub fn uses_thickness(self) -> bool {
        matches!(self, Tool::Wall | Tool::Draw)
    }

    /// Tools that work on the liquid under a round brush.
    pub fn has_brush(self) -> bool {
        !self.places_shape() && !matches!(self, Tool::Remove | Tool::Move | Tool::Pipette)
    }

    /// Accent colour of the tool on the canvas; liquid tools take the liquid's colour.
    pub fn color(self, liquid: Color32) -> Color32 {
        match self {
            Tool::Heat => theme::EMBER,
            Tool::Cool => theme::FROST,
            Tool::Erase | Tool::Remove | Tool::Drain => theme::DANGER,
            Tool::Magnet => theme::BERRY,
            t if t.uses_material() => theme::lifted(liquid),
            t => t.group().color(),
        }
    }
}

/// What a placed solid is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolidKind {
    /// Insulating, like the walls.
    Plain,
    /// Held hot: a hot plate.
    Hot,
    /// Held cold: a block of ice that never melts.
    Cold,
}

/// Range of the temperature of hot and cold solids, °C.
pub const SOLID_HOT: [f32; 2] = [40.0, 1400.0];
pub const SOLID_COLD: [f32; 2] = [-150.0, 15.0];

pub const GRIDS: [(&str, [u32; 2]); 4] = [
    ("Low  384 × 216", [384, 216]),
    ("Medium  512 × 288", [512, 288]),
    ("High  768 × 432", [768, 432]),
    ("Ultra  1024 × 576", [1024, 576]),
];

/// Time scales the speed chip steps through.
pub const SPEEDS: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

/// Rolling window of frame times.
#[derive(Default)]
pub struct FrameTimes {
    samples: VecDeque<f32>,
}

impl FrameTimes {
    pub fn push(&mut self, dt: f32) {
        self.samples.push_back(dt);
        while self.samples.len() > 120 {
            self.samples.pop_front();
        }
    }

    /// (mean, worst) frame time in milliseconds.
    pub fn summary(&self) -> (f32, f32) {
        if self.samples.is_empty() {
            return (0.0, 0.0);
        }
        let mean = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
        let worst = self.samples.iter().copied().fold(0.0, f32::max);
        (mean * 1e3, worst * 1e3)
    }

    /// The samples, oldest first, s.
    pub fn samples(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.samples.iter().copied()
    }
}

/// The pages of the settings drawer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    World,
    Display,
    Solver,
    Keys,
}

pub struct UiState {
    pub tool: Tool,
    /// The tool before this one (Tab swaps back).
    pub last_tool: Tool,
    /// The tool each group was last left on.
    pub group_tool: [Tool; 6],
    pub material: Material,
    /// Liquids on the palette, keys 1 to 8 (table rows).
    pub palette: [usize; PALETTE],
    /// Brush radius in cells.
    pub brush: f32,
    /// Half-thickness of walls and drawn lines, and of a bowl's wall, cells.
    pub thickness: f32,
    /// Flow of the liquid tools, particles per second.
    pub pour_rate: f32,
    /// Strength of the force tools relative to their default.
    pub force_strength: f32,
    /// Speed of the current and of pumps, as a multiple of `CURRENT_SPEED`.
    pub current_strength: f32,
    /// How fast a new drain empties, as a multiple of `DRAIN_RATE`.
    pub drain_strength: f32,
    /// The liquid picker and editor.
    pub editor: crate::editor::LiquidEditor,
    /// Field at the rim of the next magnet, T.
    pub magnet_strength: f32,
    /// North pole of the next magnet, degrees clockwise from straight up.
    pub magnet_angle: f32,
    /// Jet heading in degrees clockwise from straight down.
    pub jet_angle: f32,
    /// Heading of the held current, degrees clockwise from straight down.
    pub current_angle: f32,
    /// Heading of the next faucet and of the next pump, likewise.
    pub faucet_angle: f32,
    pub pump_angle: f32,
    /// Rotation of placed boxes, ramps and bowls, degrees anticlockwise.
    pub shape_angle: f32,
    /// Width / height ratio of placed boxes and ramps.
    pub shape_aspect: f32,
    pub shape_flip: bool,
    pub solid_kind: SolidKind,
    /// Temperature of hot and of cold solids, °C.
    pub solid_hot: f32,
    pub solid_cold: f32,
    pub whirl_clockwise: bool,
    pub paused: bool,
    pub time_scale: f32,
    /// Lowest frame rate the simulation may push the app down to before it slows simulated
    /// time instead of the frame rate, fps.
    pub min_fps: f32,
    pub vsync: bool,
    /// How the liquid and solids are drawn.
    pub look: crate::render::Look,
    /// Whether the interface is shown (H).
    pub show_panel: bool,
    /// The open page of the settings drawer.
    pub page: Option<Page>,
    pub maps_open: bool,
    pub stats_open: bool,
    pub grid: [u32; 2],
    /// Substeps taken in the last frame.
    pub substeps: u32,
    /// Simulated time delivered relative to real time; below 1 the scene is too heavy.
    pub speed: f32,
    pub frame_times: FrameTimes,
}

impl Default for UiState {
    fn default() -> Self {
        let m = |m: Material| m.index();
        UiState {
            tool: Tool::Pour,
            last_tool: Tool::Grab,
            group_tool: Group::ALL.map(|g| g.tools()[0]),
            material: Material::Water,
            palette: [m(Material::Water), m(Material::Oil), m(Material::Honey), m(Material::Lava), m(Material::Slime), m(Material::Mercury), m(Material::Blood), m(Material::Ferrofluid)],
            brush: 9.0,
            thickness: 2.5,
            pour_rate: 12000.0,
            force_strength: 1.0,
            current_strength: CURRENT_STRENGTH_DEFAULT,
            drain_strength: 1.0,
            editor: Default::default(),
            magnet_strength: crate::sim::MAGNET_STRENGTH_DEFAULT,
            magnet_angle: 0.0,
            jet_angle: 0.0,
            current_angle: 90.0,
            faucet_angle: 0.0,
            pump_angle: 90.0,
            shape_angle: 0.0,
            shape_aspect: 4.0,
            shape_flip: false,
            solid_kind: SolidKind::Plain,
            solid_hot: 300.0,
            solid_cold: -30.0,
            whirl_clockwise: false,
            paused: false,
            time_scale: 1.0,
            min_fps: MIN_FPS_DEFAULT,
            vsync: true,
            look: Default::default(),
            show_panel: true,
            page: None,
            maps_open: false,
            stats_open: false,
            grid: [512, 288],
            substeps: 0,
            speed: 1.0,
            frame_times: FrameTimes::default(),
        }
    }
}

fn wrap_degrees(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

impl UiState {
    /// Selects a tool, remembering the one before it and where its group was left.
    pub fn select(&mut self, tool: Tool) {
        if tool != self.tool {
            self.last_tool = self.tool;
            self.tool = tool;
        }
        self.group_tool[tool.group().index()] = tool;
    }

    /// A group's key: goes to the group (the tool it was left on), or, already there, on
    /// to its next tool (`back`: the one before).
    pub fn select_group(&mut self, group: Group, back: bool) {
        let tools = group.tools();
        if self.tool.group() == group {
            let i = tools.iter().position(|t| *t == self.tool).unwrap();
            let n = tools.len();
            self.select(tools[(i + if back { n - 1 } else { 1 }) % n]);
        } else {
            self.select(self.group_tool[group.index()]);
        }
    }

    /// Tab: back to the tool before this one.
    pub fn swap_tool(&mut self) {
        self.select(self.last_tool);
    }

    /// `[` `]` and Shift + wheel: the brush, or the thickness of walls and lines.
    pub fn scale_brush(&mut self, factor: f32) {
        if self.tool.uses_thickness() {
            self.thickness = (self.thickness * factor).clamp(THICKNESS[0], THICKNESS[1]);
        } else {
            self.brush = (self.brush * factor).clamp(BRUSH[0], BRUSH[1]);
        }
    }

    /// `-` `=` and Alt + wheel: the flow of liquid tools, the strength of the force tools,
    /// the speed of the current, the field of a magnet.
    pub fn scale_rate(&mut self, factor: f32) {
        match self.tool {
            Tool::Current | Tool::Pump => self.current_strength = (self.current_strength * factor).clamp(CURRENT_STRENGTH[0], CURRENT_STRENGTH[1]),
            Tool::Magnet => self.magnet_strength = (self.magnet_strength * factor).clamp(crate::sim::MAGNET_STRENGTH[0], crate::sim::MAGNET_STRENGTH[1]),
            Tool::Drain => self.drain_strength = (self.drain_strength * factor).clamp(0.1, 4.0),
            t if t.is_force() => self.force_strength = (self.force_strength * factor).clamp(0.1, 4.0),
            _ => self.pour_rate = (self.pour_rate * factor).clamp(FLOW[0], FLOW[1]),
        }
    }

    /// The angle the selected tool turns, if it has one.
    pub fn angle_mut(&mut self) -> Option<&mut f32> {
        Some(match self.tool {
            Tool::Jet => &mut self.jet_angle,
            Tool::Current => &mut self.current_angle,
            Tool::Faucet => &mut self.faucet_angle,
            Tool::Pump => &mut self.pump_angle,
            Tool::Magnet => &mut self.magnet_angle,
            Tool::Box | Tool::Ramp | Tool::Bowl => &mut self.shape_angle,
            _ => return None,
        })
    }

    /// T / G and Ctrl + wheel: turns what the selected tool aims or places. Returns whether
    /// it had anything to turn.
    pub fn rotate(&mut self, degrees: f32) -> bool {
        match self.angle_mut() {
            Some(a) => {
                *a = wrap_degrees(*a + degrees);
                true
            }
            None => false,
        }
    }

    /// F: flips the ramp, reverses the whirl.
    pub fn flip(&mut self) {
        match self.tool {
            Tool::Whirl => self.whirl_clockwise = !self.whirl_clockwise,
            _ => self.shape_flip = !self.shape_flip,
        }
    }

    /// Picks a liquid; a tool that does not add liquid gives way to the pouring tool used
    /// last.
    pub fn pick_material(&mut self, material: Material) {
        self.material = material;
        if !self.tool.uses_material() {
            self.select(self.group_tool[Group::Pour.index()]);
        }
    }

    /// The temperature new solids are held at, °C; `None` for a plain one.
    pub fn solid_temp(&self) -> Option<f32> {
        match self.solid_kind {
            SolidKind::Plain => None,
            SolidKind::Hot => Some(self.solid_hot),
            SolidKind::Cold => Some(self.solid_cold),
        }
    }
}

/// What a placing tool would put into the world.
#[derive(Clone, Copy, Debug)]
pub enum Placement {
    Solid(Shape),
    Faucet(Faucet),
    Pump(Current),
    Drain(Current),
    Magnet(Magnet),
}

/// The thing the selected tool places at `cursor`, or, while it is being dragged out from
/// `anchor` (where the button went down), with the size and heading the drag gives it.
/// `drag_min`: how far (cells) the cursor must be from the anchor for the drag to count;
/// `snap`: angles go to the nearest 15°.
pub fn placement(ui: &UiState, anchor: Option<[f32; 2]>, cursor: [f32; 2], drag_min: f32, snap: bool) -> Option<Placement> {
    let at = anchor.unwrap_or(cursor);
    let d = [cursor[0] - at[0], cursor[1] - at[1]];
    let dist = d[0].hypot(d[1]);
    let dragged = anchor.is_some() && dist > drag_min;
    let snapped = |a: f32| if snap { (a / 15.0).round() * 15.0 } else { a };
    // Heading of the drag: anticlockwise from +x (shapes), clockwise from straight down
    // (streams and currents), clockwise from straight up (magnets).
    let ccw = snapped(d[1].atan2(d[0]).to_degrees());
    let down = snapped(d[0].atan2(-d[1]).to_degrees());
    let up = snapped(d[0].atan2(d[1]).to_degrees());
    let r = ui.brush;
    let k = ui.shape_aspect.sqrt();
    let turn = if dragged { ccw } else { ui.shape_angle }.to_radians();
    let half = if dragged { [dist.max(2.0), (dist / ui.shape_aspect).max(1.5)] } else { [r * k, r / k] };
    Some(match ui.tool {
        Tool::Circle => Placement::Solid(Shape::Circle { c: at, r: if dragged { dist.max(2.0) } else { r } }),
        // A box is spanned like a selection: from the corner where the button went down to
        // the cursor, in the box's own frame (its set turn; 0: upright). Snapped, a square.
        Tool::Box if dragged => {
            let a = ui.shape_angle.to_radians();
            let (s, c) = a.sin_cos();
            let local = [d[0] * c + d[1] * s, -d[0] * s + d[1] * c];
            let mut half = [(local[0].abs() * 0.5).max(1.0), (local[1].abs() * 0.5).max(1.0)];
            if snap {
                let m = half[0].max(half[1]);
                half = [m, m];
            }
            let local_c = [half[0] * local[0].signum(), half[1] * local[1].signum()];
            let centre = [at[0] + local_c[0] * c - local_c[1] * s, at[1] + local_c[0] * s + local_c[1] * c];
            Placement::Solid(Shape::Box { c: centre, half, angle: a })
        }
        Tool::Box => Placement::Solid(Shape::Box { c: at, half, angle: turn }),
        Tool::Ramp => Placement::Solid(Shape::Ramp { c: at, half, angle: turn, flip: ui.shape_flip }),
        Tool::Bowl => {
            let radius = if dragged { dist.max(4.0) } else { r.max(4.0) };
            Placement::Solid(Shape::Bowl { c: at, r: radius, t: ui.thickness.min(radius * 0.5), angle: ui.shape_angle.to_radians() })
        }
        Tool::Wall => {
            let b = if dragged && snap {
                let a = ccw.to_radians();
                [at[0] + a.cos() * dist, at[1] + a.sin() * dist]
            } else {
                cursor
            };
            Placement::Solid(Shape::Capsule { a: at, b, r: ui.thickness })
        }
        Tool::Faucet => Placement::Faucet(Faucet::new(at, r, if dragged { down } else { ui.faucet_angle }, ui.pour_rate, ui.material)),
        Tool::Pump => Placement::Pump(Current::new(at, r, if dragged { down } else { ui.pump_angle }, ui.current_strength)),
        Tool::Drain => Placement::Drain(Current::drain(at, r, ui.drain_strength)),
        Tool::Magnet => Placement::Magnet(Magnet::new(at, r.max(2.0), if dragged { up } else { ui.magnet_angle }, ui.magnet_strength)),
        _ => return None,
    })
}

/// Things the interface asked for that the app has to carry out.
#[derive(Default)]
pub struct UiActions {
    pub reset: bool,
    pub load_map: Option<usize>,
    pub step: bool,
    pub grid: Option<[u32; 2]>,
    pub reset_view: bool,
    pub vsync_changed: bool,
    pub clean_stains: bool,
    pub clear_liquid: bool,
    pub undo: bool,
    pub redo: bool,
    pub quit: bool,
}

/// Installs fonts and visuals. Call once per egui context.
pub fn style(ctx: &egui::Context) {
    theme::apply(ctx);
}

/// A liquid's body colour from the live table (custom liquids included).
pub fn swatch_of(p: &crate::sim::SimParams, row: usize) -> Color32 {
    // (What it looks like: its dissolved colour and its particles together.)
    let c = crate::sim::material::appearance(&p.mat[row.min(crate::sim::MAX_MATERIALS - 1)], p.cell_size);
    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

/// A length in metres, in the unit that suits it.
pub fn length_unit(metres: f32) -> (f32, &'static str) {
    if metres < 0.01 { (1e-3, "mm") } else if metres < 1.0 { (1e-2, "cm") } else { (1.0, "m") }
}

/// Builds the interface for this frame. It slides and fades in and out with `show_panel`.
pub fn build(ctx: &egui::Context, state: &mut UiState, world: &mut World, swatches: &mut crate::editor::Swatches) -> UiActions {
    let mut a = UiActions::default();
    let shown = widgets::glide_bool(ctx, egui::Id::new("ui_shown"), state.show_panel, 14.0);
    if shown < 0.002 {
        return a;
    }
    chrome::top_left(ctx, state, world, &mut a, shown);
    chrome::top_right(ctx, state, world, &mut a, shown);
    chrome::palette(ctx, state, world, swatches, shown);
    dock::dock(ctx, state, world, shown);
    if state.show_panel {
        chrome::maps(ctx, state, world, &mut a);
        chrome::stats(ctx, state, world);
        chrome::drawer(ctx, state, world, &mut a);
        if let Some(row) = crate::editor::picker(ctx, &mut state.editor, swatches, world, state.material.index()) {
            state.pick_material(Material(row as u8));
        }
        crate::editor::editor(ctx, &mut state.editor, swatches, world);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_is_in_exactly_one_group() {
        let all: Vec<Tool> = Tool::all().collect();
        assert_eq!(all.len(), 27);
        for (i, t) in all.iter().enumerate() {
            assert!(!all[..i].contains(t), "{} is listed twice", t.name());
            assert!(t.group().tools().contains(t));
            assert!(!t.hint().is_empty() && !t.key_hint().is_empty());
        }
    }

    #[test]
    fn a_group_key_steps_through_its_tools_and_remembers_where_it_was() {
        let mut ui = UiState::default();
        ui.select_group(Group::Move, false);
        assert_eq!(ui.tool, Tool::Grab);
        ui.select_group(Group::Move, false);
        ui.select_group(Group::Move, false);
        assert_eq!(ui.tool, Tool::Pull);
        ui.select_group(Group::Move, true);
        assert_eq!(ui.tool, Tool::Push);
        ui.select_group(Group::Build, false);
        assert_eq!(ui.tool, Tool::Circle);
        ui.select_group(Group::Move, false);
        assert_eq!(ui.tool, Tool::Push, "the group comes back on the tool it was left on");
        ui.swap_tool();
        assert_eq!(ui.tool, Tool::Circle);
        // The last tool of a group wraps round to the first.
        ui.select(Tool::Cool);
        ui.select_group(Group::Heat, false);
        assert_eq!(ui.tool, Tool::Heat);
        // Picking a liquid with a tool that adds none goes back to the pouring tool last used.
        ui.select(Tool::Spray);
        ui.select(Tool::Erase);
        ui.pick_material(Material::Oil);
        assert_eq!(ui.tool, Tool::Spray);
    }

    #[test]
    fn dragging_out_a_placement_sets_its_size_and_heading() {
        let mut ui = UiState::default();
        ui.select(Tool::Box);
        // A click: the brush size and the set angle.
        let Some(Placement::Solid(Shape::Box { half, angle, .. })) = placement(&ui, Some([50.0, 50.0]), [50.5, 50.0], 1.0, false) else { panic!() };
        assert!((half[0] - 18.0).abs() < 1e-4 && (half[1] - 4.5).abs() < 1e-4 && angle == 0.0);
        // A drag spans the box from the corner it started at to the cursor, upright.
        let Some(Placement::Solid(Shape::Box { c, half, angle })) = placement(&ui, Some([50.0, 50.0]), [70.0, 40.0], 1.0, false) else { panic!() };
        assert!((c[0] - 60.0).abs() < 1e-4 && (c[1] - 45.0).abs() < 1e-4 && (half[0] - 10.0).abs() < 1e-4 && (half[1] - 5.0).abs() < 1e-4 && angle == 0.0);
        // Snapped: a square, its corner still at the anchor.
        let Some(Placement::Solid(Shape::Box { c, half, .. })) = placement(&ui, Some([50.0, 50.0]), [70.0, 40.0], 1.0, true) else { panic!() };
        assert!((half[0] - 10.0).abs() < 1e-4 && (half[1] - 10.0).abs() < 1e-4 && (c[0] - 60.0).abs() < 1e-4 && (c[1] - 40.0).abs() < 1e-4);
        // Turned, it is spanned in its own frame: a drag along the turned axis is its length.
        ui.shape_angle = 90.0;
        let Some(Placement::Solid(Shape::Box { c, half, .. })) = placement(&ui, Some([50.0, 50.0]), [44.0, 70.0], 1.0, false) else { panic!() };
        assert!((half[0] - 10.0).abs() < 1e-3 && (half[1] - 3.0).abs() < 1e-3 && (c[0] - 47.0).abs() < 1e-3 && (c[1] - 60.0).abs() < 1e-3);
        ui.shape_angle = 0.0;
        // A ramp dragged up and to the right: 45 degrees, as long as the drag.
        ui.select(Tool::Ramp);
        let Some(Placement::Solid(Shape::Ramp { half, angle, .. })) = placement(&ui, Some([50.0, 50.0]), [70.0, 70.0], 1.0, false) else { panic!() };
        assert!((half[0] - 800f32.sqrt()).abs() < 1e-3 && (angle - std::f32::consts::FRAC_PI_4).abs() < 1e-4);
        // Snapped: 40 degrees goes to 45.
        let c = [50.0 + 30.0 * 40f32.to_radians().cos(), 50.0 + 30.0 * 40f32.to_radians().sin()];
        let Some(Placement::Solid(Shape::Ramp { angle, .. })) = placement(&ui, Some([50.0, 50.0]), c, 1.0, true) else { panic!() };
        assert!((angle.to_degrees() - 45.0).abs() < 1e-3);
        // A faucet dragged to the right points right (90 degrees clockwise from down).
        ui.select(Tool::Faucet);
        let Some(Placement::Faucet(f)) = placement(&ui, Some([50.0, 50.0]), [80.0, 50.0], 1.0, false) else { panic!() };
        assert!((f.angle - 90.0).abs() < 1e-3 && f.pos == [50.0, 50.0]);
        // A magnet dragged upwards points north up.
        ui.select(Tool::Magnet);
        let Some(Placement::Magnet(m)) = placement(&ui, Some([50.0, 50.0]), [50.0, 80.0], 1.0, false) else { panic!() };
        assert!(m.dir[0].abs() < 1e-4 && (m.dir[1] - 1.0).abs() < 1e-4);
        // Tools that place nothing.
        ui.select(Tool::Pour);
        assert!(placement(&ui, None, [10.0, 10.0], 1.0, false).is_none());
    }
}
