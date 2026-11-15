//! Plain-data types shared between the CPU and the simulation shaders, and the one place
//! where physical material properties live.
//!
//! Everything the user can edit is stored in SI units (metres, seconds, N/m) together with
//! the physical size of one grid cell. The solver works in grid units (cells, seconds,
//! densities relative to water); `SimParams::gpu_units` is the only conversion.

use bytemuck::{Pod, Zeroable};

/// Particles per cell at rest (a 2 x 2 lattice).
pub const PPC: u32 = 4;

/// Length of one substep in the legacy fixed-step API (`FluidSim::step_blocking`), which
/// tests and tools use to say "advance this many 1/240 s ticks".
pub const NOMINAL_DT: f32 = 1.0 / 240.0;

pub use super::material::*;

/// `Particle::kind` of an empty slot. A live particle holds its material's table row + 1 in
/// the low byte.
pub const KIND_DEAD: u32 = 0;
/// Set on particles handed to `FluidSim::spawn` until their first substep: they did not
/// scatter to the grid yet and take no share of its heat exchange (particles.comp).
pub const KIND_FRESH: u32 = 0x10000;
/// A puff of steam that escaped from the liquid (particles.comp, README "Steam"): drawn,
/// not simulated. Its low byte is 0, so it has no material and counts nowhere; `s` holds its
/// age, s.
pub const KIND_PUFF: u32 = 0x40000;
/// Of steam: the table row + 1 of the liquid it boiled from, in bits 8 - 15 (particles.comp);
/// it condenses back into that liquid, at the concentration `s` keeps.
pub const KIND_SRC: u32 = 0xff00;
const KIND_MAT_MASK: u32 = 0xff;

/// Must match `struct Particle` in `shaders/common.glsl` (std430, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub c: [f32; 4],
    /// Concentration: the share of the particle that is its own material, the rest being
    /// that material's diluent (`MatProps::diluent`). Ignored for materials without one.
    pub s: f32,
    /// 0 = dead slot; otherwise the material's table row + 1.
    pub kind: u32,
    /// Permanent identity, assigned by `FluidSim` when the particle enters the tank. It
    /// survives the periodic sort and seeds the per-particle random numbers, so that a run
    /// does not depend on the order particles happen to sit in memory.
    pub id: u32,
    /// Enthalpy temperature, °C: heat content over heat capacity, counted so that it is the
    /// temperature in the liquid state; below the solidus it runs behind the temperature by
    /// the latent heat (`phase_of` in common.glsl, `MatProps::enthalpy_temp`).
    pub t: f32,
}

impl Particle {
    /// A particle of `material` as it is emitted: its temperature (`t` = `SPAWN_TEMP`) and,
    /// for a suspension, whether it is a grain or the liquid around it (`s` = `SPAWN_S`) are
    /// filled in from the material table when it enters the tank (`FluidSim::spawn`).
    pub fn new(material: Material, pos: [f32; 2], vel: [f32; 2]) -> Self {
        Particle { pos, vel, s: SPAWN_S, kind: material.0 as u32 + 1, t: SPAWN_TEMP, ..Default::default() }
    }

    /// The same particle at temperature `t` (°C), frozen if that is below its melting range.
    pub fn at_temp(self, t: f32) -> Self {
        Particle { t, kind: self.kind | KIND_REAL_T, ..self }
    }

    /// Fills in what `new` left to the material table (see there). `id` must be set.
    pub fn settle_spawn(&mut self, p: &SimParams) {
        let Some(m) = self.material() else { return };
        let def = &p.mat[m.index()];
        if self.t == SPAWN_TEMP {
            self.t = def.spawn_temp;
            self.kind |= KIND_REAL_T;
        }
        if self.s == SPAWN_S {
            // A concentrate is emitted at its `load` (mud: half packed silt, half water).
            self.s = if def.diluent >= 0 { def.load.clamp(0.0, 1.0) } else { 1.0 };
        }
        if self.kind & KIND_REAL_T != 0 {
            self.kind &= !KIND_REAL_T;
            self.t = p.enthalpy_temp(m, self.s, self.t);
        }
    }
}

/// Set by `Particle::at_temp`: `t` holds a temperature, to be turned into an enthalpy
/// temperature when the particle enters the tank.
pub const KIND_REAL_T: u32 = 0x20000;

/// `Particle::t` of a particle that takes its material's spawn temperature.
pub const SPAWN_TEMP: f32 = -1.0e4;
/// `Particle::s` of a particle that takes its material's emitted concentration.
pub const SPAWN_S: f32 = -1.0;

impl Particle {

    /// Its material, or `None` for a dead slot or a puff of steam.
    pub fn material(&self) -> Option<Material> {
        (self.kind != KIND_DEAD && self.kind & KIND_MAT_MASK != 0).then(|| Material(((self.kind & KIND_MAT_MASK) - 1) as u8))
    }

    /// A puff of escaped steam (`KIND_PUFF`): drawn, but no longer part of the tank.
    pub fn is_puff(&self) -> bool {
        self.kind & KIND_PUFF != 0 && self.kind & KIND_MAT_MASK == 0
    }

    pub fn is(&self, m: Material) -> bool {
        self.material() == Some(m)
    }

    /// Volume share of `m` in the particle: its concentration if it is `m`, the remainder
    /// if `m` is its diluent.
    pub fn share(&self, m: Material, p: &SimParams) -> f32 {
        match self.material() {
            Some(own) if own == m => {
                if p.mat[m.index()].diluent >= 0 { self.s } else { 1.0 }
            }
            Some(own) if p.mat[own.index()].diluent == m.0 as i32 => 1.0 - self.s,
            _ => 0.0,
        }
    }

    /// Relative density (water = 1) at the reference temperature.
    pub fn density(&self, p: &SimParams) -> f32 {
        let Some(m) = self.material() else { return 0.0 };
        let own = &p.mat[m.index()];
        let rho = if own.diluent >= 0 {
            let d = &p.mat[own.diluent as usize];
            d.density + (own.density - d.density) * self.s
        } else {
            own.density
        };
        rho / SimParams::WATER_DENSITY
    }
}

/// Physics and numerics, and the live material table.
///
/// Every field is reachable by name through `PARAMS` and `MAT_PARAMS` (`--set name=value`,
/// `sprout params`), which also hold its unit, its safe range and, for a stabilisation
/// device, the value that switches it off. The panel reads its ranges from there too.
///
/// Two former fields are definitions of the discretisation, now constants in
/// `shaders/common.glsl`: the rest occupancy of a cell (`REST_PHI` = 1, i.e. `PPC`
/// particles) and the occupancy above which a cell counts as liquid (`LIQUID_THR` = 0.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimParams {
    // ---- world -----------------------------------------------------------------------
    /// Physical edge length of one grid cell, m. Each map sets it (`MapDef::width_m`).
    pub cell_size: f32,
    /// m/s².
    pub gravity: f32,

    // ---- liquids ---------------------------------------------------------------------
    /// The live material table: row `Material(i)` is `mat[i]`; `mat_count` rows are in use.
    pub mat: [MatProps; MAX_MATERIALS],
    pub mat_count: usize,
    /// Interfacial tension between phase groups, N/m (symmetric, zero diagonal).
    pub pair_tension: [[f32; GROUPS]; GROUPS],

    // ---- heat ---------------------------------------------------------------------------
    /// Temperature of the air (and of what the tank starts at), °C.
    pub ambient_temp: f32,
    /// Speed of all heat exchange relative to real time. Conduction over centimetres takes
    /// minutes; above 1 this is a sandbox control, not physics. 0 switches conduction off.
    pub heat_rate: f32,
    /// Film coefficient of the air at a free surface, W/(m² K) (still air: 5 - 25).
    pub air_cooling: f32,
    /// Heat transfer coefficient where two different liquids meet, W/(m² K): the thermal
    /// boundary layer of convection and boiling at the interface, far thinner than a cell,
    /// which conduction across half a cell on each side cannot see (README, "Heat").
    /// Natural convection in water 100 - 1000, film boiling 200 - 500, nucleate boiling
    /// 10⁴ and more. 0 = conduction only. The default, 3000, is for subcooled water against a
    /// melt: water 80 K below its boiling point cannot keep a stable vapour film, which
    /// collapses into transition boiling (10³ - 10⁴; quenching experiments on molten
    /// material in water). Until 2026-11-14 it was 300 (stable film boiling): lava under a
    /// pool stayed molten for minutes and the pool hardly warmed (README, "Heat").
    pub contact_heat: f32,
    /// Sub-grid mixing of heat in moving liquid (README, "Heat"): the Smagorinsky constant
    /// C_s of an eddy diffusivity (C_s dx)² |S| / Pr_t (Pr_t = 0.85) added to the
    /// conductivity between cells of the same liquid, |S| the resolved strain rate between
    /// them. Eddies smaller than a cell mix heat in stirred or convecting water orders of
    /// magnitude faster than conduction (water: 1.4e-7 m²/s; a 1 cm cell took 12 minutes to
    /// even out). Lilly's value 0.17; 0 = conduction only.
    pub eddy_heat: f32,
    /// 1: the free surface of an aqueous liquid warmer than the air loses heat by
    /// evaporation as well as by the film coefficient, and the vapour shows as mist over it
    /// (README, "Heat"). By the Lewis analogy the mass transfer goes with `air_cooling`:
    /// q = h / c_p,air x L x (Y_sat(T) - Y_sat(T_air)), Y the vapour's mass fraction. At 80 °C
    /// that is ~8 kW/m², ten times the film's 600 W/m². 0 = off.
    pub evaporation: u32,
    /// Kinematic viscosity at which liquid stops flowing and moves as a rigid body, m²/s
    /// (README, "Solidification" and "Rheology"): the cap of the viscosity solve. Frozen
    /// liquid, glass or lava cooled past it, unyielded yield-stress liquid and jammed
    /// shear-thickening liquid are rigid (rigid.comp).
    pub rigid_viscosity: f32,
    /// 1: rigid bodies rest on solids (README, "Solidification"): a body cannot move into
    /// the floor or a wall it touches, and tips over the edge of its support only when its
    /// centre of mass is beyond it. Without it a body touches solids only through the liquid
    /// solve, which lets a porous body (a heap of frozen lumps) slump into the floor every
    /// substep and bounce back off the collision: the body and the liquid around it shake.
    pub rigid_contact: u32,
    /// Coulomb friction coefficient between a rigid body and the solid it rests on: each
    /// substep the body's sliding speed there falls by this multiple of the speed the
    /// contact stopped it moving into the solid. Rock on rock 0.5 - 0.8. 0 = frictionless.
    pub rigid_friction: f32,
    /// Heat / Cool brush: kelvin per second for water at the centre of the brush.
    pub heat_power: f32,
    /// 1: aqueous liquid that boils turns into steam (README, "Steam"): bubbles that rise
    /// through the liquid, condense where it is colder, and escape at the free surface as
    /// puffs. 0: boiled liquid leaves the tank at once, wherever it is.
    pub steam: u32,
    /// Vapour quality (the share of its latent heat of vaporisation a parcel holds, 0 - 1) at
    /// which boiling water becomes a bubble of steam, and below which steam condenses back
    /// into water. Even 1 % quality is ~94 % vapour by volume at 1 atm, so a parcel past a
    /// few per cent is a bubble; the gap between the two is a hysteresis, so a parcel does
    /// not flip between them every substep at the boiling point.
    pub steam_boil: f32,
    pub steam_condense: f32,
    /// Rate at which the particles of a cell that holds steam even out their temperatures,
    /// 1/s (the exchange inside a cell, README "Heat"): far faster than conduction through a
    /// parcel the size of a particle, so that bubbles condense in colder water. 30 until the
    /// third round of 2026-11-14, when bubbles popped within a cell or two of leaving hot water
    /// (reported); at 3 they rise about twice as far through cool water (`LAB_BUBBLES=1 sprout
    /// lab`: 23 -> ~45 cells through water at 20 deg C) and most still condense before the
    /// surface of a 30-cell pool (`validate thermal`). (A sandbox choice: a bubble here holds
    /// one parcel's vapour, not its real volume.)
    pub steam_mixing: f32,
    /// Speed at which escaped steam rises, m/s (a warm plume: 0.2 - 1), and how long it
    /// lasts, s: puffs lay vapour into the smoke field for this long, and the vapour mixes
    /// away into the air with 0.8 x this as its time constant (smoke.comp).
    pub steam_rise: f32,
    pub steam_life: f32,
    /// Escaped steam is drawn as tracers in the air (particles.comp: puff_air): the speed of
    /// the eddies that twist the plume into wisps, m/s (0 = it rises straight), and their
    /// size, m. A look setting: the air itself is not simulated.
    pub steam_swirl: f32,
    pub steam_eddy: f32,
    /// Diameter of the oil (or water) droplets assumed inside a cell that holds both phases,
    /// in cells: a sub-grid model parameter, so it scales with the grid. Sets how fast the
    /// phases drift apart: the droplets' terminal velocity (see particles.comp).
    pub droplet_cells: f32,

    // ---- time step and accuracy --------------------------------------------------------
    /// Courant number: the most a particle may travel in one substep, in cells.
    pub cfl: f32,
    /// Particle advection: 1 = forward Euler, 2 = midpoint rule through the grid velocity.
    pub advection_order: u32,
    /// Longest substep, s. At CFL 4 a slowly moving pool would otherwise take 1/60 s steps,
    /// in which particles converging on a spot travel up to three cells before the density
    /// correction sees them (refilling a hole then left 13 to 21 particles in one cell
    /// instead of 9).
    pub dt_max: f32,
    /// Fixed substep rate in Hz for comparisons; 0 picks the step from the CFL condition.
    pub fixed_hz: f32,
    /// Pressure solve: stop when |r| <= tol * |b|.
    pub pressure_tol: f32,
    pub pressure_max_iters: u32,
    /// Density-correction solve (same matrix), looser: it only nudges positions.
    pub correction_tol: f32,
    /// Viscosity solve: stop when |r| <= tol * |r0|.
    pub viscosity_tol: f32,
    pub viscosity_max_iters: u32,
    /// Cells whose viscous diffusion number nu dt / dx² stays below this get the explicit
    /// viscous step (accurate there, stable to ~0.12) instead of the implicit solve. 0 makes
    /// every tile of a viscous tank implicit. A cost saving, exact to the solver tolerance.
    pub implicit_viscosity_min: f32,

    // ---- stabilisation (README, "Numerical stabilisation"); see `PARAMS` for "off" -------
    /// Spray in free flight (less than half the rest density around it) may travel this many
    /// times `cfl` per substep: its path is ballistic and does not depend on the step. 1 makes
    /// the fastest droplet set the step for the whole tank again.
    pub spray_cfl: f32,
    /// Particle speed cap, as a multiple of the free-fall speed across the height of the
    /// tank, sqrt(2 g H). At 1 a particle can still reach the lid from the floor, so nothing
    /// that stays inside the tank is slowed visibly; what it removes are the few impact-jet
    /// particles (about 0.1 %) that would otherwise set the time step for everything else.
    /// 0 switches the cap off. Inactive without gravity.
    pub speed_cap: f32,
    /// Time constant of the density correction, s: a substep of length dt corrects
    /// 1 - exp(-dt / tau) of a cell's density error (at most `correction_max`). The default
    /// matches the earlier fixed 0.5 per 1/240 s substep.
    pub correction_time: f32,
    /// Most of a cell's density error corrected in one substep; 0 switches the density
    /// correction off.
    pub correction_max: f32,
    /// Buried liquid that is not under pressure is pushed apart only where a cell is this
    /// much over rest occupancy (a clump). 0 pushes every over-full cell.
    pub correction_deadband: f32,
    /// Most a particle is moved by the density correction in one substep, cells; 0 = no limit.
    pub correction_step_max: f32,
    /// Amplitude of the random shake that separates particles stacked on one spot (cells
    /// over 1.5 × rest occupancy), cells; 0 = off.
    pub crowd_jitter: f32,
    /// Occupancy (× rest) above which Push and Grab stop driving a particle. They set
    /// velocities after the pressure solve, far faster than the density correction can
    /// undo, and without this packed liquid against the floor, a wall or a frozen body into
    /// cells of thousands of particles. 0 = off.
    pub tool_crowd_max: f32,
    /// Bound on each entry of a particle's APIC affine matrix, in units of 1/dt; 0 = off.
    pub apic_clamp: f32,
    /// Largest curvature surface tension acts on, 1/cells (radius 1/curvature_max cells);
    /// 0 = no limit.
    pub curvature_max: f32,
    /// 1 caps the oil / aqueous drift at `cfl` cells per substep; 0 leaves it uncapped.
    pub slip_cap: u32,
    /// 1 gives a region that is closed on all sides and full of liquid (a sealed box, a tank
    /// filled to the lid) a pressure reference: one cell at its top, the vent, is treated as
    /// touching air at ambient pressure for as long as the region holds no air. Without it
    /// such a region has no free surface, its pressure system is singular and the solve
    /// diverges. Never acts on a region that has a free surface. 0 = off.
    pub sealed_vent: u32,
    /// Viscoelastic stress is capped at this multiple of the shear modulus (upper-convected
    /// Maxwell stress grows without bound in strong extension); 0 = no cap.
    pub elastic_stress_max: f32,
    /// Most deformation (|velocity gradient| x dt) one substep feeds into the viscoelastic
    /// stress; 0 = no limit.
    pub elastic_step_max: f32,
    /// Blocking helpers (validate, probe, shot, bench) never record a substep shorter than
    /// this share of a full one; the remainder is carried to the next call. 0 = off.
    pub min_substep_share: f32,
}

/// The built-ins in their rows; unused rows hold the neutral default.
pub fn default_table() -> [MatProps; MAX_MATERIALS] {
    let mut t = [MatProps::default(); MAX_MATERIALS];
    for (row, b) in t.iter_mut().zip(BUILTINS.iter()) {
        *row = (b.props)();
    }
    t
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            cell_size: 0.01,
            gravity: 9.81,
            mat: default_table(),
            mat_count: BUILTINS.len(),
            pair_tension: default_pair_tension(),
            ambient_temp: T_REF,
            heat_rate: 1.0,
            air_cooling: 10.0,
            contact_heat: 3000.0,
            eddy_heat: 0.17,
            evaporation: 1,
            rigid_viscosity: 1.0,
            rigid_contact: 1,
            rigid_friction: 0.6,
            heat_power: 300.0,
            steam: 1,
            steam_boil: 0.02,
            steam_condense: 0.005,
            steam_mixing: 3.0,
            steam_rise: 0.4,
            steam_life: 2.0,
            steam_swirl: 0.1,
            steam_eddy: 0.2,
            droplet_cells: 1.0,
            cfl: 4.0,
            advection_order: 2,
            dt_max: 1.0 / 80.0,
            fixed_hz: 0.0,
            pressure_tol: 1e-4,
            pressure_max_iters: 60,
            correction_tol: 1e-2,
            viscosity_tol: 1e-3,
            viscosity_max_iters: 200,
            implicit_viscosity_min: 1e-3,
            spray_cfl: 4.0,
            speed_cap: 1.0,
            correction_time: (1.0 / 240.0) / std::f32::consts::LN_2,
            correction_max: 0.5,
            correction_deadband: 0.25,
            correction_step_max: 0.5,
            crowd_jitter: 0.5,
            tool_crowd_max: 2.0,
            apic_clamp: 1.0,
            curvature_max: 0.5,
            slip_cap: 1,
            sealed_vent: 1,
            min_substep_share: 0.02,
            elastic_stress_max: 10.0,
            elastic_step_max: 0.25,
        }
    }
}

/// Who a parameter is for. The panel shows `World` and `Liquid` under Physics and a few of
/// `Accuracy` and `Stabilisation` under Solver; everything is reachable with `--set`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    World,
    Liquid,
    Accuracy,
    Stabilisation,
}

impl Group {
    pub const ALL: [Group; 4] = [Group::World, Group::Liquid, Group::Accuracy, Group::Stabilisation];

    pub fn title(self) -> &'static str {
        match self {
            Group::World => "World",
            Group::Liquid => "Liquids",
            Group::Accuracy => "Time step and accuracy (solver)",
            Group::Stabilisation => "Stabilisation (solver; each can be switched off)",
        }
    }
}

/// One named parameter: how to read and write it in its display unit, and the range that is
/// safe to use. `off`, if given, is a value outside the range that switches the device off.
#[derive(Clone, Copy)]
pub struct Param {
    pub key: &'static str,
    pub group: Group,
    pub unit: &'static str,
    pub help: &'static str,
    pub min: f32,
    pub max: f32,
    pub off: Option<f32>,
    pub integer: bool,
    /// Material row for a per-material parameter (`MAT_PARAMS`), 0 otherwise.
    pub index: usize,
    pub get: fn(&SimParams, usize) -> f32,
    pub put: fn(&mut SimParams, usize, f32),
}

impl Param {
    /// Clamps `v` into the safe range, keeping the "off" value as it is.
    pub fn clamp(&self, v: f32) -> f32 {
        if self.off == Some(v) {
            return v;
        }
        let v = v.clamp(self.min, self.max);
        if self.integer { v.round() } else { v }
    }

    /// Whether a slider over the range should be logarithmic.
    pub fn logarithmic(&self) -> bool {
        self.min > 0.0 && self.max / self.min >= 100.0
    }

    pub fn read(&self, p: &SimParams) -> f32 {
        (self.get)(p, self.index)
    }

    pub fn write(&self, p: &mut SimParams, v: f32) {
        (self.put)(p, self.index, v)
    }

    /// Looks a parameter up by name: a global one (`PARAMS`), an old name (`ALIASES`), or
    /// a material property as `<property>_<material>` or `<material>.<property>`
    /// (`MAT_PARAMS`; the material is a built-in's key or `mat<row>`).
    pub fn find(key: &str) -> Option<Param> {
        // `sigma_oil` stays refused: it was documented as the oil-water tension but set oil
        // against air, so an old command line cannot be trusted to mean either.
        if key == "sigma_oil" {
            return None;
        }
        let key = ALIASES.iter().find(|a| a.0 == key).map_or(key, |a| a.1);
        if let Some(p) = PARAMS.iter().find(|p| p.key == key) {
            return Some(*p);
        }
        let (prop, mat) = match key.split_once('.') {
            Some((m, prop)) => (prop.to_string(), m.to_string()),
            None => {
                // Longest property name (current or old) that prefixes the key (`contact_angle_oil`).
                let names = MAT_PARAMS.iter().map(|p| p.key).chain(MAT_ALIASES.iter().map(|a| a.0));
                let name = names.filter(|n| key.starts_with(n) && key[n.len()..].starts_with('_')).max_by_key(|n| n.len())?;
                (name.to_string(), key[name.len() + 1..].to_string())
            }
        };
        let prop = MAT_ALIASES.iter().find(|a| a.0 == prop).map_or(prop.as_str(), |a| a.1);
        let index = material_row(&mat)?;
        MAT_PARAMS.iter().find(|p| p.key == prop).map(|p| Param { index, ..*p })
    }
}

/// Table row of a material named in a parameter key: a built-in's key, or `mat<row>`.
pub fn material_row(name: &str) -> Option<usize> {
    if let Some(i) = BUILTINS.iter().position(|b| b.key == name) {
        return Some(i);
    }
    name.strip_prefix("mat")?.parse::<usize>().ok().filter(|&i| i < MAX_MATERIALS)
}

macro_rules! param {
    ($key:literal, $group:ident, $unit:literal, [$min:expr, $max:expr], $off:expr, $int:literal, |$p:ident| $get:expr, |$q:ident, $v:ident| $put:expr, $help:literal) => {
        Param {
            key: $key,
            group: Group::$group,
            unit: $unit,
            help: $help,
            min: $min,
            max: $max,
            off: $off,
            integer: $int,
            index: 0,
            get: |$p: &SimParams, _: usize| $get,
            put: |$q: &mut SimParams, _: usize, $v: f32| $put,
        }
    };
}

macro_rules! mat_param {
    ($key:literal, $unit:literal, [$min:expr, $max:expr], $int:literal, |$m:ident| $get:expr, |$q:ident, $v:ident| $put:expr, $help:literal) => {
        Param {
            key: $key,
            group: Group::Liquid,
            unit: $unit,
            help: $help,
            min: $min,
            max: $max,
            off: None,
            integer: $int,
            index: 0,
            get: |p: &SimParams, i: usize| {
                let $m = &p.mat[i];
                $get
            },
            put: |p: &mut SimParams, i: usize, $v: f32| {
                let $q = &mut p.mat[i];
                $put
            },
        }
    };
}

/// Every property of a material, in display units, addressed as `<property>_<material>` or
/// `<material>.<property>` (`density_oil`, `mercury.tension`). Ranges are the ones checked
/// to run (README, "How the ranges were checked").
pub static MAT_PARAMS: &[Param] = &[
    mat_param!("density", "kg/m³", [300.0, 14000.0], false, |m| m.density, |m, v| m.density = v, "Density of the liquid."),
    mat_param!("viscosity", "m²/s", [1e-7, 0.5], false, |m| m.viscosity, |m, v| m.viscosity = v, "Kinematic viscosity at the reference temperature and zero shear rate; blends with the diluent logarithmically."),
    mat_param!("tension", "N/m", [0.0, 0.75], false, |m| m.tension, |m, v| m.tension = v, "Surface tension against air."),
    mat_param!("contact_angle", "°", [0.0, 180.0], false, |m| m.contact_angle, |m, v| m.contact_angle = v, "Static contact angle on solids, through the liquid (90 = no preference)."),
    mat_param!("mix_rate", "1/s", [0.0, 20.0], false, |m| m.mix_rate, |m, v| m.mix_rate = v, "How fast its concentration evens out with its diluent."),
    mat_param!("density_solid", "kg/m³", [0.0, 15000.0], false, |m| m.density_solid, |m, v| m.density_solid = v, "Density when frozen; 0 = as the liquid."),
    mat_param!("rheology", "", [0.0, 3.0], true, |m| m.rheology as f32, |m, v| m.rheology = v as u32, "0 Newtonian, 1 Carreau, 2 Herschel-Bulkley / Bingham, 3 shear thickening."),
    mat_param!("yield_stress", "Pa", [0.0, 500.0], false, |m| m.yield_stress, |m, v| m.yield_stress = v, "Herschel-Bulkley yield stress: below it the liquid holds its shape."),
    mat_param!("flow_index", "", [0.2, 3.0], false, |m| m.flow_index, |m, v| m.flow_index = v, "Power-law index n: below 1 thins with shear, above 1 thickens."),
    mat_param!("shear_time", "s", [1e-3, 100.0], false, |m| m.shear_time, |m, v| m.shear_time = v, "Carreau / thickening time constant: the shear rate 1 / shear_time is where the viscosity starts to change."),
    mat_param!("visc_ratio", "", [1e-4, 1e4], false, |m| m.visc_ratio, |m, v| m.visc_ratio = v, "Carreau: viscosity at infinite shear over the zero-shear one. Thickening: the most the viscosity may rise."),
    mat_param!("temp_model", "", [0.0, 2.0], true, |m| m.temp_model as f32, |m, v| m.temp_model = v as u32, "Viscosity against temperature: 0 none, 1 Arrhenius, 2 Vogel-Fulcher-Tammann."),
    mat_param!("temp_b", "K", [0.0, 60000.0], false, |m| m.temp_b, |m, v| m.temp_b = v, "Arrhenius / VFT B: larger means the viscosity falls faster on heating."),
    mat_param!("temp_c", "K", [0.0, 1500.0], false, |m| m.temp_c, |m, v| m.temp_c = v, "VFT C: the temperature at which the viscosity diverges."),
    mat_param!("temp_ref", "°C", [-50.0, 1600.0], false, |m| m.temp_ref, |m, v| m.temp_ref = v, "Temperature at which `viscosity` holds."),
    mat_param!("grain", "m", [0.0, 5e-3], false, |m| m.grain, |m, v| m.grain = v, "Diameter of suspended grains: the concentrate settles through its diluent at their terminal velocity; 0 = dissolved."),
    mat_param!("load", "", [0.02, 1.0], false, |m| m.load, |m, v| m.load = v, "Concentration of freshly emitted liquid: the share that is concentrate, the rest its diluent."),
    mat_param!("heat_capacity", "J/kgK", [100.0, 6000.0], false, |m| m.heat_capacity, |m, v| m.heat_capacity = v, "Specific heat capacity."),
    mat_param!("conductivity", "W/mK", [0.01, 100.0], false, |m| m.conductivity, |m, v| m.conductivity = v, "Thermal conductivity."),
    mat_param!("expansion", "1/K", [0.0, 0.005], false, |m| m.expansion, |m, v| m.expansion = v, "Thermal expansion: the liquid's density falls by this share per kelvin above `temp_ref`, so heated liquid rises."),
    mat_param!("melt_lo", "°C", [-1e9, 1600.0], false, |m| m.melt_lo, |m, v| m.melt_lo = v, "Solidus: frozen below it."),
    mat_param!("melt_hi", "°C", [-1e9, 1600.0], false, |m| m.melt_hi, |m, v| m.melt_hi = v, "Liquidus: liquid above it; below -273 the material never freezes."),
    mat_param!("latent", "J/kg", [0.0, 1e6], false, |m| m.latent, |m, v| m.latent = v, "Latent heat of fusion."),
    mat_param!("emissivity", "", [0.0, 1.0], false, |m| m.emissivity, |m, v| m.emissivity = v, "Grey-body emissivity: radiative cooling at the free surface, and the glow."),
    mat_param!("boil", "°C", [30.0, 1e9], false, |m| m.boil, |m, v| m.boil = v, "Boiling point; 1e9 = never boils."),
    mat_param!("latent_vap", "J/kg", [0.0, 1e7], false, |m| m.latent_vap, |m, v| m.latent_vap = v, "Latent heat of vaporisation."),
    mat_param!("residue_share", "", [0.0, 1.0], false, |m| m.residue_share, |m, v| m.residue_share = v, "Volume share of the concentrate left behind as residue when it boils away."),
    mat_param!("freeze_shift", "K", [-100.0, 100.0], false, |m| m.freeze_shift, |m, v| m.freeze_shift = v, "Shift of the diluent's freezing range per unit concentration (brine: -21)."),
    mat_param!("spawn_temp", "°C", [-50.0, 1600.0], false, |m| m.spawn_temp, |m, v| m.spawn_temp = v, "Temperature of newly emitted liquid."),
    mat_param!("shear_modulus", "Pa", [0.0, 2000.0], false, |m| m.shear_modulus, |m, v| m.shear_modulus = v, "Elastic shear modulus (upper-convected Maxwell); 0 = not elastic."),
    mat_param!("relax_time", "s", [0.0, 30.0], false, |m| m.relax_time, |m, v| m.relax_time = v, "Time over which elastic stress relaxes."),
    mat_param!("susceptibility", "", [0.0, 20.0], false, |m| m.susceptibility, |m, v| m.susceptibility = v, "Magnetic volume susceptibility (SI)."),
    mat_param!("saturation", "T", [0.001, 2.0], false, |m| m.saturation, |m, v| m.saturation = v, "Saturation magnetisation (as mu0 Ms): the most a field can magnetise it."),
    mat_param!("skin", "N/m", [0.0, 2.0], false, |m| m.skin, |m, v| m.skin = v, "Strength of a surface skin: a yield stress in free-surface cells."),
    mat_param!("opacity", "", [0.0, 1.0], false, |m| m.opacity, |m, v| m.opacity = v, "How deep its colour builds up: 0 clear (water shows its colour through 30 cells), 1 opaque (half a cell)."),
    mat_param!("scatter", "", [0.0, 1.0], false, |m| m.scatter, |m, v| m.scatter = v, "How strongly it scatters light: 0 clear (light through it is tinted), 1 milky (it shows its own colour, lit from the front)."),
    mat_param!("metal", "", [0.0, 1.0], false, |m| m.metal, |m, v| m.metal = v, "0 dielectric, 1 mirror metal."),
    mat_param!("gloss", "", [0.0, 1.0], false, |m| m.gloss, |m, v| m.gloss = v, "Strength of the specular streak."),
    mat_param!("body", "", [0.0, 1.0], false, |m| m.body, |m, v| m.body = v, "Fullness of the surface and width of the translucent rim (thick liquids)."),
    mat_param!("foam", "", [0.0, 1.0], false, |m| m.foam, |m, v| m.foam = v, "1: fast, thin liquid turns to white water."),
    mat_param!("dry_time", "s", [0.2, 120.0], false, |m| m.dry_time, |m, v| m.dry_time = v, "How long a film of it stays on a solid."),
    mat_param!("stain", "", [0.0, 1.0], false, |m| m.stain, |m, v| m.stain = v, "How fast it leaves a lasting stain on solids and the back wall (blood 1); other liquids slowly wash it off."),
    mat_param!("glint", "", [0.0, 1.0], false, |m| m.glint, |m, v| m.glint = v, "Fine glints and a moving sheen on its surface, carried by the flow (honey 1)."),
    mat_param!("iridescence", "", [0.0, 1.0], false, |m| m.iridescence, |m, v| m.iridescence = v, "Thin-film colours on its surface and in thin layers (oil 1)."),
    mat_param!("crackle", "", [0.0, 1.0], false, |m| m.crackle, |m, v| m.crackle = v, "Below a bright glow, a dark crust of plates with glowing seams (lava 1)."),
];

/// Old property prefixes (`nu_oil` = `viscosity_oil`).
pub static MAT_ALIASES: &[(&str, &str)] = &[("nu", "viscosity"), ("sigma", "tension"), ("contact", "contact_angle")];


/// Every parameter, in display units. The single source of names, units, ranges and "off"
/// values for `--set`, `sprout params` and the panel.
pub static PARAMS: &[Param] = &[
    // ---- world ----
    param!("gravity", World, "m/s²", [0.0, 30.0], None, false, |p| p.gravity, |p, v| p.gravity = v, "Downward acceleration."),
    param!("cell_size", World, "m", [5e-4, 0.05], None, false, |p| p.cell_size, |p, v| p.cell_size = v,
        "Edge of one grid cell; sets the physical size of the tank. Each map sets its own (1 cm, Oil Drops ~1 mm); giving it with --set keeps it for every map."),
    param!("ambient_temp", World, "°C", [-50.0, 200.0], None, false, |p| p.ambient_temp, |p, v| p.ambient_temp = v,
        "Temperature of the air above the liquid."),
    param!("heat_rate", World, "×", [0.1, 200.0], Some(0.0), false, |p| p.heat_rate, |p, v| p.heat_rate = v,
        "Speed of conduction and cooling relative to real time (1 = physical; a 1 cm layer takes minutes); 0 = no heat exchange."),
    param!("air_cooling", World, "W/m²K", [0.0, 500.0], None, false, |p| p.air_cooling, |p, v| p.air_cooling = v,
        "Film coefficient of the air at a free surface; radiation is added from each liquid's emissivity."),
    param!("contact_heat", World, "W/m²K", [10.0, 20000.0], Some(0.0), false, |p| p.contact_heat, |p, v| p.contact_heat = v,
        "Heat transfer coefficient where two different liquids meet (convection and boiling in a boundary layer thinner than a cell); 0 = conduction across the cells only."),
    param!("eddy_heat", World, "", [0.05, 0.4], Some(0.0), false, |p| p.eddy_heat, |p, v| p.eddy_heat = v,
        "Smagorinsky constant of the sub-grid heat mixing in moving liquid (eddies smaller than a cell); 0 = conduction only."),
    param!("evaporation", World, "", [0.0, 1.0], None, true, |p| p.evaporation as f32, |p, v| p.evaporation = v as u32,
        "1: warm water loses heat by evaporating at its surface, and steams; 0 = off."),
    param!("rigid_viscosity", World, "m²/s", [0.05, 50.0], None, false, |p| p.rigid_viscosity, |p, v| p.rigid_viscosity = v,
        "Viscosity at which liquid stops flowing and moves rigidly (frozen, set glass, unyielded sludge, jammed oobleck); the cap of the viscosity solve. Higher: more viscosity iterations."),
    param!("rigid_contact", World, "", [0.0, 1.0], None, true, |p| p.rigid_contact as f32, |p, v| p.rigid_contact = v as u32,
        "1: rigid bodies rest on the floor and walls they touch (no sinking in, tipping only over the edge of their support); 0 = they touch solids only through the liquid solve."),
    param!("rigid_friction", World, "", [0.05, 2.0], Some(0.0), false, |p| p.rigid_friction, |p, v| p.rigid_friction = v,
        "Coulomb friction between a rigid body and the solid it rests on (rock on rock 0.5 - 0.8); 0 = frictionless."),
    param!("heat_power", World, "K/s", [10.0, 5000.0], None, false, |p| p.heat_power, |p, v| p.heat_power = v,
        "Heat / Cool brush: kelvin per second for water at its centre."),
    param!("steam", World, "", [0.0, 1.0], None, true, |p| p.steam as f32, |p, v| p.steam = v as u32,
        "1: boiling water turns into steam bubbles that rise, condense in colder water and escape at the surface as puffs; 0 = boiled liquid leaves the tank at once."),
    param!("steam_boil", World, "", [0.001, 1.0], None, false, |p| p.steam_boil, |p, v| p.steam_boil = v,
        "Vapour quality (share of the latent heat taken up) at which boiling water becomes a steam bubble."),
    param!("steam_condense", World, "", [0.0, 1.0], None, false, |p| p.steam_condense, |p, v| p.steam_condense = v,
        "Vapour quality below which steam condenses back into water (below steam_boil)."),
    param!("steam_mixing", World, "1/s", [0.1, 1000.0], None, false, |p| p.steam_mixing, |p, v| p.steam_mixing = v,
        "How fast the particles of a cell holding steam even out their heat: how quickly bubbles condense in colder water."),
    param!("steam_rise", World, "m/s", [0.0, 3.0], None, false, |p| p.steam_rise, |p, v| p.steam_rise = v,
        "Speed at which escaped puffs of steam rise."),
    param!("steam_life", World, "s", [0.1, 10.0], None, false, |p| p.steam_life, |p, v| p.steam_life = v,
        "How long escaped steam lasts: puffs lay vapour for this long, and it mixes away into the air over 0.8 x this."),
    param!("steam_swirl", World, "m/s", [0.0, 2.0], None, false, |p| p.steam_swirl, |p, v| p.steam_swirl = v,
        "Speed of the eddies that twist escaped steam into wisps (0 = it rises straight up)."),
    param!("steam_eddy", World, "m", [0.005, 0.5], None, false, |p| p.steam_eddy, |p, v| p.steam_eddy = v,
        "Size of the eddies in the air that escaped steam rides."),
    // ---- liquids ----
    param!("tension_oil_water", Liquid, "N/m", [0.0, 0.1], None, false, |p| p.pair_tension[0][1], |p, v| { p.pair_tension[0][1] = v; p.pair_tension[1][0] = v }, "Interfacial tension between oil-like and aqueous liquids."),
    param!("tension_metal_water", Liquid, "N/m", [0.0, 0.75], None, false, |p| p.pair_tension[0][2], |p, v| { p.pair_tension[0][2] = v; p.pair_tension[2][0] = v }, "Interfacial tension between liquid metals and aqueous liquids."),
    param!("tension_metal_oil", Liquid, "N/m", [0.0, 0.75], None, false, |p| p.pair_tension[1][2], |p, v| { p.pair_tension[1][2] = v; p.pair_tension[2][1] = v }, "Interfacial tension between liquid metals and oil-like liquids."),
    param!("tension_melt_water", Liquid, "N/m", [0.0, 0.75], None, false, |p| p.pair_tension[0][3], |p, v| { p.pair_tension[0][3] = v; p.pair_tension[3][0] = v }, "Interfacial tension between silicate melts and aqueous liquids."),
    param!("tension_melt_oil", Liquid, "N/m", [0.0, 0.75], None, false, |p| p.pair_tension[1][3], |p, v| { p.pair_tension[1][3] = v; p.pair_tension[3][1] = v }, "Interfacial tension between silicate melts and oil-like liquids."),
    param!("tension_melt_metal", Liquid, "N/m", [0.0, 0.75], None, false, |p| p.pair_tension[2][3], |p, v| { p.pair_tension[2][3] = v; p.pair_tension[3][2] = v }, "Interfacial tension between silicate melts and liquid metals."),
    param!("mix_rate", Liquid, "1/s", [0.0, 20.0], None, false, |p| p.mat[1].mix_rate, |p, v| p.mat[1].mix_rate = v, "How fast touching water and syrup blend (the same as mix_rate_syrup)."),
    param!("droplet_cells", Liquid, "cells", [0.1, 4.0], None, false, |p| p.droplet_cells, |p, v| p.droplet_cells = v,
        "Size of the oil / water droplets assumed inside a cell holding both (sub-grid model): bigger droplets separate faster."),
    // ---- time step and accuracy ----
    param!("cfl", Accuracy, "cells", [0.5, 6.0], None, false, |p| p.cfl, |p, v| p.cfl = v,
        "Most cells a particle may travel per substep. Higher is faster and less accurate (stabilisation table: 4, was 2)."),
    param!("advection_order", Accuracy, "", [1.0, 2.0], None, true, |p| p.advection_order as f32, |p, v| p.advection_order = v as u32,
        "Particle paths: 1 forward Euler, 2 midpoint rule (needed for CFL above ~2)."),
    param!("dt_max", Accuracy, "s", [1e-3, 1.0 / 60.0], None, false, |p| p.dt_max, |p, v| p.dt_max = v, "Longest substep."),
    param!("fixed_hz", Accuracy, "Hz", [30.0, 4000.0], Some(0.0), false, |p| p.fixed_hz, |p, v| p.fixed_hz = v,
        "Fixed substep rate for comparisons; 0 picks each substep from the CFL condition."),
    param!("pressure_tol", Accuracy, "", [1e-6, 1e-2], None, false, |p| p.pressure_tol, |p, v| p.pressure_tol = v, "Relative residual the pressure solve stops at."),
    param!("pressure_max_iters", Accuracy, "", [10.0, 400.0], None, true, |p| p.pressure_max_iters as f32, |p, v| p.pressure_max_iters = v as u32,
        "Most pressure iterations recorded per substep (fewer are recorded when recent solves needed fewer)."),
    param!("correction_tol", Accuracy, "", [1e-4, 0.1], None, false, |p| p.correction_tol, |p, v| p.correction_tol = v, "Relative residual the density-correction solve stops at."),
    param!("viscosity_tol", Accuracy, "", [1e-5, 0.1], None, false, |p| p.viscosity_tol, |p, v| p.viscosity_tol = v, "Relative residual the viscosity solve stops at."),
    param!("viscosity_max_iters", Accuracy, "", [20.0, 1000.0], None, true, |p| p.viscosity_max_iters as f32, |p, v| p.viscosity_max_iters = v as u32,
        "Most viscosity iterations recorded per substep."),
    param!("implicit_viscosity_min", Accuracy, "", [1e-4, 0.05], Some(0.0), false, |p| p.implicit_viscosity_min, |p, v| p.implicit_viscosity_min = v,
        "Diffusion number nu dt/dx² below which a tile takes the explicit viscous step; 0 = implicit everywhere."),
    // ---- stabilisation ----
    param!("speed_cap", Stabilisation, "× √(2gH)", [1.0, 4.0], Some(0.0), false, |p| p.speed_cap, |p, v| p.speed_cap = v,
        "Particle speed cap as a multiple of the free-fall speed over the tank height; 0 = off."),
    param!("spray_cfl", Stabilisation, "× cfl", [1.0, 8.0], None, false, |p| p.spray_cfl, |p, v| p.spray_cfl = v,
        "How much further than cfl free-flying spray may travel per substep; 1 = off."),
    param!("correction_time", Stabilisation, "s", [1e-3, 0.1], None, false, |p| p.correction_time, |p, v| p.correction_time = v,
        "Time constant of the density correction."),
    param!("correction_max", Stabilisation, "", [0.05, 1.0], Some(0.0), false, |p| p.correction_max, |p, v| p.correction_max = v,
        "Most of a cell's density error corrected per substep; 0 = density correction off."),
    param!("correction_deadband", Stabilisation, "", [0.05, 0.5], Some(0.0), false, |p| p.correction_deadband, |p, v| p.correction_deadband = v,
        "Over-fullness below which buried liquid not under pressure is left alone; 0 = off."),
    param!("correction_step_max", Stabilisation, "cells", [0.25, 2.0], Some(0.0), false, |p| p.correction_step_max, |p, v| p.correction_step_max = v,
        "Most the density correction moves a particle per substep; 0 = no limit."),
    param!("crowd_jitter", Stabilisation, "cells", [0.1, 1.0], Some(0.0), false, |p| p.crowd_jitter, |p, v| p.crowd_jitter = v,
        "Random shake that separates particles stacked on one spot; 0 = off."),
    param!("tool_crowd_max", Stabilisation, "× rest", [1.25, 8.0], Some(0.0), false, |p| p.tool_crowd_max, |p, v| p.tool_crowd_max = v,
        "Occupancy above which Push and Grab stop driving liquid; 0 = off."),
    param!("apic_clamp", Stabilisation, "× 1/dt", [1.0, 4.0], Some(0.0), false, |p| p.apic_clamp, |p, v| p.apic_clamp = v,
        "Bound on the APIC affine velocity matrix; 0 = off."),
    param!("curvature_max", Stabilisation, "1/cells", [0.25, 2.0], Some(0.0), false, |p| p.curvature_max, |p, v| p.curvature_max = v,
        "Largest interface curvature surface tension acts on; 0 = no limit."),
    param!("slip_cap", Stabilisation, "", [0.0, 1.0], None, true, |p| p.slip_cap as f32, |p, v| p.slip_cap = v as u32,
        "1 caps the oil / water drift at cfl cells per substep; 0 = off."),
    param!("sealed_vent", Stabilisation, "", [0.0, 1.0], None, true, |p| p.sealed_vent as f32, |p, v| p.sealed_vent = v as u32,
        "1 gives a sealed, completely full region a pressure reference at its top cell (its pressure system is singular otherwise); 0 = off."),
    param!("elastic_stress_max", Stabilisation, "× G", [5.0, 500.0], Some(0.0), false, |p| p.elastic_stress_max, |p, v| p.elastic_stress_max = v,
        "Cap on viscoelastic stress, as a multiple of the shear modulus (strong stretching grows it without bound)."),
    param!("elastic_step_max", Stabilisation, "", [0.05, 1.0], Some(0.0), false, |p| p.elastic_step_max, |p, v| p.elastic_step_max = v,
        "Most deformation per substep that builds viscoelastic stress (an impact would otherwise make it explode)."),
    param!("min_substep_share", Stabilisation, "", [0.005, 0.2], Some(0.0), false, |p| p.min_substep_share, |p, v| p.min_substep_share = v,
        "Head-less runs never record a substep shorter than this share of a full one; 0 = off."),
];

/// Earlier names that still work (same units). Two are gone on purpose: `rho_*` (relative
/// density), because `density_*` is in kg/m³ and an old value would silently mean something
/// else; and `sigma_oil`, which was documented as the oil-water tension but actually set oil
/// against air (the per-material lookup caught it first).
pub static ALIASES: &[(&str, &str)] = &[
    ("frozen_viscosity", "rigid_viscosity"),
    ("diffuse", "mix_rate"),
    ("droplet", "droplet_cells"),
    ("substep_hz", "fixed_hz"),
    ("pressure_iters", "pressure_max_iters"),
    ("viscosity_iters", "viscosity_max_iters"),
    ("density_tau", "correction_time"),
    ("density_k", "correction_max"),
    ("kappa_max", "curvature_max"),
];

impl SimParams {
    pub const WATER_DENSITY: f32 = 998.0;

    /// A tension (N/m) in grid units: divided by water's density, cells³/s².
    pub fn tension_cells(&self, sigma: f32) -> f32 {
        let dx = self.cell_size;
        sigma / (Self::WATER_DENSITY * dx * dx * dx)
    }

    /// Tension of material `m` against air, cells³/s² (over water's density).
    pub fn sigma_cells(&self, m: Material) -> f32 {
        self.tension_cells(self.mat[m.index()].tension)
    }

    /// Interfacial tension between two phase groups, cells³/s² (over water's density).
    pub fn pair_sigma_cells(&self, a: PhaseGroup, b: PhaseGroup) -> f32 {
        self.tension_cells(self.pair_tension[a as usize][b as usize])
    }

    /// Relative density of pure material `m` (water = 1).
    /// Solidus, liquidus (°C) and latent heat (as degrees, L / c) of a particle of `m` at
    /// concentration `s` (mirrors `thermal_of` in common.glsl).
    pub fn melting(&self, m: Material, s: f32) -> [f32; 3] {
        let own = &self.mat[m.index()];
        let lat = |d: &MatProps| if d.freezes() { d.latent / d.heat_capacity } else { 0.0 };
        if own.diluent < 0 {
            return [own.melt_lo, own.melt_hi.max(own.melt_lo + 0.25), lat(own)];
        }
        let d = &self.mat[own.diluent as usize];
        let shift = own.freeze_shift * s;
        let (cd, co) = (d.heat_capacity_vol(), own.heat_capacity_vol());
        let c = cd + (co - cd) * s;
        let l = (cd * lat(d) + (co * own.latent / own.heat_capacity - cd * lat(d)) * s) / c;
        [d.melt_lo + shift, d.melt_hi.max(d.melt_lo + 0.25) + shift, l]
    }

    /// Enthalpy temperature (`Particle::t`) of a particle of `m` at temperature `t` °C.
    pub fn enthalpy_temp(&self, m: Material, s: f32, t: f32) -> f32 {
        let [lo, hi, l] = self.melting(m, s);
        if lo < -273.0 || t >= hi {
            t
        } else if t <= lo {
            t - l
        } else {
            let fr = (t - lo) / (hi - lo).max(1e-3);
            (lo - l) + fr * (hi - lo + l)
        }
    }

    /// Temperature (°C) and frozen share of a particle (inverse of `enthalpy_temp`).
    pub fn temperature(&self, q: &Particle) -> (f32, f32) {
        let Some(m) = q.material() else { return (0.0, 0.0) };
        let [lo, hi, l] = self.melting(m, q.s);
        if lo < -273.0 || q.t >= hi {
            (q.t, 0.0)
        } else if q.t <= lo - l {
            (q.t + l, 1.0)
        } else {
            let fr = (q.t - (lo - l)) / (hi - lo + l).max(1e-3);
            (lo + fr * (hi - lo), 1.0 - fr)
        }
    }

    /// Heat taken up by one cell of water per kelvin, per metre of depth, J/(K m).
    pub fn cell_heat(&self) -> f64 {
        WATER_HEAT as f64 * (self.cell_size as f64).powi(2)
    }

    /// Volumetric heat capacity of a particle of `m` at concentration `s`, relative to
    /// water's (mirrors `thermal_of` in common.glsl).
    pub fn heat_capacity_rel(&self, m: Material, s: f32) -> f32 {
        let own = &self.mat[m.index()];
        let c = self.particle_heat_vol(m.index()) / WATER_HEAT;
        if own.diluent >= 0 {
            let d = self.mat[own.diluent as usize].heat_capacity_vol() / WATER_HEAT;
            d + (c - d) * s
        } else {
            c
        }
    }

    /// Volumetric heat capacity a particle of table row `row` carries, J/(m³ K): its own,
    /// except steam, which carries the water it boiled from (`material.rs: steam`).
    pub fn particle_heat_vol(&self, row: usize) -> f32 {
        let r = if row == Material::Steam.index() { Material::Water.index() } else { row };
        self.mat[r].heat_capacity_vol()
    }

    pub fn rho(&self, m: Material) -> f32 {
        self.mat[m.index()].density / Self::WATER_DENSITY
    }

    /// Gravity in cells/s², the unit the validation scenes work in.
    pub fn g_cells(&self) -> f32 {
        self.gravity / self.cell_size
    }

    /// Kinematic viscosity of material `m` in cells²/s (reference temperature, zero shear).
    pub fn nu_cells(&self, m: Material) -> f32 {
        self.mat[m.index()].viscosity / (self.cell_size * self.cell_size)
    }

    /// The particle speed cap in cells/s for a tank `height` cells tall (infinite when off).
    pub fn speed_cap_cells(&self, height: u32) -> f32 {
        let g = self.g_cells();
        if self.speed_cap > 0.0 && g > 0.0 { self.speed_cap * (2.0 * g * height as f32).sqrt() } else { 1e30 }
    }

    /// Longest stable step of the explicit heat conduction, s: a fifth of the fastest cell
    /// diffusion time dx² (rho c) / k among the materials in the tank, over `heat_rate`.
    pub fn dt_thermal(&self, present: &[usize]) -> f32 {
        if self.heat_rate <= 0.0 {
            return f32::INFINITY;
        }
        let dx2 = self.cell_size * self.cell_size;
        present.iter().map(|&i| 0.2 * dx2 * self.particle_heat_vol(i) / self.mat[i].conductivity.max(1e-6) / self.heat_rate).fold(f32::INFINITY, f32::min)
    }

    /// The rows a tank holding the materials `present` can contain: those, their diluents,
    /// and what they leave behind when they boil.
    pub fn closure(&self, present: &[usize]) -> Vec<usize> {
        let mut rows: Vec<usize> = present.to_vec();
        let mut k = 0;
        while k < rows.len() {
            let m = &self.mat[rows[k]];
            for extra in [m.diluent, m.residue] {
                if extra >= 0 && !rows.contains(&(extra as usize)) {
                    rows.push(extra as usize);
                }
            }
            k += 1;
        }
        rows.sort_unstable();
        rows
    }

    /// Explicit surface tension is stable only for steps below the capillary-wave limit
    /// (Brackbill et al. 1992): dt < sqrt(rho_min dx³ / (2 pi sigma)). Returns that limit with a
    /// safety factor of 0.5, in seconds (grid units: dx = 1), over the materials in the tank
    /// (`present`: table rows) and the interfaces between their phase groups. Each tension is
    /// paired with the lighter liquid it acts on: mercury's tension moves mercury.
    /// Longest substep that keeps elastic waves within half a cell per substep, over the
    /// viscoelastic materials in `present`. A stretched Maxwell liquid is stiffer than its
    /// modulus: waves travel at up to sqrt((G + |tau|) / rho), and the stress is capped at
    /// `elastic_stress_max` x G (with no cap, at 50 G).
    pub fn dt_elastic(&self, present: &[usize]) -> f32 {
        let stiff = 1.0 + if self.elastic_stress_max > 0.0 { self.elastic_stress_max } else { 50.0 };
        present
            .iter()
            .map(|&i| &self.mat[i])
            .filter(|m| m.shear_modulus > 0.0)
            .map(|m| 0.5 * self.cell_size / (stiff * m.shear_modulus / m.density).sqrt())
            .fold(f32::INFINITY, f32::min)
    }

    pub fn dt_capillary(&self, present: &[usize]) -> f32 {
        let mut worst = 0.0f32;  // largest sigma / rho
        // Steam's surface is the water around the bubble: it moves water, not vapour.
        let present: Vec<usize> = present.iter().copied().filter(|&i| i != Material::Steam.index()).collect();
        let present = &present[..];
        for &i in present {
            let m = &self.mat[i];
            worst = worst.max(self.tension_cells(m.tension) / (m.density / Self::WATER_DENSITY));
            for &j in present {
                let n = &self.mat[j];
                if n.group != m.group {
                    let sigma = self.tension_cells(self.pair_tension[m.group as usize][n.group as usize]);
                    worst = worst.max(sigma / (m.density.min(n.density) / Self::WATER_DENSITY));
                }
            }
        }
        if worst <= 0.0 {
            return f32::INFINITY;
        }
        0.5 * (1.0 / (2.0 * std::f32::consts::PI * worst)).sqrt()
    }

    /// Sets a parameter by name (`--set name=value`), in the units `sprout params` lists,
    /// clamped to its safe range. Returns false for an unknown name.
    pub fn set(&mut self, key: &str, v: f32) -> bool {
        match Param::find(key) {
            // (NaN passes `clamp` untouched and would reach the GPU.)
            Some(_) if v.is_nan() => false,
            Some(p) => {
                p.write(self, p.clamp(v));
                true
            }
            None => false,
        }
    }

    /// Restores the defaults of every parameter in `groups` (for `Group::Liquid`: the
    /// built-in materials and the pair tensions; custom rows are left alone).
    pub fn reset_groups(&mut self, groups: &[Group]) {
        let d = SimParams::default();
        for p in PARAMS.iter().filter(|p| groups.contains(&p.group)) {
            p.write(self, p.read(&d));
        }
        if groups.contains(&Group::Liquid) {
            self.mat[..BUILTINS.len()].copy_from_slice(&d.mat[..BUILTINS.len()]);
        }
    }

    /// Whether every parameter in `groups` is at its default.
    pub fn is_default(&self, groups: &[Group]) -> bool {
        let d = SimParams::default();
        PARAMS.iter().filter(|p| groups.contains(&p.group)).all(|p| p.read(self) == p.read(&d))
            && (!groups.contains(&Group::Liquid) || self.mat[..BUILTINS.len()] == d.mat[..BUILTINS.len()])
    }
}

/// A number with four significant digits, in plain or scientific notation.
fn num(v: f32) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if !(1e-3..1e4).contains(&v.abs()) {
        return format!("{v:.2e}");
    }
    let decimals = (3 - v.abs().log10().floor() as i32).max(0) as usize;
    let s = format!("{v:.decimals$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// The table `sprout params` prints.
pub fn describe() -> String {
    let d = SimParams::default();
    let mut out = String::from("Parameters for --set name=value (units as listed; values are clamped to the range).\n");
    for g in Group::ALL {
        out += &format!("\n{}\n", g.title());
        for p in PARAMS.iter().filter(|p| p.group == g) {
            let off = p.off.map_or(String::new(), |o| format!(", {} = off", num(o)));
            out += &format!(
                "  {:<24} {:>9} {:<9} [{} .. {}{}]\n      {}\n",
                p.key, num(p.read(&d)), p.unit, num(p.min), num(p.max), off, p.help
            );
        }
    }
    out += "\nPer material, as <property>_<material> or <material>.<property> (materials: ";
    out += &BUILTINS.iter().map(|b| b.key).collect::<Vec<_>>().join(", ");
    out += "; a custom liquid is mat<row>)\n";
    for p in MAT_PARAMS {
        out += &format!("  {:<24} {:<9} [{} .. {}]\n      {}\n", p.key, p.unit, num(p.min), num(p.max), p.help);
        let values: Vec<String> = BUILTINS.iter().enumerate().map(|(i, b)| format!("{} {}", b.key, num((p.get)(&d, i)))).collect();
        out += &format!("      defaults: {}\n", values.join(", "));
    }
    out += "\nOld names still accepted: ";
    out += &ALIASES.iter().map(|a| format!("{} -> {}", a.0, a.1)).collect::<Vec<_>>().join(", ");
    out += "\n";
    out
}

/// Speed the push tool drives liquid outwards at, cells/s (carried in `ToolState::vel[0]`).
pub const PUSH_SPEED: f32 = 220.0;

/// Most currents a world can hold (the size of the arrays in `GpuParams`).
pub const MAX_CURRENTS: usize = 8;
/// Most magnets in the world at once.
pub const MAX_MAGNETS: usize = 8;
/// Magnet strengths offered by the tool, as the field at the magnet's rim, T. A strong
/// neodymium magnet gives 0.3 - 0.5 T at its face.
pub const MAGNET_STRENGTH: [f32; 2] = [0.02, 0.6];
pub const MAGNET_STRENGTH_DEFAULT: f32 = 0.3;

/// A permanent magnet placed in the world (README, "Magnetism"): the field of a point
/// dipole at `pos`, evaluated in the plane of the simulation, with strength `strength`
/// (tesla on its axis at distance `radius`) and its north pole towards `dir`. No field is
/// solved: the liquid does not change the field it sits in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Magnet {
    pub pos: [f32; 2],
    pub radius: f32,
    pub strength: f32,
    pub dir: [f32; 2],
}

impl Magnet {
    /// A magnet whose north pole points `degrees` clockwise from straight up.
    pub fn new(pos: [f32; 2], radius: f32, degrees: f32, strength: f32) -> Magnet {
        let (s, c) = degrees.to_radians().sin_cos();
        Magnet { pos, radius, strength, dir: [s, c] }
    }
}
/// How fast a current brings liquid up to its speed, 1/s, at its centre and full strength.
/// Slow enough that one substep adds a small share of the speed (under 10 % at 1/240 s), so
/// the next substep's CFL limit (taken from the velocities measured in this one) keeps up.
pub const CURRENT_RATE: f32 = 20.0;
/// Speed of a current at 100 % strength, cells/s.
pub const CURRENT_SPEED: f32 = 150.0;
/// Range of the Current tool's strength, in multiples of `CURRENT_SPEED`, and where it starts.
/// At the top the particle speed cap (`speed_cap`) can bind before the current's own speed.
pub const CURRENT_STRENGTH: [f32; 2] = [0.5, 8.0];
pub const CURRENT_STRENGTH_DEFAULT: f32 = 2.0;

/// A current: a round zone that drives the liquid passing through it
/// along `vel`. Only the velocity component along the current is raised towards its speed
/// (at `rate` per second in the centre, fading to nothing at the rim); liquid already
/// faster, and motion across the current, are left alone. So it works like a pump or a
/// fan: it never brakes, and never adds or removes liquid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Current {
    pub pos: [f32; 2],
    pub radius: f32,
    /// Direction and speed the liquid is driven towards, cells/s.
    pub vel: [f32; 2],
    /// 1/s; see `CURRENT_RATE`.
    pub rate: f32,
}

impl Current {
    /// A current at `pos` flowing `degrees` clockwise from straight down, at `strength`
    /// times `CURRENT_SPEED`.
    pub fn new(pos: [f32; 2], radius: f32, degrees: f32, strength: f32) -> Current {
        let (s, c) = degrees.to_radians().sin_cos();
        let speed = CURRENT_SPEED * strength;
        Current { pos, radius, vel: [s * speed, -c * speed], rate: CURRENT_RATE }
    }
}

/// What the mouse is doing to the fluid this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ToolState {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub radius: f32,
    /// 0 = idle, 1 = grab / stir, 2 = erase, 3 = push (speed in `vel[0]`), 4 = stream nozzle,
    /// 5 = heat, 6 = cool.
    pub mode: i32,
    pub strength: f32,
}

impl ToolState {
    /// Drags the fluid under the brush along with the cursor.
    pub fn grab(pos: [f32; 2], vel: [f32; 2], radius: f32) -> ToolState {
        ToolState { pos, vel, radius, mode: 1, strength: 40.0 }
    }

    /// Deletes the fluid under the brush.
    pub fn erase(pos: [f32; 2], radius: f32) -> ToolState {
        ToolState { pos, radius, mode: 2, ..Default::default() }
    }

    /// Drives the fluid under the brush radially outwards at `PUSH_SPEED`.
    pub fn push(pos: [f32; 2], radius: f32) -> ToolState {
        ToolState { pos, vel: [PUSH_SPEED, 0.0], radius, mode: 3, strength: 40.0 }
    }

    /// Heats (or, with `cool`, cools) the liquid under the brush at `SimParams::heat_power`.
    pub fn heat(pos: [f32; 2], radius: f32, cool: bool) -> ToolState {
        ToolState { pos, radius, mode: if cool { 6 } else { 5 }, ..Default::default() }
    }

    /// Holds the liquid inside a stream's nozzle (pour or jet) at the launch velocity `vel`.
    /// The stream then leaves at exactly the speed it is laid out for, so each frame's
    /// particles join the last frame's (left alone, new liquid takes up the faster flow of
    /// the accelerating stream below at once and runs ahead); and a nozzle whose outlet is
    /// already full keeps pushing.
    pub fn nozzle(pos: [f32; 2], vel: [f32; 2], radius: f32) -> ToolState {
        ToolState { pos, vel, radius, mode: 4, strength: 1000.0 }
    }

    /// Fastest velocity the tool can impose on a particle, cells/s.
    pub fn speed(&self) -> f32 {
        match self.mode {
            1 => self.vel[0].hypot(self.vel[1]) + 4.0 * self.radius,
            3 | 4 => self.vel[0].hypot(self.vel[1]),
            _ => 0.0,
        }
    }
}

/// Must match `uniform Params` in `shaders/common.glsl` (std140).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuParams {
    pub grid: [i32; 2],
    pub n_slots: u32,
    pub spawn_count: u32,
    pub dt_fixed: f32,
    pub gravity: f32,
    pub inv_ppc: f32,
    /// Bound on APIC matrix entries in units of 1/dt (0 = off).
    pub apic_clamp: f32,
    /// Interfacial tension between phase groups a and b / rho_water, cells³/s²: `pair[a][b]`.
    pub pair: [[f32; 4]; GROUPS],
    /// Curvature limit (0 = none); sealed-region vent (1 on); cell size (m); unused.
    pub misc: [f32; 4],
    /// Materials in the tank; ints per cell of the accumulators; concentration families;
    /// bit g set: phase group g is in the tank.
    pub n_present: i32,
    pub acc_stride: i32,
    pub n_fam: i32,
    pub group_mask: u32,
    /// Capability bits (`CAP_*`); unused.
    pub caps: u32,
    /// Row of the family that shares the first row's weight (-1 none).
    pub primary: i32,
    pub group_heat: u32,
    pub pad0: u32,
    /// Table rows of the materials in the tank.
    pub present: [[i32; 4]; MAX_PRESENT / 4],
    /// Per concentration family: its row, its diluent's row; unused.
    pub fams: [[i32; 4]; MAX_PRESENT],
    /// Ambient temperature (°C); heat rate; air film coefficient (cells/s); radiation
    /// coefficient (cells/s/K³).
    pub heat: [f32; 4],
    /// ln of the frozen viscosity (cells²/s); heat tool rate (K/s); accumulator channel of
    /// the temperature sum; unused.
    pub heat2: [f32; 4],
    pub sep_tau: f32,
    pub cfl: f32,
    pub tool_crowd_max: f32,
    pub density_k: f32,
    pub dt_max: f32,
    /// Amplitude of the crowding jitter, cells (0 = off).
    pub crowd_jitter: f32,
    /// Fastest speed the tool or newly poured fluid will impose this frame, cells/s.
    pub ext_speed: f32,
    pub wall_margin: f32,
    pub tool_pos: [f32; 2],
    pub tool_vel: [f32; 2],
    pub tool_radius: f32,
    pub tool_mode: i32,
    pub tool_strength: f32,
    pub frame: u32,
    /// Relative tolerances: pressure, density correction, viscosity; then the time constant
    /// of the density correction, s.
    pub tol: [f32; 4],
    /// Particle speed cap (cells/s, huge = off); advection order; spray CFL multiple; number
    /// of currents.
    pub ext: [f32; 4],
    /// Density correction: largest displacement per substep (cells, 0 = no limit), dead
    /// band for buried unpressurised liquid; slip cap (1 on, 0 off); diffusion number below
    /// which viscosity is explicit.
    pub corr: [f32; 4],
    /// Per current: position, radius, rate (1/s).
    pub current_zone: [[f32; 4]; MAX_CURRENTS],
    /// Per current: velocity (cells/s); unused.
    pub current_vel: [[f32; 4]; MAX_CURRENTS],
    /// Accumulator channel of the stress, largest shear modulus (grid units), stress cap
    /// (x modulus, 0 = none), unused. (Kept after the magnets in the shader too.)
    /// Number of magnets; largest susceptibility in the tank; its saturation (mu0 Ms, T);
    /// Pa -> grid units (1 / (rho_water dx^2)).
    pub mag: [f32; 4],
    /// Per magnet: position (cells), radius (cells), field at the radius (T).
    pub magnet: [[f32; 4]; MAX_MAGNETS],
    /// Per magnet: pole direction.
    pub magnet_dir: [[f32; 4]; MAX_MAGNETS],
    pub elastic: [f32; 4],
    /// Steam: vapour quality below which it condenses; puff rise speed (cells/s); puff
    /// lifetime (s); vapour quality at which boiling water becomes steam.
    pub steam: [f32; 4],
    /// Steam: rate of the exchange inside a cell that holds it (1/s); 1 if boiling makes
    /// steam (`steam`); eddy speed of the air puffs ride (cells/s); eddy size (cells).
    pub steam2: [f32; 4],
    /// Heat: Smagorinsky constant of the eddy diffusivity (0 off); evaporation factor
    /// L / c_p,air in kelvin (0 off); unused; unused.
    pub heat3: [f32; 4],
}

/// What the tank needs from the solver, from the materials in it (`GpuParams::caps`).
pub const CAP_FAMILIES: u32 = 1;
pub const CAP_THERMAL: u32 = 2;
pub const CAP_RIGID: u32 = 4;
pub const CAP_RHEO: u32 = 8;
pub const CAP_MAGNET: u32 = 16;
pub const CAP_ELASTIC: u32 = 32;
pub const CAP_SETTLE: u32 = 64;
pub const CAP_STEAM: u32 = 128;

/// Volumetric heat capacity of water, J/(m³ K): the unit heat is counted in on the GPU.
pub const WATER_HEAT: f32 = SimParams::WATER_DENSITY * 4182.0;
/// Stefan-Boltzmann constant, W/(m² K⁴).
pub const STEFAN_BOLTZMANN: f32 = 5.670_374e-8;

const _: () = assert!(std::mem::size_of::<Particle>() == 48);
const _: () = assert!(KIND_SRC & (KIND_FRESH | KIND_REAL_T | KIND_PUFF | KIND_MAT_MASK) == 0);
const _: () = assert!(std::mem::size_of::<GpuParams>() == 32 + 64 + 16 + 32 + 16 * (MAX_PRESENT / 4) + 16 * MAX_PRESENT + 32 + 32 + 32 + 48 + 32 * MAX_CURRENTS + 16 + 32 * MAX_MAGNETS + 16 + 48);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_current_heads_clockwise_from_down_at_its_strength() {
        let right = Current::new([10.0, 20.0], 9.0, 90.0, 1.0);
        assert!((right.vel[0] - CURRENT_SPEED).abs() < 1e-3 && right.vel[1].abs() < 1e-3);
        let up = Current::new([10.0, 20.0], 9.0, 180.0, 2.0);
        assert!(up.vel[0].abs() < 1e-3 && (up.vel[1] - 2.0 * CURRENT_SPEED).abs() < 1e-3);
        assert_eq!((up.pos, up.radius, up.rate), ([10.0, 20.0], 9.0, CURRENT_RATE));
    }

    #[test]
    fn overrides_reach_the_named_parameter() {
        let mut p = SimParams::default();
        let (syrup, oil) = (Material::Syrup.index(), Material::Oil.index());
        assert!(p.set("viscosity_syrup", 0.09) && p.mat[syrup].viscosity == 0.09);
        assert!(p.set("nu_syrup", 0.08) && p.mat[syrup].viscosity == 0.08, "old names still work");
        assert!(p.set("syrup.viscosity", 0.07) && p.mat[syrup].viscosity == 0.07, "material.property works too");
        assert!(p.set("density_oil", 800.0) && p.mat[oil].density == 800.0);
        assert!(p.set("tension_water", 0.05) && p.mat[0].tension == 0.05);
        assert!(p.set("sigma_water", 0.06) && p.mat[0].tension == 0.06);
        assert!(p.set("contact_angle_oil", 200.0) && p.mat[oil].contact_angle == 180.0, "clamped to the range");
        assert!(p.set("contact_oil", 30.0) && p.mat[oil].contact_angle == 30.0);
        assert!(p.set("tension_oil_water", 0.03) && p.pair_tension[0][1] == 0.03 && p.pair_tension[1][0] == 0.03);
        assert!(p.set("mat40.density", 1234.0) && p.mat[40].density == 1234.0, "custom rows by number");
        assert!(p.set("pressure_iters", 50.4) && p.pressure_max_iters == 50);
        assert!(p.set("substep_hz", 1.0) && p.fixed_hz == 30.0, "a fixed rate has a floor");
        assert!(p.set("fixed_hz", 0.0) && p.fixed_hz == 0.0, "0 selects the adaptive step");
        assert!(p.set("mix_rate", -5.0) && p.mat[syrup].mix_rate == 0.0, "a negative rate would grow without bound");
        assert!(p.set("diffuse", 2.0) && p.mat[syrup].mix_rate == 2.0);
        assert!(p.set("speed_cap", 0.0) && p.speed_cap == 0.0, "0 switches the cap off");
        assert!(!p.set("gravity", f32::NAN) && p.gravity.is_finite(), "NaN is refused");
        assert!(p.set("gravity", f32::INFINITY) && p.gravity.is_finite(), "infinity is clamped to the range");
        assert!(p.set("speed_cap", 0.3) && p.speed_cap == 1.0, "a cap below free fall would brake the flow");
        assert!(!p.set("no_such_parameter", 1.0));
        assert!(!p.set("viscosity_unobtainium", 1.0));
        assert!(!p.set("rho_oil", 0.8), "relative density is gone: density_oil is in kg/m^3");
        assert!(!p.set("rest_phi", 1.0) && !p.set("fluid_thr", 0.5), "definitions, not settings");
        assert!(!p.set("sigma_oil", 0.03), "ambiguous old name");
    }

    #[test]
    fn every_parameter_is_named_once_and_defaults_are_in_range() {
        let d = SimParams::default();
        for (i, p) in PARAMS.iter().enumerate() {
            assert!(PARAMS[..i].iter().all(|q| q.key != p.key), "{} listed twice", p.key);
            assert!(p.min < p.max, "{}: empty range", p.key);
            assert!(p.off.is_none_or(|o| o < p.min || o > p.max), "{}: off value inside the range", p.key);
            let v = p.read(&d);
            assert_eq!(p.clamp(v), v, "{}: default {} outside its range", p.key, v);
            // Writing a value back through the table changes nothing else.
            let mut q = d;
            p.write(&mut q, v);
            assert!(q == d, "{}: get/put do not round-trip", p.key);
        }
        for (i, p) in MAT_PARAMS.iter().enumerate() {
            assert!(MAT_PARAMS[..i].iter().all(|q| q.key != p.key), "{} listed twice", p.key);
            assert!(p.min < p.max, "{}: empty range", p.key);
            for (row, b) in BUILTINS.iter().enumerate() {
                let v = (p.get)(&d, row);
                // (The ranges are for liquids; steam, which only boiling makes, is 0.6 kg/m³.)
                assert!(!pickable(row) || p.clamp(v) == v, "{} of {}: default {} outside its range", p.key, b.key, v);
                let q = Param::find(&format!("{}_{}", p.key, b.key)).expect("resolves");
                assert_eq!((q.index, q.key), (row, p.key), "{}_{}", p.key, b.key);
                assert_eq!(Param::find(&format!("{}.{}", b.key, p.key)).map(|q| q.index), Some(row));
            }
        }
        for (old, new) in ALIASES {
            assert!(Param::find(old).is_some_and(|p| p.key == *new), "alias {old} -> {new} is dangling");
            assert!(PARAMS.iter().all(|p| p.key != *old), "alias {old} shadows a parameter");
        }
    }

    #[test]
    fn groups_reset_independently() {
        let mut p = SimParams::default();
        p.set("gravity", 3.0);
        p.set("cfl", 2.0);
        p.set("density_oil", 700.0);
        assert!(!p.is_default(&[Group::World]) && !p.is_default(&[Group::Accuracy]) && !p.is_default(&[Group::Liquid]));
        p.reset_groups(&[Group::World, Group::Liquid]);
        assert_eq!(p.gravity, 9.81);
        assert_eq!(p.cfl, 2.0, "solver settings are kept");
        p.reset_groups(&[Group::Accuracy]);
        assert!(p == SimParams::default());
    }

    #[test]
    fn default_fluids_are_ordered_as_advertised() {
        let p = SimParams::default();
        let (water, syrup, oil) = (p.rho(Material::Water), p.rho(Material::Syrup), p.rho(Material::Oil));
        assert!(oil < water && water < syrup, "oil floats on water, syrup sinks in it");
        let nu = |m: Material| p.mat[m.index()].viscosity;
        assert!(nu(Material::Syrup) > 10.0 * nu(Material::Oil) && nu(Material::Oil) > nu(Material::Water));
        assert_eq!(nu(Material::Water), 1.0e-6, "water keeps its real viscosity");
    }

    #[test]
    fn grid_units_follow_the_cell_size() {
        let mut p = SimParams::default();
        assert!((p.g_cells() - 981.0).abs() < 1e-3, "1 cm cells: g = 981 cells/s^2");
        assert!((p.nu_cells(Material::Water) - 0.01).abs() < 1e-6, "water: 0.01 cm^2/s");
        let coarse = p.sigma_cells(Material::Water);
        assert!((coarse - 72.0 / 0.998).abs() < 0.01, "water: 72 cm^3/s^2 over its density");
        let all = [0, 1, 2];
        let dt = p.dt_capillary(&all);
        p.cell_size = 0.001;
        assert!((p.g_cells() / 981.0 - 10.0).abs() < 1e-3);
        assert!((p.sigma_cells(Material::Water) / coarse - 1000.0).abs() < 0.1);
        assert!(p.dt_capillary(&all) < dt);
        assert!(p.dt_capillary(&[2]) > p.dt_capillary(&[0]), "only what is in the tank limits the step");
    }

    #[test]
    fn particles_carry_their_material_and_shares() {
        let p = SimParams::default();
        let at = |m| Particle::new(m, [1.0, 2.0], [3.0, 4.0]);
        assert_eq!(at(Material::Water).material(), Some(Material::Water));
        assert_eq!(at(Material::Oil).pos, [1.0, 2.0]);
        assert_eq!(Particle::default().material(), None);
        let mut blend = at(Material::Syrup);
        blend.s = 0.25;
        assert_eq!((blend.share(Material::Syrup, &p), blend.share(Material::Water, &p), blend.share(Material::Oil, &p)), (0.25, 0.75, 0.0));
        assert_eq!(at(Material::Water).share(Material::Water, &p), 1.0);
        assert!((blend.density(&p) - 1.1).abs() < 1e-6);
        assert_eq!(p.closure(&[1]), vec![0, 1], "syrup brings its diluent");
    }
}
