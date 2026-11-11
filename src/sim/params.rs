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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Material {
    Water,
    Syrup,
    Oil,
}

impl Material {
    pub const ALL: [Material; 3] = [Material::Water, Material::Oil, Material::Syrup];

    pub fn name(self) -> &'static str {
        match self {
            Material::Water => "Water",
            Material::Syrup => "Syrup",
            Material::Oil => "Oil",
        }
    }

    /// Display colour (gamma space), shared by the renderer and the UI.
    pub fn color(self) -> [f32; 3] {
        match self {
            Material::Water => [0.16, 0.50, 0.93],
            Material::Syrup => [0.62, 0.20, 0.05],
            Material::Oil => [0.98, 0.80, 0.22],
        }
    }
}

pub const KIND_DEAD: u32 = 0;
pub const KIND_AQUEOUS: u32 = 1;
pub const KIND_OIL: u32 = 2;

/// Must match `struct Particle` in `shaders/common.glsl` (std430, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub c: [f32; 4],
    pub s: f32,
    pub kind: u32,
    /// Permanent identity, assigned by `FluidSim` when the particle enters the tank. It
    /// survives the periodic sort and seeds the per-particle random numbers, so that a run
    /// does not depend on the order particles happen to sit in memory.
    pub id: u32,
    pub pad: f32,
}

impl Particle {
    pub fn new(material: Material, pos: [f32; 2], vel: [f32; 2]) -> Self {
        let (kind, s) = match material {
            Material::Water => (KIND_AQUEOUS, 0.0),
            Material::Syrup => (KIND_AQUEOUS, 1.0),
            Material::Oil => (KIND_OIL, 0.0),
        };
        Particle { pos, vel, s, kind, ..Default::default() }
    }
}

/// Physics and numerics. Material arrays are indexed water, syrup, oil.
///
/// Material values (20 °C, sources in the README):
///
/// | | density | kinematic viscosity | tension against air | contact angle |
/// |---|---|---|---|---|
/// | water | 998 kg/m³ | 1.0e-6 m²/s | 0.072 N/m | 90° |
/// | syrup | 1.4 × water | 0.06 m²/s (thick golden syrup / honey) | 0.076 N/m | 90° |
/// | oil | 0.9 × water | 6.0e-5 m²/s (olive / sunflower oil) | 0.032 N/m | 90° |
///
/// Oil against water or syrup: 0.025 N/m. Water and syrup are miscible and have no
/// interfacial tension. The 90° contact angles are a neutral default (no wetting
/// preference), not measured values: wetting depends on the solid, which the scene does
/// not describe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimParams {
    // ---- scale -----------------------------------------------------------------------
    /// Physical edge length of one grid cell, m.
    pub cell_size: f32,
    /// m/s².
    pub gravity: f32,

    // ---- materials -------------------------------------------------------------------
    /// Density relative to water.
    pub rho: [f32; 3],
    /// Kinematic viscosity, m²/s.
    pub nu: [f32; 3],
    /// Surface tension against air, N/m.
    pub sigma: [f32; 3],
    /// Interfacial tension between oil and the aqueous phase (water or syrup), N/m.
    pub sigma_oil: f32,
    /// Static contact angle on solids, degrees, measured through the liquid.
    pub contact_angle: [f32; 3],
    /// Diameter of the oil (or water) droplets assumed inside a cell that holds both phases,
    /// in cells. Sets how fast they drift apart: their terminal velocity (see particles.comp).
    pub droplet: f32,
    /// Water <-> syrup mixing rate, 1/s.
    pub diffuse: f32,

    // ---- time step -------------------------------------------------------------------
    /// Courant number: the most a particle may travel in one substep, in cells.
    pub cfl: f32,
    /// Particle advection: 1 = forward Euler, 2 = midpoint rule through the grid velocity.
    pub advection_order: u32,
    /// Spray in free flight (less than half the rest density around it) may travel this many
    /// times `cfl` per substep: its path is ballistic and does not depend on the step. 1 makes
    /// the fastest droplet set the step for the whole tank again.
    pub spray_cfl: f32,
    /// Longest substep, s. At CFL 4 a slowly moving pool would otherwise take 1/60 s steps,
    /// in which particles converging on a spot travel up to three cells before the density
    /// correction sees them (refilling a hole then left 13 to 21 particles in one cell
    /// instead of 9).
    pub dt_max: f32,
    /// Fixed substep rate in Hz for comparisons; 0 picks the step from the CFL condition.
    pub fixed_hz: f32,

    // ---- solver tolerances -----------------------------------------------------------
    /// Pressure solve: stop when |r| <= tol * |b|.
    pub pressure_tol: f32,
    pub pressure_max_iters: u32,
    /// Density-correction solve (same matrix), looser: it only nudges positions.
    pub correction_tol: f32,
    /// Viscosity solve: stop when |r| <= tol * |r0|.
    pub viscosity_tol: f32,
    pub viscosity_max_iters: u32,

    // ---- stabilisation (see README, "Numerical stabilisation") -------------------------
    /// Most of a cell's density error corrected in one substep.
    pub density_k: f32,
    /// Time constant of the density correction, s: a substep of length dt corrects
    /// 1 - exp(-dt / tau) of the error (at most `density_k`). The default matches the
    /// previous fixed 0.5 per 1/240 s substep.
    pub density_tau: f32,
    pub rest_phi: f32,
    pub fluid_thr: f32,
    /// Largest curvature surface tension acts on, 1/cells (radius 1/kappa_max cells).
    pub kappa_max: f32,
    /// Particle speed cap, as a multiple of the free-fall speed across the height of the
    /// tank, sqrt(2 g H). At 1 a particle can still reach the lid from the floor, so nothing
    /// that stays inside the tank is slowed visibly; what it removes are the few impact-jet
    /// particles (about 0.1 %) that would otherwise set the time step for everything else.
    /// 0 switches the cap off. Inactive without gravity.
    pub speed_cap: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            cell_size: 0.01,
            gravity: 9.81,
            rho: [1.0, 1.4, 0.9],
            nu: [1.0e-6, 0.06, 6.0e-5],
            sigma: [0.072, 0.076, 0.032],
            sigma_oil: 0.025,
            contact_angle: [90.0, 90.0, 90.0],
            droplet: 1.0,
            diffuse: 1.5,
            cfl: 4.0,
            advection_order: 2,
            spray_cfl: 4.0,
            dt_max: 1.0 / 80.0,
            fixed_hz: 0.0,
            pressure_tol: 1e-4,
            pressure_max_iters: 60,
            correction_tol: 1e-2,
            viscosity_tol: 1e-3,
            viscosity_max_iters: 200,
            density_k: 0.5,
            density_tau: (1.0 / 240.0) / std::f32::consts::LN_2,
            rest_phi: 1.0,
            fluid_thr: 0.5,
            kappa_max: 0.5,
            speed_cap: 1.0,
        }
    }
}

/// Material properties in grid units, as the shaders see them.
#[derive(Clone, Copy, Debug)]
pub struct GridUnits {
    /// cells/s².
    pub gravity: f32,
    /// cells²/s.
    pub nu: [f32; 3],
    /// Tension divided by water's density, cells³/s².
    pub sigma: [f32; 3],
    pub sigma_oil: f32,
}

impl SimParams {
    pub const WATER_DENSITY: f32 = 998.0;

    pub fn gpu_units(&self) -> GridUnits {
        let dx = self.cell_size;
        let tension = |s: f32| s / (Self::WATER_DENSITY * dx * dx * dx);
        GridUnits {
            gravity: self.gravity / dx,
            nu: self.nu.map(|n| n / (dx * dx)),
            sigma: self.sigma.map(tension),
            sigma_oil: tension(self.sigma_oil),
        }
    }

    /// Gravity in cells/s², the unit the validation scenes work in.
    pub fn g_cells(&self) -> f32 {
        self.gravity / self.cell_size
    }

    /// Kinematic viscosity of material `i` in cells²/s.
    pub fn nu_cells(&self, i: usize) -> f32 {
        self.nu[i] / (self.cell_size * self.cell_size)
    }

    /// The particle speed cap in cells/s for a tank `height` cells tall (infinite when off).
    pub fn speed_cap_cells(&self, height: u32) -> f32 {
        let g = self.g_cells();
        if self.speed_cap > 0.0 && g > 0.0 { self.speed_cap * (2.0 * g * height as f32).sqrt() } else { 1e30 }
    }

    /// Explicit surface tension is stable only for steps below the capillary-wave limit
    /// (Brackbill et al. 1992): dt < sqrt(rho_min dx³ / (2 pi sigma)). Returns that limit with a
    /// safety factor of 0.5, in seconds (grid units: dx = 1).
    pub fn dt_capillary(&self) -> f32 {
        let u = self.gpu_units();
        let sigma = u.sigma.iter().copied().fold(u.sigma_oil, f32::max);
        if sigma <= 0.0 {
            return f32::INFINITY;
        }
        let rho_min = self.rho.iter().copied().fold(f32::INFINITY, f32::min);
        0.5 * (rho_min / (2.0 * std::f32::consts::PI * sigma)).sqrt()
    }

    /// Sets a parameter by name (`--set name=value` on the command line). Units as in the
    /// struct: SI for physical quantities.
    pub fn set(&mut self, key: &str, v: f32) -> bool {
        let m = |name: &str| ["water", "syrup", "oil"].iter().position(|&n| n == name);
        if let Some((what, mat)) = key.rsplit_once('_')
            && let Some(i) = m(mat)
        {
            match what {
                "rho" => self.rho[i] = v,
                "nu" => self.nu[i] = v.max(0.0),
                "sigma" => self.sigma[i] = v.max(0.0),
                "contact" => self.contact_angle[i] = v.clamp(0.0, 180.0),
                _ => return false,
            }
            return true;
        }
        match key {
            "cell_size" => self.cell_size = v.max(1e-5),
            "gravity" => self.gravity = v,
            "sigma_oil" => self.sigma_oil = v.max(0.0),
            "droplet" => self.droplet = v.max(0.05),
            "diffuse" => self.diffuse = v,
            "cfl" => self.cfl = v.clamp(0.05, 8.0),
            "advection_order" => self.advection_order = if v >= 2.0 { 2 } else { 1 },
            "speed_cap" => self.speed_cap = v.max(0.0),
            "spray_cfl" => self.spray_cfl = v.clamp(1.0, 8.0),
            "dt_max" => self.dt_max = v.max(1e-5),
            "fixed_hz" | "substep_hz" => self.fixed_hz = if v <= 0.0 { 0.0 } else { v.max(30.0) },
            "pressure_tol" => self.pressure_tol = v.max(1e-7),
            "pressure_iters" | "pressure_max_iters" => self.pressure_max_iters = (v as u32).max(1),
            "correction_tol" => self.correction_tol = v.max(1e-7),
            "viscosity_tol" => self.viscosity_tol = v.max(1e-7),
            "viscosity_iters" | "viscosity_max_iters" => self.viscosity_max_iters = (v as u32).max(1),
            "density_k" => self.density_k = v,
            "density_tau" => self.density_tau = v.max(1e-5),
            "rest_phi" => self.rest_phi = v,
            "fluid_thr" => self.fluid_thr = v,
            "kappa_max" => self.kappa_max = v.max(0.01),
            _ => return false,
        }
        true
    }
}

/// Speed the push tool drives liquid outwards at, cells/s (carried in `ToolState::vel[0]`).
pub const PUSH_SPEED: f32 = 220.0;

/// What the mouse is doing to the fluid this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ToolState {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub radius: f32,
    /// 0 = idle, 1 = grab / stir, 2 = erase, 3 = push (speed in `vel[0]`), 4 = jet nozzle.
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

    /// Holds the liquid inside a stream's nozzle (faucet or jet) at the launch velocity
    /// `vel`. Rows then leave at exactly the speed they were laid out for, so consecutive
    /// frames join without a gap (left alone, new rows take up the faster flow of the
    /// accelerating stream below at once and run ahead); and a nozzle whose outlet is
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

/// Must match `uniform Params` in `shaders/common.glsl` (std140, 208 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuParams {
    pub grid: [i32; 2],
    pub n_slots: u32,
    pub spawn_count: u32,
    pub dt_fixed: f32,
    pub gravity: f32,
    pub inv_ppc: f32,
    pub fluid_thr: f32,
    pub rho: [f32; 4],
    pub nu: [f32; 4],
    /// Tension against air of water, syrup, oil, and oil against the aqueous phase.
    pub sigma: [f32; 4],
    /// cos and sin of the contact angle of water and syrup.
    pub contact_aq: [f32; 4],
    /// cos and sin of the contact angle of oil; kappa_max; unused.
    pub contact_oil: [f32; 4],
    pub sep_tau: f32,
    pub cfl: f32,
    pub diffuse: f32,
    pub density_k: f32,
    pub dt_max: f32,
    pub rest_phi: f32,
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
    /// Particle speed cap (cells/s, huge = off); advection order; spray CFL multiple; unused.
    pub ext: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Particle>() == 48);
const _: () = assert!(std::mem::size_of::<GpuParams>() == 208);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_reach_the_named_parameter() {
        let mut p = SimParams::default();
        assert!(p.set("nu_syrup", 0.09) && p.nu[1] == 0.09);
        assert!(p.set("rho_oil", 0.8) && p.rho[2] == 0.8);
        assert!(p.set("sigma_water", 0.05) && p.sigma[0] == 0.05);
        assert!(p.set("contact_oil", 200.0) && p.contact_angle[2] == 180.0);
        assert!(p.set("pressure_iters", 50.0) && p.pressure_max_iters == 50);
        assert!(p.set("substep_hz", 1.0) && p.fixed_hz == 30.0, "a fixed rate has a floor");
        assert!(p.set("fixed_hz", 0.0) && p.fixed_hz == 0.0, "0 selects the adaptive step");
        assert!(!p.set("no_such_parameter", 1.0));
        assert!(!p.set("nu_mercury", 1.0));
    }

    #[test]
    fn default_fluids_are_ordered_as_advertised() {
        let p = SimParams::default();
        let [water, syrup, oil] = p.rho;
        assert!(oil < water && water < syrup, "oil floats on water, syrup sinks in it");
        assert!(p.nu[1] > 10.0 * p.nu[2] && p.nu[2] > p.nu[0], "syrup is far thicker than oil, oil thicker than water");
        assert_eq!(p.nu[0], 1.0e-6, "water keeps its real viscosity");
    }

    #[test]
    fn grid_units_follow_the_cell_size() {
        let mut p = SimParams::default();
        let u = p.gpu_units();
        assert!((u.gravity - 981.0).abs() < 1e-3, "1 cm cells: g = 981 cells/s^2");
        assert!((u.nu[0] - 0.01).abs() < 1e-6, "water: 0.01 cm^2/s");
        assert!((u.sigma[0] - 72.0 / 0.998).abs() < 0.01, "water: 72 cm^3/s^2 over its density");
        p.cell_size = 0.001;
        let fine = p.gpu_units();
        assert!((fine.gravity / u.gravity - 10.0).abs() < 1e-3);
        assert!((fine.sigma[0] / u.sigma[0] - 1000.0).abs() < 0.1);
        assert!(p.dt_capillary() < SimParams::default().dt_capillary());
    }

    #[test]
    fn materials_map_to_the_two_particle_kinds() {
        let at = |m| Particle::new(m, [1.0, 2.0], [3.0, 4.0]);
        assert_eq!((at(Material::Water).kind, at(Material::Water).s), (KIND_AQUEOUS, 0.0));
        assert_eq!((at(Material::Syrup).kind, at(Material::Syrup).s), (KIND_AQUEOUS, 1.0));
        assert_eq!(at(Material::Oil).kind, KIND_OIL);
        assert_eq!(at(Material::Oil).pos, [1.0, 2.0]);
    }
}
