//! Plain-data types shared between the CPU and the simulation shaders.

use bytemuck::{Pod, Zeroable};

/// Particles per cell at rest (a 2 x 2 lattice).
pub const PPC: u32 = 4;

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
    pub pad: [f32; 2],
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

/// Tunable physics. Lengths are in cells; one cell is nominally 1 cm, so gravity of
/// 981 cells/s^2 is Earth gravity in a tank a few metres wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimParams {
    /// Simulation substeps per second of simulated time.
    pub substep_hz: f32,
    pub gravity: f32,
    pub pressure_iters: u32,
    pub viscosity_iters: u32,
    pub sor_omega: f32,
    /// Relative density of water, syrup, oil.
    pub rho: [f32; 3],
    /// Kinematic viscosity of water, syrup, oil in cells^2/s.
    pub nu: [f32; 3],
    /// Slip relaxation time: how quickly unresolved oil droplets rise through water or syrup.
    pub sep_tau: f32,
    /// Water <-> syrup mixing rate, 1/s.
    pub diffuse: f32,
    /// Share of a cell's density error that is corrected per substep.
    pub density_k: f32,
    pub rest_phi: f32,
    pub fluid_thr: f32,
    /// Cap on particle travel per substep, in cells.
    pub cfl_max: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            substep_hz: 240.0,
            gravity: 981.0,
            pressure_iters: 30,
            viscosity_iters: 6,
            sor_omega: 1.8,
            rho: [1.0, 1.4, 0.9],
            nu: [0.0, 600.0, 30.0],
            sep_tau: 0.25,
            diffuse: 1.5,
            density_k: 0.5,
            rest_phi: 1.0,
            fluid_thr: 0.5,
            cfl_max: 2.5,
        }
    }
}

impl SimParams {
    pub fn dt(&self) -> f32 {
        1.0 / self.substep_hz
    }

    /// Sets a parameter by name (`--set name=value` on the command line).
    pub fn set(&mut self, key: &str, v: f32) -> bool {
        match key {
            "substep_hz" => self.substep_hz = v.max(30.0),
            "gravity" => self.gravity = v,
            "pressure_iters" => self.pressure_iters = v as u32,
            "viscosity_iters" => self.viscosity_iters = v as u32,
            "sor_omega" => self.sor_omega = v,
            "rho_water" => self.rho[0] = v,
            "rho_syrup" => self.rho[1] = v,
            "rho_oil" => self.rho[2] = v,
            "nu_water" => self.nu[0] = v,
            "nu_syrup" => self.nu[1] = v,
            "nu_oil" => self.nu[2] = v,
            "sep_tau" => self.sep_tau = v,
            "diffuse" => self.diffuse = v,
            "density_k" => self.density_k = v,
            "rest_phi" => self.rest_phi = v,
            "fluid_thr" => self.fluid_thr = v,
            "cfl_max" => self.cfl_max = v,
            _ => return false,
        }
        true
    }
}

/// What the mouse is doing to the fluid this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ToolState {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub radius: f32,
    /// 0 = idle, 1 = grab / stir, 2 = erase.
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
}

/// Must match `uniform Params` in `shaders/common.glsl` (std140, 128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuParams {
    pub grid: [i32; 2],
    pub n_slots: u32,
    pub spawn_count: u32,
    pub dt: f32,
    pub gravity: f32,
    pub inv_ppc: f32,
    pub fluid_thr: f32,
    pub rho: [f32; 4],
    pub nu: [f32; 4],
    pub sep_tau: f32,
    pub sep_max: f32,
    pub diffuse: f32,
    pub density_k: f32,
    pub omega: f32,
    pub rest_phi: f32,
    pub vmax: f32,
    pub wall_margin: f32,
    pub tool_pos: [f32; 2],
    pub tool_vel: [f32; 2],
    pub tool_radius: f32,
    pub tool_mode: i32,
    pub tool_strength: f32,
    pub frame: u32,
}

const _: () = assert!(std::mem::size_of::<Particle>() == 48);
const _: () = assert!(std::mem::size_of::<GpuParams>() == 128);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_reach_the_named_parameter() {
        let mut p = SimParams::default();
        assert!(p.set("nu_syrup", 900.0) && p.nu[1] == 900.0);
        assert!(p.set("rho_oil", 0.8) && p.rho[2] == 0.8);
        assert!(p.set("pressure_iters", 50.0) && p.pressure_iters == 50);
        assert!(p.set("substep_hz", 1.0) && p.substep_hz == 30.0, "the substep rate has a floor");
        assert!(!p.set("no_such_parameter", 1.0));
    }

    #[test]
    fn default_fluids_are_ordered_as_advertised() {
        let p = SimParams::default();
        let [water, syrup, oil] = p.rho;
        assert!(oil < water && water < syrup, "oil floats on water, syrup sinks in it");
        assert!(p.nu[1] > 10.0 * p.nu[2] && p.nu[2] > p.nu[0], "syrup is far thicker than oil, oil thicker than water");
        assert!((p.dt() * p.substep_hz - 1.0).abs() < 1e-6);
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
