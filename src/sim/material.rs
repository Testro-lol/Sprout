//! The material system: what a liquid *is*, as data.
//!
//! Two separate things (README, "Materials"):
//!
//! - A **material definition** (`MatProps`, plus a name and a stable id in
//!   `crate::library`) is saved, versioned data. Every capability is an opt-in field with a
//!   neutral default: a material that sets none of them is a Newtonian liquid of constant
//!   properties.
//! - **Particle state** (`Particle`: material index, concentration `s`, enthalpy
//!   temperature `t`) evolves during a run. Particles never carry properties: the shaders
//!   look them up in the GPU material table (`GpuMat`, `MatBuf` in `common.glsl`) each
//!   substep, so a changed definition takes effect on every particle of that material.
//!
//! The live table (`SimParams::mat`) is a fixed array of plain numbers, indexed by
//! `Material`. Built-ins occupy the first `BUILTIN_COUNT` rows; custom liquids follow.

use bytemuck::{Pod, Zeroable};

/// Rows of the live material table (and of the GPU table).
pub const MAX_MATERIALS: usize = 96;
/// Most materials a tank can hold at once: each needs accumulator channels on the grid.
pub const MAX_PRESENT: usize = 12;
/// Immiscible phase groups.
pub const GROUPS: usize = 4;

/// Reference temperature of the material data, °C.
pub const T_REF: f32 = 20.0;

/// Index of a material in the live table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Material(pub u8);

#[allow(non_upper_case_globals)]
impl Material {
    // The built-in rows, in table order (presets.rs: BUILTINS). Water is row 0 (common.glsl:
    // ROW_WATER); Salt, Steam and Resin are never offered by the picker (what evaporating brine
    // leaves, what boiling water turns into, what the two-part liquids set into).
    pub const Water: Material = Material(0);
    pub const Honey: Material = Material(1);
    pub const Oil: Material = Material(2);
    pub const Mud: Material = Material(3);
    pub const Ink: Material = Material(4);
    pub const Chocolate: Material = Material(5);
    pub const Clay: Material = Material(6);
    pub const Blood: Material = Material(7);
    pub const Mercury: Material = Material(8);
    pub const Lava: Material = Material(9);
    pub const Glass: Material = Material(10);
    pub const Milk: Material = Material(11);
    pub const Slime: Material = Material(12);
    pub const Ferrofluid: Material = Material(13);
    pub const Brine: Material = Material(14);
    pub const Wax: Material = Material(15);
    pub const Oobleck: Material = Material(16);
    pub const Gallium: Material = Material(17);
    pub const Beer: Material = Material(18);
    pub const Soap: Material = Material(19);
    pub const Alcohol: Material = Material(20);
    pub const Nitrogen: Material = Material(21);
    pub const Acid: Material = Material(22);
    pub const Lye: Material = Material(23);
    pub const GlowFluid: Material = Material(24);
    pub const Iron: Material = Material(25);
    pub const Polish: Material = Material(26);
    pub const Blank: Material = Material(27);
    pub const Thick: Material = Material(28);
    pub const Heavy: Material = Material(29);
    pub const Light: Material = Material(30);
    pub const Beady: Material = Material(31);
    pub const Creeper: Material = Material(32);
    pub const Putty: Material = Material(33);
    pub const Paste: Material = Material(34);
    pub const Thinner: Material = Material(35);
    pub const Jammer: Material = Material(36);
    pub const Magnetic: Material = Material(37);
    pub const Glowing: Material = Material(38);
    pub const Fizzy: Material = Material(39);
    pub const Sudsy: Material = Material(40);
    pub const Vanishing: Material = Material(41);
    pub const Grainy: Material = Material(42);
    pub const Warming: Material = Material(43);
    pub const Setting: Material = Material(44);
    pub const Mirror: Material = Material(45);
    pub const Pearly: Material = Material(46);
    pub const Glitter: Material = Material(47);
    pub const Snowy: Material = Material(48);
    pub const Mood: Material = Material(49);
    pub const Dye: Material = Material(50);
    pub const Silt: Material = Material(51);
    pub const PartA: Material = Material(52);
    pub const PartB: Material = Material(53);
    pub const Ember: Material = Material(54);
    pub const Sheen: Material = Material(55);
    pub const Salt: Material = Material(56);
    pub const Steam: Material = Material(57);
    pub const Resin: Material = Material(58);

    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Display name of a built-in row.
    pub fn name(self) -> &'static str {
        BUILTINS.get(self.index()).map_or("Custom", |b| b.name)
    }

    /// Default body colour of a built-in row (gamma space).
    pub fn color(self) -> [f32; 3] {
        BUILTINS.get(self.index()).map_or([0.6; 3], |b| (b.props)().color)
    }
}

/// Phase groups: liquids of one group mix, liquids of different groups keep an interface
/// with the tension of `SimParams::pair_tension`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseGroup {
    Aqueous = 0,
    Oily = 1,
    Metal = 2,
    Silicate = 3,
}

impl PhaseGroup {
    pub const ALL: [PhaseGroup; GROUPS] = [PhaseGroup::Aqueous, PhaseGroup::Oily, PhaseGroup::Metal, PhaseGroup::Silicate];

    pub fn name(self) -> &'static str {
        match self {
            PhaseGroup::Aqueous => "Aqueous",
            PhaseGroup::Oily => "Oil-like",
            PhaseGroup::Metal => "Liquid metal",
            PhaseGroup::Silicate => "Silicate melt",
        }
    }

    pub fn from_index(i: u32) -> PhaseGroup {
        PhaseGroup::ALL[(i as usize).min(GROUPS - 1)]
    }
}

/// How viscosity depends on the shear rate.
pub const RHEO_NEWTONIAN: u32 = 0;
/// Carreau: nu = nu_inf + (nu0 - nu_inf) (1 + (lambda g)^2)^((n-1)/2).
pub const RHEO_CARREAU: u32 = 1;
/// Herschel-Bulkley (Bingham for n = 1): tau = tau_y + K g^n, regularised (see visc_setup.comp).
pub const RHEO_HERSCHEL_BULKLEY: u32 = 2;
/// Shear thickening: nu = nu0 (1 + (lambda g)^2)^((n-1)/2) with n > 1, up to `visc_ratio` x nu0.
pub const RHEO_THICKENING: u32 = 3;

/// How viscosity depends on temperature.
pub const TEMP_NONE: u32 = 0;
/// Arrhenius: nu(T) = nu_ref exp(B (1/T - 1/T_ref)), T in kelvin, B = `temp_b`.
pub const TEMP_ARRHENIUS: u32 = 1;
/// Vogel-Fulcher-Tammann: nu(T) = nu_ref exp(B/(T - C) - B/(T_ref - C)), B = `temp_b`, C = `temp_c` (K).
pub const TEMP_VFT: u32 = 2;

/// Every number that defines a material, in SI / display units. Plain data: saved to disk
/// field by field (`crate::library`) and packed for the GPU by `SimParams::gpu_materials`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatProps {
    // ---- identity in the phase system --------------------------------------------------
    /// Phase group (`PhaseGroup` as a number).
    pub group: u32,
    /// The material this one is a concentrate of: a particle with concentration `s` is
    /// `s` parts of this material and `1 - s` parts of the diluent. -1: none.
    pub diluent: i32,

    // ---- mechanics -----------------------------------------------------------------------
    /// kg/m³, liquid.
    pub density: f32,
    /// kg/m³ when frozen; 0 = as liquid.
    pub density_solid: f32,
    /// Kinematic viscosity at `temp_ref` and zero shear rate, m²/s.
    pub viscosity: f32,
    pub rheology: u32,
    /// Herschel-Bulkley yield stress, Pa.
    pub yield_stress: f32,
    /// Power-law index n (Carreau, Herschel-Bulkley, thickening).
    pub flow_index: f32,
    /// Carreau / thickening time constant lambda, s.
    pub shear_time: f32,
    /// Carreau: nu_inf / nu0. Thickening: largest nu / nu0.
    pub visc_ratio: f32,
    pub temp_model: u32,
    /// Arrhenius or VFT B, K.
    pub temp_b: f32,
    /// VFT C, K.
    pub temp_c: f32,
    /// Temperature at which `viscosity` holds, °C.
    pub temp_ref: f32,
    /// Surface tension against air, N/m.
    pub tension: f32,
    /// Static contact angle on solids, degrees, through the liquid.
    pub contact_angle: f32,

    // ---- dissolved or suspended content ----------------------------------------------------
    /// How fast the concentration evens out between touching particles, 1/s.
    pub mix_rate: f32,
    /// Diameter of the suspended grains, m: the concentrate settles (or creams) through the
    /// diluent at the terminal velocity of such a grain. 0 = dissolved, no settling.
    pub grain: f32,
    /// Concentration of freshly emitted liquid: the share that is concentrate, the rest its
    /// diluent (mud: 0.5). 1 for a pure liquid.
    pub load: f32,

    // ---- thermal -----------------------------------------------------------------------------
    /// J/(kg K).
    pub heat_capacity: f32,
    /// W/(m K).
    pub conductivity: f32,
    /// Volumetric thermal expansion of the liquid, 1/K: its density falls by this share per
    /// kelvin above `temp_ref` (where `density` holds). Heated liquid rises: convection.
    pub expansion: f32,
    /// Solidus and liquidus, °C. `melt_hi` <= -273 means the material never freezes.
    pub melt_lo: f32,
    pub melt_hi: f32,
    /// Latent heat of fusion, J/kg.
    pub latent: f32,
    /// Grey-body emissivity (radiative loss at the free surface; glow).
    pub emissivity: f32,
    /// Boiling point, °C (1e9 = never), and latent heat of vaporisation, J/kg.
    pub boil: f32,
    pub latent_vap: f32,
    /// What boiling leaves behind: a material index (-1 none) and the volume share of it.
    pub residue: i32,
    pub residue_share: f32,
    /// Shift of the freezing range per unit concentration when this material is the
    /// concentrate of its diluent, K (brine: -21).
    pub freeze_shift: f32,
    /// Temperature newly emitted liquid has, °C.
    pub spawn_temp: f32,

    // ---- elasticity, magnetism, skin ------------------------------------------------------------
    /// Shear modulus, Pa, and relaxation time, s (upper-convected Maxwell). 0 = inelastic.
    pub shear_modulus: f32,
    pub relax_time: f32,
    /// Magnetic volume susceptibility (SI), and saturation magnetisation as mu0 Ms, T.
    pub susceptibility: f32,
    pub saturation: f32,
    /// Strength of a surface skin, N/m: a yield stress `skin / cell` in free-surface cells.
    pub skin: f32,

    // ---- optics (README, "Look") ------------------------------------------------------------------
    // A liquid's colour comes from how it absorbs and scatters light (Beer-Lambert), so that
    // mixtures add up as real solutions do: a little honey in water is pale gold, a little
    // blood a pink-red cloud, never a hue in between the two colours. What it absorbs is its
    // dissolved colour (`color`, `opacity`) plus its suspended particles (`cloud_*`); what it
    // scatters comes from those particles alone (README, "Particles in the liquid").
    /// Colour of what is dissolved in it, gamma space: the colour of light that has come
    /// through `tint_depth` cells of the liquid without its particles.
    pub color: [f32; 3],
    /// Colour when frozen.
    pub color_solid: [f32; 3],
    /// 0 clear .. 1 opaque: how deep the dissolved colour builds up, `tint_depth` =
    /// 0.5 + 30 (1 - opacity)² cells (water 30, honey 3, ink half a cell).
    pub opacity: f32,
    /// 0 dielectric .. 1 mirror metal.
    pub metal: f32,
    /// Strength of the specular streak.
    pub gloss: f32,
    /// 1: fast, thin liquid turns to white water.
    pub foam: f32,
    /// Volume fraction of what is dissolved in it and does not evaporate (sugar in honey
    /// 0.8, salt in brine 0.12, plasma proteins in blood 0.08, dye in ink 0.05). With its
    /// particles, it is what a film of it leaves behind when it dries (README, "Films, stains
    /// and drips"). A liquid that never boils does not dry at all.
    pub solids: f32,

    // ---- particles in the liquid (README, "Particles in the liquid") -------------------------
    // Two populations, each described as the real material is: how much there is (volume
    // fraction), how big the particles are, their refractive index or reflectance, their own
    // colour and shape. Everything they look like follows from that: how cloudy the liquid is
    // and in which colour (Rayleigh to Mie scattering: fine particles scatter blue and pass
    // orange, coarse ones scatter white and mostly forwards), whether platelets show the flow
    // (Jeffery orbits against Brownian rotation), how specks settle, pile up and are stirred
    // up again (Stokes drag, Shields), and how much of them shows through the liquid.
    /// Volume fraction of fine particles (colloids, droplets, pigment): the cloudiness.
    pub cloud: f32,
    /// Their diameter, m (casein micelles 0.1 µm, milk fat 3 µm, red cells 7.5 µm).
    pub cloud_size: f32,
    /// Their refractive index (fat 1.46, clay 1.56, titanium white 2.6): how strongly they
    /// scatter against the liquid's own `ior`. Equal to it: they vanish.
    pub cloud_ior: f32,
    /// Their own colour, gamma space: the colour of a deep layer of them (a powder of them,
    /// or the liquid where they dominate). Sets how much light they absorb.
    pub cloud_color: [f32; 3],
    /// Their thickness over their diameter: 1 spheres, small values platelets (mica 0.05),
    /// which the flow lines up so that the liquid shows its shear.
    pub cloud_shape: f32,
    /// Volume fraction of coarse particles, seen one by one as specks: grains, flakes, glitter.
    pub dust: f32,
    /// Their diameter, m.
    pub dust_size: f32,
    /// Their density, kg/m³: they settle through (or rise in) the liquid at their terminal
    /// velocity, rest on what is below them and are stirred up by a fast enough flow.
    pub dust_density: f32,
    /// Their colour, gamma space (also the tint of what a shiny one reflects).
    pub dust_color: [f32; 3],
    /// Their thickness over their diameter: 1 grains, small values flakes that tumble in the
    /// flow and flash as they turn to the light.
    pub dust_shape: f32,
    /// How much of the light that meets their face they reflect like a mirror (quartz 0.05,
    /// mica 0.2, aluminised glitter 0.9).
    pub dust_reflect: f32,

    // ---- the 2026-11-16 properties (README, "New properties") -------------------------------
    /// Heat the liquid makes inside itself, W/kg (a radioactive or slowly reacting liquid).
    pub self_heat: f32,
    /// Share of the liquid exposed at a free surface that evaporates per second at 20 °C
    /// (doubling every 10 K), below its boiling point. 0: only boiling removes it.
    pub volatility: f32,
    /// Internal friction (tan of the friction angle) of a granular liquid: a yield stress of
    /// `friction` x the pressure, so it piles up at its angle of repose. Scales with the
    /// concentration cubed for a concentrate (a granular paste flows when watered down).
    pub friction: f32,
    /// A reaction: in a cell that also holds `react_with` (a row, -1: none) the liquid turns
    /// into `react_into` (a row; -1: it is used up) at `react_rate` per second, giving off
    /// `react_heat` J/kg.
    pub react_with: i32,
    pub react_into: i32,
    pub react_rate: f32,
    pub react_heat: f32,
    /// Seconds the foam it makes lasts (water: half a second of white water; beer: a head).
    pub froth: f32,
    /// 0 none .. 1 soda: bubbles of dissolved gas rise through it and feed its foam.
    pub fizz: f32,
    /// 0 none .. 1: how much visible vapour it gives off where it evaporates or boils away.
    pub fume: f32,
    /// 0 none .. 2: light of its own (`glow_color`), whatever its temperature.
    pub luminous: f32,
    pub glow_color: [f32; 3],
    /// 0 .. 1: the share of that light it gives only where it is stirred (bioluminescence).
    pub glow_shear: f32,
    /// Thermochromic: its colour turns from `color` at `thermo_lo` to `color_hot` at
    /// `thermo_hi` (°C). Off while `thermo_hi` <= `thermo_lo`.
    pub color_hot: [f32; 3],
    pub thermo_lo: f32,
    pub thermo_hi: f32,
    /// Refractive index: how much its surface reflects at each angle (Fresnel) and how far
    /// it bends what is seen through it (water 1.33).
    pub ior: f32,
}

impl Default for MatProps {
    fn default() -> Self {
        MatProps {
            group: 0,
            diluent: -1,
            density: 998.0,
            density_solid: 0.0,
            viscosity: 1.0e-6,
            rheology: RHEO_NEWTONIAN,
            yield_stress: 0.0,
            flow_index: 1.0,
            shear_time: 1.0,
            visc_ratio: 1.0,
            temp_model: TEMP_NONE,
            temp_b: 0.0,
            temp_c: 0.0,
            temp_ref: T_REF,
            tension: 0.072,
            contact_angle: 90.0,
            mix_rate: 0.0,
            grain: 0.0,
            load: 1.0,
            heat_capacity: 4182.0,
            conductivity: 0.6,
            // Water: 998 kg/m³ at 20 °C, 958 at 100 °C (a linear fit over that range; it
            // misses the maximum at 4 °C).
            expansion: 4.0e-4,
            melt_lo: -1.0e9,
            melt_hi: -1.0e9,
            latent: 0.0,
            emissivity: 0.9,
            boil: 1.0e9,
            latent_vap: 0.0,
            residue: -1,
            residue_share: 0.0,
            freeze_shift: 0.0,
            spawn_temp: T_REF,
            shear_modulus: 0.0,
            relax_time: 0.0,
            susceptibility: 0.0,
            saturation: 0.05,
            skin: 0.0,
            color: [0.17, 0.52, 0.94],
            color_solid: [0.8, 0.9, 1.0],
            opacity: 0.0,
            metal: 0.0,
            gloss: 0.3,
            foam: 0.0,
            solids: 0.0,
            cloud: 0.0,
            cloud_size: 1.0e-6,
            cloud_ior: 1.5,
            cloud_color: [0.95, 0.95, 0.95],
            cloud_shape: 1.0,
            dust: 0.0,
            dust_size: 3.0e-4,
            dust_density: 2650.0,
            dust_color: [0.80, 0.75, 0.65],
            dust_shape: 1.0,
            dust_reflect: 0.05,
            self_heat: 0.0,
            volatility: 0.0,
            friction: 0.0,
            react_with: -1,
            react_into: -1,
            react_rate: 0.0,
            react_heat: 0.0,
            froth: 0.0,
            fizz: 0.0,
            fume: 0.0,
            luminous: 0.0,
            glow_color: [0.4, 1.0, 0.5],
            glow_shear: 0.0,
            color_hot: [0.95, 0.35, 0.45],
            thermo_lo: 0.0,
            thermo_hi: 0.0,
            ior: 1.33,
        }
    }
}

impl MatProps {
    /// Whether the material can freeze at all.
    pub fn freezes(&self) -> bool {
        self.melt_hi > -273.0
    }

    /// ln of the factor by which the viscosity at `t` (°C) differs from the value at
    /// `temp_ref` (mirrors `temp_law` in grid_prep.comp).
    pub fn temp_factor_ln(&self, t: f32) -> f32 {
        let tk = (t + 273.15).max(1.0);
        let tref = self.temp_ref + 273.15;
        let x = match self.temp_model {
            TEMP_ARRHENIUS => self.temp_b * (1.0 / tk - 1.0 / tref),
            TEMP_VFT => self.temp_b / (tk - self.temp_c).max(5.0) - self.temp_b / (tref - self.temp_c).max(5.0),
            _ => 0.0,
        };
        x.clamp(-30.0, 30.0)
    }

    /// Kinematic viscosity of the liquid at `t` (°C), zero shear, m²/s.
    pub fn viscosity_at(&self, t: f32) -> f32 {
        self.viscosity * self.temp_factor_ln(t).exp()
    }

    /// Volumetric heat capacity, J/(m³ K).
    pub fn heat_capacity_vol(&self) -> f32 {
        self.density * self.heat_capacity
    }

    /// Whether the liquid makes or needs heat by itself, so that a tank holding it carries
    /// temperatures from the start.
    pub fn makes_heat(&self) -> bool {
        self.self_heat > 0.0 || (self.react_with >= 0 && self.react_heat != 0.0)
    }

    pub fn phase_group(&self) -> PhaseGroup {
        PhaseGroup::from_index(self.group)
    }
}

pub use super::presets::{BUILTINS, pickable};

/// Explicit exceptions to the mixing rules (README, "Mixing rules"), keyed by
/// (concentrate, diluent). By default a concentrate's viscosity blends with its diluent's
/// logarithmically, ln nu = ln nu_d + s (ln nu_c - ln nu_d); an entry here raises s to
/// `visc_exponent` instead, for suspensions whose structure collapses on dilution.
pub struct MixOverride {
    pub concentrate: Material,
    pub diluent: Material,
    pub visc_exponent: f32,
}

pub static MIX_OVERRIDES: &[MixOverride] = &[
    // Clay and oobleck lose their gel / jammed structure steeply when watered down
    // (viscosity of concentrated suspensions rises steeply near packing, Krieger-Dougherty).
    MixOverride { concentrate: Material::Clay, diluent: Material::Water, visc_exponent: 2.0 },
    MixOverride { concentrate: Material::Oobleck, diluent: Material::Water, visc_exponent: 3.0 },
];

/// The viscosity blending exponent of `row` against its diluent (1 unless overridden).
pub fn visc_exponent(row: usize, diluent: i32) -> f32 {
    MIX_OVERRIDES.iter().find(|o| o.concentrate.index() == row && o.diluent.0 as i32 == diluent).map_or(1.0, |o| o.visc_exponent)
}

/// Interfacial tension between two phase groups, N/m, at the defaults.
pub fn default_pair_tension() -> [[f32; GROUPS]; GROUPS] {
    let mut t = [[0.0; GROUPS]; GROUPS];
    let mut set = |a: PhaseGroup, b: PhaseGroup, v: f32| {
        t[a as usize][b as usize] = v;
        t[b as usize][a as usize] = v;
    };
    // Olive / sunflower oil against water: 0.02 - 0.03 N/m.
    set(PhaseGroup::Aqueous, PhaseGroup::Oily, 0.025);
    // Mercury against water: 0.375 N/m (CRC Handbook); gallium is similar under its oxide.
    set(PhaseGroup::Aqueous, PhaseGroup::Metal, 0.375);
    // Mercury against oils: 0.35 - 0.38 N/m.
    set(PhaseGroup::Oily, PhaseGroup::Metal, 0.36);
    // Silicate melts against water or oil: a sandbox choice (they meet only while the
    // water flashes off); against liquid metal 0.4 N/m, the order of slag / metal systems.
    set(PhaseGroup::Aqueous, PhaseGroup::Silicate, 0.2);
    set(PhaseGroup::Oily, PhaseGroup::Silicate, 0.2);
    set(PhaseGroup::Metal, PhaseGroup::Silicate, 0.4);
    t
}

/// One row of the GPU material table. Must match `struct Mat` in `shaders/common.glsl`
/// (std430, 160 bytes). Everything is in grid units.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuMat {
    /// Phase group; diluent row (-1 none); accumulator channel of its weight (-1: implied);
    /// accumulator channel of its concentration (-1 none).
    pub ia: [i32; 4],
    /// Family index (-1 none); rheology model; temperature model; residue row (-1 none).
    pub ib: [i32; 4],
    /// Relative density (liquid); ln of the kinematic viscosity at the reference
    /// temperature (cells²/s); tension against air / rho_water (cells³/s²); mixing rate (1/s).
    pub a: [f32; 4],
    /// cos, sin of the contact angle; grain diameter (cells); relative density when frozen.
    pub b: [f32; 4],
    /// Yield stress / rho_water (cells²/s²); flow index; shear time (s); viscosity ratio.
    pub rheo: [f32; 4],
    /// Temperature law: B (K), C (K), reference temperature (K); skin yield / rho_water (cells³/s²).
    pub temp: [f32; 4],
    /// Volumetric heat capacity relative to water; conductivity (cells² / s, as a
    /// diffusivity of water-equivalent heat); solidus, liquidus (°C).
    pub th: [f32; 4],
    /// Latent heat as degrees (L / c); emissivity; boiling point (°C); latent heat of
    /// vaporisation as degrees.
    pub th2: [f32; 4],
    /// Shear modulus / rho_water (cells²/s²); relaxation time (s); susceptibility; residue share.
    pub el: [f32; 4],
    /// Freeze shift per unit concentration (K); load; viscosity blending exponent against
    /// the diluent (`MIX_OVERRIDES`); thermal expansion (1/K).
    pub misc: [f32; 4],
    /// Reaction: partner row (-1 none), product row (-1: used up), the partner's family index
    /// in this tank (-1: none) and its phase group.
    pub ic: [i32; 4],
    /// Reaction rate (1/s); its heat as degrees of this liquid (J/kg / c); self-heating (K/s);
    /// volatility (1/s).
    pub rx: [f32; 4],
    /// Friction coefficient; fume; fizz; unused.
    pub gr: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuMat>() == 208);

/// One row of the optics table (`struct Optics` in optics.glsl): what a material looks like,
/// in the form the render shaders blend (cell_optics.comp). Coefficients are per cell, at the
/// tank's `cell_size`: the particles' are physical (per metre, times the cell), so that a map
/// of millimetre cells sees milk as a tenth as opaque per cell as one of centimetre cells.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuOptics {
    /// Absorption coefficient (linear RGB: dissolved colour and particles) and the mean
    /// reduced scattering coefficient of the fine particles, 1/cell (`coefficients`).
    absorb: [f32; 4],
    /// Colour when frozen, linear RGB; tint depth, cells.
    solid: [f32; 4],
    /// Opacity, platelet sheen (how much of what it scatters comes from platelets the flow
    /// lines up), metal, gloss.
    opt: [f32; 4],
    /// Fullness of its shapes (from viscosity and contact angle, `fullness`), foam, time
    /// its coat takes to dry (s), emissivity.
    sty: [f32; 4],
    /// Diluent row (-1 none), phase group, how much more a dried film of it scatters (x 1000,
    /// `dry_scatter`).
    idx: [i32; 4],
    /// Volatile share (`volatile_share`), the coat it leaves (FILM_UNITs, `coat_thickness`),
    /// thin-film colours (`iridescence`), crust (`crust`).
    fx: [f32; 4],
    /// Froth (s); fizz; specks drawn per cell of it (`DUST_DRAW_MAX` at most); drainage time
    /// of its foam's films (s).
    fx2: [f32; 4],
    /// Its own light (linear RGB x `luminous`), and the share of it that needs stirring.
    lum: [f32; 4],
    /// Absorption and scattering coefficients at `thermo_hi` (as `absorb`, from `color_hot`).
    hot: [f32; 4],
    /// Thermochromic range lo, hi (°C; hi <= lo: none); how well it wets (`wetting`); 1 if it
    /// is a soapy aqueous liquid that washes any film.
    thermo: [f32; 4],
    /// The spectral shape of the scattering: the reduced scattering coefficient in red,
    /// green, blue over its mean (they sum to 3), and the asymmetry g of the fine particles.
    scat: [f32; 4],
    /// Refractive index; rotational diffusion of its platelets, rad²/s; their Bretherton
    /// constant; evaporation of a film of it, FILM_UNITs/s (`evaporation`).
    opt2: [f32; 4],
    /// What one drawn speck stands for in the liquid around it, 1/cell: absorption (RGB) and
    /// reduced scattering of the specks too many to draw (spray.comp adds it where it lies).
    dust: [f32; 4],
    /// Speck radius (m), density (kg/m³), thickness over diameter, mirror reflectance.
    dust2: [f32; 4],
    /// Speck colour (linear RGB); drawn specks per cell where they lie packed.
    dcol: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuOptics>() == 240);

/// sRGB (gamma) to linear.
pub fn linear(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) })
}

/// Linear to sRGB (gamma).
pub fn gamma(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
    })
}

/// Cells of liquid through which a clear one shows its `color` (`MatProps::opacity`).
pub fn tint_depth(m: &MatProps) -> f32 {
    let clear = 1.0 - m.opacity.clamp(0.0, 1.0);
    0.5 + 30.0 * clear * clear
}

/// Effective wavelengths of the red, green and blue primaries in vacuum, m.
pub const WAVELENGTHS: [f32; 3] = [610.0e-9, 550.0e-9, 465.0e-9];
/// Thickness of a body of liquid along the view, cells (composite.frag: LV_BODY).
pub const LV_BODY: f32 = 16.0;
/// Most specks of one liquid drawn per cell (spray.comp's pool holds `DUST_MAX` of them); a
/// denser dust is drawn as this many, and the rest is part of the liquid's colour.
pub const DUST_DRAW_MAX: f32 = 4.0;
/// Volume fraction of specks lying packed on the floor (random close packing of grains).
pub const DUST_PACKING: f32 = 0.6;
/// Drainage time of the films of water's foam, s; it scales with the viscosity (Reynolds
/// thinning of a film between two bubbles, h(t) = h0 / sqrt(1 + t / tau), tau ~ mu).
pub const FOAM_DRAIN_WATER: f32 = 0.02;
/// Boltzmann's constant times room temperature, J.
const KT: f32 = 1.380_649e-23 * 293.15;

/// Shape of a particle of diameter `d` (m) and thickness over diameter `r`: its mean
/// projected area over its volume (Cauchy: a quarter of its surface, 1/m), its mean chord
/// through it scaled so that a sphere's is its diameter, and the diameter of the sphere of
/// its volume. Above r = 2/3 it is taken as a sphere.
fn particle_shape(d: f32, r: f32) -> (f32, f32, f32) {
    let d = d.max(1e-9);
    if r >= 2.0 / 3.0 {
        return (1.5 / d, d, d);
    }
    let t = d * r.max(0.005);
    let area_over_vol = (0.5 * d + t) / (d * t);
    let chord = 1.5 * d * t / (0.5 * d + t);
    (area_over_vol, chord, d * (1.5 * r.max(0.005)).cbrt())
}

/// Scattering efficiency of a particle of relative refractive index `m`: Rayleigh's for
/// size parameter `x_vol` (of its volume's sphere) where that is the smaller, the anomalous
/// diffraction of van de Hulst (phase shift 2 x (m - 1) along a chord of size parameter
/// `x_path`) where the particle is large. Rayleigh's x⁴ makes fine particles scatter blue.
pub fn scattering_efficiency(x_vol: f32, x_path: f32, m: f32) -> f32 {
    let m = m.max(1e-3);
    let k = (m * m - 1.0) / (m * m + 2.0);
    let qr = 8.0 / 3.0 * x_vol.powi(4) * k * k;
    let rho = 2.0 * x_path * (m - 1.0).abs();
    let qa = if rho < 1e-2 { 0.5 * rho * rho } else { 2.0 - 4.0 / rho * rho.sin() + 4.0 / (rho * rho) * (1.0 - rho.cos()) };
    if qr <= 0.0 || qa <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 / qr + 1.0 / qa)
}

/// Asymmetry of the light a particle scatters (mean cosine): none for a small one, mostly
/// forwards for a large one, the more so the nearer its index is to the liquid's (red cells
/// in plasma 0.98, milk fat 0.95, titanium white ~0.5).
pub fn asymmetry(x: f32, m: f32) -> f32 {
    let large = (1.0 - 0.6 * (m - 1.0).abs()).clamp(0.3, 0.98);
    large * x * x / (x * x + 2.5)
}

/// Single-scattering albedo from the colour of a deep layer of a medium (two-stream /
/// Hapke: R = (1 - q) / (1 + q), q = sqrt(1 - albedo)), per channel, linear.
fn albedo_of(color: [f32; 3]) -> [f32; 3] {
    linear(color).map(|r| {
        let r = r.clamp(1e-3, 0.995);
        let q = (1.0 - r) / (1.0 + r);
        (1.0 - q * q).max(1e-4)
    })
}

/// The fine particles' reduced scattering per channel (1/cell, `cell` m) and their
/// asymmetry, in a liquid of refractive index `n_med`.
pub fn cloud_scattering(m: &MatProps, cell: f32) -> ([f32; 3], f32) {
    if m.cloud <= 0.0 {
        return ([0.0; 3], 0.0);
    }
    let (a_v, chord, d_eq) = particle_shape(m.cloud_size, m.cloud_shape);
    let n_med = m.ior.max(1.0);
    let rel = m.cloud_ior / n_med;
    let mut s = [0.0; 3];
    let mut g_sum = 0.0;
    for c in 0..3 {
        let k = std::f32::consts::PI * n_med / WAVELENGTHS[c];
        let q = scattering_efficiency(k * d_eq, k * chord, rel);
        let g = asymmetry(k * d_eq, rel);
        g_sum += g;
        s[c] = m.cloud.clamp(0.0, 0.74) * a_v * q * (1.0 - g) * cell;
    }
    (s, g_sum / 3.0)
}

/// Absorption (linear RGB, 1/cell) and the scattering coefficients per channel of a
/// material with dissolved colour `color`: what is dissolved absorbs so that `tint_depth`
/// cells of it pass `color`; the fine particles scatter (`cloud_scattering`) and absorb what
/// makes a deep layer of them `cloud_color` (albedo w per channel: a = s (1 - w) / w).
/// A metal is drawn by the colour of what it reflects: its coefficients blend, by `metal`,
/// to those of a deep layer of colour `color` (six scatterings per cell).
pub fn coefficients(m: &MatProps, color: [f32; 3], cell: f32) -> ([f32; 3], [f32; 3]) {
    let depth = tint_depth(m);
    let lin = linear(color).map(|v| v.clamp(1e-3, 0.995));
    let (s, _) = cloud_scattering(m, cell);
    let w = albedo_of(m.cloud_color);
    let mut a = [0.0; 3];
    let mut sc = s;
    for c in 0..3 {
        a[c] = -lin[c].ln() / depth + s[c] * (1.0 - w[c]) / w[c];
    }
    if m.metal > 0.0 {
        let wm = albedo_of(color);
        let k = m.metal.clamp(0.0, 1.0);
        for c in 0..3 {
            let sm = 6.0;
            a[c] += k * (sm * (1.0 - wm[c]) / wm[c] - a[c]);
            sc[c] += k * (sm - sc[c]);
        }
    }
    // (Beyond this a cell is opaque many times over; the cell optics are half floats.)
    (a.map(|v| v.min(MAX_COEFFICIENT)), sc.map(|v| v.min(MAX_COEFFICIENT)))
}

/// Largest absorption or scattering coefficient kept, 1/cell.
pub const MAX_COEFFICIENT: f32 = 2000.0;

/// `coefficients` packed as the shaders take them: absorption (RGB) and the mean scattering.
fn packed(a: [f32; 3], s: [f32; 3]) -> [f32; 4] {
    [a[0], a[1], a[2], (s[0] + s[1] + s[2]) / 3.0]
}

/// What the liquid looks like as a few cells of it lit from the front over a pale ground,
/// sRGB: its swatch colour in the interface (optics.glsl: film_colour).
pub fn appearance(m: &MatProps, cell: f32) -> [f32; 3] {
    let (a, s) = coefficients(m, m.color, cell);
    let film = 1.5;
    let mut out = [0.0; 3];
    for c in 0..3 {
        let w = s[c] / (a[c] + s[c]).max(1e-6);
        let q = (1.0 - w).max(0.0).sqrt();
        let r_inf = (1.0 - q) / (1.0 + q);
        let tv = (-2.0 * (a[c] + s[c]) * film).exp();
        out[c] = r_inf * (1.0 - tv) + 0.8 * tv * (-2.0 * a[c] * film).exp();
    }
    gamma(out)
}

/// How full and rounded a liquid's shapes are drawn (0 .. 1): a viscous liquid holds fat,
/// rounded forms (honey), one that does not wet beads up (mercury); water neither.
pub fn fullness(m: &MatProps) -> f32 {
    let ss = |x: f32, lo: f32, hi: f32| {
        let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    ss(m.viscosity.max(1e-9).log10(), -5.3, -2.7).max(ss(m.contact_angle, 100.0, 145.0))
}

/// Unit of film thickness, m (film.comp: 0.1 mm).
pub const FILM_UNIT: f32 = 1.0e-4;
/// Evaporation rate of water at 20 °C from a thin film or a small drop in room air, m/s
/// (diffusion-limited: D c_sat / L over features millimetres to centimetres wide, 1e-7 - 1e-6;
/// a still pond, a few millimetres a day, is ten times slower).
pub const EVAP_WATER: f32 = 1.0e-7;
/// Speed of the liquid leaving a surface for the coat it leaves behind, m/s.
pub const COAT_SPEED: f32 = 0.05;

/// Share of the liquid that evaporates when a film of it dries: all but its dissolved solids
/// and particles; nothing of a liquid that never boils (oil, the simple liquids).
pub fn volatile_share(m: &MatProps) -> f32 {
    if m.boil >= 1.0e8 && m.volatility <= 0.0 {
        return 0.0;
    }
    (1.0 - m.solids - m.cloud - m.dust).clamp(0.0, 1.0)
}

/// Saturation vapour pressure at `t` (°C) over that at the normal boiling point
/// (Clausius-Clapeyron with Trouton's rule, L = 88 J/(mol K) x T_b: ln p = 10.6 (1 - T_b / T)).
pub fn vapour_pressure(m: &MatProps, t: f32) -> f32 {
    if m.boil >= 1.0e8 {
        return 0.0;
    }
    let tb = m.boil + 273.15;
    let tk = (t + 273.15).max(1.0);
    (10.6 * (1.0 - tb / tk)).clamp(-60.0, 5.0).exp()
}

/// How fast a film of it thins by evaporation at `t` (°C), FILM_UNITs per second, at
/// `speed` times real time: water's rate scaled by its vapour pressure (alcohol dries faster,
/// liquid nitrogen at once, oil never).
pub fn evaporation(m: &MatProps, t: f32, speed: f32) -> f32 {
    let water = MatProps { boil: 100.0, ..Default::default() };
    let rel = vapour_pressure(m, t) / vapour_pressure(&water, crate::sim::material::T_REF).max(1e-12);
    let vol = if m.boil >= 1.0e8 && m.volatility > 0.0 { m.volatility * 50.0 } else { rel };
    EVAP_WATER * vol * speed / FILM_UNIT
}

/// Whether the liquid freezes on a surface at room temperature at once (a melt: solidus
/// above 300 °C): it skins over against it instead of leaving a film that runs.
pub fn freezes_on_contact(m: &MatProps) -> bool {
    m.freezes() && m.melt_lo > 300.0
}

/// Share of a surface a liquid wets: (1 + cos theta) / 2 of its contact angle (1 spreads,
/// 0 beads and rolls off: mercury leaves no film). A melt leaves none.
pub fn wetting(m: &MatProps) -> f32 {
    if freezes_on_contact(m) {
        return 0.0;
    }
    let w = 0.5 * (1.0 + m.contact_angle.to_radians().cos());
    ((w - 0.15) / 0.85).clamp(0.0, 1.0)
}

/// Thickness of the coat a liquid leaves on a surface it drains from, FILM_UNITs
/// (Landau-Levich-Derjaguin: h = 0.94 l_c Ca^(2/3), Ca = mu U / sigma, at most the capillary
/// length l_c = sqrt(sigma / (rho g)); water ~20 µm, blood ~0.3 mm, honey a few mm), times
/// how well it wets.
pub fn coat_thickness(m: &MatProps, gravity: f32) -> f32 {
    if freezes_on_contact(m) {
        return 0.0;
    }
    let sigma = m.tension.max(1e-3);
    let rho = m.density.max(1.0);
    let lc = (sigma / (rho * gravity.max(0.1))).sqrt();
    let ca = (m.viscosity * rho * COAT_SPEED / sigma).min(1.0);
    let wet = 0.5 * (1.0 + m.contact_angle.to_radians().cos());
    0.94 * lc * ca.powf(2.0 / 3.0) * wet / FILM_UNIT
}

/// How much more (or less) a dried deposit of the liquid scatters than its wet film. Rigid
/// grains (dust; fine particles of index 1.5 and above: minerals, pigment, crystals) are left
/// in air: their index step grows from n_p - n to n_p - 1, and they scatter
/// ((n_p - 1) / (n_p - n))² as much (at most 6x): dried mud and salt lighten. Soft particles
/// (cells, fat globules, below 1.5) coalesce with what is dissolved into a continuous glassy
/// layer that scatters a third as much: dried blood is a darker, deeper red than wet.
pub fn dry_scatter(m: &MatProps) -> f32 {
    let rigid_cloud = m.cloud_ior >= 1.5;
    let n = m.ior.max(1.0);
    let gain = |np: f32| (((np - 1.0) / (np - n).abs().max(0.02)).powi(2)).clamp(1.0, 6.0);
    let wr = m.dust + if rigid_cloud { m.cloud } else { 0.0 };
    let ws = if rigid_cloud { 0.0 } else { m.cloud };
    if wr + ws <= 0.0 {
        return 1.0;
    }
    (wr * gain(if m.dust > 0.0 && !rigid_cloud { 1.55 } else { m.cloud_ior.max(1.55) }) + ws * 0.33) / (wr + ws)
}

/// Thin-film colours on a liquid that spreads over water as a thin layer (spreading
/// coefficient S = sigma_water - sigma - sigma_against_water > 0, oil ~0.015 N/m) and bends
/// light differently from it: how strongly they show (0 .. 1). Only on clear liquids.
pub fn iridescence(m: &MatProps, pair: f32, cell: f32) -> f32 {
    if m.group == PhaseGroup::Aqueous as u32 || m.group == PhaseGroup::Metal as u32 {
        return 0.0;
    }
    let spread = 0.072 - m.tension - pair;
    let (a, s) = coefficients(m, m.color, cell);
    let murk = (a[0] + a[1] + a[2]) / 3.0 + (s[0] + s[1] + s[2]) / 3.0;
    (spread / 0.01).clamp(0.0, 1.0) * ((m.ior - 1.33).abs() / 0.1).clamp(0.0, 1.0) * (1.0 - (murk / 1.5).clamp(0.0, 1.0))
}

/// A crust on a glowing melt: any liquid that freezes from incandescence (solidus above
/// ~500 °C) skins over where its surface cools below its solidus (cell_optics.comp works out
/// the skin's temperature).
pub fn crust(m: &MatProps) -> f32 {
    if !m.freezes() {
        return 0.0;
    }
    ((m.melt_lo - 400.0) / 400.0).clamp(0.0, 1.0)
}

/// The specks of a material: how many are drawn per cell of it, and what each drawn one
/// stands for in the liquid's colour (absorption RGB, reduced scattering; 1/cell).
/// A column `LV_BODY` cells deep shows `(1 - exp(-tau)) / A` specks per cell of the view (A
/// their mean projected area in cells², tau the column's optical depth): all of them when
/// they are few, a face of them where they hide each other. Those not drawn are dust in the
/// liquid's colour, laid down where the drawn ones are.
pub fn dust_specks(m: &MatProps, cell: f32) -> (f32, [f32; 4]) {
    if m.dust <= 0.0 {
        return (0.0, [0.0; 4]);
    }
    let (a_v, _, _) = particle_shape(m.dust_size, m.dust_shape);
    let phi = m.dust.clamp(0.0, 0.6);
    // Geometric optics: an efficiency of 2 (half of it diffraction, sent forwards).
    let sigma = phi * a_v * 2.0 * cell;
    let s_red = 0.5 * sigma;
    let area = m.dust_size * m.dust_size * mean_area_factor(m.dust_shape) / (cell * cell);
    let tau = sigma * LV_BODY;
    let n_vis = (1.0 - (-tau).exp()) / area.max(1e-9);
    let n = n_vis.min(DUST_DRAW_MAX);
    if n <= 1e-6 {
        return (0.0, [0.0; 4]);
    }
    let undrawn = (1.0 - n / n_vis).max(0.0);
    let w = albedo_of(m.dust_color);
    let r0 = m.dust_reflect.clamp(0.0, 1.0);
    let per = undrawn / n;
    let a = w.map(|w| {
        let w = r0 + (1.0 - r0) * w;
        s_red * (1.0 - w) / w.max(1e-4) * per
    });
    (n, [a[0], a[1], a[2], s_red * per])
}

/// Mean projected area of a particle over its diameter squared (a quarter of its surface).
fn mean_area_factor(r: f32) -> f32 {
    let pi = std::f32::consts::PI;
    if r >= 2.0 / 3.0 { 0.25 * pi } else { 0.25 * (0.5 * pi + pi * r.max(0.005)) }
}

/// The optics table for the live material table.
pub fn optics_table(p: &super::SimParams) -> Vec<GpuOptics> {
    let cell = p.cell_size;
    (0..MAX_MATERIALS)
        .map(|r| {
            let m: &MatProps = &p.mat[r];
            let sl = linear(m.color_solid);
            let (a, s) = coefficients(m, m.color, cell);
            let (ah, sh) = coefficients(m, m.color_hot, cell);
            let mean = (s[0] + s[1] + s[2]) / 3.0;
            let (_, g) = cloud_scattering(m, cell);
            let ratio = if mean > 1e-9 { s.map(|v| v / mean) } else { [1.0; 3] };
            // Platelets: the share of its look that is light they send back, and how they turn.
            let plate_k = (1.0 - (m.cloud_shape / (2.0 / 3.0)).min(1.0)).max(0.0);
            let am = (a[0] + a[1] + a[2]) / 3.0;
            let plate = if m.cloud > 0.0 { plate_k * (1.0 - (-4.0 * mean).exp()) * mean / (mean + am + 1e-6) } else { 0.0 };
            let eta = (m.viscosity * m.density).max(1e-6);
            let dsz = m.cloud_size.max(1e-9);
            let dr = if plate_k > 0.0 { 3.0 * KT / (4.0 * eta * dsz * dsz * dsz) } else { 0.0 };
            let ar = m.cloud_shape.clamp(0.005, 1.0);
            let bretherton = (ar * ar - 1.0) / (ar * ar + 1.0);
            let (n_draw, per) = dust_specks(m, cell);
            let dl = linear(m.dust_color);
            let pack = if m.dust > 0.0 { n_draw * DUST_PACKING / m.dust.max(1e-6) } else { 0.0 };
            let drain = FOAM_DRAIN_WATER * eta / 1.0e-3;
            // Films (film.comp): what evaporates, how fast, and the coat it leaves.
            let vshare = volatile_share(m);
            let evap = evaporation(m, p.ambient_temp, p.drying);
            let coat = coat_thickness(m, p.gravity);
            let coat_dry_time = if vshare > 0.0 && evap > 0.0 { (coat.max(0.2) * vshare / evap).clamp(0.2, 600.0) } else { 600.0 };
            GpuOptics {
                absorb: packed(a, s),
                solid: [sl[0], sl[1], sl[2], tint_depth(m)],
                opt: [m.opacity, plate, m.metal, m.gloss],
                sty: [fullness(m), m.foam, coat_dry_time, m.emissivity],
                idx: [m.diluent, m.group as i32, (dry_scatter(m) * 1000.0).round() as i32, 0],
                fx: [vshare, coat, iridescence(m, p.pair_tension[0][(m.group as usize).min(GROUPS - 1)], cell), crust(m)],
                fx2: [m.froth, m.fizz, n_draw, drain],
                lum: {
                    let g = linear(m.glow_color);
                    [g[0] * m.luminous, g[1] * m.luminous, g[2] * m.luminous, m.glow_shear]
                },
                hot: packed(ah, sh),
                thermo: [m.thermo_lo, m.thermo_hi, wetting(m), if m.group == 0 && m.tension < 0.04 { 1.0 } else { 0.0 }],
                scat: [ratio[0], ratio[1], ratio[2], g],
                opt2: [m.ior, dr, bretherton, evap],
                dust: per,
                dust2: [0.5 * m.dust_size, m.dust_density, m.dust_shape, m.dust_reflect],
                dcol: [dl[0], dl[1], dl[2], pack],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fine particles scatter blue more than red (Rayleigh); coarse ones about evenly.
    #[test]
    fn fine_particles_scatter_blue() {
        let fine = MatProps { cloud: 0.01, cloud_size: 0.05e-6, cloud_ior: 1.6, ..Default::default() };
        let (s, g) = cloud_scattering(&fine, 0.01);
        assert!(s[2] > 2.5 * s[0], "{s:?}");
        assert!(g < 0.1);
        let coarse = MatProps { cloud: 0.01, cloud_size: 20e-6, cloud_ior: 1.6, ..Default::default() };
        let (s, g) = cloud_scattering(&coarse, 0.01);
        assert!(s[2] < 1.3 * s[0], "{s:?}");
        assert!(g > 0.8);
        // Index matched: they vanish.
        let matched = MatProps { cloud: 0.1, cloud_size: 2e-6, cloud_ior: 1.33, ..Default::default() };
        assert!(cloud_scattering(&matched, 0.01).0[1] < 1e-6);
    }

    /// Milk's reduced scattering is about 2.5 per millimetre (fat globules, 3.5 %, 3 µm).
    #[test]
    fn milk_scatters_as_milk() {
        let milk = MatProps { cloud: 0.035, cloud_size: 3e-6, cloud_ior: 1.46, ior: 1.35, ..Default::default() };
        let (s, _) = cloud_scattering(&milk, 1e-3);
        assert!(s[1] > 1.0 && s[1] < 6.0, "{s:?}");
    }

    /// A deep layer of a cloudy liquid has the colour of its particles.
    #[test]
    fn deep_layer_takes_the_particles_colour() {
        let m = MatProps { cloud: 0.2, cloud_size: 2e-6, cloud_ior: 1.56, cloud_color: [0.6, 0.4, 0.2], color: [1.0; 3], ..Default::default() };
        let (a, s) = coefficients(&m, m.color, 0.01);
        let want = linear(m.cloud_color);
        for c in 0..3 {
            let w = s[c] / (a[c] + s[c]);
            let q = (1.0 - w).sqrt();
            let r = (1.0 - q) / (1.0 + q);
            assert!((r - want[c]).abs() < 0.03, "{c}: {r} vs {}", want[c]);
        }
    }

    /// Films: liquids dry in the order of their boiling points, a liquid that never boils not
    /// at all; viscous liquids leave thicker coats; only the volatile part evaporates.
    #[test]
    fn films_follow_the_liquids() {
        let b = |k: &str| (crate::sim::BUILTINS.iter().find(|x| x.key == k).unwrap().props)();
        let e = |m: &MatProps| evaporation(m, T_REF, 1.0);
        assert!(e(&b("nitrogen")) > e(&b("alcohol")) && e(&b("alcohol")) > e(&b("water")) && e(&b("water")) > 0.0);
        assert_eq!(e(&b("oil")), 0.0);
        let g = 9.81;
        assert!(coat_thickness(&b("honey"), g) > coat_thickness(&b("blood"), g));
        assert!(coat_thickness(&b("blood"), g) > 3.0 * coat_thickness(&b("water"), g));
        assert!(coat_thickness(&b("mercury"), g) < coat_thickness(&b("water"), g));
        assert_eq!(volatile_share(&b("water")), 1.0);
        assert!(volatile_share(&b("blood")) < 0.6 && volatile_share(&b("blood")) > 0.3);
        assert!(dry_scatter(&b("mud")) > 1.5 && dry_scatter(&b("blood")) < 1.0);
    }

    /// Specks: coarse and few are all drawn (nothing left for the colour); fine and many are
    /// drawn up to the cap and the rest colours the liquid.
    #[test]
    fn specks_split_between_drawn_and_colour() {
        let few = MatProps { dust: 1e-5, dust_size: 2e-3, ..Default::default() };
        let (n, per) = dust_specks(&few, 0.01);
        assert!(n > 0.0 && n < DUST_DRAW_MAX);
        assert!(per[3] < 1e-6);
        let many = MatProps { dust: 0.05, dust_size: 5e-5, ..Default::default() };
        let (n, per) = dust_specks(&many, 0.01);
        assert_eq!(n, DUST_DRAW_MAX);
        assert!(per[3] > 0.1);
    }
}
