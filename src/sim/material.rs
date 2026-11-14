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
pub const MAX_MATERIALS: usize = 64;
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
    pub const Water: Material = Material(0);
    pub const Syrup: Material = Material(1);
    pub const Oil: Material = Material(2);
    pub const Mud: Material = Material(3);
    pub const Ink: Material = Material(4);
    pub const Chocolate: Material = Material(5);
    pub const Sludge: Material = Material(6);
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
    /// Dry salt: what evaporating brine leaves behind. Not a liquid the picker offers.
    pub const Salt: Material = Material(18);

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
    /// Body colour, gamma space.
    pub color: [f32; 3],
    /// Colour when frozen.
    pub color_solid: [f32; 3],
    /// Colour at half concentration against the diluent, if the blend should not be a
    /// straight one (syrup in water passes through violet and rose, like grenadine); any
    /// negative component: none.
    pub color_mid: [f32; 3],
    /// 0 clear (shallows show, depth darkens) .. 1 opaque paint.
    pub opacity: f32,
    /// 0 none .. 1 milky: multiple scattering lifts the body towards its own colour.
    pub scatter: f32,
    /// 0 dielectric .. 1 mirror metal.
    pub metal: f32,
    /// Strength of the specular streak.
    pub gloss: f32,
    /// Width of the translucent rim / fullness of the surface (thick liquids).
    pub body: f32,
    /// 1: fast, thin liquid turns to white water.
    pub foam: f32,
    /// Seconds a film of it stays on a solid.
    pub dry_time: f32,
    /// 0 none .. 1 blood: how fast it leaves a lasting stain on solids and the back wall,
    /// which only other liquids wash off.
    pub stain: f32,
    /// 0 none .. 1 honey: fine glints and a moving sheen on its surface, carried by the flow.
    pub glint: f32,
    /// 0 none .. 1 oil: thin-film colours on its surface and thin layers.
    pub iridescence: f32,
    /// 0 none .. 1 lava: as it cools below a bright glow, a dark crust in plates with
    /// glowing seams, carried by the flow.
    pub crackle: f32,
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
            color_mid: [-1.0; 3],
            opacity: 0.0,
            scatter: 0.0,
            metal: 0.0,
            gloss: 0.3,
            body: 0.0,
            foam: 0.0,
            dry_time: 2.0,
            stain: 0.0,
            glint: 0.25,
            iridescence: 0.0,
            crackle: 0.0,
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

    pub fn phase_group(&self) -> PhaseGroup {
        PhaseGroup::from_index(self.group)
    }
}

/// Key, display name and sources of a built-in.
pub struct Builtin {
    /// Name in `--set` keys (`density_<key>`, `<key>.density`) and the stable id `builtin.<key>`.
    pub key: &'static str,
    pub name: &'static str,
    /// Where it appears in the picker.
    pub category: &'static str,
    pub props: fn() -> MatProps,
}

fn water() -> MatProps {
    MatProps {
        // Standard 20 °C values; viscosity follows the Vogel form of the IAPWS correlation
        // (nu = 1.0e-6 at 20 °C, 0.29e-6 at 100 °C).
        temp_model: TEMP_VFT,
        temp_b: 578.9,
        temp_c: 137.5,
        melt_lo: -0.5,
        melt_hi: 0.0,
        latent: 334.0e3,
        density_solid: 917.0,
        boil: 100.0,
        latent_vap: 2257.0e3,
        emissivity: 0.96,
        foam: 1.0,
        gloss: 0.35,
        color: [0.17, 0.52, 0.94],
        color_solid: [0.78, 0.90, 0.98],
        dry_time: 2.0,
        glint: 0.3,
        ..Default::default()
    }
}

fn syrup() -> MatProps {
    MatProps {
        diluent: Material::Water.0 as i32,
        density: 1.4 * 998.0,
        viscosity: 0.06,
        // Strongly temperature dependent, as honey and golden syrup are: an Arrhenius law
        // with B = 10500 K halves the viscosity every 6 K near room temperature (a sandbox
        // choice inside the measured range for honey, 9000 - 12000 K).
        temp_model: TEMP_ARRHENIUS,
        temp_b: 10500.0,
        tension: 0.076,
        mix_rate: 1.5,
        heat_capacity: 2300.0,
        conductivity: 0.35,
        expansion: 5.0e-4,
        melt_lo: -1.0e9,
        melt_hi: -1.0e9,
        boil: 110.0,
        latent_vap: 2257.0e3,
        color: [0.74, 0.36, 0.03],
        color_mid: [1.0, 0.58, 0.0],
        opacity: 0.7,
        gloss: 0.25,
        body: 1.0,
        dry_time: 30.0,
        glint: 1.0,
        ..Default::default()
    }
}

fn oil() -> MatProps {
    MatProps {
        group: PhaseGroup::Oily as u32,
        density: 0.9 * 998.0,
        viscosity: 6.0e-5,
        // Olive oil: 84 mPa s at 20 °C, 12 mPa s at 70 °C  ->  B = 3900 K.
        temp_model: TEMP_ARRHENIUS,
        temp_b: 3900.0,
        tension: 0.032,
        heat_capacity: 1970.0,
        conductivity: 0.17,
        expansion: 7.0e-4,
        melt_lo: -8.0,
        melt_hi: -4.0,
        latent: 150.0e3,
        density_solid: 925.0,
        color: [0.99, 0.79, 0.20],
        color_solid: [0.93, 0.86, 0.55],
        opacity: 0.35,
        gloss: 1.0,
        body: 0.4,
        dry_time: 12.0,
        glint: 0.5,
        iridescence: 1.0,
        ..Default::default()
    }
}

fn mud() -> MatProps {
    MatProps {
        // A suspension: the concentrate is packed silt (60 % grains by volume, 2000 kg/m³),
        // emitted at concentration 0.5 (half water), so fresh mud is 1500 kg/m³ (natural muds and
        // debris-flow slurries: 1300 - 1700, Coussot 1997). Grains of 50 µm settle out of
        // it at about a millimetre per second (Stokes). Sandbox choices.
        diluent: Material::Water.0 as i32,
        density: 2000.0,
        viscosity: 0.05,
        rheology: RHEO_CARREAU,
        flow_index: 0.6,
        shear_time: 1.0,
        visc_ratio: 0.2,
        grain: 50.0e-6,
        load: 0.5,
        heat_capacity: 1500.0,
        conductivity: 1.5,
        expansion: 3.0e-4,
        latent: 0.0,
        freeze_shift: 0.0,
        boil: 100.0,
        latent_vap: 2257.0e3,
        tension: 0.07,
        color: [0.40, 0.29, 0.18],
        color_solid: [0.50, 0.42, 0.33],
        opacity: 1.0,
        scatter: 0.35,
        gloss: 0.25,
        body: 0.6,
        dry_time: 40.0,
        stain: 0.35,
        glint: 0.1,
        ..Default::default()
    }
}

fn ink() -> MatProps {
    MatProps {
        // Water-based ink: dye and pigment dissolved or finely dispersed, a little denser
        // than water (India ink 1.0 - 1.1 g/cm³), water-like viscosity. Its mixing rate is
        // a sandbox choice, slow enough for plumes to show.
        diluent: Material::Water.0 as i32,
        density: 1050.0,
        viscosity: 2.0e-6,
        mix_rate: 0.25,
        tension: 0.06,
        heat_capacity: 4000.0,
        conductivity: 0.58,
        latent: 330.0e3,
        boil: 101.0,
        latent_vap: 2257.0e3,
        color: [0.05, 0.07, 0.16],
        color_solid: [0.20, 0.22, 0.30],
        opacity: 1.0,
        gloss: 0.4,
        dry_time: 25.0,
        stain: 0.8,
        glint: 0.3,
        ..Default::default()
    }
}

fn chocolate() -> MatProps {
    MatProps {
        // Molten dark chocolate: fat-continuous, so it keeps an interface with water.
        // 1250 kg/m³; Casson plastic viscosity 2 - 10 Pa s and yield stress 5 - 20 Pa at
        // 40 - 45 °C (Afoakwa et al. 2007; ICA method 46), here Bingham: 6 Pa s and 10 Pa at
        // 45 °C. Cocoa butter (form V) melts at 33.8 °C; chocolate sets between 27 and 32 °C
        // with ~45 kJ/kg (about a third of cocoa butter's 150 kJ/kg). c = 1.6 kJ/(kg K),
        // k = 0.2 W/(m K) (Tscheuschner). The temperature law is a sandbox choice
        // (Arrhenius, B = 4000 K: a third thinner per 10 K).
        group: PhaseGroup::Oily as u32,
        density: 1250.0,
        density_solid: 1290.0,
        viscosity: 4.8e-3,
        rheology: RHEO_HERSCHEL_BULKLEY,
        yield_stress: 10.0,
        temp_model: TEMP_ARRHENIUS,
        temp_b: 4000.0,
        temp_ref: 45.0,
        tension: 0.035,
        heat_capacity: 1600.0,
        conductivity: 0.2,
        expansion: 6.0e-4,
        melt_lo: 27.0,
        melt_hi: 32.0,
        latent: 45.0e3,
        emissivity: 0.9,
        spawn_temp: 45.0,
        color: [0.33, 0.17, 0.09],
        color_solid: [0.38, 0.22, 0.13],
        opacity: 1.0,
        gloss: 1.0,
        body: 1.0,
        dry_time: 60.0,
        stain: 0.45,
        glint: 0.7,
        ..Default::default()
    }
}

fn sludge() -> MatProps {
    MatProps {
        // Thick, fine-grained sludge (dewatered sewage sludge, drilling mud, slurry):
        // 1450 kg/m³, Herschel-Bulkley with a 60 Pa yield stress (sludges: 10 - 100 Pa,
        // Eshtiaghi et al. 2013), n = 0.5, consistency 30 Pa s^n. Its grains (20 µm) hardly
        // settle. Diluted, it loses its strength steeply (the yield stress follows the
        // concentration; viscosity blends with the exponent override in `MIX_OVERRIDES`).
        diluent: Material::Water.0 as i32,
        density: 1450.0,
        viscosity: 0.02,
        rheology: RHEO_HERSCHEL_BULKLEY,
        yield_stress: 60.0,
        flow_index: 0.5,
        grain: 20.0e-6,
        heat_capacity: 3000.0,
        conductivity: 0.6,
        expansion: 3.0e-4,
        latent: 250.0e3,
        boil: 100.0,
        latent_vap: 2257.0e3,
        tension: 0.06,
        color: [0.24, 0.23, 0.17],
        color_solid: [0.34, 0.32, 0.26],
        opacity: 1.0,
        scatter: 0.2,
        gloss: 0.35,
        body: 1.0,
        dry_time: 90.0,
        stain: 0.5,
        glint: 0.15,
        ..Default::default()
    }
}

fn blood() -> MatProps {
    MatProps {
        // Whole blood, 1060 kg/m³; Carreau fit of Cho & Kensey (1991): mu0 = 56 mPa s,
        // mu_inf = 3.45 mPa s, lambda = 3.313 s, n = 0.3568. Surface tension 0.058 N/m
        // (Hrncir & Rosina 1997); c = 3.6 kJ/(kg K), k = 0.52 W/(m K). Disperses in water.
        diluent: Material::Water.0 as i32,
        density: 1060.0,
        viscosity: 5.28e-5,
        rheology: RHEO_CARREAU,
        flow_index: 0.3568,
        shear_time: 3.313,
        visc_ratio: 0.0616,
        mix_rate: 0.4,
        tension: 0.058,
        heat_capacity: 3617.0,
        conductivity: 0.52,
        latent: 300.0e3,
        freeze_shift: -0.6,
        boil: 101.0,
        latent_vap: 2257.0e3,
        color: [0.42, 0.02, 0.04],
        color_solid: [0.30, 0.05, 0.06],
        opacity: 0.97,
        gloss: 0.5,
        body: 0.3,
        dry_time: 45.0,
        stain: 1.0,
        glint: 0.55,
        ..Default::default()
    }
}

fn mercury() -> MatProps {
    MatProps {
        // CRC Handbook (25 °C): 13 534 kg/m³, 1.526 mPa s, 0.485 N/m against air;
        // contact angle on glass ~140°; melts at -38.83 °C (11.4 kJ/kg; solid 14 184 kg/m³),
        // boils at 356.7 °C (295 kJ/kg); c = 139.5 J/(kg K), k = 8.3 W/(m K); polished
        // emissivity ~0.1.
        group: PhaseGroup::Metal as u32,
        density: 13534.0,
        density_solid: 14184.0,
        viscosity: 1.13e-7,
        tension: 0.485,
        contact_angle: 140.0,
        heat_capacity: 139.5,
        conductivity: 8.3,
        expansion: 1.8e-4,
        melt_lo: -39.0,
        melt_hi: -38.8,
        latent: 11.4e3,
        emissivity: 0.1,
        boil: 356.7,
        latent_vap: 295.0e3,
        color: [0.80, 0.82, 0.85],
        color_solid: [0.70, 0.72, 0.75],
        opacity: 1.0,
        metal: 1.0,
        gloss: 1.0,
        body: 1.0,
        dry_time: 0.5,
        glint: 0.6,
        ..Default::default()
    }
}

fn lava() -> MatProps {
    MatProps {
        // Basaltic lava: 2700 kg/m³ (solid 2900), ~135 Pa s at 1200 °C rising tenfold by
        // ~1100 °C (Arrhenius, B = 56 000 K; measured basalts: 10² - 10⁴ Pa s at 1100 -
        // 1200 °C, Giordano et al. 2008). It crystallises between 1000 and 1150 °C
        // (400 kJ/kg); below ~1090 °C it is past `rigid_viscosity` and moves as a crust.
        // c = 1.2 kJ/(kg K), k = 1.5 W/(m K), emissivity 0.95 (Harris 2013).
        group: PhaseGroup::Silicate as u32,
        density: 2700.0,
        density_solid: 2900.0,
        viscosity: 0.05,
        temp_model: TEMP_ARRHENIUS,
        temp_b: 56000.0,
        temp_ref: 1200.0,
        tension: 0.35,
        contact_angle: 60.0,
        heat_capacity: 1200.0,
        conductivity: 1.5,
        expansion: 1.0e-4,
        melt_lo: 1000.0,
        melt_hi: 1150.0,
        latent: 400.0e3,
        emissivity: 0.95,
        spawn_temp: 1200.0,
        color: [0.10, 0.07, 0.06],
        color_solid: [0.16, 0.14, 0.13],
        opacity: 1.0,
        gloss: 0.3,
        body: 1.0,
        dry_time: 120.0,
        stain: 0.3,
        glint: 0.0,
        crackle: 1.0,
        ..Default::default()
    }
}

fn glass() -> MatProps {
    MatProps {
        // Soda-lime glass: VFT fit log10(eta / Pa s) = -2.585 + 4519 / (T - 263 °C)
        // (Lakatos et al. 1972 type fit; Shelby, "Introduction to Glass Science"):
        // B = 10 406 K, C = 536 K; 0.27 m²/s at 1100 °C, past `rigid_viscosity` below
        // ~1050 °C, so it sets without a latent heat. Melt 2450 kg/m³, 0.3 N/m;
        // c = 1.3 kJ/(kg K), effective k = 1.2 W/(m K).
        group: PhaseGroup::Silicate as u32,
        density: 2450.0,
        viscosity: 0.267,
        temp_model: TEMP_VFT,
        temp_b: 10406.0,
        temp_c: 536.0,
        temp_ref: 1100.0,
        tension: 0.3,
        contact_angle: 45.0,
        heat_capacity: 1300.0,
        conductivity: 1.2,
        expansion: 5.0e-5,
        emissivity: 0.9,
        spawn_temp: 1200.0,
        color: [0.80, 0.93, 0.88],
        color_solid: [0.80, 0.93, 0.88],
        opacity: 0.08,
        gloss: 1.0,
        body: 1.0,
        dry_time: 120.0,
        glint: 0.8,
        ..Default::default()
    }
}

fn milk() -> MatProps {
    MatProps {
        // Whole milk: 1030 kg/m³, ~2 mPa s at 20 °C (Fox & McSweeney), an emulsion of fat
        // globules (~4 µm) that is opaque white by scattering. It dilutes with water.
        // Creaming takes hours, so its grains are left out (grain 0).
        diluent: Material::Water.0 as i32,
        density: 1030.0,
        viscosity: 1.94e-6,
        temp_model: TEMP_ARRHENIUS,
        temp_b: 2500.0,
        mix_rate: 0.8,
        tension: 0.045,
        heat_capacity: 3930.0,
        conductivity: 0.55,
        latent: 300.0e3,
        freeze_shift: -0.55,
        boil: 100.5,
        latent_vap: 2257.0e3,
        color: [0.96, 0.95, 0.91],
        color_solid: [0.97, 0.97, 0.95],
        opacity: 0.9,
        scatter: 1.0,
        gloss: 0.35,
        body: 0.3,
        dry_time: 20.0,
        glint: 0.15,
        ..Default::default()
    }
}

fn slime() -> MatProps {
    MatProps {
        // PVA-borax slime: 1030 kg/m³, a Maxwell liquid with G ~ 100 Pa and a relaxation
        // time ~1 s (Cassagnau & Michel; Koopmans' measurements of "silly putty"-like
        // gels span 0.1 - 10 s). Its viscosity at short times, ~1 Pa s, is a sandbox choice;
        // it is also what holds it to walls (no-slip comes from the viscosity solve).
        density: 1030.0,
        viscosity: 1.0e-3,
        shear_modulus: 100.0,
        relax_time: 1.0,
        tension: 0.06,
        heat_capacity: 3900.0,
        conductivity: 0.55,
        melt_lo: -1.0,
        melt_hi: -0.5,
        latent: 300.0e3,
        boil: 100.0,
        latent_vap: 2257.0e3,
        color: [0.36, 0.86, 0.32],
        color_solid: [0.55, 0.85, 0.55],
        opacity: 0.45,
        gloss: 1.0,
        body: 1.0,
        dry_time: 60.0,
        stain: 0.3,
        glint: 0.85,
        ..Default::default()
    }
}

fn ferrofluid() -> MatProps {
    MatProps {
        // Ferrotec EFH1 (hydrocarbon carrier, oil-like): 1210 kg/m³, 6 mPa s, initial
        // susceptibility 2.64 (SI), saturation 44 mT; surface tension ~0.029 N/m.
        group: PhaseGroup::Oily as u32,
        density: 1210.0,
        viscosity: 5.0e-6,
        tension: 0.029,
        susceptibility: 2.64,
        saturation: 0.044,
        heat_capacity: 1800.0,
        conductivity: 0.15,
        expansion: 7.0e-4,
        melt_lo: -60.0,
        melt_hi: -55.0,
        latent: 100.0e3,
        // Glossy black: what shows is the reflection of its surroundings, so it is drawn
        // as a dark mirror (metal) over a very dark body, not as a black body with a
        // quarter of a mirror, which vanished against the dark tank (2026-11-14).
        color: [0.10, 0.10, 0.115],
        color_solid: [0.12, 0.12, 0.13],
        opacity: 1.0,
        metal: 0.85,
        gloss: 1.0,
        body: 0.7,
        dry_time: 30.0,
        stain: 0.12,
        glint: 0.7,
        iridescence: 0.15,
        ..Default::default()
    }
}

fn brine() -> MatProps {
    MatProps {
        // Saturated NaCl brine (26 % by mass): 1200 kg/m³, 1.9 mPa s (CRC). Salt lowers the
        // freezing point along the liquidus to the eutectic, -21.1 °C at 23.3 % (here
        // linear in concentration, -21 K at saturation). Saturated brine boils at 108.8 °C
        // and leaves 312 kg of salt per m³ (14 % of its volume as crystals of 2165 kg/m³).
        diluent: Material::Water.0 as i32,
        density: 1200.0,
        viscosity: 1.6e-6,
        mix_rate: 1.0,
        tension: 0.082,
        heat_capacity: 3300.0,
        conductivity: 0.57,
        latent: 250.0e3,
        freeze_shift: -21.0,
        boil: 108.8,
        latent_vap: 2257.0e3,
        residue: Material::Salt.0 as i32,
        residue_share: 0.144,
        color: [0.62, 0.80, 0.86],
        color_solid: [0.90, 0.94, 0.96],
        opacity: 0.05,
        gloss: 0.35,
        dry_time: 6.0,
        glint: 0.3,
        ..Default::default()
    }
}

fn wax() -> MatProps {
    MatProps {
        // Paraffin wax: ~780 kg/m³ molten, ~900 solid (both float on water); melts between
        // 52 and 62 °C with ~200 kJ/kg; molten ~3 mPa s at 70 °C; c = 2.9 kJ/(kg K) liquid,
        // k = 0.25 W/(m K) (CRC; Sharma et al. 2009 review of PCMs).
        group: PhaseGroup::Oily as u32,
        density: 780.0,
        density_solid: 900.0,
        viscosity: 3.8e-6,
        temp_model: TEMP_ARRHENIUS,
        temp_b: 2500.0,
        temp_ref: 70.0,
        tension: 0.028,
        heat_capacity: 2900.0,
        conductivity: 0.25,
        expansion: 1.0e-3,
        melt_lo: 52.0,
        melt_hi: 62.0,
        latent: 200.0e3,
        emissivity: 0.9,
        spawn_temp: 75.0,
        color: [0.97, 0.90, 0.72],
        color_solid: [0.95, 0.94, 0.88],
        opacity: 0.3,
        scatter: 0.3,
        gloss: 0.8,
        body: 0.6,
        dry_time: 80.0,
        stain: 0.4,
        glint: 0.4,
        ..Default::default()
    }
}

fn oobleck() -> MatProps {
    MatProps {
        // Cornstarch in water (~55 % starch by mass): 1550 kg/m³. Shear thickening, a
        // sandbox fit: 3 Pa s at rest, (1 + (0.5 g)^2) times that above 2 /s, so it jams
        // (passes `rigid_viscosity`) above ~45 /s: it flows when stirred gently and holds
        // like a solid when hit (Fall et al. 2008; Brown & Jaeger 2014). Starch grains
        // (15 µm) settle slowly out of diluted oobleck.
        diluent: Material::Water.0 as i32,
        density: 1550.0,
        viscosity: 2.0e-3,
        rheology: RHEO_THICKENING,
        flow_index: 3.0,
        shear_time: 0.5,
        visc_ratio: 2000.0,
        grain: 15.0e-6,
        heat_capacity: 2800.0,
        conductivity: 0.5,
        expansion: 3.0e-4,
        latent: 150.0e3,
        boil: 100.0,
        latent_vap: 2257.0e3,
        tension: 0.06,
        color: [0.93, 0.91, 0.85],
        color_solid: [0.95, 0.94, 0.90],
        opacity: 0.95,
        scatter: 0.9,
        gloss: 0.2,
        body: 1.0,
        dry_time: 50.0,
        stain: 0.2,
        glint: 0.1,
        ..Default::default()
    }
}

fn gallium() -> MatProps {
    MatProps {
        // Liquid gallium: 6095 kg/m³ at its melting point, 29.76 °C (80.1 kJ/kg); it expands
        // 3.1 % on freezing (solid 5904 kg/m³). 1.98 mPa s, 0.708 N/m clean; in air an oxide
        // skin with a surface yield stress of ~0.5 N/m holds irregular shapes (Dickey et al.
        // 2008). c = 410 J/(kg K), k = 29 W/(m K) (CRC).
        group: PhaseGroup::Metal as u32,
        density: 6095.0,
        density_solid: 5904.0,
        viscosity: 3.25e-7,
        tension: 0.708,
        contact_angle: 130.0,
        skin: 0.5,
        heat_capacity: 410.0,
        conductivity: 29.0,
        expansion: 1.2e-4,
        melt_lo: 29.5,
        melt_hi: 29.8,
        latent: 80.1e3,
        emissivity: 0.1,
        boil: 2400.0,
        latent_vap: 3700.0e3,
        spawn_temp: 40.0,
        color: [0.72, 0.74, 0.78],
        color_solid: [0.62, 0.64, 0.68],
        opacity: 1.0,
        metal: 1.0,
        gloss: 0.8,
        body: 1.0,
        dry_time: 1.0,
        glint: 0.6,
        ..Default::default()
    }
}

fn salt() -> MatProps {
    MatProps {
        // Halite: 2165 kg/m³, melts at 801 °C; what evaporating brine leaves behind. It is
        // solid at every temperature the sandbox reaches and does not redissolve.
        density: 2165.0,
        viscosity: 1.0e-3,
        heat_capacity: 880.0,
        conductivity: 6.5,
        expansion: 1.0e-4,
        melt_lo: 800.0,
        melt_hi: 801.0,
        latent: 482.0e3,
        tension: 0.07,
        color: [0.94, 0.94, 0.92],
        color_solid: [0.94, 0.94, 0.92],
        opacity: 1.0,
        scatter: 1.0,
        gloss: 0.2,
        dry_time: 120.0,
        glint: 0.0,
        ..Default::default()
    }
}

/// The built-in materials, in table order (`Material::Water` ...).
pub static BUILTINS: [Builtin; 19] = [
    Builtin { key: "water", name: "Water", category: "Everyday", props: water },
    Builtin { key: "syrup", name: "Syrup", category: "Everyday", props: syrup },
    Builtin { key: "oil", name: "Oil", category: "Everyday", props: oil },
    Builtin { key: "mud", name: "Mud", category: "Earth", props: mud },
    Builtin { key: "ink", name: "Ink", category: "Everyday", props: ink },
    Builtin { key: "chocolate", name: "Molten chocolate", category: "Kitchen", props: chocolate },
    Builtin { key: "sludge", name: "Sludge", category: "Earth", props: sludge },
    Builtin { key: "blood", name: "Blood", category: "Body", props: blood },
    Builtin { key: "mercury", name: "Mercury", category: "Metal", props: mercury },
    Builtin { key: "lava", name: "Lava", category: "Hot", props: lava },
    Builtin { key: "glass", name: "Molten glass", category: "Hot", props: glass },
    Builtin { key: "milk", name: "Milk", category: "Kitchen", props: milk },
    Builtin { key: "slime", name: "Slime", category: "Strange", props: slime },
    Builtin { key: "ferrofluid", name: "Ferrofluid", category: "Strange", props: ferrofluid },
    Builtin { key: "brine", name: "Brine", category: "Everyday", props: brine },
    Builtin { key: "wax", name: "Molten wax", category: "Hot", props: wax },
    Builtin { key: "oobleck", name: "Oobleck", category: "Strange", props: oobleck },
    Builtin { key: "gallium", name: "Liquid gallium", category: "Metal", props: gallium },
    Builtin { key: "salt", name: "Salt", category: "Residue", props: salt },
];

/// Rows the liquid picker offers (Salt is only ever left behind by evaporating brine).
pub fn pickable(row: usize) -> bool {
    row != Material::Salt.index()
}

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
    // Sludge and oobleck lose their gel / jammed structure steeply when watered down
    // (viscosity of concentrated suspensions rises steeply near packing, Krieger-Dougherty).
    MixOverride { concentrate: Material::Sludge, diluent: Material::Water, visc_exponent: 2.0 },
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
    /// the diluent (`MIX_OVERRIDES`); unused.
    pub misc: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuMat>() == 160);

/// One row of the optics table (`struct Optics` in optics.glsl): what a material looks like,
/// in the form the render shaders blend (cell_optics.comp).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuOptics {
    /// Body colour as OKLab L, a, b and chroma; the same when frozen.
    lab: [f32; 4],
    lab_solid: [f32; 4],
    /// Colour at half concentration (OKLab, chroma), or chroma -1: a straight blend.
    lab_mid: [f32; 4],
    /// Opacity, scattering, metal, gloss.
    opt: [f32; 4],
    /// Body (thickness of the look), foam, drying time (s), emissivity.
    sty: [f32; 4],
    /// Diluent row (-1 none), phase group.
    idx: [i32; 4],
    /// Stain, glint, iridescence, crackle.
    fx: [f32; 4],
}

/// sRGB (gamma) to OKLab (Ottosson 2020), with the chroma as a fourth value.
pub fn oklab(c: [f32; 3]) -> [f32; 4] {
    let lin = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    let (r, g, b) = (lin(c[0]), lin(c[1]), lin(c[2]));
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_99 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    let lab = [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ];
    [lab[0], lab[1], lab[2], lab[1].hypot(lab[2])]
}

/// The optics table for the live material table.
pub fn optics_table(p: &super::SimParams) -> Vec<GpuOptics> {
    (0..MAX_MATERIALS)
        .map(|r| {
            let m: &MatProps = &p.mat[r];
            GpuOptics {
                lab: oklab(m.color),
                lab_solid: oklab(m.color_solid),
                lab_mid: if m.color_mid.iter().any(|&c| c < 0.0) { [0.0, 0.0, 0.0, -1.0] } else { oklab(m.color_mid) },
                opt: [m.opacity, m.scatter, m.metal, m.gloss],
                sty: [m.body, m.foam, m.dry_time, m.emissivity],
                idx: [m.diluent, m.group as i32, 0, 0],
                fx: [m.stain, m.glint, m.iridescence, m.crackle],
            }
        })
        .collect()
}

