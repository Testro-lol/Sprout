//! The built-in liquids (README, "Liquids"): the rows of the material table every tank starts
//! with, in three sets.
//!
//! - **Real liquids** (rows 0 - 26): liquids one meets, with measured properties where there
//!   are any, each with its source. Every number is in SI / display units (`MatProps`).
//!   Refractive indices (`ior`) are at the sodium D line, about 20 °C (CRC Handbook; for
//!   mixtures and solutions the nearest tabulated one: milk and blood as their serum, beer
//!   as 5 % sucrose, the lacquers as their solvent resins), water's 1.33 where not set.
//! - **Simple liquids** (rows 27 - 55): the neutral liquid of `MatProps::default()` (a plain
//!   Newtonian liquid as dense and thin as water that never freezes or boils) with one to five
//!   properties changed, its colour among them, no two the same set of properties. Each shows
//!   what one property, or a small combination, does.
//! - **By-products** (Salt, Steam, Resin): what other liquids leave or turn into. The picker
//!   does not offer them.
//!
//! The order is the table's: `Material::Water` ... in material.rs.

use super::material::*;

/// Key, display name and sources of a built-in.
pub struct Builtin {
    /// Name in `--set` keys (`density_<key>`, `<key>.density`) and the stable id `builtin.<key>`.
    pub key: &'static str,
    pub name: &'static str,
    /// Where it appears in the picker.
    pub category: &'static str,
    pub props: fn() -> MatProps,
}

const WATER: i32 = Material::Water.0 as i32;
const OILY: u32 = PhaseGroup::Oily as u32;
const METAL: u32 = PhaseGroup::Metal as u32;
const SILICATE: u32 = PhaseGroup::Silicate as u32;

// ==== real liquids ============================================================================

fn water() -> MatProps {
    MatProps {
        // Standard 20 °C values; viscosity follows the Vogel form of the IAPWS correlation
        // (1.0e-6 m²/s at 20 °C, 0.29e-6 at 100 °C). Ice 917 kg/m³, 334 kJ/kg; boils at
        // 100 °C with 2257 kJ/kg. White water lasts about half a second.
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
        color: [0.04, 0.38, 0.70],
        color_solid: [0.78, 0.90, 0.98],
        gloss: 0.35,
        foam: 1.0,
        froth: 0.6,
        ..Default::default()
    }
}

fn honey() -> MatProps {
    MatProps {
        // Honey: 1400 kg/m³; about 80 Pa s at 20 °C in the sandbox's units (0.06 m²/s),
        // halving every 6 K (Arrhenius, B = 10 500 K; measured honeys 9000 - 12 000 K).
        // c = 2.3 kJ/(kg K), k = 0.35 W/(m K). It dissolves in water.
        diluent: WATER,
        density: 1.4 * 998.0,
        viscosity: 0.06,
        temp_model: TEMP_ARRHENIUS,
        temp_b: 10500.0,
        tension: 0.076,
        mix_rate: 1.5,
        heat_capacity: 2300.0,
        conductivity: 0.35,
        expansion: 5.0e-4,
        boil: 110.0,
        latent_vap: 2257.0e3,
        color: [0.92, 0.58, 0.10],
        opacity: 0.55,
        gloss: 0.9,
        ior: 1.49,
        // Pollen and wax crystals, a few in ten thousand: a faint haze.
        cloud: 2.0e-4,
        cloud_size: 5.0e-6,
        cloud_ior: 1.55,
        cloud_color: [0.95, 0.85, 0.60],
        solids: 0.8,
        ..Default::default()
    }
}

fn oil() -> MatProps {
    MatProps {
        // Olive oil: 900 kg/m³, 84 mPa s at 20 °C and 12 mPa s at 70 °C (Arrhenius,
        // B = 3900 K); 0.032 N/m; sets between -8 and -4 °C. c = 1.97 kJ/(kg K),
        // k = 0.17 W/(m K). A film of it on water shows thin-film colours.
        group: OILY,
        density: 0.9 * 998.0,
        viscosity: 6.0e-5,
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
        color: [0.95, 0.80, 0.30],
        color_solid: [0.93, 0.86, 0.55],
        opacity: 0.25,
        gloss: 1.0,
        ior: 1.47,
        ..Default::default()
    }
}

fn mud() -> MatProps {
    MatProps {
        // A suspension: packed silt (60 % grains by volume, 2000 kg/m³) emitted half water,
        // so fresh mud is 1500 kg/m³ (natural muds and debris flows: 1300 - 1700, Coussot
        // 1997). Grains of 50 µm settle at about a millimetre a second (Stokes). It thins
        // with shear (Carreau; sandbox numbers).
        diluent: WATER,
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
        boil: 100.0,
        latent_vap: 2257.0e3,
        tension: 0.07,
        color: [0.40, 0.29, 0.18],
        color_solid: [0.50, 0.42, 0.33],
        opacity: 1.0,
        gloss: 0.25,
        // Clay and silt fines (2 µm, quartz-like index) make it opaque in their colour; a little
        // fine sand (0.2 mm) shows as grit that settles out of muddy water.
        cloud: 0.3,
        cloud_size: 2.0e-6,
        cloud_ior: 1.55,
        cloud_color: [0.40, 0.29, 0.18],
        dust: 0.02,
        dust_size: 2.0e-4,
        dust_color: [0.55, 0.45, 0.32],
        dust_reflect: 0.06,
        ..Default::default()
    }
}

fn ink() -> MatProps {
    MatProps {
        // Water-based ink: dye and fine pigment, 1050 kg/m³ (India ink 1.0 - 1.1), about
        // 2 mPa s. Its mixing rate is a sandbox choice, slow enough for plumes to show.
        diluent: WATER,
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
        ior: 1.34,
        // Carbon-black pigment, 0.1 µm: plumes of it in water are smoky.
        cloud: 0.01,
        cloud_size: 1.0e-7,
        cloud_ior: 2.0,
        cloud_color: [0.05, 0.05, 0.06],
        solids: 0.05,
        ..Default::default()
    }
}

fn chocolate() -> MatProps {
    MatProps {
        // Molten dark chocolate: fat-continuous, so it keeps an interface with water.
        // 1250 kg/m³; Casson plastic viscosity 2 - 10 Pa s and yield stress 5 - 20 Pa at
        // 40 - 45 °C (Afoakwa et al. 2007), here Bingham: 6 Pa s and 10 Pa at 45 °C. It sets
        // between 27 and 32 °C with ~45 kJ/kg. c = 1.6 kJ/(kg K), k = 0.2 W/(m K)
        // (Tscheuschner). Arrhenius, B = 4000 K (a sandbox choice).
        group: OILY,
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
        spawn_temp: 45.0,
        color: [0.33, 0.17, 0.09],
        color_solid: [0.38, 0.22, 0.13],
        opacity: 1.0,
        gloss: 1.0,
        // Tempered chocolate folds lighter streaks of cocoa butter through itself.
        ior: 1.45,
        // Cocoa solids and fat droplets, ~8 µm, a third of its volume.
        cloud: 0.3,
        cloud_size: 8.0e-6,
        cloud_ior: 1.55,
        cloud_color: [0.33, 0.17, 0.09],
        ..Default::default()
    }
}

fn clay() -> MatProps {
    MatProps {
        // Clay slip (potter's slip; drilling muds and sludges flow alike): 1450 kg/m³,
        // Herschel-Bulkley with a 60 Pa yield stress (10 - 100 Pa, Eshtiaghi et al. 2013),
        // n = 0.5, consistency 30 Pa s^n. Its 20 µm grains hardly settle. Watered down it
        // loses its strength steeply (`MIX_OVERRIDES`).
        diluent: WATER,
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
        color: [0.55, 0.44, 0.36],
        color_solid: [0.62, 0.52, 0.45],
        opacity: 1.0,
        gloss: 0.35,
        // Its clay platelets, 2 µm, packed.
        cloud: 0.35,
        cloud_size: 2.0e-6,
        cloud_ior: 1.56,
        cloud_color: [0.55, 0.44, 0.36],
        ..Default::default()
    }
}

fn blood() -> MatProps {
    MatProps {
        // Whole blood, 1060 kg/m³; Carreau fit of Cho & Kensey (1991): mu0 = 56 mPa s,
        // mu_inf = 3.45 mPa s, lambda = 3.313 s, n = 0.3568. 0.058 N/m (Hrncir & Rosina
        // 1997); c = 3.6 kJ/(kg K), k = 0.52 W/(m K). Disperses in water.
        diluent: WATER,
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
        color: [0.95, 0.85, 0.45],
        color_solid: [0.30, 0.05, 0.06],
        opacity: 0.3,
        gloss: 0.8,
        foam: 0.3,
        froth: 1.5,
        ior: 1.35,
        // Red cells: 45 % by volume (haematocrit), discs 7.5 µm across and a third as thick, index
        // 1.40 in plasma of 1.35; haemoglobin gives them their colour. The plasma is straw yellow.
        cloud: 0.45,
        cloud_size: 7.5e-6,
        cloud_ior: 1.40,
        cloud_shape: 0.3,
        cloud_color: [0.40, 0.02, 0.035],
        solids: 0.08,
        ..Default::default()
    }
}

fn mercury() -> MatProps {
    MatProps {
        // CRC Handbook (25 °C): 13 534 kg/m³, 1.526 mPa s, 0.485 N/m; contact angle on
        // glass ~140°; melts at -38.83 °C (11.4 kJ/kg; solid 14 184 kg/m³), boils at
        // 356.7 °C (295 kJ/kg); c = 139.5 J/(kg K), k = 8.3 W/(m K); emissivity ~0.1.
        group: METAL,
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
        ..Default::default()
    }
}

fn lava() -> MatProps {
    MatProps {
        // Basaltic lava: 2700 kg/m³ (solid 2900), ~135 Pa s at 1200 °C rising tenfold by
        // ~1100 °C (Arrhenius, B = 56 000 K; Giordano et al. 2008). It crystallises between
        // 1000 and 1150 °C (400 kJ/kg). c = 1.2 kJ/(kg K), k = 1.5 W/(m K), emissivity 0.95
        // (Harris 2013).
        group: SILICATE,
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
        // Crystals and bubbles in the melt make it opaque.
        cloud: 0.3,
        cloud_size: 3.0e-5,
        cloud_ior: 1.70,
        cloud_color: [0.10, 0.07, 0.06],
        ..Default::default()
    }
}

fn glass() -> MatProps {
    MatProps {
        // Soda-lime glass: VFT fit log10(eta / Pa s) = -2.585 + 4519 / (T - 263 °C) (Lakatos
        // et al. 1972; Shelby): B = 10 406 K, C = 536 K; 0.27 m²/s at 1100 °C, too stiff to
        // flow below ~1050 °C, so it sets without a latent heat. 2450 kg/m³, 0.3 N/m;
        // c = 1.3 kJ/(kg K), effective k = 1.2 W/(m K).
        group: SILICATE,
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
        ior: 1.52,
        ..Default::default()
    }
}

fn milk() -> MatProps {
    MatProps {
        // Whole milk: 1030 kg/m³, ~2 mPa s at 20 °C (Fox & McSweeney), an emulsion of fat
        // globules that is white by scattering. It dilutes with water; creaming takes hours
        // (no grains). Its proteins hold a froth for a few seconds.
        diluent: WATER,
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
        color: [0.98, 0.96, 0.88],
        color_solid: [0.97, 0.97, 0.95],
        opacity: 0.3,
        gloss: 0.35,
        foam: 0.7,
        froth: 4.0,
        ior: 1.35,
        // Fat globules, 3.5 % by volume, ~3 µm, index 1.46 (Walstra & Jenness 1984): a reduced
        // scattering of ~2.5 per mm, as measured. Watered down, it turns bluish (Rayleigh).
        cloud: 0.035,
        cloud_size: 3.0e-6,
        cloud_ior: 1.46,
        cloud_color: [0.98, 0.97, 0.94],
        solids: 0.09,
        ..Default::default()
    }
}

fn slime() -> MatProps {
    MatProps {
        // PVA-borax slime: 1030 kg/m³, a Maxwell liquid with G ~ 100 Pa and a relaxation
        // time ~1 s (Cassagnau & Michel); solvent viscosity 10 Pa s, that of the white glue
        // it is made from (thinner, a dropped blob flung itself apart: README,
        // "Viscoelasticity").
        density: 1030.0,
        viscosity: 1.0e-2,
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
        ior: 1.34,
        cloud: 0.004,
        cloud_size: 2.0e-6,
        cloud_ior: 1.45,
        cloud_color: [0.85, 0.95, 0.85],
        solids: 0.05,
        ..Default::default()
    }
}

fn ferrofluid() -> MatProps {
    MatProps {
        // Ferrotec EFH1 (hydrocarbon carrier): 1210 kg/m³, 6 mPa s, initial susceptibility
        // 2.64 (SI), saturation 44 mT; ~0.029 N/m. Drawn as a dark mirror over a very dark
        // body: what shows of a glossy black liquid is the reflection of its surroundings.
        group: OILY,
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
        color: [0.10, 0.10, 0.115],
        color_solid: [0.12, 0.12, 0.13],
        opacity: 1.0,
        metal: 0.85,
        gloss: 1.0,
        // Magnetite nanoparticles, 10 nm, 5 % by volume, which absorb nearly all light.
        cloud: 0.05,
        cloud_size: 1.0e-8,
        cloud_ior: 2.4,
        cloud_color: [0.10, 0.10, 0.115],
        ..Default::default()
    }
}

fn brine() -> MatProps {
    MatProps {
        // Saturated NaCl brine (26 % by mass): 1200 kg/m³, 1.9 mPa s (CRC). Salt lowers the
        // freezing point to the eutectic, -21.1 °C (linear in concentration here). It boils
        // at 108.8 °C and leaves 312 kg of salt per m³ (14 % of its volume as crystals).
        diluent: WATER,
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
        color: [0.14, 0.50, 0.62],
        color_solid: [0.90, 0.94, 0.96],
        opacity: 0.05,
        gloss: 0.35,
        foam: 1.0,
        froth: 1.2,
        ior: 1.38,
        solids: 0.12,
        ..Default::default()
    }
}

fn wax() -> MatProps {
    MatProps {
        // Paraffin wax: ~780 kg/m³ molten, ~900 solid; melts between 52 and 62 °C with
        // ~200 kJ/kg; ~3 mPa s at 70 °C; c = 2.9 kJ/(kg K), k = 0.25 W/(m K) (CRC; Sharma
        // et al. 2009).
        group: OILY,
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
        spawn_temp: 75.0,
        color: [0.97, 0.90, 0.72],
        color_solid: [0.95, 0.94, 0.88],
        opacity: 0.3,
        gloss: 0.8,
        ior: 1.44,
        cloud: 0.003,
        cloud_size: 2.0e-6,
        cloud_ior: 1.50,
        ..Default::default()
    }
}

fn oobleck() -> MatProps {
    MatProps {
        // Cornstarch in water (~55 % starch by mass): 1550 kg/m³. Shear thickening, a
        // sandbox fit: 3 Pa s at rest, (1 + (0.5 g)²) times that above 2 /s, so it jams above
        // ~45 /s (Fall et al. 2008; Brown & Jaeger 2014). Starch grains (15 µm) settle slowly
        // out of diluted oobleck.
        diluent: WATER,
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
        gloss: 0.2,
        // Starch grains, half its volume.
        cloud: 0.5,
        cloud_size: 1.5e-5,
        cloud_ior: 1.53,
        cloud_color: [0.93, 0.91, 0.85],
        ..Default::default()
    }
}

fn gallium() -> MatProps {
    MatProps {
        // Liquid gallium: 6095 kg/m³ at its melting point, 29.76 °C (80.1 kJ/kg); solid
        // 5904 kg/m³. 1.98 mPa s, 0.708 N/m clean; in air an oxide skin with a surface yield
        // stress of ~0.5 N/m holds irregular shapes (Dickey et al. 2008). c = 410 J/(kg K),
        // k = 29 W/(m K) (CRC).
        group: METAL,
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
        ..Default::default()
    }
}

fn beer() -> MatProps {
    MatProps {
        // Lager: 1010 kg/m³, 1.6 mPa s, 0.045 N/m (Bamforth 2004). About 5 g/l of dissolved
        // CO2 comes out as bubbles; its proteins hold the head for half a minute or more
        // (the half-life of a lager's foam: 20 - 60 s by the Rudin and NIBEM methods).
        diluent: WATER,
        density: 1010.0,
        viscosity: 1.6e-6,
        mix_rate: 1.0,
        tension: 0.045,
        heat_capacity: 4050.0,
        conductivity: 0.58,
        latent: 320.0e3,
        freeze_shift: -2.3,
        boil: 99.0,
        latent_vap: 2257.0e3,
        color: [0.93, 0.62, 0.10],
        color_solid: [0.93, 0.82, 0.55],
        opacity: 0.35,
        gloss: 0.4,
        foam: 1.0,
        froth: 40.0,
        fizz: 0.6,
        ior: 1.34,
        solids: 0.04,
        ..Default::default()
    }
}

fn soap() -> MatProps {
    MatProps {
        // Pearly liquid hand soap: 1030 kg/m³, about 3 Pa s at rest, thinning with shear
        // (Carreau; surfactant solutions of wormlike micelles); 0.028 N/m, wets everything.
        // Glycol distearate platelets give it its pearl; shaken with air it lathers, and its
        // lather lasts minutes.
        diluent: WATER,
        density: 1030.0,
        viscosity: 3.0e-3,
        rheology: RHEO_CARREAU,
        flow_index: 0.6,
        shear_time: 0.2,
        visc_ratio: 0.05,
        mix_rate: 0.6,
        tension: 0.028,
        contact_angle: 30.0,
        heat_capacity: 3800.0,
        conductivity: 0.5,
        latent: 300.0e3,
        boil: 100.0,
        latent_vap: 2257.0e3,
        color: [0.93, 0.55, 0.72],
        color_solid: [0.95, 0.80, 0.88],
        opacity: 0.9,
        gloss: 0.9,
        foam: 1.0,
        froth: 90.0,
        ior: 1.36,
        // Glycol distearate platelets (~10 µm, a twentieth as thick): its pearl, lined up by the flow.
        cloud: 0.003,
        cloud_size: 1.0e-5,
        cloud_ior: 1.56,
        cloud_shape: 0.05,
        cloud_color: [0.97, 0.95, 0.97],
        solids: 0.3,
        ..Default::default()
    }
}

fn alcohol() -> MatProps {
    MatProps {
        // Rubbing alcohol (70 % isopropanol): 870 kg/m³, 2.4 mPa s, 0.024 N/m (CRC); boils
        // at 80.4 °C; freezes near -60 °C. It wets surfaces, mixes with water, and
        // evaporates from an open surface within minutes (here: a few per cent a second
        // of what is exposed).
        diluent: WATER,
        density: 870.0,
        viscosity: 2.8e-6,
        mix_rate: 2.0,
        tension: 0.024,
        contact_angle: 20.0,
        heat_capacity: 3000.0,
        conductivity: 0.3,
        expansion: 1.0e-3,
        latent: 150.0e3,
        freeze_shift: -60.0,
        boil: 80.4,
        latent_vap: 1000.0e3,
        volatility: 0.03,
        fume: 0.15,
        color: [0.62, 0.84, 0.92],
        color_solid: [0.85, 0.93, 0.97],
        opacity: 0.02,
        gloss: 0.35,
        foam: 0.5,
        froth: 0.2,
        ior: 1.38,
        ..Default::default()
    }
}

fn nitrogen() -> MatProps {
    MatProps {
        // Liquid nitrogen at its boiling point, -195.8 °C: 807 kg/m³, 0.16 mPa s,
        // 0.0089 N/m; c = 2.04 kJ/(kg K), k = 0.14 W/(m K), 199 kJ/kg to boil (NIST). It
        // freezes at -210 °C (25.7 kJ/kg, solid 1026 kg/m³). Everything in the tank is far
        // above its boiling point: it boils away on contact, in a white fog. Non-polar, so
        // it keeps an interface with water (the oil-like group).
        group: OILY,
        density: 807.0,
        density_solid: 1026.0,
        viscosity: 2.0e-7,
        tension: 0.0089,
        contact_angle: 10.0,
        heat_capacity: 2040.0,
        conductivity: 0.14,
        expansion: 5.0e-3,
        melt_lo: -211.0,
        melt_hi: -210.0,
        latent: 25.7e3,
        boil: -195.8,
        latent_vap: 199.0e3,
        spawn_temp: -196.0,
        fume: 1.0,
        color: [0.72, 0.86, 0.98],
        color_solid: [0.90, 0.95, 1.0],
        opacity: 0.03,
        gloss: 0.3,
        foam: 0.6,
        froth: 0.3,
        ior: 1.20,
        ..Default::default()
    }
}

fn acid() -> MatProps {
    MatProps {
        // Battery acid (35 % sulfuric acid): 1260 kg/m³, 2.8 mPa s (CRC). It neutralises lye
        // to salt water, giving off about 57 kJ per mole of water formed: of the order of
        // 180 kJ per kg of these solutions, enough to bring the mixture near boiling. Drawn
        // yellow-green (a sandbox choice: the real one is clear).
        diluent: WATER,
        density: 1260.0,
        viscosity: 2.2e-6,
        mix_rate: 1.0,
        tension: 0.074,
        heat_capacity: 3000.0,
        conductivity: 0.5,
        latent: 250.0e3,
        freeze_shift: -50.0,
        boil: 110.0,
        latent_vap: 2257.0e3,
        react_with: Material::Lye.0 as i32,
        react_into: Material::Brine.0 as i32,
        react_rate: 3.0,
        react_heat: 180.0e3,
        color: [0.78, 0.92, 0.25],
        color_solid: [0.88, 0.95, 0.70],
        opacity: 0.25,
        gloss: 0.5,
        ior: 1.38,
        solids: 0.25,
        ..Default::default()
    }
}

fn lye() -> MatProps {
    MatProps {
        // Lye (30 % sodium hydroxide, a drain cleaner): 1330 kg/m³, 13 mPa s (CRC). The
        // other half of the neutralisation (see `acid`). Drawn violet, as an indicator turns
        // in a base.
        diluent: WATER,
        density: 1330.0,
        viscosity: 1.0e-5,
        mix_rate: 1.0,
        tension: 0.09,
        heat_capacity: 3300.0,
        conductivity: 0.6,
        latent: 250.0e3,
        freeze_shift: -20.0,
        boil: 115.0,
        latent_vap: 2257.0e3,
        react_with: Material::Acid.0 as i32,
        react_into: Material::Brine.0 as i32,
        react_rate: 3.0,
        react_heat: 180.0e3,
        color: [0.62, 0.45, 0.95],
        color_solid: [0.85, 0.80, 0.97],
        opacity: 0.25,
        gloss: 0.5,
        ior: 1.40,
        solids: 0.15,
        ..Default::default()
    }
}

fn glow_fluid() -> MatProps {
    MatProps {
        // The liquid of a glow stick: a phthalate ester carrying a fluorescent dye and
        // diphenyl oxalate, 1080 kg/m³, ~12 mPa s. Its light is chemical, not thermal: it
        // glows at room temperature for hours.
        group: OILY,
        density: 1080.0,
        viscosity: 1.2e-5,
        tension: 0.035,
        heat_capacity: 1700.0,
        conductivity: 0.15,
        expansion: 7.0e-4,
        luminous: 1.0,
        glow_color: [0.35, 1.0, 0.30],
        color: [0.50, 0.95, 0.30],
        opacity: 0.5,
        gloss: 0.8,
        ior: 1.50,
        cloud: 0.001,
        cloud_size: 1.0e-6,
        cloud_ior: 1.5,
        ..Default::default()
    }
}

fn iron() -> MatProps {
    MatProps {
        // Molten iron at 1620 °C: 7000 kg/m³ (solid 7300 at its melting point, 1538 °C;
        // 247 kJ/kg), 6 mPa s; c = 820 J/(kg K), k = 35 W/(m K); emissivity ~0.35
        // (Iida & Guthrie, "The Physical Properties of Liquid Metals"). Its surface tension,
        // 1.8 N/m, is past what the solver takes: 0.75 N/m here, the top of the range.
        group: METAL,
        density: 7000.0,
        density_solid: 7300.0,
        viscosity: 8.6e-7,
        tension: 0.75,
        contact_angle: 130.0,
        heat_capacity: 820.0,
        conductivity: 35.0,
        expansion: 1.2e-4,
        melt_lo: 1530.0,
        melt_hi: 1538.0,
        latent: 247.0e3,
        emissivity: 0.35,
        boil: 2862.0,
        latent_vap: 6090.0e3,
        spawn_temp: 1620.0,
        color: [0.62, 0.60, 0.58],
        color_solid: [0.30, 0.30, 0.32],
        opacity: 1.0,
        metal: 1.0,
        gloss: 0.9,
        ..Default::default()
    }
}

fn polish() -> MatProps {
    MatProps {
        // Glitter nail polish: nitrocellulose lacquer, 1050 kg/m³, ~0.4 Pa s, 0.03 N/m,
        // loaded with reflective flakes. Solvent-based, so it keeps an interface with water.
        group: OILY,
        density: 1050.0,
        viscosity: 4.0e-4,
        tension: 0.03,
        heat_capacity: 1900.0,
        conductivity: 0.17,
        expansion: 8.0e-4,
        color: [0.80, 0.08, 0.40],
        opacity: 0.6,
        gloss: 1.0,
        ior: 1.50,
        // Pigment (0.3 µm) and aluminised flakes 0.4 mm across.
        cloud: 0.03,
        cloud_size: 3.0e-7,
        cloud_ior: 2.2,
        cloud_color: [0.80, 0.08, 0.40],
        dust: 0.004,
        dust_size: 4.0e-4,
        dust_density: 1400.0,
        dust_color: [0.85, 0.85, 0.90],
        dust_shape: 0.05,
        dust_reflect: 0.9,
        ..Default::default()
    }
}

// ==== simple liquids ===========================================================================
// `MatProps::default()` with at most five fields changed (`simple_liquids_are_simple` checks
// it), `color` being one of them. A comment says what each shows.

fn blank() -> MatProps {
    // The neutral liquid itself: as dense and thin as water, clear, nothing else.
    MatProps { color: [0.55, 0.66, 0.74], ..Default::default() }
}

fn thick() -> MatProps {
    // Viscosity alone (20 000 x water's): it folds, coils and slumps.
    MatProps { color: [0.95, 0.55, 0.20], viscosity: 0.02, ..Default::default() }
}

fn heavy() -> MatProps {
    // Density alone: five times water's. It sinks through everything but liquid metal.
    MatProps { color: [0.42, 0.30, 0.62], density: 5000.0, opacity: 0.6, ..Default::default() }
}

fn light() -> MatProps {
    // Light and immiscible with water: it floats on every other liquid.
    MatProps { color: [0.98, 0.93, 0.55], density: 450.0, group: OILY, ..Default::default() }
}

fn beady() -> MatProps {
    // High surface tension and a surface it does not wet: it pulls itself into beads.
    MatProps { color: [0.30, 0.80, 0.85], tension: 0.4, contact_angle: 150.0, ..Default::default() }
}

fn creeper() -> MatProps {
    // A small contact angle: it spreads over solids and climbs into corners.
    MatProps { color: [0.45, 0.85, 0.55], contact_angle: 15.0, ..Default::default() }
}

fn putty() -> MatProps {
    // Elastic (a Maxwell liquid): it bounces, stretches into strands, and flows in the end.
    MatProps { color: [0.95, 0.40, 0.60], shear_modulus: 150.0, relax_time: 2.0, viscosity: 5.0e-3, ..Default::default() }
}

fn paste() -> MatProps {
    // A yield stress (Bingham): it holds its shape until pushed hard enough.
    MatProps { color: [0.90, 0.88, 0.80], rheology: RHEO_HERSCHEL_BULKLEY, yield_stress: 40.0, viscosity: 5.0e-3, opacity: 1.0, ..Default::default() }
}

fn thinner() -> MatProps {
    // Shear thinning (a power law, n = 0.4): thick at rest, runny where it is stirred or poured.
    MatProps { color: [0.85, 0.25, 0.20], rheology: RHEO_HERSCHEL_BULKLEY, viscosity: 0.01, flow_index: 0.4, ..Default::default() }
}

fn jammer() -> MatProps {
    // Shear thickening: runny when left alone, solid where it is hit.
    MatProps { color: [0.75, 0.80, 0.95], rheology: RHEO_THICKENING, viscosity: 2.0e-3, flow_index: 3.0, visc_ratio: 2000.0, ..Default::default() }
}

fn magnetic() -> MatProps {
    // Magnetic susceptibility: magnets pull it and raise spikes on it.
    MatProps { color: [0.16, 0.14, 0.22], susceptibility: 3.0, cloud: 0.05, cloud_color: [0.16, 0.14, 0.22], ..Default::default() }
}

fn glowing() -> MatProps {
    // Light of its own.
    MatProps { color: [0.25, 0.75, 0.95], luminous: 1.2, glow_color: [0.25, 0.80, 1.0], ..Default::default() }
}

fn fizzy() -> MatProps {
    // Dissolved gas: bubbles rise through it and gather in a short-lived foam.
    MatProps { color: [0.95, 0.80, 0.35], fizz: 1.0, foam: 1.0, froth: 3.0, ..Default::default() }
}

fn sudsy() -> MatProps {
    // Foam that lasts: wherever it splashes it lathers, for a minute.
    MatProps { color: [0.55, 0.75, 0.98], foam: 1.0, froth: 60.0, tension: 0.03, ..Default::default() }
}

fn vanishing() -> MatProps {
    // Volatile: its open surface evaporates in a mist, a seventh of it a second.
    MatProps { color: [0.80, 0.95, 0.90], volatility: 0.15, fume: 0.7, ..Default::default() }
}

fn grainy() -> MatProps {
    // Internal friction: it pours like a liquid and piles up like dry sand. (With some
    // viscosity: on a liquid as thin as water the friction's yield stress, which follows the
    // last substep's pressure, raced. README, Limitations.)
    MatProps { color: [0.85, 0.75, 0.50], friction: 0.6, viscosity: 2.0e-3, cloud: 0.55, cloud_color: [0.85, 0.75, 0.50], ..Default::default() }
}

fn warming() -> MatProps {
    // Makes heat inside itself (400 W/kg: 0.1 K a second): it warms, convects, and warms
    // what it touches.
    MatProps { color: [0.95, 0.45, 0.30], self_heat: 400.0, ..Default::default() }
}

fn setting() -> MatProps {
    // A melting point above room temperature: poured warm, it sets as it cools.
    MatProps { color: [0.60, 0.90, 0.75], melt_lo: 28.0, melt_hi: 32.0, latent: 60.0e3, spawn_temp: 60.0, ..Default::default() }
}

fn mirror() -> MatProps {
    // A mirror surface.
    MatProps { color: [0.85, 0.87, 0.92], metal: 1.0, opacity: 1.0, gloss: 1.0, ..Default::default() }
}

fn pearly() -> MatProps {
    // Rheoscopic: mica platelets (15 µm across, a twentieth as thick, 0.2 % by volume) that
    // the flow lines up, so its shear shows in silky bands until Brownian rotation undoes it.
    MatProps { color: [0.70, 0.62, 0.90], opacity: 0.8, cloud: 0.002, cloud_size: 1.5e-5, cloud_shape: 0.05, ..Default::default() }
}

fn glitter() -> MatProps {
    // Glitter: mirror flakes a millimetre across that tumble in the flow, flash as they turn
    // to the light, and settle into a glinting layer when it stops.
    MatProps { color: [0.20, 0.30, 0.75], dust: 6.0e-4, dust_size: 1.0e-3, dust_shape: 0.05, dust_reflect: 0.9, ..Default::default() }
}

fn snowy() -> MatProps {
    // A snow globe: white grains of 1.5 mm in a liquid twenty times as viscous as water, so a
    // shake lifts them into a flurry that takes many seconds to settle (Stokes).
    MatProps { color: [0.70, 0.85, 0.95], viscosity: 2.0e-5, dust: 2.0e-3, dust_size: 1.5e-3, dust_color: [0.97, 0.97, 1.0], ..Default::default() }
}

fn mood() -> MatProps {
    // Thermochromic: blue when cool, pink when warm (22 - 45 °C). Try the Heat brush.
    MatProps { color: [0.20, 0.35, 0.90], color_hot: [0.98, 0.40, 0.55], thermo_lo: 22.0, thermo_hi: 45.0, opacity: 0.8, ..Default::default() }
}

fn dye() -> MatProps {
    // A strong colour that dissolves in water and stains what it touches.
    MatProps { color: [0.75, 0.05, 0.55], diluent: WATER, mix_rate: 0.5, opacity: 1.0, ..Default::default() }
}

fn silt() -> MatProps {
    // Grains suspended in water: they cloud it, then settle out into a bed.
    MatProps { color: [0.62, 0.50, 0.36], diluent: WATER, density: 2200.0, grain: 1.0e-4, load: 0.4, ..Default::default() }
}

fn part_a() -> MatProps {
    // One half of a two-part resin: where it meets Part B both set into solid Resin, warming.
    MatProps {
        color: [0.95, 0.80, 0.30],
        react_with: Material::PartB.0 as i32,
        react_into: Material::Resin.0 as i32,
        react_rate: 2.0,
        react_heat: 60.0e3,
        ..Default::default()
    }
}

fn part_b() -> MatProps {
    // The other half (it gives off no heat itself).
    MatProps { color: [0.35, 0.60, 0.95], react_with: Material::PartA.0 as i32, react_into: Material::Resin.0 as i32, react_rate: 2.0, ..Default::default() }
}

fn ember() -> MatProps {
    // Poured glowing hot, with a crust that cracks as it cools and freezes.
    MatProps { color: [0.12, 0.08, 0.07], spawn_temp: 1100.0, melt_lo: 850.0, melt_hi: 1000.0, ..Default::default() }
}

fn sheen() -> MatProps {
    // Thin-film colours on an oil-like liquid.
    MatProps { color: [0.30, 0.32, 0.40], group: OILY, tension: 0.03, ior: 1.47, ..Default::default() }
}

// ==== by-products ================================================================================

fn salt() -> MatProps {
    MatProps {
        // Halite: 2165 kg/m³, melts at 801 °C; what evaporating brine leaves behind. Solid
        // at every temperature the sandbox reaches; it does not redissolve.
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
        gloss: 0.2,
        cloud: 0.5,
        cloud_size: 5.0e-5,
        cloud_ior: 1.54,
        cloud_color: [0.94, 0.94, 0.92],
        ..Default::default()
    }
}

fn steam() -> MatProps {
    MatProps {
        // Saturated steam at 100 °C: 0.60 kg/m³, 1.2e-5 Pa s. A particle of steam stands for
        // the vapour of the water parcel it boiled from: it carries that parcel's mass and
        // heat (sim/mod.rs: `gpu_materials`) and condenses back into exactly that water. Its
        // density is the vapour's, so it is buoyant; its volume one particle's (README,
        // Limitations). Conductivity and surface tension are water's.
        density: 0.6,
        viscosity: 2.0e-5,
        expansion: 0.0,
        boil: 100.0,
        latent_vap: 2257.0e3,
        emissivity: 0.0,
        color: [0.92, 0.94, 0.96],
        color_solid: [0.92, 0.94, 0.96],
        gloss: 0.0,
        ..Default::default()
    }
}

fn resin() -> MatProps {
    MatProps {
        // Cured epoxy: 1150 kg/m³, c = 1.1 kJ/(kg K), k = 0.2 W/(m K). What Part A and
        // Part B set into: solid below 300 °C (it chars rather than melts; the melting
        // range only keeps it solid).
        density: 1150.0,
        viscosity: 1.0e-2,
        heat_capacity: 1100.0,
        conductivity: 0.2,
        expansion: 1.0e-4,
        melt_lo: 300.0,
        melt_hi: 320.0,
        latent: 100.0e3,
        color: [0.55, 0.80, 0.45],
        color_solid: [0.45, 0.78, 0.42],
        opacity: 0.6,
        gloss: 0.9,
        ..Default::default()
    }
}

/// How many built-in rows are real liquids, and how many simple ones follow them.
pub const REAL_COUNT: usize = 27;
pub const SIMPLE_COUNT: usize = 29;

/// The built-in materials, in table order (`Material::Water` ...).
pub static BUILTINS: [Builtin; REAL_COUNT + SIMPLE_COUNT + 3] = [
    Builtin { key: "water", name: "Water", category: "Everyday", props: water },
    Builtin { key: "honey", name: "Honey", category: "Kitchen", props: honey },
    Builtin { key: "oil", name: "Olive oil", category: "Kitchen", props: oil },
    Builtin { key: "mud", name: "Mud", category: "Earth", props: mud },
    Builtin { key: "ink", name: "Ink", category: "Everyday", props: ink },
    Builtin { key: "chocolate", name: "Molten chocolate", category: "Kitchen", props: chocolate },
    Builtin { key: "clay", name: "Clay slip", category: "Earth", props: clay },
    Builtin { key: "blood", name: "Blood", category: "Body", props: blood },
    Builtin { key: "mercury", name: "Mercury", category: "Metal", props: mercury },
    Builtin { key: "lava", name: "Lava", category: "Hot", props: lava },
    Builtin { key: "glass", name: "Molten glass", category: "Hot", props: glass },
    Builtin { key: "milk", name: "Milk", category: "Kitchen", props: milk },
    Builtin { key: "slime", name: "Slime", category: "Strange", props: slime },
    Builtin { key: "ferrofluid", name: "Ferrofluid", category: "Strange", props: ferrofluid },
    Builtin { key: "brine", name: "Brine", category: "Everyday", props: brine },
    Builtin { key: "wax", name: "Candle wax", category: "Hot", props: wax },
    Builtin { key: "oobleck", name: "Oobleck", category: "Strange", props: oobleck },
    Builtin { key: "gallium", name: "Gallium", category: "Metal", props: gallium },
    Builtin { key: "beer", name: "Beer", category: "Kitchen", props: beer },
    Builtin { key: "soap", name: "Liquid soap", category: "Everyday", props: soap },
    Builtin { key: "alcohol", name: "Rubbing alcohol", category: "Workshop", props: alcohol },
    Builtin { key: "nitrogen", name: "Liquid nitrogen", category: "Cold", props: nitrogen },
    Builtin { key: "acid", name: "Battery acid", category: "Workshop", props: acid },
    Builtin { key: "lye", name: "Lye", category: "Workshop", props: lye },
    Builtin { key: "glowfluid", name: "Glow-stick fluid", category: "Strange", props: glow_fluid },
    Builtin { key: "iron", name: "Molten iron", category: "Metal", props: iron },
    Builtin { key: "polish", name: "Glitter polish", category: "Workshop", props: polish },
    Builtin { key: "blank", name: "Blank", category: "Simple", props: blank },
    Builtin { key: "thick", name: "Thick", category: "Simple", props: thick },
    Builtin { key: "heavy", name: "Heavy", category: "Simple", props: heavy },
    Builtin { key: "light", name: "Light", category: "Simple", props: light },
    Builtin { key: "beady", name: "Beady", category: "Simple", props: beady },
    Builtin { key: "creeper", name: "Creeper", category: "Simple", props: creeper },
    Builtin { key: "putty", name: "Putty", category: "Simple", props: putty },
    Builtin { key: "paste", name: "Paste", category: "Simple", props: paste },
    Builtin { key: "thinner", name: "Thinner", category: "Simple", props: thinner },
    Builtin { key: "jammer", name: "Jammer", category: "Simple", props: jammer },
    Builtin { key: "magnetic", name: "Magnetic", category: "Simple", props: magnetic },
    Builtin { key: "glowing", name: "Glowing", category: "Simple", props: glowing },
    Builtin { key: "fizzy", name: "Fizzy", category: "Simple", props: fizzy },
    Builtin { key: "sudsy", name: "Sudsy", category: "Simple", props: sudsy },
    Builtin { key: "vanishing", name: "Vanishing", category: "Simple", props: vanishing },
    Builtin { key: "grainy", name: "Grainy", category: "Simple", props: grainy },
    Builtin { key: "warming", name: "Warming", category: "Simple", props: warming },
    Builtin { key: "setting", name: "Setting", category: "Simple", props: setting },
    Builtin { key: "mirror", name: "Mirror", category: "Simple", props: mirror },
    Builtin { key: "pearly", name: "Pearly", category: "Simple", props: pearly },
    Builtin { key: "glitter", name: "Glitter", category: "Simple", props: glitter },
    Builtin { key: "snowy", name: "Snow globe", category: "Simple", props: snowy },
    Builtin { key: "mood", name: "Mood", category: "Simple", props: mood },
    Builtin { key: "dye", name: "Dye", category: "Simple", props: dye },
    Builtin { key: "silt", name: "Silt", category: "Simple", props: silt },
    Builtin { key: "parta", name: "Part A", category: "Simple", props: part_a },
    Builtin { key: "partb", name: "Part B", category: "Simple", props: part_b },
    Builtin { key: "ember", name: "Ember", category: "Simple", props: ember },
    Builtin { key: "sheen", name: "Sheen", category: "Simple", props: sheen },
    Builtin { key: "salt", name: "Salt", category: "By-product", props: salt },
    Builtin { key: "steam", name: "Steam", category: "By-product", props: steam },
    Builtin { key: "resin", name: "Resin", category: "By-product", props: resin },
];

/// Rows the liquid picker offers: every built-in but the by-products, and every custom row.
pub fn pickable(row: usize) -> bool {
    !(REAL_COUNT + SIMPLE_COUNT..BUILTINS.len()).contains(&row)
}

/// The fields in which two definitions differ, by name (the three colours count as one
/// field each).
#[cfg(test)]
pub fn changed_fields(a: &MatProps, b: &MatProps) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = super::params::MAT_PARAMS
        .iter()
        .filter(|p| {
            let mut pa = super::SimParams::default();
            let mut pb = super::SimParams::default();
            pa.mat[0] = *a;
            pb.mat[0] = *b;
            (p.get)(&pa, 0) != (p.get)(&pb, 0)
        })
        .map(|p| p.key)
        .collect();
    for (name, x, y) in [
        ("color", a.color, b.color),
        ("color_solid", a.color_solid, b.color_solid),
        ("glow_color", a.glow_color, b.glow_color),
        ("color_hot", a.color_hot, b.color_hot),
        ("cloud_color", a.cloud_color, b.cloud_color),
        ("dust_color", a.dust_color, b.dust_color),
    ] {
        if x != y {
            out.push(name);
        }
    }
    for (name, x, y) in [("group", a.group as i32, b.group as i32), ("diluent", a.diluent, b.diluent), ("residue", a.residue, b.residue), ("react_with", a.react_with, b.react_with), ("react_into", a.react_into, b.react_into)] {
        if x != y {
            out.push(name);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_match_the_constants() {
        let at = |m: Material, key: &str| assert_eq!(BUILTINS[m.index()].key, key);
        at(Material::Water, "water");
        at(Material::Honey, "honey");
        at(Material::Gallium, "gallium");
        at(Material::Polish, "polish");
        at(Material::Blank, "blank");
        at(Material::Sheen, "sheen");
        at(Material::Salt, "salt");
        at(Material::Steam, "steam");
        at(Material::Resin, "resin");
        assert_eq!(Material::Blank.index(), REAL_COUNT);
        assert_eq!(Material::Salt.index(), REAL_COUNT + SIMPLE_COUNT);
        let mut keys: Vec<_> = BUILTINS.iter().map(|b| b.key).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), BUILTINS.len(), "a key is used twice");
    }

    /// Every simple liquid is the neutral liquid with one to five changes, its colour among
    /// them, and no two change the same set of properties.
    #[test]
    fn simple_liquids_are_simple() {
        let base = MatProps::default();
        let mut seen: Vec<(Vec<&str>, &str)> = Vec::new();
        for b in &BUILTINS[REAL_COUNT..REAL_COUNT + SIMPLE_COUNT] {
            let changed = changed_fields(&base, &(b.props)());
            assert!((1..=5).contains(&changed.len()), "{}: {} changes ({changed:?})", b.name, changed.len());
            assert!(changed.contains(&"color"), "{}: no colour of its own", b.name);
            if let Some((_, other)) = seen.iter().find(|(c, _)| *c == changed) {
                panic!("{} and {other} change the same properties: {changed:?}", b.name);
            }
            seen.push((changed, b.name));
        }
    }

    /// Reactions and residues point at rows that exist, and partners point back.
    #[test]
    fn references_are_rows() {
        for (row, b) in BUILTINS.iter().enumerate() {
            let m = (b.props)();
            for r in [m.diluent, m.residue, m.react_with, m.react_into] {
                assert!(r < BUILTINS.len() as i32, "{}", b.name);
            }
            if m.react_with >= 0 {
                let other = (BUILTINS[m.react_with as usize].props)();
                assert_eq!(other.react_with, row as i32, "{}: its partner does not react back", b.name);
            }
        }
    }
}
