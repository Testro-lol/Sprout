//! Built-in maps.
//!
//! Maps are authored in a resolution-independent space of 160 x 90 "map units" (16:9, origin
//! bottom-left, +y up) and scaled to whatever grid the simulation runs at, so the same map
//! works at every quality setting.

use crate::{
    scene::{Scene, Shape, WALL},
    sim::{Magnet, Material, PPC, Particle},
};

/// Width of the map space; a 16:9 grid makes it 90 units tall.
pub const MAP_W: f32 = 160.0;

pub struct MapDef {
    pub name: &'static str,
    pub blurb: &'static str,
    /// Physical width of the tank in metres. 0 keeps the default cell size (1 cm, a tank
    /// several metres wide, where gravity dwarfs surface tension). A tabletop map sets its
    /// width instead, so that it shows the same physics at every grid resolution.
    pub width_m: f32,
    build: fn(&mut MapBuilder),
}

impl MapDef {
    /// Cell size this map wants on a grid `grid_w` cells wide, m.
    pub fn cell_size(&self, grid_w: u32) -> Option<f32> {
        (self.width_m > 0.0).then(|| self.width_m / grid_w as f32)
    }
}

pub struct MapBuilder {
    scale: f32,
    shapes: Vec<Shape>,
    temps: Vec<Option<f32>>,
    /// Rectangle, material, temperature (°C; `None` = the material's spawn temperature).
    fills: Vec<([f32; 4], Material, Option<f32>)>,
    magnets: Vec<Magnet>,
    ambient: Option<f32>,
}

impl MapBuilder {
    /// Holds the shape added last at `t` °C (a hot plate, a cold block).
    fn held_at(&mut self, t: f32) {
        let n = self.shapes.len();
        self.temps.resize(n, None);
        self.temps[n - 1] = Some(t);
    }

    /// A magnet at (x, y), radius r (map units), north pole `degrees` clockwise from up,
    /// field `tesla` at its rim.
    fn magnet(&mut self, x: f32, y: f32, r: f32, degrees: f32, tesla: f32) {
        let s = self.scale;
        self.magnets.push(Magnet::new([x * s, y * s], r * s, degrees, tesla));
    }

    /// The air temperature of the map, °C.
    fn air(&mut self, t: f32) {
        self.ambient = Some(t);
    }

    fn circle(&mut self, x: f32, y: f32, r: f32) {
        let s = self.scale;
        self.shapes.push(Shape::Circle { c: [x * s, y * s], r: r * s });
    }

    /// Axis-aligned box given by two corners.
    fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let s = self.scale;
        self.shapes.push(Shape::Box {
            c: [(x0 + x1) * 0.5 * s, (y0 + y1) * 0.5 * s],
            half: [(x1 - x0).abs() * 0.5 * s, (y1 - y0).abs() * 0.5 * s],
            angle: 0.0,
        });
    }

    /// A straight plank of the given thickness between two points.
    fn plank(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32) {
        let s = self.scale;
        let (dx, dy) = (x1 - x0, y1 - y0);
        self.shapes.push(Shape::Box {
            c: [(x0 + x1) * 0.5 * s, (y0 + y1) * 0.5 * s],
            half: [dx.hypot(dy) * 0.5 * s, thickness * 0.5 * s],
            angle: dy.atan2(dx),
        });
    }

    /// A wedge filling the box between two corners; `high_right` puts the tall side right.
    fn wedge(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, high_right: bool) {
        let s = self.scale;
        self.shapes.push(Shape::Ramp {
            c: [(x0 + x1) * 0.5 * s, (y0 + y1) * 0.5 * s],
            half: [(x1 - x0).abs() * 0.5 * s, (y1 - y0).abs() * 0.5 * s],
            angle: 0.0,
            flip: high_right,
        });
    }

    fn fill(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, material: Material) {
        self.fills.push(([x0, y0, x1, y1], material, None));
    }

    /// A fill at `t` °C instead of the material's spawn temperature.
    fn fill_at(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, material: Material, t: f32) {
        self.fills.push(([x0, y0, x1, y1], material, Some(t)));
    }
}

pub const MAPS: &[MapDef] = &[
    MapDef {
        name: "Playground",
        blurb: "A pool, a few obstacles, and oil and honey about to fall in.",
        width_m: 0.0,
        build: |m| {
            m.fill(0.0, 0.0, 160.0, 22.0, Material::Water);
            m.plank(14.0, 60.0, 70.0, 44.0, 2.0);
            m.circle(102.0, 40.0, 8.0);
            m.rect(126.0, 30.0, 150.0, 33.0);
            m.fill(18.0, 64.0, 44.0, 84.0, Material::Oil);
            m.fill(92.0, 56.0, 112.0, 82.0, Material::Honey);
        },
    },
    MapDef {
        name: "Dam Break",
        blurb: "A tall column of water collapses across the floor and over a step.",
        width_m: 0.0,
        build: |m| {
            m.fill(0.0, 0.0, 40.0, 66.0, Material::Water);
            m.rect(84.0, 0.0, 90.0, 12.0);
            m.wedge(128.0, 0.0, 160.0, 22.0, true);
        },
    },
    MapDef {
        name: "Upside Down",
        blurb: "Honey over water over oil: the layers overturn into their stable order.",
        width_m: 0.0,
        build: |m| {
            m.fill(0.0, 0.0, 160.0, 20.0, Material::Oil);
            m.fill(0.0, 20.0, 160.0, 40.0, Material::Water);
            m.fill(0.0, 40.0, 160.0, 58.0, Material::Honey);
        },
    },
    MapDef {
        name: "Three Columns",
        blurb: "Oil, water and honey side by side: honey slides under, oil rides over.",
        width_m: 0.0,
        build: |m| {
            m.fill(0.0, 0.0, 53.0, 58.0, Material::Oil);
            m.fill(53.0, 0.0, 107.0, 58.0, Material::Water);
            m.fill(107.0, 0.0, 160.0, 58.0, Material::Honey);
        },
    },
    MapDef {
        name: "Cascade",
        blurb: "Three reservoirs drain down a zig-zag of ramps into a shared basin.",
        width_m: 0.0,
        build: |m| {
            m.plank(0.0, 70.0, 96.0, 60.0, 2.0);
            m.plank(160.0, 50.0, 62.0, 40.0, 2.0);
            m.plank(0.0, 30.0, 100.0, 20.0, 2.0);
            m.circle(118.0, 8.0, 4.0);
            m.circle(133.0, 15.0, 4.0);
            m.circle(147.0, 7.0, 4.0);
            m.fill(2.0, 73.0, 36.0, 88.0, Material::Water);
            m.fill(128.0, 54.0, 158.0, 76.0, Material::Honey);
            m.fill(2.0, 33.0, 26.0, 46.0, Material::Oil);
        },
    },
    MapDef {
        name: "Empty Tank",
        blurb: "Nothing but walls. Build your own.",
        width_m: 0.0,
        build: |_| {},
    },

    MapDef {
        name: "Terrace Falls",
        blurb: "A hillside of spill pools: blue waterfalls race the slow honey tributary.",
        width_m: 0.0,
        build: |m| {
            // Stepped bedrock leaves a broad plunge pool on the right.
            for (x, top) in [(0.0, 62.0), (25.0, 48.0), (50.0, 34.0), (75.0, 20.0)] {
                m.rect(x, 0.0, x + 25.0, top);
                m.rect(x + 21.0, top, x + 25.0, top + 4.0);
            }
            m.fill(2.0, 66.0, 23.0, 87.0, Material::Water);
            m.fill(27.0, 52.0, 48.0, 70.0, Material::Water);
            m.fill(52.0, 38.0, 73.0, 53.0, Material::Water);
            m.fill(77.0, 24.0, 98.0, 36.0, Material::Oil);
            m.plank(158.0, 65.0, 124.0, 53.0, 2.5);
            m.plank(116.0, 40.0, 143.0, 32.0, 2.5);
            m.fill(134.0, 67.0, 157.0, 85.0, Material::Honey);
            m.circle(110.0, 12.0, 5.0);
            m.circle(136.0, 8.0, 4.0);
            m.wedge(148.0, 0.0, 160.0, 17.0, true);
            m.fill(101.0, 2.0, 148.0, 10.0, Material::Water);
        },
    },
    MapDef {
        name: "Pinball Garden",
        blurb: "Three colorful showers weave through staggered pegs and splash into scalloped pools.",
        width_m: 0.0,
        build: |m| {
            m.plank(0.0, 73.0, 26.0, 65.0, 2.5);
            m.plank(160.0, 73.0, 134.0, 65.0, 2.5);
            for row in 0..4 {
                let y = 59.0 - row as f32 * 12.0;
                for col in 0..7 {
                    let x = 19.0 + col as f32 * 19.0 + (row % 2) as f32 * 9.5;
                    m.circle(x, y, 3.0);
                }
            }
            for x in [32.0, 64.0, 96.0, 128.0] {
                m.rect(x - 2.0, 0.0, x + 2.0, 9.0);
                m.circle(x, 9.0, 2.0);
            }
            m.fill(7.0, 76.0, 52.0, 87.0, Material::Water);
            m.fill(58.0, 73.0, 102.0, 87.0, Material::Honey);
            m.fill(108.0, 76.0, 153.0, 87.0, Material::Oil);
            m.fill(2.0, 2.0, 158.0, 6.0, Material::Water);
        },
    },
    MapDef {
        name: "Hourglass",
        blurb: "Layered liquid funnels through a narrow waist, fans over a crown, and fills the chamber below.",
        width_m: 0.0,
        build: |m| {
            // The ten-unit throat stays comfortably resolved even at Low quality.
            m.plank(0.0, 74.0, 74.0, 45.0, 3.0);
            m.plank(160.0, 74.0, 86.0, 45.0, 3.0);
            m.rect(72.5, 39.0, 75.5, 45.0);
            m.rect(84.5, 39.0, 87.5, 45.0);
            m.fill(6.0, 74.0, 154.0, 80.0, Material::Water);
            m.fill(6.0, 80.0, 154.0, 87.0, Material::Oil);
            m.fill(37.0, 62.0, 123.0, 74.0, Material::Honey);
            m.wedge(57.0, 23.0, 80.0, 33.0, true);
            m.wedge(80.0, 23.0, 103.0, 33.0, false);
            m.circle(45.0, 17.0, 4.0);
            m.circle(115.0, 17.0, 4.0);
            m.plank(6.0, 24.0, 30.0, 12.0, 3.0);
            m.plank(154.0, 24.0, 130.0, 12.0, 3.0);
            m.fill(2.0, 2.0, 158.0, 8.0, Material::Water);
        },
    },
    MapDef {
        name: "Twin Spillways",
        blurb: "Opposing torrents leap into a central splash, then separate around an island.",
        width_m: 0.0,
        build: |m| {
            m.plank(0.0, 64.0, 59.0, 43.0, 3.0);
            m.plank(160.0, 64.0, 101.0, 43.0, 3.0);
            m.fill(2.0, 67.0, 43.0, 87.0, Material::Water);
            m.fill(117.0, 67.0, 158.0, 87.0, Material::Oil);
            m.circle(80.0, 25.0, 8.0);
            m.rect(77.0, 0.0, 83.0, 25.0);
            m.plank(12.0, 31.0, 44.0, 20.0, 2.5);
            m.plank(148.0, 31.0, 116.0, 20.0, 2.5);
            for x in [25.0, 52.0, 108.0, 135.0] {
                m.circle(x, 10.0, 3.5);
            }
            m.wedge(0.0, 0.0, 15.0, 18.0, false);
            m.wedge(145.0, 0.0, 160.0, 18.0, true);
            m.fill(3.0, 2.0, 75.0, 9.0, Material::Honey);
            m.fill(85.0, 2.0, 157.0, 9.0, Material::Water);
        },
    },
    MapDef {
        name: "Canopy",
        blurb: "A liquid tree splits into branches, spills from leaf shelves, and gathers among the roots.",
        width_m: 0.0,
        build: |m| {
            m.fill(55.0, 74.0, 80.0, 87.0, Material::Water);
            m.fill(80.0, 74.0, 105.0, 87.0, Material::Oil);
            m.wedge(48.0, 57.0, 80.0, 69.0, true);
            m.wedge(80.0, 57.0, 112.0, 69.0, false);
            m.rect(78.5, 19.0, 81.5, 57.0);
            m.plank(12.0, 51.0, 43.0, 43.0, 2.5);
            m.plank(148.0, 51.0, 117.0, 43.0, 2.5);
            m.plank(75.0, 39.0, 54.0, 30.0, 2.5);
            m.plank(85.0, 39.0, 106.0, 30.0, 2.5);
            m.plank(25.0, 25.0, 46.0, 19.0, 2.5);
            m.plank(135.0, 25.0, 114.0, 19.0, 2.5);
            m.fill(15.0, 54.0, 34.0, 67.0, Material::Honey);
            m.fill(126.0, 54.0, 145.0, 67.0, Material::Honey);
            m.plank(80.0, 20.0, 63.0, 4.0, 3.0);
            m.plank(80.0, 20.0, 97.0, 4.0, 3.0);
            for x in [16.0, 39.0, 121.0, 144.0] {
                m.circle(x, 7.0, 3.0);
            }
            m.fill(2.0, 2.0, 158.0, 10.0, Material::Water);
        },
    },
    MapDef {
        name: "Sunken Ruins",
        blurb: "A breaking wave threads ruined towers while oil and honey spill from the rooftops.",
        width_m: 0.0,
        build: |m| {
            m.fill(2.0, 2.0, 30.0, 85.0, Material::Water);
            // Doorways under each lintel let the surge cross the whole city.
            for (x, h) in [(46.0, 31.0), (83.0, 45.0), (121.0, 27.0)] {
                m.rect(x, 12.0, x + 4.0, h);
                m.rect(x + 19.0, 12.0, x + 23.0, h);
                m.rect(x - 3.0, h, x + 26.0, h + 3.0);
            }
            m.plank(37.0, 66.0, 66.0, 58.0, 2.5);
            m.fill(40.0, 69.0, 63.0, 85.0, Material::Oil);
            m.fill(86.0, 51.0, 103.0, 76.0, Material::Honey);
            m.plank(114.0, 61.0, 145.0, 52.0, 2.5);
            m.fill(122.0, 65.0, 145.0, 82.0, Material::Oil);
            m.circle(36.0, 8.0, 3.0);
            m.circle(76.0, 9.0, 4.0);
            m.circle(114.0, 6.0, 3.0);
            m.wedge(148.0, 0.0, 160.0, 20.0, true);
            m.fill(32.0, 2.0, 158.0, 7.0, Material::Water);
        },
    },
    MapDef {
        name: "Oil Drops",
        blurb: "A tabletop tank half a metre wide: at this size oil beads into round drops and lenses.",
        // 1 mm cells at the default resolution. The capillary length of oil in water,
        // sqrt(sigma / (drho g)), is 5 mm: blobs up to about a centimetre pull themselves
        // round, larger ones flatten. Nothing but the length scale differs from the other
        // maps; the fluids are the same.
        width_m: 0.512,
        build: |m| {
            m.fill(0.0, 0.0, 160.0, 40.0, Material::Water);
            // Oil released under water: squares of 5 to 8 mm (Bond number below 1) rise as
            // round drops; the 13 mm one is above the capillary length and deforms.
            m.fill(14.0, 6.0, 15.6, 7.6, Material::Oil);
            m.fill(30.0, 5.0, 32.0, 7.0, Material::Oil);
            m.fill(50.0, 5.0, 52.5, 7.5, Material::Oil);
            m.fill(74.0, 4.0, 78.0, 8.0, Material::Oil);
            m.fill(100.0, 12.0, 102.0, 14.0, Material::Oil);
            m.fill(118.0, 18.0, 119.6, 19.6, Material::Oil);
            // Oil and honey falling in from above.
            m.fill(24.0, 60.0, 27.5, 63.5, Material::Oil);
            m.fill(62.0, 66.0, 66.0, 70.0, Material::Oil);
            m.fill(92.0, 58.0, 94.5, 60.5, Material::Oil);
            m.fill(132.0, 62.0, 136.0, 66.0, Material::Honey);
            // A shelf with two water drops and an oil drop resting on it.
            m.rect(104.0, 48.0, 156.0, 50.0);
            m.fill(110.0, 50.0, 114.0, 53.0, Material::Water);
            m.fill(124.0, 50.0, 127.0, 52.5, Material::Oil);
            m.fill(140.0, 50.0, 146.0, 54.0, Material::Water);
        },
    },
    MapDef {
        name: "Lava Meets Water",
        blurb: "Basaltic lava at 1200 °C runs down a slope into a pool: the water flashes off, the lava crusts and sets.",
        width_m: 0.0,
        build: |m| {
            m.wedge(0.0, 0.0, 70.0, 36.0, false);
            m.fill(4.0, 40.0, 30.0, 62.0, Material::Lava);
            m.fill(78.0, 0.0, 160.0, 18.0, Material::Water);
            m.rect(70.0, 0.0, 74.0, 10.0);
        },
    },
    MapDef {
        name: "Density Tower",
        blurb: "Mercury, water, oil and molten wax poured in the wrong order sort themselves by density (a warm tower, 70 °C, so the wax stays molten).",
        width_m: 0.0,
        build: |m| {
            // Warm, so that the wax stays molten (it melts at 52 - 62 °C): poured into
            // oil at room temperature it sets into flakes about as dense as the oil
            // (900 against 898 kg/m³), which then stay suspended in it, as real ones do.
            m.air(70.0);
            m.fill(40.0, 0.0, 120.0, 10.0, Material::Wax);
            m.fill_at(40.0, 10.0, 120.0, 20.0, Material::Oil, 70.0);
            m.fill_at(40.0, 20.0, 120.0, 32.0, Material::Water, 70.0);
            m.fill_at(60.0, 40.0, 100.0, 46.0, Material::Mercury, 70.0);
            m.rect(34.0, 0.0, 38.0, 50.0);
            m.rect(122.0, 0.0, 126.0, 50.0);
        },
    },
    MapDef {
        name: "Glass and Slime",
        blurb: "Molten glass droops off a ledge in strands and sets as it cools; slime drops onto a peg, stretches and springs back.",
        width_m: 0.0,
        build: |m| {
            m.rect(0.0, 48.0, 48.0, 52.0);
            m.fill(6.0, 52.0, 40.0, 70.0, Material::Glass);
            m.circle(118.0, 30.0, 4.0);
            m.fill(108.0, 52.0, 128.0, 70.0, Material::Slime);
            m.wedge(60.0, 0.0, 100.0, 14.0, true);
        },
    },
    MapDef {
        name: "Mud Flats",
        blurb: "Mud slumps and settles out, a pile of clay holds its shape, oobleck takes a falling ball of mercury.",
        width_m: 0.0,
        build: |m| {
            m.fill(4.0, 0.0, 40.0, 30.0, Material::Mud);
            m.fill(58.0, 0.0, 78.0, 28.0, Material::Clay);
            m.fill(100.0, 0.0, 156.0, 16.0, Material::Oobleck);
            m.fill(124.0, 60.0, 130.0, 66.0, Material::Mercury);
            m.fill(40.0, 0.0, 56.0, 6.0, Material::Water);
        },
    },
    MapDef {
        name: "Ferrofluid",
        blurb: "A tabletop dish of ferrofluid over a strong magnet: it heaps up and breaks into spikes (1 mm cells; spacing ~1 cm).",
        // 1 mm cells at the default resolution: the spikes' spacing, 2 pi sqrt(sigma / (rho g))
        // ~ 1 cm, spans about ten cells. On 1 cm cells it is below one cell: not resolved.
        width_m: 0.512,
        build: |m| {
            m.fill(20.0, 0.0, 140.0, 6.0, Material::Ferrofluid);
            m.magnet(80.0, -6.0, 5.0, 0.0, 0.5);
            m.fill(10.0, 60.0, 16.0, 66.0, Material::Ferrofluid);
        },
    },
    MapDef {
        name: "Kitchen",
        blurb: "Chocolate poured onto a cold plate sets; milk, ink and blood sink and spread through water in plumes.",
        width_m: 0.0,
        build: |m| {
            m.rect(4.0, 36.0, 60.0, 40.0);
            m.held_at(4.0);
            m.fill(14.0, 48.0, 40.0, 66.0, Material::Chocolate);
            m.fill(72.0, 0.0, 158.0, 30.0, Material::Water);
            m.rect(66.0, 0.0, 70.0, 34.0);
            m.fill(84.0, 50.0, 92.0, 58.0, Material::Milk);
            m.fill(110.0, 50.0, 118.0, 58.0, Material::Ink);
            m.fill(136.0, 50.0, 144.0, 58.0, Material::Blood);
        },
    },
    MapDef {
        name: "Cold Snap",
        blurb: "Air at -15 °C: the water freezes over and the brine does not; molten wax on a hot plate stays liquid, beside it it sets.",
        width_m: 0.0,
        build: |m| {
            m.air(-15.0);
            m.fill(0.0, 0.0, 48.0, 12.0, Material::Water);
            m.rect(48.0, 0.0, 52.0, 20.0);
            m.fill(52.0, 0.0, 100.0, 12.0, Material::Brine);
            m.rect(100.0, 0.0, 104.0, 20.0);
            m.rect(108.0, 0.0, 130.0, 3.0);
            m.held_at(80.0);
            m.fill(110.0, 3.0, 150.0, 14.0, Material::Wax);
        },
    },
    MapDef {
        name: "Gallium",
        blurb: "Liquid gallium on a tabletop: its oxide skin holds lumpy shapes until a knock breaks it (1 mm cells).",
        width_m: 0.512,
        build: |m| {
            m.air(35.0);
            m.fill(30.0, 0.0, 50.0, 12.0, Material::Gallium);
            m.fill(90.0, 30.0, 104.0, 44.0, Material::Gallium);
            m.rect(70.0, 20.0, 120.0, 22.0);
            m.fill(130.0, 0.0, 140.0, 30.0, Material::Gallium);
        },
    },
];

/// Deterministic jitter so maps start identically on every run.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Fills a rectangle (cell units) with rest-density particles, skipping solids.
pub fn fill_rect(scene: &Scene, rect: [f32; 4], material: Material, out: &mut Vec<Particle>) {
    let sub = (PPC as f32).sqrt().round() as i32;
    let step = 1.0 / sub as f32;
    let (w, h) = (scene.grid[0] as f32, scene.grid[1] as f32);
    let x0 = (rect[0].max(WALL) / step).round() as i32;
    let x1 = (rect[2].min(w - WALL) / step).round() as i32;
    let y0 = (rect[1].max(WALL) / step).round() as i32;
    let y1 = (rect[3].min(h - WALL) / step).round() as i32;
    let mut rng = Rng(0x9E37_79B9 ^ (out.len() as u32).wrapping_mul(747_796_405).wrapping_add(1));
    for j in y0..y1 {
        for i in x0..x1 {
            let pos = [
                (i as f32 + 0.5 + (rng.next() - 0.5) * 0.5) * step,
                (j as f32 + 0.5 + (rng.next() - 0.5) * 0.5) * step,
            ];
            // Keep the simulation's wall margin, and no more: dropping particles further
            // out would leave the liquid thin along every wall.
            if scene.sdf(pos) > 0.1 {
                out.push(Particle::new(material, pos, [0.0, 0.0]));
            }
        }
    }
}

/// What a map sets besides solids and liquid.
#[derive(Clone, Debug, Default)]
pub struct MapExtras {
    pub magnets: Vec<Magnet>,
    /// Air temperature, °C (`None`: the default).
    pub ambient: Option<f32>,
}

/// Instantiates a map on a grid: the static scene plus its initial particles.
#[cfg(test)]
pub fn build(map: &MapDef, grid: [u32; 2]) -> (Scene, Vec<Particle>) {
    let (scene, particles, _) = build_full(map, grid);
    (scene, particles)
}

/// `build`, with the map's magnets and air temperature.
pub fn build_full(map: &MapDef, grid: [u32; 2]) -> (Scene, Vec<Particle>, MapExtras) {
    let scale = grid[0] as f32 / MAP_W;
    let mut b = MapBuilder { scale, shapes: Vec::new(), temps: Vec::new(), fills: Vec::new(), magnets: Vec::new(), ambient: None };
    (map.build)(&mut b);
    let scene = Scene { grid, shapes: b.shapes, temps: b.temps };
    let mut particles = Vec::new();
    for (r, material, t) in b.fills {
        let start = particles.len();
        fill_rect(&scene, [r[0] * scale, r[1] * scale, r[2] * scale, r[3] * scale], material, &mut particles);
        if let Some(t) = t {
            particles[start..].iter_mut().for_each(|p| *p = p.at_temp(t));
        }
    }
    (scene, particles, MapExtras { magnets: b.magnets, ambient: b.ambient })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRIDS: [[u32; 2]; 4] = [[384, 216], [512, 288], [768, 432], [1024, 576]];

    #[test]
    fn every_map_builds_at_every_quality() {
        for map in MAPS {
            for grid in GRIDS {
                let (scene, particles) = build(map, grid);
                assert!(particles.len() <= (grid[0] * grid[1] * PPC) as usize, "{} overflows the particle budget", map.name);
                assert!(particles.iter().all(|p| scene.sdf(p.pos) > 0.0), "{} starts with fluid inside a solid", map.name);
                assert_eq!(particles.is_empty(), map.name == "Empty Tank", "{}", map.name);
                let smallest = scene.rasterize().smallest_region(grid);
                assert!(smallest >= crate::scene::MIN_POCKET, "{} at {grid:?} has a sealed pocket of {smallest} cells", map.name);
            }
        }
    }

    #[test]
    fn maps_are_deterministic_and_scale_with_the_grid() {
        let (_, a) = build(&MAPS[0], [512, 288]);
        let (_, b) = build(&MAPS[0], [512, 288]);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(p, q)| p.pos == q.pos && p.kind == q.kind && p.s == q.s));
        // Twice the resolution holds about four times the particles.
        let (_, fine) = build(&MAPS[0], [1024, 576]);
        let ratio = fine.len() as f32 / a.len() as f32;
        assert!((ratio - 4.0).abs() < 0.2, "ratio {ratio}");
    }

    #[test]
    fn fills_are_rest_density_and_stay_inside_the_walls() {
        let scene = Scene::new([64, 64]);
        let mut out = Vec::new();
        fill_rect(&scene, [0.0, 0.0, 64.0, 23.0], Material::Honey, &mut out);
        // The frame takes WALL cells off each side and the bottom.
        assert_eq!(out.len(), (58 * 20 * PPC) as usize);
        assert!(out.iter().all(|p| p.s == crate::sim::SPAWN_S && p.pos[0] > WALL && p.pos[0] < 64.0 - WALL && p.pos[1] > WALL));
    }
}
