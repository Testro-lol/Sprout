//! Built-in maps.
//!
//! Maps are authored in a resolution-independent space of 160 x 90 "map units" (16:9, origin
//! bottom-left, +y up) and scaled to whatever grid the simulation runs at, so the same map
//! works at every quality setting.

use crate::{
    scene::{Scene, Shape, WALL},
    sim::{Material, PPC, Particle},
};

/// Width of the map space; a 16:9 grid makes it 90 units tall.
pub const MAP_W: f32 = 160.0;

pub struct MapDef {
    pub name: &'static str,
    pub blurb: &'static str,
    build: fn(&mut MapBuilder),
}

pub struct MapBuilder {
    scale: f32,
    shapes: Vec<Shape>,
    fills: Vec<([f32; 4], Material)>,
}

impl MapBuilder {
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
        self.fills.push(([x0, y0, x1, y1], material));
    }
}

pub const MAPS: &[MapDef] = &[
    MapDef {
        name: "Playground",
        blurb: "A pool, a few obstacles, and oil and syrup about to fall in.",
        build: |m| {
            m.fill(0.0, 0.0, 160.0, 22.0, Material::Water);
            m.plank(14.0, 60.0, 70.0, 44.0, 2.0);
            m.circle(102.0, 40.0, 8.0);
            m.rect(126.0, 30.0, 150.0, 33.0);
            m.fill(18.0, 64.0, 44.0, 84.0, Material::Oil);
            m.fill(92.0, 56.0, 112.0, 82.0, Material::Syrup);
        },
    },
    MapDef {
        name: "Dam Break",
        blurb: "A tall column of water collapses across the floor and over a step.",
        build: |m| {
            m.fill(0.0, 0.0, 40.0, 66.0, Material::Water);
            m.rect(84.0, 0.0, 90.0, 12.0);
            m.wedge(128.0, 0.0, 160.0, 22.0, true);
        },
    },
    MapDef {
        name: "Upside Down",
        blurb: "Syrup over water over oil: the layers overturn into their stable order.",
        build: |m| {
            m.fill(0.0, 0.0, 160.0, 20.0, Material::Oil);
            m.fill(0.0, 20.0, 160.0, 40.0, Material::Water);
            m.fill(0.0, 40.0, 160.0, 58.0, Material::Syrup);
        },
    },
    MapDef {
        name: "Three Columns",
        blurb: "Oil, water and syrup side by side: syrup slides under, oil rides over.",
        build: |m| {
            m.fill(0.0, 0.0, 53.0, 58.0, Material::Oil);
            m.fill(53.0, 0.0, 107.0, 58.0, Material::Water);
            m.fill(107.0, 0.0, 160.0, 58.0, Material::Syrup);
        },
    },
    MapDef {
        name: "Cascade",
        blurb: "Three reservoirs drain down a zig-zag of ramps into a shared basin.",
        build: |m| {
            m.plank(0.0, 70.0, 96.0, 60.0, 2.0);
            m.plank(160.0, 50.0, 62.0, 40.0, 2.0);
            m.plank(0.0, 30.0, 100.0, 20.0, 2.0);
            m.circle(118.0, 8.0, 4.0);
            m.circle(133.0, 15.0, 4.0);
            m.circle(147.0, 7.0, 4.0);
            m.fill(2.0, 73.0, 36.0, 88.0, Material::Water);
            m.fill(128.0, 54.0, 158.0, 76.0, Material::Syrup);
            m.fill(2.0, 33.0, 26.0, 46.0, Material::Oil);
        },
    },
    MapDef {
        name: "Empty Tank",
        blurb: "Nothing but walls. Build your own.",
        build: |_| {},
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

/// Instantiates a map on a grid: the static scene plus its initial particles.
pub fn build(map: &MapDef, grid: [u32; 2]) -> (Scene, Vec<Particle>) {
    let scale = grid[0] as f32 / MAP_W;
    let mut b = MapBuilder { scale, shapes: Vec::new(), fills: Vec::new() };
    (map.build)(&mut b);
    let scene = Scene { grid, shapes: b.shapes };
    let mut particles = Vec::new();
    for (r, material) in b.fills {
        fill_rect(&scene, [r[0] * scale, r[1] * scale, r[2] * scale, r[3] * scale], material, &mut particles);
    }
    (scene, particles)
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
        fill_rect(&scene, [0.0, 0.0, 64.0, 23.0], Material::Syrup, &mut out);
        // The frame takes WALL cells off each side and the bottom.
        assert_eq!(out.len(), (58 * 20 * PPC) as usize);
        assert!(out.iter().all(|p| p.s == 1.0 && p.pos[0] > WALL && p.pos[0] < 64.0 - WALL && p.pos[1] > WALL));
    }
}
