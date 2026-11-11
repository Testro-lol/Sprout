//! Static scene geometry: analytic shapes, and their rasterisation into the fields the
//! simulation and renderer consume.
//!
//! Everything here is in grid-cell units with +y up.

/// Thickness of the solid frame around the domain, in cells. The particle kernels reach
/// 1.5 cells, so this keeps every stencil inside the grid.
pub const WALL: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Circle { c: [f32; 2], r: f32 },
    /// Box with half-extents `half`, rotated by `angle` radians.
    Box { c: [f32; 2], half: [f32; 2], angle: f32 },
    /// Right-triangle wedge inscribed in the box `half`, rotated by `angle`. The slope
    /// descends to the right, or to the left when `flip` is set.
    Ramp { c: [f32; 2], half: [f32; 2], angle: f32, flip: bool },
}

fn rotate(p: [f32; 2], angle: f32) -> [f32; 2] {
    let (s, c) = angle.sin_cos();
    [c * p[0] + s * p[1], -s * p[0] + c * p[1]]
}

fn sd_segment2(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (pa, ba) = ([p[0] - a[0], p[1] - a[1]], [b[0] - a[0], b[1] - a[1]]);
    let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / (ba[0] * ba[0] + ba[1] * ba[1])).clamp(0.0, 1.0);
    let d = [pa[0] - ba[0] * h, pa[1] - ba[1] * h];
    d[0] * d[0] + d[1] * d[1]
}

fn sd_triangle(p: [f32; 2], t: [[f32; 2]; 3]) -> f32 {
    let mut d2 = f32::MAX;
    let mut inside = true;
    // Orientation of the triangle decides which side of each edge is "in".
    let area = (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0]);
    for k in 0..3 {
        let (a, b) = (t[k], t[(k + 1) % 3]);
        d2 = d2.min(sd_segment2(p, a, b));
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if cross * area < 0.0 {
            inside = false;
        }
    }
    if inside { -d2.sqrt() } else { d2.sqrt() }
}

impl Shape {
    /// Signed distance to the shape: negative inside.
    pub fn sdf(&self, p: [f32; 2]) -> f32 {
        match *self {
            Shape::Circle { c, r } => ((p[0] - c[0]).hypot(p[1] - c[1])) - r,
            Shape::Box { c, half, angle } => {
                let q = rotate([p[0] - c[0], p[1] - c[1]], angle);
                let d = [q[0].abs() - half[0], q[1].abs() - half[1]];
                d[0].max(0.0).hypot(d[1].max(0.0)) + d[0].max(d[1]).min(0.0)
            }
            Shape::Ramp { c, half, angle, flip } => {
                let q = rotate([p[0] - c[0], p[1] - c[1]], angle);
                let top = if flip { half[0] } else { -half[0] };
                sd_triangle(q, [[-half[0], -half[1]], [half[0], -half[1]], [top, half[1]]])
            }
        }
    }

    /// Radius of a circle around the centre that contains the whole shape.
    pub fn bound(&self) -> ([f32; 2], f32) {
        match *self {
            Shape::Circle { c, r } => (c, r),
            Shape::Box { c, half, .. } | Shape::Ramp { c, half, .. } => (c, half[0].hypot(half[1])),
        }
    }
}

/// Rasterised solids, ready to upload.
pub struct SolidField {
    /// Signed distance at grid nodes, `(W + 1) * (H + 1)` values.
    pub sdf: Vec<f32>,
    /// Per cell: open fraction of the left face, open fraction of the bottom face,
    /// kernel-smoothed solid occupancy, open volume fraction.
    pub solid: Vec<[f32; 4]>,
}

#[cfg(test)]
impl SolidField {
    /// Size of the smallest group of open cells connected through open faces.
    pub fn smallest_region(&self, grid: [u32; 2]) -> usize {
        let (w, h) = (grid[0] as usize, grid[1] as usize);
        let mut seen = vec![false; w * h];
        let mut smallest = usize::MAX;
        for start in 0..w * h {
            if self.solid[start][3] == 0.0 || seen[start] {
                continue;
            }
            let (mut stack, mut size) = (vec![start], 0);
            seen[start] = true;
            while let Some(k) = stack.pop() {
                size += 1;
                let (i, j) = (k % w, k / w);
                let links = [
                    (i > 0, k.wrapping_sub(1), self.solid[k][0]),
                    (i + 1 < w, k + 1, if i + 1 < w { self.solid[k + 1][0] } else { 0.0 }),
                    (j > 0, k.wrapping_sub(w), self.solid[k][1]),
                    (j + 1 < h, k + w, if j + 1 < h { self.solid[k + w][1] } else { 0.0 }),
                ];
                for (ok, n, a) in links {
                    if ok && a > 0.0 && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
            smallest = smallest.min(size);
        }
        smallest
    }
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub grid: [u32; 2],
    pub shapes: Vec<Shape>,
}

/// Cells with less open volume than this are treated as solid.
const MIN_OPEN: f32 = 0.1;
/// Enclosed groups of open cells smaller than this are filled in (see `seal_pockets`).
pub(crate) const MIN_POCKET: usize = 16;

/// Fraction of the segment between two SDF samples that lies outside the solid.
fn open_fraction(d0: f32, d1: f32) -> f32 {
    let f = if d0 > 0.0 && d1 > 0.0 {
        1.0
    } else if d0 <= 0.0 && d1 <= 0.0 {
        0.0
    } else {
        d0.max(d1) / (d0.max(d1) - d0.min(d1))
    };
    // Slivers make the pressure system stiff for no visible gain.
    if f < 0.05 { 0.0 } else { f }
}

impl Scene {
    pub fn new(grid: [u32; 2]) -> Self {
        Scene { grid, shapes: Vec::new() }
    }

    fn wall_sdf(&self, p: [f32; 2]) -> f32 {
        let (w, h) = (self.grid[0] as f32, self.grid[1] as f32);
        (p[0] - WALL).min(w - WALL - p[0]).min(p[1] - WALL).min(h - WALL - p[1])
    }

    /// Signed distance to the nearest solid (walls included); negative inside solids.
    pub fn sdf(&self, p: [f32; 2]) -> f32 {
        self.shapes.iter().fold(self.wall_sdf(p), |d, s| d.min(s.sdf(p)))
    }

    /// Index of the topmost shape under `p`, if any.
    pub fn pick(&self, p: [f32; 2]) -> Option<usize> {
        self.shapes.iter().rposition(|s| s.sdf(p) < 0.5)
    }

    pub fn rasterize(&self) -> SolidField {
        let (w, h) = (self.grid[0] as usize, self.grid[1] as usize);
        let (nw, nh) = (w + 1, h + 1);

        // Node distances. Shapes only touch nodes near their bounding circle; far from a
        // shape only the sign and rough magnitude of the distance matter.
        let mut sdf = vec![0.0f32; nw * nh];
        for j in 0..nh {
            for i in 0..nw {
                sdf[j * nw + i] = self.wall_sdf([i as f32, j as f32]);
            }
        }
        const REACH: f32 = 6.0;
        for s in &self.shapes {
            let (c, r) = s.bound();
            let r = r + REACH;
            let i0 = (c[0] - r).floor().max(0.0) as usize;
            let i1 = ((c[0] + r).ceil() as isize).clamp(0, nw as isize - 1) as usize;
            let j0 = (c[1] - r).floor().max(0.0) as usize;
            let j1 = ((c[1] + r).ceil() as isize).clamp(0, nh as isize - 1) as usize;
            for j in j0..=j1.max(j0).min(nh - 1) {
                for i in i0..=i1.max(i0).min(nw - 1) {
                    let d = &mut sdf[j * nw + i];
                    *d = d.min(s.sdf([i as f32, j as f32]));
                }
            }
        }

        // Open volume per cell from four bilinear sub-samples.
        let mut open = vec![0.0f32; w * h];
        for j in 0..h {
            for i in 0..w {
                let d = [sdf[j * nw + i], sdf[j * nw + i + 1], sdf[(j + 1) * nw + i], sdf[(j + 1) * nw + i + 1]];
                let mut acc = 0.0;
                for (fx, fy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    let v = (d[0] * (1.0 - fx) + d[1] * fx) * (1.0 - fy) + (d[2] * (1.0 - fx) + d[3] * fx) * fy;
                    acc += (v / 0.5 + 0.5f32).clamp(0.0, 1.0);
                }
                // A cell that is all but full of solid cannot hold liquid. Count it as solid
                // outright, so that it is also sealed off below.
                open[j * w + i] = if acc * 0.25 < MIN_OPEN { 0.0 } else { acc * 0.25 };
            }
        }
        // A face is only open between two cells that can both hold liquid. A face leading
        // into a solid cell would look like a free surface to the pressure solve and
        // swallow fluid.
        let faces = |open: &[f32], sdf: &[f32], i: usize, j: usize| -> [f32; 2] {
            let here = open[j * w + i] > 0.0;
            let left = here && i > 0 && open[j * w + i - 1] > 0.0;
            let below = here && j > 0 && open[(j - 1) * w + i] > 0.0;
            [
                if left { open_fraction(sdf[j * nw + i], sdf[(j + 1) * nw + i]) } else { 0.0 },
                if below { open_fraction(sdf[j * nw + i], sdf[j * nw + i + 1]) } else { 0.0 },
            ]
        };
        Self::seal_pockets(&mut open, &mut sdf, [w, h], |open, sdf, i, j| faces(open, sdf, i, j));

        // Kernel-smoothed solid occupancy: what the particle scatter would report if the
        // solid were filled with rest-density liquid. Adding it to the liquid occupancy
        // stops cells next to walls from looking under-full.
        const K: [f32; 3] = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
        let solid_at = |i: isize, j: isize| -> f32 {
            if i < 0 || j < 0 || i >= w as isize || j >= h as isize { 1.0 } else { 1.0 - open[j as usize * w + i as usize] }
        };
        let mut solid = vec![[0.0f32; 4]; w * h];
        for j in 0..h {
            for i in 0..w {
                let mut occ = 0.0;
                for (b, kb) in K.iter().enumerate() {
                    for (a, ka) in K.iter().enumerate() {
                        occ += ka * kb * solid_at(i as isize + a as isize - 1, j as isize + b as isize - 1);
                    }
                }
                let [au, av] = faces(&open, &sdf, i, j);
                solid[j * w + i] = [au, av, occ, open[j * w + i]];
            }
        }
        SolidField { sdf, solid }
    }

    /// Turns enclosed pockets into solid: groups of open cells, connected through open
    /// faces, that are too small to hold liquid with a free surface. Two touching shapes
    /// leave such slivers in the crack between them (a cell or two, sealed on every side).
    /// Liquid that leaks in through the gap in the distance field has nowhere to go: its
    /// pressure and density-correction system is singular, the potential grows without
    /// bound, and once it overflows the whole tank loses its volume correction. The node
    /// distances of a sealed cell are pushed inside the solid as well, so particles are
    /// expelled from it and the renderer draws it filled.
    fn seal_pockets(open: &mut [f32], sdf: &mut [f32], [w, h]: [usize; 2], faces: impl Fn(&[f32], &[f32], usize, usize) -> [f32; 2]) {
        let nw = w + 1;
        let mut label = vec![u32::MAX; w * h];
        let mut stack = Vec::new();
        let mut members = Vec::new();
        for start in 0..w * h {
            if open[start] == 0.0 || label[start] != u32::MAX {
                continue;
            }
            members.clear();
            label[start] = start as u32;
            stack.push(start);
            while let Some(k) = stack.pop() {
                members.push(k);
                let (i, j) = (k % w, k / w);
                let [left, below] = faces(open, sdf, i, j);
                let right = if i + 1 < w { faces(open, sdf, i + 1, j)[0] } else { 0.0 };
                let above = if j + 1 < h { faces(open, sdf, i, j + 1)[1] } else { 0.0 };
                for (n, a) in [(k.wrapping_sub(1), left), (k + 1, right), (k.wrapping_sub(w), below), (k + w, above)] {
                    if a > 0.0 && label[n] == u32::MAX {
                        label[n] = start as u32;
                        stack.push(n);
                    }
                }
            }
            if members.len() < MIN_POCKET {
                for &k in &members {
                    open[k] = 0.0;
                    let (i, j) = (k % w, k / w);
                    for node in [j * nw + i, j * nw + i + 1, (j + 1) * nw + i, (j + 1) * nw + i + 1] {
                        sdf[node] = sdf[node].min(-0.5);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn busy_scene() -> Scene {
        let mut scene = Scene::new([256, 144]);
        // A circle whose extreme points land exactly on grid nodes, a rotated box and a ramp.
        scene.shapes.push(Shape::Circle { c: [128.0, 40.0], r: 18.0 });
        scene.shapes.push(Shape::Box { c: [60.0, 30.0], half: [25.0, 6.0], angle: 0.5 });
        scene.shapes.push(Shape::Ramp { c: [200.0, 25.0], half: [30.0, 15.0], angle: 0.0, flip: false });
        scene
    }

    #[test]
    fn shape_distances_have_the_right_sign() {
        let circle = Shape::Circle { c: [10.0, 10.0], r: 4.0 };
        assert!((circle.sdf([10.0, 10.0]) + 4.0).abs() < 1e-5);
        assert!((circle.sdf([16.0, 10.0]) - 2.0).abs() < 1e-5);

        let plank = Shape::Box { c: [0.0, 0.0], half: [10.0, 1.0], angle: std::f32::consts::FRAC_PI_2 };
        assert!(plank.sdf([0.0, 9.0]) < 0.0, "a quarter turn stands the plank upright");
        assert!(plank.sdf([9.0, 0.0]) > 0.0);

        // Tall side on the left unless flipped.
        let ramp = Shape::Ramp { c: [0.0, 0.0], half: [10.0, 5.0], angle: 0.0, flip: false };
        assert!(ramp.sdf([-8.0, 2.0]) < 0.0);
        assert!(ramp.sdf([8.0, 2.0]) > 0.0);
        let flipped = Shape::Ramp { c: [0.0, 0.0], half: [10.0, 5.0], angle: 0.0, flip: true };
        assert!(flipped.sdf([8.0, 2.0]) < 0.0);
        assert!(flipped.sdf([-8.0, 2.0]) > 0.0);
    }

    #[test]
    fn open_fraction_follows_the_crossing() {
        assert_eq!(open_fraction(1.0, 2.0), 1.0);
        assert_eq!(open_fraction(-1.0, -2.0), 0.0);
        assert!((open_fraction(-1.0, 3.0) - 0.75).abs() < 1e-6);
        assert!((open_fraction(3.0, -1.0) - 0.75).abs() < 1e-6);
        assert_eq!(open_fraction(-1.0, 0.01), 0.0, "slivers are closed");
    }

    #[test]
    fn picking_takes_the_topmost_shape() {
        let mut scene = busy_scene();
        scene.shapes.push(Shape::Circle { c: [128.0, 40.0], r: 5.0 });
        assert_eq!(scene.pick([128.0, 40.0]), Some(3));
        assert_eq!(scene.pick([128.0, 55.0]), Some(0));
        assert_eq!(scene.pick([128.0, 100.0]), None);
    }

    /// The pressure solve treats a cell without liquid as a free surface, so an open face
    /// into a cell that can never hold liquid would swallow fluid.
    #[test]
    fn open_faces_never_lead_into_solid_cells() {
        let scene = busy_scene();
        let field = scene.rasterize();
        let (w, h) = (scene.grid[0] as usize, scene.grid[1] as usize);
        let at = |i: usize, j: usize| field.solid[j * w + i];
        let mut cut_cells = 0;
        for j in 1..h {
            for i in 1..w {
                let [left, bottom, occupancy, open] = at(i, j);
                assert!((0.0..=1.0).contains(&left) && (0.0..=1.0).contains(&bottom));
                assert!((0.0..=1.0 + 1e-5).contains(&occupancy) && (0.0..=1.0).contains(&open));
                assert!(open == 0.0 || open >= MIN_OPEN);
                if left > 0.0 {
                    assert!(open > 0.0 && at(i - 1, j)[3] > 0.0, "left face of ({i}, {j})");
                }
                if bottom > 0.0 {
                    assert!(open > 0.0 && at(i, j - 1)[3] > 0.0, "bottom face of ({i}, {j})");
                }
                cut_cells += (open > 0.0 && open < 1.0) as u32;
            }
        }
        assert!(cut_cells > 100, "the shapes should cut through many cells");
    }

    /// Two ramps meeting at their tips leave a sealed crack; it must be filled in, or the
    /// liquid that leaks into it makes the pressure system singular.
    #[test]
    fn cracks_between_touching_shapes_are_filled() {
        let mut scene = Scene::new([512, 288]);
        scene.shapes.push(Shape::Ramp { c: [218.4, 89.6], half: [36.8, 16.0], angle: 0.0, flip: true });
        scene.shapes.push(Shape::Ramp { c: [292.0, 89.6], half: [36.8, 16.0], angle: 0.0, flip: false });
        let field = scene.rasterize();
        assert!(field.smallest_region(scene.grid) >= MIN_POCKET);
        // A large enclosed region (a sealed box) is left alone.
        let mut boxed = Scene::new([128, 128]);
        for (c, half) in [([64.0, 40.0], [30.0, 2.0]), ([64.0, 80.0], [30.0, 2.0]), ([36.0, 60.0], [2.0, 22.0]), ([92.0, 60.0], [2.0, 22.0])] {
            boxed.shapes.push(Shape::Box { c, half, angle: 0.0 });
        }
        let field = boxed.rasterize();
        assert_eq!(field.solid[60 * 128 + 64][3], 1.0, "the inside of a sealed box stays open");
    }

    #[test]
    fn the_tank_frame_is_solid_and_the_inside_open() {
        let scene = Scene::new([64, 64]);
        let field = scene.rasterize();
        let at = |i: usize, j: usize| field.solid[j * 64 + i];
        assert_eq!(at(1, 30)[3], 0.0);
        assert_eq!(at(30, 62)[3], 0.0);
        assert_eq!(at(3, 30)[0], 0.0, "the face on the wall is closed");
        assert_eq!(at(30, 3)[1], 0.0, "the face on the floor is closed");
        assert_eq!(at(30, 30), [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(field.sdf.len(), 65 * 65);
        assert!(scene.sdf([32.0, 32.0]) > 0.0 && scene.sdf([1.0, 32.0]) < 0.0);
    }
}
