//! Brush geometry and the admission of new liquid. CPU only: no solver or GPU dependencies.

/// Slowest and fastest stream of the faucet (Pour), cells/s. Slower than the minimum and
/// gravity stretches the stream to many times its length on the way down, which tears it
/// into ribbons.
pub const FAUCET_SPEED: [f32; 2] = [150.0, 400.0];

/// Speed of the faucet's stream for a flow of `rate` particles/s from a brush of `radius`:
/// as slow as the flow allows through the full width of the brush. Above the fastest
/// speed the brush caps the flow.
pub fn faucet_speed(rate: f32, radius: f32) -> f32 {
    (rate / (crate::sim::PPC as f32 * 2.0 * radius)).clamp(FAUCET_SPEED[0], FAUCET_SPEED[1])
}

/// Launch speed of the jet, cells/s.
pub const JET_SPEED: f32 = 320.0;
/// Launch speed of spray droplets, cells/s.
pub const SPRAY_SPEED: f32 = 260.0;
/// Spray emits this share of the pour rate.
pub const SPRAY_SHARE: f32 = 0.15;
/// Spacing of spray droplets, cells: sparse from birth, rather than a dense fill whose
/// opposing velocities cancel in the particle-to-grid transfer.
pub const SPRAY_SPACING: f32 = 1.5;

/// Lattice points (spaced `spacing` apart, aligned to the grid) inside a swept disk (a
/// capsule) from `from` to `to`. The same points come out however the stroke is cut into
/// frames, and the end caps are included even for a pointer that does not move.
pub fn stroke(from: [f32; 2], to: [f32; 2], radius: f32, grid: [u32; 2], spacing: f32) -> Vec<[f32; 2]> {
    let delta = [to[0] - from[0], to[1] - from[1]];
    let length2 = delta[0] * delta[0] + delta[1] * delta[1];
    let index = |v: f32, limit: u32, round: fn(f32) -> f32| (round(v.clamp(0.0, limit as f32) / spacing)) as i32;
    let lo = [index(from[0].min(to[0]) - radius, grid[0], f32::floor), index(from[1].min(to[1]) - radius, grid[1], f32::floor)];
    let hi = [index(from[0].max(to[0]) + radius, grid[0], f32::ceil), index(from[1].max(to[1]) + radius, grid[1], f32::ceil)];
    let mut out = Vec::new();
    for y in lo[1]..hi[1] {
        for x in lo[0]..hi[0] {
            let p = [(x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing];
            let d = offset_from_stroke(p, from, to, length2);
            if d[0] * d[0] + d[1] * d[1] <= radius * radius {
                out.push(p);
            }
        }
    }
    out
}

/// Offset of `p` from the nearest point of the segment `from`-`to` (`length2` its squared
/// length).
fn offset_from_stroke(p: [f32; 2], from: [f32; 2], to: [f32; 2], length2: f32) -> [f32; 2] {
    let delta = [to[0] - from[0], to[1] - from[1]];
    let t = if length2 > 0.0 {
        (((p[0] - from[0]) * delta[0] + (p[1] - from[1]) * delta[1]) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    [p[0] - from[0] - t * delta[0], p[1] - from[1] - t * delta[1]]
}

/// Unit vector of a jet aimed `degrees` clockwise from straight down.
pub fn jet_direction(degrees: f32) -> [f32; 2] {
    let (s, c) = degrees.to_radians().sin_cos();
    [s, -c]
}

/// Launch velocity of a spray droplet at `p` on a stroke: outwards from the stroke at
/// `SPRAY_SPEED`, and only from the outer rim. A dense centre would cancel opposite
/// velocities on the grid and turn the spray back into a blob.
pub fn spray_velocity(p: [f32; 2], from: [f32; 2], to: [f32; 2], radius: f32) -> Option<[f32; 2]> {
    let length2 = (to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2);
    let offset = offset_from_stroke(p, from, to, length2);
    let distance = offset[0].hypot(offset[1]);
    if distance < radius * 0.7 || distance < 1e-4 {
        return None;
    }
    Some([offset[0] / distance * SPRAY_SPEED, offset[1] / distance * SPRAY_SPEED])
}

/// Which cells of a window may still receive new liquid, so that a brush never packs a
/// cell above rest density and adds nothing inside a pool.
///
/// For painted liquid (spray, drop) a cell that holds any liquid already counts as full: the
/// lattice points of a paint stroke fall anywhere in a cell, and a new particle on top of
/// an existing one would stay stuck to it (both sample the same grid values). A stream
/// (`top_up`) lays its rows exactly where the previous rows have moved on from, so a cell
/// it shares with the tail of the last frame may be topped up to rest density instead;
/// otherwise every frame would leave a gap in the stream.
pub struct Admission {
    lo: [i32; 2],
    size: [i32; 2],
    room: Vec<u32>,
}

impl Admission {
    /// `counts` holds the live particles of each cell in the window starting at cell `lo`,
    /// `size` cells across, row by row; `ppc` is the rest density.
    pub fn new(lo: [i32; 2], size: [i32; 2], counts: &[u32], ppc: u32, top_up: bool) -> Self {
        let room = |n: u32| if top_up { ppc.saturating_sub(n) } else if n == 0 { ppc } else { 0 };
        Admission { lo, size, room: counts.iter().map(|&n| room(n)).collect() }
    }

    /// The window of cells covering everything within `margin` of the segment `from`-`to`.
    pub fn window(from: [f32; 2], to: [f32; 2], margin: f32) -> ([i32; 2], [i32; 2]) {
        let lo = [(from[0].min(to[0]) - margin).floor() as i32, (from[1].min(to[1]) - margin).floor() as i32];
        let hi = [(from[0].max(to[0]) + margin).ceil() as i32 + 1, (from[1].max(to[1]) + margin).ceil() as i32 + 1];
        (lo, hi)
    }

    /// Takes one particle's worth of room at `pos`, if its cell has any left.
    pub fn admit(&mut self, pos: [f32; 2]) -> bool {
        let c = [pos[0].floor() as i32 - self.lo[0], pos[1].floor() as i32 - self.lo[1]];
        if c[0] < 0 || c[1] < 0 || c[0] >= self.size[0] || c[1] >= self.size[1] {
            return false;
        }
        let room = &mut self.room[(c[1] * self.size[0] + c[0]) as usize];
        if *room == 0 {
            return false;
        }
        *room -= 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stroke_is_the_same_however_it_is_cut_into_frames() {
        let whole = stroke([10.0, 10.0], [80.0, 10.0], 2.0, [100, 30], 0.5);
        let mut parts = stroke([10.0, 10.0], [45.0, 10.0], 2.0, [100, 30], 0.5);
        parts.extend(stroke([45.0, 10.0], [80.0, 10.0], 2.0, [100, 30], 0.5));
        assert!(whole.iter().all(|p| parts.contains(p)));
        assert!(parts.iter().all(|p| whole.contains(p)));
        // A pointer that does not move still paints a full disk.
        let dot = stroke([20.0, 20.0], [20.0, 20.0], 3.0, [100, 100], 0.5);
        let area = std::f32::consts::PI * 9.0 * 4.0;
        assert!((dot.len() as f32 - area).abs() < 0.1 * area, "{} points", dot.len());
    }

    #[test]
    fn admission_fills_only_empty_cells_up_to_rest_density() {
        // A 2 x 1 window: the left cell holds a particle, the right one is empty.
        let mut admission = Admission::new([4, 7], [2, 1], &[1, 0], 4, false);
        assert!(!admission.admit([4.5, 7.5]), "a cell with liquid is full");
        for p in [[5.25, 7.25], [5.75, 7.25], [5.25, 7.75], [5.75, 7.75]] {
            assert!(admission.admit(p));
        }
        assert!(!admission.admit([5.5, 7.5]), "rest density reached");
        assert!(!admission.admit([9.5, 7.5]), "outside the window");
        // A stream tops a cell up to rest density.
        let mut stream = Admission::new([0, 0], [1, 1], &[3], 4, true);
        assert!(stream.admit([0.5, 0.5]));
        assert!(!stream.admit([0.5, 0.5]));
    }

    #[test]
    fn faucet_speed_follows_the_flow_within_its_limits() {
        assert_eq!(faucet_speed(1000.0, 9.0), FAUCET_SPEED[0]);
        assert!((faucet_speed(20_000.0, 9.0) - 20_000.0 / 72.0).abs() < 1e-3);
        assert_eq!(faucet_speed(1e6, 9.0), FAUCET_SPEED[1]);
    }

    #[test]
    fn jet_aim_covers_a_full_turn() {
        for (angle, expected) in [(0.0, [0.0, -1.0]), (90.0, [1.0, 0.0]), (180.0, [0.0, 1.0]), (270.0, [-1.0, 0.0])] {
            let d = jet_direction(angle);
            assert!((d[0] - expected[0]).abs() < 1e-5 && (d[1] - expected[1]).abs() < 1e-5);
        }
    }

    #[test]
    fn spray_has_no_dense_centre_and_keeps_its_launch_speed() {
        let c = [20.0, 20.0];
        assert!(spray_velocity(c, c, c, 9.0).is_none());
        assert!(spray_velocity([23.0, 20.0], c, c, 9.0).is_none());
        for offset in [[8.0, 0.0], [-8.0, 0.0], [0.0, 8.0], [5.0, -5.0]] {
            let v = spray_velocity([c[0] + offset[0], c[1] + offset[1]], c, c, 9.0).unwrap();
            assert!((v[0].hypot(v[1]) - SPRAY_SPEED).abs() < 0.001);
            assert!(v[0] * offset[0] + v[1] * offset[1] > 0.0);
        }
        // Along a moving stroke the spray leaves sideways, not along the stroke.
        let v = spray_velocity([30.0, 28.0], [20.0, 20.0], [40.0, 20.0], 9.0).unwrap();
        assert!(v[0].abs() < 1e-3 && v[1] > 0.0);
    }
}
