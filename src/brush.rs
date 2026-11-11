//! Brush geometry and the admission of new liquid. CPU only: no solver or GPU dependencies.

/// Slowest and fastest the pour leaves its spout, cells/s. The flow picks the speed in
/// between that fills the brush's full width.
pub const POUR_SPEED: [f32; 2] = [60.0, 400.0];
/// Launch speed of the jet, cells/s.
pub const JET_SPEED: f32 = 320.0;
/// A stream may fill a cell this far above rest density. A lattice at rest density that is
/// not aligned to the grid puts one particle more or less in some cells, so a free stream
/// has to be let through; a blocked or submerged spout is still held back.
pub const STREAM_FILL: f32 = 1.5;

/// The outlet of a stream (the pour, the jet): a round nozzle centred on the brush,
/// `width` cells across, that liquid leaves at rest density at `speed` cells/s along
/// `direction`. The liquid enters through the back half of the circle, so the stream has a
/// round top that fills the brush.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spout {
    pub direction: [f32; 2],
    pub speed: f32,
    pub width: f32,
}

impl Spout {
    /// The pour: straight down through the full width of the brush, at the speed that
    /// carries `rate` particles/s. Where that speed would be below `POUR_SPEED` the stream
    /// narrows instead, and above it the brush caps the flow.
    pub fn pour(rate: f32, radius: f32) -> Spout {
        let speed = (rate / (PPC * 2.0 * radius)).clamp(POUR_SPEED[0], POUR_SPEED[1]);
        Spout::new([0.0, -1.0], speed, rate, radius)
    }

    /// The jet: `JET_SPEED` along `degrees` clockwise from straight down; the flow sets
    /// its width, up to the brush's.
    pub fn jet(degrees: f32, rate: f32, radius: f32) -> Spout {
        Spout::new(jet_direction(degrees), JET_SPEED, rate, radius)
    }

    fn new(direction: [f32; 2], speed: f32, rate: f32, radius: f32) -> Spout {
        Spout { direction, speed, width: (rate / (PPC * speed)).min(2.0 * radius) }
    }

    /// Particles per second the spout delivers.
    pub fn rate(&self) -> f32 {
        PPC * self.speed * self.width
    }

    pub fn velocity(&self) -> [f32; 2] {
        [self.direction[0] * self.speed, self.direction[1] * self.speed]
    }
}

const PPC: f32 = crate::sim::PPC as f32;
/// 1 / golden ratio: the step across the stream from one particle to the next.
const GOLDEN: f64 = 0.618_033_988_749_894_9;

/// A stream being emitted: how long the brush has been held.
///
/// The stream is a Fibonacci lattice: particle k belongs `(k + 1/2) / rate` seconds along
/// it and at the fraction `frac(k / golden ratio)` of the way across. That packs it at rest
/// density exactly, with neighbours never closer than 0.94 of the mean spacing whatever the
/// speed and width. The lattice moves along at the stream's speed, and a particle appears
/// when it crosses the back arc of the round nozzle; so the rim comes out a little later
/// than the middle, and the stream gets its round top without the lattice being bent.
/// Each frame emits the particles that appeared during it, where they have got to since
/// (the brush sweeps from `from` to `to` over the frame). So the stream is the same however
/// time is cut into frames, and the flow is exact.
///
/// Only a sideways sweep of the brush distorts it: the leading flank of the nozzle then
/// overruns liquid it has just emitted (`Admission` holds the surplus back) and the trailing
/// flank leaves the stream thinner.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stream {
    /// Simulated time the stream has run, s.
    time: f64,
}

impl Stream {
    /// Advances the stream by `seconds` and returns where this frame's particles are at
    /// its end, at most `limit` of them (the latest; a longer backlog is dropped).
    pub fn advance(&mut self, spout: &Spout, from: [f32; 2], to: [f32; 2], seconds: f32, limit: usize) -> Vec<[f32; 2]> {
        let rate = spout.rate() as f64;
        let (start, end) = (self.time, self.time + seconds as f64);
        self.time = end;
        let half = 0.5 * spout.width;
        let side = [-spout.direction[1], spout.direction[0]];
        // Particle k is on the lattice from (k + 1/2) / rate on, and appears up to the time
        // the stream takes to cross half the nozzle later.
        let index = |time: f64| ((rate * time - 0.5).floor() + 1.0).max(0.0) as u64;
        let mut out: Vec<[f32; 2]> = (index(start - (half / spout.speed) as f64)..index(end))
            .filter_map(|k| {
                let across = ((k as f64 * GOLDEN).fract() as f32 - 0.5) * spout.width;
                let arc = (half * half - across * across).max(0.0).sqrt();
                let appears = (k as f64 + 0.5) / rate + ((half - arc) / spout.speed) as f64;
                if appears < start || appears >= end {
                    return None;
                }
                let t = ((appears - start) / (end - start)) as f32;
                let along = (end - appears) as f32 * spout.speed - arc;
                Some([
                    from[0] + (to[0] - from[0]) * t + side[0] * across + spout.direction[0] * along,
                    from[1] + (to[1] - from[1]) * t + side[1] * across + spout.direction[1] * along,
                ])
            })
            .collect();
        out.drain(..out.len().saturating_sub(limit));
        out
    }
}

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
/// arrives where its own earlier particles have just moved on from, so the cells it shares
/// with them fill up to `STREAM_FILL` times rest density instead.
pub struct Admission {
    lo: [i32; 2],
    size: [i32; 2],
    room: Vec<u32>,
}

impl Admission {
    /// `counts` holds the live particles of each cell in the window starting at cell `lo`,
    /// `size` cells across, row by row; `ppc` is the rest density. With `stream` cells fill
    /// up to `STREAM_FILL` times that, otherwise only empty cells take liquid.
    pub fn new(lo: [i32; 2], size: [i32; 2], counts: &[u32], ppc: u32, stream: bool) -> Self {
        let full = (ppc as f32 * STREAM_FILL) as u32;
        let room = |n: u32| if stream { full.saturating_sub(n) } else if n == 0 { ppc } else { 0 };
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
        // A stream fills a cell to 1.5 times rest density.
        let mut stream = Admission::new([0, 0], [1, 1], &[4], 4, true);
        assert!(stream.admit([0.5, 0.5]) && stream.admit([0.5, 0.5]));
        assert!(!stream.admit([0.5, 0.5]));
    }

    #[test]
    fn the_pour_fills_the_brush_at_the_speed_the_flow_needs() {
        // 12000 particles/s through an 18 cell spout at 4 per cell: 167 cells/s.
        let spout = Spout::pour(12_000.0, 9.0);
        assert!((spout.width - 18.0).abs() < 1e-4 && (spout.speed - 12_000.0 / 72.0).abs() < 1e-3);
        assert!((spout.rate() - 12_000.0).abs() < 1.0);
        // A trickle narrows at the slowest speed; a flood is capped at the fastest.
        let trickle = Spout::pour(1000.0, 9.0);
        assert!(trickle.speed == POUR_SPEED[0] && trickle.width < 5.0 && (trickle.rate() - 1000.0).abs() < 1.0);
        let flood = Spout::pour(1e6, 9.0);
        assert!(flood.speed == POUR_SPEED[1] && flood.width == 18.0 && flood.rate() < 30_000.0);
        let jet = Spout::jet(90.0, 12_000.0, 9.0);
        assert!((jet.velocity()[0] - JET_SPEED).abs() < 1e-3 && (jet.rate() - 12_000.0).abs() < 1.0);
    }

    /// Runs a stream for `frames` frames of `dt` from a spout moving at `sweep` cells/s,
    /// carrying every particle along at the spout's velocity.
    fn run_stream(spout: &Spout, frames: usize, dt: f32, sweep: f32) -> Vec<[f32; 2]> {
        let (mut stream, mut all, v) = (Stream::default(), Vec::<[f32; 2]>::new(), spout.velocity());
        for f in 0..frames {
            all.iter_mut().for_each(|p| *p = [p[0] + v[0] * dt, p[1] + v[1] * dt]);
            let (from, to) = ([100.0 + sweep * dt * f as f32, 100.0], [100.0 + sweep * dt * (f + 1) as f32, 100.0]);
            all.extend(stream.advance(spout, from, to, dt, usize::MAX));
        }
        all
    }

    #[test]
    fn a_stream_is_the_same_however_time_is_cut_into_frames() {
        let spout = Spout::pour(12_000.0, 9.0);
        for sweep in [0.0, 300.0] {
            let (whole, cut) = (run_stream(&spout, 1, 0.1, sweep), run_stream(&spout, 25, 0.004, sweep));
            // 12000 /s, less what has yet to cross the arc: 4 per cell of the corners that
            // the half disc leaves out of its 18 x 9 box.
            let corners = (PPC * (162.0 - 0.5 * std::f32::consts::PI * 81.0)) as usize;
            assert!(whole.len().abs_diff(1200 - corners) <= 2, "{} particles", whole.len());
            assert_eq!(whole.len(), cut.len());
            let nearest = |a: &[f32; 2]| cut.iter().map(|b| (a[0] - b[0]).hypot(a[1] - b[1])).fold(f32::MAX, f32::min);
            let worst = whole.iter().map(nearest).fold(0.0, f32::max);
            assert!(worst < 0.02, "sweep {sweep}: particles differ by up to {worst} cells");
        }
    }

    #[test]
    fn a_stream_is_evenly_packed_at_rest_density() {
        for (spout, dt) in [(Spout::pour(12_000.0, 9.0), 1.0 / 60.0), (Spout::pour(1500.0, 9.0), 1.0 / 144.0), (Spout::jet(40.0, 30_000.0, 20.0), 1.0 / 30.0)] {
            let all = run_stream(&spout, 12, dt, 0.0);
            let half = 0.5 * spout.width;
            let area = spout.width * spout.speed * 12.0 * dt - (2.0 - 0.5 * std::f32::consts::PI) * half * half;
            assert!((all.len() as f32 / area - PPC).abs() < 0.02 * PPC, "{} particles in {area} cells", all.len());
            let mut nearest = f32::MAX;
            for (i, a) in all.iter().enumerate() {
                for b in &all[..i] {
                    nearest = nearest.min((a[0] - b[0]).hypot(a[1] - b[1]));
                }
            }
            assert!(nearest > 0.9 / PPC.sqrt(), "closest pair {nearest} cells apart");
        }
    }

    #[test]
    fn a_stream_has_a_round_top_that_fills_the_brush() {
        let spout = Spout::pour(12_000.0, 9.0);
        let all = run_stream(&spout, 1, 0.02, 0.0);
        // Nothing is born outside the circle behind the spout's centre line...
        let outside = all.iter().filter(|p| p[1] > 100.0 && (p[0] - 100.0).hypot(p[1] - 100.0) > 9.0 + 1e-3).count();
        assert_eq!(outside, 0);
        // ...and the top reaches the rim in the middle and the centre line at the sides.
        let top = |lo: f32, hi: f32| all.iter().filter(|p| (lo..hi).contains(&(p[0] - 100.0))).map(|p| p[1]).fold(f32::MIN, f32::max);
        assert!((top(-1.0, 1.0) - 109.0).abs() < 0.3, "top of the middle {}", top(-1.0, 1.0));
        assert!(top(8.0, 9.0) < 104.5 && top(-9.0, -8.0) < 104.5);
    }

    #[test]
    fn a_stream_drops_a_backlog_instead_of_releasing_it() {
        let (spout, mut stream) = (Spout::pour(12_000.0, 9.0), Stream::default());
        assert_eq!(stream.advance(&spout, [0.0; 2], [0.0; 2], 1.0, 500).len(), 500);
        assert!(stream.advance(&spout, [0.0; 2], [0.0; 2], 0.01, 500).len().abs_diff(120) <= 10);
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
