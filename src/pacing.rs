//! Frame pacing: how much simulated time each rendered frame covers, and in how many
//! substeps.
//!
//! The GPU sizes every substep itself from the CFL condition (see `step.comp`); the pacer
//! only asks for a span of simulated time and records enough substeps to cover it at the
//! last known step length.
//!
//! Two things keep frame times even rather than merely fast on average:
//!
//! - How many substeps fit in a frame is decided from the measured GPU time of a substep
//!   (`FluidSim::substep_cost`), with hysteresis, not by reacting to the last frame being
//!   late. A scene too heavy for real time therefore runs the same number of substeps every
//!   frame and simulated time slows down, instead of alternating between n and n + 1.
//! - A frame that is much shorter than a substep (an uncapped frame rate) does not get a
//!   shortened substep of its own; time is banked until a full-length one is due.

/// Hard ceiling on substeps in one frame.
pub const MAX_SUBSTEPS_PER_FRAME: u32 = 24;
/// Substeps a frame may run before the GPU cost of one is known.
const UNMEASURED_SUBSTEPS: u32 = 2;
/// Simulated time left in the bank rather than given a substep of its own, as a share of a
/// substep. Without it a frame that asks for 1.02 substeps would run two half-length ones.
const SLACK: f32 = 0.1;

pub struct Pacer {
    /// Simulated time owed, in seconds.
    bank: f32,
    /// Substeps a frame may currently run.
    cap: u32,
    /// Simulated time delivered and asked for, decaying over about half a second; their
    /// ratio is the speed (1 = keeping up).
    delivered: f32,
    asked: f32,
    /// Substeps of the previous frame, and the length of the last frame that ran none, s.
    last: u32,
    idle_frame: f32,
}

/// What one frame should simulate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub substeps: u32,
    /// Simulated time the substeps should cover, s.
    pub target: f32,
    /// What they are expected to cover given the last known step length, s.
    pub expected: f32,
}

impl Pacer {
    pub fn new() -> Pacer {
        Pacer { bank: 0.0, cap: UNMEASURED_SUBSTEPS, delivered: 0.0, asked: 0.0, last: 0, idle_frame: 0.0 }
    }

    /// Forgets owed time, e.g. after a pause or a map change.
    pub fn reset(&mut self) {
        self.bank = 0.0;
    }

    /// Simulated time delivered relative to the time asked for; below 1 the scene is too
    /// heavy to run at full speed.
    pub fn speed(&self) -> f32 {
        if self.asked > 0.0 { self.delivered / self.asked } else { 1.0 }
    }

    /// Plans this frame. `frame_dt` is the time since the previous frame, `dt` the expected
    /// substep length (exact with `fixed`), `time_scale` the slow-motion factor, `cost` the
    /// measured GPU time of one substep (0 if not known yet) and `budget` the GPU time a
    /// frame may spend simulating.
    pub fn plan(&mut self, frame_dt: f32, dt: f32, fixed: bool, time_scale: f32, cost: f32, budget: f32) -> Plan {
        if cost > 0.0 {
            // Raise the cap only when one more substep clearly fits, lower it as soon as the
            // current count does not: between the two the count stays put.
            if (self.cap + 1) as f32 * cost < 0.85 * budget {
                self.cap += 1;
            }
            while self.cap > 1 && self.cap as f32 * cost > budget {
                self.cap -= 1;
            }
            self.cap = self.cap.min(MAX_SUBSTEPS_PER_FRAME);
        }
        let wanted = frame_dt.min(0.1) * time_scale;
        self.bank += wanted;
        // How long a frame without a substep lasts (what waiting one more frame would add):
        // the last frame less what its substeps cost.
        let idle = (frame_dt.min(0.1) - self.last as f32 * cost).max(1e-4);
        self.idle_frame = if self.idle_frame == 0.0 { idle } else { self.idle_frame + 0.25 * (idle - self.idle_frame) };
        let plan = if fixed {
            let n = ((self.bank / dt) as u32).min(self.cap);
            let t = n as f32 * dt;
            Plan { substeps: n, target: t, expected: t }
        } else {
            let r = self.bank / dt;
            // Step once a full substep is owed, or when waiting for the next frame would
            // owe more than one (so that frames as long as a substep each get exactly one).
            let due = r >= 1.0 || self.bank + self.idle_frame * time_scale > dt;
            let n = if due { ((r - SLACK).ceil().max(1.0) as u32).min(self.cap) } else { 0 };
            // The GPU spreads the time evenly over the substeps and never exceeds the CFL
            // limit, so the request is cut to what they can cover.
            let t = self.bank.min(n as f32 * dt);
            Plan { substeps: n, target: t, expected: t }
        };
        // Time that cannot be simulated this frame is dropped, not carried as debt. Time
        // still in the bank will be simulated and counts as delivered.
        let kept = (self.bank - plan.expected).min(dt);
        let dropped = self.bank - plan.expected - kept;
        self.bank = kept;
        let keep = 1.0 - (frame_dt * 2.0).min(1.0);
        self.delivered = self.delivered * keep + wanted - dropped;
        self.asked = self.asked * keep + wanted;
        self.last = plan.substeps;
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 240.0;
    const BUDGET: f32 = 0.8 / 60.0;

    /// Runs frames whose length is a fixed overhead plus a cost per substep, the way a GPU
    /// would deliver them. Returns (simulated seconds, wall seconds, frame times of the
    /// second half).
    fn run(pacer: &mut Pacer, overhead: f32, cost: f32, dt: f32, vsync: f32, frames: u32) -> (f32, f32, Vec<f32>) {
        let (mut simulated, mut wall) = (0.0, 0.0);
        let mut times = Vec::new();
        let mut frame_dt = vsync.max(overhead);
        for frame in 0..frames {
            let plan = pacer.plan(frame_dt, dt, false, 1.0, cost, BUDGET);
            let work = overhead + plan.substeps as f32 * cost;
            // With vsync a frame lasts a whole number of refresh intervals.
            frame_dt = if vsync > 0.0 { (work / vsync).ceil().max(1.0) * vsync } else { work };
            simulated += plan.expected;
            wall += frame_dt;
            if frame > frames / 2 {
                times.push(frame_dt);
            }
        }
        (simulated, wall, times)
    }

    #[test]
    fn a_light_scene_runs_in_real_time() {
        for vsync in [0.0, 1.0 / 60.0, 1.0 / 165.0] {
            let mut pacer = Pacer::new();
            let (simulated, wall, _) = run(&mut pacer, 0.0005, 0.001, DT, vsync, 4000);
            assert!((simulated / wall - 1.0).abs() < 0.02, "vsync {vsync}: {simulated} s in {wall} s");
            assert!((pacer.speed() - 1.0).abs() < 0.1, "vsync {vsync}: speed {}", pacer.speed());
        }
    }

    #[test]
    fn a_heavy_scene_slows_time_and_keeps_an_even_frame_time() {
        // 6 ms per substep of 1/240 s: real time would need 1.44 s of work per second.
        for vsync in [0.0, 1.0 / 60.0] {
            let mut pacer = Pacer::new();
            let (simulated, wall, times) = run(&mut pacer, 0.001, 0.006, DT, vsync, 2000);
            let (lo, hi) = times.iter().fold((f32::MAX, 0.0f32), |(lo, hi), &t| (lo.min(t), hi.max(t)));
            assert!(hi < 1.0 / 55.0, "vsync {vsync}: frames reach {} ms", hi * 1e3);
            assert!(hi - lo < 1e-4, "vsync {vsync}: frame times vary between {} and {} ms", lo * 1e3, hi * 1e3);
            let speed = simulated / wall;
            assert!(speed > 0.4 && speed < 0.75, "vsync {vsync}: speed {speed}");
            assert!((pacer.speed() - speed).abs() < 0.15);
        }
    }

    #[test]
    fn short_frames_bank_time_for_full_length_substeps() {
        // Uncapped: 0.4 ms frames, 5 ms substeps that cost 3 ms each.
        let mut pacer = Pacer::new();
        let dt = 0.005;
        let (mut steps, mut simulated, mut frame_dt) = (0u32, 0.0f32, 0.0004f32);
        for _ in 0..5000 {
            let plan = pacer.plan(frame_dt, dt, false, 1.0, 0.003, BUDGET);
            assert!(plan.substeps <= 1);
            steps += plan.substeps;
            simulated += plan.expected;
            frame_dt = 0.0004 + plan.substeps as f32 * 0.003;
        }
        let mean = simulated / steps as f32;
        assert!(mean > 0.85 * dt, "substeps average {} ms of a possible {} ms", mean * 1e3, dt * 1e3);
    }

    #[test]
    fn a_hitch_is_not_paid_back() {
        let mut pacer = Pacer::new();
        run(&mut pacer, 0.0005, 0.001, DT, 1.0 / 60.0, 200);
        // Half a second without a frame must not turn into a burst of catch-up substeps.
        let n = pacer.plan(0.5, DT, false, 1.0, 0.001, BUDGET).substeps;
        assert!(n <= MAX_SUBSTEPS_PER_FRAME);
        let next = pacer.plan(1.0 / 60.0, DT, false, 1.0, 0.001, BUDGET).substeps;
        assert!(next <= 5, "{next} substeps right after a hitch");
    }

    #[test]
    fn slow_motion_scales_the_substep_count() {
        let mut pacer = Pacer::new();
        let total: u32 = (0..600).map(|_| pacer.plan(1.0 / 60.0, DT, false, 0.25, 0.001, BUDGET).substeps).sum();
        assert!(total.abs_diff(600) <= 12, "{total} substeps in 10 s at quarter speed");
        assert!((pacer.speed() - 1.0).abs() < 0.1);
    }
}
