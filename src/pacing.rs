//! Frame pacing: how much simulated time each rendered frame covers, and in how many
//! substeps.
//!
//! The GPU sizes every substep itself from the CFL condition (see `step.comp`); the pacer
//! only asks for a span of simulated time and records enough substeps to cover it at the
//! last known step length. When a scene is too heavy for real time, it gives each frame
//! fewer substeps, so simulated time slows down and the frame rate, and with it the feel of
//! the mouse, holds.

/// Hard ceiling on substeps in one frame.
pub const MAX_SUBSTEPS_PER_FRAME: u32 = 24;

pub struct Pacer {
    /// Simulated time owed, in seconds.
    bank: f32,
    /// Substeps a frame may currently run; adapts to what the GPU manages.
    cap: f32,
    /// Smoothed ratio of simulated time to the time asked for (1 = keeping up).
    speed: f32,
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
        Pacer { bank: 0.0, cap: MAX_SUBSTEPS_PER_FRAME as f32, speed: 1.0 }
    }

    /// Forgets owed time, e.g. after a pause or a map change.
    pub fn reset(&mut self) {
        self.bank = 0.0;
    }

    /// Simulated time delivered relative to the time asked for; below 1 the scene is too
    /// heavy to run at full speed.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Plans this frame. `frame_dt` is the time since the previous frame, `dt` the expected
    /// substep length (exact with `fixed`), `time_scale` the slow-motion factor, and a frame
    /// longer than `slow_frame` counts as too slow.
    pub fn plan(&mut self, frame_dt: f32, dt: f32, fixed: bool, time_scale: f32, slow_frame: f32) -> Plan {
        // The length of the last frame says whether the GPU coped with what it was given.
        if frame_dt > slow_frame {
            self.cap = (self.cap * 0.8).max(1.0);
        } else if frame_dt < 0.9 * slow_frame {
            self.cap = (self.cap + 0.25).min(MAX_SUBSTEPS_PER_FRAME as f32);
        }
        let wanted = frame_dt.min(0.1) * time_scale;
        self.bank += wanted;
        let cap = self.cap as u32;
        let plan = if fixed {
            let n = ((self.bank / dt) as u32).min(cap);
            let t = n as f32 * dt;
            Plan { substeps: n, target: t, expected: t }
        } else {
            // Round rather than truncate: the GPU spreads the time evenly over the substeps
            // and never exceeds the CFL limit, so a slightly long request just falls short.
            let n = ((self.bank / dt).round() as u32).min(cap);
            let expected = if n == 0 { 0.0 } else { self.bank.min(n as f32 * dt) };
            Plan { substeps: n, target: if n == 0 { 0.0 } else { self.bank }, expected }
        };
        // Time that cannot be simulated this frame is dropped, not carried as debt.
        self.bank = (self.bank - plan.expected).min(dt);
        if wanted > 0.0 {
            // Weighted by frame length and averaged over about half a second, so the
            // frame-to-frame jitter of 0, 1, 2 substeps does not show.
            let blend = (frame_dt * 2.0).min(1.0);
            self.speed += (plan.expected / wanted - self.speed) * blend;
        }
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 240.0;
    const SLOW: f32 = 1.0 / 55.0;

    /// Runs frames whose length is a fixed overhead plus a cost per substep, the way a GPU
    /// would deliver them. Returns (simulated seconds, wall seconds, longest late frame).
    fn run(pacer: &mut Pacer, overhead: f32, cost: f32, vsync: f32, frames: u32) -> (f32, f32, f32) {
        let (mut simulated, mut wall, mut worst) = (0.0, 0.0, 0.0f32);
        let mut frame_dt = vsync.max(overhead);
        for frame in 0..frames {
            let plan = pacer.plan(frame_dt, DT, false, 1.0, SLOW);
            let n = plan.substeps;
            let work = overhead + n as f32 * cost;
            // With vsync a frame lasts a whole number of refresh intervals.
            frame_dt = if vsync > 0.0 { (work / vsync).ceil().max(1.0) * vsync } else { work };
            simulated += plan.expected;
            wall += frame_dt;
            if frame > frames / 2 {
                worst = worst.max(frame_dt);
            }
        }
        (simulated, wall, worst)
    }

    #[test]
    fn a_light_scene_runs_in_real_time() {
        for vsync in [0.0, 1.0 / 60.0, 1.0 / 165.0] {
            let mut pacer = Pacer::new();
            let (simulated, wall, _) = run(&mut pacer, 0.0005, 0.001, vsync, 2000);
            assert!((simulated / wall - 1.0).abs() < 0.02, "vsync {vsync}: {simulated} s in {wall} s");
            assert!((pacer.speed() - 1.0).abs() < 0.05);
        }
    }

    #[test]
    fn a_heavy_scene_slows_time_instead_of_the_frame_rate() {
        // 6 ms per substep: real time would need 1.44 s of work per second.
        for vsync in [0.0, 1.0 / 60.0, 1.0 / 165.0] {
            let mut pacer = Pacer::new();
            let (simulated, wall, worst) = run(&mut pacer, 0.001, 0.006, vsync, 2000);
            assert!(worst < 1.6 * SLOW, "vsync {vsync}: frames reach {} ms", worst * 1e3);
            let speed = simulated / wall;
            assert!(speed > 0.4 && speed < 0.75, "vsync {vsync}: speed {speed}");
            assert!((pacer.speed() - speed).abs() < 0.15);
        }
    }

    #[test]
    fn a_hitch_is_not_paid_back() {
        let mut pacer = Pacer::new();
        run(&mut pacer, 0.0005, 0.001, 1.0 / 60.0, 200);
        // Half a second without a frame must not turn into a burst of catch-up substeps.
        let n = pacer.plan(0.5, DT, false, 1.0, SLOW).substeps;
        assert!(n <= MAX_SUBSTEPS_PER_FRAME);
        let next = pacer.plan(1.0 / 60.0, DT, false, 1.0, SLOW).substeps;
        assert!(next <= 5, "{next} substeps right after a hitch");
    }

    #[test]
    fn slow_motion_scales_the_substep_count() {
        let mut pacer = Pacer::new();
        let total: u32 = (0..600).map(|_| pacer.plan(1.0 / 60.0, DT, false, 0.25, SLOW).substeps).sum();
        assert!(total.abs_diff(600) <= 2, "{total} substeps in 10 s at quarter speed");
        assert!((pacer.speed() - 1.0).abs() < 0.05);
    }
}
