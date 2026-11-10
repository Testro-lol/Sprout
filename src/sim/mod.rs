//! GPU multi-fluid solver.
//!
//! A particle-in-cell (APIC) liquid on a staggered grid, run entirely in compute shaders:
//!
//! ```text
//!   particles.comp   gather + advect + collide + scatter   (the only per-particle pass)
//!   grid_prep.comp   accumulators -> velocities, composition, liquid/air cells
//!   grid_coef.comp   variable-density face coefficients, ghost-fluid free surface
//!   viscosity.comp   implicit variable viscosity      (red-black Gauss-Seidel, n sweeps)
//!   rhs.comp         divergence + density correction
//!   pressure.comp    variable-density pressure solve  (red-black SOR, n sweeps, warm start)
//!   project.comp     subtract the pressure gradient
//! ```
//!
//! `FluidSim` owns the buffers and pipelines and records those passes into a command
//! buffer; it never blocks on the GPU except in the explicit `*_blocking` helpers used by
//! the headless tools.

pub mod params;

use std::{collections::BTreeMap, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{CopyBufferInfo, PrimaryCommandBufferAbstract},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::GpuFuture,
};

use crate::{
    gpu::{Builder, Gpu},
    scene::SolidField,
    spirv,
};
pub use params::*;

/// Most particles the pour tool can hand over in one frame.
pub const SPAWN_MAX: usize = 32768;

/// Ints per cell in the scatter accumulators; must match `ACC_STRIDE` in `common.glsl`.
const ACC_STRIDE: u64 = 8;
const PARTICLE_GROUP: u32 = 128;
/// Substeps between two particle sorts (see `sort.comp`), and the particle count below
/// which sorting is not worth its dispatches.
const SORT_INTERVAL: u32 = 128;
const SORT_MIN_PARTICLES: u32 = 16384;
const GRID_GROUP: u32 = 16;
const STATS_RUN: u32 = 64 * 64;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub live: u32,
    pub oil: u32,
    /// Syrup volume in particle-equivalents (dissolved syrup included).
    pub syrup: f32,
}

impl Stats {
    pub fn water(&self) -> f32 {
        (self.live - self.oil) as f32 - self.syrup
    }
}

struct Pass {
    pipeline: Arc<ComputePipeline>,
    set: Arc<DescriptorSet>,
}

impl Pass {
    fn new(gpu: &Gpu, spirv: &[u8], table: &BTreeMap<u32, Subbuffer<[u8]>>) -> Pass {
        let device = gpu.device.clone();
        let stage = PipelineShaderStageCreateInfo::new(gpu.shader(spirv));
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline =
            ComputePipeline::new(device, None, ComputePipelineCreateInfo::stage_layout(stage, layout)).unwrap();
        // Each shader only keeps the bindings it actually touches; bind exactly those.
        let set_layout = pipeline.layout().set_layouts()[0].clone();
        let writes: Vec<_> =
            set_layout.bindings().keys().map(|&b| WriteDescriptorSet::buffer(b, table[&b].clone())).collect();
        let set = DescriptorSet::new(gpu.descriptors.clone(), set_layout, writes, []).unwrap();
        Pass { pipeline, set }
    }

    fn bind(&self, b: &mut Builder) {
        b.bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, self.set.clone())
            .unwrap();
    }

    fn push(&self, b: &mut Builder, value: u32) {
        b.push_constants(self.pipeline.layout().clone(), 0, value).unwrap();
    }
}

fn dispatch(b: &mut Builder, groups: [u32; 3]) {
    // SAFETY: every shader bounds-checks its invocation against the grid / slot count, and
    // the descriptor sets are built from each pipeline's own layout.
    unsafe { b.dispatch(groups) }.unwrap();
}

fn device_buffer<T: BufferContents>(gpu: &Gpu, len: u64, extra: BufferUsage) -> Subbuffer<[T]> {
    Buffer::new_slice::<T>(
        gpu.memory.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC | extra,
            ..Default::default()
        },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
        len,
    )
    .expect("out of GPU memory")
}

fn staging<T: BufferContents + Copy>(gpu: &Gpu, data: &[T]) -> Subbuffer<[T]> {
    Buffer::from_iter(
        gpu.memory.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .unwrap()
}

fn readback<T: BufferContents>(gpu: &Gpu, len: u64) -> Subbuffer<[T]> {
    Buffer::new_slice::<T>(
        gpu.memory.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        len,
    )
    .unwrap()
}

pub struct FluidSim {
    gpu: Arc<Gpu>,
    pub grid: [u32; 2],
    pub capacity: u32,
    pub params: SimParams,
    /// Latest particle counts, refreshed asynchronously by `request_stats`.
    pub stats: Stats,

    n_slots: u32,
    grid_valid: bool,
    /// Substeps left that get extra pressure sweeps (the solve starts cold after a reset).
    warmup: u32,
    cleared: bool,
    viscous: bool,
    frame: u32,
    substeps_done: u64,

    particles: Subbuffer<[Particle]>,
    /// Scratch copy of the particles that the periodic sort writes into.
    sorted: Subbuffer<[Particle]>,
    since_sort: u32,
    acc: Subbuffer<[u32]>,
    pres: Subbuffer<[[f32; 2]]>,
    solid: Subbuffer<[[f32; 4]]>,
    sdf: Subbuffer<[f32]>,
    spawn: Subbuffer<[Particle]>,
    counter: Subbuffer<[u32]>,
    params_buf: Subbuffer<GpuParams>,
    /// Every float field, for the one-off clear.
    fields: Vec<Subbuffer<[u32]>>,

    pass_particles: Pass,
    pass_prep: Pass,
    pass_coef: Pass,
    pass_viscosity: Pass,
    pass_rhs: Pass,
    pass_pressure: Pass,
    pass_project: Pass,
    pass_claim: Pass,
    pass_stats: Pass,
    pass_sort: Pass,

    pending_reset: Option<Vec<Particle>>,
    pending_solids: Option<SolidField>,
    pending_spawn: Vec<Particle>,
    stats_host: Subbuffer<[u32]>,
    stats_in_flight: bool,
}

impl FluidSim {
    pub fn new(gpu: Arc<Gpu>, grid: [u32; 2], capacity: u32) -> FluidSim {
        let cells = (grid[0] * grid[1]) as u64;
        let nodes = ((grid[0] + 1) * (grid[1] + 1)) as u64;
        let none = BufferUsage::empty();

        let particles = device_buffer::<Particle>(&gpu, capacity as u64, none);
        let sorted = device_buffer::<Particle>(&gpu, capacity as u64, none);
        let acc = device_buffer::<u32>(&gpu, cells * ACC_STRIDE, none);
        let vel = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let vel0 = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let cell = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let comp = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let coef = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let pw = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let prhs = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let pres = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let solid = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let sdf = device_buffer::<f32>(&gpu, nodes, none);
        let spawn = device_buffer::<Particle>(&gpu, SPAWN_MAX as u64, none);
        let counter = device_buffer::<u32>(&gpu, 8, none);
        let params_buf = Buffer::new_sized::<GpuParams>(
            gpu.memory.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
        )
        .unwrap();

        let mut table: BTreeMap<u32, Subbuffer<[u8]>> = BTreeMap::new();
        table.insert(0, params_buf.as_bytes().clone());
        table.insert(1, particles.as_bytes().clone());
        table.insert(2, acc.as_bytes().clone());
        table.insert(3, vel.as_bytes().clone());
        table.insert(4, vel0.as_bytes().clone());
        table.insert(5, cell.as_bytes().clone());
        table.insert(6, comp.as_bytes().clone());
        table.insert(7, coef.as_bytes().clone());
        table.insert(8, pw.as_bytes().clone());
        table.insert(9, prhs.as_bytes().clone());
        table.insert(10, pres.as_bytes().clone());
        table.insert(11, solid.as_bytes().clone());
        table.insert(12, sdf.as_bytes().clone());
        table.insert(13, spawn.as_bytes().clone());
        table.insert(14, counter.as_bytes().clone());
        table.insert(15, sorted.as_bytes().clone());

        let fields = vec![
            particles.clone().reinterpret::<[u32]>(),
            vel.reinterpret::<[u32]>(),
            vel0.reinterpret::<[u32]>(),
            cell.reinterpret::<[u32]>(),
            comp.reinterpret::<[u32]>(),
            coef.reinterpret::<[u32]>(),
            pw.reinterpret::<[u32]>(),
            prhs.reinterpret::<[u32]>(),
            pres.clone().reinterpret::<[u32]>(),
            solid.clone().reinterpret::<[u32]>(),
            sdf.clone().reinterpret::<[u32]>(),
            counter.clone(),
        ];

        FluidSim {
            pass_particles: Pass::new(&gpu, spirv!("particles.comp"), &table),
            pass_prep: Pass::new(&gpu, spirv!("grid_prep.comp"), &table),
            pass_coef: Pass::new(&gpu, spirv!("grid_coef.comp"), &table),
            pass_viscosity: Pass::new(&gpu, spirv!("viscosity.comp"), &table),
            pass_rhs: Pass::new(&gpu, spirv!("rhs.comp"), &table),
            pass_pressure: Pass::new(&gpu, spirv!("pressure.comp"), &table),
            pass_project: Pass::new(&gpu, spirv!("project.comp"), &table),
            pass_claim: Pass::new(&gpu, spirv!("claim.comp"), &table),
            pass_stats: Pass::new(&gpu, spirv!("stats.comp"), &table),
            pass_sort: Pass::new(&gpu, spirv!("sort.comp"), &table),
            stats_host: readback::<u32>(&gpu, 8),
            gpu,
            grid,
            capacity,
            params: SimParams::default(),
            stats: Stats::default(),
            n_slots: 0,
            grid_valid: false,
            warmup: 0,
            cleared: false,
            viscous: false,
            frame: 0,
            substeps_done: 0,
            particles,
            sorted,
            since_sort: 0,
            acc,
            pres,
            solid,
            sdf,
            spawn,
            counter,
            params_buf,
            fields,
            pending_reset: None,
            pending_solids: None,
            pending_spawn: Vec::new(),
            stats_in_flight: false,
        }
    }

    pub fn particle_buffer(&self) -> &Subbuffer<[Particle]> {
        &self.particles
    }

    pub fn sdf_buffer(&self) -> &Subbuffer<[f32]> {
        &self.sdf
    }

    /// Number of particle slots that may hold live particles.
    pub fn slots(&self) -> u32 {
        self.n_slots
    }

    pub fn substeps_done(&self) -> u64 {
        self.substeps_done
    }

    pub fn sim_time(&self) -> f64 {
        self.substeps_done as f64 * self.params.dt() as f64
    }

    /// Replaces the solid geometry. Takes effect at the next `record`.
    pub fn set_solids(&mut self, field: SolidField) {
        self.pending_solids = Some(field);
    }

    /// Replaces every particle. Takes effect at the next `record`.
    pub fn reset(&mut self, mut particles: Vec<Particle>) {
        particles.truncate(self.capacity as usize);
        self.pending_spawn.clear();
        self.viscous = false;
        self.note_materials(&particles);
        self.stats = Stats { live: particles.len() as u32, ..Default::default() };
        self.pending_reset = Some(particles);
    }

    /// Queues new particles; they appear at the next `record`. Returns how many were
    /// accepted (the queue holds `SPAWN_MAX` per frame).
    pub fn spawn(&mut self, particles: &[Particle]) -> usize {
        let room = SPAWN_MAX - self.pending_spawn.len();
        let particles = &particles[..particles.len().min(room)];
        self.note_materials(particles);
        self.pending_spawn.extend_from_slice(particles);
        particles.len()
    }

    fn note_materials(&mut self, particles: &[Particle]) {
        // Viscosity sweeps are pure overhead while the tank only holds water.
        self.viscous |= particles.iter().any(|p| p.kind == KIND_OIL || p.s > 0.0);
    }

    fn gpu_params(&self, tool: ToolState, spawn_count: u32) -> GpuParams {
        let p = &self.params;
        let dt = p.dt();
        GpuParams {
            grid: [self.grid[0] as i32, self.grid[1] as i32],
            n_slots: self.n_slots,
            spawn_count,
            dt,
            gravity: p.gravity,
            inv_ppc: 1.0 / PPC as f32,
            fluid_thr: p.fluid_thr,
            rho: [p.rho[0], p.rho[1], p.rho[2], 0.0],
            nu: [p.nu[0], p.nu[1], p.nu[2], 0.0],
            sep_tau: p.sep_tau,
            sep_max: 0.4 / dt,
            diffuse: p.diffuse,
            density_k: p.density_k,
            omega: p.sor_omega,
            rest_phi: p.rest_phi,
            vmax: p.cfl_max / dt,
            wall_margin: 0.1,
            tool_pos: tool.pos,
            tool_vel: tool.vel,
            tool_radius: tool.radius.max(0.01),
            tool_mode: tool.mode,
            tool_strength: tool.strength,
            frame: self.frame,
        }
    }

    /// Records pending uploads plus `substeps` simulation steps.
    pub fn record(&mut self, b: &mut Builder, substeps: u32, tool: ToolState) {
        let gpu = self.gpu.clone();
        self.frame = self.frame.wrapping_add(1);

        if !self.cleared {
            for f in &self.fields {
                b.fill_buffer(f.clone(), 0).unwrap();
            }
            self.cleared = true;
        }
        if let Some(field) = self.pending_solids.take() {
            b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &field.solid), self.solid.clone())).unwrap();
            b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &field.sdf), self.sdf.clone())).unwrap();
        }
        if let Some(particles) = self.pending_reset.take() {
            self.n_slots = particles.len() as u32;
            if !particles.is_empty() {
                let dst = self.particles.clone().slice(0..particles.len() as u64);
                b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &particles), dst)).unwrap();
            }
            b.fill_buffer(self.pres.clone().reinterpret::<[u32]>(), 0).unwrap();
            self.grid_valid = false;
            self.warmup = 4;
        }

        // New particles go into fresh slots while there are any; after that they recycle
        // slots freed by the eraser.
        let mut claim = 0u32;
        if !self.pending_spawn.is_empty() {
            let spawn = std::mem::take(&mut self.pending_spawn);
            let n = spawn.len() as u32;
            let fresh = n.min(self.capacity - self.n_slots);
            let src = staging(&gpu, &spawn);
            if fresh > 0 {
                let dst = self.particles.clone().slice(self.n_slots as u64..(self.n_slots + fresh) as u64);
                b.copy_buffer(CopyBufferInfo::buffers(src.clone().slice(0..fresh as u64), dst)).unwrap();
                self.n_slots += fresh;
            }
            if n > fresh {
                claim = n - fresh;
                let dst = self.spawn.clone().slice(0..claim as u64);
                b.copy_buffer(CopyBufferInfo::buffers(src.slice(fresh as u64..n as u64), dst)).unwrap();
            }
        }

        b.update_buffer(self.params_buf.clone(), Box::new(self.gpu_params(tool, claim))).unwrap();

        let particle_groups = [self.n_slots.div_ceil(PARTICLE_GROUP).max(1), 1, 1];
        let grid_groups = [self.grid[0].div_ceil(GRID_GROUP), self.grid[1].div_ceil(GRID_GROUP), 1];

        if claim > 0 {
            b.fill_buffer(self.counter.clone(), 0).unwrap();
            self.pass_claim.bind(b);
            dispatch(b, particle_groups);
        }

        let viscosity_iters = if self.viscous { self.params.viscosity_iters } else { 0 };
        for _ in 0..substeps {
            self.since_sort += 1;
            if self.since_sort >= SORT_INTERVAL && self.n_slots >= SORT_MIN_PARTICLES {
                self.record_sort(b, particle_groups);
            }
            b.fill_buffer(self.acc.clone(), 0).unwrap();

            self.pass_particles.bind(b);
            self.pass_particles.push(b, self.grid_valid as u32);
            dispatch(b, particle_groups);

            self.pass_prep.bind(b);
            dispatch(b, grid_groups);
            self.pass_coef.bind(b);
            dispatch(b, grid_groups);

            if viscosity_iters > 0 {
                self.pass_viscosity.bind(b);
                for _ in 0..viscosity_iters {
                    for parity in 0..2 {
                        self.pass_viscosity.push(b, parity);
                        dispatch(b, grid_groups);
                    }
                }
            }

            self.pass_rhs.bind(b);
            dispatch(b, grid_groups);

            let mut pressure_iters = self.params.pressure_iters;
            if self.warmup > 0 {
                self.warmup -= 1;
                pressure_iters = (pressure_iters * 10).clamp(1, 400);
            }
            self.pass_pressure.bind(b);
            for _ in 0..pressure_iters {
                for parity in 0..2 {
                    self.pass_pressure.push(b, parity);
                    dispatch(b, grid_groups);
                }
            }

            self.pass_project.bind(b);
            dispatch(b, grid_groups);

            self.grid_valid = true;
            self.substeps_done += 1;
        }
    }

    /// Records a sort of the particle buffer by grid cell. Positions and velocities are
    /// untouched; only the order in memory changes, and dead slots move to the end.
    fn record_sort(&mut self, b: &mut Builder, particle_groups: [u32; 3]) {
        self.since_sort = 0;
        let n = self.n_slots as u64;
        let words = (size_of::<Particle>() / size_of::<u32>()) as u64;
        b.fill_buffer(self.acc.clone(), 0).unwrap();
        // Slots that no live particle lands in must read as dead afterwards.
        b.fill_buffer(self.sorted.clone().reinterpret::<[u32]>().slice(0..n * words), 0).unwrap();
        self.pass_sort.bind(b);
        let rows = [self.grid[1].div_ceil(PARTICLE_GROUP), 1, 1];
        for (stage, groups) in [particle_groups, rows, [1, 1, 1], particle_groups].into_iter().enumerate() {
            self.pass_sort.push(b, stage as u32);
            dispatch(b, groups);
        }
        b.copy_buffer(CopyBufferInfo::buffers(self.sorted.clone().slice(0..n), self.particles.clone().slice(0..n)))
            .unwrap();
    }

    /// Records a particle count. The result lands in `stats` a few frames later, once
    /// `poll_stats` sees that the GPU has finished with it.
    pub fn request_stats(&mut self, b: &mut Builder) {
        if self.stats_in_flight {
            return;
        }
        b.fill_buffer(self.counter.clone(), 0).unwrap();
        self.pass_stats.bind(b);
        dispatch(b, [self.n_slots.div_ceil(STATS_RUN).max(1), 1, 1]);
        b.copy_buffer(CopyBufferInfo::buffers(self.counter.clone(), self.stats_host.clone())).unwrap();
        self.stats_in_flight = true;
    }

    /// Picks up the result of `request_stats` if the GPU is done with it.
    pub fn poll_stats(&mut self) -> bool {
        if !self.stats_in_flight {
            return false;
        }
        let Ok(data) = self.stats_host.read() else {
            return false;
        };
        self.stats = Stats { live: data[4], oil: data[5], syrup: data[6] as f32 / 256.0 };
        self.stats_in_flight = false;
        true
    }

    // ---- blocking helpers for the headless tools ------------------------------------

    pub fn step_blocking(&mut self, substeps: u32, tool: ToolState) {
        let mut b = self.gpu.builder();
        self.record(&mut b, substeps, tool);
        submit_and_wait(&self.gpu, b);
    }

    pub fn download_particles(&self) -> Vec<Particle> {
        if self.n_slots == 0 {
            return Vec::new();
        }
        let host = readback::<Particle>(&self.gpu, self.n_slots as u64);
        let mut b = self.gpu.builder();
        b.copy_buffer(CopyBufferInfo::buffers(self.particles.clone().slice(0..self.n_slots as u64), host.clone()))
            .unwrap();
        submit_and_wait(&self.gpu, b);
        host.read().unwrap().iter().copied().filter(|p| p.kind != KIND_DEAD).collect()
    }
}

pub fn submit_and_wait(gpu: &Gpu, b: Builder) {
    b.build()
        .unwrap()
        .execute(gpu.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}
