//! GPU multi-fluid solver.
//!
//! A particle-in-cell (APIC) liquid on a staggered grid, run entirely in compute shaders.
//! One substep:
//!
//! ```text
//!   step.comp        substep length from the CFL condition (on the GPU, no read-back)
//!   particles.comp   gather + advect + collide + scatter   (the only per-particle pass)
//!   grid_prep.comp   accumulators -> velocities (+ gravity), composition, liquid/air cells
//!   grid_coef.comp   variable-density face coefficients, ghost-fluid free surface
//!   surf_*.comp      curvature; surface tension as interface pressure jumps and forces
//!   visc_*.comp      implicit viscosity, full stress tensor, conjugate gradients
//!   rhs.comp         pressure matrix, divergence + density correction
//!   mg_*.comp, cg_*  pressure: multigrid-preconditioned conjugate gradients to a tolerance
//!   project.comp     subtract the pressure gradient
//! ```
//!
//! Both linear solves stop at a relative tolerance rather than after a fixed number of
//! sweeps, so their accuracy does not depend on the resolution or the fluids. The number of
//! iterations recorded per substep adapts to what recent solves needed (read back a frame
//! later); iterations recorded after a solve converged are dispatched indirectly with zero
//! work groups and cost only an empty dispatch. A substep that runs out of recorded
//! iterations before converging is counted in `SolverStats`.
//!
//! `FluidSim` owns the buffers and pipelines and records those passes into a command
//! buffer; it never blocks on the GPU except in the explicit `*_blocking` helpers used by
//! the headless tools.

pub mod params;
mod raw;

use std::{collections::BTreeMap, sync::Arc};

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, DispatchIndirectCommand, PrimaryCommandBufferAbstract},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    sync::{GpuFuture, PipelineStage},
};

use crate::{
    gpu::{Builder, Gpu},
    scene::SolidField,
    spirv,
};
pub use params::*;
use raw::{RawSecondary, Recorder};

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
/// Largest grid the coarsest-level solve takes (cells).
const COARSEST_CELLS: u32 = 4;
/// Levels of at most this many cells are all handled by one dispatch (`mg_tail.comp`).
const TAIL_CELLS: u32 = 1500;
/// Weight of the multigrid coarse-grid correction. Aggregation (piecewise-constant)
/// coarsening under-corrects smooth errors; over-correcting compensates (Braess 1995).
const MG_OMEGA: f32 = 1.8;
/// Words of `StepBuf` and `SolveBuf`.
const STEP_WORDS: u64 = 12;
const SOLVE_WORDS: u64 = 28;
/// First word of the statistics in `SolveBuf`, cleared by the CPU after each read-back.
const SOLVE_STATS: u64 = 21;
/// The blocking helpers stop when less than this share of a substep is left (see
/// `advance_blocking`).
const SLIVER: f64 = 0.02;
/// Host read-back slots in flight.
const READBACK_RING: usize = 3;

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

/// What the solves did, accumulated over every substep whose read-back has arrived.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolverStats {
    pub substeps: u64,
    /// Pressure solves that hit the recorded iteration limit before the tolerance.
    pub pressure_unconverged: u64,
    pub viscosity_unconverged: u64,
    /// Most iterations any single solve needed.
    pub pressure_iters_max: u32,
    pub viscosity_iters_max: u32,
    /// Largest final relative residual seen.
    pub pressure_residual_max: f32,
    pub viscosity_residual_max: f32,
    /// Latest substep length and the CFL limit it was under, s.
    pub dt: f32,
    pub dt_cfl: f32,
    /// Speed (cells/s) the latest substep was sized for.
    pub umax: f32,
    /// Fastest liquid face and fastest particle displacement of the latest substep, cells/s.
    pub umax_grid: f32,
    pub umax_part: f32,
}

impl SolverStats {
    /// Clears the extremes, keeping the latest step information.
    pub fn reset_extremes(&mut self) {
        self.pressure_iters_max = 0;
        self.viscosity_iters_max = 0;
        self.pressure_residual_max = 0.0;
        self.viscosity_residual_max = 0.0;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct MgPc {
    off_f: i32,
    off_c: i32,
    dim_f: [i32; 2],
    dim_c: [i32; 2],
    mode: u32,
    omega: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct TailPc {
    first: u32,
    last: u32,
    mode: u32,
    omega: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ReducePc {
    mode: u32,
    n: u32,
    n_entries: u32,
    first_entry: u32,
    last_entry: u32,
    pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Viscosity {
    None = 0,
    Explicit = 1,
    Implicit = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ViscPc {
    mode: u32,
    half_off: u32,
}

struct Pass {
    name: &'static str,
    pipeline: Arc<ComputePipeline>,
    set: Arc<DescriptorSet>,
    /// Bytes of push constants the shader actually reads (unused members are stripped).
    push_size: u32,
}

impl Pass {
    fn new(gpu: &Gpu, name: &'static str, spirv: &[u8], table: &BTreeMap<u32, Subbuffer<[u8]>>) -> Pass {
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
        let push_size =
            pipeline.layout().push_constant_ranges().iter().map(|r| r.offset + r.size).max().unwrap_or(0);
        Pass { name, pipeline, set, push_size }
    }

    fn bind<L>(&self, b: &mut AutoCommandBufferBuilder<L>) {
        b.bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, self.set.clone())
            .unwrap();
    }

    fn bind_raw(&self, r: &mut Recorder) {
        r.bind(self.name, &self.pipeline, &self.set);
    }

    fn push_raw<T: Pod>(&self, r: &mut Recorder, value: T) {
        let bytes = bytemuck::bytes_of(&value);
        let n = (self.push_size as usize).min(bytes.len());
        if n == 0 {
            return;
        }
        let mut words = [0u32; 8];
        bytemuck::cast_slice_mut::<u32, u8>(&mut words)[..n].copy_from_slice(&bytes[..n]);
        r.push(&words[..n.div_ceil(4)]);
    }

    fn push<T: Pod, L>(&self, b: &mut AutoCommandBufferBuilder<L>, value: T) {
        let bytes = bytemuck::bytes_of(&value);
        let n = (self.push_size as usize).min(bytes.len());
        if n == 0 {
            return;
        }
        let mut words = [0u32; 8];
        bytemuck::cast_slice_mut::<u32, u8>(&mut words)[..n].copy_from_slice(&bytes[..n]);
        let layout = self.pipeline.layout().clone();
        macro_rules! go {
            ($k:literal) => {{
                let w: [u32; $k] = words[..$k].try_into().unwrap();
                b.push_constants(layout, 0, w).unwrap();
            }};
        }
        match n.div_ceil(4) {
            1 => go!(1),
            2 => go!(2),
            3 => go!(3),
            4 => go!(4),
            5 => go!(5),
            6 => go!(6),
            7 => go!(7),
            _ => go!(8),
        }
    }
}

fn dispatch<L>(b: &mut AutoCommandBufferBuilder<L>, groups: [u32; 3]) {
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
        len.max(1),
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

/// The multigrid hierarchy: level dimensions and offsets into the shared level buffers.
struct Levels {
    dims: Vec<[u32; 2]>,
    offs: Vec<u32>,
    total: u32,
    /// First level of the tail: it and every coarser level share one dispatch.
    tail: usize,
    /// Offset of each level's 16 x 16 tile flags (levels 0..=tail), and their total number.
    tile_offs: Vec<u32>,
    tiles: u32,
}

impl Levels {
    fn new(grid: [u32; 2]) -> Levels {
        let mut dims = vec![grid];
        while dims.last().unwrap()[0] * dims.last().unwrap()[1] > COARSEST_CELLS {
            let [w, h] = *dims.last().unwrap();
            dims.push([w.div_ceil(2), h.div_ceil(2)]);
        }
        let mut offs = Vec::new();
        let mut total = 0;
        for d in &dims {
            offs.push(total);
            total += d[0] * d[1];
        }
        let tail = (1..dims.len()).find(|&l| dims[l][0] * dims[l][1] <= TAIL_CELLS).unwrap_or(dims.len() - 1).max(1).min(dims.len() - 1);
        let mut tile_offs = vec![0; dims.len()];
        let mut tiles = 0;
        for l in 0..=tail {
            tile_offs[l] = tiles;
            tiles += dims[l][0].div_ceil(GRID_GROUP) * dims[l][1].div_ceil(GRID_GROUP);
        }
        Levels { dims, offs, total, tail, tile_offs, tiles }
    }

    fn tail_pc(&self, mode: u32) -> TailPc {
        TailPc { first: self.tail as u32, last: self.last() as u32, mode, omega: MG_OMEGA }
    }

    /// Index of the coarsest level.
    fn last(&self) -> usize {
        self.dims.len() - 1
    }

    fn groups(&self, l: usize) -> [u32; 3] {
        [self.dims[l][0].div_ceil(GRID_GROUP), self.dims[l][1].div_ceil(GRID_GROUP), 1]
    }

    fn pc(&self, l: usize, mode: u32) -> MgPc {
        let c = (l + 1).min(self.last());
        MgPc {
            off_f: self.offs[l] as i32,
            off_c: self.offs[c] as i32,
            dim_f: [self.dims[l][0] as i32, self.dims[l][1] as i32],
            dim_c: [self.dims[c][0] as i32, self.dims[c][1] as i32],
            mode,
            omega: MG_OMEGA,
        }
    }

    /// Push constants of a multigrid kernel on level `l`: the level rides in the mode word.
    fn mg(&self, l: usize, flag: bool) -> MgPc {
        self.pc(l, flag as u32 | (l as u32) << 8)
    }
}

struct Passes {
    step: Pass,
    particles: Pass,
    prep: Pass,
    coef: Pass,
    surf_smooth: Pass,
    surf_curv: Pass,
    surf_apply: Pass,
    visc_setup: Pass,
    visc_diag: Pass,
    visc_split: Pass,
    visc_clear: Pass,
    visc_strain: Pass,
    visc_apply: Pass,
    visc_update: Pass,
    visc_dir: Pass,
    rhs: Pass,
    mg_build: Pass,
    mg_down: Pass,
    mg_up: Pass,
    mg_tail: Pass,
    cg_scale: Pass,
    cg_init: Pass,
    /// Even and odd iterations: the two search-direction buffers swap roles.
    cg_apply: [Pass; 2],
    cg_update: [Pass; 2],
    cg_reduce: Pass,
    project: Pass,
    claim: Pass,
    count: Pass,
    stats: Pass,
    sort: Pass,
}

/// One pending host read-back of the step and solver state.
struct Readback {
    host: Subbuffer<[u32]>,
    busy: bool,
    /// Simulated time this batch asked for (to tell the CPU how far it got).
    target: f32,
    /// Resets since creation when it was requested; older ones are stale.
    epoch: u32,
    /// GPU timestamps before and after the batch's substeps, and how many there were.
    pool: Arc<QueryPool>,
    substeps: u32,
}

pub struct FluidSim {
    gpu: Arc<Gpu>,
    pub grid: [u32; 2],
    pub capacity: u32,
    pub params: SimParams,
    /// Latest particle counts, refreshed asynchronously by `request_stats`.
    pub stats: Stats,
    /// What the solvers did; see `SolverStats`.
    pub solver: SolverStats,

    n_slots: u32,
    grid_valid: bool,
    cleared: bool,
    frame: u32,
    substeps_done: u64,
    /// Simulated time of every read-back that has arrived, s.
    sim_time: f64,
    /// Simulated time and substeps before the last reset (the GPU counters restart there).
    time_base: f64,
    substeps_base: u64,
    epoch: u32,
    /// Latest CFL-limited substep length the GPU reported, s.
    dt_estimate: f32,
    /// Smoothed GPU time of one substep, s (0 until the first measurement arrives).
    substep_cost: f32,
    /// Iterations recorded per substep for each solve, adapted to what recent solves used.
    pressure_rec: u32,
    viscosity_rec: u32,
    /// Fastest speed newly spawned particles start with, cells/s (feeds the CFL limit).
    spawn_speed: f32,
    /// Fluids the tank has held since the last reset: water, syrup, oil.
    present: [bool; 3],
    /// Next permanent particle id (see `Particle::id`).
    next_id: u32,

    levels: Levels,
    nwg: u32,

    particles: Subbuffer<[Particle]>,
    /// Scratch copy of the particles that the periodic sort writes into.
    sorted: Subbuffer<[Particle]>,
    since_sort: u32,
    acc: Subbuffer<[u32]>,
    pres: Subbuffer<[[f32; 2]]>,
    solid: Subbuffer<[[f32; 4]]>,
    sdf: Subbuffer<[f32]>,
    /// One flag per 16 x 16 tile: where the implicit viscosity solve runs.
    tiles: Subbuffer<[u32]>,
    spawn: Subbuffer<[Particle]>,
    /// Particles per cell in a window of the grid (`cell_counts`).
    cell_count: Subbuffer<[u32]>,
    counter: Subbuffer<[u32]>,
    step_buf: Subbuffer<[u32]>,
    solve_buf: Subbuffer<[u32]>,
    ind: Subbuffer<[DispatchIndirectCommand]>,
    ind_ready: bool,
    lev: Subbuffer<[[i32; 4]]>,
    ptile: Subbuffer<[u32]>,
    params_buf: Subbuffer<GpuParams>,
    /// Every float field, for the one-off clear.
    fields: Vec<Subbuffer<[u32]>>,

    pass: Passes,
    /// Recorded substeps, by (particle groups, gather, pressure iterations, viscosity iterations).
    substep_cache: std::collections::HashMap<(u32, bool, u32, u32), Arc<RawSecondary>>,

    pending_reset: Option<Vec<Particle>>,
    pending_solids: Option<SolidField>,
    pending_spawn: Vec<Particle>,
    stats_host: Subbuffer<[u32]>,
    stats_in_flight: bool,
    readbacks: Vec<Readback>,
    next_readback: usize,
}

impl FluidSim {
    pub fn new(gpu: Arc<Gpu>, grid: [u32; 2], capacity: u32) -> FluidSim {
        let cells = (grid[0] * grid[1]) as u64;
        let nodes = ((grid[0] + 1) * (grid[1] + 1)) as u64;
        let none = BufferUsage::empty();
        let levels = Levels::new(grid);
        let mg = levels.total as u64;
        let nwg = grid[0].div_ceil(GRID_GROUP) * grid[1].div_ceil(GRID_GROUP);
        let entries = levels.dims.len() as u64 + 4;

        let particles = device_buffer::<Particle>(&gpu, capacity as u64, none);
        let sorted = device_buffer::<Particle>(&gpu, capacity as u64, none);
        let acc = device_buffer::<u32>(&gpu, cells * ACC_STRIDE, none);
        let vel = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let vel0 = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let cell = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let comp = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let coef = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let mga = device_buffer::<[f32; 4]>(&gpu, mg, none);
        let prhs = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let pres = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let solid = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let sdf = device_buffer::<f32>(&gpu, nodes, none);
        let spawn = device_buffer::<Particle>(&gpu, SPAWN_MAX as u64, none);
        let cell_count = device_buffer::<u32>(&gpu, cells, none);
        let counter = device_buffer::<u32>(&gpu, 8, none);
        let step_buf = device_buffer::<u32>(&gpu, STEP_WORDS, none);
        let solve_buf = device_buffer::<u32>(&gpu, SOLVE_WORDS, none);
        let ind = device_buffer::<DispatchIndirectCommand>(&gpu, 2 * entries, BufferUsage::INDIRECT_BUFFER);
        let part = device_buffer::<[f32; 4]>(&gpu, 2 * nwg as u64, none);
        let mgd = device_buffer::<f32>(&gpu, mg, none);
        let mgx = device_buffer::<[f32; 2]>(&gpu, mg, none);
        let mgr = device_buffer::<[f32; 2]>(&gpu, mg, none);
        let mgz = device_buffer::<[f32; 2]>(&gpu, mg, none);
        let cgp = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let cgp2 = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let lev = device_buffer::<[i32; 4]>(&gpu, levels.dims.len() as u64, none);
        let ptile = device_buffer::<u32>(&gpu, levels.tiles as u64, none);
        let cgq = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let vr = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let vp = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let vq = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let vw = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let vs = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let surf = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let pjump = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let tiles = device_buffer::<u32>(&gpu, nwg as u64, none);
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
        let mut put = |k: u32, b: Subbuffer<[u8]>| {
            table.insert(k, b);
        };
        put(0, params_buf.as_bytes().clone());
        put(1, particles.as_bytes().clone());
        put(2, acc.as_bytes().clone());
        put(3, vel.as_bytes().clone());
        put(4, vel0.as_bytes().clone());
        put(5, cell.as_bytes().clone());
        put(6, comp.as_bytes().clone());
        put(7, coef.as_bytes().clone());
        put(8, mga.as_bytes().clone());
        put(9, prhs.as_bytes().clone());
        put(10, pres.as_bytes().clone());
        put(11, solid.as_bytes().clone());
        put(12, sdf.as_bytes().clone());
        put(13, spawn.as_bytes().clone());
        put(14, counter.as_bytes().clone());
        put(15, sorted.as_bytes().clone());
        put(16, step_buf.as_bytes().clone());
        put(17, solve_buf.as_bytes().clone());
        put(18, ind.as_bytes().clone());
        put(19, part.as_bytes().clone());
        put(20, mgd.as_bytes().clone());
        put(21, mgx.as_bytes().clone());
        put(22, mgr.as_bytes().clone());
        put(23, cgp.as_bytes().clone());
        put(24, cgq.as_bytes().clone());
        put(25, vr.as_bytes().clone());
        put(26, vp.as_bytes().clone());
        put(27, vq.as_bytes().clone());
        put(28, vw.as_bytes().clone());
        put(29, vs.as_bytes().clone());
        put(30, surf.as_bytes().clone());
        put(31, pjump.as_bytes().clone());
        put(32, tiles.as_bytes().clone());
        put(33, lev.as_bytes().clone());
        put(34, cgp2.as_bytes().clone());
        put(35, ptile.as_bytes().clone());
        put(36, mgz.as_bytes().clone());
        put(37, cell_count.as_bytes().clone());
        // The same bindings with the two search-direction buffers swapped.
        let mut swapped = table.clone();
        swapped.insert(23, cgp2.as_bytes().clone());
        swapped.insert(34, cgp.as_bytes().clone());

        let fields = vec![
            particles.clone().reinterpret::<[u32]>(),
            vel.reinterpret::<[u32]>(),
            vel0.reinterpret::<[u32]>(),
            cell.reinterpret::<[u32]>(),
            comp.reinterpret::<[u32]>(),
            coef.reinterpret::<[u32]>(),
            mga.reinterpret::<[u32]>(),
            prhs.reinterpret::<[u32]>(),
            pres.clone().reinterpret::<[u32]>(),
            solid.clone().reinterpret::<[u32]>(),
            sdf.clone().reinterpret::<[u32]>(),
            counter.clone(),
            step_buf.clone(),
            solve_buf.clone(),
            part.reinterpret::<[u32]>(),
            mgd.reinterpret::<[u32]>(),
            mgx.reinterpret::<[u32]>(),
            mgr.reinterpret::<[u32]>(),
            mgz.reinterpret::<[u32]>(),
            cgp.reinterpret::<[u32]>(),
            cgp2.reinterpret::<[u32]>(),
            ptile.clone(),
            cgq.reinterpret::<[u32]>(),
            vr.reinterpret::<[u32]>(),
            vp.reinterpret::<[u32]>(),
            vq.reinterpret::<[u32]>(),
            vw.reinterpret::<[u32]>(),
            vs.reinterpret::<[u32]>(),
            surf.reinterpret::<[u32]>(),
            pjump.reinterpret::<[u32]>(),
        ];

        let p = |name: &'static str, s: &[u8]| Pass::new(&gpu, name, s, &table);
        let pass = Passes {
            step: p("step", spirv!("step.comp")),
            particles: p("particles", spirv!("particles.comp")),
            prep: p("grid_prep", spirv!("grid_prep.comp")),
            coef: p("grid_coef", spirv!("grid_coef.comp")),
            surf_smooth: p("surf_smooth", spirv!("surf_smooth.comp")),
            surf_curv: p("surf_curv", spirv!("surf_curv.comp")),
            surf_apply: p("surf_apply", spirv!("surf_apply.comp")),
            visc_setup: p("visc_setup", spirv!("visc_setup.comp")),
            visc_diag: p("visc_diag", spirv!("visc_diag.comp")),
            visc_split: p("visc_split", spirv!("visc_split.comp")),
            visc_clear: p("visc_clear", spirv!("visc_clear.comp")),
            visc_strain: p("visc_strain", spirv!("visc_strain.comp")),
            visc_apply: p("visc_apply", spirv!("visc_apply.comp")),
            visc_update: p("visc_update", spirv!("visc_update.comp")),
            visc_dir: p("visc_dir", spirv!("visc_dir.comp")),
            rhs: p("rhs", spirv!("rhs.comp")),
            mg_build: p("mg_build", spirv!("mg_build.comp")),
            mg_down: p("mg_down", spirv!("mg_down.comp")),
            mg_up: p("mg_up", spirv!("mg_up.comp")),
            mg_tail: p("mg_tail", spirv!("mg_tail.comp")),
            cg_scale: p("cg_scale", spirv!("cg_scale.comp")),
            cg_init: p("cg_init", spirv!("cg_init.comp")),
            cg_apply: [p("cg_apply", spirv!("cg_apply.comp")), Pass::new(&gpu, "cg_apply", spirv!("cg_apply.comp"), &swapped)],
            cg_update: [p("cg_update", spirv!("cg_update.comp")), Pass::new(&gpu, "cg_update", spirv!("cg_update.comp"), &swapped)],
            cg_reduce: p("cg_reduce", spirv!("cg_reduce.comp")),
            project: p("project", spirv!("project.comp")),
            claim: p("claim", spirv!("claim.comp")),
            count: p("count", spirv!("count.comp")),
            stats: p("stats", spirv!("stats.comp")),
            sort: p("sort", spirv!("sort.comp")),
        };
        let readbacks: Vec<Readback> = (0..READBACK_RING)
            .map(|_| Readback {
                host: readback::<u32>(&gpu, STEP_WORDS + SOLVE_WORDS),
                busy: false,
                target: 0.0,
                epoch: 0,
                pool: QueryPool::new(
                    gpu.device.clone(),
                    QueryPoolCreateInfo { query_count: 2, ..QueryPoolCreateInfo::query_type(QueryType::Timestamp) },
                )
                .unwrap(),
                substeps: 0,
            })
            .collect();
        for r in &readbacks {
            r.host.write().unwrap().fill(0);
        }

        FluidSim {
            pass,
            substep_cache: Default::default(),
            stats_host: readback::<u32>(&gpu, 8),
            gpu,
            grid,
            capacity,
            params: SimParams::default(),
            stats: Stats::default(),
            solver: SolverStats::default(),
            n_slots: 0,
            grid_valid: false,
            cleared: false,
            frame: 0,
            substeps_done: 0,
            sim_time: 0.0,
            time_base: 0.0,
            substeps_base: 0,
            epoch: 0,
            dt_estimate: NOMINAL_DT,
            substep_cost: 0.0,
            pressure_rec: 24,
            viscosity_rec: 24,
            spawn_speed: 0.0,
            present: [false; 3],
            next_id: 0,
            levels,
            nwg,
            particles,
            sorted,
            since_sort: 0,
            acc,
            pres,
            solid,
            sdf,
            tiles,
            spawn,
            cell_count,
            counter,
            step_buf,
            solve_buf,
            ind,
            ind_ready: false,
            lev,
            ptile,
            params_buf,
            fields,
            pending_reset: None,
            pending_solids: None,
            pending_spawn: Vec::new(),
            stats_in_flight: false,
            readbacks,
            next_readback: 0,
        }
    }

    pub fn gpu(&self) -> Arc<Gpu> {
        self.gpu.clone()
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

    /// Substeps recorded since the sim was created.
    pub fn substeps_done(&self) -> u64 {
        self.substeps_done
    }

    /// Simulated time covered by every batch whose read-back has arrived, s.
    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// Substep length the next batch will most likely get, s.
    pub fn dt_estimate(&self) -> f32 {
        if self.params.fixed_hz > 0.0 { 1.0 / self.params.fixed_hz } else { self.dt_estimate }
    }

    /// GPU time one substep currently takes, s, measured with timestamps around each batch
    /// and smoothed over about ten batches. 0 before the first measurement.
    pub fn substep_cost(&self) -> f32 {
        self.substep_cost
    }

    /// Replaces the solid geometry. Takes effect at the next `record`.
    pub fn set_solids(&mut self, field: SolidField) {
        self.pending_solids = Some(field);
    }

    /// Replaces every particle. Takes effect at the next `record`.
    pub fn reset(&mut self, mut particles: Vec<Particle>) {
        particles.truncate(self.capacity as usize);
        for (k, p) in particles.iter_mut().enumerate() {
            p.id = k as u32;
        }
        self.next_id = particles.len() as u32;
        self.present = [false; 3];
        self.note_materials(&particles);
        self.pending_spawn.clear();
        self.stats = Stats { live: particles.len() as u32, ..Default::default() };
        self.pending_reset = Some(particles);
    }

    /// Queues new particles; they appear at the next `record`. Returns how many were
    /// accepted (the queue holds `SPAWN_MAX` per frame).
    pub fn spawn(&mut self, particles: &[Particle]) -> usize {
        let room = SPAWN_MAX - self.pending_spawn.len();
        let particles = &particles[..particles.len().min(room)];
        for p in particles {
            self.spawn_speed = self.spawn_speed.max(p.vel[0].hypot(p.vel[1]));
        }
        self.note_materials(particles);
        for p in particles {
            self.pending_spawn.push(Particle { id: self.next_id, ..*p });
            self.next_id = self.next_id.wrapping_add(1);
        }
        particles.len()
    }

    /// Remembers which fluids the tank has held since the last reset (it picks how viscosity
    /// is integrated).
    fn note_materials(&mut self, particles: &[Particle]) {
        for p in particles {
            if p.kind == KIND_OIL {
                self.present[2] = true;
            } else {
                self.present[0] |= p.s < 1.0;
                self.present[1] |= p.s > 0.0;
            }
        }
    }

    fn gpu_params(&self, tool: ToolState, spawn_count: u32) -> GpuParams {
        let p = &self.params;
        let u = p.gpu_units();
        let rad = |deg: f32| deg.to_radians();
        let cs = |deg: f32| [rad(deg).cos(), rad(deg).sin()];
        let [cw, sw] = cs(p.contact_angle[0]);
        let [cy, sy] = cs(p.contact_angle[1]);
        let [co, so] = cs(p.contact_angle[2]);
        GpuParams {
            grid: [self.grid[0] as i32, self.grid[1] as i32],
            n_slots: self.n_slots,
            spawn_count,
            dt_fixed: if p.fixed_hz > 0.0 { 1.0 / p.fixed_hz } else { 0.0 },
            gravity: u.gravity,
            inv_ppc: 1.0 / PPC as f32,
            fluid_thr: p.fluid_thr,
            rho: [p.rho[0], p.rho[1], p.rho[2], 0.0],
            nu: [u.nu[0], u.nu[1], u.nu[2], 0.0],
            sigma: [u.sigma[0], u.sigma[1], u.sigma[2], u.sigma_oil],
            contact_aq: [cw, sw, cy, sy],
            contact_oil: [co, so, p.kappa_max, 0.0],
            sep_tau: p.droplet,
            cfl: p.cfl,
            diffuse: p.diffuse,
            density_k: p.density_k,
            dt_max: p.dt_max.min(p.dt_capillary()),
            rest_phi: p.rest_phi,
            ext_speed: tool.speed().max(self.spawn_speed),
            wall_margin: 0.1,
            tool_pos: tool.pos,
            tool_vel: tool.vel,
            tool_radius: tool.radius.max(0.01),
            tool_mode: tool.mode,
            tool_strength: tool.strength,
            frame: self.frame,
            tol: [p.pressure_tol, p.correction_tol, p.viscosity_tol, p.density_tau],
            ext: [p.speed_cap_cells(self.grid[1]), p.advection_order as f32, p.spray_cfl, 0.0],
        }
    }

    fn ind_slice(&self, entry: usize) -> Subbuffer<[DispatchIndirectCommand]> {
        self.ind.clone().slice(entry as u64..entry as u64 + 1)
    }

    /// Indirect entries: one per level, the coarse solve and the scalar reductions of the
    /// pressure solve; then the viscosity grid and its reductions.
    fn entries(&self) -> u32 {
        self.levels.dims.len() as u32 + 4
    }

    fn coarse_entry(&self) -> usize {
        self.levels.dims.len()
    }

    fn visc_entry(&self) -> usize {
        self.levels.dims.len() + 2
    }

    fn reduce(&self, b: &mut Recorder, mode: u32) {
        let pressure = mode < 4;
        let n_entries = self.entries();
        let (first, last) = if pressure { (0, n_entries - 2) } else { (n_entries - 2, n_entries) };
        self.pass.cg_reduce.bind_raw(b);
        self.pass.cg_reduce.push_raw(
            b,
            ReducePc { mode, n: self.nwg, n_entries, first_entry: first, last_entry: last, pad: [0; 3] },
        );
        // The reductions inside the iteration are skipped too once their solve has converged
        // (the last entry of each solve's range); the first and the closing one always run.
        if matches!(mode, 1 | 2 | 5 | 6) {
            b.dispatch_indirect(&self.ind_slice(last as usize - 1));
        } else {
            b.dispatch([1, 1, 1]);
        }
    }

    /// Records `substeps` substeps meant to cover `target` seconds of simulated time
    /// (ignored with a fixed step rate), plus any pending uploads. The CFL limit may make the
    /// batch fall short of `target`; `sim_time` reports what was covered once it is known.
    pub fn record(&mut self, b: &mut Builder, substeps: u32, target: f32, tool: ToolState) {
        let gpu = self.gpu.clone();
        self.frame = self.frame.wrapping_add(1);

        if !self.cleared {
            for f in &self.fields {
                b.fill_buffer(f.clone(), 0).unwrap();
            }
            self.cleared = true;
        }
        if !self.ind_ready {
            let mut cmds = Vec::new();
            for l in 0..self.levels.dims.len() {
                let [x, y, z] = self.levels.groups(l);
                cmds.push(DispatchIndirectCommand { x, y, z });
            }
            cmds.push(DispatchIndirectCommand { x: 1, y: 1, z: 1 });
            cmds.push(DispatchIndirectCommand { x: 1, y: 1, z: 1 });
            let [x, y, z] = self.levels.groups(0);
            cmds.push(DispatchIndirectCommand { x, y, z });
            cmds.push(DispatchIndirectCommand { x: 1, y: 1, z: 1 });
            let both: Vec<_> = cmds.iter().chain(cmds.iter()).copied().collect();
            b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &both), self.ind.clone())).unwrap();
            let lv = &self.levels;
            let table: Vec<[i32; 4]> = (0..lv.dims.len()).map(|l| [lv.offs[l] as i32, lv.dims[l][0] as i32, lv.dims[l][1] as i32, lv.tile_offs[l] as i32]).collect();
            b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &table), self.lev.clone())).unwrap();
            self.ind_ready = true;
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
            b.fill_buffer(self.step_buf.clone(), 0).unwrap();
            // Batches recorded before this point but not read back yet are dropped.
            self.epoch += 1;
            self.time_base = self.sim_time;
            self.substeps_base = self.solver.substeps;
            self.grid_valid = false;
            self.dt_estimate = NOMINAL_DT;
            // The first substeps of a new scene start from no pressure at all.
            self.pressure_rec = self.params.pressure_max_iters;
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
        self.spawn_speed = 0.0;
        if substeps > 0 {
            // Word 1 (cumulative simulated time) is the GPU's own and stays untouched.
            b.update_buffer(self.step_buf.clone().slice(0..1), Box::<[u32]>::from([target.max(0.0).to_bits()])).unwrap();
            b.update_buffer(self.step_buf.clone().slice(2..4), Box::<[u32]>::from([substeps, 0])).unwrap();
        }

        let particle_groups = [self.n_slots.div_ceil(PARTICLE_GROUP).max(1), 1, 1];

        if claim > 0 {
            b.fill_buffer(self.counter.clone(), 0).unwrap();
            self.pass.claim.bind(b);
            dispatch(b, particle_groups);
        }

        // Time the substeps on the GPU when a read-back slot is free to carry the result.
        let timed = (substeps > 0 && !self.readbacks[self.next_readback].busy).then(|| self.readbacks[self.next_readback].pool.clone());
        if let Some(pool) = &timed {
            // SAFETY: the pool is not in use: its slot's previous read-back has arrived.
            unsafe {
                b.reset_query_pool(pool.clone(), 0..2).unwrap();
                b.write_timestamp(pool.clone(), 0, PipelineStage::AllCommands).unwrap();
            }
        }
        for _ in 0..substeps {
            self.since_sort += 1;
            if self.since_sort >= SORT_INTERVAL && self.n_slots >= SORT_MIN_PARTICLES {
                self.record_sort(b, particle_groups);
            }
            let cb = self.substep_buffer(self.grid_valid);
            b.execute_commands(cb).unwrap();
            self.grid_valid = true;
            self.substeps_done += 1;
        }

        if let Some(pool) = &timed {
            unsafe { b.write_timestamp(pool.clone(), 1, PipelineStage::AllCommands) }.unwrap();
        }
        if substeps > 0 {
            self.request_readback(b, target, substeps);
        }
    }

    /// One whole substep, recorded raw (see `raw.rs`) once per combination of particle
    /// count, recorded iteration counts and first-step flag, and replayed every substep.
    fn substep_buffer(&mut self, gather: bool) -> Arc<RawSecondary> {
        let viscous = self.viscosity_mode();
        // Round the counts up so that small changes reuse a cached buffer.
        let quant = |n: u32| n.div_ceil(4) * 4;
        let particle_groups = self.n_slots.div_ceil(PARTICLE_GROUP).max(1);
        let key = (
            particle_groups,
            gather,
            quant(self.pressure_rec),
            if viscous == Viscosity::Implicit { quant(self.viscosity_rec) } else { viscous as u32 },
        );
        if let Some(cb) = self.substep_cache.get(&key) {
            return cb.clone();
        }
        if self.substep_cache.len() >= 32 {
            self.substep_cache.clear();
        }
        let mut r = Recorder::new(&self.gpu);
        self.record_substep(&mut r, gather, particle_groups, key.2, key.3, viscous);
        let cb = r.finish();
        self.substep_cache.insert(key, cb.clone());
        cb
    }

    fn record_substep(
        &self,
        r: &mut Recorder,
        gather: bool,
        particle_groups: u32,
        p_iters: u32,
        v_iters: u32,
        viscous: Viscosity,
    ) {
        let grid_groups = [self.grid[0].div_ceil(GRID_GROUP), self.grid[1].div_ceil(GRID_GROUP), 1];
        let ps = &self.pass;
        ps.step.bind_raw(r);
        r.dispatch([1, 1, 1]);
        r.fill(&self.acc, 0);
        ps.particles.bind_raw(r);
        ps.particles.push_raw(r, gather as u32);
        r.dispatch([particle_groups, 1, 1]);
        for pass in [&ps.prep, &ps.coef, &ps.surf_smooth, &ps.surf_curv, &ps.surf_apply] {
            pass.bind_raw(r);
            r.dispatch(grid_groups);
        }
        match viscous {
            Viscosity::None => {}
            Viscosity::Explicit => self.record_viscosity_explicit(r, grid_groups),
            Viscosity::Implicit => self.record_viscosity(r, grid_groups, v_iters),
        }
        // rhs.comp marks the tiles that hold pressure unknowns.
        r.fill(&self.ptile.clone().slice(0..self.nwg as u64), 0);
        ps.rhs.bind_raw(r);
        r.dispatch(grid_groups);
        self.record_pressure(r, p_iters);
        ps.project.bind_raw(r);
        r.dispatch(grid_groups);
    }

    /// How viscosity is integrated this frame, from the fluids present: implicitly when the
    /// viscous diffusion number nu dt / dx^2 of any of them could exceed 0.05, explicitly with
    /// the very same operator otherwise. Forward Euler on this operator is stable up to
    /// about 0.12, so the explicit path keeps a margin of more than two; it exists because
    /// water (nu dt / dx^2 ~ 1e-4 on 1 cm cells) does not need a Krylov solve.
    fn viscosity_mode(&self) -> Viscosity {
        let dt = self.params.dt_max.min(self.params.dt_capillary());
        let d = (0..3).filter(|&i| self.present[i]).map(|i| self.params.nu_cells(i) * dt).fold(0.0, f32::max);
        if d <= 0.0 {
            Viscosity::None
        } else if d <= 0.05 {
            Viscosity::Explicit
        } else {
            Viscosity::Implicit
        }
    }

    fn record_viscosity_explicit(&self, b: &mut Recorder, grid_groups: [u32; 3]) {
        let ps = &self.pass;
        let vpc = |mode| ViscPc { mode, half_off: self.nwg };
        ps.visc_setup.bind_raw(b);
        b.dispatch(grid_groups);
        ps.visc_strain.bind_raw(b);
        ps.visc_strain.push_raw(b, vpc(0));
        b.dispatch(grid_groups);
        ps.visc_apply.bind_raw(b);
        ps.visc_apply.push_raw(b, vpc(2));
        b.dispatch(grid_groups);
    }

    fn record_viscosity(&self, b: &mut Recorder, grid_groups: [u32; 3], iters: u32) {
        let ps = &self.pass;
        let vpc = |mode| ViscPc { mode, half_off: self.nwg };
        // Weights and active tiles, the strain of u* everywhere, the explicit step in tiles
        // too weakly viscous to need the implicit solve, then the solve on the rest.
        b.fill(&self.tiles, 0);
        ps.visc_setup.bind_raw(b);
        b.dispatch(grid_groups);
        ps.visc_strain.bind_raw(b);
        ps.visc_strain.push_raw(b, vpc(0));
        b.dispatch(grid_groups);
        for pass in [&ps.visc_split, &ps.visc_clear, &ps.visc_diag] {
            pass.bind_raw(b);
            b.dispatch(grid_groups);
        }
        ps.visc_apply.bind_raw(b);
        ps.visc_apply.push_raw(b, vpc(0));
        b.dispatch(grid_groups);
        self.reduce(b, 4);
        let e = self.visc_entry();
        for _ in 0..iters {
            ps.visc_strain.bind_raw(b);
            ps.visc_strain.push_raw(b, vpc(1));
            b.dispatch_indirect(&self.ind_slice(e));
            ps.visc_apply.bind_raw(b);
            ps.visc_apply.push_raw(b, vpc(1));
            b.dispatch_indirect(&self.ind_slice(e));
            self.reduce(b, 5);
            ps.visc_update.bind_raw(b);
            ps.visc_update.push_raw(b, vpc(1));
            b.dispatch_indirect(&self.ind_slice(e));
            self.reduce(b, 6);
            ps.visc_dir.bind_raw(b);
            ps.visc_dir.push_raw(b, vpc(1));
            b.dispatch_indirect(&self.ind_slice(e));
        }
        self.reduce(b, 7);
    }

    /// One V-cycle on the residual in level 0 of mgr, leaving the result in mgx. With
    /// `indirect`, every dispatch is skipped once the solve has converged. Levels from
    /// `levels.tail` down are one dispatch (`mg_tail.comp`).
    fn record_vcycle(&self, b: &mut Recorder, indirect: bool) {
        let ps = &self.pass;
        let lv = &self.levels;
        let go = |b: &mut Recorder, l: usize| {
            if indirect { b.dispatch_indirect(&self.ind_slice(l)) } else { b.dispatch(lv.groups(l)) }
        };
        for l in 0..lv.tail {
            ps.mg_down.bind_raw(b);
            ps.mg_down.push_raw(b, lv.mg(l, false));
            go(b, l);
        }
        ps.mg_tail.bind_raw(b);
        ps.mg_tail.push_raw(b, lv.tail_pc(0));
        if indirect { b.dispatch_indirect(&self.ind_slice(self.coarse_entry())) } else { b.dispatch([1, 1, 1]) }
        for l in (0..lv.tail).rev() {
            ps.mg_up.bind_raw(b);
            ps.mg_up.push_raw(b, lv.mg(l, l == 0));
            go(b, l);
        }
    }

    fn record_pressure(&self, b: &mut Recorder, iters: u32) {
        let ps = &self.pass;
        let lv = &self.levels;
        let fine = [self.grid[0].div_ceil(GRID_GROUP), self.grid[1].div_ceil(GRID_GROUP), 1];
        for l in 1..=lv.tail {
            ps.mg_build.bind_raw(b);
            ps.mg_build.push_raw(b, lv.mg(l - 1, false));
            b.dispatch(lv.groups(l));
        }
        if lv.tail < lv.last() {
            ps.mg_tail.bind_raw(b);
            ps.mg_tail.push_raw(b, lv.tail_pc(1));
            b.dispatch([1, 1, 1]);
        }
        let half = self.nwg;
        ps.cg_scale.bind_raw(b);
        ps.cg_scale.push_raw(b, lv.pc(0, 0));
        b.dispatch(fine);
        ps.cg_init.bind_raw(b);
        ps.cg_init.push_raw(b, lv.pc(0, half));
        b.dispatch(fine);
        self.record_vcycle(b, false);
        self.reduce(b, 0);
        for k in 0..iters as usize {
            // p = z + beta p (beta = 0 on the first pass), q = A p.
            let apply = &ps.cg_apply[k % 2];
            apply.bind_raw(b);
            apply.push_raw(b, lv.pc(0, half));
            b.dispatch_indirect(&self.ind_slice(0));
            self.reduce(b, 1);
            let update = &ps.cg_update[k % 2];
            update.bind_raw(b);
            update.push_raw(b, lv.pc(0, 0));
            b.dispatch_indirect(&self.ind_slice(0));
            self.record_vcycle(b, true);
            self.reduce(b, 2);
        }
        self.reduce(b, 3);
    }

    fn request_readback(&mut self, b: &mut Builder, target: f32, substeps: u32) {
        let k = self.next_readback;
        if self.readbacks[k].busy {
            // Every slot still in flight. Nothing is lost: simulated time is cumulative on
            // the GPU and the statistics keep accumulating until the next read-back.
            return;
        }
        self.next_readback = (k + 1) % READBACK_RING;
        let host = self.readbacks[k].host.clone();
        b.copy_buffer(CopyBufferInfo::buffers(self.step_buf.clone(), host.clone().slice(0..STEP_WORDS))).unwrap();
        b.copy_buffer(CopyBufferInfo::buffers(self.solve_buf.clone(), host.slice(STEP_WORDS..STEP_WORDS + SOLVE_WORDS)))
            .unwrap();
        b.fill_buffer(self.solve_buf.clone().slice(SOLVE_STATS..SOLVE_WORDS), 0).unwrap();
        self.readbacks[k].busy = true;
        self.readbacks[k].target = target;
        self.readbacks[k].epoch = self.epoch;
        self.readbacks[k].substeps = substeps;
    }

    /// Picks up every finished read-back: simulated time, the substep length, and solver
    /// statistics, which also set how many iterations the next batches record.
    pub fn poll_readbacks(&mut self) {
        for k in 0..READBACK_RING {
            let slot = (self.next_readback + k) % READBACK_RING;
            if !self.readbacks[slot].busy {
                continue;
            }
            let words: Vec<u32> = match self.readbacks[slot].host.read() {
                Ok(d) => d.to_vec(),
                Err(_) => continue,
            };
            self.readbacks[slot].busy = false;
            // A frame whose submission failed (the swapchain went out of date, say) never
            // ran its copy. The slot is kept zeroed between uses, and the substep counter
            // of a batch that did run is never zero, so such a read-back is recognisable.
            if let Ok(mut d) = self.readbacks[slot].host.write() {
                d.fill(0);
            }
            if words[9] == 0 {
                continue;
            }
            let mut ticks = [0u64; 2];
            if let Ok(true) = self.readbacks[slot].pool.get_results(0..2, &mut ticks, QueryResultFlags::empty())
                && ticks[1] > ticks[0]
            {
                let period = self.gpu.device.physical_device().properties().timestamp_period as f64 * 1e-9;
                let cost = ((ticks[1] - ticks[0]) as f64 * period / self.readbacks[slot].substeps.max(1) as f64) as f32;
                self.substep_cost = if self.substep_cost > 0.0 { self.substep_cost + 0.1 * (cost - self.substep_cost) } else { cost };
            }
            if self.readbacks[slot].epoch != self.epoch {
                continue;
            }
            let f = |i: usize| f32::from_bits(words[i]);
            let s = STEP_WORDS as usize;
            // Cumulative simulated time since the last reset, compensated sum (hi + lo).
            let t = self.time_base + f(1) as f64 - f(11) as f64;
            if t > self.sim_time {
                self.sim_time = t;
            }
            let st = &mut self.solver;
            st.substeps = self.substeps_base + words[9] as u64;
            st.dt = f(4);
            st.dt_cfl = f(8);
            st.umax = f(10);
            st.umax_grid = f(6);
            st.umax_part = f(7);
            let p_it = words[s + 21];
            let v_it = words[s + 22];
            let p_un = words[s + 23] as u64;
            let v_un = words[s + 24] as u64;
            st.pressure_iters_max = st.pressure_iters_max.max(p_it);
            st.viscosity_iters_max = st.viscosity_iters_max.max(v_it);
            st.pressure_unconverged += p_un;
            st.viscosity_unconverged += v_un;
            st.pressure_residual_max = st.pressure_residual_max.max(f(s + 26));
            st.viscosity_residual_max = st.viscosity_residual_max.max(f(s + 27));
            if st.dt_cfl > 0.0 {
                self.dt_estimate = st.dt_cfl;
            }
            // Record a little more than recent solves needed; grow fast when they ran out.
            // The margin grows with the count: a viscosity solve of 35 iterations varies by
            // several from one substep to the next, a pressure solve of 9 by one or two.
            let adapt = |rec: &mut u32, used: u32, unconverged: u64, max: u32| {
                *rec = if unconverged > 0 { (*rec + *rec / 2 + 2).min(max) } else { (used + used / 2 + 2).clamp(2, max) };
            };
            adapt(&mut self.pressure_rec, p_it, p_un, self.params.pressure_max_iters);
            adapt(&mut self.viscosity_rec, v_it, v_un, self.params.viscosity_max_iters);
        }
    }

    /// Runs one substep with a GPU timestamp after every command and returns the time per
    /// pass (ms), summed over all its dispatches, plus the total. Blocking; for `profile`.
    pub fn profile_passes(&mut self) -> Vec<(&'static str, f64)> {
        let viscous = self.viscosity_mode();
        let quant = |n: u32| n.div_ceil(4) * 4;
        let mut r = Recorder::new(&self.gpu).with_timing(&self.gpu, 4096);
        let groups = self.n_slots.div_ceil(PARTICLE_GROUP).max(1);
        self.record_substep(&mut r, true, groups, quant(self.pressure_rec), quant(self.viscosity_rec), viscous);
        let (pool, labels) = r.timing().unwrap();
        let cb = r.finish();
        let mut b = self.gpu.builder();
        let dt = self.dt_estimate();
        b.update_buffer(self.step_buf.clone().slice(0..1), Box::<[u32]>::from([dt.to_bits()])).unwrap();
        b.update_buffer(self.step_buf.clone().slice(2..4), Box::<[u32]>::from([1, 0])).unwrap();
        b.execute_commands(cb).unwrap();
        submit_and_wait(&self.gpu, b);
        let n = labels.len();
        let mut ticks = vec![0u64; n];
        pool.get_results(0..n as u32, &mut ticks, vulkano::query::QueryResultFlags::WAIT).unwrap();
        let period = self.gpu.device.physical_device().properties().timestamp_period as f64 * 1e-6;
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for k in 1..n {
            let ms = (ticks[k].saturating_sub(ticks[k - 1])) as f64 * period;
            match out.iter_mut().find(|e| e.0 == labels[k]) {
                Some(e) => e.1 += ms,
                None => out.push((labels[k], ms)),
            }
        }
        out.push(("total", (ticks[n - 1] - ticks[0]) as f64 * period));
        out
    }

    /// Times `n` trivial dispatches (ms each), to measure the fixed cost of a dispatch.
    pub fn dispatch_floor(&mut self, n: u32, rebind: bool, fine: bool) -> f64 {
        let mut r = Recorder::new(&self.gpu);
        let fine_groups = [self.grid[0].div_ceil(GRID_GROUP), self.grid[1].div_ceil(GRID_GROUP), 1];
        for k in 0..n {
            if fine {
                if rebind || k == 0 {
                    self.pass.cg_scale.bind_raw(&mut r);
                    self.pass.cg_scale.push_raw(&mut r, self.levels.pc(0, 0));
                }
                r.dispatch(fine_groups);
            } else {
                if rebind || k == 0 {
                    self.pass.cg_reduce.bind_raw(&mut r);
                    self.pass.cg_reduce.push_raw(&mut r, ReducePc { mode: 3, n: self.nwg, n_entries: self.entries(), first_entry: 0, last_entry: 0, pad: [0; 3] });
                }
                r.dispatch([1, 1, 1]);
            }
        }
        let cb = r.finish();
        let mut b = self.gpu.builder();
        b.execute_commands(cb).unwrap();
        let t = std::time::Instant::now();
        submit_and_wait(&self.gpu, b);
        t.elapsed().as_secs_f64() * 1e3 / n as f64
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
        self.pass.sort.bind(b);
        let rows = [self.grid[1].div_ceil(PARTICLE_GROUP), 1, 1];
        for (stage, groups) in [particle_groups, rows, [1, 1, 1], particle_groups].into_iter().enumerate() {
            self.pass.sort.push(b, stage as u32);
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
        self.pass.stats.bind(b);
        dispatch(b, [self.n_slots.div_ceil(STATS_RUN).max(1), 1, 1]);
        b.copy_buffer(CopyBufferInfo::buffers(self.counter.clone(), self.stats_host.clone())).unwrap();
        self.stats_in_flight = true;
    }

    /// Picks up the result of `request_stats` if the GPU is done with it.
    pub fn poll_stats(&mut self) -> bool {
        self.poll_readbacks();
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

    /// Advances exactly `seconds` of simulated time (CFL-limited substeps), blocking.
    pub fn advance_blocking(&mut self, seconds: f64, tool: ToolState) {
        let start = self.sim_time;
        let goal = start + seconds;
        let mut first = true;
        loop {
            let left = goal - self.sim_time;
            let dt = self.dt_estimate() as f64;
            // A sliver left over from rounding must not become a substep of its own: momentum
            // travels through the grid as fixed-point displacement per substep, so a step of
            // a microsecond rounds slow flow away (on 1 mm cells, where steps are 0.6 ms,
            // such slivers stopped a capillary oscillation dead). The goal is therefore met
            // to within `SLIVER` of a substep.
            if left <= SLIVER * dt {
                break;
            }
            let n = if self.params.fixed_hz > 0.0 { (left / dt).round().max(1.0) } else { (left / dt).ceil() };
            let n = (n as u32).clamp(1, 16);
            let target = if self.params.fixed_hz > 0.0 { n as f64 * dt } else { left };
            let mut b = self.gpu.builder();
            self.record(&mut b, n, target as f32, if first { tool } else { ToolState { mode: 0, ..tool } });
            submit_and_wait(&self.gpu, b);
            self.poll_readbacks();
            first = false;
            if self.params.fixed_hz > 0.0 && (goal - self.sim_time) < 0.5 * dt {
                break;
            }
        }
    }

    /// Advances `substeps` ticks of `NOMINAL_DT` (1/240 s) of simulated time, blocking.
    /// With 0, only uploads pending changes. The tool applies to the whole interval.
    pub fn step_blocking(&mut self, substeps: u32, tool: ToolState) {
        if substeps == 0 {
            let mut b = self.gpu.builder();
            self.record(&mut b, 0, 0.0, tool);
            submit_and_wait(&self.gpu, b);
            return;
        }
        let seconds = substeps as f64 * NOMINAL_DT as f64;
        let start = self.sim_time;
        loop {
            let left = seconds - (self.sim_time - start);
            let dt = self.dt_estimate() as f64;
            if left <= SLIVER * dt {
                break;
            }
            let n = ((left / dt).ceil() as u32).clamp(1, 16);
            let target = if self.params.fixed_hz > 0.0 { n as f64 * dt } else { left };
            let mut b = self.gpu.builder();
            self.record(&mut b, n, target as f32, tool);
            submit_and_wait(&self.gpu, b);
            self.poll_readbacks();
        }
    }

    /// Live particles per cell in the window of cells `[lo, hi)` (inside the grid), row by
    /// row, including those queued by `spawn` but not uploaded yet. Waits for the GPU, but
    /// reads back only the window's counts.
    pub fn cell_counts(&self, lo: [i32; 2], hi: [i32; 2]) -> Vec<u32> {
        assert!(lo[0] >= 0 && lo[1] >= 0 && hi[0] <= self.grid[0] as i32 && hi[1] <= self.grid[1] as i32);
        let size = [(hi[0] - lo[0]).max(0), (hi[1] - lo[1]).max(0)];
        let n = (size[0] * size[1]) as u64;
        if n == 0 {
            return Vec::new();
        }
        let window = self.cell_count.clone().slice(0..n);
        let host = readback::<u32>(&self.gpu, n);
        let mut b = self.gpu.builder();
        b.fill_buffer(window.clone(), 0).unwrap();
        if self.n_slots > 0 {
            self.pass.count.bind(&mut b);
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct CountPc {
                origin: [i32; 2],
                size: [i32; 2],
                n_slots: u32,
            }
            self.pass.count.push(&mut b, CountPc { origin: lo, size, n_slots: self.n_slots });
            dispatch(&mut b, [self.n_slots.div_ceil(PARTICLE_GROUP), 1, 1]);
        }
        b.copy_buffer(CopyBufferInfo::buffers(window, host.clone())).unwrap();
        submit_and_wait(&self.gpu, b);
        let mut counts = host.read().unwrap().to_vec();
        for p in &self.pending_spawn {
            let c = [p.pos[0].floor() as i32 - lo[0], p.pos[1].floor() as i32 - lo[1]];
            if c[0] >= 0 && c[1] >= 0 && c[0] < size[0] && c[1] < size[1] {
                counts[(c[1] * size[0] + c[0]) as usize] += 1;
            }
        }
        counts
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

    /// Reads back the pressure potential (p * dt, per cell) and the latest substep length,
    /// for checks against analytical pressures.
    pub fn download_pressure(&self) -> (Vec<f32>, f32) {
        let n = (self.grid[0] * self.grid[1]) as u64;
        let host = readback::<[f32; 2]>(&self.gpu, n);
        let mut b = self.gpu.builder();
        b.copy_buffer(CopyBufferInfo::buffers(self.pres.clone(), host.clone())).unwrap();
        submit_and_wait(&self.gpu, b);
        let data = host.read().unwrap();
        let p = data.iter().map(|v| v[0]).collect();
        (p, self.solver.dt)
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
