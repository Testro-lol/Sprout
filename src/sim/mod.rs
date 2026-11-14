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

pub mod material;
pub mod params;
mod raw;

use std::{collections::BTreeMap, sync::Arc};

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, DispatchIndirectCommand, PrimaryCommandBufferAbstract},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::EntryPoint,
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

/// Ints per cell in the scatter accumulators: at least `ACC_MIN` (four face sums, the cell
/// weight, three channels), at most `ACC_MAX` (what the buffer is sized for). The stride in
/// use follows the materials in the tank (`Tank`), so a tank of water, syrup and oil moves
/// as much memory as it always did.
const ACC_MIN: u32 = 8;
const ACC_MAX: u32 = 32;
/// First accumulator channel materials can be given.
const ACC_FIRST: u32 = 5;
/// Words of the counter buffer: 8 general ones, then two per table row (`stats.comp`).
const COUNTER_WORDS: u64 = 8 + 2 * MAX_MATERIALS as u64;
/// Solidus / liquidus of a material that never freezes, as the shaders see it.
const NO_MELT: f32 = -1.0e9;
/// Frozen bodies the rigid projection handles per substep (`MAX_BODIES` in rigid.comp).
const RIGID_BODIES: u64 = 4096;
/// Words per body in the body table (`BODY_WORDS` in common.glsl).
const BODY_WORDS: u64 = 32;
/// Words of the heat ledger (`LedgerBuf` in common.glsl).
const LEDGER_WORDS: u64 = 16 + 2 * MAX_MATERIALS as u64;
/// How fast a liquid with `stain` 1 stains a solid it lies against, and the back wall, 1/s;
/// how fast liquid that does not stain washes a stain off, at rest, 1/s (stain.comp).
const STAIN_WALL_RATE: f32 = 3.0;
const STAIN_BACK_RATE: f32 = 1.2;
const STAIN_WASH_RATE: f32 = 0.06;
/// Layers of the per-cell optics image (cell_optics.comp writes them, composite.frag reads them).
const CELL_OPTICS_LAYERS: u32 = 9;
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
/// Host read-back slots in flight.
const READBACK_RING: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub live: u32,
    /// Per table row: particles of that material, and the sum of their concentrations.
    pub count: [u32; MAX_MATERIALS],
    pub conc: [f32; MAX_MATERIALS],
}

impl Default for Stats {
    fn default() -> Self {
        Stats { live: 0, count: [0; MAX_MATERIALS], conc: [0.0; MAX_MATERIALS] }
    }
}

impl Stats {
    /// Volume of material `m` in the tank, in particles: its own particles weighted by
    /// their concentration, plus the diluted remainder of the particles it is the diluent of.
    pub fn volume(&self, m: Material, p: &SimParams) -> f32 {
        let own = &p.mat[m.index()];
        let mut v = if own.diluent >= 0 { self.conc[m.index()] } else { self.count[m.index()] as f32 };
        for r in 0..p.mat_count {
            if r != m.index() && p.mat[r].diluent == m.0 as i32 {
                v += self.count[r] as f32 - self.conc[r];
            }
        }
        v
    }

}

/// Per-particle viscoelastic stress and the target the periodic sort moves it into.
type StressBuffers = (Subbuffer<[[f32; 4]]>, Subbuffer<[[f32; 4]]>);
/// What a recorded substep depends on (`substep_buffer`): particle groups, first-step flag,
/// pressure and viscosity iterations, accumulator stride, CAP_* bits.
type SubstepKey = (u32, bool, u32, u32, u32, u32);

/// What crossed the boundary of the simulation since the last reset. Energies in J per metre
/// of depth (the simulation is a slice; `SimParams::cell_heat`).
#[derive(Clone, Copy, Debug)]
pub struct Ledger {
    pub from_air: f64,
    pub to_air: f64,
    pub from_solids: f64,
    pub to_solids: f64,
    pub heated: f64,
    pub cooled: f64,
    /// Latent and sensible heat carried off by liquid that boiled away.
    pub boiled: f64,
    /// Per table row: particles that boiled away, and particles left behind as its residue.
    pub boiled_count: [u32; MAX_MATERIALS],
    pub residue_count: [u32; MAX_MATERIALS],
}

impl Default for Ledger {
    fn default() -> Self {
        Ledger {
            from_air: 0.0,
            to_air: 0.0,
            from_solids: 0.0,
            to_solids: 0.0,
            heated: 0.0,
            cooled: 0.0,
            boiled: 0.0,
            boiled_count: [0; MAX_MATERIALS],
            residue_count: [0; MAX_MATERIALS],
        }
    }
}

impl Ledger {
    /// Net energy that entered the liquid, J/m.
    pub fn net_in(&self) -> f64 {
        self.from_air - self.to_air + self.from_solids - self.to_solids + self.heated - self.cooled - self.boiled
    }
}

/// Heat content of the particles, J per metre of depth: sum of volumetric heat capacity x
/// (enthalpy temperature + 273.15) x particle volume. Includes latent heat (it is part of the
/// enthalpy temperature).
pub fn heat_content(particles: &[Particle], p: &SimParams) -> f64 {
    let unit = p.cell_heat() / PPC as f64;
    particles.iter().filter_map(|q| q.material().map(|m| (m, q))).map(|(m, q)| p.heat_capacity_rel(m, q.s) as f64 * (q.t as f64 + 273.15) * unit).sum()
}

/// What the tank holds and how the grid accounts for it: the materials present since the
/// last reset (with their diluents and residues), and the accumulator channel each was given.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tank {
    /// Table rows, ascending.
    pub rows: Vec<usize>,
    /// Per row in `rows`: channel of its weight (-1: the remainder), channel of its
    /// weight x concentration (-1: none), family index (-1: none).
    pub chan: Vec<[i32; 3]>,
    pub stride: u32,
    pub families: u32,
    pub group_mask: u32,
    /// Row of the primary family (-1 none): the first concentrate of the first row.
    pub primary: i32,
    /// Channel of the temperature sum (-1: the tank is isothermal).
    pub chan_t: i32,
    /// Per phase group, the channel of the temperature sum of that group's particles alone
    /// (-1: none; the first group present takes what the others leave of `chan_t`). Each
    /// group's viscosity, density and frozen share are evaluated at its own temperature.
    pub chan_g: [i32; GROUPS],
    /// First of the three channels of the viscoelastic stress (-1: nothing elastic).
    pub chan_e: i32,
}

impl Tank {
    /// Lays the channels out for `rows` (ascending). The first row's weight is the
    /// remainder of the cell weight and costs no channel, so water, the usual first row,
    /// scatters exactly what it always did. Returns `None` if the rows do not fit.
    pub fn layout(rows: &[usize], p: &SimParams, thermal: bool) -> Option<Tank> {
        let elastic = rows.iter().any(|&r| p.mat[r].shear_modulus > 0.0);
        if rows.len() > MAX_PRESENT {
            return None;
        }
        let mut next = ACC_FIRST as i32;
        let mut group_mask = 0;
        // The primary family: the first concentrate of the first row. It shares the first
        // row's weight (the remainder of the cell weight) and is family 0.
        let primary = rows.iter().position(|&r| !rows.is_empty() && p.mat[r].diluent == rows[0] as i32);
        let mut families = primary.is_some() as i32;
        let mut chan = Vec::new();
        for (k, &r) in rows.iter().enumerate() {
            let m = &p.mat[r];
            group_mask |= 1 << m.group.min(GROUPS as u32 - 1);
            let mut c = [-1; 3];
            if k > 0 && Some(k) != primary {
                c[0] = next;
                next += 1;
            }
            if m.diluent >= 0 {
                c[1] = next;
                next += 1;
                if Some(k) == primary {
                    c[2] = 0;
                } else {
                    c[2] = families;
                    families += 1;
                }
            }
            chan.push(c);
        }
        // Capability channels after the materials': the temperature sum, the stress.
        let chan_t = if thermal { next } else { -1 };
        next += thermal as i32;
        // A cell where 1200 °C lava meets 20 °C water has a mean temperature of ~500 °C; at
        // that temperature the lava would be frozen and the water past its boiling point.
        // So each phase group after the first gets its own temperature sum.
        let mut chan_g = [-1; GROUPS];
        if thermal {
            let mut first = true;
            for (g, c) in chan_g.iter_mut().enumerate() {
                if group_mask & (1 << g) != 0 {
                    if !first {
                        *c = next;
                        next += 1;
                    }
                    first = false;
                }
            }
        }
        let chan_e = if elastic { next } else { -1 };
        next += 3 * elastic as i32;
        if next as u32 > ACC_MAX {
            return None;
        }
        let stride = (next as u32).max(ACC_MIN).next_multiple_of(4);
        Some(Tank { rows: rows.to_vec(), chan, stride, families: families as u32, group_mask, primary: primary.map_or(-1, |k| rows[k] as i32), chan_t, chan_g, chan_e })
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
        Pass::with_images(gpu, name, spirv, table, &[])
    }

    /// Like `new`, compiled for the CAP_* bits `caps` (specialisation constant 0, SPEC_CAPS
    /// in common.glsl): the code of every other capability is left out.
    fn for_caps(gpu: &Gpu, name: &'static str, spirv: &[u8], table: &BTreeMap<u32, Subbuffer<[u8]>>, caps: u32) -> Pass {
        Pass::build(gpu, name, gpu.shader_specialized(spirv, &[(0, caps)]), table, &[])
    }

    /// Like `new`, with storage images at the bindings `images` names (the rest from `table`).
    fn with_images(
        gpu: &Gpu,
        name: &'static str,
        spirv: &[u8],
        table: &BTreeMap<u32, Subbuffer<[u8]>>,
        images: &[(u32, Arc<ImageView>)],
    ) -> Pass {
        Pass::build(gpu, name, gpu.shader(spirv), table, images)
    }

    fn build(
        gpu: &Gpu,
        name: &'static str,
        entry: EntryPoint,
        table: &BTreeMap<u32, Subbuffer<[u8]>>,
        images: &[(u32, Arc<ImageView>)],
    ) -> Pass {
        let device = gpu.device.clone();
        let stage = PipelineShaderStageCreateInfo::new(entry);
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
        let writes: Vec<_> = set_layout
            .bindings()
            .keys()
            .map(|&b| match images.iter().find(|(i, _)| *i == b) {
                Some((_, view)) => WriteDescriptorSet::image_view(b, view.clone()),
                None => WriteDescriptorSet::buffer(b, table[&b].clone()),
            })
            .collect();
        let set = DescriptorSet::new(gpu.descriptors.clone(), set_layout, writes, []).unwrap();
        let push_size =
            pipeline.layout().push_constant_ranges().iter().map(|r| r.offset + r.size).max().unwrap_or(0);
        Pass { name, pipeline, set, push_size }
    }

    /// Points the pass at the buffers now in `table` (same pipeline).
    fn rebind(&mut self, gpu: &Gpu, table: &BTreeMap<u32, Subbuffer<[u8]>>) {
        let set_layout = self.pipeline.layout().set_layouts()[0].clone();
        let writes: Vec<_> = set_layout.bindings().keys().map(|&b| WriteDescriptorSet::buffer(b, table[&b].clone())).collect();
        self.set = DescriptorSet::new(gpu.descriptors.clone(), set_layout, writes, []).unwrap();
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

/// A pass built once per combination of CAP_* bits the tank has needed (`Pass::for_caps`),
/// for the hot shaders whose optional paths would otherwise cost every tank registers.
struct CapsPass {
    name: &'static str,
    spirv: &'static [u8],
    variants: Vec<(u32, Pass)>,
}

impl CapsPass {
    fn new(name: &'static str, spirv: &'static [u8]) -> CapsPass {
        CapsPass { name, spirv, variants: Vec::new() }
    }

    /// Builds the variant for `caps` unless it exists (a pipeline compile, once per combination).
    fn ensure(&mut self, gpu: &Gpu, table: &BTreeMap<u32, Subbuffer<[u8]>>, caps: u32) {
        if !self.variants.iter().any(|(c, _)| *c == caps) {
            self.variants.push((caps, Pass::for_caps(gpu, self.name, self.spirv, table, caps)));
        }
    }

    /// The variant for `caps`, which `ensure` must have built.
    fn at(&self, caps: u32) -> &Pass {
        &self.variants.iter().find(|(c, _)| *c == caps).expect("CapsPass::ensure first").1
    }

    fn rebind(&mut self, gpu: &Gpu, table: &BTreeMap<u32, Subbuffer<[u8]>>) {
        for (_, pass) in &mut self.variants {
            pass.rebind(gpu, table);
        }
    }
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
    heat: Pass,
    rigid: Pass,
    rheo: Pass,
    magnet: Pass,
    elastic: Pass,
    settle: Pass,
    cell_optics: Pass,
    optics_blur: Pass,
    glow: Pass,
    stain: Pass,
    particles: CapsPass,
    prep: CapsPass,
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
    /// Currents placed in the world (at most `MAX_CURRENTS`).
    currents: Vec<Current>,
    /// Magnets placed in the world (at most `MAX_MAGNETS`).
    magnets: Vec<Magnet>,
    /// Materials the tank has held since the last reset, and their accumulator channels.
    tank: Tank,
    /// The parameters the material table on the GPU was built from.
    mats_uploaded: Option<(SimParams, Tank)>,
    mats: Subbuffer<[GpuMat]>,
    /// The optics table (material.rs: `optics_table`), uploaded with the material table.
    optics_tab: Subbuffer<[GpuOptics]>,
    /// Per cell, the soft optical fields the renderer shades with (cell_optics.comp): one
    /// texel per cell in each of `CELL_OPTICS_LAYERS` layers, so the renderer can filter them.
    cellopt: Arc<ImageView>,
    /// What `cellopt` was last worked out from: substeps recorded, resets, table uploads,
    /// the fast speed. Frames that run no substep (fast hardware, a paused tank) reuse it.
    optics_key: std::cell::Cell<Option<(u64, u64, u64, u32)>>,
    /// The glow field the renderer samples (glow.comp; half the grid's resolution).
    glow: Arc<ImageView>,
    /// The stain map (stain.comp), and the tank and simulated time it was last updated for;
    /// `None`: to be cleared.
    stain: Arc<ImageView>,
    stain_at: std::cell::Cell<Option<(u64, f64)>>,
    /// Times the material and optics tables were uploaded.
    tables_uploaded: u64,
    /// The tank is not isothermal (since the last reset): heat is carried and conducted.
    thermal: bool,
    solid_t: Subbuffer<[f32]>,
    solids_heated: bool,
    dheat: Subbuffer<[[f32; 4]]>,
    /// Per cell, heat capacity and heat of the particles positioned in it (heat.comp).
    hcell: Subbuffer<[[i32; 2]]>,
    ledger: Subbuffer<[u32]>,
    rbody: Subbuffer<[u32]>,
    /// Face velocities, body labels and cell fields, for `download_grid`.
    dbg_vel: Subbuffer<[[f32; 4]]>,
    dbg_labels: Subbuffer<[[i32; 2]]>,
    dbg_cell: Subbuffer<[[f32; 4]]>,
    ledger_reset: bool,
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
    region: Subbuffer<[u32]>,
    /// Per cell: liquid occupancy, density, viscosity, level (`CellBuf` in common.glsl).
    cell: Subbuffer<[[f32; 4]]>,
    /// Per cell: density, oil share, syrup share, total occupancy (`CompBuf`).
    comp: Subbuffer<[[f32; 4]]>,
    resets: u64,
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
    /// Every buffer by binding (and the same with the search directions swapped), kept to
    /// re-bind passes when an optional buffer comes or goes (`set_elastic`).
    table: BTreeMap<u32, Subbuffer<[u8]>>,
    swapped: BTreeMap<u32, Subbuffer<[u8]>>,
    /// Per-particle viscoelastic stress and its sort target, while the tank holds an
    /// elastic material.
    pstress: Option<StressBuffers>,
    /// Families the settling buffers have room for per cell (0: placeholders).
    settle_fams: u32,
    /// Recorded substeps, by (particle groups, gather, pressure iterations, viscosity iterations).
    substep_cache: std::collections::HashMap<SubstepKey, Arc<RawSecondary>>,

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
        let acc = device_buffer::<u32>(&gpu, cells * ACC_MAX as u64, none);
        let mats = device_buffer::<GpuMat>(&gpu, MAX_MATERIALS as u64, none);
        let aux = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let grho = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let gbase = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let sprop = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let mixb = device_buffer::<f32>(&gpu, cells * (MAX_PRESENT as u64 - 1), none);
        let surf2 = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let therm = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let dheat = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let hcell = device_buffer::<[i32; 2]>(&gpu, cells, none);
        let solid_t = device_buffer::<f32>(&gpu, cells, none);
        let ledger = device_buffer::<u32>(&gpu, LEDGER_WORDS, none);
        let rlabel = device_buffer::<[i32; 2]>(&gpu, cells, none);
        let rbody = device_buffer::<u32>(&gpu, 16 + BODY_WORDS * RIGID_BODIES, none);
        let rh = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let magc = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let estr = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let optics_tab = device_buffer::<GpuOptics>(&gpu, MAX_MATERIALS as u64, none);
        let optics_image = || ImageView::new_default(
            Image::new(
                gpu.memory.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [grid[0], grid[1], 1],
                    array_layers: CELL_OPTICS_LAYERS,
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("out of GPU memory"),
        )
        .unwrap();
        // What cell_optics.comp works out, and the same smoothed (optics_blur.comp), which the
        // renderer samples.
        let cellopt_raw = optics_image();
        let cellopt = optics_image();
        // The glow field (glow.comp), at half the grid's resolution, and the stain map
        // (stain.comp), which lasts as long as the tank.
        let plain_image = |extent: [u32; 2], usage: ImageUsage| ImageView::new_default(
            Image::new(
                gpu.memory.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [extent[0], extent[1], 1],
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED | usage,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("out of GPU memory"),
        )
        .unwrap();
        let half = [grid[0].div_ceil(2), grid[1].div_ceil(2)];
        let glow_a = plain_image(half, ImageUsage::empty());
        let glow_b = plain_image(half, ImageUsage::empty());
        let stain = plain_image(grid, ImageUsage::TRANSFER_DST);
        // Particle stress: placeholders until an elastic material arrives (`set_elastic`).
        let pstress = device_buffer::<[f32; 4]>(&gpu, 1, none);
        // Settling: placeholders until a suspension arrives (`set_settling`).
        let famocc = device_buffer::<f32>(&gpu, 1, none);
        let vel = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let dbg_vel = vel.clone();
        let vel0 = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let cell = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let dbg_cell = cell.clone();
        let comp = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let coef = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let mga = device_buffer::<[f32; 4]>(&gpu, mg, none);
        let prhs = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let pres = device_buffer::<[f32; 2]>(&gpu, cells, none);
        let solid = device_buffer::<[f32; 4]>(&gpu, cells, none);
        let sdf = device_buffer::<f32>(&gpu, nodes, none);
        // One word for the per-substep flags, then one per cell (`RegionBuf`).
        let region = device_buffer::<u32>(&gpu, cells + 1, none);
        let spawn = device_buffer::<Particle>(&gpu, SPAWN_MAX as u64, none);
        let cell_count = device_buffer::<u32>(&gpu, cells, none);
        let counter = device_buffer::<u32>(&gpu, COUNTER_WORDS, none);
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
        put(38, region.as_bytes().clone());
        put(39, mats.as_bytes().clone());
        put(40, aux.as_bytes().clone());
        put(59, grho.as_bytes().clone());
        put(65, gbase.as_bytes().clone());
        put(41, sprop.as_bytes().clone());
        put(42, mixb.as_bytes().clone());
        put(43, surf2.as_bytes().clone());
        put(44, therm.as_bytes().clone());
        put(45, dheat.as_bytes().clone());
        put(61, hcell.as_bytes().clone());
        put(46, solid_t.as_bytes().clone());
        put(47, ledger.as_bytes().clone());
        put(48, rlabel.as_bytes().clone());
        put(49, rbody.as_bytes().clone());
        put(50, rh.as_bytes().clone());
        put(51, magc.as_bytes().clone());
        put(52, pstress.as_bytes().clone());
        put(53, pstress.as_bytes().clone());
        put(54, estr.as_bytes().clone());
        put(57, optics_tab.as_bytes().clone());
        put(55, famocc.as_bytes().clone());
        put(56, famocc.as_bytes().clone());
        // The same bindings with the two search-direction buffers swapped.
        let mut swapped = table.clone();
        swapped.insert(23, cgp2.as_bytes().clone());
        swapped.insert(34, cgp.as_bytes().clone());

        let (cell_field, comp_field) = (cell.clone(), comp.clone());
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
            aux.reinterpret::<[u32]>(),
            sprop.reinterpret::<[u32]>(),
            mixb.reinterpret::<[u32]>(),
            surf2.reinterpret::<[u32]>(),
            therm.clone().reinterpret::<[u32]>(),
            dheat.clone().reinterpret::<[u32]>(),
            hcell.clone().reinterpret::<[u32]>(),
            ledger.clone(),
        ];

        let p = |name: &'static str, s: &[u8]| Pass::new(&gpu, name, s, &table);
        let pass = Passes {
            step: p("step", spirv!("step.comp")),
            heat: p("heat", spirv!("heat.comp")),
            rigid: p("rigid", spirv!("rigid.comp")),
            rheo: p("rheo", spirv!("rheo.comp")),
            magnet: p("magnet", spirv!("magnet.comp")),
            elastic: p("elastic", spirv!("elastic.comp")),
            settle: p("settle", spirv!("settle.comp")),
            cell_optics: Pass::with_images(&gpu, "cell_optics", spirv!("cell_optics.comp"), &table, &[(58, cellopt_raw.clone())]),
            optics_blur: Pass::with_images(&gpu, "optics_blur", spirv!("optics_blur.comp"), &table, &[(58, cellopt_raw.clone()), (60, cellopt.clone())]),
            glow: Pass::with_images(&gpu, "glow", spirv!("glow.comp"), &table, &[(58, cellopt_raw.clone()), (62, glow_a.clone()), (63, glow_b.clone())]),
            stain: Pass::with_images(&gpu, "stain", spirv!("stain.comp"), &table, &[(58, cellopt_raw.clone()), (64, stain.clone())]),
            particles: CapsPass::new("particles", spirv!("particles.comp")),
            prep: CapsPass::new("grid_prep", spirv!("grid_prep.comp")),
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
            table,
            swapped,
            pstress: None,
            settle_fams: 0,
            substep_cache: Default::default(),
            stats_host: readback::<u32>(&gpu, COUNTER_WORDS),
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
            currents: Vec::new(),
            magnets: Vec::new(),
            tank: Tank::default(),
            mats_uploaded: None,
            mats,
            optics_tab,
            cellopt,
            optics_key: std::cell::Cell::new(None),
            glow: glow_b,
            stain,
            stain_at: std::cell::Cell::new(None),
            tables_uploaded: 0,
            thermal: false,
            solid_t,
            solids_heated: false,
            dheat,
            hcell,
            ledger,
            rbody: rbody.clone(),
            dbg_vel,
            dbg_labels: rlabel.clone(),
            dbg_cell,
            ledger_reset: false,
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
            region,
            cell: cell_field,
            comp: comp_field,
            resets: 0,
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

    /// Per cell, as the last substep left it: liquid occupancy in x (`CellBuf`). For the
    /// renderer's wet map (`shaders/wet.frag`).
    pub fn cell_buffer(&self) -> &Subbuffer<[[f32; 4]]> {
        &self.cell
    }

    /// Works out, for the renderer, what the liquid in each cell looks like
    /// (cell_optics.comp), from what the last substep left on the grid. `inv_speed`: 1 / the
    /// speed that counts as fast, cells / s.
    /// Records nothing if the grid has not changed since the last call.
    pub fn record_cell_optics(&self, b: &mut Builder, inv_speed: f32) {
        // A new tank, or stains to be cleaned: nothing is stained.
        let now = self.sim_time();
        let dt = match self.stain_at.get() {
            Some((tank, t)) if tank == self.resets => (now - t).clamp(0.0, 0.25) as f32,
            _ => {
                b.clear_color_image(vulkano::command_buffer::ClearColorImageInfo::image(self.stain.image().clone())).unwrap();
                0.0
            }
        };
        self.stain_at.set(Some((self.resets, now)));
        let key = (self.substeps_done, self.resets, self.tables_uploaded, inv_speed.to_bits());
        if self.optics_key.replace(Some(key)) == Some(key) {
            return;
        }
        let groups = [self.grid[0].div_ceil(GRID_GROUP), self.grid[1].div_ceil(GRID_GROUP), 1];
        self.pass.cell_optics.bind(b);
        self.pass.cell_optics.push(b, inv_speed);
        dispatch(b, groups);
        self.pass.optics_blur.bind(b);
        dispatch(b, groups);
        let half = [self.grid[0].div_ceil(2).div_ceil(GRID_GROUP), self.grid[1].div_ceil(2).div_ceil(GRID_GROUP), 1];
        self.pass.glow.bind(b);
        for dir in [0i32, 1] {
            self.pass.glow.push(b, dir);
            dispatch(b, half);
        }
        self.pass.stain.bind(b);
        self.pass.stain.push(b, [dt, STAIN_WALL_RATE, STAIN_BACK_RATE, STAIN_WASH_RATE]);
        dispatch(b, groups);
    }

    /// Wipes every stain off the solids and the back wall (Display panel).
    pub fn clean_stains(&self) {
        self.stain_at.set(None);
    }

    /// The glow field (glow.comp): an RGBA16F image at half the grid's resolution; rgb the
    /// blurred emission, a the blurred heat ((T - ambient) / 1000 K x occupancy).
    pub fn glow_view(&self) -> &Arc<ImageView> {
        &self.glow
    }

    /// The stain map (stain.comp): one RGBA16F texel per cell, the colour times its cover
    /// and the cover.
    pub fn stain_view(&self) -> &Arc<ImageView> {
        &self.stain
    }

    /// The per-cell optical fields (`record_cell_optics`): an RGBA16F array image, one texel
    /// per cell, `CELL_OPTICS_LAYERS` layers.
    pub fn cell_optics_view(&self) -> &Arc<ImageView> {
        &self.cellopt
    }

    /// The optics table (`OptBuf` in optics.glsl).
    pub fn optics_buffer(&self) -> &Subbuffer<[GpuOptics]> {
        &self.optics_tab
    }

    /// The GPU material table (`MatBuf`), for the renderer.
    pub fn material_buffer(&self) -> &Subbuffer<[GpuMat]> {
        &self.mats
    }

    /// Per cell: tension, contact angle, and in w the majority phase group plus
    /// (row + 1) / 128 of the material with the largest share (`SPropBuf`).
    pub fn surface_buffer(&self) -> Subbuffer<[[f32; 4]]> {
        self.table[&41].clone().reinterpret::<[[f32; 4]]>()
    }

    /// Per cell, as the last substep left it: oil share in y, syrup share in z (`CompBuf`).
    pub fn composition_buffer(&self) -> &Subbuffer<[[f32; 4]]> {
        &self.comp
    }

    /// How many times the tank has been emptied and refilled (`reset`): what the renderer
    /// remembers about the liquid from frame to frame is void when this changes.
    pub fn resets(&self) -> u64 {
        self.resets
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
            p.settle_spawn(&self.params);
        }
        self.next_id = particles.len() as u32;
        self.resets += 1;
        self.tank = Tank::default();
        self.thermal = self.solids_heated;
        self.set_elastic(false);
        self.set_settling(0);
        self.ledger_reset = true;
        particles.retain(|p| p.material().is_some_and(|m| m.index() < self.params.mat_count));
        let fits = self.note_materials(&particles);
        debug_assert!(fits, "a scene holds more materials than the grid has channels for");
        self.pending_spawn.clear();
        self.stats = Stats { live: particles.len() as u32, ..Default::default() };
        self.pending_reset = Some(particles);
    }

    pub fn currents(&self) -> &[Current] {
        &self.currents
    }

    /// Replaces the currents (the first `MAX_CURRENTS` are kept). They act from the next
    /// `record` on and survive a `reset`.
    pub fn set_currents(&mut self, currents: &[Current]) {
        self.currents = currents[..currents.len().min(MAX_CURRENTS)].to_vec();
    }

    pub fn magnets(&self) -> &[Magnet] {
        &self.magnets
    }

    /// Replaces the magnets (the first `MAX_MAGNETS` are kept); like the currents, they
    /// survive a `reset`.
    pub fn set_magnets(&mut self, magnets: &[Magnet]) {
        self.magnets = magnets[..magnets.len().min(MAX_MAGNETS)].to_vec();
    }

    /// Queues new particles; they appear at the next `record`. Returns how many were
    /// accepted (the queue holds `SPAWN_MAX` per frame; and a tank holds `MAX_PRESENT`
    /// materials: particles of one more are refused, see `can_hold`).
    pub fn spawn(&mut self, particles: &[Particle]) -> usize {
        let room = SPAWN_MAX - self.pending_spawn.len();
        let mut particles = particles[..particles.len().min(room)].to_vec();
        for (k, p) in particles.iter_mut().enumerate() {
            p.id = self.next_id.wrapping_add(k as u32);
            p.settle_spawn(&self.params);
        }
        let particles = &particles[..];
        if !self.note_materials(particles) {
            return 0;
        }
        for p in particles {
            self.spawn_speed = self.spawn_speed.max(p.vel[0].hypot(p.vel[1]));
        }
        for p in particles {
            self.pending_spawn.push(Particle { id: self.next_id, kind: p.kind | KIND_FRESH, ..*p });
            self.next_id = self.next_id.wrapping_add(1);
        }
        particles.len()
    }

    /// Adds the materials of `particles` to the tank's list (it sets the accumulator
    /// layout, the time-step limits and how viscosity is integrated). False, and nothing
    /// changes, if they do not fit.
    fn note_materials(&mut self, particles: &[Particle]) -> bool {
        let mut rows = self.tank.rows.clone();
        let mut last = usize::MAX;
        let mut thermal = self.thermal;
        for p in particles {
            if let Some(m) = p.material() {
                thermal |= (p.t - self.params.ambient_temp).abs() > 0.05;
                if m.index() != last {
                    last = m.index();
                    if !rows.contains(&last) {
                        rows.push(last);
                    }
                }
            }
        }
        if rows.len() == self.tank.rows.len() && thermal == self.thermal {
            return true;
        }
        match Tank::layout(&self.params.closure(&rows), &self.params, thermal) {
            Some(t) => {
                self.tank = t;
                self.thermal = thermal;
                self.set_elastic(self.tank.chan_e >= 0);
                let fams = if self.settles() { self.tank.families } else { 0 };
                self.set_settling(fams);
                true
            }
            None => false,
        }
    }

    /// Gives the particles a stress buffer while the tank holds an elastic material, and
    /// takes it away when it no longer does (README, "Viscoelasticity").
    fn set_elastic(&mut self, on: bool) {
        if on == self.pstress.is_some() {
            return;
        }
        let gpu = self.gpu.clone();
        let (a, b) = if on {
            let a = device_buffer::<[f32; 4]>(&gpu, self.capacity as u64, BufferUsage::empty());
            let b = device_buffer::<[f32; 4]>(&gpu, self.capacity as u64, BufferUsage::empty());
            let mut cb = gpu.builder();
            cb.fill_buffer(a.clone().reinterpret::<[u32]>(), 0).unwrap();
            submit_and_wait(&gpu, cb);
            self.pstress = Some((a.clone(), b.clone()));
            (a, b)
        } else {
            self.pstress = None;
            let d = device_buffer::<[f32; 4]>(&gpu, 1, BufferUsage::empty());
            (d.clone(), d)
        };
        for t in [&mut self.table, &mut self.swapped] {
            t.insert(52, a.as_bytes().clone());
            t.insert(53, b.as_bytes().clone());
        }
        let table = self.table.clone();
        self.pass.particles.rebind(&gpu, &table);
        self.pass.sort.rebind(&gpu, &table);
        self.substep_cache.clear();
    }

    /// Gives the grid the per-family settling buffers while a suspension that settles is in
    /// the tank (sized for its number of concentration families), and drops them after.
    fn set_settling(&mut self, fams: u32) {
        if fams == self.settle_fams || (fams > 0 && fams < self.settle_fams) {
            return;
        }
        let gpu = self.gpu.clone();
        let cells = (self.grid[0] * self.grid[1]) as u64;
        let n = (cells * fams as u64).max(1);
        let occ = device_buffer::<f32>(&gpu, n, BufferUsage::empty());
        let ds = device_buffer::<[f32; 2]>(&gpu, n, BufferUsage::empty());
        let mut cb = gpu.builder();
        cb.fill_buffer(ds.clone().reinterpret::<[u32]>(), 0).unwrap();
        cb.fill_buffer(occ.clone().reinterpret::<[u32]>(), 0).unwrap();
        submit_and_wait(&gpu, cb);
        for t in [&mut self.table, &mut self.swapped] {
            t.insert(55, occ.as_bytes().clone());
            t.insert(56, ds.as_bytes().clone());
        }
        let table = self.table.clone();
        self.pass.particles.rebind(&gpu, &table);
        self.pass.prep.rebind(&gpu, &table);
        self.pass.settle.rebind(&gpu, &table);
        self.settle_fams = fams;
        self.substep_cache.clear();
    }

    /// Whether a family in the tank settles (a concentrate with grains).
    fn settles(&self) -> bool {
        let p = &self.params;
        p.gravity > 0.0 && self.tank.rows.iter().any(|&r| p.mat[r].diluent >= 0 && p.mat[r].grain > 0.0)
    }

    /// Whether heat is being simulated (the tank is not isothermal).
    pub fn thermal(&self) -> bool {
        self.thermal
    }

    /// Switches the heat simulation on for the rest of this tank's life (a hot solid, the
    /// heat brush). False if the accumulators have no room for it.
    pub fn enable_heat(&mut self) -> bool {
        if self.thermal {
            return true;
        }
        match Tank::layout(&self.tank.rows, &self.params, true) {
            Some(t) => {
                self.tank = t;
                self.thermal = true;
                true
            }
            None => false,
        }
    }

    /// The material table in grid units, with the tank's channels.
    fn gpu_materials(&self) -> Vec<GpuMat> {
        let p = &self.params;
        let dx = p.cell_size;
        let rho_w = SimParams::WATER_DENSITY;
        // Stresses and moduli (Pa) over water's density, in cells²/s².
        let stress = |pa: f32| pa / (rho_w * dx * dx);
        (0..MAX_MATERIALS)
            .map(|r| {
                let m = &p.mat[r];
                let chan = self.tank.rows.iter().position(|&x| x == r).map_or([-1; 3], |k| self.tank.chan[k]);
                let (sin, cos) = m.contact_angle.to_radians().sin_cos();
                let solid = if m.density_solid > 0.0 { m.density_solid } else { m.density };
                let cvol = m.heat_capacity_vol();
                let mut ln_nu = (m.viscosity / (dx * dx)).max(1e-8).ln();
                let mut rho = m.density;
                if !self.thermal {
                    // An isothermal tank: the temperature law and the frozen state are
                    // constants, folded in here so that the shaders skip them.
                    ln_nu += m.temp_factor_ln(p.ambient_temp);
                    rho *= (1.0 - m.expansion * (p.ambient_temp - m.temp_ref)).clamp(0.7, 1.3);
                    if m.freezes() && m.diluent < 0 && p.ambient_temp < m.melt_hi {
                        let fs = ((m.melt_hi - p.ambient_temp) / (m.melt_hi - m.melt_lo).max(1e-3)).clamp(0.0, 1.0);
                        let frozen = (p.rigid_viscosity / (dx * dx)).ln().max(ln_nu);
                        ln_nu += (frozen - ln_nu) * fs;
                        rho += (solid - rho) * fs;
                    }
                }
                GpuMat {
                    ia: [m.group.min(GROUPS as u32 - 1) as i32, m.diluent, chan[0], chan[1]],
                    ib: [chan[2], m.rheology as i32, m.temp_model as i32, m.residue],
                    a: [rho / rho_w, ln_nu, p.tension_cells(m.tension), m.mix_rate],
                    b: [cos, sin, m.grain / dx, solid / rho_w],
                    rheo: [stress(m.yield_stress), m.flow_index, m.shear_time, m.visc_ratio],
                    temp: [m.temp_b, m.temp_c, m.temp_ref + 273.15, p.tension_cells(m.skin)],
                    th: [cvol / WATER_HEAT, m.conductivity / (WATER_HEAT * dx * dx), m.melt_lo.max(NO_MELT), m.melt_hi.max(m.melt_lo + 0.25).max(NO_MELT)],
                    th2: [if m.freezes() || m.diluent >= 0 { m.latent / m.heat_capacity } else { 0.0 }, m.emissivity, m.boil, m.latent_vap / m.heat_capacity],
                    el: [stress(m.shear_modulus), m.relax_time, m.susceptibility, m.residue_share],
                    misc: [m.freeze_shift, m.load, visc_exponent(r, m.diluent), m.expansion],
                }
            })
            .collect()
    }

    fn gpu_params(&self, tool: ToolState, spawn_count: u32) -> GpuParams {
        let p = &self.params;
        let mut pair = [[0.0; 4]; GROUPS];
        for (a, row) in pair.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                *v = p.tension_cells(p.pair_tension[a][b]);
            }
        }
        let mut present = [[0; 4]; MAX_PRESENT / 4];
        for (k, &r) in self.tank.rows.iter().enumerate() {
            present[k / 4][k % 4] = r as i32;
        }
        let mut fams = [[-1; 4]; MAX_PRESENT];
        for (k, &r) in self.tank.rows.iter().enumerate() {
            if self.tank.chan[k][2] >= 0 {
                fams[self.tank.chan[k][2] as usize] = [r as i32, p.mat[r].diluent, 0, 0];
            }
        }
        let caps = self.caps();
        let dx = p.cell_size;
        let mut magnet = [[0.0; 4]; MAX_MAGNETS];
        let mut magnet_dir = [[0.0; 4]; MAX_MAGNETS];
        for (i, m) in self.magnets.iter().enumerate() {
            magnet[i] = [m.pos[0], m.pos[1], m.radius.max(0.5), m.strength];
            magnet_dir[i] = [m.dir[0], m.dir[1], 0.0, 0.0];
        }
        let (chi_max, sat) = self.tank.rows.iter().map(|&r| (p.mat[r].susceptibility, p.mat[r].saturation)).fold((0.0f32, 0.05f32), |a, b| if b.0 > a.0 { b } else { a });
        let mut current_zone = [[0.0; 4]; MAX_CURRENTS];
        let mut current_vel = [[0.0; 4]; MAX_CURRENTS];
        for (i, c) in self.currents.iter().enumerate() {
            current_zone[i] = [c.pos[0], c.pos[1], c.radius.max(0.01), c.rate];
            current_vel[i] = [c.vel[0], c.vel[1], 0.0, 0.0];
        }
        GpuParams {
            grid: [self.grid[0] as i32, self.grid[1] as i32],
            n_slots: self.n_slots,
            spawn_count,
            dt_fixed: if p.fixed_hz > 0.0 { 1.0 / p.fixed_hz } else { 0.0 },
            gravity: p.g_cells(),
            inv_ppc: 1.0 / PPC as f32,
            apic_clamp: p.apic_clamp,
            pair,
            misc: [p.curvature_max, p.sealed_vent as f32, p.cell_size, if p.rigid_contact != 0 { p.rigid_friction } else { -1.0 }],
            n_present: self.tank.rows.len() as i32,
            acc_stride: self.tank.stride.max(ACC_MIN) as i32,
            n_fam: self.tank.families as i32,
            group_mask: self.tank.group_mask,
            caps,
            primary: if self.tank.rows.is_empty() { -1 } else { self.tank.primary },
            // Byte g: channel + 1 of group g's temperature sum (0: none, see `Tank::chan_g`).
            group_heat: self.tank.chan_g.iter().enumerate().map(|(g, &c)| ((c + 1) as u32 & 255) << (8 * g)).sum(),
            pad0: 0,
            present,
            fams,
            heat: [p.ambient_temp, p.heat_rate, p.air_cooling / (WATER_HEAT * dx), STEFAN_BOLTZMANN / (WATER_HEAT * dx)],
            heat2: [(p.rigid_viscosity / (dx * dx)).ln(), p.heat_power, self.tank.chan_t as f32, p.contact_heat / (WATER_HEAT * dx)],
            sep_tau: p.droplet_cells,
            cfl: p.cfl,
            tool_crowd_max: p.tool_crowd_max,
            density_k: p.correction_max,
            dt_max: self.dt_limit(),
            crowd_jitter: p.crowd_jitter,
            ext_speed: tool.speed().max(self.spawn_speed),
            wall_margin: 0.1,
            tool_pos: tool.pos,
            tool_vel: tool.vel,
            tool_radius: tool.radius.max(0.01),
            tool_mode: tool.mode,
            tool_strength: tool.strength,
            frame: self.frame,
            tol: [p.pressure_tol, p.correction_tol, p.viscosity_tol, p.correction_time],
            ext: [p.speed_cap_cells(self.grid[1]), p.advection_order as f32, p.spray_cfl, self.currents.len() as f32],
            corr: [p.correction_step_max, p.correction_deadband, p.slip_cap as f32, p.implicit_viscosity_min],
            current_zone,
            current_vel,
            mag: [self.magnets.len() as f32, chi_max.max(1e-6), sat, 1.0 / (SimParams::WATER_DENSITY * dx * dx)],
            magnet,
            magnet_dir,
            elastic: [
                self.tank.chan_e as f32,
                self.tank.rows.iter().map(|&r| p.mat[r].shear_modulus).fold(0.0f32, f32::max).max(1e-3) / (SimParams::WATER_DENSITY * dx * dx),
                p.elastic_stress_max,
                p.elastic_step_max,
            ],
        }
    }

    /// Longest substep the tank allows, s: `dt_max` and the explicit limits of what is in it
    /// (capillary waves; heat conduction while the tank is not isothermal).
    fn dt_limit(&self) -> f32 {
        let p = &self.params;
        let mut dt = p.dt_max.min(p.dt_capillary(&self.tank.rows));
        if self.thermal {
            dt = dt.min(p.dt_thermal(&self.tank.rows));
        }
        dt.min(p.dt_elastic(&self.tank.rows))
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
            let region: Vec<u32> = std::iter::once(0).chain(field.region.iter().copied()).collect();
            b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &region), self.region.clone())).unwrap();
            self.solids_heated = !field.temp.is_empty();
            if self.solids_heated {
                b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &field.temp), self.solid_t.clone())).unwrap();
                self.enable_heat();
            } else {
                b.fill_buffer(self.solid_t.clone().reinterpret::<[u32]>(), crate::scene::NO_TEMP.to_bits()).unwrap();
            }
        }
        if let Some(particles) = self.pending_reset.take() {
            self.n_slots = particles.len() as u32;
            if !particles.is_empty() {
                let dst = self.particles.clone().slice(0..particles.len() as u64);
                b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &particles), dst)).unwrap();
            }
            b.fill_buffer(self.pres.clone().reinterpret::<[u32]>(), 0).unwrap();
            b.fill_buffer(self.step_buf.clone(), 0).unwrap();
            b.fill_buffer(self.dheat.clone().reinterpret::<[u32]>(), 0).unwrap();
            if std::mem::take(&mut self.ledger_reset) {
                b.fill_buffer(self.ledger.clone(), 0).unwrap();
            }
            // Batches recorded before this point but not read back yet are dropped.
            self.epoch += 1;
            self.time_base = self.sim_time;
            self.substeps_base = self.solver.substeps;
            self.grid_valid = false;
            self.dt_estimate = NOMINAL_DT;
            // The first substeps of a new scene start from no pressure at all, and a viscous
            // scene from no idea how many viscosity iterations it needs: record the most
            // until read-backs say otherwise.
            self.pressure_rec = self.params.pressure_max_iters;
            self.viscosity_rec = self.params.viscosity_max_iters;
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

        if tool.mode >= 5 {
            self.enable_heat();
        }
        if (self.params.ambient_temp - T_REF).abs() > 0.05 {
            self.enable_heat();
        }
        b.update_buffer(self.params_buf.clone(), Box::new(self.gpu_params(tool, claim))).unwrap();
        // The material table follows the live parameters and the tank's channel layout.
        if self.mats_uploaded.as_ref().is_none_or(|(p, t)| *p != self.params || *t != self.tank) {
            self.substep_cache.clear();
            b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &self.gpu_materials()), self.mats.clone())).unwrap();
            b.copy_buffer(CopyBufferInfo::buffers(staging(&gpu, &optics_table(&self.params)), self.optics_tab.clone())).unwrap();
            self.mats_uploaded = Some((self.params, self.tank.clone()));
            self.tables_uploaded += 1;
        }
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
            self.tank.stride.max(ACC_MIN),
            self.caps(),
        );
        if let Some(cb) = self.substep_cache.get(&key) {
            return cb.clone();
        }
        if self.substep_cache.len() >= 32 {
            self.substep_cache.clear();
        }
        self.ensure_caps_passes();
        let mut r = Recorder::new(&self.gpu);
        self.record_substep(&mut r, gather, particle_groups, key.2, key.3, viscous);
        let cb = r.finish();
        self.substep_cache.insert(key, cb.clone());
        cb
    }

    /// Builds the variants of the capability-specialised passes the tank needs now.
    fn ensure_caps_passes(&mut self) {
        let caps = self.caps();
        self.pass.particles.ensure(&self.gpu, &self.table, caps);
        self.pass.prep.ensure(&self.gpu, &self.table, caps);
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
        r.fill(&self.acc.clone().slice(0..(self.grid[0] * self.grid[1]) as u64 * self.tank.stride.max(ACC_MIN) as u64), 0);
        if self.thermal {
            r.fill(&self.hcell.clone().reinterpret::<[u32]>(), 0);
        }
        let caps = self.caps();
        let particles = ps.particles.at(caps);
        particles.bind_raw(r);
        particles.push_raw(r, gather as u32);
        r.dispatch([particle_groups, 1, 1]);
        ps.prep.at(caps).bind_raw(r);
        r.dispatch(grid_groups);
        if caps & CAP_RHEO != 0 {
            ps.rheo.bind_raw(r);
            r.dispatch(grid_groups);
        }
        if caps & CAP_ELASTIC != 0 {
            ps.elastic.bind_raw(r);
            r.dispatch(grid_groups);
        }
        if caps & CAP_SETTLE != 0 {
            ps.settle.bind_raw(r);
            r.dispatch(grid_groups);
        }
        for pass in [&ps.coef, &ps.surf_smooth, &ps.surf_curv, &ps.surf_apply] {
            pass.bind_raw(r);
            r.dispatch(grid_groups);
        }
        if self.caps() & CAP_MAGNET != 0 {
            // Energy density, then its force on the faces (magnet.comp).
            for mode in 0..2u32 {
                ps.magnet.bind_raw(r);
                ps.magnet.push_raw(r, mode);
                r.dispatch(grid_groups);
            }
        }
        if self.thermal && self.params.heat_rate > 0.0 {
            ps.heat.bind_raw(r);
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
        if self.caps() & CAP_RIGID != 0 {
            for mode in 0..5u32 {
                ps.rigid.bind_raw(r);
                ps.rigid.push_raw(r, mode);
                r.dispatch(grid_groups);
            }
        }
    }

    /// Which optional parts of the solver the tank needs (`CAP_*`): each costs dispatches
    /// only while something in the tank uses it.
    fn caps(&self) -> u32 {
        let p = &self.params;
        let mut caps = 0;
        if self.tank.families > 0 {
            caps |= CAP_FAMILIES;
        }
        if self.thermal {
            caps |= CAP_THERMAL;
        }
        let mut rigid = false;
        for &r in &self.tank.rows {
            let m = &p.mat[r];
            let dil = (m.diluent >= 0).then(|| &p.mat[m.diluent as usize]);
            if m.rheology != RHEO_NEWTONIAN || m.skin > 0.0 {
                caps |= CAP_RHEO;
                rigid |= m.rheology == RHEO_HERSCHEL_BULKLEY || m.rheology == RHEO_THICKENING || m.skin > 0.0;
            }
            // Can it freeze, or get more viscous than the solve goes (cooling, or as it is)?
            if self.thermal {
                rigid |= m.freezes() || dil.is_some_and(|d| d.freezes()) || m.temp_model != TEMP_NONE || dil.is_some_and(|d| d.temp_model != TEMP_NONE);
            }
            rigid |= m.viscosity_at(p.ambient_temp) >= p.rigid_viscosity || (!self.thermal && m.freezes() && p.ambient_temp < m.melt_hi);
        }
        if rigid {
            caps |= CAP_RIGID;
        }
        if self.settle_fams >= self.tank.families && self.tank.families > 0 && self.settles() {
            caps |= CAP_SETTLE;
        }
        if self.tank.chan_e >= 0 && self.pstress.is_some() {
            caps |= CAP_ELASTIC;
        }
        if !self.magnets.is_empty() && self.tank.rows.iter().any(|&r| p.mat[r].susceptibility > 0.0) {
            caps |= CAP_MAGNET;
        }
        caps
    }

    /// How viscosity is integrated this frame, from the fluids present: implicitly when the
    /// viscous diffusion number nu dt / dx^2 of any of them could exceed 0.05, explicitly with
    /// the very same operator otherwise. Forward Euler on this operator is stable up to
    /// about 0.12, so the explicit path keeps a margin of more than two; it exists because
    /// water (nu dt / dx^2 ~ 1e-4 on 1 cm cells) does not need a Krylov solve.
    fn viscosity_mode(&self) -> Viscosity {
        let dt = self.dt_limit();
        let p = &self.params;
        // The most viscous any liquid in the tank can get: coldest (the temperature law is
        // capped at e^30) or frozen while heat is simulated, as it is otherwise.
        let nu = |i: usize| {
            let m = &p.mat[i];
            let cells = 1.0 / (p.cell_size * p.cell_size);
            if !self.thermal {
                m.viscosity_at(p.ambient_temp) * cells
            } else if m.freezes() || m.temp_model != TEMP_NONE {
                f32::INFINITY
            } else {
                m.viscosity * cells
            }
        };
        let d = self.tank.rows.iter().map(|&i| nu(i) * dt).fold(0.0, f32::max);
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
        self.ensure_caps_passes();
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
        b.fill_buffer(self.acc.clone().slice(0..(self.grid[0] * self.grid[1]) as u64 * self.tank.stride.max(ACC_MIN) as u64), 0).unwrap();
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
        if let Some((a, s)) = &self.pstress {
            b.copy_buffer(CopyBufferInfo::buffers(s.clone().slice(0..n), a.clone().slice(0..n))).unwrap();
        }
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
        let mut st = Stats { live: data[4], ..Default::default() };
        for r in 0..MAX_MATERIALS {
            st.count[r] = data[8 + 2 * r];
            st.conc[r] = data[9 + 2 * r] as f32 / 256.0;
        }
        self.stats = st;
        self.stats_in_flight = false;
        true
    }

    // ---- blocking helpers for the headless tools ------------------------------------

    /// Shortest remainder, as a share of a substep, that the blocking helpers still record.
    fn sliver(&self) -> f64 {
        (self.params.min_substep_share as f64).max(1e-6)
    }

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
            // to within `min_substep_share` of a substep (switched off, a
            // millionth of a substep still ends the loop, so float rounding cannot keep it going).
            if left <= self.sliver() * dt {
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
            if left <= self.sliver() * dt {
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

    /// One interactive frame, blocking: `substeps` substeps meant to cover `target` seconds,
    /// as the app records them. Returns the simulated time they covered.
    pub fn frame_blocking(&mut self, substeps: u32, target: f32, tool: ToolState) -> f64 {
        let start = self.sim_time;
        let mut b = self.gpu.builder();
        self.record(&mut b, substeps, target, tool);
        submit_and_wait(&self.gpu, b);
        self.poll_readbacks();
        self.sim_time - start
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

    /// Reads back what crossed the boundary of the simulation since the last reset (heat to
    /// and from air, solids and the brush; boiled-off liquid). Blocking; for the tools.
    pub fn download_ledger(&self) -> Ledger {
        let host = readback::<u32>(&self.gpu, LEDGER_WORDS);
        let mut b = self.gpu.builder();
        b.copy_buffer(CopyBufferInfo::buffers(self.ledger.clone(), host.clone())).unwrap();
        submit_and_wait(&self.gpu, b);
        let w = host.read().unwrap();
        let unit = self.params.cell_heat();
        let e = |k: usize| ((w[2 * k] as u64 | (w[2 * k + 1] as u64) << 32) as f64 / 65536.0) * unit;
        let mut l = Ledger { from_air: e(0), to_air: e(1), from_solids: e(2), to_solids: e(3), heated: e(4), cooled: e(5), boiled: e(6), ..Default::default() };
        for r in 0..MAX_MATERIALS {
            l.boiled_count[r] = w[16 + r];
            l.residue_count[r] = w[80 + r];
        }
        l
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

    /// Debug aid (`sprout lab`): face velocities (u, v, state u, state v), rigid body labels
    /// (root cell, body slot) and cell fields (occupancy, density, viscosity, level).
    pub fn download_grid(&self) -> GridDump {
        let (vel, labels, cell) = (&self.dbg_vel, &self.dbg_labels, &self.dbg_cell);
        let a = readback::<[f32; 4]>(&self.gpu, vel.len());
        let r = readback::<[i32; 2]>(&self.gpu, labels.len());
        let c = readback::<[f32; 4]>(&self.gpu, cell.len());
        let mut b = self.gpu.builder();
        b.copy_buffer(CopyBufferInfo::buffers(vel.clone(), a.clone())).unwrap();
        b.copy_buffer(CopyBufferInfo::buffers(labels.clone(), r.clone())).unwrap();
        b.copy_buffer(CopyBufferInfo::buffers(cell.clone(), c.clone())).unwrap();
        submit_and_wait(&self.gpu, b);
        GridDump { vel: a.read().unwrap().to_vec(), labels: r.read().unwrap().to_vec(), cell: c.read().unwrap().to_vec() }
    }

    /// Debug aid: the rigid body table of the latest substep, per body mass, mass-weighted
    /// offset sums, momentum, angular momentum about the root and inertia sum (grid units).
    pub fn download_bodies(&self) -> Vec<[f64; 7]> {
        let host = readback::<u32>(&self.gpu, 16 + BODY_WORDS * RIGID_BODIES);
        let mut b = self.gpu.builder();
        b.copy_buffer(CopyBufferInfo::buffers(self.rbody.clone(), host.clone())).unwrap();
        submit_and_wait(&self.gpu, b);
        let w = host.read().unwrap();
        let n = (w[0] as usize).min(RIGID_BODIES as usize);
        let g = |k: usize| ((w[k] as u64) | ((w[k + 1] as u64) << 32)) as i64 as f64 / 65536.0;
        (0..n).map(|b| { let o = 16 + BODY_WORDS as usize * b; [g(o), g(o + 2), g(o + 4), g(o + 6), g(o + 8), g(o + 10), g(o + 12)] }).collect()
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

/// Grid fields read back by `FluidSim::download_grid`.
pub struct GridDump {
    pub vel: Vec<[f32; 4]>,
    pub labels: Vec<[i32; 2]>,
    pub cell: Vec<[f32; 4]>,
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
