//! Hand-synchronised secondary command buffers for the solver's inner loop.
//!
//! A substep is hundreds of small dispatches (every Krylov iteration is ~30). Vulkano's
//! automatic synchronisation tracks every buffer each of them touches and costs several
//! milliseconds of CPU per substep to build, far more than the GPU needs to run it. The
//! substep's commands never change between substeps, so they are recorded once, raw, with
//! a full memory barrier after every command, and replayed with `execute_commands`.
//!
//! Vulkano is told that such a buffer uses no resources, so it inserts no barriers around
//! it; the buffer therefore starts and ends with a full barrier of its own, which orders it
//! against everything recorded before and after it in the primary command buffer.

use std::sync::Arc;

use vulkano::{
    VulkanObject,
    buffer::Subbuffer,
    command_buffer::{
        CommandBuffer, CommandBufferBeginInfo, CommandBufferInheritanceInfo, CommandBufferLevel, CommandBufferUsage,
        DispatchIndirectCommand, RecordingCommandBuffer, SecondaryCommandBufferAbstract,
        SecondaryCommandBufferResourcesUsage,
    },
    descriptor_set::DescriptorSet,
    device::{Device, DeviceOwned},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    query::{QueryPool, QueryPoolCreateInfo, QueryType},
    sync::{AccessFlags, DependencyInfo, MemoryBarrier, PipelineStage, PipelineStages},
};

use crate::gpu::Gpu;

/// A recorded raw secondary command buffer plus everything it references.
pub struct RawSecondary {
    inner: CommandBuffer,
    inheritance: CommandBufferInheritanceInfo,
    usage: SecondaryCommandBufferResourcesUsage,
    _pipelines: Vec<Arc<ComputePipeline>>,
    _sets: Vec<Arc<DescriptorSet>>,
    _buffers: Vec<Subbuffer<[u8]>>,
}

unsafe impl VulkanObject for RawSecondary {
    type Handle = ash::vk::CommandBuffer;

    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for RawSecondary {
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

// SAFETY: the buffer was recorded with `SimultaneousUse`, so it may be executed any number of
// times, concurrently; it synchronises itself (see the module comment), which is why it can
// report no resource usage.
unsafe impl SecondaryCommandBufferAbstract for RawSecondary {
    fn as_raw(&self) -> &CommandBuffer {
        &self.inner
    }

    fn usage(&self) -> CommandBufferUsage {
        CommandBufferUsage::SimultaneousUse
    }

    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        &self.inheritance
    }

    fn lock_record(&self) -> Result<(), Box<vulkano::ValidationError>> {
        Ok(())
    }

    unsafe fn unlock(&self) {}

    fn resources_usage(&self) -> &SecondaryCommandBufferResourcesUsage {
        &self.usage
    }
}

/// Records a `RawSecondary`. Every command is followed by a full memory barrier.
pub struct Recorder {
    rec: RecordingCommandBuffer,
    pipelines: Vec<Arc<ComputePipeline>>,
    sets: Vec<Arc<DescriptorSet>>,
    buffers: Vec<Subbuffer<[u8]>>,
    /// The layout of the bound pipeline, for push constants.
    bound: Option<Arc<ComputePipeline>>,
    bound_name: &'static str,
    /// Profiling: a timestamp after every command, labelled with the bound pass.
    timing: Option<(Arc<QueryPool>, Vec<&'static str>)>,
}

/// Between two commands of the buffer: compute and transfer writes before, compute,
/// transfer and indirect-argument reads after.
fn inner_barrier() -> DependencyInfo {
    DependencyInfo {
        memory_barriers: [MemoryBarrier {
            src_stages: PipelineStages::COMPUTE_SHADER | PipelineStages::ALL_TRANSFER,
            src_access: AccessFlags::SHADER_WRITE | AccessFlags::TRANSFER_WRITE,
            dst_stages: PipelineStages::COMPUTE_SHADER | PipelineStages::ALL_TRANSFER | PipelineStages::DRAW_INDIRECT,
            dst_access: AccessFlags::SHADER_READ
                | AccessFlags::SHADER_WRITE
                | AccessFlags::TRANSFER_READ
                | AccessFlags::TRANSFER_WRITE
                | AccessFlags::INDIRECT_COMMAND_READ,
            ..Default::default()
        }]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

/// At either end of the buffer: everything before against everything after.
fn full_barrier() -> DependencyInfo {
    DependencyInfo {
        memory_barriers: [MemoryBarrier {
            src_stages: PipelineStages::ALL_COMMANDS,
            src_access: AccessFlags::MEMORY_WRITE,
            dst_stages: PipelineStages::ALL_COMMANDS,
            dst_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
            ..Default::default()
        }]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

impl Recorder {
    pub fn new(gpu: &Gpu) -> Recorder {
        let rec = {
            RecordingCommandBuffer::new(
                gpu.commands.clone(),
                gpu.queue.queue_family_index(),
                CommandBufferLevel::Secondary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::SimultaneousUse,
                    inheritance_info: Some(CommandBufferInheritanceInfo::default()),
                    ..Default::default()
                },
            )
        }
        .expect("cannot allocate a secondary command buffer");
        let mut r = Recorder { rec, pipelines: Vec::new(), sets: Vec::new(), buffers: Vec::new(), bound: None, bound_name: "", timing: None };
        unsafe { r.rec.pipeline_barrier(&full_barrier()) }.unwrap();
        r
    }

    fn barrier(&mut self) {
        unsafe { self.rec.pipeline_barrier(&inner_barrier()) }.unwrap();
        if let Some((pool, labels)) = &mut self.timing {
            let q = labels.len() as u32;
            if q < pool.query_count() {
                unsafe { self.rec.write_timestamp(pool, q, PipelineStage::AllCommands) }.unwrap();
                labels.push(self.bound_name);
            }
        }
    }

    /// Turns on per-command GPU timestamps (for `FluidSim::profile_passes`).
    pub fn with_timing(mut self, gpu: &Gpu, capacity: u32) -> Recorder {
        let pool = QueryPool::new(
            gpu.device.clone(),
            QueryPoolCreateInfo { query_count: capacity, ..QueryPoolCreateInfo::query_type(QueryType::Timestamp) },
        )
        .unwrap();
        unsafe {
            self.rec.reset_query_pool(&pool, 0..capacity).unwrap();
            self.rec.write_timestamp(&pool, 0, PipelineStage::AllCommands).unwrap();
        }
        self.timing = Some((pool, vec!["start"]));
        self
    }

    pub fn timing(&self) -> Option<(Arc<QueryPool>, Vec<&'static str>)> {
        self.timing.clone()
    }

    pub fn bind(&mut self, name: &'static str, pipeline: &Arc<ComputePipeline>, set: &Arc<DescriptorSet>) {
        self.bound_name = name;
        unsafe {
            self.rec.bind_pipeline_compute(pipeline).unwrap();
            self.rec
                .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout(), 0, &[set.as_raw()], &[])
                .unwrap();
        }
        self.pipelines.push(pipeline.clone());
        self.sets.push(set.clone());
        self.bound = Some(pipeline.clone());
    }

    pub fn push(&mut self, words: &[u32]) {
        let layout = self.bound.as_ref().expect("push before bind").layout().clone();
        macro_rules! go {
            ($k:literal) => {{
                let w: [u32; $k] = words.try_into().unwrap();
                unsafe { self.rec.push_constants(&layout, 0, &w) }.unwrap();
            }};
        }
        match words.len() {
            1 => go!(1),
            2 => go!(2),
            3 => go!(3),
            4 => go!(4),
            5 => go!(5),
            6 => go!(6),
            7 => go!(7),
            8 => go!(8),
            n => panic!("{n} words of push constants"),
        }
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        // SAFETY: the shaders bounds-check their invocations (see `dispatch` in mod.rs).
        unsafe { self.rec.dispatch(groups) }.unwrap();
        self.barrier();
    }

    pub fn dispatch_indirect(&mut self, args: &Subbuffer<[DispatchIndirectCommand]>) {
        // SAFETY: the argument buffer only ever holds the nominal group counts or zero.
        unsafe { self.rec.dispatch_indirect(args) }.unwrap();
        self.buffers.push(args.as_bytes().clone());
        self.barrier();
    }

    pub fn fill(&mut self, buffer: &Subbuffer<[u32]>, value: u32) {
        unsafe { self.rec.fill_buffer(buffer, value) }.unwrap();
        self.bound_name = "fill";
        self.buffers.push(buffer.as_bytes().clone());
        self.barrier();
    }

    pub fn finish(mut self) -> Arc<RawSecondary> {
        unsafe { self.rec.pipeline_barrier(&full_barrier()) }.unwrap();
        let inner = unsafe { self.rec.end() }.expect("cannot finish a secondary command buffer");
        Arc::new(RawSecondary {
            inner,
            inheritance: CommandBufferInheritanceInfo::default(),
            usage: SecondaryCommandBufferResourcesUsage::default(),
            _pipelines: self.pipelines,
            _sets: self.sets,
            _buffers: self.buffers,
        })
    }
}
