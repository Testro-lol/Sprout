//! Rendering: particles are splatted additively into an off-screen accumulation target,
//! then one full-screen pass thresholds that into a liquid surface and draws the solids
//! (from the same signed-distance field the simulation collides against) and the
//! backdrop. Nothing is read back from the GPU.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    command_buffer::{RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::AllocationCreateInfo,
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};

use crate::{
    gpu::{Builder, Gpu},
    sim::{FluidSim, Material},
    spirv,
};

/// Splat radius in cells. Particles sit 0.5 cells apart at rest.
const SPLAT_RADIUS: f32 = 1.1;
/// Accumulated splat weight that counts as liquid; a lone particle peaks at 1.0.
const SURFACE_THRESHOLD: f32 = 0.8;

/// Camera: `center` is the world position (cells) at the middle of the view, `zoom` is
/// pixels per cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub center: [f32; 2],
    pub zoom: f32,
}

impl View {
    /// The view that shows the whole grid in a target of the given pixel size.
    pub fn fit(grid: [u32; 2], size: [u32; 2]) -> View {
        let zoom = (size[0] as f32 / grid[0] as f32).min(size[1] as f32 / grid[1] as f32);
        View { center: [grid[0] as f32 * 0.5, grid[1] as f32 * 0.5], zoom }
    }

    pub fn world_to_screen(&self, cells: [f32; 2], size: [u32; 2]) -> [f32; 2] {
        [
            (cells[0] - self.center[0]) * self.zoom + size[0] as f32 * 0.5,
            (self.center[1] - cells[1]) * self.zoom + size[1] as f32 * 0.5,
        ]
    }

    pub fn screen_to_world(&self, px: [f32; 2], size: [u32; 2]) -> [f32; 2] {
        [
            self.center[0] + (px[0] - size[0] as f32 * 0.5) / self.zoom,
            self.center[1] - (px[1] - size[1] as f32 * 0.5) / self.zoom,
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SplatPc {
    scale: [f32; 2],
    offset: [f32; 2],
    water: [f32; 4],
    syrup: [f32; 4],
    oil: [f32; 4],
    radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositePc {
    cam: [f32; 2],
    half_res: [f32; 2],
    grid: [i32; 2],
    inv_zoom: f32,
    threshold: f32,
}

struct Accum {
    extent: [u32; 2],
    framebuffer: Arc<Framebuffer>,
    composite_set: Arc<DescriptorSet>,
}

pub struct Renderer {
    gpu: Arc<Gpu>,
    accum_pass: Arc<RenderPass>,
    final_pass: Arc<RenderPass>,
    splat: Arc<GraphicsPipeline>,
    composite: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    splat_set: Option<Arc<DescriptorSet>>,
    accum: Option<Accum>,
    targets: Vec<(Arc<ImageView>, Arc<Framebuffer>)>,
}

const ACCUM_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

fn pipeline(
    gpu: &Gpu,
    vs: &[u8],
    fs: &[u8],
    subpass: Subpass,
    blend: Option<AttachmentBlend>,
) -> Arc<GraphicsPipeline> {
    let device = gpu.device.clone();
    let stages = [PipelineShaderStageCreateInfo::new(gpu.shader(vs)), PipelineShaderStageCreateInfo::new(gpu.shader(fs))];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::new()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState { blend, ..Default::default() },
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

impl Renderer {
    pub fn new(gpu: Arc<Gpu>, target_format: Format) -> Renderer {
        let device = gpu.device.clone();
        let accum_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: { color: { format: ACCUM_FORMAT, samples: 1, load_op: Clear, store_op: Store } },
            pass: { color: [color], depth_stencil: {} }
        )
        .unwrap();
        let final_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: { color: { format: target_format, samples: 1, load_op: Clear, store_op: Store } },
            pass: { color: [color], depth_stencil: {} }
        )
        .unwrap();

        let additive = AttachmentBlend {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::One,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::One,
            alpha_blend_op: BlendOp::Add,
        };
        let splat = pipeline(
            &gpu,
            spirv!("splat.vert"),
            spirv!("splat.frag"),
            Subpass::from(accum_pass.clone(), 0).unwrap(),
            Some(additive),
        );
        let composite = pipeline(
            &gpu,
            spirv!("composite.vert"),
            spirv!("composite.frag"),
            Subpass::from(final_pass.clone(), 0).unwrap(),
            None,
        );
        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        Renderer { gpu, accum_pass, final_pass, splat, composite, sampler, splat_set: None, accum: None, targets: Vec::new() }
    }

    /// The subpass of the final (on-screen) pass, for overlays that draw into it.
    pub fn final_subpass(&self) -> Subpass {
        Subpass::from(self.final_pass.clone(), 0).unwrap()
    }

    /// Forget framebuffers of old target images; call after the swapchain is recreated.
    pub fn invalidate_targets(&mut self) {
        self.targets.clear();
    }

    /// Forget everything tied to a simulation's buffers; call when it is replaced.
    pub fn invalidate(&mut self) {
        self.targets.clear();
        self.accum = None;
        self.splat_set = None;
    }

    fn target_framebuffer(&mut self, target: &Arc<ImageView>) -> Arc<Framebuffer> {
        if let Some((_, fb)) = self.targets.iter().find(|(view, _)| Arc::ptr_eq(view, target)) {
            return fb.clone();
        }
        let fb = Framebuffer::new(
            self.final_pass.clone(),
            FramebufferCreateInfo { attachments: vec![target.clone()], ..Default::default() },
        )
        .unwrap();
        self.targets.push((target.clone(), fb.clone()));
        fb
    }

    fn ensure_accum(&mut self, extent: [u32; 2], sim: &FluidSim) {
        if self.accum.as_ref().is_some_and(|a| a.extent == extent) {
            return;
        }
        let image = Image::new(
            self.gpu.memory.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: ACCUM_FORMAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let view = ImageView::new_default(image).unwrap();
        let framebuffer = Framebuffer::new(
            self.accum_pass.clone(),
            FramebufferCreateInfo { attachments: vec![view.clone()], ..Default::default() },
        )
        .unwrap();
        let composite_set = DescriptorSet::new(
            self.gpu.descriptors.clone(),
            self.composite.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, view, self.sampler.clone()),
                WriteDescriptorSet::buffer(1, sim.sdf_buffer().clone()),
            ],
            [],
        )
        .unwrap();
        self.accum = Some(Accum { extent, framebuffer, composite_set });
    }

    /// Records the frame into `b`. `overlay` runs inside the final render pass, after the
    /// scene has been drawn, with the viewport covering the whole target.
    pub fn render(
        &mut self,
        b: &mut Builder,
        target: Arc<ImageView>,
        sim: &FluidSim,
        view: View,
        overlay: impl FnOnce(&mut Builder),
    ) {
        let e = target.image().extent();
        let extent = [e[0], e[1]];
        self.ensure_accum(extent, sim);
        if self.splat_set.is_none() {
            self.splat_set = Some(
                DescriptorSet::new(
                    self.gpu.descriptors.clone(),
                    self.splat.layout().set_layouts()[0].clone(),
                    [WriteDescriptorSet::buffer(0, sim.particle_buffer().clone())],
                    [],
                )
                .unwrap(),
            );
        }
        let target_fb = self.target_framebuffer(&target);
        let accum = self.accum.as_ref().unwrap();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };
        let half = [extent[0] as f32 * 0.5, extent[1] as f32 * 0.5];
        let rgba = |m: Material| {
            let c = m.color();
            [c[0], c[1], c[2], 1.0]
        };

        // Pass 1: splat particles.
        let scale = [view.zoom / half[0], -view.zoom / half[1]];
        b.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0; 4].into())],
                ..RenderPassBeginInfo::framebuffer(accum.framebuffer.clone())
            },
            SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
        )
        .unwrap();
        if sim.slots() > 0 {
            b.bind_pipeline_graphics(self.splat.clone())
                .unwrap()
                .set_viewport(0, [viewport.clone()].into_iter().collect())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.splat.layout().clone(),
                    0,
                    self.splat_set.clone().unwrap(),
                )
                .unwrap()
                .push_constants(
                    self.splat.layout().clone(),
                    0,
                    SplatPc {
                        scale,
                        offset: [-view.center[0] * scale[0], -view.center[1] * scale[1]],
                        water: rgba(Material::Water),
                        syrup: rgba(Material::Syrup),
                        oil: rgba(Material::Oil),
                        radius: SPLAT_RADIUS,
                    },
                )
                .unwrap();
            // SAFETY: the vertex shader indexes the particle buffer with gl_VertexIndex / 6,
            // which stays below the slot count.
            unsafe { b.draw(sim.slots() * 6, 1, 0, 0) }.unwrap();
        }
        b.end_render_pass(SubpassEndInfo::default()).unwrap();

        // Pass 2: resolve to the target.
        b.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(target_fb)
            },
            SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
        )
        .unwrap();
        b.bind_pipeline_graphics(self.composite.clone())
            .unwrap()
            .set_viewport(0, [viewport].into_iter().collect())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.composite.layout().clone(),
                0,
                accum.composite_set.clone(),
            )
            .unwrap()
            .push_constants(
                self.composite.layout().clone(),
                0,
                CompositePc {
                    cam: view.center,
                    half_res: half,
                    grid: [sim.grid[0] as i32, sim.grid[1] as i32],
                    inv_zoom: 1.0 / view.zoom,
                    threshold: SURFACE_THRESHOLD,
                },
            )
            .unwrap();
        // SAFETY: a full-screen triangle with no vertex inputs.
        unsafe { b.draw(3, 1, 0, 0) }.unwrap();
        overlay(b);
        b.end_render_pass(SubpassEndInfo::default()).unwrap();
    }
}
