//! Rendering. Particles are splatted additively into off-screen accumulation targets that
//! hold, per pixel, accumulated optical properties rather than per-liquid channels
//! (README, "Look"): the weight of each phase group (whose level sets are the surfaces and
//! the interfaces), the body colour in OKLab with its chroma, opacity, scattering, metal,
//! gloss, emission from the particles' temperature, speed, thickness, foam and drying
//! time. Every property comes from the material table (`Optics`, built from the live
//! `SimParams` each frame) and the particles' state, so any material, built-in or custom,
//! draws with no code of its own. A blurred, low-resolution copy of the group weights says
//! how much liquid is *around* each pixel. One full-screen pass shades the liquid from
//! these and draws the solids, from the same signed-distance field the simulation collides
//! against, and the backdrop. Nothing is read back from the GPU.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    command_buffer::{ClearColorImageInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo},
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
    sim::{FluidSim, PPC},
    spirv,
};

/// Splat kernel radius in cells. Particles sit 0.5 cells apart at rest.
const SPLAT_RADIUS: f32 = 1.5;
/// The accumulation target is rendered just fine enough for the kernel to span this many
/// texels, and never coarser than one texel per `ACCUM_MAX_DIV` pixels: a kernel that covers
/// many pixels is drawn at up to half resolution (a quarter of the fill, which is most of the
/// cost of a frame), a small one at full resolution, where it would otherwise blur.
const ACCUM_KERNEL_TEXELS: f32 = 2.8;
const ACCUM_MAX_DIV: f32 = 2.0;
/// Smallest kernel radius in texels of the accumulation target. Far out, where a cell is
/// smaller than a texel, the kernel is widened to this (and its weight lowered to keep the
/// same total), so that the accumulation is an area average rather than a sparse sampling
/// that would shimmer.
const SPLAT_MIN_TEXELS: f32 = 1.5;
/// Accumulated weight that counts as liquid; a lone particle peaks at 1.0, so it draws as a
/// droplet of radius `SPLAT_RADIUS * sqrt(1 - sqrt(SURFACE_THRESHOLD))`, 0.57 cells.
const SURFACE_THRESHOLD: f32 = 0.65;
/// A particle is drawn over the path it covers in this time, s (spray becomes streaks).
const EXPOSURE: f32 = 1.0 / 160.0;
/// Standard deviation of the wide field, cells.
const WIDE_SIGMA: f32 = 2.0;
/// Standard deviation of the wide field in its own texels: it is rendered just fine enough
/// for that, so the blur costs the same at every zoom.
const WIDE_SIGMA_TEXELS: f32 = 3.0;
/// The wide field is never finer than this share of the target, nor coarser than 1 / 64.
const WIDE_MIN_FACTOR: f32 = 2.0;
const WIDE_MAX_FACTOR: f32 = 64.0;

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
    radius: f32,
    weight: f32,
    exposure: f32,
    inv_speed: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SteamPc {
    scale: [f32; 2],
    offset: [f32; 2],
    bubble: f32,
    bubble_w: f32,
}

/// The vapour field (steam.vert): kernel radius of a bubble of steam in the liquid, cells.
/// A lone one (weight 1) is drawn out to where the field crosses composite.frag's
/// `BUBBLE_CUT`: `BUBBLE_RADIUS * sqrt(1 - sqrt(0.45))`, 0.5 cells; crowded ones merge.
const BUBBLE_RADIUS: f32 = 0.85;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownPc {
    src_texel: [f32; 2],
    src_max: [f32; 2],
    factor: f32,
    taps: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurPc {
    texel: [f32; 2],
    step: [f32; 2],
    uv_max: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositePc {
    cam: [f32; 2],
    half_res: [f32; 2],
    grid: [i32; 2],
    inv_zoom: f32,
    threshold: f32,
    wide_scale: [f32; 2],
    wide_max: [f32; 2],
    rest: f32,
    mist: f32,
    accum_scale: [f32; 2],
    accum_max: [f32; 2],
    accum_div: f32,
    now: f32,
    wet_width: f32,
    wet_tint: f32,
    glow: f32,
    stains: f32,
    vapour: f32,
    film_layer: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WetPc {
    grid: [i32; 2],
    now: f32,
}

/// The wet map (`shaders/wet.frag`): one texel per cell, kept from frame to frame.
struct Wet {
    view: Arc<ImageView>,
    framebuffer: Arc<Framebuffer>,
    set: Arc<DescriptorSet>,
    grid: [u32; 2],
    /// The tank (`FluidSim::resets`) and simulated time of the last update; `None` until the
    /// map has been cleared once.
    updated: Option<(u64, f64)>,
}

/// How the wide field is laid out for one frame: it uses the top-left `size` texels of its
/// image, each covering `factor` pixels of the target.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WideLayout {
    size: [u32; 2],
    factor: f32,
    /// Bilinear taps per axis of the downsampling pass (each covers two accumulation texels).
    taps: i32,
    /// Blur step (half a standard deviation) in texels.
    step: f32,
}

impl WideLayout {
    /// `zoom` in pixels per cell, `div` pixels per accumulation texel, `extent` of the
    /// target, `image` of the wide images.
    fn new(zoom: f32, div: f32, extent: [u32; 2], image: [u32; 2]) -> WideLayout {
        let factor = (WIDE_SIGMA * zoom / WIDE_SIGMA_TEXELS).clamp(WIDE_MIN_FACTOR, WIDE_MAX_FACTOR);
        let size = [
            ((extent[0] as f32 / factor).ceil() as u32).clamp(1, image[0]),
            ((extent[1] as f32 / factor).ceil() as u32).clamp(1, image[1]),
        ];
        WideLayout { size, factor, taps: (factor / div * 0.5).ceil() as i32, step: 0.5 * WIDE_SIGMA * zoom / factor }
    }
}

struct Targets {
    extent: [u32; 2],
    accum: Arc<Framebuffer>,
    /// The two wide images: the downsampled copy and the first blur direction land in
    /// `wide[0]` and `wide[1]`, the finished field back in `wide[0]`.
    wide: [Arc<Framebuffer>; 2],
    wide_extent: [u32; 2],
    down_set: Arc<DescriptorSet>,
    blur_sets: [Arc<DescriptorSet>; 2],
    composite_set: Arc<DescriptorSet>,
    /// The vapour field (steam.vert), laid out as `accum`: bubbles, puffs.
    vapour: Arc<Framebuffer>,
    /// Whether the vapour field may hold anything (it is drawn into, or new): without steam
    /// it is cleared once, not every frame (a 1080p clear costs ~0.1 ms).
    vapour_dirty: std::cell::Cell<bool>,
}

/// Display settings for how the liquid and solids are drawn (Display panel).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// How far the wet film of a liquid reaches into a solid it touches, cells.
    pub wet_width: f32,
    /// 0..1: how strongly a wet solid takes the colour of the liquid (0: never drawn wet).
    pub wet_tint: f32,
    /// 0..2: the glow of hot liquid (bloom, and the light it throws on its surroundings).
    pub glow: f32,
    /// 0..1: how strongly stains show (0: never drawn; they are still laid down).
    pub stains: f32,
}

impl Look {
    pub const WET_WIDTH: std::ops::RangeInclusive<f32> = 0.5..=5.0;
}

impl Default for Look {
    fn default() -> Self {
        Look { wet_width: 2.2, wet_tint: 1.0, glow: 1.0, stains: 1.0 }
    }
}

pub struct Renderer {
    /// Display settings, read each frame.
    pub look: Look,
    gpu: Arc<Gpu>,
    /// A single-target pass: the splat and the wide field.
    field_pass: Arc<RenderPass>,
    wet_pass: Arc<RenderPass>,
    final_pass: Arc<RenderPass>,
    wet_pipeline: Arc<GraphicsPipeline>,
    wet: Option<Wet>,
    splat: Arc<GraphicsPipeline>,
    down: Arc<GraphicsPipeline>,
    blur: Arc<GraphicsPipeline>,
    composite: Arc<GraphicsPipeline>,
    composite_dry: Arc<GraphicsPipeline>,
    steam: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    splat_set: Option<Arc<DescriptorSet>>,
    steam_set: Option<Arc<DescriptorSet>>,
    accum: Option<Targets>,
    targets: Vec<(Arc<ImageView>, Arc<Framebuffer>)>,
}

const ACCUM_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
/// The splat target (RGBA16F), the sharp field all surfaces and interfaces are drawn from:
/// the weight of phase groups 0, 1, 2, 3 (aqueous, oil-like, liquid metal, silicate melt),
/// times the kernel. What the liquid looks like comes from the grid, per cell
/// (`FluidSim::record_cell_optics`): colour, opacity, scattering, metal, gloss, glow, speed.
/// The wet map stores simulated time, which needs more than half precision.
const WET_FORMAT: Format = Format::R32G32B32A32_SFLOAT;

fn pipeline(
    gpu: &Gpu,
    vs: &[u8],
    fs: &[u8],
    subpass: Subpass,
    blend: Option<AttachmentBlend>,
) -> Arc<GraphicsPipeline> {
    pipeline_with(gpu, vs, gpu.shader(fs), subpass, blend, None)
}

/// `pipeline` with a fragment entry point of its own (specialised), and optionally the
/// layout of another pipeline, so that the two can share descriptor sets.
fn pipeline_with(
    gpu: &Gpu,
    vs: &[u8],
    fs: vulkano::shader::EntryPoint,
    subpass: Subpass,
    blend: Option<AttachmentBlend>,
    layout: Option<Arc<PipelineLayout>>,
) -> Arc<GraphicsPipeline> {
    let device = gpu.device.clone();
    let stages = [PipelineShaderStageCreateInfo::new(gpu.shader(vs)), PipelineShaderStageCreateInfo::new(fs)];
    let layout = layout.unwrap_or_else(|| {
        PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap()
    });
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

fn viewport(size: [u32; 2]) -> Viewport {
    Viewport { offset: [0.0, 0.0], extent: [size[0] as f32, size[1] as f32], depth_range: 0.0..=1.0 }
}

impl Renderer {
    pub fn new(gpu: Arc<Gpu>, target_format: Format) -> Renderer {
        let device = gpu.device.clone();
        let field_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: { color: { format: ACCUM_FORMAT, samples: 1, load_op: Clear, store_op: Store } },
            pass: { color: [color], depth_stencil: {} }
        )
        .unwrap();
        // The wet map keeps its contents between frames.
        let wet_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: { color: { format: WET_FORMAT, samples: 1, load_op: Load, store_op: Store } },
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
        let accum_subpass = || Subpass::from(field_pass.clone(), 0).unwrap();
        let splat = pipeline(&gpu, spirv!("splat.vert"), spirv!("splat.frag"), Subpass::from(field_pass.clone(), 0).unwrap(), Some(additive.clone()));
        let wet_pipeline =
            pipeline(&gpu, spirv!("composite.vert"), spirv!("wet.frag"), Subpass::from(wet_pass.clone(), 0).unwrap(), None);
        let down = pipeline(&gpu, spirv!("composite.vert"), spirv!("down.frag"), accum_subpass(), None);
        let blur = pipeline(&gpu, spirv!("composite.vert"), spirv!("blur.frag"), accum_subpass(), None);
        let composite = pipeline(
            &gpu,
            spirv!("composite.vert"),
            spirv!("composite.frag"),
            Subpass::from(final_pass.clone(), 0).unwrap(),
            None,
        );
        // Without steam, the same shader with the vapour field compiled out (HAS_VAPOUR):
        // its code cost ~0.09 ms a 1080p frame even where it never ran.
        let composite_dry = pipeline_with(
            &gpu,
            spirv!("composite.vert"),
            gpu.shader_specialized(spirv!("composite.frag"), &[(0, 0)]),
            Subpass::from(final_pass.clone(), 0).unwrap(),
            None,
            Some(composite.layout().clone()),
        );
        let steam = pipeline(&gpu, spirv!("steam.vert"), spirv!("steam.frag"), accum_subpass(), Some(additive));
        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        Renderer {
            look: Look::default(),
            gpu,
            field_pass,
            wet_pass,
            final_pass,
            wet_pipeline,
            wet: None,
            splat,
            down,
            blur,
            composite,
            composite_dry,
            steam,
            sampler,
            splat_set: None,
            steam_set: None,
            accum: None,
            targets: Vec::new(),
        }
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
        self.steam_set = None;
        self.wet = None;
    }

    fn ensure_wet(&mut self, sim: &FluidSim) {
        if self.wet.as_ref().is_some_and(|w| w.grid == sim.grid) {
            return;
        }
        let image = Image::new(
            self.gpu.memory.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: WET_FORMAT,
                extent: [sim.grid[0], sim.grid[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let view = ImageView::new_default(image).unwrap();
        let framebuffer = Framebuffer::new(
            self.wet_pass.clone(),
            FramebufferCreateInfo { attachments: vec![view.clone()], ..Default::default() },
        )
        .unwrap();
        let set = DescriptorSet::new(
            self.gpu.descriptors.clone(),
            self.wet_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, sim.cell_buffer().clone()),
                WriteDescriptorSet::buffer(1, sim.composition_buffer().clone()),
                WriteDescriptorSet::buffer(2, sim.sdf_buffer().clone()),
                WriteDescriptorSet::buffer(3, sim.surface_buffer()),
            ],
            [],
        )
        .unwrap();
        self.wet = Some(Wet { view, framebuffer, set, grid: sim.grid, updated: None });
        // The composite pass samples the map: its descriptor set has to be rebuilt.
        self.accum = None;
    }

    /// Notes which solids the liquid is touching at this moment, so that they stay wet for a
    /// while after it has gone. `render` does this itself; call it between renders (it is
    /// one small pass) when the simulation advances a long way without one, as the
    /// off-screen tools do.
    pub fn track(&mut self, b: &mut Builder, sim: &FluidSim) {
        // The stain map is laid down with the cell optics.
        sim.record_cell_optics(b, 1.0 / fast_speed(sim));
        self.ensure_wet(sim);
        let wet = self.wet.as_mut().unwrap();
        let now = (sim.resets(), sim.sim_time());
        // A new map, or a tank that started over: nothing is wet.
        if wet.updated.is_none_or(|(tank, _)| tank != now.0) {
            b.clear_color_image(ClearColorImageInfo::image(wet.view.image().clone())).unwrap();
        } else if wet.updated == Some(now) {
            return;
        }
        wet.updated = Some(now);
        b.begin_render_pass(
            RenderPassBeginInfo { clear_values: vec![None], ..RenderPassBeginInfo::framebuffer(wet.framebuffer.clone()) },
            SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
        )
        .unwrap();
        b.bind_pipeline_graphics(self.wet_pipeline.clone())
            .unwrap()
            .set_viewport(0, [viewport(wet.grid)].into_iter().collect())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Graphics, self.wet_pipeline.layout().clone(), 0, wet.set.clone())
            .unwrap()
            .push_constants(
                self.wet_pipeline.layout().clone(),
                0,
                WetPc { grid: [wet.grid[0] as i32, wet.grid[1] as i32], now: now.1 as f32 },
            )
            .unwrap();
        // SAFETY: a full-screen triangle with no vertex inputs.
        unsafe { b.draw(3, 1, 0, 0) }.unwrap();
        b.end_render_pass(SubpassEndInfo::default()).unwrap();
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

    fn field_target(&self, extent: [u32; 2]) -> (Arc<ImageView>, Arc<Framebuffer>) {
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
            self.field_pass.clone(),
            FramebufferCreateInfo { attachments: vec![view.clone()], ..Default::default() },
        )
        .unwrap();
        (view, framebuffer)
    }

    fn ensure_targets(&mut self, extent: [u32; 2], sim: &FluidSim) {
        if self.accum.as_ref().is_some_and(|a| a.extent == extent) {
            return;
        }
        let wet_view = self.wet.as_ref().expect("the wet map comes first").view.clone();
        let (accum_view, accum) = self.field_target(extent);
        let (vapour_view, vapour) = self.field_target(extent);

        let wide_extent = [
            (extent[0] as f32 / WIDE_MIN_FACTOR).ceil().max(1.0) as u32,
            (extent[1] as f32 / WIDE_MIN_FACTOR).ceil().max(1.0) as u32,
        ];
        let (wide0_view, wide0) = self.field_target(wide_extent);
        let (wide1_view, wide1) = self.field_target(wide_extent);
        let sampled = |pipeline: &Arc<GraphicsPipeline>, view: &Arc<ImageView>| {
            DescriptorSet::new(
                self.gpu.descriptors.clone(),
                pipeline.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::image_view_sampler(0, view.clone(), self.sampler.clone())],
                [],
            )
            .unwrap()
        };
        let composite_set = DescriptorSet::new(
            self.gpu.descriptors.clone(),
            self.composite.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, accum_view.clone(), self.sampler.clone()),
                WriteDescriptorSet::buffer(1, sim.sdf_buffer().clone()),
                WriteDescriptorSet::image_view_sampler(2, wide0_view.clone(), self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(3, wet_view, self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(4, sim.cell_optics_view().clone(), self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(5, sim.glow_view().clone(), self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(6, sim.film_views().0.clone(), self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(10, sim.film_views().1.clone(), self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(7, vapour_view, self.sampler.clone()),
                WriteDescriptorSet::buffer(8, sim.optics_buffer().clone()),
                WriteDescriptorSet::image_view_sampler(9, sim.smoke_view().clone(), self.sampler.clone()),
            ],
            [],
        )
        .unwrap();
        self.accum = Some(Targets {
            extent,
            accum,
            wide: [wide0, wide1],
            wide_extent,
            down_set: sampled(&self.down, &accum_view),
            blur_sets: [sampled(&self.blur, &wide0_view), sampled(&self.blur, &wide1_view)],
            composite_set,
            vapour,
            vapour_dirty: std::cell::Cell::new(true),
        });
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
        self.track(b, sim);
        self.ensure_targets(extent, sim);
        if self.splat_set.is_none() {
            self.splat_set = Some(
                DescriptorSet::new(
                    self.gpu.descriptors.clone(),
                    self.splat.layout().set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::buffer(0, sim.particle_buffer().clone()),
                        WriteDescriptorSet::buffer(1, sim.material_buffer().clone()),
                    ],
                    [],
                )
                .unwrap(),
            );
        }
        if self.steam_set.is_none() {
            self.steam_set = Some(
                DescriptorSet::new(
                    self.gpu.descriptors.clone(),
                    self.steam.layout().set_layouts()[0].clone(),
                    [WriteDescriptorSet::buffer(0, sim.particle_buffer().clone())],
                    [],
                )
                .unwrap(),
            );
        }
        let target_fb = self.target_framebuffer(&target);
        let t = self.accum.as_ref().unwrap();
        let half = [extent[0] as f32 * 0.5, extent[1] as f32 * 0.5];
        sim.record_cell_optics(b, 1.0 / fast_speed(sim));
        let inline = || SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() };
        let clear = |fb: &Arc<Framebuffer>| RenderPassBeginInfo {
            clear_values: vec![Some([0.0; 4].into()); fb.attachments().len()],
            ..RenderPassBeginInfo::framebuffer(fb.clone())
        };

        // Pass 1: splat particles, into the top-left `accum_size` texels of the target.
        let div = (SPLAT_RADIUS * view.zoom / ACCUM_KERNEL_TEXELS).clamp(1.0, ACCUM_MAX_DIV);
        let accum_size = [
            ((extent[0] as f32 / div).ceil() as u32).clamp(1, extent[0]),
            ((extent[1] as f32 / div).ceil() as u32).clamp(1, extent[1]),
        ];
        let radius = SPLAT_RADIUS.max(SPLAT_MIN_TEXELS * div / view.zoom);
        let weight = (SPLAT_RADIUS / radius).powi(2);
        {
            let (fb, pipe, size, d, r) = (&t.accum, &self.splat, accum_size, div, radius);
            // Half the size, in pixels, of what those texels cover: the frame rounded up.
            let padded = [size[0] as f32 * d * 0.5, size[1] as f32 * d * 0.5];
            let scale = [view.zoom / padded[0], -view.zoom / padded[1]];
            b.begin_render_pass(clear(fb), inline()).unwrap();
            if sim.slots() > 0 {
                b.bind_pipeline_graphics(pipe.clone())
                    .unwrap()
                    .set_viewport(0, [viewport(size)].into_iter().collect())
                    .unwrap()
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, pipe.layout().clone(), 0, self.splat_set.clone().unwrap())
                    .unwrap()
                    .push_constants(
                        pipe.layout().clone(),
                        0,
                        SplatPc {
                            scale,
                            offset: [-view.center[0] * scale[0] + half[0] / padded[0] - 1.0, -view.center[1] * scale[1] + half[1] / padded[1] - 1.0],
                            radius: r,
                            weight,
                            exposure: EXPOSURE,
                            inv_speed: 1.0 / fast_speed(sim),
                        },
                    )
                    .unwrap();
                // SAFETY: the vertex shader indexes the particle buffer with gl_VertexIndex / 3,
                // which stays below the slot count.
                unsafe { b.draw(sim.slots() * 3, 1, 0, 0) }.unwrap();
            }
            b.end_render_pass(SubpassEndInfo::default()).unwrap();
        }

        // Pass 1b: the vapour field (bubbles and puffs of steam), in the same layout. Without
        // steam it is only cleared once: the composite pass binds it (and ignores it,
        // `vapour` 0), so it has to hold something.
        let steam = sim.has_steam() && sim.slots() > 0;
        if steam || t.vapour_dirty.replace(steam) {
            let (size, d) = (accum_size, div);
            let padded = [size[0] as f32 * d * 0.5, size[1] as f32 * d * 0.5];
            let scale = [view.zoom / padded[0], -view.zoom / padded[1]];
            b.begin_render_pass(clear(&t.vapour), inline()).unwrap();
            if steam {
                let bubble = BUBBLE_RADIUS.max(SPLAT_MIN_TEXELS * d / view.zoom);
                b.bind_pipeline_graphics(self.steam.clone())
                    .unwrap()
                    .set_viewport(0, [viewport(size)].into_iter().collect())
                    .unwrap()
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, self.steam.layout().clone(), 0, self.steam_set.clone().unwrap())
                    .unwrap()
                    .push_constants(
                        self.steam.layout().clone(),
                        0,
                        SteamPc {
                            scale,
                            offset: [-view.center[0] * scale[0] + half[0] / padded[0] - 1.0, -view.center[1] * scale[1] + half[1] / padded[1] - 1.0],
                            bubble,
                            bubble_w: (BUBBLE_RADIUS / bubble).powi(2),
                        },
                    )
                    .unwrap();
                // SAFETY: as the splat: gl_VertexIndex / 3 stays below the slot count.
                unsafe { b.draw(sim.slots() * 3, 1, 0, 0) }.unwrap();
            }
            b.end_render_pass(SubpassEndInfo::default()).unwrap();
        }

        // Passes 2 to 4: the wide field. Downsample, then blur across and down.
        let wide = WideLayout::new(view.zoom, div, extent, t.wide_extent);
        let wide_texel = [1.0 / t.wide_extent[0] as f32, 1.0 / t.wide_extent[1] as f32];
        let uv_max = [(wide.size[0] as f32 - 0.5) * wide_texel[0], (wide.size[1] as f32 - 0.5) * wide_texel[1]];
        b.begin_render_pass(clear(&t.wide[0]), inline()).unwrap();
        b.bind_pipeline_graphics(self.down.clone())
            .unwrap()
            .set_viewport(0, [viewport(wide.size)].into_iter().collect())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Graphics, self.down.layout().clone(), 0, t.down_set.clone())
            .unwrap()
            .push_constants(
                self.down.layout().clone(),
                0,
                DownPc {
                    src_texel: [1.0 / extent[0] as f32, 1.0 / extent[1] as f32],
                    src_max: [accum_size[0] as f32 - 0.5, accum_size[1] as f32 - 0.5],
                    factor: wide.factor / div,
                    taps: wide.taps,
                },
            )
            .unwrap();
        // SAFETY: a full-screen triangle with no vertex inputs (here and below).
        unsafe { b.draw(3, 1, 0, 0) }.unwrap();
        b.end_render_pass(SubpassEndInfo::default()).unwrap();
        for (pass, step) in [[wide.step * wide_texel[0], 0.0], [0.0, wide.step * wide_texel[1]]].into_iter().enumerate() {
            b.begin_render_pass(clear(&t.wide[1 - pass]), inline()).unwrap();
            b.bind_pipeline_graphics(self.blur.clone())
                .unwrap()
                .set_viewport(0, [viewport(wide.size)].into_iter().collect())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Graphics, self.blur.layout().clone(), 0, t.blur_sets[pass].clone())
                .unwrap()
                .push_constants(self.blur.layout().clone(), 0, BlurPc { texel: wide_texel, step, uv_max })
                .unwrap();
            unsafe { b.draw(3, 1, 0, 0) }.unwrap();
            b.end_render_pass(SubpassEndInfo::default()).unwrap();
        }

        // Pass 5: shade to the target.
        b.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(target_fb)
            },
            inline(),
        )
        .unwrap();
        b.bind_pipeline_graphics(if steam { self.composite.clone() } else { self.composite_dry.clone() })
            .unwrap()
            .set_viewport(0, [viewport(extent)].into_iter().collect())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.composite.layout().clone(),
                0,
                t.composite_set.clone(),
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
                    wide_scale: [wide_texel[0] / wide.factor, wide_texel[1] / wide.factor],
                    wide_max: uv_max,
                    // The kernel (1 - r^2)^2 integrates to pi R^2 / 3.
                    rest: PPC as f32 * std::f32::consts::PI * SPLAT_RADIUS * SPLAT_RADIUS / 3.0,
                    mist: 1.0 - weight,
                    accum_scale: [1.0 / (div * extent[0] as f32), 1.0 / (div * extent[1] as f32)],
                    accum_max: [(accum_size[0] as f32 - 0.5) / extent[0] as f32, (accum_size[1] as f32 - 0.5) / extent[1] as f32],
                    accum_div: div,

                    now: sim.sim_time() as f32,
                    wet_width: self.look.wet_width,
                    wet_tint: self.look.wet_tint,
                    glow: self.look.glow,
                    stains: self.look.stains,
                    vapour: if steam { 1.0 } else { 0.0 },
                    film_layer: sim.film_views().2 as f32,
                },
            )
            .unwrap();
        unsafe { b.draw(3, 1, 0, 0) }.unwrap();
        overlay(b);
        b.end_render_pass(SubpassEndInfo::default()).unwrap();
    }
}

/// The speed that reads as fast in this tank, cells/s: free fall across its height (the
/// default particle speed cap). Flow is shaded relative to it, so the look does not depend
/// on the grid resolution or the cell size.
fn fast_speed(sim: &FluidSim) -> f32 {
    let g = sim.params.g_cells();
    let g = if g > 0.0 { g } else { crate::sim::SimParams::default().g_cells() };
    (2.0 * g * sim.grid[1] as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_field_fits_its_image_at_every_zoom() {
        for extent in [[1280, 720], [1920, 1080], [641, 479], [1, 1]] {
            let image = [(extent[0] as f32 / WIDE_MIN_FACTOR).ceil() as u32, (extent[1] as f32 / WIDE_MIN_FACTOR).ceil() as u32];
            for zoom in [0.05, 0.5, 1.5, 2.5, 3.1, 8.0, 25.0, 60.0, 400.0, 1e4] {
                let div = (SPLAT_RADIUS * zoom / ACCUM_KERNEL_TEXELS).clamp(1.0, ACCUM_MAX_DIV);
                let w = WideLayout::new(zoom, div, extent, image);
                assert!(w.size[0] >= 1 && w.size[0] <= image[0] && w.size[1] >= 1 && w.size[1] <= image[1], "{w:?}");
                // Every source texel is covered by a tap of the downsampling pass.
                assert!(w.taps as f32 * 2.0 * div >= w.factor, "{w:?}");
                // The blur is as wide as asked while the factor is inside its limits.
                if w.factor > WIDE_MIN_FACTOR && w.factor < WIDE_MAX_FACTOR {
                    assert!((w.step * 2.0 - WIDE_SIGMA_TEXELS).abs() < 1e-3, "{w:?}");
                }
            }
        }
    }
}
