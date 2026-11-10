//! A small egui painter for Vulkano.
//!
//! It draws inside the renderer's final render pass, so the UI costs no extra pass, and it
//! compiles its two tiny shaders with the rest of Sprout's instead of pulling in a second
//! shader toolchain.

use std::{collections::HashMap, sync::Arc};

use egui::{ClippedPrimitive, TextureId, TexturesDelta, epaint::Primitive};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{BufferImageCopy, CopyBufferToImageInfo},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageSubresourceLayers, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{
                VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState,
            },
            viewport::{Scissor, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::Subpass,
};

use crate::{
    gpu::{Builder, Gpu},
    spirv,
};

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
struct GuiVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    color: [u8; 4],
}

pub struct GuiPainter {
    gpu: Arc<Gpu>,
    pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    textures: HashMap<TextureId, (Arc<ImageView>, Arc<DescriptorSet>)>,
}

impl GuiPainter {
    pub fn new(gpu: Arc<Gpu>, subpass: Subpass) -> GuiPainter {
        let device = gpu.device.clone();
        let vs = gpu.shader(spirv!("gui.vert"));
        let fs = gpu.shader(spirv!("gui.frag"));
        // Spelled out by location: the optimised SPIR-V carries no attribute names to match on.
        let attribute = |format, offset| VertexInputAttributeDescription { binding: 0, format, offset, ..Default::default() };
        let vertex_input_state = VertexInputState::new()
            .binding(
                0,
                VertexInputBindingDescription {
                    stride: size_of::<GuiVertex>() as u32,
                    input_rate: VertexInputRate::Vertex,
                    ..Default::default()
                },
            )
            .attribute(0, attribute(Format::R32G32_SFLOAT, 0))
            .attribute(1, attribute(Format::R32G32_SFLOAT, 8))
            .attribute(2, attribute(Format::R8G8B8A8_UNORM, 16));
        let stages = [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        // egui emits premultiplied alpha.
        let blend = AttachmentBlend {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::OneMinusDstAlpha,
            dst_alpha_blend_factor: BlendFactor::One,
            alpha_blend_op: BlendOp::Add,
        };
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState { blend: Some(blend), ..Default::default() },
                )),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();
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
        GuiPainter { gpu, pipeline, sampler, textures: HashMap::new() }
    }

    /// Uploads new or changed egui textures. Call before the render pass begins.
    pub fn update_textures(&mut self, b: &mut Builder, delta: &TexturesDelta) {
        for (id, image_delta) in &delta.set {
            let pixels: Vec<u8> = match &image_delta.image {
                egui::ImageData::Color(image) => image.pixels.iter().flat_map(|c| c.to_array()).collect(),
                egui::ImageData::Font(image) => image.srgba_pixels(None).flat_map(|c| c.to_array()).collect(),
            };
            let size = [image_delta.image.width() as u32, image_delta.image.height() as u32];
            let staging = Buffer::from_iter(
                self.gpu.memory.clone(),
                BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                pixels,
            )
            .unwrap();

            if let Some(pos) = image_delta.pos {
                // Partial update of an existing texture (the font atlas grows this way).
                let Some((view, _)) = self.textures.get(id) else { continue };
                b.copy_buffer_to_image(CopyBufferToImageInfo {
                    regions: [BufferImageCopy {
                        image_offset: [pos[0] as u32, pos[1] as u32, 0],
                        image_extent: [size[0], size[1], 1],
                        image_subresource: ImageSubresourceLayers {
                            aspects: ImageAspects::COLOR,
                            mip_level: 0,
                            array_layers: 0..1,
                        },
                        ..Default::default()
                    }]
                    .into(),
                    ..CopyBufferToImageInfo::buffer_image(staging, view.image().clone())
                })
                .unwrap();
            } else {
                // Texel values are already gamma-encoded; store them untouched.
                let image = Image::new(
                    self.gpu.memory.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R8G8B8A8_UNORM,
                        extent: [size[0], size[1], 1],
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap();
                b.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone())).unwrap();
                let view = ImageView::new_default(image).unwrap();
                let set = DescriptorSet::new(
                    self.gpu.descriptors.clone(),
                    self.pipeline.layout().set_layouts()[0].clone(),
                    [WriteDescriptorSet::image_view_sampler(0, view.clone(), self.sampler.clone())],
                    [],
                )
                .unwrap();
                self.textures.insert(*id, (view, set));
            }
        }
    }

    /// Releases textures egui no longer needs. Call after the frame has been recorded.
    pub fn free_textures(&mut self, delta: &TexturesDelta) {
        for id in &delta.free {
            self.textures.remove(id);
        }
    }

    /// Draws tessellated egui output. Must be called inside the subpass given to `new`.
    pub fn paint(&mut self, b: &mut Builder, primitives: &[ClippedPrimitive], pixels_per_point: f32, size: [u32; 2]) {
        let mut vertices: Vec<GuiVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        // (clip rect, texture, first index, index count, vertex offset)
        let mut draws: Vec<(egui::Rect, TextureId, u32, u32, i32)> = Vec::new();
        for ClippedPrimitive { clip_rect, primitive } in primitives {
            let Primitive::Mesh(mesh) = primitive else { continue };
            if mesh.indices.is_empty() || mesh.vertices.is_empty() {
                continue;
            }
            draws.push((*clip_rect, mesh.texture_id, indices.len() as u32, mesh.indices.len() as u32, vertices.len() as i32));
            vertices.extend(mesh.vertices.iter().map(|v| GuiVertex {
                position: [v.pos.x, v.pos.y],
                tex_coords: [v.uv.x, v.uv.y],
                color: v.color.to_array(),
            }));
            indices.extend_from_slice(&mesh.indices);
        }
        if draws.is_empty() {
            return;
        }

        let host = |usage| {
            (
                BufferCreateInfo { usage, ..Default::default() },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
        };
        let (info, alloc) = host(BufferUsage::VERTEX_BUFFER);
        let vertex_buffer = Buffer::from_iter(self.gpu.memory.clone(), info, alloc, vertices).unwrap();
        let (info, alloc) = host(BufferUsage::INDEX_BUFFER);
        let index_buffer = Buffer::from_iter(self.gpu.memory.clone(), info, alloc, indices).unwrap();

        let layout = self.pipeline.layout().clone();
        b.bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer)
            .unwrap()
            .bind_index_buffer(index_buffer)
            .unwrap()
            .push_constants(layout.clone(), 0, [size[0] as f32 / pixels_per_point, size[1] as f32 / pixels_per_point])
            .unwrap();

        let mut bound: Option<TextureId> = None;
        for (clip, texture, first_index, count, vertex_offset) in draws {
            let Some((_, set)) = self.textures.get(&texture) else { continue };
            let x0 = (clip.min.x * pixels_per_point).round().clamp(0.0, size[0] as f32);
            let y0 = (clip.min.y * pixels_per_point).round().clamp(0.0, size[1] as f32);
            let x1 = (clip.max.x * pixels_per_point).round().clamp(x0, size[0] as f32);
            let y1 = (clip.max.y * pixels_per_point).round().clamp(y0, size[1] as f32);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            if bound != Some(texture) {
                b.bind_descriptor_sets(PipelineBindPoint::Graphics, layout.clone(), 0, set.clone()).unwrap();
                bound = Some(texture);
            }
            b.set_scissor(
                0,
                [Scissor { offset: [x0 as u32, y0 as u32], extent: [(x1 - x0) as u32, (y1 - y0) as u32] }]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
            // SAFETY: index and vertex ranges come straight from the meshes packed above.
            unsafe { b.draw_indexed(count, 1, first_index, vertex_offset, 0) }.unwrap();
        }
    }
}
