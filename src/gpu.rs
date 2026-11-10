//! Vulkan device setup shared by the windowed app and the headless tools.

use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    shader::{EntryPoint, ShaderModule, ShaderModuleCreateInfo, spirv::bytes_to_words},
    swapchain::Surface,
};
use winit::event_loop::EventLoop;

pub type Builder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

pub struct Gpu {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory: Arc<StandardMemoryAllocator>,
    pub descriptors: Arc<StandardDescriptorSetAllocator>,
    pub commands: Arc<StandardCommandBufferAllocator>,
}

impl Gpu {
    /// Pass the event loop to get a device that can present to its windows; pass `None`
    /// for off-screen work.
    pub fn new(event_loop: Option<&EventLoop<()>>) -> Arc<Gpu> {
        let library = VulkanLibrary::new().expect("no Vulkan loader found on this system");
        let instance_extensions = match event_loop {
            Some(el) => Surface::required_extensions(el).expect("no window system support"),
            None => Default::default(),
        };
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: instance_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create a Vulkan instance");

        let device_extensions = DeviceExtensions {
            khr_swapchain: event_loop.is_some(),
            ..DeviceExtensions::empty()
        };

        let usable = |p: &Arc<PhysicalDevice>| -> Option<u32> {
            if !p.supported_extensions().contains(&device_extensions) {
                return None;
            }
            p.queue_family_properties().iter().enumerate().position(|(i, q)| {
                q.queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                    && event_loop.is_none_or(|el| p.presentation_support(i as u32, el).unwrap_or(false))
            })
            .map(|i| i as u32)
        };
        let (physical, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate Vulkan devices")
            .filter_map(|p| usable(&p).map(|i| (p, i)))
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no suitable Vulkan device found");

        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
                ..Default::default()
            },
        )
        .expect("failed to create a Vulkan device");
        let queue = queues.next().unwrap();

        Arc::new(Gpu {
            instance,
            memory: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            descriptors: Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default())),
            commands: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                // The solver caches a few secondary command buffers (see `FluidSim::grid_part`).
                StandardCommandBufferAllocatorCreateInfo { secondary_buffer_count: 4, ..Default::default() },
            )),
            device,
            queue,
        })
    }

    pub fn device_name(&self) -> String {
        let p = self.device.physical_device().properties();
        format!("{} ({:?})", p.device_name, p.device_type)
    }

    pub fn builder(&self) -> Builder {
        AutoCommandBufferBuilder::primary(
            self.commands.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    /// Loads a SPIR-V module produced by `build.rs` and returns its `main` entry point.
    pub fn shader(&self, spirv: &[u8]) -> EntryPoint {
        let words = bytes_to_words(spirv).expect("corrupt SPIR-V");
        // SAFETY: the SPIR-V comes straight from glslc at build time.
        let module = unsafe { ShaderModule::new(self.device.clone(), ShaderModuleCreateInfo::new(&words)) }
            .expect("failed to create shader module");
        module.entry_point("main").expect("shader has no main()")
    }
}

/// Embeds a shader compiled by `build.rs`, e.g. `spirv!("particles.comp")`.
#[macro_export]
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")).as_slice()
    };
}
