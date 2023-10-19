use std::borrow::Cow;

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::{
            encase, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
            BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
            BufferBinding, BufferBindingType, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, PipelineCache, ShaderDefVal, ShaderStages,
            ShaderType, StorageTextureAccess, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDimension, BufferDescriptor,
        },
        renderer::RenderDevice,
        Render, RenderApp, RenderSet,
    },
};

const SIZE: (u32, u32) = (1280, 720);
const WORKGROUP_SIZE: u32 = 256;
const NR_PARTICLES: u32 = WORKGROUP_SIZE * 128;

#[derive(Resource, Clone, ExtractResource)]
pub struct ComputeInput {
    dst_image: Handle<Image>,
}

pub struct ComputePlugin;

#[derive(Resource)]
pub struct ComputePipeline {
    bind_group_layout: BindGroupLayout,
    update_program: CachedComputePipelineId,
    draw_program: CachedComputePipelineId,
    clear_program: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct ComputeBindGroup(BindGroup);

#[derive(Default)]
pub struct ComputeNode {
    ready: bool,
}

#[derive(Clone, Resource, ExtractResource)]
pub struct ParticleBuffer {
    particles: Buffer,
    energies: Buffer,
}

#[derive(Clone, Copy, ShaderType)]
pub struct Particle {
    position: Vec2,
    velocity: Vec2,
    seed: u32,
}

pub fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(AssetPlugin::default().watch_for_changes()))
        .add_plugins(ComputePlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut images: ResMut<Assets<Image>>,
) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4 * 4],
        TextureFormat::Rgba32Float,
    );

    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let image = images.add(image);

    commands.spawn(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
            ..default()
        },
        texture: image.clone(),
        ..default()
    });

    let mut particles = vec![Particle {
        position: Vec2::ZERO,
        velocity: Vec2::ZERO,
        seed: 0,
    }; NR_PARTICLES as usize];

    for (i, p) in &mut particles.iter_mut().enumerate() {
        p.position = Vec2::new(
            rand::random::<f32>() * SIZE.0 as f32,
            rand::random::<f32>() * SIZE.1 as f32,
        );
        p.velocity = Vec2::new(
            rand::random::<f32>(),
            rand::random::<f32>(),
        );

        p.seed = i as u32;
    }

    let mut particle_byte_buffer: Vec<u8> = Vec::new();
    let mut particle_buffer = encase::StorageBuffer::new(&mut particle_byte_buffer);
    particle_buffer.write(&particles).unwrap();
    let particle_storage = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: None,
        usage: BufferUsages::STORAGE,
        contents: particle_buffer.into_inner(),
    });

    let energy_storage = render_device.create_buffer(&BufferDescriptor {
        label: None,
        size: (4 * SIZE.0 * SIZE.1) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    commands.spawn(Camera2dBundle::default());

    commands.insert_resource(ParticleBuffer {
        particles: particle_storage,
        energies: energy_storage,
    });
    commands.insert_resource(ComputeInput { dst_image: image });
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputePipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    inputs: Res<ComputeInput>,
    particles: Res<ParticleBuffer>,
    render_device: Res<RenderDevice>,
) {
    let view = gpu_images.get(&inputs.dst_image).unwrap();
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &particles.particles,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &particles.energies,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });
    commands.insert_resource(ComputeBindGroup(bind_group));
}

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<ParticleBuffer>::default());
        app.add_plugins(ExtractResourcePlugin::<ComputeInput>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
        );

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("compute", ComputeNode::default());
        render_graph.add_node_edge("compute", bevy::render::main_graph::node::CAMERA_DRIVER);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ComputePipeline>();
    }
}

impl FromWorld for ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba32Float,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/flow_field.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let from_entrypoint = |entry_point: &'static str| -> ComputePipelineDescriptor {
            ComputePipelineDescriptor {
                label: None,
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: Vec::new(),
                shader: shader.clone(),
                shader_defs: vec![
                    ShaderDefVal::UInt("NR_PARTICLES".to_string(), NR_PARTICLES),
                    ShaderDefVal::UInt("NR_PIXELS".to_string(), SIZE.0 * SIZE.1),
                    ShaderDefVal::UInt("SCREEN_WIDTH".to_string(), SIZE.0),
                ],
                entry_point: Cow::from(entry_point),
            }
        };

        let update_program = pipeline_cache.queue_compute_pipeline(from_entrypoint("update"));
        let draw_program = pipeline_cache.queue_compute_pipeline(from_entrypoint("draw"));
        let clear_program = pipeline_cache.queue_compute_pipeline(from_entrypoint("clear"));

        ComputePipeline {
            bind_group_layout,
            update_program,
            draw_program,
            clear_program,
        }
    }
}

impl render_graph::Node for ComputeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ComputePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if !self.ready {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.update_program)
            {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.draw_program)
                {
                    self.ready = true;
                }
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !self.ready {
            return Ok(());
        }

        let bind_group = &world.resource::<ComputeBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputePipeline>();
        let update_program = pipeline_cache
            .get_compute_pipeline(pipeline.update_program)
            .unwrap();
        let clear_program = pipeline_cache
            .get_compute_pipeline(pipeline.clear_program)
            .unwrap();
        let draw_program = pipeline_cache
            .get_compute_pipeline(pipeline.draw_program)
            .unwrap();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(update_program);
        pass.dispatch_workgroups(NR_PARTICLES / WORKGROUP_SIZE, 1, 1);
        pass.set_pipeline(clear_program);
        pass.dispatch_workgroups(SIZE.0 / 16, SIZE.1 / 16, 1);
        pass.set_pipeline(draw_program);
        pass.dispatch_workgroups(SIZE.0 / 16, SIZE.1 / 16, 1);

        Ok(())
    }
}
