use std::borrow::Cow;

use bevy::{
    core::FrameCount,
    core_pipeline::bloom::BloomSettings,
    prelude::*,
    reflect::{TypePath, TypeUuid},
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::{
            AsBindGroup, CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, PipelineCache, PreparedBindGroup,
            TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::RenderDevice,
        texture::FallbackImage,
        Render, RenderApp, RenderSet,
    },
};

const SIZE: (u32, u32) = (1280, 720);
const WORKGROUP_SIZE: (u32, u32) = (16, 16);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(AssetPlugin::default().watch_for_changes()))
        .add_plugins(FirePlugin)
        .insert_resource(ClearColor(Color::BLACK))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
    );

    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let mut dst_images = Vec::new();
    dst_images.push(images.add(image.clone()));
    dst_images.push(images.add(image.clone()));

    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
                ..default()
            },
            texture: dst_images[0].clone().into(),
            ..default()
        },
    ));

    commands.spawn(Camera2dBundle::default());
    commands.insert_resource(FireConfig { dst_images })
}

struct FirePlugin;

impl Plugin for FirePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<FireConfig>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_fire_bindings.in_set(RenderSet::PrepareBindGroups),
        );

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("fire", FireNode::default());
        render_graph.add_node_edge("fire", bevy::render::main_graph::node::CAMERA_DRIVER);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<FirePipeline>();
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct FireConfig {
    dst_images: Vec<Handle<Image>>,
}

#[derive(Resource, AsBindGroup)]
struct FireBindings {
    #[storage_texture(0, image_format = R32Float, access = WriteOnly)]
    dst_image: Handle<Image>,

    #[storage_texture(1, image_format = R32Float, access = ReadOnly)]
    src_image: Handle<Image>,

    #[uniform(2)]
    time: f32,

    #[uniform(3)]
    resolution: IVec2,

    bind_group: Option<PreparedBindGroup<()>>,
}

fn prepare_fire_bindings(
    mut commands: Commands,
    gpu_images: Res<RenderAssets<Image>>,
    fire_config: ResMut<FireConfig>,
    render_device: Res<RenderDevice>,
    fallback_image: Res<FallbackImage>,
    time: Res<Time>,
    frame_counter: Res<FrameCount>,
) {
    let mut bindings = FireBindings {
        dst_image: fire_config.dst_images[(frame_counter.0 + 0) as usize % 2].clone(),
        src_image: fire_config.dst_images[(frame_counter.0 + 1) as usize % 2].clone(),
        time: time.elapsed_seconds(),
        resolution: IVec2::new(SIZE.0 as i32, SIZE.1 as i32),
        bind_group: None,
    };

    bindings.bind_group = bindings
        .as_bind_group(
            &FireBindings::bind_group_layout(&render_device),
            &render_device,
            &gpu_images,
            &fallback_image,
        )
        .ok();

    commands.insert_resource(bindings);
}

#[derive(Resource)]
struct FirePipeline {
    pipeline: CachedComputePipelineId,
}

impl FromWorld for FirePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let shader = world.resource::<AssetServer>().load("shaders/fire.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![FireBindings::bind_group_layout(render_device)],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("main"),
        });

        FirePipeline { pipeline }
    }
}

#[derive(Default)]
struct FireNode {
    ready: bool,
}

impl render_graph::Node for FireNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<FirePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        self.ready = matches!(
            pipeline_cache.get_compute_pipeline_state(pipeline.pipeline),
            CachedPipelineState::Ok(_)
        );
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<FirePipeline>();
        let texture_bind_group = world
            .resource::<FireBindings>()
            .bind_group
            .as_ref()
            .unwrap();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &texture_bind_group.bind_group, &[]);

        if self.ready {
            let pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.pipeline)
                .unwrap();
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE.0, SIZE.1 / WORKGROUP_SIZE.1, 1);
        }

        Ok(())
    }
}

#[derive(AsBindGroup, TypeUuid, TypePath, Debug, Clone)]
#[uuid = "f690fdae-d598-45ab-8225-97e2a3f056e0"]
pub struct FireMaterial {
    // Uniform bindings must implement `ShaderType`, which will be used to convert the value to
    // its shader-compatible equivalent. Most core math types already implement `ShaderType`.
    #[uniform(0)]
    color: Color,
    // Images can be bound as textures in shaders. If the Image's sampler is also needed, just
    // add the sampler attribute with a different binding index.
    #[texture(1)]
    #[sampler(2)]
    color_texture: Handle<Image>,
}
