struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  seed: u32,
}

@group(0) @binding(0) var dst_image: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> energy_buffer: array<atomic<u32>>;

fn xxhash32(n: u32) -> u32 {
    var h32 = n + 374761393u;
    h32 = 668265263u * ((h32 << 17u) | (h32 >> (32u - 17u)));
    h32 = 2246822519u * (h32 ^ (h32 >> 15u));
    h32 = 3266489917u * (h32 ^ (h32 >> 13u));
    return h32^(h32 >> 16u);
}

fn randf(n: u32) -> f32 {
  particles[n].seed = xxhash32(particles[n].seed);
  return f32(particles[n].seed) / 4294967296.0;
}

//  MIT License. © Ian McEwan, Stefan Gustavson, Munrocket, Johan Helsing
//
fn mod289(x: vec2f) -> vec2f {
    return x - floor(x * (1. / 289.)) * 289.;
}

fn mod289_3(x: vec3f) -> vec3f {
    return x - floor(x * (1. / 289.)) * 289.;
}

fn permute3(x: vec3f) -> vec3f {
    return mod289_3(((x * 34.) + 1.) * x);
}

//  MIT License. © Ian McEwan, Stefan Gustavson, Munrocket
fn simplexNoise2(v: vec2f) -> f32 {
    let C = vec4(
        0.211324865405187, // (3.0-sqrt(3.0))/6.0
        0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
        -0.577350269189626, // -1.0 + 2.0 * C.x
        0.024390243902439 // 1.0 / 41.0
    );

    // First corner
    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);

    // Other corners
    var i1 = select(vec2(0., 1.), vec2(1., 0.), x0.x > x0.y);

    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    var x12 = x0.xyxy + C.xxzz;
    x12.x = x12.x - i1.x;
    x12.y = x12.y - i1.y;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation

    var p = permute3(permute3(i.y + vec3(0., i1.y, 1.)) + i.x + vec3(0., i1.x, 1.));
    var m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.));
    m *= m;
    m *= m;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
    let x = 2. * fract(p * C.www) - 1.;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;

    // Normalize gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    // Compute final noise value at P
    let g = vec3(a0.x * x0.x + h.x * x0.y, a0.yz * x12.xz + h.yz * x12.yw);
    return 130. * dot(m, g);
}

@compute @workgroup_size(256,1,1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let pid = invocation_id.x;
    let particle = particles[pid];

    let plocf = vec2<f32>(particle.position) / 100.0;

    let angle = simplexNoise2(plocf / 2.8) * 3.14159;
    let dir = vec2<f32>(cos(angle), sin(angle));

    let alpha = 0.01;

    particles[pid].velocity = (particles[pid].velocity * (1.0 - alpha)) + (dir * alpha);
    particles[pid].position += particles[pid].velocity * 0.3;

    if (particles[pid].position.x >= 1280.0
       || particles[pid].position.x < 0.0
       || particles[pid].position.y >= 720.0
       || particles[pid].position.y < 0.0) {
        particles[pid].position.x = randf(pid) * 1280.0;
        particles[pid].position.y = randf(pid) * 720.0;
        particles[pid].velocity.x = randf(pid) * 2.0 - 1.0;
        particles[pid].velocity.y = randf(pid) * 2.0 - 1.0;
    }

    let p = particles[pid].position;

    var oldValue: u32 = atomicAdd(&energy_buffer[u32(p.x) + #{SCREEN_WIDTH}u * u32(p.y)], 1u);
}

@compute @workgroup_size(16,16,1)
fn draw(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let pxl_id = invocation_id.x + #{SCREEN_WIDTH}u * invocation_id.y;
    let energy : u32 = atomicLoad(&energy_buffer[pxl_id]);

    textureStore(dst_image, vec2(invocation_id.x, invocation_id.y), vec4(vec3(0.0, 0.0, 0.01) + vec3(1.0) * f32(energy)/1000.0, 1.0));
}

@compute @workgroup_size(16,16,1)
fn clear(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<u32>(invocation_id.xy);
    let locationf = vec2<f32>(location) / 130.0;
    textureStore(dst_image, location, vec4(vec3(0.0), 1.0));
}
