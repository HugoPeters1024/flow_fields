@group(0) @binding(0) var dst_image: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var src_image: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var<uniform> time: f32;
@group(0) @binding(3) var<uniform> resolution: vec2<i32>;

fn permute4(x: vec4f) -> vec4f { return ((x * 34. + 1.) * x) % vec4f(289.); }
fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2f) -> f32 {
    var Pi: vec4f = floor(P.xyxy) + vec4f(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4f(0., 0., 1., 1.);
    Pi = Pi % vec4f(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4f = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2f = vec2f(gx.x, gy.x);
    var g10: vec2f = vec2f(gx.y, gy.y);
    var g01: vec2f = vec2f(gx.z, gy.z);
    var g11: vec2f = vec2f(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 * vec4f(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2f(fx.x, fy.x));
    let n10 = dot(g10, vec2f(fx.y, fy.y));
    let n01 = dot(g01, vec2f(fx.z, fy.z));
    let n11 = dot(g11, vec2f(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2f(n00, n01), vec2f(n10, n11), vec2f(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}


fn average_neighbours(image: texture_storage_2d<r32float, read>, location: vec2<i32>) -> f32 {
    let sum = textureLoad(image, location + vec2<i32>(-1, 0)).r + textureLoad(image, location + vec2<i32>(1, 0)).r + textureLoad(image, location + vec2<i32>(0, -1)).r + textureLoad(image, location + vec2<i32>(0, 1)).r;
    return sum / 4.0;
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let diffused = average_neighbours(src_image, location);

    var noise = 0.0;
    noise += perlinNoise2(
        13.0 * vec2f(f32(location.x) / 640.0, 0.2 * time + f32(location.y) / 480.0)
    );
    noise += perlinNoise2(
        13.0 * vec2f(0.3 + f32(location.x) / 640.0, 0.3 * time + f32(location.y) / 480.0)
    );

    let cooling = abs(noise) * pow(f32(resolution.y - location.y) / f32(resolution.y), 2.0);

    //var temperature = diffused * (1.0 - 0.2 * pow(cooling,3.0));
    var temperature = diffused - cooling;

    if location.y >= resolution.y - 1 {
        temperature = 1.0;
    }

    var velocity = 1;
    if temperature > 0.4 {
        velocity = 1;
    }


    textureStore(dst_image, location - vec2(0, velocity), vec4<f32>(temperature));
}

