/**
 * Shaders, WGSL (WebGPU) and GLSL ES 3.00 (WebGL2), with the same inputs and the same arithmetic.
 *
 * Uniform block U (240 bytes, std140-compatible): view, proj (column-major mat4), viewport = (width px, height px, dpr,
 * ortho 0/1), light = (view-space direction xyz, ambient), accent = (selection rgb, ring threshold on n_z), line =
 * (rgb, width in device px), look = (mode, outline width in device px, fog strength, 0), fog = (view depth where fog
 * starts, depth where it is full, 0, 0), bg = (background rgb, 1).
 * Lighting by look mode, with d = n.L, h = max(n.H, 0), H = normalize(L + V), V = direction to the eye:
 *   0 classic  base (a + (1 - a) max(d, 0)) + 0.22 h^60
 *   1 soft     base (A + 0.5 w^2) + 0.06 h^16, wrap w = clamp((d + 0.5) / 1.5), A = 0.38 + 0.17 (1 + n_y) / 2 (sky above)
 *   2 glossy   base (0.2 + 0.8 max(d, 0)) + 0.6 h^160 + 0.12 h^16 + 0.15 (1 - n.V)^3
 *   3 toon     base x three bands (0.56, 0.78, 1 at d = 0.15, 0.6) + 0.35 x a highlight disc (h > 0.965)
 *   4 outline  base x (0.88 + 0.12 step(d > -0.25)), i.e. flat colour
 * Outline (look.y > 0): colour -> 0.18 base within look.y device pixels of the silhouette. Spheres: the distance to the
 * edge is r_px (1 - sqrt(1 - disc / r^2)), disc of the ray-sphere intersection, r_px the projected radius; bond
 * cylinders: r_px (1 - sqrt(1 - c^2)), c = |n.V| / max(|w x V|, 0.05), w the view-space bond axis (|n.V| alone
 * would reach only sin(angle(w, V)), so a bond pointing at the eye would be all edge); surfaces darken where
 * |n.V| < 0.2. Fog (look.z > 0): mix toward bg by
 * look.z clamp((z - fog.x) / (fog.y - fog.x)), z the view depth of the fragment.
 * Spheres: a camera-facing quad of half-size s around the view-space centre c_v (s = r d / sqrt(d^2 - r^2) * d / |z_c|
 * for perspective, which covers the silhouette; s = r for orthographic). The fragment shader intersects the eye ray
 * with the sphere |p - c_v| = r: t = -b - sqrt(b^2 - c), b = d.(o - c_v), c = |o - c_v|^2 - r^2, and writes the depth of
 * the hit point. Bonds: 16-sided open cylinders from per-vertex (cos t, sin t, h) and per-bond end points; the half
 * h < 1/2 takes the first atom's colour. Lines: a quad around the projected segment, offset by +-width/2 pixels along
 * the screen normal; dashed lines drop fragments where fract(s / (7 dpr)) > 0.55, s = pixels along the segment.
 * Meshes: two-sided lighting (the normal is flipped to face the eye).
 */

export const WGSL = /* wgsl */ `
struct U { view: mat4x4f, proj: mat4x4f, viewport: vec4f, light: vec4f, accent: vec4f, line: vec4f, look: vec4f, fog: vec4f, bg: vec4f };
@group(0) @binding(0) var<uniform> u: U;
struct M { color: vec4f };
@group(1) @binding(0) var<uniform> m: M;

fn shade(base: vec3f, n: vec3f, v: vec3f) -> vec3f {
  let L = normalize(u.light.xyz);
  let d = dot(n, L);
  let h = max(dot(n, normalize(L + v)), 0.0);
  let mode = u.look.x;
  if (mode < 0.5) { return base * (u.light.w + (1.0 - u.light.w) * max(d, 0.0)) + vec3f(0.22 * pow(h, 60.0)); }
  if (mode < 1.5) {
    let w = clamp((d + 0.5) / 1.5, 0.0, 1.0);
    return base * (0.38 + 0.17 * (0.5 + 0.5 * n.y) + 0.5 * w * w) + vec3f(0.06 * pow(h, 16.0));
  }
  if (mode < 2.5) {
    let f = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0);
    return base * (0.2 + 0.8 * max(d, 0.0)) + vec3f(0.6 * pow(h, 160.0) + 0.12 * pow(h, 16.0) + 0.15 * f);
  }
  if (mode < 3.5) {
    let band = 0.56 + 0.22 * smoothstep(0.12, 0.18, d) + 0.22 * smoothstep(0.57, 0.63, d);
    return base * band + vec3f(0.35 * smoothstep(0.955, 0.975, h));
  }
  return base * (0.88 + 0.12 * smoothstep(-0.3, -0.2, d));
}
fn post(c: vec3f, base: vec3f, edge: f32, z: f32) -> vec3f {
  var o = c;
  if (u.look.y > 0.0) { o = mix(base * 0.18, o, smoothstep(u.look.y - 0.6, u.look.y + 0.6, edge)); }
  if (u.look.z > 0.0) { o = mix(o, u.bg.rgb, u.look.z * clamp((z - u.fog.x) / max(u.fog.y - u.fog.x, 1e-3), 0.0, 1.0)); }
  return o;
}

struct SV {
  @builtin(position) pos: vec4f,
  @location(0) vp: vec3f,
  @location(1) @interpolate(flat) vc: vec3f,
  @location(2) @interpolate(flat) r: f32,
  @location(3) @interpolate(flat) col: vec4f,
  @location(4) @interpolate(flat) rpx: f32,
};
@vertex fn vs_sphere(@location(0) q: vec2f, @location(1) c: vec3f, @location(2) r: f32, @location(3) col: vec4f) -> SV {
  var o: SV;
  let vc = (u.view * vec4f(c, 1.0)).xyz;
  var s = r * 1.02;
  if (u.viewport.w < 0.5) {
    let d = length(vc);
    s = r * d / sqrt(max(d * d - r * r, 1e-6)) * d / max(-vc.z, 1e-6) * 1.02;
  }
  let vp = vec3f(vc.xy + q * s, vc.z);
  o.pos = u.proj * vec4f(vp, 1.0);
  o.vp = vp;
  o.vc = vc;
  o.r = r;
  o.col = col;
  var ppu = u.proj[1][1] * 0.5 * u.viewport.y;
  if (u.viewport.w < 0.5) { ppu = ppu / max(-vc.z, 1e-6); }
  o.rpx = r * ppu;
  return o;
}
struct SF { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 };
@fragment fn fs_sphere(i: SV) -> SF {
  var ro = vec3f(0.0, 0.0, 0.0);
  var rd = normalize(i.vp);
  if (u.viewport.w > 0.5) {
    ro = vec3f(i.vp.xy, 0.0);
    rd = vec3f(0.0, 0.0, -1.0);
  }
  let oc = ro - i.vc;
  let b = dot(rd, oc);
  let cc = dot(oc, oc) - i.r * i.r;
  let disc = b * b - cc;
  if (disc < 0.0) { discard; }
  let t = -b - sqrt(disc);
  let hit = ro + t * rd;
  let n = (hit - i.vc) / i.r;
  let clip = u.proj * vec4f(hit, 1.0);
  var base = i.col.rgb;
  if (i.col.a < 0.75 && n.z < u.accent.w) { base = u.accent.rgb; }
  var o: SF;
  o.depth = clip.z / clip.w;
  let edge = i.rpx * (1.0 - sqrt(max(0.0, 1.0 - disc / (i.r * i.r))));
  o.color = vec4f(post(shade(base, n, -rd), base, edge, -hit.z), 1.0);
  return o;
}

struct BV {
  @builtin(position) pos: vec4f,
  @location(0) n: vec3f,
  @location(1) h: f32,
  @location(2) vpos: vec3f,
  @location(3) @interpolate(flat) c0: vec4f,
  @location(4) @interpolate(flat) c1: vec4f,
  @location(5) rpx: f32,
  @location(6) @interpolate(flat) wv: vec3f,
};
@vertex fn vs_bond(@location(0) g: vec3f, @location(1) p0: vec3f, @location(2) p1: vec3f, @location(3) c0: vec4f,
                   @location(4) c1: vec4f, @location(5) r: f32) -> BV {
  var o: BV;
  let ax = p1 - p0;
  let w = ax / max(length(ax), 1e-6);
  var a = vec3f(1.0, 0.0, 0.0);
  if (abs(w.x) > 0.9) { a = vec3f(0.0, 1.0, 0.0); }
  let e1 = normalize(cross(w, a));
  let e2 = cross(w, e1);
  let nw = e1 * g.x + e2 * g.y;
  let v4 = u.view * vec4f(p0 + nw * r + ax * g.z, 1.0);
  o.pos = u.proj * v4;
  o.vpos = v4.xyz;
  o.n = (u.view * vec4f(nw, 0.0)).xyz;
  o.h = g.z;
  o.c0 = c0;
  o.c1 = c1;
  var ppu = u.proj[1][1] * 0.5 * u.viewport.y;
  if (u.viewport.w < 0.5) { ppu = ppu / max(-v4.z, 1e-6); }
  o.rpx = r * ppu;
  o.wv = (u.view * vec4f(w, 0.0)).xyz;
  return o;
}
@fragment fn fs_bond(i: BV) -> @location(0) vec4f {
  var base = i.c0.rgb;
  if (i.h > 0.5) { base = i.c1.rgb; }
  var v = normalize(-i.vpos);
  if (u.viewport.w > 0.5) { v = vec3f(0.0, 0.0, 1.0); }
  let nn = normalize(i.n);
  let nv = min(abs(dot(nn, v)) / max(length(cross(i.wv, v)), 0.05), 1.0);
  return vec4f(post(shade(base, nn, v), base, i.rpx * (1.0 - sqrt(max(0.0, 1.0 - nv * nv))), -i.vpos.z), 1.0);
}

struct LV {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(linear) s: f32,
  @location(1) @interpolate(flat) dash: f32,
};
@vertex fn vs_line(@location(0) g: vec2f, @location(1) p0: vec3f, @location(2) p1: vec3f, @location(3) dash: f32) -> LV {
  var o: LV;
  let a = u.proj * u.view * vec4f(p0, 1.0);
  let b = u.proj * u.view * vec4f(p1, 1.0);
  let vp = u.viewport.xy;
  let sa = a.xy / a.w * 0.5 * vp;
  let sb = b.xy / b.w * 0.5 * vp;
  let d = sb - sa;
  let len = max(length(d), 1e-6);
  let dir = d / len;
  let nr = vec2f(-dir.y, dir.x);
  let c = mix(a, b, g.x);
  o.pos = c + vec4f(nr * g.y * u.line.w / vp * c.w, 0.0, 0.0);
  o.s = g.x * len;
  o.dash = dash;
  return o;
}
@fragment fn fs_line(i: LV) -> @location(0) vec4f {
  if (i.dash > 0.5 && fract(i.s / (7.0 * u.viewport.z)) > 0.55) { discard; }
  return vec4f(u.line.rgb, 1.0);
}

struct MV {
  @builtin(position) @invariant pos: vec4f,
  @location(0) n: vec3f,
  @location(1) vpos: vec3f,
};
@vertex fn vs_mesh(@location(0) p: vec3f, @location(1) n: vec4f) -> MV {
  var o: MV;
  let v4 = u.view * vec4f(p, 1.0);
  o.pos = u.proj * v4;
  o.vpos = v4.xyz;
  o.n = (u.view * vec4f(n.xyz, 0.0)).xyz;
  return o;
}
@fragment fn fs_mesh(i: MV) -> @location(0) vec4f {
  var n = normalize(i.n);
  var v = normalize(-i.vpos);
  if (u.viewport.w > 0.5) { v = vec3f(0.0, 0.0, 1.0); }
  if (dot(n, v) < 0.0) { n = -n; }
  var c = shade(m.color.rgb, n, v);
  if (u.look.y > 0.0) { c = mix(m.color.rgb * 0.18, c, smoothstep(0.1, 0.22, dot(n, v))); }
  return vec4f(post(c, m.color.rgb, 1e4, -i.vpos.z), m.color.a);
}
@fragment fn fs_depth(i: MV) -> @location(0) vec4f {
  return vec4f(0.0, 0.0, 0.0, 0.0);
}
`;

const GL_HEAD = `#version 300 es
precision highp float;
layout(std140) uniform U { mat4 view; mat4 proj; vec4 viewport; vec4 light; vec4 accent; vec4 line; vec4 look; vec4 fog; vec4 bg; };
vec3 shade(vec3 base, vec3 n, vec3 v) {
  vec3 L = normalize(light.xyz);
  float d = dot(n, L);
  float h = max(dot(n, normalize(L + v)), 0.0);
  float mode = look.x;
  if (mode < 0.5) return base * (light.w + (1.0 - light.w) * max(d, 0.0)) + vec3(0.22 * pow(h, 60.0));
  if (mode < 1.5) {
    float w = clamp((d + 0.5) / 1.5, 0.0, 1.0);
    return base * (0.38 + 0.17 * (0.5 + 0.5 * n.y) + 0.5 * w * w) + vec3(0.06 * pow(h, 16.0));
  }
  if (mode < 2.5) {
    float f = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0);
    return base * (0.2 + 0.8 * max(d, 0.0)) + vec3(0.6 * pow(h, 160.0) + 0.12 * pow(h, 16.0) + 0.15 * f);
  }
  if (mode < 3.5) {
    float band = 0.56 + 0.22 * smoothstep(0.12, 0.18, d) + 0.22 * smoothstep(0.57, 0.63, d);
    return base * band + vec3(0.35 * smoothstep(0.955, 0.975, h));
  }
  return base * (0.88 + 0.12 * smoothstep(-0.3, -0.2, d));
}
vec3 post(vec3 c, vec3 base, float edge, float z) {
  vec3 o = c;
  if (look.y > 0.0) o = mix(base * 0.18, o, smoothstep(look.y - 0.6, look.y + 0.6, edge));
  if (look.z > 0.0) o = mix(o, bg.rgb, look.z * clamp((z - fog.x) / max(fog.y - fog.x, 1e-3), 0.0, 1.0));
  return o;
}
`;

export const GLSL = {
  sphere: [GL_HEAD + `
layout(location = 0) in vec2 q;
layout(location = 1) in vec3 c;
layout(location = 2) in float r;
layout(location = 3) in vec4 col;
out vec3 vp;
flat out vec3 vc;
flat out float rr;
flat out vec4 vcol;
flat out float rpx;
void main() {
  vec3 cv = (view * vec4(c, 1.0)).xyz;
  float s = r * 1.02;
  if (viewport.w < 0.5) {
    float d = length(cv);
    s = r * d / sqrt(max(d * d - r * r, 1e-6)) * d / max(-cv.z, 1e-6) * 1.02;
  }
  vp = vec3(cv.xy + q * s, cv.z);
  vc = cv; rr = r; vcol = col;
  float ppu = proj[1][1] * 0.5 * viewport.y;
  if (viewport.w < 0.5) ppu = ppu / max(-cv.z, 1e-6);
  rpx = r * ppu;
  gl_Position = proj * vec4(vp, 1.0);
}`, GL_HEAD + `
in vec3 vp;
flat in vec3 vc;
flat in float rr;
flat in vec4 vcol;
flat in float rpx;
out vec4 outColor;
void main() {
  vec3 ro = vec3(0.0);
  vec3 rd = normalize(vp);
  if (viewport.w > 0.5) { ro = vec3(vp.xy, 0.0); rd = vec3(0.0, 0.0, -1.0); }
  vec3 oc = ro - vc;
  float b = dot(rd, oc);
  float cc = dot(oc, oc) - rr * rr;
  float disc = b * b - cc;
  if (disc < 0.0) discard;
  float t = -b - sqrt(disc);
  vec3 hit = ro + t * rd;
  vec3 n = (hit - vc) / rr;
  vec4 clip = proj * vec4(hit, 1.0);
  gl_FragDepth = 0.5 * clip.z / clip.w + 0.5;
  vec3 base = vcol.rgb;
  if (vcol.a < 0.75 && n.z < accent.w) base = accent.rgb;
  float edge = rpx * (1.0 - sqrt(max(0.0, 1.0 - disc / (rr * rr))));
  outColor = vec4(post(shade(base, n, -rd), base, edge, -hit.z), 1.0);
}`],
  bond: [GL_HEAD + `
layout(location = 0) in vec3 g;
layout(location = 1) in vec3 p0;
layout(location = 2) in vec3 p1;
layout(location = 3) in vec4 c0;
layout(location = 4) in vec4 c1;
layout(location = 5) in float r;
out vec3 nn;
out float hh;
out vec3 vpos;
flat out vec4 fc0;
flat out vec4 fc1;
out float rpx;
flat out vec3 fwv;
void main() {
  vec3 ax = p1 - p0;
  vec3 w = ax / max(length(ax), 1e-6);
  fwv = (view * vec4(w, 0.0)).xyz;
  vec3 a = abs(w.x) > 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  vec3 e1 = normalize(cross(w, a));
  vec3 e2 = cross(w, e1);
  vec3 nw = e1 * g.x + e2 * g.y;
  vec4 v4 = view * vec4(p0 + nw * r + ax * g.z, 1.0);
  gl_Position = proj * v4;
  vpos = v4.xyz;
  nn = (view * vec4(nw, 0.0)).xyz;
  hh = g.z; fc0 = c0; fc1 = c1;
  float ppu = proj[1][1] * 0.5 * viewport.y;
  if (viewport.w < 0.5) ppu = ppu / max(-v4.z, 1e-6);
  rpx = r * ppu;
}`, GL_HEAD + `
in vec3 nn;
in float hh;
in vec3 vpos;
flat in vec4 fc0;
flat in vec4 fc1;
in float rpx;
flat in vec3 fwv;
out vec4 outColor;
void main() {
  vec3 base = hh > 0.5 ? fc1.rgb : fc0.rgb;
  vec3 v = viewport.w > 0.5 ? vec3(0.0, 0.0, 1.0) : normalize(-vpos);
  vec3 n = normalize(nn);
  float nv = min(abs(dot(n, v)) / max(length(cross(fwv, v)), 0.05), 1.0);
  outColor = vec4(post(shade(base, n, v), base, rpx * (1.0 - sqrt(max(0.0, 1.0 - nv * nv))), -vpos.z), 1.0);
}`],
  line: [GL_HEAD + `
layout(location = 0) in vec2 g;
layout(location = 1) in vec3 p0;
layout(location = 2) in vec3 p1;
layout(location = 3) in float dash;
out float s;
flat out float fd;
void main() {
  vec4 a = proj * view * vec4(p0, 1.0);
  vec4 b = proj * view * vec4(p1, 1.0);
  vec2 vpx = viewport.xy;
  vec2 sa = a.xy / a.w * 0.5 * vpx;
  vec2 sb = b.xy / b.w * 0.5 * vpx;
  vec2 d = sb - sa;
  float len = max(length(d), 1e-6);
  vec2 dir = d / len;
  vec2 nr = vec2(-dir.y, dir.x);
  vec4 c = mix(a, b, g.x);
  gl_Position = c + vec4(nr * g.y * line.w / vpx * c.w, 0.0, 0.0);
  s = g.x * len; fd = dash;
}`, GL_HEAD + `
in float s;
flat in float fd;
out vec4 outColor;
void main() {
  if (fd > 0.5 && fract(s / (7.0 * viewport.z)) > 0.55) discard;
  outColor = vec4(line.rgb, 1.0);
}`],
  mesh: [GL_HEAD + `
layout(location = 0) in vec3 p;
layout(location = 1) in vec4 n;
out vec3 nn;
out vec3 vpos;
invariant gl_Position;
void main() {
  vec4 v4 = view * vec4(p, 1.0);
  gl_Position = proj * v4;
  vpos = v4.xyz;
  nn = (view * vec4(n.xyz, 0.0)).xyz;
}`, GL_HEAD + `
uniform vec4 meshColor;
in vec3 nn;
in vec3 vpos;
out vec4 outColor;
void main() {
  vec3 n = normalize(nn);
  vec3 v = viewport.w > 0.5 ? vec3(0.0, 0.0, 1.0) : normalize(-vpos);
  if (dot(n, v) < 0.0) n = -n;
  vec3 c = shade(meshColor.rgb, n, v);
  if (look.y > 0.0) c = mix(meshColor.rgb * 0.18, c, smoothstep(0.1, 0.22, dot(n, v)));
  outColor = vec4(post(c, meshColor.rgb, 1e4, -vpos.z), meshColor.a);
}`],
};

/** Per-vertex geometry shared by both back ends. */
export const QUAD = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
export const LINEQ = new Float32Array([0, -1, 1, -1, 0, 1, 0, 1, 1, -1, 1, 1]);
export const CYL = (() => {
  const seg = 16, out = [];
  const corner = [[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 1]];
  for (let s = 0; s < seg; s++) for (const [e, h] of corner) { const t = 2 * Math.PI * (s + e) / seg; out.push(Math.cos(t), Math.sin(t), h); }
  return new Float32Array(out);
})();

/**
 * Atom labels: a camera-facing quad on the front of the atom (centre + 1.04 r toward the eye), height 1.15 r, width
 * height x aspect, depth-tested so nearer atoms hide it. The atlas holds the glyph coverage in red and a round outline
 * in green; colour = text * fill + halo * outline * (1 - fill), premultiplied, faded out below 9 px of label height.
 * Instance (44 B): centre f32x3, radius f32, atlas rect (u0, v0, u1, v1) f32x4, aspect f32, text rgba u8, halo rgba u8.
 */
export const WGSL_LABEL = /* wgsl */ `
struct U { view: mat4x4f, proj: mat4x4f, viewport: vec4f, light: vec4f, accent: vec4f, line: vec4f, look: vec4f, fog: vec4f, bg: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var atlas: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
struct LV {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) tc: vec4f,
  @location(2) @interpolate(flat) hc: vec4f,
  @location(3) @interpolate(flat) hpx: f32,
};
@vertex fn vs_label(@location(0) q: vec2f, @location(1) c: vec3f, @location(2) r: f32, @location(3) uvr: vec4f,
                    @location(4) asp: f32, @location(5) tc: vec4f, @location(6) hc: vec4f) -> LV {
  var o: LV;
  let vc = (u.view * vec4f(c, 1.0)).xyz;
  var toEye = vec3f(0.0, 0.0, 1.0);
  if (u.viewport.w < 0.5) { toEye = normalize(-vc); }
  let front = vc + toEye * r * 1.04;
  let h = 1.15 * r;
  let w = h * asp;
  let vp = front + vec3f(q.x * 0.5 * w, q.y * 0.5 * h, 0.0);
  o.pos = u.proj * vec4f(vp, 1.0);
  let t = q * 0.5 + vec2f(0.5, 0.5);
  o.uv = vec2f(mix(uvr.x, uvr.z, t.x), mix(uvr.w, uvr.y, t.y));
  o.tc = tc;
  o.hc = hc;
  var ppu = u.proj[1][1] * 0.5 * u.viewport.y;
  if (u.viewport.w < 0.5) { ppu = ppu / max(-front.z, 1e-6); }
  o.hpx = h * ppu;
  return o;
}
@fragment fn fs_label(i: LV) -> @location(0) vec4f {
  let s = textureSample(atlas, samp, i.uv);
  let fill = s.r;
  let halo = max(s.g, s.r) * i.hc.a;
  let fade = smoothstep(5.0, 9.0, i.hpx);
  let a = (fill + halo * (1.0 - fill)) * fade;
  let rgb = (i.tc.rgb * fill + i.hc.rgb * halo * (1.0 - fill)) * fade;
  return vec4f(rgb, a);
}
`;

GLSL.label = [GL_HEAD + `
layout(location = 0) in vec2 q;
layout(location = 1) in vec3 c;
layout(location = 2) in float r;
layout(location = 3) in vec4 uvr;
layout(location = 4) in float asp;
layout(location = 5) in vec4 tcol;
layout(location = 6) in vec4 hcol;
out vec2 uv;
flat out vec4 tc;
flat out vec4 hc;
flat out float hpx;
void main() {
  vec3 vc = (view * vec4(c, 1.0)).xyz;
  vec3 toEye = viewport.w < 0.5 ? normalize(-vc) : vec3(0.0, 0.0, 1.0);
  vec3 front = vc + toEye * r * 1.04;
  float h = 1.15 * r;
  float w = h * asp;
  gl_Position = proj * vec4(front + vec3(q.x * 0.5 * w, q.y * 0.5 * h, 0.0), 1.0);
  vec2 t = q * 0.5 + vec2(0.5);
  uv = vec2(mix(uvr.x, uvr.z, t.x), mix(uvr.w, uvr.y, t.y));
  tc = tcol; hc = hcol;
  float ppu = proj[1][1] * 0.5 * viewport.y;
  if (viewport.w < 0.5) ppu = ppu / max(-front.z, 1e-6);
  hpx = h * ppu;
}`, GL_HEAD + `
uniform sampler2D atlas;
in vec2 uv;
flat in vec4 tc;
flat in vec4 hc;
flat in float hpx;
out vec4 outColor;
void main() {
  vec4 s = texture(atlas, uv);
  float fill = s.r;
  float halo = max(s.g, s.r) * hc.a;
  float fade = smoothstep(5.0, 9.0, hpx);
  outColor = vec4((tc.rgb * fill + hc.rgb * halo * (1.0 - fill)) * fade, (fill + halo * (1.0 - fill)) * fade);
}`];
