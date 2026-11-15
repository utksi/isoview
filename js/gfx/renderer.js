/**
 * Back-end choice and the packing of scene data into the instance layouts both back ends read.
 * createRenderer tries WebGPU, then WebGL2 (?gl=webgl2 forces the fallback, ?gl=webgpu forbids it).
 * Packing (little-endian): atoms 20 B each = centre f32x3, radius f32, rgba u8 (a < 192 marks a selected atom);
 * bonds 36 B = p0, p1 f32x3, rgba u8 x2, radius f32; lines 28 B = p0, p1 f32x3, dashed f32 (0 or 1).
 * Uniform block (60 floats): view (16), proj (16), viewport (4), light (4), accent (4), line (4), look (4), fog (4), bg (4).
 */
import { WebGPURenderer } from "./webgpu.js";
import { WebGL2Renderer } from "./webgl2.js";

export async function createRenderer(canvas, onError, force = "") {
  if (force !== "webgl2") {
    try { const r = await WebGPURenderer.create(canvas, onError); if (r) return r; }
    catch (e) {
      if (force === "webgpu") throw e;
      console.warn("WebGPU failed, using WebGL2:", e);
      if (onError) onError(`WebGPU could not start (${e.message}); using WebGL2.`);
    }
    if (force === "webgpu") throw new Error("WebGPU is not available in this browser");
  }
  // a canvas that has a WebGPU context cannot give a WebGL2 one: use a fresh canvas
  if (canvas.dataset.ctx === "webgpu") { const c = canvas.cloneNode(); delete c.dataset.ctx; canvas.replaceWith(c); canvas = c; }
  return WebGL2Renderer.create(canvas, onError);
}

const rgb = hex => { const h = hex.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };

/** atoms: [{x, y, z, r, color '#rrggbb', sel}] -> ArrayBuffer(20 N). */
export function packAtoms(atoms) {
  const buf = new ArrayBuffer(20 * atoms.length), f = new Float32Array(buf), b = new Uint8Array(buf);
  atoms.forEach((a, i) => {
    f[5 * i] = a.x; f[5 * i + 1] = a.y; f[5 * i + 2] = a.z; f[5 * i + 3] = a.r;
    const c = rgb(a.color); b.set([c[0], c[1], c[2], a.sel ? 128 : 255], 20 * i + 16);
  });
  return buf;
}
/** bonds: [{p0, p1, c0, c1, r}] -> ArrayBuffer(36 N). */
export function packBonds(bonds) {
  const buf = new ArrayBuffer(36 * bonds.length), f = new Float32Array(buf), b = new Uint8Array(buf);
  bonds.forEach((d, i) => {
    f.set(d.p0, 9 * i); f.set(d.p1, 9 * i + 3); f[9 * i + 8] = d.r;
    const c0 = rgb(d.c0), c1 = rgb(d.c1);
    b.set([c0[0], c0[1], c0[2], 255, c1[0], c1[1], c1[2], 255], 36 * i + 24);
  });
  return buf;
}
/** segments: [{p0, p1, dashed}] -> Float32Array(7 N). */
export function packLines(segs) {
  const f = new Float32Array(7 * segs.length);
  segs.forEach((s, i) => { f.set(s.p0, 7 * i); f.set(s.p1, 7 * i + 3); f[7 * i + 6] = s.dashed ? 1 : 0; });
  return f;
}
export function packUniforms(view, proj, viewport, light, accent, line, look = [0, 0, 0, 0], fog = [0, 1, 0, 0], bg = [1, 1, 1, 1]) {
  const u = new Float32Array(60);
  u.set(view, 0); u.set(proj, 16); u.set(viewport, 32); u.set(light, 36); u.set(accent, 40); u.set(line, 44);
  u.set(look, 48); u.set(fog, 52); u.set(bg, 56);
  return u;
}
export const hexToRgb01 = hex => rgb(hex).map(v => v / 255);
