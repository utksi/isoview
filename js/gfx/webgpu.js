// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * WebGPU back end: one render pass per frame, 4x MSAA resolved into the canvas.
 * Order: opaque spheres, bonds, lines, opaque meshes; then each translucent mesh twice, a depth-only pass (colour
 * writes off) and a blended pass with depth test <= and no depth writes, so only its front layer is blended over the
 * opaque scene (no sorting needed). Instance layouts (bytes): sphere 20 = centre f32x3, radius f32, colour unorm8x4;
 * bond 36 = p0, p1 f32x3, colours unorm8x4 x2, radius f32; line 28 = p0, p1 f32x3, dashed f32.
 * Mesh: positions f32x3, normals snorm16x4, uint32 indices. Uniforms: 240-byte block (see shaders.js).
 * Labels (after the opaque meshes, before translucent ones): 44-byte instances (gfx/labels.js), premultiplied blending,
 * depth test without depth writes; the atlas is an rgba8 texture with a full mip chain made by repeated halving.
 * Slice plane (setSlice, 40-byte instance): after the labels, before the translucent meshes; depth test and depth
 * writes, premultiplied blending; its image is a premultiplied rgba8 texture with a mip chain made the same way.
 */
import { WGSL, WGSL_LABEL, WGSL_SLICE, QUAD, LINEQ, CYL } from "./shaders.js?v=1002";

const GPUB = () => globalThis.GPUBufferUsage, GPUS = () => globalThis.GPUShaderStage;

export class WebGPURenderer {
  static async create(canvas, onError) {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const r = new WebGPURenderer();
    // WHY: any shader or pipeline validation error during setup makes the caller fall back to WebGL2
    device.pushErrorScope("validation");
    r.setup(canvas, device, onError);
    const err = await device.popErrorScope();
    if (err) { device.destroy(); throw new Error(err.message); }
    return r;
  }

  setup(canvas, device, onError) {
    this.name = "WebGPU"; this.canvas = canvas; this.device = device; this.sample = 4;
    this.maxSide = device.limits.maxTextureDimension2D; // largest canvas side (device px)
    device.addEventListener("uncapturederror", e => onError && onError("WebGPU: " + e.error.message));
    device.lost.then(info => { if (info.reason !== "destroyed" && onError) onError("WebGPU device lost: " + info.message); });
    canvas.dataset.ctx = "webgpu";
    this.ctx = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: "opaque" });
    const U = GPUB(), S = GPUS();
    const mod = device.createShaderModule({ code: WGSL });
    if (mod.getCompilationInfo) mod.getCompilationInfo().then(ci => { const e = ci.messages.filter(m => m.type === "error"); if (e.length && onError) onError("WGSL: " + e.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")); });
    this.ubuf = device.createBuffer({ size: 240, usage: U.UNIFORM | U.COPY_DST });
    const ent = [{ binding: 0, visibility: S.VERTEX | S.FRAGMENT, buffer: { type: "uniform" } }];
    this.bgl0 = device.createBindGroupLayout({ entries: ent });
    this.bgl1 = device.createBindGroupLayout({ entries: ent });
    this.bg0 = device.createBindGroup({ layout: this.bgl0, entries: [{ binding: 0, resource: { buffer: this.ubuf } }] });
    const L0 = device.createPipelineLayout({ bindGroupLayouts: [this.bgl0] });
    const L01 = device.createPipelineLayout({ bindGroupLayouts: [this.bgl0, this.bgl1] });
    const depth = (write, cmp) => ({ format: "depth24plus", depthWriteEnabled: write, depthCompare: cmp });
    const multisample = { count: this.sample };
    const vb = (stride, step, attrs) => ({ arrayStride: stride, stepMode: step, attributes: attrs.map(([loc, off, fmt]) => ({ shaderLocation: loc, offset: off, format: fmt })) });
    const pipe = (layout, vs, fs, buffers, ds, target, cull = "none") => device.createRenderPipeline({
      layout, vertex: { module: mod, entryPoint: vs, buffers }, fragment: { module: mod, entryPoint: fs, targets: [target] },
      primitive: { topology: "triangle-list", cullMode: cull }, depthStencil: ds, multisample });
    const opaque = { format: this.format };
    const blend = { format: this.format, blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" } } };
    const sphereB = [vb(8, "vertex", [[0, 0, "float32x2"]]), vb(20, "instance", [[1, 0, "float32x3"], [2, 12, "float32"], [3, 16, "unorm8x4"]])];
    const bondB = [vb(12, "vertex", [[0, 0, "float32x3"]]), vb(36, "instance", [[1, 0, "float32x3"], [2, 12, "float32x3"], [3, 24, "unorm8x4"], [4, 28, "unorm8x4"], [5, 32, "float32"]])];
    const lineB = [vb(8, "vertex", [[0, 0, "float32x2"]]), vb(28, "instance", [[1, 0, "float32x3"], [2, 12, "float32x3"], [3, 24, "float32"]])];
    const meshB = [vb(12, "vertex", [[0, 0, "float32x3"]]), vb(8, "vertex", [[1, 0, "snorm16x4"]])];
    this.pSphere = pipe(L0, "vs_sphere", "fs_sphere", sphereB, depth(true, "less"), opaque);
    this.pBond = pipe(L0, "vs_bond", "fs_bond", bondB, depth(true, "less"), opaque);
    this.pLine = pipe(L0, "vs_line", "fs_line", lineB, depth(true, "less"), opaque);
    this.pMesh = pipe(L01, "vs_mesh", "fs_mesh", meshB, depth(true, "less"), opaque);
    this.pDepth = pipe(L0, "vs_mesh", "fs_depth", meshB, depth(true, "less"), { format: this.format, writeMask: 0 });
    this.pBlend = pipe(L01, "vs_mesh", "fs_mesh", meshB, depth(false, "less-equal"), blend);
    const lmod = device.createShaderModule({ code: WGSL_LABEL });
    this.bglTex = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: S.FRAGMENT, texture: { sampleType: "float" } }, { binding: 1, visibility: S.FRAGMENT, sampler: { type: "filtering" } }] });
    const pre = { color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" } };
    this.pLabel = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bgl0, this.bglTex] }),
      vertex: { module: lmod, entryPoint: "vs_label", buffers: [vb(8, "vertex", [[0, 0, "float32x2"]]),
        vb(44, "instance", [[1, 0, "float32x3"], [2, 12, "float32"], [3, 16, "float32x4"], [4, 32, "float32"], [5, 36, "unorm8x4"], [6, 40, "unorm8x4"]])] },
      fragment: { module: lmod, entryPoint: "fs_label", targets: [{ format: this.format, blend: pre }] },
      primitive: { topology: "triangle-list" }, depthStencil: depth(false, "less"), multisample });
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" });
    const smod = device.createShaderModule({ code: WGSL_SLICE });
    this.pSlice = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bgl0, this.bglTex] }),
      vertex: { module: smod, entryPoint: "vs_slice", buffers: [vb(8, "vertex", [[0, 0, "float32x2"]]),
        vb(40, "instance", [[1, 0, "float32x3"], [2, 12, "float32x3"], [3, 24, "float32x3"], [4, 36, "float32"]])] },
      fragment: { module: smod, entryPoint: "fs_slice", targets: [{ format: this.format, blend: pre }] },
      primitive: { topology: "triangle-list" }, depthStencil: depth(true, "less"), multisample });
    this.slice = { buf: null, n: 0 };
    this.labels = { buf: null, n: 0 };
    const stat = a => { const b = device.createBuffer({ size: a.byteLength, usage: U.VERTEX | U.COPY_DST }); device.queue.writeBuffer(b, 0, a); return b; };
    this.quad = stat(QUAD); this.lineq = stat(LINEQ); this.cyl = stat(CYL);
    this.atoms = { buf: null, n: 0 }; this.bonds = { buf: null, n: 0 }; this.lines = { buf: null, n: 0 }; this.meshes = []; this.groups = new Map();
    this.bg = { r: 1, g: 1, b: 1, a: 1 };
    this.size = [0, 0];
  }

  upload(slot, data, n) {
    const d = this.device;
    if (slot.buf && slot.buf.size < data.byteLength) { slot.buf.destroy(); slot.buf = null; }
    if (!slot.buf && data.byteLength) slot.buf = d.createBuffer({ size: Math.max(256, data.byteLength), usage: GPUB().VERTEX | GPUB().COPY_DST });
    if (data.byteLength) d.queue.writeBuffer(slot.buf, 0, data);
    slot.n = n;
  }
  setAtoms(buf, n) { this.upload(this.atoms, buf, n); }
  setBonds(buf, n) { this.upload(this.bonds, buf, n); }
  setLines(buf, n) { this.upload(this.lines, buf, n); }
  /** buf: label instances; atlas: { canvas, id } from gfx/labels.js (uploaded when its id changes). */
  setLabels(buf, n, atlas) {
    this.upload(this.labels, buf, n);
    if (!atlas || atlas.id === this.atlasId) return;
    const d = this.device;
    if (this.atlasTex) this.atlasTex.destroy();
    this.atlasTex = this.mipTexture(atlas.canvas, false);
    this.bgTex = d.createBindGroup({ layout: this.bglTex, entries: [{ binding: 0, resource: this.atlasTex.createView() }, { binding: 1, resource: this.sampler }] });
    this.atlasId = atlas.id;
  }
  /** An rgba8 texture of a canvas with a full mip chain made by repeated halving (premultiplied when pre). */
  mipTexture(canvas, pre) {
    const d = this.device, T = globalThis.GPUTextureUsage, w = canvas.width, h = canvas.height;
    const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;
    const tex = d.createTexture({ size: [w, h], format: "rgba8unorm", mipLevelCount: levels, usage: T.TEXTURE_BINDING | T.COPY_DST | T.RENDER_ATTACHMENT });
    let src = canvas;
    for (let l = 0; l < levels; l++) {
      const lw = Math.max(1, w >> l), lh = Math.max(1, h >> l);
      if (l > 0) { const c = document.createElement("canvas"); c.width = lw; c.height = lh; const x = c.getContext("2d"); x.imageSmoothingQuality = "high"; x.drawImage(src, 0, 0, lw, lh); src = c; }
      d.queue.copyExternalImageToTexture({ source: src }, { texture: tex, mipLevel: l, premultipliedAlpha: pre }, [lw, lh]);
    }
    return tex;
  }
  /** Slice plane: { o, U, V (A), alpha, image: a canvas (new image) or null (keep the last one) }, or null to hide it. */
  setSlice(sl) {
    if (!sl) { this.slice.n = 0; return; }
    this.upload(this.slice, new Float32Array([...sl.o, ...sl.U, ...sl.V, sl.alpha]), 1);
    if (!sl.image) return;
    if (this.sliceTex) this.sliceTex.destroy();
    this.sliceTex = this.mipTexture(sl.image, true);
    this.bgSlice = this.device.createBindGroup({ layout: this.bglTex, entries: [{ binding: 0, resource: this.sliceTex.createView() }, { binding: 1, resource: this.sampler }] });
  }
  /** Mesh group `key` (one per field layer): list = [{ positions, normals, indices, color: [r, g, b], opacity }]. */
  setMeshGroup(key, list) {
    this.removeMeshGroup(key, true);
    const d = this.device, U = GPUB();
    this.groups.set(key, list.filter(m => m.indices.length).map(m => {
      const mk = (a, use) => { const b = d.createBuffer({ size: Math.max(16, Math.ceil(a.byteLength / 4) * 4), usage: use | U.COPY_DST }); d.queue.writeBuffer(b, 0, a); return b; };
      const ub = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
      d.queue.writeBuffer(ub, 0, new Float32Array([...m.color, m.opacity]));
      return { pos: mk(m.positions, U.VERTEX), nrm: mk(m.normals, U.VERTEX), idx: mk(m.indices, U.INDEX), n: m.indices.length, ub,
        bg: d.createBindGroup({ layout: this.bgl1, entries: [{ binding: 0, resource: { buffer: ub } }] }), opacity: m.opacity };
    }));
    this.meshes = [...this.groups.values()].flat();
  }
  removeMeshGroup(key, keepFlat = false) {
    const g = this.groups.get(key);
    if (g) { for (const m of g) { m.pos.destroy(); m.nrm.destroy(); m.idx.destroy(); m.ub.destroy(); } this.groups.delete(key); }
    if (!keepFlat) this.meshes = [...this.groups.values()].flat();
  }
  setMeshGroupStyle(key, list) {
    const g = this.groups.get(key);
    if (g) g.forEach((m, i) => { if (list[i]) { m.opacity = list[i].opacity; this.device.queue.writeBuffer(m.ub, 0, new Float32Array([...list[i].color, list[i].opacity])); } });
  }
  clearMeshes() { for (const k of [...this.groups.keys()]) this.removeMeshGroup(k); }
  setBackground(rgb) { this.bg = { r: rgb[0], g: rgb[1], b: rgb[2], a: 1 }; }

  ensureTargets() {
    const w = this.canvas.width, h = this.canvas.height;
    if (w === this.size[0] && h === this.size[1] && this.msaa) return;
    if (this.msaa) { this.msaa.destroy(); this.depthTex.destroy(); }
    const U = globalThis.GPUTextureUsage;
    this.msaa = this.device.createTexture({ size: [w, h], sampleCount: this.sample, format: this.format, usage: U.RENDER_ATTACHMENT });
    this.depthTex = this.device.createTexture({ size: [w, h], sampleCount: this.sample, format: "depth24plus", usage: U.RENDER_ATTACHMENT });
    this.size = [w, h];
  }

  /** uniforms: Float32Array(60) laid out as block U. */
  draw(uniforms) {
    const d = this.device;
    if (!this.canvas.width || !this.canvas.height) return;
    // the first frames run inside an error scope; a validation error there is reported through onFatal
    const check = (this.frames = (this.frames || 0) + 1) <= 3;
    if (check) d.pushErrorScope("validation");
    this.ensureTargets();
    d.queue.writeBuffer(this.ubuf, 0, uniforms);
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.msaa.createView(), resolveTarget: this.ctx.getCurrentTexture().createView(), clearValue: this.bg, loadOp: "clear", storeOp: "discard" }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" },
    });
    pass.setBindGroup(0, this.bg0);
    if (this.atoms.n) { pass.setPipeline(this.pSphere); pass.setVertexBuffer(0, this.quad); pass.setVertexBuffer(1, this.atoms.buf); pass.draw(6, this.atoms.n); }
    if (this.bonds.n) { pass.setPipeline(this.pBond); pass.setVertexBuffer(0, this.cyl); pass.setVertexBuffer(1, this.bonds.buf); pass.draw(CYL.length / 3, this.bonds.n); }
    if (this.lines.n) { pass.setPipeline(this.pLine); pass.setVertexBuffer(0, this.lineq); pass.setVertexBuffer(1, this.lines.buf); pass.draw(6, this.lines.n); }
    const mesh = (m, p) => { pass.setPipeline(p); pass.setBindGroup(1, m.bg); pass.setVertexBuffer(0, m.pos); pass.setVertexBuffer(1, m.nrm); pass.setIndexBuffer(m.idx, "uint32"); pass.drawIndexed(m.n); };
    for (const m of this.meshes) if (m.opacity >= 0.999) mesh(m, this.pMesh);
    if (this.labels.n && this.bgTex) { pass.setPipeline(this.pLabel); pass.setBindGroup(1, this.bgTex); pass.setVertexBuffer(0, this.quad); pass.setVertexBuffer(1, this.labels.buf); pass.draw(6, this.labels.n); }
    if (this.slice.n && this.bgSlice) { pass.setPipeline(this.pSlice); pass.setBindGroup(1, this.bgSlice); pass.setVertexBuffer(0, this.quad); pass.setVertexBuffer(1, this.slice.buf); pass.draw(6, 1); }
    for (const m of this.meshes) if (m.opacity < 0.999) {
      pass.setPipeline(this.pDepth); pass.setVertexBuffer(0, m.pos); pass.setVertexBuffer(1, m.nrm); pass.setIndexBuffer(m.idx, "uint32"); pass.drawIndexed(m.n);
      mesh(m, this.pBlend);
    }
    pass.end();
    d.queue.submit([enc.finish()]);
    if (check) d.popErrorScope().then(e => { if (e && this.onFatal) this.onFatal(e.message); });
  }
}
