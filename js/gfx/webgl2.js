/**
 * WebGL2 back end with the same interface and draw order as webgpu.js (opaque spheres, bonds, lines, opaque meshes;
 * translucent meshes as a depth-only pass then a blended pass with LEQUAL and no depth writes). Multisampling comes
 * from the default framebuffer (antialias: true). Instance attributes use vertexAttribDivisor(1); the per-vertex quad,
 * cylinder and line geometry are ordinary attributes. Uniforms: one std140 uniform block "U" at binding 0.
 * Labels: drawn after the opaque meshes with premultiplied blending and no depth writes; the atlas texture is
 * uploaded from the canvas with mipmaps (LINEAR_MIPMAP_LINEAR).
 */
import { GLSL, QUAD, LINEQ, CYL } from "./shaders.js?v=20";

export class WebGL2Renderer {
  static create(canvas, onError) {
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, depth: true, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    if (!gl) return null;
    const r = new WebGL2Renderer();
    r.setup(canvas, gl, onError);
    return r;
  }

  setup(canvas, gl, onError) {
    this.name = "WebGL2"; this.canvas = canvas; this.gl = gl; this.onError = onError;
    const prog = (name, [vs, fs]) => {
      const sh = (type, src) => {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`${name} shader: ${gl.getShaderInfoLog(s)}`);
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`${name} program: ${gl.getProgramInfoLog(p)}`);
      gl.uniformBlockBinding(p, gl.getUniformBlockIndex(p, "U"), 0);
      return p;
    };
    this.pSphere = prog("sphere", GLSL.sphere); this.pBond = prog("bond", GLSL.bond);
    this.pLine = prog("line", GLSL.line); this.pMesh = prog("mesh", GLSL.mesh);
    this.uMeshColor = gl.getUniformLocation(this.pMesh, "meshColor");
    this.pLabel = prog("label", GLSL.label);
    this.uAtlas = gl.getUniformLocation(this.pLabel, "atlas");
    this.ubo = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo); gl.bufferData(gl.UNIFORM_BUFFER, 240, gl.DYNAMIC_DRAW);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.ubo);
    const stat = a => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW); return b; };
    this.quad = stat(QUAD); this.lineq = stat(LINEQ); this.cyl = stat(CYL);
    // attribute spec: [location, size, type, normalized, stride, offset, divisor]
    const F = gl.FLOAT, UB = gl.UNSIGNED_BYTE;
    this.atoms = this.slot(this.quad, [[0, 2, F, false, 8, 0, 0]], [[1, 3, F, false, 20, 0, 1], [2, 1, F, false, 20, 12, 1], [3, 4, UB, true, 20, 16, 1]]);
    this.bonds = this.slot(this.cyl, [[0, 3, F, false, 12, 0, 0]], [[1, 3, F, false, 36, 0, 1], [2, 3, F, false, 36, 12, 1], [3, 4, UB, true, 36, 24, 1], [4, 4, UB, true, 36, 28, 1], [5, 1, F, false, 36, 32, 1]]);
    this.lines = this.slot(this.lineq, [[0, 2, F, false, 8, 0, 0]], [[1, 3, F, false, 28, 0, 1], [2, 3, F, false, 28, 12, 1], [3, 1, F, false, 28, 24, 1]]);
    this.labels = this.slot(this.quad, [[0, 2, F, false, 8, 0, 0]], [[1, 3, F, false, 44, 0, 1], [2, 1, F, false, 44, 12, 1], [3, 4, F, false, 44, 16, 1],
      [4, 1, F, false, 44, 32, 1], [5, 4, UB, true, 44, 36, 1], [6, 4, UB, true, 44, 40, 1]]);
    this.meshes = []; this.groups = new Map();
    this.bg = [1, 1, 1];
  }

  /** A VAO with a static per-vertex buffer and a dynamic per-instance buffer. */
  slot(staticBuf, vAttrs, iAttrs) {
    const gl = this.gl, vao = gl.createVertexArray(), inst = gl.createBuffer();
    gl.bindVertexArray(vao);
    const bind = (buf, list) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      for (const [loc, size, type, norm, stride, off, div] of list) {
        gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type, norm, stride, off); gl.vertexAttribDivisor(loc, div);
      }
    };
    bind(staticBuf, vAttrs); bind(inst, iAttrs);
    gl.bindVertexArray(null);
    return { vao, inst, n: 0 };
  }
  upload(slot, data, n) { const gl = this.gl; gl.bindBuffer(gl.ARRAY_BUFFER, slot.inst); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); slot.n = n; }
  setAtoms(buf, n) { this.upload(this.atoms, buf, n); }
  setBonds(buf, n) { this.upload(this.bonds, buf, n); }
  setLines(buf, n) { this.upload(this.lines, buf, n); }
  setLabels(buf, n, atlas) {
    this.upload(this.labels, buf, n);
    if (!atlas || atlas.id === this.atlasId) return;
    const gl = this.gl;
    if (!this.atlasTex) this.atlasTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, atlas.canvas);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.atlasId = atlas.id;
  }
  /** Mesh group `key` (one per field layer): list = [{ positions, normals, indices, color: [r, g, b], opacity }]. */
  setMeshGroup(key, list) {
    const gl = this.gl;
    this.removeMeshGroup(key, true);
    this.groups.set(key, list.filter(m => m.indices.length).map(m => {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const pos = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pos); gl.bufferData(gl.ARRAY_BUFFER, m.positions, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
      const nrm = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, nrm); gl.bufferData(gl.ARRAY_BUFFER, m.normals, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.SHORT, true, 8, 0);
      const idx = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.indices, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      return { vao, pos, nrm, idx, n: m.indices.length, color: m.color, opacity: m.opacity };
    }));
    this.meshes = [...this.groups.values()].flat();
  }
  removeMeshGroup(key, keepFlat = false) {
    const gl = this.gl, g = this.groups.get(key);
    if (g) { for (const m of g) { gl.deleteVertexArray(m.vao); gl.deleteBuffer(m.pos); gl.deleteBuffer(m.nrm); gl.deleteBuffer(m.idx); } this.groups.delete(key); }
    if (!keepFlat) this.meshes = [...this.groups.values()].flat();
  }
  setMeshGroupStyle(key, list) {
    const g = this.groups.get(key);
    if (g) g.forEach((m, i) => { if (list[i]) { m.color = list[i].color; m.opacity = list[i].opacity; } });
  }
  clearMeshes() { for (const k of [...this.groups.keys()]) this.removeMeshGroup(k); }
  setBackground(rgb) { this.bg = rgb; }

  draw(uniforms) {
    const gl = this.gl, w = this.canvas.width, h = this.canvas.height;
    if (!w || !h) return;
    gl.viewport(0, 0, w, h);
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo); gl.bufferSubData(gl.UNIFORM_BUFFER, 0, uniforms);
    gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1); gl.clearDepth(1);
    gl.depthMask(true); gl.colorMask(true, true, true, true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE);
    const inst = (p, s, count) => { if (!s.n) return; gl.useProgram(p); gl.bindVertexArray(s.vao); gl.drawArraysInstanced(gl.TRIANGLES, 0, count, s.n); };
    inst(this.pSphere, this.atoms, 6);
    inst(this.pBond, this.bonds, CYL.length / 3);
    inst(this.pLine, this.lines, 6);
    gl.useProgram(this.pMesh);
    const mesh = m => { gl.uniform4f(this.uMeshColor, m.color[0], m.color[1], m.color[2], m.opacity); gl.bindVertexArray(m.vao); gl.drawElements(gl.TRIANGLES, m.n, gl.UNSIGNED_INT, 0); };
    for (const m of this.meshes) if (m.opacity >= 0.999) mesh(m);
    if (this.labels.n && this.atlasTex) {
      gl.useProgram(this.pLabel); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.atlasTex); gl.uniform1i(this.uAtlas, 0);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
      gl.bindVertexArray(this.labels.vao); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.labels.n);
      gl.depthMask(true); gl.disable(gl.BLEND); gl.useProgram(this.pMesh);
    }
    for (const m of this.meshes) if (m.opacity < 0.999) {
      gl.colorMask(false, false, false, false); gl.depthMask(true); gl.depthFunc(gl.LESS); gl.disable(gl.BLEND);
      mesh(m);
      gl.colorMask(true, true, true, true); gl.depthMask(false); gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      mesh(m);
    }
    gl.depthMask(true); gl.depthFunc(gl.LESS); gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
    const err = gl.getError();
    if (err && this.onError && !this.errShown) { this.errShown = true; this.onError("WebGL2 error 0x" + err.toString(16)); }
  }
}
