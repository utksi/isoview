/**
 * isoview application: layers, UI state, camera control, picking, measurement, editing, and the hand-off to the
 * renderer (gfx/renderer.js: WebGPU or WebGL2) and the compute worker (worker.js: parsing, WAVECAR orbitals, meshing).
 *
 * Scene: every loaded file group is an item (worker dataset ds, structure, fields, WAVECAR header); items give layers,
 * an atoms layer (structure) and a field layer (one shown field, its own level, opacity and colours). Layers can be
 * added to a scene ("Add to scene"), hidden and removed.
 * Origin shift: one Cartesian translation t for every layer, entered and shown as fractions (fx, fy, fz) of the main
 * cell; each layer applies t in its own cell (structure.js for atoms, field.js for fields, exact for periodic grids),
 * so atoms, densities and derived cells (primitive, conventional: symmetry.js) move together.
 * Selection: click atoms (toggle) or draw a region with Shift + drag (or the region tool); 2-4 atoms give distances,
 * angles at the middle atoms and the dihedral (measure.js). A region also records a cut (the view frustum through the
 * drawn polygon, field.js) that "Cut surface" removes from the visible surfaces. Deleting removes the atom from its structure (all
 * periodic images). Deletions, cuts, shifts and layer removals can be undone.
 * Level slider: log10 |iso| over six decades below max |f| (densities, signed fields; signed fields get +iso and -iso
 * surfaces), linear over [min, max] (potentials, ELF); the readout is the share of sum |f| inside the surface.
 * Frames are drawn on demand. Label mode (L) puts element symbols on clicked atoms (gfx/labels.js).
 * Trajectories (traj.js): an item with traj keeps every frame; its structure holds the current frame S.frame (clamped to
 * its own frame count) for the atoms it still has (structure.ids), so deletions apply to all frames. Playback advances
 * S.step frames every 1 / S.fps seconds (loop, bounce or stop at the end). Movies render each chosen frame into a
 * canvas: GIF (gif.js) with delays of 100 / fps centiseconds, the remainder carried; video via MediaRecorder, frames
 * pushed at 1 / fps second intervals.
 * Looks (LOOKS, shaders.js): lighting mode, outline width and depth fog sent as uniforms; fog runs over the view depths
 * of the scene's bounding sphere, cam.dist -+ cam.radius.
 * The scene is kept in this browser (store.js): each item's files once, and a description (layers, edits, view,
 * camera, frame) written 0.7 s after the last change, so a reload restores it; New scene clears it (with Undo).
 */
import { params, formula, det3, norm, cross, dot, inv3, cartToFrac, fracToCart } from "./lattice.js?v=11";
import { fractionAbove, isoForFraction, autoStride, insidePolygon } from "./field.js?v=11";
import { displayAtoms, bondList, atomId, deleteAtoms, toPOSCAR, shownStructure, toCIF, toXYZ } from "./structure.js?v=11";
import { primitive, conventional, sameCellShape } from "./symmetry.js?v=11";
import { colorOf, rcov, SPACERS } from "./elements.js?v=11";
import { Camera, qfromRows, qmat, pickSphere } from "./gfx/camera.js?v=11";
import { createRenderer, packAtoms, packBonds, packLines, packUniforms, hexToRgb01 } from "./gfx/renderer.js?v=11";
import { buildAtlas, packLabels } from "./gfx/labels.js?v=11";
import { measureAll } from "./measure.js?v=11";
import { GifWriter, buildPalette } from "./gif.js?v=11";
import * as store from "./store.js?v=11";

const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const qp = new URLSearchParams(location.search);
// colour pairs (positive, negative) for successive field layers: mid-tones readable on light and dark backgrounds
const FIELD_COLORS = [["#e3a03a", "#3b86d6"], ["#2bb3a3", "#e0607e"], ["#9b7be0", "#7cb342"], ["#ef7d4b", "#5c9bd6"]];

const S = {
  items: new Map(), layers: [], active: null, activeAtoms: null, t: [0, 0, 0],
  style: "ball", scale: 1, bonds: true, tol: 1.15, cell: true, reps: [1, 1, 1], boundary: true, res: "auto",
  proj: "persp", theme: "auto", sel: [], region: null, labels: [], labelMode: false, boxMode: false, undo: [], atlas: null,
  frame: 0, playing: false, fps: 12, step: 1, loop: "loop", movie: { fmt: "gif", scale: 0.5 }, exporting: null, look: "classic",
};
// render looks (shaders.js): [mode, outline width in CSS px, depth-fog strength]
const LOOKS = { classic: [0, 0, 0], soft: [1, 0, 0.5], glossy: [2, 0, 0], toon: [3, 1.5, 0], outline: [4, 1.3, 0] };
let nextItem = 1, nextLayer = 1;
const cam = new Camera();
let R = null, canvas = $("gl"), pal = {}, disp = { atoms: [], centers: new Float32Array(0), radii: new Float32Array(0) };

/* ------------------------------------------------------------------ helpers */
const unit = v => { const n = norm(v); return v.map(x => x / n); };
const fmt = x => {
  if (!Number.isFinite(x)) return "–";
  const a = Math.abs(x);
  return a !== 0 && (a < 1e-2 || a >= 1e4) ? x.toExponential(2) : +x.toPrecision(3) + "";
};
const sup = s => s.replace(/e([+-]?)(\d+)/, (m, sg, d) => "×10" + (sg === "-" ? "⁻" : "") + String(+d).replace(/\d/g, c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let toastTimer = 0;
function toast(msg, err = false, ms = 5200, action = null) {
  const t = $("toast");
  t.textContent = msg;
  if (action) { const b = document.createElement("button"); b.type = "button"; b.textContent = action.label; b.addEventListener("click", () => { t.hidden = true; action.fn(); }); t.append(" ", b); }
  t.classList.toggle("err", err); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), ms);
}
const busy = on => ($("busy").hidden = !on);
const sameCell = (A, B) => A && B && A.every((r, i) => r.every((x, j) => Math.abs(x - B[i][j]) < 2e-3));
function readPalette() {
  pal = { bg: hexToRgb01(css("--bg")), cell: hexToRgb01(css("--cell")), accent: hexToRgb01(css("--accent")), ax: ["--ax-a", "--ax-b", "--ax-c"].map(css) };
}
const layerById = id => S.layers.find(l => l.id === id);
const itemOf = L => S.items.get(L.item);
const atomLayers = (vis = true) => S.layers.filter(l => l.kind === "atoms" && (!vis || l.visible));
const fieldLayers = (vis = false) => S.layers.filter(l => l.kind === "field" && (!vis || l.visible));
const activeField = () => S.layers.find(l => l.id === S.active && l.kind === "field") || null;
const cellOf = it => (it.structure && it.structure.lattice) || (it.wave && it.wave.lattice) || null;
const pbcOf = it => (it.structure ? it.structure.pbc : !!it.wave);
/** Corner of an item's cell: 0 for crystals, the grid origin for a non-periodic box (cube files, molecules in a box). */
const originOf = it => (it.structure && it.structure.origin) || [0, 0, 0];
/** Cell of the first visible layer that has one (else of any layer): reference for views and for the shift fractions. */
const primaryCell = () => {
  for (const vis of [true, false]) for (const L of S.layers) {
    if (vis && !L.visible) continue;
    const it = itemOf(L), c = cellOf(it);
    if (c) return { L: c, pbc: pbcOf(it), o: originOf(it) };
  }
  return null;
};
/** The origin shift as fractions of the cell L (the shift itself is the Cartesian translation S.t). */
// WHY: rounded to 1e-12 so that a typed 0.5 comes back as 0.5 after the Cartesian round trip (exact field rolls)
const fracShift = L => (L ? cartToFrac(inv3(L), S.t).map(x => Math.round(x * 1e12) / 1e12) : [0, 0, 0]);
const activeAtomsLayer = () => S.layers.find(l => l.id === S.activeAtoms && l.kind === "atoms") || atomLayers()[0] || atomLayers(false)[0] || null;

/* ------------------------------------------------------------------ worker */
let worker = null, reqId = 0;
const pending = new Map();
function startWorker() {
  worker = globalThis.__isoviewWorker ? new globalThis.__isoviewWorker() : new Worker(new URL("./worker.js?v=11", import.meta.url), { type: "module" });
  worker.onmessage = e => { const p = pending.get(e.data.id); if (!p) return; pending.delete(e.data.id); if (e.data.ok) p.res(e.data.result); else p.rej(new Error(e.data.error)); };
  worker.onerror = e => toast("The background worker failed: " + (e.message || "could not start"), true, 9000);
}
const call = (op, payload = {}) => new Promise((res, rej) => { const id = ++reqId; pending.set(id, { res, rej }); worker.postMessage({ id, op, ...payload }); });

/* ------------------------------------------------------------------ renderer and frames */
async function ensureRenderer() {
  if (R) return true;
  try { R = globalThis.__isoviewRenderer ? await globalThis.__isoviewRenderer(canvas) : await createRenderer(canvas, m => toast(m, true, 12000), qp.get("gl") || ""); }
  catch (e) { R = null; toast(e.message || String(e), true, 12000); }
  if (!R) { toast("This browser has neither WebGPU nor WebGL2, so the 3D view cannot start.", true, 12000); return false; }
  canvas = R.canvas || canvas;
  R.onFatal = msg => fallbackToWebGL2(msg);
  $("gfx").textContent = R.name;
  readPalette(); R.setBackground(pal.bg);
  resize();
  return true;
}
/** WebGPU reported an error in its first frames: rebuild everything on WebGL2. */
async function fallbackToWebGL2(msg) {
  if (!R || R.name !== "WebGPU" || qp.get("gl") === "webgpu") return;
  R = null;
  try { R = await createRenderer(canvas, m => toast(m, true, 12000), "webgl2"); } catch (e) { R = null; }
  if (!R) { toast(`WebGPU failed (${msg}) and WebGL2 is not available.`, true, 12000); return; }
  canvas = R.canvas; $("gfx").textContent = R.name + " (WebGPU failed)";
  readPalette(); R.setBackground(pal.bg); resize();
  S.atlas = buildAtlas(atlasSpecies());
  buildStructure(); refreshSurfaces();
  toast(`WebGPU reported an error (${msg}); switched to WebGL2.`, true, 9000);
}
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = window.innerWidth, h = window.innerHeight;
  canvas.width = Math.max(1, Math.round(w * dpr)); canvas.height = Math.max(1, Math.round(h * dpr));
  cam.w = w; cam.h = h;
  requestRender();
}
let dirty = false;
function requestRender() { if (!dirty) { dirty = true; requestAnimationFrame(frame); } saveSoon(); }
function frame() {
  dirty = false;
  if (!R) return;
  const dpr = canvas.width / cam.w;
  const lk = LOOKS[S.look] || LOOKS.classic;
  R.draw(packUniforms(cam.view(), cam.proj(R.name === "WebGPU" ? "gpu" : "gl"), [canvas.width, canvas.height, dpr, cam.ortho ? 1 : 0],
    [0.35, 0.55, 1.0, 0.32], [...pal.accent, 0.33], [...pal.cell, 1.3 * dpr],
    [lk[0], lk[1] * dpr, lk[2], 0], [cam.dist - cam.radius, cam.dist + cam.radius, 0, 0], [...pal.bg, 1]));
  drawOverlay(); drawTriad();
}

/* ------------------------------------------------------------------ atoms, bonds, cells */
function radiusOf(el) {
  const k = S.scale;
  if (S.style === "space") return rcov(el) * k;
  if (S.style === "stick") return 0.14 * k;
  return Math.max(0.2, 0.42 * rcov(el)) * k;
}
/** Selection or label entries mapped onto the current display atoms (same layer and atom, nearest image). */
function remap(list, atoms) {
  return list.map(s => {
    let best = -1, bd = Infinity;
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i];
      if (a.layer !== s.layer || a.src !== s.src) continue;
      const d = Math.hypot(a.x - s.p[0], a.y - s.p[1], a.z - s.p[2]);
      if (d < bd) { bd = d; best = i; }
    }
    if (best < 0) return null;
    const a = atoms[best];
    return { ...s, i: best, p: [a.x, a.y, a.z] };
  }).filter(Boolean);
}
function buildStructure() {
  if (!R) return;
  const atoms = [], bonds = [], segs = [];
  if (S.style !== "off") for (const L of atomLayers()) {
    const st = itemOf(L).structure, reps = st.pbc ? S.reps : [1, 1, 1];
    const da = displayAtoms(st, { reps, boundary: S.boundary && st.pbc, shift: fracShift(st.lattice), cart: S.t });
    for (const a of da) { a.layer = L.id; a.r = radiusOf(a.el); a.color = colorOf(a.el); }
    if (S.bonds && S.style !== "space") {
      const br = (S.style === "stick" ? 0.14 : 0.1) * S.scale;
      for (const [i, j] of bondList(da, { tol: S.tol, spacers: SPACERS })) {
        const a = da[i], b = da[j];
        bonds.push({ p0: [a.x, a.y, a.z], p1: [b.x, b.y, b.z], c0: a.color, c1: b.color, r: br });
      }
    }
    atoms.push(...da);
  }
  if (S.cell) {
    const cells = [];
    for (const L of S.layers) {
      if (!L.visible) continue;
      const it = itemOf(L), C = cellOf(it), o = originOf(it);
      if (C && !cells.some(c => sameCell(c.L, C) && c.o.every((x, k) => Math.abs(x - o[k]) < 1e-3))) cells.push({ L: C, pbc: pbcOf(it), o });
    }
    for (const { L, pbc, o } of cells) {
      const reps = pbc ? S.reps : [1, 1, 1];
      const box = (A, B, Cc, dashed) => {
        const P = (i, j, k) => [0, 1, 2].map(c => o[c] + i * A[c] + j * B[c] + k * Cc[c]);
        const E = [[0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0],
          [0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1]];
        for (const e of E) segs.push({ p0: P(e[0], e[1], e[2]), p1: P(e[3], e[4], e[5]), dashed });
      };
      box(L[0], L[1], L[2], false);
      if (reps.some(r => r > 1)) box(L[0].map(x => x * reps[0]), L[1].map(x => x * reps[1]), L[2].map(x => x * reps[2]), true);
    }
  }
  S.sel = remap(S.sel, atoms); S.labels = remap(S.labels, atoms);
  disp = { atoms, centers: Float32Array.from(atoms.flatMap(a => [a.x, a.y, a.z])), radii: Float32Array.from(atoms.map(a => a.r)) };
  uploadAtoms(); uploadLabels();
  R.setBonds(packBonds(bonds), bonds.length);
  R.setLines(packLines(segs), segs.length);
  updateSelectionBar(); updateInfo();
  requestRender();
}
function uploadAtoms() {
  const sel = new Set(S.sel.map(s => s.i));
  disp.atoms.forEach((a, i) => (a.sel = sel.has(i)));
  R.setAtoms(packAtoms(disp.atoms), disp.atoms.length);
}
const atlasSpecies = () => [...new Set(S.layers.filter(l => l.kind === "atoms").flatMap(l => itemOf(l).structure.species))];
function uploadLabels() {
  if (!R || !R.setLabels) return;
  const list = S.labels.map(s => { const a = disp.atoms[s.i]; return { x: a.x, y: a.y, z: a.z, r: Math.max(a.r, 0.36), text: a.el, color: a.color }; });
  R.setLabels(packLabels(list, S.atlas ? S.atlas.map : {}), list.length, S.atlas);
  labelBar();
}

/* ------------------------------------------------------------------ surfaces: one request at a time per layer */
function requestSurface(L, preview = false) { if (!L) return; L.next = { preview }; if (!L.busy) runSurface(L); }
function refreshSurfaces() { for (const L of fieldLayers()) requestSurface(L); }
async function runSurface(L) {
  const p = L.next;
  L.next = null;
  if (!p || !R) return;
  const it = itemOf(L), f = it && it.fields[L.fi];
  if (!f || !L.visible || !S.layers.includes(L)) { R.removeMeshGroup(L.id); L.mesh = null; if (L === activeField()) levelHint(); requestRender(); return; }
  L.busy = true;
  try {
    const reps = (f.periodic || f.closed) && pbcOf(it) ? S.reps : [1, 1, 1];
    const budget = p.preview ? 1.5e6 : 8e6, auto = autoStride(f, reps, budget);
    const stride = S.res === "auto" ? auto : Math.max(+S.res, p.preview ? auto : 1);
    const signed = L.mode === "log" && f.kind === "signed" && L.pm && f.stats.min < -1e-6 * f.stats.maxAbs;
    const { meshes } = await call("mesh", { ds: it.ds, fi: L.fi, reps, stride, isos: signed ? [L.iso, -L.iso] : [L.iso], shift: fracShift(f.span), cuts: L.cuts });
    if (S.layers.includes(L) && L.visible) {
      R.setMeshGroup(L.id, meshes.map((m, i) => ({ ...m, color: hexToRgb01(i ? L.neg : L.pos), opacity: L.opacity })));
      L.mesh = { tris: meshes.reduce((a, m) => a + m.triangles, 0), stride, truncated: meshes.some(m => m.truncated), n: meshes.length };
    }
    if (L === activeField()) { levelHint(); summaries(); }
    requestRender();
  } catch (e) { toast(e.message || String(e), true); }
  finally { L.busy = false; }
  if (L.next) runSurface(L);
}
function restyleLayer(L) {
  if (!R || !L.mesh) return;
  R.setMeshGroupStyle(L.id, [...Array(L.mesh.n)].map((_, i) => ({ color: hexToRgb01(i ? L.neg : L.pos), opacity: L.opacity })));
  requestRender();
}

/* ------------------------------------------------------------------ camera, views, triad */
function sceneSphere() {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const add = p => { for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); } };
  for (const a of disp.atoms) add([a.x, a.y, a.z]);
  for (const L of S.layers) {
    if (!L.visible) continue;
    const it = itemOf(L), C = cellOf(it);
    if (!C || (!S.cell && disp.atoms.length)) continue;
    const r = pbcOf(it) ? S.reps : [1, 1, 1];
    const o = originOf(it);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) add([0, 1, 2].map(c => o[c] + i * r[0] * C[0][c] + j * r[1] * C[1][c] + k * r[2] * C[2][c]));
  }
  if (!Number.isFinite(lo[0])) return { c: [0, 0, 0], r: 5 };
  return { c: lo.map((v, i) => (v + hi[i]) / 2), r: Math.max(1, 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])) };
}
function rows(toward, up) {
  const t = unit(toward), u0 = up.map((x, i) => x - dot(up, t) * t[i]);
  const u = norm(u0) > 1e-6 ? unit(u0) : unit(cross(t, Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  return [cross(u, t), u, t];
}
function fit() { const s = sceneSphere(); cam.fit(s.c, s.r); }
/** Re-centre the rotation pivot (and the depth range) on what is shown now, keeping the orientation and the zoom
 *  relative to the scene size; used whenever layers are added, removed, hidden or restored. */
function refit() {
  const s = sceneSphere(), k = cam.dist / cam.radius;
  cam.target = s.c.slice(); cam.radius = Math.max(s.r, 0.5); cam.dist = k * cam.radius;
  requestRender();
}
function viewAlong(i) {
  const pc = primaryCell();
  if (!pc) return;
  const L = pc.L;
  cam.q = qfromRows(rows(L[i], L[(i + 2) % 3])); fit(); requestRender();
}
function resetView() {
  const pc = primaryCell();
  if (pc) { const [a, b, c] = pc.L.map(unit); cam.q = qfromRows(rows([0, 1, 2].map(i => a[i] + 0.36 * b[i] + 0.3 * c[i]), c)); }
  else cam.q = [0, 0, 0, 1];
  fit(); requestRender();
}
function drawTriad() {
  const svg = $("triad"), pc = primaryCell();
  if (!pc) { svg.innerHTML = ""; return; }
  const M = qmat(cam.q);
  const ax = pc.L.map((a, i) => { const u = unit(a); return { r: [0, 1, 2].map(k => M[k][0] * u[0] + M[k][1] * u[1] + M[k][2] * u[2]), n: "abc"[i], c: pal.ax[i] }; })
    .sort((p, q) => p.r[2] - q.r[2]);
  svg.innerHTML = ax.map(({ r, n, c }) => {
    const x = 22 * r[0], y = -22 * r[1], o = 0.45 + 0.55 * (r[2] + 1) / 2;
    return `<g opacity="${o.toFixed(2)}"><line x1="0" y1="0" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${c}" stroke-width="2" stroke-linecap="round"/>` +
      `<text x="${(x * 1.32).toFixed(1)}" y="${(y * 1.32 + 4).toFixed(1)}" text-anchor="middle" fill="${c}">${n}</text></g>`;
  }).join("");
}

/* ------------------------------------------------------------------ selection, measurement, editing */
function atomLabel(s) {
  const L = layerById(s.layer), it = L && itemOf(L);
  if (!it) return "?";
  const multi = atomLayers(false).length > 1, k = multi ? atomLayers(false).indexOf(L) + 1 + "·" : "";
  return `${k}${it.structure.species[s.src]}${atomId(it.structure, s.src)}`;
}
function clickAt(x, y) {
  const i = disp.atoms.length ? pickSphere(cam.ray(x, y), disp.centers, disp.radii) : -1;
  if (S.labelMode) {
    if (i < 0) return;
    const k = S.labels.findIndex(s => s.i === i), a = disp.atoms[i];
    if (k >= 0) S.labels.splice(k, 1); else S.labels.push({ i, layer: a.layer, src: a.src, p: [a.x, a.y, a.z] });
    uploadLabels(); requestRender();
    return;
  }
  if (i < 0) return clearSelection();
  const k = S.sel.findIndex(s => s.i === i);
  if (k >= 0) S.sel.splice(k, 1);
  else { const a = disp.atoms[i]; S.sel.push({ i, layer: a.layer, src: a.src, p: [a.x, a.y, a.z] }); }
  S.region = null;
  uploadAtoms(); updateSelectionBar(); requestRender();
}
/** Region selection: path = [x0, y0, x1, y1, ...] in CSS pixels (a freehand polygon). Adds the atoms whose centres
 *  project inside it and records the region (polygon in normalised device coordinates, current view) for cutting. */
function regionSelect(path, add) {
  if (path.length < 6) return;
  if (!add) S.sel = [];
  const have = new Set(S.sel.map(s => s.i));
  disp.atoms.forEach((a, i) => {
    if (have.has(i)) return;
    const s = cam.project([a.x, a.y, a.z]);
    if (s && insidePolygon(path, s.x, s.y)) S.sel.push({ i, layer: a.layer, src: a.src, p: [a.x, a.y, a.z] });
  });
  const P = cam.proj("gl"), V = cam.view(), m = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let v = 0; for (let k = 0; k < 4; k++) v += P[4 * k + r] * V[4 * c + k]; m[4 * c + r] = v; }
  const poly = [];
  for (let k = 0; k < path.length; k += 2) poly.push(2 * path[k] / cam.w - 1, 1 - 2 * path[k + 1] / cam.h);
  S.region = { m: Array.from(m), poly };
  $("sec-edit").open = true;
  uploadAtoms(); updateSelectionBar(); requestRender();
}
function clearSelection() {
  if (!S.sel.length && !S.region) return;
  S.sel = []; S.region = null; uploadAtoms(); updateSelectionBar(); requestRender();
}
function updateSelectionBar() {
  const box = $("measure");
  if (!S.sel.length && !S.region) { box.hidden = true; box.innerHTML = ""; updateEdit(); return; }
  const parts = [];
  if (S.sel.length && S.sel.length <= 4) {
    const m = measureAll(S.sel.map(s => s.p));
    parts.push(`<span class="atoms">${S.sel.map(s => esc(atomLabel(s))).join(" · ")}</span>`);
    if (S.sel.length === 1) parts.push(`<span class="mv">${S.sel[0].p.map(v => v.toFixed(3)).join("  ")} Å</span>`);
    m.bonds.forEach((d, i) => parts.push(`<span class="mk">${esc(atomLabel(S.sel[i]))}–${esc(atomLabel(S.sel[i + 1]))}</span><span class="mv">${d.toFixed(3)} Å</span>`));
    m.angles.forEach((a, i) => parts.push(`<span class="mk">∠${esc(atomLabel(S.sel[i + 1]))}</span><span class="mv">${a.toFixed(2)}°</span>`));
    if (m.dihedral !== null) parts.push(`<span class="mk">dihedral</span><span class="mv">${m.dihedral.toFixed(2)}°</span>`);
  } else parts.push(`<span class="atoms">${S.sel.length ? `${S.sel.length} atoms selected` : "Region selected"}</span>`);
  // quick actions (the Edit panel holds the rest)
  const act = [];
  if (S.sel.length === 1) act.push(`<button type="button" data-act="center" title="${primaryCell() ? "Put this atom at the centre of the cell" : "Turn the view about this atom"}">Center</button>`);
  if (S.sel.length) act.push(`<button type="button" data-act="delete" class="warn" title="Delete the selected atoms in every periodic image and frame (Del)">Delete</button>`);
  if (S.region && fieldLayers(true).length) act.push(`<button type="button" data-act="cut" class="warn" title="Remove the surface inside the region">Cut surface</button>`);
  act.push(`<button type="button" data-act="edit" class="link-btn" title="Open the Edit panel">Edit ›</button>`);
  parts.push(`<span class="acts">${act.join("")}</span>`);
  parts.push(`<button type="button" class="x" id="sel-clear" aria-label="Clear the selection">×</button>`);
  box.innerHTML = parts.join("");
  box.hidden = false;
  $("sel-clear").addEventListener("click", clearSelection);
  updateEdit();
}
/** Edit panel: what is selected, which actions apply to it, the surface cuts and the last undoable change. */
function updateEdit() {
  const n = S.sel.length, pc = primaryCell(), s = n > 1 ? "s" : "";
  const info = $("sel-info");
  info.classList.toggle("none", !n && !S.region);
  info.textContent = n ? `${n} atom${s}${n <= 4 ? ": " + S.sel.map(atomLabel).join(" · ") : " selected"}` : S.region ? "A region with no atoms inside" : "Nothing selected";
  $("sel-clear2").hidden = !n && !S.region;
  $("sel-hint").hidden = !!(n || S.region);
  $("e-region").setAttribute("aria-pressed", String(S.boxMode));
  $("e-delete").disabled = !n;
  $("e-delete").textContent = n ? `Delete ${n} atom${s}` : "Delete atoms";
  $("e-cut").hidden = !fieldLayers(true).length; $("e-cut").disabled = !S.region;
  $("e-center").disabled = n !== 1;
  $("e-center").title = pc ? "Shift atoms and fields so that the selected atom sits at the centre of the cell" : "Turn the view about the selected atom";
  const cuts = Math.max(0, ...fieldLayers().map(L => L.cuts.length)), cs = cuts > 1 ? "s" : "";
  $("row-cuts").hidden = !cuts;
  $("cuts-n").textContent = `${cuts} cut${cs}`;
  const u = S.undo[S.undo.length - 1];
  $("e-undo").disabled = !u;
  $("e-undo-t").textContent = u ? `Undo: ${u.label}` : "Nothing to undo";
  const shifted = S.t.some(x => Math.abs(x) > 1e-9);
  $("sum-edit").textContent = [n ? `${n} selected` : S.region ? "region" : "", cuts ? `${cuts} cut${cs}` : "", shifted ? "shifted" : ""].filter(Boolean).join(" · ");
}
function drawOverlay() {
  const ov = $("ov");
  let h = "";
  if (dragPath && dragPath.length >= 4) {
    let d = `M${dragPath[0].toFixed(1)} ${dragPath[1].toFixed(1)}`;
    for (let k = 2; k < dragPath.length; k += 2) d += `L${dragPath[k].toFixed(1)} ${dragPath[k + 1].toFixed(1)}`;
    h += `<path class="box" d="${d}Z"/>`;
  }
  if (S.sel.length >= 2 && S.sel.length <= 4) {
    const P = S.sel.map(s => cam.project(s.p));
    if (P.every(Boolean)) {
      const m = measureAll(S.sel.map(s => s.p));
      const tag = (x, y, t) => { const w = 7 * t.length + 12; return `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)})"><rect x="${-w / 2}" y="-10" width="${w}" height="19" rx="6"/><text y="4" text-anchor="middle">${esc(t)}</text></g>`; };
      for (let i = 0; i + 1 < P.length; i++) h += `<line class="ml" x1="${P[i].x.toFixed(1)}" y1="${P[i].y.toFixed(1)}" x2="${P[i + 1].x.toFixed(1)}" y2="${P[i + 1].y.toFixed(1)}"/>`;
      m.bonds.forEach((d, i) => { h += tag((P[i].x + P[i + 1].x) / 2, (P[i].y + P[i + 1].y) / 2 - 12, d.toFixed(3) + " Å"); });
      m.angles.forEach((a, i) => { h += tag(P[i + 1].x, P[i + 1].y + 22, a.toFixed(1) + "°"); });
      if (m.dihedral !== null) h += tag((P[1].x + P[2].x) / 2, (P[1].y + P[2].y) / 2 + 22, "φ " + m.dihedral.toFixed(1) + "°");
    }
  }
  if (ov.innerHTML !== h) ov.innerHTML = h;
}
function deleteSelected() {
  if (!S.sel.length) return;
  const by = new Map();
  for (const s of S.sel) { if (!by.has(s.layer)) by.set(s.layer, new Set()); by.get(s.layer).add(s.src); }
  const changes = [], labels = S.labels.slice();
  let n = 0;
  for (const [lid, drop] of by) {
    const L = layerById(lid), it = itemOf(L), old = it.structure;
    // new index of every old atom (-1 for deleted ones), to keep labels on the same atoms
    const map = new Int32Array(old.species.length).fill(-1);
    let k = 0;
    for (let i = 0; i < old.species.length; i++) if (!drop.has(i)) map[i] = k++;
    it.structure = deleteAtoms(old, drop);
    changes.push({ it, old });
    n += drop.size;
    S.labels = S.labels.map(l => (l.layer !== lid ? l : map[l.src] < 0 ? null : { ...l, src: map[l.src] })).filter(Boolean);
  }
  pushUndo({ type: "atoms", changes, labels, label: `delete ${n} atom${n > 1 ? "s" : ""}` });
  S.sel = []; S.region = null;
  buildStructure();
  toast(`Deleted ${n} atom${n > 1 ? "s" : ""} (in every periodic image).`, false, 6000, { label: "Undo", fn: undo });
}
function cutSurface() {
  if (!S.region) return;
  const targets = fieldLayers(true).filter(L => L.mesh);
  for (const L of targets) { L.cuts.push(S.region); requestSurface(L); }
  pushUndo({ type: "cut", layers: targets.map(L => L.id), label: "cut the surface" });
  S.sel = []; S.region = null; uploadAtoms(); updateSelectionBar();
  toast("Cut the selected region out of the surface.", false, 6000, { label: "Undo", fn: undo });
}
/** Center on the selected atom: with a cell, shift so that it sits at the centre of its cell (periodic cells wrap the
 *  shift into (-1/2, 1/2]); without any cell, make it the point the view turns about. */
function centerOnSelected() {
  const s = S.sel[0];
  if (!s) return;
  const it = itemOf(layerById(s.layer)), own = cellOf(it), pc = primaryCell(), C = own || (pc && pc.L), o = own ? originOf(it) : pc ? pc.o : null;
  if (!C) { cam.target = s.p.slice(); requestRender(); toast(`The view now turns about ${atomLabel(s)}.`, false, 3000); return; }
  const mid = [0, 1, 2].map(c => o[c] + 0.5 * (C[0][c] + C[1][c] + C[2][c]));
  setShift(wrapShift(S.t.map((v, c) => v + mid[c] - s.p[c])));
}
/** A translation reduced to fractions in (-0.5, 0.5] of the main cell (shifts are periodic). */
function wrapShift(t) {
  const pc = primaryCell();
  if (!pc || !pc.pbc) return t;
  const f = cartToFrac(inv3(pc.L), t).map(x => { const y = x - Math.ceil(x - 0.5); return Math.abs(y) < 1e-9 ? 0 : +y.toFixed(9); });
  return fracToCart(pc.L, f);
}
/** Change the origin shift (Cartesian translation t, the same for every layer); selections and labels follow. */
function setShift(t, record = true) {
  const old = S.t.slice();
  if (t.every((x, c) => Math.abs(x - old[c]) < 1e-12)) return;
  if (record) pushUndo({ type: "shift", old, label: "shift the origin" });
  S.t = t.slice();
  for (const e of [...S.sel, ...S.labels]) e.p = e.p.map((x, c) => x + t[c] - old[c]);
  syncShiftInputs();
  buildStructure(); refreshSurfaces(); summaries(); updateEdit();
}
/** Shift inputs: fractions of the main cell's a, b, c, or Angstrom along x, y, z when nothing has a cell. */
function syncShiftInputs() {
  const pc = primaryCell(), f = pc ? fracShift(pc.L) : S.t;
  ["fx", "fy", "fz"].forEach((id, a) => { $(id).value = +f[a].toFixed(4); $(id).step = pc ? "0.05" : "0.5"; $(id).setAttribute("aria-label", pc ? `shift along ${"abc"[a]} (fraction)` : `shift along ${"xyz"[a]} (A)`); });
  $("shift-unit").textContent = pc ? (pc.pbc ? "fractions of a, b, c" : "fractions of the box a, b, c") : "Å along x, y, z";
}
/** Record an undoable change; u.label names it in the Edit panel ("Undo: delete 3 atoms"). */
function pushUndo(u) { S.undo.push(u); if (S.undo.length > 50) S.undo.shift(); updateEdit(); }
function undo() {
  const u = S.undo.pop();
  if (!u) return toast("Nothing to undo.");
  if (u.type === "atoms") { for (const c of u.changes) { c.it.structure = c.old; applyFrame(c.it); } S.labels = u.labels; S.sel = []; buildStructure(); }
  else if (u.type === "cut") for (const id of u.layers) { const L = layerById(id); if (L) { L.cuts.pop(); requestSurface(L); } }
  else if (u.type === "cuts") for (const c of u.list) { const L = layerById(c.layer); if (L) { L.cuts = c.cuts; requestSurface(L); } }
  else if (u.type === "shift") setShift(u.old, false);
  else if (u.type === "derive") {
    const i = S.layers.findIndex(l => l.id === u.layer);
    if (i >= 0) S.layers.splice(i, 1);
    const src = layerById(u.source);
    if (src) { src.visible = u.wasVisible; S.activeAtoms = src.id; }
    S.sel = S.sel.filter(e => e.layer !== u.layer); S.labels = S.labels.filter(e => e.layer !== u.layer);
    buildLayersUI(); updateChrome(); buildStructure(); refit();
  }
  else if (u.type === "layer") {
    S.layers.splice(Math.min(u.index, S.layers.length), 0, u.layer);
    if (u.layer.kind === "atoms") buildStructure(); else { S.active = u.layer.id; requestSurface(u.layer); syncFieldUI(); buildStructure(); }
    $("empty").hidden = true;
    buildLayersUI(); updateChrome(); refit();
  }
  updateEdit();
  toast(`Undone: ${u.label}.`, false, 2500);
}
/** Export the active atoms layer as shown (shift, deletions and supercell applied) as POSCAR, CIF or extended XYZ. */
function exportLayer(fmt) {
  const L = activeAtomsLayer();
  if (!L) return toast("There are no atoms to export.", true);
  const it = itemOf(L), st = it.structure, reps = st.pbc ? S.reps : [1, 1, 1];
  const sh = shownStructure(st, S.t, reps), pc = primaryCell(), f = pc ? fracShift(pc.L) : [0, 0, 0];
  const note = [S.t.some(x => Math.abs(x) > 1e-9) ? `shifted by ${f.map(x => +x.toFixed(4)).join(" ")}` : "", reps.some(r => r > 1) ? `${reps.join("x")} supercell` : ""].filter(Boolean).join(", ");
  const title = `${it.name}${it.traj ? ` frame ${frameOf(it) + 1}` : ""} (isoview${note ? ", " + note : ""})`;
  const txt = fmt === "cif" ? toCIF(sh, title) : fmt === "xyz" ? toXYZ(sh, title) : toPOSCAR(sh, [0, 0, 0], title);
  const ext = { cif: ".cif", xyz: ".xyz" }[fmt] || ".vasp";
  download(txt, it.name.split(" + ")[0].replace(/\.[^.]*$/, "").replace(/[^\w.-]+/g, "_") + "_isoview" + ext);
  toast(`Exported ${sh.species.length} atoms as ${fmt === "cif" ? "CIF" : fmt === "xyz" ? "extended XYZ" : "POSCAR"}${note ? " (" + note + ")" : ""}.`, false, 4000);
}
function download(data, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(data instanceof Blob ? data : new Blob([data], { type: "text/plain" }));
  a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
/** Primitive or conventional cell of the active atoms layer, added as a new layer (the source layer is hidden). */
function derive(kind) {
  const L = activeAtomsLayer();
  if (!L) return toast("Choose a structure layer first.", true);
  const it = itemOf(L), st = it.structure, tol = Math.max(1e-5, parseFloat($("t-tol").value) || 0.01);
  if (!st.pbc || !st.lattice) return toast("This needs a periodic structure.", true);
  let res;
  try { res = kind === "primitive" ? primitive(st, tol) : conventional(st, tol); }
  catch (e) { return toast(e.message || String(e), true); }
  if (kind === "primitive" && res.n === 1) return toast(`Already primitive: no lattice translations within ${tol} Å besides the cell's own.`, false, 6000);
  if (kind === "conventional" && sameCellShape(st, res.structure, tol)) return toast(`Already the conventional ${res.system} cell (${st.species.length} atoms).`, false, 6000);
  const nit = { id: nextItem++, name: `${kind} of ${it.name}${it.traj ? ` (frame ${frameOf(it) + 1})` : ""}`, ds: null, structure: res.structure, wave: null, fields: [], w: { s: 0, k: 0, b: -1, mode: "density", half: "x" } };
  S.items.set(nit.id, nit);
  const NL = { id: nextLayer++, item: nit.id, kind: "atoms", visible: true };
  S.layers.splice(S.layers.indexOf(L) + 1, 0, NL);
  const wasVisible = L.visible;
  L.visible = false; S.activeAtoms = NL.id;
  pushUndo({ type: "derive", layer: NL.id, source: L.id, wasVisible, label: `${kind} cell` });
  S.atlas = buildAtlas(atlasSpecies());
  buildLayersUI(); updateChrome(); buildStructure(); refit(); syncShiftInputs();
  const what = kind === "primitive" ? `primitive cell (${res.structure.species.length} atoms, ${res.n} lattice points in the original)` : `conventional ${res.system} cell (${res.structure.species.length} atoms, ${res.z}× the primitive)`;
  toast(`Added the ${what} as a new layer; the original is hidden.`, false, 7000, { label: "Undo", fn: undo });
}

/* ------------------------------------------------------------------ trajectories */
/** An energy (eV) for display: four decimals, or three significant digits when it is below 1 meV in size. */
const fmtE = e => (Math.abs(e) >= 1e-3 || e === 0 ? e.toFixed(4) : e.toExponential(2));
const trajItems = () => [...S.items.values()].filter(it => it.traj && S.layers.some(L => L.item === it.id));
const maxFrames = () => trajItems().reduce((m, it) => Math.max(m, it.traj.n), 0);
const frameOf = it => Math.min(S.frame, it.traj.n - 1);
/** The trajectory the panel describes: the chosen atoms layer if it is one, else the first. */
const mainTraj = () => { const L = activeAtomsLayer(), it = L && itemOf(L); return it && it.traj ? it : trajItems()[0] || null; };
/** Put frame S.frame (clamped) of a trajectory item into its structure: positions of the atoms it still has
 *  (structure.ids, 1-based numbers in the file) and the frame's cell when the cell changes. */
function applyFrame(it) {
  const t = it.traj;
  if (!t) return;
  const f = frameOf(it), st = it.structure, ids = st.ids, N = st.species.length, base = 3 * t.nat * f, pos = new Float64Array(3 * N);
  for (let i = 0; i < N; i++) { const j = 3 * (ids ? ids[i] - 1 : i); pos[3 * i] = t.pos[base + j]; pos[3 * i + 1] = t.pos[base + j + 1]; pos[3 * i + 2] = t.pos[base + j + 2]; }
  const lattice = t.cells ? [0, 1, 2].map(r => Array.from(t.cells.subarray(9 * f + 3 * r, 9 * f + 3 * r + 3))) : st.lattice;
  it.structure = { ...st, pos, lattice };
}
/** Show frame f (clamped to the longest trajectory) in every trajectory. */
function setFrame(f) {
  const n = maxFrames();
  if (!n) return;
  S.frame = Math.max(0, Math.min(n - 1, Math.round(f)));
  for (const it of trajItems()) applyFrame(it);
  buildStructure(); updatePlayer();
}
let playLast = 0, playDir = 1;
function play(on) {
  const n = maxFrames();
  S.playing = !!on && n > 1;
  if (S.playing && S.loop === "once" && S.frame >= n - 1) setFrame(0);
  if (S.playing) { playLast = performance.now(); requestAnimationFrame(tick); }
  updatePlayer();
}
function tick(t) {
  if (!S.playing) return;
  const dt = 1000 / S.fps;
  if (t - playLast >= dt) {
    // WHY: keep the phase (no drift); if drawing falls far behind, restart the clock instead of racing to catch up
    playLast = t - playLast > 4 * dt ? t : playLast + dt;
    advance();
  }
  if (S.playing) requestAnimationFrame(tick);
}
/** Next frame of the playback: S.step frames on, then loop, bounce or stop at either end. */
function advance() {
  const n = maxFrames();
  let f = S.frame + playDir * S.step;
  if (f > n - 1 || f < 0) {
    if (S.loop === "loop") f = playDir > 0 ? 0 : n - 1;
    else if (S.loop === "bounce") { playDir = -playDir; f = Math.max(0, Math.min(n - 1, S.frame + playDir * S.step)); }
    else { f = Math.max(0, Math.min(n - 1, f)); setFrame(f); play(false); return; }
  }
  setFrame(f);
}
function updatePlayer() {
  const n = maxFrames(), on = n > 1;
  $("player").hidden = !on; document.body.classList.toggle("has-player", on);
  $("sec-traj").hidden = !on;
  if (!on) return;
  const sl = $("pl-frame");
  sl.max = n - 1; sl.value = S.frame;
  const it = mainTraj(), E = it && it.traj.energy, e = E ? E[frameOf(it)] : NaN;
  $("pl-count").textContent = `${S.frame + 1} / ${n}${Number.isFinite(e) ? ` · ${fmtE(e)} eV` : ""}`;
  $("pl-play").setAttribute("aria-pressed", String(S.playing));
  const pt = S.playing ? "Pause (Space)" : "Play (Space)";
  $("pl-play").title = pt; $("pl-play").setAttribute("aria-label", pt);
  $("sum-traj").textContent = `${S.frame + 1} / ${n}${S.playing ? " · playing" : ""}`;
  if (it) $("tr-info").textContent = `${it.traj.n} of ${it.name}${it.traj.cells ? " · cell changes" : ""}`;
  drawSpark();
  movieNote();
}
/** Energy per frame of the main trajectory as a line; the current frame is marked; a click goes to that frame. */
function drawSpark() {
  const it = mainTraj(), E = it && it.traj.energy, box = $("tr-spark");
  box.hidden = !E;
  if (!E) return;
  const svg = $("tr-svg"), W = svg.clientWidth || 280, H = 52, n = E.length;
  let lo = Infinity, hi = -Infinity;
  for (const e of E) if (Number.isFinite(e)) { if (e < lo) lo = e; if (e > hi) hi = e; }
  const span = hi - lo || 1, X = f => 3 + (W - 6) * (n > 1 ? f / (n - 1) : 0.5), Y = e => 5 + (H - 10) * (hi - e) / span;
  let d = "";
  for (let f = 0; f < n; f++) if (Number.isFinite(E[f])) d += `${d ? "L" : "M"}${X(f).toFixed(1)} ${Y(E[f]).toFixed(1)}`;
  const f0 = frameOf(it), e0 = E[f0];
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `<line class="base" x1="0" x2="${W}" y1="${H - 0.5}" y2="${H - 0.5}"/><path class="ln" d="${d}"/>` + (Number.isFinite(e0) ? `<circle class="now" cx="${X(f0).toFixed(1)}" cy="${Y(e0).toFixed(1)}" r="3.5"/>` : "");
  $("tr-e-range").textContent = `E ${fmtE(lo)} … ${fmtE(hi)} eV`;
  $("tr-e-now").textContent = Number.isFinite(e0) ? `frame ${f0 + 1}: ${fmtE(e0 - lo)} eV above min` : `frame ${f0 + 1}`;
}
const sparkFrame = e => { const it = mainTraj(), svg = $("tr-svg"), r = svg.getBoundingClientRect(), W = r.width || 280; return Math.round(Math.max(0, Math.min(1, (e.clientX - r.left - 3) / (W - 6))) * (it.traj.n - 1)); };
function videoMime() {
  if (typeof MediaRecorder === "undefined" || !HTMLCanvasElement.prototype.captureStream) return null;
  for (const m of ["video/mp4;codecs=avc1.640028", "video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]) {
    try { if (MediaRecorder.isTypeSupported(m)) return m; } catch (e) { /* ignore */ }
  }
  return null;
}
/** Frames the movie will contain: from..to (1-based inputs) every S.step frames. */
function movieFrames() {
  const n = maxFrames(), a = Math.max(1, Math.min(n, +$("mv-from").value || 1)), b = Math.max(a, Math.min(n, +$("mv-to").value || n)), list = [];
  for (let f = a - 1; f <= b - 1; f += S.step) list.push(f);
  return list;
}
const movieSize = () => { const s = S.movie.scale, v = S.movie.fmt === "video"; let w = Math.max(2, Math.round(canvas.width * s)), h = Math.max(2, Math.round(canvas.height * s)); if (v) { w -= w % 2; h -= h % 2; } return [w, h]; };
function movieNote() {
  if (S.exporting) return;
  const list = movieFrames(), fps = Math.max(1, Math.min(60, +$("mv-fps").value || S.fps)), [w, h] = movieSize(), mime = videoMime();
  const ext = !mime ? "" : mime.includes("mp4") ? "MP4" : "WebM";
  $("mfmt-video").hidden = !mime;
  if (S.movie.fmt === "video" && !mime) S.movie.fmt = "gif";
  $("mv-go").textContent = S.movie.fmt === "gif" ? "Export GIF" : `Export ${ext}`;
  const capped = S.movie.fmt === "gif" && fps > 50 ? " (GIF plays at most 50 fps)" : "";
  $("mv-note").textContent = `${list.length} frame${list.length === 1 ? "" : "s"} · ${(list.length / fps).toFixed(1)} s at ${fps} fps${capped} · ${w} × ${h} px${S.movie.fmt === "video" ? " · recorded in real time" : ""}`;
}
/** Render the chosen frames and save them as an animated GIF or a video (the export button cancels while it runs). */
async function exportMovie() {
  if (S.exporting) { S.exporting.cancel = true; return; }
  if (!R || maxFrames() < 2) return;
  const list = movieFrames(), fps = Math.max(1, Math.min(60, +$("mv-fps").value || S.fps)), gif = S.movie.fmt === "gif";
  const [W, H] = movieSize(), keep = S.frame, wasPlaying = S.playing, job = { cancel: false };
  play(false); S.exporting = job;
  const btn = $("mv-go"); btn.classList.add("busy");
  const c2 = document.createElement("canvas"); c2.width = W; c2.height = H;
  const g = c2.getContext("2d", { willReadFrequently: gif }), bg = css("--bg") || "#ffffff";
  const grab = f => { setFrame(f); frame(); g.fillStyle = bg; g.fillRect(0, 0, W, H); g.drawImage(canvas, 0, 0, W, H); };
  const name = (mainTraj().name.split(" + ")[0] || "isoview").replace(/\.[^.]*$/, "").replace(/[^\w.-]+/g, "_");
  const yieldNow = () => new Promise(r => setTimeout(r, 0));
  try {
    if (gif) {
      const picks = [...new Set([0, 0.2, 0.4, 0.6, 0.8, 1].map(q => list[Math.round(q * (list.length - 1))]))], samples = [];
      for (const f of picks) { grab(f); samples.push(g.getImageData(0, 0, W, H).data); }
      const enc = new GifWriter(W, H, buildPalette(samples));
      let acc = 0;
      for (let k = 0; k < list.length; k++) {
        if (job.cancel) throw new Error("cancelled");
        grab(list[k]);
        acc += 100 / fps; const d = Math.max(2, Math.round(acc)); acc -= d;
        enc.frame(g.getImageData(0, 0, W, H).data, d);
        btn.textContent = `Cancel (${Math.round(100 * (k + 1) / list.length)}%)`;
        if (k % 2 === 1) await yieldNow();
      }
      const bytes = enc.finish();
      download(new Blob([bytes], { type: "image/gif" }), `${name}.gif`);
      toast(`Saved a GIF: ${list.length} frames, ${W} × ${H} px, ${(bytes.length / 1048576).toFixed(1)} MB.`, false, 5000);
    } else {
      const mime = videoMime(), stream = c2.captureStream(0), track = stream.getVideoTracks()[0];
      const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: Math.round(Math.min(2.5e7, Math.max(2e6, 0.2 * W * H * fps))) });
      const chunks = [], stopped = new Promise(r => (rec.onstop = r));
      rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
      grab(list[0]); rec.start();
      const t0 = performance.now();
      for (let k = 0; k < list.length; k++) {
        if (job.cancel) { rec.stop(); await stopped; throw new Error("cancelled"); }
        grab(list[k]);
        if (track.requestFrame) track.requestFrame();
        btn.textContent = `Cancel (${Math.round(100 * (k + 1) / list.length)}%)`;
        const wait = t0 + (k + 1) * 1000 / fps - performance.now();
        await new Promise(r => setTimeout(r, Math.max(0, wait)));
      }
      rec.stop(); await stopped;
      const blob = new Blob(chunks, { type: mime.split(";")[0] });
      download(blob, `${name}.${mime.includes("mp4") ? "mp4" : "webm"}`);
      toast(`Saved a video: ${list.length} frames at ${fps} fps, ${W} × ${H} px, ${(blob.size / 1048576).toFixed(1)} MB.`, false, 5000);
    }
  } catch (e) { if (e.message !== "cancelled") toast(`Could not make the movie: ${e.message || e}`, true); else toast("Movie export cancelled.", false, 2500); }
  finally {
    S.exporting = null; btn.classList.remove("busy");
    setFrame(keep); movieNote();
    if (wasPlaying) play(true);
  }
}
/** Every frame (at the playback step) of the chosen trajectory as extended XYZ, shift, deletions and supercell applied. */
function exportFrames() {
  const it = mainTraj();
  if (!it) return;
  const keep = S.frame, reps = it.structure.pbc ? S.reps : [1, 1, 1], parts = [];
  for (let f = 0; f < it.traj.n; f += S.step) {
    S.frame = f; applyFrame(it);
    const e = it.traj.energy ? it.traj.energy[f] : NaN;
    parts.push(toXYZ(shownStructure(it.structure, S.t, reps), `${it.name} frame ${f + 1}`, Number.isFinite(e) ? { energy: e } : {}));
  }
  S.frame = keep; applyFrame(it);
  download(parts.join(""), (it.name.split(" + ")[0] || "isoview").replace(/\.[^.]*$/, "").replace(/[^\w.-]+/g, "_") + "_frames.xyz");
  toast(`Exported ${parts.length} frames as extended XYZ.`, false, 4000);
}

/* ------------------------------------------------------------------ label mode */
function setLabelMode(on) {
  S.labelMode = on;
  if (on) { clearSelection(); setBoxMode(false); }
  $("btn-label").setAttribute("aria-pressed", String(on));
  $("stage").classList.toggle("labelling", on);
  labelBar();
}
function setBoxMode(on) {
  S.boxMode = on;
  if (on && S.labelMode) setLabelMode(false);
  $("stage").classList.toggle("boxing", on);
  updateEdit();
}
function labelBar() {
  $("labelbar").hidden = !S.labelMode;
  if (S.labelMode) $("label-count").textContent = S.labels.length ? `${S.labels.length} labelled` : "Click atoms to label them";
}

/* ------------------------------------------------------------------ pointer */
const stage = $("stage"), ptrs = new Map();
let drag = null, dragPath = null;
stage.addEventListener("pointerdown", e => {
  if (!R || !S.layers.length) return;
  if (stage.setPointerCapture) stage.setPointerCapture(e.pointerId);
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (ptrs.size === 1) {
    const box = e.button === 0 && (e.shiftKey || S.boxMode) && !S.labelMode;
    drag = { x0: e.clientX, y0: e.clientY, moved: false, box, add: e.shiftKey && S.sel.length > 0, pan: !box && (e.button !== 0 || e.ctrlKey || e.metaKey), button: e.button };
  } else if (drag) { drag.moved = true; drag.box = false; dragPath = null; }
});
stage.addEventListener("pointermove", e => {
  const prev = ptrs.get(e.pointerId);
  if (!prev || !drag) return;
  const cur = { x: e.clientX, y: e.clientY };
  if (ptrs.size === 2) {
    const other = [...ptrs.entries()].find(([id]) => id !== e.pointerId)[1];
    const d0 = Math.hypot(prev.x - other.x, prev.y - other.y), d1 = Math.hypot(cur.x - other.x, cur.y - other.y);
    if (d0 > 0 && d1 > 0) cam.zoomBy(d0 / d1);
    cam.panBy((cur.x - prev.x) / 2, (cur.y - prev.y) / 2);
  } else {
    if (!drag.moved && Math.hypot(cur.x - drag.x0, cur.y - drag.y0) > 4) drag.moved = true;
    if (drag.moved) {
      if (drag.box) {
        if (!dragPath) dragPath = [drag.x0, drag.y0];
        const n = dragPath.length;
        if (Math.hypot(cur.x - dragPath[n - 2], cur.y - dragPath[n - 1]) > 3) dragPath.push(cur.x, cur.y);
      }
      else if (drag.pan) cam.panBy(cur.x - prev.x, cur.y - prev.y);
      else cam.rotateBy(cur.x - prev.x, cur.y - prev.y);
    }
  }
  ptrs.set(e.pointerId, cur);
  if (drag.moved) requestRender();
});
const endPtr = e => {
  if (!ptrs.has(e.pointerId)) return;
  ptrs.delete(e.pointerId);
  if (drag && e.type === "pointerup" && ptrs.size === 0) {
    if (drag.box && drag.moved && dragPath) regionSelect(dragPath, drag.add);
    else if (!drag.moved && drag.button === 0) clickAt(e.clientX, e.clientY);
  }
  if (ptrs.size === 0) { drag = null; if (dragPath) { dragPath = null; requestRender(); } }
};
stage.addEventListener("pointerup", endPtr);
stage.addEventListener("pointercancel", endPtr);
stage.addEventListener("wheel", e => { if (!R) return; e.preventDefault(); cam.zoomBy(Math.exp(e.deltaY * (e.deltaMode ? 0.05 : 0.0015))); requestRender(); }, { passive: false });
stage.addEventListener("contextmenu", e => e.preventDefault());
window.addEventListener("resize", () => { if (R) resize(); });

/* ------------------------------------------------------------------ fields and levels (active field layer) */
function initLevel(L, keepP) {
  const f = itemOf(L).fields[L.fi];
  if (!f) return;
  L.mode = f.kind === "linear" ? "linear" : "log";
  // WHY: a total density (many electrons) at 50% enclosed hugs the cores; 90% gives the usual bonding-scale level
  if (!keepP || L.p === undefined) L.p = f.stats.integral > 3 ? 0.9 : 0.5;
  if (L.mode === "log") L.iso = isoForFraction(f.stats, L.p);
  else L.iso = /ELF/i.test(f.name) && f.stats.max >= 0.75 && f.stats.min <= 0.75 ? 0.75 : 0.5 * (f.stats.min + f.stats.max);
}
const activeStats = () => { const L = activeField(); return L ? itemOf(L).fields[L.fi] : null; };
function logRange(st) { const hi = Math.log10(st.maxAbs) - 1e-3; return [hi - 6, hi]; }
function syncFieldUI() {
  const L = activeField(), f = activeStats();
  $("sec-surface").hidden = !L;
  const it = L && itemOf(L);
  $("sec-orbital").hidden = !(it && it.wave);
  if (!L) return;
  $("field").innerHTML = it.fields.map((g, i) => `<option value="${i}">${esc(g.name)}</option>`).join("");
  $("field").value = L.fi;
  $("row-field").hidden = it.fields.length < 2;
  $("surf").checked = L.visible; $("pm").checked = L.pm; $("opacity").value = L.opacity;
  $("surf-layer").textContent = it.name;
  if (f) {
    const sl = $("iso");
    if (L.mode === "log") { const [lo, hi] = logRange(f.stats); sl.value = Math.round(1000 * (Math.log10(Math.max(L.iso, 1e-300)) - lo) / (hi - lo)); }
    else sl.value = Math.round(1000 * (L.iso - f.stats.min) / ((f.stats.max - f.stats.min) || 1));
    $("iso-num").value = fmt(L.iso);
  }
  if (it.wave) syncWaveUI(it);
  levelHint(); summaries();
}
function levelHint() {
  const L = activeField(), f = activeStats();
  if (!L || !f) return;
  const st = f.stats, u = f.unit ? " " + f.unit : "", signed = f.kind === "signed";
  if (L.mode === "log") $("iso-hint").textContent = `${signed ? "±" : ""}${sup(fmt(L.iso))}${u} · encloses ${Math.round(100 * fractionAbove(st, L.iso))}% of ${signed ? "∫|f|" : "the total"}`;
  else $("iso-hint").textContent = `${sup(fmt(L.iso))}${u}`;
  $("row-pm").hidden = !(L.mode === "log" && signed && st.min < -1e-6 * st.maxAbs);
  const I = st.integral, fi = Math.abs(I) >= 100 ? I.toFixed(2) : +I.toPrecision(5) + "";
  const m = L.mesh, ord = n => n + (["th", "st", "nd", "rd"][n] || "th");
  const tri = m ? ` · ${m.tris >= 1e6 ? (m.tris / 1e6).toFixed(2) + "M" : Math.round(m.tris / 1e3) + "k"} triangles${m.stride > 1 ? ` (every ${ord(m.stride)} point)` : ""}${m.truncated ? ", truncated" : ""}` : "";
  const fine = f.n0 && f.n0.some((x, a) => x !== f.n[a]) ? ` (drawn on ${f.n.join(" × ")})` : "";
  $("stats").textContent = `${(f.n0 || f.n).join(" × ")} grid${fine} · min ${fmt(st.min)} · max ${fmt(st.max)} · ∫ ${fi}${f.unit === "e/Å³" ? " e" : ""}${tri}`;
}
function setLevelFromSlider(preview) {
  const L = activeField(), f = activeStats();
  if (!L || !f) return;
  const u = +$("iso").value / 1000;
  if (L.mode === "log") { const [lo, hi] = logRange(f.stats); L.iso = Math.pow(10, lo + u * (hi - lo)); L.p = fractionAbove(f.stats, L.iso); }
  else L.iso = f.stats.min + u * (f.stats.max - f.stats.min);
  $("iso-num").value = fmt(L.iso);
  levelHint(); summaries();
  requestSurface(L, preview);
}

/* ------------------------------------------------------------------ WAVECAR (active field layer of a WAVECAR item) */
async function showBand(L, b) {
  const it = itemOf(L), w = it.wave, { s, k, mode, half } = it.w;
  it.w.b = b;
  busy(true);
  try {
    const { field } = await call("orbital", { ds: it.ds, s, k, b, signed: mode === "psi" && w.type === "gam", half });
    it.fields = [field]; L.fi = 0;
    initLevel(L, true);
    if (L === activeField()) { syncFieldUI(); markBand(it); }
    buildLayersUI();
    requestSurface(L);
  } catch (e) { toast(e.message || String(e), true); }
  finally { busy(false); }
}
function syncWaveUI(it) {
  const w = it.wave;
  $("row-spin").hidden = w.nspin < 2; $("row-k").hidden = w.nk < 2;
  $("row-wmode").hidden = $("row-half").hidden = w.type !== "gam";
  $("kpt").innerHTML = w.kpts.map((k, i) => `<option value="${i}">${i + 1}: ${k.map(x => x.toFixed(3)).join(" ")}</option>`).join("");
  $("kpt").value = it.w.k;
  if ($("bands").dataset.item !== String(it.id) || $("bands").dataset.sk !== `${it.w.s}|${it.w.k}`) buildBands(it);
  markBand(it); segState();
}
function buildBands(it) {
  const w = it.wave, { s, k } = it.w, e = w.eig[s][k], o = w.occ[s][k], fr = w.frontier[s][k], list = $("bands"), rows = [];
  for (let b = 0; b < w.nb; b++) {
    const cls = ["band", o[b] > 0.5 ? "occ" : "", b === fr.homo ? "homo" : "", b === fr.lumo ? "lumo" : ""].join(" ");
    rows.push(`<div class="${cls}" role="option" data-b="${b}" aria-selected="false"><span>${b + 1}</span><span class="e">${e[b].toFixed(3)}</span><span class="o">${o[b].toFixed(2)}</span></div>`);
  }
  list.innerHTML = rows.join("");
  list.dataset.item = String(it.id); list.dataset.sk = `${s}|${k}`;
  const h = list.querySelector(`[data-b="${Math.max(0, it.w.b >= 0 ? it.w.b : fr.homo)}"]`);
  if (h) list.scrollTop = h.offsetTop - list.clientHeight / 2;
}
function markBand(it) {
  for (const el of $("bands").querySelectorAll('[aria-selected="true"]')) el.setAttribute("aria-selected", "false");
  const el = $("bands").querySelector(`[data-b="${it.w.b}"]`);
  if (el) el.setAttribute("aria-selected", "true");
}
const activeWave = () => { const L = activeField(); const it = L && itemOf(L); return it && it.wave ? { L, it } : null; };

/* ------------------------------------------------------------------ layers panel */
function buildLayersUI() {
  const box = $("layers"), aa = activeAtomsLayer();
  box.innerHTML = S.layers.map(L => {
    const it = itemOf(L), f = L.kind === "field" ? it.fields[L.fi] : null;
    const kind = L.kind === "atoms" ? `${it.structure.species.length} atoms${it.traj ? ` · ${it.traj.n} frames` : ""}` : f ? f.name : it.wave ? "orbital" : "field";
    const signed = f && f.kind === "signed";
    const sw = L.kind === "field" ? `<input type="color" class="cpos" value="${L.pos}" aria-label="Surface colour">${signed ? `<input type="color" class="cneg" value="${L.neg}" aria-label="Negative surface colour">` : ""}` : "";
    const act = L.id === S.active || (L.kind === "atoms" && aa && L.id === aa.id);
    return `<div class="layer${act ? " active" : ""}${L.visible ? "" : " off"}" data-id="${L.id}">` +
      `<button type="button" class="eye" aria-pressed="${L.visible}" aria-label="Show or hide this layer"><svg viewBox="0 0 24 24"><path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="2.6" fill="currentColor"/></svg></button>` +
      `<span class="lname" title="${esc(it.name)}"><span class="lk">${esc(kind)}</span>${esc(it.name)}</span>${sw}` +
      `<button type="button" class="rm" aria-label="Remove this layer">×</button></div>`;
  }).join("");
  $("sum-layers").textContent = `${S.layers.length} layer${S.layers.length === 1 ? "" : "s"}`;
  updateTools();
}
/** Structure tools act on the active atoms layer (click an atoms row to choose it). */
function updateTools() {
  const L = activeAtomsLayer(), st = L && itemOf(L).structure;
  $("tools").hidden = !L;
  if (!L) return;
  $("tools-for").textContent = `${itemOf(L).name} · ${st.species.length} atoms`;
  $("t-prim").disabled = $("t-conv").disabled = !(st.pbc && st.lattice);
}
$("layers").addEventListener("click", e => {
  const row = e.target.closest(".layer");
  if (!row) return;
  const L = layerById(+row.dataset.id);
  if (!L) return;
  if (e.target.closest(".eye")) {
    L.visible = !L.visible;
    if (L.kind === "field") { requestSurface(L); if (L === activeField()) $("surf").checked = L.visible; }
    buildStructure(); buildLayersUI(); refit();
  } else if (e.target.closest(".rm")) removeLayer(L);
  else if (L.kind === "field" && !e.target.closest("input")) { S.active = L.id; buildLayersUI(); syncFieldUI(); $("sec-surface").open = true; }
  else if (L.kind === "atoms") { S.activeAtoms = L.id; buildLayersUI(); }
});
$("layers").addEventListener("input", e => {
  const row = e.target.closest(".layer"), L = row && layerById(+row.dataset.id);
  if (!L) return;
  if (e.target.classList.contains("cpos")) L.pos = e.target.value;
  if (e.target.classList.contains("cneg")) L.neg = e.target.value;
  restyleLayer(L);
});
function removeLayer(L) {
  const index = S.layers.indexOf(L);
  S.layers.splice(index, 1);
  pushUndo({ type: "layer", layer: L, index, label: "remove the layer" });
  if (L.kind === "field") { R.removeMeshGroup(L.id); L.mesh = null; if (S.active === L.id) S.active = (fieldLayers()[0] || {}).id ?? null; syncFieldUI(); }
  S.sel = S.sel.filter(s => s.layer !== L.id); S.labels = S.labels.filter(s => s.layer !== L.id);
  // if only hidden layers are left (e.g. a duplicate atoms layer), show them rather than an empty view
  if (S.layers.length && !S.layers.some(l => l.visible)) for (const l of S.layers) { l.visible = true; if (l.kind === "field") requestSurface(l); }
  buildStructure(); buildLayersUI(); updateChrome();
  if (!S.layers.length) { setLabelMode(false); setBoxMode(false); showHome(); }
  else refit();
  toast(S.layers.length ? "Layer removed." : "Removed the last layer.", false, 6000, { label: "Undo", fn: undo });
}

/* ------------------------------------------------------------------ loading */
/** mode: "replace", "add" or "auto" (add when it completes the scene: atoms for a WAVECAR, or pasted text). */
async function openFiles(list, mode = "auto") {
  const files = [...list];
  if (!files.length) return;
  busy(true);
  try {
    if (!(await ensureRenderer())) return;
    const d = await call("open", { files });
    let add = mode === "add" && S.layers.length > 0;
    if (mode === "auto" && S.layers.length) {
      // a structure-only file dropped on a WAVECAR without atoms, with the same cell, completes it
      const w = [...S.items.values()].find(it => it.wave && !it.structure);
      if (w && !d.fields.length && !d.wave && d.structure && sameCell(d.structure.lattice, w.wave.lattice)) add = true;
    }
    await addItem({ name: files.map(f => f.name).join(" + "), ...d, files }, add);
  } catch (e) { toast(e.message || String(e), true, 8000); console.error(e); }
  finally { busy(false); }
}
function clearScene() {
  for (const it of S.items.values()) if (it.ds) call("drop", { ds: it.ds }).catch(() => {});
  S.items.clear(); S.layers = []; S.active = null; S.activeAtoms = null; S.sel = []; S.region = null; S.labels = []; S.undo = [];
  S.frame = 0; S.playing = false;
  S.t = [0, 0, 0]; ["fx", "fy", "fz"].forEach(id => ($(id).value = 0));
  S.reps = [1, 1, 1]; ["ra", "rb", "rc"].forEach(id => ($(id).value = 1));
  if (R) R.clearMeshes();
  setLabelMode(false); setBoxMode(false);
}
function sameAtoms(a, b) {
  if (!a || !b || a.species.length !== b.species.length || !sameCell(a.lattice, b.lattice)) return false;
  for (let i = 0; i < a.pos.length; i++) if (Math.abs(a.pos[i] - b.pos[i]) > 1e-3) return false;
  return a.species.every((s, i) => s === b.species[i]);
}
async function addItem(d, adding, quiet = false) {
  if (!(await ensureRenderer())) return;
  if (!adding) clearScene();
  const notes = [...(d.notes || [])];
  const it = { id: nextItem++, name: d.name, ds: d.ds, structure: d.structure, wave: d.wave || null, traj: d.traj || null, fields: d.fields || [], w: { s: 0, k: 0, b: -1, mode: "density", half: "x" },
    key: d.key || Math.random().toString(36).slice(2) + Date.now().toString(36), files: d.files || null, example: d.example || null, kept: false };
  if (it.wave && !(it.structure && sameCell(it.structure.lattice, it.wave.lattice))) {
    if (it.structure) notes.push("the structure's cell differs from the WAVECAR's, so only the cell is drawn");
    else if (!atomLayers(false).length) notes.push("add the matching POSCAR or CONTCAR to draw the atoms");
    it.structure = null;
  }
  if (it.structure && !it.structure.species.length && !it.fields.length && !it.wave) throw new Error("No atoms or data found in that file.");
  S.items.set(it.id, it);
  if (d.alreadyKept) it.kept = true; else keepFiles(it);
  if (it.structure && it.structure.species.length) {
    const dup = adding && atomLayers().some(L => sameAtoms(itemOf(L).structure, it.structure));
    S.layers.push({ id: nextLayer++, item: it.id, kind: "atoms", visible: !dup });
    if (dup) notes.push("its atoms duplicate a visible layer, so that atoms layer starts hidden");
  }
  let FL = null;
  if (it.fields.length || it.wave) {
    const c = FIELD_COLORS[fieldLayers().length % FIELD_COLORS.length];
    FL = { id: nextLayer++, item: it.id, kind: "field", visible: true, fi: 0, pm: true, opacity: 0.85, pos: c[0], neg: c[1], cuts: [], mesh: null };
    S.layers.push(FL);
    S.active = FL.id;
    if (it.fields.length) initLevel(FL, false);
  }
  S.atlas = buildAtlas(atlasSpecies());
  $("empty").hidden = true;
  if (it.traj) { applyFrame(it); $("mv-to").value = maxFrames(); $("mv-from").max = $("mv-to").max = maxFrames(); }
  buildLayersUI(); updateChrome(); buildStructure();
  if (!adding) resetView(); else refit();
  for (const id of ["sec-structure", "sec-surface", "sec-orbital", "sec-view"]) $(id).open = false;
  if (!adding) $("sec-edit").open = false;
  $("sec-layers").open = true;
  syncFieldUI(); syncShiftInputs();
  if (FL && it.wave) { if (!adding) $("sec-orbital").open = true; await showBand(FL, Math.max(0, it.wave.frontier[0][0].homo)); }
  else if (FL) { if (!adding) $("sec-surface").open = true; requestSurface(FL); }
  else if (it.traj && !adding) $("sec-traj").open = true;
  if (notes.length && !quiet) toast(notes.join("; "), false, 7000);
}

/* ------------------------------------------------------------------ the scene kept in this browser (store.js) */
let saveTimer = 0, restoring = false;
/** Store an item's files once, while the scene's kept files stay within store.LIMIT bytes. */
function keepFiles(it) {
  if (!it.files || !it.files.length) return;
  const used = [...S.items.values()].reduce((a, o) => a + (o.kept ? o.files.reduce((b, f) => b + f.size, 0) : 0), 0);
  const size = it.files.reduce((a, f) => a + f.size, 0);
  if (used + size > store.LIMIT) return;
  it.kept = true;
  store.putFiles(it.key, it.files).then(r => { if (!r) it.kept = false; });
}
function saveSoon() {
  if (restoring) return;
  clearTimeout(saveTimer);
  // WHY: prune by every item still in memory (a removed layer can come back with Undo), not only the shown ones
  saveTimer = setTimeout(() => { if (!restoring && !S.exporting) store.putState(serialize()).then(() => store.pruneFiles(new Set([...S.items.values()].filter(i => i.kept).map(i => i.key)))); }, 700);
}
/** The scene as plain data. Items keep their key and name; an item whose files are not kept (too large), a derived
 *  cell and an edited structure also carry the structure (a trajectory carries only the ids of its atoms).
 *  withFiles adds the File objects themselves (for New scene's Undo, in memory). */
function serialize(withFiles = false) {
  const items = [...S.items.values()].filter(it => S.layers.some(L => L.item === it.id));
  const index = new Map(items.map((it, k) => [it.id, k]));
  const state = {
    v: 1, saved: Date.now(),
    items: items.map(it => ({ key: it.key, name: it.name, example: it.example, kept: it.kept, w: it.w, hadFiles: !!(it.files && it.files.length),
      structure: it.structure && (!it.kept && !it.example || it.structure.ids) && !(it.traj && it.kept) ? it.structure : null,
      ids: it.traj && it.structure && it.structure.ids ? it.structure.ids : null })),
    layers: S.layers.map(L => ({ item: index.get(L.item), kind: L.kind, visible: L.visible, ...(L.kind === "field" ? { fi: L.fi, iso: L.iso, p: L.p, mode: L.mode, pm: L.pm, opacity: L.opacity, pos: L.pos, neg: L.neg, cuts: L.cuts } : {}) })),
    view: { style: S.style, look: S.look, scale: S.scale, bonds: S.bonds, tol: S.tol, cell: S.cell, reps: S.reps.slice(), boundary: S.boundary, res: S.res, proj: S.proj, t: S.t.slice(),
      frame: S.frame, fps: S.fps, step: S.step, loop: S.loop, active: S.layers.findIndex(L => L.id === S.active), activeAtoms: S.layers.findIndex(L => L.id === S.activeAtoms),
      labels: S.labels.map(l => ({ layer: S.layers.findIndex(L => L.id === l.layer), src: l.src, p: l.p })) },
    cam: { q: cam.q.slice(), dist: cam.dist, radius: cam.radius, target: cam.target.slice() },
  };
  if (withFiles) state.files = Object.fromEntries(items.filter(it => it.files).map(it => [it.key, it.files]));
  return state;
}
/** Write the view settings into the controls. */
function syncControls() {
  $("scale").value = S.scale; $("bonds").checked = S.bonds; $("row-tol").hidden = !S.bonds; $("tol").value = S.tol; $("tol-out").textContent = S.tol.toFixed(2);
  $("cell").checked = S.cell; ["ra", "rb", "rc"].forEach((id, i) => ($(id).value = S.reps[i])); $("boundary").checked = S.boundary;
  cam.ortho = S.proj === "ortho";
  $("tr-fps").value = S.fps; $("tr-fps-out").textContent = `${S.fps} fps`; $("mv-fps").value = S.fps; $("tr-step").value = S.step;
  segState(); syncShiftInputs();
}
/** Rebuild a scene from serialize() output; files(key) gives an item's File objects, from storage (stored: they are
 *  kept already) or from memory (New scene's Undo: they are stored again). */
async function restoreScene(st, files, stored = true) {
  if (!st || st.v !== 1 || !st.items || !st.items.length) return false;
  restoring = true; busy(true);
  const missing = [], made = [];
  try {
    if (!(await ensureRenderer())) return false;
    clearScene();
    Object.assign(S, { style: st.view.style, look: LOOKS[st.view.look] ? st.view.look : "classic", scale: st.view.scale, bonds: st.view.bonds, tol: st.view.tol, cell: st.view.cell, reps: st.view.reps.slice(), boundary: st.view.boundary,
      res: st.view.res, proj: st.view.proj, t: st.view.t.slice(), fps: st.view.fps || 12, step: st.view.step || 1, loop: st.view.loop || "loop" });
    syncControls();
    for (const sp of st.items) {
      let d = null;
      try {
        if (sp.example === "h3d") d = await call("example", { name: "h3d" });
        else if (sp.kept) { const f = await files(sp.key); if (f && f.length) d = { ...(await call("open", { files: f })), files: f }; }
        if (!d && sp.structure) d = { ds: null, structure: sp.structure, fields: [], wave: null, traj: null, notes: [] };
      } catch (e) { d = null; }
      if (!d) { missing.push(sp.name); made.push(null); continue; }
      await addItem({ ...d, name: sp.name, key: sp.key, example: sp.example, alreadyKept: stored && sp.kept }, true, true);
      const it = [...S.items.values()].pop();
      if (sp.structure && !it.traj) it.structure = sp.structure;
      if (it.traj && sp.ids) { const n = sp.ids.length, ids = sp.ids; it.structure = { ...it.structure, species: Array.from(ids, id => it.traj && d.structure.species[id - 1]), ids, pos: new Float64Array(3 * n) }; }
      if (sp.hadFiles && !sp.kept) missing.push(sp.name + " (atoms only: its files were too large to keep)");
      made.push(it);
    }
    // layers in the stored order, with their settings; layers the user had removed stay removed
    const order = [];
    for (const ls of st.layers) {
      const it = made[ls.item];
      if (!it) continue;
      const L = S.layers.find(l => l.item === it.id && l.kind === ls.kind && !order.includes(l));
      if (!L) continue;
      L.visible = ls.visible;
      if (L.kind === "field") Object.assign(L, { fi: Math.min(ls.fi, Math.max(0, it.fields.length - 1)), iso: ls.iso, p: ls.p, mode: ls.mode, pm: ls.pm, opacity: ls.opacity, pos: ls.pos, neg: ls.neg, cuts: ls.cuts || [] });
      order.push(L);
    }
    for (const L of S.layers) if (!order.includes(L) && L.kind === "field") R.removeMeshGroup(L.id);
    S.layers = order;
    const at = i => (i >= 0 && order[i] ? order[i].id : null);
    S.active = at(st.view.active) ?? (fieldLayers()[0] || {}).id ?? null; S.activeAtoms = at(st.view.activeAtoms);
    S.frame = st.view.frame || 0;
    for (const it of trajItems()) applyFrame(it);
    S.labels = (st.view.labels || []).filter(l => order[l.layer]).map(l => ({ i: -1, layer: order[l.layer].id, src: l.src, p: l.p }));
    if (!S.layers.length) { showHome(); return false; }
    for (const it of made) if (it && it.wave) {
      const sp = st.items[made.indexOf(it)], FL = S.layers.find(L => L.item === it.id && L.kind === "field");
      if (FL && sp.w && sp.w.b >= 0 && (sp.w.b !== it.w.b || sp.w.s !== it.w.s || sp.w.k !== it.w.k || sp.w.mode !== it.w.mode)) { it.w = { ...sp.w }; await showBand(FL, sp.w.b); }
    }
    S.atlas = buildAtlas(atlasSpecies());
    buildLayersUI(); updateChrome(); buildStructure(); syncFieldUI(); syncControls();
    for (const L of fieldLayers()) requestSurface(L);
    Object.assign(cam, { q: st.cam.q.slice(), dist: st.cam.dist, radius: st.cam.radius, target: st.cam.target.slice() });
    $("empty").hidden = true;
    requestRender();
    if (missing.length) toast(`Restored the last scene; not kept: ${missing.join(", ")}.`, false, 8000);
    return true;
  } catch (e) { console.error(e); toast(`Could not restore the last scene: ${e.message || e}`, true); return false; }
  finally { restoring = false; busy(false); saveSoon(); }
}
/** Clear the scene and the kept copy, back to the opening page; Undo rebuilds it from memory. */
function newScene() {
  const snap = S.layers.length ? serialize(true) : null;
  clearScene(); buildLayersUI(); updateChrome(); showHome();
  store.clearAll();
  if (snap) toast("Started a new scene.", false, 8000, { label: "Undo", fn: () => restoreScene(snap, k => (snap.files || {})[k] || null, false) });
}

/* ------------------------------------------------------------------ chrome */
function updateChrome() {
  const has = S.layers.length > 0;
  $("dock").hidden = !has; $("br").hidden = !has; $("bl").hidden = !has; $("doc").hidden = !has; $("btn-new").hidden = !has;
  for (const b of document.querySelectorAll(".corner-eye")) b.hidden = !has;
  const pc = primaryCell();
  for (const el of document.querySelectorAll(".pbc")) el.hidden = !(pc && pc.pbc);
  for (const el of document.querySelectorAll(".has-cell")) el.hidden = !pc;
  syncShiftInputs();
  for (const b of document.querySelectorAll("[data-along]")) b.hidden = !pc;
  const names = [...new Set(S.layers.map(L => itemOf(L).name))];
  $("doc-name").textContent = names.join(" + ");
  const al = atomLayers();
  $("doc-sub").textContent = al.length === 1 ? formula(itemOf(al[0]).structure.species) : al.length > 1 ? `${al.length} structures` : "";
  updateTools(); updateEdit(); updatePlayer();
  updateInfo(); summaries(); segState();
}
function updateInfo() {
  const n = atomLayers().reduce((a, L) => a + itemOf(L).structure.species.length, 0), pc = primaryCell();
  let info = n ? `${n} atoms` : "";
  if (pc) {
    const p = params(pc.L);
    info += `${info ? " · " : ""}${p.slice(0, 3).map(x => x.toFixed(3)).join(" ")} Å · ${p.slice(3).map(x => x.toFixed(2)).join(" ")}° · ${Math.abs(det3(pc.L)).toFixed(1)} Å³`;
  }
  if (pc && S.t.some(x => Math.abs(x) > 1e-9)) info += ` · shifted by ${fracShift(pc.L).map(x => +x.toFixed(4)).join(" ")}`;
  if (maxFrames() > 1) info += ` · frame ${S.frame + 1} / ${maxFrames()}`;
  $("info").textContent = info;
}
function summaries() {
  const pc = primaryCell();
  $("sum-structure").textContent = !atomLayers(false).length ? "" : S.style === "off" ? "hidden" : `${S.style}${S.look !== "classic" ? " · " + S.look : ""}${pc && pc.pbc ? " · " + S.reps.join("×") : ""}`;
  const L = activeField(), f = activeStats();
  $("sum-surface").textContent = !L || !f ? "" : !L.visible ? "hidden" : L.mode === "log" ? `${fmt(L.iso)} · ${Math.round(100 * fractionAbove(f.stats, L.iso))}%` : fmt(L.iso);
  const aw = activeWave();
  if (aw && aw.it.w.b >= 0) $("sum-orbital").textContent = `${aw.it.w.b + 1}${aw.it.wave.nspin === 2 ? (aw.it.w.s ? "↓" : "↑") : ""} · ${aw.it.wave.eig[aw.it.w.s][aw.it.w.k][aw.it.w.b].toFixed(2)} eV`;
  $("sum-view").textContent = S.proj === "ortho" ? "ortho" : "perspective";
}
function segState() {
  const aw = activeWave(), w = aw ? aw.it.w : { s: 0, mode: "density", half: "x" };
  const val = { style: S.style, look: S.look, res: S.res, wspin: String(w.s), wmode: w.mode, half: w.half, proj: S.proj, theme: S.theme, loop: S.loop, mfmt: S.movie.fmt, msize: String(S.movie.scale) };
  for (const seg of document.querySelectorAll(".seg")) for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === val[seg.dataset.key]));
}

/* ------------------------------------------------------------------ home page, examples, image */
function showHome() {
  $("empty").hidden = false;
  const back = $("btn-back");
  back.hidden = !S.layers.length;
  if (S.layers.length) back.textContent = `Back to ${$("doc-name").textContent}`;
}
/** Samples on the opening page (examples/, see examples/README.md for each file's source); h3d is computed. */
const SAMPLES = {
  rutile: "rutile-TiO2.cif", sto: "SrTiO3.vasp", zno: "ZnO-wurtzite.xsf", graphene: "graphene.pw.in", nacl: "NaCl.geometry.in", c60: "C60.xyz",
  ethanol: "ethanol.pdb", pwo: "Si.pwscf.out", si: "Si-model.CHGCAR", elf: "C4.ELFCAR.gz", locpot: "Li.LOCPOT.gz", n2: "N2.WAVECAR",
  benzene: "benzene-pi.cube", outcar: "Li-relax.OUTCAR", vasprun: "Si64-md.vasprun.xml.gz", fe2o3: "Fe2O3-molten.XDATCAR", castep: "Ar-LJ.md",
  cu55: "Cu55-900K.traj", pt13: "Pt13-relax.extxyz", al: "Al-800K.XDATCAR", ag: "Ag-1600K.lammpstrj", ni: "Ni100-adatom-1100K.axsf",
};
async function example(id) {
  busy(true);
  try {
    if (id === "h3d") {
      if (!(await ensureRenderer())) return;
      await addItem({ name: "hydrogen 3d(x²−y²), computed here", ...(await call("example", { name: "h3d" })), example: "h3d" }, false);
      return;
    }
    const name = SAMPLES[id];
    if (!name) throw new Error(`There is no sample called "${id}".`);
    const r = await fetch("examples/" + name);
    if (!r.ok) throw new Error(`Could not load the sample (${r.status}).`);
    await openFiles([new File([await r.blob()], name)], "replace");
  } catch (e) { toast(e.message || String(e), true); }
  finally { busy(false); }
}
function snapshot() {
  if (!R) return;
  frame();
  canvas.toBlob(b => {
    if (!b) return toast("Could not capture the image.", true);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = ($("doc-name").textContent.split(" + ")[0] || "isoview").replace(/[^\w.-]+/g, "_") + ".png";
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, "image/png");
}
function restyle() { if (!R) return; readPalette(); R.setBackground(pal.bg); requestRender(); }

/* ------------------------------------------------------------------ corner eyes: tuck away the group in each corner */
const CORNERS = { tl: "tl", tr: "dock", bl: "bl", br: "br" };
const CORNER_NAMES = { tl: "the title", tr: "the panel", bl: "the axes and cell", br: "the view buttons" };
const EYE = '<svg viewBox="0 0 24 24" aria-hidden="true"><g class="open"><g class="lid"><path d="M2.8 12s3.4-6 9.2-6 9.2 6 9.2 6-3.4 6-9.2 6-9.2-6-9.2-6z" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="12" cy="12" r="2.4" fill="currentColor"/></g></g>' +
  '<g class="shut" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M3.4 10.6c2.4 2.9 5.3 4.4 8.6 4.4s6.2-1.5 8.6-4.4"/><path d="m6.3 13.4-1.4 1.9M12 15v2.3M17.7 13.4l1.4 1.9"/></g></svg>';
function tuck(c, on) {
  $(CORNERS[c]).classList.toggle("tucked", on);
  const b = document.querySelector(`.corner-eye[data-corner="${c}"]`), t = `${on ? "Show" : "Hide"} ${CORNER_NAMES[c]} (H for all)`;
  b.setAttribute("aria-pressed", String(on)); b.title = t; b.setAttribute("aria-label", t);
}
/** H: hide every corner if any is shown, otherwise show them all. */
function tuckAll() { const any = Object.values(CORNERS).some(id => !$(id).classList.contains("tucked")); for (const c in CORNERS) tuck(c, any); }
for (const c in CORNERS) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "corner-eye"; b.dataset.corner = c; b.hidden = true; b.innerHTML = EYE;
  b.addEventListener("click", () => tuck(c, !$(CORNERS[c]).classList.contains("tucked")));
  document.body.append(b);
  tuck(c, false);
}

/* ------------------------------------------------------------------ wiring */
function bindSeg(key, fn) {
  document.querySelector(`.seg[data-key="${key}"]`).addEventListener("click", e => { const b = e.target.closest("button"); if (b) { fn(b.value); segState(); summaries(); } });
}
bindSeg("style", v => { S.style = v; buildStructure(); });
bindSeg("look", v => { S.look = v; requestRender(); });
bindSeg("res", v => { S.res = v; refreshSurfaces(); });
bindSeg("proj", v => { S.proj = v; cam.ortho = v === "ortho"; requestRender(); });
bindSeg("theme", v => {
  S.theme = v;
  if (v === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", v);
  try { localStorage.setItem("isoview-theme", v); } catch (e) { /* storage unavailable */ }
  restyle();
});
bindSeg("loop", v => { S.loop = v; playDir = 1; });
bindSeg("mfmt", v => { S.movie.fmt = v; if (v === "video" && S.movie.scale < 0.5) S.movie.scale = 1; movieNote(); });
bindSeg("msize", v => { S.movie.scale = +v; movieNote(); });
bindSeg("wspin", v => { const aw = activeWave(); if (!aw) return; aw.it.w.s = +v; buildBands(aw.it); showBand(aw.L, Math.max(0, aw.it.w.b)); });
bindSeg("wmode", v => { const aw = activeWave(); if (!aw) return; aw.it.w.mode = v; showBand(aw.L, Math.max(0, aw.it.w.b)); });
bindSeg("half", v => { const aw = activeWave(); if (!aw) return; aw.it.w.half = v; showBand(aw.L, Math.max(0, aw.it.w.b)); });
$("kpt").addEventListener("change", e => { const aw = activeWave(); if (!aw) return; aw.it.w.k = +e.target.value; buildBands(aw.it); showBand(aw.L, Math.max(0, aw.it.w.b)); });
$("bands").addEventListener("click", e => { const r = e.target.closest(".band"), aw = activeWave(); if (r && aw) showBand(aw.L, +r.dataset.b); });
$("bands").addEventListener("keydown", e => {
  const aw = activeWave();
  if (!aw || (e.key !== "ArrowDown" && e.key !== "ArrowUp")) return;
  e.preventDefault();
  const b = Math.min(aw.it.wave.nb - 1, Math.max(0, aw.it.w.b + (e.key === "ArrowDown" ? 1 : -1)));
  showBand(aw.L, b);
  const el = $("bands").querySelector(`[data-b="${b}"]`); if (el) el.scrollIntoView({ block: "nearest" });
});
$("scale").addEventListener("input", e => { S.scale = +e.target.value; buildStructure(); });
$("bonds").addEventListener("change", e => { S.bonds = e.target.checked; $("row-tol").hidden = !S.bonds; buildStructure(); });
$("tol").addEventListener("input", e => { S.tol = +e.target.value; $("tol-out").textContent = S.tol.toFixed(2); buildStructure(); });
$("cell").addEventListener("change", e => { S.cell = e.target.checked; buildStructure(); });
$("boundary").addEventListener("change", e => { S.boundary = e.target.checked; buildStructure(); });
for (const [i, id] of ["ra", "rb", "rc"].entries()) $(id).addEventListener("change", e => {
  S.reps[i] = Math.min(6, Math.max(1, Math.round(+e.target.value) || 1)); e.target.value = S.reps[i];
  buildStructure(); fit(); refreshSurfaces(); summaries();
});
for (const [i, id] of ["fx", "fy", "fz"].entries()) $(id).addEventListener("change", e => {
  const pc = primaryCell(), v = parseFloat(e.target.value);
  if (!Number.isFinite(v)) { syncShiftInputs(); return; }
  if (!pc) { const t = S.t.slice(); t[i] = v; setShift(t); return; }
  const f = fracShift(pc.L); f[i] = v; setShift(fracToCart(pc.L, f));
});
$("shift-reset").addEventListener("click", () => setShift([0, 0, 0]));
$("t-prim").addEventListener("click", () => derive("primitive"));
$("t-conv").addEventListener("click", () => derive("conventional"));
for (const b of document.querySelectorAll("[data-fmt]")) b.addEventListener("click", () => exportLayer(b.dataset.fmt));
$("field").addEventListener("change", e => { const L = activeField(); if (!L) return; L.fi = +e.target.value; initLevel(L, true); syncFieldUI(); buildLayersUI(); requestSurface(L); });
$("surf").addEventListener("change", e => { const L = activeField(); if (!L) return; L.visible = e.target.checked; requestSurface(L); buildLayersUI(); summaries(); });
$("pm").addEventListener("change", e => { const L = activeField(); if (!L) return; L.pm = e.target.checked; requestSurface(L); });
$("opacity").addEventListener("input", e => { const L = activeField(); if (!L) return; L.opacity = +e.target.value; restyleLayer(L); });
$("iso").addEventListener("input", () => setLevelFromSlider(true));
$("iso").addEventListener("change", () => setLevelFromSlider(false));
$("iso-num").addEventListener("change", e => {
  const L = activeField(), f = activeStats();
  if (!L || !f) return;
  const v = parseFloat(e.target.value.replace("×10", "e"));
  if (!Number.isFinite(v)) return syncFieldUI();
  L.iso = L.mode === "log" ? Math.abs(v) : v;
  if (L.mode === "log") L.p = fractionAbove(f.stats, L.iso);
  syncFieldUI(); requestSurface(L);
});
$("cuts-clear").addEventListener("click", () => {
  const cut = fieldLayers().filter(L => L.cuts.length);
  if (!cut.length) return;
  const list = cut.map(L => ({ layer: L.id, cuts: L.cuts }));
  for (const L of cut) { L.cuts = []; requestSurface(L); }
  pushUndo({ type: "cuts", list, label: "restore the surface" });
});
for (const b of document.querySelectorAll("[data-along]")) b.addEventListener("click", () => viewAlong(+b.dataset.along));
$("btn-reset").addEventListener("click", resetView);
$("btn-snap").addEventListener("click", snapshot);
$("btn-home").addEventListener("click", showHome);
$("btn-label").addEventListener("click", () => setLabelMode(!S.labelMode));
$("e-region").addEventListener("click", () => setBoxMode(!S.boxMode));
$("measure").addEventListener("click", e => {
  const b = e.target.closest("[data-act]");
  if (!b) return;
  const a = b.dataset.act;
  if (a === "center") centerOnSelected();
  else if (a === "delete") deleteSelected();
  else if (a === "cut") cutSurface();
  else if (a === "edit") { tuck("tr", false); $("sec-edit").open = true; $("sec-edit").scrollIntoView({ block: "nearest" }); }
});
$("pl-play").addEventListener("click", () => play(!S.playing));
$("pl-frame").addEventListener("input", e => setFrame(+e.target.value));
$("tr-fps").addEventListener("input", e => { S.fps = +e.target.value; $("tr-fps-out").textContent = `${S.fps} fps`; $("mv-fps").value = S.fps; movieNote(); });
$("tr-step").addEventListener("change", e => { S.step = Math.max(1, Math.round(+e.target.value) || 1); e.target.value = S.step; $("tr-step-u").textContent = S.step > 1 ? `${["", "", "2nd", "3rd"][S.step] || S.step + "th"} frame` : "frame"; movieNote(); });
for (const id of ["mv-fps", "mv-from", "mv-to"]) $(id).addEventListener("change", movieNote);
$("mv-go").addEventListener("click", exportMovie);
$("tr-xyz").addEventListener("click", exportFrames);
$("tr-svg").addEventListener("click", e => { if (mainTraj()) setFrame(sparkFrame(e)); });
$("sec-traj").addEventListener("toggle", () => { if ($("sec-traj").open) drawSpark(); });
$("e-delete").addEventListener("click", deleteSelected);
$("e-cut").addEventListener("click", cutSurface);
$("e-center").addEventListener("click", centerOnSelected);
$("e-undo").addEventListener("click", undo);
$("sel-clear2").addEventListener("click", clearSelection);
$("label-done").addEventListener("click", () => setLabelMode(false));
$("label-clear").addEventListener("click", () => { S.labels = []; uploadLabels(); requestRender(); });
$("btn-back").addEventListener("click", () => { $("empty").hidden = true; });
$("btn-new").addEventListener("click", newScene);
let fileMode = "replace";
const pick = mode => { fileMode = mode; $("file").click(); };
$("btn-open2").addEventListener("click", () => pick("replace"));
$("btn-open").addEventListener("click", () => pick("replace"));
$("btn-add").addEventListener("click", () => pick("add"));
$("file").addEventListener("change", e => { openFiles(e.target.files, fileMode); e.target.value = ""; });
for (const b of document.querySelectorAll("[data-example]")) b.addEventListener("click", () => example(b.dataset.example));

/* drag and drop: with a scene loaded, the left half replaces it and the right half adds to it */
let dragDepth = 0;
const dropMode = e => (S.layers.length && $("empty").hidden && e.clientX > window.innerWidth / 2 ? "add" : "replace");
window.addEventListener("dragenter", e => {
  if (![...((e.dataTransfer && e.dataTransfer.types) || [])].includes("Files")) return;
  dragDepth++; e.preventDefault();
  $("drop").hidden = false; $("drop").classList.toggle("two", !!(S.layers.length && $("empty").hidden));
});
window.addEventListener("dragover", e => { e.preventDefault(); const m = dropMode(e); for (const z of document.querySelectorAll("#drop .zone")) z.classList.toggle("on", z.dataset.mode === m); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").hidden = true; } });
window.addEventListener("drop", e => {
  e.preventDefault(); dragDepth = 0; $("drop").hidden = true;
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) openFiles(e.dataTransfer.files, dropMode(e));
});
window.addEventListener("paste", e => {
  if (e.target && e.target.closest && e.target.closest("input, select, textarea")) return;
  const t = e.clipboardData && e.clipboardData.getData("text");
  if (t && t.trim()) openFiles([new File([t], "pasted")], S.layers.length ? "add" : "replace");
});
window.addEventListener("keydown", e => {
  if (e.target && e.target.closest && e.target.closest("input, select, textarea, .bands")) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z" && !e.shiftKey) { e.preventDefault(); undo(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (k === "o") pick("replace");
  else if (k === "escape") {
    if (!$("empty").hidden && S.layers.length) $("empty").hidden = true;
    else if (S.labelMode) setLabelMode(false);
    else if (S.sel.length || S.region) clearSelection();
    else if (S.boxMode) setBoxMode(false);
    else for (const d of document.querySelectorAll(".sec[open]")) d.open = false;
  }
  else if (!S.layers.length || !$("empty").hidden) return;
  else if ((k === "delete" || k === "backspace") && S.sel.length) { e.preventDefault(); deleteSelected(); }
  else if (k === "r") resetView();
  else if (k === "s") snapshot();
  else if (k === "l") setLabelMode(!S.labelMode);
  else if (k === "h") tuckAll();
  else if (k === " " && maxFrames() > 1) { e.preventDefault(); play(!S.playing); }
  else if ((k === "arrowright" || k === "arrowleft") && maxFrames() > 1) { e.preventDefault(); play(false); setFrame(S.frame + (k === "arrowright" ? 1 : -1) * S.step * (e.shiftKey ? 10 : 1)); }
  else if ((k === "home" || k === "end") && maxFrames() > 1) { e.preventDefault(); play(false); setFrame(k === "home" ? 0 : maxFrames() - 1); }
  else if (k.length === 1 && "abc".includes(k)) viewAlong("abc".indexOf(k));
});
try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (S.theme === "auto") restyle(); }); } catch (e) { /* old browsers */ }
try { const t = localStorage.getItem("isoview-theme"); if (t === "light" || t === "dark") { S.theme = t; document.documentElement.setAttribute("data-theme", t); } } catch (e) { /* storage unavailable */ }

startWorker();
segState();
// test hook (tests set globalThis.__isoviewTest before importing this module)
if (globalThis.__isoviewTest) globalThis.__isoview = { S, cam, get disp() { return disp; } };
if (qp.get("url")) {
  const u = qp.get("url");
  fetch(u).then(r => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.blob(); })
    .then(b => openFiles([new File([b], decodeURIComponent(u.split(/[?#]/)[0].split("/").pop() || "file"))], "replace"))
    .catch(e => toast(`Could not fetch ${u}: ${e.message}. The server must allow cross-origin requests.`, true, 9000));
} else if (qp.get("example")) example(qp.get("example"));
else store.getState().then(st => { if (st && !S.layers.length) restoreScene(st, store.getFiles); });
