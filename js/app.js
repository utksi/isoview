/**
 * isoview application: layers, UI state, camera control, picking, measurement, editing, and the hand-off to the
 * renderer (gfx/renderer.js: WebGPU or WebGL2) and the compute worker (worker.js: parsing, WAVECAR orbitals, meshing).
 *
 * Scene: every loaded file group is an item (worker dataset ds, structure, fields, WAVECAR header); items give layers,
 * an atoms layer (structure) and a field layer (one shown field, its own level, opacity and colours). Layers can be
 * added to a scene ("Add to scene"), hidden and removed.
 * Origin shift (fx, fy, fz): applied to every layer in its own cell's fractional coordinates (structure.js for atoms,
 * field.js for fields, exact for periodic grids), so atoms and densities move together.
 * Selection: click atoms (toggle) or drag a box with Shift (or the box tool); 2-4 atoms give distances, angles at the
 * middle atoms and the dihedral (measure.js). A box also records a cut region (the view frustum through the box,
 * field.js) that "Cut surface" removes from the visible surfaces. Deleting removes the atom from its structure (all
 * periodic images). Deletions, cuts, shifts and layer removals can be undone.
 * Level slider: log10 |iso| over six decades below max |f| (densities, signed fields; signed fields get +iso and -iso
 * surfaces), linear over [min, max] (potentials, ELF); the readout is the share of sum |f| inside the surface.
 * Frames are drawn on demand. Label mode (L) puts element symbols on clicked atoms (gfx/labels.js).
 */
import { params, formula, det3, norm, cross, dot, inv3, cartToFrac, fracToCart } from "./lattice.js";
import { fractionAbove, isoForFraction, autoStride } from "./field.js";
import { displayAtoms, bondList, atomId, deleteAtoms, toPOSCAR } from "./structure.js";
import { colorOf, rcov, IONIC_CATIONS } from "./elements.js";
import { Camera, qfromRows, qmat, pickSphere } from "./gfx/camera.js";
import { createRenderer, packAtoms, packBonds, packLines, packUniforms, hexToRgb01 } from "./gfx/renderer.js";
import { buildAtlas, packLabels } from "./gfx/labels.js";
import { measureAll } from "./measure.js";

const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const qp = new URLSearchParams(location.search);
// colour pairs (positive, negative) for successive field layers: mid-tones readable on light and dark backgrounds
const FIELD_COLORS = [["#e3a03a", "#3b86d6"], ["#2bb3a3", "#e0607e"], ["#9b7be0", "#7cb342"], ["#ef7d4b", "#5c9bd6"]];

const S = {
  items: new Map(), layers: [], active: null, shift: [0, 0, 0],
  style: "ball", scale: 1, bonds: true, tol: 1.15, ionic: false, cell: true, reps: [1, 1, 1], boundary: true, res: "auto",
  proj: "persp", theme: "auto", sel: [], region: null, labels: [], labelMode: false, boxMode: false, undo: [], atlas: null,
};
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
const primaryCell = () => { for (const L of S.layers) { const c = cellOf(itemOf(L)); if (c) return { L: c, pbc: pbcOf(itemOf(L)) }; } return null; };

/* ------------------------------------------------------------------ worker */
let worker = null, reqId = 0;
const pending = new Map();
function startWorker() {
  worker = globalThis.__isoviewWorker ? new globalThis.__isoviewWorker() : new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
function requestRender() { if (!dirty) { dirty = true; requestAnimationFrame(frame); } }
function frame() {
  dirty = false;
  if (!R) return;
  const dpr = canvas.width / cam.w;
  R.draw(packUniforms(cam.view(), cam.proj(R.name === "WebGPU" ? "gpu" : "gl"), [canvas.width, canvas.height, dpr, cam.ortho ? 1 : 0],
    [0.35, 0.55, 1.0, 0.32], [...pal.accent, 0.33], [...pal.cell, 1.3 * dpr]));
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
    const da = displayAtoms(st, { reps, boundary: S.boundary && st.pbc, shift: S.shift });
    for (const a of da) { a.layer = L.id; a.r = radiusOf(a.el); a.color = colorOf(a.el); }
    if (S.bonds && S.style !== "space") {
      const br = (S.style === "stick" ? 0.14 : 0.1) * S.scale;
      for (const [i, j] of bondList(da, { tol: S.tol, skip: S.ionic ? new Set() : IONIC_CATIONS })) {
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
      const it = itemOf(L), C = cellOf(it);
      if (C && !cells.some(c => sameCell(c.L, C))) cells.push({ L: C, pbc: pbcOf(it) });
    }
    for (const { L, pbc } of cells) {
      const reps = pbc ? S.reps : [1, 1, 1];
      const box = (A, B, Cc, dashed) => {
        const P = (i, j, k) => [0, 1, 2].map(c => i * A[c] + j * B[c] + k * Cc[c]);
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
    const { meshes } = await call("mesh", { ds: it.ds, fi: L.fi, reps, stride, isos: signed ? [L.iso, -L.iso] : [L.iso], shift: S.shift, cuts: L.cuts });
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
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) add([0, 1, 2].map(c => i * r[0] * C[0][c] + j * r[1] * C[1][c] + k * r[2] * C[2][c]));
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
/** Box selection in CSS pixels: add the atoms whose centres project inside, record the region for cutting. */
function boxSelect(x0, y0, x1, y1, add) {
  const [ax, bx] = [Math.min(x0, x1), Math.max(x0, x1)], [ay, by] = [Math.min(y0, y1), Math.max(y0, y1)];
  if (!add) S.sel = [];
  const have = new Set(S.sel.map(s => s.i));
  disp.atoms.forEach((a, i) => {
    if (have.has(i)) return;
    const s = cam.project([a.x, a.y, a.z]);
    if (s && s.x >= ax && s.x <= bx && s.y >= ay && s.y <= by) S.sel.push({ i, layer: a.layer, src: a.src, p: [a.x, a.y, a.z] });
  });
  const P = cam.proj("gl"), V = cam.view(), m = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let v = 0; for (let k = 0; k < 4; k++) v += P[4 * k + r] * V[4 * c + k]; m[4 * c + r] = v; }
  const nx = x => 2 * x / cam.w - 1, ny = y => 1 - 2 * y / cam.h;
  S.region = { m: Array.from(m), r: [nx(ax), ny(by), nx(bx), ny(ay)] };
  uploadAtoms(); updateSelectionBar(); requestRender();
}
function clearSelection() {
  if (!S.sel.length && !S.region) return;
  S.sel = []; S.region = null; uploadAtoms(); updateSelectionBar(); requestRender();
}
function updateSelectionBar() {
  const box = $("measure");
  if (!S.sel.length && !S.region) { box.hidden = true; box.innerHTML = ""; return; }
  const parts = [];
  if (S.sel.length && S.sel.length <= 4) {
    const m = measureAll(S.sel.map(s => s.p));
    parts.push(`<span class="atoms">${S.sel.map(s => esc(atomLabel(s))).join(" · ")}</span>`);
    if (S.sel.length === 1) parts.push(`<span class="mv">${S.sel[0].p.map(v => v.toFixed(3)).join("  ")} Å</span>`);
    m.bonds.forEach((d, i) => parts.push(`<span class="mk">${esc(atomLabel(S.sel[i]))}–${esc(atomLabel(S.sel[i + 1]))}</span><span class="mv">${d.toFixed(3)} Å</span>`));
    m.angles.forEach((a, i) => parts.push(`<span class="mk">∠${esc(atomLabel(S.sel[i + 1]))}</span><span class="mv">${a.toFixed(2)}°</span>`));
    if (m.dihedral !== null) parts.push(`<span class="mk">dihedral</span><span class="mv">${m.dihedral.toFixed(2)}°</span>`);
  } else parts.push(`<span class="atoms">${S.sel.length ? `${S.sel.length} atoms selected` : "Region selected"}</span>`);
  const btn = (id, label, cls = "") => `<button type="button" id="${id}" class="${cls}">${label}</button>`;
  if (S.sel.length === 1 && primaryCell()) parts.push(btn("act-center", "Center"));
  if (S.sel.length) parts.push(btn("act-delete", "Delete", "warn"));
  if (S.region && fieldLayers(true).some(L => L.mesh)) parts.push(btn("act-cut", "Cut surface", "warn"));
  parts.push(`<button type="button" class="x" id="sel-clear" aria-label="Clear the selection">×</button>`);
  box.innerHTML = parts.join("");
  box.hidden = false;
  $("sel-clear").addEventListener("click", clearSelection);
  if ($("act-center")) $("act-center").addEventListener("click", centerOnSelected);
  if ($("act-delete")) $("act-delete").addEventListener("click", deleteSelected);
  if ($("act-cut")) $("act-cut").addEventListener("click", cutSurface);
}
function drawOverlay() {
  const ov = $("ov");
  let h = "";
  if (dragBox) {
    const x = Math.min(dragBox.x0, dragBox.x1), y = Math.min(dragBox.y0, dragBox.y1);
    h += `<rect class="box" x="${x}" y="${y}" width="${Math.abs(dragBox.x1 - dragBox.x0)}" height="${Math.abs(dragBox.y1 - dragBox.y0)}"/>`;
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
  pushUndo({ type: "atoms", changes, labels });
  S.sel = []; S.region = null;
  buildStructure();
  toast(`Deleted ${n} atom${n > 1 ? "s" : ""} (in every periodic image).`, false, 6000, { label: "Undo", fn: undo });
}
function cutSurface() {
  if (!S.region) return;
  const targets = fieldLayers(true).filter(L => L.mesh);
  for (const L of targets) { L.cuts.push(S.region); requestSurface(L); }
  pushUndo({ type: "cut", layers: targets.map(L => L.id) });
  S.sel = []; S.region = null; uploadAtoms(); updateSelectionBar();
  toast("Cut the selected region out of the surface.", false, 6000, { label: "Undo", fn: undo });
}
function centerOnSelected() {
  const s = S.sel[0], it = itemOf(layerById(s.layer)), C = cellOf(it) || (primaryCell() || {}).L;
  if (!C) return;
  const f = cartToFrac(inv3(C), s.p);
  // shifts are periodic: keep each component in (-0.5, 0.5]
  setShift(S.shift.map((v, a) => { const x = v + 0.5 - f[a]; return +(x - Math.ceil(x - 0.5)).toFixed(6); }));
}
/** Change the origin shift; selections and labels follow their atoms. */
function setShift(v, record = true) {
  const old = S.shift.slice();
  if (v.every((x, a) => Math.abs(x - old[a]) < 1e-12)) return;
  if (record) pushUndo({ type: "shift", old });
  S.shift = v.slice();
  for (const e of [...S.sel, ...S.labels]) {
    const C = cellOf(itemOf(layerById(e.layer)));
    if (C) { const d = fracToCart(C, v.map((x, a) => x - old[a])); e.p = e.p.map((x, c) => x + d[c]); }
  }
  ["fx", "fy", "fz"].forEach((id, a) => ($(id).value = +S.shift[a].toFixed(4)));
  buildStructure(); refreshSurfaces(); summaries();
}
function pushUndo(u) { S.undo.push(u); if (S.undo.length > 50) S.undo.shift(); }
function undo() {
  const u = S.undo.pop();
  if (!u) return toast("Nothing to undo.");
  if (u.type === "atoms") { for (const c of u.changes) c.it.structure = c.old; S.labels = u.labels; S.sel = []; buildStructure(); }
  else if (u.type === "cut") for (const id of u.layers) { const L = layerById(id); if (L) { L.cuts.pop(); requestSurface(L); } }
  else if (u.type === "cuts") { const L = layerById(u.layer); if (L) { L.cuts = u.cuts; requestSurface(L); levelHint(); } }
  else if (u.type === "shift") setShift(u.old, false);
  else if (u.type === "layer") {
    S.layers.splice(Math.min(u.index, S.layers.length), 0, u.layer);
    if (u.layer.kind === "atoms") buildStructure(); else { S.active = u.layer.id; requestSurface(u.layer); syncFieldUI(); buildStructure(); }
    $("empty").hidden = true;
    buildLayersUI(); updateChrome(); refit();
  }
  toast("Undone.", false, 2500);
}
function exportPOSCAR() {
  const L = atomLayers()[0] || atomLayers(false)[0];
  if (!L) return toast("There are no atoms to export.", true);
  const it = itemOf(L), txt = toPOSCAR(it.structure, S.shift, `${it.name} (isoview${S.shift.some(Boolean) ? `, shifted by ${S.shift.join(" ")}` : ""})`);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain" }));
  a.download = it.name.split(" + ")[0].replace(/\.[^.]*$/, "").replace(/[^\w.-]+/g, "_") + "_isoview.vasp";
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
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
  $("btn-box").setAttribute("aria-pressed", String(on));
  $("stage").classList.toggle("boxing", on);
}
function labelBar() {
  $("labelbar").hidden = !S.labelMode;
  if (S.labelMode) $("label-count").textContent = S.labels.length ? `${S.labels.length} labelled` : "Click atoms to label them";
}

/* ------------------------------------------------------------------ pointer */
const stage = $("stage"), ptrs = new Map();
let drag = null, dragBox = null;
stage.addEventListener("pointerdown", e => {
  if (!R || !S.layers.length) return;
  if (stage.setPointerCapture) stage.setPointerCapture(e.pointerId);
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (ptrs.size === 1) {
    const box = e.button === 0 && (e.shiftKey || S.boxMode) && !S.labelMode;
    drag = { x0: e.clientX, y0: e.clientY, moved: false, box, add: e.shiftKey && S.sel.length > 0, pan: !box && (e.button !== 0 || e.ctrlKey || e.metaKey), button: e.button };
  } else if (drag) { drag.moved = true; drag.box = false; dragBox = null; }
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
      if (drag.box) dragBox = { x0: drag.x0, y0: drag.y0, x1: cur.x, y1: cur.y };
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
    if (drag.box && drag.moved && dragBox) boxSelect(dragBox.x0, dragBox.y0, dragBox.x1, dragBox.y1, drag.add);
    else if (!drag.moved && drag.button === 0) clickAt(e.clientX, e.clientY);
  }
  if (ptrs.size === 0) { drag = null; if (dragBox) { dragBox = null; requestRender(); } }
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
  $("stats").textContent = `${f.n.join(" × ")} grid · min ${fmt(st.min)} · max ${fmt(st.max)} · ∫ ${fi}${f.unit === "e/Å³" ? " e" : ""}${tri}`;
  $("row-cuts").hidden = !L.cuts.length;
  $("cuts-n").textContent = `${L.cuts.length} cut${L.cuts.length > 1 ? "s" : ""}`;
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
  const box = $("layers");
  box.innerHTML = S.layers.map(L => {
    const it = itemOf(L), f = L.kind === "field" ? it.fields[L.fi] : null;
    const kind = L.kind === "atoms" ? `${it.structure.species.length} atoms` : f ? f.name : it.wave ? "orbital" : "field";
    const signed = f && f.kind === "signed";
    const sw = L.kind === "field" ? `<input type="color" class="cpos" value="${L.pos}" aria-label="Surface colour">${signed ? `<input type="color" class="cneg" value="${L.neg}" aria-label="Negative surface colour">` : ""}` : "";
    return `<div class="layer${L.id === S.active ? " active" : ""}${L.visible ? "" : " off"}" data-id="${L.id}">` +
      `<button type="button" class="eye" aria-pressed="${L.visible}" aria-label="Show or hide this layer"><svg viewBox="0 0 24 24"><path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="2.6" fill="currentColor"/></svg></button>` +
      `<span class="lname" title="${esc(it.name)}"><span class="lk">${esc(kind)}</span>${esc(it.name)}</span>${sw}` +
      `<button type="button" class="rm" aria-label="Remove this layer">×</button></div>`;
  }).join("");
  $("sum-layers").textContent = `${S.layers.length} layer${S.layers.length === 1 ? "" : "s"}`;
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
  pushUndo({ type: "layer", layer: L, index });
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
    await addItem({ name: files.map(f => f.name).join(" + "), ...d }, add);
  } catch (e) { toast(e.message || String(e), true, 8000); console.error(e); }
  finally { busy(false); }
}
function clearScene() {
  for (const it of S.items.values()) call("drop", { ds: it.ds }).catch(() => {});
  S.items.clear(); S.layers = []; S.active = null; S.sel = []; S.region = null; S.labels = []; S.undo = [];
  S.shift = [0, 0, 0]; ["fx", "fy", "fz"].forEach(id => ($(id).value = 0));
  S.reps = [1, 1, 1]; ["ra", "rb", "rc"].forEach(id => ($(id).value = 1));
  if (R) R.clearMeshes();
  setLabelMode(false); setBoxMode(false);
}
function sameAtoms(a, b) {
  if (!a || !b || a.species.length !== b.species.length || !sameCell(a.lattice, b.lattice)) return false;
  for (let i = 0; i < a.pos.length; i++) if (Math.abs(a.pos[i] - b.pos[i]) > 1e-3) return false;
  return a.species.every((s, i) => s === b.species[i]);
}
async function addItem(d, adding) {
  if (!(await ensureRenderer())) return;
  if (!adding) clearScene();
  const notes = [...(d.notes || [])];
  const it = { id: nextItem++, name: d.name, ds: d.ds, structure: d.structure, wave: d.wave || null, fields: d.fields || [], w: { s: 0, k: 0, b: -1, mode: "density", half: "x" } };
  if (it.wave && !(it.structure && sameCell(it.structure.lattice, it.wave.lattice))) {
    if (it.structure) notes.push("the structure's cell differs from the WAVECAR's, so only the cell is drawn");
    else if (!atomLayers(false).length) notes.push("add the matching POSCAR or CONTCAR to draw the atoms");
    it.structure = null;
  }
  if (it.structure && !it.structure.species.length && !it.fields.length && !it.wave) throw new Error("No atoms or data found in that file.");
  S.items.set(it.id, it);
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
  buildLayersUI(); updateChrome(); buildStructure();
  if (!adding) resetView(); else refit();
  for (const id of ["sec-layers", "sec-structure", "sec-surface", "sec-orbital", "sec-view"]) $(id).open = false;
  if (adding) $("sec-layers").open = true;
  syncFieldUI();
  if (FL && it.wave) { if (!adding) $("sec-orbital").open = true; await showBand(FL, Math.max(0, it.wave.frontier[0][0].homo)); }
  else if (FL) { if (!adding) $("sec-surface").open = true; requestSurface(FL); }
  if (notes.length) toast(notes.join("; "), false, 7000);
}

/* ------------------------------------------------------------------ chrome */
function updateChrome() {
  const has = S.layers.length > 0;
  $("dock").hidden = !has; $("br").hidden = !has; $("bl").hidden = !has; $("doc").hidden = !has;
  const pc = primaryCell();
  for (const el of document.querySelectorAll(".pbc")) el.hidden = !(pc && pc.pbc);
  for (const b of document.querySelectorAll("[data-along]")) b.hidden = !pc;
  const names = [...new Set(S.layers.map(L => itemOf(L).name))];
  $("doc-name").textContent = names.join(" + ");
  const al = atomLayers();
  $("doc-sub").textContent = al.length === 1 ? formula(itemOf(al[0]).structure.species) : al.length > 1 ? `${al.length} structures` : "";
  $("row-export").hidden = !atomLayers(false).length;
  updateInfo(); summaries(); segState();
}
function updateInfo() {
  const n = atomLayers().reduce((a, L) => a + itemOf(L).structure.species.length, 0), pc = primaryCell();
  let info = n ? `${n} atoms` : "";
  if (pc) {
    const p = params(pc.L);
    info += `${info ? " · " : ""}${p.slice(0, 3).map(x => x.toFixed(3)).join(" ")} Å · ${p.slice(3).map(x => x.toFixed(2)).join(" ")}° · ${Math.abs(det3(pc.L)).toFixed(1)} Å³`;
  }
  if (S.shift.some(Boolean)) info += ` · shifted by ${S.shift.map(x => +x.toFixed(4)).join(" ")}`;
  $("info").textContent = info;
}
function summaries() {
  const pc = primaryCell();
  $("sum-structure").textContent = !atomLayers(false).length ? "" : S.style === "off" ? "hidden" : `${S.style}${pc && pc.pbc ? " · " + S.reps.join("×") : ""}`;
  const L = activeField(), f = activeStats();
  $("sum-surface").textContent = !L || !f ? "" : !L.visible ? "hidden" : L.mode === "log" ? `${fmt(L.iso)} · ${Math.round(100 * fractionAbove(f.stats, L.iso))}%` : fmt(L.iso);
  const aw = activeWave();
  if (aw && aw.it.w.b >= 0) $("sum-orbital").textContent = `${aw.it.w.b + 1}${aw.it.wave.nspin === 2 ? (aw.it.w.s ? "↓" : "↑") : ""} · ${aw.it.wave.eig[aw.it.w.s][aw.it.w.k][aw.it.w.b].toFixed(2)} eV`;
  $("sum-view").textContent = S.proj === "ortho" ? "ortho" : "perspective";
}
function segState() {
  const aw = activeWave(), w = aw ? aw.it.w : { s: 0, mode: "density", half: "x" };
  const val = { style: S.style, res: S.res, wspin: String(w.s), wmode: w.mode, half: w.half, proj: S.proj, theme: S.theme };
  for (const seg of document.querySelectorAll(".seg")) for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === val[seg.dataset.key]));
}

/* ------------------------------------------------------------------ home page, examples, image */
function showHome() {
  $("empty").hidden = false;
  const back = $("btn-back");
  back.hidden = !S.layers.length;
  if (S.layers.length) back.textContent = `Back to ${$("doc-name").textContent}`;
}
async function example(id) {
  busy(true);
  try {
    if (id === "h3d") {
      if (!(await ensureRenderer())) return;
      await addItem({ name: "hydrogen 3d(x²−y²), computed here", ...(await call("example", { name: "h3d" })) }, false);
      return;
    }
    const url = { rutile: "examples/rutile-TiO2.cif", graphene: "examples/graphene.pw.in" }[id];
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load the example (${r.status}).`);
    await openFiles([new File([await r.blob()], url.split("/").pop())], "replace");
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

/* ------------------------------------------------------------------ wiring */
function bindSeg(key, fn) {
  document.querySelector(`.seg[data-key="${key}"]`).addEventListener("click", e => { const b = e.target.closest("button"); if (b) { fn(b.value); segState(); summaries(); } });
}
bindSeg("style", v => { S.style = v; buildStructure(); });
bindSeg("res", v => { S.res = v; refreshSurfaces(); });
bindSeg("proj", v => { S.proj = v; cam.ortho = v === "ortho"; requestRender(); });
bindSeg("theme", v => {
  S.theme = v;
  if (v === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", v);
  try { localStorage.setItem("isoview-theme", v); } catch (e) { /* storage unavailable */ }
  restyle();
});
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
$("bonds").addEventListener("change", e => { S.bonds = e.target.checked; $("row-tol").hidden = $("row-ionic").hidden = !S.bonds; buildStructure(); });
$("tol").addEventListener("input", e => { S.tol = +e.target.value; $("tol-out").textContent = S.tol.toFixed(2); buildStructure(); });
$("ionic").addEventListener("change", e => { S.ionic = e.target.checked; buildStructure(); });
$("cell").addEventListener("change", e => { S.cell = e.target.checked; buildStructure(); });
$("boundary").addEventListener("change", e => { S.boundary = e.target.checked; buildStructure(); });
for (const [i, id] of ["ra", "rb", "rc"].entries()) $(id).addEventListener("change", e => {
  S.reps[i] = Math.min(6, Math.max(1, Math.round(+e.target.value) || 1)); e.target.value = S.reps[i];
  buildStructure(); fit(); refreshSurfaces(); summaries();
});
for (const [i, id] of ["fx", "fy", "fz"].entries()) $(id).addEventListener("change", e => {
  const v = parseFloat(e.target.value);
  if (!Number.isFinite(v)) { e.target.value = S.shift[i]; return; }
  const s = S.shift.slice(); s[i] = v; setShift(s);
});
$("shift-reset").addEventListener("click", () => setShift([0, 0, 0]));
$("btn-export").addEventListener("click", exportPOSCAR);
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
$("cuts-clear").addEventListener("click", () => { const L = activeField(); if (!L || !L.cuts.length) return; pushUndo({ type: "cuts", layer: L.id, cuts: L.cuts }); L.cuts = []; requestSurface(L); levelHint(); });
for (const b of document.querySelectorAll("[data-along]")) b.addEventListener("click", () => viewAlong(+b.dataset.along));
$("btn-reset").addEventListener("click", resetView);
$("btn-snap").addEventListener("click", snapshot);
$("btn-home").addEventListener("click", showHome);
$("btn-label").addEventListener("click", () => setLabelMode(!S.labelMode));
$("btn-box").addEventListener("click", () => setBoxMode(!S.boxMode));
$("label-done").addEventListener("click", () => setLabelMode(false));
$("label-clear").addEventListener("click", () => { S.labels = []; uploadLabels(); requestRender(); });
$("btn-back").addEventListener("click", () => { $("empty").hidden = true; });
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
  if (e.target && e.target.closest && e.target.closest("input, select, textarea")) return;
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
