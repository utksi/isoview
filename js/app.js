/**
 * isoview application: UI state, camera control, picking and measurement, and the hand-off to the renderer
 * (gfx/renderer.js: WebGPU or WebGL2) and the compute worker (worker.js: parsing, WAVECAR orbitals, marching cubes).
 *
 * Frames are drawn on demand (one requestAnimationFrame per change). Atoms are picked by casting the pointer ray
 * against the drawn spheres (gfx/camera.js pickSphere); 2, 3 and 4 selected atoms give distances, the angle at each
 * middle atom and the dihedral (measure.js). Level slider: log10 |iso| over six decades below max |f| (densities,
 * signed fields; signed fields get +iso and -iso surfaces), linear over [min, max] (potentials, ELF); the readout is
 * the share of sum |f| inside the surface. Views along a lattice vector use the rotation rows [right, up, toward].
 * Label mode (L): clicks put the element symbol on atoms or take it off; labels are drawn by the renderer on the front
 * of each atom (gfx/labels.js) and stay until cleared, also after the mode is switched off.
 */
import { params, formula, det3, norm, cross, dot } from "./lattice.js";
import { fractionAbove, isoForFraction, autoStride } from "./field.js";
import { displayAtoms, bondList } from "./structure.js";
import { colorOf, rcov, IONIC_CATIONS } from "./elements.js";
import { Camera, qfromRows, qmat, pickSphere } from "./gfx/camera.js";
import { createRenderer, packAtoms, packBonds, packLines, packUniforms, hexToRgb01 } from "./gfx/renderer.js";
import { measureAll } from "./measure.js";
import { buildAtlas, packLabels } from "./gfx/labels.js";

const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const qp = new URLSearchParams(location.search);

const S = {
  name: "", structure: null, fields: [], wave: null, fi: 0, st: null,
  iso: 0, p: 0.5, mode: "log", pm: true, surf: true, opacity: 0.85, res: "auto",
  style: "ball", scale: 1, bonds: true, tol: 1.15, ionic: false, cell: true, reps: [1, 1, 1], boundary: true,
  proj: "persp", theme: "auto", sel: [], mesh: null, labelMode: false, labels: [], atlas: null,
  w: { s: 0, k: 0, b: -1, mode: "density", half: "x" },
};
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
function toast(msg, err = false, ms = 5200) {
  const t = $("toast");
  t.textContent = msg; t.classList.toggle("err", err); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), ms);
}
const busy = on => ($("busy").hidden = !on);
const sameCell = (A, B) => A && B && A.every((r, i) => r.every((x, j) => Math.abs(x - B[i][j]) < 2e-3));
function readPalette() {
  pal = { bg: hexToRgb01(css("--bg")), pos: hexToRgb01(css("--pos")), neg: hexToRgb01(css("--neg")), cell: hexToRgb01(css("--cell")),
    accent: hexToRgb01(css("--accent")), ax: ["--ax-a", "--ax-b", "--ax-c"].map(css) };
}

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
  buildStructure(); requestSurface(false);
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

/* ------------------------------------------------------------------ scene */
function radiusOf(el) {
  const k = S.scale;
  if (S.style === "space") return rcov(el) * k;
  if (S.style === "stick") return 0.14 * k;
  return Math.max(0.2, 0.42 * rcov(el)) * k;
}
function buildStructure() {
  if (!R) return;
  const st = S.structure;
  let atoms = [], bonds = [];
  const segs = [];
  if (st) {
    const reps = st.pbc ? S.reps : [1, 1, 1];
    if (S.style !== "off" && st.species.length) {
      atoms = displayAtoms(st, { reps, boundary: S.boundary && st.pbc });
      for (const a of atoms) { a.r = radiusOf(a.el); a.color = colorOf(a.el); }
      if (S.bonds && S.style !== "space") {
        const br = (S.style === "stick" ? 0.14 : 0.1) * S.scale;
        bonds = bondList(atoms, { tol: S.tol, skip: S.ionic ? new Set() : IONIC_CATIONS }).map(([i, j]) => {
          const a = atoms[i], b = atoms[j];
          return { p0: [a.x, a.y, a.z], p1: [b.x, b.y, b.z], c0: a.color, c1: b.color, r: br };
        });
      }
    }
    if (S.cell && st.lattice) {
      const L = st.lattice;
      const box = (A, B, C, dashed) => {
        const P = (i, j, k) => [0, 1, 2].map(c => i * A[c] + j * B[c] + k * C[c]);
        const E = [[0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0],
          [0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1]];
        for (const e of E) segs.push({ p0: P(e[0], e[1], e[2]), p1: P(e[3], e[4], e[5]), dashed });
      };
      box(L[0], L[1], L[2], false);
      if (reps.some(r => r > 1)) box(L[0].map(x => x * reps[0]), L[1].map(x => x * reps[1]), L[2].map(x => x * reps[2]), true);
    }
  }
  // keep selected and labelled atoms that are still drawn (same structure atom at the same place)
  const remap = list => list.map(s => {
    const i = atoms.findIndex(a => a.src === s.src && Math.abs(a.x - s.p[0]) + Math.abs(a.y - s.p[1]) + Math.abs(a.z - s.p[2]) < 1e-3);
    return i < 0 ? null : { ...s, i };
  }).filter(Boolean);
  S.sel = remap(S.sel); S.labels = remap(S.labels);
  disp = { atoms, centers: Float32Array.from(atoms.flatMap(a => [a.x, a.y, a.z])), radii: Float32Array.from(atoms.map(a => a.r)) };
  uploadAtoms(); uploadLabels();
  R.setBonds(packBonds(bonds), bonds.length);
  R.setLines(packLines(segs), segs.length);
  updateMeasure();
  requestRender();
}
function uploadLabels() {
  if (!R.setLabels) return;
  const list = S.labels.map(s => { const a = disp.atoms[s.i]; return { x: a.x, y: a.y, z: a.z, r: Math.max(a.r, 0.36), text: a.el, color: a.color }; });
  R.setLabels(packLabels(list, S.atlas ? S.atlas.map : {}), list.length, S.atlas);
  labelBar();
}
function setLabelMode(on) {
  S.labelMode = on;
  if (on) clearSelection();
  $("btn-label").setAttribute("aria-pressed", String(on));
  $("stage").classList.toggle("labelling", on);
  labelBar();
}
function labelBar() {
  const bar = $("labelbar");
  bar.hidden = !S.labelMode;
  if (S.labelMode) $("label-count").textContent = S.labels.length ? `${S.labels.length} labelled` : "Click atoms to label them";
}
function uploadAtoms() {
  const sel = new Set(S.sel.map(s => s.i));
  disp.atoms.forEach((a, i) => (a.sel = sel.has(i)));
  R.setAtoms(packAtoms(disp.atoms), disp.atoms.length);
}

/* surfaces: one request at a time; the newest pending request runs when the current one returns */
let meshBusy = false, meshNext = null;
function requestSurface(preview = false) { meshNext = { preview }; if (!meshBusy) runSurface(); }
async function runSurface() {
  const p = meshNext;
  meshNext = null;
  if (!p || !R) return;
  const f = S.fields[S.fi];
  if (!f || !S.surf) { R.setMeshes([]); S.mesh = null; levelHint(); summaries(); requestRender(); return; }
  meshBusy = true;
  try {
    const reps = f.periodic && S.structure && S.structure.pbc ? S.reps : [1, 1, 1];
    const budget = p.preview ? 1.5e6 : 8e6, auto = autoStride(f, reps, budget);
    const stride = S.res === "auto" ? auto : Math.max(+S.res, p.preview ? auto : 1);
    const signed = S.mode === "log" && f.kind === "signed" && S.pm && f.stats.min < -1e-6 * f.stats.maxAbs;
    const { meshes } = await call("mesh", { fi: S.fi, reps, stride, isos: signed ? [S.iso, -S.iso] : [S.iso] });
    R.setMeshes(meshes.map((m, i) => ({ ...m, color: i ? pal.neg : pal.pos, opacity: S.opacity })));
    S.mesh = { tris: meshes.reduce((a, m) => a + m.triangles, 0), stride, truncated: meshes.some(m => m.truncated), n: meshes.length };
    levelHint(); summaries(); requestRender();
  } catch (e) { toast(e.message || String(e), true); }
  finally { meshBusy = false; }
  if (meshNext) runSurface();
}
function recolorMeshes() {
  if (!R || !S.mesh) return;
  R.setMeshColors([...Array(S.mesh.n)].map((_, i) => ({ color: i ? pal.neg : pal.pos, opacity: S.opacity })));
  requestRender();
}

/* ------------------------------------------------------------------ camera, views, triad */
function sceneSphere() {
  const st = S.structure, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const add = p => { for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); } };
  for (const a of disp.atoms) add([a.x, a.y, a.z]);
  if (st && st.lattice && (S.cell || !disp.atoms.length)) {
    const L = st.lattice, r = st.pbc ? S.reps : [1, 1, 1];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) add([0, 1, 2].map(c => i * r[0] * L[0][c] + j * r[1] * L[1][c] + k * r[2] * L[2][c]));
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
function viewAlong(i) {
  const L = S.structure && S.structure.lattice;
  if (!L) return;
  cam.q = qfromRows(rows(L[i], L[(i + 2) % 3])); fit(); requestRender();
}
function resetView() {
  const L = S.structure && S.structure.lattice;
  if (L) { const a = unit(L[0]), b = unit(L[1]), c = unit(L[2]); cam.q = qfromRows(rows([0, 1, 2].map(i => a[i] + 0.36 * b[i] + 0.3 * c[i]), c)); }
  else cam.q = [0, 0, 0, 1];
  fit(); requestRender();
}
function drawTriad() {
  const svg = $("triad"), L = S.structure && S.structure.lattice;
  if (!L) { svg.innerHTML = ""; return; }
  const M = qmat(cam.q);
  const ax = L.map((a, i) => { const u = unit(a); return { r: [0, 1, 2].map(k => M[k][0] * u[0] + M[k][1] * u[1] + M[k][2] * u[2]), n: "abc"[i], c: pal.ax[i] }; })
    .sort((p, q) => p.r[2] - q.r[2]);
  svg.innerHTML = ax.map(({ r, n, c }) => {
    const x = 22 * r[0], y = -22 * r[1], o = 0.45 + 0.55 * (r[2] + 1) / 2;
    return `<g opacity="${o.toFixed(2)}"><line x1="0" y1="0" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${c}" stroke-width="2" stroke-linecap="round"/>` +
      `<text x="${(x * 1.32).toFixed(1)}" y="${(y * 1.32 + 4).toFixed(1)}" text-anchor="middle" fill="${c}">${n}</text></g>`;
  }).join("");
}

/* ------------------------------------------------------------------ picking and measurement */
const atomLabel = s => `${s.el}${s.src + 1}`;
function clickAt(x, y) {
  const i = disp.atoms.length ? pickSphere(cam.ray(x, y), disp.centers, disp.radii) : -1;
  if (S.labelMode) {
    if (i < 0) return;
    const k = S.labels.findIndex(s => s.i === i), a = disp.atoms[i];
    if (k >= 0) S.labels.splice(k, 1); else S.labels.push({ i, src: a.src, el: a.el, p: [a.x, a.y, a.z] });
    uploadLabels(); requestRender();
    return;
  }
  if (i < 0) return clearSelection();
  const k = S.sel.findIndex(s => s.i === i);
  if (k >= 0) S.sel.splice(k, 1);
  else {
    if (S.sel.length >= 4) S.sel = [];
    const a = disp.atoms[i];
    S.sel.push({ i, src: a.src, el: a.el, p: [a.x, a.y, a.z] });
  }
  uploadAtoms(); updateMeasure(); requestRender();
}
function clearSelection() {
  if (!S.sel.length) return;
  S.sel = []; uploadAtoms(); updateMeasure(); requestRender();
}
function updateMeasure() {
  const box = $("measure");
  if (!S.sel.length) { box.hidden = true; box.innerHTML = ""; return; }
  const m = measureAll(S.sel.map(s => s.p)), parts = [];
  if (S.sel.length === 1) parts.push(`<span class="mv">${S.sel[0].p.map(v => v.toFixed(3)).join("  ")} Å</span>`);
  m.bonds.forEach((d, i) => parts.push(`<span class="mk">${esc(atomLabel(S.sel[i]))}–${esc(atomLabel(S.sel[i + 1]))}</span><span class="mv">${d.toFixed(3)} Å</span>`));
  m.angles.forEach((a, i) => parts.push(`<span class="mk">∠${esc(atomLabel(S.sel[i + 1]))}</span><span class="mv">${a.toFixed(2)}°</span>`));
  if (m.dihedral !== null) parts.push(`<span class="mk">dihedral</span><span class="mv">${m.dihedral.toFixed(2)}°</span>`);
  box.innerHTML = `<span class="atoms">${S.sel.map(s => esc(atomLabel(s))).join(" · ")}</span>${parts.join("")}` +
    `<button type="button" class="x" id="sel-clear" aria-label="Clear the selection">×</button>`;
  box.hidden = false;
  $("sel-clear").addEventListener("click", clearSelection);
}
function drawOverlay() {
  const ov = $("ov");
  if (!S.sel.length) { if (ov.firstChild) ov.innerHTML = ""; return; }
  const P = S.sel.map(s => cam.project(s.p));
  if (P.some(p => !p)) { ov.innerHTML = ""; return; }
  const m = measureAll(S.sel.map(s => s.p));
  const tag = (x, y, t) => { const w = 7 * t.length + 12; return `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)})"><rect x="${-w / 2}" y="-10" width="${w}" height="19" rx="6"/><text y="4" text-anchor="middle">${esc(t)}</text></g>`; };
  let h = "";
  for (let i = 0; i + 1 < P.length; i++) h += `<line class="ml" x1="${P[i].x.toFixed(1)}" y1="${P[i].y.toFixed(1)}" x2="${P[i + 1].x.toFixed(1)}" y2="${P[i + 1].y.toFixed(1)}"/>`;
  m.bonds.forEach((d, i) => { h += tag((P[i].x + P[i + 1].x) / 2, (P[i].y + P[i + 1].y) / 2 - 12, d.toFixed(3) + " Å"); });
  m.angles.forEach((a, i) => { h += tag(P[i + 1].x, P[i + 1].y + 22, a.toFixed(1) + "°"); });
  if (m.dihedral !== null) h += tag((P[1].x + P[2].x) / 2, (P[1].y + P[2].y) / 2 + 22, "φ " + m.dihedral.toFixed(1) + "°");
  ov.innerHTML = h;
}

/* pointer: drag rotates (right button, shift or ctrl pans), wheel and pinch zoom, a click without motion picks */
const stage = $("stage"), ptrs = new Map();
let drag = null;
stage.addEventListener("pointerdown", e => {
  if (!R || !S.structure) return;
  if (stage.setPointerCapture) stage.setPointerCapture(e.pointerId);
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (ptrs.size === 1) drag = { x0: e.clientX, y0: e.clientY, moved: false, pan: e.button !== 0 || e.shiftKey || e.ctrlKey || e.metaKey, button: e.button };
  else if (drag) drag.moved = true;
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
    if (drag.moved) { if (drag.pan) cam.panBy(cur.x - prev.x, cur.y - prev.y); else cam.rotateBy(cur.x - prev.x, cur.y - prev.y); }
  }
  ptrs.set(e.pointerId, cur);
  if (drag.moved) requestRender();
});
const endPtr = e => {
  if (!ptrs.has(e.pointerId)) return;
  ptrs.delete(e.pointerId);
  if (drag && !drag.moved && ptrs.size === 0 && e.type === "pointerup" && drag.button === 0) clickAt(e.clientX, e.clientY);
  if (ptrs.size === 0) drag = null;
};
stage.addEventListener("pointerup", endPtr);
stage.addEventListener("pointercancel", endPtr);
stage.addEventListener("wheel", e => { if (!R) return; e.preventDefault(); cam.zoomBy(Math.exp(e.deltaY * (e.deltaMode ? 0.05 : 0.0015))); requestRender(); }, { passive: false });
stage.addEventListener("contextmenu", e => e.preventDefault());
window.addEventListener("resize", () => { if (R) resize(); });

/* ------------------------------------------------------------------ fields and levels */
function selectField(i, keepP = true) {
  S.fi = i;
  const f = S.fields[i];
  if (!f) { S.st = null; return; }
  S.st = f.stats;
  S.mode = f.kind === "linear" ? "linear" : "log";
  // WHY: a total density (many electrons) at 50% enclosed hugs the cores; 90% gives the usual bonding-scale level
  if (!keepP) S.p = S.st.integral > 3 ? 0.9 : 0.5;
  if (S.mode === "log") S.iso = isoForFraction(S.st, S.p);
  else S.iso = /ELF/i.test(f.name) && S.st.max >= 0.75 && S.st.min <= 0.75 ? 0.75 : 0.5 * (S.st.min + S.st.max);
  syncLevel();
}
function logRange() { const hi = Math.log10(S.st.maxAbs) - 1e-3; return [hi - 6, hi]; }
function syncLevel() {
  if (!S.fields[S.fi]) return;
  const sl = $("iso");
  if (S.mode === "log") { const [lo, hi] = logRange(); sl.value = Math.round(1000 * (Math.log10(Math.max(S.iso, 1e-300)) - lo) / (hi - lo)); }
  else sl.value = Math.round(1000 * (S.iso - S.st.min) / ((S.st.max - S.st.min) || 1));
  $("iso-num").value = fmt(S.iso);
  levelHint();
}
function levelHint() {
  const f = S.fields[S.fi];
  if (!f) return;
  const u = f.unit ? " " + f.unit : "", signed = f.kind === "signed";
  if (S.mode === "log") $("iso-hint").textContent = `${signed ? "±" : ""}${sup(fmt(S.iso))}${u} · encloses ${Math.round(100 * fractionAbove(S.st, S.iso))}% of ${signed ? "∫|f|" : "the total"}`;
  else $("iso-hint").textContent = `${sup(fmt(S.iso))}${u}`;
  $("row-pm").hidden = !(S.mode === "log" && signed && S.st.min < -1e-6 * S.st.maxAbs);
  const I = S.st.integral, fi = Math.abs(I) >= 100 ? I.toFixed(2) : +I.toPrecision(5) + "";
  const m = S.mesh, ord = n => n + (["th", "st", "nd", "rd"][n] || "th");
  const tri = m ? ` · ${m.tris >= 1e6 ? (m.tris / 1e6).toFixed(2) + "M" : Math.round(m.tris / 1e3) + "k"} triangles${m.stride > 1 ? ` (every ${ord(m.stride)} point)` : ""}${m.truncated ? ", truncated" : ""}` : "";
  $("stats").textContent = `${f.n.join(" × ")} grid · min ${fmt(S.st.min)} · max ${fmt(S.st.max)} · ∫ ${fi}${f.unit === "e/Å³" ? " e" : ""}${tri}`;
}
function setLevelFromSlider(preview) {
  const u = +$("iso").value / 1000;
  if (S.mode === "log") { const [lo, hi] = logRange(); S.iso = Math.pow(10, lo + u * (hi - lo)); S.p = fractionAbove(S.st, S.iso); }
  else S.iso = S.st.min + u * (S.st.max - S.st.min);
  $("iso-num").value = fmt(S.iso);
  levelHint(); summaries();
  requestSurface(preview);
}

/* ------------------------------------------------------------------ WAVECAR */
async function showBand(b) {
  const w = S.wave, { s, k, mode, half } = S.w;
  S.w.b = b;
  busy(true);
  try {
    const { field } = await call("orbital", { s, k, b, signed: mode === "psi" && w.type === "gam", half });
    S.fields = [field];
    selectField(0, true); fillFieldSelect(); markBand(); summaries();
    requestSurface(false);
  } catch (e) { toast(e.message || String(e), true); }
  finally { busy(false); }
}
function buildBands() {
  const w = S.wave, { s, k } = S.w, e = w.eig[s][k], o = w.occ[s][k], fr = w.frontier[s][k], list = $("bands"), rows = [];
  for (let b = 0; b < w.nb; b++) {
    const cls = ["band", o[b] > 0.5 ? "occ" : "", b === fr.homo ? "homo" : "", b === fr.lumo ? "lumo" : ""].join(" ");
    rows.push(`<div class="${cls}" role="option" data-b="${b}" aria-selected="false"><span>${b + 1}</span><span class="e">${e[b].toFixed(3)}</span><span class="o">${o[b].toFixed(2)}</span></div>`);
  }
  list.innerHTML = rows.join("");
  const h = list.querySelector(`[data-b="${Math.max(0, fr.homo)}"]`);
  if (h) list.scrollTop = h.offsetTop - list.clientHeight / 2;
}
function markBand() {
  for (const el of $("bands").querySelectorAll('[aria-selected="true"]')) el.setAttribute("aria-selected", "false");
  const el = $("bands").querySelector(`[data-b="${S.w.b}"]`);
  if (el) el.setAttribute("aria-selected", "true");
}

/* ------------------------------------------------------------------ loading and chrome */
async function openFiles(list) {
  const files = [...list];
  if (!files.length) return;
  busy(true);
  try {
    if (!(await ensureRenderer())) return;
    const attach = !!(S.wave && S.structure && !S.structure.species.length);
    const d = await call("open", { files, attach });
    if (d.attach) {
      if (d.structure && sameCell(d.structure.lattice, S.wave.lattice)) {
        S.structure = d.structure; S.atlas = buildAtlas([...new Set(S.structure.species)]);
        buildStructure(); updateChrome(); toast("Atoms added to the WAVECAR cell.");
      }
      else toast("That structure's cell differs from the WAVECAR's, so it was not added.", true);
      return;
    }
    await load({ name: files.map(f => f.name).join(" + "), ...d });
  } catch (e) { toast(e.message || String(e), true, 8000); console.error(e); }
  finally { busy(false); }
}
async function load(d) {
  if (!(await ensureRenderer())) return;
  Object.assign(S, { name: d.name, structure: d.structure, fields: d.fields || [], wave: d.wave || null, sel: [], labels: [], mesh: null, fi: 0 });
  S.atlas = S.structure && S.structure.species.length ? buildAtlas([...new Set(S.structure.species)]) : null;
  setLabelMode(false);
  S.w = { s: 0, k: 0, b: -1, mode: "density", half: "x" };
  S.p = 0.5;
  S.reps = [1, 1, 1]; ["ra", "rb", "rc"].forEach(id => ($(id).value = 1));
  const notes = [...(d.notes || [])];
  if (S.wave && !(S.structure && sameCell(S.structure.lattice, S.wave.lattice))) {
    notes.push(S.structure ? "the structure's cell differs from the WAVECAR's, so only the cell is drawn" : "add the matching POSCAR or CONTCAR to draw the atoms");
    S.structure = { lattice: S.wave.lattice, pbc: true, species: [], pos: new Float64Array(0) };
  }
  R.setMeshes([]);
  $("empty").hidden = true;
  if (S.wave) {
    $("kpt").innerHTML = S.wave.kpts.map((k, i) => `<option value="${i}">${i + 1}: ${k.map(x => x.toFixed(3)).join(" ")}</option>`).join("");
    buildBands();
  }
  if (S.fields.length) selectField(0, false);
  fillFieldSelect(); updateChrome(); buildStructure(); resetView();
  for (const id of ["sec-structure", "sec-surface", "sec-orbital", "sec-view"]) $(id).open = false;
  if (S.wave) { $("sec-orbital").open = true; await showBand(Math.max(0, S.wave.frontier[0][0].homo)); }
  else if (S.fields.length) { $("sec-surface").open = true; requestSurface(false); }
  if (notes.length) toast(notes.join("; "), false, 7000);
}
function fillFieldSelect() {
  $("field").innerHTML = S.fields.map((f, i) => `<option value="${i}">${esc(f.name)}</option>`).join("");
  $("field").value = S.fi;
  $("row-field").hidden = S.fields.length < 2;
}
function updateChrome() {
  const st = S.structure, has = !!st;
  $("dock").hidden = !has; $("br").hidden = !has; $("bl").hidden = !has; $("doc").hidden = !has;
  $("sec-surface").hidden = !(S.fields.length || S.wave);
  $("sec-orbital").hidden = !S.wave;
  for (const el of document.querySelectorAll(".pbc")) el.hidden = !(st && st.pbc);
  for (const b of document.querySelectorAll("[data-along]")) b.hidden = !(st && st.lattice);
  if (S.wave) {
    $("row-spin").hidden = S.wave.nspin < 2; $("row-k").hidden = S.wave.nk < 2;
    $("row-wmode").hidden = $("row-half").hidden = S.wave.type !== "gam";
  }
  $("doc-name").textContent = S.name;
  $("doc-sub").textContent = st && st.species.length ? formula(st.species) : S.wave ? `WAVECAR · ${S.wave.type === "gam" ? "Γ-only" : S.wave.type} · ${S.wave.nb} bands` : "";
  let info = "";
  if (st) {
    if (st.species.length) info += `${st.species.length} atoms`;
    if (st.lattice) {
      const p = params(st.lattice);
      info += `${info ? " · " : ""}${p.slice(0, 3).map(x => x.toFixed(3)).join(" ")} Å · ${p.slice(3).map(x => x.toFixed(2)).join(" ")}° · ${Math.abs(det3(st.lattice)).toFixed(1)} Å³`;
    }
  }
  $("info").textContent = info;
  summaries(); segState();
}
function summaries() {
  const st = S.structure;
  $("sum-structure").textContent = !st ? "" : S.style === "off" ? "hidden" : `${S.style}${st.pbc ? " · " + S.reps.join("×") : ""}`;
  const f = S.fields[S.fi];
  $("sum-surface").textContent = !f ? "" : !S.surf ? "off" : S.mode === "log" ? `${fmt(S.iso)} · ${Math.round(100 * fractionAbove(S.st, S.iso))}%` : fmt(S.iso);
  if (S.wave && S.w.b >= 0) $("sum-orbital").textContent = `${S.w.b + 1}${S.wave.nspin === 2 ? (S.w.s ? "↓" : "↑") : ""} · ${S.wave.eig[S.w.s][S.w.k][S.w.b].toFixed(2)} eV`;
  $("sum-view").textContent = S.proj === "ortho" ? "ortho" : "perspective";
}
function segState() {
  const val = { style: S.style, res: S.res, wspin: String(S.w.s), wmode: S.w.mode, half: S.w.half, proj: S.proj, theme: S.theme };
  for (const seg of document.querySelectorAll(".seg")) for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === val[seg.dataset.key]));
}

/* ------------------------------------------------------------------ home page, examples, image */
function showHome() {
  $("empty").hidden = false;
  const back = $("btn-back");
  back.hidden = !S.structure;
  if (S.structure) back.textContent = `Back to ${S.name}`;
}
async function example(id) {
  busy(true);
  try {
    if (id === "h3d") {
      if (!(await ensureRenderer())) return;
      await load({ name: "hydrogen 3d(x²−y²), computed here", ...(await call("example", { name: "h3d" })) });
      return;
    }
    const url = { rutile: "examples/rutile-TiO2.cif", graphene: "examples/graphene.pw.in" }[id];
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load the example (${r.status}).`);
    await openFiles([new File([await r.blob()], url.split("/").pop())]);
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
    a.download = (S.name.split(" + ")[0] || "isoview").replace(/[^\w.-]+/g, "_") + ".png";
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, "image/png");
}
function restyle() { if (!R) return; readPalette(); R.setBackground(pal.bg); recolorMeshes(); requestRender(); }

/* ------------------------------------------------------------------ wiring */
function bindSeg(key, fn) {
  document.querySelector(`.seg[data-key="${key}"]`).addEventListener("click", e => { const b = e.target.closest("button"); if (b) { fn(b.value); segState(); summaries(); } });
}
bindSeg("style", v => { S.style = v; buildStructure(); });
bindSeg("res", v => { S.res = v; requestSurface(false); });
bindSeg("proj", v => { S.proj = v; cam.ortho = v === "ortho"; requestRender(); });
bindSeg("theme", v => {
  S.theme = v;
  if (v === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", v);
  try { localStorage.setItem("isoview-theme", v); } catch (e) { /* storage unavailable */ }
  restyle();
});
bindSeg("wspin", v => { S.w.s = +v; buildBands(); showBand(Math.max(0, S.w.b)); });
bindSeg("wmode", v => { S.w.mode = v; showBand(Math.max(0, S.w.b)); });
bindSeg("half", v => { S.w.half = v; showBand(Math.max(0, S.w.b)); });
$("kpt").addEventListener("change", e => { S.w.k = +e.target.value; buildBands(); showBand(Math.max(0, S.w.b)); });
$("bands").addEventListener("click", e => { const r = e.target.closest(".band"); if (r) showBand(+r.dataset.b); });
$("bands").addEventListener("keydown", e => {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault();
  const b = Math.min(S.wave.nb - 1, Math.max(0, S.w.b + (e.key === "ArrowDown" ? 1 : -1)));
  showBand(b);
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
  buildStructure(); fit(); requestSurface(false); summaries();
});
$("field").addEventListener("change", e => { selectField(+e.target.value, true); requestSurface(false); summaries(); });
$("surf").addEventListener("change", e => { S.surf = e.target.checked; requestSurface(false); summaries(); });
$("pm").addEventListener("change", e => { S.pm = e.target.checked; requestSurface(false); });
$("opacity").addEventListener("input", e => { S.opacity = +e.target.value; recolorMeshes(); });
$("iso").addEventListener("input", () => setLevelFromSlider(true));
$("iso").addEventListener("change", () => setLevelFromSlider(false));
$("iso-num").addEventListener("change", e => {
  const v = parseFloat(e.target.value.replace("×10", "e"));
  if (!Number.isFinite(v)) return syncLevel();
  S.iso = S.mode === "log" ? Math.abs(v) : v;
  if (S.mode === "log") S.p = fractionAbove(S.st, S.iso);
  syncLevel(); requestSurface(false); summaries();
});
for (const b of document.querySelectorAll("[data-along]")) b.addEventListener("click", () => viewAlong(+b.dataset.along));
$("btn-reset").addEventListener("click", resetView);
$("btn-snap").addEventListener("click", snapshot);
$("btn-home").addEventListener("click", showHome);
$("btn-label").addEventListener("click", () => setLabelMode(!S.labelMode));
$("label-done").addEventListener("click", () => setLabelMode(false));
$("label-clear").addEventListener("click", () => { S.labels = []; uploadLabels(); requestRender(); });
$("btn-back").addEventListener("click", () => { $("empty").hidden = true; });
$("btn-open2").addEventListener("click", () => $("file").click());
$("btn-open").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", e => { openFiles(e.target.files); e.target.value = ""; });
for (const b of document.querySelectorAll("[data-example]")) b.addEventListener("click", () => example(b.dataset.example));

let dragDepth = 0;
window.addEventListener("dragenter", e => { if ([...((e.dataTransfer && e.dataTransfer.types) || [])].includes("Files")) { dragDepth++; $("drop").hidden = false; e.preventDefault(); } });
window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").hidden = true; } });
window.addEventListener("drop", e => { e.preventDefault(); dragDepth = 0; $("drop").hidden = true; if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) openFiles(e.dataTransfer.files); });
window.addEventListener("paste", e => {
  if (e.target && e.target.closest && e.target.closest("input, select, textarea")) return;
  const t = e.clipboardData && e.clipboardData.getData("text");
  if (t && t.trim()) openFiles([new File([t], "pasted")]);
});
window.addEventListener("keydown", e => {
  if ((e.target && e.target.closest && e.target.closest("input, select, textarea")) || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "o") $("file").click();
  else if (k === "escape") {
    if (!$("empty").hidden && S.structure) $("empty").hidden = true;
    else if (S.labelMode) setLabelMode(false);
    else if (S.sel.length) clearSelection();
    else for (const d of document.querySelectorAll(".sec[open]")) d.open = false;
  }
  else if (!S.structure || !$("empty").hidden) return;
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
    .then(b => openFiles([new File([b], decodeURIComponent(u.split(/[?#]/)[0].split("/").pop() || "file"))]))
    .catch(e => toast(`Could not fetch ${u}: ${e.message}. The server must allow cross-origin requests.`, true, 9000));
} else if (qp.get("example")) example(qp.get("example"));
