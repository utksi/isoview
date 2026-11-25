// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
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
 * Embedded (?embed, in an iframe on a website or in a notebook): see initEmbed; nothing is kept in this browser.
 * Slice (SL, SV): a 2D cut through the active field on the (hkl) plane at position t of its cell (supercell for
 * periodic fields), sampled in the worker (slice.js), moved with the slice bar; drawn as a textured plane in the scene,
 * as the 2D main view and in a movable inset (the 2D figure beside the 3D view or the reverse) with contour lines, the
 * atoms within SL_NEAR of the plane and a colour bar; Photo Mode saves that picture, the Surface panel an SVG.
 * Trajectory analysis (AN): RDF, MSD with diffusion coefficients, VDOS and per-frame series of the main trajectory
 * (analysis.js, run in the worker on the atoms still present), plotted by plot.js; chips switch curves (per frame:
 * quantities, one panel each) on and off; the curves shown save as CSV, or as a 3.5 in figure in SVG or PNG.
 * Brillouin zone (S.bz): the view shows the first Brillouin zone of the active structure (bz.js: Wigner-Seitz cell of
 * the reciprocal lattice, Setyawan-Curtarolo points and path) instead of the scene, scaled by BZ_SIZE / max |k|;
 * the crystal view (camera included) comes back when it is switched off. A crystal inset (BZI) draws the structure
 * with a second renderer and the main camera's rotation.
 * Colours: element colours (S.colors, element chips) and single-atom colours (S.atomColors, by item and atom id) over
 * the defaults of elements.js, chosen in the in-page picker (picker.js); recolor() repaints without rebuilding, so the
 * scene follows the picker live.
 * X-ray diffraction (AN.kind "xrd", xrd.js in the worker): the active crystal's pattern, or for a trajectory the frame
 * shown, the average over frames, or a 2theta x time map.
 * Look: Glossy on a first visit, then the last choice (localStorage isoview-look).
 * Photo mode (PH): controls hidden, the view supersampled; saved or copied images are drawn in tiles
 * (camera.js tileProjection) at PH_SS times the output size, with matting for a transparent background.
 */
import { params, formula, det3, norm, cross, dot, inv3, cartToFrac, fracToCart } from "./lattice.js?v=37";
import { fractionAbove, isoForFraction, autoStride, insidePolygon } from "./field.js?v=37";
import { displayAtoms, bondList, atomId, deleteAtoms, toPOSCAR, shownStructure, toCIF, toXYZ } from "./structure.js?v=37";
import { primitive, conventional, sameCellShape } from "./symmetry.js?v=37";
import { colorOf, rcov, SPACERS, SYMBOLS } from "./elements.js?v=37";
import { Camera, qfromRows, qmat, pickSphere, tileProjection } from "./gfx/camera.js?v=37";
import { createRenderer, packAtoms, packBonds, packLines, packUniforms, hexToRgb01 } from "./gfx/renderer.js?v=37";
import { buildAtlas, packLabels } from "./gfx/labels.js?v=37";
import { measureAll } from "./measure.js?v=37";
import { GifWriter, buildPalette } from "./gif.js?v=37";
import * as store from "./store.js?v=37";
import { contours, scaler, paint, lut, barTicks, figureLayout, layoutClear, drawFigure, drawColorBar, colorBarWidth } from "./slice.js?v=37";
import { chart, toCSV } from "./plot.js?v=37";
import { openPicker } from "./picker.js?v=37";
import { WAVELENGTHS, hklText } from "./xrd.js?v=37";
import { colormap } from "./slice.js?v=37";
import { wignerSeitz, kpath } from "./bz.js?v=37";

const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
/** Label atlas in the page typeface (--sans), so labels in the 3D view match the interface. */
const atlasOf = strings => buildAtlas(strings, 112, css("--sans") || undefined);
const qp = new URLSearchParams(location.search);
const EMBED = qp.has("embed"); // inside an iframe on another page (initEmbed)
// colour pairs (positive, negative) for successive field layers: mid-tones readable on light and dark backgrounds
const FIELD_COLORS = [["#e3a03a", "#3b86d6"], ["#2bb3a3", "#e0607e"], ["#9b7be0", "#7cb342"], ["#ef7d4b", "#5c9bd6"]];
// spin channels (a band of a spin-polarised WAVECAR, a CHGCAR's spin up / spin down): up warm, down cool;
// a magnetization field shows spin-up excess warm and spin-down excess blue
const SPIN_COLORS = [
  [["#e3a03a", "#d6456a"], ["#ef7d4b", "#b8558f"], ["#d9b23a", "#c0392b"]],
  [["#3b86d6", "#2bb3a3"], ["#6f7fe0", "#3fb6d8"], ["#2f6db3", "#5aa36a"]],
];

const S = {
  items: new Map(), layers: [], active: null, activeAtoms: null, t: [0, 0, 0],
  style: "ball", scale: 1, bonds: true, tol: 1.15, cell: true, reps: [1, 1, 1], boundary: true, res: "auto",
  proj: "persp", theme: "auto", sel: [], region: null, labels: [], labelMode: false, boxMode: false, undo: [], atlas: null,
  frame: 0, playing: false, fps: 12, step: 1, loop: "loop", movie: { fmt: "gif", scale: 0.5 }, exporting: null, look: "glossy",
  hide: new Set(), // elements not drawn (Structure panel checkboxes); a view setting, the structure keeps them
  colors: new Map(), // element -> "#rrggbb" chosen by the user (element chips)
  atomColors: new Map(), // atomKey (item id : atom id) -> "#rrggbb" for single atoms (selection bar: Colour)
};
// slice state (see the section "2D slices")
const SL_NEAR = 0.8; // A: atoms closer than this to the plane are drawn on the slice
const SL = { on: false, hkl: [0, 0, 1], t: 0.5, scale: "auto", map: "auto", iso: true, atoms: true, in3d: true, img: null, style: null, key: "", busy: false, again: null, fine: null };
// view: main "3d" or "2d", inset and colour bar on or off, inset size (CSS px, its longer side) and position
// ([left, bottom] CSS px, null: the default corner); the last three are kept in this browser
const SV = { main: "3d", inset: true, bar: true, size: 240, pos: null };
try {
  const v = JSON.parse(localStorage.getItem(store.PREFIX + "slice") || "{}");
  if (typeof v.inset === "boolean") SV.inset = v.inset;
  if (typeof v.bar === "boolean") SV.bar = v.bar;
  if (v.size >= 120) SV.size = v.size;
  if (Array.isArray(v.pos) && v.pos.every(Number.isFinite)) SV.pos = v.pos;
} catch (e) { /* storage unavailable */ }
const saveSV = () => { try { localStorage.setItem(store.PREFIX + "slice", JSON.stringify({ inset: SV.inset, bar: SV.bar, size: SV.size, pos: SV.pos })); } catch (e) { /* storage unavailable */ } };
const SL_FS = 12, SL_FS_INSET = 10; // CSS px: colour-bar text in the main view and in the inset
const TOL0 = 1.15; // default bond reach: bonded when d < reach (R1 + R2)
// render looks (shaders.js): [mode, outline width in CSS px, depth-fog strength]
const LOOKS = { classic: [0, 0, 0], soft: [1, 0, 0.5], glossy: [2, 0, 0], toon: [3, 1.5, 0], outline: [4, 1.3, 0] };
// the look of a first visit; afterwards the last one chosen (localStorage isoview-look, also for new scenes)
const LOOK0 = "glossy";
try { const v = localStorage.getItem(store.PREFIX + "look"); if (LOOKS[v]) S.look = v; } catch (e) { /* storage unavailable */ }
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
  worker = globalThis.__isoviewWorker ? new globalThis.__isoviewWorker() : new Worker(new URL("./worker.js?v=37", import.meta.url), { type: "module" });
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
  S.atlas = atlasOf(atlasSpecies());
  buildStructure(); refreshSurfaces();
  toast(`WebGPU reported an error (${msg}); switched to WebGL2.`, true, 9000);
}
/** Sizes the 3D canvas for its box (the window, or the slice inset while the 2D slice is the main view). */
function resize() {
  const inset = canvas.parentElement && canvas.parentElement.id === "sli", b = inset ? insetBox() : null;
  const w = inset ? b.w : window.innerWidth, h = inset ? b.h : window.innerHeight, d0 = window.devicePixelRatio || 1;
  // WHY: in photo mode the view is drawn at twice the device pixels and shown scaled down (supersampled: sphere
  // impostors discard whole pixels, so multisampling alone leaves their edges stepped), within the renderer's
  // largest side and PH_LIVE pixels
  const dpr = PH.on ? Math.min(2 * d0, 4, ((R && R.maxSide) || 4096) / Math.max(w, h), Math.sqrt(PH_LIVE / (w * h))) : Math.min(d0, 2);
  // WHY: set only on a change, since assigning the size (even the same) reallocates the drawing buffer
  const cw = Math.max(1, Math.round(w * dpr)), ch = Math.max(1, Math.round(h * dpr));
  if (canvas.width !== cw) canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  cam.w = w; cam.h = h;
  requestRender();
}
let dirty = false;
function requestRender() { if (!dirty) { dirty = true; requestAnimationFrame(frame); } saveSoon(); }
/** Uniforms for drawing with camera c into a target of w x h device pixels on renderer r. bg: background colour
 *  [r, g, b] in 0..1 (fog fades toward it); dpr: device pixels per CSS pixel (line widths and outlines scale with it);
 *  tile: [px, py, VW, VH] to draw the w x h pixel tile at (px, py) of a VW x VH image (camera.js tileProjection). */
function uniformsFor(c, r, w, h, bg = pal.bg, dpr = w / c.w, tile = null) {
  const lk = LOOKS[S.look] || LOOKS.classic, P0 = c.proj(r.name === "WebGPU" ? "gpu" : "gl"), P = tile ? tileProjection(P0, tile[0], tile[1], w, h, tile[2], tile[3]) : P0;
  return packUniforms(c.view(), P, [w, h, dpr, c.ortho ? 1 : 0],
    [0.35, 0.55, 1.0, 0.32], [...pal.accent, 0.33], [...pal.cell, 1.3 * dpr],
    [lk[0], lk[1] * dpr, lk[2], 0], [c.dist - c.radius, c.dist + c.radius, 0, 0], [...bg, 1]);
}
function frame() {
  dirty = false;
  if (!R) return;
  if (fitClear) { fitClear = false; clearOfPanels(); }
  const t0 = store.CHANNEL ? performance.now() : 0;
  if (view3dShown()) R.draw(uniformsFor(cam, R, canvas.width, canvas.height, liveBg()));
  if (S.bz && BZI.R && !$("bzinset").hidden) drawInset();
  drawOverlay(); drawTriad();
  if (store.CHANNEL) perfTick(t0);
}
// test site: frames drawn in the last second and the main-thread time of the last frame (ms; GPU work not included)
const PERF = { times: [], shown: 0 };
function perfTick(t0) {
  const t = performance.now(), q = PERF.times;
  q.push(t); while (q.length && q[0] < t - 1000) q.shift();
  if (t - PERF.shown < 250) return;
  PERF.shown = t;
  $("tb-perf").textContent = `${R.name} · ${canvas.width}×${canvas.height} · ${q.length} fps · ${(t - t0).toFixed(1)} ms`;
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
/** Colour of element el: the user's choice (S.colors) or the default (elements.js colorOf). */
const elColor = el => S.colors.get(el) || colorOf(el);
/** Key of atom src of layer L for S.atomColors: item id and the atom's stable id (atomId), so it survives deletions. */
const atomKey = (L, src) => `${L.item}:${atomId(itemOf(L).structure, src)}`;
/** Colour of display atom a of layer L: its own colour, else its element's. */
const atomColor = (L, a) => (S.atomColors.size && S.atomColors.get(atomKey(L, a.src))) || elColor(a.el);
/** Atoms, bonds and cell edges of the atoms layers `layers` and of the cells of `cellLayers`, as drawn now (style,
 *  supercell, boundary atoms, hidden elements, colours). Each bond keeps its two atoms (a, b) so that colours can
 *  change later without finding the bonds again. Returns { atoms, bonds, segs }. */
function structureScene(layers, cellLayers) {
  const atoms = [], bonds = [], segs = [];
  if (S.style !== "off") for (const L of layers) {
    const st = itemOf(L).structure, reps = st.pbc ? S.reps : [1, 1, 1];
    const da = displayAtoms(st, { reps, boundary: S.boundary && st.pbc, shift: fracShift(st.lattice), cart: S.t });
    for (const a of da) { a.layer = L.id; a.r = radiusOf(a.el); a.color = atomColor(L, a); }
    if (S.bonds && S.style !== "space") {
      const br = (S.style === "stick" ? 0.14 : 0.1) * S.scale;
      // WHY: bonds are found among all atoms, then those to hidden elements dropped, so hiding (say) Cl does not
      // bond the cations it separated
      for (const [i, j] of bondList(da, { tol: S.tol, spacers: SPACERS })) {
        const a = da[i], b = da[j];
        if (S.hide.has(a.el) || S.hide.has(b.el)) continue;
        bonds.push({ p0: [a.x, a.y, a.z], p1: [b.x, b.y, b.z], a, b, c0: a.color, c1: b.color, r: br });
      }
    }
    atoms.push(...(S.hide.size ? da.filter(a => !S.hide.has(a.el)) : da));
  }
  if (S.cell) {
    const cells = [];
    for (const L of cellLayers) {
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
  return { atoms, bonds, segs };
}
let shownBonds = [];
function buildStructure() {
  if (!R) return;
  if (S.bz) { buildBZ(); return; }
  const { atoms, bonds, segs } = structureScene(atomLayers(), S.layers);
  speciesUI();
  S.sel = remap(S.sel, atoms); S.labels = remap(S.labels, atoms);
  disp = { atoms, centers: Float32Array.from(atoms.flatMap(a => [a.x, a.y, a.z])), radii: Float32Array.from(atoms.map(a => a.r)) };
  shownBonds = bonds;
  uploadAtoms(); uploadLabels();
  R.setBonds(packBonds(bonds), bonds.length);
  R.setLines(packLines(segs), segs.length);
  updateSelectionBar(); updateInfo(); updateSymmetry();
  requestRender();
}
/** New colours without rebuilding the scene (live while a colour picker is open): atoms, bonds, labels, the element
 *  swatches, the slice window; in the Brillouin-zone view, the crystal inset. */
function recolor() {
  for (const b of document.querySelectorAll("#species .sp-c")) b.querySelector(".dot").style.background = elColor(b.dataset.el);
  if (!R) return;
  if (S.bz) { buildInset(); requestRender(); return; }
  for (const a of disp.atoms) { const L = layerById(a.layer); if (L) a.color = atomColor(L, a); }
  for (const b of shownBonds) { b.c0 = b.a.color; b.c1 = b.b.color; }
  uploadAtoms(); uploadLabels();
  R.setBonds(packBonds(shownBonds), shownBonds.length);
  if (SL.on) drawSliceViews();
  requestRender(); saveSoon();
}
/** One chip per element of the atoms layers on show: a checkbox (unticked = not drawn) and the element with its colour
 *  and atom count (a click opens the in-page colour picker, picker.js; the scene follows live). */
function speciesUI() {
  const count = new Map();
  for (const L of atomLayers()) for (const el of itemOf(L).structure.species) count.set(el, (count.get(el) || 0) + 1);
  const els = [...count.keys()].sort((a, b) => SYMBOLS.indexOf(a) - SYMBOLS.indexOf(b)), box = $("species");
  $("row-species").hidden = els.length === 0;
  const key = els.map(e => `${e}${count.get(e)}${S.hide.has(e) ? "-" : ""}`).join();
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.innerHTML = els.map(e => {
    const x = esc(e);
    return `<span class="sp"><input type="checkbox" data-el="${x}"${S.hide.has(e) ? "" : " checked"} title="${S.hide.has(e) ? "Show" : "Hide"} the ${x} atoms" aria-label="Show ${x}">` +
      `<button type="button" class="sp-c" data-el="${x}" title="Colour of the ${x} atoms" aria-haspopup="dialog"><span class="dot" style="background:${elColor(e)}"></span>${x}<em>${count.get(e)}</em></button></span>`;
  }).join("");
}
function uploadAtoms() {
  if (S.bz) return;
  const sel = new Set(S.sel.map(s => s.i));
  disp.atoms.forEach((a, i) => (a.sel = !PH.on && sel.has(i))); // no selection marks in photo mode
  R.setAtoms(packAtoms(disp.atoms), disp.atoms.length);
}
const atlasSpecies = () => [...new Set(S.layers.filter(l => l.kind === "atoms").flatMap(l => itemOf(l).structure.species))];
function uploadLabels() {
  if (!R || !R.setLabels || S.bz) return;
  const list = S.labels.map(s => { const a = disp.atoms[s.i]; return { x: a.x, y: a.y, z: a.z, r: Math.max(a.r, 0.36), text: a.el, color: a.color }; });
  R.setLabels(packLabels(list, S.atlas ? S.atlas.map : {}), list.length, S.atlas);
  labelBar();
}

/* ------------------------------------------------------------------ surfaces: one request at a time per layer */
function requestSurface(L, preview = false) { if (!L) return; L.next = { preview }; if (!L.busy) runSurface(L); }
function refreshSurfaces() { for (const L of fieldLayers()) requestSurface(L); syncSlice(); }
async function runSurface(L) {
  const p = L.next;
  L.next = null;
  if (!p || !R) return;
  const it = itemOf(L), f = it && it.fields[L.fi];
  if (!f || !L.visible || !S.layers.includes(L) || S.bz) { R.removeMeshGroup(L.id); L.mesh = null; if (L === activeField()) levelHint(); requestRender(); return; }
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
  if (S.bz && S.bzData) return { c: [0, 0, 0], r: BZ_SIZE * 1.15 };
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
function fit() { const s = sceneSphere(); cam.fit(s.c, s.r); fitClear = true; }
/** Boxes (CSS px, { x, y, w, h }, clipped to the window) of the panels over the view: the panel at the right, the
 *  slice inset, the Brillouin-zone inset and the analysis window; hidden and tucked ones are left out. None in Photo
 *  Mode, where the photo lays out its own inset. */
function viewObstacles() {
  if (PH.on) return [];
  const out = [], W = window.innerWidth, H = window.innerHeight;
  for (const id of ["dock", "sli", "bzinset", "anwin"]) {
    const el = $(id);
    if (!el || el.hidden || el.classList.contains("tucked")) continue;
    const r = id === "sli" ? insetBox() : el.getBoundingClientRect(), x = r.x ?? r.left, y = r.y ?? r.top;
    const x0 = Math.max(0, x), y0 = Math.max(0, y), x1 = Math.min(W, x + (r.w ?? r.width)), y1 = Math.min(H, y + (r.h ?? r.height));
    if (x1 > x0 && y1 > y0) out.push({ x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
  }
  return out;
}
let fitClear = false; // set by fit(): the next frame moves the camera back, if needed, so that panels do not cover the scene
/**
 * After a fit, keeps the scene centred on the window but clear of the panels over it: every atom (a disc of its
 * radius) and cell corner at screen offset v from the window centre must stay outside each panel box grown by
 * 16 px. Shrinking the view about the centre by s moves v to s v; a box that the segment from the centre to v enters
 * at parameter t (slab method) bounds s <= t. Boxes over the centre are ignored, and s >= 0.6 (beyond that the
 * overlap is accepted, as on narrow windows). The camera distance is divided by s (exact for the orthographic
 * projection, close for the perspective one near the target depth). At most 20000 atoms are sampled.
 */
function clearOfPanels() {
  if (!R || S.bz || (canvas && canvas.parentElement && canvas.parentElement.id === "sli")) return;
  const cx = cam.w / 2, cy = cam.h / 2;
  const boxes = viewObstacles().map(b => ({ x0: b.x - 16, y0: b.y - 16, x1: b.x + b.w + 16, y1: b.y + b.h + 16 }))
    .filter(b => !(cx > b.x0 && cx < b.x1 && cy > b.y0 && cy < b.y1));
  if (!boxes.length) return;
  const V = cam.view(), P = cam.proj("gl"), th = Math.tan(cam.fovy / 2);
  const proj = p => {
    const v = [0, 1, 2, 3].map(r => V[r] * p[0] + V[4 + r] * p[1] + V[8 + r] * p[2] + V[12 + r]);
    const c = [0, 1, 2, 3].map(r => P[r] * v[0] + P[4 + r] * v[1] + P[8 + r] * v[2] + P[12 + r] * v[3]);
    return c[3] > 1e-9 ? { x: (c[0] / c[3] + 1) / 2 * cam.w, y: (1 - c[1] / c[3]) / 2 * cam.h, ppu: cam.h / (2 * (cam.ortho ? cam.dist * th : -v[2] * th)) } : null;
  };
  let s = 1;
  const bound = (q, rad) => {
    if (!q) return;
    const dx = q.x - cx, dy = q.y - cy;
    for (const b of boxes) {
      let t0 = 0, t1 = 1;
      for (const [p0, d, lo, hi] of [[cx, dx, b.x0 - rad, b.x1 + rad], [cy, dy, b.y0 - rad, b.y1 + rad]]) {
        if (Math.abs(d) < 1e-9) { if (p0 <= lo || p0 >= hi) { t0 = 2; break; } continue; }
        let u = (lo - p0) / d, w = (hi - p0) / d; if (u > w) [u, w] = [w, u];
        t0 = Math.max(t0, u); t1 = Math.min(t1, w);
        if (t0 > t1) break;
      }
      if (t0 <= t1 && t0 < s) s = t0;
    }
  };
  const step = Math.max(1, Math.ceil(disp.atoms.length / 20000));
  for (let i = 0; i < disp.atoms.length; i += step) { const at = disp.atoms[i], q = proj([at.x, at.y, at.z]); bound(q, q ? (at.r || 0) * q.ppu : 0); }
  for (const L of S.layers) {
    if (!L.visible) continue;
    const it = itemOf(L), C = cellOf(it);
    if (!C || (!S.cell && disp.atoms.length)) continue;
    const r = pbcOf(it) ? S.reps : [1, 1, 1], o = originOf(it);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) bound(proj([0, 1, 2].map(c => o[c] + i * r[0] * C[0][c] + j * r[1] * C[1][c] + k * r[2] * C[2][c])), 0);
  }
  s = Math.max(s, 0.6);
  if (s < 0.999) cam.dist /= s;
}
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
  if (S.sel.length) {
    act.push(`<button type="button" data-act="color" title="Colour the selected atoms (in every periodic image)" aria-haspopup="dialog">Colour</button>`);
  }
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
  let sh = shownStructure(st, S.t, reps);
  const pc = primaryCell(), f = pc ? fracShift(pc.L) : [0, 0, 0], gone = sh.species.filter(e => S.hide.has(e));
  if (gone.length) {
    const keep = sh.species.map((e, i) => (S.hide.has(e) ? -1 : i)).filter(i => i >= 0);
    sh = { ...sh, ids: undefined, species: keep.map(i => sh.species[i]), pos: Float64Array.from(keep.flatMap(i => [sh.pos[3 * i], sh.pos[3 * i + 1], sh.pos[3 * i + 2]])) };
  }
  const note = [S.t.some(x => Math.abs(x) > 1e-9) ? `shifted by ${f.map(x => +x.toFixed(4)).join(" ")}` : "", reps.some(r => r > 1) ? `${reps.join("x")} supercell` : "",
    gone.length ? `without ${[...new Set(gone)].join(", ")}` : ""].filter(Boolean).join(", ");
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
  const it = itemOf(L), st = it.structure, tol = Math.max(1e-5, parseFloat($("t-tol").value) || 1e-5);
  if (!st.pbc || !st.lattice) return toast("This needs a periodic structure.", true);
  let res;
  try { res = kind === "primitive" ? primitive(st, tol) : conventional(st, tol); }
  catch (e) { return toast(e.message || String(e), true); }
  if (kind === "primitive" && res.n === 1) return toast(`Already primitive: no lattice translations within ${tol} Å besides the cell's own.`, false, 6000);
  if (kind === "conventional" && sameCellShape(st, res.structure, tol)) return toast(`Already the conventional ${res.system} cell (${st.species.length} atoms).`, false, 6000);
  const nit = { id: nextItem++, name: `${kind} of ${it.name}${it.traj ? ` (frame ${frameOf(it) + 1})` : ""}`, ds: null, structure: res.structure, wave: null, fields: [], nextSlot: 0 };
  S.items.set(nit.id, nit);
  const NL = { id: nextLayer++, item: nit.id, kind: "atoms", visible: true };
  S.layers.splice(S.layers.indexOf(L) + 1, 0, NL);
  const wasVisible = L.visible;
  L.visible = false; S.activeAtoms = NL.id;
  pushUndo({ type: "derive", layer: NL.id, source: L.id, wasVisible, label: `${kind} cell` });
  S.atlas = atlasOf(atlasSpecies());
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
  // the XRD window follows the frame shown (Frame: a new pattern; Map: the line at the frame)
  if (!$("anwin").hidden && AN.kind === "xrd" && AN.res) { if (AN.x.mode === "frame" || !AN.res.r.map) xrdFollow(); else if (AN.res.mode === "map") drawAnalysis(); }
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

/* ------------------------------------------------------------------ space group (moyo, in the worker) */
let symTimer = 0, symKey = "";
/** Hermann-Mauguin symbol for display: "F m -3 m" -> "Fm3̄m", "P 4_2/m n m" -> "P4₂/mnm". */
const hmText = h => h.replace(/\s+/g, "").replace(/-(\d)/g, "$1\u0304").replace(/_(\d)/g, (m, d) => "₀₁₂₃₄₅₆₇₈₉"[d]);
/** Name the space group of the chosen periodic structure (its current frame) next to the file name, at the symmetry
 *  tolerance of the cell tools; debounced, and skipped when nothing relevant changed. */
function updateSymmetry() {
  clearTimeout(symTimer);
  symTimer = setTimeout(async () => {
    const chip = $("doc-sg"), L = activeAtomsLayer(), it = L && itemOf(L), st = it && it.structure;
    if (!st || !st.pbc || !st.lattice || !st.species.length || !S.layers.length) { chip.hidden = true; symKey = ""; return; }
    const tol = Math.max(1e-5, parseFloat($("t-tol").value) || 1e-5);
    const key = [it.id, st.species.length, it.traj ? frameOf(it) : -1, tol, st.pos[0], st.pos[st.pos.length - 1]].join("|");
    if (key === symKey) return;
    symKey = key;
    const Li = inv3(st.lattice), frac = [];
    for (let i = 0; i < st.species.length; i++) frac.push(cartToFrac(Li, [st.pos[3 * i], st.pos[3 * i + 1], st.pos[3 * i + 2]]));
    try {
      const r = await call("symmetry", { lattice: st.lattice, frac, numbers: st.species.map(x => Math.max(0, SYMBOLS.indexOf(x))), symprec: tol });
      if (symKey !== key) return;
      chip.textContent = `${hmText(r.hm)} (${r.number})`;
      chip.title = `Space group of ${it.name}${it.traj ? `, frame ${frameOf(it) + 1}` : ""}: ${r.hm.replace(/\s+/g, "")}, No. ${r.number} (moyo, tolerance ${tol} Å; change it next to Primitive cell)`;
      chip.hidden = false;
    } catch (e) { chip.hidden = true; console.warn("symmetry:", e); }
  }, 150);
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
  if (S.labelMode) $("label-count").textContent = S.labels.length ? `${S.labels.length} labelled · click to add or remove` : "Click atoms to label them";
}
/** Label every atom drawn (all visible structures, periodic images included). */
function labelAll() {
  S.labels = disp.atoms.map((a, i) => ({ i, layer: a.layer, src: a.src, p: [a.x, a.y, a.z] }));
  uploadLabels(); requestRender();
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
  $("sec-orbital").hidden = !(it && it.wave && L.w);
  if (!L) return;
  $("field").innerHTML = it.fields.map((g, i) => `<option value="${i}">${esc(g.name)}</option>`).join("");
  $("field").value = L.fi;
  $("row-field").hidden = it.fields.length < 2 || !!L.w;
  $("row-addfield").hidden = it.fields.length < 2 || !!L.w;
  $("surf").checked = L.visible; $("pm").checked = L.pm; $("opacity").value = L.opacity;
  $("surf-layer").textContent = it.name;
  if (f) {
    const sl = $("iso");
    if (L.mode === "log") { const [lo, hi] = logRange(f.stats); sl.value = Math.round(1000 * (Math.log10(Math.max(L.iso, 1e-300)) - lo) / (hi - lo)); }
    else sl.value = Math.round(1000 * (L.iso - f.stats.min) / ((f.stats.max - f.stats.min) || 1));
    $("iso-num").value = fmt(L.iso);
  }
  if (it.wave && L.w) syncWaveUI(L, it);
  levelHint(); summaries();
}
/** One sentence on how to read the level, by the kind of field. */
function levelNote(f) {
  if (/ELF/i.test(f.name)) return "ELF runs from 0 to 1: near 1 the electrons form localised pairs (cores, bonds, lone pairs), near ½ they behave like a uniform electron gas.";
  if (f.kind === "signed") return "The two surfaces sit at plus and minus this value; inside them the field is larger in size, and those parts hold the share given of the integral of its absolute value.";
  if (f.kind === "density") return "Inside the surface the density is higher than this value, and those regions hold the share given: a high value keeps only the densest parts (cores, bonds), a low one reaches into the voids.";
  return f.unit === "eV" ? "The surface joins the points where the potential has this value." : "The surface joins the points where the field has this value.";
}
function levelHint() {
  const L = activeField(), f = activeStats();
  if (!L || !f) return;
  const st = f.stats, u = f.unit ? " " + f.unit : "", signed = f.kind === "signed";
  if (L.mode === "log") {
    const p = fractionAbove(st, L.iso), I = st.integral;
    // WHY: for a charge density the share is also a number of electrons; an orbital density holds one electron
    const of = signed ? (/ψ/.test(f.name) ? "∫|ψ|" : "∫|f|") : /^band/.test(f.name) ? "the orbital's electron" : "the total";
    const count = !signed && f.unit === "e/Å³" && I > 0.5 ? ` (${(p * I).toFixed(I >= 100 ? 0 : 1)} of ${I.toFixed(I >= 100 ? 0 : 1)} e)` : "";
    $("iso-hint").textContent = `${signed ? "±" : ""}${sup(fmt(L.iso))}${u} · encloses ${Math.round(100 * p)}% of ${of}${count}`;
  } else $("iso-hint").textContent = `${sup(fmt(L.iso))}${u}`;
  $("iso-note").textContent = levelNote(f);
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

/* ------------------------------------------------------------------ layer colours */
/** Spin channel a field layer shows: 0 up, 1 down, -1 neither (an unpolarised field or the total density). */
function spinOf(L) {
  const it = itemOf(L);
  if (L.w) return it.wave.nspin === 2 ? L.w.s : -1;
  const f = it.fields[L.fi], n = f ? f.name : "";
  return /^spin up\b/.test(n) ? 0 : /^spin down\b/.test(n) ? 1 : -1;
}
/** [positive, negative] colours for a field layer: by spin channel where it has one (up warm, down blue), else the
 *  first pair of FIELD_COLORS that no other surface uses. */
function pickColors(L) {
  const f = itemOf(L).fields[L.fi];
  if (!L.w && f && /^magnetization/.test(f.name)) return [SPIN_COLORS[0][0][0], SPIN_COLORS[1][0][0]];
  const s = spinOf(L), list = s >= 0 ? SPIN_COLORS[s] : FIELD_COLORS;
  const used = new Set(fieldLayers().filter(l => l !== L).map(l => l.pos));
  return (list.find(c => !used.has(c[0])) || list[fieldLayers().indexOf(L) % list.length]).slice();
}
function autoColors(L) { if (L.auto) [L.pos, L.neg] = pickColors(L); }

/* ------------------------------------------------------------------ WAVECAR (one band per field layer, in L.w) */
/** Show band b in layer L, at the layer's own spin, k-point and mode (L.w); the orbital lives in field slot L.fi. */
async function showBand(L, b) {
  const it = itemOf(L), w = it.wave, { s, k, mode, half } = L.w;
  L.w.b = b;
  busy(true);
  try {
    const { field } = await call("orbital", { ds: it.ds, slot: L.fi, s, k, b, signed: mode === "psi" && w.type === "gam", half });
    it.fields[L.fi] = field;
    initLevel(L, true);
    autoColors(L);
    if (L === activeField()) { syncFieldUI(); markBand(L); }
    buildLayersUI();
    requestSurface(L);
  } catch (e) { toast(e.message || String(e), true); }
  finally { busy(false); }
}
function syncWaveUI(L, it) {
  const w = it.wave;
  $("row-spin").hidden = w.nspin < 2; $("row-k").hidden = w.nk < 2;
  $("row-wmode").hidden = $("row-half").hidden = w.type !== "gam";
  $("kpt").innerHTML = w.kpts.map((k, i) => `<option value="${i}">${i + 1}: ${k.map(x => x.toFixed(3)).join(" ")}</option>`).join("");
  $("kpt").value = L.w.k;
  if ($("bands").dataset.item !== String(it.id) || $("bands").dataset.sk !== `${L.w.s}|${L.w.k}`) buildBands(L, it);
  markBand(L); segState();
}
function buildBands(L, it) {
  const w = it.wave, { s, k } = L.w, e = w.eig[s][k], o = w.occ[s][k], fr = w.frontier[s][k], list = $("bands"), rows = [];
  for (let b = 0; b < w.nb; b++) {
    const cls = ["band", o[b] > 0.5 ? "occ" : "", b === fr.homo ? "homo" : "", b === fr.lumo ? "lumo" : ""].join(" ");
    rows.push(`<div class="${cls}" role="option" data-b="${b}" aria-selected="false"><span>${b + 1}</span><span class="e">${e[b].toFixed(3)}</span><span class="o">${o[b].toFixed(2)}</span></div>`);
  }
  list.innerHTML = rows.join("");
  list.dataset.item = String(it.id); list.dataset.sk = `${s}|${k}`;
  const h = list.querySelector(`[data-b="${Math.max(0, L.w.b >= 0 ? L.w.b : fr.homo)}"]`);
  if (h) list.scrollTop = h.offsetTop - list.clientHeight / 2;
}
/** Select the active layer's band in the list, and mark (in its surface colour) every band of this spin and k-point
 *  that a visible layer of the same WAVECAR shows. */
function markBand(L) {
  for (const el of $("bands").querySelectorAll('[aria-selected="true"], .shown')) { el.setAttribute("aria-selected", "false"); el.classList.remove("shown"); el.style.removeProperty("--dot"); }
  for (const l of fieldLayers(true)) {
    if (!l.w || l.item !== L.item || l.w.s !== L.w.s || l.w.k !== L.w.k) continue;
    const el = $("bands").querySelector(`[data-b="${l.w.b}"]`);
    if (el) { el.classList.add("shown"); el.style.setProperty("--dot", l.pos); }
  }
  const el = $("bands").querySelector(`[data-b="${L.w.b}"]`);
  if (el) { el.setAttribute("aria-selected", "true"); if (el.scrollIntoView) el.scrollIntoView({ block: "nearest" }); }
}
const activeWave = () => { const L = activeField(); const it = L && itemOf(L); return it && it.wave && L.w ? { L, it } : null; };
/** Another surface layer from the same file as layer `from`: band b (default: the same band) of a WAVECAR, or the
 *  next channel no layer shows yet of a multi-field file (CHGCAR: total, magnetization, spin up, spin down). The new
 *  layer keeps the enclosed share of `from` and becomes the active one. */
async function addSurfaceLayer(from, b = null) {
  const it = itemOf(from);
  if (!it || (!from.w && it.fields.length < 2)) return;
  const L = { id: nextLayer++, item: it.id, kind: "field", visible: true, fi: from.fi, pm: from.pm, opacity: from.opacity, pos: from.pos, neg: from.neg,
    cuts: [], mesh: null, p: from.p, mode: from.mode, iso: from.iso, auto: true };
  if (from.w) { L.w = { ...from.w }; L.fi = it.nextSlot++; }
  else {
    const shown = new Set(fieldLayers().filter(l => l.item === it.id).map(l => l.fi)), free = it.fields.findIndex((_, i) => !shown.has(i));
    L.fi = free >= 0 ? free : from.fi;
  }
  S.layers.splice(S.layers.indexOf(from) + 1, 0, L);
  S.active = L.id;
  if (L.w) { buildLayersUI(); await showBand(L, b ?? Math.max(0, from.w.b)); }
  else { initLevel(L, true); autoColors(L); buildLayersUI(); syncFieldUI(); requestSurface(L); }
}

/* ------------------------------------------------------------------ layers panel */
function buildLayersUI() {
  const box = $("layers"), aa = activeAtomsLayer();
  box.innerHTML = S.layers.map(L => {
    const it = itemOf(L), f = L.kind === "field" ? it.fields[L.fi] : null;
    const kind = L.kind === "atoms" ? `${it.structure.species.length} atoms${it.traj ? ` · ${it.traj.n} frames` : ""}` : f ? f.name : it.wave ? "orbital" : "field";
    const signed = f && f.kind === "signed";
    const sw = L.kind === "field" ? `<button type="button" class="cpos" style="--c:${L.pos}" aria-label="Surface colour" title="Surface colour" aria-haspopup="dialog"></button>${signed ? `<button type="button" class="cneg" style="--c:${L.neg}" aria-label="Negative surface colour" title="Negative surface colour" aria-haspopup="dialog"></button>` : ""}` : "";
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
  updateSymmetry();
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
  else if (e.target.closest(".cpos, .cneg")) { const b = e.target.closest(".cpos, .cneg"); pickSurfaceColor(L, b.classList.contains("cpos") ? "pos" : "neg", b); }
  else if (L.kind === "field" && !e.target.closest("input")) { S.active = L.id; buildLayersUI(); syncFieldUI(); $("sec-surface").open = true; }
  else if (L.kind === "atoms") { S.activeAtoms = L.id; buildLayersUI(); }
});
/** Surface colour of layer L (which: "pos" or "neg") from the in-page picker at swatch b, live. */
function pickSurfaceColor(L, which, b) {
  const auto = pickColors(L);
  openPicker(b, { value: L[which], def: auto[which === "pos" ? 0 : 1], title: which === "pos" ? "Surface" : "Negative surface",
    onInput: c => {
      L[which] = c; L.auto = L.pos === auto[0] && L.neg === auto[1];
      b.style.setProperty("--c", c); restyleLayer(L);
      const aw = activeWave(); if (aw) markBand(aw.L);
      saveSoon();
    } });
}
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
    if (S.bz) setBZ(false);
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
  if (S.bz) setBZ(false);
  for (const it of S.items.values()) if (it.ds) call("drop", { ds: it.ds }).catch(() => {});
  S.items.clear(); S.layers = []; S.active = null; S.activeAtoms = null; S.sel = []; S.region = null; S.labels = []; S.undo = [];
  if (PH.on) setPhoto(false);
  // the analysis window shows the scene's trajectory: it goes with the scene (a later answer is dropped by seq)
  $("anwin").hidden = true; AN.res = null; AN.seq++;
  S.srcUrl = null; // web address of the scene's file, when it came from one (?url=, samples)
  S.frame = 0; S.playing = false; S.hide = new Set(); S.colors = new Map(); S.atomColors = new Map();
  S.t = [0, 0, 0]; ["fx", "fy", "fz"].forEach(id => ($(id).value = 0));
  S.reps = [1, 1, 1]; ["ra", "rb", "rc"].forEach(id => ($(id).value = 1));
  // a new scene starts from the default bond reach unless "keep" (next to it) is on
  if (!$("tol-keep").checked) { S.tol = TOL0; $("tol").value = TOL0; $("tol-out").textContent = TOL0.toFixed(2); }
  if (R) R.clearMeshes();
  setLabelMode(false); setBoxMode(false);
  if (SL.on) setSliceOn(false);
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
  const it = { id: nextItem++, name: d.name, ds: d.ds, structure: d.structure, wave: d.wave || null, traj: d.traj || null, fields: d.fields || [], nextSlot: (d.fields || []).length,
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
    FL = { id: nextLayer++, item: it.id, kind: "field", visible: true, fi: 0, pm: true, opacity: 0.85, pos: "", neg: "", cuts: [], mesh: null, auto: true };
    // a WAVECAR layer shows one band (L.w) in a field slot of its own after the file's fields
    if (it.wave) { FL.w = { s: 0, k: 0, b: -1, mode: "density", half: "x" }; FL.fi = it.nextSlot++; }
    S.layers.push(FL);
    S.active = FL.id;
    [FL.pos, FL.neg] = pickColors(FL);
    if (it.fields.length && !it.wave) initLevel(FL, false);
  }
  S.atlas = atlasOf(atlasSpecies());
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
  if (EMBED || !it.files || !it.files.length) return;
  const used = [...S.items.values()].reduce((a, o) => a + (o.kept ? o.files.reduce((b, f) => b + f.size, 0) : 0), 0);
  const size = it.files.reduce((a, f) => a + f.size, 0);
  if (used + size > store.LIMIT) return;
  it.kept = true;
  store.putFiles(it.key, it.files).then(r => { if (!r) it.kept = false; });
}
function saveSoon() {
  if (restoring || EMBED) return;
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
    items: items.map(it => ({ key: it.key, name: it.name, example: it.example, kept: it.kept, hadFiles: !!(it.files && it.files.length), dt: it.dtUser || null,
      structure: it.structure && (!it.kept && !it.example || it.structure.ids) && !(it.traj && it.kept) ? it.structure : null,
      ids: it.traj && it.structure && it.structure.ids ? it.structure.ids : null })),
    layers: S.layers.map(L => ({ item: index.get(L.item), kind: L.kind, visible: L.visible, ...(L.kind === "field" ? { fi: L.fi, iso: L.iso, p: L.p, mode: L.mode, pm: L.pm, opacity: L.opacity, pos: L.pos, neg: L.neg, cuts: L.cuts, auto: !!L.auto, w: L.w } : {}) })),
    view: { style: S.style, look: S.look, scale: S.scale, bonds: S.bonds, tol: S.tol, cell: S.cell, reps: S.reps.slice(), boundary: S.boundary, res: S.res, proj: S.proj, t: S.t.slice(),
      frame: S.frame, fps: S.fps, step: S.step, loop: S.loop, hide: [...S.hide], colors: [...S.colors],
      atomColors: [...S.atomColors].map(([k, c]) => { const [it, id] = k.split(":").map(Number); return [index.get(it), id, c]; }).filter(a => a[0] !== undefined), active: S.layers.findIndex(L => L.id === S.active), activeAtoms: S.layers.findIndex(L => L.id === S.activeAtoms),
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
    Object.assign(S, { style: st.view.style, look: LOOKS[st.view.look] ? st.view.look : S.look, scale: st.view.scale, bonds: st.view.bonds, tol: st.view.tol, cell: st.view.cell, reps: st.view.reps.slice(), boundary: st.view.boundary,
      res: st.view.res, proj: st.view.proj, t: st.view.t.slice(), fps: st.view.fps || 12, step: st.view.step || 1, loop: st.view.loop || "loop", hide: new Set(st.view.hide || []), colors: new Map(st.view.colors || []) });
    syncControls();
    for (const sp of st.items) {
      let d = null;
      try {
        if (sp.example === "h3d") d = await call("example", { name: "h3d" });
        // WHY: Undo of New scene has every file in memory, also those too large (or unable) to be kept in storage
        else if (sp.kept || !stored) { const f = await files(sp.key); if (f && f.length) d = { ...(await call("open", { files: f })), files: f }; }
        if (!d && sp.structure) d = { ds: null, structure: sp.structure, fields: [], wave: null, traj: null, notes: [] };
      } catch (e) { d = null; }
      if (!d) { missing.push(sp.name); made.push(null); continue; }
      await addItem({ ...d, name: sp.name, key: sp.key, example: sp.example, alreadyKept: stored && sp.kept }, true, true);
      const it = [...S.items.values()].pop();
      if (sp.structure && !it.traj) it.structure = sp.structure;
      if (sp.dt > 0) it.dtUser = sp.dt;
      if (it.traj && sp.ids) { const n = sp.ids.length, ids = sp.ids; it.structure = { ...it.structure, species: Array.from(ids, id => it.traj && d.structure.species[id - 1]), ids, pos: new Float64Array(3 * n) }; }
      if (sp.hadFiles && !sp.kept) missing.push(sp.name + " (atoms only: its files were too large to keep)");
      made.push(it);
    }
    // layers in the stored order, with their settings; layers the user had removed stay removed
    const order = [];
    for (const ls of st.layers) {
      const it = made[ls.item];
      if (!it) continue;
      let L = S.layers.find(l => l.item === it.id && l.kind === ls.kind && !order.includes(l));
      // a further surface of the same file (another band or channel) gets a layer of its own
      if (!L && ls.kind === "field" && (it.wave || it.fields.length > 1)) {
        L = { id: nextLayer++, item: it.id, kind: "field", visible: true, fi: 0, pm: true, opacity: 0.85, pos: "", neg: "", cuts: [], mesh: null };
        if (it.wave) { L.w = { s: 0, k: 0, b: -1, mode: "density", half: "x" }; L.fi = it.nextSlot++; }
      }
      if (!L) continue;
      L.visible = ls.visible;
      if (L.kind === "field") {
        Object.assign(L, { iso: ls.iso, p: ls.p, mode: ls.mode, pm: ls.pm, opacity: ls.opacity, pos: ls.pos, neg: ls.neg, cuts: ls.cuts || [], auto: !!ls.auto });
        if (!L.w) L.fi = Math.min(ls.fi, Math.max(0, it.fields.length - 1));
        else { const sw = ls.w || st.items[ls.item].w; if (sw) L.want = { ...sw }; }  // st.items[].w: scenes saved before v18
      }
      order.push(L);
    }
    for (const L of S.layers) if (!order.includes(L) && L.kind === "field") R.removeMeshGroup(L.id);
    S.layers = order;
    const at = i => (i >= 0 && order[i] ? order[i].id : null);
    S.active = at(st.view.active) ?? (fieldLayers()[0] || {}).id ?? null; S.activeAtoms = at(st.view.activeAtoms);
    S.frame = st.view.frame || 0;
    for (const it of trajItems()) applyFrame(it);
    S.atomColors = new Map((st.view.atomColors || []).filter(a => made[a[0]]).map(([i, id, c]) => [`${made[i].id}:${id}`, c]));
    S.labels = (st.view.labels || []).filter(l => order[l.layer]).map(l => ({ i: -1, layer: order[l.layer].id, src: l.src, p: l.p }));
    if (!S.layers.length) { showHome(); return false; }
    for (const L of order) if (L.kind === "field" && L.w) {
      const want = L.want, w = L.w;
      delete L.want;
      if (want && want.b >= 0 && (want.b !== w.b || want.s !== w.s || want.k !== w.k || want.mode !== w.mode || want.half !== w.half || !itemOf(L).fields[L.fi])) { L.w = { ...want }; await showBand(L, want.b); }
    }
    S.atlas = atlasOf(atlasSpecies());
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
  $("dock").hidden = !has; $("br").hidden = !has; $("bl").hidden = !has; $("doc").hidden = !has; $("btn-new").hidden = !has; $("btn-photo").hidden = !has;
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
  syncSlice();
  if (AN.kind === "xrd") {
    // XRD: closes without a crystal; recomputed when the structure was replaced (another file, an edit, a new cell)
    const xi = xrdItem();
    if (!xi) $("anwin").hidden = true;
    else if (!$("anwin").hidden && AN.res && AN.res.st !== xi.structure && !(xi.traj && AN.res.it === xi)) runAnalysis();
  } else if (!mainTraj()) $("anwin").hidden = true;
  else if (!$("anwin").hidden && AN.res && AN.res.it && AN.res.it !== mainTraj()) runAnalysis();
  syncDt();
  const pc = primaryCell();
  $("sum-structure").textContent = !atomLayers(false).length ? "" : S.style === "off" ? "hidden" : `${S.style}${S.look !== LOOK0 ? " · " + S.look : ""}${pc && pc.pbc ? " · " + S.reps.join("×") : ""}${S.hide.size ? ` · no ${[...S.hide].join(", ")}` : ""}`;
  const L = activeField(), f = activeStats();
  $("sum-surface").textContent = !L || !f ? "" : !L.visible ? "hidden" : L.mode === "log" ? `${fmt(L.iso)} · ${Math.round(100 * fractionAbove(f.stats, L.iso))}%` : fmt(L.iso);
  const aw = activeWave();
  if (aw && aw.L.w.b >= 0) { const w = aw.L.w, n = fieldLayers().filter(l => l.w && l.item === aw.it.id).length; $("sum-orbital").textContent = `${w.b + 1}${aw.it.wave.nspin === 2 ? (w.s ? "↓" : "↑") : ""} · ${aw.it.wave.eig[w.s][w.k][w.b].toFixed(2)} eV${n > 1 ? ` · ${n} bands` : ""}`; }
  $("sum-view").textContent = S.proj === "ortho" ? "ortho" : "perspective";
}
function segState() {
  const aw = activeWave(), w = aw ? aw.L.w : { s: 0, mode: "density", half: "x" };
  const val = { style: S.style, look: S.look, res: S.res, wspin: String(w.s), wmode: w.mode, half: w.half, proj: S.proj, theme: S.theme, loop: S.loop, mfmt: S.movie.fmt, msize: String(S.movie.scale) };
  for (const seg of document.querySelectorAll(".seg")) for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === val[seg.dataset.key]));
}

/* ------------------------------------------------------------------ home page, examples, image */
function showHome() {
  // embedded: the wordmark opens the full site (with the same file when it came from a URL)
  if (EMBED) { window.open(location.origin + location.pathname + (qp.get("url") ? "?url=" + encodeURIComponent(qp.get("url")) : ""), "_blank", "noopener"); return; }
  $("empty").hidden = false;
  const back = $("btn-back");
  back.hidden = !S.layers.length;
  if (S.layers.length) back.textContent = `Back to ${$("doc-name").textContent}`;
}
/** Samples on the opening page (examples/, see examples/README.md for each file's source); h3d is computed. An entry
 *  may set a supercell (reps) and open a slice (slice: plane hkl, position t, scale, main view). */
const SAMPLES = {
  rutile: "rutile-TiO2.cif", sto: "SrTiO3.vasp", zno: "ZnO-wurtzite.xsf", graphene: { file: "graphene.pw.in", reps: [2, 2, 1] }, nacl: "NaCl.geometry.in", c60: "C60.xyz",
  ethanol: "ethanol.pdb", si: "Si-model.CHGCAR",
  // WHY: the (110) plane at t = 0.5 holds the zigzag Si-Si chains (x + y = a); two cells along c show two whole chains
  sislice: { file: "Si-model.CHGCAR", reps: [1, 1, 2], slice: { hkl: [1, 1, 0], t: 0.5, scale: "lin", main: "2d" } }, elf: "C4.ELFCAR.gz", locpot: "Li.LOCPOT.gz", n2: "N2.WAVECAR", benzene: "benzene-pi.cube",
  vasprun: "LGPS-800K.vasprun.xml.gz", outcar: "LiPON-relax.OUTCAR.gz", lisi: "LiSi-md.XDATCAR", pwmd: "Si8-md.pw.out", castep: "Si-relax.geom",
  cu55: "Cu55-900K.traj", pt38: "Pt38-relax.extxyz", al: "Al-800K.XDATCAR", ag: "Ag-1600K.lammpstrj", ni: "Ni100-adatom-1100K.axsf",
};
async function example(id) {
  busy(true);
  try {
    if (id === "h3d") {
      if (!(await ensureRenderer())) return;
      await addItem({ name: "hydrogen 3d(x²−y²), computed here", ...(await call("example", { name: "h3d" })), example: "h3d" }, false);
      return;
    }
    // WHY: an entry is a file name or an object; a string has its own .slice method, so normalise before reading options
    const spec = typeof SAMPLES[id] === "string" ? { file: SAMPLES[id] } : SAMPLES[id], name = spec && spec.file;
    if (!name) throw new Error(`There is no sample called "${id}".`);
    const r = await fetch("examples/" + name);
    if (!r.ok) throw new Error(`Could not load the sample (${r.status}).`);
    await openFiles([new File([await r.blob()], name)], "replace");
    if (S.layers.length) S.srcUrl = new URL("examples/" + name, SITE).href; // for the embed card
    if (spec.reps) {
      // a sample shown as a supercell (graphene 2 x 2)
      S.reps = spec.reps.slice(); ["ra", "rb", "rc"].forEach((k, i) => ($(k).value = S.reps[i]));
      buildStructure(); fit(); refreshSurfaces(); summaries(); requestRender();
    }
    if (spec.slice && S.layers.length) {
      const c = spec.slice;
      Object.assign(SL, { hkl: c.hkl.slice(), t: c.t, scale: c.scale, atoms: true });
      $("sl-hkl").value = c.hkl.join(" "); $("sl-t").value = c.t; $("sl-atoms").checked = true;
      SV.main = c.main;
      setSliceOn(true);
    }
  } catch (e) { toast(e.message || String(e), true); }
  finally { busy(false); }
}
function restyle() { if (!R) return; readPalette(); R.setBackground(liveBg()); if (BZI.R) BZI.R.setBackground(liveBg()); if (SL.on) drawSliceViews(); requestRender(); }

/* ------------------------------------------------------------------ corner eyes: tuck away the group in each corner */
const CORNERS = { tl: "tl", tr: "dock", bl: "bl", br: "br" };
const CORNER_NAMES = { tl: "the title", tr: "the panel", bl: "the axes and cell", br: "the view buttons" };
const EYE = '<svg viewBox="0 0 24 24" aria-hidden="true"><g class="open"><g class="lid"><path d="M2.8 12s3.4-6 9.2-6 9.2 6 9.2 6-3.4 6-9.2 6-9.2-6-9.2-6z" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="12" cy="12" r="2.4" fill="currentColor"/></g></g>' +
  '<g class="shut" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M3.4 10.6c2.4 2.9 5.3 4.4 8.6 4.4s6.2-1.5 8.6-4.4"/><path d="m6.3 13.4-1.4 1.9M12 15v2.3M17.7 13.4l1.4 1.9"/></g></svg>';
function tuck(c, on) {
  $(CORNERS[c]).classList.toggle("tucked", on);
  if (SL.on) drawSliceViews(); // the 2D view keeps clear of the panel
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

/* ------------------------------------------------------------------ search options (Ctrl/Cmd F) */
// Every option: t = its name, sel = the control, k = other words people use for it, self = highlight the control
// itself (toolbar buttons) rather than its row, before = make it reachable first (label mode for Label all).
const OPTIONS = [
  { t: "Layers", sel: "#layers", k: "layer files show hide remove colour color superimpose" },
  { t: "Add files to the scene", sel: "#btn-add", k: "add open another superimpose overlay combine" },
  { t: "Primitive cell", sel: "#t-prim", k: "primitive reduce reduced cell symmetry supercell niggli" },
  { t: "Conventional cell", sel: "#t-conv", k: "conventional standard cell symmetry bravais" },
  { t: "Symmetry tolerance", sel: "#t-tol", k: "tolerance symprec space group precision" },
  { t: "Export as shown (POSCAR, CIF, XYZ)", sel: ".export", k: "export save download write poscar cif xyz file", self: true },
  { t: "Draw a region", sel: "#e-region", k: "lasso region select selection area freehand" },
  { t: "Delete atoms", sel: "#e-delete", k: "delete remove atoms vacancy erase" },
  { t: "Cut surface", sel: "#e-cut", k: "cut slice clip remove surface isosurface" },
  { t: "Restore the surface", sel: "#cuts-clear", k: "restore uncut surface cuts" },
  { t: "Center on atom", sel: "#e-center", k: "center centre middle pivot atom" },
  { t: "Shift origin", sel: "#fx", k: "shift translate move origin fractional offset wrap" },
  { t: "Undo", sel: "#e-undo", k: "undo revert back ctrl z" },
  { t: "Style", sel: '.seg[data-key="style"]', k: "ball stick space filling spacefill hide atoms representation" },
  { t: "Look", sel: '.seg[data-key="look"]', k: "look toon cartoon glossy soft outline fog shading render style paper" },
  { t: "Atom size", sel: "#scale", k: "size radius scale atoms" },
  { t: "Bonds", sel: "#bonds", k: "bonds on off" },
  { t: "Bond reach", sel: "#tol", k: "bond length cutoff distance bonding reach keep" },
  { t: "Cell outline", sel: "#cell", k: "cell box unit cell lattice outline frame" },
  { t: "Repeat (supercell)", sel: "#ra", k: "repeat supercell tile replicate periodic images" },
  { t: "Boundary atoms", sel: "#boundary", k: "boundary faces edge corner atoms wrap images" },
  { t: "Field", sel: "#field", k: "field dataset magnetization spin density channel" },
  { t: "Add a field as a layer", sel: "#add-field", self: true, k: "several fields both spin up down total magnetization overlay together simultaneously at once another channel" },
  { t: "Show surface", sel: "#surf", k: "surface isosurface show hide on off" },
  { t: "Level (isovalue)", sel: "#iso", k: "level isovalue iso threshold contour value density" },
  { t: "Negative lobe", sel: "#pm", k: "negative lobe sign phase minus" },
  { t: "Opacity", sel: "#opacity", k: "opacity transparency alpha translucent" },
  { t: "Resolution", sel: '.seg[data-key="res"]', k: "resolution quality grid stride smooth coarse" },
  { t: "Band", sel: "#bands", k: "band orbital wavefunction homo lumo state eigenvalue" },
  { t: "Spin", sel: '.seg[data-key="wspin"]', k: "spin up down channel" },
  { t: "Show elements", sel: "#species", k: "hide atoms element species checkbox show only remove view cations anions tick untick" },
  { t: "Add a band as a layer", sel: "#add-band", self: true, k: "several bands orbitals both spins up down hole electron overlay together simultaneously at once another wavefunction" },
  { t: "k-point", sel: "#kpt", k: "k-point kpoint kpt" },
  { t: "|ψ|² or ψ", sel: '.seg[data-key="wmode"]', k: "psi wavefunction phase sign density" },
  { t: "Play / pause", sel: "#pl-play", k: "play pause animate animation trajectory movie", self: true },
  { t: "Frame", sel: "#pl-frame", k: "frame step time slider trajectory", self: true },
  { t: "Speed", sel: "#tr-fps", k: "speed fps frames per second playback" },
  { t: "Step (every n-th frame)", sel: "#tr-step", k: "step stride skip every frame" },
  { t: "At the end (loop, bounce, stop)", sel: '.seg[data-key="loop"]', k: "loop bounce stop repeat end" },
  { t: "Energy per frame", sel: "#tr-spark", k: "energy plot graph trajectory" },
  { t: "Movie format", sel: '.seg[data-key="mfmt"]', k: "movie gif video mp4 webm format" },
  { t: "Movie speed", sel: "#mv-fps", k: "movie fps speed frames per second" },
  { t: "Movie frames", sel: "#mv-from", k: "movie frames range from to" },
  { t: "Movie size", sel: '.seg[data-key="msize"]', k: "movie size resolution width pixels" },
  { t: "Export movie", sel: "#mv-go", k: "export movie gif video mp4 webm record save animation" },
  { t: "All frames as XYZ", sel: "#tr-xyz", k: "export trajectory frames xyz save" },
  { t: "Projection", sel: '.seg[data-key="proj"]', k: "projection perspective orthographic ortho" },
  { t: "Theme", sel: '.seg[data-key="theme"]', k: "theme dark light mode colours colors" },
  { t: "Renderer", sel: "#gfx", k: "renderer webgpu webgl gpu" },
  { t: "Keyboard shortcuts", sel: "#sec-view .keys", k: "keyboard shortcuts keys help" },
  { t: "View along a, b, c", sel: '[data-along="0"]', k: "view along axis orient direction a b c", self: true },
  { t: "Label atoms", sel: "#btn-label", k: "label labels names symbols annotate tag element", self: true },
  { t: "Label all atoms", sel: "#label-all", k: "label all labels every atom names symbols", self: true, before: () => setLabelMode(true) },
  { t: "Reset view", sel: "#btn-reset", k: "reset view fit home recenter camera", self: true },
  { t: "Open files", sel: "#btn-open", k: "open load import file replace", self: true },
  { t: "X-ray diffraction (XRD pattern)", sel: "#xrd-open", k: "xrd x-ray diffraction powder pattern bragg peaks hkl 2theta cu k alpha reflections", self: true },
  { t: "Photo Mode: save an image of the view", sel: "#btn-photo", k: "photo save image screenshot png picture snapshot export copy transparent high resolution print", self: true },
  { t: "Embed in your website, use in Jupyter or Julia", sel: "#btn-use", k: "embed iframe website share jupyter julia notebook python pluto use elsewhere", self: true },
  { t: "New scene", sel: "#btn-new", k: "new scene clear start over reset", self: true },
  { t: "Brillouin zone", sel: "#bz", k: "brillouin zone reciprocal k-path kpath high symmetry points band structure bz wigner seitz" },
  { t: "Slice", sel: "#slice", k: "slice 2d cut plane section contour map hkl miller cross section" },
  { t: "Trajectory analysis", sel: "#an-open", k: "rdf radial distribution msd diffusion vacf vdos phonon density of states temperature analysis coordination" },
  { t: "Opening page and samples", sel: "#btn-home", k: "home opening page samples examples start", self: true },
  { t: "Space group", sel: "#doc-sg", k: "space group symmetry spacegroup hermann mauguin", self: true },
  { t: "Hide the panels", sel: '.corner-eye[data-corner="tr"]', k: "hide panels eye tuck corner H", self: true },
];
const WHY = { "sec-surface": "needs a volumetric file", "sec-orbital": "needs a WAVECAR", "sec-traj": "needs a trajectory", player: "needs a trajectory",
  dock: "open a file first", br: "open a file first", bl: "open a file first", "doc-sg": "needs a periodic structure", "btn-new": "open a file first", "e-cut": "needs a surface",
  "row-cuts": "after a cut", "row-tol": "with bonds on", "row-field": "with several fields", "row-addfield": "with several fields", "row-species": "needs atoms", "row-pm": "for signed fields", "mfmt-video": "this browser cannot record video",
  "row-spin": "for spin-polarised WAVECARs", "row-k": "with several k-points", "row-wmode": "for Γ-only WAVECARs", "row-half": "for Γ-only WAVECARs", "tr-spark": "when the file has energies" };
/** Why a control is not available now ("" when it is). */
function whyNot(el) {
  if (el.id === "label-all") return S.layers.length ? "" : "open a file first";
  const h = el.closest("[hidden]");
  if (!h) return "";
  if (h.classList.contains("pbc")) return "for periodic structures";
  if (h.classList.contains("has-cell")) return "needs a cell or box";
  return WHY[h.id] || "not available for this file";
}
const whereOf = el => {
  const sec = el.closest("details.sec");
  if (sec) return sec.querySelector("summary span").textContent;
  if (el.closest("#br")) return "View buttons"; if (el.closest("#player")) return "Player"; if (el.closest(".tl")) return "Title bar";
  if (el.closest("#labelbar")) return "Label bar"; if (el.closest(".corner-eye")) return "Corners"; return "";
};
let spotHits = [], spotAt = 0;
const reEsc = x => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
/** Score of an option for the query: every word must hit the name (best), its other words or its panel. */
function spotScore(o, q) {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean), t = o.t.toLowerCase(), k = o.k.toLowerCase(), w = (o.where || "").toLowerCase();
  let sc = 0;
  for (const x of words) {
    const b = new RegExp("(^|[^a-z0-9])" + reEsc(x));
    if (t.startsWith(x)) sc += 6; else if (b.test(t)) sc += 4; else if (t.includes(x)) sc += 3;
    else if (b.test(k)) sc += 2; else if (k.includes(x) || w.includes(x)) sc += 1; else return 0;
  }
  return sc || 1;
}
function spotRender() {
  const q = $("spot-q").value.trim();
  spotHits = OPTIONS.map(o => { const el = document.querySelector(o.sel); return el ? { ...o, el, where: whereOf(el), why: whyNot(el) } : null; })
    .filter(Boolean).map(o => ({ ...o, sc: spotScore(o, q) })).filter(o => o.sc > 0)
    .sort((a, b) => (!!a.why - !!b.why) || (b.sc - a.sc));
  spotAt = Math.min(spotAt, Math.max(0, spotHits.length - 1));
  const mark = t => { if (!q) return esc(t); const i = t.toLowerCase().indexOf(q.toLowerCase()); return i < 0 ? esc(t) : esc(t.slice(0, i)) + "<b>" + esc(t.slice(i, i + q.length)) + "</b>" + esc(t.slice(i + q.length)); };
  $("spot-list").innerHTML = spotHits.length ? spotHits.map((o, i) => `<li role="option" class="sr${o.why ? " off" : ""}" data-i="${i}" aria-selected="${i === spotAt}"><span class="st">${mark(o.t)}</span><span class="sw">${esc(o.where)}</span>${o.why ? `<span class="sy">${esc(o.why)}</span>` : ""}</li>`).join("")
    : `<li class="none">No option matches “${esc(q)}”.</li>`;
  const cur = $("spot-list").querySelector('[aria-selected="true"]');
  if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: "nearest" });
}
function openSearch() {
  $("spot").hidden = false; $("spot-q").value = ""; spotAt = 0; spotRender(); $("spot-q").focus();
}
function closeSearch() { $("spot").hidden = true; }
/** Go to an option: show its corner and panel, scroll to it, outline it for a moment and focus it. */
function goToOption(o) {
  closeSearch();
  if (o.before && S.layers.length) o.before();
  const el = document.querySelector(o.sel);
  if (!el) return;
  const why = whyNot(el);
  if (why) { toast(`${o.t}: ${why}.`, false, 3500); return; }
  for (const c in CORNERS) if ($(CORNERS[c]).contains(el)) tuck(c, false);
  const sec = el.closest("details.sec");
  if (sec) sec.open = true;
  // the highlight is an ::after overlay, which inputs cannot carry: their parent takes it
  let box = o.self ? el : el.closest(".row, .tool-row, .sel-line, .spark, .keys, .row-btns, .layers") || el;
  if (box.matches("input, select, textarea")) box = box.parentElement;
  box.classList.remove("found", "self");
  void box.offsetWidth;
  box.classList.add("found");
  box.classList.toggle("self", !!o.self);
  // WHY: 200 ms flash + 300 ms outline (the CSS animation), then nothing
  setTimeout(() => box.classList.remove("found", "self"), 520);
  if (box.scrollIntoView) box.scrollIntoView({ block: "center", behavior: "smooth" });
  const f = el.matches("input, select, button, [tabindex]") ? el : box.querySelector("input, select, button, [tabindex]");
  if (f && f.focus) f.focus({ preventScroll: true });
}
$("btn-search").addEventListener("click", openSearch);
$("spot-q").addEventListener("input", () => { spotAt = 0; spotRender(); });
$("spot-q").addEventListener("keydown", e => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); spotAt = Math.max(0, Math.min(spotHits.length - 1, spotAt + (e.key === "ArrowDown" ? 1 : -1))); spotRender(); }
  else if (e.key === "Enter") { e.preventDefault(); if (spotHits[spotAt]) goToOption(spotHits[spotAt]); }
  else if (e.key === "Escape") { e.preventDefault(); closeSearch(); }
});
$("spot-list").addEventListener("click", e => { const li = e.target.closest("[data-i]"); if (li) goToOption(spotHits[+li.dataset.i]); });
$("spot").addEventListener("pointerdown", e => { if (e.target === $("spot")) closeSearch(); });
{
  const mac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");
  for (const k of document.querySelectorAll(".search-kbd")) k.textContent = mac ? "⌘F" : "Ctrl F";
  $("btn-search").title = `Search options (${mac ? "⌘F" : "Ctrl+F"})`;
}

/* ------------------------------------------------------------------ wiring */
function bindSeg(key, fn) {
  document.querySelector(`.seg[data-key="${key}"]`).addEventListener("click", e => { const b = e.target.closest("button"); if (b) { fn(b.value); segState(); summaries(); } });
}
bindSeg("style", v => { S.style = v; buildStructure(); });
bindSeg("look", v => {
  S.look = v; requestRender();
  if (!EMBED) try { localStorage.setItem(store.PREFIX + "look", v); } catch (e) { /* storage unavailable */ }
});
bindSeg("res", v => { S.res = v; refreshSurfaces(); });
bindSeg("proj", v => { S.proj = v; cam.ortho = v === "ortho"; requestRender(); });
bindSeg("theme", v => {
  S.theme = v;
  if (v === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", v);
  try { localStorage.setItem(store.PREFIX + "theme", v); } catch (e) { /* storage unavailable */ }
  restyle();
});
bindSeg("loop", v => { S.loop = v; playDir = 1; });
bindSeg("mfmt", v => { S.movie.fmt = v; if (v === "video" && S.movie.scale < 0.5) S.movie.scale = 1; movieNote(); });
bindSeg("msize", v => { S.movie.scale = +v; movieNote(); });
bindSeg("wspin", v => { const aw = activeWave(); if (!aw) return; aw.L.w.s = +v; buildBands(aw.L, aw.it); showBand(aw.L, Math.max(0, aw.L.w.b)); });
bindSeg("wmode", v => { const aw = activeWave(); if (!aw) return; aw.L.w.mode = v; showBand(aw.L, Math.max(0, aw.L.w.b)); });
bindSeg("half", v => { const aw = activeWave(); if (!aw) return; aw.L.w.half = v; showBand(aw.L, Math.max(0, aw.L.w.b)); });
$("kpt").addEventListener("change", e => { const aw = activeWave(); if (!aw) return; aw.L.w.k = +e.target.value; buildBands(aw.L, aw.it); showBand(aw.L, Math.max(0, aw.L.w.b)); });
// a plain click shows the band in the active layer; Ctrl/Cmd or Shift click adds it as a layer of its own
$("bands").addEventListener("click", e => {
  const r = e.target.closest(".band"), aw = activeWave();
  if (!r || !aw) return;
  if (e.ctrlKey || e.metaKey || e.shiftKey) addSurfaceLayer(aw.L, +r.dataset.b); else showBand(aw.L, +r.dataset.b);
});
$("add-band").addEventListener("click", () => { const aw = activeWave(); if (aw) addSurfaceLayer(aw.L); });
$("add-field").addEventListener("click", () => { const L = activeField(); if (L && !L.w) addSurfaceLayer(L); });
$("bands").addEventListener("keydown", e => {
  const aw = activeWave();
  if (!aw || (e.key !== "ArrowDown" && e.key !== "ArrowUp")) return;
  e.preventDefault();
  const b = Math.min(aw.it.wave.nb - 1, Math.max(0, aw.L.w.b + (e.key === "ArrowDown" ? 1 : -1)));
  showBand(aw.L, b);
  const el = $("bands").querySelector(`[data-b="${b}"]`); if (el) el.scrollIntoView({ block: "nearest" });
});
$("scale").addEventListener("input", e => { S.scale = +e.target.value; buildStructure(); });
$("bonds").addEventListener("change", e => { S.bonds = e.target.checked; $("row-tol").hidden = !S.bonds; buildStructure(); });
$("tol").addEventListener("input", e => { S.tol = +e.target.value; $("tol-out").textContent = S.tol.toFixed(2); buildStructure(); });
$("tol-keep").addEventListener("change", e => { try { localStorage.setItem(store.PREFIX + "keep-reach", e.target.checked ? "1" : "0"); } catch (err) { /* storage unavailable */ } });
try { $("tol-keep").checked = localStorage.getItem(store.PREFIX + "keep-reach") === "1"; } catch (e) { /* storage unavailable */ }
$("cell").addEventListener("change", e => { S.cell = e.target.checked; buildStructure(); });
$("species").addEventListener("change", e => {
  const el = e.target.dataset.el;
  if (!el) return;
  if (e.target.checked) S.hide.delete(el); else S.hide.add(el);
  buildStructure(); summaries();
});
// an element's colour: the in-page picker, live; Default (or the default colour itself) removes the choice
$("species").addEventListener("click", e => {
  const b = e.target.closest(".sp-c");
  if (!b) return;
  const el = b.dataset.el;
  openPicker(b, { value: elColor(el), def: colorOf(el), title: `${el} atoms`,
    onInput: c => { if (c === colorOf(el)) S.colors.delete(el); else S.colors.set(el, c); recolor(); } });
});
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
$("t-tol").addEventListener("change", updateSymmetry);
$("t-conv").addEventListener("click", () => derive("conventional"));
for (const b of document.querySelectorAll("[data-fmt]")) b.addEventListener("click", () => exportLayer(b.dataset.fmt));
$("field").addEventListener("change", e => { const L = activeField(); if (!L) return; L.fi = +e.target.value; initLevel(L, true); autoColors(L); syncFieldUI(); buildLayersUI(); requestSurface(L); });
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
  else if (a === "color") {
    // the selected atoms (every periodic image); Default gives each its element's colour again
    const s0 = S.sel[0], a0 = s0 && disp.atoms[s0.i], keys = S.sel.map(s => { const L = layerById(s.layer); return L ? atomKey(L, s.src) : null; }).filter(Boolean);
    if (!a0) return;
    const def = elColor(a0.el), n = keys.length;
    openPicker(b, { value: a0.color, def, title: n === 1 ? `${atomLabel(s0)}` : `${n} atoms`,
      onInput: c => { for (const k of keys) { if (c === def) S.atomColors.delete(k); else S.atomColors.set(k, c); } recolor(); } });
  }
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
$("label-all").addEventListener("click", labelAll);
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
  // Ctrl/Cmd F opens the option search from anywhere (also inside inputs), instead of the browser's find
  if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === "f") { e.preventDefault(); if ($("spot").hidden) openSearch(); else $("spot-q").select(); return; }
  if (e.target && e.target.closest && e.target.closest("input, select, textarea, .bands")) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z" && !e.shiftKey) { e.preventDefault(); undo(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (k === "o") pick("replace");
  else if (k === "escape") {
    if (PH.on) setPhoto(false);
    else if (!$("spot").hidden) closeSearch();
    else if (!$("empty").hidden && S.layers.length) $("empty").hidden = true;
    else if (S.labelMode) setLabelMode(false);
    else if (S.sel.length || S.region) clearSelection();
    else if (S.boxMode) setBoxMode(false);
    else for (const d of document.querySelectorAll(".sec[open]")) d.open = false;
  }
  else if (!S.layers.length || !$("empty").hidden) return;
  else if ((k === "delete" || k === "backspace") && S.sel.length) { e.preventDefault(); deleteSelected(); }
  else if (k === "r") resetView();
  else if (k === "s") { if (PH.on) savePhoto(); else setPhoto(true); }
  else if (k === "l") setLabelMode(!S.labelMode);
  else if (k === "k" && primaryCell() && primaryCell().pbc) setBZ(!S.bz);
  else if (k === "h") tuckAll();
  else if (k === " " && maxFrames() > 1) { e.preventDefault(); play(!S.playing); }
  else if ((k === "arrowright" || k === "arrowleft") && maxFrames() > 1) { e.preventDefault(); play(false); setFrame(S.frame + (k === "arrowright" ? 1 : -1) * S.step * (e.shiftKey ? 10 : 1)); }
  else if ((k === "home" || k === "end") && maxFrames() > 1) { e.preventDefault(); play(false); setFrame(k === "home" ? 0 : maxFrames() - 1); }
  else if (k.length === 1 && "abc".includes(k)) viewAlong("abc".indexOf(k));
});
// WHY: the 2D figure keeps clear of the panel, so opening or closing a section (the panel grows or shrinks) lays it out again
try { new ResizeObserver(() => { if (SL.on && slMode() === "2d") drawSliceViews(); }).observe($("dock")); } catch (e) { /* no ResizeObserver */ }
try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (S.theme === "auto") restyle(); }); } catch (e) { /* old browsers */ }
/* test site only (store.CHANNEL): typeface pickers for text and for numbers and file names (--sans, --mono); notes say
   which characters isoview prints that a face lacks (they come from another font). Choices are kept in this browser. */
const TEST_FONTS = [{"id":"inter","label":"Inter","kind":"text","family":"Inter","note":""},{"id":"source","label":"Source Sans 3","kind":"text","family":"isoview S3","note":""},{"id":"plex","label":"IBM Plex Sans","kind":"text","family":"isoview PX","note":"no ⁻"},{"id":"fira","label":"Fira Sans","kind":"text","family":"Fira Sans","note":""},{"id":"lato","label":"Lato","kind":"text","family":"isoview LT","note":""},{"id":"noto","label":"Noto Sans","kind":"text","family":"Noto Sans","note":"no arrows"},{"id":"opensans","label":"Open Sans","kind":"text","family":"Open Sans","note":"no ⁻"},{"id":"roboto","label":"Roboto","kind":"text","family":"Roboto","note":"no overbar"},{"id":"manrope","label":"Manrope","kind":"text","family":"Manrope","note":"no overbar"},{"id":"commissioner","label":"Commissioner","kind":"text","family":"Commissioner","note":"no ⁻"},{"id":"sourcecode","label":"Source Code Pro","kind":"mono","family":"isoview SCP","note":"no ⁻"},{"id":"jetbrains","label":"JetBrains Mono","kind":"mono","family":"JetBrains Mono","note":"no ⁻"},{"id":"plexmono","label":"IBM Plex Mono","kind":"mono","family":"isoview PXM","note":"no Greek"},{"id":"firamono","label":"Fira Mono","kind":"mono","family":"Fira Mono","note":""},{"id":"notomono","label":"Noto Sans Mono","kind":"mono","family":"Noto Sans Mono","note":""},{"id":"robotomono","label":"Roboto Mono","kind":"mono","family":"Roboto Mono","note":"no subscripts"}];
if (store.CHANNEL) {
  const SYS = { text: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', mono: 'ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace' };
  const DEF = { text: "inter", mono: "sourcecode" }; // WHY: the live site's faces, so the test site starts where live is
  const pick = (kind, id, save) => {
    const f = TEST_FONTS.find(x => x.kind === kind && x.id === id);
    // mono "text": numbers and file names in the text face, with tabular (equal-width) digits everywhere
    const v = kind === "mono" && id === "text" ? "var(--sans)" : f ? `"${f.family}", ${SYS[kind]}` : SYS[kind];
    if (kind === "mono") document.documentElement.dataset.mono = id;
    document.documentElement.style.setProperty(kind === "text" ? "--sans" : "--mono", v);
    if (kind === "text") document.body.style.fontFeatureSettings = id === "inter" ? '"cv05"' : ""; // Inter: l with a tail
    if (save) try { localStorage.setItem(store.PREFIX + "font-" + kind, id); } catch (e) { /* storage unavailable */ }
    if (document.fonts) document.fonts.load(`600 16px ${v.startsWith("var(") ? css("--sans") : v}`).then(() => { restyle(); if (R && S.layers.length) { S.atlas = atlasOf(atlasSpecies()); buildStructure(); } }, () => {});
  };
  for (const kind of ["text", "mono"]) {
    const sel = $("tb-" + kind);
    sel.innerHTML = [`<option value="system">System</option>`, ...(kind === "mono" ? [`<option value="text">Same as text</option>`] : []), ...TEST_FONTS.filter(f => f.kind === kind).map(f => `<option value="${f.id}">${esc(f.label)}${f.note ? " (" + esc(f.note) + ")" : ""}</option>`)].join("");
    let id = DEF[kind];
    try { const v = localStorage.getItem(store.PREFIX + "font-" + kind); if (v === "system" || (kind === "mono" && v === "text") || TEST_FONTS.some(f => f.kind === kind && f.id === v)) id = v; } catch (e) { /* storage unavailable */ }
    sel.value = id; pick(kind, id, false);
    sel.addEventListener("change", () => pick(kind, sel.value, true));
  }
  $("testbar").hidden = false;
}
try { const t = localStorage.getItem(store.PREFIX + "theme"); if (t === "light" || t === "dark") { S.theme = t; document.documentElement.setAttribute("data-theme", t); } } catch (e) { /* storage unavailable */ }

/* ------------------------------------------------------------------ Brillouin zone */
const BZ_SIZE = 5; // drawing units for the largest |k| of the zone (the camera works best with scenes of a few units)
/** k-point label as shown: G -> Γ, Sigma -> Σ, X_1 -> X₁. */
const kLabel = s => s.replace(/^G$/, "Γ").replace("Sigma", "Σ").replace(/_(\d)/, (m, d) => "₀₁₂₃₄₅₆₇₈₉"[d]);
/** Turns the Brillouin-zone view on or off (keeps and restores the crystal camera). */
function setBZ(on) {
  if (on) {
    const L = activeAtomsLayer(), st = L && itemOf(L).structure;
    const tol = Math.max(1e-5, parseFloat($("t-tol").value) || 1e-5);
    let k = null;
    try { k = st && kpath(st, tol); } catch (e) { console.error(e); }
    if (!k) { $("bz").checked = false; toast("The Brillouin zone needs a periodic structure.", true); return; }
    S.bzData = { k, zone: wignerSeitz(k.B), name: itemOf(L).name, layer: L.id };
    S.bzCam = { q: cam.q.slice(), target: cam.target.slice(), dist: cam.dist, radius: cam.radius };
    S.bz = true;
  } else {
    S.bz = false; S.bzData = null;
    if (R) { R.removeMeshGroup("bz"); R.setLabels && uploadLabels(); }
    buildInset();
  }
  $("bz").checked = S.bz;
  buildStructure(); refreshSurfaces();
  if (S.bz) fit();
  else if (S.bzCam) { Object.assign(cam, S.bzCam); S.bzCam = null; }
  summaries(); requestRender();
}
/** Draws the zone: edges as lines, faces as a faint shell, the k-path (accent) and b1, b2, b3 as thin rods, the
 *  high-symmetry points as small spheres with their labels. Nothing in this view can be picked. */
function buildBZ() {
  const { k, zone } = S.bzData, V = zone.vertices;
  const kmax = Math.max(...V.map(v => norm(v))), sc = BZ_SIZE / kmax, P = v => v.map(x => x * sc);
  const accent = css("--accent") || "#3058d0", ink = css("--ink2") || "#484d56", ax = ["--ax-a", "--ax-b", "--ax-c"].map(css);
  const segs = zone.edges.map(([i, j]) => ({ p0: P(V[i]), p1: P(V[j]), dashed: false }));
  const bonds = [], pts = [], labels = [];
  for (const run of k.path) for (let i = 0; i + 1 < run.length; i++) bonds.push({ p0: P(k.points[run[i]]), p1: P(k.points[run[i + 1]]), c0: accent, c1: accent, r: 0.035 * BZ_SIZE });
  k.B.forEach((b, i) => bonds.push({ p0: [0, 0, 0], p1: P(b), c0: ax[i] || ink, c1: ax[i] || ink, r: 0.012 * BZ_SIZE }));
  const onPath = new Set(k.path.flat());
  for (const [lab, q] of Object.entries(k.points)) {
    const p = P(q), r = (onPath.has(lab) ? 0.075 : 0.05) * BZ_SIZE;
    pts.push({ x: p[0], y: p[1], z: p[2], r, color: onPath.has(lab) ? accent : ink });
    const d = norm(p) > 1e-9 ? p.map(x => x / norm(p)) : [0, 0, 1], o = 0.16 * BZ_SIZE;
    labels.push({ x: p[0] + o * d[0], y: p[1] + o * d[1], z: p[2] + o * d[2], r: 0.11 * BZ_SIZE, text: kLabel(lab), color: "#ffffff" });
  }
  k.B.forEach((b, i) => { const p = P(b); labels.push({ x: p[0], y: p[1], z: p[2], r: 0.09 * BZ_SIZE, text: "b" + "₁₂₃"[i], color: "#ffffff" }); });
  // faces: a fan per face, each vertex with the face normal (outward along G), as a faint translucent mesh
  const pos = [], nrm = [], idx = [];
  zone.faces.forEach((f, fi) => {
    const g = zone.planes[fi], gn = norm(g), n = g.map(x => Math.round(32767 * x / gn)), base = pos.length / 3;
    for (const vi of f) { pos.push(...P(V[vi])); nrm.push(n[0], n[1], n[2], 0); }
    for (let t = 1; t + 1 < f.length; t++) idx.push(base, base + t, base + t + 1);
  });
  R.setMeshGroup("bz", [{ positions: new Float32Array(pos), normals: new Int16Array(nrm), indices: new Uint32Array(idx), triangles: idx.length / 3, color: hexToRgb01(accent), opacity: 0.1 }]);
  disp = { atoms: [], centers: new Float32Array(0), radii: new Float32Array(0) };
  S.sel = []; S.labels = [];
  R.setAtoms(packAtoms(pts), pts.length);
  R.setBonds(packBonds(bonds), bonds.length);
  R.setLines(packLines(segs), segs.length);
  if (R.setLabels) { const atlas = atlasOf(labels.map(l => l.text)); R.setLabels(packLabels(labels, atlas ? atlas.map : {}), atlas ? labels.length : 0, atlas); }
  updateSelectionBar();
  const pathText = k.path.map(r => r.map(kLabel).join("–")).join(" | ");
  buildInset();
  $("info").textContent = `Brillouin zone of ${S.bzData.name} · ${k.variant.toUpperCase()} · ${pathText} · |b| ${k.B.map(b => norm(b).toFixed(3)).join(" ")} Å⁻¹`;
  requestRender();
}
$("bz").addEventListener("change", e => setBZ(e.target.checked));

/* ------------------------------------------------------------------ Brillouin zone: crystal inset */
// The crystal of the zone in a small box (bottom left), drawn by a second renderer on its own canvas with the main
// camera's rotation: the zone and the crystal share one Cartesian frame (b_i . a_j = 2 pi delta_ij), so the same
// rotation shows both from the same direction. Dragging in the box turns both; the wheel zooms the box; the corner
// grip resizes it (kept in localStorage).
const BZI = { R: null, cam: new Camera(), size: 220, on: true, zoom: 1, starting: false };
try { const v = +localStorage.getItem(store.PREFIX + "bzi-size"); if (v >= 120) BZI.size = v; BZI.on = localStorage.getItem(store.PREFIX + "bzi") !== "0"; } catch (e) { /* storage unavailable */ }
$("bzi").checked = BZI.on;
/** Side of the box in CSS px: the stored size, at most half the window width and 70% of its height. */
const insetSide = () => Math.round(Math.max(120, Math.min(BZI.size, window.innerWidth * 0.5, window.innerHeight * 0.7)));
/** Shows, hides or refills the inset (atoms, bonds and cell of the zone's structure as drawn before the zone view). */
async function buildInset() {
  const box = $("bzinset"), show = !!(S.bz && BZI.on && S.bzData);
  $("row-bzi").hidden = !S.bz;
  box.hidden = !show;
  if (!show) return;
  if (!BZI.R) {
    if (BZI.starting) return;
    BZI.starting = true;
    const cv = $("bzi-cv");
    try { BZI.R = globalThis.__isoviewRenderer ? await globalThis.__isoviewRenderer(cv) : await createRenderer(cv, null, R && R.name === "WebGL2" ? "webgl2" : ""); }
    catch (e) { BZI.R = null; }
    BZI.starting = false;
    if (!BZI.R) { box.hidden = true; return; }
    if (BZI.R.canvas && BZI.R.canvas !== cv) BZI.R.canvas.id = "bzi-cv";
    if (!S.bz) { box.hidden = true; return; }
  }
  const L = layerById(S.bzData.layer);
  if (!L) { box.hidden = true; return; }
  const { atoms, bonds, segs } = structureScene([L], [L]), lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const add = p => { for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); } };
  for (const a of atoms) add([a.x, a.y, a.z]);
  for (const s of segs) { add(s.p0); add(s.p1); }
  if (Number.isFinite(lo[0])) BZI.cam.fit(lo.map((v, i) => (v + hi[i]) / 2), Math.max(1, 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])));
  BZI.cam.dist *= BZI.zoom;
  BZI.R.setBackground(liveBg());
  BZI.R.setAtoms(packAtoms(atoms), atoms.length);
  BZI.R.setBonds(packBonds(bonds), bonds.length);
  BZI.R.setLines(packLines(segs), segs.length);
  $("bzi-t").textContent = itemOf(L).name;
  sizeInset();
}
/** Box and canvas size (device pixels at up to 2 per CSS pixel), then a redraw. */
function sizeInset() {
  const s = insetSide(), cv = BZI.R ? BZI.R.canvas : $("bzi-cv"), dpr = Math.min(window.devicePixelRatio || 1, 2);
  $("bzinset").style.setProperty("--bzi", s + "px");
  cv.width = cv.height = Math.round(s * dpr);
  BZI.cam.w = BZI.cam.h = s;
  requestRender();
}
/** The inset drawn with the main camera's rotation and projection (into its own canvas, or into target r). */
function drawInset(r = BZI.R, w = r.canvas.width, h = r.canvas.height, bg = liveBg()) {
  const c = BZI.cam;
  c.q = cam.q.slice(); c.ortho = cam.ortho;
  r.draw(uniformsFor(c, r, w, h, bg));
}
$("bzi").addEventListener("change", e => {
  BZI.on = e.target.checked;
  try { localStorage.setItem(store.PREFIX + "bzi", BZI.on ? "1" : "0"); } catch (err) { /* storage unavailable */ }
  buildInset();
});
$("bzi-x").addEventListener("click", () => { $("bzi").checked = false; $("bzi").dispatchEvent(new Event("change")); });
$("bzi-cv").addEventListener("pointerdown", e => {
  const cv = e.currentTarget;
  let x = e.clientX, y = e.clientY;
  cv.setPointerCapture(e.pointerId); cv.classList.add("drag");
  const move = ev => { cam.rotateBy(ev.clientX - x, ev.clientY - y); x = ev.clientX; y = ev.clientY; requestRender(); drawTriad(); };
  const up = () => { cv.removeEventListener("pointermove", move); cv.removeEventListener("pointerup", up); cv.classList.remove("drag"); };
  cv.addEventListener("pointermove", move); cv.addEventListener("pointerup", up);
});
$("bzi-cv").addEventListener("wheel", e => {
  e.preventDefault();
  const f = Math.exp(Math.sign(e.deltaY) * 0.12);
  BZI.zoom = Math.min(4, Math.max(0.25, BZI.zoom * f)); BZI.cam.dist *= f; requestRender();
}, { passive: false });
$("bzi-grip").addEventListener("pointerdown", e => {
  e.preventDefault();
  const g = e.currentTarget, x0 = e.clientX, y0 = e.clientY, s0 = insetSide();
  g.setPointerCapture(e.pointerId);
  // the box is anchored at its bottom left corner: dragging the top right corner up or right enlarges it
  const move = ev => { BZI.size = Math.max(120, s0 + Math.max(ev.clientX - x0, y0 - ev.clientY)); sizeInset(); };
  const up = () => { g.removeEventListener("pointermove", move); g.removeEventListener("pointerup", up); try { localStorage.setItem(store.PREFIX + "bzi-size", String(insetSide())); } catch (err) { /* storage unavailable */ } };
  g.addEventListener("pointermove", move); g.addEventListener("pointerup", up);
});
window.addEventListener("resize", () => { if (S.bz && BZI.R && !$("bzinset").hidden) sizeInset(); });

/* ------------------------------------------------------------------ photo mode */
// One place for images of the 3D view. Photo (title bar, or S) hides every control and draws the view supersampled;
// Save PNG and Copy render it again at the chosen size. The image is drawn in tiles of at most PH_TILE device px, each
// with an off-centre projection (uniformsFor tile), at PH_SS times the output size and averaged down, so sphere and
// bond edges are smooth and the GPU holds one tile at a time. A transparent background comes from two renders, on
// black (C_0 = a F) and on white (C_1 = a F + (1 - a)): a = 1 - mean over r, g, b of (C_1 - C_0), F = C_0 / a.
// In the Brillouin-zone view the crystal inset goes into the image where it shows; with a slice, the 2D view, the inset
// and the colour bar as they show (renderPhoto).
const PH = { on: false, size: "2", bg: "scene" };
const PH_SS = 2, PH_TILE = 2048, PH_MAXPX = 16e6, PH_LIVE = 6e6; // PH_MAXPX: the largest 2D canvas Safari allows (2^24 px)
try { const v = JSON.parse(localStorage.getItem(store.PREFIX + "photo") || "{}"); if (["screen", "2", "4"].includes(v.size)) PH.size = v.size; if (["scene", "white", "clear"].includes(v.bg)) PH.bg = v.bg; } catch (e) { /* storage unavailable */ }
/** Background of the live view: white while photo mode shows a white-background photo, else the theme's. */
function liveBg() { return PH.on && PH.bg === "white" ? [1, 1, 1] : pal.bg; }
/** Output size: the window (CSS px) times the chosen scale (Screen: the device pixel ratio), at most PH_MAXPX px. */
function photoSize() {
  const vw = window.innerWidth, vh = window.innerHeight, k0 = PH.size === "screen" ? (window.devicePixelRatio || 1) : +PH.size, k = Math.min(k0, Math.sqrt(PH_MAXPX / (vw * vh)));
  return { w: Math.round(vw * k), h: Math.round(vh * k) };
}
function setPhoto(on) {
  if (on === !!PH.on || (on && (!R || !S.layers.length))) return;
  PH.on = on;
  document.body.classList.toggle("photo", on);
  $("photobar").hidden = !on;
  if (on) { showUse(null); photoUI(); }
  if (!S.bz && disp.atoms.length) uploadAtoms();
  R.setBackground(liveBg()); if (BZI.R) BZI.R.setBackground(liveBg());
  resize(); if (S.bz && BZI.R) sizeInset();
  if (SL.on) applySliceView(); // the 2D view's margins and the inset's handles change
}
function photoUI() {
  for (const b of $("ph-size").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === PH.size));
  for (const b of $("ph-bg").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === PH.bg));
  const { w, h } = photoSize();
  $("ph-px").textContent = `${w} × ${h} px`;
}
/** a = 1 - mean(C_1 - C_0), F = C_0 / a, into A (ImageData on black); B: the same pixels on white. */
function matte(A, B) {
  const a = A.data, b = B.data;
  for (let i = 0; i < a.length; i += 4) {
    const al = Math.min(1, 1 - (b[i] - a[i] + b[i + 1] - a[i + 1] + b[i + 2] - a[i + 2]) / 765);
    if (al < 0.004) { a[i] = a[i + 1] = a[i + 2] = a[i + 3] = 0; continue; }
    a[i] = Math.min(255, a[i] / al); a[i + 1] = Math.min(255, a[i + 1] / al); a[i + 2] = Math.min(255, a[i + 2] / al); a[i + 3] = Math.round(255 * al);
  }
}
/** Draws the view of camera c on renderer r into the 2D context g, as an ow x oh px image at (dx, dy): tiles of the
 *  PH_SS-times larger image, each averaged down into place. bgs: one background, or black and white for matting. */
function drawTiles(r, c, g, dx, dy, ow, oh, bgs) {
  const VW = ow * PH_SS, VH = oh * PH_SS, T = Math.min(PH_TILE, r.maxSide || PH_TILE) - (Math.min(PH_TILE, r.maxSide || PH_TILE) % PH_SS);
  const tmp = bgs.length > 1 ? [document.createElement("canvas"), document.createElement("canvas")] : null;
  for (let py = 0; py < VH; py += T) for (let px = 0; px < VW; px += T) {
    const tw = Math.min(T, VW - px), th = Math.min(T, VH - py), tile = [px, py, VW, VH], ox = dx + px / PH_SS, oy = dy + py / PH_SS, w2 = tw / PH_SS, h2 = th / PH_SS;
    r.canvas.width = tw; r.canvas.height = th;
    const shots = bgs.map((bg, i) => {
      r.setBackground(bg);
      r.draw(uniformsFor(c, r, tw, th, bg, VW / c.w, tile));
      if (!tmp) { g.drawImage(r.canvas, 0, 0, tw, th, ox, oy, w2, h2); return null; }
      const t = tmp[i]; t.width = w2; t.height = h2;
      const tg = t.getContext("2d"); tg.imageSmoothingQuality = "high"; tg.drawImage(r.canvas, 0, 0, tw, th, 0, 0, w2, h2);
      return tg.getImageData(0, 0, w2, h2);
    });
    if (tmp) { matte(shots[0], shots[1]); g.putImageData(shots[0], ox, oy); }
  }
  r.setBackground(liveBg());
}
/** The photo as a canvas, at photoSize(); the live view is restored afterwards. With a slice: the main view (3D, or
 *  the 2D figure), the inset (the other view) and the colour bar as on screen, the slice sampled again for the output
 *  size (sliceFine). The inset is drawn over a cleared box (transparent for a transparent photo) with a thin frame. */
async function renderPhoto() {
  const vw = window.innerWidth, { w, h } = photoSize(), s = w / vw, mode = slMode(), out = document.createElement("canvas");
  out.width = w; out.height = h;
  const g = out.getContext("2d"), bgs = PH.bg === "clear" ? [[0, 0, 0], [1, 1, 1]] : [PH.bg === "white" ? [1, 1, 1] : pal.bg];
  g.imageSmoothingQuality = "high";
  const ib = mode !== "off" && SV.inset ? insetBox() : null, I = ib && { x: Math.round(ib.x * s), y: Math.round(ib.y * s), w: Math.round(ib.w * s), h: Math.round(ib.h * s) };
  const fine = !SL.style ? null : mode === "2d" ? await sliceFine(0.9 * Math.max(w, h)) : mode === "3d" && I ? await sliceFine(Math.max(I.w, I.h)) : null;
  const framed = r => { g.strokeStyle = css("--line2") || "#c8ccd2"; g.lineWidth = Math.max(1, s); g.strokeRect(r.x + g.lineWidth / 2, r.y + g.lineWidth / 2, r.w - g.lineWidth, r.h - g.lineWidth); };
  const bg = figBg(true);
  try {
    if (mode === "2d") {
      if (bg) { g.fillStyle = bg; g.fillRect(0, 0, w, h); }
      if (fine) drawSlice2D(g, 0, 0, w, h, s, fine, mainPad(), SL_FS, ib);
      if (I) { g.clearRect(I.x, I.y, I.w, I.h); drawTiles(R, cam, g, I.x, I.y, I.w, I.h, bgs); framed(I); }
    } else {
      drawTiles(R, cam, g, 0, 0, w, h, bgs);
      if (I && fine) {
        g.clearRect(I.x, I.y, I.w, I.h);
        if (bg) { g.fillStyle = bg; g.fillRect(I.x, I.y, I.w, I.h); }
        drawSlice2D(g, I.x, I.y, I.w, I.h, s, fine, [8, 8, 8, 8], SL_FS_INSET); framed(I);
      } else if (mode === "3d" && SV.bar && SL.in3d && SL.style) drawBarAlone(g, w, h, s, barRight());
      if (S.bz && BZI.R && !$("bzinset").hidden) {
        const b = $("bzinset").getBoundingClientRect(), x = Math.round(b.left * s), y = Math.round(b.top * s), side = Math.round(b.width * s);
        BZI.cam.q = cam.q.slice(); BZI.cam.ortho = cam.ortho;
        drawTiles(BZI.R, BZI.cam, g, x, y, side, side, bgs);
        framed({ x, y, w: side, h: side });
      }
    }
  } finally {
    resize(); if (S.bz && BZI.R) sizeInset();
    frame();
  }
  return out;
}
const photoName = () => ($("doc-name").textContent.split(" + ")[0] || "isoview").replace(/[^\w.-]+/g, "_") + ".png";
/** Renders the photo and saves it as PNG. Returns a promise (settled when the file is handed to the browser). */
function savePhoto() {
  if (!PH.on || !R) return Promise.resolve();
  busy(true);
  return renderPhoto().then(out => new Promise(res => out.toBlob(b => { if (b) download(b, photoName()); else toast("Could not encode the image (too large for this browser?).", true); res(); }, "image/png")))
    .catch(e => toast("Could not draw the image: " + (e.message || e), true)).finally(() => busy(false));
}
function copyPhoto() {
  if (!PH.on || !R) return;
  if (!navigator.clipboard || !window.ClipboardItem) { toast("This browser cannot copy images; use Save PNG.", true); return; }
  // WHY: the blob is handed over as a promise so the copy still counts as part of the click
  const blob = renderPhoto().then(out => new Promise((res, rej) => out.toBlob(b => (b ? res(b) : rej(new Error("the image could not be encoded"))), "image/png")));
  navigator.clipboard.write([new ClipboardItem({ "image/png": blob })])
    .then(() => toast("Copied the image."), e => toast("Could not copy: " + (e.message || e), true));
}
const savePhotoPrefs = () => { try { localStorage.setItem(store.PREFIX + "photo", JSON.stringify({ size: PH.size, bg: PH.bg })); } catch (e) { /* storage unavailable */ } };
$("btn-photo").addEventListener("click", () => setPhoto(true));
$("ph-done").addEventListener("click", () => setPhoto(false));
$("ph-save").addEventListener("click", savePhoto);
$("ph-copy").addEventListener("click", copyPhoto);
$("ph-size").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { PH.size = b.value; photoUI(); savePhotoPrefs(); } });
$("ph-bg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  PH.bg = b.value; photoUI(); savePhotoPrefs();
  R.setBackground(liveBg()); if (BZI.R) BZI.R.setBackground(liveBg());
  if (SL.on) drawSliceViews();
  requestRender();
});
window.addEventListener("resize", () => { if (PH.on) photoUI(); });

/* ------------------------------------------------------------------ 2D slices */
// The slice (SL) is a 2D cut through the active field on the (hkl) plane at position t of its cell (supercell for
// periodic fields), sampled in the worker (slice.js: tricubic, at least 4 pixels per grid step and finer when it is
// drawn large). It shows as a coloured plane in the 3D scene, as the main view (2D: #slv, the figure fitted to the
// window) and in the inset (#sli, movable and resizable): SV.main picks the main view and the inset shows the other
// one (the 2D figure beside the 3D view, or the 3D view beside the 2D figure: the renderer's canvas moves into the
// inset box). The colour bar goes with the 2D figure, in the main view or the inset; with the 3D view alone it stands
// at the right of the view. Photo Mode draws the same picture at the output size, the slice sampled again so that one
// image pixel is at most 2 output pixels.
// (the slice state SL, SV is declared with the other state, near the top)

/** The field layer the slice is taken from (the active one, else the first visible one) with its item and field. */
function sliceTarget() {
  const L = activeField() || fieldLayers(true)[0], it = L && itemOf(L), f = it && it.fields[L.fi];
  return f ? { L, it, f } : null;
}
/** "off" (no slice shown), or the main view: "3d" or "2d". */
const slMode = () => (SL.on && !S.bz && sliceTarget() ? SV.main : "off");
/** The inset box in CSS px { x (left), y (top), w, h }: square around the 3D view; around the 2D figure, the plane's
 *  aspect (clamped to 1:3 .. 3:1) with room for the colour bar. Its longer side is SV.size (at most 60% of the window
 *  width and 70% of its height); it stays inside the window. */
function insetBox() {
  const W = window.innerWidth, H = window.innerHeight, side = Math.max(120, Math.min(SV.size, W * 0.6, H * 0.7));
  let w = side, h = side;
  if (SV.main === "3d" && SL.img) {
    const a = Math.min(3, Math.max(1 / 3, SL.img.plane.width / SL.img.plane.height));
    if (a >= 1) h = side / a; else w = side * a;
    w += 16; h += 16; // the padding around the figure
    if (SV.bar) w += 5.2 * SL_FS_INSET + 30; // the bar, its labels and title
  }
  w = Math.round(Math.min(w, W - 8)); h = Math.round(Math.min(h, H - 8));
  const narrow = W <= 1100, bars = (document.body.classList.contains("has-player") ? 48 : 0) + 48;
  const [l, b] = SV.pos || [46, narrow ? 96 + bars : 96];
  const x = Math.min(Math.max(4, l), W - w - 4), y = Math.min(Math.max(4, H - b - h), H - h - 4);
  return { x, y, w, h };
}
/** Puts the inset box where insetBox() says. */
function placeInset() {
  const b = insetBox(), box = $("sli");
  box.style.left = b.x + "px"; box.style.bottom = (window.innerHeight - b.y - b.h) + "px";
  box.style.setProperty("--sliw", b.w + "px"); box.style.setProperty("--slih", b.h + "px");
  return b;
}
/** Shows the slice bar, the 2D view, the inset (with the 2D figure, or the renderer's canvas moved into it) and the
 *  options for the current mode, then sizes and redraws everything. */
function applySliceView() {
  const mode = slMode(), on = mode !== "off", box = $("sli");
  document.body.classList.toggle("has-slicebar", on);
  document.body.classList.toggle("sl-2d", mode === "2d");
  $("slicebar").hidden = !on;
  $("sl-panel").hidden = !SL.on;
  $("ph-sl").hidden = !on;
  box.hidden = !(on && SV.inset);
  box.classList.toggle("d3", mode === "2d");
  // the 3D view lives in the inset while the 2D slice is the main view
  const home = mode === "2d" ? box : $("stage");
  if (canvas && canvas.parentElement !== home) home.prepend(canvas);
  $("sli-cv").hidden = mode === "2d";
  $("slv").hidden = !(mode === "2d" || (mode === "3d" && SV.bar && !SV.inset && SL.in3d));
  if (!box.hidden) placeInset();
  segSlice();
  resize();
  drawSliceViews();
}
/** True while the 3D view is on screen (it is hidden when the 2D slice fills the window without the inset). */
const view3dShown = () => !(slMode() === "2d" && !SV.inset);
/** Re-slices when anything the slice depends on changed (layer, field, shift, supercell, plane, position, the size
 *  it is drawn at). */
function syncSlice() {
  const T = sliceTarget();
  if (!SL.on || !T || S.bz) {
    SL.key = "";
    if (R && R.setSlice) R.setSlice(null);
    applySliceView();
    return;
  }
  const k = [T.L.id, T.L.fi, T.f.name, T.f.n.join(), T.it.ds, S.t.join(), S.reps.join(), SL.hkl.join(), SL.t, Math.ceil(liveFit() / 256)].join("|");
  if (k !== SL.key) { SL.key = k; requestSlice(false); }
  applySliceView();
}
/** Pixels wanted along the longer side of the slice image for the 2D figure on screen: half its device pixels. */
function liveFit() {
  const mode = slMode(), d = Math.min(window.devicePixelRatio || 1, 3);
  if (mode === "2d") return Math.round(0.85 * Math.max(window.innerWidth, window.innerHeight) * d / 2);
  if (mode === "3d" && SV.inset) { const b = insetBox(); return Math.round(Math.max(b.w, b.h) * d / 2); }
  return 0;
}
/** One slice request at a time; a request made meanwhile runs after it (only the latest). preview: coarse, fast. */
function requestSlice(preview) {
  if (SL.busy) { SL.again = SL.again === false ? false : preview; return; }
  SL.busy = true;
  sliceNow(preview).catch(e => toast(e.message || String(e), true)).finally(() => {
    SL.busy = false;
    if (SL.again !== null) { const p = SL.again; SL.again = null; requestSlice(p); }
  });
}
/** The worker's slice request for the current plane: px at least 4 per grid step, fit pixels along the longer side
 *  (0: no minimum), at most maxSide pixels on a side and maxPix in all. */
function sliceCall(fit, maxSide, maxPix = 8e6) {
  const { L, it, f } = sliceTarget(), per = (f.periodic || f.closed) && pbcOf(it), reps = per ? S.reps : [1, 1, 1];
  const N = f.periodic ? f.n : f.n.map(n => n - 1);
  // WHY: four pixels per grid step of the (already refined, <= 0.1 A) grid; the cubic interpolation fills between
  const step = Math.min(...f.span.map((v, a) => norm(v) / N[a]));
  return call("slice", { ds: it.ds, fi: L.fi, cell: f.span.map((v, a) => v.map(c => c * reps[a])), origin: f.origin, hkl: SL.hkl, t: SL.t,
    px: step / 4, fit, maxSide, maxPix, shift: fracShift(f.span) }).then(r => ({ ...r, f, signed: f.kind === "signed", reps }));
}
async function sliceNow(preview) {
  if (!SL.on || !sliceTarget()) return;
  const r = await sliceCall(preview ? 0 : liveFit(), preview ? 480 : 2048);
  if (!SL.on) return;
  SL.img = r; SL.fine = null;
  paintSlice();
}
/** Scale mode and colour map in use (auto: log for densities, symmetric for signed fields, else linear). */
function sliceStyle(g = SL.img) {
  const kind = g.f.kind;
  let mode;
  if (SL.scale === "log") mode = g.max > 0 ? "log" : "lin";
  else if (SL.scale === "lin") mode = g.signed ? "signed" : "lin";
  else mode = g.signed ? "signed" : kind === "density" && g.min >= 0 && g.max > 0 ? "log" : "lin";
  const map = SL.map !== "auto" ? SL.map : mode === "signed" ? "rdbu" : "viridis";
  // log: three decades below the largest value
  const lo = mode === "log" ? g.max * 1e-3 : g.min, hi = g.max;
  return { mode, map, lo, hi, sc: scaler(mode, lo, hi) };
}
/** The coloured image of slice g in style st, as a canvas of g.W x g.H (row 0 at V = 0). */
function colorCanvas(g, st) {
  const c = document.createElement("canvas");
  c.width = g.W; c.height = g.H;
  const x = c.getContext("2d");
  if (x) x.putImageData(new ImageData(paint(g.values, g.W, g.H, st.map, st.sc), g.W, g.H), 0, 0);
  return c;
}
/** Colours the slice image, hands it to the renderer and redraws the 2D views. */
function paintSlice() {
  const g = SL.img;
  if (!g) return;
  SL.style = sliceStyle(g);
  g.canvas = colorCanvas(g, SL.style);
  if (R && R.setSlice) R.setSlice(SL.in3d ? { o: g.plane.o, U: g.plane.U, V: g.plane.V, alpha: 0.95, image: g.canvas } : null);
  $("sl-d").textContent = slicePos(g.plane);
  $("sl-d").title = `The plane is ${(g.plane.d - g.plane.dRange[0]).toFixed(3)} Å into the cell (of ${g.plane.depth.toFixed(3)} Å) along the normal of (${SL.hkl.join(" ")})`;
  // the inset's shape follows the plane
  if (!$("sli").hidden) placeInset();
  requestRender();
  drawSliceViews();
  segSlice();
}
/** Position text: for (100), (010), (001) the fractional coordinate in the unit cell (x, y, z; past 1 in a
 *  supercell) and the distance into the cell; for other planes the distance only. */
function slicePos(P) {
  const d = `${(P.d - P.dRange[0]).toFixed(2)} Å`, a = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].findIndex(v => v.join() === SL.hkl.join());
  if (a < 0) return d;
  const reps = SL.img ? SL.img.reps[a] : 1;
  return `${"xyz"[a]} ${(SL.t * reps).toFixed(3)} · ${d}`;
}
/** Contour levels: 8 evenly spaced in the colour scale (symmetric pairs for signed fields), as field values. */
function sliceLevels(st) {
  const out = [];
  if (st.mode === "signed") { const m = Math.max(Math.abs(st.lo), Math.abs(st.hi)); for (let i = 1; i <= 4; i++) out.push(m * i / 5, -m * i / 5); }
  else if (st.mode === "log") { const a = Math.log10(st.lo), b = Math.log10(st.hi); for (let i = 1; i <= 8; i++) out.push(10 ** (a + (b - a) * i / 9)); }
  else for (let i = 1; i <= 8; i++) out.push(st.lo + (st.hi - st.lo) * i / 9);
  return out;
}
/** Atoms within SL_NEAR of the plane, as { s, r (fractions of U, V), rad (A: a quarter of the covalent radius, small
 *  enough to leave the density around the nucleus visible), el, color, w (closeness 0..1) }. */
function sliceAtoms(P) {
  const out = [], U2 = dot(P.U, P.U), V2 = dot(P.V, P.V);
  for (const a of disp.atoms) {
    const p = [a.x, a.y, a.z], dist = dot(p, P.n) - P.d;
    if (Math.abs(dist) > SL_NEAR) continue;
    const q = [p[0] - P.o[0], p[1] - P.o[1], p[2] - P.o[2]], s = dot(q, P.U) / U2, r = dot(q, P.V) / V2;
    if (s < -0.02 || s > 1.02 || r < -0.02 || r > 1.02) continue;
    out.push({ s, r, rad: 0.25 * rcov(a.el), el: a.el, color: a.color || elColor(a.el), w: 1 - Math.abs(dist) / SL_NEAR });
  }
  return out;
}
/** The colour bar's title: the field's name and unit. */
const barTitle = () => { const f = SL.img && SL.img.f; return f ? `${f.name}${f.unit ? ` (${f.unit})` : ""}` : ""; };
/** Text colour of slice figures: the theme's, or near black on a white or transparent photo (for paper). */
const figInk = () => (PH.on && PH.bg !== "scene" ? "#15171b" : css("--ink") || "#15171b");
/** Background of slice figures: the live view's colour (CSS rgb), or null for a transparent photo when exporting. */
const figBg = (exporting = false) => (exporting && PH.bg === "clear" ? null : `rgb(${liveBg().map(v => Math.round(255 * v)).join(",")})`);
const figFont = () => css("--sans") || "system-ui, sans-serif";
/** Padding of the 2D main view, CSS px [top, right, bottom, left], the same above and below and left and right so
 *  that the figure is centred in the window (panels over the view are avoided by layoutClear): clear of the slice
 *  bar and the player below; in Photo Mode the same on all sides. */
function mainPad() {
  const W = window.innerWidth, H = window.innerHeight;
  if (PH.on) { const p = Math.max(24, Math.round(0.05 * Math.min(W, H))); return [p, p, p, p]; }
  const v = 76 + (document.body.classList.contains("has-player") ? 48 : 0);
  return [v, 28, v, 28];
}
/** CSS px kept free at the right of the view by the panel, when it shows there (at least 24). */
function dockRight() {
  const d = $("dock"), W = window.innerWidth;
  if (!d || d.hidden || d.classList.contains("tucked")) return 24;
  const r = d.getBoundingClientRect();
  return r.width && r.left > W / 2 ? Math.max(24, W - r.left + 16) : 24;
}
/** Draws the 2D figure (and its colour bar when SV.bar) of slice g into the box (x, y, w, h) of context ctx.
 *  k: pixels per CSS px; pad: CSS px [top, right, bottom, left]; fs: text size in CSS px. avoid: for the main view,
 *  the boxes in CSS px of the panels over it (a box, a list or null): the figure stays centred and clear of them
 *  (layoutClear); without avoid (the inset itself) the figure and bar are centred as a pair. */
function drawSlice2D(ctx, x, y, w, h, k, g, pad, fs, avoid) {
  const P = g.plane, st = SL.style, F = fs * k, font = figFont(), ink = figInk(), bt = barTicks(st.mode, st.lo, st.hi);
  let title = barTitle();
  const bw = SV.bar ? colorBarWidth(ctx, F, font, bt, title) : 0, a = P.width / P.height, pk = pad.map(p => p * k);
  const Lo = avoid === undefined ? figureLayout(w, h, a, pk, bw, 6 * F)
    : layoutClear(w, h, a, pk, bw, 6 * F, (Array.isArray(avoid) ? avoid : avoid ? [avoid] : []).map(R => ({ x: R.x * k, y: R.y * k, w: R.w * k, h: R.h * k })), 16 * k);
  drawFigure(ctx, x + Lo.x, y + Lo.y, Lo.w, Lo.h, { image: g.canvas, values: g.values, iw: g.W, ih: g.H, poly: P.poly, levels: sliceLevels(st),
    lines: SL.iso, atoms: SL.atoms ? sliceAtoms(P) : [], width: P.width, k });
  if (!SV.bar) return;
  ctx.save(); ctx.font = `${F}px ${font}`;
  if (ctx.measureText(title).width > Lo.bh) title = ""; // a title longer than the bar is left out
  ctx.restore();
  drawColorBar(ctx, x + Lo.bx, y + Lo.by, Lo.bh, { fs: F, font, ink, bt, map: st.map, title, k });
}
/** The colour bar alone at the right of a W x H view (the 3D view without the inset): 40% of the height, at most
 *  320 CSS px, centred; right: CSS px kept free at the right. k: pixels per CSS px. */
function drawBarAlone(ctx, W, H, k, right) {
  const st = SL.style, F = SL_FS * k, font = figFont(), bt = barTicks(st.mode, st.lo, st.hi);
  let title = barTitle();
  const bw = colorBarWidth(ctx, F, font, bt, title), bh = Math.min(0.4 * H, 320 * k);
  ctx.save(); ctx.font = `${F}px ${font}`;
  if (ctx.measureText(title).width > bh) title = "";
  ctx.restore();
  drawColorBar(ctx, W - right * k - bw, (H - bh) / 2, bh, { fs: F, font, ink: figInk(), bt, map: st.map, title, k });
}
/** Right margin (CSS px) of the colour bar alone: clear of the panel on screen, 24 px in Photo Mode. */
function barRight() { return PH.on ? 24 : dockRight(); }
/** Redraws the 2D main view (or the colour bar over the 3D view) and the 2D inset, at device resolution. */
function drawSliceViews() {
  const mode = slMode(), g = SL.img, d = Math.min(window.devicePixelRatio || 1, 3);
  const cv = $("slv");
  if (!cv.hidden) {
    const W = window.innerWidth, H = window.innerHeight;
    cv.width = Math.round(W * d); cv.height = Math.round(H * d);
    const ctx = cv.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, cv.width, cv.height);
      if (g && SL.style && mode === "2d") { ctx.fillStyle = figBg(); ctx.fillRect(0, 0, cv.width, cv.height); drawSlice2D(ctx, 0, 0, cv.width, cv.height, d, g, mainPad(), SL_FS, viewObstacles()); }
      else if (g && SL.style && mode === "3d") drawBarAlone(ctx, cv.width, cv.height, d, barRight());
    }
  }
  const box = $("sli"), ic = $("sli-cv");
  if (!box.hidden && mode === "3d") {
    const b = insetBox();
    ic.width = Math.round(b.w * d); ic.height = Math.round(b.h * d);
    const ctx = ic.getContext("2d");
    if (ctx) {
      ctx.fillStyle = figBg(); ctx.fillRect(0, 0, ic.width, ic.height);
      if (g && SL.style) drawSlice2D(ctx, 0, 0, ic.width, ic.height, d, g, [8, 8, 8, 8], SL_FS_INSET);
    }
  }
}
/** Pressed states of the plane, scale, view, inset and colour-bar buttons (slice bar, panel, photo bar). */
function segSlice() {
  for (const b of $("sl-planes").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === SL.hkl.join(" ")));
  const mode = SL.style ? SL.style.mode : "";
  for (const b of $("sl-scale").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === (mode === "log" ? "log" : "lin")));
  for (const id of ["sl-view", "ph-view"]) for (const b of $(id).querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === SV.main));
  $("sl-inset").setAttribute("aria-pressed", String(SV.inset)); $("sl-bar").setAttribute("aria-pressed", String(SV.bar));
  for (const b of $("ph-opts").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === "inset" ? SV.inset : SV.bar));
}
function setSlicePlane(hkl) {
  if (hkl.length !== 3 || hkl.some(x => !Number.isInteger(x)) || hkl.every(x => x === 0)) { toast("Give three whole numbers h k l, not all zero, e.g. 1 1 0.", true); $("sl-hkl").value = SL.hkl.join(" "); return; }
  SL.hkl = hkl; $("sl-hkl").value = hkl.join(" "); segSlice(); syncSlice();
}
/** Switches the slice on or off (the Surface panel switch, the bar's close button, a new scene). */
function setSliceOn(on) {
  SL.on = on; $("slice").checked = on;
  if (!on) { SL.img = null; SL.style = null; SL.fine = null; SV.main = "3d"; }
  syncSlice(); requestRender();
}
/** Main view, inset or colour bar changed (slice bar or photo bar). */
function setSliceView(ch) {
  Object.assign(SV, ch);
  if ("inset" in ch || "bar" in ch) saveSV();
  syncSlice(); requestRender(); // resamples only when the 2D figure is now drawn much larger (liveFit)
}
/** The slice sampled for an image whose 2D figure has a longer side of px pixels (one image pixel per 2 output
 *  pixels, at least the live sampling), coloured in the live style; kept for the next export of the same picture. */
async function sliceFine(px) {
  const fit = Math.max(Math.ceil(px / 2), SL.img ? Math.max(SL.img.W, SL.img.H) : 0), key = SL.key + "|" + fit;
  if (SL.fine && SL.fine.key === key) return SL.fine.g;
  let g = SL.img;
  try { g = await sliceCall(fit, 4096, 8e6); g.canvas = colorCanvas(g, SL.style); } catch (e) { g = SL.img; }
  SL.fine = { key, g };
  return g;
}
/** The 2D figure as SVG: the image (PNG, embedded), contour paths, the outline, atoms, the colour bar with ticks. */
async function sliceSVG() {
  const g0 = SL.img, P = g0.plane, st = SL.style, W = 800, H = Math.round(W * P.height / P.width), g = await sliceFine(2 * W);
  const X = s => (s * W).toFixed(2), Y = r => (H - r * H).toFixed(2), fs = 13;
  const flip = document.createElement("canvas"); flip.width = g.W; flip.height = g.H;
  const fx = flip.getContext("2d"); fx.translate(0, g.H); fx.scale(1, -1); fx.drawImage(g.canvas, 0, 0);
  const bt = barTicks(st.mode, st.lo, st.hi), lab = Math.max(0, ...bt.ticks.map(t => 0.6 * fs * t.text.length)), title = barTitle();
  const bx = W + 1.1 * fs, bw = 0.85 * fs, bh = 0.8 * H, by = 0.1 * H, TW = Math.ceil(bx + bw + 0.65 * fs + lab + 2.2 * fs);
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${TW}" height="${H}" viewBox="0 0 ${TW} ${H}" font-family="Helvetica, Arial, sans-serif" font-size="${fs}">`,
    `<image x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="none" xlink:href="${flip.toDataURL("image/png")}"/>`];
  if (SL.iso) for (const c of sliceLevels(st)) {
    const seg = contours(g.values, g.W, g.H, c);
    let d = "";
    for (let i = 0; i < seg.length; i += 4) d += `M${X(seg[i] / g.W)} ${Y(seg[i + 1] / g.H)}L${X(seg[i + 2] / g.W)} ${Y(seg[i + 3] / g.H)}`;
    if (d) parts.push(`<path d="${d}" fill="none" stroke="#fff" stroke-opacity="0.6" stroke-width="0.8" stroke-linecap="round"${c < 0 ? ' stroke-dasharray="4 3"' : ""}/>`);
  }
  const poly = P.poly.map(([s, r]) => `${X(s)},${Y(r)}`).join(" ");
  if (SL.atoms) {
    parts.push(`<clipPath id="sec"><polygon points="${poly}"/></clipPath><g clip-path="url(#sec)">`);
    for (const a of sliceAtoms(P)) parts.push(`<circle cx="${X(a.s)}" cy="${Y(a.r)}" r="${Math.max(2, a.rad * W / P.width).toFixed(2)}" fill="${a.color}" fill-opacity="${(0.45 + 0.55 * a.w).toFixed(2)}" stroke="#000" stroke-opacity="0.6"><title>${esc(a.el)}</title></circle>`);
    parts.push("</g>");
  }
  parts.push(`<polygon points="${poly}" fill="none" stroke="#000" stroke-opacity="0.45"/>`);
  const t = lut(st.map), stops = [];
  for (let i = 0; i <= 32; i++) { const k = Math.round(255 * i / 32); stops.push(`<stop offset="${(i / 32).toFixed(4)}" stop-color="rgb(${t[3 * k]},${t[3 * k + 1]},${t[3 * k + 2]})"/>`); }
  parts.push(`<defs><linearGradient id="cb" x1="0" y1="1" x2="0" y2="0">${stops.join("")}</linearGradient></defs>`,
    `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" fill="url(#cb)" stroke="#15171b" stroke-opacity="0.7" stroke-width="0.8"/>`);
  for (const tk of bt.ticks) {
    const y = (by + bh - tk.x * bh).toFixed(1);
    parts.push(`<line x1="${(bx + bw).toFixed(1)}" x2="${(bx + bw + 0.35 * fs).toFixed(1)}" y1="${y}" y2="${y}" stroke="#15171b" stroke-width="0.8"/><text x="${(bx + bw + 0.65 * fs).toFixed(1)}" y="${y}" dominant-baseline="middle" fill="#15171b">${esc(tk.text)}</text>`);
  }
  if (bt.mult) parts.push(`<text x="${bx.toFixed(1)}" y="${(by - 0.45 * fs).toFixed(1)}" fill="#15171b">${esc(bt.mult)}</text>`);
  if (title) { const tx = bx + bw + 0.65 * fs + lab + 1.1 * fs, ty = by + bh / 2; parts.push(`<text transform="translate(${tx.toFixed(1)} ${ty.toFixed(1)}) rotate(-90)" text-anchor="middle" dominant-baseline="middle" fill="#15171b">${esc(title)}</text>`); }
  parts.push("</svg>");
  return parts.join("\n");
}
function sliceName(ext) { return `${($("doc-name").textContent || "slice").replace(/[^\w.+-]+/g, "_")}-slice-${SL.hkl.join("")}-${SL.t.toFixed(3)}.${ext}`; }
$("slice").addEventListener("change", e => setSliceOn(e.target.checked));
$("sl-x").addEventListener("click", () => setSliceOn(false));
$("sl-planes").addEventListener("click", e => { const b = e.target.closest("button"); if (b) setSlicePlane(b.value.split(" ").map(Number)); });
$("sl-hkl").addEventListener("change", e => setSlicePlane(e.target.value.trim().split(/[\s,]+/).map(Number)));
$("sl-t").addEventListener("input", e => { SL.t = +e.target.value; SL.key = "moving"; requestSlice(true); });
$("sl-t").addEventListener("change", e => { SL.t = +e.target.value; SL.key = ""; syncSlice(); });
$("sl-scale").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { SL.scale = b.value; SL.fine = null; paintSlice(); } });
$("sl-map").addEventListener("change", e => { SL.map = e.target.value; SL.fine = null; paintSlice(); });
$("sl-iso").addEventListener("change", e => { SL.iso = e.target.checked; drawSliceViews(); });
$("sl-atoms").addEventListener("change", e => { SL.atoms = e.target.checked; drawSliceViews(); });
$("sl-3d").addEventListener("change", e => { SL.in3d = e.target.checked; paintSlice(); applySliceView(); });
for (const id of ["sl-view", "ph-view"]) $(id).addEventListener("click", e => { const b = e.target.closest("button"); if (b && b.value !== SV.main) setSliceView({ main: b.value }); });
$("sl-inset").addEventListener("click", () => setSliceView({ inset: !SV.inset }));
$("sl-bar").addEventListener("click", () => setSliceView({ bar: !SV.bar }));
$("ph-opts").addEventListener("click", e => { const b = e.target.closest("button"); if (b) setSliceView(b.value === "inset" ? { inset: !SV.inset } : { bar: !SV.bar }); });
$("sli-x").addEventListener("click", () => setSliceView({ inset: false }));
$("sl-svg").addEventListener("click", async () => {
  if (!SL.img || !SL.style) return;
  busy(true);
  try { download(new Blob([await sliceSVG()], { type: "image/svg+xml" }), sliceName("svg")); }
  catch (e) { toast("Could not write the SVG: " + (e.message || e), true); }
  finally { busy(false); }
});
// the inset: the handle (or the 2D figure itself) moves it, the corner grip resizes it (bottom left stays), a drag on
// the 3D view turns it (right button, Ctrl or Cmd: pans) and the wheel zooms it
function moveInset(e) {
  e.preventDefault();
  const b0 = insetBox(), x0 = e.clientX, y0 = e.clientY, el = e.currentTarget;
  if (el.setPointerCapture) el.setPointerCapture(e.pointerId);
  const move = ev => { SV.pos = [b0.x + ev.clientX - x0, window.innerHeight - (b0.y + ev.clientY - y0) - b0.h]; placeInset(); if (slMode() === "2d") drawSliceViews(); };
  const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); const b = insetBox(); SV.pos = [b.x, window.innerHeight - b.y - b.h]; saveSV(); };
  el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
}
$("sli-move").addEventListener("pointerdown", moveInset);
$("sli-cv").addEventListener("pointerdown", moveInset);
$("sli-grip").addEventListener("pointerdown", e => {
  e.preventDefault();
  const g = e.currentTarget, x0 = e.clientX, y0 = e.clientY, b0 = insetBox(), s0 = Math.max(120, Math.min(SV.size, window.innerWidth * 0.6, window.innerHeight * 0.7));
  if (!SV.pos) SV.pos = [b0.x, window.innerHeight - b0.y - b0.h];
  if (g.setPointerCapture) g.setPointerCapture(e.pointerId);
  const move = ev => { SV.size = Math.max(120, s0 + Math.max(ev.clientX - x0, y0 - ev.clientY)); placeInset(); if (slMode() === "2d") resize(); drawSliceViews(); };
  const up = () => { g.removeEventListener("pointermove", move); g.removeEventListener("pointerup", up); SV.size = Math.round(Math.max(120, Math.min(SV.size, window.innerWidth * 0.6, window.innerHeight * 0.7))); saveSV(); syncSlice(); };
  g.addEventListener("pointermove", move); g.addEventListener("pointerup", up);
});
$("sli").addEventListener("pointerdown", e => {
  if (slMode() !== "2d" || e.target !== canvas) return;
  e.preventDefault();
  const cv = canvas, pan = e.button !== 0 || e.ctrlKey || e.metaKey;
  let x = e.clientX, y = e.clientY;
  if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId);
  const move = ev => { if (pan) cam.panBy(ev.clientX - x, ev.clientY - y); else cam.rotateBy(ev.clientX - x, ev.clientY - y); x = ev.clientX; y = ev.clientY; requestRender(); drawTriad(); };
  const up = () => { cv.removeEventListener("pointermove", move); cv.removeEventListener("pointerup", up); };
  cv.addEventListener("pointermove", move); cv.addEventListener("pointerup", up);
});
$("sli").addEventListener("wheel", e => { if (slMode() !== "2d" || e.target !== canvas) return; e.preventDefault(); cam.zoomBy(Math.exp(e.deltaY * (e.deltaMode ? 0.05 : 0.0015))); requestRender(); }, { passive: false });
$("sli").addEventListener("contextmenu", e => e.preventDefault());
window.addEventListener("resize", () => { if (SL.on) syncSlice(); });

/* ------------------------------------------------------------------ trajectory analysis */
// off: per tab, the curves switched off (by key); fq: the per-frame quantities shown
// fmax: upper end of the VDOS axis in THz (null: automatic)
// x: X-ray diffraction settings (radiation name or "custom" with lam in A, 2theta range t0-t1 and peak width fw in deg,
// mode for trajectories: "frame" (follows the player), "avg" (average over frames) or "map" (2theta x time))
const AN = { kind: "rdf", rmax: 8, freq: "THz", fmax: null, res: null, seq: 0, off: { rdf: new Set(), msd: new Set(), vacf: new Set(), xrd: new Set() }, fq: new Set(["temperature"]),
  x: { rad: "Cu Kα", lam: 1.54184, t0: 10, t1: 90, fw: 0.1, mode: "frame", log: true } };
const AN_COLORS = ["#3058d0", "#e3a03a", "#2bb3a3", "#d6456a", "#7cb342", "#9b7be0", "#ef7d4b", "#5c9bd6", "#a0785a", "#c05fb0"];
const AN_TOTAL = "#6b7280";
const THZ_TO_CM = 33.35641; // cm^-1 per THz
/** The main trajectory restricted to the atoms still present (structure.ids): { structure, traj } for analysis.js. */
function anData(it) {
  const t = it.traj, st = it.structure, ids = st.ids;
  let pos = t.pos, nat = t.nat;
  if (ids) {
    nat = ids.length; pos = new Float64Array(3 * nat * t.n);
    for (let f = 0; f < t.n; f++) for (let j = 0; j < nat; j++) {
      const a = 3 * (t.nat * f + ids[j] - 1), b = 3 * (nat * f + j);
      pos[b] = t.pos[a]; pos[b + 1] = t.pos[a + 1]; pos[b + 2] = t.pos[a + 2];
    }
  }
  return { structure: { species: st.species, lattice: st.lattice, pbc: st.pbc, pos: st.pos }, traj: { n: t.n, nat, pos, cells: t.cells, energy: t.energy } };
}
const anDt = () => { const v = parseFloat($("an-dt").value); return v > 0 ? v : undefined; };
// WHY: most MD codes write 0.5-2 fs steps; without a value the temperature and spectra could not be shown at all
const DT0 = 1; // fs per saved frame assumed when neither the file nor the user gives one
/** True when the main trajectory's time step is the assumed DT0 (no typed value, none in the file). */
const dtAssumed = () => { const it = mainTraj(); return !!it && !it.dtUser && !(it.traj && it.traj.dt_fs); };
/** Time step field of the main trajectory: the value typed for it (it.dtUser), else its file's (traj.dt_fs, marked
 *  "from the file"), else DT0 (marked "assumed"). Called from summaries(), so it follows the trajectory shown. */
function syncDt() {
  const it = mainTraj(), inp = $("an-dt"), v = it ? it.dtUser || (it.traj && it.traj.dt_fs) || DT0 : "";
  if (document.activeElement !== inp && String(inp.value) !== String(v)) inp.value = v;
  $("an-dt-src").textContent = !it || it.dtUser ? "" : it.traj && it.traj.dt_fs ? "from the file" : "assumed";
}
/** Opens the analysis window on tab kind and computes it. */
function openAnalysis(kind) {
  AN.kind = kind;
  $("anwin").hidden = false;
  runAnalysis();
}
/** The item XRD is computed for: the active atoms layer's, when it has a cell (a trajectory: its current frame). */
const xrdItem = () => { const L = activeAtomsLayer(), it = L && itemOf(L); return it && it.structure && it.structure.pbc && it.structure.lattice ? it : null; };
const xrdLambda = () => (AN.x.rad === "custom" ? AN.x.lam : WAVELENGTHS[AN.x.rad] || 1.54184);
/** quiet: keep the current chart until the new one is ready (XRD following the player). */
async function runAnalysis(quiet = false) {
  const xrd = AN.kind === "xrd", it = xrd ? xrdItem() : mainTraj(), hasTraj = !!mainTraj();
  // the trajectory tabs only with a trajectory, XRD only with a crystal
  for (const b of $("an-tabs").querySelectorAll("button")) { b.setAttribute("aria-pressed", String(b.value === AN.kind)); b.hidden = b.value === "xrd" ? !xrdItem() : !hasTraj; }
  anOptions();
  if ($("anwin").hidden) return;
  if (xrd && !it) { AN.res = null; $("an-keys").innerHTML = ""; $("an-res").textContent = ""; $("an-plot").innerHTML = '<div class="an-wait">X-ray diffraction needs a crystal (a structure with a cell).</div>'; return; }
  if (!it) return;
  // WHY: an assumed step goes only to the per-frame quantities (the temperature needs one, and its note says it is
  // assumed); the VDOS keeps its own marked 1 fs default, MSD and the XRD map stay in frames until a step is given
  const dt_fs = dtAssumed() && AN.kind !== "frames" ? undefined : anDt(), seq = ++AN.seq;
  if (xrd) {
    const traj = !!it.traj && AN.x.mode !== "frame", o = { lambda: xrdLambda(), range: [AN.x.t0, AN.x.t1], fwhm: AN.x.fw };
    if (!quiet) { AN.res = null; $("an-res").textContent = ""; $("an-keys").innerHTML = ""; $("an-plot").innerHTML = `<div class="an-wait">Computing…${traj ? " (every frame, up to 200)" : ""}</div>`; }
    try {
      let r;
      if (traj) { const d = anData(it); r = await call("xrd", { mode: "traj", structure: d.structure, traj: d.traj, opts: { ...o, maxFrames: 200 } }); }
      else { const st = it.structure; r = await call("xrd", { mode: "one", structure: { species: st.species, lattice: st.lattice, pos: st.pos }, opts: o }); }
      if (seq !== AN.seq) return;
      AN.res = { kind: "xrd", r, name: it.name, mode: traj ? AN.x.mode : "frame", frame: it.traj ? S.frame : null, nFrames: it.traj ? it.traj.n : 0, dt: dt_fs, href: null, st: it.structure, it };
      drawAnalysis();
    } catch (e) { if (seq === AN.seq) { AN.res = null; $("an-keys").innerHTML = ""; $("an-plot").innerHTML = `<div class="an-wait">${esc(e.message || String(e))}</div>`; } }
    return;
  }
  AN.res = null; $("an-res").textContent = "";
  $("an-plot").innerHTML = '<div class="an-wait">Computing…</div>'; $("an-keys").innerHTML = "";
  const { structure, traj } = anData(it);
  try {
    // WHY: the VDOS needs a time step; without one it is drawn for 1 fs per frame and says so (its frequencies scale
    // as 1 / time step), MSD and per-frame series stay in frames
    const assumed = AN.kind === "vacf" && !dt_fs;
    const r = await call("analysis", { kind: AN.kind, structure, traj, opts: AN.kind === "rdf" ? { rmax: AN.rmax } : { dt_fs: assumed ? 1 : dt_fs } });
    if (seq !== AN.seq) return;
    AN.res = { kind: AN.kind, r, name: it.name, it, assumed };
    drawAnalysis();
  } catch (e) { if (seq === AN.seq) $("an-plot").innerHTML = `<div class="an-wait">${esc(e.message || String(e))}</div>`; }
}
/** The options of the current tab, in the window's bottom row. */
function anOptions() {
  const o = $("an-opts");
  if (AN.kind === "rdf") o.innerHTML = `<label>r max <input type="number" id="an-rmax" min="1" max="30" step="0.5" value="${AN.rmax}"> Å</label>`;
  else if (AN.kind === "vacf") o.innerHTML = `<div class="seg2" id="an-freq"><button type="button" value="THz" aria-pressed="${AN.freq === "THz"}">THz</button><button type="button" value="cm" aria-pressed="${AN.freq === "cm"}">cm⁻¹</button></div><label>up to <input type="number" id="an-fmax" min="0" step="any" placeholder="auto" value="${AN.fmax ? +(AN.fmax * (AN.freq === "cm" ? THZ_TO_CM : 1)).toPrecision(4) : ""}"></label>`;
  else if (AN.kind === "xrd") {
    const it = xrdItem(), x = AN.x;
    o.innerHTML = `<select id="an-rad" aria-label="Radiation">${Object.keys(WAVELENGTHS).map(r => `<option${r === x.rad ? " selected" : ""}>${r}</option>`).join("")}<option value="custom"${x.rad === "custom" ? " selected" : ""}>λ in Å</option></select>` +
      (x.rad === "custom" ? `<input type="number" id="an-lam" min="0.1" max="10" step="any" value="${x.lam}" aria-label="Wavelength in Å">` : "") +
      `<label>2θ <input type="number" id="an-t0" min="0" max="170" step="1" value="${x.t0}" aria-label="2θ from (°)">–<input type="number" id="an-t1" min="1" max="179" step="1" value="${x.t1}" aria-label="2θ to (°)">°</label>` +
      `<label title="Full width at half maximum of each peak">width <input type="number" id="an-fw" min="0.01" max="5" step="0.01" value="${x.fw}">°</label>` +
      (it && it.traj && x.mode === "map" ? `<div class="seg2" id="an-xscale" title="Colour scale of the map">${[["log", "log"], ["lin", "linear"]].map(([v, t]) => `<button type="button" value="${v}" aria-pressed="${(v === "log") === x.log}">${t}</button>`).join("")}</div>` : "") +
      (it && it.traj ? `<div class="seg2" id="an-xmode" title="Frame: the pattern of the frame shown (follows the player); Average: over the frames; Map: 2θ against time">${[["frame", "Frame"], ["avg", "Average"], ["map", "Map"]].map(([v, t]) => `<button type="button" value="${v}" aria-pressed="${x.mode === v}">${t}</button>`).join("")}</div>` : "");
  }
  else o.innerHTML = "";
}
// XRD following the player: one computation at a time, the latest frame next
let xrdBusy = false, xrdAgain = false;
async function xrdFollow() {
  if (xrdBusy) { xrdAgain = true; return; }
  xrdBusy = true;
  try { await runAnalysis(true); } finally { xrdBusy = false; }
  if (xrdAgain) { xrdAgain = false; xrdFollow(); }
}
/** Map image (data URL): one row per computed frame, latest at the top, viridis over 0-100 (linear) or over
 *  log10 I from 0.1 to 100 (log); at most 1200 columns, each the largest value of the points it covers (narrow Bragg
 *  peaks stay visible when the image is drawn small). "" without a 2D canvas. */
function xrdHeatURL(r, log) {
  try {
    const nx = r.x.length, nf = r.frames.length, W = Math.min(nx, 1200), c = document.createElement("canvas");
    c.width = W; c.height = nf;
    const g = c.getContext("2d"), img = g.createImageData(W, nf), T = new Uint8Array(768);
    for (let i = 0; i < 256; i++) T.set(colormap("viridis", i / 255), 3 * i);
    const level = I => (log ? (Math.log10(Math.max(I, 0.1)) + 1) / 3 : I / 100);
    for (let row = 0; row < nf; row++) for (let col = 0; col < W; col++) {
      let m = 0;
      for (let i = Math.floor(col * nx / W), e = Math.max(i + 1, Math.floor((col + 1) * nx / W)); i < e; i++) m = Math.max(m, r.map[row * nx + i]);
      const v = Math.max(0, Math.min(255, Math.round(255 * level(m)))), o = 4 * ((nf - 1 - row) * W + col);
      img.data[o] = T[3 * v]; img.data[o + 1] = T[3 * v + 1]; img.data[o + 2] = T[3 * v + 2]; img.data[o + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return c.toDataURL("image/png");
  } catch (e) { return ""; }
}
/** XRD chart: one pattern (profile, peak sticks, hkl labels of the strongest peaks, each switchable), the average
 *  over frames, or the 2theta x time map with a line at the frame shown. */
function xrdChart(keys, key) {
  const R0 = AN.res, r = R0.r, lam = r.lambda, rad = AN.x.rad === "custom" ? "" : AN.x.rad + " ", xl = "2*θ* (°)";
  const head = `${rad}λ = ${lam} Å`;
  if (r.map) {
    const k = R0.dt ? R0.dt / 1000 : 1, t = r.frames.map(f => (R0.dt ? f * k : f + 1)), y0 = t[0], y1 = t.length > 1 ? t[t.length - 1] : t[0] + 1, ylab = R0.dt ? "*t* (ps)" : "frame";
    if (R0.mode === "avg") return { panels: [{ series: [{ label: "average", x: r.x, y: r.avg, color: AN_COLORS[0], width: 1.3 }], ylabel: "intensity (rel.)", zero: true }], keys, xlabel: xl, xr: [r.x[0], r.x[r.x.length - 1]],
      text: `${head} · average over ${r.frames.length} of ${R0.nFrames} frames · peak width ${r.fwhm}°`, xname: "2theta (deg)" };
    if (R0.href === null || R0.hrefLog !== AN.x.log) { R0.href = xrdHeatURL(r, AN.x.log); R0.hrefLog = AN.x.log; }
    const now = R0.dt ? S.frame * k : S.frame + 1, series = now >= y0 && now <= y1 ? [{ label: "", x: [r.x[0], r.x[r.x.length - 1]], y: [now, now], color: "#ffffff", dash: true, width: 1 }] : [];
    return { panels: [{ series, ylabel: ylab, yr: [y0, y1], heat: { href: R0.href, x0: r.x[0], x1: r.x[r.x.length - 1], y0, y1, stops: [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1].map(v => `rgb(${colormap("viridis", v).join(",")})`), lo: AN.x.log ? 0.1 : 0, hi: 100 } }],
      keys, xlabel: xl, xr: [r.x[0], r.x[r.x.length - 1]], text: `${head} · ${r.frames.length} of ${R0.nFrames} frames, one intensity scale (strongest 100, colours ${AN.x.log ? "logarithmic from 0.1" : "linear"}) · dashed: the frame shown`, xname: "2theta (deg)" };
  }
  const series = [], showP = key("profile", "profile", AN_COLORS[0]);
  if (showP) series.push({ label: "", x: r.x, y: r.y, color: AN_COLORS[0], width: 1.3 });
  if (key("peaks", "peaks", AN_TOTAL)) {
    const sx = [], sy = [];
    for (const p of r.peaks) sx.push(p.tth, p.tth, NaN), sy.push(0, p.I, NaN);
    series.push({ label: "", x: sx, y: sy, color: AN_TOTAL, width: 1 });
  }
  const showH = key("hkl", "hkl", AN_TOTAL), step = r.x.length > 1 ? r.x[1] - r.x[0] : 1;
  const labels = showH ? r.peaks.filter(p => p.I >= 3).sort((a, b) => b.I - a.I).slice(0, 16)
    .map(p => ({ x: p.tth, y: Math.max(p.I, showP ? r.y[Math.max(0, Math.min(r.y.length - 1, Math.round((p.tth - r.x[0]) / step)))] : 0), text: hklText(p.fam[0].hkl) })) : [];
  const strong = r.peaks.slice().sort((a, b) => b.I - a.I).slice(0, 8).sort((a, b) => a.tth - b.tth);
  const lines = strong.map(p => `${p.tth.toFixed(2).padStart(7)}°  d ${p.d.toFixed(4)} Å  I ${p.I.toFixed(1).padStart(5)}  ${p.fam.map(f => `${hklText(f.hkl)} ×${f.m}`).join(", ")}`);
  return { panels: [{ series, labels, ylabel: "intensity (rel.)", yr: [0, showH ? 128 : 105] }], keys, xlabel: xl, xr: [r.x[0], r.x[r.x.length - 1]],
    text: `${R0.frame !== null ? `frame ${R0.frame + 1} · ` : ""}${head} · ${r.peaks.length} peaks in ${r.range[0]}–${r.range[1]}° · strongest:\n${lines.join("\n")}`, xname: "2theta (deg)" };
}
/** CSV of the XRD result: one pattern (profile columns, then the peak list: 2theta, d, intensity, hkl families with
 *  multiplicities), the average, or the map (a row per frame). */
function xrdCSV() {
  const R0 = AN.res, r = R0.r, q = s => `"${String(s).replace(/"/g, '""')}"`, plainHkl = v => v.join(" ");
  if (r.map && R0.mode === "map") {
    const head = [R0.dt ? "t (ps)" : "frame", ...Array.from(r.x, v => v.toFixed(3))].join(","), nx = r.x.length;
    return [head, ...r.frames.map((f, row) => [R0.dt ? (f * R0.dt / 1000).toPrecision(6) : f + 1, ...Array.from(r.map.subarray(row * nx, (row + 1) * nx), v => +v.toFixed(4))].join(","))].join("\n") + "\n";
  }
  if (r.map) return ["2theta (deg),intensity (average)", ...Array.from(r.x, (v, i) => `${v.toFixed(3)},${r.avg[i].toFixed(5)}`)].join("\n") + "\n";
  const n = Math.max(r.x.length, r.peaks.length), rows = [["2theta (deg)", "profile", "", "peak 2theta (deg)", "d (A)", "intensity", "hkl (multiplicity)"].map(q).join(",")];
  for (let i = 0; i < n; i++) {
    const p = r.peaks[i];
    rows.push([i < r.x.length ? r.x[i].toFixed(3) : "", i < r.x.length ? r.y[i].toFixed(5) : "", "", p ? p.tth.toFixed(5) : "", p ? p.d.toFixed(6) : "", p ? p.I.toFixed(4) : "", p ? q(p.fam.map(f => `${plainHkl(f.hkl)} (${f.m})`).join("; ")) : ""].join(","));
  }
  return rows.join("\n") + "\n";
}
/** First peak of g(r) and the first minimum after it (indices), for coordination numbers. g is first smoothed by a
 *  running mean over +-w bins (w = 0.1 A / dr) so that noise does not end the shell early; the minimum is the lowest
 *  point before the smoothed curve rises again by 10% of the peak. */
function firstShell(g, dr) {
  const w = Math.max(1, Math.round(0.1 / dr)), sm = new Float64Array(g.length);
  for (let i = 0; i < g.length; i++) { let a = 0, c = 0; for (let j = Math.max(0, i - w); j <= Math.min(g.length - 1, i + w); j++) { a += g[j]; c++; } sm[i] = a / c; }
  let p = 0;
  for (let i = 1; i < sm.length; i++) if (sm[i] > sm[p]) p = i;
  let m = p;
  for (let i = p + 1; i < sm.length; i++) { if (sm[i] < sm[m]) m = i; else if (sm[i] > sm[m] + 0.1 * sm[p]) break; }
  return { p, m };
}
/** Series, axis titles and a result text for the current result: { panels: [{ series, ylabel, zero }], keys: [{ key,
 *  label, color, dash, on }] (the curves the user can switch off), xlabel, xname (CSV), text }. Only curves that are
 *  on are in panels and in the text; colours stay tied to the curve whichever others are off. */
function anChart() {
  const { kind, r } = AN.res, off = AN.off[kind] || new Set(), keys = [], key = (k, label, color, dash) => { const on = !off.has(k); keys.push({ key: k, label, color, dash, on }); return on; };
  if (kind === "xrd") return xrdChart(keys, key);
  if (kind === "rdf") {
    if (!r.total.g) return { panels: [{ series: [{ label: "all atoms", x: r.rEdge, y: r.total.n, color: AN_TOTAL }], ylabel: "neighbours within *r*" }], keys, xlabel: "*r* (Å)", text: "No cell: neighbour counts instead of g(r).", xname: "r (A)" };
    const series = [], lines = [];
    if (key("all", "all", AN_TOTAL)) series.push({ label: "all", x: r.r, y: r.total.g, color: AN_TOTAL, width: 1.2 });
    r.pairs.forEach((q, i) => {
      const c = AN_COLORS[i % AN_COLORS.length];
      if (!key(q.label, q.label, c)) return;
      series.push({ label: q.label, x: r.r, y: q.g, color: c });
      const { p, m } = firstShell(q.g, r.dr);
      if (q.g[p] > 0) lines.push(`${q.label.padEnd(7)} peak ${r.r[p].toFixed(2)} Å   ${q.b} around ${q.a}: ${q.nab[m].toFixed(2)}${q.a !== q.b ? `, ${q.a} around ${q.b}: ${q.nba[m].toFixed(2)}` : ""} (to ${r.rEdge[m].toFixed(2)} Å)`);
    });
    lines.push(`${r.frames.length} frames averaged`);
    return { panels: [{ series, ylabel: "*g*(*r*)", zero: true }], keys, xlabel: "*r* (Å)", xr: [0, r.rmax], text: lines.join("\n"), xname: "r (A)" };
  }
  if (kind === "msd") {
    const ps = r.timeUnit === "ps", series = [], lines = [], fits = r.elements.some(e => Number.isFinite(e.slope));
    r.elements.forEach((e, i) => key(e.el, e.el, AN_COLORS[i % AN_COLORS.length]));
    const showFits = fits && key("fits", "fits", AN_TOTAL, true);
    r.elements.forEach((e, i) => {
      if (off.has(e.el)) return;
      const c = AN_COLORS[i % AN_COLORS.length];
      series.push({ label: e.el, x: r.t, y: e.msd, color: c });
      if (showFits && Number.isFinite(e.slope)) series.push({ label: "", x: e.fit, y: e.fit.map(t => e.slope * t + e.intercept), color: c, dash: true, width: 1 });
      lines.push(ps && e.D_cm2s != null ? `${e.el.padEnd(3)} D = ${e.D_cm2s.toExponential(2)} cm²/s  (${e.D_A2ps.toPrecision(3)} Å²/ps, ${e.count} atoms)` : `${e.el.padEnd(3)} ${e.count} atoms`);
    });
    if (!ps) lines.push("Set Time step (Trajectory panel) for the time axis and D.");
    else lines.push(`fit over ${r.elements[0] ? r.elements[0].fit.map(t => t.toPrecision(3)).join("–") : ""} ps (dashed); drift ${r.removeDrift ? "removed" : "kept"}`);
    return { panels: [{ series, ylabel: "MSD (Å²)", zero: true }], keys, xlabel: ps ? "*t* (ps)" : "lag (frames)", text: lines.join("\n"), xname: ps ? "t (ps)" : "lag (frames)" };
  }
  if (kind === "vacf") {
    const cm = AN.freq === "cm", x = cm ? r.nu_cm : r.nu_THz, k = cm ? 1 / THZ_TO_CM : 1, series = [];
    if (key("total", "total", AN_TOTAL)) series.push({ label: "total", x, y: Array.from(r.total.vdos, v => v * k), color: AN_TOTAL, width: 1.2 });
    if (r.elements.length > 1) r.elements.forEach((e, i) => { const c = AN_COLORS[i % AN_COLORS.length]; if (key(e.el, e.el, c)) series.push({ label: e.el, x, y: Array.from(e.vdos, v => v * k), color: c }); });
    const f = v => (cm ? `${(v * THZ_TO_CM).toFixed(0)} cm⁻¹` : `${v.toPrecision(3)} THz`), top = (AN.fmax || vdosEnd(r)) * (cm ? THZ_TO_CM : 1);
    return { panels: [{ series, ylabel: cm ? "VDOS (1/cm⁻¹)" : "VDOS (1/THz)", zero: true }], keys, xlabel: cm ? "frequency (cm⁻¹)" : "frequency (THz)", xr: [0, top],
      text: `${AN.res.assumed ? "Time step not set: 1 fs per frame assumed (frequencies scale as 1 / time step; set it in the Trajectory panel)\n" : ""}resolution ${f(r.resolution_THz)}, up to ${f(r.nyquist_THz)} (Nyquist); from finite differences of positions`, xname: cm ? "frequency (cm-1)" : "frequency (THz)" };
  }
  // per frame: each chosen quantity in its own panel over a shared time axis; a, b, c share one panel
  const have = AN_Q.filter(q => r[q.k] && Array.from(r[q.k]).some(Number.isFinite));
  if (!have.some(q => AN.fq.has(q.k)) && have.length) AN.fq.add(have[0].k);
  const x = r.t ? Array.from(r.t) : Array.from({ length: r.n }, (_, i) => i + 1), panels = [], lines = [];
  let cell = null;
  for (const q of have) {
    const on = AN.fq.has(q.k);
    keys.push({ key: q.k, label: q.k, color: q.c, on });
    if (!on) continue;
    const y = Array.from(r[q.k]), vals = y.filter(Number.isFinite), mean = vals.reduce((a, v) => a + v, 0) / vals.length;
    let sd = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(1, vals.length - 1));
    if (sd < 1e-9 * Math.abs(mean)) sd = 0; // a fixed cell: rounding only
    lines.push(`${q.k.padEnd(11)} mean ${fmt(mean)} ± ${fmt(sd)} ${q.u} over ${vals.length} frames`);
    const s = { label: q.cell ? q.k : `${q.k} (${q.u})`, x, y, color: q.c };
    if (q.cell) { if (!cell) panels.push(cell = { series: [], ylabel: "cell length (Å)" }); cell.series.push(s); }
    else panels.push({ series: [s], ylabel: q.t });
  }
  if (!have.length) lines.push("Not available for this file.");
  if (AN.fq.has("temperature") && have.some(q => q.k === "temperature")) {
    lines.push("temperature: from finite differences of saved positions, right only when every MD step (or every other) is saved");
    if (dtAssumed()) lines.push(`temperature: the time step is assumed (${DT0} fs per frame); type the real one, as T scales with 1/Δt²`);
  }
  else if (!r.temperature) lines.push("temperature: needs the time step (Trajectory panel)");
  return { panels, keys, xlabel: r.t ? "*t* (ps)" : "frame", text: lines.join("\n"), xname: r.t ? "t (ps)" : "frame" };
}
/** Automatic upper end of the VDOS axis (THz): 1.25 times the frequency below which 99.5% of the total VDOS lies
 *  (cumulative sum over the frequency bins), at most the Nyquist frequency. WHY: the spectrum usually ends far below
 *  the Nyquist frequency of the saved frames, and a figure up to it is mostly empty. */
function vdosEnd(r) {
  const g = r.total.vdos, nu = r.nu_THz;
  let sum = 0, c = 0;
  for (let i = 0; i < g.length; i++) sum += Math.max(0, g[i]);
  for (let i = 0; i < g.length; i++) { c += Math.max(0, g[i]); if (c >= 0.995 * sum) return Math.min(r.nyquist_THz, 1.25 * nu[i]) || r.nyquist_THz; }
  return r.nyquist_THz;
}
/** Quantities per frame: key, axis title, unit, colour; cell: drawn together in one panel. */
const AN_Q = [
  { k: "temperature", t: "temperature estimate (K)", u: "K" }, { k: "energy", t: "energy (eV)", u: "eV" }, { k: "volume", t: "volume (Å³)", u: "Å³" },
  { k: "density", t: "density (g/cm³)", u: "g/cm³" }, { k: "a", t: "a (Å)", u: "Å", cell: true }, { k: "b", t: "b (Å)", u: "Å", cell: true }, { k: "c", t: "c (Å)", u: "Å", cell: true },
].map((q, i) => ({ ...q, c: AN_COLORS[i % AN_COLORS.length] }));
/** Chart options for the window (screen) or for a saved figure (paper: 3.5 in wide, the width of one journal column,
 *  black ink, Helvetica; 11-unit tick labels print at about 8 pt). */
function anFigure(c, paper) {
  const k = Math.max(1, c.panels.length);
  if (paper) { const W = 340, H = 240 + 92 * (k - 1); return { ...c, W, H, fs: 11, ink: "#000", font: "Helvetica, Arial, sans-serif", size: ["3.5in", `${(3.5 * H / W).toFixed(3)}in`] }; }
  const big = $("anwin").classList.contains("big");
  return { ...c, W: big ? 800 : 520, H: (big ? 420 : 300) + (big ? 150 : 110) * (k - 1), fs: big ? 12 : 11 };
}
/** Draws the current result. The chart's height is lowered (to a floor of 150 + 70 per extra panel SVG units) so the
 *  whole window fits between its anchor and the edge of the page: the bottom anchor when it was not moved, else its
 *  top; the window then scrolls rather than leave the page. */
function drawAnalysis() {
  if (!AN.res) return;
  anOptions();
  const c = anChart(), fig = anFigure(c, false), win = $("anwin"), plotBox = $("an-plot");
  $("an-keys").innerHTML = c.keys.length > 1 ? c.keys.map(q => `<button type="button" data-key="${esc(q.key)}" aria-pressed="${q.on}" title="${q.on ? "Hide" : "Show"} ${esc(q.label)}"><i class="${q.dash ? "d" : ""}" style="--c:${q.color}"></i>${esc(q.label)}</button>`).join("") : "";
  $("an-res").textContent = c.text || "";
  const pw = plotBox.clientWidth;
  if (pw > 0 && !win.hidden) {
    plotBox.innerHTML = "";
    const r = win.getBoundingClientRect(), moved = win.style.top && win.style.top !== "auto";
    const room = (moved ? window.innerHeight - r.top : r.bottom) - 12, rest = win.offsetHeight - plotBox.offsetHeight;
    const k = Math.max(1, c.panels.length), floor = 150 + 70 * (k - 1), maxH = (room - rest) * fig.W / pw;
    fig.H = Math.round(Math.max(floor, Math.min(fig.H, maxH)));
  }
  plotBox.innerHTML = chart(fig);
}
$("an-open").addEventListener("click", e => { const b = e.target.closest("[data-an]"); if (b) openAnalysis(b.dataset.an); });
$("an-tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { AN.kind = b.value; runAnalysis(); } });
$("an-x").addEventListener("click", () => { $("anwin").hidden = true; AN.seq++; });
window.addEventListener("resize", () => { if (!$("anwin").hidden && AN.res) drawAnalysis(); });
$("an-big").addEventListener("click", () => { const on = !$("anwin").classList.contains("big"); $("anwin").classList.toggle("big", on); $("an-big").setAttribute("aria-pressed", String(on)); drawAnalysis(); });
$("an-dt").addEventListener("change", () => {
  const it = mainTraj(); if (it) it.dtUser = anDt() || null;
  syncDt(); saveSoon();
  if (!$("anwin").hidden && AN.kind !== "rdf") runAnalysis();
});
$("an-opts").addEventListener("change", e => {
  const xi = { "an-t0": "t0", "an-t1": "t1", "an-fw": "fw", "an-lam": "lam" }[e.target.id];
  if (xi) {
    const v = parseFloat(e.target.value), lim = { t0: [0, 170], t1: [1, 179], fw: [0.01, 5], lam: [0.1, 10] }[xi];
    if (Number.isFinite(v)) AN.x[xi] = Math.min(lim[1], Math.max(lim[0], v));
    if (AN.x.t1 <= AN.x.t0 + 1) AN.x.t1 = Math.min(179, AN.x.t0 + 1);
    runAnalysis(); return;
  }
  if (e.target.id === "an-rad") { AN.x.rad = e.target.value; runAnalysis(); return; }
  if (e.target.id === "an-fmax") { const v = parseFloat(e.target.value); AN.fmax = v > 0 ? v / (AN.freq === "cm" ? THZ_TO_CM : 1) : null; drawAnalysis(); }
  if (e.target.id === "an-rmax") { AN.rmax = Math.min(30, Math.max(1, +e.target.value || 8)); runAnalysis(); }
});
$("an-opts").addEventListener("click", e => {
  const b = e.target.closest("#an-freq button");
  if (b) { AN.freq = b.value; drawAnalysis(); }
  const sc = e.target.closest("#an-xscale button");
  if (sc) { AN.x.log = sc.value === "log"; drawAnalysis(); }
  const m = e.target.closest("#an-xmode button");
  if (m && m.value !== AN.x.mode) { AN.x.mode = m.value; runAnalysis(); }
});
$("xrd-open").addEventListener("click", () => openAnalysis("xrd"));
// a curve chip switches its curve on or off; per frame, at least one quantity stays on
$("an-keys").addEventListener("click", e => {
  const b = e.target.closest("[data-key]");
  if (!b || !AN.res) return;
  const k = b.dataset.key, set = AN.res.kind === "frames" ? AN.fq : AN.off[AN.res.kind];
  if (AN.res.kind === "frames" && set.has(k) && set.size === 1) return;
  if (set.has(k)) set.delete(k); else set.add(k);
  drawAnalysis();
});
const anFile = ext => `${(AN.res ? AN.res.name : "trajectory").replace(/[^\w.+-]+/g, "_")}-${AN.kind === "vacf" ? "vdos" : AN.kind}.${ext}`;
$("an-csv").addEventListener("click", () => {
  if (!AN.res) return;
  if (AN.res.kind === "xrd") { download(new Blob([xrdCSV()], { type: "text/csv" }), anFile("csv")); return; }
  const c = anChart(); download(new Blob([toCSV(c.panels.flatMap(p => p.series).filter(s => s.label), c.xname || "x")], { type: "text/csv" }), anFile("csv")); });
$("an-svg").addEventListener("click", () => { if (!AN.res) return; download(new Blob([chart(anFigure(anChart(), true))], { type: "image/svg+xml" }), anFile("svg")); });
/** The saved figure as PNG at 600 dpi on white: the paper SVG drawn onto a canvas. */
$("an-png").addEventListener("click", () => {
  if (!AN.res) return;
  const f = anFigure(anChart(), true), svg = chart(f), dpi = 600, w = Math.round(3.5 * dpi), h = Math.round(w * f.H / f.W);
  const img = new Image(), url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  img.onload = () => {
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d"); g.fillStyle = "#fff"; g.fillRect(0, 0, w, h); g.drawImage(img, 0, 0, w, h);
    URL.revokeObjectURL(url);
    c.toBlob(b => download(b, anFile("png")), "image/png");
  };
  img.src = url;
});
$("an-head").addEventListener("pointerdown", e => {
  if (e.target.closest("button, input")) return;
  const win = $("anwin"), r = win.getBoundingClientRect(), x0 = e.clientX - r.left, y0 = e.clientY - r.top;
  const move = ev => { win.style.left = Math.max(0, ev.clientX - x0) + "px"; win.style.top = Math.max(0, ev.clientY - y0) + "px"; win.style.right = win.style.bottom = "auto"; };
  const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
});

/* ------------------------------------------------------------------ use elsewhere: website embed, notebooks */
/** Cards behind "Embed in your website" and the "Use in" symbols (top right of the opening page): install lines and
 *  code to copy. The packages live in this repo (python/, julia/) and load the viewer from SITE with ?embed. */
const SITE = "https://utkarsh.science/isoview/";
const COPY_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8.5" y="8.5" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M15.5 8.5V6A1.5 1.5 0 0 0 14 4.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>';
const codeBlock = (t, id) => `<div class="code"><pre${id ? ` id="${id}"` : ""}>${esc(t)}</pre><button type="button" class="copy" title="Copy" aria-label="Copy">${COPY_SVG}</button></div>`;
const embedSnippet = u => `<iframe src="${SITE}?embed&url=${encodeURIComponent(u || "https://example.org/files/CHGCAR")}" width="100%" height="480" style="border:0;border-radius:12px" loading="lazy" allowfullscreen></iframe>`;
/** The files of the scene for the notebook snippets: names in quotes, at most three (default "CHGCAR"). */
function useNames() {
  const n = [...S.items.values()].flatMap(it => (it.files && it.files.length ? it.files.map(f => f.name) : [it.name])).filter(s => s && !/computed here/.test(s));
  return n.length ? n.slice(0, 3) : ["CHGCAR"];
}
const pyArgs = () => useNames().map(s => JSON.stringify(s)).join(", ");
const USE = {
  web: ["Embed in your website", () => `<p>Show a structure, density or trajectory in any web page. The file's address:</p><input type="url" id="use-url" placeholder="https://example.org/files/CHGCAR" spellcheck="false" aria-label="Address of the file" value="${esc(S.layers.length && S.srcUrl ? S.srcUrl : "")}"><p class="lbl">Paste into your page</p>${codeBlock(embedSnippet(""), "use-snip")}<p class="fine2">The server with the file must allow cross-origin requests (GitHub Pages and raw.githubusercontent.com do). Visitors can turn it, open the panel and save images; the file is read in their browser.</p>`],
  jupyter: ["Use in Jupyter", () => `<p class="lbl">Install</p>${codeBlock('pip install "isoview-notebook @ git+https://github.com/utksi/isoview#subdirectory=python"')}<p class="lbl">In a notebook</p>${codeBlock(S.layers.length ? `import isoview\nisoview.show(${pyArgs()})\nisoview.show(atoms)  # ASE Atoms, pymatgen Structure` : 'import isoview\nisoview.show("CHGCAR")             # any file isoview opens\nisoview.show("WAVECAR", "CONTCAR")  # several files, one scene\nisoview.show(atoms)                 # ASE Atoms, pymatgen Structure')}<p class="fine2">JupyterLab, Notebook, VS Code and Colab. The viewer loads from this site and reads the files in your browser; nothing is uploaded.</p>`],
  julia: ["Use in Julia", () => `<p class="lbl">Install</p>${codeBlock('using Pkg\nPkg.add(url="https://github.com/utksi/isoview", subdir="julia")')}<p class="lbl">In a notebook</p>${codeBlock(S.layers.length ? `using Isoview\nisoview(${pyArgs()})` : 'using Isoview\nisoview("CHGCAR")             # any file isoview opens\nisoview("WAVECAR", "CONTCAR")  # several files, one scene')}<p class="fine2">Pluto, IJulia (Jupyter) and VS Code. The viewer loads from this site and reads the files in your browser; nothing is uploaded.</p>`],
};
let useOpen = null;
/** Opens card key (web, jupyter, julia) or closes it (null). From the opening page the card hangs below the row of
 *  buttons; from a scene (the </> view button) it sits above the view buttons with tabs for the three cards. */
function showUse(key, scene = false) {
  const pop = $("use-pop");
  for (const b of document.querySelectorAll("[data-use]")) b.setAttribute("aria-expanded", String(b.dataset.use === key));
  $("btn-use").setAttribute("aria-expanded", String(!!key && scene));
  useOpen = key;
  if (!key) { pop.hidden = true; return; }
  pop.classList.toggle("scene", scene);
  $("use-tabs").hidden = !scene;
  for (const b of $("use-tabs").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === key));
  $("use-h").textContent = USE[key][0];
  $("use-body").innerHTML = USE[key][1]();
  pop.hidden = false;
  const u = $("use-url");
  if (u) {
    const upd = () => { $("use-snip").textContent = embedSnippet(u.value.trim()); };
    u.addEventListener("input", upd); upd();
    if (!u.value) u.focus();
  }
}
$("btn-use").addEventListener("click", e => { e.stopPropagation(); showUse(useOpen && $("use-pop").classList.contains("scene") ? null : "web", true); });
$("use-tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { e.stopPropagation(); showUse(b.value, true); } });
for (const b of document.querySelectorAll("[data-use]")) b.addEventListener("click", e => { e.stopPropagation(); showUse(useOpen === b.dataset.use ? null : b.dataset.use); });
$("use-x").addEventListener("click", () => showUse(null));
$("use-pop").addEventListener("click", e => {
  const c = e.target.closest(".copy");
  if (!c) return;
  const t = c.parentElement.querySelector("pre").textContent;
  (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => toast("Copied."), () => toast("Select the text and copy it.", true));
});
document.addEventListener("click", e => { if (useOpen && !e.target.closest("#use-pop") && !e.target.closest("#btn-use")) showUse(null); });
document.addEventListener("keydown", e => { if (useOpen && e.key === "Escape") { e.stopPropagation(); showUse(null); } }, true);

startWorker();
// WHY: a canvas does not wait for a web font; once the label face has loaded, redraw the label atlas with it
if (typeof document !== "undefined" && document.fonts && document.fonts.load) document.fonts.load(`600 16px ${css("--sans")}`)
  .then(() => { if (R && S.layers.length) { S.atlas = atlasOf(atlasSpecies()); buildStructure(); } }, () => {});
segState();
// test hook (tests set globalThis.__isoviewTest before importing this module)
if (globalThis.__isoviewTest) globalThis.__isoview = { S, SL, SV, AN, cam, serialize, savePhoto, requestRender, get disp() { return disp; }, get canvas() { return canvas; } };
/* ------------------------------------------------------------------ embedding */
/**
 * Embedded mode (?embed, in an iframe on a website or in a notebook): nothing is kept in or restored from this
 * browser, the opening page is never shown, the panel (unless ?panel=1) and the axes corner start tucked away, and the
 * wordmark opens the full site in a new tab. Options: ?theme=light|dark, ?style=<Structure style>, ?look=<Look>.
 * A parent page (the notebook packages in python/ and julia/) sends files by postMessage:
 *   { isoview: "open", files: [{ name, base64 } or { name, data: ArrayBuffer | Uint8Array | string }], mode: "replace" | "add" }
 * and receives { isoview: "ready" } once this page listens, then { isoview: "opened", layers } after each open.
 */
function initEmbed() {
  document.documentElement.classList.add("embed");
  $("empty").hidden = true;
  $("btn-home").title = "Open in isoview (new tab)";
  if (qp.get("panel") !== "1") tuck("tr", true);
  tuck("bl", true);
  const th = qp.get("theme");
  // WHY: set directly, not through the Theme buttons, which would also store the choice for the full site
  if (th === "light" || th === "dark") { S.theme = th; document.documentElement.setAttribute("data-theme", th); restyle(); }
  for (const k of ["style", "look"]) {
    const v = qp.get(k), b = v && /^\w+$/.test(v) && document.querySelector(`.seg[data-key="${k}"] button[value="${v}"]`);
    if (b) b.click();
  }
  const b64 = s => { const t = window.atob(s), u = new Uint8Array(t.length); for (let i = 0; i < t.length; i++) u[i] = t.charCodeAt(i); return u; };
  window.addEventListener("message", async e => {
    const m = e.data;
    if (!m || m.isoview !== "open" || !Array.isArray(m.files) || !m.files.length) return;
    let files;
    try { files = m.files.map(f => new File([f.base64 != null ? b64(f.base64) : f.data], String(f.name || "file"))); }
    catch (err) { toast("Could not read the files sent to the viewer.", true); return; }
    await openFiles(files, m.mode === "add" ? "add" : "replace");
    if (e.source) e.source.postMessage({ isoview: "opened", layers: S.layers.length }, "*");
  });
  if (window.parent !== window) window.parent.postMessage({ isoview: "ready" }, "*");
}
if (EMBED) initEmbed();
if (qp.get("url")) {
  const u = qp.get("url");
  fetch(u).then(r => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.blob(); })
    .then(b => openFiles([new File([b], decodeURIComponent(u.split(/[?#]/)[0].split("/").pop() || "file"))], "replace"))
    .then(() => { if (S.layers.length) S.srcUrl = u; })
    .catch(e => toast(`Could not fetch ${u}: ${e.message}. The server must allow cross-origin requests.`, true, 9000));
} else if (qp.get("example")) example(qp.get("example"));
else if (!EMBED) store.getState().then(st => { if (st && !S.layers.length) restoreScene(st, store.getFiles); });
