/**
 * isoview application: file loading, UI state and drawing with 3Dmol.js (global $3Dmol).
 *
 * Isosurfaces: a field is resampled by field.renderGrid (periodic closure, supercell tiling, stride) and handed to
 * 3Dmol as a VolumeData whose index -> Cartesian map is the matrix [va vb vc origin] built from the voxel vectors
 * (a 2x2x2 cube header sets the matrix, then size and data are replaced). Level slider: log10 |iso| over six decades
 * below max |f| for densities and signed fields (+iso and, when the field has a negative part, -iso), linear over
 * [min, max] for potentials and ELF. The readout gives the share of sum |f| inside the surface (field.fractionAbove).
 * Views along a lattice vector: rows [right, up, toward] of the world-to-view rotation, converted to the quaternion
 * that 3Dmol stores in getView()[4..7].
 */
import { detectKind, parseText, isWavecarHeader } from "./formats.js";
import { Wavecar } from "./wavecar.js";
import { stats, fractionAbove, isoForFraction, autoStride, renderGrid } from "./field.js";
import { displayAtoms, bondList } from "./structure.js";
import { colorOf, rcov, IONIC_CATIONS } from "./elements.js";
import { params, formula, det3, norm, cross, dot } from "./lattice.js";

const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const S = {
  name: "", structure: null, fields: [], wave: null, fi: 0, st: null,
  iso: 0, p: 0.5, mode: "log", pm: true, surf: true, opacity: 0.85, res: "auto",
  style: "ball", scale: 1, bonds: true, tol: 1.15, ionic: false, cell: true, reps: [1, 1, 1], boundary: true,
  proj: "persp", theme: "auto",
  w: { s: 0, k: 0, b: -1, mode: "density", half: "x", cache: new Map() },
};
let viewer = null, model = null, cellShapes = [], isoShapes = [];
const vols = new Map();

/* ------------------------------------------------------------------ small helpers */
const unit = v => { const n = norm(v); return v.map(x => x / n); };
const fmt = x => {
  if (!Number.isFinite(x)) return "–";
  const a = Math.abs(x);
  return a !== 0 && (a < 1e-2 || a >= 1e4) ? x.toExponential(2).replace("e", "e") : +x.toPrecision(3) + "";
};
const sup = s => s.replace(/e([+-]?)(\d+)/, (m, sg, d) => "×10" + (sg === "-" ? "⁻" : "") + String(+d).replace(/\d/g, c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]));
let toastTimer = 0;
function toast(msg, err = false, ms = 5200) {
  const t = $("toast");
  t.textContent = msg; t.classList.toggle("err", err); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), ms);
}
const busy = on => ($("busy").hidden = !on);
const tick = () => new Promise(r => setTimeout(r, 16));

/* ------------------------------------------------------------------ viewer */
function ensureViewer() {
  if (viewer) return true;
  if (typeof $3Dmol === "undefined") { toast("The 3D library did not load. Check the connection and reload.", true, 9000); return false; }
  try { viewer = $3Dmol.createViewer($("stage"), { backgroundColor: css("--bg"), antialias: true }); }
  catch (e) { viewer = null; }
  if (!viewer) { toast("This browser has WebGL turned off, so the 3D view cannot start.", true, 9000); return false; }
  viewer.setViewChangeCallback(() => requestAnimationFrame(drawTriad));
  return true;
}

/** 3Dmol VolumeData for a render grid (see the file docstring). */
function makeVolume(g) {
  const [va, vb, vc] = g.vox, o = g.origin, f = x => x.toFixed(9);
  const head = `isoview\nvolume\n0 ${f(o[0])} ${f(o[1])} ${f(o[2])}\n-2 ${f(va[0])} ${f(va[1])} ${f(va[2])}\n` +
    `-2 ${f(vb[0])} ${f(vb[1])} ${f(vb[2])}\n-2 ${f(vc[0])} ${f(vc[1])} ${f(vc[2])}\n0 0 0 0 0 0 0 0\n`;
  const vol = new $3Dmol.VolumeData(head, "cube");
  vol.size = { x: g.size[0], y: g.size[1], z: g.size[2] };
  vol.data = g.data;
  vol.inversematrix = null;
  return vol;
}

function atomStyle(el) {
  const r = rcov(el), k = S.scale;
  if (S.style === "space") return { sphere: { radius: r * k } };
  if (S.style === "stick") return { stick: { radius: 0.14 * k }, sphere: { radius: 0.14 * k } };
  const s = { sphere: { radius: Math.max(0.2, 0.42 * r) * k } };
  if (S.bonds) s.stick = { radius: 0.1 * k };
  return s;
}

function drawStructure() {
  if (!viewer) return;
  if (model) { viewer.removeModel(model); model = null; }
  for (const s of cellShapes) viewer.removeShape(s);
  cellShapes = [];
  const st = S.structure;
  if (!st) return;
  const reps = st.pbc ? S.reps : [1, 1, 1];
  if (S.style !== "off" && st.species.length) {
    const atoms = displayAtoms(st, { reps, boundary: S.boundary && st.pbc });
    const bonds = S.bonds && S.style !== "space" ? bondList(atoms, { tol: S.tol, skip: S.ionic ? new Set() : IONIC_CATIONS }) : [];
    const spec = atoms.map((a, i) => ({ elem: a.el, x: a.x, y: a.y, z: a.z, index: i, serial: i, bonds: [], bondOrder: [], color: colorOf(a.el) }));
    for (const [i, j] of bonds) { spec[i].bonds.push(j); spec[i].bondOrder.push(1); spec[j].bonds.push(i); spec[j].bondOrder.push(1); }
    model = viewer.addModel();
    model.addAtoms(spec);
    for (const el of new Set(atoms.map(a => a.el))) model.setStyle({ elem: el }, atomStyle(el));
  }
  if (S.cell && st.lattice) {
    const col = css("--cell");
    const box = (A, B, C, dashed) => {
      const P = (i, j, k) => ({ x: i * A[0] + j * B[0] + k * C[0], y: i * A[1] + j * B[1] + k * C[1], z: i * A[2] + j * B[2] + k * C[2] });
      const E = [[0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1]];
      for (const e of E) cellShapes.push(viewer.addLine({ start: P(e[0], e[1], e[2]), end: P(e[3], e[4], e[5]), color: col, dashed }));
    };
    const L = st.lattice;
    box(L[0], L[1], L[2], false);
    if (reps.some(r => r > 1)) box(L[0].map(x => x * reps[0]), L[1].map(x => x * reps[1]), L[2].map(x => x * reps[2]), true);
  }
}

function drawSurface(preview = false) {
  if (!viewer) return;
  for (const s of isoShapes) viewer.removeShape(s);
  isoShapes = [];
  const f = S.fields[S.fi];
  if (f && S.surf) {
    const reps = f.periodic && S.structure && S.structure.pbc ? S.reps : [1, 1, 1];
    const budget = preview ? 1.2e6 : 6e6;
    let stride = S.res === "auto" ? autoStride(f, reps, budget) : +S.res;
    if (preview) stride = Math.max(stride, autoStride(f, reps, budget));
    const key = S.fi + "|" + reps.join("x") + "|" + stride;
    let vol = vols.get(key);
    if (!vol) {
      vol = makeVolume(renderGrid(f, reps, stride));
      vols.set(key, vol);
      if (vols.size > 4) vols.delete(vols.keys().next().value);
    }
    const spec = c => ({ color: c, opacity: S.opacity, smoothness: 1 });
    isoShapes.push(viewer.addIsosurface(vol, { isoval: S.iso, ...spec(css("--pos")) }));
    if (S.mode === "log" && S.pm && f.kind === "signed" && S.st.min < -1e-6 * S.st.maxAbs) isoShapes.push(viewer.addIsosurface(vol, { isoval: -S.iso, ...spec(css("--neg")) }));
  }
  viewer.render();
}

let rafSurf = 0;
function surfaceSoon(preview) {
  cancelAnimationFrame(rafSurf);
  rafSurf = requestAnimationFrame(() => { drawSurface(preview); summaries(); });
}

/* ------------------------------------------------------------------ orientation */
function quatFromRows(R) {
  const [m11, m12, m13] = R[0], [m21, m22, m23] = R[1], [m31, m32, m33] = R[2], tr = m11 + m22 + m33;
  let x, y, z, w, s;
  if (tr > 0) { s = 0.5 / Math.sqrt(tr + 1); w = 0.25 / s; x = (m32 - m23) * s; y = (m13 - m31) * s; z = (m21 - m12) * s; }
  else if (m11 > m22 && m11 > m33) { s = 2 * Math.sqrt(1 + m11 - m22 - m33); w = (m32 - m23) / s; x = 0.25 * s; y = (m12 + m21) / s; z = (m13 + m31) / s; }
  else if (m22 > m33) { s = 2 * Math.sqrt(1 + m22 - m11 - m33); w = (m13 - m31) / s; x = (m12 + m21) / s; y = 0.25 * s; z = (m23 + m32) / s; }
  else { s = 2 * Math.sqrt(1 + m33 - m11 - m22); w = (m21 - m12) / s; x = (m13 + m31) / s; y = (m23 + m32) / s; z = 0.25 * s; }
  return [x, y, z, w];
}
/** Rotation rows for a view with `toward` pointing at the viewer and `up` (orthogonalised) pointing up. */
function rows(toward, up) {
  const t = unit(toward), u0 = up.map((x, i) => x - dot(up, t) * t[i]);
  const u = norm(u0) > 1e-6 ? unit(u0) : unit(cross(t, [1, 0, 0]));
  return [cross(u, t), u, t];
}
function setRotation(R) {
  const q = quatFromRows(R), v = viewer.getView();
  v[4] = q[0]; v[5] = q[1]; v[6] = q[2]; v[7] = q[3];
  viewer.setView(v);
}
function viewAlong(i) {
  const L = S.structure && S.structure.lattice;
  if (!viewer || !L) return;
  setRotation(rows(L[i], L[(i + 2) % 3]));
  viewer.zoomTo(); viewer.render(); drawTriad();
}
function resetView() {
  if (!viewer) return;
  const L = S.structure && S.structure.lattice;
  if (L) {
    const a = unit(L[0]), b = unit(L[1]), c = unit(L[2]);
    setRotation(rows([0, 1, 2].map(i => a[i] + 0.36 * b[i] + 0.3 * c[i]), c));
  }
  viewer.zoomTo(); viewer.zoom(0.92); viewer.render(); drawTriad();
}
function drawTriad() {
  const svg = $("triad"), L = S.structure && S.structure.lattice;
  if (!viewer || !L) { svg.innerHTML = ""; return; }
  const v = viewer.getView(), [qx, qy, qz, qw] = [v[4], v[5], v[6], v[7]];
  const rot = p => { // standard active rotation by unit quaternion q (matches 3Dmol's group matrix)
    const tx = 2 * (qy * p[2] - qz * p[1]), ty = 2 * (qz * p[0] - qx * p[2]), tz = 2 * (qx * p[1] - qy * p[0]);
    return [p[0] + qw * tx + (qy * tz - qz * ty), p[1] + qw * ty + (qz * tx - qx * tz), p[2] + qw * tz + (qx * ty - qy * tx)];
  };
  const ax = L.map((a, i) => ({ r: rot(unit(a)), n: "abc"[i], c: css(["--ax-a", "--ax-b", "--ax-c"][i]) })).sort((p, q) => p.r[2] - q.r[2]);
  svg.innerHTML = ax.map(({ r, n, c }) => {
    const x = 22 * r[0], y = -22 * r[1], o = 0.45 + 0.55 * (r[2] + 1) / 2;
    return `<g opacity="${o.toFixed(2)}"><line x1="0" y1="0" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${c}" stroke-width="2" stroke-linecap="round"/>` +
      `<text x="${(x * 1.32).toFixed(1)}" y="${(y * 1.32 + 4).toFixed(1)}" text-anchor="middle" fill="${c}">${n}</text></g>`;
  }).join("");
}

/* ------------------------------------------------------------------ fields and levels */
function selectField(i, keepP = true) {
  S.fi = i;
  const f = S.fields[i];
  if (!f) { S.st = null; return; }
  S.st = stats(f);
  S.mode = f.kind === "linear" ? "linear" : "log";
  // WHY: a total density (many electrons) at 50% enclosed hugs the cores; 90% gives the usual bonding-scale level
  if (!keepP) S.p = S.st.integral > 3 ? 0.9 : 0.5;
  if (S.mode === "log") S.iso = isoForFraction(S.st, S.p);
  else S.iso = /ELF/i.test(f.name) && S.st.max >= 0.75 && S.st.min <= 0.75 ? 0.75 : 0.5 * (S.st.min + S.st.max);
  syncLevel();
}
function logRange() { const hi = Math.log10(S.st.maxAbs) - 1e-3; return [hi - 6, hi]; }
function syncLevel() {
  const f = S.fields[S.fi], sl = $("iso");
  if (!f) return;
  if (S.mode === "log") { const [lo, hi] = logRange(); sl.value = Math.round(1000 * (Math.log10(Math.max(S.iso, 1e-300)) - lo) / (hi - lo)); }
  else sl.value = Math.round(1000 * (S.iso - S.st.min) / ((S.st.max - S.st.min) || 1));
  $("iso-num").value = fmt(S.iso);
  levelHint();
}
function levelHint() {
  const f = S.fields[S.fi];
  if (!f) return;
  const u = f.unit ? " " + f.unit : "";
  if (S.mode === "log") {
    const p = fractionAbove(S.st, S.iso), signed = f.kind === "signed";
    $("iso-hint").textContent = `±${sup(fmt(S.iso))}${u} · encloses ${Math.round(100 * p)}% of ${signed ? "∫|f|" : "the total"}`.replace(/^±/, signed ? "±" : "");
  } else $("iso-hint").textContent = `${sup(fmt(S.iso))}${u}`;
  $("row-pm").hidden = !(S.mode === "log" && f.kind === "signed" && S.st.min < -1e-6 * S.st.maxAbs);
  const I = S.st.integral, fi = Math.abs(I) >= 100 ? I.toFixed(2) : +I.toPrecision(5) + "";
  $("stats").textContent = `${f.n.join(" × ")} grid · min ${fmt(S.st.min)} · max ${fmt(S.st.max)} · ∫ ${fi}${f.unit === "e/Å³" ? " e" : ""}`;
}
function setLevelFromSlider(preview) {
  const u = +$("iso").value / 1000;
  if (S.mode === "log") { const [lo, hi] = logRange(); S.iso = Math.pow(10, lo + u * (hi - lo)); S.p = fractionAbove(S.st, S.iso); }
  else S.iso = S.st.min + u * (S.st.max - S.st.min);
  $("iso-num").value = fmt(S.iso);
  levelHint();
  surfaceSoon(preview);
}

/* ------------------------------------------------------------------ WAVECAR */
async function showBand(b) {
  const w = S.wave, { s, k, mode, half } = S.w;
  S.w.b = b;
  const key = [s, k, b, mode, half].join("|");
  let f = S.w.cache.get(key);
  if (!f) {
    busy(true); await tick();
    try {
      const r = await w.orbital(s, k, b, { signed: mode === "psi", half });
      f = { name: `band ${b + 1}${w.nspin === 2 ? (s ? " ↓" : " ↑") : ""}${mode === "psi" ? " ψ" : " |ψ|²"}`, unit: mode === "psi" ? "Å⁻³ᐟ²" : "Å⁻³",
        n: r.n, data: r.data, origin: [0, 0, 0], span: w.lattice, periodic: true, closed: false, kind: r.signed ? "signed" : "density" };
      S.w.cache.set(key, f);
      if (S.w.cache.size > 6) S.w.cache.delete(S.w.cache.keys().next().value);
    } catch (e) { toast(e.message, true); console.error(e); return; }
    finally { busy(false); }
  }
  S.fields = [f]; vols.clear();
  selectField(0, true);
  fillFieldSelect();
  markBand();
  drawSurface(); summaries();
}
function buildBands() {
  const w = S.wave, { s, k } = S.w, e = w.eig[s][k], o = w.occ[s][k], fr = w.frontier(s, k), list = $("bands");
  const rows = [];
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

/* ------------------------------------------------------------------ loading */
const sameCell = (A, B) => A && B && A.every((r, i) => r.every((x, j) => Math.abs(x - B[i][j]) < 2e-3));

async function openFiles(list) {
  const files = [...list];
  if (!files.length) return;
  busy(true); await tick();
  try {
    let structure = null, fields = [], wave = null, notes = [], title = "";
    for (const f of files) {
      const head = await f.slice(0, 65536).arrayBuffer();
      if (isWavecarHeader(head)) { wave = await Wavecar.open((o, l) => f.slice(o, o + l).arrayBuffer()); continue; }
      const text = await f.text();
      const d = parseText(detectKind(f.name, text.slice(0, 65536)), text, f.name);
      notes.push(...d.notes); title = title || d.title;
      if (d.fields.length && !fields.length) { fields = d.fields; if (!structure || !structure.species.length) structure = d.structure; }
      else if (!structure || (!structure.species.length && d.structure.species.length)) structure = d.structure;
    }
    if (!wave && !fields.length && structure && S.wave && S.structure && !S.structure.species.length) {
      if (sameCell(structure.lattice, S.wave.lattice)) { S.structure = structure; drawStructure(); viewer.render(); updateChrome(); toast("Atoms added to the WAVECAR cell."); return; }
    }
    if (wave && !(structure && sameCell(structure.lattice, wave.lattice))) {
      if (structure) notes.push("the structure's cell differs from the WAVECAR's, so only the cell is drawn");
      else notes.push("add the matching POSCAR or CONTCAR to draw the atoms");
      structure = { lattice: wave.lattice, pbc: true, species: [], pos: new Float64Array(0) };
    }
    load({ name: files.map(f => f.name).join(" + "), title, structure, fields, wave, notes });
  } catch (e) { toast(e.message || String(e), true, 8000); console.error(e); }
  finally { busy(false); }
}

function load(d) {
  if (!ensureViewer()) return;
  Object.assign(S, { name: d.name, structure: d.structure, fields: d.fields || [], wave: d.wave || null });
  S.w = { s: 0, k: 0, b: -1, mode: "density", half: "x", cache: new Map() };
  S.reps = [1, 1, 1]; ["ra", "rb", "rc"].forEach(id => ($(id).value = 1));
  vols.clear();
  $("empty").hidden = true;
  if (S.wave) {
    const w = S.wave;
    $("kpt").innerHTML = w.kpts.map((k, i) => `<option value="${i}">${i + 1}: ${k.map(x => x.toFixed(3)).join(" ")}</option>`).join("");
    buildBands();
  }
  S.fi = 0;
  if (S.fields.length) selectField(0, false);
  fillFieldSelect();
  updateChrome();
  drawStructure(); drawSurface();
  resetView();
  if (S.wave) { const fr = S.wave.frontier(0, 0); showBand(Math.max(0, fr.homo)); $("sec-orbital").open = true; }
  else if (S.fields.length) $("sec-surface").open = true;
  if (d.notes && d.notes.length) toast(d.notes.join("; "), false, 7000);
}

function fillFieldSelect() {
  $("field").innerHTML = S.fields.map((f, i) => `<option value="${i}">${f.name}</option>`).join("");
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
  summaries();
  segState();
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

/* ------------------------------------------------------------------ examples */
function hydrogen3d() {
  // psi_3d(x2-y2) = (x^2 - y^2) exp(-r / 3 a0) / (81 sqrt(2 pi) a0^(7/2)), normalised over all space (A^-3/2)
  const n = 72, L = 26, h = L / (n - 1), a0 = 0.529177210903, c = 1 / (81 * Math.sqrt(2 * Math.PI) * Math.pow(a0, 3.5));
  const data = new Float32Array(n * n * n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = -L / 2 + i * h, y = -L / 2 + j * h, z = -L / 2 + k * h, r = Math.hypot(x, y, z);
    data[i + n * (j + n * k)] = c * (x * x - y * y) * Math.exp(-r / (3 * a0));
  }
  const box = [[L, 0, 0], [0, L, 0], [0, 0, L]];
  return { name: "hydrogen 3d(x²−y²), computed here", structure: { lattice: box, pbc: false, species: ["H"], pos: Float64Array.from([0, 0, 0]) },
    fields: [{ name: "ψ 3d(x²−y²)", unit: "Å⁻³ᐟ²", n: [n, n, n], data, origin: [-L / 2, -L / 2, -L / 2], span: box, periodic: false, closed: false, kind: "signed" }], notes: [] };
}
async function example(id) {
  if (id === "h3d") return load(hydrogen3d());
  const url = { rutile: "examples/rutile-TiO2.cif", graphene: "examples/graphene.pw.in" }[id];
  const r = await fetch(url);
  if (!r.ok) return toast(`Could not load the example (${r.status}).`, true);
  openFiles([new File([await r.blob()], url.split("/").pop())]);
}

/* ------------------------------------------------------------------ wiring */
function bindSeg(key, fn) {
  const seg = document.querySelector(`.seg[data-key="${key}"]`);
  seg.addEventListener("click", e => { const b = e.target.closest("button"); if (b) { fn(b.value); segState(); summaries(); } });
}
bindSeg("style", v => { S.style = v; drawStructure(); viewer.render(); });
bindSeg("res", v => { S.res = v; drawSurface(); });
bindSeg("proj", v => { S.proj = v; viewer && (viewer.setProjection(v === "ortho" ? "orthographic" : "perspective"), viewer.render()); });
bindSeg("theme", v => {
  S.theme = v;
  if (v === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", v);
  try { localStorage.setItem("isoview-theme", v); } catch (e) { /* storage unavailable */ }
  restyle();
});
bindSeg("wspin", v => { S.w.s = +v; buildBands(); showBand(Math.max(0, S.w.b)); });
bindSeg("wmode", v => { S.w.mode = v; showBand(Math.max(0, S.w.b)); });
bindSeg("half", v => { S.w.half = v; S.w.cache.clear(); showBand(Math.max(0, S.w.b)); });
$("kpt").addEventListener("change", e => { S.w.k = +e.target.value; buildBands(); showBand(Math.max(0, S.w.b)); });
$("bands").addEventListener("click", e => { const r = e.target.closest(".band"); if (r) showBand(+r.dataset.b); });
$("bands").addEventListener("keydown", e => {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault();
  const b = Math.min(S.wave.nb - 1, Math.max(0, S.w.b + (e.key === "ArrowDown" ? 1 : -1)));
  showBand(b);
  const el = $("bands").querySelector(`[data-b="${b}"]`); if (el) el.scrollIntoView({ block: "nearest" });
});

$("scale").addEventListener("input", e => { S.scale = +e.target.value; drawStructure(); viewer.render(); });
$("bonds").addEventListener("change", e => { S.bonds = e.target.checked; $("row-tol").hidden = $("row-ionic").hidden = !S.bonds; drawStructure(); viewer.render(); });
$("tol").addEventListener("input", e => { S.tol = +e.target.value; $("tol-out").textContent = S.tol.toFixed(2); drawStructure(); viewer.render(); });
$("ionic").addEventListener("change", e => { S.ionic = e.target.checked; drawStructure(); viewer.render(); });
$("cell").addEventListener("change", e => { S.cell = e.target.checked; drawStructure(); viewer.render(); });
$("boundary").addEventListener("change", e => { S.boundary = e.target.checked; drawStructure(); viewer.render(); });
for (const [i, id] of ["ra", "rb", "rc"].entries()) $(id).addEventListener("change", e => {
  S.reps[i] = Math.min(6, Math.max(1, Math.round(+e.target.value) || 1)); e.target.value = S.reps[i];
  drawStructure(); drawSurface(); viewer.zoomTo(); viewer.render(); summaries();
});
$("field").addEventListener("change", e => { selectField(+e.target.value, true); drawSurface(); summaries(); });
$("surf").addEventListener("change", e => { S.surf = e.target.checked; drawSurface(); summaries(); });
$("pm").addEventListener("change", e => { S.pm = e.target.checked; drawSurface(); });
$("opacity").addEventListener("input", e => { S.opacity = +e.target.value; surfaceSoon(false); });
$("iso").addEventListener("input", () => setLevelFromSlider(true));
$("iso").addEventListener("change", () => setLevelFromSlider(false));
$("iso-num").addEventListener("change", e => {
  const v = Math.abs(parseFloat(e.target.value.replace("×10", "e")));
  if (!Number.isFinite(v)) return syncLevel();
  S.iso = S.mode === "log" ? v : parseFloat(e.target.value);
  if (S.mode === "log") S.p = fractionAbove(S.st, S.iso);
  syncLevel(); drawSurface(); summaries();
});

for (const b of document.querySelectorAll("[data-along]")) b.addEventListener("click", () => viewAlong(+b.dataset.along));
$("btn-reset").addEventListener("click", resetView);
$("btn-snap").addEventListener("click", snapshot);
$("btn-open").addEventListener("click", () => $("file").click());
$("btn-open2").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", e => { openFiles(e.target.files); e.target.value = ""; });
for (const b of document.querySelectorAll("[data-example]")) b.addEventListener("click", () => example(b.dataset.example));

function snapshot() {
  if (!viewer) return;
  const a = document.createElement("a");
  a.href = viewer.pngURI();
  a.download = (S.name.split(" + ")[0] || "isoview").replace(/[^\w.-]+/g, "_") + ".png";
  document.body.append(a); a.click(); a.remove();
}
function restyle() {
  if (!viewer) return;
  viewer.setBackgroundColor(css("--bg"));
  drawStructure(); drawSurface(); drawTriad();
}

let dragDepth = 0;
addEventListener("dragenter", e => { if ([...(e.dataTransfer?.types || [])].includes("Files")) { dragDepth++; $("drop").hidden = false; e.preventDefault(); } });
addEventListener("dragover", e => e.preventDefault());
addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").hidden = true; } });
addEventListener("drop", e => { e.preventDefault(); dragDepth = 0; $("drop").hidden = true; if (e.dataTransfer?.files?.length) openFiles(e.dataTransfer.files); });
addEventListener("paste", e => {
  if (e.target.closest && e.target.closest("input, select, textarea")) return;
  const t = e.clipboardData && e.clipboardData.getData("text");
  if (t && t.trim()) openFiles([new File([t], "pasted")]);
});
addEventListener("keydown", e => {
  if (e.target.closest && e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "o") $("file").click();
  else if (!S.structure) return;
  else if (k === "r") resetView();
  else if (k === "s") snapshot();
  else if ("abc".includes(k) && k.length === 1) viewAlong("abc".indexOf(k));
  else if (k === "escape") for (const d of document.querySelectorAll(".sec[open]")) d.open = false;
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (S.theme === "auto") restyle(); });

try { const t = localStorage.getItem("isoview-theme"); if (t === "light" || t === "dark") { S.theme = t; document.documentElement.setAttribute("data-theme", t); } } catch (e) { /* storage unavailable */ }
segState();

const qp = new URLSearchParams(location.search);
if (qp.get("url")) {
  const u = qp.get("url");
  fetch(u).then(r => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.blob(); })
    .then(b => openFiles([new File([b], decodeURIComponent(u.split(/[?#]/)[0].split("/").pop() || "file"))]))
    .catch(e => toast(`Could not fetch ${u}: ${e.message}. The server must allow cross-origin requests.`, true, 9000));
} else if (qp.get("example")) example(qp.get("example"));
