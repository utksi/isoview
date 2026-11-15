/**
 * Compute worker: file parsing, WAVECAR orbitals and marching cubes, off the main thread. The large arrays (field
 * values, the WAVECAR reader) stay here; the page gets metadata, statistics (stats() of field.js, for the level
 * slider) and finished meshes (transferred, not copied).
 * Requests { id, op, ... } -> replies { id, ok, result | error }:
 *   open { files, attach }               -> { structure, fields: [meta], wave: header | null, notes, title }; with attach
 *                                           and a structure-only drop while a WAVECAR is loaded: { structure, attach: true }
 *                                           (the WAVECAR and its orbital stay loaded)
 *   example { name: 'h3d' }              -> same shape (hydrogen 3d(x2 - y2) orbital computed here)
 *   orbital { s, k, b, signed, half }    -> { field: meta }  (replaces the current fields)
 *   mesh { fi, reps, stride, isos: [..] } -> { meshes: [{ positions, normals, indices, triangles, truncated }] }
 * meta = the field without `data`, plus `stats`.
 */
import { detectKind, parseText, isWavecarHeader } from "./formats.js";
import { Wavecar } from "./wavecar.js";
import { stats, renderGrid } from "./field.js";
import { marchingCubes } from "./gfx/mc.js";

const W = { fields: [], wave: null };

function meta(f) { const { data, ...m } = f; return { ...m, stats: stats(f) }; }
const transfers = m => [m.stats.sorted.buffer, m.stats.cum.buffer];

/** hydrogen 3d(x2-y2): psi = (x^2 - y^2) exp(-r / 3 a0) / (81 sqrt(2 pi) a0^(7/2)), normalised (A^-3/2). */
function hydrogen3d() {
  const n = 72, L = 26, h = L / (n - 1), a0 = 0.529177210903, c = 1 / (81 * Math.sqrt(2 * Math.PI) * Math.pow(a0, 3.5));
  const data = new Float32Array(n * n * n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = -L / 2 + i * h, y = -L / 2 + j * h, z = -L / 2 + k * h;
    data[i + n * (j + n * k)] = c * (x * x - y * y) * Math.exp(-Math.hypot(x, y, z) / (3 * a0));
  }
  const box = [[L, 0, 0], [0, L, 0], [0, 0, L]];
  return { title: "hydrogen 3d(x²−y²)", structure: { lattice: box, pbc: false, species: ["H"], pos: Float64Array.from([0, 0, 0]) },
    fields: [{ name: "ψ 3d(x²−y²)", unit: "Å⁻³ᐟ²", n: [n, n, n], data, origin: [-L / 2, -L / 2, -L / 2], span: box, periodic: false, closed: false, kind: "signed" }], notes: [] };
}

async function open(files) {
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
  return { structure, fields, wave, notes, title };
}

function waveHeader(w) {
  return { nspin: w.nspin, nk: w.nk, nb: w.nb, encut: w.encut, lattice: w.lattice, kpts: w.kpts, eig: w.eig, occ: w.occ, type: w.type,
    frontier: [...Array(w.nspin).keys()].map(s => [...Array(w.nk).keys()].map(k => w.frontier(s, k))) };
}

export async function handle(msg) {
  const { op } = msg;
  if (op === "open" || op === "example") {
    const d = op === "open" ? await open(msg.files) : hydrogen3d();
    // keep the previous WAVECAR when only a structure was added to it
    if (op === "open" && msg.attach && !d.wave && !d.fields.length && W.wave) return { result: { ...d, fields: [], wave: null, attach: true }, transfer: [] };
    W.fields = d.fields; W.wave = d.wave || null;
    const metas = d.fields.map(meta);
    return { result: { structure: d.structure, fields: metas, wave: d.wave ? waveHeader(d.wave) : null, notes: d.notes, title: d.title },
      transfer: metas.flatMap(transfers) };
  }
  if (op === "orbital") {
    const w = W.wave, r = await w.orbital(msg.s, msg.k, msg.b, { signed: msg.signed, half: msg.half });
    const f = { name: `band ${msg.b + 1}${w.nspin === 2 ? (msg.s ? " ↓" : " ↑") : ""}${r.signed ? " ψ" : " |ψ|²"}`, unit: r.signed ? "Å⁻³ᐟ²" : "Å⁻³",
      n: r.n, data: r.data, origin: [0, 0, 0], span: w.lattice, periodic: true, closed: false, kind: r.signed ? "signed" : "density" };
    W.fields = [f];
    const m = meta(f);
    return { result: { field: m }, transfer: transfers(m) };
  }
  if (op === "mesh") {
    const f = W.fields[msg.fi];
    if (!f) throw new Error("no field " + msg.fi);
    const g = renderGrid(f, msg.reps, msg.stride);
    const meshes = msg.isos.map(iso => marchingCubes(g, iso, msg.maxTris || 6e6));
    return { result: { meshes }, transfer: meshes.flatMap(m => [m.positions.buffer, m.normals.buffer, m.indices.buffer]) };
  }
  throw new Error("unknown op " + op);
}

if (typeof self !== "undefined" && typeof window === "undefined" && typeof self.postMessage === "function") {
  self.onmessage = async e => {
    const { id } = e.data;
    try { const { result, transfer } = await handle(e.data); self.postMessage({ id, ok: true, result }, transfer); }
    catch (err) { self.postMessage({ id, ok: false, error: err && err.message ? err.message : String(err) }); }
  };
}
