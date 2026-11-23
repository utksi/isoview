/**
 * Compute worker: file parsing, WAVECAR orbitals and marching cubes, off the main thread. The large arrays (field
 * values, the WAVECAR reader) stay here; the page gets metadata, statistics (stats() of field.js, for the level
 * slider) and finished meshes (transferred, not copied).
 * Several datasets (one per loaded file group) are kept at once, keyed by the id that open returns, so layers from
 * different files can be shown together. Requests { id, op, ... } -> replies { id, ok, result | error }:
 *   open { files }                        -> { ds, structure, fields: [meta], wave: header | null, traj | null, notes, title }
 *                                            (traj: the frames of a trajectory file, traj.js; its arrays are transferred)
 *   example { name: 'h3d' }               -> same shape (hydrogen 3d(x2 - y2) orbital computed here)
 *   orbital { ds, slot, s, k, b, signed, half } -> { field: meta }  (becomes field `slot` of dataset ds)
 *   mesh { ds, fi, reps, stride, isos: [..], shift, cuts }
 *                                         -> { meshes: [{ positions, normals, indices, triangles, truncated }] }
 *   drop { ds }                           -> {}  (frees the dataset)
 *   symmetry { lattice, frac, numbers, symprec }
 *                                         -> { number, hm, hall } (space group by moyo, vendor/moyo; symprec in A)
 * meta = the field without `data`, plus `stats`.
 * Fields are refined on opening (field.js refine: grid step <= 0.1 A, at most 4 M points) and WAVECAR orbitals are
 * built on an FFT box of the same size limits (wavecar.js), so isosurfaces of coarse data are smooth.
 * Gzip-compressed files (magic bytes 1f 8b, e.g. vasprun.xml.gz, CHGCAR.gz) are unpacked first with the browser's
 * DecompressionStream; the name loses its ".gz".
 */
import { detectKind, parseText, isWavecarHeader } from "./formats.js?v=26";
import { isUlm, parseAseTraj } from "./traj.js?v=26";
import { Wavecar } from "./wavecar.js?v=26";
import { stats, renderGrid, refine } from "./field.js?v=26";
import { marchingCubes } from "./gfx/mc.js?v=26";

const DS = new Map();
let nextDs = 1;

function meta(f) { const { data, ...m } = f; return { ...m, stats: stats(f) }; }
const transfers = m => [m.stats.sorted.buffer, m.stats.cum.buffer];

/** hydrogen 3d(x2-y2): psi = (x^2 - y^2) exp(-r / 3 a0) / (81 sqrt(2 pi) a0^(7/2)), normalised (A^-3/2). */
function hydrogen3d() {
  const n = 128, L = 26, h = L / (n - 1), a0 = 0.529177210903, c = 1 / (81 * Math.sqrt(2 * Math.PI) * Math.pow(a0, 3.5));
  const data = new Float32Array(n * n * n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = -L / 2 + i * h, y = -L / 2 + j * h, z = -L / 2 + k * h;
    data[i + n * (j + n * k)] = c * (x * x - y * y) * Math.exp(-Math.hypot(x, y, z) / (3 * a0));
  }
  const box = [[L, 0, 0], [0, L, 0], [0, 0, L]];
  return { title: "hydrogen 3d(x²−y²)", structure: { lattice: box, pbc: false, species: ["H"], pos: Float64Array.from([0, 0, 0]), origin: [-L / 2, -L / 2, -L / 2] },
    fields: [{ name: "ψ 3d(x²−y²)", unit: "Å⁻³ᐟ²", n: [n, n, n], data, origin: [-L / 2, -L / 2, -L / 2], span: box, periodic: false, closed: false, kind: "signed" }], notes: [] };
}

/** The file itself, or its gzip-decompressed content (as a File without the .gz suffix). */
async function gunzip(f) {
  const m = new Uint8Array(await f.slice(0, 2).arrayBuffer());
  if (m[0] !== 0x1f || m[1] !== 0x8b) return f;
  if (typeof DecompressionStream === "undefined") throw new Error(`${f.name} is gzip-compressed and this browser cannot unpack it; unpack it first`);
  const blob = await new Response(f.stream().pipeThrough(new DecompressionStream("gzip"))).blob();
  return new File([blob], f.name.replace(/\.gz$/i, ""));
}

async function open(list) {
  let structure = null, fields = [], wave = null, traj = null, notes = [], title = "";
  for (const f0 of list) {
    const f = await gunzip(f0);
    const head = await f.slice(0, 65536).arrayBuffer();
    if (isWavecarHeader(head)) { wave = await Wavecar.open((o, l) => f.slice(o, o + l).arrayBuffer()); continue; }
    let d;
    if (isUlm(head)) d = parseAseTraj(await f.arrayBuffer());
    else { const text = await f.text(); d = parseText(detectKind(f.name, text.slice(0, 65536)), text, f.name); }
    notes.push(...d.notes); title = title || d.title;
    if (d.traj && !traj && !fields.length) { traj = d.traj; structure = d.structure; continue; }
    if (d.fields.length && !fields.length) { fields = d.fields; if (!structure || !structure.species.length) structure = d.structure; }
    else if (!structure || (!structure.species.length && d.structure.species.length)) structure = d.structure;
  }
  // WHY: coarse grids give faceted surfaces; a finer copy (Fourier for periodic grids, cubic otherwise) is smooth
  return { structure, fields: fields.map(f => refine(f)), wave, traj, notes, title };
}

function waveHeader(w) {
  return { nspin: w.nspin, nk: w.nk, nb: w.nb, encut: w.encut, lattice: w.lattice, kpts: w.kpts, eig: w.eig, occ: w.occ, type: w.type,
    frontier: [...Array(w.nspin).keys()].map(s => [...Array(w.nk).keys()].map(k => w.frontier(s, k))) };
}

export async function handle(msg) {
  const { op } = msg;
  if (op === "open" || op === "example") {
    const d = op === "open" ? await open(msg.files) : hydrogen3d();
    const ds = nextDs++;
    DS.set(ds, { fields: d.fields, wave: d.wave || null });
    const metas = d.fields.map(meta);
    const tr = d.traj || null;
    return { result: { ds, structure: d.structure, fields: metas, wave: d.wave ? waveHeader(d.wave) : null, traj: tr, notes: d.notes, title: d.title },
      transfer: [...metas.flatMap(transfers), ...(tr ? [tr.pos.buffer, tr.cells && tr.cells.buffer, tr.energy && tr.energy.buffer].filter(Boolean) : [])] };
  }
  if (op === "drop") { DS.delete(msg.ds); return { result: {}, transfer: [] }; }
  if (op === "symmetry") {
    const m = await moyoReady();
    const d = m.analyze_cell(JSON.stringify({ lattice: { basis: msg.lattice.flat() }, positions: msg.frac, numbers: msg.numbers }), msg.symprec, "Standard");
    return { result: { number: d.number, hm: d.hm_symbol, hall: d.hall_number }, transfer: [] };
  }
  const W = DS.get(msg.ds);
  if (!W) throw new Error("dataset " + msg.ds + " is not loaded");
  if (op === "orbital") {
    const w = W.wave, r = await w.orbital(msg.s, msg.k, msg.b, { signed: msg.signed, half: msg.half });
    const f = { name: `band ${msg.b + 1}${w.nspin === 2 ? (msg.s ? " ↓" : " ↑") : ""}${r.signed ? " ψ" : " |ψ|²"}`, unit: r.signed ? "Å⁻³ᐟ²" : "Å⁻³",
      n: r.n, n0: r.n0, data: r.data, origin: [0, 0, 0], span: w.lattice, periodic: true, closed: false, kind: r.signed ? "signed" : "density" };
    // WHY: one slot per orbital layer (msg.slot, after the file's own fields), so several bands can be shown at once
    W.fields[msg.slot ?? 0] = f;
    const m = meta(f);
    return { result: { field: m }, transfer: transfers(m) };
  }
  if (op === "mesh") {
    const f = W.fields[msg.fi];
    if (!f) throw new Error("no field " + msg.fi);
    const g = renderGrid(f, msg.reps, msg.stride, msg.shift || [0, 0, 0], msg.cuts || []);
    const meshes = msg.isos.map(iso => marchingCubes(g, iso, msg.maxTris || 6e6));
    return { result: { meshes }, transfer: meshes.flatMap(m => [m.positions.buffer, m.normals.buffer, m.indices.buffer]) };
  }
  throw new Error("unknown op " + op);
}

/** moyo (WebAssembly), loaded on first use; in Node (tests) the .wasm file is read from disk. */
let moyo = null;
function moyoReady() {
  if (!moyo) moyo = import("./vendor/moyo/moyo_wasm.js?v=26").then(async m => {
    const url = new URL("./vendor/moyo/moyo_wasm_bg.wasm?v=26", import.meta.url);
    const src = url.protocol === "file:" ? (await import("node:fs")).readFileSync(new URL(url.pathname, "file://")) : url;
    await m.default({ module_or_path: src });
    return m;
  });
  return moyo;
}

if (typeof self !== "undefined" && typeof window === "undefined" && typeof self.postMessage === "function") {
  self.onmessage = async e => {
    const { id } = e.data;
    try { const { result, transfer } = await handle(e.data); self.postMessage({ id, ok: true, result }, transfer); }
    catch (err) { self.postMessage({ id, ok: false, error: err && err.message ? err.message : String(err) }); }
  };
}
