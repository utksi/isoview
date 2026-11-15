/**
 * Trajectory readers: files holding a sequence of structures of the same atoms (molecular dynamics, relaxations,
 * NEB images, animations). Every reader returns a dataset shaped like those of formats.js,
 *   { title, structure, fields: [], notes, traj }
 * with structure = frame 0 and, when there are two or more frames,
 *   traj = { n, nat, pos: Float64Array(3 nat n), cells: Float64Array(9 n) | null, energy: Float64Array(n) | null }
 * pos: Cartesian coordinates (Angstrom), atom i of frame f at index 3 (nat f + i). cells: lattice rows a, b, c of
 * every frame (Angstrom, 9 numbers per frame) when the cell changes (NPT, variable-cell relaxation), else null and
 * structure.lattice holds for all frames. energy: eV per frame as the code prints it (VASP free energy TOTEN,
 * QE total energy, ASE calculator energy, extended-XYZ energy=), NaN where a frame has none; null if no frame has one.
 * Atoms keep one order in every frame (LAMMPS dumps are sorted by atom id). A frame with a different number of atoms
 * ends the trajectory (said in notes).
 * Units in: VASP, extended XYZ, XSF, PDB, LAMMPS (metal/real units) and ASE in Angstrom and eV; QE pw.x output in Bohr
 * or alat and Ry; CASTEP .md/.geom in Bohr and Hartree; GROMACS .gro in nm. Conversions: BOHR = 0.529177210903 A,
 * Ry = 13.605693122994 eV, Ha = 2 Ry, nm = 10 A.
 */
import { Cursor, tokens } from "./numbers.js?v=15";
import { elementOf, SYMBOLS } from "./elements.js?v=15";
import { BOHR, det3, fracToCart, fromParams } from "./lattice.js?v=15";

const RY = 13.605693122994; // eV
const HA = 2 * RY; // eV

/* ------------------------------------------------------------------ frames to a dataset */

/**
 * Collect frames into a dataset. species: [nat] symbols; frames: [Float64Array(3 nat)] Cartesian A; cells: [rows or
 * null] per frame; energies: [eV or NaN] per frame. The cell counts as changing if any entry differs from frame 0 by
 * more than 1e-6 A.
 */
export function finish(species, frames, cells, energies, { title = "", pbc = true, notes = [] } = {}) {
  const n = frames.length, nat = species.length;
  if (!n) throw new Error("no structures found");
  const c0 = cells[0] || cells.find(Boolean) || null;
  const varies = !!c0 && cells.some(c => c && c.some((r, i) => r.some((x, j) => Math.abs(x - c0[i][j]) > 1e-6)));
  const structure = { lattice: c0 ? c0.map(r => r.slice()) : null, pbc: pbc && !!c0, species: species.slice(), pos: Float64Array.from(frames[0]) };
  if (n === 1) return { title, structure, fields: [], notes, traj: null };
  const pos = new Float64Array(3 * nat * n);
  frames.forEach((p, f) => pos.set(p, 3 * nat * f));
  let C = null;
  if (varies) { C = new Float64Array(9 * n); cells.forEach((c, f) => C.set((c || c0).flat(), 9 * f)); }
  const E = energies.some(Number.isFinite) ? Float64Array.from({ length: n }, (_, f) => (Number.isFinite(energies[f]) ? energies[f] : NaN)) : null;
  return { title, structure, fields: [], notes, traj: { n, nat, pos, cells: C, energy: E } };
}

/** Cartesian coordinates (Float64Array 3N) from fractional ones f (3N) and lattice rows L. */
function fracArray(L, f) {
  const out = new Float64Array(f.length);
  for (let i = 0; i < f.length; i += 3) {
    const a = f[i], b = f[i + 1], c = f[i + 2];
    for (let k = 0; k < 3; k++) out[i + k] = a * L[0][k] + b * L[1][k] + c * L[2][k];
  }
  return out;
}

/** True if the frame has the expected atom count; otherwise adds a note once and returns false (ends the reader). */
function sameCount(n, nat, notes, what) {
  if (n === nat) return true;
  notes.push(`${what} has ${n} atoms instead of ${nat}; the trajectory stops before it`);
  return false;
}

/* ------------------------------------------------------------------ VASP XDATCAR */

/** One XDATCAR header (VASP 5): title, scale, lattice rows, symbols, counts. Scale < 0 is a target volume (A^3). */
function xdatHeader(cur) {
  const title = (cur.nonEmpty() ?? "").trim();
  const sc = Number(tokens(cur.line() ?? "")[0]);
  const A = [0, 1, 2].map(() => tokens(cur.line() ?? "").slice(0, 3).map(Number));
  if (!Number.isFinite(sc) || A.some(r => r.length < 3 || r.some(x => !Number.isFinite(x)))) throw new Error("XDATCAR: bad header");
  const t = tokens(cur.line() ?? "");
  let symbols, counts;
  if (/^[A-Za-z]/.test(t[0] || "")) { symbols = t.map(elementOf); counts = tokens(cur.line() ?? "").map(Number); }
  else { counts = t.map(Number); const g = title.split(/\s+/).map(elementOf); symbols = counts.map((_, i) => g[i] || "X"); }
  if (!counts.length || counts.some(x => !Number.isInteger(x))) throw new Error("XDATCAR: bad atom counts (VASP 4 XDATCAR files are not supported)");
  const s = sc < 0 ? Math.cbrt(-sc / Math.abs(det3(A))) : sc;
  const species = [];
  counts.forEach((c, i) => { for (let k = 0; k < c; k++) species.push(symbols[i] || "X"); });
  return { title, L: A.map(r => r.map(x => x * s)), species };
}

/**
 * XDATCAR: a header, then blocks "Direct configuration= k" of nat fractional positions. Variable-cell runs repeat
 * the header before every block. Position lines may carry more columns (pymatgen appends the element); the first
 * three numbers are used.
 */
export function parseXDATCAR(text) {
  const cur = new Cursor(text), frames = [], cells = [], notes = [];
  let h = null;
  for (;;) {
    const save = cur.pos, l = cur.nonEmpty();
    if (l === null) break;
    if (/configuration/i.test(l)) {
      if (!h) throw new Error("XDATCAR: a configuration before the header");
      const nat = h.species.length, at = cur.pos;
      let v;
      try { v = cur.numbers(3 * nat, new Float64Array(3 * nat)); } catch (e) { v = null; }
      if (!v || Array.prototype.some.call(v, x => Number.isNaN(x))) {
        // lines with more than three columns (pymatgen appends the element): take the first three numbers per line
        cur.pos = at; v = new Float64Array(3 * nat);
        let okC = true;
        for (let i = 0; i < nat && okC; i++) {
          const t = tokens(cur.line() ?? "");
          if (t.length < 3 || t.slice(0, 3).some(x => !Number.isFinite(+x))) okC = false;
          else { v[3 * i] = +t[0]; v[3 * i + 1] = +t[1]; v[3 * i + 2] = +t[2]; }
        }
        if (!okC) { notes.push(frames.length ? "the last configuration is incomplete and was skipped" : "XDATCAR: unreadable positions"); break; }
      }
      frames.push(/^\s*c/i.test(l) ? v : fracArray(h.L, v)); cells.push(h.L);
      continue;
    }
    cur.pos = save;
    const nh = xdatHeader(cur);
    if (h && !sameCount(nh.species.length, h.species.length, notes, `configuration ${frames.length + 1}`)) break;
    h = nh;
  }
  if (!h) throw new Error("XDATCAR: no header");
  return finish(h.species, frames, cells, [], { title: h.title, notes });
}

/* ------------------------------------------------------------------ VASP OUTCAR */

/**
 * OUTCAR: species from the TITEL lines (one per POTCAR) and "ions per type"; for every ionic step the Cartesian
 * positions of the "POSITION ... TOTAL-FORCE" block, the last "direct lattice vectors" before it, and the energy of
 * the "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" block after it (free energy TOTEN, eV).
 */
export function parseOUTCAR(text) {
  const notes = [];
  const ipt = text.match(/ions per type\s*=\s*([\d \t]+)/);
  if (!ipt) throw new Error("OUTCAR: no 'ions per type' line");
  const counts = ipt[1].trim().split(/\s+/).map(Number);
  let names = [...text.matchAll(/^\s*TITEL\s*=\s*\S+\s+(\S+)/gm)].map(m => elementOf(m[1]));
  if (names.length === 2 * counts.length) names = names.slice(0, counts.length);
  if (names.length !== counts.length) names = [...text.matchAll(/^\s*VRHFIN\s*=\s*([A-Za-z]+)/gm)].map(m => elementOf(m[1])).slice(0, counts.length);
  const species = [];
  counts.forEach((c, i) => { for (let k = 0; k < c; k++) species.push(names[i] || "X"); });
  const nat = species.length, frames = [], cells = [], energies = [];
  const cur = new Cursor(text), re = /direct lattice vectors|POSITION\s+TOTAL-FORCE|FREE ENERGIE OF THE ION-ELECTRON SYSTEM/g;
  let L = null, m;
  while ((m = re.exec(text))) {
    cur.pos = text.indexOf("\n", m.index) + 1;
    if (m[0][0] === "d") {
      const rows = [0, 1, 2].map(() => tokens(cur.line() ?? "").slice(0, 3).map(Number));
      if (rows.every(r => r.length === 3 && r.every(Number.isFinite))) L = rows;
    } else if (m[0][0] === "P") {
      cur.line();
      const v = new Float64Array(3 * nat);
      let okF = true;
      for (let i = 0; i < nat; i++) {
        const t = tokens(cur.line() ?? "");
        if (t.length < 3) { okF = false; break; }
        v[3 * i] = +t[0]; v[3 * i + 1] = +t[1]; v[3 * i + 2] = +t[2];
      }
      if (!okF) { notes.push("the last ionic step is incomplete and was skipped"); break; }
      frames.push(v); cells.push(L); energies.push(NaN);
    } else if (frames.length) {
      const e = text.slice(m.index, m.index + 400).match(/free\s+energy\s+TOTEN\s*=\s*([-+\d.Ee]+)/);
      if (e) energies[frames.length - 1] = +e[1];
    }
    re.lastIndex = cur.pos;
  }
  return finish(species, frames, cells, energies, { title: "OUTCAR", notes });
}

/* ------------------------------------------------------------------ VASP vasprun.xml */

/** Rows of numbers of <varray name="name"> starting at or after index `from` of s, or null. */
function varray(s, name, from = 0) {
  const a = s.indexOf(`<varray name="${name}"`, from);
  if (a < 0) return null;
  const b = s.indexOf("</varray>", a);
  if (b < 0) return null;
  return [...s.slice(a, b).matchAll(/<v>([^<]*)<\/v>/g)].map(r => r[1].trim().split(/\s+/).map(Number));
}

/**
 * vasprun.xml: species from <atominfo> <array name="atoms">; one frame per <calculation> from its <structure> (basis
 * rows in A, fractional positions) and its last e_fr_energy (eV). Without calculations, the initial structure.
 */
export function parseVasprun(text) {
  const ai = text.indexOf('<array name="atoms"');
  if (ai < 0) throw new Error("vasprun.xml: no atom list");
  const species = [...text.slice(ai, text.indexOf("</array>", ai)).matchAll(/<rc>\s*<c>\s*([A-Za-z]+)\s*<\/c>/g)].map(r => elementOf(r[1]));
  const nat = species.length, frames = [], cells = [], energies = [], notes = [];
  const add = (blk, at) => {
    const B = varray(blk, "basis", at), P = varray(blk, "positions", at);
    if (!B || !P || B.length < 3 || P.length !== nat) return false;
    frames.push(fracArray(B, Float64Array.from(P.flat()))); cells.push(B.slice(0, 3));
    return true;
  };
  for (let p = 0; ;) {
    const c0 = text.indexOf("<calculation>", p);
    if (c0 < 0) break;
    let c1 = text.indexOf("</calculation>", c0);
    if (c1 < 0) { c1 = text.length; notes.push("the file ends inside a calculation (a run still going or cut off)"); }
    const blk = text.slice(c0, c1), s0 = blk.indexOf("<structure");
    if (s0 >= 0 && add(blk, s0)) {
      const es = [...blk.matchAll(/<i name="e_fr_energy">\s*([^<]+)<\/i>/g)];
      energies.push(es.length ? +es[es.length - 1][1] : NaN);
    }
    p = c1 + 1;
  }
  if (!frames.length) {
    const s0 = text.indexOf('<structure name="initialpos"');
    if (s0 < 0 || !add(text, s0)) throw new Error("vasprun.xml: no structures");
    energies.push(NaN);
  }
  return finish(species, frames, cells, energies, { title: "vasprun.xml", notes });
}

/* ------------------------------------------------------------------ extended XYZ (one or many frames) */

/**
 * XYZ and extended XYZ: blocks of "N", a comment line, N atom lines. Extended XYZ comments give Lattice="9 numbers"
 * (rows a, b, c), Properties=name:type:ncols:... (columns of species and pos) and energy=... (eV).
 */
export function parseXYZFrames(text) {
  const cur = new Cursor(text), frames = [], cells = [], energies = [], notes = [];
  let species = null, pbc = false;
  for (;;) {
    const l = cur.nonEmpty();
    if (l === null) break;
    const n = parseInt(l);
    if (!Number.isInteger(n) || n < 0) break;
    const comment = cur.line() ?? "";
    let si = 0, pi = 1;
    const pr = comment.match(/Properties\s*=\s*(\S+)/i);
    if (pr) {
      const p = pr[1].split(":");
      for (let k = 0, c = 0; k + 2 < p.length; c += +p[k + 2], k += 3) {
        if (/^species$/i.test(p[k])) si = c;
        if (/^pos(itions)?$/i.test(p[k])) pi = c;
      }
    }
    const sp = [], v = new Float64Array(3 * n);
    let okF = true;
    for (let i = 0; i < n; i++) {
      const r = tokens(cur.line() ?? "");
      if (r.length < pi + 3) { okF = false; break; }
      sp.push(elementOf(r[si])); v[3 * i] = +r[pi]; v[3 * i + 1] = +r[pi + 1]; v[3 * i + 2] = +r[pi + 2];
    }
    if (!okF) { if (frames.length) notes.push("the last frame is incomplete and was skipped"); else throw new Error("xyz: incomplete atom lines"); break; }
    if (species && (!sameCount(n, species.length, notes, `frame ${frames.length + 1}`) || sp.some((s, i) => s !== species[i]))) {
      if (n === species.length) notes.push(`frame ${frames.length + 1} lists the atoms in another order; the trajectory stops before it`);
      break;
    }
    species = species || sp;
    const lat = comment.match(/Lattice\s*=\s*"([^"]+)"/i);
    let L = null;
    if (lat) { const q = lat[1].trim().split(/\s+/).map(Number); if (q.length >= 9) L = [q.slice(0, 3), q.slice(3, 6), q.slice(6, 9)]; }
    const pb = comment.match(/pbc\s*=\s*"([^"]+)"/i);
    pbc = L ? !(pb && /^\s*(F|false|0)(\s+(F|false|0)){2}\s*$/i.test(pb[1])) : false;
    const e = comment.match(/(?:^|\s)energy\s*=\s*"?([-+]?[\d.]+(?:[eEdD][-+]?\d+)?)/i);
    frames.push(v); cells.push(L); energies.push(e ? +e[1].replace(/[dD]/, "e") : NaN);
  }
  if (!frames.length) throw new Error("xyz: no frames");
  return finish(species, frames, cells, energies, { pbc, notes });
}

/* ------------------------------------------------------------------ LAMMPS text dump */

/**
 * LAMMPS dump (ITEM: TIMESTEP / NUMBER OF ATOMS / BOX BOUNDS / ATOMS ...). Box: bounds lines "lo hi [tilt]"; for a
 * triclinic box xlo = xlo_bound - min(0, xy, xz, xy + xz), xhi = xhi_bound - max(0, xy, xz, xy + xz),
 * ylo = ylo_bound - min(0, yz), yhi = yhi_bound - max(0, yz); a = (xhi - xlo, 0, 0), b = (xy, yhi - ylo, 0),
 * c = (xz, yz, zhi - zlo), origin (xlo, ylo, zlo) moved to 0. Coordinates: x y z, else xu yu zu, else scaled xs ys zs
 * (r = origin + s L), else xsu ysu zsu. Species: the element column; else the element nearest to the mass column;
 * else the type number read as an atomic number (as ASE does). Atoms are sorted by id.
 */
export function parseLAMMPS(text) {
  const cur = new Cursor(text), frames = [], cells = [], notes = [];
  let species = null, typeNote = false;
  for (;;) {
    let l = cur.nonEmpty();
    if (l === null) break;
    if (!/^ITEM:\s*TIMESTEP/.test(l)) continue;
    cur.line();
    l = cur.nonEmpty();
    if (!/^ITEM:\s*NUMBER OF ATOMS/.test(l || "")) break;
    const n = parseInt(cur.line());
    l = cur.nonEmpty() || "";
    if (!/^ITEM:\s*BOX BOUNDS/.test(l)) break;
    const tri = /xy\s+xz\s+yz/.test(l);
    const B = [0, 1, 2].map(() => tokens(cur.line() ?? "").map(Number));
    let [xlo, xhi, xy = 0] = B[0], [ylo, yhi, xz = 0] = B[1], [zlo, zhi, yz = 0] = B[2];
    if (tri) {
      xlo -= Math.min(0, xy, xz, xy + xz); xhi -= Math.max(0, xy, xz, xy + xz);
      ylo -= Math.min(0, yz); yhi -= Math.max(0, yz);
    } else { xy = xz = yz = 0; }
    const L = [[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]], o = [xlo, ylo, zlo];
    const cols = tokens((cur.nonEmpty() || "").replace(/^ITEM:\s*ATOMS/, ""));
    const ci = k => cols.indexOf(k);
    const xyz = [["x", "y", "z"], ["xu", "yu", "zu"], ["xs", "ys", "zs"], ["xsu", "ysu", "zsu"]].find(s => s.every(k => ci(k) >= 0));
    if (!xyz) throw new Error("LAMMPS dump: no coordinate columns (x y z, xu yu zu or xs ys zs)");
    const scaled = xyz[0].startsWith("xs"), ix = xyz.map(ci), iid = ci("id"), itype = ci("type"), iel = ci("element"), imass = ci("mass");
    const rows = [];
    for (let i = 0; i < n; i++) rows.push(tokens(cur.line() ?? ""));
    if (iid >= 0) rows.sort((a, b) => +a[iid] - +b[iid]);
    const sp = rows.map(r => {
      if (iel >= 0) return elementOf(r[iel]);
      if (imass >= 0) return nearestByMass(+r[imass]);
      typeNote = true;
      return SYMBOLS[+r[itype]] || "X";
    });
    if (species && !sameCount(n, species.length, notes, `timestep ${frames.length + 1}`)) break;
    species = species || sp;
    const v = new Float64Array(3 * n);
    rows.forEach((r, i) => {
      const q = ix.map(k => +r[k]);
      const p = scaled ? fracToCart(L, q).map((x, k) => x + o[k]) : q;
      for (let k = 0; k < 3; k++) v[3 * i + k] = p[k] - o[k];
    });
    frames.push(v); cells.push(L);
  }
  if (typeNote) notes.push("atom types are drawn as the elements with that atomic number (1 = H, 2 = He, ...); add an element or mass column to the dump to get the real elements");
  if (!frames.length) throw new Error("LAMMPS dump: no frames");
  return finish(species, frames, cells, [], { title: "LAMMPS dump", notes });
}

/** Standard atomic masses (u) for Z = 1..86, for LAMMPS dumps that give masses instead of elements. */
const MASS = [0, 1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974,
  32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723,
  72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 97, 101.07, 102.91, 106.42, 107.87,
  112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145, 150.36,
  151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22,
  195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 209, 210, 222];
function nearestByMass(m) {
  let best = 0, d = Infinity;
  for (let z = 1; z < MASS.length; z++) { const e = Math.abs(MASS[z] - m); if (e < d) { d = e; best = z; } }
  return SYMBOLS[best] || "X";
}

/* ------------------------------------------------------------------ Quantum ESPRESSO pw.x output */

/**
 * pw.x output (scf, relax, md, vc-relax). Frame 0: "crystal axes" (alat units) and "site n. atom positions (alat
 * units)"; then every ATOMIC_POSITIONS block (crystal, alat, bohr or angstrom), with the latest CELL_PARAMETERS
 * (alat=, bohr or angstrom). The repeated block between "Begin final coordinates" and "End final coordinates" is
 * skipped when it equals the last frame. Energies: the "!    total energy" lines, in order, Ry -> eV.
 */
export function parseQEout(text) {
  const notes = [];
  const am = text.match(/lattice parameter \(alat\)\s*=\s*([-\d.Ee+]+)/);
  if (!am) throw new Error("pw.x output: no lattice parameter");
  const alat = +am[1] * BOHR;
  const ax = text.match(/crystal axes: \(cart\. coord\. in units of alat\)\s*\n\s*a\(1\) = \(([^)]+)\)\s*\n\s*a\(2\) = \(([^)]+)\)\s*\n\s*a\(3\) = \(([^)]+)\)/);
  if (!ax) throw new Error("pw.x output: no crystal axes");
  let L = [ax[1], ax[2], ax[3]].map(s => s.trim().split(/\s+/).map(x => +x * alat));
  const site = text.indexOf("positions (alat units)");
  if (site < 0) throw new Error("pw.x output: no atomic positions");
  const cur = new Cursor(text, text.indexOf("\n", site) + 1), species = [], p0 = [];
  for (;;) {
    const l = cur.line();
    const m = l && l.match(/^\s*\d+\s+(\S+)\s+tau\(\s*\d+\)\s*=\s*\(([^)]+)\)/);
    if (!m) break;
    species.push(elementOf(m[1])); p0.push(...m[2].trim().split(/\s+/).map(x => +x * alat));
  }
  const nat = species.length, frames = [Float64Array.from(p0)], cells = [L];
  // WHY: the card name alone on its line, optionally with (unit) or {unit}: QE also prints messages that start with
  // the card name ("ATOMIC_POSITIONS: units set to alat"), which are not blocks
  const re = /^[ \t]*(CELL_PARAMETERS|ATOMIC_POSITIONS)[ \t]*(?:[({][ \t]*([^)}\n]*)[)}])?[ \t]*$|Begin final coordinates|End final coordinates/gm;
  let m, final = false;
  while ((m = re.exec(text))) {
    if (m[0].includes("Begin final")) { final = true; continue; }
    if (m[0].includes("End final")) { final = false; continue; }
    const c = new Cursor(text, m.index + m[0].length + 1), unit = (m[2] || "").trim().toLowerCase();
    if (m[1] === "CELL_PARAMETERS") {
      const rows = [0, 1, 2].map(() => tokens(c.line() ?? "").slice(0, 3).map(Number));
      const a2 = unit.match(/alat\s*=\s*([-\d.Ee+]+)/);
      const s = a2 ? +a2[1] * BOHR : unit.startsWith("bohr") ? BOHR : unit.startsWith("ang") ? 1 : alat;
      L = rows.map(r => r.map(x => x * s));
    } else {
      const v = new Float64Array(3 * nat);
      for (let i = 0; i < nat; i++) {
        const t = tokens(c.line() ?? "");
        const q = t.slice(1, 4).map(Number);
        const r = unit.startsWith("crystal") ? fracToCart(L, q) : unit.startsWith("bohr") ? q.map(x => x * BOHR) : unit.startsWith("ang") ? q : q.map(x => x * alat);
        v.set(r, 3 * i);
      }
      const last = frames[frames.length - 1];
      if (final && v.every((x, k) => Math.abs(x - last[k]) < 1e-6)) continue;
      frames.push(v); cells.push(L);
    }
  }
  const energies = [...text.matchAll(/^!\s+total energy\s*=\s*([-\d.Ee+]+)\s*Ry/gm)].map(r => +r[1] * RY).slice(0, frames.length);
  if (frames.length > 1 && energies.length < frames.length - 1) notes.push("some steps have no converged energy");
  return finish(species, frames, cells, energies, { title: "pw.x output", notes });
}

/* ------------------------------------------------------------------ PDB (one or many MODELs) */

/** PDB: CRYST1 a b c alpha beta gamma (A, degrees); MODEL ... ENDMDL frames; ATOM/HETATM x, y, z in columns 31-54. */
export function parsePDB(text) {
  const frames = [], cells = [], notes = [];
  let species = null, L = null, sp = [], ps = [];
  const flush = () => {
    if (!sp.length) return true;
    if (species && (!sameCount(sp.length, species.length, notes, `model ${frames.length + 1}`))) return false;
    species = species || sp; frames.push(Float64Array.from(ps)); cells.push(L); sp = []; ps = [];
    return true;
  };
  for (const raw of text.split(/\r?\n/)) {
    const rec = raw.slice(0, 6);
    if (rec === "CRYST1") {
      const v = [raw.slice(6, 15), raw.slice(15, 24), raw.slice(24, 33), raw.slice(33, 40), raw.slice(40, 47), raw.slice(47, 54)].map(Number);
      // WHY: a 1 x 1 x 1 cell is the PDB placeholder for "no cell"
      L = v.every(Number.isFinite) && !(v[0] === 1 && v[1] === 1 && v[2] === 1) ? fromParams(...v) : null;
    } else if (rec === "ATOM  " || rec === "HETATM") {
      const el = raw.slice(76, 78).trim() || raw.slice(12, 16).trim().replace(/^\d+/, "");
      sp.push(elementOf(el)); ps.push(+raw.slice(30, 38), +raw.slice(38, 46), +raw.slice(46, 54));
    } else if (rec === "ENDMDL" || rec.startsWith("END ") || rec === "END") { if (!flush()) break; }
  }
  flush();
  return finish(species || [], frames, cells, [], { title: "PDB", notes, pbc: true });
}

/* ------------------------------------------------------------------ GROMACS .gro (one or many frames) */

/**
 * .gro: title line, atom count, atom lines (residue 5, residue name 5, atom name 5, atom number 5 columns, then x y z
 * in nm with fixed width; the width is read from the spacing of the decimal points), box line
 * v1x v2y v3z [v1y v1z v2x v2z v3x v3y] (nm).
 */
export function parseGRO(text) {
  const cur = new Cursor(text), frames = [], cells = [], notes = [];
  let species = null;
  for (;;) {
    const title = cur.nonEmpty();
    if (title === null) break;
    const n = parseInt(cur.line() ?? "");
    if (!Number.isInteger(n)) break;
    const lines = [];
    for (let i = 0; i < n; i++) lines.push(cur.line() ?? "");
    const p1 = lines[0].indexOf(".", 20), p2 = lines[0].indexOf(".", p1 + 1), w = p2 - p1 || 8;
    const sp = [], v = new Float64Array(3 * n);
    lines.forEach((l, i) => {
      sp.push(elementOf(l.slice(10, 15).trim().replace(/\d+$/, "")));
      for (let k = 0; k < 3; k++) v[3 * i + k] = 10 * +l.slice(20 + k * w, 20 + (k + 1) * w);
    });
    const b = tokens(cur.line() ?? "").map(x => 10 * +x);
    const L = b.length >= 9 ? [[b[0], b[3], b[4]], [b[5], b[1], b[6]], [b[7], b[8], b[2]]] : b.length >= 3 ? [[b[0], 0, 0], [0, b[1], 0], [0, 0, b[2]]] : null;
    if (species && !sameCount(n, species.length, notes, `frame ${frames.length + 1}`)) break;
    species = species || sp;
    frames.push(v); cells.push(L);
  }
  if (!frames.length) throw new Error(".gro: no frames");
  return finish(species, frames, cells, [], { title: "GROMACS", notes });
}

/* ------------------------------------------------------------------ CASTEP .md / .geom */

/**
 * CASTEP .md and .geom: after the header, blocks separated by blank lines with lines tagged "<-- E" (energies, Ha;
 * the first number is taken), "<-- h" (three cell rows, Bohr) and "<-- R" (species, index, x y z, Bohr).
 */
export function parseCASTEP(text) {
  const frames = [], cells = [], energies = [], notes = [];
  let species = null, sp = [], ps = [], h = [], L = null, E = NaN;
  const flush = () => {
    if (!sp.length) return true;
    if (h.length === 3) L = h.map(r => r.map(x => x * BOHR));
    if (species && !sameCount(sp.length, species.length, notes, `block ${frames.length + 1}`)) return false;
    species = species || sp; frames.push(Float64Array.from(ps.map(x => x * BOHR))); cells.push(L); energies.push(E * HA);
    sp = []; ps = []; h = []; E = NaN;
    return true;
  };
  const body = text.includes("END header") ? text.slice(text.indexOf("END header") + 10) : text;
  for (const raw of body.split(/\r?\n/)) {
    if (!raw.trim()) { if (!flush()) break; continue; }
    const tag = raw.match(/<--\s*(\w)\s*$/);
    if (!tag) continue;
    const t = tokens(raw.replace(/<--.*$/, ""));
    if (tag[1] === "E") E = +t[0];
    else if (tag[1] === "h") h.push(t.slice(0, 3).map(Number));
    else if (tag[1] === "R") { sp.push(elementOf(t[0])); ps.push(+t[2], +t[3], +t[4]); }
  }
  flush();
  if (!frames.length) throw new Error("CASTEP: no frames");
  return finish(species, frames, cells, energies, { title: "CASTEP", notes });
}

/* ------------------------------------------------------------------ ASE .traj (ULM binary) */

/** True if the buffer starts with the ULM magic of ASE trajectory files. */
export function isUlm(buf) {
  if (!buf || buf.byteLength < 48) return false;
  const m = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  return m === "- of Ulm" || m === "AFFormat";
}

/**
 * ASE .traj (ULM format, ase.io.ulm): magic "- of Ulm", a 16-byte tag, int64 version, item count and the offset of
 * the int64 item-offset table. Item k at offset o: int64 length n, then n bytes of JSON. Keys ending in "." hold
 * arrays as {"ndarray": [shape, dtype, byte offset]}, stored raw (little-endian unless "_little_endian": false).
 * Per item: "positions." (A), "cell" (rows, A), "pbc", and when they change "numbers." (atomic numbers); the energy
 * is "calculator."["energy"] (eV).
 */
export function parseAseTraj(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf), notes = [];
  const i64 = o => Number(dv.getBigInt64(o, true));
  const tag = new TextDecoder().decode(u8.subarray(8, 24)).trim();
  const nitems = i64(32), pos0 = i64(40);
  if (!(nitems > 0) || pos0 + 8 * nitems > buf.byteLength) throw new Error("ASE trajectory: empty or truncated file");
  const frames = [], cells = [], energies = [];
  let species = null, numbers = null, pbc = true;
  const dec = new TextDecoder();
  const arr = (ref, le) => {
    const [shape, dtype, off] = ref.ndarray, n = shape.reduce((a, b) => a * b, 1), out = new Float64Array(n);
    const get = { float64: [8, o => dv.getFloat64(o, le)], float32: [4, o => dv.getFloat32(o, le)], int64: [8, o => Number(dv.getBigInt64(o, le))],
      int32: [4, o => dv.getInt32(o, le)], int8: [1, o => dv.getInt8(o)], uint8: [1, o => dv.getUint8(o)], bool: [1, o => dv.getUint8(o)] }[dtype];
    if (!get) throw new Error(`ASE trajectory: array type ${dtype} is not supported`);
    for (let k = 0; k < n; k++) out[k] = get[1](off + k * get[0]);
    return out;
  };
  for (let k = 0; k < nitems; k++) {
    const o = i64(pos0 + 8 * k);
    if (o <= 0 || o + 8 > buf.byteLength) { notes.push("the file ends before the last frame (a run still writing?)"); break; }
    const n = i64(o);
    // WHY: Python's json writes NaN and Infinity, which JSON.parse rejects
    const d = JSON.parse(dec.decode(u8.subarray(o + 8, o + 8 + n)).replace(/\bNaN\b|-?\bInfinity\b/g, "null"));
    const le = d._little_endian !== false;
    const num = d["numbers."] ? arr(d["numbers."], le) : d.numbers;
    if (num) numbers = Array.from(num);
    if (!numbers) throw new Error("ASE trajectory: no atomic numbers");
    const sp = numbers.map(z => SYMBOLS[z] || "X");
    if (species && (!sameCount(sp.length, species.length, notes, `frame ${frames.length + 1}`))) break;
    species = species || sp;
    const P = d["positions."] ? arr(d["positions."], le) : d.positions ? Float64Array.from(d.positions.flat()) : null;
    if (!P || P.length !== 3 * species.length) { notes.push(`frame ${k + 1} has no positions; the trajectory stops before it`); break; }
    let C = d["cell."] ? Array.from(arr(d["cell."], le)) : d.cell ? d.cell.flat() : null;
    if (d.pbc) pbc = (Array.isArray(d.pbc) ? d.pbc : [d.pbc]).some(Boolean);
    const L = C && C.length === 9 && Math.abs(det3([C.slice(0, 3), C.slice(3, 6), C.slice(6, 9)])) > 1e-9 ? [C.slice(0, 3), C.slice(3, 6), C.slice(6, 9)] : cells[cells.length - 1] || null;
    const calc = d["calculator."] || d.calculator;
    frames.push(P); cells.push(L); energies.push(calc && typeof calc.energy === "number" ? calc.energy : NaN);
  }
  if (tag && tag !== "ASE-Trajectory") notes.push(`ULM file tagged "${tag}", read as an ASE trajectory`);
  return finish(species || [], frames, cells, energies, { title: "ASE trajectory", notes, pbc });
}
