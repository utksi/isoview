/**
 * Atoms to draw and the bonds between them.
 *
 * Origin shift: fractional coordinates f -> f + shift before wrapping (periodic), r -> r + shift . L (non-periodic
 * with a box) or r -> r + cart (no cell, Angstrom); the same shift moves the fields (field.js), so atoms and densities
 * stay together.
 * Periodic structures: fractional coordinates are wrapped into [0, 1) (values within eps of 1 map to 0), repeated
 * over the r_a x r_b x r_c supercell, and, with boundary atoms on, atoms on a cell face (fractional coordinate below
 * eps along an axis) are also drawn on the opposite face of the supercell (edges and corners by combination).
 * Bonds: atoms i, j are bonded when 0.1 A < |r_i - r_j| < tol (R_i + R_j), R = covalent radius. Pairs involving a
 * spacer element (alkali and heavy alkaline-earth cations) are dropped when the other atoms form at least one bond among
 * themselves (Cs in Cs2AgInCl6, Li in LiFePO4), and kept when they do not (NaCl, CaF2, Li2O would otherwise show no
 * bonds at all). Neighbours are found with a cubic cell list of edge = the largest cutoff.
 */
import { inv3, cartToFrac, fracToCart, det3 } from "./lattice.js?v=26";
import { rcov } from "./elements.js?v=26";

/** List of { el, x, y, z, src } with src the index of the atom in the structure. */
export function displayAtoms(st, { reps = [1, 1, 1], boundary = true, eps = 1e-3, shift = [0, 0, 0], cart = [0, 0, 0] } = {}) {
  const out = [], N = st.species.length, P = st.pos;
  if (!st.pbc || !st.lattice) {
    const d = st.lattice ? fracToCart(st.lattice, shift) : cart;
    for (let i = 0; i < N; i++) out.push({ el: st.species[i], x: P[3 * i] + d[0], y: P[3 * i + 1] + d[1], z: P[3 * i + 2] + d[2], src: i });
    return out;
  }
  const L = st.lattice, Li = inv3(L);
  for (let i = 0; i < N; i++) {
    const f = cartToFrac(Li, [P[3 * i], P[3 * i + 1], P[3 * i + 2]]).map((x, a) => { x += shift[a]; let y = x - Math.floor(x); if (y > 1 - eps) y -= 1; return y; });
    const sh = f.map((x, a) => { const s = []; for (let k = 0; k < reps[a]; k++) s.push(k); if (boundary && Math.abs(x) < eps) s.push(reps[a]); return s; });
    for (const a of sh[0]) for (const b of sh[1]) for (const c of sh[2]) {
      const r = fracToCart(L, [f[0] + a, f[1] + b, f[2] + c]);
      out.push({ el: st.species[i], x: r[0], y: r[1], z: r[2], src: i });
    }
  }
  return out;
}

/** Bonded index pairs [i, j] (i < j) among display atoms; spacer pairs dropped as in the file header. */
export function bondList(atoms, { tol = 1.15, spacers = new Set() } = {}) {
  if (spacers.size && atoms.some(a => spacers.has(a.el))) {
    const framework = pairs(atoms, tol, spacers);
    if (framework.length) return framework;
  }
  return pairs(atoms, tol, new Set());
}

/** Pairs [i, j] (i < j) with 0.1 A < d_ij < tol (R_i + R_j), ignoring atoms whose element is in `skip`. */
function pairs(atoms, tol, skip) {
  const n = atoms.length, R = atoms.map(a => rcov(a.el));
  let rmax = 0;
  for (let i = 0; i < n; i++) if (!skip.has(atoms[i].el) && R[i] > rmax) rmax = R[i];
  const h = Math.max(0.5, 2 * tol * rmax), cells = new Map(), key = (a, b, c) => a + "," + b + "," + c;
  const cell = atoms.map(a => [Math.floor(a.x / h), Math.floor(a.y / h), Math.floor(a.z / h)]);
  for (let i = 0; i < n; i++) {
    if (skip.has(atoms[i].el)) continue;
    const k = key(...cell[i]);
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(i);
  }
  const out = [];
  for (let i = 0; i < n; i++) {
    if (skip.has(atoms[i].el)) continue;
    const [cx, cy, cz] = cell[i], a = atoms[i];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const list = cells.get(key(cx + dx, cy + dy, cz + dz));
      if (!list) continue;
      for (const j of list) {
        if (j <= i) continue;
        const b = atoms[j], d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (d > 0.1 && d < tol * (R[i] + R[j])) out.push([i, j]);
      }
    }
  }
  return out;
}

/** Atom number shown to the user: the 1-based index in the file, kept through deletions (st.ids). */
export const atomId = (st, i) => (st.ids ? st.ids[i] : i + 1);

/** The structure without the atoms whose indices are in `drop` (a Set); original numbers are kept in `ids`. */
export function deleteAtoms(st, drop) {
  const keep = [];
  for (let i = 0; i < st.species.length; i++) if (!drop.has(i)) keep.push(i);
  const pos = new Float64Array(3 * keep.length);
  keep.forEach((i, k) => pos.set(st.pos.subarray(3 * i, 3 * i + 3), 3 * k));
  return { ...st, species: keep.map(i => st.species[i]), pos, ids: Int32Array.from(keep.map(i => atomId(st, i))) };
}

/**
 * POSCAR text (VASP 5, Direct coordinates) of a structure after the origin shift: atoms grouped by species in order of
 * first appearance, fractional coordinates wrapped into [0, 1). A structure without a cell gets a cubic box with
 * 10 A of vacuum around the atoms (Cartesian coordinates).
 */
export function toPOSCAR(st, shift = [0, 0, 0], title = "isoview") {
  const order = [];
  for (const s of st.species) if (!order.includes(s)) order.push(s);
  const idx = order.flatMap(sp => st.species.map((s, i) => (s === sp ? i : -1)).filter(i => i >= 0));
  const f9 = x => x.toFixed(10).padStart(16);
  const lines = [title.replace(/\s+/g, " ").slice(0, 80), "1.0"];
  let coords;
  if (st.lattice && Math.abs(det3(st.lattice)) > 1e-9) {
    const Li = inv3(st.lattice);
    lines.push(...st.lattice.map(v => v.map(f9).join(" ")));
    coords = idx.map(i => cartToFrac(Li, [st.pos[3 * i], st.pos[3 * i + 1], st.pos[3 * i + 2]]).map((x, a) => {
      x += shift[a];
      return st.pbc ? x - Math.floor(x) : x;
    }));
    lines.push("  " + order.join("  "), "  " + order.map(sp => st.species.filter(s => s === sp).length).join("  "), "Direct");
  } else {
    const lo = [0, 1, 2].map(c => Math.min(...idx.map(i => st.pos[3 * i + c]))), hi = [0, 1, 2].map(c => Math.max(...idx.map(i => st.pos[3 * i + c])));
    const L = Math.max(...hi.map((h, c) => h - lo[c])) + 10;
    lines.push(f9(L) + f9(0) + f9(0), f9(0) + f9(L) + f9(0), f9(0) + f9(0) + f9(L));
    lines.push("  " + order.join("  "), "  " + order.map(sp => st.species.filter(s => s === sp).length).join("  "), "Cartesian");
    coords = idx.map(i => [0, 1, 2].map(c => st.pos[3 * i + c] - lo[c] + 5));
  }
  for (const c of coords) lines.push(c.map(f9).join(" "));
  return lines.join("\n") + "\n";
}

/**
 * The structure as shown, for export: atoms translated by the Cartesian vector t, wrapped into the cell (periodic)
 * and repeated over the reps supercell (lattice rows scaled by reps); boundary duplicates are not included. A
 * non-periodic box is moved to start at 0 (positions taken relative to its origin).
 */
export function shownStructure(st, t = [0, 0, 0], reps = [1, 1, 1]) {
  if (!st.pbc || !st.lattice) {
    // a non-periodic box is written from 0: positions relative to its corner
    const o = st.origin || [0, 0, 0], pos = Float64Array.from(st.pos, (x, k) => x + t[k % 3] - (st.lattice ? o[k % 3] : 0));
    return { ...st, pos, origin: undefined };
  }
  const L = st.lattice, Li = inv3(L), species = [], pos = [];
  for (let i = 0; i < st.species.length; i++) {
    const f = cartToFrac(Li, [st.pos[3 * i] + t[0], st.pos[3 * i + 1] + t[1], st.pos[3 * i + 2] + t[2]]).map(x => { const y = x - Math.floor(x); return y > 1 - 1e-9 ? 0 : y; });
    for (let a = 0; a < reps[0]; a++) for (let b = 0; b < reps[1]; b++) for (let c = 0; c < reps[2]; c++) {
      pos.push(...fracToCart(L, [f[0] + a, f[1] + b, f[2] + c])); species.push(st.species[i]);
    }
  }
  return { lattice: L.map((v, a) => v.map(x => x * reps[a])), pbc: true, species, pos: Float64Array.from(pos) };
}

/** CIF text in space group P 1 (cell lengths in A, angles in degrees, fractional coordinates). */
export function toCIF(st, title = "isoview") {
  if (!st.lattice) throw new Error("CIF needs a cell");
  const [a, b, c] = st.lattice.map(v => Math.hypot(...v));
  const an = (u, v) => Math.acos((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v))) * 180 / Math.PI;
  const L = st.lattice, Li = inv3(L), cnt = {};
  const lines = [`# ${title.replace(/\s+/g, " ")}`, "data_isoview", `_cell_length_a ${a.toFixed(6)}`, `_cell_length_b ${b.toFixed(6)}`, `_cell_length_c ${c.toFixed(6)}`,
    `_cell_angle_alpha ${an(L[1], L[2]).toFixed(4)}`, `_cell_angle_beta ${an(L[0], L[2]).toFixed(4)}`, `_cell_angle_gamma ${an(L[0], L[1]).toFixed(4)}`,
    "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_Int_Tables_number 1", "loop_", "_symmetry_equiv_pos_as_xyz", "'x, y, z'",
    "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"];
  st.species.forEach((s, i) => {
    cnt[s] = (cnt[s] || 0) + 1;
    const f = cartToFrac(Li, [st.pos[3 * i], st.pos[3 * i + 1], st.pos[3 * i + 2]]);
    lines.push(`${s}${cnt[s]} ${s} ${f.map(x => x.toFixed(8)).join(" ")}`);
  });
  return lines.join("\n") + "\n";
}

/** Extended XYZ text: Lattice (A) and pbc in the comment line when the structure has a cell; props.energy (eV) is
 *  written as energy=... when given. */
export function toXYZ(st, title = "isoview", props = {}) {
  const en = Number.isFinite(props.energy) ? ` energy=${props.energy.toFixed(8)}` : "";
  const head = st.lattice ? `Lattice="${st.lattice.flat().map(x => x.toFixed(8)).join(" ")}" Properties=species:S:1:pos:R:3${en} pbc="${st.pbc ? "T T T" : "F F F"}" comment="${title.replace(/"/g, "'")}"` : title + en;
  const lines = [String(st.species.length), head];
  st.species.forEach((s, i) => lines.push(`${s.padEnd(3)} ${[0, 1, 2].map(c => st.pos[3 * i + c].toFixed(8).padStart(14)).join(" ")}`));
  return lines.join("\n") + "\n";
}
