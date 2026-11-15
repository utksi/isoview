/**
 * Atoms to draw and the bonds between them.
 *
 * Origin shift: fractional coordinates f -> f + shift before wrapping (periodic), or r -> r + shift . L (non-periodic
 * with a box); the same shift moves the fields (field.js), so atoms and densities stay together.
 * Periodic structures: fractional coordinates are wrapped into [0, 1) (values within eps of 1 map to 0), repeated
 * over the r_a x r_b x r_c supercell, and, with boundary atoms on, atoms on a cell face (fractional coordinate below
 * eps along an axis) are also drawn on the opposite face of the supercell (edges and corners by combination).
 * Bonds: atoms i, j are bonded when 0.1 A < |r_i - r_j| < tol (R_i + R_j), R = covalent radius; pairs involving an
 * element in `skip` are not bonded. Neighbours are found with a cubic cell list of edge = the largest cutoff.
 */
import { inv3, cartToFrac, fracToCart, det3 } from "./lattice.js";
import { rcov } from "./elements.js";

/** List of { el, x, y, z, src } with src the index of the atom in the structure. */
export function displayAtoms(st, { reps = [1, 1, 1], boundary = true, eps = 1e-3, shift = [0, 0, 0] } = {}) {
  const out = [], N = st.species.length, P = st.pos;
  if (!st.pbc || !st.lattice) {
    const d = st.lattice ? fracToCart(st.lattice, shift) : [0, 0, 0];
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

/** Bonded index pairs [i, j] (i < j) among display atoms. */
export function bondList(atoms, { tol = 1.15, skip = new Set() } = {}) {
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
