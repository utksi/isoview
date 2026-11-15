/**
 * Atoms to draw and the bonds between them.
 *
 * Periodic structures: fractional coordinates are wrapped into [0, 1) (values within eps of 1 map to 0), repeated
 * over the r_a x r_b x r_c supercell, and, with boundary atoms on, atoms on a cell face (fractional coordinate below
 * eps along an axis) are also drawn on the opposite face of the supercell (edges and corners by combination).
 * Bonds: atoms i, j are bonded when 0.1 A < |r_i - r_j| < tol (R_i + R_j), R = covalent radius; pairs involving an
 * element in `skip` are not bonded. Neighbours are found with a cubic cell list of edge = the largest cutoff.
 */
import { inv3, cartToFrac, fracToCart } from "./lattice.js";
import { rcov } from "./elements.js";

/** List of { el, x, y, z, src } with src the index of the atom in the structure. */
export function displayAtoms(st, { reps = [1, 1, 1], boundary = true, eps = 1e-3 } = {}) {
  const out = [], N = st.species.length, P = st.pos;
  if (!st.pbc || !st.lattice) {
    for (let i = 0; i < N; i++) out.push({ el: st.species[i], x: P[3 * i], y: P[3 * i + 1], z: P[3 * i + 2], src: i });
    return out;
  }
  const L = st.lattice, Li = inv3(L);
  for (let i = 0; i < N; i++) {
    const f = cartToFrac(Li, [P[3 * i], P[3 * i + 1], P[3 * i + 2]]).map(x => { let y = x - Math.floor(x); if (y > 1 - eps) y -= 1; return y; });
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
