/**
 * Lattice algebra. A lattice is a 3x3 array of row vectors a, b, c in Angstrom; Cartesian r = f0 a + f1 b + f2 c for
 * fractional coordinates f. Angles are in degrees.
 *
 * fromParams: a along x, b in the xy plane, c completing a right-handed cell:
 *   a = (a, 0, 0); b = (b cos g, b sin g, 0);
 *   c = (c cos be, c (cos al - cos be cos g) / sin g, V / (a b sin g)),  V = a b c sqrt(1 - cos^2 al - cos^2 be
 *   - cos^2 g + 2 cos al cos be cos g).
 */

export const BOHR = 0.529177210903; // Angstrom

export function det3(m) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/** Inverse of a 3x3 matrix. */
export function inv3(m) {
  const d = det3(m);
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d],
  ];
}

/** Cartesian position (Angstrom) of fractional coordinates f in lattice L. */
export function fracToCart(L, f) {
  return [0, 1, 2].map(j => f[0] * L[0][j] + f[1] * L[1][j] + f[2] * L[2][j]);
}

/** Fractional coordinates of Cartesian r, given Li = inv3(L). */
export function cartToFrac(Li, r) {
  return [0, 1, 2].map(j => r[0] * Li[0][j] + r[1] * Li[1][j] + r[2] * Li[2][j]);
}

export const norm = v => Math.hypot(v[0], v[1], v[2]);
export const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
export const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];

/** Row vectors from cell parameters (Angstrom, degrees), in the orientation of the file docstring. */
export function fromParams(a, b, c, al, be, ga) {
  const r = Math.PI / 180, ca = Math.cos(al * r), cb = Math.cos(be * r), cg = Math.cos(ga * r), sg = Math.sin(ga * r);
  const v = Math.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);
  return [[a, 0, 0], [b * cg, b * sg, 0], [c * cb, c * (ca - cb * cg) / sg, c * v / sg]];
}

/** Cell parameters [a, b, c, alpha, beta, gamma] (Angstrom, degrees). */
export function params(L) {
  const [a, b, c] = L.map(norm);
  const ang = (u, v) => Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (norm(u) * norm(v))))) * 180 / Math.PI;
  return [a, b, c, ang(L[1], L[2]), ang(L[0], L[2]), ang(L[0], L[1])];
}

/** Composition string with Unicode subscripts, elements in order of first appearance: "Cs₂AgInCl₆". */
export function formula(species, reduce = false) {
  const order = [], cnt = new Map();
  for (const s of species) { if (!cnt.has(s)) { order.push(s); cnt.set(s, 0); } cnt.set(s, cnt.get(s) + 1); }
  let g = 0;
  if (reduce) { const gcd = (x, y) => (y ? gcd(y, x % y) : x); for (const v of cnt.values()) g = gcd(g, v); }
  const sub = n => String(n).replace(/\d/g, d => "₀₁₂₃₄₅₆₇₈₉"[d]);
  return order.map(s => { const n = cnt.get(s) / (g || 1); return s + (n === 1 ? "" : sub(n)); }).join("");
}
