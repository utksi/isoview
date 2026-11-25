// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * First Brillouin zone and the high-symmetry k-path of Setyawan and Curtarolo, Comput. Mater. Sci. 49, 299 (2010)
 * ("SC"), as pymatgen's HighSymmKpath(path_type="setyawan_curtarolo") gives them.
 *
 * Reciprocal lattice: for real-space rows a_1, a_2, a_3 (A), the rows b_i (1/A) satisfy b_i . a_j = 2 pi delta_ij,
 * i.e. B = 2 pi (L^-1)^T with L the matrix of rows a_j.
 *
 * Wigner-Seitz cell of a lattice with rows B: the set of k with k . G <= |G|^2 / 2 for every lattice vector G =
 * n1 b1 + n2 b2 + n3 b3 != 0. The basis is first reduced to its successive minima (shortest vector, shortest one not
 * parallel to it, shortest one not in their plane), for which every facet vector has |n_i| <= 1, so the search over
 * |n_i| <= 2 contains all of them. A vector G is a facet vector (Voronoi-relevant) when G/2 lies strictly inside
 * every other half-space: G/2 . H < |H|^2 / 2 for all H != G. Vertices are the solutions of three facet equations
 * k . G = |G|^2 / 2 that satisfy all half-spaces; vertices closer than 1e-6 s (s = the longest facet vector) are
 * merged. The zone volume is |det B| = (2 pi)^3 / V_cell.
 *
 * k-path: the Bravais lattice of the structure is found from its lattice alone. The primitive cell P (rows, A) comes
 * from the pure translations of the structure (symmetry.js primitive), and the conventional cell C, the crystal
 * system and the centring multiplicity z = V_C / V_P from the lattice point group of P (symmetry.js conventional
 * applied to the bare lattice). C is then relabelled into the SC conventional cell and P_SC = T C, with T the SC
 * centring matrix:
 *   F: T = [[0,1,1],[1,0,1],[1,1,0]] / 2        I: T = [[-1,1,1],[1,-1,1],[1,1,-1]] / 2
 *   C (orthorhombic): T = [[1,-1,0],[1,1,0],[0,0,2]] / 2     C (monoclinic): T = [[1,1,0],[-1,1,0],[0,0,2]] / 2
 *   R: P_SC = the three vectors (2A+B+C)/3, (-A+B+C)/3, (-A-2B+C)/3 of the hexagonal triple cell (A, B, C) (obverse;
 *      A, B -> -A, -B for the reverse setting), equal lengths a and equal angles alpha.
 * SC conventions for the conventional cell (the same as pymatgen's get_conventional_standard_structure with
 * international_monoclinic=False): ORC, ORCF, ORCI: a < b < c; ORCC: c the axis normal to the centred face, a < b;
 * MCL: a the unique axis, b < c the reduced pair of the plane normal to it, alpha = angle(b, c) < 90 deg; MCLC: a the
 * unique axis, b the in-plane vector of the centring (a + b)/2 (the shortest of its class), c the shortest in-plane
 * vector completing the plane's basis, alpha < 90 deg; TRI: the reduced primitive basis with an even number of vectors
 * negated so that the reciprocal angles k_alpha, k_beta, k_gamma are all > 90 deg (TRI1a) or all < 90 deg (TRI1b).
 * Variants (a, b, c, alpha of the SC conventional cell; k_gamma of the reciprocal of P_SC):
 *   BCT1 c < a, BCT2 c > a; ORCF1/2/3 1/a^2 >, <, = 1/b^2 + 1/c^2; RHL1 alpha < 90, RHL2 alpha > 90;
 *   MCLC1/2 k_gamma >, = 90; MCLC3/4/5 k_gamma < 90 and b cos(alpha)/c + b^2 sin^2(alpha)/a^2 <, =, > 1;
 *   TRI2a: k_gamma = 90 (the axes cycled to put the right angle there), k_alpha, k_beta > 90 (otherwise TRI1a/1b);
 *   TRI2b (k_alpha, k_beta < 90) is another setting of the same lattices and is not returned.
 * Equalities are tested with the relative tolerance eps = tol / (shortest conventional length), so a cell within
 * the symmetry tolerance of a boundary takes the boundary variant (ORCF3, MCLC2, MCLC4, TRI2a); pymatgen tests
 * them with exact float equality and practically never returns those. The points of each variant are the fractional
 * coordinates of the SC tables in the basis of the reciprocal rows of P_SC; Cartesian k = sum_i f_i b_i (1/A).
 * Since P_SC is built from lattice vectors of the input, B and the points are in the input's own Cartesian frame.
 *
 * Structures with lower symmetry than their lattice: this module uses the holohedry of the lattice (the point group
 * of the metric of the primitive cell), so the zone, the type and the labels are those of the Bravais lattice. pymatgen
 * instead takes the crystal system and centring letter from the space group (spglib) and the conventional cell of
 * that space group; for a structure whose atoms break the lattice symmetry (e.g. a tetragonal space group in a cell
 * with a = c, or a monoclinic space group in a cell with beta = 90 deg) pymatgen returns the path of the lower system
 * while this module returns that of the lattice's higher system. For structures whose space group has the full
 * holohedry of the lattice the two agree (up to a symmetry operation of the lattice, which leaves the labels
 * equivalent). One deliberate difference: pymatgen's MCL table gives M_2 = M_1 (a typo); here M_2 = (1/2, eta, -nu)
 * as in the SC paper. M_2 is not on the MCL path.
 *
 * Points outside the zone: a few SC points lie outside the Wigner-Seitz zone of their own B (MCL D_1, TRI1b N, found
 * in the tests). kpath keeps their SC coordinates in frac and returns in points the equivalent k - G nearest to Gamma
 * (G the reciprocal lattice vector nearest to k), with the integer shift in folded.
 *
 * Zone tolerances: facets are kept when G/2 is inside every other half-space by more than 2e-9 |H|^2; for a cell
 * strained by 1e-7 to 1e-6 from a higher symmetry the tiny facets this drops leave a relative volume error of about
 * the strain, and the polyhedron stays closed (V - E + F = 2).
 */
import { inv3, det3, dot, cross, norm, fracToCart, cartToFrac } from "./lattice.js?v=37";
import { primitive, conventional } from "./symmetry.js?v=37";

const TWO_PI = 2 * Math.PI;

/**
 * Reciprocal lattice rows b1, b2, b3 (1/A) of the real-space rows L = [a, b, c] (A): b_i . a_j = 2 pi delta_ij,
 * B = 2 pi (L^-1)^T. No factor-2pi-free convention: |b_i| = 2 pi / d_i.
 */
export function reciprocal(L) {
  const M = inv3(L);
  return [0, 1, 2].map(i => [0, 1, 2].map(j => TWO_PI * M[j][i]));
}

/**
 * Successive minima of the lattice with rows B (any units): the shortest nonzero vector, the shortest one not
 * parallel to it and the shortest one not in their plane, among n1 b1 + n2 b2 + n3 b3, |n_i| <= 2, after a
 * pairwise (Gauss) reduction of B. Returns rows spanning the same lattice (same |det|), sign of det not fixed.
 */
function reduce(B) {
  let R = B.map(v => v.slice());
  // WHY: pairwise reduction first, so the |n| <= 2 window below holds the minima even for very oblique input
  for (let it = 0, changed = true; changed && it < 100; it++) {
    changed = false;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      if (i === j) continue;
      const m = Math.round(dot(R[i], R[j]) / dot(R[j], R[j]));
      if (m !== 0) {
        const v = R[i].map((x, k) => x - m * R[j][k]);
        if (norm(v) < norm(R[i]) * (1 - 1e-12)) { R[i] = v; changed = true; }
      }
    }
  }
  const cand = [];
  for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) for (let k = -2; k <= 2; k++) {
    if (!i && !j && !k) continue;
    const v = fracToCart(R, [i, j, k]);
    cand.push({ v, n: norm(v) });
  }
  cand.sort((p, q) => p.n - q.n);
  const v1 = cand[0].v;
  const c2 = cand.find(c => norm(cross(v1, c.v)) > 1e-8 * cand[0].n * c.n);
  const n12 = norm(cross(v1, c2.v));
  const c3 = cand.find(c => Math.abs(det3([v1, c2.v, c.v])) > 1e-8 * n12 * c.n);
  const S = [v1, c2.v, c3.v];
  return Math.abs(Math.abs(det3(S)) - Math.abs(det3(R))) < 1e-8 * Math.abs(det3(R)) ? S : R;
}

/**
 * Wigner-Seitz cell of the lattice with rows B (for the Brillouin zone: B = reciprocal rows, 1/A; centred at 0).
 * Returns { vertices: [[x,y,z]], faces: [[vertex indices, counter-clockwise seen from outside]], edges: [[i, j]]
 * (i < j), planes: [[Gx,Gy,Gz]] (the lattice vector of each face: the face lies on k . G = |G|^2 / 2, outward
 * normal G / |G|) }. Method in the file header; tolerances relative to s = the longest facet vector.
 */
export function wignerSeitz(B) {
  const R = reduce(B);
  const all = [];
  for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) for (let k = -2; k <= 2; k++) {
    if (!i && !j && !k) continue;
    const g = fracToCart(R, [i, j, k]);
    all.push({ g, h: dot(g, g) / 2 });
  }
  const rel = 1e-9;
  // facet vectors: G/2 strictly inside every other half-space
  const P = all.filter(p => all.every(q => q === p || dot(p.g, q.g) / 2 < q.h - rel * q.h * 2));
  const s = Math.max(...P.map(p => norm(p.g)));
  const vtol = 1e-7 * s * s, mtol = 1e-6 * s;
  const V = [];
  for (let a = 0; a < P.length; a++) for (let b = a + 1; b < P.length; b++) for (let c = b + 1; c < P.length; c++) {
    const M = [P[a].g, P[b].g, P[c].g], d = det3(M);
    if (Math.abs(d) < 1e-9 * norm(M[0]) * norm(M[1]) * norm(M[2])) continue;
    const Mi = inv3(M), rhs = [P[a].h, P[b].h, P[c].h];
    const x = [0, 1, 2].map(r => Mi[r][0] * rhs[0] + Mi[r][1] * rhs[1] + Mi[r][2] * rhs[2]);
    if (!P.every(p => dot(x, p.g) <= p.h + vtol)) continue;
    if (!V.some(v => norm([v[0] - x[0], v[1] - x[1], v[2] - x[2]]) < mtol)) V.push(x);
  }
  const faces = [], planes = [];
  for (const p of P) {
    const gn = norm(p.g), n = p.g.map(x => x / gn);
    const idx = V.map((v, i) => i).filter(i => Math.abs(dot(V[i], n) - gn / 2) < mtol);
    if (idx.length < 3) continue;
    const cen = [0, 1, 2].map(k => idx.reduce((t, i) => t + V[i][k], 0) / idx.length);
    const ref = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u0 = cross(n, ref), u = u0.map(x => x / norm(u0)), w = cross(n, u);
    const ang = i => { const d = V[i].map((x, k) => x - cen[k]); return Math.atan2(dot(d, w), dot(d, u)); };
    idx.sort((i, j) => ang(i) - ang(j));
    faces.push(idx);
    planes.push(p.g.slice());
  }
  const seen = new Set(), edges = [];
  for (const f of faces) for (let k = 0; k < f.length; k++) {
    const i = Math.min(f[k], f[(k + 1) % f.length]), j = Math.max(f[k], f[(k + 1) % f.length]), key = i + "," + j;
    if (!seen.has(key)) { seen.add(key); edges.push([i, j]); }
  }
  return { vertices: V, faces, edges, planes };
}

/**
 * Volume enclosed by a zone from wignerSeitz (same units cubed): sum over faces of the fan triangles
 * (v0 . (v_i x v_{i+1})) / 6, positive for outward counter-clockwise faces.
 */
export function zoneVolume(zone) {
  let s = 0;
  for (const f of zone.faces) for (let k = 1; k + 1 < f.length; k++) {
    s += det3([zone.vertices[f[0]], zone.vertices[f[k]], zone.vertices[f[k + 1]]]);
  }
  return s / 6;
}

/**
 * True if k (same units as the zone) lies in or on the zone: k . G <= |G|^2 / 2 + rtol |G|^2 for every face vector G
 * (rtol relative, default 1e-6).
 */
export function inZone(zone, k, rtol = 1e-6) {
  return zone.planes.every(g => dot(k, g) <= dot(g, g) * (0.5 + rtol));
}

/* ------------------------------------------------------------------ SC tables */

/**
 * Points (fractional, in the reciprocal basis of P_SC) and path of the SC variant v; q = { a, b, c, al } holds the
 * SC conventional lengths (A) and alpha (radians) that the variant's parameters need. Gamma is "G"; subscripts are
 * written "_1"; the Greek label of BCT2 is "Sigma". Formulas as in pymatgen kpath.py (KPathSetyawanCurtarolo).
 */
function table(v, q) {
  const { a, b, c, al } = q, h = 0.5, ca = Math.cos(al), sa = Math.sin(al);
  switch (v) {
    case "cub":
      return { pts: { G: [0, 0, 0], X: [0, h, 0], R: [h, h, h], M: [h, h, 0] },
        path: [["G", "X", "M", "G", "R", "X"], ["M", "R"]] };
    case "fcc":
      return { pts: { G: [0, 0, 0], K: [3 / 8, 3 / 8, 3 / 4], L: [h, h, h], U: [5 / 8, 1 / 4, 5 / 8], W: [h, 1 / 4, 3 / 4], X: [h, 0, h] },
        path: [["G", "X", "W", "K", "G", "L", "U", "W", "L", "K"], ["U", "X"]] };
    case "bcc":
      return { pts: { G: [0, 0, 0], H: [h, -h, h], P: [1 / 4, 1 / 4, 1 / 4], N: [0, 0, h] },
        path: [["G", "H", "N", "G", "P", "H"], ["P", "N"]] };
    case "tet":
      return { pts: { G: [0, 0, 0], A: [h, h, h], M: [h, h, 0], R: [0, h, h], X: [0, h, 0], Z: [0, 0, h] },
        path: [["G", "X", "M", "G", "Z", "R", "A", "Z"], ["X", "R"], ["M", "A"]] };
    case "bct1": {
      const e = (1 + c * c / (a * a)) / 4;
      return { pts: { G: [0, 0, 0], M: [-h, h, h], N: [0, h, 0], P: [1 / 4, 1 / 4, 1 / 4], X: [0, 0, h], Z: [e, e, -e], Z_1: [-e, 1 - e, e] },
        path: [["G", "X", "M", "G", "Z", "P", "N", "Z_1", "M"], ["X", "P"]] };
    }
    case "bct2": {
      const e = (1 + a * a / (c * c)) / 4, z = a * a / (2 * c * c);
      return { pts: { G: [0, 0, 0], N: [0, h, 0], P: [1 / 4, 1 / 4, 1 / 4], Sigma: [-e, e, e], Sigma_1: [e, 1 - e, -e], X: [0, 0, h], Y: [-z, z, h], Y_1: [h, h, -z], Z: [h, h, -h] },
        path: [["G", "X", "Y", "Sigma", "G", "Z", "Sigma_1", "N", "P", "Y_1", "Z"], ["X", "P"]] };
    }
    case "orc":
      return { pts: { G: [0, 0, 0], R: [h, h, h], S: [h, h, 0], T: [0, h, h], U: [h, 0, h], X: [h, 0, 0], Y: [0, h, 0], Z: [0, 0, h] },
        path: [["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z"], ["Y", "T"], ["U", "X"], ["S", "R"]] };
    case "orcf1": case "orcf3": {
      const z = (1 + a * a / (b * b) - a * a / (c * c)) / 4, e = (1 + a * a / (b * b) + a * a / (c * c)) / 4;
      const pts = { G: [0, 0, 0], A: [h, h + z, z], A_1: [h, h - z, 1 - z], L: [h, h, h], T: [1, h, h], X: [0, e, e], X_1: [1, 1 - e, 1 - e], Y: [h, 0, h], Z: [h, h, 0] };
      return v === "orcf1"
        ? { pts, path: [["G", "Y", "T", "Z", "G", "X", "A_1", "Y"], ["T", "X_1"], ["X", "A", "Z"], ["L", "G"]] }
        : { pts, path: [["G", "Y", "T", "Z", "G", "X", "A_1", "Y"], ["X", "A", "Z"], ["L", "G"]] };
    }
    case "orcf2": {
      const f = (1 + c * c / (b * b) - c * c / (a * a)) / 4, e = (1 + a * a / (b * b) - a * a / (c * c)) / 4, d = (1 + b * b / (a * a) - b * b / (c * c)) / 4;
      return { pts: { G: [0, 0, 0], C: [h, h - e, 1 - e], C_1: [h, h + e, e], D: [h - d, h, 1 - d], D_1: [h + d, h, d], L: [h, h, h], H: [1 - f, h - f, h], H_1: [f, h + f, h], X: [0, h, h], Y: [h, 0, h], Z: [h, h, 0] },
        path: [["G", "Y", "C", "D", "X", "G", "Z", "D_1", "H", "C"], ["C_1", "Z"], ["X", "H_1"], ["H", "Y"], ["L", "G"]] };
    }
    case "orci": {
      const z = (1 + a * a / (c * c)) / 4, e = (1 + b * b / (c * c)) / 4, d = (b * b - a * a) / (4 * c * c), m = (a * a + b * b) / (4 * c * c);
      return { pts: { G: [0, 0, 0], L: [-m, m, h - d], L_1: [m, -m, h + d], L_2: [h - d, h + d, -m], R: [0, h, 0], S: [h, 0, 0], T: [0, 0, h], W: [1 / 4, 1 / 4, 1 / 4], X: [-z, z, z], X_1: [z, 1 - z, -z], Y: [e, -e, e], Y_1: [1 - e, e, -e], Z: [h, h, -h] },
        path: [["G", "X", "L", "T", "W", "R", "X_1", "Z", "G", "Y", "S", "W"], ["L_1", "Y"], ["Y_1", "Z"]] };
    }
    case "orcc": {
      const z = (1 + a * a / (b * b)) / 4;
      return { pts: { G: [0, 0, 0], A: [z, z, h], A_1: [-z, 1 - z, h], R: [0, h, h], S: [0, h, 0], T: [-h, h, h], X: [z, z, 0], X_1: [-z, 1 - z, 0], Y: [-h, h, 0], Z: [0, 0, h] },
        path: [["G", "X", "S", "R", "A", "Z", "G", "Y", "X_1", "A_1", "T", "Y"], ["Z", "T"]] };
    }
    case "hex":
      return { pts: { G: [0, 0, 0], A: [0, 0, h], H: [1 / 3, 1 / 3, h], K: [1 / 3, 1 / 3, 0], L: [h, 0, h], M: [h, 0, 0] },
        path: [["G", "M", "K", "G", "A", "L", "H", "A"], ["L", "M"], ["K", "H"]] };
    case "rhl1": {
      const e = (1 + 4 * ca) / (2 + 4 * ca), n = 3 / 4 - e / 2;
      return { pts: { G: [0, 0, 0], B: [e, h, 1 - e], B_1: [h, 1 - e, e - 1], F: [h, h, 0], L: [h, 0, 0], L_1: [0, 0, -h], P: [e, n, n], P_1: [1 - n, 1 - n, 1 - e], P_2: [n, n, e - 1], Q: [1 - n, n, 0], X: [n, 0, -n], Z: [h, h, h] },
        path: [["G", "L", "B_1"], ["B", "Z", "G", "X"], ["Q", "F", "P_1", "Z"], ["L", "P"]] };
    }
    case "rhl2": {
      const e = 1 / (2 * Math.tan(al / 2) ** 2), n = 3 / 4 - e / 2;
      return { pts: { G: [0, 0, 0], F: [h, -h, 0], L: [h, 0, 0], P: [1 - n, -n, 1 - n], P_1: [n, n - 1, n - 1], Q: [e, e, e], Q_1: [1 - e, -e, -e], Z: [h, -h, h] },
        path: [["G", "P", "Z", "Q", "G", "F", "P_1", "Q_1", "L", "Z"]] };
    }
    case "mcl": {
      const e = (1 - b * ca / c) / (2 * sa * sa), n = h - e * c * ca / b;
      return { pts: { G: [0, 0, 0], A: [h, h, 0], C: [0, h, h], D: [h, 0, h], D_1: [h, h, -h], E: [h, h, h], H: [0, e, 1 - n], H_1: [0, 1 - e, n], H_2: [0, e, -n], M: [h, e, 1 - n], M_1: [h, 1 - e, n], M_2: [h, e, -n], X: [0, h, 0], Y: [0, 0, h], Y_1: [0, 0, -h], Z: [h, 0, 0] },
        path: [["G", "Y", "H", "C", "E", "M_1", "A", "X", "H_1"], ["M", "D", "Z"], ["Y", "D"]] };
    }
    case "mclc1": case "mclc2": {
      const z = (2 - b * ca / c) / (4 * sa * sa), e = h + 2 * z * c * ca / b, p = 3 / 4 - a * a / (4 * b * b * sa * sa), f = p + (3 / 4 - p) * b * ca / c;
      const pts = { G: [0, 0, 0], N: [h, 0, 0], N_1: [0, -h, 0], F: [1 - z, 1 - z, 1 - e], F_1: [z, z, e], F_2: [-z, -z, 1 - e], I: [f, 1 - f, h], I_1: [1 - f, f - 1, h], L: [h, h, h], M: [h, 0, h], X: [1 - p, p - 1, 0], X_1: [p, 1 - p, 0], X_2: [p - 1, -p, 0], Y: [h, h, 0], Y_1: [-h, -h, 0], Z: [0, 0, h] };
      if (v === "mclc1") return { pts, path: [["G", "Y", "F", "L", "I"], ["I_1", "Z", "F_1"], ["Y", "X_1"], ["X", "G", "N"], ["M", "G"]] };
      pts.F_3 = [1 - z, -z, 1 - e];
      return { pts, path: [["G", "Y", "F", "L", "I"], ["I_1", "Z", "F_1"], ["N", "G", "M"]] };
    }
    case "mclc3": case "mclc4": {
      const m = (1 + b * b / (a * a)) / 4, d = b * c * ca / (2 * a * a), z = m - 1 / 4 + (1 - b * ca / c) / (4 * sa * sa), e = h + 2 * z * c * ca / b, f = 1 + z - 2 * m, p = e - 2 * d;
      const pts = { G: [0, 0, 0], F: [1 - f, 1 - f, 1 - p], F_1: [f, f - 1, p], F_2: [1 - f, -f, 1 - p], H: [z, z, e], H_1: [1 - z, -z, 1 - e], H_2: [-z, -z, 1 - e], I: [h, -h, h], M: [h, 0, h], N: [h, 0, 0], N_1: [0, -h, 0], X: [h, -h, 0], Y: [m, m, d], Y_1: [1 - m, -m, -d], Y_2: [-m, -m, -d], Y_3: [m, m - 1, d], Z: [0, 0, h] };
      return v === "mclc3"
        ? { pts, path: [["G", "Y", "F", "H", "Z", "I", "F_1"], ["H_1", "Y_1", "X", "G", "N"], ["M", "G"]] }
        : { pts, path: [["G", "Y", "F", "H", "Z", "I"], ["H_1", "Y_1", "X", "G", "N"], ["M", "G"]] };
    }
    case "mclc5": {
      const z = (b * b / (a * a) + (1 - b * ca / c) / (sa * sa)) / 4, e = h + 2 * z * c * ca / b, m = e / 2 + b * b / (4 * a * a) - b * c * ca / (2 * a * a);
      const n = 2 * m - z, r = 1 - z * a * a / (b * b), o = (4 * n - 1 - b * b * sa * sa / (a * a)) * c / (2 * b * ca), d = z * c * ca / b + o / 2 - 1 / 4;
      return { pts: { G: [0, 0, 0], F: [n, n, o], F_1: [1 - n, 1 - n, 1 - o], F_2: [n, n - 1, o], H: [z, z, e], H_1: [1 - z, -z, 1 - e], H_2: [-z, -z, 1 - e], I: [r, 1 - r, h], I_1: [1 - r, r - 1, h], L: [h, h, h], M: [h, 0, h], N: [h, 0, 0], N_1: [0, -h, 0], X: [h, -h, 0], Y: [m, m, d], Y_1: [1 - m, -m, -d], Y_2: [-m, -m, -d], Y_3: [m, m - 1, d], Z: [0, 0, h] },
        path: [["G", "Y", "F", "L", "I"], ["I_1", "Z", "H", "F_1"], ["H_1", "Y_1", "X", "G", "N"], ["M", "G"]] };
    }
    case "tri1a": case "tri2a":
      return { pts: { G: [0, 0, 0], L: [h, h, 0], M: [0, h, h], N: [h, 0, h], R: [h, h, h], X: [h, 0, 0], Y: [0, h, 0], Z: [0, 0, h] },
        path: [["X", "G", "Y"], ["L", "G", "Z"], ["N", "G", "M"], ["R", "G"]] };
    case "tri1b": case "tri2b":
      return { pts: { G: [0, 0, 0], L: [h, -h, 0], M: [0, 0, h], N: [-h, -h, h], R: [0, -h, h], X: [0, -h, 0], Y: [h, 0, 0], Z: [-h, 0, h] },
        path: [["X", "G", "Y"], ["L", "G", "Z"], ["N", "G", "M"], ["R", "G"]] };
  }
  throw new Error("unknown lattice variant " + v);
}

/* ------------------------------------------------------------------ Bravais lattice and SC cell */

/**
 * LLL reduction (delta = 0.75) of the rows S, a port of pymatgen Lattice._calculate_lll (pymatgen 2024.8.9), so that
 * a triclinic cell gets the basis of pymatgen's SC path. Gram-Schmidt vectors g_i, coefficients u_ij = (a_i . g_j) /
 * |g_j|^2; size reduction a_k -= round(u_kj) a_j (j = k-1 .. 1); a_k and a_(k-1) swap when |g_k|^2 < (delta -
 * u_k,k-1^2) |g_(k-1)|^2. Returns new rows spanning the same lattice.
 */
function lll(S, delta = 0.75) {
  const a = S.map(v => v.slice()), g = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], u = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], m = [0, 0, 0];
  const gs = s => {
    for (let j = 0; j < s; j++) u[s][j] = dot(a[s], g[j]) / m[j];
    g[s] = a[s].map((x, c) => { let t = x; for (let j = 0; j < s; j++) t -= u[s][j] * g[j][c]; return t; });
    m[s] = dot(g[s], g[s]);
  };
  for (let s = 0; s < 3; s++) gs(s);
  let k = 2;
  for (let it = 0; k <= 3 && it < 1000; it++) {
    for (let i = k - 1; i > 0; i--) {
      // WHY: numpy rounds half to even, Math.round half up; they differ only at exact half-integers
      const q = Math.round(u[k - 1][i - 1]);
      if (q !== 0) {
        a[k - 1] = a[k - 1].map((x, c) => x - q * a[i - 1][c]);
        for (let j = 0; j < i - 1; j++) u[k - 1][j] -= q * u[i - 1][j];
        u[k - 1][i - 1] -= q;
      }
    }
    if (m[k - 1] >= (delta - u[k - 1][k - 2] ** 2) * m[k - 2]) k++;
    else {
      [a[k - 1], a[k - 2]] = [a[k - 2], a[k - 1]];
      gs(k - 2); gs(k - 1);
      if (k > 2) k--;
      else for (let r = k; r < 3; r++) for (let j = k - 2; j < k; j++) u[r][j] = dot(a[r], g[j]) / m[j];
    }
  }
  return a;
}

/** Small vector helpers: mulT(T, C) = rows of the matrix product T C; add(u, v, ...) = sum of vectors; scale(v, s) = s v;
 *  angle(u, v) = angle between u and v (rad). T_F, T_I, T_OC, T_MC: the SC centring matrices of the file header. */
const mulT = (T, C) => T.map(r => [0, 1, 2].map(j => r[0] * C[0][j] + r[1] * C[1][j] + r[2] * C[2][j]));
const add = (...vs) => [0, 1, 2].map(k => vs.reduce((s, v) => s + v[k], 0));
const scale = (v, s) => v.map(x => x * s);
const angle = (u, v) => Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (norm(u) * norm(v)))));
const T_F = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
const T_I = [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]];
const T_OC = [[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]];
const T_MC = [[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]];

/** True if the Cartesian vector v (A) is a lattice vector of the rows P, i.e. its fractional coordinates in P are
 *  integers to within 1e-4. */
function isLatticeVector(P, v) {
  return cartToFrac(inv3(P), v).every(x => Math.abs(x - Math.round(x)) < 1e-4);
}

/**
 * Shortest vector of { v0 + i e1 + j e2 : |i|, |j| <= 6 } with |cross(result, w)| != 0 when w is given and
 * |cross(result, w)| = area (to 1e-6 relative) when area is given; 2D helper for the monoclinic cells (A).
 */
function shortestIn(v0, e1, e2, w = null, area = 0) {
  let best = null, bn = Infinity;
  for (let i = -6; i <= 6; i++) for (let j = -6; j <= 6; j++) {
    const v = add(v0, scale(e1, i), scale(e2, j)), n = norm(v);
    if (n < 1e-9) continue;
    if (w && Math.abs(norm(cross(v, w)) - area) > 1e-6 * area) continue;
    if (n < bn - 1e-9 * n) { bn = n; best = v; }
  }
  return best;
}

/**
 * The SC lattice of a primitive lattice P (rows, A): { type, variant, P (SC primitive rows, A), q } with q = { a, b,
 * c, al } the SC conventional lengths (A) and alpha (rad) used by the variant formulas. tol (A) is the symmetry
 * tolerance of symmetry.js; the variant equalities use eps = tol / (shortest length). Rules in the file header.
 */
function scLattice(P, tol) {
  let conv;
  try { conv = conventional({ lattice: P, pbc: true, species: ["X"], pos: new Float64Array(3) }, tol); }
  catch (e) { conv = { system: "triclinic", z: 1, structure: { lattice: P } }; }
  const C = conv.structure.lattice, z = conv.z, sys = conv.system;
  const byLen = rows => rows.slice().sort((u, v) => norm(u) - norm(v));
  const eps = tol / Math.min(...P.map(norm));
  if (sys === "cubic") {
    const a = norm(C[0]), q = { a, b: a, c: a, al: Math.PI / 2 };
    if (z === 4) return { type: "fcc", variant: "fcc", P: mulT(T_F, C), q };
    if (z === 2) return { type: "bcc", variant: "bcc", P: mulT(T_I, C), q };
    return { type: "cub", variant: "cub", P: C, q };
  }
  if (sys === "tetragonal") {
    const a = norm(C[0]), c = norm(C[2]), q = { a, b: a, c, al: Math.PI / 2 };
    if (z === 2) return { type: "bct", variant: c < a ? "bct1" : "bct2", P: mulT(T_I, C), q };
    return { type: "tet", variant: "tet", P: C, q };
  }
  if (sys === "orthorhombic") {
    if (z === 2 && !isLatticeVector(P, scale(add(C[0], C[1], C[2]), 0.5))) {
      // base-centred: c is the axis normal to the centred face, a < b
      const k = [0, 1, 2].find(k => { const [i, j] = [0, 1, 2].filter(x => x !== k); return isLatticeVector(P, scale(add(C[i], C[j]), 0.5)); });
      const [u, v] = byLen([0, 1, 2].filter(x => x !== k).map(i => C[i]));
      const S = [u, v, C[k]], q = { a: norm(u), b: norm(v), c: norm(C[k]), al: Math.PI / 2 };
      return { type: "orcc", variant: "orcc", P: mulT(T_OC, S), q };
    }
    const S = byLen(C), [a, b, c] = S.map(norm), q = { a, b, c, al: Math.PI / 2 };
    if (z === 2) return { type: "orci", variant: "orci", P: mulT(T_I, S), q };
    if (z === 4) {
      const d = 1 / (a * a) - 1 / (b * b) - 1 / (c * c);
      // WHY: ORCF3 is the boundary 1/a^2 = 1/b^2 + 1/c^2, tested relative to 1/a^2 at the symmetry tolerance
      const v = Math.abs(d) <= 2 * eps / (a * a) ? "orcf3" : d > 0 ? "orcf1" : "orcf2";
      return { type: "orcf", variant: v, P: mulT(T_F, S), q };
    }
    return { type: "orc", variant: "orc", P: S, q };
  }
  if (sys === "hexagonal") {
    return { type: "hex", variant: "hex", P: C, q: { a: norm(C[0]), b: norm(C[0]), c: norm(C[2]), al: Math.PI / 2 } };
  }
  if (sys === "rhombohedral") {
    const [A, Bv, Cv] = C;
    // obverse setting; the reverse one has A, B -> -A, -B
    const s = isLatticeVector(P, add(scale(A, 2 / 3), scale(Bv, 1 / 3), scale(Cv, 1 / 3))) ? 1 : -1;
    const R = [[2, 1], [-1, 1], [-1, -2]].map(([i, j]) => add(scale(A, s * i / 3), scale(Bv, s * j / 3), scale(Cv, 1 / 3)));
    const al = angle(R[0], R[1]), a = norm(R[0]);
    return { type: "rhl", variant: al < Math.PI / 2 ? "rhl1" : "rhl2", P: R, q: { a, b: a, c: a, al } };
  }
  if (sys === "monoclinic") {
    const u = C[1], e1 = C[0], e2 = C[2], area = norm(cross(e1, e2));
    if (z === 2) {
      // centring (u + v)/2 with v in the plane normal to u; b = the shortest vector of v's class modulo 2 x plane
      const v0 = [e1, e2, add(e1, e2)].find(v => isLatticeVector(P, scale(add(u, v), 0.5)));
      const bv = shortestIn(v0, scale(e1, 2), scale(e2, 2));
      let cv = shortestIn([0, 0, 0], e1, e2, bv, area);
      if (dot(bv, cv) < 0) cv = scale(cv, -1);
      const S = [u, bv, cv], q = { a: norm(u), b: norm(bv), c: norm(cv), al: angle(bv, cv) };
      const Pm = mulT(T_MC, S), Br = reciprocal(Pm), kg = angle(Br[0], Br[1]);
      const cg = Math.cos(kg);
      let v;
      if (Math.abs(cg) <= eps) v = "mclc2";
      else if (cg < 0) v = "mclc1";
      else {
        const t = q.b * Math.cos(q.al) / q.c + q.b * q.b * Math.sin(q.al) ** 2 / (q.a * q.a) - 1;
        v = Math.abs(t) <= eps ? "mclc4" : t < 0 ? "mclc3" : "mclc5";
      }
      return { type: "mclc", variant: v, P: Pm, q };
    }
    let [bv, cv] = byLen([e1, e2]);
    if (dot(bv, cv) < 0) cv = scale(cv, -1);
    return { type: "mcl", variant: "mcl", P: [u, bv, cv], q: { a: norm(u), b: norm(bv), c: norm(cv), al: angle(bv, cv) } };
  }
  // triclinic: the metric of the Niggli cell (P has the successive minima a <= b <= c; type I, all cos > 0, when the
  // product of the cosines is positive, else type II, all cos <= 0), LLL-reduced as pymatgen does, then an even number
  // of sign flips so that the reciprocal angles are all obtuse or all acute
  const flips = [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]];
  const cosk = S => { const B = reciprocal(S), n = B.map(norm); return [dot(B[1], B[2]) / (n[1] * n[2]), dot(B[0], B[2]) / (n[0] * n[2]), dot(B[0], B[1]) / (n[0] * n[1])]; };
  const r = [dot(P[1], P[2]), dot(P[0], P[2]), dot(P[0], P[1])], sgn = r[0] * r[1] * r[2] > 0 ? 1 : -1;
  let N = P;
  for (const f of flips) {
    const T = P.map((v, i) => scale(v, f[i])), rr = [dot(T[1], T[2]), dot(T[0], T[2]), dot(T[0], T[1])];
    if (rr.every(x => (sgn > 0 ? x > 0 : x <= 0))) { N = T; break; }
  }
  let S = lll(N), k = cosk(S);
  const zero = k.map(x => Math.abs(x) <= eps);
  if (zero.filter(Boolean).length === 1 && !zero[2]) {
    // WHY: SC's TRI2 has k_gamma = 90 deg; cycle the axes (keeps handedness) to put the right angle there
    S = zero[0] ? [S[1], S[2], S[0]] : [S[2], S[0], S[1]];
    k = cosk(S);
  }
  // WHY: with k_gamma = 90 deg, sign flips reach both k_alpha, k_beta > 90 (TRI2a) and < 90 (TRI2b): the two are
  // settings of one lattice, and TRI2a is taken. Otherwise the sign of the product of the three cosines is invariant.
  const want = Math.abs(k[2]) <= eps ? -1 : k[0] * k[1] * k[2] < 0 ? -1 : 1;
  for (const f of flips) {
    const T = S.map((r, i) => scale(r, f[i])), kk = cosk(T);
    if (kk.every(x => Math.abs(x) <= eps || Math.sign(x) === want)) { S = T; k = kk; break; }
  }
  const two = Math.abs(k[2]) <= eps;
  const variant = (two ? "tri2" : "tri1") + (want < 0 ? "a" : "b");
  return { type: "tri", variant, P: S, q: { a: norm(S[0]), b: norm(S[1]), c: norm(S[2]), al: angle(S[1], S[2]) } };
}

/**
 * Bravais lattice, SC k-points and path of a periodic structure st = { lattice (rows, A), pos (Float64Array,
 * Cartesian A, 3N), species, pbc }; tol (A) the symmetry tolerance (as in symmetry.js). Returns null without a
 * lattice or periodicity, else { type ("cub", "fcc", "bcc", "tet", "bct", "orc", "orcf", "orci", "orcc", "hex",
 * "rhl", "mcl", "mclc", "tri"), variant ("bct1", "orcf2", "mclc3", "tri1a", ...; equal to type when there is none),
 * lattice (SC primitive rows, A, in the input's Cartesian frame), B (its reciprocal rows, 1/A), points ({ label:
 * [kx, ky, kz] } Cartesian 1/A, "G" = Gamma, each in or on wignerSeitz(B)), frac ({ label: [f1, f2, f3] }, the SC
 * table coordinates in the basis B, identical to pymatgen's), folded ({ label: [n1, n2, n3] } for the points whose SC
 * coordinates lie outside the zone: points[label] = (frac[label] - n) . B, the equivalent point nearest to Gamma;
 * empty for most lattices), path ([[labels]...], each inner list one continuous segment) }.
 * Lattice-only: see the file header for structures with lower symmetry than their lattice.
 */
export function kpath(st, tol = 1e-5) {
  if (!st || !st.pbc || !st.lattice || Math.abs(det3(st.lattice)) < 1e-12) return null;
  let P = st.lattice;
  if (st.species && st.species.length) {
    try { P = primitive(st, tol).structure.lattice; } catch (e) { P = st.lattice; }
  }
  P = reduce(P);
  if (det3(P) < 0) P = P.map(v => scale(v, -1));
  const sc = scLattice(P, tol);
  const B = reciprocal(sc.P), t = table(sc.variant, sc.q), points = {}, frac = {}, folded = {};
  const zone = wignerSeitz(B), R = reduce(B), Ri = inv3(R), Bi = inv3(B);
  for (const [lab, f] of Object.entries(t.pts)) {
    frac[lab] = f;
    const k = fracToCart(B, f);
    points[lab] = k;
    if (inZone(zone, k)) continue;
    // WHY: some SC points (e.g. MCL D_1, TRI1b N) lie outside the Wigner-Seitz zone; show the equivalent k - G
    // nearest to Gamma, G the reciprocal lattice vector closest to k
    const r = cartToFrac(Ri, k).map(Math.round);
    let best = k, bn = norm(k);
    for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) for (let l = -2; l <= 2; l++) {
      const G = fracToCart(R, [r[0] + i, r[1] + j, r[2] + l]), d = [k[0] - G[0], k[1] - G[1], k[2] - G[2]];
      if (norm(d) < bn - 1e-9 * bn) { bn = norm(d); best = d; }
    }
    points[lab] = best;
    folded[lab] = cartToFrac(Bi, [k[0] - best[0], k[1] - best[1], k[2] - best[2]]).map(Math.round);
  }
  return { type: sc.type, variant: sc.variant, lattice: sc.P, B, points, frac, folded, path: t.path };
}
