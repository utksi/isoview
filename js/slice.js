// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * 2D slices through a scalar field on a grid: the plane, the sampled image, contour lines and colour maps.
 *
 * Plane (hkl): cell rows A = (a1, a2, a3) (Angstrom) with origin O. Reciprocal rows (without 2 pi) B = (A^-1)^T, so
 * a_i . b_j = delta_ij; the plane normal is n = G / |G| with G = h b1 + k b2 + l b3, i.e. the lattice planes (hkl).
 * The position t in [0, 1] runs over the cell: d = d_min + t (d_max - d_min), with d_min, d_max the least and greatest
 * of c . n over the 8 corners c of the cell; the plane is { p : p . n = d }. For (001) this is the plane at fractional
 * coordinate t along a3. The cross-section is the convex polygon where the plane cuts the 12 cell edges. In-plane
 * axes: u = the cell vector most nearly in the plane, projected onto it and normalised; v = n x u. The image is the
 * rectangle o + s U + r V (s, r in [0, 1]) bounding the polygon, U = (u_max - u_min) u, V = (v_max - v_min) v.
 *
 * Sampling: pixel (i, j) of a W x H image is the point p = o + (i + 1/2) / W U + (j + 1/2) / H V (row j = 0 along
 * V = 0). Its fractional coordinates in the field cell are x = (p - origin_f) S^-1 (S = rows span_a), minus the origin
 * shift; grid coordinates g_a = x_a N_a with N_a = n_a (periodic, wrapped), n_a - 1 (closed general grid, wrapped) or
 * n_a - 1 (open general grid, NaN outside).
 * The value is the tricubic Catmull-Rom interpolation of the 4 x 4 x 4 nearest grid values (exact at grid points,
 * C1 continuous), with weights for fraction s in [0, 1):
 *   w_-1 = (-s^3 + 2 s^2 - s) / 2, w_0 = (3 s^3 - 5 s^2 + 2) / 2, w_1 = (-3 s^3 + 4 s^2 + s) / 2, w_2 = (s^3 - s^2) / 2.
 * Fields in isoview are already refined to a step of at most 0.1 A by Fourier interpolation (field.js), so the cubic
 * step only fills in between points that are themselves exact for plane-wave data. Pixels outside the cross-section
 * polygon are NaN (not drawn).
 *
 * Contours: marching squares on the pixel centres at a level c, edges cut where the linear interpolant equals c;
 * the two saddle cases are split by the mean of the four corners. Segments in pixel units (x right, y along V).
 * Colour maps: viridis (polynomial fit of the matplotlib map, as published by M. Zucker, 2019, under CC0) and
 * ColorBrewer RdBu (11 classes, reversed: negative blue, positive red), linear in between, and grey.
 */
import { inv3, cartToFrac, dot, cross, norm } from "./lattice.js?v=31";

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const unit = a => mul(a, 1 / norm(a));

/**
 * The slice plane of the cell A (3 rows, A) with origin O for Miller indices hkl (integers, not all zero) at
 * position t in [0, 1] (see the file header). Returns { n (unit normal), d (A, p . n of the plane), dRange: [d_min,
 * d_max], depth (A, d_max - d_min), u, v (unit in-plane axes), o, U, V (image rectangle: corner and edge vectors, A),
 * poly: [[s, r], ...] (the cross-section, fractions of U and V, counter-clockwise) }.
 */
export function slicePlane(A, O, hkl, t) {
  const Bt = inv3(A); // columns of A^-1 are the reciprocal vectors b_j (without 2 pi)
  const b = [0, 1, 2].map(j => [Bt[0][j], Bt[1][j], Bt[2][j]]);
  const G = add(add(mul(b[0], hkl[0]), mul(b[1], hkl[1])), mul(b[2], hkl[2]));
  if (norm(G) < 1e-12) throw new Error("the Miller indices (0 0 0) give no plane");
  const n = unit(G);
  const corners = [];
  for (let i = 0; i < 8; i++) corners.push(add(O, add(add(mul(A[0], i & 1), mul(A[1], (i >> 1) & 1)), mul(A[2], (i >> 2) & 1))));
  const ds = corners.map(c => dot(c, n)), dmin = Math.min(...ds), dmax = Math.max(...ds);
  const d = dmin + Math.min(1, Math.max(0, t)) * (dmax - dmin);
  // in-plane axis u: the cell vector most nearly in the plane
  let best = 0, bq = Infinity;
  for (let a = 0; a < 3; a++) { const q = Math.abs(dot(A[a], n)) / norm(A[a]); if (q < bq - 1e-9) { bq = q; best = a; } }
  const u = unit(sub(A[best], mul(n, dot(A[best], n)))), v = cross(n, u);
  // cross-section: the plane against the 12 edges (corner i to corner i | bit)
  const pts = [], scale = Math.max(...A.map(norm)), eps = 1e-9 * scale;
  const push = p => { if (!pts.some(q => norm(sub(p, q)) < 1e-7 * scale)) pts.push(p); };
  for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) {
    if (i & bit) continue;
    const p0 = corners[i], p1 = corners[i | bit], d0 = ds[i] - d, d1 = ds[i | bit] - d;
    if (Math.abs(d0) <= eps) push(p0);
    if (Math.abs(d1) <= eps) push(p1);
    if ((d0 < -eps && d1 > eps) || (d0 > eps && d1 < -eps)) push(add(p0, mul(sub(p1, p0), d0 / (d0 - d1))));
  }
  const uv = pts.map(p => [dot(p, u), dot(p, v)]);
  const umin = Math.min(...uv.map(q => q[0])), umax = Math.max(...uv.map(q => q[0]));
  const vmin = Math.min(...uv.map(q => q[1])), vmax = Math.max(...uv.map(q => q[1]));
  const cu = uv.reduce((s, q) => s + q[0], 0) / uv.length, cv = uv.reduce((s, q) => s + q[1], 0) / uv.length;
  uv.sort((p, q) => Math.atan2(p[1] - cv, p[0] - cu) - Math.atan2(q[1] - cv, q[0] - cu));
  const du = Math.max(umax - umin, 1e-9), dv = Math.max(vmax - vmin, 1e-9);
  const o = add(add(mul(n, d), mul(u, umin)), mul(v, vmin));
  return { n, d, dRange: [dmin, dmax], depth: dmax - dmin, u, v, o, U: mul(u, du), V: mul(v, dv), width: du, height: dv,
    poly: uv.map(q => [(q[0] - umin) / du, (q[1] - vmin) / dv]) };
}

/** Catmull-Rom weights for fraction s, written into w[0..3] (taps -1, 0, 1, 2). */
function crw(s, w) {
  const s2 = s * s, s3 = s2 * s;
  w[0] = 0.5 * (-s3 + 2 * s2 - s); w[1] = 0.5 * (3 * s3 - 5 * s2 + 2); w[2] = 0.5 * (-3 * s3 + 4 * s2 + s); w[3] = 0.5 * (s3 - s2);
}

/** True when the point (s, r) (fractions of U, V) is inside the convex polygon poly (counter-clockwise). */
function inConvex(poly, s, r) {
  for (let i = 0, m = poly.length; i < m; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % m];
    if ((x1 - x0) * (r - y0) - (y1 - y0) * (s - x0) < -1e-12) return false;
  }
  return true;
}

/**
 * Sample field f ({ data (x fastest), n, span, origin, periodic, closed }) on plane P (slicePlane) as a W x H image.
 * shift: origin shift in fractions of the field cell (as field.js renderGrid). Returns { values: Float32Array(W H)
 * (row j = 0 along V = 0; NaN outside the cross-section or a general grid), min, max } (min, max over drawn pixels).
 */
export function sampleSlice(f, P, W, H, shift = [0, 0, 0]) {
  // periodic grids wrap over n points; closed general grids (last plane repeats the first) wrap over n - 1; open
  // general grids span n - 1 steps and are not continued outside (as field.js)
  const per = !!(f.periodic || f.closed), [nx, ny, nz] = f.n;
  const N = f.periodic ? [nx, ny, nz] : [nx - 1, ny - 1, nz - 1];
  const Si = inv3(f.span), d = f.data, out = new Float32Array(W * H);
  const wx = new Float64Array(4), wy = new Float64Array(4), wz = new Float64Array(4);
  const ix = new Int32Array(4), iy = new Int32Array(4), iz = new Int32Array(4);
  const idx = (i, a) => (per ? ((i % N[a]) + N[a]) % N[a] : Math.min(Math.max(i, 0), N[a]));
  let mn = Infinity, mx = -Infinity;
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const s = (i + 0.5) / W, r = (j + 0.5) / H;
    if (!inConvex(P.poly, s, r)) { out[i + W * j] = NaN; continue; }
    const p = add(add(P.o, mul(P.U, s)), mul(P.V, r));
    const x = cartToFrac(Si, sub(p, f.origin));
    const g = [0, 1, 2].map(a => (x[a] - shift[a]) * N[a]);
    if (!per && g.some((q, a) => q < -1e-6 || q > N[a] + 1e-6)) { out[i + W * j] = NaN; continue; }
    const g0 = g.map(Math.floor);
    crw(g[0] - g0[0], wx); crw(g[1] - g0[1], wy); crw(g[2] - g0[2], wz);
    for (let q = 0; q < 4; q++) { ix[q] = idx(g0[0] + q - 1, 0); iy[q] = idx(g0[1] + q - 1, 1); iz[q] = idx(g0[2] + q - 1, 2); }
    let v = 0;
    for (let c = 0; c < 4; c++) {
      const zc = iz[c] * nx * ny;
      for (let b = 0; b < 4; b++) {
        const yb = zc + iy[b] * nx, wb = wz[c] * wy[b];
        v += wb * (wx[0] * d[yb + ix[0]] + wx[1] * d[yb + ix[1]] + wx[2] * d[yb + ix[2]] + wx[3] * d[yb + ix[3]]);
      }
    }
    out[i + W * j] = v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return { values: out, min: mn, max: mx };
}

/**
 * Contour segments of a W x H image (NaN cells skipped) at level c by marching squares on the pixel centres.
 * Returns a Float32Array [x0, y0, x1, y1, ...] in pixel units (pixel i's centre at x = i + 1/2).
 */
export function contours(values, W, H, c) {
  const out = [];
  const at = (i, j) => values[i + W * j];
  const lerp = (a, b) => (c - a) / (b - a);
  for (let j = 0; j < H - 1; j++) for (let i = 0; i < W - 1; i++) {
    const a = at(i, j), b = at(i + 1, j), e = at(i + 1, j + 1), g = at(i, j + 1);
    if (Number.isNaN(a + b + e + g)) continue;
    const k = (a > c) | ((b > c) << 1) | ((e > c) << 2) | ((g > c) << 3);
    if (k === 0 || k === 15) continue;
    const x = i + 0.5, y = j + 0.5;
    const B = [x + lerp(a, b), y], R = [x + 1, y + lerp(b, e)], T = [x + lerp(g, e), y + 1], L = [x, y + lerp(a, g)];
    const seg = (p, q) => out.push(p[0], p[1], q[0], q[1]);
    switch (k) {
      case 1: case 14: seg(L, B); break;
      case 2: case 13: seg(B, R); break;
      case 3: case 12: seg(L, R); break;
      case 4: case 11: seg(R, T); break;
      case 6: case 9: seg(B, T); break;
      case 7: case 8: seg(L, T); break;
      case 5: case 10: {
        // saddle: the mean of the corners decides which pairs of edges join
        const up = (a + b + e + g) / 4 > c;
        if ((k === 5) === up) { seg(L, T); seg(B, R); } else { seg(L, B); seg(R, T); }
        break;
      }
    }
  }
  return new Float32Array(out);
}

const VIRIDIS = [[0.2777273272234177, 0.005407344544966578, 0.3340998053353061], [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659], [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105], [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124352495832]];
const RDBU = [[5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240], [247, 247, 247], [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31]];

/** Colour of map name ("viridis", "rdbu", "grey") at x in [0, 1], as [r, g, b] in 0..255. */
export function colormap(name, x) {
  x = Math.min(1, Math.max(0, x));
  if (name === "grey") { const g = Math.round(20 + 225 * x); return [g, g, g]; }
  if (name === "rdbu") {
    const s = x * (RDBU.length - 1), i = Math.min(RDBU.length - 2, Math.floor(s)), w = s - i;
    return [0, 1, 2].map(k => Math.round(RDBU[i][k] * (1 - w) + RDBU[i + 1][k] * w));
  }
  return [0, 1, 2].map(k => { let v = 0; for (let p = VIRIDIS.length - 1; p >= 0; p--) v = v * x + VIRIDIS[p][k]; return Math.round(255 * Math.min(1, Math.max(0, v))); });
}

/** 256-entry lookup table (Uint8Array(768), rgb) of a colour map. */
export function lut(name) {
  const t = new Uint8Array(768);
  for (let i = 0; i < 256; i++) t.set(colormap(name, i / 255), 3 * i);
  return t;
}

/**
 * Map values to [0, 1] for colouring: linear over [lo, hi], or log10 over [lo, hi] (lo > 0; values below lo map to 0),
 * or symmetric about zero over [-m, m] (m = max(|lo|, |hi|)) for "signed". Returns a function value -> x (NaN stays NaN).
 */
export function scaler(mode, lo, hi) {
  if (mode === "log") { const a = Math.log10(lo), b = Math.log10(hi); return v => (v <= lo ? 0 : (Math.log10(v) - a) / (b - a)); }
  if (mode === "signed") { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return v => 0.5 + 0.5 * v / m; }
  return v => (v - lo) / ((hi - lo) || 1);
}

/** RGBA image (Uint8ClampedArray(4 W H)) of values with map name and scale function sc; NaN pixels are transparent. */
export function paint(values, W, H, name, sc) {
  const t = lut(name), px = new Uint8ClampedArray(4 * W * H);
  for (let p = 0; p < W * H; p++) {
    const v = values[p];
    if (Number.isNaN(v)) continue;
    const k = 3 * Math.round(255 * Math.min(1, Math.max(0, sc(v))));
    px[4 * p] = t[k]; px[4 * p + 1] = t[k + 1]; px[4 * p + 2] = t[k + 2]; px[4 * p + 3] = 255;
  }
  return px;
}
