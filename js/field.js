/**
 * Scalar-field utilities: statistics for isovalue choice, and the resampled grid handed to the isosurface renderer.
 *
 * Integral: periodic grid  int f dV ~ sum_all f * V_span / (nx ny nz);
 *           closed grid    (general grid whose last plane repeats the first) sum over i < n - 1 per axis * V_span /
 *                          ((nx-1)(ny-1)(nz-1));  open grid  sum_all f * V_span / ((nx-1)(ny-1)(nz-1)).
 * Enclosed fraction: p(iso) = sum_{|f| > iso} |f| / sum |f|, estimated from a uniform sample of at most 2^20 grid
 * values (sorted |f| with suffix sums), so the isovalue that encloses a share p of the charge is a table lookup.
 * Render grid: voxel (I, J, K) takes the value at source index (I s - k_x, J s - k_y, K s - k_z) (stride s), wrapped
 * periodically for periodic grids, which are extended by one plane per axis (and tiled over the supercell repetition
 * r) so the surface closes at the far faces. Closed general grids (XSF over the cell) are periodic with n - 1 points.
 * Voxel vectors span[a] * s / d_a with d = n (periodic) or n - 1 (general grid).
 * Origin shift by fractions (f_x, f_y, f_z) of the cell: periodic grids roll by k_a = floor(f_a n_a) points and move the
 * origin by the remainder (f_a n_a - k_a) grid steps, so the field moves exactly with the atoms (no interpolation);
 * other grids move their origin by sum_a f_a span_a.
 * Cuts: a grid point p is set to NaN (no surface, gfx/mc.js skips the cell) when, for any cut { m, poly } (or the
 * rectangle { m, r }), the clip position c = m (p, 1) has c_w > 0 and c_xy / c_w inside the polygon poly = [x0, y0,
 * x1, y1, ...] (normalised device coordinates of the view that made the cut, m = proj view in the WebGL convention).
 * The polygon is rasterised once on a 384 x 384 grid over its bounding box (even-odd rule), so each point costs one
 * lookup.
 * Output order: x index fastest (I + mx (J + my K)), as read by the marching cubes (gfx/mc.js).
 * Refinement (refine): coarse grids are resampled onto m_a times finer grids, so isosurfaces are smooth. Periodic grids
 * use Fourier interpolation, exact for band-limited data such as plane-wave densities and potentials: along each axis
 * the line is transformed, its spectrum padded with zeros from n to m n points (an even n's Nyquist term split in half
 * between +n/2 and -n/2) and transformed back, which keeps the original points and the mean (so the integral). General
 * grids use cubic Catmull-Rom interpolation along each axis, (n - 1) m + 1 points.
 */
import { det3 } from "./lattice.js?v=23";
import { fft1 } from "./fft.js?v=23";

/** Field statistics: min, max, maxAbs, integral, the sorted |f| sample and its suffix sums. */
export function stats(f) {
  const d = f.data, [nx, ny, nz] = f.n;
  let mn = Infinity, mx = -Infinity, sum = 0;
  if (f.closed) {
    for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) sum += d[i + nx * (j + ny * k)];
  } else for (let i = 0; i < d.length; i++) sum += d[i];
  for (let i = 0; i < d.length; i++) { const v = d[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
  const V = Math.abs(det3(f.span));
  const cells = f.periodic ? nx * ny * nz : (nx - 1) * (ny - 1) * (nz - 1);
  const m = Math.min(d.length, 1 << 20), step = d.length / m, s = new Float32Array(m);
  for (let i = 0; i < m; i++) s[i] = Math.abs(d[Math.floor(i * step)]);
  s.sort();
  const cum = new Float64Array(m + 1);
  for (let i = m - 1; i >= 0; i--) cum[i] = cum[i + 1] + s[i];
  return { min: mn, max: mx, maxAbs: Math.max(Math.abs(mn), Math.abs(mx)), integral: sum * V / cells, sorted: s, cum };
}

/** Share of sum |f| in voxels with |f| > iso. */
export function fractionAbove(st, iso) {
  const s = st.sorted;
  let lo = 0, hi = s.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (s[mid] <= iso) lo = mid + 1; else hi = mid; }
  return st.cum[0] ? st.cum[lo] / st.cum[0] : 0;
}

/** Isovalue whose superlevel set {|f| > iso} holds the share p of sum |f|. */
export function isoForFraction(st, p) {
  const s = st.sorted, c = st.cum, tot = c[0];
  let lo = 0, hi = s.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (c[mid] / tot > p) lo = mid + 1; else hi = mid; }
  return s[Math.max(0, lo - 1)] || s[s.length - 1];
}

/** Voxel counts per axis for stride s and repetition r. */
const periodicN = f => (f.periodic ? f.n : f.closed ? f.n.map(n => n - 1) : null);
function counts(f, r, s) {
  const pn = periodicN(f);
  return pn ? pn.map((n, a) => Math.floor(n * r[a] / s) + 1) : f.n.map(n => Math.floor((n - 1) / s) + 1);
}

/** Smallest stride whose render grid holds at most `budget` voxels. */
export function autoStride(f, r, budget) {
  for (let s = 1; s < 16; s++) { const m = counts(f, r, s); if (m[0] * m[1] * m[2] <= budget) return s; }
  return 16;
}

/** Resampled grid { data (x fastest), size [mx, my, mz], vox [3 vectors], origin } for the marching cubes. */
export function renderGrid(f, reps = [1, 1, 1], s = 1, shift = [0, 0, 0], cuts = []) {
  const pn = periodicN(f), r = pn ? reps : [1, 1, 1];
  const [nx, ny] = f.n, m = counts(f, r, s), d = f.data;
  const den = pn || f.n.map(n => n - 1);
  const step = f.span.map((v, a) => v.map(c => c / den[a]));
  const vox = step.map(v => v.map(c => c * s));
  const origin = f.origin.slice(), k = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    if (pn) { const t = shift[a] * pn[a]; k[a] = Math.floor(t); for (let c = 0; c < 3; c++) origin[c] += (t - k[a]) * step[a][c]; }
    else for (let c = 0; c < 3; c++) origin[c] += shift[a] * f.span[a][c];
  }
  const wrap = (v, n) => ((v % n) + n) % n;
  const idx = (a, I) => (pn ? wrap(I * s - k[a], pn[a]) : I * s);
  const out = new Float32Array(m[0] * m[1] * m[2]);
  const ix = new Int32Array(m[0]);
  for (let I = 0; I < m[0]; I++) ix[I] = idx(0, I);
  let p = 0;
  for (let K = 0; K < m[2]; K++) {
    const kz = idx(2, K) * nx * ny;
    for (let J = 0; J < m[1]; J++) {
      const base = kz + nx * idx(1, J);
      for (let I = 0; I < m[0]; I++) out[p++] = d[base + ix[I]];
    }
  }
  if (cuts.length) maskCuts(out, m, vox, origin, cuts);
  return { data: out, size: m, vox, origin };
}

/** Even-odd point-in-polygon test; P = [x0, y0, x1, y1, ...]. */
export function insidePolygon(P, x, y) {
  let inside = false;
  for (let i = 0, j = P.length - 2; i < P.length; j = i, i += 2) {
    const xi = P[i], yi = P[i + 1], xj = P[j], yj = P[j + 1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
/** A cut as { M, inside(X, Y) } in normalised device coordinates. */
function prepCut(c) {
  if (!c.poly) return { M: c.m, inside: (X, Y) => X >= c.r[0] && X <= c.r[2] && Y >= c.r[1] && Y <= c.r[3] };
  const P = c.poly, R = 384;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < P.length; i += 2) { x0 = Math.min(x0, P[i]); x1 = Math.max(x1, P[i]); y0 = Math.min(y0, P[i + 1]); y1 = Math.max(y1, P[i + 1]); }
  const gx = (x1 - x0) / R || 1, gy = (y1 - y0) / R || 1, ras = new Uint8Array(R * R);
  for (let iy = 0; iy < R; iy++) for (let ix = 0; ix < R; ix++) ras[iy * R + ix] = insidePolygon(P, x0 + (ix + 0.5) * gx, y0 + (iy + 0.5) * gy) ? 1 : 0;
  return { M: c.m, inside: (X, Y) => X >= x0 && X <= x1 && Y >= y0 && Y <= y1 && ras[Math.min(R - 1, Math.floor((Y - y0) / gy)) * R + Math.min(R - 1, Math.floor((X - x0) / gx))] === 1 };
}
/** Set grid points inside any cut region to NaN (see the file docstring). */
function maskCuts(out, m, vox, o, cutList) {
  const cuts = cutList.map(prepCut);
  let p = 0;
  for (let K = 0; K < m[2]; K++) for (let J = 0; J < m[1]; J++) {
    const b = [0, 1, 2].map(c => o[c] + J * vox[1][c] + K * vox[2][c]);
    for (let I = 0; I < m[0]; I++, p++) {
      const x = b[0] + I * vox[0][0], y = b[1] + I * vox[0][1], z = b[2] + I * vox[0][2];
      for (const { M, inside } of cuts) {
        const w = M[3] * x + M[7] * y + M[11] * z + M[15];
        if (w <= 0) continue;
        if (inside((M[0] * x + M[4] * y + M[8] * z + M[12]) / w, (M[1] * x + M[5] * y + M[9] * z + M[13]) / w)) { out[p] = NaN; break; }
      }
    }
  }
}

/**
 * A finer copy of a coarse field (see the file header): the factor m_a per axis is the smallest integer giving a grid
 * step <= h (A), reduced on the axis with the largest factor until the grid has at most maxPts points. Returns the
 * field itself when no axis needs refining; otherwise a new field with data, n, n0 (the grid as read) and refined = [m_a].
 */
export function refine(f, h = 0.1, maxPts = 4e6) {
  const per = f.periodic && !f.closed, n = f.n;
  const len = v => Math.hypot(v[0], v[1], v[2]);
  const m = n.map((na, a) => (na < 2 ? 1 : Math.max(1, Math.ceil(len(f.span[a]) / (per ? na : na - 1) / h - 1e-9))));
  const size = mm => n.reduce((acc, na, a) => acc * (per ? na * mm[a] : (na - 1) * mm[a] + 1), 1);
  while (size(m) > maxPts && m.some(x => x > 1)) m[m.indexOf(Math.max(...m))]--;
  if (m.every(x => x === 1)) return f;
  let data = f.data, cur = n.slice();
  for (let a = 0; a < 3; a++) {
    if (m[a] === 1) continue;
    const r = per ? fourierAxis(data, cur, a, m[a]) : cubicAxis(data, cur, a, m[a]);
    data = r.data; cur = r.n;
  }
  return { ...f, data, n: cur, n0: n.slice(), refined: m };
}

/** Lines along axis a of a grid of shape n (x fastest): fn(base, p, q) with base the index of the line's first point
 *  and p, q its indices along the other two axes (in increasing axis order). */
function eachLine(n, a, fn) {
  const st = [1, n[0], n[0] * n[1]], o = [0, 1, 2].filter(b => b !== a);
  for (let q = 0; q < n[o[1]]; q++) for (let p = 0; p < n[o[0]]; p++) fn(p * st[o[0]] + q * st[o[1]], p, q);
}

/** Fourier interpolation of every line along axis a of a periodic grid from n_a to m n_a points. */
function fourierAxis(data, n, a, m) {
  const na = n[a], M = na * m, out = n.slice(); out[a] = M;
  const res = new Float32Array(data.length / na * M), re = new Float64Array(na), im = new Float64Array(na);
  const R = new Float64Array(M), I = new Float64Array(M), half = Math.floor(na / 2);
  const ost = [1, out[0], out[0] * out[1]], o = [0, 1, 2].filter(b => b !== a);
  const stride = [1, n[0], n[0] * n[1]][a];
  eachLine(n, a, (base, p, q) => {
    for (let k = 0; k < na; k++) { re[k] = data[base + k * stride]; im[k] = 0; }
    fft1(re, im, -1);
    R.fill(0); I.fill(0);
    for (let k = 0; k <= half; k++) { R[k] = re[k]; I[k] = im[k]; }
    for (let k = half + 1; k < na; k++) { R[M - na + k] = re[k]; I[M - na + k] = im[k]; }
    if (na % 2 === 0) { R[half] *= 0.5; I[half] *= 0.5; R[M - half] = R[half]; I[M - half] = I[half]; }
    fft1(R, I, 1);
    const ob = p * ost[o[0]] + q * ost[o[1]];
    for (let l = 0; l < M; l++) res[ob + l * ost[a]] = R[l] / na;
  });
  return { data: res, n: out };
}

/** Catmull-Rom interpolation of every line along axis a of a general grid from n_a to (n_a - 1) m + 1 points. */
function cubicAxis(data, n, a, m) {
  const na = n[a], M = (na - 1) * m + 1, out = n.slice(); out[a] = M;
  const res = new Float32Array(data.length / na * M), x = new Float64Array(na);
  const ost = [1, out[0], out[0] * out[1]], o = [0, 1, 2].filter(b => b !== a);
  const stride = [1, n[0], n[0] * n[1]][a];
  eachLine(n, a, (base, p, q) => {
    for (let k = 0; k < na; k++) x[k] = data[base + k * stride];
    const ob = p * ost[o[0]] + q * ost[o[1]];
    for (let l = 0; l < M; l++) {
      const i = Math.min(Math.floor(l / m), na - 2), u = l / m - i;
      const p0 = x[Math.max(i - 1, 0)], p1 = x[i], p2 = x[i + 1], p3 = x[Math.min(i + 2, na - 1)];
      res[ob + l * ost[a]] = 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (3 * p1 - p0 - 3 * p2 + p3) * u * u * u);
    }
  });
  return { data: res, n: out };
}
