/**
 * Scalar-field utilities: statistics for isovalue choice, and the resampled grid handed to the isosurface renderer.
 *
 * Integral: periodic grid  int f dV ~ sum_all f * V_span / (nx ny nz);
 *           closed grid    (general grid whose last plane repeats the first) sum over i < n - 1 per axis * V_span /
 *                          ((nx-1)(ny-1)(nz-1));  open grid  sum_all f * V_span / ((nx-1)(ny-1)(nz-1)).
 * Enclosed fraction: p(iso) = sum_{|f| > iso} |f| / sum |f|, estimated from a uniform sample of at most 2^20 grid
 * values (sorted |f| with suffix sums), so the isovalue that encloses a share p of the charge is a table lookup.
 * Render grid: voxel (I, J, K) takes the value at source index (I s, J s, K s) (stride s), wrapped periodically for
 * periodic grids, which are extended by one plane per axis (and tiled over the supercell repetition r) so the
 * surface closes at the far faces. Voxel vectors span[a] * s / d_a with d = n (periodic) or n - 1 (general grid).
 * Output order is x outermost, z innermost (the order the renderer indexes).
 */
import { det3 } from "./lattice.js";

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
function counts(f, r, s) {
  return f.periodic ? f.n.map((n, a) => Math.floor(n * r[a] / s) + 1) : f.n.map(n => Math.floor((n - 1) / s) + 1);
}

/** Smallest stride whose render grid holds at most `budget` voxels. */
export function autoStride(f, r, budget) {
  for (let s = 1; s < 16; s++) { const m = counts(f, r, s); if (m[0] * m[1] * m[2] <= budget) return s; }
  return 16;
}

/** Resampled grid { data (x outer, z inner), size [mx, my, mz], vox [3 vectors], origin } for the renderer. */
export function renderGrid(f, reps = [1, 1, 1], s = 1) {
  const r = f.periodic ? reps : [1, 1, 1];
  const [nx, ny, nz] = f.n, m = counts(f, r, s), d = f.data;
  const den = f.periodic ? f.n : f.n.map(n => n - 1);
  const vox = f.span.map((v, a) => v.map(c => c * s / den[a]));
  const out = new Float32Array(m[0] * m[1] * m[2]);
  const iz = new Int32Array(m[2]);
  for (let K = 0; K < m[2]; K++) iz[K] = ((K * s) % nz) * nx * ny;
  let p = 0;
  for (let I = 0; I < m[0]; I++) {
    const ix = (I * s) % nx;
    for (let J = 0; J < m[1]; J++) {
      const base = ix + nx * ((J * s) % ny);
      for (let K = 0; K < m[2]; K++) out[p++] = d[base + iz[K]];
    }
  }
  return { data: out, size: m, vox, origin: f.origin.slice() };
}
