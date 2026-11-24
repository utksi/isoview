// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Trajectory analysis: radial distribution functions, mean-squared displacement and self-diffusion, velocity
 * autocorrelation and vibrational density of states, and per-frame series (energy, volume, density, cell lengths,
 * kinetic temperature estimate). Pure numerics on the trajectory data model of traj.js; no DOM, runs in a worker.
 *
 * Data: structure = { lattice: 3 rows (A) | null, pbc: boolean, species: [symbol], pos }; traj = { n, nat,
 * pos: Float64Array(3 nat n) Cartesian A (atom i of frame f at 3 (nat f + i)), cells: Float64Array(9 n) | null,
 * energy: Float64Array(n) | null (eV) }. The cell of frame f is traj.cells[9f .. 9f+8] (rows a, b, c) when present,
 * else structure.lattice. Units: length A, time fs (input dt_fs) and ps (output), mass amu, energy eV.
 *
 * Notation: N atoms, N_a atoms of element a, V cell volume (A^3) = |det L|, L rows a, b, c (A), dt time between
 * consecutive frames (fs), m_i mass of atom i (amu), r_i(t) position (A), u_i(t) unwrapped position (A).
 *
 * 1. Unwrapping (periodic cells) by image flags. With s_i(f) = r_i(f) L_f^{-1} the fractional position (row vector)
 *    in the cell L_f of frame f,
 *      n_i(f) = n_i(f-1) - round(s_i(f) - s_i(f-1)),  n_i(0) = 0,  u_i(f) = r_i(f) + n_i(f) L_f.
 *    Exact for fixed and variable cells, whether or not the file wraps positions, when no atom moves more than half a
 *    cell width between saved frames. (Adding minimum-image Cartesian steps instead is wrong for variable cells with
 *    wrapped positions: it leaves an error n (L_f - L_{f-1}).) Convention: each atom follows the image it has in
 *    frame 0 of the file; for a deforming cell a different starting image would add n (L_f - L_0) to its path.
 *
 * 2. Radial distribution function. For elements a, b and a bin [r, r + dr):
 *      g_ab(r) = < V / (N_a N_b) * h_ab(r) / dV(r) >_frames,   dV(r) = (4/3) pi ((r + dr)^3 - r^3)  (A^3),
 *    h_ab(r) = number of ordered pairs (i in a, j in b, j != i or j a periodic image of i) with |r_j + T - r_i| in
 *    the bin, over all lattice translations T (all images, not only the nearest). Total g(r) uses N^2 and all pairs.
 *    Running coordination n_ab(R) = < sum_{bins below R} h_ab > / N_a = number of b atoms within R of an a atom.
 *    An ideal gas gives g_ab -> 1 (g_aa -> (N_a - 1) / N_a).
 *
 * 3. Mean-squared displacement, averaged over time origins t0 and the atoms of an element:
 *      MSD_a(tau) = (1 / N_a) sum_{i in a} < |u_i(t0 + tau) - u_i(t0)|^2 >_{t0}   (A^2),
 *    computed as S1 - 2 S2 with S2(m) = (1 / (K - m)) sum_k u_k . u_{k+m} by FFT (zero-padded to >= 2K) and
 *    S1(m) = (1 / (K - m)) sum_{k=0}^{K-m-1} (|u_k|^2 + |u_{k+m}|^2), K samples (Calandrini et al., Collection SFN 12,
 *    201 (2011)). Einstein relation in 3D: D = lim (1 / 6) d MSD / d tau, fitted by least squares on a lag window.
 *    1 A^2/ps = 1e-4 cm^2/s.
 *
 * 4. Velocity autocorrelation. v_i(t) = (u_i(t + dt) - u_i(t - dt)) / (2 dt) (A/fs, central difference),
 *      C_a(tau) = (1 / N_a) sum_{i in a} < v_i(t0) . v_i(t0 + tau) >_{t0},   Z_a(tau) = C_a(tau) / C_a(0),
 *      Z_m(tau) = sum_a m_a N_a C_a(tau) / sum_a m_a N_a C_a(0)   (mass-weighted total).
 *    Vibrational density of states (VDOS), one-sided cosine transform with a Hann window w over lags 0..tau_max:
 *      G(nu) = 2 int_0^tau_max Z_m(tau) w(tau) cos(2 pi nu tau) dtau,  w(tau) = (1 + cos(pi tau / tau_max)) / 2,
 *    normalised to int_0^nu_N G dnu = 1; nu in THz (1/ps), 1 THz = 33.35640952 cm^-1, nu_N = 1 / (2 dt) the Nyquist
 *    frequency. The central difference multiplies the velocity amplitude at frequency nu by sin(2 pi nu dt) /
 *    (2 pi nu dt), so G is damped toward nu_N (to 64% in power at nu_N / 2) and vanishes at nu_N.
 *
 * 5. Kinetic temperature estimate, T = 2 KE / (n_dof k_B), KE = (1/2) sum_i m_i |v_i - v_com|^2, n_dof = 3N - 3 (the
 *    centre-of-mass motion removed), k_B = 8.617333262e-5 eV/K, 1 amu A^2/fs^2 = 103.6426965 eV.
 *    Density rho = sum_i m_i / V; 1 amu/A^3 = 1.66053906660 g/cm^3.
 *
 * Time step: dt must come from the user or from the file (VASP vasprun.xml has POTIM; XDATCAR, extended XYZ and
 * most dumps do not, and when frames are saved every k MD steps dt is k times the MD step). The readers do not set
 * it; timeStepOf returns traj.dt_fs if a reader ever provides it, else null, and the UI asks.
 *
 * Masses: IUPAC 2013/2016 standard atomic weights (conventional values; the mass number of the longest-lived
 * isotope for elements without a standard weight), as tabulated in ase.data.atomic_masses_iupac2016 (ASE 3.26);
 * unknown "X" = 1 amu.
 */
import { fft1, nextPow2 } from "./fft.js?v=28";
import { zOf } from "./elements.js?v=28";
import { det3, inv3, params } from "./lattice.js?v=28";

const KB = 8.617333262e-5; // eV/K
const AMU_A2_FS2 = 103.6426965; // eV per amu A^2/fs^2
const AMU_A3 = 1.6605390666; // g/cm^3 per amu/A^3
const THZ_CM = 33.35640952; // cm^-1 per THz

/** Standard atomic weights (amu) indexed by Z = 0..103 (index 0 is the unknown "X"); source in the file header. */
const MASS = Float64Array.from(("1 1.008 4.002602 6.94 9.0121831 10.81 12.011 14.007 15.999 18.998403163 20.1797 22.98976928 24.305 26.9815385 " +
  "28.085 30.973761998 32.06 35.45 39.948 39.0983 40.078 44.955908 47.867 50.9415 51.9961 54.938044 55.845 " +
  "58.933194 58.6934 63.546 65.38 69.723 72.63 74.921595 78.971 79.904 83.798 85.4678 87.62 88.90584 91.224 " +
  "92.90637 95.95 97.90721 101.07 102.9055 106.42 107.8682 112.414 114.818 118.71 121.76 127.6 126.90447 131.293 " +
  "132.90545196 137.327 138.90547 140.116 140.90766 144.242 144.91276 150.36 151.964 157.25 158.92535 162.5 " +
  "164.93033 167.259 168.93422 173.054 174.9668 178.49 180.94788 183.84 186.207 190.23 192.217 195.084 196.966569 " +
  "200.592 204.38 207.2 208.9804 208.98243 209.98715 222.01758 223.01974 226.02541 227.02775 232.0377 231.03588 " +
  "238.02891 237.04817 244.06421 243.06138 247.07035 247.07031 251.07959 252.083 257.09511 258.09843 259.101 262.11").split(" ").map(Number));

/** Atomic mass (amu) of an element symbol: standard atomic weight, 1 for unknown symbols. */
export function massOf(sym) { return MASS[zOf(sym)] || 1; }

/**
 * Time step between saved frames (fs) if the trajectory carries one (traj.dt_fs), else null. No reader sets it at
 * present: dt must come from the user or the file (POTIM in vasprun.xml times the save interval).
 */
export function timeStepOf(traj) { return traj && Number.isFinite(traj.dt_fs) && traj.dt_fs > 0 ? traj.dt_fs : null; }

/* ------------------------------------------------------------------ shared helpers */

/** Elements in order of first appearance and the element index of every atom (Int32Array N). */
function elementIndex(species) {
  const els = [], map = new Map(), idx = new Int32Array(species.length);
  species.forEach((s, i) => { if (!map.has(s)) { map.set(s, els.length); els.push(s); } idx[i] = map.get(s); });
  return { els, idx };
}

/** True if the structure is periodic and has a cell (fixed or per frame). */
const isPeriodic = (structure, traj) => !!structure.pbc && !!(structure.lattice || (traj && traj.cells));

/** Lattice rows (3x3 arrays, A) of frame f: traj.cells when present, else structure.lattice. */
function cellRows(structure, traj, f) {
  if (traj && traj.cells) { const c = traj.cells, o = 9 * f; return [[c[o], c[o + 1], c[o + 2]], [c[o + 3], c[o + 4], c[o + 5]], [c[o + 6], c[o + 7], c[o + 8]]]; }
  return structure.lattice;
}

/**
 * Flat cell and inverse per frame for unwrapping: { L: Float64Array(9 n), Li: Float64Array(9 n), stride: 9 } (rows,
 * row-major; fractional s = r Li for a row vector r), or one 9-vector each with stride 0 when the cell is fixed.
 * null if not periodic.
 */
function cellTable(structure, traj) {
  if (!isPeriodic(structure, traj)) return null;
  const nc = traj.cells ? traj.n : 1, L = new Float64Array(9 * nc), Li = new Float64Array(9 * nc);
  for (let f = 0; f < nc; f++) {
    const A = cellRows(structure, traj, f), B = inv3(A);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) { L[9 * f + 3 * r + c] = A[r][c]; Li[9 * f + 3 * r + c] = B[r][c]; }
  }
  return { L, Li, stride: traj.cells ? 9 : 0 };
}

/**
 * Frame-by-frame unwrapper of all atoms by image flags. Call next(f, out) for f = 0, 1, 2, ... in order; it writes
 * the unwrapped positions of frame f into out (Float64Array 3N, A): with s_i(f) = r_i(f) L_f^{-1} (fractional),
 * n_i(f) = n_i(f-1) - round(s_i(f) - s_i(f-1)), n_i(0) = 0, u_i(f) = r_i(f) + n_i(f) L_f. Without a cell (T null)
 * u = r. Exact for fixed and variable cells as long as no atom moves more than half a cell between saved frames.
 */
function unwrapper(traj, T) {
  const { nat, pos } = traj, s = new Float64Array(3 * nat), img = new Float64Array(3 * nat);
  return (f, out) => {
    for (let i = 0; i < nat; i++) unwrapStep(T, f, pos, 3 * (nat * f + i), s, img, 3 * i, out, 3 * i);
  };
}

/**
 * One unwrapping step of one atom (see unwrapper): reads r = pos[p..p+2] of frame f, updates its last fractional
 * position s[o..o+2] and image flags img[o..o+2], writes u to out[q..q+2]. f = 0 initialises (img = 0).
 */
function unwrapStep(T, f, pos, p, s, img, o, out, q) {
  const x = pos[p], y = pos[p + 1], z = pos[p + 2];
  if (!T) { out[q] = x; out[q + 1] = y; out[q + 2] = z; return; }
  const c = T.stride * f, { L, Li } = T;
  for (let k = 0; k < 3; k++) {
    const sk = x * Li[c + k] + y * Li[c + 3 + k] + z * Li[c + 6 + k];
    if (f === 0) img[o + k] = 0; else img[o + k] -= Math.round(sk - s[o + k]);
    s[o + k] = sk;
  }
  const a = img[o], b = img[o + 1], g = img[o + 2];
  out[q] = x + a * L[c] + b * L[c + 3] + g * L[c + 6];
  out[q + 1] = y + a * L[c + 1] + b * L[c + 4] + g * L[c + 7];
  out[q + 2] = z + a * L[c + 2] + b * L[c + 5] + g * L[c + 8];
}

/**
 * Unwrapped centre-of-mass trajectory, Float64Array(3 n) (A): R(f) = sum_i m_i u_i(f) / sum_i m_i, u_i from
 * unwrapper (frame-major pass, O(N) memory).
 */
function comTrajectory(traj, T, mass) {
  const { n, nat } = traj, next = unwrapper(traj, T), u = new Float64Array(3 * nat), R = new Float64Array(3 * n);
  let M = 0;
  for (let i = 0; i < nat; i++) M += mass[i];
  for (let f = 0; f < n; f++) {
    next(f, u);
    let X = 0, Y = 0, Z = 0;
    for (let i = 0; i < nat; i++) { X += mass[i] * u[3 * i]; Y += mass[i] * u[3 * i + 1]; Z += mass[i] * u[3 * i + 2]; }
    R[3 * f] = X / M; R[3 * f + 1] = Y / M; R[3 * f + 2] = Z / M;
  }
  return R;
}

/**
 * Unwrapped path of atom i relative to its start, sampled every `stride` frames: out[3k..3k+2] = u_i(k stride) -
 * u_i(0) - (R(k stride) - R(0)) (A), with R the centre-of-mass path (or no subtraction when R is null). out:
 * Float64Array(3 K), K = floor((n - 1) / stride) + 1. Image flags are followed through every frame, not only the
 * sampled ones.
 */
function unwrapAtom(traj, T, i, stride, R, out) {
  const { n, nat, pos } = traj, s = new Float64Array(3), img = new Float64Array(3), u = new Float64Array(3);
  let x0 = 0, y0 = 0, z0 = 0;
  for (let f = 0; f < n; f++) {
    unwrapStep(T, f, pos, 3 * (nat * f + i), s, img, 0, u, 0);
    if (f === 0) { x0 = u[0]; y0 = u[1]; z0 = u[2]; }
    if (f % stride === 0) {
      const k = 3 * (f / stride);
      out[k] = u[0] - x0; out[k + 1] = u[1] - y0; out[k + 2] = u[2] - z0;
      if (R) { out[k] -= R[3 * f] - R[0]; out[k + 1] -= R[3 * f + 1] - R[1]; out[k + 2] -= R[3 * f + 2] - R[2]; }
    }
  }
}

/**
 * Sum of power spectra |X_k|^2 of real signals x (length K each), zero-padded to P = nextPow2(2K) so that the
 * inverse transform gives the linear (not circular) autocorrelation. Two real signals share one complex FFT
 * (x + i y): X_k = (C_k + conj C_{P-k}) / 2, Y_k = (C_k - conj C_{P-k}) / (2i).
 */
class PowerSum {
  constructor(K) {
    this.K = K; this.P = nextPow2(2 * K);
    this.re = new Float64Array(this.P); this.im = new Float64Array(this.P); this.acc = new Float64Array(this.P);
    this.pending = null;
  }
  /** Add one real signal (array-like, read at x[off + k * step], k = 0..K-1). */
  add(x, off = 0, step = 1) {
    const s = new Float64Array(this.K);
    for (let k = 0; k < this.K; k++) s[k] = x[off + k * step];
    if (!this.pending) { this.pending = s; return; }
    this.flush(s);
  }
  /** Transform the pending signal, with y as the imaginary part if given (else zero). */
  flush(y = null) {
    const a = this.pending;
    if (!a) return;
    const { re, im, acc, P, K } = this;
    re.fill(0); im.fill(0); re.set(a);
    if (y) im.set(y.subarray(0, K));
    fft1(re, im, -1);
    for (let k = 0; k < P; k++) {
      const j = (P - k) % P, ar = re[k], ai = im[k], cr = re[j], ci = im[j];
      acc[k] += ((ar + cr) * (ar + cr) + (ai - ci) * (ai - ci)) / 4 + ((ai + ci) * (ai + ci) + (ar - cr) * (ar - cr)) / 4;
    }
    this.pending = null;
  }
  /** Autocorrelation sums A(m) = sum_k x_k x_{k+m} over all added signals, m = 0..maxLag (Float64Array). */
  correlation(maxLag) {
    this.flush();
    const { re, im, acc, P } = this;
    re.set(acc); im.fill(0);
    fft1(re, im, 1);
    const out = new Float64Array(maxLag + 1);
    for (let m = 0; m <= maxLag; m++) out[m] = re[m] / P;
    return out;
  }
}

/** Selected element symbols (all present if `elements` is empty or absent), checked against the structure. */
function selectElements(els, elements) {
  if (!elements || !elements.length) return els.slice();
  const want = els.filter(e => elements.includes(e));
  if (!want.length) throw new Error(`none of ${elements.join(", ")} is in the structure`);
  return want;
}

/** Ordinary least-squares line y = s x + c over x[i0..i1]: { slope, intercept }. */
function linfit(x, y, i0, i1) {
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = i0; i <= i1; i++) { n++; sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
  const den = n * sxx - sx * sx;
  const slope = den ? (n * sxy - sx * sy) / den : NaN;
  return { slope, intercept: (sy - slope * sx) / n };
}

/* ------------------------------------------------------------------ radial distribution function */

// WHY: cell-list bins at least rmax / SUB wide; SUB = 2 scans (5 rmax/2)^3 = 15.6 rmax^3 around an atom instead of
// 27 rmax^3 with rmax-wide bins, without too many nearly empty bins.
const SUB = 2;

/**
 * Pair-distance histogram of one frame. P: positions (Float64Array 3N, A); eidx: element index per atom; ne: number
 * of elements; L: lattice rows (A) or null (non-periodic, no images); counts ordered pairs (i, j) with r_ij < rmax
 * into hist[(e_i ne + e_j) nbins + floor(r_ij / dr)], including all periodic images (lattice translations T with
 * |r_j + T - r_i| < rmax, the self-image of i included when T != 0). Returns the cell volume (A^3), or the bounding
 * box volume when L is null.
 *
 * Method: fractional coordinates s = r L^{-1} wrapped to [0, 1); bins nb_k = floor(SUB w_k / rmax) along each
 * fractional axis (w_k = V / |a_{k+1} x a_{k+2}| the perpendicular width); a neighbour within rmax differs by at most
 * rmax / w_k in s_k, so bin offsets -m_k..m_k with m_k = ceil(rmax nb_k / w_k) cover it, each offset b + o mapped to
 * bin mod(b + o, nb_k) and translation floor((b + o) / nb_k) (exact for any triclinic cell and any rmax).
 */
function rdfFrame(P, nat, eidx, ne, L, rmax, dr, nbins, hist) {
  const periodic = !!L;
  let A = L, org = [0, 0, 0];
  if (!periodic) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < nat; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[3 * i + k]); hi[k] = Math.max(hi[k], P[3 * i + k]); }
    const e = lo.map((x, k) => (hi[k] - x) * (1 + 1e-9) + 1e-6);
    A = [[e[0], 0, 0], [0, e[1], 0], [0, 0, e[2]]]; org = lo;
  }
  const Li = inv3(A), V = Math.abs(det3(A));
  const cr = (u, v) => Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
  const w = [0, 1, 2].map(k => V / cr(A[(k + 1) % 3], A[(k + 2) % 3]));
  const nb = w.map(x => Math.max(1, Math.floor(SUB * x / rmax)));
  // WHY: cap the bin count near 2 N so a large sparse box does not allocate and scan mostly empty bins.
  while (nb[0] * nb[1] * nb[2] > Math.max(64, 2 * nat)) { const k = nb.indexOf(Math.max(...nb)); nb[k] = Math.max(1, nb[k] >> 1); }
  const m = nb.map((b, k) => Math.ceil(rmax * b / w[k]));
  const nB = nb[0] * nb[1] * nb[2], binOf = new Int32Array(nat), start = new Int32Array(nB + 1), W = new Float64Array(3 * nat);
  for (let i = 0; i < nat; i++) {
    const x = P[3 * i] - org[0], y = P[3 * i + 1] - org[1], z = P[3 * i + 2] - org[2], b = [0, 0, 0];
    let wx = x, wy = y, wz = z;
    for (let k = 0; k < 3; k++) {
      let s = x * Li[0][k] + y * Li[1][k] + z * Li[2][k];
      if (periodic) {
        const fl = Math.floor(s);
        s -= fl;
        if (fl) { wx -= fl * A[k][0]; wy -= fl * A[k][1]; wz -= fl * A[k][2]; }
      }
      b[k] = Math.min(nb[k] - 1, Math.max(0, Math.floor(s * nb[k])));
    }
    W[3 * i] = wx; W[3 * i + 1] = wy; W[3 * i + 2] = wz;
    binOf[i] = b[0] + nb[0] * (b[1] + nb[1] * b[2]);
    start[binOf[i] + 1]++;
  }
  for (let b = 0; b < nB; b++) start[b + 1] += start[b];
  const fill = start.slice(0, nB), px = new Float64Array(nat), py = new Float64Array(nat), pz = new Float64Array(nat), pe = new Int32Array(nat);
  for (let i = 0; i < nat; i++) { const j = fill[binOf[i]]++; px[j] = W[3 * i]; py[j] = W[3 * i + 1]; pz[j] = W[3 * i + 2]; pe[j] = eidx[i]; }
  const r2max = rmax * rmax, inv = 1 / dr;
  for (let b2 = 0; b2 < nb[2]; b2++) for (let b1 = 0; b1 < nb[1]; b1++) for (let b0 = 0; b0 < nb[0]; b0++) {
    const B = b0 + nb[0] * (b1 + nb[1] * b2), i0 = start[B], i1 = start[B + 1];
    if (i0 === i1) continue;
    for (let o2 = -m[2]; o2 <= m[2]; o2++) {
      const t2 = b2 + o2;
      if (!periodic && (t2 < 0 || t2 >= nb[2])) continue;
      const q2 = Math.floor(t2 / nb[2]), c2 = t2 - q2 * nb[2];
      for (let o1 = -m[1]; o1 <= m[1]; o1++) {
        const t1 = b1 + o1;
        if (!periodic && (t1 < 0 || t1 >= nb[1])) continue;
        const q1 = Math.floor(t1 / nb[1]), c1 = t1 - q1 * nb[1];
        for (let o0 = -m[0]; o0 <= m[0]; o0++) {
          const t0 = b0 + o0;
          if (!periodic && (t0 < 0 || t0 >= nb[0])) continue;
          const q0 = Math.floor(t0 / nb[0]), c0 = t0 - q0 * nb[0];
          const C = c0 + nb[0] * (c1 + nb[1] * c2), j0 = start[C], j1 = start[C + 1];
          if (j0 === j1) continue;
          const Tx = q0 * A[0][0] + q1 * A[1][0] + q2 * A[2][0], Ty = q0 * A[0][1] + q1 * A[1][1] + q2 * A[2][1];
          const Tz = q0 * A[0][2] + q1 * A[1][2] + q2 * A[2][2], self = C === B && !q0 && !q1 && !q2;
          for (let i = i0; i < i1; i++) {
            const xi = px[i] - Tx, yi = py[i] - Ty, zi = pz[i] - Tz, base = pe[i] * ne;
            for (let j = j0; j < j1; j++) {
              if (self && j === i) continue;
              const dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi, r2 = dx * dx + dy * dy + dz * dz;
              if (r2 >= r2max) continue;
              const k = Math.floor(Math.sqrt(r2) * inv);
              if (k < nbins) hist[(base + pe[j]) * nbins + k]++;
            }
          }
        }
      }
    }
  }
  return V;
}

/**
 * Frame indices to use: `frames` an array of indices (kept if 0 <= f < n), or a stride s >= 1 (frames 0, s, 2s, ...),
 * or absent: the stride that keeps at most `auto` frames. Returns Int32Array.
 */
function pickFrames(n, frames, auto) {
  if (Array.isArray(frames) || ArrayBuffer.isView(frames)) {
    const out = Int32Array.from(frames).filter(f => f >= 0 && f < n);
    if (!out.length) throw new Error("no valid frame indices");
    return out;
  }
  const s = Number.isFinite(frames) && frames >= 1 ? Math.floor(frames) : Math.max(1, Math.ceil(n / auto));
  return Int32Array.from({ length: Math.floor((n - 1) / s) + 1 }, (_, k) => k * s);
}

/**
 * Radial distribution functions, total and per unordered element pair, averaged over frames (file header, section 2).
 *
 * structure: as in traj.js; traj: trajectory or null (then the single structure is analysed).
 * Options:
 *   rmax (A, default 8): largest distance; rounded up to a whole number of bins. Any rmax is exact: all periodic
 *     images are counted (cost grows as rmax^3 per atom, and as (rmax / w)^3 images when rmax exceeds the cell
 *     width w).
 *   dr (A, default 0.02): bin width.
 *   frames: array of frame indices, or a stride (integer >= 1); default the stride that uses at most 100 frames.
 *   pairs: ["A-B", ...] element pairs to report (either order); default every unordered pair of elements present.
 * Returns {
 *   r: Float64Array(nbins) bin centres (A); rEdge: Float64Array(nbins) outer bin edges (A), where n(r) is evaluated;
 *   dr, rmax (A); frames: Int32Array used; periodic: boolean; elements: [symbol]; counts: { symbol: N_a };
 *   total: { g: Float64Array | null, n: Float64Array }  (n = mean number of atoms within rEdge of an atom);
 *   pairs: [{ a, b, label: "a-b", g: Float64Array | null, nab: Float64Array (b atoms around an a atom),
 *             nba: Float64Array (a atoms around a b atom) }];
 *   notes: [string] }.
 * Non-periodic structures (pbc false or no cell): g is null (no volume defines the ideal-gas density); n and nab,
 * nba are the plain running neighbour counts within rEdge, with no images.
 * Variable cells: each frame is normalised with its own volume V_f, then frames are averaged.
 */
export function rdf(structure, traj, { rmax = 8, dr = 0.02, frames, pairs } = {}) {
  if (!(rmax > 0 && dr > 0)) throw new Error("rmax and dr must be positive");
  const nat = structure.species.length, nf = traj ? traj.n : 1;
  const nbins = Math.ceil(rmax / dr - 1e-9);
  rmax = nbins * dr;
  const periodic = isPeriodic(structure, traj), used = pickFrames(nf, frames, 100);
  const { els, idx } = elementIndex(structure.species), ne = els.length;
  const Na = new Float64Array(ne);
  for (let i = 0; i < nat; i++) Na[idx[i]]++;
  const h = new Float64Array(ne * ne * nbins), hv = new Float64Array(ne * ne * nbins), hc = new Float64Array(ne * ne * nbins);
  const notes = [];
  for (const f of used) {
    const P = traj ? traj.pos.subarray(3 * nat * f, 3 * nat * (f + 1)) : structure.pos;
    h.fill(0);
    const V = rdfFrame(P, nat, idx, ne, periodic ? cellRows(structure, traj, f) : null, rmax, dr, nbins, h);
    for (let k = 0; k < h.length; k++) if (h[k]) { hv[k] += h[k] * V; hc[k] += h[k]; }
  }
  const F = used.length, r = new Float64Array(nbins), rEdge = new Float64Array(nbins), shell = new Float64Array(nbins);
  for (let k = 0; k < nbins; k++) {
    r[k] = (k + 0.5) * dr; rEdge[k] = (k + 1) * dr;
    shell[k] = (4 / 3) * Math.PI * (Math.pow((k + 1) * dr, 3) - Math.pow(k * dr, 3));
  }
  const gOf = (sumV, norm) => { const g = new Float64Array(nbins); for (let k = 0; k < nbins; k++) g[k] = sumV[k] / (F * norm * shell[k]); return g; };
  const nOf = (cnt, per) => { const n = new Float64Array(nbins); let s = 0; for (let k = 0; k < nbins; k++) { s += cnt[k]; n[k] = s / (F * per); } return n; };
  const tv = new Float64Array(nbins), tc = new Float64Array(nbins);
  for (let p = 0; p < ne * ne; p++) for (let k = 0; k < nbins; k++) { tv[k] += hv[p * nbins + k]; tc[k] += hc[p * nbins + k]; }
  const total = { g: periodic ? gOf(tv, nat * nat) : null, n: nOf(tc, nat) };
  let want = [];
  for (let a = 0; a < ne; a++) for (let b = a; b < ne; b++) want.push([a, b]);
  if (pairs && pairs.length) {
    const keys = new Set(pairs.map(s => s.split("-").map(t => t.trim()).sort().join("-")));
    want = want.filter(([a, b]) => keys.has([els[a], els[b]].sort().join("-")));
  }
  const out = want.map(([a, b]) => {
    const sab = hv.subarray((a * ne + b) * nbins, (a * ne + b + 1) * nbins), cab = hc.subarray((a * ne + b) * nbins, (a * ne + b + 1) * nbins);
    const cba = hc.subarray((b * ne + a) * nbins, (b * ne + a + 1) * nbins);
    return { a: els[a], b: els[b], label: `${els[a]}-${els[b]}`, g: periodic ? gOf(sab, Na[a] * Na[b]) : null, nab: nOf(cab, Na[a]), nba: nOf(cba, Na[b]) };
  });
  if (!periodic) notes.push("non-periodic: g(r) is not defined without a volume; n(r) are plain neighbour counts");
  const counts = {};
  els.forEach((e, a) => { counts[e] = Na[a]; });
  return { r, rEdge, dr, rmax, frames: used, periodic, elements: els, counts, total, pairs: out, notes };
}

/* ------------------------------------------------------------------ mean-squared displacement */

/**
 * Mean-squared displacement per element and self-diffusion coefficients (file header, sections 1 and 3).
 *
 * Options:
 *   dt_fs: time between consecutive frames (fs). Without it the time axis is in frames and D is null.
 *   stride (default 1): use every stride-th frame as a time origin and lag step (unwrapping still uses every frame).
 *   maxLag (in sampled frames, default floor(K / 2), K the number of sampled frames): longest lag returned.
 *   elements: [symbol] to analyse, default all.
 *   fit: [lo, hi] fractions of maxLag for the linear fit (default [0.2, 0.8]).
 *   removeDrift (default true): subtract the mass-weighted centre-of-mass displacement of all atoms from every atom
 *     (thermostats and numerical noise make the whole cell drift, which would add to D).
 * Returns {
 *   t: Float64Array(maxLag + 1) lag times (ps, or frames without dt); timeUnit: "ps" | "frame";
 *   dtSample_fs: time between sampled frames (fs) | null; removeDrift;
 *   elements: [{ el, count, msd: Float64Array (A^2), D_A2ps, D_cm2s, slope (A^2 per time unit), intercept (A^2),
 *                fit: [t0, t1] }];
 *   total: same fields for all selected atoms together (plain average over atoms, no mass weighting);
 *   notes: [string] }.
 * Non-periodic structures are not unwrapped. Wrong unwrapping (an atom moving more than half a cell width between
 * saved frames) gives jumps of a lattice vector in the MSD.
 */
export function msd(structure, traj, { dt_fs, stride = 1, maxLag, elements, fit = [0.2, 0.8], removeDrift = true } = {}) {
  if (!traj || traj.n < 3) throw new Error("MSD needs a trajectory of at least 3 frames");
  stride = Math.max(1, Math.floor(stride));
  const { nat, n } = traj, K = Math.floor((n - 1) / stride) + 1;
  if (K < 3) throw new Error("fewer than 3 frames after the stride");
  maxLag = Math.min(K - 1, Math.max(1, Math.floor(maxLag ?? K / 2)));
  const T = cellTable(structure, traj), { els, idx } = elementIndex(structure.species);
  const sel = selectElements(els, elements);
  const mass = Float64Array.from(structure.species, massOf);
  const R = removeDrift ? comTrajectory(traj, T, mass) : null;
  const ps = new Map(sel.map(e => [e, new PowerSum(K)])), sq = new Map(sel.map(e => [e, new Float64Array(K)])), cnt = new Map(sel.map(e => [e, 0]));
  const u = new Float64Array(3 * K);
  for (let i = 0; i < nat; i++) {
    const e = els[idx[i]];
    if (!ps.has(e)) continue;
    unwrapAtom(traj, T, i, stride, R, u);
    const P = ps.get(e), D = sq.get(e);
    for (let c = 0; c < 3; c++) P.add(u, c, 3);
    for (let k = 0; k < K; k++) D[k] += u[3 * k] * u[3 * k] + u[3 * k + 1] * u[3 * k + 1] + u[3 * k + 2] * u[3 * k + 2];
    cnt.set(e, cnt.get(e) + 1);
  }
  const dts = Number.isFinite(dt_fs) && dt_fs > 0 ? dt_fs * stride : null;
  const t = Float64Array.from({ length: maxLag + 1 }, (_, m) => (dts ? m * dts / 1000 : m * stride));
  const i0 = Math.max(0, Math.round(fit[0] * maxLag)), i1 = Math.min(maxLag, Math.max(i0 + 1, Math.round(fit[1] * maxLag)));
  const summarise = (el, count, curve) => {
    const { slope, intercept } = linfit(t, curve, i0, i1);
    const D = dts ? slope / 6 : null;
    return { el, count, msd: curve, D_A2ps: D, D_cm2s: D === null ? null : D * 1e-4, slope, intercept, fit: [t[i0], t[i1]] };
  };
  const tot = new Float64Array(maxLag + 1);
  let ntot = 0;
  const out = sel.map(e => {
    const A = ps.get(e).correlation(maxLag), D = sq.get(e), Ne = cnt.get(e), curve = new Float64Array(maxLag + 1);
    // WHY: S1 recursion of Calandrini et al.; Q(m) = sum_{k=0}^{K-m-1} (D_k + D_{k+m}).
    let Q = 0;
    for (let k = 0; k < K; k++) Q += 2 * D[k];
    for (let m = 0; m <= maxLag; m++) {
      if (m > 0) Q -= D[m - 1] + D[K - m];
      curve[m] = Math.max(0, (Q - 2 * A[m]) / ((K - m) * Ne));
      tot[m] += curve[m] * Ne;
    }
    ntot += Ne;
    return summarise(e, Ne, curve);
  });
  for (let m = 0; m <= maxLag; m++) tot[m] /= ntot;
  const notes = [];
  if (!T) notes.push("non-periodic: positions are used as they are, without unwrapping");
  if (!dts) notes.push("no time step: lags are in frames and D is not given");
  return { t, timeUnit: dts ? "ps" : "frame", dtSample_fs: dts, removeDrift, elements: out, total: summarise("all", ntot, tot), notes };
}

/* ------------------------------------------------------------------ velocity autocorrelation and VDOS */

/**
 * Velocity autocorrelation per element, mass-weighted total and vibrational density of states (file header,
 * sections 1 and 4).
 *
 * Options:
 *   dt_fs (required): time between consecutive frames (fs).
 *   stride (default 1): sample every stride-th frame; velocities are central differences over 2 stride dt.
 *   maxLag (sampled frames, default floor(K_v / 2), K_v = K - 2 velocities from K samples): longest lag; sets the
 *     frequency resolution about 1 / (maxLag dt_s) (the Hann window doubles the width of a line).
 *   elements: [symbol] to include, default all (the mass-weighted total and the VDOS use the selected elements only).
 *   removeDrift (default true): subtract the centre-of-mass velocity of all atoms (removes the zero-frequency peak
 *     of a drifting cell).
 *   pad (default 4): zero-padding factor of the cosine transform; finer frequency sampling, not finer resolution.
 * Returns {
 *   t: Float64Array(maxLag + 1) lags (ps); dtSample_fs;
 *   elements: [{ el, count, mass (amu), v2: <|v|^2> (A^2/fs^2), z: Float64Array normalised VACF (z[0] = 1),
 *                vdos: Float64Array partial VDOS (1/THz; the partials sum to the total) }];
 *   total: { z: mass-weighted normalised VACF, vdos: Float64Array (1/THz, unit area over 0..nyquist) };
 *   nu_THz: Float64Array frequencies, nu_cm: same in cm^-1; nyquist_THz = 1 / (2 dt_s); resolution_THz =
 *   1 / (maxLag dt_s); notes }.
 * dt_s = stride dt_fs. Approximations: central differences damp the spectrum by sinc^2(2 pi nu dt_s) (zero at the
 * Nyquist frequency); saving frames every k MD steps lowers the Nyquist frequency and aliases faster modes.
 */
export function vacf(structure, traj, { dt_fs, stride = 1, maxLag, elements, removeDrift = true, pad = 4 } = {}) {
  if (!(Number.isFinite(dt_fs) && dt_fs > 0)) throw new Error("VACF needs the time step dt_fs");
  if (!traj) throw new Error("VACF needs a trajectory");
  stride = Math.max(1, Math.floor(stride));
  const { nat, n } = traj, K = Math.floor((n - 1) / stride) + 1, Kv = K - 2;
  if (Kv < 4) throw new Error("too few frames for velocities after the stride");
  maxLag = Math.min(Kv - 1, Math.max(2, Math.floor(maxLag ?? Kv / 2)));
  const dts = dt_fs * stride;
  const T = cellTable(structure, traj), { els, idx } = elementIndex(structure.species);
  const sel = selectElements(els, elements);
  const mass = Float64Array.from(structure.species, massOf);
  // WHY: a constant COM velocity offset only adds a constant to every VACF; subtracting the COM path before the
  // differences removes it exactly (differences are linear).
  const R = removeDrift ? comTrajectory(traj, T, mass) : null;
  const ps = new Map(sel.map(e => [e, new PowerSum(Kv)])), cnt = new Map(sel.map(e => [e, 0]));
  const u = new Float64Array(3 * K), v = new Float64Array(3 * Kv);
  for (let i = 0; i < nat; i++) {
    const e = els[idx[i]];
    if (!ps.has(e)) continue;
    unwrapAtom(traj, T, i, stride, R, u);
    for (let k = 0; k < Kv; k++) for (let c = 0; c < 3; c++) v[3 * k + c] = (u[3 * (k + 2) + c] - u[3 * k + c]) / (2 * dts);
    const P = ps.get(e);
    for (let c = 0; c < 3; c++) P.add(v, c, 3);
    cnt.set(e, cnt.get(e) + 1);
  }
  const C = sel.map(e => {
    const A = ps.get(e).correlation(maxLag), Ne = cnt.get(e);
    for (let m = 0; m <= maxLag; m++) A[m] /= (Kv - m) * Ne;
    return A; // C_a(m), A^2/fs^2
  });
  const me = sel.map(massOf);
  let W0 = 0;
  sel.forEach((e, a) => { W0 += me[a] * cnt.get(e) * C[a][0]; });
  const ztot = new Float64Array(maxLag + 1);
  sel.forEach((e, a) => { for (let m = 0; m <= maxLag; m++) ztot[m] += me[a] * cnt.get(e) * C[a][m] / W0; });
  const Lf = Math.max(2, Math.floor(pad)) * nextPow2(2 * maxLag), nfreq = Lf / 2 + 1;
  const nu = Float64Array.from({ length: nfreq }, (_, k) => 1000 * k / (Lf * dts));
  const hann = Float64Array.from({ length: maxLag + 1 }, (_, m) => 0.5 * (1 + Math.cos(Math.PI * m / maxLag)));
  const re = new Float64Array(Lf), im = new Float64Array(Lf);
  // Cosine transform of y (Float64Array maxLag + 1, already windowed): G_k = dts (y_0 + 2 sum_m y_m cos(2 pi k m / Lf)),
  // as the real part of the FFT of the even extension y_{Lf - m} = y_m. Units: fs times the units of y.
  const cosT = y => {
    re.fill(0); im.fill(0);
    re[0] = y[0];
    for (let m = 1; m <= maxLag; m++) re[m] = re[Lf - m] = y[m];
    fft1(re, im, -1);
    return Float64Array.from({ length: nfreq }, (_, k) => re[k] * dts);
  };
  const gtot = cosT(ztot.map((z, m) => z * hann[m]));
  let area = 0;
  for (let k = 1; k < nfreq; k++) area += 0.5 * (gtot[k] + gtot[k - 1]) * (nu[k] - nu[k - 1]);
  for (let k = 0; k < nfreq; k++) gtot[k] /= area;
  const out = sel.map((e, a) => {
    const z = C[a].map(x => x / C[a][0]);
    const part = cosT(C[a].map((x, m) => me[a] * cnt.get(e) * x / W0 * hann[m]));
    for (let k = 0; k < nfreq; k++) part[k] /= area;
    return { el: e, count: cnt.get(e), mass: me[a], v2: C[a][0], z, vdos: part };
  });
  return {
    t: Float64Array.from({ length: maxLag + 1 }, (_, m) => m * dts / 1000), dtSample_fs: dts, elements: out,
    total: { z: ztot, vdos: gtot }, nu_THz: nu, nu_cm: nu.map(x => x * THZ_CM), nyquist_THz: 1000 / (2 * dts),
    resolution_THz: 1000 / (maxLag * dts), removeDrift,
    notes: ["VDOS: Hann-windowed cosine transform of the mass-weighted VACF, unit area; central-difference velocities damp it toward the Nyquist frequency"],
  };
}

/* ------------------------------------------------------------------ per-frame series */

/**
 * Per-frame series for plotting against frame index or time.
 *
 * Options: dt_fs (time between frames, fs): needed for the time axis and the temperature estimate.
 * Returns {
 *   n; t: Float64Array(n) time (ps) | null without dt_fs;
 *   energy: Float64Array(n) (eV, NaN where missing) | null;
 *   volume: Float64Array(n) (A^3) | null (non-periodic); density: Float64Array(n) (g/cm^3) | null;
 *   a, b, c (A), alpha, beta, gamma (degrees): Float64Array(n) | null (non-periodic);
 *   temperature: Float64Array(n) (K, NaN at the first and last frame) | null without dt_fs or with fewer than 2
 *   atoms; dof: 3N - 3; notes }.
 * temperature is an ESTIMATE: T(f) = sum_i m_i |v_i - v_com|^2 / ((3N - 3) k_B), with v_i = (u_i(f + 1) - u_i(f - 1))
 * / (2 dt) from saved positions (A/fs). It is close to the thermostat temperature only when frames are saved every
 * MD step or two; with frames saved every k steps the finite difference averages the velocity over 2 k dt and
 * underestimates T (strongly when 2 k dt approaches a vibrational period), and a wrong dt scales T by (dt_true /
 * dt)^2. Fixed atoms (selective dynamics) still count as degrees of freedom here.
 */
export function perFrame(structure, traj, { dt_fs } = {}) {
  if (!traj) throw new Error("per-frame series need a trajectory");
  const { n, nat } = traj, periodic = isPeriodic(structure, traj);
  const mass = Float64Array.from(structure.species, massOf);
  let M = 0;
  for (let i = 0; i < nat; i++) M += mass[i];
  const out = { n, t: null, energy: traj.energy ? Float64Array.from(traj.energy) : null, volume: null, density: null, a: null, b: null, c: null, alpha: null, beta: null, gamma: null, temperature: null, dof: 3 * nat - 3, notes: [] };
  const dt = Number.isFinite(dt_fs) && dt_fs > 0 ? dt_fs : null;
  if (dt) out.t = Float64Array.from({ length: n }, (_, f) => f * dt / 1000);
  if (periodic) {
    for (const k of ["volume", "density", "a", "b", "c", "alpha", "beta", "gamma"]) out[k] = new Float64Array(n);
    let p0 = null, v0 = 0;
    for (let f = 0; f < n; f++) {
      if (!traj.cells && p0) {
        out.volume[f] = v0; out.density[f] = out.density[0];
        ["a", "b", "c", "alpha", "beta", "gamma"].forEach((k, j) => { out[k][f] = p0[j]; });
        continue;
      }
      const L = cellRows(structure, traj, f), V = Math.abs(det3(L)), p = params(L);
      out.volume[f] = V; out.density[f] = M * AMU_A3 / V;
      ["a", "b", "c", "alpha", "beta", "gamma"].forEach((k, j) => { out[k][f] = p[j]; });
      p0 = p; v0 = V;
    }
  }
  if (dt && nat >= 2 && n >= 3) {
    const T = cellTable(structure, traj), Tk = new Float64Array(n).fill(NaN);
    // Rolling unwrapped positions u(f - 1), u(f), u(f + 1); d = u(f + 1) - u(f - 1) = 2 dt v(f) (A).
    const next = unwrapper(traj, T);
    let um = new Float64Array(3 * nat), u0 = new Float64Array(3 * nat), up = new Float64Array(3 * nat);
    next(0, um); next(1, u0);
    for (let f = 1; f < n - 1; f++) {
      next(f + 1, up);
      let px = 0, py = 0, pz = 0;
      for (let i = 0; i < nat; i++) { px += mass[i] * (up[3 * i] - um[3 * i]); py += mass[i] * (up[3 * i + 1] - um[3 * i + 1]); pz += mass[i] * (up[3 * i + 2] - um[3 * i + 2]); }
      px /= M; py /= M; pz /= M;
      let s = 0;
      for (let i = 0; i < nat; i++) {
        const vx = up[3 * i] - um[3 * i] - px, vy = up[3 * i + 1] - um[3 * i + 1] - py, vz = up[3 * i + 2] - um[3 * i + 2] - pz;
        s += mass[i] * (vx * vx + vy * vy + vz * vz);
      }
      // sum m v^2 with v = d / (2 dt); T = sum m v^2 / (dof kB).
      Tk[f] = s / (4 * dt * dt) * AMU_A2_FS2 / (out.dof * KB);
      [um, u0, up] = [u0, up, um];
    }
    out.temperature = Tk;
    out.notes.push("temperature is an estimate from finite-difference velocities of saved positions (3N - 3 degrees of freedom)");
  }
  return out;
}
