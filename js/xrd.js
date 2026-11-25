// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Utkarsh Singh
/**
 * Powder X-ray diffraction of a crystal, computed as pymatgen's XRDCalculator (2024.8) does, a broadened profile, and
 * patterns along a trajectory (one per frame, their average, and the frames as a map). Pure numerics; runs in a worker.
 *
 * Lattice rows a_i (A), positions r_j (A); fractional x_j = r_j L^-1 (row vector). Crystallographic reciprocal
 * lattice: rows g_i with g_i . a_k = delta_ik (no factor 2 pi), the columns of L^-1. Reflection hkl:
 * G = h g_1 + k g_2 + l g_3, d = 1 / |G| (A), Bragg angle theta = asin(lambda |G| / 2), lambda the wavelength (A).
 * Every G with |G| <= 2 sin(theta_max) / lambda is taken (all symmetry equivalents, so no multiplicity factor).
 * Atomic form factor (X-ray, in electrons), s = sin(theta) / lambda = |G| / 2 (1/A):
 *   f(s) = Z - 41.78214 s^2 sum_{i=1..4} a_i exp(-b_i s^2)
 * (the Mott-Bethe relation applied to 4-Gaussian fits of the electron scattering factors, a_i in A, b_i in A^2; the
 * coefficients are those pymatgen distributes, atomic_scattering_params.json, MIT licence: see NOTICE).
 * Structure factor F = sum_j f_j exp(2 pi i (h x_j1 + k x_j2 + l x_j3)); no thermal factor, no anomalous dispersion.
 * Intensity I = |F|^2 LP with the Lorentz-polarisation factor LP = (1 + cos^2 2theta) / (sin^2 theta cos theta).
 * Friedel pairs hkl and -h-k-l have equal |F| (real f), so F is computed once per pair.
 * Reflections sorted by (|G|, -h, -k, -l); one within 1e-5 deg (2theta) of the previous peak joins it (intensities
 * added, hkl listed); peaks below 1e-3 of the strongest are dropped; scaled, the strongest is 100.
 * Profile: y(2theta) = sum_k I_k exp(-4 ln 2 (2theta - 2theta_k)^2 / w^2), w the full width at half maximum (deg).
 */
import { inv3, cartToFrac, norm } from "./lattice.js?v=1005";
import { zOf } from "./elements.js?v=1005";

/** Common X-ray tube lines (A), as pymatgen gives them (K alpha: the weighted K alpha1 / K alpha2 mean). */
export const WAVELENGTHS = { "Cu Kα": 1.54184, "Mo Kα": 0.71073, "Co Kα": 1.79026, "Fe Kα": 1.93735, "Cr Kα": 2.291, "Ag Kα": 0.560885 };
/** a1, b1, ..., a4, b4 per element (a_i in A, b_i in A^2). */
const FF = {
  H:[0.202,30.868,0.244,8.544,0.082,1.273,0,0], D:[0.202,30.868,0.244,8.544,0.082,1.273,0,0],
  He:[0.091,18.183,0.181,6.212,0.11,1.803,0.036,0.284], Li:[1.611,107.638,1.246,30.48,0.326,4.533,0.099,0.495],
  Be:[1.25,60.804,1.334,18.591,0.36,3.653,0.106,0.416], B:[0.945,46.444,1.312,14.178,0.419,3.223,0.116,0.377],
  C:[0.731,36.995,1.195,11.297,0.456,2.814,0.125,0.346], N:[0.572,28.847,1.043,9.054,0.465,2.421,0.131,0.317],
  O:[0.455,23.78,0.917,7.622,0.472,2.144,0.138,0.296], F:[0.387,20.239,0.811,6.609,0.475,1.931,0.146,0.279],
  Ne:[0.303,17.64,0.72,5.86,0.475,1.762,0.153,0.266], Na:[2.241,108.004,1.333,24.505,0.907,3.391,0.286,0.435],
  Mg:[2.268,73.67,1.803,20.175,0.839,3.013,0.289,0.405], Al:[2.276,72.322,2.428,19.773,0.858,3.08,0.317,0.408],
  Si:[2.129,57.775,2.533,16.476,0.835,2.88,0.322,0.386], P:[1.888,44.876,2.469,13.538,0.805,2.642,0.32,0.361],
  S:[1.659,36.65,2.386,11.488,0.79,2.469,0.321,0.34], Cl:[1.452,30.935,2.292,9.98,0.787,2.234,0.322,0.323],
  Ar:[1.274,26.682,2.19,8.813,0.793,2.219,0.326,0.307], K:[3.951,137.075,2.545,22.402,1.98,4.532,0.482,0.434],
  Ca:[4.47,99.523,2.971,22.696,1.97,4.195,0.482,0.417], Sc:[3.966,88.96,2.917,20.606,1.925,3.856,0.48,0.399],
  Ti:[3.565,81.982,2.818,19.049,1.893,3.59,0.483,0.386], V:[3.245,76.379,2.698,17.726,1.86,3.363,0.486,0.374],
  Cr:[2.307,78.405,2.334,15.785,1.823,3.157,0.49,0.364], Mn:[2.747,67.786,2.456,15.674,1.792,3,0.498,0.357],
  Fe:[2.544,64.424,2.343,14.88,1.759,2.854,0.506,0.35], Co:[2.367,61.431,2.236,14.18,1.724,2.725,0.515,0.344],
  Ni:[2.21,58.727,2.134,13.553,1.689,2.609,0.524,0.339], Cu:[1.579,62.94,1.82,12.453,1.658,2.504,0.532,0.333],
  Zn:[1.942,54.162,1.95,12.518,1.619,2.416,0.543,0.33], Ga:[2.321,65.602,2.486,15.458,1.688,2.581,0.599,0.351],
  Ge:[2.447,55.893,2.702,14.393,1.616,2.446,0.601,0.342], As:[2.399,45.718,2.79,12.817,1.529,2.28,0.594,0.328],
  Se:[2.298,38.83,2.854,11.536,1.456,2.146,0.59,0.316], Br:[2.166,33.899,2.904,10.497,1.395,2.041,0.589,0.307],
  Kr:[2.034,29.999,2.927,9.598,1.342,1.952,0.589,0.299], Rb:[4.776,140.782,3.859,18.991,2.234,3.701,0.868,0.419],
  Sr:[5.848,104.972,4.003,19.367,2.342,3.737,0.88,0.414], Y:[4.129,27.548,3.012,5.088,1.179,0.591,0,0],
  Zr:[4.105,28.492,3.144,5.277,1.229,0.601,0,0], Nb:[4.237,27.415,3.105,5.074,1.234,0.593,0,0],
  Mo:[3.12,72.464,3.906,14.642,2.361,3.237,0.85,0.366], Tc:[4.318,28.246,3.27,5.148,1.287,0.59,0,0],
  Ru:[4.358,27.881,3.298,5.179,1.323,0.594,0,0], Rh:[4.431,27.911,3.343,5.153,1.345,0.592,0,0],
  Pd:[4.436,28.67,3.454,5.269,1.383,0.595,0,0], Ag:[2.036,61.497,3.272,11.824,2.511,2.846,0.837,0.327],
  Cd:[2.574,55.675,3.259,11.838,2.547,2.784,0.838,0.322], In:[3.153,66.649,3.557,14.449,2.818,2.976,0.884,0.335],
  Sn:[3.45,59.104,3.735,14.179,2.118,2.855,0.877,0.327], Sb:[3.564,50.487,3.844,13.316,2.687,2.691,0.864,0.316],
  Te:[4.785,27.999,3.688,5.083,1.5,0.581,0,0], I:[3.473,39.441,4.06,11.816,2.522,2.415,0.84,0.298],
  Xe:[3.366,35.509,4.147,11.117,2.443,2.294,0.829,0.289], Cs:[6.062,155.837,5.986,19.695,3.303,3.335,1.096,0.379],
  Ba:[7.821,117.657,6.004,18.778,3.28,3.263,1.103,0.376], La:[4.94,28.716,3.968,5.245,1.663,0.594,0,0],
  Ce:[5.007,28.283,3.98,5.183,1.678,0.589,0,0], Pr:[5.085,28.588,4.043,5.143,1.684,0.581,0,0],
  Nd:[5.151,28.304,4.075,5.073,1.683,0.571,0,0], Pm:[5.201,28.079,4.094,5.081,1.719,0.576,0,0],
  Sm:[5.255,28.016,4.113,5.037,1.743,0.577,0,0], Eu:[6.267,100.298,4.844,16.066,3.202,2.98,1.2,0.367],
  Gd:[5.225,29.158,4.314,5.259,1.827,0.586,0,0], Tb:[5.272,29.046,4.347,5.226,1.844,0.585,0,0],
  Dy:[5.332,28.888,4.37,5.198,1.863,0.581,0,0], Ho:[5.376,28.773,4.403,5.174,1.884,0.582,0,0],
  Er:[5.436,28.655,4.437,5.117,1.891,0.577,0,0], Tm:[5.441,29.149,4.51,5.264,1.956,0.59,0,0],
  Yb:[5.529,28.927,4.533,5.144,1.945,0.578,0,0], Lu:[5.553,28.907,4.58,5.16,1.969,0.577,0,0],
  Hf:[5.588,29.001,4.619,5.164,1.997,0.579,0,0], Ta:[5.659,28.807,4.63,5.114,2.014,0.578,0,0],
  W:[5.709,28.782,4.677,5.084,2.019,0.572,0,0], Re:[5.695,28.968,4.74,5.156,2.064,0.575,0,0],
  Os:[5.75,28.933,4.773,5.139,2.079,0.573,0,0], Ir:[5.754,29.159,4.851,5.152,2.096,0.57,0,0],
  Pt:[5.803,29.016,4.87,5.15,2.127,0.572,0,0], Au:[2.388,42.866,4.226,9.743,2.689,2.264,1.255,0.307],
  Hg:[2.682,42.822,4.241,9.856,2.755,2.295,1.27,0.307], Tl:[5.932,29.086,4.972,5.126,2.195,0.572,0,0],
  Pb:[3.51,52.914,4.552,11.884,3.154,2.571,1.359,0.321], Bi:[3.841,50.261,4.679,11.999,3.192,2.56,1.363,0.318],
  Po:[6.07,28.075,4.997,4.999,2.232,0.563,0,0], At:[6.133,28.047,5.031,4.957,2.239,0.558,0,0],
  Rn:[4.078,38.406,4.978,11.02,3.096,2.355,1.326,0.299], Fr:[6.201,28.2,5.121,4.954,2.275,0.556,0,0],
  Ra:[6.215,28.382,5.17,5.002,2.316,0.562,0,0], Ac:[6.278,28.323,5.195,4.949,2.321,0.557,0,0],
  Th:[6.264,28.651,5.263,5.03,2.367,0.563,0,0], Pa:[6.306,28.688,5.303,5.026,2.386,0.561,0,0],
  U:[6.767,85.951,6.729,15.642,4.014,2.936,1.561,0.335], Np:[6.323,29.142,5.414,5.096,2.453,0.568,0,0],
  Pu:[6.415,28.836,5.419,5.022,2.449,0.561,0,0], Am:[6.378,29.156,5.495,5.102,2.495,0.565,0,0],
  Cm:[6.46,28.396,5.469,4.97,2.471,0.554,0,0], Bk:[6.502,28.375,5.478,4.975,2.51,0.561,0,0],
  Cf:[6.548,28.461,5.526,4.965,2.52,0.557,0,0],
};
const MB = 41.78214; // Mott-Bethe constant of f(s) above (A^-1)
const TOL = 1e-5, SCALED_TOL = 1e-3; // deg; percent of the strongest peak

/** X-ray form factor of element el at s^2 = (sin theta / lambda)^2 (1/A^2), in electrons. */
export function formFactor(el, s2) {
  const c = FF[el];
  if (!c) throw new Error(`No X-ray scattering factors for ${el}.`);
  const Z = el === "D" ? 1 : zOf(el);
  return Z - MB * s2 * (c[0] * Math.exp(-c[1] * s2) + c[2] * Math.exp(-c[3] * s2) + c[4] * Math.exp(-c[5] * s2) + c[6] * Math.exp(-c[7] * s2));
}

/**
 * Peaks of the powder pattern of one cell. L: lattice rows (A); species: symbols; pos: Cartesian positions (3N, A).
 * opts: lambda (A), range [2theta min, max] (deg, max below 180), scaled (strongest = 100, default true).
 * Returns { peaks: [{ tth (deg), d (A), I, hkls: [[h, k, l], ...] }] sorted by 2theta, max: the strongest raw I }.
 */
export function xrdPeaks(L, species, pos, { lambda = 1.54184, range = [0, 90], scaled = true } = {}) {
  const Li = inv3(L), g = [0, 1, 2].map(i => [Li[0][i], Li[1][i], Li[2][i]]), deg = Math.PI / 180;
  const rmin = 2 * Math.sin(Math.max(0, range[0]) * deg / 2) / lambda, rmax = 2 * Math.sin(Math.min(179.9, range[1]) * deg / 2) / lambda;
  const els = [...new Set(species)], n = species.length, X = new Float64Array(3 * n), group = els.map(() => []);
  for (const e of els) if (!FF[e]) throw new Error(`No X-ray scattering factors for ${e}.`);
  for (let j = 0; j < n; j++) { X.set(cartToFrac(Li, [pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]]), 3 * j); group[els.indexOf(species[j])].push(j); }
  // |h_i| = |G . a_i| <= |G| |a_i|
  const hm = [0, 1, 2].map(i => Math.ceil(rmax * norm(L[i]) + 1e-9)), refl = [], TAU = 2 * Math.PI;
  for (let h = -hm[0]; h <= hm[0]; h++) for (let k = -hm[1]; k <= hm[1]; k++) for (let l = -hm[2]; l <= hm[2]; l++) {
    // one of each Friedel pair: the first non-zero index positive
    if (!(h > 0 || (h === 0 && (k > 0 || (k === 0 && l > 0))))) continue;
    const G0 = h * g[0][0] + k * g[1][0] + l * g[2][0], G1 = h * g[0][1] + k * g[1][1] + l * g[2][1], G2 = h * g[0][2] + k * g[1][2] + l * g[2][2];
    const q = Math.sqrt(G0 * G0 + G1 * G1 + G2 * G2);
    if (q > rmax * (1 + 1e-12) || q < rmin) continue;
    const s2 = q * q / 4;
    let re = 0, im = 0;
    els.forEach((e, ei) => {
      const f = formFactor(e, s2);
      let c = 0, s = 0;
      for (const j of group[ei]) { const ph = TAU * (h * X[3 * j] + k * X[3 * j + 1] + l * X[3 * j + 2]); c += Math.cos(ph); s += Math.sin(ph); }
      re += f * c; im += f * s;
    });
    const th = Math.asin(lambda * q / 2), lp = (1 + Math.cos(2 * th) ** 2) / (Math.sin(th) ** 2 * Math.cos(th)), I = (re * re + im * im) * lp;
    refl.push({ q, h: [h, k, l], I }, { q, h: [-h, -k, -l], I });
  }
  refl.sort((a, b) => a.q - b.q || b.h[0] - a.h[0] || b.h[1] - a.h[1] || b.h[2] - a.h[2]);
  const peaks = [];
  for (const r of refl) {
    const tth = 2 * Math.asin(lambda * r.q / 2) / deg, last = peaks[peaks.length - 1];
    if (last && Math.abs(tth - last.tth) < TOL) { last.I += r.I; last.hkls.push(r.h); }
    else peaks.push({ tth, d: 1 / r.q, I: r.I, hkls: [r.h] });
  }
  const max = peaks.reduce((m, p) => Math.max(m, p.I), 0);
  const kept = peaks.filter(p => max > 0 && p.I / max * 100 > SCALED_TOL);
  if (scaled && max > 0) for (const p of kept) p.I *= 100 / max;
  return { peaks: kept, max };
}

/**
 * Families of the hkl of one peak, in the basis of cell C (rows, A; the conventional cell) when every G maps to whole
 * indices there (h_C,i = G . c_i), else in the computed cell L. Hexagonal cells (|a| = |b|, angle 120 deg) get
 * Miller-Bravais indices (h, k, -h-k, l). A family: indices equal up to order and signs; its name is the largest
 * member (lexicographic), its multiplicity the number of members. Returns [{ hkl, m }] by decreasing m.
 */
export function families(hkls, L, C = null) {
  const Li = inv3(L), g = [0, 1, 2].map(i => [Li[0][i], Li[1][i], Li[2][i]]);
  let list = hkls, B = L;
  if (C) {
    const conv = hkls.map(([h, k, l]) => { const G = [0, 1, 2].map(c => h * g[0][c] + k * g[1][c] + l * g[2][c]); return C.map(ci => ci[0] * G[0] + ci[1] * G[1] + ci[2] * G[2]); });
    if (conv.every(v => v.every(x => Math.abs(x - Math.round(x)) < 1e-4))) { list = conv.map(v => v.map(Math.round)); B = C; }
  }
  const a = norm(B[0]), b = norm(B[1]), cosg = (B[0][0] * B[1][0] + B[0][1] * B[1][1] + B[0][2] * B[1][2]) / (a * b);
  if (Math.abs(a - b) < 1e-4 * a && Math.abs(cosg + 0.5) < 1e-4) list = list.map(([h, k, l]) => [h, k, -h - k, l]);
  const fam = new Map();
  for (const v of list) {
    const key = v.map(Math.abs).sort((x, y) => x - y).join(",");
    const f = fam.get(key);
    if (!f) fam.set(key, { hkl: v, m: 1 });
    else { f.m++; if (cmpTuple(v, f.hkl) > 0) f.hkl = v; }
  }
  return [...fam.values()].sort((x, y) => y.m - x.m);
}
const cmpTuple = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i]; return 0; };
/** Indices as text, negative ones with a bar: [1, -1, 0] -> "11̄0"; indices above 9 separated by spaces. */
export function hklText(v) {
  const sp = v.some(x => Math.abs(x) > 9) ? " " : "";
  return v.map(x => (x < 0 ? `${-x}`.split("").map(c => c + "̅").join("") : String(x))).join(sp);
}

/** Profile on [range] in steps of step (deg): Gaussians of FWHM fwhm (deg) and height I at each peak, cut at 3 fwhm.
 *  Returns { x: Float64Array, y: Float64Array }. */
export function profile(peaks, { range = [0, 90], fwhm = 0.1, step = 0.02 } = {}) {
  const n = Math.max(2, Math.floor((range[1] - range[0]) / step + 1e-9) + 1), x = new Float64Array(n), y = new Float64Array(n), k = 4 * Math.LN2 / (fwhm * fwhm);
  for (let i = 0; i < n; i++) x[i] = range[0] + i * step;
  for (const p of peaks) {
    const i0 = Math.max(0, Math.ceil((p.tth - 3 * fwhm - range[0]) / step)), i1 = Math.min(n - 1, Math.floor((p.tth + 3 * fwhm - range[0]) / step));
    for (let i = i0; i <= i1; i++) { const d = x[i] - p.tth; y[i] += p.I * Math.exp(-k * d * d); }
  }
  return { x, y };
}

/**
 * One cell, ready to plot: scaled peaks (strongest 100), each with its families in the conventional cell C (optional),
 * and the profile scaled so that its highest point is 100. opts: lambda, range, fwhm (deg).
 */
export function xrdPattern(L, species, pos, { lambda = 1.54184, range = [10, 90], fwhm = 0.1, C = null } = {}) {
  const { peaks } = xrdPeaks(L, species, pos, { lambda, range });
  for (const p of peaks) p.fam = families(p.hkls, L, C);
  const pr = profile(peaks, { range, fwhm, step: Math.min(0.02, fwhm / 5) }), top = pr.y.reduce((m, v) => Math.max(m, v), 0) || 1;
  for (let i = 0; i < pr.y.length; i++) pr.y[i] *= 100 / top;
  return { peaks: peaks.map(p => ({ tth: p.tth, d: p.d, I: p.I, fam: p.fam, n: p.hkls.length })), x: pr.x, y: pr.y, lambda, range, fwhm };
}

/**
 * Patterns along a trajectory (data model of analysis.js): up to maxFrames frames evenly spaced, each with its own
 * cell; unscaled profiles on one grid (step max(0.02, fwhm / 4) deg). Returns { frames: frame indices, x,
 * map: Float32Array(frames x points), row f = frame frames[f], scaled so the highest value over all frames is 100
 * (frames stay comparable), avg: their mean scaled to a highest point of 100 }.
 */
export function xrdTrajectory(structure, traj, { lambda = 1.54184, range = [10, 90], fwhm = 0.1, maxFrames = 200 } = {}) {
  const nf = Math.min(traj.n, maxFrames), frames = [...new Set(Array.from({ length: nf }, (_, i) => Math.round(i * (traj.n - 1) / Math.max(1, nf - 1))))];
  const step = Math.max(0.02, fwhm / 4);
  let x = null, map = null;
  frames.forEach((f, r) => {
    const L = traj.cells ? [0, 1, 2].map(i => Array.from(traj.cells.subarray(9 * f + 3 * i, 9 * f + 3 * i + 3))) : structure.lattice;
    const pos = traj.pos.subarray(3 * traj.nat * f, 3 * traj.nat * (f + 1));
    const { peaks } = xrdPeaks(L, structure.species, pos, { lambda, range, scaled: false });
    const pr = profile(peaks, { range, fwhm, step });
    if (!map) { x = pr.x; map = new Float32Array(frames.length * x.length); }
    map.set(pr.y, r * x.length);
  });
  const nx = x.length, avg = new Float64Array(nx);
  let top = 0;
  for (let i = 0; i < map.length; i++) top = Math.max(top, map[i]);
  for (let i = 0; i < map.length; i++) map[i] *= top ? 100 / top : 0;
  for (let r = 0; r < frames.length; r++) for (let i = 0; i < nx; i++) avg[i] += map[r * nx + i] / frames.length;
  const atop = avg.reduce((m, v) => Math.max(m, v), 0) || 1;
  for (let i = 0; i < nx; i++) avg[i] *= 100 / atop;
  return { frames, x, map, avg, lambda, range, fwhm };
}
