// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * VASP WAVECAR reader: band energies, occupations and real-space orbital densities, read lazily from a File.
 *
 * File layout (fixed-length records of rl bytes, little-endian float64 headers):
 *   record 0: rl, nspin, tag (45200: complex64 coefficients, 45210: complex128)
 *   record 1: nk, nbands, ENCUT (eV), lattice a, b, c (Angstrom, 9 values)
 *   record 2 + s nk (nb + 1) + k (nb + 1):        nplw, k (fractional, 3), then (Re eps, Im eps, occ) per band
 *   that record + 1 + b:                          nplw coefficients of band b (0-based)
 * A complete file holds 2 + nspin nk (nb + 1) records; open() refuses a file that ends before the last one starts.
 * Plane waves: G = n . B (B rows = 2 pi inv(A)^T), kept if |k + G|^2 / C < ENCUT with C = 2 m_e / hbar^2 =
 * 0.262465831 1/(eV A^2); order n3 outermost, n2, n1 innermost, each running 0, 1, ..., +m, -m, ..., -1.
 * Build types, told apart by the stored plane-wave count:
 *   standard (vasp_std): the full sphere;  non-collinear (vasp_ncl): two spinor components of one sphere each;
 *   Gamma-only (vasp_gam, k = 0): half the sphere. 'x' half (VASP 5.4 and later): n1 > 0, or n1 = 0 and n2 > 0,
 *   or n1 = n2 = 0 and n3 >= 0. 'z' half (older -DNGZhalf builds): the same with n3 and n1 exchanged.
 *   The other half follows from c(-G) = conj c(G), and the stored G != 0 coefficients carry a factor sqrt 2,
 *   which is removed.
 * Orbital: psi(r_j) = sum_G c_G exp(i (k + G) . r_j); |psi|^2 equals |u_k|^2, the periodic part. The grid has
 * N_i >= 2 m_i + 1 points per axis (exact psi at the grid points), a power of two, raised to a step <= h = 0.1 A while
 * the grid stays within about 4 M points (a finer grid is still exact: it only samples the same sum more densely). Output densities are normalised to one
 * electron over the cell (1/A^3); the signed Gamma-only orbital to int psi^2 dV = 1 (A^-3/2). Only the smooth
 * part of the PAW orbital is available (no augmentation inside the spheres).
 */
import { fft3, nextPow2 } from "./fft.js?v=31";
import { det3, inv3 } from "./lattice.js?v=31";

const C = 0.262465831;

export class Wavecar {
  /** read(offset, length) -> Promise<ArrayBuffer>. */
  static async open(read) {
    const w = new Wavecar();
    w.read = read;
    const h0 = new Float64Array(await read(0, 24));
    w.rl = Math.round(h0[0]); w.nspin = Math.round(h0[1]); w.tag = Math.round(h0[2]);
    if (w.tag !== 45200 && w.tag !== 45210) throw new Error(`WAVECAR format tag ${w.tag} is not supported (45200 and 45210 are)`);
    const h1 = new Float64Array(await read(w.rl, 96));
    w.nk = Math.round(h1[0]); w.nb = Math.round(h1[1]); w.encut = h1[2];
    w.lattice = [[h1[3], h1[4], h1[5]], [h1[6], h1[7], h1[8]], [h1[9], h1[10], h1[11]]];
    w.volume = Math.abs(det3(w.lattice));
    // WHY: a cut-off file (a download that stopped early) reads short past its end, which would give NaN orbitals
    // for the later spin channel; the last record must exist (it may be shorter than rl, so only its start is tested)
    const nrec = w.rec(w.nspin - 1, w.nk - 1) + w.nb + 1;
    if ((await read((nrec - 1) * w.rl, 8)).byteLength < 8)
      throw new Error(`This WAVECAR is incomplete: it ends before its last band (a complete one has about ${(nrec * w.rl).toLocaleString("en")} bytes). It was probably cut off while downloading or copying.`);
    w.kpts = []; w.nplw = []; w.eig = []; w.occ = [];
    for (let s = 0; s < w.nspin; s++) {
      w.nplw.push([]); w.eig.push([]); w.occ.push([]);
      for (let k = 0; k < w.nk; k++) {
        const b = new Float64Array(await read(w.rec(s, k) * w.rl, 8 * (4 + 3 * w.nb)));
        w.nplw[s].push(Math.round(b[0]));
        if (s === 0) w.kpts.push([b[1], b[2], b[3]]);
        const e = new Float64Array(w.nb), o = new Float64Array(w.nb);
        for (let i = 0; i < w.nb; i++) { e[i] = b[4 + 3 * i]; o[i] = b[6 + 3 * i]; }
        w.eig[s].push(e); w.occ[s].push(o);
      }
    }
    w.gcache = new Map();
    w.type = w.detectType(0);
    return w;
  }

  rec(s, k) { return 2 + s * this.nk * (this.nb + 1) + k * (this.nb + 1); }

  /** Integer G triplets (Int32Array, 3 per G) for k-point index k in the given mode ('std', 'gamx', 'gamz'). */
  gvectors(k, mode) {
    const key = k + mode;
    if (this.gcache.has(key)) return this.gcache.get(key);
    const A = this.lattice, Ai = inv3(A);
    const B = [0, 1, 2].map(i => [0, 1, 2].map(j => 2 * Math.PI * Ai[j][i]));
    const kf = this.kpts[k], gmax = Math.sqrt(this.encut * C);
    const m = A.map(a => Math.floor(gmax * Math.hypot(...a) / (2 * Math.PI)) + 2);
    const ord = n => { const o = []; for (let i = 0; i <= n; i++) o.push(i); for (let i = -n; i < 0; i++) o.push(i); return o; };
    const nonneg = n => { const o = []; for (let i = 0; i <= n; i++) o.push(i); return o; };
    const o1 = mode === "gamx" ? nonneg(m[0]) : ord(m[0]);
    const o3 = mode === "gamz" ? nonneg(m[2]) : ord(m[2]);
    const o2 = ord(m[1]);
    const out = [];
    for (const n3 of o3) for (const n2 of o2) for (const n1 of o1) {
      if (mode === "gamx" && n1 === 0 && (n2 < 0 || (n2 === 0 && n3 < 0))) continue;
      if (mode === "gamz" && n3 === 0 && (n2 < 0 || (n2 === 0 && n1 < 0))) continue;
      const f0 = kf[0] + n1, f1 = kf[1] + n2, f2 = kf[2] + n3;
      const gx = f0 * B[0][0] + f1 * B[1][0] + f2 * B[2][0];
      const gy = f0 * B[0][1] + f1 * B[1][1] + f2 * B[2][1];
      const gz = f0 * B[0][2] + f1 * B[1][2] + f2 * B[2][2];
      if ((gx * gx + gy * gy + gz * gz) / C < this.encut) out.push(n1, n2, n3);
    }
    const g = Int32Array.from(out);
    this.gcache.set(key, g);
    return g;
  }

  /** 'std', 'ncl' or 'gam' from the plane-wave count at k-point k (spin 0). */
  detectType(k) {
    const np = this.nplw[0][k], ns = this.gvectors(k, "std").length / 3;
    if (np === ns) return "std";
    if (np === 2 * ns) return "ncl";
    const kz = this.kpts[k].every(x => Math.abs(x) < 1e-8);
    if (kz && np === this.gvectors(k, "gamx").length / 3) return "gam";
    throw new Error(`WAVECAR: ${np} stored plane waves do not match the cutoff sphere (${ns} for a standard build)`);
  }

  /** Lowest-occupation index pairs per spin at k: { homo, lumo } (0-based), from occupations 0.5 split. */
  frontier(s, k) {
    const o = this.occ[s][k];
    let homo = -1;
    for (let b = 0; b < this.nb; b++) if (o[b] > 0.5) homo = b;
    return { homo, lumo: homo + 1 < this.nb ? homo + 1 : -1 };
  }

  /**
   * Orbital field of band b (0-based), spin s, k-point k.
   * opts.signed: return psi (Gamma-only build) instead of |psi|^2; opts.half: 'x' or 'z' (Gamma-only half sphere);
   * opts.h, opts.maxPts: target grid step (A) and point budget (file header).
   * Returns { n, n0 (the smallest exact grid), data (Float32Array, x fastest), signed, maxImag }.
   */
  async orbital(s, k, b, opts = {}) {
    const np = this.nplw[s][k], bytes = this.tag === 45200 ? 8 : 16;
    const buf = await this.read((this.rec(s, k) + 1 + b) * this.rl, np * bytes);
    if (buf.byteLength < np * bytes) throw new Error(`band ${b + 1}${this.nspin === 2 ? (s ? " ↓" : " ↑") : ""} lies past the end of the file: the WAVECAR is incomplete`);
    const c = this.tag === 45200 ? new Float32Array(buf) : new Float64Array(buf);
    const type = this.type, gam = type === "gam";
    const mode = gam ? (opts.half === "z" ? "gamz" : "gamx") : "std";
    const G = this.gvectors(k, mode), npw = G.length / 3;
    let mx = [0, 0, 0];
    for (let i = 0; i < npw; i++) for (let a = 0; a < 3; a++) mx[a] = Math.max(mx[a], Math.abs(G[3 * i + a]));
    let n = mx.map(v => nextPow2(2 * v + 1));
    if (n[0] * n[1] * n[2] > 1 << 23) n = mx.map(v => 2 * v + 1);
    // WHY: a larger FFT box samples the same plane-wave orbital more finely (exact): grid step <= h (A), at most
    // maxPts points (halving the largest axis that is above its minimum until it fits)
    const h = opts.h || 0.1, maxPts = opts.maxPts || 4.2e6, n0 = n.slice();
    n = n.map((v, a) => Math.max(v, nextPow2(Math.ceil(Math.hypot(...this.lattice[a]) / h))));
    while (n[0] * n[1] * n[2] > maxPts) {
      let a = -1;
      for (let b = 0; b < 3; b++) if (n[b] > n0[b] && (a < 0 || n[b] > n[a])) a = b;
      if (a < 0) break;
      n[a] = Math.max(n0[a], n[a] >> 1);
    }
    const N = n[0] * n[1] * n[2];
    const idx = (i1, i2, i3) => ((i1 % n[0]) + n[0]) % n[0] + n[0] * ((((i2 % n[1]) + n[1]) % n[1]) + n[1] * (((i3 % n[2]) + n[2]) % n[2]));
    const comps = type === "ncl" ? 2 : 1;
    const rho = new Float64Array(N);
    let psiReal = null, maxImag = 0;
    for (let sp = 0; sp < comps; sp++) {
      const re = new Float64Array(N), im = new Float64Array(N), off = sp * npw;
      for (let i = 0; i < npw; i++) {
        let cr = c[2 * (off + i)], ci = c[2 * (off + i) + 1];
        const g1 = G[3 * i], g2 = G[3 * i + 1], g3 = G[3 * i + 2];
        if (gam && (g1 || g2 || g3)) { cr /= Math.SQRT2; ci /= Math.SQRT2; const q = idx(-g1, -g2, -g3); re[q] = cr; im[q] = -ci; }
        const p = idx(g1, g2, g3);
        re[p] = cr; im[p] = ci;
      }
      fft3(re, im, n, 1);
      for (let p = 0; p < N; p++) rho[p] += re[p] * re[p] + im[p] * im[p];
      if (gam && opts.signed) {
        let ar = 0, ai = 0;
        for (let p = 0; p < N; p++) { ar = Math.max(ar, Math.abs(re[p])); ai = Math.max(ai, Math.abs(im[p])); }
        maxImag = ai / (ar || 1);
        psiReal = re;
      }
    }
    const dV = this.volume / N;
    let sum = 0;
    for (let p = 0; p < N; p++) sum += rho[p];
    const out = new Float32Array(N);
    if (psiReal) { const f = 1 / Math.sqrt(sum * dV); for (let p = 0; p < N; p++) out[p] = psiReal[p] * f; }
    else { const f = 1 / (sum * dV); for (let p = 0; p < N; p++) out[p] = rho[p] * f; }
    return { n, n0, data: out, signed: !!psiReal, maxImag };
  }
}
