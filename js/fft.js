// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Complex discrete Fourier transforms on separate real/imaginary Float64Arrays.
 *
 *   X_j = sum_{k=0}^{n-1} x_k exp(sign * 2 pi i j k / n),  sign = +1 (inverse, unnormalised) or -1 (forward).
 *
 * Powers of two use an iterative radix-2 transform. Other lengths use Bluestein's chirp-z identity
 *   j k = (j^2 + k^2 - (j - k)^2) / 2  =>  X_j = w_j sum_k (x_k w_k) conj(w_{j-k}),  w_k = exp(sign i pi k^2 / n),
 * evaluated as a circular convolution of length m = 2^p >= 2n - 1 with radix-2 transforms.
 * fft3 applies the 1D transform along x, y and z of an array with x as the fastest index, skipping lines that are
 * identically zero (the plane-wave sphere fills only part of the box).
 */

const radixCache = new Map();
const blueCache = new Map();

function radixTables(n) {
  let t = radixCache.get(n);
  if (t) return t;
  const rev = new Uint32Array(n), bits = Math.round(Math.log2(n));
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0, x = i; b < bits; b++, x >>= 1) r = (r << 1) | (x & 1);
    rev[i] = r;
  }
  const cos = new Float64Array(n >> 1), sin = new Float64Array(n >> 1);
  for (let i = 0; i < n >> 1; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
  t = { rev, cos, sin };
  radixCache.set(n, t);
  return t;
}

/** In-place radix-2 transform; n must be a power of two. */
export function fftRadix2(re, im, sign) {
  const n = re.length;
  if (n <= 1) return;
  const { rev, cos, sin } = radixTables(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0, k = 0; j < half; j++, k += step) {
        const wr = cos[k], wi = sign * sin[k], a = i + j, b = a + half;
        const tr = wr * re[b] - wi * im[b], ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}

const isPow2 = n => (n & (n - 1)) === 0;

function bluePlan(n, sign) {
  const key = n * 2 + (sign > 0 ? 1 : 0);
  let p = blueCache.get(key);
  if (p) return p;
  let m = 1;
  while (m < 2 * n - 1) m <<= 1;
  const wr = new Float64Array(n), wi = new Float64Array(n);
  for (let k = 0; k < n; k++) { const a = Math.PI * ((k * k) % (2 * n)) / n; wr[k] = Math.cos(a); wi[k] = sign * Math.sin(a); }
  const br = new Float64Array(m), bi = new Float64Array(m);
  br[0] = wr[0]; bi[0] = -wi[0];
  for (let k = 1; k < n; k++) { br[k] = br[m - k] = wr[k]; bi[k] = bi[m - k] = -wi[k]; }
  fftRadix2(br, bi, -1);
  p = { m, wr, wi, br, bi, ar: new Float64Array(m), ai: new Float64Array(m) };
  blueCache.set(key, p);
  return p;
}

/** In-place transform of any length. */
export function fft1(re, im, sign) {
  const n = re.length;
  if (isPow2(n)) return fftRadix2(re, im, sign);
  const { m, wr, wi, br, bi, ar, ai } = bluePlan(n, sign);
  ar.fill(0); ai.fill(0);
  for (let k = 0; k < n; k++) { ar[k] = re[k] * wr[k] - im[k] * wi[k]; ai[k] = re[k] * wi[k] + im[k] * wr[k]; }
  fftRadix2(ar, ai, -1);
  for (let i = 0; i < m; i++) { const r = ar[i] * br[i] - ai[i] * bi[i]; ai[i] = ar[i] * bi[i] + ai[i] * br[i]; ar[i] = r; }
  fftRadix2(ar, ai, 1);
  for (let k = 0; k < n; k++) { const r = ar[k] / m, s = ai[k] / m; re[k] = r * wr[k] - s * wi[k]; im[k] = r * wi[k] + s * wr[k]; }
}

/** In-place 3D transform of arrays of length nx*ny*nz, index i + nx (j + ny k). */
export function fft3(re, im, [nx, ny, nz], sign) {
  const pass = (n, count, start, stride) => {
    const tr = new Float64Array(n), ti = new Float64Array(n);
    for (let l = 0; l < count; l++) {
      const s0 = start(l);
      let nz0 = false;
      for (let i = 0, p = s0; i < n; i++, p += stride) { tr[i] = re[p]; ti[i] = im[p]; if (tr[i] !== 0 || ti[i] !== 0) nz0 = true; }
      if (!nz0) continue;
      fft1(tr, ti, sign);
      for (let i = 0, p = s0; i < n; i++, p += stride) { re[p] = tr[i]; im[p] = ti[i]; }
    }
  };
  pass(nx, ny * nz, l => l * nx, 1);
  pass(ny, nx * nz, l => (l % nx) + nx * ny * Math.floor(l / nx), nx);
  pass(nz, nx * ny, l => l, nx * ny);
}

/** Smallest power of two >= n. */
export function nextPow2(n) { let m = 1; while (m < n) m <<= 1; return m; }
