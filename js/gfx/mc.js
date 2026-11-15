/**
 * Marching cubes on a regular grid (runs in the worker).
 *
 * grid = { data: Float32Array, x index fastest (i + nx (j + ny k)), size: [nx, ny, nz], vox: [va, vb, vc] (Cartesian step
 * per index, A), origin: [3] (A) }; grid point (i, j, k) sits at r = origin + J (i, j, k), J = [va vb vc] (columns).
 * Surface f = iso. A vertex sits on each grid edge whose end values straddle iso, at t = (iso - f0) / (f1 - f0) by
 * linear interpolation; it is created once and shared by the four cells around the edge (edge-id caches over two
 * z planes). Normal n = -grad f / |grad f| (toward lower values), with grad f = J^-T g and g the index-space gradient
 * from central differences (one-sided on the grid faces), interpolated with the same t.
 * Output { positions: Float32Array(3V), normals: Int16Array(4V) (snorm16 x, y, z, 0), indices: Uint32Array(3T),
 * truncated } with at most maxTris triangles.
 */
import { EDGE, COUNT, TRI } from "./mctables.js";

/** Growable typed array. */
class Buf {
  constructor(T, n) { this.T = T; this.a = new T(n); this.n = 0; }
  need(k) { if (this.n + k > this.a.length) { const b = new this.T(Math.max(2 * this.a.length, this.n + k)); b.set(this.a); this.a = b; } }
  done() { return this.a.slice(0, this.n); }
}

function inv3(m) {
  const d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d],
  ];
}

export function marchingCubes(grid, iso, maxTris = 6e6) {
  const f = grid.data, [nx, ny, nz] = grid.size, [va, vb, vc] = grid.vox, o = grid.origin;
  const J = [[va[0], vb[0], vc[0]], [va[1], vb[1], vc[1]], [va[2], vb[2], vc[2]]], Ji = inv3(J);
  const sxy = nx * ny, at = (i, j, k) => f[i + nx * j + sxy * k];
  const pos = new Buf(Float32Array, 1 << 15), nrm = new Buf(Int16Array, 1 << 15), idx = new Buf(Uint32Array, 1 << 16);
  let nv = 0, nt = 0, truncated = false;
  // edge-id caches: x and y edges of the bottom (0) and top (1) planes, z edges of the current slab
  let X0 = new Int32Array(sxy).fill(-1), X1 = new Int32Array(sxy).fill(-1);
  let Y0 = new Int32Array(sxy).fill(-1), Y1 = new Int32Array(sxy).fill(-1);
  const Z = new Int32Array(sxy);
  const g0 = [0, 0, 0], g1 = [0, 0, 0];
  const grad = (i, j, k, g) => {
    const i0 = i > 0 ? i - 1 : i, i1 = i < nx - 1 ? i + 1 : i, j0 = j > 0 ? j - 1 : j, j1 = j < ny - 1 ? j + 1 : j;
    const k0 = k > 0 ? k - 1 : k, k1 = k < nz - 1 ? k + 1 : k;
    g[0] = i1 > i0 ? (at(i1, j, k) - at(i0, j, k)) / (i1 - i0) : 0;
    g[1] = j1 > j0 ? (at(i, j1, k) - at(i, j0, k)) / (j1 - j0) : 0;
    g[2] = k1 > k0 ? (at(i, j, k1) - at(i, j, k0)) / (k1 - k0) : 0;
  };
  const vertex = (i, j, k, ax) => {
    const di = ax === 0 ? 1 : 0, dj = ax === 1 ? 1 : 0, dk = ax === 2 ? 1 : 0;
    const f0 = at(i, j, k), f1 = at(i + di, j + dj, k + dk), d = f1 - f0;
    const t = d !== 0 ? Math.min(1, Math.max(0, (iso - f0) / d)) : 0.5;
    const x = i + t * di, y = j + t * dj, z = k + t * dk;
    pos.need(3);
    pos.a[pos.n++] = o[0] + J[0][0] * x + J[0][1] * y + J[0][2] * z;
    pos.a[pos.n++] = o[1] + J[1][0] * x + J[1][1] * y + J[1][2] * z;
    pos.a[pos.n++] = o[2] + J[2][0] * x + J[2][1] * y + J[2][2] * z;
    grad(i, j, k, g0); grad(i + di, j + dj, k + dk, g1);
    const gi = [g0[0] + t * (g1[0] - g0[0]), g0[1] + t * (g1[1] - g0[1]), g0[2] + t * (g1[2] - g0[2])];
    let cx = -(Ji[0][0] * gi[0] + Ji[1][0] * gi[1] + Ji[2][0] * gi[2]);
    let cy = -(Ji[0][1] * gi[0] + Ji[1][1] * gi[1] + Ji[2][1] * gi[2]);
    let cz = -(Ji[0][2] * gi[0] + Ji[1][2] * gi[1] + Ji[2][2] * gi[2]);
    const L = Math.hypot(cx, cy, cz) || 1;
    nrm.need(4);
    nrm.a[nrm.n++] = Math.round(32767 * cx / L); nrm.a[nrm.n++] = Math.round(32767 * cy / L); nrm.a[nrm.n++] = Math.round(32767 * cz / L); nrm.a[nrm.n++] = 0;
    return nv++;
  };
  const ids = new Int32Array(12);
  for (let k = 0; k < nz - 1 && !truncated; k++) {
    Z.fill(-1);
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const b = i + nx * j + sxy * k;
        let c = 0;
        if (f[b] < iso) c |= 1;
        if (f[b + 1] < iso) c |= 2;
        if (f[b + nx] < iso) c |= 4;
        if (f[b + nx + 1] < iso) c |= 8;
        if (f[b + sxy] < iso) c |= 16;
        if (f[b + sxy + 1] < iso) c |= 32;
        if (f[b + sxy + nx] < iso) c |= 64;
        if (f[b + sxy + nx + 1] < iso) c |= 128;
        const nTri = COUNT[c];
        if (!nTri) continue;
        if (nt + nTri > maxTris) { truncated = true; break; }
        for (let q = 0; q < 3 * nTri; q++) {
          const e = TRI[16 * c + q];
          const ax = EDGE[4 * e], ox = EDGE[4 * e + 1], oy = EDGE[4 * e + 2], oz = EDGE[4 * e + 3];
          const key = i + ox + nx * (j + oy);
          let arr;
          if (ax === 2) arr = Z; else if (ax === 0) arr = oz ? X1 : X0; else arr = oz ? Y1 : Y0;
          let v = arr[key];
          if (v < 0) { v = vertex(i + ox, j + oy, k + oz, ax); arr[key] = v; }
          ids[q] = v;
        }
        idx.need(3 * nTri);
        for (let q = 0; q < 3 * nTri; q++) idx.a[idx.n++] = ids[q];
        nt += nTri;
      }
      if (truncated) break;
    }
    // the top plane of this slab is the bottom plane of the next one
    let t = X0; X0 = X1; X1 = t; X1.fill(-1);
    t = Y0; Y0 = Y1; Y1 = t; Y1.fill(-1);
  }
  return { positions: pos.done(), normals: nrm.done(), indices: idx.done(), triangles: nt, truncated };
}
