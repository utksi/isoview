// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Orbit camera: unit quaternion q (world -> view rotation), target point, distance, vertical field of view.
 *
 * View matrix   V = T(0, 0, -dist) R(q) T(-target)   (view space looks down -z, x right, y up).
 * Projection    perspective (fovy, aspect, near, far) or orthographic with half-height h = dist tan(fovy / 2), so both
 *               modes frame the same region at the target. clip z maps to [-1, 1] for WebGL and to [0, 1] for WebGPU
 *               (zRange). Matrices are column-major Float32Array(16).
 * Rotation by a drag (dx, dy) in CSS pixels: q <- rot(axis (dy, dx, 0) / |d|, k |d|) q, so the front of the scene
 * follows the pointer. Picking ray through a pixel: view-space origin 0 (perspective) or (x, y, 0) (orthographic),
 * mapped to world space by the inverse view transform. Tiles of a large image (tileProjection): the projection
 * followed by a scale and shift in clip space, so each tile shows its part of the same view.
 */

export function qmul(a, b) {
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
export function qaxis(ax, ang) { const s = Math.sin(ang / 2), n = Math.hypot(ax[0], ax[1], ax[2]) || 1; return [s * ax[0] / n, s * ax[1] / n, s * ax[2] / n, Math.cos(ang / 2)]; }
export function qnorm(q) { const n = Math.hypot(...q) || 1; return q.map(x => x / n); }
/** 3x3 rotation (rows) of unit quaternion q = (x, y, z, w). */
export function qmat(q) {
  const [x, y, z, w] = q;
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
}
/** Quaternion of a rotation given by its rows (an orthonormal, right-handed basis). */
export function qfromRows(R) {
  const [m11, m12, m13] = R[0], [m21, m22, m23] = R[1], [m31, m32, m33] = R[2], tr = m11 + m22 + m33;
  let s;
  if (tr > 0) { s = 0.5 / Math.sqrt(tr + 1); return qnorm([(m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s]); }
  if (m11 > m22 && m11 > m33) { s = 2 * Math.sqrt(1 + m11 - m22 - m33); return qnorm([0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s]); }
  if (m22 > m33) { s = 2 * Math.sqrt(1 + m22 - m11 - m33); return qnorm([(m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s]); }
  s = 2 * Math.sqrt(1 + m33 - m11 - m22);
  return qnorm([(m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s]);
}

export class Camera {
  constructor() { this.q = [0, 0, 0, 1]; this.target = [0, 0, 0]; this.dist = 10; this.fovy = 30 * Math.PI / 180; this.ortho = false; this.radius = 5; this.w = 1; this.h = 1; }
  /** Frame a sphere of the given centre and radius. */
  fit(center, radius) { this.target = center.slice(); this.radius = Math.max(radius, 0.5); this.dist = this.radius / Math.sin(this.fovy / 2) * 1.08; }
  rotateBy(dx, dy) { const d = Math.hypot(dx, dy); if (d > 0) this.q = qnorm(qmul(qaxis([dy, dx, 0], 0.0085 * d), this.q)); }
  zoomBy(factor) { this.dist = Math.min(Math.max(this.dist * factor, this.radius * 0.05), this.radius * 60); }
  /** World units per CSS pixel at the target depth. */
  unitsPerPixel() { return 2 * this.dist * Math.tan(this.fovy / 2) / this.h; }
  panBy(dx, dy) {
    const R = qmat(this.q), s = this.unitsPerPixel();
    for (let i = 0; i < 3; i++) this.target[i] -= s * (dx * R[0][i] - dy * R[1][i]);
  }
  view() {
    const R = qmat(this.q), t = this.target, m = new Float32Array(16);
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[4 * c + r] = R[r][c];
    for (let r = 0; r < 3; r++) m[12 + r] = -(R[r][0] * t[0] + R[r][1] * t[1] + R[r][2] * t[2]);
    m[14] -= this.dist; m[15] = 1;
    return m;
  }
  clip() { const n = Math.max(this.dist - 3 * this.radius, this.dist * 0.01), fa = this.dist + 3 * this.radius; return [n, fa]; }
  proj(zRange = "gl") {
    const [n, f] = this.clip(), a = this.w / this.h, m = new Float32Array(16);
    if (this.ortho) {
      const h = this.dist * Math.tan(this.fovy / 2);
      m[0] = 1 / (h * a); m[5] = 1 / h; m[15] = 1;
      if (zRange === "gl") { m[10] = -2 / (f - n); m[14] = -(f + n) / (f - n); } else { m[10] = -1 / (f - n); m[14] = -n / (f - n); }
    } else {
      const c = 1 / Math.tan(this.fovy / 2);
      m[0] = c / a; m[5] = c; m[11] = -1;
      if (zRange === "gl") { m[10] = (f + n) / (n - f); m[14] = 2 * f * n / (n - f); } else { m[10] = f / (n - f); m[14] = n * f / (n - f); }
    }
    return m;
  }
  /** Screen position (CSS px, origin top left) and view-space depth of world point p; null if behind the camera. */
  project(p) {
    const V = this.view(), P = this.proj("gl");
    const v = [0, 1, 2, 3].map(r => V[r] * p[0] + V[4 + r] * p[1] + V[8 + r] * p[2] + V[12 + r]);
    const c = [0, 1, 2, 3].map(r => P[r] * v[0] + P[4 + r] * v[1] + P[8 + r] * v[2] + P[12 + r] * v[3]);
    if (c[3] <= 1e-9) return null;
    return { x: (c[0] / c[3] + 1) / 2 * this.w, y: (1 - c[1] / c[3]) / 2 * this.h, z: v[2] };
  }
  /** CSS pixels per world unit at view-space depth z (z < 0). */
  pixelsPerUnit(z) { const h = this.ortho ? this.dist * Math.tan(this.fovy / 2) : -z * Math.tan(this.fovy / 2); return this.h / (2 * h); }
  /** World-space ray { o, d } through CSS pixel (x, y). */
  ray(x, y) {
    const R = qmat(this.q), nx = 2 * x / this.w - 1, ny = 1 - 2 * y / this.h, th = Math.tan(this.fovy / 2), a = this.w / this.h;
    let ov, dv;
    if (this.ortho) { const h = this.dist * th; ov = [nx * h * a, ny * h, 0]; dv = [0, 0, -1]; }
    else { ov = [0, 0, 0]; dv = [nx * th * a, ny * th, -1]; const l = Math.hypot(...dv); dv = dv.map(v => v / l); }
    // world = R^T (v - (0, 0, -dist)) + target
    const toW = v => [0, 1, 2].map(i => R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * (v[2] + this.dist) + this.target[i]);
    const o = toW(ov), d = [0, 1, 2].map(i => R[0][i] * dv[0] + R[1][i] * dv[1] + R[2][i] * dv[2]);
    return { o, d };
  }
}

/** Index of the nearest sphere hit by ray { o, d } (centres Float32Array 3N, radii Float32Array N), or -1. */
export function pickSphere(ray, centers, radii) {
  const { o, d } = ray;
  let best = -1, bt = Infinity;
  for (let i = 0; i < radii.length; i++) {
    const cx = centers[3 * i] - o[0], cy = centers[3 * i + 1] - o[1], cz = centers[3 * i + 2] - o[2];
    const b = cx * d[0] + cy * d[1] + cz * d[2], c2 = cx * cx + cy * cy + cz * cz - b * b, r = radii[i];
    if (c2 > r * r) continue;
    const t = b - Math.sqrt(r * r - c2);
    if (t > 0 && t < bt) { bt = t; best = i; }
  }
  return best;
}

/**
 * Projection for one tile of a larger image: P (column-major 4x4) followed by clip x' = sx x + tx w, y' = sy y + ty w,
 * so that the pixel rectangle [px, px + tw) x [py, py + th) (origin top left) of a VW x VH image fills the tile's
 * viewport. With x0, x1 and y0, y1 the tile's edges in normalised device coordinates of the whole image:
 * sx = 2 / (x1 - x0), tx = -(x1 + x0) / (x1 - x0), and likewise for y. Returns a new Float32Array(16).
 */
export function tileProjection(P, px, py, tw, th, VW, VH) {
  const x0 = -1 + 2 * px / VW, x1 = -1 + 2 * (px + tw) / VW, y1 = 1 - 2 * py / VH, y0 = 1 - 2 * (py + th) / VH;
  const sx = 2 / (x1 - x0), tx = -(x1 + x0) / (x1 - x0), sy = 2 / (y1 - y0), ty = -(y1 + y0) / (y1 - y0), Q = Float32Array.from(P);
  for (let k = 0; k < 4; k++) { const w = P[4 * k + 3]; Q[4 * k] = sx * P[4 * k] + tx * w; Q[4 * k + 1] = sy * P[4 * k + 1] + ty * w; }
  return Q;
}
