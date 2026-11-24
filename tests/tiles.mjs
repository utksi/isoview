// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Tests of tileProjection (js/gfx/camera.js), used to draw large photos in tiles: a world point at pixel (X, Y) of the
 * whole VW x VH image lands at pixel (X - px, Y - py) of the tile at (px, py), for perspective and orthographic
 * cameras and both clip depth ranges; depth is unchanged; proj[1][1] times the tile height equals proj[1][1] of the
 * whole image times VH (the pixels-per-unit the sphere and bond shaders use). Run: node tests/tiles.mjs
 */
import { Camera, tileProjection } from "../js/gfx/camera.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) pass++; else { fail++; console.log("FAIL", m); } };
/** Pixel (origin top left) and NDC depth of world point p under view V and projection P on a w x h target. */
function pix(V, P, p, w, h) {
  const v = [0, 1, 2, 3].map(r => V[r] * p[0] + V[4 + r] * p[1] + V[8 + r] * p[2] + V[12 + r]);
  const c = [0, 1, 2, 3].map(r => P[r] * v[0] + P[4 + r] * v[1] + P[8 + r] * v[2] + P[12 + r] * v[3]);
  return [(c[0] / c[3] + 1) / 2 * w, (1 - c[1] / c[3]) / 2 * h, c[2] / c[3]];
}
for (const ortho of [false, true]) for (const z of ["gl", "gpu"]) {
  const cam = new Camera();
  cam.fit([1, 2, 3], 6); cam.q = [0.2, -0.3, 0.1, 0.927]; cam.ortho = ortho; cam.w = 1200; cam.h = 700;
  const V = cam.view(), P = cam.proj(z), VW = 4800, VH = 2800;
  let err = 0, derr = 0, n = 0;
  for (const [px, py, tw, th] of [[0, 0, 2048, 2048], [2048, 2048, 2048, 752], [4096, 0, 704, 2048]]) {
    const Q = tileProjection(P, px, py, tw, th, VW, VH);
    for (let i = 0; i < 200; i++) {
      const p = [1 + 8 * Math.sin(i * 1.7), 2 + 8 * Math.cos(i * 2.3), 3 + 8 * Math.sin(i * 0.9)];
      const a = pix(V, P, p, VW, VH), b = pix(V, Q, p, tw, th);
      err = Math.max(err, Math.abs(a[0] - px - b[0]), Math.abs(a[1] - py - b[1])); derr = Math.max(derr, Math.abs(a[2] - b[2])); n++;
    }
    ok(Math.abs(Q[5] * th - P[5] * VH) < 1e-3 * P[5] * VH, `${ortho ? "ortho" : "persp"} ${z}: proj[1][1] x tile height = proj[1][1] x image height`);
  }
  ok(err < 2e-3 && derr < 1e-6, `${ortho ? "ortho" : "persp"} ${z}: ${n} points, max pixel error ${err.toExponential(2)}, depth error ${derr.toExponential(2)}`);
}
console.log(`${pass}/${pass + fail} passed`);
if (fail) process.exitCode = 1;
