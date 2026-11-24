// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Tests of js/slice.js: plane geometry (cross-section polygons of known shape and area), sampling accuracy on an
 * analytic periodic field, open grids, contours of a circle, colour-map end points. Run: node tests/slice.mjs
 */
import { slicePlane, sampleSlice, contours, colormap } from "../js/slice.js";
import { fromParams, norm, cross } from "../js/lattice.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) pass++; else { fail++; console.log("FAIL", m); } };
const near = (a, b, t) => Math.abs(a - b) <= t;
/** Area of the cross-section polygon in A^2 (shoelace over the fractions, times |U x V|). */
const area = P => { let s = 0; const p = P.poly; for (let i = 0; i < p.length; i++) { const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length]; s += x0 * y1 - x1 * y0; } return 0.5 * Math.abs(s) * norm(cross(P.U, P.V)); };

const cube = [[4, 0, 0], [0, 4, 0], [0, 0, 4]];
let P = slicePlane(cube, [0, 0, 0], [0, 0, 1], 0.25);
ok(P.poly.length === 4 && near(P.d, 1, 1e-12) && near(area(P), 16, 1e-9) && near(P.depth, 4, 1e-12), `cube (001) t=0.25: square of 16 A^2 at d = 1 (${P.poly.length} corners, ${area(P)})`);
P = slicePlane(cube, [0, 0, 0], [1, 1, 1], 0.5);
ok(P.poly.length === 6 && near(area(P), 3 * Math.sqrt(3) * 16 / 4, 1e-9), `cube (111) t=0.5: regular hexagon of area 3 sqrt3 a^2 / 4 (${P.poly.length} corners, ${area(P).toFixed(6)})`);
P = slicePlane(cube, [0, 0, 0], [0, 0, 1], 0);
ok(P.poly.length === 4 && near(area(P), 16, 1e-9), "cube (001) t=0: the bottom face");
const hex = fromParams(3, 3, 5, 90, 90, 120);
P = slicePlane(hex, [0, 0, 0], [0, 0, 1], 0.5);
ok(P.poly.length === 4 && near(area(P), norm(cross(hex[0], hex[1])), 1e-9) && near(Math.abs(P.u[0] * hex[0][0] + P.u[1] * hex[0][1]) / 3, 1, 1e-9), `hexagonal (001): the a-b rhombus, u along a (${area(P).toFixed(6)})`);
P = slicePlane(hex, [0, 0, 0], [1, 1, 0], 0.5);
ok(P.poly.length === 4 && near(P.n[2], 0, 1e-12), `hexagonal (110): a rectangle containing c (${P.poly.length} corners)`);
let threw = false; try { slicePlane(cube, [0, 0, 0], [0, 0, 0], 0.5); } catch (e) { threw = true; }
ok(threw, "(000) is refused");

// sampling: f = cos(2 pi x) + 0.5 sin(4 pi y) + 0.2 cos(2 pi z) on a periodic 24^3 grid of a monoclinic cell
const L = fromParams(5, 6, 7, 90, 104, 90), n = 24, data = new Float32Array(n ** 3);
const fx = (x, y, z) => Math.cos(2 * Math.PI * x) + 0.5 * Math.sin(4 * Math.PI * y) + 0.2 * Math.cos(2 * Math.PI * z);
for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) data[i + n * (j + n * k)] = fx(i / n, j / n, k / n);
const f = { data, n: [n, n, n], span: L, origin: [0, 0, 0], periodic: true, closed: false };
P = slicePlane(L, [0, 0, 0], [0, 0, 1], 0.3);
let W = 64, H = 64, s = sampleSlice(f, P, W, H), err = 0, cnt = 0;
// on a (001) plane at t, fractional z = 0.3 everywhere; x, y from the pixel position
const Li = (await import("../js/lattice.js")).inv3(L), { cartToFrac } = await import("../js/lattice.js");
for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
  const v = s.values[i + W * j];
  if (Number.isNaN(v)) continue;
  const p = [0, 1, 2].map(c => P.o[c] + (i + 0.5) / W * P.U[c] + (j + 0.5) / H * P.V[c]), x = cartToFrac(Li, p);
  err = Math.max(err, Math.abs(v - fx(...x))); cnt++;
}
ok(cnt > 0.95 * W * H * area(P) / (P.width * P.height) && err < 4e-3, `monoclinic (001) slice of an analytic field: max error ${err.toExponential(2)} over ${cnt} pixels (Catmull-Rom on 24 points per period)`);
// shift by a quarter of a: the slice of the shifted field equals the original at x - 1/4
const s2 = sampleSlice(f, P, W, H, [0.25, 0, 0]);
err = 0;
for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
  const v = s2.values[i + W * j];
  if (Number.isNaN(v)) continue;
  const p = [0, 1, 2].map(c => P.o[c] + (i + 0.5) / W * P.U[c] + (j + 0.5) / H * P.V[c]), x = cartToFrac(Li, p);
  err = Math.max(err, Math.abs(v - fx(x[0] - 0.25, x[1], x[2])));
}
ok(err < 4e-3, `origin shift (1/4, 0, 0): max error ${err.toExponential(2)}`);
// exact at grid points: a plane through grid points (001) at t = 6/24 and a pixel grid matching the field grid
P = slicePlane(cube, [0, 0, 0], [0, 0, 1], 6 / 24);
const fc = { ...f, span: cube }, sg = sampleSlice(fc, { ...P, o: [P.o[0] - 2 / 24, P.o[1] - 2 / 24, P.o[2]] }, 24, 24);
let exact = 0;
for (let j = 0; j < 24; j++) for (let i = 0; i < 24; i++) { const v = sg.values[i + 24 * j]; if (!Number.isNaN(v)) exact = Math.max(exact, Math.abs(v - data[i + 24 * (j + 24 * 6)])); }
ok(exact < 1e-6, `at grid points the slice equals the grid values (max difference ${exact.toExponential(2)})`);

// open general grid (cube file): outside the grid is NaN, inside finite
const g = { data: new Float32Array(11 ** 3).fill(2), n: [11, 11, 11], span: cube, origin: [1, 1, 1], periodic: false, closed: false };
P = slicePlane([[6, 0, 0], [0, 6, 0], [0, 0, 6]], [0, 0, 0], [0, 0, 1], 0.5);
s = sampleSlice(g, P, 60, 60);
const inside = s.values[30 + 60 * 30], outside = s.values[2 + 60 * 2];
ok(near(inside, 2, 1e-6) && Number.isNaN(outside), `open grid: value inside ${inside}, outside ${outside}`);

// contours of r - 20 on a 64 x 64 image: every segment end within 0.05 px of the circle of radius 20 about (32, 32)
W = 64; H = 64;
const img = new Float32Array(W * H);
for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) img[i + W * j] = Math.hypot(i + 0.5 - 32, j + 0.5 - 32) - 20;
const seg = contours(img, W, H, 0);
let dev = 0;
for (let q = 0; q < seg.length; q += 2) dev = Math.max(dev, Math.abs(Math.hypot(seg[q] - 32, seg[q + 1] - 32) - 20));
ok(seg.length / 4 > 100 && dev < 0.05, `circle contour: ${seg.length / 4} segments, max radial deviation ${dev.toFixed(4)} px`);

const v0 = colormap("viridis", 0), v1 = colormap("viridis", 1), r5 = colormap("rdbu", 0.5);
ok(Math.abs(v0[0] - 68) <= 4 && Math.abs(v0[2] - 84) <= 4 && Math.abs(v1[0] - 253) <= 4 && Math.abs(v1[1] - 231) <= 4 && Math.abs(v1[2] - 37) <= 5 && r5.join() === "247,247,247",
  `colour maps (the viridis fit is within 4 of the matplotlib table): viridis(0) ${v0}, viridis(1) ${v1}, RdBu(0.5) ${r5}`);

console.log(`${pass}/${pass + fail} passed`);
if (fail) process.exitCode = 1;
