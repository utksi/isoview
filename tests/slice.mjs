// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Tests of js/slice.js: plane geometry (cross-section polygons of known shape and area), sampling accuracy on an
 * analytic periodic field, open grids, contours of a circle, colour-map end points; colour-bar ticks (linear, signed,
 * log, factored powers of ten) at the positions the colour scale gives their values; the figure layout (aspect kept,
 * inside the padding, beside the bar, centred as a pair or the figure itself; kept clear of an inset with symmetric
 * padding); the figure and colour bar drawn on a recording context.
 * Run: node tests/slice.mjs
 */
import { slicePlane, sampleSlice, contours, colormap, barTicks, figureLayout, layoutClear, drawFigure, drawColorBar, colorBarWidth, scaler } from "../js/slice.js";
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

// colour-bar ticks: each tick's position is the colour-scale position of its value
const at = (mode, lo, hi) => { const t = barTicks(mode, lo, hi), sc = scaler(mode, lo, hi); return { ...t, err: Math.max(0, ...t.ticks.map(k => Math.abs(k.x - sc(k.v)))) }; };
let b = at("lin", 0, 0.47);
ok(b.ticks.map(t => t.text).join() === "0.0,0.1,0.2,0.3,0.4" && b.mult === "" && b.err < 1e-12, `linear [0, 0.47]: ${b.ticks.map(t => t.text)}`);
b = at("signed", -0.8, 1.3);
ok(b.ticks.map(t => t.text).join() === "−1.0,−0.5,0.0,0.5,1.0" && b.err < 1e-12 && b.ticks.find(t => t.v === 0).x === 0.5, `signed (m = 1.3): ${b.ticks.map(t => t.text)}, zero at the middle`);
b = at("log", 2.5e-4, 0.25);
ok(b.ticks.map(t => t.text).join() === "10⁻³,10⁻²,10⁻¹" && b.err < 1e-12, `log [2.5e-4, 0.25]: ${b.ticks.map(t => t.text)}`);
b = at("log", 1e-9, 10);
ok(b.ticks.length === 6 && b.ticks[0].text === "10⁻⁹" && b.ticks[1].text === "10⁻⁷", `log over 11 decades: every second (${b.ticks.map(t => t.text)})`);
b = at("lin", 1e-5, 4.6e-5);
ok(b.mult === "×10⁻⁵" && b.ticks.map(t => t.text).join() === "1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5" && b.err < 1e-12, `small values factored: ${b.mult} x ${b.ticks.map(t => t.text)}`);
b = at("lin", -12000, 31000);
ok(b.mult === "×10⁴" && b.ticks[0].text === "−1" && b.ticks.at(-1).text === "3", `large values factored: ${b.mult} x ${b.ticks.map(t => t.text)}`);
// layout: a 2:1 figure in 1000 x 600 with padding 20 and a 100 px bar block
let Lo = figureLayout(1000, 600, 2, [20, 20, 20, 20], 100, 0);
ok(near(Lo.w / Lo.h, 2, 1e-12) && near(Lo.w, 860, 1e-9) && near(Lo.x, 20, 1e-9) && near(Lo.bx, 880, 1e-9) && Lo.y >= 20 - 1e-9 && Lo.y + Lo.h <= 580 + 1e-9 && near(Lo.bh, 0.8 * Lo.h, 1e-9),
  `layout, width-limited: figure ${Lo.w} x ${Lo.h} at (${Lo.x}, ${Lo.y}), bar at ${Lo.bx}`);
Lo = figureLayout(1000, 600, 0.5, [40, 20, 20, 20], 0, 0);
ok(near(Lo.h, 540, 1e-9) && near(Lo.w, 270, 1e-9) && near(Lo.x + Lo.w / 2, 500, 1e-9) && near(Lo.y, 40, 1e-9), `layout, height-limited and centred: ${Lo.w} x ${Lo.h} at (${Lo.x}, ${Lo.y})`);
Lo = figureLayout(800, 200, 8, [10, 10, 10, 10], 60, 120);
ok(near(Lo.bh, 120, 1e-9) && Lo.by + Lo.bh <= 190 + 1e-9 && Lo.by >= 10 - 1e-9, `a flat figure keeps a readable bar: ${Lo.bh} px`);
// the figure itself centred (bar on its right), and kept clear of an inset at the bottom left while staying centred
Lo = figureLayout(1200, 800, 1, [76, 28, 76, 28], 90, 0, true);
ok(near(Lo.x + Lo.w / 2, 600, 1e-9) && near(Lo.y + Lo.h / 2, 400, 1e-9) && near(Lo.h, 648, 1e-9) && near(Lo.bx, Lo.x + Lo.w, 1e-9), `figure centred in the window: ${Lo.w} x ${Lo.h} at (${Lo.x}, ${Lo.y})`);
const inset = { x: 46, y: 464, w: 240, h: 240 }, hits = (L, R) => L.x < R.x + R.w && R.x < L.x + L.w && L.y < R.y + R.h && R.y < L.y + L.h;
ok(hits(Lo, inset), "unobstructed, the centred figure would run under a 240 px inset at the bottom left");
const Lc = layoutClear(1200, 800, 1, [76, 28, 76, 28], 90, 0, inset, 16);
ok(!hits(Lc, inset) && !hits({ x: Lc.bx, y: Lc.by, w: 90, h: Lc.bh }, inset) && near(Lc.x + Lc.w / 2, 600, 1e-9) && near(Lc.y + Lc.h / 2, 400, 1e-9) && near(Lc.w, 2 * (600 - (46 + 240 + 16)), 1e-9),
  `kept clear of the inset and still centred: ${Lc.w.toFixed(0)} x ${Lc.h.toFixed(0)} at (${Lc.x.toFixed(0)}, ${Lc.y.toFixed(0)})`);
const mid = { x: 500, y: 300, w: 200, h: 200 }, Lm = layoutClear(1200, 800, 1, [76, 28, 76, 28], 90, 0, mid, 16);
ok(layoutClear(1200, 800, 1, [76, 28, 76, 28], 90, 0, { x: 1000, y: 20, w: 60, h: 60 }, 16).w === Lo.w && !hits(Lm, mid) && Lm.w > 300,
  `an inset out of the way changes nothing; one over the centre moves the figure into the free band (${Lm.w.toFixed(0)} px at x ${Lm.x.toFixed(0)})`);
// a panel at the right (the dock) and the inset at the bottom left: figure and bar clear both
const dock = { x: 828, y: 12, w: 332, h: 520 }, Ld = layoutClear(1200, 800, 1, [76, 28, 76, 28], 90, 0, [dock, inset], 16), barOf = L => ({ x: L.bx, y: L.by, w: 90, h: L.bh });
ok(!hits(Ld, dock) && !hits(barOf(Ld), dock) && !hits(Ld, inset) && Ld.w > 200, `clear of panel and inset: ${Ld.w.toFixed(0)} x ${Ld.h.toFixed(0)} at (${Ld.x.toFixed(0)}, ${Ld.y.toFixed(0)})`);
const wide = layoutClear(1600, 900, 1.6, [76, 28, 76, 28], 90, 0, [{ x: 1220, y: 12, w: 332, h: 800 }], 16);
ok(near(wide.x + wide.w / 2, 800, 1e-6) && wide.bx + 90 <= 1220 - 16 + 1e-6, `a wide window keeps the figure centred and the bar left of the panel: ${wide.w.toFixed(0)} px, bar ends at ${(wide.bx + 90).toFixed(0)}`);
const narrow = layoutClear(900, 800, 1.6, [76, 28, 76, 28], 90, 0, [{ x: 554, y: 12, w: 332, h: 780 }], 16);
ok(!hits(narrow, { x: 554, y: 12, w: 332, h: 780 }) && narrow.x + narrow.w / 2 < 450, `a narrow window moves the figure into the free band: ${narrow.w.toFixed(0)} px centred at ${(narrow.x + narrow.w / 2).toFixed(0)}`);
// drawing on a recording context: image first, contours, outline, atoms; bar, ticks, labels, title
const calls = [], rec = new Proxy({ measureText: t => ({ width: 6 * String(t).length }), createLinearGradient: () => ({ addColorStop() {} }) }, {
  get: (o, k) => (k in o ? o[k] : (...a) => calls.push([k, ...a])), set: (o, k, v) => { calls.push(["=" + k, v]); return true; } });
const vals = new Float32Array(16 * 16).map((_, i) => Math.hypot((i % 16) - 7.5, Math.floor(i / 16) - 7.5));
drawFigure(rec, 10, 20, 160, 160, { image: {}, values: vals, iw: 16, ih: 16, poly: [[0, 0], [1, 0], [1, 1], [0, 1]], levels: [3, 5], lines: true, atoms: [{ s: 0.5, r: 0.5, rad: 1, color: "#f00", w: 1 }], width: 8, k: 2 });
const names = calls.map(c => c[0]), di = names.indexOf("drawImage"), st = names.lastIndexOf("stroke"), ar = names.indexOf("arc");
ok(di >= 0 && names.indexOf("lineTo") > di && ar > names.indexOf("closePath") && calls[ar][3] === 20, `figure: image, then contours and outline, then atoms (radius 1 A = ${calls[ar] && calls[ar][3]} px at 20 px/A)`);
calls.length = 0;
const bt = barTicks("lin", 0, 1), bw = colorBarWidth(rec, 12, "sans-serif", bt, "density (e/Å³)");
drawColorBar(rec, 500, 100, 300, { fs: 12, font: "sans-serif", ink: "#000", bt, map: "viridis", title: "density (e/Å³)", k: 1 });
const texts = calls.filter(c => c[0] === "fillText").map(c => c[1]), fr = calls.find(c => c[0] === "fillRect");
ok(texts.join() === "0.0,0.2,0.4,0.6,0.8,1.0,density (e/Å³)" && fr[1] > 500 && fr[4] === 300 && bw > 12 * 4, `colour bar: labels ${texts.slice(0, 6)}, title; block width ${bw.toFixed(1)} px`);

console.log(`${pass}/${pass + fail} passed`);
if (fail) process.exitCode = 1;
