// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Tests of js/xrd.js (node tests/xrd.mjs).
 *
 * Reference (fixtures/xrd_pymatgen.json): pymatgen 2024.8.9 XRDCalculator.get_pattern on seven cells: Si conventional
 * and primitive, NaCl, rutile, wurtzite ZnO (Cu K alpha, 10-90 deg), a random monoclinic Ba-S-O cell (Mo K alpha,
 * 5-50 deg) and a random triclinic Fe-O-H cell (Co K alpha, 15-100 deg). Per peak: 2theta (deg), scaled intensity,
 * d (A), multiplicity, and for the five small cases the hkl families. Script: .dev/checks/xrd_ref.py.
 *
 * Checks: the same peaks (count, 2theta to 1e-7 deg, intensity to 1e-6 on the 0-100 scale, d to 1e-9 A, number of
 * hkl per peak, families and their names); Si in its primitive cell gives the conventional pattern and, with the
 * conventional cell for labels, the indices 111, 220, 311 with multiplicities 8, 12, 24; ZnO uses four indices; the
 * profile peaks at each strong reflection with height I and half height at fwhm / 2; a trajectory of identical
 * frames gives identical map rows and an average equal to the single pattern; a translated cell changes nothing;
 * random displacements in a 3x3x3 Si supercell weaken a high-angle Bragg peak more than a low-angle one.
 */
import fs from "node:fs";
import { xrdPeaks, families, hklText, profile, xrdPattern, xrdTrajectory, formFactor } from "../js/xrd.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) pass++; else { fail++; console.log("FAIL", m); } };
const REF = JSON.parse(fs.readFileSync(new URL("./fixtures/xrd_pymatgen.json", import.meta.url), "utf8"));

for (const c of REF) {
  const { peaks } = xrdPeaks(c.L, c.s, c.c.flat(), { lambda: c.wl, range: c.tr });
  let dt = 0, dI = 0, dd = 0, mult = 0, fam = 0;
  const n = Math.min(peaks.length, c.p.length);
  for (let i = 0; i < n; i++) {
    const p = peaks[i], r = c.p[i];
    dt = Math.max(dt, Math.abs(p.tth - r[0])); dI = Math.max(dI, Math.abs(p.I - r[1])); dd = Math.max(dd, Math.abs(p.d - r[2]));
    if (p.hkls.length !== r[3]) mult++;
    if (r[4]) {
      const f = families(p.hkls, c.L).map(q => `${q.hkl.join(",")}:${q.m}`).sort().join(" "), g = r[4].map(([h, m]) => `${h.join(",")}:${m}`).sort().join(" ");
      if (f !== g) fam++;
    }
  }
  ok(peaks.length === c.p.length && dt < 1e-7 && dI < 1e-6 && dd < 1e-9 && !mult && !fam,
    `${c.n}: ${peaks.length} peaks as pymatgen (${c.p.length}); max diff 2theta ${dt.toExponential(1)} deg, I ${dI.toExponential(1)}, d ${dd.toExponential(1)} A; hkl counts ${mult ? "differ" : "equal"}, families ${fam ? "differ" : "equal"}`);
}

// Si primitive: conventional labels
const a = 5.431, prim = REF.find(c => c.n === "Si-prim"), conv = REF.find(c => c.n === "Si-conv");
const pat = xrdPattern(prim.L, prim.s, prim.c.flat(), { lambda: prim.wl, range: prim.tr, C: conv.L });
const lab = pat.peaks.slice(0, 3).map(p => `${p.fam[0].hkl.join("")}x${p.fam[0].m}`).join(" ");
ok(lab === "111x8 220x12 311x24", `Si primitive cell, labels in the conventional cell: ${lab}`);
const zno = REF.find(c => c.n === "ZnO"), zp = xrdPattern(zno.L, zno.s, zno.c.flat(), { lambda: zno.wl, range: zno.tr });
ok(zp.peaks.every(p => p.fam[0].hkl.length === 4 && p.fam[0].hkl[2] === -p.fam[0].hkl[0] - p.fam[0].hkl[1]), `ZnO: Miller-Bravais indices (${zp.peaks.slice(0, 3).map(p => hklText(p.fam[0].hkl)).join(", ")})`);
ok(hklText([1, -1, 0]) === "11̅0" && hklText([10, 0, -2]) === "10 0 2̅", `index text: ${hklText([1, -1, 0])}, ${hklText([10, 0, -2])}`);
ok(Math.abs(formFactor("Si", 0) - 14) < 1e-12 && formFactor("Si", 0.25) < 14, `f(Si, s = 0) = Z = ${formFactor("Si", 0)}`);

// profile: Gaussian of height I and FWHM w
const pr = profile([{ tth: 30, I: 50 }], { range: [20, 40], fwhm: 0.2, step: 0.01 }), at = t => pr.y[Math.round((t - 20) / 0.01)];
ok(Math.abs(at(30) - 50) < 1e-9 && Math.abs(at(30.1) - 25) < 1e-6 && Math.abs(at(29.9) - 25) < 1e-6 && at(31) === 0, `profile: height ${at(30)}, half height at +-fwhm/2 (${at(30.1).toFixed(6)})`);
ok(Math.max(...pat.y) > 99.999 && Math.max(...pat.y) <= 100 + 1e-9, "pattern profile scaled to 100");

// translation invariance
const sh = conv.c.flat().map((x, i) => x + [0.37, -1.2, 2.9][i % 3]), p0 = xrdPeaks(conv.L, conv.s, conv.c.flat(), { lambda: conv.wl, range: conv.tr }).peaks, p1 = xrdPeaks(conv.L, conv.s, sh, { lambda: conv.wl, range: conv.tr }).peaks;
ok(p0.length === p1.length && p0.every((p, i) => Math.abs(p.I - p1[i].I) < 1e-9), "a translated crystal gives the same intensities");

// trajectory of identical frames (variable-cell layout)
const nat = conv.s.length, nF = 5, pos = new Float64Array(3 * nat * nF), cells = new Float64Array(9 * nF);
for (let f = 0; f < nF; f++) { pos.set(conv.c.flat(), 3 * nat * f); cells.set(conv.L.flat(), 9 * f); }
const tr = xrdTrajectory({ species: conv.s, lattice: conv.L }, { n: nF, nat, pos, cells }, { lambda: conv.wl, range: [20, 80], fwhm: 0.15 });
const nx = tr.x.length, same = [...Array(nF).keys()].every(r => tr.map.subarray(r * nx, (r + 1) * nx).every((v, i) => Math.abs(v - tr.map[i]) < 1e-4));
const single = profile(xrdPeaks(conv.L, conv.s, conv.c.flat(), { lambda: conv.wl, range: [20, 80] }).peaks, { range: [20, 80], fwhm: 0.15, step: Math.max(0.02, 0.15 / 4) });
const stop = Math.max(...single.y), dev = Math.max(...Array.from(tr.avg, (v, i) => Math.abs(v - 100 * single.y[i] / stop)));
ok(tr.frames.length === nF && same && dev < 1e-3, `trajectory of ${nF} identical frames: equal rows, average = the single pattern (max diff ${dev.toExponential(1)})`);
// thermal-like disorder lowers high-angle Bragg intensity relative to low angle (Debye-Waller from displacements)
const rnd = (() => { let s = 12345; return () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5); })();
const big = [], bigS = [], A = 3 * a;
for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) for (const p of conv.c) { big.push(p[0] + i * a, p[1] + j * a, p[2] + k * a); bigS.push("Si"); }
const Lb = [[A, 0, 0], [0, A, 0], [0, 0, A]], shaken = big.map(x => x + 0.25 * rnd());
const I0 = xrdPeaks(Lb, bigS, big, { lambda: 1.54184, range: [20, 90], scaled: false }).peaks, I1 = xrdPeaks(Lb, bigS, shaken, { lambda: 1.54184, range: [20, 90], scaled: false }).peaks;
const near = (P, t) => P.filter(p => Math.abs(p.tth - t) < 0.05).reduce((s, p) => s + p.I, 0);
const strong = I0.filter(p => p.I > 0.05 * Math.max(...I0.map(q => q.I))), lo = strong[0].tth, hi = strong[strong.length - 1].tth;
const rlo = near(I1, lo) / near(I0, lo), rhi = near(I1, hi) / near(I0, hi);
ok(rlo < 1 && rhi < rlo, `displaced atoms: the Bragg peak at ${lo.toFixed(2)} deg keeps ${(100 * rlo).toFixed(1)}%, at ${hi.toFixed(2)} deg ${(100 * rhi).toFixed(1)}% (Debye-Waller-like fall with angle)`);

console.log(`${pass}/${pass + fail} passed`);
if (fail) process.exitCode = 1;
