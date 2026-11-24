// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Tests of js/plot.js: nice tick steps and minor ticks, a tight x range, one path per series, the legend in the
 * corner with the fewest data points, stacked panels with x tick labels only below the last, italic marks, the true
 * minus sign and CSV. Run: node tests/plot.mjs
 */
import { niceTicks, chart, lineChart, toCSV } from "../js/plot.js";

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) pass++; else { fail++; console.log("FAIL", m); } };

let t = niceTicks(0, 8);
ok(t.step === 2 && t.minor === 0.5 && t.ticks.join() === "0,2,4,6,8", `ticks of [0, 8]: step ${t.step}, minor ${t.minor}, ${t.ticks}`);
t = niceTicks(0.013, 0.047);
ok(Math.abs(t.step - 0.005) < 1e-15 && Math.abs(t.minor - 0.001) < 1e-15, `ticks of [0.013, 0.047]: step ${t.step}, minor ${t.minor}`);

// a curve rising to the right: the legend must not go top right (where the data is) but top left
const x = Array.from({ length: 101 }, (_, i) => i / 10), up = x.map(v => v * v), up2 = x.map(v => 0.8 * v * v);
let s = lineChart({ series: [{ label: "Li", x, y: up, color: "#f00" }, { label: "Si", x, y: up2, color: "#00f" }], xlabel: "*t* (ps)", ylabel: "MSD", W: 400, H: 300 });
const paths = s.match(/<path d="M[\d.]+ [\d.]+L/g) || [];
ok(paths.length === 2, `one path per series (${paths.length})`);
const leg = s.match(/<line x1="([\d.]+)"[^>]*stroke="#f00"/);
ok(leg && +leg[1] < 200, `legend top left for a curve rising to the right (x = ${leg && leg[1]})`);
ok(/<tspan font-style="italic">t<\/tspan> \(ps\)/.test(s), "italic marks in axis titles");
// tight x range: the first and last x tick labels are the data ends 0 and 10
ok(/>0<\/text>/.test(s) && />10<\/text>/.test(s) && !/>12<\/text>/.test(s), "x range is the data range");
// negative tick labels use U+2212
s = lineChart({ series: [{ label: "a", x, y: x.map(v => Math.sin(v)), color: "#000" }], W: 400, H: 300 });
ok(/>−1<\/text>/.test(s) && !/>-1<\/text>/.test(s), "minus sign in tick labels");
ok(!/<line x1="[\d.]+" x2="[\d.]+" y1="[\d.]+" y2="[\d.]+" stroke="#000"/.test(s), "no legend for a single series");

// two panels: x tick labels only once, two frames, two y titles
s = chart({ panels: [{ series: [{ label: "T", x, y: up, color: "#f00" }], ylabel: "T (K)" }, { series: [{ label: "V", x, y: up2, color: "#00f" }], ylabel: "V" }], xlabel: "frame", W: 400, H: 400, size: ["3.5in", "3.5in"] });
ok((s.match(/>10<\/text>/g) || []).length === 1 && (s.match(/<rect x=[^>]*fill="none"/g) || []).length === 2 && /T \(K\)/.test(s) && /width="3.5in"/.test(s), "stacked panels share one x axis; physical size");

const csv = toCSV([{ label: "*g*", x: [1, 2], y: [3, 4] }, { label: "b", x: [1], y: [5] }], "*r* (A)");
ok(csv === '"r (A) (g)","g","r (A) (b)","b"\n1,3,1,5\n2,4,,\n', `CSV:\n${csv}`);

console.log(`${pass}/${pass + fail} passed`);
if (fail) process.exitCode = 1;
