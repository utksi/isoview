/**
 * Tests of the file readers and numerics (node tests/run.mjs). Each check states the exact or physical value it
 * expects. Optional checks on real VASP output run when the paths are given in the environment:
 *   ISOVIEW_CHGCAR  spin-polarised CHGCAR: int rho = NELECT (ISOVIEW_NELECT), int m = ISOVIEW_MAG
 *   ISOVIEW_WAVECAR Gamma-only WAVECAR (with ISOVIEW_BAND = "band:spin:Q", band 1-based, spin 0/1, and
 *                   ISOVIEW_CENTER = "x,y,z" Cartesian A): Q = charge of |psi|^2 within 2.5 A of the centre, compared
 *                   with an independent evaluation (tools/orbital_density.py), tolerance 0.02
 *   ISOVIEW_POSCAR  any POSCAR/CONTCAR: parses, atom count printed
 */
import fs from "node:fs";
import { readNumbers } from "../js/numbers.js";
import { SYMBOLS, elementOf, rcov, colorOf, SPACERS } from "../js/elements.js";
import { fromParams, params, det3, formula, norm, dot, fracToCart } from "../js/lattice.js";
import { parseText, detectKind, parseSymop, qeLattice, isWavecarHeader } from "../js/formats.js";
import { fft1 } from "../js/fft.js";
import { stats, renderGrid, isoForFraction, fractionAbove, insidePolygon } from "../js/field.js";
import { displayAtoms, bondList, deleteAtoms, atomId, toPOSCAR, shownStructure, toCIF, toXYZ } from "../js/structure.js";
import { primitive, conventional } from "../js/symmetry.js";
import { Wavecar } from "../js/wavecar.js";
import { marchingCubes } from "../js/gfx/mc.js";
import { Camera, qfromRows, qmat, pickSphere } from "../js/gfx/camera.js";
import { distance, angle, dihedral, measureAll } from "../js/measure.js";
import { packAtoms, packBonds } from "../js/gfx/renderer.js";
import { packLabels } from "../js/gfx/labels.js";
import { handle } from "../js/worker.js";

let fails = 0, n = 0;
const ok = (cond, msg) => { n++; if (!cond) { fails++; console.log("FAIL", msg); } else console.log("ok  ", msg); };
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const dist = (P, i, j) => Math.hypot(P[3 * i] - P[3 * j], P[3 * i + 1] - P[3 * j + 1], P[3 * i + 2] - P[3 * j + 2]);

/* numbers */
{
  const s = " 1.5  -.12E-02 3D+2 ,7 0.123456789012345678E+01 **** 1.0-2.0 ";
  const r = readNumbers(s, 0, 8, new Float64Array(8)).out;
  ok(r[0] === 1.5 && near(r[1], -0.0012, 1e-18) && r[2] === 300 && r[3] === 7, "readNumbers: plain, leading dot, Fortran D, comma");
  ok(near(r[4], 1.23456789012345678, 1e-15) && Number.isNaN(r[5]) && r[6] === 1 && r[7] === -2, "readNumbers: 17 digits, NaN token, glued sign");
}
/* elements and lattice */
ok(SYMBOLS.length === 104 && SYMBOLS[55] === "Cs" && SYMBOLS[47] === "Ag", "element table Z = 0..103");
ok(elementOf("Cs_sv") === "Cs" && elementOf("Fe1") === "Fe" && elementOf("O2-") === "O" && elementOf("CL") === "Cl" && elementOf("17") === "Cl", "elementOf labels");
ok(rcov("Cs") === 2.44 && rcov("Cl") === 1.02 && /^#[0-9a-f]{6}$/.test(colorOf("Ag")), "radii and colours");
{
  const L = fromParams(5, 6, 7, 80, 95, 110), p = params(L);
  ok([5, 6, 7, 80, 95, 110].every((v, i) => near(p[i], v, 1e-9)), "fromParams / params round trip (triclinic)");
  ok(formula(["Cs", "Cs", "Ag", "In", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl"]) === "Cs₂AgInCl₆", "formula subscripts");
}
/* FFT: radix-2 and Bluestein against a direct DFT */
for (const len of [8, 7, 12]) {
  const re = Float64Array.from({ length: len }, (_, i) => Math.sin(1.3 * i) + 0.2 * i), im = Float64Array.from({ length: len }, (_, i) => Math.cos(0.7 * i * i));
  const dr = new Float64Array(len), di = new Float64Array(len);
  for (let j = 0; j < len; j++) for (let k = 0; k < len; k++) { const a = 2 * Math.PI * j * k / len; dr[j] += re[k] * Math.cos(a) - im[k] * Math.sin(a); di[j] += re[k] * Math.sin(a) + im[k] * Math.cos(a); }
  fft1(re, im, 1);
  let e = 0; for (let j = 0; j < len; j++) e = Math.max(e, Math.abs(re[j] - dr[j]), Math.abs(im[j] - di[j]));
  ok(e < 1e-10, `fft1 n = ${len} vs direct DFT (max error ${e.toExponential(1)})`);
}
/* CIF with symmetry: rutile */
{
  const d = parseText("cif", fs.readFileSync(new URL("../examples/rutile-TiO2.cif", import.meta.url), "utf8"));
  const sp = d.structure.species, P = d.structure.pos;
  ok(sp.length === 6 && sp.filter(s => s === "Ti").length === 2, "rutile CIF: 16 symmetry operations give Ti2O4");
  let dmin = 9;
  for (let j = 0; j < 6; j++) if (sp[j] === "O") dmin = Math.min(dmin, dist(P, 0, j));
  ok(near(dmin, 1.98, 0.02), `rutile Ti-O bond ${dmin.toFixed(3)} A (expected 1.95-1.98)`);
  const op = parseSymop("-y+1/2, x-y, 1/2+z");
  ok(op[0].join() === "0,-1,0,0.5" && op[1].join() === "1,-1,0,0" && op[2].join() === "0,0,1,0.5", "symmetry operation parsing");
}
/* pw.in */
{
  const d = parseText("pw", fs.readFileSync(new URL("../examples/graphene.pw.in", import.meta.url), "utf8"));
  ok(d.structure.species.join() === "C,C" && near(dist(d.structure.pos, 0, 1), 2.46 / Math.sqrt(3), 1e-3), "graphene pw.in (ibrav 4, crystal): C-C = a/sqrt3");
  const si = `&control\n/\n&system\n ibrav=2, celldm(1)=10.26, nat=2, ntyp=1, ecutwfc=30\n/\n&electrons\n/\nATOMIC_SPECIES\nSi 28.086 Si.pbe.UPF\nATOMIC_POSITIONS alat\nSi 0.00 0.00 0.00\nSi 0.25 0.25 0.25\nK_POINTS automatic\n4 4 4 0 0 0\n`;
  const s = parseText("pw", si);
  ok(near(Math.abs(det3(s.structure.lattice)), Math.pow(10.26 * 0.529177210903, 3) / 4, 1e-6) && near(dist(s.structure.pos, 0, 1), 10.26 * 0.529177210903 * Math.sqrt(3) / 4, 1e-6), "Si pw.in (ibrav 2, alat): fcc volume a^3/4, Si-Si = sqrt3 a/4");
  const A = `&system\n ibrav = 0, A = 3.0, nat = 1, ntyp = 1\n/\nCELL_PARAMETERS alat\n1 0 0\n0 1 0\n0 0 2\nATOMIC_POSITIONS angstrom\nFe 0 0 0\n`;
  ok(near(Math.abs(det3(parseText("pw", A).structure.lattice)), 54, 1e-9), "pw.in ibrav 0 with A and CELL_PARAMETERS alat");
  for (const ib of [1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14]) {
    const cd = [0, 10, 1.1, 1.3, 0.2, -0.1, 0.15];
    const L = qeLattice(ib, cd);
    ok(Math.abs(det3(L)) > 1 && L.every(v => v.every(Number.isFinite)), `qeLattice ibrav ${ib} gives a finite cell`);
  }
}
/* XSF with a datagrid: f(i,j,k) = i + 10 j + 100 k */
{
  const nx = 3, ny = 4, nz = 2, vals = [];
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) vals.push(i + 10 * j + 100 * k);
  const x = `CRYSTAL\nPRIMVEC\n 4 0 0\n 0 5 0\n 0 0 6\nPRIMCOORD\n 2 1\n 14 0 0 0\n Si 1 1 1\nBEGIN_BLOCK_DATAGRID_3D\n demo\n BEGIN_DATAGRID_3D_rho\n ${nx} ${ny} ${nz}\n 0 0 0\n 4 0 0\n 0 5 0\n 0 0 6\n ${vals.join(" ")}\n END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n`;
  ok(detectKind("a.xsf", x) === "xsf" && detectKind("noname", x) === "xsf", "detect XSF by name and content");
  const d = parseText("xsf", x), f = d.fields[0];
  ok(d.structure.species.join() === "Si,Si" && f.name === "rho" && f.data[1 + nx * (2 + ny * 1)] === 121 && f.closed === true, "XSF: atoms, grid name, x-fastest order, closed periodic grid");
}
/* cube: z fastest in the file -> x fastest internally; Bohr -> Angstrom */
{
  const nx = 2, ny = 3, nz = 4, v = [];
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) v.push(i + 10 * j + 100 * k);
  const c = `t1\nt2\n 1 0.0 0.0 0.0\n ${nx} 1.0 0 0\n ${ny} 0 1.0 0\n ${nz} 0 0 1.0\n 8 0.0 1.0 0.0 0.0\n ${v.join(" ")}\n`;
  const d = parseText("cube", c), f = d.fields[0];
  ok(f.data[1 + nx * (2 + ny * 3)] === 321 && near(d.structure.pos[0], 0.529177210903, 1e-12) && d.structure.species[0] === "O", "cube: order, Bohr units, atom");
}
/* xyz, extended xyz, FHI-aims */
{
  const e = `2\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3\nNa 0 0 0\nCl 2.5 0 0\n`;
  const d = parseText("xyz", e);
  ok(d.structure.pbc && d.structure.species.join() === "Na,Cl" && d.structure.pos[3] === 2.5, "extended xyz with Lattice");
  const g = `lattice_vector 4 0 0\nlattice_vector 0 4 0\nlattice_vector 0 0 4\natom_frac 0.5 0.5 0.5 Cu\natom 0 0 0 Au\n`;
  const a = parseText("aims", g);
  ok(a.structure.species.join() === "Cu,Au" && a.structure.pos[0] === 2, "geometry.in atom_frac and atom");
}
/* POSCAR (VASP 5 and VASP 4, selective dynamics, cartesian, negative scale) */
{
  const p5 = `NaCl\n1.0\n 5.64 0 0\n 0 5.64 0\n 0 0 5.64\n Na Cl\n 1 1\nSelective dynamics\nDirect\n 0 0 0 T T T\n 0.5 0 0 F F F\n`;
  const d = parseText("vasp", p5);
  ok(d.structure.species.join() === "Na,Cl" && near(d.structure.pos[3], 2.82, 1e-12), "POSCAR VASP5, selective dynamics");
  const p4 = `Na Cl\n-178.453\n 1 0 0\n 0 1 0\n 0 0 1\n 1 1\nCartesian\n 0 0 0\n 0.5 0 0\n`;
  const d4 = parseText("vasp", p4);
  ok(d4.structure.species.join() === "Na,Cl" && near(Math.abs(det3(d4.structure.lattice)), 178.453, 1e-6), "POSCAR VASP4 (species from title), negative scale = volume");
}
/* periodic render grid and boundary atoms */
{
  const f = { n: [2, 2, 2], data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), origin: [0, 0, 0], span: [[2, 0, 0], [0, 2, 0], [0, 0, 2]], periodic: true, closed: false };
  const g = renderGrid(f, [1, 1, 1], 1);
  ok(g.size.join() === "3,3,3" && g.data[0] === 1 && g.data[1] === 2 && g.data[2] === 1 && g.data[3] === 3 && g.data[9] === 5 && g.data[26] === 1 && g.vox[0][0] === 1, "render grid: periodic closure, x-fastest order");
  const st = stats(f);
  ok(near(st.integral, 36, 1e-9) && near(fractionAbove(st, 4.5), 26 / 36, 1e-12) && isoForFraction(st, 0.5) > 0, "field stats: integral and enclosed fraction");
  const at = displayAtoms({ lattice: [[4, 0, 0], [0, 4, 0], [0, 0, 4]], pbc: true, species: ["Cu"], pos: Float64Array.from([0, 0, 0]) }, { reps: [1, 1, 1], boundary: true });
  ok(at.length === 8, "boundary atoms: corner atom drawn at the 8 cube corners");
  const b = bondList(displayAtoms({ lattice: null, pbc: false, species: ["O", "H", "H"], pos: Float64Array.from([0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0]) }));
  ok(b.length === 2, "bonds: water has two O-H bonds and no H-H bond");
  // spacer rule: rock salt NaCl keeps Na-Cl bonds (nothing else bonds); CsPbCl3 drops Cs-Cl and keeps Pb-Cl
  const nacl = { lattice: [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]], pbc: true, species: ["Na", "Cl"], pos: Float64Array.from([0, 0, 0, 2.82, 0, 0]) };
  const bn = bondList(displayAtoms(nacl, { reps: [2, 2, 2], boundary: false }), { spacers: SPACERS });
  const a0 = 5.6, cspb = { lattice: [[a0, 0, 0], [0, a0, 0], [0, 0, a0]], pbc: true, species: ["Cs", "Pb", "Cl", "Cl", "Cl"], pos: Float64Array.from([0, 0, 0, a0 / 2, a0 / 2, a0 / 2, 0, a0 / 2, a0 / 2, a0 / 2, 0, a0 / 2, a0 / 2, a0 / 2, 0]) };
  const da = displayAtoms(cspb, { reps: [1, 1, 1], boundary: true }), bc = bondList(da, { spacers: SPACERS });
  ok(bn.length > 0 && bc.length === 6 && bc.every(([i, j]) => da[i].el !== "Cs" && da[j].el !== "Cs"), "bonds: spacer cations bonded only when nothing else is (NaCl yes, Cs in CsPbCl3 no)");
}
/* WAVECAR header detection */
{
  const h = new Float64Array([4096, 2, 45200]).buffer;
  ok(isWavecarHeader(h) && !isWavecarHeader(new TextEncoder().encode("Cs2AgInCl6 POSCAR line one....").buffer), "WAVECAR header detection");
}

/* marching cubes: sphere |r| = R in a periodic-free box -> closed mesh, area 4 pi R^2, radial normals */
{
  const n = 41, L = 10, h = L / (n - 1), R0 = 3.1, data = new Float32Array(n * n * n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) data[i + n * (j + n * k)] = R0 - Math.hypot(-L / 2 + i * h, -L / 2 + j * h, -L / 2 + k * h);
  const m = marchingCubes({ data, size: [n, n, n], vox: [[h, 0, 0], [0, h, 0], [0, 0, h]], origin: [-L / 2, -L / 2, -L / 2] }, 0);
  const P = m.positions, I = m.indices, edges = new Map();
  let area = 0, rerr = 0, nerr = 0;
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t], b = I[t + 1], c = I[t + 2];
    const u = [0, 1, 2].map(q => P[3 * b + q] - P[3 * a + q]), v = [0, 1, 2].map(q => P[3 * c + q] - P[3 * a + q]);
    area += 0.5 * norm([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]);
    for (const [x, y] of [[a, b], [b, c], [c, a]]) { const key = Math.min(x, y) + "," + Math.max(x, y); edges.set(key, (edges.get(key) || 0) + 1); }
  }
  for (let v = 0; v < P.length / 3; v++) {
    const p = [P[3 * v], P[3 * v + 1], P[3 * v + 2]], r = norm(p);
    rerr = Math.max(rerr, Math.abs(r - R0));
    const nn = [m.normals[4 * v], m.normals[4 * v + 1], m.normals[4 * v + 2]].map(x => x / 32767);
    nerr = Math.max(nerr, 1 - (nn[0] * p[0] + nn[1] * p[1] + nn[2] * p[2]) / (r * norm(nn)));
  }
  const open = [...edges.values()].filter(c => c !== 2).length;
  ok(open === 0 && near(area, 4 * Math.PI * R0 * R0, 0.01 * 4 * Math.PI * R0 * R0) && rerr < 0.02 && nerr < 1e-3 && P.length / 3 < m.triangles,
    `marching cubes sphere: ${m.triangles} triangles, ${P.length / 3} shared vertices, open edges ${open}, area ${area.toFixed(3)} vs ${(4 * Math.PI * R0 * R0).toFixed(3)}, max |r - R| ${rerr.toFixed(3)} A, normal error ${nerr.toExponential(1)}`);
}
/* camera: the pixel ray through a projected point passes through it; view-along rotations are right-handed */
{
  const c = new Camera(); c.w = 800; c.h = 600; c.fit([1, 2, 3], 6); c.rotateBy(120, -45);
  for (const ortho of [false, true]) {
    c.ortho = ortho;
    const p = [2.5, 1.0, 4.2], s = c.project(p), r = c.ray(s.x, s.y);
    const w = [0, 1, 2].map(i => p[i] - r.o[i]), t = dot(w, r.d), miss = norm([0, 1, 2].map(i => w[i] - t * r.d[i]));
    ok(miss < 1e-6 && t > 0, `camera (${ortho ? "orthographic" : "perspective"}): ray through the projected pixel misses the point by ${miss.toExponential(1)} A`);
  }
  const q = qfromRows([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), M = qmat(q);
  ok(near(M[2][0], 1, 1e-12) && near(M[0][1], 1, 1e-12) && near(M[1][2], 1, 1e-12), "quaternion from view rows: along x, b right, c up");
  const idx = pickSphere({ o: [0, 0, 10], d: [0, 0, -1] }, Float32Array.from([0, 0, 0, 0, 0, 3, 5, 0, 0]), Float32Array.from([1, 1, 1]));
  ok(idx === 1, "pickSphere returns the nearest sphere on the ray");
}
/* measurements */
{
  ok(near(distance([0, 0, 0], [3, 4, 0]), 5, 1e-12) && near(angle([1, 0, 0], [0, 0, 0], [0, 1, 0]), 90, 1e-9), "distance and angle");
  ok(near(dihedral([1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]), 90, 1e-9) && near(dihedral([1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 1]), 0, 1e-9), "dihedral 90 and 0 degrees");
  const m = measureAll([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]);
  ok(m.bonds.length === 3 && m.angles.length === 2 && near(m.angles[0], 90, 1e-9) && near(Math.abs(m.dihedral), 90, 1e-9), "measureAll for four atoms");
}
/* instance packing */
{
  const a = new Float32Array(packAtoms([{ x: 1, y: 2, z: 3, r: 0.5, color: "#ff8000", sel: true }])), ab = new Uint8Array(a.buffer);
  ok(a.length === 5 && a[2] === 3 && a[3] === 0.5 && ab[16] === 255 && ab[17] === 128 && ab[19] === 128, "atom instance layout (20 B, selection flag in alpha)");
  const b = new Float32Array(packBonds([{ p0: [1, 2, 3], p1: [4, 5, 6], c0: "#010203", c1: "#040506", r: 0.1 }])), bb = new Uint8Array(b.buffer);
  ok(b.length === 9 && b[5] === 6 && near(b[8], 0.1, 1e-7) && bb[24] === 1 && bb[28] === 4, "bond instance layout (36 B)");
  const l = new Float32Array(packLabels([{ x: 1, y: 2, z: 3, r: 0.6, text: "Ag", color: "#c0c0c0" }], { Ag: { uv: [0.1, 0.2, 0.3, 0.4], aspect: 1.3 } })), lb = new Uint8Array(l.buffer);
  ok(l.length === 11 && near(l[4], 0.1, 1e-7) && near(l[8], 1.3, 1e-6) && lb[36] === 23 && lb[39] === 255, "label instance layout (44 B, dark text on a light atom)");
}
/* worker: open a CIF and a cube, then mesh the cube field */
{
  const cif = new File([fs.readFileSync(new URL("../examples/rutile-TiO2.cif", import.meta.url))], "rutile-TiO2.cif");
  const r1 = await handle({ op: "open", files: [cif] });
  ok(r1.result.structure.species.length === 6 && r1.result.fields.length === 0, "worker open: CIF structure");
  const n = 21, v = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) v.push(Math.exp(-((i - 10) ** 2 + (j - 10) ** 2 + (k - 10) ** 2) / 20));
  const cube = new File([`t\nt\n 1 0 0 0\n ${n} 0.3 0 0\n ${n} 0 0.3 0\n ${n} 0 0 0.3\n 1 0 3.0 3.0 3.0\n ${v.join(" ")}\n`], "g.cube");
  const r2 = await handle({ op: "open", files: [cube] });
  const f = r2.result.fields[0];
  ok(f && f.kind === "density" && f.stats && f.stats.sorted.length === n * n * n && !("data" in f), "worker open: cube field metadata and stats, no raw data");
  const r3 = await handle({ op: "mesh", ds: r2.result.ds, fi: 0, reps: [1, 1, 1], stride: 1, isos: [0.5] });
  const mm = r3.result.meshes[0];
  ok(mm.triangles > 100 && mm.indices.length === 3 * mm.triangles && r3.transfer.length === 3, `worker mesh: ${mm.triangles} triangles, buffers transferred`);
  const r4 = await handle({ op: "example", name: "h3d" });
  ok(r4.result.fields[0].kind === "signed" && near(r4.result.fields[0].stats.integral, 0, 1e-6), "worker example: hydrogen 3d orbital (signed, integral 0)");
}

/* origin shift and cuts on fields; shifted atoms; deletion; POSCAR export */
{
  const f = { n: [4, 1, 1], data: Float32Array.from([0, 1, 2, 3]), origin: [0, 0, 0], span: [[4, 0, 0], [0, 1, 0], [0, 0, 1]], periodic: true, closed: false };
  const g = renderGrid(f, [1, 1, 1], 1, [0.5, 0, 0]);
  ok(Array.from(g.data.slice(0, 5)).join() === "2,3,0,1,2" && g.origin[0] === 0, "field shift 0.5 = roll by 2 grid points");
  const h = renderGrid(f, [1, 1, 1], 1, [0.3, 0, 0]);
  ok(Array.from(h.data.slice(0, 5)).join() === "3,0,1,2,3" && near(h.origin[0], 0.2, 1e-12), "field shift 0.3 = roll by 1 point + origin moved 0.2 of a step (exact, no interpolation)");
  const all = { m: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], r: [-100, -100, 100, 100] };
  const c = renderGrid(f, [1, 1, 1], 1, [0, 0, 0], [all]);
  ok(Array.from(c.data).every(Number.isNaN), "a cut covering the view masks every point (NaN)");
  const n = 31, L = 10, hh = L / (n - 1), d = new Float32Array(n * n * n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) d[i + n * (j + n * k)] = 3 - Math.hypot(-5 + i * hh, -5 + j * hh, -5 + k * hh);
  const sph = { data: d, size: [n, n, n], vox: [[hh, 0, 0], [0, hh, 0], [0, 0, hh]], origin: [-5, -5, -5] };
  const full = marchingCubes(sph, 0).triangles;
  const half = { m: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], r: [0, -100, 100, 100] };
  const g2 = renderGrid({ n: [n, n, n], data: d, origin: [-5, -5, -5], span: [[L, 0, 0], [0, L, 0], [0, 0, L]], periodic: false, closed: false }, [1, 1, 1], 1, [0, 0, 0], [half]);
  const cut = marchingCubes(g2, 0);
  ok(cut.triangles > 0.4 * full && cut.triangles < 0.6 * full && Array.from(cut.normals).every(Number.isFinite), `cutting away x > 0 leaves half the sphere (${cut.triangles} of ${full} triangles), finite normals`);
  const st = { lattice: [[4, 0, 0], [0, 4, 0], [0, 0, 4]], pbc: true, species: ["Na", "Cl", "Na"], pos: Float64Array.from([0.4, 0, 0, 2, 0, 0, 3.6, 2, 0]) };
  const da = displayAtoms(st, { reps: [1, 1, 1], boundary: false, shift: [0.25, 0, 0] });
  ok(near(da[0].x, 1.4, 1e-12) && near(da[2].x, 0.6, 1e-9), "atoms shift by 0.25 a and wrap into the cell");
  const del = deleteAtoms(st, new Set([1]));
  ok(del.species.join() === "Na,Na" && atomId(del, 1) === 3, "deleting an atom keeps the original numbers of the others");
  const back = parseText("vasp", toPOSCAR(del, [0.25, 0, 0], "test"));
  ok(back.structure.species.join() === "Na,Na" && near(back.structure.pos[0], 1.4, 1e-8) && near(back.structure.pos[3], 0.6, 1e-8), "POSCAR export (shifted, edited) reads back");
}
/* worker: two datasets at once, mesh with shift and cuts, drop */
{
  const a = await handle({ op: "example", name: "h3d" }), b = await handle({ op: "example", name: "h3d" });
  ok(a.result.ds !== b.result.ds, "worker keeps several datasets (distinct ids)");
  const m0 = (await handle({ op: "mesh", ds: a.result.ds, fi: 0, reps: [1, 1, 1], stride: 1, isos: [0.01] })).result.meshes[0].triangles;
  const m1 = (await handle({ op: "mesh", ds: b.result.ds, fi: 0, reps: [1, 1, 1], stride: 1, isos: [0.01], cuts: [{ m: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], r: [0, -100, 100, 100] }] })).result.meshes[0].triangles;
  ok(m0 > 0 && m1 > 0 && m1 < m0, `worker mesh honours cuts (${m1} < ${m0} triangles)`);
  await handle({ op: "drop", ds: a.result.ds });
  let threw = false; try { await handle({ op: "mesh", ds: a.result.ds, fi: 0, reps: [1, 1, 1], stride: 1, isos: [0.01] }); } catch (e) { threw = true; }
  ok(threw, "a dropped dataset is gone");
}

/* symmetry: primitive and conventional cells */
{
  const mk = (L, sp, F) => ({ lattice: L, pbc: true, species: sp, pos: Float64Array.from(F.flatMap(f => fracToCart(L, f))) });
  const cub = a => [[a, 0, 0], [0, a, 0], [0, 0, a]];
  const cu = mk(cub(3.615), ["Cu", "Cu", "Cu", "Cu"], [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]);
  const p = primitive(cu), c = conventional(p.structure);
  ok(p.n === 4 && p.structure.species.length === 1 && near(params(p.structure.lattice)[3], 60, 1e-6), "fcc: primitive cell 1 atom, 60 degree rhombohedron");
  ok(c.system === "cubic" && c.structure.species.length === 4 && near(params(c.structure.lattice)[0], 3.615, 1e-6), "fcc primitive -> conventional cubic, 4 atoms");
  const nacl = mk(cub(5.64), ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"], [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]]);
  ok(primitive(nacl).structure.species.length === 2 && conventional(nacl).structure.species.length === 8, "NaCl: primitive 2, conventional 8");
  const rut = parseText("cif", fs.readFileSync(new URL("../examples/rutile-TiO2.cif", import.meta.url), "utf8")).structure, cr = conventional(rut);
  ok(primitive(rut).n === 1 && cr.system === "tetragonal" && near(params(cr.structure.lattice)[2], 2.9587, 1e-6), "rutile: primitive, conventional tetragonal");
  const rh = conventional(mk(fromParams(4, 4, 4, 70, 70, 70), ["Bi"], [[0, 0, 0]]));
  ok(rh.system === "rhombohedral" && rh.z === 3 && near(params(rh.structure.lattice)[5], 120, 1e-6), "rhombohedral lattice -> hexagonal triple cell");
  const oc = conventional(mk([[3, 0, 0], [0, 4, 0], [0, 0, 5]], ["C", "C"], [[0, 0, 0], [0.5, 0.5, 0]]));
  ok(oc.system === "orthorhombic" && oc.structure.species.length === 2 && params(oc.structure.lattice).slice(0, 3).map(x => +x.toFixed(6)).join() === "3,4,5", "C-centred orthorhombic -> 3 x 4 x 5 C cell");
  // supercells: a 3x3x3 NaCl supercell (216 atoms), exact and with ${noise} A noise, reduces to 2 / 8 atoms
  const sc = (st, r, noise = 0) => {
    const L = st.lattice, sp = [], pos = [];
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2 * noise;
    for (let i = 0; i < st.species.length; i++) for (let a = 0; a < r; a++) for (let b = 0; b < r; b++) for (let c = 0; c < r; c++) {
      sp.push(st.species[i]);
      pos.push(...[0, 1, 2].map(k => st.pos[3 * i + k] + a * L[0][k] + b * L[1][k] + c * L[2][k] + rnd()));
    }
    return { lattice: L.map(v => v.map(x => x * r)), pbc: true, species: sp, pos: Float64Array.from(pos) };
  };
  for (const [noise, tol] of [[0, 0.01], [0.002, 0.01], [0.004, 0.02]]) {
    const big = sc(nacl, 3, noise), pp = primitive(big, tol), cc = conventional(big, tol);
    ok(big.species.length === 216 && pp.structure.species.length === 2 && pp.n === 108 && cc.system === "cubic" && cc.structure.species.length === 8 && near(params(cc.structure.lattice)[0], 5.64, 0.01),
      `NaCl 3x3x3 supercell${noise ? ` with ${noise} A noise (tol ${tol} A)` : ""}: primitive 2 atoms (108 lattice points), conventional cubic 8 atoms, a = ${params(cc.structure.lattice)[0].toFixed(4)} A`);
  }
  // noise beyond the tolerance: no translation is accepted, the cell is reported as it is (no error)
  const loose = sc(nacl, 3, 0.004), lp = primitive(loose, 0.005);
  ok(lp.n === 1 && lp.structure.species.length === 216, "noise beyond the tolerance: the supercell is kept as it is, without an error");

}
/* export as shown: shift and supercell applied, CIF and XYZ read back */
{
  const st = { lattice: [[4, 0, 0], [0, 4, 0], [0, 0, 4]], pbc: true, species: ["Na", "Cl"], pos: Float64Array.from([0, 0, 0, 2, 0, 0]) };
  const sh = shownStructure(st, [1, 0, 0], [2, 1, 1]);
  ok(sh.species.length === 4 && near(det3(sh.lattice), 128, 1e-9) && near(sh.pos[0], 1, 1e-12) && near(sh.pos[3], 5, 1e-12), "shown structure: translated by 1 A, 2x1x1 supercell");
  const c = parseText("cif", toCIF(sh, "t")).structure, x = parseText("xyz", toXYZ(sh, "t")).structure, v = parseText("vasp", toPOSCAR(sh, [0, 0, 0], "t")).structure;
  ok([c, x, v].every(t => t.species.length === 4 && near(det3(t.lattice), 128, 1e-6)) && near(c.pos[0], 1, 1e-6) && near(x.pos[0], 1, 1e-6), "CIF, extended XYZ and POSCAR exports read back");
}
/* freehand cut: an L-shaped polygon masks only its own area */
{
  const f = { n: [4, 4, 2], data: new Float32Array(32).fill(1), origin: [0, 0, 0], span: [[3, 0, 0], [0, 3, 0], [0, 0, 1]], periodic: false, closed: false };
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const poly = [-0.5, -0.5, 2.5, -0.5, 2.5, 0.5, 0.5, 0.5, 0.5, 2.5, -0.5, 2.5];
  const g = renderGrid(f, [1, 1, 1], 1, [0, 0, 0], [{ m: I, poly }]);
  const masked = [...new Set([...g.data].map((v, k) => (Number.isNaN(v) ? `${k % 4},${Math.floor(k / 4) % 4}` : null)).filter(Boolean))];
  ok(masked.length && [...g.data].filter(Number.isNaN).length === 10 && masked.sort().join(" ") === "0,0 0,1 0,2 1,0 2,0" && insidePolygon(poly, 2, 0) && !insidePolygon(poly, 2, 2), "freehand (L-shaped) cut masks only the points inside");
}

/* optional: real files */
const env = process.env;
if (env.ISOVIEW_POSCAR) {
  const d = parseText("vasp", fs.readFileSync(env.ISOVIEW_POSCAR, "utf8"), env.ISOVIEW_POSCAR);
  ok(d.structure.species.length > 0, `POSCAR ${env.ISOVIEW_POSCAR}: ${d.structure.species.length} atoms, ${formula(d.structure.species)}`);
}
if (env.ISOVIEW_CHGCAR) {
  const t0 = Date.now(), txt = fs.readFileSync(env.ISOVIEW_CHGCAR, "utf8");
  const d = parseText("vasp", txt, "CHGCAR"), t1 = Date.now();
  const I = d.fields.map(f => stats(f).integral);
  ok(d.fields.length === 4 && near(I[0], +env.ISOVIEW_NELECT, 1e-3 * +env.ISOVIEW_NELECT) && near(I[1], +env.ISOVIEW_MAG, 0.02),
    `CHGCAR (${(txt.length / 1e6).toFixed(0)} MB, ${d.fields[0].n.join("x")}, parsed in ${((t1 - t0) / 1000).toFixed(1)} s): int rho = ${I[0].toFixed(3)}, int m = ${I[1].toFixed(4)}, int up/down = ${I[2].toFixed(3)}/${I[3].toFixed(3)}`);
}
if (env.ISOVIEW_WAVECAR) {
  const fd = fs.openSync(env.ISOVIEW_WAVECAR, "r");
  const read = async (off, len) => { const b = Buffer.alloc(len); fs.readSync(fd, b, 0, len, off); return b.buffer.slice(b.byteOffset, b.byteOffset + len); };
  const t0 = Date.now(), w = await Wavecar.open(read);
  ok(w.type === "gam", `WAVECAR: type ${w.type}, ${w.nspin} spins, ${w.nk} k, ${w.nb} bands, ENCUT ${w.encut}`);
  const [band, spin, Qref] = env.ISOVIEW_BAND.split(":").map(Number), c = env.ISOVIEW_CENTER.split(",").map(Number);
  const r = await w.orbital(spin, 0, band - 1, {}), t1 = Date.now();
  const [nx, ny, nz] = r.n, L = w.lattice, V = w.volume, dV = V / (nx * ny * nz);
  const Li = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  { const A = L, d = det3(A); Li[0] = [(A[1][1] * A[2][2] - A[1][2] * A[2][1]) / d, (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / d, (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / d];
    Li[1] = [(A[1][2] * A[2][0] - A[1][0] * A[2][2]) / d, (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / d, (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / d];
    Li[2] = [(A[1][0] * A[2][1] - A[1][1] * A[2][0]) / d, (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / d, (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / d]; }
  const fc = [0, 1, 2].map(j => c[0] * Li[0][j] + c[1] * Li[1][j] + c[2] * Li[2][j]);
  let Q = 0, tot = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = r.data[i + nx * (j + ny * k)];
    tot += v * dV;
    const f = [i / nx - fc[0], j / ny - fc[1], k / nz - fc[2]].map(x => x - Math.round(x));
    let best = 1e9;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let e = -1; e <= 1; e++) {
      const g = [f[0] + a, f[1] + b, f[2] + e];
      best = Math.min(best, norm([0, 1, 2].map(m => g[0] * L[0][m] + g[1] * L[1][m] + g[2] * L[2][m])));
    }
    if (best < 2.5) Q += v * dV;
  }
  ok(near(tot, 1, 1e-6) && near(Q, Qref, 0.02), `WAVECAR band ${band} spin ${spin}: grid ${r.n.join("x")}, ${((t1 - t0) / 1000).toFixed(1)} s, int = ${tot.toFixed(6)}, Q(2.5 A) = ${Q.toFixed(3)} (reference ${Qref})`);
  if (w.type === "gam") {
    const s = await w.orbital(spin, 0, band - 1, { signed: true });
    ok(s.signed && s.maxImag < 1e-6, `Gamma-only psi is real (max |Im|/max |Re| = ${s.maxImag.toExponential(1)})`);
  }
}
console.log(`\n${n - fails}/${n} passed`);
process.exit(fails ? 1 : 0);
