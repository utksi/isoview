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
import { SYMBOLS, elementOf, rcov, colorOf } from "../js/elements.js";
import { fromParams, params, det3, formula, norm } from "../js/lattice.js";
import { parseText, detectKind, parseSymop, qeLattice, isWavecarHeader } from "../js/formats.js";
import { fft1 } from "../js/fft.js";
import { stats, renderGrid, isoForFraction, fractionAbove } from "../js/field.js";
import { displayAtoms, bondList } from "../js/structure.js";
import { Wavecar } from "../js/wavecar.js";

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
  ok(g.size.join() === "3,3,3" && g.data[0] === 1 && g.data[1] === 5 && g.data[2] === 1 && g.data[3] === 3 && g.data[9] === 2 && g.data[26] === 1 && g.vox[0][0] === 1, "render grid: periodic closure, x-outer order");
  const st = stats(f);
  ok(near(st.integral, 36, 1e-9) && near(fractionAbove(st, 4.5), 26 / 36, 1e-12) && isoForFraction(st, 0.5) > 0, "field stats: integral and enclosed fraction");
  const at = displayAtoms({ lattice: [[4, 0, 0], [0, 4, 0], [0, 0, 4]], pbc: true, species: ["Cu"], pos: Float64Array.from([0, 0, 0]) }, { reps: [1, 1, 1], boundary: true });
  ok(at.length === 8, "boundary atoms: corner atom drawn at the 8 cube corners");
  const b = bondList(displayAtoms({ lattice: null, pbc: false, species: ["O", "H", "H"], pos: Float64Array.from([0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0]) }));
  ok(b.length === 2, "bonds: water has two O-H bonds and no H-H bond");
}
/* WAVECAR header detection */
{
  const h = new Float64Array([4096, 2, 45200]).buffer;
  ok(isWavecarHeader(h) && !isWavecarHeader(new TextEncoder().encode("Cs2AgInCl6 POSCAR line one....").buffer), "WAVECAR header detection");
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
