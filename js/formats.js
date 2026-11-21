/**
 * Readers for structure and volumetric text formats. Every reader returns a dataset
 *   { title, structure, fields: [Field], notes: [string], traj? }
 * (multi-frame files: XDATCAR, OUTCAR, vasprun.xml, xyz, AXSF, LAMMPS dumps, pw.x output, PDB, .gro, CASTEP .md/.geom
 * are read by traj.js and add traj, the frames; structure is then frame 0)
 * structure = { lattice: [[3],[3],[3]] | null (rows a, b, c in Angstrom), pbc: boolean, species: [symbol],
 *               pos: Float64Array(3N) Cartesian Angstrom, origin?: [3] (corner of a non-periodic box, A; default 0) }
 * Field     = { name, unit, n: [nx, ny, nz], data: Float32Array (x index fastest: i + nx (j + ny k)),
 *               origin: [3] (Angstrom), span: [[3],[3],[3]], periodic, closed, kind }
 * Grid point (i, j, k) sits at origin + (i/dx) span[0] + (j/dy) span[1] + (k/dz) span[2], with d = n for periodic
 * grids (VASP, WAVECAR: point n coincides with point 0) and d = n - 1 for general grids (XSF, cube), whose last
 * point is the far face. closed = a general grid whose span is the periodic cell, so the last plane repeats the
 * first. kind = 'density' (non-negative), 'signed' (both signs, shown as +/- surfaces) or 'linear' (potentials,
 * ELF: a single surface at a chosen value).
 *
 * Units: VASP CHGCAR-type files store rho * V_cell; they are divided by V_cell here (e/A^3). LOCPOT (eV) and ELFCAR
 * are kept as stored. Cube files in Bohr (positive voxel counts) are converted to Angstrom; their values are kept.
 */
import { Cursor, readNumbers, tokens } from "./numbers.js?v=23";
import { elementOf, SYMBOLS } from "./elements.js?v=23";
import { BOHR, det3, inv3, fracToCart, fromParams, norm } from "./lattice.js?v=23";
import { finish, parseXDATCAR, parseOUTCAR, parseVasprun, parseXYZFrames, parseLAMMPS, parseQEout, parsePDB, parseGRO, parseCASTEP } from "./traj.js?v=23";

/* ------------------------------------------------------------------ detection */

/** True if the first 24 bytes look like a WAVECAR header (record length, nspin in {1, 2}, known format tag). */
export function isWavecarHeader(buf) {
  if (!buf || buf.byteLength < 24) return false;
  const d = new Float64Array(buf.slice(0, 24));
  const rl = d[0], ns = d[1], tag = d[2];
  return rl > 0 && rl < 2 ** 40 && Number.isInteger(rl) && (ns === 1 || ns === 2) &&
    [45200, 45210, 53300, 53310].includes(Math.round(tag));
}

/** Format of a text file from its name and its first ~64 kB. */
export function detectKind(name, head) {
  const n = (name || "").toLowerCase(), base = n.split(/[\\/]/).pop();
  if (/xdatcar/.test(base) || /^\s*(Direct|Cartesian) configuration=/m.test(head)) return "xdatcar";
  if (/outcar/.test(base) || (/^\s*vasp\.\d/.test(head) && /POTCAR:/.test(head))) return "outcar";
  if (/\.xml$/.test(n) && /<modeling>/.test(head)) return "vasprun";
  if (/\.(lammpstrj|dump|lammps)$/.test(n) || /^ITEM:\s*TIMESTEP/.test(head)) return "lammps";
  if (/Program PWSCF/.test(head)) return "qeout";
  if (/\.(pdb|ent)$/.test(n) || /^(CRYST1|MODEL |HETATM|ATOM  )/m.test(head.slice(0, 4096))) return "pdb";
  if (/\.gro$/.test(n)) return "gro";
  if (/\.(md|geom)$/.test(n) && /<--\s*[RhE]\s*$/m.test(head)) return "castep";
  if (/\.cif$/.test(n)) return "cif";
  if (/\.a?xsf$/.test(n)) return "xsf";
  if (/\.cube?$/.test(n)) return "cube";
  if (/\.(ext)?xyz$/.test(n)) return "xyz";
  if (/geometry\.in/.test(base)) return "aims";
  if (/(chgcar|^chg|parchg|locpot|elfcar|aeccar|chgdiff)/.test(base)) return "vasp";
  if (/(poscar|contcar|\.vasp$|\.poscar$)/.test(base)) return "vasp";
  if (/&system/i.test(head)) return "pw";
  if (/^\s*data_/m.test(head) && /_cell_length_a/i.test(head)) return "cif";
  if (/^\s*(CRYSTAL|MOLECULE|SLAB|POLYMER|ATOMS|PRIMVEC|BEGIN_BLOCK_DATAGRID_3D)\b/m.test(head)) return "xsf";
  if (/^\s*(lattice_vector|atom_frac)\s/m.test(head) || /^\s*atom\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[A-Za-z]/m.test(head)) return "aims";
  const first = head.split("\n", 2);
  if (/^\s*\d+\s*$/.test(first[0] || "")) return "xyz";
  return "vasp";
}

/** Parse text of the given kind. */
export function parseText(kind, text, name = "") {
  switch (kind) {
    case "vasp": return parseVasp(text, name);
    case "cif": return parseCIF(text);
    case "xsf": return parseXSF(text);
    case "cube": return parseCube(text);
    case "pw": return parsePW(text);
    case "xyz": return parseXYZFrames(text);
    case "aims": return parseAims(text);
    case "xdatcar": return parseXDATCAR(text);
    case "outcar": return parseOUTCAR(text);
    case "vasprun": return parseVasprun(text);
    case "lammps": return parseLAMMPS(text);
    case "qeout": return parseQEout(text);
    case "pdb": return parsePDB(text);
    case "gro": return parseGRO(text);
    case "castep": return parseCASTEP(text);
    default: throw new Error("unknown format " + kind);
  }
}

/* ------------------------------------------------------------------ helpers */

/** kind of a field from its value range: non-negative -> density; both signs of similar size -> signed; else linear. */
export function classify(data) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < data.length; i++) { const v = data[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
  if (mx <= 0 && mn < 0) return Math.abs(mx) < 1e-3 * Math.abs(mn) ? "signed" : "linear";
  if (mn >= -1e-3 * Math.abs(mx)) return "density";
  const r = -mn / mx;
  return r > 0.02 && r < 50 ? "signed" : "linear";
}

const almost = (A, B, tol = 1e-4) => A.every((r, i) => r.every((x, j) => Math.abs(x - B[i][j]) < tol));

function field(name, unit, n, data, origin, span, periodic, kind, closed = false) {
  return { name, unit, n, data, origin, span, periodic, closed, kind: kind || classify(data) };
}

/* ------------------------------------------------------------------ VASP: POSCAR, CONTCAR, CHGCAR, LOCPOT ... */

/** Read a POSCAR-format header at the cursor (VASP 4 or 5+). Returns { title, structure }. */
function readPoscar(cur) {
  const title = (cur.line() ?? "").trim();
  const sc = tokens(cur.line() ?? "").map(Number);
  const A = [0, 1, 2].map(() => tokens(cur.line() ?? "").slice(0, 3).map(Number));
  if (A.some(r => r.length < 3 || r.some(x => !Number.isFinite(x)))) throw new Error("not a POSCAR: bad lattice lines");
  let t = tokens(cur.line() ?? "");
  let symbols, counts;
  if (/^[A-Za-z]/.test(t[0] || "")) {
    symbols = t.map(elementOf);
    counts = tokens(cur.line() ?? "").map(Number);
  } else {
    counts = t.map(Number);
    const guess = title.split(/\s+/).map(elementOf);
    symbols = counts.map((_, i) => guess[i] || "X");
  }
  const nt = counts.findIndex(x => !Number.isInteger(x));
  if (nt >= 0) counts = counts.slice(0, nt);
  if (!counts.length) throw new Error("not a POSCAR: no atom counts");
  let s;
  if (sc.length >= 3 && sc.slice(0, 3).every(Number.isFinite)) s = sc.slice(0, 3);
  else if (sc[0] < 0) { const f = Math.cbrt(-sc[0] / Math.abs(det3(A))); s = [f, f, f]; }
  else s = [sc[0], sc[0], sc[0]];
  const L = A.map(r => r.map((x, j) => x * s[j]));
  let mode = (cur.line() ?? "").trim();
  if (/^s/i.test(mode)) mode = (cur.line() ?? "").trim();
  const cart = /^[ck]/i.test(mode);
  const N = counts.reduce((a, b) => a + b, 0);
  const pos = new Float64Array(3 * N), species = [];
  counts.forEach((c, i) => { for (let k = 0; k < c; k++) species.push(symbols[i] || "X"); });
  for (let i = 0; i < N; i++) {
    const v = tokens(cur.line() ?? "").slice(0, 3).map(Number);
    if (v.length < 3 || v.some(x => !Number.isFinite(x))) throw new Error(`POSCAR: bad position line ${i + 1}`);
    pos.set(cart ? v.map((x, j) => x * s[j]) : fracToCart(L, v), 3 * i);
  }
  return { title, structure: { lattice: L, pbc: true, species, pos } };
}

/** POSCAR/CONTCAR, or a VASP volumetric file (CHGCAR, CHG, PARCHG, LOCPOT, ELFCAR, AECCAR*). */
export function parseVasp(text, name = "") {
  const cur = new Cursor(text);
  const { title, structure } = readPoscar(cur);
  const g = tokens(cur.nonEmpty() ?? "");
  if (!(g.length === 3 && g.every(x => /^\d+$/.test(x)))) return { title, structure, fields: [], notes: [] };
  const n = g.map(Number), N = n[0] * n[1] * n[2];
  const re = new RegExp(`^[ \\t]*${n[0]}[ \\t]+${n[1]}[ \\t]+${n[2]}[ \\t]*\\r?$`, "gm");
  const blocks = [];
  let pos = cur.pos;
  for (;;) {
    const r = readNumbers(text, pos, N);
    blocks.push(r.out);
    pos = r.pos;
    if (blocks.length === 4) break;
    re.lastIndex = pos;
    const m = re.exec(text);
    if (!m) break;
    pos = m.index + m[0].length;
  }
  const up = (name || "").toUpperCase();
  const L = structure.lattice, V = Math.abs(det3(L)), O = [0, 0, 0];
  const mk = (nm, unit, d, kind) => field(nm, unit, n, d, O, L, true, kind);
  let fields;
  if (/LOCPOT/.test(up)) fields = blocks.map((d, i) => mk(blocks.length > 1 ? `potential ${i + 1}` : "potential", "eV", d, "linear"));
  else if (/ELF/.test(up)) fields = blocks.map((d, i) => mk(blocks.length > 1 ? `ELF ${i + 1}` : "ELF", "", d, "linear"));
  else {
    for (const d of blocks) for (let i = 0; i < d.length; i++) d[i] /= V;
    // WHY: PAW charge densities dip below zero near the cores; they are still densities (no negative surface)
    const u = "e/Å³";
    if (blocks.length === 1) fields = [mk("density", u, blocks[0], "density")];
    else if (blocks.length === 2) {
      const [t, m] = blocks, su = new Float32Array(N), sd = new Float32Array(N);
      for (let i = 0; i < N; i++) { su[i] = 0.5 * (t[i] + m[i]); sd[i] = 0.5 * (t[i] - m[i]); }
      fields = [mk("total", u, t, "density"), mk("magnetization", u, m), mk("spin up", u, su, "density"), mk("spin down", u, sd, "density")];
    } else if (blocks.length === 4) fields = ["total", "m_x", "m_y", "m_z"].map((nm, i) => mk(nm, u, blocks[i], i ? null : "density"));
    else fields = blocks.map((d, i) => mk(`block ${i + 1}`, u, d));
  }
  return { title, structure, fields, notes: [] };
}

/* ------------------------------------------------------------------ XSF (XCrySDen) */

export function parseXSF(text) {
  const cur = new Cursor(text);
  let lattice = null, pbc = false, species = [], pos = [], block = "", anim = 0;
  const fields = [], notes = [], frames = [], cells = [];
  const KW = /^(ATOMS|CRYSTAL|SLAB|POLYMER|MOLECULE|PRIMVEC|CONVVEC|PRIMCOORD|CONVCOORD|ANIMSTEPS|BEGIN_|END_|DATAGRID)/i;
  const read3 = () => [0, 1, 2].map(() => tokens(cur.nonEmpty() ?? "").slice(0, 3).map(Number));
  const atomLine = t => ({ el: elementOf(t[0]), r: t.slice(1, 4).map(Number) });
  let line;
  while ((line = cur.line()) !== null) {
    const t = tokens(line);
    if (!t.length || t[0][0] === "#") continue;
    const kw = t[0].toUpperCase();
    if (kw === "CRYSTAL" || kw === "SLAB" || kw === "POLYMER") pbc = true;
    else if (kw === "MOLECULE") pbc = false;
    else if (kw === "ANIMSTEPS") anim = parseInt(t[1]) || 0;
    else if (kw === "PRIMVEC") lattice = read3();
    else if (kw === "CONVVEC") read3();
    else if (kw === "PRIMCOORD" || kw === "CONVCOORD") {
      const nat = parseInt(tokens(cur.nonEmpty())[0]);
      const sp = [], ps = [];
      for (let i = 0; i < nat; i++) { const a = atomLine(tokens(cur.nonEmpty())); sp.push(a.el); ps.push(...a.r); }
      if (kw === "PRIMCOORD") { species = sp; pos = ps; if (anim) { frames.push({ sp, ps }); cells.push(lattice); } }
    } else if (kw === "ATOMS") {
      const sp = [], ps = [];
      for (;;) {
        const save = cur.pos, l = cur.line();
        if (l === null) break;
        const u = tokens(l);
        if (!u.length) continue;
        if (KW.test(u[0]) || u.length < 4 || u.slice(1, 4).some(x => !Number.isFinite(+x))) { cur.pos = save; break; }
        const a = atomLine(u); sp.push(a.el); ps.push(...a.r);
      }
      species = sp; pos = ps;
      if (anim) { frames.push({ sp, ps }); cells.push(lattice); }
    } else if (kw.startsWith("BEGIN_BLOCK_DATAGRID_3D") || kw === "BLOCK_DATAGRID_3D") {
      block = (cur.nonEmpty() ?? "").trim();
    } else if (/^(BEGIN_)?DATAGRID_3D/.test(kw)) {
      const gname = t[0].replace(/^(BEGIN_)?DATAGRID_3D_?/i, "") || block || "grid";
      const n = tokens(cur.nonEmpty()).slice(0, 3).map(Number);
      const origin = tokens(cur.nonEmpty()).slice(0, 3).map(Number);
      const span = read3();
      const data = cur.numbers(n[0] * n[1] * n[2]);
      fields.push({ gname, n, origin, span, data });
    } else if (kw.startsWith("BEGIN_BLOCK_DATAGRID_2D")) {
      let l;
      while ((l = cur.line()) !== null && !/END_BLOCK_DATAGRID_2D/i.test(l));
      notes.push("2D data grids are skipped");
    }
  }
  if (frames.length > 1 && !fields.length) {
    // animated XSF: frames with the atoms of the first one (a frame with other atoms ends the animation)
    const sp0 = frames[0].sp, ok = frames.findIndex(f => f.sp.length !== sp0.length || f.sp.some((s, i) => s !== sp0[i]));
    const use = ok < 0 ? frames : frames.slice(0, ok);
    if (ok >= 0) notes.push(`step ${ok + 1} of the animation has other atoms; it stops before it`);
    return finish(sp0, use.map(f => Float64Array.from(f.ps)), cells.slice(0, use.length), [], { pbc, notes });
  }
  if (frames.length > 1) notes.push(`animated XSF with data grids: showing the last of ${frames.length} structures`);
  const structure = { lattice: lattice, pbc: pbc && !!lattice, species, pos: Float64Array.from(pos) };
  const out = fields.map(f => {
    const closed = !!(structure.pbc && almost(f.span, lattice, 1e-3) && f.origin.every(x => Math.abs(x) < 1e-4));
    return field(f.gname, "", f.n, f.data, f.origin, f.span, false, null, closed);
  });
  if (!lattice && out.length) { structure.lattice = out[0].span.map(r => r.slice()); structure.origin = out[0].origin.slice(); }
  return { title: "", structure, fields: out, notes };
}

/* ------------------------------------------------------------------ Gaussian cube */

export function parseCube(text) {
  const cur = new Cursor(text);
  const title = [(cur.line() ?? "").trim(), (cur.line() ?? "").trim()].filter(Boolean).join(" · ");
  const t = tokens(cur.line());
  const nat = parseInt(t[0]);
  let nval = t.length > 4 ? parseInt(t[4]) : 1;
  const ax = [0, 1, 2].map(() => tokens(cur.line()).slice(0, 4).map(Number));
  const f = ax[0][0] > 0 ? BOHR : 1;
  const n = ax.map(r => Math.abs(Math.round(r[0])));
  const v = ax.map(r => r.slice(1, 4).map(x => x * f));
  const origin = t.slice(1, 4).map(x => +x * f);
  const species = [], pos = [];
  for (let i = 0; i < Math.abs(nat); i++) {
    const a = tokens(cur.line());
    species.push(SYMBOLS[Math.round(+a[0])] || "X");
    pos.push(+a[2] * f, +a[3] * f, +a[4] * f);
  }
  let ids = null;
  if (nat < 0) {
    const r = readNumbers(text, cur.pos, 1, new Float64Array(1));
    const m = Math.round(r.out[0]);
    const rr = readNumbers(text, r.pos, m, new Float64Array(m));
    ids = Array.from(rr.out);
    cur.pos = rr.pos;
    nval = m;
  }
  const [nx, ny, nz] = n, N = nx * ny * nz;
  const raw = cur.numbers(N * nval);
  const fields = [];
  for (let s = 0; s < nval; s++) {
    const d = new Float32Array(N);
    let p = s;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++, p += nval) d[i + nx * (j + ny * k)] = raw[p];
    const span = v.map((r, a) => r.map(x => x * (n[a] - 1)));
    fields.push(field(ids ? `orbital ${ids[s]}` : nval > 1 ? `value ${s + 1}` : "cube", "", n, d, origin, span, false, null));
  }
  // WHY: the box drawn is the data's own extent, from the grid origin, so the molecule sits where its field is
  const lattice = v.map((r, a) => r.map(x => x * (n[a] - 1)));
  return { title, structure: { lattice, pbc: false, species, pos: Float64Array.from(pos), origin }, fields, notes: [] };
}

/* ------------------------------------------------------------------ CIF */

/** CIF tokens: bare words, quoted strings (a quote ends only before whitespace), ;-delimited text fields. */
function cifTokens(text) {
  const out = [], L = text.length, ws = c => c === 32 || c === 9 || c === 10 || c === 13;
  let i = 0;
  while (i < L) {
    const c = text.charCodeAt(i);
    if (ws(c)) { i++; continue; }
    if (c === 35) { const e = text.indexOf("\n", i); i = e < 0 ? L : e + 1; continue; }
    if (c === 59 && (i === 0 || text.charCodeAt(i - 1) === 10)) {
      const e = text.indexOf("\n;", i + 1);
      out.push({ v: text.slice(i + 1, e < 0 ? L : e).trim(), q: true });
      i = e < 0 ? L : e + 2;
      continue;
    }
    if (c === 39 || c === 34) {
      const q = text[i];
      let j = i + 1;
      while (j < L && !(text[j] === q && (j + 1 >= L || ws(text.charCodeAt(j + 1))))) j++;
      out.push({ v: text.slice(i + 1, j), q: true });
      i = j + 1;
      continue;
    }
    let j = i;
    while (j < L && !ws(text.charCodeAt(j))) j++;
    out.push({ v: text.slice(i, j), q: false });
    i = j;
  }
  return out;
}

/** Data blocks: { items: {tag: value}, loops: [{ tags, rows }] }; tags lower-cased with '.' replaced by '_'. */
function cifBlocks(tok) {
  const blocks = [], key = /^(loop_|data_|save_|global_|stop_)/i, tag = t => !t.q && t.v[0] === "_";
  const nm = s => s.toLowerCase().replace(/\./g, "_");
  let cur = null, i = 0;
  while (i < tok.length) {
    const t = tok[i];
    if (!t.q && /^data_/i.test(t.v)) { cur = { name: t.v.slice(5), items: {}, loops: [] }; blocks.push(cur); i++; continue; }
    if (!cur) { cur = { name: "", items: {}, loops: [] }; blocks.push(cur); }
    if (!t.q && /^loop_$/i.test(t.v)) {
      i++;
      const tags = [];
      while (i < tok.length && tag(tok[i])) tags.push(nm(tok[i++].v));
      const vals = [];
      while (i < tok.length && !tag(tok[i]) && !(!tok[i].q && key.test(tok[i].v))) vals.push(tok[i++].v);
      const rows = [];
      for (let r = 0; r + tags.length <= vals.length; r += tags.length) rows.push(vals.slice(r, r + tags.length));
      cur.loops.push({ tags, rows });
      continue;
    }
    if (tag(t)) { cur.items[nm(t.v)] = tok[i + 1] ? tok[i + 1].v : ""; i += 2; continue; }
    i++;
  }
  return blocks;
}

const cifNum = v => parseFloat(String(v).replace(/\([^)]*\)/g, ""));
const frac = s => (s.includes("/") ? s.split("/").reduce((a, b) => +a / +b) : +s);

/** Symmetry operation "x,y,z" form -> 3 rows [cx, cy, cz, t] acting on fractional coordinates. */
export function parseSymop(s) {
  const parts = s.toLowerCase().replace(/[\s'"]/g, "").split(",");
  if (parts.length !== 3) throw new Error("bad symmetry operation: " + s);
  return parts.map(p => {
    const row = [0, 0, 0, 0];
    for (let term of p.replace(/-/g, "+-").split("+").filter(Boolean)) {
      let sg = 1;
      if (term[0] === "-") { sg = -1; term = term.slice(1); }
      const ax = term.match(/[xyz]/);
      if (ax) { const c = term.replace(/[xyz*]/g, ""); row["xyz".indexOf(ax[0])] += sg * (c ? frac(c) : 1); }
      else row[3] += sg * frac(term);
    }
    return row;
  });
}

export function parseCIF(text) {
  const blocks = cifBlocks(cifTokens(text));
  const b = blocks.find(x => x.items._cell_length_a !== undefined) || blocks[0];
  if (!b) throw new Error("CIF: no data block");
  const it = b.items, notes = [];
  const p = ["a", "b", "c"].map(k => cifNum(it["_cell_length_" + k]));
  const ang = ["alpha", "beta", "gamma"].map(k => (it["_cell_angle_" + k] !== undefined ? cifNum(it["_cell_angle_" + k]) : 90));
  if (p.some(x => !Number.isFinite(x))) throw new Error("CIF: missing cell lengths");
  const L = fromParams(...p, ...ang), Li = inv3(L);
  const site = b.loops.find(l => l.tags.includes("_atom_site_fract_x"));
  if (!site) throw new Error("CIF: no _atom_site_fract_x loop");
  const col = k => site.tags.indexOf(k);
  const ix = col("_atom_site_fract_x"), iy = col("_atom_site_fract_y"), iz = col("_atom_site_fract_z");
  const ity = col("_atom_site_type_symbol"), ila = col("_atom_site_label"), ioc = col("_atom_site_occupancy");
  const symTag = /(_symmetry_equiv_pos_as_xyz|_space_group_symop_operation_xyz)$/;
  const sl = b.loops.find(l => l.tags.some(t => symTag.test(t)));
  let ops = sl ? sl.rows.map(r => r[sl.tags.findIndex(t => symTag.test(t))]) : [];
  if (!ops.length) {
    const single = Object.keys(it).find(k => symTag.test(k));
    ops = single ? [it[single]] : ["x,y,z"];
    const hm = it["_symmetry_space_group_name_h-m"] || it["_space_group_name_h-m_alt"] || "";
    if (!single && hm && !/^p\s*1$/i.test(hm.trim())) notes.push(`no symmetry operations in the file (space group ${hm}): only the listed sites are shown`);
  }
  const M = ops.map(parseSymop);
  const species = [], pos = [];
  let partial = 0;
  for (const r of site.rows) {
    const f0 = [cifNum(r[ix]), cifNum(r[iy]), cifNum(r[iz])];
    if (f0.some(x => !Number.isFinite(x))) continue;
    const el = elementOf(ity >= 0 && r[ity] !== "?" && r[ity] !== "." ? r[ity] : r[ila]);
    if (ioc >= 0 && cifNum(r[ioc]) < 0.999) partial++;
    const seen = [];
    for (const m of M) {
      const f = m.map(row => { const v = row[0] * f0[0] + row[1] * f0[1] + row[2] * f0[2] + row[3]; return v - Math.floor(v); });
      const dup = seen.some(g => {
        const d = [0, 1, 2].map(j => { const x = f[j] - g[j]; return x - Math.round(x); });
        return norm(fracToCart(L, d)) < 0.01;
      });
      if (!dup) { seen.push(f); species.push(el); pos.push(...fracToCart(L, f)); }
    }
  }
  if (partial) notes.push(`${partial} partially occupied site${partial > 1 ? "s" : ""}: all positions are drawn`);
  void Li;
  return { title: b.name, structure: { lattice: L, pbc: true, species, pos: Float64Array.from(pos) }, fields: [], notes };
}

/* ------------------------------------------------------------------ Quantum ESPRESSO pw.x input */

const PW_CARDS = /^(ATOMIC_SPECIES|ATOMIC_POSITIONS|CELL_PARAMETERS|K_POINTS|OCCUPATIONS|CONSTRAINTS|ATOMIC_FORCES|ATOMIC_VELOCITIES|ADDITIONAL_K_POINTS|SOLVENTS|HUBBARD)\b\s*[{(]?\s*([A-Za-z_]*)/i;

function pwValue(v) {
  if (/^['"].*['"]$/.test(v)) return v.slice(1, -1);
  if (/^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$/.test(v)) return +v.replace(/[dD]/, "e");
  return v;
}

/** Lattice vectors (Angstrom) for a QE Bravais-lattice index and celldm(1..6) (celldm[0] unused). */
export function qeLattice(ibrav, cd) {
  const a = cd[1] * BOHR, ba = cd[2], ca = cd[3], s3 = Math.sqrt(3);
  switch (ibrav) {
    case 1: return [[a, 0, 0], [0, a, 0], [0, 0, a]];
    case 2: return [[-a / 2, 0, a / 2], [0, a / 2, a / 2], [-a / 2, a / 2, 0]];
    case 3: return [[a / 2, a / 2, a / 2], [-a / 2, a / 2, a / 2], [-a / 2, -a / 2, a / 2]];
    case -3: return [[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]];
    case 4: return [[a, 0, 0], [-a / 2, a * s3 / 2, 0], [0, 0, a * ca]];
    case 5: case -5: {
      const c = cd[4], tx = Math.sqrt((1 - c) / 2), ty = Math.sqrt((1 - c) / 6), tz = Math.sqrt((1 + 2 * c) / 3);
      if (ibrav === 5) return [[a * tx, -a * ty, a * tz], [0, 2 * a * ty, a * tz], [-a * tx, -a * ty, a * tz]];
      const ap = a / s3, u = tz - 2 * Math.SQRT2 * ty, v = tz + Math.SQRT2 * ty;
      return [[ap * u, ap * v, ap * v], [ap * v, ap * u, ap * v], [ap * v, ap * v, ap * u]];
    }
    case 6: return [[a, 0, 0], [0, a, 0], [0, 0, a * ca]];
    case 7: return [[a / 2, -a / 2, a * ca / 2], [a / 2, a / 2, a * ca / 2], [-a / 2, -a / 2, a * ca / 2]];
    case 8: return [[a, 0, 0], [0, a * ba, 0], [0, 0, a * ca]];
    case 9: return [[a / 2, a * ba / 2, 0], [-a / 2, a * ba / 2, 0], [0, 0, a * ca]];
    case -9: return [[a / 2, -a * ba / 2, 0], [a / 2, a * ba / 2, 0], [0, 0, a * ca]];
    case 91: return [[a, 0, 0], [0, a * ba / 2, -a * ca / 2], [0, a * ba / 2, a * ca / 2]];
    case 10: return [[a / 2, 0, a * ca / 2], [a / 2, a * ba / 2, 0], [0, a * ba / 2, a * ca / 2]];
    case 11: return [[a / 2, a * ba / 2, a * ca / 2], [-a / 2, a * ba / 2, a * ca / 2], [-a / 2, -a * ba / 2, a * ca / 2]];
    case 12: { const g = cd[4], sg = Math.sqrt(1 - g * g); return [[a, 0, 0], [a * ba * g, a * ba * sg, 0], [0, 0, a * ca]]; }
    case -12: { const c5 = cd[5], s = Math.sqrt(1 - c5 * c5); return [[a, 0, 0], [0, a * ba, 0], [a * ca * c5, 0, a * ca * s]]; }
    case 13: { const g = cd[4], sg = Math.sqrt(1 - g * g); return [[a / 2, 0, -a * ca / 2], [a * ba * g, a * ba * sg, 0], [a / 2, 0, a * ca / 2]]; }
    case -13: { const c5 = cd[5], s = Math.sqrt(1 - c5 * c5); return [[a / 2, a * ba / 2, 0], [-a / 2, a * ba / 2, 0], [a * ca * c5, 0, a * ca * s]]; }
    case 14: {
      const cal = cd[4], cbe = cd[5], cga = cd[6], sga = Math.sqrt(1 - cga * cga);
      const z = Math.sqrt(1 + 2 * cal * cbe * cga - cal * cal - cbe * cbe - cga * cga) / sga;
      return [[a, 0, 0], [a * ba * cga, a * ba * sga, 0], [a * ca * cbe, a * ca * (cal - cbe * cga) / sga, a * ca * z]];
    }
    default: throw new Error(`pw.in: ibrav = ${ibrav} is not supported`);
  }
}

export function parsePW(text) {
  const lines = text.split(/\r?\n/).map(l => l.replace(/[!#].*$/, ""));
  const nl = {}, cards = [];
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    const m = t.match(/^&(\w+)/);
    if (m) {
      let body = t.slice(m[0].length);
      let done = /(^|[\s,])\/$/.test(body.trim()) || body.trim() === "/";
      while (!done && ++i < lines.length) {
        const u = lines[i].trim();
        if (u === "/" || /(^|[\s,])\/$/.test(u)) { body += "\n" + u.replace(/\/$/, ""); done = true; }
        else body += "\n" + u;
      }
      const d = {};
      const re = /([A-Za-z_][\w%]*)\s*(\(\s*[\d,\s]+\))?\s*=\s*('[^']*'|"[^"]*"|[^\s,]+)/g;
      let a;
      while ((a = re.exec(body))) d[(a[1] + (a[2] ? "(" + a[2].replace(/[()\s]/g, "") + ")" : "")).toLowerCase()] = pwValue(a[3]);
      nl[m[1].toLowerCase()] = d;
      continue;
    }
    const c = t.match(PW_CARDS);
    if (c) cards.push({ name: c[1].toUpperCase(), opt: (c[2] || "").toLowerCase(), line: i });
  }
  const sys = nl.system;
  if (!sys) throw new Error("pw.in: no &SYSTEM namelist");
  const ibrav = Math.round(+(sys.ibrav ?? 0)), nat = Math.round(+sys.nat);
  const cd = [0, 1, 2, 3, 4, 5, 6].map(k => +(sys[`celldm(${k})`] ?? 0));
  if (sys.a !== undefined) {
    const A = +sys.a;
    cd[1] = A / BOHR; cd[2] = (+sys.b || 0) / A; cd[3] = (+sys.c || 0) / A;
    const cab = +(sys.cosab ?? 0), cac = +(sys.cosac ?? 0), cbc = +(sys.cosbc ?? 0);
    if (ibrav === 14) { cd[4] = cbc; cd[5] = cac; cd[6] = cab; }
    else if (ibrav === -12 || ibrav === -13) cd[5] = cac;
    else cd[4] = cab;
  }
  const alat = cd[1] * BOHR, hasAlat = cd[1] > 0;
  const card = nm => cards.find(c => c.name === nm);
  const body = (c, n) => { const out = []; for (let i = c.line + 1; i < lines.length && out.length < n; i++) { const t = lines[i].trim(); if (t) out.push(tokens(t)); } return out; };
  let L;
  if (ibrav === 0) {
    const c = card("CELL_PARAMETERS");
    if (!c) throw new Error("pw.in: ibrav = 0 needs CELL_PARAMETERS");
    const unit = c.opt || (hasAlat ? "alat" : "bohr");
    const f = unit === "alat" ? alat : unit === "bohr" ? BOHR : 1;
    L = body(c, 3).map(r => r.slice(0, 3).map(x => +x.replace(/[dD]/, "e") * f));
  } else L = qeLattice(ibrav, cd);
  const ap = card("ATOMIC_POSITIONS");
  if (!ap) throw new Error("pw.in: no ATOMIC_POSITIONS");
  const unit = ap.opt || "alat";
  if (unit === "crystal_sg") throw new Error("pw.in: ATOMIC_POSITIONS crystal_sg is not supported");
  const spc = card("ATOMIC_SPECIES"), pseudo = {};
  if (spc) for (const r of body(spc, Math.round(+sys.ntyp) || 0)) pseudo[r[0]] = r[2] || "";
  const species = [], pos = [];
  for (const r of body(ap, nat)) {
    let el = elementOf(r[0]);
    if (el === "X" && pseudo[r[0]]) el = elementOf(pseudo[r[0]]);
    species.push(el);
    const v = r.slice(1, 4).map(x => +x.replace(/[dD]/, "e"));
    if (unit === "crystal") pos.push(...fracToCart(L, v));
    else { const f = unit === "alat" ? alat : unit === "bohr" ? BOHR : 1; pos.push(...v.map(x => x * f)); }
  }
  const title = typeof nl.control?.prefix === "string" ? nl.control.prefix : "";
  return { title, structure: { lattice: L, pbc: true, species, pos: Float64Array.from(pos) }, fields: [], notes: [] };
}

/* ------------------------------------------------------------------ xyz / extended xyz: traj.js parseXYZFrames */

/* ------------------------------------------------------------------ FHI-aims geometry.in */

export function parseAims(text) {
  const lat = [], species = [], pos = [], fr = [];
  for (const raw of text.split(/\r?\n/)) {
    const t = tokens(raw.replace(/#.*$/, ""));
    if (!t.length) continue;
    const k = t[0].toLowerCase();
    if (k === "lattice_vector") lat.push(t.slice(1, 4).map(Number));
    else if (k === "atom") { species.push(elementOf(t[4])); pos.push(t.slice(1, 4).map(Number)); fr.push(false); }
    else if (k === "atom_frac") { species.push(elementOf(t[4])); pos.push(t.slice(1, 4).map(Number)); fr.push(true); }
  }
  const lattice = lat.length === 3 ? lat : null;
  if (fr.some(Boolean) && !lattice) throw new Error("geometry.in: atom_frac needs three lattice_vector lines");
  const P = pos.flatMap((p, i) => (fr[i] ? fracToCart(lattice, p) : p));
  return { title: "", structure: { lattice, pbc: !!lattice, species, pos: Float64Array.from(P) }, fields: [], notes: [] };
}
