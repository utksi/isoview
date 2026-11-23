// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Set the deploy version N in every URL a browser caches: the stylesheet and app.js in index.html, every relative
 * module import in js/ (the page and the worker alike: a module worker does not use the page's import map) and the
 * worker URL. Browsers keep the site's files for hours, so a changed module must also change its URL, or a new
 * module can end up next to an old cached one.
 * Usage: node tools/version.mjs N
 */
import fs from "node:fs";
import path from "node:path";

const v = process.argv[2];
if (!/^\d+$/.test(v || "")) { console.error("usage: node tools/version.mjs N"); process.exit(2); }
const root = path.join(path.dirname(new URL(import.meta.url).pathname), "..");
const walk = d => fs.readdirSync(path.join(root, d), { withFileTypes: true }).flatMap(e => (e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith(".js") ? [path.join(d, e.name)] : []));
let changed = 0;
for (const f of ["index.html", ...walk("js")]) {
  const p = path.join(root, f), s = fs.readFileSync(p, "utf8");
  const t = s
    .replace(/((?:from|import\()\s*"\.{1,2}\/[^"?]+\.js)(?:\?v=\d+)?"/g, `$1?v=${v}"`)
    .replace(/(new URL\("\.\/[^"?]+\.(?:js|wasm))(?:\?v=\d+)?"/g, `$1?v=${v}"`)
    .replace(/((?:href|src)="(?:css\/style\.css|js\/app\.js))(?:\?v=\d+)?"/g, `$1?v=${v}"`);
  if (t !== s) { fs.writeFileSync(p, t); changed++; }
}
console.log(`version ${v}: ${changed} files updated`);
