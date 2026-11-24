// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Line charts as SVG text (for the analysis window and for saving): linear axes with "nice" ticks (steps of 1, 2 or
 * 5 times a power of ten, about 5 per axis), series as polylines, a legend, and CSV of the same data.
 * Data x -> pixel: px = left + (x - x0) / (x1 - x0) * (width - left - right); y likewise, upward.
 * Non-finite points break a line.
 */

/** Ticks covering [a, b]: about n values at a step of 1, 2 or 5 x 10^k. Returns { lo, hi, step, ticks }. */
export function niceTicks(a, b, n = 5) {
  if (!(b > a)) { const d = Math.abs(a) || 1; a -= d / 2; b += d / 2; }
  const raw = (b - a) / n, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  const lo = Math.floor(a / step + 1e-9) * step, hi = Math.ceil(b / step - 1e-9) * step, ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { lo, hi, step, ticks };
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
/** Tick label: up to 4 significant digits, exponent form for very small or large values. */
const lab = v => { const a = Math.abs(v); return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(1) : String(+v.toPrecision(4)); };

/**
 * SVG of a line chart. series: [{ label, x, y (arrays of equal length), color, dash (bool), width }]; xlabel, ylabel:
 * axis titles; xr, yr: optional [min, max] ranges (default: the data, y including 0 when zero: true); W, H: size in
 * px; ink: colour of axes and text ("currentColor" on screen, a fixed colour for files).
 */
export function lineChart({ series, xlabel = "", ylabel = "", xr, yr, zero = false, W = 520, H = 300, ink = "currentColor", font = "system-ui, sans-serif" }) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series) for (let i = 0; i < s.x.length; i++) {
    const x = s.x[i], y = s.y[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (!Number.isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
  if (zero) { y0 = Math.min(0, y0); y1 = Math.max(0, y1); }
  if (xr) [x0, x1] = xr;
  if (yr) [y0, y1] = yr;
  const tx = niceTicks(x0, x1), ty = niceTicks(y0, y1);
  if (!xr) { x0 = tx.lo; x1 = tx.hi; }
  y0 = yr ? y0 : ty.lo; y1 = yr ? y1 : ty.hi;
  const L = 58, R = 12, T = 12, B = 40, pw = W - L - R, ph = H - T - B;
  const X = x => L + (x - x0) / (x1 - x0 || 1) * pw, Y = y => T + ph - (y - y0) / (y1 - y0 || 1) * ph;
  const o = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="${esc(font)}" font-size="11">`];
  o.push(`<g stroke="${ink}" stroke-opacity="0.12">`);
  for (const v of ty.ticks) if (v >= y0 - 1e-12 && v <= y1 + 1e-12) o.push(`<line x1="${L}" x2="${L + pw}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/>`);
  o.push("</g>");
  o.push(`<rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="${ink}" stroke-opacity="0.45"/>`);
  o.push(`<g fill="${ink}" fill-opacity="0.75">`);
  for (const v of tx.ticks) if (v >= x0 - 1e-12 && v <= x1 + 1e-12) o.push(`<text x="${X(v).toFixed(1)}" y="${T + ph + 15}" text-anchor="middle">${lab(v)}</text>`);
  for (const v of ty.ticks) if (v >= y0 - 1e-12 && v <= y1 + 1e-12) o.push(`<text x="${L - 6}" y="${(Y(v) + 3.5).toFixed(1)}" text-anchor="end">${lab(v)}</text>`);
  o.push(`<text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle">${esc(xlabel)}</text>`);
  o.push(`<text transform="translate(13 ${T + ph / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text></g>`);
  o.push(`<clipPath id="pc"><rect x="${L}" y="${T}" width="${pw}" height="${ph}"/></clipPath><g clip-path="url(#pc)" fill="none" stroke-linejoin="round">`);
  for (const s of series) {
    let d = "", pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const x = s.x[i], y = s.y[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${X(x).toFixed(1)} ${Y(y).toFixed(1)}`; pen = true;
    }
    if (d) o.push(`<path d="${d}" stroke="${s.color}" stroke-width="${s.width || 1.5}"${s.dash ? ' stroke-dasharray="5 3"' : ""}/>`);
  }
  o.push("</g>");
  const named = series.filter(s => s.label);
  if (named.length > 1 || (named.length === 1 && series.length > 1)) {
    o.push(`<g font-size="10.5">`);
    named.forEach((s, i) => {
      const y = T + 12 + 14 * i, x = L + pw - 8;
      o.push(`<line x1="${x - 34}" x2="${x - 18}" y1="${y - 3.5}" y2="${y - 3.5}" stroke="${s.color}" stroke-width="2"${s.dash ? ' stroke-dasharray="4 2"' : ""}/><text x="${x - 38}" y="${y}" text-anchor="end" fill="${ink}">${esc(s.label)}</text>`);
    });
    o.push("</g>");
  }
  o.push("</svg>");
  return o.join("");
}

/** CSV text of series sharing one x axis per series: columns "x label, y label" pairs, rows padded with blanks. */
export function toCSV(series, xname) {
  const n = Math.max(...series.map(s => s.x.length)), head = series.flatMap(s => [`${xname} (${s.label})`, s.label]);
  const rows = [head.map(h => `"${h.replace(/"/g, '""')}"`).join(",")];
  for (let i = 0; i < n; i++) rows.push(series.flatMap(s => (i < s.x.length ? [s.x[i], s.y[i]] : ["", ""])).join(","));
  return rows.join("\n") + "\n";
}
