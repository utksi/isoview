// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Line charts as SVG text (for the analysis window and for saving as figures): one or more panels stacked over a
 * shared x axis, each a framed box with inward major and minor ticks on all four sides, tick labels left and below,
 * series as polylines and a legend placed in the corner of the panel that covers the fewest data points. CSV of the
 * same data.
 * Data x -> pixel: px = left + (x - x0) / (x1 - x0) * plot width; y likewise, upward. The x range is the data range
 * (tight) unless given; a y range covers the points inside the x range and is extended to the enclosing major ticks. Major ticks are 1, 2 or 5 x 10^k (about 5 per axis),
 * minor ticks divide a major step into 5 (step 1 or 5 x 10^k) or 4 (step 2 x 10^k); a panel gets about one major y tick
 * per 4 font heights. A constant series is shown over +-2% of its value. Non-finite points break a line.
 * Axis titles may mark variables as *italic*.
 */

/** Major ticks covering [a, b]: about n values at a step of 1, 2 or 5 x 10^k. Returns { lo, hi, step, ticks, minor }
 *  (minor: the minor tick step). */
export function niceTicks(a, b, n = 5) {
  if (!(b > a)) { const d = Math.abs(a) * 0.02 || 1; a -= d; b += d; } // a constant: +-2% around it
  const raw = (b - a) / n, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  const mm = m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10, step = mm * p;
  const lo = Math.floor(a / step + 1e-9) * step, hi = Math.ceil(b / step - 1e-9) * step, ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { lo, hi, step, ticks, minor: step / (mm === 2 ? 4 : 5) };
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
/** Axis or legend text as SVG content: escaped, with *x* set in italics. */
const rich = s => esc(s).replace(/\*([^*]+)\*/g, '<tspan font-style="italic">$1</tspan>');
/** Visible length of a text in characters (italic marks removed). */
const vlen = s => String(s).replace(/\*/g, "").length;
/** Tick label: up to 4 significant digits, exponent form for very small or large values, a true minus sign. */
const lab = v => { const a = Math.abs(v), s = a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(1) : String(+v.toPrecision(4)); return s.replace("-", "−"); };
let uid = 0;

/** The finite data range of a list of series: [x0, x1, y0, y1] (Infinity when empty); with xr = [a, b], only the
 *  points with a <= x <= b. */
function range(series, xr) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series) for (let i = 0; i < s.x.length; i++) {
    const x = s.x[i], y = s.y[i];
    if (!Number.isFinite(x) || !Number.isFinite(y) || (xr && (x < xr[0] || x > xr[1]))) continue;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return [x0, x1, y0, y1];
}

/**
 * SVG of stacked line-chart panels over one x axis.
 * panels: [{ series: [{ label, x, y (arrays of equal length), color, dash (bool), width }], ylabel, yr: optional
 * [min, max], zero: include y = 0 }]; xlabel: x axis title; xr: optional [min, max]; W, H: size in SVG units;
 * fs: tick label size (titles fs + 1); ink: colour of frame and text ("currentColor" on screen); font: font family;
 * size: optional [width, height] strings for the root element (physical size of a saved figure), default W, H px.
 */
export function chart({ panels, xlabel = "", xr, W = 520, H = 300, fs = 11, ink = "currentColor", font = "system-ui, sans-serif", size }) {
  const id = `pc${++uid}`, all = panels.flatMap(p => p.series);
  let [x0, x1] = range(all);
  if (!Number.isFinite(x0)) { x0 = 0; x1 = 1; }
  if (xr) [x0, x1] = xr;
  if (!(x1 > x0)) { x0 -= 0.5; x1 += 0.5; }
  const tx = niceTicks(x0, x1);
  const k = panels.length, T = Math.round(fs * 0.8), B = Math.round(fs * 3.3), gap = Math.round(fs * 0.8), ph = (H - T - B - gap * (k - 1)) / k;
  const ny = Math.max(3, Math.min(6, Math.round(ph / (fs * 4))));
  // y ranges and ticks per panel, for the left margin
  const ys = panels.map(p => {
    let [, , y0, y1] = range(p.series, [x0, x1]);
    if (!Number.isFinite(y0)) { y0 = 0; y1 = 1; }
    if (p.zero) { y0 = Math.min(0, y0); y1 = Math.max(0, y1); }
    if (p.yr) [y0, y1] = p.yr;
    const t = niceTicks(y0, y1, ny);
    return p.yr ? { y0, y1, t } : { y0: t.lo, y1: t.hi, t };
  });
  const wide = Math.max(1, ...ys.flatMap(q => q.t.ticks.filter(v => v >= q.y0 - 1e-12 && v <= q.y1 + 1e-12).map(v => lab(v).length)));
  const L = Math.round(fs * 1.9 + wide * fs * 0.6 + 8), R = Math.round(fs * 0.9), pw = W - L - R, tk = fs * 0.4, tm = fs * 0.22;
  const X = x => L + (x - x0) / (x1 - x0) * pw;
  const o = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${size ? size[0] : W}" height="${size ? size[1] : H}" font-family="${esc(font)}" font-size="${fs}">`];
  const inX = v => v >= x0 - 1e-9 * (x1 - x0) && v <= x1 + 1e-9 * (x1 - x0);
  panels.forEach((p, n) => {
    const top = T + n * (ph + gap), { y0, y1, t } = ys[n], last = n === k - 1;
    const Y = y => top + ph - (y - y0) / (y1 - y0 || 1) * ph, inY = v => v >= y0 - 1e-9 * (y1 - y0) && v <= y1 + 1e-9 * (y1 - y0);
    // ticks: major and minor, inward, on all four sides
    let d = "";
    const xt = (v, l) => { const px = X(v).toFixed(2); d += `M${px} ${(top + ph).toFixed(2)}v${-l}M${px} ${top.toFixed(2)}v${l}`; };
    const yt = (v, l) => { const py = Y(v).toFixed(2); d += `M${L} ${py}h${l}M${L + pw} ${py}h${-l}`; };
    for (let v = Math.ceil(x0 / tx.minor - 1e-9) * tx.minor; v <= x1 + 1e-9 * (x1 - x0); v += tx.minor) xt(v, Math.abs(v / tx.step - Math.round(v / tx.step)) < 1e-6 ? tk : tm);
    for (let v = Math.ceil(y0 / t.minor - 1e-9) * t.minor; v <= y1 + 1e-9 * (y1 - y0); v += t.minor) yt(v, Math.abs(v / t.step - Math.round(v / t.step)) < 1e-6 ? tk : tm);
    o.push(`<path d="${d}" stroke="${ink}" stroke-width="0.8" fill="none"/>`);
    o.push(`<g fill="${ink}">`);
    for (const v of t.ticks) if (inY(v) && !(n > 0 && Y(v) - top < fs * 0.6)) o.push(`<text x="${L - fs * 0.45}" y="${(Y(v) + fs * 0.35).toFixed(1)}" text-anchor="end">${lab(v)}</text>`);
    if (last) for (const v of tx.ticks) if (inX(v)) o.push(`<text x="${X(v).toFixed(1)}" y="${(top + ph + fs * 1.3).toFixed(1)}" text-anchor="middle">${lab(v)}</text>`);
    if (p.ylabel) o.push(`<text transform="translate(${(fs * 1.05).toFixed(1)} ${(top + ph / 2).toFixed(1)}) rotate(-90)" text-anchor="middle" font-size="${fs + 1}">${rich(p.ylabel)}</text>`);
    o.push("</g>");
    o.push(`<clipPath id="${id}-${n}"><rect x="${L}" y="${top.toFixed(2)}" width="${pw}" height="${ph.toFixed(2)}"/></clipPath><g clip-path="url(#${id}-${n})" fill="none" stroke-linejoin="round" stroke-linecap="round">`);
    const pts = [];
    for (const s of p.series) {
      let dd = "", pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i], y = s.y[i];
        if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; continue; }
        const px = X(x), py = Y(y);
        dd += `${pen ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`; pen = true; pts.push(px, py);
      }
      if (dd) o.push(`<path d="${dd}" stroke="${s.color}" stroke-width="${s.width || 1.5}"${s.dash ? ` stroke-dasharray="${fs * 0.45} ${fs * 0.3}"` : ""}/>`);
    }
    o.push("</g>");
    o.push(`<rect x="${L}" y="${top.toFixed(2)}" width="${pw}" height="${ph.toFixed(2)}" fill="none" stroke="${ink}" stroke-width="0.8"/>`);
    const named = p.series.filter(s => s.label);
    if (named.length > 1 || (named.length === 1 && p.series.length > 1)) o.push(legend(named, pts, { L, top, pw, ph, fs, ink }));
  });
  o.push(`<text x="${L + pw / 2}" y="${H - fs * 0.55}" text-anchor="middle" fill="${ink}" font-size="${fs + 1}">${rich(xlabel)}</text>`);
  o.push("</svg>");
  return o.join("");
}

/** Legend of the named series in the panel corner (top right, top left, bottom right, bottom left, in that order of
 *  preference) whose box contains the fewest data points pts ([px, py, ...]). Rows: a line sample, then the label. */
function legend(named, pts, { L, top, pw, ph, fs, ink }) {
  const row = fs * 1.3, sw = fs * 1.8, w = sw + fs * 0.5 + Math.max(...named.map(s => vlen(s.label))) * fs * 0.58, h = named.length * row + fs * 0.3, m = fs * 0.6;
  const corners = [[L + pw - m - w, top + m], [L + m, top + m], [L + pw - m - w, top + ph - m - h], [L + m, top + ph - m - h]];
  let best = 0, bestN = Infinity;
  corners.forEach(([bx, by], c) => {
    let cnt = 0;
    for (let i = 0; i < pts.length; i += 2) if (pts[i] >= bx - 2 && pts[i] <= bx + w + 2 && pts[i + 1] >= by - 2 && pts[i + 1] <= by + h + 2) cnt++;
    if (cnt < bestN) { bestN = cnt; best = c; }
  });
  const [bx, by] = corners[best], o = [`<g font-size="${fs * 0.95}">`];
  named.forEach((s, i) => {
    const y = by + fs * 0.15 + row * (i + 0.5);
    o.push(`<line x1="${bx.toFixed(1)}" x2="${(bx + sw).toFixed(1)}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${s.color}" stroke-width="${Math.max(1.6, s.width || 1.5)}"${s.dash ? ` stroke-dasharray="${fs * 0.45} ${fs * 0.3}"` : ""}/>`);
    o.push(`<text x="${(bx + sw + fs * 0.5).toFixed(1)}" y="${(y + fs * 0.33).toFixed(1)}" fill="${ink}">${rich(s.label)}</text>`);
  });
  o.push("</g>");
  return o.join("");
}

/** One panel: chart() with panels = [{ series, ylabel, yr, zero }]. */
export function lineChart({ series, ylabel = "", yr, zero = false, ...rest }) {
  return chart({ ...rest, panels: [{ series, ylabel, yr, zero }] });
}

/** CSV text of series, one x column and one y column per series: header "x name (label)", "label"; rows padded with
 *  blanks. Labels lose their *italic* marks. */
export function toCSV(series, xname) {
  const n = Math.max(...series.map(s => s.x.length)), plain = s => s.replace(/\*/g, ""), head = series.flatMap(s => [`${plain(xname)} (${plain(s.label)})`, plain(s.label)]);
  const rows = [head.map(h => `"${h.replace(/"/g, '""')}"`).join(",")];
  for (let i = 0; i < n; i++) rows.push(series.flatMap(s => (i < s.x.length ? [s.x[i], s.y[i]] : ["", ""])).join(","));
  return rows.join("\n") + "\n";
}
