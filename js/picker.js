// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * In-page colour picker, so that choosing a colour never opens the system's colour dialog: a popover anchored to the
 * element that opened it, with a saturation-value square, a hue slider, a row of swatches, a hex field and, when the
 * caller gives one, a Default button. Every change calls onInput(hex) at once (the scene follows live); a click
 * outside, Enter or the anchor again closes it and keeps the colour; Esc closes it and restores the colour it opened
 * with.
 *
 * Colour model HSV, h in [0, 360) degrees, s and v in [0, 1]. HSV -> RGB: c = v s, h' = h / 60,
 * x = c (1 - |h' mod 2 - 1|), (r1, g1, b1) = (c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x) for
 * h' in [0, 1), [1, 2), ..., [5, 6); m = v - c; channel = round(255 (channel1 + m)).
 * RGB -> HSV (channels in [0, 1]): v = max, d = max - min, s = d / v (0 when v = 0), h = 60 ((g - b) / d mod 6),
 * 60 ((b - r) / d + 2) or 60 ((r - g) / d + 4) for max = r, g or b (0 when d = 0).
 * In the square, x is s (0 left to 1 right) and y is v (1 top to 0 bottom).
 */

/** "#rrggbb" of HSV (h degrees, s and v in [0, 1]). */
export function hsvToHex(h, s, v) {
  const c = v * s, hp = (((h % 360) + 360) % 360) / 60, x = c * (1 - Math.abs((hp % 2) - 1)), m = v - c;
  const [r, g, b] = hp < 1 ? [c, x, 0] : hp < 2 ? [x, c, 0] : hp < 3 ? [0, c, x] : hp < 4 ? [0, x, c] : hp < 5 ? [x, 0, c] : [c, 0, x];
  return "#" + [r, g, b].map(t => Math.round(255 * (t + m)).toString(16).padStart(2, "0")).join("");
}
/** [h, s, v] of "#rrggbb" (h degrees, s and v in [0, 1]). */
export function hexToHsv(hex) {
  const n = parseInt(hex.slice(1), 16), r = (n >> 16 & 255) / 255, g = (n >> 8 & 255) / 255, b = (n & 255) / 255;
  const v = Math.max(r, g, b), d = v - Math.min(r, g, b), s = v ? d / v : 0;
  let h = 0;
  if (d) h = v === r ? 60 * (((g - b) / d) % 6) : v === g ? 60 * ((b - r) / d + 2) : 60 * ((r - g) / d + 4);
  return [(h + 360) % 360, s, v];
}
/** "#rrggbb" for a typed colour (#rgb, #rrggbb, with or without #), else null. */
export function parseHex(t) {
  const m = String(t).trim().replace(/^#/, "").toLowerCase();
  if (/^[0-9a-f]{6}$/.test(m)) return "#" + m;
  if (/^[0-9a-f]{3}$/.test(m)) return "#" + [...m].map(c => c + c).join("");
  return null;
}

// a quiet palette for common choices: greys, warm, cool, and the app accent
const SWATCHES = ["#ffffff", "#c9ccd1", "#8c9099", "#3d4047", "#d94a4a", "#ef7d4b", "#f2c94c", "#7cb342", "#2bb3a3", "#4aa3df", "#3058d0", "#8e6ad8", "#d65db1", "#a0785a"];
let pop = null, cur = null;

function build() {
  pop = document.createElement("div");
  pop.className = "cpick glass"; pop.hidden = true;
  pop.setAttribute("role", "dialog"); pop.setAttribute("aria-label", "Colour");
  pop.innerHTML = `<div class="cp-t"></div><div class="cp-sv" tabindex="0" aria-label="Saturation and brightness"><span class="cp-knob"></span></div>` +
    `<input type="range" class="cp-hue" min="0" max="359" step="1" aria-label="Hue">` +
    `<div class="cp-sw">${SWATCHES.map(c => `<button type="button" data-c="${c}" style="--c:${c}" title="${c}" aria-label="${c}"></button>`).join("")}</div>` +
    `<div class="cp-row"><span class="cp-now"></span><input class="cp-hex" spellcheck="false" maxlength="7" aria-label="Hex colour"><button type="button" class="cp-def">Default</button></div>`;
  document.body.append(pop);
  const sv = pop.querySelector(".cp-sv");
  const fromPointer = e => {
    const r = sv.getBoundingClientRect();
    cur.s = Math.min(1, Math.max(0, (e.clientX - r.left) / (r.width || 1)));
    cur.v = Math.min(1, Math.max(0, 1 - (e.clientY - r.top) / (r.height || 1)));
    set(hsvToHex(cur.h, cur.s, cur.v), true);
  };
  sv.addEventListener("pointerdown", e => {
    e.preventDefault(); fromPointer(e);
    if (sv.setPointerCapture) sv.setPointerCapture(e.pointerId);
    const move = ev => fromPointer(ev), up = () => { sv.removeEventListener("pointermove", move); sv.removeEventListener("pointerup", up); };
    sv.addEventListener("pointermove", move); sv.addEventListener("pointerup", up);
  });
  sv.addEventListener("keydown", e => {
    const d = { ArrowLeft: [-0.02, 0], ArrowRight: [0.02, 0], ArrowUp: [0, 0.02], ArrowDown: [0, -0.02] }[e.key];
    if (!d) return;
    e.preventDefault();
    cur.s = Math.min(1, Math.max(0, cur.s + d[0])); cur.v = Math.min(1, Math.max(0, cur.v + d[1]));
    set(hsvToHex(cur.h, cur.s, cur.v), true);
  });
  pop.querySelector(".cp-hue").addEventListener("input", e => { cur.h = +e.target.value; set(hsvToHex(cur.h, cur.s, cur.v), true); });
  pop.querySelector(".cp-sw").addEventListener("click", e => { const b = e.target.closest("[data-c]"); if (b) set(b.dataset.c); });
  const hexIn = pop.querySelector(".cp-hex");
  hexIn.addEventListener("input", () => { const c = parseHex(hexIn.value); if (c) set(c, false, true); });
  hexIn.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); closePicker(); } });
  pop.querySelector(".cp-def").addEventListener("click", () => { if (cur && cur.def) set(cur.def); });
  // outside pointer: close and keep; Esc: close and restore
  document.addEventListener("pointerdown", e => { if (cur && !pop.contains(e.target) && !(cur.anchor && cur.anchor.contains(e.target))) closePicker(); }, true);
  document.addEventListener("keydown", e => { if (cur && e.key === "Escape") { e.stopPropagation(); e.preventDefault(); closePicker(true); } }, true);
  window.addEventListener("resize", () => { if (cur) place(); });
}
/** Shows colour c in the picker and reports it. keepHsv: c came from the square or the slider (keeps the hue of greys);
 *  fromHex: c came from the hex field (leave the field as typed). */
function set(c, keepHsv = false, fromHex = false) {
  if (!keepHsv) { const [h, s, v] = hexToHsv(c); if (s) cur.h = h; cur.s = s; cur.v = v; }
  cur.value = c;
  paint(fromHex);
  if (cur.onInput) cur.onInput(c);
}
function paint(fromHex = false) {
  pop.style.setProperty("--h", String(cur.h));
  const k = pop.querySelector(".cp-knob");
  k.style.left = `${100 * cur.s}%`; k.style.top = `${100 * (1 - cur.v)}%`; k.style.background = cur.value;
  pop.querySelector(".cp-hue").value = String(Math.round(cur.h));
  pop.querySelector(".cp-now").style.background = cur.value;
  if (!fromHex) pop.querySelector(".cp-hex").value = cur.value;
  for (const b of pop.querySelectorAll(".cp-sw [data-c]")) b.setAttribute("aria-pressed", String(b.dataset.c === cur.value));
  const def = pop.querySelector(".cp-def");
  def.hidden = !cur.def; def.disabled = cur.value === cur.def;
}
/** Below the anchor (above it when there is no room), inside the window. */
function place() {
  const a = cur.anchor.getBoundingClientRect(), w = pop.offsetWidth || 236, h = pop.offsetHeight || 260, W = window.innerWidth, H = window.innerHeight;
  let x = Math.min(Math.max(8, a.left), W - w - 8), y = a.bottom + 6;
  if (y + h > H - 8) y = Math.max(8, a.top - h - 6);
  pop.style.left = `${x}px`; pop.style.top = `${y}px`;
}
/**
 * Opens the picker at element anchor. opts: value ("#rrggbb"), def (the default colour, shown as Default; omit for
 * none), title (a line at the top), onInput(hex) at every change, onClose(hex, cancelled) when it closes. Opening it
 * again from the same anchor closes it.
 */
export function openPicker(anchor, opts) {
  if (!pop) build();
  if (cur && cur.anchor === anchor) { closePicker(); return; }
  if (cur) closePicker();
  const v = parseHex(opts.value) || "#888888", [h, s, val] = hexToHsv(v);
  cur = { anchor, h, s, v: val, value: v, start: v, def: opts.def ? parseHex(opts.def) : null, onInput: opts.onInput, onClose: opts.onClose };
  pop.querySelector(".cp-t").textContent = opts.title || "";
  pop.querySelector(".cp-t").hidden = !opts.title;
  pop.hidden = false;
  anchor.setAttribute("aria-expanded", "true");
  paint(); place();
}
/** Closes the picker; cancel restores the colour it opened with. */
export function closePicker(cancel = false) {
  if (!cur) return;
  const c = cur;
  cur = null;
  pop.hidden = true;
  c.anchor.setAttribute("aria-expanded", "false");
  if (cancel && c.value !== c.start && c.onInput) c.onInput(c.start);
  if (c.onClose) c.onClose(cancel ? c.start : c.value, cancel);
}
/** True while the picker is open. */
export const pickerOpen = () => !!cur;
