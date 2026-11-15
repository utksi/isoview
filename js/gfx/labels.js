/**
 * Label atlas and label instances.
 *
 * Atlas: every distinct string is drawn once into rows of height H px on an opaque black canvas with additive
 * compositing: a round-joined outline (width 0.16 H) in the green channel and the glyph fill in the red channel, so
 * red = fill coverage and green = outline coverage, anti-aliased, independent of alpha premultiplication. The atlas
 * map gives each string its rectangle (u0, v0, u1, v1) in [0, 1] and aspect = width / H.
 * Colours: text is dark (#17191e) on light atoms and white on dark ones (relative luminance Y > 0.55 with
 * Y = 0.2126 R + 0.7152 G + 0.0722 B), with a halo of the opposite tone at 55 % (dark text) or 45 % (white text).
 * Instance layout (44 B): centre f32x3, radius f32, rect f32x4, aspect f32, text rgba u8, halo rgba u8.
 */

export function buildAtlas(strings, H = 112) {
  const c = typeof document !== "undefined" ? document.createElement("canvas") : null;
  const ctx = c && c.getContext ? c.getContext("2d") : null;
  if (!ctx) return null;
  const font = `600 ${Math.round(H * 0.74)}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`;
  ctx.font = font;
  const pad = Math.round(H * 0.22), W = 1024, rects = {};
  let x = 0, y = 0;
  const items = [...new Set(strings)].map(s => ({ s, w: Math.ceil(ctx.measureText(s).width) + 2 * pad }));
  for (const it of items) { if (x + it.w > W) { x = 0; y += H; } rects[it.s] = { x, y, w: Math.min(it.w, W) }; x += it.w; }
  let Ht = 1;
  while (Ht < y + H) Ht <<= 1;
  c.width = W; c.height = Ht;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, Ht);
  ctx.font = font; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.globalCompositeOperation = "lighter"; ctx.lineJoin = "round"; ctx.lineWidth = H * 0.16;
  for (const it of items) {
    const r = rects[it.s], cx = r.x + r.w / 2, cy = r.y + H * 0.54;
    ctx.strokeStyle = "rgb(0,255,0)"; ctx.strokeText(it.s, cx, cy);
    ctx.fillStyle = "rgb(255,0,0)"; ctx.fillText(it.s, cx, cy);
  }
  const map = {};
  for (const it of items) { const r = rects[it.s]; map[it.s] = { uv: [r.x / W, r.y / Ht, (r.x + r.w) / W, (r.y + H) / Ht], aspect: r.w / H }; }
  return { canvas: c, map, id: Math.random() };
}

const rgb = hex => { const h = hex.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };

/** Text and halo colours (rgba bytes) for a label on an atom of colour `hex`. */
export function labelColors(hex) {
  const [r, g, b] = rgb(hex), Y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return Y > 0.55 ? { text: [23, 25, 30, 255], halo: [255, 255, 255, 140] } : { text: [255, 255, 255, 255], halo: [10, 12, 16, 115] };
}

/** labels: [{x, y, z, r, text, color}] with atlas map -> ArrayBuffer(44 N). */
export function packLabels(labels, map) {
  const buf = new ArrayBuffer(44 * labels.length), f = new Float32Array(buf), b = new Uint8Array(buf);
  labels.forEach((l, i) => {
    const m = map[l.text], o = 11 * i;
    f[o] = l.x; f[o + 1] = l.y; f[o + 2] = l.z; f[o + 3] = l.r;
    f.set(m ? m.uv : [0, 0, 0, 0], o + 4); f[o + 8] = m ? m.aspect : 1;
    const c = labelColors(l.color);
    b.set(c.text, 44 * i + 36); b.set(c.halo, 44 * i + 40);
  });
  return buf;
}
