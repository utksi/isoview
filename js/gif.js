/**
 * Animated GIF (GIF89a) writer for exported movies.
 *
 * Palette: one global table for all frames, so colours do not flicker. buildPalette runs median cut on RGB samples
 * (every k-th pixel of a few frames): starting from one box holding all samples, the box with the widest channel range
 * is split at the median of that channel until there are 255 boxes; each colour is the mean of its box. Index 255 is
 * reserved as the transparent colour.
 * Mapping: a pixel takes the palette colour nearest in squared RGB distance; results are cached on the colour
 * truncated to 5 bits per channel (32768 entries), so the search runs once per distinct cell.
 * Frames: frame 0 is stored whole. Each later frame stores the bounding box of the pixels whose palette index changed,
 * with unchanged pixels inside the box set to the transparent index; disposal method 1 (keep) leaves the rest of the
 * previous frame on screen. A frame with no change only extends the previous delay.
 * LZW: minimum code size 8, clear code 256, end code 257, codes of 9 to 12 bits packed LSB first, a clear code when
 * the table reaches 4096 entries, data in sub-blocks of at most 255 bytes.
 * Delays are integer centiseconds (GIF units); callers carry the rounding remainder so the mean rate is the chosen fps.
 */

/** Median-cut palette of 255 colours (+ transparent index 255) from RGBA sample arrays. Returns Uint8Array(768). */
export function buildPalette(images, maxSamples = 120000) {
  const total = images.reduce((a, d) => a + d.length / 4, 0), every = Math.max(1, Math.floor(total / maxSamples));
  const px = [];
  for (const d of images) for (let i = 0; i < d.length; i += 4 * every) px.push((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
  const ch = (c, k) => (c >> (16 - 8 * k)) & 255;
  const boxes = [px];
  const range = b => {
    let best = -1, bk = 0;
    for (let k = 0; k < 3; k++) {
      let lo = 255, hi = 0;
      for (const c of b) { const v = ch(c, k); if (v < lo) lo = v; if (v > hi) hi = v; }
      if (hi - lo > best) { best = hi - lo; bk = k; }
    }
    return { r: best, k: bk };
  };
  while (boxes.length < 255) {
    let bi = -1, br = 0, bk = 0;
    boxes.forEach((b, i) => { if (b.length < 2) return; const { r, k } = range(b); if (r > br) { br = r; bi = i; bk = k; } });
    if (bi < 0) break;
    const b = boxes[bi].sort((p, q) => ch(p, bk) - ch(q, bk)), m = b.length >> 1;
    boxes.splice(bi, 1, b.slice(0, m), b.slice(m));
  }
  const pal = new Uint8Array(768);
  boxes.forEach((b, i) => {
    let r = 0, g = 0, bl = 0;
    for (const c of b) { r += ch(c, 0); g += ch(c, 1); bl += ch(c, 2); }
    const n = b.length || 1;
    pal[3 * i] = Math.round(r / n); pal[3 * i + 1] = Math.round(g / n); pal[3 * i + 2] = Math.round(bl / n);
  });
  return pal;
}

/** Growable byte buffer. */
class Bytes {
  constructor(n = 1 << 16) { this.a = new Uint8Array(n); this.n = 0; }
  need(k) { if (this.n + k > this.a.length) { const b = new Uint8Array(Math.max(2 * this.a.length, this.n + k)); b.set(this.a.subarray(0, this.n)); this.a = b; } }
  byte(v) { this.need(1); this.a[this.n++] = v; }
  u16(v) { this.need(2); this.a[this.n++] = v & 255; this.a[this.n++] = (v >> 8) & 255; }
  bytes(arr) { this.need(arr.length); this.a.set(arr, this.n); this.n += arr.length; }
  done() { return this.a.slice(0, this.n); }
}

/** LZW-compress palette indices (min code size 8) into out, as GIF image data sub-blocks. */
function lzw(idx, out) {
  const CLEAR = 256, EOI = 257, HS = 8192, keys = new Int32Array(HS), vals = new Int16Array(HS), gen = new Int32Array(HS);
  let g = 1, next = EOI + 1, size = 9, acc = 0, nbits = 0;
  const block = new Uint8Array(255);
  let bn = 0;
  const flushByte = b => { block[bn++] = b; if (bn === 255) { out.byte(255); out.bytes(block); bn = 0; } };
  const emit = code => { acc |= code << nbits; nbits += size; while (nbits >= 8) { flushByte(acc & 255); acc >>>= 8; nbits -= 8; } };
  // WHY: open-addressed table with a generation stamp, so a clear code empties it without touching 8192 slots
  const find = key => { let h = Math.imul(key, 2654435761) >>> 19; while (gen[h] === g) { if (keys[h] === key) return h; h = (h + 1) & (HS - 1); } return ~h; };
  out.byte(8);
  emit(CLEAR);
  let cur = idx[0];
  for (let i = 1; i < idx.length; i++) {
    const k = idx[i], key = (cur << 8) | k, h = find(key);
    if (h >= 0) { cur = vals[h]; continue; }
    emit(cur);
    if (next === 4096) { emit(CLEAR); next = EOI + 1; size = 9; g++; }
    else {
      if (next >= 1 << size) size++;
      const s = ~h; gen[s] = g; keys[s] = key; vals[s] = next++;
    }
    cur = k;
  }
  emit(cur); emit(EOI);
  if (nbits > 0) flushByte(acc & 255);
  if (bn) { out.byte(bn); out.bytes(block.subarray(0, bn)); }
  out.byte(0);
}

/** GIF89a writer: new GifWriter(w, h, palette); frame(rgba, delayCs) for each frame; finish() -> Uint8Array. */
export class GifWriter {
  constructor(w, h, palette, { loop = 0 } = {}) {
    this.w = w; this.h = h; this.pal = palette; this.out = new Bytes(w * h);
    this.cache = new Int16Array(32768).fill(-1); this.prev = null; this.pending = null;
    const o = this.out;
    o.bytes([71, 73, 70, 56, 57, 97]); o.u16(w); o.u16(h);
    o.byte(0xf7); o.byte(0); o.byte(0); // global table of 256 colours, background 0, no aspect ratio
    o.bytes(palette);
    o.bytes([0x21, 0xff, 11, ...[..."NETSCAPE2.0"].map(c => c.charCodeAt(0)), 3, 1]); o.u16(loop); o.byte(0);
  }
  /** Palette index of an RGB colour (nearest colour among indices 0..254, cached on 5-bit cells). */
  index(r, g, b) {
    const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
    let v = this.cache[key];
    if (v >= 0) return v;
    let best = 0, bd = Infinity;
    const p = this.pal;
    for (let i = 0; i < 255; i++) { const dr = p[3 * i] - r, dg = p[3 * i + 1] - g, db = p[3 * i + 2] - b, d = dr * dr + dg * dg + db * db; if (d < bd) { bd = d; best = i; } }
    this.cache[key] = best;
    return best;
  }
  /** Add one frame (RGBA bytes, w*h*4) shown for delayCs centiseconds. */
  frame(rgba, delayCs) {
    const { w, h } = this, n = w * h, cur = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) cur[i] = this.index(rgba[j], rgba[j + 1], rgba[j + 2]);
    let x0 = 0, y0 = 0, x1 = w - 1, y1 = h - 1;
    if (this.prev) {
      x0 = w; y0 = h; x1 = -1; y1 = -1;
      for (let y = 0; y < h; y++) for (let x = 0, i = y * w; x < w; x++, i++) if (cur[i] !== this.prev[i]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      if (x1 < 0) { if (this.pending) this.pending.delay += delayCs; return; }
    }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1, sub = new Uint8Array(bw * bh);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const i = (y + y0) * w + x + x0;
      sub[y * bw + x] = this.prev && cur[i] === this.prev[i] ? 255 : cur[i];
    }
    this.flush();
    this.pending = { x0, y0, bw, bh, sub, delay: delayCs, transparent: !!this.prev };
    this.prev = cur;
  }
  /** Write the frame held back (its delay may still grow while the next frames are unchanged). */
  flush() {
    const f = this.pending, o = this.out;
    if (!f) return;
    o.bytes([0x21, 0xf9, 4, (1 << 2) | (f.transparent ? 1 : 0)]); o.u16(Math.min(65535, f.delay)); o.byte(255); o.byte(0);
    o.byte(0x2c); o.u16(f.x0); o.u16(f.y0); o.u16(f.bw); o.u16(f.bh); o.byte(0);
    lzw(f.sub, o);
    this.pending = null;
  }
  /** The finished file. */
  finish() { this.flush(); this.out.byte(0x3b); return this.out.done(); }
}
