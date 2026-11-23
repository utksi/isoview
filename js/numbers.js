// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Text scanning for structure and volumetric files.
 *
 * readNumbers parses whitespace- or comma-separated decimal numbers directly from a string, without splitting it
 * into substrings, so files with tens of millions of values (CHGCAR, XSF, cube) parse in one pass.
 * Accepted syntax: [+-] digits [. digits] [(e|E|d|D) [+-] digits]; a leading '.' is allowed (".5", "-.12E-02").
 * A sign directly after a number starts the next number ("1.0-2.0" is two values), as Fortran fixed formats
 * sometimes print. Tokens with no digits (e.g. "****", "NaN") are stored as NaN.
 * Value = mantissa * 10^exponent with up to 17 significant mantissa digits; the power of ten comes from an exact
 * table, so results match parseFloat to within one unit in the last place for the formats seen in practice.
 */

const P10 = Array.from({ length: 309 }, (_, i) => Number("1e" + i));

/** True for space, tab, CR, LF and comma (character codes). */
function sep(c) {
  return c === 32 || c === 10 || c === 13 || c === 9 || c === 44;
}

/**
 * Parse n numbers from s starting at character index pos.
 * Returns { out, pos } with out the filled array (Float32Array by default) and pos the index after the last number.
 * Throws if the string ends before n numbers were read.
 */
export function readNumbers(s, pos, n, out = new Float32Array(n)) {
  const L = s.length;
  let i = pos;
  for (let k = 0; k < n; k++) {
    let c = s.charCodeAt(i);
    while (i < L && sep(c)) c = s.charCodeAt(++i);
    if (i >= L) throw new Error(`expected ${n} numbers, found ${k}`);
    let neg = false;
    if (c === 45) { neg = true; c = s.charCodeAt(++i); } else if (c === 43) { c = s.charCodeAt(++i); }
    let m = 0, e = 0, nd = 0;
    while (c >= 48 && c <= 57) { if (nd < 17) { m = m * 10 + (c - 48); nd++; } else e++; c = s.charCodeAt(++i); }
    if (c === 46) {
      c = s.charCodeAt(++i);
      while (c >= 48 && c <= 57) { if (nd < 17) { m = m * 10 + (c - 48); nd++; e--; } c = s.charCodeAt(++i); }
    }
    if (nd === 0) {
      // not a number: skip the token and store NaN
      const start = i;
      while (i < L && !sep(s.charCodeAt(i))) i++;
      if (i === start) i++;
      out[k] = NaN;
      continue;
    }
    if (c === 69 || c === 101 || c === 68 || c === 100) {
      c = s.charCodeAt(++i);
      let en = false;
      if (c === 45) { en = true; c = s.charCodeAt(++i); } else if (c === 43) { c = s.charCodeAt(++i); }
      let x = 0;
      while (c >= 48 && c <= 57) { x = x * 10 + (c - 48); c = s.charCodeAt(++i); }
      e += en ? -x : x;
    }
    let v;
    if (e === 0) v = m;
    else if (e > 0) v = e > 308 ? Infinity : m * P10[e];
    else v = -e > 308 ? (m / P10[308]) / Math.pow(10, -e - 308) : m / P10[-e];
    out[k] = neg ? -v : v;
  }
  return { out, pos: i };
}

/** Sequential reader over a string: lines (without the newline, CR stripped) and blocks of numbers. */
export class Cursor {
  constructor(s, pos = 0) { this.s = s; this.pos = pos; }
  /** Next line, or null at the end of the string. */
  line() {
    const s = this.s;
    if (this.pos >= s.length) return null;
    let e = s.indexOf("\n", this.pos);
    if (e < 0) e = s.length;
    let l = s.slice(this.pos, e);
    if (l.charCodeAt(l.length - 1) === 13) l = l.slice(0, -1);
    this.pos = e + 1;
    return l;
  }
  /** Next line that is not blank (null at the end). */
  nonEmpty() {
    let l;
    while ((l = this.line()) !== null && !l.trim());
    return l;
  }
  /** n numbers from the current position (Float32Array unless out is given). */
  numbers(n, out) {
    const r = readNumbers(this.s, this.pos, n, out);
    this.pos = r.pos;
    return r.out;
  }
}

/** Whitespace-split tokens of a line (empty array for blank lines). */
export function tokens(line) {
  const t = line.trim();
  return t ? t.split(/\s+/) : [];
}
