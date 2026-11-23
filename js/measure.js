// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Geometry of a selection of 2 to 4 atoms (Cartesian positions in A).
 *   distance  |b - a|
 *   angle     at the middle atom b: acos( (a - b).(c - b) / (|a - b| |c - b|) ), degrees
 *   dihedral  a-b-c-d: atan2( |b2| b1.(b2 x b3), (b1 x b2).(b2 x b3) ), b1 = b - a, b2 = c - b, b3 = d - c, degrees in
 *             (-180, 180] (IUPAC sign: positive for clockwise rotation of the front bond seen along b -> c).
 */
const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const len = u => Math.hypot(u[0], u[1], u[2]);

export const distance = (a, b) => len(sub(b, a));
export function angle(a, b, c) {
  const u = sub(a, b), v = sub(c, b);
  return Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (len(u) * len(v))))) * 180 / Math.PI;
}
export function dihedral(a, b, c, d) {
  const b1 = sub(b, a), b2 = sub(c, b), b3 = sub(d, c), n1 = cross(b1, b2), n2 = cross(b2, b3);
  return Math.atan2(len(b2) * dot(b1, n2), dot(n1, n2)) * 180 / Math.PI;
}
/** Measurements of a selection: { bonds: [d01, d12, ...], angles: [...], dihedral }. */
export function measureAll(p) {
  const out = { bonds: [], angles: [], dihedral: null };
  for (let i = 0; i + 1 < p.length; i++) out.bonds.push(distance(p[i], p[i + 1]));
  for (let i = 0; i + 2 < p.length; i++) out.angles.push(angle(p[i], p[i + 1], p[i + 2]));
  if (p.length === 4) out.dihedral = dihedral(p[0], p[1], p[2], p[3]);
  return out;
}
