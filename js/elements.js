/**
 * Element data: symbols (Z = 0..103, Z = 0 is the unknown "X"), covalent radii and display colours.
 *
 * Covalent radii (Angstrom): B. Cordero et al., Dalton Trans. 2832 (2008); low-spin values for Mn, Fe, Co; 1.50 A
 * where no value is tabulated. Display colours: the Jmol scheme mixed 20% toward neutral grey, which keeps the
 * familiar hues but tones down the saturated ones (Cl, Mg, Sr ...).
 */

export const SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As " +
  "Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb " +
  "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split(" ");

const RCOV = [1.5, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05,
  1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20,
  1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
  2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70,
  1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96,
  1.90, 1.87, 1.80, 1.69, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];

const JMOL = ("808080 FFFFFF D9FFFF CC80FF C2FF00 FFB5B5 909090 3050F8 FF0D0D 90E050 B3E3F5 AB5CF2 8AFF00 BFA6A6 " +
  "F0C8A0 FF8000 FFFF30 1FF01F 80D1E3 8F40D4 3DFF00 E6E6E6 BFC2C7 A6A6AB 8A99C7 9C7AC7 E06633 F090A0 50D050 C88033 " +
  "7D80B0 C28F8F 668F8F BD80E3 FFA100 A62929 5CB8D1 702EB0 00FF00 94FFFF 94E0E0 73C2C9 54B5B5 3B9E9E 248F8F 0A7D8C " +
  "006985 C0C0C0 FFD98F A67573 668080 9E63B5 D47A00 940094 429EB0 57178F 00C900 70D4FF FFFFC7 D9FFC7 C7FFC7 A3FFC7 " +
  "8FFFC7 61FFC7 45FFC7 30FFC7 1FFFC7 00FF9C 00E675 00D452 00BF38 00AB24 4DC2FF 4DA6FF 2194D6 267DAB 266696 175487 " +
  "D0D0E0 FFD123 B8B8D0 A6544D 575961 9E4FB5 AB5C00 754F45 428296 420066 007D00 70ABFA 00BAFF 00A1FF 008FFF 0080FF " +
  "006BFF 545CF2 785CE3 8A4FE3 A136D4 B31FD4 B31FBA B30DA6 BD0D87 C70066").split(" ");

const Z_OF = new Map(SYMBOLS.map((s, z) => [s, z]));

/** Atomic number of a symbol (0 if unknown). */
export function zOf(sym) { return Z_OF.get(sym) ?? 0; }

/** Element symbol for a label such as "Cs_sv", "Fe1", "O2-", "CL" or an atomic number; "X" if none matches. */
export function elementOf(label) {
  const t = String(label).trim();
  if (/^\d+$/.test(t)) return SYMBOLS[+t] ?? "X";
  const m = t.match(/^[A-Za-z]+/);
  if (!m) return "X";
  const w = m[0];
  const two = w[0].toUpperCase() + (w[1] ? w[1].toLowerCase() : "");
  if (w.length >= 2 && Z_OF.has(two)) return two;
  const one = w[0].toUpperCase();
  return Z_OF.has(one) && one !== "X" ? one : "X";
}

/** Covalent radius (Angstrom) of an element symbol. */
export function rcov(sym) { return RCOV[zOf(sym)] ?? 1.5; }

/** Display colour "#rrggbb": Jmol colour mixed 20% toward grey #8c9099. */
export function colorOf(sym) {
  const h = JMOL[zOf(sym)] ?? "808080";
  const c = [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
  const g = [0x8c, 0x90, 0x99];
  return "#" + c.map((v, i) => Math.round(0.8 * v + 0.2 * g[i]).toString(16).padStart(2, "0")).join("");
}

/** Group 1 and the heavier group 2 elements, which are drawn without bonds by default (A-site cations). */
export const IONIC_CATIONS = new Set(["Li", "Na", "K", "Rb", "Cs", "Fr", "Ca", "Sr", "Ba", "Ra"]);
