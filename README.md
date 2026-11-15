# isoview

Crystal structures and volumetric data in the browser. Open a file, turn it around, pick an isovalue, click atoms to
measure them. Everything runs locally: files are read on your device and never uploaded.

**Open it:** https://utkarsh.science/isoview/

## Formats

| Kind | Files |
|---|---|
| Structures | POSCAR / CONTCAR (VASP 4 and 5+), CIF (symmetry operations applied), XSF, Quantum ESPRESSO `pw.x` input (all `ibrav`, every unit option), cube, xyz / extended xyz, FHI-aims `geometry.in` |
| Volumetric | CHGCAR, CHG, PARCHG, AECCAR (as e/Å³; spin-polarised files give total, magnetization, up, down), LOCPOT (eV), ELFCAR, XSF `DATAGRID_3D`, Gaussian cube |
| Wavefunctions | WAVECAR from `vasp_std`, `vasp_gam` and `vasp_ncl`: pick spin, k-point and band, and see \|ψ\|² (or ψ itself for Γ-only builds) |

Drop several files at once to combine them, e.g. a `WAVECAR` with its `CONTCAR`. You can also paste a structure
as text, or link to a file with `?url=https://...` (the host must allow cross-origin requests).

## Using it

- **Structure**: ball-and-stick, space filling or sticks; bond reach; supercell repeat; atoms on cell faces.
  Alkali and heavy alkaline-earth cations are drawn without bonds unless you switch them on.
- **Surface**: the level slider is logarithmic for densities and orbitals and linear for potentials and ELF.
  The readout gives the share of the charge (or of ∫|f| for signed fields) inside the surface.
  Signed fields show a positive and a negative surface.
- **Orbital** (WAVECAR): the band list marks occupied bands; arrow keys step through bands.
  Only the requested band is read from disk, so multi-gigabyte files open in seconds.
- **Layers**: every file adds layers (its atoms, its field). Drag files onto the right half of the window, or use
  "Add files to the scene", to superimpose them: a PARCHG on a CHGCAR, a CONTCAR on a WAVECAR, two orbitals from
  different runs. Each field layer has its own level, opacity and colours; layers can be hidden or removed.
- **Shift origin** by fractions (fx, fy, fz) of the cell: atoms and fields move together. Periodic fields (CHGCAR,
  WAVECAR, periodic XSF) shift exactly, by whole grid steps plus a sub-step origin offset. With one atom selected,
  Center moves it to the middle of the cell.
- **Measure**: click two atoms for their distance, three for the angle at the middle one, four for the dihedral.
  Click empty space to clear.
- **Edit**: Shift + drag (or the box tool) selects the atoms in a box. Delete removes the selected atoms (in every
  periodic image); Cut surface removes the isosurface inside the box (the region stays fixed as you rotate).
  Ctrl/⌘ Z undoes deletions, cuts, shifts and removed layers. Export POSCAR saves the edited, shifted structure.
- **Labels**: switch on label mode (tag button or `L`) and click atoms to put their symbol on them; click again to remove.
  Labels stay when you leave the mode and are hidden by atoms in front of them, like the rest of the scene.
- **View**: perspective or orthographic, light or dark. Keys: `O` open, `R` reset, `A` `B` `C` view along a lattice
  vector, `L` labels, `S` save a PNG, `Del` delete, `Esc` clear. Right-drag pans. The isoview wordmark returns to the
  opening page.

## How it draws

WebGPU where the browser has it, WebGL2 otherwise (`?gl=webgl2` forces the fallback). Atoms are ray-cast spheres on
camera-facing quads (exact silhouettes and depth, one instanced draw call), bonds are instanced cylinders, cell edges
constant-width lines, and frames are drawn only when something changes. Isosurfaces come from marching cubes in a Web
Worker (shared vertices, gradient normals); translucent surfaces use a depth pre-pass so only their front layer is
blended. Parsing, WAVECAR transforms and meshing all run off the main thread.

## How the numbers are made

- VASP charge files store ρ·V; isoview divides by the cell volume, so ∫ρ dV over a CHGCAR equals the number of electrons.
- WAVECAR orbitals are rebuilt from their plane-wave coefficients with a 3D FFT (the G-vector list is regenerated from
  ENCUT and checked against the stored count). They are the smooth part of the PAW orbitals, normalised to one
  electron; the augmentation inside the PAW spheres is not included.
- Bonds join atoms closer than reach × (R₁ + R₂), with covalent radii from Cordero et al. (2008).

## Develop

A static site with no build step. Serve the folder with any static server (`python3 -m http.server`) and open it.
Tests of the readers, meshing, camera and measurement maths run in Node: `node tests/run.mjs`. The marching-cubes case
table is the classic Lorensen-Cline one, taken from [3Dmol.js](https://3dmol.org) (BSD-3-Clause).

MIT licence.
