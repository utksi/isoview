# isoview

Visualize, edit and create crystal structures and volumetric data, in the browser. Open a file, turn it around, pick
an isovalue, click atoms to measure them, cut and shift the cell, and save what you made. Everything runs locally:
files are read on your device and never uploaded.

**Open it:** https://utkarsh.science/isoview/

## Formats

| Kind | Files |
|---|---|
| Structures | POSCAR / CONTCAR (VASP 4 and 5+), CIF (symmetry operations applied), XSF, Quantum ESPRESSO `pw.x` input (all `ibrav`, every unit option), cube, xyz / extended xyz, FHI-aims `geometry.in` |
| Volumetric | CHGCAR, CHG, PARCHG, AECCAR (as e/Å³; spin-polarised files give total, magnetization, up, down), LOCPOT (eV), ELFCAR, XSF `DATAGRID_3D`, Gaussian cube |
| Wavefunctions | WAVECAR from `vasp_std`, `vasp_gam` and `vasp_ncl`: pick spin, k-point and band, and see \|ψ\|² (or ψ itself for Γ-only builds) |
| Trajectories | XDATCAR (fixed or changing cell), OUTCAR and vasprun.xml (every ionic step, with its energy), ASE `.traj`, xyz / extended xyz (several frames, `energy=`), animated XSF (AXSF), LAMMPS text dumps (orthogonal or triclinic box; element, mass or type columns), Quantum ESPRESSO `pw.x` output (relax, md, vc-relax), PDB with MODELs, GROMACS `.gro`, CASTEP `.md` / `.geom` |

Gzip-compressed files (`vasprun.xml.gz`, `CHGCAR.gz`, ...) open directly. The opening page offers a sample of most
formats, including real VASP, Quantum ESPRESSO and CASTEP outputs from public test data; see
[`examples/README.md`](examples/README.md) for where each comes from.

Drop several files at once to combine them, e.g. a `WAVECAR` with its `CONTCAR`. You can also paste a structure
as text, or link to a file with `?url=https://...` (the host must allow cross-origin requests).

## In your website, notebooks and editor

- **Website**: "Embed in your website" (top right of the opening page, or the `</>` view button once a file is open,
  which also has the notebook snippets with the scene's file names) gives an `<iframe>` that shows a file from any
  address that allows cross-origin requests (GitHub Pages and raw.githubusercontent.com do):
  `https://utkarsh.science/isoview/?embed&url=https://.../CHGCAR`. Options: `&theme=light|dark`, `&style=stick`,
  `&look=soft`, `&panel=1` (start with the panel open). In embedded mode nothing is kept in the visitor's browser and
  the wordmark opens the full site.
- **Jupyter** ([`python/`](python/)): `pip install "isoview-notebook @ git+https://github.com/utksi/isoview#subdirectory=python"`,
  then `isoview.show("CHGCAR")`, `isoview.show("WAVECAR", "CONTCAR")` or `isoview.show(atoms)` (ASE Atoms, a list of
  them as a trajectory, or a pymatgen Structure). JupyterLab, Notebook, VS Code and Colab.
- **Julia** ([`julia/`](julia/)): `Pkg.add(url="https://github.com/utksi/isoview", subdir="julia")`, then
  `using Isoview; isoview("CHGCAR")`. Pluto, IJulia and VS Code.
- **VS Code** ([`vscode/`](vscode/)): "Open in isoview" on a file in the Explorer, also over Remote SSH (not yet on
  the Marketplace).

The notebook packages and the extension load the viewer from this site in the notebook's or editor's own window and
hand it the files there, so nothing is uploaded; that window needs internet access.

## Using it

- **Structure**: ball-and-stick, space filling or sticks; bond reach (back to 1.15 when another file is opened, unless "keep" beside it is on); supercell repeat; atoms on cell faces; the cell
  or box outline can be switched off for anything that has one (crystals, molecules in a box, cube grids, WAVECARs).
  Alkali and heavy alkaline-earth cations get no bonds when the rest of the structure is bonded (Cs in a halide perovskite), and do get them when nothing else would be (NaCl, CaF₂).
  **Show elements** has a chip for each element, with its atom count: click the element to pick its colour in a
  small picker on the page (a saturation-brightness square, a hue slider, swatches, a hex field and Default; the
  scene follows as you move), untick it to hide those atoms and their
  bonds: the bonds are still found among all atoms, so hiding Cl does not bond the cations it kept apart. Hiding
  only changes the view; Export as shown leaves hidden elements out and says so. Opening a new file shows every
  element again.
- **Surface**: coarse grids are drawn on a finer grid so surfaces are smooth (a step of at most 0.1 Å, up to 4 M
  points): periodic data by Fourier interpolation, which is exact for plane-wave densities and potentials and keeps
  the integral; other grids (cube) by cubic interpolation. WAVECAR orbitals are built on a larger FFT box, which is
  exact. The stats line gives the grid as read and the one drawn. The level slider is logarithmic for densities and orbitals and linear for potentials and ELF.
  The readout gives the share of the charge (or of ∫|f| for signed fields) inside the surface, and for a charge
  density also the number of electrons there; a short note under it says how to read the level for that kind of
  field (density, signed field, potential, ELF).
  Signed fields show a positive and a negative surface.
- **Slice** (Surface panel): a 2D cut through the field on a lattice plane. The slice bar (bottom centre) picks
  (100), (010) or (001) and moves the plane through the cell with a slider (its position as a fractional coordinate
  and in Å); any plane (hkl) is set in the panel. The bar also chooses the main view, **3D** (the coloured plane in
  the scene) or **2D** (the slice filling the window), the **inset** (the other view in a small box: drag its top
  handle to move it, its corner to resize it, and turn the 3D view inside it) and the **colour bar** (beside the 2D
  slice, or at the right of the 3D view alone). The 2D slice has contour lines, the atoms within 0.8 Å of the plane
  and a colour bar with ticks and the field's unit: densities on a log scale over three decades, signed fields
  (orbitals, magnetization) blue to red about zero, potentials and ELF linear (scale and colours can be changed in
  the panel). It is sampled with tricubic interpolation of the (already Fourier-refined) grid, at four pixels per
  grid step or finer when it is drawn large. Photo Mode saves the views as shown; the panel also saves the 2D slice
  as SVG (image, vector contour lines, atoms and colour bar).
- **Orbital** (WAVECAR): the band list marks occupied bands; arrow keys step through bands.
  Only the requested band is read from disk, so multi-gigabyte files open in seconds.
  **Add a band as a layer** (or Ctrl/⌘-click a band) shows several bands at once, for example a hole in spin ↓
  and an electron in spin ↑: each band layer has its own spin, k-point, level and colour, and the band list marks
  the bands on show in their surface colours. Spin-up bands are drawn in warm colours and spin-down bands in blues.
- **Several fields of one file**: **Add a field as a layer** (Surface panel) shows another channel of the same
  file next to the first, such as a CHGCAR's spin up and spin down, or its total density and magnetization. Spin up is
  warm, spin down blue, and a magnetization shows its spin-up excess warm and its spin-down excess blue.
- **Layers**: every file adds layers (its atoms, its field). Drag files onto the right half of the window, or use
  "Add files to the scene", to superimpose them: a PARCHG on a CHGCAR, a CONTCAR on a WAVECAR, two orbitals from
  different runs. Each field layer has its own level, opacity and colours; layers can be hidden or removed.
  Under the list, for the chosen structure (click its row):
  - **Primitive cell** and **Conventional cell** add the reduced cell as a new layer (the original is hidden, and
    Undo removes it); a cell that already is primitive or conventional is left as it is. Both work from supercells: a 3×3×3 rock-salt supercell gives the 2-atom primitive and the
    8-atom cubic cell, also with positional noise below the tolerance (copies are averaged). The conventional cell
    follows the lattice's own symmetry (cubic, tetragonal, hexagonal, rhombohedral as its triple hexagonal cell,
    orthorhombic with the centred face as C, monoclinic, triclinic).
    The tolerance (± next to the buttons, 1e-5 Å by default) applies to the atom positions and to the cell vectors
    alike, so a cell 1e-4 off cubic is tetragonal. Raise it (e.g. to 0.01 Å) for noisy structures such as an MD frame.
  - **Export as shown** saves the structure as you see it (shift, deletions and supercell applied) as POSCAR, CIF or
    extended XYZ.
- **Measure**: click two atoms for their distance, three for the angle at the middle one, four for the dihedral.
  Click empty space to clear. The bar that shows the selection also offers Center and Delete (Cut surface for a
  region), so the Edit panel need not be open.
- **Edit** (its own panel): draw a region around atoms with Shift + drag (or switch on "Draw a region"); the region
  is freehand, not a box. Delete removes the selected atoms in every periodic image; Cut surface removes the
  isosurface inside the region (the region stays fixed as you rotate), and Restore the surface brings it back.
  **Shift origin** moves atoms, fields and every trajectory frame together: by fractions of a, b, c for crystals
  (periodic fields shift exactly, by whole grid steps plus a sub-step origin offset), by fractions of the box for a
  molecule in a box (cube files, clusters), and in Å when nothing has a cell (xyz, PDB). **Center on atom** puts the
  selected atom in the middle of its cell or box; without a cell it makes the view turn about that atom. The Undo button (or Ctrl/⌘ Z) names what it will undo: deletions, cuts, shifts,
  removed layers, derived cells.
- **Trajectories**: a player appears under the view: play and pause (`Space`), drag through frames, step with `←` `→`
  (`Shift` for ten, `Home` / `End` for the ends). The Trajectory panel sets the speed (1-60 frames per second), shows
  every n-th frame, and loops, bounces or stops at the end; when the file has energies it draws them per frame (click
  the line to jump to a frame). Edits apply to every frame: a deleted atom is gone from the whole trajectory, and
  measurements follow the atoms as they move.
  **Movies**: GIF or video (MP4 or WebM, whichever the browser records) at a chosen speed, frame range and size. GIFs
  share one palette over the whole movie (no flicker) and store only what changed between frames, so they stay small.
  Videos are recorded in real time. **All frames as XYZ** saves the trajectory with the shift and deletions applied.
  **Analysis** (Trajectory panel) opens a chart window: the radial distribution function g(r), total and for each
  pair of elements, with the first peak and the coordination number to the first minimum; the mean squared
  displacement of each element with the diffusion coefficient from its slope; the vibrational density of states
  from the velocity autocorrelation; and per-frame energy, temperature estimate, volume, density and cell lengths.
  Time-dependent results need the time between saved frames (Time step, in fs). It is filled in from VASP
  molecular dynamics (IBRION = 0: POTIM, in vasprun.xml and OUTCAR) and pw.x md output; other files do not store it,
  and the value typed is kept with the trajectory. Without one, the VDOS is drawn for 1 fs per frame and says so (its
  frequencies scale as 1 / time step); MSD stays in frames. Each curve has a chip below the chart that switches it off and on; per frame, the chips choose the
  quantities, drawn as panels over one time axis. Figures save as SVG or 600 dpi PNG, 3.5 in wide (one journal
  column), with only the curves shown, and the same curves as CSV; deleted atoms are left out. The window belongs to
  the scene: opening other files closes it.
- **X-ray diffraction** (Structure panel, XRD pattern; for a trajectory also a tab of Analysis): the powder pattern
  as pymatgen's XRDCalculator computes it (the same peaks, intensities and hkl to 1e-6; atomic form factors with
  the Mott-Bethe relation, Lorentz-polarisation factor, no thermal factor). Cu, Mo, Co, Fe, Cr or Ag Kα or any
  wavelength, a 2θ range and a peak width. The chart has the broadened profile, the peaks as sticks and the hkl of
  the strongest peaks (each can be switched off), named in the conventional cell (four indices for hexagonal cells),
  so a primitive fcc cell shows 111, 220, 311; the result lists the strongest peaks with d and multiplicities, and
  the CSV has the profile and every peak. For a trajectory: **Frame** (the pattern of the frame shown, following the
  player), **Average** (over up to 200 frames, each with its own cell) or **Map** (2θ against time on one intensity
  scale, logarithmic or linear colours, a dashed line at the frame shown).
- **Brillouin zone** (Structure panel, or `K`): the view switches from the crystal to its first Brillouin zone,
  with b₁ b₂ b₃, the high-symmetry points and the standard k-path of Setyawan and Curtarolo (2010), as used by
  pymatgen and most band-structure plots; the line under the view names the lattice and the path. The zone follows
  the lattice of the structure (at the cell tolerance), and switching back restores the crystal and its view.
  A **crystal inset** (bottom left) shows the structure turning with the zone, since both share one Cartesian frame:
  drag in it to turn both, scroll to zoom it, drag its corner to resize it; Crystal inset (Structure panel) hides it.
- **Look**: Classic, Soft (light from above and depth fog, as in many papers), Glossy, Toon (three tones, a highlight
  and outlines) and Outline (flat colours with ink lines, like a drawing). Outlines are pixel-exact on atoms and bonds.
  A first visit starts in Glossy; after that the look last chosen stays, also for new scenes and later visits.
- **Labels**: switch on label mode (tag button or `L`) and click atoms to put their symbol on them; click again to remove.
  The bar that appears also has **Label all** (every atom drawn) and Clear.
  Labels stay when you leave the mode and are hidden by atoms in front of them, like the rest of the scene.
- **View**: perspective or orthographic, light or dark. Keys: `O` open, `R` reset, `A` `B` `C` view along a lattice
  vector, `L` labels, `S` photo mode, `H` hide the panels, `Del` delete, `Esc` clear. Right-drag pans. The isoview
  wordmark returns to the opening page.
- **Colour single atoms**: select atoms and choose **Colour** in the bar that appears; every periodic image of them
  takes the colour, it stays through supercells, deletions and reloads, and the picker's Default undoes it. Surface
  colours (Layers panel) use the same picker; no colour opens the system's colour dialog.
- **Photo Mode** (top left, or `S`): hides every control and draws the view supersampled, so edges are smooth. Turn and
  zoom to frame it, then **Save PNG** or **Copy** at the screen's size, 2× or 4× the window (up to 16 Mpx), on the
  theme's background, white, or transparent. Large images are drawn in tiles at twice the output size and
  averaged down; a transparent background comes from one render on black and one on white. In the Brillouin-zone
  view the crystal inset is part of the photo. With a slice, the photo bar also switches the main view (3D or 2D),
  the inset and the colour bar; the slice is sampled again for the output size, so contour lines stay smooth, and
  its text is black on white or transparent photos. `Esc` or Done returns.
- **Kept between visits**: the last scene (its files, layers, edits, view and camera) is stored in this browser, so a
  reload brings it back. Scenes up to 512 MB of files are kept whole; larger files come back as atoms only. **New
  scene** (top left) clears it, and can be undone.
- **Search options**: `Ctrl F` / `⌘ F`, or the field at the top of the panel, searches every option by name and by
  the words people use for it ("isovalue" finds Level, "supercell" finds Repeat, "toon" finds Look). Choosing one
  opens its panel, scrolls to it and outlines it; an option that needs another kind of file says which.
- **Hide**: each corner (title, panel, axes and cell, view buttons) has a small eye at the very corner that tucks it
  away; `H` hides or shows all four. The panel never scrolls sideways; its rows wrap to fit.

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
- Slices: plane normal along h b₁ + k b₂ + l b₃ (reciprocal vectors); tricubic Catmull-Rom interpolation of the grid;
  contour lines by marching squares.
- Brillouin zone: the Wigner-Seitz cell of the reciprocal lattice; the Bravais lattice and its points are found from
  the lattice alone (via the primitive and conventional cells), checked against pymatgen for every lattice type and
  variant.
- Trajectory analysis: g(r) over all periodic images (each frame normalised by its own volume); MSD averaged over
  all time origins (by FFT) on positions unwrapped with image flags, centre-of-mass drift removed, D = slope / 6
  fitted over 20-80 % of the lags; velocities by central differences; VDOS as the Hann-windowed cosine transform of
  the mass-weighted velocity autocorrelation; temperature T = 2 KE / ((3N - 3) k_B). Masses: IUPAC 2016 standard
  atomic weights.
- The space group next to the file name comes from [moyo](https://github.com/spglib/moyo) (by the spglib
  developers; its WebAssembly build is in `js/vendor/moyo`, MIT or Apache-2.0), at the same tolerance as the cell tools
  (1e-5 Å by default). For a trajectory it is that of the frame shown; molecules without a cell get none.

## Develop

A static site with no build step. Serve the folder with any static server (`python3 -m http.server`) and open it.
Before deploying a change, run `node tools/version.mjs N` with a new N: it puts `?v=N` on the stylesheet, the app,
every module import and the worker, so browsers never mix cached old modules with new ones (a module worker does not
use import maps, so the version has to be in the import specifiers themselves).
Tests of the readers (including every trajectory format, against ASE on the files in `tests/fixtures`), meshing,
symmetry, camera and measurement maths run in Node: `node tests/run.mjs`; slices, the Brillouin zone and trajectory
analysis have their own (`node tests/slice.mjs`, `tests/bz.mjs`, `tests/analysis.mjs`, `tests/plot.mjs`, `tests/tiles.mjs`, `tests/xrd.mjs`). The marching-cubes case
table is the classic Lorensen-Cline one, taken from [3Dmol.js](https://3dmol.org) (BSD-3-Clause, licence text in
`js/gfx/LICENSE-3Dmol.txt`).

## Desktop app

The same isoview as a desktop app for macOS, Windows and Linux, built with [Tauri](https://tauri.app) (Rust, using
the system's own web view, so installers are a few MB). The site's files are built into the app, so it works
offline. In the app, Export, PNG, GIF and movies open a native Save dialog.

Installers are built by GitHub Actions (`.github/workflows/desktop.yml`): `.dmg` for macOS (Apple silicon and
Intel), `.msi` and `.exe` for Windows, `.deb`, `.rpm` and `.AppImage` for Linux. The apps are not signed. On
macOS, allow isoview in System Settings > Privacy & Security after the first launch. On Windows, SmartScreen asks
first ("More info > Run anyway"); Windows 11 with Smart App Control on does not run unsigned apps at all, so use the
web version there.

To build one yourself you need Node and Rust, plus WebKitGTK on Linux (see Tauri's
[prerequisites](https://tauri.app/start/prerequisites/)). Then run `npm install`, then `npx tauri build` (or
`npx tauri dev` for a window that runs from your checkout). `tools/stage-desktop.mjs` copies the site into `dist/`
for the app; `src-tauri/` holds the Rust side. The app also contains Tauri and the Rust crates it uses, each under
its own licence (mostly MIT or Apache-2.0, a few MPL-2.0, Zlib, BSD or Unicode-3.0; `src-tauri/Cargo.lock` lists
the crates).

## Licence

Copyright 2026 Utkarsh Singh. isoview is licensed under the [Apache License 2.0](LICENSE): you may use, change and
share it, also commercially. Copies and works built on it must keep the licence and the [`NOTICE`](NOTICE) file and
say which files were changed. The licence gives no right to use the name isoview for them. Third-party parts (moyo,
the 3Dmol.js tables, the sample files from real calculations) keep their own licences, listed in `NOTICE`.
