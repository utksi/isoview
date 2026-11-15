# Samples

The files offered under "Try a sample" on isoview's opening page.

## Made for isoview

Written by [`tools/make_samples.py`](../tools/make_samples.py) (ASE 3.26, fixed random seeds). The model fields are
not DFT results, and the trajectories use effective-medium theory (EMT), a simple model of metals. EMT energies
are relative to the bulk fcc metal, so a cluster's energy is positive.

| File | What it is |
|---|---|
| `rutile-TiO2.cif` | rutile TiO₂, space group P4₂/mnm (symmetry operations applied on reading) |
| `SrTiO3.vasp` | cubic perovskite SrTiO₃, a = 3.905 Å (POSCAR) |
| `ZnO-wurtzite.xsf` | wurtzite ZnO, a = 3.25 Å, c = 5.207 Å |
| `graphene.pw.in` | graphene, Quantum ESPRESSO input, layer in the middle of 14.8 Å of vacuum (shown as a 2 × 2 supercell) |
| `NaCl.geometry.in` | rock-salt NaCl, conventional cell (FHI-aims) |
| `C60.xyz` | C₆₀ (ASE's molecule set) |
| `ethanol.pdb` | ethanol (ASE's molecule set) |
| `Si-model.CHGCAR` | model valence density of diamond Si: Gaussians on the atoms (2 e each) and on the 16 bond centres (1 e each), 32 e in the cell |
| `benzene-pi.cube` | Hückel π orbital of benzene (one of the two HOMOs) from carbon 2p_z Slater orbitals |
| `Cu55-900K.traj` | Cu₅₅ icosahedron at 900 K, Langevin dynamics, 101 frames with energies (ASE trajectory) |
| `Pt38-relax.extxyz` | Pt₃₈ cluster melted at 1400 K, then relaxed (FIRE), 104 frames with energies |
| `Al-800K.XDATCAR` | fcc Al, 108 atoms at 800 K, 121 frames |
| `Ag-1600K.lammpstrj` | fcc Ag, 108 atoms at 1600 K, 121 frames (LAMMPS text dump format) |
| `Ni100-adatom-1100K.axsf` | Ni adatom on Ni(100) at 1100 K, bottom layers fixed, 121 frames (animated XSF) |

## From real calculations

Taken unchanged (only renamed) from public test data:

| File | Source | Licence |
|---|---|---|
| `LGPS-800K.vasprun.xml.gz` | [pymatgen-test-files](https://github.com/materialsproject/pymatgen-test-files) `io/vasp/outputs/vasprun.bad_random_seed.xml.gz`: VASP molecular dynamics of Li₁₀GeP₂S₁₂ (50 atoms) at 800 K, Nosé-Hoover, 2 fs steps, 264 steps with energies | MIT |
| `LiPON-relax.OUTCAR.gz` | pymatgen-test-files `io/vasp/outputs/OUTCAR.CL.gz`: VASP conjugate-gradient relaxation of a 14-atom Li phosphorus oxynitride, 79 ionic steps | MIT |
| `LiSi-md.XDATCAR` | pymatgen-test-files `io/vasp/outputs/XDATCAR_traj`: VASP molecular dynamics of Li₃₈Si₃₈, 100 frames (written out by pymatgen, with the element after each position) | MIT |
| `N2.WAVECAR` | pymatgen-test-files `io/vasp/outputs/WAVECAR.N2`: N₂ molecule, 9 bands (no structure file comes with it, so only the cell is drawn) | MIT |
| `C4.ELFCAR.gz` | pymatgen-test-files `io/vasp/outputs/ELFCAR.gz`: electron localisation function of a four-atom carbon cell, two spin channels | MIT |
| `Li.LOCPOT.gz` | pymatgen-test-files `io/vasp/outputs/LOCPOT.gz`: local potential of Li (eV) | MIT |
| `Si8-md.pw.out` | [Quantum ESPRESSO](https://github.com/QEF/q-e) `PW/examples/example03/reference/si.md8.out`: pw.x 6.0 molecular dynamics of eight Si atoms, 100 steps | GPL-2.0 |
| `Si-relax.geom` | [aiida-castep](https://github.com/zhubonan/aiida-castep) `tests/data/Si-geom-stress/aiida.geom`: CASTEP geometry and cell optimisation of Si, 31 steps | MIT |

The licence texts are in [`LICENSES/`](LICENSES/).
