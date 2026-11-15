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
| `graphene.pw.in` | graphene, Quantum ESPRESSO input |
| `NaCl.geometry.in` | rock-salt NaCl, conventional cell (FHI-aims) |
| `C60.xyz` | C₆₀ (ASE's molecule set) |
| `ethanol.pdb` | ethanol (ASE's molecule set) |
| `Si-model.CHGCAR` | model valence density of diamond Si: Gaussians on the atoms (2 e each) and on the 16 bond centres (1 e each), 32 e in the cell |
| `benzene-pi.cube` | Hückel π orbital of benzene (one of the two HOMOs) from carbon 2p_z Slater orbitals |
| `Cu55-900K.traj` | Cu₅₅ icosahedron at 900 K, Langevin dynamics, 101 frames with energies (ASE trajectory) |
| `Pt13-relax.extxyz` | Pt₁₃ cluster relaxing from a distorted icosahedron (BFGS), 34 frames with energies |
| `Al-800K.XDATCAR` | fcc Al, 108 atoms at 800 K, 61 frames |
| `Ag-1600K.lammpstrj` | fcc Ag, 108 atoms at 1600 K, 51 frames (LAMMPS text dump format) |
| `Ni100-adatom-1100K.axsf` | Ni adatom on Ni(100) at 1100 K, bottom layers fixed, 51 frames (animated XSF) |

## From real calculations

Taken unchanged (only renamed) from public test data:

| File | Source | Licence |
|---|---|---|
| `Li-relax.OUTCAR` | [pymatgen-test-files](https://github.com/materialsproject/pymatgen-test-files) `io/vasp/fixtures/relaxation/OUTCAR`: VASP relaxation of Li, 3 ionic steps | MIT |
| `Si64-md.vasprun.xml.gz` | pymatgen-test-files `io/vasp/outputs/vasprun.md.xml.gz`: VASP molecular dynamics of 64 Si atoms, 10 steps | MIT |
| `Fe2O3-molten.XDATCAR` | pymatgen-test-files `io/vasp/outputs/XDATCAR.MD`: VASP molecular dynamics of molten Fe₂O₃, 80 atoms | MIT |
| `N2.WAVECAR` | pymatgen-test-files `io/vasp/outputs/WAVECAR.N2`: N₂ molecule, 9 bands (no structure file comes with it, so only the cell is drawn) | MIT |
| `C4.ELFCAR.gz` | pymatgen-test-files `io/vasp/outputs/ELFCAR.gz`: electron localisation function of a four-atom carbon cell, two spin channels | MIT |
| `Li.LOCPOT.gz` | pymatgen-test-files `io/vasp/outputs/LOCPOT.gz`: local potential of Li (eV) | MIT |
| `Si.pwscf.out` | pymatgen-test-files `io/pwscf/Si.pwscf.out`: Quantum ESPRESSO 5.1.2 SCF of Si | MIT |
| `Ar-LJ.md` | [castep_outputs](https://github.com/oerc0122/castep_outputs) `docs/source/example_data/pair-pot-lj.md`: CASTEP molecular dynamics of argon with a Lennard-Jones pair potential | BSD-3-Clause |

The licence texts are in [`LICENSES/`](LICENSES/).
