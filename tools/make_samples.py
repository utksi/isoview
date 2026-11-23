# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Utkarsh Singh
"""Write the sample files offered on isoview's opening page (examples/), from model structures and short EMT runs.

Structures: textbook cells and ASE's molecule set, written with ase.io (POSCAR, XSF, FHI-aims, xyz, PDB).

Fields (model data, not DFT):
  Si model valence density (CHGCAR): diamond Si, a = 5.431 A, eight atoms. rho(r) = sum_i q_a g(r - R_i; s_a)
    + sum_b q_b g(r - B_b; s_b) over all periodic images, with g(r; s) = (2 pi s^2)^(-3/2) exp(-|r|^2 / 2 s^2),
    q_a = 2 e on each atom (s_a = 0.75 A) and q_b = 1 e on each of the 16 bond centres B_b (s_b = 0.55 A), so that
    the integral over the cell is 32 valence electrons. CHGCAR stores rho * V_cell (e), x index fastest.
  Benzene pi orbital (cube): Hueckel molecular orbital psi = sum_j c_j phi_j over the six carbon 2p_z Slater
    orbitals phi_j(r) = sqrt(zeta^5 / pi) z_j exp(-zeta r_j) (atomic units, zeta = 1.625 from Slater's rules),
    c_j = sqrt(1/3) cos(2 pi j / 6) for the carbons in ring order j = 0..5 (one of the two degenerate HOMOs,
    E = alpha + beta); overlap between neighbours is neglected (Hueckel). Grid step 0.25 A, values in bohr^-3/2.

Trajectories (effective-medium theory, ase.calculators.emt; energies in eV relative to the bulk fcc metals, so a
cluster's energy is positive):
  Cu55 icosahedron at 900 K: Langevin dynamics, 5 fs steps, friction 0.02, every 4th of 400 steps (ASE .traj).
  Al fcc 3x3x3 (108 atoms) at 800 K: Langevin, 5 fs, every 5th of 600 steps (XDATCAR, fixed cell), 121 frames.
  Pt38 truncated octahedron melted at 1400 K (1 ps of Langevin dynamics), then relaxed with FIRE (max step 0.1 A)
    to fmax 0.005 eV/A, every step (extended XYZ, energy=), 104 frames.
  Ag fcc 3x3x3 (108 atoms) at 1600 K: Langevin, 5 fs, every 8th of 960 steps (LAMMPS text dump, wrapped x y z).
  Ni(100) slab 3x3x4 with an adatom in a hollow site at 1100 K, bottom two layers fixed: Langevin, 5 fs, every
    8th of 960 steps (animated XSF).
Random seeds are fixed, so the files are reproducible. Run: OMP_NUM_THREADS=1 python tools/make_samples.py (needs ase,
numpy).
"""
import os
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule, fcc100, add_adsorbate
from ase.cluster import Icosahedron, Octahedron
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from ase import units

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
BOHR = 0.529177210903  # A


def gauss(d2, s):
    """Normalised 3D Gaussian (2 pi s^2)^(-3/2) exp(-d2 / 2 s^2); d2 in A^2, s in A, result in A^-3."""
    return (2 * np.pi * s * s) ** -1.5 * np.exp(-d2 / (2 * s * s))


def si_chgcar(path, n=32):
    """Model Si valence density on an n^3 grid of the conventional cell, written as a VASP CHGCAR (rho * V)."""
    a = 5.431
    si = bulk("Si", "diamond", a=a, cubic=True)
    L = si.cell[:]
    R = si.get_scaled_positions()
    # bond centres: midpoints between each atom and its four nearest neighbours (each bond once)
    B = []
    for i in range(len(R)):
        for j in range(len(R)):
            for sh in np.ndindex(3, 3, 3):
                d = R[j] + np.array(sh) - 1 - R[i]
                if abs(np.linalg.norm(d @ L) - a * np.sqrt(3) / 4) < 1e-6 and (i, j) < (j, i):
                    B.append(R[i] + d / 2)
    B = np.array(B) % 1.0
    assert len(B) == 16, len(B)
    g = (np.arange(n) / n)
    F = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)  # (i, j, k) -> fractional, i slowest here
    rho = np.zeros(len(F))
    for centres, q, s in ((R, 2.0, 0.75), (B, 1.0, 0.55)):
        for c in centres:
            d = F - c
            d -= np.round(d)
            for sh in np.ndindex(3, 3, 3):
                r = (d + np.array(sh) - 1) @ L
                rho += q * gauss((r * r).sum(1), s)
    V = abs(np.linalg.det(L))
    rho = rho.reshape(n, n, n)
    total = rho.sum() * V / n ** 3
    vals = (rho * V).transpose(2, 1, 0).ravel()  # CHGCAR order: x fastest, then y, then z
    with open(path, "w") as f:
        f.write("Si model valence density (sum of Gaussians on atoms and bond centres), isoview sample\n   1.0\n")
        for v in L:
            f.write("  %12.6f %12.6f %12.6f\n" % tuple(v))
        f.write("   Si\n   8\nDirect\n")
        for p in R:
            f.write("  %.6f %.6f %.6f\n" % tuple(p))
        f.write("\n  %d  %d  %d\n" % (n, n, n))
        for k in range(0, len(vals), 5):
            f.write(" " + " ".join("%.5E" % x for x in vals[k:k + 5]) + "\n")
    return total


def benzene_cube(path, step=0.25, pad=(2.6, 2.6, 3.0)):
    """Hueckel HOMO of benzene on a grid (cube file, bohr units for the grid and the atoms)."""
    m = molecule("C6H6")
    m.positions -= m.positions.mean(0)
    C = [i for i, s in enumerate(m.get_chemical_symbols()) if s == "C"]
    C.sort(key=lambda i: np.arctan2(m.positions[i, 1], m.positions[i, 0]))
    c = [np.sqrt(1 / 3) * np.cos(2 * np.pi * j / 6) for j in range(6)]
    zeta = 1.625
    lo = m.positions.min(0) - pad
    hi = m.positions.max(0) + pad
    nx, ny, nz = [int(np.ceil((hi[k] - lo[k]) / step)) + 1 for k in range(3)]
    X, Y, Z = np.meshgrid(*[lo[k] + step * np.arange(nn) for k, nn in enumerate((nx, ny, nz))], indexing="ij")
    psi = np.zeros_like(X)
    for cj, i in zip(c, C):
        dx, dy, dz = (X - m.positions[i, 0]) / BOHR, (Y - m.positions[i, 1]) / BOHR, (Z - m.positions[i, 2]) / BOHR
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        psi += cj * np.sqrt(zeta ** 5 / np.pi) * dz * np.exp(-zeta * r)
    with open(path, "w") as f:
        f.write("benzene pi orbital: Hueckel HOMO from carbon 2p_z Slater orbitals (zeta 1.625), isoview sample\n")
        f.write("values in bohr^-3/2; grid and atoms in bohr\n")
        o = lo / BOHR
        f.write("%5d %12.6f %12.6f %12.6f\n" % (len(m), *o))
        for k, nn in enumerate((nx, ny, nz)):
            v = [0.0, 0.0, 0.0]
            v[k] = step / BOHR
            f.write("%5d %12.6f %12.6f %12.6f\n" % (nn, *v))
        for z, p in zip(m.numbers, m.positions / BOHR):
            f.write("%5d %12.6f %12.6f %12.6f %12.6f\n" % (z, z, *p))
        for i in range(nx):
            for j in range(ny):
                row = psi[i, j]
                for k in range(0, nz, 6):
                    f.write(" ".join("%12.5E" % x for x in row[k:k + 6]) + "\n")


def run_md(atoms, T, steps, every, seed, friction=0.02):
    """Langevin dynamics (5 fs) with EMT; returns snapshots with their potential energies (eV)."""
    atoms.calc = EMT()
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, 5 * units.fs, temperature_K=T, friction=friction, rng=np.random.default_rng(seed + 1))
    frames = []

    def snap():
        a = atoms.copy()
        a.arrays.pop("momenta", None)
        a.calc = SinglePointCalculator(a, energy=atoms.get_potential_energy())
        frames.append(a)

    dyn.attach(snap, interval=every)  # also called at step 0
    dyn.run(steps)
    return frames


def lammps_dump(path, frames, every):
    """LAMMPS text dump (orthogonal box from 0, wrapped coordinates, id type element x y z)."""
    els = sorted(set(frames[0].get_chemical_symbols()))
    with open(path, "w") as f:
        for k, a in enumerate(frames):
            L = np.diag(a.cell[:])
            a = a.copy()
            a.wrap()
            f.write("ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\nITEM: BOX BOUNDS pp pp pp\n" % (k * every, len(a)))
            for x in L:
                f.write("0.0 %.6f\n" % x)
            f.write("ITEM: ATOMS id type element x y z\n")
            for i, (s, p) in enumerate(zip(a.get_chemical_symbols(), a.positions)):
                f.write("%d %d %s %.5f %.5f %.5f\n" % (i + 1, els.index(s) + 1, s, *p))


def main():
    os.makedirs(OUT, exist_ok=True)
    p = lambda name: os.path.join(OUT, name)
    # structures
    sto = Atoms("SrTiO3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], cell=[3.905] * 3, pbc=True)
    write(p("SrTiO3.vasp"), sto, format="vasp", direct=True)
    write(p("ZnO-wurtzite.xsf"), bulk("ZnO", "wurtzite", a=3.25, c=5.207), format="xsf")
    write(p("NaCl.geometry.in"), bulk("NaCl", "rocksalt", a=5.64, cubic=True), format="aims", scaled=True)
    write(p("C60.xyz"), molecule("C60"), format="xyz")
    write(p("ethanol.pdb"), molecule("CH3CH2OH"), format="proteindatabank")
    # fields
    total = si_chgcar(p("Si-model.CHGCAR"))
    benzene_cube(p("benzene-pi.cube"))
    # trajectories
    cu = Icosahedron("Cu", 3)
    cu.center(vacuum=6.0)
    fr = run_md(cu, 900, 400, 4, seed=11)
    with Trajectory(p("Cu55-900K.traj"), "w") as t:
        for a in fr:
            t.write(a)
    al = bulk("Al", "fcc", a=4.05, cubic=True) * (3, 3, 3)
    write(p("Al-800K.XDATCAR"), run_md(al, 800, 600, 5, seed=21), format="vasp-xdatcar")
    pt = Octahedron("Pt", 4, cutoff=1)
    pt.center(vacuum=6.0)
    pt.calc = EMT()
    MaxwellBoltzmannDistribution(pt, temperature_K=1400, rng=np.random.default_rng(61))
    Langevin(pt, 5 * units.fs, temperature_K=1400, friction=0.02, rng=np.random.default_rng(62)).run(200)
    pt.arrays.pop("momenta", None)
    relax = []

    def keep():
        a = pt.copy()
        a.calc = SinglePointCalculator(a, energy=pt.get_potential_energy())
        relax.append(a)

    opt = FIRE(pt, maxstep=0.1, logfile=None)
    opt.attach(keep, interval=1)  # also called before the first step
    opt.run(fmax=0.005)
    write(p("Pt38-relax.extxyz"), relax, format="extxyz")
    ag = bulk("Ag", "fcc", a=4.09, cubic=True) * (3, 3, 3)
    lammps_dump(p("Ag-1600K.lammpstrj"), run_md(ag, 1600, 960, 8, seed=41), 8)
    ni = fcc100("Ni", size=(3, 3, 4), vacuum=6.0)
    add_adsorbate(ni, "Ni", 1.7, "hollow")
    ni.set_constraint(FixAtoms(indices=[i for i, z in enumerate(ni.positions[:, 2]) if z < ni.positions[:, 2].min() + 2.0]))
    write(p("Ni100-adatom-1100K.axsf"), run_md(ni, 1100, 960, 8, seed=51), format="xsf")
    print("Si model density integrates to %.4f e; Pt38 relaxation %d frames" % (total, len(relax)))
    for f in sorted(os.listdir(OUT)):
        print("%8.1f kB  %s" % (os.path.getsize(p(f)) / 1024, f))


if __name__ == "__main__":
    main()
