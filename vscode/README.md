# isoview for VS Code

Right-click a file in the Explorer (or an editor tab) and choose **Open in isoview** to see it in 3D next to your
code: crystal structures (POSCAR, CIF, XSF, pw.x input, xyz, ...), volumetric data (CHGCAR, PARCHG, LOCPOT, ELFCAR,
cube), WAVECAR orbitals and trajectories (XDATCAR, OUTCAR, vasprun.xml, extended xyz, LAMMPS dumps, ...). Select
several files to show them together, e.g. a WAVECAR with its CONTCAR.

Works over Remote SSH, WSL and in containers: the file is read where it lives and drawn on your machine. The viewer is
[isoview](https://utkarsh.science/isoview/), loaded from the web (internet access is needed); your files are passed
to it inside VS Code and never uploaded.

Apache License 2.0, like isoview.
