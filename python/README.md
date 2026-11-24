# isoview in Jupyter

Show crystal structures, volumetric data (CHGCAR, LOCPOT, ELFCAR, cube, ...), WAVECAR orbitals and trajectories
with [isoview](https://utkarsh.science/isoview/) in a notebook cell.

```
pip install "isoview-notebook @ git+https://github.com/utksi/isoview#subdirectory=python"
```

```python
import isoview
isoview.show("CHGCAR")              # any file isoview opens
isoview.show("WAVECAR", "CONTCAR")  # several files in one scene
isoview.show(atoms)                 # an ASE Atoms (a list of them is a trajectory) or a pymatgen Structure
isoview.show("POSCAR", height=600, theme="dark", style="stick", look="soft", panel=True)
```

Works in JupyterLab, Jupyter Notebook, VS Code and Google Colab. The viewer loads from utkarsh.science/isoview in
the notebook's own browser tab, and the files are handed to it there (gzip-compressed inside the notebook output),
so nothing is uploaded; the browser needs internet access. Files larger than 200 MB after compression are refused
(they would be stored in the notebook); open those in the full site or the desktop app instead.

Apache License 2.0, like isoview.
