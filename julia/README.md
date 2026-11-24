# isoview in Julia notebooks

Show crystal structures, volumetric data (CHGCAR, LOCPOT, ELFCAR, cube, ...), WAVECAR orbitals and trajectories
with [isoview](https://utkarsh.science/isoview/) in Pluto, IJulia (Jupyter) or VS Code.

```julia
using Pkg
Pkg.add(url="https://github.com/utksi/isoview", subdir="julia")

using Isoview
isoview("CHGCAR")                  # any file isoview opens
isoview("WAVECAR", "CONTCAR")      # several files in one scene
isoview("POSCAR"; height=600, theme="dark", style="stick", look="soft", panel=true)
isoview("POSCAR" => read("POSCAR"))  # a name and the file's bytes (or text)
```

The viewer loads from utkarsh.science/isoview in the notebook's own browser tab, and the files are handed to it
there (inside the notebook output), so nothing is uploaded; the browser needs internet access. Files above 200 MB
are refused (they would be stored in the notebook); open those in the full site or the desktop app instead.

Apache License 2.0, like isoview.
