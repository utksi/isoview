# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Utkarsh Singh
"""Show files in isoview (https://utkarsh.science/isoview/) inside a Jupyter notebook cell.

show(...) returns a View whose HTML is an iframe of the isoview site in embedded mode (?embed) plus a short script.
When the embedded page reports {isoview: "ready"} by postMessage, the script sends it the files as
{isoview: "open", files: [{name, base64}], mode: "replace"}. Every file is gzip-compressed first (isoview unpacks
gzip by its magic bytes 1f 8b), so a notebook stores about 1.37 x the compressed size (base64 is 4/3 the bytes,
plus the name). Nothing is uploaded: the iframe and the files meet in the user's browser.
"""
import base64
import gzip
import html
import io
import json
import os
import uuid

__all__ = ["show", "View", "SITE"]
__version__ = "0.1.0"

SITE = "https://utkarsh.science/isoview/"
LIMIT = 200 * 2**20  # bytes of compressed files per view; larger ones would bloat the notebook


def _from_object(x):
    """(name, bytes) for an in-memory structure: an ASE Atoms (extended XYZ), a list or tuple of ASE Atoms (a
    multi-frame extended XYZ, i.e. a trajectory), or a pymatgen Structure (CIF) or Molecule (XYZ). None otherwise."""
    frames = list(x) if isinstance(x, (list, tuple)) and x and all(hasattr(a, "get_chemical_symbols") for a in x) else None
    if frames is not None or hasattr(x, "get_chemical_symbols"):
        from ase.io import write  # only needed when the caller already has ASE objects
        buf = io.StringIO()
        write(buf, frames if frames is not None else x, format="extxyz")
        return ("structure.extxyz", buf.getvalue().encode())
    if hasattr(x, "lattice") and hasattr(x, "to"):
        return ("structure.cif", x.to(fmt="cif").encode())
    if hasattr(x, "to") and hasattr(x, "species"):
        return ("molecule.xyz", x.to(fmt="xyz").encode())
    return None


def _item(x):
    """(name, bytes) for one argument of show: a path (str or os.PathLike), a (name, bytes or str) pair, or a
    structure object (see _from_object)."""
    if isinstance(x, (str, os.PathLike)):
        p = os.fspath(x)
        with open(p, "rb") as f:
            return (os.path.basename(p), f.read())
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], (bytes, bytearray, str)):
        data = x[1].encode() if isinstance(x[1], str) else bytes(x[1])
        return (x[0], data)
    o = _from_object(x)
    if o is None:
        raise TypeError(f"isoview.show cannot use a {type(x).__name__}: give a file path, a (name, bytes) pair, "
                        "an ASE Atoms (or a list of them) or a pymatgen Structure")
    return o


class View:
    """An isoview viewer for a notebook cell. files: [(name, bytes)]; height in CSS pixels; query: the extra URL
    parameters of the embedded page (theme, style, look, panel); site: the isoview address."""

    def __init__(self, files, height=480, query=None, site=SITE):
        self.files, self.height, self.query, self.site = files, int(height), dict(query or {}), site

    def _payload(self):
        """[{name, base64}] with each file gzip-compressed (already compressed files are sent as they are)."""
        out, total = [], 0
        for name, data in self.files:
            if data[:2] != b"\x1f\x8b":
                # WHY: the name stays as it is (isoview unpacks by magic bytes, and shows the name as the title)
                data = gzip.compress(data, compresslevel=6, mtime=0)
            total += len(data)
            out.append({"name": name, "base64": base64.b64encode(data).decode("ascii")})
        if total > LIMIT:
            raise ValueError(f"{total / 2**20:.0f} MB after compression is too much to keep in a notebook (limit "
                             f"{LIMIT / 2**20:.0f} MB); open these files in {self.site} or the desktop app")
        return out

    def _repr_html_(self):
        """The iframe and the script that hands it the files once it is ready."""
        q = "&".join(["embed"] + [f"{k}={v}" for k, v in self.query.items()])
        src = f"{self.site}?{q}"
        origin = "/".join(self.site.split("/")[:3])
        fid = "isoview-" + uuid.uuid4().hex[:12]
        files = json.dumps(self._payload())
        return (
            f'<iframe id="{fid}" src="{html.escape(src)}" style="width:100%;height:{self.height}px;border:0;'
            f'border-radius:10px" allowfullscreen title="isoview"></iframe>'
            f"<script>(function(){{var f=document.getElementById({json.dumps(fid)}),files={files};"
            f"window.addEventListener('message',function(e){{if(f&&e.source===f.contentWindow&&e.data&&"
            f"e.data.isoview==='ready')f.contentWindow.postMessage({{isoview:'open',files:files,mode:'replace'}},"
            f"{json.dumps(origin)});}});}})();</script>"
        )


def show(*items, height=480, theme=None, style=None, look=None, panel=False, site=SITE):
    """Show the given files or structures in one isoview scene in the notebook cell.

    items: file paths (any format isoview opens: POSCAR, CIF, CHGCAR, LOCPOT, cube, WAVECAR with its POSCAR,
    trajectories, ...), (name, bytes) pairs, ASE Atoms (a list of them is shown as a trajectory) or pymatgen
    Structure / Molecule objects. height: pixels. theme: "light" or "dark" (default: follows the browser). style:
    "ball", "stick", "space" (see the Structure panel). look: "classic", "soft", "glossy", "toon", "outline".
    panel: show the side panel at start (it can always be opened from its corner).
    """
    if not items:
        raise TypeError("isoview.show needs at least one file or structure")
    query = {k: v for k, v in (("theme", theme), ("style", style), ("look", look)) if v}
    if panel:
        query["panel"] = 1
    return View([_item(x) for x in items], height=height, query=query, site=site)
