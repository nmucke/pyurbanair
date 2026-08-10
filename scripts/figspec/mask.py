"""Building (solid-cell) masks on the common evaluation grid (spec §3 / §10.5).

Two independent sources, in order of preference:

1. **STL voxelization** -- the Xie & Castro (2008) geometry is a set of
   axis-aligned cubes. We read the binary STL, and a grid cell is "solid" when a
   vertical ray from its centre crosses an odd number of mesh triangles
   (point-in-mesh). This works on any (z, y, x) grid, so the same mask serves
   every model after interpolation onto the truth grid. The geometry itself
   lives in :mod:`evaluation.style`; this module only binds it to the repo's
   data locations via :mod:`figspec.dataio`.
2. **LBM ``blanking``** -- the pylbm runs carry an explicit 0/1 solid field; we
   can interpolate it onto the truth grid as a cross-check / fallback.

If the STL is unavailable the mask is ``None`` (metrics then run over all cells;
this is recorded in ``figures/NOTES.md``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from evaluation.style import stl_solid_mask

from . import dataio


@lru_cache(maxsize=1)
def truth_solid_mask() -> np.ndarray | None:
    """Solid mask on the full truth (z, y, x) grid, or None if no STL."""
    g = dataio.truth_grid()
    return stl_solid_mask(len(g["z"]), len(g["y"]), len(g["x"]), dataio.STL_PATH, g)


def mask_for_slice(z_index: int) -> np.ndarray | None:
    """2-D (y, x) solid mask at a single z-level of the truth grid."""
    m = truth_solid_mask()
    return None if m is None else m[z_index]


def nearest_z_index(z_meters: float) -> int:
    """Index of the truth z-level nearest a physical height (e.g. pedestrian z)."""
    zc = dataio.truth_grid()["z"]
    return int(np.argmin(np.abs(np.asarray(zc, dtype=float) - z_meters)))
