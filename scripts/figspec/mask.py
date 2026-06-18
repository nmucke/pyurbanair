"""Building (solid-cell) masks on the common evaluation grid (spec §3 / §10.5).

Two independent sources, in order of preference:

1. **STL voxelization** -- the Xie & Castro (2008) geometry is a set of
   axis-aligned cubes. We read the binary STL, and a grid cell is "solid" when a
   vertical ray from its centre crosses an odd number of mesh triangles
   (point-in-mesh). This works on any (z, y, x) grid, so the same mask serves
   every model after interpolation onto the truth grid.
2. **LBM ``blanking``** -- the pylbm runs carry an explicit 0/1 solid field; we
   can interpolate it onto the truth grid as a cross-check / fallback.

If the STL is unavailable the mask is ``None`` (metrics then run over all cells;
this is recorded in ``figures/NOTES.md``).
"""
from __future__ import annotations

import pathlib
import struct
from functools import lru_cache

import numpy as np

from . import dataio


# ---------------------------------------------------------------------------
# Binary STL reader
# ---------------------------------------------------------------------------
def read_binary_stl(path: pathlib.Path) -> np.ndarray:
    """Return triangles as an ``(n_tri, 3, 3)`` float array (vertices x xyz)."""
    data = pathlib.Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    tris = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        # 12 floats: normal(3) + v0(3) + v1(3) + v2(3); skip 2-byte attr
        vals = struct.unpack_from("<12f", data, off)
        tris[i, 0] = vals[3:6]
        tris[i, 1] = vals[6:9]
        tris[i, 2] = vals[9:12]
        off += 50
    return tris


def _ray_z_crossings(tris: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each (x, y) column, the sorted z-heights where +z rays hit the mesh.

    Returns an object array (len = n_points) of 1-D z-crossing arrays. Uses the
    2-D point-in-triangle test in the xy-plane and barycentric interpolation of
    the triangle's z at the hit.
    """
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    # edge vectors in xy
    e1 = (v1 - v0)[:, :2]
    e2 = (v2 - v0)[:, :2]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    good = np.abs(det) > 1e-12
    inv_det = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)

    pts = np.column_stack([x, y])
    out = np.empty(len(pts), dtype=object)
    for i, (px, py) in enumerate(pts):
        rx = px - v0[:, 0]
        ry = py - v0[:, 1]
        u = (rx * e2[:, 1] - ry * e2[:, 0]) * inv_det
        v = (e1[:, 0] * ry - e1[:, 1] * rx) * inv_det
        inside = good & (u >= 0) & (v >= 0) & (u + v <= 1.0)
        if not inside.any():
            out[i] = np.empty(0)
            continue
        w = 1.0 - u - v
        z_hit = (w[inside] * v0[inside, 2] + u[inside] * v1[inside, 2]
                 + v[inside] * v2[inside, 2])
        out[i] = np.sort(z_hit)
    return out


@lru_cache(maxsize=4)
def stl_solid_mask(nz: int, ny: int, nx: int,
                   stl_path: str | None = None) -> np.ndarray | None:
    """Boolean solid mask of shape (nz, ny, nx) on the truth grid.

    A cell centre is solid if it lies below an odd-numbered z-crossing of the
    mesh above it (i.e. inside a closed solid). Cached by grid shape.
    """
    path = pathlib.Path(stl_path) if stl_path else dataio.STL_PATH
    if not path.exists():
        return None
    g = dataio.truth_grid()
    zc, yc, xc = g["z"], g["y"], g["x"]
    if (len(zc), len(yc), len(xc)) != (nz, ny, nx):
        # caller asked for a non-truth shape; only the truth grid is supported
        return None

    tris = read_binary_stl(path)
    # Build the (y, x) column grid.
    XX, YY = np.meshgrid(xc, yc)  # (ny, nx)
    crossings = _ray_z_crossings(tris, XX.ravel(), YY.ravel())

    mask = np.zeros((nz, ny, nx), dtype=bool)
    zc_arr = np.asarray(zc, dtype=float)
    for col, cz in enumerate(crossings):
        if cz.size < 2:
            continue
        iy, ix = divmod(col, nx)
        # number of crossings strictly above each z level; odd => inside solid
        for k, zlev in enumerate(zc_arr):
            n_above = int(np.count_nonzero(cz > zlev))
            if n_above % 2 == 1:
                mask[k, iy, ix] = True
    return mask


@lru_cache(maxsize=1)
def truth_solid_mask() -> np.ndarray | None:
    """Solid mask on the full truth (z, y, x) grid, or None if no STL."""
    g = dataio.truth_grid()
    return stl_solid_mask(len(g["z"]), len(g["y"]), len(g["x"]))


def mask_for_slice(z_index: int) -> np.ndarray | None:
    """2-D (y, x) solid mask at a single z-level of the truth grid."""
    m = truth_solid_mask()
    return None if m is None else m[z_index]


def nearest_z_index(z_meters: float) -> int:
    """Index of the truth z-level nearest a physical height (e.g. pedestrian z)."""
    zc = dataio.truth_grid()["z"]
    return int(np.argmin(np.abs(np.asarray(zc, dtype=float) - z_meters)))
