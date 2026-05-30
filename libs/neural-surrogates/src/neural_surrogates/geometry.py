"""Geometry-mask construction for the neural-surrogate forward model.

The surrogate's network takes a binary geometry mask (``1`` = fluid,
``0`` = obstacle) as an extra channel, exactly like the
:class:`~neural_surrogates.data.TransitionDataset` it was trained on.
At inference time the mask has to be produced from whatever geometry the
caller is simulating; this module voxelises a ``.stl`` building geometry
onto the simulation grid so the same convention is reproduced.

The mask is built directly against a *template* state variable so that it
inherits the backend's spatial dimension order and per-cell coordinates
(pylbm uses ``x/y/z``; uDALES uses staggered ``xt/xm`` etc.). This keeps
the mask aligned with the state tensors regardless of backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _axis_of(dim: str) -> str:
    """Map a coordinate/dim name to its physical axis via its first letter.

    Works for both pylbm (``x``, ``y``, ``z``) and uDALES staggered
    (``xt``, ``xm``, ``yt``, ...) conventions.
    """
    first = dim[0].lower()
    if first not in ("x", "y", "z"):
        raise ValueError(
            f"cannot infer physical axis from dimension name '{dim}'; "
            "expected it to start with x, y or z"
        )
    return first


def stl_to_fluid_mask(
    stl_path: str | Path,
    template_var: xr.DataArray,
) -> np.ndarray:
    """Voxelise ``stl_path`` onto the grid of ``template_var``.

    Args:
        stl_path: Path to the building geometry ``.stl`` file.
        template_var: A single state variable (no ``time`` dimension) whose
            dims and coordinates define the target grid. Its dimension names
            are mapped to physical axes by first letter.

    Returns:
        A float array shaped like ``template_var`` with ``1`` for fluid
        cells and ``0`` for cells whose centre falls inside the geometry.
    """
    import trimesh

    dims = list(template_var.dims)
    axes = [_axis_of(d) for d in dims]
    coords = [np.asarray(template_var.coords[d].values, dtype=float) for d in dims]

    mesh = trimesh.load(str(stl_path), force="mesh")

    # Cell-centre coordinates for every grid point, in template dim order.
    grids = np.meshgrid(*coords, indexing="ij")
    columns = {axis: grid.ravel() for axis, grid in zip(axes, grids)}
    points = np.stack([columns["x"], columns["y"], columns["z"]], axis=-1)

    try:
        inside = mesh.contains(points)
    except Exception as exc:  # pragma: no cover - depends on mesh watertightness
        raise RuntimeError(
            f"Could not classify grid points against {stl_path}: {exc}. "
            "The STL must be watertight for inside/outside voxelisation."
        ) from exc

    obstacle = inside.reshape([c.size for c in coords])
    return (~obstacle).astype(np.float32)


def nonzero_fluid_mask(
    template: xr.Dataset,
    state_vars: tuple[str, ...],
) -> np.ndarray:
    """Fallback mask: fluid cells are those with non-zero state.

    Mirrors :class:`~neural_surrogates.data.TransitionDataset`'s fallback
    when a backend ships no obstacle indicator. ``template`` must be a
    single snapshot (no ``time`` dimension).
    """
    stacked = np.stack(
        [np.asarray(template[v].values) for v in state_vars], axis=0
    )
    return (np.abs(stacked).sum(axis=0) != 0).astype(np.float32)
