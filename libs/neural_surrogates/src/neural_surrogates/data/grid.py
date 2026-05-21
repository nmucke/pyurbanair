"""Collocated grid metadata + STL→occupancy/SDF baking (D3, D5).

Train and predict on the **collocated** ``x, y, z`` grid
(``docs/neural_surrogate_plan.md`` D3). pylbm/pypalm are already collocated;
uDALES staggered grids are interpolated to this grid before tensors are built.

Geometry is a **static, baked-in input** (D5): one model per (solver,
geometry, grid). The STL is voxelized once, offline, into a solid/fluid mask on
the collocated grid — bit-identical to where the source solver placed solid
cells (reusing ``pylbm.stl_to_lbm.get_building_grid_indices``) — and the SDF
derived from it is the preferred static channel.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

# Tensor axis order is channels-first space-last: [C, Z, Y, X].
SPATIAL_DIMS: tuple[str, str, str] = ("z", "y", "x")


@dataclass(frozen=True)
class GridMeta:
    """Collocated grid description shared by the corpus, checkpoint, and IO.

    Coordinates follow the source-solver convention (cell centers spanning the
    physical ``bounds``), matching how pylbm assigns ``x_grid/y_grid/z_grid``.
    """

    nx: int
    ny: int
    nz: int
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    def _centers(self, axis: int) -> np.ndarray:
        n = (self.nx, self.ny, self.nz)[axis]
        lo, hi = self.bounds[axis]
        edges = np.linspace(lo, hi, n + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def x(self) -> np.ndarray:
        return self._centers(0)

    @property
    def y(self) -> np.ndarray:
        return self._centers(1)

    @property
    def z(self) -> np.ndarray:
        return self._centers(2)

    def coords(self) -> dict[str, np.ndarray]:
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_dict(self) -> dict:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "bounds": [list(b) for b in self.bounds],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridMeta":
        bounds = tuple(tuple(float(v) for v in b) for b in data["bounds"])
        return cls(nx=int(data["nx"]), ny=int(data["ny"]), nz=int(data["nz"]), bounds=bounds)  # type: ignore[arg-type]

    def matches(self, other: "GridMeta", *, atol: float = 1e-6) -> bool:
        """Whether two grids agree in resolution and bounds (validation, §8.1)."""
        if (self.nx, self.ny, self.nz) != (other.nx, other.ny, other.nz):
            return False
        return bool(
            np.allclose(np.asarray(self.bounds), np.asarray(other.bounds), atol=atol)
        )


def build_occupancy_mask(
    stl_path: str | pathlib.Path,
    grid: GridMeta,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Voxelize an STL into a ``[Z, Y, X]`` solid (1) / fluid (0) mask.

    Reuses ``pylbm.stl_to_lbm.get_building_grid_indices`` so the solid cells
    coincide with where the LBM solver places blanked cells (D5). The helper
    returns 1-based Fortran interior index boxes ``{is,ie,js,je,ks,ke}`` over
    ``1..nx`` etc.; we fill those boxes (inclusive) in ``[X, Y, Z]`` order and
    transpose to the channels-first ``[Z, Y, X]`` convention.
    """
    from pylbm.stl_to_lbm import get_building_grid_indices

    (x0, x1), (y0, y1), (z0, z1) = grid.bounds
    domain_bounds = {
        "xmin": x0,
        "xmax": x1,
        "ymin": y0,
        "ymax": y1,
        "zmin": z0,
        "zmax": z1,
    }
    boxes = get_building_grid_indices(
        stl_path, grid.nx, grid.ny, grid.nz, domain_bounds=domain_bounds, verbose=verbose
    )

    mask_xyz = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float32)
    for b in boxes:
        # 1-based inclusive -> 0-based slices.
        mask_xyz[
            b["is"] - 1 : b["ie"],
            b["js"] - 1 : b["je"],
            b["ks"] - 1 : b["ke"],
        ] = 1.0
    return np.transpose(mask_xyz, (2, 1, 0))  # [Z, Y, X]


def mask_to_sdf(mask: np.ndarray, grid: GridMeta | None = None) -> np.ndarray:
    """Signed distance field from a solid/fluid mask (negative inside solid).

    Smooth wall-proximity gradients pool/convolve more gracefully than a 0/1
    edge (D5). Distances are in grid cells (isotropic); pass ``grid`` only if a
    physical-unit SDF is later needed. Uses the exact Euclidean distance
    transform from SciPy.
    """
    from scipy.ndimage import distance_transform_edt

    solid = mask > 0.5
    fluid = ~solid
    # Distance from each fluid cell to the nearest solid cell, and vice versa.
    dist_to_solid = distance_transform_edt(fluid)
    dist_to_fluid = distance_transform_edt(solid)
    sdf = dist_to_solid - dist_to_fluid
    return sdf.astype(np.float32)


def build_static_channels(
    mask: np.ndarray,
    *,
    include_sdf: bool = True,
    include_mask: bool = True,
    grid: GridMeta | None = None,
) -> np.ndarray:
    """Stack the static geometry channels ``[S, Z, Y, X]`` fed every step (D5).

    Default is ``[sdf, mask]``. The SDF is normalized to unit scale (divided by
    its max absolute value) so it sits in a network-friendly range.
    """
    channels: list[np.ndarray] = []
    if include_sdf:
        sdf = mask_to_sdf(mask, grid)
        scale = float(np.max(np.abs(sdf))) or 1.0
        channels.append(sdf / scale)
    if include_mask:
        channels.append(mask.astype(np.float32))
    if not channels:
        raise ValueError("At least one of include_sdf/include_mask must be True.")
    return np.stack(channels, axis=0).astype(np.float32)
