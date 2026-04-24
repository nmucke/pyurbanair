"""Convert a triangulated STL to PALM's 2D topography format.

PALM reads building topography as an ASCII grid of surface heights: one float
per (y, x) cell, ``ny`` rows of ``nx`` whitespace-separated values. By
convention the file is written with y running from top (highest y) to bottom
(lowest y) — see PALM's topography documentation for the canonical reference.
Heights should be expressed in meters and (if ``snap_to_dz=True``) rounded to
integer multiples of the vertical grid spacing ``dz``.

Placed at the package root (not under utils/) to mirror pylbm's
``stl_to_lbm.py``.
"""

import logging
import pathlib

import numpy as np
import trimesh

from .utils.dir_utils import PALMDirectoryPaths

logger = logging.getLogger(__name__)


def _load_mesh(stl_path: pathlib.Path) -> trimesh.Trimesh:
    loaded = trimesh.load(stl_path)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"STL scene at {stl_path} contains no meshes")
        return trimesh.util.concatenate(geometries)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"Could not load a valid mesh from {stl_path}")


def _vertical_ray_heights(
    mesh: trimesh.Trimesh,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_top: float,
) -> np.ndarray:
    """Cast one downward ray per (x, y) cell; return the highest intersection z.

    Cells with no intersection (no building above) get height 0.
    """
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    origins = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_top + 1.0)])
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    heights = np.zeros(origins.shape[0], dtype=float)
    if len(locations) > 0:
        for hit_point, ray_id in zip(locations, index_ray):
            z_hit = float(hit_point[2])
            if z_hit > heights[ray_id]:
                heights[ray_id] = z_hit

    return heights.reshape(yy.shape)


def stl_to_palm_topography(
    stl_path: str | pathlib.Path,
    dirs: PALMDirectoryPaths,
    nx: int,
    ny: int,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    dz: float,
    snap_to_dz: bool = True,
) -> np.ndarray:
    """Rasterize the STL buildings onto an ``(ny, nx)`` height grid and write ``_topo``.

    Args:
        stl_path: Path to the STL file containing building geometry.
        dirs: Directory layout for this experiment — the file is written to
            ``<input_dir>/<experiment_name>_topo``.
        nx, ny: Horizontal grid resolution.
        bounds: Physical domain ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
        dz: Vertical grid spacing in meters (used only when ``snap_to_dz``).
        snap_to_dz: If True, snap heights to integer multiples of ``dz``.

    Returns:
        The ``(ny, nx)`` height array that was written. y index 0 is the
        FIRST row in the file, which corresponds to the HIGHEST y. (PALM's
        topography file convention.)
    """
    stl_path = pathlib.Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    mesh = _load_mesh(stl_path)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    dx = (xmax - xmin) / nx
    dy_cell = (ymax - ymin) / ny
    x_centers = xmin + (np.arange(nx) + 0.5) * dx
    y_centers = ymin + (np.arange(ny) + 0.5) * dy_cell

    heights = _vertical_ray_heights(
        mesh=mesh,
        x_centers=x_centers,
        y_centers=y_centers,
        z_top=float(zmax),
    )
    heights = np.clip(heights - zmin, 0.0, None)

    if snap_to_dz and dz > 0:
        heights = np.round(heights / dz) * dz

    heights_to_write = np.flipud(heights)

    topo_path = dirs.input_dir / f"{dirs.experiment_name}_topo"
    np.savetxt(topo_path, heights_to_write, fmt="%.3f")
    logger.info(
        "Wrote PALM topography %s (%d x %d, max height %.3f m)",
        topo_path, ny, nx, float(heights.max()),
    )

    return heights
