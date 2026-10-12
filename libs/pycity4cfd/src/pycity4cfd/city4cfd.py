"""Run City4CFD on building footprints with heights: GeoJSON -> buildings STL.

City4CFD (``libs/pycity4cfd/City4CFD``; on PATH in the ``city4cfd`` pixi env,
or built by ``pixi run build-city4cfd``) runs here without a measured point cloud: every building is an LoD1.2
extrusion of its footprint to its ``height`` attribute, closed at the bottom,
standing on flat ground at z = 0. Coordinates are in the buildings' projected
(metric) CRS, and City4CFD translates every output coordinate by
``-point_of_interest``. A building without a positive ``height`` is skipped and
listed in the City4CFD log.

Every footprint must lie inside the square of ``half_width`` around
``point_of_interest`` (``write_config`` checks it). City4CFD itself needs more
room than that, so its domain is ``DOMAIN_MARGIN`` larger:

- it takes the ground elevation of each footprint vertex from a terrain point
  cloud and aborts the whole run ("Can't calculate average of a zero-sized
  vector!") on a vertex outside that cloud's hull. Without a ground cloud it
  builds one from the footprint vertices, drops them as lying on buildings and
  fails. So ``write_config`` writes a flat ground grid, ``ground.xyz``, that
  reaches past the footprints;
- it drops ground points on or near its domain edge, and wants the region of
  buildings it reconstructs strictly inside the domain.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyurbanair.utils.solver_process import run_solver

# The pixi env this runs in (its prefix is .pixi/envs/<name> when not activated).
PIXI_ENV = os.environ.get("PIXI_ENVIRONMENT_NAME") or Path(sys.prefix).name
# libs/pycity4cfd/build/<env>/, where build_city4cfd.sh puts the binary: one
# build per env, since CMake records that env's absolute library paths.
BUILD_DIR = Path(__file__).resolve().parents[2] / "build" / PIXI_ENV

# Output file stem; City4CFD appends "_<surface>.stl" per surface.
OUTPUT_NAME = "city4cfd"

# City4CFD's domain and the ground grid, relative to the footprint square.
DOMAIN_MARGIN = 0.02
GROUND_MARGIN = 0.01
GROUND_POINTS = 11  # per side, edges included


def city4cfd_executable() -> Path:
    """Return the City4CFD binary: ``city4cfd`` on PATH, else this env's build.

    The ``city4cfd`` pixi env installs it on its PATH; a local build
    (``pixi run build-city4cfd``) lands in ``BUILD_DIR``.
    """
    found = shutil.which("city4cfd")
    if found:
        return Path(found)
    path = BUILD_DIR / "city4cfd"
    if not path.is_file():
        raise FileNotFoundError(
            f"City4CFD is not on PATH nor built for the {PIXI_ENV} env ({path}); "
            "install the city4cfd env (`pixi install -e city4cfd`) and run from "
            "it, e.g. `pixi run -e city4cfd python ...`."
        )
    return path


def _points(coordinates: Any) -> list[list[float]]:
    """Flatten GeoJSON (Multi)Polygon coordinates into ``[x, y]`` points."""
    if isinstance(coordinates[0], (int, float)):
        return [list(coordinates[:2])]
    return [p for c in coordinates for p in _points(c)]


def write_config(
    buildings_path: Path,
    out_dir: Path,
    *,
    point_of_interest: Sequence[float],
    half_width: float,
    top_height: float,
    edge_max_len: float = 10.0,
) -> Path:
    """Write the City4CFD config for ``buildings_path`` into ``out_dir``.

    Args:
        buildings_path: GeoJSON of building footprints in a projected metric
            CRS with a ``height`` property (m) and an ``id``.
        out_dir: Directory for the config, City4CFD's outputs and its log.
        point_of_interest: Centre ``(x, y)`` in the buildings' CRS; the origin
            of City4CFD's output coordinates.
        half_width: Half the side of the square around ``point_of_interest``
            that holds every footprint (m).
        top_height: Height of City4CFD's domain top (m).
        edge_max_len: Longest polygon edge before City4CFD splits it (m).

    Returns:
        Path of the written ``config.json`` (``ground.xyz`` next to it).

    Raises:
        ValueError: A footprint reaches outside the square.
    """
    buildings_path = Path(buildings_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, y0 = (float(v) for v in point_of_interest)

    outside = [
        str(f["properties"].get("id"))
        for f in json.loads(buildings_path.read_text())["features"]
        if (
            np.abs(np.array(_points(f["geometry"]["coordinates"])) - (x0, y0))
            > half_width
        ).any()
    ]
    if outside:
        raise ValueError(
            f"Footprints reach outside the {2 * half_width} m square around "
            f"{(x0, y0)}: {', '.join(outside)}"
        )

    def square(h: float) -> list[list[float]]:
        """Counter-clockwise square of half-side ``h``, absolute coordinates."""
        return [[x0 - h, y0 - h], [x0 + h, y0 - h], [x0 + h, y0 + h], [x0 - h, y0 + h]]

    ground_path = out_dir / "ground.xyz"
    offsets = np.linspace(-1, 1, GROUND_POINTS) * half_width * (1 + GROUND_MARGIN)
    gx, gy = np.meshgrid(x0 + offsets, y0 + offsets)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    np.savetxt(ground_path, ground, fmt="%.3f")

    config = {
        "point_clouds": {"ground": str(ground_path)},
        "polygons": [
            {
                "type": "Building",
                "path": str(buildings_path),
                "unique_id": "id",
                "height_attribute": "height",
                "height_attribute_advantage": True,
            }
        ],
        "reconstruction_regions": [
            {"influence_region": square(half_width), "lod": "1.2"}
        ],
        "point_of_interest": [x0, y0],
        "domain_bnd": square(half_width * (1 + DOMAIN_MARGIN)),
        "top_height": float(top_height),
        "flat_terrain": True,
        "building_percentile": 90,
        "edge_max_len": float(edge_max_len),
        "output_file_name": OUTPUT_NAME,
        "output_format": "stl",
        "output_separately": True,
        "output_log": True,
        "log_file": "city4cfd_log",
    }
    path = out_dir / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def run_city4cfd(config_path: Path) -> Path:
    """Run City4CFD on ``config_path``; return the buildings STL it wrote.

    Outputs land next to the config: ``city4cfd_Buildings.stl`` (closed
    buildings, no ground), ``city4cfd_Terrain.stl`` (the flat ground with the
    footprints cut out), City4CFD's ``city4cfd_log`` and its stdout in
    ``city4cfd.out``.
    """
    config_path = Path(config_path).resolve()
    out_dir = config_path.parent
    stdout_path = out_dir / "city4cfd.out"
    with open(stdout_path, "w") as stdout:
        run_solver(
            [
                str(city4cfd_executable()),
                str(config_path),
                "--output_dir",
                str(out_dir),
            ],
            env=os.environ,
            cwd=out_dir,
            stdout=stdout,
            stderr=stdout,
            log_path=stdout_path,
        )
    stl = out_dir / f"{OUTPUT_NAME}_Buildings.stl"
    if not stl.is_file() or stl.stat().st_size == 0:
        raise RuntimeError(f"City4CFD wrote no buildings STL; see {stdout_path}")
    return stl
