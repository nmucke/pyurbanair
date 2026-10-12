"""Map area -> simulation case: OSM buildings, City4CFD STL, domain overrides.

``make_case`` writes one case folder (``buildings.geojson``, City4CFD's run in
``city4cfd/``, the domain-frame ``buildings.stl``, the uDALES ``namoptions.300``
and PALM ``_p3d`` templates, and ``case.json``) and returns the Hydra overrides
that point a forward run at it, e.g.
``python scripts/run_forward.py model=pylbm <overrides>`` (or ``model=pypalm``,
``model=pyudales``).

The domain is the bbox's projected envelope plus an open ``margin`` on every
side, so the inflow and outflow boundaries are clear of buildings. The STL
follows the repo's case contract (``scripts/tools/prepare_case_stl.py``):
metres, ``(xmin, ymin) -> (0, 0)``, flat ground at ``z = 0`` (no ground
mesh), z up.

A change that makes older case folders unusable (files, ``case.json`` keys,
overrides) needs a bump of ``CASE_FORMAT`` in ``mcp_server/jobs/city_case.py``,
whose case cache would otherwise reuse them.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import trimesh
from pycity4cfd.city4cfd import run_city4cfd, write_config
from pycity4cfd.osm import fetch_buildings, write_buildings
from shapely.geometry import Point, box

N_SENSORS = 6
N_VALIDATION = 3  # held-out sensors, after the assimilated ones
SENSOR_HEIGHT = 3.0  # m above ground
# The xie_and_castro uDALES/PALM templates: what the wrappers do not overwrite
# (courant 0.7, neutral, no Coriolis, CD2 advection) is generic and matched
# between the two solvers.
TEMPLATE_DIR = Path(__file__).resolve().parents[4] / "geometries" / "xie_and_castro"


def _even(n: float) -> int:
    """Round up to an even integer."""
    return 2 * math.ceil(n / 2)


def _fft_size(n: float) -> int:
    """Round up to an even integer with no prime factor above 5.

    PALM's poisfft needs even nx/ny and its Temperton FFT only factors 2, 3, 5
    (PAC0317).
    """
    m = _even(n)
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 2


def _sensors(
    footprints: gpd.GeoSeries, lx: float, ly: float, resolution: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Street-level points spread over the domain, outside buildings.

    Returns up to ``N_SENSORS`` assimilated and ``N_VALIDATION`` held-out
    points, distinct, from one farthest-point sampling (at least one each).
    """
    clearance = footprints.buffer(2 * resolution).union_all()
    candidates = [
        (lx * fx, ly * fy)
        for fy in np.linspace(0.2, 0.8, 9)
        for fx in np.linspace(0.2, 0.8, 9)
        if not clearance.contains(Point(lx * fx, ly * fy))
    ]
    if len(candidates) < 2:
        raise ValueError("Too few open street-level points for sensors in this area")
    # Farthest-point sampling from the domain centre for an even spread.
    pts = np.array(candidates)
    chosen = [int(np.argmin(np.linalg.norm(pts - (lx / 2, ly / 2), axis=1)))]
    while len(chosen) < min(N_SENSORS + N_VALIDATION, len(pts)):
        d = np.linalg.norm(pts[:, None] - pts[chosen][None], axis=2).min(axis=1)
        chosen.append(int(np.argmax(d)))
    points = [(round(float(x), 1), round(float(y), 1)) for x, y in pts[chosen]]
    n_assimilated = min(N_SENSORS, len(points) - 1)
    return points[:n_assimilated], points[n_assimilated:]


def _set(text: str, values: dict[str, Any]) -> str:
    """Replace ``key = value`` lines of a Fortran namelist."""
    for key, value in values.items():
        text, n = re.subn(
            rf"(?m)^(\s*{key}\s*=\s*).*$", rf"\g<1>{value}", text, count=1
        )
        if n != 1:
            raise KeyError(f"{key} not in the namelist template")
    return text


def _write_solver_templates(
    case_dir: Path, n_facets: int, n: tuple[int, int, int], lengths: tuple[float, ...]
) -> None:
    """Write ``namoptions.300`` (uDALES) and ``_p3d`` (PALM) into ``case_dir``.

    Both wrappers overwrite the grid, runtime, boundary conditions, inflow and
    closure at staging; the grid is written here only so the files are
    consistent as-is. uDALES reads ``stl_file`` from here; its IBM
    preprocessing recounts ``nfcts`` (one facet per STL triangle) and
    regenerates the other ``&WALLS`` counts, so those are zeroed.
    """
    (nx, ny, nz), (lx, ly, lz) = n, lengths
    namoptions = (TEMPLATE_DIR / "namoptions.300").read_text()
    walls = {
        f"{count}_{grid}": 0
        for count in ("nsolpts", "nbndpts", "nfctsecs")
        for grid in "uvwc"
    }
    namoptions = _set(
        namoptions,
        {
            "itot": nx,
            "jtot": ny,
            "ktot": nz,
            "xlen": lx,
            "ylen": ly,
            "zsize": lz,
            "stl_file": "buildings.stl",
            "nfcts": n_facets,
            **walls,
        },
    )
    (case_dir / "namoptions.300").write_text(namoptions)
    # PALM counts nx/ny from 0; nz is the number of levels.
    p3d = (TEMPLATE_DIR / "_p3d").read_text()
    p3d = _set(
        p3d,
        {
            "nx": f"{nx - 1},",
            "ny": f"{ny - 1},",
            "nz": f"{nz},",
            "dx": f"{lx / nx},",
            "dy": f"{ly / ny},",
            "dz": f"{lz / nz},",
        },
    )
    (case_dir / "_p3d").write_text(p3d)


def make_case(
    bbox: tuple[float, float, float, float],
    case_dir: str | Path,
    *,
    resolution: float = 4.0,
    margin: float = 50.0,
    floor_height: float = 3.0,
    default_height: float = 10.0,
) -> dict[str, Any]:
    """Build the case folder for ``bbox`` and return its forward-run overrides.

    Args:
        bbox: ``(west, south, east, north)`` in EPSG:4326 degrees.
        case_dir: Output folder (created; existing files are overwritten).
        resolution: Grid spacing in x, y and z (m).
        margin: Open ground around the selected area on every side (m).
        floor_height: Height per ``building:levels`` (m).
        default_height: Height of buildings without OSM height tags (m).

    Returns:
        ``{"case_dir", "overrides", "n_buildings", "height_sources",
        "max_height", "domain", "crs", "origin"}``; also written to
        ``case_dir/case.json``. ``origin`` is the domain's ``(0, 0)`` corner in
        ``crs`` (the local UTM zone).
    """
    case_dir = Path(case_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    buildings = fetch_buildings(
        bbox, floor_height=floor_height, default_height=default_height
    )
    buildings_path = write_buildings(buildings, case_dir / "buildings.geojson")

    # The selected area's envelope in the buildings' UTM CRS.
    area = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(buildings.crs)
    xmin, ymin, xmax, ymax = area.total_bounds
    poi = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    fx0, fy0, fx1, fy1 = buildings.total_bounds
    half_width = max(
        np.abs([xmin - poi[0], xmax - poi[0], ymin - poi[1], ymax - poi[1]]).max(),
        np.abs([fx0 - poi[0], fx1 - poi[0], fy0 - poi[1], fy1 - poi[1]]).max(),
    )

    max_height = float(buildings["height"].max())
    # Domain top: three times the tallest building, at least 15 cells.
    nz = _even(max(3 * max_height, 15 * resolution) / resolution)
    lz = nz * resolution
    nx = _fft_size((xmax - xmin + 2 * margin) / resolution)
    ny = _fft_size((ymax - ymin + 2 * margin) / resolution)
    lx, ly = nx * resolution, ny * resolution

    config = write_config(
        buildings_path,
        case_dir / "city4cfd",
        point_of_interest=poi,
        half_width=float(half_width),
        top_height=lz,
    )
    mesh = trimesh.load_mesh(run_city4cfd(config))
    # City4CFD's origin is the point of interest; the domain's is (xmin, ymin)
    # minus the margin.
    shift = (poi[0] - xmin + margin, poi[1] - ymin + margin)
    mesh.apply_translation((*shift, 0.0))
    stl_path = case_dir / "buildings.stl"
    mesh.export(stl_path)
    # nfcts as uDALES counts it: the faces trimesh keeps on loading the STL.
    n_facets = len(trimesh.load_mesh(stl_path).faces)
    _write_solver_templates(case_dir, n_facets, (nx, ny, nz), (lx, ly, lz))

    footprints = buildings.geometry.translate(-(xmin - margin), -(ymin - margin))
    sensors, validation = _sensors(footprints, lx, ly, resolution)
    xs, ys = [s[0] for s in sensors], [s[1] for s in sensors]

    overrides = [
        f"case_name=city_{case_dir.name}",
        f"geometry.case_dir={case_dir}",
        f"geometry.stl_path={stl_path}",
        # uDALES generates its IBM geometry from buildings.stl (a case's
        # precomputed one, e.g. barcelona's, is for another STL and grid).
        "++geometry.udales_precomputed_geom_dir=null",
        f"domain.nx={nx}",
        f"domain.ny={ny}",
        f"domain.nz={nz}",
        f"domain.bounds=[[0.0,{lx}],[0.0,{ly}],[0.0,{lz}]]",
        "obs.mode=points",
        f"obs.x_points={xs}",
        f"obs.y_points={ys}",
        f"obs.z_points={[SENSOR_HEIGHT] * len(sensors)}",
        # Replace (or add, ++) the case's held-out sensors: the default case's
        # would sit inside buildings.
        f"++obs.validation_x_points={[v[0] for v in validation]}",
        f"++obs.validation_y_points={[v[1] for v in validation]}",
        f"++obs.validation_z_points={[SENSOR_HEIGHT] * len(validation)}",
    ]
    result = {
        "case_dir": str(case_dir),
        "overrides": overrides,
        "n_buildings": len(buildings),
        "height_sources": buildings["height_source"].value_counts().to_dict(),
        "max_height": max_height,
        "domain": {"nx": nx, "ny": ny, "nz": nz, "lengths_m": [lx, ly, lz]},
        # The domain's (0, 0) corner in the buildings' projected (UTM) CRS, to
        # map domain coordinates (e.g. sensors) back onto the map.
        "crs": buildings.crs.to_string(),
        "origin": [float(xmin - margin), float(ymin - margin)],
    }
    # case.json marks the folder finished: write it atomically, so a reader
    # never sees it half written and a killed build leaves none.
    temporary = case_dir / f".case.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps({"bbox": list(bbox), **result}, indent=2) + "\n")
    os.replace(temporary, case_dir / "case.json")
    return result
