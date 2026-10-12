"""
Map area -> case folder and forward overrides (``pycity4cfd.case``).

OpenStreetMap is replaced by two hand-built buildings. The unit test also
replaces City4CFD (one box per building, in City4CFD's point-of-interest
frame); the ``integration`` test runs the real binary (skipped when it is not
built). Both check the STL lands in the domain frame, the grid and the sensors.
"""

import json
import pathlib

import geopandas as gpd
import numpy as np
import pytest
import trimesh
from pycity4cfd import case, city4cfd
from shapely.geometry import Point, box

# About 300 m x 300 m in central Delft (EPSG:4326 degrees).
BBOX = (4.3555, 52.0095, 4.3600, 52.0122)
RESOLUTION = 4.0
MARGIN = 20.0


def _buildings() -> gpd.GeoDataFrame:
    """Two footprints inside BBOX, in its UTM zone; the second is the tallest."""
    centre = gpd.GeoSeries([box(*BBOX).centroid], crs="EPSG:4326")
    utm = centre.estimate_utm_crs()
    x, y = centre.to_crs(utm).iloc[0].coords[0]
    return gpd.GeoDataFrame(
        {
            "id": ["way/1", "way/2"],
            "height": [12.0, 30.0],
            "height_source": ["levels", "height"],
        },
        geometry=[box(x - 60, y - 40, x - 30, y), box(x + 10, y + 10, x + 30, y + 50)],
        crs=utm,
    )


def _fake_city4cfd(config_path: pathlib.Path) -> pathlib.Path:
    """One box per footprint, translated by -point_of_interest like City4CFD."""
    config = json.loads(pathlib.Path(config_path).read_text())
    x0, y0 = config["point_of_interest"]
    boxes = []
    for b in gpd.read_file(config["polygons"][0]["path"]).itertuples():
        minx, miny, maxx, maxy = b.geometry.bounds
        boxes.append(
            trimesh.creation.box(
                bounds=[[minx - x0, miny - y0, 0.0], [maxx - x0, maxy - y0, b.height]]
            )
        )
    out = pathlib.Path(config_path).parent / "city4cfd_Buildings.stl"
    trimesh.util.concatenate(boxes).export(out)
    return out


def _check_case(result: dict, buildings: gpd.GeoDataFrame) -> None:
    case_dir = pathlib.Path(result["case_dir"])
    overrides = dict(o.split("=", 1) for o in result["overrides"] if "=" in o)
    nx, ny, nz = (result["domain"][k] for k in ("nx", "ny", "nz"))
    lx, ly, lz = result["domain"]["lengths_m"]

    assert nx % 2 == ny % 2 == 0
    assert (lx, ly, lz) == (nx * RESOLUTION, ny * RESOLUTION, nz * RESOLUTION)
    assert lz >= 3 * 30.0
    assert overrides["geometry.stl_path"] == str(case_dir / "buildings.stl")
    assert json.loads((case_dir / "case.json").read_text())["bbox"] == list(BBOX)

    # The STL keeps the footprints' layout, shifted into the domain frame.
    mesh = trimesh.load_mesh(case_dir / "buildings.stl")
    (mx0, my0, mz0), (mx1, my1, mz1) = mesh.bounds
    assert mz0 == pytest.approx(0.0) and mz1 == pytest.approx(30.0)
    assert 0 < mx0 and mx1 < lx and 0 < my0 and my1 < ly
    fx0, fy0, fx1, fy1 = buildings.total_bounds
    assert mx1 - mx0 == pytest.approx(fx1 - fx0, abs=0.01)
    assert my1 - my0 == pytest.approx(fy1 - fy0, abs=0.01)

    # Sensors sit in the domain, outside every building.
    xs, ys = json.loads(overrides["obs.x_points"]), json.loads(
        overrides["obs.y_points"]
    )
    assert len(xs) == len(ys) == case.N_SENSORS
    shifted = buildings.geometry.translate(mx0 - fx0, my0 - fy0)
    for x, y in zip(xs, ys):
        assert 0 < x < lx and 0 < y < ly
        assert not shifted.contains(Point(x, y)).any()


def test_make_case(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buildings = _buildings()
    monkeypatch.setattr(case, "fetch_buildings", lambda *a, **k: buildings)
    monkeypatch.setattr(case, "run_city4cfd", _fake_city4cfd)
    result = case.make_case(
        BBOX, tmp_path / "delft", resolution=RESOLUTION, margin=MARGIN
    )
    assert result["n_buildings"] == 2
    assert result["height_sources"] == {"levels": 1, "height": 1}
    _check_case(result, buildings)


@pytest.mark.integration  # type: ignore[misc]
def test_make_case_runs_city4cfd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        city4cfd.city4cfd_executable()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    buildings = _buildings()
    monkeypatch.setattr(case, "fetch_buildings", lambda *a, **k: buildings)
    result = case.make_case(
        BBOX, tmp_path / "delft", resolution=RESOLUTION, margin=MARGIN
    )
    _check_case(result, buildings)
    assert np.isfinite(trimesh.load_mesh(tmp_path / "delft" / "buildings.stl").volume)
