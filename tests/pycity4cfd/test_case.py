"""
Map area -> case folder and forward overrides (``pycity4cfd.case``).

OpenStreetMap is replaced by two hand-built buildings. The unit test also
replaces City4CFD (one box per building, in City4CFD's point-of-interest
frame); the ``integration`` test runs the real binary (skipped when it is not
built). Both check the STL lands in the domain frame, the grid, the sensors and
the uDALES/PALM templates; the overrides compose on every case and backend.
"""

import json
import pathlib
import re

import geopandas as gpd
import numpy as np
import pytest
import trimesh
from pycity4cfd import case, city4cfd
from shapely.geometry import Point, box

from tests.conftest import compose

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
    # PALM's Temperton FFT only factors 2, 3 and 5.
    for n in (nx, ny):
        while n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
            n //= next(p for p in (2, 3, 5) if n % p == 0)
        assert n == 1
    assert (lx, ly, lz) == (nx * RESOLUTION, ny * RESOLUTION, nz * RESOLUTION)
    assert lz >= 3 * 30.0
    assert overrides["geometry.stl_path"] == str(case_dir / "buildings.stl")
    case_json = json.loads((case_dir / "case.json").read_text())
    assert case_json["bbox"] == list(BBOX)
    assert case_json["crs"] == result["crs"] == buildings.crs.to_string()
    assert case_json["origin"] == result["origin"]

    # The STL keeps the footprints' layout, shifted into the domain frame.
    mesh = trimesh.load_mesh(case_dir / "buildings.stl")
    (mx0, my0, mz0), (mx1, my1, mz1) = mesh.bounds
    assert mz0 == pytest.approx(0.0) and mz1 == pytest.approx(30.0)
    assert 0 < mx0 and mx1 < lx and 0 < my0 and my1 < ly
    fx0, fy0, fx1, fy1 = buildings.total_bounds
    # origin + domain coordinates = projected coordinates.
    ox, oy = result["origin"]
    assert (mx0 + ox, my0 + oy) == pytest.approx((fx0, fy0), abs=0.01)
    assert mx1 - mx0 == pytest.approx(fx1 - fx0, abs=0.01)
    assert my1 - my0 == pytest.approx(fy1 - fy0, abs=0.01)

    # uDALES reads the STL name and one facet per triangle from namoptions;
    # both templates carry the case grid.
    def value(text: str, key: str) -> str:
        match = re.search(rf"(?m)^\s*{key}\s*=\s*([^,\s]+)", text)
        assert match is not None, key
        return match.group(1)

    namoptions = (case_dir / "namoptions.300").read_text()
    assert value(namoptions, "stl_file") == "buildings.stl"
    assert int(value(namoptions, "nfcts")) == len(mesh.faces)
    assert int(value(namoptions, "nsolpts_u")) == 0
    assert [int(value(namoptions, k)) for k in ("itot", "jtot", "ktot")] == [
        nx,
        ny,
        nz,
    ]
    assert float(value(namoptions, "zsize")) == lz
    p3d = (case_dir / "_p3d").read_text()
    assert [int(value(p3d, k)) for k in ("nx", "ny", "nz")] == [nx - 1, ny - 1, nz]
    assert float(value(p3d, "dz")) == RESOLUTION

    # Sensors sit in the domain, outside every building.
    xs, ys = json.loads(overrides["obs.x_points"]), json.loads(
        overrides["obs.y_points"]
    )
    vxs = json.loads(overrides["++obs.validation_x_points"])
    vys = json.loads(overrides["++obs.validation_y_points"])
    assert len(xs) == len(ys) == case.N_SENSORS
    assert len(vxs) == len(vys) == case.N_VALIDATION
    assert not set(zip(xs, ys)) & set(zip(vxs, vys))
    shifted = buildings.geometry.translate(mx0 - fx0, my0 - fy0)
    for x, y in zip(xs + vxs, ys + vys):
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


@pytest.mark.parametrize("model", ["pylbm", "pypalm", "pyudales"])  # type: ignore[misc]
@pytest.mark.parametrize("case_name", ["xie_and_castro", "barcelona"])  # type: ignore[misc]
def test_overrides_compose(
    case_name: str, model: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overrides compose on every backend, with or without held-out sensors."""
    buildings = _buildings()
    monkeypatch.setattr(case, "fetch_buildings", lambda *a, **k: buildings)
    monkeypatch.setattr(case, "run_city4cfd", _fake_city4cfd)
    result = case.make_case(
        BBOX, tmp_path / "delft", resolution=RESOLUTION, margin=MARGIN
    )
    cfg = compose(
        "forward",
        f"case={case_name}",
        f"model={model}",
        *result["overrides"],
        root=tmp_path,
    )
    overrides = dict(o.split("=", 1) for o in result["overrides"])
    assert cfg.domain.nx == result["domain"]["nx"]
    assert cfg.geometry.case_dir == result["case_dir"]
    # barcelona's precomputed uDALES geometry must not leak into the new case.
    assert cfg.geometry.udales_precomputed_geom_dir is None
    if model == "pyudales":
        assert cfg.model.forward_model.precomputed_geom_dir is None
    for axis in "xyz":
        assert list(cfg.obs[f"{axis}_points"]) == json.loads(
            overrides[f"obs.{axis}_points"]
        )
        assert list(cfg.obs[f"validation_{axis}_points"]) == json.loads(
            overrides[f"++obs.validation_{axis}_points"]
        )


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
