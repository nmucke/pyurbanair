"""
City4CFD on hand-written building footprints (``pycity4cfd.city4cfd``).

The unit tests check the config ``write_config`` writes; the ``integration``
test runs the binary from ``pixi run build-city4cfd`` (skipped when it is not
built) on three rectangular buildings and checks the buildings STL.
"""

import json
import pathlib

import numpy as np
import pytest
import trimesh
from pycity4cfd import city4cfd
from pycity4cfd.city4cfd import run_city4cfd, write_config

# Footprint corner (x, y) in EPSG:32631 (UTM 31N, around Delft), size (m), height.
Building = tuple[tuple[float, float], tuple[float, float], float]
BUILDINGS: list[Building] = [
    ((85000.0, 446000.0), (20.0, 10.0), 12.0),
    ((85040.0, 446010.0), (15.0, 15.0), 25.0),
    ((85010.0, 446040.0), (30.0, 8.0), 6.0),
]
POINT_OF_INTEREST = (85025.0, 446025.0)
# The second building touches the square's east edge (x = 85055).
HALF_WIDTH = 30.0


def _write_buildings(
    path: pathlib.Path, buildings: list[Building] = BUILDINGS
) -> pathlib.Path:
    features = []
    for i, ((x, y), (dx, dy), height) in enumerate(buildings):
        ring = [[x, y], [x + dx, y], [x + dx, y + dy], [x, y + dy], [x, y]]
        features.append(
            {
                "type": "Feature",
                "properties": {"id": f"b{i}", "height": height},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:EPSG::32631"},
                },
                "features": features,
            }
        )
    )
    return path


def _config(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_config(
        _write_buildings(tmp_path / "buildings.geojson"),
        tmp_path / "city4cfd",
        point_of_interest=POINT_OF_INTEREST,
        half_width=HALF_WIDTH,
        top_height=150.0,
    )


def test_write_config(tmp_path: pathlib.Path) -> None:
    path = _config(tmp_path)
    assert path == (tmp_path / "city4cfd" / "config.json").resolve()
    config = json.loads(path.read_text())

    (building,) = config["polygons"]
    assert building["type"] == "Building"
    assert pathlib.Path(building["path"]).is_absolute()
    assert building["height_attribute"] == "height"
    assert building["height_attribute_advantage"] is True
    assert config["flat_terrain"] is True
    assert config["output_format"] == "stl"
    assert config["output_separately"] is True
    assert config["point_of_interest"] == list(POINT_OF_INTEREST)
    assert config["top_height"] == 150.0

    # Counter-clockwise squares around the point of interest in absolute
    # coordinates: the buildings' one, and City4CFD's domain 2 % larger.
    (region,) = config["reconstruction_regions"]
    assert region == {
        "influence_region": [
            [84995.0, 445995.0],
            [85055.0, 445995.0],
            [85055.0, 446055.0],
            [84995.0, 446055.0],
        ],
        "lod": "1.2",
    }
    assert np.allclose(
        config["domain_bnd"],
        [
            [84994.4, 445994.4],
            [85055.6, 445994.4],
            [85055.6, 446055.6],
            [84994.4, 446055.6],
        ],
    )

    # Flat ground just past the buildings' square, inside the domain.
    ground = np.loadtxt(config["point_clouds"]["ground"])
    assert (ground[:, 2] == 0.0).all()
    assert np.allclose(np.abs(ground[:, :2] - POINT_OF_INTEREST).max(), 30.3)


def test_write_config_rejects_footprints_outside(tmp_path: pathlib.Path) -> None:
    buildings = BUILDINGS + [((85050.0, 446050.0), (10.0, 10.0), 9.0)]
    path = _write_buildings(tmp_path / "buildings.geojson", buildings)
    with pytest.raises(ValueError, match="b3"):
        write_config(
            path,
            tmp_path / "city4cfd",
            point_of_interest=POINT_OF_INTEREST,
            half_width=HALF_WIDTH,
            top_height=150.0,
        )


def test_executable_missing_names_the_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(city4cfd, "BUILD_DIR", tmp_path)
    monkeypatch.setattr(city4cfd, "PIXI_ENV", "someenv")
    with pytest.raises(FileNotFoundError, match="pixi install -e city4cfd"):
        city4cfd.city4cfd_executable()


def test_executable_on_path_comes_first(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for directory in ("bin", "build"):
        binary = tmp_path / directory / "city4cfd"
        binary.parent.mkdir()
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
    monkeypatch.setattr(city4cfd, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert city4cfd.city4cfd_executable() == tmp_path / "bin" / "city4cfd"
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    assert city4cfd.city4cfd_executable() == tmp_path / "build" / "city4cfd"


def test_build_dir_is_per_pixi_env() -> None:
    assert city4cfd.BUILD_DIR.name == city4cfd.PIXI_ENV
    assert city4cfd.BUILD_DIR.parent.name == "build"


def _has_city4cfd() -> bool:
    try:
        city4cfd.city4cfd_executable()
    except FileNotFoundError:
        return False
    return True


requires_city4cfd = pytest.mark.skipif(
    not _has_city4cfd(), reason="City4CFD is not installed (pixi install -e city4cfd)"
)


@pytest.mark.integration  # type: ignore[misc]
@requires_city4cfd  # type: ignore[misc]
def test_run_city4cfd(tmp_path: pathlib.Path) -> None:
    stl = run_city4cfd(_config(tmp_path))
    assert stl.name == "city4cfd_Buildings.stl"
    assert stl.stat().st_size > 0

    mesh = trimesh.load_mesh(stl)
    # One closed body per building, flat on z = 0, as tall as its height.
    bodies = mesh.split(only_watertight=False)
    assert len(bodies) == len(BUILDINGS)
    assert all(body.is_watertight for body in bodies)
    zmin, zmax = mesh.bounds[:, 2]
    assert zmin == pytest.approx(0.0, abs=1e-6)
    assert zmax == pytest.approx(max(h for *_, h in BUILDINGS), abs=1e-3)
    # Coordinates are relative to the point of interest.
    assert np.allclose(mesh.bounds[:, :2], [[-25.0, -25.0], [30.0, 23.0]], atol=1e-3)
