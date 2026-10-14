"""Ground coverage and the outflow buffer of the uDALES STL, without CFD.

uDALES' floor is a wall only under STL ground facets (``utils.stl_utils``), so
a domain reaching past the STL's ground gets ground added there, and the
outflow buffer replaces the buildings beyond the bounds by flat ground, as
pypalm's buffer does.
"""

import pathlib
import shutil

import numpy as np
import pytest
import trimesh
import xarray
from pyudales.forward_model import ForwardModel
from pyudales.utils.namoptions_utils import NamoptionsFile
from pyudales.utils.stl_utils import _ground_faces, fit_stl_to_domain

CASE = pathlib.Path(__file__).resolve().parents[2] / "geometries" / "xie_and_castro"
STL = CASE / "xie_castro_2008_STL.stl"
# The STL's ground is [0, 160] x [0, 160] with 64 cube footprints of 100 m^2.
GROUND_AREA = 160.0 * 160.0 - 64 * 100.0


def _shifted_stl(dx: float) -> trimesh.Trimesh:
    mesh = trimesh.load(STL)
    mesh.vertices[:, 0] += dx
    return mesh


def _ground_area(mesh: trimesh.Trimesh) -> float:
    return float(mesh.area_faces[_ground_faces(mesh)].sum())


def test_covered_floor_is_left_alone() -> None:
    # One periodic repeat unit inside the STL's ground: nothing to change.
    assert fit_stl_to_domain(_shifted_stl(-5.0), x_length=80.0, y_length=80.0) is None


def test_upstream_fetch_gets_ground() -> None:
    # bounds x = [-20, 80]: the STL starts 20 m into the domain.
    fitted = fit_stl_to_domain(_shifted_stl(20.0), x_length=100.0, y_length=80.0)
    assert fitted is not None
    np.testing.assert_allclose(_ground_area(fitted), GROUND_AREA + 20.0 * 80.0)
    assert fitted.bounds[0, 0] == pytest.approx(0.0)
    assert np.all(fitted.face_normals[_ground_faces(fitted)][:, 2] > 0.99)


def test_buffer_replaces_buildings_by_ground() -> None:
    # bounds end at 100 m (uDALES frame) in the gap after the fourth column;
    # 20 m of buffer.
    fitted = fit_stl_to_domain(
        _shifted_stl(20.0), x_length=120.0, y_length=80.0, x_keep=100.0
    )
    assert fitted is not None
    buildings = fitted.triangles[~_ground_faces(fitted)]
    assert buildings[:, :, 0].max() == pytest.approx(95.0)
    # Ground: the STL's over x = 20-100 (four columns of footprints cut out),
    # plus the upstream strip and the buffer.
    expected = 80.0 * 160.0 - 32 * 100.0 + 20.0 * 80.0 + 20.0 * 80.0
    np.testing.assert_allclose(_ground_area(fitted), expected)


def test_buffer_edge_through_a_building_raises() -> None:
    with pytest.raises(ValueError, match="across the end of the domain bounds"):
        fit_stl_to_domain(_shifted_stl(20.0), 120.0, 80.0, x_keep=50.0)


def _model(tmp_path: pathlib.Path, outflow_buffer: float | None) -> ForwardModel:
    case = tmp_path / "case"
    shutil.copytree(CASE, case)
    return ForwardModel(
        case_dir=case,
        temp_dir=tmp_path / "temp",
        ncpu=1,
        nx=50,
        ny=40,
        nz=16,
        bounds=((-20.0, 80.0), (0.0, 80.0), (0.0, 32.0)),
        boundary_condition="inflow_outflow",
        outflow_buffer=outflow_buffer,
        verbose=False,
    )


def test_buffer_extends_the_grid_and_crops_the_state(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path, 19.0)  # rounded up to 10 cells
    namoptions = NamoptionsFile(
        model.dirs.experiment_dir / f"namoptions.{model.dirs.experiment_name}"
    )
    assert namoptions.get_value_as_int("DOMAIN", "itot") == 60
    assert namoptions.get_value_as_float("DOMAIN", "xlen") == pytest.approx(120.0)
    stl = trimesh.load(model.dirs.experiment_dir / STL.name)
    assert stl.triangles[~_ground_faces(stl)][:, :, 0].max() == pytest.approx(95.0)

    full = xarray.Dataset(
        {
            "u": (("zt", "yt", "xm"), np.random.default_rng(0).random((16, 40, 60))),
            "v": (("zt", "ym", "xt"), np.zeros((16, 40, 60))),
        },
        coords={
            "xm": np.arange(60) * 2.0 - 20.0,
            "xt": np.arange(60) * 2.0 - 19.0,
        },
    )
    cropped = model._crop_outflow_buffer(full)
    assert cropped.sizes["xm"] == cropped.sizes["xt"] == 50
    assert float(cropped.xt[-1]) == pytest.approx(79.0)
    padded = model._pad_outflow_buffer(cropped)
    assert padded.sizes["xm"] == padded.sizes["xt"] == 60
    np.testing.assert_allclose(padded.xt.values, full.xt.values)
    np.testing.assert_array_equal(
        padded.u.isel(xm=-1).values, cropped.u.isel(xm=-1).values
    )


def test_no_buffer_keeps_the_grid(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path, None)
    namoptions = NamoptionsFile(
        model.dirs.experiment_dir / f"namoptions.{model.dirs.experiment_name}"
    )
    assert namoptions.get_value_as_int("DOMAIN", "itot") == 50
    state = xarray.Dataset({"u": (("xm",), np.zeros(50))})
    assert model._crop_outflow_buffer(state) is state
    assert model._pad_outflow_buffer(state) is state


def test_buffer_needs_inflow_outflow(tmp_path: pathlib.Path) -> None:
    case = tmp_path / "case"
    shutil.copytree(CASE, case)
    with pytest.raises(ValueError, match="inflow_outflow"):
        ForwardModel(
            case_dir=case,
            temp_dir=tmp_path / "temp",
            ncpu=1,
            nx=40,
            ny=40,
            nz=16,
            bounds=((5.0, 85.0), (5.0, 85.0), (0.0, 32.0)),
            boundary_condition="periodic",
            outflow_buffer=20.0,
        )
