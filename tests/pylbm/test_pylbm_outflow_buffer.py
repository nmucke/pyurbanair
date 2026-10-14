"""pylbm's ``outflow_buffer``: empty ground past ``bounds``' xmax, cropped away.

As in pyudales and pypalm, the grid is extended downstream by whole cells, the
buffer gets no buildings, every state is cropped back to ``bounds``, and a warm
start fills the buffer by repeating the last column.
"""

import pathlib
import re
from typing import Any, Optional

import numpy as np
import pytest
import xarray
from hydra.utils import instantiate
from pylbm.stl_to_lbm import compute_solid_occupancy
from pylbm.utils.warm_start_utils import (
    _try_load_restart_distribution,
    identify_latest_restart_iteration,
    restart_file_name,
)

from tests.conftest import compose

REPO = pathlib.Path(__file__).resolve().parents[2]
LBM_SRC = REPO / "libs" / "pylbm" / "LBM" / "src"
STL_PATH = REPO / "geometries" / "xie_and_castro" / "xie_castro_2008_STL.stl"
# The case's x range, cut short in y and z to keep the voxelisation quick.
NX, NY, NZ = 50, 8, 10
BOUNDS = ((-20.0, 80.0), (0.0, 16.0), (0.0, 20.0))
PARAMS = xarray.Dataset({"inflow_angle": 10.0, "velocity_magnitude": 5.0})

pytestmark = pytest.mark.skipif(
    not (LBM_SRC / "m_mkinfile.F90").exists(), reason="LBM sources not checked out"
)


def _solver_template_infile() -> str:
    """The infile.in the solver writes for itself (m_mkinfile.F90)."""
    source = (LBM_SRC / "m_mkinfile.F90").read_text()
    return "\n".join(re.findall(r"write\(10,'\(a\)'\)'(.*)'", source)) + "\n"


def _model(
    tmp_path: pathlib.Path,
    outflow_buffer: Optional[float],
    boundary_condition: str = "inflow_outflow",
) -> Any:
    from pylbm.forward_model import ForwardModel

    return ForwardModel(
        stl_path=STL_PATH,
        temp_dir=tmp_path,
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=BOUNDS,
        cuda=False,
        verbose=False,
        boundary_condition=boundary_condition,
        outflow_buffer=outflow_buffer,
        simulation_time=2.0,
        output_frequency=1.0,
    )


def _state(nx: int, x: np.ndarray) -> xarray.Dataset:
    """A (z, y, x) state whose fields vary along x, blanking 1 in one column."""
    shape = (NZ, NY, nx)
    ramp = np.broadcast_to(np.arange(nx, dtype=np.float32), shape)
    blanking = np.zeros(shape, dtype=np.float32)
    blanking[:, :, nx // 2] = 1.0
    return xarray.Dataset(
        {
            "rho": (("z", "y", "x"), np.ones(shape, dtype=np.float32)),
            "u": (("z", "y", "x"), 1.0 + 0.01 * ramp),
            "v": (("z", "y", "x"), 0.5 * np.ones(shape, dtype=np.float32)),
            "w": (("z", "y", "x"), np.zeros(shape, dtype=np.float32)),
            "blanking": (("z", "y", "x"), blanking),
        },
        coords={"x": x},
    )


def test_the_grid_is_extended_by_whole_cells(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path, outflow_buffer=19.0)  # 9.5 cells -> 10 (20 m)

    assert model.x_grid.size == NX + 10
    np.testing.assert_allclose(model.x_grid[-1], 99.0)
    dims = model.dirs.mod_dimensions_path.read_text()
    assert re.search(rf"nx\s*=\s*{NX + 10}\b", dims)


def test_the_buffer_holds_no_buildings(tmp_path: pathlib.Path) -> None:
    """Past xmax the voxelisation is empty; inside it is the unbuffered one."""
    bounds = dict(xmin=-20.0, xmax=100.0, ymin=0.0, ymax=16.0, zmin=0.0, zmax=20.0)
    # The STL has a fifth block column at x = 85..95, inside the buffer.
    uncut = compute_solid_occupancy(STL_PATH, NX + 10, NY, NZ, bounds)
    assert uncut[NX:].any()
    cut = compute_solid_occupancy(STL_PATH, NX + 10, NY, NZ, bounds, x_max=80.0)
    assert not cut[NX:].any()
    original = compute_solid_occupancy(STL_PATH, NX, NY, NZ, dict(bounds, xmax=80.0))
    np.testing.assert_array_equal(cut[:NX], original)


def test_absent_leaves_the_grid_alone(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path, outflow_buffer=None)
    assert model.x_grid.size == NX
    state = _state(NX, model.x_grid)
    assert model._crop_outflow_buffer(state) is state
    assert model._pad_outflow_buffer(state) is state


@pytest.mark.parametrize(  # type: ignore[misc]
    "kwargs, match",
    [
        ({"outflow_buffer": 20.0, "boundary_condition": "periodic"}, "inflow_outflow"),
        ({"outflow_buffer": 0.0}, "> 0"),
        ({"outflow_buffer": -2.0}, "> 0"),
    ],
)
def test_bad_buffers_raise(tmp_path: pathlib.Path, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _model(tmp_path, **kwargs)


def test_crop_and_pad(tmp_path: pathlib.Path) -> None:
    model = _model(tmp_path, outflow_buffer=20.0)
    full = _state(NX + 10, model.x_grid)

    cropped = model._crop_outflow_buffer(full)
    assert cropped.sizes["x"] == NX
    np.testing.assert_allclose(cropped.x, (np.arange(NX) + 0.5) * 2.0 - 20.0)

    padded = model._pad_outflow_buffer(cropped)
    assert padded.sizes["x"] == NX + 10
    np.testing.assert_allclose(padded.x, model.x_grid)
    for name in ("rho", "u", "v", "w"):
        tail = padded[name].isel(x=slice(NX, None)).values
        last = cropped[name].isel(x=-1).values[..., None]
        np.testing.assert_array_equal(tail, np.broadcast_to(last, tail.shape))
    assert float(padded.blanking.isel(x=slice(NX, None)).max()) == 0.0
    with pytest.raises(ValueError, match="x cells"):
        model._pad_outflow_buffer(full.isel(x=slice(0, NX - 1)))


def test_a_warm_start_writes_the_extended_restart(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cropped state a window returns warm-starts the extended grid."""
    model = _model(tmp_path, outflow_buffer=20.0)
    model.dirs.infile_path.write_text(_solver_template_infile())
    monkeypatch.setattr(model, "_verify_prebuilt_binary", lambda: None)
    model.compile(compile=False)
    model._set_scaling_factors(PARAMS)

    cropped = _state(NX, (np.arange(NX) + 0.5) * 2.0 - 20.0)
    model._prepare_warmstart(cropped)

    iteration = identify_latest_restart_iteration(model.dirs)
    assert iteration is not None
    restart = model.dirs.experiment_dir / "restart" / restart_file_name(iteration)
    f = _try_load_restart_distribution(restart, NX + 10, NY, NZ)
    assert f is not None and f.shape == (27, NX + 12, NY + 2, NZ + 2)


def _tiny_run(root: pathlib.Path) -> tuple[Any, xarray.Dataset, xarray.Dataset]:
    """A cold and a warm window of the tiny LBM case with a 4 m buffer."""
    cfg = compose(
        "forward",
        "+test=forward",
        "model=pylbm_tiny",
        "model.forward_model.boundary_condition=inflow_outflow",
        "flow.outflow_buffer=4.0",
        "++model.forward_model.profile_config.z_ref=6.0",
        root=root,
    )
    model = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=model)
    cold = model.run_single(params=PARAMS).load()
    warm = model.run_single(state=cold, params=PARAMS).load()
    return cfg, cold, warm


@pytest.mark.integration  # type: ignore[misc]
def test_a_buffered_run_returns_the_case_grid(tmp_path: pathlib.Path) -> None:
    cfg, cold, warm = _tiny_run(tmp_path)
    (xmin, xmax) = cfg.domain.bounds[0]
    dx = (xmax - xmin) / cfg.domain.nx
    case_x = xmin + (np.arange(cfg.domain.nx) + 0.5) * dx
    for state in (cold, warm):
        np.testing.assert_allclose(state.x, case_x)
        for name in ("u", "v", "w", "rho"):
            assert np.isfinite(state[name]).all(), name
