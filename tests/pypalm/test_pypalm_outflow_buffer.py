"""Staging tests for pypalm's ``outflow_buffer`` knob.

Like ``test_pypalm_outflow_damping.py`` these compose a real smoke model and
check the staged files WITHOUT running PALM. With a buffer PALM runs on a grid
extended downstream by flat ground, and the output is cropped back to
``domain.bounds``; absent, nothing changes. See docs/pypalm.md §8.
"""

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray
from hydra.errors import InstantiationException
from hydra.utils import instantiate

from tests.conftest import compose

NX, NY, NZ = 20, 20, 4
_SMOKE = [
    f"domain.nx={NX}",
    f"domain.ny={NY}",
    f"domain.nz={NZ}",
    # The Xie & Castro cubes reach x=15 here; dx = 1 m.
    "domain.bounds=[[0.0,20.0],[0.0,20.0],[0.0,10.0]]",
    "time.simulation_time=3.0",
    "time.output_frequency=1.0",
    "time.spinup_time=0.0",
    "model=pypalm",
    "model.forward_model.boundary_condition=inflow_outflow",
    "model.forward_model.outflow_damping.factor=1.0",
    "model.forward_model.outflow_damping.width=4.0",
]


def _make_model(tmp_path: pathlib.Path, *extra_overrides: str) -> Any:
    cfg = compose("forward", "+test=forward", *_SMOKE, *extra_overrides, root=tmp_path)
    return instantiate(cfg.model.forward_model)


def _p3d_int(fm: Any, key: str) -> int:
    from pypalm.utils.p3d_utils import P3DFile

    return int(P3DFile(fm.p3d_path).get_value("initialization_parameters", key))


def _topo(fm: Any) -> np.ndarray:
    path = fm.dirs.input_dir / f"{fm.experiment_name}_topo"
    return np.flipud(np.loadtxt(path))  # file rows run from the highest y


def test_absent_runs_on_the_bounds(tmp_path: pathlib.Path) -> None:
    fm = _make_model(tmp_path, "model.forward_model.ncpu=4")
    assert fm.nx == NX and tuple(fm.bounds[0]) == (0.0, 20.0)
    assert _p3d_int(fm, "nx") == NX - 1
    assert _topo(fm).shape == (NY, NX)


def test_buffer_extends_the_grid_with_flat_ground(tmp_path: pathlib.Path) -> None:
    plain = _make_model(tmp_path / "a", "model.forward_model.ncpu=4")
    fm = _make_model(
        tmp_path / "b",
        "model.forward_model.ncpu=4",
        # 7.5 m rounds up to 8 whole cells.
        "model.forward_model.outflow_buffer=7.5",
    )
    assert fm.nx == NX + 8
    assert tuple(fm.bounds[0]) == (0.0, 28.0)
    assert _p3d_int(fm, "nx") == NX + 8 - 1
    topo = _topo(fm)
    assert topo.shape == (NY, NX + 8)
    # Inside bounds the geometry is unchanged; the buffer is empty.
    assert np.array_equal(topo[:, :NX], _topo(plain))
    assert topo[:, :NX].max() > 0.0
    assert np.all(topo[:, NX:] == 0.0)


def test_output_is_cropped_back_to_the_bounds(tmp_path: pathlib.Path) -> None:
    fm = _make_model(
        tmp_path, "model.forward_model.ncpu=4", "model.forward_model.outflow_buffer=8"
    )
    n = fm.nx
    # A PALM-shaped 3D output on the extended grid (native, 0-based axes).
    ds = xarray.Dataset(
        {
            "u": (("time", "zu_3d", "y", "xu"), np.ones((3, NZ + 2, NY, n))),
            "v": (("time", "zu_3d", "yv", "x"), np.ones((3, NZ + 2, NY, n))),
            "w": (("time", "zw_3d", "y", "x"), np.full((3, NZ + 2, NY, n), 0.1)),
        },
        coords={
            "time": [1.0, 2.0, 3.0],
            "zu_3d": np.arange(NZ + 2) - 0.5,
            "zw_3d": np.arange(NZ + 2) * 1.0,
            "x": np.arange(n) + 0.5,
            "xu": np.arange(n) * 1.0,
            "y": np.arange(NY) + 0.5,
            "yv": np.arange(NY) * 1.0,
        },
    )
    fm.dirs.output_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(fm.dirs.output_dir / f"{fm.experiment_name}_3d.nc")

    state = fm._load_and_postprocess_state()
    assert state.sizes["x"] == NX and state.sizes["xu"] == NX
    assert float(state.x[-1]) == 19.5 and float(state.xu[-1]) == 19.0


def test_warm_start_fills_the_buffer_from_the_last_column(
    tmp_path: pathlib.Path,
) -> None:
    from pypalm.utils.warm_start_utils import write_warmstart_driver

    fm = _make_model(
        tmp_path, "model.forward_model.ncpu=4", "model.forward_model.outflow_buffer=8"
    )
    xu = np.arange(NX) * 1.0
    z = np.arange(NZ) * 2.5 + 1.25
    state = xarray.Dataset(
        {
            "u": (("z", "y", "xu"), np.broadcast_to(xu, (NZ, NY, NX))),
            "v": (("z", "yv", "x"), np.zeros((NZ, NY, NX))),
            "w": (("z", "y", "x"), np.zeros((NZ, NY, NX))),
        },
        coords={
            "z": z,
            "xu": xu,
            "x": xu + 0.5,
            "y": np.arange(NY) + 0.5,
            "yv": np.arange(NY) * 1.0,
        },
    )
    write_warmstart_driver(
        fm.dynamic_driver_path, state, fm.bounds, fm.nx, NY, NZ, pt_surface=300.0
    )
    u = xarray.open_dataset(fm.dynamic_driver_path)["init_atmosphere_u"].values
    # u(i) for i = 1..nx-1 at x = i: the state up to its last column, then
    # that column repeated through the buffer.
    assert np.allclose(u[0, 0, : NX - 1], np.arange(1, NX))
    assert np.allclose(u[0, 0, NX - 1 :], NX - 1)


@pytest.mark.parametrize(  # type: ignore[misc]
    "override, match",
    [
        ("model.forward_model.outflow_buffer=0.0", "> 0"),
        ("model.forward_model.boundary_condition=periodic", "inflow_outflow"),
    ],
)
def test_invalid_buffer_raises(
    tmp_path: pathlib.Path, override: str, match: str
) -> None:
    with pytest.raises(InstantiationException, match=match):
        _make_model(tmp_path, "model.forward_model.outflow_buffer=8", override)


def test_damping_wider_than_the_buffer_warns(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="pypalm.forward_model"):
        _make_model(
            tmp_path,
            "model.forward_model.ncpu=1",
            "model.forward_model.outflow_buffer=2",
        )
    assert "wider than" in caplog.text
