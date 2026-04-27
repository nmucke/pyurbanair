"""Unit tests for pypalm's time-varying inflow driver."""

import pathlib

import numpy as np
import pytest
import xarray

from pypalm.utils.dynamic_driver_utils import (
    _extract_schedule,
    _prepend_spinup_plateau,
    is_time_varying_params,
    write_dynamic_driver_file,
)


def _make_time_varying_params(
    times, angles, speeds
) -> xarray.Dataset:
    data_vars: dict = {}
    if np.ndim(angles) == 0:
        data_vars["inflow_angle"] = float(angles)
    else:
        data_vars["inflow_angle"] = ("time", np.asarray(angles, dtype=float))
    if np.ndim(speeds) == 0:
        data_vars["velocity_magnitude"] = float(speeds)
    else:
        data_vars["velocity_magnitude"] = ("time", np.asarray(speeds, dtype=float))
    return xarray.Dataset(
        data_vars=data_vars, coords={"time": np.asarray(times, dtype=float)}
    )


def test_is_time_varying_detects_per_variable_time_dim() -> None:
    # Time-varying angle, scalar speed — per-variable check detects it
    params = _make_time_varying_params([0.0, 50.0, 100.0], [30.0, 45.0, 60.0], 5.0)
    assert is_time_varying_params(params) is True


def test_is_time_varying_false_for_scalar_params() -> None:
    params = xarray.Dataset(
        data_vars={"inflow_angle": 30.0, "velocity_magnitude": 5.0}
    )
    assert is_time_varying_params(params) is False


def test_is_time_varying_false_for_none() -> None:
    assert is_time_varying_params(None) is False


def test_extract_schedule_broadcasts_scalar_speed() -> None:
    params = _make_time_varying_params([0.0, 50.0, 100.0], [30.0, 45.0, 60.0], 5.0)
    times, angles, speeds = _extract_schedule(params)
    assert times.tolist() == [0.0, 50.0, 100.0]
    assert angles.tolist() == [30.0, 45.0, 60.0]
    # Scalar speed broadcast to match time length
    assert speeds.tolist() == [5.0, 5.0, 5.0]


def test_spinup_plateau_prepends_initial_values() -> None:
    times = np.array([0.0, 100.0])
    angles = np.array([30.0, 60.0])
    speeds = np.array([5.0, 4.0])
    new_t, new_a, new_s = _prepend_spinup_plateau(times, angles, speeds, 20.0)
    # User times shifted by spinup, t=0 plateau prepended with initial values
    assert new_t.tolist() == [0.0, 20.0, 120.0]
    # Values at t=0 and t=20 identical (constant plateau during spinup)
    assert new_a[0] == new_a[1] == 30.0
    assert new_s[0] == new_s[1] == 5.0


def test_spinup_plateau_noop_when_zero() -> None:
    times = np.array([0.0, 50.0])
    angles = np.array([30.0, 45.0])
    speeds = np.array([5.0, 4.0])
    new_t, new_a, new_s = _prepend_spinup_plateau(times, angles, speeds, 0.0)
    np.testing.assert_array_equal(new_t, times)
    np.testing.assert_array_equal(new_a, angles)
    np.testing.assert_array_equal(new_s, speeds)


def test_write_dynamic_driver_schema(tmp_path: pathlib.Path) -> None:
    n_time = 3
    n_z = 5
    n_y = 4
    times = np.linspace(0.0, 100.0, n_time)
    u = np.random.default_rng(0).standard_normal((n_time, n_z)).astype(np.float32)
    v = np.random.default_rng(1).standard_normal((n_time, n_z)).astype(np.float32)
    z = np.linspace(0.5, 39.5, n_z)
    zw = np.linspace(1.0, 39.0, n_z - 1)
    y = np.linspace(1.0, 39.0, n_y)

    path = tmp_path / "urban_run_dynamic"
    write_dynamic_driver_file(path, times, u, v, z, zw, y)

    ds = xarray.open_dataset(path, engine="netcdf4")
    assert set(ds.dims) >= {"time_inflow", "z", "zw", "y"}
    assert ds.sizes["time_inflow"] == n_time
    assert ds.sizes["z"] == n_z
    assert ds.sizes["zw"] == n_z - 1
    assert ds.sizes["y"] == n_y
    assert ds["time_inflow"].values[0] == 0.0
    # Each time/z slice is uniform across y (broadcast from the 1D profile)
    np.testing.assert_allclose(
        ds["inflow_plane_u"].values,
        np.broadcast_to(u[:, :, None], (n_time, n_z, n_y)),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ds["inflow_plane_v"].values,
        np.broadcast_to(v[:, :, None], (n_time, n_z, n_y)),
        atol=1e-5,
    )
    # w is zero everywhere (mean inflow only, no turbulent w forcing)
    np.testing.assert_array_equal(
        ds["inflow_plane_w"].values,
        np.zeros((n_time, n_z - 1, n_y), dtype=np.float32),
    )
    # e is zero and pt is a constant 300 K
    np.testing.assert_array_equal(
        ds["inflow_plane_e"].values, np.zeros((n_time, n_z, n_y), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        ds["inflow_plane_pt"].values,
        np.full((n_time, n_z, n_y), 300.0, dtype=np.float32),
    )
    ds.close()


def test_write_dynamic_driver_rejects_nonzero_first_time(tmp_path: pathlib.Path) -> None:
    times = np.array([10.0, 20.0, 30.0])
    u = np.zeros((3, 2), dtype=np.float32)
    v = np.zeros((3, 2), dtype=np.float32)
    z = np.array([1.0, 3.0])
    zw = np.array([2.0])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="time_inflow must start at 0.0"):
        write_dynamic_driver_file(tmp_path / "run_dynamic", times, u, v, z, zw, y)
