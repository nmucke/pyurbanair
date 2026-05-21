"""P0 round-trip tests for state_io / params_io (no NN, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray

from neural_surrogates.data.grid import GridMeta
from neural_surrogates.utils import params_io, state_io
from neural_surrogates.utils.schema import ParamSchema


def _grid() -> GridMeta:
    return GridMeta(nx=8, ny=6, nz=4, bounds=((0.0, 8.0), (0.0, 6.0), (0.0, 4.0)))


def _state(grid: GridMeta, n_t: int, var_names=("u", "v", "w")) -> xarray.Dataset:
    rng = np.random.default_rng(0)
    data_vars = {
        name: (("time", "z", "y", "x"), rng.standard_normal((n_t, grid.nz, grid.ny, grid.nx)))
        for name in var_names
    }
    return xarray.Dataset(
        data_vars=data_vars,
        coords={"time": np.arange(n_t), "z": grid.z, "y": grid.y, "x": grid.x},
    )


def test_state_tensor_roundtrip() -> None:
    grid = _grid()
    var_names = ("u", "v", "w")
    ds = _state(grid, n_t=5)

    tensor = state_io.state_to_tensor(ds, grid, var_names)
    assert tensor.shape == (5, 3, grid.nz, grid.ny, grid.nx)

    back = state_io.tensor_to_state(tensor, grid, var_names)
    for name in var_names:
        np.testing.assert_allclose(back[name].values, ds[name].values, rtol=1e-5)


def test_timeless_warm_start_frame_normalizes_to_one_frame() -> None:
    grid = _grid()
    var_names = ("u", "v", "w")
    ds = _state(grid, n_t=3).isel(time=-1)  # strip the time dim
    assert "time" not in ds.dims

    tensor = state_io.state_to_tensor(ds, grid, var_names)
    assert tensor.shape == (1, 3, grid.nz, grid.ny, grid.nx)


def test_extract_history_left_pads_and_masks() -> None:
    grid = _grid()
    tensor = state_io.state_to_tensor(_state(grid, n_t=2), grid, ("u", "v", "w"))

    hist, mask = state_io.extract_history(tensor, history_len=4)
    assert hist.shape == (4, 3, grid.nz, grid.ny, grid.nx)
    np.testing.assert_array_equal(mask, [0.0, 0.0, 1.0, 1.0])
    # padded slots are zero; last two equal the real frames
    np.testing.assert_array_equal(hist[:2], 0.0)
    np.testing.assert_allclose(hist[2:], tensor)


def test_extract_history_k1_takes_last_frame() -> None:
    grid = _grid()
    tensor = state_io.state_to_tensor(_state(grid, n_t=5), grid, ("u", "v", "w"))
    hist, mask = state_io.extract_history(tensor, history_len=1)
    assert hist.shape == (1, 3, grid.nz, grid.ny, grid.nx)
    np.testing.assert_array_equal(mask, [1.0])
    np.testing.assert_allclose(hist[0], tensor[-1])


def test_trim_to_window_matches_pylbm_contract() -> None:
    grid = _grid()
    ds = _state(grid, n_t=10)
    trimmed = state_io.trim_to_window(ds, num_outputs=4)
    assert trimmed.sizes["time"] == 4
    np.testing.assert_array_equal(trimmed["time"].values, [0, 1, 2, 3])
    # keeps the LAST four frames (drops the spin-up prefix)
    np.testing.assert_allclose(trimmed["u"].values, ds["u"].isel(time=slice(-4, None)).values)


def test_params_conditioning_scalar_broadcast_and_sincos() -> None:
    schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    params = xarray.Dataset(
        data_vars={"inflow_angle": 90.0, "velocity_magnitude": 5.0}
    )
    cond = params_io.params_to_conditioning(
        params, schema, num_steps=3, output_frequency=1.0
    )
    assert cond.shape == (3, schema.conditioning_dim)
    # angle 90deg -> sin=1, cos=0; velocity broadcast to 5.0
    np.testing.assert_allclose(cond[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(cond[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(cond[:, 2], 5.0)


def test_params_conditioning_linear_interp_over_time() -> None:
    schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": ("time", [0.0, 90.0]),
            "velocity_magnitude": ("time", [0.0, 4.0]),
        },
        coords={"time": [0.0, 4.0]},
    )
    cond = params_io.params_to_conditioning(
        params, schema, num_steps=5, output_frequency=1.0
    )
    # target_times = [0,1,2,3,4]; velocity interpolates 0..4 linearly
    np.testing.assert_allclose(cond[:, 2], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_schema_includes_pressure_gradient_only_when_requested() -> None:
    pylbm_schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    udales_schema = ParamSchema(
        names=("inflow_angle", "velocity_magnitude", "pressure_gradient_magnitude")
    )
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": 0.0,
            "velocity_magnitude": 1.0,
            "pressure_gradient_magnitude": 0.01,
        }
    )
    pylbm_cond = params_io.params_to_conditioning(
        params, pylbm_schema, num_steps=2, output_frequency=1.0
    )
    udales_cond = params_io.params_to_conditioning(
        params, udales_schema, num_steps=2, output_frequency=1.0
    )
    assert pylbm_cond.shape[1] == 3  # sin, cos, velocity
    assert udales_cond.shape[1] == 4  # + pressure gradient

    # a pylbm-style params Dataset (no pressure gradient) must raise under the
    # uDALES schema rather than silently dropping it.
    pylbm_params = xarray.Dataset(
        data_vars={"inflow_angle": 0.0, "velocity_magnitude": 1.0}
    )
    with pytest.raises(KeyError):
        params_io.params_to_conditioning(
            pylbm_params, udales_schema, num_steps=2, output_frequency=1.0
        )
