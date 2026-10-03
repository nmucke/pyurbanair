"""Native Vreman discrepancy regressions on the small frozen uDALES case."""

import json
import re
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr
from hydra.utils import instantiate

from tests.conftest import compose

SETTINGS = {
    "enabled": True,
    "canopy_height": 10.0,
    "height_band_over_H": [0.5, 1.5],
    "gradient_regularization": 1e-6,
    "log_multiplier_cap": 1.0,
}
VELOCITY = ("u", "v", "w")


def _forecast(root: Path, *, ncpu: int, enabled: bool) -> xr.Dataset:
    cfg = compose(
        "forward",
        "+test=forward",
        "model=pyudales_stock",
        f"model.forward_model.ncpu={ncpu}",
        # The tolerances below were calibrated after a 3 s spin-up.
        "time.spinup_time=3.0",
        root=root,
    )
    model = instantiate(
        cfg.model.forward_model,
        model_discrepancy=SETTINGS if enabled else None,
        results_dir=None,
    )
    model.run_preprocessing()
    result = model.run_single(
        params=(
            xr.Dataset(
                {name: 0.0 for name in ("sgs_bias_b0", "sgs_bias_b1", "sgs_bias_b2")}
            )
            if enabled
            else None
        )
    )
    state = cast(xr.Dataset, result.load())
    namoptions = model.dirs.experiment_dir / f"namoptions.{model.dirs.experiment_name}"
    match = re.search(r"courant\s*=\s*([0-9.]+)", namoptions.read_text())
    assert match is not None
    state.attrs["courant"] = float(match.group(1))
    return state


def _max_step(state: xr.Dataset) -> float:
    """Courant bound on uDALES's adaptive step at the output instants.

    ``dt <= courant * dx_i / |u_i|`` in every direction and cell. Each run
    writes an output at the first step past the output time, so two runs'
    output times differ by less than one such step.
    """
    return min(
        state.attrs["courant"]
        * float(np.diff(state[dim].values).max())
        / float(np.abs(state[name].values).max())
        for name, dim in (("u", "xm"), ("v", "ym"), ("w", "zm"))
    )


def _assert_finite_velocity(state: xr.Dataset) -> None:
    for name in VELOCITY:
        assert state[name].size > 0
        assert np.isfinite(state[name].values).all(), name


def _assert_velocity_close(
    actual: xr.Dataset,
    expected: xr.Dataset,
    *,
    max_range_fraction: float,
    rms_range_fraction: float,
) -> None:
    time_atol = max(_max_step(actual), _max_step(expected))
    for name in VELOCITY:
        assert actual[name].dims == expected[name].dims
        assert actual[name].shape == expected[name].shape
        for dim in actual[name].dims:
            np.testing.assert_allclose(
                actual[dim],
                expected[dim],
                rtol=0,
                atol=time_atol if dim == "time" else 1e-12,
            )
        error = actual[name].values.astype(float) - expected[name].values.astype(float)
        scale = max(float(np.ptp(expected[name].values)), 1e-12)
        assert np.max(np.abs(error)) / scale < max_range_fraction, name
        assert np.sqrt(np.mean(error**2)) / scale < rms_range_fraction, name


def _diagnostics(state: xr.Dataset) -> dict[str, float]:
    metadata = json.loads(state.attrs["model_discrepancy"])
    return {
        key: float(value)
        for line in metadata["native_diagnostics"].splitlines()
        for key, value in [line.split("=", 1)]
    }


@pytest.mark.integration  # type: ignore[misc]
def test_zero_coefficients_recover_stock_and_agree_across_mpi_ranks(
    tmp_path: Path,
) -> None:
    """Compare complete finite forecasts, not just a compiled helper kernel."""
    stock_one = _forecast(tmp_path / "stock_one", ncpu=1, enabled=False)
    stock_two = _forecast(tmp_path / "stock_two", ncpu=2, enabled=False)
    zero_one = _forecast(tmp_path / "zero_one", ncpu=1, enabled=True)
    zero_two = _forecast(tmp_path / "zero_two", ncpu=2, enabled=True)

    for state in (stock_one, stock_two, zero_one, zero_two):
        _assert_finite_velocity(state)
    for state in (zero_one, zero_two):
        diagnostics = _diagnostics(state)
        assert diagnostics["evaluated_cells"] > 0
        assert diagnostics["saturation_fraction"] == 0
        np.testing.assert_allclose(
            [diagnostics["multiplier_min"], diagnostics["multiplier_max"]],
            [1.0, 1.0],
            rtol=0,
            atol=1e-14,
        )

    # Enabled startup refresh changes the first adaptive dt despite a unit
    # multiplier. The two MPI decompositions already differ in stock uDALES.
    # Compare the extension to stock on each rank, then isolate its added rank
    # dependence by subtracting each rank's stock result.
    _assert_velocity_close(
        zero_one,
        stock_one,
        max_range_fraction=1e-3,
        rms_range_fraction=1e-4,
    )
    _assert_velocity_close(
        zero_two,
        stock_two,
        max_range_fraction=1e-3,
        rms_range_fraction=1e-4,
    )
    for name in VELOCITY:
        stock_rank_error = stock_two[name].values.astype(float) - stock_one[
            name
        ].values.astype(float)
        extended_rank_error = zero_two[name].values.astype(float) - zero_one[
            name
        ].values.astype(float)
        scale = max(float(np.ptp(stock_one[name].values)), 1e-12)
        assert np.max(np.abs(stock_rank_error)) / scale < 1e-2, name
        assert np.max(np.abs(extended_rank_error)) < (
            np.max(np.abs(stock_rank_error)) + 1e-4
        ), name
        np.testing.assert_allclose(
            extended_rank_error,
            stock_rank_error,
            rtol=0,
            atol=1e-4,
            err_msg=f"{name}: zero-coefficient extension added MPI rank dependence",
        )


@pytest.mark.integration  # type: ignore[misc]
@pytest.mark.parametrize("ncpu", [1, 2])  # type: ignore[misc]
def test_native_forecast_window_replays_cold_and_warm_state(
    tmp_path: Path, ncpu: int
) -> None:
    """Replaying one window restores native carry, clocks and input settings."""
    cfg = compose(
        "forward",
        "+test=forward",
        "model=pyudales_stock",
        f"model.forward_model.ncpu={ncpu}",
        root=tmp_path,
    )
    model = instantiate(cfg.model.forward_model, model_discrepancy=SETTINGS)
    model.run_preprocessing()
    reference_coefficients = xr.Dataset(
        {"sgs_bias_b0": 0.1, "sgs_bias_b1": 0.0, "sgs_bias_b2": 0.0}
    )
    changed_coefficients = xr.Dataset(
        {"sgs_bias_b0": -0.3, "sgs_bias_b1": 0.0, "sgs_bias_b2": 0.0}
    )

    model.begin_forecast_window()
    cold = model.run_single(params=reference_coefficients).load()
    cold_replay = model.run_single(params=reference_coefficients).load()
    xr.testing.assert_equal(cold_replay, cold)
    model.end_forecast_window(commit=True)

    initial_state = cold.isel(time=[-1])
    changed_state = initial_state.copy(deep=True)
    changed_state["u"] = changed_state["u"] + 0.05
    model.begin_forecast_window()
    warm = model.run_single(state=initial_state, params=reference_coefficients).load()
    changed_coefficient_result = model.run_single(
        state=initial_state, params=changed_coefficients
    ).load()
    changed_state_result = model.run_single(
        state=changed_state, params=reference_coefficients
    ).load()
    warm_replay = model.run_single(
        state=initial_state, params=reference_coefficients
    ).load()
    xr.testing.assert_equal(warm_replay, warm)
    assert (
        np.max(np.abs(changed_coefficient_result["u"].values - warm["u"].values)) > 1e-7
    )
    assert np.max(np.abs(changed_state_result["u"].values - warm["u"].values)) > 1e-7
    model.end_forecast_window(commit=True)
