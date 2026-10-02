"""Resolved TKE histories keep temporal and ensemble reductions in order."""

import numpy as np
import pytest
import xarray as xr
from evaluation.turbulence import sensor_tke_evolution


def test_sensor_tke_evolution_preserves_member_energy() -> None:
    time = np.arange(5, dtype=float)
    truth_velocity = np.zeros((3, 5, 1))
    truth_velocity[0, :, 0] = [0, 2, 0, 2, 0]
    member_velocity = np.stack([truth_velocity, 2 * truth_velocity], axis=1)
    truth = xr.DataArray(
        truth_velocity,
        dims=("component", "time", "sensor"),
        coords={"component": ["u", "v", "w"], "time": time},
    )
    predicted = xr.DataArray(
        member_velocity,
        dims=("component", "ensemble", "time", "sensor"),
        coords={"component": ["u", "v", "w"], "time": time},
    )

    result = sensor_tke_evolution(truth, predicted, window_seconds=2.0)

    assert result is not None
    assert result.window_frames == 3
    assert result.members[0] == pytest.approx(result.truth)
    assert result.members[1] == pytest.approx(4 * result.truth)
    assert result.mean == pytest.approx(2.5 * result.truth)
    assert result.mean_error == pytest.approx(1.5 * result.truth)
    assert np.any(result.truth > 0)


def test_sensor_tke_evolution_uses_physical_cycle_times() -> None:
    values = np.zeros((3, 4, 1))
    values[0, :, 0] = [0, 1, 0, 1]
    truth = xr.DataArray(
        values,
        dims=("component", "time", "sensor"),
        coords={"component": ["u", "v", "w"], "time": np.arange(4)},
    )
    predicted = truth.expand_dims(ensemble=[0])

    result = sensor_tke_evolution(
        truth,
        predicted,
        window_seconds=4.0,
        time_seconds=np.array([2.0, 4.0, 6.0, 8.0]),
    )

    assert result is not None
    assert result.time == pytest.approx([2, 4, 6, 8])
    assert result.window_frames == 3
    assert result.window_span_seconds == pytest.approx(4.0)
