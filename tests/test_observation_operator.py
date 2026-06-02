"""Unit tests for TemporalObservationOperator's seconds-based interval mode."""

import numpy as np
import pytest
import xarray
from data_assimilation.observation_operator import (
    ObservationOperator,
    TemporalObservationOperator,
)


def _make_state(time_values: list[float]) -> xarray.Dataset:
    """Build a tiny pylbm-style state whose `u` equals the time at every cell.

    With ``u[t] == t`` everywhere, the mean over any set of frames is just the
    mean of their time coordinates, which makes interval aggregation easy to
    assert.
    """
    nt = len(time_values)
    nz = ny = nx = 2
    time = np.asarray(time_values, dtype=float)
    u = np.broadcast_to(time[:, None, None, None], (nt, nz, ny, nx)).astype(float)
    return xarray.Dataset(
        {"u": (("time", "z", "y", "x"), u.copy())},
        coords={
            "time": time,
            "z": np.arange(nz),
            "y": np.arange(ny),
            "x": np.arange(nx),
        },
    )


def _single_sensor_op() -> ObservationOperator:
    return ObservationOperator(
        obs_ids_x=[0],
        obs_ids_y=[0],
        obs_ids_z=[0],
        obs_states=["u"],
        solver_name="pylbm",
    )


def test_intervals_bin_by_seconds() -> None:
    # Frames every 2 s; a 4 s interval bins them as {0,2}, {4,6}, {8,10}.
    state = _make_state([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    top = TemporalObservationOperator(
        _single_sensor_op(),
        mode="intervals",
        interval_seconds=4.0,
        aggregation_mode="mean",
    )

    obs = top(state)

    np.testing.assert_allclose(obs, [1.0, 5.0, 9.0])
    assert top.num_obs == 3


def test_interval_seconds_independent_of_step_count() -> None:
    # Doubling the sampling cadence within the same 4 s windows must not change
    # the number of intervals — binning is by seconds, not step count.
    state = _make_state([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    top = TemporalObservationOperator(
        _single_sensor_op(),
        mode="intervals",
        interval_seconds=4.0,
        aggregation_mode="mean",
    )

    obs = top(state)

    # Bins {0,1,2,3} and {4,5,6,7} -> means 1.5 and 5.5.
    np.testing.assert_allclose(obs, [1.5, 5.5])
    assert top.num_obs == 2


def test_interval_larger_than_span_yields_single_bin() -> None:
    state = _make_state([0.0, 2.0, 4.0, 6.0])
    top = TemporalObservationOperator(
        _single_sensor_op(),
        mode="intervals",
        interval_seconds=1000.0,
        aggregation_mode="mean",
    )

    obs = top(state)

    np.testing.assert_allclose(obs, [3.0])
    assert top.num_obs == 1


def test_missing_interval_seconds_raises() -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        TemporalObservationOperator(_single_sensor_op(), mode="intervals")


def test_non_positive_interval_seconds_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        TemporalObservationOperator(
            _single_sensor_op(), mode="intervals", interval_seconds=0.0
        )
