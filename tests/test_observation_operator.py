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


def test_full_mode_num_obs_before_first_call_raises() -> None:
    # "full" mode with num_time_steps unknown must not crash with a TypeError
    # on ``num_obs * None`` -- it raises a clear RuntimeError like intervals.
    top = TemporalObservationOperator(_single_sensor_op(), mode="full")
    with pytest.raises(RuntimeError, match="number of time steps"):
        _ = top.num_obs


def test_full_mode_num_obs_after_first_call() -> None:
    state = _make_state([0.0, 1.0, 2.0])
    top = TemporalObservationOperator(_single_sensor_op(), mode="full")
    top(state)
    assert top.num_obs == 3  # 3 time steps * 1 sensor * 1 state


def test_interval_with_empty_absolute_bin_raises() -> None:
    # Times 0, 2 fall in absolute interval 0; 10 falls in interval 2; 12 falls
    # in interval 3. Interval 1 ([4, 8)) has no frames at all -- binning only
    # over populated bins would silently shift every element after it, so this
    # must raise instead.
    state = _make_state([0.0, 2.0, 10.0, 12.0])
    top = TemporalObservationOperator(
        _single_sensor_op(),
        mode="intervals",
        interval_seconds=4.0,
        aggregation_mode="mean",
    )

    with pytest.raises(ValueError, match="Absolute interval 1"):
        top(state)


def test_interval_all_absolute_bins_populated_unchanged() -> None:
    # Irregular frame spacing (0, 3, 4, 9) still lands one frame in each of the
    # 3 absolute intervals spanned ([0,4), [4,8), [8,12)) -- no empty bin, so
    # behavior must be identical to the pre-fix populated-bins-only binning.
    state = _make_state([0.0, 3.0, 4.0, 9.0])
    top = TemporalObservationOperator(
        _single_sensor_op(),
        mode="intervals",
        interval_seconds=4.0,
        aggregation_mode="mean",
    )

    obs = top(state)

    np.testing.assert_allclose(obs, [1.5, 4.0, 9.0])
    assert top.num_obs == 3


def test_interval_count_change_between_calls_raises() -> None:
    # First call establishes 3 intervals (C_D would be sized from it); a later
    # window that produces a different count must raise, not return a
    # mismatched-length vector.
    top = TemporalObservationOperator(
        _single_sensor_op(), mode="intervals", interval_seconds=4.0
    )
    top(_make_state([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))  # 3 bins
    assert top.num_obs == 3
    with pytest.raises(ValueError, match="Interval count changed"):
        top(_make_state([0.0, 2.0, 4.0, 6.0]))  # 2 bins
