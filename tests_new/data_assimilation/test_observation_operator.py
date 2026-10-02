"""Unit tests for the temporal observation operator and observation aggregation."""

import numpy as np
import pytest
import xarray
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    TemporalObservationOperator,
    flatten_observations,
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


def _make_observations(
    time_values: list[float],
    num_obs: int = 1,
    ensemble_size: int | None = None,
) -> xarray.DataArray:
    """Observation DataArray whose value at time t is ``t + obs_index``.

    Ensemble members are offset by ``100 * member`` so a flattening/transpose
    mistake is immediately visible.
    """
    time = np.asarray(time_values, dtype=float)
    values = time[:, None] + np.arange(num_obs, dtype=float)[None, :]
    if ensemble_size is None:
        return xarray.DataArray(values, dims=("time", "obs"), coords={"time": time})
    stacked = np.stack(
        [values + 100.0 * member for member in range(ensemble_size)], axis=0
    )
    return xarray.DataArray(
        stacked, dims=("ensemble", "time", "obs"), coords={"time": time}
    )


def test_temporal_operator_returns_time_resolved_dataarray() -> None:
    state = _make_state([0.0, 2.0, 4.0])
    top = TemporalObservationOperator(_single_sensor_op())

    obs = top(state)

    assert isinstance(obs, xarray.DataArray)
    assert obs.dims == ("time", "obs")
    assert obs.shape == (3, 1)
    # The state's seconds-valued time coordinate is carried through.
    np.testing.assert_allclose(obs["time"].values, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(obs.values.ravel(), [0.0, 2.0, 4.0])


def test_temporal_operator_obs_axis_layout() -> None:
    # "obs" layout is [var0 all sensors, var1 all sensors, ...]; with two
    # sensors and u == time, v == 0 the block structure is visible directly.
    state = _make_state([0.0, 1.0])
    state["v"] = xarray.zeros_like(state["u"])
    operator = ObservationOperator(
        obs_ids_x=[0, 1],
        obs_ids_y=[0, 0],
        obs_ids_z=[0, 0],
        obs_states=["u", "v"],
        solver_name="pylbm",
    )
    top = TemporalObservationOperator(operator)

    obs = top(state)

    assert obs.shape == (2, 4)  # 2 frames x (2 states * 2 sensors)
    np.testing.assert_allclose(obs.isel(time=1).values, [1.0, 1.0, 0.0, 0.0])


def test_temporal_operator_ensemble_dims() -> None:
    states = xarray.concat(
        [_make_state([0.0, 1.0, 2.0]), _make_state([0.0, 1.0, 2.0]) + 10.0],
        dim="ensemble",
    )
    top = TemporalObservationOperator(_single_sensor_op())

    obs = top(states)

    assert obs.dims == ("ensemble", "time", "obs")
    assert obs.shape == (2, 3, 1)
    np.testing.assert_allclose(obs["time"].values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(obs.isel(ensemble=1).values.ravel(), [10.0, 11.0, 12.0])


def test_temporal_operator_requires_a_time_dim() -> None:
    state = _make_state([0.0, 1.0]).isel(time=-1)
    top = TemporalObservationOperator(_single_sensor_op())

    with pytest.raises(ValueError, match="'time' dimension"):
        top(state)


def test_intervals_bin_by_seconds() -> None:
    # Frames every 2 s; a 4 s interval bins them as {0,2}, {4,6}, {8,10}.
    obs = _make_observations([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    aggregate = AggregateObservations(interval_seconds=4.0, mode="mean")

    aggregated = aggregate(obs)

    assert aggregated.dims == ("time", "obs")
    np.testing.assert_allclose(aggregated.values.ravel(), [1.0, 5.0, 9.0])
    # Each interval is labelled by its start time.
    np.testing.assert_allclose(aggregated["time"].values, [0.0, 4.0, 8.0])


def test_interval_seconds_independent_of_step_count() -> None:
    # Doubling the sampling cadence within the same 4 s windows must not change
    # the number of intervals — binning is by seconds, not step count.
    obs = _make_observations([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    aggregate = AggregateObservations(interval_seconds=4.0, mode="mean")

    aggregated = aggregate(obs)

    # Bins {0,1,2,3} and {4,5,6,7} -> means 1.5 and 5.5.
    np.testing.assert_allclose(aggregated.values.ravel(), [1.5, 5.5])


def test_interval_larger_than_span_yields_single_bin() -> None:
    obs = _make_observations([0.0, 2.0, 4.0, 6.0])
    aggregate = AggregateObservations(interval_seconds=1000.0, mode="mean")

    aggregated = aggregate(obs)

    np.testing.assert_allclose(aggregated.values.ravel(), [3.0])


def test_aggregation_modes() -> None:
    times = [0.0, 1.0, 2.0, 3.0]
    expected = {"mean": 1.5, "median": 1.5, "max": 3.0, "min": 0.0}

    for mode, value in expected.items():
        aggregated = AggregateObservations(interval_seconds=10.0, mode=mode)(
            _make_observations(times)
        )
        np.testing.assert_allclose(aggregated.values.ravel(), [value])


def test_aggregation_preserves_ensemble_dim_order() -> None:
    obs = _make_observations([0.0, 1.0, 2.0, 3.0], num_obs=2, ensemble_size=3)
    aggregate = AggregateObservations(interval_seconds=2.0, mode="mean")

    aggregated = aggregate(obs)

    assert aggregated.dims == ("ensemble", "time", "obs")
    assert aggregated.shape == (3, 2, 2)
    # Member 1, interval 0: times {0,1} -> mean 0.5, plus the obs offset and the
    # member offset of 100.
    np.testing.assert_allclose(
        aggregated.isel(ensemble=1, time=0).values, [100.5, 101.5]
    )


def test_non_positive_interval_seconds_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        AggregateObservations(interval_seconds=0.0)


def test_invalid_aggregation_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid mode"):
        AggregateObservations(interval_seconds=4.0, mode="sum")


def test_interval_with_empty_absolute_bin_raises() -> None:
    # Times 0, 2 fall in absolute interval 0; 10 falls in interval 2; 12 falls
    # in interval 3. Interval 1 ([4, 8)) has no frames at all -- binning only
    # over populated bins would silently shift every element after it, so this
    # must raise instead.
    obs = _make_observations([0.0, 2.0, 10.0, 12.0])
    aggregate = AggregateObservations(interval_seconds=4.0, mode="mean")

    with pytest.raises(ValueError, match="Absolute interval 1"):
        aggregate(obs)


def test_interval_all_absolute_bins_populated_unchanged() -> None:
    # Irregular frame spacing (0, 3, 4, 9) still lands one frame in each of the
    # 3 absolute intervals spanned ([0,4), [4,8), [8,12)) -- no empty bin, so
    # behavior must be identical to populated-bins-only binning.
    obs = _make_observations([0.0, 3.0, 4.0, 9.0])
    aggregate = AggregateObservations(interval_seconds=4.0, mode="mean")

    aggregated = aggregate(obs)

    np.testing.assert_allclose(aggregated.values.ravel(), [1.5, 4.0, 9.0])


def test_interval_count_change_between_calls_raises() -> None:
    # First call establishes 3 intervals (C_D would be sized from it); a later
    # window that produces a different count must raise, not return a
    # mismatched-length vector.
    aggregate = AggregateObservations(interval_seconds=4.0)
    aggregate(_make_observations([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))  # 3 bins
    with pytest.raises(ValueError, match="Interval count changed"):
        aggregate(_make_observations([0.0, 2.0, 4.0, 6.0]))  # 2 bins


def test_flatten_observations_is_time_major() -> None:
    obs = _make_observations([0.0, 1.0, 2.0], num_obs=3)

    flat = flatten_observations(obs)

    assert flat.shape == (9,)
    # Identical to concatenating the per-time observation blocks in order.
    expected = np.concatenate([obs.isel(time=t).values for t in range(3)])
    np.testing.assert_allclose(flat, expected)


def test_flatten_observations_ensemble() -> None:
    obs = _make_observations([0.0, 1.0, 2.0], num_obs=3, ensemble_size=4)

    flat = flatten_observations(obs)

    assert flat.shape == (4, 9)
    for member in range(4):
        expected = np.concatenate(
            [obs.isel(ensemble=member, time=t).values for t in range(3)]
        )
        np.testing.assert_allclose(flat[member], expected)


def test_flatten_observations_transposes_before_flattening() -> None:
    # A DataArray whose dims arrive in a different order must still flatten
    # time-major per member.
    obs = _make_observations([0.0, 1.0], num_obs=2, ensemble_size=2)
    permuted = obs.transpose("time", "obs", "ensemble")

    np.testing.assert_allclose(
        flatten_observations(permuted), flatten_observations(obs)
    )


def test_create_observation_operator_skips_the_temporal_wrapper_without_a_mode() -> (
    None
):
    """No ``temporal_mode`` in the obs config -> the bare spatial operator.

    Configs that never aggregate in time (forward-only runs, and the filtering
    entry point when it observes instantaneous state) leave ``temporal_mode`` out
    or set it to null. Wrapping those in a ``TemporalObservationOperator`` would
    add a time axis those callers do not want -- the same "no-op when the field
    is absent" rule the backends follow for optional parameters.
    """
    from pyurbanair.config.hydra_helpers import create_observation_operator

    base = {
        "mode": "points",
        "x_points": [1.0, 2.0],
        "y_points": [1.0, 2.0],
        "z_points": [2.0, 2.0],
        "states": ["u", "v"],
    }

    for obs in (base, {**base, "temporal_mode": None}):
        operator = create_observation_operator(obs, "pylbm")
        assert type(operator) is ObservationOperator
        assert operator.num_sensors == 2

    temporal = create_observation_operator({**base, "temporal_mode": "full"}, "pylbm")
    assert isinstance(temporal, TemporalObservationOperator)
    assert temporal.observation_operator.num_sensors == 2

    # Aggregation is no longer an operator mode; the old values must fail loudly.
    with pytest.raises(ValueError, match="AggregateObservations"):
        create_observation_operator({**base, "temporal_mode": "intervals"}, "pylbm")


def test_create_aggregate_observations() -> None:
    from pyurbanair.config.hydra_helpers import create_aggregate_observations

    assert create_aggregate_observations({}) is None
    assert create_aggregate_observations({"interval_seconds": None}) is None

    aggregate = create_aggregate_observations({"interval_seconds": 30.0})
    assert isinstance(aggregate, AggregateObservations)
    assert aggregate.interval_seconds == 30.0
    assert aggregate.mode == "mean"  # default

    aggregate = create_aggregate_observations(
        {"interval_seconds": 4, "aggregation_mode": "max"}
    )
    assert aggregate is not None
    assert aggregate.interval_seconds == 4.0
    assert aggregate.mode == "max"

    # A null aggregation_mode in the config falls back to the default.
    aggregate = create_aggregate_observations(
        {"interval_seconds": 4.0, "aggregation_mode": None}
    )
    assert aggregate is not None
    assert aggregate.mode == "mean"
