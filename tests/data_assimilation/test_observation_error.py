"""Analytical checks for the opt-in diagonal observation likelihood."""

from typing import Any

import numpy as np
import pytest
import xarray as xr
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
)
from data_assimilation.smoothing.esmda import ParameterESMDA


def _operator() -> ObservationOperator:
    return ObservationOperator(
        obs_states=["u", "v"],
        obs_x=[0.0, 1.0],
        obs_y=[0.0, 0.0],
        obs_z=[1.0, 3.0],
    )


def _observations(times: list[float]) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((len(times), 4)),
        dims=("time", "obs"),
        coords={"time": times},
    )


def test_mean_covariance_matches_explicit_matrix_with_unequal_bins() -> None:
    raw = _observations([0.0, 1.0, 2.0, 4.0, 5.0, 8.0])
    aggregate = AggregateObservations(4.0)
    spec = ObservationErrorSpec(
        instrument_std={"default": 2.0, "components": {"v": 3.0}},
        representation_std=1.0,
    )
    resolved = spec.resolve(raw, _operator(), aggregate)

    assert resolved.frame_ids == ((0, 1, 2), (3, 4), (5,))
    np.testing.assert_allclose(resolved.times, [0.0, 4.0, 8.0])
    np.testing.assert_allclose([sum(w) for w in resolved.weights], 1.0)
    np.testing.assert_allclose(aggregate(raw).time, resolved.times)
    np.testing.assert_allclose([len(ids) for ids in resolved.frame_ids], [3, 2, 1])

    # Explicit A R A^T on the time axis, separately for every obs label.
    a = np.zeros((3, 6))
    for b, (ids, weights) in enumerate(zip(resolved.frame_ids, resolved.weights)):
        a[b, list(ids)] = weights
    for j in range(4):
        raw_variance = (
            resolved.raw_instrument_variance[:, j]
            + resolved.raw_representation_variance[:, j]
        )
        expected = a @ np.diag(raw_variance) @ a.T
        np.testing.assert_allclose(resolved.variance[:, j], np.diag(expected))
    np.testing.assert_allclose(resolved.covariance_diag, resolved.variance.ravel())


def test_component_sensor_and_height_labels_follow_operator_order() -> None:
    raw = _observations([0.0, 1.0])
    spec = ObservationErrorSpec(
        instrument_std={
            "default": 1.0,
            "height_bands": [{"min_z": 2.0, "max_z": 4.0, "std": 2.0}],
            "components": {"v": 3.0},
            "sensors": {1: 4.0},
        }
    )
    resolved = spec.resolve(raw, _operator())
    assert resolved.components == ("u", "u", "v", "v")
    assert resolved.sensor_indices == (0, 1, 0, 1)
    np.testing.assert_allclose(resolved.raw_instrument_std[0], [1, 4, 3, 4])
    assert not resolved.variance.flags.writeable


def test_new_window_recomputes_bin_counts_even_for_equal_output_length() -> None:
    aggregate = AggregateObservations(4.0)
    spec = ObservationErrorSpec(instrument_std=2.0)
    first = spec.resolve(_observations([0.0, 2.0, 4.0, 6.0]), _operator(), aggregate)
    second = spec.resolve(_observations([10.0, 14.0, 15.0]), _operator(), aggregate)
    np.testing.assert_allclose(first.variance[:, 0], [2.0, 2.0])
    np.testing.assert_allclose(second.variance[:, 0], [4.0, 2.0])
    second_obs = aggregate(
        _observations([10.0, 14.0, 15.0]), allow_interval_count_change=True
    )
    np.testing.assert_allclose(second_obs.time, second.times)


def test_strided_frames_keep_time_major_covariance_alignment() -> None:
    raw = _observations([0.0, 2.0, 4.0, 6.0, 8.0]).isel(time=[0, 2, 4])
    spec = ObservationErrorSpec(
        instrument_std={"default": 1.0, "components": {"v": 2.0}}
    )
    resolved = spec.resolve(raw, _operator(), AggregateObservations(6.0))
    assert resolved.frame_ids == ((0, 1), (2,))
    np.testing.assert_allclose(resolved.raw_times, [0.0, 4.0, 8.0])
    np.testing.assert_allclose(
        resolved.covariance_diag, [0.5, 0.5, 2.0, 2.0, 1.0, 1.0, 4.0, 4.0]
    )


@pytest.mark.parametrize("mode", ["median", "min", "max"])  # type: ignore[misc, unused-ignore]
def test_nonlinear_aggregation_rejected_only_in_corrected_mode(mode: str) -> None:
    aggregate = AggregateObservations(4.0, mode=mode)
    raw = _observations([0.0, 1.0])
    aggregate(raw)  # legacy path remains supported
    with pytest.raises(ValueError, match="only mean aggregation"):
        ObservationErrorSpec(1.0).resolve(raw, _operator(), aggregate)


def test_invalid_labels_and_variance_rejected() -> None:
    raw = _observations([0.0])
    with pytest.raises(ValueError, match="unknown component"):
        ObservationErrorSpec({"default": 1.0, "components": {"w": 2.0}}).resolve(
            raw, _operator()
        )
    with pytest.raises(ValueError, match="positive"):
        ObservationErrorSpec(0.0).resolve(raw, _operator())
    with pytest.raises(ValueError, match="'independent' or 'persistent'"):
        ObservationErrorSpec(1.0, representation_time_model="other").resolve(
            raw, _operator()
        )


def test_spec_copies_nested_overrides_and_validates_masked_values() -> None:
    config: dict[str, Any] = {"default": 1.0, "components": {"u": 2.0}}
    spec = ObservationErrorSpec(config)
    config["components"]["u"] = 9.0
    assert (
        spec.resolve(_observations([0.0]), _operator()).raw_instrument_std[0, 0] == 2.0
    )
    with pytest.raises(ValueError, match="default"):
        ObservationErrorSpec({"default": -1.0, "sensors": {0: 1.0, 1: 1.0}}).resolve(
            _observations([0.0]), _operator()
        )
    with pytest.raises(ValueError, match="overlap"):
        ObservationErrorSpec(
            {
                "default": 1.0,
                "height_bands": [
                    {"min_z": 0.0, "max_z": 2.0, "std": 1.0},
                    {"min_z": 1.0, "max_z": 3.0, "std": 2.0},
                ],
            }
        ).resolve(_observations([0.0]), _operator())


def test_bad_obs_labels_and_nonfinite_data_rejected() -> None:
    reordered = _observations([0.0]).assign_coords(obs=[1, 0, 2, 3])
    with pytest.raises(ValueError, match="coordinate labels"):
        ObservationErrorSpec(1.0).resolve(reordered, _operator())
    missing = _observations([0.0, 1.0])
    missing.values[0, 0] = np.nan
    corrected_aggregate = AggregateObservations(2.0, allow_interval_count_change=True)
    with pytest.raises(ValueError, match="skipping NaNs"):
        corrected_aggregate(missing)
    with pytest.raises(ValueError, match="finite"):
        ObservationErrorSpec(1.0).resolve(missing, _operator())


def test_empty_sensor_labels_rejected() -> None:
    empty_operator = ObservationOperator(obs_states=["u"], obs_x=[], obs_y=[], obs_z=[])
    raw = xr.DataArray(np.empty((1, 0)), dims=("time", "obs"), coords={"time": [0.0]})
    with pytest.raises(ValueError, match="non-empty"):
        ObservationErrorSpec(1.0).resolve(raw, empty_operator)


def test_smoother_observation_path_handles_changed_window_bin_counts() -> None:
    aggregate = AggregateObservations(4.0, allow_interval_count_change=True)
    smoother = ParameterESMDA.__new__(ParameterESMDA)
    smoother.aggregate_observations = aggregate
    spec = ObservationErrorSpec(2.0)
    first = _observations([0.0, 2.0, 4.0, 6.0])
    second = _observations([10.0, 14.0, 15.0, 18.0])
    first.values[:] = np.arange(4)[:, None]
    second.values[:] = np.arange(4)[:, None]

    first_product = np.asarray(smoother._get_observations(first))
    second_product = np.asarray(smoother._get_observations(second))
    first_error = spec.resolve(first, _operator(), aggregate)
    second_error = spec.resolve(second, _operator(), aggregate)
    np.testing.assert_allclose(first_product.reshape(2, 4)[:, 0], [0.5, 2.5])
    np.testing.assert_allclose(second_product.reshape(3, 4)[:, 0], [0.0, 1.5, 3.0])
    np.testing.assert_allclose(first_error.variance[:, 0], [2.0, 2.0])
    np.testing.assert_allclose(second_error.variance[:, 0], [4.0, 2.0, 4.0])


def test_none_preserves_configured_errors_across_unequal_mean_bins() -> None:
    raw = _observations([0.0, 1.0, 2.0, 4.0, 5.0, 8.0])
    raw.values[:] = np.arange(6)[:, None]
    aggregate = AggregateObservations(4.0)
    settings: dict[str, Any] = dict(
        instrument_std={"default": 2.0, "components": {"v": 3.0}},
        representation_std={"default": 1.0, "sensors": {1: 2.0}},
    )
    unchanged = ObservationErrorSpec(**settings, aggregation="none").resolve(
        raw, _operator(), aggregate
    )
    propagated = ObservationErrorSpec(**settings).resolve(raw, _operator(), aggregate)
    np.testing.assert_allclose(unchanged.instrument_variance, [[4, 4, 9, 9]] * 3)
    np.testing.assert_allclose(unchanged.representation_variance, [[1, 4, 1, 4]] * 3)
    np.testing.assert_allclose(unchanged.variance, [[5, 8, 10, 13]] * 3)
    np.testing.assert_allclose(
        propagated.variance, unchanged.variance / np.array([3, 2, 1])[:, None]
    )
    np.testing.assert_array_equal(
        unchanged.raw_instrument_std, propagated.raw_instrument_std
    )
    assert unchanged.frame_ids == propagated.frame_ids
    assert unchanged.weights == propagated.weights
    assert unchanged.provenance.endswith(":none")
    smoother = ParameterESMDA.__new__(ParameterESMDA)
    smoother.aggregate_observations = aggregate
    np.testing.assert_allclose(
        np.asarray(smoother._get_observations(raw)).reshape(3, 4)[:, 0], [1, 3.5, 5]
    )
    for mode in ("none", "propagate_mean"):
        frame = ObservationErrorSpec(**settings, aggregation=mode).resolve(
            raw, _operator()
        )
        np.testing.assert_allclose(frame.variance, [[5, 8, 10, 13]] * 6)


def test_persistent_representation_error_does_not_average_down() -> None:
    raw = _observations([float(t) for t in range(20)])
    aggregate = AggregateObservations(20.0)
    settings: dict[str, Any] = dict(instrument_std=2.0, representation_std=0.5)
    independent = ObservationErrorSpec(**settings).resolve(raw, _operator(), aggregate)
    persistent = ObservationErrorSpec(
        **settings, representation_time_model="persistent"
    ).resolve(raw, _operator(), aggregate)
    assert persistent.frame_ids == ((tuple(range(20))),)
    np.testing.assert_allclose(independent.representation_variance, 0.25 / 20)
    np.testing.assert_allclose(persistent.representation_variance, 0.25)
    np.testing.assert_array_equal(
        independent.instrument_variance, persistent.instrument_variance
    )
    np.testing.assert_allclose(persistent.variance, 4.0 / 20 + 0.25)
    assert independent.provenance.endswith(":diagonal:independent:propagate_mean")
    assert persistent.provenance.endswith(":diagonal:persistent:propagate_mean")
    # Without aggregation every bin is one frame and the two models coincide.
    frames = [
        ObservationErrorSpec(**settings, representation_time_model=model).resolve(
            raw, _operator()
        )
        for model in ("independent", "persistent")
    ]
    np.testing.assert_array_equal(frames[0].variance, frames[1].variance)


@pytest.mark.parametrize("mode", [None, "invalid"])  # type: ignore[misc, unused-ignore]
def test_invalid_error_aggregation_rejected(mode: Any) -> None:
    with pytest.raises(ValueError, match="aggregation must be"):
        ObservationErrorSpec(1.0, aggregation=mode).resolve(
            _observations([0.0]), _operator()
        )
