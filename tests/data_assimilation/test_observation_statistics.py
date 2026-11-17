"""Turbulence-statistics observations: definitions, layout, error and an ESMDA toy."""

from typing import Any, cast

import jax
import numpy as np
import pytest
import xarray as xr
from data_assimilation.adaptive_observation_error import (
    observation_heights,
    observation_rows,
)
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    TemporalObservationOperator,
    flatten_observations,
    sensor_observation_coords,
)
from data_assimilation.observation_statistics import ObservationStatistics
from data_assimilation.smoothing.esmda import ParameterESMDA
from scipy import signal

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel
from pyurbanair.base_forward_model import BaseForwardModel

_NUM_SENSORS = 3


def _operator() -> ObservationOperator:
    return ObservationOperator(
        obs_states=["u", "v"],
        obs_x=[0.0, 1.0, 2.0],
        obs_y=[0.0, 0.0, 0.0],
        obs_z=[1.0, 2.0, 4.0],
    )


def _series(
    num_frames: int, dt: float = 0.5, seed: int = 0, std: Any = 1.0
) -> xr.DataArray:
    """White noise of ``std`` (scalar or per obs) about a per-obs mean."""
    rng = np.random.default_rng(seed)
    mean = np.array([3.0, 4.0, 5.0, 0.5, -0.5, 1.0])
    values = mean + np.asarray(std) * rng.normal(size=(num_frames, 6))
    return xr.DataArray(
        values, dims=("time", "obs"), coords={"time": 10.0 + dt * np.arange(num_frames)}
    )


def _all(**kwargs: Any) -> ObservationStatistics:
    return ObservationStatistics(
        ["spectral", "bins", "tke"],
        AggregateObservations(25.0),
        num_sensors=_NUM_SENSORS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# raw and bins reproduce the existing paths exactly
# ---------------------------------------------------------------------------


def test_bins_only_reproduces_aggregation_and_error_exactly() -> None:
    raw = _series(200)
    spec = ObservationErrorSpec(instrument_std=0.3, representation_std=0.2)
    stats = ObservationStatistics(["bins"], AggregateObservations(25.0))
    expected = AggregateObservations(25.0)(raw)

    out = stats(raw)
    np.testing.assert_array_equal(out.values, expected.values)
    np.testing.assert_array_equal(out.time.values, expected.time.values)
    np.testing.assert_array_equal(
        flatten_observations(out), flatten_observations(expected)
    )
    assert set(out.block.values) == {"bins"}

    resolved = stats.resolve_error(raw, _operator(), spec)
    reference = spec.resolve(raw, _operator(), AggregateObservations(25.0))
    np.testing.assert_array_equal(resolved.covariance_diag, reference.covariance_diag)
    np.testing.assert_array_equal(resolved.std, reference.std)
    assert resolved.provenance == reference.provenance


def test_raw_only_reproduces_the_unaggregated_path_exactly() -> None:
    raw = _series(40)
    spec = ObservationErrorSpec(instrument_std={"default": 0.3, "sensors": {1: 0.5}})
    stats = ObservationStatistics(["raw"])

    out = stats(raw)
    np.testing.assert_array_equal(flatten_observations(out), flatten_observations(raw))
    np.testing.assert_array_equal(out.time.values, raw.time.values)
    resolved = stats.resolve_error(raw, _operator(), spec)
    reference = spec.resolve(raw, _operator(), None)
    np.testing.assert_array_equal(resolved.covariance_diag, reference.covariance_diag)
    assert resolved.block_slices == {"raw": slice(0, raw.size)}


def test_raw_with_statistics_is_stacked_in_order() -> None:
    raw = _series(200)
    stats = ObservationStatistics(["tke", "raw"], num_sensors=_NUM_SENSORS)
    out = stats(raw)
    assert list(out.block.values) == ["raw"] * 200 + ["tke"]
    np.testing.assert_array_equal(out.isel(time=slice(0, 200)).values, raw.values)


# ---------------------------------------------------------------------------
# Statistic definitions
# ---------------------------------------------------------------------------


def test_tke_is_half_the_population_variance() -> None:
    """Without bins: half the variance about the window mean, normalized by
    default."""
    raw = _series(500, std=np.array([1.0, 2.0, 0.5, 1.0, 1.5, 3.0]))
    plain = ObservationStatistics(
        ["tke"], tke={"normalize": False, "log": False, "pool_std": "none"}
    )(raw)
    np.testing.assert_allclose(plain.values[0], 0.5 * raw.values.var(axis=0))

    logged = ObservationStatistics(["tke"], tke={"normalize": False})(raw)
    np.testing.assert_allclose(logged.values[0], np.log(plain.values[0]))

    default = ObservationStatistics(["tke"], num_sensors=_NUM_SENSORS)
    assert default.tke["normalize"] and default.tke["log"]
    normalized = ObservationStatistics(
        ["tke"], tke={"log": False}, num_sensors=_NUM_SENSORS
    )(raw)
    mean = raw.values.mean(axis=0).reshape(2, _NUM_SENSORS)
    speed2 = np.tile((mean**2).sum(axis=0), 2)
    np.testing.assert_allclose(normalized.values[0], plain.values[0] / (0.5 * speed2))
    np.testing.assert_allclose(default(raw).values[0], np.log(normalized.values[0]))


def _tke_row(out: xr.DataArray) -> np.ndarray:
    return np.asarray(out.isel(time=np.flatnonzero(out.block == "tke")).values)


def test_tke_with_bins_leaves_out_the_variation_between_bins() -> None:
    """With bins the tke is taken about each bin's mean: a slow per-bin
    offset (what the bins observe) leaves it unchanged, whatever the
    aggregation mode; without bins it inflates the variance."""
    raw = _red_series(480, seed=7)  # 240 s: twelve 20 s bins of 40 frames
    rng = np.random.default_rng(8)
    offsets = rng.normal(0.0, 2.0, size=(12, 6))
    offsets -= offsets.mean(axis=0)  # the window mean, so |U|, stays
    shifted = raw + np.repeat(offsets, 40, axis=0)

    with_bins = ObservationStatistics(
        ["bins", "tke"], AggregateObservations(20.0), num_sensors=_NUM_SENSORS
    )
    reference = _tke_row(with_bins(raw))
    np.testing.assert_allclose(_tke_row(with_bins(shifted)), reference, atol=1e-10)
    median = ObservationStatistics(
        ["bins", "tke"],
        AggregateObservations(20.0, mode="median"),
        num_sensors=_NUM_SENSORS,
    )
    np.testing.assert_allclose(_tke_row(median(shifted)), reference, atol=1e-10)
    # Half the mean of the bins' own (population) variances, normalized.
    plain = ObservationStatistics(
        ["bins", "tke"], AggregateObservations(20.0), tke={"normalize": False}
    )
    within = raw.values.reshape(12, 40, 6).var(axis=1).mean(axis=0)
    np.testing.assert_allclose(_tke_row(plain(shifted))[0], np.log(0.5 * within))

    without_bins = ObservationStatistics(["tke"], num_sensors=_NUM_SENSORS)
    change = _tke_row(without_bins(shifted)) - _tke_row(without_bins(raw))
    assert np.all(change > 0.3)


def test_predicted_tke_noise_about_bin_means_is_exact() -> None:
    """Removing the mean of n frames removes 1/n of their white-noise
    variance: over many windows the noisy observed tke about 4-frame bin
    means equals the clean prediction with (1 - 1/4) sigma**2 added, not
    sigma**2."""
    noise = np.array([0.5, 0.5, 1.0, 0.2, 0.2, 2.0])
    clean = xr.concat([_red_series(400, seed) for seed in range(50)], "ensemble")
    rng = np.random.default_rng(9)
    noisy = clean + np.sqrt(noise) * rng.normal(size=clean.shape)
    stats = ObservationStatistics(
        ["bins", "tke"],
        AggregateObservations(2.0),
        tke={"normalize": False, "log": False, "pool_std": "none"},
        noise_variance=noise,
    )
    observed = _tke_row(stats(noisy, observed=True)).mean(axis=0)
    predicted = _tke_row(stats(clean)).mean(axis=0)
    np.testing.assert_allclose(predicted, observed, rtol=0.03)
    stats.noise_variance = None
    naive = _tke_row(stats(clean)).mean(axis=0) + 0.5 * noise
    assert np.max(np.abs(naive / observed - 1.0)) > 0.15


def _red_series(num_frames: int, seed: int) -> xr.DataArray:
    """AR(1) series (a red spectrum, unlike the white instrument noise)."""
    rng = np.random.default_rng(seed)
    values = signal.lfilter(
        [1.0], [1.0, -0.9], rng.normal(size=(num_frames, 6)), axis=0
    )
    mean = np.array([3.0, 4.0, 5.0, 0.5, -0.5, 1.0])
    return xr.DataArray(
        mean + values,
        dims=("time", "obs"),
        coords={"time": 0.5 * np.arange(num_frames)},
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "types, options",
    [
        (["tke"], dict(tke={"normalize": False, "log": False, "pool_std": "none"})),
        (["tke"], dict(num_sensors=_NUM_SENSORS)),
        (["spectral"], dict(spectral={"segment_seconds": 32.0})),
        (["spectral"], dict(spectral={"segment_seconds": 32.0, "normalize": False})),
    ],
)
def test_predictions_carry_the_instrument_noise_of_the_observations(
    types: list[str], options: dict
) -> None:
    clean = _red_series(40000, seed=1)
    noise = np.array([0.5, 0.5, 1.0, 0.2, 0.2, 2.0])
    rng = np.random.default_rng(2)
    noisy = clean + np.sqrt(noise) * rng.normal(size=clean.shape)
    stats = ObservationStatistics(types, noise_variance=noise, **options)
    observed = stats(noisy, observed=True).values
    predicted = stats(clean).values
    np.testing.assert_allclose(predicted, observed, rtol=0.05, atol=0.05)
    # Without the noise the predictions miss the bias (here: by a lot).
    stats.noise_variance = None
    assert np.abs(stats(clean).values - observed).max() > 0.2
    # A zero noise variance is no noise.
    stats.noise_variance = 0.0
    np.testing.assert_array_equal(stats(clean).values, stats(clean, observed=True))


def _spectral(**options: Any) -> ObservationStatistics:
    return ObservationStatistics(["spectral"], spectral={"pool_std": "none", **options})


def test_a_sinusoid_dominates_its_band() -> None:
    dt, frames = 0.5, 2048
    stats = _spectral(normalize=False, log=False, num_bands=4)
    times = dt * np.arange(frames)
    rng = np.random.default_rng(3)
    out = stats(
        xr.DataArray(
            rng.normal(size=(frames, 6)), dims=("time", "obs"), coords={"time": times}
        )
    )
    centres = out.band_frequency.values
    # A sine at the centre of band 2, plus a little noise.
    values = np.sin(2 * np.pi * centres[2] * times)[:, None] + 0.01 * rng.normal(
        size=(frames, 6)
    )
    series = xr.DataArray(values, dims=("time", "obs"), coords={"time": times})
    bands = stats(series).values
    assert np.all(np.argmax(bands, axis=0) == 2)
    assert np.all(bands[2] > 100 * np.delete(bands, 2, axis=0).max(axis=0))


def test_normalized_bands_ignore_the_amplitude() -> None:
    raw = _series(800)
    stats = _spectral()
    scaled = (raw - raw.mean("time")) * 7.0 + raw.mean("time")
    np.testing.assert_allclose(stats(scaled).values, stats(raw).values, atol=1e-10)


def test_spectral_tolerates_small_time_jitter_but_not_gaps() -> None:
    raw = _series(400)
    rng = np.random.default_rng(4)
    jittered = raw.assign_coords(time=raw.time + rng.uniform(-0.01, 0.01, 400))
    assert _spectral()(jittered).sizes["time"] == 4
    gap = raw.time.values.copy()
    gap[200:] += 2.0
    with pytest.raises(ValueError, match="near-uniform"):
        _spectral()(raw.assign_coords(time=gap))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_layout_blocks_and_sensor_innermost() -> None:
    std = np.array([1.0, 2.0, 4.0, 1.0, 2.0, 4.0])  # by sensor, both components
    raw = _series(400, std=std)
    stats = _all(
        tke={"normalize": False, "log": False, "pool_std": "none"},
        spectral={"normalize": False, "log": False, "pool_std": "none"},
    )
    out = stats(raw)
    blocks = list(out.block.values)
    assert blocks == ["bins"] * 8 + ["tke"] + ["spectral"] * 4
    assert np.isnan(out.band_frequency.values[:9]).all()
    assert np.all(np.diff(out.band_frequency.values[9:]) > 0)

    flat = flatten_observations(out)
    slices = stats.block_slices(raw)
    assert flat.size == 13 * 6 == slices["spectral"].stop
    assert slices == {
        "bins": slice(0, 48),
        "tke": slice(48, 54),
        "spectral": slice(54, 78),
    }
    # Observation j lives at sensor j % num_sensors: the variance grows by sensor.
    tke = flat[slices["tke"]]
    sensors = np.arange(tke.size) % _NUM_SENSORS
    np.testing.assert_allclose(tke, 0.5 * std[sensors] ** 2, rtol=0.2)
    spectral = flat[slices["spectral"]].reshape(4, 6)
    sensors = np.arange(6) % _NUM_SENSORS
    assert np.all(np.diff(spectral.sum(axis=0)[sensors.argsort()][::2]) > 0)


def test_ensemble_rows_equal_single_member_calls() -> None:
    members = [_series(300, seed=s) for s in range(3)]
    ensemble = xr.concat(members, dim="ensemble")
    stats = _all()
    out = stats(ensemble)
    assert out.dims == ("ensemble", "time", "obs")
    flat = flatten_observations(out)
    for i, member in enumerate(members):
        np.testing.assert_allclose(flat[i], flatten_observations(stats(member)))


def test_interval_count_change_is_forwarded() -> None:
    aggregation = AggregateObservations(25.0)
    stats = ObservationStatistics(
        ["bins", "tke"], aggregation, num_sensors=_NUM_SENSORS
    )
    stats(_series(200))
    with pytest.raises(ValueError, match="Interval count changed"):
        stats(_series(100))
    stats.allow_interval_count_change = True
    assert aggregation.allow_interval_count_change
    assert stats(_series(100)).sizes["time"] == 3  # two bins and the tke row


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


def test_auto_std_shrinks_with_the_window_and_floors_apply() -> None:
    spec = ObservationErrorSpec(0.1)
    # A fixed Welch segment, so a longer window averages more segments.
    stats = ObservationStatistics(
        ["tke", "spectral"],
        tke={"std_floor": 0.0},
        spectral={"std_floor": 0.0, "segment_seconds": 50.0},
        num_sensors=_NUM_SENSORS,
    )
    ratios = []
    for seed in range(12):
        short = stats.resolve_error(_series(1000, seed=seed), _operator(), spec)
        long = stats.resolve_error(_series(4000, seed=seed), _operator(), spec)
        ratios.append(
            [
                np.mean(short.std[short.block_slices[block]])
                / np.mean(long.std[long.block_slices[block]])
                for block in ("tke", "spectral")
            ]
        )
    # White noise: 4x the window halves the sampling error.
    assert np.all((1.6 < np.mean(ratios, axis=0)) & (np.mean(ratios, axis=0) < 2.5))

    floored = ObservationStatistics(
        ["tke", "spectral"],
        tke={"std_floor": 5.0},
        spectral={"std": 0.3},
        num_sensors=_NUM_SENSORS,
    ).resolve_error(_series(1000), _operator(), spec)
    np.testing.assert_allclose(floored.std[floored.block_slices["tke"]], 5.0)
    np.testing.assert_allclose(floored.std[floored.block_slices["spectral"]], 0.3)
    assert "tke(std=auto" in floored.provenance
    assert floored.covariance_diag.size == 6 * (1 + 4)  # tke + 4 bands


def _pooled_operator() -> ObservationOperator:
    """Sensors 0 and 1 share a height, sensor 2 is alone."""
    return ObservationOperator(
        obs_states=["u", "v"],
        obs_x=[0.0, 1.0, 2.0],
        obs_y=[0.0, 0.0, 0.0],
        obs_z=[2.0, 2.0, 4.0],
    )


def test_auto_std_is_pooled_per_height() -> None:
    raw = _series(1000, std=np.array([1.0, 3.0, 2.0, 1.0, 1.0, 1.0]))
    spec = ObservationErrorSpec(0.1)
    options = {"std_floor": 0.0}
    pooled = ObservationStatistics(
        ["tke"], tke=options, num_sensors=_NUM_SENSORS
    ).resolve_error(raw, _pooled_operator(), spec)
    alone = ObservationStatistics(
        ["tke"], tke={**options, "pool_std": "none"}, num_sensors=_NUM_SENSORS
    ).resolve_error(raw, _pooled_operator(), spec)
    # Median over the two sensors at z=2 (of two: their mean), per component.
    np.testing.assert_allclose(pooled.std[[0, 1]], alone.std[[0, 1]].mean())
    np.testing.assert_allclose(pooled.std[[3, 4]], alone.std[[3, 4]].mean())
    np.testing.assert_allclose(pooled.std[[2, 5]], alone.std[[2, 5]])
    assert alone.std[0] != alone.std[1]


def test_representation_std_adds_in_quadrature() -> None:
    raw = _series(1000)
    spec = ObservationErrorSpec(0.1)
    fixed = ObservationStatistics(
        ["tke"],
        tke={"std": 0.3, "representation_std": 0.4},
        num_sensors=_NUM_SENSORS,
    ).resolve_error(raw, _operator(), spec)
    np.testing.assert_allclose(fixed.std, 0.5)
    sampling = ObservationStatistics(["spectral"], spectral={"std_floor": 0.0})
    auto = sampling.resolve_error(raw, _operator(), spec).std
    sampling.spectral["representation_std"] = 0.2
    np.testing.assert_allclose(
        sampling.resolve_error(raw, _operator(), spec).std,
        np.sqrt(auto**2 + 0.2**2),
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "types, interval, tke",
    [
        (["tke", "spectral"], None, {}),
        (["bins", "tke"], 20.0, {}),
        (["bins", "tke"], 20.0, {"normalize": False}),
        (["bins", "tke"], 5.0, {"normalize": False}),
    ],
)
def test_auto_std_matches_the_monte_carlo_sampling_error(
    types: list[str], interval: float | None, tke: dict
) -> None:
    """Default tke and spectral rows of a red (AR(1), correlation time ~5 s)
    series plus white noise: the median auto std is sqrt(2) (observed plus
    predicted sampling error) times the spread over independent windows.
    Also for the tke about bin means of 4 and 1 correlation times (without
    normalize: the residual variance's own error, not the mean speed's)."""
    stats = ObservationStatistics(
        types,
        None if interval is None else AggregateObservations(interval),
        tke={"std_floor": 0.0, **tke},
        spectral={"std_floor": 0.0},
        num_sensors=_NUM_SENSORS,
    )
    spec = ObservationErrorSpec(0.1)
    windows = [
        _red_series(240, seed) + 0.5 * np.random.default_rng(seed).normal(size=(240, 6))
        for seed in range(400)
    ]
    rows = slice(stats.block_slices(windows[0])["tke"].start, None)
    values = np.stack(
        [flatten_observations(stats(w, observed=True))[rows] for w in windows]
    )
    auto = np.stack(
        [stats.resolve_error(w, _operator(), spec).std[rows] for w in windows[:100]]
    )
    ratio = np.median(auto, axis=0) / (np.sqrt(2.0) * values.std(axis=0))
    assert np.all((0.8 < ratio) & (ratio < 1.25)), ratio


def test_predicted_white_noise_level_is_exact_in_every_band() -> None:
    """Pure white noise observed vs. a constant series predicted with that
    noise: the Welch level after the segment-mean removal, also in the lowest
    band (5/6 of 2 sigma**2 dt at the first bin for a Hann window)."""
    noise = 0.49
    stats = _spectral(normalize=False, log=False, segment_seconds=16.0)
    raw = _series(200000, std=np.sqrt(noise))
    observed = stats(raw, observed=True).values
    stats.noise_variance = noise
    predicted = stats(xr.zeros_like(raw)).values
    np.testing.assert_allclose(predicted, observed, rtol=0.03)


def test_predictions_reuse_the_observed_bands() -> None:
    stats = _spectral()
    observed = stats(_series(400), observed=True)
    predicted = stats(_series(400, seed=1))
    np.testing.assert_array_equal(
        predicted.band_frequency.values, observed.band_frequency.values
    )
    with pytest.raises(ValueError, match="do not match the observed"):
        stats(_series(399))
    with pytest.raises(ValueError, match="do not match the observed"):
        stats(_series(400, dt=0.6))
    # A new observed window resets the bands.
    stats(_series(399), observed=True)
    assert stats(_series(399)).sizes["time"] == 4


def test_a_constant_predicted_series_is_rejected() -> None:
    """A member with a dead sensor (zeros, no noise) has no tke or spectrum:
    raise rather than log a floored zero."""
    raw = _series(400)
    raw[:, 4] = 0.0
    for types in (["tke"], ["spectral"]):
        stats = ObservationStatistics(types, num_sensors=_NUM_SENSORS)
        with pytest.raises(ValueError, match="constant or non-finite"):
            stats(raw)


def test_a_constant_observed_series_is_rejected() -> None:
    raw = _series(400)
    raw[:, 4] = 2.0
    for types in (["tke"], ["spectral"]):
        with pytest.raises(ValueError, match="obs 4 \\(sensor 1\\)"):
            ObservationStatistics(types, num_sensors=_NUM_SENSORS).resolve_error(
                raw, _operator(), ObservationErrorSpec(0.1)
            )


@pytest.mark.parametrize(  # type: ignore[misc]
    "kwargs, match",
    [
        (dict(types=[]), "at least one"),
        (dict(types=["bins", "variance"]), "Unknown statistic"),
        (dict(types=["all"]), "Unknown statistic"),
        (dict(types=["tke", "tke"]), "Duplicate"),
        (dict(types="tke"), "list"),
        (dict(types=["bins"]), "requires an aggregation"),
        (
            dict(types=["raw", "bins"], aggregation=AggregateObservations(1.0)),
            "counts the data twice",
        ),
        (dict(types=["tke"]), "num_sensors"),  # normalize is the default
        (dict(types=["tke"], tke={"window": 3}), "Unknown tke options"),
        (dict(types=["spectral"], spectral={"num_batches": 3}), "Unknown spectral"),
        (
            dict(types=["tke"], tke={"log": False, "normalize": False}),
            "needs log or normalize",
        ),
        (dict(types=["tke"], tke={"pool_std": "sensor"}), "pool_std"),
        (dict(types=["tke"], tke={"representation_std": -1.0}), "non-negative"),
        (dict(types=["stress"]), "stress requires num_sensors"),
        (
            dict(types=["stress"], num_sensors=3, stress={"log": True}),
            "Unknown stress options",
        ),
        (
            dict(types=["stress"], num_sensors=3, stress={"normalize": False}),
            "stress.pool_std 'height' needs normalize:",
        ),
        (dict(types=["stress"], num_sensors=3, stress={"std": -0.1}), "positive"),
    ],
)
def test_invalid_construction_is_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ObservationStatistics(**kwargs)


def test_short_windows_and_empty_bands_are_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        _spectral()(_series(12))
    with pytest.raises(ValueError, match="no Welch frequency"):
        _spectral(num_bands=20)(_series(64))
    one_segment = _spectral(segment_seconds=100.0)
    one_segment(_series(200))  # a single Welch segment is a statistic...
    with pytest.raises(ValueError, match="at least two Welch segments"):
        one_segment.resolve_error(_series(200), _operator(), ObservationErrorSpec(0.1))


# ---------------------------------------------------------------------------
# stress: the covariances between components
# ---------------------------------------------------------------------------

#: Component correlation of the innovations of `_correlated_series`.
_CORRELATION = np.array([[1.0, -0.4, 0.2], [-0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])


def _correlated_series(
    num_frames: int, seed: int, num_components: int, phi: float = 0.9
) -> xr.DataArray:
    """AR(1) series per component and sensor whose components are
    cross-correlated (`_CORRELATION`), about a per-obs mean; `(component,
    sensor)` layout, 0.5 s frames."""
    rng = np.random.default_rng(seed)
    cholesky = np.linalg.cholesky(_CORRELATION[:num_components, :num_components])
    shocks = rng.normal(size=(num_frames, _NUM_SENSORS, num_components)) @ cholesky.T
    shocks = shocks.transpose(0, 2, 1).reshape(num_frames, -1)
    values = signal.lfilter([1.0], [1.0, -phi], shocks, axis=0)
    mean = np.tile([3.0, 4.0, 5.0], num_components)
    return xr.DataArray(
        mean + values,
        dims=("time", "obs"),
        coords={"time": 0.5 * np.arange(num_frames)},
    )


def _operator_for(num_components: int) -> ObservationOperator:
    return ObservationOperator(
        obs_states=["u", "v", "w"][:num_components],
        obs_x=[0.0, 1.0, 2.0],
        obs_y=[0.0, 0.0, 0.0],
        obs_z=[1.0, 2.0, 4.0],
    )


def _stress_values(stats: ObservationStatistics, raw: xr.DataArray) -> np.ndarray:
    """The flattened stress entries of the statistics of one raw series."""
    flat = flatten_observations(stats(raw))
    return np.asarray(flat)[stats.block_slices(raw)["stress"]]


def _direct_stress(
    values: np.ndarray, num_components: int, bin_frames: int | None, normalize: bool
) -> np.ndarray:
    """The stress row by hand: per pair (0,1), (0,2), (1,2) and sensor."""
    if bin_frames is None:
        residual = values - values.mean(axis=0)
    else:
        binned = values.reshape(-1, bin_frames, values.shape[-1])
        residual = (binned - binned.mean(axis=1, keepdims=True)).reshape(values.shape)
    per_component = residual.reshape(-1, num_components, _NUM_SENSORS)
    entries = []
    for a, b in [(0, 1), (0, 2), (1, 2)][: 1 if num_components == 2 else 3]:
        x, y = per_component[:, a], per_component[:, b]
        stress = np.mean(x * y, axis=0)
        if normalize:
            stress = stress / np.sqrt(np.mean(x**2, axis=0) * np.mean(y**2, axis=0))
        entries.append(stress)
    return np.concatenate(entries)


@pytest.mark.parametrize("num_components", [2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("normalize", [True, False])  # type: ignore[misc]
@pytest.mark.parametrize("interval", [None, 20.0])  # type: ignore[misc]
def test_stress_equals_the_direct_computation(
    num_components: int, normalize: bool, interval: float | None
) -> None:
    """The covariance (or correlation coefficient) of the tke fluctuations,
    about the window mean or each bin's; with bins a per-bin offset leaves it
    unchanged."""
    raw = _correlated_series(480, seed=3, num_components=num_components)
    types = ["stress"] if interval is None else ["bins", "stress"]
    stats = ObservationStatistics(
        types,
        None if interval is None else AggregateObservations(interval),
        stress={"normalize": normalize, "pool_std": "none"},
        num_sensors=_NUM_SENSORS,
    )
    out = stats(raw)
    bin_frames = None if interval is None else 40
    expected = _direct_stress(raw.values, num_components, bin_frames, normalize)
    np.testing.assert_allclose(_stress_values(stats, raw), expected, rtol=1e-10)
    if normalize:
        assert np.all(np.abs(expected) < 1.0)
    # The two-component row fills the first num_sensors slots only.
    if num_components == 2:
        used = out.slot_used.values
        assert used.sum() == used.size - _NUM_SENSORS
        assert not used[-1, _NUM_SENSORS:].any() and used[-1, :_NUM_SENSORS].all()
    else:
        assert "slot_used" not in out.coords
    if interval is not None:
        rng = np.random.default_rng(4)
        offsets = rng.normal(0.0, 2.0, size=(12, raw.sizes["obs"]))
        shifted = raw + np.repeat(offsets, 40, axis=0)
        np.testing.assert_allclose(_stress_values(stats, shifted), expected, atol=1e-10)


def test_without_stress_no_slot_coordinate_is_added() -> None:
    """Runs without stress keep the exact output they had."""
    out = _all()(_series(400))
    assert "slot_used" not in out.coords


@pytest.mark.parametrize("num_components", [2, 3])  # type: ignore[misc]
def test_stress_layout_sizes_and_sensor_innermost(num_components: int) -> None:
    """[bins, tke, stress, spectral]: the flat vector, block_slices,
    block_rows, the error vector and the adaptive rows agree, and entry j
    belongs to sensor j % num_sensors (as the localization tiles it)."""
    num_obs = num_components * _NUM_SENSORS
    num_pairs = 1 if num_components == 2 else 3
    raw = _correlated_series(400, seed=5, num_components=num_components)
    # Sensor s's components correlate more with s: entry j's correlation
    # coefficient then tells its sensor.
    weight = np.tile([0.0, 2.0, 6.0], num_components)
    common = np.random.default_rng(6).normal(size=(400, 1))
    raw = raw + weight * common
    stats = ObservationStatistics(
        ["spectral", "stress", "bins", "tke"],
        AggregateObservations(25.0),
        num_sensors=_NUM_SENSORS,
    )
    operator = _operator_for(num_components)
    out = stats(raw)
    assert list(dict.fromkeys(out.block.values)) == [
        "bins",
        "tke",
        "stress",
        "spectral",
    ]
    flat = flatten_observations(out)
    slices = stats.block_slices(raw)
    stress_size = num_pairs * _NUM_SENSORS
    assert slices["stress"] == slice(9 * num_obs, 9 * num_obs + stress_size)
    assert slices["spectral"].stop == flat.size == 13 * num_obs + stress_size

    resolved = stats.resolve_error(raw, operator, ObservationErrorSpec(0.1))
    assert resolved.covariance_diag.size == flat.size
    assert resolved.block_slices == slices
    assert resolved.block_rows == {"bins": 8, "tke": 1, "stress": 1, "spectral": 4}
    rows = observation_rows(resolved)
    assert rows is not None
    assert (rows == "stress").sum() == stress_size
    heights = observation_heights(resolved, operator)
    coords = sensor_observation_coords(operator, flat.size)
    sensors = np.arange(flat.size) % _NUM_SENSORS
    np.testing.assert_array_equal(heights, operator.obs_z[sensors])
    np.testing.assert_array_equal(coords[:, 2], operator.obs_z[sensors])
    # The stress entries follow their sensor's correlation, pair after pair.
    stress = flat[slices["stress"]].reshape(num_pairs, _NUM_SENSORS)
    assert np.all(np.diff(stress, axis=1) > 0.2)
    # Every member of an ensemble gets the same entries as alone.
    members = [raw, raw + 0.1 * _correlated_series(400, 7, num_components)]
    ensemble = flatten_observations(stats(xr.concat(members, dim="ensemble")))
    assert ensemble.shape == (2, flat.size)
    for i, member in enumerate(members):
        np.testing.assert_allclose(ensemble[i], flatten_observations(stats(member)))


@pytest.mark.parametrize("normalize", [True, False])  # type: ignore[misc]
def test_predicted_stress_carries_the_instrument_noise(normalize: bool) -> None:
    """Independent noise per component: the covariance is unbiased, and the
    correlation coefficient of a noisy series matches the clean prediction
    with noise_variance in its variances (about 4-frame bin means here)."""
    noise = np.array([0.5, 0.5, 1.0, 0.2, 0.2, 2.0, 0.3, 0.3, 0.3])
    clean = xr.concat(
        [_correlated_series(400, seed, 3) for seed in range(50)], "ensemble"
    )
    rng = np.random.default_rng(9)
    noisy = clean + np.sqrt(noise) * rng.normal(size=clean.shape)
    stats = ObservationStatistics(
        ["bins", "stress"],
        AggregateObservations(2.0),
        stress={"normalize": normalize, "pool_std": "none"},
        num_sensors=_NUM_SENSORS,
        noise_variance=noise,
    )
    stress = stats.block_slices(clean.isel(ensemble=0))["stress"]
    observed = flatten_observations(stats(noisy, observed=True))[:, stress].mean(0)
    predicted = flatten_observations(stats(clean))[:, stress].mean(axis=0)
    np.testing.assert_allclose(predicted, observed, rtol=0.05, atol=0.01)
    stats.noise_variance = None
    naive = flatten_observations(stats(clean))[:, stress].mean(axis=0)
    if normalize:  # without the noise the correlations come out too strong
        assert np.max(np.abs(naive / observed - 1.0)) > 0.1
    else:  # the covariance does not see it
        np.testing.assert_array_equal(naive, predicted)


@pytest.mark.parametrize(  # type: ignore[misc]
    "num_components, interval, normalize",
    [
        (2, None, True),
        (2, 20.0, True),
        (3, None, False),
        (3, None, True),
        (3, 20.0, True),
        (3, 5.0, False),
    ],
)
def test_stress_auto_std_matches_the_monte_carlo_sampling_error(
    num_components: int, interval: float | None, normalize: bool
) -> None:
    """Cross-correlated AR(1) components (correlation time ~5 s) plus white
    noise: the median auto std is sqrt(2) times the spread over independent
    windows, without bins and about bin means of 4 and 1 correlation times."""
    types = ["stress"] if interval is None else ["bins", "stress"]
    stats = ObservationStatistics(
        types,
        None if interval is None else AggregateObservations(interval),
        stress={"normalize": normalize, "std_floor": 0.0, "pool_std": "none"},
        num_sensors=_NUM_SENSORS,
    )
    spec = ObservationErrorSpec(0.1)
    windows = [
        _correlated_series(240, seed, num_components)
        + 0.5 * np.random.default_rng(seed).normal(size=(240, 3 * num_components))
        for seed in range(400)
    ]
    stress = stats.block_slices(windows[0])["stress"]
    values = np.stack(
        [flatten_observations(stats(w, observed=True))[stress] for w in windows]
    )
    operator = _operator_for(num_components)
    auto = np.stack(
        [stats.resolve_error(w, operator, spec).std[stress] for w in windows[:100]]
    )
    ratio = np.median(auto, axis=0) / (np.sqrt(2.0) * values.std(axis=0))
    assert np.all((0.8 < ratio) & (ratio < 1.25)), ratio


def test_stress_error_options() -> None:
    raw = _correlated_series(400, seed=1, num_components=2)
    operator = _operator_for(2)
    spec = ObservationErrorSpec(0.1)
    fixed = ObservationStatistics(
        ["stress"],
        stress={"std": 0.3, "representation_std": 0.4},
        num_sensors=_NUM_SENSORS,
    ).resolve_error(raw, operator, spec)
    np.testing.assert_allclose(fixed.std, 0.5)
    assert fixed.std.size == _NUM_SENSORS
    assert "stress(std=0.3" in fixed.provenance and "log" not in fixed.provenance
    floored = ObservationStatistics(
        ["stress"], stress={"std_floor": 2.0}, num_sensors=_NUM_SENSORS
    ).resolve_error(raw, operator, spec)
    np.testing.assert_allclose(floored.std, 2.0)
    # Pooled per height: sensors 0 and 1 share one.
    pooled = ObservationStatistics(
        ["stress"], stress={"std_floor": 0.0}, num_sensors=_NUM_SENSORS
    ).resolve_error(raw, _pooled_operator(), spec)
    assert pooled.std[0] == pooled.std[1] != pooled.std[2]


def test_stress_needs_two_or_three_components() -> None:
    stats = ObservationStatistics(["stress"], num_sensors=_NUM_SENSORS)
    single = _series(200).isel(obs=slice(0, _NUM_SENSORS))
    with pytest.raises(ValueError, match="two or three observed velocity"):
        stats(single)
    operator = ObservationOperator(
        obs_states=["u"], obs_x=[0.0, 1.0, 2.0], obs_y=[0.0] * 3, obs_z=[1.0] * 3
    )
    with pytest.raises(ValueError, match="two or three observed velocity"):
        stats.resolve_error(single, operator, ObservationErrorSpec(0.1))


# ---------------------------------------------------------------------------
# ESMDA toy: the tke row constrains a fluctuation amplitude
# ---------------------------------------------------------------------------


class _AmplitudeEnsemble(BaseEnsembleForwardModel):
    """Sensor series ``mean + amp * noise`` whose amplitude is the parameter."""

    def __init__(self, noise: np.ndarray, times: np.ndarray) -> None:
        self._set_save_mode(None)
        self._last_failure_substitutions: dict[int, int] = {}
        self._failure_jitter_scale = 0.0
        self.noise = noise  # (time, z, y, x), shared by all members
        self.times = times

    def _create_new_forward_model(self, *args: Any, **kwargs: Any) -> BaseForwardModel:
        raise NotImplementedError

    def _pre_run_ensemble(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _post_run_ensemble(self, sim_name: str) -> xr.Dataset | None:
        return None

    def run_ensemble(
        self,
        state: Any = None,
        params: xr.Dataset | None = None,
        sim_name: str | None = "state",
    ) -> xr.Dataset:
        assert params is not None
        amp = np.asarray(params.amp.values)[:, None, None, None, None]
        u = 4.0 + amp * self.noise[None]
        return _field(u, self.times, ensemble=True)


def _field(u: np.ndarray, times: np.ndarray, ensemble: bool = False) -> xr.Dataset:
    dims = ("time", "z", "y", "x")
    coords: dict[str, Any] = {"time": times, "z": [0.0], "y": [0.0], "x": [0.0, 1.0]}
    if ensemble:
        dims = ("ensemble", *dims)
        coords["ensemble"] = np.arange(u.shape[0])
    return xr.Dataset({"u": (dims, u)}, coords=coords)


def test_esmda_moves_the_amplitude_toward_the_truth() -> None:
    rng = np.random.default_rng(5)
    times = 0.5 * np.arange(600)
    operator = TemporalObservationOperator(
        ObservationOperator(
            obs_ids_x=[0, 1], obs_ids_y=[0, 0], obs_ids_z=[0, 0], obs_states=["u"]
        )
    )
    truth_amp, noise_std = 2.0, 0.3
    clean = operator(_field(4.0 + truth_amp * rng.normal(size=(600, 1, 1, 2)), times))
    noisy = clean + noise_std * rng.normal(size=clean.shape)

    stats = ObservationStatistics(
        ["tke", "spectral"], spectral={"std": 0.5}, num_sensors=2
    )
    spec = ObservationErrorSpec(noise_std)
    resolved = stats.resolve_error(noisy, operator, spec)
    stats.noise_variance = resolved.noise_variance
    observed = flatten_observations(stats(noisy, observed=True))

    smoother = ParameterESMDA(
        observation_operator=cast(Any, operator),
        forward_model=_AmplitudeEnsemble(rng.normal(size=(600, 1, 1, 2)), times),
        C_D=resolved.covariance_diag,
        num_steps=3,
        rng_key=jax.random.PRNGKey(0),
        aggregate_observations=cast(Any, stats),
        log_parameter_names={"amp": None},
    )
    prior = xr.Dataset({"amp": ("ensemble", np.exp(rng.normal(0.0, 0.5, 40)))})
    posterior = smoother(params=prior, observations=observed)
    posterior = posterior[0] if isinstance(posterior, tuple) else posterior
    prior_error = abs(np.log(np.median(prior.amp.values) / truth_amp))
    posterior_error = abs(np.log(np.median(posterior.amp.values) / truth_amp))
    assert posterior_error < 0.1 < prior_error
    assert posterior.amp.values.std() < prior.amp.values.std()
