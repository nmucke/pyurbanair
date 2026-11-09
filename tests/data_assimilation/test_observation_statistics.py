"""Turbulence-statistics observations: definitions, layout, error and an ESMDA toy."""

from typing import Any, cast

import jax
import numpy as np
import pytest
import xarray as xr
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    TemporalObservationOperator,
    flatten_observations,
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
    stats = ObservationStatistics(["tke", "raw"])
    out = stats(raw)
    assert list(out.block.values) == ["raw"] * 200 + ["tke"]
    np.testing.assert_array_equal(out.isel(time=slice(0, 200)).values, raw.values)


# ---------------------------------------------------------------------------
# Statistic definitions
# ---------------------------------------------------------------------------


def test_tke_is_half_the_population_variance() -> None:
    raw = _series(500, std=np.array([1.0, 2.0, 0.5, 1.0, 1.5, 3.0]))
    plain = ObservationStatistics(["tke"], tke={"log": False, "pool_std": "none"})(raw)
    np.testing.assert_allclose(plain.values[0], 0.5 * raw.values.var(axis=0))

    logged = ObservationStatistics(["tke"])(raw)
    np.testing.assert_allclose(logged.values[0], np.log(plain.values[0]))

    normalized = ObservationStatistics(
        ["tke"], tke={"normalize": True, "log": False}, num_sensors=_NUM_SENSORS
    )(raw)
    mean = raw.values.mean(axis=0).reshape(2, _NUM_SENSORS)
    speed2 = np.tile((mean**2).sum(axis=0), 2)
    np.testing.assert_allclose(normalized.values[0], plain.values[0] / (0.5 * speed2))


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
        (["tke"], dict(tke={"log": False, "pool_std": "none"})),
        (["tke"], dict(tke={"normalize": True}, num_sensors=_NUM_SENSORS)),
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
        tke={"log": False, "pool_std": "none"},
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
    stats = _all(tke={"normalize": True})
    out = stats(ensemble)
    assert out.dims == ("ensemble", "time", "obs")
    flat = flatten_observations(out)
    for i, member in enumerate(members):
        np.testing.assert_allclose(flat[i], flatten_observations(stats(member)))


def test_interval_count_change_is_forwarded() -> None:
    aggregation = AggregateObservations(25.0)
    stats = ObservationStatistics(["bins", "tke"], aggregation)
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
        ["tke", "spectral"], tke={"std_floor": 5.0}, spectral={"std": 0.3}
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
    pooled = ObservationStatistics(["tke"], tke=options).resolve_error(
        raw, _pooled_operator(), spec
    )
    alone = ObservationStatistics(
        ["tke"], tke={**options, "pool_std": "none"}
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
        ["tke"], tke={"std": 0.3, "representation_std": 0.4}
    ).resolve_error(raw, _operator(), spec)
    np.testing.assert_allclose(fixed.std, 0.5)
    sampling = ObservationStatistics(["spectral"], spectral={"std_floor": 0.0})
    auto = sampling.resolve_error(raw, _operator(), spec).std
    sampling.spectral["representation_std"] = 0.2
    np.testing.assert_allclose(
        sampling.resolve_error(raw, _operator(), spec).std,
        np.sqrt(auto**2 + 0.2**2),
    )


def test_auto_std_matches_the_monte_carlo_sampling_error() -> None:
    """Default tke and spectral rows of a red (AR(1)) series plus white noise:
    the median auto std is sqrt(2) (observed plus predicted sampling error)
    times the spread over independent windows."""
    stats = ObservationStatistics(
        ["tke", "spectral"], tke={"std_floor": 0.0}, spectral={"std_floor": 0.0}
    )
    spec = ObservationErrorSpec(0.1)
    windows = [
        _red_series(240, seed) + 0.5 * np.random.default_rng(seed).normal(size=(240, 6))
        for seed in range(400)
    ]
    values = np.stack([stats(w, observed=True).values.reshape(-1) for w in windows])
    auto = np.stack(
        [stats.resolve_error(w, _operator(), spec).std for w in windows[:100]]
    )
    ratio = np.median(auto, axis=0) / (np.sqrt(2.0) * values.std(axis=0))
    assert np.all((0.7 < ratio) & (ratio < 1.3)), ratio


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
        stats = ObservationStatistics(types)
        with pytest.raises(ValueError, match="constant or non-finite"):
            stats(raw)


def test_a_constant_observed_series_is_rejected() -> None:
    raw = _series(400)
    raw[:, 4] = 2.0
    for types in (["tke"], ["spectral"]):
        with pytest.raises(ValueError, match="obs 4 \\(sensor 1\\)"):
            ObservationStatistics(types).resolve_error(
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
        (dict(types=["tke"], tke={"normalize": True}), "num_sensors"),
        (dict(types=["tke"], tke={"window": 3}), "Unknown tke options"),
        (dict(types=["spectral"], spectral={"num_batches": 3}), "Unknown spectral"),
        (dict(types=["tke"], tke={"log": False}), "needs log or normalize"),
        (dict(types=["tke"], tke={"pool_std": "sensor"}), "pool_std"),
        (dict(types=["tke"], tke={"representation_std": -1.0}), "non-negative"),
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

    stats = ObservationStatistics(["tke", "spectral"], spectral={"std": 0.5})
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
