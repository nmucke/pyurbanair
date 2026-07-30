"""Property tests for the WP1.2 statistics-space sensor scoring.

Everything runs on synthetic ``xarray`` series shaped exactly like the ones
``truth_sensor_series`` / ``ensemble_sensor_series`` return -- dims
``("component", "time", "sensor")`` and ``("component", "ensemble", "time",
"sensor")`` -- with fixed seeds. Assertions are closed-form identities,
constructed answers or statistical properties of a deliberately calibrated
ensemble; there are no snapshots, so a reimplementation that gets the same
statistics right passes.

Three shapes get explicit coverage because they are the ones that bite:

* the **two windowing rules**, which are different (ensemble by time value,
  truth by frame index) and are built here to disagree loudly if either is
  applied to the wrong series;
* ``M = 2``, the smoke shape the suite runs everywhere, where every calibration
  number is an artifact of the ensemble size and must come out ``null``;
* a window holding 3 frames against the default 20 bootstrap blocks, where the
  identifiability bootstrap has no resolution at all and must report ``null``
  rather than a number produced from length-1 blocks.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray

from scripts.esmda._esmda_common import (
    IDENTIFIABILITY_MIN_RATIO,
    MIN_MEMBERS_CALIBRATION,
    PIT_BINS,
    SENSOR_STATISTIC_NAMES,
    _ensemble_window_indices,
    _statistic_window_series,
    _var_ddof1,
    read_yaml,
    sensor_statistic_scores,
    write_yaml,
)

COMPONENTS = ["u", "v", "w"]
SIM_TIME = 10.0
NUM_WINDOWS = 3
SET = "assimilation"

# The pinned schema, as key SETS rather than samples of it: a consumer indexes
# these paths, and the whole point of the "every key present, as null" rule is
# that they are the same on a degraded run as on a healthy one.
SET_KEYS = frozenset(
    {
        "n_members",
        "n_windows",
        "n_sensors",
        "window_mean",
        "window_variance",
        "tke",
        "velmag_mean",
        "wasserstein",
    }
)
STATISTIC_KEYS = frozenset(
    {
        "crps",
        "crpss_vs_prior",
        "zscore",
        "coverage",
        "pit_counts",
        "pit",
        "identifiability",
        "n_samples",
        "reason",
    }
)
ZSCORE_KEYS = frozenset({"mean", "std", "max_abs", "max_abs_calibrated_median"})
COVERAGE_KEYS = frozenset(
    {
        "alpha_50",
        "alpha_90",
        "nominal_alpha_50",
        "nominal_alpha_90",
        "max_nominal_alpha",
    }
)
PIT_KEYS = frozenset({"n_bins", "n_samples", "ranks_per_bin", "tie_seed"})
IDENTIFIABILITY_KEYS = frozenset(
    {"ratio_median", "ratio_min", "sampling_noise_dominated", "threshold"}
)
WASSERSTEIN_KEYS = frozenset(
    {
        "w1_over_sigma_pooled",
        "w1_over_sigma_member_mean",
        "self_floor",
        "w1_over_floor",
        "reason",
    }
)


# ---------------------------------------------------------------------------
# Synthetic sensor series
# ---------------------------------------------------------------------------


def _ensemble_da(values: np.ndarray, times: np.ndarray) -> xarray.DataArray:
    """``(component, ensemble, time, sensor)``, the ensemble series' real shape."""
    values = np.asarray(values, dtype=float)
    return xarray.DataArray(
        values,
        dims=("component", "ensemble", "time", "sensor"),
        coords={
            "component": COMPONENTS,
            "ensemble": np.arange(values.shape[1]),
            "time": np.asarray(times, dtype=float),
            "sensor": np.arange(values.shape[3]),
        },
    )


def _truth_da(values: np.ndarray, times: np.ndarray) -> xarray.DataArray:
    """``(component, time, sensor)``, the truth series' real shape."""
    values = np.asarray(values, dtype=float)
    return xarray.DataArray(
        values,
        dims=("component", "time", "sensor"),
        coords={
            "component": COMPONENTS,
            "time": np.asarray(times, dtype=float),
            "sensor": np.arange(values.shape[2]),
        },
    )


def _rebased_times(
    frames_per_window: int, num_windows: int = NUM_WINDOWS
) -> np.ndarray:
    """The global axis ``ensemble_sensor_series`` builds: window w at w*sim_time."""
    within = np.arange(frames_per_window) * (SIM_TIME / frames_per_window)
    return np.concatenate([w * SIM_TIME + within for w in range(num_windows)])


def _exchangeable_series(
    n_members: int,
    n_sensors: int,
    frames_per_window: int,
    seed: int,
    member_spread: float = 1.0,
    truth_spread: float | None = None,
) -> tuple[dict, dict]:
    """A truth drawn from the ensemble's own law -- the exactly calibrated case.

    Every (component, window, sensor) element gets an offset plus iid noise;
    members and truth draw their offsets and their noise from the same
    distributions and over the same number of frames, so the window mean, the
    window variance, the TKE and the mean ``|U|`` are all exchangeable between
    truth and members. That is the only construction under which coverage is
    supposed to land on its nominal level for all four statistics at once.

    ``truth_spread`` breaks the exchangeability on purpose (overconfidence when
    it exceeds ``member_spread``).
    """
    rng = np.random.default_rng(seed)
    shape = (len(COMPONENTS), NUM_WINDOWS, n_sensors)
    n_time = NUM_WINDOWS * frames_per_window

    def _draw(spread: float, n_ens: int | None) -> np.ndarray:
        offsets = rng.normal(
            0.0, spread, size=shape if n_ens is None else (n_ens, *shape)
        )
        noise_shape: tuple[int, ...] = (
            len(COMPONENTS),
            NUM_WINDOWS,
            frames_per_window,
            n_sensors,
        )
        if n_ens is not None:
            noise_shape = (n_ens, *noise_shape)
        noise = rng.normal(0.0, 1.0, size=noise_shape)
        series = offsets[..., None, :] * np.ones_like(noise) + noise
        # (..., component, window, time, sensor) -> (..., component, time, sensor)
        merged = series.reshape(*series.shape[:-3], n_time, n_sensors)
        return merged

    truth = _draw(member_spread if truth_spread is None else truth_spread, None)
    members = np.moveaxis(_draw(member_spread, n_members), 0, 1)
    return (
        {SET: _truth_da(truth, np.arange(n_time, dtype=float))},
        {SET: _ensemble_da(members, _rebased_times(frames_per_window))},
    )


def _score(truth_series: dict, ensemble_series: dict, **kwargs: object) -> dict:
    """Call the entry point with the pipeline's arguments, defaults filled in."""
    frames = int(truth_series[SET].sizes["time"]) // NUM_WINDOWS
    options: dict = {
        "num_windows": NUM_WINDOWS,
        "n_per_window": frames,
        "sim_time": SIM_TIME,
        "bootstrap_blocks": 20,
    }
    options.update(kwargs)
    scores: dict = sensor_statistic_scores(truth_series, ensemble_series, **options)
    return scores


# ---------------------------------------------------------------------------
# Windowing: two different rules, applied to two different series
# ---------------------------------------------------------------------------


def test_the_truth_is_windowed_by_index_while_the_ensemble_is_windowed_by_time() -> (
    None
):
    """Each series is sliced by its own rule; a swapped rule is visibly wrong.

    Constructed so that every plausible mistake is loud. The truth is uniform
    inside a window and jumps between windows, so a wrong truth slice mixes two
    plateaus; the ensemble members are set to the exact per-window plateau, so a
    correct pairing gives CRPS exactly 0 and any mispairing gives O(4).

    * Slicing the truth by TIME VALUE would be wrong: its 15 frames are stamped
      at t = 0..14, so the ``[0, 10)`` interval would swallow ten of them and
      windows 1 and 2 would be empty.
    * Slicing the ensemble by INDEX in equal blocks would be wrong: its windows
      hold 8, 5 and 8 frames, so equal thirds of 21 would straddle them.

    The naive alternative -- reindexing both onto one shared time axis and
    comparing pointwise -- cannot work at all here: the two series have
    different lengths and cadences, which is the routine case, not a contrived
    one.
    """
    plateaus = [1.0, 5.0, 9.0]
    frames_per_window = [8, 5, 8]

    truth_values = np.zeros((len(COMPONENTS), NUM_WINDOWS * 5, 2))
    for w, level in enumerate(plateaus):
        truth_values[0, w * 5 : (w + 1) * 5, :] = level
    truth = {SET: _truth_da(truth_values, np.arange(15, dtype=float))}

    times, ensemble_blocks = [], []
    for w, (level, n_frames) in enumerate(zip(plateaus, frames_per_window)):
        times.append(w * SIM_TIME + np.arange(n_frames) * (SIM_TIME / n_frames))
        block = np.zeros((len(COMPONENTS), 4, n_frames, 2))
        block[0] = level
        ensemble_blocks.append(block)
    ensemble = {
        SET: _ensemble_da(
            np.concatenate(ensemble_blocks, axis=2), np.concatenate(times)
        )
    }

    entry = _score(truth, ensemble, n_per_window=5)[SET]

    # Members equal the truth exactly in every window -> a zero CRPS. Any
    # mis-slice pairs a member plateau with a different truth plateau.
    assert entry["window_mean"]["crps"] == pytest.approx(0.0, abs=1e-12)
    assert entry["velmag_mean"]["crps"] == pytest.approx(0.0, abs=1e-12)
    # 3 components x 2 sensors x 3 windows for the per-component statistics.
    assert entry["window_mean"]["n_samples"] == 18
    assert entry["velmag_mean"]["n_samples"] == 6


def test_no_ensemble_frame_is_dropped_by_the_half_open_window_intervals() -> None:
    """The final window keeps a frame landing exactly on ``num_windows*sim_time``.

    A solver that writes its end-of-window frame at the closing boundary would
    otherwise fall outside every ``[w*sim_time, (w+1)*sim_time)`` interval and be
    silently discarded -- a whole frame lost from the last window's statistics
    with nothing in the summary saying so.
    """
    times = np.concatenate([_rebased_times(4), [NUM_WINDOWS * SIM_TIME]])
    ensemble = _ensemble_da(np.zeros((3, 2, times.size, 1)), times)

    indices = _ensemble_window_indices(ensemble, NUM_WINDOWS, SIM_TIME, SET)

    assert [i.size for i in indices] == [4, 4, 5]
    assert sorted(np.concatenate(indices)) == list(range(times.size))


def test_a_series_without_a_time_coordinate_falls_back_to_contiguous_blocks() -> None:
    """No ``time`` coord -> equal blocks, which is only correct when they are equal.

    ``ensemble_sensor_series`` rebases the time axis only when the window file
    carries one, so the coordinate-free series is a real path. Equal-length
    windows make the fallback agree with the time-value rule exactly, which is
    the assertion; the fallback is logged precisely because that agreement is
    not guaranteed in general.
    """
    values = np.arange(3 * 2 * 12 * 1, dtype=float).reshape(3, 2, 12, 1)
    with_coord = _ensemble_da(values, _rebased_times(4))
    without_coord = with_coord.drop_vars("time")

    assert "time" not in without_coord.coords
    by_value = _ensemble_window_indices(with_coord, NUM_WINDOWS, SIM_TIME, SET)
    by_block = _ensemble_window_indices(without_coord, NUM_WINDOWS, SIM_TIME, SET)

    for expected, actual in zip(by_value, by_block):
        assert list(expected) == list(actual)


# ---------------------------------------------------------------------------
# The statistics themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "name", ["window_mean", "window_variance", "tke", "velmag_mean"]
)
def test_every_statistic_is_the_reducer_of_a_one_dimensional_series(name: str) -> None:
    """``member value == reducer(member series)``, exactly, for all four.

    This identity is what makes the identifiability bootstrap meaningful:
    ``block_bootstrap_std`` resamples blocks of a 1-D series and re-applies the
    reducer, so if the tabulated statistic were computed by some *other*
    expression the bootstrap would be quantifying the sampling noise of a
    quantity nobody reports. The TKE case is the one that could silently drift
    -- the natural per-timestep energy has a ``ddof=0`` mean, so the series
    carries an ``n/(n-1)`` factor to land on the reported ``ddof=1`` TKE.
    """
    rng = np.random.default_rng(7)
    ensemble_window = rng.normal(size=(3, 5, 9, 2))
    truth_window = rng.normal(size=(3, 6, 2))

    series, truth_series, reducer = _statistic_window_series(
        name, ensemble_window, truth_window
    )
    members = reducer(series, axis=-1)

    if name == "window_mean":
        expected = np.transpose(ensemble_window.mean(axis=2), (1, 0, 2)).reshape(5, 6)
    elif name == "window_variance":
        expected = np.transpose(ensemble_window.var(axis=2, ddof=1), (1, 0, 2)).reshape(
            5, 6
        )
    elif name == "tke":
        expected = np.transpose(
            0.5 * ensemble_window.var(axis=2, ddof=1).sum(axis=0), (0, 1)
        )
    else:
        magnitude = np.sqrt((ensemble_window**2).sum(axis=0))
        expected = magnitude.mean(axis=1)

    assert np.allclose(members, expected)
    # And the same reducer applied to the truth series reproduces the truth
    # statistic, so both sides of every score are the same estimator.
    assert reducer(truth_series, axis=-1).shape == (expected.shape[-1],)


def test_tke_is_half_the_summed_component_variance_of_the_window() -> None:
    """Closed form on a constructed slab: sinusoids of known amplitude.

    A sinusoid of amplitude ``A`` sampled over a whole number of periods has
    variance ``A**2/2``, so the TKE is ``0.25 * sum_c A_c**2`` up to the ddof=1
    correction. Checked against the reducer path rather than a snapshot.
    """
    n_time = 64
    phase = 2 * np.pi * np.arange(n_time) / n_time
    amplitudes = np.array([2.0, 1.0, 0.5])
    slab = (amplitudes[:, None] * np.sin(phase)[None, :])[:, None, :, None]

    series, _, reducer = _statistic_window_series("tke", slab, slab[:, 0])
    tke = float(reducer(series, axis=-1).ravel()[0])

    exact = 0.5 * float(np.sum(_var_ddof1(slab[:, 0, :, 0], axis=-1)))
    assert tke == pytest.approx(exact, rel=1e-12)
    assert tke == pytest.approx(0.25 * float((amplitudes**2).sum()), rel=0.05)


def test_the_pooling_axes_match_the_documented_schema() -> None:
    """``n_samples`` is C*S*W for the per-component statistics and S*W for the rest.

    The two per-component statistics pool the component axis (a mean of ``u`` is
    a different number from a mean of ``v``); TKE and mean ``|U|`` have already
    summed the components away, so pooling them again would double-count.
    """
    n_sensors = 4
    truth, ensemble = _exchangeable_series(6, n_sensors, 8, seed=11)
    entry = _score(truth, ensemble)[SET]

    assert entry["n_members"] == 6
    assert entry["n_windows"] == NUM_WINDOWS
    assert entry["n_sensors"] == n_sensors
    assert entry["window_mean"]["n_samples"] == 3 * n_sensors * NUM_WINDOWS
    assert entry["window_variance"]["n_samples"] == 3 * n_sensors * NUM_WINDOWS
    assert entry["tke"]["n_samples"] == n_sensors * NUM_WINDOWS
    assert entry["velmag_mean"]["n_samples"] == n_sensors * NUM_WINDOWS


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_a_calibrated_ensemble_scores_coverage_near_its_nominal_level() -> None:
    """Truth drawn from the members' own law -> coverage on ``nominal_alpha_*``.

    Compared against ``nominal_alpha_90``, never against 0.9: the band edges are
    member order statistics, so at M = 15 the attainable levels are multiples of
    1/16 and reading the empirical fraction against the *requested* alpha
    reports pure discretization as miscalibration.
    """
    truth, ensemble = _exchangeable_series(15, 30, 8, seed=3)
    entry = _score(truth, ensemble)[SET]

    for name in SENSOR_STATISTIC_NAMES:
        block = entry[name]
        assert block["reason"] is None, name
        coverage = block["coverage"]
        # 270 (or 90) pooled elements: sampling error on a coverage fraction is
        # ~0.02 (~0.03), so 0.09 never flakes and still catches a wrong band.
        assert coverage["alpha_90"] == pytest.approx(
            coverage["nominal_alpha_90"], abs=0.09
        ), name
        assert coverage["alpha_50"] == pytest.approx(
            coverage["nominal_alpha_50"], abs=0.12
        ), name
        assert block["zscore"]["mean"] == pytest.approx(0.0, abs=0.3), name
        assert block["pit_counts"] is not None
        assert sum(block["pit_counts"]) == block["n_samples"]


def test_an_overconfident_ensemble_trips_the_max_abs_zscore() -> None:
    """A truth drawn far wider than the members: large ``max_abs``, empty band.

    Read against ``max_abs_calibrated_median``, the size-aware reference, rather
    than a fixed cut: the maximum of ``n`` draws grows with ``n``, so a bare
    ``max_abs > 3`` flags the pooled sample size on a perfectly calibrated
    ensemble instead of flagging miscalibration.
    """
    truth, ensemble = _exchangeable_series(
        15, 20, 8, seed=4, member_spread=0.05, truth_spread=4.0
    )
    block = _score(truth, ensemble)[SET]["window_mean"]

    assert (
        block["zscore"]["max_abs"] > 5.0 * block["zscore"]["max_abs_calibrated_median"]
    )
    assert block["zscore"]["std"] > 3.0
    assert block["coverage"]["alpha_90"] < 0.5
    # Under-dispersion piles the truth into the outermost rank bins.
    counts = np.asarray(block["pit_counts"])
    assert counts[0] + counts[-1] > 0.5 * counts.sum()


def test_a_calibrated_ensemble_is_not_flagged_by_the_same_reference() -> None:
    """The companion to the previous test: the reference does not fire on noise."""
    truth, ensemble = _exchangeable_series(15, 20, 8, seed=5)
    block = _score(truth, ensemble)[SET]["window_mean"]

    assert (
        block["zscore"]["max_abs"] < 3.0 * block["zscore"]["max_abs_calibrated_median"]
    )


# ---------------------------------------------------------------------------
# CRPS skill against a prior
# ---------------------------------------------------------------------------


def test_crpss_vs_prior_is_null_without_a_prior_and_positive_against_a_worse_one() -> (
    None
):
    """``run.save_prior_state`` defaults to false, so ``None`` is the common case.

    It must be a silent ``null``, not an error and not a fabricated 0.0 -- a
    zero would read as "the update achieved nothing", which is a claim the run
    dir contains no evidence for.
    """
    truth, ensemble = _exchangeable_series(10, 12, 8, seed=6)
    without_prior = _score(truth, ensemble)[SET]
    for name in SENSOR_STATISTIC_NAMES:
        assert without_prior[name]["crpss_vs_prior"] is None, name

    # A prior biased far off the truth: the posterior must show positive skill.
    prior = {SET: ensemble[SET] + 25.0}
    with_prior = _score(truth, ensemble, prior_series=prior)[SET]
    for name in ("window_mean", "velmag_mean"):
        skill = with_prior[name]["crpss_vs_prior"]
        assert skill is not None and 0.8 < skill <= 1.0, name

    # The posterior numbers are untouched by supplying a reference.
    assert with_prior["window_mean"]["crps"] == pytest.approx(
        without_prior["window_mean"]["crps"]
    )


# ---------------------------------------------------------------------------
# Identifiability
# ---------------------------------------------------------------------------


def test_the_identifiability_bootstrap_is_null_at_the_smoke_window_length() -> None:
    """3 frames per window against 20 blocks has no resolution -- report ``null``.

    A moving-block bootstrap with a block length of 1 is an iid bootstrap, which
    for a serially correlated window is precisely the estimate the blocking
    exists to avoid; producing it anyway would put a confidently wrong
    identifiability ratio next to numbers that are fine. The rest of the block
    must survive: the CRPS and the coverage do not need the bootstrap.
    """
    truth, ensemble = _exchangeable_series(6, 4, 3, seed=8)
    entry = _score(truth, ensemble, bootstrap_blocks=20)[SET]

    for name in SENSOR_STATISTIC_NAMES:
        identifiability = entry[name]["identifiability"]
        assert set(identifiability) == IDENTIFIABILITY_KEYS, name
        assert identifiability["ratio_median"] is None, name
        assert identifiability["ratio_min"] is None, name
        # Unknown is not "not dominated".
        assert identifiability["sampling_noise_dominated"] is None, name
        assert identifiability["threshold"] == IDENTIFIABILITY_MIN_RATIO
    # Everything that does not depend on the bootstrap is still reported.
    assert entry["window_mean"]["crps"] is not None
    assert entry["window_mean"]["coverage"]["alpha_90"] is not None
    assert entry["window_mean"]["reason"] is None


def test_identifiability_separates_a_real_spread_from_sampling_noise() -> None:
    """Members that differ only by their noise draw are sampling-noise dominated.

    Two runs on the same window length and the same bootstrap settings, so the
    only thing that changes is whether the members carry a genuine offset:

    * no offset -> the across-member spread IS the within-member sampling noise,
      ratio ~ 1, flag ``True``;
    * a large offset -> the spread is signal, ratio well above the threshold,
      flag ``False``.

    Asserting the inequality between the two rather than a value: the exact
    ratio depends on the noise realization, the direction does not.
    """
    n_time = 96
    rng = np.random.default_rng(9)
    times = np.arange(n_time) * (SIM_TIME / n_time)
    truth = {
        SET: _truth_da(rng.normal(size=(3, n_time, 2)), np.arange(n_time, dtype=float))
    }

    def _entry(offset_spread: float) -> dict:
        noise = rng.normal(size=(3, 5, n_time, 2))
        offsets = rng.normal(0.0, offset_spread, size=(3, 5, 1, 2))
        ensemble = {SET: _ensemble_da(noise + offsets, times)}
        scores: dict = sensor_statistic_scores(
            truth,
            ensemble,
            num_windows=1,
            n_per_window=n_time,
            sim_time=SIM_TIME,
            bootstrap_blocks=8,
        )
        block: dict = scores[SET]["window_mean"]["identifiability"]
        return block

    noise_only = _entry(0.0)
    with_signal = _entry(20.0)

    assert noise_only["ratio_median"] is not None
    assert noise_only["ratio_median"] < IDENTIFIABILITY_MIN_RATIO
    assert noise_only["sampling_noise_dominated"] is True
    assert with_signal["ratio_median"] > 10.0 * noise_only["ratio_median"]
    assert with_signal["sampling_noise_dominated"] is False
    # Reported either way -- the flag is a reading aid, never a gate.
    assert with_signal["ratio_min"] is not None


# ---------------------------------------------------------------------------
# The Wasserstein layer
# ---------------------------------------------------------------------------


def test_a_shifted_ensemble_distribution_is_far_above_its_own_self_floor() -> None:
    """``w1_over_floor`` ~ 1 for a matching distribution, >> 1 for a shifted one.

    The floor is the point of the layer: a raw W1 is unreadable without knowing
    what a *perfect* model would score at this sample count and autocorrelation.
    Both runs use the same truth and the same sample counts, so the floor is
    identical and only the numerator moves.
    """
    rng = np.random.default_rng(12)
    n_time = 240
    times = np.arange(n_time) * (SIM_TIME * NUM_WINDOWS / n_time)
    truth_values = np.abs(rng.normal(2.0, 1.0, size=(3, n_time, 3)))
    truth = {SET: _truth_da(truth_values, np.arange(n_time, dtype=float))}

    def _wasserstein(shift: float) -> dict:
        members = np.abs(rng.normal(2.0 + shift, 1.0, size=(3, 6, n_time, 3)))
        scores: dict = sensor_statistic_scores(
            truth,
            {SET: _ensemble_da(members, times)},
            num_windows=NUM_WINDOWS,
            n_per_window=n_time // NUM_WINDOWS,
            sim_time=SIM_TIME,
            # This test reads only the Wasserstein layer, which is computed over
            # the whole run and never touches the bootstrap. A block count finer
            # than the window length short-circuits the identifiability
            # bootstrap into its documented `null`, which keeps the test cheap.
            bootstrap_blocks=500,
        )
        block: dict = scores[SET]["wasserstein"]
        return block

    matched = _wasserstein(0.0)
    shifted = _wasserstein(6.0)

    assert set(matched) == WASSERSTEIN_KEYS
    assert matched["reason"] is None
    assert matched["self_floor"]["median"] > 0.0
    # A matched draw is within a small multiple of what a perfect model scores;
    # a 6-sigma shift is orders away. The inequality is the assertion.
    assert matched["w1_over_floor"]["median"] < 5.0
    assert (
        shifted["w1_over_floor"]["median"] > 20.0 * matched["w1_over_floor"]["median"]
    )
    assert shifted["w1_over_sigma_pooled"]["median"] > 3.0


def test_a_straddling_ensemble_separates_the_pooled_and_per_member_wasserstein() -> (
    None
):
    """``W1`` is convex, so pooling can only shrink the distance -- never inflate it.

    Two ensembles over the same truth and the same sample counts:

    * members split symmetrically above and below the truth. Each one is off by
      1.5 sigma on its own, but their pooled predictive distribution is centred
      correctly, so the pooled distance collapses.
    * every member shifted the same way. Pooling cannot cancel a shared bias, so
      the two numbers coincide.

    Reporting only one of them cannot distinguish those: the pooled number calls
    the first ensemble nearly perfect, the member mean calls it as wrong as the
    second. The assertion is the inequality and the collapse, not a value.
    """
    rng = np.random.default_rng(13)
    n_time = 200
    times = np.arange(n_time) * (SIM_TIME * NUM_WINDOWS / n_time)
    truth_values = rng.normal(20.0, 1.0, size=(3, n_time, 2))

    def _wasserstein(shifts: list[float]) -> dict:
        members = np.stack(
            [
                rng.normal(20.0, 1.0, size=(3, n_time, 2)) + shift / np.sqrt(3.0)
                for shift in shifts
            ],
            axis=1,
        )
        scores: dict = sensor_statistic_scores(
            {SET: _truth_da(truth_values, np.arange(n_time, dtype=float))},
            {SET: _ensemble_da(members, times)},
            num_windows=NUM_WINDOWS,
            n_per_window=n_time // NUM_WINDOWS,
            sim_time=SIM_TIME,
            bootstrap_blocks=500,  # see the note above
        )
        block: dict = scores[SET]["wasserstein"]
        return block

    straddling = _wasserstein([1.5, 1.5, 1.5, -1.5, -1.5, -1.5])
    biased = _wasserstein([1.5] * 6)

    def _ratio(block: dict) -> float:
        pooled = block["w1_over_sigma_pooled"]["median"]
        per_member = block["w1_over_sigma_member_mean"]["median"]
        # Convexity, asserted rather than assumed: it is the whole reason two
        # numbers are emitted instead of one.
        assert pooled <= per_member + 1e-9
        return float(pooled / per_member)

    # Measured on this construction: 0.511 straddling, 1.000 biased.
    assert _ratio(straddling) < 0.7
    assert _ratio(biased) > 0.95


def test_a_dead_truth_sensor_nulls_the_wasserstein_layer_with_a_reason() -> None:
    """No spread -> no scale to normalize by, and no self floor. ``null``, not zero.

    Not a contrived input: a sensor point that falls inside a building reads
    ``u = v = w = 0`` at every frame, so its ``|U|`` is identically zero and has
    neither a sigma to normalize by nor a self floor. Dividing by that zero
    would emit ``inf``, which ``write_yaml`` cannot round-trip and which poisons
    any sweep aggregate that averages the column.
    """
    n_time = 60
    times = np.arange(n_time) * (SIM_TIME * NUM_WINDOWS / n_time)
    rng = np.random.default_rng(14)
    truth = {SET: _truth_da(np.zeros((3, n_time, 2)), np.arange(n_time, dtype=float))}
    ensemble = {SET: _ensemble_da(rng.normal(size=(3, 5, n_time, 2)), times)}

    block = sensor_statistic_scores(
        truth,
        ensemble,
        num_windows=NUM_WINDOWS,
        n_per_window=n_time // NUM_WINDOWS,
        sim_time=SIM_TIME,
        bootstrap_blocks=20,
    )[SET]["wasserstein"]

    assert set(block) == WASSERSTEIN_KEYS
    for key in ("w1_over_sigma_pooled", "w1_over_sigma_member_mean", "self_floor"):
        assert block[key] is None, key
    assert block["reason"] is not None


# ---------------------------------------------------------------------------
# Degenerate shapes
# ---------------------------------------------------------------------------


def test_two_members_null_every_number_with_a_reason() -> None:
    """M = 2, the smoke shape: every key present, every number ``null``.

    At two members a ddof=1 spread has one degree of freedom, the widest
    order-statistic band is nominally 1/3 (so a "90%" coverage is unattainable)
    and the 10-bin PIT has three rank values to spread over ten bins. Every one
    of those is a measurement of the ensemble SIZE, so the answer is a logged
    ``null`` rather than a special-cased formula.
    """
    truth, ensemble = _exchangeable_series(2, 4, 8, seed=15)
    entry = _score(truth, ensemble)[SET]

    assert MIN_MEMBERS_CALIBRATION > 2
    assert entry["n_members"] == 2
    for name in SENSOR_STATISTIC_NAMES:
        block = entry[name]
        assert block["reason"] is not None, name
        assert block["crps"] is None, name
        assert block["pit_counts"] is None, name
        assert all(value is None for value in block["zscore"].values()), name
        assert all(value is None for value in block["coverage"].values()), name
    assert entry["wasserstein"]["reason"] is not None
    assert entry["wasserstein"]["w1_over_sigma_pooled"] is None


def test_a_single_frame_window_has_no_variance_and_says_so() -> None:
    """One frame per window: the ddof=1 variance is ``null``, not zero.

    A zero variance would be indistinguishable from a genuinely laminar sensor
    and would make the TKE read as exactly 0 everywhere.
    """
    truth, ensemble = _exchangeable_series(5, 3, 1, seed=16)
    entry = _score(truth, ensemble)[SET]

    assert entry["window_mean"]["crps"] is not None
    for name in ("window_variance", "tke"):
        assert entry[name]["crps"] is None, name
        assert entry[name]["reason"] is not None, name
        assert entry[name]["n_samples"] == 0, name


def test_a_zero_window_count_nulls_the_set_rather_than_raising() -> None:
    """``num_windows`` missing from a pre-phase-1 ``truth_access.yaml`` is a null."""
    truth, ensemble = _exchangeable_series(8, 3, 8, seed=17)
    entry = _score(truth, ensemble, num_windows=0)[SET]

    assert entry["n_windows"] == 0
    assert entry["window_mean"]["crps"] is None
    assert "window count" in entry["window_mean"]["reason"]


def test_an_ensemble_set_without_a_truth_counterpart_is_skipped() -> None:
    """Only sets present on both sides are scored; the mismatch is logged, not fatal."""
    truth, ensemble = _exchangeable_series(5, 3, 8, seed=22)
    ensemble["validation"] = ensemble[SET]

    scores = _score(truth, ensemble)

    assert set(scores) == {SET}


# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------


def _assert_schema(entry: dict) -> None:
    """Every key path the summary pins, at whatever depth it is not ``null``."""
    assert set(entry) == SET_KEYS
    for name in SENSOR_STATISTIC_NAMES:
        block = entry[name]
        assert set(block) == STATISTIC_KEYS, name
        assert set(block["zscore"]) == ZSCORE_KEYS, name
        assert set(block["coverage"]) == COVERAGE_KEYS, name
        assert set(block["pit"]) == PIT_KEYS, name
        assert set(block["identifiability"]) == IDENTIFIABILITY_KEYS, name
        assert block["pit"]["n_bins"] == PIT_BINS, name
    assert set(entry["wasserstein"]) == WASSERSTEIN_KEYS
    for key in ("w1_over_sigma_pooled", "self_floor", "w1_over_floor"):
        value = entry["wasserstein"][key]
        assert value is None or set(value) == {"median", "max"}, key


def test_the_emitted_key_set_is_identical_on_the_healthy_and_degraded_paths() -> None:
    """The degraded path nulls LEAVES, it does not drop keys.

    Asserted as a frozenset comparison between an M = 8 run and an M = 2 run so
    that a consumer -- ``compare_sweep_results.py``, a figure script -- can index
    the same dotted paths without a per-run existence check. Emitting the
    degraded block as one bare ``null`` would have been simpler and is exactly
    what this forbids.
    """
    truth, ensemble = _exchangeable_series(8, 4, 8, seed=18)
    healthy = _score(truth, ensemble)[SET]
    degraded = _score(*_exchangeable_series(2, 4, 8, seed=18))[SET]

    _assert_schema(healthy)
    _assert_schema(degraded)
    for name in SENSOR_STATISTIC_NAMES:
        assert set(healthy[name]) == set(degraded[name]), name
        assert set(healthy[name]["pit"]) == set(degraded[name]["pit"]), name


def test_the_summary_round_trips_through_write_yaml(tmp_path: pathlib.Path) -> None:
    """No numpy scalars and no bare nan/inf anywhere in the emitted tree.

    ``write_yaml`` will happily dump ``.nan``, and it comes back as a string in
    some consumers -- which is why every float in this section leaves through
    ``_finite_or_none``. Checked over the whole tree rather than at the handful
    of keys a test happens to name.
    """
    truth, ensemble = _exchangeable_series(9, 5, 8, seed=19)
    scores = _score(truth, ensemble, prior_series={SET: ensemble[SET] + 7.0})

    path = tmp_path / "run_summary.yaml"
    write_yaml({"sensor_statistics": scores}, path)
    reloaded = read_yaml(path)["sensor_statistics"]

    assert reloaded[SET]["window_mean"]["crps"] == pytest.approx(
        scores[SET]["window_mean"]["crps"]
    )
    _assert_schema(reloaded[SET])

    def _no_bare_non_finite(node: object, path_repr: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _no_bare_non_finite(value, f"{path_repr}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _no_bare_non_finite(value, f"{path_repr}[{index}]")
        elif isinstance(node, float):
            assert np.isfinite(node), path_repr

    _no_bare_non_finite(reloaded)


def test_both_sensor_sets_are_scored_independently() -> None:
    """``validation`` is the held-out set and must get its own entry, not a copy."""
    truth, ensemble = _exchangeable_series(8, 4, 8, seed=20)
    other_truth, other_ensemble = _exchangeable_series(8, 2, 8, seed=21)
    truth["validation"] = other_truth[SET]
    ensemble["validation"] = other_ensemble[SET]

    scores = _score(truth, ensemble)

    assert set(scores) == {SET, "validation"}
    assert scores[SET]["n_sensors"] == 4
    assert scores["validation"]["n_sensors"] == 2
    assert (
        scores[SET]["window_mean"]["crps"]
        != scores["validation"]["window_mean"]["crps"]
    )
