"""WP1.3: window statistics as the verification object (metrics doc §4.2).

``sensor_metrics`` scores the *instantaneous* sensor series, and for turbulence
that is mostly a phase measurement: two members with identical parameters
decorrelate within an eddy turnover, so the pointwise error is dominated by
something no parameter estimate can control. What the parameters act on is the
statistics, so WP1.3 scores those instead. What is pinned here:

  * the windowing is by the **time coordinate**, not by frame count, because
    the truth and the assimilation routinely write at different cadences and
    the same reduction has to serve both (``..._bins_by_time_...``);
  * the block bootstrap is the sampling floor those statistics are read
    against. It lands on ``std/sqrt(n)`` when there is no correlation to
    preserve and rises well above it when there is -- the whole point, since
    the iid formula understates a turbulent series by 2-3x
    (``..._matches_the_iid_formula_...``, ``..._exceeds_the_iid_formula_...``);
  * its blocks are as long as the flow's own correlation time, and when the
    window is too short to hold enough of them the floor is **refused** rather
    than measured over blocks that are too short. A floor is a denominator, so
    one that is too small does not read as a worse estimate, it reads as a
    better verdict (``..._never_reports_a_floor_below_...``,
    ``..._blocks_grow_with_the_correlation_...``, ``..._refuses_...``);
  * the identifiability guard is the ratio of those two spreads. A statistic
    whose across-member spread does not clear its own sampling noise is not
    identifiable at this window length however good its CRPS looks
    (``..._flags_a_statistic_...``);
  * ranks are uniform for a calibrated ensemble and pile at an end for a biased
    one -- the shape check the CRPS and the z-score cannot make
    (``..._uniform_...``, ``..._pile_up_...``);
  * ``ensemble_sensor_series`` reads **one member at a time**. It used to
    ``.load()`` whole multi-GB window files, the last such site in the
    post-processing stack, and the rewrite has to be numerically inert
    (``..._streams_member_at_a_time``, ``..._is_inert``).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
import xarray
from evaluation.scores import ensemble_rank, window_statistics_summary
from evaluation.sensors import window_masks, window_sampling_std, window_statistics
from evaluation.turbulence import block_bootstrap_std, integral_time_scale

SIM_TIME = 10.0
NUM_WINDOWS = 3


def _series(
    rng: np.random.Generator,
    n_time: int,
    n_ensemble: int | None = None,
    n_sensors: int = 2,
    offset: float = 0.0,
) -> xarray.DataArray:
    """A ``(component, [ensemble,] time, sensor)`` series on a global time axis."""
    time = np.linspace(0.0, NUM_WINDOWS * SIM_TIME, n_time, endpoint=False)
    shape: tuple[int, ...] = (3, n_time, n_sensors)
    dims: tuple[str, ...] = ("component", "time", "sensor")
    if n_ensemble is not None:
        shape = (3, n_ensemble, n_time, n_sensors)
        dims = ("component", "ensemble", "time", "sensor")
    return xarray.DataArray(
        rng.normal(size=shape) + 2.0 + offset,
        dims=dims,
        coords={
            "component": ["u", "v", "w"],
            "time": time,
            "sensor": np.arange(n_sensors),
        },
    )


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def test_window_masks_bin_by_time_not_by_frame_count() -> None:
    # A boundary frame opens its window; anything past the last boundary (float
    # drift on the final frame) falls in the last window rather than off the end.
    times = np.array([0.0, 9.9, 10.0, 19.9, 20.0, 30.0])
    masks = window_masks(times, SIM_TIME, NUM_WINDOWS)

    assert [list(np.flatnonzero(m)) for m in masks] == [[0, 1], [2, 3], [4, 5]]


def test_window_masks_survive_float_drift_on_a_boundary() -> None:
    # The extraction rebases window w to start at exactly ``w*sim_time``, but
    # ``(w*sim_time)/sim_time`` is not exactly ``w`` in IEEE double for most
    # sim_time. At 10.76 (200 frames at pylbm's default 0.0538 s cadence)
    # windows 7 and 14 come out a ULP short, and a bare ``floor`` scores their
    # opening frame into the *previous* window -- silently, and not necessarily
    # for the same frame on the truth axis as on the ensemble axis.
    sim_time = 10.76
    starts = np.array([w * sim_time for w in range(16)])
    masks = window_masks(starts, sim_time, 16)

    assert [int(np.flatnonzero(m)[0]) for m in masks] == list(range(16))


def test_window_masks_reject_a_run_with_no_windows() -> None:
    with pytest.raises(ValueError, match="num_windows"):
        window_masks(np.array([0.0]), SIM_TIME, 0)


def test_window_statistics_bin_by_time_so_truth_and_ensemble_cadences_may_differ() -> (
    None
):
    # The truth writes 30 frames per window, the assimilation 9. Binning by frame
    # count would put the two on different horizons and silently compare window 0
    # of one against most of window 1 of the other.
    rng = np.random.default_rng(0)
    truth = _series(rng, 90)
    members = _series(rng, 27, n_ensemble=4)

    truth_stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS)
    member_stats = window_statistics(members, SIM_TIME, NUM_WINDOWS)

    assert truth_stats["mean"].sizes["window"] == NUM_WINDOWS
    assert member_stats["mean"].sizes["window"] == NUM_WINDOWS
    expected = truth.sel(component="u").isel(sensor=0).values[30:60].mean()
    assert truth_stats["mean"].sel(quantity="u", window=1).isel(
        sensor=0
    ).item() == pytest.approx(expected)


def test_window_statistics_variance_is_ddof_one() -> None:
    # ddof=1: the window variance estimates the flow's variance from a finite
    # window, not the window's own second moment. Truth and members are reduced
    # identically, so the choice cannot bias the comparison -- but it has to be
    # the same on both sides, which a hand-rolled reduction gets wrong.
    rng = np.random.default_rng(1)
    truth = _series(rng, 60)

    stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS)
    window0 = truth.sel(component="v").isel(sensor=0).values[:20]

    assert stats["variance"].sel(quantity="v", window=0).isel(
        sensor=0
    ).item() == pytest.approx(window0.var(ddof=1))


def test_window_statistics_carry_the_magnitude_beside_the_components() -> None:
    rng = np.random.default_rng(2)
    truth = _series(rng, 60)

    stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS)

    assert list(stats["mean"]["quantity"].values) == ["u", "v", "w", "magnitude"]
    magnitude = np.sqrt((truth**2).sum("component"))
    assert stats["mean"].sel(quantity="magnitude", window=0).isel(
        sensor=0
    ).item() == pytest.approx(magnitude.isel(sensor=0).values[:20].mean())


def test_window_statistics_null_a_window_with_no_frames(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Invariant 3: a truncated run costs the windows it lost, not the block.
    rng = np.random.default_rng(3)
    truth = _series(rng, 40).isel(time=slice(0, 20))  # only window 0 is covered

    with caplog.at_level("WARNING"):
        stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS, label="validation")

    assert np.isfinite(stats["mean"].sel(window=0).values).all()
    assert np.isnan(stats["mean"].sel(window=2).values).all()
    # Labelled, so a run with several sensor sets does not emit six
    # indistinguishable copies of this line.
    assert "validation: no sensor frames fall in window 2" in caplog.text


# ---------------------------------------------------------------------------
# The block bootstrap: the sampling floor
# ---------------------------------------------------------------------------


def _ar1(rng: np.random.Generator, n: int, phi: float, n_series: int = 1) -> np.ndarray:
    """AR(1) series of unit marginal variance."""
    noise = rng.normal(scale=np.sqrt(1.0 - phi**2), size=(n_series, n))
    out = np.empty((n_series, n))
    out[:, 0] = rng.normal(size=n_series)
    for t in range(1, n):
        out[:, t] = phi * out[:, t - 1] + noise[:, t]
    return out


def _true_mean_spread(n: int, phi: float) -> float:
    """Exact sd of the sample mean of a stationary AR(1) of unit variance.

    ``var(mean) = (n + 2*sum_k (n-k) phi^k) / n^2`` -- the reference the
    bootstrap is measured against, computed rather than resampled so the
    comparison is against the answer and not against another estimate of it.
    """
    lags = np.arange(1, n)
    return float(np.sqrt((n + 2.0 * np.sum((n - lags) * phi**lags)) / n**2))


def test_block_bootstrap_matches_the_iid_formula_on_independent_data() -> None:
    # Blocking costs nothing when there is no correlation to preserve, which is
    # what makes it safe to apply unconditionally.
    rng = np.random.default_rng(4)
    series = rng.normal(size=(8, 400))

    std = block_bootstrap_std(series, n_blocks=20, n_resamples=400)

    assert std.shape == (8,)
    assert np.median(std) == pytest.approx(1.0 / np.sqrt(400), rel=0.25)


def test_block_bootstrap_exceeds_the_iid_formula_on_correlated_data() -> None:
    # The reason this exists: a correlated series has far fewer independent
    # samples than frames, so std/sqrt(n) understates its sampling spread and
    # would make every statistic look identifiable.
    rng = np.random.default_rng(5)
    series = _ar1(rng, 400, phi=0.9, n_series=8)

    blocked = np.median(block_bootstrap_std(series, n_blocks=20, n_resamples=400))
    iid = np.median(series.std(axis=1, ddof=1) / np.sqrt(400))

    assert blocked > 2.0 * iid


@pytest.mark.parametrize(  # type: ignore[misc]
    ("n_time", "phi"),
    [
        (30, 0.0),  # barcelona's window length, uncorrelated
        (30, 0.9),  # barcelona's window length, a turbulent correlation time
        (60, 0.9),  # xie_and_castro's window length
        (200, 0.8),
        (480, 0.9),
    ],
)
def test_block_bootstrap_never_reports_a_floor_below_the_true_sampling_spread(
    n_time: int, phi: float
) -> None:
    # THE property, and the one the old fixed-block-count implementation broke:
    # the number this returns is a *floor* other code divides by, so a floor that
    # is too small is not a worse estimate, it is a wrong verdict. Measured
    # against the exact sd of an AR(1) sample mean -- an independent reference,
    # not another run of the estimator -- it may come back nan, but it may not
    # come back materially under the truth.
    #
    # Pre-fix this fails at the shipped window shapes: the block count was fixed
    # at 20, so L = ceil(n/20) = 2 at n=30 and 3 at n=60, both far under a
    # turbulent correlation time, and the reported floor was 0.24x (n=30,
    # phi=0.9) and 0.32x the truth.
    rng = np.random.default_rng(31)
    series = _ar1(rng, n_time, phi, n_series=400)
    true = _true_mean_spread(n_time, phi)
    # The algebra above, checked against the realizations actually resampled.
    assert np.std(series.mean(axis=1), ddof=1) == pytest.approx(true, rel=0.1)

    floor = np.median(block_bootstrap_std(series, n_resamples=400))

    if np.isnan(floor):
        return  # refused: the honest answer when the window is too short
    assert 0.65 * true <= floor <= 1.25 * true


def test_block_bootstrap_refuses_a_correlated_window_it_cannot_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The refusal has to be audible. Losing the floor means losing the
    # identifiability guard for that window, and the only place that can be
    # said is here -- the summary just omits the key.
    rng = np.random.default_rng(32)
    series = _ar1(rng, 30, phi=0.9, n_series=64)

    with caplog.at_level("WARNING"):
        floor = block_bootstrap_std(series, n_resamples=64)

    assert np.isnan(floor).all()
    assert "decorrelates over" in caplog.text
    assert "independent samples" in caplog.text


def test_block_bootstrap_blocks_grow_with_the_correlation_not_with_the_window() -> None:
    # The inversion at the heart of the defect: with the block *count* fixed,
    # L = ceil(n/20) shrinks as the window shortens, so the same flow sampled
    # over a shorter window got shorter blocks -- exactly backwards for a fixed
    # physical correlation time. The length must be set by the flow.
    rng = np.random.default_rng(33)
    correlated = _ar1(rng, 480, phi=0.9, n_series=32)
    independent = rng.normal(size=(32, 480))

    assert integral_time_scale(correlated) > 8.0 * integral_time_scale(independent)
    # Same n, same n_blocks: only the correlation differs, and the floor must
    # follow it rather than the record length.
    assert np.median(
        block_bootstrap_std(correlated, n_resamples=200)
    ) > 3.0 * np.median(block_bootstrap_std(independent, n_resamples=200))


def test_integral_time_scale_recovers_the_ar1_correlation_time() -> None:
    # tau = (1+phi)/(1-phi) for AR(1) -- an analytic reference, so this pins the
    # estimator against the answer rather than against itself. A long record,
    # because the estimate is knowingly biased low on a short one (the sample
    # autocorrelation of an n-frame series is biased by ~tau/n) and that bias is
    # what the refusal policy exists to absorb.
    rng = np.random.default_rng(34)

    for phi in (0.5, 0.8, 0.9):
        tau = integral_time_scale(_ar1(rng, 4000, phi, n_series=16))
        assert tau == pytest.approx((1.0 + phi) / (1.0 - phi), rel=0.15)


def test_integral_time_scale_is_one_when_nothing_decorrelates() -> None:
    # Both ends of the degenerate range: independent samples carry one
    # independent sample each, and a constant series has no correlation time at
    # all (its bootstrap is a measured zero at any block length).
    rng = np.random.default_rng(35)

    assert integral_time_scale(rng.normal(size=(64, 400))) == pytest.approx(
        1.0, abs=0.1
    )
    assert integral_time_scale(np.full((4, 50), 2.5)) == 1.0


def test_block_bootstrap_is_vectorized_over_leading_dims_not_approximated() -> None:
    # Every row shares one time axis and therefore one index matrix, so the fast
    # path is the same estimator -- not a batched approximation of it.
    rng = np.random.default_rng(6)
    series = rng.normal(size=(3, 5, 200))

    batched = block_bootstrap_std(series, n_blocks=10, n_resamples=64)
    per_row = np.array(
        [
            [
                float(block_bootstrap_std(series[i, j], n_blocks=10, n_resamples=64))
                for j in range(5)
            ]
            for i in range(3)
        ]
    )

    assert batched.shape == (3, 5)
    assert np.allclose(batched, per_row)


def test_block_bootstrap_chunking_moves_no_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The replicate axis is split to bound the temporary. Every shape the tests
    # use fits in one chunk, so without this the "changes no value" claim is
    # never exercised -- and the chunk loop only runs at production scale.
    from evaluation import turbulence

    rng = np.random.default_rng(23)
    series = rng.normal(size=(6, 120))
    unchunked = block_bootstrap_std(series, n_blocks=10, n_resamples=64)

    monkeypatch.setattr(turbulence, "_BOOTSTRAP_CHUNK_BYTES", 1)
    chunked = block_bootstrap_std(series, n_blocks=10, n_resamples=64)

    assert np.array_equal(unchunked, chunked)


def test_block_bootstrap_reproduces_without_a_seed() -> None:
    # Re-running the metric stage on the same run dir must reproduce
    # run_summary.yaml; the default must therefore not be OS entropy.
    rng = np.random.default_rng(7)
    series = rng.normal(size=(4, 100))

    assert np.array_equal(block_bootstrap_std(series), block_bootstrap_std(series))
    assert not np.array_equal(
        block_bootstrap_std(series), block_bootstrap_std(series, rng=1)
    )


def test_block_bootstrap_is_null_on_windows_too_short_to_block() -> None:
    # Routine, not exotic: even an uncorrelated record needs 15 frames (three
    # per block, five blocks) and the CI smoke shape has four frames per window,
    # so callers must handle the null.
    rng = np.random.default_rng(8)

    assert np.isnan(block_bootstrap_std(rng.normal(size=(2, 3)))).all()  # n < 4
    assert np.isnan(block_bootstrap_std(rng.normal(size=(2, 10)))).all()  # < 5 blocks
    # 15 is the shortest an *uncorrelated* record can be, and only when the
    # measured time scale comes out at exactly one sample: with a couple of rows
    # to pool over, its own scatter can put the length at four and cost the
    # floor a 15-frame window as well. It is a floor, not a promise.
    assert np.isfinite(block_bootstrap_std(rng.normal(size=(64, 15)))).all()


def test_block_bootstrap_nulls_only_the_rows_that_are_not_finite() -> None:
    # A diverged member must not blank the floor for the members beside it.
    rng = np.random.default_rng(9)
    series = rng.normal(size=(3, 100))
    series[1, 5] = np.nan

    std = block_bootstrap_std(series, n_blocks=10, n_resamples=64)

    assert np.isnan(std[1])
    assert np.isfinite(std[[0, 2]]).all()


def test_block_bootstrap_of_a_constant_series_is_an_exact_zero() -> None:
    # Every replicate of a constant series is that same constant, so the spread
    # really is zero. Exactly zero, not the 1e-17 of float rounding ``np.std``
    # leaves behind, which would survive the ``> 0`` filter downstream and turn
    # an identifiability ratio into ~1e17 instead of a clean null.
    std = block_bootstrap_std(np.full((2, 60), 3.25), n_resamples=16)

    assert np.array_equal(std, np.zeros(2))


def test_block_bootstrap_refuses_a_single_block_instead_of_reporting_zero() -> None:
    # ``n_blocks=1`` asks for blocks as long as the record. That is a point mass
    # -- one legal start, every replicate the original -- and a point mass is not
    # a measurement of anything, so it takes the same refusal path as any other
    # block count under five rather than reporting a floor of 0.0 that reads as
    # "this statistic has no sampling noise".
    rng = np.random.default_rng(10)

    std = block_bootstrap_std(rng.normal(size=(2, 60)), n_blocks=1, n_resamples=16)

    assert np.isnan(std).all()


def test_block_bootstrap_spread_over_replicates_is_the_ddof_one_one() -> None:
    # ddof=0 over the replicates rescales every floor by sqrt((R-1)/R): 0.25 % at
    # the 200 replicates production uses, which no statistical assertion can
    # see, and 29 % at the two used here. So it is pinned exactly, against the
    # closed form for two samples (|a - b|/sqrt(2)) rather than against another
    # call of the estimator.
    from evaluation.turbulence import _block_resample_indices, _bootstrap_block_length

    rng = np.random.default_rng(36)
    series = rng.normal(size=(3, 60))

    std = block_bootstrap_std(series, n_resamples=2)

    block_len = _bootstrap_block_length(series, 20)
    index = _block_resample_indices(60, block_len, 2, np.random.default_rng(0))
    replicates = series[:, index].mean(axis=-1)  # (3, 2)
    assert std == pytest.approx(
        np.abs(replicates[:, 0] - replicates[:, 1]) / np.sqrt(2.0)
    )


def test_block_bootstrap_rejects_a_statistic_that_ignores_its_axis() -> None:
    rng = np.random.default_rng(11)

    with pytest.raises(ValueError, match="reduce only the axis"):
        block_bootstrap_std(
            rng.normal(size=(2, 100)), statistic=lambda x, axis: np.mean(x)
        )


def test_window_sampling_std_matches_the_statistics_it_floors() -> None:
    # The floor has to be shaped exactly like the statistic it divides into, or
    # the identifiability ratio silently broadcasts across sensors. At the
    # SHIPPED settings: 90 frames over 3 windows is barcelona's 30 frames per
    # window, and the block count is the default rather than a test-only one, so
    # the production block length is what gets exercised.
    rng = np.random.default_rng(12)
    members = _series(rng, 90, n_ensemble=3)

    stats = window_statistics(members, SIM_TIME, NUM_WINDOWS)
    floor = window_sampling_std(members, SIM_TIME, NUM_WINDOWS, n_resamples=32)

    for name in ("mean", "variance"):
        assert floor[name].dims == stats[name].dims
        assert floor[name].shape == stats[name].shape
        assert np.isfinite(floor[name].values).all()


def test_window_sampling_std_nulls_the_shipped_window_when_the_flow_is_correlated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # End to end at the shape that shipped: barcelona's 300 s window at a 10 s
    # cadence is 30 frames, and 30 frames of a flow that decorrelates over ~19 of
    # them cannot measure their own sampling floor. Pre-fix this returned a
    # number ~4x too small at every sensor, which is what let the identifiability
    # guard pass silently; the floor is now absent and the reason is logged.
    rng = np.random.default_rng(37)
    n_time = NUM_WINDOWS * 30
    correlated = _ar1(rng, n_time, phi=0.9, n_series=3 * 4 * 2).reshape(3, 4, n_time, 2)
    members = xarray.DataArray(
        correlated,
        dims=("component", "ensemble", "time", "sensor"),
        coords={
            "component": ["u", "v", "w"],
            "time": np.linspace(0.0, NUM_WINDOWS * SIM_TIME, n_time, endpoint=False),
            "sensor": np.arange(2),
        },
    )

    with caplog.at_level("WARNING"):
        floor = window_sampling_std(members, SIM_TIME, NUM_WINDOWS, n_resamples=32)

    assert np.isnan(floor["mean"].values).all()
    assert "decorrelates over" in caplog.text


# ---------------------------------------------------------------------------
# Ranks
# ---------------------------------------------------------------------------


def test_ensemble_rank_counts_the_members_below_the_truth() -> None:
    members = np.array([[0.0], [1.0], [2.0], [3.0]])

    assert ensemble_rank(members, np.array([-1.0])) == 0
    assert ensemble_rank(members, np.array([1.5])) == 2
    assert ensemble_rank(members, np.array([9.0])) == 4


def test_ensemble_ranks_are_uniform_for_a_calibrated_ensemble() -> None:
    # The third calibration view: ranks use only the ordering, so they catch a
    # shape failure the CRPS and the z-score cannot see.
    rng = np.random.default_rng(13)
    n_members, n_knots = 8, 4000
    ranks = ensemble_rank(
        rng.normal(size=(n_members, n_knots)), rng.normal(size=n_knots), rng=rng
    )

    counts = np.bincount(ranks.astype(int), minlength=n_members + 1)
    expected = n_knots / (n_members + 1)
    assert np.abs(counts - expected).max() < 4.0 * np.sqrt(expected)


def test_ensemble_ranks_pile_up_when_the_ensemble_is_biased() -> None:
    rng = np.random.default_rng(14)
    n_knots = 2000
    ranks = ensemble_rank(
        rng.normal(size=(8, n_knots)) - 3.0, rng.normal(size=n_knots), rng=rng
    )

    assert (ranks == 8).mean() > 0.9


def test_ensemble_rank_ties_are_drawn_not_piled_at_one_end() -> None:
    # A fully collapsed ensemble matching the truth is all ties. A deterministic
    # tie-break would put every one of them at rank 0 (or M) and read as a
    # catastrophic bias that is not there.
    members = np.zeros((4, 2000))
    ranks = ensemble_rank(members, np.zeros(2000))

    assert set(np.unique(ranks)) == {0.0, 1.0, 2.0, 3.0, 4.0}
    assert ranks.mean() == pytest.approx(2.0, abs=0.15)


def test_ensemble_rank_is_null_where_a_member_is_not_finite() -> None:
    members = np.array([[0.0, 0.0], [1.0, np.nan]])

    ranks = ensemble_rank(members, np.array([0.5, 0.5]))

    assert ranks[0] == 1
    assert np.isnan(ranks[1])


# ---------------------------------------------------------------------------
# The summary block
# ---------------------------------------------------------------------------


def _summary(
    rng: np.random.Generator,
    n_members: int = 8,
    n_time: int = 120,
    with_prior: bool = True,
    prior_offset: float = 1.0,
) -> dict:
    truth = _series(rng, n_time)
    posterior = _series(rng, n_time, n_ensemble=n_members)
    prior = (
        _series(rng, n_time, n_ensemble=n_members, offset=prior_offset)
        if with_prior
        else None
    )
    # Spelled out at both call sites rather than ``**boot``: unpacking an
    # inferred ``dict[str, int]`` into a signature whose tail is ``label: str``
    # is not something mypy can check.
    boot_blocks, boot_resamples = 5, 32
    # ``dict(...)``: ``evaluation.scores`` is mypy-waived and returns ``Any``.
    return dict(
        window_statistics_summary(
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
            prior_stats=(
                window_statistics(prior, SIM_TIME, NUM_WINDOWS)
                if prior is not None
                else None
            ),
            posterior_sampling_std=window_sampling_std(
                posterior,
                SIM_TIME,
                NUM_WINDOWS,
                n_blocks=boot_blocks,
                n_resamples=boot_resamples,
            ),
            prior_sampling_std=(
                window_sampling_std(
                    prior,
                    SIM_TIME,
                    NUM_WINDOWS,
                    n_blocks=boot_blocks,
                    n_resamples=boot_resamples,
                )
                if prior is not None
                else None
            ),
        )
    )


def test_summary_scores_every_statistic_and_quantity_separately() -> None:
    # Separately identifiable: a parameter that fixes the mean wind while
    # leaving the resolved variance halved is exactly what this block names.
    summary = _summary(np.random.default_rng(15))

    assert set(summary["posterior"]) == {
        f"{stat}_{q}"
        for stat in ("mean", "variance")
        for q in ("u", "v", "w", "magnitude")
    }
    entry = summary["posterior"]["mean_magnitude"]
    assert set(entry) >= {"crps", "z_score", "rank_counts", "identifiability"}
    assert entry["crps"]["mean"] > 0
    # One count per rank 0..M; they total the scored knots (windows x sensors).
    assert len(entry["rank_counts"]) == summary["n_members"] + 1
    assert sum(entry["rank_counts"]) == NUM_WINDOWS * 2


def test_summary_scores_the_prior_and_the_skill_against_it() -> None:
    # A prior displaced from the truth must score worse, so the skill is positive.
    summary = _summary(np.random.default_rng(16), prior_offset=3.0)

    assert set(summary["prior"]) == set(summary["posterior"])
    entry = summary["posterior"]["mean_u"]
    assert entry["prior_crps_mean"] > entry["crps"]["mean"]
    assert entry["crps_reduction_vs_prior"] > 0.5


def test_summary_no_ops_on_the_prior_when_it_was_not_saved() -> None:
    # run.save_prior_state is off by default, so absence is the common case.
    summary = _summary(np.random.default_rng(17), with_prior=False)

    assert "prior" not in summary
    assert "crps_reduction_vs_prior" not in summary["posterior"]["mean_u"]


def test_summary_identifiability_flags_a_statistic_the_window_cannot_resolve(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Members that differ only by their own turbulent noise: the across-member
    # spread is the sampling spread, so the ratio sits near 1 and the statistic
    # carries no parameter information however good its CRPS reads.
    with caplog.at_level("WARNING"):
        summary = _summary(np.random.default_rng(18), n_members=6)

    assert summary["posterior"]["mean_u"]["identifiability"]["mean"] < 3.0
    assert "not identifiable at this window length" in caplog.text


def test_summary_identifiability_rises_when_the_members_genuinely_differ() -> None:
    # The direction check the noise-only case cannot make: with members offset
    # well beyond their own sampling spread the ratio must go UP. Inverted
    # (floor/spread) it would go down, and the "< 3" test below still passes.
    rng = np.random.default_rng(24)
    truth = _series(rng, 120)
    members = xarray.concat(
        [_series(rng, 120) + offset for offset in np.linspace(-3.0, 3.0, 6)],
        dim="ensemble",
    ).transpose("component", "ensemble", "time", "sensor")

    summary = window_statistics_summary(
        window_statistics(truth, SIM_TIME, NUM_WINDOWS),
        window_statistics(members, SIM_TIME, NUM_WINDOWS),
        posterior_sampling_std=window_sampling_std(
            members, SIM_TIME, NUM_WINDOWS, n_blocks=5, n_resamples=32
        ),
    )

    assert summary["posterior"]["mean_u"]["identifiability"]["mean"] > 3.0


def test_summary_is_omitted_when_the_members_axis_is_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An ensemble-mean-only artifact has nothing to score probabilistically.
    # Invariant 3: the block goes away, the metric stage does not.
    rng = np.random.default_rng(25)
    truth = _series(rng, 60)

    with caplog.at_level("WARNING"):
        summary = window_statistics_summary(
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            label="assimilation",
        )

    assert summary == {}
    assert "no ensemble dimension" in caplog.text


def test_summary_z_score_is_signed_toward_the_truth() -> None:
    # The sign of z_score.mean is the ONLY thing that says "the posterior sits
    # below the truth" rather than above it. z = (truth - mean)/sigma, so a
    # posterior displaced downward must give a positive mean; written the other
    # way round every other assertion in this file still passes.
    rng = np.random.default_rng(26)
    truth = _series(rng, 120)
    posterior = _series(rng, 120, n_ensemble=8, offset=-2.0)

    summary = window_statistics_summary(
        window_statistics(truth, SIM_TIME, NUM_WINDOWS),
        window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
    )

    assert summary["posterior"]["mean_u"]["z_score"]["mean"] > 1.0


def test_summary_z_score_denominator_is_the_ddof_one_spread() -> None:
    # ddof=0 would scale every z by sqrt(M/(M-1)) -- 1.41 at M=2, 1.07 at M=8 --
    # and silently break the expected_std reference, which is derived for
    # t_(M-1). Four members at fixed offsets from the truth make z exact:
    # member means are truth+{0,2,4,6}, so the ensemble mean is truth+3 and the
    # ddof=1 spread is sqrt(20/3); ddof=0 would give sqrt(20/4) and a z of
    # -1.342 instead of -1.108.
    rng = np.random.default_rng(27)
    truth = _series(rng, 60, n_sensors=1)
    members = xarray.concat(
        [truth + offset for offset in (0.0, 2.0, 4.0, 6.0)], dim="ensemble"
    ).transpose("component", "ensemble", "time", "sensor")

    summary = window_statistics_summary(
        window_statistics(truth, SIM_TIME, NUM_WINDOWS),
        window_statistics(members, SIM_TIME, NUM_WINDOWS),
    )

    z = summary["posterior"]["mean_u"]["z_score"]
    assert z["mean"] == pytest.approx(-3.0 / np.sqrt(20.0 / 3.0))
    assert z["std"] == pytest.approx(0.0, abs=1e-12)  # identical at every knot


def test_summary_identifiability_is_measured_per_sensor() -> None:
    # The floor is reduced over MEMBERS, per sensor and window. Reduced over the
    # wrong axis it would be constant across sensors, and every test built on
    # homogeneous synthetic data still passes. Two sensors, same member spread,
    # 10x the within-member noise on the second: only its ratio may collapse.
    rng = np.random.default_rng(28)
    n_time = 160
    time = np.linspace(0.0, NUM_WINDOWS * SIM_TIME, n_time, endpoint=False)
    noise = rng.normal(size=(3, 6, n_time, 2)) * np.array([0.1, 10.0])
    offsets = np.linspace(-2.0, 2.0, 6).reshape(1, 6, 1, 1)
    members = xarray.DataArray(
        noise + offsets,
        dims=("component", "ensemble", "time", "sensor"),
        coords={"component": ["u", "v", "w"], "time": time, "sensor": [0, 1]},
    )

    stats = window_statistics(members, SIM_TIME, NUM_WINDOWS)["mean"].sel(quantity="u")
    floor = window_sampling_std(
        members, SIM_TIME, NUM_WINDOWS, n_blocks=8, n_resamples=32
    )["mean"].sel(quantity="u")
    ratio = stats.std("ensemble", ddof=1) / floor.median("ensemble")

    quiet, noisy = float(ratio.isel(sensor=0).mean()), float(
        ratio.isel(sensor=1).mean()
    )
    assert quiet > 3.0 > noisy


def test_summary_rank_counts_exclude_the_knots_that_have_no_rank() -> None:
    # A window with no frames has no rank. Counting it as rank 0 (what
    # ``nan_to_num`` would do) piles unrankable knots at one end and reads as a
    # catastrophic bias in figure D1.
    rng = np.random.default_rng(29)
    truth = _series(rng, 90).isel(time=slice(0, 60))  # window 2 never ran
    posterior = _series(rng, 90, n_ensemble=4).isel(time=slice(0, 60))

    summary = window_statistics_summary(
        window_statistics(truth, SIM_TIME, NUM_WINDOWS),
        window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
    )

    counts = summary["posterior"]["mean_u"]["rank_counts"]
    assert sum(counts) == (NUM_WINDOWS - 1) * 2  # the two windows that ran


def test_summary_nulls_the_skill_when_prior_and_posterior_span_different_windows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A prior state file that exists but stops short leaves its last window
    # empty. Averaging a 2-window prior against a 3-window posterior puts two
    # horizons in one skill score.
    rng = np.random.default_rng(30)
    truth = _series(rng, 90)
    posterior = _series(rng, 90, n_ensemble=4)
    prior = _series(rng, 90, n_ensemble=4).isel(time=slice(0, 60))

    with caplog.at_level("WARNING"):
        summary = window_statistics_summary(
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
            prior_stats=window_statistics(prior, SIM_TIME, NUM_WINDOWS),
        )

    entry = summary["posterior"]["mean_u"]
    assert entry["crps_reduction_vs_prior"] is None
    assert entry["prior_crps_mean"] is None
    assert "two different horizons" in caplog.text


def test_summary_identifiability_is_absent_when_no_floor_was_measured() -> None:
    # Short windows give no bootstrap, and a ratio over an unmeasured floor is
    # not "infinitely identifiable" -- it is unknown. 10 frames per window is
    # under the 15 an uncorrelated record needs to be blocked at all.
    rng = np.random.default_rng(19)
    truth = _series(rng, 30)
    posterior = _series(rng, 30, n_ensemble=4)

    summary = window_statistics_summary(
        window_statistics(truth, SIM_TIME, NUM_WINDOWS),
        window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
        posterior_sampling_std=window_sampling_std(posterior, SIM_TIME, NUM_WINDOWS),
    )

    assert "identifiability" not in summary["posterior"]["mean_u"]


def test_summary_degrades_on_the_smoke_shape_rather_than_inventing_numbers() -> None:
    # M=2: z is Cauchy, so no moment of it converges and the whole z_score entry
    # is null by policy (WP1.2). The CRPS and the ranks stay well defined.
    summary = _summary(np.random.default_rng(20), n_members=2, n_time=12)

    entry = summary["posterior"]["mean_magnitude"]
    assert summary["n_members"] == 2
    assert entry["z_score"] is None
    assert entry["crps"]["mean"] >= 0
    assert len(entry["rank_counts"]) == 3  # ranks 0, 1, 2 at M=2


def test_summary_survives_a_window_whose_frames_are_missing() -> None:
    # Invariant 3 end to end: a truncated run nulls the windows it lost and
    # scores the ones it has, silently (no numpy all-NaN-slice warnings) and
    # without taking the block down.
    rng = np.random.default_rng(22)
    truth = _series(rng, 90).isel(time=slice(0, 60))  # window 2 never ran
    posterior = _series(rng, 90, n_ensemble=4).isel(time=slice(0, 60))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = window_statistics_summary(
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
        )

    crps = summary["posterior"]["mean_u"]["crps"]
    assert crps["final"] is None  # the missing last window
    assert crps["mean"] > 0  # the two that ran


def test_summary_is_yaml_safe(tmp_path: pathlib.Path) -> None:
    # The block is written straight to run_summary.yaml through the pipeline's
    # own writer; a numpy scalar surviving into it would raise there (safe_dump
    # has no representer for one), so this goes through the real round trip
    # rather than asserting about types.
    from scripts.esmda._esmda_common import read_yaml, write_yaml

    summary = _summary(np.random.default_rng(21))
    path = tmp_path / "run_summary.yaml"
    write_yaml({"sensor_statistics": {"assimilation": summary}}, path)

    assert "!!python" not in path.read_text()
    assert read_yaml(path)["sensor_statistics"]["assimilation"]["n_members"] == 8


# ---------------------------------------------------------------------------
# The streaming extraction this block is built on
# ---------------------------------------------------------------------------


def _window_state_file(
    tmp_path: pathlib.Path,
    window: int,
    n_ensemble: int = 3,
    n_time: int = 4,
    n_cells: int = 5,
) -> pathlib.Path:
    """A ``(ensemble, time, z, y, x)`` window state file on a pylbm-shaped grid."""
    rng = np.random.default_rng(100 + window)
    axis = np.linspace(0.0, 20.0, n_cells)
    shape = (n_ensemble, n_time, n_cells, n_cells, n_cells)
    ds = xarray.Dataset(
        {
            name: (("ensemble", "time", "z", "y", "x"), rng.normal(size=shape))
            for name in ("u", "v", "w")
        },
        coords={
            "ensemble": np.arange(n_ensemble),
            "time": np.arange(n_time, dtype=float),
            "z": axis,
            "y": axis,
            "x": axis,
        },
    )
    path = tmp_path / f"window_{window}_posterior_state.nc"
    ds.to_netcdf(path)
    return path


_SENSOR_SETS = {
    "assimilation": (
        np.array([3.0, 11.0]),
        np.array([4.0, 12.0]),
        np.array([5.0, 6.0]),
    ),
    "validation": (np.array([7.0]), np.array([8.0]), np.array([9.0])),
}


def test_ensemble_sensor_series_streams_member_at_a_time(
    tmp_path: pathlib.Path,
) -> None:
    # Master-plan invariant 2: full-ensemble window state files are gigabytes and
    # must never be read whole. This was the last site that did.
    from scripts.esmda import _esmda_common

    paths = [_window_state_file(tmp_path, w) for w in range(2)]
    seen: list[int | None] = []
    original = _esmda_common._sensor_component_timeseries

    def _spy(state: xarray.Dataset, *args: object, **kwargs: object) -> object:
        seen.append(state.sizes.get("ensemble"))
        return original(state, *args, **kwargs)

    _esmda_common._sensor_component_timeseries = _spy
    try:
        _esmda_common.ensemble_sensor_series(paths, _SENSOR_SETS, "pylbm", 4.0)
    finally:
        _esmda_common._sensor_component_timeseries = original

    # 2 windows x 3 members x 2 sensor sets, never more than one member at a time.
    assert seen == [1] * 12


def test_ensemble_sensor_series_preserves_the_member_axis(
    tmp_path: pathlib.Path,
) -> None:
    # The per-member slices are concatenated back; a lost or reordered
    # ``ensemble`` coord would silently re-label every member's scores.
    from scripts.esmda._esmda_common import ensemble_sensor_series

    paths = [_window_state_file(tmp_path, w) for w in range(2)]
    series = ensemble_sensor_series(paths, _SENSOR_SETS, "pylbm", 4.0)["assimilation"]

    assert series.sizes["ensemble"] == 3
    assert list(series["ensemble"].values) == [0, 1, 2]


def test_ensemble_sensor_series_is_inert_under_the_streaming_rewrite(
    tmp_path: pathlib.Path,
) -> None:
    # The rewrite is a memory change and nothing else: the same numbers, the same
    # dims, the same global time axis, in the same member order.
    from scripts.esmda._esmda_common import (
        _sensor_component_timeseries,
        ensemble_sensor_series,
    )

    sim_time = 4.0
    paths = [_window_state_file(tmp_path, w) for w in range(2)]
    streamed = ensemble_sensor_series(paths, _SENSOR_SETS, "pylbm", sim_time)

    for name, (ox, oy, oz) in _SENSOR_SETS.items():
        # The pre-WP1.3 implementation: open each window whole, interpolate once.
        pieces = []
        for w, path in enumerate(paths):
            whole = xarray.open_dataset(path).load()
            t = np.asarray(whole["time"].values, dtype=float)
            vel = _sensor_component_timeseries(whole, ox, oy, oz, "pylbm")
            pieces.append(vel.assign_coords(time=(t - t[0]) + w * sim_time))
            whole.close()
        expected = xarray.concat(pieces, dim="time", join="override")

        assert streamed[name].dims == expected.dims
        assert np.array_equal(streamed[name].values, expected.values)
        assert np.array_equal(streamed[name]["time"].values, expected["time"].values)


# ---------------------------------------------------------------------------
# run_summary.yaml wiring
# ---------------------------------------------------------------------------

_RUN_WINDOWS = 2
_RUN_SIM_TIME = 4.0
_RUN_FRAMES = 6  # per window


def _state_dataset(
    n_ensemble: int | None, n_time: int, seed: int, n_cells: int = 5
) -> xarray.Dataset:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 20.0, n_cells)
    dims: tuple[str, ...] = ("time", "z", "y", "x")
    shape: tuple[int, ...] = (n_time, n_cells, n_cells, n_cells)
    coords = {
        "time": np.arange(n_time, dtype=float) * (_RUN_SIM_TIME / n_time),
        "z": axis,
        "y": axis,
        "x": axis,
    }
    if n_ensemble is not None:
        dims = ("ensemble",) + dims
        shape = (n_ensemble,) + shape
        coords["ensemble"] = np.arange(n_ensemble)
    return xarray.Dataset(
        {n: (dims, rng.normal(size=shape) + 2.0) for n in ("u", "v", "w")},
        coords=coords,
    )


def _full_run_dir(
    tmp_path: pathlib.Path, n_members: int = 4, save_prior_state: bool = False
) -> pathlib.Path:
    """An ESMDA run dir complete enough for the non-``skip_viz`` metric path."""
    from scripts.esmda._esmda_common import write_yaml

    run_dir = tmp_path / "run"
    (run_dir / "windows").mkdir(parents=True)

    write_yaml(
        {
            "run": {"skip_viz": False},
            "obs": {
                "mode": "points",
                "x_points": [4.0, 12.0],
                "y_points": [5.0, 13.0],
                "z_points": [6.0, 7.0],
                "validation_x_points": [8.0],
                "validation_y_points": [9.0],
                "validation_z_points": [10.0],
            },
        },
        run_dir / "config.yaml",
    )
    write_yaml(
        {"configuration": {"ensemble_size": n_members}}, run_dir / "run_info.yaml"
    )

    # Truth: one global time axis across both windows.
    truth = _state_dataset(None, _RUN_WINDOWS * _RUN_FRAMES, seed=1)
    truth = truth.assign_coords(
        time=np.linspace(
            0.0,
            _RUN_WINDOWS * _RUN_SIM_TIME,
            _RUN_WINDOWS * _RUN_FRAMES,
            endpoint=False,
        )
    )
    truth.to_netcdf(run_dir / "truth_state.nc")
    write_yaml(
        {
            "true_state_path": str(run_dir / "truth_state.nc"),
            "n_total": _RUN_WINDOWS * _RUN_FRAMES,
            "x_offset": 0.0,
            "start_idx": 0,
            "t_offset": 0.0,
            "num_windows": _RUN_WINDOWS,
            "n_per_window": _RUN_FRAMES,
            "sim_time": _RUN_SIM_TIME,
            "truth_solver_name": "pylbm",
            "assim_solver_name": "pylbm",
        },
        run_dir / "truth_access.yaml",
    )

    for w in range(_RUN_WINDOWS):
        _state_dataset(n_members, _RUN_FRAMES, seed=10 + w).to_netcdf(
            run_dir / "windows" / f"window_{w}_posterior_state.nc"
        )
        if save_prior_state:
            _state_dataset(n_members, _RUN_FRAMES, seed=50 + w).to_netcdf(
                run_dir / "windows" / f"window_{w}_prior_state.nc"
            )
    _state_dataset(None, _RUN_WINDOWS * _RUN_FRAMES, seed=20).to_netcdf(
        run_dir / "posterior_state_mean.nc"
    )

    members = np.linspace(-1.0, 1.0, n_members)
    for name, values in (
        ("posterior_params", members),
        ("prior_params", members * 4.0),
    ):
        xarray.Dataset(
            {"inflow_angle": (("ensemble", "time"), np.repeat(values[:, None], 2, 1))},
            coords={"ensemble": np.arange(n_members), "time": [0.0, 1.0]},
        ).to_netcdf(run_dir / f"{name}.nc")
    xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    ).to_netcdf(run_dir / "true_params.nc")
    return run_dir


def test_run_summary_carries_sensor_statistics_for_every_sensor_set(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    block = summary["sensor_statistics"]
    assert set(block) == {"assimilation", "validation"}
    assimilation = block["assimilation"]
    assert assimilation["n_members"] == 4
    assert assimilation["n_windows"] == _RUN_WINDOWS
    assert assimilation["num_sensors"] == 2
    assert block["validation"]["num_sensors"] == 1
    # Held out from the update but scored all the same -- the point of §4.2.
    assert block["validation"]["posterior"]["mean_magnitude"]["crps"]["mean"] > 0

    # Additive: WP1.1/WP1.2 keys are untouched.
    assert summary["metrics_version"] == 2
    assert "parameter_metrics" in summary and "sensor_metrics" in summary


def test_run_summary_survives_state_files_without_a_members_axis(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # An ensemble-mean-only artifact. Both sensor blocks are probabilistic and
    # need the members -- but the guard has to be at the SCRIPT level, because
    # `vector_sensor_metrics` runs first and would raise before the library's
    # own guard is ever reached. Invariant 3: the set is dropped, the file is
    # still written, and everything already computed survives.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path)
    for w in range(_RUN_WINDOWS):
        _state_dataset(None, _RUN_FRAMES, seed=10 + w).to_netcdf(
            run_dir / "windows" / f"window_{w}_posterior_state.nc"
        )

    with caplog.at_level("WARNING"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "No ensemble dimension in the assimilation sensor series" in caplog.text
    assert summary["sensor_metrics"] == {}
    assert summary["sensor_statistics"] == {}
    assert summary["parameter_metrics"] and summary["state_metrics"]


def test_run_summary_sensor_statistics_no_op_without_saved_prior_states(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path, save_prior_state=False)
    with caplog.at_level("INFO"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "prior" not in summary["sensor_statistics"]["assimilation"]
    assert "No prior sensor statistics" in caplog.text


def test_run_summary_survives_an_unreadable_prior_state_file(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A job killed mid-``to_netcdf`` leaves a prior state file that exists but
    # cannot be opened. Invariant 3: that costs the prior half of one block, not
    # run_summary.yaml -- which would take the parameter, state and sensor
    # metrics down with it.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path, save_prior_state=True)
    (run_dir / "windows" / "window_1_prior_state.nc").write_bytes(b"not a netcdf")

    with caplog.at_level("WARNING"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "Cannot read the prior sensor series" in caplog.text
    assert "prior" not in summary["sensor_statistics"]["assimilation"]
    # Everything the stage had already computed survives.
    assert summary["parameter_metrics"] and summary["sensor_metrics"]
    assert summary["sensor_statistics"]["assimilation"]["posterior"]


def test_run_summary_sensor_statistics_score_the_prior_when_it_was_saved(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path, save_prior_state=True)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    posterior = summary["sensor_statistics"]["assimilation"]["posterior"]
    assert set(summary["sensor_statistics"]["assimilation"]["prior"]) == set(posterior)
    assert "crps_reduction_vs_prior" in posterior["mean_u"]
