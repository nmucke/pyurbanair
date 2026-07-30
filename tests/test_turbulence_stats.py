"""Property tests for the shared flow-statistics helpers.

No snapshots: every assertion is a closed form, an inequality that states why
the naive alternative is wrong, or a seeded Monte-Carlo property of a
synthetic series whose correlation structure is known by construction. The
smoke shape (three frames per window) is exercised explicitly, because it is
the shape that pushes ``block_bootstrap_std`` onto its undefined path.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyurbanair.utils.turbulence_stats import (
    block_bootstrap_std,
    block_bootstrap_std_batch,
)


def _ar1(n_samples: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) with unit marginal variance: a series with a correlation time.

    ``phi = 0`` is iid, ``phi = 0.9`` has an integrated correlation time of
    ~19 samples, so a 400-sample series carries ~20 independent ones.
    """
    noise = rng.normal(scale=np.sqrt(1.0 - phi**2), size=n_samples)
    series = np.empty(n_samples)
    series[0] = rng.normal()
    for i in range(1, n_samples):
        series[i] = phi * series[i - 1] + noise[i]
    return series


def test_block_bootstrap_std_of_an_iid_mean_matches_the_closed_form() -> None:
    """On independent data, blocking must cost nothing.

    The standard error of a mean over ``n`` iid samples is ``std/sqrt(n)``.
    A block bootstrap that missed this would be preserving a correlation that
    is not there, i.e. inflating every sampling floor it is asked for.
    """
    rng = np.random.default_rng(2)
    ratios = [
        block_bootstrap_std(series, rng=rng)
        / (float(np.std(series, ddof=1)) / np.sqrt(series.size))
        for series in (rng.normal(size=400) for _ in range(60))
    ]

    # measured over 200 series: median 0.965, 5th-95th pct 0.78-1.21. The
    # per-series spread is the reason this is a median, not a single draw.
    assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.15)


def test_block_bootstrap_std_exceeds_the_iid_formula_for_a_correlated_series() -> None:
    """The whole point of blocking, asserted as an inequality rather than a value.

    ``std/sqrt(n)`` treats every sample as independent, so on a correlated
    probe series it understates the sampling spread of a window statistic by a
    large factor -- which is precisely the factor that decides whether an
    ensemble's across-member spread is signal or sampling noise. The exact
    multiplier depends on the correlation time and on how much of it fits in a
    block, so only the direction and the order of magnitude are pinned.
    """
    rng = np.random.default_rng(3)
    series = _ar1(400, 0.9, rng)
    iid_formula = float(np.std(series, ddof=1)) / np.sqrt(series.size)

    blocked = block_bootstrap_std(series, rng=rng)

    # measured over 200 series: ratio median 3.20, min 2.34 -- and the true
    # sampling spread of the mean at this phi is ~4.9x the iid formula, so the
    # bootstrap is still a LOWER bound. It closes most of a gap the iid
    # formula does not see at all.
    assert blocked > 1.5 * iid_formula
    # ... and on the same generator with no correlation it does not inflate.
    white = _ar1(400, 0.0, rng)
    assert block_bootstrap_std(white, rng=rng) < 1.5 * (
        float(np.std(white, ddof=1)) / np.sqrt(white.size)
    )


def test_block_bootstrap_std_is_nan_at_the_smoke_shape() -> None:
    """Three frames against twenty blocks is undefined, and says so.

    ``L = ceil(3/20) = 1`` is point resampling, which is the iid answer wearing
    a block bootstrap's name -- returning it would silently hand the caller a
    number with no correlation structure in it. This path is routine, not
    exotic: it fires on every smoke-shaped run.
    """
    assert np.isnan(block_bootstrap_std(np.array([1.0, 2.0, 3.0])))
    assert np.isnan(block_bootstrap_std(np.arange(20.0)))  # L = 1, still degenerate
    # 21 samples is where the default 20 blocks first gives L = 2.
    assert np.isfinite(block_bootstrap_std(np.linspace(0.0, 1.0, 21)))
    # A production window (36 frames) works at the default, and a short window
    # works if the caller asks for fewer, longer blocks.
    assert np.isfinite(block_bootstrap_std(np.linspace(0.0, 1.0, 36)))
    assert np.isfinite(block_bootstrap_std(np.arange(8.0), n_blocks=4))
    # Fewer than four finite samples has no bootstrap at any block count.
    assert np.isnan(block_bootstrap_std(np.arange(3.0), n_blocks=1))


def test_block_bootstrap_std_drops_non_finite_samples_before_blocking() -> None:
    """A gap must not become a block full of ``nan``.

    Masked frames are real (a sensor inside a building), and a block spanning
    a gap is still a contiguous stretch of the samples that exist.
    """
    rng = np.random.default_rng(4)
    clean = rng.normal(size=40)
    gappy = np.concatenate([clean[:20], [np.nan, np.inf], clean[20:]])

    assert block_bootstrap_std(gappy, rng=0) == pytest.approx(
        block_bootstrap_std(clean, rng=0)
    )
    assert np.isnan(block_bootstrap_std(np.full(40, np.nan)))


def test_block_bootstrap_std_is_reproducible_under_a_fixed_seed() -> None:
    """Re-running the metrics stage on the same run directory must not move numbers.

    The default is a fixed module seed rather than OS entropy, so even the
    no-argument call is reproducible -- the failure mode this guards against is
    a summary file whose bootstrap columns change on every re-run.
    """
    rng = np.random.default_rng(5)
    series = _ar1(200, 0.7, rng)

    assert block_bootstrap_std(series) == block_bootstrap_std(series)
    assert block_bootstrap_std(series, rng=7) == block_bootstrap_std(series, rng=7)
    assert block_bootstrap_std(series, rng=7) != block_bootstrap_std(series, rng=8)
    # An explicit generator is accepted too, and advances -- so a caller
    # looping over members gets independent resampling, not one repeated draw.
    generator = np.random.default_rng(0)
    first = block_bootstrap_std(series, rng=generator)
    assert first != block_bootstrap_std(series, rng=generator)


def test_block_bootstrap_std_accepts_any_statistic_not_just_the_mean() -> None:
    """The sensor statistics it feeds are means, variances and a TKE closure."""
    rng = np.random.default_rng(6)
    series = _ar1(400, 0.7, rng)

    for statistic in (np.mean, np.var, np.median, lambda x: float(np.ptp(x))):
        value = block_bootstrap_std(series, statistic=statistic, rng=0)
        assert np.isfinite(value) and value > 0.0

    # A constant statistic has no sampling spread at all -- zero, not nan.
    assert block_bootstrap_std(series, statistic=lambda x: 1.0, rng=0) == 0.0


def test_block_bootstrap_std_shape_arguments_reject_nonsense() -> None:
    series = np.linspace(0.0, 1.0, 50)

    with pytest.raises(ValueError):
        block_bootstrap_std(series, n_blocks=0)
    with pytest.raises(ValueError):
        block_bootstrap_std(series, n_resamples=1)  # a spread needs two replicates


def test_block_bootstrap_std_replicates_have_the_length_of_the_original() -> None:
    """Truncating to ``n`` keeps sample-count-dependent statistics comparable.

    ``ceil(n/L)`` blocks of length ``L`` overshoot whenever ``L`` does not
    divide ``n``; without the truncation the replicates would be longer than
    the original and a statistic like a variance would be evaluated at the
    wrong sample size. Here ``n = 50`` with 7 blocks gives ``L = 8`` and
    ``ceil(50/8) = 7`` blocks, i.e. 56 samples before truncation.
    """
    seen: list[int] = []

    def record_length(values: np.ndarray) -> float:
        seen.append(int(values.size))
        return float(np.mean(values))

    block_bootstrap_std(
        np.linspace(0.0, 1.0, 50), statistic=record_length, n_blocks=7, n_resamples=5
    )

    assert seen == [50] * 5


def test_block_bootstrap_std_is_exactly_zero_for_a_single_block() -> None:
    """``n_blocks=1`` is a measured zero, not an undefined result -- and not a raise.

    One block spans the whole series, so there is exactly one legal start
    position and every replicate *is* the original series: the spread over them
    really is zero. ``bootstrap_blocks: 1`` passes ``resolve_metrics_settings``'
    ``>= 1`` check, so this is reachable from a YAML knob; raising here would
    move a cheap config mistake into the middle of a streaming pass that has
    already read gigabytes, which is what that validator exists to avoid. The
    WP1.2 consumer's ``within > 0`` filter turns the zero into a clean null.
    """
    rng = np.random.default_rng(11)
    series = _ar1(60, 0.7, rng)

    assert block_bootstrap_std(series, n_blocks=1) == 0.0
    assert block_bootstrap_std(series, statistic=np.var, n_blocks=1) == 0.0
    # Same for the batch form, per row.
    batch = block_bootstrap_std_batch(np.stack([series, 2.0 * series]), n_blocks=1)
    assert batch.tolist() == [0.0, 0.0]


def test_block_bootstrap_std_batch_of_one_row_equals_the_scalar_exactly() -> None:
    """The load-bearing property: the fast path IS the documented estimator.

    Not "close" -- bit-for-bit equal, at the same seed, because both functions
    build their replicates from the same private index matrix. Without this a
    reader has to take on trust that the vectorized rewrite kept the estimator;
    with it, every property proved of ``block_bootstrap_std`` above transfers.
    """
    rng = np.random.default_rng(12)
    for n_samples, phi, n_blocks, seed in ((36, 0.9, 20, 0), (200, 0.7, 13, 5)):
        series = _ar1(n_samples, phi, rng)
        for statistic in (np.mean, np.var):
            scalar = block_bootstrap_std(
                series, statistic=statistic, n_blocks=n_blocks, rng=seed
            )
            batch = block_bootstrap_std_batch(
                series[None, :], statistic=statistic, n_blocks=n_blocks, rng=seed
            )
            assert batch.shape == (1,)
            assert batch[0] == scalar  # exact, not approx


def test_block_bootstrap_std_batch_equals_a_scalar_loop_row_by_row() -> None:
    """A whole batch reproduces the double loop it replaces, row for row.

    All rows share one time axis, so they share one index matrix -- which is
    the same matrix the scalar function builds from that seed. So the batch is
    not merely a good approximation of the loop WP1.2 used to run, it is the
    same arithmetic in a different order.
    """
    rng = np.random.default_rng(13)
    rows = np.stack([_ar1(48, phi, rng) for phi in (0.0, 0.5, 0.9, 0.95, 0.7)])

    batch = block_bootstrap_std_batch(rows, rng=3)
    scalar = np.array([block_bootstrap_std(row, rng=3) for row in rows])

    assert np.array_equal(batch, scalar)
    # ... including for an axis-taking closure, which is how WP1.2 passes a
    # ddof=1 variance (the scalar equivalent takes a flat series).
    batch_var = block_bootstrap_std_batch(
        rows, statistic=lambda x, axis: np.var(x, axis=axis, ddof=1), rng=3
    )
    scalar_var = np.array(
        [
            block_bootstrap_std(row, statistic=lambda x: np.var(x, ddof=1), rng=3)
            for row in rows
        ]
    )
    assert np.array_equal(batch_var, scalar_var)


def test_block_bootstrap_std_batch_nans_rows_with_gaps_and_keeps_the_rest() -> None:
    """A gappy row is refused, not silently approximated -- and it is contained.

    The scalar form drops non-finite samples before blocking; the batch cannot,
    because a row with gaps has a different finite count, hence a different
    block length and a different index matrix, and sharing that matrix is the
    entire speedup. So such a row reports ``nan``, and -- the part that matters
    at a call site -- its neighbours in the same batch are untouched.
    """
    rng = np.random.default_rng(14)
    clean = np.stack([_ar1(40, 0.8, rng) for _ in range(3)])
    gappy = clean.copy()
    gappy[1, 7] = np.nan
    gappy[2, 0] = np.inf

    observed = block_bootstrap_std_batch(gappy, rng=1)

    assert np.isnan(observed[1]) and np.isnan(observed[2])
    assert observed[0] == block_bootstrap_std(clean[0], rng=1)
    # Every row gappy is not an error either, just an all-nan answer.
    assert np.all(np.isnan(block_bootstrap_std_batch(np.full((2, 40), np.nan))))


def test_block_bootstrap_std_batch_is_nan_for_every_row_at_the_smoke_shape() -> None:
    """The shared time axis decides definedness for the whole batch.

    Three frames against twenty blocks gives ``L = 1`` -- point resampling
    wearing a block bootstrap's name -- for every row at once, since every row
    is bootstrapped over the same axis. This fires on every smoke-shaped run.
    """
    assert np.all(np.isnan(block_bootstrap_std_batch(np.zeros((4, 3)) + 1.0)))
    assert np.all(np.isnan(block_bootstrap_std_batch(np.tile(np.arange(20.0), (2, 1)))))
    assert np.all(
        np.isfinite(
            block_bootstrap_std_batch(np.tile(np.linspace(0.0, 1.0, 36), (2, 1)))
        )
    )
    # No rows at all is an empty answer, not a crash.
    assert block_bootstrap_std_batch(np.zeros((0, 36))).shape == (0,)


def test_block_bootstrap_std_batch_is_unchanged_by_its_memory_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunking the replicate axis bounds the temporary; it must not move a number.

    The statistic reduces along time only, so a row's replicate cannot depend
    on which chunk it was computed in. This pins that, because the chunk size
    is derived from the batch shape and so differs between the shipped shape
    and any test fixture.
    """
    from pyurbanair.utils import turbulence_stats

    rng = np.random.default_rng(15)
    rows = np.stack([_ar1(64, 0.85, rng) for _ in range(6)])

    unchunked = block_bootstrap_std_batch(rows, rng=2)
    monkeypatch.setattr(turbulence_stats, "_BATCH_CHUNK_BYTES", 1)
    assert np.array_equal(
        turbulence_stats.block_bootstrap_std_batch(rows, rng=2), unchunked
    )


def test_block_bootstrap_std_batch_rejects_bad_shapes_and_bad_statistics() -> None:
    """The axis contract is checked, because getting it wrong broadcasts silently."""
    rows = np.tile(np.linspace(0.0, 1.0, 40), (3, 1))

    with pytest.raises(ValueError):
        block_bootstrap_std_batch(rows[0])  # 1-D: no element axis
    with pytest.raises(ValueError):
        block_bootstrap_std_batch(rows, n_blocks=0)
    with pytest.raises(ValueError):
        block_bootstrap_std_batch(rows, n_resamples=1)
    with pytest.raises(ValueError, match="axis"):
        # A flat-series statistic ignoring ``axis`` returns a scalar, which
        # would otherwise broadcast into every replicate of every row.
        block_bootstrap_std_batch(rows, statistic=lambda x, axis: np.mean(x))


def test_block_bootstrap_std_batch_is_reproducible_and_advances_a_generator() -> None:
    """Same seed, same numbers; an explicit generator still advances once per call."""
    rng = np.random.default_rng(16)
    rows = np.stack([_ar1(50, 0.6, rng) for _ in range(4)])

    assert np.array_equal(
        block_bootstrap_std_batch(rows), block_bootstrap_std_batch(rows)
    )
    generator = np.random.default_rng(0)
    first = block_bootstrap_std_batch(rows, rng=generator)
    assert not np.array_equal(first, block_bootstrap_std_batch(rows, rng=generator))
    # The default is the module seed, not OS entropy -- so are the batch's.
    assert np.array_equal(
        block_bootstrap_std_batch(rows), block_bootstrap_std_batch(rows, rng=0)
    )
