"""Property tests for the shared flow-statistics helpers.

No snapshots: every assertion is a closed form, an inequality that states why
the naive alternative is wrong, or a seeded Monte-Carlo property of a
synthetic series whose correlation structure is known by construction. The
smoke shape (three frames per window) is exercised explicitly, because it is
the shape that pushes ``block_bootstrap_std`` onto its undefined path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import xarray

from pyurbanair.utils.turbulence_stats import (
    StreamingMoments,
    block_bootstrap_std,
    block_bootstrap_std_batch,
    colocate_components,
    extrapolated_centre_dims,
    fac2,
    fractional_bias,
    hit_rate,
    nmse,
    nmse_split,
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


# ---------------------------------------------------------------------------
# StreamingMoments
# ---------------------------------------------------------------------------


def test_streaming_moments_match_numpy_on_random_data() -> None:
    """The reference case: one chunk, no gaps, against ``np.mean``/``np.cov``.

    ``np.cov`` with ``rowvar=True`` and the default ``ddof=1`` is the same
    estimator ``reynolds_stress()`` reports, so this pins the convention as
    well as the arithmetic.
    """
    rng = np.random.default_rng(20)
    field = rng.normal(loc=[[3.0], [-1.0], [0.2]], scale=1.0, size=(3, 40))

    moments = StreamingMoments()
    moments.update(*(component[:, None] for component in field))

    assert moments.n_components == 3
    assert moments.cell_shape == (1,)
    assert moments.count().tolist() == [40]
    assert moments.mean()[:, 0] == pytest.approx(field.mean(axis=1))
    assert moments.reynolds_stress()[:, :, 0] == pytest.approx(np.cov(field))
    assert moments.tke()[0] == pytest.approx(0.5 * np.var(field, axis=1, ddof=1).sum())
    # ddof=0 is the plain time average, exactly n/(n-1) smaller.
    assert moments.reynolds_stress(ddof=0)[:, :, 0] == pytest.approx(
        np.cov(field, ddof=0)
    )


def test_streaming_moments_are_unchanged_by_how_the_time_axis_is_chunked() -> None:
    """Chunk boundaries must not appear in the answer.

    The driver's chunk length is an I/O decision (and differs between the truth
    pass, whose grid and cadence are its own, and the member pass), so a
    statistic that depended on it would not be comparable between the two sides
    of the very comparison this layer exists to make.
    """
    rng = np.random.default_rng(21)
    field = (
        rng.normal(size=(3, 37, 2, 5)) + np.array([5.0, 0.5, -2.0])[:, None, None, None]
    )

    whole = StreamingMoments()
    whole.update(*field)

    for chunks in ([1] * 37, [10, 10, 10, 7], [36, 1]):
        pieces = StreamingMoments()
        start = 0
        for length in chunks:
            pieces.update(*(component[start : start + length] for component in field))
            start += length
        assert np.allclose(pieces.mean(), whole.mean(), rtol=0, atol=1e-13)
        assert np.allclose(
            pieces.reynolds_stress(), whole.reynolds_stress(), rtol=0, atol=1e-13
        )
        assert np.array_equal(pieces.count(), whole.count())


def test_streaming_moments_beat_the_naive_sum_of_squares_on_an_offset_field() -> None:
    """The cancellation claim, asserted rather than described.

    ``sum uu - (sum u)**2/n`` on a field whose mean dwarfs its fluctuation
    loses most of its digits; this is the regime the module docstring tabulates
    (mean 1e4, fluctuation 1e-3). The accumulator has to stay at machine
    precision there, because the resulting variance is compared against a
    bootstrap sampling floor of the same order.
    """
    rng = np.random.default_rng(22)
    fluctuation = rng.normal(size=(360, 1))
    series = 1e4 + 1e-3 * fluctuation
    exact = float(
        np.var(series.astype(np.longdouble), ddof=1)
    )  # exact for the stored samples

    moments = StreamingMoments()
    for start in range(0, 360, 36):
        moments.update(series[start : start + 36])
    streaming = float(moments.reynolds_stress()[0, 0, 0])

    naive = (float((series**2).sum()) - float(series.sum()) ** 2 / series.size) / (
        series.size - 1
    )

    # measured: naive 3.9e-2 relative error, streaming 7.1e-11.
    assert abs(naive / exact - 1.0) > 1e-3
    assert abs(streaming / exact - 1.0) < 1e-9


def test_streaming_moments_count_per_cell_so_one_masked_cell_is_contained() -> None:
    """A gap at one cell must not cost its neighbours a frame.

    This is why ``n`` is an array and not a scalar: blocked cells and the edge
    cells of an interpolated truth field are per-cell facts, and a scalar count
    would have to either drop the whole frame or count the gap as a sample.
    """
    rng = np.random.default_rng(23)
    field = rng.normal(size=(3, 20, 4))
    holed = field.copy()
    holed[0, 5:9, 1] = np.nan  # only component 0, only cell 1, only 4 frames
    holed[2, 0, 3] = np.inf  # inf counts as non-finite too

    moments = StreamingMoments()
    moments.update(*holed)

    assert moments.count().tolist() == [20, 16, 20, 19]
    # The untouched cells are bit-comparable with a clean accumulation.
    clean = StreamingMoments()
    clean.update(*field)
    assert moments.mean()[:, 0] == pytest.approx(clean.mean()[:, 0])
    assert moments.mean()[:, 2] == pytest.approx(clean.mean()[:, 2])
    # ... and the holed cell reports the statistics of the frames it kept, in
    # *every* component: the mask is casewise, so component 1 at cell 1 also
    # skips frames 5-8.
    kept = np.r_[0:5, 9:20]
    assert moments.mean()[:, 1] == pytest.approx(field[:, kept, 1].mean(axis=1))
    assert moments.reynolds_stress()[:, :, 1] == pytest.approx(
        np.cov(field[:, kept, 1])
    )


def test_streaming_moments_casewise_masking_keeps_the_stress_a_covariance() -> None:
    """Per-pair sample sets would let ``tke`` go negative; one count cannot.

    Constructed adversarially: component 0 is finite only on the first half of
    the record and component 1 only on the second, with opposite offsets. Under
    available-case deletion the cross term would be estimated from no shared
    frames at all while the diagonals used all of them, which is how an
    indefinite "covariance" arises. Casewise deletion reports the two frames
    they actually share.
    """
    a = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0])
    b = np.array([np.nan, 5.0, 6.0, 7.0, 8.0, np.nan])
    shared = np.isfinite(a) & np.isfinite(b)
    assert shared.tolist() == [False, True, False, False, True, False]

    moments = StreamingMoments()
    moments.update(a[:, None], b[:, None])

    # Three finite frames in ``a`` and four in ``b``, but only two frames in
    # which both are finite -- and two is the count every entry of the tensor
    # is built from, diagonal included.
    assert moments.count().tolist() == [2]
    pairs = np.stack([a[shared], b[shared]])
    assert moments.mean()[:, 0] == pytest.approx(pairs.mean(axis=1))
    assert moments.reynolds_stress()[:, :, 0] == pytest.approx(np.cov(pairs))
    stress = moments.reynolds_stress()[:, :, 0]
    # The whole point: a genuine covariance matrix, so TKE cannot be negative.
    assert np.linalg.eigvalsh(stress).min() >= -1e-12
    assert moments.tke()[0] > 0.0


def test_streaming_moments_report_nan_where_nothing_was_seen() -> None:
    """A fully masked cell is ``nan``, not zero, and does not raise.

    Zero would be indistinguishable from a genuine stagnation point, and the
    consumers of this layer (the mean-field scores) drop non-finite cells as
    their "fluid cells only" rule -- so ``nan`` is the value that makes the
    masking work end to end.
    """
    field = np.zeros((2, 6, 3))
    field[:, :, 1] = np.nan  # cell 1 never observed
    field[:, 1:, 2] = np.nan  # cell 2 observed once

    moments = StreamingMoments()
    moments.update(*field)

    assert moments.count().tolist() == [6, 0, 1]
    assert np.isnan(moments.mean()[:, 1]).all()
    assert not np.isnan(moments.mean()[:, 2]).any()  # a mean needs one frame
    # ... but a ddof=1 second moment needs two, so the single-frame cell is nan.
    assert np.isnan(moments.reynolds_stress()[:, :, 2]).all()
    assert np.isnan(moments.tke()[2])
    assert np.isfinite(moments.reynolds_stress(ddof=0)[:, :, 2]).all()


def test_streaming_moments_reject_shapes_that_would_silently_mismatch() -> None:
    """Including the one that matters: selecting levels between chunks."""
    moments = StreamingMoments()

    with pytest.raises(ValueError, match="at least one"):
        moments.update()
    with pytest.raises(ValueError, match="one shape"):
        moments.update(np.zeros((4, 3)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="time axis"):
        moments.update(np.float64(1.0))
    with pytest.raises(ValueError, match="no frames accumulated"):
        moments.mean()

    moments.update(np.zeros((4, 6)), np.zeros((4, 6)))
    with pytest.raises(ValueError, match="fixed by the first update"):
        moments.update(np.zeros((4, 3)), np.zeros((4, 3)))  # z-levels selected
    with pytest.raises(ValueError, match="fixed by the first update"):
        moments.update(np.zeros((4, 6)))  # a component dropped
    with pytest.raises(ValueError, match="ddof"):
        moments.reynolds_stress(ddof=-1)
    # An empty chunk is a no-op, not an error: a window can hold no frames.
    before = moments.count().copy()
    moments.update(np.zeros((0, 6)), np.zeros((0, 6)))
    assert np.array_equal(moments.count(), before)


# ---------------------------------------------------------------------------
# colocate_components
# ---------------------------------------------------------------------------


def _linear(z: np.ndarray, y: np.ndarray, x: np.ndarray, scale: float) -> np.ndarray:
    """``scale * (2x + 3y + 5z + 7)`` on the outer product of three axes.

    A linear field is the test with teeth for linear interpolation: the
    interpolant is exact, *including* the half-cell extrapolation at the top
    edge, so any indexing error shows up as a finite discrepancy rather than as
    a small one.
    """
    grid = (
        2.0 * x[None, None, :] + 3.0 * y[None, :, None] + 5.0 * z[:, None, None] + 7.0
    )
    return scale * grid[None]  # a length-1 time axis


def _udales_state() -> xarray.Dataset:
    """A 3 x 4 x 5 uDALES grid: faces are the lower faces, same length as centres."""
    zt, yt, xt = np.arange(3) + 0.5, np.arange(4) + 0.5, np.arange(5) + 0.5
    zm, ym, xm = zt - 0.5, yt - 0.5, xt - 0.5
    return xarray.Dataset(
        {
            "u": (("time", "zt", "yt", "xm"), _linear(zt, yt, xm, 1.0)),
            "v": (("time", "zt", "ym", "xt"), _linear(zt, ym, xt, 2.0)),
            "w": (("time", "zm", "yt", "xt"), _linear(zm, yt, xt, 3.0)),
        },
        coords={"zt": zt, "yt": yt, "xt": xt, "zm": zm, "ym": ym, "xm": xm},
    )


def test_colocate_udales_is_exact_on_a_linear_field() -> None:
    """Every component lands on ``(zt, yt, xt)`` with the exact centre value.

    Exact, not close: linear interpolation of a linear field has no truncation
    error, and neither does the half-cell extrapolation of the last column, so
    ``atol=1e-12`` here is a statement about float rounding and nothing else.
    A swapped axis or an off-by-one shift moves a value by a whole cell (2, 3
    or 5 units by construction), so it cannot hide inside that tolerance.
    """
    state = _udales_state()
    zt, yt, xt = (np.asarray(state[name].values) for name in ("zt", "yt", "xt"))

    u, v, w = colocate_components(state, "udales")

    for field, scale in ((u, 1.0), (v, 2.0), (w, 3.0)):
        assert field.dims == ("time", "zt", "yt", "xt")
        assert np.allclose(
            np.asarray(field.values), _linear(zt, yt, xt, scale), rtol=0, atol=1e-12
        )
        assert np.allclose(np.asarray(field["xt"].values), xt)


def test_colocate_palm_moves_xu_and_yv_and_leaves_w_alone() -> None:
    """PALM stages ``u`` on ``xu`` and ``v`` on ``yv``; pypalm already fixed ``w``.

    The ``w`` pass-through is not an omission: pypalm's postprocess interpolates
    ``w`` from ``zw_3d`` onto ``z`` and drops ``zw``, so by the time a state file
    is read there is nothing left to move. The table still lists ``zw -> z``, and
    the second half of this test feeds a pre-unification layout to show that the
    entry fires when the dim is present.
    """
    z, y, x = np.arange(3) + 0.5, np.arange(4) + 0.5, np.arange(5) + 0.5
    yv, xu = y - 0.5, x - 0.5
    state = xarray.Dataset(
        {
            "u": (("time", "z", "y", "xu"), _linear(z, y, xu, 1.0)),
            "v": (("time", "z", "yv", "x"), _linear(z, yv, x, 2.0)),
            "w": (("time", "z", "y", "x"), _linear(z, y, x, 3.0)),
        },
        coords={"z": z, "y": y, "x": x, "yv": yv, "xu": xu},
    )

    u, v, w = colocate_components(state, "palm")

    for field, scale in ((u, 1.0), (v, 2.0), (w, 3.0)):
        assert field.dims == ("time", "z", "y", "x")
        assert np.allclose(
            np.asarray(field.values), _linear(z, y, x, scale), rtol=0, atol=1e-12
        )
    # ``w`` was not touched at all -- same object identity as in the Dataset.
    assert w.equals(state["w"])

    zw = z - 0.5
    raw = state.drop_vars("w").assign(
        w=(("time", "zw", "y", "x"), _linear(zw, y, x, 3.0))
    )
    raw = raw.assign_coords(zw=zw)
    _, _, w_raw = colocate_components(raw, "palm")
    assert w_raw.dims == ("time", "z", "y", "x")
    assert np.allclose(np.asarray(w_raw.values), _linear(z, y, x, 3.0), atol=1e-12)


def test_colocate_pylbm_and_the_surrogate_pass_straight_through() -> None:
    """A uniform grid is already co-located, so nothing may be interpolated.

    Asserted by identity rather than by value: any interpolation at all would
    smooth a field that is already at cell centres, and the surrogate reports
    ``solver_name: pylbm`` under every shipped config, so this is the path
    almost every run takes.
    """
    z, y, x = np.arange(3) + 0.5, np.arange(4) + 0.5, np.arange(5) + 0.5
    state = xarray.Dataset(
        {
            name: (("time", "z", "y", "x"), _linear(z, y, x, scale))
            for name, scale in (("u", 1.0), ("v", 2.0), ("w", 3.0))
        },
        coords={"z": z, "y": y, "x": x},
    )

    for solver in ("pylbm", "neural_surrogate"):
        for field, name in zip(colocate_components(state, solver), "uvw"):
            assert field.equals(state[name])


def test_colocate_refuses_a_dataset_that_was_sliced_before_colocation() -> None:
    """The silent-wrong-number case, turned into an error.

    Selecting a subset of z-levels leaves both axes uniformly spaced, so only
    the midpoint identity catches it -- and it has to be caught, because
    blending two levels five cells apart produces a plausible-looking field
    rather than a visible failure.
    """
    state = _udales_state()

    with pytest.raises(ValueError, match="midpoints of the faces"):
        colocate_components(state.isel(zt=[0, 2], zm=[0, 2]), "udales")
    # A single level leaves no interpolant at all.
    with pytest.raises(ValueError, match="defines no interpolant"):
        colocate_components(state.isel(zm=[0], zt=[0]), "udales")
    # Slicing an axis that does not need colocating is fine (this is how the
    # driver selects its z-slabs -- after colocation, on the centre axis).
    u, _, _ = colocate_components(state.isel(yt=[1, 2], ym=[1, 2]), "udales")
    assert u.sizes["yt"] == 2


def test_colocate_reports_what_it_cannot_do_instead_of_guessing() -> None:
    state = _udales_state()

    with pytest.raises(ValueError, match="no known staggering"):
        colocate_components(state, "openfoam")
    with pytest.raises(ValueError, match="missing the 'w' component"):
        colocate_components(state.drop_vars("w"), "udales")
    with pytest.raises(ValueError, match="no coordinate values"):
        colocate_components(state.drop_vars("xt"), "udales")


def test_colocate_spreads_a_masked_face_to_its_two_neighbouring_centres() -> None:
    """Documented, not desirable: interpolation is linear, so NaN propagates.

    A masked face cell makes both centres that average it NaN, which is the
    conservative direction (a cell whose value depends on a missing sample is
    not reported) and is exactly what ``StreamingMoments``' per-cell counting is
    built to absorb. Pinned here so the interaction between the two layers is a
    tested contract rather than an assumption.
    """
    state = _udales_state()
    poisoned = state.copy()
    values = np.asarray(poisoned["u"].values).copy()
    values[0, 0, 0, 2] = np.nan
    poisoned["u"] = (poisoned["u"].dims, values)

    u, _, _ = colocate_components(poisoned, "udales")

    missing = np.isnan(np.asarray(u.values))[0, 0, 0]
    assert missing.tolist() == [False, True, True, False, False]


def test_extrapolated_centre_dims_names_every_axis_carrying_an_edge() -> None:
    """Which axes' last index came out of extrapolation rather than the data.

    A caller cannot see this in the returned array -- keeping the grid the same
    length is the whole point of extrapolating -- and it matters because an
    evenly spaced level selection ALWAYS includes the last index, so the edge
    lands on a scored level unless something excludes it deliberately.
    """
    assert extrapolated_centre_dims(_udales_state(), "udales") == ("xt", "yt", "zt")
    # pylbm writes one unstaggered grid, so nothing is extrapolated at all.
    zt, yt, xt = np.arange(3) + 0.5, np.arange(4) + 0.5, np.arange(5) + 0.5
    pylbm = xarray.Dataset(
        {
            name: (("time", "z", "y", "x"), _linear(zt, yt, xt, 1.0))
            for name in ("u", "v", "w")
        },
        coords={"z": zt, "y": yt, "x": xt},
    )
    assert extrapolated_centre_dims(pylbm, "pylbm") == ()

    # Same skip rule as `colocate_components`: pypalm's postprocess has already
    # moved `w` off `zw`, so a postprocessed PALM state extrapolates only the
    # two horizontal axes -- the entry for `w` is a no-op rather than a claim.
    palm = xarray.Dataset(
        {
            "u": (("time", "z", "y", "xu"), _linear(zt, yt, xt - 0.5, 1.0)),
            "v": (("time", "z", "yv", "x"), _linear(zt, yt - 0.5, xt, 1.0)),
            "w": (("time", "z", "y", "x"), _linear(zt, yt, xt, 1.0)),
        },
        coords={
            "z": zt,
            "y": yt,
            "x": xt,
            "xu": xt - 0.5,
            "yv": yt - 0.5,
        },
    )
    assert extrapolated_centre_dims(palm, "palm") == ("x", "y")

    with pytest.raises(ValueError, match="no known staggering"):
        extrapolated_centre_dims(_udales_state(), "openfoam")


def test_the_extrapolated_edge_inflates_variance_against_the_interior() -> None:
    """The cost of the edge, measured against the reference that matters.

    ``1.5**2 + 0.5**2 = 2.5`` is the amplification against ONE UNFILTERED
    sample, but no interior cell of a colocated field is one unfiltered sample
    -- it is the midpoint average, whose variance is ``0.5 * (1 + rho)``. The
    ratio a consumer needs is ``(2.5 - 1.5 rho) / (0.5 + 0.5 rho)``: 5x for
    face-to-face white noise, ~1.2x once the field is resolved. The docstring
    quotes those numbers, so they are pinned here against the actual weights.
    """
    rng = np.random.default_rng(3)
    n = 400_000
    for rho, expected in ((0.0, 5.0), (0.5, 7.0 / 3.0), (0.9, 1.15 / 0.95)):
        first = rng.normal(size=n)
        second = rho * first + np.sqrt(1.0 - rho**2) * rng.normal(size=n)
        interior = 0.5 * (first + second)
        edge = 1.5 * first - 0.5 * second
        assert np.var(edge) / np.var(interior) == pytest.approx(expected, rel=0.02)


# ---------------------------------------------------------------------------
# Mean-field error norms
# ---------------------------------------------------------------------------


def test_hit_rate_counts_the_relative_or_the_absolute_criterion() -> None:
    """Hand-computed: the ``or`` is what makes this work on a signed component.

    ``obs = 1`` throughout, so the relative allowance is 0.25 in absolute
    terms; the four predictions are off by 0.2, 0.3, 0.0 and 0.3.
    """
    obs = np.ones(4)
    pred = np.array([1.2, 1.3, 1.0, 0.7])

    assert hit_rate(pred, obs, allowance=0.0) == pytest.approx(0.5)
    assert hit_rate(pred, obs, allowance=0.35) == pytest.approx(1.0)
    assert hit_rate(pred, obs, allowance=0.0, relative_tolerance=0.05) == pytest.approx(
        0.25
    )
    # A zero observation is decided by the absolute clause alone -- the point of
    # having one, since a velocity component passes through zero.
    assert hit_rate(np.array([0.1]), np.array([0.0]), allowance=0.2) == 1.0
    assert hit_rate(np.array([0.1]), np.array([0.0]), allowance=0.05) == 0.0


def test_hit_rate_without_a_sampling_floor_falls_back_to_the_relative_test() -> None:
    """``nan`` allowance is "not available", and it can only lower ``q``.

    The floor comes from ``block_bootstrap_std``, which is undefined at the
    smoke shape (three frames against twenty blocks), so this path is routine.
    The direction matters: a missing floor must never flatter the run.
    """
    obs = np.ones(4)
    pred = np.array([1.2, 1.3, 1.0, 0.7])

    relative_only = hit_rate(pred, obs, allowance=0.0)
    assert hit_rate(pred, obs, allowance=float("nan")) == relative_only
    assert relative_only <= hit_rate(pred, obs, allowance=0.35)

    with pytest.raises(ValueError, match="allowance"):
        hit_rate(pred, obs, allowance=-0.1)
    with pytest.raises(ValueError, match="relative_tolerance"):
        hit_rate(pred, obs, allowance=0.0, relative_tolerance=-0.25)


def test_fac2_is_hand_computable_and_refuses_signed_input() -> None:
    """A ratio test on a sign-changing quantity is meaningless, so it raises.

    That is the only place the plan's "positive quantities only" rule can be
    enforced: by the time the number is in ``run_summary.yaml`` it is
    indistinguishable from a valid one.
    """
    obs = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    pred = np.array([0.5, 2.0, 0.49, 2.01, 2.0])

    assert fac2(pred, obs) == pytest.approx(0.6)  # the two boundaries count
    # Zero against zero counts as agreement -- correct for a stagnation point,
    # optimistic for an unmasked (zero-filled) solid cell.
    assert fac2(np.array([0.0]), np.array([0.0])) == 1.0
    assert fac2(np.array([0.1]), np.array([0.0])) == 0.0

    with pytest.raises(ValueError, match="positive quantities only"):
        fac2(np.array([-1.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="positive quantities only"):
        fac2(np.array([1.0, 1.0]), np.array([-1.0, 1.0]))


def test_fractional_bias_and_nmse_match_their_definitions() -> None:
    """Both hand-computed from metrics-doc section 4.1, including the sign."""
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = obs - 1.0  # a pure -1 bias: mean 2.5 vs 1.5

    assert fractional_bias(pred, obs) == pytest.approx(1.0 / 2.0)  # (2.5-1.5)/2.0
    assert nmse(pred, obs) == pytest.approx(1.0 / (2.5 * 1.5))
    # Positive FB means the prediction is too *small*.
    assert fractional_bias(obs, pred) == pytest.approx(-0.5)
    # A perfect prediction: zero bias, zero error.
    assert fractional_bias(obs, obs) == pytest.approx(0.0)
    assert nmse(obs, obs) == pytest.approx(0.0)
    # NMSE's normalization is undefined when the mean product is not positive,
    # which a signed component reaches routinely.
    assert np.isnan(nmse(np.array([-1.0, 1.0]), np.array([2.0, -2.0])))
    assert np.isnan(fractional_bias(np.array([-1.0]), np.array([1.0])))


def test_nmse_split_is_the_squared_bias_and_the_error_variance() -> None:
    """The identity, not an inequality: ``NMSE_s`` *is* the bias term.

    ``4FB**2/(4-FB**2)`` reduces to ``(o_mean - p_mean)**2/(o_mean*p_mean)``, so
    the remainder is ``var(o - p)/(o_mean*p_mean)`` -- non-negative, which makes
    the plan's ``NMSE_s <= NMSE`` an algebraic consequence rather than a check.
    """
    rng = np.random.default_rng(24)
    obs = 5.0 + rng.normal(size=200)
    pred = 4.0 + 1.3 * rng.normal(size=200)

    fb = fractional_bias(pred, obs)
    total = nmse(pred, obs)
    systematic, unsystematic = nmse_split(fb, total)

    scale = float(np.mean(obs)) * float(np.mean(pred))
    assert systematic == pytest.approx((np.mean(obs) - np.mean(pred)) ** 2 / scale)
    assert unsystematic == pytest.approx(np.var(obs - pred) / scale)
    assert systematic <= total
    assert unsystematic >= 0.0

    # An unbiased prediction puts everything in the irreducible part.
    fb_unbiased = fractional_bias(obs, obs + rng.normal(size=200) * 0.0)
    assert nmse_split(fb_unbiased, 0.4) == (pytest.approx(0.0), pytest.approx(0.4))

    # |FB| >= 2 is only reachable for a sign-changing quantity, and there the
    # formula would return a negative systematic part -- reported as undefined.
    assert all(np.isnan(value) for value in nmse_split(2.0, 1.0))
    assert all(np.isnan(value) for value in nmse_split(-2.5, 1.0))
    assert all(np.isnan(value) for value in nmse_split(float("nan"), 1.0))
    assert all(np.isnan(value) for value in nmse_split(0.1, float("nan")))


def test_mean_field_scores_use_fluid_cells_only_and_the_same_ones() -> None:
    """Pairwise finiteness, shared by all four scores.

    Two properties at once: masked cells are dropped (which is how "fluid cells
    only" is implemented, since no backend NaNs its solid cells for us), and
    every score drops the *same* cells -- if they did not,
    :func:`nmse_split`'s identity would silently stop holding.
    """
    obs = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    pred = np.array([1.1, 2.2, 3.3, np.inf, 5.5])
    fluid = np.array([0, 1, 4])

    # Annotated because the scorers have different signatures: without it mypy
    # joins them to an unknown callable type and rejects the call below.
    scorers: tuple[tuple[Callable[..., float], dict[str, float]], ...] = (
        (hit_rate, {"allowance": 0.05}),
        (fac2, {}),
        (fractional_bias, {}),
        (nmse, {}),
    )
    for score, kwargs in scorers:
        assert score(pred, obs, **kwargs) == pytest.approx(
            score(pred[fluid], obs[fluid], **kwargs)
        )

    # Nothing finite at all is a null, not a crash or a zero -- the degenerate
    # shape rule (report null, never special-case the math).
    empty = np.full(3, np.nan)
    assert np.isnan(hit_rate(empty, empty, allowance=1.0))
    assert np.isnan(fac2(empty, empty))
    assert np.isnan(fractional_bias(empty, empty))
    assert np.isnan(nmse(empty, empty))

    with pytest.raises(ValueError, match="same shape"):
        nmse(np.zeros(3), np.zeros(4))
