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

from pyurbanair.utils.turbulence_stats import block_bootstrap_std


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
