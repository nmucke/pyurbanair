"""Probabilistic ensemble scores (pure numpy, ensemble axis first).

The shared scoring math for the ESMDA evaluation stack lives here so the
metrics stage (``scripts/esmda/compute_esmda_metrics.py``), the figure stage
and the sweep/figspec pipelines all use one implementation instead of three
drifting copies. Nothing in here knows about xarray, Hydra or run
directories -- callers hand over plain arrays and get plain arrays back.

Conventions, applied without exception:

* The **ensemble axis is axis 0**; every remaining axis is a "batch" axis
  (knot, time, sensor, grid point, ...) and is preserved in the output.
* Only **fair (unbiased) finite-ensemble estimators** are used: pairwise
  terms exclude the zero diagonal and divide by ``M(M - 1)``, spread ratios
  carry the Fortin ``sqrt((M + 1)/M)`` factor. The biased ``M**2`` forms
  reward under-dispersion and must not reappear.
* Degenerate ensembles (``M = 1``, zero spread) degrade to a documented
  value -- ``nan`` where the score is undefined -- rather than raising or
  silently returning an infinity. The two-member smoke shape used by the
  test suite is a real, supported case.
* **A calibration diagnostic ships with its own reference.** At production
  ``M`` almost none of them have the textbook large-sample reference: PIT
  bins are unequal (:func:`rank_histogram_weights`), the z-score null is a
  scaled ``t`` rather than a normal (:func:`zscore_nominal_exceedance`), an
  order-statistic band cannot hit an arbitrary ``alpha``
  (:func:`coverage_nominal_alpha`), and the Wasserstein-over-floor ratio is
  not calibrated at 1 (:func:`wasserstein_over_floor_reference`). Every one of
  those is a function here, so the plotting and metrics layers read the
  reference instead of re-deriving it -- a re-derived reference drifts, and
  each of these drifts *toward* reporting a perfectly calibrated ensemble as
  broken.

See ``docs/plans/esmda_evaluation/phase1_postprocessing_metrics.md`` (WP1.0)
for how these are wired into the metrics schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

# The block resampler is shared rather than reimplemented: WP1.3's bootstrap
# already owns the moving-block draw (and its scalar/batch pair is pinned to
# agree because both call it), so the perfect-model reference below draws its
# synthetic members from the same place. ``turbulence_stats`` imports nothing
# from this module, so the dependency is one-way.
from pyurbanair.utils.turbulence_stats import (
    _block_resample_indices,
    _resolve_generator,
)

# Tie-breaking in ``pit_rank`` must not depend on OS entropy: re-running the
# metrics stage on the same run directory has to reproduce the same numbers.
_DEFAULT_TIE_SEED = 0

# Thresholds the z-score exceedance reference is reported at by default. Two
# levels, not one: |z| > 2 has enough expected counts to be estimable from a
# few hundred pooled knots, while |z| > 3 is the tail a reader actually cares
# about but is nearly unpopulated at that sample size.
DEFAULT_ZSCORE_THRESHOLDS: tuple[float, ...] = (2.0, 3.0)

# Division guard for :func:`wasserstein_over_floor`. A floor of exactly zero
# needs two bit-identical halves, which a nonzero-spread series can produce
# (a perfectly periodic probe signal); clamping keeps the ratio a large finite
# number instead of an ``inf`` that ``write_yaml`` cannot emit.
_WASSERSTEIN_FLOOR_EPS = 1e-12

__all__ = [
    "DEFAULT_ZSCORE_THRESHOLDS",
    "coverage",
    "coverage_indicator",
    "coverage_nominal_alpha",
    "crpss",
    "fair_crps",
    "fair_energy_score",
    "max_abs_zscore_reference",
    "max_nominal_alpha",
    "pit_rank",
    "rank_histogram",
    "rank_histogram_weights",
    "spread_skill_ratio",
    "wasserstein_over_floor",
    "wasserstein_over_floor_reference",
    "wasserstein_over_sigma",
    "wasserstein_self_floor",
    "zscore",
    "zscore_exceedance",
    "zscore_nominal_exceedance",
]


# ---------------------------------------------------------------------------
# CRPS and its multivariate generalization
# ---------------------------------------------------------------------------


def fair_crps(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair sample CRPS of an ensemble against a deterministic truth.

    ``CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|`` with ``X, X'`` independent
    draws from ``F``. With a finite ensemble of ``M`` members the pairwise
    term excludes the zero diagonal and divides by ``M(M - 1)``; this is the
    fair estimator, which -- unlike the all-pairs ``M**2`` form -- does not
    reward under-dispersion at small ``M``.

    The pairwise term uses the sorted-sample identity
    ``sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - M + 1) * x_(i)``, algebraically
    identical to forming the full pairwise-difference tensor but ``O(M log M)``
    in time and free of its ``O(M**2)`` memory cost.

    Args:
        ens: ``(n_members, *batch)`` ensemble values.
        truth: ``(*batch,)`` truth, broadcastable against one member.

    Returns:
        ``(*batch,)`` CRPS, in the units of the quantity (lower is better).
        A one-member ensemble returns the absolute error ``|x - y|``.
    """
    ens = np.asarray(ens)
    truth = np.asarray(truth)
    n_members = ens.shape[0]
    term1 = np.mean(np.abs(ens - truth[None, ...]), axis=0)
    if n_members < 2:
        return term1
    sorted_ens = np.sort(ens, axis=0)
    weights = (2 * np.arange(n_members) - n_members + 1).reshape(
        (n_members,) + (1,) * (sorted_ens.ndim - 1)
    )
    term2 = np.sum(weights * sorted_ens, axis=0) / (n_members * (n_members - 1))
    return term1 - term2


def fair_energy_score(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair energy score: the multivariate generalization of the CRPS.

    For a vector ensemble ``{v_m}`` and truth ``v`` (Gneiting & Raftery 2007)

        ES = mean_m ||v_m - v|| - 0.5 / (M(M-1)) * sum_{m != m'} ||v_m - v_m'||,

    which reduces exactly to :func:`fair_crps` when the vector is 1-D. It
    rewards accuracy (first term) and calibrated spread (second term) in the
    units of the vector.

    Unlike :func:`fair_crps` there is no sorting identity for vector norms, so
    the pairwise term is materialized -- but only one slab at a time. The
    leading batch axis (time, for the sensor callers) is looped over, so peak
    memory is one slab rather than the whole-run pairwise tensor. Keep it that
    way: at production scale the latter does not fit.

    Memory bound, stated exactly because WP1.3 will point this at fields rather
    than sensors: everything is up-cast to float64, and peak allocation is
    ``max(M * prod(batch) * C, M**2 * prod(batch[1:]) * C) * 8`` bytes -- the
    ``term1`` difference over the *whole* batch, and the per-slab ``diff``
    tensor, which carries the ``C`` components (the norm is only taken
    afterwards). Negligible at sensor scale (M=32, ~1e3 times, 3 sensors, C=3:
    a few MB) but ``M**2 * C`` per grid point, so a field caller must chunk the
    grid into the leading batch axis rather than passing a whole slab.

    Args:
        members: ``(n_members, *batch, n_components)`` member vectors, with
            the vector components on the **last** axis.
        truth: ``(*batch, n_components)`` truth vectors.

    Returns:
        ``(*batch,)`` energy score. A one-member ensemble returns the
        Euclidean error ``||v_1 - v||``.
    """
    members = np.asarray(members, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if members.ndim < 2:
        raise ValueError(
            "members must have shape (n_members, *batch, n_components); "
            f"got array with shape {members.shape}"
        )
    if members.shape[1:] != truth.shape:
        raise ValueError(
            f"truth shape {truth.shape} does not match the per-member shape "
            f"{members.shape[1:]}"
        )

    n_members = members.shape[0]
    # Distance to truth never needs the pairwise tensor, so it is done whole.
    term1 = np.sqrt(np.sum((members - truth[None, ...]) ** 2, axis=-1)).mean(axis=0)
    if n_members < 2:
        return term1

    batch = members.shape[1:-1]
    if not batch:
        diff = members[:, None, :] - members[None, :, :]
        pairwise = np.sqrt(np.sum(diff**2, axis=-1))
        term2 = 0.5 * pairwise.sum() / (n_members * (n_members - 1))
        return term1 - term2

    scores = np.empty(batch, dtype=float)
    for i in range(batch[0]):
        slab = members[:, i, ...]  # (M, *batch[1:], C)
        diff = slab[:, None, ...] - slab[None, :, ...]
        pairwise = np.sqrt(np.sum(diff**2, axis=-1))  # (M, M, *batch[1:])
        scores[i] = 0.5 * pairwise.sum(axis=(0, 1)) / (n_members * (n_members - 1))
    return term1 - scores


def crpss(post_score: float, prior_score: float) -> float | None:
    """Skill score ``1 - post/prior`` (positive = the update helped).

    Guarded exactly like the existing reduction ratios in
    ``scripts/esmda/_esmda_common.parameter_metric_summary``: a zero,
    negative or non-finite reference score makes the ratio meaningless, so
    the answer is ``None`` (which serializes to a YAML ``null``) rather than
    an infinity or a ``nan`` that would poison downstream aggregation.

    Args:
        post_score: Posterior score (CRPS, energy score, RMSE, ...).
        prior_score: The same score for the reference/prior ensemble.

    Returns:
        The skill score, or ``None`` when it is undefined.
    """
    post = float(post_score)
    prior = float(prior_score)
    if not np.isfinite(post) or not np.isfinite(prior) or prior <= 0.0:
        return None
    return float(1.0 - post / prior)


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


def zscore(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Standardized truth-vs-ensemble error ``(truth - mean) / std(ddof=1)``.

    **A calibrated ensemble does not give standard normal z-scores**, and the
    difference is not academic at production ``M``. Both the mean and the
    spread are estimated from the same ``M`` members, so under exchangeability
    the null is ``sqrt((M + 1)/M) * t(M - 1)``: the ``t`` from the ``ddof=1``
    denominator, the Fortin factor from ``truth - mean_m`` having variance
    ``sigma**2 (1 + 1/M)``. At ``M = 32`` that puts ``P(|z| > 3)`` at 0.59%,
    2.2x the normal 0.27% (measured over 4e5 trials: 0.583%). Compare against
    :func:`zscore_nominal_exceedance` / :func:`zscore_exceedance`, never
    against a normal table, and never against a fixed cut on ``max |z|``
    (see :func:`max_abs_zscore_reference` for why).

    Undefined cases return ``nan`` without raising or warning: a one-member
    ensemble (no spread estimate exists) and any batch element whose spread is
    zero or non-finite.

    Args:
        ens: ``(n_members, *batch)`` ensemble values.
        truth: ``(*batch,)`` truth.

    Returns:
        ``(*batch,)`` z-scores.
    """
    ens = np.asarray(ens, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if ens.shape[0] < 2:
        return np.full(truth.shape, np.nan)
    mean = ens.mean(axis=0)
    std = ens.std(axis=0, ddof=1)
    valid = np.isfinite(std) & (std > 0.0)
    out = np.full(truth.shape, np.nan)
    np.divide(truth - mean, std, out=out, where=valid)
    return out


def _zscore_null_scale(n_members: int) -> float:
    """Fortin-style inflation of the z-score null: ``sqrt((M + 1)/M)``."""
    return float(np.sqrt((n_members + 1) / n_members))


def zscore_nominal_exceedance(n_members: int, threshold: float) -> float:
    """``P(|z| > threshold)`` for a perfectly calibrated ``M``-member ensemble.

    The reference level a measured exceedance fraction must be read against.
    Under exchangeability the null of :func:`zscore` is
    ``sqrt((M + 1)/M) * t(M - 1)`` (see that docstring), so

        P(|z| > c) = 2 * sf_t(c / sqrt((M + 1)/M); M - 1).

    Measured against 4e5 synthetic calibrated draws, both corrections earn
    their place -- the normal reference is wrong by a factor, and dropping the
    Fortin scale is wrong by ~10-25%:

    ======  ====  =========  =========  =========  =======
    M       c     empirical  this fn    plain t    normal
    ======  ====  =========  =========  =========  =======
    32      3.0   0.00583    0.00594    0.00529    0.00270
    32      2.0   0.05789    0.05789    0.05433    0.04550
    4       2.0   0.17156    0.17159    0.13933    0.04550
    ======  ====  =========  =========  =========  =======

    Args:
        n_members: Ensemble size the z-scores were computed from (``>= 2``).
        threshold: Positive cut on ``|z|``.

    Returns:
        The two-sided nominal exceedance probability, in ``(0, 1)``.
    """
    if n_members < 2:
        raise ValueError(f"n_members must be at least 2, got {n_members}")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"threshold must be positive and finite, got {threshold}")
    scaled = float(threshold) / _zscore_null_scale(n_members)
    return float(2.0 * stats.t.sf(scaled, df=n_members - 1))


def zscore_exceedance(
    z: np.ndarray,
    n_members: int,
    thresholds: Sequence[float] = DEFAULT_ZSCORE_THRESHOLDS,
) -> dict[str, Any]:
    """Observed ``|z|`` exceedance fractions **next to their nominal levels**.

    This is the sample-size-aware replacement for a fixed cut on ``max |z|``.
    A tail *fraction* has a sample-size-independent expectation, so pooling
    more knots sharpens it instead of inflating it -- the failure mode of
    ``max |z| > 3``, whose false-positive rate on a perfectly calibrated
    ensemble is 12% at 21 pooled knots and **82% at 315** (M = 32, 400 trials
    each; 315 knots is the routine 2-parameter/21-knot/3-window shape).

    The nominal levels are emitted alongside the observed ones for the same
    reason :func:`rank_histogram_weights` is emitted alongside the PIT counts:
    the reference depends on ``M``, the consumer (WP1.4) sees only the numbers,
    and a reference the plot layer has to re-derive is a reference that will
    drift. Both are reported -- ``nominal`` (the correct
    ``sqrt((M + 1)/M) * t(M - 1)`` null) and ``nominal_normal`` (the large-``M``
    limit), because the gap between them is itself the diagnostic for whether
    the ensemble is big enough for a normal reading.

    Note on independence: the fractions are *unbiased* however correlated the
    pooled knots are, but their sampling error is not ``sqrt(p(1-p)/n)`` when
    knots share members. This function deliberately emits no p-value or
    boolean; deciding "is this miscalibrated" needs an effective sample size
    the caller owns (``n_knots_effective``, itself an upper bound).

    Args:
        z: Any-shaped z-scores from :func:`zscore` (flattened; non-finite
            entries are dropped, matching the ``nan`` contract there).
        n_members: Ensemble size the z-scores were computed from (``>= 2``).
        thresholds: Positive cuts on ``|z|``, ascending by convention.

    Returns:
        A plain-Python mapping, YAML-safe apart from a ``nan`` ``observed``
        when nothing is finite:

        * ``n_samples`` (``int``) -- finite z-scores actually used.
        * ``df`` (``int``) -- ``n_members - 1``, the t degrees of freedom.
        * ``null_scale`` (``float``) -- ``sqrt((M + 1)/M)``.
        * ``thresholds`` (``list[float]``) -- echoed, so the four parallel
          lists below are self-describing in a summary file.
        * ``counts`` (``list[int]``) -- ``#{|z| > threshold}``.
        * ``observed`` (``list[float]``) -- ``counts / n_samples``; ``nan``
          when ``n_samples == 0``.
        * ``nominal`` (``list[float]``) -- :func:`zscore_nominal_exceedance`,
          **the level to compare ``observed`` against**.
        * ``nominal_normal`` (``list[float]``) -- ``2 * sf_norm(threshold)``,
          for context only.
    """
    if n_members < 2:
        raise ValueError(f"n_members must be at least 2, got {n_members}")
    cuts = [float(t) for t in thresholds]
    if not cuts:
        raise ValueError("thresholds must not be empty")

    flat = np.asarray(z, dtype=float).ravel()
    finite = flat[np.isfinite(flat)]
    n_samples = int(finite.size)
    abs_z = np.abs(finite)

    counts = [int(np.count_nonzero(abs_z > cut)) for cut in cuts]
    observed = [float(c) / n_samples if n_samples > 0 else float("nan") for c in counts]
    return {
        "n_samples": n_samples,
        "df": int(n_members - 1),
        "null_scale": _zscore_null_scale(n_members),
        "thresholds": cuts,
        "counts": counts,
        "observed": observed,
        "nominal": [zscore_nominal_exceedance(n_members, cut) for cut in cuts],
        "nominal_normal": [float(2.0 * stats.norm.sf(cut)) for cut in cuts],
    }


def max_abs_zscore_reference(
    n_members: int, n_samples: int, quantile: float = 0.5
) -> float:
    """Where ``max |z|`` sits for a *calibrated* ensemble of this size.

    Secondary to :func:`zscore_exceedance`, and offered only so the reported
    ``max_abs`` -- an informative number worth keeping -- can be read at all.
    The maximum of ``n`` draws is an order statistic whose distribution grows
    with ``n``, so the answer is a quantile, closed form: with
    ``F(c) = P(|z| <= c)`` under the ``sqrt((M + 1)/M) * t(M - 1)`` null,
    ``P(max |z| <= c) = F(c)**n``, hence

        c_q = sqrt((M + 1)/M) * ppf_t((1 + q**(1/n)) / 2; M - 1).

    At ``M = 32`` the median calibrated ``max |z|`` is 2.27 over 21 knots,
    2.96 over 105 and 3.39 over 315 -- so a fixed cut at 3 sits above the
    median at 21 knots, on it at 105 and below it at 315, matching the
    measured false-positive rates of that cut (12% / 45% / 82%) to within
    simulation noise.

    **Use with care, and prefer the fractions.** This is keyed on an exact
    independent ``n``, and the only ``n`` the metrics stage has
    (``n_knots_effective``) is a *stated upper bound* on independence, not a
    count -- pass a bound and you get a bound, biased toward calling a
    calibrated ensemble overconfident. The exceedance fractions need no such
    number to be unbiased.

    Args:
        n_members: Ensemble size (``>= 2``).
        n_samples: Number of (assumed independent) z-scores the max is over.
        quantile: Which quantile of ``max |z|`` to return; ``0.5`` is the
            typical value, ``0.95`` a one-sided calibrated ceiling.

    Returns:
        The ``quantile``-quantile of ``max |z|`` under the calibrated null.
    """
    if n_members < 2:
        raise ValueError(f"n_members must be at least 2, got {n_members}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}")
    per_draw = 0.5 * (1.0 + quantile ** (1.0 / n_samples))
    return _zscore_null_scale(n_members) * float(
        stats.t.ppf(per_draw, df=n_members - 1)
    )


def pit_rank(
    ens: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Rank of the truth among the members, with ties randomized (seeded).

    The rank is the number of members strictly below the truth, so it lies in
    ``[0, n_members]`` -- ``n_members + 1`` possible values, one per gap in the
    sorted ensemble. For an exchangeable (calibrated) ensemble the rank is
    uniform over those values; systematic U- or dome-shapes indicate under- or
    over-dispersion.

    Ties (identical member/truth values, common with clipped or saturated
    quantities) are broken by a uniform draw over the tied gaps rather than
    always rounding one way, which would put a spurious spike in the
    histogram. The draw uses an explicit generator so that re-running the
    metrics stage on the same run directory reproduces the same numbers.

    Args:
        ens: ``(n_members, *batch)`` ensemble values.
        truth: ``(*batch,)`` truth.
        rng: ``numpy`` generator, or a seed for one. ``None`` means the fixed
            default seed -- never OS entropy, so the default path is
            reproducible too.

    Returns:
        ``(*batch,)`` integer ranks in ``[0, n_members]``.
    """
    ens = np.asarray(ens)
    truth = np.asarray(truth)
    if isinstance(rng, np.random.Generator):
        generator = rng
    else:
        generator = np.random.default_rng(_DEFAULT_TIE_SEED if rng is None else rng)

    below = np.sum(ens < truth[None, ...], axis=0)
    tied = np.sum(ens == truth[None, ...], axis=0)
    # ``integers`` is exclusive on the high end, hence ``tied + 1`` gaps.
    jitter = generator.integers(0, np.asarray(tied) + 1)
    return np.asarray(below + jitter, dtype=int)


def rank_histogram_weights(n_members: int, n_bins: int = 10) -> np.ndarray:
    """How many of the ``n_members + 1`` rank values land in each bin.

    This is the **expected shape of a perfectly calibrated histogram**, up to
    the sample size: a calibrated ensemble draws uniformly over the ranks, so
    bin ``i`` gets ``len(ranks) * weights[i] / (n_members + 1)`` counts, not
    ``len(ranks) / n_bins``. Consumers that plot :func:`rank_histogram` against
    a reference must use this, or a calibrated ensemble reads as structured
    (see the note in :func:`rank_histogram`).

    Args:
        n_members: Ensemble size the ranks were computed from.
        n_bins: Number of histogram bins.

    Returns:
        ``(n_bins,)`` integer weights summing to ``n_members + 1``.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be positive, got {n_members}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    ranks = np.arange(n_members + 1)
    return np.bincount((ranks * n_bins) // (n_members + 1), minlength=n_bins)[:n_bins]


def rank_histogram(ranks: np.ndarray, n_members: int, n_bins: int = 10) -> np.ndarray:
    """Bin PIT ranks into a rank (Talagrand) histogram.

    Ranks from :func:`pit_rank` take ``n_members + 1`` values, which rarely
    divides evenly into ``n_bins``; the mapping ``bin = rank * n_bins //
    (n_members + 1)`` is exactly uniform only when ``n_bins`` divides
    ``n_members + 1``.

    **Otherwise the bins are genuinely unequal, and a flat reference is the
    wrong one.** At ``M = 32`` and ``n_bins = 10`` the rank values split
    ``[4, 3, 3, 4, 3, 3, 4, 3, 3, 3]``, so a perfectly calibrated ensemble
    produces a fixed three-bin comb of +21% / -9% about ``len(ranks)/n_bins``;
    at ``M < n_bins`` some bins cannot be reached at all. Compare against
    :func:`rank_histogram_weights` (rescaled by ``len(ranks)/(n_members + 1)``),
    never against a flat line.

    Args:
        ranks: Any-shaped integer ranks in ``[0, n_members]`` (flattened).
        n_members: Ensemble size the ranks were computed from.
        n_bins: Number of histogram bins.

    Returns:
        ``(n_bins,)`` integer counts.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be positive, got {n_members}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    flat = np.asarray(ranks).ravel().astype(int)
    flat = np.clip(flat, 0, n_members)
    bins = (flat * n_bins) // (n_members + 1)
    return np.bincount(bins, minlength=n_bins)[:n_bins]


def _band_indices(n_members: int, alpha: float) -> tuple[int, int]:
    """Zero-based order-statistic indices delimiting the central band."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    q_lo = 0.5 - alpha / 2.0
    q_hi = 0.5 + alpha / 2.0
    k_lo = int(np.ceil(q_lo * (n_members + 1)))
    k_hi = int(np.ceil(q_hi * (n_members + 1)))
    lo = min(max(k_lo, 1), n_members) - 1
    hi = min(max(k_hi, 1), n_members) - 1
    return lo, hi


def coverage_indicator(
    ens: np.ndarray, truth: np.ndarray, alpha: float = 0.9
) -> np.ndarray:
    """Per-element indicator: truth inside the central ``alpha`` band.

    The band edges are **actual member order statistics** -- member
    ``ceil(q(M + 1))`` for ``q = (1 -/+ alpha)/2``, clamped into ``[1, M]`` --
    not interpolated quantiles. For an exchangeable ensemble the truth falls
    uniformly into one of the ``M + 1`` gaps between sorted members, so the
    order-statistic band has exactly the nominal coverage whenever
    ``q(M + 1)`` is an integer, whereas ``np.quantile`` interpolation is
    biased low at small ``M``. The clamp means small ensembles cannot claim
    more than ``(M - 1)/(M + 1)`` coverage -- honest, not a bug: with two
    members the widest available band is ``[x_(1), x_(2)]``.

    (The legacy ``da_metrics.per_knot_in_band`` keeps its ``np.quantile``
    behaviour for backwards-compatible numbers; new code uses this.)

    Args:
        ens: ``(n_members, *batch)`` ensemble values.
        truth: ``(*batch,)`` truth.
        alpha: Nominal central probability, e.g. ``0.9``.

    Returns:
        ``(*batch,)`` boolean array.
    """
    ens = np.asarray(ens)
    truth = np.asarray(truth)
    lo_idx, hi_idx = _band_indices(ens.shape[0], alpha)
    ordered = np.sort(ens, axis=0)
    return (truth >= ordered[lo_idx]) & (truth <= ordered[hi_idx])


def coverage(ens: np.ndarray, truth: np.ndarray, alpha: float = 0.9) -> float:
    """Fraction of the batch covered by the central ``alpha`` band.

    Thin reduction over :func:`coverage_indicator`. A calibrated ensemble
    scores ``coverage_nominal_alpha(M, alpha)`` -- **not** ``alpha``, which the
    order-statistic edges can only hit when ``alpha (M + 1)/2`` lands on a
    half-integer. Report :func:`coverage_nominal_alpha` next to this number.
    Returns ``nan`` for an empty batch.
    """
    inside = coverage_indicator(ens, truth, alpha=alpha)
    if inside.size == 0:
        return float("nan")
    return float(np.mean(inside))


def coverage_nominal_alpha(n_members: int, alpha: float = 0.9) -> float:
    """The level :func:`coverage` at this ``(M, alpha)`` is actually targeting.

    **Emit this next to every coverage number.** The band edges are member
    order statistics, so the attainable nominal levels are the ``M + 1``
    multiples of ``1/(M + 1)`` and the requested ``alpha`` is rounded to one of
    them; the realized target is ``(k_hi - k_lo)/(M + 1)`` for the same edges
    :func:`coverage_indicator` uses. A consumer comparing an empirical coverage
    against the *requested* ``alpha`` reads pure discretization as
    miscalibration -- measured on calibrated ensembles (2e5 samples):

    ======  =====  ==================  =======  =========
    M       alpha  band                nominal  empirical
    ======  =====  ==================  =======  =========
    32      0.5    ``[x_(9), x_(25)]``  0.4848   0.4841
    32      0.9    ``[x_(2), x_(32)]``  0.9091   0.9088
    8       0.5    ``[x_(3), x_(7)]``   0.4444   0.4440
    4       0.5    ``[x_(2), x_(4)]``   0.4000   0.4014
    ======  =====  ==================  =======  =========

    The empirical column tracks ``nominal``, never ``alpha``. This exists so
    the metrics layer never grows a second copy of the edge convention:
    ``_band_indices`` stays private and this is its one public consequence.

    :func:`max_nominal_alpha` answers a different question (the *widest* band
    this ``M`` can offer, i.e. the ceiling a large ``alpha`` clamps to) and is
    unaffected by ``alpha``; both are worth reporting.

    **A return of exactly 0.0 is possible and is not an error**: when
    ``alpha < 1/(M + 1)`` both edges round to the same order statistic, so the
    "band" is a single member and only an exact tie is inside it. The caller
    should treat a zero nominal level as "this ``alpha`` is unresolvable at
    this ``M``" rather than dividing by it. Out of reach for the levels the
    metrics stage uses (``alpha >= 0.5`` needs ``M < 1``), reachable for a
    2-member ensemble at ``alpha = 0.1``.

    Args:
        n_members: Ensemble size.
        alpha: The nominal level as *requested* by the caller.

    Returns:
        The realized nominal level, a multiple of ``1/(n_members + 1)``, in
        ``[0, (M - 1)/(M + 1)]``.
    """
    lo, hi = _band_indices(n_members, alpha)
    return float(hi - lo) / float(n_members + 1)


def max_nominal_alpha(n_members: int) -> float:
    """Widest nominal level an order-statistic band can reach: ``(M-1)/(M+1)``.

    The band ``[x_(1), x_(M)]`` leaves the two outer gaps of the ``M + 1``
    uncovered, so a request above this clamps silently -- 0.94 at ``M = 32``,
    0.6 at ``M = 4``, 1/3 at ``M = 2``. Reporting it stops a clamped
    ``alpha_90`` from reading as a calibration failure. It bounds the
    **nominal** level only; a realized fraction can sit above it by chance.

    Equal to ``max(coverage_nominal_alpha(M, a) for a in (0, 1))``, and pinned
    to that by a test -- the two must never disagree.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be positive, got {n_members}")
    return float(n_members - 1) / float(n_members + 1)


# ---------------------------------------------------------------------------
# Spread/skill
# ---------------------------------------------------------------------------


def spread_skill_ratio(
    variances: np.ndarray, sq_errors: np.ndarray, n_members: int
) -> float:
    """Finite-ensemble-corrected spread/skill ratio (calibrated ~ 1).

    ``sqrt((M + 1)/M) * sqrt(mean(variances)) / sqrt(mean(sq_errors))``.

    This is the **only** implementation: ``scripts/figspec/metrics.spread_skill``
    is a thin adapter over it, keeping the figspec vocabulary of a
    standard-deviation series and an RMSE series and squaring them at the call
    site. Two conventions matter:

    * the average is over **variances**, not standard deviations -- an RMS
      spread, which is what the spread/skill relation is derived for;
    * the Fortin et al. (2014) factor ``sqrt((M + 1)/M)`` corrects the
      finite-ensemble bias, without which a perfectly calibrated ensemble of
      ``M`` members scores ``sqrt(M/(M + 1)) < 1``.

    Non-finite handling, stated because it is now the shared contract rather
    than one caller's convention: ``nan`` entries are ignored (``nanmean``) so
    masked cells/sensors can be passed through unfiltered, and an input with no
    finite entries at all short-circuits to ``nan``. That guard keys on
    ``isfinite``, so an ``inf``-valued spread series is treated as **masked and
    returns ``nan``**, not propagated to an ``inf`` ratio -- deliberate: an
    infinite spread makes the diagnostic undefined, not "infinitely
    over-dispersed", and a ``nan`` is what the rest of this module returns for
    an undefined score. (This is the one case where the phase-0 figspec body
    differed; it returned ``+inf``.)

    Args:
        variances: Per-element ensemble variance (``ddof=1``).
        sq_errors: Per-element squared ensemble-mean error, same shape.
        n_members: Ensemble size.

    Returns:
        The ratio, or ``nan`` when the error norm is zero or non-finite.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be positive, got {n_members}")
    var = np.asarray(variances, dtype=float)
    err = np.asarray(sq_errors, dtype=float)
    # A fully masked input is a legitimate case (all-solid slab); short-circuit
    # it rather than letting nanmean warn about an empty slice.
    if not np.any(np.isfinite(var)) or not np.any(np.isfinite(err)):
        return float("nan")
    num = float(np.sqrt(np.nanmean(var)))
    den = float(np.sqrt(np.nanmean(err)))
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return float(np.sqrt((n_members + 1) / n_members)) * num / den


# ---------------------------------------------------------------------------
# Distributional distance
# ---------------------------------------------------------------------------


def _finite_samples(values: np.ndarray) -> np.ndarray:
    """Flattened finite entries, **in the order they were given**.

    Order is load-bearing for :func:`wasserstein_self_floor`, which splits the
    result in time; the Wasserstein distance itself is order-blind.
    """
    flat = np.asarray(values, dtype=float).ravel()
    finite: np.ndarray = flat[np.isfinite(flat)]
    return finite


def _sigma_or_nan(samples: np.ndarray) -> float:
    """``std(ddof=1)``, or ``nan`` when it is not meaningfully positive.

    The test is relative, not ``sigma > 0``. A sensor series that is physically
    constant is rarely *bitwise* constant by the time it reaches here: the
    magnitude ``sqrt(sum_c u_c**2)`` of a constant unit vector comes back with a
    spread of ~4.5e-16, and dividing by that turns a zero distance into ~1e15 --
    finite, so it survives the ``_finite_or_none`` filter and lands in
    ``run_summary.yaml`` as a headline number. Anything below ``1e-12`` times
    the series' own magnitude is float64 rounding, not spread, so it is
    ``nan`` -- undefined, which is the truth.
    """
    if samples.size < 2:
        return float("nan")
    sigma = float(np.std(samples, ddof=1))
    scale = float(np.max(np.abs(samples)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return float("nan")
    if sigma < 1e-12 * scale:
        return float("nan")
    return sigma


def wasserstein_over_sigma(samples: np.ndarray, truth_samples: np.ndarray) -> float:
    """1-Wasserstein distance between two sample sets, in truth sigmas.

    ``W1(samples, truth_samples) / std(truth_samples, ddof=1)``. ``W1`` is the
    L1 area between the two empirical CDFs, so it is in the units of the
    quantity and responds to the *whole* distribution -- a shifted mean, a
    widened tail and a changed shape all register, where a moment-by-moment
    comparison sees only what it was asked for. Dividing by the truth spread
    makes it dimensionless and comparable across sensors and components.

    Neither series has to be sampled at the same cadence or length as the
    other, and neither is aligned in time: this compares *distributions*, which
    is the whole reason it is used on probe series whose ensemble and truth
    time axes do not line up.

    **A distance of zero is not the calibrated value.** Two independent draws
    from the *same* distribution sit a finite distance apart at finite sample
    size, and that offset is large at the window lengths this stack has.
    Measured on white noise, 400 trials, median:

    ==============================  ======  =======
    comparison                      n = 36  n = 400
    ==============================  ======  =======
    same distribution, independent  0.269   0.085
    :func:`wasserstein_self_floor`  0.356   0.117
    mean shifted by 1 sigma         0.994   1.002
    sigma doubled                   0.888   0.804
    ==============================  ======  =======

    So at a 36-frame window a raw 0.3 is where a *perfect* model sits. Report it
    against :func:`wasserstein_self_floor` via :func:`wasserstein_over_floor` --
    never against 0 -- and read *that* ratio against
    :func:`wasserstein_over_floor_reference` at the same sample count, because it
    is not calibrated at 1 either (a perfect model scores ~0.55 of the floor, and
    the value moves with ``n``, ``M`` and the series' autocorrelation).

    Note for callers pooling an ensemble: ``W1`` is convex in its first
    argument, so pooling all members into one sample set can only *lower* the
    distance relative to scoring members separately and averaging
    (``W1(mean of F_m, G) <= mean_m W1(F_m, G)``), with equality only when the
    members agree. The two numbers answer different questions -- pooled asks
    whether the ensemble *as a distribution* matches, per-member whether a
    typical member does -- and the gap between them is the ensemble spread.

    Args:
        samples: Any-shaped sample set (flattened; non-finite entries dropped).
        truth_samples: The reference sample set, likewise flattened.

    Returns:
        The normalized distance (>= 0, lower is better), or ``nan`` when either
        set has fewer than two finite samples or the truth spread is not
        meaningfully positive.
    """
    sample = _finite_samples(samples)
    truth = _finite_samples(truth_samples)
    if sample.size < 2 or truth.size < 2:
        return float("nan")
    sigma = _sigma_or_nan(truth)
    if not np.isfinite(sigma):
        return float("nan")
    return float(stats.wasserstein_distance(sample, truth)) / sigma


def wasserstein_self_floor(truth_samples: np.ndarray) -> float:
    """The distance a *perfect* model scores against this series.

    ``W1(first half, second half) / std(truth, ddof=1)`` -- the truth compared
    against itself, which is the finite-sample, finite-correlation-time offset
    that :func:`wasserstein_over_sigma` cannot go below no matter how good the
    model is.

    **The split is contiguous (halves in time order), not random.** A random
    split interleaves the two halves through the same slow excursions, so each
    half inherits the series' full range and the two empirical CDFs agree far
    better than two genuinely independent stretches of the flow would. It
    reports the white-noise floor whatever the series actually does. Median
    over 400 AR(1) series of unit marginal variance:

    ======  ======  ==========  ======  =====
    n       phi     contiguous  random  ratio
    ======  ======  ==========  ======  =====
    36      0.0     0.382       0.368   1.04
    36      0.7     0.588       0.356   1.65
    36      0.95    1.024       0.346   2.96
    400     0.7     0.199       0.114   1.75
    400     0.95    0.506       0.110   4.61
    ======  ======  ==========  ======  =====

    The random column is flat in ``phi`` -- it does not see the correlation at
    all -- so on a strongly correlated probe it understates the floor by 3-5x,
    and a perfectly good model then reads as 3-5x worse than the floor.

    Two known conservatisms, both in the safe direction (they inflate the floor,
    i.e. they make a bad model look acceptable rather than the reverse): the
    halves have ``n/2`` samples where the compared sample sets have ``n``, worth
    a factor ~``sqrt(2)`` (0.356 vs 0.269 on white noise at ``n = 36``), and a
    single split is one draw of a noisy quantity rather than an expectation.

    **A third, larger one: the window must be stationary.** The table above is
    measured on stationary AR(1) series, where the contiguous split reports the
    series' own sampling variability -- which is the quantity a model should be
    read against. Put a *deterministic* trend inside the window and the two
    halves are drawn from different parts of that trend, so the split measures
    the trend instead. Measured on a 108-frame ``|U|`` series with
    ``sigma_turb = 0.5``; perfect model = an independent realization of the same
    law, bad model = ``+20%`` in ``|U|`` with half the turbulence:

    ==============================  ==========  =============  =========
    truth                           self_floor  perfect ratio  bad ratio
    ==============================  ==========  =============  =========
    stationary                      0.168       0.75           18.2
    magnitude cosine only           0.366       0.38           5.8
    both cosines (shipped default)  0.575       0.12           1.9
    ==============================  ==========  =============  =========

    Read the last row. On the shipped default truth (``params@truth_params:
    dynamic_cosine`` -- a 400 s inflow-angle cosine and a 200 s magnitude
    cosine) a *clearly wrong* model scores ``w1_over_floor ~ 1.9``, which a reader
    comparing against 1 would call marginal. The floor has absorbed the forcing,
    and good and bad models are then both divided by it. Unlike the two
    conservatisms above this is not a small factor: it is the difference between
    18 and 2.

    :func:`wasserstein_over_floor_reference` softens the *reading* of that but
    does not fix the floor, and it is worth knowing which. Its stand-in windows
    are block resamples, which scramble a trend the way the rejected random split
    does, so the reference barely moves when the floor triples -- measured on the
    108-frame probe above (``M = 32``, 60 windows): floor 0.210 stationary /
    0.733 forced, perfect model 0.55 / 0.12, bad model 9.5 / 1.66, reference 0.52
    / 0.51. Reading against the reference, the bad model is 18x it stationary and
    still 3.3x it forced -- recognizably bad rather than marginal. But the
    *perfect* model reads 0.24x the reference on the forced window, i.e. the
    reference now over-states what perfect is, so a mediocre model can hide below
    it. Windowing is still the mitigation; the reference restores the bad-model
    signal, not the floor.

    **Mitigation, and its limit.** Callers must take the split over a window
    short enough to be *locally* stationary. WP1.2 does this: every Wasserstein
    quantity is computed per assimilation window and then reduced, rather than
    over the pooled run, which removes the cross-window part of the forcing --
    the dominant term in the table (0.168 stationary vs 0.575 with both cosines,
    over the full 540 s run). It does **not** remove a trend whose period is
    comparable to one window: a 180 s window against a 200 s magnitude cosine
    still contains most of a cycle. Measured on the same synthetic, a cosine of
    period ~2x the window inflates the floor ~3x whether the window is 108 or 36
    frames, while a cosine cycling several times *within* the window does not
    inflate it at all. So a ratio near 1 on a strongly forced truth means "the
    floor may be doing the work", not "the model passed".

    An odd sample count drops the **middle** sample so the halves are equal
    length; equal halves keep the ``n``-dependence of the distance the same on
    both sides.

    Args:
        truth_samples: Any-shaped truth series, flattened **in time order**
            (non-finite entries are dropped first, then the split is taken).

    Returns:
        The normalized self-distance (>= 0), or ``nan`` when fewer than four
        finite samples remain (each half needs two to be a distribution) or the
        spread is not meaningfully positive.
    """
    truth = _finite_samples(truth_samples)
    n_samples = truth.size
    if n_samples < 4:
        return float("nan")
    sigma = _sigma_or_nan(truth)
    if not np.isfinite(sigma):
        return float("nan")
    half = n_samples // 2
    # ``n - half`` skips the middle sample when the count is odd.
    first = truth[:half]
    second = truth[n_samples - half :]
    return float(stats.wasserstein_distance(first, second)) / sigma


def wasserstein_over_floor(w1: float, floor: float) -> float:
    """The headline ratio: ``w1 / max(floor, tiny)``. **Not calibrated at 1.**

    A distance read against the only reference that means anything at these
    window lengths (see the table in :func:`wasserstein_over_sigma`) -- but the
    quotient is *not* 1 for a perfect model, and reading it that way is a factor
    of ~2 wrong. The numerator compares ``M * n`` pooled member samples against
    ``n`` truth samples while the denominator compares ``n/2`` truth samples
    against ``n/2``, so the two sides are not like-for-like on sample count:

    * a perfect model scores **~0.55, roughly flat in ``n``** (numerator and
      floor both fall as ``1/sqrt(n)``), so a model reading exactly 1.0 is
      already about twice as far from the truth as a perfect one;
    * a *genuine* bias has a numerator that does not shrink with ``n``, so its
      ratio **grows as ``sqrt(n)``** -- which also means the number is not
      comparable across windows of different length.

    Both statements are numbers, not intuitions: see the n-scaling table in
    :func:`wasserstein_over_floor_reference`, which computes the perfect-model
    value for the series and ``M`` at hand (~0.55 is one ``phi`` and one ``M``;
    it moves with both). **Interpret this ratio only against that reference,
    evaluated at the same sample count** -- WP1.2 emits the pair as
    ``w1_over_floor`` and ``w1_over_floor_calibrated_median``. With the reference
    in hand, whether the ratio was computed per window or over the pooled run
    stops mattering for interpretation.

    Both arguments must come from the *same* normalization --
    :func:`wasserstein_over_sigma` and :func:`wasserstein_self_floor` divide by
    the same truth sigma, so it cancels and the ratio is a pure
    distance-over-distance. Passing an un-normalized ``w1`` silently rescales
    the answer by that sigma.

    A floor of exactly zero needs two bit-identical halves, which a
    nonzero-spread series can produce (an exactly periodic probe signal whose
    period divides the window). The denominator is clamped rather than nulled:
    the honest reading of that case is "the perfect-model distance is
    unresolvable here, so the ratio is enormous", and a large finite number
    says that where an ``inf`` would be dropped by the YAML writer.

    Args:
        w1: Normalized distance, typically from :func:`wasserstein_over_sigma`.
        floor: Normalized self-distance from :func:`wasserstein_self_floor`.

    Returns:
        The ratio, or ``nan`` when either input is non-finite (which is how
        both producers report an undefined distance).

    Raises:
        ValueError: If either argument is negative -- ``W1`` is a distance, so
            a negative one is a caller bug, not a degenerate sample.
    """
    numerator = float(w1)
    denominator = float(floor)
    if numerator < 0.0 or denominator < 0.0:
        raise ValueError(
            "w1 and floor are distances and must be non-negative; got "
            f"w1={numerator}, floor={denominator}"
        )
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    return numerator / max(denominator, _WASSERSTEIN_FLOOR_EPS)


def wasserstein_over_floor_reference(
    truth_samples: np.ndarray,
    n_members: int,
    *,
    n_resamples: int = 64,
    n_blocks: int = 20,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Median ``w1_over_floor`` a PERFECT model of this series would score.

    **The reference :func:`wasserstein_over_floor` has to be read against.**
    That ratio is *not* calibrated at 1: its numerator compares ``M * n`` pooled
    member samples against ``n`` truth samples while its denominator compares
    ``n/2`` truth samples against ``n/2``, so the two sides are not like-for-like
    on sample count and the quotient lands well below 1 for a perfect model.
    Both sides fall as ``1/sqrt(n)``, so the perfect-model value is roughly flat
    in the window length while a *genuine* bias -- whose numerator does not
    shrink with ``n`` -- makes the ratio grow as ``sqrt(n)``. A single number
    therefore cannot be interpreted without knowing ``n`` and ``M``, which is
    what this function supplies. Measured on AR(1) (``phi = 0.6``, stationary
    mean, ``M = 32``, median over 200 trials):

    ========  =============  ==========  ===========
    n scored  perfect model  +0.5 sigma  sigma x 0.5
    ========  =============  ==========  ===========
    18        0.77           0.93        0.75
    36        0.56           0.99        0.80
    108       0.53           1.48        1.25
    216       0.50           2.02        1.72
    432       0.54           3.00        2.47
    ========  =============  ==========  ===========

    A model reading exactly 1.0 is already ~2x worse than perfect at these
    window lengths. Emit this beside the ratio (WP1.2 does, as
    ``w1_over_floor_calibrated_median``) and compare the two element for
    element; do not compare against 1, and do not carry a prose constant --
    0.55 is the value at *one* ``phi`` and *one* ``M``, and it moves with both.

    **Method: a two-sample block bootstrap.** Each of ``n_resamples`` replicates
    draws ``n_members + 1`` series of ``n`` samples each from the truth by
    moving-block resampling -- block length ``ceil(n / n_blocks)``, drawn by the
    same ``turbulence_stats._block_resample_indices`` the WP1.3 bootstraps use,
    so the synthetic series inherit the truth's marginal distribution *and* (up to
    the block length) its autocorrelation. ``n_members`` of them are pooled into
    a synthetic perfect model exactly as the real caller pools its members; the
    remaining one is a **stand-in truth window**, and the replicate's ratio is
    ``wasserstein_over_sigma(pooled, stand_in) / wasserstein_self_floor(stand_in)``.
    The answer is the median over replicates.

    **Why the stand-in, and not the truth itself.** Scoring the pooled synthetic
    model against the real truth array -- the obvious construction, since the
    floor is deterministic given the truth -- is wrong by a factor of 4-6 and
    gets worse as ``M`` grows. The pooled sample is ``M * n`` draws *from the
    truth's own empirical distribution*, so its ECDF converges to that ECDF and
    ``W1(pooled, truth) -> 0``: it measures the model's sampling noise, which
    pooling averages away, and misses the term that actually dominates -- the
    truth window's own deviation from the law it was drawn from. Measured at
    ``M = 32``, that construction returns 0.05-0.13 where the answer is ~0.55.
    Resampling both sides restores the real comparison (two independent samples
    of ``M * n`` and ``n`` from one law) at the cost of conditioning: the median
    is over the sampling variability of the truth window as well as the model's,
    which is unavoidable -- the dominant term *is* the window's own sampling
    error, and no estimator conditioned on that one window can see it. So this is
    "the ratio a perfect model of this series and this length typically scores",
    not "the ratio conditional on this exact window".

    **The bootstrap caveat, stated plainly.** A resample reuses the truth's own
    values, so neither side is a fully independent realization of the same law --
    the two synthetic sides share the truth's ties and extremes, and nothing here
    guarantees the estimate is unbiased. This is the same device
    :func:`wasserstein_self_floor` rests on and the standard bootstrap argument for
    sampling variability, but it has to be measured rather than asserted, and it is
    (by ``test_wasserstein_reference_matches_a_directly_simulated_perfect_model``).
    Median ratio of this function to a directly
    simulated independent perfect model, paired per truth window (AR(1),
    ``M = 32``, 120 windows each, 5 independent model realizations per window):

    ======  ======  ======  ======
    n       phi 0   phi .6  phi .9
    ======  ======  ======  ======
    36      1.02    1.05    0.55
    108     0.99    0.91    0.88
    432     0.98    1.06    1.08
    ======  ======  ======  ======

    Within 12% at eight of the nine points, and the exception is low, not high.
    It has a clean mechanism: at ``phi = 0.9`` with ``n = 36`` the block length is
    2, which cannot reproduce a correlation spanning ten frames, so the resampled
    sides under-inherit the slow excursions that make a short strongly-correlated
    window a poor sample of its own law. **Note the direction.** Unlike the floor's
    conservatisms this one is *not* in the safe direction -- an under-stated
    reference makes a good model look bad -- so on a strongly autocorrelated probe
    scored over a short window, read a ratio moderately above the reference as
    inconclusive rather than as a failure. Longer windows and weakly correlated
    series do not have the problem.

    **Cost, and what ``n_resamples`` buys.** Per call: ``n_resamples`` Wasserstein
    evaluations on ``M * n``-sample arrays, ``n_resamples`` floors on ``n/2``-sample
    halves, and one ``(n_resamples * (M + 1), n)`` index draw (0.6 MB of ``int64``
    at the shipped shape, 26 MB at ``M = 128`` / ``n = 400``). Measured on this
    hardware at ``M = 32``, ``n = 36`` -- the shipped shape, where WP1.2 makes 24
    ``(sensor, window)`` calls -- with the seed-to-seed spread of the returned
    median over 200 seeds:

    ===========  ===========  ==========  ===========
    n_resamples  ms per call  24 calls    MC spread
    ===========  ===========  ==========  ===========
    16           1.9          0.05 s      0.100 (19%)
    64           7.3          0.18 s      0.061 (12%)
    128          14.4         0.35 s      0.041 (8%)
    ===========  ===========  ==========  ===========

    The default 64 keeps the addition an order of magnitude inside the ~1 s budget
    round-1's vectorization bought (0.37 s per call at ``M = 128`` / ``n = 400``).
    Its 12% per-element Monte-Carlo error is well below the ~2x effect being
    diagnosed, and WP1.2's ``{median, max}`` reduction over 24 elements averages
    the median's down by ``~sqrt(24)``; raise ``n_resamples`` if a *per-element*
    number is being plotted, since the error falls as ``1/sqrt(n_resamples)``
    while the cost rises linearly.

    Args:
        truth_samples: The truth series for one element, flattened **in time
            order** (non-finite entries dropped first, as everywhere else in
            this family) -- the same array handed to
            :func:`wasserstein_self_floor`, or the answer describes a different
            series than the floor does.
        n_members: Ensemble size the real ``w1_over_floor`` pooled, so the
            synthetic model is pooled from as many samples.
        n_resamples: Bootstrap replicates the median is taken over.
        n_blocks: Target block count, as in
            :func:`~pyurbanair.utils.turbulence_stats.block_bootstrap_std`; the
            block length is derived from the series length so one call site
            serves windows of different lengths. Two degenerate ends, both
            reachable from a config knob: fewer samples than blocks gives a block
            length of 1 -- an iid resample, which carries none of the series'
            correlation (unreachable at the shipped 36-frame window, where
            ``L = 2``, but not on a shorter one) -- and ``n_blocks=1`` makes the
            block the whole series, so there is one legal start, every replicate
            *is* the truth, and the answer is exactly ``0.0``. That zero is
            measured rather than undefined, the same way
            :func:`~pyurbanair.utils.turbulence_stats.block_bootstrap_std`
            returns ``0.0`` there.
        rng: ``numpy`` generator, or a seed for one. ``None`` means the fixed
            default seed -- never OS entropy, so re-running the metrics stage on
            a run directory reproduces the number.

    Returns:
        The median perfect-model ratio (``>= 0``; exactly ``0.0`` only in the
        degenerate ``n_blocks=1`` case above), or ``nan`` when
        :func:`wasserstein_self_floor` is ``nan`` for the real truth (fewer than
        four finite samples -- the smoke shape -- or a spread that is only float
        rounding), when that floor is zero (there is no perfect-model *ratio* to
        report against a floor :func:`wasserstein_over_floor` can only clamp), or
        when no replicate produced a usable stand-in floor.

    Raises:
        ValueError: If ``n_members < 1``, ``n_resamples < 1`` or ``n_blocks < 1``.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be positive, got {n_members}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")

    truth = _finite_samples(truth_samples)
    # The real floor is not used as the divisor (see the docstring), but it is
    # the gate: this function exists to calibrate ``w1_over_floor``, and where
    # that ratio is undefined (``nan`` floor) or only clamped (zero floor) there
    # is nothing to calibrate.
    real_floor = wasserstein_self_floor(truth)
    if not np.isfinite(real_floor) or real_floor <= 0.0:
        return float("nan")
    n_samples = int(truth.size)
    block_len = min(max(int(np.ceil(n_samples / n_blocks)), 1), n_samples)
    generator = _resolve_generator(rng)
    # ``n_members + 1`` rows per replicate: the pooled synthetic model, then the
    # stand-in truth window it is scored against. One draw for all of them --
    # rows come off the generator in the order a per-row loop would consume them.
    index = _block_resample_indices(
        n_samples, block_len, n_resamples * (n_members + 1), generator
    ).reshape(n_resamples, n_members + 1, n_samples)

    ratios = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        stand_in = truth[index[i, n_members]]
        stand_in_floor = wasserstein_self_floor(stand_in)
        # A replicate can degenerate where the truth cannot: a stand-in drawn
        # from one repeated block has no spread (``nan``) or two identical halves
        # (``0.0``). Those replicates drop out of the median rather than being
        # clamped -- the clamp would put ~1e12 in it and move the median off the
        # surviving replicates entirely.
        if not np.isfinite(stand_in_floor) or stand_in_floor <= 0.0:
            ratios[i] = float("nan")
            continue
        pooled = truth[index[i, :n_members]].ravel()
        ratios[i] = wasserstein_over_sigma(pooled, stand_in) / stand_in_floor

    usable = ratios[np.isfinite(ratios)]
    if usable.size == 0:
        return float("nan")
    return float(np.median(usable))
