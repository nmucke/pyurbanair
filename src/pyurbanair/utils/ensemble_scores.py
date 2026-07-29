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

See ``docs/plans/esmda_evaluation/phase1_postprocessing_metrics.md`` (WP1.0)
for how these are wired into the metrics schema.
"""

from __future__ import annotations

import numpy as np

# Tie-breaking in ``pit_rank`` must not depend on OS entropy: re-running the
# metrics stage on the same run directory has to reproduce the same numbers.
_DEFAULT_TIE_SEED = 0

__all__ = [
    "coverage",
    "coverage_indicator",
    "crpss",
    "fair_crps",
    "fair_energy_score",
    "pit_rank",
    "rank_histogram",
    "spread_skill_ratio",
    "zscore",
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
    memory is ``(M, M, *batch[1:])`` rather than ``(M, M, *batch)``. Keep it
    that way: at production scale a whole-run pairwise tensor does not fit.

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

    A calibrated ensemble gives z-scores that look like standard normal draws;
    ``|z| > 3`` flags overconfidence.

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


def rank_histogram(ranks: np.ndarray, n_members: int, n_bins: int = 10) -> np.ndarray:
    """Bin PIT ranks into a rank (Talagrand) histogram.

    Ranks from :func:`pit_rank` take ``n_members + 1`` values, which rarely
    divides evenly into ``n_bins``; the mapping ``bin = rank * n_bins //
    (n_members + 1)`` spreads them as evenly as the arithmetic allows and is
    exactly uniform whenever ``n_bins`` divides ``n_members + 1``. Bin counts
    are therefore comparable to a flat ``len(ranks) / n_bins`` reference to
    within the usual rounding of the last bin.

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

    Thin reduction over :func:`coverage_indicator`; a calibrated ensemble
    scores ``~alpha``. Returns ``nan`` for an empty batch.
    """
    inside = coverage_indicator(ens, truth, alpha=alpha)
    if inside.size == 0:
        return float("nan")
    return float(np.mean(inside))


# ---------------------------------------------------------------------------
# Spread/skill
# ---------------------------------------------------------------------------


def spread_skill_ratio(
    variances: np.ndarray, sq_errors: np.ndarray, n_members: int
) -> float:
    """Finite-ensemble-corrected spread/skill ratio (calibrated ~ 1).

    ``sqrt((M + 1)/M) * sqrt(mean(variances)) / sqrt(mean(sq_errors))``.

    Two conventions matter and are shared with the phase-0 implementation in
    ``scripts/figspec/metrics.spread_skill`` (that function takes the standard
    deviation and error series instead of their squares; the two agree
    element-for-element):

    * the average is over **variances**, not standard deviations -- an RMS
      spread, which is what the spread/skill relation is derived for;
    * the Fortin et al. (2014) factor ``sqrt((M + 1)/M)`` corrects the
      finite-ensemble bias, without which a perfectly calibrated ensemble of
      ``M`` members scores ``sqrt(M/(M + 1)) < 1``.

    NaNs are ignored (``nanmean``) so masked cells/sensors can be passed
    through unfiltered.

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
