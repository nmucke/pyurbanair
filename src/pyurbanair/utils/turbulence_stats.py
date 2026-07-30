"""Flow-statistics helpers for the ESMDA evaluation stack (WP1.3).

Companion to :mod:`pyurbanair.utils.ensemble_scores`: that module scores
ensembles probabilistically, this one turns raw solver output into the
turbulence quantities worth scoring. Both are pure numpy so the metrics
stage, the figure stage and ``scripts/figspec`` share one implementation.

Nearly empty for now. WP1.0 creates the module so the import path is
settled; WP1.3 fills it with

* ``StreamingMoments`` -- chunk-wise accumulation of ``n``, ``sum u_i`` and
  ``sum u_i u_j`` giving means, Reynolds stresses and TKE without ever
  holding a window state file in memory;
* ``colocate_components`` -- staggered-grid velocity components interpolated
  onto cell centres, per backend (uDALES/PALM staggering, pass-through for
  the uniform pylbm and surrogate grids);
* the mean-field scores (hit rate, FAC2, fractional bias, NMSE and its
  systematic/unsystematic split).

Specification: ``docs/plans/esmda_evaluation/phase1_postprocessing_metrics.md``
(WP1.3). The one exception is :func:`block_bootstrap_std`, the sampling floor
those scores are read against: WP1.2's sensor statistics need it first, so it
landed early rather than being written twice. Nothing else is stubbed --
names added ahead of that work would only have to be deleted.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# The resampling draw must not depend on OS entropy: re-running the metrics
# stage on the same run directory has to reproduce the same numbers. Same
# reasoning, and the same value, as ``ensemble_scores._DEFAULT_TIE_SEED``.
_DEFAULT_BOOTSTRAP_SEED = 0

__all__: list[str] = ["block_bootstrap_std"]


def block_bootstrap_std(
    series: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_blocks: int = 20,
    n_resamples: int = 200,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Moving-block bootstrap standard error of a statistic of one series.

    How much of a window statistic (a time-mean, a variance, a TKE) is just
    the finite length of the window? The iid answer ``std/sqrt(n)`` is wrong
    for a probe series by a large factor, because turbulent samples are not
    independent -- a 400-sample AR(1) series at ``phi = 0.9`` has roughly the
    sampling spread of 20 independent ones. The moving-block bootstrap keeps
    the within-block correlation by resampling *contiguous stretches* rather
    than points: block length ``L = ceil(n / n_blocks)``, then ``ceil(n / L)``
    blocks drawn with replacement from all ``n - L + 1`` start positions,
    concatenated and truncated back to ``n``, repeated ``n_resamples`` times.
    The answer is the ``ddof=1`` spread of the statistic over those replicates.

    Measured on AR(1) series of unit marginal variance, statistic ``np.mean``,
    median over 200 series; "true" is the spread of the sample mean across
    those same series, i.e. the quantity being estimated:

    ====  ====  ======  ==  =======  ===========  =====
    n     phi   blocks  L   this fn  std/sqrt(n)  true
    ====  ====  ======  ==  =======  ===========  =====
    400   0.0   20      20  0.049    0.050        0.055
    400   0.7   20      20  0.106    0.050        0.131
    400   0.9   20      20  0.154    0.048        0.238
    36    0.9   20      2   0.168    0.128        0.596
    ====  ====  ======  ==  =======  ===========  =====

    Two things to read off that table. On independent data the estimate lands
    on the iid formula, so blocking costs nothing when there is no correlation
    to preserve. On correlated data it is 2-3x the iid formula -- and still
    **below** the truth, increasingly so as ``L`` falls below the correlation
    time (the last row, a 36-frame window: ``L = 2`` cannot represent a
    correlation spanning ten frames). So this is a *lower bound* on the
    sampling spread of a correlated series, which is the safe direction for
    its WP1.2 consumer: it can only make the identifiability ratio it feeds
    look better than it is, never worse.

    Args:
        series: One member's (or the truth's) time series, any shape; it is
            flattened **in time order** and non-finite entries are dropped
            before blocking, so a block spanning a gap is a contiguous stretch
            of the samples that exist rather than a window of ``nan``.
        statistic: Reduction applied to each resampled series, e.g.
            ``np.mean``, ``np.var``, or a closure computing TKE.
        n_blocks: Target number of blocks; the block length is derived from it
            so one call site works on windows of different lengths.
        n_resamples: Number of bootstrap replicates.
        rng: ``numpy`` generator, or a seed for one. ``None`` means the fixed
            default seed -- never OS entropy, so the default path is
            reproducible too.

    Returns:
        The bootstrap standard error, or ``nan`` when it is undefined: fewer
        than four finite samples, or ``L < 2`` because ``n_blocks`` is finer
        than the series is long. **The second case is routine, not exotic** --
        the default 20 blocks needs 21 samples, and the smoke shape has three
        frames per window -- so callers must handle the ``nan``.

    Raises:
        ValueError: If ``n_blocks < 1`` or ``n_resamples < 2`` (a spread over
            fewer than two replicates does not exist).
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples}")

    values = np.asarray(series, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n_samples = int(values.size)
    if n_samples < 4:
        return float("nan")
    block_len = int(np.ceil(n_samples / n_blocks))
    if block_len < 2:
        return float("nan")

    if isinstance(rng, np.random.Generator):
        generator = rng
    else:
        generator = np.random.default_rng(
            _DEFAULT_BOOTSTRAP_SEED if rng is None else rng
        )

    n_starts = n_samples - block_len + 1
    n_draw = int(np.ceil(n_samples / block_len))
    offsets = np.arange(block_len)
    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        starts = generator.integers(0, n_starts, size=n_draw)
        # Truncated back to ``n`` so every replicate has the length of the
        # original; a statistic that depends on the sample count (a ``ddof=1``
        # variance, say) then stays comparable across replicates.
        index = (starts[:, None] + offsets[None, :]).ravel()[:n_samples]
        replicates[i] = float(statistic(values[index]))

    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1))
