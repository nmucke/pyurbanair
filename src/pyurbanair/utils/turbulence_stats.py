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
(WP1.3). The one exception is :func:`block_bootstrap_std` (with its batched
sibling :func:`block_bootstrap_std_batch`), the sampling floor those scores are
read against: WP1.2's sensor statistics need it first, so it landed early
rather than being written twice. Nothing else is stubbed -- names added ahead
of that work would only have to be deleted.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# The resampling draw must not depend on OS entropy: re-running the metrics
# stage on the same run directory has to reproduce the same numbers. Same
# reasoning, and the same value, as ``ensemble_scores._DEFAULT_TIE_SEED``.
_DEFAULT_BOOTSTRAP_SEED = 0

# Ceiling on the temporary :func:`block_bootstrap_std_batch` builds. The
# resampled array is ``(n_resamples, n_elements, n_time)`` floats, which is
# 16 MB at WP1.2's shipped shape but ~440 MB at M=128 / S=20 / W=10 -- the very
# shape the batch path exists for. Splitting the replicate axis into chunks
# bounds it without moving a single number: the statistic reduces along the
# time axis only, so a row's replicate does not depend on which chunk it was
# computed in. (Momentarily twice this while the gathered block is made
# contiguous, which is still an order of magnitude below the unchunked array.)
_BATCH_CHUNK_BYTES = 32 << 20

__all__: list[str] = ["block_bootstrap_std", "block_bootstrap_std_batch"]


def _validate_bootstrap_shape(n_blocks: int, n_resamples: int) -> None:
    """Shared argument check for the scalar and batch bootstraps."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples}")


def _resolve_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """A generator from a generator, a seed, or the fixed module default."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(_DEFAULT_BOOTSTRAP_SEED if rng is None else rng)


def _block_resample_indices(
    n_samples: int,
    block_len: int,
    n_resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """The moving-block resample index matrix, shape ``(n_resamples, n_samples)``.

    The single place the draw happens: :func:`block_bootstrap_std` and
    :func:`block_bootstrap_std_batch` both build their replicates from this, so
    the documented estimator and the fast one cannot drift apart. Drawing all
    replicates in one ``integers`` call consumes the generator's stream in
    exactly the order a per-replicate loop consumes it, so the matrix -- and
    therefore every number either function returns -- is identical to the
    per-replicate loop this replaced.

    Each row is ``ceil(n / L)`` block starts drawn with replacement from the
    ``n - L + 1`` legal positions, expanded to their offsets, concatenated and
    truncated back to ``n``. The truncation matters: the blocks overshoot
    whenever ``L`` does not divide ``n``, and a statistic that depends on the
    sample count (a ``ddof=1`` variance, say) has to be evaluated at the
    original length in every replicate to stay comparable.
    """
    n_starts = n_samples - block_len + 1
    n_draw = int(np.ceil(n_samples / block_len))
    starts = generator.integers(0, n_starts, size=(n_resamples, n_draw))
    offsets = np.arange(block_len)
    index = starts[:, :, None] + offsets[None, None, :]
    return index.reshape(n_resamples, -1)[:, :n_samples]


def _replicate_spread(replicates: np.ndarray) -> float:
    """``ddof=1`` spread over one row of replicates; the shared reduction step.

    Identical replicates are collapsed to a true ``0.0`` rather than left to
    ``np.std``, whose mean subtraction leaves ~1e-17 of float rounding behind
    (measured 2.8e-17 on ``n_blocks=1`` at n=60). That residue is not a
    measurement: when every replicate is the same number the bootstrap
    distribution is a point mass and its spread is zero. It also matters
    downstream -- WP1.2 nulls ``identifiability`` on ``within > 0``, and 1e-17
    passes that filter and turns the ratio into ~1e17 instead of a clean null.
    """
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        return float("nan")
    if float(finite.min()) == float(finite.max()):
        return 0.0
    return float(np.std(finite, ddof=1))


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

        ``n_blocks=1`` returns exactly ``0.0``, which is a *measured* zero and
        not an undefined result: one block spans the whole series, there is a
        single legal start position, so every replicate is the original series
        and the spread over them really is zero. (Exactly zero because
        identical replicates are collapsed rather than handed to ``np.std``,
        which would leave ~1e-17 of float rounding -- enough to survive a
        ``> 0`` test.) ``bootstrap_blocks: 1`` is a
        legal config (``resolve_metrics_settings`` validates ``>= 1``), so this
        is reachable from a YAML knob and deliberately does not raise -- that
        validator exists so a bad knob is reported cheaply rather than crashing
        deep inside a streaming pass. WP1.2's consumer filters on ``within >
        0``, which turns the ``0.0`` into a clean null ``identifiability``.

    Raises:
        ValueError: If ``n_blocks < 1`` or ``n_resamples < 2`` (a spread over
            fewer than two replicates does not exist).
    """
    _validate_bootstrap_shape(n_blocks, n_resamples)

    values = np.asarray(series, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n_samples = int(values.size)
    if n_samples < 4:
        return float("nan")
    block_len = int(np.ceil(n_samples / n_blocks))
    if block_len < 2:
        return float("nan")

    generator = _resolve_generator(rng)
    index = _block_resample_indices(n_samples, block_len, n_resamples, generator)
    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        replicates[i] = float(statistic(values[index[i]]))

    return _replicate_spread(replicates)


def block_bootstrap_std_batch(
    series: np.ndarray,
    statistic: Callable[..., np.ndarray] = np.mean,
    n_blocks: int = 20,
    n_resamples: int = 200,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """:func:`block_bootstrap_std` for many equal-length series at once.

    Same estimator, same seed, same numbers -- both functions build their
    replicates from :func:`_block_resample_indices`, and a one-row batch is
    *exactly* equal to the scalar call (pinned by a test), so this is a speed
    change and nothing else. It exists because WP1.2's caller needs one
    bootstrap per ``(member, sensor, statistic)``: the Python double loop that
    calls the scalar form costs ~1 ms each, ~4.6 s at the shipped shape
    (M=32, C=3, S=8, W=3) and ~3.4 min at M=128 / S=20 / W=10.

    All rows share one time axis, so they share one resample index matrix: the
    draw happens once and the whole ``(n_resamples, n_elements, n_time)``
    resampled array is reduced in a single ``statistic(..., axis=-1)`` call
    (chunked over replicates to bound the temporary, which does not change any
    value -- the reduction is along time only).

    **The statistic contract differs from the scalar function's.** There it is
    ``statistic(1-D array) -> float``; here it must be an *axis-taking reducer*,
    called as ``statistic(x, axis=-1)`` on a 3-D array and returning ``x``'s
    leading shape. That is what makes the vectorization possible. ``np.mean``
    and ``np.var`` qualify as they are; a closure must be written to take and
    forward ``axis`` (e.g. ``lambda x, axis: np.var(x, axis=axis, ddof=1)``)
    rather than assuming a flat series.

    **Non-finite handling: a row containing any non-finite sample returns
    ``nan``**, rather than falling back to the scalar path for that row. The
    scalar version drops non-finite samples before blocking, which the batch
    cannot do uniformly -- a row with gaps has a different finite count, hence a
    different block length and a different index matrix, which is precisely the
    sharing that makes this fast. WP1.2's rows are either fully finite or
    absent altogether (a masked sensor is dropped upstream, not passed as a row
    of gaps), so the ``nan`` is a report of an input this path does not serve
    rather than a silent approximation. A caller with genuinely gappy rows
    should loop :func:`block_bootstrap_std` over them and get the documented
    gap-dropping behaviour.

    Args:
        series: ``(n_elements, n_time)``, **time last**, one series per row.
            Every row is bootstrapped over the same time axis.
        statistic: Axis-taking reduction, called as ``statistic(x, axis=-1)``;
            see the contract above.
        n_blocks: Target number of blocks; the block length ``L = ceil(n /
            n_blocks)`` is derived from the row length, as in the scalar form.
        n_resamples: Number of bootstrap replicates.
        rng: ``numpy`` generator, or a seed for one. ``None`` means the fixed
            default seed, so the default path is reproducible.

    Returns:
        ``(n_elements,)`` bootstrap standard errors, ``nan`` where undefined:
        rows containing a non-finite sample, and *all* rows when the shared
        time axis is shorter than four samples or gives ``L < 2`` (the smoke
        shape). ``n_blocks=1`` returns exactly ``0.0`` for every finite row, a
        measured zero -- see :func:`block_bootstrap_std`'s Returns section for
        why that is not an error.

    Raises:
        ValueError: If ``series`` is not 2-D, if ``n_blocks < 1`` or
            ``n_resamples < 2``, or if ``statistic`` does not honour the
            ``axis=-1`` contract (detected as a wrong output shape).
    """
    _validate_bootstrap_shape(n_blocks, n_resamples)

    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            "series must be 2-D (n_elements, n_time) with time last, got shape "
            f"{values.shape}"
        )
    n_elements, n_time = values.shape
    result = np.full(n_elements, float("nan"))
    if n_elements == 0 or n_time < 4:
        return result
    block_len = int(np.ceil(n_time / n_blocks))
    if block_len < 2:
        return result

    # Whole rows, not individual samples: see the docstring.
    usable = np.isfinite(values).all(axis=1)
    if not bool(usable.any()):
        return result
    rows = values[usable]
    n_rows = int(rows.shape[0])

    generator = _resolve_generator(rng)
    index = _block_resample_indices(n_time, block_len, n_resamples, generator)

    # ``replicates`` is (rows, resamples) so each row's replicate vector is
    # contiguous and its ``ddof=1`` spread is taken over exactly the 1-D array
    # the scalar function would have taken it over.
    replicates = np.empty((n_rows, n_resamples), dtype=float)
    chunk = int(np.clip(_BATCH_CHUNK_BYTES // (n_rows * n_time * 8), 1, n_resamples))
    for start in range(0, n_resamples, chunk):
        stop = min(start + chunk, n_resamples)
        # ``ascontiguousarray`` so every reduced series is a contiguous run of
        # ``n_time`` doubles, exactly as in the scalar path -- a transposed view
        # would be free to sum in a different order and move the last bits.
        resampled = np.ascontiguousarray(rows[:, index[start:stop]].swapaxes(0, 1))
        reduced = np.asarray(statistic(resampled, axis=-1), dtype=float)
        if reduced.shape != resampled.shape[:-1]:
            raise ValueError(
                "statistic must reduce along axis=-1 and return the leading "
                f"shape {resampled.shape[:-1]}, got {reduced.shape}; see the "
                "axis contract in block_bootstrap_std_batch's docstring"
            )
        replicates[:, start:stop] = reduced.T

    out = np.array([_replicate_spread(replicates[i]) for i in range(n_rows)])
    result[usable] = out
    return result
