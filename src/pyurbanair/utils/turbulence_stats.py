"""Flow-statistics helpers for the ESMDA evaluation stack (WP1.3).

Companion to :mod:`pyurbanair.utils.ensemble_scores`: that module scores
ensembles probabilistically, this one turns raw solver output into the
turbulence quantities worth scoring, so the metrics stage, the figure stage
and ``scripts/figspec`` share one implementation.

Four layers, in the order a driver uses them:

* :func:`colocate_components` -- staggered-grid velocity components
  interpolated onto cell centres, per backend (uDALES/PALM staggering,
  pass-through for the uniform pylbm and surrogate grids). The only part of
  this module that touches xarray; everything below is plain numpy.
* :class:`StreamingMoments` -- chunk-wise accumulation of ``n``, ``sum u_i``
  and ``sum u_i u_j`` (``i <= j``) giving means, Reynolds stresses and TKE
  without ever holding a window state file in memory.
* the mean-field scores over fluid cells -- :func:`hit_rate`, :func:`fac2`,
  :func:`fractional_bias`, :func:`nmse` and :func:`nmse_split` -- the
  urban-CFD standards of ``docs/plans/esmda_turbulence_evaluation.md``
  section 4.1.
* :func:`block_bootstrap_std` (and its batched sibling
  :func:`block_bootstrap_std_batch`), the sampling floor those scores are read
  against -- ``hit_rate``'s absolute allowance among them. It landed with
  WP1.2, whose sensor statistics needed it first, rather than being written
  twice.

Specification: ``docs/plans/esmda_evaluation/phase1_postprocessing_metrics.md``
(WP1.3); metric definitions and acceptance thresholds:
``docs/plans/esmda_turbulence_evaluation.md`` section 4.1.

One thing no function here does: decide which cells are solid. No backend
marks blocked cells with NaN (measured -- see :func:`colocate_components`), so
the "fluid cells only" rule is a mask the *caller* applies; these functions
only promise to handle the NaN they are given without letting it spread.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for the annotations
    # Annotation-only: :func:`colocate_components` reaches xarray objects
    # through their own methods, so nothing here needs xarray at import time
    # and this module stays a numpy-only import for ``ensemble_scores``.
    import xarray

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

__all__: list[str] = [
    "DEFAULT_HIT_RATE_TOLERANCE",
    "StreamingMoments",
    "block_bootstrap_std",
    "block_bootstrap_std_batch",
    "colocate_components",
    "extrapolated_centre_dims",
    "fac2",
    "fractional_bias",
    "hit_rate",
    "nmse",
    "nmse_split",
]


# ---------------------------------------------------------------------------
# Sampling floors: the moving-block bootstrap (WP1.2)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Streaming one-point moments
# ---------------------------------------------------------------------------


class StreamingMoments:
    """Per-cell means, Reynolds stresses and TKE accumulated chunk by chunk.

    The one-point statistics layer of WP1.3: fed time chunks of already
    co-located components (see :func:`colocate_components`), it holds only
    per-cell accumulators, so the window state files are never materialised.
    One instance covers one (member, region) pair and is fed **across all
    windows** -- ``update`` has no notion of a window, so the statistics
    reported are over the whole run, which is what the plan's ``n_windows``
    bookkeeping records.

    **State, and why it is small.** Per cell: one sample count, ``C`` running
    means and the ``C(C+1)/2`` co-moments of the ``i <= j`` pairs -- 10 arrays
    for the usual ``C = 3``, independent of how many frames go through it:

    ====================================  ==============  ==========
    region (per member)                   cells           bytes
    ====================================  ==============  ==========
    4 z-slabs of 64 x 64                  16384           1.3 MB
    4 z-slabs of 256 x 256                262144          21 MB
    4 z-slabs of 512 x 512                1048576         84 MB
    8 station columns, 64 levels          512             41 kB
    ====================================  ==============  ==========

    So a 32-member ensemble's slab accumulators are ~0.7 GB at the 256 x 256
    shape and its station columns are free -- the driver can hold one per
    member for the whole run, which is what the shared-pass requirement needs.
    At the plan's Barcelona target (M = 128, 4 z-slabs of 512 x 512) the same
    arithmetic is **10 GB**, held for the whole pass, so that shape needs the
    caller's horizontal stride rather than more memory (see
    ``_esmda_common.MeanFieldAccumulator``).
    The *transient* cost is the caller's chunk: ``update`` allocates about
    ``C + 1`` chunk-sized temporaries, so chunk length is what to turn down if
    the pass gets tight, not the number of accumulators. Measured throughput on
    a 36-frame chunk of three components: 29 ms at 4 x 128 x 128 (57 MB of
    chunk, 1.9 GB/s) and 141 ms at 4 x 256 x 256 (226 MB, 1.6 GB/s), i.e.
    ~0.14 s per member per window at the larger shape -- bandwidth-bound and
    well under the cost of reading the same bytes off disk, which is the point
    of accumulating inside the shared read pass. A chunk containing NaN costs
    the same (measured 141 ms), so there is no fast path to miss.

    **NaN policy: per-cell counts, casewise deletion.** ``n`` is an array over
    cells, not a scalar, and a frame contributes to a cell only if *every*
    component is finite there. Both halves of that are deliberate:

    * A scalar ``n`` would force the accumulator to choose between dropping a
      whole frame because one cell is masked and counting a masked cell as a
      sample. Blocked cells are a property of a cell, not of a frame, and a
      truth field interpolated onto the assimilation grid comes back with NaN
      at the edge cells the interpolation cannot reach -- so a per-cell count
      is not a refinement, it is the only version that gives the interior
      cells their correct sample size.
    * Casewise (rather than pairwise) deletion keeps ``R_ij`` a *covariance*.
      Counting each pair over its own available frames -- "available-case"
      deletion -- mixes sample sets between the entries of one matrix, which
      can make ``R`` indefinite and ``tke`` negative at a cell that is masked
      intermittently in only one component. One count per cell, shared by the
      mean and every co-moment, cannot do that.

    Note what this does **not** do: it never decides which cells are solid.
    No backend marks blocked cells with NaN (measured -- see
    :func:`colocate_components`), so masking is the caller's job; this class
    only promises to handle the NaN it is given without poisoning its
    neighbours in time or in space.

    **Bessel correction: ``ddof = 1`` by default, on the reduced moment.**
    ``reynolds_stress()`` returns
    ``sum_t (u_i - U_i)(u_j - U_j) / (n - 1)``, i.e. the unbiased sample
    covariance, and ``tke()`` is ``0.5 * trace`` of exactly that -- so
    ``tke()`` equals ``0.5 * sum_i var_ddof1(u_i)``, the same estimator WP1.2's
    sensor TKE reports (there the ``n/(n-1)`` factor is applied to the
    per-timestep integrand so that a bootstrap resamples the reported
    estimator; here there is no resampling, so the factor sits on the reduced
    value and the two agree exactly). ``ddof=0`` is available for a caller that
    wants the plain time average ``<u_i u_j> - U_i U_j``; the difference is
    ``n/(n-1)``, i.e. 3% at a 36-frame window and 0.3% at 360 frames, and it
    matters most exactly where the sample is smallest.

    **Numerical formulation, and the measurement that forced it.** The
    accumulation is the chunk-wise (Chan/Welford) combine, *not* the literal
    ``sum u_i u_j - (sum u_i)(sum u_j)/n`` of the plan text: each chunk's
    co-moment is formed about that chunk's own mean and merged with a
    ``delta_i * delta_j * n*m/(n+m)`` correction term. The plan's textbook form
    subtracts two nearly equal large numbers, and this layer's regime is
    exactly the one where that fails -- urban flow at a mean of 5 m/s carries
    fluctuations of ~0.2 m/s, so the mean square is ~600x the variance being
    recovered. Measured relative error of the ``ddof=1`` variance against the
    exact variance of the stored samples (two-pass in ``np.longdouble``), n =
    360 frames fed in 10 chunks of 36; "naive" is the plan's
    ``sum uu - (sum u)**2/n`` accumulated the same way, "two-pass" is a
    ``float64`` two-pass over the whole series held in memory:

    ==================  ========  ==========  ========
    mean / fluctuation  naive     streaming   two-pass
    ==================  ========  ==========  ========
    5 / 0.2 (25x)       8.0e-14   1.7e-16     <1e-16
    5 / 0.002 (2500x)   1.6e-09   4.4e-15     <1e-16
    1e4 / 1e-3 (1e7x)   3.9e-02   7.1e-11     <1e-16
    ==================  ========  ==========  ========

    The cross moment behaves the same (``<u'w'>`` at correlation 0.6, same
    shapes: naive 4.2e-14 / 9.9e-10 / 1.1e-02, streaming 0.0 / 7.5e-15 /
    7.7e-11). Read the rows in order. At the shipped regime the naive form is
    still accurate enough to report, so this is insurance rather than a rescue;
    by 2500x it has lost six digits of a number the identifiability layer
    compares against a bootstrap floor; and the last row -- **a 3.9% error, on
    data that is not pathological**, just a field with a large offset relative
    to its fluctuation -- is what rules the naive form out. The streaming form
    is within a factor of ~2 of a two-pass computation over the whole series at
    the shipped regime, i.e. chunking costs nothing numerically. The plan's
    ``sum u_i u_j`` is a description of what is accumulated, not a prescription
    of the subtraction order.

    Args:
        None. The component count and cell shape are taken from the first
        :meth:`update`; every later call must match them.
    """

    def __init__(self) -> None:
        # All state is allocated on the first update, so one class serves the
        # z-slabs and the station columns without the caller restating shapes
        # the data already carries.
        self._count: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._comoment: np.ndarray | None = None
        self._pairs: tuple[tuple[int, int], ...] = ()
        self._n_components = 0
        self._cell_shape: tuple[int, ...] = ()

    @property
    def n_components(self) -> int:
        """Number of components being accumulated; ``0`` before the first update."""
        return self._n_components

    @property
    def cell_shape(self) -> tuple[int, ...]:
        """Shape of one frame (the chunk shape minus its leading time axis)."""
        return self._cell_shape

    def count(self) -> np.ndarray:
        """Per-cell number of frames in which **all** components were finite.

        Returns:
            ``cell_shape`` array of ``int64``, a copy (so a caller cannot
            corrupt the accumulator by writing into the reported counts).

        Raises:
            ValueError: If nothing has been accumulated yet.
        """
        return self._state()[0].copy()

    def update(self, *components: np.ndarray) -> None:
        """Fold one time chunk of co-located components into the accumulators.

        Args:
            *components: One array per component, **time first**, all of the
                same shape ``(n_time, *cell_shape)``. ``colocate_components``'s
                three returns, sliced to a chunk of frames and to the region of
                interest, are the intended input; ``np.asarray`` is applied, so
                DataArrays and ``float32`` fields are accepted (accumulation is
                always in ``float64``). A single frame must be passed with an
                explicit length-1 time axis -- there is no way to detect a
                missing one, and the leading axis is *always* read as time.

        Raises:
            ValueError: If no components are given, if their shapes differ from
                each other, if an array has no time axis at all, or if the
                component count or cell shape differs from the first call. The
                last case is what catches a caller who selected z-levels
                between chunks.
        """
        if not components:
            raise ValueError("update needs at least one component array")
        chunk = [np.asarray(component, dtype=float) for component in components]
        shapes = {component.shape for component in chunk}
        if len(shapes) != 1:
            raise ValueError(
                f"all components must share one shape, got {sorted(shapes)}"
            )
        if chunk[0].ndim < 1:
            raise ValueError(
                "components need a leading time axis; pass a single frame as "
                "field[None] rather than as a bare frame"
            )
        n_time = int(chunk[0].shape[0])
        cell_shape = tuple(chunk[0].shape[1:])

        if self._count is None:
            self._initialise(len(chunk), cell_shape)
        elif (len(chunk), cell_shape) != (self._n_components, self._cell_shape):
            raise ValueError(
                f"expected {self._n_components} components of cell shape "
                f"{self._cell_shape} (fixed by the first update), got "
                f"{len(chunk)} of {cell_shape}"
            )
        if n_time == 0:
            return

        count, mean, comoment = self._state()

        # Casewise validity: one mask for all components, so the mean and every
        # co-moment are taken over the same frames at every cell.
        valid = np.isfinite(chunk[0])
        for component in chunk[1:]:
            valid &= np.isfinite(component)
        n_new = valid.sum(axis=0)
        if not bool(np.any(n_new)):
            return
        contributing = n_new > 0

        # Deviations from the *running* mean rather than from zero: on every
        # chunk but the first this is already the small quantity, which is what
        # keeps the chunk mean itself accurate for large-offset fields.
        deviation = [
            np.where(valid, component - mean[i], 0.0)
            for i, component in enumerate(chunk)
        ]
        safe_new = np.where(contributing, n_new, 1)
        delta = np.stack([d.sum(axis=0) / safe_new for d in deviation])

        # Each chunk's co-moment is formed about the chunk's own mean (a
        # two-pass computation *within* the chunk), and the shift between the
        # chunk mean and the running mean is added back exactly. This is the
        # whole numerical argument for the class; see the class docstring's
        # measured table.
        centred = [np.where(valid, d - delta[i], 0.0) for i, d in enumerate(deviation)]
        total = count + n_new
        safe_total = np.where(total > 0, total, 1)
        cross_weight = count * n_new / safe_total
        for pair, (i, j) in enumerate(self._pairs):
            comoment[pair] += (centred[i] * centred[j]).sum(axis=0) + (
                delta[i] * delta[j] * cross_weight
            )
        mean += delta * (n_new / safe_total)
        self._count = total

    def mean(self) -> np.ndarray:
        """Per-cell time means ``U_i``.

        Returns:
            ``(n_components, *cell_shape)`` array; ``nan`` at cells with no
            finite frames (a solid cell the caller masked, or a cell outside
            an interpolated truth field).

        Raises:
            ValueError: If nothing has been accumulated yet.
        """
        count, mean, _ = self._state()
        return np.where(count > 0, mean, np.nan)

    def reynolds_stress(self, ddof: int = 1) -> np.ndarray:
        """Per-cell resolved Reynolds stresses ``R_ij = <u_i u_j> - U_i U_j``.

        **Resolved only.** These are moments of the fields the solver wrote;
        the subgrid contribution is not in them and is not negligible inside a
        canopy (metrics doc section 4.1 asks for this to be stated wherever the
        stresses are reported).

        Args:
            ddof: Denominator correction. The default ``1`` is the unbiased
                sample covariance ``sum/(n - 1)``, matching the ``ddof=1``
                variances WP1.2 reports for the same quantity at sensors;
                ``ddof=0`` gives the plain time average.

        Returns:
            ``(n_components, n_components, *cell_shape)``, symmetric by
            construction (the ``i > j`` entries are the accumulated ``i <= j``
            ones, not a second estimate). ``nan`` at cells with ``n <= ddof``,
            which includes every masked cell and -- at ``ddof=1`` -- a cell
            seen in exactly one frame.

        Raises:
            ValueError: If nothing has been accumulated yet, or ``ddof < 0``.
        """
        scale = self._moment_scale(ddof)
        comoment = self._state()[2]
        n_components = self._n_components
        stress = np.empty((n_components, n_components) + self._cell_shape)
        for pair, (i, j) in enumerate(self._pairs):
            value = comoment[pair] * scale
            stress[i, j] = value
            stress[j, i] = value
        return stress

    def tke(self, ddof: int = 1) -> np.ndarray:
        """Per-cell resolved TKE ``k = 0.5 * R_ii``.

        Computed from the diagonal co-moments directly rather than by tracing
        :meth:`reynolds_stress`, so it costs one array instead of ``C**2``.

        Args:
            ddof: As in :meth:`reynolds_stress`; ``1`` (the default) makes this
                exactly ``0.5 * sum_i var_ddof1(u_i)``.

        Returns:
            ``cell_shape`` array, ``nan`` where the stresses are.

        Raises:
            ValueError: If nothing has been accumulated yet, or ``ddof < 0``.
        """
        scale = self._moment_scale(ddof)
        comoment = self._state()[2]
        diagonal = np.zeros(self._cell_shape, dtype=float)
        for pair, (i, j) in enumerate(self._pairs):
            if i == j:
                diagonal += comoment[pair]
        return np.asarray(0.5 * diagonal * scale)

    def _initialise(self, n_components: int, cell_shape: tuple[int, ...]) -> None:
        """Allocate the accumulators once the first chunk has fixed the shapes."""
        self._n_components = n_components
        self._cell_shape = cell_shape
        self._pairs = tuple(
            (i, j) for i in range(n_components) for j in range(i, n_components)
        )
        self._count = np.zeros(cell_shape, dtype=np.int64)
        self._mean = np.zeros((n_components, *cell_shape), dtype=float)
        self._comoment = np.zeros((len(self._pairs), *cell_shape), dtype=float)

    def _state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(count, mean, comoment)``, or a readable error if nothing was fed yet.

        The accumulators are allocated on the first :meth:`update` (only the
        data knows the shapes), so every reader goes through here rather than
        letting an unfed instance fail with an ``AttributeError`` deep in numpy.
        """
        if self._count is None or self._mean is None or self._comoment is None:
            raise ValueError(
                "no frames accumulated yet: StreamingMoments takes its shapes "
                "from the first update, so there is nothing to report"
            )
        return self._count, self._mean, self._comoment

    def _moment_scale(self, ddof: int) -> np.ndarray:
        """``1/(n - ddof)`` per cell, ``nan`` where that is not a sample size."""
        if ddof < 0:
            raise ValueError(f"ddof must be non-negative, got {ddof}")
        count = self._state()[0]
        denominator = count - ddof
        usable = denominator > 0
        return np.where(usable, 1.0 / np.where(usable, denominator, 1), np.nan)


# ---------------------------------------------------------------------------
# Staggered-grid colocation
# ---------------------------------------------------------------------------

# Per solver, the staggered dim each velocity component must be moved off and
# the cell-centre dim it must land on. Authority: ``ObservationOperator``'s
# ``dim_mapping`` (``libs/data-assimilation/src/data_assimilation/
# observation_operator.py``), read rather than remembered -- the same three
# solver names, the same axes.
#
# An empty tuple means "already co-located": pylbm writes one uniform grid, and
# ``conf/model/neural_surrogate.yaml`` sets ``solver_name: pylbm`` (the
# surrogate reuses its spin-up backend's grid, and ``ObservationOperator``
# rejects the name ``neural_surrogate`` outright), so the surrogate never
# actually reaches this table under a shipped config -- the alias is here so a
# future config that does name it gets the pass-through rather than a raise.
#
# ``palm``'s ``w`` is listed even though pypalm's postprocess has already
# interpolated it from ``zw_3d`` onto ``z`` and dropped ``zw``
# (``libs/pypalm/src/pypalm/forward_model.py``, the ``w_on_z`` block). Pairs
# whose staggered dim is absent are skipped, so the entry is a no-op on a
# post-processed state and correct on a raw one.
_STAGGERED_TO_CENTRE: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "udales": {"u": (("xm", "xt"),), "v": (("ym", "yt"),), "w": (("zm", "zt"),)},
    "palm": {"u": (("xu", "x"),), "v": (("yv", "y"),), "w": (("zw", "z"),)},
    "pylbm": {"u": (), "v": (), "w": ()},
    "neural_surrogate": {"u": (), "v": (), "w": ()},
}

# The velocity triple, in the order every consumer expects.
_VELOCITY_COMPONENTS = ("u", "v", "w")

# How far a cell centre may sit from the midpoint of its two bounding faces,
# as a fraction of the local face spacing, before colocation refuses to run.
# ``centre[i] == 0.5 * (face[i] + face[i+1])`` is the *definition* of a cell
# centre, so in exact arithmetic the tolerance could be zero; it exists only to
# absorb the rounding of coordinates stored as ``float32`` and offset by the
# case bounds (~1e-7 relative). At 1e-3 of a cell it is ~1e4 times that
# rounding and still 100x smaller than the smallest slicing mistake it has to
# catch (a stride-2 level selection displaces the midpoint by half a cell).
_FACE_CENTRE_TOLERANCE = 1e-3


def colocate_components(
    ds: "xarray.Dataset", solver_name: str
) -> tuple["xarray.DataArray", "xarray.DataArray", "xarray.DataArray"]:
    """Velocity components interpolated onto cell centres, per backend.

    Reynolds stresses are *one-point* moments, so every component has to be
    evaluated at the same point before they are multiplied together. On a
    staggered grid they are not: uDALES stores ``u`` on the ``x``-faces, ``v``
    on the ``y``-faces and ``w`` on the ``z``-faces (``ObservationOperator.
    dim_mapping``), and PALM stores ``u`` on ``xu`` and ``v`` on ``yv``.

    **Why not the by-index shortcut.** ``_esmda_common._vel_field_4z`` combines
    the three components by array index -- ``sqrt(u[k,j,i]**2 + v[k,j,i]**2 +
    w[k,j,i]**2)`` -- and the plan is explicit that this must not be reused
    here. ``u[k,j,i]`` and ``w[k,j,i]`` sit half a cell apart in *both* x and z,
    so an ``<u'w'>`` formed from them is a **two-point** cross correlation at
    lag ``(dx/2, 0, dz/2)``, while the diagonal ``<u'u'>`` is a one-point
    moment at the face -- every entry of the tensor is evaluated somewhere else.
    Measured on a homogeneous synthetic field with a prescribed Gaussian
    correlation length ``L``, fluctuation correlation 0.6, 6000 frames, as
    relative error against the exact centre-point tensor:

    ======  ==========  ==========  ==========
    L / dx  quantity    by index    colocated
    ======  ==========  ==========  ==========
    1.0     <u'w'>        -22.1%      -22.2%
    1.0     <u'u'>          0.0%      -19.7%
    1.0     <u'w'>/k      -22.1%       -3.1%
    2.0     <u'w'>         -6.0%       -6.1%
    2.0     <u'u'>          0.0%       -5.9%
    2.0     <u'w'>/k       -6.0%       -0.2%
    4.0     <u'w'>         -1.5%       -1.5%
    4.0     <u'u'>         -0.0%       -1.5%
    4.0     <u'w'>/k       -1.5%       -0.0%
    ======  ==========  ==========  ==========

    Read that table before believing the obvious story. Colocating does **not**
    recover the amplitude of a grid-scale stress: linear interpolation is a
    low-pass filter, so it loses the same 22% on ``<u'w'>`` at ``L = dx`` that
    the half-cell lag loses, and it *adds* a 20% loss on the diagonal, where
    the by-index value was unbiased. What it fixes is the **consistency** of
    the tensor: the filter is the same for every entry, so the anisotropy ratio
    ``<u'w'>/k`` -- which the metrics doc names as far more discriminating than
    ``k`` alone, since a model can match TKE with entirely wrong anisotropy --
    goes from a 22% error to 3%, and from 6% to 0.2% as soon as the
    fluctuations are resolved at all. By-index biases that ratio by the
    *staggering pattern* rather than by the flow, which is a bias no amount of
    ensemble averaging removes and which differs between backends. Neither
    method is a substitute for resolving the fluctuation: at ``L = dx`` both
    numbers are bad, and that is a statement about the grid.

    The mean field is the unambiguous half: with a linear profile of gradient
    ``0.1`` per cell, by-index reports ``U`` half a cell off (error ``0.05``,
    i.e. ``0.5*dx*dU/dx``, first order and not averaged away by time or
    ensemble) while colocation is exact to ``2e-16``. For a ``|U|`` magnitude
    summary that displacement is an ordinary smooth-field interpolation error
    and harmless, which is why the shortcut is fine where it is used and wrong
    here.

    **Edge handling: linear extrapolation over at most half a cell.** Each
    solver stores one face per cell (the lower one: measured
    ``xm - xt = -dx/2`` on ``.temp/pyudales_to_pyudales/true_state.nc``), so the
    face and centre axes have the *same* length and the last centre has no
    upper face. It is filled by linear extrapolation from the last two faces
    (weights 1.5 and -0.5), the same ``fill_value="extrapolate"`` choice
    pypalm's own ``zw -> z`` interpolation of ``w`` makes. The alternative --
    dropping the last column -- was rejected because it silently changes the
    returned grid: the caller interpolates truth onto these coordinates and
    scores cell against cell, so a component that is one column shorter than
    its centre axis either misaligns the comparison or forces every consumer to
    re-derive which column went missing. Two honest costs of extrapolating: the
    last column's second moments are inflated, and a periodic domain would be
    better served by wrapping -- which cannot be done here, since the boundary
    condition is not recoverable from the state file.

    **How inflated, measured against the right reference.** The weights amplify
    the variance of two adjacent face samples by ``1.5**2 + 0.5**2 = 2.5``
    relative to *one unfiltered sample*, but no interior cell of the returned
    field is one unfiltered sample either -- it is the midpoint average, whose
    variance is ``0.5 * (1 + rho)``. The number a consumer actually needs is the
    ratio of the two, ``(2.5 - 1.5*rho) / (0.5 + 0.5*rho)`` at face-to-face
    correlation ``rho``:

    ======  =======================
    rho     last column / interior
    ======  =======================
    0.0     5.0x  (measured 5.06x)
    0.5     2.33x
    0.9     1.21x
    ======  =======================

    So it is ~20% for a well-resolved field and 5x only for face-to-face white
    noise, i.e. the penalty is worst exactly where the field is least resolved.
    It is also not avoided by telling a caller to "drop the last index": an
    evenly spaced level selection (``np.linspace(0, n-1, k)``) always *includes*
    it, so the edge lands on a scored level unless something excludes it on
    purpose. :func:`extrapolated_centre_dims` reports which axes carry the edge
    so a caller can keep that index out of its *aggregates* -- rather than out
    of the returned data, which would change the grid.

    **Solid cells are not marked, in any backend.** The plan's parenthesis
    ("solid cells are the NaN/blocked cells -- verify") does not hold for the
    shipped artifacts, verified in code and on the artifacts under ``.temp``:
    pylbm writes near-exact zeros in obstacle interiors *plus* a ``blanking``
    variable (1 = solid) in the same Dataset; pypalm's postprocess explicitly
    replaces PALM's NaN with 0.0 (no-slip) and keeps no mask; uDALES fielddumps
    carry small non-zero junk inside buildings and ship no mask at all.
    Measured NaN fraction of ``u``/``v``/``w`` in every state artifact under
    ``.temp``: 0.0. This function therefore does not attempt to mask, and
    interpolation is *not* a way to discover the mask either -- a zero-filled
    solid face pulls its neighbouring centre halfway to zero. Masking is the
    caller's step, from ``blanking`` where the backend provides it.

    Args:
        ds: A state Dataset (or one member of one) carrying ``u``, ``v``, ``w``
            and the coordinates of both the staggered and the centre axes. It
            must **not** have been subset along an axis that still needs
            colocating -- colocate first, then select z-levels or stride the
            grid. Interpolating between two z-levels five cells apart is
            arithmetically fine and physically meaningless, so it is refused
            (see Raises) rather than silently performed.
        solver_name: ``udales``, ``palm``, ``pylbm`` or ``neural_surrogate``,
            i.e. ``cfg.assim_model.solver_name`` / the ``assim_solver_name``
            recorded in the run's ``truth_access.yaml``.

    Returns:
        ``(u, v, w)`` as ``xarray.DataArray``s, all three on the centre axes
        and carrying the centre coordinates, so they share one grid and can be
        multiplied cell by cell (and handed to ``.interp`` for a cross-grid
        truth comparison). Every non-spatial dim (``time``, and ``ensemble`` if
        present) is passed through untouched. **DataArrays, not bare arrays**
        -- ``np.asarray`` is one call away, while re-attaching the coordinates
        is not.

    Raises:
        ValueError: If ``solver_name`` is not one of the four; if a component
            is missing from ``ds``; if a staggered axis lacks either its own or
            its centre axis's coordinate values (there is then no way to know
            where the samples are); if a staggered axis has a single level (one
            point defines no interpolant); or if the centres are not the
            midpoints of the faces, which is what a pre-colocation subset looks
            like from in here.
    """
    try:
        mapping = _STAGGERED_TO_CENTRE[solver_name]
    except KeyError:
        raise ValueError(
            f"solver {solver_name!r} has no known staggering; expected one of "
            f"{sorted(_STAGGERED_TO_CENTRE)}"
        ) from None

    colocated = []
    for name in _VELOCITY_COMPONENTS:
        if name not in ds:
            raise ValueError(
                f"state is missing the {name!r} component; colocation needs all "
                f"of {_VELOCITY_COMPONENTS}"
            )
        field = ds[name]
        for face_dim, centre_dim in mapping[name]:
            if face_dim not in field.dims:
                # Already co-located on this axis (pypalm's postprocessed w),
                # so there is nothing to move.
                continue
            field = _interp_face_to_centre(ds, field, name, face_dim, centre_dim)
        colocated.append(field)
    return colocated[0], colocated[1], colocated[2]


def extrapolated_centre_dims(ds: "xarray.Dataset", solver_name: str) -> tuple[str, ...]:
    """Centre axes whose **last** index :func:`colocate_components` extrapolates.

    The companion to the edge-handling paragraph of
    :func:`colocate_components`: each solver stores one face per cell, so the
    last centre of every colocated axis is filled from outside the data with
    weights ``(1.5, -0.5)`` and its second moments are inflated (5x against an
    interior cell for face-to-face white noise, ~1.2x for a well-resolved
    field). That index is not a normal cell, and an evenly spaced level
    selection always lands on it.

    This reports **every** axis carrying an edge; what a caller does about each
    is the caller's decision, and they are not equivalent. The only consumer
    today (``_esmda_common.mean_field_summary``) acts on the **z** entry alone,
    because its z-levels are selected by an evenly spaced rule that always
    includes the top index, so the edge lands on a scored level and biases a
    whole plane. Its x and y slabs are reduced over in full, so the last x-row
    and y-column carry the same extrapolation into every aggregate -- a known
    residual, one row and one column out of the horizontal grid, left in
    deliberately rather than overlooked.

    Cheap and read-only -- it inspects dims, never values, and mirrors
    :func:`colocate_components`'s own skip rule (a pair whose staggered dim is
    absent, e.g. pypalm's already-postprocessed ``w``, moves nothing and
    therefore extrapolates nothing).

    Args:
        ds: The state Dataset that would be handed to
            :func:`colocate_components`.
        solver_name: Same value as :func:`colocate_components` would receive.

    Returns:
        The centre dim names, in ``(u, v, w)`` order and without duplicates. An
        unstaggered backend (``pylbm``) returns ``()``.

    Raises:
        ValueError: If ``solver_name`` has no known staggering, matching
            :func:`colocate_components` so the two cannot disagree about which
            names are supported.
    """
    try:
        mapping = _STAGGERED_TO_CENTRE[solver_name]
    except KeyError:
        raise ValueError(
            f"solver {solver_name!r} has no known staggering; expected one of "
            f"{sorted(_STAGGERED_TO_CENTRE)}"
        ) from None

    edges: list[str] = []
    for name in _VELOCITY_COMPONENTS:
        if name not in ds:
            continue
        for face_dim, centre_dim in mapping[name]:
            if face_dim in ds[name].dims and centre_dim not in edges:
                edges.append(centre_dim)
    return tuple(edges)


def _interp_face_to_centre(
    ds: "xarray.Dataset",
    field: "xarray.DataArray",
    name: str,
    face_dim: str,
    centre_dim: str,
) -> "xarray.DataArray":
    """Move one component off one staggered axis onto the matching centre axis.

    Linear interpolation, extrapolating at most the half cell at the top end
    (see :func:`colocate_components` for why extrapolation and not truncation).
    The interpolated dim is renamed to the centre dim so all three components
    come out on one grid.
    """
    face = _axis_values(ds, field, face_dim, name)
    centre = _axis_values(ds, None, centre_dim, name)
    if face.size < 2:
        raise ValueError(
            f"cannot colocate {name!r}: staggered axis {face_dim!r} has "
            f"{face.size} level(s), which defines no interpolant"
        )
    _check_face_centre_alignment(face, centre, name, face_dim, centre_dim)

    if face_dim not in field.coords:
        # ``interp`` needs the source positions on the array itself; a Dataset
        # can legitimately carry them at the top level only (e.g. after a
        # ``drop_vars`` round-trip), and re-attaching them is free.
        field = field.assign_coords({face_dim: face})
    moved = field.interp(
        {face_dim: centre}, method="linear", kwargs={"fill_value": "extrapolate"}
    )
    moved = moved.rename({face_dim: centre_dim}).assign_coords({centre_dim: centre})
    return moved


def _axis_values(
    ds: "xarray.Dataset",
    field: "xarray.DataArray | None",
    dim: str,
    name: str,
) -> np.ndarray:
    """Coordinate values of ``dim``, from the field if it has them, else from ``ds``."""
    if field is not None and dim in field.coords:
        return np.asarray(field[dim].values, dtype=float)
    if dim in ds.coords:
        return np.asarray(ds[dim].values, dtype=float)
    raise ValueError(
        f"cannot colocate {name!r}: axis {dim!r} carries no coordinate values, "
        "so the sample positions are unknown"
    )


def _check_face_centre_alignment(
    face: np.ndarray,
    centre: np.ndarray,
    name: str,
    face_dim: str,
    centre_dim: str,
) -> None:
    """Refuse to interpolate when the centres are not the midpoints of the faces.

    ``centre[i] == 0.5 * (face[i] + face[i+1])`` holds by definition on every
    staggered grid this repo reads, stretched or not. When it fails, the
    dataset was almost certainly subset along this axis before colocation --
    selecting four of twenty z-levels leaves both axes uniformly spaced, so a
    spacing check would pass while the interpolation quietly blended cells five
    apart. This is the cheap guard that turns that into an error at the top of
    the pass instead of a plausible-looking stress field at the end of it.
    """
    n_check = min(face.size - 1, centre.size)
    if n_check < 1:
        return
    spacing = np.diff(face)[:n_check]
    offset = centre[:n_check] - face[:n_check]
    with np.errstate(invalid="ignore", divide="ignore"):
        misfit = np.abs(offset - 0.5 * spacing) / np.abs(spacing)
    worst = float(np.nanmax(misfit)) if misfit.size else 0.0
    if worst > _FACE_CENTRE_TOLERANCE:
        raise ValueError(
            f"cannot colocate {name!r} from {face_dim!r} onto {centre_dim!r}: the "
            f"centres are not the midpoints of the faces (worst relative misfit "
            f"{worst:.3g} > {_FACE_CENTRE_TOLERANCE:g}). Either the dataset was "
            "subset along this axis before colocation -- colocate first, then "
            "select levels or stride -- or this solver does not store the lower "
            "face, which is the convention this repo's backends use"
        )


# ---------------------------------------------------------------------------
# Mean-field error norms (urban-CFD standards)
# ---------------------------------------------------------------------------

# The VDI 3783/9 relative tolerance the hit rate is defined at. Named so the
# metrics schema and the figure captions read the same number this module
# scores with, rather than each restating 0.25.
DEFAULT_HIT_RATE_TOLERANCE = 0.25


def hit_rate(
    pred: np.ndarray,
    obs: np.ndarray,
    *,
    allowance: float,
    relative_tolerance: float = DEFAULT_HIT_RATE_TOLERANCE,
) -> float:
    """VDI 3783/9 hit rate ``q``: fraction of points that agree within tolerance.

    A point counts if the relative error is within ``relative_tolerance``
    **or** the absolute error is within ``allowance``. The ``or`` is the whole
    design: this is the one urban-CFD standard metric built for *velocity*, and
    a velocity component passes through zero, where any relative criterion is
    meaningless. See ``docs/plans/esmda_turbulence_evaluation.md`` section 4.1
    (acceptance: ``q >= 0.66``).

    The relative test is evaluated as ``|p - o| <= D * |o|`` rather than
    ``|p/o - 1| <= D``, which is the same set of points without the division --
    so a zero-valued observation is decided by the absolute clause instead of
    producing an ``inf``.

    Args:
        pred: Predicted field (the ensemble-mean time-mean, typically).
        obs: Truth field, same shape. Pairs where either side is non-finite are
            dropped -- that is the "fluid cells only" filter, and it assumes the
            caller has already NaN-masked the solid cells, which no backend does
            for it (see :func:`colocate_components`).
        allowance: ``W``, the absolute error allowed regardless of relative
            error, in the field's units. **Required, with no default**, because
            there is no universal value: it is the truth's own sampling
            uncertainty ``sigma_u / sqrt(N_eff)``, which the driver gets from
            :func:`block_bootstrap_std` on the truth series at each point, and
            it turns ``q`` into an "indistinguishable within sampling error"
            test rather than an arbitrary threshold. Pass ``nan`` when the
            floor is unavailable (the smoke shape, where the bootstrap is
            undefined): the absolute clause is then skipped entirely and ``q``
            becomes the pure relative test, which can only be *lower* -- a
            missing floor never flatters the run.
        relative_tolerance: ``D``, default ``DEFAULT_HIT_RATE_TOLERANCE``.

    Returns:
        The hit fraction in ``[0, 1]``, or ``nan`` when no finite pair
        survives the mask.

    Raises:
        ValueError: If the shapes differ, if ``relative_tolerance`` is negative,
            or if ``allowance`` is negative (a negative tolerance admits no
            points and is always a bug, unlike ``nan``, which is a documented
            "not available").
    """
    if relative_tolerance < 0.0:
        raise ValueError(
            f"relative_tolerance must be non-negative, got {relative_tolerance}"
        )
    width = float(allowance)
    if width < 0.0:
        raise ValueError(
            f"allowance must be non-negative (or nan when unavailable), got {width}"
        )

    predicted, observed = _fluid_pairs(pred, obs)
    if predicted.size == 0:
        return float("nan")
    absolute_error = np.abs(predicted - observed)
    hit = absolute_error <= relative_tolerance * np.abs(observed)
    if np.isfinite(width):
        hit |= absolute_error <= width
    return float(np.mean(hit))


def fac2(pred: np.ndarray, obs: np.ndarray) -> float:
    """Fraction of points within a factor of two: ``0.5 <= pred/obs <= 2``.

    ``docs/plans/esmda_turbulence_evaluation.md`` section 4.1 (acceptance
    ``>= 0.5`` for dispersion, ``>= 0.3`` in use for urban-LES velocity).

    **Positive quantities only, enforced.** On a sign-changing quantity the
    ratio test is not a weak metric, it is a meaningless one: two values of
    opposite sign give a negative ratio that fails, while two negative values
    pass, so the score mixes "wrong" with "agrees" according to the sign of the
    truth. This function therefore *raises* on a negative input rather than
    documenting the trap -- the plan's guidance ("apply to ``|U|``, never to
    signed components") is only enforceable here, since by the time the number
    reaches ``run_summary.yaml`` it looks exactly like a valid one. Use
    :func:`hit_rate`, whose absolute allowance handles sign changes, for the
    signed components.

    Evaluated as ``0.5*o <= p <= 2*o``, the division-free form of the same
    test. One consequence to be aware of: a pair that is zero on both sides
    counts as a hit. That is the right answer for a stagnation point and the
    wrong one for a solid cell -- and pypalm zero-*fills* solid cells while
    pylbm writes near-zeros there, so an unmasked FAC2 is optimistic by
    whatever fraction of the domain is buildings. Mask first.

    Args:
        pred: Predicted field, non-negative.
        obs: Truth field, same shape, non-negative. Non-finite pairs are
            dropped (fluid cells only).

    Returns:
        The fraction in ``[0, 1]``, or ``nan`` when no finite pair survives.

    Raises:
        ValueError: If the shapes differ, or if any surviving pair holds a
            negative value.
    """
    predicted, observed = _fluid_pairs(pred, obs)
    negative = int(np.count_nonzero((predicted < 0.0) | (observed < 0.0)))
    if negative:
        raise ValueError(
            f"fac2 is defined for positive quantities only, but {negative} of "
            f"{predicted.size} fluid points are negative. A ratio test on a "
            "sign-changing quantity is meaningless -- score |U| or TKE with "
            "fac2 and the signed components with hit_rate's absolute allowance"
        )
    if predicted.size == 0:
        return float("nan")
    within = (predicted >= 0.5 * observed) & (predicted <= 2.0 * observed)
    return float(np.mean(within))


def fractional_bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Fractional bias ``FB = (obs_mean - pred_mean) / (0.5 * (obs_mean + pred_mean))``.

    ``docs/plans/esmda_turbulence_evaluation.md`` section 4.1 (acceptance
    ``|FB| <= 0.3``). Sign convention follows that formula: **positive FB means
    the prediction is too small**.

    For non-negative quantities ``FB`` is bounded in ``[-2, 2]`` and reads as a
    normalized bias. For a signed velocity component it is not bounded at all:
    the denominator is the *sum* of the means, which can pass through zero
    while both fields are perfectly ordinary, and ``FB`` then blows up. The
    number is returned as defined rather than clamped -- but a large ``|FB|``
    on a component whose mean is near zero is a statement about the
    denominator, not about the run, and ``|FB| >= 2`` makes
    :func:`nmse_split` undefined (it reports ``nan`` there).

    Args:
        pred: Predicted field.
        obs: Truth field, same shape. Non-finite pairs are dropped (fluid cells
            only).

    Returns:
        ``FB``, or ``nan`` when no finite pair survives or the two means cancel
        exactly.

    Raises:
        ValueError: If the shapes differ.
    """
    predicted, observed = _fluid_pairs(pred, obs)
    if predicted.size == 0:
        return float("nan")
    mean_pred = float(np.mean(predicted))
    mean_obs = float(np.mean(observed))
    denominator = 0.5 * (mean_obs + mean_pred)
    if denominator == 0.0:
        return float("nan")
    return (mean_obs - mean_pred) / denominator


def nmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Normalized mean square error ``<(o - p)**2> / (obs_mean * pred_mean)``.

    ``docs/plans/esmda_turbulence_evaluation.md`` section 4.1 (acceptance
    ``NMSE <= 4``). The normalization by the product of the means is what makes
    the systematic/unsystematic split of :func:`nmse_split` exact, so it is not
    interchangeable with a variance-normalized error.

    That normalization also inherits the positivity assumption of the
    urban-CFD standards it comes from: if the two means have opposite signs the
    ratio is negative, and if either is near zero it diverges. Rather than
    report a negative "square error", this returns ``nan`` whenever
    ``obs_mean * pred_mean <= 0``. Per-component NMSE on a signed velocity is
    therefore fragile by construction and the metrics doc's own advice applies
    -- score ``|U|``, TKE and the stresses' magnitudes, and read the components
    through :func:`hit_rate`.

    Args:
        pred: Predicted field.
        obs: Truth field, same shape. Non-finite pairs are dropped (fluid cells
            only).

    Returns:
        ``NMSE >= 0``, or ``nan`` when no finite pair survives or the mean
        product is not positive.

    Raises:
        ValueError: If the shapes differ.
    """
    predicted, observed = _fluid_pairs(pred, obs)
    if predicted.size == 0:
        return float("nan")
    denominator = float(np.mean(observed)) * float(np.mean(predicted))
    if not denominator > 0.0:
        return float("nan")
    return float(np.mean((observed - predicted) ** 2)) / denominator


def nmse_split(fb: float, nmse_total: float) -> tuple[float, float]:
    """Split NMSE into its systematic and unsystematic parts.

    ``NMSE_s = 4*FB**2 / (4 - FB**2)`` and ``NMSE_u = NMSE - NMSE_s``
    (``docs/plans/esmda_turbulence_evaluation.md`` section 4.1). The split is
    algebra, not an approximation: with ``d = obs_mean - pred_mean`` and
    ``s = obs_mean + pred_mean``, ``FB = 2d/s`` gives
    ``4FB**2/(4 - FB**2) = 4d**2/(s**2 - d**2) = d**2/(obs_mean * pred_mean)``,
    which is exactly the squared-bias term of the NMSE. So the remainder is
    ``var(o - p) / (obs_mean * pred_mean)``, always non-negative, and
    ``NMSE_s <= NMSE`` is an identity rather than a check -- provided both
    arguments were computed from the *same* pairs, which is the caller's job.

    Reading it is the conceptual payoff of the whole mean-field layer:
    assimilation should collapse ``NMSE_s`` (a bias is a parameter error), and
    ``NMSE_u`` is chaotic decorrelation, which is irreducible.

    Args:
        fb: Fractional bias from :func:`fractional_bias`.
        nmse_total: Total NMSE from :func:`nmse`, on the same pairs.

    Returns:
        ``(systematic, unsystematic)``. Both are ``nan`` if either input is
        non-finite, or if ``|fb| >= 2``: at ``|FB| = 2`` one of the means is
        zero (or they cancel) and the formula divides by zero, while beyond it
        the formula would return a *negative* "systematic" part and an
        unsystematic part larger than the total. That regime is unreachable for
        positive fields and reachable only for a signed component -- exactly
        the case :func:`nmse`'s docstring warns about -- so it is reported as
        undefined instead of as a number.
    """
    bias = float(fb)
    total = float(nmse_total)
    if not np.isfinite(bias) or not np.isfinite(total) or abs(bias) >= 2.0:
        return float("nan"), float("nan")
    systematic = 4.0 * bias**2 / (4.0 - bias**2)
    return systematic, total - systematic


def _fluid_pairs(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flattened prediction/truth pairs at cells where **both** are finite.

    The single implementation of the plan's "fluid cells only" rule, shared by
    all four scores so they are never computed over different point sets (which
    would silently break :func:`nmse_split`'s identity). Pairwise, not
    per-field: a cell the truth cannot resolve is not scored against a
    prediction that happens to be finite there.
    """
    predicted = np.asarray(pred, dtype=float)
    observed = np.asarray(obs, dtype=float)
    if predicted.shape != observed.shape:
        raise ValueError(
            f"pred and obs must have the same shape, got {predicted.shape} and "
            f"{observed.shape}"
        )
    keep = np.isfinite(predicted) & np.isfinite(observed)
    return predicted[keep], observed[keep]
