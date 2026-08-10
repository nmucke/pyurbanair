"""Reductions of pre-extracted sensor and probe series to window statistics.

Consumes ``(ensemble, time, sensor)`` arrays a script has already pulled out of
the state files. Extraction itself stays in ``scripts/esmda/_esmda_common.py``:
it needs ``data_assimilation``'s observation operator (jax) and the run-dir
layout, both forbidden here.

WP1.3's premise (metrics doc §4.2): the verification object is the **window
statistic**, not the time series. Instantaneous turbulence is chaotic -- two
members with identical parameters decorrelate within an eddy turnover -- so a
pointwise time-series error mostly measures phase, which no parameter estimate
can control. What the estimated parameters do act on is the *statistics*, so
those are the identifiable quantities and the thing worth scoring.

Populated in WP0.2 (move), extended in WP1.3.
"""

# WP0.2 moved this module out of ``scripts/esmda/_esmda_common.py`` under a
# file-level ``# mypy: ignore-errors``. The waiver is gone: every function here
# is annotated and the module passes the repo's strict config on its own.

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import xarray as xr
from evaluation.turbulence import block_bootstrap_std

logger = logging.getLogger(__name__)

# The scored statistics, in the order they appear in ``run_summary.yaml``.
# ``magnitude`` is |U|, which is what a cup anemometer reports and what the
# sensor figures plot; the components carry the direction information |U|
# discards.
QUANTITIES = ("u", "v", "w", "magnitude")

# Slack, in units of windows, on the window-boundary test. See :func:`window_masks`
# for why a bare ``floor`` puts a boundary frame in the wrong window.
_BOUNDARY_TOLERANCE = 1e-9


def sensor_magnitude(components: xr.DataArray) -> xr.DataArray:
    """Velocity magnitude |U| from a ``(component, ...)`` sensor series."""
    return np.sqrt((components**2).sum("component"))


def _quantity_series(series: xr.DataArray) -> xr.DataArray:
    """``(component, ...)`` series -> ``(quantity, ...)`` with |U| appended.

    One array over :data:`QUANTITIES` so every downstream reduction is a single
    vectorized call rather than four near-copies.
    """
    magnitude = sensor_magnitude(series).expand_dims(quantity=["magnitude"])
    components = series.rename(component="quantity")
    return xr.concat([components, magnitude], dim="quantity").sel(
        quantity=list(QUANTITIES)
    )


def window_masks(
    times: np.ndarray, sim_time: float, num_windows: int
) -> list[np.ndarray]:
    """Boolean time masks, one per assimilation window.

    Binning by the *time coordinate* rather than by frame count is what lets the
    truth and the ensemble be reduced by the same function: both series carry a
    global time axis on which window ``w`` starts at ``w*sim_time`` (the
    extraction rebases them), but they need not have the same output cadence --
    a truth pre-simulated at one interval against an assimilation writing at
    another is the normal case, not the exception.

    A frame exactly on a boundary belongs to the window it opens, matching the
    run's own ``[w*sim_time, (w+1)*sim_time)`` convention; anything past the
    last boundary (float drift on the final frame) falls in the last window.

    The ``+_BOUNDARY_TOLERANCE`` is what makes the first half of that true. The
    extraction rebases window ``w`` to start at exactly ``w*sim_time``, but
    ``(w*sim_time)/sim_time`` is not exactly ``w`` in IEEE double for most
    values of ``sim_time`` -- at ``sim_time = 10.76`` (200 frames at pylbm's
    default 0.0538 s cadence) windows 7 and 14 come out a ULP short, and a bare
    ``floor`` scores their opening frame into the *previous* window. Roughly a
    fifth of two-decimal ``sim_time`` values misbin at least one boundary in
    the first ten windows, and the truth's global axis drifts independently of
    the ensemble's, so the two need not even misbin the same frame. The
    tolerance is in units of windows: 1e-9 of a window is a nanosecond at
    ``sim_time = 1``, orders of magnitude below any output cadence.
    """
    t = np.asarray(times, dtype=float)
    if not sim_time > 0:
        raise ValueError(f"sim_time must be positive, got {sim_time}")
    if num_windows < 1:
        raise ValueError(f"num_windows must be at least 1, got {num_windows}")
    index = np.floor(t / sim_time + _BOUNDARY_TOLERANCE).astype(int)
    index = np.clip(index, 0, num_windows - 1)
    return [index == w for w in range(num_windows)]


def _windowed(
    series: xr.DataArray,
    sim_time: float,
    num_windows: int,
    # ``...`` rather than ``[xr.DataArray]``: it is always called with the one
    # window slice, but :func:`window_sampling_std` passes a lambda that binds
    # its reducer through a default argument, which mypy will not infer against
    # a fixed-arity signature. The return type is the part worth pinning.
    reduce_fn: Callable[..., xr.DataArray],
    label: str = "",
) -> xr.DataArray:
    """Apply ``reduce_fn(window_slice)`` per window and stack on a ``window`` dim."""
    quantities = _quantity_series(series)
    masks = window_masks(quantities["time"].values, sim_time, num_windows)
    reduced = []
    for w, mask in enumerate(masks):
        if not mask.any():
            # Invariant 3: a window whose frames are missing (a killed job, a
            # truncated file) costs its own entry, not the whole block.
            logger.warning(
                "%sno sensor frames fall in window %d -- its statistics are null",
                f"{label}: " if label else "",
                w,
            )
            # ``mean`` over an empty selection, not ``isel(time=0)``: it yields
            # the right dims and coords even when the series itself is empty.
            empty = quantities.isel(time=slice(0, 0)).mean("time")
            reduced.append(xr.full_like(empty, np.nan, dtype=float))
            continue
        reduced.append(reduce_fn(quantities.isel(time=mask)))
    return xr.concat(reduced, dim="window").assign_coords(window=np.arange(num_windows))


def window_statistics(
    series: xr.DataArray, sim_time: float, num_windows: int, label: str = ""
) -> dict[str, xr.DataArray]:
    """Per-window mean and variance of each quantity at each sensor.

    Args:
        series: ``(component, [ensemble,] time, sensor)`` with a global ``time``
            coordinate, as produced by the extraction helpers.
        sim_time: Simulated seconds per assimilation window.
        num_windows: Number of windows.
        label: Prefix for the log lines (e.g. the sensor-set name), so a run
            with several sensor sets does not emit indistinguishable warnings.

    Returns:
        ``{"mean": DataArray, "variance": DataArray}``, each
        ``(window, quantity, [ensemble,] sensor)`` -- ``window`` leads because
        it is the concat dim. Index by name (``.sel(quantity=...)``), never
        positionally.

    The variance is ``ddof=1``: it is an estimate of the flow's variance from a
    finite window, not the window's own second moment, and the truth and the
    members are reduced identically so the choice cannot bias the comparison.
    A single-frame window gives ``nan``, which is correct rather than zero.
    """
    return {
        "mean": _windowed(
            series, sim_time, num_windows, lambda w: w.mean("time"), label
        ),
        "variance": _windowed(
            series, sim_time, num_windows, lambda w: w.var("time", ddof=1), label
        ),
    }


def window_sampling_std(
    series: xr.DataArray,
    sim_time: float,
    num_windows: int,
    n_blocks: int = 20,
    n_resamples: int = 200,
    rng: np.random.Generator | None = None,
    label: str = "",
) -> dict[str, xr.DataArray]:
    """Block-bootstrap sampling std of each :func:`window_statistics` entry.

    The noise floor the statistics are read against: how much a member's window
    mean or variance would move if the same member were simulated over a
    different stretch of the same flow. WP1.3's identifiability guard is the
    ensemble's spread in a statistic divided by this -- a statistic whose
    across-member spread does not clear its own sampling noise is not
    identifiable at this window length, however good its CRPS looks.

    Same shapes as :func:`window_statistics`. ``nan`` wherever the bootstrap is
    undefined (see :func:`~evaluation.turbulence.block_bootstrap_std`); short
    windows -- including the CI smoke shape -- are the common case, so callers
    must treat an all-``nan`` result as "no floor measured", not as a failure.
    """
    reducers = {
        "mean": lambda x, axis: np.mean(x, axis=axis),
        "variance": lambda x, axis: np.var(x, axis=axis, ddof=1),
    }
    return {
        name: _windowed(
            series,
            sim_time,
            num_windows,
            lambda w, fn=fn: _bootstrap_window(w, fn, n_blocks, n_resamples, rng),
            label,
        )
        for name, fn in reducers.items()
    }


def _bootstrap_window(
    window: xr.DataArray,
    statistic: Callable[..., np.ndarray],
    n_blocks: int,
    n_resamples: int,
    rng: np.random.Generator | None,
) -> xr.DataArray:
    """One window's bootstrap std, shaped exactly like that window's ``mean``."""
    # ``block_bootstrap_std`` wants time last; the extracted series carries it in
    # the middle (``..., time, sensor``). Dropping ``time`` in place leaves the
    # dim order xarray's own ``mean("time")`` would produce, so the statistics
    # and their sampling floor stack identically.
    reduced_dims = tuple(d for d in window.dims if d != "time")
    ordered = window.transpose(*reduced_dims, "time")
    std = block_bootstrap_std(
        np.asarray(ordered.values, dtype=float),
        statistic=statistic,
        n_blocks=n_blocks,
        n_resamples=n_resamples,
        rng=rng,
    )
    coords = {d: ordered.coords[d] for d in reduced_dims if d in ordered.coords}
    return xr.DataArray(std, dims=reduced_dims, coords=coords)
