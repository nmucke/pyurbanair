"""Flow statistics over ensemble state fields.

Time-mean velocities, resolved second moments, block bootstrap, and (phase 3)
Welch spectra with the log-spectral distance.

Everything streams: window state files are ``(ensemble, time, z, y, x)`` and
run to gigabytes, so nothing here may ``.load()`` one whole. The streaming
moment accumulator is the one class the library allows.

Populated in WP0.2 (move), extended in WP1.3 (the block bootstrap that gives
every window statistic its sampling floor), WP1.4 and phase 3.
"""

# mypy: ignore-errors
# Moved in WP0.2 from ``scripts/esmda/_esmda_common.py``, which carries a
# file-level mypy waiver; kept here rather than annotated during a pure
# refactor.

from __future__ import annotations

import numpy as np
import xarray

# z-like dims across the backends: pylbm / the surrogate are cell-centred
# (``z``), uDALES and PALM stagger u/v on ``zt`` and w on ``zm``.
_Z_DIMS = ("z", "zm", "zt")

# The resample draw must not depend on OS entropy: re-running the metric stage
# on the same run directory has to reproduce the same ``run_summary.yaml``.
_DEFAULT_BOOTSTRAP_SEED = 0

# Ceiling on the temporary the bootstrap builds. The resampled array is
# ``(n_rows, n_replicates_in_chunk, n_time)`` float64, which at M=128 / 4
# quantities / 20 sensors would be ~1.6 GB unchunked. Splitting the replicate
# axis bounds it without moving a single number -- the statistic reduces along
# the time axis only, so a row's replicate does not depend on which chunk it
# was computed in.
_BOOTSTRAP_CHUNK_BYTES = 32 << 20


def select_z_plane(ds, z_level):
    """Select a single z-layer (kept as a size-1 dim) on every z-like dim present.

    udales staggers the components on different vertical axes (u/v on ``zt``,
    w on ``zm``); selecting ``z_level`` on each keeps the velocity-magnitude
    computation aligned while loading only one horizontal plane per variable.
    """
    sel = {d: slice(z_level, z_level + 1) for d in _Z_DIMS if d in ds.dims}
    return ds.isel(sel) if sel else ds


def _horizontal_coord(ds, names):
    for n in names:
        if n in ds.coords:
            return np.asarray(ds[n].values, dtype=float)
    return None


def _vel_field_4z(state, n_time, n_z_slices=4):
    """Velocity-magnitude field on ``n_z_slices`` evenly-spaced z-levels.

    Returns a ``(time, zlev, y, x)`` DataArray on nominal cell-centre coords.
    Only the selected z-slices (across all time) are read from disk, bounding
    memory to a small fraction of the full 3-D field. The components are combined
    by index (matching ``get_velocity_magnitude_field``).
    """
    zdim = next((d for d in _Z_DIMS if d in state.dims), None)
    nz = state.sizes[zdim] if zdim is not None else 1
    z_idx = np.unique(np.linspace(0, nz - 1, n_z_slices).round().astype(int))

    s = state.isel(time=slice(0, n_time))

    def _sel_var(name):
        da = s[name]
        for d in _Z_DIMS:
            if d in da.dims:
                da = da.isel({d: z_idx})
                break
        return np.asarray(da.values)

    vel = np.sqrt(_sel_var("u") ** 2 + _sel_var("v") ** 2 + _sel_var("w") ** 2)

    coords = {}
    y = _horizontal_coord(state, ("yt", "y"))
    x = _horizontal_coord(state, ("xt", "x"))
    if y is not None and y.size == vel.shape[2]:
        coords["y"] = y
    if x is not None and x.size == vel.shape[3]:
        coords["x"] = x
    return xarray.DataArray(vel, dims=("time", "zlev", "y", "x"), coords=coords)


def streaming_state_rmse(true_state, esmda_state, n_z_slices=4):
    """Per-timestep RMSE of |U| between truth and the ensemble-mean state.

    Streams over ``n_z_slices`` evenly-spaced z-levels and all time steps rather
    than materialising the full 4-D velocity field. When the truth and
    assimilation grids differ, the truth planes are interpolated onto the
    assimilation grid before differencing.
    """
    true_s = (
        true_state.mean(dim="ensemble") if "ensemble" in true_state.dims else true_state
    )
    esmda_s = (
        esmda_state.mean(dim="ensemble")
        if "ensemble" in esmda_state.dims
        else esmda_state
    )

    n_time = min(true_s.sizes["time"], esmda_s.sizes["time"])

    true_vel = _vel_field_4z(true_s, n_time, n_z_slices)
    esmda_vel = _vel_field_4z(esmda_s, n_time, n_z_slices)

    have_coords = all(
        "y" in da.coords and "x" in da.coords for da in (true_vel, esmda_vel)
    )
    grids_match = (
        have_coords
        and true_vel.sizes.get("y") == esmda_vel.sizes.get("y")
        and true_vel.sizes.get("x") == esmda_vel.sizes.get("x")
        and np.allclose(true_vel["y"], esmda_vel["y"])
        and np.allclose(true_vel["x"], esmda_vel["x"])
    )
    if not grids_match and have_coords:
        # Coordinates don't line up -> interpolate the truth onto the assim grid.
        true_vel = true_vel.interp(y=esmda_vel["y"], x=esmda_vel["x"])

    nz_common = min(true_vel.sizes["zlev"], esmda_vel.sizes["zlev"])
    diff = np.asarray(true_vel.isel(zlev=slice(0, nz_common)).values) - np.asarray(
        esmda_vel.isel(zlev=slice(0, nz_common)).values
    )
    return np.sqrt(np.nanmean(diff**2, axis=tuple(range(1, diff.ndim))))


# ---------------------------------------------------------------------------
# Moving-block bootstrap: the sampling floor of a window statistic
# ---------------------------------------------------------------------------


def _block_resample_indices(n_samples, block_len, n_resamples, generator):
    """Moving-block resample index matrix, shape ``(n_resamples, n_samples)``.

    Each row is ``ceil(n / L)`` block starts drawn with replacement from the
    ``n - L + 1`` legal positions, expanded to their offsets, concatenated and
    truncated back to ``n``. The truncation matters: the blocks overshoot
    whenever ``L`` does not divide ``n``, and a statistic that depends on the
    sample count (a ``ddof=1`` variance) has to be evaluated at the original
    length in every replicate to stay comparable.
    """
    n_starts = n_samples - block_len + 1
    n_draw = int(np.ceil(n_samples / block_len))
    starts = generator.integers(0, n_starts, size=(n_resamples, n_draw))
    index = starts[:, :, None] + np.arange(block_len)[None, None, :]
    return index.reshape(n_resamples, -1)[:, :n_samples]


def _replicate_spread(replicates):
    """``ddof=1`` spread down axis 0 of a ``(n_resamples, n_rows)`` array.

    Identical replicates collapse to a true ``0.0`` rather than being handed to
    ``np.std``, whose mean subtraction leaves ~1e-17 of float rounding behind.
    That residue is not a measurement -- when every replicate is the same number
    the bootstrap distribution is a point mass -- and it matters downstream,
    where an identifiability ratio filters on a positive floor and 1e-17 turns
    a clean ``null`` into ~1e17.
    """
    n_rows = replicates.shape[1]
    spread = np.full(n_rows, np.nan)
    if replicates.shape[0] < 2:
        return spread
    finite = np.isfinite(replicates)
    usable = finite.sum(axis=0) >= 2
    if not usable.any():
        return spread
    columns = replicates[:, usable]
    spread[usable] = np.nanstd(columns, axis=0, ddof=1)
    degenerate = np.nanmin(columns, axis=0) == np.nanmax(columns, axis=0)
    spread[np.flatnonzero(usable)[degenerate]] = 0.0
    return spread


def block_bootstrap_std(
    series, statistic=np.mean, n_blocks=20, n_resamples=200, rng=None
):
    """Moving-block bootstrap standard error of a statistic, time axis last.

    How much of a window statistic (a time-mean, a variance) is just the finite
    length of the window? The iid answer ``std/sqrt(n)`` is wrong for a probe
    series by a large factor, because turbulent samples are not independent --
    a 400-sample AR(1) series at ``phi = 0.9`` has roughly the sampling spread
    of 20 independent ones. The moving-block bootstrap keeps the within-block
    correlation by resampling *contiguous stretches* rather than points: block
    length ``L = ceil(n / n_blocks)``, then ``ceil(n / L)`` blocks drawn with
    replacement from all ``n - L + 1`` start positions, concatenated and
    truncated back to ``n``, repeated ``n_resamples`` times. The answer is the
    ``ddof=1`` spread of the statistic over those replicates.

    Measured on AR(1) series of unit marginal variance with ``statistic =
    np.mean``, median over 200 series; "true" is the spread of the sample mean
    across those same series, i.e. the quantity being estimated:

    ====  ====  ==  =======  ===========  =====
    n     phi   L   this fn  std/sqrt(n)  true
    ====  ====  ==  =======  ===========  =====
    400   0.0   20  0.049    0.050        0.055
    400   0.7   20  0.106    0.050        0.131
    400   0.9   20  0.154    0.048        0.238
    36    0.9   2   0.168    0.128        0.596
    ====  ====  ==  =======  ===========  =====

    Two readings. On independent data it lands on the iid formula, so blocking
    costs nothing when there is no correlation to preserve. On correlated data
    it is 2-3x the iid formula and still **below** the truth, increasingly so as
    ``L`` falls under the correlation time. So it is a *lower bound* on the
    sampling spread of a correlated series -- the safe direction for its WP1.3
    consumer, where it is a denominator: it can only make an identifiability
    ratio look better than it is, never worse.

    Every row shares one time axis and therefore one resample index matrix,
    which is what makes the vectorized form possible: the draw happens once and
    the whole resampled block is reduced in a single ``statistic(..., axis=-1)``
    call (chunked over replicates to bound the temporary, which changes no
    value -- the reduction is along time only).

    Args:
        series: ``(..., n_time)``, **time last**. Leading dims are independent
            series; a 1-D input is one series.
        statistic: An *axis-taking* reducer, called as ``statistic(x, axis=-1)``
            on a 3-D array and returning ``x``'s leading shape. ``np.mean`` and
            ``np.var`` qualify as they are; a closure must take and forward
            ``axis`` (``lambda x, axis: np.var(x, axis=axis, ddof=1)``) rather
            than assuming a flat series.
        n_blocks: Target block count; the block length is derived from it, so
            one call site works on windows of different lengths.
        n_resamples: Number of bootstrap replicates.
        rng: A ``numpy`` generator or a seed for one. ``None`` means the fixed
            default seed -- never OS entropy, so the default path reproduces.

    Returns:
        ``series.shape[:-1]`` bootstrap standard errors (a 0-d array for a 1-D
        input), ``nan`` where undefined:

        * every row, when the shared time axis has fewer than four samples or
          gives ``L < 2``. **This is routine, not exotic** -- the default 20
          blocks needs 21 samples and the CI smoke shape has three frames per
          window -- so callers must handle it.
        * a row containing any non-finite sample. Dropping the gaps per row
          would give that row a different finite count, hence a different block
          length and index matrix, which is exactly the sharing that makes this
          vectorized; reporting ``nan`` is honest about an input this path does
          not serve.

        ``n_blocks=1`` returns exactly ``0.0`` for finite rows, a *measured*
        zero: one block spans the series, there is one legal start, so every
        replicate is the original series.

    Raises:
        ValueError: If ``series`` has no dimensions, ``n_blocks < 1``,
            ``n_resamples < 2``, or ``statistic`` breaks the ``axis`` contract.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples}")

    values = np.asarray(series, dtype=float)
    if values.ndim < 1:
        raise ValueError("series must have at least a time dimension")
    lead_shape = values.shape[:-1]
    n_time = values.shape[-1]
    flat = values.reshape(-1, n_time)
    out = np.full(flat.shape[0], np.nan)

    block_len = int(np.ceil(n_time / n_blocks)) if n_time else 0
    if flat.shape[0] == 0 or n_time < 4 or block_len < 2:
        return out.reshape(lead_shape)

    finite_rows = np.isfinite(flat).all(axis=1)
    if not finite_rows.any():
        return out.reshape(lead_shape)
    rows = flat[finite_rows]

    generator = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(_DEFAULT_BOOTSTRAP_SEED if rng is None else rng)
    )
    index = _block_resample_indices(n_time, block_len, n_resamples, generator)

    replicates = np.empty((n_resamples, rows.shape[0]))
    chunk = max(1, int(_BOOTSTRAP_CHUNK_BYTES // (8 * rows.shape[0] * n_time)))
    for start in range(0, n_resamples, chunk):
        block = rows[:, index[start : start + chunk]]  # (n_rows, chunk, n_time)
        reduced = np.asarray(statistic(block, axis=-1))
        if reduced.shape != block.shape[:-1]:
            raise ValueError(
                "statistic must reduce only the axis it is given: expected shape "
                f"{block.shape[:-1]} from statistic(x, axis=-1), got {reduced.shape}"
            )
        replicates[start : start + chunk] = reduced.T

    out[finite_rows] = _replicate_spread(replicates)
    return out.reshape(lead_shape)
