"""Flow statistics over ensemble state fields.

Time-mean velocities, resolved second moments, block bootstrap, and (phase 3)
Welch spectra with the log-spectral distance.

Everything streams: window state files are ``(ensemble, time, z, y, x)`` and
run to gigabytes, so nothing here may ``.load()`` one whole. The streaming
moment accumulator is the one class the library allows.

Populated in WP0.2 (move), extended in WP1.3 (the block bootstrap that gives
every window statistic its sampling floor), WP1.4 (colocation and the moment
accumulator behind the mean-field metrics) and phase 3.
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
    z_idx = evenly_spaced_levels(nz, n_z_slices)

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

    Identical replicates collapse to a true ``0.0`` rather than being left to
    ``np.std``, whose mean subtraction leaves ~1e-17 of float rounding behind.
    That residue is not a measurement -- when every replicate is the same number
    the bootstrap distribution is a point mass -- and it matters downstream,
    where an identifiability ratio filters on a positive floor and 1e-17 turns
    a clean ``null`` into ~1e17.

    No non-finite handling: the caller drops every row containing a non-finite
    sample before resampling, and validates ``n_resamples >= 2``, so the only
    way a ``nan`` arrives here is a statistic that produced one -- which should
    propagate rather than be filtered away.
    """
    spread = np.std(replicates, axis=0, ddof=1)
    spread[replicates.min(axis=0) == replicates.max(axis=0)] = 0.0
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
          blocks needs 21 samples and the CI smoke shape has four frames per
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


# ---------------------------------------------------------------------------
# Staggered-grid colocation (WP1.4)
# ---------------------------------------------------------------------------

# Per solver, the staggered dim each velocity component sits on and the
# cell-centre dim it must be moved to. Authority: ``ObservationOperator``'s
# ``dim_mapping`` in ``libs/data-assimilation`` -- the same solver names and the
# same axes, restated here because this library may not import that one.
#
# An empty tuple means "already co-located": pylbm writes one uniform grid, and
# ``conf/model/neural_surrogate.yaml`` sets ``solver_name: pylbm``, so the
# surrogate reaches this table under its spin-up backend's name and needs no
# entry of its own. ``palm``'s ``w`` is listed although pypalm's
# postprocess already interpolates it from ``zw_3d`` onto ``z``; a pair whose
# staggered dim is absent is skipped, so the entry is a no-op on a
# post-processed state and correct on a raw one.
_STAGGERED_TO_CENTRE = {
    "udales": {"u": (("xm", "xt"),), "v": (("ym", "yt"),), "w": (("zm", "zt"),)},
    "palm": {"u": (("xu", "x"),), "v": (("yv", "y"),), "w": (("zw", "z"),)},
    "pylbm": {"u": (), "v": (), "w": ()},
}

_VELOCITY_COMPONENTS = ("u", "v", "w")

# How far a cell centre may sit from the midpoint of its two bounding faces, as
# a fraction of the local face spacing, before colocation refuses to run.
# ``centre[i] == 0.5*(face[i] + face[i+1])`` is the *definition* of a cell
# centre, so in exact arithmetic this could be zero; it absorbs the rounding of
# coordinates stored as float32 (~1e-7 relative) and is still far below the
# smallest slicing mistake it must catch -- a stride-2 level selection displaces
# the midpoint by half a cell.
_FACE_CENTRE_TOLERANCE = 1e-3


def _faces_to_centres(da, face_dim, centre_dim, centres):
    """Linearly interpolate ``da`` from face samples onto cell centres.

    Each solver stores one face per cell (the lower one), so the face and centre
    axes have the same length and the last centre has no upper face; it is filled
    by linear extrapolation from the last two faces (weights 1.5 and -0.5) --
    the same ``fill_value="extrapolate"`` choice pypalm's own ``zw -> z``
    interpolation makes. Dropping the column instead would silently shorten the
    returned grid, which every consumer then has to re-derive.

    Works on ``.values``: adding two ``isel`` slices as DataArrays would make
    xarray align them on their (different) face coordinates and return nothing.
    """
    axis = da.get_axis_num(face_dim)
    values = np.asarray(da.values)
    n = values.shape[axis]

    def _take(index):
        return np.take(values, index, axis=axis)

    interior = 0.5 * (_take(np.arange(n - 1)) + _take(np.arange(1, n)))
    edge = 1.5 * _take([n - 1]) - 0.5 * _take([n - 2])
    moved = np.concatenate([interior, edge], axis=axis)

    dims = tuple(centre_dim if d == face_dim else d for d in da.dims)
    coords = {
        name: coord
        for name, coord in da.coords.items()
        if face_dim not in coord.dims and name != centre_dim
    }
    coords[centre_dim] = centres
    return xarray.DataArray(moved, dims=dims, coords=coords, name=da.name)


def _centre_values(ds, face_dim, centre_dim):
    """The centre coordinate for a face axis, checked to *be* the midpoints.

    The check is what makes colocation safe to call blindly: interpolating
    between two z-levels five cells apart is arithmetically fine and physically
    meaningless, so a state that was strided or level-selected before colocation
    is refused rather than silently averaged.
    """
    if face_dim not in ds.coords or centre_dim not in ds.coords:
        raise ValueError(
            f"cannot colocate off {face_dim!r}: the state carries "
            f"{'the centre' if face_dim in ds.coords else 'neither'} coordinate "
            f"but not both {face_dim!r} and {centre_dim!r}, so there is no way "
            "to know where the samples are"
        )
    faces = np.asarray(ds[face_dim].values, dtype=float)
    centres = np.asarray(ds[centre_dim].values, dtype=float)
    if faces.size < 2:
        raise ValueError(f"cannot colocate off a single {face_dim!r} level")
    if centres.size != faces.size:
        raise ValueError(
            f"{centre_dim!r} has {centres.size} levels but {face_dim!r} has "
            f"{faces.size}; one face per cell is the layout every backend writes"
        )
    spacing = np.diff(faces)
    midpoints = faces[:-1] + 0.5 * spacing
    offset = np.abs(centres[:-1] - midpoints)
    if np.any(offset > _FACE_CENTRE_TOLERANCE * np.abs(spacing)):
        raise ValueError(
            f"{centre_dim!r} centres are not the midpoints of the {face_dim!r} "
            "faces (worst offset "
            f"{float(np.max(offset / np.abs(spacing))):.3g} of a cell) -- the "
            "state was subset or strided along that axis before colocation; "
            "colocate first, then select"
        )
    return centres


def colocate_components(ds, solver_name):
    """``(u, v, w)`` interpolated onto cell centres, per backend.

    Reynolds stresses are *one-point* moments, so the components have to be
    evaluated at the same point before they are multiplied. On a staggered grid
    they are not: uDALES stores ``u`` on the x-faces, ``v`` on the y-faces and
    ``w`` on the z-faces; PALM stores ``u`` on ``xu`` and ``v`` on ``yv``.

    Why not combine by index, as :func:`streaming_state_rmse` does for ``|U|``:
    ``u[k,j,i]`` and ``w[k,j,i]`` sit half a cell apart in both x and z, so an
    ``<u'w'>`` formed from them is a *two-point* correlation at lag
    ``(dx/2, 0, dz/2)`` while the diagonal ``<u'u'>`` is a one-point moment --
    every entry of the tensor is evaluated somewhere else, and the resulting
    bias in the anisotropy ratio follows the staggering pattern rather than the
    flow, so no amount of ensemble averaging removes it and it differs between
    backends. For the mean field the displacement is a plain first-order error
    (``0.5*dx*dU/dx``), which is harmless in the ``|U|`` summary and wrong here.

    What colocation does *not* buy: linear interpolation is a low-pass filter,
    so it does not recover the amplitude of a grid-scale stress -- it makes the
    filter the *same* for every entry of the tensor, which is what the ratio
    needs. Neither method substitutes for resolving the fluctuation.

    Solid cells are not marked by any backend (pylbm writes near-zeros plus a
    ``blanking`` variable, pypalm replaces PALM's NaN with 0.0, uDALES ships no
    mask), so this function never decides which cells are solid -- masking is
    the caller's step, and interpolation is not a way to discover the mask
    either, since a zero-filled solid face pulls its neighbouring centre
    halfway to zero.

    Args:
        ds: A state Dataset (or one member of one) carrying ``u``, ``v``, ``w``
            and the coordinates of both the staggered and the centre axes. It
            must not have been subset along an axis that still needs colocating
            (see :func:`_centre_values`).
        solver_name: ``udales``, ``palm`` or ``pylbm`` -- the
            ``assim_solver_name`` recorded in a run's ``truth_access.yaml``.
            The surrogate is not listed because it records its spin-up
            backend's name (``pylbm``), which ``ObservationOperator`` also
            requires.

    Returns:
        Three DataArrays on the centre axes, carrying the centre coordinates, so
        they share one grid and can be multiplied cell by cell (and handed to
        ``.interp`` for a cross-grid truth comparison). Non-spatial dims
        (``time``, and ``ensemble`` when present) pass through untouched.

    Raises:
        ValueError: If the solver is unknown, a component is missing, or an axis
            cannot be colocated (see :func:`_centre_values`). The staggering
            table restates ``ObservationOperator.dim_mapping``; a test pins the
            two together, since silent drift would have the sensor series and
            the mean fields read one component off different axes in the same
            pass.
    """
    if solver_name not in _STAGGERED_TO_CENTRE:
        raise ValueError(
            f"unknown solver {solver_name!r}: colocation needs the staggering, "
            f"known are {sorted(_STAGGERED_TO_CENTRE)}"
        )
    mapping = _STAGGERED_TO_CENTRE[solver_name]
    missing = [name for name in _VELOCITY_COMPONENTS if name not in ds]
    if missing:
        raise ValueError(f"state is missing the velocity components {missing}")

    out = []
    for name in _VELOCITY_COMPONENTS:
        field = ds[name]
        for face_dim, centre_dim in mapping[name]:
            if face_dim not in field.dims:
                continue  # already co-located on this axis (e.g. pypalm's w)
            field = _faces_to_centres(
                field, face_dim, centre_dim, _centre_values(ds, face_dim, centre_dim)
            )
        out.append(field)
    return tuple(out)


def evenly_spaced_levels(n_levels, n_wanted):
    """Indices of ``n_wanted`` evenly spaced levels, endpoints included.

    Shared with :func:`_vel_field_4z` -- one implementation, so the accumulated
    slabs and the z-levels the ``|U|`` RMSE is streamed over cannot drift apart.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be positive, got {n_levels}")
    if n_wanted < 1:
        raise ValueError(f"n_wanted must be positive, got {n_wanted}")
    return np.unique(np.linspace(0, n_levels - 1, n_wanted).round().astype(int))


# ---------------------------------------------------------------------------
# Streaming moments (WP1.4)
# ---------------------------------------------------------------------------


class MomentAccumulator:
    """Chunk-wise ``n``, ``U_i`` and ``<u_i'u_j'>`` without holding the frames.

    The library's one class, and stateful for a reason: a run's mean fields are
    an average over every frame of every window, and the window state files
    total tens of GB, so the moments have to be folded in as the frames stream
    past. One instance covers one member (or the truth) over one region; feed it
    chunk after chunk with :meth:`update` and read :meth:`mean`, :meth:`tke` and
    :meth:`reynolds_stress` at the end.

    State is per cell and independent of how many frames pass through: one count
    plus ``C`` running means plus the ``C(C+1)/2`` co-moments of the ``i <= j``
    pairs -- 10 float64 arrays at the usual ``C = 3``, so 1.3 MB for four
    64x64 slabs and 21 MB for four 256x256 ones. It is the *ensemble* total that
    bites (one accumulator per member), which is the caller's problem to bound.

    **NaN policy: per-cell counts, casewise deletion.** ``n`` is an array over
    cells, not a scalar, and a frame contributes to a cell only if every
    component is finite there. A scalar count would have to either drop a whole
    frame because one cell is masked or count a masked cell as a sample; and a
    truth field interpolated onto the assimilation grid comes back NaN at the
    cells the interpolation cannot reach, so per-cell counts are the only
    version that gives the interior its correct sample size. Casewise (rather
    than pairwise) deletion keeps ``R_ij`` a covariance: counting each pair over
    its own available frames can make ``R`` indefinite and ``tke`` negative at a
    cell that is masked intermittently in one component only.

    **Numerics: the chunk-wise (Chan) combine, not ``sum uu - (sum u)^2/n``.**
    Each chunk's co-moment is formed about that chunk's own mean and merged with
    a ``delta_i*delta_j*n*m/(n+m)`` correction. The textbook form subtracts two
    nearly equal large numbers and this is exactly its bad regime -- urban flow
    at a 5 m/s mean carries ~0.2 m/s fluctuations, so the mean square is ~600x
    the variance being recovered. Relative error of the ``ddof=1`` variance
    against an exact ``longdouble`` two-pass over the stored samples, 360 frames
    fed in 10 chunks of 36 (measured, ``float64`` throughout):

    ==================  ========  ==========
    mean / fluctuation  naive     this class
    ==================  ========  ==========
    5 / 0.2                8e-14      2e-16
    5 / 0.002              2e-09      4e-15
    1e4 / 1e-3             4e-02      7e-11
    ==================  ========  ==========

    The last row is a 4% error on data that is not pathological -- just a field
    with a large offset relative to its fluctuation -- which is what rules the
    naive form out.
    """

    def __init__(self):
        # Everything is allocated on the first update: only the data knows the
        # cell shape, and one class then serves slabs and station columns
        # without the caller restating shapes the arrays already carry.
        self._count = None
        self._mean = None
        self._comoment = None
        self._pairs = ()
        self._n_components = 0
        self._cell_shape = ()

    def update(self, *components):
        """Fold one time chunk of co-located components into the accumulators.

        Args:
            *components: One array per component, **time first**, all of the
                same shape ``(n_time, *cell_shape)``. ``np.asarray`` is applied,
                so DataArrays and float32 fields are accepted; accumulation is
                always in float64. A single frame must be passed with an
                explicit length-1 time axis -- the leading axis is always read
                as time and a missing one cannot be detected.

        Raises:
            ValueError: If no components are given, if their shapes differ, if
                an array has no time axis, or if the component count or cell
                shape differs from the first call -- which is what catches a
                caller that changed its z-level selection between chunks.
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

        # Casewise validity: one mask for every component, so the mean and every
        # co-moment are taken over the same frames at every cell.
        valid = np.isfinite(chunk[0])
        for component in chunk[1:]:
            valid &= np.isfinite(component)
        n_new = valid.sum(axis=0)
        if not bool(np.any(n_new)):
            return

        # Deviations from the *running* mean rather than from zero: on every
        # chunk but the first this is already the small quantity, which is what
        # keeps the chunk mean itself accurate for large-offset fields.
        deviation = [
            np.where(valid, component - mean[i], 0.0)
            for i, component in enumerate(chunk)
        ]
        safe_new = np.where(n_new > 0, n_new, 1)
        delta = np.stack([d.sum(axis=0) / safe_new for d in deviation])

        # Each chunk's co-moment is formed about the chunk's own mean (a
        # two-pass computation *within* the chunk) and the shift between that
        # mean and the running one is added back exactly.
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

    def count(self):
        """Per-cell frames in which **all** components were finite (a copy)."""
        return self._state()[0].copy()

    def mean(self):
        """Per-cell time means ``U_i``.

        Returns:
            ``(n_components, *cell_shape)``; ``nan`` at cells with no finite
            frames (a masked solid cell, or a cell outside an interpolated truth
            field).
        """
        count, mean, _ = self._state()
        return np.where(count > 0, mean, np.nan)

    def reynolds_stress(self, ddof=1):
        """Per-cell resolved stresses ``R_ij = <u_i'u_j'>``.

        **Resolved only** -- these are moments of the fields the solver wrote;
        the subgrid contribution is not in them and is not negligible inside a
        canopy (metrics doc section 4.1 asks for that to be stated wherever the
        stresses are reported).

        Args:
            ddof: Denominator correction. ``1`` (the default) is the unbiased
                sample covariance, matching the ``ddof=1`` variances WP1.3
                reports for the same quantity at sensors; ``0`` gives the plain
                time average ``<u_i u_j> - U_i U_j``.

        Returns:
            ``(n_components, n_components, *cell_shape)``, symmetric by
            construction (the ``i > j`` entries are the accumulated ``i <= j``
            ones, not a second estimate); ``nan`` where ``n <= ddof``.
        """
        scale = self._moment_scale(ddof)
        comoment = self._state()[2]
        n = self._n_components
        stress = np.empty((n, n) + self._cell_shape)
        for pair, (i, j) in enumerate(self._pairs):
            value = comoment[pair] * scale
            stress[i, j] = value
            stress[j, i] = value
        return stress

    def tke(self, ddof=1):
        """Per-cell resolved TKE ``k = 0.5*R_ii``.

        Formed from the diagonal co-moments directly rather than by tracing
        :meth:`reynolds_stress`, so it costs one array instead of ``C**2``. At
        the default ``ddof=1`` this is exactly ``0.5*sum_i var_ddof1(u_i)``, the
        estimator WP1.3 reports at sensors.
        """
        scale = self._moment_scale(ddof)
        comoment = self._state()[2]
        diagonal = np.zeros(self._cell_shape, dtype=float)
        for pair, (i, j) in enumerate(self._pairs):
            if i == j:
                diagonal += comoment[pair]
        return np.asarray(0.5 * diagonal * scale)

    def _initialise(self, n_components, cell_shape):
        """Allocate once the first chunk has fixed the shapes."""
        self._n_components = n_components
        self._cell_shape = cell_shape
        self._pairs = tuple(
            (i, j) for i in range(n_components) for j in range(i, n_components)
        )
        self._count = np.zeros(cell_shape, dtype=np.int64)
        self._mean = np.zeros((n_components, *cell_shape), dtype=float)
        self._comoment = np.zeros((len(self._pairs), *cell_shape), dtype=float)

    def _state(self):
        """``(count, mean, comoment)``, or a readable error if nothing was fed."""
        if self._count is None or self._mean is None or self._comoment is None:
            raise ValueError(
                "no frames accumulated yet: MomentAccumulator takes its shapes "
                "from the first update, so there is nothing to report"
            )
        return self._count, self._mean, self._comoment

    def _moment_scale(self, ddof):
        """``1/(n - ddof)`` per cell, ``nan`` where that is not a sample size."""
        if ddof < 0:
            raise ValueError(f"ddof must be non-negative, got {ddof}")
        count = self._state()[0]
        denominator = count - ddof
        usable = denominator > 0
        return np.where(usable, 1.0 / np.where(usable, denominator, 1), np.nan)
