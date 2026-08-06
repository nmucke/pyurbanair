"""Flow statistics over ensemble state fields.

Time-mean velocities, resolved second moments, block bootstrap, and (phase 3)
Welch spectra with the log-spectral distance.

Everything streams: window state files are ``(ensemble, time, z, y, x)`` and
run to gigabytes, so nothing here may ``.load()`` one whole. The streaming
moment accumulator is the one class the library allows.

Populated in WP0.2 (move), extended in WP1.3 (the block bootstrap that gives
every window statistic its sampling floor), WP1.4 (colocation and the moment
accumulator behind the mean-field metrics) and phase 3 (Welch spectra at the
probes and the log-spectral distance).
"""

# mypy: ignore-errors
# Moved in WP0.2 from ``scripts/esmda/_esmda_common.py``, which carries a
# file-level mypy waiver; kept here rather than annotated during a pure
# refactor. The phase-3 spectrum section below is annotated.

from __future__ import annotations

import logging
import warnings

import numpy as np
import xarray
from scipy import signal

logger = logging.getLogger(__name__)

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


def extrapolated_centre_dims(ds, solver_name):
    """Centre dims whose **last** index colocation fills by extrapolation.

    Every axis :func:`colocate_components` moves gets its final centre from
    ``1.5*face[-1] - 0.5*face[-2]`` rather than from two bounding faces, because
    the backends store one face per cell (see :func:`_faces_to_centres`). That
    column's second moments are inflated -- ~20 % for a well-resolved field, up
    to 5x for face-to-face white noise -- and it is **not** only the vertical:
    uDALES moves x, y *and* z, PALM moves x and y. An evenly spaced level or
    cell selection always includes the last index, so the affected cells are
    scored unless a consumer knows to exclude them.

    Returns the centre dim names, so a caller can record which cells of a
    reduced field carry the artefact. Empty for an unstaggered grid (pylbm) and
    for an axis the state does not carry (pypalm pre-interpolates ``w`` off
    ``zw``), which is exactly when no extrapolation happens.

    Raises:
        ValueError: If the solver is unknown -- same table as
            :func:`colocate_components`, so the two cannot disagree.
    """
    if solver_name not in _STAGGERED_TO_CENTRE:
        raise ValueError(
            f"unknown solver {solver_name!r}: colocation needs the staggering, "
            f"known are {sorted(_STAGGERED_TO_CENTRE)}"
        )
    moved = []
    for pairs in _STAGGERED_TO_CENTRE[solver_name].values():
        for face_dim, centre_dim in pairs:
            if face_dim in ds.dims and centre_dim not in moved:
                moved.append(centre_dim)
    return tuple(moved)


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


# ---------------------------------------------------------------------------
# Frequency spectra at the probes (phase 3, metrics doc section 4.3)
#
# The one structural check the suite keeps. An over-smoothed or
# surrogate-collapsed flow can pass every mean-field metric -- the right
# time-mean, and a variance a parameter can be tuned to reproduce -- while
# carrying that variance at the wrong frequencies; nothing else here looks at
# *where in frequency* the energy sits.
#
# Frequency spectra at probe points, never wavenumber spectra: converting one to
# the other needs Taylor's frozen-turbulence hypothesis, and inside a canopy the
# convection velocity is comparable to the fluctuations, so it does not hold.
# ---------------------------------------------------------------------------

# Welch segments per record: ``nperseg = n_truth // SPECTRUM_SEGMENTS``, which at
# 50 % overlap averages ``2*S - 1 = 15`` segments. That is the trade this
# diagnostic wants -- the estimator's own scatter has to sit well below the
# truth-vs-model differences it is meant to resolve -- and the alternative (fewer,
# longer segments) buys low-frequency resolution that the cutoff below discards
# anyway.
SPECTRUM_SEGMENTS = 8

# Comparisons stop at this fraction of the Nyquist frequency. The top of a
# sampled band is not data: a solver damps its own smallest resolved scales, and
# whatever the flow carried above Nyquist has aliased into the last octave. Both
# effects are discretisation-specific, so scoring there would compare numerics
# rather than flows.
SPECTRUM_CUTOFF_FRACTION = 0.25

# Fewest frequency bins below the cutoff for a comparison to mean anything. Not a
# resolution claim -- it is the floor under which the LSD is one or two bins of a
# noisy periodogram. Note what it implies about record length: the band holds
# about ``nperseg/8`` bins and ``nperseg`` is ``n/8``, so four bins needs ~260
# samples, which is the metrics doc's "≳10^3 samples per sensor" stated as a
# minimum rather than as a comfort level.
_MIN_BAND_BINS = 4

_PROBE_COMPONENTS = ("u", "v", "w")


def minimum_spectral_samples() -> int:
    """Samples per sensor a probe record needs before a spectrum can be scored.

    Worth having as a function rather than a comment because the caller that can
    act on it -- whatever writes the probe records -- pays for a full solver
    re-run first, and the check costs nothing. The band holds
    ``n * SPECTRUM_CUTOFF_FRACTION / (2 * SPECTRUM_SEGMENTS)`` bins: the bin width
    is ``fs/nperseg`` with ``nperseg = n // SPECTRUM_SEGMENTS`` and the cutoff is
    ``SPECTRUM_CUTOFF_FRACTION * fs/2``, so ``fs`` cancels and **the cadence drops
    out entirely** -- only the sample count matters.

    Two details make the naive inversion (``2*B*S/c``) wrong by an octave-edge
    bin, so they are carried explicitly: the DC bin is dropped, and the band is
    ``f < cutoff`` strictly. Scored bins are the integers ``k >= 1`` with
    ``k/nperseg < c/2``, i.e. ``ceil(c*nperseg/2) - 1`` of them, so the floor is
    the smallest ``nperseg`` with ``c*nperseg/2 > B``.
    """
    nperseg = int(np.floor(2 * _MIN_BAND_BINS / SPECTRUM_CUTOFF_FRACTION)) + 1
    return nperseg * SPECTRUM_SEGMENTS


# The non-dimension coordinates (xarray's term) a probe file carries on ``sensor``.
# Two probe files are matched *on these*, not on sensor order: comparing the
# truth at one point against a member at another renders exactly like a correct
# figure, and the writer's ordering is not something this side can verify.
_SENSOR_COORDS = ("sensor_x", "sensor_y", "sensor_z")

# Decimals the sensor coordinates are rounded to before they are matched. The
# same probe points travel through a YAML config and two netCDF files, so they
# can differ in the last float bits; a micrometre is far below any sensor spacing
# a case defines.
_SENSOR_MATCH_DECIMALS = 6

# How far two sources' bin SPACINGS may differ before the comparison is refused.
#
# The grids are ``k/segment_seconds``, so sharing the segment duration makes the
# spacing identical -- which is the point of deriving ``nperseg`` from a shared
# duration rather than from a shared sample count, and is why a genuinely coarser
# record is truncated to its own Nyquist rather than rejected. What survives is a
# residual: ``nperseg`` is rounded to an integer per record, so two records whose
# achieved cadences differ by a relative eps end up with spacings differing by
# ~eps. On the run this was verified against, truth and posterior came out at
# 1.00079 s and 0.99917 s (eps = 1.6e-3) because the LBM timestep is derived from
# each solve's own velocity scale.
#
# The criterion is on the SPACING RATIO, not on the accumulated frequency offset:
# that offset grows as k*eps, so an offset budget would tighten as 1/n and start
# refusing exactly the long records the diagnostic wants (measured: an
# offset-based rule accepted n=480 but refused n=1024 and n=2048 at this run's
# own cadence pair). Mislabelling costs nothing at this size -- bin k truly sits
# at f_k*(1+eps) and is scored as f_k, so for a local slope s the amplitude error
# is 10*s*log10(1+eps) ~ 0.012 dB at eps=1.6e-3, four orders below the 1-3 dB the
# LSD reports, and it is the same at every k.
_GRID_SPACING_RTOL = 1e-2

# How much the truth-halves distance exceeds the like-for-like scatter of the
# reported truth-vs-posterior-median distance. Halving the record at fixed
# ``nperseg`` halves the segment count, so each half's log-spectrum carries ~2x a
# full-record estimate's variance (``LSD_halves^2 ~ 4*var_full``), while the
# posterior side is a median over M members whose own scatter is ~1/M of one
# estimate's and negligible by M ~ 8 (``LSD_null^2 ~ var_full``). Hence 2, measured
# at 1.99 (M=8) and 2.08 (M=32) over 40 trials of statistically identical
# ``f^-5/3`` flows. Not a fitted constant and not exact at small M, where the
# median's own scatter pulls the true ratio below 2 -- which makes the halved
# reference the *stricter* of the two readings there, the safe direction for a
# number whose whole job is to stop "under the floor" being read as a pass.
_HALVES_INFLATION = 2.0


def welch_spectrum(
    series: np.ndarray, fs: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch power spectral density: Hann window, 50 % overlap, detrended.

    Args:
        series: ``(..., n_time)``, **time last**. Leading dims are independent
            series (members, sensors, components).
        fs: Sampling frequency in Hz -- the *achieved* cadence of the record, not
            a nominal one (see :func:`probe_spectra`).
        nperseg: Samples per segment. **The same ``(fs, nperseg)`` must be used
            for the truth and for every member**: the window and the segment
            length fix the estimator's own bias and variance, so two spectra
            computed with different ones differ by the estimator as well as by
            the flow, and the LSD cannot tell those apart.

    Returns:
        ``(f, psd)`` with ``f`` the positive frequencies and ``psd`` of shape
        ``series.shape[:-1] + f.shape``, in units of ``series**2`` per Hz.

        The **zero-frequency bin is dropped**. Linear detrending removes the mean
        and the trend from every segment, so what is left at DC is leakage rather
        than energy; it is also the one bin that has no place on a log frequency
        axis and contributes ``0`` to a premultiplied ``f*E``.

        A row containing any non-finite sample comes back all-``nan``, the same
        policy as :func:`block_bootstrap_std`: consumers drop such rows rather
        than being handed a spectrum computed over a gap. Those rows are held out
        of the transform rather than left to propagate, because the linear detrend
        solves a least-squares fit per segment and *raises* on a gap -- one
        diverged member would otherwise take the whole ensemble's spectra down.

    Raises:
        ValueError: If ``series`` has no time axis, ``fs`` is not positive, or
            ``nperseg`` is under four samples or over the record length. The
            record length is checked here rather than left to scipy, which
            silently shortens the segment and would hand back a spectrum on a
            *different* frequency grid than its siblings -- the one failure this
            function exists to prevent.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim < 1:
        raise ValueError("series must have at least a time dimension")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be a positive frequency, got {fs}")
    n_time = values.shape[-1]
    n_segment = int(nperseg)
    if n_segment < 4:
        raise ValueError(f"nperseg must be at least 4 samples, got {n_segment}")
    if n_segment > n_time:
        raise ValueError(
            f"nperseg={n_segment} exceeds the {n_time} samples in the record; "
            "shortening it here would put this spectrum on a different frequency "
            "grid than the ones it is compared against"
        )

    # The grid a one-sided Welch estimate lands on, DC already dropped. Derived
    # rather than read off the transform's output so it is available even when
    # every row is held out below.
    frequencies = np.fft.rfftfreq(n_segment, d=1.0 / float(fs))[1:]

    rows = values.reshape(-1, n_time)
    usable = np.isfinite(rows).all(axis=1)
    psd = np.full((rows.shape[0], frequencies.size), np.nan)
    if usable.any():
        _, estimated = signal.welch(
            rows[usable],
            fs=float(fs),
            window="hann",
            nperseg=n_segment,
            noverlap=n_segment // 2,
            detrend="linear",
            scaling="density",
            axis=-1,
        )
        psd[usable] = np.asarray(estimated)[..., 1:]
    return frequencies, psd.reshape(values.shape[:-1] + frequencies.shape)


def log_spectral_distance(
    truth_spectrum: np.ndarray, member_spectrum: np.ndarray
) -> np.ndarray:
    """``LSD = sqrt(mean_k [10*log10(E_t/E_m)]^2)`` over the last (frequency) axis.

    The one scalar of the spectral check, in dB. It is a *shape and amplitude*
    distance: 3 dB of it means the model's spectrum sits, in the
    root-mean-square over the band, a factor of two off the truth's at a typical
    frequency. Being a log measure it weights a factor-of-two deficit in the
    smallest resolved eddies exactly as heavily as one in the largest, which is
    the point -- an over-smoothed flow loses energy at the top of the band, where
    a linear-scale distance would not notice.

    Read it against a floor, never on its own: two finite records of the *same*
    flow have a non-zero LSD purely from the estimator's scatter (see
    :func:`probe_spectra`'s ``truth_halves``).

    Args:
        truth_spectrum: ``(..., n_freq)``.
        member_spectrum: ``(..., n_freq)``, broadcast against the truth -- so
            ``(M, n_sensor, n_freq)`` against ``(n_sensor, n_freq)`` gives one
            distance per member and sensor.

    Returns:
        The broadcast leading shape (a 0-d array for two 1-D spectra). A bin
        where either spectrum is not finite or not strictly positive is dropped
        from the mean rather than turning the whole distance into ``inf``: a
        zero-energy bin is a property of one estimate, not of the pair. Rows with
        no usable bin left come back ``nan``.

    Raises:
        ValueError: If the two frequency axes have different lengths -- the
            spectra are then not on one grid, and a distance between them would
            be comparing different frequencies.
    """
    truth = np.asarray(truth_spectrum, dtype=float)
    member = np.asarray(member_spectrum, dtype=float)
    if truth.shape[-1] != member.shape[-1]:
        raise ValueError(
            f"spectra must share one frequency grid: got {truth.shape[-1]} bins "
            f"against {member.shape[-1]}"
        )

    # ``&`` broadcasts, so ``usable`` already has the pair's full shape and its
    # per-row count below is taken over the bins each row actually contributed.
    usable = np.isfinite(truth) & np.isfinite(member) & (truth > 0.0) & (member > 0.0)
    ratio = np.where(usable, truth, 1.0) / np.where(usable, member, 1.0)
    squared = np.where(usable, (10.0 * np.log10(ratio)) ** 2, 0.0)
    n_usable = usable.sum(axis=-1)
    mean = np.where(
        n_usable > 0, squared.sum(axis=-1) / np.where(n_usable > 0, n_usable, 1), np.nan
    )
    return np.asarray(np.sqrt(mean))


def _sample_frequency(probes: xarray.Dataset) -> float | None:
    """The achieved sampling frequency of a probe record, in Hz.

    Read from the record's own ``time`` axis, with the ``output_frequency``
    attribute (a cadence in *seconds*) as the fallback: what the run asked for and
    what the solver wrote are not always the same number, and Welch is a function
    of the latter. A cadence that varies across the record is logged -- Welch
    assumes uniform sampling, and a median step silently papers over a record with
    gaps in it.

    ``None`` when neither source gives a usable cadence, which the caller reads as
    "there is no spectrum to compute here".
    """
    recorded = probes.attrs.get("output_frequency")
    period = float(recorded) if recorded not in (None, "") else None

    times = (
        np.asarray(probes["time"].values, dtype=float)
        if "time" in probes.coords
        else None
    )
    if times is not None and times.size >= 2:
        steps = np.diff(times)
        steps = steps[np.isfinite(steps) & (steps > 0.0)]
        if steps.size:
            spacing = float(np.median(steps))
            spread = float(steps.max() - steps.min())
            if spread > 0.05 * spacing:
                logger.warning(
                    "Probe cadence is not uniform: steps span %.4g s about a "
                    "median of %.4g s. Welch assumes even sampling, so the "
                    "spectrum's frequency axis is only as good as that median",
                    spread,
                    spacing,
                )
            if period is not None and abs(period - spacing) > 0.01 * spacing:
                logger.info(
                    "Probe record says output_frequency=%.4g s but its time axis "
                    "steps by %.4g s; using the time axis",
                    period,
                    spacing,
                )
            return 1.0 / spacing

    if period is not None and period > 0.0:
        return 1.0 / period
    return None


def _sensor_keys(probes: xarray.Dataset) -> list[tuple[float, ...]] | None:
    """Rounded ``(x, y, z)`` per sensor, or ``None`` when the file carries no points."""
    if not all(name in probes.coords for name in _SENSOR_COORDS):
        return None
    columns = [
        np.asarray(probes[name].values, dtype=float).ravel() for name in _SENSOR_COORDS
    ]
    if any(column.size != int(probes.sizes["sensor"]) for column in columns):
        return None
    rounded = np.round(np.stack(columns, axis=1), _SENSOR_MATCH_DECIMALS)
    return [tuple(float(v) for v in row) for row in rounded]


def _aligned_sensors(
    sources: dict[str, xarray.Dataset]
) -> dict[str, np.ndarray] | None:
    """Per source, the sensor indices of the points **every** source holds.

    Matched on the ``sensor_x/y/z`` coordinates and ordered as the truth holds
    them, so every array the bundle returns is indexed by the same physical point.
    Falls back to positional indexing only when a source carries no sensor
    coordinates *and* all the counts agree -- the assumption is then stated in the
    log rather than made silently.

    ``None`` when the sources share no sensor, which is the caller's cue that
    there is nothing to compare.
    """
    keys = {name: _sensor_keys(ds) for name, ds in sources.items()}
    if any(value is None for value in keys.values()):
        sizes = {name: int(ds.sizes["sensor"]) for name, ds in sources.items()}
        if len(set(sizes.values())) != 1 or not min(sizes.values()):
            logger.info(
                "probe_spectra: probe files carry no sensor coordinates and "
                "different sensor counts (%s), so their sensors cannot be paired",
                sizes,
            )
            return None
        logger.info(
            "probe_spectra: at least one probe file carries no sensor_x/y/z "
            "coordinates; pairing the %d sensors by position instead",
            next(iter(sizes.values())),
        )
        count = next(iter(sizes.values()))
        return {name: np.arange(count) for name in sources}

    shared = set.intersection(*(set(value) for value in keys.values()))
    truth_keys = keys["truth"]
    assert truth_keys is not None  # every entry is non-None in this branch
    # ``dict.fromkeys`` deduplicates while keeping the truth's order: a probe point
    # listed twice (a case config with a repeated sensor) would otherwise be drawn
    # as two identical panels and counted twice in the median over sensors, and the
    # index map below keeps only one position for it anyway.
    order = [key for key in dict.fromkeys(truth_keys) if key in shared]
    if not order:
        logger.info(
            "probe_spectra: the truth and member probe files share no sensor "
            "location, so there is nothing to compare"
        )
        return None
    dropped = len(set(truth_keys)) - len(order)
    if dropped:
        logger.info(
            "probe_spectra: %d of the truth's %d probe sensors are absent from a "
            "member probe file and are not scored",
            dropped,
            len(set(truth_keys)),
        )
    index = {}
    for name, value in keys.items():
        assert value is not None
        position = {key: i for i, key in enumerate(value)}
        index[name] = np.array([position[key] for key in order], dtype=int)
    return index


def _probe_series(probes: xarray.Dataset, sensors: np.ndarray) -> np.ndarray:
    """``(component, ..., sensor, time)`` velocity series at the paired sensors."""
    stacked = []
    for name in _PROBE_COMPONENTS:
        field = probes[name].transpose(..., "sensor", "time")
        # One advanced index among slices keeps the axis in place, so the sensor
        # axis stays second-to-last whether or not there is a member axis.
        stacked.append(np.asarray(field.values, dtype=float)[..., sensors, :])
    return np.stack(stacked)


def _energy_spectra(
    probes: xarray.Dataset, sensors: np.ndarray, fs: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(f, E)`` with ``E`` the trace ``E_uu + E_vv + E_ww`` of the spectral tensor.

    One spectrum per sensor rather than three. The component spectra are what
    ``welch_spectrum`` produces, and the trace is the quantity the diagnostic is
    about -- the turbulent energy's distribution over frequency, whose inertial
    range is the ``-5/3`` this figure draws a guide for. Anisotropy is already
    covered, and covered better, by the resolved ``<u'w'>`` in the mean-field
    block; splitting the spectra per component instead would triple the panels
    and the reported scalars to say something a third metric already says.

    A member axis, when the record has one, stays leading: ``E`` is
    ``(M, sensor, freq)`` for an ensemble record and ``(sensor, freq)`` for the
    truth.
    """
    frequencies, psd = welch_spectrum(_probe_series(probes, sensors), fs, nperseg)
    return frequencies, np.asarray(psd.sum(axis=0))


def _usable_record(name: str, probes: xarray.Dataset) -> bool:
    """Whether a probe record carries the components and axes a spectrum needs."""
    missing = [v for v in _PROBE_COMPONENTS if v not in probes.data_vars]
    if missing or "time" not in probes.dims or "sensor" not in probes.dims:
        logger.info(
            "probe_spectra: the %s probe record is missing %s",
            name,
            ", ".join(missing) or "its time/sensor dims",
        )
        return False
    return True


def _grids_are_the_same_bins(
    name: str,
    grid: np.ndarray,
    reference: np.ndarray,
    *,
    consequence: str = "the spectral comparison is skipped",
) -> bool:
    """Whether ``grid`` resolves the same frequency bins as ``reference``.

    Judged on bin SPACING (see :data:`_GRID_SPACING_RTOL`), which is
    length-independent, rather than on how far the two grids have drifted apart by
    the last bin, which is not. Records built on a shared segment duration always
    pass; what this rejects is a record whose spacing genuinely disagrees, i.e.
    one that was not built against the same duration at all. When a small residual
    is tolerated the comparison proceeds on ``reference``'s bins, so it is logged.
    """
    if grid.size < 2 or reference.size < 2:
        # One bin carries no spacing; compare the frequencies themselves, where
        # the same relative tolerance is the honest test.
        return bool(
            grid.size
            and reference.size
            and abs(float(grid[0]) / float(reference[0]) - 1.0) <= _GRID_SPACING_RTOL
        )
    spacing = float(grid[1] - grid[0])
    reference_spacing = float(reference[1] - reference[0])
    mismatch = abs(spacing / reference_spacing - 1.0)
    if mismatch > _GRID_SPACING_RTOL:
        logger.info(
            "probe_spectra: the %s record's bin spacing is %.6g Hz against the "
            "truth's %.6g Hz (%.2f%% apart), so its bins are not the same "
            "frequencies -- %s",
            name,
            spacing,
            reference_spacing,
            100.0 * mismatch,
            consequence,
        )
        return False
    if mismatch > 0.0:
        logger.info(
            "probe_spectra: the %s record's cadence differs from the truth's, so its "
            "bin spacing sits %.3f%% off; scoring it on the truth's bins (worth "
            "~%.3g dB of mislabelling, far below what the LSD resolves)",
            name,
            100.0 * mismatch,
            4.343 * mismatch,
        )
    return True


def _matched_spectra(
    name: str,
    probes: xarray.Dataset,
    sensors: np.ndarray,
    cadence: float,
    segment_seconds: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One record's spectra on the shared segment *duration*, member axis kept.

    ``None`` with a log line when the record cannot carry that duration -- the one
    place the shared-estimator rule is enforced per record, so the truth, the
    posterior and the prior all reach it the same way.
    """
    n_segment = int(round(segment_seconds * cadence))
    available = int(probes.sizes["time"])
    if n_segment < 4 or n_segment > available:
        logger.info(
            "probe_spectra: a %.4g s Welch segment is %d samples at the %s record's "
            "cadence, which its %d samples cannot carry",
            segment_seconds,
            n_segment,
            name,
            available,
        )
        return None
    frequencies, spectra = _energy_spectra(probes, sensors, cadence, n_segment)
    if name != "truth" and spectra.ndim == 2:
        # No ensemble axis (a single-member probe run): keep the member axis so
        # every consumer indexes the arrays the same way.
        spectra = spectra[None]
    return frequencies, spectra


def _fitted_prior(
    prior: xarray.Dataset,
    core: dict[str, xarray.Dataset],
    sensors: dict[str, np.ndarray],
    frequencies: np.ndarray,
    segment_seconds: float,
) -> tuple[np.ndarray | None, float | None]:
    """The prior's spectra on the *already fixed* sensors and frequency grid.

    Returns ``(spectra, cadence)``, both ``None`` when the prior cannot be fitted
    onto that grid. It is fitted rather than negotiated with because the prior
    record is optional: letting it take part in choosing the sensors, the shared
    band or the cutoff would make ``lsd_posterior_median`` -- a mandatory number --
    depend on whether an optional rerun happened, and a prior at half the cadence
    would silently halve the band the posterior is scored on.

    A prior that resolves *fewer* bins than the scored grid is kept and padded with
    ``nan`` above its own Nyquist rather than dropped: the per-bin masking in
    :func:`log_spectral_distance` then scores it over the bins it does have, and S4
    draws its envelope where it exists. Only a prior that cannot be aligned at all
    -- other sensors, another frequency spacing, a record too short for the shared
    segment -- is dropped.
    """
    if not _usable_record("prior", prior):
        return None, None
    cadence = _sample_frequency(prior)
    if cadence is None:
        logger.info("probe_spectra: the prior probe record carries no usable cadence")
        return None, None

    with_prior = _aligned_sensors({**core, "prior": prior})
    if with_prior is None or not np.array_equal(with_prior["truth"], sensors["truth"]):
        logger.info(
            "probe_spectra: the prior probe record does not cover the same sensors "
            "as the truth and the posterior; S4 loses its prior envelope rather "
            "than the comparison losing sensors"
        )
        return None, None

    estimated = _matched_spectra(
        "prior", prior, with_prior["prior"], cadence, segment_seconds
    )
    if estimated is None:
        return None, None
    grid, spectra = estimated

    shared = min(grid.size, frequencies.size)
    if not _grids_are_the_same_bins(
        "prior",
        grid[:shared],
        frequencies[:shared],
        consequence="S4 loses its prior envelope",
    ):
        return None, None
    if shared < frequencies.size:
        logger.info(
            "probe_spectra: the prior probe record resolves %d of the %d scored "
            "frequency bins (it was written at a coarser cadence); its envelope "
            "stops there and its LSD is scored over those bins only",
            shared,
            frequencies.size,
        )
    padded = np.full(spectra.shape[:-1] + (frequencies.size,), np.nan)
    padded[..., :shared] = spectra[..., :shared]
    return padded, cadence


def probe_spectra(
    truth: xarray.Dataset,
    posterior: xarray.Dataset,
    prior: xarray.Dataset | None = None,
    segments: int = SPECTRUM_SEGMENTS,
) -> dict | None:
    """Matched Welch energy spectra at every probe sensor, truth and members.

    The shared reduction behind both consumers of the probe files -- the
    ``spectral_metrics`` block (:func:`spectral_metric_summary`) and figure S4
    (``evaluation.figures.plot_spectra``) -- so the numbers in the summary and the
    curves in the figure cannot be computed two different ways.

    Args:
        truth: Probe record of the truth: ``u``, ``v``, ``w`` on
            ``(time, sensor)``, spin-up already excluded, with the sensor
            coordinates and the achieved cadence on it (see
            :func:`_sample_frequency`).
        posterior: The same for the posterior members, with a leading
            ``ensemble`` dim. A record without one is treated as a single member.
        prior: Optional prior-member record. Prior probe reruns are optional, so
            its absence is the common case and costs only S4's prior envelope and
            ``lsd_prior_median`` -- see "the prior never moves a reported number"
            below, which is a property this function enforces rather than a hope.
        segments: Welch segments per record; see :data:`SPECTRUM_SEGMENTS`.

    Returns:
        ``None`` -- with the reason logged -- whenever the records cannot be
        compared: a missing component or axis, a cadence neither the time axis nor
        the attributes give, no shared sensor, frequency grids that do not line
        up, or a record too short to leave :data:`_MIN_BAND_BINS` bins under the
        cutoff. Every one of those is an input property rather than a bug, and
        each costs this block alone (master-plan invariant 3).

        Otherwise a dict:

        * ``f`` ``(K,)`` -- the shared positive frequencies, up to the lowest
          Nyquist among the records. It runs **past** the cutoff on purpose: S4
          draws the whole resolved band and marks where scoring stops.
        * ``band`` ``(K,)`` bool -- ``f < f_cutoff``, the bins comparisons use.
        * ``f_cutoff``, ``segment_seconds``, ``sample_frequency`` -- the cutoff in
          Hz, the Welch segment duration in seconds, and each record's achieved
          cadence in Hz.
        * ``truth`` ``(sensor, K)``, ``posterior`` ``(M, sensor, K)``, ``prior``
          ``(M, sensor, K)`` or ``None`` -- the trace spectra.
        * ``truth_halves`` ``(2, sensor, K)`` -- the same spectrum from the first
          and second half of the truth record: the truth measured against itself,
          which is the only reference this diagnostic can produce without a second
          truth run. Deliberately computed with the **same** ``nperseg`` as the full
          record, so both spectra come off the same estimator.

          **It is ~2x the like-for-like scatter, and it is not a pass threshold.**
          Two effects, both in the same direction. Halving the record at fixed
          ``nperseg`` halves the segment count, so each half's log-spectrum carries
          ~2x the variance of a full-record estimate: ``LSD_halves^2 ~ 2*var_half =
          4*var_full``. The reported ``lsd_posterior_median`` instead compares the
          full truth against a *median over M members*, whose own scatter is ~1/M of
          one estimate and negligible by M ~ 8: ``LSD_posterior^2 ~ var_full``.
          Hence ``LSD_halves / LSD_null -> 2``, measured at 1.99 (M=8) and 2.08
          (M=32) over 40 trials of statistically identical ``f^-5/3`` flows. So a
          posterior at 2 dB against a 2.5 dB halves distance is **not**
          indistinguishable from the truth -- its like-for-like reference is ~1.2 dB
          and it is well outside it. :func:`spectral_metric_summary` reports the
          halved value beside it as ``lsd_truth_floor_comparable`` for exactly this
          reason.
        * ``variance`` ``(sensor,)`` -- the truth's total resolved variance
          ``sum_i var(u_i)`` (``ddof=1``), i.e. twice its resolved TKE. It is what
          S4 premultiplies by, and it comes from the truth for both records: a
          curve normalized by its *own* variance would let a member with half the
          energy overlay the truth exactly, which is the failure this check
          exists to catch.
        * ``sensor_sets`` ``(sensor,)`` -- ``"assimilation"`` / ``"validation"``
          per scored sensor when the file labels them, so a figure can put the
          held-out sensors first.

    **One shared segment duration, not one shared sample count.** When the truth
    and the members were written at the same cadence those are the same statement
    and ``nperseg`` is identical. When they differ -- a pre-existing truth that
    could not be re-run at the probe cadence -- the shared quantity has to be the
    segment's length *in seconds*, since that is what fixes the frequency
    resolution and the spectral window in physical units; the sample count then
    differs per record and the comparison is restricted to the band both resolve
    (the lower Nyquist), which is the metrics doc's "common resolved band". No
    resampling happens anywhere: interpolating a coarse record onto a fine axis
    invents high-frequency content, which is precisely the quantity under test.

    **The prior never moves a reported number.** The scored sensors, the shared
    band, the cutoff and the segment duration are all derived from the truth and
    the posterior alone; the prior record is then fitted onto that result and
    dropped (with a log line) if it does not fit -- see :func:`_fitted_prior`. Doing
    it the other way round is a real trap rather than a hypothetical one: a prior
    written at half the cadence would drag ``min(fs)`` down, halve ``f_cutoff`` and
    the bins under it, and so change ``lsd_posterior_median`` -- a mandatory
    number -- depending on whether an optional rerun happened.
    """
    if int(segments) < 2:
        # A programmer error rather than an input property, hence a raise: at one
        # segment the record cannot also be halved for the self-distance floor,
        # which is not an optional part of this bundle.
        raise ValueError(f"segments must be at least 2, got {segments}")
    if "ensemble" in truth.dims:
        # The truth is one realisation; a member axis on its record means the
        # rerun wrote it through the ensemble path. Drop it so the truth is one
        # series per sensor, as everything below assumes.
        truth = truth.isel(ensemble=0)
    # ``core`` is what every reported number is derived from. The prior is fitted
    # onto that result afterwards and can only ever be *dropped* -- see the
    # "the prior never moves a reported number" paragraph in this docstring.
    core: dict[str, xarray.Dataset] = {"truth": truth, "posterior": posterior}

    for name, ds in core.items():
        if not _usable_record(name, ds):
            return None

    fs: dict[str, float] = {}
    for name, ds in core.items():
        cadence = _sample_frequency(ds)
        if cadence is None:
            logger.info(
                "probe_spectra: the %s probe record carries no usable cadence "
                "(neither a time axis with two frames nor output_frequency)",
                name,
            )
            return None
        fs[name] = cadence

    n_truth = int(truth.sizes["time"])
    nperseg_truth = n_truth // int(segments)
    if nperseg_truth < 4:
        logger.info(
            "probe_spectra: %d truth samples leave %d per segment at %d segments, "
            "which is not a spectrum",
            n_truth,
            nperseg_truth,
            segments,
        )
        return None
    segment_seconds = nperseg_truth / fs["truth"]

    sensors = _aligned_sensors(core)
    if sensors is None:
        return None

    grids: dict[str, np.ndarray] = {}
    spectra: dict[str, np.ndarray] = {}
    for name, ds in core.items():
        estimated = _matched_spectra(name, ds, sensors[name], fs[name], segment_seconds)
        if estimated is None:
            return None
        grids[name], spectra[name] = estimated

    n_common = min(grid.size for grid in grids.values())
    frequencies = grids["truth"][:n_common]
    if not _grids_are_the_same_bins(
        "posterior", grids["posterior"][:n_common], frequencies
    ):
        return None

    f_cutoff = SPECTRUM_CUTOFF_FRACTION * 0.5 * min(fs.values())
    band = frequencies < f_cutoff
    if int(band.sum()) < _MIN_BAND_BINS:
        logger.info(
            "probe_spectra: only %d frequency bins fall under the %.4g Hz cutoff "
            "(need %d) -- the records are too short, or too coarsely sampled, for "
            "a spectral comparison",
            int(band.sum()),
            f_cutoff,
            _MIN_BAND_BINS,
        )
        return None

    half = n_truth // 2
    halves = np.stack(
        [
            _energy_spectra(
                truth.isel(time=slice(0, half)),
                sensors["truth"],
                fs["truth"],
                nperseg_truth,
            )[1],
            _energy_spectra(
                truth.isel(time=slice(n_truth - half, n_truth)),
                sensors["truth"],
                fs["truth"],
                nperseg_truth,
            )[1],
        ]
    )

    series = _probe_series(truth, sensors["truth"])  # (component, sensor, time)
    variance = np.asarray(np.var(series, axis=-1, ddof=1).sum(axis=0))

    labels = (
        [str(s) for s in np.asarray(truth["sensor_set"].values).ravel()]
        if "sensor_set" in truth.coords
        else ["" for _ in range(int(truth.sizes["sensor"]))]
    )
    if prior is not None:
        prior_spectra, prior_cadence = _fitted_prior(
            prior, core, sensors, frequencies, segment_seconds
        )
    else:
        prior_spectra, prior_cadence = None, None
    if prior_cadence is not None:
        fs["prior"] = prior_cadence
    return {
        "f": frequencies,
        "band": band,
        "f_cutoff": float(f_cutoff),
        "segment_seconds": float(segment_seconds),
        "sample_frequency": {name: float(value) for name, value in fs.items()},
        "truth": spectra["truth"][..., :n_common],
        "truth_halves": halves[..., :n_common],
        "posterior": spectra["posterior"][..., :n_common],
        "prior": prior_spectra,
        "variance": variance,
        "sensor_sets": tuple(labels[i] for i in sensors["truth"]),
    }


def median_spectrum(members: np.ndarray) -> np.ndarray:
    """Bin-wise median over the leading member axis, ignoring non-finite members.

    The posterior median spectrum -- the object the LSD scores and S4 draws --
    with the same member policy as the S5 fan: a member whose probe series had a
    gap has an all-``nan`` spectrum (see :func:`welch_spectrum`) and is dropped
    rather than blanking the median. A bin where no member is finite stays
    ``nan``. The median commutes with the log, so this is also the median in dB.
    """
    values = np.asarray(members, dtype=float)
    with warnings.catch_warnings():
        # A bin where every member is non-finite is a legitimate outcome (an
        # ensemble that all diverged, a sensor inside a solid cell), and ``nan``
        # is the right answer for it -- but ``nanmedian`` calls it a RuntimeWarning
        # per bin, which would bury the operator's log in thousands of lines.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmedian(values, axis=0))


def _median_over_sensors(distances: np.ndarray) -> float | None:
    """Median LSD over the sensors, or ``None`` when no sensor produced one.

    The median rather than the mean: one sensor sitting in a solid cell or in a
    stagnation point can carry a wild distance, and the headline number is meant
    to describe a typical probe. ``None`` (not ``nan``) because this goes into
    YAML, where the rest of the summary spells an unmeasurable entry ``null``.
    """
    values = np.asarray(distances, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def spectral_metric_summary(spectra: dict) -> dict:
    """The ``spectral_metrics`` block: the LSD beside the truth's own references.

    Args:
        spectra: A :func:`probe_spectra` bundle.

    Returns:
        A YAML-safe dict. ``lsd_posterior_median`` is the log-spectral distance
        between the truth's spectrum and the **member-median** spectrum, per
        sensor, reduced over sensors by the median -- two medians, one over
        members (the object being scored) and one over sensors (the reduction).

        Two references beside it, and they are not interchangeable.
        ``lsd_truth_floor`` is the distance between the two halves of the truth
        record; it is what the halves measure and nothing more, and it runs **~2x**
        the scatter this comparison would show under a null of identical flows (the
        derivation and the measured 1.99--2.08 are in :func:`probe_spectra`).
        ``lsd_truth_floor_comparable`` is that value halved, which *is* the
        like-for-like reference: read ``lsd_posterior_median`` against **it**.
        Being under ``lsd_truth_floor`` is not a pass criterion and neither number
        is a threshold -- there is no acceptance level for the LSD in the metrics
        doc, only the comparison against the prior and against these references.

        ``lsd_prior_median`` appears only when prior probes were run, and is what
        makes the posterior number readable as a change rather than as an absolute.
        ``f_cutoff`` (Hz) with ``n_band_bins``, ``segment_seconds`` and
        ``sample_frequency`` record the band the comparison ran on; ``n_sensors``
        how many probe points it covered and ``n_members`` how many members the
        median was taken over. **``n_members`` is not bookkeeping**: the median of
        M spectra is smoother the larger M is, so the distance falls with ensemble
        size on identical flows (measured 1.409 dB at M=2 against 1.171 dB at
        M=32), and two runs' LSDs are comparable only at equal M.

        Every distance is in dB and is ``null`` when no sensor produced a finite
        one.
    """
    band = spectra["band"]
    truth = spectra["truth"][:, band]
    halves_distance = _median_over_sensors(
        log_spectral_distance(
            spectra["truth_halves"][0][:, band], spectra["truth_halves"][1][:, band]
        )
    )
    entry: dict[str, object] = {
        "n_sensors": int(truth.shape[0]),
        "n_members": int(np.asarray(spectra["posterior"]).shape[0]),
        "n_band_bins": int(np.count_nonzero(band)),
        "f_cutoff": float(spectra["f_cutoff"]),
        "segment_seconds": float(spectra["segment_seconds"]),
        "sample_frequency": dict(spectra["sample_frequency"]),
        "lsd_posterior_median": _median_over_sensors(
            log_spectral_distance(truth, median_spectrum(spectra["posterior"])[:, band])
        ),
        "lsd_truth_floor": halves_distance,
        # The halves' distance divided by the factor derived in ``probe_spectra``:
        # halving the record doubles each estimate's log-spectral variance (4x in
        # LSD^2) while the posterior side's median over members contributes ~1/M of
        # one estimate's, so the like-for-like reference is half the halves'.
        "lsd_truth_floor_comparable": (
            None if halves_distance is None else halves_distance / _HALVES_INFLATION
        ),
    }
    if spectra.get("prior") is not None:
        entry["lsd_prior_median"] = _median_over_sensors(
            log_spectral_distance(truth, median_spectrum(spectra["prior"])[:, band])
        )
    return entry
