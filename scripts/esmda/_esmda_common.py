"""Shared post-processing helpers for the ESMDA three-script pipeline.

``scripts/esmda/run_esmda.py`` runs the assimilation and saves the raw artifacts;
``scripts/esmda/compute_esmda_metrics.py`` turns those into ``run_summary.yaml`` and
``scripts/esmda/make_esmda_figures.py`` draws the figures. The metric and figure
stages both need the same lazy truth access, sensor-series extraction and
small scalar/YAML helpers, so they live here instead of being duplicated.

Everything is read-only with respect to the run directory except the explicit
``*.to_netcdf`` / ``_write_yaml`` writes the callers perform; the truth state is
always opened lazily (see :func:`open_truth`) so a multi-GB truth is never held
in memory in full.
"""

# mypy: ignore-errors
# Legacy untyped helper module: it predates the strict mypy config and is
# type-checked transitively whenever a script importing it is committed.
# Waived wholesale rather than annotated piecemeal; drop this when typed.

from __future__ import annotations

import logging
import pathlib

import numpy as np
import scipy.linalg
import xarray
import yaml
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_operator import ObservationOperator
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_validation_points,
)
from pyurbanair.plotting import (
    compute_parameter_metrics,
    compute_sensor_metrics,
    param_members_and_x,
    plotted_param_names,
)
from pyurbanair.utils.ensemble_scores import (
    coverage,
    coverage_nominal_alpha,
    crpss,
    fair_crps,
    fair_energy_score,
    max_abs_zscore_reference,
    max_nominal_alpha,
    pit_rank,
    rank_histogram,
    rank_histogram_weights,
    wasserstein_over_floor,
    wasserstein_over_floor_reference,
    wasserstein_over_sigma,
    wasserstein_self_floor,
    zscore,
    zscore_exceedance,
)
from pyurbanair.utils.turbulence_stats import (
    DEFAULT_HIT_RATE_TOLERANCE,
    StreamingMoments,
    block_bootstrap_std_batch,
    colocate_components,
    extrapolated_centre_dims,
    fac2,
    fractional_bias,
    hit_rate,
    nmse,
    nmse_split,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run-directory loaders (config + persisted truth-access view + sensor sets)
# ---------------------------------------------------------------------------


def load_run_config(run_dir: pathlib.Path):
    """Re-load the composed Hydra config saved by run_esmda.py."""
    return OmegaConf.load(run_dir / "config.yaml")


def read_yaml(path: pathlib.Path) -> dict:
    """Load a small YAML mapping (``{}`` if the file is missing/empty)."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_sensor_sets(cfg) -> dict:
    """``{"assimilation": (x, y, z)[, "validation": (x, y, z)]}`` from the obs config.

    The validation set is the held-out sensors (scored but never assimilated);
    it is only present when the obs config defines it.
    """
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)
    sensor_sets = {"assimilation": (obs_x, obs_y, obs_z)}
    validation_points = create_validation_points(cfg.obs)
    if validation_points is not None:
        sensor_sets["validation"] = validation_points
    return sensor_sets


# ---------------------------------------------------------------------------
# Lazy truth-state access (only load the slices that are actually needed)
# ---------------------------------------------------------------------------

_Z_DIMS = ("z", "zm", "zt")
_X_COORDS = ("x", "xt", "xm")


def truth_x_min(ds):
    """Smallest x face/centre coordinate of a (truth) dataset, or 0.0 if none.

    Prefers the staggered face coordinate (``xm``) so the domain *edges* align
    when computing the offset; falls back to ``x``/``xt`` for non-udales grids.
    """
    for c in ("xm", "x", "xt"):
        if c in ds.coords:
            return float(np.asarray(ds[c].values).min())
    return 0.0


def open_truth(true_state_path, n_total, x_offset=0.0, start_idx=0, t_offset=0.0):
    """Lazily open the truth state, limited to ``n_total`` frames from ``start_idx``.

    Uses ``open_dataset`` (not ``load_dataset``) so the data stays on disk; the
    caller's subsequent ``.isel``/reduction then materialises only the slice it
    needs. A multi-GB truth is therefore never pulled into memory in full -- the
    window loop reads one window at a time, the plots one z-plane at a time.

    ``start_idx`` drops the leading frames before the chosen start time, so the
    assimilation horizon begins partway into a pre-simulated truth (e.g. after a
    spin-up). ``t_offset`` then rebases the kept frames' time coordinate so the
    chosen start time becomes t=0, keeping the window loop and plots on a
    [0, final_time) axis regardless of where the truth was sliced.

    ``x_offset`` shifts every x coordinate so a truth saved in its own frame
    (e.g. x in [0, 100]) lines up with the simulation domain (e.g. x in
    [-20, 80]). Applied on every open so the observation operator, the window
    loop and the plots all see the truth in domain coordinates.
    """
    ds = xarray.open_dataset(true_state_path)
    if n_total is not None:
        ds = ds.isel(time=slice(start_idx, start_idx + n_total))
    elif start_idx:
        ds = ds.isel(time=slice(start_idx, None))
    if t_offset and "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"] - t_offset)
    if x_offset:
        shifted = {c: ds[c] + x_offset for c in _X_COORDS if c in ds.coords}
        if shifted:
            ds = ds.assign_coords(shifted)
    return ds


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
# Per-member streaming reads of the full-ensemble window state files
# ---------------------------------------------------------------------------

# What the window state files are read for. Restricting the per-member read to
# the velocity triple is not tidiness: ``run_esmda._stream_concat_members``
# copies *every* data variable the solver wrote into the window file, so a
# solver carrying pressure or scalars would otherwise have its extra fields
# paid for in full by post-processing that reads none of them.
_STATE_VELOCITY_VARS = ("u", "v", "w")


def stream_window_members(state_paths, variables=_STATE_VELOCITY_VARS):
    """Yield ``(window_index, member_index, member_state)``, one member at a time.

    The sanctioned reader for ``windows/window_*_{posterior,prior}_state.nc``.
    Those files hold the whole ensemble -- ~1 GB at smoke scale, tens of GB at
    Barcelona scale -- so they are opened lazily and sliced ``.isel(ensemble=m)``;
    nothing here ever holds more than one member. Reading is sequential (one
    reader), which is under the two-reader cap by construction.

    **Every** consumer that needs the window states is fed from this one pass:
    at production scale a second full pass over tens of GB is not affordable, so
    the pass is shared rather than duplicated. Two properties of the yielded
    member follow from that, and both are deliberate:

    *It is materialised* (in memory, one member's fields), because xarray does
    not cache reads taken through ``.isel``: two consumers slicing the same
    *lazy* member read the same bytes twice. Measured by counting
    ``NetCDF4ArrayWrapper._getitem`` calls on a fabricated
    ``(ensemble 8, time 8, z 6, y 7, x 9)`` window file, in elements read, where
    one full ensemble is 72576 elements:

    ==============================================  =====  ========
    shape                                           reads  elements
    ==============================================  =====  ========
    full-ensemble ``.load()`` (what this replaced)      3     72576
    lazy member, one consumer                          24     72576
    lazy member, two consumers                         48    145152
    materialised member, two consumers                 24     72576
    ==============================================  =====  ========

    So a materialised member reads each member's bytes exactly once however many
    consumers are attached, at a peak of one member's velocity fields instead of
    the ensemble's. It also lets a consumer keep the yielded object past its loop
    iteration without reading through a closed file handle.

    *It is not co-located onto cell centres*, although the WP1.3 plan text has
    this generator applying ``colocate_components``. The consumers need
    different things from the same bytes: :func:`member_sensor_series`
    trilinearly interpolates each component on its *own* staggered grid (which
    is what makes the sensor series grid- and solver-independent), while the
    ``turbulence_stats.StreamingMoments`` accumulators need co-located arrays.
    Co-locating here would destroy the former and charge its cost to every
    consumer, so deriving stays with the consumer.

    Args:
        state_paths: Per-window state file paths in window order. The position
            in the sequence *is* the ``window_index`` yielded, which is what the
            time rebasing ``(t - t[0]) + w*sim_time`` is keyed on.
        variables: Data variables to read per member; ``None`` reads every data
            variable in the file. Defaults to ``("u", "v", "w")``, all any
            current consumer uses.

    Yields:
        ``(window_index, member_index, member_state)`` with ``member_state`` an
        in-memory ``Dataset`` on the file's own (possibly staggered) dims. Its
        scalar ``ensemble`` coordinate is present only when the file has one --
        the disk-backed run path writes the ``ensemble`` *dimension* with no
        coordinate variable, the in-memory path writes both.

    Raises:
        KeyError: a requested variable is missing from a window file.
        ValueError: a window file has no ``ensemble`` dimension, i.e. it is not
            a full-ensemble state file.
    """
    for window_index, path in enumerate(state_paths):
        # A context manager rather than a trailing close(): the handle has to go
        # away when a consumer raises and when the generator is abandoned
        # part-way through a window (GeneratorExit is thrown at the yield below
        # and unwinds through here), not only on the happy path.
        with xarray.open_dataset(path) as ds:
            names = list(ds.data_vars) if variables is None else list(variables)
            missing = [name for name in names if name not in ds.data_vars]
            if missing:
                raise KeyError(
                    f"{path} is missing data variable(s) {missing}; it carries "
                    f"{tuple(ds.data_vars)}"
                )
            if "ensemble" not in ds.dims:
                raise ValueError(
                    f"{path} has no 'ensemble' dimension (dims {tuple(ds.dims)}); "
                    "stream_window_members reads full-ensemble window state files"
                )
            subset = ds[names]
            for member_index in range(ds.sizes["ensemble"]):
                # `.compute()` is `.load()` on a shallow copy, so state this
                # plainly: this *is* a read into memory -- of one member, which
                # is the granularity the never-`.load()`-a-window-file rule
                # prescribes. `subset` itself stays lazy, so the next member is
                # still a fresh slab read rather than a slice of something
                # already resident.
                yield (
                    window_index,
                    member_index,
                    subset.isel(ensemble=member_index).compute(),
                )


# ---------------------------------------------------------------------------
# Sensor time-series extraction (truth vs ensemble at fixed points)
# ---------------------------------------------------------------------------


def _sensor_component_timeseries(state, obs_x, obs_y, obs_z, solver_name):
    """Per-component ``(u, v, w)`` velocity time series at each sensor point.

    Trilinearly interpolates u/v/w (each on its own staggered grid, resolved via
    an ``ObservationOperator``'s solver-specific dim mapping) at the sensor
    locations, keeping any leading dims (``ensemble``, ``time``). Returns a
    DataArray with a leading ``component`` dim: ``(component, ..., time, sensor)``.
    The velocity magnitude |U| is :func:`sensor_magnitude` of this (used for the
    sensor figures); the full vector is used for the sensor error metrics.
    """
    op = ObservationOperator(
        obs_x=list(np.asarray(obs_x, dtype=float)),
        obs_y=list(np.asarray(obs_y, dtype=float)),
        obs_z=list(np.asarray(obs_z, dtype=float)),
        obs_states=["u", "v", "w"],
        solver_name=solver_name,
    )
    comps = []
    for var in ("u", "v", "w"):
        dims = op.dim_mapping[var]
        comps.append(
            interpolate_dataarray_at_points(
                state[var],
                x_dim=dims["x"],
                y_dim=dims["y"],
                z_dim=dims["z"],
                obs_x=op.obs_x,
                obs_y=op.obs_y,
                obs_z=op.obs_z,
            )
        )
    return xarray.concat(comps, dim="component").assign_coords(
        component=["u", "v", "w"]
    )


def sensor_magnitude(components):
    """Velocity magnitude |U| from a ``(component, ...)`` sensor series."""
    return np.sqrt((components**2).sum("component"))


def _concat_sensor_pieces(pieces):
    """Concatenate per-window sensor series along ``time`` for each sensor set."""
    return {
        name: (
            parts[0]
            if len(parts) == 1
            else xarray.concat(parts, dim="time", join="override")
        )
        for name, parts in pieces.items()
    }


def member_sensor_series(
    member_state, sensor_sets, solver_name, window_index=0, sim_time=0.0
):
    """Per-component sensor series for **one** already-open member of one window.

    The per-member unit of :func:`ensemble_sensor_series`, exposed so the WP1.3
    mean-field pass can extract the sensor series from the same
    :func:`stream_window_members` read that feeds its moment accumulators,
    instead of re-reading the window files.

    The window-local -> global time rebasing lives here and is load-bearing:
    window ``w``'s frames are placed on ``[w*sim_time, (w+1)*sim_time)``, which
    is the rule :func:`sensor_statistic_scores` slices the ensemble by (by time
    *value*, against the truth's by-index slicing). Rebasing only happens when
    the state carries a ``time`` coordinate, exactly as before -- a file without
    one keeps its positional axis.

    Args:
        member_state: One member's state, e.g. a ``stream_window_members`` yield.
            Components stay on their own staggered dims; this function needs
            them raw, since that is what the interpolation resolves against.
        sensor_sets: ``{name: (x, y, z)}`` from :func:`build_sensor_sets`.
        solver_name: Solver whose staggered dim mapping the state follows.
        window_index: Which rollout window this member came from.
        sim_time: Seconds per window; the rebasing stride.

    Returns:
        ``{name: DataArray(component, time, sensor)}``, carrying the member's
        scalar ``ensemble`` coordinate when the state had one.
    """
    t = (
        np.asarray(member_state["time"].values, dtype=float)
        if "time" in member_state.coords
        else None
    )
    # `interpolate_dataarray_at_points` rebuilds its output coords from the
    # output *dims* only, so the member label is dropped on the way through.
    # Re-attaching it here is what lets the ensemble concat in
    # `SensorSeriesAccumulator.result` rebuild the `ensemble` coordinate the
    # old full-ensemble read got straight off the file.
    member_id = member_state["ensemble"] if "ensemble" in member_state.coords else None
    series = {}
    for name, (ox, oy, oz) in sensor_sets.items():
        vel = _sensor_component_timeseries(member_state, ox, oy, oz, solver_name)
        if t is not None and "time" in vel.dims:
            vel = vel.assign_coords(time=(t - t[0]) + window_index * sim_time)
        if member_id is not None:
            vel = vel.assign_coords(ensemble=member_id)
        series[name] = vel
    return series


_MEMBER_SERIES_DIMS = ("component", "time", "sensor")
_WINDOW_SERIES_DIMS = ("component", "ensemble", "time", "sensor")


def _stack_window_members(members):
    """One window's per-member sensor series -> ``(component, ensemble, time, sensor)``.

    The dims and the values are the easy part: any stacking gives those. The
    *memory layout* is the hard part, and it is not cosmetic. Every consumer of
    this series reduces over some of its axes (``fair_energy_score`` over
    components, the WP1.2 statistics over time and sensors), numpy's pairwise
    summation walks a reduction in memory order, and floating-point addition is
    not associative -- so two arrays with identical contents in different layouts
    give answers that differ in the last ULP. That is visible in
    ``run_summary.yaml``, whose numbers WP1.2's calibration references were
    measured against, so this function reproduces the layout the pre-WP1.3
    full-ensemble read produced. Measured by diffing every leaf of the summary
    computed on the WP1.2 wiring fixture (587 leaves, 260 of them floats)
    against the same summary from the old code path:

    ==========================================================  ==============
    how the window's members are stacked                        leaves changed
    ==========================================================  ==============
    ``concat(dim="ensemble")`` then ``transpose``                            4
    ...then forced C-contiguous                                             20
    this function                                                            0
    ==========================================================  ==============

    All the differences were 1-2 ULP (the largest relative move, 2e-14, was
    ``crpss_vs_prior``, a ratio of near-equal CRPS values). They are not a
    change to any estimator -- but "the metric is unchanged to 15 digits" is a
    much weaker claim to hand a reviewer than "the summary is byte-identical",
    and the four lines below buy the stronger one.

    The layout being matched is ``(component, sensor, ensemble, time)``, which
    is where the old path landed for a reason worth recording: for a
    full-ensemble read ``interpolate_dataarray_at_points`` gathers the eight
    trilinear corners with ``arr[..., z0, y0, x0]``, and numpy's advanced
    indexing lays that result out with the sensor axis outermost; xarray's
    concatenations then preserve it. Nothing documents that, so the guard below
    checks the shapes rather than trusting them: if the member series are not
    exactly ``(component, time, sensor)`` -- a window file with no time axis,
    say -- the plain concatenation is returned and the layout match is silently
    given up, which costs those last ULPs and nothing else.

    Args:
        members: One window's :func:`member_sensor_series` outputs for one sensor
            set, in member order.

    Returns:
        ``DataArray(component, ensemble, time, sensor)``.
    """
    stacked = xarray.concat(
        [
            xarray.concat(
                [member.isel(component=i) for member in members], dim="ensemble"
            )
            for i in range(members[0].sizes["component"])
        ],
        dim="component",
    )
    if any(member.dims != _MEMBER_SERIES_DIMS for member in members):
        return stacked
    # (component, time, sensor) per member -> a C-contiguous
    # (component, sensor, ensemble, time) buffer -> a view of it in the dim order
    # every consumer expects.
    buffer = np.ascontiguousarray(
        np.stack([member.values.transpose(0, 2, 1) for member in members], axis=2)
    ).transpose(0, 2, 3, 1)
    if stacked.dims != _WINDOW_SERIES_DIMS or stacked.shape != buffer.shape:
        return stacked
    return stacked.copy(data=buffer)


class SensorSeriesAccumulator:
    """Collects per-member sensor series from a :func:`stream_window_members` pass.

    Holds one :func:`member_sensor_series` result per ``(window, member, sensor
    set)`` and assembles them into the same
    ``{name: DataArray(component, ensemble, time, sensor)}``
    :func:`ensemble_sensor_series` has always returned. What it accumulates is
    tiny -- ``components x members x frames x sensors`` -- so the memory bound of
    a streaming pass is set by the yielded member, never by this.

    Members may be fed in any order: pieces are keyed by ``(window, member)``
    and sorted in :meth:`result`, so a caller is free to reorder its own pass.
    """

    def __init__(self, sensor_sets, solver_name, sim_time):
        self.sensor_sets = sensor_sets
        self.solver_name = solver_name
        self.sim_time = sim_time
        # {window_index: {set_name: {member_index: DataArray}}}
        self._pieces = {}

    def add_member(self, window_index, member_index, member_state):
        """Extract and stash one member's sensor series for every sensor set."""
        series = member_sensor_series(
            member_state,
            self.sensor_sets,
            self.solver_name,
            window_index=window_index,
            sim_time=self.sim_time,
        )
        by_set = self._pieces.setdefault(window_index, {})
        for name, vel in series.items():
            by_set.setdefault(name, {})[member_index] = vel

    def result(self):
        """``{name: DataArray(component, ensemble, time, sensor)}`` over all windows.

        Members within a window, then windows along the (already rebased, hence
        global) ``time`` axis -- the same two concatenations, in the same order,
        the full-ensemble read did.
        """
        pieces = {name: [] for name in self.sensor_sets}
        for window_index in sorted(self._pieces):
            for name, by_member in self._pieces[window_index].items():
                members = [by_member[m] for m in sorted(by_member)]
                pieces[name].append(_stack_window_members(members))
        return _concat_sensor_pieces(pieces)


def ensemble_sensor_series(state_paths, sensor_sets, solver_name, sim_time):
    """Ensemble per-component ``(u, v, w)`` sensor series across rollout windows.

    Streams each window's full-ensemble state file **member at a time**
    (:func:`stream_window_members`) and interpolates u/v/w at every sensor set's
    points (keeping ``component`` + ``ensemble`` + ``time``), rebasing each
    window's local time onto a single global axis (window ``w`` starts at
    ``w*sim_time``) so it lines up with the truth. Returns
    ``{name: DataArray(component, ensemble, time, sensor)}``.

    The return value is unchanged from the full-ensemble ``.load()`` this
    replaced -- bit-identical, pinned by
    ``tests/test_esmda_stream_members.py`` against a verbatim copy of the old
    implementation, because WP1.2's ``sensor_statistic_scores`` numbers are
    calibrated on it. What changed is the memory bound: one member's velocity
    fields rather than the ensemble's, which is what the ~1 GB (smoke) to
    tens-of-GB (Barcelona) window files require.
    """
    accumulator = SensorSeriesAccumulator(sensor_sets, solver_name, sim_time)
    for window_index, member_index, member_state in stream_window_members(state_paths):
        accumulator.add_member(window_index, member_index, member_state)
    return accumulator.result()


def truth_sensor_series(
    true_state_path,
    n_total,
    x_offset,
    start_idx,
    t_offset,
    sensor_sets,
    solver_name,
    num_windows,
    n_per_window,
):
    """Truth per-component ``(u, v, w)`` sensor series, one window at a time.

    Mirrors the window loop's memory discipline: only one window's worth of the
    (potentially multi-GB) truth is held at once. The truth's ``time`` axis is
    already global, so the per-window pieces concatenate directly. Returns
    ``{name: DataArray(component, time, sensor)}``.
    """
    pieces = {name: [] for name in sensor_sets}
    for w in range(num_windows):
        ts = open_truth(true_state_path, n_total, x_offset, start_idx, t_offset).isel(
            time=slice(w * n_per_window, (w + 1) * n_per_window)
        )
        for name, (ox, oy, oz) in sensor_sets.items():
            pieces[name].append(
                _sensor_component_timeseries(ts, ox, oy, oz, solver_name)
            )
        ts.close()
    return _concat_sensor_pieces(pieces)


# ---------------------------------------------------------------------------
# Vector (u,v,w) sensor error metrics
# ---------------------------------------------------------------------------


def _energy_score(members, truth):
    """Per-timestep energy score, averaged over sensors.

    Sensor-shaped adapter around
    :func:`pyurbanair.utils.ensemble_scores.fair_energy_score` (the
    multivariate CRPS; see there for the estimator). The only work done here
    is the axis convention: the sensor series carry the component axis first,
    the shared score wants ensemble first and components last.

    Memory, stated exactly because it is *not* identical to the hand-rolled
    loop this replaced. The **pairwise** term is unchanged: the shared
    implementation loops over the leading batch axis, which after the move is
    time, so it holds one ``(E, E, S, C)`` slab -- the same element count as
    the old ``(C, E, E, S)``. The **distance-to-truth** term is not: the shared
    implementation takes it over the whole batch in one ``(E, T, S, C)``
    allocation, where the old code held ``(C, E, S)`` inside the time loop.
    That is a factor ``n_time`` on that term.

    Accepted rather than chunked, because it does not change what binds: peak
    is ``max(E*T*S*C, E*E*S*C)``, so the pairwise slab still dominates whenever
    ``M > n_time``, and where it does not the term1 array is a plain dense
    float64 buffer (M=32, T=1e3, S=3, C=3: ~7 MB). A **field**-shaped caller
    has neither property and must chunk the grid into the leading batch axis --
    see the memory bound in :func:`fair_energy_score`.

    Args:
        members: ``(component, ensemble, time, sensor)`` aligned member vectors.
        truth: ``(component, time, sensor)`` aligned truth vectors.

    Returns:
        ``(time,)`` energy score, averaged over the sensors (matching the
        per-time, over-sensors reduction of ``compute_sensor_metrics``).
    """
    per_sensor = fair_energy_score(
        np.moveaxis(np.asarray(members), 0, -1),  # (E, T, S, C)
        np.moveaxis(np.asarray(truth), 0, -1),  # (T, S, C)
    )
    return per_sensor.mean(axis=-1)  # average over sensors -> (T,)


def vector_sensor_metrics(truth_comp, ensemble_comp):
    """Full-vector ``(u, v, w)`` sensor error, reduced over sensors per timestep.

    One scalar per sensor per timestep is formed from the whole velocity vector
    (not just its magnitude), then reduced over sensors:

      * ``rmse(t) = sqrt(mean_s || <v>_ens - v_truth ||^2)`` -- the ensemble-mean
        vector error, obtained by combining the per-component
        :func:`compute_sensor_metrics` RMSEs as ``sqrt(sum_c rmse_c**2)`` (so the
        shared, time-aligning metric is reused unchanged).
      * ``energy_score(t)`` -- the multivariate CRPS (:func:`_energy_score`) over
        the aligned member/truth vectors.

    Args:
        truth_comp: ``(component, time, sensor)`` truth series.
        ensemble_comp: ``(component, ensemble, time, sensor)`` ensemble series.

    Returns:
        ``{"rmse": (T,), "energy_score": (T,)}``.
    """
    components = [str(c) for c in np.asarray(truth_comp["component"].values)]
    # Per-component metrics reuse the shared compute_sensor_metrics, which also
    # time-aligns the truth onto the ensemble axis identically for each
    # component, so the returned members/truth stack into consistent vectors.
    per = {
        c: compute_sensor_metrics(
            truth_comp.sel(component=c), ensemble_comp.sel(component=c)
        )
        for c in components
    }
    rmse = np.sqrt(np.sum([per[c]["rmse"] ** 2 for c in components], axis=0))  # (T,)
    members = np.stack([per[c]["members"] for c in components], axis=0)  # (C,E,T,S)
    truth = np.stack([per[c]["truth"] for c in components], axis=0)  # (C,T,S)
    return {"rmse": rmse, "energy_score": _energy_score(members, truth)}


# ---------------------------------------------------------------------------
# Scalar / YAML helpers
# ---------------------------------------------------------------------------


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python for safe YAML."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_yaml(data, path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(_to_native(data), f, sort_keys=False, default_flow_style=False)


def series_stats(arr):
    """{mean, final, max, min} of a 1-D series, or ``None`` if it has no values.

    ``final`` is the last element (the end-of-rollout value); the rest reduce
    over the whole series. NaNs are ignored.
    """
    a = np.asarray(arr, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    return {
        "mean": float(finite.mean()),
        "final": float(a[-1]) if np.isfinite(a[-1]) else None,
        "max": float(finite.max()),
        "min": float(finite.min()),
    }


def parameter_metric_summary(posterior_params, true_params, prior_params):
    """Per-parameter RMSE/CRPS summary stats (posterior, with a prior reference).

    Both ``*_reduction_vs_prior`` ratios are :func:`ensemble_scores.crpss`
    (``1 - post/prior``), not a local copy: the shared function was written
    against this code's guard and evaluates the identical expression on
    identical inputs. Verified bit-identical over 2e5 random positive
    ``(post, prior)`` pairs -- 0 differing.

    Only the **non-finite** corners move, and every one of them moves from a
    number that was wrong to a ``null``, because the old guard tested the
    denominator alone (``prior_mean > 0``) while ``crpss`` tests both operands:

    ======================  =========  ======
    case                    was        now
    ======================  =========  ======
    post ``nan``, prior 4   ``nan``    ``null``
    post ``inf``, prior 4   ``-inf``   ``null``
    post 2, prior ``inf``   ``1.0``    ``null``
    ======================  =========  ======

    The last is the substantive one: an infinite reference score was reported as
    a 100% reduction. The first two are the convention every other float in this
    section already follows (:func:`_finite_or_none`) -- a bare ``nan``/``inf``
    reaches ``compare_sweep_results.py`` as a number and poisons the aggregate.
    """
    metrics = compute_parameter_metrics(posterior_params, true_params, prior_params)
    summary = {}
    for name, m in metrics.items():
        entry = {"rmse": series_stats(m["rmse"]), "crps": series_stats(m["crps"])}
        for score in ("rmse", "crps"):
            if f"prior_{score}" not in m:
                continue
            prior_mean = float(np.nanmean(m[f"prior_{score}"]))
            entry[f"prior_{score}_mean"] = prior_mean
            entry[f"{score}_reduction_vs_prior"] = crpss(
                float(np.nanmean(m[score])), prior_mean
            )
        summary[name] = entry
    return summary


# ---------------------------------------------------------------------------
# Parameter calibration bundle (WP1.1)
# ---------------------------------------------------------------------------

# Number of PIT/rank-histogram bins. Fixed rather than configurable so counts
# from different runs stack in one figure.
PIT_BINS = 10

# Central credible levels scored by the coverage block.
COVERAGE_ALPHAS = (0.5, 0.9)

# Below this ensemble size every calibration diagnostic in the bundle measures
# the ensemble SIZE rather than the ensemble: a ddof=1 spread has a single
# degree of freedom, the widest available order-statistic band is [x_(1), x_(M)]
# (nominal (M-1)/(M+1) = 1/3 at M = 2, so a "90%" coverage cannot be attained),
# and PIT ranks take M + 1 = 3 values spread over 10 bins. The two-member smoke
# shape hits exactly this. Per the master plan's rule for degenerate shapes the
# answer is a logged `null`, never a special-cased formula.
MIN_MEMBERS_CALIBRATION = 3

# The joint block additionally needs a covariance with more than one degree of
# freedom; at M = 2 every sample correlation is exactly +/-1 and the generalized
# spectrum is an artifact of the regularizer.
MIN_MEMBERS_JOINT = 3

# Full K x K correlation matrices are written to run_summary.yaml only below
# this size. `yaml.safe_dump(default_flow_style=False)` puts one number per
# line, so two K x K matrices cost ~2*K*(K+1) lines: measured on a real summary,
# K = 8 adds 130 lines to a 99-line file and K = 16 adds ~514 -- i.e. the
# summary stops being a file a human reads at a glance well before the K = 42 of
# a routine 2-parameter/21-knot run (~3.5k lines). The cap keeps the matrices for
# the small joint vectors where they are genuinely readable; above it they are
# omitted and `corr_summary` carries the off-diagonal scalars. Nothing consumes
# the full matrices yet -- when something does, the place for them is WP1.3's
# `eval_fields.nc`-style sidecar, not this file.
JOINT_CORR_MAX_K = 8

# Same reasoning for the eigenvalue list, which is only `r` long.
JOINT_EIGENVALUE_MAX = 64

# How many parameter-vector entries to report per eigenvector direction.
JOINT_LOADINGS = 5

# Tie-breaking seed for `pit_rank`, recorded in the summary so a reader can
# reproduce the counts.
PIT_TIE_SEED = 0

# Which of `ensemble_scores.DEFAULT_ZSCORE_THRESHOLDS` the `overconfident` flag
# is read off, and by how much the observed exceedance fraction must beat its
# nominal level. Index 0 is |z| > 2, NOT the |z| > 3 tail: at the pooled sample
# sizes this stage actually has (tens to a few hundred knots) the 3-sigma tail
# holds O(1) expected counts, so any rule on it is a coin flip on a single knot.
# Measured false-positive rate on a PERFECTLY CALIBRATED M = 32 ensemble
# (4000 trials, independent knots), against the rule this replaces:
#
#   pooled knots        21     63    105    315
#   max|z| > 3        0.118  0.316  0.451  0.852   <- the old flag
#   |z|>3, x2         0.118  0.316  0.124  0.122
#   |z|>2, x2         0.114  0.029  0.007  0.000   <- this rule
#   |z|>2, x3         0.031  0.001  0.000  0.000
#
# Only the last two shrink with sample size, which is the whole point: pooling
# more knots must sharpen a calibration verdict, not manufacture one. Between
# them the multiplier is a power trade, measured at M = 32 / 315 knots on an
# ensemble whose spread is a factor `s` too small:
#
#   s                  1.0    0.8    0.7    0.5
#   |z|>2, x2         0.000  0.649  0.998  1.000   <- this rule
#   |z|>2, x3         0.000  0.003  0.521  1.000
#
# x3 cannot see a 20%-too-narrow ensemble at all, so x2 it is: ~0 false alarms
# at the routine shape and it still catches the miscalibration that matters.
OVERCONFIDENT_THRESHOLD_INDEX = 0
OVERCONFIDENT_MULTIPLIER = 2.0


def _finite_or_none(value):
    """Plain float, or ``None`` when the value is nan/inf.

    ``write_yaml`` round-trips numpy scalars fine but not ``nan``/``inf``
    (``.nan``/``.inf`` are not read back as floats by every YAML consumer), so
    every float leaving this section goes through here -- matching how
    ``parameter_metric_summary`` and ``ensemble_scores.crpss`` already guard.
    """
    number = float(value)
    return number if np.isfinite(number) else None


def parameter_vector_labels(params) -> list[str]:
    """``["inflow_angle[0]", ...]`` naming each column of the flattened members.

    Mirrors ``compute_esmda_metrics._flatten_parameter_members`` exactly --
    same variable order (sorted), same ensemble-first transpose, same row-major
    reshape -- so label ``i`` names column ``i`` of that flattening. The two
    live in different modules only because the flattener sits next to its other
    caller; ``tests/test_parameter_bundle.py`` pins them together.
    """
    labels: list[str] = []
    for name in sorted(params.data_vars):
        variable = params[name]
        if "ensemble" not in variable.dims:
            continue
        values = variable.transpose("ensemble", ...)
        n_entries = int(np.prod(values.shape[1:], dtype=int)) if values.ndim > 1 else 1
        if n_entries == 1:
            labels.append(str(name))
        else:
            labels.extend(f"{name}[{i}]" for i in range(n_entries))
    return labels


def _aligned_parameter_arrays(posterior_params, true_params, prior_params=None):
    """Per parameter: members, truth on the members' x-axis, and prior members.

    The alignment is the one :func:`pyurbanair.plotting.compute_parameter_metrics`
    performs (``param_members_and_x`` + ``np.interp`` onto the posterior
    x-axis), reused rather than re-derived: it already handles a static truth
    (one point -> constant) and a time-varying truth sampled on a different
    knot grid (the routine case -- e.g. 19 truth knots against 21 posterior
    knots) with the same two lines.

    Yields ``(name, members, truth, prior_members, x_is_time)`` where
    ``members`` is ``(n_members, n_x)`` and ``prior_members`` is ``None`` unless
    the prior exists on the same x-axis.

    This re-implements ``compute_parameter_metrics``'s two alignment lines
    rather than calling it, because that helper returns *scores* and never
    exposes the aligned members this bundle needs. The shared piece is the pair
    of public helpers below, so there is still exactly one definition of a
    parameter's x-axis and of the parameter order.

    ``x_is_time`` is the per-parameter dynamic/static discriminator -- there is
    no run-level split, because a single ``conf/params/dynamic.yaml`` run mounts
    time-varying ``external_parameters`` AND a ``static_parameters`` block into
    one Dataset, so both PIT branches fire in the same run. The test is
    ``param_members_and_x``'s own, repeated here because that helper returns
    the axis but not the reason for it: a **``time`` coordinate**, not merely a
    ``time`` dimension. The dimension alone is not enough -- ``_concat_windows``
    stacks per-window parameter files along ``time``, so in a purely static run
    every parameter comes out with a length-``num_windows`` ``time`` dimension
    and no ``time`` coordinate, and its x-axis is the window index.
    """
    for name in plotted_param_names(posterior_params, true_params):
        posterior_da = posterior_params[name]
        x_est, members = param_members_and_x(posterior_da)

        true_da = true_params[name]
        if "ensemble" in true_da.dims:
            true_da = true_da.isel(ensemble=0)
        x_true, true_members = param_members_and_x(true_da.expand_dims("ensemble"))
        order = np.argsort(x_true)
        truth = np.interp(x_est, np.asarray(x_true)[order], true_members[0][order])

        prior_members = None
        if prior_params is not None and name in prior_params.data_vars:
            _, candidate = param_members_and_x(prior_params[name])
            if candidate.shape == members.shape:
                prior_members = np.asarray(candidate, dtype=float)

        non_ensemble_dims = [d for d in posterior_da.dims if d != "ensemble"]
        yield (
            name,
            np.asarray(members, dtype=float),
            np.asarray(truth, dtype=float),
            prior_members,
            non_ensemble_dims == ["time"] and "time" in posterior_da.coords,
        )


def _knot_correlation_config(cfg):
    """``(correlation_length, seconds_per_knot)`` from a run's saved config.

    Both live at the top level of the ``prior_params`` block (as
    interpolations, e.g. ``seconds_per_knot: ${time.seconds_per_knot}``, hence
    ``OmegaConf.select`` rather than a raw YAML read). Either may be absent --
    a run whose parameters are all static never sets them -- and an absent
    value is reported as ``null``, never guessed.
    """
    if cfg is None:
        return None, None

    def _select(key):
        try:
            value = OmegaConf.select(cfg, f"prior_params.{key}", default=None)
        except Exception:  # unresolvable interpolation in a partial config
            return None
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) and number > 0 else None

    return _select("correlation_length"), _select("seconds_per_knot")


# A knot-to-knot step counts as a real step once it exceeds this fraction of the
# parameter's own ensemble spread. Scaled to the spread, not to the values: the
# `np.isclose` default (rtol = 1e-5) is relative to the *magnitude*, which is
# 2.7e-3 degrees against an inflow angle near 270 but 1e-7 against an SGS
# constant near 0.01 -- i.e. a tolerance set by where the parameter's origin
# happens to be. A broadcast static parameter repeats bitwise-identical values
# (difference exactly 0) and a genuinely time-varying one steps by O(spread),
# so the two are separated by many orders of magnitude and the exact fraction
# below is not delicate.
_SEGMENT_STEP_FRACTION = 1e-6


def _n_constant_segments(members):
    """Number of piecewise-constant runs along the x-axis of ``(M, n_x)`` members.

    A parameter that is static but rides on a dynamic run's time axis (the
    ``static_parameters`` block of ``conf/params/dynamic.yaml`` is broadcast to
    every knot by ``_concat_windows``) is piecewise constant with one step per
    assimilation window. Counting the steps recovers that window count without
    needing to know which block the parameter came from, and returns ``n_knots``
    (i.e. no clamp) for a genuinely time-varying parameter, whose every knot
    differs by O(ensemble spread) -- far above ``_SEGMENT_STEP_FRACTION``.
    """
    if members.shape[1] < 2:
        return int(members.shape[1])
    scale = float(np.nanstd(members)) if np.any(np.isfinite(members)) else 0.0
    tolerance = _SEGMENT_STEP_FRACTION * scale if np.isfinite(scale) else 0.0
    steps = np.abs(members[:, 1:] - members[:, :-1]) > tolerance
    return int(np.sum(np.any(steps, axis=0))) + 1


def _effective_knot_count(
    members, x_is_time, correlation_length, seconds_per_knot, num_windows, name
):
    """``(n_knots_effective, pooling)`` -- the caveat attached to the PIT counts.

    Pooled PIT ranks are only as informative as the number of *independent*
    samples behind them, which is never the raw knot count:

    * time-varying parameters are a GP with correlation length ``L``, so
      ``n_knots * seconds_per_knot / L`` knots are independent -- clamped at
      ``n_knots`` because a sub-knot correlation length cannot manufacture more
      independent samples than there are knots, and at the number of distinct
      piecewise-constant segments, which is what actually bounds a static
      parameter broadcast onto a time axis;
    * a parameter whose x-axis is the window index gets one value per
      assimilation window, so the pool is ``num_windows`` deep -- and the
      windows are linked by the cross-window carry-over, making even that an
      upper bound (``pooling: windows_correlated``).
    """
    n_knots = int(members.shape[1])
    if not x_is_time:
        if num_windows is None:
            logger.info(
                "Parameter %r has no time dimension and num_windows is unknown; "
                "reporting n_knots_effective as null",
                name,
            )
            return None, "windows_correlated"
        return int(num_windows), "windows_correlated"

    if correlation_length is None or seconds_per_knot is None:
        logger.info(
            "Parameter %r is time-varying but the saved config has no "
            "correlation_length/seconds_per_knot; reporting n_knots_effective "
            "as null rather than assuming one",
            name,
        )
        return None, "knots_correlated"
    independent = int(np.ceil(n_knots * seconds_per_knot / correlation_length))
    segments = _n_constant_segments(members)
    return int(max(1, min(n_knots, segments, independent))), "knots_correlated"


def _zscore_block(members, truth):
    """Pooled per-knot z-score summary, **each number next to its reference**.

    ``mean`` / ``std`` / ``max_abs`` are the raw pooled moments, unchanged. What
    is new is that nothing here may be read against a standard normal:

    * ``exceedance`` -- :func:`ensemble_scores.zscore_exceedance`, the observed
      ``|z|`` tail fractions **with the calibrated levels they must be compared
      against**. A calibrated ensemble's z-scores are not N(0,1) but
      ``sqrt((M + 1)/M) * t(M - 1)`` (mean and spread come from the same M
      members), which at M = 32 puts ``P(|z| > 3)`` at 0.59% -- 2.2x the normal
      table. WP1.4 plots ``observed`` against ``nominal`` from this block and
      must not re-derive either.
    * ``max_abs_calibrated_median`` -- where ``max |z|`` sits for a *calibrated*
      ensemble of this size over this many knots
      (:func:`ensemble_scores.max_abs_zscore_reference`). Without it ``max_abs``
      is unreadable: the max of ``n`` draws grows with ``n``, so the same 3.0
      means "high" at 21 knots and "low" at 315. Deliberately evaluated at the
      RAW knot count rather than ``sampling.n_knots_effective``: the effective
      count is a stated upper bound, and passing a bound would *shrink* the
      reference and bias it toward calling a calibrated ensemble broken. The raw
      count errs the safe way.

    ``overconfident`` is the one boolean, and it is read off the exceedance
    fractions rather than ``max_abs``, because a fixed cut on a maximum flags
    SAMPLE SIZE: at M = 32 a perfectly calibrated ensemble trips ``max|z| > 3``
    12% of the time at 21 pooled knots and 85% at 315 -- and 315 is the routine
    2-parameter/21-knot/3-window shape, so the old flag was very nearly a
    constant `true` on real runs. A tail *fraction* has a sample-size-independent
    expectation, so more knots sharpen it instead of inflating it (measured rates
    at ``OVERCONFIDENT_MULTIPLIER`` above).

    The rule is emitted as ``overconfident_rule``, a string naming the exact
    keys it is computed from, so the verdict is reproducible from the summary
    alone and no consumer has to guess the basis.

    Caveat, and the reason this is the only boolean here: the pooled knots are
    NOT independent (see the ``sampling`` block on the parameter entry). The
    fractions stay unbiased under correlation, but their sampling error does not
    follow ``sqrt(p(1-p)/n)``, so correlation degrades this flag toward its
    small-``n`` behaviour (~11% false alarms) rather than toward the ~0% of the
    nominal 315-knot shape. Treat it as a screen, not a test.
    """
    z = zscore(members, truth)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return None
    n_members = int(np.asarray(members).shape[0])
    exceedance = zscore_exceedance(z, n_members)

    index = OVERCONFIDENT_THRESHOLD_INDEX
    threshold = exceedance["thresholds"][index]
    observed = exceedance["observed"][index]
    nominal = exceedance["nominal"][index]
    return {
        "mean": _finite_or_none(finite.mean()),
        # ddof=1 over knots: one knot gives no spread estimate, not a zero one.
        "std": _finite_or_none(finite.std(ddof=1)) if finite.size > 1 else None,
        "max_abs": _finite_or_none(np.max(np.abs(finite))),
        "max_abs_calibrated_median": _finite_or_none(
            max_abs_zscore_reference(n_members, int(finite.size))
        ),
        "exceedance": exceedance,
        "overconfident": bool(
            np.isfinite(observed) and observed > OVERCONFIDENT_MULTIPLIER * nominal
        ),
        "overconfident_rule": (
            f"exceedance.observed[{index}] > {OVERCONFIDENT_MULTIPLIER} * "
            f"exceedance.nominal[{index}]  (|z| > {threshold})"
        ),
    }


def _pit_block(
    members, truth, x_is_time, correlation_length, seconds_per_knot, num_windows, name
):
    """``(counts, metadata)`` -- pooled rank histogram plus its sample-size caveat."""
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
    if not np.any(valid):
        return None, None
    ranks = pit_rank(members[:, valid], truth[valid], rng=PIT_TIE_SEED)
    counts = rank_histogram(ranks, members.shape[0], n_bins=PIT_BINS)
    n_effective, pooling = _effective_knot_count(
        members[:, valid],
        x_is_time,
        correlation_length,
        seconds_per_knot,
        num_windows,
        name,
    )
    meta = {
        "n_bins": PIT_BINS,
        "n_samples": int(valid.sum()),
        "n_knots_effective": n_effective,
        "pooling": pooling,
        "tie_seed": PIT_TIE_SEED,
        # The reference a plot must divide by. Ranks take M + 1 values, which
        # only divides evenly into PIT_BINS for particular M, so a calibrated
        # ensemble is flat in `count / ranks_per_bin`, NOT in `count`: at
        # M = 32 the bins hold [4, 3, 3, 4, ...] rank values and a perfectly
        # calibrated ensemble shows a +21%/-9% three-bin comb against a flat
        # line. Emitted per parameter because M is a run property but the
        # consumer (WP1.4) sees only the counts.
        "ranks_per_bin": [
            int(w) for w in rank_histogram_weights(members.shape[0], n_bins=PIT_BINS)
        ],
    }
    return [int(c) for c in counts], meta


def _coverage_block(members, truth):
    """Order-statistic coverage at each ``COVERAGE_ALPHAS`` level, **with the
    level it is actually targeting**.

    ``alpha_50`` / ``alpha_90`` are empirical fractions; ``nominal_alpha_50`` /
    ``nominal_alpha_90`` are what a *perfectly calibrated* ensemble of this size
    scores, and they are **not** 0.5 and 0.9. The band edges are member order
    statistics, so the attainable nominal levels are the ``M + 1`` multiples of
    ``1/(M + 1)`` and the requested alpha is rounded to one of them: at M = 32,
    ``alpha = 0.5`` is the band ``[x_(9), x_(25)]``, nominal 0.4848. A consumer
    comparing the empirical 0.4841 measured there against the *requested* 0.5
    reads ~13 sampling sigma of pure discretization as miscalibration.

    Compare ``alpha_X`` against ``nominal_alpha_X``, never against ``X/100``.

    ``max_nominal_alpha`` answers a different question -- the widest band this
    ``M`` can offer at all, i.e. the ceiling a large alpha clamps to -- and is
    :func:`ensemble_scores.max_nominal_alpha` rather than a local
    ``(M - 1)/(M + 1)``, so the two definitions cannot drift apart.

    The band-edge convention itself is not repeated here: ``_band_indices``
    stays private to ``ensemble_scores`` and these two functions are its only
    public consequence.
    """
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
    if not np.any(valid):
        return None
    n_members = int(members.shape[0])
    block = {}
    for alpha in COVERAGE_ALPHAS:
        key = f"alpha_{int(round(alpha * 100))}"
        block[key] = _finite_or_none(
            coverage(members[:, valid], truth[valid], alpha=alpha)
        )
        block[f"nominal_{key}"] = _finite_or_none(
            coverage_nominal_alpha(n_members, alpha)
        )
    block["max_nominal_alpha"] = _finite_or_none(max_nominal_alpha(n_members))
    return block


def _spread_ratio_stats(post_std, prior_std):
    """``{mean, min}`` of ``post_std/prior_std``, skipping collapsed prior knots."""
    ratio = np.full(post_std.shape, np.nan)
    valid = np.isfinite(post_std) & np.isfinite(prior_std) & (prior_std > 0)
    np.divide(post_std, prior_std, out=ratio, where=valid)
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return None
    return {
        "mean": _finite_or_none(finite.mean()),
        "min": _finite_or_none(finite.min()),
    }


def _initial_prior_std(prior_std, num_windows, name):
    """Window 0's prior spread, tiled across the windows -- or ``(None, reason)``.

    ``prior_params.nc`` is a per-window CONCATENATION and only its first block is
    a genuine prior (see :func:`_contraction_block`), so the run-long reference
    is block 0 repeated. One slice covers both artifact layouts, which is why
    there is no branch on the parameter's x-axis:

    * a **dynamic** parameter has ``n_knots = num_windows * knots_per_window``
      columns, so block 0 is the first ``knots_per_window`` of them and knot
      ``j`` of window ``w`` lines up with knot ``j % knots_per_window`` of
      window 0 (the per-window knot grid is the prior sampler's own, shifted);
    * a **static** parameter's x-axis IS the window index -- one column per
      window -- so ``knots_per_window`` comes out 1 and the tile degenerates to
      "column 0, repeated". Same expression, no special case.

    Returns ``(tiled_std, None)`` or ``(None, reason)``; an unusable shape is a
    logged ``null``, never a silent mis-slice.
    """
    if num_windows is None:
        return None, "num_windows is unknown (truth_access.yaml predates the key)"
    n_windows = int(num_windows)
    if n_windows < 1:
        return None, f"num_windows = {n_windows} is not a window count"
    n_knots = int(prior_std.shape[0])
    if n_knots % n_windows:
        reason = (
            f"{n_knots} knots do not divide into {n_windows} windows, so window "
            "0's prior block cannot be identified"
        )
        logger.info(
            "Parameter %r: %s; reporting contraction_ratio.vs_initial_prior as null",
            name,
            reason,
        )
        return None, reason
    knots_per_window = n_knots // n_windows
    return np.tile(prior_std[:knots_per_window], n_windows), None


def _contraction_block(members, prior_members, num_windows, name):
    """Posterior/prior spread ratio against **both** priors that exist.

    The distinction is the whole point of this block, and it is not visible from
    the artifact: ``run_esmda.py`` sets each window's prior to the previous
    window's posterior (GP-extrapolated when the parameter is dynamic, assigned
    bitwise when it is static), so in the concatenated ``prior_params.nc`` block
    ``w`` IS posterior ``w - 1`` and only block 0 is a genuine prior. A ratio
    taken elementwise against that file therefore measures what the LAST update
    did, not what the run did:

    * ``vs_window_prior`` -- ``std_post/std_prior`` knot by knot. Per-window
      contraction. This is the pre-existing number, unchanged.
    * ``vs_initial_prior`` -- the same posterior against window 0's prior block,
      tiled (:func:`_initial_prior_std`). Cumulative contraction, i.e. the
      "how much did assimilation shrink the uncertainty" reading that the name
      ``contraction_ratio`` invites.

    They differ by the window count and the gap is not cosmetic: on a 3-window
    M = 32 construction with a true per-window ratio of 0.6, the per-window
    number reports ``{mean: 0.600, min: 0.600}`` while the cumulative one
    reports ``{mean: 0.392, min: 0.216}`` -- a run that cut spread by 78%
    reading as 40%.

    ``mean`` / ``min`` remain at the top level as aliases of ``vs_window_prior``
    (the plan's schema sketch and any existing consumer index them directly);
    they are assigned from the same mapping, so the two cannot drift.
    ``vs_initial_prior`` always has the same three keys, with ``reason``
    non-null exactly when its numbers are null.
    """
    if prior_members is None:
        return None
    post_std = members.std(axis=0, ddof=1)
    prior_std = prior_members.std(axis=0, ddof=1)

    per_window = _spread_ratio_stats(post_std, prior_std)
    if per_window is None:
        return None

    initial_std, reason = _initial_prior_std(prior_std, num_windows, name)
    cumulative = (
        None if initial_std is None else _spread_ratio_stats(post_std, initial_std)
    )
    if cumulative is None and reason is None:
        reason = "window 0's prior block has no knot with a positive spread"

    return {
        "mean": per_window["mean"],
        "min": per_window["min"],
        "vs_window_prior": per_window,
        "vs_initial_prior": {
            "mean": None if cumulative is None else cumulative["mean"],
            "min": None if cumulative is None else cumulative["min"],
            "reason": reason,
        },
    }


def _correlation_matrix(cov):
    """Correlation matrix of a covariance, with zero-variance rows left as nan."""
    scale = np.sqrt(np.diag(cov))
    outer = scale[:, None] * scale[None, :]
    corr = np.full(cov.shape, np.nan)
    np.divide(cov, outer, out=corr, where=np.isfinite(outer) & (outer > 0))
    return np.clip(corr, -1.0, 1.0)


def _offdiag_abs_stats(corr):
    """``(mean, max)`` of ``|corr|`` off the diagonal, nan-safe."""
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    finite = np.abs(off[np.isfinite(off)])
    if finite.size == 0:
        return None, None
    return _finite_or_none(finite.mean()), _finite_or_none(finite.max())


def _loadings(vector, labels, eigenvalue):
    """The ``JOINT_LOADINGS`` largest-magnitude entries of one direction."""
    norm = float(np.linalg.norm(vector))
    unit = vector / norm if norm > 0 else vector
    order = np.argsort(-np.abs(unit))[:JOINT_LOADINGS]
    return {
        "eigenvalue": _finite_or_none(eigenvalue),
        "loadings": [
            {"parameter": labels[i], "loading": _finite_or_none(unit[i])} for i in order
        ],
    }


def joint_parameter_directions(posterior_flat, prior_flat, labels=None):
    """Generalized posterior/prior spread directions over the joint parameter vector.

    ``posterior_flat`` / ``prior_flat`` are the ``(M, K)`` member matrices from
    ``compute_esmda_metrics._flatten_parameter_members``. The generalized
    eigenvalues of ``(C_post, C_prior)`` are per-direction variance ratios --
    ``lambda < 0.5`` marks a direction whose spread the update at least halved --
    and are invariant to any rescaling of the parameters, so the mixed units of
    the joint vector do not matter.

    **Which update, though.** ``prior_flat`` spans every window of the
    concatenated ``prior_params.nc``, and all but its first block are previous
    windows' posteriors (``run_esmda.py`` chains them). So ``C_prior`` here is a
    per-window reference and ``n_constrained_directions`` counts directions the
    *per-window* updates halved -- the same shift ``_contraction_block``
    documents, applied to the pencil. The callers' answer to the cumulative
    question is a second, smaller pencil built from the final posterior window
    against the window-0 prior block; see
    :func:`joint_directions_vs_initial_prior`. The two are emitted side by side
    (``prior_reference`` names which is which) rather than one silently
    standing in for the other.

    Two deliberate departures from a naive ``scipy.linalg.eigh(C_a, C_b)``:

    * **Rank truncation first.** Ensembles are routinely smaller than the joint
      parameter vector (M = 32 against K = 42 on a 2-parameter, 21-knot run), so
      both covariances are singular with *different* null spaces and the raw
      pencil is meaningless: on real run data an eps-ridged pair returns a
      negative eigenvalue and an 11-order-of-magnitude spread. The problem is
      therefore projected onto the prior's retained eigenbasis (rank cut
      ``lambda_max * finfo.eps * max(shape)``, the ``numpy.linalg.matrix_rank``
      convention that ``data_assimilation.reduction`` also uses), leaving the
      ``r = min(M - 1, K)`` directions the sample actually resolves.
    * The eps ridge ``C + eps * tr(C)/K * I`` is then applied inside that
      subspace, where it only guards an exactly-degenerate posterior direction.

    The truncation is onto the *prior's* basis, which is lossless only when the
    posterior lives in the prior's span -- true within one ESMDA update, not
    across the concatenated windows of ``posterior_params.nc``. What the
    projection kept is therefore reported as ``posterior_variance_retained``
    (``tr(Q' C_post Q) / tr(C_post)``) rather than assumed.

    Returns a mapping with ``null`` leaves (plus a ``reason``) whenever the
    decomposition is not defined.
    """
    posterior_flat = np.asarray(posterior_flat, dtype=float)
    n_members, n_parameters = posterior_flat.shape

    def _unavailable(reason):
        logger.info("Joint parameter directions unavailable: %s", reason)
        return {
            "n_members": int(n_members),
            "n_parameters": int(n_parameters),
            "n_sample_directions": None,
            "rank_deficient": None,
            "posterior_variance_retained": None,
            "n_constrained_directions": None,
            "generalized_eigenvalues": None,
            "eigenvalue_quantiles": None,
            "most_constrained": None,
            "least_constrained": None,
            "posterior_corr": None,
            "prior_corr": None,
            "corr_summary": None,
            "reason": reason,
        }

    if prior_flat is None:
        return _unavailable("no prior parameter ensemble was saved")
    prior_flat = np.asarray(prior_flat, dtype=float)
    if prior_flat.shape != posterior_flat.shape:
        return _unavailable(
            f"prior members {prior_flat.shape} do not match the posterior "
            f"{posterior_flat.shape}"
        )
    if n_members < MIN_MEMBERS_JOINT:
        return _unavailable(
            f"{n_members} members is below the {MIN_MEMBERS_JOINT} needed for a "
            "covariance with more than one degree of freedom"
        )
    if n_parameters < 2:
        return _unavailable("the joint parameter vector has fewer than 2 entries")
    if not (np.all(np.isfinite(posterior_flat)) and np.all(np.isfinite(prior_flat))):
        return _unavailable("the flattened parameter members contain non-finite values")

    if labels is None or len(labels) != n_parameters:
        labels = [f"param[{i}]" for i in range(n_parameters)]

    cov_post = np.cov(posterior_flat, rowvar=False)
    cov_prior = np.cov(prior_flat, rowvar=False)
    corr_post = _correlation_matrix(cov_post)
    corr_prior = _correlation_matrix(cov_prior)

    prior_eigenvalues, prior_vectors = np.linalg.eigh(cov_prior)
    largest = float(prior_eigenvalues.max())
    if not np.isfinite(largest) or largest <= 0:
        return _unavailable("the prior covariance has no positive eigenvalue")
    tol = largest * np.finfo(float).eps * max(n_members, n_parameters)
    retained = prior_eigenvalues > tol
    rank = int(retained.sum())
    if rank < 1:
        return _unavailable("the prior covariance has no numerically nonzero direction")
    basis = prior_vectors[:, retained]

    reduced_post = basis.T @ cov_post @ basis
    reduced_prior = basis.T @ cov_prior @ basis

    # How much posterior variance the prior-eigenbasis projection keeps.
    # For a SINGLE-window ESMDA update the posterior anomalies are linear
    # combinations of the prior anomalies, so the projection is lossless (1.0)
    # by construction. `posterior_params.nc` is a multi-window CONCATENATION,
    # though, and each window block applies its own M x M transform, so the
    # joint row space is NOT contained in the prior's span: a posterior
    # direction the prior never had (inflation, a re-sampled window, a
    # cross-window carry-over) is silently dropped by the truncation. This
    # ratio is the only place that loss is visible -- read `n_constrained_
    # directions` as covering this fraction of the posterior spread, not all
    # of it. Computed pre-ridge, so the regularizer does not flatter it.
    trace_post = float(np.trace(cov_post))
    variance_retained = (
        _finite_or_none(float(np.trace(reduced_post)) / trace_post)
        if np.isfinite(trace_post) and trace_post > 0
        else None
    )

    eps = np.finfo(float).eps * max(n_members, rank)
    identity = np.eye(rank)
    reduced_post = reduced_post + eps * np.trace(reduced_post) / rank * identity
    reduced_prior = reduced_prior + eps * np.trace(reduced_prior) / rank * identity

    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(reduced_post, reduced_prior)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError) as exc:
        return _unavailable(f"the generalized eigendecomposition failed ({exc})")

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    posterior_offdiag = _offdiag_abs_stats(corr_post)
    prior_offdiag = _offdiag_abs_stats(corr_prior)

    joint = {
        "n_members": int(n_members),
        "n_parameters": int(n_parameters),
        # Only the first r directions are sample-resolved; the remaining
        # K - r carry no ensemble information and are dropped, not scored.
        "n_sample_directions": rank,
        "rank_deficient": bool(rank < n_parameters),
        "posterior_variance_retained": variance_retained,
        "n_constrained_directions": int(np.sum(eigenvalues < 0.5)),
        "generalized_eigenvalues": (
            [_finite_or_none(v) for v in eigenvalues]
            if rank <= JOINT_EIGENVALUE_MAX
            else None
        ),
        "eigenvalue_quantiles": {
            key: _finite_or_none(np.quantile(eigenvalues, q))
            for key, q in (
                ("min", 0.0),
                ("p10", 0.1),
                ("median", 0.5),
                ("p90", 0.9),
                ("max", 1.0),
            )
        },
        "most_constrained": _loadings(
            basis @ eigenvectors[:, 0], labels, eigenvalues[0]
        ),
        "least_constrained": _loadings(
            basis @ eigenvectors[:, -1], labels, eigenvalues[-1]
        ),
        "corr_summary": {
            "posterior_offdiag_abs_mean": posterior_offdiag[0],
            "posterior_offdiag_abs_max": posterior_offdiag[1],
            "prior_offdiag_abs_mean": prior_offdiag[0],
            "prior_offdiag_abs_max": prior_offdiag[1],
        },
    }
    if n_parameters <= JOINT_CORR_MAX_K:
        joint["posterior_corr"] = [
            [_finite_or_none(v) for v in row] for row in corr_post
        ]
        joint["prior_corr"] = [[_finite_or_none(v) for v in row] for row in corr_prior]
    else:
        joint["posterior_corr"] = None
        joint["prior_corr"] = None
        joint["corr_matrices_omitted"] = (
            f"K = {n_parameters} exceeds JOINT_CORR_MAX_K = {JOINT_CORR_MAX_K}; "
            "see corr_summary"
        )
    return joint


# Which keys of a `joint_parameter_directions` result the cumulative pencil
# re-reports. A strict subset: the full block already costs ~40 YAML lines, the
# question this one answers is a scalar count, and the loadings/matrices would
# be near-duplicates of the per-window ones. `reason` travels so a degraded
# path is self-explaining, exactly as in the parent block.
JOINT_VS_INITIAL_KEYS = (
    "n_members",
    "n_parameters",
    "n_sample_directions",
    "rank_deficient",
    "posterior_variance_retained",
    "n_constrained_directions",
    "eigenvalue_quantiles",
    "reason",
)


def joint_directions_vs_initial_prior(final_posterior_flat, initial_prior_flat):
    """The joint pencil answering the CUMULATIVE question, summarized.

    :func:`joint_parameter_directions` scores the concatenated posterior against
    the concatenated prior, which -- because ``run_esmda.py`` seeds each window's
    prior from the previous window's posterior -- is a per-window reference.
    This scores the **final window's posterior block** against the **first
    window's prior block**, the only genuine prior in the artifact, so
    ``n_constrained_directions`` means "directions the run as a whole at least
    halved".

    Both blocks are one window wide, so the pencil is dimensionally consistent
    without inventing anything: the two blocks carry the same parameters on the
    same per-window knot grid (the prior sampler reuses its own grid, shifted).
    Comparing different *time* intervals is sound because the knot prior is the
    stationary GP -- window 0's prior covariance is the prior covariance for any
    window's knots -- and that stationarity is the same assumption the run stage
    makes when it extrapolates.

    A side benefit worth stating, since it is why this is cheap: the blocks are
    ``K/num_windows`` wide, so at the routine M = 32 / K = 42 / 3-window shape
    the pencil is 14-dimensional against 31 sample directions -- **full rank**,
    where the parent block is rank-truncated. No new math: this is the same
    function on smaller inputs, reduced to ``JOINT_VS_INITIAL_KEYS``.

    Args:
        final_posterior_flat: ``(M, K/W)`` last window's posterior block, or
            ``None`` when the caller could not slice one.
        initial_prior_flat: ``(M, K/W)`` window 0's prior block, or ``None``.

    Returns:
        A mapping over :data:`JOINT_VS_INITIAL_KEYS`, all-``null`` with a
        ``reason`` when the blocks are unavailable or the pencil degrades.
    """
    if final_posterior_flat is None or initial_prior_flat is None:
        reason = (
            "the per-window blocks could not be sliced from the concatenated "
            "parameter artifacts (see the log for which guard fired)"
        )
        logger.info("Joint cumulative directions unavailable: %s", reason)
        return dict.fromkeys(JOINT_VS_INITIAL_KEYS) | {"reason": reason}
    # Labels are omitted deliberately: this block reports no loadings, so the
    # parent's K-long label list would only be wrong for a K/W-wide vector.
    full = joint_parameter_directions(final_posterior_flat, initial_prior_flat)
    return {key: full.get(key) for key in JOINT_VS_INITIAL_KEYS}


def parameter_bundle_summary(
    base_summary,
    posterior_params,
    true_params,
    prior_params,
    *,
    posterior_flat,
    prior_flat,
    final_posterior_window_flat=None,
    initial_prior_window_flat=None,
    cfg=None,
    num_windows=None,
):
    """Add the WP1.1 calibration bundle to an existing ``parameter_metrics`` mapping.

    Purely additive: every key :func:`parameter_metric_summary` wrote is copied
    through untouched (their paths are hard-coded in
    ``scripts/figure_creation/``), each per-parameter entry gains ``zscore``,
    ``pit_counts``/``pit``, ``sampling``, ``coverage`` and ``contraction_ratio``,
    and a sibling ``joint`` entry is added.

    ``sampling`` sits on the parameter entry rather than inside ``pit`` because
    the caveat it carries is not PIT's. ``pit``, ``zscore.exceedance`` and
    ``coverage.alpha_*`` all pool over the *same* correlated knots, so
    ``n_knots_effective`` bounds the information behind every one of them; the
    same three numbers stay mirrored under ``pit`` for schema stability, copied
    from this one computation so they cannot drift.

    ``posterior_flat`` / ``prior_flat`` are the ``(M, K)`` matrices from
    ``compute_esmda_metrics._flatten_parameter_members``; they are passed in
    rather than recomputed so the pipeline keeps exactly one flattener. The two
    per-window blocks are that same flattener applied to a ``time``-sliced
    Dataset, for the same reason.

    Args:
        base_summary: The mapping returned by :func:`parameter_metric_summary`.
        posterior_params: Posterior parameter Dataset.
        true_params: Truth parameter Dataset (any knot grid).
        prior_params: Prior parameter Dataset, or ``None``.
        posterior_flat: ``(M, K)`` flattened posterior members.
        prior_flat: ``(M, K)`` flattened prior members, or ``None``.
        final_posterior_window_flat: ``(M, K/W)`` last window's posterior block,
            or ``None`` when it could not be sliced.
        initial_prior_window_flat: ``(M, K/W)`` window 0's prior block, or
            ``None``. See :func:`joint_directions_vs_initial_prior`.
        cfg: The run's saved config, read for the GP knot correlation.
        num_windows: Assimilation window count. Two uses: the pooling depth for
            parameters without a ``time`` dimension, and the width of window 0's
            prior block in ``contraction_ratio.vs_initial_prior``.

    Returns:
        A new mapping; ``base_summary`` is not mutated.
    """
    summary = {name: dict(entry) for name, entry in base_summary.items()}
    correlation_length, seconds_per_knot = _knot_correlation_config(cfg)
    n_members = int(np.asarray(posterior_flat).shape[0])

    degenerate = n_members < MIN_MEMBERS_CALIBRATION
    if degenerate:
        logger.info(
            "Ensemble of %d members is below the %d needed for the parameter "
            "calibration bundle (ddof=1 spreads, order-statistic bands and "
            "%d-bin PIT all degenerate); emitting nulls",
            n_members,
            MIN_MEMBERS_CALIBRATION,
            PIT_BINS,
        )

    scored = set()
    for name, members, truth, prior_members, x_is_time in _aligned_parameter_arrays(
        posterior_params, true_params, prior_params
    ):
        scored.add(name)
        entry = summary.setdefault(name, {})
        if degenerate:
            entry.update(
                {
                    "zscore": None,
                    "pit_counts": None,
                    "pit": None,
                    "sampling": None,
                    "coverage": None,
                    "contraction_ratio": None,
                }
            )
            continue
        pit_counts, pit_meta = _pit_block(
            members,
            truth,
            x_is_time,
            correlation_length,
            seconds_per_knot,
            num_windows,
            name,
        )
        entry.update(
            {
                "zscore": _zscore_block(members, truth),
                "pit_counts": pit_counts,
                "pit": pit_meta,
                # The pooling caveat applies to zscore and coverage just as much
                # as to PIT; hoisted so it is read once per parameter instead of
                # being found under one of the three blocks it qualifies.
                "sampling": (
                    None
                    if pit_meta is None
                    else {
                        key: pit_meta[key]
                        for key in ("n_samples", "n_knots_effective", "pooling")
                    }
                ),
                "coverage": _coverage_block(members, truth),
                "contraction_ratio": _contraction_block(
                    members, prior_members, num_windows, name
                ),
            }
        )

    # The joint vector spans every variable carrying an `ensemble` dim, but the
    # per-parameter bundle iterates `plotted_param_names`, a fixed tuple. They
    # agree today. A future estimated parameter missing from that tuple would
    # enter `joint` while silently getting no calibration entry, so say so the
    # first time it happens rather than leaving it to be noticed in a figure.
    skipped = sorted(
        name
        for name in posterior_params.data_vars
        if "ensemble" in posterior_params[name].dims and name not in scored
    )
    if skipped:
        logger.info(
            "Parameters %s carry an ensemble dimension and are in the joint "
            "vector, but are not in plotting.plotted_param_names, so they get "
            "no per-parameter calibration entry",
            ", ".join(repr(name) for name in skipped),
        )

    joint = joint_parameter_directions(
        posterior_flat, prior_flat, parameter_vector_labels(posterior_params)
    )
    # Names the reference the block above is scored against, so the per-window
    # reading is explicit rather than inferred from the plan text.
    joint["prior_reference"] = "per_window_prior"
    joint["vs_initial_prior"] = joint_directions_vs_initial_prior(
        final_posterior_window_flat, initial_prior_window_flat
    )
    summary["joint"] = joint
    return summary


# ---------------------------------------------------------------------------
# Statistics-space sensor scoring (WP1.2)
# ---------------------------------------------------------------------------

# The four window statistics scored against the truth, in the order they are
# written to the summary. Named once so the schema, the null path and the
# scoring loop cannot drift apart.
SENSOR_STATISTIC_NAMES = ("window_mean", "window_variance", "tke", "velmag_mean")

# Which keys of a `_zscore_block` result the statistics-space blocks re-report.
# A strict subset, and the omission is deliberate rather than cosmetic:
# `exceedance` / `overconfident` / `overconfident_rule` are calibrated in
# `_zscore_block` against POOLED INDEPENDENT knots (the measured false-alarm
# table there assumes it), and nothing pooled here is independent -- the same
# window's sensors see one realization of one flow, so the C x S x W elements
# behind these z-scores are strongly cross-correlated and the multiplier rule's
# measured operating point does not transfer. The raw moments and the
# size-aware `max_abs` reference survive that correlation (they stay unbiased;
# only their sampling error inflates), so those are what is emitted.
STAT_ZSCORE_KEYS = ("mean", "std", "max_abs", "max_abs_calibrated_median")


def _coverage_block_keys():
    """The exact key set (and order) :func:`_coverage_block` emits.

    Derived from ``COVERAGE_ALPHAS`` by the same expression the block uses, so
    adding a credible level cannot leave the degraded path emitting a stale key
    set -- the whole point of the "every key present, as ``null``" rule is that
    a consumer can index the same paths on a degraded run.
    """
    keys = []
    for alpha in COVERAGE_ALPHAS:
        key = f"alpha_{int(round(alpha * 100))}"
        keys.extend((key, f"nominal_{key}"))
    keys.append("max_nominal_alpha")
    return tuple(keys)


_COVERAGE_KEYS = _coverage_block_keys()

# An ensemble is identifiable in a statistic when the spread ACROSS members is
# large next to the sampling noise WITHIN one member's finite averaging window.
# Below this ratio the spread being scored is a property of the window length,
# not of the parameters, and every calibration number above it is measuring the
# averaging rather than the assimilation: at a ratio r the within-member
# sampling variance is 1/r**2 of the across-member variance, so r = 3 is the
# point where sampling noise stops contributing more than ~11% of the spread a
# CRPS / coverage / PIT reads. The plan writes the cut as "~< 3"; it is a screen
# on how to read the block, never a gate -- the numbers are always reported.
IDENTIFIABILITY_MIN_RATIO = 3.0

# Block count handed to `turbulence_stats.block_bootstrap_std_batch` when the
# caller does not choose one. Mirrors that function's own default so the metrics
# stage has a name for it; a caller that passes `bootstrap_blocks` overrides it.
DEFAULT_BOOTSTRAP_BLOCKS = 20

# Base entropy for the Monte-Carlo `w1_over_floor` reference
# (`ensemble_scores.wasserstein_over_floor_reference`, which synthesizes perfect
# models by block-resampling the truth). Spelled as one seed per (window,
# sensor) rather than one shared generator threaded through the loop so an
# element's reference depends on nothing but its own truth samples and the
# member count: the same sensor scores the same number whichever position it
# holds in the set and whether or not its neighbours were scorable at all.
WASSERSTEIN_REFERENCE_SEED = 0


def _finite_mean(values):
    """Mean over the finite entries of an array, ``nan`` when there are none."""
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _var_ddof1(values, axis=-1):
    """``ddof=1`` variance that returns ``nan`` -- not a warning -- at n < 2.

    Doubles as the per-member reducer handed to ``block_bootstrap_std``, which
    calls it on a bare 1-D resample, hence the ``axis=-1`` default.
    """
    array = np.asarray(values, dtype=float)
    if array.shape[axis] < 2:
        return np.full(np.delete(np.asarray(array.shape), axis), np.nan)
    return np.var(array, axis=axis, ddof=1)


def _median_max(values):
    """``{median, max}`` over the finite entries, or ``None`` when there are none."""
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return {
        "median": _finite_or_none(np.median(finite)),
        "max": _finite_or_none(finite.max()),
    }


def _null_statistic_block(reason, n_samples=0):
    """One statistic's entry with every measured leaf ``null``.

    The nesting is identical to the healthy block, not collapsed to a single
    ``null``, so a consumer indexes ``...window_mean.coverage.alpha_90`` on a
    smoke run exactly as on a production one. Only the three genuine constants
    (``pit.n_bins``, ``pit.tie_seed``, ``identifiability.threshold``) keep their
    values: they are configuration, not measurement, and nulling them would hide
    which settings the (absent) numbers would have been produced under.

    ``n_windows_scored`` is ``0`` rather than ``null``: it is a count of what
    contributed, and on this path nothing did. That is a measurement, and a
    consumer dividing by it must see the zero rather than an absent key.
    """
    return {
        "crps": None,
        "crpss_vs_prior": None,
        "zscore": dict.fromkeys(STAT_ZSCORE_KEYS),
        "coverage": dict.fromkeys(_COVERAGE_KEYS),
        "pit_counts": None,
        "pit": {
            "n_bins": PIT_BINS,
            "n_samples": int(n_samples),
            "ranks_per_bin": None,
            "tie_seed": PIT_TIE_SEED,
        },
        "identifiability": {
            "ratio_median": None,
            "ratio_min": None,
            "sampling_noise_dominated": None,
            "threshold": IDENTIFIABILITY_MIN_RATIO,
        },
        "n_samples": int(n_samples),
        "n_windows_scored": 0,
        "reason": reason,
    }


def _null_wasserstein_block(reason):
    """The Wasserstein layer with every sensor-reduced entry ``null``."""
    return {
        "w1_over_sigma_pooled": None,
        "w1_over_sigma_member_mean": None,
        "self_floor": None,
        "w1_over_floor": None,
        "w1_over_floor_calibrated_median": None,
        "reason": reason,
    }


def _ensemble_window_indices(ensemble_da, num_windows, sim_time, set_name):
    """Per-window integer indices into the ensemble series' ``time`` axis.

    The ensemble and the truth are windowed by **different rules** and this is
    the ensemble's: ``ensemble_sensor_series`` rebases window ``w`` onto
    ``[w*sim_time, (w+1)*sim_time)``, so the window is recovered from the time
    *values*, never from a frame count. The two series generally have different
    lengths and cadences (the truth is saved on its own schedule), so there is
    no shared index to slice and no attempt is made to build one -- each series
    is sliced by its own rule and reduced independently.

    The final window's upper bound is dropped: a solver that writes its
    end-of-window frame at exactly ``num_windows*sim_time`` would otherwise have
    it fall outside every half-open interval and be silently discarded.

    Falls back to contiguous equal blocks (with one log line) when the series
    carries no ``time`` coordinate -- ``ensemble_sensor_series`` only rebases
    when the window file has one -- or when ``sim_time`` is not a positive
    number, in which case the interval rule is undefined.
    """
    n_time = int(ensemble_da.sizes["time"])

    def _contiguous(why):
        logger.info(
            "Sensor set %r: %s; falling back to %d contiguous equal blocks of "
            "the %d ensemble frames for the WP1.2 window statistics",
            set_name,
            why,
            num_windows,
            n_time,
        )
        edges = np.linspace(0, n_time, num_windows + 1).round().astype(int)
        return [np.arange(edges[w], edges[w + 1]) for w in range(num_windows)]

    if "time" not in ensemble_da.coords:
        return _contiguous("the ensemble series carries no time coordinate")
    if sim_time is None or not np.isfinite(float(sim_time)) or float(sim_time) <= 0:
        return _contiguous(f"sim_time = {sim_time!r} is not a positive duration")

    times = np.asarray(ensemble_da["time"].values, dtype=float)
    span = float(sim_time)
    indices = []
    for w in range(num_windows):
        inside = times >= w * span
        if w < num_windows - 1:
            inside &= times < (w + 1) * span
        indices.append(np.flatnonzero(inside))
    return indices


def _statistic_window_series(name, ensemble_window, truth_window):
    """``(member_series, truth_series, reducer)`` for one statistic in one window.

    Every one of the four statistics is expressed as **one reducer applied to a
    1-D per-element time series**, which is what makes the identifiability
    bootstrap possible at all: ``block_bootstrap_std`` resamples blocks of a
    serially correlated series and re-applies the statistic, so the statistic
    has to be a function of a resampleable series rather than of the raw
    ``(component, time)`` slab.

      * ``window_mean`` / ``window_variance`` -- the component series itself,
        reduced by ``mean`` / ``_var_ddof1``, pooled over component x sensor.
      * ``tke`` -- the *instantaneous* fluctuation energy
        ``0.5 * n/(n-1) * sum_c (u_c(t) - <u_c>)**2``, reduced by ``mean``.
        The ``n/(n-1)`` factor is not cosmetic: it makes the time-mean of this
        series **exactly** ``0.5 * sum_c var_ddof1(u_c)``, i.e. the tabulated
        TKE, so the bootstrap is resampling the same estimator that is reported
        rather than a ``ddof=0`` cousin of it. The component means are held at
        their full-window values while the series is resampled, which is the
        usual convention -- the fluctuation field is a property of the window,
        not of the resample.
      * ``velmag_mean`` -- ``|U|(t)``, reduced by ``mean``, pooled over sensor.

    Args:
        ensemble_window: ``(component, ensemble, time, sensor)`` window slab.
        truth_window: ``(component, time, sensor)`` window slab.

    Returns:
        ``member_series`` ``(n_members, n_elements, n_time)``, ``truth_series``
        ``(n_elements, n_time_truth)`` and the shared reducer. ``n_elements`` is
        ``n_components * n_sensors`` for the two per-component statistics and
        ``n_sensors`` for the two pooled ones; the element order is row-major
        ``(component, sensor)`` on both sides so the two line up.
    """
    if name in ("window_mean", "window_variance"):
        n_components, n_members, n_time, n_sensors = ensemble_window.shape
        member = np.transpose(ensemble_window, (1, 0, 3, 2)).reshape(
            n_members, n_components * n_sensors, n_time
        )
        truth = np.transpose(truth_window, (0, 2, 1)).reshape(
            n_components * n_sensors, truth_window.shape[1]
        )
        reducer = np.mean if name == "window_mean" else _var_ddof1
        return member, truth, reducer

    if name == "tke":
        member = np.transpose(_fluctuation_energy(ensemble_window, axis=2), (0, 2, 1))
        truth = _fluctuation_energy(truth_window, axis=1).T
        return member, truth, np.mean

    if name == "velmag_mean":
        member = np.transpose(np.sqrt((ensemble_window**2).sum(axis=0)), (0, 2, 1))
        truth = np.sqrt((truth_window**2).sum(axis=0)).T
        return member, truth, np.mean

    raise ValueError(f"unknown sensor statistic {name!r}")


def _fluctuation_energy(slab, axis):
    """``0.5 * n/(n-1) * sum_c (u_c - <u_c>)**2``, the per-timestep TKE integrand.

    ``slab`` has ``component`` first and its time axis at ``axis``; the
    component axis is summed out, so the result drops it. See
    :func:`_statistic_window_series` for why the Bessel factor is applied to the
    integrand rather than to the reduced value.
    """
    n_time = slab.shape[axis]
    scale = n_time / (n_time - 1) if n_time > 1 else np.nan
    fluctuation = slab - slab.mean(axis=axis, keepdims=True)
    return 0.5 * scale * (fluctuation**2).sum(axis=0)


def _within_member_std(member_series, reducer, bootstrap_blocks):
    """Block-bootstrap sampling std of ``reducer`` for every (member, element).

    The blocking is the point: a window's frames are serially correlated, so the
    iid ``std/sqrt(n)`` would understate the sampling noise of a window mean and
    the identifiability ratio would come out flatteringly large.

    One **batched** call over the flattened ``(member x element, time)`` rows.
    The per-row Python loop this replaces ran one ``block_bootstrap_std`` per
    (member, element) -- 6144 calls at the shipped shape (M = 32, C = 3, S = 8,
    W = 3), measured at 3.7 s, and growing with M x S x W to minutes at
    production shapes for a number that is a footnote in the summary.
    ``block_bootstrap_std_batch`` builds one resample index matrix and applies
    it to every row, and is exact against the scalar function on the same seed,
    so this is a speed change and not a numerical one.

    Both reducers are handed over as axis-taking callables, which is the batch
    function's contract (``statistic(x, axis=-1)`` on a 3-D array): ``np.mean``
    already is one and ``_var_ddof1`` was written as one for this reason.

    Two ``nan`` paths, both left for the caller rather than repaired here. A row
    shorter than the blocking can resolve -- the smoke shape, 3 frames against
    20 blocks -- makes *every* entry ``nan``, which is the degenerate case the
    caller reports as a ``null`` identifiability block. A row holding any
    non-finite sample is ``nan`` too: the batch cannot drop gaps per row the way
    the scalar function does, so a masked sensor loses its identifiability
    element instead of having it estimated from a shortened series. That is the
    conservative direction (a dropped element, not a silently different sample
    count) and the elements are dropped by the caller's finiteness filter
    anyway.
    """
    n_members, n_elements, n_time = member_series.shape
    flat = np.asarray(member_series, dtype=float).reshape(
        n_members * n_elements, n_time
    )
    within = block_bootstrap_std_batch(
        flat, statistic=reducer, n_blocks=bootstrap_blocks
    )
    return np.asarray(within, dtype=float).reshape(n_members, n_elements)


def _pooled_statistic(
    name,
    ensemble,
    truth,
    ensemble_windows,
    truth_windows,
    bootstrap_blocks,
    with_bootstrap=True,
    windows=None,
):
    """Pool one statistic over windows into ``(members, truth, within_std, cols)``.

    Windows with no ensemble frames or no truth frames are dropped rather than
    contributing an empty reduction; a run where that leaves nothing returns
    ``(None, None, None, None)`` and the caller nulls the block.

    ``cols`` is the window index each pooled COLUMN came from, which is what
    makes both of the drop paths visible downstream. Two callers need it and
    neither can reconstruct it: ``_statistic_block`` reports
    ``n_windows_scored`` (a window can also vanish later, when every one of its
    elements fails the finiteness filter -- a single-frame window has no ddof=1
    variance), and the CRPSS path has to know which windows a pool actually
    holds, because the posterior and the prior drop windows INDEPENDENTLY and an
    equal column count is not evidence that they dropped the same ones.

    Args:
        windows: Optional explicit window indices to pool, in order. ``None``
            pools every window. Used to rebuild a pool on the subset the
            posterior and the prior share -- see :func:`sensor_statistic_scores`.
        with_bootstrap: ``False`` skips the (M x N) bootstrap entirely -- used
            for the prior ensemble, whose only contribution is a CRPS reference.
    """
    selected = range(len(ensemble_windows)) if windows is None else list(windows)
    member_parts, truth_parts, within_parts, column_windows = [], [], [], []
    for window in selected:
        indices = np.asarray(ensemble_windows[window])
        if indices.size == 0:
            continue
        ensemble_window = ensemble[:, :, indices, :]
        truth_window = truth[:, truth_windows[window], :]
        if truth_window.shape[1] == 0:
            continue
        member_series, truth_series, reducer = _statistic_window_series(
            name, ensemble_window, truth_window
        )
        member_parts.append(np.asarray(reducer(member_series, axis=-1), dtype=float))
        truth_parts.append(np.asarray(reducer(truth_series, axis=-1), dtype=float))
        within_parts.append(
            _within_member_std(member_series, reducer, bootstrap_blocks)
            if with_bootstrap
            else np.full(member_parts[-1].shape, np.nan)
        )
        column_windows.append(np.full(member_parts[-1].shape[1], int(window)))
    if not member_parts:
        return None, None, None, None
    return (
        np.concatenate(member_parts, axis=1),
        np.concatenate(truth_parts, axis=0),
        np.concatenate(within_parts, axis=1),
        np.concatenate(column_windows),
    )


def _pooled_windows(column_windows):
    """The ordered, de-duplicated window indices behind a pool's columns."""
    if column_windows is None:
        return []
    seen = []
    for window in np.asarray(column_windows).tolist():
        if window not in seen:
            seen.append(int(window))
    return seen


def _column_medians(values):
    """Per-column median over the finite entries, ``nan`` for an all-nan column.

    ``np.nanmedian`` would do this but warns on an all-nan slice, and an all-nan
    column is the *expected* smoke-scale outcome here, not an anomaly.
    """
    out = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        if finite.size:
            out[column] = np.median(finite)
    return out


def _identifiability_block(members, within_std, name, set_name):
    """Across-member spread measured in units of within-member sampling noise.

    ``ratio = std_across_members / median_over_members(within-member bootstrap
    std)``, per pooled element, then reduced over elements by median and min.
    A ratio near 1 means the members differ by no more than one member's own
    finite-window sampling noise, so the CRPS / coverage / PIT sitting next to
    it are scoring the averaging window rather than the assimilation.

    Reported, never enforced: ``sampling_noise_dominated`` is a flag on the
    median ratio and the numbers are emitted whether or not it fires.

    Degrades to all-``null`` (the flag included, since an unknown ratio is not
    a "not dominated") when the bootstrap could not run -- the smoke
    shape puts 3 frames in a window against 20 requested blocks, which is below
    ``block_bootstrap_std``'s resolution and yields ``nan`` everywhere.
    """
    block = {
        "ratio_median": None,
        "ratio_min": None,
        "sampling_noise_dominated": None,
        "threshold": IDENTIFIABILITY_MIN_RATIO,
    }
    if within_std is None or not np.any(np.isfinite(within_std)):
        logger.info(
            "Sensor set %r, statistic %r: the within-member block bootstrap "
            "returned no finite sampling std (too few frames per window for the "
            "requested block count); reporting identifiability as null and "
            "leaving the calibration numbers as they are",
            set_name,
            name,
        )
        return block

    across = np.asarray(members, dtype=float).std(axis=0, ddof=1)
    within = _column_medians(np.asarray(within_std, dtype=float))
    ratio = np.full(across.shape, np.nan)
    usable = np.isfinite(across) & np.isfinite(within) & (within > 0)
    np.divide(across, within, out=ratio, where=usable)
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return block

    median = float(np.median(finite))
    block["ratio_median"] = _finite_or_none(median)
    block["ratio_min"] = _finite_or_none(finite.min())
    block["sampling_noise_dominated"] = bool(median < IDENTIFIABILITY_MIN_RATIO)
    return block


def _pool_crps(members, truth, valid):
    """Mean fair CRPS over the elements ``valid`` selects."""
    return _finite_mean(
        fair_crps(
            np.asarray(members, dtype=float)[:, valid],
            np.asarray(truth, dtype=float)[valid],
        )
    )


def _crpss_from_pools(posterior, prior, name, set_name):
    """CRPSS of one pooled statistic against a prior pool of the SAME elements.

    Both pools must already be built on the same window list -- the caller
    guarantees that (see :func:`sensor_statistic_scores`), and the shape guard
    here is only a backstop for a pooling bug, not the alignment mechanism.
    Element selection is a JOINT filter: an element is scored only when the
    truth, every posterior member and every prior member are finite there, so
    the two CRPS values in the ratio are means over exactly the same sample. A
    per-pool filter would divide a mean over one element set by a mean over
    another and call the difference skill.

    Returns ``None`` (with one ``logger.info``) when nothing survives.
    """
    post_members, post_truth = posterior
    prior_members, prior_truth = prior
    if post_members is None or prior_members is None:
        return None
    if prior_members.shape != post_members.shape:
        logger.info(
            "Sensor set %r: the prior %s pool is %s against the posterior's %s; "
            "reporting crpss_vs_prior as null",
            set_name,
            name,
            prior_members.shape,
            post_members.shape,
        )
        return None

    valid = (
        np.isfinite(post_truth)
        & np.all(np.isfinite(post_members), axis=0)
        & np.isfinite(prior_truth)
        & np.all(np.isfinite(prior_members), axis=0)
    )
    if not np.any(valid):
        logger.info(
            "Sensor set %r, statistic %r: no element is finite in both the "
            "posterior and the prior pool; reporting crpss_vs_prior as null",
            set_name,
            name,
        )
        return None
    return _finite_or_none(
        crpss(
            _pool_crps(post_members, post_truth, valid),
            _pool_crps(prior_members, prior_truth, valid),
        )
    )


def _aligned_crpss(
    name,
    posterior_pool,
    posterior_source,
    prior_source,
    truth,
    truth_windows,
    bootstrap_blocks,
    set_name,
):
    """``crpss_vs_prior`` for one statistic, on the windows BOTH pools resolve.

    The posterior and the prior are windowed independently -- they are different
    ensembles with different time axes, and ``_pooled_statistic`` drops a window
    from whichever pool lacks frames for it. Two pools can therefore hold the
    same NUMBER of columns while holding different windows, and an equal-shape
    check then compares window 0's posterior against window 1's prior and
    reports the mismatch as skill. Measured on identical member values with the
    prior shifted by one window, that produced a ``crpss_vs_prior`` of -0.26
    where the honest answer is exactly 0.

    So the window lists are intersected explicitly:

      * identical lists -- score the pools as they are;
      * a strict subset -- rebuild BOTH pools on the intersection and score
        those, so the metric survives a partly-missing prior instead of
        vanishing;
      * an empty intersection -- ``None`` plus one ``logger.info``.

    Only ``crpss_vs_prior`` is affected. The posterior's own keys keep their
    full window list: a short prior must never degrade the posterior's numbers,
    which is why this returns a scalar instead of re-pooling in place.
    """
    members, statistic_truth, column_windows = posterior_pool
    ensemble, ensemble_windows = posterior_source
    prior, prior_windows = prior_source

    prior_members, prior_truth, _, prior_columns = _pooled_statistic(
        name,
        prior,
        truth,
        prior_windows,
        truth_windows,
        bootstrap_blocks,
        with_bootstrap=False,
    )
    if prior_members is None:
        logger.info(
            "Sensor set %r, statistic %r: no window has both prior and truth "
            "frames; reporting crpss_vs_prior as null",
            set_name,
            name,
        )
        return None

    used = _pooled_windows(column_windows)
    prior_used = _pooled_windows(prior_columns)
    shared = [window for window in used if window in prior_used]
    if not shared:
        logger.info(
            "Sensor set %r, statistic %r: the posterior pooled windows %s and "
            "the prior pooled %s, which share none; reporting crpss_vs_prior "
            "as null",
            set_name,
            name,
            used,
            prior_used,
        )
        return None
    if shared == used == prior_used:
        return _crpss_from_pools(
            (members, statistic_truth), (prior_members, prior_truth), name, set_name
        )

    logger.info(
        "Sensor set %r, statistic %r: the posterior pooled windows %s and the "
        "prior pooled %s; rebuilding both pools on the shared %s so "
        "crpss_vs_prior compares like with like (the posterior's own keys keep "
        "the full window list)",
        set_name,
        name,
        used,
        prior_used,
        shared,
    )
    posterior_shared = _pooled_statistic(
        name,
        ensemble,
        truth,
        ensemble_windows,
        truth_windows,
        bootstrap_blocks,
        with_bootstrap=False,
        windows=shared,
    )
    prior_shared = _pooled_statistic(
        name,
        prior,
        truth,
        prior_windows,
        truth_windows,
        bootstrap_blocks,
        with_bootstrap=False,
        windows=shared,
    )
    return _crpss_from_pools(posterior_shared[:2], prior_shared[:2], name, set_name)


def _statistic_block(
    name, members, truth, within_std, column_windows, n_windows, skill, set_name
):
    """Score one pooled statistic: CRPS, z-score, coverage, PIT, identifiability.

    The pooled elements are the columns of ``members``; the ensemble axis is
    axis 0, which is the convention every ``ensemble_scores`` function takes.
    Elements whose truth or members are non-finite are dropped once, here, so
    each score sees the same sample.

    ``skill`` is ``crpss_vs_prior``, computed by the caller because it needs the
    prior pool rebuilt on the shared windows; it is passed in rather than
    derived here so this function scores exactly one pool.

    ``n_windows_scored`` counts the windows that survive to the reported
    numbers, which is NOT the configured ``n_windows``: a window is lost either
    by contributing no columns (no ensemble or no truth frame) or by having
    every one of its columns fail the finiteness filter (a single-frame window
    has no ddof=1 variance, so it kills ``window_variance`` and ``tke`` while
    ``window_mean`` survives -- which is why this count is per statistic and not
    per sensor set).
    """
    if members is None:
        reason = (
            "no assimilation window has both ensemble and truth frames, so the "
            f"{name} statistic has nothing to pool"
        )
        logger.info("Sensor set %r: %s; emitting nulls", set_name, reason)
        return _null_statistic_block(reason)

    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
    if not np.any(valid):
        reason = (
            f"every pooled {name} element is non-finite (a window with a single "
            "frame has no ddof=1 variance, and a masked sensor has no series)"
        )
        logger.info("Sensor set %r: %s; emitting nulls", set_name, reason)
        return _null_statistic_block(reason)

    scored = np.asarray(members, dtype=float)[:, valid]
    scored_truth = np.asarray(truth, dtype=float)[valid]
    n_members = int(scored.shape[0])

    scored_windows = _pooled_windows(np.asarray(column_windows)[valid])
    dropped = [w for w in range(int(n_windows)) if w not in scored_windows]
    if dropped:
        logger.info(
            "Sensor set %r, statistic %r: windows %s contribute no finite "
            "element (no ensemble or truth frame in the window, or too few "
            "frames for this statistic); scoring %d of the %d configured "
            "windows -- n_windows_scored records this",
            set_name,
            name,
            dropped,
            len(scored_windows),
            int(n_windows),
        )

    crps_mean = _finite_mean(fair_crps(scored, scored_truth))
    z_block = _zscore_block(scored, scored_truth)
    ranks = pit_rank(scored, scored_truth, rng=PIT_TIE_SEED)
    counts = rank_histogram(ranks, n_members, n_bins=PIT_BINS)

    return {
        "crps": _finite_or_none(crps_mean),
        "crpss_vs_prior": None if skill is None else _finite_or_none(skill),
        "zscore": (
            dict.fromkeys(STAT_ZSCORE_KEYS)
            if z_block is None
            else {key: z_block[key] for key in STAT_ZSCORE_KEYS}
        ),
        "coverage": _coverage_block(scored, scored_truth)
        or dict.fromkeys(_COVERAGE_KEYS),
        "pit_counts": [int(c) for c in counts],
        "pit": {
            "n_bins": PIT_BINS,
            "n_samples": int(valid.sum()),
            # Ranks take M + 1 values, which divides evenly into PIT_BINS only
            # for particular M, so a calibrated ensemble is flat in
            # `count / ranks_per_bin`, NOT in `count`. Same reference the WP1.1
            # bundle carries, for the same reason.
            "ranks_per_bin": [
                int(w) for w in rank_histogram_weights(n_members, n_bins=PIT_BINS)
            ],
            "tie_seed": PIT_TIE_SEED,
        },
        "identifiability": _identifiability_block(
            scored,
            None if within_std is None else np.asarray(within_std)[:, valid],
            name,
            set_name,
        ),
        "n_samples": int(valid.sum()),
        "n_windows_scored": len(scored_windows),
        "reason": None,
    }


def _wasserstein_block(
    ensemble_da, truth_da, ensemble_windows, truth_windows, set_name
):
    """Distribution distance between modelled and observed ``|U|``, per sensor-window.

    The layer the window statistics cannot provide: a window mean can match
    while the *distribution* of the probe signal is wrong, and it is the
    distribution the plan's probe figures compare. Scored on ``|U|`` -- positive
    and directly the probe quantity -- **within each assimilation window**, over
    the same window slices the four statistics use (the ensemble by time value,
    the truth by frame index), then reduced over the ``S x W`` sensor-window
    elements.

    Per window rather than over the whole run because of ``self_floor``. That
    floor splits the truth series in two contiguous halves, so it absorbs
    everything that makes the two halves differ -- including any *deterministic*
    drift across the split, which is not sampling variability at all. Pooling
    the whole run puts the run's slowest forcing inside the split: on the
    shipped default truth (``params@truth_params: dynamic_cosine``, a 400 s
    inflow-angle cosine and a 200 s magnitude cosine) the measured floor rises
    from 0.168 on a stationary series to 0.575, and a clearly-wrong model (+20%
    ``|U|``, half the turbulence) falls from ``w1_over_floor`` 18.2 to 1.9 --
    within a factor of ~3 of what a perfect model scores. Windowing removes the
    cross-window part of that; a trend whose period is comparable to ONE window
    still inflates the floor, and no reduction here can fix that.

    Five numbers per (sensor, window), then reduced by median and max:

      * ``w1_over_sigma_pooled`` -- every member's samples pooled into one
        empirical distribution against the truth's. This is the ensemble's
        *predictive* distribution.
      * ``w1_over_sigma_member_mean`` -- each member scored on its own, then
        averaged: the typical single-member error.

    The pair is emitted because of the inequality between them, which holds
    always and in one direction: ``W1`` is convex in its first argument, so
    ``W1(pooled, truth) <= mean_m W1(member_m, truth)``. They therefore
    **coincide** when every member is wrong the same way (a shared bias, which
    pooling cannot cancel) and **separate** when the members straddle the truth
    -- spread that brackets the observed distribution. A single number cannot
    tell those apart: the pooled distance alone credits a wide ensemble for
    covering the truth, the member mean alone refuses to.
      * ``self_floor`` -- what a *perfect* model scores at this sample count and
        autocorrelation (``wasserstein_self_floor``: contiguous halves of the
        window's truth against each other). Without it the raw distance is
        unreadable.
      * ``w1_over_floor`` -- each element's OWN distance over its OWN floor,
        reduced afterwards. Deliberately not
        ``w1_over_sigma_pooled.median / self_floor.median``: the floor is a
        property of one sensor in one window (its sample count and its
        autocorrelation), so dividing a median by a median would read one
        sensor's distance against another's floor.
      * ``w1_over_floor_calibrated_median`` -- what a PERFECT model of this
        element's truth would score on ``w1_over_floor``
        (:func:`ensemble_scores.wasserstein_over_floor_reference`, evaluated on
        the very same truth samples and the same member count, so it is
        comparable to ``w1_over_floor`` element for element and under the same
        median/max reduction).

    That reference is not decoration: ``w1_over_floor`` is **not** calibrated at
    1. The floor compares ``n/2`` truth samples against ``n/2`` while the
    numerator compares ``M*n`` model samples against ``n``, so the two are still
    not like-for-like on sample count. A perfect model measures ~0.55 and is
    *flat* in ``n`` (numerator and floor both shrink as ``1/sqrt(n)``), whereas a
    genuine bias has an ``n``-independent numerator and so grows as ``sqrt(n)``.
    Two consequences a reader has to have: a model reading exactly 1.0 is already
    ~2x worse than perfect, and at a short window a real error can read *below*
    1. Read the ratio against this key, never against 1. The reference is a
    Monte-Carlo median, seeded per (window, sensor) from
    ``WASSERSTEIN_REFERENCE_SEED`` so a re-run reproduces it; its resampling
    block count is the estimator's own default and is deliberately NOT tied to
    this stage's ``bootstrap_blocks``, which callers set to defeat the
    identifiability bootstrap and which would silently strip the synthetic
    model's autocorrelation here.

    One reading caveat, and it applies to the shipped default truth: the same
    within-window trend that inflates ``self_floor`` does *not* survive the
    reference's block resampling, so on a forced window the ratio is pushed down
    while its reference is not. Measured on a cosine of period ~2x the window, a
    perfect model reads ~0.12 against a reference of ~0.51 while a clearly-wrong
    one still reads ~1.66, i.e. 3.3x the reference. The comparison keeps its
    sign and still separates bad from good; it is the equality "perfect scores
    its reference" that holds only on a stationary window.

    Elements whose distance or floor is undefined drop out of the reduction --
    the four-finite-sample minimum of ``wasserstein_self_floor`` is routine at
    smoke scale, where a window holds 3 frames and ``self_floor`` is ``null``
    while the two distances survive.

    ``reason`` is non-null whenever any of the five reduces to ``null``, naming
    which -- a constant truth series has no ``sigma`` to divide by and no floor,
    and that is a property of the sensor, not a bug to hide.
    """
    ensemble_magnitude = np.asarray(
        sensor_magnitude(
            ensemble_da.transpose("component", "ensemble", "time", "sensor")
        ).values,
        dtype=float,
    )  # (ensemble, time, sensor)
    truth_magnitude = np.asarray(
        sensor_magnitude(truth_da.transpose("component", "time", "sensor")).values,
        dtype=float,
    )  # (time, sensor)

    n_members, _, n_sensors = ensemble_magnitude.shape
    n_windows = len(ensemble_windows)
    if n_sensors == 0 or truth_magnitude.shape[0] == 0:
        reason = "the sensor set has no sensor or no truth frame"
        logger.info("Sensor set %r: %s; emitting Wasserstein nulls", set_name, reason)
        return _null_wasserstein_block(reason)

    shape = (n_windows, n_sensors)
    pooled = np.full(shape, np.nan)
    member_mean = np.full(shape, np.nan)
    floor = np.full(shape, np.nan)
    over_floor = np.full(shape, np.nan)
    calibrated = np.full(shape, np.nan)
    for window in range(n_windows):
        indices = np.asarray(ensemble_windows[window])
        window_truth = truth_magnitude[truth_windows[window], :]
        if indices.size == 0 or window_truth.shape[0] == 0:
            continue
        window_members = ensemble_magnitude[:, indices, :]
        for sensor in range(n_sensors):
            truth_samples = window_truth[:, sensor]
            pooled[window, sensor] = wasserstein_over_sigma(
                window_members[:, :, sensor].ravel(), truth_samples
            )
            per_member = np.array(
                [
                    wasserstein_over_sigma(window_members[m, :, sensor], truth_samples)
                    for m in range(n_members)
                ]
            )
            member_mean[window, sensor] = _finite_mean(per_member)
            floor[window, sensor] = wasserstein_self_floor(truth_samples)
            over_floor[window, sensor] = wasserstein_over_floor(
                pooled[window, sensor], floor[window, sensor]
            )
            # Same `truth_samples` the floor and the distances were taken on --
            # the window slices are the caller's and are never re-derived here.
            calibrated[window, sensor] = wasserstein_over_floor_reference(
                truth_samples,
                n_members,
                rng=np.random.default_rng([WASSERSTEIN_REFERENCE_SEED, window, sensor]),
            )

    block = {
        "w1_over_sigma_pooled": _median_max(pooled),
        "w1_over_sigma_member_mean": _median_max(member_mean),
        "self_floor": _median_max(floor),
        "w1_over_floor": _median_max(over_floor),
        "w1_over_floor_calibrated_median": _median_max(calibrated),
    }
    missing = sorted(key for key, value in block.items() if value is None)
    reason = None
    if missing:
        reason = (
            f"no (sensor, window) element has a finite {', '.join(missing)} (a "
            "window truth series with fewer than two finite samples or zero "
            "spread has no scale to normalize by, and the self floor -- with "
            "the calibrated reference that divides by it -- needs four)"
        )
        logger.info(
            "Sensor set %r: %s; reporting those entries as null and keeping the rest",
            set_name,
            reason,
        )
    block["reason"] = reason
    return block


def sensor_statistic_scores(
    truth_series,
    ensemble_series,
    *,
    num_windows,
    n_per_window,
    sim_time,
    bootstrap_blocks=DEFAULT_BOOTSTRAP_BLOCKS,
    prior_series=None,
):
    """Score the ensemble in STATISTICS space rather than pointwise (WP1.2).

    The pointwise sensor metrics (:func:`vector_sensor_metrics`) ask whether the
    ensemble reproduces the truth *at each instant*, which a turbulent flow
    cannot be expected to do: two realizations of the same statistics decorrelate
    within a few eddy turnover times and the pointwise score then measures phase,
    not physics. This function reduces each assimilation window to statistics
    that ARE reproducible -- window mean, window variance, TKE, mean ``|U|`` --
    scores those against the truth's own window statistics, and adds a
    distribution-distance layer on the raw ``|U|`` samples.

    Windowing is the trap. The two series are sliced by **different rules**,
    because they are produced by different code paths:

      * the ensemble by **time value** -- ``ensemble_sensor_series`` rebases
        window ``w`` onto ``[w*sim_time, (w+1)*sim_time)``;
      * the truth by **index** -- ``truth_sensor_series`` concatenates
        ``slice(w*n_per_window, (w+1)*n_per_window)`` of a globally-timed series.

    They generally differ in both length and cadence, so nothing here aligns
    them pointwise; each is sliced by its own rule and reduced independently,
    which is exactly what makes a statistics-space comparison possible in the
    first place.

    **What ``crpss_vs_prior`` measures.** ``window_{w}_prior_state.nc`` is window
    ``w``'s ESMDA step-0 forecast, and ``run_esmda.py`` seeds window ``w``'s
    prior from window ``w-1``'s posterior. Pooling every window therefore makes
    this **one-window-ahead** skill -- how much the update improved on the state
    it started that window from -- and NOT skill against the run's initial
    prior. Those are different quantities, and the same ``run_summary.yaml``
    carries the other one: WP1.1's parameter bundle splits exactly this
    distinction into ``crpss.vs_window_prior`` and ``crpss.vs_initial_prior``
    (``compute_esmda_metrics.py:376-385``). Read the sensor number as the
    former; there is no statistics-space analogue of the latter, because the
    initial prior's sensor series is not saved.

    Does no file IO: the caller passes the series it already extracted. That
    keeps WP1.3's streaming refactor of the extraction from touching this
    function at all.

    Degradation follows the section's rule -- every key present with a ``null``
    value plus a non-null ``reason``, one ``logger.info``, never a warning. The
    paths that fire in practice are ``M < MIN_MEMBERS_CALIBRATION`` (the
    two-member smoke shape, which nulls the whole set) and the identifiability
    bootstrap at a window length below its block resolution (3 frames against
    20 blocks), which nulls only that sub-block.

    Args:
        truth_series: ``{set_name: DataArray(component, time, sensor)}`` from
            :func:`truth_sensor_series`.
        ensemble_series: ``{set_name: DataArray(component, ensemble, time,
            sensor)}`` from :func:`ensemble_sensor_series`.
        num_windows: Assimilation window count.
        n_per_window: Truth frames per window -- the truth's slicing rule.
        sim_time: Seconds per window -- the ensemble's slicing rule.
        bootstrap_blocks: Block count for the identifiability bootstrap
            (:func:`turbulence_stats.block_bootstrap_std_batch`).
        prior_series: The same mapping for the prior ensemble, or ``None``.
            ``None`` is the common case (``run.save_prior_state`` defaults to
            false), and it makes ``crpss_vs_prior`` ``null`` -- not an error.

    Returns:
        ``{set_name: {n_members, n_windows, n_sensors, window_mean,
        window_variance, tke, velmag_mean, wasserstein}}``, ready to assign to
        ``summary["sensor_statistics"]``.
    """
    scores = {}
    windows = 0 if num_windows is None else int(num_windows)
    frames_per_window = 0 if n_per_window is None else int(n_per_window)

    for set_name, ensemble_da in ensemble_series.items():
        if set_name not in truth_series:
            logger.info(
                "Sensor set %r has an ensemble series but no truth series; "
                "skipping it in the WP1.2 statistics",
                set_name,
            )
            continue
        ensemble_da = ensemble_da.transpose("component", "ensemble", "time", "sensor")
        truth_da = truth_series[set_name].transpose("component", "time", "sensor")
        n_members = int(ensemble_da.sizes["ensemble"])
        entry = {
            "n_members": n_members,
            "n_windows": windows,
            "n_sensors": int(ensemble_da.sizes["sensor"]),
        }

        # Same floor, and the same reasoning, as the WP1.1 parameter bundle: a
        # ddof=1 spread with one degree of freedom, an order-statistic band that
        # cannot reach 90%, and a 10-bin PIT over M + 1 = 3 rank values are all
        # measuring the ensemble SIZE. Nulling the Wasserstein layer alongside
        # them is the contract's call rather than a mathematical necessity --
        # W1 is defined at M = 2 -- and is noted as such.
        if n_members < MIN_MEMBERS_CALIBRATION or windows < 1:
            if n_members < MIN_MEMBERS_CALIBRATION:
                reason = (
                    f"{n_members} members is below the "
                    f"{MIN_MEMBERS_CALIBRATION} needed for a spread, an "
                    f"order-statistic band and a {PIT_BINS}-bin PIT to mean "
                    "anything"
                )
            else:
                reason = f"num_windows = {num_windows!r} is not a window count"
            logger.info(
                "Sensor set %r: %s; emitting nulls for every WP1.2 statistic "
                "and for the Wasserstein layer",
                set_name,
                reason,
            )
            for name in SENSOR_STATISTIC_NAMES:
                entry[name] = _null_statistic_block(reason)
            entry["wasserstein"] = _null_wasserstein_block(reason)
            scores[set_name] = entry
            continue

        ensemble = np.asarray(ensemble_da.values, dtype=float)
        truth = np.asarray(truth_da.values, dtype=float)
        ensemble_windows = _ensemble_window_indices(
            ensemble_da, windows, sim_time, set_name
        )
        truth_windows = [
            slice(w * frames_per_window, (w + 1) * frames_per_window)
            for w in range(windows)
        ]

        prior_da = None if prior_series is None else prior_series.get(set_name)
        prior, prior_windows = None, None
        if prior_da is not None:
            prior_da = prior_da.transpose("component", "ensemble", "time", "sensor")
            prior = np.asarray(prior_da.values, dtype=float)
            prior_windows = _ensemble_window_indices(
                prior_da, windows, sim_time, set_name
            )

        for name in SENSOR_STATISTIC_NAMES:
            members, statistic_truth, within_std, column_windows = _pooled_statistic(
                name,
                ensemble,
                truth,
                ensemble_windows,
                truth_windows,
                bootstrap_blocks,
            )
            skill = None
            if members is not None and prior is not None:
                skill = _aligned_crpss(
                    name,
                    (members, statistic_truth, column_windows),
                    (ensemble, ensemble_windows),
                    (prior, prior_windows),
                    truth,
                    truth_windows,
                    bootstrap_blocks,
                    set_name,
                )
            entry[name] = _statistic_block(
                name,
                members,
                statistic_truth,
                within_std,
                column_windows,
                windows,
                skill,
                set_name,
            )

        entry["wasserstein"] = _wasserstein_block(
            ensemble_da, truth_da, ensemble_windows, truth_windows, set_name
        )
        scores[set_name] = entry

    return scores


# ---------------------------------------------------------------------------
# Mean-field / Reynolds-stress layer (WP1.3)
# ---------------------------------------------------------------------------

# Cell-centre axis names, in the order `turbulence_stats.colocate_components`
# leaves them: pylbm/PALM write `z/y/x`, uDALES `zt/yt/xt` (its staggered
# `xm/ym/zm` are renamed onto the centre axes on the way through). Read off the
# colocated array rather than passed in, so this layer cannot disagree with
# `turbulence_stats._STAGGERED_TO_CENTRE` about where the data landed.
_CENTRE_Z_DIMS = ("z", "zt")
_CENTRE_Y_DIMS = ("y", "yt")
_CENTRE_X_DIMS = ("x", "xt")

# The velocity triple, in the order every quantity below indexes it.
MEAN_FIELD_COMPONENTS = ("u", "v", "w")

# Per region, the per-member quantities the reduction actually forms. They are
# deliberately different: the z-slabs are what the summary scores (mean field,
# TKE, <u'w'>) and the station columns are what WP1.4's S1 profile draws (the
# component means and their RMS fluctuation). Each entry costs one region-sized
# array PER MEMBER, so a quantity nobody consumes is not a stray dict key, it
# is work at member scale.
_SLAB_QUANTITIES = ("mean", "velmag", "tke", "uw")
_STATION_QUANTITIES = ("mean", "rms", "tke", "uw")

# Quantities carried for their across-member SPREAD only. The mean of the
# members' speeds is not a quantity this layer reports -- the scored magnitude
# is |ensemble-mean vector| (`velmag_of_mean`), which is a different number --
# but `ensemble_velmag_std` needs each member's own speed to spread over.
_SPREAD_ONLY_QUANTITIES = ("velmag",)

# The across-member quantiles persisted per station in `eval_fields.nc`. Enough
# for a 5-95 fan with a 25-75 box and the median, which is what an ensemble
# profile panel wants; mean +- k*sigma is not that band unless the ensemble is
# Gaussian, and the alternative to persisting them is re-reading the window
# state files in the figure stage, which is exactly what this file prevents.
STATION_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# The one solid-cell mask any shipped backend actually writes. pylbm emits it
# (1 = solid) into both its truth state and its window state files; pypalm
# `fillna(0.0)`s PALM's NaN before returning state, so a PALM mask is not
# recoverable from the artifact; uDALES fielddumps carry small non-zero junk
# inside buildings and no mask at all. Measured NaN fraction of `u`/`v`/`w` in
# every state artifact under `.temp`: 0.0 -- i.e. the plan's "fluid cells only
# (`~np.isnan` after masking)" premise does not hold and this variable is the
# whole of the mask story. See :func:`_fluid_indicator`.
BLANKING_VAR = "blanking"

# Byte ceiling on one truth time chunk's colocated components. The truth is
# opened lazily and read chunk-wise because a Barcelona-scale truth is tens of
# GB; `StreamingMoments` makes the chunk length a free parameter (the
# accumulators are per cell, not per frame), so this is a pure memory knob and
# moves no number -- pinned by `tests/test_turbulence_stats.py`'s chunking
# tests on the class itself.
#
# It bounds the chunk, NOT the layer's floor, and the difference matters at
# production scale: `colocate_components` refuses to run on an axis-subset
# dataset (a pre-colocation z selection is exactly what its face/centre check
# catches), so every chunk is colocated at the truth's **full 3-D** resolution
# before the z-slabs are taken. One frame is therefore the smallest chunk
# possible, and at a 512 x 512 x 128 truth that single frame is ~0.8 GB for the
# velocity triple plus the interpolation's own temporaries. That is the same
# order as the member `stream_window_members` already materialises, so it is
# the accepted shape of this stage rather than a regression -- but a truth
# grown past it needs colocation to gain a level-wise mode, not a smaller
# number here.
_TRUTH_CHUNK_BYTES = 64 << 20

# Byte ceiling on the retained per-cell truth *series* -- the one array in this
# layer whose size grows with the number of frames, because a moving-block
# bootstrap needs the series and not a moment of it. Above the ceiling the
# bootstrap cells are subsampled (never the frames: the blocking is what the
# estimator is about). The consequence is only Monte-Carlo noise on the
# reductions below, since every bootstrap output here is reduced over cells to
# a single scalar anyway.
_TRUTH_SERIES_MAX_BYTES = 128 << 20

# Cap on the number of truth cells the moving-block bootstrap is run over, and
# the ONLY bound on this layer's dominant cost. `_TRUTH_SERIES_MAX_BYTES`
# bounds *memory*, which at 4 x 128 x 128 x 36 is not binding at all (the cell
# stride it implies is 1), while the bootstrap is O(cells x frames x
# resamples). The cost was invisible until now for a specific reason:
# `block_bootstrap_std_batch` returns all-`nan` immediately below 4 frames and
# the smoke truth has 3, so the shape this layer was benchmarked at is the one
# shape where its dominant cost is identically zero. Measured
# `_truth_sampling_floors`, 3 components, 20 blocks:
#
# =================  ==========  ==========================================
# cells              wall time   note
# =================  ==========  ==========================================
# 1600, 3 frames        0.1 ms   the smoke shape: `nan` before any work
# 1600, 36 frames     157.5 ms
# 4096, 36 frames     448.0 ms   the cap
# 16384, 36 frames      1.9 s
# 65536, 36 frames     13.2 s    4 x 128 x 128 uncapped
# =================  ==========  ==========================================
#
# 4096 cells (12288 bootstrap rows) holds the whole `standard` stage at
# 4 x 128 x 128 x 36 to 1.6 s against 0.35 s at `basic`; uncapped, the truth
# pass alone would add the 13.2 s above. It does not materially move the
# floors it bounds: over 8 seeds at that shape the capped values sit within
# 0.5% (rms; 1.3% worst case) of the all-cell ones, against a floor that is an
# order-of-magnitude statement about the truth's window length. The realised
# count and the seed are both reported (`averaging.n_bootstrap_cells`,
# `averaging.n_bootstrap_cells_max`, `averaging.bootstrap_seed`), so the number
# is reproducible rather than merely stable.
_TRUTH_BOOTSTRAP_MAX_CELLS = 4096

# Seed for that subsample. Fixed rather than configurable: a floor that moved
# between two re-runs of the same metrics stage would be indistinguishable from
# a floor that moved because the run changed, and the cells are drawn once per
# run from a Generator of this seed alone.
_TRUTH_BOOTSTRAP_SEED = 0

# How close an interpolated fluid indicator must sit to 1 for the target cell
# to count as fluid. The indicator is 1 in fluid and 0 in solid, so a linear
# interpolation returns exactly 1 only when *every* source cell in the stencil
# was fluid: this is the conservative-threshold rule, expressed as a tolerance
# rather than an equality only to absorb float rounding.
_FLUID_INTERP_TOLERANCE = 1e-9


def _null_mean_field_block(reason):
    """The WP1.3 block with every measured leaf ``null``.

    Same contract as :func:`_null_statistic_block`: the nesting is identical to
    the healthy block so a consumer can index
    ``mean_field_metrics.nmse.u.systematic`` on a degraded run exactly as on a
    production one, and only the genuine constants (the hit rate's relative
    tolerance) keep their values, because they are configuration rather than
    measurement.
    """
    per_component = dict.fromkeys(MEAN_FIELD_COMPONENTS)
    return {
        "hit_rate": {
            **per_component,
            "per_z": None,
            "allowance": {**per_component, "reason": reason},
            "relative_tolerance": DEFAULT_HIT_RATE_TOLERANCE,
        },
        "fac2_velmag": None,
        "fb": {**per_component, "velmag": None},
        "nmse": {
            name: {"total": None, "systematic": None, "unsystematic": None}
            for name in (*MEAN_FIELD_COMPONENTS, "velmag")
        },
        "tke_rmse": {"value": None, "sampling_floor": None},
        "uw_stress_rmse": {"value": None, "sampling_floor": None},
        "averaging": {
            "n_time": None,
            "n_time_truth": None,
            "n_windows": None,
            "n_members": None,
            "n_members_scored": None,
            "n_stations": None,
            "n_stations_with_truth": None,
            "stride": None,
            "z_levels": None,
            "z_indices": None,
            "z_levels_extrapolated": None,
            "truth_z_levels": None,
            "z_offset_max": None,
            "n_cells": None,
            "n_scored_cells": None,
            "n_aggregate_cells": None,
            "n_stress_cells": None,
            "n_bootstrap_cells": None,
            "n_bootstrap_cells_max": None,
            "bootstrap_seed": None,
        },
        "masking": {
            "source": "none",
            "truth_source": "none",
            "ensemble_source": "none",
            "fluid_fraction": None,
            "scored_fraction": None,
            "ensemble_fluid_fraction": None,
            "truth_finite_fraction": None,
            "note": None,
        },
        "eval_fields": None,
        "reason": reason,
    }


def _centre_dims(field):
    """``(z, y, x)`` dim names of an already-colocated component.

    Raises:
        ValueError: If any of the three is missing, which is what a state
            without a spatial axis (or a solver whose centre axes are named
            something this repo has never written) looks like from here.
    """
    dims = []
    for candidates in (_CENTRE_Z_DIMS, _CENTRE_Y_DIMS, _CENTRE_X_DIMS):
        name = next((dim for dim in candidates if dim in field.dims), None)
        if name is None:
            raise ValueError(
                f"colocated field has no {candidates[0]!r}-like axis (dims "
                f"{tuple(field.dims)}); the mean-field layer needs all three "
                "cell-centre axes"
            )
        dims.append(name)
    return tuple(dims)


def evenly_spaced_levels(n_levels, n_slices):
    """``_vel_field_4z``'s z-level selection, shared rather than re-derived.

    The plan asks for this layer's z-slabs to line up with the existing |U|
    RMSE slices, which is a property of the *selection rule* and not of the
    count: ``np.linspace(0, n-1, k).round()`` is not the same set of levels as
    an evenly-strided one whenever ``k`` does not divide ``n``. Duplicates are
    dropped (``k > n`` asks for more levels than exist), so the returned array
    can be shorter than ``n_slices``.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be positive, got {n_levels}")
    return np.unique(np.linspace(0, n_levels - 1, n_slices).round().astype(int))


def _slab(field, dims, z_index, stride):
    """The ``(time, zlev, y, x)`` z-slab region of one colocated component."""
    zdim, ydim, xdim = dims
    selected = field.isel(
        {
            zdim: z_index,
            ydim: slice(None, None, stride),
            xdim: slice(None, None, stride),
        }
    )
    return selected.transpose("time", zdim, ydim, xdim)


def _columns(field, dims, station_x, station_y):
    """The ``(time, z, station)`` full-z station columns of one component.

    Pointwise (advanced) interpolation: the station coordinates share one
    ``station`` dim, so the result is one column per station rather than the
    outer product of the two coordinate lists. Stations outside the grid come
    back as ``nan`` rather than extrapolated -- an invented profile at a
    mis-specified station is worse than a missing one.
    """
    zdim, ydim, xdim = dims
    columns = field.interp(
        {
            xdim: xarray.DataArray(np.asarray(station_x, dtype=float), dims="station"),
            ydim: xarray.DataArray(np.asarray(station_y, dtype=float), dims="station"),
        },
        method="linear",
    )
    return columns.transpose("time", zdim, "station")


def _fluid_indicator(state, dims):
    """``(z, y, x)`` float indicator, 1 in fluid and 0 in solid, or ``None``.

    Built from :data:`BLANKING_VAR` when the state carries it and from nothing
    otherwise -- there is no second source. A cell counts as solid if it is
    blanked in **any** frame: the geometry is static in every shipped case, so
    an ``any`` and an ``all`` agree, and where they would not the conservative
    reading is the one that never scores a building interior.

    Float rather than boolean because the station columns need it interpolated,
    and a linear interpolation of the indicator is exactly the conservative
    threshold this layer wants (see :data:`_FLUID_INTERP_TOLERANCE`).
    """
    if BLANKING_VAR not in state:
        return None
    blanking = state[BLANKING_VAR]
    solid = blanking > 0.5
    for dim in ("time", "ensemble"):
        if dim in solid.dims:
            solid = solid.any(dim)
    zdim, ydim, xdim = dims
    return (~solid).astype(float).transpose(zdim, ydim, xdim)


def _fluctuation_covariance(slab, axis, i, j):
    """``n/(n-1) * (u_i - <u_i>)(u_j - <u_j>)``, the per-timestep covariance integrand.

    The ``<u'w'>`` sibling of :func:`_fluctuation_energy`, and built the same
    way for the same reason: a moving-block bootstrap resamples a *series*, so
    the reported second moment has to be expressible as the plain mean of one.
    The Bessel factor sits on the integrand so that mean is exactly the
    ``ddof=1`` covariance ``StreamingMoments.reynolds_stress`` reports, rather
    than a ``ddof=0`` cousin of it. (``_fluctuation_energy`` is
    ``0.5 * sum_c`` of this at ``i = j = c``; it is reused rather than
    re-expressed here, so the two cannot drift.)
    """
    n_time = slab.shape[axis]
    scale = n_time / (n_time - 1) if n_time > 1 else np.nan
    fluctuation = slab - slab.mean(axis=axis, keepdims=True)
    return scale * fluctuation[i] * fluctuation[j]


class MeanFieldAccumulator:
    """Per-member mean fields and Reynolds stresses from the shared read pass.

    The ensemble half of WP1.3, shaped exactly like
    :class:`SensorSeriesAccumulator` so both hang off the *one*
    :func:`stream_window_members` pass over the window state files: at
    Barcelona scale those files total tens of GB and a second full read pass is
    not affordable, which is why this class takes an already-open member rather
    than a path.

    **What it holds, and why that is affordable.** Two
    :class:`turbulence_stats.StreamingMoments` per member -- one for the
    ``n_z_slices`` z-slabs, one for the full-z station columns -- each holding
    ``C`` running means and ``C(C+1)/2`` co-moments *per cell*, independent of
    how many frames go through them. At the shipped ``C = 3`` that is 10 arrays
    of the slab shape per member, and the ensemble total is what to watch:

    ==============================  ==============  ==========
    ensemble x slab shape           per member      total
    ==============================  ==============  ==========
    M = 8, 4 x 64 x 64              1.3 MB          10 MB
    M = 32, 4 x 256 x 256           21 MB           0.62 GB
    M = 128, 4 x 512 x 512          84 MB           **10 GB**
    ==============================  ==============  ==========

    The last row is the plan's own Barcelona target, and it is held for the
    whole pass on top of the materialised member and colocation's float64
    temporaries -- so that shape needs ``run.metrics.mean_field_stride`` (2
    quarters it, 4 divides it by 16) rather than more memory. ``n_z_slices`` is
    the other relief valve. The station columns are free at every shape.

    **Accumulation spans windows.** ``StreamingMoments`` has no notion of a
    window, so member ``m``'s statistics cover every frame of every window it
    was fed -- which is what makes them comparable to a truth pass over the same
    time range. The window count is recorded so that range is reported rather
    than assumed.

    **Failures degrade the layer, they do not break the pass.** The sensor
    extraction shares this loop and must not be taken down by, say, a solver
    whose staggering ``colocate_components`` refuses; so a failure here sets
    :attr:`reason`, stops accumulating and logs a warning, and :meth:`result`
    returns ``None``. The driver turns that into a null ``mean_field_metrics``
    block per the master plan's degenerate-shape rule.

    Args:
        solver_name: ``assim_solver_name``, the staggering the window files
            follow.
        n_z_slices: ``run.metrics.n_z_slices``; the level *selection* follows
            :func:`evenly_spaced_levels` so the slabs coincide with the |U|
            RMSE slices ``streaming_state_rmse`` already reports.
        stride: ``run.metrics.mean_field_stride``, applied to the two
            horizontal axes of the slabs only -- never to the station columns
            (they are already a handful of cells) and never to time.
        station_x: Station x coordinates, in domain coordinates.
        station_y: Station y coordinates, same length.
        state_paths: The window state files being streamed, in window order, so
            the *static* solid-cell geometry can be read once per window
            instead of once per member (see :meth:`_record_mask`). ``None``
            falls back to reading it out of the member handed to
            :meth:`add_member`, which is what a caller with no paths (a unit
            test, an in-memory ensemble) has.
    """

    def __init__(
        self, solver_name, n_z_slices, stride, station_x, station_y, state_paths=None
    ):
        self.solver_name = solver_name
        self.n_z_slices = int(n_z_slices)
        self.stride = int(stride)
        self.station_x = np.asarray(station_x, dtype=float)
        self.station_y = np.asarray(station_y, dtype=float)
        self.state_paths = None if state_paths is None else list(state_paths)
        self.reason = None
        self._failed = False
        self._geometry = None
        self._moments = {}
        self._frames = {}
        self._windows = set()
        self._masked_windows = set()
        self._fluid_slab = None
        self._fluid_station = None
        self._mask_source = "none"
        self._z_edge_extrapolated = False

    def add_member(self, window_index, member_index, member_state):
        """Fold one member's window into that member's two accumulators."""
        if self._failed:
            return
        try:
            self._add_member(window_index, member_index, member_state)
        except (ValueError, KeyError) as error:
            self._failed = True
            self.reason = f"mean-field accumulation failed: {error}"
            logger.warning(
                "Mean-field layer disabled after window %d member %d: %s",
                window_index,
                member_index,
                error,
            )

    def _add_member(self, window_index, member_index, member_state):
        components = colocate_components(member_state, self.solver_name)
        dims = _centre_dims(components[0])
        if "time" not in components[0].dims:
            raise ValueError(
                "window state has no 'time' axis, so there is nothing to "
                "average over"
            )
        if self._geometry is None:
            self._initialise(components[0], dims)
            # Whether the TOP z-level of the returned grid was filled by
            # colocation's edge extrapolation rather than interpolated between
            # two faces. Read off the raw member, because it is a property of
            # the staggering and the file's dims -- the colocated array carries
            # no trace of it (that is the whole point of keeping the grid the
            # same length). `evenly_spaced_levels` always includes the top
            # index, so on a staggered backend this is always a scored level.
            self._z_edge_extrapolated = dims[0] in extrapolated_centre_dims(
                member_state, self.solver_name
            )
        zdim, _, _ = dims
        z_index, _, _, _ = self._geometry

        moments = self._moments.setdefault(
            member_index, {"slab": StreamingMoments(), "station": StreamingMoments()}
        )
        moments["slab"].update(
            *(
                np.asarray(_slab(field, dims, z_index, self.stride).values)
                for field in components
            )
        )
        moments["station"].update(
            *(
                np.asarray(_columns(field, dims, self.station_x, self.station_y).values)
                for field in components
            )
        )

        self._record_mask(window_index, member_state, dims, z_index)
        self._frames[member_index] = self._frames.get(member_index, 0) + int(
            components[0].sizes["time"]
        )
        self._windows.add(int(window_index))

    def _initialise(self, field, dims):
        """Fix the scored cells from the first member's colocated grid."""
        zdim, ydim, xdim = dims
        z_index = evenly_spaced_levels(int(field.sizes[zdim]), self.n_z_slices)
        self._geometry = (
            z_index,
            np.asarray(field[zdim].values, dtype=float),
            np.asarray(field[ydim].values, dtype=float)[:: self.stride],
            np.asarray(field[xdim].values, dtype=float)[:: self.stride],
        )

    def _record_mask(self, window_index, member_state, dims, z_index):
        """Fold one window's fluid indicator into the run's, reading it ONCE.

        The solid-cell geometry is static -- ``run_esmda`` writes
        :data:`BLANKING_VAR` replicated over ``(ensemble, time)``, and every
        shipped case's buildings stand still -- so reading it per member costs
        a second full ensemble-sized read of the window file for one 3-D
        frame's worth of information (measured on a 3-window / 8-member run
        dir: 7,680 blanking elements per window against 23,040 for ``u/v/w``,
        i.e. +33% on the window-state bytes, plus re-deriving and
        re-interpolating the indicator M times). So: one read per window, off
        ``isel(ensemble=0, time=0)`` of a separate open, and
        :func:`stream_window_members` keeps reading only ``u/v/w``.

        Where a member's own state carries the variable anyway (no
        ``state_paths``, e.g. an in-memory caller), it is taken from there
        instead -- same indicator, no extra read.
        """
        if int(window_index) in self._masked_windows:
            return
        self._masked_windows.add(int(window_index))
        indicator = self._window_indicator(window_index, member_state, dims)
        if indicator is None:
            return
        slab = np.asarray(
            _slab_indicator(indicator, dims, z_index, self.stride), dtype=float
        )
        columns = np.asarray(
            _column_indicator(indicator, dims, self.station_x, self.station_y),
            dtype=float,
        )
        # Conservative in both directions: a cell is fluid only if it is fluid
        # in every window (the geometry is static, so this is a no-op that would
        # catch a window whose blanking disagreed) and only if no solid cell
        # entered its interpolation stencil.
        slab_fluid = slab >= 1.0 - _FLUID_INTERP_TOLERANCE
        station_fluid = columns >= 1.0 - _FLUID_INTERP_TOLERANCE
        self._fluid_slab = (
            slab_fluid if self._fluid_slab is None else self._fluid_slab & slab_fluid
        )
        self._fluid_station = (
            station_fluid
            if self._fluid_station is None
            else self._fluid_station & station_fluid
        )
        self._mask_source = BLANKING_VAR

    def _window_indicator(self, window_index, member_state, dims):
        """One window's fluid indicator, from its file or from the member."""
        if self.state_paths is None or window_index >= len(self.state_paths):
            return _fluid_indicator(member_state, dims)
        with xarray.open_dataset(self.state_paths[window_index]) as state:
            if BLANKING_VAR not in state.data_vars:
                return None
            # One geometry frame, materialised before the handle closes. The
            # `isel` is written defensively over the dims the variable actually
            # has: `run_esmda` replicates it over `(ensemble, time)`, but a file
            # written with the bare 3-D geometry is equally valid input.
            frame = state[[BLANKING_VAR]]
            selection = {
                dim: 0
                for dim in ("ensemble", "time")
                if dim in frame[BLANKING_VAR].dims
            }
            return _fluid_indicator(frame.isel(selection).compute(), dims)

    def result(self):
        """Ensemble mean/std maps and station columns, or ``None`` if degraded.

        The reduction order is the plan's: each member's time-mean field first
        (that is what the accumulators hold), then the ensemble mean and the
        ``ddof=1`` ensemble std across members. Reversing it -- averaging the
        ensemble before averaging in time -- would report the mean field of the
        ensemble-mean *state*, which is a different quantity and is already
        covered by ``state_metrics``.

        **A non-finite member is excluded, not averaged in.** A diverged CFD
        member -- NaN over the whole domain or over the sub-region it blew up
        in -- is routine, and a plain ``mean`` over members propagates its NaN
        to every cell it touches: one member takes the whole layer down while
        ``n_members`` still says M. Members whose slab mean field is not finite
        everywhere are therefore dropped from *every* reduction here, the
        surviving count travels as ``n_members_scored`` and the excluded
        indices are logged. Dropping the member wholesale (rather than per
        cell) is the same casewise rule ``StreamingMoments`` applies in time
        and for the same reason: an ensemble mean and an ensemble ``ddof=1``
        spread computed over different member sets at neighbouring cells are
        not a field, and the spread is what ``eval_fields.nc`` publishes.
        Finiteness is judged on the **slab** mean alone: the station columns
        are legitimately NaN at a station outside the domain or inside a
        building, which is a station's problem and never a member's.

        Returns:
            A mapping of numpy arrays plus the grid the driver scores on, or
            ``None`` when nothing was accumulated, every member was non-finite,
            or the pass was disabled.
        """
        if self._failed or self._geometry is None or not self._moments:
            if self.reason is None:
                self.reason = "no ensemble member was accumulated"
            return None
        z_index, z_all, y, x = self._geometry
        accumulated = sorted(self._moments)
        members = [m for m in accumulated if self._member_is_finite(m)]
        excluded = [m for m in accumulated if m not in members]
        if excluded:
            logger.warning(
                "Mean-field layer: excluding ensemble member(s) %s from the "
                "mean field -- their time-mean slab is not finite everywhere "
                "(a diverged member); %d of %d members scored",
                ", ".join(str(m) for m in excluded),
                len(members),
                len(accumulated),
            )
        if not members:
            self.reason = (
                f"every one of the {len(accumulated)} ensemble members has a "
                "non-finite time-mean field, so there is nothing to score"
            )
            return None

        slab, station = {}, {}
        for key, stacks, quantities in (
            ("slab", slab, _SLAB_QUANTITIES),
            ("station", station, _STATION_QUANTITIES),
        ):
            per_member = {name: [] for name in quantities}
            for member in members:
                for name, value in self._member_quantities(member, key, quantities):
                    per_member[name].append(value)
            for name, values in per_member.items():
                stacked = np.stack(values)
                if name not in _SPREAD_ONLY_QUANTITIES:
                    stacks[name] = stacked.mean(axis=0)
                stacks[f"{name}_std"] = (
                    stacked.std(axis=0, ddof=1)
                    if len(members) > 1
                    else np.full(stacked.shape[1:], np.nan)
                )
                if key == "station":
                    # The empirical across-member fan WP1.4's S1 draws. Stations
                    # are a handful of columns, so persisting the quantiles
                    # costs nothing and removes the only reason stage 3 would
                    # have to re-read the window states: mean +- k*sigma is not
                    # a 5-95 band unless the ensemble happens to be Gaussian.
                    stacks[f"{name}_quantile"] = np.quantile(
                        stacked, STATION_QUANTILES, axis=0
                    )
            # Scored against the truth: the magnitude of the ENSEMBLE-MEAN mean
            # vector, not the ensemble mean of the members' magnitudes. The two
            # differ by the ensemble spread of the direction, and it is the
            # former the plan's "ensemble-mean-of-means vs truth" names.
            if key == "slab":
                stacks["velmag_of_mean"] = np.sqrt((stacks["mean"] ** 2).sum(axis=0))

        frames = sorted(self._frames[member] for member in members)
        return {
            "n_members": len(accumulated),
            "n_members_scored": len(members),
            "excluded_members": excluded,
            "n_windows": len(self._windows),
            "n_time": int(frames[0]),
            "n_time_ragged": frames[0] != frames[-1],
            "z_index": z_index,
            "z_levels": z_all[z_index],
            "z_all": z_all,
            "z_edge_extrapolated": self._z_edge_extrapolated,
            "y": y,
            "x": x,
            "station_x": self.station_x,
            "station_y": self.station_y,
            "slab": slab,
            "station": station,
            "fluid_slab": self._fluid_slab,
            "fluid_station": self._fluid_station,
            "mask_source": self._mask_source,
        }

    def _member_quantities(self, member, key, quantities):
        """One member's reduced fields for one region, computing only what is used.

        Yields ``(name, array)`` for the names in ``quantities`` and nothing
        else: every entry here is an array the size of the region *per member*,
        so a quantity nobody reads is real work at member scale rather than a
        stray dict key.
        """
        moments = self._moments[member][key]
        mean = moments.mean()
        stress = moments.reynolds_stress()
        for name in quantities:
            if name == "mean":
                yield name, mean
            elif name == "velmag":
                yield name, np.sqrt((mean**2).sum(axis=0))
            elif name == "tke":
                yield name, moments.tke()
            elif name == "uw":
                yield name, stress[0, 2]
            elif name == "rms":
                yield name, np.sqrt(np.stack([stress[i, i] for i in range(len(mean))]))
            else:  # pragma: no cover -- the tuples above are the whole domain
                raise ValueError(f"unknown mean-field quantity {name!r}")

    def _member_is_finite(self, member):
        """Whether this member's time-mean slab field is finite at every SCORED cell.

        Masked before the test, not after: a member that is NaN only inside
        buildings has not diverged, it has marked its solid cells -- which is
        exactly what the WP1.3 plan text assumes every backend does (no shipped
        one currently does; pypalm zero-fills PALM's NaN). Testing the raw slab
        would read that as a diverged member and drop it, and a backend where
        *every* member marks its solid cells would null the whole layer.
        """
        mean = self._moments[member]["slab"].mean()
        if self._fluid_slab is not None:
            mean = mean[:, self._fluid_slab]
        return bool(np.all(np.isfinite(mean)))


def _truth_chunk_length(n_time, cell_count):
    """Frames per truth read chunk under :data:`_TRUTH_CHUNK_BYTES`."""
    per_frame = max(int(cell_count), 1) * len(MEAN_FIELD_COMPONENTS) * 8
    return int(np.clip(_TRUTH_CHUNK_BYTES // per_frame, 1, max(int(n_time), 1)))


def _bootstrap_cells(candidates, n_time, seed=_TRUTH_BOOTSTRAP_SEED):
    """Which truth cells the sampling-floor bootstrap runs over.

    A **seeded random subsample without replacement** of the cells offered,
    bounded by both a time cap (:data:`_TRUTH_BOOTSTRAP_MAX_CELLS`, the only
    thing standing between this layer and a tens-of-seconds truth pass) and the
    memory ceiling (:data:`_TRUTH_SERIES_MAX_BYTES`, which the retained series
    has to fit under).

    Random rather than strided, which is what this used to be. A uniform stride
    over a **raveled** ``(zlev, y, x)`` index is not a uniform sample of the
    domain: at 4 x 512 x 512 over 360 frames the stride works out at 68, and
    ``gcd(68, 512) = 4``, so every retained cell sits at ``x = 0 (mod 4)``. On a
    regular building array -- which is what an urban case is -- that locks the
    sample onto one geometric phase (all street, or all facade), and the floors
    it produces describe that phase rather than the domain. A random subsample
    of the same size costs the same and cannot phase-lock; the seed is fixed
    and reported so it stays reproducible.

    Args:
        candidates: Flat indices of the cells eligible to be sampled -- the
            **fluid, strided** ones, i.e. the cells the scores are computed on.
        n_time: Frames per cell, which is what the memory bound scales with.
        seed: Seed of the ``Generator`` that draws the subsample.

    Returns:
        A sorted array of the retained flat indices (sorted so the gather is a
        forward pass over memory and the result does not depend on draw order).
    """
    candidates = np.asarray(candidates)
    per_cell = max(int(n_time), 1) * len(MEAN_FIELD_COMPONENTS) * 8
    affordable = max(int(_TRUTH_SERIES_MAX_BYTES // per_cell), 1)
    keep = min(_TRUTH_BOOTSTRAP_MAX_CELLS, affordable)
    if candidates.size <= keep:
        return candidates
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=keep, replace=False))


def truth_mean_field_stats(
    true_state_path,
    n_total,
    x_offset,
    start_idx,
    t_offset,
    solver_name,
    z_levels,
    station_x,
    station_y,
    stride=1,
    bootstrap_blocks=DEFAULT_BOOTSTRAP_BLOCKS,
):
    """The truth's own time-mean fields, stresses and sampling floors (WP1.3).

    Runs on the **truth's own grid and its own cadence** -- nothing is
    interpolated here. The plan's alignment rule for fields is "average in time
    first, then ``.interp`` the truth onto the assim grid", and every
    interpolation this layer performs therefore happens in
    :func:`mean_field_summary`, after this function has reduced the time axis
    away. Interpolating first would smear the truth's fluctuations before their
    second moments are taken, which is precisely the quantity the layer scores.

    **Vertical alignment is by nearest level, not by interpolation.** The
    scored z-levels are the assimilation grid's (``z_levels``), and the truth's
    slab is accumulated at the truth levels nearest to them. A vertical
    interpolation would be the more faithful thing to do, but it cannot be done
    *after* averaging without keeping the bracketing levels through the whole
    streaming pass -- i.e. paying for the full 3-D truth moments -- so the
    nearest level is taken and the residual offset is reported
    (``averaging.z_offset_max``) rather than hidden. The station columns have
    no such constraint: they are a handful of cells, so they keep the truth's
    **full** z axis and are interpolated in z after averaging.

    **Sampling floors.** ``hit_rate``'s absolute allowance ``W`` and the
    TKE / ``<u'w'>`` RMSE floors are all the truth's own sampling uncertainty,
    which needs the truth *series* at each cell and not a moment of it -- hence
    the retained ``(component, cell, time)`` array. Each is a moving-block
    bootstrap of a plain mean:

    * ``W_c`` bootstraps ``mean(u_c)`` per cell and reduces over cells by the
      **median**. The reduction is forced by ``hit_rate``'s signature -- its
      ``allowance`` is one scalar for the whole comparison -- and the median is
      the robust choice for a quantity whose per-cell values span a wake.
    * the TKE and ``<u'w'>`` floors bootstrap the per-timestep integrands
      (:func:`_fluctuation_energy`, :func:`_fluctuation_covariance`, both
      carrying the Bessel factor so their means are exactly the ``ddof=1``
      moments reported) and reduce over cells by the **RMS**, because the
      number they sit beside is an RMSE: the RMS of the per-cell sampling
      errors is the RMSE a perfect model would still score.

    **The floors describe the cells the scores are computed on.** The bootstrap
    runs over the truth's **fluid** cells at the scored **stride**, subsampled
    by :func:`_bootstrap_cells`, and not over the raw slab. The distinction is
    not cosmetic: a floor computed over building interiors as well as fluid is
    pulled down by the near-zero, near-constant series inside the buildings
    (measured on a 25%-solid truth with pylbm-like interiors: allowance 0.918x,
    TKE floor 0.866x, ``<u'w'>`` floor 0.866x of the fluid-only values), and a
    floor that is too low is **anti-conservative for its stated purpose** -- an
    RMSE genuinely inside the truth's sampling noise gets reported as though it
    were above the floor. The bias grows with the solid fraction. The stride is
    applied on the truth's own axes, which is a sample of the same *density* as
    the scores rather than a cell-for-cell alignment with them; the truth grid
    need not be the assimilation grid at all, and the floors are a property of
    the truth's own sampling.

    Args:
        true_state_path, n_total, x_offset, start_idx, t_offset: As for
            :func:`open_truth`, straight out of ``truth_access.yaml``.
        solver_name: ``truth_solver_name``; the truth's staggering, which is
            generally *not* the assimilation model's.
        z_levels: The assimilation grid's scored z coordinates.
        station_x, station_y: Station coordinates, in domain coordinates.
        stride: ``run.metrics.mean_field_stride``. Applied to the bootstrap
            sample only -- the moments themselves stay at the truth's full
            horizontal resolution, because they are interpolated onto the
            (already strided) assimilation grid afterwards.
        bootstrap_blocks: ``run.metrics.bootstrap_blocks``.

    Returns:
        A mapping of numpy arrays on the truth grid plus that grid's
        coordinates and the three sampling floors. Floors are ``nan`` whenever
        the bootstrap is undefined -- fewer than four frames, or a block length
        below 2, which **is** the smoke shape (3 frames against 20 blocks) --
        and ``hit_rate`` documents ``nan`` as its sanctioned "no floor
        available" value, so no special case is needed downstream.

    Raises:
        ValueError: If the truth cannot be colocated or has no time axis. The
            driver catches this and nulls the layer.
    """
    dataset = open_truth(true_state_path, n_total, x_offset, start_idx, t_offset)
    try:
        if "time" not in dataset.dims:
            raise ValueError("truth state has no 'time' axis to average over")
        n_time = int(dataset.sizes["time"])
        slab_moments, station_moments = StreamingMoments(), StreamingMoments()
        series_pieces = []
        geometry = None
        fluid = None
        station_fluid = None
        # Both budgets are sized from the file's own dims, before anything is
        # read: a component's non-time shape is the same cell count colocation
        # returns (colocation moves samples between axes, it does not add
        # cells), so no probe read is needed to bound the chunk.
        cells_per_frame = int(np.prod(dataset["u"].isel(time=0).shape))
        chunk_length = _truth_chunk_length(n_time, cells_per_frame)

        for start in range(0, n_time, chunk_length):
            piece = dataset.isel(time=slice(start, start + chunk_length))
            components = colocate_components(piece, solver_name)
            dims = _centre_dims(components[0])
            if geometry is None:
                geometry = _truth_geometry(components[0], dims, z_levels)
                # The solid-cell geometry is static, so it is read ONCE, off
                # the first frame of the first chunk, rather than re-read and
                # re-interpolated for every chunk of a multi-GB truth.
                fluid, station_fluid = _truth_fluid_masks(
                    piece, dims, geometry[0], station_x, station_y
                )
                bootstrap_cells = _bootstrap_cells(
                    _bootstrap_candidates(fluid, geometry, components, dims, stride),
                    n_time,
                )
            z_index = geometry[0]

            slabs = [
                np.asarray(_slab(field, dims, z_index, 1).values)
                for field in components
            ]
            slab_moments.update(*slabs)
            station_moments.update(
                *(
                    np.asarray(_columns(field, dims, station_x, station_y).values)
                    for field in components
                )
            )
            # (component, cell, time): time last, which is what the batched
            # bootstrap's row contract wants. Strided and subsampled to the
            # cells the SCORES see -- see this function's "sampling floors"
            # docstring section for why that is not optional.
            stacked = np.stack(slabs)[:, :, :, ::stride, ::stride]
            series_pieces.append(
                stacked.reshape(len(slabs), stacked.shape[1], -1)[:, :, bootstrap_cells]
                .transpose(0, 2, 1)
                .copy()
            )
    finally:
        dataset.close()

    if geometry is None:
        raise ValueError("truth state carries no frames to average")
    z_index, truth_z, y, x, z_all = geometry
    series = np.concatenate(series_pieces, axis=2)

    mean = slab_moments.mean()
    stress = slab_moments.reynolds_stress()
    station_mean = station_moments.mean()
    station_stress = station_moments.reynolds_stress()
    return {
        "n_time": n_time,
        "z_index": z_index,
        "z_levels": truth_z,
        "z_all": z_all,
        "y": y,
        "x": x,
        "mean": mean,
        "velmag": np.sqrt((mean**2).sum(axis=0)),
        "tke": slab_moments.tke(),
        "uw": stress[0, 2],
        # Masked here rather than in the reduction, so the z-interpolation onto
        # the assimilation levels propagates the NaN exactly as the horizontal
        # one does for the slabs -- one conservative rule, applied once.
        "station_mean": _masked(station_mean, station_fluid),
        "station_rms": _masked(
            np.sqrt(np.stack([station_stress[i, i] for i in range(len(station_mean))])),
            station_fluid,
        ),
        "station_tke": _masked(station_moments.tke(), station_fluid),
        "station_uw": _masked(station_stress[0, 2], station_fluid),
        "fluid_slab": fluid,
        "mask_source": BLANKING_VAR if fluid is not None else "none",
        "bootstrap_seed": _TRUTH_BOOTSTRAP_SEED,
        "n_bootstrap_cells_max": _TRUTH_BOOTSTRAP_MAX_CELLS,
        **_truth_sampling_floors(series, bootstrap_blocks),
    }


def _truth_fluid_masks(piece, dims, z_index, station_x, station_y):
    """``(slab, station)`` fluid masks of the truth, from ONE geometry frame.

    :func:`_fluid_indicator`'s ``any``-over-time reduction is what a caller
    handing it a whole chunk gets; here the first frame is selected instead, so
    a multi-GB truth pays one 3-D frame for its static geometry rather than one
    per chunk. The two agree wherever the geometry does not move, which is
    every shipped case (see :func:`_fluid_indicator`).

    The slab mask is at the truth's **full** horizontal resolution: it masks
    the moments, which are interpolated onto the assimilation grid afterwards.
    Only the bootstrap sample is strided.
    """
    frame = piece.isel(time=0) if "time" in piece.dims else piece
    indicator = _fluid_indicator(frame, dims)
    if indicator is None:
        return None, None
    slab = (
        np.asarray(_slab_indicator(indicator, dims, z_index, 1))
        >= 1.0 - _FLUID_INTERP_TOLERANCE
    )
    station = (
        np.asarray(_column_indicator(indicator, dims, station_x, station_y))
        >= 1.0 - _FLUID_INTERP_TOLERANCE
    )
    return slab, station


def _bootstrap_candidates(fluid, geometry, components, dims, stride):
    """Flat indices of the strided slab cells the sampling floors may use.

    The fluid cells of the strided slab, i.e. the sample the scores are
    computed on, flattened in the same ``(zlev, y, x)`` order the series is
    reshaped into. With no mask on the truth every cell is a candidate --
    unmasked floors are then wrong in the same way the unmasked scores beside
    them are, which ``masking.source`` already says out loud.
    """
    shape = (
        int(np.asarray(geometry[0]).size),
        len(range(0, int(components[0].sizes[dims[1]]), stride)),
        len(range(0, int(components[0].sizes[dims[2]]), stride)),
    )
    if fluid is None:
        return np.arange(int(np.prod(shape)))
    return np.flatnonzero(fluid[:, ::stride, ::stride].reshape(-1))


def _slab_indicator(indicator, dims, z_index, stride):
    """The z-slab region of a ``(z, y, x)`` fluid indicator."""
    zdim, ydim, xdim = dims
    return indicator.isel(
        {
            zdim: z_index,
            ydim: slice(None, None, stride),
            xdim: slice(None, None, stride),
        }
    ).values


def _column_indicator(indicator, dims, station_x, station_y):
    """The ``(z, station)`` station columns of a ``(z, y, x)`` fluid indicator."""
    zdim, ydim, xdim = dims
    return (
        indicator.interp(
            {
                xdim: xarray.DataArray(
                    np.asarray(station_x, dtype=float), dims="station"
                ),
                ydim: xarray.DataArray(
                    np.asarray(station_y, dtype=float), dims="station"
                ),
            },
            method="linear",
        )
        .transpose(zdim, "station")
        .values
    )


def _truth_geometry(field, dims, z_levels):
    """``(z_index, z_selected, y, x, z_all)`` for the truth's own grid.

    The z levels are the truth levels **nearest** the assimilation grid's; see
    :func:`truth_mean_field_stats` for why nearest rather than interpolated.
    Duplicates are kept deliberately -- a truth coarser in z than the
    assimilation grid legitimately maps two scored levels onto one truth level,
    and silently deduplicating would leave the two sides with different level
    counts.
    """
    zdim, ydim, xdim = dims
    truth_z = np.asarray(field[zdim].values, dtype=float)
    targets = np.asarray(z_levels, dtype=float)
    z_index = np.abs(truth_z[:, None] - targets[None, :]).argmin(axis=0)
    return (
        z_index,
        truth_z[z_index],
        np.asarray(field[ydim].values, dtype=float),
        np.asarray(field[xdim].values, dtype=float),
        truth_z,
    )


def _truth_sampling_floors(series, bootstrap_blocks):
    """The hit-rate allowance and the two RMSE floors from the truth series.

    Args:
        series: ``(component, cell, time)`` truth samples at the scored,
            strided, fluid cells (see :func:`_bootstrap_cells`).
        bootstrap_blocks: ``run.metrics.bootstrap_blocks``.

    Returns:
        ``{"allowance": {u, v, w}, "tke_floor": float, "uw_floor": float,
        "n_bootstrap_cells": int}``, every float ``nan`` where the bootstrap is
        undefined at this window length -- or where the mask left no fluid cell
        to run it over, which is a truth entirely inside a building and so a
        degenerate shape rather than a failure.
    """
    n_components, n_cells, n_time = series.shape
    if n_cells == 0:
        return {
            "allowance": dict.fromkeys(MEAN_FIELD_COMPONENTS, float("nan")),
            "tke_floor": float("nan"),
            "uw_floor": float("nan"),
            "n_bootstrap_cells": 0,
        }
    rows = series.reshape(n_components * n_cells, n_time)
    per_cell = block_bootstrap_std_batch(
        rows, statistic=np.mean, n_blocks=bootstrap_blocks
    ).reshape(n_components, n_cells)
    allowance = {
        name: _column_median(per_cell[i])
        for i, name in enumerate(MEAN_FIELD_COMPONENTS)
    }

    energy = _fluctuation_energy(series, axis=2)
    covariance = _fluctuation_covariance(series, 2, 0, 2)
    return {
        "allowance": allowance,
        "tke_floor": _rms(
            block_bootstrap_std_batch(
                energy, statistic=np.mean, n_blocks=bootstrap_blocks
            )
        ),
        "uw_floor": _rms(
            block_bootstrap_std_batch(
                covariance, statistic=np.mean, n_blocks=bootstrap_blocks
            )
        ),
        "n_bootstrap_cells": int(n_cells),
    }


def _column_median(values):
    """Median over the finite entries, ``nan`` when there are none."""
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _rms(values):
    """Root mean square over the finite entries, ``nan`` when there are none."""
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def _same_axis(left, right):
    """Whether two coordinate axes are the same grid (length and values)."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return left.size == right.size and bool(np.allclose(left, right))


def _interp_axes(array, dims, coords, targets):
    """``array`` linearly interpolated onto ``targets`` along named axes.

    A thin xarray wrapper, used for the *only* interpolations this layer
    performs -- the truth's already-time-averaged fields onto the assimilation
    grid. Kept in one place because the alignment rule ("average in time first,
    then interp") is a correctness property, and a second call site is a second
    chance to interpolate something that has not been averaged yet.

    NaN propagation is load-bearing rather than incidental: a truth field
    NaN-masked at its solid cells comes back NaN at **every** target cell whose
    interpolation stencil touched a building, which is exactly the conservative
    combination rule this layer wants -- a building's velocity can never bleed
    into a scored fluid cell. The cost is that the fluid cells immediately
    downstream of a wall are dropped from the score along with the wall itself,
    so the scored set shrinks by roughly the building *perimeter* (reported as
    ``masking.truth_finite_fraction``). The alternative -- interpolating the
    field unmasked and the mask separately -- keeps those cells but scores them
    against a value that is partly the inside of a building, which is a wrong
    number rather than a missing one.
    """
    field = xarray.DataArray(np.asarray(array, dtype=float), dims=dims, coords=coords)
    return np.asarray(field.interp(**targets).values)


def _masked(field, fluid):
    """``field`` with the non-fluid cells set to ``nan``, or unchanged if unmasked."""
    if fluid is None:
        return np.asarray(field, dtype=float)
    return np.where(fluid, np.asarray(field, dtype=float), np.nan)


def _masked_rmse(pred, obs):
    """RMSE over the cells where both fields are finite, ``nan`` if none are."""
    predicted = np.asarray(pred, dtype=float)
    observed = np.asarray(obs, dtype=float)
    keep = np.isfinite(predicted) & np.isfinite(observed)
    if not bool(keep.any()):
        return float("nan")
    return float(np.sqrt(np.mean((predicted[keep] - observed[keep]) ** 2)))


def _nmse_entry(pred, obs):
    """``{total, systematic, unsystematic}`` for one field pair.

    ``nmse`` is ``nan`` whenever the mean product is not positive and
    ``nmse_split`` is ``(nan, nan)`` at ``|FB| >= 2``; both are **routine** for
    a signed velocity component whose mean passes through zero, not a sign that
    anything failed. They arrive in the summary as ``null``.
    """
    total = nmse(pred, obs)
    systematic, unsystematic = nmse_split(fractional_bias(pred, obs), total)
    return {
        "total": _finite_or_none(total),
        "systematic": _finite_or_none(systematic),
        "unsystematic": _finite_or_none(unsystematic),
    }


def mean_field_summary(ensemble, truth, stride, eval_fields_name="eval_fields.nc"):
    """Score the ensemble's mean field against the truth's (WP1.3 step 4).

    The reduction the two streaming passes exist for. In order: the truth's
    time-mean fields are interpolated onto the assimilation grid (**after**
    averaging -- see :func:`_interp_axes`), the fluid mask is assembled from
    whatever the two sides actually carry, and the ensemble-mean-of-means is
    scored against the truth with the urban-CFD standards of
    ``docs/plans/esmda_turbulence_evaluation.md`` section 4.1: hit rate per
    signed component, FAC2 on ``|U|`` only, fractional bias, and NMSE with its
    systematic/unsystematic split. TKE and ``<u'w'>`` are reported as RMSE maps
    reduced to a scalar, each beside the truth's own sampling floor -- an RMSE
    below its floor is not skill, it is the truth's window length.

    **Masking, stated because the plan's premise is false.** The plan says to
    take fluid cells as the non-NaN ones. No backend marks solid cells with NaN
    (measured NaN fraction 0.0 in every ``.temp`` artifact), so the mask comes
    from :data:`BLANKING_VAR` where a side writes it and from nowhere
    otherwise. When neither side does, the scores include building interiors
    and are optimistic by roughly the building fraction -- a ``logger.warning``
    says so and ``masking.source`` records it, but nothing is invented.

    **The reported sample is the scored sample, by construction.** Every *cell
    count* under ``averaging`` and ``masking.scored_fraction`` come from
    :func:`_fluid_pairs` -- the cells that survived everything, not from the
    mask alone, which keeps its own ``masking.fluid_fraction`` -- and the
    aggregate scores exclude the extrapolated edge level
    (:func:`_aggregated_levels`) while ``hit_rate.per_z`` still reports it.
    All of it is one rule: every number in this block names the sample it was
    computed on, and where a name and a sample would drift apart the sample
    wins and the name changes.

    Args:
        ensemble: :meth:`MeanFieldAccumulator.result` output.
        truth: :func:`truth_mean_field_stats` output.
        stride: ``run.metrics.mean_field_stride``, reported under ``averaging``.
        eval_fields_name: File name recorded in the summary; the driver writes
            the returned Dataset there.

    Returns:
        ``(block, dataset)`` -- the ``mean_field_metrics`` mapping and the
        ``eval_fields.nc`` Dataset (truth mean, ensemble mean and ensemble std
        per quantity, the full-z station columns and the fluid mask).
    """
    y, x = ensemble["y"], ensemble["x"]
    grids_match = _same_axis(truth["y"], y) and _same_axis(truth["x"], x)

    def _to_assim(array, leading):
        masked = _masked(array, truth["fluid_slab"])
        if grids_match:
            return masked
        return _interp_axes(
            masked,
            (*leading, "y", "x"),
            {"y": truth["y"], "x": truth["x"]},
            {"y": y, "x": x},
        )

    truth_slab = {
        "mean": _to_assim(truth["mean"], ("component", "zlev")),
        "velmag": _to_assim(truth["velmag"], ("zlev",)),
        "tke": _to_assim(truth["tke"], ("zlev",)),
        "uw": _to_assim(truth["uw"], ("zlev",)),
    }

    # The scored mask: fluid on the assimilation side AND resolvable on the
    # truth side. The second half is not only the truth's own buildings -- a
    # truth grid that does not cover the whole assimilation domain leaves NaN
    # at the edge cells too, and neither is a cell this layer can score.
    fluid = ensemble["fluid_slab"]
    truth_finite = np.isfinite(truth_slab["velmag"])
    scored = truth_finite if fluid is None else (fluid & truth_finite)
    for key in ("mean", "velmag", "tke", "uw"):
        truth_slab[key] = _masked(truth_slab[key], scored)

    slab = ensemble["slab"]
    predicted = {
        "mean": _masked(slab["mean"], scored),
        "velmag": _masked(slab["velmag_of_mean"], scored),
        "tke": _masked(slab["tke"], scored),
        "uw": _masked(slab["uw"], scored),
    }
    # The cells that actually carry a scoreable pair, which is what the counts
    # below report. `scored` is the MASK; the pairs are the mask minus whatever
    # else came out non-finite (a member that diverged over part of the domain,
    # a cell the truth interpolation could not reach). Reporting the mask as
    # though it were the sample is how a summary ends up claiming 104 scored
    # cells for a comparison that scored none.
    retained = _fluid_pairs(predicted, truth_slab)
    for key in ("mean", "velmag", "tke", "uw"):
        truth_slab[key] = _masked(truth_slab[key], retained)
        predicted[key] = _masked(predicted[key], retained)

    # Levels whose values came out of colocation's edge extrapolation. They are
    # reported per z but kept out of the aggregates: `evenly_spaced_levels`
    # always includes the top index, so on a staggered backend the extrapolated
    # level is ALWAYS one of the scored ones (1 of 4 at the shipped
    # `n_z_slices`), and its second moments are inflated by up to 5x. At
    # `nz == 1` the two indices coincide and excluding would leave nothing, so
    # `_aggregated_levels` keeps everything and this says so out loud.
    aggregated, is_extrapolated = _aggregated_levels(ensemble)
    extrapolated = [
        float(level)
        for level, edge in zip(ensemble["z_levels"], is_extrapolated)
        if edge
    ]
    notes = []
    if is_extrapolated.any() and bool(np.all(aggregated)):
        notes.append(
            f"the only scored z-level ({extrapolated[0]:g}) is the one "
            "colocation extrapolated, so it could not be excluded from the "
            "aggregate scores; their second moments carry that edge"
        )
        logger.warning("Mean-field layer: %s", notes[-1])

    block = _score_mean_fields(predicted, truth_slab, truth, ensemble, aggregated)
    block["averaging"] = {
        "n_time": int(ensemble["n_time"]),
        "n_time_truth": int(truth["n_time"]),
        "n_windows": int(ensemble["n_windows"]),
        "n_members": int(ensemble["n_members"]),
        "n_members_scored": int(ensemble["n_members_scored"]),
        "n_stations": int(ensemble["station_x"].size),
        "n_stations_with_truth": _n_stations_with_truth(ensemble, truth),
        "stride": int(stride),
        "z_levels": [float(v) for v in ensemble["z_levels"]],
        "z_indices": [int(v) for v in ensemble["z_index"]],
        "z_levels_extrapolated": extrapolated,
        "truth_z_levels": [float(v) for v in truth["z_levels"]],
        "z_offset_max": float(
            np.max(np.abs(np.asarray(truth["z_levels"]) - ensemble["z_levels"]))
        ),
        "n_cells": int(scored.size),
        # Three counts, because three samples exist: `n_scored_cells` is every
        # cell carrying a pair (the sample `hit_rate.per_z` and `masking`
        # describe), `n_aggregate_cells` is the subset on the levels the
        # top-level scores reduce over (they differ exactly when a level was
        # extrapolated), and `n_stress_cells` is the subset of THAT which the
        # two RMSEs could use -- a `ddof=1` moment needs two frames per cell
        # where a mean needs one, so the stresses legitimately see fewer cells
        # (76 of 104 on the divergence fixture). Each number sits with the
        # sample it was computed on; see :func:`_stress_pairs`.
        "n_scored_cells": int(np.count_nonzero(retained)),
        "n_aggregate_cells": int(np.count_nonzero(retained[aggregated])),
        "n_stress_cells": int(
            np.count_nonzero(_stress_pairs(predicted, truth_slab)[aggregated])
        ),
        "n_bootstrap_cells": int(truth["n_bootstrap_cells"]),
        "n_bootstrap_cells_max": int(truth["n_bootstrap_cells_max"]),
        "bootstrap_seed": int(truth["bootstrap_seed"]),
    }
    block["masking"] = _masking_block(
        ensemble, truth, fluid, truth_finite, scored, retained
    )
    block["eval_fields"] = eval_fields_name
    block["reason"] = _degraded_reason(ensemble, notes)
    return block, _eval_fields_dataset(ensemble, truth, truth_slab, fluid, grids_match)


def _fluid_pairs(predicted, observed):
    """The cells where every mean-field score has a finite pair to score.

    One retained set for all of them, rather than each score dropping its own
    non-finite cells: ``nmse_split``'s identity and the per-component hit rates
    are only comparable across components if they are computed on the same
    sample, and the summary reports **one** ``n_scored_cells``. The stress
    RMSEs keep their own pairwise finiteness on top of this (their inputs can
    be ``nan`` from a one-frame window while the mean field is perfectly
    finite), so this is the base sample and never a widening of it.
    """
    keep = np.ones(np.asarray(observed["velmag"]).shape, dtype=bool)
    for fields in (predicted, observed):
        keep &= np.isfinite(fields["velmag"])
        keep &= np.all(np.isfinite(fields["mean"]), axis=0)
    return keep


def _aggregated_levels(ensemble):
    """``(aggregated, extrapolated)``, two boolean masks over the scored z-levels.

    ``extrapolated`` is a fact about the grid -- which scored levels are the top
    index of an axis colocation had to extrapolate -- and is reported whatever
    happens next. ``aggregated`` is the decision: normally its complement, so
    the inflated edge stays out of the aggregate scores.

    **Except when that would leave nothing.** At ``nz == 1`` the only scored
    level *is* the extrapolated one (index 0 and ``nz - 1`` are the same level),
    and a udales-staggered state with a 2-level ``zm`` and a 1-level ``zt``
    colocates cleanly, so this is reachable rather than hypothetical. Excluding
    there would null every aggregate score with no ``reason`` -- precisely the
    signature this layer has already been rewritten once to remove. So every
    level is kept instead and the caller says so in ``reason``: a contaminated
    number that announces its contamination beats a silent null.
    """
    z_index = np.asarray(ensemble["z_index"])
    extrapolated = np.zeros(z_index.shape, dtype=bool)
    if ensemble["z_edge_extrapolated"]:
        extrapolated = z_index == int(ensemble["z_all"].size) - 1
    aggregated = ~extrapolated
    if not aggregated.any():
        aggregated = np.ones_like(extrapolated)
    return aggregated, extrapolated


def _stress_pairs(predicted, observed):
    """The cells the two stress RMSEs are computed on.

    Deliberately not :func:`_fluid_pairs`: ``tke`` and ``<u'w'>`` are ``ddof=1``
    moments, so they are ``nan`` wherever a cell's per-cell sample count is
    below 2 -- a member that lost all but one frame over a sub-region, a
    one-frame window -- while the mean field there is perfectly finite. Folding
    that into the base sample would let a short window empty the hit rate's
    sample too, so the RMSEs keep their own pairwise rule and this is the count
    that travels with them.

    ``tke`` and ``<u'w'>`` share one sample by construction rather than by
    luck: both come from the same :class:`StreamingMoments` per-cell count and
    the same mask, so they are ``nan`` in exactly the same cells (measured
    equal on the divergence fixture, 76 of 104). The intersection is taken
    anyway, so the count can only ever understate, never overstate.
    """
    keep = np.ones(np.asarray(observed["tke"]).shape, dtype=bool)
    for fields in (predicted, observed):
        keep &= np.isfinite(fields["tke"]) & np.isfinite(fields["uw"])
    return keep


def _n_stations_with_truth(ensemble, truth):
    """How many stations have a truth column at all, warning about those that do not.

    ``_column_indicator``'s all-fluid stencil rule is strict on purpose -- a
    station whose 2x2 horizontal stencil touches a solid cell loses its whole
    truth column rather than being profiled against a building interior -- and
    stations default to the sensor x/y, which in an urban case sit on facades
    and roofs. So an empty truth column is the common case, not the corner, and
    the one thing it must not be is silent: WP1.4's profile panels would draw a
    posterior band with no truth line and nothing would say why.
    """
    columns = np.asarray(truth["station_mean"], dtype=float)
    has_truth = np.isfinite(columns).any(axis=tuple(range(columns.ndim - 1)))
    missing = np.flatnonzero(~has_truth)
    if missing.size:
        logger.warning(
            "Mean-field layer: %d of %d station(s) have no truth column -- "
            "their horizontal stencil touches a solid cell, so the truth "
            "profile is empty at station(s) %s (x, y): %s",
            missing.size,
            has_truth.size,
            ", ".join(str(int(i)) for i in missing),
            ", ".join(
                f"({ensemble['station_x'][i]:g}, {ensemble['station_y'][i]:g})"
                for i in missing
            ),
        )
    return int(has_truth.sum())


def _degraded_reason(ensemble, extra=()):
    """``None`` on a healthy layer; what was dropped or kept anyway when not.

    The block still carries numbers here -- the alternative, nulling every
    score because one member of eight diverged, throws away a perfectly good
    7-member answer -- so the ``reason`` is what tells a reader that
    ``n_members`` and ``n_members_scored`` differ on purpose, or that a level
    the layer would normally exclude had to be scored anyway.

    Args:
        ensemble: :meth:`MeanFieldAccumulator.result` output.
        extra: Further clauses from the reduction, already phrased.
    """
    causes = []
    excluded = ensemble["excluded_members"]
    if excluded:
        causes.append(
            f"{len(excluded)} of {ensemble['n_members']} ensemble members "
            f"({', '.join(str(m) for m in excluded)}) have a non-finite "
            "time-mean field and were excluded; every score here is over the "
            f"remaining {ensemble['n_members_scored']}"
        )
    causes.extend(extra)
    return "; ".join(causes) if causes else None


def _score_mean_fields(predicted, observed, truth, ensemble, aggregated):
    """The four urban-CFD scores plus the two stress RMSEs, per component and z.

    ``aggregated`` selects the z-levels the **top-level** numbers reduce over
    (see :func:`_aggregated_levels`); ``per_z`` always reports every scored
    level, so an excluded level is visible rather than missing.
    """
    allowance = truth["allowance"]
    aggregate = {
        name: np.asarray(field)[..., aggregated, :, :]
        for name, field in predicted.items()
    }
    observed_aggregate = {
        name: np.asarray(field)[..., aggregated, :, :]
        for name, field in observed.items()
    }
    hit = {}
    for i, name in enumerate(MEAN_FIELD_COMPONENTS):
        hit[name] = _finite_or_none(
            hit_rate(
                aggregate["mean"][i],
                observed_aggregate["mean"][i],
                allowance=allowance[name],
            )
        )
    per_z = []
    for level in range(observed["velmag"].shape[0]):
        entry = {"z": float(ensemble["z_levels"][level])}
        for i, name in enumerate(MEAN_FIELD_COMPONENTS):
            entry[name] = _finite_or_none(
                hit_rate(
                    predicted["mean"][i, level],
                    observed["mean"][i, level],
                    allowance=allowance[name],
                )
            )
        entry["fac2_velmag"] = _finite_or_none(
            fac2(predicted["velmag"][level], observed["velmag"][level])
        )
        # Named on the entry itself, not only in `averaging`: a per-z table is
        # read row by row, and this row's second moments carry colocation's
        # edge extrapolation (up to 5x the interior variance).
        entry["extrapolated"] = not bool(aggregated[level])
        per_z.append(entry)

    fb = {
        name: _finite_or_none(
            fractional_bias(aggregate["mean"][i], observed_aggregate["mean"][i])
        )
        for i, name in enumerate(MEAN_FIELD_COMPONENTS)
    }
    fb["velmag"] = _finite_or_none(
        fractional_bias(aggregate["velmag"], observed_aggregate["velmag"])
    )
    nmse_block = {
        name: _nmse_entry(aggregate["mean"][i], observed_aggregate["mean"][i])
        for i, name in enumerate(MEAN_FIELD_COMPONENTS)
    }
    nmse_block["velmag"] = _nmse_entry(
        aggregate["velmag"], observed_aggregate["velmag"]
    )

    unavailable = None
    if not any(np.isfinite(allowance[name]) for name in MEAN_FIELD_COMPONENTS):
        unavailable = (
            "the truth's block bootstrap is undefined at this window length, so "
            "the hit rate is the pure relative test (which can only be lower)"
        )
    return {
        "hit_rate": {
            **hit,
            "per_z": per_z,
            "allowance": {
                **{
                    name: _finite_or_none(allowance[name])
                    for name in MEAN_FIELD_COMPONENTS
                },
                "reason": unavailable,
            },
            "relative_tolerance": DEFAULT_HIT_RATE_TOLERANCE,
        },
        "fac2_velmag": _finite_or_none(
            fac2(aggregate["velmag"], observed_aggregate["velmag"])
        ),
        "fb": fb,
        "nmse": nmse_block,
        "tke_rmse": {
            "value": _finite_or_none(
                _masked_rmse(aggregate["tke"], observed_aggregate["tke"])
            ),
            "sampling_floor": _finite_or_none(truth["tke_floor"]),
        },
        "uw_stress_rmse": {
            "value": _finite_or_none(
                _masked_rmse(aggregate["uw"], observed_aggregate["uw"])
            ),
            "sampling_floor": _finite_or_none(truth["uw_floor"]),
        },
    }


def _masking_block(ensemble, truth, fluid, truth_finite, scored, retained):
    """What masked the scores, and how much of the domain survived it.

    Two fractions, because there are two samples and giving them one name is
    the whole failure mode this layer keeps having to defend against:

    * ``fluid_fraction`` is the **mask's** own number -- the combined mask,
      fluid on the assimilation side and resolvable on the truth side -- which
      is what its name says and what the sibling ``ensemble_fluid_fraction`` /
      ``truth_finite_fraction`` entries report for each side separately.
    * ``scored_fraction`` is the fraction that carried a scoreable pair
      (:func:`_fluid_pairs`), i.e. the sample the numbers in this block were
      actually computed on. It equals ``n_scored_cells / n_cells`` by
      construction, and it is at most ``fluid_fraction``.

    The two are **currently equal on every reachable path**, and that is a
    result rather than a redundancy: the only sources of non-finiteness left
    are the mask itself and a diverged member, and members are excluded whole
    (:meth:`MeanFieldAccumulator.result`), so the scored sample comes back to
    the mask. They are separate keys so that a future third source shows up in
    the summary as a gap between them instead of quietly redefining one number.
    """
    sources = {
        "ensemble_source": ensemble["mask_source"],
        "truth_source": truth["mask_source"],
    }
    source = (
        BLANKING_VAR if BLANKING_VAR in sources.values() else "none"  # noqa: SIM300
    )
    note = None
    if source == "none":
        note = (
            "no backend on either side writes a solid-cell mask (pyudales ships "
            "none and pypalm fills PALM's NaN with 0.0), so these scores include "
            "building interiors and are optimistic by roughly the building "
            "fraction; parsing uDALES solid_c.txt and preserving PALM's NaN are "
            "both out of phase 1"
        )
        logger.warning(
            "Mean-field layer is UNMASKED: neither the truth (%s) nor the "
            "assimilation state (%s) carries a %r variable, so FAC2 and the hit "
            "rate are computed over building interiors as well as fluid",
            truth["mask_source"],
            ensemble["mask_source"],
            BLANKING_VAR,
        )
    return {
        "source": source,
        **sources,
        "fluid_fraction": _finite_or_none(np.mean(scored)),
        "scored_fraction": _finite_or_none(np.mean(retained)),
        "ensemble_fluid_fraction": (
            None if fluid is None else _finite_or_none(np.mean(fluid))
        ),
        "truth_finite_fraction": _finite_or_none(np.mean(truth_finite)),
        "note": note,
    }


def _eval_fields_dataset(ensemble, truth, truth_slab, fluid, grids_match):
    """Build ``eval_fields.nc``: the reduced fields WP1.4's figures need.

    The sanctioned handoff between the metrics stage and the figure stage.
    Stage 3 currently re-derives its inputs from the raw artifacts; everything
    here costs a streaming pass over the window state files, so recomputing it
    per figure is exactly what this file exists to prevent.

    Two grids in one file, which is why the vertical dims are named
    differently. ``zlev`` is the ``n_z_slices`` scored planes (F1's mean-field
    comparison and F2's spread maps); ``z`` is the **full** assimilation column,
    carried only at the station points, because a 4-plane sampling is not a
    vertical profile and S1 needs one.

    ``fluid_mask`` is the **assimilation-side** indicator (1 = fluid), which is
    what a figure needs for ``set_bad`` on the posterior panels; every other
    exclusion the summary applied (the truth's own buildings, the perimeter the
    cross-grid interpolation erodes, a diverged member's footprint) is already
    baked into the ``truth_*`` variables as NaN, so ``isfinite(truth_velmag)``
    is exactly the set of cells the summary scored -- and ``fluid_mask`` alone
    is a **superset** of it, equal on a matched-grid run where both sides carry
    the same mask (measured 104 == 104 on the wiring fixture) and strictly
    larger as soon as the truth's mask or grid differs (16 vs 8 at stride 3).
    Use ``isfinite(truth_velmag)`` when the question is "what was scored".

    ``station_*_quantile`` carries the across-member 5/25/50/75/95 percentiles
    of each station column, so a profile panel can draw the **empirical** fan
    rather than mean +- k*sigma, which is only the same band for a Gaussian
    ensemble. Stations are a handful of columns, so this costs nothing and
    removes the last reason for the figure stage to re-read the window states.
    """
    slab, station = ensemble["slab"], ensemble["station"]
    z_all = ensemble["z_all"]
    station_dims = ("component", "z", "station")
    truth_z = truth["z_all"]

    def _to_assim_z(array, dims):
        """Truth station columns onto the assimilation levels, after averaging."""
        if _same_axis(truth_z, z_all):
            return np.asarray(array, dtype=float)
        return _interp_axes(array, dims, {"z": truth_z}, {"z": z_all})

    slab_dims = ("component", "zlev", "y", "x")
    plane_dims = ("zlev", "y", "x")
    column_dims = ("z", "station")
    variables = {
        "truth_mean": (slab_dims, truth_slab["mean"]),
        "ensemble_mean": (slab_dims, slab["mean"]),
        "ensemble_std": (slab_dims, slab["mean_std"]),
        "truth_velmag": (plane_dims, truth_slab["velmag"]),
        "ensemble_velmag": (plane_dims, slab["velmag_of_mean"]),
        "ensemble_velmag_std": (plane_dims, slab["velmag_std"]),
        "truth_tke": (plane_dims, truth_slab["tke"]),
        "ensemble_tke": (plane_dims, slab["tke"]),
        "ensemble_tke_std": (plane_dims, slab["tke_std"]),
        "truth_uw": (plane_dims, truth_slab["uw"]),
        "ensemble_uw": (plane_dims, slab["uw"]),
        "ensemble_uw_std": (plane_dims, slab["uw_std"]),
        "fluid_mask": (
            plane_dims,
            (
                np.ones(slab["tke"].shape, dtype=np.int8)
                if fluid is None
                else fluid.astype(np.int8)
            ),
        ),
        "station_truth_mean": (
            station_dims,
            _to_assim_z(truth["station_mean"], station_dims),
        ),
        "station_ensemble_mean": (station_dims, station["mean"]),
        "station_ensemble_std": (station_dims, station["mean_std"]),
        "station_truth_rms": (
            station_dims,
            _to_assim_z(truth["station_rms"], station_dims),
        ),
        "station_ensemble_rms": (station_dims, station["rms"]),
        "station_ensemble_rms_std": (station_dims, station["rms_std"]),
        "station_truth_tke": (
            column_dims,
            _to_assim_z(truth["station_tke"], column_dims),
        ),
        "station_ensemble_tke": (column_dims, station["tke"]),
        "station_ensemble_tke_std": (column_dims, station["tke_std"]),
        "station_truth_uw": (
            column_dims,
            _to_assim_z(truth["station_uw"], column_dims),
        ),
        "station_ensemble_uw": (column_dims, station["uw"]),
        "station_ensemble_uw_std": (column_dims, station["uw_std"]),
        "station_fluid_mask": (
            column_dims,
            (
                np.ones(station["tke"].shape, dtype=np.int8)
                if ensemble["fluid_station"] is None
                else ensemble["fluid_station"].astype(np.int8)
            ),
        ),
        "station_x": (("station",), ensemble["station_x"]),
        "station_y": (("station",), ensemble["station_y"]),
    }
    # The empirical across-member fan, one variable per profile row S1 draws.
    # `quantile` leads the dims so a figure can `.sel(quantile=0.05)` and get
    # the column back in the same layout as its `station_ensemble_*` sibling.
    for name, dims in (
        ("mean", station_dims),
        ("rms", station_dims),
        ("tke", column_dims),
        ("uw", column_dims),
    ):
        variables[f"station_ensemble_{name}_quantile"] = (
            ("quantile", *dims),
            station[f"{name}_quantile"],
        )
    dataset = xarray.Dataset(
        {
            name: (dims, np.asarray(values))
            for name, (dims, values) in variables.items()
        },
        coords={
            "component": list(MEAN_FIELD_COMPONENTS),
            "zlev": np.asarray(ensemble["z_levels"], dtype=float),
            "y": np.asarray(ensemble["y"], dtype=float),
            "x": np.asarray(ensemble["x"], dtype=float),
            "z": np.asarray(z_all, dtype=float),
            "station": np.arange(int(ensemble["station_x"].size)),
            "quantile": np.asarray(STATION_QUANTILES, dtype=float),
        },
        attrs={
            "description": (
                "WP1.3 mean-field / Reynolds-stress fields: time-mean over the "
                "whole assimilation horizon, per quantity, for the truth and "
                "for the posterior ensemble"
            ),
            "n_members": int(ensemble["n_members"]),
            # The ensemble every field here was reduced over, which is not
            # `n_members` when a member diverged (see
            # `MeanFieldAccumulator.result`) -- a figure labelling a band "M =
            # 8" off the wrong one of these would be captioning a lie.
            "n_members_scored": int(ensemble["n_members_scored"]),
            "n_windows": int(ensemble["n_windows"]),
            "n_time": int(ensemble["n_time"]),
            "n_time_truth": int(truth["n_time"]),
            "mask_source": ensemble["mask_source"],
            "truth_mask_source": truth["mask_source"],
            "truth_regridded": int(not grids_match),
            "stress_kind": "resolved only (no subgrid contribution)",
        },
    )
    dataset["ensemble_velmag"].attrs["long_name"] = (
        "magnitude of the ensemble-mean time-mean velocity vector (the scored "
        "prediction), NOT the ensemble mean of the members' speeds"
    )
    dataset["ensemble_velmag_std"].attrs[
        "long_name"
    ] = "across-member std of each member's own |time-mean velocity|"
    dataset["fluid_mask"].attrs["source"] = ensemble["mask_source"]
    dataset["station_fluid_mask"].attrs["source"] = ensemble["mask_source"]
    return dataset


def mean_field_scores(
    accumulator,
    true_state_path,
    n_total,
    x_offset,
    start_idx,
    t_offset,
    truth_solver_name,
    stride=1,
    bootstrap_blocks=DEFAULT_BOOTSTRAP_BLOCKS,
    eval_fields_name="eval_fields.nc",
):
    """The WP1.3 entry point: run the truth pass and reduce against the ensemble.

    The ensemble half has already happened by the time this is called -- its
    accumulators were filled from the *shared* :func:`stream_window_members`
    pass that also feeds the sensor extraction, which is the production-scale
    requirement the plan states as a review-blocker, not an optimization. So
    the only IO here is the truth's, on its own grid and its own cadence.

    Every failure path returns the full schema with ``null`` leaves and a
    non-null ``reason`` plus one log line, per the master plan's
    degenerate-shape rule: an old run dir, an absent truth, a solver whose
    staggering cannot be colocated, or a level at which the layer never ran.

    Args:
        accumulator: A filled :class:`MeanFieldAccumulator`, or ``None`` when
            the layer was not enabled for this run.
        true_state_path, n_total, x_offset, start_idx, t_offset: As for
            :func:`open_truth`, straight out of ``truth_access.yaml``.
        truth_solver_name: ``truth_solver_name`` -- generally not the
            assimilation model's.
        stride: ``run.metrics.mean_field_stride``. Reported, already applied by
            the accumulator, and passed on to the truth pass -- the sampling
            floors have to describe the same sample density as the scores they
            gate, and a stride-blind floor is a floor for a run that was not
            performed.
        bootstrap_blocks: ``run.metrics.bootstrap_blocks``.
        eval_fields_name: File name recorded in the summary.

    Returns:
        ``(block, dataset)``; ``dataset`` is ``None`` on every degraded path,
        and the caller writes it when it is not.
    """
    if accumulator is None:
        return _null_mean_field_block("the mean-field layer did not run"), None
    ensemble = accumulator.result()
    if ensemble is None:
        logger.info("Mean-field layer: %s; reporting nulls", accumulator.reason)
        return _null_mean_field_block(accumulator.reason), None

    try:
        truth = truth_mean_field_stats(
            true_state_path,
            n_total,
            x_offset,
            start_idx,
            t_offset,
            truth_solver_name,
            ensemble["z_levels"],
            ensemble["station_x"],
            ensemble["station_y"],
            stride=stride,
            bootstrap_blocks=bootstrap_blocks,
        )
    except (ValueError, KeyError, OSError) as error:
        reason = f"truth mean-field pass failed: {error}"
        logger.warning("Mean-field layer disabled: %s", reason)
        return _null_mean_field_block(reason), None

    if ensemble["n_time_ragged"]:
        # Not fatal -- the moments are per cell and weight every frame equally,
        # so a member with more frames is simply better sampled -- but it means
        # `averaging.n_time` is the shortest member's count rather than a shared
        # one, and a reader comparing two members' spreads should know.
        logger.info(
            "Mean-field layer: ensemble members carry different frame counts; "
            "averaging.n_time reports the smallest"
        )
    return mean_field_summary(ensemble, truth, stride, eval_fields_name)
