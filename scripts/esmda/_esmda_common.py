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
import warnings

import numpy as np
import xarray
import yaml
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_operator import ObservationOperator
from evaluation.turbulence import (
    MomentAccumulator,
    block_bootstrap_std,
    colocate_components,
    evenly_spaced_levels,
)
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_validation_points,
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


# ---------------------------------------------------------------------------
# Sensor time-series extraction (truth vs ensemble at fixed points)
# ---------------------------------------------------------------------------


def _sensor_component_timeseries(state, obs_x, obs_y, obs_z, solver_name):
    """Per-component ``(u, v, w)`` velocity time series at each sensor point.

    Trilinearly interpolates u/v/w (each on its own staggered grid, resolved via
    an ``ObservationOperator``'s solver-specific dim mapping) at the sensor
    locations, keeping any leading dims (``ensemble``, ``time``). Returns a
    DataArray with a leading ``component`` dim: ``(component, ..., time, sensor)``.
    The velocity magnitude |U| is :func:`evaluation.sensors.sensor_magnitude` of
    this (used for the sensor figures); the full vector is used for the sensor
    error metrics.
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


def ensemble_sensor_series(
    state_paths, sensor_sets, solver_name, sim_time, on_member=None
):
    """Ensemble per-component ``(u, v, w)`` sensor series across rollout windows.

    Interpolates u/v/w at every sensor set's points (keeping ``component`` +
    ``ensemble`` + ``time``), rebasing each window's local time onto a single
    global axis (window ``w`` starts at ``w*sim_time``) so it lines up with the
    truth. Returns ``{name: DataArray(component, ensemble, time, sensor)}``.

    **One member at a time.** A window state file is
    ``(ensemble, time, z, y, x)`` per component and runs to gigabytes, so it is
    opened lazily and sliced per member: peak memory is three components of one
    member's window, not the whole ensemble. (This function used to ``.load()``
    each file whole -- the last such site in the post-processing stack.) The
    per-member slice *is* materialised, because the interpolation reads
    ``.values``; every sensor set is interpolated from that one slice so a
    second set costs no extra read.

    ``on_member`` is called as ``on_member(window_index, member_index, member)``
    with that same materialised slice, before it is dropped. It exists so the
    WP1.4 mean fields can be accumulated inside this pass rather than in a
    second one: at Barcelona scale the window files total tens of GB and a
    second full read of them is not affordable, while a callback on a slice
    already in memory costs no I/O at all. It is only called when the state
    carries an ``ensemble`` axis -- everything hanging off it is per member.
    """
    pieces = {name: [] for name in sensor_sets}
    for w, path in enumerate(state_paths):
        with xarray.open_dataset(path) as ds:
            t = (
                np.asarray(ds["time"].values, dtype=float)
                if "time" in ds.coords
                else None
            )
            for name, vel in _window_sensor_series(
                ds, sensor_sets, solver_name, on_member, w
            ).items():
                if t is not None and "time" in vel.dims:
                    vel = vel.assign_coords(time=(t - t[0]) + w * sim_time)
                pieces[name].append(vel)
    return _concat_sensor_pieces(pieces)


def _window_sensor_series(ds, sensor_sets, solver_name, on_member=None, window_index=0):
    """One window's ``{name: DataArray(component, ensemble, time, sensor)}``.

    Members are read and interpolated one at a time from the lazily-opened
    ``ds``; every sensor set is interpolated from the member slice already in
    memory, so a second set costs no extra read, and ``on_member`` (see
    :func:`ensemble_sensor_series`) sees the same slice.
    """
    if "ensemble" not in ds.dims:
        return {
            name: _sensor_component_timeseries(ds, ox, oy, oz, solver_name)
            for name, (ox, oy, oz) in sensor_sets.items()
        }

    members = {name: [] for name in sensor_sets}
    for m in range(ds.sizes["ensemble"]):
        # A slice, not an index: it keeps the ``ensemble`` dim, so the pieces
        # concatenate back into the layout the whole-file load produced.
        member = ds[["u", "v", "w"]].isel(ensemble=slice(m, m + 1)).load()
        for name, (ox, oy, oz) in sensor_sets.items():
            members[name].append(
                _sensor_component_timeseries(member, ox, oy, oz, solver_name)
            )
        if on_member is not None:
            on_member(window_index, m, member)
        # No ``member.close()``: ``.load()``'s result owns no file handle, so
        # the call would be a no-op that reads as if it released something. The
        # slice is dropped at the end of the iteration, which is the real thing.
    return {
        name: (parts[0] if len(parts) == 1 else xarray.concat(parts, dim="ensemble"))
        for name, parts in members.items()
    }


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
# Mean fields and resolved second moments (WP1.4, metrics doc section 4.1)
# ---------------------------------------------------------------------------

# Cell-centre dims, in (z, y, x) order, as they can appear after colocation:
# ``zt/yt/xt`` for uDALES, ``z/y/x`` for PALM, pylbm and the surrogate.
_CENTRE_DIM_CANDIDATES = (("z", "zt"), ("y", "yt"), ("x", "xt"))

# Evenly spaced z-levels the horizontal slabs are accumulated on. The same
# count (and the same selection rule) as the ``|U|`` RMSE slices, so the two
# state blocks describe the same heights.
_MEAN_FIELD_Z_SLICES = 4

# Ceiling on the accumulator state held for the *whole ensemble* during the
# pass. One member costs 80 bytes per slab cell (an int64 count, three float64
# means and six float64 co-moments), independent of how many frames go past, so
# at M=128 and four 512x512 slabs the unstrided cost would be 10 GB. The
# horizontal stride derived from this budget bounds it without a config knob;
# it is applied to the slabs only -- never to the station columns, which are a
# handful of cells, and never to time.
_MEAN_FIELD_BUDGET_BYTES = 1 << 30
_MEAN_FIELD_BYTES_PER_CELL = 80

# Points at which the truth's series is kept during its pass so the hit rate's
# absolute tolerance ``W`` can be block-bootstrapped from it. Kept per
# component: 64 points x a run's frames is a few hundred kB, against the whole
# field it is sampled from.
_W_SAMPLE_POINTS = 64
_W_SAMPLE_SEED = 0

# Quantiles stored at the station columns (figure S1's nested bands). The slabs
# get per-cell ensemble mean and std only -- quantiles over a full 3-D field
# would multiply ``eval_fields.nc`` by the number of levels stored here.
STATION_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _centre_dims(field):
    """``(z, y, x)`` dim names of a co-located field."""
    dims = []
    for candidates in _CENTRE_DIM_CANDIDATES:
        found = [d for d in candidates if d in field.dims]
        if not found:
            raise ValueError(
                f"co-located field has none of {candidates} among its dims "
                f"{tuple(field.dims)}"
            )
        dims.append(found[0])
    return tuple(dims)


def _drop_member_axis(field):
    """Drop the length-1 ``ensemble`` axis the per-member slices carry."""
    return field.isel(ensemble=0) if "ensemble" in field.dims else field


def _mean_field_stride(n_members, n_z, n_y, n_x):
    """Horizontal stride keeping the ensemble's accumulators inside the budget.

    A knob would be the obvious alternative, and WP1.3 deliberately shipped no
    metric knobs: nothing here needs to vary per run, and the only value a
    caller could sensibly pick is the one that fits in memory -- which is what
    this computes. Logged when it bites, so a strided run says so.
    """
    stride = 1
    while (
        n_members * n_z * -(-n_y // stride) * -(-n_x // stride)
    ) * _MEAN_FIELD_BYTES_PER_CELL > _MEAN_FIELD_BUDGET_BYTES and stride < min(
        n_y, n_x
    ):
        stride += 1
    if stride > 1:
        logger.info(
            "Mean fields: %d members x %d x %d x %d cells would exceed the "
            "%.1f GB accumulator budget -- scoring every %d-th cell horizontally",
            n_members,
            n_z,
            n_y,
            n_x,
            _MEAN_FIELD_BUDGET_BYTES / (1 << 30),
            stride,
        )
    return stride


class MeanFieldCollector:
    """Per-member time-mean velocity, TKE and ``<u'w'>`` on a fixed region.

    The ensemble half of WP1.4. One :class:`~evaluation.turbulence.
    MomentAccumulator` per member and region, fed from the member slices the
    sensor pass already materialises (see ``ensemble_sensor_series``'s
    ``on_member``), so the mean fields cost no extra read of the window files.

    Two regions, because the two consumers want different things. The
    **slabs** -- a few evenly spaced z-levels at full (or strided) horizontal
    resolution -- are what the hit rate scores and what figure F1 draws. The
    **station columns** -- full-depth profiles at the sensor (x, y) -- are what
    figure S1 plots, and they are small enough to carry per-member quantiles.

    Accumulation spans windows: a member's statistics cover every frame of every
    window it was fed, which is what makes them comparable with a truth pass
    over the same time range.

    Args:
        solver_name: The staggering the states follow, for colocation.
        station_x, station_y: Station coordinates (the sensor points), in domain
            coordinates.
        n_members: Ensemble size, used only to derive the horizontal stride.
        target: ``None`` to define the scored region from the first state seen
            (what the ensemble collector does); a target from another
            collector's :attr:`target` to sample onto *that* region instead,
            interpolating (what the truth collector does -- see :meth:`_slab`).
        keep_samples: Retain a fixed sample of slab cells' series so
            :meth:`sampling_tolerance` can bootstrap the hit rate's ``W`` from
            them. Only the truth collector needs it -- ``W`` is a property of
            the truth -- and it is off by default so the ensemble pass does not
            carry M copies of series nothing reads.

    Failures degrade rather than propagate: the sensor extraction shares the
    pass and must not be taken down by, say, a solver whose staggering
    colocation refuses, so a failure sets :attr:`reason`, stops accumulation and
    makes :meth:`result` return ``None``.
    """

    def __init__(
        self,
        solver_name,
        station_x,
        station_y,
        n_members=1,
        target=None,
        keep_samples=False,
    ):
        self.solver_name = solver_name
        self.station_x = np.asarray(station_x, dtype=float)
        self.station_y = np.asarray(station_y, dtype=float)
        self.n_members = int(n_members)
        self.target = target
        self.keep_samples = bool(keep_samples)
        self.reason = None
        self.failed = False
        self._moments = {}
        self._frames = {}
        self._windows = set()
        self._samples = []
        self._sample_index = None

    # -- accumulation -------------------------------------------------------

    def add_member(self, window_index, member_index, member_state):
        """``on_member`` callback: fold one member's window into its accumulators."""
        self.add(window_index, member_index, member_state)

    def add(self, window_index, member_index, state):
        """Fold one chunk of frames into ``member_index``'s accumulators."""
        if self.failed:
            return
        try:
            self._add(window_index, member_index, state)
        except (ValueError, KeyError) as error:
            self.failed = True
            self.reason = str(error)
            logger.warning(
                "Mean fields disabled after window %s member %s: %s",
                window_index,
                member_index,
                error,
            )

    def _add(self, window_index, member_index, state):
        components = [
            _drop_member_axis(field)
            for field in colocate_components(state, self.solver_name)
        ]
        if "time" not in components[0].dims:
            raise ValueError("state has no 'time' axis, so there is nothing to average")
        dims = _centre_dims(components[0])
        if self.target is None:
            self.target = self._derive_target(components[0], dims)

        slabs = [self._slab(field, dims) for field in components]
        columns = [self._columns(field, dims) for field in components]
        moments = self._moments.setdefault(
            member_index, {"slab": MomentAccumulator(), "station": MomentAccumulator()}
        )
        moments["slab"].update(*slabs)
        moments["station"].update(*columns)

        self._keep_samples(slabs)
        self._frames[member_index] = self._frames.get(member_index, 0) + int(
            components[0].sizes["time"]
        )
        self._windows.add(int(window_index))

    def _derive_target(self, field, dims):
        """Fix the scored region from the first state: the collector owns the grid."""
        zdim, ydim, xdim = dims
        z_index = evenly_spaced_levels(int(field.sizes[zdim]), _MEAN_FIELD_Z_SLICES)
        stride = _mean_field_stride(
            self.n_members,
            int(z_index.size),
            int(field.sizes[ydim]),
            int(field.sizes[xdim]),
        )
        native_z = np.asarray(field[zdim].values, dtype=float)
        native_y = np.asarray(field[ydim].values, dtype=float)
        native_x = np.asarray(field[xdim].values, dtype=float)
        return {
            "z": native_z[z_index],
            "y": native_y[::stride],
            "x": native_x[::stride],
            "station_z": native_z,
            "station_x": self.station_x,
            "station_y": self.station_y,
            "z_index": z_index,
            "stride": int(stride),
            # The full grid the target was cut from, so a later state can be
            # recognised as living on it (and sliced) rather than interpolated.
            "native": (native_z, native_y, native_x),
        }

    def _slab(self, field, dims):
        """``(time, zlev, y, x)`` on the target slab cells.

        Interpolation, always, when the target came from elsewhere: that is the
        truth collector, whose grid is a *different* grid in the general case
        (a PALM truth against a surrogate ensemble is the shipped default), and
        the metrics doc's rule is to interpolate the truth onto the assimilation
        grid before scoring. On identical grids linear interpolation returns the
        samples themselves, so the cross-grid path costs accuracy only where
        there is a genuine grid difference; target cells outside the truth's
        domain come back ``nan`` and are dropped by every consumer.
        """
        zdim, ydim, xdim = dims
        target = self.target
        if self._is_native(field, dims):
            stride = target["stride"]
            selected = field.isel(
                {
                    zdim: target["z_index"],
                    ydim: slice(None, None, stride),
                    xdim: slice(None, None, stride),
                }
            )
        else:
            selected = field.interp(
                {zdim: target["z"], ydim: target["y"], xdim: target["x"]}
            )
        return np.asarray(
            selected.transpose("time", zdim, ydim, xdim).values, dtype=float
        )

    def _columns(self, field, dims):
        """``(time, z, station)`` profiles at the station points.

        Always an interpolation: a station is an arbitrary (x, y) point, not a
        cell centre. For a foreign grid the vertical is interpolated too, so the
        truth's profiles land on the assimilation grid's levels.
        """
        zdim, ydim, xdim = dims
        target = self.target
        n_station = int(target["station_x"].size)
        if self._is_native(field, dims):
            points = {
                xdim: xarray.DataArray(target["station_x"], dims="station"),
                ydim: xarray.DataArray(target["station_y"], dims="station"),
            }
        else:
            n_z = int(target["station_z"].size)
            points = {
                zdim: xarray.DataArray(
                    np.repeat(target["station_z"][:, None], n_station, axis=1),
                    dims=("z_target", "station"),
                ),
                xdim: xarray.DataArray(
                    np.repeat(target["station_x"][None, :], n_z, axis=0),
                    dims=("z_target", "station"),
                ),
                ydim: xarray.DataArray(
                    np.repeat(target["station_y"][None, :], n_z, axis=0),
                    dims=("z_target", "station"),
                ),
            }
        sampled = field.interp(points)
        vertical = zdim if self._is_native(field, dims) else "z_target"
        return np.asarray(
            sampled.transpose("time", vertical, "station").values, dtype=float
        )

    def _is_native(self, field, dims):
        """Whether ``field`` lives on the very grid the target was cut from.

        Compared against the *full* stored axes rather than the strided target
        ones, so a grid that merely happens to share every fourth coordinate
        cannot be mistaken for the native one and sliced.
        """
        native = self.target["native"]
        return all(
            int(field.sizes[dim]) == axis.size
            and np.allclose(np.asarray(field[dim].values, dtype=float), axis)
            for dim, axis in zip(dims, native)
        )

    def _keep_samples(self, slabs):
        """Retain a fixed sample of slab cells' series for the ``W`` bootstrap.

        Inside the streaming pass because that is the only place the series
        exist; a fixed cell sample rather than the whole slab because ``W`` is a
        median over cells and 64 of them settle it.
        """
        if not self.keep_samples:
            return
        flat = [component.reshape(component.shape[0], -1) for component in slabs]
        if self._sample_index is None:
            n_cells = flat[0].shape[1]
            rng = np.random.default_rng(_W_SAMPLE_SEED)
            size = min(_W_SAMPLE_POINTS, n_cells)
            self._sample_index = np.sort(rng.choice(n_cells, size=size, replace=False))
        self._samples.append(
            np.stack([component[:, self._sample_index] for component in flat])
        )

    # -- reduction ----------------------------------------------------------

    def result(self):
        """Per-member fields, or ``None`` if nothing usable was accumulated.

        Returns:
            ``{"n_members", "n_windows", "frames_per_member", "target",
            "slab_mean" (M, 3, zlev, y, x), "slab_tke", "slab_uw",
            "station_mean" (M, 3, z, station), "station_tke", "station_uw"}``.
            ``slab_uw`` is ``<u'w'>``, the off-diagonal the metrics doc singles
            out because it discriminates anisotropy that TKE alone cannot.
        """
        if self.failed or not self._moments:
            return None
        members = sorted(self._moments)
        stacked = {}
        for region in ("slab", "station"):
            accumulators = [self._moments[m][region] for m in members]
            stacked[f"{region}_mean"] = np.stack([a.mean() for a in accumulators])
            stacked[f"{region}_tke"] = np.stack([a.tke() for a in accumulators])
            stacked[f"{region}_uw"] = np.stack(
                [a.reynolds_stress()[0, 2] for a in accumulators]
            )
        return {
            "n_members": len(members),
            "n_windows": len(self._windows),
            "frames_per_member": [int(self._frames[m]) for m in members],
            "target": self.target,
            **stacked,
        }

    def truth_pass(self, truth_access, n_chunks, chunk_frames):
        """Stream the truth through this collector, one window's frames at a time.

        Mirrors the window loop's memory discipline (only one chunk of the
        potentially multi-GB truth is held at once) and, because the collector
        was given the ensemble's target, lands the truth's statistics on the
        assimilation grid -- the metrics doc's cross-grid rule.
        """
        for chunk in range(n_chunks):
            ds = open_truth(
                truth_access["true_state_path"],
                truth_access["n_total"],
                truth_access["x_offset"],
                truth_access["start_idx"],
                truth_access["t_offset"],
            ).isel(time=slice(chunk * chunk_frames, (chunk + 1) * chunk_frames))
            if ds.sizes.get("time", 0):
                self.add(chunk, 0, ds)
            ds.close()
        return self

    def sampling_tolerance(self):
        """Per-component ``W``: the truth's own sampling error on its time-mean.

        The block-bootstrap standard error of the time-mean at each sampled slab
        cell (which is ``sigma_u/sqrt(N_eff)`` with the correlation kept),
        reduced by the **median** over cells: a single cell inside a
        recirculation can carry a wild floor, and ``W`` is meant to describe a
        typical one. ``nan`` per component where no floor could be measured --
        short runs cannot be blocked, which includes the CI smoke shape -- and
        :func:`~evaluation.scores.hit_rate` reads that as "relative test only".
        """
        if not self._samples:
            return None
        series = np.concatenate(self._samples, axis=1)  # (component, time, point)
        floors = block_bootstrap_std(np.moveaxis(series, 1, -1))  # (component, point)
        with warnings.catch_warnings():
            # Every sampled cell unmeasurable (a run too short to block) is the
            # common case in the smoke shape, not a fault.
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(floors, axis=1)


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
