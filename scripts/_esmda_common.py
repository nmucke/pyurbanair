"""Shared post-processing helpers for the ESMDA three-script pipeline.

``scripts/run_esmda.py`` runs the assimilation and saves the raw artifacts;
``scripts/compute_esmda_metrics.py`` turns those into ``run_summary.yaml`` and
``scripts/make_esmda_figures.py`` draws the figures. The metric and figure
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

import pathlib

import numpy as np
import xarray
import yaml
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_operator import ObservationOperator
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_validation_points,
)
from pyurbanair.plotting import compute_parameter_metrics, compute_sensor_metrics

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


def ensemble_sensor_series(state_paths, sensor_sets, solver_name, sim_time):
    """Ensemble per-component ``(u, v, w)`` sensor series across rollout windows.

    Opens each window's full-ensemble state file once and interpolates u/v/w at
    every sensor set's points (keeping ``component`` + ``ensemble`` + ``time``),
    rebasing each window's local time onto a single global axis (window ``w``
    starts at ``w*sim_time``) so it lines up with the truth. Returns
    ``{name: DataArray(component, ensemble, time, sensor)}``.
    """
    pieces = {name: [] for name in sensor_sets}
    for w, path in enumerate(state_paths):
        ds = xarray.open_dataset(path).load()
        t = np.asarray(ds["time"].values, dtype=float) if "time" in ds.coords else None
        for name, (ox, oy, oz) in sensor_sets.items():
            vel = _sensor_component_timeseries(ds, ox, oy, oz, solver_name)
            if t is not None and "time" in vel.dims:
                vel = vel.assign_coords(time=(t - t[0]) + w * sim_time)
            pieces[name].append(vel)
        ds.close()
    return _concat_sensor_pieces(pieces)


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

    The energy score is the multivariate generalization of the CRPS (Gneiting &
    Raftery 2007): for a vector forecast ensemble ``{v_m}`` and truth ``v``,

        ES = mean_m ||v_m - v|| - 0.5 * mean_{m,m'} ||v_m - v_{m'}||,

    which reduces to the CRPS in 1-D. It rewards both accuracy (term 1) and a
    calibrated spread (term 2), in the same |U| units as the velocity.

    Args:
        members: ``(component, ensemble, time, sensor)`` aligned member vectors.
        truth: ``(component, time, sensor)`` aligned truth vectors.

    Returns:
        ``(time,)`` energy score, averaged over the sensors (matching the
        per-time, over-sensors reduction of ``compute_sensor_metrics``).
    """
    n_time = members.shape[2]
    es = np.empty(n_time)
    # Loop over time so the pairwise term never materializes more than
    # ``(component, ensemble, ensemble, sensor)`` at once.
    for t in range(n_time):
        m = members[:, :, t, :]  # (C, E, S)
        v = truth[:, t, :]  # (C, S)
        d_truth = np.sqrt(np.sum((m - v[:, None, :]) ** 2, axis=0))  # (E, S)
        term1 = d_truth.mean(axis=0)  # (S,)
        diff = m[:, :, None, :] - m[:, None, :, :]  # (C, E, E, S)
        d_pair = np.sqrt(np.sum(diff**2, axis=0))  # (E, E, S)
        term2 = 0.5 * d_pair.mean(axis=(0, 1))  # (S,)
        es[t] = float((term1 - term2).mean())  # average over sensors
    return es


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
    """Per-parameter RMSE/CRPS summary stats (posterior, with a prior reference)."""
    metrics = compute_parameter_metrics(posterior_params, true_params, prior_params)
    summary = {}
    for name, m in metrics.items():
        entry = {"rmse": series_stats(m["rmse"]), "crps": series_stats(m["crps"])}
        if "prior_rmse" in m:
            prior_mean = float(np.nanmean(m["prior_rmse"]))
            post_mean = float(np.nanmean(m["rmse"]))
            entry["prior_rmse_mean"] = prior_mean
            entry["rmse_reduction_vs_prior"] = (
                float(1.0 - post_mean / prior_mean) if prior_mean > 0 else None
            )
        summary[name] = entry
    return summary
