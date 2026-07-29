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
    _param_members_and_x,
    _plotted_param_names,
    compute_parameter_metrics,
    compute_sensor_metrics,
)
from pyurbanair.utils.ensemble_scores import (
    coverage,
    fair_energy_score,
    pit_rank,
    rank_histogram,
    zscore,
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

    Sensor-shaped adapter around
    :func:`pyurbanair.utils.ensemble_scores.fair_energy_score` (the
    multivariate CRPS; see there for the estimator). The only work done here
    is the axis convention: the sensor series carry the component axis first,
    the shared score wants ensemble first and components last. The shared
    implementation keeps the same memory bound -- it loops over the leading
    batch axis, which after the move is time, so the pairwise term never
    materializes more than ``(ensemble, ensemble, sensor)`` at once.

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
        if "prior_crps" in m:
            prior_mean = float(np.nanmean(m["prior_crps"]))
            post_mean = float(np.nanmean(m["crps"]))
            entry["prior_crps_mean"] = prior_mean
            entry["crps_reduction_vs_prior"] = (
                float(1.0 - post_mean / prior_mean) if prior_mean > 0 else None
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
# line, so the K = 42 of a routine 2-parameter/21-knot run would add ~3.5k lines
# to a ~100-line summary, and production cases are larger still. Above the cap
# the matrices are omitted and `corr_summary` carries the off-diagonal scalars.
JOINT_CORR_MAX_K = 16

# Same reasoning for the eigenvalue list, which is only `r` long.
JOINT_EIGENVALUE_MAX = 64

# How many parameter-vector entries to report per eigenvector direction.
JOINT_LOADINGS = 5

# Tie-breaking seed for `pit_rank`, recorded in the summary so a reader can
# reproduce the counts.
PIT_TIE_SEED = 0


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
    performs (``_param_members_and_x`` + ``np.interp`` onto the posterior
    x-axis), reused rather than re-derived: it already handles a static truth
    (one point -> constant) and a time-varying truth sampled on a different
    knot grid (the routine case -- e.g. 19 truth knots against 21 posterior
    knots) with the same two lines.

    Yields ``(name, members, truth, prior_members, x_is_time)`` where
    ``members`` is ``(n_members, n_x)`` and ``prior_members`` is ``None`` unless
    the prior exists on the same x-axis.

    ``x_is_time`` is the per-parameter dynamic/static discriminator -- there is
    no run-level split, because a single ``conf/params/dynamic.yaml`` run mounts
    time-varying ``external_parameters`` AND a ``static_parameters`` block into
    one Dataset, so both PIT branches fire in the same run. The test is
    ``_param_members_and_x``'s own, repeated here because that helper returns
    the axis but not the reason for it: a **``time`` coordinate**, not merely a
    ``time`` dimension. The dimension alone is not enough -- ``_concat_windows``
    stacks per-window parameter files along ``time``, so in a purely static run
    every parameter comes out with a length-``num_windows`` ``time`` dimension
    and no ``time`` coordinate, and its x-axis is the window index.
    """
    for name in _plotted_param_names(posterior_params, true_params):
        posterior_da = posterior_params[name]
        x_est, members = _param_members_and_x(posterior_da)

        true_da = true_params[name]
        if "ensemble" in true_da.dims:
            true_da = true_da.isel(ensemble=0)
        x_true, true_members = _param_members_and_x(true_da.expand_dims("ensemble"))
        order = np.argsort(x_true)
        truth = np.interp(x_est, np.asarray(x_true)[order], true_members[0][order])

        prior_members = None
        if prior_params is not None and name in prior_params.data_vars:
            _, candidate = _param_members_and_x(prior_params[name])
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


def _n_constant_segments(members):
    """Number of piecewise-constant runs along the x-axis of ``(M, n_x)`` members.

    A parameter that is static but rides on a dynamic run's time axis (the
    ``static_parameters`` block of ``conf/params/dynamic.yaml`` is broadcast to
    every knot by ``_concat_windows``) is piecewise constant with one step per
    assimilation window. Counting the steps recovers that window count without
    needing to know which block the parameter came from, and is a no-op for a
    genuinely time-varying parameter, whose every knot differs.
    """
    if members.shape[1] < 2:
        return int(members.shape[1])
    steps = ~np.isclose(members[:, 1:], members[:, :-1])
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
    """``{mean, std, max_abs, overconfident}`` of the per-knot z-scores."""
    z = zscore(members, truth)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return None
    max_abs = float(np.max(np.abs(finite)))
    return {
        "mean": _finite_or_none(finite.mean()),
        # ddof=1 over knots: one knot gives no spread estimate, not a zero one.
        "std": _finite_or_none(finite.std(ddof=1)) if finite.size > 1 else None,
        "max_abs": _finite_or_none(max_abs),
        "overconfident": bool(max_abs > 3.0),
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
    }
    return [int(c) for c in counts], meta


def _coverage_block(members, truth):
    """Order-statistic coverage at each ``COVERAGE_ALPHAS`` level."""
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
    if not np.any(valid):
        return None
    n_members = members.shape[0]
    block = {
        f"alpha_{int(round(alpha * 100))}": _finite_or_none(
            coverage(members[:, valid], truth[valid], alpha=alpha)
        )
        for alpha in COVERAGE_ALPHAS
    }
    # The band edges are member order statistics, so with M members the widest
    # band available is [x_(1), x_(M)], whose nominal level is (M-1)/(M+1) --
    # 0.94 at M = 32, 0.6 at M = 4. Requesting alpha above that silently clamps,
    # so the ceiling is reported next to the numbers rather than letting a
    # capped `alpha_90` read as a calibration failure. It bounds the NOMINAL
    # level, not the realized fraction, which can sit above it by chance.
    block["max_nominal_alpha"] = _finite_or_none((n_members - 1) / (n_members + 1))
    return block


def _contraction_block(members, prior_members):
    """``{mean, min}`` of the per-knot posterior/prior spread ratio."""
    if prior_members is None:
        return None
    post_std = members.std(axis=0, ddof=1)
    prior_std = prior_members.std(axis=0, ddof=1)
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


def parameter_bundle_summary(
    base_summary,
    posterior_params,
    true_params,
    prior_params,
    *,
    posterior_flat,
    prior_flat,
    cfg=None,
    num_windows=None,
):
    """Add the WP1.1 calibration bundle to an existing ``parameter_metrics`` mapping.

    Purely additive: every key :func:`parameter_metric_summary` wrote is copied
    through untouched (their paths are hard-coded in
    ``scripts/figure_creation/``), each per-parameter entry gains ``zscore``,
    ``pit_counts``/``pit``, ``coverage`` and ``contraction_ratio``, and a
    sibling ``joint`` entry is added.

    ``posterior_flat`` / ``prior_flat`` are the ``(M, K)`` matrices from
    ``compute_esmda_metrics._flatten_parameter_members``; they are passed in
    rather than recomputed so the pipeline keeps exactly one flattener.

    Args:
        base_summary: The mapping returned by :func:`parameter_metric_summary`.
        posterior_params: Posterior parameter Dataset.
        true_params: Truth parameter Dataset (any knot grid).
        prior_params: Prior parameter Dataset, or ``None``.
        posterior_flat: ``(M, K)`` flattened posterior members.
        prior_flat: ``(M, K)`` flattened prior members, or ``None``.
        cfg: The run's saved config, read for the GP knot correlation.
        num_windows: Assimilation window count, the pooling depth for
            parameters without a ``time`` dimension.

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

    for name, members, truth, prior_members, x_is_time in _aligned_parameter_arrays(
        posterior_params, true_params, prior_params
    ):
        entry = summary.setdefault(name, {})
        if degenerate:
            entry.update(
                {
                    "zscore": None,
                    "pit_counts": None,
                    "pit": None,
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
                "coverage": _coverage_block(members, truth),
                "contraction_ratio": _contraction_block(members, prior_members),
            }
        )

    summary["joint"] = joint_parameter_directions(
        posterior_flat, prior_flat, parameter_vector_labels(posterior_params)
    )
    return summary
