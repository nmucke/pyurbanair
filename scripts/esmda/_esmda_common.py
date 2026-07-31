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
    fair_energy_score,
    max_abs_zscore_reference,
    max_nominal_alpha,
    pit_rank,
    rank_histogram,
    rank_histogram_weights,
    zscore,
    zscore_exceedance,
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
