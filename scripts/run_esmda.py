"""Run ESMDA: parameter-only / joint state+parameter, static / time-varying
parameters, single-window / multi-window rollout, with truth simulated inline
or loaded from disk.

This single script replaces the former
run_{parameter,state_and_parameter,rollout,time_varying_parameter,
time_varying_parameters_rollout}_esmda.py family. Three declarative axes select
the mode (see conf/run_esmda.yaml):

  * ``esmda/smoother=parameter|state_and_parameter|time_varying``
        which augmented state the Kalman update acts on.
  * ``params@prior_params=static|dynamic``
        static scalar parameters vs a time-varying (AR(2)) prior.
  * ``esmda.num_assimilation_windows=1|N``
        a single assimilation window vs an N-window rollout.

and the truth source:

  * ``run.ground_truth_dir=null``    simulate the truth inline (default).
  * ``run.ground_truth_dir=<path>``  load a state.nc/params.nc artifact written
                                     by run_forward_model.py run.time_varying=true.

Truth (states + parameters) for every window is generated up front, before any
assimilation runs. The window loop then consumes the precomputed truth.

Examples::

    python scripts/run_esmda.py esmda/smoother=parameter \
        params@prior_params=static params@truth_params=static_truth
    python scripts/run_esmda.py esmda/smoother=state_and_parameter \
        params@prior_params=static esmda.num_assimilation_windows=3
    python scripts/run_esmda.py esmda/smoother=time_varying \
        params@prior_params=dynamic params@truth_params=dynamic_truth \
        esmda.num_assimilation_windows=3
"""

import pathlib
import sys
import time

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_operator import ObservationOperator
from data_assimilation.smoothing.esmda import (
    StateAndParameterESMDA,
)
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_observation_operator,
    create_observation_points,
    create_validation_points,
    resolve_output_dir,
)
from pyurbanair.plotting import (
    compute_parameter_metrics,
    compute_sensor_metrics,
    plot_final_state_with_obs,
    plot_parameter_error,
    plot_rollout_time_evolution,
    plot_sensor_timeseries,
)
from pyurbanair.utils.animation_utils import animate_rollout_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, get_ensemble_mean_field


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Small helpers (in the style of run_forward_model.py)
# ---------------------------------------------------------------------------

def _concat_windows(paths, sim_time, rebase, transform=None):
    """Concatenate per-window NetCDF files along ``time``.

    Files are opened one at a time and (optionally) reduced by ``transform``
    before being appended, so a memory-heavy per-window reduction (e.g. taking
    the ensemble mean of a state field) keeps only the reduced result instead of
    holding every window's full data in memory at once.

    ``rebase`` (used for the time-varying case) shifts each window's local time
    onto a single monotonic global axis (window ``w`` starts at ``w*sim_time``);
    the static case stacks the windows as-is, matching the old rollout script.
    """
    pieces = []
    for w, path in enumerate(paths):
        ds = xarray.open_dataset(path).load()
        if transform is not None:
            ds = transform(ds)
        if rebase and "time" in ds.dims:
            t = np.asarray(ds["time"].values, dtype=float)
            ds = ds.assign_coords(time=(t - t[0]) + w * sim_time)
        pieces.append(ds)
    if len(pieces) == 1:
        return pieces[0]
    return xarray.concat(pieces, dim="time", join="override")


# ---------------------------------------------------------------------------
# Lazy truth-state access (only load the slices that are actually needed)
# ---------------------------------------------------------------------------

_Z_DIMS = ("z", "zm", "zt")
_X_COORDS = ("x", "xt", "xm")


def _truth_x_min(ds):
    """Smallest x face/centre coordinate of a (truth) dataset, or 0.0 if none.

    Prefers the staggered face coordinate (``xm``) so the domain *edges* align
    when computing the offset; falls back to ``x``/``xt`` for non-udales grids.
    """
    for c in ("xm", "x", "xt"):
        if c in ds.coords:
            return float(np.asarray(ds[c].values).min())
    return 0.0


def _open_truth(true_state_path, n_total, x_offset=0.0, start_idx=0, t_offset=0.0):
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


def _select_z_plane(ds, z_level):
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


def _streaming_state_rmse(true_state, esmda_state, n_z_slices=4):
    """Per-timestep RMSE of |U| between truth and the ensemble-mean state.

    Streams over ``n_z_slices`` evenly-spaced z-levels and all time steps rather
    than materialising the full 4-D velocity field. When the truth and
    assimilation grids differ, the truth planes are interpolated onto the
    assimilation grid before differencing.
    """
    true_s = true_state.mean(dim="ensemble") if "ensemble" in true_state.dims else true_state
    esmda_s = esmda_state.mean(dim="ensemble") if "ensemble" in esmda_state.dims else esmda_state

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
    diff = (
        np.asarray(true_vel.isel(zlev=slice(0, nz_common)).values)
        - np.asarray(esmda_vel.isel(zlev=slice(0, nz_common)).values)
    )
    return np.sqrt(np.nanmean(diff ** 2, axis=tuple(range(1, diff.ndim))))


# ---------------------------------------------------------------------------
# Sensor time-series extraction (truth vs ensemble at fixed points)
# ---------------------------------------------------------------------------

def _sensor_vel_timeseries(state, obs_x, obs_y, obs_z, solver_name):
    """Velocity-magnitude time series at each sensor point.

    Trilinearly interpolates u/v/w (each on its own staggered grid, resolved via
    an ``ObservationOperator``'s solver-specific dim mapping) at the sensor
    locations, keeping any leading dims (``ensemble``, ``time``), and combines
    them into |U|. Returns a DataArray with dims ``(..., time, sensor)``.
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
                x_dim=dims["x"], y_dim=dims["y"], z_dim=dims["z"],
                obs_x=op.obs_x, obs_y=op.obs_y, obs_z=op.obs_z,
            )
        )
    return np.sqrt(comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2)


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


def _ensemble_sensor_series(state_paths, sensor_sets, solver_name, sim_time):
    """Ensemble |U| sensor series across rollout windows.

    Opens each window's full-ensemble state file once and interpolates |U| at
    every sensor set's points (keeping ``ensemble`` + ``time``), rebasing each
    window's local time onto a single global axis (window ``w`` starts at
    ``w*sim_time``) so it lines up with the truth. Returns
    ``{name: DataArray(ensemble, time, sensor)}``.
    """
    pieces = {name: [] for name in sensor_sets}
    for w, path in enumerate(state_paths):
        ds = xarray.open_dataset(path).load()
        t = np.asarray(ds["time"].values, dtype=float) if "time" in ds.coords else None
        for name, (ox, oy, oz) in sensor_sets.items():
            vel = _sensor_vel_timeseries(ds, ox, oy, oz, solver_name)
            if t is not None and "time" in vel.dims:
                vel = vel.assign_coords(time=(t - t[0]) + w * sim_time)
            pieces[name].append(vel)
        ds.close()
    return _concat_sensor_pieces(pieces)


def _truth_sensor_series(
    true_state_path, n_total, x_offset, start_idx, t_offset,
    sensor_sets, solver_name, num_windows, n_per_window,
):
    """Truth |U| sensor series, read one assimilation window at a time.

    Mirrors the window loop's memory discipline: only one window's worth of the
    (potentially multi-GB) truth is held at once. The truth's ``time`` axis is
    already global, so the per-window pieces concatenate directly.
    """
    pieces = {name: [] for name in sensor_sets}
    for w in range(num_windows):
        ts = _open_truth(true_state_path, n_total, x_offset, start_idx, t_offset).isel(
            time=slice(w * n_per_window, (w + 1) * n_per_window)
        )
        for name, (ox, oy, oz) in sensor_sets.items():
            pieces[name].append(_sensor_vel_timeseries(ts, ox, oy, oz, solver_name))
        ts.close()
    return _concat_sensor_pieces(pieces)


# ---------------------------------------------------------------------------
# Run summary (YAML)
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


def _write_yaml(data, path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(_to_native(data), f, sort_keys=False, default_flow_style=False)


def _series_stats(arr):
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


def _parameter_metric_summary(posterior_params, true_params, prior_params):
    """Per-parameter RMSE/CRPS summary stats (posterior, with a prior reference)."""
    metrics = compute_parameter_metrics(posterior_params, true_params, prior_params)
    summary = {}
    for name, m in metrics.items():
        entry = {"rmse": _series_stats(m["rmse"]), "crps": _series_stats(m["crps"])}
        if "prior_rmse" in m:
            prior_mean = float(np.nanmean(m["prior_rmse"]))
            post_mean = float(np.nanmean(m["rmse"]))
            entry["prior_rmse_mean"] = prior_mean
            entry["rmse_reduction_vs_prior"] = (
                float(1.0 - post_mean / prior_mean) if prior_mean > 0 else None
            )
        summary[name] = entry
    return summary


# ---------------------------------------------------------------------------
# Output / plotting
# ---------------------------------------------------------------------------

def _finish_rollout(
    cfg, out_dir, windows_dir, num_windows, sim_time, is_dynamic,
    true_state_path, n_total, x_offset=0.0, start_idx=0, t_offset=0.0,
    run_info=None,
):
    """Assemble rollout outputs from the per-window files in ``windows_dir``.

    State files are reduced one window at a time to the ensemble mean state plus
    the ensemble mean/std of the velocity magnitude, so the full ensemble is
    never held across windows. Parameter files are small enough to load in full,
    so the whole ensemble is kept in memory for distribution plotting.

    ``run_info`` (run metadata + ESMDA timing assembled by the caller) seeds the
    ``run_summary.yaml`` written alongside the figures; this function augments it
    with the parameter, state and sensor estimation metrics it computes here.
    """
    state_paths = [
        windows_dir / f"window_{w}_posterior_state.nc" for w in range(num_windows)
    ]
    posterior_param_paths = [
        windows_dir / f"window_{w}_posterior_params.nc" for w in range(num_windows)
    ]
    prior_param_paths = [
        windows_dir / f"window_{w}_prior_params.nc" for w in range(num_windows)
    ]

    # Parameters: full ensemble in memory.
    posterior_params = _concat_windows(posterior_param_paths, sim_time, rebase=is_dynamic)
    prior_params = _concat_windows(prior_param_paths, sim_time, rebase=is_dynamic)
    posterior_params.to_netcdf(out_dir / "posterior_params.nc")
    prior_params.to_netcdf(out_dir / "prior_params.nc")

    # States: reduce each window's ensemble in a single pass before
    # concatenating, so the full ensemble is never held across windows. Keep the
    # mean state (u/v/w) plus the ensemble mean/std of the velocity magnitude
    # (``vel_mean``/``vel_std``).
    def _state_summary(ds):
        vmag = add_velocity_magnitude(ds)["vel_magnitude"]
        reduced = ds.mean(dim="ensemble")
        reduced["vel_mean"] = vmag.mean(dim="ensemble")
        reduced["vel_std"] = vmag.std(dim="ensemble")
        return reduced

    posterior_state = _concat_windows(
        state_paths, sim_time, rebase=is_dynamic, transform=_state_summary
    )
    posterior_state.to_netcdf(out_dir / "posterior_state_mean.nc")

    # Run summary: start from the caller's run metadata/timing and add the
    # parameter estimation metrics (always available). State and sensor metrics
    # are added below once the truth has been opened (skipped with the figures).
    true_params = xarray.open_dataset(out_dir / "true_params.nc")
    summary = dict(run_info or {})
    summary["parameter_metrics"] = _parameter_metric_summary(
        posterior_params, true_params, prior_params
    )

    if cfg.run.skip_viz:
        _write_yaml(summary, out_dir / "run_summary.yaml")
        print(f"Saved run summary in {out_dir / 'run_summary.yaml'}")
        return

    # Open the (potentially multi-GB) truth lazily. The plots below each pull
    # only the slice they need: a single z-plane for the animation/final state
    # and a few z-slices for the streamed error curve.
    true_state = _open_truth(true_state_path, n_total, x_offset, start_idx, t_offset)
    obs_x, obs_y, _ = create_observation_points(cfg.obs)

    # Truth reduced to the z=0 horizontal plane (single layer kept), so the
    # animation and final-state plot never load the full 3-D velocity field.
    true_state_plane = _select_z_plane(true_state, z_level=0)
    true_vel = add_velocity_magnitude(true_state_plane)["vel_magnitude"]

    # State error: stream over 4 z-slices and all time steps instead of the
    # whole 4-D field (interpolating onto the assim grid if coords differ).
    rmse = _streaming_state_rmse(true_state, posterior_state)
    summary["state_metrics"] = {"vel_magnitude_rmse": _series_stats(rmse)}

    # Boundaries between assimilation windows on the (rebased) global time axis,
    # used to lightly shade alternating windows in the parameter plot.
    window_edges = (
        list(np.linspace(0.0, sim_time * num_windows, num_windows + 1))
        if is_dynamic and num_windows > 1
        else None
    )

    plot_rollout_time_evolution(
        esmda_params=posterior_params,
        true_params=true_params,
        esmda_state=None,
        true_state=None,
        output_path=out_dir / "rollout_time_evolution.png",
        prior_params=prior_params,
        window_edges=window_edges,
        rmse=rmse,
    )
    plot_parameter_error(
        esmda_params=posterior_params,
        true_params=true_params,
        output_path=out_dir / "parameter_error.png",
        window_edges=window_edges,
    )
    animate_rollout_state(
        true_state=true_state_plane,
        mean_vel=posterior_state["vel_mean"],
        std_vel=posterior_state["vel_std"],
        output_path=out_dir / "rollout_animation.mp4",
        z_level=0,
    )
    plot_final_state_with_obs(
        mean_vel=posterior_state["vel_mean"],
        std_vel=posterior_state["vel_std"],
        output_path=out_dir / "final_state_with_obs.png",
        true_vel=true_vel,
        obs_x=obs_x,
        obs_y=obs_y,
        z_level=0,
    )

    # Sensor time series: true vs ensemble |U| at the assimilation sensors and at
    # a held-out validation set, each with an RMSE/CRPS skill panel. The truth
    # and ensemble are interpolated at the same physical points on their own
    # grids, so the two figures are grid-independent. Extracted one window at a
    # time to keep the (potentially multi-GB) truth and full ensemble out of
    # memory in full.
    _, _, obs_z = create_observation_points(cfg.obs)
    sensor_sets = {"assimilation": (obs_x, obs_y, obs_z)}
    validation_points = create_validation_points(cfg.obs)
    if validation_points is not None:
        sensor_sets["validation"] = validation_points

    n_per_window = n_total // max(num_windows, 1)
    truth_series = _truth_sensor_series(
        true_state_path, n_total, x_offset, start_idx, t_offset,
        sensor_sets, cfg.truth_model.solver_name, num_windows, n_per_window,
    )
    ensemble_series = _ensemble_sensor_series(
        state_paths, sensor_sets, cfg.assim_model.solver_name, sim_time,
    )

    sensor_metrics = {}
    for name, (sx, sy, sz) in sensor_sets.items():
        plot_sensor_timeseries(
            true_sensor=truth_series[name],
            ensemble_sensor=ensemble_series[name],
            output_path=out_dir / f"sensor_timeseries_{name}.png",
            title=f"State at {name} sensors",
            sensor_x=sx,
            sensor_y=sy,
            sensor_z=sz,
        )
        m = compute_sensor_metrics(truth_series[name], ensemble_series[name])
        sensor_metrics[name] = {
            "num_sensors": int(np.asarray(sx).size),
            "vel_magnitude_rmse": _series_stats(m["rmse"]),
            "vel_magnitude_crps": _series_stats(m["crps"]),
        }
    summary["sensor_metrics"] = sensor_metrics

    _write_yaml(summary, out_dir / "run_summary.yaml")
    print(f"Saved run summary in {out_dir / 'run_summary.yaml'}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    num_windows = int(cfg.esmda.num_assimilation_windows)
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    is_dynamic = "time_coords" in list(cfg.truth_params.keys())
    final_time = sim_time * num_windows
    domain_x_min = float(cfg.domain.bounds[0][0])
    rng_key = jax.random.PRNGKey(cfg.esmda.seed)

    # --- Output and windows dir ---------------------------------------------------------
    out_dir = resolve_output_dir(cfg, "esmda")
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"

    # Persist the raw composed Hydra config used to launch this run (interpolations
    # intact), so the run is fully reproducible from the output folder alone.
    OmegaConf.save(config=cfg, f=out_dir / "config.yaml")

    # --- True parameter sampler and state -----------------------------------------------------------
    # ``true_state_path`` points at the truth on disk; ``n_total`` is the number
    # of truth frames inside the assimilation horizon [0, final_time). The state
    # itself is opened lazily everywhere downstream (see ``_open_truth``) so a
    # large truth is never held in memory in full.
    if cfg.run.truth_dir is None:
        truth_sampler = instantiate(cfg.truth_params)
        if num_windows > 0:
            true_forward_model = instantiate(
                cfg.truth_model.forward_model,
                results_dir=None,
                simulation_time=sim_time * num_windows
            )
            if is_dynamic:
                # Match the assimilation grid: each window owns `num` control
                # points spaced `sim_time/(num-1)` (conf/params/dynamic.yaml's
                # per-window linspace(0, sim_time, num)). Sampling the truth on
                # the same spacing puts its control points on the window
                # boundaries instead of on a `num*num_windows`-point grid whose
                # spacing (sim_time*num_windows/(num*num_windows-1)) drifts off
                # the window edges, so truth and assim curves share x-locations.
                num = cfg.truth_params.time_coords.num
                time_coords = jnp.linspace(
                    0, sim_time * num_windows, (num - 1) * num_windows + 1
                )
                truth_sampler = instantiate(cfg.truth_params, time_coords=time_coords)
        else:
            true_forward_model = instantiate(
                cfg.truth_model.forward_model,
                results_dir=None,
                simulation_time=sim_time
            )

        true_params = truth_sampler.sample(1)

        instantiate(cfg.truth_model.prepare, forward_model=true_forward_model)
        clean_outputs(model_name=cfg.truth_model.name, forward_model=true_forward_model)
        true_state = true_forward_model(params=true_params.isel(ensemble=0))

        # Persist the simulated truth so the window loop and final plots can
        # re-open it lazily (one window / one z-plane at a time) instead of
        # holding the whole rollout in memory.
        true_state_path = out_dir / "true_state.nc"
        true_state.to_netcdf(true_state_path)
        n_total = int(true_state.sizes["time"])
        # Inline truth is already simulated on the domain grid, starting at t=0.
        x_offset = domain_x_min - _truth_x_min(true_state)
        start_idx = 0
        t_offset = 0.0
        del true_state

    else:
        truth_dir = pathlib.Path(cfg.run.truth_dir)
        true_params = xarray.load_dataset(truth_dir / "params.nc")

        # Optionally begin the assimilation horizon partway into a pre-simulated
        # truth (e.g. skip a spin-up): keep the truth from ``start_time`` onward
        # and shift its time axis so ``start_time`` becomes t=0. Both the state
        # and the parameters are sliced and rebased by the same offset.
        start_time = float(cfg.run.truth_start_time or 0.0)

        if "time" in true_params.dims:
            if start_time:
                true_params = true_params.sel(time=true_params.time >= start_time)
                true_params = true_params.assign_coords(
                    time=true_params.time - start_time
                )
            true_params = true_params.sel(time=true_params.time < final_time)

        # The truth state can be very large (tens of GB). Reference the existing
        # artifact directly -- never load or copy the whole file -- and only read
        # the frames inside the assimilation horizon [start_time, start_time +
        # final_time).
        true_state_path = truth_dir / "state.nc"
        with xarray.open_dataset(true_state_path) as _truth_meta:
            true_times = np.asarray(_truth_meta["time"].values, dtype=float)
            # Shift the truth's x onto the domain frame: a truth saved in its own
            # coordinates (e.g. x in [0, 100]) is offset so its upstream edge
            # lines up with the domain's (e.g. x_min = -20). Other axes (y, z)
            # already share the domain origin.
            x_offset = domain_x_min - _truth_x_min(_truth_meta)
        # Drop the frames before ``start_time`` and rebase the kept frames onto a
        # [0, final_time) axis (t_offset = start_time), so frame-index slicing in
        # the window loop counts from the chosen start.
        start_idx = int((true_times < start_time).sum())
        t_offset = start_time
        n_total = int(((true_times[start_idx:] - t_offset) < final_time).sum())

    if x_offset:
        print(f"Shifting truth x by {x_offset:+g} to align with domain x_min={domain_x_min:g}")

    # Number of truth frames per assimilation window (contiguous, half-open).
    n_per_window = n_total // max(num_windows, 1)

    # Save the (small) truth parameters for the final plots.
    true_params.to_netcdf(out_dir / "true_params.nc")

    # --- Assimilation ensemble model -------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    assim_model = instantiate(
        cfg.assim_model.forward_model, results_dir=assim_results_dir
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model, forward_model=assim_model
    )

    # --- Prior parameter sampler -----------------------------------------------------------
    prior_sampler = instantiate(cfg.prior_params)
    prior_params = prior_sampler.sample(ensemble_size)

    # --- Observation operator -----------------------------------------------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    # --- Observation error covariance ---------
    # Truth frames sit on a uniform grid over [0, sim_time*num_windows); each
    # window owns exactly `n_per_window` of them. Size C_D from the first such
    # block so it matches every window's observation vector (and the per-window
    # count the assimilation model emits). Opened lazily and sliced, so only the
    # first window's frames are read.
    truth_first_window = _open_truth(
        true_state_path, n_total, x_offset, start_idx, t_offset
    ).isel(time=slice(0, n_per_window))
    obs = jnp.asarray(truth_obs_op(truth_first_window))
    truth_first_window.close()
    C_D = jnp.diag((cfg.esmda.obs_error_std**2) * jnp.ones(obs.shape[0]))

    # --- Smoother -----------------------------------------------------------
    rng_key, esmda_key = jax.random.split(rng_key)
    esmda = instantiate(
        cfg.esmda.smoother,
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
        rng_key=esmda_key,
    )
    include_state = isinstance(esmda, StateAndParameterESMDA)

    # --- Run ESMDA -----------------------------------------------------------
    # Time the assimilation. ``window_seconds`` is each window's full wall-clock
    # cost (observation extraction + Kalman solve + I/O + extrapolation);
    # ``solve_seconds`` isolates the ESMDA Kalman solve itself.
    state_input = None
    window_seconds: list[float] = []
    solve_seconds: list[float] = []
    esmda_start = time.perf_counter()
    for window in tqdm(range(num_windows)):
        window_start = time.perf_counter()
        windows_dir.mkdir(parents=True, exist_ok=True)
        prior_params.to_netcdf(windows_dir / f"window_{window}_prior_params.nc")

        # Pin the t=0 knot from window 1 onward so the Kalman update preserves
        # the cross-window continuity that the GP extrapolation established at
        # each window boundary. Window 0's prior t=0 is just a cold-start GP
        # draw (over a spun-up flow), so ESMDA is free to fit it. Only the
        # time-varying smoother carries this flag.
        if hasattr(esmda, "pin_initial_time_point"):
            esmda.pin_initial_time_point = window > 0

        # Get observations in window and add noise. Select the w-th contiguous
        # block of frames (half-open) rather than an inclusive time-slice: the
        # frame at the next window's start (t=(window+1)*sim_time) must NOT be
        # double-counted, or interior windows would be one frame longer than the
        # assimilation model emits and the observation vector would misalign.
        window_true_state = _open_truth(
            true_state_path, n_total, x_offset, start_idx, t_offset
        ).isel(time=slice(window * n_per_window, (window + 1) * n_per_window))
        window_obs = jnp.asarray(truth_obs_op(window_true_state))
        window_true_state.close()
        rng_key, subkey = jax.random.split(rng_key)
        window_obs = window_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, window_obs.shape)
        
        # Sample posterior
        solve_start = time.perf_counter()
        output = esmda(
            state=state_input,
            params=prior_params,
            observations=window_obs,
            return_params_history=True,
            return_state_history=False,
        )
        solve_seconds.append(time.perf_counter() - solve_start)

        posterior_params = output[0].isel(esmda_step=-1)
        posterior_params.to_netcdf(windows_dir / f"window_{window}_posterior_params.nc")
        output[1].to_netcdf(windows_dir / f"window_{window}_posterior_state.nc")

        state_input = output[1].isel(time=-1)

        # Next window's prior: extrapolate the posterior
        if window < num_windows - 1:
            if is_dynamic:
                prediction_times = jnp.linspace(
                    sim_time, 2.0 * sim_time, cfg.prior_params.time_coords.num
                )
                rng_key, subkey = jax.random.split(rng_key)
                extrapolated = prior_sampler.extrapolate(
                    posterior_params, prediction_times, subkey
                )
                prior_params = extrapolated.assign_coords(
                    time=np.asarray(jnp.linspace(0.0, sim_time, cfg.prior_params.time_coords.num))
                )
            else:
                prior_params = posterior_params

        del output
        window_seconds.append(time.perf_counter() - window_start)

    esmda_seconds = time.perf_counter() - esmda_start

    # Run metadata + timing handed to ``_finish_rollout``, which augments it with
    # the estimation metrics and writes ``run_summary.yaml``.
    run_info = {
        "configuration": {
            "smoother": type(esmda).__name__,
            "joint_state_and_parameter": bool(include_state),
            "time_varying_parameters": bool(is_dynamic),
            "num_assimilation_windows": int(num_windows),
            "ensemble_size": int(ensemble_size),
            "simulation_time_per_window": float(sim_time),
            "final_time": float(final_time),
            "observation_error_std": float(cfg.esmda.obs_error_std),
            "num_esmda_steps": int(cfg.esmda.num_steps),
            "seed": int(cfg.esmda.seed),
            "truth_model": str(cfg.truth_model.name),
            "assimilation_model": str(cfg.assim_model.name),
            "truth_source": "disk" if cfg.run.truth_dir is not None else "inline",
            "truth_dir": str(cfg.run.truth_dir) if cfg.run.truth_dir is not None else None,
            "num_truth_frames": int(n_total),
        },
        "timing": {
            "esmda_total_seconds": float(esmda_seconds),
            "esmda_solve_seconds": float(sum(solve_seconds)),
            "mean_window_seconds": float(np.mean(window_seconds)) if window_seconds else None,
            "per_window_seconds": [float(s) for s in window_seconds],
            "per_window_solve_seconds": [float(s) for s in solve_seconds],
        },
    }

    _finish_rollout(
        cfg,
        out_dir=out_dir,
        windows_dir=windows_dir,
        num_windows=num_windows,
        sim_time=sim_time,
        is_dynamic=is_dynamic,
        true_state_path=true_state_path,
        n_total=n_total,
        x_offset=x_offset,
        start_idx=start_idx,
        t_offset=t_offset,
        run_info=run_info,
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="run_esmda")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
