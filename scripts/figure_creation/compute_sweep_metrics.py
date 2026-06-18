"""Compute all sweep metrics + metric time series from ESMDA posterior results.

Middle stage of the three-script sweep pipeline:

  1. scripts/run_esmda.py            -- runs the DA, writes the (large) posterior
                                        states/params + a base run_summary.yaml +
                                        per-window prior/posterior states +
                                        truth_access.yaml, all under the project
                                        results root.
  2. scripts/compute_sweep_metrics.py (THIS) -- reads those posterior results and
                                        the ground truth, computes every metric
                                        and metric time series, and writes SMALL
                                        artifacts (no full states) to
                                        pyurbanair/sweep_metrics/<run>/.
  3. scripts/compare_sweep_results.py -- reads pyurbanair/sweep_metrics/ and draws
                                        the comparison figures + the big CSV.

Per run it writes ``pyurbanair/sweep_metrics/<run>/``:

  * ``metrics.yaml``                -- configuration + parameter / state / sensor
                                       metrics. Sensor metrics now cover |U| AND
                                       each velocity component (u/v/w), per sensor
                                       set (assimilation + validation). Same schema
                                       as run_summary.yaml so the comparison script
                                       parses it unchanged.
  * ``sensor_timeseries_<set>.nc``  -- per sensor set: truth, prior ensemble and
                                       posterior ensemble |U|/u/v/w time series at
                                       each sensor (small; no full fields).
  * ``posterior_params.nc`` / ``prior_params.nc`` / ``true_params.nc`` -- copied
                                       (tiny) so the comparison script is fully
                                       self-contained off sweep_metrics/.

The prior sensor series require the per-window prior states that run_esmda.py now
saves (``window_*_prior_state.nc``); runs produced before that change are still
processed for everything else and the prior series are simply skipped (logged).

Usage::

    python scripts/compute_sweep_metrics.py
    python scripts/compute_sweep_metrics.py \
        --root /projects/prjs2075/urbanair/assim_from_ground_truth \
        --out  pyurbanair/sweep_metrics --models pyudales pylbm
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import numpy as np
import xarray as xr
import yaml
from omegaconf import OmegaConf

from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_operator import ObservationOperator
from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_validation_points,
)
from pyurbanair.plotting import compute_parameter_metrics, compute_sensor_metrics


# Velocity components plus magnitude; ``vel`` keeps the historical summary key.
QUANTITIES = ("u", "v", "w", "vel")
_Q_KEY = {"vel": "vel_magnitude", "u": "u", "v": "v", "w": "w"}
_X_COORDS = ("x", "xt", "xm")


# ---------------------------------------------------------------------------
# Truth access (mirrors run_esmda.py's lazy view, driven by truth_access.yaml)
# ---------------------------------------------------------------------------

def _open_truth(true_state_path, n_total, x_offset=0.0, start_idx=0, t_offset=0.0):
    """Lazily open the truth state, limited to ``n_total`` frames from ``start_idx``.

    Same offsets/slicing as run_esmda.py so the truth lines up frame-for-frame
    with the assimilation windows. Kept lazy (``open_dataset``) so a multi-GB
    truth is never loaded in full -- the caller slices one window at a time.
    """
    ds = xr.open_dataset(true_state_path)
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
# Sensor interpolation (per component) + per-window series assembly
# ---------------------------------------------------------------------------

def _sensor_components(state, obs_x, obs_y, obs_z, solver_name):
    """Interpolate u/v/w (+ |U|) at the sensor points, keeping leading dims.

    Returns ``{"u","v","w","vel": DataArray(..., time, sensor)}``. Interpolating
    once and deriving every quantity avoids repeating the (expensive) trilinear
    interpolation per component.
    """
    op = ObservationOperator(
        obs_x=list(np.asarray(obs_x, dtype=float)),
        obs_y=list(np.asarray(obs_y, dtype=float)),
        obs_z=list(np.asarray(obs_z, dtype=float)),
        obs_states=["u", "v", "w"],
        solver_name=solver_name,
    )
    comps = {}
    for var in ("u", "v", "w"):
        dims = op.dim_mapping[var]
        comps[var] = interpolate_dataarray_at_points(
            state[var],
            x_dim=dims["x"], y_dim=dims["y"], z_dim=dims["z"],
            obs_x=op.obs_x, obs_y=op.obs_y, obs_z=op.obs_z,
        )
    comps["vel"] = np.sqrt(comps["u"] ** 2 + comps["v"] ** 2 + comps["w"] ** 2)
    return comps


def _concat(parts):
    return parts[0] if len(parts) == 1 else xr.concat(parts, dim="time", join="override")


def _ensemble_series(state_paths, sensor_sets, solver_name, sim_time):
    """Per-component sensor series from per-window ensemble state files.

    ``{name: {quantity: DataArray(ensemble, time, sensor)}}``, with each window's
    local time rebased onto a single global axis (window ``w`` starts at
    ``w*sim_time``). Returns ``None`` if any window file is missing.
    """
    if not all(p.exists() for p in state_paths):
        return None
    pieces = {name: {q: [] for q in QUANTITIES} for name in sensor_sets}
    for w, path in enumerate(state_paths):
        ds = xr.open_dataset(path).load()
        t = np.asarray(ds["time"].values, dtype=float) if "time" in ds.coords else None
        for name, (ox, oy, oz) in sensor_sets.items():
            fields = _sensor_components(ds, ox, oy, oz, solver_name)
            for q, da in fields.items():
                if t is not None and "time" in da.dims:
                    da = da.assign_coords(time=(t - t[0]) + w * sim_time)
                pieces[name][q].append(da)
        ds.close()
    return {n: {q: _concat(parts) for q, parts in by_q.items()} for n, by_q in pieces.items()}


def _truth_series(ta, sensor_sets, solver_name):
    """Per-component truth sensor series, read one window at a time.

    ``{name: {quantity: DataArray(time, sensor)}}``. The truth's time axis is
    already global, so the per-window pieces concatenate directly.
    """
    pieces = {name: {q: [] for q in QUANTITIES} for name in sensor_sets}
    for w in range(ta["num_windows"]):
        ts = _open_truth(
            ta["true_state_path"], ta["n_total"], ta["x_offset"],
            ta["start_idx"], ta["t_offset"],
        ).isel(time=slice(w * ta["n_per_window"], (w + 1) * ta["n_per_window"]))
        for name, (ox, oy, oz) in sensor_sets.items():
            fields = _sensor_components(ts, ox, oy, oz, solver_name)
            for q, da in fields.items():
                pieces[name][q].append(da)
        ts.close()
    return {n: {q: _concat(parts) for q, parts in by_q.items()} for n, by_q in pieces.items()}


# ---------------------------------------------------------------------------
# Scalar metric helpers
# ---------------------------------------------------------------------------

def _series_stats(arr):
    """{mean, final, max, min} of a 1-D series, or ``None`` if it has no values."""
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


def _to_native(obj):
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


def _parameter_metrics(post, true, prior):
    """Per-parameter RMSE/CRPS summary + reduction vs prior (library compute)."""
    metrics = compute_parameter_metrics(post, true, prior)
    out = {}
    for name, m in metrics.items():
        entry = {"rmse": _series_stats(m["rmse"]), "crps": _series_stats(m["crps"])}
        if "prior_rmse" in m:
            prior_mean = float(np.nanmean(m["prior_rmse"]))
            post_mean = float(np.nanmean(m["rmse"]))
            entry["prior_rmse_mean"] = prior_mean
            entry["rmse_reduction_vs_prior"] = (
                float(1.0 - post_mean / prior_mean) if prior_mean > 0 else None
            )
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Sensor time-series persistence
# ---------------------------------------------------------------------------

def _save_sensor_timeseries(out_run, name, coords, truth_q, prior_q, post_q):
    """Write one sensor set's truth/prior/posterior u/v/w/|U| series to NetCDF.

    The truth and the ensemble forecasts are sampled on slightly different time
    grids (truth at the truth cadence, the ensemble at the assimilation output
    cadence). They are interpolated onto a single common axis -- the ensemble's
    (rebased, global) time -- before being combined. Without this, ``xr.Dataset``
    would *union* the two grids and fill every mismatch with NaN, so each line
    would render at only ~half its points -- broken, gappy and time-offset.
    """
    ref = post_q if post_q is not None else prior_q
    tc = ref["vel"]["time"] if (ref is not None and "time" in ref["vel"].dims) else None

    def _align(da):
        if tc is None or "time" not in da.dims:
            return da
        return da.interp(time=tc, kwargs={"fill_value": "extrapolate"})

    data_vars = {}
    for q in QUANTITIES:
        data_vars[f"{q}_truth"] = _align(truth_q[q])
        if prior_q is not None:
            data_vars[f"{q}_prior"] = _align(prior_q[q])
        if post_q is not None:
            data_vars[f"{q}_post"] = _align(post_q[q])
    ds = xr.Dataset(data_vars)
    ox, oy, oz = coords
    ds = ds.assign_coords(
        sensor_x=("sensor", np.asarray(ox, dtype=float)),
        sensor_y=("sensor", np.asarray(oy, dtype=float)),
        sensor_z=("sensor", np.asarray(oz, dtype=float)),
    )
    ds.to_netcdf(out_run / f"sensor_timeseries_{name}.nc")


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(run_dir: pathlib.Path, out_run: pathlib.Path) -> dict:
    """Compute every metric + time series for one run; returns a short status dict."""
    out_run.mkdir(parents=True, exist_ok=True)
    status = {"name": run_dir.name, "sensor_timeseries": False, "components": False}

    with open(run_dir / "run_summary.yaml") as f:
        summary = yaml.safe_load(f) or {}
    cfg = OmegaConf.load(run_dir / "config.yaml")

    metrics: dict = {"configuration": summary.get("configuration", {}),
                     "timing": summary.get("timing", {})}

    # --- Parameters (recomputed from the small param NetCDFs) ----------------
    post_p = xr.open_dataset(run_dir / "posterior_params.nc")
    true_p = xr.open_dataset(run_dir / "true_params.nc")
    prior_p = xr.open_dataset(run_dir / "prior_params.nc")
    metrics["parameter_metrics"] = _parameter_metrics(post_p, true_p, prior_p)
    # Copy the (tiny) param files so the comparison script is self-contained.
    for fn in ("posterior_params.nc", "prior_params.nc", "true_params.nc"):
        shutil.copyfile(run_dir / fn, out_run / fn)

    # --- State field RMSE (reuse run_esmda's streamed base metric) -----------
    if "state_metrics" in summary:
        metrics["state_metrics"] = summary["state_metrics"]

    # --- Sensors: |U| + u/v/w, truth/prior/posterior series + metrics --------
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)
    sensor_sets = {"assimilation": (obs_x, obs_y, obs_z)}
    val = create_validation_points(cfg.obs)
    if val is not None:
        sensor_sets["validation"] = val

    ta_path = run_dir / "truth_access.yaml"
    sensor_metrics = {}
    if ta_path.exists():
        with open(ta_path) as f:
            ta = yaml.safe_load(f)
        nwin = int(ta["num_windows"])
        windows = run_dir / "windows"
        post_paths = [windows / f"window_{w}_posterior_state.nc" for w in range(nwin)]
        prior_paths = [windows / f"window_{w}_prior_state.nc" for w in range(nwin)]

        truth_s = _truth_series(ta, sensor_sets, ta["truth_solver_name"])
        post_s = _ensemble_series(post_paths, sensor_sets, ta["assim_solver_name"], ta["sim_time"])
        prior_s = _ensemble_series(prior_paths, sensor_sets, ta["assim_solver_name"], ta["sim_time"])
        status["sensor_timeseries"] = post_s is not None
        status["components"] = post_s is not None

        for name, (sx, sy, sz) in sensor_sets.items():
            entry = {"num_sensors": int(np.asarray(sx).size)}
            if post_s is not None:
                for q in QUANTITIES:
                    m = compute_sensor_metrics(truth_s[name][q], post_s[name][q])
                    entry[f"{_Q_KEY[q]}_rmse"] = _series_stats(m["rmse"])
                    entry[f"{_Q_KEY[q]}_crps"] = _series_stats(m["crps"])
                _save_sensor_timeseries(
                    out_run, name, (sx, sy, sz),
                    truth_s[name], prior_s[name] if prior_s is not None else None,
                    post_s[name],
                )
            sensor_metrics[name] = entry

    if not sensor_metrics:
        # No truth_access (pre-update run): fall back to the base |U| summary.
        sensor_metrics = summary.get("sensor_metrics", {})
        status["note"] = "no truth_access.yaml -> base |U| sensor metrics only (re-run ESMDA)"
    metrics["sensor_metrics"] = sensor_metrics

    _write_yaml(metrics, out_run / "metrics.yaml")
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path,
                    default=pathlib.Path("/projects/prjs2075/urbanair/assim_from_ground_truth"),
                    help="Root holding the per-run ESMDA result directories.")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="Output dir for the small metric artifacts "
                         "(default: <repo>/sweep_metrics).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these assim backends (by dir-name prefix).")
    args = ap.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out_root = args.out or (repo_root / "sweep_metrics")
    if not args.root.exists():
        raise SystemExit(f"results root not found: {args.root}")
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(p.parent for p in args.root.glob("*/run_summary.yaml"))
    if args.models:
        run_dirs = [d for d in run_dirs if any(d.name.startswith(m) for m in args.models)]
    if not run_dirs:
        raise SystemExit("No runs with run_summary.yaml found.")

    print(f"Computing metrics for {len(run_dirs)} run(s) -> {out_root}")
    n_ts = 0
    for run_dir in run_dirs:
        try:
            st = process_run(run_dir, out_root / run_dir.name)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {run_dir.name}: FAILED ({type(e).__name__}: {e})")
            continue
        tag = "with u/v/w series" if st["components"] else st.get("note", "summary only")
        n_ts += int(st["sensor_timeseries"])
        print(f"  {run_dir.name}: {tag}")

    print(f"\nDone. {n_ts}/{len(run_dirs)} run(s) have per-component sensor series. "
          f"Metrics in {out_root}")


if __name__ == "__main__":
    main()
