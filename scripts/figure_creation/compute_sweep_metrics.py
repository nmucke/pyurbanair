"""Compute all sweep metrics + metric time series from ESMDA posterior results.

Middle stage of the three-script sweep pipeline:

  1. scripts/esmda/run_esmda.py            -- runs the DA, writes the (large) posterior
                                        states/params + a base run_summary.yaml +
                                        per-window prior/posterior states +
                                        truth_access.yaml, all under the project
                                        results root.
  2. scripts/figure_creation/compute_sweep_metrics.py (THIS) -- reads those
                                        posterior results and the ground truth,
                                        computes every metric and metric time
                                        series, and writes SMALL artifacts (no
                                        full states) to
                                        pyurbanair/sweep_metrics/<run>/.
  3. scripts/figure_creation/compare_sweep_results.py -- reads
                                        pyurbanair/sweep_metrics/ and draws the
                                        comparison figures + the big CSV.

Per run it writes ``pyurbanair/sweep_metrics/<run>/``:

  * ``metrics.yaml``                -- estimator version + configuration + parameter /
                                       state / sensor metrics. Sensor metrics now
                                       cover |U| AND each velocity component (u/v/w),
                                       per sensor set (assimilation + validation).
                                       Same schema as run_summary.yaml so the
                                       comparison script parses it unchanged.
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

    python scripts/figure_creation/compute_sweep_metrics.py
    python scripts/figure_creation/compute_sweep_metrics.py \
        --root /projects/prjs2075/urbanair/assim_from_ground_truth \
        --out  pyurbanair/sweep_metrics --models pyudales pylbm
"""

# mypy: ignore-errors
# Legacy untyped CLI module. Keep the waiver local until its existing helpers
# are annotated; runtime behavior remains covered by the metric tests.

from __future__ import annotations

import argparse
import logging
import pathlib
import shutil
import sys

import numpy as np
import xarray as xr
import yaml
from omegaconf import OmegaConf

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_validation_points,
)
from pyurbanair.plotting import compute_parameter_metrics, compute_sensor_metrics
from scripts.esmda._esmda_common import (
    ensemble_sensor_series,
    sensor_magnitude,
    truth_sensor_series,
)

logger = logging.getLogger(__name__)

# The components this stage knows how to key an artifact by. Deliberately a
# closed set, not whatever the state file happens to carry: `_Q_KEY` below,
# `metrics.yaml`'s key names and the comparison script's columns are all written
# per component, so a fourth one has nowhere to go -- `_split_quantities` raises
# rather than dropping it.
_COMPONENTS = ("u", "v", "w")
# Velocity components plus magnitude; ``vel`` keeps the historical summary key.
QUANTITIES = _COMPONENTS + ("vel",)
_Q_KEY = {"vel": "vel_magnitude", "u": "u", "v": "v", "w": "w"}


# ---------------------------------------------------------------------------
# Sensor series (shared with the ESMDA stage, reshaped per quantity here)
# ---------------------------------------------------------------------------


def _split_quantities(components):
    """``DataArray(component, ...)`` -> ``{"u", "v", "w", "vel": DataArray(...)}``.

    The one structural difference between this stage's sensor series and the
    ESMDA stage's: ESMDA keeps the three components on a ``component`` dim,
    while ``QUANTITIES`` and every sweep artifact are keyed per quantity with
    |U| alongside u/v/w.

    ``.sel`` is deliberately left as a *view* of the stacked buffer, so each
    component keeps the memory layout the old full-ensemble interpolation
    produced. That is not cosmetic: consumers reduce over these axes, numpy
    walks a reduction in memory order and float addition is not associative, so
    a re-laid-out buffer moves the artifact. Measured on the wiring fixture, by
    diffing every leaf of ``metrics.yaml`` computed both ways:

    ============================================  ==============  ============
    how the per-quantity arrays are laid out      leaves changed  largest move
    ============================================  ==============  ============
    ``.sel`` view (this function)                  0 of 92         --
    ``np.ascontiguousarray`` per quantity         16 of 92         2.1e-15
    ============================================  ==============  ============

    All sixteen were sensor CRPS entries, and the moves are 1-2 ULP -- no
    estimator changes either way. The view is kept anyway because sweep
    comparisons are cross-run: ``metrics_version`` exists so that historical
    numbers are only allowed to move deliberately.

    |U| is ``_esmda_common.sensor_magnitude``, the same definition the ESMDA
    stage scores, rather than a second copy of the deleted
    ``_sensor_components``' elementwise ``sqrt(u**2 + v**2 + w**2)``. Its
    ``sqrt((x**2).sum("component"))`` was measured to give a bit-identical
    buffer with identical strides on every fixture -- the reduction is over
    three elements, so it sums in the same order as the elementwise chain. The
    bare ``.rename()`` after it is the whole adaptation: the reduction inherits
    the stacked array's ``name`` (``"u"``, from ``xr.concat``), and calling
    ``rename`` with no argument clears it back to ``None``, which is what the
    old |U| carried.

    ``.rename(q)`` likewise restores the per-component names, for the same
    reason: ``xr.concat`` labels the stacked array with its first input's name,
    so every ``.sel`` would otherwise come back called ``"u"``. Nothing
    downstream reads the name (NetCDF variables are named by
    ``_save_sensor_timeseries``' dict keys), but leaving it wrong would make an
    ``identical()`` check on these arrays lie.

    Args:
        components: ``DataArray`` with a ``component`` coordinate holding
            exactly ``("u", "v", "w")``, in that order.

    Returns:
        ``{quantity: DataArray}`` over :data:`QUANTITIES`.

    Raises:
        ValueError: if the ``component`` coordinate is missing or is anything
            other than ``("u", "v", "w")``. Silently taking the three it knows
            is the failure mode being removed: every artifact this stage writes
            is keyed per component, so an extra component would vanish from the
            summary *and* from |U| with nothing to read anywhere.
    """
    # Membership rather than ``.coords.get``: for a dim with no coordinate the
    # latter hands back a virtual 0..n-1 range, which would report the component
    # set as integers instead of as absent.
    present = (
        tuple(str(c) for c in np.atleast_1d(components.coords["component"].values))
        if "component" in components.coords
        else ()
    )
    if present != _COMPONENTS:
        raise ValueError(
            "sweep sensor series must carry exactly the components "
            f"{_COMPONENTS}; got "
            f"{present if present else 'no component coordinate'}. The per-quantity "
            "artifact keys (QUANTITIES / _Q_KEY) and |U| are defined over those "
            "three, so anything else has to be added here deliberately."
        )
    quantities = {
        q: components.sel(component=q, drop=True).rename(q) for q in _COMPONENTS
    }
    quantities["vel"] = sensor_magnitude(components).rename()
    return quantities


def _ensemble_series(state_paths, sensor_sets, solver_name, sim_time):
    """Per-component sensor series from per-window ensemble state files.

    ``{name: {quantity: DataArray(ensemble, time, sensor)}}``, with each window's
    local time rebased onto a single global axis (window ``w`` starts at
    ``w*sim_time``). Returns ``None`` if any window file is missing.

    The read is delegated to ``_esmda_common.ensemble_sensor_series``, which
    streams each window file **one member at a time**. This function used to
    ``xr.open_dataset(path).load()`` the file whole -- but those files carry the
    entire ensemble (~1 GB at smoke scale, tens of GB at Barcelona scale), which
    is exactly what phase 1 forbids materialising. Sharing the ESMDA stage's
    extraction rather than keeping a second copy also means the two stages
    cannot drift apart in what a "sensor series" is; the per-quantity reshaping
    in :func:`_split_quantities` is all that remains specific to this one.

    The result is bit-identical to the pre-streaming implementation, memory
    layout included -- pinned by ``tests/test_da_metrics.py``, which scores it
    against a verbatim copy of the old code. That matters because sweep
    comparisons are cross-run: silently moving a historical number is what
    ``metrics_version`` exists to prevent.

    Raises:
        ValueError: from the shared reader, if a window file exists but has no
            ``ensemble`` dimension. A *missing* file is not an error (``None``
            is returned) -- this is the file that is present and unusable.
            :func:`process_run` degrades on it rather than losing the run.
    """
    if not all(p.exists() for p in state_paths):
        return None
    series = ensemble_sensor_series(state_paths, sensor_sets, solver_name, sim_time)
    return {name: _split_quantities(vel) for name, vel in series.items()}


def _truth_series(ta, sensor_sets, solver_name):
    """Per-component truth sensor series, read one window at a time.

    ``{name: {quantity: DataArray(time, sensor)}}``. The truth's time axis is
    already global, so the per-window pieces concatenate directly.

    Delegated to ``_esmda_common.truth_sensor_series`` for the same reason
    :func:`_ensemble_series` is: this stage's copy of the truth window loop --
    and of ``open_truth``'s offset/slicing rules, which are the contract with
    ``truth_access.yaml`` -- was character-for-character the ESMDA stage's, so
    keeping both only created somewhere for them to disagree. Bit-identical to
    the copy it replaces, layout included.
    """
    return {
        name: _split_quantities(vel)
        for name, vel in truth_sensor_series(
            ta["true_state_path"],
            ta["n_total"],
            ta["x_offset"],
            ta["start_idx"],
            ta["t_offset"],
            sensor_sets,
            solver_name,
            ta["num_windows"],
            ta["n_per_window"],
        ).items()
    }


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
    """Per-parameter RMSE/CRPS summary + skill vs prior (library compute)."""
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
        if "prior_crps" in m:
            prior_mean = float(np.nanmean(m["prior_crps"]))
            post_mean = float(np.nanmean(m["crps"]))
            entry["prior_crps_mean"] = prior_mean
            entry["crps_reduction_vs_prior"] = (
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

    metrics: dict = {
        # This stage recomputes every CRPS with the current fair estimator, even
        # when processing a raw run whose older run_summary has no version key.
        "metrics_version": 2,
        "configuration": summary.get("configuration", {}),
        "timing": summary.get("timing", {}),
    }

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

        # A malformed / legacy state file makes the sensor stage unusable but
        # says nothing about the parameter and state metrics, which are already
        # computed above. Letting the exception out would lose those too: `main`
        # catches per run and moves on, so the run would end up with no
        # `metrics.yaml` at all and one printed line as its only trace. Degrade
        # the way the missing-`truth_access` branch below does instead -- write
        # what is computable, record why the rest is absent, and log it against
        # the run. `ValueError` only: it is what `stream_window_members` raises
        # on a state file with no `ensemble` dimension (a hand-made or
        # pre-`run_esmda` file; the current writer always produces one) and what
        # the sensor interpolation raises on out-of-domain sensor points. A
        # missing *file* is not an error here -- `_ensemble_series` returns
        # ``None`` for that, and the prior series are routinely absent.
        try:
            truth_s = _truth_series(ta, sensor_sets, ta["truth_solver_name"])
            post_s = _ensemble_series(
                post_paths, sensor_sets, ta["assim_solver_name"], ta["sim_time"]
            )
            prior_s = _ensemble_series(
                prior_paths, sensor_sets, ta["assim_solver_name"], ta["sim_time"]
            )
        except ValueError as exc:
            logger.warning(
                "%s: sensor series unreadable (%s: %s) -- sensor metrics and "
                "sensor_timeseries_*.nc omitted; the rest of metrics.yaml is "
                "still written",
                run_dir.name,
                type(exc).__name__,
                exc,
            )
            status["note"] = f"sensor series unreadable ({type(exc).__name__}: {exc})"
            truth_s = post_s = prior_s = None
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
                    out_run,
                    name,
                    (sx, sy, sz),
                    truth_s[name],
                    prior_s[name] if prior_s is not None else None,
                    post_s[name],
                )
            sensor_metrics[name] = entry

    if sensor_metrics:
        metrics["sensor_metrics"] = sensor_metrics
    else:
        # No truth_access (pre-update run): the source summary's sensor scores
        # use the legacy estimator and cannot live under this stage's version-2
        # marker. Omit them rather than producing a mixed-semantics artifact.
        status["note"] = "no truth_access.yaml -> sensor metrics omitted (re-run ESMDA)"

    _write_yaml(metrics, out_run / "metrics.yaml")
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("/projects/prjs2075/urbanair/assim_from_ground_truth"),
        help="Root holding the per-run ESMDA result directories.",
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Output dir for the small metric artifacts "
        "(default: <repo>/sweep_metrics).",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these assim backends (by dir-name prefix).",
    )
    args = ap.parse_args()

    # So the per-run degradations (`process_run`'s `logger.warning`) and the
    # tracebacks below are readable on a sweep of dozens of runs, rather than
    # arriving through logging's unformatted last-resort handler.
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out_root = args.out or (repo_root / "sweep_metrics")
    if not args.root.exists():
        raise SystemExit(f"results root not found: {args.root}")
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(p.parent for p in args.root.glob("*/run_summary.yaml"))
    if args.models:
        run_dirs = [
            d for d in run_dirs if any(d.name.startswith(m) for m in args.models)
        ]
    if not run_dirs:
        raise SystemExit("No runs with run_summary.yaml found.")

    print(f"Computing metrics for {len(run_dirs)} run(s) -> {out_root}")
    n_ts = 0
    for run_dir in run_dirs:
        try:
            st = process_run(run_dir, out_root / run_dir.name)
        except Exception as e:  # noqa: BLE001
            # One unprocessable run must not abandon the other forty, so the
            # loop continues -- but the run then has *no* `metrics.yaml`, which
            # the comparison script reads as "absent", so the traceback is the
            # only way to tell a broken run from one that was never processed.
            # Logged, not just printed, for that reason.
            logger.exception("%s: metrics stage failed, no metrics.yaml", run_dir.name)
            print(f"  ! {run_dir.name}: FAILED ({type(e).__name__}: {e})")
            continue
        tag = (
            "with u/v/w series" if st["components"] else st.get("note", "summary only")
        )
        n_ts += int(st["sensor_timeseries"])
        print(f"  {run_dir.name}: {tag}")

    print(
        f"\nDone. {n_ts}/{len(run_dirs)} run(s) have per-component sensor series. "
        f"Metrics in {out_root}"
    )


if __name__ == "__main__":
    main()
