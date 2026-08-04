"""Compute the metrics for a finished ESMDA run and write ``run_summary.yaml``.

Second stage of the three-script single-run ESMDA pipeline (see
``scripts/run_esmda_pipeline.sh``):

  1. scripts/esmda/run_esmda.py            -- runs the assimilation, saves the raw
                                       artifacts + truth_access.yaml + run_info.yaml.
  2. scripts/esmda/compute_esmda_metrics.py (THIS) -- reads those, computes the
                                       parameter / state / sensor metrics and
                                       writes run_summary.yaml.
  3. scripts/esmda/make_esmda_figures.py   -- reads those, draws the figures.

``run_summary.yaml`` is the run_info (configuration + timing) saved by
run_esmda.py, augmented with:

  * ``metrics_version``    -- estimator-semantics marker; 2 = fair (``M(M-1)``)
                              pairwise scores. Absent or 1 means the older
                              biased estimators, whose CRPS / energy-score
                              numbers sit ~O(1/M) higher and must not be
                              compared with version-2 ones.
  * ``parameter_metrics``  -- per-parameter RMSE/CRPS summary (+ skill vs prior)
                              and calibration: z-score, normalized error and
                              contraction ratio, plus a pooled z-score entry.
  * ``ensemble_health``    -- duplicate-member counts, run-wide and per window.
  * ``state_metrics``      -- |U| field RMSE summary (streamed over a few z-slices).
  * ``sensor_metrics``     -- per sensor set, the full-vector (u, v, w) RMSE and
                              energy score (multivariate CRPS). The comprehensive
                              per-component sweep series are computed separately by
                              scripts/figure_creation/compute_sweep_metrics.py.

Usage::

    python scripts/esmda/compute_esmda_metrics.py --run-dir <esmda output dir>
"""

import argparse
import logging
import pathlib
import re
import sys

import numpy as np
import xarray

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.scores import (
    METRICS_VERSION,
    ensemble_uniqueness,
    parameter_metric_summary,
    series_stats,
    vector_sensor_metrics,
)
from evaluation.turbulence import streaming_state_rmse

from scripts.esmda._esmda_common import (
    build_sensor_sets,
    ensemble_sensor_series,
    load_run_config,
    open_truth,
    read_yaml,
    truth_sensor_series,
    write_yaml,
)

logger = logging.getLogger(__name__)


def _flatten_parameter_members(params: xarray.Dataset) -> np.ndarray:
    """Flatten every ensemble parameter variable into one row per member."""
    arrays: list[np.ndarray] = []
    n_members: int | None = None
    for name in sorted(str(v) for v in params.data_vars):
        variable = params[name]
        if "ensemble" not in variable.dims:
            continue
        values = np.asarray(variable.transpose("ensemble", ...).values)
        if n_members is None:
            n_members = values.shape[0]
        elif values.shape[0] != n_members:
            raise ValueError(
                f"parameter {name!r} has {values.shape[0]} ensemble members; "
                f"expected {n_members}"
            )
        arrays.append(values.reshape(values.shape[0], -1))
    if not arrays:
        raise ValueError(
            "parameter dataset has no variables with an ensemble dimension"
        )
    return np.concatenate(arrays, axis=1)


def _ensemble_health(
    run_dir: pathlib.Path, posterior_params: xarray.Dataset
) -> dict[str, object] | None:
    """Duplicate-member counts for the assembled posterior and each window.

    The assembled posterior concatenates the windows, so two members that were
    cloned in one window but diverge in another are distinct there and only the
    per-window counts see it -- hence both.

    This is a diagnostic, not a metric: a parameter file it cannot read (an old
    layout, a window truncated by a killed job) costs its own count -- a
    ``null`` entry, so the list stays aligned with the window index -- and a log
    line, never the whole metric stage. Returns ``None`` when even the assembled
    posterior cannot be read.

    ``min_over_median_pairwise`` is the near-duplicate detector exact row
    matching cannot be: a ratio near 0 means two members sit far closer to each
    other than the ensemble typically does. It is an unweighted L2 over all
    parameters concatenated, so a parameter with a much larger numerical range
    (an inflow angle in degrees against an SGS constant) dominates it; read it
    as a coarse flag, not a calibrated distance.
    """
    try:
        health = ensemble_uniqueness(_flatten_parameter_members(posterior_params))
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Cannot assess ensemble health: %s", exc)
        return None

    if health["n_unique"] < health["n_members"]:
        logger.warning(
            "Duplicate posterior parameter members: %s/%s unique -- the fair "
            "scores' M(M-1) corrections overstate the ensemble's spread",
            health["n_unique"],
            health["n_members"],
        )

    per_window: list[int | None] = []

    # Sorted by window index, with anything that does not carry one last: a
    # stray or hand-renamed file in windows/ must not take the sort (and with
    # it the whole metric stage) down.
    def _window_index(path: pathlib.Path) -> tuple[int, int, str]:
        match = re.fullmatch(r"window_(\d+)_posterior_params", path.stem)
        return (0, int(match.group(1)), "") if match else (1, 0, path.stem)

    window_paths = sorted(
        (run_dir / "windows").glob("window_*_posterior_params.nc"), key=_window_index
    )
    for path in window_paths:
        try:
            with xarray.open_dataset(path) as params:
                window_health = ensemble_uniqueness(_flatten_parameter_members(params))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Skipping ensemble health for %s: %s", path.name, exc)
            # Keep the slot: consumers index this list by window.
            per_window.append(None)
            continue
        per_window.append(int(window_health["n_unique"]))
        if window_health["n_unique"] < window_health["n_members"]:
            logger.warning(
                "Duplicate posterior parameter members in %s: %s/%s unique",
                path.name,
                window_health["n_unique"],
                window_health["n_members"],
            )

    minimum, median = health["min_pairwise"], health["median_pairwise"]
    return {
        "n_members": int(health["n_members"]),
        "n_unique": int(health["n_unique"]),
        "n_unique_per_window": per_window,
        "min_over_median_pairwise": (
            float(minimum / median)
            if minimum is not None and median is not None and median > 0
            else None
        ),
    }


def compute_metrics(run_dir: pathlib.Path) -> None:
    """Compute every run metric from the saved artifacts and write run_summary.yaml."""
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")

    # Seed the summary from the run metadata/timing saved by run_esmda.py.
    summary = read_yaml(run_dir / "run_info.yaml")
    summary["metrics_version"] = METRICS_VERSION

    # --- Parameters (always available) --------------------------------------
    posterior_params = xarray.open_dataset(run_dir / "posterior_params.nc")
    prior_params = xarray.open_dataset(run_dir / "prior_params.nc")
    true_params = xarray.open_dataset(run_dir / "true_params.nc")
    summary["parameter_metrics"] = parameter_metric_summary(
        posterior_params, true_params, prior_params
    )
    health = _ensemble_health(run_dir, posterior_params)
    if health is not None:
        summary["ensemble_health"] = health

    # The parameter metrics are always available. The state and sensor metrics
    # both open the (potentially multi-GB) truth, so -- matching run_esmda.py's
    # old in-script behaviour -- they are skipped under run.skip_viz (the fast
    # path for large sweeps), exactly as the figures are.
    if cfg.run.skip_viz:
        write_yaml(summary, run_dir / "run_summary.yaml")
        print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")
        return

    # --- State field |U| RMSE -----------------------------------------------
    # Stream over a few z-slices and all time steps instead of the whole 4-D
    # field (interpolating onto the assim grid if coords differ).
    posterior_state = xarray.open_dataset(run_dir / "posterior_state_mean.nc")
    true_state = open_truth(
        ta["true_state_path"],
        ta["n_total"],
        ta["x_offset"],
        ta["start_idx"],
        ta["t_offset"],
    )
    rmse = streaming_state_rmse(true_state, posterior_state)
    summary["state_metrics"] = {"vel_magnitude_rmse": series_stats(rmse)}

    # --- Sensors: full-vector (u, v, w) RMSE + energy score ------------------
    # The truth and the ensemble are interpolated at the same physical points on
    # their own grids (one window at a time), so the metric is grid-independent
    # and never holds the multi-GB truth / full ensemble in memory in full.
    sensor_sets = build_sensor_sets(cfg)
    num_windows = int(ta["num_windows"])
    truth_series = truth_sensor_series(
        ta["true_state_path"],
        ta["n_total"],
        ta["x_offset"],
        ta["start_idx"],
        ta["t_offset"],
        sensor_sets,
        ta["truth_solver_name"],
        num_windows,
        int(ta["n_per_window"]),
    )
    state_paths = [
        run_dir / "windows" / f"window_{w}_posterior_state.nc"
        for w in range(num_windows)
    ]
    ensemble_series = ensemble_sensor_series(
        state_paths,
        sensor_sets,
        ta["assim_solver_name"],
        float(ta["sim_time"]),
    )

    sensor_metrics = {}
    for name, (sx, sy, sz) in sensor_sets.items():
        m = vector_sensor_metrics(truth_series[name], ensemble_series[name])
        sensor_metrics[name] = {
            "num_sensors": int(np.asarray(sx).size),
            "velocity_vector_rmse": series_stats(m["rmse"]),
            "velocity_vector_energy_score": series_stats(m["energy_score"]),
        }
    summary["sensor_metrics"] = sensor_metrics

    write_yaml(summary, run_dir / "run_summary.yaml")
    print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run-dir",
        type=pathlib.Path,
        required=True,
        help="The ESMDA run output directory written by scripts/esmda/run_esmda.py.",
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    compute_metrics(args.run_dir)


if __name__ == "__main__":
    main()
