"""Compute the metrics for a finished filtering run and write ``run_summary.yaml``.

Second stage of the three-script single-run filtering pipeline (see
``scripts/run_filtering_pipeline.sh``), mirroring compute_esmda_metrics.py:

  1. scripts/filtering/run_filtering.py            -- runs the filter, saves the
                                       raw artifacts + truth_access.yaml +
                                       cycle_diagnostics.yaml + run_info.yaml.
  2. scripts/filtering/compute_filtering_metrics.py (THIS) -- reads those,
                                       computes the parameter / state / sensor /
                                       filter-diagnostic metrics and writes
                                       run_summary.yaml.
  3. scripts/filtering/make_filtering_figures.py   -- reads those, draws figures.

``run_summary.yaml`` is the run_info (configuration + timing) saved by
run_filtering.py, augmented with:

  * ``metrics_version``     -- estimator-semantics marker shared with the ESMDA
                               pipeline; 2 = fair (``M(M-1)``) pairwise scores
                               (see scripts/esmda/compute_esmda_metrics.py).
  * ``parameter_metrics``   -- per-parameter RMSE/CRPS of the final analyzed
                               ensemble vs truth (+ skill vs prior) and its
                               calibration (z-score, normalized error,
                               contraction ratio). Absent in ``mode='state'``
                               (no parameters estimated).
  * ``filter_diagnostics``  -- summary stats of the per-cycle innovation chi2 and
                               observation-space prior/posterior RMSE.
  * ``state_metrics``       -- |U| field RMSE of the analyzed end-of-cycle states
                               vs the truth's end-of-cycle frames.
  * ``sensor_metrics``      -- per sensor set, the full-vector (u, v, w) RMSE and
                               energy score (multivariate CRPS) over the cycles.

The time axis throughout is the cycle: the truth is compared at each cycle's
end-of-segment frame (see scripts/filtering/_filtering_common.py).

Usage::

    python scripts/filtering/compute_filtering_metrics.py --run-dir <filtering output dir>
"""

import argparse
import pathlib
import sys

import numpy as np
import xarray

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.scores import (
    METRICS_VERSION,
    parameter_metric_summary,
    series_stats,
    vector_sensor_metrics,
)
from evaluation.turbulence import streaming_state_rmse

from scripts.filtering._filtering_common import (
    build_sensor_sets,
    cycle_diagnostics_series,
    ensemble_cycle_sensor_series,
    load_analyzed_states,
    load_run_config,
    read_yaml,
    truth_cycle_sensor_series,
    truth_end_of_cycle,
    write_yaml,
)


def compute_metrics(run_dir: pathlib.Path) -> None:
    """Compute every run metric from the saved artifacts and write run_summary.yaml."""
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")

    # Seed the summary from the run metadata/timing saved by run_filtering.py.
    summary = read_yaml(run_dir / "run_info.yaml")
    summary["metrics_version"] = METRICS_VERSION

    # --- Filter health diagnostics (always available) -----------------------
    # Summarize the per-cycle innovation chi2 and observation-space RMSE that the
    # filter recorded; these need no truth access and exist for every mode.
    diag = cycle_diagnostics_series(run_dir)
    if diag:
        summary["filter_diagnostics"] = {
            field: series_stats(values) for field, values in diag.items()
        }

    # --- Parameters (present unless mode='state') ---------------------------
    posterior_path = run_dir / "posterior_params.nc"
    if posterior_path.exists():
        posterior_params = xarray.open_dataset(posterior_path)
        prior_params = xarray.open_dataset(run_dir / "prior_params.nc")
        true_params = xarray.open_dataset(run_dir / "true_params.nc")
        # The posterior is the filter's final scalar estimate (end of run). Label
        # it with the run's end time so a drifting (time-varying) truth is
        # compared at that time rather than at t=0 (the truth is interpolated
        # onto the estimate's x-axis; see compute_parameter_metrics). A static
        # truth is constant, so the label is a no-op for it.
        if ta and "time" not in posterior_params.dims:
            final_time = float(ta["sim_time"]) * int(ta["num_cycles"])
            posterior_params = posterior_params.expand_dims(time=[final_time])
        summary["parameter_metrics"] = parameter_metric_summary(
            posterior_params, true_params, prior_params
        )

    # The state and sensor metrics both open the (potentially multi-GB) truth.
    # If the truth-access record is missing (e.g. an older run) there is nothing
    # to compare against, so write what we have and stop.
    if not ta:
        write_yaml(summary, run_dir / "run_summary.yaml")
        print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")
        return

    # --- State field |U| RMSE -----------------------------------------------
    # Compare each cycle's analyzed ensemble-mean state against the truth's
    # end-of-cycle frame, streaming over a few z-slices (interpolating onto the
    # assim grid if the coords differ). One RMSE per cycle.
    analyzed_states = load_analyzed_states(run_dir, ta)
    n_frames = int(analyzed_states.sizes["time"])
    truth_end = truth_end_of_cycle(ta, n_frames)
    rmse = streaming_state_rmse(truth_end, analyzed_states)
    summary["state_metrics"] = {"vel_magnitude_rmse": series_stats(rmse)}

    # --- Sensors: full-vector (u, v, w) RMSE + energy score ------------------
    # Truth and ensemble are interpolated at the same physical points on their
    # own grids, so the metric is grid-independent. One value per cycle.
    sensor_sets = build_sensor_sets(cfg)
    truth_series = truth_cycle_sensor_series(
        truth_end, sensor_sets, ta["truth_solver_name"]
    )
    ensemble_series = ensemble_cycle_sensor_series(
        analyzed_states, sensor_sets, ta["assim_solver_name"]
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
        help="The filtering run output directory written by scripts/filtering/run_filtering.py.",
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    compute_metrics(args.run_dir)


if __name__ == "__main__":
    main()
