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

  * ``metrics_version``    -- estimator-semantics marker (2 = fair scores).
  * ``parameter_metrics``  -- per-parameter RMSE/CRPS summary (+ skill vs prior).
  * ``ensemble_health``    -- exact duplicate counts and pairwise-distance ratio.
  * ``state_metrics``      -- |U| field RMSE summary (streamed over a few z-slices).
  * ``sensor_metrics``     -- per sensor set, the full-vector (u, v, w) RMSE and
                              energy score (multivariate CRPS). The comprehensive
                              per-component sweep series are computed separately by
                              scripts/figure_creation/compute_sweep_metrics.py.

How much of that is computed is gated by ``run.metrics.level`` in the run dir's
saved config (``basic`` = exactly the keys above, ``standard`` additionally
computes the evaluation layers, ``full`` is reserved for later phases); run dirs
saved before that config block existed fall back to the shipped defaults. See
``resolve_metrics_settings`` below.

Usage::

    python scripts/esmda/compute_esmda_metrics.py --run-dir <esmda output dir> \
        [--metrics-level basic|standard|full]
"""

import argparse
import dataclasses
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import xarray
from omegaconf import DictConfig, OmegaConf

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pyurbanair.utils.da_metrics import ensemble_uniqueness
from scripts.esmda._esmda_common import (
    build_sensor_sets,
    ensemble_sensor_series,
    load_run_config,
    open_truth,
    parameter_bundle_summary,
    parameter_metric_summary,
    read_yaml,
    series_stats,
    streaming_state_rmse,
    truth_sensor_series,
    vector_sensor_metrics,
    write_yaml,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics level / settings resolution
# ---------------------------------------------------------------------------

# Ordered cheapest-first; `at_least` compares by position, so the order is the
# semantics (each level is a superset of the ones before it).
METRICS_LEVELS = ("basic", "standard", "full")

# Defaults, kept in sync with the `run.metrics` block of conf/run_esmda.yaml.
# They are also the fallback for run dirs saved BEFORE that block existed: the
# phase-0/1 metric layers are meant to apply retroactively to old runs, and each
# individual layer is separately required to no-op when its inputs are missing,
# so defaulting an absent block to `standard` degrades gracefully rather than
# silently pinning old runs to a reduced metric set.
DEFAULT_METRICS = {
    "level": "standard",
    "n_z_slices": 4,
    "mean_field_stride": 1,
    "bootstrap_blocks": 20,
    "stations": None,
}


@dataclasses.dataclass(frozen=True)
class MetricsSettings:
    """Resolved ``run.metrics`` knobs for one metrics-stage invocation.

    Constructed once per run in :func:`compute_metrics` and handed to the metric
    sections, so the gating decision is made in exactly one place and the
    sections stay independently testable.
    """

    level: str
    n_z_slices: int
    mean_field_stride: int
    bootstrap_blocks: int
    stations: list[list[float]] | None

    def at_least(self, level: str) -> bool:
        """Whether the resolved level includes the layers gated at ``level``."""
        return METRICS_LEVELS.index(self.level) >= METRICS_LEVELS.index(level)


def resolve_metrics_settings(
    cfg: DictConfig | dict, level_override: str | None = None
) -> MetricsSettings:
    """Resolve ``run.metrics`` from a run dir's saved config.

    Every setting is read with a default: configs saved before the
    ``run.metrics`` block existed have no ``run.metrics`` key at all, and
    re-processing those old run dirs must keep working rather than dying on a
    missing key. ``level_override`` (the ``--metrics-level`` CLI flag) wins over
    the config so one run dir can be re-processed at another depth without
    editing its saved config.
    """
    container = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))

    def _get(key: str) -> Any:
        value = OmegaConf.select(container, f"run.metrics.{key}", default=None)
        return DEFAULT_METRICS[key] if value is None else value

    level = str(level_override if level_override is not None else _get("level"))
    if level not in METRICS_LEVELS:
        raise ValueError(
            f"unknown run.metrics.level {level!r}; expected one of "
            f"{', '.join(METRICS_LEVELS)}"
        )

    stations = _get("stations")
    if stations is not None:
        stations = [
            [float(v) for v in station]
            for station in OmegaConf.to_container(OmegaConf.create(stations))
        ]

    return MetricsSettings(
        level=level,
        n_z_slices=int(_get("n_z_slices")),
        mean_field_stride=int(_get("mean_field_stride")),
        bootstrap_blocks=int(_get("bootstrap_blocks")),
        stations=stations,
    )


def _flatten_parameter_members(params: xarray.Dataset) -> np.ndarray:
    """Flatten every parameter variable into one row per ensemble member."""
    arrays: list[np.ndarray] = []
    n_members: int | None = None
    for name in sorted(params.data_vars):
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
) -> dict[str, object]:
    """Duplicate-member diagnostics for the assembled and per-window posteriors."""
    health = ensemble_uniqueness(_flatten_parameter_members(posterior_params))
    per_window: list[int] = []
    window_paths = sorted(
        (run_dir / "windows").glob("window_*_posterior_params.nc"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    for path in window_paths:
        with xarray.open_dataset(path) as params:
            window_health = ensemble_uniqueness(_flatten_parameter_members(params))
        per_window.append(int(window_health["n_unique"]))
        if window_health["n_unique"] < window_health["n_members"]:
            logger.warning(
                "Duplicate posterior parameter members detected in %s: %s/%s unique",
                path.name,
                window_health["n_unique"],
                window_health["n_members"],
            )

    if health["n_unique"] < health["n_members"]:
        logger.warning(
            "Duplicate posterior parameter members detected: %s/%s unique",
            health["n_unique"],
            health["n_members"],
        )

    minimum = health["min_pairwise"]
    median = health["median_pairwise"]
    ratio = (
        float(minimum / median)
        if minimum is not None and median is not None and median > 0
        else None
    )
    return {
        "n_members": int(health["n_members"]),
        "n_unique": int(health["n_unique"]),
        "n_unique_per_window": per_window,
        "min_over_median_pairwise": ratio,
    }


def compute_metrics(run_dir: pathlib.Path, metrics_level: str | None = None) -> None:
    """Compute every run metric from the saved artifacts and write run_summary.yaml."""
    cfg = load_run_config(run_dir)
    metrics = resolve_metrics_settings(cfg, metrics_level)
    logger.info("Computing metrics at level %r for %s", metrics.level, run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")

    # Seed the summary from the run metadata/timing saved by run_esmda.py.
    summary = read_yaml(run_dir / "run_info.yaml")
    summary["metrics_version"] = 2

    # --- Parameters (always available) --------------------------------------
    posterior_params = xarray.open_dataset(run_dir / "posterior_params.nc")
    prior_params = xarray.open_dataset(run_dir / "prior_params.nc")
    true_params = xarray.open_dataset(run_dir / "true_params.nc")
    summary["parameter_metrics"] = parameter_metric_summary(
        posterior_params, true_params, prior_params
    )
    summary["ensemble_health"] = _ensemble_health(run_dir, posterior_params)

    # Standard-level parameter layers attach HERE (WP1.1: per-parameter z-score /
    # PIT counts / coverage / contraction ratio under the existing
    # `parameter_metrics.<name>` mapping, plus `parameter_metrics.joint`). They
    # deliberately sit BEFORE the skip_viz early return below: they read only the
    # already-open prior/posterior/true parameter datasets, never the truth
    # state, so they are cheap enough to run on the fast sweep path too.
    if metrics.at_least("standard"):
        summary["parameter_metrics"] = parameter_bundle_summary(
            summary["parameter_metrics"],
            posterior_params,
            true_params,
            prior_params,
            # Flattened here rather than inside the bundle so the pipeline keeps
            # exactly one (M, K) flattener, shared with `_ensemble_health`.
            posterior_flat=_flatten_parameter_members(posterior_params),
            prior_flat=_flatten_parameter_members(prior_params),
            cfg=cfg,
            num_windows=ta.get("num_windows"),
        )

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

    # Standard-level truth-consuming layers attach HERE: WP1.2's
    # `sensor_statistics` (window statistics scored against truth, reusing the
    # series already extracted above) and WP1.3's `mean_field_metrics` +
    # `eval_fields.nc`, which shares the single per-member read pass over the
    # window state files with the sensor extraction above.
    if metrics.at_least("standard"):
        logger.debug("standard-level sensor/field layers pending (WP1.2, WP1.3)")

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
    ap.add_argument(
        "--metrics-level",
        choices=METRICS_LEVELS,
        default=None,
        help=(
            "Override run.metrics.level from the run dir's saved config, so an "
            "existing run can be re-processed at another depth without editing "
            "that config. Default: whatever the saved config says (or "
            f"{DEFAULT_METRICS['level']!r} for run dirs predating the block)."
        ),
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    compute_metrics(args.run_dir, args.metrics_level)


if __name__ == "__main__":
    main()
