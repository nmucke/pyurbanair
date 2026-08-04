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
  * ``sensor_statistics``  -- per sensor set, the per-window mean and variance of
                              u/v/w/|U| scored as the verification object (fair
                              CRPS, z-score, rank), prior and posterior, with the
                              identifiability guard. This is the statistics-space
                              counterpart of ``sensor_metrics``: a pointwise
                              time-series error mostly measures turbulent phase,
                              which no parameter estimate controls.
  * ``field_metrics``      -- the VDI 3783/9 hit rate ``q`` of the time-mean
                              velocity field over evenly spaced z-slabs, prior
                              and posterior, with the absolute tolerance ``W``
                              block-bootstrapped from the truth's own sampling
                              error. The reduced fields behind it (mean, TKE,
                              ``<u'w'>``, on the slabs and at the station
                              columns) are written beside it as
                              ``eval_fields.nc`` for the figure stage.

Usage::

    python scripts/esmda/compute_esmda_metrics.py --run-dir <esmda output dir>
"""

import argparse
import logging
import pathlib
import re
import sys
from collections.abc import Callable

import numpy as np
import xarray

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.scores import (
    METRICS_VERSION,
    ensemble_uniqueness,
    hit_rate,
    parameter_metric_summary,
    series_stats,
    vector_sensor_metrics,
    window_statistics_summary,
)
from evaluation.sensors import window_sampling_std, window_statistics
from evaluation.turbulence import streaming_state_rmse

from scripts.esmda._esmda_common import (
    STATION_QUANTILES,
    MeanFieldCollector,
    build_sensor_sets,
    ensemble_sensor_series,
    load_run_config,
    open_truth,
    read_yaml,
    truth_sensor_series,
    write_yaml,
)

logger = logging.getLogger(__name__)

# The ``on_member`` callback the streaming extraction hands each materialised
# member slice: ``(window_index, member_index, member_state)``.
_OnMember = Callable[[int, int, xarray.Dataset], None] | None


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


def _prior_sensor_series(
    run_dir: pathlib.Path,
    sensor_sets: dict,
    num_windows: int,
    sim_time: float,
    solver_name: str,
    on_member: _OnMember = None,
) -> dict | None:
    """Sensor series from the saved prior states, or ``None`` with a log line.

    ``run.save_prior_state`` is off by default (the prior states double the
    state-file bytes a run writes), so the prior half of the statistics block is
    genuinely optional -- absence is the common case, not a fault. A partially
    written set is treated the same way: scoring a prior over some windows and a
    posterior over all of them would put two different horizons in one skill
    score.
    """
    paths = [
        run_dir / "windows" / f"window_{w}_prior_state.nc" for w in range(num_windows)
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.info(
            "No prior sensor statistics: %d of %d prior state files are absent "
            "(run.save_prior_state was off, or the run was interrupted)",
            len(missing),
            num_windows,
        )
        return None
    try:
        # ``dict(...)``: the extraction module is mypy-waived and returns ``Any``.
        return dict(
            ensemble_sensor_series(
                paths, sensor_sets, solver_name, sim_time, on_member=on_member
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        # A prior state file that *exists* but cannot be read -- a job killed
        # mid-``to_netcdf``, which is the same interrupted run the absence
        # branch above names. Invariant 3: it costs the prior half, not the
        # whole summary.
        logger.warning("Cannot read the prior sensor series: %s", exc)
        return None


def _sensor_statistics(
    run_dir: pathlib.Path,
    sensor_sets: dict,
    truth_series: dict,
    ensemble_series: dict,
    num_windows: int,
    sim_time: float,
    solver_name: str,
    on_prior_member: _OnMember = None,
) -> dict:
    """The ``sensor_statistics`` block, per sensor set (metrics doc §4.2).

    Reuses the series the vector sensor metrics already extracted, so the
    posterior half costs no extra pass over the state files; only the prior
    (when saved) is read here.

    One caveat on the last window: ``truth_sensor_series`` extracts exactly
    ``num_windows * n_per_window`` frames, so when the truth's frame count is
    not a multiple of the window count the trailing frames are dropped and the
    last window's truth statistic is computed over slightly fewer samples than
    its peers. The members are unaffected (each window state file is its own
    file), so this shows up as a marginally noisier truth in the final window.
    """
    prior_series = _prior_sensor_series(
        run_dir, sensor_sets, num_windows, sim_time, solver_name, on_prior_member
    )

    # No try/except around the scoring. Everything below is pure computation on
    # series the stage already holds in memory -- the only I/O is the prior read
    # above, which has its own guard -- so the exceptions it can raise are bugs,
    # and swallowing them would ship a green run with an empty block.
    statistics = {}
    for name in sensor_sets:
        prior = prior_series[name] if prior_series is not None else None
        statistics[name] = window_statistics_summary(
            window_statistics(truth_series[name], sim_time, num_windows, label=name),
            window_statistics(ensemble_series[name], sim_time, num_windows, label=name),
            prior_stats=(
                window_statistics(prior, sim_time, num_windows, label=name)
                if prior is not None
                else None
            ),
            posterior_sampling_std=window_sampling_std(
                ensemble_series[name], sim_time, num_windows
            ),
            prior_sampling_std=(
                window_sampling_std(prior, sim_time, num_windows)
                if prior is not None
                else None
            ),
            label=name,
        )
    return statistics


# ---------------------------------------------------------------------------
# Mean fields (WP1.4, metrics doc section 4.1)
# ---------------------------------------------------------------------------

_COMPONENTS = ("u", "v", "w")

# Region -> the dims each accumulated quantity carries, once the member axis is
# reduced away. ``mean`` is a vector (one entry per velocity component); ``tke``
# and ``uw`` are scalars per cell.
_FIELD_DIMS = {
    "slab": {
        "mean": ("component", "zlev", "y", "x"),
        "tke": ("zlev", "y", "x"),
        "uw": ("zlev", "y", "x"),
    },
    "station": {
        "mean": ("component", "z", "station"),
        "tke": ("z", "station"),
        "uw": ("z", "station"),
    },
}


def _ensemble_size(state_paths: list[pathlib.Path]) -> int:
    """Members in the window state files, from metadata alone.

    Read before the pass because the accumulators' horizontal stride has to be
    fixed at the first member, which is too late to count them. A metadata-only
    open, so it costs no data read; 1 when the files are unreadable or carry no
    ensemble axis, which is the conservative direction (no stride).
    """
    for path in state_paths:
        try:
            with xarray.open_dataset(path) as ds:
                return int(ds.sizes.get("ensemble", 1))
        except OSError:
            continue
    return 1


def _ensemble_reduction(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, ddof=1 std)`` over the leading member axis.

    NaN propagates rather than being skipped: a member whose field is not
    finite is not a sample from the posterior, so a mean that silently drops it
    would describe an ensemble that does not exist (WP1.2 settled the same
    question the same way for the parameter calibration block). Cells that are
    NaN in *every* member -- masked solids, the edges an interpolated truth
    cannot reach -- come out NaN either way.
    """
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, np.full_like(mean, np.nan)
    return mean, values.std(axis=0, ddof=1)


def _source_variables(prefix: str, result: dict, ensemble: bool) -> dict:
    """``{variable name: (dims, array)}`` for one source (truth / prior / posterior)."""
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for region, quantities in _FIELD_DIMS.items():
        for quantity, dims in quantities.items():
            members = result[f"{region}_{quantity}"]
            name = f"{prefix}_{region}_{quantity}"
            if not ensemble:
                # The truth is a single "member"; keep the same layout minus the
                # spread, so every consumer indexes the two the same way.
                variables[name] = (dims, members[0])
                continue
            mean, spread = _ensemble_reduction(members)
            variables[name] = (dims, mean)
            variables[f"{name}_spread"] = (dims, spread)
            if region == "station":
                # Nested quantile bands, at the station columns only: the same
                # thing over a 3-D slab would multiply the file by the number of
                # quantiles for a figure nobody draws.
                variables[f"{name}_quantile"] = (
                    ("quantile",) + dims,
                    np.quantile(members, STATION_QUANTILES, axis=0),
                )
    return variables


def _eval_fields_dataset(target: dict, sources: dict) -> xarray.Dataset:
    """``eval_fields.nc``: the reduced fields the WP1.5 figures read.

    Reductions only -- never the per-member fields, which are what the window
    state files already hold and what would make this file grow with the
    ensemble. float32 throughout: these are plotted and differenced, not
    accumulated, and the accumulation that needed float64 already happened.
    """
    variables = {}
    for prefix, (result, ensemble) in sources.items():
        for name, (dims, values) in _source_variables(prefix, result, ensemble).items():
            variables[name] = (dims, np.asarray(values, dtype=np.float32))
    return xarray.Dataset(
        variables,
        coords={
            "component": list(_COMPONENTS),
            "zlev": target["z"],
            "y": target["y"],
            "x": target["x"],
            "z": target["station_z"],
            "station": np.arange(target["station_x"].size),
            "station_x": ("station", target["station_x"]),
            "station_y": ("station", target["station_y"]),
            "quantile": list(STATION_QUANTILES),
        },
        attrs={
            "description": (
                "Time-mean velocity, resolved TKE and <u'w'> on evenly spaced "
                "z-slabs and at the sensor station columns, reduced across the "
                "ensemble. Stresses are resolved-only: the subgrid contribution "
                "is not included and is not negligible inside a canopy."
            ),
            "horizontal_stride": target["stride"],
        },
    )


def _hit_rates(predicted: np.ndarray, observed: np.ndarray, w: np.ndarray) -> dict:
    """Hit rate over the slab cells: pooled over the components, and per component.

    Pooled is the headline number the metrics doc asks for (one scalar for the
    mean field); the per-component entries are what says *which* component a
    poor one came from, and each is scored against its own sampling floor.
    """
    pooled = hit_rate(predicted, observed, tolerance_w=w[:, None, None, None])
    entry = {"q": pooled["q"], "n_points": pooled["n_points"]}
    for i, name in enumerate(_COMPONENTS):
        entry[name] = hit_rate(predicted[i], observed[i], tolerance_w=w[i])["q"]
    return entry


def _mean_field_block(
    run_dir: pathlib.Path,
    posterior: MeanFieldCollector,
    truth: MeanFieldCollector,
    prior: MeanFieldCollector | None,
) -> dict | None:
    """Write ``eval_fields.nc`` and return the ``field_metrics`` summary block.

    ``None`` (with a log line) whenever the fields could not be built --
    invariant 3: a state layout colocation refuses, or a truth that shares no
    cell with the assimilation grid, costs this block and nothing else.
    """
    posterior_fields = posterior.result()
    truth_fields = truth.result()
    if posterior_fields is None or truth_fields is None:
        logger.warning(
            "No mean-field metrics: %s",
            posterior.reason or truth.reason or "no frames were accumulated",
        )
        return None

    target = posterior_fields["target"]
    sources = {"truth": (truth_fields, False), "posterior": (posterior_fields, True)}
    prior_fields = prior.result() if prior is not None else None
    if prior_fields is not None:
        if prior_fields["slab_mean"].shape == posterior_fields["slab_mean"].shape:
            sources["prior"] = (prior_fields, True)
        else:
            # A prior written on a different grid than the posterior: comparing
            # the two hit rates would compare two different sets of cells.
            logger.warning(
                "Prior mean fields cover %s cells and the posterior %s -- the "
                "prior half is dropped rather than scored on another grid",
                prior_fields["slab_mean"].shape,
                posterior_fields["slab_mean"].shape,
            )
            prior_fields = None

    _eval_fields_dataset(target, sources).to_netcdf(run_dir / "eval_fields.nc")

    tolerance = truth.sampling_tolerance()
    if tolerance is None:
        tolerance = np.full(len(_COMPONENTS), np.nan)
    if not np.any(np.isfinite(tolerance)):
        logger.info(
            "Mean fields: the truth's window is too short to block-bootstrap a "
            "sampling floor, so the hit rate runs on its relative criterion "
            "alone (W = 0)"
        )

    observed = truth_fields["slab_mean"][0]
    block: dict[str, object] = {
        "n_windows": posterior_fields["n_windows"],
        "frames_per_member": posterior_fields["frames_per_member"],
        "z_levels": [float(v) for v in target["z"]],
        "horizontal_stride": target["stride"],
        "hit_rate_tolerance_w": {
            name: (float(v) if np.isfinite(v) else None)
            for name, v in zip(_COMPONENTS, tolerance)
        },
        "hit_rate_posterior": _hit_rates(
            _ensemble_reduction(posterior_fields["slab_mean"])[0], observed, tolerance
        ),
    }
    if prior_fields is not None:
        block["hit_rate_prior"] = _hit_rates(
            _ensemble_reduction(prior_fields["slab_mean"])[0], observed, tolerance
        )
    return block


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
    # The mean fields ride along on that same pass: the window files total tens
    # of GB at Barcelona scale, and the accumulators are fed from the member
    # slices the sensor extraction already materialises, so they cost no extra
    # read (metrics doc §4.1, master-plan invariant 2).
    station_x, station_y, _ = sensor_sets["assimilation"]
    field_collector = MeanFieldCollector(
        ta["assim_solver_name"],
        station_x,
        station_y,
        n_members=_ensemble_size(state_paths),
    )
    ensemble_series = ensemble_sensor_series(
        state_paths,
        sensor_sets,
        ta["assim_solver_name"],
        float(ta["sim_time"]),
        on_member=field_collector.add_member,
    )

    # Invariant 3, once for both sensor blocks: every score below is a
    # *probabilistic* one and needs the members. A state file written without an
    # ensemble axis (an old ensemble-mean-only artifact) has nothing to score,
    # and must cost its own sensor set rather than the whole metric stage --
    # which would take the parameter, health and state blocks with it.
    scorable = {
        name: coords
        for name, coords in sensor_sets.items()
        if "ensemble" in ensemble_series[name].dims
    }
    for name in sensor_sets:
        if name not in scorable:
            logger.warning(
                "No ensemble dimension in the %s sensor series -- the sensor "
                "metrics and statistics both need the members, so that set is "
                "omitted from both blocks",
                name,
            )

    sensor_metrics = {}
    for name, (sx, sy, sz) in scorable.items():
        m = vector_sensor_metrics(truth_series[name], ensemble_series[name])
        sensor_metrics[name] = {
            "num_sensors": int(np.asarray(sx).size),
            "velocity_vector_rmse": series_stats(m["rmse"]),
            "velocity_vector_energy_score": series_stats(m["energy_score"]),
        }
    summary["sensor_metrics"] = sensor_metrics

    # --- Sensors: window statistics as the verification object (§4.2) --------
    prior_collector = MeanFieldCollector(
        ta["assim_solver_name"],
        station_x,
        station_y,
        n_members=_ensemble_size(state_paths),
        target=field_collector.target,
    )
    summary["sensor_statistics"] = _sensor_statistics(
        run_dir,
        scorable,
        truth_series,
        ensemble_series,
        num_windows,
        float(ta["sim_time"]),
        ta["assim_solver_name"],
        on_prior_member=prior_collector.add_member,
    )

    # --- Mean fields: hit rate + eval_fields.nc (§4.1) ------------------------
    truth_collector = MeanFieldCollector(
        ta["truth_solver_name"],
        station_x,
        station_y,
        target=field_collector.target,
        keep_samples=True,
    )
    if field_collector.target is not None and not field_collector.failed:
        # Only once the ensemble pass has fixed the region: the truth is scored
        # *on the assimilation grid*, so without a target there is nothing to
        # interpolate onto -- and re-reading the truth for a posterior that
        # failed mid-pass would buy nothing either. The block is dropped below
        # in both cases.
        truth_collector.truth_pass(ta, num_windows, int(ta["n_per_window"]))
    field_metrics = _mean_field_block(
        run_dir, field_collector, truth_collector, prior_collector
    )
    if field_metrics is not None:
        summary["field_metrics"] = field_metrics

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
