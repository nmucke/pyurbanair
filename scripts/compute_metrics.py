"""Compute the metrics of a finished assimilation run and write `metrics.yaml`.

    python scripts/compute_metrics.py <run dir>

The run dir is what an assimilation script wrote (e.g.
`<paths.results_dir>/smoother`): `config.yaml`, `true_params.nc`, the truth
state, `run_info.yaml` and `windows/window_{w}_{prior,posterior}_{params,state}.nc`,
`window_{w}_obs.nc` (and a hybrid's `window_{w}_filter_obs.nc`).

Metrics (each summarised as {mean, final, max, min} over its series; the
window-indexed ones also list `per_window` values):
  parameters      per parameter: RMSE and CRPS of the posterior ensemble vs the
                  truth, the prior's, and the reduction 1 - posterior/prior.
  parameter_correlation
                  posterior and prior correlation matrix of the estimated
                  parameters over the members, final window (every window in
                  diagnostics.nc); a time-varying parameter enters as its
                  window mean.
  sgs_health      with the assimilation model's model_discrepancy enabled, per
                  window: the SGS multiplier min and max and the largest
                  saturation fraction over the members, posterior (and prior).
  state           RMSE of the ensemble-mean velocity magnitude |U| vs the truth
                  over time (on a few z-levels, building cells left out).
  sensors         per sensor set (assimilated and held-out validation): RMSE of
                  the ensemble-mean (u, v, w) vector and its energy score
                  (multivariate CRPS), per time step.
  spread_skill    per sensor set: ensemble spread on the same vector norm and the
                  spread-skill ratio (~1 when calibrated); the prior's ratio when
                  the prior states were saved.
  climatology     per sensor set: RMSE of predicting each sensor's time mean of
                  the truth, and the posterior's skill against it.
  sensor_statistics
                  per sensor set: the per-window mean and variance of u/v/w/|U|
                  scored with CRPS, z-score and rank, posterior (and prior, when
                  the prior states were saved; and forecast, when the forecast
                  states were saved), on the posterior's time stamps. With
                  assimilation.replica_dir, the truth replica (a forward run
                  with another turbulence seed) is scored as a one-member
                  prediction under `replica`: the noise floor.
  observation     per stage (smoother: one value per window; filter: one per
                  cycle): RMSE of the forecast and analysis predicted
                  observations vs the noisy observations, their ratio and the
                  diagonal innovation chi^2 (~1 when R and the spread are
                  honest); the smoother's normalized data mismatch O_N. A
                  hybrid's smoother stops before the posterior forecast, so
                  its stage has only O_N.
  desroziers      per stage with an analysis: the observation error std
                  estimated from the forecast and analysis residuals
                  (Desroziers), the obs_std the run used, and their ratio.

Arrays too large for metrics.yaml go into `diagnostics.nc` next to it:
  posterior_parameter_members, prior_parameter_members  (window, ensemble, parameter)
  posterior_parameter_correlation, prior_parameter_correlation
                                          (window, parameter, parameter_j)
  true_parameter                          (parameter), NaN where it varies in time

The window state files can be tens of GB, so they are read one member at a
time; the truth (and the replica) one window at a time.
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import xarray
from evaluation.scores import (
    _skill_score,
    compute_parameter_metrics,
    crps_ensemble,
    data_mismatch,
    data_mismatch_summary,
    observation_fit,
    series_stats,
    spread_skill,
    vector_sensor_metrics,
    window_statistics_summary,
)
from evaluation.sensors import window_sampling_std, window_statistics
from evaluation.turbulence import streaming_state_rmse
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "utils"))

from helper_functions import (  # noqa: E402
    case_stl_path,
    concat_windows,
    global_time,
    open_forward_run,
    open_truth,
    save_yaml,
    sensor_series,
    sensor_sets,
)


def run(run_dir: pathlib.Path) -> None:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    run_info = OmegaConf.load(run_dir / "run_info.yaml")
    num_windows = int(cfg.assimilation.num_windows)
    sim_time = float(cfg.time.simulation_time)
    # Truth frames are matched to the ensemble's by time, within half a frame.
    tolerance = 0.5 * float(cfg.time.output_frequency)
    windows = run_dir / "windows"

    def window_files(name: str) -> list[pathlib.Path]:
        return [windows / f"window_{w}_{name}.nc" for w in range(num_windows)]

    metrics: dict = {}
    # Arrays for the figures, written to diagnostics.nc at the end.
    diagnostics: dict[str, xarray.DataArray] = {}

    # --- Parameters -------------------------------------------------------------
    true_params = xarray.load_dataset(run_dir / "true_params.nc")
    metrics["parameters"] = _parameter_metrics(
        concat_windows(window_files("posterior_params"), sim_time),
        true_params,
        concat_windows(window_files("prior_params"), sim_time),
        num_windows,
    )
    names = _estimated_names(cfg, window_files("posterior_params")[0])
    if names:
        metrics["parameter_correlation"] = {}
        for kind in ("posterior", "prior"):
            members, correlation = _parameter_correlation(
                window_files(f"{kind}_params"), names
            )
            diagnostics[f"{kind}_parameter_members"] = members
            diagnostics[f"{kind}_parameter_correlation"] = correlation
            final = correlation.isel(window=-1).values
            metrics["parameter_correlation"][kind] = {
                a: {b: _finite(final[i, j]) for j, b in enumerate(names)}
                for i, a in enumerate(names)
            }
        diagnostics["true_parameter"] = _static_truth(true_params, names)

    # --- One pass over truth and ensemble, window by window ----------------------
    sets = sensor_sets(cfg)
    # Prior: the free run with the prior parameters; forecast: the filter's
    # forecasts. Both are scored on the posterior's time stamps.
    extra_files = {
        kind: window_files(f"{kind}_state") for kind in ("prior", "forecast")
    }
    extra_files = {
        kind: files
        for kind, files in extra_files.items()
        if all(f.exists() for f in files)
    }
    truth = open_truth(cfg, run_dir)
    # The noise floor: the truth configuration with another turbulence seed,
    # scored like the prior/forecast ensembles, as one member.
    replica_dir = cfg.assimilation.get("replica_dir")
    replica = None if replica_dir is None else open_forward_run(cfg, replica_dir)
    stl_path = case_stl_path(cfg)
    frames_per_window = truth.sizes["time"] // num_windows

    truth_series: dict[str, list] = {name: [] for name in sets}
    posterior_series: dict[str, list] = {name: [] for name in sets}
    extra_series: dict[str, dict[str, list]] = {
        kind: {name: [] for name in sets}
        for kind in [*extra_files, *(["replica"] if replica is not None else [])]
    }
    state_rmse = []
    for w in range(num_windows):
        window = slice(w * frames_per_window, (w + 1) * frames_per_window)
        window_truth = truth.isel(time=window)
        for name, points in sets.items():
            truth_series[name].append(
                sensor_series(window_truth, points, cfg.truth_model.solver_name)
            )
            if replica is not None:
                window_replica = replica.isel(time=window)
                extra_series["replica"][name].append(
                    sensor_series(
                        window_replica, points, cfg.truth_model.solver_name
                    ).expand_dims(ensemble=1)
                )

        series, mean_state = _read_ensemble(
            window_files("posterior_state")[w],
            sets,
            cfg.assim_model.solver_name,
            w,
            sim_time,
        )
        for name in sets:
            posterior_series[name].append(series[name])
        state_rmse.append(
            streaming_state_rmse(
                window_truth.sel(
                    time=mean_state.time, method="nearest", tolerance=tolerance
                ),
                mean_state,
                stl_path,
            )
        )

        times = posterior_series[next(iter(sets))][-1].time.values
        for kind, files in extra_files.items():
            series, _ = _read_ensemble(
                files[w], sets, cfg.assim_model.solver_name, w, sim_time, times
            )
            for name in sets:
                extra_series[kind][name].append(series[name])
    truth.close()
    if replica is not None:
        replica.close()

    # --- SGS health ---------------------------------------------------------------
    discrepancy = OmegaConf.select(
        cfg, "assim_model.forward_model.model_discrepancy.enabled", default=False
    )
    if discrepancy:
        metrics["sgs_health"] = {
            kind: _sgs_health(files)
            for kind, files in (
                ("posterior", window_files("posterior_state")),
                ("prior", extra_files.get("prior")),
            )
            if files is not None
        }

    metrics["state"] = {"vel_magnitude_rmse": series_stats(np.concatenate(state_rmse))}

    # --- Sensor metrics -----------------------------------------------------------
    for block in ("sensors", "spread_skill", "climatology", "sensor_statistics"):
        metrics[block] = {}
    for name in sets:
        truth_s = xarray.concat(truth_series[name], dim="time")
        posterior_s = xarray.concat(posterior_series[name], dim="time")
        extra = {
            kind: xarray.concat(series[name], dim="time")
            for kind, series in extra_series.items()
        }
        n_members = int(posterior_s.sizes["ensemble"])

        vector = vector_sensor_metrics(truth_s, posterior_s)
        metrics["sensors"][name] = {
            "num_sensors": int(posterior_s.sizes["sensor"]),
            "velocity_vector_rmse": series_stats(vector["rmse"]),
            "velocity_vector_energy_score": series_stats(vector["energy_score"]),
        }
        metrics["spread_skill"][name] = {
            "ratio": _finite(spread_skill(vector["spread"], vector["rmse"], n_members)),
            "spread": series_stats(vector["spread"]),
        }
        if "prior" in extra:
            prior = vector_sensor_metrics(truth_s, extra["prior"])
            metrics["spread_skill"][name]["prior_ratio"] = _finite(
                spread_skill(prior["spread"], prior["rmse"], n_members)
            )

        # Each sensor's time mean of the truth as a one-member ensemble. It is
        # taken from the scored period itself, so the baseline is in-sample.
        climatology = vector_sensor_metrics(
            truth_s,
            truth_s.mean("time").expand_dims(ensemble=1, time=posterior_s.time.values),
        )
        metrics["climatology"][name] = {
            "velocity_vector_rmse": series_stats(climatology["rmse"]),
            "rmse_skill_vs_climatology": _skill_score(
                vector["rmse"], climatology["rmse"], name, "climatology RMSE"
            )[1],
        }

        truth_stats = window_statistics(truth_s, sim_time, num_windows, label=name)
        posterior_stats = window_statistics(
            posterior_s, sim_time, num_windows, label=name
        )
        posterior_std = window_sampling_std(posterior_s, sim_time, num_windows)
        stats = window_statistics_summary(
            truth_stats,
            posterior_stats,
            posterior_sampling_std=posterior_std,
            label=name,
        )
        if "posterior" in stats:
            _add_per_window_crps(stats["posterior"], posterior_stats, truth_stats)
        # One summary per reference; the posterior is scored the same each time,
        # so only the reference's block and skill keys are added.
        for kind, series in extra.items():
            reference_stats = window_statistics(
                series, sim_time, num_windows, label=name
            )
            scored = window_statistics_summary(
                truth_stats,
                posterior_stats,
                prior_stats=reference_stats,
                posterior_sampling_std=posterior_std,
                prior_sampling_std=window_sampling_std(series, sim_time, num_windows),
                label=name,
                reference=kind,
            )
            if kind not in scored:  # no ensemble dimension: the block is empty
                continue
            _add_per_window_crps(scored[kind], reference_stats, truth_stats)
            stats[kind] = scored[kind]
            for key, entry in stats["posterior"].items():
                for skill in (f"{kind}_crps_mean", f"crps_reduction_vs_{kind}"):
                    entry[skill] = scored["posterior"][key][skill]
        metrics["sensor_statistics"][name] = stats

    # --- Observation space ----------------------------------------------------------
    # The smoother stage reads a smoother's (or hybrid's) ESMDA steps per window;
    # the filter stage a filter's forecast/analysis per cycle.
    # A hybrid's smoother stops before the posterior forecast (its filter takes
    # over), so it has no analysis to fit: only its O_N is scored.
    hybrid = "smoother" in run_info and "filter" in run_info
    metrics["observation"] = {}
    metrics["desroziers"] = {}
    if "smoother" in run_info:
        files = window_files("obs")
        smoother: dict = {}
        if not hybrid:
            smoother, metrics["desroziers"]["smoother"] = _observation_metrics(files, 1)
        smoother["data_mismatch"] = _data_mismatch(files)
        metrics["observation"]["smoother"] = smoother
    if "filter" in run_info:
        metrics["observation"]["filter"], metrics["desroziers"]["filter"] = (
            _observation_metrics(
                window_files("filter_obs" if hybrid else "obs"),
                int(run_info.cycles_per_window),
            )
        )

    save_yaml(metrics, run_dir / "metrics.yaml")
    print(f"Saved metrics in {run_dir / 'metrics.yaml'}")
    if diagnostics:
        xarray.Dataset(diagnostics).to_netcdf(run_dir / "diagnostics.nc")
        print(f"Saved diagnostics in {run_dir / 'diagnostics.nc'}")


def _finite(value: float) -> float | None:
    """`value`, or None (null in the YAML) when it is not finite."""
    return float(value) if np.isfinite(value) else None


def _observation_metrics(files: list[pathlib.Path], n_cycles: int) -> tuple[dict, dict]:
    """The observation and desroziers blocks of one stage, one value per update.

    Each window's observations hold `n_cycles` equal chunks, one per update; its
    first step is the forecast and its last the analysis.
    """
    fits = []
    obs_std = []
    for path in files:
        ds = xarray.load_dataset(path)
        obs, std = ds.obs.values, ds.obs_std.values
        pred = ds.pred_obs.transpose("esmda_step", "observation", "ensemble").values
        for chunk in np.split(np.arange(obs.size), n_cycles):
            fits.append(
                observation_fit(obs[chunk], std[chunk], pred[0, chunk], pred[-1, chunk])
            )
        obs_std.append(std)
    series = {key: np.array([fit[key] for fit in fits]) for key in fits[0]}

    observation = {
        key: series_stats(series[key])
        for key in (
            "forecast_rmse",
            "analysis_rmse",
            "rmse_ratio",
            "innovation_chi2_diag",
        )
    }
    estimated = series_stats(series["obs_std_estimated"])
    used = float(np.sqrt(np.mean(np.concatenate(obs_std) ** 2)))
    desroziers = {
        "obs_std_estimated": estimated,
        "obs_std_used": used,
        "ratio": estimated["mean"] / used if estimated is not None and used else None,
    }
    return observation, desroziers


def _data_mismatch(files: list[pathlib.Path]) -> dict | None:
    """O_N of every ESMDA step's forecast, pooled over the windows."""
    per_window = []
    for path in files:
        ds = xarray.load_dataset(path)
        pred = ds.pred_obs.transpose("esmda_step", "observation", "ensemble").values
        per_window.append(
            np.stack([data_mismatch(ds.obs.values, p, ds.obs_std.values) for p in pred])
        )
    return data_mismatch_summary(
        np.concatenate(per_window, axis=1),  # (step, window * member)
        ds.sizes["observation"],
        per_window=per_window,
    )


def _per_window(series: np.ndarray, num_windows: int) -> list[float | None]:
    """A window-indexed series' value per window; a per-knot or per-cycle series
    (time-varying parameters) is averaged within each window."""
    values = np.asarray(series, dtype=float).reshape(num_windows, -1)
    with warnings.catch_warnings():  # an all-NaN window is a null, not a warning
        warnings.simplefilter("ignore", RuntimeWarning)
        return [_finite(v) for v in np.nanmean(values, axis=1)]


def _with_per_window(series: np.ndarray, num_windows: int) -> dict | None:
    """`series_stats` plus the `per_window` values."""
    stats = series_stats(series)
    if stats is not None:
        stats["per_window"] = _per_window(series, num_windows)
    return stats


def _add_per_window_crps(
    block: dict, stats: dict[str, xarray.DataArray], truth: dict[str, xarray.DataArray]
) -> None:
    """Add `per_window` to each `crps` of a sensor_statistics block: the CRPS
    averaged over sensors per window, as `window_statistics_summary` scores it."""
    for statistic, members in stats.items():
        for quantity in members.quantity.values:
            entry = block.get(f"{statistic}_{quantity}", {})
            if entry.get("crps") is None:
                continue
            ens = members.sel(quantity=quantity).transpose("ensemble", "window", ...)
            target = truth[statistic].sel(quantity=quantity).transpose("window", ...)
            crps = crps_ensemble(
                np.asarray(ens.values, dtype=float).reshape(ens.sizes["ensemble"], -1),
                np.asarray(target.values, dtype=float).ravel(),
            )
            entry["crps"]["per_window"] = _per_window(crps, ens.sizes["window"])


def _estimated_names(cfg: DictConfig, params_file: pathlib.Path) -> list[str]:
    """The estimated parameters in a window parameter file, in config order."""
    with xarray.open_dataset(params_file) as ds:
        available = list(ds.data_vars)
    selected = cfg.assimilation.params_to_estimate
    return [n for n in (available if selected is None else selected) if n in available]


def _parameter_correlation(
    files: list[pathlib.Path], names: list[str]
) -> tuple[xarray.DataArray, xarray.DataArray]:
    """Per window: the members (window, ensemble, parameter) and their correlation
    matrix (window, parameter, parameter_j). A time-varying parameter enters as
    its window mean; a parameter without spread has NaN correlations."""
    members = []
    for path in files:
        ds = xarray.load_dataset(path)
        members.append(
            np.stack(
                [
                    ds[n].mean([d for d in ds[n].dims if d != "ensemble"]).values
                    for n in names
                ],
                axis=1,
            )
        )
    stacked = np.stack(members)  # (window, ensemble, parameter)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.stack(
            [np.atleast_2d(np.corrcoef(m, rowvar=False)) for m in stacked]
        )
    window = np.arange(len(files))
    return (
        xarray.DataArray(
            stacked,
            dims=("window", "ensemble", "parameter"),
            coords={"window": window, "parameter": names},
        ),
        xarray.DataArray(
            correlation,
            dims=("window", "parameter", "parameter_j"),
            coords={"window": window, "parameter": names, "parameter_j": names},
        ),
    )


def _static_truth(truth: xarray.Dataset, names: list[str]) -> xarray.DataArray:
    """The truth value per parameter; NaN for a parameter it varies in time."""
    values = []
    for name in names:
        knots = np.asarray(truth[name].values if name in truth else np.nan).ravel()
        values.append(float(knots[0]) if np.allclose(knots, knots[0]) else np.nan)
    return xarray.DataArray(values, dims="parameter", coords={"parameter": names})


def _sgs_health(files: list[pathlib.Path]) -> dict[str, list[float | None]]:
    """Per window: the SGS multiplier min and max over the members and the
    largest member saturation fraction.

    From each state file's `model_discrepancy_by_member` attribute (JSON, one
    entry per member), whose `native_diagnostics` is the solver's
    `key=value` report (pyudales `_record_discrepancy`).
    """
    out: dict[str, list[float | None]] = {
        "multiplier_min": [],
        "multiplier_max": [],
        "saturation_fraction_max": [],
    }
    for path in files:
        with xarray.open_dataset(path) as ds:
            raw = ds.attrs.get("model_discrepancy_by_member")
        reports = [
            {
                key: float(value)
                for line in member["native_diagnostics"].splitlines()
                if "=" in line
                for key, value in [line.split("=", 1)]
            }
            for member in (json.loads(raw) if raw else [])
            if member and "native_diagnostics" in member
        ]
        if not reports:
            print(f"{path.name}: no per-member SGS diagnostics; sgs_health is null")
        for key, source, reduce in (
            ("multiplier_min", "multiplier_min", min),
            ("multiplier_max", "multiplier_max", max),
            ("saturation_fraction_max", "saturation_fraction", max),
        ):
            values = [report[source] for report in reports]
            out[key].append(reduce(values) if values else None)
    return out


def _parameter_metrics(
    posterior: xarray.Dataset,
    truth: xarray.Dataset,
    prior: xarray.Dataset,
    num_windows: int,
) -> dict:
    """RMSE and CRPS per parameter, posterior and prior, and the reduction."""
    out = {}
    for name, m in compute_parameter_metrics(posterior, truth, prior).items():
        entry: dict[str, object] = {
            "rmse": _with_per_window(m["rmse"], num_windows),
            "crps": _with_per_window(m["crps"], num_windows),
        }
        for score in ("rmse", "crps"):
            if f"prior_{score}" in m:
                post = float(np.nanmean(m[score]))
                pri = float(np.nanmean(m[f"prior_{score}"]))
                entry[f"prior_{score}_mean"] = pri
                entry[f"{score}_reduction_vs_prior"] = 1 - post / pri if pri else None
        out[name] = entry
    return out


def _read_ensemble(
    path: pathlib.Path,
    sets: dict,
    solver_name: str,
    window: int,
    sim_time: float,
    times: np.ndarray | None = None,
) -> tuple[dict[str, xarray.DataArray], xarray.Dataset]:
    """Sensor series and ensemble-mean state of one window file, member by member.

    The series take the run's global time axis. Given the global `times` of
    another file of the same window (the posterior's), only the frames at
    those times are read.
    """
    with xarray.open_dataset(path) as ds:
        ds = ds.assign_coords(time=global_time(ds.time, window, sim_time))
        if times is not None:
            frames = np.abs(ds.time.values[None, :] - times[:, None]).argmin(axis=1)
            assert np.allclose(ds.time.values[frames], times, atol=1e-6), path
            ds = ds.isel(time=frames)
        n_members = ds.sizes["ensemble"]
        series: dict[str, list] = {name: [] for name in sets}
        total = None
        for m in range(n_members):
            member = ds[["u", "v", "w"]].isel(ensemble=slice(m, m + 1)).load()
            for name, points in sets.items():
                series[name].append(sensor_series(member, points, solver_name))
            total = (
                member.isel(ensemble=0)
                if total is None
                else total + member.isel(ensemble=0)
            )
    assert total is not None
    return (
        {name: xarray.concat(parts, dim="ensemble") for name, parts in series.items()},
        total / n_members,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    run(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
