"""Compute the metrics of a finished assimilation run and write `metrics.yaml`.

    python scripts/compute_metrics.py <run dir>

The run dir is what an assimilation script wrote (e.g.
`<paths.results_dir>/smoother`): `config.yaml`, `true_params.nc`, the truth
state, `run_info.yaml` and `windows/window_{w}_{prior,posterior}_{params,state}.nc`,
`window_{w}_obs.nc` (and a hybrid's `window_{w}_filter_obs.nc`).

Metrics (each summarised as {mean, final, max, min} over its series):
  parameters      per parameter: RMSE and CRPS of the posterior ensemble vs the
                  truth, the prior's, and the reduction 1 - posterior/prior.
  state           RMSE of the ensemble-mean velocity magnitude |U| vs the truth
                  over time (on a few z-levels).
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
                  states were saved), on the posterior's time stamps.
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

The window state files can be tens of GB, so they are read one member at a
time; the truth is read one window at a time.
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import argparse
import pathlib
import sys

import numpy as np
import xarray
from evaluation.scores import (
    _skill_score,
    compute_parameter_metrics,
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
    concat_windows,
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
    windows = run_dir / "windows"

    def window_files(name: str) -> list[pathlib.Path]:
        return [windows / f"window_{w}_{name}.nc" for w in range(num_windows)]

    metrics: dict = {}

    # --- Parameters -------------------------------------------------------------
    metrics["parameters"] = _parameter_metrics(
        concat_windows(window_files("posterior_params"), sim_time),
        xarray.load_dataset(run_dir / "true_params.nc"),
        concat_windows(window_files("prior_params"), sim_time),
    )

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
    frames_per_window = truth.sizes["time"] // num_windows

    truth_series: dict[str, list] = {name: [] for name in sets}
    posterior_series: dict[str, list] = {name: [] for name in sets}
    extra_series: dict[str, dict[str, list]] = {
        kind: {name: [] for name in sets} for kind in extra_files
    }
    state_rmse = []
    for w in range(num_windows):
        window_truth = truth.isel(
            time=slice(w * frames_per_window, (w + 1) * frames_per_window)
        )
        for name, points in sets.items():
            truth_series[name].append(
                sensor_series(window_truth, points, cfg.truth_model.solver_name)
            )

        series, mean_state = _read_ensemble(
            window_files("posterior_state")[w],
            sets,
            cfg.assim_model.solver_name,
            w * sim_time,
        )
        for name in sets:
            posterior_series[name].append(series[name])
        state_rmse.append(streaming_state_rmse(window_truth, mean_state))

        times = posterior_series[next(iter(sets))][-1].time.values
        for kind, files in extra_files.items():
            series, _ = _read_ensemble(
                files[w], sets, cfg.assim_model.solver_name, w * sim_time, times
            )
            for name in sets:
                extra_series[kind][name].append(series[name])
    truth.close()

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
        # One summary per reference; the posterior is scored the same each time,
        # so only the reference's block and skill keys are added.
        for kind, series in extra.items():
            scored = window_statistics_summary(
                truth_stats,
                posterior_stats,
                prior_stats=window_statistics(
                    series, sim_time, num_windows, label=name
                ),
                posterior_sampling_std=posterior_std,
                prior_sampling_std=window_sampling_std(series, sim_time, num_windows),
                label=name,
                reference=kind,
            )
            if kind not in scored:  # no ensemble dimension: the block is empty
                continue
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


def _parameter_metrics(
    posterior: xarray.Dataset, truth: xarray.Dataset, prior: xarray.Dataset
) -> dict:
    """RMSE and CRPS per parameter, posterior and prior, and the reduction."""
    out = {}
    for name, m in compute_parameter_metrics(posterior, truth, prior).items():
        entry: dict[str, object] = {
            "rmse": series_stats(m["rmse"]),
            "crps": series_stats(m["crps"]),
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
    t_start: float,
    times: np.ndarray | None = None,
) -> tuple[dict[str, xarray.DataArray], xarray.Dataset]:
    """Sensor series and ensemble-mean state of one window file, member by member.

    The sensor series get a global time axis starting at `t_start`. Given the
    global `times` of another file of the same window (the posterior's), only
    the frames at those times are read and they take that axis. Files differ in
    how they stamp time, but every one ends on the window's end, so frames are
    matched by their time before it.
    """
    with xarray.open_dataset(path) as ds:
        if times is not None:
            before_end = ds.time.values.astype(float)
            before_end = before_end - before_end[-1]
            wanted = times - times[-1]
            frames = np.abs(before_end[None, :] - wanted[:, None]).argmin(axis=1)
            assert np.allclose(before_end[frames], wanted, atol=1e-6), path
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
        time = ds.time.values.astype(float)
    assert total is not None
    global_time = time - time[0] + t_start if times is None else times
    return (
        {
            name: xarray.concat(parts, dim="ensemble").assign_coords(time=global_time)
            for name, parts in series.items()
        },
        total / n_members,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    run(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
