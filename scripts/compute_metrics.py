"""Compute the metrics of a finished assimilation run and write `metrics.yaml`.

    python scripts/compute_metrics.py <run dir>

The run dir is what an assimilation script wrote (e.g.
`<paths.results_dir>/smoother`): `config.yaml`, `true_params.nc`, the truth
state and `windows/window_{w}_{prior,posterior}_{params,state}.nc`.

Metrics (each summarised as {mean, final, max, min} over its series):
  parameters      per parameter: RMSE and CRPS of the posterior ensemble vs the
                  truth, the prior's, and the reduction 1 - posterior/prior.
  state           RMSE of the ensemble-mean velocity magnitude |U| vs the truth
                  over time (on a few z-levels).
  sensors         per sensor set (assimilated and held-out validation): RMSE of
                  the ensemble-mean (u, v, w) vector and its energy score
                  (multivariate CRPS), per time step.
  sensor_statistics
                  per sensor set: the per-window mean and variance of u/v/w/|U|
                  scored with CRPS, z-score and rank, posterior (and prior, when
                  the prior states were saved).

The window state files can be tens of GB, so they are read one member at a
time; the truth is read one window at a time.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import xarray
from evaluation.scores import (
    compute_parameter_metrics,
    series_stats,
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
    prior_files = window_files("prior_state")
    has_prior = all(f.exists() for f in prior_files)
    truth = open_truth(cfg, run_dir)
    frames_per_window = truth.sizes["time"] // num_windows

    truth_series: dict[str, list] = {name: [] for name in sets}
    posterior_series: dict[str, list] = {name: [] for name in sets}
    prior_series: dict[str, list] = {name: [] for name in sets}
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

        if has_prior:
            series, _ = _read_ensemble(
                prior_files[w], sets, cfg.assim_model.solver_name, w * sim_time
            )
            for name in sets:
                prior_series[name].append(series[name])
    truth.close()

    metrics["state"] = {"vel_magnitude_rmse": series_stats(np.concatenate(state_rmse))}

    # --- Sensor metrics -----------------------------------------------------------
    metrics["sensors"] = {}
    metrics["sensor_statistics"] = {}
    for name in sets:
        truth_s = xarray.concat(truth_series[name], dim="time")
        posterior_s = xarray.concat(posterior_series[name], dim="time")
        prior_s = xarray.concat(prior_series[name], dim="time") if has_prior else None

        vector = vector_sensor_metrics(truth_s, posterior_s)
        metrics["sensors"][name] = {
            "num_sensors": int(posterior_s.sizes["sensor"]),
            "velocity_vector_rmse": series_stats(vector["rmse"]),
            "velocity_vector_energy_score": series_stats(vector["energy_score"]),
        }
        metrics["sensor_statistics"][name] = window_statistics_summary(
            window_statistics(truth_s, sim_time, num_windows, label=name),
            window_statistics(posterior_s, sim_time, num_windows, label=name),
            prior_stats=(
                window_statistics(prior_s, sim_time, num_windows, label=name)
                if prior_s is not None
                else None
            ),
            posterior_sampling_std=window_sampling_std(
                posterior_s, sim_time, num_windows
            ),
            prior_sampling_std=(
                window_sampling_std(prior_s, sim_time, num_windows)
                if prior_s is not None
                else None
            ),
            label=name,
        )

    save_yaml(metrics, run_dir / "metrics.yaml")
    print(f"Saved metrics in {run_dir / 'metrics.yaml'}")


def _parameter_metrics(
    posterior: xarray.Dataset, truth: xarray.Dataset, prior: xarray.Dataset
) -> dict:
    """RMSE and CRPS per parameter, posterior and prior, and the reduction."""
    out = {}
    for name, m in compute_parameter_metrics(posterior, truth, prior).items():
        entry = {"rmse": series_stats(m["rmse"]), "crps": series_stats(m["crps"])}
        for score in ("rmse", "crps"):
            if f"prior_{score}" in m:
                post = float(np.nanmean(m[score]))
                pri = float(np.nanmean(m[f"prior_{score}"]))
                entry[f"prior_{score}_mean"] = pri
                entry[f"{score}_reduction_vs_prior"] = 1 - post / pri if pri else None
        out[name] = entry
    return out


def _read_ensemble(
    path: pathlib.Path, sets: dict, solver_name: str, t_start: float
) -> tuple[dict[str, xarray.DataArray], xarray.Dataset]:
    """Sensor series and ensemble-mean state of one window file, member by member.

    The sensor series get a global time axis starting at `t_start`.
    """
    with xarray.open_dataset(path) as ds:
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
    global_time = time - time[0] + t_start
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
