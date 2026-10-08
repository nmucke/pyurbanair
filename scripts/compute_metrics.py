"""Compute the metrics of a finished assimilation run and write `metrics.yaml`.

    python scripts/compute_metrics.py <run dir> [key=value ...]

The overrides are applied to the run's config.yaml, e.g.
`assimilation.replica_dir=<forward run dir>` to score an existing run against
a replica made after it.

The run dir is what an assimilation script wrote (e.g.
`<paths.results_dir>/smoother`): `config.yaml`, `true_params.nc`, the truth
state, `run_info.yaml` and `windows/window_{w}_{prior,posterior}_{params,state}.nc`,
`window_{w}_obs.nc` (and a hybrid's `window_{w}_filter_obs.nc`).

The scores are the statistics: field_statistics, canopy_profiles,
sensor_statistics, sensor_distributions and spectra. Truth and members are
different turbulent realisations, so the instantaneous blocks (sensors,
spread_skill) are sanity checks only. Read every score against the `replica`
(and `truth_halves`) floor when there is one.

"Sources" are the posterior, the prior (when the prior states were saved), the
forecast (when the forecast states were saved) and, with
assimilation.replica_dir, the truth replica (a forward run with another
turbulence seed) as a one-member prediction under `replica`: the noise floor.

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
  field_statistics
                  per statistic (time-mean u, v, w; TKE; resolved u'w', each per
                  member and window on the cell centres, averaged over the
                  members) and source: its RMSE against the truth's over the
                  fluid cells per window, and `level_rmse` per height (z listed
                  once), RMS over the windows. A truth on another grid is
                  interpolated onto the ensemble's cell centres.
  canopy_profiles per profile (u, tke, uw: the averages over the fluid cells of
                  each level) and source: RMSE over z of the ensemble-mean
                  profile against the truth's, per window.
  spectra         per component (u, v, w), height group (above_canopy,
                  in_canopy: the fully fluid lines along the periodic y) and
                  source: the energy ratio to the truth's spanwise spectrum in
                  dB in the large (> 8 dy), mid (4-8 dy) and near_cutoff
                  (2-4 dy) wavelength bands, and the log-spectral distance, of
                  the ensemble's median spectrum, per window. Native grids,
                  no interpolation.
  sensors         sanity check, per sensor set (assimilated and held-out
                  validation): RMSE of the ensemble-mean (u, v, w) vector and
                  its energy score (multivariate CRPS), per time step.
  spread_skill    sanity check, per sensor set: ensemble spread on the same vector norm and the
                  spread-skill ratio (~1 when calibrated); the prior's ratio when
                  the prior states were saved.
  climatology     per sensor set: RMSE of predicting each sensor's time mean of
                  the truth, and the posterior's skill against it.
  sensor_statistics
                  per sensor set: the per-window mean and variance of u/v/w/|U|
                  scored with CRPS, z-score and rank, per source, on the
                  posterior's time stamps.
  sensor_distributions
                  per sensor set, quantity (u, v, w, magnitude) and source: the
                  distance between its sensor values (frames x sensors x
                  members) and the truth's on the posterior's time stamps, per
                  window: W2 and its location/scale/shape split and KL, pooled
                  (the headline); `*_per_sensor` (normalised by each sensor's
                  truth std, averaged); `*_member_median`. `truth_halves`: the
                  truth's first half-window against its second.
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
  profile_<source>       (window, [ensemble,] profile_quantity, z) canopy profiles
  spectrum_<source>      (window, [ensemble,] component, group, k) spanwise
                         spectra, k in cycles/m; spectrum_dy the grid spacing
  building_height        (bound) lowest and highest building top
  sensor_bin_edges_<set>, sensor_density_<set>, sensor_quantiles_<set>
                         per quantity: shared bins and each source's density and
                         quantiles, pooled over sensors, windows and members

The window state files can be tens of GB, so they are read one member at a
time; the truth (and the replica) one window at a time.
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import argparse
import json
import pathlib
import sys

import numpy as np
import xarray
from evaluation.scores import (
    _skill_score,
    compute_parameter_metrics,
    data_mismatch,
    data_mismatch_summary,
    member_correlation,
    observation_fit,
    sensor_distribution_summary,
    series_stats,
    spread_skill,
    vector_sensor_metrics,
    window_series_stats,
    window_statistics_summary,
)
from evaluation.sensors import (
    QUANTITIES,
    quantity_series,
    window_sampling_std,
    window_statistics,
)
from evaluation.turbulence import (
    ON_TRUTH_GRID,
    field_metric_blocks,
    member_field_reductions,
    score_window_fields,
)
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


def run(run_dir: pathlib.Path, overrides: list[str] | None = None) -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(run_dir / "config.yaml"), OmegaConf.from_dotlist(overrides or [])
    )
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
    posterior_params = concat_windows(window_files("posterior_params"), sim_time)
    metrics["parameters"] = _parameter_metrics(
        posterior_params,
        true_params,
        concat_windows(window_files("prior_params"), sim_time),
        num_windows,
    )
    available = [str(n) for n in posterior_params.data_vars]
    selected = cfg.assimilation.params_to_estimate
    names = [n for n in (available if selected is None else selected) if n in available]
    if names:
        metrics["parameter_correlation"] = {}
        for kind in ("posterior", "prior"):
            members = _parameter_members(window_files(f"{kind}_params"), names)
            correlation = member_correlation(members)
            diagnostics[f"{kind}_parameter_members"] = members
            diagnostics[f"{kind}_parameter_correlation"] = correlation
            metrics["parameter_correlation"][kind] = {
                a: {b: _finite(v) for b, v in zip(names, row)}
                for a, row in zip(names, correlation.values[-1])
            }
        diagnostics["true_parameter"] = _static_truth(true_params, names)

    # --- One pass over truth and ensembles, window by window ---------------------
    sets = sensor_sets(cfg)
    # The truth and the noise floor (the truth configuration with another
    # turbulence seed) are one-member ensembles on the truth's grid.
    states = {"truth": open_truth(cfg, run_dir)}
    if cfg.assimilation.get("replica_dir") is not None:
        states["replica"] = open_forward_run(cfg, cfg.assimilation.replica_dir)
    # Prior: the free run with the prior parameters; forecast: the filter's
    # forecasts. Both are scored on the posterior's time stamps.
    ensembles = {"posterior": window_files("posterior_state")}
    for kind in ("prior", "forecast"):
        if all(f.exists() for f in window_files(f"{kind}_state")):
            ensembles[kind] = window_files(f"{kind}_state")
    stl_path = case_stl_path(cfg)
    frames_per_window = states["truth"].sizes["time"] // num_windows

    # Per source and window: the sensor series of each set (`series_by`) and
    # the scored field reductions (`fields`).
    series_by: dict[str, list[dict]] = {kind: [] for kind in (*states, *ensembles)}
    fields = []
    for w in range(num_windows):
        window = slice(w * frames_per_window, (w + 1) * frames_per_window)
        reduced = {}
        for kind, state in states.items():
            series, reduced[kind] = _read_ensemble(
                state.isel(time=window).expand_dims("ensemble"),
                sets,
                cfg.truth_model.solver_name,
                stl_path,
            )
            series_by[kind].append(series)
        times = None  # the posterior's, which the others are read on
        for kind, paths in ensembles.items():
            with xarray.open_dataset(paths[w]) as ds:
                ds = ds.assign_coords(time=global_time(ds.time, w, sim_time))
                series, reduced[kind] = _read_ensemble(
                    ds, sets, cfg.assim_model.solver_name, stl_path, times
                )
                if times is None:
                    times = ds.time.values
            series_by[kind].append(series)
        fields.append(score_window_fields(reduced, stl_path))
    for state in states.values():
        state.close()

    # --- SGS health ---------------------------------------------------------------
    if OmegaConf.select(
        cfg, "assim_model.forward_model.model_discrepancy.enabled", default=False
    ):
        metrics["sgs_health"] = {
            kind: _sgs_health(ensembles[kind])
            for kind in ("posterior", "prior")
            if kind in ensembles
        }

    # --- Field statistics, canopy profiles and spectra ------------------------------
    blocks, arrays = field_metric_blocks(fields)
    metrics.update(blocks)
    diagnostics.update(arrays)

    # --- Sensor metrics -----------------------------------------------------------
    for block in (
        "sensors",
        "spread_skill",
        "climatology",
        "sensor_statistics",
        "sensor_distributions",
    ):
        metrics[block] = {}
    for name in sets:
        per_window = {
            kind: [s[name] for s in per_source]
            for kind, per_source in series_by.items()
        }
        extra = {
            kind: xarray.concat(series, dim="time")
            for kind, series in per_window.items()
        }
        truth_s = extra.pop("truth").isel(ensemble=0)
        posterior_s = extra.pop("posterior")
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
            stats[kind] = scored[kind]
            for key, entry in stats["posterior"].items():
                for skill in (f"{kind}_crps_mean", f"crps_reduction_vs_{kind}"):
                    entry[skill] = scored["posterior"][key][skill]
        metrics["sensor_statistics"][name] = stats

        # Every source sampled on the posterior's time stamps.
        samples = {
            kind: [
                quantity_series(
                    s.sel(time=p.time, method="nearest", tolerance=tolerance)
                )
                .transpose("quantity", "ensemble", "time", "sensor")
                .values
                for s, p in zip(series, per_window["posterior"])
            ]
            for kind, series in per_window.items()
        }
        metrics["sensor_distributions"][name], figure = sensor_distribution_summary(
            samples, QUANTITIES
        )
        diagnostics.update({f"{key}_{name}": da for key, da in figure.items()})

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
            "rmse": window_series_stats(m["rmse"], num_windows),
            "crps": window_series_stats(m["crps"], num_windows),
        }
        for score in ("rmse", "crps"):
            if f"prior_{score}" in m:
                post = float(np.nanmean(m[score]))
                pri = float(np.nanmean(m[f"prior_{score}"]))
                entry[f"prior_{score}_mean"] = pri
                entry[f"{score}_reduction_vs_prior"] = 1 - post / pri if pri else None
        out[name] = entry
    return out


def _parameter_members(files: list[pathlib.Path], names: list[str]) -> xarray.DataArray:
    """The members of the parameters `names` per window, (window, ensemble,
    parameter); a time-varying parameter enters as its window mean."""
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
    return xarray.DataArray(
        np.stack(members),
        dims=("window", "ensemble", "parameter"),
        coords={"window": np.arange(len(files)), "parameter": names},
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


def _read_ensemble(
    ds: xarray.Dataset,
    sets: dict,
    solver_name: str,
    stl_path: pathlib.Path,
    times: np.ndarray | None = None,
) -> tuple[dict[str, xarray.DataArray], xarray.Dataset]:
    """Sensor series and field reductions of one window, member by member.

    `ds` is `(ensemble, time, ...)` on the run's global time axis. Given the
    global `times` of another source of the same window (the posterior's), only
    the frames at those times are read. Returns the series per sensor set and
    the members' `member_field_reductions`: the statistics averaged (never the
    statistics of the mean field), the profiles and spectra stacked on
    `ensemble`.
    """
    if times is not None:
        frames = np.abs(ds.time.values[None, :] - times[:, None]).argmin(axis=1)
        assert np.allclose(ds.time.values[frames], times, atol=1e-6)
        ds = ds.isel(time=frames)
    n_members = ds.sizes["ensemble"]
    series: dict[str, list] = {name: [] for name in sets}
    total, profiles, spectra = None, [], []
    for m in range(n_members):
        member = ds[["u", "v", "w"]].isel(ensemble=slice(m, m + 1)).load()
        for name, points in sets.items():
            series[name].append(sensor_series(member, points, solver_name))
        reduced = member_field_reductions(
            member.isel(ensemble=0), solver_name, stl_path
        )
        total = reduced if total is None else total + reduced
        profiles.append(reduced.profile)
        spectra.append(reduced.spectrum)
    assert total is not None
    mean = (total / n_members).assign(
        profile=xarray.concat(profiles, dim="ensemble"),
        spectrum=xarray.concat(spectra, dim="ensemble"),
    )
    mean.attrs = reduced.attrs
    return (
        {name: xarray.concat(parts, dim="ensemble") for name, parts in series.items()},
        mean,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="config overrides, e.g. assimilation.replica_dir=...",
    )
    args = parser.parse_args()
    run(args.run_dir, args.overrides)


if __name__ == "__main__":
    main()
