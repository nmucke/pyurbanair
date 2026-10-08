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
import functools
import json
import pathlib
import sys
import warnings

import numpy as np
import xarray
from evaluation.scores import (
    _skill_score,
    compute_parameter_metrics,
    data_mismatch,
    data_mismatch_summary,
    distribution_scores,
    observation_fit,
    series_stats,
    shared_histograms,
    spread_skill,
    vector_sensor_metrics,
    window_statistics_summary,
)
from evaluation.sensors import (
    QUANTITIES,
    quantity_series,
    window_sampling_std,
    window_statistics,
)
from evaluation.turbulence import (
    band_energy_ratio,
    colocate_components,
    field_rmse,
    field_statistics,
    fluid_mask,
    intrinsic_profile,
    log_spectral_distance,
    median_spectrum,
    on_grid,
    spanwise_spectra,
    stl_solid_mask,
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

# The canopy profiles: time-mean u, TKE and resolved u'w' (field_statistics keys).
PROFILE_QUANTITIES = ("u", "tke", "uw")
# The height groups of `spanwise_spectra`.
SPECTRUM_GROUPS = ("above_canopy", "in_canopy")
# The sources read on the truth's grid, as one-member ensembles.
ON_TRUTH_GRID = ("truth", "replica")


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

    # --- One pass over truth and ensembles, window by window ---------------------
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

    # Per source (truth, replica, posterior, prior, forecast), per window: the
    # sensor series of each set and the field reductions (`_read_ensemble`).
    # The truth and the replica are one-member ensembles on the truth's grid.
    # The field statistics are scored per window, so only the profiles and
    # spectra are kept across windows.
    series_by: dict[str, dict[str, list]] = {}
    fields: list[dict[str, xarray.Dataset]] = []
    reduced: dict[str, xarray.Dataset] = {}

    def collect(kind: str, series: dict, window_reduced: xarray.Dataset) -> None:
        for name in sets:
            series_by.setdefault(kind, {n: [] for n in sets})[name].append(series[name])
        reduced[kind] = window_reduced

    for w in range(num_windows):
        window = slice(w * frames_per_window, (w + 1) * frames_per_window)
        reduced.clear()
        for kind, state in zip(ON_TRUTH_GRID, (truth, replica)):
            if state is not None:
                collect(
                    kind,
                    *_read_ensemble(
                        state.isel(time=window).expand_dims("ensemble"),
                        sets,
                        cfg.truth_model.solver_name,
                        stl_path,
                    ),
                )
        times = None  # the posterior's, which the others are read on
        ensembles = {"posterior": window_files("posterior_state"), **extra_files}
        for kind, paths in ensembles.items():
            with xarray.open_dataset(paths[w]) as ds:
                ds = ds.assign_coords(time=global_time(ds.time, w, sim_time))
                series, window_reduced = _read_ensemble(
                    ds, sets, cfg.assim_model.solver_name, stl_path, times
                )
            if times is None:
                times = series[next(iter(sets))].time.values
            collect(kind, series, window_reduced)
        fields.append(_score_fields(reduced, stl_path))
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

    # --- Field statistics, canopy profiles and spectra ------------------------------
    field_metrics = _field_metrics(fields)
    diagnostics.update(field_metrics.pop("diagnostics"))
    metrics.update(field_metrics)

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
        truth_s = xarray.concat(series_by["truth"][name], dim="time").isel(ensemble=0)
        posterior_s = xarray.concat(series_by["posterior"][name], dim="time")
        extra = {
            kind: xarray.concat(series[name], dim="time")
            for kind, series in series_by.items()
            if kind not in ("truth", "posterior")
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
        metrics["sensor_distributions"][name], figure = _sensor_distributions(
            {kind: series[name] for kind, series in series_by.items()},
            tolerance,
            num_windows,
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
    a Dataset of: `stat_<q>` (z, y, x), the ensemble mean of each member's
    field statistics on the cell centres (`field_statistics`; never the
    statistics of the mean field); `profile` (ensemble, profile_quantity, z), each
    member's canopy profiles; `spectrum` (ensemble, component, group, k), each
    member's spanwise spectra on the native grids.
    """
    if times is not None:
        frames = np.abs(ds.time.values[None, :] - times[:, None]).argmin(axis=1)
        assert np.allclose(ds.time.values[frames], times, atol=1e-6)
        ds = ds.isel(time=frames)
    n_members = ds.sizes["ensemble"]
    series: dict[str, list] = {name: [] for name in sets}
    totals: dict[str, xarray.DataArray] = {}
    profiles, spectra = [], []
    for m in range(n_members):
        member = ds[["u", "v", "w"]].isel(ensemble=slice(m, m + 1)).load()
        for name, points in sets.items():
            series[name].append(sensor_series(member, points, solver_name))
        stats, profile, k, dy, spectrum = _member_reductions(
            member.isel(ensemble=0), solver_name, stl_path
        )
        totals = {q: totals[q] + s if totals else s for q, s in stats.items()}
        profiles.append(profile)
        spectra.append(spectrum)
    reduced = xarray.Dataset(
        {
            **{f"stat_{q}": total / n_members for q, total in totals.items()},
            "profile": (("ensemble", "profile_quantity", "z"), np.stack(profiles)),
            "spectrum": (("ensemble", "component", "group", "k"), np.stack(spectra)),
        },
        coords={
            "profile_quantity": list(PROFILE_QUANTITIES),
            "component": ["u", "v", "w"],
            "group": list(SPECTRUM_GROUPS),
            "k": k,
        },
        attrs={"dy": dy},
    )
    return (
        {name: xarray.concat(parts, dim="ensemble") for name, parts in series.items()},
        reduced,
    )


def _member_reductions(
    member: xarray.Dataset, solver_name: str, stl_path: pathlib.Path
) -> tuple[dict[str, xarray.DataArray], np.ndarray, np.ndarray, float, np.ndarray]:
    """One member's field statistics `(z, y, x)` on the cell centres, its canopy
    profiles `(quantity, z)` and its spanwise spectra `(component, group, k)`
    on each component's native grid (interpolation would low-pass them), with
    the wavenumbers `k` and the spanwise spacing `dy`."""
    centred = colocate_components(member, solver_name)
    coords = {d: centred[0][c].values for d, c in zip("zyx", centred[0].dims[-3:])}
    stats = {
        q: xarray.DataArray(values, dims=("z", "y", "x"), coords=coords)
        for q, values in field_statistics(*(c.values for c in centred)).items()
    }
    fluid = _fluid_cells(stl_path, list(coords.values()), dilated=True)
    profile = np.stack([intrinsic_profile(stats[q], fluid) for q in PROFILE_QUANTITIES])
    spectra = []
    for component in ("u", "v", "w"):
        field = member[component]
        native = [field[d].values for d in field.dims[-3:]]
        dy = float(native[1][1] - native[1][0])
        k, groups = spanwise_spectra(
            field.values, _fluid_cells(stl_path, native, dilated=False), dy
        )
        spectra.append(
            [
                np.full(k.size, np.nan) if groups[g] is None else groups[g]
                for g in SPECTRUM_GROUPS
            ]
        )
    return stats, profile, k, dy, np.array(spectra)


def _fluid_cells(
    stl_path: pathlib.Path, coords: list[np.ndarray], dilated: bool
) -> np.ndarray:
    """The fluid cells of a `(z, y, x)` grid, once per grid: `fluid_mask` (dilated,
    for colocated statistics) or the plain complement of the solid mask."""
    return _fluid_cells_cached(
        stl_path, *(tuple(np.asarray(c, dtype=float).tolist()) for c in coords), dilated
    )


@functools.cache
def _fluid_cells_cached(
    stl_path: pathlib.Path, z: tuple, y: tuple, x: tuple, dilated: bool
) -> np.ndarray:
    coords = (np.array(z), np.array(y), np.array(x))
    if dilated:
        return fluid_mask(stl_path, *coords)
    return ~stl_solid_mask(stl_path, *coords)


def _on_axis(da: xarray.DataArray, dim: str, values: np.ndarray) -> xarray.DataArray:
    """`da` at `values` along `dim`: as it is when it is already there, else
    linearly interpolated (NaN outside its range)."""
    if da.sizes[dim] == len(values) and np.allclose(da[dim], values):
        return da.assign_coords({dim: values})
    return da.interp({dim: values})


def _score_fields(
    reduced: dict[str, xarray.Dataset], stl_path: pathlib.Path
) -> dict[str, xarray.Dataset]:
    """One window's field reductions per source, on the posterior's grid.

    The truth's (and the replica's) statistics are interpolated onto the
    posterior's cell centres and their profiles recomputed there; every
    spectrum is put on the posterior's wavenumbers. Returns per source its
    `profile` (ensemble, profile_quantity, z) and `spectrum` (ensemble, component,
    group, k) and, but for the truth, the `rmse` (statistic) and per-level
    `level_rmse` (statistic, z) of its statistics against the truth's. The
    posterior's carries `dy` and the `building_height` range as attributes.
    """
    grid = reduced["posterior"]
    centres = [grid[d].values for d in "zyx"]
    fluid = _fluid_cells(stl_path, centres, dilated=True)
    statistics = [str(v).removeprefix("stat_") for v in grid.data_vars if "stat_" in v]
    stats = {
        kind: {q: on_grid(ds[f"stat_{q}"], grid[f"stat_{q}"]) for q in statistics}
        for kind, ds in reduced.items()
    }
    out = {}
    for kind, ds in reduced.items():
        profile = ds.profile
        if kind in ON_TRUTH_GRID:
            profile = xarray.DataArray(
                [
                    [
                        intrinsic_profile(stats[kind][q], fluid)
                        for q in PROFILE_QUANTITIES
                    ]
                ],
                dims=profile.dims,
                coords={"profile_quantity": profile.profile_quantity, "z": grid.z},
            )
        scored = xarray.Dataset(
            {"profile": profile, "spectrum": _on_axis(ds.spectrum, "k", grid.k.values)}
        )
        if kind != "truth":
            errors = [
                field_rmse(stats[kind][q], stats["truth"][q], fluid) for q in statistics
            ]
            scored["rmse"] = ("statistic", [e[0] for e in errors])
            scored["level_rmse"] = (("statistic", "z"), [e[1] for e in errors])
            scored = scored.assign_coords(statistic=statistics)
        out[kind] = scored

    solid = ~_fluid_cells(stl_path, centres, dilated=False)
    z = centres[0][:, None, None]
    tops = np.where(solid, z, -np.inf).max(axis=0)[solid.any(axis=0)]
    out["posterior"].attrs = {
        "dy": grid.attrs["dy"],
        "building_height": [tops.min(), tops.max()] if tops.size else [np.nan] * 2,
    }
    return out


def _field_metrics(fields: list[dict[str, xarray.Dataset]]) -> dict:
    """The field_statistics, canopy_profiles and spectra blocks from the
    per-window `_score_fields`, and the `profile_<source>`,
    `spectrum_<source>`, `spectrum_dy` and `building_height` arrays for
    diagnostics.nc (the truth's and replica's without an ensemble dim)."""
    num_windows = len(fields)
    by_kind = {
        kind: xarray.concat([f[kind] for f in fields], dim="window", join="override")
        for kind in fields[0]
    }
    truth = by_kind["truth"].isel(ensemble=0)
    first = fields[0]["posterior"]
    k, dy = first.k.values, float(first.attrs["dy"])
    blocks: dict = {
        "field_statistics": {"z": first.z.values.tolist()},
        "canopy_profiles": {},
        "spectra": {},
    }
    with warnings.catch_warnings():  # a level or band with nothing in it is null
        warnings.simplefilter("ignore", RuntimeWarning)
        for kind, ds in by_kind.items():
            if kind == "truth":
                continue
            for q in ds.statistic.values:
                level = np.sqrt((ds.level_rmse.sel(statistic=q) ** 2).mean("window"))
                blocks["field_statistics"].setdefault(str(q), {})[kind] = {
                    "rmse": _with_per_window(ds.rmse.sel(statistic=q), num_windows),
                    "level_rmse": [_finite(v) for v in level.values],
                }
            error = ds.profile.mean("ensemble") - truth.profile
            profile_rmse = np.sqrt((error**2).mean("z"))
            for q in PROFILE_QUANTITIES:
                blocks["canopy_profiles"].setdefault(q, {})[kind] = {
                    "profile_rmse": _with_per_window(
                        profile_rmse.sel(profile_quantity=q), num_windows
                    )
                }
            median = median_spectrum(ds.spectrum.transpose("ensemble", ...).values)
            scores = band_energy_ratio(k, median, truth.spectrum.values, dy)
            scores["log_spectral_distance"] = log_spectral_distance(
                truth.spectrum.values, median
            )
            for c, component in enumerate(ds.component.values):
                for g, group in enumerate(ds.group.values):
                    blocks["spectra"].setdefault(str(component), {}).setdefault(
                        str(group), {}
                    )[kind] = {
                        name: _with_per_window(values[:, c, g], num_windows)
                        for name, values in scores.items()
                    }
    for group in SPECTRUM_GROUPS:
        if by_kind["posterior"].spectrum.sel(group=group).isnull().all():
            print(f"No fully fluid {group} lines along y: no {group} spectra")

    diagnostics = {
        "spectrum_dy": xarray.DataArray(dy),
        "building_height": xarray.DataArray(
            fields[-1]["posterior"].attrs["building_height"], dims="bound"
        ),
    }
    for kind, ds in by_kind.items():
        if kind in ON_TRUTH_GRID:
            ds = ds.isel(ensemble=0)
        diagnostics[f"profile_{kind}"] = ds.profile
        diagnostics[f"spectrum_{kind}"] = ds.spectrum
    return {**blocks, "diagnostics": diagnostics}


def _sensor_distributions(
    series: dict[str, list[xarray.DataArray]], tolerance: float, num_windows: int
) -> tuple[dict, dict[str, xarray.DataArray]]:
    """The sensor_distributions block of one sensor set, and its figure arrays.

    `series` maps each source to its per-window `(component, ensemble, time,
    sensor)` series. Every source is sampled on the posterior's time stamps
    (frames x sensors x members). Per quantity and window, `distribution_scores`
    of each prediction against the truth, and of the truth's first half-window
    against its second (`truth_halves`, a floor that assumes stationarity).
    The figure arrays are the densities on shared bins and the quantiles of
    each source, pooled over sensors, windows and members.
    """
    samples: dict[str, list[np.ndarray]] = {}  # (quantity, ensemble, time, sensor)
    for kind, windows in series.items():
        for w, s in enumerate(windows):
            times = series["posterior"][w].time
            s = s.sel(time=times, method="nearest", tolerance=tolerance)
            samples.setdefault(kind, []).append(
                quantity_series(s)
                .transpose("quantity", "ensemble", "time", "sensor")
                .values
            )

    block: dict = {}
    edges, densities, quantiles = [], [], []
    for i, quantity in enumerate(QUANTITIES):
        truth = [t[i, 0] for t in samples["truth"]]  # (time, sensor) per window
        scored = {
            kind: [distribution_scores(t, s[w][i]) for w, t in enumerate(truth)]
            for kind, s in samples.items()
            if kind != "truth"
        }
        scored["truth_halves"] = [
            distribution_scores(t[: len(t) // 2], t[len(t) // 2 :][None]) for t in truth
        ]
        block[quantity] = {
            kind: {
                key: _with_per_window(np.array([v[key] for v in values]), num_windows)
                for key in values[0]
            }
            for kind, values in scored.items()
        }
        e, d, q = shared_histograms(
            {
                kind: np.concatenate([s[i].ravel() for s in s_w])
                for kind, s_w in samples.items()
            }
        )
        edges.append(e)
        densities.append(list(d.values()))
        quantiles.append(list(q.values()))
    coords = {"quantity": list(QUANTITIES), "source": list(samples)}
    return block, {
        "sensor_bin_edges": xarray.DataArray(
            np.array(edges),
            dims=("quantity", "edge"),
            coords={"quantity": list(QUANTITIES)},
        ),
        "sensor_density": xarray.DataArray(
            np.array(densities), dims=("quantity", "source", "bin"), coords=coords
        ),
        "sensor_quantiles": xarray.DataArray(
            np.array(quantiles), dims=("quantity", "source", "level"), coords=coords
        ),
    }


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
