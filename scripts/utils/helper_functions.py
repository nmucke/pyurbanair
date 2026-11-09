"""Helpers shared by the scripts in scripts/.

Only what more than one script needs lives here: building the truth, the
observation pieces and the assimilation ensemble from the config, and a few
small I/O utilities. Each script keeps its own workflow logic.
"""

from __future__ import annotations

import pathlib
import shutil
from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray
import yaml
from data_assimilation.augmentation import ParamAugmentation
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    flatten_observations,
    sensor_observation_coords,
)
from data_assimilation.observation_statistics import ObservationStatistics
from data_assimilation.parameter_selection import select_parameters
from data_assimilation.smoothing.field_analysis import field_analysis_netcdf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.discrepancy import SGS_BIAS_PARAMETER_NAMES
from pyurbanair.config.hydra_helpers import clean_outputs
from pyurbanair.static_parameters import log_space_parameter_names


def save_yaml(data: dict, path: pathlib.Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def make_run_dir(cfg: DictConfig, name: str) -> tuple[pathlib.Path, pathlib.Path]:
    """`<paths.results_dir>/<name>/` and its `windows/` dir, with config.yaml saved."""
    out_dir = pathlib.Path(cfg.paths.results_dir) / name
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    return out_dir, windows_dir


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------


def make_truth(cfg: DictConfig, out_dir: pathlib.Path) -> xarray.Dataset:
    """Truth state over the whole horizon, opened lazily from disk.

    Simulated with the truth model when `assimilation.truth_dir` is null, else
    read from `<truth_dir>`. The truth parameters are saved to
    `out_dir/true_params.nc` either way.
    """
    horizon = cfg.assimilation.num_windows * cfg.time.simulation_time
    truth_dir = cfg.assimilation.truth_dir

    if truth_dir is None:
        # A time-varying sampler draws its knots over the full horizon.
        sampler_kwargs = (
            {"simulation_time": horizon}
            if "seconds_per_knot" in cfg.truth_params
            else {}
        )
        params = instantiate(cfg.truth_params, **sampler_kwargs).sample(1)
        model = instantiate(
            cfg.truth_model.forward_model, results_dir=None, simulation_time=horizon
        )
        instantiate(cfg.truth_model.prepare, forward_model=model)
        clean_outputs(cfg.truth_model.name, model)
        model(params=params.isel(ensemble=0)).to_netcdf(out_dir / "true_state.nc")
    else:
        params = xarray.load_dataset(pathlib.Path(truth_dir) / "params.nc")
        if "time" in params.dims:
            start = float(cfg.assimilation.truth_start_time or 0.0)
            params = _time_window(params, start, horizon, keep_start=True)

    params.to_netcdf(out_dir / "true_params.nc")
    return open_truth(cfg, out_dir)


def case_stl_path(cfg: DictConfig) -> pathlib.Path:
    """The case STL, `geometry.stl_path`; a relative path is from the repo root."""
    path = pathlib.Path(cfg.geometry.stl_path)
    if not path.is_absolute():
        path = pathlib.Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Case STL {path} (geometry.stl_path) not found; the state metrics "
            "need it to leave the building cells out."
        )
    return path


def open_truth(cfg: DictConfig, run_dir: pathlib.Path) -> xarray.Dataset:
    """A run's truth state over its horizon, opened lazily.

    `run_dir/true_state.nc` for a simulated truth, else the state of
    `assimilation.truth_dir` (see `open_forward_run`).
    """
    truth_dir = cfg.assimilation.truth_dir
    if truth_dir is None:
        return xarray.open_dataset(run_dir / "true_state.nc")
    return open_forward_run(cfg, truth_dir)


def open_forward_run(cfg: DictConfig, directory: str) -> xarray.Dataset:
    """`<directory>/state.nc` of a forward run over the assimilation horizon.

    Read from `assimilation.truth_start_time` on, with that time rebased to
    t=0, so it lines up with the truth: used for `truth_dir` and `replica_dir`.
    Both ends are matched within half an output interval: uDALES output times
    jitter (the last frame of a 60 s run may sit at 60.05 s).
    """
    horizon = cfg.assimilation.num_windows * cfg.time.simulation_time
    start = float(cfg.assimilation.truth_start_time or 0.0)
    path = pathlib.Path(directory) / "state.nc"
    state = xarray.open_dataset(path)
    # Output frames sit in (0, simulation_time], so a frame at t=0 marks a
    # state.nc written before the backends stamped time that way (PR #163).
    if start == 0.0 and float(state.time[0]) <= 1e-6:
        state.close()
        raise ValueError(
            f"{path} has a frame at t=0: it predates output on "
            "(0, simulation_time] and would be read one frame off. Regenerate it."
        )
    tolerance = 0.5 * float(cfg.time.output_frequency)
    return _time_window(state, start, horizon, tolerance=tolerance)


def _time_window(
    ds: xarray.Dataset,
    start: float,
    length: float,
    keep_start: bool = False,
    tolerance: float = 1e-6,
) -> xarray.Dataset:
    """Keep (start, start + length] and shift the time axis so `start` is t=0.

    That is where a run's output frames sit (the first one is one output
    interval in). `keep_start` also keeps t=start, for parameter knots. Both
    ends are matched within `tolerance`.
    """
    eps = tolerance
    after_start = ds.time >= start - eps if keep_start else ds.time > start + eps
    ds = ds.sel(time=after_start & (ds.time <= start + length + eps))
    return ds.assign_coords(time=ds.time - start)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def make_observation_operator(cfg: DictConfig, solver_name: str) -> Any:
    """The configured operator, set up for one solver's grid layout."""
    operator = cfg.observation.operator
    if "observation_operator" in operator:  # temporal wrapper around a spatial one
        return instantiate(operator, observation_operator={"solver_name": solver_name})
    return instantiate(operator, solver_name=solver_name)


def make_observation_error(cfg: DictConfig) -> ObservationErrorSpec:
    error = dict(OmegaConf.to_container(cfg.observation.error, resolve=True))  # type: ignore[arg-type, unused-ignore]
    # TODO: drop once ObservationErrorSpec.aggregation is renamed to `propagation`.
    error["aggregation"] = error.pop("propagation", "propagate_mean")
    spec: ObservationErrorSpec = instantiate(error)
    return spec


def observation_types(cfg: DictConfig) -> list[str]:
    """`observation.types`, [bins] when unset.

    bins without `aggregation.interval_seconds` bins every frame on its own,
    which is raw, and is returned as raw.
    """
    types = list(OmegaConf.select(cfg, "observation.types", default=None) or ["bins"])
    aggregation = cfg.observation.aggregation
    if aggregation is None or aggregation.get("interval_seconds") is None:
        types = ["raw" if name == "bins" else name for name in types]
    return types


def make_aggregation(
    cfg: DictConfig,
) -> AggregateObservations | ObservationStatistics | None:
    """What turns a window's observed series into the vector the update sees.

    For a single raw or bins type (the default), the interval aggregator, or
    None to assimilate every frame (raw); otherwise the `ObservationStatistics`
    of the listed types.
    """
    types = observation_types(cfg)
    aggregator = None
    if "bins" in types:
        aggregator = instantiate(cfg.observation.aggregation)
        # Windows may hold a different number of intervals (e.g. a shorter last one).
        aggregator.allow_interval_count_change = True
    if types in (["raw"], ["bins"]):
        return aggregator
    options = {
        name: OmegaConf.to_container(cfg.observation[name], resolve=True)
        for name in ("tke", "spectral")
        if name in types
    }
    statistics = ObservationStatistics(
        types=types,
        aggregation=aggregator,
        num_sensors=len(cfg.obs.x_points),
        **options,
    )
    statistics.allow_interval_count_change = True
    return statistics


def observe(
    truth: xarray.Dataset,
    operator: Any,
    error: ObservationErrorSpec,
    aggregation: AggregateObservations | ObservationStatistics | None,
    rng_key: jax.Array,
) -> tuple[xarray.DataArray, xarray.DataArray, Any]:
    """Noisy and clean observations of `truth`, plus the resolved error.

    Instrument noise is drawn on the raw frames; the resolved error carries the
    covariance of the (possibly aggregated) observations the update sees. For
    observation statistics it is block-diagonal, the tke and spectral blocks
    estimated from the noisy series.
    """
    clean = operator(truth)
    noise = np.asarray(jax.random.normal(rng_key, clean.shape))
    if not isinstance(aggregation, ObservationStatistics):
        resolved = error.resolve(clean, operator, aggregation)
        return clean + resolved.raw_instrument_std * noise, clean, resolved
    raw_std = error.resolve(clean, operator, None).raw_instrument_std
    noisy = clean + raw_std * noise
    return noisy, clean, aggregation.resolve_error(noisy, operator, error)


def window_observations(
    truth: xarray.Dataset,
    frames_per_window: int,
    num_windows: int,
    operator: Any,
    error: ObservationErrorSpec,
    aggregation: AggregateObservations | ObservationStatistics | None,
    rng_key: jax.Array,
) -> tuple[list[tuple], jax.Array]:
    """`observe` per window of the smoother, and the advanced key.

    The noise depends only on the key and the frames, so the same call gives a
    finished run's observations again.
    """
    observations = []
    for w in range(num_windows):
        rng_key, key = jax.random.split(rng_key)
        window_truth = truth.isel(
            time=slice(w * frames_per_window, (w + 1) * frames_per_window)
        )
        observations.append(observe(window_truth, operator, error, aggregation, key))
    return observations, rng_key


def cycle_observations(
    cfg: DictConfig, truth: xarray.Dataset, num_cycles: int, rng_key: jax.Array
) -> list[tuple]:
    """One noisy observation per filter cycle, in global cycle order.

    A cycle covers `assimilation.assimilate_every_n_step` truth frames and
    assimilates the last one. Returns (noisy, clean, resolved error, time) per
    cycle, `time` being the truth's time of the assimilated frame.
    """
    stride = int(cfg.assimilation.assimilate_every_n_step)
    operator = make_observation_operator(cfg, cfg.truth_model.solver_name)
    error = make_observation_error(cfg)
    out = []
    for cycle in range(num_cycles):
        rng_key, key = jax.random.split(rng_key)
        frame = (cycle + 1) * stride - 1
        cycle_truth = truth.isel(time=slice(frame, frame + 1))
        noisy, clean, resolved = observe(cycle_truth, operator, error, None, key)
        out.append((noisy, clean, resolved, float(cycle_truth.time[0])))
    return out


def cycles_to_time(ds: xarray.Dataset, times: np.ndarray) -> xarray.Dataset:
    """A cycle-stacked filter history as an (ensemble, time, ...) dataset."""
    ds = ds.drop_vars("time", errors="ignore").rename(cycle="time")
    return ds.assign_coords(time=times).transpose("ensemble", "time", ...)


def flatten_obs(
    observations: xarray.DataArray,
    aggregation: AggregateObservations | ObservationStatistics | None,
) -> np.ndarray:
    """Aggregate and flatten observations into the vector the update uses.

    Observation statistics of an observed series (`observed=True`) carry no
    instrument-noise bias term; the predictions get it added instead
    (`set_prediction_noise`). So a clean series' statistics differ from a
    noisy one's by that bias, by design.
    """
    if isinstance(aggregation, ObservationStatistics):
        observations = aggregation(observations, observed=True)
    elif aggregation is not None:
        observations = aggregation(observations)
    return np.asarray(flatten_observations(observations))


def set_prediction_noise(aggregation: Any, resolved: Any) -> None:
    """Give observation statistics the raw instrument variance per observation
    (from their resolved error), which they add to the predicted tke and
    spectra to match the noisy observed series. No-op otherwise."""
    if isinstance(aggregation, ObservationStatistics):
        aggregation.noise_variance = resolved.noise_variance


class StridedOperator:
    """An observation operator that keeps every n-th frame, ending on the last.

    A filter cycle spanning n output frames assimilates only its last one, so
    its predicted observations are thinned the same way as the truth's.
    """

    def __init__(self, operator: Any, stride: int) -> None:
        self.operator = operator
        self.stride = stride

    def __call__(self, state: xarray.Dataset) -> Any:
        return self.operator(state).isel(time=slice(self.stride - 1, None, self.stride))

    def __getattr__(self, name: str) -> Any:
        if name in ("operator", "stride"):
            raise AttributeError(name)
        return getattr(self.operator, name)


def save_obs(
    path: pathlib.Path,
    obs: np.ndarray,
    obs_clean: np.ndarray,
    obs_std: np.ndarray,
    pred_obs: list[np.ndarray],
    extra: dict[str, np.ndarray] | None = None,
    block_slices: dict[str, slice] | None = None,
) -> None:
    """One window's observations and predicted observations per update step.

    `pred_obs` holds one (observation, ensemble) array per step: the ESMDA
    iterations for a smoother (prior first, posterior last), forecast and
    analysis for a filter. `extra` adds more per-observation variables;
    `block_slices` (observation statistics) labels each observation with its
    block in `obs_block` (raw, bins, tke, spectral).
    """
    if block_slices is not None:
        labels = np.full(np.asarray(obs).size, "", dtype=object)
        for name, rows in block_slices.items():
            labels[rows] = name
        extra = {**(extra or {}), "obs_block": labels.astype(str)}
    xarray.Dataset(
        {
            "obs": ("observation", np.asarray(obs).ravel()),
            "obs_clean": ("observation", np.asarray(obs_clean).ravel()),
            "obs_std": ("observation", np.asarray(obs_std).ravel()),
            "pred_obs": (
                ("esmda_step", "observation", "ensemble"),
                np.stack([np.asarray(p) for p in pred_obs]),
            ),
            **{
                name: ("observation", np.asarray(value).ravel())
                for name, value in (extra or {}).items()
            },
        }
    ).to_netcdf(path)


# ---------------------------------------------------------------------------
# Field analysis
# ---------------------------------------------------------------------------


def field_observations(
    noisy: xarray.DataArray,
    clean: xarray.DataArray,
    truth_operator: Any,
    error: ObservationErrorSpec,
    operator: Any,
    forecast_path: pathlib.Path,
    substitutions: dict[int, int],
) -> xarray.Dataset:
    """Everything the field analysis needs of one window but the state.

    `noisy` and `clean` are the window's raw sensor frames, whatever Stage 1
    assimilated; `operator` reads the assimilation model's frames in
    `forecast_path` member by member; `substitutions` are the failed members of
    that forecast. The Dataset holds per frame and observation `obs`,
    `obs_clean`, `obs_var` (instrument plus representation variance, no
    aggregation and no sampling error) and `pred_obs` (ensemble, time, obs), per
    observation its position `obs_x/y/z` and `obs_component`, and per member
    `member_mask` (False for the slots holding a donor's field).

    `obs_var` treats the frames as independent, which they are not.
    `innovation_lag1` (obs) measures that: the lag-1 autocorrelation in time of
    the innovation, about (1 + r) / (1 - r) frames per independent sample.
    """
    base = getattr(operator, "observation_operator", operator)
    with xarray.open_dataset(forecast_path) as state:
        times = state.time.values
        member_mask = ~np.isin(np.arange(state.sizes["ensemble"]), list(substitutions))
        predicted = np.stack(
            [
                operator(state[list(base.obs_states)].isel(ensemble=m).load()).values
                for m in range(state.sizes["ensemble"])
            ]
        )
    if noisy.sizes["time"] != times.size:
        raise ValueError(
            f"The truth has {noisy.sizes['time']} frames in a window and the "
            f"ensemble {times.size}: the field analysis pairs them one to one."
        )
    variance = np.asarray(error.resolve(clean, truth_operator, None).covariance_diag)
    num_obs = noisy.sizes["obs"]
    coords = sensor_observation_coords(operator, num_obs)
    innovation = noisy.values - predicted[member_mask].mean(axis=0)
    innovation = innovation - innovation.mean(axis=0)
    lag1 = (innovation[:-1] * innovation[1:]).sum(axis=0) / (innovation**2).sum(axis=0)
    frame = ("time", "obs")
    return xarray.Dataset(
        {
            "obs": (frame, noisy.values),
            "obs_clean": (frame, clean.values),
            "obs_var": (frame, variance.reshape(times.size, num_obs)),
            "pred_obs": (("ensemble", *frame), predicted),
            "obs_x": ("obs", coords[:, 0]),
            "obs_y": ("obs", coords[:, 1]),
            "obs_z": ("obs", coords[:, 2]),
            "obs_component": (
                "obs",
                np.repeat(list(base.obs_states), base.num_sensors),
            ),
            "member_mask": ("ensemble", member_mask),
            "innovation_lag1": ("obs", lag1),
        },
        coords={"time": times},
    )


def field_parameter_rows(cfg: DictConfig, params: xarray.Dataset) -> np.ndarray:
    """(N_p, J) analysis-space rows of every parameter, one per time knot."""
    log_names = log_space_parameter_names(instantiate(cfg.prior_params))
    augmentation = ParamAugmentation(params.sizes.get("time"))
    flat = augmentation.flatten(select_parameters(params, None, log_names))
    return np.asarray(ParamAugmentation.to_array(flat))


def correct_window(
    cfg: DictConfig, windows_dir: pathlib.Path, w: int
) -> dict[str, float | int]:
    """Write window `w`'s corrected frames to its _posterior_state.nc.

    Reads the frames as the model wrote them from _posterior_forecast_state.nc,
    which is left alone, so a correction can never be applied twice, and
    window_{w}_field_obs.nc and _posterior_params.nc. Returns the
    `FieldAnalysisInfo`, plus the median of `innovation_lag1`, for run_info.yaml.
    """
    settings = cfg.field_analysis
    field_obs = xarray.load_dataset(windows_dir / f"window_{w}_field_obs.nc")
    params = xarray.load_dataset(windows_dir / f"window_{w}_posterior_params.nc")
    num_frames, num_obs = field_obs.sizes["time"], field_obs.sizes["obs"]
    coords = np.stack([field_obs[f"obs_{a}"].values for a in "xyz"], axis=1)
    # The corrected file only replaces the posterior once it is complete.
    partial = windows_dir / f"window_{w}_posterior_state.nc.partial"
    shutil.copyfile(windows_dir / f"window_{w}_posterior_forecast_state.nc", partial)
    info = field_analysis_netcdf(
        partial,
        variables=list(settings.variables),
        pred_obs=field_obs.pred_obs.values.reshape(-1, num_frames * num_obs).T,
        obs=field_obs.obs.values.ravel(),
        obs_var=field_obs.obs_var.values.ravel(),
        obs_time=np.repeat(field_obs.time.values, num_obs),
        obs_coords=np.tile(coords, (num_frames, 1)),
        params=field_parameter_rows(cfg, params),
        member_mask=field_obs.member_mask.values.astype(bool),
        L_r=settings.localization_radius,
        L_t=settings.localization_time,
        min_dof=int(settings.min_dof),
        chunk_size=int(settings.chunking),
    )
    partial.replace(windows_dir / f"window_{w}_posterior_state.nc")
    lag1 = float(np.nanmedian(field_obs.innovation_lag1.values))
    if lag1 > 0.5:
        print(
            f"Window {w}: the innovations' median lag-1 autocorrelation is "
            f"{lag1:.2f}, about {(1 + lag1) / (1 - lag1):.1f} frames per "
            "independent sample: an R of independent frames is overconfident."
        )
    return {**asdict(info), "innovation_lag1_median": lag1}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def parameter_names(
    cfg: DictConfig, params: xarray.Dataset, sampler: Any
) -> tuple[list[str] | None, list[str], dict[str, tuple[float, float] | None]]:
    """The parameters to estimate (None for all), the global ones and the
    log-space ones (name -> None or (min, max) bounds).

    SGS-discrepancy coefficients are global (never localized) when the
    assimilation model's discrepancy is enabled. Parameters with a `LogNormal`
    prior in `sampler` are analysed as log(value), bounded to (min, max) when
    the prior sets both.
    """
    selected = cfg.assimilation.params_to_estimate
    discrepancy = OmegaConf.select(
        cfg, "assim_model.forward_model.model_discrepancy.enabled", default=False
    )
    global_names = [n for n in SGS_BIAS_PARAMETER_NAMES if discrepancy and n in params]
    log_names = log_space_parameter_names(sampler)
    return (None if selected is None else list(selected)), global_names, log_names


def next_window_params(
    sampler: Any, params: xarray.Dataset, sim_time: float, rng_key: jax.Array
) -> xarray.Dataset:
    """Time-varying parameters extrapolated one window of `sim_time` ahead.

    The result sits on the sampler's own (window-local) knot times.
    """
    knot_times = np.asarray(sampler.time_coords)
    extrapolated: xarray.Dataset = sampler.extrapolate(
        params, jnp.asarray(knot_times) + sim_time, rng_key
    )
    return extrapolated.assign_coords(time=knot_times)


# ---------------------------------------------------------------------------
# Assimilation ensemble
# ---------------------------------------------------------------------------


def make_ensemble_model(
    cfg: DictConfig, results_dir: pathlib.Path | None, **forward_kwargs: Any
) -> Any:
    """The assimilation model's ensemble wrapper.

    With `results_dir` set, each member's forecast is written to its own NetCDF
    file there instead of being held in memory. `forward_kwargs` override the
    forward model's config (e.g. a filter's one-cycle `simulation_time`).
    """
    forward_model = instantiate(
        cfg.assim_model.forward_model, results_dir=None, **forward_kwargs
    )
    instantiate(cfg.assim_model.prepare, forward_model=forward_model)
    return instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=forward_model,
        results_dir=results_dir,
    )


# ---------------------------------------------------------------------------
# Reading run outputs
# ---------------------------------------------------------------------------


def global_time(time: Any, window: int, sim_time: float) -> np.ndarray:
    """Window `window`'s time stamps on the run's global time axis.

    Every window file ends on its window's end, (window + 1) * sim_time, whether
    it is stamped on the window's own clock (model output on (0, sim_time],
    parameter knots on [0, sim_time]) or already on the global one (a filter's
    cycle times).
    """
    time = np.asarray(time, dtype=float)
    return time - time[-1] + (window + 1) * sim_time


def concat_windows(paths: list[pathlib.Path], sim_time: float) -> xarray.Dataset:
    """Stack per-window parameter files into one dataset.

    Time-varying parameters go on one global time axis; static ones get one
    entry per window.
    """
    pieces = [xarray.load_dataset(p) for p in paths]
    if "time" not in pieces[0].dims:
        return xarray.concat(pieces, dim="window")
    pieces = [
        ds.assign_coords(time=global_time(ds.time, w, sim_time))
        for w, ds in enumerate(pieces)
    ]
    return xarray.concat(pieces, dim="time")


def sensor_sets(cfg: DictConfig) -> dict[str, tuple]:
    """Assimilated sensors, plus the held-out validation sensors if configured."""
    obs = cfg.obs
    sets = {"assimilation": (obs.x_points, obs.y_points, obs.z_points)}
    if "validation_x_points" in obs:
        sets["validation"] = (
            obs.validation_x_points,
            obs.validation_y_points,
            obs.validation_z_points,
        )
    return sets


def sensor_series(
    state: xarray.Dataset, points: tuple, solver_name: str
) -> xarray.DataArray:
    """(u, v, w) at the sensor points: (component, [ensemble,] time, sensor).

    Each component is interpolated on its own (staggered) grid.
    """
    x, y, z = (list(np.asarray(p, dtype=float)) for p in points)
    operator = ObservationOperator(
        obs_x=x, obs_y=y, obs_z=z, obs_states=["u", "v", "w"], solver_name=solver_name
    )
    components = []
    for var in ("u", "v", "w"):
        dims = operator.dim_mapping[var]
        components.append(
            interpolate_dataarray_at_points(
                state[var],
                x_dim=dims["x"],
                y_dim=dims["y"],
                z_dim=dims["z"],
                obs_x=operator.obs_x,
                obs_y=operator.obs_y,
                obs_z=operator.obs_z,
            )
        )
    series: xarray.DataArray = xarray.concat(components, dim="component")
    return series.assign_coords(component=["u", "v", "w"])
