"""Helpers shared by the scripts in scripts/.

Only what more than one script needs lives here: building the truth, the
observation pieces and the assimilation ensemble from the config, and a few
small I/O utilities. Each script keeps its own workflow logic.
"""

from __future__ import annotations

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import xarray
import yaml
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    flatten_observations,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.discrepancy import SGS_BIAS_PARAMETER_NAMES
from pyurbanair.config.hydra_helpers import clean_outputs


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


def make_aggregation(cfg: DictConfig) -> AggregateObservations | None:
    """The observation aggregator, or None to assimilate every frame."""
    aggregation = cfg.observation.aggregation
    if aggregation is None or aggregation.get("interval_seconds") is None:
        return None
    aggregator: AggregateObservations = instantiate(aggregation)
    # Windows may hold a different number of intervals (e.g. a shorter last one).
    aggregator.allow_interval_count_change = True
    return aggregator


def observe(
    truth: xarray.Dataset,
    operator: Any,
    error: ObservationErrorSpec,
    aggregation: AggregateObservations | None,
    rng_key: jax.Array,
) -> tuple[xarray.DataArray, xarray.DataArray, Any]:
    """Noisy and clean observations of `truth`, plus the resolved error.

    Instrument noise is drawn on the raw frames; the resolved error carries the
    covariance of the (possibly aggregated) observations the update sees.
    """
    clean = operator(truth)
    resolved = error.resolve(clean, operator, aggregation)
    noise = np.asarray(jax.random.normal(rng_key, clean.shape))
    return clean + resolved.raw_instrument_std * noise, clean, resolved


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
    observations: xarray.DataArray, aggregation: AggregateObservations | None
) -> np.ndarray:
    """Aggregate and flatten observations into the vector the update uses."""
    if aggregation is not None:
        observations = aggregation(observations)
    return np.asarray(flatten_observations(observations))


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
) -> None:
    """One window's observations and predicted observations per update step.

    `pred_obs` holds one (observation, ensemble) array per step: the ESMDA
    iterations for a smoother (prior first, posterior last), forecast and
    analysis for a filter.
    """
    xarray.Dataset(
        {
            "obs": ("observation", np.asarray(obs).ravel()),
            "obs_clean": ("observation", np.asarray(obs_clean).ravel()),
            "obs_std": ("observation", np.asarray(obs_std).ravel()),
            "pred_obs": (
                ("esmda_step", "observation", "ensemble"),
                np.stack([np.asarray(p) for p in pred_obs]),
            ),
        }
    ).to_netcdf(path)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def parameter_names(
    cfg: DictConfig, params: xarray.Dataset
) -> tuple[list[str] | None, list[str]]:
    """The parameters to estimate (None for all) and the global ones.

    SGS-discrepancy coefficients are global (never localized) when the
    assimilation model's discrepancy is enabled.
    """
    selected = cfg.assimilation.params_to_estimate
    discrepancy = OmegaConf.select(
        cfg, "assim_model.forward_model.model_discrepancy.enabled", default=False
    )
    global_names = [n for n in SGS_BIAS_PARAMETER_NAMES if discrepancy and n in params]
    return (None if selected is None else list(selected)), global_names


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


def member_files(step_dir: pathlib.Path, ensemble_size: int) -> list[pathlib.Path]:
    return [step_dir / f"state_{m}.nc" for m in range(ensemble_size)]


def concat_member_files(files: list[pathlib.Path], out_path: pathlib.Path) -> None:
    """Stack per-member state files along a new `ensemble` dim, one at a time.

    Same result as `xarray.concat(members, dim="ensemble")` without holding the
    whole ensemble in memory.
    """
    with netCDF4.Dataset(files[0]) as ref, netCDF4.Dataset(out_path, "w") as out:
        out.setncatts({k: ref.getncattr(k) for k in ref.ncattrs()})
        out.createDimension("ensemble", len(files))
        for name, dim in ref.dimensions.items():
            out.createDimension(name, len(dim))
        data_vars = [v for v in ref.variables if v not in ref.dimensions]
        for name, var in ref.variables.items():
            dims = (
                var.dimensions
                if name in ref.dimensions
                else ("ensemble", *var.dimensions)
            )
            new = out.createVariable(name, var.dtype, dims)
            new.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            if name in ref.dimensions:
                new[...] = var[...]
        for m, path in enumerate(files):
            with netCDF4.Dataset(path) as src:
                for name in data_vars:
                    out.variables[name][m, ...] = src.variables[name][...]


def last_frames(files: list[pathlib.Path]) -> xarray.Dataset:
    """Ensemble state at the final time, reading one frame per member file."""
    frames = [xarray.open_dataset(f).isel(time=-1).load() for f in files]
    return xarray.concat(frames, dim="ensemble", join="override")


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
