from __future__ import annotations

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.observation_operator import (
    ObservationOperator,
    TemporalObservationOperator,
)
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pylbm.utils.warm_start_utils import (
    clean_all_restart_files as clean_lbm_restart_files,
)
from pylbm.utils.warm_start_utils import clean_output_files as clean_lbm_output_files
from pyudales.utils.clean_up_utils import clean_output_dir as clean_udales_output_dir


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _unwrap_forward_model(forward_model: Any) -> Any:
    return (
        forward_model.forward_model
        if hasattr(forward_model, "forward_model")
        else forward_model
    )


def prepare_compile(forward_model: Any, compile: bool) -> None:
    _unwrap_forward_model(forward_model).compile(compile=compile)


def prepare_lbm(forward_model: Any, compile: bool) -> None:
    prepare_compile(forward_model, compile)


def prepare_palm(forward_model: Any, compile: bool) -> None:
    prepare_compile(forward_model, compile)


def prepare_udales(
    forward_model: Any,
    python_or_matlab: str = "python",
) -> None:
    _unwrap_forward_model(forward_model).run_preprocessing(
        python_or_matlab=python_or_matlab
    )


def clean_outputs(model_name: str, forward_model: Any) -> None:
    model = _unwrap_forward_model(forward_model)
    if model_name == "pylbm":
        clean_lbm_output_files(model.dirs)
    elif model_name == "pypalm":
        from pypalm.utils.clean_up_utils import clean_palm_output_dir

        clean_palm_output_dir(model.dirs)
    else:
        clean_udales_output_dir(model.dirs)


def clean_restarts(model_name: str, forward_model: Any) -> None:
    if model_name == "pylbm":
        clean_lbm_restart_files(_unwrap_forward_model(forward_model).dirs)


def configure_failure_policy(ensemble_model: Any, failure_cfg: Any) -> Any:
    failure = _plain(failure_cfg) or {}
    ensemble_model.configure_failure_policy(
        policy=failure.get("policy", "raise"),
        jitter_scale=failure.get("jitter_scale", 0.05),
        seed=failure.get("seed", 0),
    )
    return ensemble_model


def create_true_params(model_name: str, true_cfg: Any) -> xarray.Dataset:
    true = _plain(true_cfg)
    data_vars = {
        "inflow_angle": true["inflow_angle"],
        "velocity_magnitude": true["velocity_magnitude"],
    }
    if model_name == "pyudales":
        data_vars["pressure_gradient_magnitude"] = true[
            "pressure_gradient_magnitude"
        ]
    return xarray.Dataset(data_vars=data_vars)


def create_parameter_ensemble(
    model_name: str,
    prior_cfg: Any,
    ensemble_size: int,
    seed: int,
) -> xarray.Dataset:
    prior = _plain(prior_cfg)
    rng_key = jax.random.PRNGKey(seed)

    rng_key, subkey = jax.random.split(rng_key)
    inflow_spec = prior["inflow_angle"]
    inflow = (
        jax.random.normal(subkey, (ensemble_size,)) * inflow_spec["std"]
        + inflow_spec["mean"]
    )

    rng_key, subkey = jax.random.split(rng_key)
    velocity_spec = prior["velocity_magnitude"]
    velocity = (
        jax.random.normal(subkey, (ensemble_size,)) * velocity_spec["std"]
        + velocity_spec["mean"]
    )
    if velocity_spec.get("min") is not None:
        velocity = jnp.maximum(velocity, velocity_spec["min"])
    if velocity_spec.get("max") is not None:
        velocity = jnp.minimum(velocity, velocity_spec["max"])

    return xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", inflow),
            "velocity_magnitude": ("ensemble", velocity),
        },
        coords={"ensemble": jnp.arange(ensemble_size)},
    )


def _prior_cfg_as_external_priors(prior_cfg: Any) -> dict[str, dict[str, float]]:
    prior = _plain(prior_cfg)
    names = ("inflow_angle", "velocity_magnitude")
    return {name: dict(prior[name]) for name in names}


def create_time_varying_true_params(
    model_name: str,
    tv_cfg: Any,
    true_cfg: Any,
    prior_cfg: Any,
    simulation_time: float,
    num_time_points: int,
    seed: int,
) -> xarray.Dataset:
    from pyurbanair.parameter_time_series import build_parameter_time_series

    tv = _plain(tv_cfg)
    true = _plain(true_cfg)
    time_coords = make_time_coords(simulation_time, num_time_points)
    truth_model = build_parameter_time_series(
        method=tv["truth_method"],
        external_priors=_prior_cfg_as_external_priors(prior_cfg),
        ensemble_size=1,
        method_kwargs=tv.get("truth_method_kwargs") or {},
    )
    sampled = truth_model.sample_prior(
        time_coords=time_coords,
        rng_key=jax.random.PRNGKey(seed + 1),
    )

    data_vars: dict[str, Any] = {}
    for name in ("inflow_angle", "velocity_magnitude"):
        data_vars[name] = ("time", np.asarray(sampled[name].isel(ensemble=0)))
    if model_name == "pyudales":
        data_vars["pressure_gradient_magnitude"] = true[
            "pressure_gradient_magnitude"
        ]
    return xarray.Dataset(
        data_vars=data_vars,
        coords={"time": np.asarray(time_coords)},
    )


def create_initial_state_ensemble(
    state: xarray.Dataset,
    ensemble_size: int,
) -> xarray.Dataset:
    member_state = state.isel(time=-1) if "time" in state.dims else state
    members = [member_state.copy(deep=True) for _ in range(ensemble_size)]
    return xarray.concat(members, dim="ensemble", join="override")


def create_observation_points(obs_cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = _plain(obs_cfg)
    mode = obs.get("mode")
    if mode == "points":
        return (
            np.asarray(obs["x_points"]),
            np.asarray(obs["y_points"]),
            np.asarray(obs["z_points"]),
        )
    if mode == "grid":
        obs_x_ax = np.linspace(obs["x_min"], obs["x_max"], obs["n_per_axis"])
        obs_y_ax = np.linspace(obs["y_min"], obs["y_max"], obs["n_per_axis"])
        obs_xx, obs_yy = np.meshgrid(obs_x_ax, obs_y_ax)
        obs_x = obs_xx.flatten()
        obs_y = obs_yy.flatten()
        obs_z = np.full(obs_x.shape[0], obs["z"])
        return obs_x, obs_y, obs_z
    raise ValueError(f"Unknown observation mode: {mode!r}")


def create_observation_operator(
    obs_cfg: Any,
    solver_name: str,
) -> TemporalObservationOperator:
    obs = _plain(obs_cfg)
    obs_x, obs_y, obs_z = create_observation_points(obs)
    operator = ObservationOperator(
        obs_x=obs_x.tolist(),
        obs_y=obs_y.tolist(),
        obs_z=obs_z.tolist(),
        obs_states=obs["states"],
        solver_name=solver_name,
    )
    return TemporalObservationOperator(
        operator,
        mode=obs["temporal_mode"],
        interval_size=obs.get("interval_size"),
        aggregation_mode=obs.get("aggregation_mode", "mean"),
    )


def create_C_D(num_obs: int, obs_error_std: float) -> jnp.ndarray:
    return jnp.diag((obs_error_std**2) * jnp.ones(num_obs))


def load_init_conditions_for_esmda(
    model_name: str,
    init_conditions_dir: str | pathlib.Path,
    ensemble_size: int,
    true_sim_id: int,
    init_subdir: str,
) -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset] | None:
    init_dir = pathlib.Path(init_conditions_dir) / init_subdir

    if not init_dir.exists():
        return None
    params_path = init_dir / "params.nc"
    if not params_path.exists():
        return None

    params_all = xarray.open_dataset(params_path).load()
    n_available = int(params_all.sizes.get("ensemble", 0))
    if n_available < ensemble_size or true_sim_id >= n_available:
        return None

    true_params = params_all.isel(ensemble=true_sim_id)
    init_params = params_all.isel(ensemble=slice(0, ensemble_size))

    state_path = init_dir / f"state_{true_sim_id}.nc"
    if not state_path.exists():
        return None
    true_init_state = xarray.open_dataset(state_path).load()
    if "time" in true_init_state.dims:
        true_init_state = true_init_state.isel(time=-1)

    init_states_list: list[xarray.Dataset] = []
    for i in range(ensemble_size):
        member_state_path = init_dir / f"state_{i}.nc"
        if not member_state_path.exists():
            return None
        state = xarray.open_dataset(member_state_path).load()
        if "time" in state.dims:
            state = state.isel(time=-1)
        init_states_list.append(state)
    init_states = xarray.concat(init_states_list, dim="ensemble", join="override")

    return init_states, init_params, true_params, true_init_state


def make_rng_key(seed: int) -> jax.Array:
    return jax.random.PRNGKey(seed)


def make_time_coords(simulation_time: float, num_time_points: int) -> jnp.ndarray:
    return jnp.linspace(0, simulation_time, num_time_points)


def resolve_output_dir(cfg: DictConfig, run_name: str) -> pathlib.Path:
    if HydraConfig.initialized():
        return pathlib.Path(HydraConfig.get().runtime.output_dir)
    return pathlib.Path(cfg.paths.base_results_dir) / run_name
