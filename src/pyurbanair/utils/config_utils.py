import pathlib
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.observation_operator import (
    ObservationOperator,
    TemporalObservationOperator,
)
from pylbm.ensemble_forward_model import EnsembleForwardModel as LBMEnsembleForwardModel
from pylbm.forward_model import ForwardModel as LBMForwardModel
from pylbm.rollout_forward_model import RolloutForwardModel as LBMRolloutForwardModel
from pylbm.utils.warm_start_utils import clean_output_files as clean_lbm_output_files
from pyudales.ensemble_forward_model import (
    EnsembleForwardModel as UDALESEnsembleForwardModel,
)
from pyudales.forward_model import ForwardModel as UDALESForwardModel
from pyudales.rollout_forward_model import (
    RolloutForwardModel as UDALESRolloutForwardModel,
)
from pyudales.utils.clean_up_utils import clean_output_dir as clean_udales_output_dir

ModelName = Literal["pylbm", "pyudales"]


def _cfg() -> Any:
    # Lazily import config to avoid import cycles.
    from scripts import config

    return config


def solver_name(model_name: ModelName) -> str:
    return "pylbm" if model_name == "pylbm" else "udales"


def model_args(model_name: ModelName) -> dict:
    cfg = _cfg()
    if model_name == "pylbm":
        return {
            **cfg.LBM_ARGS,
            **cfg.DOMAIN,
            "simulation_time": cfg.TIME["simulation_time"],
            "output_frequency": cfg.TIME["output_frequency"],
        }
    return {
        **cfg.UDALES_ARGS,
        **cfg.DOMAIN,
        "simulation_time": cfg.TIME["simulation_time"],
        "output_frequency": cfg.TIME["output_frequency"],
    }


def create_forward_model(
    model_name: ModelName,
    rollout: bool = False,
    results_dir: pathlib.Path | None = None,
) -> Any:
    args = model_args(model_name)
    if results_dir is not None:
        args["results_dir"] = results_dir

    if model_name == "pylbm":
        cls = LBMRolloutForwardModel if rollout else LBMForwardModel
        return cls(**args)

    cls = UDALESRolloutForwardModel if rollout else UDALESForwardModel
    return cls(**args)


def prepare_forward_model(model_name: ModelName, forward_model: Any) -> None:
    if model_name == "pylbm":
        forward_model.compile()
    else:
        forward_model.run_preprocessing(python_or_matlab="python")


def clean_forward_model_outputs(model_name: ModelName, forward_model: Any) -> None:
    if model_name == "pylbm":
        clean_lbm_output_files(forward_model.dirs)
    else:
        clean_udales_output_dir(forward_model.dirs)


def create_ensemble_forward_model(model_name: ModelName, forward_model: Any) -> Any:
    cfg = _cfg()
    if model_name == "pylbm":
        return LBMEnsembleForwardModel(
            forward_model=forward_model,
            ensemble_size=cfg.ESMDA["ensemble_size"],
            num_parallel_processes=cfg.ESMDA["num_parallel_processes"],
            num_cpus_per_process=cfg.ESMDA["num_cpus_per_process"],
        )
    return UDALESEnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=cfg.ESMDA["ensemble_size"],
        num_parallel_processes=cfg.ESMDA["num_parallel_processes"],
        num_cpus_per_process=cfg.ESMDA["num_cpus_per_process"],
    )


def create_true_params(model_name: ModelName) -> xarray.Dataset:
    cfg = _cfg()
    data_vars = {
        "inflow_angle": cfg.TRUE_PARAMS["inflow_angle"],
        "velocity_magnitude": cfg.TRUE_PARAMS["velocity_magnitude"],
    }
    if model_name == "pyudales":
        data_vars["pressure_gradient_magnitude"] = cfg.TRUE_PARAMS[
            "pressure_gradient_magnitude"
        ]
    return xarray.Dataset(data_vars=data_vars)


def create_parameter_ensemble(model_name: ModelName) -> xarray.Dataset:
    cfg = _cfg()
    n = int(cfg.ESMDA["ensemble_size"])
    rng_key = jax.random.PRNGKey(cfg.ESMDA["seed"])

    rng_key, subkey = jax.random.split(rng_key)
    inflow = (
        jax.random.normal(subkey, (n,)) * cfg.PARAM_PRIORS["inflow_angle_std"]
        + cfg.PARAM_PRIORS["inflow_angle_mean"]
    )

    rng_key, subkey = jax.random.split(rng_key)
    vel = (
        jax.random.normal(subkey, (n,)) * cfg.PARAM_PRIORS["velocity_std"]
        + cfg.PARAM_PRIORS["velocity_mean"]
    )
    vel = jnp.maximum(vel, 0.1)

    data_vars = {
        "inflow_angle": ("ensemble", inflow),
        "velocity_magnitude": ("ensemble", vel),
    }

    if model_name == "pyudales":
        rng_key, subkey = jax.random.split(rng_key)
        pressure = (
            jax.random.normal(subkey, (n,)) * cfg.PARAM_PRIORS["pressure_std"]
            + cfg.PARAM_PRIORS["pressure_mean"]
        )
        data_vars["pressure_gradient_magnitude"] = ("ensemble", pressure)

    return xarray.Dataset(data_vars=data_vars, coords={"ensemble": jnp.arange(n)})


def create_initial_state_ensemble(state: xarray.Dataset) -> xarray.Dataset:
    cfg = _cfg()
    n = int(cfg.ESMDA["ensemble_size"])
    member_state = state.isel(time=-1) if "time" in state.dims else state
    members = [member_state.copy(deep=True) for _ in range(n)]
    return xarray.concat(members, dim="ensemble", join="override")


def create_observation_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = _cfg()
    obs_x = np.linspace(cfg.OBS["x_min"], cfg.OBS["x_max"], cfg.OBS["n_per_axis"])
    obs_y = np.linspace(cfg.OBS["y_min"], cfg.OBS["y_max"], cfg.OBS["n_per_axis"])
    obs_xx, obs_yy = np.meshgrid(obs_x, obs_y)
    obs_x_flat = obs_xx.flatten()
    obs_y_flat = obs_yy.flatten()
    obs_z_flat = np.full(obs_x_flat.shape[0], cfg.OBS["z"])
    return obs_x_flat, obs_y_flat, obs_z_flat


def create_observation_operator(model_name: ModelName) -> TemporalObservationOperator:
    cfg = _cfg()
    obs_x, obs_y, obs_z = create_observation_points()
    operator = ObservationOperator(
        obs_x=obs_x.tolist(),
        obs_y=obs_y.tolist(),
        obs_z=obs_z.tolist(),
        obs_states=cfg.OBS["states"],
        solver_name=solver_name(model_name),
    )
    return TemporalObservationOperator(operator, mode=cfg.OBS["temporal_mode"])


def create_C_D(num_obs: int) -> jnp.ndarray:
    cfg = _cfg()
    return jnp.diag((cfg.ESMDA["obs_error_std"] ** 2) * jnp.ones(num_obs))
