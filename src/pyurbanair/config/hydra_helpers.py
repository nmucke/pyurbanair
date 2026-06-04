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


def prepare_udales(
    forward_model: Any,
    python_or_matlab: str = "python",
) -> None:
    _unwrap_forward_model(forward_model).run_preprocessing(
        python_or_matlab=python_or_matlab
    )


def prepare_neural_surrogate(
    forward_model: Any,
    spinup_backend: str,
    compile: bool = True,
    python_or_matlab: str = "python",
) -> None:
    """Prepare the surrogate's spin-up backend (compile / preprocess).

    The neural surrogate itself needs no preparation, but the CFD backend it
    uses to bootstrap cold starts does. ``spinup_backend`` selects which
    preparation to run on ``forward_model.spinup_forward_model``.
    """
    surrogate = _unwrap_forward_model(forward_model)
    spinup = surrogate.spinup_forward_model
    if spinup_backend == "pyudales":
        spinup.run_preprocessing(python_or_matlab=python_or_matlab)
    elif spinup_backend in ("pylbm", "pypalm"):
        spinup.compile(compile=compile)
    else:
        raise ValueError(
            f"prepare_neural_surrogate: unknown spinup_backend {spinup_backend!r}."
        )


def clean_outputs(model_name: str, forward_model: Any) -> None:
    model = _unwrap_forward_model(forward_model)
    if model_name == "pylbm":
        clean_lbm_output_files(model.dirs)
    elif model_name == "pypalm":
        from pypalm.utils.clean_up_utils import clean_palm_output_dir

        clean_palm_output_dir(model.dirs)
    elif model_name == "pyudales":
        clean_udales_output_dir(model.dirs)
    elif model_name == "neural_surrogate":
        # The surrogate keeps no solver output of its own; its spin-up
        # backend cleans up after each call via BaseForwardModel.__call__.
        return
    else:
        # Previously the else arm fell through to uDALES cleanup; raise instead
        # so an unrecognized backend can't silently get the wrong cleanup
        # (docs/codebase_guide.md §8).
        raise ValueError(f"clean_outputs: unknown model_name {model_name!r}.")


def resolve_parameter_schema(model_name: str) -> tuple[str, ...]:
    """Resolve the ordered parameter names a model consumes.

    Keyed off ``model_name``: ``pressure_gradient_magnitude`` is uDALES-only.
    """
    base = ("inflow_angle", "velocity_magnitude")
    if model_name == "pyudales":
        return base + ("pressure_gradient_magnitude",)
    return base


def create_initial_state_ensemble(
    state: xarray.Dataset,
    ensemble_size: int,
) -> xarray.Dataset:
    member_state = state.isel(time=-1) if "time" in state.dims else state
    members = [member_state.copy(deep=True) for _ in range(ensemble_size)]
    return xarray.concat(members, dim="ensemble", join="override")


def create_observation_points(
    obs_cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        interval_seconds=obs.get("interval_seconds"),
        aggregation_mode=obs.get("aggregation_mode", "mean"),
    )


def create_C_D(num_obs: int, obs_error_std: float) -> jnp.ndarray:
    return jnp.diag((obs_error_std**2) * jnp.ones(num_obs))



def make_time_coords(simulation_time: float, num_time_points: int) -> jnp.ndarray:
    return jnp.linspace(0, simulation_time, num_time_points)


def resolve_output_dir(cfg: DictConfig, run_name: str) -> pathlib.Path:
    if HydraConfig.initialized():
        return pathlib.Path(HydraConfig.get().runtime.output_dir)
    return pathlib.Path(cfg.paths.base_results_dir) / run_name
