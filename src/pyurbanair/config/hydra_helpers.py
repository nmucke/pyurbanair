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


def configure_failure_policy(ensemble_model: Any, failure_cfg: Any) -> Any:
    failure = _plain(failure_cfg) or {}
    ensemble_model.configure_failure_policy(
        policy=failure.get("policy", "raise"),
        jitter_scale=failure.get("jitter_scale", 0.05),
        seed=failure.get("seed", 0),
    )
    return ensemble_model


def resolve_parameter_schema(model_name: str) -> tuple[str, ...]:
    """Resolve the ordered parameter names a model consumes.

    Keyed off ``model_name``: ``pressure_gradient_magnitude`` is uDALES-only.
    """
    base = ("inflow_angle", "velocity_magnitude")
    if model_name == "pyudales":
        return base + ("pressure_gradient_magnitude",)
    return base


def create_true_params(
    model_name: str,
    true_cfg: Any,
    param_names: Any = None,
) -> xarray.Dataset:
    true = _plain(true_cfg)
    names = param_names if param_names is not None else resolve_parameter_schema(model_name)
    data_vars = {
        "inflow_angle": true["inflow_angle"],
        "velocity_magnitude": true["velocity_magnitude"],
    }
    if "pressure_gradient_magnitude" in names:
        data_vars["pressure_gradient_magnitude"] = true[
            "pressure_gradient_magnitude"
        ]
    return xarray.Dataset(data_vars=data_vars)


def _sample_gaussian_param(
    rng_key: jax.Array,
    spec: Any,
    ensemble_size: int,
) -> tuple[jax.Array, jnp.ndarray]:
    """Sample one Gaussian parameter, applying optional min/max clamps."""
    rng_key, subkey = jax.random.split(rng_key)
    values = jax.random.normal(subkey, (ensemble_size,)) * spec["std"] + spec["mean"]
    if spec.get("min") is not None:
        values = jnp.maximum(values, spec["min"])
    if spec.get("max") is not None:
        values = jnp.minimum(values, spec["max"])
    return rng_key, values


def create_parameter_ensemble(
    model_name: str,
    prior_cfg: Any,
    ensemble_size: int,
    seed: int,
    param_names: Any = None,
) -> xarray.Dataset:
    prior = _plain(prior_cfg)
    rng_key = jax.random.PRNGKey(seed)
    names = param_names if param_names is not None else resolve_parameter_schema(model_name)

    rng_key, inflow = _sample_gaussian_param(rng_key, prior["inflow_angle"], ensemble_size)
    rng_key, velocity = _sample_gaussian_param(
        rng_key, prior["velocity_magnitude"], ensemble_size
    )
    data_vars = {
        "inflow_angle": ("ensemble", inflow),
        "velocity_magnitude": ("ensemble", velocity),
    }
    # Include pressure_gradient_magnitude only when the resolved schema requires
    # it (uDALES).
    if "pressure_gradient_magnitude" in names:
        if "pressure_gradient_magnitude" not in prior:
            raise KeyError(
                "Schema requires 'pressure_gradient_magnitude' but the prior "
                "config has no such entry."
            )
        rng_key, pgm = _sample_gaussian_param(
            rng_key, prior["pressure_gradient_magnitude"], ensemble_size
        )
        data_vars["pressure_gradient_magnitude"] = ("ensemble", pgm)

    return xarray.Dataset(
        data_vars=data_vars,
        coords={"ensemble": jnp.arange(ensemble_size)},
    )


def build_truth_ts_model(
    tv_cfg: Any,
    external_cfg: Any,
    ensemble_size: int = 1,
) -> Any:
    """Construct the parameter time-series model used to draw the truth.

    Uses ``tv_cfg.truth_method`` and ``tv_cfg.truth_method_kwargs`` — kept
    distinct from ``tv_cfg.method`` / ``tv_cfg.method_kwargs`` so the truth
    and assimilation priors never collapse to identical generative
    processes (anti-inverse-crime invariant).
    """
    from pyurbanair.parameter_time_series import build_parameter_time_series

    tv = _plain(tv_cfg)
    external_priors = _plain(external_cfg)
    return build_parameter_time_series(
        method=tv["truth_method"],
        external_priors=external_priors,
        ensemble_size=ensemble_size,
        method_kwargs=tv.get("truth_method_kwargs") or {},
    )


def create_time_varying_true_params(
    model_name: str,
    tv_cfg: Any,
    true_cfg: Any,
    external_cfg: Any,
    simulation_time: float,
    num_time_points: int,
    seed: int,
) -> xarray.Dataset:
    true = _plain(true_cfg)
    time_coords = make_time_coords(simulation_time, num_time_points)
    truth_model = build_truth_ts_model(tv_cfg, external_cfg, ensemble_size=1)
    sampled = truth_model.sample_prior(
        time_coords=time_coords,
        rng_key=jax.random.PRNGKey(seed + 1),
    )

    data_vars: dict[str, Any] = {}
    for name in sampled.data_vars:
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
        interval_seconds=obs.get("interval_seconds"),
        aggregation_mode=obs.get("aggregation_mode", "mean"),
    )


def create_C_D(num_obs: int, obs_error_std: float) -> jnp.ndarray:
    return jnp.diag((obs_error_std**2) * jnp.ones(num_obs))


def make_rng_key(seed: int) -> jax.Array:
    return jax.random.PRNGKey(seed)


def make_time_coords(simulation_time: float, num_time_points: int) -> jnp.ndarray:
    return jnp.linspace(0, simulation_time, num_time_points)


def resolve_output_dir(cfg: DictConfig, run_name: str) -> pathlib.Path:
    if HydraConfig.initialized():
        return pathlib.Path(HydraConfig.get().runtime.output_dir)
    return pathlib.Path(cfg.paths.base_results_dir) / run_name
