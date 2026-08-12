from __future__ import annotations

import copy
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.observation_operator import (
    AggregateObservations,
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

    When ``spinup_source == "training_data"`` the surrogate never runs a spin-up
    (the assimilation warm-starts every window from provided states), so the CFD
    backend is never invoked and there is nothing to prepare — skip the
    preprocessing/compile entirely. This keeps a
    training-data surrogate (e.g. a pypalm-trained net assimilated with a
    pyudales spin-up template) from running an unused uDALES preprocessing pass.
    """
    surrogate = _unwrap_forward_model(forward_model)
    if getattr(surrogate, "spinup_source", None) == "training_data":
        return
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
    ``vertical_inflow_exponent`` (power-law shear exponent α) and ``sgs_constant``
    (sub-grid-scale mixing constant) are model-error compensation knobs every
    backend can consume per-member; see docs/esmda_model_error_parameters.md.
    """
    base = (
        "inflow_angle",
        "velocity_magnitude",
        "vertical_inflow_exponent",
        "sgs_constant",
    )
    if model_name == "pyudales":
        return base + ("pressure_gradient_magnitude",)
    return base


# Config blocks (in the static / dynamic params sampler configs) that hold the
# per-parameter Distribution entries. ``parameters`` is the static sampler's
# block; ``external_parameters`` / ``static_parameters`` are the time-varying
# (AR(2)) sampler's dynamic and constant-in-time blocks.
_PARAM_CONFIG_BLOCKS = ("parameters", "external_parameters", "static_parameters")


def filter_parameter_config(params_cfg: DictConfig, selected: Any) -> DictConfig:
    """Restrict a params sampler config to the parameters in ``selected``.

    Lets a run choose *which* parameters ESMDA estimates from
    ``conf/run_esmda.yaml`` (``params_to_estimate``) without editing the sampler
    configs. ``selected`` is an iterable of parameter names, or ``None`` to keep
    every parameter the config defines. Parameters dropped here are absent from
    the sampled prior/truth Dataset, so the forward models fall back to their
    construction-time/template defaults for them (see
    docs/esmda_model_error_parameters.md §4 default-absent behaviour).

    The same filter is applied to both the prior and truth samplers so excluding
    a parameter reproduces the run as if that knob did not exist on either side.
    """
    if selected is None:
        return params_cfg
    keep = set(selected)
    cfg = copy.deepcopy(params_cfg)
    # A config composed by Hydra is in struct mode, which forbids key deletion;
    # relax it on the copy (the original is untouched).
    OmegaConf.set_struct(cfg, False)
    for block in _PARAM_CONFIG_BLOCKS:
        if block in cfg and cfg[block] is not None:
            for name in list(cfg[block].keys()):
                if name not in keep:
                    del cfg[block][name]
    return cfg


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


def create_validation_points(
    obs_cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the validation sensor coordinates, or ``None`` if not configured.

    Validation sensors are a held-out set, distinct from the assimilation
    sensors (``*_points``), used only to score the posterior at locations the
    Kalman update never saw. Configured via ``validation_{x,y,z}_points`` on the
    obs config (points mode only).
    """
    obs = _plain(obs_cfg)
    if "validation_x_points" not in obs:
        return None
    return (
        np.asarray(obs["validation_x_points"]),
        np.asarray(obs["validation_y_points"]),
        np.asarray(obs["validation_z_points"]),
    )


def create_observation_operator(
    obs_cfg: Any,
    solver_name: str,
) -> ObservationOperator | TemporalObservationOperator:
    obs = _plain(obs_cfg)
    obs_x, obs_y, obs_z = create_observation_points(obs)
    operator = ObservationOperator(
        obs_x=obs_x.tolist(),
        obs_y=obs_y.tolist(),
        obs_z=obs_z.tolist(),
        obs_states=obs["states"],
        solver_name=solver_name,
    )

    # No temporal_mode (or null) -> the bare spatial operator, for configs that
    # observe instantaneous state and never look at the time dimension.
    if ("temporal_mode" not in obs) or (obs["temporal_mode"] is None):
        return operator

    if obs["temporal_mode"] != "full":
        raise ValueError(
            f"Invalid obs.temporal_mode '{obs['temporal_mode']}'. The temporal "
            "operator now always returns full time-resolved observations; "
            "temporal aggregation moved to AggregateObservations. Set "
            "obs.temporal_mode=full and configure interval_seconds (and "
            "aggregation_mode) on the run config's algorithm node "
            "(esmda/filtering/filter_smoothing) instead."
        )

    return TemporalObservationOperator(operator)


def create_aggregate_observations(cfg: Any) -> AggregateObservations | None:
    """Build the observation aggregator, or None for full-resolution assimilation.

    Reads ``interval_seconds`` / ``aggregation_mode`` off the run config's
    algorithm node (``esmda`` / ``filtering`` / ``filter_smoothing``):
    aggregation is a data-assimilation choice, not an observation-operator
    argument. An absent or null ``interval_seconds`` means the data
    assimilation assimilates the full time-resolved observation vector.
    """
    node = _plain(cfg)
    interval_seconds = node.get("interval_seconds")
    if interval_seconds is None:
        return None
    # A null aggregation_mode in the config means "use the default".
    mode = node.get("aggregation_mode") or "mean"
    return AggregateObservations(interval_seconds=float(interval_seconds), mode=mode)


def create_C_D(num_obs: int, obs_error_std: float) -> jnp.ndarray:
    return jnp.diag((obs_error_std**2) * jnp.ones(num_obs))


def make_time_coords(simulation_time: float, num_time_points: int) -> jnp.ndarray:
    return jnp.linspace(0, simulation_time, num_time_points)


def resolve_output_dir(cfg: DictConfig, run_name: str) -> pathlib.Path:
    if HydraConfig.initialized():
        return pathlib.Path(HydraConfig.get().runtime.output_dir)
    return pathlib.Path(cfg.paths.base_results_dir) / run_name
