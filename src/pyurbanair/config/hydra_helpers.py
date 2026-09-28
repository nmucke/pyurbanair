from __future__ import annotations

import copy
import json
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.observation_error import ObservationErrorSpec
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
    The same holds for ``"generative"``: the cold start is sampled from the
    latent generator and the CFD backend is not even built, so there is nothing
    to prepare.
    """
    surrogate = _unwrap_forward_model(forward_model)
    if getattr(surrogate, "spinup_source", None) in ("training_data", "generative"):
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
            "(esmda/filtering) instead."
        )

    return TemporalObservationOperator(operator)


def create_aggregate_observations(cfg: Any) -> AggregateObservations | None:
    """Build the observation aggregator, or None for full-resolution assimilation.

    Reads ``interval_seconds`` / ``aggregation_mode`` off the run config's
    algorithm node (``esmda`` / ``filtering``):
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


def create_observation_error(
    cfg: Any, obs_cfg: Any, legacy_error_keys: tuple[str, ...]
) -> ObservationErrorSpec | None:
    """Resolve the opt-in observation likelihood, retaining null as legacy.

    The old scalar keys remain in the composed configuration for reproducible
    legacy runs. In corrected mode their untouched defaults have no effect,
    while an explicit Hydra override of either contract is ambiguous and fails.
    """
    error = _plain(cfg.get("observation_error"))
    if error is None:
        return None
    if not isinstance(error, dict):
        raise ValueError("observation_error must be a mapping or null")
    allowed = {
        "instrument_std",
        "representation_std",
        "representation_time_model",
        "aggregation",
    }
    unknown = set(error) - allowed
    if unknown:
        raise ValueError(f"Unknown observation_error keys: {sorted(unknown)}")
    if "instrument_std" not in error:
        raise ValueError("observation_error.instrument_std is required")
    if HydraConfig.initialized():
        overrides = HydraConfig.get().overrides.task
        overridden_old = [
            key
            for key in legacy_error_keys
            if any(override.lstrip("+").startswith(f"{key}=") for override in overrides)
        ]
        if overridden_old:
            raise ValueError(
                "Explicit legacy observation-error override conflicts with "
                "observation_error: " + ", ".join(overridden_old)
            )
    if obs_cfg.get("temporal_mode") != "full":
        raise ValueError(
            "Corrected observation_error requires obs.temporal_mode=full "
            "to retain physical frame labels."
        )
    if error.get("representation_time_model", "independent") != "independent":
        raise ValueError("Only independent representation_time_model is supported")
    if error.get("aggregation", "propagate_mean") != "propagate_mean":
        raise ValueError("Only aggregation='propagate_mean' is supported")
    return ObservationErrorSpec(
        instrument_std=error["instrument_std"],
        representation_std=error.get("representation_std", 0.0),
        representation_time_model=error.get("representation_time_model", "independent"),
        aggregation=error.get("aggregation", "propagate_mean"),
    )


def add_observation_error_metadata(
    dataset: xarray.Dataset, resolved: Any, dimension: str = "observation"
) -> None:
    """Record physical error components and the exact raw-frame bin weights."""
    products = resolved if isinstance(resolved, list) else [resolved]
    dataset["obs_instrument_variance"] = (
        dimension,
        np.concatenate([p.instrument_variance.reshape(-1) for p in products]),
    )
    dataset["obs_representation_variance"] = (
        dimension,
        np.concatenate([p.representation_variance.reshape(-1) for p in products]),
    )
    dataset["obs_time"] = (
        dimension,
        np.concatenate([np.repeat(p.times, p.variance.shape[1]) for p in products]),
    )
    dataset["obs_bin_count"] = (
        dimension,
        np.concatenate(
            [
                np.repeat([len(ids) for ids in p.frame_ids], p.variance.shape[1])
                for p in products
            ]
        ),
    )
    dataset.attrs["observation_error_model"] = products[0].provenance
    dataset.attrs["raw_frame_times_json"] = json.dumps(
        [p.raw_times.tolist() for p in products]
    )
    dataset.attrs["aggregation_frame_ids_json"] = json.dumps(
        [p.frame_ids for p in products]
    )
    dataset.attrs["aggregation_weights_json"] = json.dumps(
        [p.weights for p in products]
    )


def add_prior_innovation_diagnostics(
    dataset: xarray.Dataset,
    observations: Any,
    predicted_observations: Any,
    physical_variance: Any,
    dimension: str = "obs_index",
    block_size: int | None = None,
) -> None:
    """Record signed innovations and NIS using physical R plus prior spread."""
    pred = np.asarray(predicted_observations, dtype=float)
    obs = np.asarray(observations, dtype=float).ravel()
    variance = np.asarray(physical_variance, dtype=float).ravel()
    innovation = obs - pred.mean(axis=1)
    if block_size is None:
        block_size = obs.size
    if obs.size % block_size:
        raise ValueError("Observation count is not divisible by NIS block size")
    nis = 0.0
    for start in range(0, obs.size, block_size):
        stop = start + block_size
        pred_block = pred[start:stop]
        forecast_covariance = (
            np.cov(pred_block, rowvar=True) if pred.shape[1] > 1 else 0.0
        )
        innovation_covariance = np.atleast_2d(forecast_covariance) + np.diag(
            variance[start:stop]
        )
        residual = innovation[start:stop]
        nis += float(residual @ np.linalg.solve(innovation_covariance, residual))
    dataset["obs_innovation_prior"] = (dimension, innovation)
    dataset["obs_squared_residual_over_R_prior"] = (
        dimension,
        innovation**2 / variance,
    )
    dataset.attrs["physical_nis_prior"] = nis
    dataset.attrs["physical_nis_prior_per_observation"] = nis / obs.size
    dataset.attrs["innovation_bias_prior"] = float(np.mean(innovation))
    predictive_std = np.sqrt(
        np.var(pred, axis=1, ddof=1 if pred.shape[1] > 1 else 0) + variance
    )
    standardized = np.abs(innovation) / predictive_std
    dataset.attrs["predictive_coverage_1sigma_prior"] = float(
        np.mean(standardized <= 1)
    )
    dataset.attrs["predictive_coverage_2sigma_prior"] = float(
        np.mean(standardized <= 2)
    )
    if "obs_sensor" in dataset.coords and "obs_state" in dataset.coords:
        sensors = np.asarray(dataset.coords["obs_sensor"].values)
        states = np.asarray(dataset.coords["obs_state"].values)
        channels = [(s, c) for c in np.unique(states) for s in np.unique(sensors)]
        series = [innovation[(sensors == s) & (states == c)] for s, c in channels]
        autocorr = []
        for values in series:
            if values.size > 2 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
                autocorr.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]))
        dataset.attrs["innovation_lag1_autocorrelation_prior"] = (
            float(np.mean(autocorr)) if autocorr else np.nan
        )
        paired = [
            (a, b)
            for i, a in enumerate(series)
            for b in series[i + 1 :]
            if a.size == b.size and a.size > 1 and np.std(a) > 0 and np.std(b) > 0
        ]
        dataset.attrs["innovation_cross_channel_correlation_prior"] = (
            float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in paired]))
            if paired
            else np.nan
        )


def make_time_coords(simulation_time: float, num_time_points: int) -> jnp.ndarray:
    return jnp.linspace(0, simulation_time, num_time_points)


def resolve_output_dir(cfg: DictConfig, run_name: str) -> pathlib.Path:
    if HydraConfig.initialized():
        return pathlib.Path(HydraConfig.get().runtime.output_dir)
    return pathlib.Path(cfg.paths.base_results_dir) / run_name
