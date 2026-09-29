"""Configuration-only support for static SGS discrepancy inference.

The coefficient prior belongs to the assimilation sampler. Truth parameters
are deliberately independent: an existing truth artifact need not contain the
three discrepancy fields.
"""

from __future__ import annotations

import copy
import math
import struct
from numbers import Real
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

SGS_BIAS_PARAMETER_NAMES = ("sgs_bias_b0", "sgs_bias_b1", "sgs_bias_b2")
SGS_BIAS_PARAMETER_METADATA = {
    name: {"scope": "global", "localization": "global"}
    for name in SGS_BIAS_PARAMETER_NAMES
}

_STATIC_SAMPLER = "pyurbanair.static_parameters.ParameterSampler"
_STATIC_SMOOTHER = "data_assimilation.smoothing.esmda.ParameterESMDA"
_IDENTITY_EVOLUTION = (
    "data_assimilation.filtering.parameter_evolution.IdentityEvolution"
)
_NORMAL = "pyurbanair.static_parameters.Normal"
_SETTING_NAMES = {
    "enabled",
    "kind",
    "coefficient_model",
    "canopy_height",
    "height_band_over_H",
    "gradient_regularization",
    "log_multiplier_cap",
    "prior_std",
}
# Mirror pyudales's conservative float32 bound, even when a native build uses
# default-real-8. The native expression rounds log(float32.tiny) to float32,
# negates it and leaves one unit of margin for exp(): -log(2^-126) - 1.
_MAX_LOG_MULTIPLIER_CAP = (
    -struct.unpack("f", struct.pack("f", math.log(2.0**-126)))[0] - 1.0
)


def _enabled(discrepancy_cfg: Any) -> bool:
    if discrepancy_cfg is None:
        return False
    return bool(discrepancy_cfg.get("enabled", False))


def _prior_scales(discrepancy_cfg: Any) -> tuple[float, float, float]:
    scales = discrepancy_cfg.get("prior_std")
    if not isinstance(scales, (list, tuple, ListConfig)) or len(scales) != 3:
        raise ValueError(
            "model_discrepancy.prior_std must contain three positive finite scales "
            "for sgs_bias_b0/b1/b2."
        )
    for scale in scales:
        if (
            isinstance(scale, bool)
            or not isinstance(scale, Real)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError(
                "model_discrepancy.prior_std must contain three positive finite "
                "numeric scales."
            )
    return tuple(float(scale) for scale in scales)  # type: ignore[return-value]


def _positive_finite(value: Any, key: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"model_discrepancy.{key} must be positive and finite.")
    return float(value)


def validate_sgs_discrepancy_settings(discrepancy_cfg: Any) -> None:
    """Validate solver feature settings without importing a backend.

    This mirrors the enabled pyudales contract at preview time. A fixed forward
    run may omit ``prior_std``; inference requires it separately.
    """
    if discrepancy_cfg is None:
        return
    enabled = discrepancy_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("model_discrepancy.enabled must be a boolean.")
    if not enabled:
        return
    unknown = set(discrepancy_cfg) - _SETTING_NAMES
    if unknown:
        raise ValueError(f"Unknown model_discrepancy settings: {sorted(unknown)}")
    if discrepancy_cfg.get("kind", "sgs_strain_rotation") != "sgs_strain_rotation":
        raise ValueError("model_discrepancy.kind must be sgs_strain_rotation.")
    if discrepancy_cfg.get("coefficient_model", "persistent") != "persistent":
        raise ValueError("model_discrepancy.coefficient_model must be persistent.")
    height = _positive_finite(discrepancy_cfg.get("canopy_height"), "canopy_height")
    _positive_finite(
        discrepancy_cfg.get("gradient_regularization"),
        "gradient_regularization",
    )
    cap = _positive_finite(
        discrepancy_cfg.get("log_multiplier_cap"), "log_multiplier_cap"
    )
    if cap > _MAX_LOG_MULTIPLIER_CAP:
        raise ValueError(
            "model_discrepancy.log_multiplier_cap exponentials must be safe "
            "in native REAL precision."
        )
    band = discrepancy_cfg.get("height_band_over_H")
    if not isinstance(band, (list, tuple, ListConfig)) or len(band) != 2:
        raise ValueError(
            "model_discrepancy.height_band_over_H must contain two finite heights."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in band
    ):
        raise ValueError(
            "model_discrepancy.height_band_over_H must contain two finite heights."
        )
    lower, upper = (float(value) for value in band)
    if upper <= lower:
        raise ValueError("model_discrepancy.height_band_over_H requires z_b > z_a.")
    lower_height, upper_height = lower * height, upper * height
    if (
        not math.isfinite(lower_height)
        or not math.isfinite(upper_height)
        or not math.isfinite(upper_height - lower_height)
        or upper_height <= lower_height
    ):
        raise ValueError(
            "model_discrepancy.height_band_over_H must define a finite "
            "positive physical width."
        )
    if discrepancy_cfg.get("prior_std") is not None:
        _prior_scales(discrepancy_cfg)


def validate_sgs_discrepancy_inference(cfg: DictConfig, workflow: str) -> None:
    """Check the first supported inference scope before a runner has side effects.

    Fixed-coefficient forward runs and truth models are outside this check.
    The initial inverse problem estimates all three global coefficients and
    holds forcing and the native SGS constant at their configured values.
    """
    discrepancy = OmegaConf.select(cfg, "assim_model.forward_model.model_discrepancy")
    if not _enabled(discrepancy):
        return
    validate_sgs_discrepancy_settings(discrepancy)
    if workflow not in {"esmda", "filtering", "filter_smoothing"}:
        raise ValueError(
            "SGS discrepancy inference currently supports only ESMDA, "
            "filtering, and filter_smoothing workflows."
        )
    if OmegaConf.select(cfg, "assim_model.name") != "pyudales":
        raise ValueError("SGS discrepancy inference requires the pyudales backend.")
    if OmegaConf.select(cfg, "assim_model.forward_model.closure") != "vreman":
        raise ValueError("SGS discrepancy inference requires the Vreman closure.")
    if OmegaConf.select(cfg, "prior_params._target_") != _STATIC_SAMPLER:
        raise ValueError("SGS discrepancy inference requires a static prior sampler.")
    parameters = OmegaConf.select(cfg, "prior_params.parameters")
    if not isinstance(parameters, DictConfig):
        raise ValueError("Static prior sampler must define a parameters mapping.")
    _check_coefficient_collisions(parameters)
    if (
        workflow in {"esmda", "filter_smoothing"}
        and OmegaConf.select(cfg, "esmda.smoother._target_") != _STATIC_SMOOTHER
    ):
        raise ValueError(
            "SGS discrepancy inference requires the static parameter-only ESMDA smoother."
        )
    if workflow == "filtering" and OmegaConf.select(cfg, "filtering.mode") not in {
        "parameter",
        "joint",
    }:
        raise ValueError(
            "SGS discrepancy filtering requires filtering.mode=parameter or joint."
        )
    if (
        workflow == "filtering"
        and OmegaConf.select(cfg, "filtering.mode") == "parameter"
        and "DistanceLocalization"
        in str(OmegaConf.select(cfg, "filtering.localization._target_", default=""))
    ):
        raise ValueError(
            "SGS discrepancy parameter-only filtering cannot use distance localization."
        )
    if (
        workflow == "filter_smoothing"
        and OmegaConf.select(cfg, "filtering.mode") != "state"
    ):
        raise ValueError(
            "SGS discrepancy hybrid requires filtering.mode=state so the ESMDA "
            "coefficients remain fixed through the filter phase."
        )
    if (
        workflow == "filter_smoothing"
        and OmegaConf.select(cfg, "ensemble.failure.policy", default="raise") != "raise"
    ):
        raise ValueError(
            "SGS discrepancy hybrid requires ensemble.failure.policy=raise "
            "until cross-phase donor handoff is supported."
        )
    if workflow == "filtering":
        evolution = OmegaConf.select(cfg, "filtering.parameter_evolution")
        if evolution is not None and evolution.get("_target_") != _IDENTITY_EVOLUTION:
            raise ValueError(
                "SGS discrepancy filtering requires identity parameter evolution "
                "(filtering/evolution=none or IdentityEvolution)."
            )
    selected = OmegaConf.select(cfg, "params_to_estimate")
    if (
        not isinstance(selected, (list, tuple, ListConfig))
        or len(selected) != len(SGS_BIAS_PARAMETER_NAMES)
        or not all(isinstance(name, str) for name in selected)
        or set(selected) != set(SGS_BIAS_PARAMETER_NAMES)
    ):
        raise ValueError(
            "SGS discrepancy inference must estimate exactly sgs_bias_b0/b1/b2; "
            "hold physical forcing and sgs_constant fixed."
        )
    _prior_scales(discrepancy)


def augment_sgs_discrepancy_prior(
    prior_params_cfg: DictConfig, discrepancy_cfg: Any
) -> DictConfig:
    """Append three independent zero-mean Normal priors to a static sampler.

    Call after the normal parameter filter and only for the assimilation prior.
    Existing entries stay in their original order, which preserves their JAX
    random-key sequence. Missing/disabled discrepancy returns the input object.
    """
    if not _enabled(discrepancy_cfg):
        return prior_params_cfg
    if prior_params_cfg.get("_target_") != _STATIC_SAMPLER:
        raise ValueError("SGS discrepancy prior requires a static ParameterSampler.")
    scales = _prior_scales(discrepancy_cfg)
    parameters = prior_params_cfg.get("parameters")
    if not isinstance(parameters, DictConfig):
        raise ValueError("Static prior sampler must define a parameters mapping.")
    _check_coefficient_collisions(parameters)
    result = copy.deepcopy(prior_params_cfg)
    original_struct = OmegaConf.is_struct(result)
    OmegaConf.set_struct(result, False)
    for name, scale in zip(SGS_BIAS_PARAMETER_NAMES, scales):
        result.parameters[name] = {"_target_": _NORMAL, "mean": 0.0, "std": scale}
    OmegaConf.set_struct(result, original_struct)
    return result


def _check_coefficient_collisions(parameters: DictConfig) -> None:
    collisions = set(SGS_BIAS_PARAMETER_NAMES).intersection(parameters)
    if collisions:
        raise ValueError(
            "SGS discrepancy prior already defines coefficient distributions: "
            + ", ".join(sorted(collisions))
        )
