"""Configuration-only support for persistent SGS discrepancy inference.

The coefficient prior belongs to the assimilation sampler. Truth parameters
are deliberately independent: an existing truth artifact need not contain the
three discrepancy fields.
"""

from __future__ import annotations

import math
import struct
from numbers import Real
from typing import Any

from omegaconf import ListConfig

SGS_BIAS_PARAMETER_NAMES = ("sgs_bias_b0", "sgs_bias_b1", "sgs_bias_b2")
SGS_BIAS_PARAMETER_METADATA = {
    name: {"scope": "global", "localization": "global"}
    for name in SGS_BIAS_PARAMETER_NAMES
}

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
