"""The fixed-coefficient Vreman discrepancy contract and reference algebra.

This module owns solver serialization and has no data-assimilation dependency.
Features use the native vertical datum and gradient units of inverse seconds.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from .namoptions_utils import NamoptionsFile, parse_fortran_logical

DISCREPANCY_PARAM_NAMES = ("sgs_bias_b0", "sgs_bias_b1", "sgs_bias_b2")
DISCREPANCY_NAMELIST_KEYS = (
    "lsgs_discrepancy",
    *DISCREPANCY_PARAM_NAMES,
    "sgs_discrepancy_height",
    "sgs_discrepancy_za_over_h",
    "sgs_discrepancy_zb_over_h",
    "sgs_discrepancy_epsilon",
    "sgs_discrepancy_cap",
)
# Safe even for builds whose default REAL is single precision.
MAX_LOG_MULTIPLIER_CAP = (
    min(
        float(np.log(np.finfo(np.float32).max)),
        -float(np.log(np.finfo(np.float32).tiny)),
    )
    - 1.0
)
_CONFIG_KEYS = {
    "enabled",
    "kind",
    "coefficient_model",
    "canopy_height",
    "height_band_over_H",
    "gradient_regularization",
    "log_multiplier_cap",
    "prior_std",
}


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or type(value).__name__ in {"bool", "bool_"}
        or np.ndim(value) != 0
    ):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        requirement = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}")
    return result


def validate_model_discrepancy(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate opt-in settings; disabled settings require no feature values.

    Prior scales are optional for a fixed forward run. No defaults are chosen
    for calibrated feature scales, height or the multiplier cap.
    """
    if config is None:
        return {"enabled": False}
    enabled = config.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("model_discrepancy.enabled must be a boolean")
    if not enabled:
        return {"enabled": False}
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown model_discrepancy settings: {sorted(unknown)}")
    kind = config.get("kind", "sgs_strain_rotation")
    if kind != "sgs_strain_rotation":
        raise ValueError("model_discrepancy.kind must be 'sgs_strain_rotation'")
    coefficient_model = config.get("coefficient_model", "persistent")
    if coefficient_model != "persistent":
        raise ValueError("model_discrepancy.coefficient_model must be 'persistent'")
    result: dict[str, Any] = {
        "enabled": True,
        "kind": kind,
        "coefficient_model": coefficient_model,
    }
    for key in ("canopy_height", "gradient_regularization", "log_multiplier_cap"):
        result[key] = _finite_float(config.get(key), key, positive=True)
    if result["log_multiplier_cap"] > MAX_LOG_MULTIPLIER_CAP:
        raise ValueError(
            "log_multiplier_cap exponentials must be safe in native REAL precision"
        )
    band = config.get("height_band_over_H")
    if band is None or isinstance(band, (str, bytes)):
        raise ValueError("height_band_over_H must contain two finite heights")
    try:
        if len(band) != 2:
            raise ValueError("height_band_over_H must contain two finite heights")
        za, zb = (_finite_float(value, "height_band_over_H") for value in band)
    except TypeError as exc:
        raise ValueError("height_band_over_H must contain two finite heights") from exc
    if zb <= za:
        raise ValueError("height_band_over_H requires z_b > z_a")
    with np.errstate(over="ignore", invalid="ignore"):
        heights = np.array([za, zb]) * result["canopy_height"]
        width = heights[1] - heights[0]
    if not np.all(np.isfinite(heights)) or not np.isfinite(width) or width <= 0:
        raise ValueError(
            "height_band_over_H must define a finite positive physical width"
        )
    result["height_band_over_H"] = [za, zb]
    prior = config.get("prior_std")
    if prior is not None:
        try:
            if isinstance(prior, (str, bytes)) or len(prior) != 3:
                raise ValueError("prior_std requires three positive scales")
            result["prior_std"] = [
                _finite_float(value, "prior_std", positive=True) for value in prior
            ]
        except TypeError as exc:
            raise ValueError("prior_std requires three positive scales") from exc
    return result


def extract_discrepancy_coefficients(params: xr.Dataset | None) -> dict[str, float]:
    """Extract static member coefficients, resetting absent fields to zero.

    Even a length-one time axis is rejected: ensemble selection and explicit
    static coefficients must occur before the solver-facing call.
    """
    values = dict.fromkeys(DISCREPANCY_PARAM_NAMES, 0.0)
    if params is not None:
        for name in DISCREPANCY_PARAM_NAMES:
            if name in params:
                if params[name].ndim != 0:
                    raise ValueError(
                        f"{name} must be scalar and constant during a forecast; time arrays are unsupported"
                    )
                values[name] = _finite_float(params[name].item(), name)
    return values


def discrepancy_metadata(
    config: Mapping[str, Any], coefficients: Mapping[str, float]
) -> dict[str, Any]:
    """Return JSON-safe configured settings, without inventing flow diagnostics."""
    settings = validate_model_discrepancy(config)
    if not settings["enabled"]:
        return settings
    za, zb = settings["height_band_over_H"]
    cap = settings["log_multiplier_cap"]
    return {
        **settings,
        "coefficients": {
            name: float(coefficients[name]) for name in DISCREPANCY_PARAM_NAMES
        },
        "height_band_m": [
            za * settings["canopy_height"],
            zb * settings["canopy_height"],
        ],
        "height_datum": "native zf vertical datum",
        "gradient_regularization_units": "s^-1",
        "multiplier_bounds": [float(np.exp(-cap)), float(np.exp(cap))],
    }


def write_model_discrepancy(
    namoptions_path: pathlib.Path,
    config: Mapping[str, Any] | None,
    params: xr.Dataset | None = None,
) -> dict[str, Any] | None:
    """Write every enabled coefficient after preprocessing, or remove stale keys.

    An untouched disabled namoptions is never rewritten. Active Vreman is
    determined using upstream defaults and closure branch precedence.
    """
    settings = validate_model_discrepancy(config)
    namoptions = NamoptionsFile(namoptions_path)
    section = next(
        (name for name in namoptions.sections if name.lower() == "namsubgrid"),
        "NAMSUBGRID",
    )
    keys = {name.lower(): name for name in namoptions.get_section_keys(section)}
    if not settings["enabled"]:
        changed = False
        for name in DISCREPANCY_NAMELIST_KEYS:
            if name in keys:
                changed = namoptions.remove_value(section, keys[name]) or changed
        if changed:
            namoptions.write()
        return None

    def logical(name: str, default: bool) -> bool:
        raw = namoptions.get_value(section, keys.get(name, name))
        if raw is None:
            return default
        parsed = parse_fortran_logical(raw)
        if parsed is None:
            raise ValueError(f"Invalid NAMSUBGRID logical {name}: {raw}")
        return parsed

    if (
        logical("lsmagorinsky", False)
        or not logical("lvreman", True)
        or logical("loneeqn", False)
    ):
        raise ValueError("Enabled model_discrepancy requires the active Vreman closure")
    coefficients = extract_discrepancy_coefficients(params)
    za, zb = settings["height_band_over_H"]
    native: dict[str, str | float] = {
        "lsgs_discrepancy": ".true.",
        **coefficients,
        "sgs_discrepancy_height": settings["canopy_height"],
        "sgs_discrepancy_za_over_h": za,
        "sgs_discrepancy_zb_over_h": zb,
        "sgs_discrepancy_epsilon": settings["gradient_regularization"],
        "sgs_discrepancy_cap": settings["log_multiplier_cap"],
    }
    for name, value in native.items():
        namoptions.set_value(section, keys.get(name, name), value)
    namoptions.write()
    return discrepancy_metadata(settings, coefficients)


def strain_rotation_feature(
    gradient: ArrayLike, gradient_regularization: float
) -> NDArray[np.float64]:
    """Compute q from (..., 3, 3) gradients with finite, nonnegative norms."""
    epsilon = _finite_float(
        gradient_regularization, "gradient_regularization", positive=True
    )
    tensor = np.asarray(gradient, dtype=float)
    if tensor.shape[-2:] != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("gradient must have finite (..., 3, 3) entries")
    # Scaling all terms prevents squaring overflow, and preserves q exactly in
    # real arithmetic, including extremely weak gradients relative to epsilon.
    scale = np.maximum(np.max(np.abs(tensor), axis=(-2, -1)), epsilon)
    scaled = tensor / scale[..., None, None]
    transpose = np.swapaxes(scaled, -1, -2)
    strain = 0.5 * (scaled + transpose)
    rotation = 0.5 * (scaled - transpose)
    s2 = np.sum(strain * strain, axis=(-2, -1))
    o2 = np.sum(rotation * rotation, axis=(-2, -1))
    return np.asarray((o2 - s2) / (o2 + s2 + (epsilon / scale) ** 2))


def height_feature(
    z: ArrayLike, canopy_height: float, height_band_over_H: Sequence[float]
) -> NDArray[np.float64]:
    """Evaluate the compact sin-squared band at native viscosity heights."""
    settings = validate_model_discrepancy(
        {
            "enabled": True,
            "canopy_height": canopy_height,
            "height_band_over_H": height_band_over_H,
            "gradient_regularization": 1.0,
            "log_multiplier_cap": 1.0,
        }
    )
    heights = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(heights)):
        raise ValueError("height coordinates must be finite")
    za, zb = np.asarray(settings["height_band_over_H"]) * settings["canopy_height"]
    result = np.zeros_like(heights)
    inside = (heights > za) & (heights < zb)
    result[inside] = np.sin(np.pi * ((heights[inside] - za) / (zb - za))) ** 2
    return result


def viscosity_multiplier(
    q: ArrayLike,
    phi: ArrayLike,
    coefficients: Sequence[float],
    log_multiplier_cap: float,
) -> NDArray[np.float64]:
    """Return exp(L tanh((b0 + b1 phi + b2 q)/L))."""
    if len(coefficients) != 3:
        raise ValueError("Exactly three discrepancy coefficients are required")
    b0, b1, b2 = (_finite_float(value, "coefficient") for value in coefficients)
    cap = _finite_float(log_multiplier_cap, "log_multiplier_cap", positive=True)
    if cap > MAX_LOG_MULTIPLIER_CAP:
        raise ValueError(
            "log_multiplier_cap exponentials must be safe in native REAL precision"
        )
    rotation = np.asarray(q, dtype=float)
    height = np.asarray(phi, dtype=float)
    if not np.all(np.isfinite(rotation)) or not np.all(np.abs(rotation) <= 1):
        raise ValueError("q must be finite and between -1 and 1")
    if not np.all(np.isfinite(height)) or not np.all((height >= 0) & (height <= 1)):
        raise ValueError("phi must be finite and between 0 and 1")
    # Long double avoids intermediate overflow for finite extreme coefficients.
    g = (
        np.longdouble(b0)
        + np.longdouble(b1) * height.astype(np.longdouble)
        + np.longdouble(b2) * rotation.astype(np.longdouble)
    )
    return np.asarray(np.exp(cap * np.tanh(g / cap)), dtype=float)
