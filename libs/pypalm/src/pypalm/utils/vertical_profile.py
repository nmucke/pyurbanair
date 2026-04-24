"""Pluggable vertical profile shapes for PALM inflow.

Given a reference speed ``u_ref`` and heights ``z``, a profile returns a
dimensionless shape ``s(z)`` with ``s(z_ref) = 1``. The target velocity at
height z is ``u(z) = u_ref * s(z)``.

Structurally identical to pyudales/utils/vertical_profile.py. Duplicated rather
than imported across packages so pypalm has no runtime dependency on pyudales.
"""

from typing import Any, Callable

import numpy as np


def _uniform(
    config: dict[str, Any],
    heights: np.ndarray,
    zsize: float,
) -> np.ndarray:
    return np.ones_like(heights, dtype=float)


def _power_law(
    config: dict[str, Any],
    heights: np.ndarray,
    zsize: float,
) -> np.ndarray:
    if "alpha" not in config:
        raise ValueError("power_law profile requires 'alpha'")
    alpha = float(config["alpha"])
    z_ref = float(config.get("z_ref") or zsize)
    z = np.maximum(heights, heights[0])
    return (z / z_ref) ** alpha


_BUILDERS: dict[str, Callable[[dict[str, Any], np.ndarray, float], np.ndarray]] = {
    "uniform": _uniform,
    "power_law": _power_law,
}


def build_profile_shape(
    profile_config: dict[str, Any] | None,
    heights: np.ndarray,
    zsize: float,
) -> np.ndarray:
    if profile_config is None:
        return _uniform({}, heights, zsize)

    kind = profile_config.get("type", "uniform")
    builder = _BUILDERS.get(kind)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"Unknown profile type '{kind}'; known: {known}")
    return builder(profile_config, heights, zsize)
