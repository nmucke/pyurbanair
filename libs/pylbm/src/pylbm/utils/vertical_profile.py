"""Pluggable vertical profile shapes for LBM inflow.

Given a reference speed ``u_ref`` and a height array ``z``, a profile returns a
dimensionless shape ``s(z)`` such that the target velocity is ``u(z) = u_ref *
s(z)``. ``s(z_ref) = 1`` by convention, so ``u_ref`` is the speed at the
reference height (typically the domain top).

This mirrors ``pyudales.utils.vertical_profile`` so that the two backends share
the same inflow-profile convention (``power_law`` with the same ``alpha`` /
``z_ref`` produces the same shape).  Kept as a self-contained copy to avoid a
cross-backend import, matching the repo's per-backend utils layout.

Add new profiles by registering a builder in ``_BUILDERS``.
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
    """Return a (nz,) shape array ``s(z)`` for the requested profile.

    Args:
        profile_config: ``{"type": <name>, ...}`` or None (defaults to uniform).
        heights: Cell-center heights in meters, shape (nz,).
        zsize: Domain top in meters. Used as default ``z_ref``.

    Returns:
        Dimensionless shape array of length ``nz``. Multiply by a reference
        speed to obtain the velocity profile.

    Raises:
        ValueError: If ``profile_config["type"]`` is not registered or required
            fields are missing.
    """
    if profile_config is None:
        return _uniform({}, heights, zsize)

    kind = profile_config.get("type", "uniform")
    builder = _BUILDERS.get(kind)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"Unknown profile type '{kind}'; known: {known}")
    return builder(profile_config, heights, zsize)
