"""Utilities for scaling LBM state between lattice and physical units."""

import xarray

# LBM outputs velocity in lattice units. Multiply by this factor to get m/s.
VELOCITY_SCALE_TO_PHYSICAL = 75.0


def scale_velocity_to_physical(state: xarray.Dataset) -> xarray.Dataset:
    """
    Scale velocity fields (u, v, w) from lattice units to m/s.

    LBM outputs scaled/lattice-unit velocities. Multiply by VELOCITY_SCALE_TO_PHYSICAL
    to convert to physical units (m/s).
    """
    result = state.copy(deep=False)
    for var in ("u", "v", "w"):
        if var in result.data_vars:
            result[var] = result[var] * VELOCITY_SCALE_TO_PHYSICAL
    return result


def scale_velocity_to_lattice(state: xarray.Dataset) -> xarray.Dataset:
    """
    Scale velocity fields (u, v, w) from m/s to lattice units.

    Used when preparing state for LBM restart files, which expect lattice units.
    """
    result = state.copy(deep=False)
    scale = 1.0 / VELOCITY_SCALE_TO_PHYSICAL
    for var in ("u", "v", "w"):
        if var in result.data_vars:
            result[var] = result[var] * scale
    return result
