"""Utilities for scaling LBM state between lattice and physical units."""

from typing import TYPE_CHECKING, Optional

import xarray

if TYPE_CHECKING:
    from .dir_utils import DirectoryPaths

# Fallback when C_u cannot be read from infile (e.g. C_u = 15 * 5 = 75 for 5 m/s inflow).
DEFAULT_C_U = 75.0
# Backward compatibility alias.
VELOCITY_SCALE_TO_PHYSICAL = DEFAULT_C_U


def _get_c_u(dirs: Optional["DirectoryPaths"]) -> float:
    """Read C_u from infile.in, or return default if unavailable."""
    if dirs is None:
        return DEFAULT_C_U
    from .infile_utils import Infile

    infile = Infile(dirs.infile_path)
    c_u = infile.get_value_as_float("C_u")
    return c_u if c_u is not None and c_u > 0 else DEFAULT_C_U


def scale_velocity_to_physical(
    state: xarray.Dataset,
    dirs: Optional["DirectoryPaths"] = None,
) -> xarray.Dataset:
    """
    Scale velocity fields (u, v, w) from lattice units to m/s.

    LBM outputs scaled/lattice-unit velocities. Multiply by C_u (from infile.in)
    to convert to physical units (m/s). C_u is set as 15 * inflow_velocity.
    """
    c_u = _get_c_u(dirs)
    result = state.copy(deep=False)
    for var in ("u", "v", "w"):
        if var in result.data_vars:
            result[var] = result[var] * c_u
    return result


def scale_velocity_to_lattice(
    state: xarray.Dataset,
    dirs: Optional["DirectoryPaths"] = None,
) -> xarray.Dataset:
    """
    Scale velocity fields (u, v, w) from m/s to lattice units.

    Used when preparing state for LBM restart files, which expect lattice units.
    """
    c_u = _get_c_u(dirs)
    scale = 1.0 / c_u
    result = state.copy(deep=False)
    for var in ("u", "v", "w"):
        if var in result.data_vars:
            result[var] = result[var] * scale
    return result
