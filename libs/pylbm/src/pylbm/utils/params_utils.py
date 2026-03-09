"""Utilities for handling parameter extraction and application for ForwardModel."""

import logging
from typing import TYPE_CHECKING, Optional

import xarray

if TYPE_CHECKING:
    from .dir_utils import DirectoryPaths

from .infile_utils import Infile

logger = logging.getLogger(__name__)


def get_param_value(params: xarray.Dataset, param_name: str) -> Optional[float]:
    """
    Extract a parameter value from an xarray.Dataset.

    Args:
        params: xarray.Dataset containing parameters.
        param_name: Name of the parameter to extract.

    Returns:
        The parameter value as a float, or None if not found.
    """
    if param_name not in params:
        return None

    return params[param_name].item()  # type: ignore[no-any-return]


def apply_inflow_settings(
    params: xarray.Dataset,
    dirs: "DirectoryPaths",
) -> None:
    """
    Apply the inflow settings to infile.in.

    Args:
        params: xarray.Dataset containing inflow parameters. Must contain
               inflow_angle and velocity_magnitude.
        dirs: DirectoryPaths instance containing infile_path.

    Raises:
        ValueError: If required parameters are missing or uini key is not found.
    """
    # Extract parameter values from params Dataset
    inflow_angle = get_param_value(params, "inflow_angle")
    velocity_magnitude = get_param_value(params, "velocity_magnitude")

    # Check if required parameters are present
    if inflow_angle is None:
        raise ValueError("inflow_angle is required but not found in params")

    if velocity_magnitude is None:
        raise ValueError("velocity_magnitude is required but not found in params")

    # pylbm uses opposite sign convention for inflow angle compared to pyudales.
    # Negate the angle so that the same params produce matching flow direction
    # (pyudales is the ground truth).
    # inflow_angle_for_lbm = -inflow_angle
    inflow_angle_for_lbm = inflow_angle

    # Update infile.in using Infile class
    infile = Infile(dirs.infile_path)

    # Key is first word after "!" on the uini, udir line (e.g. "uini,")
    uini_key = next((k for k in infile.get_keys() if k.startswith("uini")), None)
    if uini_key is None:
        raise ValueError(
            f"Could not find 'uini, udir' in infile.in at {dirs.infile_path}"
        )

    infile.set_value(uini_key, f"{velocity_magnitude:.1f} {inflow_angle_for_lbm:.1f}")

    # C_u (wind velocity conversion) scales with inflow velocity: C_u = 15 * uini
    c_u = 15.0 * velocity_magnitude
    infile.set_value("C_u", c_u)
    infile.write()

    logger.info(
        f"Updated inflow settings: angle={inflow_angle_for_lbm:.1f}°, "
        f"velocity={velocity_magnitude:.1f} m/s, C_u={c_u:.1f}"
    )
