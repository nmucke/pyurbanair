"""Utilities for handling parameter extraction and application for ForwardModel."""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray

if TYPE_CHECKING:
    from .dir_utils import DirectoryPaths

from .infile_utils import Infile

logger = logging.getLogger(__name__)


def is_time_varying_params(params: Optional[xarray.Dataset]) -> bool:
    """Check if parameters contain time-varying inflow_angle or velocity_magnitude.

    Args:
        params: Optional xarray.Dataset that may contain inflow parameters.

    Returns:
        True if any inflow parameter has a ``time`` dimension.
    """
    if params is None:
        return False
    for var_name in ("inflow_angle", "velocity_magnitude"):
        if var_name in params and "time" in params[var_name].dims:
            return True
    return False


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

    # LBM's effective y-direction response is opposite to the user-facing
    # convention used by pyudales, so negate the angle before writing udir.
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
    infile.write()

    logger.info(
        f"Updated inflow settings: angle={inflow_angle:.1f}° "
        f"(written to LBM as {inflow_angle_for_lbm:.1f}°), "
        f"velocity={velocity_magnitude:.1f} m/s"
    )


def extract_initial_params(params: xarray.Dataset) -> xarray.Dataset:
    """Extract scalar initial-time values from a time-varying params Dataset.

    Takes the first time point of each time-varying variable and returns a
    scalar Dataset suitable for ``apply_inflow_settings``.
    """
    data_vars: dict = {}
    for name in ("inflow_angle", "velocity_magnitude"):
        if name not in params:
            continue
        var = params[name]
        if "time" in var.dims:
            data_vars[name] = float(var.isel(time=0).item())
        else:
            data_vars[name] = float(var.item())
    return xarray.Dataset(data_vars=data_vars)


def write_uvel_time_file(
    params: xarray.Dataset,
    dirs: "DirectoryPaths",
    spinup_time: float = 0.0,
) -> None:
    """Write ``uvel_time.dat`` for the Fortran LBM code.

    The file format expected by ``m_inflow.F90`` is one row per data point::

        time_seconds  velocity_m_s  direction_degrees

    Times are in physical seconds, velocity in m/s, direction in degrees.
    The Fortran code handles non-dimensionalization internally.

    Args:
        params: xarray.Dataset with ``velocity_magnitude`` and ``inflow_angle``
                having a ``time`` dimension.
        dirs: DirectoryPaths with ``experiment_dir``.
        spinup_time: If > 0, prepend a constant row at t=0 with initial values
                     and shift user-provided times by this offset.
    """
    times = params["time"].values.astype(float)
    velocities = params["velocity_magnitude"].values.astype(float)

    # Broadcast scalar inflow_angle to match time dimension if needed
    if "time" in params["inflow_angle"].dims:
        angles = params["inflow_angle"].values.astype(float)
    else:
        angles = np.full_like(times, float(params["inflow_angle"].item()))

    # Negate angle for LBM convention (same as apply_inflow_settings)
    angles_for_lbm = -angles

    if spinup_time > 0:
        # Prepend constant plateau at initial values for spinup period
        times = np.concatenate([[0.0], times + spinup_time])
        velocities = np.concatenate([[velocities[0]], velocities])
        angles_for_lbm = np.concatenate([[angles_for_lbm[0]], angles_for_lbm])

    file_path = dirs.experiment_dir / "uvel_time.dat"
    with open(file_path, "w") as f:
        for t, vel, ang in zip(times, velocities, angles_for_lbm):
            f.write(f"{t:.6f}  {vel:.6f}  {ang:.6f}\n")

    logger.info("Wrote %d time-varying inflow entries to %s", len(times), file_path)


def remove_uvel_time_file(dirs: "DirectoryPaths") -> None:
    """Remove ``uvel_time.dat`` if it exists, preventing stale data."""
    file_path = dirs.experiment_dir / "uvel_time.dat"
    if file_path.exists():
        file_path.unlink()
        logger.info("Removed stale %s", file_path)
