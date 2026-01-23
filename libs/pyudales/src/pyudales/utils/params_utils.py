"""Utilities for handling parameter extraction and merging for ForwardModel."""

import logging
import pathlib
from typing import Optional

import xarray

from .dir_utils import DirectoryPaths
from .file_update_utils import update_lscale_file, update_prof_file
from .inflow_utils import angle_to_pressure_gradient, angle_to_velocity
from .namoptions_utils import NamoptionsFile

logger = logging.getLogger(__name__)


def extract_inflow_params(params: Optional[xarray.Dataset]) -> Optional[xarray.Dataset]:
    """
    Extract only the inflow parameters from an xarray.Dataset.

    Returns a new Dataset containing only the inflow parameters that are present.
    If params is None or contains no inflow parameters, returns None.

    Args:
        params: Optional xarray.Dataset that may contain inflow parameters.

    Returns:
        xarray.Dataset containing only inflow_angle, velocity_magnitude,
        and/or pressure_gradient_magnitude if present, or None if none are present.
    """
    if params is None:
        return None

    inflow_param_names = [
        "inflow_angle",
        "velocity_magnitude",
        "pressure_gradient_magnitude",
    ]
    data_vars = {}

    for param_name in inflow_param_names:
        if param_name in params:
            data_vars[param_name] = params[param_name]

    if not data_vars:
        return None

    return xarray.Dataset(data_vars=data_vars)


def merge_params(
    existing_params: Optional[xarray.Dataset],
    new_params: Optional[xarray.Dataset],
) -> Optional[xarray.Dataset]:
    """
    Merge new parameters with existing parameters.

    Creates a new xarray.Dataset that combines existing and new parameters.
    New parameters override existing ones if present. Only merges inflow parameters.

    Args:
        existing_params: Existing parameters Dataset (can be None).
        new_params: New parameters Dataset to merge in (can be None).

    Returns:
        New xarray.Dataset with merged inflow parameters, or None if both are None.
    """
    # Extract inflow params from both
    existing_inflow = extract_inflow_params(existing_params)
    new_inflow = extract_inflow_params(new_params)

    if existing_inflow is None and new_inflow is None:
        return None

    if existing_inflow is None:
        return new_inflow

    if new_inflow is None:
        return existing_inflow

    # Merge: new params override existing ones
    merged = existing_inflow.copy(deep=True)
    for param_name in [
        "inflow_angle",
        "velocity_magnitude",
        "pressure_gradient_magnitude",
    ]:
        if param_name in new_inflow:
            merged[param_name] = new_inflow[param_name]

    return merged


def create_params_dataset(
    inflow_angle: Optional[float] = None,
    velocity_magnitude: Optional[float] = None,
    pressure_gradient_magnitude: Optional[float] = None,
) -> Optional[xarray.Dataset]:
    """
    Create an xarray.Dataset from individual parameter values.

    Only includes parameters that are not None.

    Args:
        inflow_angle: Optional inflow angle in degrees.
        velocity_magnitude: Optional velocity magnitude in m/s.
        pressure_gradient_magnitude: Optional pressure gradient magnitude in Pa/m.

    Returns:
        xarray.Dataset with provided parameters, or None if all are None.
    """
    data_vars = {}
    if inflow_angle is not None:
        data_vars["inflow_angle"] = inflow_angle
    if velocity_magnitude is not None:
        data_vars["velocity_magnitude"] = velocity_magnitude
    if pressure_gradient_magnitude is not None:
        data_vars["pressure_gradient_magnitude"] = pressure_gradient_magnitude

    if not data_vars:
        return None

    return xarray.Dataset(data_vars=data_vars)


def get_param_value(
    params: Optional[xarray.Dataset],
    param_name: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Get a single parameter value from a Dataset with optional default.

    Args:
        params: Optional xarray.Dataset containing parameters.
        param_name: Name of the parameter to extract.
        default: Default value to return if parameter is not present.

    Returns:
        Parameter value as float, or default if not present, or None.
    """
    if params is None or param_name not in params:
        return default

    return params[param_name].item()  # type: ignore[no-any-return]


def apply_inflow_settings(
    params: xarray.Dataset,
    dirs: DirectoryPaths,
) -> None:
    """
    Apply the inflow settings to namoptions file and update affected input files.

    Args:
        params: xarray.Dataset containing inflow parameters. Only applies settings
               if inflow_angle and at least one magnitude (velocity_magnitude or
               pressure_gradient_magnitude) are provided.
        dirs: DirectoryPaths instance containing experiment_dir and experiment_name.

    Returns:
        Updated params Dataset if settings were applied, None otherwise.
    """
    # Extract parameter values from params Dataset
    inflow_angle = get_param_value(params, "inflow_angle")
    velocity_magnitude = get_param_value(params, "velocity_magnitude")
    pressure_gradient_magnitude = get_param_value(params, "pressure_gradient_magnitude")

    # Only apply if we have angle and at least one magnitude
    if inflow_angle is None:
        logger.warning("inflow_angle not provided, skipping inflow settings update")
        return None

    if velocity_magnitude is None and pressure_gradient_magnitude is None:
        logger.warning(
            "Neither velocity_magnitude nor pressure_gradient_magnitude provided, "
            "skipping inflow settings update"
        )
        return None

    # Calculate velocity and pressure gradient components from angle and magnitudes
    # Use defaults of 0.0 if magnitudes are not provided (though we check above)
    u0, v0 = angle_to_velocity(
        inflow_angle, velocity_magnitude if velocity_magnitude is not None else 0.0
    )
    dpdx, dpdy = angle_to_pressure_gradient(
        inflow_angle,
        pressure_gradient_magnitude if pressure_gradient_magnitude is not None else 0.0,
    )

    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"

    # Update namoptions file using NamoptionsFile
    namoptions = NamoptionsFile(namoptions_path)
    namoptions.set_value("INPS", "u0", f"{u0:.7f}")
    namoptions.set_value("INPS", "v0", f"{v0:.7f}")
    namoptions.set_value("INPS", "dpdx", f"{dpdx:.7f}")
    namoptions.set_value("INPS", "dpdy", f"{dpdy:.7f}")
    namoptions.write()

    # Update the affected input files
    prof_path = dirs.experiment_dir / f"prof.inp.{dirs.experiment_name}"
    update_prof_file(prof_path, u0=u0, v0=v0)

    lscale_path = dirs.experiment_dir / f"lscale.inp.{dirs.experiment_name}"
    update_lscale_file(lscale_path, u0=u0, v0=v0, dpdx=dpdx, dpdy=dpdy)

    logger.info(
        f"Updated inflow settings: angle={inflow_angle}°, "
        f"u0={u0:.7f}, v0={v0:.7f}, dpdx={dpdx:.7f}, dpdy={dpdy:.7f}"
    )
