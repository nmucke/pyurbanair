"""Utilities for handling parameter extraction and merging for ForwardModel."""

from typing import Optional

import xarray

# Parameter names that survive extract/merge into the solver-facing Dataset.
# Beyond the inflow trio, the model-error knobs ``vertical_inflow_exponent`` (α)
# and ``sgs_constant`` must be whitelisted here too — otherwise they are silently
# dropped before reaching the solver and their ESMDA estimates never take effect
# (docs/archive/esmda_model_error_parameters.md §6.3).
INFLOW_PARAM_NAMES = (
    "inflow_angle",
    "velocity_magnitude",
    "pressure_gradient_magnitude",
    "vertical_inflow_exponent",
    "sgs_constant",
)


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

    data_vars = {}

    for param_name in INFLOW_PARAM_NAMES:
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
    for param_name in INFLOW_PARAM_NAMES:
        if param_name in new_inflow:
            merged[param_name] = new_inflow[param_name]

    return merged


def create_params_dataset(
    inflow_angle: Optional[float] = None,
    velocity_magnitude: Optional[float] = None,
    pressure_gradient_magnitude: Optional[float] = None,
    vertical_inflow_exponent: Optional[float] = None,
    sgs_constant: Optional[float] = None,
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
    if vertical_inflow_exponent is not None:
        data_vars["vertical_inflow_exponent"] = vertical_inflow_exponent
    if sgs_constant is not None:
        data_vars["sgs_constant"] = sgs_constant

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
