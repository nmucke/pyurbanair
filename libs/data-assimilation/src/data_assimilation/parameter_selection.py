"""Select analysis parameters while preserving the full forecast configuration.

Parameters named in ``log_names`` are analysed in log space: ``select_parameters``
returns ``log(value)`` for them and ``merge_parameters`` maps the analysed values
back with ``exp``, so the Kalman update works on ``log(value)`` while every
forecast and saved Dataset stays in physical units (and stays positive).
"""

from typing import Optional, Sequence

import numpy as np
import xarray


def validate_parameter_names(
    names: Optional[Sequence[str]],
) -> Optional[tuple[str, ...]]:
    """None estimates all supplied fields; an empty sequence estimates none."""
    if names is None:
        return None
    if isinstance(names, str):
        raise ValueError("parameter_names_to_estimate must be a sequence of names.")
    selected = tuple(names)
    if any(not isinstance(name, str) or not name for name in selected):
        raise ValueError("parameter_names_to_estimate must contain nonempty strings.")
    if len(set(selected)) != len(selected):
        raise ValueError("parameter_names_to_estimate must not contain duplicates.")
    return selected


def select_parameters(
    params: xarray.Dataset,
    names: Optional[Sequence[str]],
    log_names: Sequence[str] = (),
) -> xarray.Dataset:
    """Keep analysis fields in Dataset order, including an empty ensemble axis.

    Selected fields named in ``log_names`` are returned as ``log(value)``.
    """
    if names is None:
        selected = params
    else:
        missing = [name for name in names if name not in params.data_vars]
        if missing:
            raise ValueError(
                f"Parameters to estimate absent from params: {sorted(missing)}"
            )
        selected = params.drop_vars(
            [name for name in params.data_vars if name not in names]
        )
        if "ensemble" in params.sizes and "ensemble" not in selected.sizes:
            selected = selected.assign_coords(ensemble=params["ensemble"])
    if not log_names:
        return selected
    missing = [name for name in log_names if name not in params.data_vars]
    if missing:
        raise ValueError(f"Log-space parameters absent from params: {sorted(missing)}")
    selected = selected.copy(deep=False)
    for name in log_names:
        if name not in selected.data_vars:
            continue
        values = selected[name]
        if values.dims != ("ensemble",):
            raise ValueError(
                f"Log-space parameter {name!r} must be static with dims "
                f"('ensemble',), got {values.dims}."
            )
        if not bool(np.all(np.asarray(values) > 0.0)):
            raise ValueError(f"Log-space parameter {name!r} must be strictly positive.")
        selected[name] = np.log(values)
    return selected


def merge_parameters(
    original: xarray.Dataset,
    updated: xarray.Dataset,
    log_names: Sequence[str] = (),
) -> xarray.Dataset:
    """Replace analyzed variables, retaining fixed fields and labelled metadata.

    Analysed fields named in ``log_names`` hold ``log(value)`` and are written
    back as ``exp`` of it, in physical units.
    """
    result = original.copy(deep=False)
    for name in updated.data_vars:
        values = updated[name]
        if name in log_names:
            values = np.exp(values)
        result[name] = values.transpose(*original[name].dims)
        result[name].attrs = dict(original[name].attrs)
    return result
