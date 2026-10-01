"""Select analysis parameters while preserving the full forecast configuration."""

from typing import Optional, Sequence

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
    params: xarray.Dataset, names: Optional[Sequence[str]]
) -> xarray.Dataset:
    """Keep analysis fields in Dataset order, including an empty ensemble axis."""
    if names is None:
        return params
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
    return selected


def merge_parameters(
    original: xarray.Dataset, updated: xarray.Dataset
) -> xarray.Dataset:
    """Replace analyzed variables, retaining fixed fields and labelled metadata."""
    result = original.copy(deep=False)
    for name in updated.data_vars:
        result[name] = updated[name].transpose(*original[name].dims)
        result[name].attrs = dict(original[name].attrs)
    return result
