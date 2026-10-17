"""Select analysis parameters while preserving the full forecast configuration.

Parameters named in ``log_names`` are analysed in log space: ``select_parameters``
returns ``log(value)`` for them and ``merge_parameters`` maps the analysed values
back with ``exp``, so the Kalman update works on ``log(value)`` while every
forecast and saved Dataset stays in physical units (and stays positive).

``log_names`` may instead map each name to ``None`` (plain log) or to bounds
``(min, max)``: such a parameter is analysed as the logit of its log-scaled
position, ``logit((log(value) - log(min)) / (log(max) - log(min)))``, so every
value written back lies strictly inside ``(min, max)``.
"""

from typing import Mapping, Optional, Sequence, Union

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


#: Names analysed in log space, or a mapping name -> None (log) | (min, max).
LogParameters = Union[Sequence[str], Mapping[str, Optional[Sequence[float]]]]

#: Margin on the log-scaled position in (0, 1) that keeps a bounded value
#: strictly inside (min, max): logit is infinite at the bounds.
BOUND_MARGIN = 1e-6


def validate_log_parameters(
    log_names: Optional[LogParameters],
) -> dict[str, Optional[tuple[float, float]]]:
    """Normalise ``log_names`` to ``{name: None | (min, max)}``."""
    if log_names is None:
        return {}
    if not isinstance(log_names, Mapping):
        return dict.fromkeys(validate_parameter_names(log_names) or (), None)
    validate_parameter_names(list(log_names))
    spec: dict[str, Optional[tuple[float, float]]] = {}
    for name, bounds in log_names.items():
        if bounds is None:
            spec[name] = None
            continue
        low, high = (float(bound) for bound in bounds)
        if not 0.0 < low < high:
            raise ValueError(
                f"Bounds of log-space parameter {name!r} need 0 < min < max, "
                f"got ({low}, {high})."
            )
        spec[name] = (low, high)
    return spec


def _to_bounded_space(
    values: xarray.DataArray, low: float, high: float
) -> xarray.DataArray:
    """``logit`` of the log-scaled position, clamped a hair inside the bounds."""
    span = np.log(high) - np.log(low)
    position = (np.log(values.astype(float)) - np.log(low)) / span
    position = position.clip(BOUND_MARGIN, 1.0 - BOUND_MARGIN)
    return (np.log(position) - np.log1p(-position)).astype(values.dtype)


def _from_bounded_space(
    values: xarray.DataArray, low: float, high: float
) -> xarray.DataArray:
    """Inverse of :func:`_to_bounded_space`: a value strictly inside the bounds."""
    span = np.log(high) - np.log(low)
    position = 0.5 * (1.0 + np.tanh(0.5 * values.astype(float)))
    position = position.clip(BOUND_MARGIN, 1.0 - BOUND_MARGIN)
    return np.exp(np.log(low) + position * span).astype(values.dtype)


def select_parameters(
    params: xarray.Dataset,
    names: Optional[Sequence[str]],
    log_names: Optional[LogParameters] = (),
) -> xarray.Dataset:
    """Keep analysis fields in Dataset order, including an empty ensemble axis.

    Selected fields named in ``log_names`` are returned as ``log(value)``, or
    in the bounded space when it gives them bounds; values at or beyond a bound
    are clamped just inside it.
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
    log_spec = validate_log_parameters(log_names)
    if not log_spec:
        return selected
    missing = [name for name in log_spec if name not in params.data_vars]
    if missing:
        raise ValueError(f"Log-space parameters absent from params: {sorted(missing)}")
    selected = selected.copy(deep=False)
    for name, bounds in log_spec.items():
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
        selected[name] = (
            np.log(values) if bounds is None else _to_bounded_space(values, *bounds)
        )
    return selected


def merge_parameters(
    original: xarray.Dataset,
    updated: xarray.Dataset,
    log_names: Optional[LogParameters] = (),
) -> xarray.Dataset:
    """Replace analyzed variables, retaining fixed fields and labelled metadata.

    Analysed fields named in ``log_names`` hold ``log(value)`` (or the bounded
    value) and are written back in physical units.
    """
    log_spec = validate_log_parameters(log_names)
    result = original.copy(deep=False)
    for name in updated.data_vars:
        values = updated[name]
        if name in log_spec:
            bounds = log_spec[name]
            values = (
                np.exp(values)
                if bounds is None
                else _from_bounded_space(values, *bounds)
            )
        result[name] = values.transpose(*original[name].dims)
        result[name].attrs = dict(original[name].attrs)
    return result
