"""Interpolation utilities for extracting state values at point locations."""

from __future__ import annotations

import numpy as np
import xarray

_AXIS_DIM_CANDIDATES: dict[str, tuple[str, ...]] = {
    "x": ("x", "xt", "xm"),
    "y": ("y", "yt", "ym"),
    "z": ("z", "zt", "zm"),
}


def _resolve_axis_dim_name(
    data_array: xarray.DataArray, requested_dim: str, axis_name: str
) -> str:
    """Resolve requested axis dimension name against available staggered variants."""
    if requested_dim in data_array.dims:
        return requested_dim

    candidates = _AXIS_DIM_CANDIDATES.get(axis_name, (requested_dim,))
    for candidate in candidates:
        if candidate in data_array.dims:
            return candidate

    raise ValueError(
        f"Dimension '{requested_dim}' required for interpolation is not present in "
        f"state variable '{data_array.name}'. Available dimensions are "
        f"{tuple(data_array.dims)}."
    )


def _compute_linear_indices_and_weights(
    coords: np.ndarray, points: np.ndarray, axis_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute lower/upper interpolation indices and weights for 1D coordinates."""
    if coords.ndim != 1:
        raise ValueError(f"Coordinate array for axis '{axis_name}' must be 1D.")
    if coords.size < 2:
        raise ValueError(
            f"Coordinate array for axis '{axis_name}' must have at least 2 points."
        )

    if np.any(np.diff(coords) <= 0):
        raise ValueError(
            f"Coordinate array for axis '{axis_name}' must be strictly increasing."
        )

    min_coord = float(coords[0])
    max_coord = float(coords[-1])

    # Staggered grids can place observations up to half a cell beyond a variable's
    # native coordinates (e.g., center observations against face velocities).
    spacing = np.diff(coords)
    extrapolation_margin = 0.5 * float(np.median(spacing))

    if np.any(points < (min_coord - extrapolation_margin)) or np.any(
        points > (max_coord + extrapolation_margin)
    ):
        raise ValueError(
            f"Observation points for axis '{axis_name}' are outside the grid bounds "
            f"[{min_coord}, {max_coord}] (including allowed staggered extrapolation "
            f"margin {extrapolation_margin})."
        )

    upper_idx = np.searchsorted(coords, points, side="right")
    upper_idx = np.clip(upper_idx, 1, coords.size - 1)
    lower_idx = upper_idx - 1

    lower_coord = coords[lower_idx]
    upper_coord = coords[upper_idx]
    weights = (points - lower_coord) / (upper_coord - lower_coord)

    return lower_idx, upper_idx, weights


def interpolate_dataarray_at_points(
    data_array: xarray.DataArray,
    *,
    x_dim: str,
    y_dim: str,
    z_dim: str,
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    obs_z: np.ndarray,
) -> xarray.DataArray:
    """Trilinearly interpolate a state variable at paired sensor points.

    Args:
        data_array: State variable with 3D spatial dimensions.
        x_dim: Name of the x dimension for this variable.
        y_dim: Name of the y dimension for this variable.
        z_dim: Name of the z dimension for this variable.
        obs_x: Observation x locations with shape (num_sensors,).
        obs_y: Observation y locations with shape (num_sensors,).
        obs_z: Observation z locations with shape (num_sensors,).

    Returns:
        DataArray with shape (..., sensor), where "..." are non-spatial dimensions
        from the original data_array (e.g., time).
    """
    z_dim = _resolve_axis_dim_name(data_array, z_dim, "z")
    y_dim = _resolve_axis_dim_name(data_array, y_dim, "y")
    x_dim = _resolve_axis_dim_name(data_array, x_dim, "x")

    num_sensors = obs_x.size
    if obs_y.size != num_sensors or obs_z.size != num_sensors:
        raise ValueError("obs_x, obs_y, and obs_z must have the same length.")

    ordered_dims = [dim for dim in data_array.dims if dim not in (z_dim, y_dim, x_dim)]
    ordered_dims.extend([z_dim, y_dim, x_dim])
    arr = np.asarray(data_array.transpose(*ordered_dims).values)

    z_coords = np.asarray(data_array.coords[z_dim].values, dtype=float)
    y_coords = np.asarray(data_array.coords[y_dim].values, dtype=float)
    x_coords = np.asarray(data_array.coords[x_dim].values, dtype=float)

    z0, z1, wz = _compute_linear_indices_and_weights(z_coords, obs_z, z_dim)
    y0, y1, wy = _compute_linear_indices_and_weights(y_coords, obs_y, y_dim)
    x0, x1, wx = _compute_linear_indices_and_weights(x_coords, obs_x, x_dim)

    v000 = arr[..., z0, y0, x0]
    v001 = arr[..., z0, y0, x1]
    v010 = arr[..., z0, y1, x0]
    v011 = arr[..., z0, y1, x1]
    v100 = arr[..., z1, y0, x0]
    v101 = arr[..., z1, y0, x1]
    v110 = arr[..., z1, y1, x0]
    v111 = arr[..., z1, y1, x1]

    weight_shape = (1,) * (arr.ndim - 3) + (num_sensors,)
    wz = wz.reshape(weight_shape)
    wy = wy.reshape(weight_shape)
    wx = wx.reshape(weight_shape)

    v00 = v000 * (1.0 - wx) + v001 * wx
    v01 = v010 * (1.0 - wx) + v011 * wx
    v10 = v100 * (1.0 - wx) + v101 * wx
    v11 = v110 * (1.0 - wx) + v111 * wx
    v0 = v00 * (1.0 - wy) + v01 * wy
    v1 = v10 * (1.0 - wy) + v11 * wy
    interpolated = v0 * (1.0 - wz) + v1 * wz

    output_dims = [dim for dim in ordered_dims if dim not in (z_dim, y_dim, x_dim)]
    output_dims.append("sensor")
    output_coords = {
        dim: data_array.coords[dim]
        for dim in output_dims
        if dim in data_array.coords and dim != "sensor"
    }
    output_coords["sensor"] = np.arange(num_sensors)

    return xarray.DataArray(
        interpolated,
        dims=output_dims,
        coords=output_coords,
        name=data_array.name,
    )
