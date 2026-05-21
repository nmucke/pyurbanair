"""xarray.Dataset <-> tensor, K-frame history, and the time-axis contract (§4).

These helpers own the translation between the solver-shaped ``xarray.Dataset``
and the channels-first ``[T, C, Z, Y, X]`` tensors the architectures consume,
and the **time-axis trimming** that must match the Fortran backends exactly
(``docs/neural_surrogate_plan.md`` §4; mirrors pylbm ``forward_model.py``
lines 401-408).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import xarray

from ..data.grid import GridMeta

# Center-grid (collocated) dim names some solvers emit after staggered→common
# interpolation (e.g. uDALES via ``pyudales.utils.grid_utils.interpolate_grid``
# returns ``zt/yt/xt``). Renamed to the surrogate's collocated ``z/y/x`` (D3).
_COLLOCATED_DIM_ALIASES = {"zt": "z", "yt": "y", "xt": "x"}


def to_collocated_dims(ds: xarray.Dataset) -> xarray.Dataset:
    """Rename known center-grid dims/coords to the collocated ``z/y/x`` (D3).

    A no-op if the dataset already uses ``z/y/x``. Lets ``state_to_tensor``
    accept uDALES collocated output (``zt/yt/xt``) without a separate step.
    """
    rename = {
        old: new
        for old, new in _COLLOCATED_DIM_ALIASES.items()
        if old in ds.dims and new not in ds.dims
    }
    return ds.rename(rename) if rename else ds


def state_to_tensor(
    ds: xarray.Dataset,
    grid: GridMeta,
    var_names: Sequence[str],
) -> np.ndarray:
    """Convert a state Dataset to a ``[T, C, Z, Y, X]`` float array.

    Accepts both a time-indexed history (``time`` length >= 1) and a single
    time-less warm-start frame (``docs/neural_surrogate_plan.md`` §4): the
    latter is normalized to ``T = 1``. Variables are stacked channels-first in
    ``var_names`` order; spatial axes are transposed to ``(z, y, x)``. Solver
    center-grid dim names (``zt/yt/xt``) are renamed to ``z/y/x`` first (D3).
    """
    ds = to_collocated_dims(ds)
    has_time = "time" in ds.dims
    channels: list[np.ndarray] = []
    for name in var_names:
        if name not in ds:
            raise KeyError(
                f"State is missing variable {name!r}; has {list(ds.data_vars)}."
            )
        da = ds[name]
        if has_time:
            da = da.transpose("time", "z", "y", "x")
        else:
            da = da.transpose("z", "y", "x")
        arr = np.asarray(da.values, dtype=np.float32)
        if not has_time:
            arr = arr[None, ...]  # add T=1
        channels.append(arr)
    # stack along channel axis -> [T, C, Z, Y, X]
    tensor = np.stack(channels, axis=1)
    expected = (grid.nz, grid.ny, grid.nx)
    if tensor.shape[2:] != expected:
        raise ValueError(
            f"State grid {tensor.shape[2:]} does not match GridMeta {expected}."
        )
    return tensor


def tensor_to_state(
    arr: np.ndarray,
    grid: GridMeta,
    var_names: Sequence[str],
    time_coords: Optional[Sequence[float]] = None,
) -> xarray.Dataset:
    """Convert a ``[T, C, Z, Y, X]`` array back to a state Dataset.

    The output carries ``time`` + collocated ``z, y, x`` coords (D3) and one
    data variable per ``var_names`` entry, matching the source solver's axes so
    the observation operator and plotting are unchanged.
    """
    arr = np.asarray(arr)
    if arr.ndim != 5:
        raise ValueError(f"Expected [T, C, Z, Y, X], got shape {arr.shape}.")
    n_t, n_c = arr.shape[0], arr.shape[1]
    if n_c != len(var_names):
        raise ValueError(f"Channel count {n_c} != len(var_names) {len(var_names)}.")
    if time_coords is None:
        time_coords = list(range(n_t))
    data_vars = {
        name: (("time", "z", "y", "x"), arr[:, c]) for c, name in enumerate(var_names)
    }
    return xarray.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray(time_coords),
            "z": grid.z,
            "y": grid.y,
            "x": grid.x,
        },
    )


def trim_to_window(ds: xarray.Dataset, num_outputs: int) -> xarray.Dataset:
    """Trim a time-indexed state to the **last** ``num_outputs`` frames (§4).

    Matches the Fortran backends, which drop the spin-up-output prefix and keep
    ``simulation_time / output_frequency`` outputs
    (``docs/codebase_guide.md`` §7; pylbm ``forward_model.py`` lines 404-408),
    then reset the ``time`` coord to ``0..num_outputs-1``. The
    ``TemporalObservationOperator`` aggregates in fixed ``interval_size``
    chunks, so the surrogate's output ``time`` length and spacing must equal
    the source backend's *after trimming*.
    """
    if "time" not in ds.dims:
        return ds
    if ds.sizes["time"] > num_outputs:
        ds = ds.isel(time=slice(-num_outputs, None))
    return ds.assign_coords(time=range(ds.sizes["time"]))


def extract_history(tensor: np.ndarray, history_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Take the last ``history_len`` frames as a left-padded K-frame buffer (D4).

    Args:
        tensor: ``[T, C, Z, Y, X]`` frames in time order.
        history_len: ``K``.

    Returns:
        ``(hist_fields, hist_mask)``: ``hist_fields`` is ``[K, C, Z, Y, X]``
        with the most recent frame last; if ``T < K`` the leading slots are
        zero-padded and ``hist_mask`` is ``0`` there, ``1`` for real frames.
    """
    if tensor.ndim != 5:
        raise ValueError(f"Expected [T, C, Z, Y, X], got shape {tensor.shape}.")
    n_t = tensor.shape[0]
    take = min(history_len, n_t)
    real = tensor[-take:]
    if take < history_len:
        pad = np.zeros((history_len - take,) + tensor.shape[1:], dtype=tensor.dtype)
        hist_fields = np.concatenate([pad, real], axis=0)
        mask = np.concatenate(
            [np.zeros(history_len - take, dtype=np.float32), np.ones(take, dtype=np.float32)]
        )
    else:
        hist_fields = real
        mask = np.ones(history_len, dtype=np.float32)
    return hist_fields.astype(np.float32), mask


def state_to_history(
    ds: xarray.Dataset,
    grid: GridMeta,
    var_names: Sequence[str],
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``state_to_tensor`` then ``extract_history`` — Dataset to K-frame buffer."""
    tensor = state_to_tensor(ds, grid, var_names)
    return extract_history(tensor, history_len)
