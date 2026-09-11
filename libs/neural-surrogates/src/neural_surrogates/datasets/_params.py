"""Shared per-trajectory parameter-table reader.

Both :class:`~neural_surrogates.datasets.transition.TransitionDataset` and the
history-conditioned :class:`~neural_surrogates.datasets.snapshot_history.SnapshotHistoryDataset`
need the same thing from a ``<root>/param/<split>/sample_XXXX.nc`` file: a
``(T, P)`` table of parameter values over the trajectory's saved times, with
scalar (static) parameters broadcast along time and 1-D parameters checked for
the trajectory's length. The reader lived as ``TransitionDataset._load_params``;
it is hoisted here so the second dataset shares one implementation (and one set
of error messages) instead of a copy that could drift. ``TransitionDataset``
keeps its method as a thin wrapper, so nothing about its behaviour changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr


def load_param_table(
    param_path: Path,
    t_len: int,
    param_vars: Sequence[str] | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Read a ``(T, P)`` parameter table from one trajectory's param file.

    ``param_vars`` fixes the column order; ``None`` takes every data variable
    in file order. Scalar variables are broadcast to ``t_len`` rows, 1-D
    variables must have exactly ``t_len`` entries, and anything else raises.
    Returns the table (as ``dtype``) and the resolved variable names.
    """
    with xr.open_dataset(param_path) as ds:
        names = (
            tuple(param_vars)
            if param_vars is not None
            else tuple(str(k) for k in ds.data_vars)
        )
        cols = []
        for name in names:
            arr = np.asarray(ds[name].values)
            if arr.ndim == 0:
                cols.append(np.full((t_len,), float(arr)))
            elif arr.ndim == 1:
                if arr.shape[0] != t_len:
                    raise ValueError(
                        f"param '{name}' in {param_path.name} has length "
                        f"{arr.shape[0]}, expected {t_len}"
                    )
                cols.append(arr.astype(np.float64))
            else:
                raise ValueError(
                    f"param '{name}' in {param_path.name} has unsupported "
                    f"shape {arr.shape}; expected scalar or 1-D over time"
                )
    return torch.from_numpy(np.stack(cols, axis=-1)).to(dtype), names
