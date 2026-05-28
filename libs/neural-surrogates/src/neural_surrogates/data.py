"""PyTorch dataset over a pyurbanair `training_data/` split.

The training-data layout is documented in `docs/training_data.md`. This
module exposes `TransitionDataset`, which flattens every trajectory in a
split into individual `(state_n, params_n, geometry) -> state_{n+1}`
transition pairs so a one-step neural surrogate can be trained on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """One-step transition pairs flattened across every trajectory in a split.

    A trajectory with `T` saved time steps contributes `T - 1` pairs; the
    dataset length is the sum across all trajectories in
    `<root>/state/<split>/`.

    Each item is a `dict` of `torch.Tensor`:

    - ``state_n``    — ``(C, *grid)``  velocity channels stacked from `state_vars`.
    - ``state_next`` — ``(C, *grid)``  the next snapshot in the same trajectory.
    - ``params_n``   — ``(P,)``        parameter values at step `n` (time-varying
      params sliced at `n`; scalar params broadcast).
    - ``geometry``   — ``(*grid,)``    binary mask, `1` = fluid, `0` = obstacle
      (buildings + ground). Same tensor for every item.

    Geometry is sourced from the state file's ``geometry_var`` variable
    (``"blanking"`` by default — pylbm's per-cell obstacle indicator,
    inverted to match the requested convention). When that variable is
    absent, the mask falls back to the cells where the stacked state is
    non-zero in the first trajectory's first snapshot.

    State snapshots are read lazily from netCDF on each ``__getitem__``
    via ``xr.open_dataset(..., cache=cache).isel(time=...)`` so only two
    slices per pair leave disk. With ``cache=False`` (default) nothing
    accumulates between calls; with ``cache=True`` xarray keeps every
    already-read slice in memory, so after one epoch the full trajectories
    are resident and later epochs hit RAM instead of disk. Per-trajectory
    parameter tensors are small and are kept in memory. State file handles
    are cached per process, so the cache is rebuilt independently in each
    ``DataLoader`` worker.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        state_vars: Sequence[str] = ("u", "v", "w"),
        param_vars: Sequence[str] | None = None,
        geometry_var: str | None = "blanking",
        cache: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.root = Path(root_dir)
        self.split = split
        self.state_vars = tuple(state_vars)
        self.geometry_var = geometry_var
        self.cache = cache
        self.dtype = dtype

        state_dir = self.root / "state" / split
        param_dir = self.root / "param" / split
        if not state_dir.is_dir():
            raise FileNotFoundError(f"missing state split dir: {state_dir}")
        if not param_dir.is_dir():
            raise FileNotFoundError(f"missing param split dir: {param_dir}")

        self._state_files: list[Path] = sorted(state_dir.glob("sample_*.nc"))
        param_files: list[Path] = sorted(param_dir.glob("sample_*.nc"))
        if not self._state_files:
            raise ValueError(f"split '{split}' is empty under {self.root}")
        if len(self._state_files) != len(param_files):
            raise ValueError(
                f"sample count mismatch in split '{split}': "
                f"{len(self._state_files)} state vs {len(param_files)} param"
            )

        self._params: list[torch.Tensor] = []
        self.param_names: tuple[str, ...] = ()
        traj_lengths: list[int] = []

        for state_path, param_path in zip(self._state_files, param_files):
            t_len = self._read_time_length(state_path)
            params, names = self._load_params(param_path, t_len, param_vars)
            if not self.param_names:
                self.param_names = names
            elif names != self.param_names:
                raise ValueError(
                    f"param variable set differs between samples: "
                    f"{self.param_names} vs {names} (in {param_path})"
                )
            if t_len < 2:
                raise ValueError(
                    f"trajectory {state_path.name} has <2 time steps; "
                    f"cannot form a transition pair"
                )
            self._params.append(params)
            traj_lengths.append(t_len)

        self._index: list[tuple[int, int]] = [
            (traj, t)
            for traj, t_len in enumerate(traj_lengths)
            for t in range(t_len - 1)
        ]

        self._geometry = self._load_geometry(self._state_files[0])
        self._state_cache: dict[int, xr.Dataset] | None = None

    @staticmethod
    def _read_time_length(state_path: Path) -> int:
        with xr.open_dataset(state_path) as ds:
            return int(ds.sizes["time"])

    def _load_params(
        self,
        param_path: Path,
        t_len: int,
        param_vars: Sequence[str] | None,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        with xr.open_dataset(param_path) as ds:
            names = (
                tuple(param_vars) if param_vars is not None else tuple(ds.data_vars)
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
        return torch.from_numpy(np.stack(cols, axis=-1)).to(self.dtype), names

    def _load_geometry(self, state_path: Path) -> torch.Tensor:
        with xr.open_dataset(state_path) as ds:
            if self.geometry_var is not None and self.geometry_var in ds.data_vars:
                obstacle = np.asarray(ds[self.geometry_var].isel(time=0).values)
                return torch.from_numpy(1.0 - obstacle).to(self.dtype)
            channels = [
                np.asarray(ds[v].isel(time=0).values) for v in self.state_vars
            ]
        stacked = np.stack(channels, axis=0)
        nonzero = torch.from_numpy(stacked).ne(0).any(dim=0)
        return nonzero.to(self.dtype)

    def _get_state_ds(self, traj: int) -> xr.Dataset:
        if self._state_cache is None:
            self._state_cache = {}
        ds = self._state_cache.get(traj)
        if ds is None:
            ds = xr.open_dataset(self._state_files[traj], cache=self.cache)
            self._state_cache[traj] = ds
        return ds

    def __getstate__(self) -> dict:
        """Drop the open-file cache before pickling so each `DataLoader`
        worker rebuilds it lazily against its own file descriptors.
        """
        state = self.__dict__.copy()
        state["_state_cache"] = None
        return state

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, t = self._index[idx]
        ds = self._get_state_ds(traj)
        snap = ds.isel(time=slice(t, t + 2))
        channels = np.stack(
            [np.asarray(snap[v].values) for v in self.state_vars], axis=1
        )
        pair = torch.from_numpy(channels).to(self.dtype)
        return {
            "state_n": pair[0],
            "state_next": pair[1],
            "params_n": self._params[traj][t],
            "geometry": self._geometry,
        }
