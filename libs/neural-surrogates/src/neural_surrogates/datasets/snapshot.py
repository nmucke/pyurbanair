"""Single-snapshot dataset for Tadpole-style autoencoder pre-training (plan 02).

:class:`SnapshotDataset` is the representation-learning sibling of
:class:`~neural_surrogates.datasets.transition.TransitionDataset`: it reads the
same on-disk ``training_data/`` split but yields **single time slices** (every
``(trajectory, t)`` pair is a sample) with no next-step target and no
parameters -- the autoencoder learns to reconstruct a snapshot, not to step it.

It reuses ``TransitionDataset``'s file discovery, lazy ``xr.open_dataset``
pattern, geometry-mask loading and content-deduped SDF-feature computation, so
the shared ``get_normalization_stats`` path (``training/data_utils.py``) works
unchanged -- ``param_names`` is empty and ``_params`` holds zero-width tensors,
so the param branch of ``_compute_normalization_stats`` degenerates cleanly to
empty stats.

Each item is a ``dict`` of ``torch.Tensor``:

- ``state``    -- ``(C, *grid)`` state channels stacked from ``state_vars`` at
  ``(traj, t)``.
- ``geometry`` -- ``(*grid,)`` binary fluid mask (``1`` = fluid). Trajectories
  with equal masks share one tensor object (deduped at init), so a
  single-geometry split ships the same tensor for every item -- which lets
  :func:`snapshot_collate` ship it once per batch.
- ``geom_features`` -- ``(C, *grid)`` SDF / ∇SDF channels, present only when
  built with a non-``"none"`` ``sdf_features`` mode.

With ``random_crop_size`` set, each item is a random spatial crop of the full
field (the paper's "intermediate pre-cropping" -- more crop diversity per epoch,
smaller batches in memory); the geometry (and SDF) are cropped to match, so a
batch no longer shares one geometry object and :func:`snapshot_collate` falls
back to stacking per-sample geometry. Default ``None`` (full field) keeps the
shared-geometry fast path and the smoke tests trivial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from neural_surrogates.sdf import normalize_sdf_mode
from neural_surrogates.sdf import sdf_features as compute_sdf_features
from torch.utils.data import Dataset, default_collate


def snapshot_collate(batch: list[dict]) -> dict:
    """Collate snapshot items, shipping a shared geometry (+ SDF) once.

    When every item carries the *same* geometry tensor (the full-field default,
    where equal masks are deduped to one object) the mask and SDF features are
    emitted once with a leading singleton dim -- ``(1, *grid)`` / ``(1, C, *grid)``
    -- exactly like :func:`~neural_surrogates.datasets.transition.transition_collate`,
    so ``B-1`` redundant host->GPU copies are avoided. When they differ (random
    cropping ships a per-sample geometry crop), everything is stacked with the
    default collate into ``(B, *grid)`` / ``(B, C, *grid)``; the trainer handles
    both leading shapes.
    """
    geometry = batch[0]["geometry"]
    shared = all(item["geometry"] is geometry for item in batch)
    if not shared:
        return default_collate(batch)
    keys_shared = {"geometry"}
    has_features = "geom_features" in batch[0]
    if has_features:
        keys_shared.add("geom_features")
    rest = [{k: v for k, v in item.items() if k not in keys_shared} for item in batch]
    collated = default_collate(rest)
    collated["geometry"] = geometry.unsqueeze(0)
    if has_features:
        collated["geom_features"] = batch[0]["geom_features"].unsqueeze(0)
    return collated


class SnapshotDataset(Dataset):
    """Single-time-slice samples flattened across every trajectory in a split.

    A trajectory with ``T`` saved time steps contributes ``ceil(T /
    time_stride)`` samples; the dataset length is the sum across all
    trajectories in ``<root>/state/<split>/``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        state_vars: Sequence[str] = ("u", "v", "w"),
        geometry_var: str | None = "blanking",
        cache: bool = False,
        dtype: torch.dtype = torch.float32,
        time_stride: int = 1,
        random_crop_size: int | None = None,
        sdf_features: bool | str = "none",
        sdf_clamp_cells: float = 32.0,
    ) -> None:
        if time_stride < 1:
            raise ValueError(f"time_stride must be >= 1, got {time_stride}")
        if random_crop_size is not None and random_crop_size < 1:
            raise ValueError(
                f"random_crop_size must be >= 1 or None, got {random_crop_size}"
            )
        self.root = Path(root_dir)
        self.split = split
        self.state_vars = tuple(state_vars)
        self.geometry_var = geometry_var
        self.cache = cache
        self.dtype = dtype
        self.time_stride = int(time_stride)
        self.random_crop_size = random_crop_size
        self.sdf_feature_mode = normalize_sdf_mode(sdf_features)
        self.sdf_clamp_cells = float(sdf_clamp_cells)

        # No params for an AE: expose the empty attributes the shared
        # normalization helper reads so its param branch degenerates to empty
        # stats without a special case.
        self.param_names: tuple[str, ...] = ()

        state_dir = self.root / "state" / split
        if not state_dir.is_dir():
            raise FileNotFoundError(f"missing state split dir: {state_dir}")
        self._state_files: list[Path] = sorted(state_dir.glob("sample_*.nc"))
        if not self._state_files:
            raise ValueError(f"split '{split}' is empty under {self.root}")

        self._traj_lengths: list[int] = [
            self._read_time_length(p) for p in self._state_files
        ]
        # Zero-width per-trajectory param tensors: torch.cat over them yields a
        # (sum_T, 0) array whose mean/std are empty -- see the module docstring.
        self._params: list[torch.Tensor] = [
            torch.zeros((t_len, 0), dtype=self.dtype) for t_len in self._traj_lengths
        ]

        # Flat (trajectory, t) index, strided in time to decorrelate snapshots.
        self._index: list[tuple[int, int]] = [
            (traj, t)
            for traj, t_len in enumerate(self._traj_lengths)
            for t in range(0, t_len, self.time_stride)
        ]

        # Per-trajectory geometry, content-deduped (equal masks share one object)
        # -- mirrors TransitionDataset so single-geometry splits keep one mask in
        # memory and snapshot_collate's identity check fires.
        self._geometries: list[torch.Tensor] = []
        self._traj_geometry: list[int] = []
        buckets: dict[tuple, list[int]] = {}
        for state_path in self._state_files:
            geom = self._load_geometry(state_path)
            key = (tuple(geom.shape), int(geom.sum().item()))
            gi = next(
                (
                    i
                    for i in buckets.get(key, [])
                    if torch.equal(self._geometries[i], geom)
                ),
                None,
            )
            if gi is None:
                self._geometries.append(geom)
                gi = len(self._geometries) - 1
                buckets.setdefault(key, []).append(gi)
            self._traj_geometry.append(gi)
        # One EDT per unique geometry, at init (never per __getitem__).
        self._geom_features: list[torch.Tensor] | None = (
            [
                compute_sdf_features(
                    g, clamp_cells=self.sdf_clamp_cells, mode=self.sdf_feature_mode
                )
                for g in self._geometries
            ]
            if self.sdf_feature_mode != "none"
            else None
        )
        self._state_cache: dict[int, xr.Dataset] | None = None

    @property
    def sample_index(self) -> list[tuple[int, int]]:
        """Flat ``(trajectory, t)`` pair per sample (read-only).

        Exposed under the same name as ``TransitionDataset.sample_index`` so
        :class:`~neural_surrogates.datasets.sampler.TrajectoryBatchSampler` can
        bucket multi-geometry snapshot splits by trajectory (it reads only
        ``sample_index`` + ``grid_shape``).
        """
        return self._index

    # -- geometry accessors (mirror TransitionDataset) --------------------- #

    def geometry_for(self, traj: int) -> torch.Tensor:
        """Fluid mask ``(*grid,)`` of trajectory ``traj`` (shared when equal)."""
        return self._geometries[self._traj_geometry[traj]]

    def geom_features_for(self, traj: int) -> torch.Tensor | None:
        if self._geom_features is None:
            return None
        return self._geom_features[self._traj_geometry[traj]]

    def grid_shape(self, traj: int) -> tuple[int, ...]:
        return tuple(self.geometry_for(traj).shape)

    # -- loading helpers --------------------------------------------------- #

    @staticmethod
    def _read_time_length(state_path: Path) -> int:
        with xr.open_dataset(state_path) as ds:
            return int(ds.sizes["time"])

    def _load_geometry(self, state_path: Path) -> torch.Tensor:
        with xr.open_dataset(state_path) as ds:
            if self.geometry_var is not None and self.geometry_var in ds.data_vars:
                da = ds[self.geometry_var]
                if "time" in da.dims:
                    da = da.isel(time=0)
                obstacle = np.asarray(da.values, dtype=np.float64)
                return torch.from_numpy(1.0 - obstacle).to(self.dtype)
            channels = [np.asarray(ds[v].isel(time=0).values) for v in self.state_vars]
        # Fallback (no ``geometry_var``): mark fluid where the first snapshot's
        # stacked state is non-zero. This ASSUMES obstacles are exact zeros
        # (pylbm) and would misclassify a genuinely zero-velocity fluid cell as
        # obstacle; uDALES/PALM datasets must ship ``blanking`` (the primary path
        # above). Mirrors ``TransitionDataset._load_geometry``.
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
        state = self.__dict__.copy()
        state["_state_cache"] = None
        return state

    def __len__(self) -> int:
        return len(self._index)

    def _random_crop(
        self, state: torch.Tensor, geometry: torch.Tensor, features: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Random ``random_crop_size`` cube of ``(state, geometry, features)``.

        A dim smaller than the crop size is taken whole (no padding here -- the
        model pads to the encoder-crop multiple)."""
        cs = self.random_crop_size
        assert cs is not None
        grid = geometry.shape  # (nz, ny, nx)
        starts = []
        sizes = []
        for n in grid:
            c = min(cs, n)
            start = int(torch.randint(0, n - c + 1, (1,)).item()) if n > c else 0
            starts.append(start)
            sizes.append(c)
        zs, ys, xs = starts
        zc, yc, xc = sizes
        sl = (slice(zs, zs + zc), slice(ys, ys + yc), slice(xs, xs + xc))
        state = state[:, sl[0], sl[1], sl[2]]
        geometry = geometry[sl]
        if features is not None:
            features = features[:, sl[0], sl[1], sl[2]]
        return (
            state.contiguous(),
            geometry.contiguous(),
            (features.contiguous() if features is not None else None),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, t = self._index[idx]
        ds = self._get_state_ds(traj)
        snap = ds.isel(time=t)
        channels = np.stack([np.asarray(snap[v].values) for v in self.state_vars], 0)
        state = torch.from_numpy(channels).to(self.dtype)
        geometry = self.geometry_for(traj)
        features = self.geom_features_for(traj)
        if self.random_crop_size is not None:
            state, geometry, features = self._random_crop(state, geometry, features)
        item = {"state": state, "geometry": geometry}
        if features is not None:
            item["geom_features"] = features
        return item
