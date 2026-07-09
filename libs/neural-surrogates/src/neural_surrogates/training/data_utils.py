"""Shared data-side helpers for the surrogate train / fine-tune scripts.

Both ``scripts/neural_surrogate/train_neural_surrogate.py`` and
``scripts/neural_surrogate/finetune_neural_surrogate.py`` need the same two
things: build the (optionally trajectory-bucketed) ``DataLoader`` for a split,
and compute-or-load the cached per-channel normalization statistics. These lived
as underscore-prefixed locals in the train script and were reached into by the
fine-tune script via ``importlib`` -- a brittle coupling a rename would break
silently. They live here so both scripts import them normally.

``build_loader`` uses ``hydra.utils.instantiate`` (already a library dependency,
see ``forward_model.py``); the normalization helpers use only numpy/xarray/torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    # The per-split dataset the normalization pass reads (state_vars,
    # _state_files, geometry_for, _params, split, param_names, geometry_var,
    # root). ``from __future__ import annotations`` keeps this a lazy string so
    # there is no runtime import (and no import cycle with datasets/).
    from neural_surrogates.datasets.transition import TransitionDataset

# Bump when the meaning of the cached arrays changes so old caches are ignored.
_NORM_STATS_VERSION = 1


def build_loader(cfg: DictConfig, dataset: Dataset, *, train: bool) -> DataLoader:
    """DataLoader for one split, honouring the optional ``batch_sampler`` block.

    Without it (``batch_sampler: null``, the default) this is the original
    ``instantiate(cfg.dataloader, ...)`` with shuffle forced off for val. With
    it, the sampler groups every batch by trajectory -- required for
    multi-geometry datasets, whose grids cannot be stacked by plain shuffled
    batching -- and DataLoader's own ``batch_size`` / ``shuffle`` / ``drop_last``
    must be neutralised (they are mutually exclusive with a batch sampler; the
    sampler config carries them instead). Val keeps a deterministic order.
    """
    sampler_cfg = cfg.get("batch_sampler")
    if sampler_cfg is None:
        if train:
            return instantiate(cfg.dataloader, dataset=dataset)
        return instantiate(cfg.dataloader, dataset=dataset, shuffle=False)
    overrides = {} if train else {"shuffle": False}
    sampler = instantiate(sampler_cfg, dataset=dataset, **overrides)
    return instantiate(
        cfg.dataloader,
        dataset=dataset,
        batch_sampler=sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )


def _compute_normalization_stats(
    train_ds: "TransitionDataset",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel state and param mean/std over the training split's fluid cells.

    Mirrors upstream UPT's dataset standardisation: the network sees zero-mean,
    unit-variance fields. Stats are streamed file-by-file (sum / sum-of-squares
    in float64) and restricted to fluid cells via each trajectory's own geometry
    mask (multi-geometry splits have per-trajectory masks and grids) so the
    masked-out obstacle zeros do not bias them.
    """
    n_state = len(train_ds.state_vars)
    s_sum: np.ndarray = np.zeros(n_state, dtype=np.float64)
    s_sqsum: np.ndarray = np.zeros(n_state, dtype=np.float64)
    s_count = 0
    for traj, state_path in enumerate(train_ds._state_files):
        fluid = train_ds.geometry_for(traj).cpu().numpy().astype(bool)
        with xr.open_dataset(state_path) as ds:
            for c, var in enumerate(train_ds.state_vars):
                vals = np.asarray(ds[var].values)  # (T, *grid)
                masked = vals[:, fluid]  # (T, n_fluid)
                s_sum[c] += masked.sum(dtype=np.float64)
                s_sqsum[c] += np.square(masked, dtype=np.float64).sum()
            s_count += masked.shape[0] * masked.shape[1]
    state_mean = s_sum / s_count
    state_std = np.sqrt(np.maximum(s_sqsum / s_count - state_mean**2, 0.0))

    params = torch.cat([p for p in train_ds._params], dim=0).cpu().numpy()  # (sum_T,P)
    param_mean = params.mean(axis=0)
    param_std = params.std(axis=0)
    return state_mean, state_std, param_mean, param_std


def _normalization_signature(train_ds: "TransitionDataset") -> str:
    """A stable fingerprint of every input `_compute_normalization_stats` reads.

    The cached stats are only valid for the exact split, channel/param order and
    on-disk state files they were computed from. We key on the dataset class, the
    split name, the state/param variable tuples, the geometry variable (it selects
    the fluid mask), and each state file's ``(name, size, mtime)`` -- so
    regenerating or editing the training data, or changing any of these knobs,
    invalidates the cache and forces a recompute. ``pushforward_steps`` is
    deliberately absent: the stats stream every snapshot regardless of the rollout
    horizon. The dataset class is included so a ``SnapshotDataset`` and a
    ``TransitionDataset`` over the same root/split never read each other's cache
    (the values coincide today, but the two could diverge in how they compute
    stats).
    """
    files = [
        [p.name, st.st_size, int(st.st_mtime)]
        for p in train_ds._state_files
        for st in (p.stat(),)
    ]
    return json.dumps(
        {
            "version": _NORM_STATS_VERSION,
            "dataset_class": type(train_ds).__name__,
            "split": train_ds.split,
            "state_vars": list(train_ds.state_vars),
            "param_names": list(train_ds.param_names),
            "geometry_var": train_ds.geometry_var,
            "files": files,
        },
        sort_keys=True,
    )


def _normalization_cache_path(train_ds: "TransitionDataset") -> Path:
    """Where the cached stats for this split live, inside the data folder."""
    return Path(train_ds.root) / "normalization_stats" / f"{train_ds.split}.npz"


def _load_cached_normalization_stats(
    train_ds: "TransitionDataset",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return cached stats if present and still matching the data, else ``None``."""
    path = _normalization_cache_path(train_ds)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != _normalization_signature(train_ds):
                return None
            return (
                data["state_mean"],
                data["state_std"],
                data["param_mean"],
                data["param_std"],
            )
    except (OSError, KeyError, ValueError) as exc:
        # A corrupt / stale-format cache file must never break training; just
        # fall back to recomputing (which overwrites it).
        print(f"ignoring unreadable normalization cache {path}: {exc}")
        return None


def _save_normalization_stats(
    train_ds: "TransitionDataset",
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Persist stats next to the training data; a write failure is non-fatal."""
    path = _normalization_cache_path(train_ds)
    state_mean, state_std, param_mean, param_std = stats
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            state_mean=state_mean,
            state_std=state_std,
            param_mean=param_mean,
            param_std=param_std,
            signature=_normalization_signature(train_ds),
        )
    except OSError as exc:
        print(f"could not cache normalization stats to {path}: {exc}")


def get_normalization_stats(
    train_ds: "TransitionDataset",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached normalization stats when available, else compute and cache.

    Computing the stats streams the whole training split from disk, which is
    slow on large datasets; caching the result in the data folder makes reruns
    on the same data effectively free.
    """
    cached = _load_cached_normalization_stats(train_ds)
    if cached is not None:
        print(
            f"loaded cached normalization stats from "
            f"{_normalization_cache_path(train_ds)}"
        )
        return cached
    stats = _compute_normalization_stats(train_ds)
    _save_normalization_stats(train_ds, stats)
    print(f"cached normalization stats to {_normalization_cache_path(train_ds)}")
    return stats
