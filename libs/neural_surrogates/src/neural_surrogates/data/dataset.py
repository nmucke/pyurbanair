"""Lazy index-map windowing of the trajectory corpus (§6.1.1).

Trajectories are stored **whole** (never pre-windowed, §5). This module builds
a flat index ``sample_id -> (trajectory_id, t_anchor, history_len)`` and slices
lazily, returning architecture-independent **field-space window records**:

    hist_fields    [K, C, Z, Y, X]   # K history frames; left-padded if short
    hist_params    [K, P]            # dense per-step params at those frames
    hist_mask      [K]               # 1 = real frame, 0 = left-pad
    future_params  [H, P]            # boundary conditions for the H rollout steps
    target_fields  [H, C, Z, Y, X]   # next H *true* frames (pushforward targets)

The horizon ``H`` is curriculum-controlled: bumping it just recomputes the
index (nothing is re-materialized). Splitting is by trajectory, done by the
corpus before the index is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .grid import GridMeta
from .normalization import Normalization
from ..utils.schema import ParamSchema


class Corpus:
    """Read interface every corpus backend (in-memory, Zarr) implements."""

    grid: GridMeta
    param_schema: ParamSchema
    var_names: tuple[str, ...]
    static_channels: np.ndarray  # [S, Z, Y, X]

    def split_ids(self, split: str) -> list[str]:
        raise NotImplementedError

    def num_frames(self, traj_id: str) -> int:
        raise NotImplementedError

    def load_fields(self, traj_id: str) -> np.ndarray:
        """Return ``[T, C, Z, Y, X]`` raw (un-normalized) fields."""
        raise NotImplementedError

    def load_params(self, traj_id: str) -> np.ndarray:
        """Return ``[T, P]`` per-frame encoded conditioning (§1.5)."""
        raise NotImplementedError


class InMemoryCorpus(Corpus):
    """A corpus held entirely in RAM — used by tests and the CPU smoke stage."""

    def __init__(
        self,
        fields: dict[str, np.ndarray],
        params: dict[str, np.ndarray],
        grid: GridMeta,
        param_schema: ParamSchema,
        var_names: Sequence[str],
        static_channels: np.ndarray,
        splits: dict[str, list[str]],
    ) -> None:
        self._fields = {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}
        self._params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        self.grid = grid
        self.param_schema = param_schema
        self.var_names = tuple(var_names)
        self.static_channels = np.asarray(static_channels, dtype=np.float32)
        self._splits = {k: list(v) for k, v in splits.items()}

    def split_ids(self, split: str) -> list[str]:
        return list(self._splits.get(split, []))

    def num_frames(self, traj_id: str) -> int:
        return int(self._fields[traj_id].shape[0])

    def load_fields(self, traj_id: str) -> np.ndarray:
        return self._fields[traj_id]

    def load_params(self, traj_id: str) -> np.ndarray:
        return self._params[traj_id]


@dataclass(frozen=True)
class WindowRecord:
    hist_fields: np.ndarray
    hist_params: np.ndarray
    hist_mask: np.ndarray
    future_params: np.ndarray
    target_fields: np.ndarray


class WindowDataset:
    """Sliding-window view over a corpus split (§6.1.1).

    Args:
        corpus: Any :class:`Corpus`.
        split: Split name (``"train"``/``"val"``/``"test"``).
        history_len: ``K``.
        horizon: ``H`` (pushforward length); change with :meth:`set_horizon`.
        stride: Sliding-window stride between anchors.
        normalization: Applied to fields/targets on load (None = raw).
    """

    def __init__(
        self,
        corpus: Corpus,
        split: str,
        history_len: int,
        horizon: int,
        *,
        stride: int = 1,
        normalization: Optional[Normalization] = None,
    ) -> None:
        if history_len < 1 or horizon < 1 or stride < 1:
            raise ValueError("history_len, horizon, stride must all be >= 1.")
        self.corpus = corpus
        self.split = split
        self.history_len = history_len
        self.stride = stride
        self.normalization = normalization
        self._ids = corpus.split_ids(split)
        self.set_horizon(horizon)

    def set_horizon(self, horizon: int) -> None:
        """Rebuild the index for a new pushforward horizon (curriculum, §6.1)."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        self.horizon = horizon
        index: list[tuple[str, int, int]] = []
        for traj_id in self._ids:
            n_t = self.corpus.num_frames(traj_id)
            # need t_anchor + H <= n_t - 1  =>  t_anchor <= n_t - 1 - H
            last_anchor = n_t - 1 - horizon
            for t_anchor in range(0, last_anchor + 1, self.stride):
                history_len = min(self.history_len, t_anchor + 1)
                index.append((traj_id, t_anchor, history_len))
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def _normalize(self, fields: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return fields
        return self.normalization.apply(fields)

    def __getitem__(self, i: int) -> WindowRecord:
        traj_id, t_anchor, hl = self._index[i]
        k, h = self.history_len, self.horizon
        fields = self.corpus.load_fields(traj_id)
        params = self.corpus.load_params(traj_id)

        # history frames: [t_anchor-hl+1 .. t_anchor]
        hist = self._normalize(fields[t_anchor - hl + 1 : t_anchor + 1])
        hist_params = params[t_anchor - hl + 1 : t_anchor + 1]

        c, z, y, x = fields.shape[1:]
        p = params.shape[1]
        if hl < k:  # left-pad to K
            pad_f = np.zeros((k - hl, c, z, y, x), dtype=np.float32)
            pad_p = np.zeros((k - hl, p), dtype=np.float32)
            hist = np.concatenate([pad_f, hist], axis=0)
            hist_params = np.concatenate([pad_p, hist_params], axis=0)
        mask = np.concatenate(
            [np.zeros(k - hl, dtype=np.float32), np.ones(hl, dtype=np.float32)]
        )

        future_params = params[t_anchor + 1 : t_anchor + 1 + h]
        target = self._normalize(fields[t_anchor + 1 : t_anchor + 1 + h])

        return WindowRecord(
            hist_fields=hist.astype(np.float32),
            hist_params=hist_params.astype(np.float32),
            hist_mask=mask,
            future_params=future_params.astype(np.float32),
            target_fields=target.astype(np.float32),
        )


def collate(records: Sequence[WindowRecord]) -> dict[str, np.ndarray]:
    """Stack window records into a batch of arrays (leading axis ``B``)."""
    return {
        "hist_fields": np.stack([r.hist_fields for r in records]),
        "hist_params": np.stack([r.hist_params for r in records]),
        "hist_mask": np.stack([r.hist_mask for r in records]),
        "future_params": np.stack([r.future_params for r in records]),
        "target_fields": np.stack([r.target_fields for r in records]),
    }


def iterate_batches(
    dataset: WindowDataset,
    batch_size: int,
    *,
    rng: Optional[np.random.Generator] = None,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Yield collated batches over the dataset, optionally shuffled."""
    n = len(dataset)
    order = np.arange(n)
    if shuffle:
        (rng or np.random.default_rng()).shuffle(order)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        yield collate([dataset[int(j)] for j in idx])
