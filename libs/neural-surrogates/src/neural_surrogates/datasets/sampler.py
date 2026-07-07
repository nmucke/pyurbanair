"""Trajectory-bucketed batch sampling for multi-geometry transition datasets.

A :class:`~neural_surrogates.datasets.transition.TransitionDataset` over
random-geometry training data mixes trajectories with different spatial grids
(and different geometry masks). Plain shuffled batching breaks twice there:
``default_collate`` cannot stack states of different shapes, and
``transition_collate``'s ship-geometry-once contract requires one mask per
batch. :class:`TrajectoryBatchSampler` restores both invariants by drawing
every batch from a single trajectory — no padding, no per-sample geometry
copies, and the architectures (P3D pads any grid to a multiple of 16
internally) need no changes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:  # annotation-only: the sampler touches no dataset internals
    from neural_surrogates.datasets.transition import TransitionDataset


class TrajectoryBatchSampler(Sampler[list[int]]):
    """Yield batches whose samples all come from one trajectory.

    Samples of one trajectory share a grid and a geometry, so grouping batches
    per trajectory keeps ``transition_collate`` (and the trainer's per-batch
    geometry cache) valid on multi-geometry splits. Every trajectory has
    ``T - K`` samples, so same-trajectory batches are always available;
    batches are shuffled across trajectories each epoch, so geometries still
    mix step to step.

    Parameters
    ----------
    dataset:
        A :class:`TransitionDataset` (anything exposing ``sample_index`` and
        ``grid_shape``).
    batch_size:
        Upper bound on samples per batch.
    cell_budget:
        Upper bound on *total grid cells* per batch: a trajectory on ``N``
        cells gets batches of ``max(1, cell_budget // N)``. This keeps memory
        roughly flat when domain sizes span a wide range (the UrbanTALES
        realistic pool spans ~25x in cell count); set it to
        ``batch_size_ref * cells_ref`` of a known-good configuration. At least
        one of ``batch_size`` / ``cell_budget`` must be set; when both are,
        the smaller bound wins.
    shuffle:
        Shuffle samples within each trajectory and the batch order across
        trajectories (seeded, advancing per epoch). ``False`` gives a
        deterministic trajectory-major order (validation).
    drop_last:
        Drop each trajectory's final short batch so every batch of that
        trajectory has the full per-trajectory size (cheaper ``torch.compile``
        / cudnn-benchmark churn). A trajectory with fewer samples than its
        batch size is then dropped entirely.
    seed:
        Base seed for the per-epoch shuffles.

    The dataset's flat index is re-read at every ``__iter__`` (and ``__len__``),
    so a mid-training ``set_pushforward_steps`` — the pushforward curriculum —
    is picked up on the next epoch without rebuilding the sampler or loader.
    """

    def __init__(
        self,
        dataset: "TransitionDataset",
        batch_size: int | None = None,
        cell_budget: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size is None and cell_budget is None:
            raise ValueError("set batch_size, cell_budget, or both")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if cell_budget is not None and cell_budget < 1:
            raise ValueError(f"cell_budget must be >= 1, got {cell_budget}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.cell_budget = cell_budget
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0

    def _batch_size_for(self, traj: int) -> int:
        if self.cell_budget is None:
            assert self.batch_size is not None  # enforced in __init__
            return self.batch_size
        budget = max(1, self.cell_budget // math.prod(self.dataset.grid_shape(traj)))
        return budget if self.batch_size is None else min(self.batch_size, budget)

    def _batches(self, generator: torch.Generator | None) -> list[list[int]]:
        by_traj: dict[int, list[int]] = {}
        for flat, (traj, _t) in enumerate(self.dataset.sample_index):
            by_traj.setdefault(traj, []).append(flat)
        batches: list[list[int]] = []
        for traj, flats in by_traj.items():
            if generator is not None:
                order = torch.randperm(len(flats), generator=generator).tolist()
                flats = [flats[i] for i in order]
            size = self._batch_size_for(traj)
            for start in range(0, len(flats), size):
                batch = flats[start : start + size]
                if self.drop_last and len(batch) < size:
                    continue
                batches.append(batch)
        if generator is not None:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = None
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self._epoch)
            self._epoch += 1
        yield from self._batches(generator)

    def __len__(self) -> int:
        return len(self._batches(None))
