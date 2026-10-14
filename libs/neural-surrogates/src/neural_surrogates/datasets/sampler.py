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

``group_by="grid"`` relaxes this to one *grid shape* per batch: frames of
different trajectories (different geometries) on the same grid share a batch,
for more diverse gradients per step. The tensors still stack; the geometry
then travels per sample, which the snapshot collates and latent-cache items
support (``transition_collate`` does not, and says so).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:  # annotation-only: the sampler touches no dataset internals
    from neural_surrogates.datasets.transition import TransitionDataset


class TrajectoryBatchSampler(Sampler[list[int]]):
    """Yield batches whose samples all come from one trajectory (or one grid).

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
    group_by:
        ``"trajectory"`` (default): every batch holds frames of one
        trajectory, as described above. ``"grid"``: every batch holds frames
        of one grid shape, drawn across all trajectories on that grid
        (shuffled within the grid, batch order shuffled across grids;
        ``shuffle=False`` gives a deterministic grid-major order). The batch
        size, ``cell_budget`` and ``drop_last`` then apply per grid shape
        instead of per trajectory. Geometries then differ within a batch, so
        it needs a collate that ships per-sample geometry
        (``snapshot_collate``, ``snapshot_history_collate``, the latent
        cache's default collate), not ``transition_collate``.

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
        group_by: str = "trajectory",
    ) -> None:
        if group_by not in ("trajectory", "grid"):
            raise ValueError(
                f"group_by must be 'trajectory' or 'grid', got {group_by!r}"
            )
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
        self.group_by = group_by
        self._epoch = 0

    def _batch_size_for(self, key: int | tuple[int, ...]) -> int:
        """Batch size of a group: a trajectory index or a grid shape."""
        if self.cell_budget is None:
            assert self.batch_size is not None  # enforced in __init__
            return self.batch_size
        grid = key if isinstance(key, tuple) else self.dataset.grid_shape(key)
        budget = max(1, self.cell_budget // math.prod(grid))
        return budget if self.batch_size is None else min(self.batch_size, budget)

    def _groups(self) -> dict[int | tuple[int, ...], list[int]]:
        """Flat indices per batch group (a trajectory, or a grid shape), in
        first-seen order."""
        groups: dict[int | tuple[int, ...], list[int]] = {}
        grids: dict[int, tuple[int, ...]] = {}
        for flat, (traj, _t) in enumerate(self.dataset.sample_index):
            key: int | tuple[int, ...] = traj
            if self.group_by == "grid":
                if traj not in grids:
                    grids[traj] = tuple(self.dataset.grid_shape(traj))
                key = grids[traj]
            groups.setdefault(key, []).append(flat)
        return groups

    def _batches(self, generator: torch.Generator | None) -> list[list[int]]:
        batches: list[list[int]] = []
        for key, flats in self._groups().items():
            if generator is not None:
                order = torch.randperm(len(flats), generator=generator).tolist()
                flats = [flats[i] for i in order]
            size = self._batch_size_for(key)
            for start in range(0, len(flats), size):
                batch = flats[start : start + size]
                if self.drop_last and len(batch) < size:
                    continue
                batches.append(batch)
        if generator is not None:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        return batches

    def set_epoch(self, epoch: int) -> None:
        """Shuffle the next ``__iter__`` as epoch ``epoch`` (0-based) of a
        fresh run; the trainer calls it so a resumed run does not replay the
        shuffles of its first epochs."""
        self._epoch = int(epoch)

    def _generator(self, epoch: int) -> torch.Generator | None:
        if not self.shuffle:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return generator

    def __iter__(self) -> Iterator[list[int]]:
        generator = self._generator(self._epoch)
        if self.shuffle:
            self._epoch += 1
        yield from self._batches(generator)

    def __len__(self) -> int:
        return len(self._batches(None))

    def batch_sizes(self, epoch: int | None = None) -> list[int]:
        """Samples in each batch of an epoch, in the order epoch ``epoch``
        (0-based, see ``set_epoch``) draws them; ``None``: unshuffled order
        (shuffling reorders the batches but keeps their sizes)."""
        generator = None if epoch is None else self._generator(epoch)
        return [len(batch) for batch in self._batches(generator)]
