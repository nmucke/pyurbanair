"""PyTorch datasets over a pyurbanair ``training_data/`` split.

The training-data layout is documented in ``docs/training_data.md``. Two
datasets read the same on-disk split:

* :class:`TransitionDataset` -- flattens every trajectory into full-field
  ``(state_n, params_n, geometry) -> state_{n+K}`` one-step / K-step
  (pushforward-trick) samples.
* :class:`PatchTransitionDataset` -- a subclass that instead yields one sample
  *per spatial patch* of a two-level overlapping domain decomposition.
* :class:`SnapshotDataset` -- single time slices (no next-step target, no
  params) for Tadpole-style autoencoder pre-training.

:class:`TrajectoryBatchSampler` batches multi-geometry splits (one trajectory
-- one grid/geometry -- per batch) with an optional per-batch cell budget.
"""

from neural_surrogates.datasets.patch import PatchTransitionDataset
from neural_surrogates.datasets.sampler import TrajectoryBatchSampler
from neural_surrogates.datasets.snapshot import SnapshotDataset, snapshot_collate
from neural_surrogates.datasets.transition import TransitionDataset

__all__ = [
    "TransitionDataset",
    "PatchTransitionDataset",
    "TrajectoryBatchSampler",
    "SnapshotDataset",
    "snapshot_collate",
]
