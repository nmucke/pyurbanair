"""PyTorch datasets over a pyurbanair ``training_data/`` split.

The training-data layout is documented in ``docs/training_data.md``. Two
datasets read the same on-disk split:

* :class:`TransitionDataset` -- flattens every trajectory into full-field
  ``(state_n, params_n, geometry) -> state_{n+K}`` one-step / K-step
  (pushforward-trick) samples.
* :class:`PatchTransitionDataset` -- a subclass that instead yields one sample
  *per spatial patch* of a two-level overlapping domain decomposition.
"""

from neural_surrogates.datasets.patch import PatchTransitionDataset
from neural_surrogates.datasets.transition import TransitionDataset

__all__ = ["TransitionDataset", "PatchTransitionDataset"]
