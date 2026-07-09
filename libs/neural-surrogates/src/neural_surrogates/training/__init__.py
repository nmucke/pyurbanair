"""Neural-surrogate trainers.

All trainers share :class:`BaseTraining`, which holds the architecture-agnostic
machinery (LR scheduling, mixed precision, ``torch.compile``, pushforward
curriculum, gradient clipping, checkpoint/resume and metrics logging). The two
concrete trainers differ only in how a batch becomes a loss:

* :class:`Trainer` -- generic full-grid masked element-wise loss.
* :class:`PatchTrainer` -- the four-term Eq (9) domain-decomposition loss over
  the model's per-patch intermediates.
* :class:`AutoencoderTrainer` -- snapshot (V)AE reconstruction + KL loss (no
  rollout / pushforward), for Tadpole-style pre-training.
"""

from neural_surrogates.training.autoencoder import AutoencoderTrainer
from neural_surrogates.training.base import BaseTraining
from neural_surrogates.training.patch import PatchTrainer
from neural_surrogates.training.standard import Trainer

__all__ = ["BaseTraining", "Trainer", "PatchTrainer", "AutoencoderTrainer"]
