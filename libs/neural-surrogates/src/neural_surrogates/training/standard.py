"""Generic single-loop trainer for the neural-surrogate baselines."""

from __future__ import annotations

import torch

from neural_surrogates.training.base import BaseTraining


class Trainer(BaseTraining):
    """Full-grid trainer: a masked element-wise ``loss_fn(pred, target)`` over
    the pushforward rollout.

    All of the scheduling / mixed-precision / ``torch.compile`` / checkpointing /
    early-stopping machinery lives in :class:`BaseTraining`; this subclass only
    defines how the final rollout step's prediction becomes a scalar loss.
    """

    def _final_loss(
        self,
        state: torch.Tensor,
        state_next: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        K = params.shape[1]
        pred = self.model(state, params[:, K - 1, :], geometry)
        if self.mask_loss:
            # uDALES targets carry junk values inside obstacles, which would
            # otherwise be penalised against the model's (masked) zero output.
            pred = pred[..., self._fluid_mask]
            state_next = state_next[..., self._fluid_mask]
        return self.loss_fn(pred, state_next)
