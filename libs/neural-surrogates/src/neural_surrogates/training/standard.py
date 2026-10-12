"""Generic single-loop trainer for the neural-surrogate baselines."""

from __future__ import annotations

import torch
from neural_surrogates.training.base import BaseTraining

# Element-wise losses that are 0 where prediction and target agree, so a masked
# mean can be taken over the full grid with both sides zeroed outside the mask.
_ZERO_AT_EQUAL = (
    torch.nn.MSELoss,
    torch.nn.L1Loss,
    torch.nn.HuberLoss,
    torch.nn.SmoothL1Loss,
)


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
        # ``state`` is whatever the model consumes -- a single frame, or the
        # flattened (B, H*C, *grid) history window -- while the prediction and
        # the target are always a single (B, C, *grid) frame, so the masked
        # comparison below is history-agnostic.
        pred = self._model_forward(state, params[:, K - 1, :], geometry)
        if not self.mask_loss:
            return self.loss_fn(pred, state_next)
        # uDALES targets carry junk values inside obstacles, which would
        # otherwise be penalised against the model's (masked) zero output.
        mask = self._fluid_mask
        assert mask is not None and self._fluid_scale is not None
        if not (
            isinstance(self.loss_fn, _ZERO_AT_EQUAL)
            and self.loss_fn.reduction == "mean"
        ):
            return self.loss_fn(pred[..., mask], state_next[..., mask])
        # Same value without boolean indexing, which waits on the GPU to count
        # the selected cells: zero both sides outside the fluid (these losses
        # are 0 there) and rescale the full-grid mean to the fluid mean.
        loss: torch.Tensor = self.loss_fn(
            torch.where(mask, pred, 0.0), torch.where(mask, state_next, 0.0)
        )
        return loss * self._fluid_scale
