"""SSRollingUrbanNet's rollout training (Park & Lee 2026, §II E) on our trainer.

* Roll-K: the model is unrolled over the ``K = pushforward_steps`` targets of a
  :class:`~neural_surrogate_baselines.datasets.RolloutTransitionDataset`, with
  gradients through every step (the paper trains Roll-1, then fine-tunes
  Roll-3);
* loss ``Σ_i [MSE_i + α·L_spec,i]``, summed over the steps, not averaged:
  ``MSE_i`` is ``loss_fn`` on the fluid cells (``mask_loss``), ``L_spec,i`` the
  spectral loss of the masked fields along the lateral axis y.

Adam at a constant 1e-5, batch 2, patience 50 and no gradient clipping are set
in the configs.
"""

from __future__ import annotations

from typing import Any

import torch
from neural_surrogate_baselines.losses import spectral_loss
from neural_surrogates.training.base import BaseTraining


class RolloutTrainer(BaseTraining):
    def __init__(
        self,
        *args: Any,
        alpha: float = 1.0,
        spectral_norm: str = "backward",
        **kwargs: Any,
    ) -> None:
        self.alpha = float(alpha)
        self.spectral_norm = spectral_norm
        super().__init__(*args, **kwargs)

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state, _, params, geometry = self._prepare_batch(batch)
        targets = batch["state_targets"].to(self.device, non_blocking=True)
        assert self._geometry is not None and self._fluid_mask is not None
        mask = self._geometry
        fluid = self._fluid_mask
        mse = spec = torch.zeros((), device=self.device)
        with self._autocast():
            for i in range(targets.shape[1]):
                pred = self._model_forward(state, params[:, i, :], geometry)
                target = targets[:, i]
                if self.mask_loss:
                    mse = mse + self.loss_fn(pred[..., fluid], target[..., fluid])
                else:
                    mse = mse + self.loss_fn(pred, target)
                spec = spec + spectral_loss(
                    pred * mask, target * mask, dim=-2, norm=self.spectral_norm
                )
                state = self._advance_history(state, pred)
        self._aux_terms = {"mse": mse.detach(), "spec": spec.detach()}
        return mse + self.alpha * spec
