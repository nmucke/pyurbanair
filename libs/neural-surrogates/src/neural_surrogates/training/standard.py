"""Generic single-loop trainer for the neural-surrogate baselines."""

from __future__ import annotations

from typing import Any

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
    defines how the rollout's predictions become a scalar loss.

    ``rollout_loss`` picks which steps of a ``K``-step training rollout are
    supervised:

    - ``"final"`` (default): only the last one, against ``state_next``
      (:meth:`BaseTraining._forward`).
    - ``"sum"``: every one, against the dataset's ``state_targets``
      (:class:`~neural_surrogates.datasets.transition.RolloutTransitionDataset`);
      the loss is the *mean* of the ``K`` per-step losses, so its scale (and the
      learning rate) matches ``"final"``'s. See :meth:`_rollout_sum_loss`.

    Validation always uses ``"final"``, so its loss stays comparable between
    the two.
    """

    def __init__(self, *args: Any, rollout_loss: str = "final", **kwargs: Any) -> None:
        if rollout_loss not in ("final", "sum"):
            raise ValueError(
                f"rollout_loss must be 'final' or 'sum', got {rollout_loss!r}"
            )
        self.rollout_loss = rollout_loss
        super().__init__(*args, **kwargs)

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.rollout_loss == "final":
            return super()._forward(batch)
        if not torch.is_grad_enabled():
            # Validation: the final-state loss, without the training batches'
            # per-step terms.
            self._aux_terms = None
            return super()._forward(batch)
        return self._rollout_sum_loss(batch)

    def _rollout_sum_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """The mean of the masked losses of every rollout step.

        The last ``g = grad_unroll_steps`` steps backpropagate through their
        chain, as for ``"final"``. Each earlier step is trained on its own
        one-step loss from its detached input -- the true state for step 1, the
        previous (detached) prediction after it -- and feeds its detached
        prediction forward. During training (``_backward_part`` set) each of
        those losses is backpropagated at once, so its graph is freed before the
        next step and the peak memory stays that of the ``g``-step chain; the
        returned loss holds their values detached. Called directly with
        gradients on, the whole graph is kept (same gradient). Each early step
        has its own autocast context (left before its backward); no step runs
        under ``no_grad``, so no cast cache is poisoned.
        """
        if "state_targets" not in batch:
            raise ValueError(
                "rollout_loss='sum' needs every intermediate rollout target "
                "('state_targets'): build the dataset as "
                "neural_surrogates.RolloutTransitionDataset"
            )
        state, _, params, geometry = self._prepare_batch(batch)
        targets = batch["state_targets"].to(self.device, non_blocking=True)
        K = params.shape[1]
        g = min(self.grad_unroll_steps, K)
        losses: list[torch.Tensor] = []
        for i in range(K - g):
            with self._autocast():
                pred = self._model_forward(state, params[:, i, :], geometry)
                loss = self._masked_loss(pred, targets[:, i])
            if self._backward_part is not None:
                self._backward_part(loss / K)
                loss = loss.detach()
            losses.append(loss)
            state = self._advance_history(state, pred.detach())
        with self._autocast():
            for i in range(K - g, K):
                for module in self._unroll_no_grad_modules:
                    module._unroll_no_grad = i < K - 1
                try:
                    pred = self._model_forward(state, params[:, i, :], geometry)
                finally:
                    for module in self._unroll_no_grad_modules:
                        module._unroll_no_grad = False
                losses.append(self._masked_loss(pred, targets[:, i]))
                state = self._advance_history(state, pred)
        self._aux_terms = {f"step_{i + 1}": x.detach() for i, x in enumerate(losses)}
        return torch.stack(losses).mean()

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
        return self._masked_loss(pred, state_next)

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``loss_fn(pred, target)``, over the fluid cells with ``mask_loss``."""
        if not self.mask_loss:
            return self.loss_fn(pred, target)
        # uDALES targets carry junk values inside obstacles, which would
        # otherwise be penalised against the model's (masked) zero output.
        mask = self._fluid_mask
        assert mask is not None and self._fluid_scale is not None
        if not (
            isinstance(self.loss_fn, _ZERO_AT_EQUAL)
            and self.loss_fn.reduction == "mean"
        ):
            return self.loss_fn(pred[..., mask], target[..., mask])
        # Same value without boolean indexing, which waits on the GPU to count
        # the selected cells: zero both sides outside the fluid (these losses
        # are 0 there) and rescale the full-grid mean to the fluid mean.
        loss: torch.Tensor = self.loss_fn(
            torch.where(mask, pred, 0.0), torch.where(mask, target, 0.0)
        )
        return loss * self._fluid_scale
