"""Trainer for the Eq (9) domain-decomposed patch objective.

:class:`PatchTrainer` trains a :class:`DomainDecomposed` model end-to-end under
the four-term loss of :class:`neural_surrogates.dd_loss.DomainDecompositionLoss`.
It shares all of its scaffolding with the generic
:class:`neural_surrogates.training.Trainer` via :class:`BaseTraining` -- LR
scheduling, mixed precision, ``torch.compile``, the pushforward-rollout
curriculum, gradient clipping, checkpoint/resume and metrics logging are all
available here too -- and differs only in :meth:`_final_loss`, where the model's
per-patch intermediates feed the Eq (9) objective.

Why full-field batches (not ``PatchTransitionDataset``)
-------------------------------------------------------
The interface and divergence terms of Eq (9) couple *neighbouring* patches.
A per-patch :class:`PatchTransitionDataset` item cannot supply its neighbours'
predictions within an arbitrary minibatch (the neighbours may land in different
batches, or be absent entirely), so those terms are not computable from
isolated patch items. Instead this trainer feeds **full-field** transition
batches (the existing :class:`neural_surrogates.datasets.transition.TransitionDataset`): the
model tiles the field internally and ``forward(..., return_intermediates=True)``
exposes every patch's prediction block, so all neighbours are present every
step and the interface/divergence terms are well defined.

``PatchTransitionDataset`` remains available for one-step, delta-only patch
training (no interface/divergence coupling), where isolated patch items suffice.

Pushforward + Eq (9)
--------------------
Under a pushforward horizon > 1 the rollout prefix advances the merged field
with the plain ``forward`` (no intermediates), and only the final step is run
with ``return_intermediates=True`` so the interface / divergence / coarse terms
-- which are defined per single step -- are evaluated on the last transition.
With the default one-step horizon this reduces exactly to the original
single-step patch objective.
"""

from __future__ import annotations

import torch

from neural_surrogates.training.base import BaseTraining


class PatchTrainer(BaseTraining):
    """End-to-end trainer for :class:`DomainDecomposed` under Eq (9).

    Accepts the full :class:`BaseTraining` knob set; ``loss_fn`` must be a
    :class:`DomainDecompositionLoss` (or a compatible callable returning
    ``(total, terms)`` from the ``info`` dict of
    ``model(..., return_intermediates=True)``). The model's ``forward`` must
    accept ``return_intermediates=True`` and return ``(state_next, info)``.

    Note ``mask_loss`` is handled *inside* the DD loss (its own ``mask_loss``
    flag), so the trainer-level ``mask_loss`` does not slice the prediction here.
    """

    def _final_loss(
        self,
        state: torch.Tensor,
        state_next: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        K = params.shape[1]
        state_next_pred, info = self.model(
            state, params[:, K - 1, :], geometry, return_intermediates=True
        )
        loss, terms = self.loss_fn(
            info=info,
            state_next=state_next_pred,
            target_next=state_next,
            geometry=geometry,
            dd=getattr(self.model, "dd", None),
        )
        # Expose the four-term breakdown so BaseTraining logs it per epoch.
        self._aux_terms = terms
        return loss
