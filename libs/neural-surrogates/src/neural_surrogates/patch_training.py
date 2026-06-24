"""Trainer for the Eq (9) domain-decomposed patch objective.

:class:`PatchTrainer` trains a :class:`DomainDecomposed` model end-to-end under
the four-term loss of :class:`neural_surrogates.dd_loss.DomainDecompositionLoss`.
It is a separate, lean class -- the generic
:class:`neural_surrogates.training.Trainer` is left untouched -- but mirrors its
structure (device handling, best-checkpoint saving, train/validate loop).

Why full-field batches (not ``PatchTransitionDataset``)
-------------------------------------------------------
The interface and divergence terms of Eq (9) couple *neighbouring* patches.
A per-patch :class:`PatchTransitionDataset` item cannot supply its neighbours'
predictions within an arbitrary minibatch (the neighbours may land in different
batches, or be absent entirely), so those terms are not computable from
isolated patch items. Instead this trainer feeds **full-field** transition
batches (the existing :class:`neural_surrogates.data.TransitionDataset`): the
model tiles the field internally and ``forward(..., return_intermediates=True)``
exposes every patch's prediction block, so all neighbours are present every
step and the interface/divergence terms are well defined.

``PatchTransitionDataset`` remains available for one-step, delta-only patch
training (no interface/divergence coupling), where isolated patch items suffice.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neural_surrogates.dd_loss import DomainDecompositionLoss


class PatchTrainer:
    """End-to-end trainer for :class:`DomainDecomposed` under Eq (9).

    Parameters
    ----------
    model:
        A :class:`DomainDecomposed` (anything whose ``forward`` accepts
        ``return_intermediates=True`` and returns ``(state_next, info)``).
    train_loader, val_loader:
        Loaders over a **full-field** ``TransitionDataset`` (items with
        ``state_n``, ``state_next``, ``params_n``, ``geometry``).
    optimizer:
        A torch optimizer over ``model.parameters()``.
    loss_fn:
        A :class:`DomainDecompositionLoss` (or compatible callable returning
        ``(total, terms)``).
    num_epochs:
        Number of training epochs.
    device:
        Torch device.
    weights_path:
        Where to save the best (lowest val) state-dict; ``None`` disables saving.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: DomainDecompositionLoss,
        num_epochs: int,
        device: str | torch.device = "cpu",
        weights_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = int(num_epochs)
        self.weights_path = (
            Path(weights_path) if weights_path is not None else None
        )
        # The geometry mask is identical for every sample, so it is moved to
        # the device once, lazily from the first batch (mirrors Trainer).
        self._geometry: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """One forward + loss on a full-field transition batch. The model uses
        only the first pushforward step's params (``params_n[:, 0]``); the
        objective here is one-step (no pushforward unroll)."""
        state_n = batch["state_n"].to(self.device, non_blocking=True)
        target_next = batch["state_next"].to(self.device, non_blocking=True)
        params = batch["params_n"].to(self.device, non_blocking=True)
        if self._geometry is None:
            self._geometry = batch["geometry"][0].to(self.device)
        geometry = self._geometry.expand(
            state_n.shape[0], *self._geometry.shape
        )

        state_next, info = self.model(
            state_n, params[:, 0, :], geometry, return_intermediates=True
        )
        loss, terms = self.loss_fn(
            info=info,
            state_next=state_next,
            target_next=target_next,
            geometry=geometry,
            dd=getattr(self.model, "dd", None),
        )
        return loss, terms

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for batch in self.train_loader:
            loss, _ = self._step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for batch in self.val_loader:
            loss, _ = self._step(batch)
            total += float(loss.item())
            n += 1
        return total / max(n, 1)

    def fit(self) -> dict:
        """Train + validate for ``num_epochs``, saving the best (lowest val)
        weights. Returns a small history dict (``train``/``val`` loss lists)."""
        if self.weights_path is not None:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        history = {"train": [], "val": []}
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate()
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            print(
                f"epoch {epoch + 1}/{self.num_epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                if self.weights_path is not None:
                    torch.save(self.model.state_dict(), self.weights_path)
        if self.weights_path is not None and self.weights_path.exists():
            self.model.load_state_dict(
                torch.load(self.weights_path, map_location=self.device)
            )
        return history
