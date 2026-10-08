"""Local-FNO's training recipe (Qin et al. 2025, §2.5) on our trainer.

* RMSE over every patch prediction, overlap included, on its real fluid cells;
* the learning rate halves after every epoch (``lr_step_gamma``);
* early stopping at the first validation rise (``patience: 1``) within 12
  epochs (``num_epochs``), set in the config.

Our corpora hold many more samples per epoch than their 50-minute record, so
``epoch_batches`` / ``val_batches`` cap an epoch at about their length; the
schedule then carries over unchanged. ``loss_fn`` is unused: the paper fixes
the RMSE.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator

import torch
from neural_surrogate_baselines.losses import masked_rmse
from neural_surrogates.training.base import BaseTraining


class _Batches:
    """At most ``n`` batches of ``loader`` per pass.

    ``cycle`` continues where the last pass stopped (restarting the loader when
    it runs out), so capped training epochs still see the whole dataset;
    otherwise every pass takes the loader's first ``n`` batches, a fixed
    validation subset.
    """

    def __init__(self, loader: Any, n: int, cycle: bool) -> None:
        self.loader, self.n, self.cycle = loader, int(n), cycle
        self.dataset = loader.dataset
        self._it: Iterator | None = None

    def __len__(self) -> int:
        return min(self.n, len(self.loader))

    def __iter__(self) -> Iterator:
        if not self.cycle:
            yield from itertools.islice(self.loader, self.n)
            return
        for _ in range(self.n):
            batch = None
            if self._it is not None:
                batch = next(self._it, None)
            if batch is None:
                it = iter(self.loader)
                self._it = it
                batch = next(it)
            yield batch


class LocalFNOTrainer(BaseTraining):
    def __init__(
        self,
        *args: Any,
        lr_step_gamma: float | None = 0.5,
        epoch_batches: int | None = None,
        val_batches: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.lr_step_gamma = lr_step_gamma
        super().__init__(*args, **kwargs)
        if epoch_batches is not None:
            self.train_loader = _Batches(self.train_loader, epoch_batches, cycle=True)  # type: ignore[has-type]
        if val_batches is not None:
            self.val_loader = _Batches(self.val_loader, val_batches, cycle=False)  # type: ignore[has-type]

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.lr_step_gamma is None:
            return super()._build_lr_scheduler()
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.lr_step_gamma
        )

    def _final_loss(
        self,
        state: torch.Tensor,
        state_next: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        model = self._eager_model
        feat = None
        if self._geom_features is not None:
            feat = self._geom_features.expand(
                state.shape[0], *self._geom_features.shape
            )
        pred = model.forward_patches(
            state, params[:, -1, :], geometry, geom_features=feat
        )
        grid = model.patch_grid(*state.shape[-2:])
        target = grid.extract(state_next)
        valid = grid.valid(state.shape[-3], device=pred.device).repeat(
            state.shape[0], 1, 1, 1, 1
        )
        if self.mask_loss:
            valid = valid & grid.extract(geometry.unsqueeze(1)).bool()
        return masked_rmse(pred, target, valid)
