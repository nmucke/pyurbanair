"""Generic single-loop trainer for the neural-surrogate baselines."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        num_epochs: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state_n = batch["state_n"].to(self.device)
        state_next = batch["state_next"].to(self.device)
        params_n = batch["params_n"].to(self.device)
        geometry = batch["geometry"].to(self.device)
        pred = self.model(state_n, params_n, geometry)
        return self.loss_fn(pred, state_next)

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for batch in self.train_loader:
            loss = self._forward(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for batch in self.val_loader:
            total += self._forward(batch).item()
            n += 1
        return total / max(n, 1)

    def fit(self) -> None:
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate()
            print(
                f"epoch {epoch + 1}/{self.num_epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}"
            )
