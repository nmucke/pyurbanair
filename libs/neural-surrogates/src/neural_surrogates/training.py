"""Generic single-loop trainer for the neural-surrogate baselines."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import tqdm


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
        patience: int | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.patience = patience
        self.weights_path = Path(weights_path) if weights_path is not None else None

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
        for batch in tqdm.tqdm(self.train_loader):
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
        best_val = float("inf")
        epochs_since_improvement = 0
        if self.weights_path is not None:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate()
            print(
                f"epoch {epoch + 1}/{self.num_epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                epochs_since_improvement = 0
                if self.weights_path is not None:
                    torch.save(self.model.state_dict(), self.weights_path)
                    print(f"  saved new best weights to {self.weights_path}")
            else:
                epochs_since_improvement += 1
                if self.patience is not None and epochs_since_improvement >= self.patience:
                    print(
                        f"early stopping: no val improvement for {self.patience} epochs "
                        f"(best val={best_val:.6f})"
                    )
                    break
        if self.weights_path is not None and self.weights_path.exists():
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
