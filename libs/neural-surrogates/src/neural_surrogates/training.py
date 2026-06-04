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
        amp: bool = False,
        pushforward_epochs_per_step: int | None = None,
        pushforward_start_steps: int = 1,
        lr_warmup_epochs: int | None = None,
        lr_warmup_start: float = 0.0,
        lr_min: float = 0.0,
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
        # Pushforward-horizon curriculum: start the rollout at
        # ``pushforward_start_steps`` and add one step every
        # ``pushforward_epochs_per_step`` epochs, up to the target horizon the
        # datasets were built with. ``None`` disables the ramp and keeps the
        # horizon fixed at that target (the original behavior).
        self.pushforward_epochs_per_step = pushforward_epochs_per_step
        self.pushforward_start_steps = int(pushforward_start_steps)
        # Only the *training* dataset's horizon ramps; the validation dataset
        # stays at the target horizon it was built with so val loss is a fixed,
        # comparable yardstick across the whole run (not a moving target that
        # would make early stopping / best-weight selection meaningless).
        train_ds = self.train_loader.dataset
        self._train_pushforward_dataset = (
            train_ds if hasattr(train_ds, "set_pushforward_steps") else None
        )
        self.pushforward_max_steps = (
            getattr(train_ds, "pushforward_steps", 1)
            if self._train_pushforward_dataset is not None
            else 1
        )
        # Warmup + cosine LR schedule: linearly ramp from ``lr_warmup_start`` up
        # to the optimizer's configured (peak) LR over ``lr_warmup_epochs``,
        # then cosine-anneal down to ``lr_min`` over the remaining epochs.
        # ``lr_warmup_epochs is None`` keeps the LR fixed (original behavior).
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_start = lr_warmup_start
        self.lr_min = lr_min
        self.scheduler = self._build_lr_scheduler()
        # Mixed precision: autocast runs the forward pass in fp16/bf16; the loss
        # scaler guards against fp16 gradient underflow and is a no-op unless we
        # are actually autocasting fp16 on CUDA.
        self.amp = amp
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=amp and self.device.type == "cuda"
        )

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.lr_warmup_epochs is None:
            return None
        peak_lr = self.optimizer.param_groups[0]["lr"]
        warmup = max(int(self.lr_warmup_epochs), 0)
        # CosineAnnealingLR needs at least one step to anneal over.
        cosine_epochs = max(self.num_epochs - warmup, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=self.lr_min
        )
        if warmup == 0:
            return cosine
        # LinearLR scales the peak LR by start_factor on epoch 0 and reaches 1.0
        # (the peak) at epoch ``warmup``; start_factor must be > 0.
        start_factor = self.lr_warmup_start / peak_lr if peak_lr > 0 else 1.0
        start_factor = min(max(start_factor, 1e-8), 1.0)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine],
            milestones=[warmup],
        )

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = batch["state_n"].to(self.device)
        state_next = batch["state_next"].to(self.device)
        params = batch["params_n"].to(self.device)
        geometry = batch["geometry"].to(self.device)
        # params: (B, K, P). The K-1 pushforward steps run under no_grad so
        # the model sees its own predictions without backprop through the
        # unroll; only the final step contributes gradients.
        K = params.shape[1]
        # Two separate autocast contexts on purpose: autocast caches its weight
        # casts and clears the cache on exit. Running the no_grad pushforward in
        # the *same* context would cache casts with requires_grad=False and
        # poison the final, gradient-bearing forward.
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                for i in range(K - 1):
                    state = self.model(state, params[:, i, :], geometry)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            pred = self.model(state, params[:, K - 1, :], geometry)
            return self.loss_fn(pred, state_next)

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for batch in tqdm.tqdm(self.train_loader):
            loss = self._forward(batch)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
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

    def _pushforward_steps_for_epoch(self, epoch: int) -> int:
        """Active rollout horizon at ``epoch`` (0-based) under the curriculum."""
        if self.pushforward_epochs_per_step is None:
            return self.pushforward_max_steps
        steps = self.pushforward_start_steps + epoch // self.pushforward_epochs_per_step
        return min(steps, self.pushforward_max_steps)

    def fit(self) -> None:
        best_val = float("inf")
        epochs_since_improvement = 0
        current_steps = 0
        if self.weights_path is not None:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.num_epochs):
            steps = self._pushforward_steps_for_epoch(epoch)
            if steps != current_steps:
                # Ramp only the training horizon; validation stays at the target
                # horizon so its loss remains comparable across all epochs and
                # best_val / patience need no per-stage reset.
                if self._train_pushforward_dataset is not None:
                    self._train_pushforward_dataset.set_pushforward_steps(steps)
                current_steps = steps
                # best_val stays global (validation is fixed at the target
                # horizon, so it's comparable across stages), but give each new
                # stage a fresh patience window so a plateau during the ramp
                # can't immediately trip early stopping at the final horizon.
                epochs_since_improvement = 0
                print(f"  train pushforward horizon -> {steps} step(s)")
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss = self._train_epoch()
            val_loss = self._validate()
            if self.scheduler is not None:
                self.scheduler.step()
            print(
                f"epoch {epoch + 1}/{self.num_epochs}  "
                f"lr={lr:.2e}  train={train_loss:.6f}  val={val_loss:.6f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                epochs_since_improvement = 0
                if self.weights_path is not None:
                    torch.save(self.model.state_dict(), self.weights_path)
                    print(f"  saved new best weights to {self.weights_path}")
            else:
                epochs_since_improvement += 1
                # Only stop once the curriculum has reached the target horizon;
                # an intermediate-stage plateau must not cut the ramp short.
                at_final_horizon = steps >= self.pushforward_max_steps
                if (
                    at_final_horizon
                    and self.patience is not None
                    and epochs_since_improvement >= self.patience
                ):
                    print(
                        f"early stopping: no val improvement for {self.patience} epochs "
                        f"(best val={best_val:.6f})"
                    )
                    break
        if self.weights_path is not None and self.weights_path.exists():
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
