"""Shared training scaffolding for the neural-surrogate trainers.

:class:`BaseTraining` holds every architecture-agnostic capability -- device &
memory-format handling, ``torch.compile``, mixed-precision autocast + grad
scaling, the warmup+cosine LR schedule, the pushforward-rollout curriculum,
gradient clipping, best-weight saving, checkpoint/resume and per-epoch metrics
logging -- and drives the train / validate / early-stop loop.

Subclasses implement a single hook, :meth:`_final_loss`, that turns the final
rollout step into a scalar loss. That one method is the *only* place the generic
full-grid trainer (:class:`neural_surrogates.training.Trainer`) and the
domain-decomposed patch trainer (:class:`neural_surrogates.training.PatchTrainer`)
differ: the former applies a masked element-wise ``loss_fn(pred, target)``, the
latter the four-term Eq (9) loss over the model's per-patch intermediates.
Everything else -- including the pushforward rollout itself -- is shared.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import tqdm


class BaseTraining:
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
        amp_dtype: str = "bfloat16",
        compile_model: bool = False,
        compile_dynamic: bool | None = None,
        channels_last: bool = False,
        pushforward_epochs_per_step: int | None = None,
        pushforward_start_steps: int = 1,
        lr_warmup_epochs: int | None = None,
        lr_warmup_start: float = 0.0,
        lr_min: float = 0.0,
        mask_loss: bool = True,
        grad_clip_norm: float | None = None,
        grad_unroll_steps: int = 2,
        resume: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        # channels-last memory layout lets cuDNN pick faster 3D-conv kernels
        # under autocast; a no-op for correctness either way.
        self.channels_last = channels_last
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last_3d)
        # Keep a handle on the eager module: torch.compile wraps the model and
        # prefixes its state-dict keys with `_orig_mod.`, so saving/loading
        # must go through the unwrapped module (parameters are shared).
        self._eager_model = self.model
        if compile_model:
            # Inductor's CUDA backend generates triton kernels; without a
            # working triton, compilation raises mid-epoch. Fall back to
            # eager instead of crashing the run.
            from torch.utils._triton import has_triton

            if self.device.type == "cuda" and not has_triton():
                print(
                    "compile_model=true but no working triton installation for "
                    "CUDA inductor; running eager. Install triton in this env "
                    "to enable torch.compile."
                )
            else:
                # `dynamic` controls whether inductor specialises on concrete
                # spatial sizes. An explicit `compile_dynamic` wins; otherwise
                # the model may request a setting via a `compile_dynamic`
                # attribute (P3D sets False: its shape-dependent internal padding
                # makes dynamo introduce symbolic spatial dims even on the first
                # trace, and inductor then mis-infers the backward strides of its
                # reshape/PixelShuffle ops -- an `assert_size_stride` crash in the
                # compiled backward. Forcing static specialisation avoids it).
                dynamic = compile_dynamic
                if dynamic is None:
                    dynamic = getattr(self._eager_model, "compile_dynamic", None)
                # The pushforward loop calls the model under both no_grad and
                # grad (plus separate autocast contexts), so a single forward
                # legitimately needs several compiled graph variants. The
                # default dynamo recompile limit (8) is tight for that and makes
                # a frame silently fall back to eager; give it headroom.
                import torch._dynamo as _dynamo

                for _attr in ("recompile_limit", "cache_size_limit"):
                    if hasattr(_dynamo.config, _attr):
                        setattr(_dynamo.config, _attr, max(getattr(_dynamo.config, _attr), 64))
                self.model = torch.compile(self.model, dynamic=dynamic)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = int(num_epochs)
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
        # Mixed precision: autocast runs the forward pass in bf16 (default) or
        # fp16. bf16 has fp32's exponent range, so no gradient under/overflow
        # and no loss scaling; the GradScaler is only enabled when we actually
        # autocast fp16 on CUDA (pre-Ampere GPUs without bf16 support).
        self.amp = amp
        resolved_dtype = getattr(torch, amp_dtype)
        if resolved_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"amp_dtype must be bfloat16 or float16, got {amp_dtype}")
        if (
            amp
            and resolved_dtype is torch.bfloat16
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            print("bf16 autocast not supported on this GPU; falling back to fp16")
            resolved_dtype = torch.float16
        self.amp_dtype = resolved_dtype
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=amp
            and self.device.type == "cuda"
            and resolved_dtype is torch.float16,
        )
        # The geometry mask is identical for every sample (see
        # TransitionDataset), so it is moved to the device once, lazily from
        # the first batch, instead of shipping B copies host->GPU every step.
        self._geometry: torch.Tensor | None = None
        self._fluid_mask: torch.Tensor | None = None
        # mask_loss restricts the loss to fluid cells: uDALES targets carry
        # junk values inside obstacles, which would otherwise be penalised
        # against the model's (masked) zero output there.
        self.mask_loss = mask_loss
        self.grad_clip_norm = grad_clip_norm
        # Number of final unroll steps that carry gradients. 1 reproduces the
        # pure pushforward trick (gradient through the last step only);
        # larger values backprop through the last N model calls, which
        # improves long-rollout error at the cost of holding N steps'
        # activations.
        self.grad_unroll_steps = max(1, int(grad_unroll_steps))
        self.resume = resume
        # Per-batch auxiliary scalars a subclass may expose for logging (e.g.
        # the DD loss term breakdown). ``None`` until/unless a subclass sets it
        # inside ``_final_loss``; the generic full-grid trainer leaves it unset.
        self._aux_terms: dict[str, torch.Tensor] | None = None

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

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp
        )

    def _prepare_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move one transition batch to the device and return
        ``(state, state_next, params, geometry)``. The geometry mask is cached
        on the device on the first batch (it is identical for every sample) and
        broadcast to the batch via ``expand`` (a view, not B copies)."""
        to_kwargs: dict = {"non_blocking": True}
        if self.channels_last:
            to_kwargs["memory_format"] = torch.channels_last_3d
        state = batch["state_n"].to(self.device, **to_kwargs)
        state_next = batch["state_next"].to(self.device, non_blocking=True)
        params = batch["params_n"].to(self.device, non_blocking=True)
        if self._geometry is None:
            self._geometry = batch["geometry"][0].to(self.device)
            self._fluid_mask = self._geometry.bool()
        geometry = self._geometry.expand(state.shape[0], *self._geometry.shape)
        return state, state_next, params, geometry

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """One pushforward rollout + loss on a transition batch.

        ``params`` is ``(B, K, P)``. The first ``K - g`` steps run under
        ``no_grad`` so the model consumes its own predictions without backprop
        through the unroll; the final ``g = grad_unroll_steps`` calls carry
        gradients (``g = 1`` is the pure pushforward trick). The very last step
        and its loss are delegated to the subclass hook :meth:`_final_loss`.

        Two separate autocast contexts on purpose: autocast caches its weight
        casts and clears the cache on exit. Running the no_grad pushforward in
        the *same* context would cache casts with ``requires_grad=False`` and
        poison the final, gradient-bearing forwards.
        """
        state, state_next, params, geometry = self._prepare_batch(batch)
        K = params.shape[1]
        g = min(self.grad_unroll_steps, K)
        with torch.no_grad():
            with self._autocast():
                for i in range(K - g):
                    state = self.model(state, params[:, i, :], geometry)
        with self._autocast():
            for i in range(K - g, K - 1):
                state = self.model(state, params[:, i, :], geometry)
            return self._final_loss(state, state_next, params, geometry)

    def _final_loss(
        self,
        state: torch.Tensor,
        state_next: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Run the final rollout step (using ``params[:, -1]``) and return the
        scalar loss for the batch. Called inside the gradient-bearing autocast
        context. Subclasses define how the prediction becomes a loss; a subclass
        may set ``self._aux_terms`` to a dict of detached scalars for logging."""
        raise NotImplementedError

    # -- term logging (no-op unless a subclass populates ``_aux_terms``) ----- #
    def _accumulate_terms(self, sums: dict[str, float]) -> None:
        if self._aux_terms is None:
            return
        for k, v in self._aux_terms.items():
            sums[k] = sums.get(k, 0.0) + float(v)

    @staticmethod
    def _mean_terms(sums: dict[str, float], n: int) -> dict[str, float]:
        return {k: v / max(n, 1) for k, v in sums.items()}

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        n = 0
        term_sums: dict[str, float] = {}
        for batch in tqdm.tqdm(self.train_loader):
            loss = self._forward(batch)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                # No-op unscale when the scaler is disabled; scaler.step
                # detects the explicit unscale and does not repeat it.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
            n += 1
            self._accumulate_terms(term_sums)
        self._train_terms = self._mean_terms(term_sums, n)
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        term_sums: dict[str, float] = {}
        for batch in self.val_loader:
            total += self._forward(batch).item()
            n += 1
            self._accumulate_terms(term_sums)
        self._val_terms = self._mean_terms(term_sums, n)
        return total / max(n, 1)

    def _pushforward_steps_for_epoch(self, epoch: int) -> int:
        """Active rollout horizon at ``epoch`` (0-based) under the curriculum."""
        if self.pushforward_epochs_per_step is None:
            return self.pushforward_max_steps
        steps = self.pushforward_start_steps + epoch // self.pushforward_epochs_per_step
        return min(steps, self.pushforward_max_steps)

    def _checkpoint_path(self) -> Path | None:
        if self.weights_path is None:
            return None
        return self.weights_path.parent / "checkpoint.pt"

    def _metrics_path(self) -> Path | None:
        if self.weights_path is None:
            return None
        return self.weights_path.parent / "metrics.csv"

    def _log_metrics(self, path: Path, row: dict) -> None:
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_val: float,
        epochs_since_improvement: int,
        pushforward_steps: int,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self._eager_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "scaler": self.scaler.state_dict(),
                "best_val": best_val,
                "epochs_since_improvement": epochs_since_improvement,
                "pushforward_steps": pushforward_steps,
            },
            path,
        )

    def fit(self) -> dict:
        """Train + validate for ``num_epochs`` with the warmup/cosine schedule,
        pushforward curriculum, per-epoch metrics + checkpoint logging and
        early stopping. Saves the best (lowest val) weights and reloads them at
        the end. Returns a small history dict (``train`` / ``val`` loss lists)."""
        best_val = float("inf")
        epochs_since_improvement = 0
        current_steps = 0
        start_epoch = 0
        history: dict[str, list[float]] = {"train": [], "val": []}
        self._train_terms: dict[str, float] = {}
        self._val_terms: dict[str, float] = {}
        ckpt_path = self._checkpoint_path()
        metrics_path = self._metrics_path()
        if self.weights_path is not None:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume and ckpt_path is not None and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self._eager_model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.scheduler is not None and ckpt.get("scheduler") is not None:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            best_val = ckpt["best_val"]
            epochs_since_improvement = ckpt["epochs_since_improvement"]
            # Restore the curriculum horizon so the stage-change branch below
            # does not fire (it would reset the patience window).
            current_steps = ckpt["pushforward_steps"]
            if self._train_pushforward_dataset is not None and current_steps > 0:
                self._train_pushforward_dataset.set_pushforward_steps(current_steps)
            start_epoch = ckpt["epoch"] + 1
            print(
                f"resumed from {ckpt_path}: continuing at epoch {start_epoch + 1} "
                f"(best val={best_val:.6f}, horizon={current_steps})"
            )
        for epoch in range(start_epoch, self.num_epochs):
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
            epoch_start = time.monotonic()
            train_loss = self._train_epoch()
            val_loss = self._validate()
            epoch_seconds = time.monotonic() - epoch_start
            if self.scheduler is not None:
                self.scheduler.step()
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            msg = (
                f"epoch {epoch + 1}/{self.num_epochs}  "
                f"lr={lr:.2e}  train={train_loss:.6f}  val={val_loss:.6f}"
            )
            # Append the loss-term breakdown when a subclass exposes one (e.g.
            # the DD patch trainer's one-step / interface / divergence / coarse).
            if self._train_terms:
                terms = "  ".join(
                    f"{k}={self._train_terms[k]:.4f}" for k in self._train_terms
                )
                msg += f"  [{terms}]"
            print(msg)
            if val_loss < best_val:
                best_val = val_loss
                epochs_since_improvement = 0
                if self.weights_path is not None:
                    torch.save(self._eager_model.state_dict(), self.weights_path)
                    print(f"  saved new best weights to {self.weights_path}")
            else:
                epochs_since_improvement += 1
            if metrics_path is not None:
                row = {
                    "epoch": epoch + 1,
                    "pushforward_steps": steps,
                    "lr": f"{lr:.6e}",
                    "train_loss": f"{train_loss:.8f}",
                    "val_loss": f"{val_loss:.8f}",
                    "best_val": f"{best_val:.8f}",
                    "seconds": f"{epoch_seconds:.1f}",
                }
                for k, v in self._train_terms.items():
                    row[f"train_{k}"] = f"{v:.8f}"
                for k, v in self._val_terms.items():
                    row[f"val_{k}"] = f"{v:.8f}"
                self._log_metrics(metrics_path, row)
            if ckpt_path is not None:
                self._save_checkpoint(
                    ckpt_path, epoch, best_val, epochs_since_improvement, current_steps
                )
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
            self._eager_model.load_state_dict(
                torch.load(self.weights_path, map_location=self.device)
            )
        return history
