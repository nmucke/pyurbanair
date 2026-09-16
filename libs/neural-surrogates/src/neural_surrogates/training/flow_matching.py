"""Latent flow-matching training loop for ``TadpoleLatentGenerator`` (plan 07).

:class:`LatentFlowMatchingTrainer` reuses :class:`BaseTraining`'s machinery --
device/AMP, warmup+cosine LR, gradient clipping, early stopping,
checkpoint/resume, ``metrics.csv`` and best-weights saving -- and replaces the
rollout-shaped parts, exactly as :class:`AutoencoderTrainer` does. There is no
pushforward curriculum (a generator has no time axis), no ``_final_loss`` hook,
and none of the DFT machinery (skips, LoRA, dynamics): ``_forward`` computes one
conditional flow-matching draw on a
:class:`~neural_surrogates.datasets.snapshot_history.SnapshotHistoryDataset`
batch,

    v_pred, v_target = model(state, params_hist, geometry, geom_features)
    loss = mse(v_pred, v_target)            # fp32, all state-latent channels

where the model itself runs the frozen AE encoding in fp32 with autocast
disabled and only the velocity network sees the trainer's autocast context.

Why no voxel mask. ``mask_loss`` (fluid-cell masking, the convention of every
physical-space trainer here) is meaningless for this objective and is ignored:
the regression target lives on the AE's latent grid, one token per 16^3 block
of *padded* physical cells, so there is no per-voxel fluid indicator to apply.
The loss deliberately covers every state-latent channel at every latent
position, padded positions included -- latent attention and halo decoding
propagate errors at padded positions into the retained domain, so leaving them
untrained would not be harmless (plan 07 §2).

Deterministic validation. The flow objective is stochastic (a fresh ``z0`` and
``tau`` per sample), so a validation loss drawn with the training RNG would ride
on sampling noise and make best-weight selection / early stopping meaningless.
Every validation pass therefore (i) iterates a loader that must not shuffle --
checked at construction, since re-seeding the flow draws alone would not fix a
reshuffled example order -- and (ii) draws its noise and flow times from a
dedicated ``torch.Generator`` re-seeded from ``val_seed`` at the start of the
pass. Training keeps the global RNG (``generator=None``), so validation never
perturbs the training stream and vice versa.

Latent statistics. The generator's per-channel latent mean/std are model
buffers, so they travel in ``model.state_dict()`` -- ``weights.pt`` and
``checkpoint.pt`` both carry them and ``BaseTraining.fit`` restores them on
resume. They must be installed *before* training (the script computes them from
raw latents once and caches them), are never recomputed here, and a checkpoint
without them is refused rather than silently resumed with identity statistics.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import torch
from neural_surrogates.training.base import BaseTraining
from torch.utils.data import BatchSampler, DataLoader, RandomSampler


class LatentFlowMatchingTrainer(BaseTraining):
    """Flow-matching trainer for a frozen-AE latent generator.

    Parameters (beyond :class:`BaseTraining`'s)
    ------------------------------------------
    val_seed:
        Seed of the validation generator that draws ``z0`` / ``tau``; re-applied
        at the start of every validation pass so the val loss is a fixed
        yardstick across epochs and runs.
    """

    def __init__(self, *args: Any, val_seed: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.val_seed = int(val_seed)
        model = self._eager_model
        for attr in ("ae", "latent_stats_installed", "velocity_net"):
            if not hasattr(model, attr):
                raise TypeError(
                    "LatentFlowMatchingTrainer expects a TadpoleLatentGenerator-"
                    f"shaped model (missing attribute {attr!r}); got "
                    f"{type(model).__name__}"
                )

        # The AE is frozen by construction; a thawed parameter here means the
        # objective would start re-fitting the representation it generates in.
        ae: Any = model.ae
        thawed = [n for n, p in ae.named_parameters() if p.requires_grad]
        if thawed:
            raise ValueError(
                f"{len(thawed)} frozen-AE parameter(s) have requires_grad=True "
                f"(e.g. ae.{thawed[0]}); the generator's AE must stay frozen."
            )
        # ... and the optimizer must only ever see the trainable velocity-net
        # parameters (weight decay on frozen tensors would still move them).
        opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]
        ae_ids = {id(p) for p in ae.parameters()}
        if any(id(p) in ae_ids for p in opt_params):
            raise ValueError(
                "the optimizer holds frozen-AE parameters; build it from "
                "[p for p in model.parameters() if p.requires_grad]"
            )
        if any(not p.requires_grad for p in opt_params):
            raise ValueError("the optimizer holds parameters with requires_grad=False")

        for name, loader in (("train", self.train_loader), ("val", self.val_loader)):
            if len(loader) == 0:
                raise ValueError(
                    f"the {name} loader yields no batches (empty split, or a "
                    "TrajectoryBatchSampler with drop_last=True and fewer samples "
                    "per trajectory than its batch size)"
                )
        self._check_val_loader_is_deterministic(self.val_loader)

    # ---------------------------------------------------------------- checks #
    @staticmethod
    def _check_val_loader_is_deterministic(loader: DataLoader) -> None:
        """Refuse a validation loader whose example order changes per epoch.

        A stock ``DataLoader(shuffle=True)`` installs a ``RandomSampler``; a
        custom batch sampler (``TrajectoryBatchSampler``) carries its own
        ``shuffle`` flag. Either would make the fixed-seed flow draws land on
        different examples each epoch, so the val loss would no longer compare
        across epochs.
        """
        if isinstance(loader.sampler, RandomSampler):
            raise ValueError(
                "the validation loader shuffles (RandomSampler); build it with "
                "shuffle=False so validation scores a fixed example order"
            )
        batch_sampler = loader.batch_sampler
        if batch_sampler is not None and not isinstance(batch_sampler, BatchSampler):
            if bool(getattr(batch_sampler, "shuffle", False)):
                raise ValueError(
                    "the validation loader's batch sampler shuffles "
                    f"({type(batch_sampler).__name__}(shuffle=True)); build it "
                    "with shuffle=False"
                )

    def _require_latent_stats(self, when: str) -> None:
        if not bool(self._eager_model.latent_stats_installed):
            raise RuntimeError(
                f"latent normalisation statistics are not installed ({when}); "
                "compute_latent_normalization(...) must run before training and "
                "its buffers must be present in any checkpoint being resumed."
            )

    # --------------------------------------------------------------- batches #
    def prepared_batches(
        self, loader: Iterable[dict[str, torch.Tensor]]
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """``(state, geometry, geom_features)`` device tuples for every batch of
        ``loader`` -- the input ``TadpoleLatentGenerator.compute_latent_normalization``
        takes, produced by the very same :meth:`_prepare_snapshot_batch` the
        training step uses (same upload, cache and broadcast), so the statistics
        are estimated on exactly what the objective later encodes."""
        for batch in loader:
            yield self._prepare_snapshot_batch(batch)

    def _forward(
        self,
        batch: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One flow-matching draw + fp32 MSE on a snapshot-history batch.

        ``generator`` seeds the model's ``z0`` / ``tau`` draws; ``None`` (the
        training path) uses the global RNG. The loss is formed outside the
        autocast region on fp32 copies so a bf16 velocity prediction is
        compared to its fp32 target at full precision.
        """
        state, geometry, features = self._prepare_snapshot_batch(batch)
        params_hist = batch["params_hist"].to(self.device, non_blocking=True)
        with self._autocast():
            v_pred, v_target = self.model(
                state, params_hist, geometry, features, generator=generator
            )
        # No fluid mask: see the module docstring (latent grid, padded positions
        # included on purpose).
        loss: torch.Tensor = self.loss_fn(v_pred.float(), v_target.float())
        return loss

    def _train_epoch(self) -> float:
        self._require_latent_stats("before the training epoch")
        return super()._train_epoch()

    def _val_generator(self) -> torch.Generator:
        """A generator on the model's device (the model draws noise on
        ``z1.device``, and torch requires the generator to live there too),
        freshly seeded from ``val_seed``."""
        device = getattr(self._eager_model, "_device", self.device)
        return torch.Generator(device=device).manual_seed(self.val_seed)

    @torch.no_grad()
    def _validate(self) -> float:
        """Validation with fixed examples, order, noise and flow times.

        The loader's order is fixed by construction (see
        :meth:`_check_val_loader_is_deterministic`); the flow draws come from a
        generator re-seeded here, so two validation passes on the same weights
        return identical losses.
        """
        self._require_latent_stats("before validation")
        self.model.eval()
        generator = self._val_generator()
        total = torch.zeros((), device=self.device)
        n = 0
        for batch in self.val_loader:
            total = total + self._forward(batch, generator=generator).detach()
            n += 1
        self._val_terms = {}
        return (total / max(n, 1)).item()

    # ------------------------------------------------------ checkpoint/resume #
    def fit(self) -> dict:
        """Train, refusing to resume from a checkpoint without latent statistics.

        ``BaseTraining.fit`` restores ``model.state_dict()`` from ``checkpoint.pt``
        (buffers included, so the latent mean/std come back with it). A checkpoint
        whose ``latent_stats_installed`` flag is unset would silently train on
        identity statistics, so it is rejected up front; the flag is re-checked
        after ``fit`` so a stale in-memory state cannot slip through either.
        """
        ckpt_path = self._checkpoint_path()
        if self.resume and ckpt_path is not None and ckpt_path.exists():
            saved = torch.load(ckpt_path, map_location="cpu")["model"]
            flag = saved.get("latent_stats_installed")
            if flag is None or not bool(flag):
                raise RuntimeError(
                    f"checkpoint {ckpt_path} carries no installed latent "
                    "statistics (latent_stats_installed is missing/False); refusing "
                    "to resume with identity latent normalisation."
                )
        self._require_latent_stats("before fit()")
        history = super().fit()
        self._require_latent_stats("after fit()")
        return history
