"""Autoencoder (VAE) pre-training loop for ``TadpoleAE`` (plan 02).

:class:`AutoencoderTrainer` reuses :class:`BaseTraining`'s architecture-agnostic
machinery -- device/AMP/GradScaler, the warmup+cosine LR schedule, gradient
clipping, early stopping, checkpoint/resume, ``metrics.csv`` and best-weights
saving -- but replaces the rollout-shaped parts. There is no pushforward
curriculum (a snapshot AE has no time dimension) and no ``_final_loss`` rollout
hook; ``_forward`` is overridden to compute a single reconstruction + VAE loss on
a :class:`~neural_surrogates.datasets.snapshot.SnapshotDataset` batch:

    loss = masked_mse(state_recon, state)
         + geometry_recon_weight * mse(geometry_recon, geometry_block)
         + kl_weight * kl_elem.mean()

* the state reconstruction MSE is restricted to fluid cells (``mask_loss``), the
  same convention as :class:`~neural_surrogates.training.Trainer` -- obstacle
  cells carry no signal;
* the geometry/SDF channels (present only when the model encodes geometry) get
  their own small weight so the total loss stays dominated by state
  reconstruction;
* ``kl_weight`` (β) defaults tiny (latent-diffusion convention); ``kl_weight=0``
  with ``latent_type="mode"`` degrades gracefully to a plain deterministic
  autoencoder -- the "AE core" of the staged scope, one config knob away.

The per-term breakdown is exposed via ``_aux_terms`` so it lands in
``metrics.csv`` (the same mechanism the DD patch trainer uses).
"""

from __future__ import annotations

import torch
from neural_surrogates.training.base import BaseTraining


class AutoencoderTrainer(BaseTraining):
    def __init__(
        self,
        *args,
        kl_weight: float = 1.0e-6,
        geometry_recon_weight: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kl_weight = float(kl_weight)
        self.geometry_recon_weight = float(geometry_recon_weight)
        self._n_state_channels = int(self._eager_model.n_state_channels)
        self._encode_geometry = bool(
            getattr(self._eager_model, "encode_geometry", False)
        )

    def _prepare_ae_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Move a snapshot batch to the device and broadcast the (possibly
        once-shipped) geometry / SDF features to the batch size.

        :func:`~neural_surrogates.datasets.snapshot.snapshot_collate` ships a
        shared geometry once as ``(1, *grid)``; expand it to ``(B, *grid)`` (a
        view) so the model sees one geometry per member with no broadcast
        ambiguity. Random-crop batches already arrive per-sample as
        ``(B, *grid)`` and pass through unchanged.
        """
        to_kwargs: dict = {"non_blocking": True}
        if self.channels_last:
            to_kwargs["memory_format"] = torch.channels_last_3d
        state = batch["state"].to(self.device, **to_kwargs)
        b = state.shape[0]
        geometry = batch["geometry"].to(self.device, non_blocking=True)
        if geometry.shape[0] == 1 and b != 1:
            geometry = geometry.expand(b, *geometry.shape[1:])
        features = batch.get("geom_features")
        if features is not None:
            features = features.to(self.device, non_blocking=True)
            if features.shape[0] == 1 and b != 1:
                features = features.expand(b, *features.shape[1:])
        return state, geometry, features

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state, geometry, features = self._prepare_ae_batch(batch)
        with self._autocast():
            recon, target, kl_elem = self.model(
                state,
                geometry,
                features,
                return_kl_element=True,
                working_space=True,
            )
            return self._loss(recon, target, geometry, kl_elem)

    def _loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        geometry: torch.Tensor,
        kl_elem: torch.Tensor,
    ) -> torch.Tensor:
        """Masked state recon + weighted geometry recon + KL (working space)."""
        c = self._n_state_channels
        state_recon = recon[:, :c]
        state_target = target[:, :c]
        # (B, 1, *grid) fluid mask, broadcast over channels. Multiplicative
        # masking (not boolean indexing) handles a per-sample mask uniformly.
        mask = geometry.unsqueeze(1).to(dtype=recon.dtype)
        if self.mask_loss:
            sq = (state_recon - state_target) ** 2
            denom = mask.sum().clamp_min(1.0) * c
            state_loss = (sq * mask).sum() / denom
        else:
            state_loss = self.loss_fn(state_recon, state_target)

        geom_loss = torch.zeros((), device=recon.device, dtype=recon.dtype)
        if self._encode_geometry and recon.shape[1] > c:
            geom_loss = self.loss_fn(recon[:, c:], target[:, c:])

        kl_loss = kl_elem.mean()
        total = (
            state_loss
            + self.geometry_recon_weight * geom_loss
            + self.kl_weight * kl_loss
        )
        # Detached per-term breakdown for metrics.csv (synced once at epoch end).
        self._aux_terms = {
            "recon": state_loss.detach(),
            "geom": geom_loss.detach(),
            "kl": kl_loss.detach(),
        }
        return total
