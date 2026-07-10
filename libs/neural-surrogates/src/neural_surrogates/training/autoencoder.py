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
        shared geometry once as ``(1, *grid)``; that upload is cached on the
        device (identity fast path + ``torch.equal`` content revalidation,
        mirroring :meth:`BaseTraining._prepare_batch`) so a same-geometry stream
        does not re-upload the mask + SDF features each step -- and expanded to
        ``(B, *grid)`` (a view) so the model sees one geometry per member with
        no broadcast ambiguity. Random-crop batches arrive per-sample as
        ``(B, *grid)`` (leading dim != 1): they take the direct-upload branch and
        are never cached, so a stale crop can never be served.
        """
        to_kwargs: dict = {"non_blocking": True}
        if self.channels_last:
            to_kwargs["memory_format"] = torch.channels_last_3d
        state = batch["state"].to(self.device, **to_kwargs)
        b = state.shape[0]
        geom_batch = batch["geometry"]
        feat_batch = batch.get("geom_features")

        if geom_batch.shape[0] != 1:
            # Per-sample geometry (random-crop batch): upload directly, no cache
            # -- the crops differ across the batch and across steps.
            geometry = geom_batch.to(self.device, non_blocking=True)
            features = (
                feat_batch.to(self.device, non_blocking=True)
                if feat_batch is not None
                else None
            )
            return state, geometry, features

        # Shared geometry shipped once as (1, *grid): device-side cache keyed on
        # the host tensor. Identity (``is``) hits for workerless loaders; the
        # content compare keeps a same-geometry stream from re-uploading each
        # step; a genuinely different geometry refreshes the cache.
        geom_host = geom_batch[0]
        cached = self._geometry_host
        stale = cached is None or (
            cached is not geom_host
            and not (cached.shape == geom_host.shape and torch.equal(cached, geom_host))
        )
        if stale:
            self._geometry_host = geom_host
            self._geometry = geom_host.to(self.device)
            self._geom_features = (
                feat_batch[0].to(self.device) if feat_batch is not None else None
            )
        assert self._geometry is not None  # set on the first (always-stale) batch
        geometry = self._geometry.expand(b, *self._geometry.shape)
        features = None
        if self._geom_features is not None:
            features = self._geom_features.expand(b, *self._geom_features.shape)
        return state, geometry, features

    def _validate(self) -> float:
        """Score validation with a *deterministic* latent.

        Under ``latent_type="sample"`` the (V)AE draws a fresh latent each
        forward, so the validation loss -- and thus best-weights selection and
        early stopping (``patience``) -- would ride on sampling noise. Switch the
        wrapped autoencoder to ``"mode"`` (the latent mean) for the duration of
        validation and restore the training setting afterwards. ``try/finally``
        guarantees the restore even if a validation batch raises."""
        ae = getattr(self._eager_model, "ae", None)
        if ae is None or not hasattr(ae, "latent_type"):
            return float(super()._validate())
        saved = ae.latent_type
        ae.latent_type = "mode"
        try:
            return float(super()._validate())
        finally:
            ae.latent_type = saved

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
