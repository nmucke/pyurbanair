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
  reconstruction. In **geometry-branch** mode (``TadpoleAE(geometry_branch=...)``)
  there are no geometry channels at all -- geometry conditions the
  encoder/decoder instead of being reconstructed -- so that term is identically
  zero and the objective degenerates to masked state recon + KL, no config
  change needed;
* ``kl_weight`` (β) defaults tiny (latent-diffusion convention); ``kl_weight=0``
  with ``latent_type="mode"`` degrades gracefully to a plain deterministic
  autoencoder -- the "AE core" of the staged scope, one config knob away.

The per-term breakdown is exposed via ``_aux_terms`` so it lands in
``metrics.csv`` (the same mechanism the DD patch trainer uses).

Optional adversarial (GAN) extension
------------------------------------
Pass a ``discriminator`` (+ its own ``disc_optimizer``) to add the VQGAN/latent-
diffusion adversarial term on top of the objective above -- the fix for the
characteristic blurriness of a pure MSE+KL autoencoder (MSE is minimised by the
conditional *mean*, so it averages away exactly the small-scale structure an
urban flow lives on). ``discriminator=None`` (the default) is a strict no-op:
same loss, same ``_aux_terms`` keys, same checkpoint payload, no extra forward.

When it is on, each training step is the standard 1:1 alternating update:

1. the generator (AE) loss gains ``coeff * hinge_g_loss(D(fake))``, where
   ``fake`` is the reconstruction and ``real`` the target, each concatenated with
   the **true** geometry block (never the reconstructed one -- the critic must
   judge the flow, not the AE's guess at where the buildings are);
2. after the AE's optimizer step, :meth:`_after_optimizer_step` updates the
   discriminator on the *detached* pair stashed by (1), so the D step costs no
   second autoencoder forward.

``coeff = ramp * adv_weight * lambda`` (see :meth:`_adversarial_generator_term`).
The warm-up is counted in **global optimizer steps**, not epochs, following the
paper (Appendix C.2: adversarial feedback is off for the first ~1000 iterations
and then ramped over a further ~1000) -- and because our snapshot corpora differ
in size by more than an order of magnitude, so "epoch" is not a portable unit
here. ``lambda`` is the gradient-balancing adaptive weight of Esser et al. (2021)
and ``adv_weight`` is the *maximum* adversarial scale (paper: 1e-4).

**Deliberate deviation from upstream** (documented here and in
``tadpole_discriminator.py`` so nobody "fixes" it back): upstream feeds its
critic the same *folded, single-channel* crops the encoder sees, so its D never
sees state and geometry together. We feed the **unfolded, multi-channel** field
with the true geometry block concatenated, because the requirement here is that
the critic judge the reconstruction in the presence of its obstacles. Everything
else (hinge losses, adaptive lambda, step-based warm-up, a separate AdamW for the
critic, 1:1 D:G steps, no perceptual/R1/EMA/spectral-norm) follows the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from neural_surrogates.training.base import BaseTraining


class AutoencoderTrainer(BaseTraining):
    def __init__(
        self,
        *args,
        kl_weight: float = 1.0e-6,
        geometry_recon_weight: float = 0.1,
        discriminator: torch.nn.Module | None = None,
        disc_optimizer: torch.optim.Optimizer | None = None,
        adv_weight: float = 1.0e-4,
        adv_start_step: int = 1000,
        adv_ramp_steps: int = 1000,
        adaptive_adv_weight: bool = True,
        disc_recon_threshold: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kl_weight = float(kl_weight)
        self.geometry_recon_weight = float(geometry_recon_weight)
        self._n_state_channels = int(self._eager_model.n_state_channels)
        self._encode_geometry = bool(
            getattr(self._eager_model, "encode_geometry", False)
        )

        # -- optional adversarial extension (everything below is inert when
        #    ``discriminator is None``) -------------------------------------- #
        if (discriminator is None) != (disc_optimizer is None):
            raise ValueError(
                "the adversarial extension needs both `discriminator` and "
                "`disc_optimizer` or neither; got discriminator="
                f"{type(discriminator).__name__ if discriminator else None}, "
                f"disc_optimizer="
                f"{type(disc_optimizer).__name__ if disc_optimizer else None}."
            )
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.adv_weight = float(adv_weight)
        self.adv_start_step = int(adv_start_step)
        self.adv_ramp_steps = int(adv_ramp_steps)
        self.adaptive_adv_weight = bool(adaptive_adv_weight)
        self.disc_recon_threshold = (
            None if disc_recon_threshold is None else float(disc_recon_threshold)
        )
        # Global optimizer-step counter driving the adversarial warm-up/ramp.
        # Incremented once per AE optimizer step in ``_after_optimizer_step`` and
        # persisted in the checkpoint, so a resumed run continues the schedule
        # instead of restarting the warm-up.
        self._global_step = 0
        # Detached (real, fake) pair + the batch's recon loss, handed from the
        # generator step to the discriminator step. ``None`` whenever the
        # adversarial path did not run for this batch.
        self._adv_real: torch.Tensor | None = None
        self._adv_fake: torch.Tensor | None = None
        self._adv_recon: torch.Tensor | None = None
        self._adv_last_layer: torch.nn.Parameter | None = None
        self._adv_geom_source = "none"
        self.disc_scaler: torch.amp.GradScaler | None = None
        if self.discriminator is not None:
            self._setup_adversarial()

    # ----------------------------------------------------------------- setup #
    def _setup_adversarial(self) -> None:
        """Place the critic on the trainer's device, give it its own GradScaler,
        and settle *once* how the geometry channels it expects are assembled.

        The channel contract is validated here rather than at the first batch so
        a mismatched discriminator/architecture pairing fails at construction --
        before a long run has burned an epoch -- with a message naming both
        sides. ``resolve_last_decoder_layer`` is likewise resolved once: the
        adaptive weight needs the decoder's output-layer parameter every step,
        and parameters are updated in place (never replaced), so the handle stays
        valid for the whole run.
        """
        from neural_surrogates.architectures.tadpole_discriminator import (
            resolve_last_decoder_layer,
        )

        assert self.discriminator is not None
        self.discriminator = self.discriminator.to(self.device)
        if self.channels_last:
            self.discriminator = self.discriminator.to(  # type: ignore[call-overload]
                memory_format=torch.channels_last_3d
            )
        # Mirror BaseTraining's scaler: only fp16-on-CUDA autocast needs loss
        # scaling; bf16 (the default) has fp32's exponent range, so the scaler is
        # constructed disabled and every scaler call below is a pass-through.
        self.disc_scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.amp
            and self.device.type == "cuda"
            and self.amp_dtype is torch.float16,
        )
        self._adv_last_layer = resolve_last_decoder_layer(self._eager_model)

        # How many channels does the critic want on top of the state block, and
        # where do they come from? ``target``/``recon`` already carry the AE's
        # geometry block when it encodes geometry; when it does not -- a
        # geometry-blind AE, or one conditioned through a geometry branch, both
        # of which have ``n_geometry_channels == 0`` -- a critic that still wants
        # a mask gets it from the raw ``geometry`` argument (``"mask"``).
        expected = int(getattr(self.discriminator, "n_input_channels"))
        available = int(getattr(self._eager_model, "n_geometry_channels", 0))
        extra = expected - self._n_state_channels
        if extra == available:
            self._adv_geom_source = "target" if available else "none"
        elif extra == 1 and available == 0:
            self._adv_geom_source = "mask"
        else:
            raise ValueError(
                f"discriminator expects n_input_channels={expected}, but the "
                f"autoencoder produces {self._n_state_channels} state channel(s) "
                f"+ {available} geometry channel(s). Build the discriminator with "
                "the same n_state_channels / encode_geometry / sdf_features as "
                "the architecture (scripts/neural_surrogate/pretrain_autoencoder.py "
                "injects all three)."
            )

    # ---------------------------------------------------------------- batches #
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
        guarantees the restore even if a validation batch raises.

        Note the adversarial term is *never* part of the validation loss (see
        :meth:`_adversarial_engaged`): it would make the val curve jump at the
        warm-up boundary and stop being a comparable yardstick for best-weight
        selection, and it cannot be differentiated under ``no_grad`` anyway."""
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
        if self._adversarial_engaged():
            total = total + self._adversarial_generator_term(
                recon, target, mask, state_loss
            )
        return total

    # ----------------------------------------------------------- adversarial #
    def _adversarial_engaged(self) -> bool:
        """Is the adversarial path in play for *this* ``_loss`` call?

        Only while genuinely training: ``_forward`` is also called from
        ``_validate`` under ``model.eval()`` + ``torch.no_grad()``, where the
        adaptive weight's ``torch.autograd.grad`` would raise and the extra term
        would break the val curve's comparability across the warm-up boundary.
        Deliberately *not* gated on the warm-up step, so the ``adv`` / ``adv_w`` /
        ``d`` columns exist from the first row of ``metrics.csv`` onwards (a
        column appearing mid-run would break the appended CSV's header)."""
        return (
            self.discriminator is not None
            and self.model.training
            and torch.is_grad_enabled()
        )

    def _adv_ramp(self) -> float:
        """Warm-up/ramp factor in ``[0, 1]`` at the current global step.

        Zero before ``adv_start_step`` (the AE learns to reconstruct before a
        critic starts pushing it), then a linear ramp to 1.0 over
        ``adv_ramp_steps`` further steps. ``adv_ramp_steps=0`` switches the term
        on at full strength at ``adv_start_step``."""
        if self._global_step < self.adv_start_step:
            return 0.0
        if self.adv_ramp_steps <= 0:
            return 1.0
        progress = (self._global_step - self.adv_start_step) / self.adv_ramp_steps
        return min(1.0, progress)

    def _adv_geometry_block(
        self, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor | None:
        """The **true** geometry channels to condition the critic on, or ``None``
        when it wants none. Sourced per ``_adv_geom_source``, settled in
        :meth:`_setup_adversarial`: ``"target"`` slices the AE's own working-space
        geometry block off the target (mask + SDF channels, already in the right
        order and dtype); ``"mask"`` covers a geometry-blind AE paired with a
        geometry-aware critic, where the only available channel is the fluid mask
        itself. Both the real and the fake pass get this same block -- never the
        *reconstructed* geometry, which would let the AE hide flow errors behind a
        distorted obstacle field."""
        if self._adv_geom_source == "none":
            return None
        if self._adv_geom_source == "mask":
            return mask
        return target[:, self._n_state_channels :]

    def _adversarial_generator_term(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        state_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Generator-side adversarial loss, and the stash for the D step.

        Returns ``coeff * hinge_g_loss(D(fake))`` with

            coeff = ramp * adv_weight * lambda

        -- ``ramp`` the step-based warm-up, ``adv_weight`` the *maximum*
        adversarial scale (paper: 1e-4) and ``lambda`` the gradient-balancing
        adaptive weight of Esser et al. (2021), which returns values in ``[0, 1]``
        (its own ``max_weight``), so the effective coefficient lives in
        ``[0, adv_weight]``. With ``adaptive_adv_weight=False`` lambda is fixed at
        1.0 and the coefficient is just ``ramp * adv_weight``.

        During the warm-up (``ramp == 0``) the critic is not run on the generator
        side at all -- the term would be multiplied by zero -- but the detached
        real/fake pair is still stashed so the discriminator itself can start
        training the moment ``adv_start_step`` is reached.

        AMP note. ``adaptive_adv_weight`` calls ``torch.autograd.grad`` on the
        **unscaled** losses, straight out of the autocast region. That is correct
        for this repo's default ``amp_dtype: bfloat16``, where ``GradScaler`` is
        constructed disabled (bf16 has fp32's exponent range) -- the ratio of two
        equally-unscaled gradient norms is the ratio we want, and the loss the
        caller later backprops through the scaler is untouched by these
        ``retain_graph=True`` probes. Under **fp16** scaling the probe gradients
        would be computed unscaled and could underflow to zero, making lambda
        collapse to its ``eps`` fallback; making that exact would mean scaling the
        two losses by ``scaler.get_scale()`` before the probe (the common factor
        cancels in the ratio) and skipping the step when the scaler later finds
        infs. We deliberately do not build that machinery: bf16 is the default and
        fp16 is a pre-Ampere fallback. Set ``adaptive_adv_weight: false`` if you
        must run the adversarial extension under fp16 scaling.
        """
        from neural_surrogates.architectures.tadpole_discriminator import (
            adaptive_adv_weight,
            hinge_g_loss,
        )

        assert self.discriminator is not None
        c = self._n_state_channels
        geom_block = self._adv_geometry_block(target, mask)
        state_fake, state_real = recon[:, :c], target[:, :c]
        if geom_block is None:
            fake, real = state_fake, state_real
        else:
            fake = torch.cat([state_fake, geom_block], dim=1)
            real = torch.cat([state_real, geom_block], dim=1)

        ramp = self._adv_ramp()
        if ramp > 0.0:
            g_loss = hinge_g_loss(self.discriminator(fake))
            lam: float | torch.Tensor = 1.0
            if self.adaptive_adv_weight and self._adv_last_layer is not None:
                lam = adaptive_adv_weight(state_loss, g_loss, self._adv_last_layer)
            coeff = ramp * self.adv_weight * lam
            term = coeff * g_loss
        else:
            g_loss = torch.zeros((), device=recon.device, dtype=recon.dtype)
            coeff = 0.0
            term = g_loss

        # Hand the D step a detached pair so it needs no second AE forward, and
        # the recon value so the optional `disc_recon_threshold` gate can read it.
        self._adv_real = real.detach()
        self._adv_fake = fake.detach()
        self._adv_recon = state_loss.detach()
        assert self._aux_terms is not None
        self._aux_terms["adv"] = g_loss.detach()
        # Effective coefficient actually applied to `adv` this step (ramp x
        # adv_weight x lambda); `d` is filled in by the discriminator step below
        # and stays 0 while the critic is not being updated.
        self._aux_terms["adv_w"] = torch.as_tensor(
            coeff, device=recon.device, dtype=torch.float32
        )
        self._aux_terms["d"] = torch.zeros((), device=recon.device)
        return term

    def _after_optimizer_step(self, batch: dict[str, torch.Tensor]) -> None:
        """Discriminator update: one hinge step on the stashed (real, fake) pair.

        Runs immediately after the AE's optimizer step (1:1 D:G ratio). The
        ``zero_grad`` matters beyond hygiene: the generator's backward pass ran
        *through* the critic, so its parameters carry generator gradients that
        must not be stepped on. Everything is skipped before ``adv_start_step``
        and whenever the optional ``disc_recon_threshold`` gate says the AE is not
        yet reconstructing well enough to be worth criticising."""
        if self.discriminator is None:
            return
        # Count every AE optimizer step, including the ones the critic sits out --
        # this is the clock the warm-up/ramp schedule is written against.
        self._global_step += 1
        real, fake, recon = self._adv_real, self._adv_fake, self._adv_recon
        self._adv_real = self._adv_fake = self._adv_recon = None
        if real is None or fake is None:
            return
        if self._global_step <= self.adv_start_step:
            return
        # Optional upstream-style gate: don't let the critic overpower an
        # untrained AE. `.item()` costs a host sync, hence opt-in (default None).
        if self.disc_recon_threshold is not None and (
            recon is None or recon.item() >= self.disc_recon_threshold
        ):
            return

        from neural_surrogates.architectures.tadpole_discriminator import hinge_d_loss

        assert self.disc_optimizer is not None and self.disc_scaler is not None
        self.disc_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            d_loss = hinge_d_loss(self.discriminator(real), self.discriminator(fake))
        self.disc_scaler.scale(d_loss).backward()
        if self.grad_clip_norm is not None:
            # No-op unscale when the scaler is disabled; scaler.step detects the
            # explicit unscale and does not repeat it (as in the main step).
            self.disc_scaler.unscale_(self.disc_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.grad_clip_norm
            )
        self.disc_scaler.step(self.disc_optimizer)
        self.disc_scaler.update()
        if self._aux_terms is not None:
            self._aux_terms["d"] = d_loss.detach()

    # ------------------------------------------------------ checkpoint/resume #
    def _adversarial_checkpoint_state(self) -> dict:
        assert self.discriminator is not None
        assert self.disc_optimizer is not None and self.disc_scaler is not None
        return {
            "discriminator": self.discriminator.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "disc_scaler": self.disc_scaler.state_dict(),
            "global_step": self._global_step,
        }

    def _load_adversarial_state(self, ckpt: dict) -> None:
        """Restore the critic half of a checkpoint, tolerating its absence.

        A checkpoint written before this extension (or by a non-adversarial run)
        simply carries none of these keys, and the run continues with a freshly
        initialised critic and a warm-up starting from step 0."""
        if self.discriminator is None or "discriminator" not in ckpt:
            return
        assert self.disc_optimizer is not None and self.disc_scaler is not None
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        self.disc_scaler.load_state_dict(ckpt["disc_scaler"])
        self._global_step = int(ckpt.get("global_step", 0))

    def _save_checkpoint(self, path: Path, *args: Any, **kwargs: Any) -> None:
        """Append the critic's state to the base checkpoint.

        Done as a read-back-and-rewrite rather than a hook inside
        ``BaseTraining._save_checkpoint`` so the shared trainer stays untouched:
        the extra round trip only ever happens on the opt-in adversarial path,
        once every ``checkpoint_every`` epochs. The rewrite goes to a sibling
        temp file and is renamed over the original -- ``torch.save`` truncates as
        it writes, so writing straight back into the file we just read would
        leave a corrupt checkpoint if it failed part-way (a full disk is the
        realistic case); ``Path.replace`` is atomic within the directory."""
        super()._save_checkpoint(path, *args, **kwargs)
        if self.discriminator is None:
            return
        ckpt = torch.load(path, map_location="cpu")
        ckpt.update(self._adversarial_checkpoint_state())
        tmp = path.with_name(path.name + ".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    def fit(self) -> dict:
        """Train, restoring the critic first when resuming.

        ``BaseTraining.fit()`` restores the model/optimizer/scheduler/scaler half
        of ``checkpoint.pt`` itself; the critic half is this subclass's, so read
        it from the same file under the same guard rather than threading another
        hook through the base class."""
        if self.discriminator is not None and self.resume:
            ckpt_path = self._checkpoint_path()
            if ckpt_path is not None and ckpt_path.exists():
                self._load_adversarial_state(
                    torch.load(ckpt_path, map_location=self.device)
                )
        return super().fit()
