"""Tadpole-style autoencoder pre-training (plan 02).

Pre-trains a ``TadpoleAE`` on flow snapshots as a representation-learning (V)AE
-- no next-step objective. Mirrors ``train_neural_surrogate.py``: build the
``SnapshotDataset`` splits, compute + install per-channel normalization stats,
save the resolved config, then run ``AutoencoderTrainer.fit()``.

Artifacts land in ``model_weights/<model_name>/``: the full ``TadpoleAE``
``weights.pt`` (our standard, plus ``config.yaml`` / ``checkpoint.pt`` /
``metrics.csv`` as usual), and ``encoder.pt`` / ``decoder.pt`` via the
autoencoder's ``save_separate_weights`` -- the natural handoff format for plan 03
(AE -> time-stepper) and for HF-style reuse -- plus ``geometry_branch.pt`` when
the architecture runs in geometry-branch mode (the branch is a separate module;
its zero-init projections travel inside encoder.pt / decoder.pt).

The optional adversarial (GAN) extension is off unless the config carries a
``discriminator:`` block (it ships as ``null``); when present, this script builds
the critic with the *architecture's* own channel/geometry contract so the two can
never disagree, and hands it plus its own optimizer to the trainer.

    pixi run -e dev python scripts/neural_surrogate/pretrain_autoencoder.py \
        dataset.root_dir=training_data/pylbm_barcelona model_name=tadpole_ae_s
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from neural_surrogates.training.data_utils import build_loader, get_normalization_stats
from omegaconf import DictConfig, OmegaConf


def run(cfg: DictConfig) -> None:
    dtype = getattr(torch, cfg.dataset.dtype)

    train_ds = instantiate(cfg.dataset, split="train", dtype=dtype)
    val_ds = instantiate(cfg.dataset, split="val", dtype=dtype)
    # persistent_workers needs worker processes; force it off for workerless runs
    # (CPU smoke tests / debugging) so the DataLoader accepts the config.
    if int(cfg.dataloader.get("num_workers", 0)) == 0:
        cfg.dataloader.persistent_workers = False
    train_loader = build_loader(cfg, train_ds, train=True)
    val_loader = build_loader(cfg, val_ds, train=False)

    model = instantiate(
        cfg.architecture,
        n_state_channels=len(cfg.dataset.state_vars),
    ).to(dtype=dtype)

    # Cross-check the SDF-feature modes: a model whose stem encodes a specific set
    # of SDF channels -- or whose geometry branch is fed them -- must be paired
    # with a dataset that ships exactly those (and at the same clamp radius). Both
    # default to "none", so this is a no-op for standard runs. The check is on the
    # MODE alone, so it covers both consumers (folded stem / branch); which one is
    # in play is the architecture's business.
    model_mode = getattr(model, "sdf_feature_mode", "none")
    dataset_mode = getattr(train_ds, "sdf_feature_mode", "none")
    if model_mode != dataset_mode:
        raise ValueError(
            "SDF-feature mismatch: architecture.sdf_features="
            f"{model_mode!r} but dataset.sdf_features={dataset_mode!r}. "
            "They must select the same channels (set both to the same mode)."
        )
    if model_mode != "none" and float(train_ds.sdf_clamp_cells) != float(
        model.sdf_clamp_cells
    ):
        raise ValueError(
            "SDF clamp mismatch: architecture.sdf_clamp_cells="
            f"{model.sdf_clamp_cells} but dataset.sdf_clamp_cells="
            f"{train_ds.sdf_clamp_cells}; they must match."
        )

    # Warn on a wasteful crop/encoder pairing: a random crop SMALLER than the
    # encoder crop size is zero-padded up to it inside the model, so most of every
    # forward is padding and the reconstruction is dominated by padded cells before
    # the crop-back. Pick random_crop_size as a multiple of encoder_crop_size.
    crop = train_ds.random_crop_size
    global_spatial = getattr(model, "spatial_mode", "local") == "global"
    enc_crop = 16 if global_spatial else int(model.encoder_crop_size)
    padding_name = (
        "encoder stride" if global_spatial else "architecture.encoder_crop_size"
    )
    padding_advice = (
        "Use a grid or crop whose dimensions are multiples of 16."
        if global_spatial
        else "Pick an encoder_crop_size that divides the grid, or crop to a multiple."
    )
    if crop is not None and int(crop) < enc_crop:
        print(
            f"WARNING: dataset.random_crop_size={crop} < "
            f"{padding_name}={enc_crop}: every crop is zero-padded "
            f"{crop}->{enc_crop} per spatial dim (wasted compute, padding-dominated "
            "reconstruction). Set random_crop_size to a multiple of "
            f"{padding_name}."
        )
    elif crop is None:
        # Full-field path: the model tiles each grid into encoder_crop_size cubes
        # and zero-pads any dim that is not a multiple of it. Padding tiles waste
        # compute AND pollute the KL metric (their all-zero latents are counted).
        # Warn once per distinct grid shape that is not cleanly divisible.
        bad_shapes = {
            tuple(g.shape)
            for g in train_ds._geometries
            if any(int(d) % enc_crop != 0 for d in g.shape)
        }
        for shape in sorted(bad_shapes):
            offenders = [int(d) for d in shape if int(d) % enc_crop != 0]
            print(
                f"WARNING: full-field grid {shape} has dim(s) {offenders} not a "
                f"multiple of {padding_name}={enc_crop}: those axes "
                "are zero-padded up to the next multiple every forward (wasted "
                "compute, and the padded tiles affect the logged KL metric). "
                f"{padding_advice}"
            )

    print(
        f"train snapshots={len(train_ds)}  val snapshots={len(val_ds)}  "
        f"n_state_channels={len(cfg.dataset.state_vars)}"
    )
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total model parameters={num_params:,} (trainable={num_trainable:,})")

    # Optional adversarial (GAN) extension: built ONLY when the config carries a
    # `discriminator` block, so the default (`discriminator: null`) path below is
    # exactly the pure-VAE call it has always been -- no critic, no second
    # optimizer, no extra trainer kwargs.
    adv_kwargs: dict = {}
    disc_cfg = cfg.get("discriminator")
    if disc_cfg is not None:
        if cfg.get("disc_optimizer") is None:
            raise ValueError(
                "a `discriminator` block needs a `disc_optimizer` block: the "
                "critic's loss is adversarial to the autoencoder's, so the two "
                "must never share an optimizer (or its moments)."
            )
        # Channel/geometry contract is read off the BUILT model, not the config,
        # so the critic can never disagree with the autoencoder about how many
        # channels the working-space field has. The paper pairs the critic with a
        # same-size autoencoder, so `size` defaults to the architecture's.
        disc_size = disc_cfg.get("size", cfg.architecture.size)
        # In geometry-branch mode the AE's working space has NO geometry channels,
        # but the critic should still judge the flow in the presence of its
        # obstacles: give it the bare mask (the trainer's "mask" source feeds it
        # from the raw geometry argument). Outside branch mode this is exactly the
        # architecture's own contract, unchanged.
        branch_mode = getattr(model, "geometry_branch", None) is not None
        discriminator = instantiate(
            disc_cfg,
            size=disc_size,
            n_state_channels=model.n_state_channels,
            encode_geometry=model.encode_geometry or branch_mode,
            sdf_features="none" if branch_mode else model.sdf_feature_mode,
        ).to(dtype=dtype)
        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(
            f"adversarial loss ON: discriminator size={disc_size} "
            f"parameters={disc_params:,} "
            f"(in_channels={discriminator.n_input_channels})"
        )
        adv_kwargs = dict(
            discriminator=discriminator,
            disc_optimizer=instantiate(
                cfg.disc_optimizer, params=discriminator.parameters()
            ),
            adv_weight=cfg.loss.adv_weight,
            adv_start_step=cfg.loss.adv_start_step,
            adv_ramp_steps=cfg.loss.adv_ramp_steps,
            adaptive_adv_weight=cfg.loss.adaptive_adv_weight,
            disc_recon_threshold=cfg.loss.get("disc_recon_threshold"),
        )

    # Per-channel standardisation stats over the training split's fluid cells.
    # Stored as model buffers (travel with the checkpoint); param stats are empty
    # for a snapshot AE and ignored by TadpoleAE.set_normalization.
    s_mean, s_std, p_mean, p_std = get_normalization_stats(train_ds)
    model.set_normalization(s_mean, s_std, p_mean, p_std)
    print(
        f"normalization set:\n"
        f"  state_mean={np.round(s_mean, 4)} state_std={np.round(s_std, 4)}"
    )

    out_dir = Path("model_weights") / cfg.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    trainer = instantiate(
        cfg.trainer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=instantiate(cfg.optimizer, params=model.parameters()),
        # The reconstruction metric is plain MSE (masked to fluid cells in the
        # trainer when mask_loss=true); the KL / geometry weights come from the
        # loss block as trainer kwargs.
        loss_fn=torch.nn.MSELoss(),
        weights_path=out_dir / "weights.pt",
        kl_weight=cfg.loss.kl_weight,
        geometry_recon_weight=cfg.loss.geometry_recon_weight,
        **adv_kwargs,
    )
    trainer.fit()

    # Also export encoder/decoder separately -- the handoff format for plan 03 and
    # for sharing/HF-style reuse. Uses the best-val weights the trainer reloaded.
    model.ae.save_separate_weights(
        str(out_dir / "encoder.pt"), str(out_dir / "decoder.pt")
    )
    # The geometry branch is a separate module (the encoder/decoder only carry its
    # zero-init projections), so it needs its own handoff file -- written ONLY in
    # branch mode, so a standard run's artifact set is unchanged.
    if getattr(model, "geometry_branch", None) is not None:
        torch.save(model.geometry_branch.state_dict(), out_dir / "geometry_branch.pt")
        print("geometry branch saved to geometry_branch.pt")
    print(f"config, best weights and encoder/decoder saved to {out_dir}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/pretrain_autoencoder",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
