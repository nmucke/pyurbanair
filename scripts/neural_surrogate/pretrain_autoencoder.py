"""Tadpole-style autoencoder pre-training (plan 02).

Pre-trains a ``TadpoleAE`` on flow snapshots as a representation-learning (V)AE
-- no next-step objective. Mirrors ``train_neural_surrogate.py``: build the
``SnapshotDataset`` splits, compute + install per-channel normalization stats,
save the resolved config, then run ``AutoencoderTrainer.fit()``.

Artifacts land in ``model_weights/<model_name>/``: the full ``TadpoleAE``
``weights.pt`` (our standard, plus ``config.yaml`` / ``checkpoint.pt`` /
``metrics.csv`` as usual), and ``encoder.pt`` / ``decoder.pt`` via the
autoencoder's ``save_separate_weights`` -- the natural handoff format for plan 03
(AE -> time-stepper) and for HF-style reuse.

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
    # of SDF channels must be paired with a dataset that ships exactly those (and
    # at the same clamp radius). Both default to "none", so this is a no-op for
    # standard runs.
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
    enc_crop = int(model.encoder_crop_size)
    if crop is not None and int(crop) < enc_crop:
        print(
            f"WARNING: dataset.random_crop_size={crop} < "
            f"architecture.encoder_crop_size={enc_crop}: every crop is zero-padded "
            f"{crop}->{enc_crop} per spatial dim (wasted compute, padding-dominated "
            "reconstruction). Set random_crop_size to a multiple of "
            "encoder_crop_size."
        )

    print(
        f"train snapshots={len(train_ds)}  val snapshots={len(val_ds)}  "
        f"n_state_channels={len(cfg.dataset.state_vars)}"
    )
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total model parameters={num_params:,} (trainable={num_trainable:,})")

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
    )
    trainer.fit()

    # Also export encoder/decoder separately -- the handoff format for plan 03 and
    # for sharing/HF-style reuse. Uses the best-val weights the trainer reloaded.
    model.ae.save_separate_weights(
        str(out_dir / "encoder.pt"), str(out_dir / "decoder.pt")
    )
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
