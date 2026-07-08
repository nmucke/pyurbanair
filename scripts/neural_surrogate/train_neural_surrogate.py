"""Train the simple-conv neural surrogate on a training-data split.

This is a thin loop intended to validate the end-to-end stack
(dataloader → model → optimizer); the model itself is the single-conv
baseline in `neural_surrogates.architectures.SimpleConv`.

Usage:

    pixi run -e dev python scripts/train_neural_surrogate.py
    pixi run -e dev python scripts/train_neural_surrogate.py \
        dataset.root_dir=training_data/pylbm_small trainer.num_epochs=10
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
    # persistent_workers needs worker processes; force it off for workerless
    # runs (CPU smoke tests / debugging) so the DataLoader accepts the config.
    if int(cfg.dataloader.get("num_workers", 0)) == 0:
        cfg.dataloader.persistent_workers = False
    train_loader = build_loader(cfg, train_ds, train=True)
    val_loader = build_loader(cfg, val_ds, train=False)

    model = instantiate(
        cfg.architecture,
        n_state_channels=len(cfg.dataset.state_vars),
        n_params=len(train_ds.param_names),
    ).to(dtype=dtype)

    # Cross-check the SDF-feature modes: a model whose stem was widened for a
    # specific set of SDF channels must be paired with a dataset that ships
    # exactly those (and at the same clamp radius), or training silently trains
    # on the wrong stem. Fail loud on any mismatch of the selected mode. Both
    # default to "none", so this is a no-op for standard runs.
    model_mode = getattr(model, "sdf_feature_mode", "none")
    dataset_mode = getattr(train_ds, "sdf_feature_mode", "none")
    model_wants_features = getattr(model, "n_geom_feature_channels", 0) > 0
    if model_mode != dataset_mode:
        raise ValueError(
            "SDF-feature mismatch: architecture.sdf_features="
            f"{model_mode!r} but dataset.sdf_features={dataset_mode!r}. "
            "They must select the same channels (set both to the same mode)."
        )
    if model_wants_features and float(train_ds.sdf_clamp_cells) != float(
        getattr(model, "sdf_clamp_cells", train_ds.sdf_clamp_cells)
    ):
        raise ValueError(
            "SDF clamp mismatch: architecture.sdf_clamp_cells="
            f"{model.sdf_clamp_cells} but dataset.sdf_clamp_cells="
            f"{train_ds.sdf_clamp_cells}; they must match."
        )

    print(
        f"train pairs={len(train_ds)}  param_names={train_ds.param_names}  "
        f"n_state_channels={len(cfg.dataset.state_vars)}"
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total model parameters={num_params:,} (trainable={num_trainable:,})")

    # Optionally warm-start from previously trained weights. This loads the full
    # state dict (including any normalization buffers) before we recompute the
    # standardisation stats below, so the stats always reflect the *current*
    # training split -- the right invariant whether continuing on the same data
    # or fine-tuning on a new dataset.
    init_weights_path = cfg.get("init_weights_path")
    if init_weights_path is not None:
        init_weights_path = Path(init_weights_path)
        if not init_weights_path.exists():
            raise FileNotFoundError(
                f"init_weights_path does not exist: {init_weights_path}"
            )
        state_dict = torch.load(init_weights_path, map_location=cfg.trainer.device)
        model.load_state_dict(state_dict)
        print(f"initialized model weights from {init_weights_path}")

    # Install standardisation statistics if the architecture supports it (UPT).
    # The stats are stored as model buffers, so they are saved with the weights
    # and restored automatically at rollout/test time -- no other call site
    # needs to know about normalisation.
    if hasattr(model, "set_normalization"):
        s_mean, s_std, p_mean, p_std = get_normalization_stats(train_ds)
        model.set_normalization(s_mean, s_std, p_mean, p_std)
        print(
            f"normalization set:\n"
            f"  state_mean={np.round(s_mean, 4)} state_std={np.round(s_std, 4)}\n"
            f"  param_mean={np.round(p_mean, 4)} param_std={np.round(p_std, 4)}"
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
        loss_fn=instantiate(cfg.loss),
        weights_path=out_dir / "weights.pt",
    )
    trainer.fit()
    print(f"config and best weights saved to {out_dir}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/training",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
