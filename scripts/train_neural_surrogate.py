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

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from neural_surrogates.architectures import SimpleConv


def run(cfg: DictConfig) -> None:
    dtype = getattr(torch, cfg.dataset.dtype)

    train_ds = instantiate(cfg.dataset, split="train", dtype=dtype)
    val_ds = instantiate(cfg.dataset, split="val", dtype=dtype)
    train_loader = instantiate(cfg.dataloader, dataset=train_ds)
    val_loader = instantiate(cfg.dataloader, dataset=val_ds, shuffle=False)

    model = SimpleConv(
        n_state_channels=len(cfg.dataset.state_vars),
        n_params=len(train_ds.param_names),
        kernel_size=cfg.kernel_size,
    ).to(dtype=dtype)

    print(
        f"train pairs={len(train_ds)}  param_names={train_ds.param_names}  "
        f"n_state_channels={len(cfg.dataset.state_vars)}"
    )

    trainer = instantiate(
        cfg.trainer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=instantiate(cfg.optimizer, params=model.parameters()),
        loss_fn=instantiate(cfg.loss),
    )
    trainer.fit()


@hydra.main(
    version_base=None,
    config_path="../conf/neural_surrogate_training",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
