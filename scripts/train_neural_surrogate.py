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
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def _compute_normalization_stats(
    train_ds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel state and param mean/std over the training split's fluid cells.

    Mirrors upstream UPT's dataset standardisation: the network sees zero-mean,
    unit-variance fields. Stats are streamed file-by-file (sum / sum-of-squares
    in float64) and restricted to fluid cells via the dataset geometry mask so
    the masked-out obstacle zeros do not bias them.
    """
    fluid = train_ds._geometry.cpu().numpy().astype(bool)
    n_state = len(train_ds.state_vars)
    s_sum = np.zeros(n_state, dtype=np.float64)
    s_sqsum = np.zeros(n_state, dtype=np.float64)
    s_count = 0
    for state_path in train_ds._state_files:
        with xr.open_dataset(state_path) as ds:
            for c, var in enumerate(train_ds.state_vars):
                vals = np.asarray(ds[var].values)  # (T, *grid)
                masked = vals[:, fluid]  # (T, n_fluid)
                s_sum[c] += masked.sum(dtype=np.float64)
                s_sqsum[c] += np.square(masked, dtype=np.float64).sum()
            s_count += masked.shape[0] * masked.shape[1]
    state_mean = s_sum / s_count
    state_std = np.sqrt(np.maximum(s_sqsum / s_count - state_mean**2, 0.0))

    params = torch.cat([p for p in train_ds._params], dim=0).cpu().numpy()  # (sum_T, P)
    param_mean = params.mean(axis=0)
    param_std = params.std(axis=0)
    return state_mean, state_std, param_mean, param_std


def run(cfg: DictConfig) -> None:
    dtype = getattr(torch, cfg.dataset.dtype)

    train_ds = instantiate(cfg.dataset, split="train", dtype=dtype)
    val_ds = instantiate(cfg.dataset, split="val", dtype=dtype)
    train_loader = instantiate(cfg.dataloader, dataset=train_ds)
    val_loader = instantiate(cfg.dataloader, dataset=val_ds, shuffle=False)

    model = instantiate(
        cfg.architecture,
        n_state_channels=len(cfg.dataset.state_vars),
        n_params=len(train_ds.param_names),
    ).to(dtype=dtype)

    print(
        f"train pairs={len(train_ds)}  param_names={train_ds.param_names}  "
        f"n_state_channels={len(cfg.dataset.state_vars)}"
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total model parameters={num_params:,} (trainable={num_trainable:,})")

    # Install standardisation statistics if the architecture supports it (UPT).
    # The stats are stored as model buffers, so they are saved with the weights
    # and restored automatically at rollout/test time -- no other call site
    # needs to know about normalisation.
    if hasattr(model, "set_normalization"):
        s_mean, s_std, p_mean, p_std = _compute_normalization_stats(train_ds)
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
    config_path="../conf",
    config_name="neural_surrogate_training/train",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
