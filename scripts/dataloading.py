"""Smoke test for the neural-surrogate transition dataloader.

Builds a `TransitionDataset` over the configured training-data split,
wraps it in a `DataLoader`, and prints the shape of the first few batches
so we can sanity-check the pair layout.

Usage:

    pixi run -e dev python scripts/dataloading.py
    pixi run -e dev python scripts/dataloading.py data_dir=training_data/pylbm_small
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from neural_surrogates import TransitionDataset


def _plot_batch(
    batch: dict[str, torch.Tensor],
    param_names: tuple[str, ...],
    out_dir: Path,
) -> None:
    state_n = batch["state_n"]
    state_next = batch["state_next"]
    params = batch["params_n"]
    geometry = batch["geometry"][0]

    z_mid = state_n.shape[2] // 2
    mag_n = state_n[:, :, z_mid].norm(dim=1)
    mag_next = state_next[:, :, z_mid].norm(dim=1)
    n_show = min(state_n.shape[0], 4)
    vmax = max(mag_n.max().item(), mag_next.max().item())
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6), squeeze=False)
    for i in range(n_show):
        axes[0, i].imshow(mag_n[i].numpy(), origin="lower", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"state_n[{i}]")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].imshow(mag_next[i].numpy(), origin="lower", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"state_next[{i}]")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    fig.suptitle(f"|u| at z-index {z_mid}")
    fig.tight_layout()
    fig.savefig(out_dir / "states.png", dpi=120)
    plt.close(fig)

    name_to_col = {name: i for i, name in enumerate(param_names)}
    angle_key = next((n for n in param_names if "angle" in n), param_names[0])
    vel_key = next(
        (n for n in param_names if "vel" in n),
        param_names[1] if len(param_names) > 1 else param_names[0],
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        params[:, name_to_col[angle_key]].numpy(),
        params[:, name_to_col[vel_key]].numpy(),
    )
    ax.set_xlabel(angle_key)
    ax.set_ylabel(vel_key)
    ax.set_title("Batch parameters")
    fig.tight_layout()
    fig.savefig(out_dir / "params.png", dpi=120)
    plt.close(fig)

    nz = geometry.shape[0]
    fig, axes = plt.subplots(1, nz, figsize=(3 * nz, 3), squeeze=False)
    for k in range(nz):
        axes[0, k].imshow(
            geometry[k].numpy(), origin="lower", cmap="gray", vmin=0, vmax=1
        )
        axes[0, k].set_title(f"z-index {k}")
        axes[0, k].set_xticks([])
        axes[0, k].set_yticks([])
    fig.suptitle("Geometry (1=fluid, 0=obstacle)")
    fig.tight_layout()
    fig.savefig(out_dir / "geometry.png", dpi=120)
    plt.close(fig)


def run(cfg: DictConfig) -> None:
    dtype = getattr(torch, cfg.dtype)
    param_vars = list(cfg.param_vars) if cfg.param_vars else None

    dataset = TransitionDataset(
        root_dir=cfg.data_dir,
        split=cfg.split,
        state_vars=tuple(cfg.state_vars),
        param_vars=param_vars,
        cache=cfg.cache,
        dtype=dtype,
    )
    print(
        f"split='{cfg.split}'  trajectories={len(dataset._state_files)}  "
        f"pairs={len(dataset)}  param_names={dataset.param_names}"
    )
    print(f"geometry mask: shape={tuple(dataset._geometry.shape)}  "
          f"fluid_fraction={dataset._geometry.mean().item():.3f}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )

    last_batch = None
    for i, batch in enumerate(loader):
        print(f"\nbatch {i}:")
        print(f"  state_n     shape={tuple(batch['state_n'].shape)}     "
              f"dtype={batch['state_n'].dtype}")
        print(f"  state_next  shape={tuple(batch['state_next'].shape)}  "
              f"dtype={batch['state_next'].dtype}")
        print(f"  params_n    shape={tuple(batch['params_n'].shape)}    "
              f"dtype={batch['params_n'].dtype}")
        print(f"  geometry    shape={tuple(batch['geometry'].shape)}    "
              f"dtype={batch['geometry'].dtype}")
        last_batch = batch
        if i >= 2:
            break

    out_dir = Path(cfg.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_batch(last_batch, dataset.param_names, out_dir)
    print(f"\nplots written to {out_dir}/")


@hydra.main(
    version_base=None,
    config_path="../conf/neural_surrogate_training",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
