"""Autoregressively roll out a trained neural surrogate on a test trajectory.

Loads the architecture and dataset from the saved
`model_weights/<model_name>/config.yaml`, restores `weights.pt`, picks
one test trajectory, and steps the model from its initial condition for
the same number of steps as the ground-truth trajectory.

Usage:

    pixi run -e dev python scripts/test_neural_surrogate.py
    pixi run -e dev python scripts/test_neural_surrogate.py \
        model_dir=model_weights/unet_convnext_small sample_idx=2
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def _load_trajectory(
    dataset, sample_idx: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_path = dataset._state_files[sample_idx]
    with xr.open_dataset(state_path) as ds:
        channels = np.stack(
            [np.asarray(ds[v].values) for v in dataset.state_vars], axis=1
        )
    truth = torch.from_numpy(channels).to(dtype)
    params = dataset._params[sample_idx]
    geometry = dataset._geometry
    return truth, params, geometry


@torch.no_grad()
def _rollout(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    params: torch.Tensor,
    geometry: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    pred = torch.empty((n_steps + 1, *initial_state.shape), dtype=initial_state.dtype)
    pred[0] = initial_state
    state = initial_state.unsqueeze(0).to(device)
    geom = geometry.unsqueeze(0).to(device)
    for t in range(n_steps):
        param_t = params[t].unsqueeze(0).to(device)
        next_state = model(state, param_t, geom)
        pred[t + 1] = next_state[0].cpu()
        state = next_state
    return pred


def _plot_rollout(
    truth: torch.Tensor, pred: torch.Tensor, out_dir: Path, n_show: int = 5
) -> None:
    T = truth.shape[0]
    z_mid = truth.shape[-3] // 2
    times = np.linspace(0, T - 1, min(n_show, T), dtype=int)
    mag_t = truth.norm(dim=1)[:, z_mid]
    mag_p = pred.norm(dim=1)[:, z_mid]
    vmax = mag_t.max().item()

    fig, axes = plt.subplots(
        3, len(times), figsize=(3 * len(times), 9), squeeze=False
    )
    for i, t in enumerate(times):
        axes[0, i].imshow(mag_t[t].numpy(), origin="lower", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"truth t={t}")
        axes[1, i].imshow(mag_p[t].numpy(), origin="lower", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"pred t={t}")
        diff = (mag_p[t] - mag_t[t]).abs().numpy()
        axes[2, i].imshow(diff, origin="lower", vmin=0, vmax=vmax)
        axes[2, i].set_title(f"|err| t={t}")
        for r in range(3):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    fig.suptitle(f"|u| rollout at z-index {z_mid}")
    fig.tight_layout()
    fig.savefig(out_dir / "rollout.png", dpi=120)
    plt.close(fig)

    per_step = ((pred - truth) ** 2).mean(dim=tuple(range(1, pred.ndim))).sqrt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(per_step.numpy())
    ax.set_xlabel("time step")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-step rollout RMSE")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse.png", dpi=120)
    plt.close(fig)


def _plot_params(
    params: torch.Tensor,
    param_names: tuple[str, ...],
    out_dir: Path,
) -> None:
    arr = params.numpy()
    T, P = arr.shape
    fig, axes = plt.subplots(P, 1, figsize=(8, 2.5 * max(P, 1)), squeeze=False)
    for i, name in enumerate(param_names):
        axes[i, 0].plot(np.arange(T), arr[:, i])
        axes[i, 0].set_xlabel("time step")
        axes[i, 0].set_ylabel(name)
        axes[i, 0].grid(True, alpha=0.3)
    fig.suptitle("Rollout parameters")
    fig.tight_layout()
    fig.savefig(out_dir / "params.png", dpi=120)
    plt.close(fig)


def _animate_rollout(
    truth: torch.Tensor, pred: torch.Tensor, out_path: Path, fps: int = 10
) -> None:
    T = truth.shape[0]
    z_mid = truth.shape[-3] // 2
    mag_t = truth.norm(dim=1)[:, z_mid].numpy()
    mag_p = pred.norm(dim=1)[:, z_mid].numpy()
    err = np.abs(mag_p - mag_t)
    vmax = float(mag_t.max())
    err_max = float(err.max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ims = [
        axes[0].imshow(mag_t[0], origin="lower", vmin=0, vmax=vmax, cmap="viridis"),
        axes[1].imshow(mag_p[0], origin="lower", vmin=0, vmax=vmax, cmap="viridis"),
        axes[2].imshow(err[0], origin="lower", vmin=0, vmax=err_max, cmap="magma"),
    ]
    titles = ["truth", "pred", "|err|"]
    for ax, title in zip(axes, titles):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    for im, ax in zip(ims[:2], axes[:2]):
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.colorbar(ims[2], ax=axes[2], fraction=0.046)
    suptitle = fig.suptitle(f"|u| at z={z_mid}  t=0/{T - 1}")

    def update(frame: int):
        ims[0].set_array(mag_t[frame])
        ims[1].set_array(mag_p[frame])
        ims[2].set_array(err[frame])
        suptitle.set_text(f"|u| at z={z_mid}  t={frame}/{T - 1}")
        return [*ims, suptitle]

    if animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps)
        save_path = out_path
    else:
        writer = animation.PillowWriter(fps=fps)
        save_path = out_path.with_suffix(".gif")

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(str(save_path), writer=writer, dpi=120)
    plt.close(fig)
    return save_path


def run(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir)
    train_cfg = OmegaConf.load(model_dir / "config.yaml")
    dtype = getattr(torch, train_cfg.dataset.dtype)
    device = torch.device(cfg.device)

    test_ds = instantiate(train_cfg.dataset, split="test", dtype=dtype)

    model = (
        instantiate(
            train_cfg.architecture,
            n_state_channels=len(train_cfg.dataset.state_vars),
            n_params=len(test_ds.param_names),
        )
        .to(dtype=dtype)
        .to(device)
    )
    model.load_state_dict(
        torch.load(model_dir / "weights.pt", map_location=device)
    )
    model.eval()

    truth, params, geometry = _load_trajectory(test_ds, cfg.sample_idx, dtype)
    T = truth.shape[0]
    print(
        f"loaded trajectory {cfg.sample_idx}  "
        f"shape={tuple(truth.shape)}  param_names={test_ds.param_names}"
    )

    pred = _rollout(
        model=model,
        initial_state=truth[0],
        params=params,
        geometry=geometry,
        n_steps=T - 1,
        device=device,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"truth": truth, "pred": pred}, out_dir / "trajectory.pt")
    _plot_rollout(truth, pred, out_dir)
    _plot_params(params, test_ds.param_names, out_dir)
    anim_path = _animate_rollout(truth, pred, out_dir / "rollout.mp4")

    rmse = ((pred - truth) ** 2).mean().sqrt().item()
    print(f"overall RMSE={rmse:.6f}  outputs in {out_dir}  animation={anim_path.name}")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="neural_surrogate_testing/test",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
