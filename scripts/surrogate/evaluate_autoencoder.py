"""Evaluate a trained autoencoder: how well it reconstructs held-out snapshots.

Config: configs/surrogate/eval.yaml, block `autoencoder`.

    python scripts/surrogate/evaluate_autoencoder.py
    python scripts/surrogate/evaluate_autoencoder.py autoencoder.model_dir=model_weights/ae

Snapshots are reconstructed whole (no random crop). Every metric is over fluid
cells only.

Outputs, in `autoencoder.output_dir`:
  metrics.json               overall RMSE, relative L2 and PSNR; RMSE, MAE,
                             relative L2 and R² per variable; mean KL and the
                             fraction of active latent channels
  reconstruction_*.png       |U| at a few heights: truth, reconstruction, error
                             (one figure per snapshot)
  per_channel_metrics.png    RMSE and relative L2 per variable
  height_profile.png         |U| RMSE against normalised height
  error_hist.png             histogram of the |U| error
  pred_vs_true.png           reconstructed against true |U|
  latent_stats.png           KL per snapshot and spread per latent channel
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import (  # noqa: E402
    device,
    expand,
    load_model,
    load_states,
    open_dataset,
    plot_slice_rows,
    speed,
    write_json,
)

HEIGHT_BINS = 20
VALUES_PER_SNAPSHOT = 4000  # |U| values kept per snapshot for the histograms
ACTIVE_STD = 1e-2  # a latent channel with less spread than this is unused


def run(cfg: DictConfig) -> None:
    ev = cfg.autoencoder
    dev = device(cfg.device)
    out = pathlib.Path(ev.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, train_cfg = load_model(ev.model_dir, dev)
    model.ae.latent_type = str(ev.latent_type)
    data = open_dataset(train_cfg, cfg.data_dir, cfg.split, random_crop_size=None)
    names = list(data.state_vars)
    rng = np.random.default_rng(int(cfg.seed))

    # Reconstruction figures: one snapshot from each of a few trajectories.
    for traj in _spread(len(data._state_files), int(ev.num_figures)):
        t = int(rng.integers(data._traj_lengths[traj]))
        truth = load_states(data, traj, [t])
        recon, _, _ = _reconstruct(model, data, traj, truth, dev)
        _plot_reconstruction(
            truth[0],
            recon[0],
            data,
            traj,
            int(ev.num_heights),
            out / f"reconstruction_traj{traj}_t{t}.png",
        )

    # Metrics on a random set of snapshots, batched per trajectory (one grid).
    pairs = list(data.sample_index)
    chosen = rng.choice(
        len(pairs), min(int(ev.num_metric_samples), len(pairs)), replace=False
    )
    by_traj: dict[int, list[int]] = {}
    for i in sorted(chosen):
        by_traj.setdefault(pairs[i][0], []).append(pairs[i][1])

    acc = _Metrics(len(names), rng)
    kl, latent_std = [], []
    for traj, times in by_traj.items():
        fluid = data.geometry_for(traj).numpy().astype(bool)
        for start in range(0, len(times), int(ev.batch_size)):
            truth = load_states(data, traj, times[start : start + int(ev.batch_size)])
            recon, batch_kl, latent = _reconstruct(model, data, traj, truth, dev)
            kl += batch_kl.tolist()
            latent_std.append(
                np.moveaxis(latent, 1, 0).reshape(latent.shape[1], -1).std(axis=1)
            )
            for true_state, recon_state in zip(truth, recon):
                acc.add(true_state, recon_state, fluid)

    summary = acc.summary(names)
    spread = np.mean(latent_std, axis=0)
    summary |= {
        "mean_kl": float(np.mean(kl)),
        "latent_active_fraction": float(np.mean(spread > ACTIVE_STD)),
        "num_snapshots": len(kl),
        "latent_type": str(ev.latent_type),
    }
    write_json(summary, out / "metrics.json")
    _plot_per_channel(summary, names, out / "per_channel_metrics.png")
    _plot_height_profile(acc, out / "height_profile.png")
    _plot_error_hist(acc, out / "error_hist.png")
    _plot_pred_vs_true(acc, out / "pred_vs_true.png")
    _plot_latent_stats(kl, spread, out / "latent_stats.png")
    print(
        f"rmse={summary['rmse']:.4g} rel_l2={summary['rel_l2']:.4g} "
        f"psnr={summary['psnr_db']:.1f} dB, active latents {summary['latent_active_fraction']:.0%}"
    )
    print(f"Saved evaluation in {out}")


def _spread(n: int, k: int) -> list[int]:
    """`k` indices spread evenly over range(n)."""
    return sorted({int(i) for i in np.linspace(0, n - 1, min(k, n))})


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _reconstruct(
    model: Any, data: Any, traj: int, truth: np.ndarray, dev: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruction, KL per snapshot and the mode latent of a batch of one trajectory."""
    b = len(truth)
    state = torch.from_numpy(truth).to(dev)
    geom = expand(data.geometry_for(traj), b, dev)
    feat = expand(data.geom_features_for(traj), b, dev)
    recon = model(state, geom, feat)
    _, kl = model(state, geom, feat, return_kl_element=True)
    latent = model.encode(state, geom, feat, latent_type="mode")
    return (
        recon.float().cpu().numpy(),
        kl.reshape(b, -1).mean(dim=1).cpu().numpy(),
        latent.float().cpu().numpy(),
    )


class _Metrics:
    """Running sums over fluid cells: per variable, per height bin, plus a
    random sample of |U| values for the histograms."""

    def __init__(self, n_vars: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.sse, self.sae = np.zeros(n_vars), np.zeros(n_vars)
        self.sum, self.sum2, self.count = (
            np.zeros(n_vars),
            np.zeros(n_vars),
            np.zeros(n_vars),
        )
        self.height_sse, self.height_count = np.zeros(HEIGHT_BINS), np.zeros(
            HEIGHT_BINS
        )
        self.true_speed: list[np.ndarray] = []
        self.pred_speed: list[np.ndarray] = []

    def add(self, truth: np.ndarray, recon: np.ndarray, fluid: np.ndarray) -> None:
        t, r = truth[:, fluid], recon[:, fluid]  # (C, n_fluid)
        error = r - t
        self.sse += (error**2).sum(axis=1)
        self.sae += np.abs(error).sum(axis=1)
        self.sum += t.sum(axis=1)
        self.sum2 += (t**2).sum(axis=1)
        self.count += t.shape[1]

        speed_error = (speed(recon) - speed(truth)) ** 2  # (z, y, x)
        nz = fluid.shape[0]
        for z in range(nz):
            b = min(int(z / max(nz - 1, 1) * HEIGHT_BINS), HEIGHT_BINS - 1)
            self.height_sse[b] += speed_error[z][fluid[z]].sum()
            self.height_count[b] += fluid[z].sum()

        true_speed, pred_speed = np.sqrt((t**2).sum(axis=0)), np.sqrt(
            (r**2).sum(axis=0)
        )
        keep = self.rng.choice(
            true_speed.size, min(true_speed.size, VALUES_PER_SNAPSHOT), replace=False
        )
        self.true_speed.append(true_speed[keep])
        self.pred_speed.append(pred_speed[keep])

    def summary(self, names: list[str]) -> dict:
        rmse = np.sqrt(self.sse / self.count)
        variance = self.sum2 / self.count - (self.sum / self.count) ** 2
        true_speed = np.concatenate(self.true_speed)
        overall_rmse = float(np.sqrt(self.sse.sum() / self.count.sum()))
        data_range = float(true_speed.max() - true_speed.min()) or 1.0
        return {
            "rmse": overall_rmse,
            "rel_l2": float(np.sqrt(self.sse.sum() / self.sum2.sum())),
            "psnr_db": float(20 * np.log10(data_range / max(overall_rmse, 1e-12))),
            "per_variable": {
                name: {
                    "rmse": float(rmse[c]),
                    "mae": float(self.sae[c] / self.count[c]),
                    "rel_l2": float(np.sqrt(self.sse[c] / self.sum2[c])),
                    "r2": float(
                        1 - self.sse[c] / max(variance[c] * self.count[c], 1e-12)
                    ),
                }
                for c, name in enumerate(names)
            },
        }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_reconstruction(
    truth: np.ndarray,
    recon: np.ndarray,
    data: Any,
    traj: int,
    num_heights: int,
    path: pathlib.Path,
) -> None:
    fluid = data.geometry_for(traj).numpy().astype(bool)
    levels = _spread(fluid.shape[0], num_heights)
    true_speed, recon_speed = speed(truth), speed(recon)
    rows = [
        ("truth", [true_speed[z] for z in levels], False),
        ("reconstruction", [recon_speed[z] for z in levels], False),
        (
            "reconstruction - truth",
            [recon_speed[z] - true_speed[z] for z in levels],
            True,
        ),
    ]
    plot_slice_rows(rows, [f"z index {z}" for z in levels], fluid[levels], path)


def _plot_per_channel(summary: dict, names: list[str], path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key in zip(axes, ["rmse", "rel_l2"]):
        ax.bar(
            names, [summary["per_variable"][n][key] for n in names], color="tab:blue"
        )
        ax.set_title(key)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_height_profile(acc: _Metrics, path: pathlib.Path) -> None:
    filled = acc.height_count > 0
    heights = (np.arange(HEIGHT_BINS) + 0.5) / HEIGHT_BINS
    fig, ax = plt.subplots(figsize=(4, 4.5))
    ax.plot(
        np.sqrt(acc.height_sse[filled] / acc.height_count[filled]),
        heights[filled],
        "o-",
    )
    ax.set_xlabel("|U| RMSE [m/s]")
    ax.set_ylabel("normalised height")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_error_hist(acc: _Metrics, path: pathlib.Path) -> None:
    error = np.concatenate(acc.pred_speed) - np.concatenate(acc.true_speed)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(error, bins=80, color="tab:blue")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("|U| reconstruction - truth [m/s]")
    ax.set_ylabel("cells")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_vs_true(acc: _Metrics, path: pathlib.Path) -> None:
    true_speed, pred_speed = np.concatenate(acc.true_speed), np.concatenate(
        acc.pred_speed
    )
    lim = [0, float(max(true_speed.max(), pred_speed.max()))]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(true_speed, pred_speed, s=1, alpha=0.2)
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("true |U| [m/s]")
    ax.set_ylabel("reconstructed |U| [m/s]")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_stats(kl: list[float], spread: np.ndarray, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(kl, bins=30, color="darkorange")
    axes[0].set_xlabel("KL per snapshot")
    axes[1].bar(np.arange(len(spread)), spread, color="tab:blue")
    axes[1].axhline(ACTIVE_STD, color="k", ls="--", lw=0.8, label="active threshold")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("latent channel")
    axes[1].set_ylabel("std (mode latent)")
    axes[1].legend()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs", config_name="surrogate/eval"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
