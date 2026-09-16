"""Autoregressively roll out a trained neural surrogate on a test trajectory.

Loads the architecture and dataset from the saved
`model_weights/<model_name>/config.yaml`, restores the model weights,
picks one test trajectory, and steps the model from its initial
condition for the same number of steps as the ground-truth trajectory.

The model is restored from ``weights.pt`` — the best-validation weights.
(The sibling ``checkpoint.pt`` is the trainer's full latest-epoch state and
exists only for resuming training, not for evaluation.) ``metrics.csv`` is
echoed for context when present.

Alongside the velocity error the rollout is also scored on resolved turbulent
kinetic energy — a surrogate can track ``|u|`` closely while carrying the wrong
amount of fluctuation, and nothing in the RMSE separates the two. ``k`` is
formed per frame from a sliding Reynolds average (``tke_window`` frames,
``null`` = the whole rollout) by :func:`evaluation.turbulence.rolling_tke`, and
reported three ways: one scalar over the whole domain and rollout, a per-step
spatial error curve (``tke_error.png``), and a per-grid-point error panel in the
animation.

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
from evaluation.turbulence import rolling_tke
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
    geometry = dataset.geometry_for(sample_idx)
    return truth, params, geometry


@torch.no_grad()
def _rollout(
    model: torch.nn.Module,
    truth: torch.Tensor,
    params: torch.Tensor,
    geometry: torch.Tensor,
    n_steps: int,
    device: torch.device,
    num_history_steps: int = 1,
) -> torch.Tensor:
    """Autoregressive rollout seeded from the ground truth.

    A one-step network (``num_history_steps == 1``) is seeded with ``truth[0]``
    and predicts ``t = 1 … n_steps``. A history-conditioned network needs ``H``
    frames before it can predict, so it is seeded with the ground-truth window
    ``truth[0:H]`` (flattened oldest-first into the channel axis) and predicts
    from ``t = H`` onward; ``pred[0:H]`` is set to that same window so the
    returned trajectory keeps the ground truth's length and time indexing.
    """
    H = num_history_steps
    C = truth.shape[1]
    grid = truth.shape[2:]
    pred = torch.empty((n_steps + 1, *truth.shape[1:]), dtype=truth.dtype)
    pred[:H] = truth[:H]
    state = truth[:H].reshape(1, H * C, *grid).to(device)
    geom = geometry.unsqueeze(0).to(device)
    for t in range(H - 1, n_steps):
        param_t = params[t].unsqueeze(0).to(device)
        next_state = model(state, param_t, geom)
        pred[t + 1] = next_state[0].cpu()
        state = next_state if H == 1 else torch.cat([state[:, C:], next_state], dim=1)
    return pred


def _plot_rollout(
    truth: torch.Tensor, pred: torch.Tensor, out_dir: Path, n_show: int = 5
) -> None:
    T = truth.shape[0]
    z_mid = 0  # truth.shape[-3] // 2
    times = np.linspace(0, T - 1, min(n_show, T), dtype=int)
    mag_t = truth.norm(dim=1)[:, z_mid]
    mag_p = pred.norm(dim=1)[:, z_mid]
    vmax = mag_t.max().item()

    fig, axes = plt.subplots(3, len(times), figsize=(3 * len(times), 9), squeeze=False)
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


def _upper_limit(values: np.ndarray, percentile: float = 100.0) -> float:
    """A positive colour-scale top, robust to an all-NaN or perfectly flat panel."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    top = float(np.percentile(finite, percentile))
    return top if top > 0.0 else 1.0


def _tke_window(n_time: int, configured: int | None) -> int:
    """Frames in the sliding Reynolds average behind the TKE diagnostics.

    ``None`` (the config default) is a fifth of the rollout, floored at 8 frames
    and capped at its length. Both halves of that are deliberate. The *fraction*
    is what keeps the per-step error curve a curve: averaging over the whole
    rollout makes ``k`` one static field, so the curve goes flat and says
    nothing a single scalar does not — that pass-long view is what
    :class:`evaluation.turbulence.MomentAccumulator` is for, and it is still
    reachable by configuring a window at or above the rollout length. The
    *floor* is the other side of the trade: a variance over fewer than ~8 frames
    is mostly its own sampling scatter.
    """
    if configured is not None:
        return max(1, min(int(configured), n_time))
    return max(1, min(max(8, n_time // 5), n_time))


def _tke_fields(
    truth: torch.Tensor,
    pred: torch.Tensor,
    state_vars: tuple[str, ...],
    window: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-frame resolved TKE of the truth and the rollout, on the state grid.

    ``None`` when the trained state carries no velocity component at all: ``k``
    is a moment of the velocity field and there is nothing to form it from. A
    state carrying only some of ``u``/``v``/``w`` is scored on the components it
    has, which is the same partial sum on both trajectories and so still a fair
    comparison.
    """
    channels = [state_vars.index(v) for v in ("u", "v", "w") if v in state_vars]
    if not channels:
        return None
    return (
        rolling_tke(*(truth[:, c].numpy() for c in channels), window=window),
        rolling_tke(*(pred[:, c].numpy() for c in channels), window=window),
    )


def _plot_tke_error(
    tke_truth: np.ndarray,
    tke_pred: np.ndarray,
    fluid: np.ndarray,
    window_label: str,
    out_dir: Path,
) -> dict[str, float]:
    """Domain-mean TKE and the per-step spatial TKE error; returns the scalars.

    Every statistic is taken over fluid cells only — the solid cells the solver
    holds at rest carry no turbulence, and leaving them in would dilute the
    error by whatever fraction of the domain the buildings happen to occupy.
    """
    cells_truth = tke_truth[:, fluid]  # (time, fluid cell)
    cells_pred = tke_pred[:, fluid]
    error = cells_pred - cells_truth

    per_step_mae = np.nanmean(np.abs(error), axis=1)
    per_step_rmse = np.sqrt(np.nanmean(error**2, axis=1))
    scalars = {
        "tke_mae": float(np.nanmean(np.abs(error))),
        "tke_rmse": float(np.sqrt(np.nanmean(error**2))),
        "tke_bias": float(np.nanmean(error)),
        "tke_truth_mean": float(np.nanmean(cells_truth)),
    }

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(np.nanmean(cells_truth, axis=1), label="truth", lw=1.8)
    axes[0].plot(np.nanmean(cells_pred, axis=1), label="pred", lw=1.8)
    axes[0].set_ylabel("domain-mean k [m²/s²]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(per_step_mae, label="MAE", lw=1.8)
    axes[1].plot(per_step_rmse, label="RMSE", lw=1.8, ls="--")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("spatial k error [m²/s²]")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Resolved TKE ({window_label}, fluid cells)  "
        f"MAE={scalars['tke_mae']:.4g}  bias={scalars['tke_bias']:+.4g} m²/s²"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "tke_error.png", dpi=120)
    plt.close(fig)
    return scalars


def _animate_rollout(
    truth: torch.Tensor,
    pred: torch.Tensor,
    out_path: Path,
    fps: int = 10,
    tke_error: np.ndarray | None = None,
) -> Path:
    """Animate per-slice |u| truth/pred/error, plus the TKE error when given.

    ``tke_error`` is the signed per-grid-point ``k_pred - k_truth`` field
    ``(T, Z, Y, X)``; it is drawn as ``|Δk|`` in a fourth column.
    """
    T = truth.shape[0]
    z_slices = [2, 10, 25, 50]
    n_z = truth.shape[2]  # (T, C, Z, Y, X) → shape[2] is Z
    z_slices = [z for z in z_slices if z < n_z]

    mag_t_full = truth.norm(dim=1).numpy()  # (T, Z, Y, X)
    mag_p_full = pred.norm(dim=1).numpy()

    # per-slice arrays: list of (T, Y, X)
    slices_t = [mag_t_full[:, z] for z in z_slices]
    slices_p = [mag_p_full[:, z] for z in z_slices]
    slices_e = [np.abs(p - t) for t, p in zip(slices_t, slices_p)]
    slices_k = (
        None if tke_error is None else [np.abs(tke_error[:, z]) for z in z_slices]
    )

    titles = ["truth", "pred", "|err|"] + ([] if slices_k is None else ["TKE |Δk|"])
    n_rows, n_cols = len(z_slices), len(titles)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    ims = []
    for row, z in enumerate(z_slices):
        vmax = _upper_limit(slices_t[row])
        panels = [
            (slices_t[row], vmax, "viridis"),
            (slices_p[row], vmax, "viridis"),
            (slices_e[row], _upper_limit(slices_e[row]), "magma"),
        ]
        if slices_k is not None:
            # Squared quantities have far heavier tails than |u| does, so a
            # single hot cell would flatten the whole panel on a max scale.
            panels.append((slices_k[row], _upper_limit(slices_k[row], 99.0), "magma"))
        for col, (frames, top, cmap) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(frames[0], origin="lower", vmin=0, vmax=top, cmap=cmap)
            ims.append((image, frames))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"z={z}", labelpad=2) if col == 0 else None
            ax.set_title(titles[col]) if row == 0 else None
            fig.colorbar(image, ax=ax, fraction=0.046)

    quantities = "|u|" if slices_k is None else "|u| and resolved-TKE error"
    suptitle = fig.suptitle(f"{quantities} vertical slices  t=0/{T - 1}")

    def update(frame: int):
        artists = [suptitle]
        for image, frames in ims:
            image.set_array(frames[frame])
            artists.append(image)
        suptitle.set_text(f"{quantities} vertical slices  t={frame}/{T - 1}")
        return artists

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

    # How many past frames the trained network consumes (H). Legacy configs
    # predate the key -> the classic one-step surrogate.
    num_history_steps = int(train_cfg.dataset.get("num_history_steps", 1))

    test_ds = instantiate(
        train_cfg.dataset,
        split="test",
        dtype=dtype,
        **({} if num_history_steps == 1 else {"num_history_steps": num_history_steps}),
    )

    arch_overrides: dict = {}
    if num_history_steps != 1 and "num_history_steps" not in train_cfg.architecture:
        arch_overrides["num_history_steps"] = num_history_steps
    model = (
        instantiate(
            train_cfg.architecture,
            n_state_channels=len(train_cfg.dataset.state_vars),
            n_params=len(test_ds.param_names),
            **arch_overrides,
        )
        .to(dtype=dtype)
        .to(device)
    )
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location=device))
    model.eval()
    print("loaded weights.pt (best-validation weights)")

    metrics_path = model_dir / "metrics.csv"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        print(f"training metrics: {len(lines) - 1} epochs logged; last: {lines[-1]}")

    truth, params, geometry = _load_trajectory(test_ds, cfg.sample_idx, dtype)
    truth = truth[0:150]
    params = params[0:150]
    T = truth.shape[0]
    print(
        f"loaded trajectory {cfg.sample_idx}  "
        f"shape={tuple(truth.shape)}  param_names={test_ds.param_names}  "
        f"num_history_steps={num_history_steps}"
    )

    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(geometry[0, :, :])
    plt.subplot(1, 5, 2)
    plt.imshow(geometry[3, :, :])
    plt.subplot(1, 5, 3)
    plt.imshow(geometry[8, :, :])
    plt.subplot(1, 5, 4)
    plt.imshow(geometry[10, :, :])
    plt.subplot(1, 5, 5)
    plt.imshow(geometry[15, :, :])
    plt.savefig("lol.png")

    pred = _rollout(
        model=model,
        truth=truth,
        params=params,
        geometry=geometry,
        n_steps=T - 1,
        device=device,
        num_history_steps=num_history_steps,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"truth": truth, "pred": pred}, out_dir / "trajectory.pt")
    _plot_rollout(truth, pred, out_dir)
    _plot_params(params, test_ds.param_names, out_dir)

    # Resolved TKE: the fluctuation the rollout carries, which the velocity RMSE
    # does not separate from the mean flow.
    configured = cfg.get("tke_window")
    tke_window = _tke_window(T, None if configured is None else int(configured))
    window_label = f"{tke_window}-frame window" + (
        " = whole rollout" if tke_window == T else ""
    )
    tke = _tke_fields(truth, pred, test_ds.state_vars, tke_window)
    tke_error = None
    if tke is None:
        print(f"state_vars={test_ds.state_vars} carry no velocity; skipping TKE")
    else:
        tke_truth, tke_pred = tke
        tke_error = tke_pred - tke_truth
        fluid = geometry.numpy() > 0.5
        if not fluid.any():  # degenerate mask: score every cell, not none
            fluid = np.ones_like(fluid)
        scalars = _plot_tke_error(tke_truth, tke_pred, fluid, window_label, out_dir)
        print(
            f"TKE over the domain and rollout ({window_label}): "
            f"MAE={scalars['tke_mae']:.6f}  RMSE={scalars['tke_rmse']:.6f}  "
            f"bias={scalars['tke_bias']:+.6f}  "
            f"(truth mean k={scalars['tke_truth_mean']:.6f} m²/s²)"
        )

    anim_path = _animate_rollout(
        truth, pred, out_dir / "rollout.mp4", tke_error=tke_error
    )

    rmse = ((pred - truth) ** 2).mean().sqrt().item()
    print(f"overall RMSE={rmse:.6f}  outputs in {out_dir}  animation={anim_path.name}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/testing",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
