"""Compare several trained neural surrogates on the same test trajectories.

For every model listed in the config this rebuilds the architecture + dataset
from its saved ``model_weights/<name>/config.yaml``, restores ``weights.pt``,
and autoregressively rolls it out on a shared set of test trajectories (the
same initial conditions and ground truth for every model, so the comparison is
apples-to-apples). It then writes side-by-side plots and a metrics table.

This is the multi-model sibling of ``test_neural_surrogate.py`` -- the plots
here are the same diagnostics, restructured so models overlay / stack instead
of standing alone.

Outputs land in ``output_dir`` (default ``model_comparison/``):

| File                         | Contents |
|------------------------------|----------|
| ``metrics.csv``              | per-(model, sample) + aggregate RMSE / MAE / rel-L2 / timing |
| ``per_step_rmse_sample_*.png`` | per-step rollout RMSE, all models overlaid, one per sample |
| ``per_step_rmse_mean.png``   | per-step RMSE averaged across samples, all models overlaid |
| ``summary_metrics.png``      | bar charts of the aggregate metrics per model |
| ``slices_sample_*.png``      | ``|u|`` slice grid: truth row + pred/|err| rows per model |
| ``rollout_sample_*.mp4``     | stacked truth/pred/|err| animation, one row per model (if ``animate``) |

Usage:

    pixi run -e dev python scripts/neural_surrogate/compare_surrogate_models.py
    pixi run -e dev python scripts/neural_surrogate/compare_surrogate_models.py \
        data.root_dir=training_data/pyudales_idealized \
        'data.sample_indices=[0,1,2]' data.max_steps=100
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from neural_surrogates import TransitionDataset
from omegaconf import DictConfig, OmegaConf


class LoadedModel:
    """A trained surrogate plus the dataset it is evaluated on."""

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        dataset: TransitionDataset,
        dtype: torch.dtype,
    ) -> None:
        self.name = name
        self.model = model
        self.dataset = dataset
        self.dtype = dtype
        self.n_params = sum(p.numel() for p in model.parameters())


def _load_model(
    name: str,
    model_dir: Path,
    data_root: str | None,
    split: str,
    device: torch.device,
) -> LoadedModel:
    """Rebuild architecture + dataset from a model dir and restore weights.

    ``data_root`` overrides the dataset ``root_dir`` from the training config so
    every model can be pointed at one shared evaluation dataset; ``None`` keeps
    each model's own training dataset. SDF features are forced off -- the rollout
    only needs the geometry mask (SDF-consuming models self-compute the features
    at inference), and skipping the EDT keeps dataset construction fast.
    """
    train_cfg = OmegaConf.load(model_dir / "config.yaml")
    dtype = getattr(torch, train_cfg.dataset.dtype)

    ds_overrides: dict = dict(split=split, dtype=dtype, sdf_features="none")
    if data_root is not None:
        ds_overrides["root_dir"] = data_root
    dataset = instantiate(train_cfg.dataset, **ds_overrides)

    model = (
        instantiate(
            train_cfg.architecture,
            n_state_channels=len(train_cfg.dataset.state_vars),
            n_params=len(dataset.param_names),
        )
        .to(dtype=dtype)
        .to(device)
    )
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location=device))
    model.eval()

    root = data_root if data_root is not None else train_cfg.dataset.root_dir
    print(
        f"[{name}] loaded {train_cfg.architecture._target_.split('.')[-1]} "
        f"({sum(p.numel() for p in model.parameters()):,} params) "
        f"on {Path(str(root)).name}/{split}"
    )
    return LoadedModel(name, model, dataset, dtype)


def _load_trajectory(
    dataset: TransitionDataset, sample_idx: int, dtype: torch.dtype
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


def _per_step_rmse(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """RMSE at each rollout step, over all channels and cells."""
    return ((pred - truth) ** 2).mean(dim=tuple(range(1, pred.ndim))).sqrt().numpy()


def _metrics(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    diff = pred - truth
    rmse = diff.pow(2).mean().sqrt().item()
    mae = diff.abs().mean().item()
    rel_l2 = (diff.norm() / truth.norm().clamp_min(1e-12)).item()
    per_step = _per_step_rmse(pred, truth)
    return {
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
        "final_rmse": float(per_step[-1]),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_per_step_rmse(
    per_step: dict[str, np.ndarray], out_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in per_step.items():
        ax.plot(curve, label=name, linewidth=1.8)
    ax.set_xlabel("time step")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_slice_comparison(
    truth: torch.Tensor,
    preds: dict[str, torch.Tensor],
    out_path: Path,
    z_idx: int,
    n_show: int = 5,
) -> None:
    """Grid of ``|u|`` slices: a truth row, then a pred + |err| row per model.

    Columns are evenly spaced rollout times; magnitude panels share the truth
    color scale and error panels share their own so models are directly
    comparable panel-to-panel.
    """
    T = truth.shape[0]
    times = np.linspace(0, T - 1, min(n_show, T), dtype=int)
    mag_t = truth.norm(dim=1)[:, z_idx]
    mag_p = {name: p.norm(dim=1)[:, z_idx] for name, p in preds.items()}
    vmax = mag_t.max().item()
    err_max = max((mag_p[n] - mag_t).abs().max().item() for n in preds) or 1.0

    rows: list[tuple[str, torch.Tensor, float, str]] = [
        ("truth", mag_t, vmax, "viridis")
    ]
    for name in preds:
        rows.append((f"{name}\npred", mag_p[name], vmax, "viridis"))
        rows.append((f"{name}\n|err|", (mag_p[name] - mag_t).abs(), err_max, "magma"))

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, len(times), figsize=(2.6 * len(times), 2.6 * n_rows), squeeze=False
    )
    for r, (label, data, vm, cmap) in enumerate(rows):
        for c, t in enumerate(times):
            im = axes[r, c].imshow(
                data[t].numpy(), origin="lower", vmin=0, vmax=vm, cmap=cmap
            )
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"t={t}")
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=9)
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(f"|u| rollout comparison at z-index {z_idx}")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_metrics(summary: dict[str, dict[str, float]], out_path: Path) -> None:
    """Bar charts of aggregate per-model metrics (lower is better except size)."""
    names = list(summary.keys())
    metrics = ["rmse", "mae", "rel_l2", "final_rmse", "ms_per_step", "n_params_M"]
    titles = {
        "rmse": "overall RMSE",
        "mae": "overall MAE",
        "rel_l2": "relative L2 error",
        "final_rmse": "final-step RMSE",
        "ms_per_step": "inference (ms / step)",
        "n_params_M": "model size (M params)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    x = np.arange(len(names))
    for ax, metric in zip(axes.ravel(), metrics):
        vals = [summary[n][metric] for n in names]
        ax.bar(x, vals, color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(titles[metric])
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Model comparison — aggregate metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _animate_comparison(
    truth: torch.Tensor,
    preds: dict[str, torch.Tensor],
    out_path: Path,
    z_idx: int,
    fps: int = 10,
) -> Path:
    """One row per model: truth | pred | |err| at a fixed z-slice, over time."""
    T = truth.shape[0]
    mag_t = truth.norm(dim=1)[:, z_idx].numpy()
    vmax = float(mag_t.max()) or 1.0
    rows = []
    for name, p in preds.items():
        mp = p.norm(dim=1)[:, z_idx].numpy()
        me = np.abs(mp - mag_t)
        rows.append((name, mp, me, float(me.max()) or 1.0))

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(11, 3.6 * n_rows), constrained_layout=True, squeeze=False
    )
    ims = []
    for r, (name, mp, me, emax) in enumerate(rows):
        im_t = axes[r, 0].imshow(mag_t[0], origin="lower", vmin=0, vmax=vmax)
        im_p = axes[r, 1].imshow(mp[0], origin="lower", vmin=0, vmax=vmax)
        im_e = axes[r, 2].imshow(me[0], origin="lower", vmin=0, vmax=emax, cmap="magma")
        ims.append((im_t, im_p, im_e, mp, me))
        for col, title in enumerate(["truth", "pred", "|err|"]):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
            if r == 0:
                axes[r, col].set_title(title)
        axes[r, 0].set_ylabel(name, fontsize=9)
        fig.colorbar(im_e, ax=axes[r, 2], fraction=0.046)

    suptitle = fig.suptitle(f"|u| at z={z_idx}  t=0/{T - 1}")

    def update(frame: int) -> list:
        artists: list = [suptitle]
        for im_t, im_p, im_e, mp, me in ims:
            im_t.set_array(mag_t[frame])
            im_p.set_array(mp[frame])
            im_e.set_array(me[frame])
            artists.extend([im_t, im_p, im_e])
        suptitle.set_text(f"|u| at z={z_idx}  t={frame}/{T - 1}")
        return artists

    if animation.writers.is_available("ffmpeg"):
        writer: animation.AbstractMovieWriter = animation.FFMpegWriter(fps=fps)
        save_path = out_path
    else:
        writer = animation.PillowWriter(fps=fps)
        save_path = out_path.with_suffix(".gif")
    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(str(save_path), writer=writer, dpi=110)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
def run(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable — falling back to cpu")
        device = torch.device("cpu")

    data_root = cfg.data.root_dir
    split = cfg.data.split
    sample_indices = list(cfg.data.sample_indices)
    max_steps = cfg.data.max_steps

    models = [
        _load_model(
            name=str(m.get("name", Path(m.dir).name)),
            model_dir=Path(m.dir),
            data_root=data_root,
            split=split,
            device=device,
        )
        for m in cfg.models
    ]
    if data_root is None:
        roots = {
            str(OmegaConf.load(Path(m.dir) / "config.yaml").dataset.root_dir)
            for m in cfg.models
        }
        if len(roots) > 1:
            print(
                "WARNING: data.root_dir is null and the models were trained on "
                f"different datasets ({roots}); they are NOT being compared on "
                "the same trajectories. Set data.root_dir to force a shared set."
            )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_step[sample_idx][model_name] -> per-step RMSE curve
    per_step: dict[int, dict[str, np.ndarray]] = {}
    # rows accumulated for metrics.csv
    rows: list[dict] = []
    # accumulate for aggregate summary: model_name -> list of metric dicts
    agg: dict[str, list[dict[str, float]]] = {m.name: [] for m in models}
    timing: dict[str, list[float]] = {m.name: [] for m in models}

    for sample_idx in sample_indices:
        per_step[sample_idx] = {}
        ref_truth: torch.Tensor | None = None
        preds: dict[str, torch.Tensor] = {}

        for m in models:
            if sample_idx >= len(m.dataset._state_files):
                print(f"[{m.name}] no sample {sample_idx} in split — skipping")
                continue
            truth, params, geometry = _load_trajectory(m.dataset, sample_idx, m.dtype)
            if max_steps is not None:
                truth = truth[:max_steps]
                params = params[:max_steps]
            T = truth.shape[0]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = _rollout(
                model=m.model,
                initial_state=truth[0],
                params=params,
                geometry=geometry,
                n_steps=T - 1,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            ms_per_step = 1e3 * elapsed / max(T - 1, 1)
            timing[m.name].append(ms_per_step)

            met = _metrics(pred, truth)
            per_step[sample_idx][m.name] = _per_step_rmse(pred, truth)
            preds[m.name] = pred
            agg[m.name].append(met)
            rows.append(
                dict(
                    model=m.name,
                    sample=sample_idx,
                    ms_per_step=ms_per_step,
                    n_params_M=m.n_params / 1e6,
                    **met,
                )
            )
            print(
                f"[{m.name}] sample {sample_idx}: rmse={met['rmse']:.5f} "
                f"rel_l2={met['rel_l2']:.4f} final_rmse={met['final_rmse']:.5f} "
                f"({ms_per_step:.1f} ms/step)"
            )
            if ref_truth is None:
                ref_truth = truth

        if ref_truth is None or not preds:
            continue

        z_idx = ref_truth.shape[-3] // 2
        _plot_per_step_rmse(
            per_step[sample_idx],
            out_dir / f"per_step_rmse_sample_{sample_idx}.png",
            f"Per-step rollout RMSE — sample {sample_idx}",
        )
        _plot_slice_comparison(
            ref_truth, preds, out_dir / f"slices_sample_{sample_idx}.png", z_idx
        )
        if cfg.animate:
            path = _animate_comparison(
                ref_truth, preds, out_dir / f"rollout_sample_{sample_idx}.mp4", z_idx
            )
            print(f"  animation -> {path.name}")

    # ---- aggregate across samples ---------------------------------------- #
    summary: dict[str, dict[str, float]] = {}
    for m in models:
        mets = agg[m.name]
        if not mets:
            continue
        summary[m.name] = {
            "rmse": float(np.mean([x["rmse"] for x in mets])),
            "mae": float(np.mean([x["mae"] for x in mets])),
            "rel_l2": float(np.mean([x["rel_l2"] for x in mets])),
            "final_rmse": float(np.mean([x["final_rmse"] for x in mets])),
            "ms_per_step": float(np.mean(timing[m.name])),
            "n_params_M": m.n_params / 1e6,
        }
        rows.append(dict(model=m.name, sample="mean", **summary[m.name]))

    # mean per-step RMSE across samples (truncate to shortest for safety)
    mean_curves: dict[str, np.ndarray] = {}
    for m in models:
        curves = [per_step[s][m.name] for s in sample_indices if m.name in per_step[s]]
        if not curves:
            continue
        L = min(len(c) for c in curves)
        mean_curves[m.name] = np.mean([c[:L] for c in curves], axis=0)
    if mean_curves:
        _plot_per_step_rmse(
            mean_curves,
            out_dir / "per_step_rmse_mean.png",
            "Per-step rollout RMSE — mean over samples",
        )
    if summary:
        _plot_summary_metrics(summary, out_dir / "summary_metrics.png")

    # ---- metrics.csv ----------------------------------------------------- #
    fieldnames = [
        "model",
        "sample",
        "rmse",
        "mae",
        "rel_l2",
        "final_rmse",
        "ms_per_step",
        "n_params_M",
    ]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\ncomparison of {len(summary)} model(s) written to {out_dir}")
    for name, s in summary.items():
        print(f"  {name}: rmse={s['rmse']:.5f}  rel_l2={s['rel_l2']:.4f}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/comparison",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
