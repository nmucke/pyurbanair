"""Evaluate one or several trained steppers by rolling them out on test data.

Config: configs/surrogate/eval.yaml, block `stepper`.

    python scripts/surrogate/evaluate_stepper.py
    python scripts/surrogate/evaluate_stepper.py \
        'stepper.models=[model_weights/p3d_a,model_weights/p3d_b]' 'stepper.sample_indices=[0,1]'

Each model starts from a test trajectory's first state(s) and steps forward
with the trajectory's own parameters, for as many frames as the truth has.

Outputs, in `stepper.output_dir`:
  metrics.csv          RMSE, MAE, relative L2, final RMSE and rollout time per
                       model and sample, plus the mean over samples
  rmse.png             RMSE per rollout step, one panel per sample
  rmse_mean.png        the same averaged over the samples
  summary_metrics.png  the mean metrics as bars per model
  slices.png           |U| at a few times: truth, then prediction and error per
                       model (first sample)
  tke_slices.png       resolved TKE (time variance) at a few heights: truth,
                       then prediction and error per model (first sample)
  params.png           the trajectory's parameters, with inflow angle and speed
                       recovered at the inlet from the truth and from each model
  rollout.mp4          truth, prediction and error over time (if `animate`)
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Any

import hydra
import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from omegaconf import DictConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import (  # noqa: E402
    device,
    load_model,
    load_states,
    open_dataset,
    plot_slice_rows,
    speed,
    write_csv,
)


def run(cfg: DictConfig) -> None:
    ev = cfg.stepper
    dev = device(cfg.device)
    out = pathlib.Path(ev.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = {}
    for model_dir in ev.models:
        model, train_cfg = load_model(model_dir, dev)
        # Steppers that take geometry features compute them from the mask.
        data = open_dataset(train_cfg, cfg.data_dir, cfg.split, sdf_features="none")
        history = int(train_cfg.dataset.get("num_history_steps") or 1)
        models[pathlib.Path(model_dir).name] = (model, data, history)

    rows: list[dict] = []
    rmse_curves: dict[int, dict[str, np.ndarray]] = {}
    first: dict[str, Any] = {}
    for sample in ev.sample_indices:
        rmse_curves[sample] = {}
        for name, (model, data, history) in models.items():
            truth = _truth(data, sample, ev.max_steps)
            params = data._params[sample][: len(truth)]
            start = time.perf_counter()
            pred = _rollout(
                model, truth, params, data.geometry_for(sample), history, dev
            )
            seconds = time.perf_counter() - start
            error = pred - truth
            per_step = np.sqrt((error**2).mean(axis=(1, 2, 3, 4)))
            rmse_curves[sample][name] = per_step
            rows.append(
                {
                    "model": name,
                    "sample": sample,
                    "rmse": float(np.sqrt((error**2).mean())),
                    "mae": float(np.abs(error).mean()),
                    "rel_l2": float(np.linalg.norm(error) / np.linalg.norm(truth)),
                    "final_rmse": float(per_step[-1]),
                    "rollout_seconds": seconds,
                }
            )
            if sample == ev.sample_indices[0]:
                first.setdefault("truth", truth)
                first.setdefault("params", (data.param_names, params.numpy()))
                first.setdefault(
                    "fluid", data.geometry_for(sample).numpy().astype(bool)
                )
                first.setdefault("state_vars", list(data.state_vars))
                first.setdefault("preds", {})[name] = pred

    means = _mean_rows(rows, list(models))
    write_csv(rows + means, out / "metrics.csv")
    _plot_rmse(rmse_curves, out / "rmse.png", out / "rmse_mean.png")
    _plot_summary(means, out / "summary_metrics.png")
    _plot_slices(first, out / "slices.png")
    _plot_tke_slices(first, out / "tke_slices.png")
    _plot_params(first, out / "params.png")
    if ev.animate:
        _animate(first, out / "rollout.mp4")
    for row in means:
        print(f"{row['model']}: rmse={row['rmse']:.4g} rel_l2={row['rel_l2']:.4g}")
    print(f"Saved evaluation in {out}")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _truth(data: Any, sample: int, max_steps: int | None) -> np.ndarray:
    truth = load_states(data, sample)
    return truth if max_steps is None else truth[:max_steps]


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _rollout(
    model: Any,
    truth: np.ndarray,
    params: torch.Tensor,
    geometry: torch.Tensor,
    history: int,
    dev: torch.device,
) -> np.ndarray:
    """Step from the first `history` true frames to the trajectory's end.

    The first `history` frames of the result are the truth itself.
    """
    channels, grid = truth.shape[1], truth.shape[2:]
    pred = truth.copy()
    state = (
        torch.from_numpy(truth[:history]).reshape(1, history * channels, *grid).to(dev)
    )
    geom = geometry.unsqueeze(0).to(dev)
    for t in range(history - 1, len(truth) - 1):
        step = model(state, params[t].unsqueeze(0).to(dev), geom)
        pred[t + 1] = step[0].float().cpu().numpy()
        state = step if history == 1 else torch.cat([state[:, channels:], step], dim=1)
    return pred


def _mean_rows(rows: list[dict], names: list[str]) -> list[dict]:
    means = []
    for name in names:
        own = [r for r in rows if r["model"] == name]
        means.append(
            {"model": name, "sample": "mean"}
            | {
                k: float(np.mean([r[k] for r in own]))
                for k in own[0]
                if k not in ("model", "sample")
            }
        )
    return means


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_rmse(curves: dict, path: pathlib.Path, mean_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(
        1, len(curves), figsize=(5 * len(curves), 3.5), squeeze=False
    )
    for ax, (sample, per_model) in zip(axes[0], curves.items()):
        for name, curve in per_model.items():
            ax.plot(curve, label=name)
        ax.set_title(f"sample {sample}")
        ax.set_xlabel("rollout step")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in next(iter(curves.values())):
        stacked = [c[name] for c in curves.values()]
        n = min(len(c) for c in stacked)
        ax.plot(np.mean([c[:n] for c in stacked], axis=0), label=name)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(f"RMSE (mean of {len(curves)} samples)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(mean_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(means: list[dict], path: pathlib.Path) -> None:
    keys = ["rmse", "mae", "rel_l2", "final_rmse"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 3.2))
    for ax, key in zip(axes, keys):
        ax.bar([r["model"] for r in means], [r[key] for r in means], color="tab:blue")
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=45)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_slices(first: dict, path: pathlib.Path) -> None:
    """|U| at four times (lowest quarter of the domain): truth, then per model."""
    truth = first["truth"]
    z = truth.shape[2] // 4
    times = np.unique(np.linspace(0, len(truth) - 1, 4).astype(int))
    rows = [("truth", [speed(truth[t])[z] for t in times], False)]
    for name, pred in first["preds"].items():
        rows.append((name, [speed(pred[t])[z] for t in times], False))
        rows.append(
            (
                f"{name} - truth",
                [speed(pred[t])[z] - speed(truth[t])[z] for t in times],
                True,
            )
        )
    plot_slice_rows(rows, [f"step {t}" for t in times], first["fluid"][z], path)


def _plot_tke_slices(first: dict, path: pathlib.Path) -> None:
    """Resolved TKE (half the summed time variance) at three heights."""
    truth = first["truth"]
    levels = np.unique(np.linspace(0, truth.shape[2] - 2, 3).astype(int))

    def tke(fields: np.ndarray) -> np.ndarray:
        return 0.5 * fields.var(axis=0).sum(axis=0)

    reference = tke(truth)
    rows = [("truth", [reference[z] for z in levels], False)]
    for name, pred in first["preds"].items():
        k = tke(pred)
        rows.append((name, [k[z] for z in levels], False))
        rows.append((f"{name} - truth", [k[z] - reference[z] for z in levels], True))
    fluid = first["fluid"][levels]
    plot_slice_rows(rows, [f"z index {z}" for z in levels], fluid, path, "TKE [m²/s²]")


def _plot_params(first: dict, path: pathlib.Path) -> None:
    """Prescribed parameters; inflow angle and speed also recovered at the inlet."""
    names, values = first["params"]
    recovered = {"truth": _inlet_flow(first["truth"], first["state_vars"])}
    recovered |= {
        n: _inlet_flow(p, first["state_vars"]) for n, p in first["preds"].items()
    }
    fig, axes = plt.subplots(
        len(names), 1, figsize=(8, 2.6 * len(names)), squeeze=False
    )
    for i, (ax, name) in enumerate(zip(axes[:, 0], names)):
        ax.plot(values[:, i], color="k", label="prescribed")
        if name in ("inflow_angle", "velocity_magnitude"):
            for j, (source, flow) in enumerate(recovered.items()):
                style = dict(color="k") if source == "truth" else dict(color=f"C{j}")
                ax.plot(flow[name], ":", label=f"recovered, {source}", **style)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("step")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _inlet_flow(fields: np.ndarray, state_vars: list[str]) -> dict[str, np.ndarray]:
    """Inflow angle [deg] and speed from u, v at three probes across the inlet
    (second x cell, mid height), averaged over the probes."""
    u = fields[:, state_vars.index("u")]
    v = fields[:, state_vars.index("v")]
    nz, ny = u.shape[1], u.shape[2]
    ys = [int(f * (ny - 1)) for f in (0.2, 0.5, 0.8)]
    u_p, v_p = u[:, nz // 2, ys, 1], v[:, nz // 2, ys, 1]
    return {
        "inflow_angle": np.degrees(np.arctan2(v_p, u_p)).mean(axis=1),
        "velocity_magnitude": np.hypot(u_p, v_p).mean(axis=1),
    }


def _animate(first: dict, path: pathlib.Path) -> None:
    """|U| over time (lowest quarter of the domain): truth, prediction, error."""
    truth = first["truth"]
    z = truth.shape[2] // 4
    fluid = first["fluid"][z]
    panels = [("truth", speed(truth)[:, z])]
    for name, pred in first["preds"].items():
        panels += [
            (name, speed(pred)[:, z]),
            (f"|{name} - truth|", np.abs(speed(pred) - speed(truth))[:, z]),
        ]
    vmax = float(np.nanmax(panels[0][1]))
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.4 * len(panels), 3.2), squeeze=False
    )
    images = []
    for ax, (title, frames) in zip(axes[0], panels):
        cmap = "magma" if title.startswith("|") else "viridis"
        images.append(
            ax.imshow(
                np.where(fluid, frames[0], np.nan),
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    def update(t: int) -> list:
        for image, (_, frames) in zip(images, panels):
            image.set_data(np.where(fluid, frames[t], np.nan))
        fig.suptitle(f"step {t}")
        return images

    animation.FuncAnimation(fig, update, frames=len(truth)).save(path, fps=5, dpi=100)
    plt.close(fig)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs", config_name="surrogate/eval"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
