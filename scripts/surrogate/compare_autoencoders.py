"""Compare trained autoencoders on the same held-out snapshots.

Config: configs/surrogate/eval.yaml, block `autoencoder_compare`.

    python scripts/surrogate/compare_autoencoders.py \
        'autoencoder_compare.models=[model_weights/tadpole_ae_s,model_weights/tadpole_ae_b]'

Every model reconstructs the same snapshots (no random crop), drawn as
scripts/surrogate/evaluate_autoencoder.py draws them for the same `seed`,
`num_figures` and `num_metric_samples`: each model's metrics equal that
script's for it alone. The models must have been trained on the same state
variables and trajectories of the split; each reads its own geometry features.
Every metric is over fluid cells only.

Outputs, in `autoencoder_compare.output_dir`:
  metrics.csv                one row per model: overall RMSE, relative L2 and
                             PSNR; RMSE, MAE, relative L2 and R² per variable;
                             mean KL, active latent fraction, parameter count
  metrics.json               the same, per model as evaluate_autoencoder.py
  reconstruction_traj<i>_t<t>.png
                             |U| at a few heights: truth, then each model's
                             reconstruction and error (one figure per snapshot)
  per_channel_metrics.png    RMSE and relative L2 per variable, per model
  height_profile.png         |U| RMSE against normalised height, per model
  vertical_profiles_<nz>x<ny>x<nx>.png
                             per grid shape: mean and fluctuation RMS of each
                             variable and |U| per z level, truth and each model
  error_hist.png             histograms of the |U| error, per model
  pred_vs_true.png           reconstructed against true |U|, one panel per model
  latent_stats.png           KL per snapshot and the sorted spread per latent
                             channel, per model
"""

from __future__ import annotations

import copy
import pathlib
import sys
from typing import Any

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

import evaluate_autoencoder as ea  # noqa: E402
from eval_common import (  # noqa: E402
    device,
    load_model,
    load_states,
    open_dataset,
    plot_slice_rows,
    speed,
    write_csv,
    write_json,
)


def run(cfg: DictConfig) -> None:
    cc = cfg.autoencoder_compare
    if len(cc.models) < 2:
        raise ValueError("list at least two autoencoders in autoencoder_compare.models")
    tags = (
        [str(t) for t in cc.tags]
        if cc.tags is not None
        else [pathlib.Path(m).name for m in cc.models]
    )
    if len(tags) != len(cc.models) or len(set(tags)) != len(tags):
        raise ValueError(f"need one distinct tag per model, got {tags}")
    dev = device(cfg.device)
    out = pathlib.Path(cc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arch: dict[str, Any] = {}
    if cc.max_internal_batchsize is not None:
        arch["max_internal_batchsize"] = int(cc.max_internal_batchsize)

    models, datasets = {}, {}
    for tag, model_dir in zip(tags, cc.models):
        model, train_cfg = load_model(model_dir, dev, **arch)
        model.ae.latent_type = str(cc.latent_type)
        models[tag] = model
        datasets[tag] = open_dataset(
            train_cfg, cfg.data_dir, cfg.split, random_crop_size=None
        )
    data = datasets[tags[0]]
    names = list(data.state_vars)
    for tag, other in datasets.items():
        if list(other.state_vars) != names or [
            pathlib.Path(f).name for f in other._state_files
        ] != [pathlib.Path(f).name for f in data._state_files]:
            raise ValueError(
                f"{tag} was trained on other variables or trajectories than "
                f"{tags[0]}; set data_dir to compare them on one corpus"
            )

    # Reconstruction figures, drawn as evaluate_autoencoder.py draws them.
    rng = np.random.default_rng(int(cfg.seed))
    for traj in ea._spread(len(data._state_files), int(cc.num_figures)):
        t = int(rng.integers(data._traj_lengths[traj]))
        truth = load_states(data, traj, [t])
        recons = {
            tag: ea._reconstruct(models[tag], datasets[tag], traj, truth, dev)[0][0]
            for tag in tags
        }
        _plot_reconstructions(
            truth[0],
            recons,
            data,
            traj,
            int(cc.num_heights),
            out / f"reconstruction_traj{traj}_t{t}.png",
        )

    # Metrics on the same random snapshots, batched per trajectory (one grid).
    pairs = list(data.sample_index)
    chosen = rng.choice(
        len(pairs), min(int(cc.num_metric_samples), len(pairs)), replace=False
    )
    by_traj: dict[int, list[int]] = {}
    for i in sorted(chosen):
        by_traj.setdefault(pairs[i][0], []).append(pairs[i][1])

    # One copy of the generator per model: the same |U| values are kept.
    acc = {tag: ea._Metrics(len(names), copy.deepcopy(rng)) for tag in tags}
    profiles: dict[tuple[int, ...], dict[str, ea._Profiles]] = {}
    kl: dict[str, list[float]] = {tag: [] for tag in tags}
    latent_std: dict[str, list[np.ndarray]] = {tag: [] for tag in tags}
    for traj, times in by_traj.items():
        fluid = data.geometry_for(traj).numpy().astype(bool)
        grid = profiles.setdefault(
            fluid.shape,
            {tag: ea._Profiles(fluid.shape[0], len(names)) for tag in tags},
        )
        for start in range(0, len(times), int(cc.batch_size)):
            truth = load_states(data, traj, times[start : start + int(cc.batch_size)])
            for tag in tags:
                recon, batch_kl, latent = ea._reconstruct(
                    models[tag], datasets[tag], traj, truth, dev
                )
                kl[tag] += batch_kl.tolist()
                latent_std[tag].append(
                    np.moveaxis(latent, 1, 0).reshape(latent.shape[1], -1).std(axis=1)
                )
                for true_state, recon_state in zip(truth, recon):
                    acc[tag].add(true_state, recon_state, fluid)
                    grid[tag].add(true_state, recon_state, fluid)

    summaries, spreads, rows = {}, {}, []
    for tag, model_dir in zip(tags, cc.models):
        spreads[tag] = np.mean(latent_std[tag], axis=0)
        summary = acc[tag].summary(names) | {
            "mean_kl": float(np.mean(kl[tag])),
            "latent_active_fraction": float(np.mean(spreads[tag] > ea.ACTIVE_STD)),
            "num_snapshots": len(kl[tag]),
            "latent_type": str(cc.latent_type),
            "n_parameters": sum(p.numel() for p in models[tag].parameters()),
            "model_dir": str(model_dir),
        }
        summaries[tag] = summary
        rows.append(
            {"model": tag}
            | {k: summary[k] for k in ("rmse", "rel_l2", "psnr_db")}
            | {
                f"{key}_{name}": summary["per_variable"][name][key]
                for key in ("rmse", "mae", "rel_l2", "r2")
                for name in names
            }
            | {
                k: summary[k]
                for k in (
                    "mean_kl",
                    "latent_active_fraction",
                    "n_parameters",
                    "num_snapshots",
                )
            }
        )
        print(
            f"{tag}: rmse={summary['rmse']:.4g} rel_l2={summary['rel_l2']:.4g} "
            f"psnr={summary['psnr_db']:.1f} dB, active latents "
            f"{summary['latent_active_fraction']:.0%}"
        )
    write_csv(rows, out / "metrics.csv")
    write_json(summaries, out / "metrics.json")

    _plot_per_channel(summaries, names, out / "per_channel_metrics.png")
    _plot_height_profile(acc, out / "height_profile.png")
    for shape, grid in profiles.items():
        _plot_vertical_profiles(
            grid,
            names + ["|U|"],
            out / f"vertical_profiles_{'x'.join(map(str, shape))}.png",
        )
    _plot_error_hist(acc, out / "error_hist.png")
    _plot_pred_vs_true(acc, out / "pred_vs_true.png")
    _plot_latent_stats(kl, spreads, out / "latent_stats.png")
    print(f"Saved comparison in {out}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _colors(tags: list[str]) -> dict[str, Any]:
    return {tag: plt.get_cmap("tab10")(i) for i, tag in enumerate(tags)}


def _plot_reconstructions(
    truth: np.ndarray,
    recons: dict[str, np.ndarray],
    data: Any,
    traj: int,
    num_heights: int,
    path: pathlib.Path,
) -> None:
    fluid = data.geometry_for(traj).numpy().astype(bool)
    levels = ea._spread(fluid.shape[0], num_heights)
    true_speed = speed(truth)
    rows = [("truth", [true_speed[z] for z in levels], False)]
    for tag, recon in recons.items():
        recon_speed = speed(recon)
        rows += [
            (tag, [recon_speed[z] for z in levels], False),
            (f"{tag} - truth", [recon_speed[z] - true_speed[z] for z in levels], True),
        ]
    plot_slice_rows(rows, [f"z index {z}" for z in levels], fluid[levels], path)


def _plot_per_channel(
    summaries: dict[str, dict], names: list[str], path: pathlib.Path
) -> None:
    tags = list(summaries)
    colors = _colors(tags)
    width = 0.8 / len(tags)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key in zip(axes, ["rmse", "rel_l2"]):
        for i, tag in enumerate(tags):
            ax.bar(
                x + (i - (len(tags) - 1) / 2) * width,
                [summaries[tag]["per_variable"][n][key] for n in names],
                width,
                color=colors[tag],
                label=tag,
            )
        ax.set_xticks(x, names)
        ax.set_title(key)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_height_profile(acc: dict[str, ea._Metrics], path: pathlib.Path) -> None:
    colors = _colors(list(acc))
    heights = (np.arange(ea.HEIGHT_BINS) + 0.5) / ea.HEIGHT_BINS
    fig, ax = plt.subplots(figsize=(4, 4.5))
    for tag, metrics in acc.items():
        filled = metrics.height_count > 0
        ax.plot(
            np.sqrt(metrics.height_sse[filled] / metrics.height_count[filled]),
            heights[filled],
            "o-",
            color=colors[tag],
            label=tag,
        )
    ax.set_xlabel("|U| RMSE [m/s]")
    ax.set_ylabel("normalised height")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_vertical_profiles(
    grid: dict[str, ea._Profiles], names: list[str], path: pathlib.Path
) -> None:
    colors = _colors(list(grid))
    stats = {tag: profile.mean_and_rms() for tag, profile in grid.items()}
    first = next(iter(grid.values()))
    levels = np.arange(len(first.count))
    fig, axes = plt.subplots(
        2, len(names), figsize=(3 * len(names), 6), sharey=True, squeeze=False
    )
    for c, name in enumerate(names):
        for row, kind in enumerate(("mean", "fluctuation RMS")):
            ax = axes[row, c]
            # The truth is the same in every model's sums.
            ax.plot(stats[next(iter(grid))][row][0, c], levels, "k--", label="truth")
            for tag, values in stats.items():
                ax.plot(values[row][1, c], levels, color=colors[tag], label=tag)
            ax.set_title(f"{name} {kind}")
            ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("z index")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"fluid cells, {first.snapshots} snapshots")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_error_hist(acc: dict[str, ea._Metrics], path: pathlib.Path) -> None:
    colors = _colors(list(acc))
    errors = {
        tag: np.concatenate(m.pred_speed) - np.concatenate(m.true_speed)
        for tag, m in acc.items()
    }
    bins = np.linspace(
        min(e.min() for e in errors.values()),
        max(e.max() for e in errors.values()),
        81,
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for tag, error in errors.items():
        ax.hist(error, bins=bins, histtype="step", color=colors[tag], label=tag)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("|U| reconstruction - truth [m/s]")
    ax.set_ylabel("cells")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_vs_true(acc: dict[str, ea._Metrics], path: pathlib.Path) -> None:
    colors = _colors(list(acc))
    pairs = {
        tag: (np.concatenate(m.true_speed), np.concatenate(m.pred_speed))
        for tag, m in acc.items()
    }
    lim = [0, float(max(max(t.max(), p.max()) for t, p in pairs.values()))]
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(4.5 * len(pairs), 4.5), squeeze=False
    )
    for ax, (tag, (true_speed, pred_speed)) in zip(axes[0], pairs.items()):
        ax.scatter(true_speed, pred_speed, s=1, alpha=0.2, color=colors[tag])
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(tag)
        ax.set_xlabel("true |U| [m/s]")
    axes[0, 0].set_ylabel("reconstructed |U| [m/s]")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_stats(
    kl: dict[str, list[float]], spreads: dict[str, np.ndarray], path: pathlib.Path
) -> None:
    colors = _colors(list(kl))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for tag in kl:
        axes[0].hist(kl[tag], bins=30, histtype="step", color=colors[tag], label=tag)
        axes[1].plot(np.sort(spreads[tag])[::-1], color=colors[tag], label=tag)
    axes[0].set_xlabel("KL per snapshot")
    axes[0].legend(fontsize=8)
    axes[1].axhline(ea.ACTIVE_STD, color="k", ls="--", lw=0.8, label="active threshold")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("latent channel (sorted)")
    axes[1].set_ylabel("std (mode latent)")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs", config_name="surrogate/eval"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
