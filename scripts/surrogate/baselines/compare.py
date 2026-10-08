"""Compare trained steppers fairly: same trajectories, same start, fluid cells
only, against persistence, with turbulence statistics.

Config: configs/surrogate/baselines/compare.yaml.

    python scripts/surrogate/baselines/compare.py \
        'models=[model_weights/local_fno,model_weights/p3d_idealized]'

Every model starts from the same frame, the latest any model needs as history
(``max(H) - 1``), each from its own last ``H`` true frames up to it, and rolls
out to the trajectory's end with the trajectory's parameters. The models must
have been trained on the same data: the same training-data folder name, state
and parameter variables and, where the corpus' config.yaml is readable, the
same output frequency; the script refuses to compare them otherwise. Models are
labelled by their folder name (by the path given, if two names clash). Persistence (repeat the start
frame) is the reference every model must beat.

Outputs, in `output_dir`:
  metrics.csv      per model and sample, plus the mean: fluid-masked RMSE, MAE
                   and final RMSE, and the RMSE relative to persistence
  per_step.csv     fluid-masked RMSE and MAE per lead time (mean over samples),
                   and each velocity component's spatial std relative to the truth
  statistics.csv   per model and height: mean-u, TKE and -<u'w'> relative to the
                   truth, over the rollout (first sample)
  rmse.png         per-step RMSE against lead time
  energy.png       spatial std relative to the truth against lead time
  spectra.png      lateral (y) spectrum of u' at `levels` (first sample)
  profiles.png     mean u, TKE and -<u'w'> against height (first sample)
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from neural_surrogate_baselines import diagnostics
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "utils"))

from eval_common import (  # noqa: E402
    device,
    grid_spacing,
    load_model,
    load_states,
    open_dataset,
    write_csv,
)

PERSISTENCE = "persistence"


def run(cfg: DictConfig) -> None:
    dev = device(cfg.device)
    out = pathlib.Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.models:
        raise ValueError("list at least one trained model in `models`")
    if cfg.max_steps is not None and int(cfg.max_steps) < 1:
        raise ValueError(f"max_steps must be at least 1, got {cfg.max_steps}")

    labels = _labels(list(cfg.models))
    _check_same_data(
        {
            label: OmegaConf.load(pathlib.Path(d) / "config.yaml")
            for label, d in labels.items()
        }
    )
    models = {}
    for label, model_dir in labels.items():
        models[label] = load_model(model_dir, dev)
    first_cfg = next(iter(models.values()))[1]
    data = open_dataset(first_cfg, cfg.data_dir, cfg.split, sdf_features="none")
    histories = {
        name: int(c.dataset.get("num_history_steps") or 1)
        for name, (_, c) in models.items()
    }
    start = max(histories.values()) - 1
    state_vars = list(data.state_vars)
    u, w = state_vars.index("u"), state_vars.index("w")
    velocity = [state_vars.index(v) for v in ("u", "v", "w") if v in state_vars]

    rows: list[dict] = []
    dy = 1.0
    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    first: dict[str, Any] = {}
    for sample in cfg.sample_indices:
        truth = load_states(data, sample)
        if cfg.max_steps is not None:
            truth = truth[: start + 1 + int(cfg.max_steps)]
        params = data._params[sample][: len(truth)]
        geometry = data.geometry_for(sample)
        fluid = geometry.numpy().astype(bool)
        preds = {
            name: _rollout(model, truth, params, geometry, histories[name], start, dev)
            for name, (model, _) in models.items()
        }
        if cfg.persistence:
            preds[PERSISTENCE] = diagnostics.persistence(truth, start)
        lead = slice(start + 1, None)
        truth_std = diagnostics.spatial_std(truth[lead], fluid)
        reference = None
        if cfg.persistence:
            reference = diagnostics.masked_errors(
                preds[PERSISTENCE][lead], truth[lead], fluid
            )[0]
        for name, pred in preds.items():
            rmse, mae = diagnostics.masked_errors(pred[lead], truth[lead], fluid)
            std_ratio = diagnostics.spatial_std(pred[lead], fluid) / truth_std
            curves.setdefault(name, []).append((rmse, mae, std_ratio))
            row: dict[str, Any] = {
                "model": name,
                "sample": sample,
                "rmse": float(np.sqrt((rmse**2).mean())),
                "mae": float(mae.mean()),
                "final_rmse": float(rmse[-1]),
            }
            if reference is not None:
                row["rmse_vs_persistence"] = row["rmse"] / float(
                    np.sqrt((reference**2).mean())
                )
            rows.append(row)
        if not first:
            first = {"truth": truth[lead], "fluid": fluid} | {
                name: pred[lead] for name, pred in preds.items()
            }
            dy = grid_spacing(data, sample)[1]

    names = list(curves)
    write_csv(rows + _means(rows, names), out / "metrics.csv")
    write_csv(_per_step(curves, state_vars), out / "per_step.csv")
    write_csv(_statistics(first, names, u, w, velocity), out / "statistics.csv")
    _plot_curves(curves, out / "rmse.png", out / "energy.png", state_vars)
    levels = [lv for lv in cfg.levels if 0 <= lv < first["fluid"].shape[0]]
    if levels:
        _plot_spectra(first, names, levels, u, dy, out / "spectra.png")
    else:
        print(f"no `levels` inside the grid's {first['fluid'].shape[0]}: no spectra")
    _plot_profiles(first, names, u, w, velocity, out / "profiles.png")
    print(f"Saved comparison in {out}")


def _labels(model_dirs: list[str]) -> dict[str, str]:
    """A unique label per model: its folder name, or the path if names clash."""
    names = [pathlib.Path(d).name for d in model_dirs]
    if len(set(model_dirs)) != len(model_dirs):
        raise ValueError(f"a model is listed twice: {model_dirs}")
    labels = names if len(set(names)) == len(names) else [str(d) for d in model_dirs]
    reserved = {"truth", "fluid", PERSISTENCE} & set(labels)
    if reserved:
        raise ValueError(
            f"model labels {sorted(reserved)} are reserved; rename the folder"
        )
    return dict(zip(labels, model_dirs))


def _check_same_data(train_cfgs: dict[str, Any]) -> None:
    """Refuse models trained on different data.

    Compares the training-data folder name (paths may differ between machines),
    the state and parameter variables, and the corpus' output frequency where
    its config.yaml can be read. A root that does not resolve on this machine
    is left out of the folder comparison rather than compared as raw text.
    """

    def signature(cfg: Any) -> dict[str, Any]:
        dataset = cfg.dataset
        try:
            root = pathlib.Path(str(OmegaConf.select(cfg, "dataset.root_dir")))
        except Exception:
            root = None
        frequency = None
        if root is not None and (root / "config.yaml").exists():
            frequency = OmegaConf.select(
                OmegaConf.load(root / "config.yaml"), "time.output_frequency"
            )
        return {
            "data": root.name if root is not None else None,
            "output_frequency": frequency,
            "state_vars": list(dataset.state_vars),
            "param_vars": list(dataset.get("param_vars") or []),
        }

    signatures = {name: signature(cfg) for name, cfg in train_cfgs.items()}
    different = False
    for key in ("data", "output_frequency", "state_vars", "param_vars"):
        known = {str(s[key]) for s in signatures.values() if s[key] is not None}
        different |= len(known) > 1
    if different:
        raise ValueError(
            "models trained on different data:\n  "
            + "\n  ".join(f"{n}: {s}" for n, s in signatures.items())
        )


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _rollout(
    model: Any,
    truth: np.ndarray,
    params: torch.Tensor,
    geometry: torch.Tensor,
    history: int,
    start: int,
    dev: torch.device,
) -> np.ndarray:
    """Step from frame `start` (and the `history - 1` before it) to the end."""
    channels, grid = truth.shape[1], truth.shape[2:]
    pred = truth.copy()
    window = truth[start - history + 1 : start + 1]
    state = torch.from_numpy(window).reshape(1, history * channels, *grid).to(dev)
    geom = geometry.unsqueeze(0).to(dev)
    for t in range(start, len(truth) - 1):
        step = model(state, params[t].unsqueeze(0).to(dev), geom)
        pred[t + 1] = step[0].float().cpu().numpy()
        state = step if history == 1 else torch.cat([state[:, channels:], step], dim=1)
    return pred


def _means(rows: list[dict], names: list[str]) -> list[dict]:
    means = []
    for name in names:
        own = [r for r in rows if r["model"] == name]
        keys = [k for k in own[0] if k not in ("model", "sample")]
        means.append(
            {"model": name, "sample": "mean"}
            | {k: float(np.mean([r[k] for r in own])) for k in keys}
        )
    return means


def _per_step(curves: dict, state_vars: list[str]) -> list[dict]:
    rows = []
    for name, per_sample in curves.items():
        n = min(len(c[0]) for c in per_sample)
        rmse = np.mean([c[0][:n] for c in per_sample], axis=0)
        mae = np.mean([c[1][:n] for c in per_sample], axis=0)
        std = np.mean([c[2][:n] for c in per_sample], axis=0)
        for step in range(n):
            rows.append(
                {
                    "model": name,
                    "lead_step": step + 1,
                    "rmse": rmse[step],
                    "mae": mae[step],
                }
                | {f"std_ratio_{v}": std[step, c] for c, v in enumerate(state_vars)}
            )
    return rows


def _statistics(
    first: dict, names: list[str], u: int, w: int, velocity: list[int]
) -> list[dict]:
    truth = diagnostics.profiles(first["truth"], first["fluid"], u, w, velocity)
    rows = []
    for name in names:
        pred = diagnostics.profiles(first[name], first["fluid"], u, w, velocity)
        for level in range(len(truth["tke"])):
            rows.append(
                {"model": name, "level": level}
                | {
                    f"{k}_ratio": (
                        float(pred[k][level] / truth[k][level])
                        if truth[k][level] != 0
                        else float("nan")
                    )
                    for k in truth
                }
            )
    return rows


def _plot_curves(
    curves: dict,
    rmse_path: pathlib.Path,
    energy_path: pathlib.Path,
    state_vars: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, per_sample in curves.items():
        n = min(len(c[0]) for c in per_sample)
        ax.plot(
            np.arange(1, n + 1),
            np.mean([c[0][:n] for c in per_sample], axis=0),
            label=name,
        )
    ax.set_xlabel("lead time (steps)")
    ax.set_ylabel("RMSE over fluid cells")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(rmse_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(state_vars), figsize=(4 * len(state_vars), 3.2), squeeze=False
    )
    for c, (ax, var) in enumerate(zip(axes[0], state_vars)):
        for name, per_sample in curves.items():
            n = min(len(s[2]) for s in per_sample)
            ax.plot(
                np.arange(1, n + 1),
                np.mean([s[2][:n, c] for s in per_sample], axis=0),
                label=name,
            )
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_title(f"spatial std of {var} / truth")
        ax.set_xlabel("lead time (steps)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.savefig(energy_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_spectra(
    first: dict,
    names: list[str],
    levels: list[int],
    u: int,
    dy: float,
    path: pathlib.Path,
) -> None:
    fluid = first["fluid"]
    levels = [lv for lv in levels if lv < fluid.shape[0]]
    fig, axes = plt.subplots(
        1, len(levels), figsize=(4.5 * len(levels), 3.5), squeeze=False
    )
    for ax, level in zip(axes[0], levels):
        for name in ["truth", *names]:
            prime = diagnostics.fluctuations(first[name], fluid)[:, u, level]
            k, energy = diagnostics.lateral_spectrum(prime, fluid[level], dy)
            style = {"color": "k", "lw": 2} if name == "truth" else {}
            ax.loglog(k[1:], energy[1:], label=name, **style)
        ax.set_title(f"u' lateral spectrum, level {level}")
        ax.set_xlabel("k_y (rad/m)")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_profiles(
    first: dict,
    names: list[str],
    u: int,
    w: int,
    velocity: list[int],
    path: pathlib.Path,
) -> None:
    labels = {"mean_u": "mean u", "tke": "resolved TKE", "uw": "-<u'w'>"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for name in ["truth", *names]:
        prof = diagnostics.profiles(first[name], first["fluid"], u, w, velocity)
        style = {"color": "k", "lw": 2} if name == "truth" else {}
        for ax, key in zip(axes, labels):
            ax.plot(prof[key], np.arange(len(prof[key])), label=name, **style)
    for ax, key in zip(axes, labels):
        ax.set_xlabel(labels[key])
        ax.set_ylabel("level")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None,
    config_path="../../../configs",
    config_name="surrogate/baselines/compare",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
