"""Helpers shared by the scripts_new/surrogate/evaluate_*.py scripts.

Loading a trained model and its data, reading states, and the one figure type
they all use: rows of horizontal slices (truth, prediction, error, ...).
Fields are (..., C, z, y, x) arrays with C the state variables (u, v, w).
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def device(name: str) -> torch.device:
    """The requested device, or the CPU when CUDA is not available."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; evaluating on the CPU")
        return torch.device("cpu")
    return torch.device(name)


def load_model(
    model_dir: str | pathlib.Path, dev: torch.device, **arch_kwargs: Any
) -> tuple[Any, DictConfig]:
    """A trained model rebuilt from `config.yaml` + `weights.pt`, in eval mode."""
    model_dir = pathlib.Path(model_dir)
    cfg = OmegaConf.load(model_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    if cfg.dataset.get("param_vars") is not None:  # steppers and the generator
        arch_kwargs.setdefault("n_params", len(cfg.dataset.param_vars))
    model = instantiate(
        cfg.architecture, n_state_channels=len(cfg.dataset.state_vars), **arch_kwargs
    )
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu"))
    return model.to(dev).eval(), cfg


def open_dataset(
    train_cfg: DictConfig, data_dir: str | None, split: str, **overrides: Any
) -> Any:
    """The model's training dataset class on `split` (of `data_dir`, if given)."""
    if data_dir is not None:
        overrides["root_dir"] = data_dir
    dtype = getattr(torch, train_cfg.dataset.dtype)
    return instantiate(train_cfg.dataset, split=split, dtype=dtype, **overrides)


def load_states(
    dataset: Any, traj: int, times: Sequence[int] | None = None
) -> np.ndarray:
    """(T, C, z, y, x) states of one trajectory (all frames, or `times`)."""
    with xarray.open_dataset(dataset._state_files[traj]) as ds:
        if times is not None:
            ds = ds.isel(time=list(times))
        return np.stack([ds[v].values for v in dataset.state_vars], axis=1).astype(
            np.float32
        )


def grid_spacing(dataset: Any, traj: int) -> tuple[float, float, float]:
    """(dz, dy, dx) of one trajectory's grid, from its cell-centre coordinates."""
    with xarray.open_dataset(dataset._state_files[traj]) as ds:
        dims = [d for d in ds[dataset.state_vars[0]].dims if d != "time"]
        dz, dy, dx = (float(np.median(np.diff(ds[d].values))) for d in dims)
    return dz, dy, dx


def expand(
    t: torch.Tensor | None, batch: int, dev: torch.device
) -> torch.Tensor | None:
    """Repeat a per-trajectory tensor (geometry, features) over a batch."""
    return None if t is None else t.unsqueeze(0).expand(batch, *t.shape).to(dev)


def speed(fields: np.ndarray) -> np.ndarray:
    """|U| from (..., C, z, y, x) fields: the norm over the C axis."""
    return np.sqrt((np.asarray(fields) ** 2).sum(axis=-4))


def plot_slice_rows(
    rows: Sequence[tuple[str, Sequence[np.ndarray], bool]],
    column_titles: Sequence[str],
    fluid: np.ndarray,
    path: pathlib.Path,
    label: str = "|U| [m/s]",
) -> None:
    """A grid of 2-D (y, x) slices: one row per entry, one column per title.

    Each row is (name, slices, is_error). Value rows share one colour scale;
    error rows get their own. Building cells (`fluid` False) are blanked.
    """
    values = [s for _, slices, err in rows if not err for s in slices]
    errors = [s for _, slices, err in rows if err for s in slices]
    vlim = (np.nanmin(values), np.nanmax(values)) if values else (0.0, 1.0)
    elim = np.nanmax([np.nanmax(np.abs(e)) for e in errors]) if errors else 1.0
    n_rows, n_cols = len(rows), len(column_titles)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows), squeeze=False
    )
    for r, (name, slices, is_error) in enumerate(rows):
        for c, field in enumerate(slices):
            masked = np.where(fluid[c] if fluid.ndim == 3 else fluid, field, np.nan)
            if is_error:
                im = axes[r, c].imshow(
                    masked, origin="lower", cmap="RdBu_r", vmin=-elim, vmax=elim
                )
            else:
                im = axes[r, c].imshow(
                    masked, origin="lower", vmin=vlim[0], vmax=vlim[1]
                )
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(column_titles[c], fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=9)
        fig.colorbar(
            im, ax=axes[r, :].tolist(), shrink=0.8, label="error" if is_error else label
        )
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: Sequence[dict], path: pathlib.Path) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Any, path: pathlib.Path) -> None:
    def native(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        raise TypeError(type(x))

    path.write_text(json.dumps(data, indent=2, default=native))
