"""Shared helpers for the scripts/ runners.

Small, dependency-light glue that several top-level scripts would otherwise
copy-paste. Pure config->object construction belongs in
``pyurbanair.config.hydra_helpers``; this module is only for script-level
plumbing (results-dir resolution, the forward-model visualization block).
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.run_utils import extract_2d_slice


def resolve_results_dir(cfg: DictConfig) -> pathlib.Path | None:
    """On-disk results dir from ``cfg.run.results_dir`` (None -> in-memory)."""
    return (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )


def visualize_forward_state(
    state,
    model_name: str,
    out_dir: pathlib.Path,
    title_prefix: str,
    z_level: int = 0,
) -> None:
    """Write the standard forward-model field snapshot + animation.

    ``state`` must already be reduced to a single member (callers pass the
    ensemble mean for ensemble runs). uDALES states are projected onto a common
    grid first; solver-only vars (rho/blanking/pres) are dropped before
    animating.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_var = "vel_magnitude" if "vel_magnitude" in state.data_vars else "u"
    plot_2d = extract_2d_slice(state[plot_var], z_level=z_level)
    plt.figure(figsize=(6, 5))
    plt.imshow(plot_2d, origin="lower")
    plt.colorbar(label=plot_var)
    plt.title(f"{title_prefix} - {plot_var} (last time, mid z)")
    plt.tight_layout()
    plt.savefig(out_dir / "field_snapshot.png")
    plt.close()

    if model_name == "pyudales":
        from pyudales.utils.grid_utils import interpolate_grid

        state = interpolate_grid(state)

    drop = [v for v in ("rho", "blanking", "pres") if v in state]
    if drop:
        state = state.drop_vars(drop)

    animate_state(
        state=state,
        output_path=out_dir / "state_animation.mp4",
        z_level=z_level,
    )
    print(f"Saved visualization outputs in {out_dir}")
