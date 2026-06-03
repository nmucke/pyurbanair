"""Shared helpers for the scripts/ runners.

Small, dependency-light glue that several top-level scripts would otherwise
copy-paste. Pure config->object construction belongs in
``pyurbanair.config.hydra_helpers``; this module is only for script-level
plumbing (results-dir resolution, the forward-model visualization block).
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
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


def plot_derived_inflow_angle(state, params, out_dir: pathlib.Path) -> None:
    """Plot the inflow angle derived from the simulated field vs. the prescribed.

    Compares the angle recovered from (u, v) at three probes near the inlet
    against the prescribed time-varying ``inflow_angle``. PALM/uDALES store
    u/v on staggered grids (u on xu, v on yv), so the spatial dim names are
    resolved per-variable.
    """

    def _pick_dim(da, candidates):
        return next(d for d in candidates if d in da.dims)

    x_cands = ("x", "xt", "xm", "xu")
    y_cands = ("y", "yt", "ym", "yv")
    z_cands = ("z", "zt", "zm", "zu")
    u_x_dim = _pick_dim(state["u"], x_cands)
    u_y_dim = _pick_dim(state["u"], y_cands)
    u_z_dim = _pick_dim(state["u"], z_cands)
    v_x_dim = _pick_dim(state["v"], x_cands)
    v_y_dim = _pick_dim(state["v"], y_cands)
    v_z_dim = _pick_dim(state["v"], z_cands)

    x_left = float(state[u_x_dim].min())
    y_min = float(state[u_y_dim].min())
    y_max = float(state[u_y_dim].max())
    y_probes = [
        y_min + 0.2 * (y_max - y_min),
        y_min + 0.5 * (y_max - y_min),
        y_min + 0.8 * (y_max - y_min),
    ]
    z_probe = 0.5 * (float(state[u_z_dim].min()) + float(state[u_z_dim].max()))

    fig, ax = plt.subplots(figsize=(8, 4))
    for y_p in y_probes:
        u_sel = {u_x_dim: x_left, u_y_dim: y_p, u_z_dim: z_probe}
        v_sel = {v_x_dim: x_left, v_y_dim: y_p, v_z_dim: z_probe}
        u_t = state["u"].sel(**u_sel, method="nearest")
        v_t = state["v"].sel(**v_sel, method="nearest")
        angle_sim = np.degrees(np.arctan2(v_t.values, u_t.values))
        ax.plot(state.time.values, angle_sim, label=f"y={y_p:.1f} m")
    ax.plot(
        params["time"].values,
        params["inflow_angle"].values,
        "k--",
        alpha=0.5,
        label="prescribed",
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("inflow angle [deg]")
    ax.set_title(f"Derived inflow angle near x={x_left:.1f} m (z={z_probe:.1f} m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "derived_inflow_angle.png")
    plt.close()


def plot_derived_velocity_magnitude(state, params, out_dir: pathlib.Path) -> None:
    """Plot the speed derived from the simulated field vs. the prescribed.

    Mirrors :func:`plot_derived_inflow_angle`: recovers the horizontal speed
    ``sqrt(u^2 + v^2)`` from the same three inlet probes and compares it against
    the prescribed time-varying ``velocity_magnitude``.
    """

    def _pick_dim(da, candidates):
        return next(d for d in candidates if d in da.dims)

    x_cands = ("x", "xt", "xm", "xu")
    y_cands = ("y", "yt", "ym", "yv")
    z_cands = ("z", "zt", "zm", "zu")
    u_x_dim = _pick_dim(state["u"], x_cands)
    u_y_dim = _pick_dim(state["u"], y_cands)
    u_z_dim = _pick_dim(state["u"], z_cands)
    v_x_dim = _pick_dim(state["v"], x_cands)
    v_y_dim = _pick_dim(state["v"], y_cands)
    v_z_dim = _pick_dim(state["v"], z_cands)

    x_left = float(state[u_x_dim].min())
    y_min = float(state[u_y_dim].min())
    y_max = float(state[u_y_dim].max())
    y_probes = [
        y_min + 0.2 * (y_max - y_min),
        y_min + 0.5 * (y_max - y_min),
        y_min + 0.8 * (y_max - y_min),
    ]
    z_probe = 0.5 * (float(state[u_z_dim].min()) + float(state[u_z_dim].max()))

    fig, ax = plt.subplots(figsize=(8, 4))
    for y_p in y_probes:
        u_sel = {u_x_dim: x_left, u_y_dim: y_p, u_z_dim: z_probe}
        v_sel = {v_x_dim: x_left, v_y_dim: y_p, v_z_dim: z_probe}
        u_t = state["u"].sel(**u_sel, method="nearest")
        v_t = state["v"].sel(**v_sel, method="nearest")
        speed_sim = np.hypot(u_t.values, v_t.values)
        ax.plot(state.time.values, speed_sim, label=f"y={y_p:.1f} m")
    ax.plot(
        params["time"].values,
        params["velocity_magnitude"].values,
        "k--",
        alpha=0.5,
        label="prescribed",
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity magnitude [m/s]")
    ax.set_title(
        f"Derived velocity magnitude near x={x_left:.1f} m (z={z_probe:.1f} m)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "derived_velocity_magnitude.png")
    plt.close()
