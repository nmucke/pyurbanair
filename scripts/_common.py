"""Shared helpers for the scripts/ runners.

Small, dependency-light glue that several top-level scripts would otherwise
copy-paste. Pure config->object construction belongs in
``pyurbanair.config.hydra_helpers``; this module is only for script-level
plumbing (results-dir resolution, the forward-model visualization block).
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xarray
from omegaconf import DictConfig

from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.da_metrics import (
    per_knot_crps,
    per_knot_error,
    per_knot_in_band,
    per_knot_spread,
    summary_scalars,
)
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


# ---------------------------------------------------------------------------
# Time-varying ESMDA diagnostics (shared by run_esmda.py for dynamic params).
# ---------------------------------------------------------------------------


def plot_time_varying_params(
    params_history: xarray.Dataset,
    true_params: xarray.Dataset,
    time_coords: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    """Plot true vs estimated time-varying parameters.

    For each parameter, the true profile is shown as a solid line and the
    final ESMDA step's ensemble mean is shown with a shaded +/- 1 std band.
    """
    param_names = [
        name for name in true_params.data_vars if "time" in true_params[name].dims
    ]
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params), squeeze=False)

    for ax, name in zip(axes[:, 0], param_names):
        true_vals = np.asarray(true_params[name].values)
        ax.plot(time_coords, true_vals, color="C0", linewidth=2, label="True")

        # Use the final ESMDA step
        final = params_history[name].isel(esmda_step=-1)
        ens_mean = np.asarray(final.mean(dim="ensemble").values)
        ens_std = np.asarray(final.std(dim="ensemble").values)

        ax.plot(time_coords, ens_mean, color="C1", linewidth=2, label="Estimated mean")
        ax.fill_between(
            time_coords,
            ens_mean - ens_std,
            ens_mean + ens_std,
            color="C1",
            alpha=0.3,
            label="Estimated std",
        )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(name)
        ax.legend()
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-varying parameter plot to {output_path}")


def compute_time_varying_metrics(
    params_history: xarray.Dataset,
    true_params: xarray.Dataset,
    time_coords: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Compute per-knot and per-step summary metrics.

    Returns ``(rows, summary_rows)``: long-format per-knot records and
    per-step summary records, ready for CSV writing.
    """
    param_names = [
        name for name in true_params.data_vars if "time" in true_params[name].dims
    ]
    rows: list[dict] = []
    summary_rows: list[dict] = []
    n_steps = int(params_history.sizes["esmda_step"])
    for k in range(n_steps):
        for name in param_names:
            ens = np.asarray(
                params_history[name]
                .isel(esmda_step=k)
                .transpose("ensemble", "time")
                .values
            )
            truth = np.asarray(true_params[name].values)
            err = per_knot_error(ens, truth)
            spr = per_knot_spread(ens)
            crps = per_knot_crps(ens, truth)
            band = per_knot_in_band(ens, truth)
            for t_idx, t in enumerate(time_coords):
                rows.append(
                    {
                        "esmda_step": k,
                        "parameter": name,
                        "time": float(t),
                        "error": float(err[t_idx]),
                        "spread": float(spr[t_idx]),
                        "crps": float(crps[t_idx]),
                        "in_band": int(bool(band[t_idx])),
                    }
                )
            summary = summary_scalars(ens, truth)
            summary_rows.append({"esmda_step": k, "parameter": name, **summary})
    return rows, summary_rows


def write_metrics_csv(rows: list[dict], path: pathlib.Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_metrics_summary(summary_rows: list[dict]) -> None:
    by_step: dict[int, list[dict]] = {}
    for row in summary_rows:
        by_step.setdefault(int(row["esmda_step"]), []).append(row)
    for step in sorted(by_step):
        print(f"--- ESMDA step {step} ---")
        for row in by_step[step]:
            print(
                f"  {row['parameter']:20s} "
                f"rmse={row['time_avg_error']:.4f}  "
                f"spread={row['time_avg_spread']:.4f}  "
                f"crps={row['mean_crps']:.4f}  "
                f"coverage={row['coverage']:.2f}"
            )


def plot_time_varying_metrics(
    params_history: xarray.Dataset,
    true_params: xarray.Dataset,
    time_coords: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    """Per-parameter diagnostic: error/spread/CRPS/in-band over time, one
    line per ESMDA step (color-graded so step 0 is light, final is dark)."""
    param_names = [
        name for name in true_params.data_vars if "time" in true_params[name].dims
    ]
    n_params = len(param_names)
    n_steps = int(params_history.sizes["esmda_step"])
    metric_titles = ["|mean - truth|", "ensemble spread", "CRPS", "in 90% band"]
    fig, axes = plt.subplots(n_params, 4, figsize=(16, 3.5 * n_params), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for row_idx, name in enumerate(param_names):
        truth = np.asarray(true_params[name].values)
        for k in range(n_steps):
            ens = np.asarray(
                params_history[name]
                .isel(esmda_step=k)
                .transpose("ensemble", "time")
                .values
            )
            err = per_knot_error(ens, truth)
            spr = per_knot_spread(ens)
            crps = per_knot_crps(ens, truth)
            band = per_knot_in_band(ens, truth).astype(float)
            shade = 0.2 + 0.8 * (k / max(n_steps - 1, 1))
            color = cmap(shade)
            label = f"step {k}"
            axes[row_idx, 0].plot(time_coords, err, color=color, label=label)
            axes[row_idx, 1].plot(time_coords, spr, color=color, label=label)
            axes[row_idx, 2].plot(time_coords, crps, color=color, label=label)
            axes[row_idx, 3].plot(
                time_coords, band, "o-", color=color, label=label, alpha=0.7
            )
        for col, title in enumerate(metric_titles):
            ax = axes[row_idx, col]
            ax.set_xlabel("Time [s]")
            ax.set_title(f"{name}: {title}")
            if col == 3:
                ax.set_ylim(-0.1, 1.1)
        axes[row_idx, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-varying metrics plot to {output_path}")
