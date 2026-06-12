"""Visualize all relevant metrics and results for a SINGLE rollout-ESMDA run.

Point this at a successful ``scripts/run_esmda.py`` output folder (the per-run
directory that holds ``run_summary.yaml``, the ``*_params.nc`` files and
``posterior_state_mean.nc``). It regenerates a consolidated set of figures into
``result_figures/<case>/`` -- by default ``<case>`` is the run folder's name.

It works almost entirely from the small saved artifacts (parameter NetCDFs,
the posterior mean/std rollout, and ``run_summary.yaml``), so it is cheap and
does NOT re-run the forward model. The only optionally-heavy step is loading a
single final frame of the ground truth (via ``truth_access.yaml``) for the
truth-vs-posterior final-state comparison; if the truth is unavailable that
panel is skipped.

Figures written:
  * parameter_trajectories.png -- truth vs prior vs posterior ensemble per param
  * parameter_error.png        -- posterior RMSE & CRPS vs truth per param
  * parameter_metrics.png      -- prior/posterior skill bars from run_summary
  * final_state.png            -- posterior mean/std |U| (+ truth if available)
  * state_montage.png          -- posterior mean |U| at several times
  * metrics_summary.png        -- scalar metrics table from run_summary
Existing run-folder figures that need the full ensemble (sensor time series,
the rollout animation) are copied across so everything lives in one place.

Usage:

    python scripts/visualize_run.py /path/to/run_dir
    python scripts/visualize_run.py /path/to/run_dir --out result_figures/my_case
    python scripts/visualize_run.py /path/to/run_dir --no-truth
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pyurbanair.plotting import plot_final_state_with_obs, plot_parameter_error
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice

_PARAMS = ("inflow_angle", "velocity_magnitude")
_PARAM_LABELS = {
    "inflow_angle": "Inflow angle [deg]",
    "velocity_magnitude": "Velocity magnitude [m/s]",
}
_COLOR_PRIOR = "#ff7f0e"
_COLOR_POSTERIOR = "#1f77b4"
_COLOR_TRUTH = "k"

# Run-folder figures that depend on the full ensemble / truth and are too
# expensive to recompute here -- copied across verbatim if present.
_COPY_FIGURES = (
    "sensor_timeseries_assimilation.png",
    "sensor_timeseries_validation.png",
    "rollout_animation.mp4",
    "rollout_animation.gif",
)


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _window_edges(time: np.ndarray, num_windows: int | None) -> np.ndarray | None:
    """Equal-width window boundaries across the param time axis (for shading)."""
    if not num_windows or num_windows < 2:
        return None
    return np.linspace(float(time.min()), float(time.max()), num_windows + 1)


def _shade_windows(ax, edges: np.ndarray | None) -> None:
    if edges is None:
        return
    for k in range(0, len(edges) - 1, 2):
        ax.axvspan(edges[k], edges[k + 1], color="0.92", zorder=0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_param_trajectories(
    prior: xr.Dataset,
    post: xr.Dataset,
    truth: xr.Dataset,
    out_path: pathlib.Path,
    num_windows: int | None,
) -> None:
    """Truth vs prior vs posterior ensemble for every time-varying parameter."""
    params = [p for p in _PARAMS if p in post.data_vars]
    if not params:
        return
    t = np.asarray(post["time"].values)
    edges = _window_edges(t, num_windows)

    fig, axes = plt.subplots(
        len(params), 1, figsize=(11, 3.4 * len(params)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, params):
        _shade_windows(ax, edges)
        if p in prior.data_vars:
            pr = np.asarray(prior[p].transpose("ensemble", "time").values)
            ax.plot(t, pr.T, color=_COLOR_PRIOR, alpha=0.18, linewidth=0.7)
            ax.plot(t, pr.mean(0), color=_COLOR_PRIOR, linewidth=2.0, label="Prior mean")
        po = np.asarray(post[p].transpose("ensemble", "time").values)
        ax.plot(t, po.T, color=_COLOR_POSTERIOR, alpha=0.18, linewidth=0.7)
        ax.plot(t, po.mean(0), color=_COLOR_POSTERIOR, linewidth=2.5, label="Posterior mean")
        if p in truth.data_vars:
            tr = np.asarray(truth[p].values)
            ax.plot(t, tr, color=_COLOR_TRUTH, linewidth=2.0, linestyle="--",
                    zorder=5, label="Truth")
        ax.set_ylabel(_PARAM_LABELS.get(p, p))
        ax.set_xlabel("Time")
        ax.margins(x=0.01)
        ax.legend(loc="best")
    fig.suptitle("Parameter trajectories: truth vs prior vs posterior",
                 fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_parameter_metric_bars(summary: dict, out_path: pathlib.Path) -> None:
    """Prior vs posterior RMSE (mean & final) + CRPS bars, from run_summary."""
    pm = summary.get("parameter_metrics", {}) or {}
    params = [p for p in _PARAMS if p in pm]
    if not params:
        return

    fig, axes = plt.subplots(
        1, len(params), figsize=(5.5 * len(params), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, params):
        entry = pm[p]
        rmse = entry.get("rmse", {}) or {}
        crps = entry.get("crps", {}) or {}
        labels = ["prior\nRMSE", "post\nRMSE\n(mean)", "post\nRMSE\n(final)", "post\nCRPS\n(mean)"]
        values = [
            entry.get("prior_rmse_mean"),
            rmse.get("mean"),
            rmse.get("final"),
            crps.get("mean"),
        ]
        values = [np.nan if v is None else float(v) for v in values]
        colors = [_COLOR_PRIOR, _COLOR_POSTERIOR, _COLOR_POSTERIOR, "#6baed6"]
        bars = ax.bar(labels, values, color=colors)
        for b, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                        ha="center", va="bottom", fontsize=9)
        red = entry.get("rmse_reduction_vs_prior")
        title = _PARAM_LABELS.get(p, p)
        if red is not None:
            title += f"\nRMSE reduction vs prior: {float(red):+.0%}"
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("error")
        ax.margins(y=0.15)
    fig.suptitle("Parameter estimation skill", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_state_montage(
    mean_vel: xr.DataArray, out_path: pathlib.Path, z_level: int, n_frames: int = 5
) -> None:
    """Posterior mean |U| at n evenly-spaced times (z-plane)."""
    n_t = mean_vel.sizes.get("time", 1)
    idxs = np.unique(np.linspace(0, n_t - 1, n_frames, dtype=int)) if n_t > 1 else [0]
    slabs = [extract_2d_slice(mean_vel.isel(time=int(i)) if "time" in mean_vel.dims
                              else mean_vel, z_level=z_level) for i in idxs]
    vmin = float(np.nanmin([np.nanmin(s) for s in slabs]))
    vmax = float(np.nanmax([np.nanmax(s) for s in slabs]))

    fig, axes = plt.subplots(
        1, len(slabs), figsize=(4.2 * len(slabs), 4.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    im = None
    for ax, i, slab in zip(axes, idxs, slabs):
        im = ax.imshow(slab, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(f"t index {int(i)}/{n_t - 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Velocity magnitude")
    fig.suptitle("Posterior mean |U| over time", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_summary(summary: dict, out_path: pathlib.Path) -> None:
    """Render the scalar metrics from run_summary.yaml as a text table figure."""
    cfg = summary.get("configuration", {}) or {}
    timing = summary.get("timing", {}) or {}
    pm = summary.get("parameter_metrics", {}) or {}
    sm = summary.get("state_metrics", {}) or {}
    sen = summary.get("sensor_metrics", {}) or {}

    def fmt(v, nd=4):
        if isinstance(v, (int, float)):
            return f"{v:.{nd}g}"
        return str(v)

    rows: list[tuple[str, str]] = []
    rows.append(("== CONFIGURATION ==", ""))
    for k in ("smoother", "assimilation_model", "truth_model", "ensemble_size",
              "num_esmda_steps", "num_assimilation_windows",
              "simulation_time_per_window", "final_time", "obs_error_std", "seed"):
        if k in cfg:
            rows.append((k, fmt(cfg[k])))

    if timing:
        rows.append(("", ""))
        rows.append(("== TIMING [s] ==", ""))
        for k in ("esmda_total_seconds", "esmda_solve_seconds", "mean_window_seconds"):
            if k in timing:
                rows.append((k, fmt(timing[k])))

    for p in _PARAMS:
        if p in pm:
            e = pm[p]
            rows.append(("", ""))
            rows.append((f"== PARAM: {p} ==", ""))
            rows.append(("rmse mean / final", f"{fmt(e.get('rmse',{}).get('mean'))} / {fmt(e.get('rmse',{}).get('final'))}"))
            rows.append(("crps mean", fmt(e.get("crps", {}).get("mean"))))
            rows.append(("prior rmse mean", fmt(e.get("prior_rmse_mean"))))
            rows.append(("rmse reduction vs prior", fmt(e.get("rmse_reduction_vs_prior"))))

    if sm.get("vel_magnitude_rmse"):
        rows.append(("", ""))
        rows.append(("== STATE ==", ""))
        v = sm["vel_magnitude_rmse"]
        rows.append(("vel |U| rmse mean / final", f"{fmt(v.get('mean'))} / {fmt(v.get('final'))}"))

    for s in ("assimilation", "validation"):
        if s in sen:
            e = sen[s]
            rows.append(("", ""))
            rows.append((f"== SENSORS: {s} ({e.get('num_sensors','?')}) ==", ""))
            rmse = e.get("velocity_vector_rmse", {}) or {}
            es = e.get("velocity_vector_energy_score", {}) or {}
            rows.append(("vector rmse mean / final", f"{fmt(rmse.get('mean'))} / {fmt(rmse.get('final'))}"))
            rows.append(("energy score mean", fmt(es.get("mean"))))

    fig_h = max(4.0, 0.32 * len(rows))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.axis("off")
    y = 1.0
    dy = 1.0 / (len(rows) + 1)
    for label, value in rows:
        bold = label.startswith("==")
        ax.text(0.01, y, label, family="monospace", fontsize=10,
                fontweight="bold" if bold else "normal", va="top")
        if value:
            ax.text(0.62, y, value, family="monospace", fontsize=10, va="top")
        y -= dy
    ax.set_title("Run metrics summary", fontsize=14, fontweight="bold", loc="left")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Truth / obs loading (best-effort)
# ---------------------------------------------------------------------------

def _load_truth_final_vel(run_dir: pathlib.Path) -> xr.DataArray | None:
    """Final-frame truth |U| (one z-plane's worth of field), or None on failure."""
    ta = _load_yaml(run_dir / "truth_access.yaml")
    tpath = ta.get("true_state_path")
    if not tpath or not pathlib.Path(tpath).exists():
        return None
    try:
        ds = xr.open_dataset(tpath)
        final = ds.isel(time=-1)
        final = add_velocity_magnitude(final.load())
        ds.close()
        if "vel_magnitude" not in final.data_vars:
            return None
        return final["vel_magnitude"]
    except Exception as e:  # noqa: BLE001
        print(f"  ! truth final frame unavailable, skipping truth panel ({e})")
        return None


def _obs_points(cfg: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    obs = cfg.get("obs", {}) or {}
    x = obs.get("x_points")
    y = obs.get("y_points")
    if x is None or y is None:
        return None, None
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def visualize(run_dir: pathlib.Path, out_dir: pathlib.Path, *,
              z_level: int, want_truth: bool, copy_existing: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_yaml(run_dir / "run_summary.yaml")
    cfg = _load_yaml(run_dir / "config.yaml")
    num_windows = (summary.get("configuration", {}) or {}).get("num_assimilation_windows")

    print(f"Run:    {run_dir}")
    print(f"Output: {out_dir}")

    # --- Parameter figures (from the small *_params.nc files) ----------------
    prior = post = truth = None
    try:
        prior = xr.open_dataset(run_dir / "prior_params.nc")
        post = xr.open_dataset(run_dir / "posterior_params.nc")
        truth = xr.open_dataset(run_dir / "true_params.nc")
    except FileNotFoundError as e:
        print(f"  ! parameter NetCDFs missing ({e}); skipping parameter figures")

    if post is not None and truth is not None:
        plot_param_trajectories(prior, post, truth, out_dir / "parameter_trajectories.png", num_windows)
        print("  + parameter_trajectories.png")
        edges = _window_edges(np.asarray(post["time"].values), num_windows)
        plot_parameter_error(
            esmda_params=post, true_params=truth,
            output_path=out_dir / "parameter_error.png",
            window_edges=list(edges) if edges is not None else None,
        )
        print("  + parameter_error.png")

    plot_parameter_metric_bars(summary, out_dir / "parameter_metrics.png")
    print("  + parameter_metrics.png")

    # --- State figures (from posterior_state_mean.nc) ------------------------
    psm_path = run_dir / "posterior_state_mean.nc"
    if psm_path.exists():
        psm = xr.open_dataset(psm_path)
        if "vel_mean" in psm.data_vars and "vel_std" in psm.data_vars:
            true_vel = _load_truth_final_vel(run_dir) if want_truth else None
            obs_x, obs_y = _obs_points(cfg)
            plot_final_state_with_obs(
                mean_vel=psm["vel_mean"], std_vel=psm["vel_std"],
                output_path=out_dir / "final_state.png",
                true_vel=true_vel, obs_x=obs_x, obs_y=obs_y, z_level=z_level,
            )
            print("  + final_state.png")
            plot_state_montage(psm["vel_mean"], out_dir / "state_montage.png", z_level)
            print("  + state_montage.png")
        psm.close()
    else:
        print("  ! posterior_state_mean.nc missing; skipping state figures")

    # --- Scalar metrics table ------------------------------------------------
    if summary:
        plot_metrics_summary(summary, out_dir / "metrics_summary.png")
        print("  + metrics_summary.png")

    # --- Copy ensemble-dependent figures already in the run folder -----------
    if copy_existing:
        for name in _COPY_FIGURES:
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
                print(f"  > copied {name}")

    for ds in (prior, post, truth):
        if ds is not None:
            ds.close()
    print(f"Done. Figures in {out_dir}")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=pathlib.Path, help="run_esmda output folder")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output dir (default: <repo>/result_figures/<case>)")
    ap.add_argument("--case", default=None,
                    help="case name for result_figures/<case> (default: run folder name)")
    ap.add_argument("--z-level", type=int, default=0, help="z-plane index for field plots")
    ap.add_argument("--no-truth", action="store_true",
                    help="skip loading the ground truth for the final-state panel")
    ap.add_argument("--no-copy-existing", action="store_true",
                    help="do not copy sensor-timeseries / animation figures from the run folder")
    args = ap.parse_args()

    run_dir = args.run.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run folder not found: {run_dir}")

    case = args.case or run_dir.name
    out_dir = args.out or (repo_root / "result_figures" / case)

    visualize(
        run_dir, out_dir,
        z_level=args.z_level,
        want_truth=not args.no_truth,
        copy_existing=not args.no_copy_existing,
    )


if __name__ == "__main__":
    main()
