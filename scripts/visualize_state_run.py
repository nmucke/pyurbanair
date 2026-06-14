"""Visualize a SINGLE ESMDA run that estimates the STATE as well as parameters.

This is the state-estimation counterpart to ``scripts/visualize_run.py``. Point
it at an output folder under e.g. ``/projects/prjs2075/urbanair/assim_with_state``.
Those runs come in two flavours, encoded by the smoother's
``final_time_smoothing`` flag (and mirrored by the folder-name suffix):

  * ``_ic``  -- estimate only the INITIAL CONDITION of each window
               (``final_time_smoothing: false``)
  * ``_all`` -- also adjust the STATE THROUGHOUT each window
               (``final_time_smoothing: true``)

orthogonally, a "method" axis selects how the high-dimensional state update is
made tractable: ``svd`` (online SVD state reduction), ``localization_corr``
(correlation localization) or ``localization_dist_dist`` (distance localization).

The script keeps the cheap parameter + posterior-mean figures from
``visualize_run.py`` (so the parameter-estimation results are still shown), and
ADDS state-estimation-specific figures built from the per-window ensemble state:

  Parameter / light-state figures (reused, work from small artifacts):
    * parameter_trajectories.png -- truth vs prior vs posterior ensemble per param
    * parameter_error.png        -- posterior RMSE & CRPS vs truth per param
    * parameter_metrics.png      -- prior/posterior skill bars from run_summary
    * final_state.png            -- posterior mean/std |U| (+ truth if available)
    * state_montage.png          -- posterior mean |U| at several times
    * metrics_summary.png        -- scalar metrics table from run_summary

  State-estimation figures (new, read per-window ensemble state hyperslabs):
    * state_spread_reduction.png -- per-window ensemble |U| spread, prior vs
                                    posterior (does the update shrink uncertainty?)
    * state_window_increment.png -- prior mean / posterior mean / increment /
                                    posterior std maps for one window's final frame

The per-window ensemble state files are large (~0.5 GB each), so the new figures
read only a single z-slice at a single time via lazy hyperslab selection. Pass
``--no-window-state`` to skip them entirely (then only the light figures run).

Usage:

    python scripts/visualize_state_run.py /path/to/run_dir
    python scripts/visualize_state_run.py /path/to/run_dir --window 4 --z-level 2
    python scripts/visualize_state_run.py /path/to/run_dir --no-window-state --no-truth
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
# Reuse the parameter + light-state figures (and IO helpers) from the
# parameter-only visualizer so the two stay in lockstep.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from visualize_run import (  # noqa: E402
    _load_truth_final_vel,
    _load_yaml,
    _obs_points,
    _window_edges,
    plot_metrics_summary,
    plot_param_trajectories,
    plot_parameter_metric_bars,
    plot_state_montage,
)

from pyurbanair.plotting import plot_final_state_with_obs, plot_parameter_error  # noqa: E402

_COLOR_PRIOR = "#ff7f0e"
_COLOR_POSTERIOR = "#1f77b4"


# ---------------------------------------------------------------------------
# Mode / method detection (shared with the comparison script)
# ---------------------------------------------------------------------------

def detect_mode_and_method(run_dir: pathlib.Path) -> dict:
    """Classify a run's state-estimation mode and method.

    ``mode`` is ``"all"`` (state adjusted throughout each window) or ``"ic"``
    (initial condition only), derived from the smoother's
    ``final_time_smoothing`` flag and cross-checked against the folder suffix.
    ``method`` is ``svd`` / ``localization_corr`` / ``localization_dist_dist`` /
    ``plain`` from the configured state reduction / localization.
    """
    cfg = _load_yaml(run_dir / "config.yaml")
    esmda = cfg.get("esmda", {}) or {}

    # --- mode: prefer the config flag, fall back to the folder-name suffix ---
    fts = esmda.get("final_time_smoothing")
    name = run_dir.name
    if fts is None:
        mode = "all" if name.endswith("_all") else ("ic" if name.endswith("_ic") else "unknown")
    else:
        mode = "all" if fts else "ic"

    # --- method: SVD state reduction vs a localization scheme ----------------
    method = "plain"
    sr = esmda.get("state_reduction")
    loc = esmda.get("localization")
    if isinstance(sr, dict) and sr.get("_target_"):
        method = "svd"
    elif isinstance(loc, dict) and loc.get("_target_"):
        tgt = loc["_target_"].split(".")[-1].lower()
        if "correlation" in tgt:
            method = "localization_corr"
        elif "distance" in tgt:
            method = "localization_dist_dist"
        else:
            method = tgt
    return {"mode": mode, "method": method}


# ---------------------------------------------------------------------------
# Per-window state helpers (lazy hyperslab reads)
# ---------------------------------------------------------------------------

def _available_windows(run_dir: pathlib.Path) -> list[int]:
    """Window indices that have BOTH a prior and posterior state file present."""
    wdir = run_dir / "windows"
    if not wdir.is_dir():
        return []
    have: dict[int, set[str]] = {}
    for p in wdir.glob("window_*_*_state.nc"):
        m = re.match(r"window_(\d+)_(prior|posterior)_state\.nc", p.name)
        if m:
            have.setdefault(int(m.group(1)), set()).add(m.group(2))
    return sorted(w for w, kinds in have.items() if {"prior", "posterior"} <= kinds)


def _velmag_slice(path: pathlib.Path, z_level: int, time_idx: int = -1) -> np.ndarray | None:
    """Ensemble |U| at one z-level and one time as ``(ensemble, ny, nx)``.

    Reads only the requested hyperslab from disk (the per-window ensemble state
    files are far too large to load whole). ``u``/``v``/``w`` live on staggered
    grids but share a shape, so |U| is formed element-wise as elsewhere.
    """
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    try:
        if not all(v in ds.data_vars for v in ("u", "v", "w")):
            return None

        def pick(name: str) -> np.ndarray:
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=time_idx)
            for zd in ("zt", "zm", "z"):
                if zd in da.dims:
                    z = min(max(z_level, 0), da.sizes[zd] - 1)
                    da = da.isel({zd: z})
                    break
            return np.asarray(da.values, dtype=float)

        u, v, w = pick("u"), pick("v"), pick("w")
        return np.sqrt(u**2 + v**2 + w**2)
    finally:
        ds.close()


def plot_state_spread_reduction(
    run_dir: pathlib.Path, windows: list[int], out_path: pathlib.Path, z_level: int
) -> None:
    """Per-window ensemble |U| spread (prior vs posterior) at each window's end.

    The ensemble std of |U|, averaged over the z-slice, is a proxy for state
    uncertainty. A drop from prior to posterior means the assimilation is
    informing the state; this is the headline diagnostic for a state run.
    """
    prior_spread, post_spread, prior_mean_err, kept = [], [], [], []
    for w in windows:
        pr = _velmag_slice(run_dir / "windows" / f"window_{w}_prior_state.nc", z_level)
        po = _velmag_slice(run_dir / "windows" / f"window_{w}_posterior_state.nc", z_level)
        if pr is None or po is None:
            continue
        kept.append(w)
        prior_spread.append(float(np.nanmean(np.nanstd(pr, axis=0))))
        post_spread.append(float(np.nanmean(np.nanstd(po, axis=0))))
        # RMS magnitude of the mean update (how much the state actually moved).
        prior_mean_err.append(float(np.sqrt(np.nanmean((po.mean(0) - pr.mean(0)) ** 2))))
    if not kept:
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax0.plot(kept, prior_spread, "-o", color=_COLOR_PRIOR, label="Prior spread")
    ax0.plot(kept, post_spread, "-s", color=_COLOR_POSTERIOR, label="Posterior spread")
    ax0.set_xlabel("Assimilation window")
    ax0.set_ylabel("Mean ensemble std of |U|")
    ax0.set_title("State uncertainty: prior vs posterior")
    ax0.set_ylim(bottom=0.0)
    ax0.legend(loc="best")

    ax1.bar([str(w) for w in kept], prior_mean_err, color=_COLOR_POSTERIOR)
    ax1.set_xlabel("Assimilation window")
    ax1.set_ylabel("RMS state increment |U|")
    ax1.set_title("Magnitude of the state update")
    ax1.margins(y=0.12)

    fig.suptitle("Per-window state assimilation", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_window_increment(
    run_dir: pathlib.Path, window: int, out_path: pathlib.Path, z_level: int
) -> None:
    """Prior mean / posterior mean / increment / posterior std maps for a window."""
    pr = _velmag_slice(run_dir / "windows" / f"window_{window}_prior_state.nc", z_level)
    po = _velmag_slice(run_dir / "windows" / f"window_{window}_posterior_state.nc", z_level)
    if pr is None or po is None:
        return
    prior_mean, post_mean = pr.mean(0), po.mean(0)
    incr = post_mean - prior_mean
    post_std = np.nanstd(po, axis=0)

    vmax_field = float(np.nanmax([np.nanmax(prior_mean), np.nanmax(post_mean)]))
    vlim = float(np.nanmax(np.abs(incr))) or 1.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.0), constrained_layout=True)
    panels = [
        (prior_mean, "Prior mean |U|", dict(vmin=0.0, vmax=vmax_field), "viridis"),
        (post_mean, "Posterior mean |U|", dict(vmin=0.0, vmax=vmax_field), "viridis"),
        (incr, "Increment (post - prior)", dict(vmin=-vlim, vmax=vlim), "RdBu_r"),
        (post_std, "Posterior std |U|", dict(vmin=0.0), "magma"),
    ]
    for ax, (field, title, kw, cmap) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", aspect="equal", cmap=cmap, **kw)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(
        f"State update at window {window} (final frame, z-level {z_level})",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def visualize(run_dir: pathlib.Path, out_dir: pathlib.Path, *,
              z_level: int, want_truth: bool, want_window_state: bool,
              window: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_yaml(run_dir / "run_summary.yaml")
    cfg = _load_yaml(run_dir / "config.yaml")
    num_windows = (summary.get("configuration", {}) or {}).get("num_assimilation_windows")
    info = detect_mode_and_method(run_dir)

    print(f"Run:    {run_dir}")
    print(f"Output: {out_dir}")
    print(f"Mode:   {info['mode']}  ({'state throughout window' if info['mode']=='all' else 'initial condition only'})")
    print(f"Method: {info['method']}")

    # --- Parameter figures (reused; the run still estimates parameters) -------
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

    # --- Light state figures (posterior mean/std rollout) --------------------
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
        print("  ! posterior_state_mean.nc missing; skipping light state figures")

    # --- State-estimation figures (heavy per-window ensemble reads) ----------
    if want_window_state:
        windows = _available_windows(run_dir)
        if not windows:
            print("  ! no complete per-window state files; skipping state-update figures")
        else:
            plot_state_spread_reduction(run_dir, windows, out_dir / "state_spread_reduction.png", z_level)
            print(f"  + state_spread_reduction.png ({len(windows)} windows)")
            sel = window if window is not None and window in windows else windows[-1]
            plot_window_increment(run_dir, sel, out_dir / "state_window_increment.png", z_level)
            print(f"  + state_window_increment.png (window {sel})")
    else:
        print("  (skipping per-window state figures: --no-window-state)")

    # --- Scalar metrics table ------------------------------------------------
    if summary:
        plot_metrics_summary(summary, out_dir / "metrics_summary.png")
        print("  + metrics_summary.png")

    for ds in (prior, post, truth):
        if ds is not None:
            ds.close()
    print(f"Done. Figures in {out_dir}")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=pathlib.Path, help="state-estimation run output folder")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output dir (default: <repo>/result_figures/<case>)")
    ap.add_argument("--case", default=None,
                    help="case name for result_figures/<case> (default: run folder name)")
    ap.add_argument("--z-level", type=int, default=2, help="z-plane index for field plots")
    ap.add_argument("--window", type=int, default=None,
                    help="window index for the increment map (default: last available)")
    ap.add_argument("--no-truth", action="store_true",
                    help="skip loading the ground truth for the final-state panel")
    ap.add_argument("--no-window-state", action="store_true",
                    help="skip the heavy per-window ensemble-state figures")
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
        want_window_state=not args.no_window_state,
        window=args.window,
    )


if __name__ == "__main__":
    main()
