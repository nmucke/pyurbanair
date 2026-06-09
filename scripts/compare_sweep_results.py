"""Visualize the sweep metrics + write the big CSV (final stage of the pipeline).

Third stage of the three-script sweep pipeline:

  1. scripts/run_esmda.py             -- runs the DA, writes the large posterior
                                         states/params + base run_summary.yaml.
  2. scripts/compute_sweep_metrics.py -- computes every metric + metric time series
                                         and writes SMALL artifacts to
                                         pyurbanair/sweep_metrics/<run>/.
  3. scripts/compare_sweep_results.py (THIS) -- reads pyurbanair/sweep_metrics/ and
                                         draws all comparison figures + the CSV.

It reads per run, under ``--root`` (default ``pyurbanair/sweep_metrics``):
``metrics.yaml`` (configuration + parameter / state / sensor metrics, |U| AND
per-component u/v/w), the copied parameter NetCDFs, and per sensor set
``sensor_timeseries_<set>.nc`` (truth + prior/posterior ensemble series). It:

Choose the sweep with ``--sweep``; each writes to its own subfolder:

  * ``--sweep domain``   -> ``comparison/domain``   (x-axis = grid cells; the
                            ensemble size + ESMDA steps are held at their modal
                            value).
  * ``--sweep ensemble`` -> ``comparison/ensemble`` (x-axis = ensemble size; the
                            domain is fixed to a single grid first).
  * ``--sweep all`` (default) does both.

For the chosen sweep it:

  1. Flattens every ``metrics.yaml`` + the dir-name tag into one tidy CSV.
  2. Plots each metric CATEGORY in its own figure -- parameters, assimilation
     sensors, validation sensors, state field -- RMSE next to CRPS, one line per
     (backend, localization).
  3. Compares backends side by side; plots ESMDA wall-clock scaling vs the axis.
  4. Overlays posterior parameter trajectories vs the truth across the sweep.
  5. Draws per-model sensor time-series grids (rows = sensors, cols = sweep
     values; one figure per component) showing ground truth, prior members
     (alpha 0.3), posterior members (alpha 0.5) and the posterior mean.

It is read-only and plain argparse (not Hydra).

Usage::

    python scripts/compare_sweep_results.py                       # both sweeps
    python scripts/compare_sweep_results.py --sweep ensemble      # ensemble only
    python scripts/compare_sweep_results.py --sweep domain \
        --root pyurbanair/sweep_metrics --out pyurbanair/comparison --models pyudales pylbm
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import yaml


# Directory tag written by the slurm scripts, e.g.
# ``pyudales_nx100_ny80_nz32_ens96_steps2_localization``.
_TAG_RE = re.compile(
    r"^(?P<model>[a-zA-Z]+)_nx(?P<nx>\d+)_ny(?P<ny>\d+)_nz(?P<nz>\d+)"
    r"_ens(?P<ens>\d+)_steps(?P<steps>\d+)(?P<loc>_localization)?$"
)

# Sweep axes: (column, human label, log-scale x?).
AXES = [
    ("grid_cells", "grid cells (nx*ny*nz)", True),
    ("ensemble_size", "ensemble size", True),
    ("num_esmda_steps", "ESMDA steps", False),
]

# --- Metric groups (each becomes its own figure) ---------------------------
_PARAMS = ["inflow_angle", "velocity_magnitude"]
_PARAM_PRETTY = {"inflow_angle": "inflow angle", "velocity_magnitude": "velocity"}
_PARAM_UNIT = {"inflow_angle": "deg", "velocity_magnitude": "m/s"}

# Sensor quantities scored per sensor set: column suffix -> (label, summary key).
_SENSOR_Q = {
    "vel": ("|U|", "vel_magnitude"),
    "u": ("u", "u"),
    "v": ("v", "v"),
    "w": ("w", "w"),
}


def param_metrics() -> list[tuple[str, str, bool]]:
    """(column, label, lower-is-better) for every parameter metric (RMSE + CRPS)."""
    out = []
    for p in _PARAMS:
        pretty, unit = _PARAM_PRETTY[p], _PARAM_UNIT[p]
        out.append((f"param_{p}_rmse_mean", f"{pretty} RMSE [{unit}]", True))
        out.append((f"param_{p}_crps_mean", f"{pretty} CRPS [{unit}]", True))
        out.append((f"param_{p}_rmse_reduction", f"{pretty} RMSE reduction vs prior", False))
    return out


def sensor_metrics_group(setname: str) -> list[tuple[str, str, bool]]:
    """(column, label, lower-is-better) for one sensor set: |U| + u/v/w, RMSE+CRPS."""
    out = []
    for q, (lbl, _key) in _SENSOR_Q.items():
        out.append((f"sensor_{setname}_{q}_rmse_mean", f"{lbl} RMSE [m/s]", True))
        out.append((f"sensor_{setname}_{q}_crps_mean", f"{lbl} CRPS [m/s]", True))
    return out


STATE_METRICS = [
    ("state_vel_rmse_mean", "state |U| RMSE (mean) [m/s]", True),
    ("state_vel_rmse_final", "state |U| RMSE (final) [m/s]", True),
]

# Concise cross-category selection used only for the backend bar chart.
SUMMARY_METRICS = [
    ("param_inflow_angle_rmse_mean", "inflow angle RMSE [deg]", True),
    ("param_velocity_magnitude_rmse_mean", "velocity RMSE [m/s]", True),
    ("state_vel_rmse_mean", "state |U| RMSE [m/s]", True),
    ("sensor_assimilation_vel_rmse_mean", "assim |U| RMSE [m/s]", True),
    ("sensor_validation_vel_rmse_mean", "valid |U| RMSE [m/s]", True),
    ("sensor_validation_vel_crps_mean", "valid |U| CRPS [m/s]", True),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _safe(d: dict, *keys, default=None):
    """Nested ``dict.get`` that tolerates missing intermediate keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _flatten_summary(run_dir: pathlib.Path, summary: dict) -> dict:
    """One flat record per run: tag-derived axes + scalar metrics."""
    cfg = summary.get("configuration", {}) or {}
    timing = summary.get("timing", {}) or {}
    pm = summary.get("parameter_metrics", {}) or {}
    sm = summary.get("state_metrics", {}) or {}
    sen = summary.get("sensor_metrics", {}) or {}

    rec: dict = {"run_dir": str(run_dir), "name": run_dir.name}

    # Axes: prefer the dir-name tag (carries nx/ny/nz, not in the summary);
    # fall back to the summary configuration for the rest.
    m = _TAG_RE.match(run_dir.name)
    if m:
        rec["assim_model"] = m.group("model")
        rec["nx"], rec["ny"], rec["nz"] = (int(m.group(k)) for k in ("nx", "ny", "nz"))
        rec["ensemble_size"] = int(m.group("ens"))
        rec["num_esmda_steps"] = int(m.group("steps"))
        rec["localization"] = bool(m.group("loc"))
        rec["grid_cells"] = rec["nx"] * rec["ny"] * rec["nz"]
        rec["grid"] = f"{rec['nx']}x{rec['ny']}x{rec['nz']}"
    # Summary fallbacks / extras (override tag only when tag was absent).
    rec.setdefault("assim_model", cfg.get("assimilation_model"))
    rec.setdefault("ensemble_size", cfg.get("ensemble_size"))
    rec.setdefault("num_esmda_steps", cfg.get("num_esmda_steps"))
    rec["truth_model"] = cfg.get("truth_model")
    rec["num_windows"] = cfg.get("num_assimilation_windows")
    rec["sim_time_per_window"] = cfg.get("simulation_time_per_window")
    rec["num_truth_frames"] = cfg.get("num_truth_frames")

    # Timing.
    rec["esmda_total_seconds"] = timing.get("esmda_total_seconds")
    rec["esmda_solve_seconds"] = timing.get("esmda_solve_seconds")
    rec["mean_window_seconds"] = timing.get("mean_window_seconds")

    # Parameter metrics.
    for p in _PARAMS:
        rec[f"param_{p}_rmse_mean"] = _safe(pm, p, "rmse", "mean")
        rec[f"param_{p}_rmse_final"] = _safe(pm, p, "rmse", "final")
        rec[f"param_{p}_crps_mean"] = _safe(pm, p, "crps", "mean")
        rec[f"param_{p}_rmse_reduction"] = _safe(pm, p, "rmse_reduction_vs_prior")

    # State metrics.
    rec["state_vel_rmse_mean"] = _safe(sm, "vel_magnitude_rmse", "mean")
    rec["state_vel_rmse_final"] = _safe(sm, "vel_magnitude_rmse", "final")

    # Sensor metrics (assimilation + validation sets), |U| + per-component u/v/w.
    # Older summaries only carry the |U| (vel_magnitude) keys; the component keys
    # then come back as NaN and are simply skipped when plotting.
    for s in ("assimilation", "validation"):
        rec[f"sensor_{s}_num"] = _safe(sen, s, "num_sensors")
        for q, (_lbl, key) in _SENSOR_Q.items():
            rec[f"sensor_{s}_{q}_rmse_mean"] = _safe(sen, s, f"{key}_rmse", "mean")
            rec[f"sensor_{s}_{q}_crps_mean"] = _safe(sen, s, f"{key}_crps", "mean")
    return rec


def load_runs(root: pathlib.Path, models: list[str] | None) -> pd.DataFrame:
    records = []
    # Prefer compute_sweep_metrics' metrics.yaml; fall back to run_summary.yaml so
    # the script also works when pointed straight at the raw ESMDA results root.
    summaries = sorted(root.glob("*/metrics.yaml")) or sorted(root.glob("*/run_summary.yaml"))
    for summ in summaries:
        run_dir = summ.parent
        try:
            with open(summ) as f:
                summary = yaml.safe_load(f) or {}
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipping {run_dir.name}: cannot read summary ({e})")
            continue
        rec = _flatten_summary(run_dir, summary)
        if models and rec.get("assim_model") not in models:
            continue
        records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["assim_model", "grid_cells", "ensemble_size", "num_esmda_steps"])
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _present_metrics(df: pd.DataFrame,
                     metrics: list[tuple[str, str, bool]]) -> list[tuple[str, str, bool]]:
    return [(c, lbl, lo) for (c, lbl, lo) in metrics
            if c in df.columns and df[c].notna().any()]


# Curated, colour-blind-friendly palette + a distinct marker per backend.
PALETTE = {"pyudales": "#2A6F97", "pypalm": "#E8743B", "pylbm": "#3CA06A"}
_FALLBACK_COLORS = ["#8E44AD", "#C0392B", "#16A085"]
MARKERS = {"pyudales": "o", "pypalm": "s", "pylbm": "D"}
_FALLBACK_MARKERS = ["^", "v", "P", "X"]


def _setup_style() -> None:
    """A clean, consistent house style for every figure."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFBFD",
        "axes.edgecolor": "#5A5A5A",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 10.5,
        "axes.labelweight": "medium",
        "axes.labelcolor": "#1A1A1A",
        "axes.prop_cycle": plt.cycler(color=list(PALETTE.values()) + _FALLBACK_COLORS),
        "grid.color": "#C7C7D1",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.55,
        "xtick.color": "#3A3A3A",
        "ytick.color": "#3A3A3A",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#D0D0D0",
        "legend.borderpad": 0.5,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2.3,
        "lines.markersize": 7.5,
        "lines.markeredgewidth": 1.3,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _model_color(model: str) -> str:
    if model in PALETTE:
        return PALETTE[model]
    return _FALLBACK_COLORS[hash(model) % len(_FALLBACK_COLORS)]


def _model_marker(model: str) -> str:
    if model in MARKERS:
        return MARKERS[model]
    return _FALLBACK_MARKERS[hash(model) % len(_FALLBACK_MARKERS)]


def _series_style(model: str, loc: bool):
    """Stable colour per backend, linestyle per localization."""
    return _model_color(model), ("--" if loc else "-")


def _plot_series(ax, x, y, model: str, loc: bool, label: str) -> None:
    """One styled line: solid colour, white-filled marker, dashed if localized."""
    color, ls = _series_style(model, loc)
    ax.plot(
        x, y,
        color=color, ls=ls, marker=_model_marker(model),
        markerfacecolor="white", markeredgecolor=color,
        label=label, zorder=3, clip_on=False,
    )


def _grid_dims(n: int) -> tuple[int, int]:
    cols = min(3, n) if n else 1
    rows = math.ceil(n / cols) if n else 1
    return rows, cols


def _hold_other_axes(df: pd.DataFrame, sweep_col: str) -> tuple[pd.DataFrame, dict]:
    """Fix the non-swept axes at their modal value so a line varies in one axis."""
    held = {}
    sub = df
    for col, _, _ in AXES:
        if col == sweep_col or col not in sub.columns:
            continue
        vals = sub[col].dropna()
        if vals.nunique() <= 1:
            continue
        mode = Counter(vals).most_common(1)[0][0]
        held[col] = mode
        sub = sub[sub[col] == mode]
    return sub, held


def plot_group_vs_axis(df: pd.DataFrame, sweep_col: str, xlabel: str, logx: bool,
                       out_dir: pathlib.Path, metrics: list[tuple[str, str, bool]],
                       title: str, fname_stub: str, ncols: int = 2) -> None:
    """One figure: every metric in ``metrics`` vs ``sweep_col`` (RMSE next to CRPS)."""
    sub, held = _hold_other_axes(df, sweep_col)
    if sub[sweep_col].dropna().nunique() < 2:
        return
    present = _present_metrics(sub, metrics)
    if not present:
        return
    cols = min(ncols, len(present))
    rows = math.ceil(len(present) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.7 * rows),
                             squeeze=False, constrained_layout=True)
    held_str = ", ".join(f"{k}={v}" for k, v in held.items()) or "none"

    handles_labels = {}
    for ax, (col, lbl, lower_better) in zip(axes.flat, present):
        for (model, loc), g in sub.groupby(["assim_model", "localization"]):
            g = g[[sweep_col, col]].dropna().sort_values(sweep_col)
            if g.empty:
                continue
            label = f"{model}{' + loc' if loc else ''}"
            _plot_series(ax, g[sweep_col], g[col], model, bool(loc), label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lbl)
        arrow = "↓ lower better" if lower_better else "↑ higher better"
        ax.set_title(f"{lbl}\n{arrow}", fontsize=9.5)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.margins(x=0.08, y=0.14)
        _despine(ax)
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles_labels.setdefault(l, h)
    for ax in axes.flat[len(present):]:
        ax.set_visible(False)

    # Shared legend BELOW the panels so it never collides with the suptitle
    # (constrained_layout reserves the space).
    if handles_labels:
        fig.legend(handles_labels.values(), handles_labels.keys(),
                   loc="outside lower center", ncol=min(len(handles_labels), 6))
    fig.suptitle(f"{title}  vs  {xlabel}    ·    other axes held at: {held_str}")
    out = out_dir / f"{fname_stub}_vs_{sweep_col}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_backend_comparison(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Bar chart comparing backends at the single most common (grid, ens, steps)."""
    if df["assim_model"].nunique() < 2:
        return
    keys = [c for c in ("grid_cells", "ensemble_size", "num_esmda_steps") if c in df.columns]
    combo = df.groupby(keys).size().sort_values(ascending=False)
    # Prefer a combo covered by the most backends, then most frequent.
    best, best_score = None, (-1, -1)
    for combo_vals in combo.index:
        mask = np.logical_and.reduce([df[k] == v for k, v in zip(keys, np.atleast_1d(combo_vals))])
        score = (df[mask]["assim_model"].nunique(), int(mask.sum()))
        if score > best_score:
            best, best_score = combo_vals, score
    if best is None or best_score[0] < 2:
        return
    mask = np.logical_and.reduce([df[k] == v for k, v in zip(keys, np.atleast_1d(best))])
    sub = df[mask]
    metrics = _present_metrics(sub, SUMMARY_METRICS)
    if not metrics:
        return

    rows, cols = _grid_dims(len(metrics))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.9 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, (col, lbl, lower_better) in zip(axes.flat, metrics):
        g = sub[["assim_model", "localization", col]].dropna()
        labels = [f"{m}{' + loc' if l else ''}" for m, l in zip(g["assim_model"], g["localization"])]
        colors = [_series_style(m, bool(l))[0] for m, l in zip(g["assim_model"], g["localization"])]
        bars = ax.bar(range(len(g)), g[col], color=colors, edgecolor="white",
                      linewidth=1.2, width=0.68, zorder=3)
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8, color="#333333")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        arrow = "↓ lower better" if lower_better else "↑ higher better"
        ax.set_title(f"{lbl}\n{arrow}", fontsize=9.5)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.margins(y=0.18)
        _despine(ax)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    combo_str = ", ".join(f"{k}={v}" for k, v in zip(keys, np.atleast_1d(best)))
    fig.suptitle(f"Backend comparison at {combo_str}")
    out = out_dir / "backend_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_timing(df: pd.DataFrame, out_dir: pathlib.Path,
                only_cols: list[str] | None = None) -> None:
    if "esmda_total_seconds" not in df.columns or df["esmda_total_seconds"].isna().all():
        return
    sweepable = [(c, lbl, logx) for (c, lbl, logx) in AXES
                 if c in df.columns and df[c].dropna().nunique() >= 2
                 and (only_cols is None or c in only_cols)]
    if not sweepable:
        return
    fig, axes = plt.subplots(1, len(sweepable), figsize=(5.6 * len(sweepable), 4.3),
                             squeeze=False, constrained_layout=True)
    for ax, (col, xlabel, logx) in zip(axes.flat, sweepable):
        sub, held = _hold_other_axes(df, col)
        for (model, loc), g in sub.groupby(["assim_model", "localization"]):
            g = g[[col, "esmda_total_seconds"]].dropna().sort_values(col)
            if g.empty:
                continue
            _plot_series(ax, g[col], g["esmda_total_seconds"] / 60.0, model, bool(loc),
                         f"{model}{' + loc' if loc else ''}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ESMDA total [min]")
        if logx:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.margins(x=0.08, y=0.14)
        ax.legend()
        _despine(ax)
        held_str = ", ".join(f"{k}={v}" for k, v in held.items()) or "none"
        ax.set_title(f"vs {xlabel}\nheld: {held_str}", fontsize=9.5)
    fig.suptitle("ESMDA wall-clock scaling")
    out = out_dir / "timing_scaling.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


def _load_param_mean_std(run_dir: pathlib.Path, fname: str):
    p = run_dir / fname
    if not p.exists():
        return None
    ds = xr.open_dataset(p)
    return ds


def plot_param_trajectories(df: pd.DataFrame, sweep_col: str, xlabel: str,
                            out_dir: pathlib.Path) -> None:
    """Overlay posterior parameter trajectories (mean +/- std) vs truth across a sweep."""
    sub, held = _hold_other_axes(df, sweep_col)
    sub = sub[sub[sweep_col].notna()]
    if sub[sweep_col].nunique() < 2:
        return
    _PRETTY = {"inflow_angle": "inflow angle [deg]", "velocity_magnitude": "velocity magnitude [m/s]"}
    # One figure per backend present (truth is shared).
    for model, gm in sub.groupby("assim_model"):
        gm = gm.sort_values(sweep_col)
        fig, axes = plt.subplots(1, len(_PARAMS), figsize=(6.8 * len(_PARAMS), 4.6),
                                 squeeze=False, constrained_layout=True)
        cmap = plt.get_cmap("plasma")
        vals = gm[sweep_col].to_numpy(dtype=float)
        vmin, vmax = (vals.min(), vals.max()) if len(vals) else (0, 1)
        norm = plt.Normalize(vmin, vmax if vmax > vmin else vmin + 1)
        truth_plotted = False
        any_data = False
        for _, row in gm.iterrows():
            rd = pathlib.Path(row["run_dir"])
            post = _load_param_mean_std(rd, "posterior_params.nc")
            truth = _load_param_mean_std(rd, "true_params.nc")
            if post is None:
                continue
            color = cmap(0.12 + 0.78 * norm(float(row[sweep_col])))
            for j, pname in enumerate(_PARAMS):
                ax = axes[0, j]
                if pname not in post:
                    continue
                any_data = True
                t = np.asarray(post["time"].values, dtype=float)
                da = post[pname]
                mean = da.mean(dim="ensemble").values if "ensemble" in da.dims else da.values
                ax.plot(t, mean, color=color, lw=2.0, zorder=3)
                if "ensemble" in da.dims:
                    std = da.std(dim="ensemble").values
                    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.10, zorder=1)
                if truth is not None and pname in truth and not truth_plotted:
                    tt = np.asarray(truth["time"].values, dtype=float)
                    ax.plot(tt, truth[pname].values, color="#111111", lw=3.0, ls=(0, (4, 2)),
                            label="truth", zorder=5)
            truth_plotted = truth_plotted or (truth is not None)
        if not any_data:
            plt.close(fig)
            continue
        for j, pname in enumerate(_PARAMS):
            ax = axes[0, j]
            ax.set_xlabel("time [s]")
            ax.set_ylabel(_PRETTY.get(pname, pname))
            ax.set_title(_PRETTY.get(pname, pname))
            ax.margins(x=0.02, y=0.12)
            _despine(ax)
            h, l = ax.get_legend_handles_labels()
            if l:
                ax.legend(loc="best")
        # Colour bar encodes the swept value (replaces a long per-line legend).
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.015, fraction=0.045)
        cbar.set_label(xlabel)
        if len(vals):
            cbar.set_ticks(sorted(set(vals)))
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        held_str = ", ".join(f"{k}={v}" for k, v in held.items()) or "none"
        fig.suptitle(f"{model} · posterior parameter trajectories vs truth, coloured by {xlabel}"
                     f"    ·    held: {held_str}")
        out = out_dir / f"param_trajectories_{model}_vs_{sweep_col}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Sensor time-series grids (truth + prior/posterior members + posterior mean)
# ---------------------------------------------------------------------------

_Q_PRETTY = {"vel": "|U|", "u": "u", "v": "v", "w": "w"}


def plot_sensor_timeseries_grid(df: pd.DataFrame, sweep_col: str, xlabel: str,
                                out_dir: pathlib.Path, set_name: str, q: str,
                                max_members: int = 40) -> None:
    """One figure per (model, sensor set, sweep axis, component).

    Grid: rows = sensors, cols = sweep values. Each panel overlays, for component
    ``q``: ground truth (black), prior ensemble members (alpha 0.3), posterior
    ensemble members (alpha 0.5) and the posterior mean (bold). Reads the small
    ``sensor_timeseries_<set>.nc`` files written by compute_sweep_metrics.py.
    """
    sub, held = _hold_other_axes(df, sweep_col)
    sub = sub[sub[sweep_col].notna()]
    if sub.empty:
        return
    for model, gm in sub.groupby("assim_model"):
        gm = gm.sort_values(sweep_col)
        runs = []
        for _, row in gm.iterrows():
            p = pathlib.Path(row["run_dir"]) / f"sensor_timeseries_{set_name}.nc"
            if p.exists():
                runs.append((row[sweep_col], p))
        if len(runs) < 2:
            continue

        with xr.open_dataset(runs[0][1]) as ds0:
            if f"{q}_truth" not in ds0:
                continue
            n_sensors = ds0.sizes["sensor"]
            sx = np.asarray(ds0["sensor_x"].values) if "sensor_x" in ds0 else None
            sy = np.asarray(ds0["sensor_y"].values) if "sensor_y" in ds0 else None
            sz = np.asarray(ds0["sensor_z"].values) if "sensor_z" in ds0 else None

        ncols, nrows = len(runs), n_sensors
        color = _model_color(model)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.4 * nrows),
                                 squeeze=False, sharex=True, constrained_layout=True)
        for ci, (val, path) in enumerate(runs):
            ds = xr.open_dataset(path)
            t = np.asarray(ds["time"].values, dtype=float)
            truth = np.asarray(ds[f"{q}_truth"].values)            # (time, sensor)
            prior = ds[f"{q}_prior"].values if f"{q}_prior" in ds else None
            post = ds[f"{q}_post"].values if f"{q}_post" in ds else None
            for si in range(n_sensors):
                ax = axes[si, ci]
                if prior is not None:
                    mem = prior[:, :, si]
                    idx = np.linspace(0, mem.shape[0] - 1, min(max_members, mem.shape[0])).astype(int)
                    ax.plot(t, mem[idx].T, color="#9AA0A6", lw=0.6, alpha=0.3, zorder=1)
                if post is not None:
                    mem = post[:, :, si]
                    idx = np.linspace(0, mem.shape[0] - 1, min(max_members, mem.shape[0])).astype(int)
                    ax.plot(t, mem[idx].T, color=color, lw=0.6, alpha=0.5, zorder=2)
                    ax.plot(t, mem.mean(axis=0), color=color, lw=2.4, zorder=4,
                            label="posterior mean")
                ax.plot(t, truth[:, si], color="black", lw=2.0, ls=(0, (4, 2)),
                        zorder=5, label="truth")
                _despine(ax)
                ax.margins(x=0.02, y=0.12)
                if ci == 0:
                    loc = f"({sx[si]:g},{sy[si]:g},{sz[si]:g})" if sx is not None else f"S{si}"
                    ax.set_ylabel(f"sensor {si}\n{loc}", fontsize=8)
                if si == 0:
                    ax.set_title(f"{xlabel}\n= {val:g}", fontsize=9.5)
                if si == nrows - 1:
                    ax.set_xlabel("time [s]")

        # Compact shared legend (truth + posterior mean), de-duplicated.
        h, l = axes[0, 0].get_legend_handles_labels()
        seen = dict(zip(l, h))
        if seen:
            fig.legend(seen.values(), seen.keys(), loc="outside lower center",
                       ncol=2)
        held_str = ", ".join(f"{k}={v}" for k, v in held.items()) or "none"
        fig.suptitle(f"{model} · {set_name} sensors · {_Q_PRETTY.get(q, q)} time series "
                     f"vs {xlabel}    ·    truth (black), prior α0.3, posterior α0.5 + mean"
                     f"    ·    held: {held_str}")
        out = out_dir / f"timeseries_{set_name}_{model}_{q}_vs_{sweep_col}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Sweep selection (domain vs ensemble)
# ---------------------------------------------------------------------------

# One sweep == one x-axis. Each writes to its own ``comparison/<sweep>`` folder.
SWEEPS = {
    "domain": ("grid_cells", "grid cells (nx*ny*nz)", True),
    "ensemble": ("ensemble_size", "ensemble size", True),
}


def _pick_domain(df: pd.DataFrame) -> str | None:
    """Grid (``NXxNYxNZ``) with the most distinct ensemble sizes; tie-break most runs."""
    if "grid" not in df.columns:
        return None
    sub = df.dropna(subset=["grid"])
    if sub.empty:
        return None
    stats = sub.groupby("grid").agg(
        n_ens=("ensemble_size", "nunique"),
        n_runs=("name", "size"),
    ).sort_values(["n_ens", "n_runs"], ascending=False)
    return stats.index[0]


def run_one_sweep(df: pd.DataFrame, sweep: str, out_dir: pathlib.Path, args) -> None:
    """Draw every figure for ONE sweep axis into ``out_dir``.

    ``domain``   -- x = grid cells; the ensemble size (+ ESMDA steps) is held at
                    its modal value by ``_hold_other_axes`` inside each plot.
    ``ensemble`` -- x = ensemble size; the domain is FIXED to a single grid first
                    (``--domain`` or the grid with the most ensemble sizes) so it
                    is the only structural axis that varies.
    """
    col, xlabel, logx = SWEEPS[sweep]
    if args.linear_x:
        logx = False

    if sweep == "ensemble":
        target = args.domain or _pick_domain(df)
        if target is not None and "grid" in df.columns:
            df = df[df["grid"] == target]
        if df.empty:
            print(f"[ensemble] no runs at domain {target}; skipping.")
            return
        print(f"[ensemble] domain fixed at {target}")

    present = df[col].dropna() if col in df.columns else df.get(col, pd.Series(dtype=float))
    if present.nunique() < 2:
        print(f"[{sweep}] need >= 2 distinct {col} values to draw a sweep; "
              f"found {sorted(present.dropna().unique().tolist())} -- skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{sweep}] {len(df)} run(s) -> {out_dir}")
    csv = out_dir / "all_runs_metrics.csv"
    df.to_csv(csv, index=False)
    print(f"  wrote {csv.name} ({len(df.columns)} columns)")

    # One figure per metric category (params / assim sensors / validation
    # sensors / state field), RMSE next to CRPS.
    groups = [
        (param_metrics(), "Parameter metrics", "parameters", 3),
        (sensor_metrics_group("assimilation"), "Assimilation sensor metrics", "assim_sensors", 2),
        (sensor_metrics_group("validation"), "Validation sensor metrics", "validation_sensors", 2),
        (STATE_METRICS, "State-field metrics", "state", 2),
    ]
    for metrics, title, stub, ncols in groups:
        plot_group_vs_axis(df, col, xlabel, logx, out_dir, metrics, title, stub, ncols)
    if not args.no_trajectories:
        plot_param_trajectories(df, col, xlabel, out_dir)
    if not args.no_timeseries:
        for set_name in ("assimilation", "validation"):
            for q in args.components:
                plot_sensor_timeseries_grid(df, col, xlabel, out_dir, set_name, q)

    plot_backend_comparison(df, out_dir)
    plot_timing(df, out_dir, only_cols=[col])
    print(f"  done -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--sweep", choices=["domain", "ensemble", "all"], default="all",
                    help="Which sweep to visualize. 'domain' -> figures in "
                         "comparison/domain (x=grid cells); 'ensemble' -> "
                         "comparison/ensemble (x=ensemble size, domain fixed); "
                         "'all' (default) does both.")
    ap.add_argument("--root", type=pathlib.Path, default=repo_root / "sweep_metrics",
                    help="Root holding compute_sweep_metrics output "
                         "(default: <repo>/sweep_metrics). May also point at the raw "
                         "ESMDA results root, in which case run_summary.yaml is used.")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="Base output dir; each sweep lands in <out>/<sweep> "
                         "(default: <repo>/comparison).")
    ap.add_argument("--domain", default=None,
                    help="For the ensemble sweep: fix the grid to this NXxNYxNZ "
                         "(default: the grid with the most ensemble sizes).")
    ap.add_argument("--linear-x", action="store_true",
                    help="Use a linear sweep x-axis instead of log.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these assim backends (e.g. pyudales pylbm).")
    ap.add_argument("--no-trajectories", action="store_true",
                    help="Skip the (slower) per-run NetCDF parameter-trajectory overlays.")
    ap.add_argument("--no-timeseries", action="store_true",
                    help="Skip the per-model sensor time-series grids.")
    ap.add_argument("--components", nargs="*", default=["vel", "u", "v", "w"],
                    help="Velocity quantities for the sensor time-series grids.")
    args = ap.parse_args()

    _setup_style()

    root = args.root
    base_out = args.out or (repo_root / "comparison")
    if not root.exists():
        raise SystemExit(f"metrics root not found: {root}")

    print(f"Scanning {root} ...")
    df = load_runs(root, args.models)
    if df.empty:
        raise SystemExit("No runs with metrics.yaml / run_summary.yaml found.")
    print(f"Loaded {len(df)} run(s): "
          f"{df['assim_model'].value_counts().to_dict()}")

    sweeps = ["domain", "ensemble"] if args.sweep == "all" else [args.sweep]
    for sweep in sweeps:
        run_one_sweep(df.copy(), sweep, base_out / sweep, args)

    print(f"\nDone. Comparison outputs under {base_out}")


if __name__ == "__main__":
    main()
