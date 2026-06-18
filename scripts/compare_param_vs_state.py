"""Compare parameter-only ESMDA against state+parameter ESMDA runs.

Where ``compare_state_runs.py`` compares runs that all estimate the state, this
script compares ACROSS the estimation type: a baseline that estimates only
parameters versus runs that also estimate the state. The headline question is
whether adding state estimation improves the reconstruction of the state, so the
state-error metrics are given their own dedicated figure.

You choose exactly which run folders take part by listing them as arguments
(they may live in different directories -- e.g. parameter-only runs under
``assim_from_ground_truth`` and state runs under ``assim_with_state``). Each run
is classified into one of three categories from its config:

    param_only  -- estimates parameters only      (no state in the control vector)
    state_ic    -- parameters + INITIAL CONDITION  (joint_state_and_parameter, final_time_smoothing off)
    state_all   -- parameters + state THROUGHOUT    (final_time_smoothing on)

Both state categories (ic-only and ic+state) are included so you can see the
progression param_only -> state_ic -> state_all. Bars are coloured by category.

Because a fair comparison needs a shared truth and domain, the script prints a
compatibility check and warns (it does not abort) when the selected runs differ
in domain size, window interval, or ground truth -- pick comparable runs.

Figures (under result_figures/<out> ; default result_figures/param_vs_state):
    * state_reconstruction.png -- state |U| RMSE (mean & final) per run, by
                                  category, with % improvement vs the param_only
                                  baseline annotated -- the headline result
    * metric_comparison.png    -- full grid: parameter, state and sensor metrics
    * comparison_table.png / comparison.csv -- every metric for every run

Usage:

    python scripts/compare_param_vs_state.py RUN_PARAM_ONLY RUN_STATE_IC RUN_STATE_ALL [...]
    python scripts/compare_param_vs_state.py /p/.../assim_from_ground_truth/pyudales_... \\
        /p/.../assim_with_state/pyudales_..._svd_ic /p/.../assim_with_state/pyudales_..._svd_all
    python scripts/compare_param_vs_state.py RUN_A RUN_B --out result_figures/my_cmp
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from visualize_run import _load_yaml  # noqa: E402
from visualize_state_run import detect_mode_and_method  # noqa: E402
from compare_state_runs import _METRICS, _dig  # noqa: E402

# Category drives the colour; ordered so the legend / x-axis read as a
# progression from the baseline to the richest control vector.
_CATEGORY_ORDER = ["param_only", "state_ic", "state_all"]
_CATEGORY_COLORS = {
    "param_only": "#7f7f7f",   # grey baseline
    "state_ic": "#1f77b4",     # blue
    "state_all": "#2ca02c",    # green
}
_CATEGORY_LABEL = {
    "param_only": "parameters only",
    "state_ic": "params + initial condition",
    "state_all": "params + state throughout",
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def detect_estimation_type(run_dir: pathlib.Path) -> str:
    """Classify a run as ``param_only`` / ``state_ic`` / ``state_all``.

    Uses the smoother's ``final_time_smoothing`` (state adjusted throughout the
    window => ``state_all``) and ``joint_state_and_parameter`` (initial-condition
    state in the control vector => ``state_ic``); neither => ``param_only``.
    """
    cfg = _load_yaml(run_dir / "config.yaml")
    esmda = cfg.get("esmda", {}) or {}
    summary = _load_yaml(run_dir / "run_summary.yaml")
    conf = summary.get("configuration", {}) or {}

    final_time_smoothing = esmda.get("final_time_smoothing")
    joint = conf.get("joint_state_and_parameter")
    if joint is None:  # fall back to the smoother class name
        smoother = (esmda.get("smoother") or {}).get("_target_", "")
        joint = "StateAnd" in smoother

    if final_time_smoothing:
        return "state_all"
    if joint:
        return "state_ic"
    return "param_only"


def _short_label(run_dir: pathlib.Path, category: str) -> str:
    """Compact x-axis label, e.g. ``ens64 int10 [param_only]``."""
    name = run_dir.name
    ens = re.search(r"_ens(\d+)", name)
    interval = re.search(r"_int([\d.]+)", name)
    method = detect_mode_and_method(run_dir)["method"]
    parts = []
    if ens:
        parts.append(f"ens{ens.group(1)}")
    if interval:
        parts.append(f"int{interval.group(1).rstrip('.')}")
    if method not in ("plain", None):
        parts.append(method.replace("localization_", "loc_"))
    return " ".join(parts) + f" [{category}]"


def _context(run_dir: pathlib.Path) -> dict:
    """Domain / truth / setup fields used for the compatibility check."""
    cfg = _load_yaml(run_dir / "config.yaml")
    dom = cfg.get("domain", {}) or {}
    conf = (_load_yaml(run_dir / "run_summary.yaml").get("configuration", {}) or {})
    return {
        "domain": (dom.get("nx"), dom.get("ny"), dom.get("nz")),
        "interval": conf.get("simulation_time_per_window"),
        "final_time": conf.get("final_time"),
        "num_truth_frames": conf.get("num_truth_frames"),
        "truth_model": conf.get("truth_model"),
        "assim_model": conf.get("assimilation_model"),
    }


def collect_runs(run_dirs: list[pathlib.Path]) -> list[dict]:
    rows: list[dict] = []
    for d in run_dirs:
        d = d.resolve()
        if not d.is_dir():
            print(f"  ! not a directory, skipping: {d}")
            continue
        summary = _load_yaml(d / "run_summary.yaml")
        if not summary or "parameter_metrics" not in summary:
            print(f"  ! incomplete run (no metrics), skipping: {d.name}")
            continue
        category = detect_estimation_type(d)
        rows.append({
            "name": d.name,
            "category": category,
            "label": _short_label(d, category),
            "metrics": {key: _dig(summary, *path) for key, _, path in _METRICS},
            "context": _context(d),
        })
    rows.sort(key=lambda r: (_CATEGORY_ORDER.index(r["category"])
                             if r["category"] in _CATEGORY_ORDER else 99, r["name"]))
    return rows


def compatibility_check(rows: list[dict]) -> None:
    """Warn when selected runs are not directly comparable (different setups)."""
    def uniq(field):
        return {repr(r["context"].get(field)) for r in rows}

    issues = []
    for field, msg in [
        ("domain", "domain size (nx,ny,nz)"),
        ("interval", "window interval (simulation_time_per_window)"),
        ("num_truth_frames", "ground-truth length (num_truth_frames)"),
        ("truth_model", "truth model"),
    ]:
        if len(uniq(field)) > 1:
            vals = ", ".join(sorted(uniq(field)))
            issues.append(f"    - differing {msg}: {vals}")
    if issues:
        print("  ! WARNING: selected runs differ in setup; the comparison may not")
        print("            be apples-to-apples. Differences:")
        print("\n".join(issues))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_state_reconstruction(rows: list[dict], out_path: pathlib.Path) -> None:
    """Headline: state |U| RMSE per run, by category, with improvement vs baseline."""
    x = np.arange(len(rows))
    colors = [_CATEGORY_COLORS.get(r["category"], "#7f7f7f") for r in rows]
    labels = [r["label"] for r in rows]

    def vals(key):
        return np.array([np.nan if r["metrics"].get(key) is None
                         else float(r["metrics"][key]) for r in rows])

    rmse_mean = vals("state_rmse_mean")
    rmse_final = vals("state_rmse_final")

    # Baseline = mean state RMSE over the param_only runs (if any).
    base_idx = [i for i, r in enumerate(rows) if r["category"] == "param_only"]
    baseline = float(np.nanmean(rmse_mean[base_idx])) if base_idx else None

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for ax, data, title in [
        (ax0, rmse_mean, "State |U| RMSE (window mean)"),
        (ax1, rmse_final, "State |U| RMSE (final window)"),
    ]:
        bars = ax.bar(x, data, color=colors)
        for b, v in zip(bars, data):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                        ha="center", va="bottom", fontsize=9)
        if baseline is not None:
            ax.axhline(baseline, color="#7f7f7f", linestyle="--", linewidth=1.3,
                       label=f"param_only baseline ({baseline:.3g})")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("RMSE  (lower = better reconstruction)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.3)

    # Annotate % improvement of each state run's mean RMSE vs the baseline.
    if baseline is not None:
        for i, r in enumerate(rows):
            if r["category"] != "param_only" and not np.isnan(rmse_mean[i]):
                impr = 100.0 * (baseline - rmse_mean[i]) / baseline
                ax0.annotate(f"{impr:+.0f}%", (i, rmse_mean[i]),
                             textcoords="offset points", xytext=(0, 14),
                             ha="center", fontsize=8,
                             color="#2ca02c" if impr > 0 else "#d62728", fontweight="bold")

    from matplotlib.patches import Patch
    cats = [c for c in _CATEGORY_ORDER if any(r["category"] == c for r in rows)]
    handles = [Patch(facecolor=_CATEGORY_COLORS[c], label=_CATEGORY_LABEL[c]) for c in cats]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.04), fontsize=10, frameon=False)
    fig.suptitle("Does state estimation improve state reconstruction?",
                 fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_panels(rows: list[dict], out_path: pathlib.Path) -> None:
    """Full grid of parameter / state / sensor metrics; bars coloured by category."""
    x = np.arange(len(rows))
    colors = [_CATEGORY_COLORS.get(r["category"], "#7f7f7f") for r in rows]
    labels = [r["label"] for r in rows]

    ncols = 3
    nrows = int(np.ceil(len(_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, label, _) in zip(axes, _METRICS):
        data = [np.nan if r["metrics"].get(key) is None else float(r["metrics"][key]) for r in rows]
        bars = ax.bar(x, data, color=colors)
        for b, v in zip(bars, data):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[len(_METRICS):]:
        ax.axis("off")

    from matplotlib.patches import Patch
    cats = [c for c in _CATEGORY_ORDER if any(r["category"] == c for r in rows)]
    handles = [Patch(facecolor=_CATEGORY_COLORS[c], label=_CATEGORY_LABEL[c]) for c in cats]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)
    fig.suptitle("Parameter-only vs state+parameter: all metrics",
                 fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_table_and_csv(rows: list[dict], png_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    header = ["run", "category"] + [key for key, _, _ in _METRICS]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["name"], r["category"]] + [r["metrics"].get(k) for k, _, _ in _METRICS])

    show_cols = [
        ("state_rmse_mean", "state RMSE\n(mean)"),
        ("state_rmse_final", "state RMSE\n(final)"),
        ("param_vel_red", "vel RMSE\nreduction"),
        ("param_angle_red", "angle RMSE\nreduction"),
        ("sensor_valid_rmse", "sensor RMSE\n(valid)"),
    ]
    cell_text = []
    for r in rows:
        line = [r["label"]]
        for key, _ in show_cols:
            v = r["metrics"].get(key)
            line.append("-" if v is None else f"{float(v):.3g}")
        cell_text.append(line)

    fig_h = max(2.0, 0.5 + 0.4 * len(rows))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text,
                     colLabels=["run"] + [lbl for _, lbl in show_cols],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Parameter-only vs state+parameter summary", fontsize=14, fontweight="bold")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", type=pathlib.Path, nargs="+",
                    help="run folders to compare (parameter-only and/or state runs)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output dir (default: <repo>/result_figures/param_vs_state)")
    args = ap.parse_args()

    out_dir = args.out or (repo_root / "result_figures" / "param_vs_state")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(args.runs)
    if not rows:
        raise SystemExit("no complete runs to compare")

    print(f"Comparing {len(rows)} run(s):")
    for r in rows:
        print(f"    [{r['category']:<10}] {r['name']}")
    compatibility_check(rows)
    if not any(r["category"] == "param_only" for r in rows):
        print("  (note: no param_only run selected -> no baseline line / improvement %)")

    plot_state_reconstruction(rows, out_dir / "state_reconstruction.png")
    print(f"  + {out_dir/'state_reconstruction.png'}")
    plot_metric_panels(rows, out_dir / "metric_comparison.png")
    print(f"  + {out_dir/'metric_comparison.png'}")
    write_table_and_csv(rows, out_dir / "comparison_table.png", out_dir / "comparison.csv")
    print(f"  + {out_dir/'comparison_table.png'}")
    print(f"  + {out_dir/'comparison.csv'}")
    print(f"Done. Comparison in {out_dir}")


if __name__ == "__main__":
    main()
