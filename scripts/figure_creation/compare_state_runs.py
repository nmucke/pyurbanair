"""Compare multiple state-estimation ESMDA runs on a common set of metrics.

Companion to ``scripts/visualize_state_run.py``. It scans a directory of run
folders (default ``/projects/prjs2075/urbanair/assim_with_state``), reads each
run's ``run_summary.yaml``, and draws grouped comparison figures plus a summary
table / CSV. Bars are coloured by METHOD (svd / localization_corr /
localization_dist_dist) and labelled by MODE so the method's effect is visible
within each mode filter.

The ``--mode`` filter selects which runs go into the comparison:

    --mode ic     only initial-condition runs   (final_time_smoothing: false, "_ic")
    --mode all    only state-throughout runs     (final_time_smoothing: true,  "_all")
    --mode both   every run, ic and all together (default)

Runs that are still incomplete (no ``run_summary.yaml`` with metrics yet -- e.g.
results mid-transfer) are skipped with a printed note, so this is safe to run
while results are still arriving.

Metrics compared (from run_summary.yaml), each as one bar-chart panel:
    * parameter RMSE (mean) -- inflow_angle, velocity_magnitude
    * parameter RMSE reduction vs prior
    * parameter CRPS (mean)
    * state |U| RMSE (mean & final)
    * sensor velocity-vector RMSE (mean) -- assimilation & validation
    * sensor energy score (mean)         -- assimilation & validation

Usage:

    python scripts/compare_state_runs.py                      # all runs, default root
    python scripts/compare_state_runs.py --mode ic
    python scripts/compare_state_runs.py --mode all
    python scripts/compare_state_runs.py --root /path/to/runs --mode both --out result_figures/cmp
"""

# mypy: ignore-errors
# Legacy untyped plotting CLI. Keep the waiver local until the module receives
# a dedicated typing pass.

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from visualize_run import _load_yaml  # noqa: E402
from visualize_state_run import detect_mode_and_method  # noqa: E402

_DEFAULT_ROOT = "/projects/prjs2075/urbanair/assim_with_state"

# Stable colour per method so the same method reads the same across figures.
_METHOD_COLORS = {
    "svd": "#1f77b4",
    "localization_corr": "#2ca02c",
    "localization_dist_dist": "#d62728",
    "plain": "#7f7f7f",
}
_MODE_HATCH = {"ic": "", "all": "//"}


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _dig(d: dict, *keys, default=None):
    """Nested ``dict.get`` that tolerates missing intermediate keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


# (column key, human label, nested path into run_summary). Kept declarative so
# the panels, the table and the CSV all stay in sync.
_METRICS: list[tuple[str, str, tuple]] = [
    (
        "param_angle_rmse",
        "inflow_angle RMSE (mean)",
        ("parameter_metrics", "inflow_angle", "rmse", "mean"),
    ),
    (
        "param_vel_rmse",
        "velocity_mag RMSE (mean)",
        ("parameter_metrics", "velocity_magnitude", "rmse", "mean"),
    ),
    (
        "param_angle_red",
        "inflow_angle RMSE reduction",
        ("parameter_metrics", "inflow_angle", "rmse_reduction_vs_prior"),
    ),
    (
        "param_vel_red",
        "velocity_mag RMSE reduction",
        ("parameter_metrics", "velocity_magnitude", "rmse_reduction_vs_prior"),
    ),
    (
        "param_angle_crps",
        "inflow_angle CRPS (mean)",
        ("parameter_metrics", "inflow_angle", "crps", "mean"),
    ),
    (
        "param_vel_crps",
        "velocity_mag CRPS (mean)",
        ("parameter_metrics", "velocity_magnitude", "crps", "mean"),
    ),
    (
        "state_rmse_mean",
        "state |U| RMSE (mean)",
        ("state_metrics", "vel_magnitude_rmse", "mean"),
    ),
    (
        "state_rmse_final",
        "state |U| RMSE (final)",
        ("state_metrics", "vel_magnitude_rmse", "final"),
    ),
    (
        "sensor_assim_rmse",
        "sensor RMSE assim (mean)",
        ("sensor_metrics", "assimilation", "velocity_vector_rmse", "mean"),
    ),
    (
        "sensor_valid_rmse",
        "sensor RMSE valid (mean)",
        ("sensor_metrics", "validation", "velocity_vector_rmse", "mean"),
    ),
    (
        "sensor_assim_es",
        "sensor energy assim (mean)",
        ("sensor_metrics", "assimilation", "velocity_vector_energy_score", "mean"),
    ),
    (
        "sensor_valid_es",
        "sensor energy valid (mean)",
        ("sensor_metrics", "validation", "velocity_vector_energy_score", "mean"),
    ),
]


def _short_label(name: str, mode: str, method: str) -> str:
    """Compact x-axis label, e.g. ``ens128 int6 svd [all]``."""
    ens = re.search(r"_ens(\d+)", name)
    interval = re.search(r"_int([\d.]+)", name)
    parts = []
    if ens:
        parts.append(f"ens{ens.group(1)}")
    if interval:
        parts.append(f"int{interval.group(1).rstrip('.')}")
    parts.append(method.replace("localization_", "loc_"))
    return " ".join(parts) + f" [{mode}]"


def collect_runs(root: pathlib.Path, mode_filter: str) -> list[dict]:
    """Discover complete runs under ``root`` matching the mode filter."""
    rows: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        info = detect_mode_and_method(d)
        mode = info["mode"]
        if mode_filter != "both" and mode != mode_filter:
            continue
        summary = _load_yaml(d / "run_summary.yaml")
        if not summary or "parameter_metrics" not in summary:
            print(f"  - skipping incomplete run (no metrics yet): {d.name}")
            continue
        row = {
            "name": d.name,
            "mode": mode,
            "method": info["method"],
            "label": _short_label(d.name, mode, info["method"]),
            "metrics_version": int(summary.get("metrics_version", 1)),
            "metrics": {key: _dig(summary, *path) for key, _, path in _METRICS},
            "timing": _dig(summary, "timing", "esmda_total_seconds"),
        }
        rows.append(row)
    # Group visually by mode then method then name.
    rows.sort(key=lambda r: (r["mode"], r["method"], r["name"]))
    versions = sorted({row["metrics_version"] for row in rows})
    if len(versions) > 1:
        warnings.warn(
            "Comparing summaries with mismatched metrics versions "
            f"{versions}; CRPS, energy-score, and spread values are not "
            "comparable across the version-1/version-2 boundary.",
            UserWarning,
            stacklevel=2,
        )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_metric_panels(
    rows: list[dict], out_path: pathlib.Path, mode_filter: str
) -> None:
    """One bar-chart panel per metric; bars coloured by method, hatched by mode."""
    n = len(rows)
    x = np.arange(n)
    colors = [_METHOD_COLORS.get(r["method"], "#7f7f7f") for r in rows]
    hatches = [_MODE_HATCH.get(r["mode"], "") for r in rows]
    labels = [r["label"] for r in rows]

    ncols = 3
    nrows = int(np.ceil(len(_METRICS) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, label, _) in zip(axes, _METRICS):
        vals = [
            np.nan if r["metrics"].get(key) is None else float(r["metrics"][key])
            for r in rows
        ]
        bars = ax.bar(x, vals, color=colors)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v,
                    f"{v:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[len(_METRICS) :]:
        ax.axis("off")

    # Legends: method -> colour, mode -> hatch.
    from matplotlib.patches import Patch

    methods = sorted({r["method"] for r in rows})
    modes = sorted({r["mode"] for r in rows})
    method_handles = [
        Patch(facecolor=_METHOD_COLORS.get(m, "#7f7f7f"), label=m) for m in methods
    ]
    mode_handles = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=_MODE_HATCH.get(m, ""),
            label=f"mode: {m}",
        )
        for m in modes
    ]
    fig.legend(
        handles=method_handles + mode_handles,
        loc="lower center",
        ncol=max(2, len(method_handles) + len(mode_handles)),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        f"State-estimation run comparison (mode: {mode_filter}, n={n})",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_table_and_csv(
    rows: list[dict], png_path: pathlib.Path, csv_path: pathlib.Path
) -> None:
    """A text-table PNG and a CSV of every metric for every run."""
    header = ["run", "mode", "method"] + [key for key, _, _ in _METRICS]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(
                [r["name"], r["mode"], r["method"]]
                + [r["metrics"].get(k) for k, _, _ in _METRICS]
            )

    # Compact PNG table: rows = runs, cols = a readable subset of metrics.
    show_cols = [
        ("param_vel_red", "vel RMSE\nreduction"),
        ("param_angle_red", "angle RMSE\nreduction"),
        ("state_rmse_mean", "state\nRMSE"),
        ("sensor_valid_rmse", "sensor RMSE\n(valid)"),
        ("sensor_valid_es", "energy\n(valid)"),
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
    table = ax.table(
        cellText=cell_text,
        colLabels=["run"] + [lbl for _, lbl in show_cols],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("State-estimation comparison summary", fontsize=14, fontweight="bold")
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
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(_DEFAULT_ROOT),
        help=f"directory of run folders (default: {_DEFAULT_ROOT})",
    )
    ap.add_argument(
        "--mode",
        choices=["ic", "all", "both"],
        default="both",
        help="which runs to compare (default: both)",
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="output dir (default: <repo>/result_figures/comparison_<mode>)",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root folder not found: {root}")

    out_dir = args.out or (repo_root / "result_figures" / f"comparison_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root:   {root}")
    print(f"Mode:   {args.mode}")
    rows = collect_runs(root, args.mode)
    if not rows:
        raise SystemExit(f"no complete runs found for mode '{args.mode}' under {root}")

    print(f"Comparing {len(rows)} run(s):")
    for r in rows:
        print(f"    [{r['mode']:>3}] {r['method']:<22} {r['name']}")

    plot_metric_panels(rows, out_dir / "metric_comparison.png", args.mode)
    print(f"  + {out_dir/'metric_comparison.png'}")
    write_table_and_csv(
        rows, out_dir / "comparison_table.png", out_dir / "comparison.csv"
    )
    print(f"  + {out_dir/'comparison_table.png'}")
    print(f"  + {out_dir/'comparison.csv'}")
    print(f"Done. Comparison in {out_dir}")


if __name__ == "__main__":
    main()
