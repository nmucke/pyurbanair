"""Cross-cutting summary figures (spec §7).

Produces, under ``<out>/summary/``:
  S1_cost_vs_accuracy.pdf  -- scatter of computational cost vs state |U| field RMSE,
                              one point per Block-A run (model x resolution) and per
                              Block-B method (uDALES); model->color, resolution->size,
                              method->marker.
  S2_headline_metric.pdf   -- grouped bar of one headline number (validation-sensor
                              |U| RMSE): models at finest res (param-only) vs uDALES
                              Block-B methods.
  S3_walltime.pdf          -- total ESMDA wall-clock per run: Block-A model x
                              resolution (left) and Block-B method (right), from
                              each run's run_summary.yaml ``timing`` block.

This script is LIGHT: it reads the small per-run field-metric caches
``<out>/_cache/block_{a,b}_metrics.json`` (written by the block drivers' ``--heavy``
pass) plus each run's ``run_summary.yaml``. It runs fully on the login node.

If a cache is missing/empty the per-run state |U| field RMSE is taken from
``run_summary.yaml`` ``state_metrics.vel_magnitude_rmse.mean``. CAVEAT: that RMSE is
computed on each run's OWN grid (not the common truth grid), so cross-resolution
comparisons are only approximate in that fallback mode. The provenance is annotated
on the figures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figspec import dataio
from figspec import style as S


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------
COST_LABELS = {
    "walltime": "ESMDA wall-clock total [s]",
    "forward_runs": r"# forward model runs ($N_e \cdot N_{win} \cdot (N_{step}+1)$)",
    "state_dim": "updated state-vector dimension",
}


def _cfg_int(cfg: dict, key: str) -> int | None:
    v = cfg.get(key)
    return None if v is None else int(v)


def forward_runs(cfg: dict) -> float | None:
    """N_e * num_windows * (num_esmda_steps + 1) forward model evaluations."""
    ne = _cfg_int(cfg, "ensemble_size")
    nwin = _cfg_int(cfg, "num_assimilation_windows")
    nstep = _cfg_int(cfg, "num_esmda_steps")
    if ne is None or nwin is None or nstep is None:
        return None
    return float(ne * nwin * (nstep + 1))


def state_dim(run: dataio.Run) -> float | None:
    """Updated state-vector dimension (fluid cells * components; here nx*ny*nz)."""
    return float(run.nx * run.ny * run.nz_eff)


def run_cost(run: dataio.Run, summary: dict, cost: str) -> float | None:
    """Return the requested cost for a run, or None if unavailable."""
    cfg = summary.get("configuration", {}) or {}
    timing = summary.get("timing", {}) or {}
    if cost == "walltime":
        return timing.get("esmda_total_seconds")
    if cost == "forward_runs":
        return forward_runs(cfg)
    if cost == "state_dim":
        return state_dim(run)
    return None


def field_rmse_from_cache(cache: dict, run: dataio.Run) -> float | None:
    entry = cache.get(run.name)
    if entry is None:
        return None
    v = entry.get("field_rmse")
    return None if v is None else float(v)


def field_rmse_fallback(summary: dict) -> float | None:
    """Own-grid |U| RMSE from run_summary (used when a cache entry is absent)."""
    sm = (summary.get("state_metrics", {}) or {}).get("vel_magnitude_rmse", {}) or {}
    v = sm.get("mean")
    return None if v is None else float(v)


def valsensor_rmse(cache: dict, run: dataio.Run, summary: dict) -> float | None:
    """Validation-sensor |U| RMSE (cache) or vector RMSE fallback (run_summary)."""
    entry = cache.get(run.name)
    if entry is not None and entry.get("valsensor_rmse") is not None:
        return float(entry["valsensor_rmse"])
    val = (summary.get("sensor_metrics", {}) or {}).get("validation", {}) or {}
    vrmse = (val.get("velocity_vector_rmse", {}) or {}).get("mean")
    return None if vrmse is None else float(vrmse)


def load_cache(path: pathlib.Path, label: str) -> dict:
    if not path.exists():
        print(
            f"  WARNING: {label} cache missing ({path}); "
            "falling back to run_summary own-grid |U| RMSE."
        )
        return {}
    try:
        with open(path) as f:
            data = json.load(f) or {}
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: could not read {label} cache ({exc}); falling back.")
        return {}
    if not data:
        print(f"  WARNING: {label} cache empty; falling back to run_summary.")
    return data


# ---------------------------------------------------------------------------
# Build the point records shared by S1/S2
# ---------------------------------------------------------------------------
def collect_records(
    out: pathlib.Path, cost: str
) -> tuple[list[dict], list[dict], bool]:
    """Return (block_a_records, block_b_records, used_fallback_field_rmse)."""
    cache_a = load_cache(out / "_cache" / "block_a_metrics.json", "Block A")
    cache_b = load_cache(out / "_cache" / "block_b_metrics.json", "Block B")

    used_fallback = False
    rec_a: list[dict] = []
    for r in dataio.discover_block_a():
        if not r.has("run_summary.yaml"):
            continue
        summary = dataio.load_summary(r)
        rmse = field_rmse_from_cache(cache_a, r)
        if rmse is None:
            rmse = field_rmse_fallback(summary)
            if rmse is not None:
                used_fallback = True
        c = run_cost(r, summary, cost)
        if rmse is None or c is None:
            continue
        rec_a.append(
            dict(
                name=r.name,
                model=r.model,
                method="param_only",
                dx=r.dx,
                nx=r.nx,
                res_label=r.res_label_dx,
                cost=c,
                field_rmse=rmse,
                valsensor=valsensor_rmse(cache_a, r, summary),
            )
        )

    rec_b: list[dict] = []
    for r in dataio.discover_block_b():
        if not r.has("run_summary.yaml"):
            continue
        method = r.method
        if method is None:
            continue
        summary = dataio.load_summary(r)
        rmse = field_rmse_from_cache(cache_b, r)
        if rmse is None:
            rmse = field_rmse_fallback(summary)
            if rmse is not None:
                used_fallback = True
        c = run_cost(r, summary, cost)
        if rmse is None or c is None:
            continue
        rec_b.append(
            dict(
                name=r.name,
                model=r.model,
                method=method,
                dx=r.dx,
                nx=r.nx,
                res_label=r.res_label_dx,
                cost=c,
                field_rmse=rmse,
                valsensor=valsensor_rmse(cache_b, r, summary),
            )
        )

    return rec_a, rec_b, used_fallback


def _resolve_cost(
    out: pathlib.Path, cost: str
) -> tuple[list[dict], list[dict], str, bool]:
    """Collect records, auto-falling back to a cost metric every run has."""
    order = [cost] + [c for c in ("walltime", "forward_runs", "state_dim") if c != cost]
    for cand in order:
        rec_a, rec_b, used_fallback = collect_records(out, cand)
        if rec_a or rec_b:
            if cand != cost:
                print(
                    f"  NOTE: cost '{cost}' unavailable for all runs; "
                    f"using '{cand}' instead."
                )
            return rec_a, rec_b, cand, used_fallback
    return [], [], cost, False


# ---------------------------------------------------------------------------
# S1 -- cost vs accuracy scatter
# ---------------------------------------------------------------------------
def _marker_size(nx: int, nx_all: list[int]) -> float:
    """Map nx -> marker area; finer (larger nx) -> bigger marker."""
    lo, hi = min(nx_all), max(nx_all)
    if hi == lo:
        return 90.0
    frac = (nx - lo) / (hi - lo)
    return 45.0 + frac * 230.0


def s1_cost_vs_accuracy(
    rec_a: list[dict],
    rec_b: list[dict],
    outdir: pathlib.Path,
    *,
    cost: str,
    used_fallback: bool,
    also_png: bool = True,
) -> pathlib.Path | None:
    records = rec_a + rec_b
    if not records:
        print("  S1: no records to plot; skipping.")
        return None

    nx_all = [r["nx"] for r in records]
    costs = [r["cost"] for r in records]
    log_x = (max(costs) / max(min(costs), 1e-12)) >= 50.0

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)

    for r in records:
        color = S.MODEL_COLORS.get(r["model"], S.COLORS["charcoal"])
        marker = S.METHOD_MARKERS.get(r["method"], S.MODEL_MARKERS.get(r["model"], "o"))
        ax.scatter(
            r["cost"],
            r["field_rmse"],
            s=_marker_size(r["nx"], nx_all),
            facecolor=color,
            edgecolor="0.2",
            linewidth=0.7,
            marker=marker,
            alpha=0.85,
            zorder=3,
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(COST_LABELS.get(cost, cost))
    ax.set_ylabel(
        r"state $|U|$ field RMSE [m/s]"
        + (" (own-grid)" if used_fallback else " (common grid)")
    )
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.05)

    _s1_legends(ax, records, nx_all)

    if used_fallback:
        fig.text(
            0.5,
            -0.01,
            "Note: |U| RMSE from run_summary own-grid (caches absent) -- "
            "cross-resolution comparison approximate.",
            ha="center",
            va="top",
            fontsize=7,
            color="0.35",
        )

    if also_png:
        S.save_png(fig, outdir / "S1_cost_vs_accuracy.png", dpi=160)
    return S.save_pdf(fig, outdir / "S1_cost_vs_accuracy.pdf")


def _s1_legends(ax, records: list[dict], nx_all: list[int]) -> None:
    """Three legend blocks: model colour, method marker, resolution size."""
    models = sorted(
        {r["model"] for r in records},
        key=lambda m: list(S.MODEL_LABELS).index(m) if m in S.MODEL_LABELS else 99,
    )
    color_h = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=S.MODEL_COLORS.get(m, S.COLORS["charcoal"]),
            markeredgecolor="0.2",
            markersize=8,
            label=S.MODEL_LABELS.get(m, m),
        )
        for m in models
    ]

    methods = sorted(
        {r["method"] for r in records},
        key=lambda m: S.METHOD_ORDER.index(m) if m in S.METHOD_ORDER else 99,
    )
    method_h = [
        Line2D(
            [0],
            [0],
            marker=S.METHOD_MARKERS.get(m, "o"),
            linestyle="none",
            markerfacecolor="0.6",
            markeredgecolor="0.2",
            markersize=8,
            label=S.METHOD_LABELS.get(m, m),
        )
        for m in methods
    ]

    nx_set = sorted(set(nx_all))
    size_nx = [nx_set[0], nx_set[-1]] if len(nx_set) > 1 else nx_set
    size_h = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.6",
            markeredgecolor="0.2",
            markersize=np.sqrt(_marker_size(nx, nx_all)),
            label=f"nx={nx} (Δx={60.0/nx:.2g} m)",
        )
        for nx in size_nx
    ]

    leg1 = ax.legend(
        handles=color_h,
        title="model (colour)",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
    )
    leg2 = ax.legend(
        handles=method_h,
        title="method (marker)",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.74),
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(leg1)
    if size_h:
        ax.legend(
            handles=size_h,
            title="resolution (size)",
            loc="upper right",
            bbox_to_anchor=(1.0, 0.40),
            fontsize=8,
            title_fontsize=8,
        )
        ax.add_artist(leg2)


# ---------------------------------------------------------------------------
# S2 -- headline metric grouped bar
# ---------------------------------------------------------------------------
def _finest_param_only(rec_a: list[dict]) -> dict[str, dict]:
    """Finest-resolution (largest nx) param-only record per model."""
    best: dict[str, dict] = {}
    for r in rec_a:
        if r["model"] not in best or r["nx"] > best[r["model"]]["nx"]:
            best[r["model"]] = r
    return best


def s2_headline_metric(
    rec_a: list[dict], rec_b: list[dict], outdir: pathlib.Path, *, used_fallback: bool
) -> pathlib.Path | None:
    finest = _finest_param_only(rec_a)
    g1 = [finest[m] for m in ("palm", "udales", "lbm") if m in finest]
    g1 = [r for r in g1 if r.get("valsensor") is not None]
    g2 = sorted(
        (r for r in rec_b if r.get("valsensor") is not None),
        key=lambda r: (
            S.METHOD_ORDER.index(r["method"]) if r["method"] in S.METHOD_ORDER else 99
        ),
    )
    if not g1 and not g2:
        print("  S2: no validation-sensor RMSE available; skipping.")
        return None

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    labels, values, colors = [], [], []
    for r in g1:
        labels.append(
            f"{S.MODEL_LABELS.get(r['model'], r['model'])}\n({r['res_label']})"
        )
        values.append(r["valsensor"])
        colors.append(S.MODEL_COLORS.get(r["model"], S.COLORS["charcoal"]))
    gap = 1 if (g1 and g2) else 0
    for r in g2:
        labels.append(S.METHOD_LABELS.get(r["method"], r["method"]))
        values.append(r["valsensor"])
        colors.append(S.METHOD_COLORS.get(r["method"], S.COLORS["posterior"]))

    x = list(range(len(g1))) + [len(g1) + gap + i for i in range(len(g2))]
    ax.bar(x, values, color=colors, edgecolor="0.2", linewidth=0.7, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(
        r"validation-sensor $|U|$ RMSE [m/s]"
        + (" (vector RMSE proxy)" if used_fallback else "")
    )
    ax.set_ylim(bottom=0.0)

    if g1 and g2:
        sep = len(g1) - 0.5 + gap / 2.0
        ax.axvline(sep, color="0.7", ls="--", lw=0.9)
        ymax = ax.get_ylim()[1]
        ax.text(
            (len(g1) - 1) / 2.0,
            ymax * 0.97,
            "models (finest res)",
            ha="center",
            va="top",
            fontsize=8,
            color="0.4",
        )
        ax.text(
            len(g1) + gap + (len(g2) - 1) / 2.0,
            ymax * 0.97,
            "uDALES methods",
            ha="center",
            va="top",
            fontsize=8,
            color="0.4",
        )
    if used_fallback:
        fig.text(
            0.5,
            -0.02,
            "Note: validation-sensor metric from run_summary (vector RMSE; "
            "caches absent).",
            ha="center",
            va="top",
            fontsize=7,
            color="0.35",
        )
    return S.save_pdf(fig, outdir / "S2_headline_metric.pdf")


# ---------------------------------------------------------------------------
# S3 -- wall-clock totals
# ---------------------------------------------------------------------------
def _walltime(summary: dict) -> float | None:
    t = (summary.get("timing", {}) or {}).get("esmda_total_seconds")
    return None if t is None else float(t)


def s3_walltime(outdir: pathlib.Path) -> pathlib.Path | None:
    """Total ESMDA wall-clock per run: Block A (model x resolution) | Block B (method)."""
    # Block A: one bar per run, grouped by model, ordered by resolution.
    a_runs = []
    for r in dataio.discover_block_a():
        t = _walltime(dataio.load_summary(r))
        if t is not None:
            a_runs.append((r, t))
    a_runs.sort(
        key=lambda rt: (
            (
                list(S.MODEL_LABELS).index(rt[0].model)
                if rt[0].model in S.MODEL_LABELS
                else 99
            ),
            rt[0].nx,
            rt[0].nz,
        )
    )

    # Block B: one bar per method (canonical order).
    b_runs = []
    for r in dataio.discover_block_b():
        if r.method is None:
            continue
        t = _walltime(dataio.load_summary(r))
        if t is not None:
            b_runs.append((r, t))
    b_runs.sort(
        key=lambda rt: (
            S.METHOD_ORDER.index(rt[0].method) if rt[0].method in S.METHOD_ORDER else 99
        )
    )

    if not a_runs and not b_runs:
        print("  S3: no timing available; skipping.")
        return None

    ncol = (1 if a_runs else 0) + (1 if b_runs else 0)
    widths = [max(len(a_runs), 1), max(len(b_runs), 1)] if ncol == 2 else [1]
    fig, axes = plt.subplots(
        1,
        ncol,
        figsize=(0.7 * sum(widths) + 3.0, 4.8),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"width_ratios": widths},
    )
    axes = axes[0]
    ai = 0

    def _annotate_hours(ax, xs, ts):
        ymax = max(ts)
        for x, t in zip(xs, ts):
            ax.text(
                x,
                t,
                f"{t/3600:.1f} h",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if ymax / max(min(ts), 1) > 20 else 0,
            )

    if a_runs:
        ax = axes[ai]
        ai += 1
        xs = list(range(len(a_runs)))
        ts = [t for _, t in a_runs]
        colors = [S.MODEL_COLORS.get(r.model, S.COLORS["charcoal"]) for r, _ in a_runs]
        ax.bar(xs, ts, color=colors, edgecolor="0.2", linewidth=0.7, width=0.8)
        _annotate_hours(ax, xs, ts)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                f"{r.res_label_dx}" + (" (nz32)" if r.nz == 32 else "")
                for r, _ in a_runs
            ],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("ESMDA wall-clock total [s]")
        ax.set_yscale("log")
        ax.set_title("Block A: model × resolution (param-only)", fontsize=10)
        # model colour legend
        models = sorted(
            {r.model for r, _ in a_runs}, key=lambda m: list(S.MODEL_LABELS).index(m)
        )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="none",
                    markerfacecolor=S.MODEL_COLORS[m],
                    markeredgecolor="0.2",
                    markersize=8,
                    label=S.MODEL_LABELS[m],
                )
                for m in models
            ],
            fontsize=8,
            loc="upper left",
        )

    if b_runs:
        ax = axes[ai]
        xs = list(range(len(b_runs)))
        ts = [t for _, t in b_runs]
        colors = [
            S.METHOD_COLORS.get(r.method, S.COLORS["posterior"]) for r, _ in b_runs
        ]
        ax.bar(xs, ts, color=colors, edgecolor="0.2", linewidth=0.7, width=0.7)
        _annotate_hours(ax, xs, ts)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [S.METHOD_LABELS.get(r.method, r.method) for r, _ in b_runs],
            rotation=25,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("ESMDA wall-clock total [s]")
        ax.set_title("Block B: uDALES method (IC+state)", fontsize=10)

    return S.save_pdf(fig, outdir / "S3_walltime.pdf")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "figures",
    )
    ap.add_argument(
        "--cost",
        choices=("walltime", "forward_runs", "state_dim"),
        default="walltime",
        help="cost axis for S1; auto-falls-back if unavailable",
    )
    args = ap.parse_args()

    S.apply_style()
    outdir = args.out / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    rec_a, rec_b, cost, used_fallback = _resolve_cost(args.out, args.cost)
    print(
        f"Summary: {len(rec_a)} Block-A + {len(rec_b)} Block-B records; "
        f"cost='{cost}'; field-RMSE source="
        f"{'run_summary (own-grid)' if used_fallback else 'cache (common grid)'}."
    )

    s1_cost_vs_accuracy(rec_a, rec_b, outdir, cost=cost, used_fallback=used_fallback)
    s2_headline_metric(rec_a, rec_b, outdir, used_fallback=used_fallback)
    s3_walltime(outdir)
    print("Summary figures done.")


if __name__ == "__main__":
    main()
