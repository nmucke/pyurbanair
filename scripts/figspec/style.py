"""Plot styling: UrbanAIR palette, rcParams, and small drawing/save helpers.

All conventions follow ``docs/figure_specs.md`` §2. Import :data:`COLORS`,
:func:`apply_style`, :func:`shade_windows`, :func:`mark_windows`, and the
``save_pdf`` / ``save_png`` helpers from here so every figure looks identical.
"""
from __future__ import annotations

import pathlib
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# §2.1 Colours (UrbanAIR palette)
# ---------------------------------------------------------------------------
COLORS = {
    "truth": "#000000",
    "posterior": "#009BC2",   # teal
    "prior": "#9AA6AC",       # grey
    "orange": "#D0661C",
    "amber": "#DC911B",
    "lightblue": "#92C7DF",
    "charcoal": "#343434",
    "window": "#9AA6AC",      # window boundary lines (grey, dashed)
}

CMAP_FIELD = "viridis"   # sequential field heatmaps
CMAP_DIFF = "RdBu_r"     # diverging difference maps (centre 0)
CMAP_STD = "magma"       # ensemble std

# Per-model identity (Block A: model x resolution comparisons).
MODEL_LABELS = {"palm": "PALM", "udales": "uDALES", "lbm": "LBM"}
MODEL_COLORS = {
    "udales": COLORS["posterior"],
    "palm": COLORS["orange"],
    "lbm": COLORS["amber"],
}
MODEL_MARKERS = {"udales": "o", "palm": "s", "lbm": "^"}

# Per-method identity (Block B/C: estimation-strategy comparisons). Verbatim
# legend labels per spec §1.
METHOD_ORDER = [
    "param_only",
    "ic_corr",
    "ic_dist",
    "ic_red",
    "icstate_red",
]
METHOD_LABELS = {
    "param_only": "param-only",
    "ic_corr": "IC+param (corr)",
    "ic_dist": "IC+param (dist)",
    "ic_red": "IC+param (reduction)",
    "icstate_red": "IC+state+param (reduction)",
}
METHOD_COLORS = {
    "param_only": COLORS["charcoal"],
    "ic_corr": COLORS["orange"],
    "ic_dist": COLORS["amber"],
    "ic_red": COLORS["lightblue"],
    "icstate_red": COLORS["posterior"],
}
METHOD_MARKERS = {
    "param_only": "o",
    "ic_corr": "s",
    "ic_dist": "D",
    "ic_red": "^",
    "icstate_red": "v",
}

PARAM_LABELS = {
    "inflow_angle": r"Inflow angle $\alpha$ [deg]",
    "velocity_magnitude": r"Velocity magnitude $|U|$ [m/s]",
    "pressure_gradient_magnitude": r"Pressure gradient [Pa/m]",
}
PARAM_UNITS = {
    "inflow_angle": "deg",
    "velocity_magnitude": "m/s",
    "pressure_gradient_magnitude": "Pa/m",
}


def apply_style() -> None:
    """Set global rcParams for talk-quality vector/PNG output (spec §2.2)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "axes.grid": False,
            "lines.linewidth": 1.6,
            "figure.dpi": 120,
            "pdf.fonttype": 42,   # editable text in vector PDFs
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Window boundaries on time-axis plots (spec §2.1 / §2.2)
# ---------------------------------------------------------------------------

def shade_windows(ax, edges: np.ndarray | None) -> None:
    """Light alternating shading of assimilation windows (subtle)."""
    if edges is None:
        return
    for k in range(0, len(edges) - 1, 2):
        ax.axvspan(edges[k], edges[k + 1], color="0.93", zorder=0)


def mark_windows(ax, edges: np.ndarray | None, *, annotate: bool = True) -> None:
    """Thin dashed grey vertical lines at window boundaries + window indices."""
    if edges is None:
        return
    for e in edges:
        ax.axvline(float(e), color=COLORS["window"], linestyle="--",
                   linewidth=0.7, alpha=0.8, zorder=1)
    if annotate and len(edges) >= 2:
        ymin, ymax = ax.get_ylim()
        y = ymax - 0.04 * (ymax - ymin)
        for k in range(len(edges) - 1):
            xc = 0.5 * (edges[k] + edges[k + 1])
            ax.text(xc, y, f"W{k}", ha="center", va="top", fontsize=7,
                    color=COLORS["window"], zorder=2)


def band(ax, x, mean, std, color, *, alpha=0.25, label=None, lw=2.0, ls="-",
         nsig=1.0):
    """Mean line + ±nσ shaded band in a single colour."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    line, = ax.plot(x, mean, color=color, lw=lw, ls=ls, label=label, zorder=4)
    if std is not None:
        std = np.asarray(std, dtype=float)
        ax.fill_between(x, mean - nsig * std, mean + nsig * std,
                        color=color, alpha=alpha, lw=0, zorder=3)
    return line


# ---------------------------------------------------------------------------
# Saving (spec §2.3: line/scatter/bar -> PDF; heatmaps -> PNG @ 300 dpi)
# ---------------------------------------------------------------------------

def _ensure_parent(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_pdf(fig, path) -> pathlib.Path:
    path = _ensure_parent(path)
    fig.savefig(path, format="pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {path}")
    return path


def save_png(fig, path, dpi: int = 300) -> pathlib.Path:
    path = _ensure_parent(path)
    fig.savefig(path, format="png", dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {path}")
    return path


# ---------------------------------------------------------------------------
# booktabs LaTeX table emission (spec §2.3 / §3)
# ---------------------------------------------------------------------------

def write_table(path_stem, header: list[str], rows: list[list],
                *, caption: str = "", label: str = "",
                bold_min_cols: Iterable[int] = (),
                bold_max_cols: Iterable[int] = ()) -> None:
    """Write a CSV and a booktabs ``.tex`` snippet for the same table.

    ``rows`` holds cell values (numbers or strings). For numeric columns in
    ``bold_min_cols`` / ``bold_max_cols`` the best cell is wrapped in ``\\textbf``
    in the ``.tex`` only.
    """
    import csv

    path_stem = pathlib.Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    # CSV (raw values).
    with open(path_stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    # Identify best cell per flagged numeric column.
    def col_vals(c):
        out = []
        for r in rows:
            try:
                out.append(float(r[c]))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out)

    best = {}
    for c in bold_min_cols:
        v = col_vals(c)
        if np.isfinite(v).any():
            best[c] = int(np.nanargmin(v))
    for c in bold_max_cols:
        v = col_vals(c)
        if np.isfinite(v).any():
            best[c] = int(np.nanargmax(v))

    def fmt_cell(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    lines = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule"]
    lines.append(" & ".join(_tex_escape(h) for h in header) + r" \\")
    lines.append(r"\midrule")
    for i, r in enumerate(rows):
        cells = []
        for c, v in enumerate(r):
            s = fmt_cell(v)
            if c in best and best[c] == i:
                s = r"\textbf{" + s + "}"
            cells.append(s if c == 0 else s)
        cells[0] = _tex_escape(str(r[0]))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tex = "\n".join(lines)
    if caption or label:
        tex = (
            "% Auto-generated booktabs snippet (requires \\usepackage{booktabs}).\n"
            + tex
        )
    with open(path_stem.with_suffix(".tex"), "w") as f:
        f.write(tex + "\n")
    print(f"  + {path_stem.with_suffix('.csv')} / .tex")


def _tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )
