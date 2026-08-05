"""Figure conventions: colors, quantile bands, shared norms, solid-cell masking.

Shared by every evaluation plot so prior/posterior panels can never disagree on
a color scale. All conventions follow ``docs/figure_specs.md`` §2: import
:data:`COLORS`, :func:`apply_style`, :func:`shade_windows`, :func:`mark_windows`
and the ``save_pdf`` / ``save_png`` helpers from here so every figure looks
identical. See ``docs/plans/esmda_turbulence_evaluation.md`` §7 for the
conventions themselves.

Geometry helpers take the STL path and grid as mandatory arguments -- where the
repo keeps its data is the caller's business.

Populated in WP0.2 (move).
"""

# mypy: ignore-errors
# Moved wholesale in WP0.2 from ``scripts/figspec/style.py`` and the pure
# geometry of ``scripts/figspec/mask.py`` -- largely unannotated code predating
# the strict mypy config. Waived rather than annotated as part of a pure
# refactor; dropping the waiver is later cleanup. The WP1.5 additions
# (:func:`finite_limits`, :func:`nested_bands`) are fully annotated and pass the
# strict config on their own -- checked by deleting this line, which leaves 11
# errors, all in the moved code -- but the waiver covers them too, so nothing
# enforces it.

from __future__ import annotations

import pathlib
import struct
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# §2.1 Colours (UrbanAIR palette)
# ---------------------------------------------------------------------------
COLORS = {
    "truth": "#000000",
    "posterior": "#009BC2",  # teal
    "prior": "#9AA6AC",  # grey
    "orange": "#D0661C",
    "amber": "#DC911B",
    "lightblue": "#92C7DF",
    "charcoal": "#343434",
    "window": "#9AA6AC",  # window boundary lines (grey, dashed)
}

CMAP_FIELD = "viridis"  # sequential field heatmaps
CMAP_DIFF = "RdBu_r"  # diverging difference maps (centre 0)
CMAP_STD = "magma"  # ensemble std

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
            "pdf.fonttype": 42,  # editable text in vector PDFs
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
        ax.axvline(
            float(e),
            color=COLORS["window"],
            linestyle="--",
            linewidth=0.7,
            alpha=0.8,
            zorder=1,
        )
    if annotate and len(edges) >= 2:
        ymin, ymax = ax.get_ylim()
        y = ymax - 0.04 * (ymax - ymin)
        for k in range(len(edges) - 1):
            xc = 0.5 * (edges[k] + edges[k + 1])
            ax.text(
                xc,
                y,
                f"W{k}",
                ha="center",
                va="top",
                fontsize=7,
                color=COLORS["window"],
                zorder=2,
            )


def band(ax, x, mean, std, color, *, alpha=0.25, label=None, lw=2.0, ls="-", nsig=1.0):
    """Mean line + ±nσ shaded band in a single colour."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    (line,) = ax.plot(x, mean, color=color, lw=lw, ls=ls, label=label, zorder=4)
    if std is not None:
        std = np.asarray(std, dtype=float)
        ax.fill_between(
            x,
            mean - nsig * std,
            mean + nsig * std,
            color=color,
            alpha=alpha,
            lw=0,
            zorder=3,
        )
    return line


# ---------------------------------------------------------------------------
# Shared limits and nested quantile bands (WP1.5)
#
# Two conventions the evaluation figure set repeats: every prior/posterior pair
# shares one scale, and an ensemble is drawn as nested quantile bands rather
# than as a mean +- sigma (which is what :func:`band` above does, and which
# reads as symmetric when the posterior is not). Both live here because more
# than one figure needs them -- anything only one figure needs stays private in
# ``figures.py``.
# ---------------------------------------------------------------------------


def finite_limits(
    *arrays: np.ndarray | None, pad: float = 0.0
) -> tuple[float, float] | None:
    """``(min, max)`` over the finite entries of every array, or ``None``.

    The one place non-finite values are excluded before an axis limit or a
    colour norm is set: a single ``inf`` (a pinned parameter's ratio, a member
    that diverged) rescales an axis into uselessness, and NaN makes ``min``
    return NaN outright. ``None`` means nothing finite was passed, which every
    caller reads as "there is no figure to draw here".

    ``pad`` widens the interval by that fraction of its span on both sides; a
    *degenerate* interval (a collapsed ensemble, a pinned parameter -- every
    value identical) is instead widened by ``pad`` of its own magnitude, so the
    result is always usable as a limit.
    """
    finite = [
        np.asarray(a, dtype=float).ravel() for a in arrays if a is not None
    ]  # keep the empty ones: concatenate handles them, an empty list does not
    if not finite:
        return None
    values = np.concatenate(finite)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    low, high = float(values.min()), float(values.max())
    if pad:
        span = high - low
        margin = pad * (span if span > 0 else max(abs(high), 1.0))
        low, high = low - margin, high + margin
    return low, high


def nested_bands(
    ax: Axes,
    coord: np.ndarray,
    values: np.ndarray,
    quantiles: Sequence[float],
    color: str,
    *,
    orient: str = "vertical",
    alphas: tuple[float, float] = (0.18, 0.38),
    label: str | None = None,
    lw: float = 1.8,
    zorder: int = 3,
) -> Line2D | None:
    """Nested quantile bands (5--95 % outside, 25--75 % inside) plus the median.

    Args:
        ax: Target axes.
        coord: ``(N,)`` values of the coordinate the bands run along.
        values: ``(Q, N)`` quantiles of the ensemble, one row per level in
            ``quantiles`` (any order; sorted here).
        quantiles: The ``Q`` quantile levels.
        color: Band and median colour (a ``COLORS`` entry).
        orient: ``"vertical"`` puts ``coord`` on x and the values on y (a time
            series); ``"horizontal"`` puts ``coord`` on y (a vertical profile).
        alphas: Fill alpha of the outermost and innermost band; intermediate
            pairs interpolate.

    Returns the median line, or ``None`` when no level is a median.
    """
    coord = np.asarray(coord, dtype=float)
    values = np.asarray(values, dtype=float)
    levels = np.asarray(quantiles, dtype=float)
    if values.ndim != 2 or values.shape[0] != levels.size:
        raise ValueError(
            f"values must have shape ({levels.size}, n_coord) to match quantiles; "
            f"got {values.shape}"
        )

    order = np.argsort(levels)
    levels, values = levels[order], values[order]
    n_pairs = levels.size // 2
    for i in range(n_pairs):  # outermost pair first, so the inner ones sit on top
        alpha = (
            alphas[1]
            if n_pairs == 1
            else alphas[0] + (alphas[1] - alphas[0]) * i / (n_pairs - 1)
        )
        lo, hi = values[i], values[-1 - i]
        fill = ax.fill_between if orient == "vertical" else ax.fill_betweenx
        fill(coord, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder)

    median = np.isclose(levels, 0.5)
    if not median.any():
        return None
    med = values[int(np.argmax(median))]
    x, y = (coord, med) if orient == "vertical" else (med, coord)
    (line,) = ax.plot(
        x,
        y,
        color=color,
        lw=lw,
        label=label,
        zorder=zorder + 1,
        # The median is the same colour as its own band; the halo is what keeps
        # it visible inside it.
        path_effects=[
            patheffects.Stroke(linewidth=lw + 1.4, foreground="white", alpha=0.8),
            patheffects.Normal(),
        ],
    )
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


def write_table(
    path_stem,
    header: list[str],
    rows: list[list],
    *,
    caption: str = "",
    label: str = "",
    bold_min_cols: Iterable[int] = (),
    bold_max_cols: Iterable[int] = (),
) -> None:
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


# ---------------------------------------------------------------------------
# Building (solid-cell) masks from a case STL (spec §3 / §10.5)
#
# The Xie & Castro (2008) geometry is a set of axis-aligned cubes. We read the
# binary STL, and a grid cell is "solid" when a vertical ray from its centre
# crosses an odd number of mesh triangles (point-in-mesh). This works on any
# (z, y, x) grid, so the same mask serves every model after interpolation onto
# the truth grid. If the STL is unavailable the mask is ``None`` (metrics then
# run over all cells).
# ---------------------------------------------------------------------------
def read_binary_stl(path: pathlib.Path) -> np.ndarray:
    """Return triangles as an ``(n_tri, 3, 3)`` float array (vertices x xyz)."""
    data = pathlib.Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    tris = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        # 12 floats: normal(3) + v0(3) + v1(3) + v2(3); skip 2-byte attr
        vals = struct.unpack_from("<12f", data, off)
        tris[i, 0] = vals[3:6]
        tris[i, 1] = vals[6:9]
        tris[i, 2] = vals[9:12]
        off += 50
    return tris


def _ray_z_crossings(tris: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each (x, y) column, the sorted z-heights where +z rays hit the mesh.

    Returns an object array (len = n_points) of 1-D z-crossing arrays. Uses the
    2-D point-in-triangle test in the xy-plane and barycentric interpolation of
    the triangle's z at the hit.
    """
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    # edge vectors in xy
    e1 = (v1 - v0)[:, :2]
    e2 = (v2 - v0)[:, :2]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    good = np.abs(det) > 1e-12
    inv_det = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)

    pts = np.column_stack([x, y])
    out = np.empty(len(pts), dtype=object)
    for i, (px, py) in enumerate(pts):
        rx = px - v0[:, 0]
        ry = py - v0[:, 1]
        u = (rx * e2[:, 1] - ry * e2[:, 0]) * inv_det
        v = (e1[:, 0] * ry - e1[:, 1] * rx) * inv_det
        inside = good & (u >= 0) & (v >= 0) & (u + v <= 1.0)
        if not inside.any():
            out[i] = np.empty(0)
            continue
        w = 1.0 - u - v
        z_hit = (
            w[inside] * v0[inside, 2]
            + u[inside] * v1[inside, 2]
            + v[inside] * v2[inside, 2]
        )
        out[i] = np.sort(z_hit)
    return out


def stl_solid_mask(
    nz: int, ny: int, nx: int, stl_path: pathlib.Path | str, grid: dict
) -> np.ndarray | None:
    """Boolean solid mask of shape (nz, ny, nx) on ``grid``.

    A cell centre is solid if it lies below an odd-numbered z-crossing of the
    mesh above it (i.e. inside a closed solid). ``grid`` is a mapping with 1-D
    ``"z"``/``"y"``/``"x"`` cell-centre arrays (the caller's evaluation grid);
    ``None`` is returned when the STL is missing or when ``(nz, ny, nx)`` does
    not match ``grid`` -- only that one grid is supported.

    Uncached: the grid arrays are unhashable, and the one caller
    (``figspec.mask.truth_solid_mask``) caches the result itself.
    """
    path = pathlib.Path(stl_path)
    if not path.exists():
        return None
    zc, yc, xc = grid["z"], grid["y"], grid["x"]
    if (len(zc), len(yc), len(xc)) != (nz, ny, nx):
        # caller asked for a non-truth shape; only the truth grid is supported
        return None

    tris = read_binary_stl(path)
    # Build the (y, x) column grid.
    XX, YY = np.meshgrid(xc, yc)  # (ny, nx)
    crossings = _ray_z_crossings(tris, XX.ravel(), YY.ravel())

    mask = np.zeros((nz, ny, nx), dtype=bool)
    zc_arr = np.asarray(zc, dtype=float)
    for col, cz in enumerate(crossings):
        if cz.size < 2:
            continue
        iy, ix = divmod(col, nx)
        # number of crossings strictly above each z level; odd => inside solid
        for k, zlev in enumerate(zc_arr):
            n_above = int(np.count_nonzero(cz > zlev))
            if n_above % 2 == 1:
                mask[k, iy, ix] = True
    return mask
