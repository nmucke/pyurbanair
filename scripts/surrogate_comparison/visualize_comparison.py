"""The paper figures of a surrogate comparison run.

    python scripts/surrogate_comparison/visualize_comparison.py <run dir> [key=value ...]

The overrides are applied to the run's config.yaml (e.g. `figures.sample=1`;
eval_common.comparison_config). The run dir is what run_models.py wrote and
compute_metrics.py scored. The curve figures read metrics.nc only;
station_profiles and the slice figures read the rollouts of trajectory
`figures.sample` and its truth from the corpus over metrics.nc's window (its
`lead` steps), scaled by that sample's reference scales in metrics.nc. Every
quantity is dimensionless (velocities / u_ref, lengths / H, lead time / T, see
docs/plans/implemented/surrogate_comparison_paper_metrics.md).

One style for all (STYLE, below): serif 8 pt text, single-column 3.5 in or
double-column 7.2 in wide, one Okabe-Ito colour per model in models.yaml order
(named by `display_names`), the truth solid black, persistence dashed grey,
one legend above the panels, panel letters (a), (b), ... Curves that leave the
axis (an unstable model) are clipped and marked "(off scale)" in the legend.
Persistence (a frozen frame) is left out of the figures drawn from the
rollouts. The model slices share the truth's colour scale and saturate.

Outputs, in the run dir, each in every format of `figures.formats`:
  rmse                 RMSE / u_ref vs t / T: pooled over u, v, w, then u, v, w
  acc                  anomaly correlation vs t / T: pooled, u, v, w;
                       `metrics.acc_threshold` line
  vorticity            vorticity RMSE x H / u_ref vs t / T: pooled, x, y, z
  physics              TKE / u_ref^2, divergence RMS x H / u_ref and spanwise
                       integral length / H vs t / T, the truth as reference
  pedestrian           RMSE / u_ref and ACC at the pedestrian level vs t / T
  spectra              premultiplied spanwise spectra k_y E / u_ref^2 vs k_y H
                       (k_y = 2 pi / lambda) at the canopy top; rows u', w',
                       columns the snapshot leads
  spectral_ratio       10 log10(E_model / E_truth) of u' over (k_y H, t / T),
                       one panel per model
  pdfs                 densities of u', v', w' / u_ref at the canopy top;
                       rows u', v', w', columns the snapshot leads
  profiles             plane-averaged window mean u, TKE and -u'w' vs z / H
  station_profiles     the same window statistics in single columns behind the
                       most downstream building (lee face + 0.5, 1, 2, 4 x its
                       height); skipped without a building
  slices               |U| / u_ref at the pedestrian level, rows truth + models,
                       columns the snapshot leads (those of spectra and pdfs)
  vertical_slices      x-z plane through the most downstream building at lead
                       `figures.vertical_lead` (negative: from the end, as
                       `metrics.snapshot_leads`): u, w / u_ref and
                       omega_y H / u_ref; skipped without a building
  statistics_slices    window mean u / u_ref and TKE / u_ref^2 at the pedestrian level
  tke_heights          window resolved TKE / u_ref^2 on horizontal slices at the
                       heights `figures.tke_heights` (units of H), rows truth + models
  animation_speed.mp4  |U| at the pedestrian level against lead time: truth and
                       each model on the first row, each model - truth on the
                       second (a review aid, not a paper figure); a .gif when
                       ffmpeg is unavailable. `figures.animation: null` skips it
  animation_tke_z<h>H.mp4  instantaneous resolved TKE / u_ref^2 about each source's
                       window mean at each height h of `figures.tke_heights`
                       (1.0 -> z1p0H), truth and each model side by side; also
                       skipped by `figures.animation: null`
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Sequence

import numpy as np
import xarray
import yaml
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, StrMethodFormatter
from omegaconf import DictConfig, OmegaConf
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import (  # noqa: E402
    comparison_config,
    grid_spacing,
    load_states,
    open_dataset,
    speed,
)
from slice_animation import Panel, animate_slices, frame_indices  # noqa: E402
from surrogate_metrics import interior_cells, tke_field, vorticity  # noqa: E402

# ---------------------------------------------------------------------------
# The paper style, shared by every figure
# ---------------------------------------------------------------------------
SINGLE, DOUBLE = 3.5, 7.2  # column widths [in]
# Okabe & Ito's colour-blind-safe palette (yellow last: faint on white); one
# colour per model in models.yaml order.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
REFERENCE = {
    "persistence": {"color": "0.55", "ls": (0, (4, 2)), "lw": 1.0},
    "truth": {"color": "black", "ls": "-", "lw": 1.3},
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
SOLID = "0.72"  # building cells in the slices
VARS = ("u", "v", "w")
LEAD = r"$t/\overline{T}$"
U_REF = r"u_\mathrm{ref}"
ZH = r"$z/\overline{H}$"


def run(run_dir: pathlib.Path, overrides: list[str] | None = None) -> None:
    run_dir = pathlib.Path(run_dir)
    cfg = comparison_config(run_dir, overrides)
    models = dict(yaml.safe_load((run_dir / "models.yaml").read_text()))
    names = dict(cfg.get("display_names") or {})
    styles = {
        m: {
            "color": PALETTE[i % len(PALETTE)],
            "ls": "-",
            "lw": 1.0,
            "label": names.get(m, m),
        }
        for i, m in enumerate(models)
    } | {
        s: style | {"label": names.get(s, s.capitalize())}
        for s, style in REFERENCE.items()
    }
    metrics = xarray.load_dataset(run_dir / "metrics.nc")
    threshold = float(cfg.metrics.acc_threshold)
    formats = list(cfg.figures.formats)

    with plt.rc_context(STYLE):
        figures = {
            "rmse": rmse_figure(metrics, styles),
            "acc": acc_figure(metrics, styles, threshold),
            "vorticity": vorticity_figure(metrics, styles),
            "physics": physics_figure(metrics, styles),
            "pedestrian": pedestrian_figure(metrics, styles, threshold),
            "spectra": spectra_figure(metrics, styles),
            "spectral_ratio": spectral_ratio_figure(metrics, styles),
            "pdfs": pdfs_figure(metrics, styles),
            "profiles": profiles_figure(metrics, styles),
        }
        for name, fig in figures.items():
            _save(fig, run_dir / name, formats)
        sample = _load_sample(cfg, run_dir, models, metrics)
        if sample is not None:
            leads = [int(t) for t in metrics["pdf_lead"].values]
            figures = {
                "station_profiles": station_profiles_figure(sample, styles),
                "slices": slices_figure(sample, styles, metrics, leads, cfg),
                "vertical_slices": vertical_slices_figure(
                    sample, styles, int(cfg.figures.vertical_lead)
                ),
                "statistics_slices": statistics_slices_figure(sample, styles, cfg),
                "tke_heights": tke_heights_figure(sample, styles, cfg),
            }
            for name, fig in figures.items():
                _save(fig, run_dir / name, formats)
            if cfg.figures.get("animation") is not None:
                _animate_speed(cfg, run_dir, sample, styles)
                _animate_tke(cfg, run_dir, sample, styles)
    print(f"Saved comparison figures in {run_dir}")


# ---------------------------------------------------------------------------
# 1-5: metrics against lead time
# ---------------------------------------------------------------------------


def rmse_figure(metrics: xarray.Dataset, styles: dict) -> Figure | None:
    """RMSE / u_ref over the fluid cells against t / T (metrics.nc `rmse`,
    `rmse_{u,v,w}`): (a) pooled over the three components, (b-d) u, v, w.
    Each sample's RMSE is divided by its u_ref before the sample mean."""
    label = rf"$\mathrm{{RMSE}}/{U_REF}$"
    return _lead_figure(
        metrics,
        styles,
        [("rmse", "pooled", label), *((f"rmse_{v}", f"${v}$", label) for v in VARS)],
    )


def acc_figure(
    metrics: xarray.Dataset, styles: dict, threshold: float
) -> Figure | None:
    """Anomaly correlation about the truth's time mean over the fluid cells
    against t / T (metrics.nc `acc`, `acc_{u,v,w}`): (a) pooled, (b-d) u, v, w.
    The dotted line is `metrics.acc_threshold` (the ACC horizon)."""
    return _lead_figure(
        metrics,
        styles,
        [("acc", "pooled", "ACC"), *((f"acc_{v}", f"${v}$", "ACC") for v in VARS)],
        limits=(-0.3, 1.02),
        reference=threshold,
    )


def vorticity_figure(metrics: xarray.Dataset, styles: dict) -> Figure | None:
    """RMSE of the vorticity x H / u_ref against t / T, on the fluid cells with
    six fluid neighbours (metrics.nc `vorticity_rmse`, `vorticity_rmse_{x,y,z}`):
    (a) pooled over the components, (b-d) omega_x, omega_y, omega_z."""
    label = rf"$\mathrm{{RMSE}}(\omega)\,H/{U_REF}$"
    return _lead_figure(
        metrics,
        styles,
        [
            ("vorticity_rmse", "pooled", label),
            *((f"vorticity_rmse_{c}", rf"$\omega_{c}$", label) for c in "xyz"),
        ],
    )


def physics_figure(metrics: xarray.Dataset, styles: dict) -> Figure | None:
    """Physical consistency against t / T, the truth as the reference curve:
    (a) fluid-mean resolved TKE / u_ref^2, each source about its own time mean
    (`tke`); (b) RMS of the velocity divergence x H / u_ref on the interior
    fluid cells (`divergence_rms`); (c) spanwise integral length Lambda_y / H of
    u' at the canopy top (`integral_length`)."""
    return _lead_figure(
        metrics,
        styles,
        [
            ("tke", "resolved TKE", rf"$k/{U_REF}^2$"),
            (
                "divergence_rms",
                "divergence",
                rf"$\mathrm{{RMS}}(\nabla\!\cdot\!\mathbf{{u}})\,H/{U_REF}$",
            ),
            ("integral_length", "integral length", r"$\Lambda_y/H$"),
        ],
        width=DOUBLE,
    )


def pedestrian_figure(
    metrics: xarray.Dataset, styles: dict, threshold: float
) -> Figure | None:
    """Scores at the pedestrian level `metrics.pedestrian_level` against t / T,
    pooled over u, v, w: (a) RMSE / u_ref (`rmse_pedestrian`), (b) ACC about
    the truth's time mean (`acc_pedestrian`), with the threshold line."""
    return _lead_figure(
        metrics,
        styles,
        [
            ("rmse_pedestrian", "RMSE", rf"$\mathrm{{RMSE}}/{U_REF}$"),
            ("acc_pedestrian", "ACC", "ACC", (-0.3, 1.02), threshold),
        ],
        width=SINGLE,
        height=1.75,
    )


def _lead_figure(
    metrics: xarray.Dataset,
    styles: dict,
    panels: Sequence[tuple],
    limits: tuple[float, float] | None = None,
    reference: float | None = None,
    width: float = DOUBLE,
    height: float = 1.85,
) -> Figure | None:
    """One panel per (metrics.nc name, title, y label[, limits, reference line])
    against t / T, one line per source. Without limits, the y axis starts at 0
    and its top clips a diverging curve (_clip_top)."""
    panels = [p for p in panels if _curves(metrics, styles, p[0])]
    if not panels:
        return None
    x = metrics["lead_over_T"].values
    fig, axes = plt.subplots(
        1, len(panels), figsize=(width, height), layout="constrained", squeeze=False
    )
    off: set[str] = set()
    drawn: set[str] = set()
    for i, (ax, panel) in enumerate(zip(axes[0], panels)):
        name, title, ylabel = panel[:3]
        low, high = panel[3] if len(panel) > 3 else limits or (0.0, None)
        line = panel[4] if len(panel) > 4 else reference
        curves = _curves(metrics, styles, name)
        drawn |= set(curves)
        if high is None:
            high = _clip_top(curves)
        for source, curve in curves.items():
            if (high is not None and np.nanmax(curve) > high) or np.nanmin(curve) < low:
                off.add(source)
            ax.plot(x, curve, **_line(styles, source))
        if line is not None:
            ax.axhline(line, color="0.3", lw=0.7, ls=":", zorder=1)
        ax.set_ylim(low, high)
        ax.set_xlim(0, float(np.nanmax(x)))
        ax.set_xlabel(LEAD)
        if i == 0 or ylabel != panels[i - 1][2]:
            ax.set_ylabel(ylabel)
        _title(ax, i, title)
    _legend(fig, styles, list(drawn), {s: "off scale" for s in off})
    return fig


def _curves(metrics: xarray.Dataset, styles: dict, name: str) -> dict[str, np.ndarray]:
    """`name` per source with a finite value, in legend order (truth last)."""
    if name not in metrics:
        return {}
    have = {str(s) for s in metrics.source.values}
    return {
        s: metrics[name].sel(source=s).values
        for s in styles
        if s in have and np.isfinite(metrics[name].sel(source=s).values).any()
    }


def _clip_top(curves: dict[str, np.ndarray]) -> float | None:
    """An upper y-limit that keeps the typical curves readable: 3 x the median
    of the sources' maxima, or None when every curve fits under it."""
    peaks = [float(np.nanmax(c)) for c in curves.values()]
    top = 3 * float(np.median(peaks))
    return top if max(peaks) > top else None


# ---------------------------------------------------------------------------
# 6-9: spectra, PDFs and profiles at the canopy top / vs height
# ---------------------------------------------------------------------------


def spectra_figure(metrics: xarray.Dataset, styles: dict) -> Figure | None:
    """Premultiplied spanwise spectra k_y H x E / (u_ref^2 H) = k_y E / u_ref^2
    against k_y H, k_y the angular wavenumber 2 pi / lambda (E normalised so
    sum E dk = variance), at the canopy-top level, of the fluctuation about
    the truth's time mean (metrics.nc `spectrum_u`, `spectrum_w`, sample mean).
    Rows u', w'; columns the snapshot leads (the `pdf_lead` coordinate). The
    dotted guide is the inertial range, k^(-2/3) premultiplied, through the
    truth at the middle wavenumber."""
    rows = [v for v in ("u", "w") if _curves(metrics, styles, f"spectrum_{v}")]
    if not rows:
        return None
    leads = metrics["pdf_lead"].values
    kh = metrics["kh"].values
    fig, axes = plt.subplots(
        len(rows),
        len(leads),
        figsize=(DOUBLE, 1.55 * len(rows) + 0.35),
        layout="constrained",
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for r, var in enumerate(rows):
        spectra = _curves(metrics, styles, f"spectrum_{var}")
        for c, lead in enumerate(leads):
            ax = axes[r, c]
            for source, spectrum in spectra.items():
                e = kh * spectrum[metrics.indexes["lead"].get_loc(lead)]
                if (e > 0).any():
                    ax.loglog(kh, np.where(e > 0, e, np.nan), **_line(styles, source))
            mid = len(kh) // 2
            truth = (
                kh * spectra["truth"][metrics.indexes["lead"].get_loc(lead)]
                if "truth" in spectra
                else np.full(len(kh), np.nan)
            )
            if truth[mid] > 0:
                guide = 1.6 * truth[mid] * (kh / kh[mid]) ** (-2 / 3)
                ax.loglog(kh[mid - 3 :], guide[mid - 3 :], color="0.45", lw=0.7, ls=":")
                if r == 0 and c == 0:
                    ax.annotate(
                        r"$k^{-2/3}$",
                        (kh[-1], guide[-1]),
                        xytext=(-2, 3),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        color="0.35",
                    )
            _log_kh(ax)
            ax.grid(which="minor", color="0.94", lw=0.3)
            if r == len(rows) - 1:
                ax.set_xlabel(r"$k_y H$")
            _title(ax, r * len(leads) + c, _lead_title(metrics, lead) if r == 0 else "")
        axes[r, 0].set_ylabel(rf"$k_y E_{{{var}{var}}}/{U_REF}^2$")
    _legend(fig, styles, [s for s in styles if s in spectra])
    return fig


def spectral_ratio_figure(metrics: xarray.Dataset, styles: dict) -> Figure | None:
    """10 log10(E_model / E_truth) of the spanwise u' spectrum at the canopy top,
    the ratio of the sample-mean spectra (metrics.nc `spectrum_u`, k_y the
    angular wavenumber 2 pi / lambda) over k_y H (x, log) and t / T (y), one panel per
    model (persistence left out). Blue: the model lacks energy at that scale,
    red: it has too much; the limits are +-6 dB and the contours mark ratios
    0.8 and 1.2."""
    spectra = _curves(metrics, styles, "spectrum_u")
    models = [s for s in spectra if s not in REFERENCE]
    if "truth" not in spectra or not models:
        return None
    kh = metrics["kh"].values
    t = metrics["lead_over_T"].values
    ncol = len(models) if len(models) <= 3 else -(-len(models) // 2)
    nrow = -(-len(models) // ncol)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(DOUBLE, 1.75 * nrow + 0.2),
        layout="constrained",
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    norm = mcolors.Normalize(-6, 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, (ax, model) in enumerate(zip(axes.flat, models)):
            db = 10 * np.log10(spectra[model] / spectra["truth"])
            image = ax.pcolormesh(
                kh, t, db, cmap="RdBu_r", norm=norm, shading="nearest", rasterized=True
            )
            ax.contour(
                kh,
                t,
                db,
                levels=10 * np.log10([0.8, 1.2]),
                colors="k",
                linewidths=0.5,
                linestyles=["--", "-"],
            )
            _log_kh(ax)
            ax.grid(False)
            _title(ax, i, styles[model]["label"])
    for i, ax in enumerate(axes.flat):
        if i >= len(models):
            ax.set_visible(False)
        elif i + ncol >= len(models):  # nothing drawn below it
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel(r"$k_y H$")
    for ax in axes[:, 0]:
        ax.set_ylabel(LEAD)
    bar = fig.colorbar(image, ax=axes, extend="both", shrink=0.9, aspect=30, pad=0.01)
    bar.set_label(r"$10\log_{10}(E_{uu}/E_{uu}^\mathrm{truth})$ [dB]")
    bar.set_ticks([-6, -3, 0, 3, 6])
    return fig


def pdfs_figure(metrics: xarray.Dataset, styles: dict) -> Figure | None:
    """Probability densities of u', v', w' / u_ref at the canopy top, about the
    truth's time mean, pooled over the samples (metrics.nc `pdf` on the shared
    bins `pdf_bin_centre`, spanning the truth's range: a diverging model's
    tails leave the axis), log density from 1e-4 of the truth's peak (over the
    leads) to 3x it; densities <= 0 are not drawn.
    Rows u', v', w'; columns the snapshot leads."""
    sources = _curves(metrics, styles, "pdf")
    if not sources:
        return None
    variables = [str(v) for v in metrics["var"].values]
    leads = metrics["pdf_lead"].values
    fig, axes = plt.subplots(
        len(variables),
        len(leads),
        figsize=(DOUBLE, 1.3 * len(variables) + 0.35),
        layout="constrained",
        sharex="row",
        sharey="row",
        squeeze=False,
    )
    for r, var in enumerate(variables):
        centres = metrics["pdf_bin_centre"].sel(var=var).values
        for c, lead in enumerate(leads):
            ax = axes[r, c]
            for source in sources:
                density = (
                    metrics["pdf"].sel(source=source, var=var, pdf_lead=lead).values
                )
                ax.semilogy(
                    centres,
                    np.where(density > 0, density, np.nan),
                    **_line(styles, source),
                )
            truth = metrics["pdf"].sel(source="truth", var=var).values
            peak = float(np.nanmax(truth)) if "truth" in sources else 0.0
            if peak > 0:
                ax.set_ylim(1e-4 * peak, 3 * peak)
            half = 0.5 * float(centres[1] - centres[0])
            ax.set_xlim(float(centres[0]) - half, float(centres[-1]) + half)
            ax.set_xlabel(rf"${var}'/{U_REF}$")
            _title(ax, r * len(leads) + c, _lead_title(metrics, lead) if r == 0 else "")
        axes[r, 0].set_ylabel("PDF")
    _legend(fig, styles, list(sources))
    return fig


def profiles_figure(metrics: xarray.Dataset, styles: dict) -> Figure | None:
    """Window statistics averaged over the fluid cells of each horizontal plane
    against z / H (metrics.nc `profile_mean_u`, `profile_tke`, `profile_uw`):
    (a) mean u / u_ref, (b) resolved TKE / u_ref^2, (c) -u'w' / u_ref^2, each
    source about its own time mean. The dotted line is the mean building
    height."""
    keys = [
        k
        for k in ("profile_mean_u", "profile_tke", "profile_uw")
        if _curves(metrics, styles, k)
    ]
    if not keys:
        return None
    z = metrics["z_over_H"].values
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=(DOUBLE, 2.3),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    off: set[str] = set()
    for i, (ax, key) in enumerate(zip(axes[0], keys)):
        off |= _profile_panel(ax, i, key, z, _curves(metrics, styles, key), styles)
    axes[0, 0].set_ylabel(ZH)
    sources = _curves(metrics, styles, keys[0])
    _legend(fig, styles, list(sources), {s: "off scale" for s in off})
    return fig


PROFILE_LABELS = {
    "profile_mean_u": (r"mean $u$", rf"$\overline{{u}}/{U_REF}$"),
    "profile_tke": ("resolved TKE", rf"$k/{U_REF}^2$"),
    "profile_uw": (r"shear stress", rf"$-\overline{{u'w'}}/{U_REF}^2$"),
}


def _profile_panel(
    ax: Any,
    i: int,
    key: str,
    z: np.ndarray,
    curves: dict,
    styles: dict,
    roof: float = 1.0,
) -> set[str]:
    """One profile panel: each source's `key` against z, a dotted roof line.
    The x axis spans the truth's range (TKE, stress: and 0) plus 40 % of it on each side;
    returns the sources that leave it."""
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.axhline(roof, color="0.45", lw=0.6, ls=":", zorder=1)
    for source, values in curves.items():
        ax.plot(values, z, **_line(styles, source))
    ax.set_ylim(0, float(np.nanmax(z)))
    off: set[str] = set()
    if "truth" in curves:
        zero = [] if key == "profile_mean_u" else [0.0]
        low = min([float(np.nanmin(curves["truth"])), *zero])
        high = max([float(np.nanmax(curves["truth"])), *zero])
        pad = 0.4 * (high - low) or 1.0
        ax.set_xlim(low - pad, high + pad)
        off = {
            s
            for s, v in curves.items()
            if np.nanmin(v) < low - pad or np.nanmax(v) > high + pad
        }
    title, label = PROFILE_LABELS[key]
    ax.set_xlabel(label)
    _title(ax, i, title)
    return off


# ---------------------------------------------------------------------------
# 10-13: figures of trajectory `figures.sample`, from the rollouts
# ---------------------------------------------------------------------------


def station_profiles_figure(sample: dict, styles: dict) -> Figure | None:
    """Window statistics in single columns behind the most downstream building
    of trajectory `figures.sample` (rollouts + truth, no persistence): rows
    mean u / u_ref, resolved TKE / u_ref^2 and -u'w' / u_ref^2, each source
    about its own time mean over the compared window; columns the stations at
    x = lee face + 0.5, 1, 2, 4 H_b, H_b the building's own height, at its
    spanwise centre (those inside the domain). The lee face is the footprint's
    largest x: the flow is assumed along +x. Height is z / H, H this sample's
    mean building height; the dotted line is H_b / H. Each panel's x axis
    follows the truth (_profile_panel). Skipped without a building."""
    if sample["building"] is None:
        print("no building in figures.sample: station_profiles skipped")
        return None
    k_top, y, lee = sample["building"]
    H = sample["refs"]["H"]
    H_b = sample["dz"] * (k_top + 1)
    x = sample["x"]
    stations = [
        (d, int(np.argmin(np.abs(x - (lee + d * H_b)))))
        for d in (0.5, 1, 2, 4)
        if lee + d * H_b <= x[-1]
    ]
    if not stations:
        return None
    sources = ["truth", *sample["preds"]]
    stats = {s: _column_statistics(sample, s, y) for s in sources}
    z = sample["z"] / H
    roof = H_b / H
    fig, axes = plt.subplots(
        3,
        len(stations),
        figsize=(DOUBLE, 4.6),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    off: set[str] = set()
    for r, key in enumerate(PROFILE_LABELS):
        for c, (d, i) in enumerate(stations):
            curves = {
                s: v[r][:, i] for s, v in stats.items() if np.isfinite(v[r][:, i]).any()
            }
            ax = axes[r, c]
            off |= _profile_panel(
                ax, r * len(stations) + c, key, z, curves, styles, roof
            )
            title = rf"$x_\mathrm{{lee}} + {d:g}\,H_b$" if r == 0 else ""
            _title(ax, r * len(stations) + c, title)
        axes[r, 0].set_ylabel(r"$z/H$")
    notes = {s: "off scale" for s in off}
    notes |= {s: "unstable" for s, v in stats.items() if not np.isfinite(v[0]).any()}
    _legend(fig, styles, sources, notes)
    return fig


def _column_statistics(sample: dict, source: str, y: int) -> list[np.ndarray]:
    """Mean u / u_ref, TKE / u_ref^2 and -u'w' / u_ref^2 of `source` in the
    x-z plane at `y`, (nz, nx) each, about its own window mean; NaN in solid
    cells and for a rollout with a non-finite frame."""
    fields = (sample["truth"] if source == "truth" else sample["preds"][source])[
        :, :, :, y
    ]
    u_ref = sample["refs"]["u_ref"]
    with np.errstate(all="ignore"):
        prime = fields - fields.mean(axis=0)
        out = [
            fields[:, 0].mean(axis=0) / u_ref,
            0.5 * (prime**2).mean(axis=0).sum(axis=0) / u_ref**2,
            -(prime[:, 0] * prime[:, 2]).mean(axis=0) / u_ref**2,
        ]
    return [np.where(sample["fluid"][:, y], o, np.nan) for o in out]


def slices_figure(
    sample: dict,
    styles: dict,
    metrics: xarray.Dataset,
    leads: list[int],
    cfg: DictConfig,
) -> Figure | None:
    """|U| / u_ref at the pedestrian level `metrics.pedestrian_level` of
    trajectory `figures.sample`: rows the truth and each model (no
    persistence), columns the snapshot leads (metrics.nc `pdf_lead`, the
    columns of the spectra and PDFs). One colour scale, 0 to the truth's
    maximum over the panels: a model above it saturates. Buildings grey; flow
    from left to right. Skipped when the level has no fluid cell."""
    level = int(cfg.metrics.pedestrian_level)
    if not leads or not _has_fluid(sample, level, "slices"):
        return None
    u_ref = sample["refs"]["u_ref"]
    rows = {
        s: [speed(f[t - 1])[level] / u_ref for t in leads]
        for s, f in [("truth", sample["truth"]), *sample["preds"].items()]
    }
    vmax = max(
        float(np.nanmax(np.where(sample["fluid"][level], r, np.nan)))
        for r in rows["truth"]
    )
    columns = [
        (
            _lead_title(metrics, t),
            "viridis",
            mcolors.Normalize(0, vmax),
        )
        for t in leads
    ]
    return _slice_grid(
        rows,
        columns,
        ~sample["fluid"][level],
        styles,
        sample["extent_xy"],
        DOUBLE,
        bar=rf"$|\mathbf{{U}}|/{U_REF}$",
    )


def vertical_slices_figure(sample: dict, styles: dict, lead: int) -> Figure | None:
    """The x-z plane through the spanwise centre of the most downstream building of
    trajectory `figures.sample` at lead step `figures.vertical_lead` (a negative
    lead counts from the end, n + 1 + lead, as `metrics.snapshot_leads`;
    skipped outside 1..n or without a building): columns
    u / u_ref, w / u_ref and the spanwise vorticity omega_y H / u_ref (central
    differences, surrogate_metrics.vorticity; white on the fluid cells next to a
    wall, where they would read solid cells), rows the truth and each model.
    Per column one colour scale from the truth's 1-99 % range (symmetric about 0,
    so reversed flow is blue): a model beyond it saturates."""
    n = len(sample["truth"])
    lead = lead if lead > 0 else n + 1 + lead
    if not 1 <= lead <= n:
        print(f"figures.vertical_lead not in 1..{n}: vertical_slices skipped")
        return None
    if sample["building"] is None:
        print("no building in figures.sample: vertical_slices skipped")
        return None
    _, y, _ = sample["building"]
    refs = sample["refs"]
    solid = ~sample["fluid"][:, y]
    interior = interior_cells(sample["fluid"])[:, y]
    rows = {}
    for source, fields in [("truth", sample["truth"]), *sample["preds"].items()]:
        frame = fields[lead - 1 : lead]
        with np.errstate(all="ignore"):
            omega = (
                vorticity(frame, sample["spacing"])[0, 1, :, y]
                * refs["H"]
                / refs["u_ref"]
            )
        omega = np.where(interior, omega, np.nan)
        rows[source] = [
            frame[0, 0, :, y] / refs["u_ref"],
            frame[0, 2, :, y] / refs["u_ref"],
            omega,
        ]
    columns = []
    for c, (title, symbol) in enumerate(
        [
            ("$u$", rf"$u/{U_REF}$"),
            ("$w$", rf"$w/{U_REF}$"),
            (r"$\omega_y$", rf"$\omega_y H/{U_REF}$"),
        ]
    ):
        low, high = np.nanpercentile(rows["truth"][c][~solid], [1, 99])
        bound = max(abs(low), abs(high))
        columns.append((title, "RdBu_r", mcolors.Normalize(-bound, bound), symbol))
    return _slice_grid(rows, columns, solid, styles, sample["extent_xz"], DOUBLE)


def statistics_slices_figure(
    sample: dict, styles: dict, cfg: DictConfig
) -> Figure | None:
    """Window statistics at the pedestrian level of trajectory `figures.sample`,
    each source about its own time mean over the compared window: columns mean
    u / u_ref and resolved TKE / u_ref^2, rows the truth and each model. Per
    column one scale from the truth (mean u: its 1-99 % range made
    symmetric about 0; TKE: 0 to its 99th percentile): a model beyond it
    saturates. Skipped when the level has no fluid cell."""
    level = int(cfg.metrics.pedestrian_level)
    if not _has_fluid(sample, level, "statistics_slices"):
        return None
    u_ref = sample["refs"]["u_ref"]
    rows = {}
    for source, fields in [("truth", sample["truth"]), *sample["preds"].items()]:
        layer = fields[:, :, level]
        with np.errstate(all="ignore"):
            rows[source] = [
                layer[:, 0].mean(axis=0) / u_ref,
                0.5 * layer.var(axis=0).sum(axis=0) / u_ref**2,
            ]
    solid = ~sample["fluid"][level]
    mean, tke = (rows["truth"][c][~solid] for c in range(2))
    bound = float(np.nanmax(np.abs(np.nanpercentile(mean, [1, 99]))))
    columns = [
        (
            r"mean $u$",
            "RdBu_r",
            mcolors.Normalize(-bound, bound),
            rf"$\overline{{u}}/{U_REF}$",
        ),
        (
            "resolved TKE",
            "magma",
            mcolors.Normalize(0, float(np.nanpercentile(tke, 99))),
            rf"$k/{U_REF}^2$",
        ),
    ]
    return _slice_grid(rows, columns, solid, styles, sample["extent_xy"], SINGLE)


def tke_heights_figure(sample: dict, styles: dict, cfg: DictConfig) -> Figure | None:
    """Window resolved TKE / u_ref^2 of trajectory `figures.sample`, each source
    about its own time mean over the compared window (surrogate_metrics.tke_field),
    on horizontal slices: rows the truth and each model (no persistence: its
    TKE is 0), columns the heights `figures.tke_heights` (_tke_levels), titled
    with the level's actual z / H. Per column one scale, 0 to the truth's 99th
    percentile: a model above it saturates."""
    levels = _tke_levels(sample, cfg.figures.tke_heights)
    if not levels:
        return None
    u_ref = sample["refs"]["u_ref"]
    rows = {}
    for source, fields in [("truth", sample["truth"]), *sample["preds"].items()]:
        with np.errstate(all="ignore"):
            rows[source] = [tke_field(fields[:, :, k]) / u_ref**2 for _, k, _ in levels]
    solid = [~sample["fluid"][k] for _, k, _ in levels]
    columns = [
        (
            rf"$z/H = {z:.2f}$",
            "magma",
            mcolors.Normalize(0, float(np.nanpercentile(truth[~mask], 99))),
            rf"$k/{U_REF}^2$",
            "max",
        )
        for (_, _, z), truth, mask in zip(levels, rows["truth"], solid)
    ]
    return _slice_grid(rows, columns, solid, styles, sample["extent_xy"], DOUBLE)


def _tke_levels(
    sample: dict, heights: Sequence[float]
) -> list[tuple[float, int, float]]:
    """(height, level k, z / H of its centre) per height h of `figures.tke_heights`
    (units of H, this sample's mean building height): the level k containing
    h H (= the nearest cell centre (k + 1/2) dz); heights outside the domain and
    repeated levels are dropped."""
    H, dz = sample["refs"]["H"], sample["dz"]
    nz = sample["fluid"].shape[0]
    levels: dict[int, tuple[float, int, float]] = {}
    for h in heights:
        k = int(np.floor(float(h) * H / dz))
        if 0 <= float(h) and k < nz and k not in levels:
            levels[k] = (float(h), k, (k + 0.5) * dz / H)
    return list(levels.values())


def _slice_grid(
    rows: dict[str, list[np.ndarray]],
    columns: list[tuple],
    solid: np.ndarray | list[np.ndarray],
    styles: dict,
    extent: tuple[float, float, float, float],
    width: float,
    bar: str | None = None,
) -> Figure:
    """A grid of 2-D slices, one row per source (named on the left), one column
    per (title, cmap, norm[, colour-bar label]). Each column gets a colour bar
    below it, or one bar under the whole grid when `bar` labels a shared one.
    `solid` cells (one mask, or one per column) are grey, other NaN cells
    white, a panel without a finite value says so; a column's optional fifth
    entry is its bar's `extend` (default "both"); letters
    (a), (b), ... run row by row, the first panel of a row names the source."""
    n_rows, n_cols = len(rows), len(columns)
    masks = solid if isinstance(solid, list) else [solid] * n_cols
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    panel_h = (width - 0.3) / n_cols * aspect
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, n_rows * (panel_h + 0.06) + 0.5),
        layout="constrained",
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    images = []
    for r, (source, fields) in enumerate(rows.items()):
        for c, (field, column, solid) in enumerate(zip(fields, columns, masks)):
            ax = axes[r, c]
            ax.imshow(
                np.where(solid, 1.0, np.nan),
                origin="lower",
                extent=extent,
                cmap=mcolors.ListedColormap([SOLID]),
                interpolation="nearest",
                rasterized=True,
            )
            images.append(
                ax.imshow(
                    np.where(solid, np.nan, field),
                    origin="lower",
                    extent=extent,
                    cmap=column[1],
                    norm=column[2],
                    interpolation="nearest",
                    rasterized=True,
                )
            )
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            letter = f"({chr(97 + r * n_cols + c)})"
            if not np.isfinite(field[~solid]).any():
                ax.text(
                    0.5,
                    0.5,
                    "non-finite",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            ax.text(
                0.015,
                0.96,
                f"{letter} {styles[source]['label']}" if c == 0 else letter,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 0.6,
                    "alpha": 0.8,
                },
            )
            if r == 0:
                ax.set_title(column[0])
    if bar is not None:
        colorbar = fig.colorbar(
            images[0],
            ax=axes,
            location="bottom",
            shrink=0.4,
            aspect=40,
            pad=0.01,
            extend="max",
        )
        colorbar.set_label(bar)
    else:
        for c, column in enumerate(columns):
            colorbar = fig.colorbar(
                images[c],
                ax=axes[:, c],
                location="bottom",
                shrink=0.8,
                aspect=25,
                pad=0.01,
                extend=column[4] if len(column) > 4 else "both",
            )
            colorbar.set_label(column[3])
    return fig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _log_kh(ax: Any) -> None:
    """A log k_y H axis labelled 0.1, 0.2, 0.5, 1, 2, 5, ... in plain numbers."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _line(styles: dict, source: str) -> dict:
    """Plot keywords of `source`: the truth on top of the models."""
    style: dict = styles[source] | {"zorder": 3 if source == "truth" else 2}
    return style


def _title(ax: Any, index: int, text: str) -> None:
    """The panel letter, then a short panel title, at the top left."""
    ax.set_title(f"({chr(97 + index)}) {text}".rstrip(), loc="left")


def _legend(
    fig: Figure, styles: dict, sources: Sequence[str], notes: dict | None = None
) -> None:
    """One legend above the panels: every source once in style order, a
    source in `notes` marked e.g. "(off scale)"."""
    notes = notes or {}
    sources = [s for s in styles if s in sources]
    handles = [Line2D([], [], **_line(styles, s)) for s in sources]
    labels = [
        styles[s]["label"] + (f" ({notes[s]})" if s in notes else "") for s in sources
    ]
    ncol = len(sources) if fig.get_figwidth() > SINGLE + 0.1 else min(len(sources), 3)
    fig.legend(
        handles, labels, loc="outside upper center", ncol=ncol, columnspacing=1.2
    )


def _lead_title(metrics: xarray.Dataset, lead: int) -> str:
    """'t/T = 0.42' for lead step `lead`, from metrics.nc's `lead_over_T`."""
    value = float(metrics["lead_over_T"].sel(lead=lead))
    return rf"$t/\overline{{T}} = {value:.2f}$"


def _has_fluid(sample: dict, level: int, figure: str) -> bool:
    """Whether level `level` of `figures.sample` has a fluid cell; if not, says
    that `figure` is skipped."""
    if 0 <= level < len(sample["fluid"]) and sample["fluid"][level].any():
        return True
    print(f"no fluid cell at z index {level} of figures.sample: {figure} skipped")
    return False


def _save(fig: Figure | None, stem: pathlib.Path, formats: Sequence[str]) -> None:
    """`stem`.<format> for each format (raster at 300 dpi), then close."""
    if fig is None:
        print(f"no data for {stem.name}: skipped")
        return
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  + {stem.name}")


def _load_sample(
    cfg: DictConfig, run_dir: pathlib.Path, models: dict, metrics: xarray.Dataset
) -> dict | None:
    """Trajectory `figures.sample`: its truth u, v, w and every model's rollout
    over metrics.nc's window (n = its `lead` length, the shortest rollout over
    all models and samples, as compute_metrics.py), (n, 3, nz, ny, nx) each,
    the fluid mask, grid, this sample's reference scales from metrics.nc
    (`reference_H`, `reference_u`, `reference_T`, `canopy_level`) and the most
    downstream building (_downstream_building, None without one)."""
    samples = list(cfg.sample_indices)  # set by run_models.py
    index = int(cfg.figures.sample)
    if not 0 <= index < len(samples):
        print(f"figures.sample={index} not among the {len(samples)} samples: no slices")
        return None
    name_nc = f"sample_{int(samples[index]):04d}.nc"
    n = int(metrics.sizes["lead"])
    preds = {}
    for label in models:
        with xarray.open_dataset(run_dir / "rollouts" / label / name_nc) as ds:
            window = ds.isel(time=slice(0, n))
            preds[label] = np.stack([window[v].values for v in VARS], axis=1)
            start, sample = int(ds.attrs["start"]), int(ds.attrs["sample"])
            z, y, x = (ds[d].values.astype(float) for d in ("z", "y", "x"))
    first = next(iter(models))
    train_cfg = OmegaConf.load(pathlib.Path(models[first]["model_dir"]) / "config.yaml")
    assert isinstance(train_cfg, DictConfig)
    data = open_dataset(train_cfg, cfg.data_dir, cfg.split, sdf_features="none")
    fluid = data.geometry_for(sample).numpy().astype(bool)
    frames = list(range(start, start + n + 1))
    with xarray.open_dataset(data._state_files[sample]) as ds:
        times = ds["time"].values[frames].astype(float)
    velocity = [list(data.state_vars).index(v) for v in VARS]  # as compute_metrics
    truth = load_states(data, sample, frames[1:])[:, velocity]
    spacing = grid_spacing(data, sample)  # as compute_metrics.py
    scales = metrics.isel(sample=index)
    refs = {
        "H": float(scales["reference_H"]),
        "u_ref": float(scales["reference_u"]),
        "T": float(scales["reference_T"]),
        "k_c": int(scales["canopy_level"]),
    }
    H = refs["H"]
    half = [s / 2 for s in spacing]
    return {
        "truth": truth,
        "preds": {label: p[:n] for label, p in preds.items()},
        "fluid": fluid,
        "spacing": spacing,
        "dz": spacing[0],
        "z": z - z[0] + half[0],  # cell centres above the ground
        "x": x,
        "times": times - times[0],
        "refs": refs,
        "building": _downstream_building(fluid, x, spacing[2]),
        "extent_xy": tuple(
            v / H
            for v in (x[0] - half[2], x[-1] + half[2], y[0] - half[1], y[-1] + half[1])
        ),
        "extent_xz": tuple(
            v / H for v in (x[0] - half[2], x[-1] + half[2], 0.0, len(z) * spacing[0])
        ),
    }


def _downstream_building(
    fluid: np.ndarray, x: np.ndarray, dx: float
) -> tuple[int, int, float] | None:
    """(top level, spanwise centre index, lee-face x [m]) of the most
    downstream building, so the stations behind it are in its open wake: the
    connected footprint (columns with a solid cell) reaching the largest x
    (ties: the first in y), its top the highest solid level in it. None when
    no cell is solid."""
    footprint = (~fluid).any(axis=0)
    if not footprint.any():
        return None
    labels, _ = ndimage.label(footprint)
    ys, xs = np.nonzero(footprint)
    first = int(np.argmax(xs))  # np.nonzero is (y, x)-ordered: the first in y
    ys, xs = np.nonzero(labels == labels[ys[first], xs[first]])
    top = int(np.nonzero((~fluid)[:, ys, xs].any(axis=1))[0].max())
    return top, int(round(float(ys.mean()))), float(x[xs.max()] + dx / 2)


def _animate_speed(
    cfg: DictConfig, run_dir: pathlib.Path, sample: dict, styles: dict
) -> pathlib.Path | None:
    """animation_speed.mp4: |U| at the pedestrian level per lead step (thinned
    to `max_frames`): truth and each model on the first row, each model -
    truth on the second. Colour limits are fixed over the whole rollout, the
    models' clipped to 1.5 x the truth's maximum; the error scale is the median
    model's, so a diverging model saturates rather than washing out the others.
    Skipped when the pedestrian level has no fluid cell."""
    opts = cfg.figures.animation
    level = int(cfg.metrics.pedestrian_level)
    if not _has_fluid(sample, level, "animation_speed"):
        return None
    n = len(sample["truth"])
    frames = frame_indices(n, int(opts.max_frames))
    fluid = sample["fluid"][level]

    def layer(states: np.ndarray) -> np.ndarray:
        return np.where(fluid, speed(states[frames])[:, level], np.nan)

    reference = layer(sample["truth"])
    bound = 1.5 * float(np.nanmax(reference))
    fields = {s: np.clip(layer(p), 0, bound) for s, p in sample["preds"].items()}
    vmax = float(np.nanpercentile(reference, 99.5))
    errors = {s: f - reference for s, f in fields.items()}
    spreads = [float(np.nanpercentile(np.abs(e), 99.5)) for e in errors.values()]
    spread = float(np.median(spreads)) if spreads else 1.0
    values = [Panel("truth", reference, "viridis", 0.0, vmax, "value")]
    values += [
        Panel(styles[s]["label"], f, "viridis", 0.0, vmax, "value")
        for s, f in fields.items()
    ]
    differences = [
        Panel(f"{styles[s]['label']}\n- truth", e, "RdBu_r", -spread, spread, "error")
        for s, e in errors.items()
    ]
    path: pathlib.Path = animate_slices(
        [values, differences],
        sample["times"][1:][frames],
        run_dir / "animation_speed.mp4",
        title=f"|U| [m/s] at z index {level}, lead",
        fps=int(opts.fps),
        dpi=int(opts.dpi),
        labels=("x index", "y index"),
    )
    return path


def _animate_tke(
    cfg: DictConfig, run_dir: pathlib.Path, sample: dict, styles: dict
) -> list[pathlib.Path]:
    """animation_tke_z<h>H.mp4 per height of `figures.tke_heights` (_tke_levels):
    the instantaneous resolved TKE 1/2 sum_c (u_c - mean_c)^2 / u_ref^2 per
    lead step (thinned to `max_frames`), the mean each source's own over the
    compared window (its time average is the tke_heights panel); truth and each
    model on one row. The colour limits are the truth's 0 to its 99.5th
    percentile over the whole rollout, so a diverging model saturates."""
    opts = cfg.figures.animation
    frames = frame_indices(len(sample["truth"]), int(opts.max_frames))
    u_ref = sample["refs"]["u_ref"]
    paths = []
    for h, k, z in _tke_levels(sample, cfg.figures.tke_heights):
        fluid = sample["fluid"][k]

        def layer(states: np.ndarray) -> np.ndarray:
            plane = states[:, :, k]
            with np.errstate(all="ignore"):
                prime = plane[frames] - plane.mean(axis=0, dtype=np.float64)
                return np.where(fluid, 0.5 * (prime**2).sum(axis=1) / u_ref**2, np.nan)

        reference = layer(sample["truth"])
        vmax = float(np.nanpercentile(reference, 99.5))
        panels = [Panel("truth", reference, "magma", 0.0, vmax, "value")]
        panels += [
            Panel(styles[s]["label"], layer(p), "magma", 0.0, vmax, "value")
            for s, p in sample["preds"].items()
        ]
        label = f"{h:.2f}".rstrip("0")
        label = (label + "0" if label.endswith(".") else label).replace(".", "p")
        paths.append(
            animate_slices(
                [panels],
                sample["times"][1:][frames],
                run_dir / f"animation_tke_z{label}H.mp4",
                title=f"k / u_ref^2 at z/H = {z:.2f} (z index {k}), lead",
                fps=int(opts.fps),
                dpi=int(opts.dpi),
                labels=("x index", "y index"),
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "overrides", nargs="*", help="config overrides, e.g. figures.sample=1"
    )
    args = parser.parse_args()
    run(args.run_dir, args.overrides)


if __name__ == "__main__":
    main()
