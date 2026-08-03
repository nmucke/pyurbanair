"""Reusable plotting primitives shared by the block drivers.

These build directly on :mod:`figspec.dataio` / :mod:`figspec.style` and produce
the recurring figure shapes from ``docs/figure_specs.md``: parameter-trajectory
panels, error-vs-time / error-vs-resolution line plots, validation-sensor panels,
and the field + difference heatmap grids.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from evaluation import style as S

from . import dataio, mask

# ---------------------------------------------------------------------------
# Field slice helpers (common evaluation grid = truth grid)
# ---------------------------------------------------------------------------
PED_Z = 2.0  # pedestrian-height evaluation level [m] (spec §4 A3: z~2-10 m)


def _extent() -> list[float]:
    g = dataio.truth_grid()
    x, y = g["x"], g["y"]
    return [float(x.min()), float(x.max()), float(y.min()), float(y.max())]


def truth_slice(time_s: float, z_m: float = PED_Z) -> tuple[np.ndarray, list[float]]:
    zi = mask.nearest_z_index(z_m)
    tv = dataio.truth_velmag()
    da = tv.sel(time=time_s, method="nearest").isel(z=zi)
    return np.asarray(da.values, dtype=float), _extent()


def model_slice(
    run, time_s: float, z_m: float = PED_Z
) -> tuple[np.ndarray, list[float]] | None:
    """Posterior-mean |U| of a run on the truth grid at (time, z)."""
    sm = dataio.load_state_mean(run)
    if sm is None:
        return None
    da = dataio.velmag_field(sm).sel(time=time_s, method="nearest")
    da = dataio.interp_to_truth(da)
    zi = mask.nearest_z_index(z_m)
    da = da.isel(z=zi)
    sm.close()
    return np.asarray(da.values, dtype=float), _extent()


# ---------------------------------------------------------------------------
# Parameter trajectories (A1, B1, A1-grid)
# ---------------------------------------------------------------------------
def param_panel(
    ax,
    *,
    time,
    post_mean,
    post_std=None,
    prior_mean=None,
    prior_std=None,
    truth_t=None,
    truth=None,
    edges=None,
    extra: Sequence[dict] = (),
    shade=True,
    post_color=None,
    post_label="posterior",
):
    """One parameter panel: truth + prior band + posterior band (+ extra series)."""
    if shade:
        S.shade_windows(ax, edges)
    if prior_mean is not None:
        S.band(
            ax,
            time,
            prior_mean,
            prior_std,
            S.COLORS["prior"],
            alpha=0.18,
            label="prior",
            lw=1.6,
            ls="--",
        )
    for s in extra:
        S.band(
            ax,
            s["time"],
            s["mean"],
            s.get("std"),
            s["color"],
            alpha=s.get("alpha", 0.0),
            label=s.get("label"),
            lw=s.get("lw", 1.6),
            ls=s.get("ls", "-"),
        )
    if post_mean is not None:
        S.band(
            ax,
            time,
            post_mean,
            post_std,
            post_color or S.COLORS["posterior"],
            alpha=0.25,
            label=post_label,
            lw=2.2,
        )
    if truth is not None:
        ax.plot(
            truth_t if truth_t is not None else time,
            truth,
            color=S.COLORS["truth"],
            lw=2.0,
            label="truth",
            zorder=6,
        )
    S.mark_windows(ax, edges, annotate=False)
    ax.margins(x=0.01)


def plot_param_trajectories(
    out_path,
    *,
    prior,
    post,
    true,
    edges,
    extra_by_param=None,
    post_label="posterior",
    title=None,
):
    """Stacked α / |U| panels for a single run (truth/prior/posterior)."""
    params = [p for p in dataio.PARAMS if p in post.data_vars]
    fig, axes = plt.subplots(
        len(params),
        1,
        figsize=(8.2, 2.7 * len(params)),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, p in zip(axes, params):
        t = np.asarray(post["time"].values, dtype=float)
        pm = np.asarray(post[p].transpose("ensemble", "time").values)
        prm = (
            np.asarray(prior[p].transpose("ensemble", "time").values)
            if prior is not None and p in prior.data_vars
            else None
        )
        tr_t = np.asarray(true["time"].values) if "time" in true.dims else t
        param_panel(
            ax,
            time=t,
            post_mean=pm.mean(0),
            post_std=pm.std(0),
            prior_mean=None if prm is None else prm.mean(0),
            prior_std=None if prm is None else prm.std(0),
            truth_t=tr_t,
            truth=np.asarray(true[p].values) if p in true else None,
            edges=edges,
            extra=(extra_by_param or {}).get(p, ()),
            post_label=post_label,
        )
        ax.set_ylabel(S.PARAM_LABELS.get(p, p))
        ax.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time [s]")
    return S.save_pdf(fig, out_path)


# ---------------------------------------------------------------------------
# Error vs time (B2) and generic multi-series line
# ---------------------------------------------------------------------------
def plot_lines_vs_time(
    out_path, series, *, edges=None, ylabel="", xlabel="Time [s]", title=None
):
    """series: list of dict(time, y, color, label, lw?, ls?)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.0), constrained_layout=True)
    S.shade_windows(ax, edges)
    for s in series:
        ax.plot(
            s["time"],
            s["y"],
            color=s["color"],
            label=s.get("label"),
            lw=s.get("lw", 1.8),
            ls=s.get("ls", "-"),
        )
    S.mark_windows(ax, edges)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.01)
    ax.legend(loc="best", ncol=2)
    return S.save_pdf(fig, out_path)


# ---------------------------------------------------------------------------
# Error / metric vs resolution (A2, A5)
# ---------------------------------------------------------------------------
def plot_vs_resolution(
    out_path, panels, *, xlabel=r"Grid spacing $\Delta x$ [m]", truth_dx=None
):
    """panels: list of dict(title, ylabel, by_model={model: (dx[], y[])}).

    One subplot per entry; one line+marker per model. Linear x with explicit
    ticks at the actual Δx values (the resolution set isn't geometric), finer to
    the right.
    """
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(5.2 * n, 4.0), constrained_layout=True, squeeze=False
    )
    axes = axes[0]
    all_dx = sorted(
        {v for panel in panels for dx, _ in panel["by_model"].values() for v in dx}
    )
    for ax, panel in zip(axes, panels):
        for model, (dx, y) in panel["by_model"].items():
            order = np.argsort(dx)
            ax.plot(
                np.asarray(dx)[order],
                np.asarray(y)[order],
                marker=S.MODEL_MARKERS.get(model, "o"),
                ms=7,
                color=S.MODEL_COLORS.get(model),
                label=S.MODEL_LABELS.get(model, model),
            )
        if truth_dx is not None and truth_dx not in all_dx:
            ax.axvline(
                truth_dx,
                color=S.COLORS["truth"],
                ls=":",
                lw=1.2,
                label="truth resolution",
            )
        ax.set_xticks(all_dx)
        ax.set_xticklabels([f"{v:.2g}" for v in all_dx])
        ax.minorticks_off()
        ax.invert_xaxis()  # finer (smaller dx) to the right
        ax.set_xlabel(xlabel)
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel.get("title", ""))
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.08)
        ax.legend(loc="best")
    return S.save_pdf(fig, out_path)


# ---------------------------------------------------------------------------
# Validation-sensor time series (A4, B4)
# ---------------------------------------------------------------------------
def plot_sensor_timeseries(
    out_path, *, time, truth_ts, series, sensor_xyz, edges=None, title=None
):
    """One small panel per sensor.

    truth_ts: (n_sensor, n_time); series: list of dict(label, color, ts(n_sensor,n_time),
    std(optional same shape)).
    """
    n = truth_ts.shape[0]
    fig, axes = plt.subplots(
        n, 1, figsize=(8.2, 2.4 * n), constrained_layout=True, squeeze=False
    )
    axes = axes[:, 0]
    for i, ax in enumerate(axes):
        S.shade_windows(ax, edges)
        for s in series:
            std = s.get("std")
            S.band(
                ax,
                time,
                s["ts"][i],
                None if std is None else std[i],
                s["color"],
                alpha=0.2,
                label=s.get("label"),
                lw=1.8,
                ls=s.get("ls", "-"),
            )
        ax.plot(
            time, truth_ts[i], color=S.COLORS["truth"], lw=2.0, label="truth", zorder=6
        )
        S.mark_windows(ax, edges, annotate=False)
        x, y, z = sensor_xyz[i]
        ax.set_ylabel("|U| [m/s]")
        ax.set_title(f"sensor ({x:.0f}, {y:.0f}, {z:.0f})", fontsize=9)
        ax.margins(x=0.01)
        if i == 0:
            ax.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("Time [s]")
    return S.save_pdf(fig, out_path)


# ---------------------------------------------------------------------------
# Field + difference heatmap grid (A3, B3, C1)
# ---------------------------------------------------------------------------
def plot_field_error_grid(
    out_path,
    *,
    columns,
    truth_field,
    extent,
    z_label="",
    mask2d=None,
    assim_xy=None,
    val_xy=None,
    dpi=300,
):
    """Two-row image grid: |U| fields (row 0) and (model-truth) error (row 1).

    columns: list of (label, field2d). The first column is conventionally the
    truth (its error row is left blank and used for the sensor legend).
    """
    ncol = len(columns)
    fig, axes = plt.subplots(
        2, ncol, figsize=(2.7 * ncol, 5.6), constrained_layout=True, squeeze=False
    )

    def _mask(f):
        if mask2d is not None and f is not None and f.shape == mask2d.shape:
            return np.where(mask2d, np.nan, f)
        return f

    fields = [_mask(f) for _, f in columns]
    vmax = np.nanmax([np.nanmax(f) for f in fields if f is not None])
    vmin = 0.0
    tf = _mask(truth_field)
    errs = [
        None if lbl_is_truth else (_mask(f) - tf)
        for (lbl_is_truth, f) in [(i == 0, f) for i, (_, f) in enumerate(columns)]
    ]
    emax = np.nanmax([np.nanmax(np.abs(e)) for e in errs if e is not None] or [1.0])

    im0 = im1 = None
    for j, (label, _) in enumerate(columns):
        ax = axes[0, j]
        im0 = ax.imshow(
            fields[j],
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=S.CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(label, fontsize=10)
        _hatch(ax, mask2d, extent)
        if j == 0:
            if assim_xy is not None:
                ax.plot(
                    assim_xy[:, 0],
                    assim_xy[:, 1],
                    "o",
                    mfc="none",
                    mec="w",
                    ms=5,
                    mew=1.2,
                    label="assim",
                )
            if val_xy is not None:
                ax.plot(
                    val_xy[:, 0],
                    val_xy[:, 1],
                    "^",
                    mfc="none",
                    mec="w",
                    ms=6,
                    mew=1.2,
                    label="valid",
                )
        ax.set_xlabel("x [m]")
        if j == 0:
            ax.set_ylabel("y [m]")

        axe = axes[1, j]
        if errs[j] is None:
            axe.axis("off")
        else:
            im1 = axe.imshow(
                errs[j],
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=S.CMAP_DIFF,
                vmin=-emax,
                vmax=emax,
            )
            _hatch(axe, mask2d, extent)
            axe.set_xlabel("x [m]")
            if j == 0 or (j == 1):
                axe.set_ylabel("y [m]")
    if im0 is not None:
        fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label="|U| [m/s]")
    if im1 is not None:
        fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label="error [m/s]")
    return S.save_png(fig, out_path, dpi=dpi)


def plot_field_time_montage(
    out_path,
    *,
    times,
    truth_fields,
    est_fields,
    extent,
    est_label="estimate",
    z_label="",
    mask2d=None,
    assim_xy=None,
    val_xy=None,
    dpi=300,
):
    """3-row montage over times: row 0 truth, row 1 estimate, row 2 error.

    Each column is one time; the error is ``estimate(t) - truth(t)`` (per-column
    truth, unlike :func:`plot_field_error_grid` which shares one truth). Used by
    Block C's full-joint showcase.
    """
    ncol = len(times)

    def _mask(f):
        if mask2d is not None and f is not None and f.shape == mask2d.shape:
            return np.where(mask2d, np.nan, f)
        return f

    tr = [_mask(f) for f in truth_fields]
    es = [_mask(f) for f in est_fields]
    errs = [e - t for e, t in zip(es, tr)]
    vmax = float(np.nanmax([np.nanmax(f) for f in (tr + es)]))
    emax = float(np.nanmax([np.nanmax(np.abs(e)) for e in errs] or [1.0]))

    fig, axes = plt.subplots(
        3, ncol, figsize=(2.7 * ncol, 7.8), constrained_layout=True, squeeze=False
    )
    row_data = [
        ("truth", tr, S.CMAP_FIELD, dict(vmin=0, vmax=vmax)),
        (est_label, es, S.CMAP_FIELD, dict(vmin=0, vmax=vmax)),
        ("error", errs, S.CMAP_DIFF, dict(vmin=-emax, vmax=emax)),
    ]
    ims = [None, None, None]
    for ri, (rlabel, data, cmap, kw) in enumerate(row_data):
        for j, t in enumerate(times):
            ax = axes[ri, j]
            ims[ri] = ax.imshow(
                data[j], origin="lower", extent=extent, aspect="equal", cmap=cmap, **kw
            )
            _hatch(ax, mask2d, extent)
            if ri == 0:
                ax.set_title(f"t = {t:.0f} s", fontsize=10)
                if assim_xy is not None:
                    ax.plot(
                        assim_xy[:, 0],
                        assim_xy[:, 1],
                        "o",
                        mfc="none",
                        mec="w",
                        ms=5,
                        mew=1.2,
                    )
                if val_xy is not None:
                    ax.plot(
                        val_xy[:, 0],
                        val_xy[:, 1],
                        "^",
                        mfc="none",
                        mec="w",
                        ms=6,
                        mew=1.2,
                    )
            if j == 0:
                ax.set_ylabel(f"{rlabel}\ny [m]")
            if ri == 2:
                ax.set_xlabel("x [m]")
    fig.colorbar(ims[0], ax=axes[:2, :].ravel().tolist(), shrink=0.6, label="|U| [m/s]")
    fig.colorbar(
        ims[2], ax=axes[2, :].ravel().tolist(), shrink=0.8, label="error [m/s]"
    )
    return S.save_png(fig, out_path, dpi=dpi)


def _hatch(ax, mask2d, extent):
    """Hatch solid (building) cells."""
    if mask2d is None:
        return
    ax.contourf(
        np.linspace(extent[0], extent[1], mask2d.shape[1]),
        np.linspace(extent[2], extent[3], mask2d.shape[0]),
        mask2d.astype(float),
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
    )
    ax.contour(
        np.linspace(extent[0], extent[1], mask2d.shape[1]),
        np.linspace(extent[2], extent[3], mask2d.shape[0]),
        mask2d.astype(float),
        levels=[0.5],
        colors="0.3",
        linewidths=0.5,
    )
