"""Figures of a forward-model comparison.

    python scripts/comparison/visualize_comparison.py <run dir> [key=value ...]

The run dir is what run_models.py wrote and compute_metrics.py scored; the
overrides are applied to its config.yaml (e.g. `figures.slice_heights=[0.2]`).
Each model keeps one colour in every figure (the backend's colour of
evaluation.style where it is free); its replica 0 is drawn solid, the other
replicas thin.

Outputs, in the run dir:
  profiles.png          horizontally averaged mean u, mean v, TKE and u'w' vs
                        height, canopy height dashed
  slices_speed.png      time-mean |U| at `figures.slice_heights` (x H; the
                        nearest level): one column per model (replica mean),
                        then one per pair difference a - b
  slices_tke.png        the same for the resolved TKE
  sensor_series.png     |U| at every sensor against time
  sensor_spectra.png    frequency spectra of u, v, w and |U|, averaged over the
                        sensors (only when compute_metrics.py computed them)
  difference_noise.png  the difference / noise ratio of every pair: the
                        headline statistics on all fluid cells, then the
                        field statistics per height band; 1 marks a difference
                        no larger than the models' realisation noise
  animation_speed.mp4   instantaneous |U| at the slice heights against time,
                        laid out as slices_speed.png (replica
                        `figures.animation.replica` of every model); a .gif
                        when ffmpeg is unavailable. `figures.animation: null`
                        skips it
All figures have a white background, so they read in a dark-mode viewer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import xarray
import yaml
from evaluation.style import (
    CMAP_DIFF,
    CMAP_FIELD,
    COLORS,
    MODEL_COLORS,
    apply_style,
    save_png,
)
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from slice_animation import Panel, animate_slices, frame_indices  # noqa: E402

SOLVER_COLOR = {
    "udales": MODEL_COLORS["udales"],
    "palm": MODEL_COLORS["palm"],
    "pylbm": MODEL_COLORS["lbm"],
}
PALETTE = ["#6A3D9A", "#33A02C", "#E31A1C", COLORS["lightblue"], COLORS["charcoal"]]
PROFILE_LABELS = {
    "u": "mean u [m/s]",
    "v": "mean v [m/s]",
    "tke": "TKE [m²/s²]",
    "uw": "u'w' [m²/s²]",
}
VELOCITY = ("u", "v", "w")
SLICES = {"speed": "time-mean |U| [m/s]", "tke": "TKE [m²/s²]"}
# The bars of difference_noise.png: (metric, quantity) on all fluid cells / sensors.
HEADLINE = (
    ("field", "mean_u"),
    ("field", "mean_v"),
    ("field", "speed"),
    ("field", "tke"),
    ("field", "uw"),
    ("profile", "u"),
    ("profile", "tke"),
    ("sensor_mean", "speed"),
    ("sensor_std", "speed"),
    ("sensor_w2", "speed"),
    ("sensor_lsd", "u"),
)
BANDED = ("mean_u", "speed", "tke")


def run(run_dir: pathlib.Path, overrides: list[str] | None = None) -> None:
    run_dir = pathlib.Path(run_dir)
    cfg = OmegaConf.merge(
        OmegaConf.load(run_dir / "config.yaml"), OmegaConf.from_dotlist(overrides or [])
    )
    assert isinstance(cfg, DictConfig)
    apply_style()
    models = yaml.safe_load((run_dir / "models.yaml").read_text())
    colors = model_colors(models)
    metrics = xarray.load_dataset(run_dir / "metrics.nc")
    rows = json.loads((run_dir / "metrics.json").read_text())["rows"]

    _plot_profiles(metrics, colors, run_dir / "profiles.png")
    for quantity, label in SLICES.items():
        _plot_slices(metrics, quantity, label, cfg, run_dir / f"slices_{quantity}.png")
    _plot_sensor_series(run_dir, models, colors, run_dir / "sensor_series.png")
    if "spectrum" in metrics:
        _plot_spectra(metrics, colors, run_dir / "sensor_spectra.png")
    _plot_ratios(rows, list(models), run_dir / "difference_noise.png")
    if cfg.figures.get("animation") is not None:
        _animate_speed(run_dir, metrics, cfg)
    print(f"Saved comparison figures in {run_dir}")


def model_colors(models: dict) -> dict[str, str]:
    """One colour per label: its backend's, unless an earlier label took it."""
    colors: dict[str, str] = {}
    spare = iter(PALETTE * len(models))
    for label, entry in models.items():
        color = SOLVER_COLOR.get(entry["solver_name"])
        colors[label] = color if color and color not in colors.values() else next(spare)
    return colors


def _plot_profiles(metrics: xarray.Dataset, colors: dict, path: pathlib.Path) -> None:
    quantities = list(metrics.profile_quantity.values)
    fig, axes = plt.subplots(
        1, len(quantities), figsize=(3.2 * len(quantities), 4.2), sharey=True
    )
    height = float(metrics.attrs["canopy_height"])
    for ax, q in zip(np.atleast_1d(axes), quantities):
        for label in metrics.label.values:
            profile = metrics.profile.sel(label=label, profile_quantity=q)
            for r in profile.replica.values:
                values = profile.sel(replica=r).values
                if np.isfinite(values).any():
                    ax.plot(
                        values,
                        metrics.z,
                        color=colors[str(label)],
                        lw=1.8 if r == 0 else 0.8,
                        alpha=1.0 if r == 0 else 0.6,
                        label=str(label) if r == 0 else None,
                    )
        ax.axhline(height, color=COLORS["prior"], ls="--", lw=1)
        ax.set_xlabel(PROFILE_LABELS.get(str(q), str(q)))
    axes_list = np.atleast_1d(axes)
    axes_list[0].set_ylabel("z [m]")
    axes_list[0].legend()
    save_png(fig, path, transparent=False)


def _levels(metrics: xarray.Dataset, cfg: DictConfig) -> list[int]:
    """The z indices nearest to `figures.slice_heights` x H."""
    height = float(metrics.attrs["canopy_height"])
    z = metrics.z.values
    return [int(np.abs(z - f * height).argmin()) for f in cfg.figures.slice_heights]


def _plot_slices(
    metrics: xarray.Dataset,
    quantity: str,
    label: str,
    cfg: DictConfig,
    path: pathlib.Path,
) -> None:
    labels = [str(lb) for lb in metrics.label.values]
    pairs = list(itertools.combinations(labels, 2))
    levels = _levels(metrics, cfg)
    stat = metrics.stat.sel(quantity=quantity).where(metrics.solid == 0)
    columns = len(labels) + len(pairs)
    fig, axes = plt.subplots(
        len(levels),
        columns,
        figsize=(3.0 * columns, 2.8 * len(levels)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    extent = [
        float(metrics.x[0]),
        float(metrics.x[-1]),
        float(metrics.y[0]),
        float(metrics.y[-1]),
    ]
    for row, k in enumerate(levels):
        plane = {lb: stat.sel(label=lb).isel(z=k).values for lb in labels}
        finite = np.concatenate([p[np.isfinite(p)] for p in plane.values()])
        vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
        diffs = {(a, b): plane[a] - plane[b] for a, b in pairs}
        spread = max(
            [np.nanmax(np.abs(d)) for d in diffs.values() if np.isfinite(d).any()],
            default=1.0,
        )
        panels = [(lb, plane[lb], CMAP_FIELD, vmin, vmax) for lb in labels]
        panels += [
            (f"{a} - {b}", diffs[(a, b)], CMAP_DIFF, -spread, spread) for a, b in pairs
        ]
        for col, (title, values, cmap, lo, hi) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                aspect="equal",
            )
            ax.set_title(f"{title}, z = {float(metrics.z[k]):.3g} m", fontsize=9)
            if col == len(labels) - 1:  # one bar per group keeps panels equal
                fig.colorbar(image, ax=list(axes[row, : len(labels)]), shrink=0.8)
            elif col == columns - 1:
                fig.colorbar(image, ax=list(axes[row, len(labels) :]), shrink=0.8)
        axes[row, 0].set_ylabel("y [m]")
    for ax in axes[-1]:
        ax.set_xlabel("x [m]")
    fig.suptitle(label)
    save_png(fig, path, transparent=False)


def _animate_speed(
    run_dir: pathlib.Path, metrics: xarray.Dataset, cfg: DictConfig
) -> pathlib.Path:
    """Instantaneous |U| at the slice heights, one column per model then per
    pair difference, one frame per output time (thinned to `max_frames`).
    Colour limits are fixed per row over the whole run, so frames compare."""
    opts = cfg.figures.animation
    labels = [str(lb) for lb in metrics.label.values]
    pairs = list(itertools.combinations(labels, 2))
    levels = _levels(metrics, cfg)
    speed = {}
    for label in labels:
        path = run_dir / "fields" / label / f"replica_{int(opts.replica)}.nc"
        with xarray.open_dataset(path) as ds:
            planes = ds[["u", "v", "w"]].isel(z=levels).load()
            time = ds.time.values
        speed[label] = np.sqrt(sum(planes[c].values ** 2 for c in VELOCITY))
    n_time = min(len(s) for s in speed.values())
    frames = frame_indices(n_time, int(opts.max_frames))
    rows = []
    for row, k in enumerate(levels):
        at = f"z = {float(metrics.z[k]):.3g} m"
        plane = {lb: speed[lb][frames, row] for lb in labels}
        vmax = float(np.nanpercentile(np.concatenate(list(plane.values())), 99.5))
        diffs = {(a, b): plane[a] - plane[b] for a, b in pairs}
        spread = max(
            (float(np.nanpercentile(np.abs(d), 99.5)) for d in diffs.values()),
            default=1.0,
        )
        rows.append(
            [Panel(f"{lb}, {at}", plane[lb], CMAP_FIELD, 0.0, vmax, "value") for lb in labels]
            + [
                Panel(f"{a} - {b}, {at}", d, CMAP_DIFF, -spread, spread, "diff")
                for (a, b), d in diffs.items()
            ]
        )
    return animate_slices(
        rows,
        time[frames],
        run_dir / "animation_speed.mp4",
        title="instantaneous |U| [m/s]",
        fps=int(opts.fps),
        dpi=int(opts.dpi),
        extent=[
            float(metrics.x[0]),
            float(metrics.x[-1]),
            float(metrics.y[0]),
            float(metrics.y[-1]),
        ],
    )

def _plot_sensor_series(
    run_dir: pathlib.Path, models: dict, colors: dict, path: pathlib.Path
) -> None:
    series = {}
    for label, entry in models.items():
        for r in range(len(entry["replicas"])):
            with xarray.open_dataset(
                run_dir / "fields" / label / f"replica_{r}.nc"
            ) as ds:
                speed = np.sqrt(
                    sum(ds[f"sensor_{c}"].values ** 2 for c in ("u", "v", "w"))
                )
                series[(label, r)] = (ds.time.values, speed)
                sensors = [
                    f"{s} ({x:g}, {y:g}, {z:g})"
                    for s, x, y, z in zip(
                        ds.sensor_set.values,
                        ds.sensor_x.values,
                        ds.sensor_y.values,
                        ds.sensor_z.values,
                    )
                ]
    fig, axes = plt.subplots(
        len(sensors), 1, figsize=(8, 1.6 * len(sensors)), sharex=True, squeeze=False
    )
    for s, ax in enumerate(axes[:, 0]):
        for (label, r), (time, speed) in series.items():
            ax.plot(
                time,
                speed[:, s],
                color=colors[label],
                lw=1.4 if r == 0 else 0.7,
                alpha=1.0 if r == 0 else 0.6,
                label=label if r == 0 else None,
            )
        ax.set_title(sensors[s], fontsize=9, loc="left")
        ax.set_ylabel("|U| [m/s]")
    axes[0, 0].legend(ncol=len(models))
    axes[-1, 0].set_xlabel("time [s]")
    save_png(fig, path, transparent=False)


def _plot_spectra(metrics: xarray.Dataset, colors: dict, path: pathlib.Path) -> None:
    quantities = list(metrics.sensor_quantity.values)
    fig, axes = plt.subplots(
        1, len(quantities), figsize=(3.4 * len(quantities), 3.4), sharey=True
    )
    f = metrics.frequency.values
    for ax, q in zip(np.atleast_1d(axes), quantities):
        for label in metrics.label.values:
            spectra = metrics.spectrum.sel(label=label, sensor_quantity=q)
            for r in spectra.replica.values:
                mean = spectra.sel(replica=r).mean("sensor").values
                if np.isfinite(mean).any():
                    ax.loglog(
                        f,
                        mean,
                        color=colors[str(label)],
                        lw=1.6 if r == 0 else 0.8,
                        alpha=1.0 if r == 0 else 0.6,
                        label=str(label) if r == 0 else None,
                    )
        ax.set_title("|U|" if q == "speed" else str(q))
        ax.set_xlabel("frequency [Hz]")
    axes_list = np.atleast_1d(axes)
    axes_list[0].set_ylabel("PSD [m²/s² / Hz], sensor mean")
    axes_list[0].legend()
    save_png(fig, path, transparent=False)


def _plot_ratios(rows: list[dict], labels: list[str], path: pathlib.Path) -> None:
    pairs = list(itertools.combinations(labels, 2))
    index = {
        (r["metric"], r["quantity"], r["region"], r["model_a"], r["model_b"]): r[
            "ratio"
        ]
        for r in rows
    }
    regions = list(dict.fromkeys(r["region"] for r in rows if r["metric"] == "field"))
    groups = [
        [(f"{m}\n{q}", (m, q, "all")) for m, q in HEADLINE],
        [
            (f"{q}\n{region}", ("field", q, region))
            for q in BANDED
            for region in regions
        ],
    ]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.9 * len(groups[1])), 7))
    palette = PALETTE + [COLORS["posterior"], COLORS["orange"]]
    width = 0.8 / max(len(pairs), 1)
    for ax, group in zip(axes, groups):
        names = [name for name, _ in group]
        top = 1.0
        for p, (a, b) in enumerate(pairs):
            found = [index.get((*key, a, b)) for _, key in group]
            values = [np.nan if v is None else float(v) for v in found]
            top = max([top, *(v for v in values if np.isfinite(v))])
            ax.bar(
                np.arange(len(group)) + (p - (len(pairs) - 1) / 2) * width,
                values,
                width,
                color=palette[p % len(palette)],
                label=f"{a} vs {b}",
            )
        ax.axhline(1.0, color=COLORS["truth"], lw=1, ls="--")
        ax.set_xticks(np.arange(len(group)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("difference / noise")
        if top > 20:
            ax.set_yscale("log")
    if pairs:
        axes[0].legend()
    save_png(fig, path, transparent=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="config overrides, e.g. figures.slice_heights=[0.2]",
    )
    args = parser.parse_args()
    run(args.run_dir, args.overrides)


if __name__ == "__main__":
    main()
