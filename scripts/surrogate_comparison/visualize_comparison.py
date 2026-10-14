"""Figures and the headline table of a surrogate comparison run.

    python scripts/surrogate_comparison/visualize_comparison.py <run dir> [key=value ...]

The overrides are applied to the run's config.yaml (e.g. `figures.level=4`).
The run dir is what run_models.py wrote and compute_metrics.py scored. Every
figure reads metrics.nc / metrics.yaml only, except the two slice figures,
which read the rollouts and the truth from the corpus. Each model keeps one
colour in every figure; persistence is dashed grey, the truth solid black.
Curves that leave the axis (an unstable model) are clipped and marked
"off scale" in the legend, so the others stay readable.

Outputs, in the run dir:
  rmse.png               RMSE vs lead time: all components, then u, v, w
  acc.png                anomaly correlation vs lead time, `metrics.acc_threshold` line
  energy.png             spatial std of u, v, w relative to the truth vs lead time
  divergence.png         RMS of the velocity divergence vs lead time, truth as reference
  profiles.png           time-mean u, resolved TKE and -<u'w'> vs height
  spectra.png            spanwise spectra of u' and w' at `figures.spectra_levels`
  pdfs.png               velocity PDFs per component, log density
  slices.png             |U| at z index `figures.level` at `figures.lead_steps`:
                         truth, then one row per model (trajectory `figures.sample`)
  statistics_slices.png  time-mean u and resolved TKE at `figures.level`: truth,
                         then one row per model (trajectory `figures.sample`)
  summary.csv/.tex       headline scalars per model and persistence (mean over the samples)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Any, Collection

import numpy as np
import xarray
import yaml
from evaluation.style import COLORS, apply_style, save_png, write_table
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import load_states, open_dataset, plot_slice_rows, speed  # noqa: E402

# One colour per model, in models.yaml order; the reference sources are fixed.
PALETTE = [
    COLORS["posterior"],
    COLORS["orange"],
    "#6A3D9A",
    "#33A02C",
    COLORS["amber"],
    "#E31A1C",
    COLORS["lightblue"],
    COLORS["charcoal"],
]
REFERENCE = {
    "persistence": {"color": COLORS["prior"], "ls": "--"},
    "truth": {"color": COLORS["truth"], "ls": "-", "lw": 2.0},
}
VARS = ("u", "v", "w")
LEAD_LABEL = "lead time [s]"


def run(run_dir: pathlib.Path, overrides: list[str] | None = None) -> None:
    run_dir = pathlib.Path(run_dir)
    cfg = OmegaConf.merge(
        OmegaConf.load(run_dir / "config.yaml"), OmegaConf.from_dotlist(overrides or [])
    )
    assert isinstance(cfg, DictConfig)
    apply_style()
    models = dict(yaml.safe_load((run_dir / "models.yaml").read_text()))
    styles = {
        name: {"color": PALETTE[i % len(PALETTE)], "ls": "-"}
        for i, name in enumerate(models)
    } | REFERENCE
    metrics = xarray.load_dataset(run_dir / "metrics.nc")
    scalars = yaml.safe_load((run_dir / "metrics.yaml").read_text())

    _lead_figure(
        metrics,
        styles,
        ["rmse", *(f"rmse_{v}" for v in VARS)],
        ["all components", *VARS],
        "RMSE [m/s]",
        run_dir / "rmse.png",
    )
    threshold = float(cfg.metrics.acc_threshold)
    _lead_figure(
        metrics,
        styles,
        ["acc"],
        ["anomaly correlation"],
        "ACC",
        run_dir / "acc.png",
        reference=threshold,
        limits=(-0.2, 1.02),
    )
    _lead_figure(
        metrics,
        styles,
        [f"std_ratio_{v}" for v in VARS],
        [f"spatial std of {v} / truth" for v in VARS],
        "ratio",
        run_dir / "energy.png",
        reference=1.0,
        limits=(0.0, 2.5),
    )
    _lead_figure(
        metrics,
        styles,
        ["divergence_rms"],
        ["divergence RMS (fluid interior)"],
        "RMS of ∇·u [1/s]",
        run_dir / "divergence.png",
    )
    _plot_profiles(metrics, styles, run_dir / "profiles.png")
    _plot_spectra(metrics, styles, run_dir / "spectra.png")
    _plot_pdfs(metrics, styles, run_dir / "pdfs.png")
    _plot_slices(cfg, run_dir, models)
    _write_summary(scalars, [*models, "persistence"], run_dir)
    print(f"Saved comparison figures in {run_dir}")


def _sources(metrics: xarray.Dataset, styles: dict, name: str) -> list[str]:
    """The sources with a finite value of `name`, models first, truth last."""
    if name not in metrics:
        return []
    have = [
        str(s)
        for s in metrics.source.values
        if np.isfinite(metrics[name].sel(source=s).values).any()
    ]
    return [s for s in styles if s in have]


def _clip_top(curves: dict[str, np.ndarray]) -> float | None:
    """An upper y-limit that keeps the typical curves readable: 3 x the median
    of the sources' maxima, or None when every curve fits under it."""
    peaks = [
        np.nanmax(c[np.isfinite(c)]) for c in curves.values() if np.isfinite(c).any()
    ]
    if not peaks:
        return None
    top = 3 * float(np.median(peaks))
    return top if max(peaks) > top else None


def _lead_figure(
    metrics: xarray.Dataset,
    styles: dict,
    names: list[str],
    titles: list[str],
    ylabel: str,
    path: pathlib.Path,
    reference: float | None = None,
    limits: tuple[float, float] | None = None,
) -> None:
    """One panel per metrics.nc variable against lead time, one line per source."""
    panels = [(n, t) for n, t in zip(names, titles) if _sources(metrics, styles, n)]
    if not panels:
        print(f"no {names} in metrics.nc: no {path.name}")
        return
    lead_time = metrics["lead_time"].values
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False
    )
    off: set[str] = set()
    for ax, (name, title) in zip(axes[0], panels):
        curves = {
            s: metrics[name].sel(source=s).values
            for s in _sources(metrics, styles, name)
        }
        top = _clip_top(curves) if limits is None else limits[1]
        for source, curve in curves.items():
            if top is not None and bool(np.nanmax(curve) > top):
                off.add(source)
            ax.plot(lead_time, curve, label=source, **styles[source])
        if reference is not None:
            ax.axhline(reference, color="0.4", lw=0.8, ls=":")
        if limits is not None:
            low, high = ax.get_ylim()
            ax.set_ylim(max(low, limits[0]), min(high, limits[1]))
        elif top is not None:
            ax.set_ylim(0, top)
        ax.set_title(title)
        ax.set_xlabel(LEAD_LABEL)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(ylabel)
    fig.tight_layout()
    _legend(fig, off)
    save_png(fig, path, dpi=150, transparent=False)


def _legend(fig: Any, off: Collection[str] = ()) -> None:
    """One legend for the figure, right of the panels, every source once;
    the sources in `off` are marked as clipped."""
    entries: dict[str, Any] = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    labels = [f"{s} (off scale)" if s in off else s for s in entries]
    fig.legend(entries.values(), labels, loc="center left", bbox_to_anchor=(1.0, 0.5))


def _plot_profiles(metrics: xarray.Dataset, styles: dict, path: pathlib.Path) -> None:
    keys = {
        "profile_mean_u": "mean u [m/s]",
        "profile_tke": "resolved TKE [m²/s²]",
        "profile_uw": "-<u'w'> [m²/s²]",
    }
    keys = {k: v for k, v in keys.items() if _sources(metrics, styles, k)}
    if not keys:
        print("no profiles in metrics.nc: no profiles.png")
        return
    z = metrics["z"].values
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, keys.items()):
        for source in _sources(metrics, styles, key):
            ax.plot(metrics[key].sel(source=source), z, label=source, **styles[source])
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("z [m]")
    fig.tight_layout()
    _legend(fig)
    save_png(fig, path, dpi=150, transparent=False)


def _plot_spectra(metrics: xarray.Dataset, styles: dict, path: pathlib.Path) -> None:
    rows = [v for v in ("u", "w") if _sources(metrics, styles, f"spectrum_{v}")]
    if not rows or "level" not in metrics.dims:
        print("no spectra in metrics.nc: no spectra.png")
        return
    levels = metrics["level"].values
    k = metrics["k"].values
    fig, axes = plt.subplots(
        len(rows),
        len(levels),
        figsize=(4.3 * len(levels), 3.4 * len(rows)),
        squeeze=False,
        sharex=True,
    )
    positive = k > 0
    for r, var in enumerate(rows):
        name = f"spectrum_{var}"
        for c, level in enumerate(levels):
            ax = axes[r, c]
            for source in _sources(metrics, styles, name):
                energy = metrics[name].sel(source=source, level=level).values
                if not (energy[positive] > 0).any():  # persistence: no fluctuations
                    continue
                ax.loglog(k[positive], energy[positive], label=source, **styles[source])
            ax.set_title(f"{var}', z index {int(level)}")
            ax.grid(alpha=0.3, which="both")
            if r == len(rows) - 1:
                ax.set_xlabel("k [1/m]")
        axes[r, 0].set_ylabel("E(k) [m³/s²]")
    fig.tight_layout()
    _legend(fig)
    save_png(fig, path, dpi=150, transparent=False)


def _plot_pdfs(metrics: xarray.Dataset, styles: dict, path: pathlib.Path) -> None:
    sources = _sources(metrics, styles, "pdf")
    if not sources:
        print("no pdf in metrics.nc: no pdfs.png")
        return
    variables = [str(v) for v in metrics["var"].values]
    fig, axes = plt.subplots(
        1, len(variables), figsize=(4.2 * len(variables), 3.4), squeeze=False
    )
    for ax, var in zip(axes[0], variables):
        centres = metrics["pdf_bin_centre"].sel(var=var).values
        for source in sources:
            density = metrics["pdf"].sel(source=source, var=var).values
            ax.semilogy(centres, density, label=source, **styles[source])
        truth = metrics["pdf"].sel(source="truth", var=var).values
        if "truth" in sources and (truth > 0).any():
            # Down to 1e-4 of the truth's peak: the tails, not single-count noise.
            ax.set_ylim(1e-4 * np.nanmax(truth), 3 * np.nanmax(truth))
        ax.set_xlabel(f"{var} [m/s]")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("density [s/m]")
    fig.tight_layout()
    _legend(fig)
    save_png(fig, path, dpi=150, transparent=False)


def _plot_slices(cfg: DictConfig, run_dir: pathlib.Path, models: dict) -> None:
    """slices.png and statistics_slices.png of trajectory `figures.sample`."""
    first = next(iter(models))
    samples = list(cfg.sample_indices)  # set by run_models.py
    index = int(cfg.figures.sample)
    if not 0 <= index < len(samples):
        print(f"figures.sample={index} not among the {len(samples)} samples: no slices")
        return
    name_nc = f"sample_{int(samples[index]):04d}.nc"
    preds = {}
    for name in models:
        with xarray.open_dataset(run_dir / "rollouts" / name / name_nc) as ds:
            preds[name] = np.stack([ds[v].values for v in VARS], axis=1)
            lead, start, sample = (
                ds["lead"].values,
                ds.attrs["start"],
                ds.attrs["sample"],
            )
    level = int(cfg.figures.level)
    nz = preds[first].shape[2]
    if not 0 <= level < nz:
        print(f"figures.level={level} outside the grid's {nz} levels: no slices")
        return

    train_cfg = OmegaConf.load(pathlib.Path(models[first]["model_dir"]) / "config.yaml")
    assert isinstance(train_cfg, DictConfig)
    data = open_dataset(train_cfg, cfg.data_dir, cfg.split, sdf_features="none")
    fluid = data.geometry_for(int(sample)).numpy().astype(bool)[level]
    n = min(len(p) for p in preds.values())
    frames = [int(start), *(int(start) + int(t) for t in lead[:n])]
    with xarray.open_dataset(data._state_files[int(sample)]) as ds:
        times = ds["time"].values[frames]  # start frame first: lead s at times[s]
    truth = load_states(data, int(sample), frames[1:])

    # |U| at the lead steps; model slices clipped to the truth's range (with a
    # margin) so a diverging model does not wash out the shared colour scale.
    steps = []
    for step in cfg.figures.lead_steps:
        step = n if int(step) == -1 else int(step)
        if 1 <= step <= n:
            steps.append(step)
        else:
            print(f"lead step {step} beyond the {n}-step rollout: left out of slices")
    steps = list(dict.fromkeys(steps))
    if steps:
        reference = [speed(truth[s - 1])[level] for s in steps]
        bound = 1.5 * max(float(np.nanmax(r)) for r in reference)
        rows = [("truth", reference, False)]
        for name, pred in preds.items():
            rows.append(
                (
                    name,
                    [np.clip(speed(pred[s - 1])[level], 0, bound) for s in steps],
                    False,
                )
            )
        titles = [f"lead {s} ({times[s] - times[0]:.0f} s)" for s in steps]
        plot_slice_rows(rows, titles, fluid, run_dir / "slices.png")
        print(f"  + {run_dir / 'slices.png'}")

    # Time-mean u and resolved TKE over the rollout window, one scale per column.
    stats = {"truth": _statistics(truth, level)} | {
        name: _statistics(pred[:n], level) for name, pred in preds.items()
    }
    titles = ["time-mean u [m/s]", "resolved TKE [m²/s²]"]
    fig, axes = plt.subplots(
        len(stats),
        2,
        figsize=(8, 0.6 + 4 * fluid.shape[0] / fluid.shape[1] * len(stats)),
        squeeze=False,
        layout="constrained",
    )
    for c, title in enumerate(titles):
        # The truth's 1-99 % range: a hot spot or a diverging model saturates.
        ref = stats["truth"][c][fluid]
        low = float(np.nanpercentile(ref, 1)) if c == 0 else 0.0
        high = float(np.nanpercentile(ref, 99))
        for r, (name, fields) in enumerate(stats.items()):
            ax = axes[r, c]
            image = ax.imshow(
                np.where(fluid, fields[c], np.nan),
                origin="lower",
                vmin=low,
                vmax=high,
                cmap="viridis" if c == 0 else "magma",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
        fig.colorbar(image, ax=axes[:, c].tolist(), shrink=0.6, extend="both")
    fig.suptitle(f"rollout statistics at z index {level} (sample {sample})")
    save_png(fig, run_dir / "statistics_slices.png", dpi=150, transparent=False)


def _statistics(fields: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-mean u and resolved TKE (half the summed time variance) at `level`."""
    layer = fields[:, :, level]
    return layer[:, 0].mean(axis=0), 0.5 * layer.var(axis=0).sum(axis=0)


def _write_summary(scalars: dict, order: list[str], run_dir: pathlib.Path) -> None:
    # Per source, each scalar is {mean, per_sample} (a model's cost is plain).
    sources = [s for s in order if s in scalars]
    means = {
        s: {k: v["mean"] if isinstance(v, dict) else v for k, v in scalars[s].items()}
        for s in sources
    }
    steps = sorted(
        {k for m in means.values() for k in m if re.fullmatch(r"rmse_step\d+", k)},
        key=lambda k: int(k[len("rmse_step") :]),
    )
    keys = [
        "rmse",
        "rmse_vs_persistence",
        *steps,
        "acc_horizon_seconds",
        "tke_rmse",
        "profile_rmse_tke",
        "spectra_db_u_mid",
        "seconds_per_step",
        "n_parameters",
        "stable",
    ]

    def cell(key: str, value: Any) -> Any:
        if value is None:  # an ACC that never drops below the threshold
            return "never" if key == "acc_horizon_seconds" else ""
        if isinstance(value, bool):
            return str(value).lower()
        return float(f"{value:.4g}") if isinstance(value, float) else value

    rows = [
        [s, *(cell(k, means[s][k]) if k in means[s] else "" for k in keys)]
        for s in sources
    ]
    lower = [1 + keys.index(k) for k in keys if k.startswith(("rmse", "tke", "prof"))]
    write_table(run_dir / "summary", ["source", *keys], rows, bold_min_cols=lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "overrides", nargs="*", help="config overrides, e.g. figures.level=4"
    )
    args = parser.parse_args()
    run(args.run_dir, args.overrides)


if __name__ == "__main__":
    main()
