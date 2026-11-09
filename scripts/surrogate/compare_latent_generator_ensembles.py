"""Compare latent generators evaluated against the same seed ensemble.

Config: configs/surrogate/eval.yaml, block `latent_generator_ensemble_compare`.

    python scripts/surrogate/compare_latent_generator_ensembles.py \
        'latent_generator_ensemble_compare.eval_dirs=[model_weights/lg_a,model_weights/lg_b]' \
        'latent_generator_ensemble_compare.tags=[A,B]' \
        latent_generator_ensemble_compare.reference=B

Reads each model's scripts/surrogate/evaluate_latent_generator_ensemble.py
output (`eval_dirs`: the output dirs, or the model dirs holding
`eval_seed_ensemble/`). CPU only. The `real` floors and the `ae_recon` row come
from the first model (they do not depend on the generator). Model differences
are paired per geometry (same members, same noise): model minus `reference`,
mean over the geometries both evaluated, with a geometry-bootstrap CI.

Outputs, in `latent_generator_ensemble_compare.output_dir`:
  comparison.csv          mean and CI per model x source x metric
  comparison.md           the headline table (generated per model, AE, floors)
                          and the paired differences
  paired_differences.csv  model - reference per metric, mean and CI
  spread_ratio_profiles.png  mean spread ratio per height, per model
  rank_histograms.png     generated rank histograms per model, summed over
                          geometries, with the real leave-one-out one
  scores.png              CRPS ratio, energy distance (half size, with the
                          split-half floor) and spread ratio per model, with CIs
  paired_differences.png  model - reference for the key metrics, with CIs
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import hydra
import numpy as np
from matplotlib import pyplot as plt
from neural_surrogates import generator_evaluation as ge
from omegaconf import DictConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import write_csv  # noqa: E402

# The headline metrics, in table order (generated rows; `real` and `ae_recon`
# where they have them).
HEADLINE = (
    "spread_ratio_total",
    "spread_ratio_u",
    "spread_ratio_v",
    "spread_ratio_w",
    "spread_corr",
    "spread_log_rms_half",
    "crps_ratio",
    "rank_outer",
    "rank_chi2",
    "energy_distance_rel",
    "energy_distance_rel_half",
    "w1_points_half",
    "bias_standardised",
    "diversity_ratio",
    "profile_rmse",
    "spectra_lsd_db",
    "divergence_rms",
)
SUBDIR = "eval_seed_ensemble"


def run(cfg: DictConfig) -> None:
    cc = cfg.latent_generator_ensemble_compare
    if not cc.eval_dirs:
        raise ValueError("latent_generator_ensemble_compare.eval_dirs is empty")
    dirs = [_eval_dir(pathlib.Path(d)) for d in cc.eval_dirs]
    tags = (
        [str(t) for t in cc.tags]
        if cc.tags is not None
        else [d.parent.name if d.name == SUBDIR else d.name for d in dirs]
    )
    if len(tags) != len(dirs) or len(set(tags)) != len(tags):
        raise ValueError(f"need one distinct tag per eval dir, got {tags}")
    reference = tags[0] if cc.reference is None else str(cc.reference)
    if reference not in tags:
        raise ValueError(f"reference {reference!r} is not one of {tags}")
    out = pathlib.Path(cc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {
        t: json.loads((d / "summary.json").read_text()) for t, d in zip(tags, dirs)
    }
    n_boot, seed = int(cc.bootstrap_resamples), int(cfg.seed)

    rows = []
    for tag, summary in summaries.items():
        for source, metrics in summary["aggregate"].items():
            for key, ci in metrics.items():
                rows.append({"model": tag, "source": source, "metric": key, **ci})
    write_csv(rows, out / "comparison.csv")

    paired = []
    for tag in tags:
        if tag == reference:
            continue
        for key in HEADLINE + _probe_keys():
            diffs = _paired(summaries[tag], summaries[reference], key)
            paired.append(
                {"model": tag, "reference": reference, "metric": key}
                | ge.bootstrap_ci(diffs, n_boot, seed)
            )
    if paired:
        write_csv(paired, out / "paired_differences.csv")
    (out / "comparison.md").write_text(_markdown(summaries, tags, reference, paired))
    _plot_profiles(dirs, tags, summaries, out / "spread_ratio_profiles.png")
    _plot_ranks(dirs, tags, summaries, out / "rank_histograms.png")
    _plot_scores(summaries, tags, out / "scores.png")
    if paired:
        _plot_paired(paired, out / "paired_differences.png")
    print(f"Saved comparison of {tags} in {out}")


def _eval_dir(path: pathlib.Path) -> pathlib.Path:
    if (path / "summary.json").exists():
        return path
    if (path / SUBDIR / "summary.json").exists():
        return path / SUBDIR
    raise FileNotFoundError(f"no summary.json in {path} or {path / SUBDIR}")


def _probe_keys() -> tuple[str, ...]:
    return ("probe_energy_distance_rel_gain", "probe_crps_ratio_gain")


def _value(geometry: dict, source: str, key: str) -> float:
    """One per-geometry number; `probe_*_gain` is the shuffled-parameter probe
    minus the true conditioning (positive: the parameters matter)."""
    scalars = geometry["scalars"]
    if key.startswith("probe_") and key.endswith("_gain"):
        metric = key[len("probe_") : -len("_gain")]
        probe = scalars.get("generated_shuffled_params", {}).get(metric, np.nan)
        return float(probe) - float(scalars["generated"].get(metric, np.nan))
    return float(scalars.get(source, {}).get(key, np.nan))


def _paired(model: dict, reference: dict, key: str) -> list[float]:
    """Per-geometry `generated` differences model - reference, on the
    geometries both evaluated."""
    shared = sorted(set(model["per_geometry"]) & set(reference["per_geometry"]))
    return [
        _value(model["per_geometry"][g], "generated", key)
        - _value(reference["per_geometry"][g], "generated", key)
        for g in shared
    ]


def _ci(summary: dict, source: str, key: str, n_boot: int = 2000) -> dict[str, float]:
    if key in summary["aggregate"].get(source, {}):
        return dict(summary["aggregate"][source][key])
    values = [_value(g, source, key) for g in summary["per_geometry"].values()]
    return ge.bootstrap_ci(values, n_boot, 0)


def _fmt(ci: dict[str, float], significance: bool = False) -> str:
    """`mean [lo, hi]`; with `significance`, bold when the CI excludes 0."""
    if not np.isfinite(ci["mean"]):
        return ""
    if not (np.isfinite(ci["lo"]) and np.isfinite(ci["hi"])):
        return f"{ci['mean']:.4g}"
    text = f"{ci['mean']:.4g} [{ci['lo']:.3g}, {ci['hi']:.3g}]"
    if significance and (ci["lo"] > 0 or ci["hi"] < 0):
        return f"**{text}**"
    return text


def _markdown(
    summaries: dict[str, dict], tags: list[str], reference: str, paired: list[dict]
) -> str:
    first = summaries[tags[0]]
    keys = HEADLINE + _probe_keys()
    lines = [
        "# Latent generators against the seed ensemble",
        "",
        f"Mean over geometries [95 % geometry-bootstrap CI]; {first['sampler']}, "
        f"{first['num_steps']} steps, {first['num_draws']} draws per geometry. "
        "`floor`: the real ensemble against itself (leave-one-out CRPS and ranks, "
        "split-half `_half` values); `_half` model values are at the same size.",
        "",
        "| model | " + " | ".join(keys) + " |",
        "|---" * (len(keys) + 1) + "|",
    ]
    for tag in tags:
        cells = [_fmt(_ci(summaries[tag], "generated", k)) for k in keys]
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    for label, source in (("AE", "ae_recon"), ("floor", "real")):
        cells = [
            "" if k.startswith("probe_") else _fmt(_ci(first, source, k)) for k in keys
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if paired:
        lines += [
            "",
            f"## Paired differences: model - {reference} (generated, per geometry)",
            "",
            "Same members and same noise per geometry, so these differences are "
            "what separates the models; bold: the 95 % geometry-bootstrap CI "
            "excludes 0. The sampling floors (`*_half_std` on the floor row) are "
            "far smaller than the spread between geometries, so the CI decides.",
            "",
            "| model | " + " | ".join(keys) + " |",
            "|---" * (len(keys) + 1) + "|",
        ]
        for tag in tags:
            if tag == reference:
                continue
            by_key = {r["metric"]: r for r in paired if r["model"] == tag}
            lines.append(
                f"| {tag} | " + " | ".join(_fmt(by_key[k], True) for k in keys) + " |"
            )
    return "\n".join(lines) + "\n"


def _arrays(eval_dir: pathlib.Path, summary: dict) -> list[dict]:
    loaded = []
    for g in summary["per_geometry"]:
        with np.load(eval_dir / "arrays" / f"{g}.npz") as f:
            loaded.append(dict(f))
    return loaded


def _colors(tags: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(tags)}


def _plot_profiles(
    dirs: list[pathlib.Path], tags: list[str], summaries: dict, path: pathlib.Path
) -> None:
    names = summaries[tags[0]]["state_vars"]
    colors = _colors(tags)
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.6 * len(names), 4.0), squeeze=False
    )
    for i, (d, tag) in enumerate(zip(dirs, tags)):
        arrays = _arrays(d, summaries[tag])
        height = arrays[0]["height_m"]
        sources = [("generated", tag, colors[tag], "-")]
        if i == 0:
            sources.append(("ae_recon", "AE", "k", ":"))
        for source, label, color, ls in sources:
            key = f"{source}.spread_ratio_profile"
            for c in range(len(names)):
                curves = [
                    np.interp(
                        height, a["height_m"], a[key][c], left=np.nan, right=np.nan
                    )
                    for a in arrays
                ]
                axes[0, c].plot(
                    np.nanmean(curves, axis=0), height, color=color, ls=ls, label=label
                )
    for c, name in enumerate(names):
        axes[0, c].axvline(1.0, color="k", lw=0.8)
        axes[0, c].set_title(f"{name}: spread ratio (mean over geometries)")
        axes[0, c].set_xlabel("sigma_S / sigma_R")
        axes[0, c].set_ylabel("height [m]")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_ranks(
    dirs: list[pathlib.Path], tags: list[str], summaries: dict, path: pathlib.Path
) -> None:
    names = summaries[tags[0]]["state_vars"]
    colors = _colors(tags)
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.6 * len(names), 3.4), squeeze=False
    )
    for i, (d, tag) in enumerate(zip(dirs, tags)):
        arrays = _arrays(d, summaries[tag])
        sources = [("generated", tag, colors[tag], "-")]
        if i == 0:
            sources += [("real", "real (leave-one-out)", "k", "--")]
            sources += [("ae_recon", "AE", "tab:gray", ":")]
        for source, label, color, ls in sources:
            for c in range(len(names)):
                counts = sum(a[f"{source}.rank_counts"][c] for a in arrays)
                density = counts / (counts.sum() / len(counts))
                x = np.arange(len(counts))
                axes[0, c].step(
                    x, density, where="mid", color=color, ls=ls, label=label
                )
    for c, name in enumerate(names):
        axes[0, c].axhline(1.0, color="k", lw=0.6)
        axes[0, c].set_title(f"{name}: rank histogram")
        axes[0, c].set_xlabel("rank bin")
        axes[0, c].set_ylabel("density (1 = flat)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_scores(summaries: dict, tags: list[str], path: pathlib.Path) -> None:
    """Per model, with CIs: CRPS ratio (1 = the solver), energy distance at half
    size against the split-half floor, and the spread ratio; AE as a marker."""
    first = summaries[tags[0]]
    panels = (
        ("crps_ratio", 1.0),
        ("energy_distance_rel_half", None),
        ("spread_ratio_total", 1.0),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    labels = [*tags, "AE"]
    for ax, (key, target) in zip(axes, panels):
        cis = [_ci(summaries[t], "generated", key) for t in tags]
        cis.append(_ci(first, "ae_recon", key))
        x = np.arange(len(labels))
        mean = np.array([c["mean"] for c in cis])
        err = np.array(
            [
                [c["mean"] - c["lo"] for c in cis],
                [c["hi"] - c["mean"] for c in cis],
            ]
        )
        ax.errorbar(x, mean, yerr=np.nan_to_num(err), fmt="o", capsize=3)
        if target is not None:
            ax.axhline(target, color="k", lw=0.8)
        else:
            floor = _ci(first, "real", key)
            ax.axhline(
                floor["mean"], color="k", ls="--", lw=1, label="split-half floor"
            )
            std = first["aggregate"]["real"].get(f"{key}_std")
            if std is not None:
                ax.axhspan(
                    floor["mean"] - std["mean"],
                    floor["mean"] + std["mean"],
                    color="k",
                    alpha=0.1,
                )
            ax.legend(fontsize=7, frameon=False)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# Metrics shown in paired_differences.png.
PAIRED_SHOWN = (
    "spread_ratio_total",
    "crps_ratio",
    "energy_distance_rel",
    "bias_standardised",
    "rank_outer",
    "diversity_ratio",
)


def _plot_paired(paired: list[dict], path: pathlib.Path) -> None:
    """Per-geometry paired differences model - reference, mean and CI."""
    models = list(dict.fromkeys(r["model"] for r in paired))
    fig, axes = plt.subplots(
        1, len(PAIRED_SHOWN), figsize=(2.8 * len(PAIRED_SHOWN), 3.4), squeeze=False
    )
    for ax, key in zip(axes[0], PAIRED_SHOWN):
        rows = [
            next(r for r in paired if r["model"] == m and r["metric"] == key)
            for m in models
        ]
        mean = np.array([r["mean"] for r in rows])
        err = np.array(
            [[r["mean"] - r["lo"] for r in rows], [r["hi"] - r["mean"] for r in rows]]
        )
        ax.errorbar(
            np.arange(len(models)), mean, yerr=np.nan_to_num(err), fmt="o", capsize=3
        )
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(models)), models, rotation=30, ha="right")
        ax.set_title(key, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"paired difference vs {paired[0]['reference']} (95 % CI over geometries)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs", config_name="surrogate/eval"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
