"""Compare the models of a forward-model comparison against their own noise.

    python scripts/comparison/compute_metrics.py <run dir> [key=value ...]

The run dir is what scripts/comparison/run_models.py wrote: `config.yaml`,
`models.yaml` and `fields/<label>/replica_<r>.nc`. The overrides are applied to
its config.yaml (e.g. `metrics.spectrum_segment=32`).

There is no truth. Every metric is a distance D between two runs, taken for
every model pair (a, b) over all replica combinations (a_i, b_j), and for every
model with itself over its replica pairs (a_i, a_j), i < j: the noise floor,
how far two realisations of one model are apart. The ratio of a pair is its
distance over the noise it would show if the two models were the same flow,
sqrt((noise_a^2 + noise_b^2) / 2) (one model's noise if only it has replicas):
about 1 when the models differ by realisation noise only, well above 1 when
they differ systematically. Every statistic is on the case grid's fluid cells
(`fluid` of the fields files: the STL's building cells and their neighbours
left out, the same for every model), in m/s units.

Metrics (`metric`, `quantity`):
  field     time-mean statistics, (z, y, x) per run: mean_u, mean_v, mean_w,
            speed (the time mean of |U|), tke (0.5 trace of the resolved
            stresses), and the resolved stresses uu, vv, ww, uv, uw, vw
            (<u_i' u_j'>, time covariance, ddof 1). D: RMSE over the cells, and
            the bias, the mean of a - b; per region: `all` the fluid cells, and
            each of `metrics.height_bands` (fractions [lo, hi) of the canopy
            height H, from z of the cell centre)
  profile   horizontally averaged profiles over the fluid cells of each level:
            u, v (mean), tke, uw. D: RMSE over the levels, and the bias
  sensor_mean, sensor_std
            the time mean and std of u, v, w and speed at each sensor of the
            case's obs (`assimilation`) and validation points (`validation`;
            `all`: both). D: RMSE over the sensors, and the bias
  sensor_w2 the 1-D Wasserstein-2 distance between the value distributions at
            each sensor, averaged over the sensors
  sensor_lsd
            the log-spectral distance in dB between the Welch frequency spectra
            at each sensor (`metrics.spectrum_segment` samples per segment,
            capped by the shortest run), averaged over the sensors; skipped
            when a run has fewer than 8 frames

Outputs, in the run dir:
  metrics.csv   one row per metric, quantity, region and pair: model_a, model_b
                (the same label: the noise floor), value (D), bias (a - b, for
                RMSE metrics), noise (the pair's combined noise) and ratio
  metrics.json  the same rows, plus `canopy_height`, `height_bands` (m) and the
                replicas per model
  metrics.nc    what the figures read: `stat` (label, quantity, z, y, x), the
                replica mean of the field statistics; `profile` (label, replica,
                profile_quantity, z); `spectrum` (label, replica, sensor,
                sensor_quantity, frequency); `fluid` and `solid` (z, y, x); NaN where a model
                has fewer replicas
  summary.md    the headline ratios and noise floors
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from typing import Any, Callable

import numpy as np
import xarray
import yaml
from evaluation.scores import wasserstein2
from evaluation.turbulence import (
    intrinsic_profile,
    log_spectral_distance,
    welch_spectrum,
)
from omegaconf import DictConfig, OmegaConf

FIELD_QUANTITIES = (
    "mean_u",
    "mean_v",
    "mean_w",
    "speed",
    "tke",
    "uu",
    "vv",
    "ww",
    "uv",
    "uw",
    "vw",
)
PROFILE_QUANTITIES = {"u": "mean_u", "v": "mean_v", "tke": "tke", "uw": "uw"}
SENSOR_QUANTITIES = ("u", "v", "w", "speed")
STRESSES = {"uu": (0, 0), "vv": (1, 1), "ww": (2, 2), "uv": (0, 1), "uw": (0, 2)}
STRESSES["vw"] = (1, 2)
SENSOR_COORDS = ("sensor_set", "sensor_x", "sensor_y", "sensor_z")
# Fewest frames for a sensor spectrum (two segments of four samples).
MIN_SPECTRUM_FRAMES = 8
COLUMNS = (
    "metric",
    "quantity",
    "region",
    "model_a",
    "model_b",
    "value",
    "bias",
    "noise",
    "ratio",
)


def run(run_dir: pathlib.Path, overrides: list[str] | None = None) -> None:
    run_dir = pathlib.Path(run_dir)
    cfg = OmegaConf.merge(
        OmegaConf.load(run_dir / "config.yaml"), OmegaConf.from_dotlist(overrides or [])
    )
    assert isinstance(cfg, DictConfig)
    models = yaml.safe_load((run_dir / "models.yaml").read_text())
    files = {
        label: [
            run_dir / "fields" / label / f"replica_{r}.nc"
            for r in range(len(entry["replicas"]))
        ]
        for label, entry in models.items()
    }
    with xarray.open_dataset(files[next(iter(files))][0]) as first:
        grid = {d: first[d].values for d in ("z", "y", "x")}
        fluid = first["fluid"].values.astype(bool)
        solid = first["solid"].values.astype(bool)
        sensors = {k: first[k].values for k in SENSOR_COORDS}
    height = canopy_height(cfg, solid, grid["z"])
    bands = height_bands(cfg, height)
    regions = {"all": fluid} | {
        name: fluid & ((grid["z"] >= lo) & (grid["z"] < hi))[:, None, None]
        for name, (lo, hi) in bands.items()
    }

    stats: dict[str, list[dict[str, np.ndarray]]] = {}
    series: dict[str, list[np.ndarray]] = {}
    times: dict[str, list[np.ndarray]] = {}
    for label, paths in files.items():
        stats[label], series[label], times[label] = [], [], []
        for path in paths:
            with xarray.open_dataset(path) as ds:
                if not np.array_equal(ds["fluid"].values.astype(bool), fluid):
                    raise ValueError(f"{path} is on another grid or mask")
                stats[label].append(field_statistics(ds))
                series[label].append(sensor_values(ds))
                times[label].append(ds["time"].values)

    profiles = {
        label: [
            {
                q: intrinsic_profile(s[stat], fluid)
                for q, stat in PROFILE_QUANTITIES.items()
            }
            for s in runs
        ]
        for label, runs in stats.items()
    }
    shortest = min(len(t) for runs in times.values() for t in runs)
    segment = min(int(cfg.metrics.spectrum_segment), shortest)
    spectra: dict[str, list[np.ndarray]] | None = None
    frequency: np.ndarray | None = None
    if shortest >= MIN_SPECTRUM_FRAMES:
        step = float(np.median(np.diff(times[next(iter(times))][0])))
        spectra = {}
        for label, runs in series.items():
            spectra[label] = []
            for values in runs:  # (quantity, time, sensor) -> (sensor, quantity, f)
                frequency, psd = welch_spectrum(
                    values.transpose(2, 0, 1), 1.0 / step, segment
                )
                spectra[label].append(psd)

    rows: list[dict[str, Any]] = []
    for q in FIELD_QUANTITIES:
        for region, mask in regions.items():
            items = {lb: [s[q][mask] for s in runs] for lb, runs in stats.items()}
            rows += pair_rows("field", q, region, items, squared_error)
    for q in PROFILE_QUANTITIES:
        items = {lb: [p[q] for p in runs] for lb, runs in profiles.items()}
        rows += pair_rows("profile", q, "all", items, squared_error)
    sets = sensors["sensor_set"]
    for region in ["all", *dict.fromkeys(sets)]:
        chosen = np.ones(sets.size, bool) if region == "all" else sets == region
        for c, q in enumerate(SENSOR_QUANTITIES):
            values = {
                lb: [v[c][:, chosen] for v in runs] for lb, runs in series.items()
            }
            for metric, reduce in (("sensor_mean", np.mean), ("sensor_std", np.std)):
                items = {
                    lb: [reduce(v, axis=0) for v in runs] for lb, runs in values.items()
                }
                rows += pair_rows(metric, q, region, items, squared_error)
            rows += pair_rows("sensor_w2", q, region, values, sensor_w2)
            if spectra is not None:
                items = {
                    lb: [psd[chosen, c] for psd in runs] for lb, runs in spectra.items()
                }
                rows += pair_rows("sensor_lsd", q, region, items, spectral_distance)

    with open(run_dir / "metrics.csv", "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(_cell(row[c]) for c in COLUMNS) + "\n")
    meta = {
        "canopy_height": height,
        "height_bands": {k: list(v) for k, v in bands.items()},
        "replicas": {label: len(paths) for label, paths in files.items()},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(meta | {"rows": [_native(r) for r in rows]}, indent=1)
    )
    masks = {"fluid": fluid, "solid": solid}
    _dataset(stats, profiles, spectra, frequency, grid, masks, sensors, meta).to_netcdf(
        run_dir / "metrics.nc"
    )
    (run_dir / "summary.md").write_text(summary(rows, models, meta))
    print(f"Saved metrics.csv, metrics.json, metrics.nc and summary.md in {run_dir}")


# ---------------------------------------------------------------------------
# Statistics of one run
# ---------------------------------------------------------------------------


def field_statistics(ds: xarray.Dataset) -> dict[str, np.ndarray]:
    """Time-mean statistics (FIELD_QUANTITIES) of a fields file, each (z, y, x)."""
    velocity = np.stack([ds[c].values for c in ("u", "v", "w")]).astype(np.float64)
    mean = velocity.mean(axis=1)
    anomaly = velocity - mean[:, None]
    n = velocity.shape[1]
    out = {
        "mean_u": mean[0],
        "mean_v": mean[1],
        "mean_w": mean[2],
        "speed": np.sqrt((velocity**2).sum(axis=0)).mean(axis=0),
    }
    for name, (i, j) in STRESSES.items():
        out[name] = (anomaly[i] * anomaly[j]).sum(axis=0) / max(n - 1, 1)
    out["tke"] = 0.5 * (out["uu"] + out["vv"] + out["ww"])
    return out


def sensor_values(ds: xarray.Dataset) -> np.ndarray:
    """(SENSOR_QUANTITIES, time, sensor) at the sensors of a fields file."""
    u, v, w = (ds[f"sensor_{c}"].values.astype(np.float64) for c in ("u", "v", "w"))
    return np.stack([u, v, w, np.sqrt(u**2 + v**2 + w**2)])


def canopy_height(cfg: DictConfig, solid: np.ndarray, z: np.ndarray) -> float:
    """`metrics.canopy_height`, or the top face of the highest building cell
    (on a level that is not entirely solid, such as a ground plane)."""
    if cfg.metrics.get("canopy_height") is not None:
        return float(cfg.metrics.canopy_height)
    dz = float(np.diff(z).mean()) if z.size > 1 else 1.0
    built = solid.any(axis=(1, 2)) & ~solid.all(axis=(1, 2))
    if not built.any():
        raise ValueError("no building on the case grid: set metrics.canopy_height")
    return float(z[np.flatnonzero(built)[-1]] + 0.5 * dz)


def height_bands(cfg: DictConfig, height: float) -> dict[str, tuple[float, float]]:
    """`metrics.height_bands` in metres: {name: (lo, hi)}, open ends infinite."""
    out = {}
    for name, (lo, hi) in cfg.metrics.height_bands.items():
        out[str(name)] = (
            -np.inf if lo is None else float(lo) * height,
            np.inf if hi is None else float(hi) * height,
        )
    return out


# ---------------------------------------------------------------------------
# Distances, pairs and noise
# ---------------------------------------------------------------------------


def squared_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean squared difference, mean difference a - b) over the finite values."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if not diff.size:
        return np.nan, np.nan
    return float(np.mean(diff**2)), float(np.mean(diff))


def sensor_w2(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean over the sensors (last axis) of the W2 distance of the time series."""
    w2 = [wasserstein2(a[:, s], b[:, s])["w2"] for s in range(a.shape[1])]
    return float(np.nanmean(w2)) if np.isfinite(w2).any() else np.nan, np.nan


def spectral_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean over the sensors (first axis) of the log-spectral distance in dB."""
    lsd = log_spectral_distance(a, b)
    return float(np.nanmean(lsd)) if np.isfinite(lsd).any() else np.nan, np.nan


def pair_rows(
    metric: str,
    quantity: str,
    region: str,
    items: dict[str, list[np.ndarray]],
    distance: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
) -> list[dict[str, Any]]:
    """The rows of one metric: every model with itself (the noise), then every pair.

    `items` holds each model's per-replica values. `squared_error` distances
    are averaged as squares and reported as their root (an RMSE); the others
    are averaged as they are.
    """
    rms = distance is squared_error

    def combine(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
        if not pairs:
            return np.nan, np.nan
        values = np.array([distance(a, b) for a, b in pairs], dtype=float)
        with np.errstate(invalid="ignore"):
            d = np.mean(values[:, 0])
            return (float(np.sqrt(d)) if rms else float(d)), float(
                np.mean(values[:, 1])
            )

    noise = {
        label: combine(list(itertools.combinations(runs, 2)))[0]
        for label, runs in items.items()
    }
    rows = []
    for label in items:
        rows.append(
            _row(metric, quantity, region, label, label, noise[label], np.nan)
            | {"noise": noise[label]}
        )
    for a, b in itertools.combinations(items, 2):
        value, bias = combine(list(itertools.product(items[a], items[b])))
        floor = [n for n in (noise[a], noise[b]) if np.isfinite(n)]
        pair_noise = float(np.sqrt(np.mean(np.square(floor)))) if floor else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = value / pair_noise if pair_noise > 0 else np.nan
        rows.append(
            _row(metric, quantity, region, a, b, value, bias if rms else np.nan)
            | {"noise": pair_noise, "ratio": ratio}
        )
    return rows


def _row(
    metric: str, quantity: str, region: str, a: str, b: str, value: float, bias: float
) -> dict[str, Any]:
    return {
        "metric": metric,
        "quantity": quantity,
        "region": region,
        "model_a": a,
        "model_b": b,
        "value": value,
        "bias": bias,
        "noise": np.nan,
        "ratio": np.nan,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if not np.isfinite(value) else f"{value:.6g}"
    return str(value)


def _native(row: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (None if isinstance(v, float) and not np.isfinite(v) else v)
        for k, v in row.items()
    }


def _dataset(
    stats: dict[str, list[dict[str, np.ndarray]]],
    profiles: dict[str, list[dict[str, np.ndarray]]],
    spectra: dict[str, list[np.ndarray]] | None,
    frequency: np.ndarray | None,
    grid: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    sensors: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> xarray.Dataset:
    labels = list(stats)
    n_rep = max(len(runs) for runs in stats.values())
    stat = np.stack(
        [
            np.stack(
                [np.mean([s[q] for s in stats[lb]], axis=0) for q in FIELD_QUANTITIES]
            )
            for lb in labels
        ]
    )
    profile = np.full(
        (len(labels), n_rep, len(PROFILE_QUANTITIES), grid["z"].size), np.nan
    )
    for i, lb in enumerate(labels):
        for r, p in enumerate(profiles[lb]):
            profile[i, r] = [p[q] for q in PROFILE_QUANTITIES]
    data_vars: dict[str, Any] = {
        "stat": (("label", "quantity", "z", "y", "x"), stat.astype(np.float32)),
        "profile": (("label", "replica", "profile_quantity", "z"), profile),
        **{k: (("z", "y", "x"), m.astype(np.int8)) for k, m in masks.items()},
    }
    coords: dict[str, Any] = {
        "label": labels,
        "replica": np.arange(n_rep),
        "quantity": list(FIELD_QUANTITIES),
        "profile_quantity": list(PROFILE_QUANTITIES),
        **grid,
        **{k: ("sensor", v) for k, v in sensors.items()},
    }
    if spectra is not None and frequency is not None:
        first = spectra[labels[0]][0]
        table = np.full((len(labels), n_rep, *first.shape), np.nan)
        for i, lb in enumerate(labels):
            for r, psd in enumerate(spectra[lb]):
                table[i, r] = psd
        data_vars["spectrum"] = (
            ("label", "replica", "sensor", "sensor_quantity", "frequency"),
            table,
        )
        coords |= {"frequency": frequency, "sensor_quantity": list(SENSOR_QUANTITIES)}
    attrs = {"canopy_height": meta["canopy_height"]}
    attrs |= {
        f"band_{name}": [float(np.clip(v, -1e9, 1e9)) for v in bounds]
        for name, bounds in meta["height_bands"].items()
    }
    return xarray.Dataset(data_vars, coords=coords, attrs=attrs)


# The headline rows of summary.md: (metric, quantity, region).
HEADLINE = (
    ("field", "mean_u", "all"),
    ("field", "speed", "all"),
    ("field", "tke", "all"),
    ("field", "uw", "all"),
    ("profile", "u", "all"),
    ("profile", "tke", "all"),
    ("sensor_mean", "speed", "all"),
    ("sensor_std", "speed", "all"),
    ("sensor_w2", "speed", "all"),
    ("sensor_lsd", "u", "all"),
)


def summary(rows: list[dict[str, Any]], models: dict[str, Any], meta: dict) -> str:
    """summary.md: runs, the noise floors and the pairs' distances and ratios."""
    index = {
        (r["metric"], r["quantity"], r["region"], r["model_a"], r["model_b"]): r
        for r in rows
    }
    labels = list(models)
    pairs = list(itertools.combinations(labels, 2))
    lines = ["# Model comparison", ""]
    lines += [
        "| model | backend | replicas | wall time per run [s] |",
        "|---|---|---|---|",
    ]
    for label, entry in models.items():
        walls = [r["wall_time_s"] for r in entry["replicas"]]
        lines.append(
            f"| {label} | {entry['model']} | {len(walls)} | " f"{np.mean(walls):.0f} |"
        )
    lines += [
        "",
        f"Canopy height H = {meta['canopy_height']:.3g} m. Distances in m/s "
        "(m²/s² for tke and the stresses, dB for sensor_lsd), on the fluid cells. "
        "`noise`: the distance between two replicas of a model; ratio: the "
        "pair's distance over sqrt((noise_a² + noise_b²)/2), about 1 when the "
        "models differ by realisation noise only.",
        "",
    ]
    header = "| metric | quantity | " + " | ".join(f"noise {lb}" for lb in labels)
    header += "".join(f" | {a} vs {b} | ratio" for a, b in pairs) + " |"
    lines += [header, "|" + "---|" * (2 + len(labels) + 2 * len(pairs))]
    for metric, quantity, region in HEADLINE:
        if (metric, quantity, region, labels[0], labels[0]) not in index:
            continue
        cells = [metric, quantity]
        cells += [
            _fmt(index[(metric, quantity, region, lb, lb)]["value"]) for lb in labels
        ]
        for a, b in pairs:
            row = index[(metric, quantity, region, a, b)]
            cells += [_fmt(row["value"]), _fmt(row["ratio"])]
        lines.append("| " + " | ".join(cells) + " |")
    if all(len(e["replicas"]) < 2 for e in models.values()):
        lines += ["", "No model has two replicas: no noise floor, no ratios."]
    return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="config overrides, e.g. metrics.spectrum_segment=32",
    )
    args = parser.parse_args()
    run(args.run_dir, args.overrides)


if __name__ == "__main__":
    main()
