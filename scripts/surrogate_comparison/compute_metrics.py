"""Score the rollouts of a surrogate comparison and write `metrics.yaml` and `metrics.nc`.

    python scripts/surrogate_comparison/compute_metrics.py <run dir> [key=value ...]

The run dir is what scripts/surrogate_comparison/run_models.py wrote: `config.yaml`,
`models.yaml` and `rollouts/<label>/sample_<NNNN>.nc`. The overrides are
applied to its config.yaml (e.g. `metrics.max_pdf_values=50000`). The truth and
the fluid mask are read from the corpus again, with the first model's training
config, `data_dir` and `split`.

Sources: every model, `persistence` (the start frame repeated) and, for the
window statistics, `truth_halves` (the first half of the truth's window against
the second half: a rough sampling-noise floor of a time-averaged statistic,
about sqrt(2) above that of the full window, meaningful only for long rollouts). Everything is on the fluid cells, in m/s,
per compared trajectory and then averaged over them (plain mean: one unstable
sample makes a model's mean NaN).

Per lead step (metrics.nc, dims source x lead, truncated to the shortest rollout):
  rmse, rmse_<var>  RMSE over the fluid cells, all components and per component
  acc               anomaly correlation of prediction and truth about the truth's
                    time mean over the window (per cell), pooled over fluid cells
                    and components
  std_ratio_<var>   spatial std of the prediction / the truth's (energy retention)
  divergence_rms    RMS of du/dx + dv/dy + dw/dz, central differences on the cells
                    whose six neighbours are fluid; `truth` is the reference
  tke, tke_<var>    resolved TKE averaged over the fluid cells, 0.5 sum_c (u_c -
                    mean u_c)^2, and per component 0.5 (u_c - mean u_c)^2, with
                    each source's own time mean over its rollout (per cell, over
                    the finite frames): a smoothing model loses it, a diverging
                    one grows it; `truth` is the reference, persistence is 0

Window statistics (over the rollout's lead steps; NaN for an unstable rollout):
  mean_u_rmse, tke_rmse   RMSE of the time-mean u and resolved-TKE fields
  profile_rmse_{mean_u,tke,uw}
                          RMSE over z of the fluid-averaged profiles of mean u,
                          resolved TKE and -<u'w'>; the profiles go to metrics.nc
                          (profile_<name>, dims source x z)
  spectra_db_<var>_<band> spanwise (y) spectrum energy ratio to the truth in dB
                          for u' and w' in the large (>= 8 dy), mid (4-8 dy) and
                          near_cutoff (2-4 dy) wavelength bands, on the fully fluid
                          lines of both height groups of
                          evaluation.turbulence.spanwise_spectra (summed); the
                          first sample's spectra at `figures.spectra_levels` go to
                          metrics.nc (spectrum_<var>, dims source x level x k,
                          k in cycles/m)
  wasserstein_<var>       1-D Wasserstein-2 distance between the velocity values of
                          prediction and truth (`metrics.max_pdf_values` random
                          fluid values per sample); the PDFs of the values pooled
                          over the stable samples go to metrics.nc (pdf, dims
                          source x var x bin, with pdf_bin_centre)

Outputs, in the run dir:
  metrics.yaml  per source: every scalar as {mean, per_sample: {sample: value}}
                (rmse over all leads, rmse_vs_persistence, final_rmse,
                rmse_step<k> for `metrics.report_leads` inside every rollout,
                acc_horizon_steps / acc_horizon_seconds: the first lead where ACC
                drops below `metrics.acc_threshold`, null if it never does (and the
                mean null if any sample is), stable: the rollout is finite, and the
                statistics above), plus seconds_per_step and n_parameters from
                models.yaml
  metrics.nc    the curves, profiles, spectra and PDFs above; coordinate `source`
                (models, persistence, truth), `lead` with `lead_time` in seconds;
                NaN where a source has no value
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import xarray
import yaml
from evaluation.scores import N_BINS, shared_histograms, wasserstein2
from evaluation.turbulence import (
    SPECTRAL_BANDS,
    band_energy_ratio,
    spanwise_spectra,
    statistic_rmse,
)
from neural_surrogate_baselines import diagnostics
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import grid_spacing, load_states, open_dataset  # noqa: E402

PERSISTENCE, TRUTH, HALVES = "persistence", "truth", "truth_halves"
VELOCITY = ("u", "v", "w")
SPECTRA_VARS = ("u", "w")
PROFILES = ("mean_u", "tke", "uw")


def run(run_dir: pathlib.Path, overrides: list[str] | None = None) -> None:
    run_dir = pathlib.Path(run_dir)
    cfg = OmegaConf.merge(
        OmegaConf.load(run_dir / "config.yaml"), OmegaConf.from_dotlist(overrides or [])
    )
    assert isinstance(cfg, DictConfig)
    models = yaml.safe_load((run_dir / "models.yaml").read_text())
    labels = list(models)
    train_cfg = OmegaConf.load(
        pathlib.Path(models[labels[0]]["model_dir"]) / "config.yaml"
    )
    assert isinstance(train_cfg, DictConfig)
    data = open_dataset(train_cfg, cfg.data_dir, cfg.split, sdf_features="none")
    samples = [int(s) for s in cfg.sample_indices]  # set by run_models.py
    pdf_share = int(cfg.metrics.max_pdf_values) // len(samples)

    scores = [
        _score_sample(run_dir, labels, data, s, cfg, first=i == 0, pdf_share=pdf_share)
        for i, s in enumerate(samples)
    ]
    metrics = _scalars(scores, samples, labels, models, cfg)
    (run_dir / "metrics.yaml").write_text(yaml.safe_dump(metrics, sort_keys=False))
    _dataset(scores, labels).to_netcdf(run_dir / "metrics.nc")
    print(f"Saved metrics.yaml and metrics.nc in {run_dir}")


def _score_sample(
    run_dir: pathlib.Path,
    labels: list[str],
    data: Any,
    sample: int,
    cfg: DictConfig,
    first: bool,
    pdf_share: int,
) -> dict[str, Any]:
    """Curves, statistics and PDF values of every source on one trajectory.

    Holds the truth and one model's rollout at a time.
    """
    files = {
        label: run_dir / "rollouts" / label / f"sample_{sample:04d}.nc"
        for label in labels
    }
    with xarray.open_dataset(files[labels[0]]) as ds:
        start, z = int(ds.attrs["start"]), ds["z"].values
    with xarray.open_dataset(data._state_files[sample]) as ds:
        corpus_times = ds["time"].values
    n = 0
    for path in files.values():
        with xarray.open_dataset(path) as ds:
            own = corpus_times[start + 1 : start + 1 + ds.sizes["time"]]
            if (
                (int(ds.attrs["sample"]), int(ds.attrs["start"])) != (sample, start)
                or ds.attrs["split"] != str(cfg.split)
                or not np.array_equal(ds["time"].values, own)
            ):
                raise ValueError(
                    f"{path} is not trajectory {sample} of split {cfg.split!r} "
                    f"from frame {start}: rerun run_models.py"
                )
            n = max(n, ds.sizes["time"])
    window = load_states(data, sample, range(start, start + n + 1))
    truth = window[1:]
    times = corpus_times[start : start + n + 1].astype(float)
    fluid = data.geometry_for(sample).numpy().astype(bool)
    state_vars = list(data.state_vars)
    spacing = grid_spacing(data, sample)
    ctx = {
        "fluid": fluid,
        "vars": state_vars,
        "dy": spacing[1],
        "seed": sample,
        "levels": (
            [lv for lv in cfg.figures.spectra_levels if 0 <= lv < len(fluid)]
            if first
            else []
        ),
        "max_values": int(cfg.metrics.max_pdf_values),
    }

    truth_stats = _statistics(truth, ctx)
    half = len(truth) // 2
    out: dict[str, Any] = {
        "sample": sample,
        "times": times - times[0],
        "z": z,
        "curves": {
            TRUTH: {
                "divergence_rms": _divergence_rms(truth, ctx, spacing),
                **_tke(_fluid_values(truth, fluid), state_vars),
            }
        },
        "stats": {
            HALVES: _compare(
                _statistics(truth[:half], ctx), _statistics(truth[half:], ctx), ctx
            )
        },
        "stable": {},
        "profiles": {TRUTH: truth_stats["profiles"]},
        "spectra": {TRUTH: truth_stats["level_spectra"]},
        "values": {TRUTH: truth_stats["values"][:, :pdf_share]},
        "levels": ctx["levels"],
        "k": truth_stats["k"],
    }
    for name in [PERSISTENCE, *labels]:
        pred = (
            np.broadcast_to(window[0], truth.shape)
            if name == PERSISTENCE
            else _read_rollout(files[name], state_vars)
        )
        own = truth[: len(pred)]
        stable = bool(np.isfinite(pred).all())
        out["stable"][name] = stable
        out["curves"][name] = _curves(pred, own, ctx, spacing)
        stats = _statistics(pred, ctx) if stable else None
        reference = truth_stats if len(pred) == len(truth) else _statistics(own, ctx)
        out["stats"][name] = _compare(stats, reference, ctx)
        if stats is not None:
            out["profiles"][name] = stats["profiles"]
            out["spectra"][name] = stats["level_spectra"]
            out["values"][name] = stats["values"][:, :pdf_share]
        del pred
    return out


def _read_rollout(path: pathlib.Path, state_vars: list[str]) -> np.ndarray:
    with xarray.open_dataset(path) as ds:
        return np.stack([ds[v].values for v in state_vars], axis=1).astype(np.float32)


def _curves(
    pred: np.ndarray,
    truth: np.ndarray,
    ctx: dict[str, Any],
    spacing: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    fluid, state_vars = ctx["fluid"], ctx["vars"]
    with np.errstate(all="ignore"):  # an unstable rollout gives inf / NaN curves
        p, t = _fluid_values(pred, fluid), _fluid_values(truth, fluid)
        f8 = np.float64  # float32 sums over ~1e8 values lose digits
        mse = ((p - t) ** 2).mean(axis=-1, dtype=f8)
        anomaly_t = t - t.mean(axis=0, dtype=f8).astype(t.dtype)
        anomaly_p = p - t.mean(axis=0, dtype=f8).astype(t.dtype)
        acc = (anomaly_p * anomaly_t).sum(axis=(1, 2), dtype=f8) / np.sqrt(
            (anomaly_p**2).sum(axis=(1, 2), dtype=f8)
            * (anomaly_t**2).sum(axis=(1, 2), dtype=f8)
        )
        std_ratio = p.std(axis=-1, dtype=f8) / t.std(axis=-1, dtype=f8)
        curves = {"rmse": np.sqrt(mse.mean(axis=1)), "acc": acc}
        for c, var in enumerate(state_vars):
            curves[f"rmse_{var}"] = np.sqrt(mse[:, c])
            curves[f"std_ratio_{var}"] = std_ratio[:, c]
        curves["divergence_rms"] = _divergence_rms(pred, ctx, spacing)
        curves |= _tke(p, state_vars)
    return curves


def _fluid_values(fields: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """(T, C, n_fluid) values on the fluid cells; `take` on the flat cells is
    much faster than a mask."""
    return np.take(fields.reshape(*fields.shape[:2], -1), np.flatnonzero(fluid), -1)


def _tke(values: np.ndarray, state_vars: list[str]) -> dict[str, np.ndarray]:
    """(T,) resolved TKE, total and per component, averaged over the fluid cells
    of (T, C, n_fluid) `values`, about their own time mean over the finite frames."""
    velocity = [state_vars.index(v) for v in VELOCITY]
    v = values[:, velocity]
    finite = np.isfinite(v).all(axis=(1, 2))
    with np.errstate(all="ignore"):
        mean = v[finite].mean(axis=0, dtype=np.float64).astype(v.dtype)
        energy = 0.5 * ((v - mean) ** 2).mean(axis=-1, dtype=np.float64)  # (T, 3)
    return {"tke": energy.sum(axis=1)} | {
        f"tke_{var}": energy[:, c] for c, var in enumerate(VELOCITY)
    }


def _divergence_rms(
    fields: np.ndarray, ctx: dict[str, Any], spacing: tuple[float, float, float]
) -> np.ndarray:
    """(T,) RMS of the cell-centred divergence on cells with six fluid neighbours."""
    fluid, state_vars = ctx["fluid"], ctx["vars"]
    inner = (slice(1, -1),) * 3
    interior = fluid[inner].copy()
    for axis in range(3):
        for shift in (-1, 1):
            interior &= np.roll(fluid, shift, axis=axis)[inner]
    divergence = np.zeros((len(fields), *interior.shape), dtype=np.float64)
    for var, axis, h in zip(("w", "v", "u"), (0, 1, 2), spacing):
        component = fields[:, state_vars.index(var)]
        ahead = [slice(None)] + [slice(1, -1)] * 3
        behind = list(ahead)
        ahead[axis + 1], behind[axis + 1] = slice(2, None), slice(None, -2)
        divergence += (component[tuple(ahead)] - component[tuple(behind)]) / (2 * h)
    result: np.ndarray = np.sqrt((divergence[:, interior] ** 2).mean(axis=-1))
    return result


def _statistics(fields: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    """Time-averaged statistics of one window: the inputs of :func:`_compare`."""
    fluid, state_vars = ctx["fluid"], ctx["vars"]
    u, w = state_vars.index("u"), state_vars.index("w")
    velocity = [state_vars.index(v) for v in VELOCITY]
    spectra, level_spectra = {}, {}
    for var in SPECTRA_VARS:
        k, groups = spanwise_spectra(fields[:, state_vars.index(var)], fluid, ctx["dy"])
        spectra[var] = sum(s for s in groups.values() if s is not None)
        level_spectra[var] = [
            _level_spectrum(fields[:, state_vars.index(var)], fluid, lv, ctx["dy"])
            for lv in ctx["levels"]
        ]
    # The same random fluid (frame, cell) draws for every source of a sample.
    rng = np.random.default_rng(ctx["seed"])
    frames = rng.integers(len(fields), size=ctx["max_values"])
    cells = np.flatnonzero(fluid)[rng.integers(fluid.sum(), size=ctx["max_values"])]
    flat = fields.reshape(*fields.shape[:2], -1)
    values = flat[frames, :, cells][:, velocity].T  # (3, max_values)
    return {
        "mean_u": fields[:, u].mean(axis=0),
        "tke": 0.5 * fields[:, velocity].var(axis=0).sum(axis=0),
        "profiles": diagnostics.profiles(fields, fluid, u, w, velocity),
        "k": k,
        "spectra": spectra,
        "level_spectra": level_spectra,
        "values": values,
    }


def _level_spectrum(
    field: np.ndarray, fluid: np.ndarray, level: int, dy: float
) -> np.ndarray:
    """spanwise_spectra of one level: every other level masked as solid.

    Same fully-fluid lines as the grouped spectra; only the start of the x range
    (the first column with a solid cell) is then taken from this level alone.
    """
    alone = np.zeros_like(fluid)
    alone[level] = fluid[level]
    _, groups = spanwise_spectra(field, alone, dy)
    spectrum = next((s for s in groups.values() if s is not None), None)
    return np.full(field.shape[-2] // 2, np.nan) if spectrum is None else spectrum


def _compare(
    pred: dict[str, Any] | None, truth: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, float]:
    """The scalar statistics of `pred` against `truth` (NaN when `pred` is None)."""
    keys = (
        ["mean_u_rmse", "tke_rmse"]
        + [f"profile_rmse_{p}" for p in PROFILES]
        + [f"spectra_db_{v}_{b}" for v in SPECTRA_VARS for b in SPECTRAL_BANDS]
        + [f"wasserstein_{v}" for v in VELOCITY]
    )
    if pred is None:
        return dict.fromkeys(keys, float("nan"))
    fluid = ctx["fluid"]
    out = {
        "mean_u_rmse": statistic_rmse(pred["mean_u"], truth["mean_u"], fluid)[0],
        "tke_rmse": statistic_rmse(pred["tke"], truth["tke"], fluid)[0],
    }
    for p in PROFILES:
        error = pred["profiles"][p] - truth["profiles"][p]
        out[f"profile_rmse_{p}"] = float(np.sqrt(np.mean(error**2)))
    for var in SPECTRA_VARS:
        ratios = band_energy_ratio(
            truth["k"], pred["spectra"][var], truth["spectra"][var], ctx["dy"]
        )
        out |= {f"spectra_db_{var}_{b}": float(r) for b, r in ratios.items()}
    for c, var in enumerate(VELOCITY):
        out[f"wasserstein_{var}"] = wasserstein2(truth["values"][c], pred["values"][c])[
            "w2"
        ]
    return out


def _scalars(
    scores: list[dict[str, Any]],
    samples: list[int],
    labels: list[str],
    models: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """metrics.yaml: per source, each scalar's mean and per-sample values."""
    shortest = min(
        len(s["curves"][name]["rmse"])
        for s in scores
        for name in [*labels, PERSISTENCE]
    )
    threshold = float(cfg.metrics.acc_threshold)
    per_sample: dict[str, dict[str, list[Any]]] = {}
    for score in scores:
        persistence_rmse = _total(score["curves"][PERSISTENCE]["rmse"])
        for name in [*labels, PERSISTENCE]:
            curves = score["curves"][name]
            below = np.flatnonzero(~(curves["acc"] >= threshold))
            row: dict[str, Any] = {
                "rmse": _total(curves["rmse"]),
                "rmse_vs_persistence": _total(curves["rmse"]) / persistence_rmse,
                "final_rmse": float(curves["rmse"][-1]),
            }
            row |= {
                f"rmse_step{k}": float(curves["rmse"][k - 1])
                for k in cfg.metrics.report_leads
                if 1 <= k <= shortest
            }
            row["acc_horizon_steps"] = int(below[0]) + 1 if below.size else None
            row["acc_horizon_seconds"] = (
                float(score["times"][below[0] + 1]) if below.size else None
            )
            row["stable"] = score["stable"][name]
            row |= score["stats"][name]
            for key, value in row.items():
                per_sample.setdefault(name, {}).setdefault(key, []).append(value)
        for key, value in score["stats"][HALVES].items():
            per_sample.setdefault(HALVES, {}).setdefault(key, []).append(value)

    metrics: dict[str, Any] = {}
    for name, rows in per_sample.items():
        entry: dict[str, Any] = {}
        for key, values in rows.items():
            entry[key] = {
                "mean": _mean(values),
                "per_sample": dict(zip(samples, (_native(v) for v in values))),
            }
        if name in models:
            entry["seconds_per_step"] = models[name].get("seconds_per_step")
            entry["n_parameters"] = models[name].get("n_parameters")
        metrics[name] = entry
    return metrics


def _total(curve: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float(np.sqrt(np.mean(np.asarray(curve, dtype=float) ** 2)))


def _mean(values: list[Any]) -> Any:
    if any(v is None for v in values):
        return None
    if isinstance(values[0], bool):
        return all(values)
    return float(np.mean(values))


def _native(value: Any) -> Any:
    return float(value) if isinstance(value, (float, np.floating)) else value


def _dataset(scores: list[dict[str, Any]], labels: list[str]) -> xarray.Dataset:
    """metrics.nc: curves (source x lead), profiles, spectra and PDFs."""
    sources = [*labels, PERSISTENCE, TRUTH]
    n = min(len(c["divergence_rms"]) for s in scores for c in s["curves"].values())
    curve_names = list(scores[0]["curves"][labels[0]])
    data_vars: dict[str, Any] = {}
    for key in curve_names:
        table = np.full((len(sources), n), np.nan)
        for i, name in enumerate(sources):
            per = [
                s["curves"][name][key][:n] for s in scores if key in s["curves"][name]
            ]
            if per:
                with np.errstate(all="ignore"):
                    table[i] = np.mean(per, axis=0)
        data_vars[key] = (("source", "lead"), table)

    z = scores[0]["z"]
    if any(len(s["z"]) != len(z) for s in scores):
        raise ValueError(
            "compared trajectories differ in nz: profiles cannot be averaged"
        )
    for p in PROFILES:
        table = np.full((len(sources), len(z)), np.nan)
        for i, name in enumerate(sources):
            per = [s["profiles"][name][p] for s in scores if name in s["profiles"]]
            if len(per) == len(scores):  # an unstable sample leaves it NaN
                table[i] = np.mean(per, axis=0)
        data_vars[f"profile_{p}"] = (("source", "z"), table)

    coords: dict[str, Any] = {
        "source": sources,
        "lead": np.arange(1, n + 1),
        "lead_time": ("lead", scores[0]["times"][1 : n + 1]),
        "z": z,
    }
    first = scores[0]
    if first["levels"]:
        coords |= {"level": first["levels"], "k": first["k"]}
        for var in SPECTRA_VARS:
            table = np.full(
                (len(sources), len(first["levels"]), len(first["k"])), np.nan
            )
            for i, name in enumerate(sources):
                if name in first["spectra"]:
                    table[i] = np.stack(first["spectra"][name][var])
            data_vars[f"spectrum_{var}"] = (("source", "level", "k"), table)
    pooled = {
        name: np.concatenate(
            [s["values"][name] for s in scores if name in s["values"]], 1
        )
        for name in sources
        if any(name in s["values"] for s in scores)
    }
    density = np.full((len(sources), len(VELOCITY), N_BINS), np.nan)
    centres = []
    for c, _ in enumerate(VELOCITY):
        edges, densities, _ = shared_histograms({k: v[c] for k, v in pooled.items()})
        centres.append(0.5 * (edges[1:] + edges[:-1]))
        for i, name in enumerate(sources):
            if name in densities:
                density[i, c] = densities[name]
    data_vars["pdf"] = (("source", "var", "bin"), density)
    data_vars["pdf_bin_centre"] = (("var", "bin"), np.array(centres))
    coords["var"] = list(VELOCITY)
    return xarray.Dataset(data_vars, coords=coords)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "overrides", nargs="*", help="config overrides, e.g. metrics.acc_threshold=0.6"
    )
    args = parser.parse_args()
    run(args.run_dir, args.overrides)


if __name__ == "__main__":
    main()
