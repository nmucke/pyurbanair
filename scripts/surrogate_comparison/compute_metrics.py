"""Score the rollouts of a surrogate comparison and write `metrics.yaml` and `metrics.nc`.

    python scripts/surrogate_comparison/compute_metrics.py <run dir> [key=value ...]

The run dir is what scripts/surrogate_comparison/run_models.py wrote: `config.yaml`,
`models.yaml` and `rollouts/<label>/sample_<NNNN>.nc`. The overrides are
applied to its config.yaml (e.g. `metrics.max_pdf_values=50000`). The truth and
the fluid mask are read from the corpus again, with the first model's training
config, `data_dir` and `split`. Definitions: scripts/utils/surrogate_metrics.py
(one function per metric) and docs/plans/implemented/surrogate_comparison_paper_metrics.md.

Sources: every model, `persistence` (the start frame repeated), `truth` (the
reference curve where it has one) and, for the window statistics,
`truth_halves` (the first half of the truth's window against the second half;
not a noise floor when the inflow varies in time: the models share that
forcing and can beat it).

Per compared trajectory (sample): the window is lead steps 1..N, N the
shortest rollout of all models and samples; ū is the truth's per-cell time mean
over it, the one reference of every per-lead fluctuation (ACC, spectra, Λ,
PDFs); the window statistics keep each source's own time mean. The reference
scales: H the mean building height, k_c the canopy-top level (the lowest
building-free level, so z_c / H varies by sample; spectra, Λ and PDFs are taken
there) at height z_c, u_ref the plane mean of ū_u at k_c, T the truth's
integral time scale (trapezoid rule). Everything is made dimensionless per
sample (velocities / u_ref, vorticity and divergence x H / u_ref, TKE and
stresses / u_ref^2, lengths / H, spectra / (u_ref^2 H) against κ H, κ = 2π/λ
the angular wavenumber in rad/m), then averaged over the samples (plain mean: one
unstable sample makes a model's mean NaN). An unstable (non-finite) rollout has
NaN curves from its first non-finite frame and NaN window statistics.

Per lead step (metrics.nc, dims source x lead):
  rmse, rmse_<u,v,w>   RMSE over the fluid cells, pooled and per component
  acc, acc_<u,v,w>     anomaly correlation of prediction and truth about ū over
                       the fluid cells, pooled and per component
  rmse_pedestrian, acc_pedestrian
                       the same, pooled, on the fluid cells of level
                       `metrics.pedestrian_level` only
  vorticity_rmse, vorticity_rmse_<x,y,z>
                       RMSE of the vorticity (central differences, on the cells
                       whose six neighbours are fluid), pooled and per component
  divergence_rms       RMS of du/dx + dv/dy + dw/dz on the same cells (truth too)
  tke                  resolved TKE averaged over the fluid cells, 0.5 sum_c
                       (u_c - mean u_c)^2 about the source's own time mean over
                       its finite frames (truth too; persistence is 0)
  integral_length      Λ_y / H at k_c: dy (sum R(Δy) - 1/2) up to the first zero
                       crossing (trapezoid rule), R the spanwise autocorrelation
                       of u - ū with each line's spanwise mean removed, on the
                       fully fluid y-lines (truth too)
  spectrum_<u,w>       spanwise spectrum of u - ū (w - ū_w) at k_c on the fully
                       fluid y-lines, sum E dκ = the variance, binned on shared
                       log-spaced κ H bins and averaged over the samples (the
                       spectral-ratio figure takes the ratio of these means;
                       dims source x lead x kh; truth too)
Also in metrics.nc:
  pdf                  density of the fluctuations about ū at k_c at the
                       `metrics.snapshot_leads` (-1: the last lead), pooled over
                       the samples, on 60 bins per variable over the truth's
                       0.05-99.95 percentile range at those leads, the same for
                       every source (values outside fall out but count in the
                       normalisation) (dims source x pdf_lead x var x bin, with
                       pdf_bin_centre: var x bin; at most
                       `metrics.max_pdf_values` values per source and lead)
  profile_<mean_u,tke,uw>
                       fluid-plane-averaged window profiles of mean u, resolved
                       TKE and -<u'w'> (dims source x z)
  reference_H, reference_u, reference_T, canopy_level
                       the reference scales per sample (dim sample)
  coordinates source, lead, lead_time (s), lead_over_T (t / T̄, T̄ the mean T),
  kh, pdf_lead (steps), var, bin, z, z_over_H (z / H̄), sample; attrs T_mean,
  H_mean. NaN where a source has no value.

metrics.yaml: per source, every scalar as {mean, per_sample: {sample: value}}:
  rmse                 RMS of the rmse curve over the window
  rmse_vs_persistence  rmse / persistence's
  acc_horizon_seconds  the first lead time where acc < `metrics.acc_threshold`;
                       the window's last lead time if it never does (censored)
  acc_horizon_censored 1 if acc never drops below the threshold, else 0 (the
                       mean is the censored fraction of the samples)
  stable               the rollout is finite over the window
  vorticity_rmse, rmse_pedestrian
                       RMS of those curves over the window
  mean_u_rmse, tke_rmse
                       RMSE of the time-mean u and resolved-TKE fields over the
                       fluid cells
  profile_rmse_<mean_u,tke,uw>
                       RMSE over z of the profiles above
  spectra_db_<u,w>_<large,mid,near_cutoff>
                       10 log10 of the window-mean, band-summed spectrum against
                       the truth's, in the wavelength bands
                       evaluation.turbulence.SPECTRAL_BANDS (in dy)
  integral_length      window mean of Λ_y / H (truth too)
  wasserstein_<u,v,w>  1-D Wasserstein-2 distance between the fluctuations about
                       ū at k_c of prediction and truth, pooled over the window
                       (`metrics.max_pdf_values` random (lead, cell) draws, the
                       same for every source)
  divergence_ratio     window mean of divergence_rms / the truth's
plus seconds_per_step, n_parameters and speedup (`metrics.solver_seconds_per_step`
/ seconds_per_step, when set) of every model; truth_halves has the window
statistics (mean_u_rmse to wasserstein); a top-level `reference` entry has the
reference scales H, u_ref, T, k_c and z_c per sample.

metrics_reference.pdf  the definitions of every metric (prebuilt from
                       metrics_reference.tex)
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Any

import numpy as np
import xarray
import yaml
from evaluation.scores import wasserstein2
from evaluation.turbulence import SPECTRAL_BANDS, band_energy_ratio, statistic_rmse
from neural_surrogate_baselines import diagnostics
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

import surrogate_metrics as sm  # noqa: E402
from eval_common import grid_spacing, load_states, open_dataset  # noqa: E402

PERSISTENCE, TRUTH, HALVES = "persistence", "truth", "truth_halves"
VELOCITY = ("u", "v", "w")
SPECTRA_VARS = ("u", "w")
PROFILES = ("mean_u", "tke", "uw")
N_KH_BINS = 32
N_PDF_BINS = 60
PDF_PERCENTILES = (0.05, 99.95)  # of the truth's values: the PDF bin range
STATISTICS = (
    ["mean_u_rmse", "tke_rmse"]
    + [f"profile_rmse_{p}" for p in PROFILES]
    + [f"spectra_db_{v}_{b}" for v in SPECTRA_VARS for b in SPECTRAL_BANDS]
    + [f"wasserstein_{v}" for v in VELOCITY]
)


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
    n = min(_check_rollouts(run_dir, labels, data, s, cfg) for s in samples)
    pdf_leads = sorted(
        {lead if lead > 0 else n + 1 + lead for lead in cfg.metrics.snapshot_leads}
        & set(range(1, n + 1))
    )
    ctx = {
        "n": n,
        "pdf_leads": pdf_leads,
        "pedestrian": int(cfg.metrics.pedestrian_level),
        "max_values": int(cfg.metrics.max_pdf_values),
        "pdf_share": int(cfg.metrics.max_pdf_values) // len(samples),
    }
    scores = [_score_sample(run_dir, labels, data, s, ctx) for s in samples]
    metrics = _scalars(scores, samples, labels, models, cfg)
    (run_dir / "metrics.yaml").write_text(yaml.safe_dump(metrics, sort_keys=False))
    _dataset(scores, labels, pdf_leads).to_netcdf(run_dir / "metrics.nc")
    reference = "metrics_reference.pdf"
    shutil.copyfile(pathlib.Path(__file__).parent / reference, run_dir / reference)
    print(f"Saved metrics.yaml, metrics.nc and {reference} in {run_dir}")


def _rollout_file(run_dir: pathlib.Path, label: str, sample: int) -> pathlib.Path:
    return run_dir / "rollouts" / label / f"sample_{sample:04d}.nc"


def _check_rollouts(
    run_dir: pathlib.Path, labels: list[str], data: Any, sample: int, cfg: DictConfig
) -> int:
    """The shortest rollout of `sample`, after checking every rollout is of this
    trajectory and split, from the same start frame."""
    with xarray.open_dataset(_rollout_file(run_dir, labels[0], sample)) as ds:
        start = int(ds.attrs["start"])
    with xarray.open_dataset(data._state_files[sample]) as ds:
        corpus_times = ds["time"].values
    lengths = []
    for label in labels:
        path = _rollout_file(run_dir, label, sample)
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
            lengths.append(int(ds.sizes["time"]))
    return min(lengths)


def _score_sample(
    run_dir: pathlib.Path,
    labels: list[str],
    data: Any,
    sample: int,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Curves, statistics and PDF values of every source on one trajectory.

    Holds the truth and one model's rollout at a time.
    """
    n = common["n"]
    with xarray.open_dataset(_rollout_file(run_dir, labels[0], sample)) as ds:
        start, z = int(ds.attrs["start"]), ds["z"].values
    with xarray.open_dataset(data._state_files[sample]) as ds:
        times = ds["time"].values[start : start + n + 1].astype(float)
    velocity = [list(data.state_vars).index(v) for v in VELOCITY]
    window = load_states(data, sample, range(start, start + n + 1))[:, velocity]
    truth = window[1:]
    fluid = data.geometry_for(sample).numpy().astype(bool)
    spacing = grid_spacing(data, sample)
    dt = (times[-1] - times[0]) / n  # the mean output interval (it jitters)
    reference = sm.reference_scales(truth, fluid, spacing, dt)
    level = reference["k_c"]
    mean = truth.mean(axis=0, dtype=np.float64).astype(truth.dtype)
    cells = fluid[level].size  # every cell of k_c is fluid
    rng = np.random.default_rng(sample)  # the same draws for every source
    ctx = {
        **common,
        "fluid": fluid,
        "spacing": spacing,
        "interior": sm.interior_cells(fluid),
        "H": reference["H"],
        "u": reference["u_ref"],
        "level": level,
        "mean": mean,
        "mean_values": sm.fluid_values(mean[None], fluid)[0],
        "truth": truth,
        "truth_values": sm.fluid_values(truth, fluid),
        "draws": (
            rng.integers(n, size=common["max_values"]),
            rng.integers(cells, size=common["max_values"]),
        ),
        "pdf_cells": rng.permutation(cells)[: common["pdf_share"]],
    }
    ctx["k"] = 2 * np.pi * np.fft.rfftfreq(fluid.shape[1], d=spacing[1])[1:]  # rad/m

    truth_physics = _physics(truth, ctx)
    truth_stats = _statistics(truth, ctx)
    half = n // 2
    first, second = _statistics(truth[:half], ctx), _statistics(truth[half:], ctx)
    halves = _compare(first, second, ctx)
    halves |= _spectra_db(
        {v: truth_physics["curves"][f"spectrum_{v}"][:half] for v in SPECTRA_VARS},
        {v: truth_physics["curves"][f"spectrum_{v}"][half:] for v in SPECTRA_VARS},
        ctx,
    )
    early = ctx["draws"][0] < half
    halves |= _wasserstein(
        truth_physics["w2_values"][:, early], truth_physics["w2_values"][:, ~early]
    )
    out: dict[str, Any] = {
        "sample": sample,
        "times": times - times[0],
        "z": z,
        "reference": reference,
        "kh": ctx["k"] * reference["H"],
        "curves": {TRUTH: truth_physics["curves"]},
        "pdf": {TRUTH: truth_physics["pdf"]},
        "profiles": {TRUTH: truth_stats["profiles"]},
        "stats": {HALVES: halves},
        "stable": {},
    }
    for name in [PERSISTENCE, *labels]:
        pred = (
            np.broadcast_to(window[0], truth.shape)
            if name == PERSISTENCE
            else _read_rollout(_rollout_file(run_dir, name, sample), n)
        )
        finite = np.isfinite(pred).all(axis=(1, 2, 3, 4))
        bad = n if finite.all() else int(np.argmin(finite))
        physics = _physics(pred[:bad], ctx)
        out["curves"][name] = {
            key: _pad(curve, n)
            for key, curve in (_errors(pred[:bad], ctx) | physics["curves"]).items()
        }
        out["pdf"][name] = physics["pdf"]
        out["stable"][name] = stable = bad == n
        stats = dict.fromkeys(STATISTICS, float("nan"))
        if stable:
            pred_stats = _statistics(pred, ctx)
            out["profiles"][name] = pred_stats["profiles"]
            stats = _compare(pred_stats, truth_stats, ctx)
            stats |= _spectra_db(
                {v: physics["curves"][f"spectrum_{v}"] for v in SPECTRA_VARS},
                {v: truth_physics["curves"][f"spectrum_{v}"] for v in SPECTRA_VARS},
                ctx,
            )
            stats |= _wasserstein(physics["w2_values"], truth_physics["w2_values"])
        out["stats"][name] = stats
        del pred
    return out


def _read_rollout(path: pathlib.Path, n: int) -> np.ndarray:
    """(n, 3, z, y, x) the first n lead steps of a rollout, as float32."""
    with xarray.open_dataset(path) as ds:
        return np.stack([ds[v].values[:n] for v in VELOCITY], axis=1, dtype=np.float32)


def _pad(curve: np.ndarray, n: int) -> np.ndarray:
    """`curve` (lead first) extended with NaN to n leads."""
    out = np.full((n, *curve.shape[1:]), np.nan)
    out[: len(curve)] = curve
    return out


def _errors(pred: np.ndarray, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    """The dimensionless error curves of the finite frames `pred` against the truth."""
    u, H, k_p = ctx["u"], ctx["H"], ctx["pedestrian"]
    truth = ctx["truth"][: len(pred)]
    p, t = sm.fluid_values(pred, ctx["fluid"]), ctx["truth_values"][: len(pred)]
    curves: dict[str, np.ndarray] = {}
    with np.errstate(all="ignore"):  # an empty or diverging rollout
        rmse, rmse_c = sm.rmse(p, t)
        acc, acc_c = sm.acc(p, t, ctx["mean_values"])
        level = np.zeros_like(ctx["fluid"])
        level[k_p] = ctx["fluid"][k_p]
        p_ped, t_ped = sm.fluid_values(pred, level), sm.fluid_values(truth, level)
        rmse_ped, _ = sm.rmse(p_ped, t_ped)
        acc_ped, _ = sm.acc(p_ped, t_ped, sm.fluid_values(ctx["mean"][None], level)[0])
        vort, vort_c = sm.vorticity_rmse(pred, truth, ctx["interior"], ctx["spacing"])
    curves["rmse"], curves["acc"] = rmse / u, acc
    for c, var in enumerate(VELOCITY):
        curves[f"rmse_{var}"], curves[f"acc_{var}"] = rmse_c[:, c] / u, acc_c[:, c]
    curves["rmse_pedestrian"], curves["acc_pedestrian"] = rmse_ped / u, acc_ped
    curves["vorticity_rmse"] = vort * H / u
    for c, axis in enumerate("xyz"):
        curves[f"vorticity_rmse_{axis}"] = vort_c[:, c] * H / u
    return curves


def _physics(fields: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    """The dimensionless curves every source has (truth too), the PDF values at
    the snapshot leads ((n_pdf_leads, 3, m), NaN past the frames given) and the
    Wasserstein values ((3, max_values) at the shared draws, NaN-free only for a
    full window)."""
    u, H, dy, level = ctx["u"], ctx["H"], ctx["spacing"][1], ctx["level"]
    with np.errstate(all="ignore"):
        plane = (fields[:, :, level] - ctx["mean"][:, level]) / u  # (T, 3, ny, nx)
        lines = {
            var: sm.spanwise_lines(plane[:, VELOCITY.index(var)], ctx["fluid"][level])
            for var in SPECTRA_VARS
        }
        divergence = sm.divergence_rms(fields, ctx["interior"], ctx["spacing"])
        curves = {
            "divergence_rms": divergence * H / u,
            "tke": sm.tke(sm.fluid_values(fields, ctx["fluid"])) / u**2,
            "integral_length": sm.integral_length(lines["u"], dy) / H,
        }
        for var in SPECTRA_VARS:
            curves[f"spectrum_{var}"] = sm.spanwise_spectrum(lines[var], dy)[1] / H
    flat = plane.reshape(*plane.shape[:2], -1)
    pdf = np.full((len(ctx["pdf_leads"]), 3, len(ctx["pdf_cells"])), np.nan)
    for j, lead in enumerate(ctx["pdf_leads"]):
        if lead <= len(fields):
            pdf[j] = flat[lead - 1][:, ctx["pdf_cells"]]
    frames, cells = ctx["draws"]
    w2_values = (
        flat[frames, :, cells].T
        if len(fields) == ctx["n"]
        else np.full((3, len(frames)), np.nan)
    )
    return {"curves": curves, "pdf": pdf, "w2_values": w2_values}


def _statistics(fields: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    """Dimensionless time-averaged fields and profiles of one window, about its
    own time mean: the inputs of :func:`_compare`."""
    u = ctx["u"]
    profiles = diagnostics.profiles(fields, ctx["fluid"])
    return {
        "mean_u": fields[:, 0].mean(axis=0, dtype=np.float64) / u,
        "tke": sm.tke_field(fields) / u**2,
        "profiles": {
            "mean_u": profiles["mean_u"] / u,
            "tke": profiles["tke"] / u**2,
            "uw": profiles["uw"] / u**2,
        },
    }


def _compare(
    pred: dict[str, Any], truth: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, float]:
    """The window-statistic scalars of `pred` against `truth`."""
    fluid = ctx["fluid"]
    out = {
        "mean_u_rmse": statistic_rmse(pred["mean_u"], truth["mean_u"], fluid)[0],
        "tke_rmse": statistic_rmse(pred["tke"], truth["tke"], fluid)[0],
    }
    for p in PROFILES:
        error = pred["profiles"][p] - truth["profiles"][p]
        out[f"profile_rmse_{p}"] = float(np.sqrt(np.mean(error**2)))
    return out


def _spectra_db(
    pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], ctx: dict[str, Any]
) -> dict[str, float]:
    """spectra_db_<var>_<band> of the (T, n_k) per-lead spectra averaged over the
    window (a mean, so windows of different length compare)."""
    out: dict[str, float] = {}
    cycles = ctx["k"] / (2 * np.pi)  # band_energy_ratio takes cycles/m
    for var in SPECTRA_VARS:
        ratios = band_energy_ratio(
            cycles, pred[var].mean(axis=0), truth[var].mean(axis=0), ctx["spacing"][1]
        )
        out |= {f"spectra_db_{var}_{b}": float(r) for b, r in ratios.items()}
    return out


def _wasserstein(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """wasserstein_<var> of the (3, m) fluctuation values."""
    return {
        f"wasserstein_{var}": wasserstein2(truth[c], pred[c])["w2"]
        for c, var in enumerate(VELOCITY)
    }


def _scalars(
    scores: list[dict[str, Any]],
    samples: list[int],
    labels: list[str],
    models: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """metrics.yaml: per source, each scalar's mean and per-sample values."""
    threshold = float(cfg.metrics.acc_threshold)
    per_sample: dict[str, dict[str, list[Any]]] = {}

    def add(name: str, row: dict[str, Any]) -> None:
        for key, value in row.items():
            per_sample.setdefault(name, {}).setdefault(key, []).append(value)

    for score in scores:
        curves = score["curves"]
        persistence_rmse = sm.window_rms(curves[PERSISTENCE]["rmse"])
        truth_divergence = np.mean(curves[TRUTH]["divergence_rms"])
        for name in [*labels, PERSISTENCE]:
            own = curves[name]
            below = np.flatnonzero(~(own["acc"] >= threshold))
            # Censored at the window end when ACC never drops below the threshold.
            horizon = below[0] + 1 if below.size else len(own["acc"])
            with np.errstate(all="ignore"):
                add(
                    name,
                    {
                        "rmse": sm.window_rms(own["rmse"]),
                        "rmse_vs_persistence": sm.window_rms(own["rmse"])
                        / persistence_rmse,
                        "acc_horizon_seconds": float(score["times"][horizon]),
                        "acc_horizon_censored": int(not below.size),
                        "stable": score["stable"][name],
                        "vorticity_rmse": sm.window_rms(own["vorticity_rmse"]),
                        "rmse_pedestrian": sm.window_rms(own["rmse_pedestrian"]),
                        **score["stats"][name],
                        "integral_length": float(np.mean(own["integral_length"])),
                        "divergence_ratio": float(
                            np.mean(own["divergence_rms"]) / truth_divergence
                        ),
                    },
                )
        add(
            TRUTH, {"integral_length": float(np.mean(curves[TRUTH]["integral_length"]))}
        )
        add(HALVES, score["stats"][HALVES])
        add("reference", score["reference"])

    solver = cfg.metrics.get("solver_seconds_per_step")
    metrics: dict[str, Any] = {}
    for name, rows in per_sample.items():
        entry: dict[str, Any] = {
            key: {
                "mean": _mean(values),
                "per_sample": dict(zip(samples, (_native(v) for v in values))),
            }
            for key, values in rows.items()
        }
        if name in models:
            seconds = models[name].get("seconds_per_step")
            entry["seconds_per_step"] = seconds
            entry["n_parameters"] = models[name].get("n_parameters")
            if solver is not None and seconds:
                entry["speedup"] = float(solver) / seconds
        metrics[name] = entry
    return metrics


def _mean(values: list[Any]) -> Any:
    if isinstance(values[0], bool):
        return all(values)
    return float(np.mean(values))


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return float(value) if isinstance(value, (float, np.floating)) else value


def _dataset(
    scores: list[dict[str, Any]], labels: list[str], pdf_leads: list[int]
) -> xarray.Dataset:
    """metrics.nc: curves (source x lead), spectra, PDFs, profiles, reference scales."""
    sources = [*labels, PERSISTENCE, TRUTH]
    n = len(scores[0]["curves"][TRUTH]["tke"])
    reference = {
        key: np.array([s["reference"][key] for s in scores])
        for key in ("H", "u_ref", "T", "k_c")
    }
    T_mean, H_mean = float(reference["T"].mean()), float(reference["H"].mean())
    data_vars: dict[str, Any] = {}
    for key in scores[0]["curves"][labels[0]]:
        if key.startswith("spectrum_"):
            continue
        table = np.full((len(sources), n), np.nan)
        for i, name in enumerate(sources):
            if key in scores[0]["curves"][name]:
                table[i] = np.mean([s["curves"][name][key] for s in scores], axis=0)
        data_vars[key] = (("source", "lead"), table)

    # Spectra: each sample's on shared log-spaced k H bins, averaged over the
    # samples whose wavenumbers reach the bin.
    lo = min(s["kh"][0] for s in scores)
    hi = max(s["kh"][-1] for s in scores)
    edges = np.geomspace(lo, hi * (1 + 1e-9), N_KH_BINS + 1)
    covered = np.array(
        [
            np.isfinite(sm.bin_spectrum(s["kh"], np.ones(len(s["kh"])), edges))
            for s in scores
        ]
    )  # (sample, kh)
    for var in SPECTRA_VARS:
        table = np.full((len(sources), n, N_KH_BINS), np.nan)
        for i, name in enumerate(sources):
            binned = np.stack(
                [
                    np.where(
                        c,
                        sm.bin_spectrum(
                            s["kh"], s["curves"][name][f"spectrum_{var}"], edges
                        ),
                        0.0,
                    )
                    for s, c in zip(scores, covered)
                ]
            )
            with np.errstate(all="ignore"):
                table[i] = binned.sum(axis=0) / covered.sum(axis=0)
        data_vars[f"spectrum_{var}"] = (("source", "lead", "kh"), table)

    # PDFs: pooled over the samples, on bins over the truth's range at every
    # snapshot lead (a diverging model must not squeeze it); NaN if any sample
    # has a non-finite value.
    density = np.full((len(sources), len(pdf_leads), len(VELOCITY), N_PDF_BINS), np.nan)
    centres = []
    for c, _ in enumerate(VELOCITY):
        lo, hi = np.percentile(
            np.concatenate([s["pdf"][TRUTH][:, c].ravel() for s in scores]),
            PDF_PERCENTILES,
        )
        edges_c = np.linspace(lo, hi, N_PDF_BINS + 1)
        centres.append(0.5 * (edges_c[1:] + edges_c[:-1]))
        for i, name in enumerate(sources):
            for j in range(len(pdf_leads)):
                pooled = np.concatenate([s["pdf"][name][j, c] for s in scores])
                if np.isfinite(pooled).all():
                    density[i, j, c] = sm.density(pooled, edges_c)
    data_vars["pdf"] = (("source", "pdf_lead", "var", "bin"), density)
    data_vars["pdf_bin_centre"] = (("var", "bin"), np.array(centres))

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

    data_vars |= {
        "reference_H": ("sample", reference["H"]),
        "reference_u": ("sample", reference["u_ref"]),
        "reference_T": ("sample", reference["T"]),
        "canopy_level": ("sample", reference["k_c"]),
    }
    lead_time = scores[0]["times"][1 : n + 1]
    coords: dict[str, Any] = {
        "source": sources,
        "lead": np.arange(1, n + 1),
        "lead_time": ("lead", lead_time),
        "lead_over_T": ("lead", lead_time / T_mean),
        "kh": np.sqrt(edges[1:] * edges[:-1]),
        "pdf_lead": pdf_leads,
        "var": list(VELOCITY),
        "bin": np.arange(N_PDF_BINS),
        "z": z,
        "z_over_H": ("z", z / H_mean),
        "sample": [s["sample"] for s in scores],
    }
    return xarray.Dataset(
        data_vars, coords=coords, attrs={"T_mean": T_mean, "H_mean": H_mean}
    )


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
