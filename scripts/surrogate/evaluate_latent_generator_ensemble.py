"""Evaluate a latent generator against the solver's seed ensemble.

Config: configs/surrogate/eval.yaml, block `latent_generator_ensemble`.

    python scripts/surrogate/evaluate_latent_generator_ensemble.py \
        latent_generator_ensemble.model_dir=model_weights/lg \
        latent_generator_ensemble.ensemble_dir=training_data/pyudales_realistic_seed_ensemble

A seed ensemble (scripts/surrogate/generate_data.py, "Seed ensemble") holds,
per held-out geometry, `state/sample_XXXX/member_YYYY.nc`: spun-up states with
the geometry, parameters and spin-up of test trajectory XXXX of the reference
corpus, differing only in their seeds. They sample p(x | geometry,
parameters) at the reference's first frame, whose generator conditioning is
the t = 0 history `params_hist_for(traj, 0)`. Per geometry, one at a time:

  real                       the members, read like the corpus loader reads a
                             frame (member 0 is checked against frame 0)
  ae_recon                   each member through the frozen AE (paired with
                             its member: never scored against its own truth)
  generated                  `num_draws` draws with the t = 0 history
  generated_shuffled_params  `probe_draws` draws with the next test
                             trajectory's t = 0 history (the conditioning probe)

Draw k's initial noise depends only on (`seed`, sample, k), so every model
sees the same noise (common random numbers for paired model comparisons), and
probe draw k uses draw k's noise. The metrics, per geometry and variable, over
fluid cells (neural_surrogates.generator_evaluation):

  bias_*        squared ensemble-mean bias, raw / debiased / standardised by
                the mean real variance
  spread_*      sqrt(sum var_S / sum var_R) (global, per height in the
                arrays), the correlation of the sigma maps and the RMS of
                log(sigma_S / sigma_R)
  crps_*        fair CRPS at `num_points` fixed fluid cells (where the
                members vary; `zero_variance_cells` on the `real` row), every member as
                truth; crps_ratio = CRPS(S) / leave-one-out CRPS(R)
  rank_*        rank histogram of the members within S (`rank_bins` bins):
                chi2 per bin and outer-bin mass (flat: 2 / rank_bins)
  energy_distance  unbiased energy distance of the whole fluid fields (RMS
                distance over u, v, w), and relative to the real pairwise one
  diversity_ratio  mean pairwise distance of S over that of R
  *_half        the same at half size (`num_splits` random splits of the
                members for energy distance and W1, `num_field_splits` for the
                full-field spread and bias): on the `real` row half vs half
                (the floor, `*_half_std` its spread), on a source row a
                half-size subset of S vs the other real half; w1_points
                (per-point marginal W1) is only reported this way
  physics       profile / fluctuation-RMS RMSE, spectra LSD, Reynolds
                stresses and divergence of each source against the members
  latent_var_ratio_*  generated vs encoded-member latent variance per
                eigen-band of the members' latent PCA (`latent_spectrum`)

The `real` row holds the floors: the leave-one-out CRPS and rank histogram
and the split-half values. Aggregates are the mean over geometries with a
geometry-bootstrap CI (`bootstrap_resamples`).

Outputs, in `latent_generator_ensemble.output_dir`:
  summary.json            settings, per-geometry scalars, member-0 check
                          (report only: per variable max / RMS difference
                          and RMS over the median member distance), wall
                          times, aggregates with CIs
  metrics.csv             one row per geometry x source, plus the mean rows
  arrays/sample_XXXX.npz  per-height spread ratio and standardised bias, rank
                          histogram counts, sigma maps at one height (read by
                          compare_latent_generator_ensembles.py)
  spread_ratio_profiles.png  spread ratio per height and variable
  rank_histograms.png     rank histograms summed over geometries
  std_maps_sample_XXXX.png   sigma of u, v, w at one height: real, AE,
                          generated, probe (first `num_figures` geometries)
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Any

import hydra
import numpy as np
import torch
import xarray
from matplotlib import pyplot as plt
from neural_surrogates import generator_evaluation as ge
from neural_surrogates.architectures.tadpole_latent_flow import SAMPLER_NFE_PER_STEP
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import (  # noqa: E402
    ae_reconstruct,
    device,
    encoded_latent_moments,
    generator_samples,
    grid_spacing,
    load_latent_generator,
    load_states,
    open_dataset,
    plot_slice_rows,
    read_states,
    write_csv,
    write_json,
)

SOURCES = ("ae_recon", "generated", "generated_shuffled_params")
# Sources whose member i is derived from real member i.
PAIRED = ("ae_recon",)
# Sources with split-half (matched-size) values; the probe is too small.
HALVED = ("ae_recon", "generated")
PHYSICS = ("ae_recon", "generated")
STYLE = {
    "real": {"color": "k", "ls": "--"},
    "ae_recon": {"color": "tab:gray", "ls": ":"},
    "generated": {"color": "tab:blue", "ls": "-"},
    "generated_shuffled_params": {"color": "tab:orange", "ls": "-"},
}


def run(cfg: DictConfig) -> None:
    ev = cfg.latent_generator_ensemble
    _check(ev)
    dev = device(cfg.device)
    out = pathlib.Path(ev.output_dir)
    (out / "arrays").mkdir(parents=True, exist_ok=True)
    ensemble_dir = pathlib.Path(ev.ensemble_dir)
    made = OmegaConf.load(ensemble_dir / "config.yaml").data.seed_ensemble
    data_dir = cfg.data_dir if cfg.data_dir is not None else made.reference_dir

    model, train_cfg = load_latent_generator(
        ev.model_dir, dev, ev.max_internal_batchsize
    )
    data = open_dataset(train_cfg, data_dir, str(made.split))
    names = list(data.state_vars)
    traj_of = {int(p.stem.split("_")[-1]): i for i, p in enumerate(data._state_files)}
    samples = (
        sorted(int(p.name.split("_")[-1]) for p in ensemble_dir.glob("state/sample_*"))
        if ev.samples is None
        else [int(s) for s in ev.samples]
    )
    missing = [s for s in samples if s not in traj_of]
    if missing:
        raise ValueError(
            f"samples {missing} of {ensemble_dir} are not in the {made.split} "
            f"split of {data_dir}"
        )
    steps = int(
        train_cfg.generator.sampling.num_steps if ev.num_steps is None else ev.num_steps
    )

    per_geometry: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        begin = time.perf_counter()
        result, arrays = _evaluate_geometry(
            cfg, model, data, ensemble_dir, sample, traj_of[sample], steps, dev
        )
        result["seconds"]["total"] = time.perf_counter() - begin
        key = f"sample_{sample:04d}"
        per_geometry[key] = result
        np.savez_compressed(out / "arrays" / f"{key}.npz", **arrays)
        for source, scalars in result["scalars"].items():
            rows.append({"geometry": key, "source": source, **scalars})
        if i < int(ev.num_figures):
            _plot_std_maps(arrays, names, out / f"std_maps_{key}.png")
        seconds = ", ".join(f"{k} {v:.0f} s" for k, v in result["seconds"].items())
        print(f"{key} ({result['geometry']}, n_real {result['n_real']}): {seconds}")

    aggregate: dict[str, dict[str, Any]] = {}
    for source in ("real", *SOURCES):
        per = [r for r in rows if r["source"] == source]
        if not per:
            continue
        keys = [k for k in per[0] if k not in ("geometry", "source")]
        aggregate[source] = {
            k: ge.bootstrap_ci(
                [r.get(k, np.nan) for r in per],
                int(ev.bootstrap_resamples),
                int(cfg.seed),
            )
            for k in keys
        }
        rows.append(
            {"geometry": "mean", "source": source}
            | {k: v["mean"] for k, v in aggregate[source].items()}
        )
    checks = [g["member0_check"] for g in per_geometry.values() if g["member0_check"]]
    summary = {
        "model_dir": str(ev.model_dir),
        "ensemble_dir": str(ensemble_dir),
        "data_dir": str(data_dir),
        "split": str(made.split),
        "state_vars": names,
        "sampler": str(ev.sampler),
        "num_steps": steps,
        "nfe": steps * SAMPLER_NFE_PER_STEP[str(ev.sampler)],
        "num_draws": int(ev.num_draws),
        "probe_draws": int(ev.probe_draws),
        "num_points": int(ev.num_points),
        "rank_bins": int(ev.rank_bins),
        "num_splits": int(ev.num_splits),
        "num_field_splits": int(ev.num_field_splits),
        "seed": int(cfg.seed),
        "member0_max_abs_diff": max(
            (c["max_abs_diff"] for c in checks), default=float("nan")
        ),
        "member0_rms_diff_over_median_pair": max(
            (c["rms_diff_all_over_median_pair"] for c in checks), default=float("nan")
        ),
        "aggregate": aggregate,
        "per_geometry": per_geometry,
    }
    write_json(summary, out / "summary.json")
    write_csv(rows, out / "metrics.csv")
    _plot_spread_profiles(out, samples, names, out / "spread_ratio_profiles.png")
    _plot_rank_histograms(out, samples, names, out / "rank_histograms.png")
    gen = aggregate["generated"]
    print(
        f"generated: spread ratio {gen['spread_ratio']['mean']:.3f}, CRPS ratio "
        f"{gen['crps_ratio']['mean']:.3f}, energy distance (rel) "
        f"{gen['energy_distance_rel']['mean']:.4f} vs floor "
        f"{aggregate['real']['energy_distance_rel_half']['mean']:.4f}"
    )
    print(f"Saved evaluation in {out}")


def _check(ev: DictConfig) -> None:
    """Every bad setting at once, before the model loads."""
    problems = []
    if int(ev.num_draws) < 2:
        problems.append(f"num_draws={ev.num_draws}: need at least 2")
    if not 0 <= int(ev.probe_draws) <= int(ev.num_draws):
        problems.append(
            f"probe_draws={ev.probe_draws}: must be in [0, num_draws] (probe draw k "
            "reuses draw k's noise)"
        )
    if str(ev.sampler) not in SAMPLER_NFE_PER_STEP:
        problems.append(
            f"sampler={ev.sampler}: use one of {sorted(SAMPLER_NFE_PER_STEP)}"
        )
    for key, low in (
        ("num_points", 1),
        ("rank_bins", 2),
        ("num_splits", 1),
        ("num_field_splits", 1),
    ):
        if int(ev[key]) < low:
            problems.append(f"{key}={ev[key]}: must be >= {low}")
    if problems:
        raise ValueError(
            "Inconsistent latent_generator_ensemble config:\n  - "
            + "\n  - ".join(problems)
        )


def _seed(*keys: int) -> int:
    """A seed that depends only on `keys` (e.g. seed, sample, draw)."""
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0])


def _members(
    ensemble_dir: pathlib.Path, sample: int, data: Any, traj: int
) -> tuple[list[pathlib.Path], str]:
    """The sample's member files, checked against reference trajectory `traj`:
    attrs `reference_sample` / `geometry` and the obstacle mask."""
    paths = sorted(
        (ensemble_dir / "state" / f"sample_{sample:04d}").glob("member_*.nc")
    )
    if len(paths) < 4:
        raise ValueError(f"sample {sample}: {len(paths)} members, need at least 4")
    with xarray.open_dataset(data._state_files[traj]) as ref:
        stl = ref.attrs.get("geometry_stl")
    expected = None if stl is None else pathlib.Path(str(stl)).stem
    fluid = data.geometry_for(traj).numpy()
    name = ""
    for path in paths:
        with xarray.open_dataset(path) as ds:
            name = str(ds.attrs.get("geometry", ""))
            if int(ds.attrs.get("reference_sample", sample)) != sample or (
                expected is not None and name != expected
            ):
                raise ValueError(
                    f"{path}: reference_sample {ds.attrs.get('reference_sample')} / "
                    f"geometry {name!r} do not match test trajectory {sample} "
                    f"({expected})"
                )
            if data.geometry_var in ds and not np.array_equal(
                1.0 - ds[data.geometry_var].values, fluid
            ):
                raise ValueError(f"{path}: obstacle mask differs from the reference")
    return paths, name or (expected or f"sample_{sample:04d}")


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _evaluate_geometry(
    cfg: DictConfig,
    model: Any,
    data: Any,
    ensemble_dir: pathlib.Path,
    sample: int,
    traj: int,
    steps: int,
    dev: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ev = cfg.latent_generator_ensemble
    batch, seed = int(ev.batch_size), int(cfg.seed)
    seconds: dict[str, float] = {}
    begin = time.perf_counter()
    paths, name = _members(ensemble_dir, sample, data, traj)
    real = np.concatenate([read_states(p, data.state_vars) for p in paths])
    fluid = data.geometry_for(traj).numpy().astype(bool)
    hist = data.params_hist_for(traj, 0)
    n_traj = len(data._state_files)
    probe_traj = (traj + 1) % n_traj
    check = _member0_check(paths, real, data, traj, hist, fluid)
    seconds["load"] = time.perf_counter() - begin

    begin = time.perf_counter()
    shape = (model.state_latent_dim, *model.latent_grid_for(fluid.shape))
    noise = torch.stack(
        [
            torch.randn(
                shape, generator=torch.Generator().manual_seed(_seed(seed, sample, k))
            )
            for k in range(int(ev.num_draws))
        ]
    )
    sampled: list[dict] | None = [] if ev.latent_spectrum else None

    def generate(h: torch.Tensor, n: int, latents: list[dict] | None = None) -> Any:
        return generator_samples(
            model,
            h.unsqueeze(0).expand(n, -1, -1),
            data,
            traj,
            0,
            steps,
            batch,
            dev,
            sampler=str(ev.sampler),
            latents=latents,
            noise=noise[:n],
        )

    stacks = {"real": real}
    stacks["generated"] = generate(hist, int(ev.num_draws), sampled)
    if int(ev.probe_draws) > 0:
        stacks["generated_shuffled_params"] = generate(
            data.params_hist_for(probe_traj, 0), int(ev.probe_draws)
        )
    seconds["generate"] = time.perf_counter() - begin
    begin = time.perf_counter()
    stacks["ae_recon"] = ae_reconstruct(model, real, data, traj, batch, dev)
    latent_real = (
        encoded_latent_moments(model, real, data, traj, batch, dev)
        if ev.latent_spectrum
        else None
    )
    seconds["autoencoder"] = time.perf_counter() - begin

    begin = time.perf_counter()
    scalars, arrays = _metrics(cfg, stacks, fluid, data, traj, sample)
    if sampled is not None and latent_real is not None:
        spectrum = ge.latent_spectrum(
            ge.merge_latent_moments(latent_real), ge.merge_latent_moments(sampled)
        )
        scalars["generated"] |= {
            f"latent_var_ratio_{band}": r for band, r in spectrum["bands"].items()
        }
    seconds["metrics"] = time.perf_counter() - begin
    return {
        "geometry": name,
        "traj": traj,
        "n_real": len(real),
        "probe_sample": int(data._state_files[probe_traj].stem.split("_")[-1]),
        "member0_check": check,
        "scalars": scalars,
        "seconds": seconds,
    }, arrays


def _member0_check(
    paths: list[pathlib.Path],
    real: np.ndarray,
    data: Any,
    traj: int,
    hist: torch.Tensor,
    fluid: np.ndarray,
) -> dict[str, Any]:
    """Member 0 (the reference's own seeds) against frame 0 of the reference,
    read by the corpus loader: per variable the max and RMS difference over
    fluid cells and the RMS over the median distance between two members
    (`*_over_median_pair`), and the members' parameters against the last row
    of the t = 0 history. Recorded, not asserted: the seed ensemble need not
    reproduce the corpus run (and labels the frame t = 5 s, not 0)."""
    if paths[0].stem != "member_0000":
        return {}
    names = list(data.state_vars)
    frame = load_states(data, traj, [0])[0]
    diff = (real[0] - frame)[:, fluid]
    pair_sq = np.stack(
        [ge.rms_distances([real[:, c : c + 1]], fluid) ** 2 for c in range(len(names))]
    )
    off = ~np.eye(len(real), dtype=bool)
    median = {
        n: float(np.median(np.sqrt(pair_sq[c][off]))) for c, n in enumerate(names)
    }
    median["all"] = float(np.median(np.sqrt(pair_sq.mean(axis=0)[off])))
    rms = {n: float(np.sqrt(np.mean(diff[c] ** 2))) for c, n in enumerate(names)}
    rms["all"] = float(np.sqrt(np.mean(diff**2)))
    param_file = paths[0].parents[2] / "param" / paths[0].parent.name / paths[0].name
    params: dict[str, float] = {}
    if param_file.exists():
        with xarray.open_dataset(param_file) as ds:
            for i, p in enumerate(data.param_names):
                if p in ds:
                    value = float(np.asarray(ds[p].values).ravel()[-1])
                    params[p] = value - float(hist[-1, i])
    return {
        "max_abs_diff": float(np.abs(diff).max()),
        **{
            f"max_abs_diff_{n}": float(np.abs(diff[c]).max())
            for c, n in enumerate(names)
        },
        **{f"rms_diff_{n}": v for n, v in rms.items()},
        **{f"rms_diff_{n}_over_median_pair": rms[n] / median[n] for n in rms},
        "median_pair_distance": median,
        "member_params_minus_history": params,
    }


def _per_var(prefix: str, values: Any, names: list[str]) -> dict[str, float]:
    """`prefix_<var>` per variable and `prefix` the mean over variables."""
    v = np.asarray(values, dtype=np.float64)
    return {f"{prefix}_{n}": float(x) for n, x in zip(names, v)} | {
        prefix: ge.nanmean(v)
    }


def _physics(fields: np.ndarray, fluid: np.ndarray, spacing: Any) -> dict[str, Any]:
    return {
        "profiles": ge.profiles(fields, fluid),
        "spectra": ge.spectra(fields, fluid, dx=spacing[2]),
        "reynolds": ge.reynolds_stresses(fields, fluid),
        "divergence": ge.divergence(fields, ge.stencil_fluid_mask(fluid), spacing),
        "n": len(fields),
    }


def _metrics(
    cfg: DictConfig,
    stacks: dict[str, np.ndarray],
    fluid: np.ndarray,
    data: Any,
    traj: int,
    sample: int,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Every per-geometry scalar (one dict per source; `real` holds the
    floors) and the arrays the comparison reads."""
    ev = cfg.latent_generator_ensemble
    seed, bins = _seed(int(cfg.seed), sample), int(ev.rank_bins)
    names = list(data.state_vars)
    real = stacks["real"]
    n_real = len(real)
    order = list(stacks)
    offsets = np.cumsum([0] + [len(stacks[s]) for s in order])
    index = {s: np.arange(offsets[i], offsets[i + 1]) for i, s in enumerate(order)}
    distances = ge.rms_distances([stacks[s] for s in order], fluid)
    real_pairwise = float(distances[np.ix_(index["real"], index["real"])].sum()) / (
        n_real * (n_real - 1)
    )
    moments = {s: ge.ensemble_moments(f, fluid) for s, f in stacks.items()}
    # Points only where the members vary: a constant truth (an enclosed
    # courtyard) has no CRPS or rank to speak of.
    varying = fluid.copy()
    varying[fluid] = (moments["real"]["var"] > 0).all(axis=0)
    points = ge.select_points(varying, int(ev.num_points), seed)
    values = {s: ge.point_values(f, points) for s, f in stacks.items()}
    level = fluid.shape[0] // 4
    spacing = grid_spacing(data, traj)
    arrays: dict[str, np.ndarray] = {
        "height_m": (np.arange(fluid.shape[0]) + 0.5) * spacing[0],
        "std_map_level": np.asarray(level),
        "fluid_level": fluid[level],
    }
    for s, mom in moments.items():
        std = np.full((len(names), *fluid.shape), np.nan)
        std[:, fluid] = np.sqrt(np.maximum(mom["var"], 0.0))
        arrays[f"{s}.std_map"] = std[:, level]

    oracle = ge.fair_crps(values["real"], values["real"], exclude_self=True)
    ranks = ge.rank_histogram(values["real"], values["real"], bins, True, seed)
    arrays["real.rank_counts"] = ranks["counts"]
    scalars: dict[str, dict[str, float]] = {
        "real": {
            **_per_var("crps", oracle, names),
            **_per_var("rank_chi2", ranks["chi2_per_bin"], names),
            **_per_var("rank_outer", ranks["outer_mass"], names),
            "diversity": real_pairwise,
            "zero_variance_cells": int(fluid.sum() - varying.sum()),
            "zero_variance_fraction": float(1.0 - varying.sum() / fluid.sum()),
        }
    }
    real_physics = _physics(real, fluid, spacing)
    scalars["real"]["divergence_rms"] = float(real_physics["divergence"]["rms"])

    # Split-half metrics: the cheap ones (from the distance matrix and the
    # points) over `num_splits` splits, the full-field moments over
    # `num_field_splits`. The real-vs-real floor is cached across sources.
    caches: dict[str, dict[tuple, dict[str, Any]]] = {"cheap": {}, "fields": {}}

    def halves(source: str, kind: str) -> Any:
        def metric(from_source: bool, x: np.ndarray, b: np.ndarray) -> dict:
            key = (tuple(x), tuple(b))
            if not from_source and key in caches[kind]:
                return caches[kind][key]
            s = source if from_source else "real"
            if kind == "cheap":
                ed = ge.energy_distance(distances, index[s][x], index["real"][b])
                result = {
                    "energy_distance": ed["energy_distance"],
                    "energy_distance_rel": ed["relative"],
                    "w1_points": ge.pointwise_w1(values[s][x], values["real"][b]),
                }
            else:
                xm = ge.ensemble_moments(stacks[s][x], fluid)
                bm = ge.ensemble_moments(real[b], fluid)
                sp, bias = ge.spread(xm, bm), ge.mean_bias(xm, bm)
                result = {
                    "spread_ratio": sp["ratio"],
                    "spread_corr": sp["map_corr"],
                    "spread_log_rms": sp["log_ratio_rms"],
                    "bias_standardised": bias["standardised"],
                }
            if not from_source:
                caches[kind][key] = result
            return result

        return metric

    for source in (s for s in SOURCES if s in stacks):
        fields, paired = stacks[source], source in PAIRED
        mom = moments[source]
        diff_var = ge.ensemble_moments(fields - real, fluid)["var"] if paired else None
        bias = ge.mean_bias(mom, moments["real"], diff_var)
        sp = ge.spread(mom, moments["real"])
        crps = ge.fair_crps(values[source], values["real"], exclude_self=paired)
        ranks = ge.rank_histogram(values[source], values["real"], bins, paired, seed)
        ed = ge.energy_distance(distances, index[source], index["real"], paired)
        own = distances[np.ix_(index[source], index[source])]
        row = {
            **_per_var("bias_raw", bias["raw"], names),
            **_per_var("bias_debiased", bias["debiased"], names),
            **_per_var("bias_standardised", bias["standardised"], names),
            **_per_var("spread_ratio", sp["ratio"], names),
            "spread_ratio_total": sp["ratio_total"],
            **_per_var("spread_corr", sp["map_corr"], names),
            **_per_var("spread_log_rms", sp["log_ratio_rms"], names),
            **_per_var("crps", crps, names),
            **_per_var("crps_ratio", crps / oracle, names),
            **_per_var("rank_chi2", ranks["chi2_per_bin"], names),
            **_per_var("rank_outer", ranks["outer_mass"], names),
            "energy_distance": ed["energy_distance"],
            "energy_distance_rel": ed["relative"],
            "diversity": float(own.sum()) / (len(own) * (len(own) - 1)),
        }
        row["diversity_ratio"] = row["diversity"] / real_pairwise
        if source in PHYSICS:
            row |= ge.compare_to_real(_physics(fields, fluid, spacing), real_physics)
        kinds = (
            (("cheap", int(ev.num_splits)), ("fields", int(ev.num_field_splits)))
            if source in HALVED
            else ()
        )
        for kind, splits in kinds:
            split = ge.split_half_floor(
                halves(source, kind),
                n_real,
                splits,
                seed,
                n_source=len(fields),
                paired=paired,
            )
            for k, v in split["matched"].items():
                row |= _half(k, v, names)
            for k, v in split["floor"].items():
                scalars["real"] |= _half(k, v, names)
            for k, v in split["floor_std"].items():
                scalars["real"] |= _half(k, v, names, "_std")
        scalars[source] = row
        arrays[f"{source}.spread_ratio_profile"] = sp["ratio_profile"]
        arrays[f"{source}.bias_profile_standardised"] = bias["profile_standardised"]
        arrays[f"{source}.rank_counts"] = ranks["counts"]
    return scalars, arrays


def _half(key: str, value: Any, names: list[str], suffix: str = "") -> dict:
    """A split-half value as `<key>_half[<suffix>]` (per variable when an array)."""
    if np.ndim(value) == 0:
        return {f"{key}_half{suffix}": float(value)}
    return {
        f"{k.replace(key, key + '_half', 1)}{suffix}": v
        for k, v in _per_var(key, value, names).items()
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _load_arrays(out: pathlib.Path, samples: list[int]) -> list[dict]:
    loaded = []
    for s in samples:
        with np.load(out / "arrays" / f"sample_{s:04d}.npz") as f:
            loaded.append(dict(f))
    return loaded


def _plot_spread_profiles(
    out: pathlib.Path, samples: list[int], names: list[str], path: pathlib.Path
) -> None:
    """Spread ratio per height: each geometry thin, the mean thick."""
    arrays = _load_arrays(out, samples)
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.4 * len(names), 3.8), squeeze=False
    )
    for source in ("ae_recon", "generated", "generated_shuffled_params"):
        key = f"{source}.spread_ratio_profile"
        if key not in arrays[0]:
            continue
        for c, name in enumerate(names):
            ax = axes[0, c]
            curves = []
            for a in arrays:
                ax.plot(a[key][c], a["height_m"], lw=0.5, alpha=0.4, **STYLE[source])
                curves.append(
                    np.interp(
                        arrays[0]["height_m"],
                        a["height_m"],
                        a[key][c],
                        left=np.nan,
                        right=np.nan,
                    )
                )
            ax.plot(
                np.nanmean(curves, axis=0),
                arrays[0]["height_m"],
                lw=2,
                label=source,
                **STYLE[source],
            )
            ax.axvline(1.0, color="k", lw=0.8)
            ax.set_title(f"{name}: spread ratio")
            ax.set_xlabel("sigma_S / sigma_R")
            ax.set_ylabel("height [m]")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_rank_histograms(
    out: pathlib.Path, samples: list[int], names: list[str], path: pathlib.Path
) -> None:
    """Rank histograms summed over geometries (density: 1 = flat)."""
    arrays = _load_arrays(out, samples)
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.4 * len(names), 3.2), squeeze=False
    )
    shown = [s for s in ("real", *SOURCES) if f"{s}.rank_counts" in arrays[0]]
    width = 0.8 / len(shown)
    for c, name in enumerate(names):
        for j, source in enumerate(shown):
            counts = sum(a[f"{source}.rank_counts"][c] for a in arrays)
            density = counts / (counts.sum() / len(counts))
            x = np.arange(len(counts)) + (j - (len(shown) - 1) / 2) * width
            label = "real (leave-one-out)" if source == "real" else source
            ax = axes[0, c]
            ax.bar(x, density, width, color=STYLE[source]["color"], label=label)
        axes[0, c].axhline(1.0, color="k", lw=0.8)
        axes[0, c].set_title(f"{name}: rank histogram")
        axes[0, c].set_xlabel("rank bin")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_std_maps(arrays: dict, names: list[str], path: pathlib.Path) -> None:
    rows = [
        (source, list(arrays[f"{source}.std_map"]), False)
        for source in ("real", *SOURCES)
        if f"{source}.std_map" in arrays
    ]
    plot_slice_rows(
        rows,
        [f"{n}, z index {int(arrays['std_map_level'])}" for n in names],
        arrays["fluid_level"],
        path,
        label="sigma [m/s]",
    )


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs", config_name="surrogate/eval"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
