"""Evaluate a trained latent generator: are its fields statistically like real ones?

Config: configs_new/surrogate/eval.yaml, block `latent_generator`.

    python scripts_new/surrogate/evaluate_latent_generator.py
    python scripts_new/surrogate/evaluate_latent_generator.py \
        latent_generator.model_dir=model_weights/lg latent_generator.rollout_stepper_dir=model_weights/p3d

For held-out snapshots, fields from several sources are compared with the real
ones. The sources are: the autoencoder reconstruction (`ae_recon`, the best the
decoder can do), generated fields with the true parameter history
(`generated`, several noise seeds), with a constant history
(`generated_const_history`), with histories shuffled between snapshots
(`generated_shuffled_history`), with the training-mean parameters
(`generated_omitted_history`), and with other numbers of sampling steps
(`generated_steps<k>`). Statistics are computed per grid and merged.

Outputs, in `latent_generator.output_dir`:
  summary.json        acceptance verdict, held-out reference scales, per-source
                      scalars, diversity, conditioning probe and step sweep
  metrics.csv         the per-source scalars (profile RMSE, W1, spectra,
                      Reynolds stresses, divergence, diversity)
  states_*.png        |U| at a few heights: real, AE, generated draws
  profiles.png        mean and fluctuation-RMS vertical profiles per variable
  histograms.png      value densities per variable, with W1 to real
  spectra.png         1-D energy spectra along x
  divergence.png      divergence RMS per source
  step_sweep.png      profile RMSE, W1 and sampling time against the step count
  rollout_transients.png  kinetic energy of a stepper rolled out from real,
                      AE and generated fields (if `rollout_stepper_dir`)
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Any

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from neural_surrogates import generator_evaluation as ge
from omegaconf import DictConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_common import (  # noqa: E402
    device,
    expand,
    grid_spacing,
    load_model,
    load_states,
    open_dataset,
    plot_slice_rows,
    speed,
    write_csv,
    write_json,
)

# Sources drawn in the profile, histogram and spectrum figures.
PLOTTED = ("real", "ae_recon", "generated", "generated_const_history")
SCALARS = (
    "profile_rmse",
    "rms_profile_rmse",
    "spectra_lsd_db",
    "reynolds_abs_err",
    "divergence_rms",
)


def run(cfg: DictConfig) -> None:
    ev = cfg.latent_generator
    dev = device(cfg.device)
    out = pathlib.Path(ev.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    model, train_cfg = load_model(ev.model_dir, dev)
    schema = train_cfg.generator.physical_schema
    model.set_conditioning_schema(
        list(schema.param_vars), float(schema.history_dt_seconds)
    )
    data = open_dataset(train_cfg, cfg.data_dir, cfg.split)
    names = list(data.state_vars)
    steps = int(train_cfg.generator.sampling.num_steps)
    seeds = int(ev.num_noise_seeds)
    selection = _select(data, int(ev.max_snapshots), rng)

    # Conditioning probes over the whole selection: histories shuffled between
    # snapshots, and the training-mean parameters.
    order = [(traj, t) for traj, times in selection.items() for t in times]
    histories = torch.stack([data.params_hist_for(traj, t) for traj, t in order])
    shuffled = histories[torch.as_tensor(np.roll(rng.permutation(len(order)), 1))]
    omitted = (
        model.param_mean.detach().float().cpu().reshape(1, 1, -1).expand_as(histories)
    )

    def generate(
        hist: torch.Tensor, traj: int, seed: int, num_steps: int = steps
    ) -> np.ndarray:
        return _generate(
            model, hist, data, traj, seed, num_steps, int(ev.batch_size), dev
        )

    groups: dict[str, dict[str, list[dict]]] = {}  # source -> grid -> metric groups
    values: dict[str, list[np.ndarray]] = {}  # source -> pooled value samples
    diversity: dict[str, list[np.ndarray]] = {
        "generated": [],
        "generated_const_history": [],
    }
    real_spread, sweep_seconds = [], {}
    rollout_inputs: list[tuple[int, dict[str, np.ndarray], torch.Tensor]] = []
    start = 0
    for traj, times in selection.items():
        fluid = data.geometry_for(traj).numpy().astype(bool)
        stencil = ge.stencil_fluid_mask(fluid)
        spacing = grid_spacing(data, traj)
        grid = (
            "x".join(str(s) for s in fluid.shape)
            + " @ "
            + ",".join(f"{d:g}" for d in spacing)
        )
        hist = histories[start : start + len(times)]
        idx = slice(start, start + len(times))
        start += len(times)

        def add(source: str, fields: np.ndarray) -> None:
            metrics = {
                "profiles": ge.profiles(fields, fluid),
                "spectra": ge.spectra(fields, fluid, dx=spacing[2]),
                "reynolds": ge.reynolds_stresses(fields, fluid),
                "divergence": ge.divergence(fields, stencil, spacing),
                "n": len(fields),
            }
            groups.setdefault(source, {}).setdefault(grid, []).append(metrics)
            values.setdefault(source, []).append(
                ge.fluid_values(fields, fluid, int(ev.max_values), rng)
            )

        real = load_states(data, traj, times)
        add("real", real)
        real_spread.append(ge.pairwise_rms_distance(real, fluid))
        ae = _ae_reconstruct(model, real, data, traj, int(ev.batch_size), dev)
        add("ae_recon", ae)

        generated = np.stack(
            [generate(hist, traj, 1000 * int(cfg.seed) + s) for s in range(seeds)]
        )
        add("generated", generated.reshape(-1, *generated.shape[2:]))
        diversity["generated"].append(ge.diversity(generated, fluid)["per_condition"])
        constant = hist[:, -1:].expand_as(hist).contiguous()
        const = np.stack(
            [
                generate(constant, traj, 2000 * (int(cfg.seed) + 1) + s)
                for s in range(seeds)
            ]
        )
        add("generated_const_history", const.reshape(-1, *const.shape[2:]))
        diversity["generated_const_history"].append(
            ge.diversity(const, fluid)["per_condition"]
        )
        add(
            "generated_shuffled_history",
            generate(shuffled[idx], traj, 1000 * int(cfg.seed)),
        )
        add(
            "generated_omitted_history",
            generate(omitted[idx], traj, 1000 * int(cfg.seed)),
        )

        for k in ev.num_steps_sweep:
            add(
                f"generated_steps{k}",
                (
                    generated[0]
                    if k == steps
                    else generate(hist, traj, 1000 * int(cfg.seed), k)
                ),
            )
            if k not in sweep_seconds:  # timed once, on the first trajectory
                begin = time.perf_counter()
                generate(hist[:1].expand(int(ev.batch_size), -1, -1), traj, 0, k)
                sweep_seconds[k] = (time.perf_counter() - begin) / int(ev.batch_size)

        if len(rollout_inputs) < int(ev.num_figures):
            rollout_inputs.append(
                (
                    traj,
                    {"real": real, "ae_recon": ae, "generated": generated[0]},
                    hist[:, -1],
                )
            )
            _plot_states(
                real,
                ae,
                generated,
                const,
                fluid,
                out / f"states_traj{traj}_t{times[0]}.png",
            )

    # Merge per grid, then compare every source with the real fields.
    merged = {
        s: {g: ge.merge_group_metrics(m) for g, m in per.items()}
        for s, per in groups.items()
    }
    pooled = {
        s: np.asarray(ge.pool_values(v, int(ev.max_values), rng))
        for s, v in values.items()
    }
    edges = ge.histogram_edges(pooled["real"], int(ev.n_bins))
    hists = {
        s: ge.histograms_and_w1(v, pooled["real"], edges) for s, v in pooled.items()
    }
    n_boot = int(ev.bootstrap_resamples)
    w1_boot = ge.bootstrap_w1(pooled["real"], min(n_boot, 50), int(cfg.seed))
    reference = {
        "bootstrap_profile_rmse": ge.nanmean(
            [
                ge.bootstrap_profile_rmse(
                    m["profiles"]["sample_mean"], n_boot, int(cfg.seed)
                )
                for m in merged["real"].values()
            ]
        ),
        "bootstrap_w1": ge.nanmean(w1_boot),
        "real_divergence_rms": _weighted(
            {g: m["divergence"]["rms"] for g, m in merged["real"].items()},
            merged["real"],
        ),
        "real_pairwise_spread": ge.nanmean(real_spread),
    }
    scalars: dict[str, dict[str, float]] = {}
    for source, per in merged.items():
        compared = {g: ge.compare_to_real(m, merged["real"][g]) for g, m in per.items()}
        scalars[source] = {
            k: _weighted({g: c[k] for g, c in compared.items()}, per) for k in SCALARS
        }
        scalars[source]["w1"] = ge.nanmean(hists[source]["w1"])
        scalars[source] |= {
            f"w1_{n}": float(w) for n, w in zip(names, hists[source]["w1"])
        }
    scalars["real"]["diversity"] = reference["real_pairwise_spread"]
    for source, per_condition in diversity.items():
        scalars[source]["diversity"] = ge.nanmean(np.concatenate(per_condition))
    verdict = ge.aggregate_report(scalars, reference, dict(ev.acceptance))

    true_history = scalars["generated"]
    conditioning = {
        f"{probe}_{key}": scalars[f"generated_{probe}_history"][key] - true_history[key]
        for probe in ("shuffled", "omitted", "const")
        for key in ("profile_rmse", "w1")
    }
    sweep = [
        {
            "num_steps": k,
            **{
                m: scalars[f"generated_steps{k}"][m]
                for m in ("profile_rmse", "w1", "divergence_rms")
            },
            "seconds_per_sample": sweep_seconds[k],
        }
        for k in ev.num_steps_sweep
    ]
    summary = {
        "acceptance": verdict["acceptance"],
        "checks": verdict["checks"],
        "reference": reference,
        "scalars": scalars,
        "conditioning_gain": conditioning,
        "conditioning_note": "error with the probe history minus error with the true history; positive = the true history helps",
        "step_sweep": sweep,
        "num_snapshots": len(order),
        "num_noise_seeds": seeds,
        "num_steps": steps,
    }

    if ev.rollout_stepper_dir is not None:
        transients = _rollout_transients(ev, rollout_inputs, data, dev)
        summary["rollout_kinetic_energy"] = transients
        _plot_rollout(transients, out / "rollout_transients.png")

    write_json(summary, out / "summary.json")
    write_csv([{"source": s, **v} for s, v in scalars.items()], out / "metrics.csv")
    plotted = {s: merged[s] for s in PLOTTED}
    _plot_profiles(plotted, names, out / "profiles.png")
    _plot_spectra(plotted, names, out / "spectra.png")
    _plot_histograms(
        {s: hists[s] for s in PLOTTED}, edges, names, out / "histograms.png"
    )
    _plot_divergence(scalars, out / "divergence.png")
    _plot_sweep(sweep, out / "step_sweep.png")
    print(
        "PASS"
        if verdict["acceptance"]["passed"]
        else "FAIL: " + "; ".join(verdict["acceptance"]["failures"])
    )
    print(f"Saved evaluation in {out}")


def _select(
    data: Any, max_snapshots: int, rng: np.random.Generator
) -> dict[int, list[int]]:
    """Up to `max_snapshots` random snapshots, taken in turn from each trajectory."""
    queues: dict[int, list[int]] = {}
    for traj, t in data.sample_index:
        queues.setdefault(int(traj), []).append(int(t))
    queues = {traj: list(rng.permutation(ts)) for traj, ts in queues.items()}
    chosen: dict[int, list[int]] = {}
    while sum(map(len, chosen.values())) < max_snapshots and any(queues.values()):
        for traj, queue in queues.items():
            if queue and sum(map(len, chosen.values())) < max_snapshots:
                chosen.setdefault(traj, []).append(int(queue.pop()))
    return {traj: sorted(ts) for traj, ts in chosen.items()}


def _weighted(per_grid: dict[str, float], merged: dict[str, dict]) -> float:
    """Mean of a per-grid number, weighted by the grids' sample counts."""
    pairs = [(v, merged[g]["n"]) for g, v in per_grid.items() if np.isfinite(v)]
    return (
        float(sum(v * n for v, n in pairs) / sum(n for _, n in pairs))
        if pairs
        else float("nan")
    )


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _ae_reconstruct(
    model: Any, real: np.ndarray, data: Any, traj: int, batch: int, dev: torch.device
) -> np.ndarray:
    out = []
    for start in range(0, len(real), batch):
        state = torch.from_numpy(real[start : start + batch]).to(dev)
        b = len(state)
        enc = model.encode_latents(
            state,
            expand(data.geometry_for(traj), b, dev),
            expand(data.geom_features_for(traj), b, dev),
        )
        out.append(model.decode_latents(enc.z, enc).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _generate(
    model: Any,
    hist: torch.Tensor,
    data: Any,
    traj: int,
    seed: int,
    num_steps: int,
    batch: int,
    dev: torch.device,
) -> np.ndarray:
    """Generated fields for parameter histories (B, Hp, P); the noise is fixed by `seed`."""
    noise = torch.Generator(device=dev).manual_seed(int(seed))
    out = []
    for start in range(0, len(hist), batch):
        h = hist[start : start + batch].to(dev)
        b = len(h)
        sample = model.sample(
            h,
            expand(data.geometry_for(traj), b, dev),
            expand(data.geom_features_for(traj), b, dev),
            generator=noise,
            num_steps=int(num_steps),
            param_names=model.param_names,
            history_dt_seconds=model.history_dt_seconds,
        )
        out.append(sample.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _rollout_transients(
    ev: DictConfig, inputs: list, data: Any, dev: torch.device
) -> dict[str, list[float]]:
    """Fluid kinetic energy per step of a stepper rolled out from each source's
    fields, under each snapshot's latest parameters. A stepper with a history
    starts from the field repeated over the history."""
    stepper, cfg = load_model(ev.rollout_stepper_dir, dev)
    if list(cfg.dataset.param_vars) != list(data.param_names):
        raise ValueError(
            f"stepper parameters {list(cfg.dataset.param_vars)} differ from the generator's {list(data.param_names)}"
        )
    history = int(cfg.dataset.get("num_history_steps") or 1)
    energy: dict[str, list[np.ndarray]] = {}
    for traj, sources, params in inputs:
        fluid = data.geometry_for(traj).numpy().astype(bool)
        for source, fields in sources.items():
            current = torch.from_numpy(fields).to(dev)
            state = current.repeat(1, history, 1, 1, 1)
            geom = expand(data.geometry_for(traj), len(fields), dev)
            per_step = [ge.kinetic_energy(fields, fluid)]
            for _ in range(int(ev.rollout_steps)):
                current = stepper(state, params.to(dev), geom)
                state = (
                    current
                    if history == 1
                    else torch.cat([state[:, current.shape[1] :], current], dim=1)
                )
                per_step.append(ge.kinetic_energy(current.float().cpu().numpy(), fluid))
            energy.setdefault(source, []).append(np.stack(per_step))  # (steps + 1, B)
    return {
        s: np.concatenate(e, axis=1).mean(axis=1).tolist() for s, e in energy.items()
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(source: str) -> dict[str, Any]:
    """Line style: real dashed black, every other source solid."""
    return {"ls": "--", "color": "k", "lw": 2} if source == "real" else {"lw": 1.4}


def _plot_states(
    real: np.ndarray,
    ae: np.ndarray,
    generated: np.ndarray,
    const: np.ndarray,
    fluid: np.ndarray,
    path: pathlib.Path,
) -> None:
    """|U| of the first snapshot at three heights: real, AE and two generated draws per history."""
    levels = np.unique(np.linspace(0, fluid.shape[0] - 1, 3).astype(int))
    rows = [("real", real[0]), ("AE reconstruction", ae[0])]
    rows += [
        (f"generated, seed {s}", generated[s, 0]) for s in range(min(2, len(generated)))
    ]
    rows += [
        (f"const history, seed {s}", const[s, 0]) for s in range(min(2, len(const)))
    ]
    plot_slice_rows(
        [(name, [speed(f)[z] for z in levels], False) for name, f in rows],
        [f"z index {z}" for z in levels],
        fluid[levels],
        path,
    )


def _per_grid_figure(
    merged: dict, path: pathlib.Path, draw: Any, n_rows: int, n_cols: int
) -> None:
    """One figure per grid (suffixed `_grid<i>` when there are several)."""
    grids = list(merged["real"])
    for i, grid in enumerate(grids):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows), squeeze=False
        )
        for source, per in merged.items():
            style = _style(source)
            draw(axes, per[grid], dict(label=source, **style))
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"grid {grid}")
        fig.tight_layout()
        fig.savefig(
            path if len(grids) == 1 else path.with_stem(f"{path.stem}_grid{i}"), dpi=110
        )
        plt.close(fig)


def _plot_profiles(merged: dict, names: list[str], path: pathlib.Path) -> None:
    def draw(axes: Any, metrics: dict, style: dict) -> None:
        prof = metrics["profiles"]
        for c, name in enumerate(names):
            for r, key in enumerate(("mean", "rms")):
                axes[r, c].plot(prof[key][c], np.arange(prof[key].shape[1]), **style)
                axes[r, c].set_title(
                    f"{name}: {'mean' if key == 'mean' else 'fluctuation RMS'}"
                )
                axes[r, c].set_ylabel("z index")

    _per_grid_figure(merged, path, draw, 2, len(names))


def _plot_spectra(merged: dict, names: list[str], path: pathlib.Path) -> None:
    def draw(axes: Any, metrics: dict, style: dict) -> None:
        k, energy = metrics["spectra"]["k"][1:], metrics["spectra"]["energy"][:, 1:]
        for c, name in enumerate(names):
            ok = np.isfinite(energy[c]) & (energy[c] > 0)
            if ok.any():
                axes[0, c].loglog(k[ok], energy[c][ok], **style)
            axes[0, c].set_title(f"{name}: spectrum along x")
            axes[0, c].set_xlabel("k [rad/m]")

    _per_grid_figure(merged, path, draw, 1, len(names))


def _plot_histograms(
    hists: dict, edges: np.ndarray, names: list[str], path: pathlib.Path
) -> None:
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.8 * len(names), 3.4), squeeze=False
    )
    for c, name in enumerate(names):
        centres = 0.5 * (edges[c, 1:] + edges[c, :-1])
        for source, h in hists.items():
            style = _style(source)
            axes[0, c].plot(
                centres,
                h["density"][c],
                label=f"{source} (W1={h['w1'][c]:.3g})",
                **style,
            )
        axes[0, c].set_xlabel(f"{name} [m/s]")
        axes[0, c].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_divergence(scalars: dict, path: pathlib.Path) -> None:
    sources = [s for s in scalars if not s.startswith("generated_steps")]
    fig, ax = plt.subplots(figsize=(1.2 * len(sources) + 2, 3.6))
    ax.bar(sources, [scalars[s]["divergence_rms"] for s in sources], color="tab:blue")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("divergence RMS [1/s]")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_sweep(sweep: list[dict], path: pathlib.Path) -> None:
    keys = ["profile_rmse", "w1", "divergence_rms", "seconds_per_sample"]
    steps = [row["num_steps"] for row in sweep]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.2))
    for ax, key in zip(axes, keys):
        ax.plot(steps, [row[key] for row in sweep], "o-")
        ax.set_xscale("log")
        ax.set_xlabel("sampling steps")
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_rollout(transients: dict[str, list[float]], path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for source, energy in transients.items():
        style = _style(source)
        ax.plot(energy, label=source, **style)
    ax.set_xlabel("stepper rollout step")
    ax.set_ylabel("fluid kinetic energy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs_new", config_name="surrogate/eval"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
