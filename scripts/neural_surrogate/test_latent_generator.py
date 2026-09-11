"""Statistical acceptance of a trained latent generator (plan 07 §3, phase 3).

Loads a ``train_latent_generator.py`` export and, on a HELD-OUT split of the
artifact's corpus, compares four initial-state sources under matched geometry
and parameter history:

* ``real`` -- the held-out states themselves;
* ``ae_recon`` -- frozen-AE reconstructions ``decode(encode(real))``: the best
  the decoder can do, i.e. the baseline every generation error is judged
  against (AE error and generation error are separate quantities);
* ``generated`` -- ``sample()`` with the snapshot's true history, several noise
  seeds per conditioning;
* ``generated_const_history`` -- the cold-start case: the snapshot's LAST
  parameter row repeated ``Hp`` times (what an ESMDA member with only a current
  parameter value can supply).

Per source (fluid cells only): conditional mean / RMS vertical profiles,
velocity histograms + 1-Wasserstein distances to real, 1-D energy spectra
along ``x``, Reynolds stresses about the per-source mean field, divergence on
a stencil-valid fluid mask with the schema's grid spacing, and diversity across
noise seeds against the real pairwise spread. Then the conditioning probe
(true vs shuffled vs omitted history), the Euler step sweep (metrics + wall
time + peak memory at the deployment batch shape), the padding-band check and
-- optionally -- rollout transients through a trained stepper. The metric
functions live in :mod:`neural_surrogates.generator_evaluation`; this script
owns the model, the batching and the files.

Acceptance is declared, not eyeballed: ``summary.json`` carries the tolerance
block (``acceptance.*`` factors relative to the AE baseline and to the held-out
bootstrap spread) and ``acceptance: {passed, failures}``; the same verdict is
printed as PASS / FAIL. Low flow loss alone never establishes a useful spin-up
distribution -- run this before enabling generative spin-up in ESMDA.

    pixi run -e dev python scripts/neural_surrogate/test_latent_generator.py \
        model_dir=model_weights/latent_generator_s \
        output_dir=model_weights/latent_generator_s/acceptance
"""

from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from neural_surrogates import generator_evaluation as ge  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

# Fixed source -> colour assignment (never cycled): real is the black
# reference, the rest follow the categorical order of the evaluation palette.
SOURCE_COLORS: dict[str, str] = {
    "real": "#0b0b0b",
    "ae_recon": "#eb6834",
    "generated": "#2a78d6",
    "generated_const_history": "#1baf7a",
    "generated_shuffled_history": "#e87ba4",
    "generated_omitted_history": "#4a3aa7",
}
SOURCE_LABELS: dict[str, str] = {
    "real": "real",
    "ae_recon": "AE recon",
    "generated": "generated (true history)",
    "generated_const_history": "generated (constant history)",
    "generated_shuffled_history": "generated (shuffled history)",
    "generated_omitted_history": "generated (omitted history)",
}
PADDING_EDGE = 16


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_generator(model_dir: Path, device: torch.device) -> tuple[Any, DictConfig]:
    """Rebuild the generator from ``config.yaml`` + strict ``weights.pt``."""
    train_cfg = OmegaConf.load(model_dir / "config.yaml")
    assert isinstance(train_cfg, DictConfig)
    dtype = getattr(torch, train_cfg.dataset.dtype)
    model = instantiate(
        train_cfg.architecture,
        n_state_channels=len(train_cfg.dataset.state_vars),
        n_params=len(train_cfg.dataset.param_vars),
    ).to(dtype=dtype)
    state = torch.load(model_dir / "weights.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    if not bool(model.latent_stats_installed):
        raise RuntimeError(f"{model_dir / 'weights.pt'} has no latent statistics")
    model.to(device).eval()
    return model, train_cfg


def _select_snapshots(
    dataset: Any, max_snapshots: int, seed: int
) -> dict[int, list[int]]:
    """``{traj: [t, ...]}`` -- up to ``max_snapshots`` anchors drawn round-robin
    over the split's trajectories (shuffled within each), so every geometry is
    represented and no trajectory group grows beyond its fair share."""
    rng = np.random.default_rng(seed)
    per_traj: dict[int, list[int]] = {}
    for traj, t in dataset.sample_index:
        per_traj.setdefault(int(traj), []).append(int(t))
    queues = {traj: list(rng.permutation(ts)) for traj, ts in per_traj.items()}
    chosen: dict[int, list[int]] = {}
    total = 0
    while total < max_snapshots and any(queues.values()):
        for traj in sorted(queues):
            if queues[traj] and total < max_snapshots:
                chosen.setdefault(traj, []).append(int(queues[traj].pop()))
                total += 1
    return {traj: sorted(ts) for traj, ts in chosen.items()}


def _load_states(
    dataset: Any, traj: int, ts: list[int], dtype: torch.dtype
) -> torch.Tensor:
    """Stack snapshots ``(len(ts), C, *grid)`` of one trajectory."""
    with xr.open_dataset(dataset._state_files[traj]) as ds:
        arr = np.stack(
            [np.asarray(ds[v].isel(time=ts).values) for v in dataset.state_vars],
            axis=1,
        )
    return torch.from_numpy(arr).to(dtype)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _expand(
    t: torch.Tensor | None, b: int, device: torch.device
) -> torch.Tensor | None:
    if t is None:
        return None
    return t.unsqueeze(0).expand(b, *t.shape).to(device)


@torch.no_grad()
def _ae_reconstruct(
    model: Any,
    state: torch.Tensor,
    geometry: torch.Tensor,
    features: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """``decode_latents(encode_latents(state))`` in ``batch_size`` chunks."""
    out = []
    for start in range(0, state.shape[0], batch_size):
        chunk = state[start : start + batch_size].to(device)
        b = chunk.shape[0]
        enc = model.encode_latents(
            chunk, _expand(geometry, b, device), _expand(features, b, device)
        )
        out.append(model.decode_latents(enc.z, enc).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def _generate(
    model: Any,
    params_hist: torch.Tensor,
    geometry: torch.Tensor,
    features: torch.Tensor | None,
    *,
    seed: int,
    num_steps: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """``sample()`` for ``params_hist`` ``(B, Hp, P)`` in ``batch_size`` chunks
    with one seeded generator (noise is a deterministic function of ``seed``
    and the chunking)."""
    gen = torch.Generator(device=device).manual_seed(int(seed))
    out = []
    for start in range(0, params_hist.shape[0], batch_size):
        hist = params_hist[start : start + batch_size].to(device)
        b = hist.shape[0]
        sample = model.sample(
            hist,
            _expand(geometry, b, device),
            _expand(features, b, device),
            generator=gen,
            num_steps=int(num_steps),
        )
        out.append(sample.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / 2**20
    # Process-lifetime high-water mark (kB on Linux): monotone, so a later
    # step count can only report >= an earlier one.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


@torch.no_grad()
def _benchmark_sampling(
    model: Any,
    params_hist_row: torch.Tensor,
    geometry: torch.Tensor,
    features: torch.Tensor | None,
    *,
    num_steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Wall time / peak memory of one ``sample()`` at the deployment shape
    (``batch_size`` members of one geometry)."""
    hist = (
        params_hist_row.unsqueeze(0)
        .expand(batch_size, *params_hist_row.shape)
        .to(device)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model.sample(
        hist,
        _expand(geometry, batch_size, device),
        _expand(features, batch_size, device),
        generator=torch.Generator(device=device).manual_seed(0),
        num_steps=int(num_steps),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    return {
        "wall_time_s": float(wall),
        "wall_time_per_member_s": float(wall / batch_size),
        "peak_memory_mb": _peak_memory_mb(device),
        "batch_size": float(batch_size),
    }


# --------------------------------------------------------------------------- #
# Metrics per group
# --------------------------------------------------------------------------- #


def _group_metrics(
    fields: np.ndarray,
    fluid: np.ndarray,
    stencil: np.ndarray,
    spacing: tuple[float, float, float],
    max_values: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Every per-stack metric of one source on one trajectory group."""
    return {
        "profiles": ge.profiles(fields, fluid),
        "spectra": ge.spectra(fields, fluid, dx=spacing[2]),
        "reynolds": ge.reynolds_stresses(fields, fluid),
        "divergence": ge.divergence(fields, stencil, spacing),
        "values": ge.fluid_values(fields, fluid, max_values, rng),
        "n": int(fields.shape[0]),
    }


class _SourceStore:
    """Per-source, per-grid-shape group results (merged at the end)."""

    def __init__(self) -> None:
        self.groups: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def add(self, source: str, grid_key: str, metrics: dict[str, Any]) -> None:
        self.groups.setdefault(source, {}).setdefault(grid_key, []).append(metrics)

    def sources(self) -> list[str]:
        return list(self.groups)

    def merged(self, source: str) -> dict[str, dict[str, Any]]:
        return {
            key: ge.merge_group_metrics(groups)
            for key, groups in self.groups[source].items()
        }

    def values(
        self, source: str, max_values: int, rng: np.random.Generator
    ) -> np.ndarray:
        chunks = [
            g["values"] for groups in self.groups[source].values() for g in groups
        ]
        return ge.pool_values(chunks, max_values, rng)

    def n(self, source: str) -> int:
        return sum(g["n"] for groups in self.groups[source].values() for g in groups)


def _weighted_scalar(per_grid: dict[str, dict[str, Any]], key: str) -> float:
    """Sample-count-weighted mean of a per-grid scalar (nan-aware)."""
    num = den = 0.0
    for m in per_grid.values():
        v, n = float(m[key]), float(m["n"])
        if np.isfinite(v):
            num += v * n
            den += n
    return num / den if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Optional rollout transients
# --------------------------------------------------------------------------- #


def _load_stepper(
    stepper_dir: Path, gen_cfg: DictConfig, device: torch.device
) -> tuple[Any, int]:
    """A trained time stepper rebuilt as ``test_neural_surrogate.py`` does; its
    parameter schema must equal the generator's (checked when recorded)."""
    cfg = OmegaConf.load(stepper_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    theirs = cfg.dataset.get("param_vars")
    ours = list(gen_cfg.dataset.param_vars)
    if theirs is not None and list(theirs) != ours:
        raise ValueError(
            f"stepper param_vars {list(theirs)} != generator param_vars {ours}"
        )
    history = int(cfg.dataset.get("num_history_steps", 1))
    overrides: dict[str, Any] = {}
    if history != 1 and "num_history_steps" not in cfg.architecture:
        overrides["num_history_steps"] = history
    model = instantiate(
        cfg.architecture,
        n_state_channels=len(cfg.dataset.state_vars),
        n_params=len(ours),
        **overrides,
    )
    model.load_state_dict(torch.load(stepper_dir / "weights.pt", map_location="cpu"))
    model.to(device).eval()
    return model, history


@torch.no_grad()
def _rollout_transients(
    stepper: Any,
    history: int,
    initial: np.ndarray,
    params: torch.Tensor,
    geometry: torch.Tensor,
    fluid: np.ndarray,
    num_steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Kinetic energy / fluctuation RMS per rollout step (mean over members)
    from the ``(B, C, *grid)`` initial states under a constant parameter
    vector. A history-conditioned stepper is cold-started by repeating the
    initial state ``H`` times (the generator emits one state, not a window)."""
    c = initial.shape[1]
    ke = np.zeros((num_steps + 1, initial.shape[0]))
    rms = np.zeros_like(ke)
    for start in range(0, initial.shape[0], batch_size):
        chunk = torch.from_numpy(initial[start : start + batch_size]).to(device)
        b = chunk.shape[0]
        geom = _expand(geometry, b, device)
        p = params.unsqueeze(0).expand(b, *params.shape).to(device)
        state = chunk.repeat(1, history, 1, 1, 1) if history > 1 else chunk
        current = chunk
        for step in range(num_steps + 1):
            f = current.float().cpu().numpy()
            ke[step, start : start + b] = ge.kinetic_energy(f, fluid)
            rms[step, start : start + b] = np.sqrt(
                np.mean(
                    (f[:, :, fluid] - f[:, :, fluid].mean(axis=2, keepdims=True)) ** 2,
                    axis=(1, 2),
                )
            )
            if step == num_steps:
                break
            nxt = stepper(state, p, geom)
            state = nxt if history == 1 else torch.cat([state[:, c:], nxt], dim=1)
            current = nxt
    return {"kinetic_energy": ke.mean(axis=1), "rms": rms.mean(axis=1)}


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def _rows_for_source(
    source: str, per_grid: dict[str, dict[str, Any]], components: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(metric: str, component: str, level: Any, value: float, grid: str) -> None:
        rows.append(
            {
                "source": source,
                "metric": metric,
                "component": component,
                "level": level,
                "grid": grid,
                "value": float(value),
            }
        )

    for grid, m in per_grid.items():
        prof = m["profiles"]
        for ci, name in enumerate(components):
            for z in range(prof["mean"].shape[1]):
                add("mean_profile", name, z, prof["mean"][ci, z], grid)
                add("rms_profile", name, z, prof["rms"][ci, z], grid)
            for ki, k in enumerate(m["spectra"]["k"]):
                add(
                    "energy_spectrum",
                    name,
                    float(k),
                    m["spectra"]["energy"][ci, ki],
                    grid,
                )
        for i, j in ge.STRESS_PAIRS:
            if i < len(components) and j < len(components):
                add(
                    "reynolds_stress",
                    f"{components[i]}{components[j]}",
                    "",
                    m["reynolds"]["stress"][i, j],
                    grid,
                )
        add("divergence_rms", "", "", m["divergence"]["rms"], grid)
        add("divergence_mean_abs", "", "", m["divergence"]["mean_abs"], grid)
        add("n_samples", "", "", m["n"], grid)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "metric", "component", "level", "grid", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _style(ax: Any) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot_profiles(
    merged: dict[str, dict[str, dict[str, Any]]], components: list[str], path: Path
) -> None:
    grids = sorted({g for per in merged.values() for g in per})
    for grid in grids:
        fig, axes = plt.subplots(
            2, len(components), figsize=(3.2 * len(components), 6.4), squeeze=False
        )
        for source, per in merged.items():
            if grid not in per:
                continue
            prof = per[grid]["profiles"]
            z = np.arange(prof["mean"].shape[1])
            for ci, name in enumerate(components):
                for r, key in enumerate(("mean", "rms")):
                    axes[r, ci].plot(
                        prof[key][ci],
                        z,
                        color=SOURCE_COLORS.get(source, "#52514e"),
                        linewidth=1.6 if source != "real" else 2.0,
                        linestyle="--" if source == "real" else "-",
                        label=SOURCE_LABELS.get(source, source),
                    )
        for ci, name in enumerate(components):
            axes[0, ci].set_title(f"{name}: conditional mean")
            axes[1, ci].set_title(f"{name}: fluctuation RMS")
            for r in range(2):
                axes[r, ci].set_ylabel("z level")
                _style(axes[r, ci])
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Vertical profiles (fluid cells), grid {grid}")
        fig.tight_layout()
        suffix = "" if len(grids) == 1 else f"_{grid.replace(' ', '')}"
        fig.savefig(path.with_name(f"{path.stem}{suffix}{path.suffix}"), dpi=120)
        plt.close(fig)


def _plot_histograms(
    hists: dict[str, dict[str, Any]],
    edges: np.ndarray,
    components: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        1, len(components), figsize=(3.6 * len(components), 3.4), squeeze=False
    )
    for ci, name in enumerate(components):
        centres = 0.5 * (edges[ci, 1:] + edges[ci, :-1])
        for source, h in hists.items():
            axes[0, ci].plot(
                centres,
                h["density"][ci],
                color=SOURCE_COLORS.get(source, "#52514e"),
                linewidth=1.6 if source != "real" else 2.0,
                linestyle="--" if source == "real" else "-",
                label=f"{SOURCE_LABELS.get(source, source)} (W1={h['w1'][ci]:.3g})",
            )
        axes[0, ci].set_title(f"{name} density (fluid cells)")
        axes[0, ci].set_xlabel(name)
        axes[0, ci].legend(fontsize=6, frameon=False)
        _style(axes[0, ci])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_spectra(
    merged: dict[str, dict[str, dict[str, Any]]], components: list[str], path: Path
) -> None:
    grids = sorted({g for per in merged.values() for g in per})
    for grid in grids:
        fig, axes = plt.subplots(
            1, len(components), figsize=(3.6 * len(components), 3.4), squeeze=False
        )
        for source, per in merged.items():
            if grid not in per:
                continue
            spec = per[grid]["spectra"]
            k, e = spec["k"][1:], spec["energy"][:, 1:]
            for ci, name in enumerate(components):
                ok = np.isfinite(e[ci]) & (e[ci] > 0)
                if ok.any():
                    axes[0, ci].loglog(
                        k[ok],
                        e[ci][ok],
                        color=SOURCE_COLORS.get(source, "#52514e"),
                        linewidth=1.6 if source != "real" else 2.0,
                        linestyle="--" if source == "real" else "-",
                        label=SOURCE_LABELS.get(source, source),
                    )
        for ci, name in enumerate(components):
            axes[0, ci].set_title(f"{name}: 1-D spectrum along x")
            axes[0, ci].set_xlabel("k [rad/m]")
            axes[0, ci].set_ylabel("E(k)")
            _style(axes[0, ci])
        axes[0, 0].legend(fontsize=6, frameon=False)
        fig.suptitle(f"Energy spectra over fully fluid rows, grid {grid}")
        fig.tight_layout()
        suffix = "" if len(grids) == 1 else f"_{grid.replace(' ', '')}"
        fig.savefig(path.with_name(f"{path.stem}{suffix}{path.suffix}"), dpi=120)
        plt.close(fig)


def _plot_divergence(scalars: dict[str, dict[str, float]], path: Path) -> None:
    names = [
        s for s in scalars if np.isfinite(scalars[s].get("divergence_rms", np.nan))
    ]
    fig, ax = plt.subplots(figsize=(1.4 * max(len(names), 3) + 2, 3.6))
    ax.bar(
        range(len(names)),
        [scalars[s]["divergence_rms"] for s in names],
        color=[SOURCE_COLORS.get(s, "#52514e") for s in names],
        width=0.6,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [SOURCE_LABELS.get(s, s) for s in names], rotation=20, ha="right", fontsize=8
    )
    ax.set_ylabel("divergence RMS [1/s]")
    ax.set_title("Divergence on stencil-valid fluid cells")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_sweep(sweep: list[dict[str, Any]], path: Path) -> None:
    steps = [s["num_steps"] for s in sweep]
    panels = [
        ("profile_rmse", "profile RMSE vs real"),
        ("w1", "mean W1 vs real"),
        ("divergence_rms", "divergence RMS"),
        ("wall_time_s", "wall time [s]"),
        ("peak_memory_mb", "peak memory [MB]"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, (key, title) in zip(axes, panels):
        ax.plot(
            steps,
            [s.get(key, np.nan) for s in sweep],
            "-o",
            color="#2a78d6",
            markersize=5,
            linewidth=1.6,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Euler steps")
        _style(ax)
    fig.suptitle("Euler step sweep (seed 0; benchmark at the deployment batch)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_rollout(transients: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for source, tr in transients.items():
        for ax, key in zip(axes, ("kinetic_energy", "rms")):
            ax.plot(
                tr[key],
                color=SOURCE_COLORS.get(source, "#52514e"),
                linewidth=1.6,
                linestyle="--" if source == "real" else "-",
                label=SOURCE_LABELS.get(source, source),
            )
    axes[0].set_title("mean kinetic energy")
    axes[1].set_title("fluctuation RMS")
    for ax in axes:
        ax.set_xlabel("rollout step")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.4g}"


def _write_report(summary: dict[str, Any], path: Path) -> None:
    acc = summary["acceptance"]
    lines = [
        "# Latent generator acceptance report",
        "",
        f"- model_dir: `{summary['model_dir']}`",
        f"- split: `{summary['split']}` from `{summary['data_root']}`",
        f"- snapshots evaluated: {summary['n_snapshots']} over "
        f"{summary['n_trajectories']} trajectories; grids: {summary['grids']}",
        f"- chosen sampling steps (primary): {summary['chosen_sampling_steps']}; "
        f"noise seeds per conditioning: {summary['num_noise_seeds']}",
        f"- Hp = {summary['param_history_steps']}, params = {summary['param_vars']}",
        f"- device: {summary['device']}",
        "",
        f"## Verdict: **{'PASS' if acc['passed'] else 'FAIL'}**",
        "",
    ]
    if acc["failures"]:
        lines += [f"- {f}" for f in acc["failures"]] + [""]
    lines += ["### Declared tolerances", ""]
    for k, v in summary["tolerances"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "| check | value | bound | passed |", "|---|---|---|---|"]
    for name, chk in summary["checks"].items():
        lines.append(
            f"| {name} | {_fmt(chk['value'])} | {_fmt(chk['bound'])} | {chk['passed']} |"
        )
    lines += ["", "### Held-out reference scales", ""]
    for k, v in summary["reference"].items():
        lines.append(f"- {k}: {_fmt(v)}")
    keys = [
        "profile_rmse",
        "rms_profile_rmse",
        "w1",
        "spectra_lsd_db",
        "reynolds_abs_err",
        "divergence_rms",
        "diversity",
    ]
    lines += ["", "## Per-source statistics (vs real)", ""]
    lines.append("| source | n | " + " | ".join(keys) + " |")
    lines.append("|---|---|" + "---|" * len(keys))
    for source, sc in summary["sources"].items():
        lines.append(
            f"| {source} | {sc.get('n', '')} | "
            + " | ".join(_fmt(sc.get(k, "")) for k in keys)
            + " |"
        )
    lines += ["", "## Conditioning sensitivity", ""]
    for k, v in summary["conditioning"].items():
        lines.append(f"- {k}: {_fmt(v) if not isinstance(v, str) else v}")
    lines += ["", "## Euler step sweep", ""]
    lines.append(
        "| steps | profile_rmse | w1 | divergence_rms | spectra_lsd_db | wall_time_s | per member | peak_memory_mb |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summary["sweep"]:
        lines.append(
            f"| {s['num_steps']} | {_fmt(s['profile_rmse'])} | {_fmt(s['w1'])} | "
            f"{_fmt(s['divergence_rms'])} | {_fmt(s['spectra_lsd_db'])} | "
            f"{_fmt(s['wall_time_s'])} | {_fmt(s['wall_time_per_member_s'])} | "
            f"{_fmt(s['peak_memory_mb'])} |"
        )
    lines += [
        "",
        f"Benchmark shape: batch {summary['sampling_batch_size']} members; memory is "
        + (
            "torch.cuda.max_memory_allocated"
            if summary["device"].startswith("cuda")
            else "the process RSS high-water mark (monotone across the sweep)"
        )
        + ".",
        "",
        "## Padding sensitivity",
        "",
    ]
    pad = summary["padding_sensitivity"]
    if isinstance(pad, str):
        lines.append(pad)
    else:
        lines += [
            f"Last {PADDING_EDGE} cells of each padded axis (the crop block the "
            "AE's padding sits in) against the rest of the domain:",
            "",
        ]
        for source, per_axis in pad.items():
            for key, regions in per_axis.items():
                grid, axis = key.split(":")
                lines.append(
                    f"- {source}, grid {grid}, axis {axis}: edge "
                    f"rms_velocity={_fmt(regions['edge']['rms_velocity'])}, "
                    f"divergence={_fmt(regions['edge']['divergence_rms'])}; interior "
                    f"rms_velocity={_fmt(regions['interior']['rms_velocity'])}, "
                    f"divergence={_fmt(regions['interior']['divergence_rms'])}"
                )
    lines += ["", "## Rollout transients", ""]
    roll = summary["rollout"]
    if isinstance(roll, str):
        lines.append(roll)
    else:
        lines.append(
            f"stepper: `{roll['stepper_model_dir']}`, {roll['num_steps']} steps"
        )
        for source, tr in roll["transients"].items():
            lines.append(
                f"- {source}: KE {_fmt(tr['kinetic_energy'][0])} -> {_fmt(tr['kinetic_energy'][-1])}, "
                f"RMS {_fmt(tr['rms'][0])} -> {_fmt(tr['rms'][-1])}"
            )
    # Per-grid figures carry the grid in their name (see _plot_profiles).
    grids = summary["grids"]
    suffixes = [""] if len(grids) == 1 else [f"_{g}" for g in grids]
    figures = (
        [f"profiles{sfx}.png" for sfx in suffixes]
        + [f"spectra{sfx}.png" for sfx in suffixes]
        + ["histograms.png", "divergence.png", "step_sweep.png"]
        + ([] if isinstance(roll, str) else ["rollout_transients.png"])
    )
    lines += ["", "Figures: " + ", ".join(figures), ""]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate the artifact; returns the summary dict (tests inspect it)."""
    OmegaConf.set_struct(cfg, False)
    model_dir = Path(cfg.model_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(cfg.divergence.get("stencil", "central")) != "central":
        raise ValueError("divergence.stencil: only 'central' is implemented")
    want = torch.device(str(cfg.device))
    device = (
        want
        if want.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device != want:
        print(f"device {want} unavailable; evaluating on {device}")
    torch.manual_seed(int(cfg.data.seed))

    model, train_cfg = _load_generator(model_dir, device)
    dtype = getattr(torch, train_cfg.dataset.dtype)
    schema = train_cfg.generator.physical_schema
    components = [str(v) for v in schema.state_vars]
    param_vars = [str(v) for v in schema.param_vars]
    spacing = (float(schema.grid.dz), float(schema.grid.dy), float(schema.grid.dx))
    primary_steps = int(train_cfg.generator.sampling.num_steps)
    hp = int(model.param_history_steps)

    root = cfg.data.root_dir or train_cfg.dataset.root_dir
    train_split = str(train_cfg.generator.data_provenance.split)
    if str(cfg.data.split) == train_split and str(root) == str(
        train_cfg.dataset.root_dir
    ):
        print(
            f"WARNING: data.split={cfg.data.split!r} is the artifact's training split"
        )
    dataset = instantiate(
        train_cfg.dataset, root_dir=str(root), split=str(cfg.data.split), dtype=dtype
    )
    selection = _select_snapshots(
        dataset, int(cfg.data.max_snapshots), int(cfg.data.seed)
    )
    n_snapshots = sum(len(ts) for ts in selection.values())
    if n_snapshots == 0:
        raise ValueError(f"split {cfg.data.split!r} under {root} has no snapshots")
    print(
        f"loaded generator from {model_dir} (Hp={hp}, primary steps={primary_steps}) on {device}; "
        f"evaluating {n_snapshots} snapshots over {len(selection)} trajectories of split "
        f"{cfg.data.split!r}"
    )

    batch_size = int(cfg.sampling.batch_size)
    n_seeds = int(cfg.sampling.num_noise_seeds)
    sweep_steps = [int(s) for s in cfg.sampling.num_steps_sweep]
    max_values = int(cfg.distribution.max_values)
    rng = np.random.default_rng(int(cfg.data.seed))

    # Conditioning probes are defined over the WHOLE selection: shuffled
    # histories are a permutation across snapshots (and trajectories), omitted
    # histories the training-set mean parameter vector.
    order = [(traj, t) for traj in sorted(selection) for t in selection[traj]]
    all_hist = torch.stack(
        [dataset.params_hist_for(traj, t) for traj, t in order]
    )  # (N, Hp, P)
    perm = rng.permutation(len(order))
    if len(order) > 1 and np.all(perm == np.arange(len(order))):
        perm = np.roll(perm, 1)
    shuffled_hist = all_hist[torch.as_tensor(perm)]
    mean_params = model.param_mean.detach().float().cpu()
    omitted_hist = (
        mean_params.reshape(1, 1, -1).expand(len(order), hp, -1).to(all_hist.dtype)
    )
    position = {pair: i for i, pair in enumerate(order)}

    store = _SourceStore()
    diversity_gen: list[np.ndarray] = []
    diversity_const: list[np.ndarray] = []
    real_spread: list[float] = []
    sweep_time: dict[int, dict[str, float]] = {}
    padding: dict[str, dict[str, Any]] = {}
    padding_applicable = False
    grids_seen: dict[str, int] = {}
    rollout_inputs: dict[str, list[tuple[np.ndarray, torch.Tensor, np.ndarray]]] = {}

    def _metrics(
        fields: np.ndarray, fluid: np.ndarray, stencil: np.ndarray
    ) -> dict[str, Any]:
        return _group_metrics(fields, fluid, stencil, spacing, max_values, rng)

    for traj, ts in selection.items():
        geometry = dataset.geometry_for(traj)
        features = dataset.geom_features_for(traj)
        fluid = geometry.numpy().astype(bool)
        stencil = ge.stencil_fluid_mask(fluid)
        grid = tuple(int(s) for s in geometry.shape)
        grid_key = "x".join(str(s) for s in grid)
        grids_seen[grid_key] = grids_seen.get(grid_key, 0) + 1
        padded_axes = tuple(p != g for p, g in zip(model._padded_shape(grid), grid))
        padding_applicable = padding_applicable or any(padded_axes)
        idx = [position[(traj, t)] for t in ts]
        hist_true = all_hist[idx]
        hist_const = hist_true[:, -1:, :].expand(-1, hp, -1).contiguous()
        print(
            f"trajectory {traj}: grid {grid_key}, {len(ts)} snapshots, padded axes (z,y,x)={padded_axes}"
        )

        real = _load_states(dataset, traj, ts, dtype)
        real_np = real.float().numpy()
        store.add("real", grid_key, _metrics(real_np, fluid, stencil))
        real_spread.append(ge.pairwise_rms_distance(real_np, fluid))

        ae = _ae_reconstruct(model, real, geometry, features, batch_size, device)
        store.add("ae_recon", grid_key, _metrics(ae, fluid, stencil))

        sources_this: dict[str, np.ndarray] = {"real": real_np, "ae_recon": ae}
        gen_seeds = []
        for s in range(n_seeds):
            gen_seeds.append(
                _generate(
                    model,
                    hist_true,
                    geometry,
                    features,
                    seed=1000 * int(cfg.data.seed) + s,
                    num_steps=primary_steps,
                    batch_size=batch_size,
                    device=device,
                )
            )
        gen_stack = np.stack(gen_seeds)  # (S, B, C, *grid)
        store.add(
            "generated",
            grid_key,
            _metrics(gen_stack.reshape(-1, *gen_stack.shape[2:]), fluid, stencil),
        )
        diversity_gen.append(ge.diversity(gen_stack, fluid)["per_condition"])
        sources_this["generated"] = gen_stack[0]

        if bool(cfg.conditioning.constant_history):
            const_seeds = [
                _generate(
                    model,
                    hist_const,
                    geometry,
                    features,
                    seed=2000 * (int(cfg.data.seed) + 1) + s,
                    num_steps=primary_steps,
                    batch_size=batch_size,
                    device=device,
                )
                for s in range(n_seeds)
            ]
            const_stack = np.stack(const_seeds)
            store.add(
                "generated_const_history",
                grid_key,
                _metrics(
                    const_stack.reshape(-1, *const_stack.shape[2:]), fluid, stencil
                ),
            )
            diversity_const.append(ge.diversity(const_stack, fluid)["per_condition"])
            sources_this["generated_const_history"] = const_stack[0]
        if bool(cfg.conditioning.shuffled_history):
            shuf = _generate(
                model,
                shuffled_hist[idx],
                geometry,
                features,
                seed=1000 * int(cfg.data.seed),
                num_steps=primary_steps,
                batch_size=batch_size,
                device=device,
            )
            store.add(
                "generated_shuffled_history", grid_key, _metrics(shuf, fluid, stencil)
            )
        if bool(cfg.conditioning.omitted_history):
            omit = _generate(
                model,
                omitted_hist[idx],
                geometry,
                features,
                seed=1000 * int(cfg.data.seed),
                num_steps=primary_steps,
                batch_size=batch_size,
                device=device,
            )
            store.add(
                "generated_omitted_history", grid_key, _metrics(omit, fluid, stencil)
            )

        # Step sweep: seed 0 per count (the primary count reuses the seed-0
        # stack); the timing benchmark runs once per count on the first group.
        for steps in sweep_steps:
            if steps == primary_steps:
                fields = gen_stack[0]
            else:
                fields = _generate(
                    model,
                    hist_true,
                    geometry,
                    features,
                    seed=1000 * int(cfg.data.seed),
                    num_steps=steps,
                    batch_size=batch_size,
                    device=device,
                )
            store.add(
                f"generated_steps{steps}", grid_key, _metrics(fields, fluid, stencil)
            )
            if steps not in sweep_time:
                sweep_time[steps] = _benchmark_sampling(
                    model,
                    hist_true[0],
                    geometry,
                    features,
                    num_steps=steps,
                    batch_size=batch_size,
                    device=device,
                )

        if any(padded_axes):
            for source, fields in sources_this.items():
                res = ge.padding_sensitivity(
                    fields, fluid, spacing, padded_axes, PADDING_EDGE
                )
                if res is not None:
                    padding.setdefault(source, {}).update(
                        {f"{grid_key}:{k}": v for k, v in res.items()}
                    )
        if bool(cfg.rollout.enabled):
            for source, fields in sources_this.items():
                rollout_inputs.setdefault(source, []).append((fields, geometry, fluid))

    # -- merge, distributions, scalars ---------------------------------------- #
    merged = {source: store.merged(source) for source in store.sources()}
    # Pool each source's value reservoir ONCE: the real pool is both the
    # histogram support and the W1 reference, and drawing it twice would give
    # "real" a non-zero distance to itself whenever the pool exceeds
    # distribution.max_values.
    values = {
        source: store.values(source, max_values, rng) for source in store.sources()
    }
    real_values = values["real"]
    edges = ge.histogram_edges(real_values, int(cfg.distribution.n_bins))
    hists = {
        source: ge.histograms_and_w1(vals, real_values, edges)
        for source, vals in values.items()
    }
    n_boot = int(cfg.bootstrap.n_resamples)
    # One bootstrap per grid shape (a profile is only defined within a grid),
    # averaged: the scale a single grid's mean profile can be pinned down to.
    boot_profile = ge.nanmean(
        [
            ge.bootstrap_profile_rmse(
                m["profiles"]["sample_mean"], n_boot, int(cfg.data.seed)
            )
            for m in merged["real"].values()
        ]
    )
    boot_w1 = ge.bootstrap_w1(real_values, min(n_boot, 50), int(cfg.data.seed))
    reference: dict[str, Any] = {
        "bootstrap_profile_rmse": float(boot_profile),
        "bootstrap_w1": ge.nanmean(boot_w1),
        "bootstrap_w1_per_component": {
            c: float(v) for c, v in zip(components, boot_w1)
        },
        "real_divergence_rms": _weighted_scalar(
            {
                k: {"divergence_rms": m["divergence"]["rms"], "n": m["n"]}
                for k, m in merged["real"].items()
            },
            "divergence_rms",
        ),
        "real_pairwise_spread": ge.nanmean(real_spread),
    }

    scalars: dict[str, dict[str, float]] = {}
    for source in store.sources():
        per_grid = {
            key: {**ge.compare_to_real(m, merged["real"][key]), "n": m["n"]}
            for key, m in merged[source].items()
            if key in merged["real"]
        }
        sc = {
            k: _weighted_scalar(per_grid, k)
            for k in (
                "profile_rmse",
                "rms_profile_rmse",
                "spectra_lsd_db",
                "reynolds_abs_err",
                "divergence_rms",
            )
        }
        sc["w1"] = ge.nanmean(hists[source]["w1"])
        for c, v in zip(components, hists[source]["w1"]):
            sc[f"w1_{c}"] = float(v)
        sc["n"] = float(store.n(source))
        scalars[source] = sc
    if diversity_gen:
        scalars["generated"]["diversity"] = ge.nanmean(np.concatenate(diversity_gen))
    if diversity_const:
        scalars["generated_const_history"]["diversity"] = ge.nanmean(
            np.concatenate(diversity_const)
        )
    scalars["real"]["diversity"] = reference["real_pairwise_spread"]

    verdict = ge.aggregate_report(scalars, reference, dict(cfg.acceptance))

    def _gain(other: str) -> float:
        if other not in scalars:
            return float("nan")
        return float(
            scalars[other]["profile_rmse"] - scalars["generated"]["profile_rmse"]
        )

    conditioning = {
        "true_history_profile_rmse": scalars["generated"]["profile_rmse"],
        "shuffled_history_profile_rmse": scalars.get(
            "generated_shuffled_history", {}
        ).get("profile_rmse", float("nan")),
        "omitted_history_profile_rmse": scalars.get(
            "generated_omitted_history", {}
        ).get("profile_rmse", float("nan")),
        "constant_history_profile_rmse": scalars.get("generated_const_history", {}).get(
            "profile_rmse", float("nan")
        ),
        "true_history_w1": scalars["generated"]["w1"],
        "shuffled_history_w1": scalars.get("generated_shuffled_history", {}).get(
            "w1", float("nan")
        ),
        "omitted_history_w1": scalars.get("generated_omitted_history", {}).get(
            "w1", float("nan")
        ),
        "profile_rmse_gain_vs_shuffled": _gain("generated_shuffled_history"),
        "profile_rmse_gain_vs_omitted": _gain("generated_omitted_history"),
        "note": "positive gain = the true history lowers the error to the real conditional statistics",
    }
    sweep: list[dict[str, Any]] = []
    for steps in sweep_steps:
        key = f"generated_steps{steps}"
        sweep.append(
            {
                "num_steps": steps,
                **{k: v for k, v in scalars[key].items() if k != "n"},
                **sweep_time[steps],
            }
        )

    rollout: Any = "disabled (rollout.enabled=false)"
    if bool(cfg.rollout.enabled):
        if cfg.rollout.stepper_model_dir is None:
            raise ValueError("rollout.enabled=true needs rollout.stepper_model_dir")
        stepper, history = _load_stepper(
            Path(str(cfg.rollout.stepper_model_dir)), train_cfg, device
        )
        transients: dict[str, dict[str, np.ndarray]] = {}
        for source, groups in rollout_inputs.items():
            runs = [
                _rollout_transients(
                    stepper,
                    history,
                    fields,
                    mean_params,
                    geometry,
                    fluid,
                    int(cfg.rollout.num_steps),
                    batch_size,
                    device,
                )
                for fields, geometry, fluid in groups
            ]
            transients[source] = {
                k: np.mean([r[k] for r in runs], axis=0)
                for k in ("kinetic_energy", "rms")
            }
        rollout = {
            "stepper_model_dir": str(cfg.rollout.stepper_model_dir),
            "num_steps": int(cfg.rollout.num_steps),
            "history_steps": history,
            "transients": transients,
        }
        _plot_rollout(transients, out_dir / "rollout_transients.png")

    # -- files ---------------------------------------------------------------- #
    rows: list[dict[str, Any]] = []
    for source in store.sources():
        rows += _rows_for_source(source, merged[source], components)
        for ci, name in enumerate(components):
            rows.append(
                {
                    "source": source,
                    "metric": "w1",
                    "component": name,
                    "level": "",
                    "grid": "",
                    "value": float(hists[source]["w1"][ci]),
                }
            )
            centres = 0.5 * (edges[ci, 1:] + edges[ci, :-1])
            for bi, centre in enumerate(centres):
                rows.append(
                    {
                        "source": source,
                        "metric": "histogram_density",
                        "component": name,
                        "level": float(centre),
                        "grid": "",
                        "value": float(hists[source]["density"][ci, bi]),
                    }
                )
        if "diversity" in scalars[source]:
            rows.append(
                {
                    "source": source,
                    "metric": "diversity",
                    "component": "",
                    "level": "",
                    "grid": "",
                    "value": scalars[source]["diversity"],
                }
            )
    for entry in sweep:
        for k in ("wall_time_s", "wall_time_per_member_s", "peak_memory_mb"):
            rows.append(
                {
                    "source": f"generated_steps{entry['num_steps']}",
                    "metric": k,
                    "component": "",
                    "level": "",
                    "grid": "",
                    "value": float(entry[k]),
                }
            )
    _write_csv(rows, out_dir / "metrics.csv")

    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "data_root": str(root),
        "split": str(cfg.data.split),
        "device": str(device),
        "n_snapshots": n_snapshots,
        "n_trajectories": len(selection),
        "snapshots": {str(traj): ts for traj, ts in selection.items()},
        "grids": sorted(grids_seen),
        "geometries": [
            {
                "trajectory": int(traj),
                "shape": list(dataset.grid_shape(traj)),
                "fluid_cells": int(dataset.geometry_for(traj).sum().item()),
            }
            for traj in selection
        ],
        "chosen_sampling_steps": primary_steps,
        "num_steps_sweep": sweep_steps,
        "num_noise_seeds": n_seeds,
        "sampling_batch_size": batch_size,
        "param_history_steps": hp,
        "param_vars": param_vars,
        "state_vars": components,
        "grid_spacing_dz_dy_dx": list(spacing),
        "sources": scalars,
        "reference": reference,
        "conditioning": conditioning,
        "sweep": sweep,
        "padding_sensitivity": (
            padding
            if padding_applicable
            else "n/a (grid is a multiple of the AE padding multiple)"
        ),
        "rollout": rollout,
        **verdict,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    plot_sources = {s: merged[s] for s in ge.SOURCE_ORDER if s in merged}
    _plot_profiles(plot_sources, components, out_dir / "profiles.png")
    _plot_histograms(
        {s: hists[s] for s in plot_sources},
        edges,
        components,
        out_dir / "histograms.png",
    )
    _plot_spectra(plot_sources, components, out_dir / "spectra.png")
    _plot_divergence({s: scalars[s] for s in plot_sources}, out_dir / "divergence.png")
    _plot_sweep(sweep, out_dir / "step_sweep.png")
    _write_report(summary, out_dir / "report.md")

    print("\n================ latent generator acceptance ================")
    for source, sc in scalars.items():
        if source.startswith("generated_steps"):
            continue
        print(
            f"{source:>28}: n={int(sc['n'])} profile_rmse={_fmt(sc['profile_rmse'])} "
            f"w1={_fmt(sc['w1'])} div_rms={_fmt(sc['divergence_rms'])} "
            f"lsd={_fmt(sc['spectra_lsd_db'])}dB diversity={_fmt(sc.get('diversity', 'n/a'))}"
        )
    for entry in sweep:
        print(
            f"steps={entry['num_steps']:>4}: profile_rmse={_fmt(entry['profile_rmse'])} "
            f"w1={_fmt(entry['w1'])} wall={_fmt(entry['wall_time_s'])}s "
            f"peak_mem={_fmt(entry['peak_memory_mb'])}MB"
        )
    for name, chk in verdict["checks"].items():
        print(
            f"  [{'ok ' if chk['passed'] else 'BAD'}] {name}: {_fmt(chk['value'])} vs {_fmt(chk['bound'])}"
        )
    print(f"ACCEPTANCE: {'PASS' if verdict['acceptance']['passed'] else 'FAIL'}")
    print(f"metrics.csv, summary.json, figures and report.md in {out_dir}")
    print("=============================================================")
    return summary


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/testing_latent_generator",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
