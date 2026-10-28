"""Helpers shared by the scripts/surrogate/evaluate_*.py scripts.

Loading a trained model and its data, reading states, sampling and
reconstructing with a latent generator, and the one figure type they all use:
rows of horizontal slices (truth, prediction, error, ...). Fields are
(..., C, z, y, x) arrays with C the state variables (u, v, w).
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def device(name: str) -> torch.device:
    """The requested device, or the CPU when CUDA is not available."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; evaluating on the CPU")
        return torch.device("cpu")
    return torch.device(name)


def load_model(
    model_dir: str | pathlib.Path,
    dev: torch.device,
    periodic_y: bool = False,
    **arch_kwargs: Any,
) -> tuple[Any, DictConfig]:
    """A trained model rebuilt from `config.yaml` + `weights.pt`, in eval mode.

    `periodic_y` makes it y-periodic at inference
    (neural_surrogates.periodic.make_periodic_y)."""
    model_dir = pathlib.Path(model_dir)
    cfg = OmegaConf.load(model_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    if cfg.dataset.get("param_vars") is not None:  # steppers and the generator
        arch_kwargs.setdefault("n_params", len(cfg.dataset.param_vars))
    model = instantiate(
        cfg.architecture, n_state_channels=len(cfg.dataset.state_vars), **arch_kwargs
    )
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu"))
    if periodic_y:
        from neural_surrogates.periodic import make_periodic_y

        model = make_periodic_y(model)
    return model.to(dev).eval(), cfg


def open_dataset(
    train_cfg: DictConfig, data_dir: str | None, split: str, **overrides: Any
) -> Any:
    """The model's training dataset class on `split` (of `data_dir`, if given)."""
    if data_dir is not None:
        overrides["root_dir"] = data_dir
    dtype = getattr(torch, train_cfg.dataset.dtype)
    return instantiate(train_cfg.dataset, split=split, dtype=dtype, **overrides)


def load_states(
    dataset: Any, traj: int, times: Sequence[int] | None = None
) -> np.ndarray:
    """(T, C, z, y, x) states of one trajectory (all frames, or `times`)."""
    return read_states(dataset._state_files[traj], dataset.state_vars, times)


def read_states(
    path: str | pathlib.Path,
    state_vars: Sequence[str],
    times: Sequence[int] | None = None,
) -> np.ndarray:
    """(T, C, z, y, x) float32 states of one state file (all frames, or `times`)."""
    with xarray.open_dataset(path) as ds:
        if times is not None:
            ds = ds.isel(time=list(times))
        return np.stack([ds[v].values for v in state_vars], axis=1).astype(np.float32)


def load_latent_generator(
    model_dir: str | pathlib.Path,
    dev: torch.device,
    max_internal_batchsize: int | None = None,
) -> tuple[Any, DictConfig]:
    """A trained latent generator with its conditioning schema reinstalled;
    `max_internal_batchsize` caps the crops its frozen AE runs at once."""
    model, train_cfg = load_model(model_dir, dev)
    if max_internal_batchsize is not None:
        model.ae.ae.max_internal_batchsize = int(max_internal_batchsize)
    schema = train_cfg.generator.physical_schema
    model.set_conditioning_schema(
        list(schema.param_vars), float(schema.history_dt_seconds)
    )
    return model, train_cfg


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def ae_reconstruct(
    model: Any,
    real: np.ndarray,
    data: Any,
    traj: int,
    batch: int,
    dev: torch.device,
    decode: tuple[str | None, int | None] = (None, None),
) -> np.ndarray:
    """Real fields through a latent generator's frozen AE; `decode` = (spatial
    mode, halo size) of the decoder, `None` for the trained ones."""
    out = []
    for start in range(0, len(real), batch):
        state = torch.from_numpy(real[start : start + batch]).to(dev)
        b = len(state)
        enc = model.encode_latents(
            state,
            expand(data.geometry_for(traj), b, dev),
            expand(data.geom_features_for(traj), b, dev),
        )
        recon = model.decode_latents(
            enc.z, enc, spatial_mode=decode[0], halo_size=decode[1]
        )
        out.append(recon.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def encoded_latent_moments(
    model: Any, fields: np.ndarray, data: Any, traj: int, batch: int, dev: torch.device
) -> list[dict]:
    """Token moments (`generator_evaluation.latent_moments`) of the normalised
    state latents the frozen encoder gives `fields`, one dict per batch."""
    from neural_surrogates import generator_evaluation as ge

    out = []
    for start in range(0, len(fields), batch):
        state = torch.from_numpy(fields[start : start + batch]).to(dev)
        b = len(state)
        enc = model.encode_latents(
            state,
            expand(data.geometry_for(traj), b, dev),
            expand(data.geom_features_for(traj), b, dev),
        )
        out.append(ge.latent_moments(enc.z.float().cpu().numpy()))
    return out


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def generator_samples(
    model: Any,
    hist: torch.Tensor,
    data: Any,
    traj: int,
    seed: int,
    num_steps: int,
    batch: int,
    dev: torch.device,
    sampler: str = "euler",
    decode: tuple[str | None, int | None] = (None, None),
    latents: list[dict] | None = None,
    noise: torch.Tensor | None = None,
) -> np.ndarray:
    """Generated fields for parameter histories (B, Hp, P) on trajectory
    `traj`'s geometry. The noise is fixed by `seed` (one generator over the
    batches), or given per sample as `noise` (B, D, Zl, Yl, Xl). `decode` =
    (spatial mode, halo size) of the decoder, `None` for the trained ones.
    `latents`, if given, gets the token moments of the sampled latents
    (`generator_evaluation.latent_moments`), one dict per batch."""
    from neural_surrogates import generator_evaluation as ge

    generator = (
        None
        if noise is not None
        else torch.Generator(device=dev).manual_seed(int(seed))
    )
    out = []
    for start in range(0, len(hist), batch):
        h = hist[start : start + batch].to(dev)
        b = len(h)
        sample = model.sample(
            h,
            expand(data.geometry_for(traj), b, dev),
            expand(data.geom_features_for(traj), b, dev),
            initial_noise=None if noise is None else noise[start : start + b],
            generator=generator,
            num_steps=int(num_steps),
            param_names=model.param_names,
            history_dt_seconds=model.history_dt_seconds,
            sampler=sampler,
            decode_spatial_mode=decode[0],
            decode_halo_size=decode[1],
            return_latents=latents is not None,
        )
        if latents is not None:
            sample, z = sample
            latents.append(ge.latent_moments(z.float().cpu().numpy()))
        out.append(sample.float().cpu().numpy())
    return np.concatenate(out)


def inlet_turbulence(settings: Any, dataset: Any, traj: int) -> Any:
    """The inlet-turbulence forcing of one trajectory (an `InletTurbulence`,
    called as `forcing(pred, frame)`), or None when `settings.enabled` is off."""
    if not settings.enabled:
        return None
    from neural_surrogates.inlet_turbulence import InletTurbulence

    seed = None if settings.seed is None else int(settings.seed)
    return InletTurbulence.from_dataset(dataset, traj, seed=seed)


def grid_spacing(dataset: Any, traj: int) -> tuple[float, float, float]:
    """(dz, dy, dx) of one trajectory's grid, from its cell-centre coordinates."""
    with xarray.open_dataset(dataset._state_files[traj]) as ds:
        dims = [d for d in ds[dataset.state_vars[0]].dims if d != "time"]
        dz, dy, dx = (float(np.median(np.diff(ds[d].values))) for d in dims)
    return dz, dy, dx


def expand(
    t: torch.Tensor | None, batch: int, dev: torch.device
) -> torch.Tensor | None:
    """Repeat a per-trajectory tensor (geometry, features) over a batch."""
    return None if t is None else t.unsqueeze(0).expand(batch, *t.shape).to(dev)


def speed(fields: np.ndarray) -> np.ndarray:
    """|U| from (..., C, z, y, x) fields: the norm over the C axis."""
    return np.sqrt((np.asarray(fields) ** 2).sum(axis=-4))


def plot_slice_rows(
    rows: Sequence[tuple[str, Sequence[np.ndarray], bool]],
    column_titles: Sequence[str],
    fluid: np.ndarray,
    path: pathlib.Path,
    label: str = "|U| [m/s]",
) -> None:
    """A grid of 2-D (y, x) slices: one row per entry, one column per title.

    Each row is (name, slices, is_error). Value rows share one colour scale;
    error rows get their own. Building cells (`fluid` False) are blanked.
    """
    values = [s for _, slices, err in rows if not err for s in slices]
    errors = [s for _, slices, err in rows if err for s in slices]
    vlim = (np.nanmin(values), np.nanmax(values)) if values else (0.0, 1.0)
    elim = np.nanmax([np.nanmax(np.abs(e)) for e in errors]) if errors else 1.0
    n_rows, n_cols = len(rows), len(column_titles)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows), squeeze=False
    )
    for r, (name, slices, is_error) in enumerate(rows):
        for c, field in enumerate(slices):
            masked = np.where(fluid[c] if fluid.ndim == 3 else fluid, field, np.nan)
            if is_error:
                im = axes[r, c].imshow(
                    masked, origin="lower", cmap="RdBu_r", vmin=-elim, vmax=elim
                )
            else:
                im = axes[r, c].imshow(
                    masked, origin="lower", vmin=vlim[0], vmax=vlim[1]
                )
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(column_titles[c], fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=9)
        fig.colorbar(
            im, ax=axes[r, :].tolist(), shrink=0.8, label="error" if is_error else label
        )
    # White, not the style's transparent background: read in a dark viewer.
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(rows: Sequence[dict], path: pathlib.Path) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Any, path: pathlib.Path) -> None:
    def native(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        raise TypeError(type(x))

    path.write_text(json.dumps(data, indent=2, default=native))
