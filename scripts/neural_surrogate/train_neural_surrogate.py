"""Train the simple-conv neural surrogate on a training-data split.

This is a thin loop intended to validate the end-to-end stack
(dataloader → model → optimizer); the model itself is the single-conv
baseline in `neural_surrogates.architectures.SimpleConv`.

Usage:

    pixi run -e dev python scripts/train_neural_surrogate.py
    pixi run -e dev python scripts/train_neural_surrogate.py \
        dataset.root_dir=training_data/pylbm_small trainer.num_epochs=10
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Bump when the meaning of the cached arrays changes so old caches are ignored.
_NORM_STATS_VERSION = 1


def _compute_normalization_stats(
    train_ds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel state and param mean/std over the training split's fluid cells.

    Mirrors upstream UPT's dataset standardisation: the network sees zero-mean,
    unit-variance fields. Stats are streamed file-by-file (sum / sum-of-squares
    in float64) and restricted to fluid cells via each trajectory's own geometry
    mask (multi-geometry splits have per-trajectory masks and grids) so the
    masked-out obstacle zeros do not bias them.
    """
    n_state = len(train_ds.state_vars)
    s_sum = np.zeros(n_state, dtype=np.float64)
    s_sqsum = np.zeros(n_state, dtype=np.float64)
    s_count = 0
    for traj, state_path in enumerate(train_ds._state_files):
        fluid = train_ds.geometry_for(traj).cpu().numpy().astype(bool)
        with xr.open_dataset(state_path) as ds:
            for c, var in enumerate(train_ds.state_vars):
                vals = np.asarray(ds[var].values)  # (T, *grid)
                masked = vals[:, fluid]  # (T, n_fluid)
                s_sum[c] += masked.sum(dtype=np.float64)
                s_sqsum[c] += np.square(masked, dtype=np.float64).sum()
            s_count += masked.shape[0] * masked.shape[1]
    state_mean = s_sum / s_count
    state_std = np.sqrt(np.maximum(s_sqsum / s_count - state_mean**2, 0.0))

    params = torch.cat([p for p in train_ds._params], dim=0).cpu().numpy()  # (sum_T, P)
    param_mean = params.mean(axis=0)
    param_std = params.std(axis=0)
    return state_mean, state_std, param_mean, param_std


def _normalization_signature(train_ds) -> str:
    """A stable fingerprint of every input `_compute_normalization_stats` reads.

    The cached stats are only valid for the exact split, channel/param order and
    on-disk state files they were computed from. We key on the split name, the
    state/param variable tuples, the geometry variable (it selects the fluid
    mask), and each state file's ``(name, size, mtime)`` — so regenerating or
    editing the training data, or changing any of these knobs, invalidates the
    cache and forces a recompute. ``pushforward_steps`` is deliberately absent:
    the stats stream every snapshot regardless of the rollout horizon.
    """
    files = [
        [p.name, st.st_size, int(st.st_mtime)]
        for p in train_ds._state_files
        for st in (p.stat(),)
    ]
    return json.dumps(
        {
            "version": _NORM_STATS_VERSION,
            "split": train_ds.split,
            "state_vars": list(train_ds.state_vars),
            "param_names": list(train_ds.param_names),
            "geometry_var": train_ds.geometry_var,
            "files": files,
        },
        sort_keys=True,
    )


def _normalization_cache_path(train_ds) -> Path:
    """Where the cached stats for this split live, inside the data folder."""
    return Path(train_ds.root) / "normalization_stats" / f"{train_ds.split}.npz"


def _load_cached_normalization_stats(
    train_ds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return cached stats if present and still matching the data, else ``None``."""
    path = _normalization_cache_path(train_ds)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != _normalization_signature(train_ds):
                return None
            return (
                data["state_mean"],
                data["state_std"],
                data["param_mean"],
                data["param_std"],
            )
    except (OSError, KeyError, ValueError) as exc:
        # A corrupt / stale-format cache file must never break training; just
        # fall back to recomputing (which overwrites it).
        print(f"ignoring unreadable normalization cache {path}: {exc}")
        return None


def _save_normalization_stats(
    train_ds,
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Persist stats next to the training data; a write failure is non-fatal."""
    path = _normalization_cache_path(train_ds)
    state_mean, state_std, param_mean, param_std = stats
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            state_mean=state_mean,
            state_std=state_std,
            param_mean=param_mean,
            param_std=param_std,
            signature=_normalization_signature(train_ds),
        )
    except OSError as exc:
        print(f"could not cache normalization stats to {path}: {exc}")


def _get_normalization_stats(
    train_ds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached normalization stats when available, else compute and cache.

    Computing the stats streams the whole training split from disk, which is
    slow on large datasets; caching the result in the data folder makes reruns
    on the same data effectively free.
    """
    cached = _load_cached_normalization_stats(train_ds)
    if cached is not None:
        print(
            f"loaded cached normalization stats from {_normalization_cache_path(train_ds)}"
        )
        return cached
    stats = _compute_normalization_stats(train_ds)
    _save_normalization_stats(train_ds, stats)
    print(f"cached normalization stats to {_normalization_cache_path(train_ds)}")
    return stats


def _build_loader(cfg: DictConfig, dataset: Dataset, *, train: bool) -> DataLoader:
    """DataLoader for one split, honouring the optional ``batch_sampler`` block.

    Without it (``batch_sampler: null``, the default) this is the original
    ``instantiate(cfg.dataloader, ...)`` with shuffle forced off for val. With
    it, the sampler groups every batch by trajectory — required for
    multi-geometry datasets, whose grids cannot be stacked by plain shuffled
    batching — and DataLoader's own ``batch_size`` / ``shuffle`` / ``drop_last``
    must be neutralised (they are mutually exclusive with a batch sampler; the
    sampler config carries them instead). Val keeps a deterministic order.
    """
    sampler_cfg = cfg.get("batch_sampler")
    if sampler_cfg is None:
        if train:
            return instantiate(cfg.dataloader, dataset=dataset)
        return instantiate(cfg.dataloader, dataset=dataset, shuffle=False)
    overrides = {} if train else {"shuffle": False}
    sampler = instantiate(sampler_cfg, dataset=dataset, **overrides)
    return instantiate(
        cfg.dataloader,
        dataset=dataset,
        batch_sampler=sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )


def run(cfg: DictConfig) -> None:
    dtype = getattr(torch, cfg.dataset.dtype)

    train_ds = instantiate(cfg.dataset, split="train", dtype=dtype)
    val_ds = instantiate(cfg.dataset, split="val", dtype=dtype)
    # persistent_workers needs worker processes; force it off for workerless
    # runs (CPU smoke tests / debugging) so the DataLoader accepts the config.
    if int(cfg.dataloader.get("num_workers", 0)) == 0:
        cfg.dataloader.persistent_workers = False
    train_loader = _build_loader(cfg, train_ds, train=True)
    val_loader = _build_loader(cfg, val_ds, train=False)

    model = instantiate(
        cfg.architecture,
        n_state_channels=len(cfg.dataset.state_vars),
        n_params=len(train_ds.param_names),
    ).to(dtype=dtype)

    # Cross-check the SDF-feature flags: a model whose stem was widened for the
    # SDF channels must be paired with a dataset that ships them (and at the same
    # clamp radius), or training silently trains on the wrong stem. Fail loud on
    # any mismatch. Both default off, so this is a no-op for standard runs.
    model_wants_features = getattr(model, "n_geom_feature_channels", 0) > 0
    dataset_has_features = bool(getattr(train_ds, "sdf_features", False))
    if model_wants_features != dataset_has_features:
        raise ValueError(
            "SDF-feature mismatch: architecture.sdf_features="
            f"{model_wants_features} but dataset.sdf_features={dataset_has_features}. "
            "Enable (or disable) both together."
        )
    if model_wants_features and float(train_ds.sdf_clamp_cells) != float(
        getattr(model, "sdf_clamp_cells", train_ds.sdf_clamp_cells)
    ):
        raise ValueError(
            "SDF clamp mismatch: architecture.sdf_clamp_cells="
            f"{model.sdf_clamp_cells} but dataset.sdf_clamp_cells="
            f"{train_ds.sdf_clamp_cells}; they must match."
        )

    print(
        f"train pairs={len(train_ds)}  param_names={train_ds.param_names}  "
        f"n_state_channels={len(cfg.dataset.state_vars)}"
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total model parameters={num_params:,} (trainable={num_trainable:,})")

    # Optionally warm-start from previously trained weights. This loads the full
    # state dict (including any normalization buffers) before we recompute the
    # standardisation stats below, so the stats always reflect the *current*
    # training split -- the right invariant whether continuing on the same data
    # or fine-tuning on a new dataset.
    init_weights_path = cfg.get("init_weights_path")
    if init_weights_path is not None:
        init_weights_path = Path(init_weights_path)
        if not init_weights_path.exists():
            raise FileNotFoundError(
                f"init_weights_path does not exist: {init_weights_path}"
            )
        state_dict = torch.load(init_weights_path, map_location=cfg.trainer.device)
        model.load_state_dict(state_dict)
        print(f"initialized model weights from {init_weights_path}")

    # Install standardisation statistics if the architecture supports it (UPT).
    # The stats are stored as model buffers, so they are saved with the weights
    # and restored automatically at rollout/test time -- no other call site
    # needs to know about normalisation.
    if hasattr(model, "set_normalization"):
        s_mean, s_std, p_mean, p_std = _get_normalization_stats(train_ds)
        model.set_normalization(s_mean, s_std, p_mean, p_std)
        print(
            f"normalization set:\n"
            f"  state_mean={np.round(s_mean, 4)} state_std={np.round(s_std, 4)}\n"
            f"  param_mean={np.round(p_mean, 4)} param_std={np.round(p_std, 4)}"
        )

    out_dir = Path("model_weights") / cfg.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    trainer = instantiate(
        cfg.trainer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=instantiate(cfg.optimizer, params=model.parameters()),
        loss_fn=instantiate(cfg.loss),
        weights_path=out_dir / "weights.pt",
    )
    trainer.fit()
    print(f"config and best weights saved to {out_dir}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/training",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
