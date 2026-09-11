"""Conditional latent flow matching for generative spin-up (plan 07).

Trains a ``TadpoleLatentGenerator`` -- a flow-matching velocity network in the
latent space of a FROZEN, pre-trained ``TadpoleAE`` -- on a
``SnapshotHistoryDataset`` split and writes a self-contained export. Mirrors
``pretrain_autoencoder.py`` (datasets -> model -> normalization -> config.yaml ->
``fit()``) with the plan-07 additions:

* the AE export's ``config.yaml`` is the single source of truth for the frozen
  representation: its ``dataset.state_vars`` must equal ours and its
  ``sdf_features`` / ``sdf_clamp_cells`` are inherited by the dataset (an
  explicit override must agree);
* the physical conditioning contract -- ordered ``param_vars``, ``Hp``, the
  saved cadence, units, mask convention, coordinate order, grid and the
  supported training geometries -- is REQUIRED and recorded verbatim under
  ``generator.physical_schema`` so deployment can never substitute another
  (each supported geometry carries the sha256 of its fluid mask, so a
  relocation of the same obstacles cannot pass as a trained one, and the mask
  convention must be the canonical ``MASK_CONVENTION`` polarity);
* ``dataset.constant_prehistory`` is a claim about the DATA, so it is verified
  against the corpus' own ``config.yaml`` (the constant-forcing spin-up must
  cover the repeated plateau) and the verdict recorded in the artifact;
* the latent-attention budget is checked BEFORE training (naive attention
  allocates ``B * heads * N * N`` latent tokens; a physical-cell budget alone
  does not bound it);
* the raw latent mean/std are estimated once over a representative, seeded
  subset of the train split, installed as model buffers and cached to
  ``latent_stats.pt`` keyed on their provenance -- reused on resume, never
  recomputed from already-normalised latents.

Artifacts land in ``<paths.output_dir>/<model_name>/``: ``config.yaml`` (the
``architecture`` node stamped ``skip_pretrained_load: true`` /
``pretrained_ae_dir: null`` with the resolved ``ae_kwargs`` inline, plus the
``generator:`` block), best-val ``weights.pt`` (the FULL state dict incl. the
frozen ``ae.*`` weights and every normalisation buffer), ``checkpoint.pt``,
``metrics.csv`` and ``latent_stats.pt``. Deployment rebuilds from
``config.yaml`` + ``weights.pt`` alone -- the script verifies that reload
strictly after ``fit()``.

    pixi run -e dev python scripts/neural_surrogate/train_latent_generator.py \
        pretrained_ae_dir=model_weights/tadpole_ae_s \
        dataset.root_dir=training_data/pyudales_idealized \
        'dataset.param_vars=[inflow_angle,velocity_magnitude,pressure_gradient_magnitude]' \
        physical_metadata.boundary_conditions='...' model_name=latent_generator_s
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from neural_surrogates.generative_spinup import MASK_CONVENTION, geometry_fingerprint
from neural_surrogates.sdf import normalize_sdf_mode
from neural_surrogates.training.data_utils import build_loader, get_normalization_stats
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

# Bump when the meaning of the cached latent statistics changes.
_LATENT_STATS_VERSION = 1

# On-disk spatial dim name -> canonical axis. The corpora write cell-centred
# coordinates either plainly (pylbm / the surrogate's own output) or with the
# backend's ``*t`` suffix (pyudales), and `_to_regular_grid` renames the latter
# onto the former at deploy time.
_CELL_CENTRE_DIMS = {"z": "z", "y": "y", "x": "x", "zt": "z", "yt": "y", "xt": "x"}


def _plain(node: Any) -> Any:
    """OmegaConf container -> plain python (resolved); passthrough otherwise."""
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _require(cfg: DictConfig, key: str, what: str) -> Any:
    """``cfg[key]`` or a clear error when it is absent / ``???`` / null."""
    try:
        value = OmegaConf.select(cfg, key, throw_on_missing=True)
    except MissingMandatoryValue:
        value = None
    if value is None:
        raise ValueError(f"{key} must be set ({what}).")
    return value


# --------------------------------------------------------------------------- #
# Physical metadata / grid
# --------------------------------------------------------------------------- #


def _validated_physical_metadata(cfg: DictConfig) -> dict:
    """The ``physical_metadata`` block as a plain dict, every field filled.

    Source files carry no units or conventions, so the config must (plan 07
    §1): a missing block, a ``???`` inside it, or a ``units`` map that lacks a
    state or parameter variable all fail here, before any data is read.
    """
    node = cfg.get("physical_metadata")
    if node is None:
        raise ValueError(
            "physical_metadata must be set (units, geometry_mask_convention, "
            "coordinate_order, boundary_conditions, constant_forcing_notes)."
        )
    try:
        meta = OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        raise ValueError(
            f"physical_metadata has an unfilled '???' entry: {exc}. Every field "
            "is recorded in the generator artifact and must be stated explicitly."
        ) from exc
    assert isinstance(meta, dict)
    for key in (
        "units",
        "geometry_mask_convention",
        "coordinate_order",
        "boundary_conditions",
        "constant_forcing_notes",
    ):
        if key not in meta or meta[key] is None:
            raise ValueError(f"physical_metadata.{key} must be set.")
    units = meta["units"]
    if not isinstance(units, dict):
        raise ValueError("physical_metadata.units must be a {variable: unit} map.")
    needed = list(cfg.dataset.state_vars) + list(cfg.dataset.param_vars)
    missing = [v for v in needed if v not in units or units[v] in (None, "")]
    if missing:
        raise ValueError(
            f"physical_metadata.units lacks an entry for {missing}; every state "
            "and parameter variable needs a unit."
        )
    order = [str(a) for a in meta["coordinate_order"]]
    if order != ["z", "y", "x"]:
        raise ValueError(
            f"physical_metadata.coordinate_order must be [z, y, x] (the axis "
            f"order of every (C, *grid) tensor here), got {order}."
        )
    # The mask polarity is not a free-text note: deployment applies exactly
    # MASK_CONVENTION, so an artifact may not claim a different one (it would
    # invert the fluid mask at sampling time).
    convention = str(meta["geometry_mask_convention"])
    if convention != MASK_CONVENTION:
        raise ValueError(
            f"physical_metadata.geometry_mask_convention must be exactly "
            f"{MASK_CONVENTION!r} (the one polarity the generator and the "
            f"generative spin-up speak), got {convention!r}."
        )
    return meta


def _grid_metadata(state_path: Path, state_var: str, order: Sequence[str]) -> dict:
    """``{nz, ny, nx, dz, dy, dx, bounds, dims, first_center}`` off a state file.

    Read from the FIRST state variable's spatial coordinates -- not from the
    training-data ``config.yaml`` ``domain`` block, which for a random-geometry
    corpus is the generation *template* (one nominal domain), not the grid any
    trajectory was actually run on. Spacing is the median coordinate step
    (1.0 for index-only coordinates); ``bounds`` are the cell EDGES (centres
    +- half a spacing), ordered ``[[x0, x1], [y0, y1], [z0, z1]]`` as in the
    training-data domain block.

    The file's spatial dims must appear in ``order`` (the validated
    ``physical_metadata.coordinate_order``): every ``(C, *grid)`` tensor in this
    stack is positional, so a corpus stored ``(x, y, z)`` would be trained,
    recorded and deployed with silently transposed axes. Cell-centre dims are
    accepted in both the canonical (``z``/``y``/``x``) and the backend ``*t``
    (``zt``/``yt``/``xt``) spelling, exactly as
    ``NeuralSurrogateForwardModel._to_regular_grid`` renames them.
    """
    with xr.open_dataset(state_path) as ds:
        da = ds[state_var]
        dims = [str(d) for d in da.dims if d != "time"]
        if len(dims) != 3:
            raise ValueError(
                f"{state_path.name}: state var {state_var!r} has spatial dims "
                f"{dims}; expected exactly three (z, y, x)."
            )
        canonical = [_CELL_CENTRE_DIMS.get(d) for d in dims]
        if canonical != [str(a) for a in order]:
            raise ValueError(
                f"{state_path.name}: state var {state_var!r} has spatial dims "
                f"{dims}, which is not the declared "
                f"physical_metadata.coordinate_order {[str(a) for a in order]} "
                f"(cell-centre dims may also be spelled "
                f"{sorted(set(_CELL_CENTRE_DIMS) - set('zyx'))}). Every tensor "
                "here is positional, so a transposed corpus would train and "
                "deploy on silently swapped axes."
            )
        sizes = [int(da.sizes[d]) for d in dims]
        spacing: list[float] = []
        first: list[float] = []
        for d, n in zip(dims, sizes):
            if d in ds.coords and n > 1:
                c = np.asarray(ds[d].values, dtype=np.float64)
                spacing.append(float(np.median(np.diff(c))))
                first.append(float(c[0]))
            elif d in ds.coords:
                first.append(float(np.asarray(ds[d].values, dtype=np.float64)[0]))
                spacing.append(1.0)
            else:
                first.append(0.0)
                spacing.append(1.0)
    (nz, ny, nx), (dz, dy, dx), (z0, y0, x0) = sizes, spacing, first
    bounds = [
        [x0 - 0.5 * dx, x0 + (nx - 0.5) * dx],
        [y0 - 0.5 * dy, y0 + (ny - 0.5) * dy],
        [z0 - 0.5 * dz, z0 + (nz - 0.5) * dz],
    ]
    return {
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "dz": dz,
        "dy": dy,
        "dx": dx,
        "bounds": bounds,
        # Provenance for the numbers above: on-disk dim names + first centres.
        "dims": dims,
        "first_center": [z0, y0, x0],
    }


def _verified_prehistory(train_ds: Any, root: Path) -> Optional[dict]:
    """Provenance gate for ``dataset.constant_prehistory`` (plan 07 §1).

    Repeating the first recorded parameter row for the missing leading history
    is only legitimate when the corpus really was forced at those values before
    its first save. That is a claim about the DATA, so it is checked against the
    corpus' own ``config.yaml``: the constant-forcing spin-up must be at least
    as long as the plateau it stands in for, ``(Hp - 1) * history_dt_seconds``.
    Returns the record written to ``generator.data_provenance``; ``None`` when
    the flag is off (anchors then start at ``t = Hp-1`` and no history is
    invented).
    """
    if not train_ds.constant_prehistory:
        return None
    hp = int(train_ds.param_history_steps)
    required = (hp - 1) * float(train_ds.history_dt_seconds)
    time_block = _training_data_provenance(root).get("time") or {}
    spinup = time_block.get("spinup_time")
    if spinup is None:
        raise ValueError(
            f"dataset.constant_prehistory=true needs the corpus' own "
            f"{root / 'config.yaml'} to record time.spinup_time: the repeated "
            "leading history is only valid if the forcing really was constant "
            "at the first saved values before the first save, and nothing else "
            "in the corpus states that. Set constant_prehistory=false or "
            "regenerate the data with its config."
        )
    spinup = float(spinup)
    if spinup + 1e-9 < required:
        raise ValueError(
            f"dataset.constant_prehistory=true requires a constant-forcing "
            f"spin-up at least as long as the repeated plateau: "
            f"time.spinup_time={spinup:g} s < (Hp - 1) * history_dt_seconds = "
            f"({hp} - 1) * {train_ds.history_dt_seconds:g} = {required:g} s. The "
            "invented rows would reach back before the constant forcing began."
        )
    # State and parameter times are validated element-wise by the dataset, so
    # one number describes both; across trajectories they must agree too, or
    # 'the first saved time' is not a single, auditable instant.
    firsts = sorted({round(float(t[0]), 9) for t in train_ds._times})
    if len(firsts) != 1:
        raise ValueError(
            f"dataset.constant_prehistory=true requires one common first saved "
            f"time across the split, got {firsts}; the spin-up duration cannot "
            "vouch for the plateau of every trajectory otherwise."
        )
    return {
        "spinup_time": spinup,
        "required_seconds": required,
        "first_saved_time": firsts[0],
    }


def _training_data_provenance(root: Path) -> dict:
    """``domain`` / ``time`` blocks of the corpus' ``config.yaml`` (if any) --
    recorded as provenance only (see :func:`_grid_metadata`)."""
    path = root / "config.yaml"
    if not path.exists():
        return {}
    data_cfg: Any = OmegaConf.load(path)
    out: dict = {}
    for key in ("domain", "time"):
        block = data_cfg.get(key)
        if block is not None:
            out[key] = _plain(block)
    return out


# --------------------------------------------------------------------------- #
# AE cross-checks
# --------------------------------------------------------------------------- #


def _resolve_ae_inherited_dataset_settings(cfg: DictConfig, ae_cfg: DictConfig) -> None:
    """Inherit ``sdf_features`` / ``sdf_clamp_cells`` from the AE export and
    require ``dataset.state_vars`` to equal the AE's.

    The frozen encoder was pre-trained on exactly the AE's geometry channels
    and state ordering; a different dataset would feed it out-of-distribution
    inputs while every tensor shape still matched (strict weight loading
    checks tensors, not physical contracts -- plan 07 §1)."""
    ae_arch: Any = ae_cfg.get("architecture")
    if ae_arch is None:
        raise ValueError("the AE export's config.yaml has no 'architecture' node")
    ae_ds = ae_cfg.get("dataset")
    ae_state_vars = None if ae_ds is None else ae_ds.get("state_vars")
    if ae_state_vars is None:
        raise ValueError(
            "the AE export's config.yaml has no dataset.state_vars; cannot verify "
            "the generator dataset's ordered state variables against the AE."
        )
    ours = [str(v) for v in cfg.dataset.state_vars]
    theirs = [str(v) for v in ae_state_vars]
    if ours != theirs:
        raise ValueError(
            f"state_vars mismatch: dataset.state_vars={ours} but the AE export "
            f"{cfg.pretrained_ae_dir} was pre-trained on {theirs}; the ordered "
            "state variables must match the frozen AE exactly."
        )

    ae_sdf = normalize_sdf_mode(ae_arch.get("sdf_features", "none"))
    ae_clamp = float(ae_arch.get("sdf_clamp_cells", 32.0))
    ours_sdf = cfg.dataset.get("sdf_features")
    if ours_sdf is None:
        cfg.dataset.sdf_features = ae_sdf
    elif normalize_sdf_mode(ours_sdf) != ae_sdf:
        raise ValueError(
            f"sdf_features mismatch: dataset.sdf_features={ours_sdf!r} but the AE "
            f"export uses {ae_sdf!r}; leave it null to inherit from the AE."
        )
    ours_clamp = cfg.dataset.get("sdf_clamp_cells")
    if ours_clamp is None:
        cfg.dataset.sdf_clamp_cells = ae_clamp
    elif ae_sdf != "none" and float(ours_clamp) != ae_clamp:
        raise ValueError(
            f"sdf_clamp_cells mismatch: dataset.sdf_clamp_cells={ours_clamp} but "
            f"the AE export uses {ae_clamp}; leave it null to inherit from the AE."
        )


def _check_attention_budget(model: Any, ds: Any, loader: Any, split: str) -> None:
    """Refuse the run if any trajectory's batch exceeds ``max_latent_tokens``.

    ``B`` is the per-trajectory batch the loader will actually form (the
    ``TrajectoryBatchSampler``'s cell-budgeted size, else the DataLoader's), and
    ``N = prod(latent_grid_for(grid))`` the latent tokens per sample -- the
    same product ``velocity()`` re-checks per call, evaluated here for every
    grid up front so the failure comes before an epoch is burned. Run on BOTH
    splits: validation forwards the same velocity net, and a val-only grid that
    blows the budget would otherwise surface only at the first epoch's end."""
    budget = model.max_latent_tokens
    if budget is None:
        return
    sampler = loader.batch_sampler
    from torch.utils.data import BatchSampler

    custom = sampler is not None and not isinstance(sampler, BatchSampler)
    worst: tuple[int, tuple[int, ...], int, int] | None = None
    for traj in range(len(ds._state_files)):
        grid = tuple(ds.grid_shape(traj))
        b = int(sampler._batch_size_for(traj)) if custom else int(loader.batch_size)
        n = int(math.prod(model.latent_grid_for(grid)))
        if worst is None or b * n > worst[0]:
            worst = (b * n, grid, b, n)
    assert worst is not None
    tokens, grid, b, n = worst
    if tokens > budget:
        raise ValueError(
            f"latent attention budget exceeded before training on the {split} "
            f"split: grid {grid} gives {n} latent tokens per sample x batch {b} "
            f"= {tokens} > architecture.max_latent_tokens={budget} (naive "
            "attention allocates B*heads*N*N). Reduce the batch size / "
            "cell_budget or the domain, or raise the budget after profiling."
        )
    print(
        f"latent attention budget OK ({split}): worst grid {grid} -> {b} x {n} = "
        f"{tokens} tokens <= {budget}"
    )


# --------------------------------------------------------------------------- #
# Latent statistics
# --------------------------------------------------------------------------- #


def _latent_stats_loader(cfg: DictConfig, train_ds: Any, seed: int) -> Any:
    """A seeded, shuffled loader over the train split for the statistics pass.

    Shuffled so ``max_batches`` batches sample the corpus rather than the first
    trajectory in file order (a multi-geometry split's first trajectory is one
    geometry); seeded so the subset -- and thus the cached statistics -- is
    reproducible and recorded in the cache provenance."""
    sampler_cfg = cfg.get("batch_sampler")
    if sampler_cfg is not None:
        sampler = instantiate(sampler_cfg, dataset=train_ds, shuffle=True, seed=seed)
        return instantiate(
            cfg.dataloader,
            dataset=train_ds,
            batch_sampler=sampler,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
    return instantiate(
        cfg.dataloader,
        dataset=train_ds,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _latent_stats_provenance(cfg: DictConfig, model: Any, train_ds: Any) -> dict:
    """Everything the cached latent statistics are only valid for."""
    return {
        "version": _LATENT_STATS_VERSION,
        "ae_fingerprint": model.ae_fingerprint,
        "spatial_mode": model.spatial_mode,
        "encoder_crop_size": int(model.encoder_crop_size),
        "halo_size": int(model.halo_size),
        "root_dir": str(Path(train_ds.root).resolve()),
        "split": str(train_ds.split),
        "state_vars": list(train_ds.state_vars),
        "sdf_features": str(train_ds.sdf_feature_mode),
        "sdf_clamp_cells": float(train_ds.sdf_clamp_cells),
        "time_stride": int(train_ds.time_stride),
        "precision": "fp32",
        "latent_mode": "mode",
        "max_batches": _plain(cfg.latent_stats.max_batches),
        "seed": int(cfg.latent_stats.seed),
    }


def _install_latent_stats(
    cfg: DictConfig, model: Any, trainer: Any, train_ds: Any, out_dir: Path
) -> dict:
    """Install the raw-latent statistics, from the cache on resume or freshly.

    On ``trainer.resume`` a cache whose provenance matches is reused verbatim
    (the checkpoint's buffers will agree with it); a cache that does NOT match
    while a checkpoint exists is refused, because ``fit()`` would then restore
    the checkpoint's stale buffers over freshly computed ones without a
    trace. Otherwise the statistics are computed through the trainer's own
    batch preparation (same upload / geometry cache the objective uses) and
    written to ``latent_stats.pt``.
    """
    cache_path = out_dir / "latent_stats.pt"
    provenance = _latent_stats_provenance(cfg, model, train_ds)
    ckpt_exists = (out_dir / "checkpoint.pt").exists()
    if trainer.resume and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("provenance") == provenance:
            model.set_latent_normalization(cached["mean"], cached["std"])
            print(f"reusing cached latent stats from {cache_path}")
            return provenance
        if ckpt_exists:
            raise RuntimeError(
                f"{cache_path} was computed under a different provenance "
                f"({cached.get('provenance')} != {provenance}) and a checkpoint "
                "exists in the same directory; refusing to resume with mismatched "
                "latent statistics. Use a fresh model_name or fix the config."
            )
        print(f"latent stats cache {cache_path} is stale; recomputing")
    elif trainer.resume and ckpt_exists:
        raise RuntimeError(
            f"resume=true and {out_dir / 'checkpoint.pt'} exists but "
            f"{cache_path} is missing; the checkpoint's latent buffers cannot be "
            "audited. Restore latent_stats.pt or start a fresh model_name."
        )

    loader = _latent_stats_loader(cfg, train_ds, int(cfg.latent_stats.seed))
    max_batches = cfg.latent_stats.max_batches
    mean, std = model.compute_latent_normalization(
        trainer.prepared_batches(loader),
        max_batches=None if max_batches is None else int(max_batches),
    )
    torch.save(
        {"provenance": provenance, "mean": mean.cpu(), "std": std.cpu()}, cache_path
    )
    print(f"cached latent stats to {cache_path}")
    return provenance


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _stamp_export_config(
    cfg: DictConfig,
    model: Any,
    train_ds: Any,
    val_ds: Any,
    ae_dir: Path,
    physical: dict,
    provenance: dict,
    prehistory: Optional[dict],
) -> None:
    """Rewrite ``cfg`` into the self-contained artifact config (in place)."""
    cfg.architecture.skip_pretrained_load = True
    cfg.architecture.pretrained_ae_dir = None
    cfg.architecture.ae_kwargs = model.ae_kwargs
    # Resolved width (null -> D rounded up); idempotent on reload.
    cfg.architecture.hidden_size = int(model.hidden_size)
    # Stamped like hidden_size: both shape the velocity net's tensors, so the
    # deploy rebuild must not depend on the defaults of the day.
    cfg.architecture.mlp_ratio = int(model.mlp_ratio)
    cfg.architecture.normalize = bool(model.normalize)
    cfg.dataset.param_vars = list(train_ds.param_names)
    cfg.dataset.state_vars = list(train_ds.state_vars)

    root = Path(train_ds.root)
    grid = _grid_metadata(
        train_ds._state_files[0], train_ds.state_vars[0], physical["coordinate_order"]
    )
    # Shape + fluid-cell count do not identify a geometry (relocating the
    # obstacles preserves both), so each entry also carries the mask's hash.
    supported = [
        {
            "shape": [int(s) for s in g.shape],
            "fluid_cells": int(g.sum().item()),
            "mask_sha256": geometry_fingerprint(g),
        }
        for g in train_ds._geometries
    ]
    cfg.generator = {
        "physical_schema": {
            "state_vars": list(train_ds.state_vars),
            "param_vars": list(train_ds.param_names),
            "param_history_steps": int(train_ds.param_history_steps),
            "history_dt_seconds": float(train_ds.history_dt_seconds),
            "units": {
                k: physical["units"][k]
                for k in list(train_ds.state_vars) + list(train_ds.param_names)
            },
            "geometry_mask_convention": str(physical["geometry_mask_convention"]),
            "coordinate_order": [str(a) for a in physical["coordinate_order"]],
            "grid": grid,
            "supported_geometries": supported,
            "boundary_conditions": str(physical["boundary_conditions"]),
            "constant_forcing_notes": str(physical["constant_forcing_notes"]),
        },
        "ae_fingerprint": model.ae_fingerprint,
        "ae_dir": str(ae_dir),
        "sampling": {"num_steps": int(model.num_sampling_steps)},
        "data_provenance": {
            "root_dir": str(root),
            "split": str(train_ds.split),
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "constant_prehistory": bool(train_ds.constant_prehistory),
            # What the constant_prehistory claim was checked against (null when
            # the flag is off); see _verified_prehistory.
            "verified_prehistory": prehistory,
            "cadence_rtol": float(train_ds.cadence_rtol),
            "training_data_config": _training_data_provenance(root),
        },
        "latent_stats": {
            "max_batches": provenance["max_batches"],
            "seed": provenance["seed"],
            "n_channels": int(model.working_latent_dim),
        },
    }


def _verify_export_reloads(out_dir: Path, n_state: int, n_params: int) -> None:
    """Rebuild from ``config.yaml`` + ``weights.pt`` alone (no AE dir, no data)."""
    saved = OmegaConf.load(out_dir / "config.yaml")
    fresh = instantiate(saved.architecture, n_state_channels=n_state, n_params=n_params)
    state = torch.load(out_dir / "weights.pt", map_location="cpu")
    fresh.load_state_dict(state, strict=True)
    if not bool(fresh.latent_stats_installed):
        raise RuntimeError(
            f"{out_dir / 'weights.pt'} reloaded without installed latent statistics"
        )
    print(f"export verified: {out_dir} rebuilds strictly without the AE dir")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(cfg: DictConfig) -> Any:
    """Train and export; returns the trainer (tests inspect it)."""
    OmegaConf.set_struct(cfg, False)

    # -- required inputs, checked before any data is touched ----------------- #
    ae_dir = Path(str(_require(cfg, "pretrained_ae_dir", "a TadpoleAE export dir")))
    _require(cfg, "model_name", "the export dir name")
    _require(cfg, "dataset.root_dir", "the training-data root")
    param_vars = _require(cfg, "dataset.param_vars", "the ordered conditioning schema")
    if len(list(param_vars)) == 0:
        raise ValueError("dataset.param_vars must list at least one parameter.")
    physical = _validated_physical_metadata(cfg)
    ae_cfg_path = ae_dir / "config.yaml"
    if not ae_cfg_path.exists():
        raise FileNotFoundError(f"AE export config not found: {ae_cfg_path}")
    ae_cfg = OmegaConf.load(ae_cfg_path)
    assert isinstance(ae_cfg, DictConfig)
    _resolve_ae_inherited_dataset_settings(cfg, ae_cfg)

    # -- data ----------------------------------------------------------------- #
    dtype = getattr(torch, cfg.dataset.dtype)
    train_ds = instantiate(cfg.dataset, split="train", dtype=dtype)
    val_ds = instantiate(cfg.dataset, split="val", dtype=dtype)
    if int(cfg.dataloader.get("num_workers", 0)) == 0:
        cfg.dataloader.persistent_workers = False
    for name, ds in (("train", train_ds), ("val", val_ds)):
        shapes = {tuple(g.shape) for g in ds._geometries}
        if len(shapes) > 1 and cfg.get("batch_sampler") is None:
            raise ValueError(
                f"the {name} split mixes grid shapes {sorted(shapes)}; a plain "
                "shuffled DataLoader cannot stack them. Set the batch_sampler "
                "block (TrajectoryBatchSampler) so every batch comes from one "
                "trajectory."
            )
    train_loader = build_loader(cfg, train_ds, train=True)
    val_loader = build_loader(cfg, val_ds, train=False)
    if list(train_ds.param_names) != list(val_ds.param_names):
        raise ValueError(
            f"param order differs between splits: train {train_ds.param_names} "
            f"vs val {val_ds.param_names}"
        )
    print(
        f"train anchors={len(train_ds)}  val anchors={len(val_ds)}  "
        f"param_names={train_ds.param_names}  Hp={train_ds.param_history_steps}  "
        f"history_dt={train_ds.history_dt_seconds:.4g}s  "
        f"geometries={len(train_ds._geometries)}"
    )
    prehistory = _verified_prehistory(train_ds, Path(train_ds.root))
    if prehistory is not None:
        print(f"constant_prehistory verified against the corpus: {prehistory}")

    # -- model ---------------------------------------------------------------- #
    model = instantiate(
        cfg.architecture,
        n_state_channels=len(train_ds.state_vars),
        n_params=len(train_ds.param_names),
        pretrained_ae_dir=str(ae_dir),
    ).to(dtype=dtype)
    print(
        f"TadpoleLatentGenerator: D={model.state_latent_dim} "
        f"hidden={model.hidden_size} layers={model.n_layers} "
        f"spatial_mode={model.spatial_mode!r} "
        f"trainable parameters={model.count_parameters():,} "
        f"(total incl. frozen AE={model.count_parameters(trainable_only=False):,})"
    )

    # Param stats only: the frozen AE owns the state statistics (they travel in
    # its weights.pt), so set_normalization ignores s_mean / s_std by design.
    s_mean, s_std, p_mean, p_std = get_normalization_stats(train_ds)
    model.set_normalization(s_mean, s_std, p_mean, p_std)
    print(
        f"param normalization set: mean={np.round(p_mean, 4)} std={np.round(p_std, 4)}"
    )
    # The physical meaning of the params_hist columns, carried on the model
    # itself so sample()/velocity() can re-check a caller's claim (deployment
    # re-installs the same schema from the exported physical_schema).
    model.set_conditioning_schema(
        train_ds.param_names, float(train_ds.history_dt_seconds)
    )

    _check_attention_budget(model, train_ds, train_loader, "train")
    _check_attention_budget(model, val_ds, val_loader, "val")

    # -- trainer (before the stats pass: it shares the batch preparation) ----- #
    out_dir = Path(cfg.paths.output_dir) / cfg.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainer = instantiate(
        cfg.trainer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=instantiate(cfg.optimizer, params=trainable),
        loss_fn=torch.nn.MSELoss(),
        weights_path=out_dir / "weights.pt",
    )

    provenance = _install_latent_stats(cfg, model, trainer, train_ds, out_dir)

    # -- export config first, so a killed run still leaves a loadable schema --- #
    _stamp_export_config(
        cfg, model, train_ds, val_ds, ae_dir, physical, provenance, prehistory
    )
    OmegaConf.save(cfg, out_dir / "config.yaml")

    trainer.fit()

    _verify_export_reloads(out_dir, len(train_ds.state_vars), len(train_ds.param_names))
    print(f"config, best weights, checkpoint, metrics and latent stats in {out_dir}")
    return trainer


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/train_latent_generator",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
