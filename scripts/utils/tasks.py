"""How scripts/surrogate/train.py builds each model (the config's `task`).

Each task builds the train/val datasets and the model, and returns them in a
`Setup` together with anything extra the trainer needs and what to do just
before and after fitting. `TASKS` maps a config's `task` to its builder:

  stepper           a next-step model from `architecture`
  autoencoder       the field autoencoder (+ optional discriminator)
  latent_generator  a flow-matching generator on a frozen autoencoder
  dft               a stepper built around a pretrained autoencoder, trained
                    through LoRA adapters plus `trainable_modules`
  finetune_stepper  a pretrained stepper, trained fully or through LoRA
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Callable

import numpy as np
import torch
import xarray
from hydra.utils import instantiate
from neural_surrogates.datasets.latent_cache import (
    LatentCacheDataset,
    load_latent_stats,
    prepare_latent_cache,
)
from neural_surrogates.finetuning import (
    inject_lora,
    merge_to_state_dict,
    resolve_target_modules,
    save_adapter,
)
from neural_surrogates.generative_spinup import geometry_fingerprint
from neural_surrogates.training.data_utils import (
    build_loader,
    get_normalization_stats,
    get_param_normalization_stats,
)
from omegaconf import DictConfig, OmegaConf


@dataclasses.dataclass
class Setup:
    """What a task builds; `run` does the rest."""

    model: torch.nn.Module
    train: Any
    val: Any
    trainer_kwargs: dict = dataclasses.field(default_factory=dict)
    before_fit: Callable[[Any], None] | None = None  # gets the trainer
    after_fit: Callable[[Any], None] | None = None  # gets the trainer


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _stepper(cfg: DictConfig, out_dir: pathlib.Path) -> Setup:
    # The dataset ships the geometry features the architecture takes.
    cfg.dataset.sdf_features = cfg.architecture.get("sdf_features", "none")
    cfg.dataset.sdf_clamp_cells = cfg.architecture.get(
        "sdf_clamp_cells", cfg.dataset.sdf_clamp_cells
    )
    train, val, source = _prechunked_datasets(cfg)
    model = _build_stepper(cfg, train)
    if cfg.init_weights is not None:
        model.load_state_dict(torch.load(cfg.init_weights, map_location="cpu"))
    if hasattr(model, "set_normalization"):
        model.set_normalization(*get_normalization_stats(train, source_root=source))
    return Setup(model, train, val)


def _autoencoder(cfg: DictConfig, out_dir: pathlib.Path) -> Setup:
    # A prechunked copy is only read: config.yaml keeps the source root_dir.
    root = prechunked_root(cfg)
    train, val = _datasets(cfg) if root is None else _datasets(cfg, root_dir=root)
    model = instantiate(cfg.architecture, n_state_channels=len(train.state_vars))
    model.set_normalization(*get_normalization_stats(train))
    weights = cfg.loss_weights
    kwargs: dict = {
        "kl_weight": weights.kl_weight,
        "geometry_recon_weight": weights.geometry_recon_weight,
    }
    if cfg.discriminator is not None:
        # An adversarial term: the critic gets its own optimizer instance.
        branch = getattr(model, "geometry_branch", None) is not None
        discriminator = instantiate(
            cfg.discriminator,
            size=cfg.discriminator.get("size", cfg.architecture.size),
            n_state_channels=model.n_state_channels,
            encode_geometry=model.encode_geometry or branch,
            sdf_features="none" if branch else model.sdf_feature_mode,
        )
        kwargs.update(
            discriminator=discriminator,
            disc_optimizer=instantiate(
                cfg.optimizer, params=discriminator.parameters()
            ),
            adv_weight=weights.adv_weight,
            adv_start_step=weights.adv_start_step,
            adv_ramp_steps=weights.adv_ramp_steps,
            adaptive_adv_weight=weights.adaptive_adv_weight,
            disc_recon_threshold=weights.disc_recon_threshold,
        )

    def export_encoder_decoder(trainer: Any) -> None:
        # Separate files the latent generator and the DFT stepper load, cut
        # from weights.pt so they always hold the same (best) weights.
        if not (out_dir / "weights.pt").exists():
            return  # no finite validation loss yet, so no best weights
        state = torch.load(out_dir / "weights.pt", map_location="cpu")
        for prefix, file in [
            ("ae.encoder.", "encoder.pt"),
            ("ae.decoder.", "decoder.pt"),
            ("geometry_branch.", "geometry_branch.pt"),
        ]:
            part = {
                k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
            }
            if part:
                torch.save(part, out_dir / file)

    def export_with_best_weights(trainer: Any) -> None:
        # Re-export whenever new best weights are written, so a run killed at
        # the time limit still leaves them for the latent generator and DFT.
        # A resumed run first re-cuts them from the best weights on disk.
        export_encoder_decoder(trainer)
        write_best_val = trainer._write_best_val

        def write_and_export(best_val: float) -> None:
            write_best_val(best_val)
            export_encoder_decoder(trainer)

        trainer._write_best_val = write_and_export

    return Setup(
        model,
        train,
        val,
        kwargs,
        before_fit=export_with_best_weights,
        after_fit=export_encoder_decoder,
    )


def _latent_generator(cfg: DictConfig, out_dir: pathlib.Path) -> Setup:
    ae_dir = pathlib.Path(cfg.autoencoder_dir)
    model, train, val = _latent_generator_inputs(cfg)
    root = _latent_cache(cfg, model, train, val)

    # The saved config must rebuild the model without the autoencoder dir.
    cfg.architecture.update(
        skip_pretrained_load=True,
        pretrained_ae_dir=None,
        ae_kwargs=model.ae_kwargs,
        hidden_size=int(model.hidden_size),
        mlp_ratio=int(model.mlp_ratio),
    )
    cfg.dataset.param_vars = list(train.param_names)
    cfg.generator = _generator_block(cfg, model, train, ae_dir)

    if root is not None:
        # Exact latent statistics of the whole train split, from the cache.
        model.set_latent_normalization(*load_latent_stats(root))
        cfg.dataloader.collate_fn = None  # cached items stack as they are
        return Setup(
            model,
            LatentCacheDataset(root, train),
            LatentCacheDataset(root, val),
            {"_recursive_": False, "_convert_": "all"},
            # The seed the latent_statistics pass would set before fitting.
            before_fit=lambda trainer: torch.manual_seed(int(cfg.latent_stats.seed)),
        )

    def latent_statistics(trainer: Any) -> None:
        # Normalization of the encoder's latents, estimated on training batches.
        torch.manual_seed(int(cfg.latent_stats.seed))
        loader = build_loader(cfg, train, train=True)
        model.compute_latent_normalization(
            trainer.prepared_batches(loader),
            max_batches=int(cfg.latent_stats.max_batches),
        )

    return Setup(
        model,
        train,
        val,
        {"_recursive_": False, "_convert_": "all"},
        before_fit=latent_statistics,
    )


def _dft(cfg: DictConfig, out_dir: pathlib.Path) -> Setup:
    ae_dir = pathlib.Path(cfg.pretrained_dir)
    _inherit_dataset(cfg, OmegaConf.load(ae_dir / "config.yaml"))
    train, val, source = _prechunked_datasets(cfg)
    cfg.architecture.pretrained_ae_dir = str(ae_dir)
    # Keep the autoencoder's state normalization unless asked to recompute it.
    cfg.architecture.require_ae_state_stats = not cfg.recompute_normalization
    model = _build_stepper(cfg, train)
    if cfg.recompute_normalization:
        model.set_normalization(*get_normalization_stats(train, source_root=source))
    else:  # the state stats would stream the whole split only to be dropped
        model.set_normalization(None, None, *get_param_normalization_stats(train))

    peft = _with_lora(model, cfg.lora)
    for name, p in peft.named_parameters():  # the new modules train fully
        path = "." + name.replace("base_model.model.", "") + "."
        if any(f".{module}." in path for module in cfg.trainable_modules):
            p.requires_grad_(True)
    # The saved config rebuilds the stepper from weights.pt alone.
    cfg.architecture.skip_pretrained_load = True
    cfg.architecture.pretrained_ae_dir = None
    return Setup(
        peft,
        train,
        val,
        {"weights_transform": merge_to_state_dict},
        after_fit=lambda trainer: _save_lora(trainer, peft, out_dir),
    )


def prepare_latents(cfg: DictConfig) -> str | None:
    """Make (or validate) the latent cache, without training."""
    return _latent_cache(cfg, *_latent_generator_inputs(cfg))


def _finetune_stepper(cfg: DictConfig, out_dir: pathlib.Path) -> Setup:
    pretrained_dir = pathlib.Path(cfg.pretrained_dir)
    pretrained = OmegaConf.load(pretrained_dir / "config.yaml")
    _inherit_dataset(cfg, pretrained)
    cfg.architecture = pretrained.architecture
    train, val, source = _prechunked_datasets(cfg)
    model = _build_stepper(cfg, train)
    model.load_state_dict(torch.load(pretrained_dir / "weights.pt", map_location="cpu"))
    if cfg.recompute_normalization:
        model.set_normalization(*get_normalization_stats(train, source_root=source))
    if cfg.method == "full":
        return Setup(model, train, val)
    peft = _with_lora(model, cfg.lora)
    return Setup(
        peft,
        train,
        val,
        {"weights_transform": merge_to_state_dict},
        after_fit=lambda trainer: _save_lora(trainer, peft, out_dir),
    )


TASKS: dict[str, Callable[[DictConfig, pathlib.Path], Setup]] = {
    "stepper": _stepper,
    "autoencoder": _autoencoder,
    "latent_generator": _latent_generator,
    "dft": _dft,
    "finetune_stepper": _finetune_stepper,
}


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _datasets(cfg: DictConfig, **overrides: Any) -> tuple[Any, Any]:
    dtype = getattr(torch, cfg.dataset.dtype)
    return (
        instantiate(cfg.dataset, split="train", dtype=dtype, **overrides),
        instantiate(cfg.dataset, split="val", dtype=dtype, **overrides),
    )


def _prechunked_datasets(cfg: DictConfig) -> tuple[Any, Any, str | None]:
    """Train/val data, read from the `prechunk` copy when there is one, and
    the source root it was copied from (None without a copy). The copy holds
    only the states: params stay in the source, and the source's big chunks
    stream the whole split faster for the normalization stats."""
    root = prechunked_root(cfg)
    if root is None:
        return (*_datasets(cfg), None)
    source = str(cfg.dataset.root_dir)
    return (*_datasets(cfg, root_dir=root, param_root=source), source)


def prechunked_root(cfg: DictConfig) -> str | None:
    """The re-chunked copy of `dataset.root_dir` (`prechunk` block), made or
    validated first; None without one."""
    prechunk = cfg.get("prechunk")
    if prechunk is None or prechunk.output_root is None:
        return None
    from neural_surrogates.datasets.rechunk import prepare_rechunked_dataset

    return str(
        prepare_rechunked_dataset(
            cfg.dataset.root_dir,
            prechunk.output_root,
            spatial_chunks=prechunk.spatial_chunks,
        )
    )


def _latent_generator_inputs(cfg: DictConfig) -> tuple[Any, Any, Any]:
    """The generator around its frozen autoencoder, and the train/val data."""
    ae_dir = pathlib.Path(cfg.autoencoder_dir)
    ae = OmegaConf.load(ae_dir / "config.yaml")
    # The frozen encoder only understands the autoencoder's inputs.
    cfg.dataset.state_vars = list(ae.dataset.state_vars)
    cfg.dataset.sdf_features = ae.architecture.sdf_features
    cfg.dataset.sdf_clamp_cells = ae.architecture.sdf_clamp_cells
    cfg.dataset.dtype = "float32"  # the frozen encoder runs in fp32
    train, val = _datasets(cfg)
    model = instantiate(
        cfg.architecture,
        n_state_channels=len(train.state_vars),
        n_params=len(train.param_names),
        pretrained_ae_dir=str(ae_dir),
    )
    # The frozen autoencoder keeps its own state statistics.
    model.set_normalization(None, None, *get_param_normalization_stats(train))
    model.set_conditioning_schema(train.param_names, float(train.history_dt_seconds))
    return model, train, val


def _latent_cache(cfg: DictConfig, model: Any, train: Any, val: Any) -> str | None:
    """The cache of the frozen encoder's latents (`latent_cache` block), made
    or validated first; None without one."""
    cache = cfg.get("latent_cache")
    if cache is None or cache.output_root is None:
        return None
    return str(
        prepare_latent_cache(
            model,
            {"train": train, "val": val},
            cache.output_root,
            device=cfg.trainer.device,
        )
    )


def _build_stepper(cfg: DictConfig, train: Any) -> Any:
    """A stepper sized to the data; its inputs are recorded in cfg."""
    history = int(train.num_history_steps)
    cfg.dataset.param_vars = list(train.param_names)
    cfg.dataset.num_history_steps = history
    cfg.architecture.num_history_steps = history
    return instantiate(
        cfg.architecture,
        n_state_channels=len(train.state_vars),
        n_params=len(train.param_names),
    )


def _inherit_dataset(cfg: DictConfig, pretrained: Any) -> None:
    """Use the pretrained model's inputs: variables, geometry features, history."""
    for key in ("state_vars", "sdf_features", "sdf_clamp_cells"):
        cfg.dataset[key] = pretrained.dataset.get(key)
    if cfg.dataset.get("param_vars") is None:
        cfg.dataset.param_vars = pretrained.dataset.get("param_vars")
    if cfg.dataset.get("num_history_steps") is None:
        # An autoencoder's data has no history: then one step.
        cfg.dataset.num_history_steps = pretrained.dataset.get("num_history_steps") or 1


def _with_lora(model: torch.nn.Module, lora: DictConfig) -> Any:
    """Freeze `model` and wrap it with trainable low-rank adapters."""
    model.requires_grad_(False)
    targets = resolve_target_modules(
        model, preset=lora.target_preset, target_modules=lora.target_modules
    )
    return inject_lora(
        model,
        rank=lora.rank,
        alpha=lora.alpha,
        dropout=lora.dropout,
        target_modules=targets,
        variant=lora.variant,
        modules_to_save=list(lora.modules_to_save) or None,
    )


def _save_lora(trainer: Any, peft: Any, out_dir: pathlib.Path) -> None:
    """Save the adapters and the best weights merged into the base model."""
    save_adapter(peft, out_dir / "adapter")
    if trainer.restored_best_weights:
        torch.save(merge_to_state_dict(peft), out_dir / "weights.pt")
        (out_dir / "best_val.json").write_text(
            json.dumps({"best_val": float(trainer.best_val)})
        )


def _generator_block(
    cfg: DictConfig, model: Any, train: Any, ae_dir: pathlib.Path
) -> dict:
    """What the generative spin-up checks before it uses the generator.

    The variables, units and parameter history it was trained on, the grid,
    and a fingerprint of every training geometry (it refuses unseen ones).
    """
    meta = cfg.physical_metadata
    variables = [*train.state_vars, *train.param_names]
    geometries = {}
    for i, path in enumerate(train._state_files):
        mask = train.geometry_for(i)
        entry = {
            "shape": [int(s) for s in mask.shape],
            "fluid_cells": int(mask.sum().item()),
            "mask_sha256": geometry_fingerprint(mask),
            "grid": _grid(path, train.state_vars[0]),
        }
        geometries[json.dumps(entry, sort_keys=True)] = entry
    return {
        "physical_schema": {
            "state_vars": list(train.state_vars),
            "param_vars": list(train.param_names),
            "param_history_steps": int(train.param_history_steps),
            "history_dt_seconds": float(train.history_dt_seconds),
            "units": {v: meta.units[v] for v in variables},
            "coordinate_order": list(meta.coordinate_order),
            "geometry_mask_convention": meta.geometry_mask_convention,
            "boundary_conditions": meta.notes,
            "constant_forcing_notes": "",
            "grid": _grid(train._state_files[0], train.state_vars[0]),
            "supported_geometries": list(geometries.values()),
        },
        "ae_fingerprint": model.ae_fingerprint,
        "ae_dir": str(ae_dir),
        "sampling": {"num_steps": int(model.num_sampling_steps)},
    }


def _grid(path: pathlib.Path, var: str) -> dict:
    """Size, spacing and cell-edge bounds of a training state's (z, y, x) grid."""
    with xarray.open_dataset(path) as ds:
        dims = [str(d) for d in ds[var].dims if d != "time"]
        sizes = [int(ds.sizes[d]) for d in dims]
        coords = [np.asarray(ds[d].values, dtype=float) for d in dims]
    spacing = [float(np.median(np.diff(c))) if c.size > 1 else 1.0 for c in coords]
    first = [float(c[0]) for c in coords]
    (nz, ny, nx), (dz, dy, dx), (z0, y0, x0) = sizes, spacing, first
    return {
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "dz": dz,
        "dy": dy,
        "dx": dx,
        "bounds": [
            [x0 - dx / 2, x0 + (nx - 0.5) * dx],
            [y0 - dy / 2, y0 + (ny - 0.5) * dy],
            [z0 - dz / 2, z0 + (nz - 0.5) * dz],
        ],
        "dims": dims,
        "first_center": first,
    }
