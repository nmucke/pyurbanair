"""Shared fixtures for the plan-07 latent generator (model, trainer, deploy tests).

Not a test module (no ``test_`` prefix): it fabricates the three inputs every
plan-07 test needs and is imported as ``tests._latent_generator_fixtures`` by
``test_tadpole_latent_flow.py``, ``test_latent_generator_training.py`` and the
deployment / evaluation tests.

* :func:`make_ae_export` -- a ``pretrain_autoencoder.py``-shaped ``TadpoleAE``
  export (``weights.pt`` + ``config.yaml``) for one spatial mode x geometry
  path, with the zero-init heads randomised so parity checks cannot pass
  vacuously and non-trivial state statistics installed.
* :func:`write_history_dataset` -- a tiny ``SnapshotHistoryDataset`` corpus:
  ``state/<split>/sample_XXXX.nc`` (``u, v, w`` + ``blanking``, a ``time``
  coordinate at a constant 5 s cadence, ``z/y/x`` index coordinates) paired
  with ``param/<split>/sample_XXXX.nc`` (two time-varying parameters) and a
  training-data ``config.yaml`` carrying ``domain`` / ``time`` blocks like the
  real corpora (mirrors ``_write_transition_dataset`` in
  ``test_ae_to_timestepper.py``).
* :func:`compose_generator_cfg` / :func:`train_tiny_generator` -- compose
  ``conf/neural_surrogate/train_latent_generator.yaml`` shrunk to CPU smoke
  shapes (AE size S, crop 16, grid 16x16x32, Hp=3, P=2, 2 epochs, batch 2) and
  run ``scripts/neural_surrogate/train_latent_generator.py::run``.

Callers gate on the vendored Tadpole runtime deps themselves
(``pytest.importorskip("diffusers")`` / ``timm`` / ``einops``) before
importing this module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_WORKTREE = Path(__file__).resolve().parents[1]
CONF_DIR = _WORKTREE / "conf"
SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "train_latent_generator.py"

STATE_VARS: tuple[str, ...] = ("u", "v", "w")
PARAM_VARS: tuple[str, ...] = ("inflow_angle", "velocity_magnitude")
UNITS = {
    "u": "m/s",
    "v": "m/s",
    "w": "m/s",
    "inflow_angle": "deg",
    "velocity_magnitude": "m/s",
}
C = len(STATE_VARS)
P = len(PARAM_VARS)
CROP = 16
GRID: tuple[int, int, int] = (16, 16, 32)
HP = 3
T_LEN = 6
DT = 5.0
SPLITS = {"train": 2, "val": 1}

# The tiny velocity net used throughout: D = 768 (3 channels x Cl=256 for size
# S) stays the token width (hidden_size=None -> D); only depth / heads shrink.
NET: dict[str, Any] = dict(n_layers=1, num_heads=2, film_hidden=8, time_embed_dim=8)


# --------------------------------------------------------------------------- #
# AE export
# --------------------------------------------------------------------------- #


def ae_arch(spatial_mode: str, geometry: str, latent_type: str = "mode") -> dict:
    """The ``architecture`` node of a fabricated AE export."""
    arch: dict[str, Any] = {
        "_target_": "neural_surrogates.TadpoleAE",
        "size": "S",
        "encoder_crop_size": CROP,
        "latent_type": latent_type,
        "normalize": True,
        "sdf_clamp_cells": 8.0,
        "spatial_mode": spatial_mode,
        "halo_size": 16,
    }
    if geometry == "fold":
        arch.update(encode_geometry=True, sdf_features="both", geometry_branch=None)
    elif geometry == "branch":
        arch.update(
            encode_geometry=False,
            sdf_features="sdf",
            geometry_branch={"width": 8},
        )
    else:
        raise ValueError(f"geometry must be 'fold' or 'branch', got {geometry!r}")
    return arch


def make_ae_export(
    root: Path,
    spatial_mode: str = "local",
    geometry: str = "fold",
    *,
    latent_type: str = "mode",
) -> Path:
    """Fabricate a ``pretrain_autoencoder.py``-shaped AE export under
    ``root/model_weights/ae_<spatial_mode>_<geometry>`` and return its path.

    Zero-init heads (the decoder's output projection and, in branch mode, the
    geometry projections) are randomised so representation-parity checks are
    non-trivial; non-trivial state statistics are installed."""
    from neural_surrogates import TadpoleAE

    arch = ae_arch(spatial_mode, geometry, latent_type)
    kwargs = {k: v for k, v in arch.items() if k != "_target_"}
    ae = TadpoleAE(n_state_channels=C, **kwargs)
    with torch.no_grad():
        raw: Any = ae.ae
        torch.nn.init.normal_(
            raw.decoder.transformer_decoder.final_layer.out_proj.weight, std=0.05
        )
        if geometry == "branch":
            projections = (
                list(raw.encoder.conv_encoder.geom_proj)
                + list(raw.decoder.conv_decoder.geom_proj)
                + [raw.decoder.geom_latent_proj]
            )
            for proj in projections:
                torch.nn.init.normal_(proj.weight, std=0.05)
    ae.set_normalization([0.1, -0.2, 0.3], [1.0, 1.5, 0.7])

    model_dir = root / "model_weights" / f"ae_{spatial_mode}_{geometry}"
    model_dir.mkdir(parents=True)
    torch.save(ae.state_dict(), model_dir / "weights.pt")
    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": arch,
                "dataset": {
                    "root_dir": str(root / "ae_data"),
                    "state_vars": list(STATE_VARS),
                    "sdf_features": arch["sdf_features"],
                    "sdf_clamp_cells": 8.0,
                },
            }
        ),
        model_dir / "config.yaml",
    )
    return model_dir


# --------------------------------------------------------------------------- #
# History dataset
# --------------------------------------------------------------------------- #


def write_history_dataset(
    root: Path,
    *,
    splits: dict[str, int] | None = None,
    grid: tuple[int, int, int] = GRID,
    grids: Sequence[tuple[int, int, int]] | None = None,
    t_len: int = T_LEN,
    dt: float = DT,
    seed: int = 0,
) -> Path:
    """Write a tiny ``SnapshotHistoryDataset`` corpus under ``root``; returns it.

    ``splits`` maps split name -> number of trajectories (default
    ``{"train": 2, "val": 1}``). ``grids`` (one shape per train trajectory,
    cycled over the other splits) builds a multi-geometry corpus; ``grid``
    alone gives a single-geometry one. Every trajectory has ``t_len`` saves at
    a constant ``dt`` cadence, an obstacle block in ``blanking`` (1 = obstacle)
    and two time-varying parameters in ``PARAM_VARS`` order.
    """
    splits = dict(SPLITS if splits is None else splits)
    rng = np.random.default_rng(seed)
    times = np.arange(t_len, dtype="f8") * dt
    for split, n in splits.items():
        state_dir = root / "state" / split
        param_dir = root / "param" / split
        state_dir.mkdir(parents=True, exist_ok=True)
        param_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            shape = grid if grids is None else tuple(grids[i % len(grids)])
            nz, ny, nx = shape
            blank = np.zeros(shape, "f4")
            blank[: nz // 2, ny // 4 : ny // 2, nx // 4 : nx // 2] = 1.0  # one block
            state: dict[str, Any] = {
                v: (
                    ("time", "z", "y", "x"),
                    rng.standard_normal((t_len, nz, ny, nx)).astype("f4"),
                )
                for v in STATE_VARS
            }
            state["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                state,
                coords=dict(
                    time=times,
                    z=np.arange(nz, dtype="f8"),
                    y=np.arange(ny, dtype="f8"),
                    x=np.arange(nx, dtype="f8"),
                ),
            ).to_netcdf(state_dir / f"sample_{i:04d}.nc")
            xr.Dataset(
                {
                    "inflow_angle": (("time",), rng.uniform(-30.0, 30.0, t_len)),
                    "velocity_magnitude": (("time",), rng.uniform(2.0, 6.0, t_len)),
                },
                coords=dict(time=times),
            ).to_netcdf(param_dir / f"sample_{i:04d}.nc")
    nz, ny, nx = grid
    OmegaConf.save(
        OmegaConf.create(
            {
                "domain": {
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "bounds": [[0.0, float(nx)], [0.0, float(ny)], [0.0, float(nz)]],
                },
                "time": {
                    "simulation_time": float(t_len * dt),
                    "output_frequency": float(dt),
                    "spinup_time": 0.0,
                },
            }
        ),
        root / "config.yaml",
    )
    return root


# --------------------------------------------------------------------------- #
# Config + script
# --------------------------------------------------------------------------- #


def load_run() -> Callable[[DictConfig], Any]:
    """``train_latent_generator.py::run`` imported by path (scripts/ is not a
    package); returns the trainer."""
    spec = importlib.util.spec_from_file_location("train_latent_generator_ut", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run  # type: ignore[no-any-return]


def shrink_for_cpu(cfg: DictConfig) -> None:
    """CPU smoke shapes: tiny velocity net, 2 epochs, batch 2, no sampler."""
    OmegaConf.set_struct(cfg, False)
    for k, v in NET.items():
        cfg.architecture[k] = v
    cfg.architecture.num_sampling_steps = 2
    cfg.architecture.param_history_steps = HP
    cfg.batch_sampler = None
    cfg.dataloader.batch_size = 2
    cfg.dataloader.drop_last = False
    cfg.dataloader.num_workers = 0
    cfg.dataloader.pin_memory = False
    cfg.latent_stats.max_batches = 2
    cfg.trainer.num_epochs = 2
    cfg.trainer.device = "cpu"
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.patience = None
    cfg.trainer.lr_warmup_epochs = None
    cfg.trainer.cudnn_benchmark = False
    cfg.trainer.tf32 = False
    cfg.trainer.resume = False
    cfg.trainer.checkpoint_every = 1


def compose_generator_cfg(
    ae_dir: Path,
    data_dir: Path,
    out_root: Path,
    *,
    model_name: str = "latent_generator_test",
    overrides: Sequence[str] = (),
    values: Mapping[str, Any] | None = None,
    **dotted: Any,
) -> DictConfig:
    """Compose ``train_latent_generator.yaml`` for the fixture inputs, shrunk for
    CPU. ``overrides`` are Hydra CLI-style strings applied at compose time;
    dotted-key values (``values={"trainer.num_epochs": 1}``, or the same as
    ``**{"trainer.num_epochs": 1}``) are applied after the shrink so they win
    over it. ``values`` is the mypy-clean spelling of ``**dotted``."""
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(
            config_name="neural_surrogate/train_latent_generator",
            overrides=[
                f"pretrained_ae_dir={ae_dir}",
                f"dataset.root_dir={data_dir}",
                f"dataset.param_vars=[{','.join(PARAM_VARS)}]",
                f"model_name={model_name}",
                f"paths.output_dir={out_root / 'model_weights'}",
                "physical_metadata.boundary_conditions='synthetic test corpus'",
                *overrides,
            ],
        )
    shrink_for_cpu(cfg)
    cfg.physical_metadata.units = dict(UNITS)
    for key, value in {**(values or {}), **dotted}.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return cfg


def fixture_inputs(
    tmp_root: Path, *, spatial_mode: str = "local", geometry: str = "fold"
) -> tuple[Path, Path]:
    """``(ae_dir, data_dir)`` under ``tmp_root``, created on first use and
    reused afterwards (so a resume test can call :func:`train_tiny_generator`
    twice on the same root)."""
    ae_dir = tmp_root / "model_weights" / f"ae_{spatial_mode}_{geometry}"
    if not ae_dir.exists():
        make_ae_export(tmp_root, spatial_mode, geometry)
    data_dir = tmp_root / "data"
    if not data_dir.exists():
        write_history_dataset(data_dir)
    return ae_dir, data_dir


def train_tiny_generator(
    tmp_root: Path,
    *,
    spatial_mode: str = "local",
    geometry: str = "fold",
    model_name: str = "latent_generator_test",
    overrides: Sequence[str] = (),
    values: Mapping[str, Any] | None = None,
    **dotted: Any,
) -> Path:
    """Fabricate the AE export + corpus under ``tmp_root`` (if absent), train the
    smoke generator via ``run(cfg)`` and return its export ``model_dir``
    (``tmp_root/model_weights/<model_name>``). ``values`` / ``**dotted`` are
    dotted-key config overrides (see :func:`compose_generator_cfg`)."""
    ae_dir, data_dir = fixture_inputs(
        tmp_root, spatial_mode=spatial_mode, geometry=geometry
    )
    cfg = compose_generator_cfg(
        ae_dir,
        data_dir,
        tmp_root,
        model_name=model_name,
        overrides=overrides,
        values={**(values or {}), **dotted},
    )
    load_run()(cfg)
    return tmp_root / "model_weights" / model_name
