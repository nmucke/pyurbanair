"""Shared fixtures for the plan-07 latent generator (model, trainer, deploy tests).

Not a test module (no ``test_`` prefix): it fabricates the three inputs every
plan-07 test needs and is imported as ``tests.neural_surrogates._latent_generator_fixtures`` by
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
* :func:`fixture_inputs` -- both of the above under one root, created once.

Callers gate on the vendored Tadpole runtime deps themselves
(``pytest.importorskip("diffusers")`` / ``timm`` / ``einops``) before
importing this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

STATE_VARS: tuple[str, ...] = ("u", "v", "w")
PARAM_VARS: tuple[str, ...] = ("inflow_angle", "velocity_magnitude")
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
                # The generators save their whole Hydra config: the top-level
                # ``time`` block is the unused default (deliberately wrong
                # here, incl. a spin-up that would satisfy any plateau), and
                # the horizon the data were generated with is training_data's.
                "time": {
                    "simulation_time": 60.0,
                    "output_frequency": 2.5 * float(dt),
                    "spinup_time": 1.0e6,
                },
                "training_data": {
                    "simulation_time": float(t_len * dt),
                    "output_frequency": float(dt),
                    "spinup_time": 0.0,
                },
            }
        ),
        root / "config.yaml",
    )
    return root


def fixture_inputs(
    tmp_root: Path, *, spatial_mode: str = "local", geometry: str = "fold"
) -> tuple[Path, Path]:
    """``(ae_dir, data_dir)`` under ``tmp_root``, created on first use and
    reused afterwards."""
    ae_dir = tmp_root / "model_weights" / f"ae_{spatial_mode}_{geometry}"
    if not ae_dir.exists():
        make_ae_export(tmp_root, spatial_mode, geometry)
    data_dir = tmp_root / "data"
    if not data_dir.exists():
        write_history_dataset(data_dir)
    return ae_dir, data_dir
