"""Orbax checkpoint + the §7 artifact set (architecture / schema / grid / norm).

A trained model is **weights + everything needed to reproduce inference**,
including the **architecture identity** (``docs/neural_surrogate_plan.md`` §7).
``save_checkpoint`` writes the full artifact set; ``load_checkpoint``
reconstructs the exact network (resolving ``architecture.json`` through the
registry) and returns it together with the normalization, grid, geometry, and
source-solver schema the forward model needs.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax
import numpy as np
import orbax.checkpoint as ocp

from ..architectures.base import SurrogateArchitecture
from ..architectures.registry import resolve_architecture
from ..data.grid import GridMeta
from ..data.normalization import Normalization
from ..utils import registry
from ..utils.schema import ContractSchema

_STATIC_FILE = "static.npy"
_IC_BANK_FILE = "ic_bank.npz"


def _write_json(path: pathlib.Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_checkpoint(
    directory: str | pathlib.Path,
    arch: SurrogateArchitecture,
    *,
    arch_name: str,
    arch_config: dict[str, Any],
    history_len: int,
    normalization: Normalization,
    grid: GridMeta,
    geometry_mask: np.ndarray,
    static_channels: np.ndarray,
    schema: ContractSchema,
    native_output_frequency: Optional[float] = None,
    metrics: Optional[dict] = None,
    manifest_extra: Optional[dict] = None,
    ic_bank: Optional[dict] = None,
) -> pathlib.Path:
    """Persist a full checkpoint and return its directory.

    Writes ``weights/`` (Orbax), ``architecture.json`` (name + config + ``K``),
    ``normalization.json``, ``grid.json``, ``geometry.npy`` (+ ``static.npy``),
    ``schema.json``, ``manifest.json``, and ``metrics.json`` (§7).
    """
    directory = pathlib.Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    # Weights: save only the inexact-array leaves (the trainable params).
    params = eqx.filter(arch, eqx.is_inexact_array)
    weights_dir = directory / registry.WEIGHTS_DIR
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(weights_dir, params)

    _write_json(
        directory / registry.ARCHITECTURE_FILE,
        {"name": arch_name, "config": arch_config, "history_len": int(history_len)},
    )
    _write_json(directory / registry.NORMALIZATION_FILE, normalization.to_dict())
    _write_json(directory / registry.GRID_FILE, grid.to_dict())
    _write_json(directory / registry.SCHEMA_FILE, schema.to_dict())
    np.save(directory / registry.GEOMETRY_FILE, np.asarray(geometry_mask, np.float32))
    np.save(directory / _STATIC_FILE, np.asarray(static_channels, np.float32))

    # Canned initial-condition bank (§4, recommended cold-start default): spun-up
    # frames keyed by per-frame encoded conditioning; state=None picks nearest.
    if ic_bank is not None:
        np.savez(
            directory / _IC_BANK_FILE,
            params=np.asarray(ic_bank["params"], np.float32),
            fields=np.asarray(ic_bank["fields"], np.float32),
        )

    _write_json(
        directory / "metrics.json",
        metrics or {},
    )
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "architecture": arch_name,
        "history_len": int(history_len),
        "source_solver_name": schema.source_solver_name,
        # Native autoregressive step size: the corpus output_frequency the model
        # was trained on. One ``arch.step`` advances the flow by this much
        # physical time. Inference resamples the rollout from this to the
        # requested output_frequency, so the model is self-describing (§4).
        "native_output_frequency": (
            None if native_output_frequency is None else float(native_output_frequency)
        ),
    }
    manifest.update(manifest_extra or {})
    _write_json(directory / registry.MANIFEST_FILE, manifest)
    return directory


@dataclass
class LoadedCheckpoint:
    """Everything inference needs from a checkpoint (§7)."""

    arch: SurrogateArchitecture
    arch_name: str
    history_len: int
    normalization: Normalization
    grid: GridMeta
    schema: ContractSchema
    geometry_mask: np.ndarray
    static_channels: np.ndarray
    manifest: dict
    native_output_frequency: Optional[float] = None
    ic_bank: Optional[dict] = None


def load_checkpoint(
    directory: str | pathlib.Path,
    *,
    key: Optional[jax.Array] = None,
    expected_architecture: Optional[str] = None,
) -> LoadedCheckpoint:
    """Reconstruct a checkpoint, validating the architecture name (§7).

    Args:
        directory: Checkpoint directory.
        key: PRNG key for building the template architecture (weights are then
            overwritten from disk). Defaults to a fixed key.
        expected_architecture: If given, assert it matches the checkpoint's
            recorded architecture (the loader's contract check, §7).
    """
    directory = pathlib.Path(directory).resolve()
    arch_meta = registry.load_json_artifact(directory, registry.ARCHITECTURE_FILE)
    arch_name = arch_meta["name"]
    arch_config = arch_meta["config"]
    history_len = int(arch_meta["history_len"])

    if expected_architecture is not None and expected_architecture != arch_name:
        raise ValueError(
            f"Requested architecture {expected_architecture!r} does not match "
            f"checkpoint architecture {arch_name!r}."
        )

    if key is None:
        key = jax.random.PRNGKey(0)
    template = resolve_architecture(arch_name, arch_config, key=key)
    template_params = eqx.filter(template, eqx.is_inexact_array)

    weights_dir = directory / registry.WEIGHTS_DIR
    with ocp.StandardCheckpointer() as ckptr:
        restored_params = ckptr.restore(weights_dir, target=template_params)
    arch = eqx.combine(restored_params, template)

    normalization = Normalization.from_dict(
        registry.load_json_artifact(directory, registry.NORMALIZATION_FILE)
    )
    grid = GridMeta.from_dict(
        registry.load_json_artifact(directory, registry.GRID_FILE)
    )
    schema = ContractSchema.from_dict(
        registry.load_json_artifact(directory, registry.SCHEMA_FILE)
    )
    geometry_mask = np.load(directory / registry.GEOMETRY_FILE)
    static_channels = np.load(directory / _STATIC_FILE)
    manifest = registry.load_manifest(directory)
    native_output_frequency = manifest.get("native_output_frequency")
    if native_output_frequency is not None:
        native_output_frequency = float(native_output_frequency)

    ic_bank = None
    ic_path = directory / _IC_BANK_FILE
    if ic_path.exists():
        with np.load(ic_path) as data:
            ic_bank = {"params": data["params"], "fields": data["fields"]}

    return LoadedCheckpoint(
        arch=arch,
        arch_name=arch_name,
        history_len=history_len,
        normalization=normalization,
        grid=grid,
        schema=schema,
        geometry_mask=geometry_mask,
        static_channels=static_channels,
        manifest=manifest,
        native_output_frequency=native_output_frequency,
        ic_bank=ic_bank,
    )
