"""Checkpoint resolution + manifest loading (``docs/neural_surrogate_plan.md`` §7).

A trained model is **weights + everything needed to reproduce inference**,
stored under ``models/neural_surrogates/<run_id>/``. Inference resolves a
checkpoint by **explicit path** or ``run_id`` (with a ``latest`` symlink per
(solver, geometry, architecture)).
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

DEFAULT_MODELS_ROOT = pathlib.Path("models/neural_surrogates")

# Artifact filenames inside a checkpoint directory (§7).
ARCHITECTURE_FILE = "architecture.json"
NORMALIZATION_FILE = "normalization.json"
GRID_FILE = "grid.json"
GEOMETRY_FILE = "geometry.npy"
SCHEMA_FILE = "schema.json"
MANIFEST_FILE = "manifest.json"
WEIGHTS_DIR = "weights"


def resolve_checkpoint(
    path_or_run_id: str | pathlib.Path,
    models_root: Optional[str | pathlib.Path] = None,
) -> pathlib.Path:
    """Resolve an explicit path or a ``run_id`` to a checkpoint directory.

    Tries, in order: the value as a filesystem path; then
    ``<models_root>/<run_id>``. Raises if neither exists.
    """
    candidate = pathlib.Path(path_or_run_id)
    if candidate.exists():
        return candidate.resolve()

    root = pathlib.Path(models_root) if models_root is not None else DEFAULT_MODELS_ROOT
    by_run_id = root / str(path_or_run_id)
    if by_run_id.exists():
        return by_run_id.resolve()

    raise FileNotFoundError(
        f"Could not resolve checkpoint {path_or_run_id!r}: tried {candidate} and "
        f"{by_run_id}."
    )


def _read_json(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_manifest(checkpoint_dir: str | pathlib.Path) -> dict:
    """Load ``manifest.json`` from a checkpoint directory."""
    return _read_json(pathlib.Path(checkpoint_dir) / MANIFEST_FILE)


def load_json_artifact(checkpoint_dir: str | pathlib.Path, filename: str) -> dict:
    """Load one of the JSON artifacts (architecture/normalization/grid/schema)."""
    return _read_json(pathlib.Path(checkpoint_dir) / filename)
