"""Trajectory corpus on disk: Zarr writer + loader (§5).

The existing Fortran backends are the data generator
(``scripts/generate_neural_surrogate_data.py``); this module owns the
**storage format**. Each trajectory is stored **whole** (never pre-windowed,
§5) as a Zarr group; the geometry mask/SDF is identical across trajectories
(fixed geometry, D5) and stored **once at the corpus root**. A JSON manifest
records solver, grid, schema, split assignment, counts, and git SHA.

On-disk layout::

    <corpus>/
      manifest.json            # grid, schema, splits, counts, git SHA, var_names
      geometry.npy             # solid/fluid mask [Z, Y, X]
      static.npy               # baked static channels [S, Z, Y, X] (SDF [+ mask])
      normalization.json       # written by the normalization-fit step (§6.3)
      trajectories/<id>.zarr   # group: fields [T,C,Z,Y,X], params [T,P], times [T]
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional, Sequence

import numpy as np
import zarr

from .dataset import Corpus
from .grid import GridMeta
from ..utils.schema import ContractSchema

_MANIFEST = "manifest.json"
_GEOMETRY = "geometry.npy"
_STATIC = "static.npy"
_TRAJ_DIR = "trajectories"


class CorpusWriter:
    """Incrementally write a trajectory corpus to disk."""

    def __init__(
        self,
        path: str | pathlib.Path,
        grid: GridMeta,
        schema: ContractSchema,
        var_names: Sequence[str],
        static_channels: np.ndarray,
        geometry_mask: np.ndarray,
    ) -> None:
        self.path = pathlib.Path(path)
        (self.path / _TRAJ_DIR).mkdir(parents=True, exist_ok=True)
        self.grid = grid
        self.schema = schema
        self.var_names = tuple(var_names)
        np.save(self.path / _GEOMETRY, np.asarray(geometry_mask, np.float32))
        np.save(self.path / _STATIC, np.asarray(static_channels, np.float32))
        self._entries: list[dict] = []

    def add_trajectory(
        self,
        traj_id: str,
        fields: np.ndarray,
        params: np.ndarray,
        times: np.ndarray,
        split: str,
    ) -> None:
        """Write one whole trajectory.

        Args:
            fields: ``[T, C, Z, Y, X]`` raw state.
            params: ``[T, P]`` per-frame **encoded** conditioning (§1.5).
            times: ``[T]`` output-frame times.
            split: ``"train"`` / ``"val"`` / ``"test"`` (split by trajectory, §5).
        """
        fields = np.asarray(fields, dtype=np.float32)
        params = np.asarray(params, dtype=np.float32)
        if fields.ndim != 5:
            raise ValueError(f"fields must be [T,C,Z,Y,X], got {fields.shape}.")
        store = zarr.open_group(str(self.path / _TRAJ_DIR / f"{traj_id}.zarr"), mode="w")
        # chunk along time so windowing reads are cheap (§5).
        store.create_dataset(
            "fields", data=fields, chunks=(1,) + fields.shape[1:], dtype="f4"
        )
        store.create_dataset("params", data=params, chunks=(1, params.shape[1]), dtype="f4")
        store.create_dataset("times", data=np.asarray(times, np.float64))
        self._entries.append({"id": traj_id, "split": split, "n_frames": int(fields.shape[0])})

    def finalize(self, extra: Optional[dict] = None) -> pathlib.Path:
        """Write ``manifest.json`` and return the corpus path."""
        splits: dict[str, list[str]] = {}
        for e in self._entries:
            splits.setdefault(e["split"], []).append(e["id"])
        manifest = {
            "grid": self.grid.to_dict(),
            "schema": self.schema.to_dict(),
            "var_names": list(self.var_names),
            "splits": splits,
            "counts": {k: len(v) for k, v in splits.items()},
            "n_trajectories": len(self._entries),
        }
        manifest.update(extra or {})
        with open(self.path / _MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return self.path


class ZarrCorpus(Corpus):
    """Read a corpus written by :class:`CorpusWriter` (lazy field slicing)."""

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        with open(self.path / _MANIFEST) as f:
            self.manifest = json.load(f)
        self.grid = GridMeta.from_dict(self.manifest["grid"])
        contract = ContractSchema.from_dict(self.manifest["schema"])
        self.contract = contract
        self.param_schema = contract.param_schema
        self.var_names = tuple(self.manifest["var_names"])
        self.static_channels = np.load(self.path / _STATIC)
        self.geometry_mask = np.load(self.path / _GEOMETRY)
        self._splits = self.manifest["splits"]
        self._cache: dict[str, zarr.Group] = {}

    def _group(self, traj_id: str) -> zarr.Group:
        if traj_id not in self._cache:
            self._cache[traj_id] = zarr.open_group(
                str(self.path / _TRAJ_DIR / f"{traj_id}.zarr"), mode="r"
            )
        return self._cache[traj_id]

    def split_ids(self, split: str) -> list[str]:
        return list(self._splits.get(split, []))

    def num_frames(self, traj_id: str) -> int:
        return int(self._group(traj_id)["fields"].shape[0])

    def load_fields(self, traj_id: str) -> np.ndarray:
        # zarr array supports lazy slicing; return as-is so WindowDataset only
        # reads the frames it needs.
        return self._group(traj_id)["fields"]

    def load_params(self, traj_id: str) -> np.ndarray:
        return self._group(traj_id)["params"][:]


def open_corpus(path: str | pathlib.Path) -> ZarrCorpus:
    """Open a corpus directory for reading."""
    return ZarrCorpus(path)
