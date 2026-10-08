"""Cache of the frozen autoencoder's raw latents for latent-generator training.

:class:`~neural_surrogates.TadpoleLatentGenerator` encodes every training
snapshot with its frozen autoencoder on each step. The encoding is
deterministic (latent type ``"mode"``), so :func:`prepare_latent_cache` runs it
once over every saved frame of every trajectory and
:class:`LatentCacheDataset` serves the stored raw latents instead of states.

Layout under the cache root (``<stem>`` is the state file's ``sample_XXXX``)::

    <split>/<stem>.npy       raw state latents, float32 (T, D, Zl, Yl, Xl), all
                             T saved frames (memory-mappable)
    <split>/<stem>.geom.npy  raw geometry conditioning, float32: the geometry
                             branch output (G, Zl, Yl, Xl) or, without a
                             branch, the geometry latents (D_geom, Zl, Yl, Xl)
    <split>/<stem>.sums.npz  float64 per-channel ``sum`` / ``sumsq`` of the
                             working latents over all frames, and their ``count``
    latent_stats.npz         float64 ``mean`` / ``std`` of the working latents
                             over every train frame (std not floored)
    manifest.json            AE fingerprint, sources, encode settings, progress

Like :func:`~neural_surrogates.datasets.rechunk.prepare_rechunked_dataset`,
preparation resumes after an interruption, reuses completed files and fails
closed when the AE, a source file or an encode setting changed. Files are
fingerprinted by size and mtime only, so a ``chmod``, ``chown`` or ``cp -a``
keeps the cache valid. A complete cache is validated without locking or
writing, so concurrent training runs can share it, read-only too. Preparing
takes an exclusive lock, but ``flock`` is node-local on BeeGFS: run only one
preparation of a cache at a time.
"""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import xarray as xr
from neural_surrogates.datasets.snapshot_history import SnapshotHistoryDataset
from torch.utils.data import Dataset

_LOG = logging.getLogger(__name__)
_MANIFEST = "manifest.json"
_LOCK = ".latent-cache.lock"
_STATS = "latent_stats.npz"
_VERSION = 2


def _fingerprint(path: Path) -> dict[str, int]:
    """Size and mtime of ``path``: unlike ``rechunk._signature`` no ctime, which
    a ``chmod``, ``chown`` or copy preserving mtime changes."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _outputs(stem: str) -> list[str]:
    return [f"{stem}{suffix}" for suffix in (".npy", ".geom.npy", ".sums.npz")]


def _unchanged(directory: Path, stem: str, record: dict[str, Any]) -> bool:
    """Whether the completed outputs of ``stem`` still match ``record``."""
    return all(
        (directory / name).is_file()
        and _fingerprint(directory / name) == record["outputs"][name]
        for name in _outputs(stem)
    )


def _plain(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` as it reads back from JSON (tuples become lists)."""
    plain: dict[str, Any] = json.loads(json.dumps(record))
    return plain


def _split_record(base: SnapshotHistoryDataset) -> dict[str, Any]:
    """What the cached latents of ``base``'s split depend on."""
    return _plain(
        {
            "source_root": str(base.root.resolve()),
            "state_vars": list(base.state_vars),
            "geometry_var": base.geometry_var,
            "dtype": str(base.dtype),
            "sdf_features": base.sdf_feature_mode,
            "sdf_clamp_cells": base.sdf_clamp_cells,
            "sources": {path.stem: _fingerprint(path) for path in base._state_files},
        }
    )


def _save_manifest(root: Path, manifest: dict[str, Any]) -> None:
    temporary = root / f"{_MANIFEST}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / _MANIFEST)


def _read_complete_manifest(root: Path) -> dict[str, Any]:
    path = root / _MANIFEST
    manifest: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    if not manifest.get("complete"):
        raise ValueError(
            f"Latent cache is missing or incomplete: {root}; run prepare_latent_cache"
        )
    return manifest


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:  # a handle: np.savez keeps the name
        np.savez(handle, **arrays)
    temporary.replace(path)


def _encode_trajectory(
    model: Any,
    base: SnapshotHistoryDataset,
    traj: int,
    directory: Path,
    device: torch.device,
) -> int:
    """Write the three files of one trajectory; returns its frame count."""
    path = base._state_files[traj]
    frames = base._traj_lengths[traj]
    geometry = base.geometry_for(traj).unsqueeze(0).to(device)
    features = base.geom_features_for(traj)
    if features is not None:
        features = features.unsqueeze(0).to(device)
    with torch.no_grad(), model._autocast_off(device):
        _, branch_cond, geom_raw = model._geometry_conditioning(
            geometry, features, torch.float32
        )
    geom = branch_cond if geom_raw is None else geom_raw
    if not torch.isfinite(geom).all():
        raise ValueError(f"Non-finite geometry conditioning for {path}")
    temporary = directory / f"{path.stem}.geom.npy.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, geom[0].cpu().numpy())
    temporary.replace(directory / f"{path.stem}.geom.npy")

    latent_grid = model.latent_grid_for(base.grid_shape(traj))
    temporary = directory / f"{path.stem}.npy.tmp"
    latents = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(frames, model.state_latent_dim, *latent_grid),
    )
    n_work = model.working_latent_dim
    total = torch.zeros(n_work, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = 0
    with xr.open_dataset(path, cache=False) as ds:
        # Read whole source time chunks: a frame-by-frame read decompresses
        # every chunk once per frame it holds.
        first = ds[base.state_vars[0]]
        chunks = first.encoding.get("chunksizes")
        step = int(chunks[first.dims.index("time")]) if chunks else 1
        for t0 in range(0, frames, step):
            block = [
                # Time first, whatever the source's dim order.
                np.asarray(
                    ds[v].isel(time=slice(t0, t0 + step)).transpose("time", ...).values
                )
                for v in base.state_vars
            ]
            for k in range(block[0].shape[0]):
                # Exactly SnapshotDataset.__getitem__'s state, one frame.
                state = torch.from_numpy(np.stack([b[k] for b in block], 0))
                state = state.to(base.dtype).unsqueeze(0).to(device)
                with torch.no_grad(), model._autocast_off(device):
                    z_raw, geom_b, *_ = model._encode_raw(state, geometry, features)
                # The sums compute_latent_normalization accumulates.
                work = z_raw if geom_b is None else torch.cat([z_raw, geom_b], dim=1)
                if not torch.isfinite(work).all():
                    raise ValueError(f"Non-finite latents at frame {t0 + k} of {path}")
                latents[t0 + k] = z_raw[0].cpu().numpy()
                w64 = work.cpu().to(torch.float64)
                total += w64.sum(dim=(0, 2, 3, 4))
                total_sq += (w64 * w64).sum(dim=(0, 2, 3, 4))
                count += w64.shape[2] * w64.shape[3] * w64.shape[4]
    latents.flush()
    del latents
    temporary.replace(directory / f"{path.stem}.npy")
    _save_npz(
        directory / f"{path.stem}.sums.npz",
        sum=total.numpy(),
        sumsq=total_sq.numpy(),
        count=np.int64(count),
    )
    return frames


def prepare_latent_cache(
    model: Any,
    datasets: Mapping[str, SnapshotHistoryDataset],
    output_root: str | Path,
    *,
    device: torch.device | str,
) -> Path:
    """Create, resume or validate the raw-latent cache of ``datasets``' splits.

    ``model`` is a :class:`~neural_surrogates.TadpoleLatentGenerator` with its
    pretrained AE loaded; it is moved to ``device`` and put in eval mode.
    Every saved frame of every trajectory is encoded in fp32, without
    autocast, one frame at a time. ``datasets`` must hold a ``"train"``
    split: ``latent_stats.npz`` holds the statistics of all its frames.

    The manifest records the AE weights' fingerprint and settings
    (``ae_kwargs``) and every split's source state files and dataset
    settings. If any of it changed, or a completed file is missing or
    changed, this raises: use a new ``output_root``. Interrupted trajectories
    are encoded again, and a trajectory with non-finite latents raises.
    A complete, unchanged cache returns without locking or writing.
    """
    if model.ae_fingerprint is None:
        raise ValueError(
            "The latent cache needs the AE weights' fingerprint; build the "
            "model from its pretrained_ae_dir (skip_pretrained_load=False)"
        )
    if "train" not in datasets:
        raise ValueError("prepare_latent_cache needs a 'train' split")
    device = torch.device(device)
    model.to(device).eval()
    splits = ["train"] + [split for split in datasets if split != "train"]
    recorded = _plain(
        {
            "version": _VERSION,
            "ae_fingerprint": model.ae_fingerprint,
            "ae_kwargs": model.ae_kwargs,
            "splits": {split: _split_record(datasets[split]) for split in splits},
        }
    )
    output = Path(output_root).expanduser().resolve()
    manifest_path = output / _MANIFEST
    if manifest_path.exists():
        # Validate a complete cache without the lock: concurrent and
        # read-only users never block each other.
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("files", {})
        if (
            manifest.get("complete")
            and {key: manifest.get(key) for key in recorded} == recorded
            and all(
                files.get(f"{split}/{path.stem}") is not None
                and _unchanged(output / split, path.stem, files[f"{split}/{path.stem}"])
                for split in splits
                for path in datasets[split]._state_files
            )
        ):
            _LOG.info("Latent cache valid: %d trajectories in %s", len(files), output)
            return output
    output.mkdir(parents=True, exist_ok=True)
    with (output / _LOCK).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is preparing {output}") from exc
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if {key: manifest.get(key) for key in recorded} != recorded:
                raise ValueError(
                    "Latent cache AE, sources or encode settings changed; use a "
                    f"new output_root: {output}"
                )
        else:
            manifest = {**recorded, "files": {}, "complete": False}
        files = manifest["files"]
        total = sum(len(datasets[split]._state_files) for split in splits)
        index = 0
        for split in splits:
            base = datasets[split]
            directory = output / split
            directory.mkdir(exist_ok=True)
            for traj, path in enumerate(base._state_files):
                index += 1
                key = f"{split}/{path.stem}"
                record = files.get(key)
                if record is not None:
                    if not _unchanged(directory, path.stem, record):
                        raise ValueError(
                            f"Completed latent cache file changed or is missing: {key}"
                        )
                    continue
                files[key] = None
                manifest["complete"] = False
                _save_manifest(output, manifest)
                _LOG.info("Encoding latents %d/%d: %s", index, total, key)
                frames = _encode_trajectory(model, base, traj, directory, device)
                if (
                    _fingerprint(path)
                    != recorded["splits"][split]["sources"][path.stem]
                ):
                    raise ValueError(f"Source changed during encoding: {path}")
                files[key] = {
                    "frames": frames,
                    "outputs": {
                        name: _fingerprint(directory / name)
                        for name in _outputs(path.stem)
                    },
                }
                _save_manifest(output, manifest)
            if split == "train":
                train_sum = np.zeros(model.working_latent_dim)
                train_sq = np.zeros_like(train_sum)
                count = 0
                for path in base._state_files:
                    with np.load(directory / f"{path.stem}.sums.npz") as sums:
                        train_sum += sums["sum"]
                        train_sq += sums["sumsq"]
                        count += int(sums["count"])
                mean = train_sum / count
                std = np.sqrt(np.maximum(train_sq / count - mean * mean, 0.0))
                if not (np.isfinite(mean).all() and np.isfinite(std).all()):
                    raise ValueError(
                        "Latent statistics are not finite; check the AE/data"
                    )
                _save_npz(output / _STATS, mean=mean, std=std)
                _LOG.info("Latent statistics over %d train latent cells", count)
        manifest["complete"] = True
        _save_manifest(output, manifest)
        _LOG.info("Latent cache ready: %d trajectories in %s", len(files), output)
    return output


def load_latent_stats(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """float64 ``(mean, std)`` of the working latents over every train frame
    (std not yet floored: ``set_latent_normalization`` floors it)."""
    root = Path(root)
    _read_complete_manifest(root)
    with np.load(root / _STATS) as stats:
        return stats["mean"], stats["std"]


class LatentCacheDataset(Dataset):
    """Cached raw latents of ``base``'s samples (split ``base.split``).

    Same length and ``sample_index`` as ``base``. Each item is
    ``{"latent": (D, Zl, Yl, Xl), "geom": raw geometry conditioning (one
    tensor per trajectory), "params_hist": (Hp, P)}``; the default collate
    batches them. The latent file is memory-mapped lazily per process.
    """

    def __init__(self, root: str | Path, base: SnapshotHistoryDataset) -> None:
        self.root = Path(root)
        self.base = base
        manifest = _read_complete_manifest(self.root)
        if manifest["splits"].get(base.split) != _split_record(base):
            raise ValueError(
                f"Latent cache {self.root} was not built from this '{base.split}' "
                "dataset (sources or settings differ)"
            )
        directory = self.root / base.split
        stems = [path.stem for path in base._state_files]
        self._files = [directory / f"{stem}.npy" for stem in stems]
        self._geoms = [
            torch.from_numpy(np.load(directory / f"{stem}.geom.npy")) for stem in stems
        ]
        # One open memory map, like SnapshotDataset's file handle: batches
        # come from one trajectory at a time.
        self._open: tuple[int, np.ndarray] | None = None

    @property
    def sample_index(self) -> list[tuple[int, int]]:
        return self.base.sample_index

    def grid_shape(self, traj: int) -> tuple[int, ...]:
        return self.base.grid_shape(traj)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open"] = None
        return state

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, t = self.base.sample_index[idx]
        if self._open is None or self._open[0] != traj:
            self._open = (traj, np.load(self._files[traj], mmap_mode="r"))
        return {
            "latent": torch.from_numpy(np.array(self._open[1][t])),
            "geom": self._geoms[traj],
            "params_hist": self.base.params_hist_for(traj, t),
        }
