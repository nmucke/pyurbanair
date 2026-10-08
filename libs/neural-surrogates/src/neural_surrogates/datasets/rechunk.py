"""Prepare a complete, lossless NetCDF cache for single-snapshot training reads."""

from __future__ import annotations

import fcntl
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import netCDF4
import numpy as np

_LOG = logging.getLogger(__name__)
_MANIFEST = ".rechunk-manifest.json"
_LOCK = ".rechunk.lock"
_VERSION = 1


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _inventory(root: Path) -> dict[str, dict[str, int]]:
    for split in ("train", "val"):
        directory = root / "state" / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing state split directory: {directory}")
        if not any(directory.glob("sample_*.nc")):
            raise ValueError(f"Empty state split: {directory}")
    return {
        str(path.relative_to(root)): _signature(path)
        for path in sorted((root / "state").glob("*/sample_*.nc"))
        if path.is_file()
    }


def _save_manifest(root: Path, manifest: dict[str, Any]) -> None:
    temporary = root / f"{_MANIFEST}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / _MANIFEST)


def _chunks(
    variable: Any, time_chunk: int, spatial_chunks: Sequence[int] | None
) -> tuple[int, ...] | None:
    if not variable.dimensions:
        return None
    chunks = [max(1, int(n)) for n in variable.shape]
    non_time = [i for i, name in enumerate(variable.dimensions) if name != "time"]
    # Spatial fields use their final three non-time axes (also staggered grids).
    # Coordinate vectors remain whole; their size is negligible beside fields.
    if len(non_time) >= 3 and spatial_chunks is not None:
        for axis, size in zip(non_time[-3:], spatial_chunks):
            chunks[axis] = min(chunks[axis], size)
    if "time" in variable.dimensions:
        axis = variable.dimensions.index("time")
        chunks[axis] = min(chunks[axis], time_chunk)
    return tuple(chunks)


def _blocks(
    variable: Any, budget: int, chunks: Sequence[int]
) -> Iterator[tuple[slice, ...]]:
    shape = tuple(variable.shape)
    if not shape:
        yield ()
        return
    if any(n == 0 for n in shape):
        return
    source_chunks = variable.chunking()
    source = source_chunks if isinstance(source_chunks, list) else shape
    # Whole output chunks covering a source chunk: a partly written chunk is
    # recompressed on every later write to it, which dominates the copy time.
    block = [min(n, -(-s // c) * c) for n, s, c in zip(shape, source, chunks)]
    itemsize = max(1, np.dtype(variable.dtype).itemsize)
    # Allow both the source block and its verification read in the byte budget.
    while math.prod(block) * itemsize * 2 > budget:
        counts = [-(-b // c) for b, c in zip(block, chunks)]
        axis = max(range(len(block)), key=lambda i: counts[i])
        if counts[axis] > 1:  # drop whole output chunks first
            block[axis] = counts[axis] // 2 * chunks[axis]
            continue
        axis = max(range(len(block)), key=lambda i: block[i])
        if block[axis] == 1:
            raise ValueError("max_buffer_mb is too small for one variable element")
        block[axis] = max(1, block[axis] // 2)
    for starts in itertools.product(*(range(0, n, b) for n, b in zip(shape, block))):
        yield tuple(
            slice(start, min(start + b, n)) for start, b, n in zip(starts, block, shape)
        )


def _same_values(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _copy_file(
    source: Path,
    destination: Path,
    *,
    time_chunk: int,
    spatial_chunks: Sequence[int] | None,
    compression_level: int,
    max_buffer_mb: float,
) -> int:
    budget = int(max_buffer_mb * 1024**2)
    with netCDF4.Dataset(source, "r") as original:
        if (
            original.groups
            or original.cmptypes
            or original.enumtypes
            or original.vltypes
        ):
            raise ValueError(
                f"Unsupported NetCDF groups or user-defined types: {source}"
            )
        if "time" not in original.dimensions:
            raise ValueError(f"Missing time dimension in {source}")
        count = len(original.dimensions["time"])
        with netCDF4.Dataset(destination, "w", format="NETCDF4") as target:
            target.setncatts(
                {name: original.getncattr(name) for name in original.ncattrs()}
            )
            for name, dim in original.dimensions.items():
                target.createDimension(name, None if dim.isunlimited() else len(dim))
            for name, variable in original.variables.items():
                variable.set_auto_maskandscale(False)
                variable.set_auto_chartostring(False)
                attrs = {key: variable.getncattr(key) for key in variable.ncattrs()}
                options: dict[str, Any] = {}
                if "_FillValue" in attrs:
                    options["fill_value"] = attrs.pop("_FillValue")
                chunks = _chunks(variable, time_chunk, spatial_chunks)
                is_string = variable.datatype is str
                if chunks is not None:
                    options["chunksizes"] = chunks
                    if not is_string:
                        options.update(
                            zlib=compression_level > 0, complevel=compression_level
                        )
                copied = target.createVariable(
                    name, variable.datatype, variable.dimensions, **options
                )
                copied.set_auto_maskandscale(False)
                copied.set_auto_chartostring(False)
                # Bound per-variable HDF5 caches as well as explicit numpy buffers.
                for handle in (variable, copied):
                    if handle.chunking() != "contiguous":
                        handle.set_var_chunk_cache(
                            size=min(budget // 4, 16 * 1024**2),
                            nelems=1009,
                            preemption=0.75,
                        )
                for slab in _blocks(variable, budget, chunks or ()):
                    values = np.asarray(variable[slab])
                    copied[slab] = values
                    target.sync()
                    if not _same_values(values, np.asarray(copied[slab])):
                        raise ValueError(
                            f"Rechunked values differ in {source}: {name}{slab}"
                        )
                # Avoid applying least_significant_digit quantization a second
                # time: attributes are restored only after raw writes finish.
                copied.setncatts(attrs)
            if len(target.dimensions["time"]) != count:
                raise ValueError(f"Rechunking changed the time count in {source}")
    return count


def prepare_rechunked_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    time_chunk: int = 1,
    spatial_chunks: Sequence[int] | None = (16, 64, 64),
    compression_level: int = 1,
    # Room for a whole-frame block spanning a 40-frame source chunk, so each
    # source chunk is decompressed once (halves the whole-frame copy time).
    max_buffer_mb: float = 1024,
) -> Path:
    """Repack every state trajectory before returning a training-ready root.

    All splits, frames, cells, coordinates and encoded values are retained;
    only physical NetCDF storage changes. The existing SnapshotDataset can read
    this root without any indexing or crop-sampling changes.
    ``spatial_chunks=None`` keeps each frame whole (one chunk per frame and
    variable), for models that read entire domains. Non-state files
    (parameters, model artifacts, etc.) are outside this autoencoder cache.

    Preparation streams bounded blocks and verifies every copied value. A
    manifest permits interrupted preparation to resume and completed files to
    be reused using size/mtime/ctime fingerprints. Changed sources, options or
    completed outputs fail closed: select a new cache directory to rebuild.
    An exclusive lock prevents concurrent preparation in the same directory.
    """
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and rechunk output directories must not overlap")
    if isinstance(time_chunk, bool) or int(time_chunk) != time_chunk or time_chunk < 1:
        raise ValueError("time_chunk must be a positive integer")
    if spatial_chunks is not None:
        spatial_chunks = tuple(spatial_chunks)
        if len(spatial_chunks) != 3 or any(
            isinstance(n, bool) or int(n) != n or n < 1 for n in spatial_chunks
        ):
            raise ValueError(
                "spatial_chunks must be None or contain three positive integers"
            )
    if (
        isinstance(compression_level, bool)
        or int(compression_level) != compression_level
        or not 0 <= compression_level <= 9
    ):
        raise ValueError("compression_level must be an integer from 0 through 9")
    if not math.isfinite(max_buffer_mb) or max_buffer_mb <= 0:
        raise ValueError("max_buffer_mb must be positive and finite")
    options: dict[str, Any] = {
        "time_chunk": int(time_chunk),
        "spatial_chunks": (
            None if spatial_chunks is None else [int(n) for n in spatial_chunks]
        ),
        "compression_level": int(compression_level),
    }
    sources = _inventory(source)
    output.mkdir(parents=True, exist_ok=True)
    with (output / _LOCK).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is preparing {output}") from exc
        manifest_path = output / _MANIFEST
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            expected = (_VERSION, str(source), options, sources)
            actual = tuple(
                manifest.get(k)
                for k in ("version", "source_root", "options", "sources")
            )
            if actual != expected:
                raise ValueError(
                    f"Rechunk source or options changed; use a new output directory: {output}"
                )
        else:
            manifest = {
                "version": _VERSION,
                "source_root": str(source),
                "options": options,
                "sources": sources,
                "files": {},
                "complete": False,
            }
        files = manifest["files"]
        allowed = {_LOCK, _MANIFEST, f"{_MANIFEST}.tmp"}
        # Training computes these against the prepared files on its first run.
        # They are derived artifacts, never copied from the original dataset.
        allowed.update(
            f"normalization_stats/{Path(relative).parent.name}.npz"
            for relative in sources
        )
        for relative in files:
            if relative not in sources:
                raise ValueError(f"Unexpected manifest entry: {relative}")
            allowed.update((relative, f"{relative}.rechunking"))
        unknown = [
            str(p.relative_to(output))
            for p in output.rglob("*")
            if (p.is_file() or p.is_symlink())
            and str(p.relative_to(output)) not in allowed
        ]
        if unknown:
            raise ValueError(
                f"Untracked files in rechunk output {output}: {unknown[:5]}"
            )
        _save_manifest(output, manifest)
        for index, (relative, signature) in enumerate(sources.items(), start=1):
            destination = output / relative
            record = files.get(relative)
            if record is not None and record.get("output") is not None:
                if (
                    not destination.is_file()
                    or _signature(destination) != record["output"]
                ):
                    raise ValueError(
                        f"Completed rechunk file changed or is missing: {destination}"
                    )
                continue
            files[relative] = {"output": None}
            manifest["complete"] = False
            _save_manifest(output, manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".rechunking")
            _LOG.info("Rechunking trajectory %d/%d: %s", index, len(sources), relative)
            count = _copy_file(
                source / relative, temporary, max_buffer_mb=max_buffer_mb, **options
            )
            if _signature(source / relative) != signature:
                raise ValueError(
                    f"Source changed during rechunking: {source / relative}"
                )
            temporary.replace(destination)
            files[relative] = {"output": _signature(destination), "snapshots": count}
            _save_manifest(output, manifest)
        if _inventory(source) != sources:
            raise ValueError("Source inventory changed during rechunking")
        manifest["complete"] = True
        manifest["snapshot_count"] = sum(
            record["snapshots"] for record in files.values()
        )
        _save_manifest(output, manifest)
        _LOG.info(
            "Rechunk cache ready: %d trajectories, %d snapshots in %s",
            len(files),
            manifest["snapshot_count"],
            output,
        )
    return output
