"""Shared on-disk ensemble-state I/O helpers.

Used by both the smoothing (``smoothing/``) and filtering (``filtering/``)
packages so neither has to import the other for file handling.
"""

import os
import pathlib
import re

import xarray


def load_dataset(path: os.PathLike) -> xarray.Dataset:
    """Open a NetCDF file, read it fully into memory, and close the handle.

    Every call site consumes the arrays immediately (concatenated or observed),
    so eager loading costs nothing and avoids leaking netCDF4 file handles --
    long rollouts open ``ensemble_size x (num_steps + 1)`` files per window and
    would otherwise hit ``Too many open files`` / HDF5 locking on clusters.
    """
    with xarray.open_dataset(path) as ds:
        return ds.load()


def get_sorted_state_files(results_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return state files sorted by ensemble index.

    Only files matching state_<int>.nc are considered to avoid stale or
    unrelated NetCDF files from previous runs polluting ensemble size.
    """
    state_file_regex = re.compile(r"state_(\d+)\.nc")
    state_files_with_idx: list[tuple[int, pathlib.Path]] = []

    for file_path in results_dir.iterdir():
        if not file_path.is_file():
            continue
        match = state_file_regex.fullmatch(file_path.name)
        if match is None:
            continue
        state_files_with_idx.append((int(match.group(1)), file_path))

    state_files_with_idx.sort(key=lambda item: item[0])
    return [file_path for _, file_path in state_files_with_idx]
