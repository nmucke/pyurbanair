"""Shared on-disk ensemble-state I/O helpers.

Used by both the smoothing (``smoothing/``) and filtering (``filtering/``)
packages so neither has to import the other for file handling.
"""

import os
import pathlib
import re

import netCDF4
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


def concat_member_files(files: list[pathlib.Path], out_path: pathlib.Path) -> None:
    """Stack per-member state files along a new `ensemble` dim, one at a time.

    Same file as `xarray.concat(members, dim="ensemble")` written to NetCDF,
    without holding the whole ensemble in memory.
    """
    with netCDF4.Dataset(files[0]) as ref, netCDF4.Dataset(out_path, "w") as out:
        out.setncatts({k: ref.getncattr(k) for k in ref.ncattrs()})
        out.createDimension("ensemble", len(files))
        for name, dim in ref.dimensions.items():
            out.createDimension(name, len(dim))
        data_vars = [v for v in ref.variables if v not in ref.dimensions]
        for name, var in ref.variables.items():
            dims = (
                var.dimensions
                if name in ref.dimensions
                else ("ensemble", *var.dimensions)
            )
            new = out.createVariable(name, var.dtype, dims)
            new.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            if name in ref.dimensions:
                new[...] = var[...]
        for m, path in enumerate(files):
            with netCDF4.Dataset(path) as src:
                for name in data_vars:
                    out.variables[name][m, ...] = src.variables[name][...]


def last_frames(files: list[pathlib.Path]) -> xarray.Dataset:
    """Ensemble state at the final time, reading one frame per member file."""
    frames = []
    for path in files:
        with xarray.open_dataset(path) as ds:
            frames.append(ds.isel(time=-1).load())
    return xarray.concat(frames, dim="ensemble", join="override")
