"""Bounded, read-only inspection of a run's NetCDF results."""

from __future__ import annotations

import pathlib
from typing import Any


def contained_file(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Artifact must be an existing file inside the owned run")
    return path


def inspect_results(
    results_dir: str | pathlib.Path,
    artifact_id: int | None = None,
    variable: str | None = None,
    selection: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    """List the NetCDF files run_forward.py wrote, or inspect one of them.

    `artifact_id` is a file's position in the sorted listing: `state.nc` and
    `params.nc` once the run has finished, `windows/<kind>_<window>.nc` as
    each window finishes.
    """
    root = pathlib.Path(results_dir).resolve()
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError("offset >= 0 and 1 <= limit <= 200 required")
    paths = sorted(root.rglob("*.nc")) if root.is_dir() else []
    entries: list[dict[str, Any]] = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
        for path in paths
    ]
    if not entries:
        return {
            "status": "unavailable",
            "artifacts": [],
            "message": "No NetCDF results are available yet",
        }
    if artifact_id is None:
        return {
            "artifacts": [
                {"artifact_id": i, **entry}
                for i, entry in enumerate(
                    entries[offset : offset + limit], start=offset
                )
            ],
            "total": len(entries),
            "next_offset": offset + limit if offset + limit < len(entries) else None,
        }
    if not 0 <= artifact_id < len(entries):
        raise ValueError("Unknown artifact ID")
    entry = entries[artifact_id]
    path = contained_file(root, entry["path"])
    import numpy as np
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        result: dict[str, Any] = {
            "artifact": entry,
            "dimensions": dict(dataset.sizes),
            "variables": {
                key: {
                    "dimensions": list(value.dims),
                    "dtype": str(value.dtype),
                    "units": value.attrs.get("units"),
                }
                for key, value in dataset.data_vars.items()
            },
            "coordinates": {
                key: {
                    "size": value.size,
                    "first": (
                        str(value.isel({dim: 0 for dim in value.dims}).values)
                        if value.size
                        else None
                    ),
                    "last": (
                        str(value.isel({dim: -1 for dim in value.dims}).values)
                        if value.size
                        else None
                    ),
                }
                for key, value in dataset.coords.items()
            },
        }
        if variable is not None:
            if variable not in dataset:
                raise ValueError(f"Unknown variable: {variable}")
            array = dataset[variable]
            selectors: dict[str, Any] = {}
            for axis, value in (selection or {}).items():
                if axis not in array.dims:
                    raise ValueError(f"Unknown dimension: {axis}")
                if isinstance(value, int):
                    selectors[axis] = value
                elif (
                    isinstance(value, list)
                    and len(value) in (2, 3)
                    and all(isinstance(v, int) for v in value)
                ):
                    selectors[axis] = slice(*value)
                else:
                    raise ValueError(
                        "Selection values must be integer indices or [start, stop, stride] lists"
                    )
            array = array.isel(selectors)
            if array.size > 4096:
                raise ValueError(
                    "Selection exceeds 4096 values; select fewer times/cells"
                )
            values = array.values
            if values.dtype.kind not in "biuf":
                raise ValueError("Only numeric variable slices can be returned")
            finite_values = values.astype(object)
            finite_values[~np.isfinite(values)] = None
            result["selection"] = {
                "dimensions": list(array.dims),
                "shape": list(array.shape),
                "values": finite_values.tolist(),
            }
        return result
