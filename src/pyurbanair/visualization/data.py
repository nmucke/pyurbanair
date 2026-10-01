"""Backend-free, frame-at-a-time reading and physical grid collocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

AXES = {
    "x": ("x", "xt", "xm", "xu"),
    "y": ("y", "yt", "ym", "yv"),
    "z": ("z", "zt", "zm", "zw"),
}
VELOCITY_UNITS = {"m/s", "m s-1", "m s^-1", "m s**-1", "m.s-1"}


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def _coordinates(ds: xr.Dataset) -> dict[str, np.ndarray]:
    coordinates = {}
    for axis, aliases in AXES.items():
        name = next((name for name in aliases if name in ds.coords), None)
        if name is None:
            raise ValueError(f"Missing named physical {axis} coordinate")
        coord = ds.coords[name]
        units = coord.attrs.get("units", "m")
        if units not in ("m", "meter", "meters", "metre", "metres"):
            raise ValueError(f"Coordinate {name} must be in metres, got {units!r}")
        values = np.asarray(coord.values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Coordinate {name} must be finite and one-dimensional")
        delta = np.diff(values)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"Coordinate {name} must be strictly monotone")
        coordinates[axis] = np.sort(values)
    return coordinates


def _collocate(
    da: xr.DataArray, coords: dict[str, np.ndarray], *, mask: bool = False
) -> xr.DataArray:
    rename = {}
    for axis, aliases in AXES.items():
        names = [name for name in aliases if name in da.dims]
        if len(names) != 1:
            raise ValueError(f"{da.name}: expected one {axis} dimension, got {names}")
        coord_units = da[names[0]].attrs.get("units", "m")
        if coord_units not in ("m", "meter", "meters", "metre", "metres"):
            raise ValueError(
                f"Coordinate {names[0]} must be in metres, got {coord_units!r}"
            )
        rename[names[0]] = axis
    da = da.rename(rename)
    extra = set(da.dims) - set(AXES)
    if extra:
        raise ValueError(
            f"Select time/member before collocation; extra dimensions: {extra}"
        )
    for axis, target in coords.items():
        source = np.asarray(da[axis].values, dtype=float)
        if not np.isfinite(source).all() or len(np.unique(source)) != len(source):
            raise ValueError(f"Invalid {axis} coordinates for {da.name}")
        if len(source) > 1 and not (
            np.all(np.diff(source) > 0) or np.all(np.diff(source) < 0)
        ):
            raise ValueError(f"Nonmonotone {axis} coordinates for {da.name}")
        da = da.sortby(axis)
        if not np.array_equal(da[axis].values, target):
            if len(source) == 1:
                # A singleton axis has no support for interpolation elsewhere.
                da = da.reindex({axis: target})
            else:
                da = da.interp({axis: target}, method="nearest" if mask else "linear")
    return da.transpose("z", "y", "x")


def normalize(ds: xr.Dataset) -> xr.Dataset:
    """Collocate vectors at named centre coordinates; never extrapolate boundaries.

    ``blanking != 0`` means solid. Unknown/out-of-domain interpolation remains NaN,
    distinct from valid zero velocity. An absent mask is recorded, never inferred.
    """
    coords = _coordinates(ds)
    fields = {}
    warnings = []
    for name in ("u", "v", "w"):
        if name not in ds:
            raise ValueError(f"Missing velocity component {name}")
        units = ds[name].attrs.get("units")
        if units is None:
            warnings.append(
                f"{name} units absent; repository velocity convention m/s assumed"
            )
        elif str(units).strip() not in VELOCITY_UNITS:
            raise ValueError(f"Unsupported {name} units {units!r}; expected m/s")
        fields[name] = _collocate(ds[name], coords)
    result = xr.Dataset(fields)
    if "blanking" in ds:
        blanking = _collocate(ds.blanking, coords, mask=True)
        result["blanking"] = blanking
        for name in fields:
            result[name] = result[name].where(blanking == 0)
    else:
        warnings.append(
            "No explicit solid mask; zero velocity is treated as valid fluid"
        )
    result.attrs["warnings"] = warnings
    return add_magnitudes(result)


def add_magnitudes(ds: xr.Dataset) -> xr.Dataset:
    ds["horizontal_speed"] = np.sqrt(ds.u**2 + ds.v**2)
    ds["speed"] = np.sqrt(ds.u**2 + ds.v**2 + ds.w**2)
    for name in ("u", "v", "w", "horizontal_speed", "speed"):
        ds[name].attrs["units"] = "m/s"
    return ds


class ArtifactReader:
    """Read indexed windows/members without materializing an ensemble history."""

    def __init__(
        self,
        run_root: str | Path,
        *,
        member: Any = None,
        reduction: str | None = None,
        max_cells: int = 8_000_000,
    ):
        self.root = Path(run_root).resolve()
        self.member = member
        self.reduction = reduction
        self.max_cells = max_cells
        if member is not None and reduction is not None:
            raise ValueError("Choose a member or a reduction, not both")
        if reduction not in (None, "mean_velocity", "mean_speed"):
            raise ValueError("Reduction must be mean_velocity or mean_speed")
        index_path = self.root / "artifact_index.json"
        self.index = json.loads(index_path.read_text()) if index_path.exists() else {}
        if self.index and self.index.get("status") != "complete":
            raise ValueError("Visualization requires complete numerical artifacts")
        entries = [
            item for item in self.index.get("artifacts", []) if item["kind"] == "state"
        ]
        if not entries:
            entries = [{"path": "state.nc", "window": 0}]
        windows = sorted({entry.get("window", 0) for entry in entries})
        if windows != list(range(len(windows))) or self.index.get(
            "total_windows", len(windows)
        ) != len(windows):
            raise ValueError("Artifact index contains missing rollout windows")
        self.sources = []
        self.frames: dict[float, list[tuple[Path, int, Any]]] = {}
        members = set()
        window_ranges: dict[int, tuple[float, float]] = {}
        for entry in entries:
            path = contained(self.root, entry["path"])
            digest = fingerprint(path)
            if entry.get("sha256") and digest != entry["sha256"]:
                raise ValueError(f"Artifact fingerprint changed: {entry['path']}")
            self.sources.append(
                {
                    "path": entry["path"],
                    "sha256": digest,
                    "window": entry.get("window", 0),
                }
            )
            with xr.open_dataset(path, decode_times=False) as ds:
                if "time" not in ds.coords:
                    raise ValueError(
                        "State artifacts require an explicit time coordinate"
                    )
                times = np.asarray(ds.time.values)
                if not np.issubdtype(times.dtype, np.number):
                    raise ValueError("State time must be numeric simulation seconds")
                time_units = str(ds.time.attrs.get("units", "s"))
                if time_units not in ("s", "second", "seconds"):
                    raise ValueError(
                        f"State time must be in simulation seconds, got {time_units!r}"
                    )
                times = times.astype(float)
                if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                    raise ValueError(
                        "State times must be finite and strictly increasing"
                    )
                if len(times):
                    window = int(entry.get("window", 0))
                    extent = (float(times[0]), float(times[-1]))
                    if window in window_ranges and window_ranges[window] != extent:
                        raise ValueError(
                            f"Members have inconsistent times in window {window}"
                        )
                    window_ranges[window] = extent
                labels = list(ds.ensemble.values) if "ensemble" in ds.dims else [None]
                members.update(labels)
                selected = [
                    label for label in labels if member is None or label == member
                ]
                for position, time in enumerate(times):
                    for label in selected:
                        self.frames.setdefault(float(time), []).append(
                            (path, position, label)
                        )
        for previous, current in zip(windows, windows[1:]):
            if (
                previous in window_ranges
                and current in window_ranges
                and window_ranges[current][0] < window_ranges[previous][1]
            ):
                raise ValueError("Rollout windows overlap beyond a shared endpoint")
        self.ensemble = len(members) > 1
        if self.ensemble and member is None and reduction is None:
            raise ValueError(
                "Ensemble visualization requires an explicit member or reduction"
            )
        if member is not None and member not in members:
            raise ValueError(
                f"Unknown ensemble member {member!r}; available: {sorted(str(m) for m in members)}"
            )
        if not self.frames:
            raise ValueError("No state frames available")
        self.times = sorted(self.frames)
        self.members = members if member is None else {member}
        # Every selected time must have every requested member; never bridge omissions.
        for time, refs in self.frames.items():
            if {ref[2] for ref in refs} != self.members:
                raise ValueError(f"Missing ensemble members at simulation time {time}")

    def frame(self, time: float) -> xr.Dataset:
        by_member: dict[Any, xr.Dataset] = {}
        warnings = set()
        for path, position, label in self.frames[time]:
            with xr.open_dataset(path, decode_times=False) as source:
                ds = source.isel(time=position, drop=True)
                if "ensemble" in ds.dims:
                    ds = ds.sel(ensemble=label, drop=True)
                ds = ds[[name for name in ("u", "v", "w", "blanking") if name in ds]]
                cells = sum(ds[name].size for name in ("u", "v", "w"))
                if cells * len(self.members) > self.max_cells * 3:
                    raise ValueError(
                        f"Frame exceeds max_cells={self.max_cells}; use a smaller simulation/grid"
                    )
                normalized = normalize(ds.load())
            if label in by_member:
                # Shared rollout endpoints may be deduplicated only if scientifically equal.
                if not normalized.equals(by_member[label]):
                    raise ValueError(
                        f"Conflicting shared-window endpoint at t={time}, member={label}"
                    )
            else:
                by_member[label] = normalized
            warnings.update(normalized.attrs["warnings"])
        frames = list(by_member.values())
        if len(frames) == 1:
            result = frames[0]
        else:
            if sum(ds.u.size for ds in frames) > self.max_cells:
                raise ValueError("Selected ensemble frame exceeds max_cells budget")
            stacked = xr.concat(frames, dim="selected_member", join="exact")
            result = stacked.mean("selected_member", skipna=False)
            if self.reduction == "mean_velocity":
                result = add_magnitudes(result)
        result.attrs["warnings"] = sorted(warnings)
        return result
