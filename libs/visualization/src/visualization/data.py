"""Backend-free, frame-at-a-time reading and physical grid collocation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml
from evaluation.turbulence import stl_solid_mask

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


def _locate(path: str | Path, root: Path) -> Path | None:
    """A run config's repository-relative path, from the run dir up or the cwd."""
    path = Path(path)
    if path.is_absolute():
        return path if path.is_file() else None
    candidates = [parent / path for parent in root.parents] + [Path.cwd() / path]
    return next((c.resolve() for c in candidates if c.is_file()), None)


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
    """Read a run's ``state.nc`` frame by frame, never a whole ensemble history.

    ``run_root`` is a forward run directory (``scripts/run_forward.py``): one
    ``state.nc`` with numeric ``time`` in seconds and an ``ensemble`` dimension
    only for ensembles. Its ``config.yaml``, when present, names backend and case.

    Without a ``blanking`` variable the solid cells come from the buildings' STL:
    ``geometry`` or else the config's ``geometry.stl_path``, through
    ``evaluation.turbulence.stl_solid_mask``, the mask the metrics use.
    """

    def __init__(
        self,
        run_root: str | Path,
        *,
        member: Any = None,
        reduction: str | None = None,
        max_cells: int = 8_000_000,
        geometry: str | Path | None = None,
    ):
        self.root = Path(run_root).resolve()
        self.member = member
        self.reduction = reduction
        self.max_cells = max_cells
        if member is not None and reduction is not None:
            raise ValueError("Choose a member or a reduction, not both")
        if reduction not in (None, "mean_velocity", "mean_speed"):
            raise ValueError("Reduction must be mean_velocity or mean_speed")
        self.path = self.root / "state.nc"
        self.sources = [{"path": "state.nc", "sha256": fingerprint(self.path)}]
        config_path = self.root / "config.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        self.backend = str((config.get("model") or {}).get("name", "unknown"))
        self.case = str(config.get("case_name", self.root.name))
        stl = geometry or (config.get("geometry") or {}).get("stl_path")
        self.geometry = _locate(stl, self.root) if stl else None
        if geometry and self.geometry is None:
            raise ValueError(f"Geometry STL not found: {geometry}")
        self.solid = None
        self.warnings = []
        if stl and self.geometry is None:
            self.warnings.append(f"Case STL {stl} not found; buildings not masked")
        with xr.open_dataset(self.path, decode_times=False) as ds:
            if "time" not in ds.coords:
                raise ValueError("State artifacts require an explicit time coordinate")
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
                raise ValueError("State times must be finite and strictly increasing")
            members = list(ds.ensemble.values) if "ensemble" in ds.dims else [None]
            if self.geometry is not None and "blanking" not in ds:
                coords = _coordinates(ds)
                solid = stl_solid_mask(
                    self.geometry, coords["z"], coords["y"], coords["x"]
                )
                self.solid = xr.DataArray(
                    solid.astype(np.int8), dims=("z", "y", "x"), coords=coords
                )
        if self.geometry is not None:
            self.sources.append(
                {"path": str(self.geometry), "sha256": fingerprint(self.geometry)}
            )
        self.ensemble = len(members) > 1
        if self.ensemble and member is None and reduction is None:
            raise ValueError(
                "Ensemble visualization requires an explicit member or reduction"
            )
        if member is not None and member not in members:
            raise ValueError(
                f"Unknown ensemble member {member!r}; available: {sorted(str(m) for m in members)}"
            )
        if not len(times):
            raise ValueError("No state frames available")
        self.members = members if member is None else [member]
        self.positions = {float(time): i for i, time in enumerate(times)}
        self.times = sorted(self.positions)

    def frame(self, time: float) -> xr.Dataset:
        frames = []
        warnings = set(self.warnings)
        with xr.open_dataset(self.path, decode_times=False) as source:
            snapshot = source.isel(time=self.positions[time], drop=True)
            snapshot = snapshot[
                [name for name in ("u", "v", "w", "blanking") if name in snapshot]
            ]
            if self.solid is not None:
                snapshot["blanking"] = self.solid
            cells = sum(snapshot[name].size for name in ("u", "v", "w"))
            if "ensemble" in snapshot.dims:
                cells //= snapshot.sizes["ensemble"]
            if cells * len(self.members) > self.max_cells * 3:
                raise ValueError(
                    f"Frame exceeds max_cells={self.max_cells}; use a smaller simulation/grid"
                )
            for label in self.members:
                ds = snapshot
                if "ensemble" in ds.dims:
                    ds = ds.sel(ensemble=label, drop=True)
                normalized = normalize(ds.load())
                frames.append(normalized)
                warnings.update(normalized.attrs["warnings"])
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
