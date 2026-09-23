"""Discover the inputs of a render case: state file, building geometry, inflow params.

A *case* is a folder (or a single state file) holding an LES state NetCDF with
``u, v, w`` on cell centres ``(time, zt, yt, xt)``. Everything else is optional
and resolved with fallbacks, so both a hand-assembled case folder and a sample
inside a ``training_data/<dataset>/state/<split>/`` tree work unchanged:

========== ================================================================
state      ``state.nc``, else the only ``*.nc`` holding ``u, v, w``
geometry   ``*.stl`` in the folder, else ``attrs["geometry_stl"]`` looked up
           in the folder and in ``<dataset>/geometries/``, else the solid
           voxels of the ``blanking`` variable
params     ``params.nc``, else a ``*.nc`` holding ``inflow_angle``, else the
           training-data twin ``<dataset>/param/<split>/<state name>``
render.yaml optional per-case overrides merged over the preset
========== ================================================================
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Optional

import numpy as np
import trimesh
import xarray as xr

_STATE_VARS = ("u", "v", "w")


@dataclasses.dataclass
class Case:
    name: str
    state_path: pathlib.Path
    geometry_path: Optional[pathlib.Path]
    params_path: Optional[pathlib.Path]
    overrides: dict[str, Any]

    def open_state(self) -> xr.Dataset:
        return xr.open_dataset(self.state_path)

    def open_params(self) -> Optional[xr.Dataset]:
        if self.params_path is None:
            return None
        return xr.open_dataset(self.params_path)

    def buildings(self) -> trimesh.Trimesh:
        """Building surface mesh in the simulation frame (metres, z-up)."""
        if self.geometry_path is not None:
            mesh = trimesh.load(self.geometry_path, force="mesh")
            assert isinstance(mesh, trimesh.Trimesh)
            return mesh
        with self.open_state() as ds:
            return mesh_from_blanking(ds)


def _has_vars(path: pathlib.Path, names: tuple[str, ...]) -> bool:
    try:
        with xr.open_dataset(path) as ds:
            return all(n in ds.variables for n in names)
    except Exception:
        return False


def _find_state(folder: pathlib.Path) -> pathlib.Path:
    if (folder / "state.nc").is_file():
        return folder / "state.nc"
    candidates = [p for p in sorted(folder.glob("*.nc")) if _has_vars(p, _STATE_VARS)]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one state file (state.nc or one *.nc with u, v, w) "
            f"in {folder}, found {[p.name for p in candidates]}. Pass the file "
            f"path directly to pick one."
        )
    return candidates[0]


def _dataset_root(state_path: pathlib.Path) -> Optional[pathlib.Path]:
    """``training_data/<dataset>`` if the state file lives in ``state/<split>/``."""
    parts = state_path.parent.parts
    if len(parts) >= 2 and parts[-2] == "state":
        return state_path.parent.parent.parent
    return None


def _find_geometry(
    folder: pathlib.Path, state_path: pathlib.Path
) -> Optional[pathlib.Path]:
    stls = sorted(folder.glob("*.stl"))
    if len(stls) == 1:
        return stls[0]
    with xr.open_dataset(state_path) as ds:
        stl_name = ds.attrs.get("geometry_stl")
    if stl_name:
        search = [folder, state_path.parent]
        root = _dataset_root(state_path)
        if root is not None:
            search.append(root / "geometries")
        for d in search:
            if (d / str(stl_name)).is_file():
                return d / str(stl_name)
    return None


def _find_params(
    folder: pathlib.Path, state_path: pathlib.Path
) -> Optional[pathlib.Path]:
    if (folder / "params.nc").is_file():
        return folder / "params.nc"
    for p in sorted(folder.glob("*.nc")):
        if p != state_path and _has_vars(p, ("inflow_angle",)):
            return p
    root = _dataset_root(state_path)
    if root is not None:
        twin = root / "param" / state_path.parent.name / state_path.name
        if twin.is_file():
            return twin
    return None


def discover_case(
    path: str | pathlib.Path,
    state: Optional[str | pathlib.Path] = None,
    geometry: Optional[str | pathlib.Path] = None,
    params: Optional[str | pathlib.Path] = None,
) -> Case:
    """Resolve a case from a folder or a state-file path; explicit args win."""
    path = pathlib.Path(path).expanduser().resolve()
    if path.is_file():
        folder, state_path = path.parent, path
    else:
        folder = path
        state_path = pathlib.Path(state) if state else _find_state(folder)
    if state:
        state_path = pathlib.Path(state).expanduser().resolve()

    geometry_path = (
        pathlib.Path(geometry).expanduser().resolve()
        if geometry
        else _find_geometry(folder, state_path)
    )
    params_path = (
        pathlib.Path(params).expanduser().resolve()
        if params
        else _find_params(folder, state_path)
    )

    overrides: dict[str, Any] = {}
    render_yaml = folder / "render.yaml"
    if render_yaml.is_file():
        import yaml

        overrides = yaml.safe_load(render_yaml.read_text()) or {}

    name = state_path.stem if state_path.stem != "state" else folder.name
    return Case(name, state_path, geometry_path, params_path, overrides)


def mesh_from_blanking(ds: xr.Dataset) -> trimesh.Trimesh:
    """Box mesh of the solid cells when no STL is available."""
    from les_render.fields import solid_mask

    solid = solid_mask(ds)  # (x, y, z)
    dx = float(ds.xt[1] - ds.xt[0])
    dy = float(ds.yt[1] - ds.yt[0])
    dz = float(ds.zt[1] - ds.zt[0])
    boxes = []
    for i, j, k in np.argwhere(solid):
        box = trimesh.creation.box(extents=(dx, dy, dz))
        box.apply_translation((float(ds.xt[i]), float(ds.yt[j]), float(ds.zt[k])))
        boxes.append(box)
    mesh: trimesh.Trimesh = trimesh.util.concatenate(boxes)
    mesh.merge_vertices()
    return mesh
