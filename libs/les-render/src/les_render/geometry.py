"""Export building and ground meshes.

Files are written in the simulation frame (metres, right-handed, z-up):

* ``geometry/buildings.glb`` / ``ground.glb`` -- glTF binary. glTF is y-up, so
  the exporter rotates z-up -> y-up; UE's Interchange and Blender's importer
  both convert glTF back to their z-up frames (UE also to cm and left-handed).
* ``geometry/buildings.obj`` / ``ground.obj`` -- raw z-up metres, no axis
  metadata (Blender preview reads these 1:1).
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import trimesh

# z-up (x, y, z) -> glTF y-up (x, z, -y)
_ZUP_TO_YUP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def ground_plane(
    lower: np.ndarray, upper: np.ndarray, margin: float = 0.0
) -> trimesh.Trimesh:
    """A single quad at z=0 covering the domain footprint (+ margin), UV 0..1."""
    x0, y0 = lower[0] - margin, lower[1] - margin
    x1, y1 = upper[0] + margin, upper[1] + margin
    verts = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(verts, faces, process=False)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    return mesh


def _write(mesh: trimesh.Trimesh, stem: pathlib.Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(stem.with_suffix(".obj"))
    yup = mesh.copy()
    yup.apply_transform(_ZUP_TO_YUP)
    yup.export(stem.with_suffix(".glb"))
    return {
        "obj": f"{stem.parent.name}/{stem.name}.obj",
        "glb": f"{stem.parent.name}/{stem.name}.glb",
    }


def export_geometry(
    buildings: trimesh.Trimesh,
    lower: np.ndarray,
    upper: np.ndarray,
    out_dir: pathlib.Path,
    ground_margin: float = 0.0,
) -> dict[str, Any]:
    """Write buildings + ground; return the manifest ``geometry`` block."""
    geo = out_dir / "geometry"
    b = buildings.copy()
    b.merge_vertices()
    b.fix_normals()
    return {
        "buildings": _write(b, geo / "buildings"),
        "ground": _write(ground_plane(lower, upper, ground_margin), geo / "ground"),
        "buildings_bounds": b.bounds.tolist(),
        "max_building_height": float(b.bounds[1, 2]),
        "footprints": building_footprints(b),
    }


def building_footprints(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    """Axis-aligned bounds of each connected building (used for camera framing)."""
    out = []
    for part in mesh.split(only_watertight=False):
        lo, hi = part.bounds
        out.append({"min": lo.round(3).tolist(), "max": hi.round(3).tolist()})
    return out
