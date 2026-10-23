"""Fit a case STL to the uDALES domain floor: ground coverage and outflow buffer.

uDALES' floor is a wall only where the STL has ground facets. The domain
bottom itself (``BCbotm``) is never a wall when the geometry comes from an STL:
``modibm.f90::bottom`` applies its wall function only with ``lbottom``, which
the wrapper leaves at ``.false.``, and the ghost value ``u0(:,:,kb-1)`` is never
set, so an uncovered floor carries only molecular stress, i.e. it is free-slip.
A domain that reaches past the STL's ground (the upstream fetch of
``xie_and_castro``, x < 0, or an outflow buffer) therefore needs ground facets
added there, or its floor differs from PALM's, where every ground cell is a
rough wall.

Coordinates are the shifted ones uDALES runs in: the domain is
``[0, x_length] x [0, y_length]`` with the floor at the STL's lowest point.
"""

import logging
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# Facet heights and positions in the case STLs are exact to well below this.
TOLERANCE = 1e-6


def _ground_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Mask of upward-facing faces lying on the STL's lowest plane."""
    zmin = mesh.bounds[0, 2]
    on_floor = np.all(np.abs(mesh.triangles[:, :, 2] - zmin) < TOLERANCE, axis=1)
    return on_floor & (mesh.face_normals[:, 2] > 0.99)


def _ground_rectangle(
    x0: float, x1: float, y0: float, y1: float, z: float
) -> trimesh.Trimesh:
    """Two upward-facing triangles covering ``[x0, x1] x [y0, y1]`` at height z."""
    vertices = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _clip_ground(ground: trimesh.Trimesh, x_keep: float) -> trimesh.Trimesh:
    """Keep the part of the (planar, z = const) ground with ``x <= x_keep``.

    Each triangle is clipped against the half-plane (Sutherland-Hodgman) and
    fan-triangulated; vertex order, hence the upward normal, is preserved.
    """
    triangles = []
    for tri in ground.triangles:
        polygon = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            a_in, b_in = a[0] <= x_keep, b[0] <= x_keep
            if a_in:
                polygon.append(a)
            if a_in != b_in:
                t = (x_keep - a[0]) / (b[0] - a[0])
                polygon.append(a + t * (b - a))
        for k in range(1, len(polygon) - 1):
            triangles.append([polygon[0], polygon[k], polygon[k + 1]])
    vertices = np.asarray(triangles, dtype=float).reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    clipped = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    clipped.update_faces(clipped.area_faces > TOLERANCE)
    return clipped


def _drop_buildings_beyond(
    buildings: trimesh.Trimesh, x_keep: float
) -> trimesh.Trimesh:
    """Remove the buildings that lie entirely downstream of ``x_keep``.

    Buildings are the connected components of the non-ground faces. One that
    straddles ``x_keep`` cannot be cut without leaving an open shell, which the
    IBM preprocessing cannot classify, so it raises.
    """
    kept = []
    for part in buildings.split(only_watertight=False):
        xlo, xhi = part.bounds[0, 0], part.bounds[1, 0]
        if xhi <= x_keep + TOLERANCE:
            kept.append(part)
        elif xlo < x_keep - TOLERANCE:
            raise ValueError(
                f"A building spans x = {xlo:.2f}..{xhi:.2f} m (uDALES frame) across "
                f"the end of the domain bounds at x = {x_keep:.2f} m. The outflow "
                "buffer needs bounds that end in a gap between buildings."
            )
    if not kept:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(kept)


def fit_stl_to_domain(
    mesh: trimesh.Trimesh,
    x_length: float,
    y_length: float,
    x_keep: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    """Return ``mesh`` with ground over the whole floor, or None if unchanged.

    Args:
        mesh: Case STL, already shifted into the uDALES frame.
        x_length: Domain length in x (including any outflow buffer).
        y_length: Domain length in y.
        x_keep: End of the domain bounds when an outflow buffer extends the
            grid beyond them. Buildings downstream of it and the ground beyond
            it are removed and replaced by flat ground, as PALM's buffer does
            (``stl_to_palm_topography(x_max=...)``). None keeps every building.

    Returns:
        The modified mesh, or None when the STL already covers the floor along
        x and nothing is removed (so the caller can leave the file untouched).
        STLs without ground facets are returned unchanged (None): their floor
        was never a wall, so there is nothing to extend.
    """
    ground_mask = _ground_faces(mesh)
    if not ground_mask.any():
        return None
    ground = mesh.submesh([np.nonzero(ground_mask)[0]], append=True)
    buildings = mesh.submesh([np.nonzero(~ground_mask)[0]], append=True)
    z_floor = float(mesh.bounds[0, 2])

    changed = False
    if x_keep is not None and x_keep < x_length - TOLERANCE:
        n_before = len(buildings.faces)
        buildings = _drop_buildings_beyond(buildings, x_keep)
        if ground.bounds[1, 0] > x_keep + TOLERANCE:
            ground = _clip_ground(ground, x_keep)
            changed = True
        changed = changed or len(buildings.faces) != n_before

    gx0, gx1 = float(ground.bounds[0, 0]), float(ground.bounds[1, 0])
    strips = []
    if gx0 > TOLERANCE:
        strips.append((0.0, gx0))
    if gx1 < x_length - TOLERANCE:
        strips.append((gx1, x_length))
    if not strips and not changed:
        return None

    parts = [buildings, ground] if len(buildings.faces) else [ground]
    for x0, x1 in strips:
        parts.append(_ground_rectangle(x0, x1, 0.0, y_length, z_floor))
        logger.info(
            "Added ground facets over x = %.2f..%.2f m (uDALES frame): the STL's "
            "ground does not reach there, which would leave a free-slip floor.",
            x0,
            x1,
        )
    return trimesh.util.concatenate(parts)
