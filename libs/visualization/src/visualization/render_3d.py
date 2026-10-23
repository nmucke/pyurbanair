"""Export of the interactive 3D view: the scene the viewer draws with WebGL.

``media/flow-3d/scene.json`` (version 1), every path relative to it:

- ``x``, ``y``, ``z``: cell-centre coordinates in metres (increasing)
- ``times``: the saved simulation seconds shown; ``velocity[i]`` is an Int16
  little-endian file of planar u, v, w, each (nz, ny, nx) with x fastest;
  physical = int × ``velocity_scale[i]``; solid/invalid cells are stored as 0
- ``fluid``: Uint8 (nz, ny, nx), x fastest, 1 = fluid (``blanking == 0`` and
  finite velocity in the first frame)
- ``speed_limits`` [0, 99.5th percentile] m/s of the full 3D speed over the
  fluid cells of all exported frames (or ``color_limits`` when given), ``units``
- ``level_speed_limits``: per z level, [0, 99th percentile] of the same speed
  over that level's fluid cells and all exported frames (or ``color_limits``);
  null without fluid
- ``buildings``: Float32 triangle soup (n_tri × 9, metres), the STL cut to the
  cell-face domain, or null without an STL
- ``case``, ``backend``, ``camera`` (null or ``[position, focal_point, up]``)

Frames are read one at a time; ``check_scene_size`` bounds the export.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from evaluation.turbulence import read_binary_stl

from .data import fingerprint

if TYPE_CHECKING:
    from .data import ArtifactReader
    from .render import RenderOptions

MAX_SCENE_BYTES = 512 * 1024**2
# The speed percentile comes from a histogram: at most a bin above the exact one.
SPEED_BINS = 4096


def scene_frames(sizes: Mapping[Any, int]) -> int:
    """How many frames fit a 3D scene in ``MAX_SCENE_BYTES``.

    The scene is a Uint8 fluid mask plus Int16 u, v, w per frame. Raises
    ``ValueError`` when an axis has fewer than two cells.
    """
    if any(sizes[axis] < 2 for axis in "xyz"):
        raise ValueError("The 3D view needs at least two cells on each axis")
    cells = sizes["x"] * sizes["y"] * sizes["z"]
    return (MAX_SCENE_BYTES - cells) // (3 * 2 * cells)


def check_scene_size(ds: xr.Dataset, frames: int) -> None:
    """Refuse an export over ``MAX_SCENE_BYTES`` before anything is written."""
    fit = scene_frames(ds.sizes)
    if frames > fit:
        raise ValueError(
            f"The 3D scene holds at most {fit} frames in "
            f"{MAX_SCENE_BYTES // 1024**2} MiB, not {frames}; increase stride or "
            "shorten the time interval"
        )


def cell_faces(coordinates: np.ndarray) -> tuple[float, float]:
    """Outer cell faces of a centre axis: half a spacing beyond each end."""
    return (
        float(coordinates[0] - (coordinates[1] - coordinates[0]) / 2),
        float(coordinates[-1] + (coordinates[-1] - coordinates[-2]) / 2),
    )


def clip_triangles(tri: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Cut ``(n, 3, 3)`` triangles to the box [lo, hi]; inner ones stay as they are."""
    inside = ((tri >= lo) & (tri <= hi)).all((1, 2))
    crossing = ((tri.max(1) >= lo) & (tri.min(1) <= hi)).all(1) & ~inside
    pieces = [tri[inside]]
    for polygon in tri[crossing]:
        # Sutherland-Hodgman against the six faces, then a triangle fan.
        for axis in range(3):
            for bound, sign in ((lo[axis], 1), (hi[axis], -1)):
                distance = sign * (polygon[:, axis] - bound)
                clipped = []
                for i in range(len(polygon)):
                    j = (i + 1) % len(polygon)
                    if distance[i] >= 0:
                        clipped.append(polygon[i])
                    if (distance[i] >= 0) != (distance[j] >= 0):
                        weight = distance[i] / (distance[i] - distance[j])
                        clipped.append(polygon[i] + weight * (polygon[j] - polygon[i]))
                polygon = np.reshape(clipped, (-1, 3))
        if len(polygon) >= 3:
            fan = [np.repeat(polygon[:1], len(polygon) - 2, 0), polygon[1:-1]]
            pieces.append(np.stack(fan + [polygon[2:]], 1))
    return np.concatenate(pieces)


def render_3d(
    reader: ArtifactReader,
    output: Path,
    times: list[float],
    repeats: list[int],
    opts: RenderOptions,
    limits: list[float] | None = None,
) -> dict[str, Any]:
    scene_dir = output / "media" / "flow-3d"
    (scene_dir / "velocity").mkdir(parents=True)
    first = reader.frame(times[0])
    x, y, z = (first[axis].values.astype(float) for axis in "xyz")
    faces = np.array([cell_faces(values) for values in (x, y, z)])
    geometry = None
    buildings = None
    if reader.geometry is not None:
        path = reader.geometry
        if path.suffix.lower() != ".stl":
            raise ValueError("3D geometry must be an STL in the state coordinate frame")
        tri = read_binary_stl(path)
        if (
            not len(tri)
            or np.any(tri.max((0, 1)) < faces[:, 0])
            or np.any(tri.min((0, 1)) > faces[:, 1])
        ):
            raise ValueError(
                "STL does not overlap the state domain; check coordinate alignment"
            )
        # An STL can cover a much larger city than this saved simulation. Show
        # only its intersection with the cell-face domain, in physical metres.
        tri = clip_triangles(tri, faces[:, 0], faces[:, 1])
        tri.astype("<f4").tofile(scene_dir / "buildings.bin")
        buildings = "buildings.bin"
        geometry = {
            "path": str(path),
            "sha256": fingerprint(path),
            "scale": 1.0,
            "display_bounds": faces.ravel().tolist(),
            "clipping": "triangles cut at the saved field's cell-face domain; no rescaling",
            "triangles": len(tri),
        }
    fluid = np.isfinite(
        np.stack([first[name].transpose("z", "y", "x").values for name in "uvw"])
    ).all(0)
    if "blanking" in first:
        fluid &= first.blanking.transpose("z", "y", "x").values == 0
    fluid.astype(np.uint8).tofile(scene_dir / "fluid.bin")
    paths, scales, peak = [], [], 0.0
    for index, time in enumerate(times):
        ds = first if index == 0 else reader.frame(time)
        velocity = np.stack(
            [ds[name].transpose("z", "y", "x").values for name in "uvw"]
        )
        velocity = np.where(fluid & np.isfinite(velocity), velocity, 0.0)
        scale = max(float(np.abs(velocity).max()), 1e-6) / 32767
        relative = f"velocity/{index:05d}.bin"
        np.round(velocity / scale).astype("<i2").tofile(scene_dir / relative)
        paths.append(relative)
        scales.append(scale)
        peak = max(peak, float(np.linalg.norm(velocity, axis=0).max()))
    # A second pass over the written frames, one at a time: a speed histogram
    # per z level, for the per-level and the overall percentiles.
    width = (peak or 1.0) / SPEED_BINS
    counts = np.zeros((len(z), SPEED_BINS))
    levels = np.broadcast_to(np.arange(len(z))[:, None, None], fluid.shape)[fluid]
    for relative, scale in zip(paths, scales):
        velocity = np.fromfile(scene_dir / relative, "<i2").reshape(3, *fluid.shape)
        speed = np.linalg.norm(velocity * scale, axis=0)[fluid]
        bins = np.minimum(speed / width, SPEED_BINS - 1).astype(int)
        counts += np.bincount(
            levels * SPEED_BINS + bins, minlength=counts.size
        ).reshape(counts.shape)

    def percentile(histogram: np.ndarray, q: float) -> float:
        rank = np.searchsorted(np.cumsum(histogram), q * histogram.sum())
        return float(min(rank, SPEED_BINS - 1) + 1) * width if peak > 0 else 1.0

    # Explicit limits hold everywhere, so both browser modes show them.
    level_limits = [
        (limits or [0.0, percentile(level, 0.99)]) if level.sum() else None
        for level in counts
    ]
    if limits is None:
        limits = [0.0, percentile(counts.sum(0), 0.995)]
    scene = {
        "version": 1,
        "case": reader.case,
        "backend": reader.backend,
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "times": times,
        "speed_limits": limits,
        "level_speed_limits": level_limits,
        "units": "m/s",
        "fluid": "fluid.bin",
        "velocity": paths,
        "velocity_scale": scales,
        "buildings": buildings,
        "camera": opts.camera,
    }
    (scene_dir / "scene.json").write_text(json.dumps(scene, allow_nan=False))
    # The 2D movies' pacing: a saved time starts where its 2D frames start.
    starts = np.cumsum([0] + repeats[:-1])
    return {
        "id": "flow-3d",
        "kind": "3d",
        "label": "3D flow",
        "scene": "media/flow-3d/scene.json",
        "field": "speed",
        "units": "m/s",
        "color_limits": limits,
        "geometry": geometry,
        "frames": [
            {"video_time": int(start) / opts.fps, "simulation_time": time}
            for start, time in zip(starts, times)
        ],
        "duration": sum(repeats) / opts.fps,
    }
