"""Per-frame marching-cubes isosurface export (binary PLY).

Writes one ``.ply`` file per exported frame under
``<out_dir>/isosurfaces/<name>/<name>.<FFFF>.ply``: a triangle mesh of
``iso_variable == level`` (default: ``q_criterion``, the standard
rotation-vs-strain vortex-core indicator), with vertices coloured by
``color_variable`` (default ``speed``). ``export_isosurfaces(fields,
timeline, spec, out_dir) -> dict`` returns the manifest ``layer`` entry for
``type: "isosurface"``.

Mesh construction, per frame
-----------------------------
1. ``fields.scalar_field(fields, iso_variable, t, upsample)`` -- tricubic
   upsample of velocity *before* differentiating, so Q is smooth rather than
   blocky (solid cells already zeroed by ``fields.py``).
2. A small Gaussian pre-smooth (``smooth_sigma`` voxels, default ``0.7``)
   knocks down single-voxel numerical noise that would otherwise show up as
   isolated fuzzy islands. Smoothing can leak positive Q from fluid cells
   into solid ones (diffusion doesn't know about walls), which would draw a
   spurious shell *inside* buildings -- so solid cells are explicitly
   re-zeroed after smoothing, every frame. This is the only source of wall
   artifacts we found; without the re-zero, a thin shell hugs every solid
   voxel face.
3. ``skimage.measure.marching_cubes(volume, level=level, spacing=grid.spacing)``,
   then vertices are offset by ``grid.origin`` (marching_cubes returns
   coordinates in the volume's own index-times-spacing frame, i.e. index
   ``(0,0,0)`` at world ``(0,0,0)``; adding the grid origin puts them in the
   simulation frame). If ``level`` falls outside ``[volume.min(),
   volume.max()]`` for a given frame (e.g. an early, still-laminar frame),
   marching_cubes has no surface to report -- that frame's file is written
   with zero vertices/faces rather than raising.
4. Face-count control: connected components are found on the *face*
   adjacency graph (``trimesh.graph.connected_components``, operating
   directly on the marching-cubes topology -- no vertex merging, so vertex
   order used for colour sampling stays untouched). Components smaller than
   ``min_component_faces`` (default 50) are dropped outright -- these are
   almost always single-voxel numerical specks, not real vortex cores, so
   this alone is usually enough to keep meshes small without any true
   decimation. If the mesh is still over ``max_faces`` (default 400,000)
   after that, remaining components are kept largest-first until the budget
   is hit and the rest dropped. This is a cheap approximation to real
   decimation (trimesh's ``simplify_quadric_decimation`` needs an optional
   extra dependency) -- it can only remove whole components, so the actual
   result can undershoot ``max_faces`` by up to one component's worth of
   triangles; it never exceeds it except when a single component alone is
   already over budget (never split).
5. Colour: ``color_variable`` is computed on the same (upsample, time)
   scalar field and sampled at each kept vertex with ``fields.trilinear``
   (trilinear in the refined grid's index space -- vertices don't sit on
   grid nodes). The manifest colour block's LUT maps it to sRGB uint8
   ``red/green/blue``; the raw sampled value is also stored per-vertex as a
   float ``value`` property, so a renderer can re-map colour without
   re-deriving it from geometry.

Auto ``level``
--------------
There's no principled absolute threshold for Q (it's a signed, unbounded,
extremely peaked quantity), and re-deriving it per frame would make the
surface visibly swell/shrink/flicker as turbulence intensity fluctuates
frame to frame. Instead ``level`` is computed *once*, from a handful of
sample frames spread across the timeline: it's ``level_fraction`` (default
``0.15``) times the mean, over those samples, of the ``level_percentile``th
(default 99th) percentile of *positive*, post-smoothing Q. That is: find
roughly how strong the strongest vortex cores get over the run, then draw
the surface at a fixed fraction of that -- strong enough to exclude
background numerical noise, low enough to still catch real structures in
quieter frames. Held constant afterwards so the surface doesn't flicker.

Spec keys (all optional)
-------------------------
``name`` (str, default ``"q_criterion"``) -- layer name / file stem.
``iso_variable`` (str, default ``"q_criterion"``) -- one of ``fields.SCALARS``.
``level`` (float, default: auto, see above).
``level_percentile`` / ``level_fraction`` (float, default ``99.0`` / ``0.15``)
    Only used to compute the auto level.
``upsample`` (int, default ``2``) -- applied to both ``iso_variable`` and
``color_variable``.
``smooth_sigma`` (float, default ``0.7``) -- Gaussian pre-smooth, in voxels
of the upsampled grid; ``0`` disables it.
``color_variable`` (str, default ``"speed"``).
``colormap`` (str, default ``"viridis"``) -- perceptually uniform, reads
well for a velocity-magnitude-like quantity under any lighting.
``color_range`` ([lo, hi], default: auto robust range of ``color_variable``,
computed the same way / at the same time as the auto level).
``max_faces`` (int, default ``400_000``).
``min_component_faces`` (int, default ``50``).
``frame_step`` (int, default ``2``).
``workers`` (int, default ``4``) -- see the parallelism note in
``volumes.py``; the same forkserver-process/thread-pool split is used here,
against the same repo DRAM-bandwidth-bound-past-~4-8-workers caveat.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import pathlib
from typing import Any, Optional

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from . import colormaps
from .fields import FieldSeries, robust_range, scalar_field, trilinear, upsample_mask
from .timeline import Timeline

_N_AUTO_SAMPLES = 4

_FIELDS: Optional[FieldSeries] = None


def default_isosurface_specs() -> list[dict[str, Any]]:
    return [{"name": "q_criterion"}]


def _file_frames(timeline: Timeline, frame_step: int) -> list[tuple[int, int, float]]:
    times = timeline.frame_times
    video_frames = list(range(0, timeline.n_frames, frame_step))
    return [(f, vf, float(times[vf])) for f, vf in enumerate(video_frames)]


def _smoothed(vals: np.ndarray, solid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return vals
    out = np.asarray(ndimage.gaussian_filter(vals, sigma=sigma, mode="nearest"))
    out[solid] = 0.0
    return out


def _auto_level_and_color_range(
    fields: FieldSeries,
    iso_variable: str,
    color_variable: str,
    timeline: Timeline,
    upsample: int,
    smooth_sigma: float,
    level_percentile: float,
    level_fraction: float,
    n_samples: int = _N_AUTO_SAMPLES,
) -> tuple[float, float, float]:
    times = np.linspace(timeline.frame_times[0], timeline.frame_times[-1], n_samples)
    solid = upsample_mask(fields.solid, upsample) if upsample > 1 else fields.solid
    fluid = ~solid
    peaks = []
    color_samples = []
    for t in times:
        q, _ = scalar_field(fields, iso_variable, float(t), upsample)
        q = _smoothed(q, solid, smooth_sigma)
        pos = q[q > 0]
        if pos.size:
            peaks.append(float(np.percentile(pos, level_percentile)))
        c, _ = scalar_field(fields, color_variable, float(t), upsample)
        color_samples.append(c[fluid])
    high = float(np.mean(peaks)) if peaks else 1.0
    level = max(level_fraction * high, 1e-9)
    color_flat = (
        np.concatenate(color_samples)
        if color_samples
        else np.zeros(1, dtype=np.float32)
    )
    lo, hi = robust_range(color_flat, lo=1.0, hi=99.5)
    return level, float(lo), float(hi)


def _limit_faces(
    verts: np.ndarray, faces: np.ndarray, min_component_faces: int, max_faces: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop tiny connected components, then cap total faces by dropping the
    smallest remaining components. Returns (verts_kept, faces_reindexed,
    original_vertex_indices_kept)."""
    if faces.shape[0] == 0:
        return verts[:0], faces[:0], np.zeros(0, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1
    )
    groups = [g for g in groups if len(g) >= min_component_faces]
    groups.sort(key=len, reverse=True)

    kept: list[np.ndarray] = []
    total = 0
    for g in groups:
        if total >= max_faces:
            break
        kept.append(g)
        total += len(g)

    if not kept:
        return verts[:0], faces[:0], np.zeros(0, dtype=np.int64)

    kept_face_idx = np.concatenate(kept)
    kept_faces = faces[kept_face_idx]
    used_verts, remap = np.unique(kept_faces, return_inverse=True)
    new_faces = remap.reshape(kept_faces.shape).astype(np.int64)
    return verts[used_verts], new_faces, used_verts


def _write_ply(
    path: pathlib.Path,
    verts: np.ndarray,
    faces: np.ndarray,
    colors_u8: np.ndarray,
    values: np.ndarray,
) -> None:
    """Binary-little-endian PLY with float xyz, uchar rgb, float `value`."""
    n_v, n_f = verts.shape[0], faces.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float value\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    vdt = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("value", "<f4"),
        ]
    )
    varr = np.zeros(n_v, dtype=vdt)
    if n_v:
        varr["x"], varr["y"], varr["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        varr["red"], varr["green"], varr["blue"] = (
            colors_u8[:, 0],
            colors_u8[:, 1],
            colors_u8[:, 2],
        )
        varr["value"] = values

    fdt = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
    farr = np.zeros(n_f, dtype=fdt)
    if n_f:
        farr["n"] = 3
        farr["idx"] = faces.astype(np.int32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(varr.tobytes())
        fh.write(farr.tobytes())


def _init_pool(source: str) -> None:
    global _FIELDS
    from .fields import open_fields

    _FIELDS = open_fields(source)


def _render_isosurface_file(job: dict[str, Any]) -> dict[str, Any]:
    fields = _FIELDS
    assert fields is not None, "isosurface pool worker used before initialisation"

    upsample = job["upsample"]
    solid = upsample_mask(fields.solid, upsample) if upsample > 1 else fields.solid

    q, grid = scalar_field(fields, job["iso_variable"], job["t"], upsample)
    q = _smoothed(q, solid, job["smooth_sigma"])

    level = job["level"]
    verts_world = np.zeros((0, 3), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.int64)
    if q.size and q.min() < level < q.max():
        verts, faces, _normals, _values = measure.marching_cubes(
            q, level=level, spacing=tuple(grid.spacing)
        )
        verts_world = verts + grid.origin

    verts_world, faces, _used = _limit_faces(
        verts_world, faces, job["min_component_faces"], job["max_faces"]
    )

    colors_u8 = np.zeros((verts_world.shape[0], 3), dtype=np.uint8)
    values = np.zeros(verts_world.shape[0], dtype=np.float32)
    if verts_world.shape[0]:
        color_field, color_grid = scalar_field(
            fields, job["color_variable"], job["t"], upsample
        )
        idx = color_grid.to_index(verts_world)
        values = trilinear(color_field, idx).astype(np.float32)
        lo, hi = job["color_range"]
        rgb = colormaps.apply(values, lo, hi, job["colormap"], linear=False)
        colors_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)

    out_dir = pathlib.Path(job["out_dir"])
    path = out_dir / f"{job['name']}.{job['f']:04d}.ply"
    _write_ply(path, verts_world, faces, colors_u8, values)
    return {
        "f": job["f"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "n_verts": int(verts_world.shape[0]),
        "n_faces": int(faces.shape[0]),
    }


def export_isosurfaces(
    fields: FieldSeries, timeline: Timeline, spec: dict[str, Any], out_dir: pathlib.Path
) -> dict[str, Any]:
    spec = dict(spec)
    name = str(spec.get("name", "q_criterion"))
    iso_variable = str(spec.get("iso_variable", "q_criterion"))
    color_variable = str(spec.get("color_variable", "speed"))
    upsample = int(spec.get("upsample", 2))
    smooth_sigma = float(spec.get("smooth_sigma", 0.7))
    level_percentile = float(spec.get("level_percentile", 99.0))
    level_fraction = float(spec.get("level_fraction", 0.15))
    max_faces = int(spec.get("max_faces", 400_000))
    min_component_faces = int(spec.get("min_component_faces", 50))
    frame_step = int(spec.get("frame_step", 2))
    workers = int(spec.get("workers", 4))
    colormap = str(spec.get("colormap", "viridis"))

    auto_level, auto_lo, auto_hi = _auto_level_and_color_range(
        fields,
        iso_variable,
        color_variable,
        timeline,
        upsample,
        smooth_sigma,
        level_percentile,
        level_fraction,
    )
    level = float(spec.get("level", auto_level))
    color_range = tuple(float(v) for v in spec.get("color_range", (auto_lo, auto_hi)))

    layer_dir = pathlib.Path(out_dir) / "isosurfaces" / name
    layer_dir.mkdir(parents=True, exist_ok=True)

    frames = _file_frames(timeline, frame_step)
    jobs = [
        dict(
            f=f,
            t=t,
            out_dir=str(layer_dir),
            name=name,
            iso_variable=iso_variable,
            level=level,
            upsample=upsample,
            smooth_sigma=smooth_sigma,
            color_variable=color_variable,
            colormap=colormap,
            color_range=color_range,
            max_faces=max_faces,
            min_component_faces=min_component_faces,
        )
        for f, _vf, t in frames
    ]

    source = None
    encoding = getattr(fields.ds, "encoding", None)
    if encoding:
        source = encoding.get("source")

    global _FIELDS
    if workers <= 1 or len(jobs) <= 1:
        _FIELDS = fields
        results = [_render_isosurface_file(j) for j in jobs]
    elif source:
        ctx = mp.get_context("forkserver")
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_pool,
            initargs=(source,),
        ) as ex:
            results = list(ex.map(_render_isosurface_file, jobs))
    else:
        _FIELDS = fields
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_render_isosurface_file, jobs))
    del results

    layer: dict[str, Any] = {
        "name": name,
        "type": "isosurface",
        "pattern": f"isosurfaces/{name}/{name}.{{frame:04d}}.ply",
        "frame_step": frame_step,
        "n_files": len(frames),
        "iso_variable": iso_variable,
        "level": level,
    }
    layer.update(
        colormaps.layer_color_spec(
            colormap, color_range[0], color_range[1], color_variable
        )
    )
    return layer


__all__ = ["export_isosurfaces", "default_isosurface_specs"]
