"""Slice-plane RGBA texture export, with Line Integral Convolution (LIC).

Writes one RGBA ``.png`` per exported frame under
``<out_dir>/slices/<name>/<name>.<FFFF>.png``: a colormapped scalar field on
an axis-aligned plane (default: ``speed`` at ``z = 2`` m, "pedestrian
level"), modulated by an LIC texture of the in-plane velocity so the flow
direction/shear is visible even though the colour alone only encodes
magnitude. ``export_slices(fields, timeline, spec, out_dir) -> dict``
returns the manifest ``layer`` entry for ``type: "slice"``.

Plane / pixel layout
---------------------
``axis`` picks the plane normal; the two remaining axes of ``(x, y, z)``, in
that order, are the in-plane ``(u, v)`` axes (contract: for ``axis="z"``,
``u=x, v=y``; likewise ``axis="y"`` -> ``u=x, v=z``, ``axis="x"`` -> ``u=y,
v=z``). ``extent = [[u0, v0], [u1, v1]]`` defaults to the full domain in
those two axes. Per the bundle contract, PNG row 0 is the *max-v* edge
(standard image orientation, top of the image = max v): column ``j`` maps to
``u0 + (j + 0.5)/W * (u1-u0)`` and row ``i`` maps to ``v1 - (i +
0.5)/H * (v1-v0)`` (row increases -> v decreases).

Colour: the scalar field (upsampled tricubically, see ``fields.py``) is
sampled at every pixel centre with ``fields.trilinear`` (so the texture
isn't limited to the raw grid's resolution) and mapped through the colour
block's LUT. ``variable="w"`` (or any of ``u``/``v``/``pressure``) defaults
to the diverging ``"RdBu_r"`` colormap centred at 0 (auto range
``[-m, m]``, ``m`` = a robust percentile of ``|variable|``); everything else
defaults to the sequential ``"inferno"`` colormap (0..max, better perceptual
"how much" reading for a strictly-positive quantity like speed than a
diverging map would give).

Alpha: 0 inside solids (per contract), from ``fields.is_solid`` at each
pixel's plane position -- i.e. the actual building footprint at that
height/position, not a coarse rasterisation of the geometry mesh.

LIC (in-plane flow texture)
----------------------------
A fixed (seeded) white-noise texture at the output resolution is advected
along the in-plane velocity direction, sampled with
``fields.sample_velocity`` (trilinear in space, linear in time, so this
works at plane positions between stored snapshots) and converted to a
texel-space unit direction each step (texel pitch is uniform since
``resolution`` is derived from a single ``px_per_metre``). For every pixel,
the streamline is integrated ``lic_length`` texels forward *and* backward,
recomputing direction at each step (a true curved-streamline LIC, not a
straight-line approximation), sampling the noise texture bilinearly along
the way and accumulating a weighted average -- weight is a Hann window by
default (``lic_kernel="hann"``; ``"box"`` -- uniform weight -- is also
available and cheaper/noisier). The whole integration is vectorised over
every pixel at once (no per-pixel Python loop), which is what keeps this
inside the ~1 s/frame budget at 1024x512 -- see the runtime measurement in
the final report. The noise texture is generated from a fixed
``lic_noise_seed`` and is otherwise identical every frame -- an
independently-reseeded texture per frame would flicker instead of "flowing"
coherently, which is the entire point of using LIC over, say, overlaid
streamline glyphs. The resulting grayscale LIC pattern is contrast-stretched
(its raw dynamic range shrinks with averaging) and blended multiplicatively
into the colour image's luminance, weighted by ``lic_strength`` (0 = colour
only, 1 = LIC fully modulates brightness).

``animate_noise`` (nice-to-have, default ``False``) advects the *noise
texture itself* over time by an integer-texel roll proportional to ``t``
along the mean in-plane flow direction (computed once, held fixed) -- a
cheap way to make the pattern visibly flow across frames instead of only
showing static streamline shape. It is a texel-grid wraparound roll, not a
true phase-continuous advection, so at high ``playback_speed`` the roll can
jump by more than one texel per video frame; documented as an approximation
deliberately traded for simplicity/speed.

Spec keys (all optional)
-------------------------
``name`` (str, default ``"pedestrian_speed"``).
``axis`` (``"x"|"y"|"z"``, default ``"z"``).
``position`` (float, m, default ``2.0``).
``variable`` (str, default ``"speed"``) -- one of ``fields.SCALARS``.
``colormap`` (str, default: ``"RdBu_r"`` if ``variable`` in
``{"u","v","w","pressure"}`` else ``"inferno"``).
``range`` ([lo, hi], default: auto -- symmetric ``[-m, m]`` for the
diverging colormaps, else a robust ``[0, high-percentile]`` range; computed
once from a handful of sample frames, held constant).
``upsample`` (int, default ``2``) -- 3D field refinement before sampling
onto the texture (independent of texture pixel resolution).
``extent`` (default: full domain in the plane's two in-plane axes).
``px_per_metre`` (float, default ``8.0``) and ``max_resolution`` (int,
default ``2048``) -- ``resolution = extent_size * px_per_metre``, uniformly
downscaled if its long side would exceed ``max_resolution``.
``resolution`` ([w, h], default: derived from ``px_per_metre``) -- explicit
override.
``lic`` (bool, default ``True``).
``lic_length`` (int, texels, default ``25``).
``lic_kernel`` (``"hann"|"box"``, default ``"hann"``).
``lic_noise_seed`` (int, default ``0``).
``lic_strength`` (float in [0, 1], default ``0.6``).
``animate_noise`` (bool, default ``False``).
``animate_speed`` (float, texels per sim second, default ``2.0``) -- only
used when ``animate_noise`` is on.
``frame_step`` (int, default ``1``) -- slices are the cheapest layer (target
~1 s/frame), so by default every video frame gets its own texture.
``workers`` (int, default ``4``).
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import pathlib
from typing import Any, Optional

import numba
import numpy as np
from PIL import Image

from . import colormaps
from .fields import FieldSeries, robust_range, scalar_field, trilinear
from .timeline import Timeline

_DIVERGING_VARS = {"u", "v", "w", "pressure"}
_IN_PLANE_AXES = {"z": ("x", "y"), "y": ("x", "z"), "x": ("y", "z")}
_N_AUTO_SAMPLES = 4

_FIELDS: Optional[FieldSeries] = None


def default_slice_specs() -> list[dict[str, Any]]:
    return [{"name": "pedestrian_speed"}]


def _file_frames(timeline: Timeline, frame_step: int) -> list[tuple[int, int, float]]:
    times = timeline.frame_times
    video_frames = list(range(0, timeline.n_frames, frame_step))
    return [(f, vf, float(times[vf])) for f, vf in enumerate(video_frames)]


def _domain_extent(
    fields: FieldSeries, axis: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    u_axis, v_axis = _IN_PLANE_AXES[axis]
    lower = dict(zip(("x", "y", "z"), fields.grid.lower))
    upper = dict(zip(("x", "y", "z"), fields.grid.upper))
    return (lower[u_axis], lower[v_axis]), (upper[u_axis], upper[v_axis])


def _resolution(
    extent: tuple, px_per_metre: float, max_resolution: int
) -> tuple[int, int]:
    (u0, v0), (u1, v1) = extent
    w = max(int(round((u1 - u0) * px_per_metre)), 2)
    h = max(int(round((v1 - v0) * px_per_metre)), 2)
    long_side = max(w, h)
    if long_side > max_resolution:
        scale = max_resolution / long_side
        w = max(int(round(w * scale)), 2)
        h = max(int(round(h * scale)), 2)
    return w, h


def _plane_points(
    axis: str, position: float, extent: tuple, resolution: tuple[int, int]
) -> np.ndarray:
    """World (H, W, 3) points at pixel centres; row 0 = max-v edge."""
    (u0, v0), (u1, v1) = extent
    w, h = resolution
    u = u0 + (np.arange(w) + 0.5) / w * (u1 - u0)
    v = v1 - (np.arange(h) + 0.5) / h * (v1 - v0)  # row 0 -> v close to v1 (max)
    uu, vv = np.meshgrid(u, v)  # (h, w) each

    u_axis, v_axis = _IN_PLANE_AXES[axis]
    coords = {u_axis: uu, v_axis: vv, axis: np.full_like(uu, position)}
    return np.stack([coords["x"], coords["y"], coords["z"]], axis=-1).astype(np.float64)


def _auto_range(
    fields: FieldSeries,
    variable: str,
    diverging: bool,
    timeline: Timeline,
    axis: str,
    position: float,
    n_samples: int = _N_AUTO_SAMPLES,
) -> tuple[float, float]:
    """Colour range from the values *on the slice plane* (fluid only).

    A pedestrian-level plane sees far slower wind than the domain as a whole,
    so a domain-wide range would squash it into the dark end of the colormap.
    """
    grid = fields.grid
    u_axis, v_axis = _IN_PLANE_AXES[axis]
    coords = {"x": grid.x, "y": grid.y, "z": grid.z}
    uu, vv = np.meshgrid(coords[u_axis], coords[v_axis])
    plane = {u_axis: uu.ravel(), v_axis: vv.ravel(), axis: np.full(uu.size, position)}
    points = np.stack([plane["x"], plane["y"], plane["z"]], axis=-1)
    fluid = ~fields.is_solid(points)
    idx = grid.to_index(points[fluid])
    times = np.linspace(timeline.frame_times[0], timeline.frame_times[-1], n_samples)
    samples = []
    for t in times:
        vals, _ = scalar_field(fields, variable, float(t), upsample=1)
        samples.append(trilinear(vals, idx))
    flat = np.concatenate(samples) if samples else np.zeros(1, dtype=np.float32)
    if diverging:
        m = float(np.percentile(np.abs(flat), 99.0))
        m = max(m, 1e-6)
        return -m, m
    lo, hi = robust_range(flat, lo=0.0, hi=99.0)
    return max(0.0, lo), hi


def _lic_kernel_weights(length: int, kernel: str) -> np.ndarray:
    k = np.arange(-length, length + 1, dtype=np.float64)
    if kernel == "box":
        return np.ones_like(k)
    if kernel == "hann":
        return 0.5 * (1.0 + np.cos(np.pi * k / max(length, 1)))
    raise ValueError(f"unknown lic_kernel {kernel!r}")


@numba.njit(cache=True, fastmath=True, inline="always")  # type: ignore[misc]
def _bilinear_nb(
    f: np.ndarray, r: float, c: float
) -> float:  # pragma: no cover - jitted
    h, w = f.shape
    r = min(max(r, 0.0), h - 1.0)
    c = min(max(c, 0.0), w - 1.0)
    r0 = max(min(int(r), h - 2), 0)
    c0 = max(min(int(c), w - 2), 0)
    fr = r - r0
    fc = c - c0
    return float(
        f[r0, c0] * (1 - fr) * (1 - fc)
        + f[r0, c0 + 1] * (1 - fr) * fc
        + f[r0 + 1, c0] * fr * (1 - fc)
        + f[r0 + 1, c0 + 1] * fr * fc
    )


@numba.njit(parallel=True, cache=True, fastmath=True)  # type: ignore[misc]
def _lic_kernel(
    dir_row: np.ndarray,
    dir_col: np.ndarray,
    noise: np.ndarray,
    weights: np.ndarray,
    length: int,
) -> np.ndarray:  # pragma: no cover - jitted
    h, w = noise.shape
    out = np.empty((h, w), np.float32)
    for i in numba.prange(h):
        for j in range(w):
            acc = noise[i, j] * weights[length]
            wsum = weights[length]
            for sign in (1.0, -1.0):
                r = float(i)
                c = float(j)
                for step in range(1, length + 1):
                    dr = _bilinear_nb(dir_row, r, c)
                    dc = _bilinear_nb(dir_col, r, c)
                    r = min(max(r + sign * dr, 0.0), h - 1.0)
                    c = min(max(c + sign * dc, 0.0), w - 1.0)
                    wt = weights[length + int(sign) * step]
                    acc += wt * _bilinear_nb(noise, r, c)
                    wsum += wt
            out[i, j] = acc / max(wsum, 1e-9)
    return out


def _compute_lic(
    vu: np.ndarray, vv: np.ndarray, noise: np.ndarray, length: int, kernel: str
) -> np.ndarray:
    """Curved-streamline LIC. vu/vv/noise are (H, W); vu is the world-space
    in-plane velocity component along +u (maps to +col), vv along +v (maps to
    -row, since row increases as v decreases).

    Each texel integrates its own streamline ``length`` steps forward and
    backward (unit texel steps, direction re-sampled every step). This is a
    per-texel loop, so it runs as a numba kernel parallel over rows (~70x the
    vectorised-numpy version, which was bound by fancy-indexing traffic).
    """
    vu = vu.astype(np.float32, copy=False)
    vv = vv.astype(np.float32, copy=False)
    speed = np.sqrt(vu**2 + vv**2) + np.float32(1e-9)
    dir_col = np.ascontiguousarray(vu / speed, dtype=np.float32)
    dir_row = np.ascontiguousarray(-vv / speed, dtype=np.float32)
    weights = _lic_kernel_weights(length, kernel).astype(np.float32)
    lic = _lic_kernel(
        dir_row,
        dir_col,
        np.ascontiguousarray(noise, dtype=np.float32),
        weights,
        int(length),
    )
    lo, hi = np.percentile(lic, [1.0, 99.0])
    if hi <= lo:
        return np.full_like(lic, 0.5)
    return np.asarray(np.clip((lic - lo) / (hi - lo), 0.0, 1.0))


def _init_pool(source: str, numba_threads: int) -> None:
    global _FIELDS
    from .fields import open_fields

    # Each worker runs the parallel LIC kernel; split the cores between them.
    numba.set_num_threads(max(1, min(numba_threads, numba.config.NUMBA_NUM_THREADS)))

    _FIELDS = open_fields(source)


def _render_slice_file(job: dict[str, Any]) -> dict[str, Any]:
    fields = _FIELDS
    assert fields is not None, "slice pool worker used before initialisation"

    axis, position, extent, resolution = (
        job["axis"],
        job["position"],
        job["extent"],
        job["resolution"],
    )
    points_hw3 = _plane_points(axis, position, extent, resolution)
    h, w = points_hw3.shape[:2]
    points_flat = points_hw3.reshape(-1, 3)

    field3d, grid = scalar_field(fields, job["variable"], job["t"], job["upsample"])
    idx = grid.to_index(points_flat)
    values = trilinear(field3d, idx).astype(np.float32).reshape(h, w)

    lo, hi = job["value_range"]
    rgb = colormaps.apply(
        values, lo, hi, job["colormap"], linear=False
    )  # (h, w, 3) sRGB in [0,1]

    solid = fields.is_solid(points_flat).reshape(h, w)
    alpha = np.where(solid, 0.0, 1.0)

    if job["lic"]:
        vel = fields.sample_velocity(points_flat, job["t"]).reshape(h, w, 3)
        u_axis, v_axis = _IN_PLANE_AXES[axis]
        comp = {"x": 0, "y": 1, "z": 2}
        vu, vv = vel[..., comp[u_axis]], vel[..., comp[v_axis]]

        rng = np.random.default_rng(job["lic_noise_seed"])
        noise = rng.random((h, w))
        if job["animate_noise"]:
            shift = int(round(job["t"] * job["animate_speed"])) % w
            noise = np.roll(noise, shift, axis=1)

        lic = _compute_lic(vu, vv, noise, job["lic_length"], job["lic_kernel"])
        strength = job["lic_strength"]
        modulation = (1.0 - strength) + strength * lic
        rgb = np.clip(rgb * modulation[..., None], 0.0, 1.0)

    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)

    out_dir = pathlib.Path(job["out_dir"])
    path = out_dir / f"{job['name']}.{job['f']:04d}.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    return {"f": job["f"], "path": str(path), "bytes": path.stat().st_size}


def export_slices(
    fields: FieldSeries, timeline: Timeline, spec: dict[str, Any], out_dir: pathlib.Path
) -> dict[str, Any]:
    spec = dict(spec)
    name = str(spec.get("name", "pedestrian_speed"))
    axis = str(spec.get("axis", "z"))
    if axis not in _IN_PLANE_AXES:
        raise ValueError(f"axis must be one of {sorted(_IN_PLANE_AXES)}, got {axis!r}")
    position = float(spec.get("position", 2.0))
    variable = str(spec.get("variable", "speed"))
    diverging = variable in _DIVERGING_VARS
    colormap = str(spec.get("colormap", "RdBu_r" if diverging else "inferno"))
    upsample = int(spec.get("upsample", 2))
    px_per_metre = float(spec.get("px_per_metre", 8.0))
    max_resolution = int(spec.get("max_resolution", 2048))
    frame_step = int(spec.get("frame_step", 1))
    workers = int(spec.get("workers", 4))

    lo_domain, hi_domain = _domain_extent(fields, axis)
    extent = tuple(tuple(p) for p in spec.get("extent", (lo_domain, hi_domain)))
    resolution = tuple(
        spec.get("resolution") or _resolution(extent, px_per_metre, max_resolution)
    )

    auto_lo, auto_hi = _auto_range(
        fields, variable, diverging, timeline, axis, position
    )
    value_range = tuple(float(v) for v in spec.get("range", (auto_lo, auto_hi)))

    lic = bool(spec.get("lic", True))
    lic_length = int(spec.get("lic_length", 25))
    lic_kernel = str(spec.get("lic_kernel", "hann"))
    lic_noise_seed = int(spec.get("lic_noise_seed", 0))
    lic_strength = float(spec.get("lic_strength", 0.6))
    animate_noise = bool(spec.get("animate_noise", False))
    animate_speed = float(spec.get("animate_speed", 2.0))

    layer_dir = pathlib.Path(out_dir) / "slices" / name
    layer_dir.mkdir(parents=True, exist_ok=True)

    frames = _file_frames(timeline, frame_step)
    jobs = [
        dict(
            f=f,
            t=t,
            out_dir=str(layer_dir),
            name=name,
            axis=axis,
            position=position,
            extent=extent,
            resolution=resolution,
            variable=variable,
            colormap=colormap,
            value_range=value_range,
            upsample=upsample,
            lic=lic,
            lic_length=lic_length,
            lic_kernel=lic_kernel,
            lic_noise_seed=lic_noise_seed,
            lic_strength=lic_strength,
            animate_noise=animate_noise,
            animate_speed=animate_speed,
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
        results = [_render_slice_file(j) for j in jobs]
    elif source:
        ctx = mp.get_context("forkserver")
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_pool,
            initargs=(source, max(1, (os.cpu_count() or 4) // workers)),
        ) as ex:
            results = list(ex.map(_render_slice_file, jobs))
    else:
        _FIELDS = fields
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_render_slice_file, jobs))
    del results

    layer: dict[str, Any] = {
        "name": name,
        "type": "slice",
        "pattern": f"slices/{name}/{name}.{{frame:04d}}.png",
        "frame_step": frame_step,
        "n_files": len(frames),
        "axis": axis,
        "position": position,
        "extent": [list(extent[0]), list(extent[1])],
        "resolution": list(resolution),
    }
    layer.update(
        colormaps.layer_color_spec(colormap, value_range[0], value_range[1], variable)
    )
    return layer


__all__ = ["export_slices", "default_slice_specs"]
