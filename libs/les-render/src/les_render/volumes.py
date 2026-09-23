"""OpenVDB volume export for Unreal Sparse Volume Textures and Blender.

Writes one ``.vdb`` file per exported frame under
``<out_dir>/volumes/<name>/<name>.<FFFF>.vdb``, holding a single
``openvdb.FloatGrid`` named ``spec["name"]`` (plus an optional second
``"density"`` grid, see below). ``export_volumes(fields, timeline, spec,
out_dir) -> dict`` returns the manifest ``layer`` entry for
``type: "volume"``.

Index/world mapping
--------------------
The grid transform is a (possibly anisotropic) linear map: voxel size is the
*refined* grid spacing (``fields.grid.refined(upsample).spacing``, metres),
and the translation is set so that index-space voxel ``(0, 0, 0)``'s centre
lands exactly on ``grid.origin`` (world metres) -- OpenVDB index-space cell
centres sit at integer coordinates, so a plain scale + translate transform
does this exactly (verified: ``transform.indexToWorld((0, 0, 0)) ==
grid.origin``). The dataset used in this repo is isotropic, but the
transform is built from a full 3x3 diagonal + translation matrix so
anisotropic spacing works unchanged.

Spec keys (all optional; ``spec = {}`` uses every default)
------------------------------------------------------------
``name`` (str, default ``"speed_glow"``)
    Grid name and file stem. One of the built-in presets below, or any name
    paired with an explicit ``variable``.
``variable`` (str, default from preset)
    One of ``fields.SCALARS``. Required if ``name`` is not a preset.
``transform`` (str, default from preset, else ``"linear"``)
    How the raw scalar is mapped to the value actually stored in the grid
    (see "Transfer functions" below): ``"linear"``, ``"excess"``,
    ``"abs_excess"``, ``"gamma_norm"``.
``gamma`` (float, default ``0.45``)
    Exponent for the ``"gamma_norm"`` transform.
``reference`` / ``vscale`` (float, default: auto)
    Override the auto-computed reference value / normalisation scale used by
    the transforms.
``upsample`` (int, default ``2``)
    Tricubic upsample factor applied before the transform (see
    ``fields.scalar_field``).
``density_range`` ([lo, hi], default: auto)
    Values <= lo become *inactive* voxels (sparse, background); values >= hi
    saturate to full density. Auto value is
    ``(percentile(floor_pct), percentile(ceiling_pct))`` of the *transformed*
    field, computed once from a handful of sample frames spread across the
    timeline and held constant so the volume doesn't visibly re-normalise
    frame to frame. ``floor_pct`` defaults high (70th) rather than the
    usual display-range low percentile (~1st): the transforms below are
    built so most of the domain sits near zero (undisturbed ambient flow /
    low-vorticity fluid), and a low floor leaves that entire near-zero
    majority *active* -- technically non-zero but visually and
    storage-wise noise. A 70th-percentile floor actually culls the boring
    majority of the volume, which is what "keep files sparse" means here.
``density_floor_percentile`` / ``density_ceiling_percentile`` (float,
default ``70.0`` / ``99.5``)
    Percentiles of the transformed, fluid-only sample distribution used for
    the auto ``density_range`` above.
``colormap`` (str, default from preset, else ``"viridis"``)
    Colour LUT name written into the manifest colour block (``range`` =
    ``density_range`` unless ``color_range`` is given).
``color_range`` ([lo, hi], default: ``density_range``)
``emission_strength`` (float, default ``1.5``)
    Passed straight through to the manifest; a render-side multiplier.
``density_scale`` (float, default ``1.0``)
    Passed straight through to the manifest; a render-side multiplier.
``write_normalized_density`` (bool, default ``False``)
    Also write a second grid named ``"density"`` in the same file, holding
    the transformed value normalised into [0, 1] via ``density_range``. Some
    engines/materials prefer to drive density from a pre-normalised channel
    and use the named grid (``speed_glow``/``vorticity``/...) only for
    colour/emission; this grid gives them that for free without re-deriving
    the mapping from the manifest at render time. Defaults to off: Unreal's
    Sparse Volume Texture importer is reported to be picky about a file
    holding more than one grid, so opt in only for renderers (e.g. the
    Blender preview) that you've confirmed handle a second grid fine. When
    on, both grids in a file always share the exact same transform.
``half`` (bool, default ``True``)
    Store voxel values as half-float (``saveFloatAsHalf``); halves file size
    at a precision cost that's invisible for display-only glow volumes (see
    measurements in the final report).
``frame_step`` (int, default ``4``)
    Volumes are the heaviest per-file layer, so they are written more
    sparsely than isosurfaces/slices by default.
``workers`` (int, default ``4``)
    Frame parallelism. This machine is DRAM-bandwidth-bound past ~4-8
    workers (see repo memory), so raising this blindly does not help.

Presets
-------
``"speed_glow"`` -> ``variable="speed"``, ``transform="abs_excess"``:
stores ``|speed - reference|`` where ``reference`` is the auto-computed
median fluid speed (i.e. roughly the ambient/inflow speed away from
buildings). A *raw* speed volume lights up almost the entire inflow region
uniformly (it's all close to the freestream speed) and tells you nothing
about the interesting structure. Absolute deviation from the ambient speed
is near-zero for undisturbed flow and lights up *both* wakes (local speed
deficit behind buildings) and jets (local speed excess through gaps/over
roofs) -- the sparsity this creates is also what keeps the VDB files small.

``"vorticity"`` -> ``variable="vorticity_magnitude"``, ``transform="gamma_norm"``:
stores ``clip(|omega|, 0, vscale) / vscale) ** gamma`` (``vscale`` = the
auto 99.5th-percentile magnitude). Vorticity magnitude is extremely peaked
(a few near-wall/shear-layer cells dominate the raw range), so a raw or
linear mapping makes everything except those cells invisible. Normalising
by a robust upper percentile and gamma-compressing (gamma < 1 boosts low
values) spreads the dynamic range back out so fainter turbulent structures
away from walls stay visible. Because this transform's output is already in
[0, 1], its manifest ``density_range``/colour ``range`` end up close to
[0, 1] too (still auto-computed, still held constant across the animation).

Parallelism
-----------
Frames are independent, so they're farmed out over ``spec["workers"]`` via
``_frames.map_frames`` (shared with ``isosurfaces.py`` / ``slices.py`` -- see
that module's docstring for the forkserver-process-pool / thread-pool /
serial dispatch rules).
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, Optional

import numpy as np

try:
    import pyopenvdb as vdb
except ImportError:  # pragma: no cover - depends on the openvdb build in the env
    import openvdb as vdb

from . import colormaps
from ._frames import file_frames, map_frames, opt
from .fields import FieldSeries, Grid, scalar_field, upsample_mask
from .timeline import Timeline

_PRESETS: dict[str, dict[str, str]] = {
    "speed_glow": {
        "variable": "speed",
        "transform": "abs_excess",
        "colormap": "inferno",
    },
    "vorticity": {
        "variable": "vorticity_magnitude",
        "transform": "gamma_norm",
        "colormap": "viridis",
    },
}

_N_AUTO_SAMPLES = 4

# Per-process global, populated by _set_fields -- either directly (thread
# pool / sequential, same process as the caller) or via _frames.map_frames's
# pool initializer (forkserver worker process).
_FIELDS: Optional[FieldSeries] = None


def default_volume_specs() -> list[dict[str, Any]]:
    """The two built-in presets, with every other key left at its default."""
    return [{"name": "speed_glow"}, {"name": "vorticity"}]


def _openvdb_transform(spacing: np.ndarray, origin: np.ndarray) -> "vdb.Transform":
    sx, sy, sz = (float(v) for v in spacing)
    ox, oy, oz = (float(v) for v in origin)
    if not (np.isclose(sx, sy) and np.isclose(sy, sz)):
        # Supported here (a plain diagonal + translation matrix handles it
        # fine, and Blender reads it correctly), but Unreal's OpenVDB/SVT
        # importer hard-rejects non-uniform voxel size ("OpenVDB importer
        # cannot handle non uniform voxels") -- there is no workaround on
        # the UE side, only resampling to cubic voxels on this side. Every
        # dataset this repo currently renders is isotropic, so this is a
        # heads-up for whoever adds the first anisotropic one, not a live bug.
        warnings.warn(
            f"volume grid spacing {(sx, sy, sz)} is anisotropic; Unreal's SVT "
            "importer rejects non-uniform voxel size. Blender will render this "
            "fine, but resample to cubic voxels before targeting Unreal.",
            stacklevel=2,
        )
    # Row-vector convention (confirmed against openvdb.Transform.indexToWorld):
    # world = index @ diag(sx, sy, sz) + (ox, oy, oz).
    matrix = [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [ox, oy, oz, 1.0],
    ]
    return vdb.createLinearTransform(matrix=matrix)


def _transform_values(
    raw: np.ndarray, transform: str, reference: float, gamma: float, vscale: float
) -> np.ndarray:
    if transform == "linear":
        return raw
    if transform == "excess":
        return np.clip(raw - reference, 0.0, None)
    if transform == "abs_excess":
        return np.abs(raw - reference)
    if transform == "gamma_norm":
        v = np.clip(raw, 0.0, None) / vscale
        np.clip(v, 0.0, 1.0, out=v)
        return np.asarray(v**gamma)
    raise ValueError(f"unknown volume transform {transform!r}")


def _auto_stats(
    fields: FieldSeries,
    variable: str,
    transform: str,
    gamma: float,
    timeline: Timeline,
    upsample: int,
    floor_pct: float,
    ceiling_pct: float,
    n_samples: int = _N_AUTO_SAMPLES,
) -> tuple[float, float, float, float]:
    """Reference / scale / density_range, computed once and held constant."""
    times = np.linspace(timeline.frame_times[0], timeline.frame_times[-1], n_samples)
    solid = upsample_mask(fields.solid, upsample) if upsample > 1 else fields.solid
    fluid = ~solid
    raws = []
    for t in times:
        vals, _ = scalar_field(fields, variable, float(t), upsample)
        raws.append(vals[fluid])
    flat = np.concatenate(raws) if raws else np.zeros(1, dtype=np.float32)

    reference = float(np.median(flat)) if transform in ("excess", "abs_excess") else 0.0
    vscale = 1.0
    if transform == "gamma_norm":
        vscale = max(float(np.percentile(np.clip(flat, 0.0, None), 99.5)), 1e-6)
    transformed = _transform_values(flat, transform, reference, gamma, vscale)

    lo = float(np.percentile(transformed, floor_pct))
    hi = float(np.percentile(transformed, ceiling_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return reference, vscale, max(lo, 0.0), hi


def _set_fields(f: Optional[FieldSeries]) -> None:
    global _FIELDS
    _FIELDS = f


def _render_volume_file(job: dict[str, Any]) -> dict[str, Any]:
    fields = _FIELDS
    assert fields is not None, "volume pool worker used before initialisation"

    upsample = job["upsample"]
    raw, grid = scalar_field(fields, job["variable"], job["t"], upsample)
    solid = upsample_mask(fields.solid, upsample) if upsample > 1 else fields.solid
    values = _transform_values(
        raw, job["transform"], job["reference"], job["gamma"], job["vscale"]
    )
    values = values.astype(np.float32)
    values[solid] = 0.0

    lo, hi = job["density_range"]
    span = max(hi - lo, 1e-9)
    # Sparsity: zero out everything below the density floor, then let
    # copyFromArray's tolerance drop those exact zeros to inactive/background
    # voxels. tolerance is a small fraction of the value range, not `lo`
    # itself, so it works even when lo == 0 (the common case for these
    # non-negative transforms).
    eps = max(span * 1e-4, 1e-6)
    masked = np.where(values >= lo, values, 0.0).astype(np.float32)

    xform = _openvdb_transform(grid.spacing, grid.origin)

    g = vdb.FloatGrid()
    g.name = job["grid_name"]
    g.copyFromArray(masked, tolerance=eps)
    g.transform = xform
    g.saveFloatAsHalf = job["half"]
    grids = [g]

    if job["write_density"]:
        norm = np.clip((values - lo) / span, 0.0, 1.0).astype(np.float32)
        norm[solid] = 0.0
        d = vdb.FloatGrid()
        d.name = "density"
        d.copyFromArray(norm, tolerance=1e-4)
        d.transform = xform
        d.saveFloatAsHalf = job["half"]
        grids.append(d)

    out_dir = pathlib.Path(job["out_dir"])
    path = out_dir / f"{job['grid_name']}.{job['f']:04d}.vdb"
    vdb.write(str(path), grids)
    return {
        "f": job["f"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "active_voxels": int(g.activeVoxelCount()),
    }


def export_volumes(
    fields: FieldSeries, timeline: Timeline, spec: dict[str, Any], out_dir: pathlib.Path
) -> dict[str, Any]:
    spec = dict(spec)
    name = str(opt(spec, "name", "speed_glow"))
    preset = _PRESETS.get(name, {})

    variable = opt(spec, "variable", preset.get("variable"))
    if variable is None:
        raise ValueError(
            f"volume spec {name!r} is not a built-in preset ({sorted(_PRESETS)}); "
            "pass an explicit 'variable'."
        )
    transform = str(opt(spec, "transform", preset.get("transform", "linear")))
    gamma = float(opt(spec, "gamma", 0.45))
    upsample = int(opt(spec, "upsample", 2))
    frame_step = int(opt(spec, "frame_step", 4))
    half = bool(opt(spec, "half", True))
    write_density = bool(opt(spec, "write_normalized_density", False))
    emission_strength = float(opt(spec, "emission_strength", 1.5))
    density_scale = float(opt(spec, "density_scale", 1.0))
    workers = int(opt(spec, "workers", 4))
    colormap = str(opt(spec, "colormap", preset.get("colormap", "viridis")))
    floor_pct = float(opt(spec, "density_floor_percentile", 70.0))
    ceiling_pct = float(opt(spec, "density_ceiling_percentile", 99.5))

    reference, vscale, auto_lo, auto_hi = _auto_stats(
        fields, variable, transform, gamma, timeline, upsample, floor_pct, ceiling_pct
    )
    reference = float(opt(spec, "reference", reference))
    vscale = float(opt(spec, "vscale", vscale))
    density_range = tuple(
        float(v) for v in opt(spec, "density_range", (auto_lo, auto_hi))
    )
    color_range = tuple(float(v) for v in opt(spec, "color_range", density_range))

    refined_grid: Grid = fields.grid.refined(upsample) if upsample > 1 else fields.grid

    layer_dir = pathlib.Path(out_dir) / "volumes" / name
    layer_dir.mkdir(parents=True, exist_ok=True)

    frames = file_frames(timeline, frame_step)
    jobs = [
        dict(
            f=f,
            t=t,
            out_dir=str(layer_dir),
            grid_name=name,
            variable=variable,
            transform=transform,
            gamma=gamma,
            reference=reference,
            vscale=vscale,
            upsample=upsample,
            density_range=density_range,
            half=half,
            write_density=write_density,
        )
        for f, _vf, t in frames
    ]

    results = map_frames(fields, jobs, _render_volume_file, workers, _set_fields)

    layer: dict[str, Any] = {
        "name": name,
        "type": "volume",
        "pattern": f"volumes/{name}/{name}.{{frame:04d}}.vdb",
        "frame_step": frame_step,
        "n_files": len(frames),
        "grid": name,
        "voxel_size": [round(float(v), 6) for v in refined_grid.spacing],
        "origin": [round(float(v), 6) for v in refined_grid.origin],
        "shape": list(refined_grid.shape),
        "density_range": [density_range[0], density_range[1]],
        "emission_strength": emission_strength,
        "density_scale": density_scale,
        "transform": transform,
    }
    layer.update(
        colormaps.layer_color_spec(colormap, color_range[0], color_range[1], variable)
    )
    del results  # per-file stats (bytes, active voxel count); not part of the manifest contract
    return layer


__all__ = ["export_volumes", "default_volume_specs"]
