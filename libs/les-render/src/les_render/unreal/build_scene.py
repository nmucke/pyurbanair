# mypy: ignore-errors
# (Unreal Editor script: untyped by design, runs against the dynamic `unreal` module.)
"""Build an Unreal Engine 5 scene from an les-render bundle.

Runs INSIDE the Unreal Editor (Python Editor Script Plugin). It imports the
bundle's assets under ``/Game/LES/<case>/``, builds a level (geometry, lights,
post-process, heterogeneous volumes, grooms / geometry caches, slice planes),
a Level Sequence (one CineCamera per shot, camera cuts, per-shot layer
visibility, volume/cache/media playback) and a Movie Render Queue config, and
saves everything. See ``<bundle>/unreal/README.md`` for the full walkthrough.

Usage
-----
Editor GUI:  Tools > Execute Python Script... > <bundle>/unreal/build_scene.py
             (the bundle path is baked in by ``prepare_unreal``; override with
             the ``LES_BUNDLE`` environment variable).
Headless (full editor, recommended; waits for the startup map to load):
    UnrealEditor-Cmd <Project.uproject> -ExecutePythonScript="<bundle>/unreal/build_scene.py --bundle <bundle> --quit" -unattended -nosplash
Headless (commandlet; faster, but no viewport and some editor subsystems may
be missing):
    UnrealEditor-Cmd <Project.uproject> -run=pythonscript -script="<bundle>/unreal/build_scene.py --bundle <bundle>"

Arguments after the script path land in ``sys.argv`` (Unreal runs the file in
"ExecuteFile" mode). ``--bundle`` wins over ``LES_BUNDLE`` which wins over the
path baked in by ``prepare_unreal``.

Coordinate frame
----------------
Bundle files are in the simulation frame (metres, right-handed, z-up).
Unreal is centimetres, left-handed, z-up:
``(x, y, z)_m -> (100 x, -100 y, 100 z)_cm``. glTF geometry is converted by
Unreal's glTF importer; Alembic and VDB are converted by the knobs below.

Design
------
Every stage is wrapped so a failure is logged with ``unreal.log_warning``
(including the manual fallback) and the remaining stages still run. All
maths (manifest parsing, frame/time mapping, camera baking, transforms) is
in pure functions at the top of this file that never touch ``unreal``, so
they are unit tested outside the editor (``libs/les-render/tests``).
Only ``unreal``, ``json``, ``math``, ``os``, ``pathlib`` and ``sys`` are
imported.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

try:  # inside the editor
    import unreal  # type: ignore[import-not-found]
except ImportError:  # outside the editor: the pure helpers below stay importable
    unreal = None  # type: ignore[assignment]

# Replaced with the absolute bundle path by ``prepare_unreal`` when this file is
# copied into ``<bundle>/unreal/``.
DEFAULT_BUNDLE = None

# =============================================================================
# Knobs. Edit here, or set the environment variable shown next to each one.
# =============================================================================

CONTENT_ROOT = "/Game/LES"
UE_UNITS_PER_METRE = 100.0

# Heterogeneous-volume placement. UE >= 5.4 applies the VDB grid transform
# (scale/rotation/translation) at import, with 1 VDB unit = 1 UE unit (cm) and
# no axis conversion, so "world" = actor at the origin with scale
# (100, -100, 100). UE 5.3 keeps the volume in index space ("index" = scale by
# voxel size and translate to the manifest origin). "auto" measures the actor
# bounds after spawning and chooses. If the volume ends up mirrored, flip the
# Y sign in VDB_AXIS_SIGN (one-line fix).
VDB_PLACEMENT = os.environ.get("LES_VDB_PLACEMENT", "auto")  # auto | world | index
VDB_AXIS_SIGN = (1.0, -1.0, 1.0)

# Alembic written by Blender's exporter is Y-up metres: (x, y, z)_blender ->
# (x, z, -y)_abc. Rotation (90, 0, 0) + scale (1, -1, 1) is Unreal's "Maya"
# preset (Y-up right-handed -> Z-up left-handed); x100 converts metres to cm.
ABC_ROTATION = (90.0, 0.0, 0.0)
ABC_SCALE = (100.0, -100.0, 100.0)

# Isosurface vertex colours ("Cd") arrive sRGB-encoded from Blender; the
# material raises them to this power (2.2 ~ sRGB -> linear). Set 1.0 if the
# Alembic importer turns out to decode sRGB itself (colours look too dark).
VERTEX_COLOR_GAMMA = 2.2

# Particles: "auto" and "groom" both import alembic/<layer>.abc as a Groom
# asset + Groom Cache; "none" skips particle layers entirely. (There is no
# Geometry Cache fallback: nothing in the bundle pipeline writes the
# alembic/<layer>_mesh.abc a fallback would need. If groom import fails, the
# particles stage logs the manual re-import steps instead.)
PARTICLE_MODE = os.environ.get("LES_PARTICLE_MODE", "auto")

# After spawning glTF geometry, compare its bounds with the manifest and fix
# a missing x100 or a missing Y mirror automatically.
GEOMETRY_AUTOFIX = True

# Camera: key every N video frames (1 = every frame; linear between keys).
CAMERA_KEY_STEP = int(os.environ.get("LES_CAMERA_KEY_STEP", "1"))
SENSOR_WIDTH_MM = 36.0  # matches the Blender preview (36 mm horizontal fit)

# Slices: in-plane (u, v) axes for each slice normal axis. Must match the
# slice exporter; PNG row 0 is the max-v edge.
SLICE_UV_AXES = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}

# Renderer used by the Movie Render Queue config: "path_tracer" (documented
# cinematic path for heterogeneous volumes) or "lumen" (deferred).
RENDERER = os.environ.get("LES_RENDERER", "path_tracer")
OUTPUT_FORMAT = os.environ.get("LES_OUTPUT_FORMAT", "png")  # png | exr

MRQ = {
    "path_tracer": {
        "spatial_samples": 1,
        "temporal_samples": 64,
        "engine_warm_up": 16,
        "render_warm_up": 0,
    },
    "lumen": {
        "spatial_samples": 1,
        "temporal_samples": 8,
        "engine_warm_up": 32,
        "render_warm_up": 32,
    },
}
MRQ_CVARS = {
    "path_tracer": {
        "r.PathTracing.HeterogeneousVolumes": 1,
        "r.HeterogeneousVolumes.MaxTraceDistance": 10000000,
        "r.HeterogeneousVolumes.OrthoGrid.MaxBottomLevelMemoryInMegabytes": 512,
        "r.HeterogeneousVolumes.OrthoGridShadingRate": 1,
        "r.HairStrands.RaytracingProceduralSplits": 2,
    },
    "lumen": {
        "r.HeterogeneousVolumes.MaxTraceDistance": 10000000,
        "r.HeterogeneousVolumes.OrthoGrid.MaxBottomLevelMemoryInMegabytes": 512,
        "r.HeterogeneousVolumes.OrthoGridShadingRate": 1,
        "r.HeterogeneousVolumes.IndirectLighting": 1,
    },
}

# Looks. Exposure is manual (EV100 = exposure_bias with physical camera
# exposure off) so the render never pumps; emissive_gain multiplies every
# emissive layer; volume_* are the per-cm integration scales of the
# heterogeneous-volume material (tune these first if the volume is too
# faint / blown out).
LOOKS = {
    "dark": {
        "sun_lux": 0.0,
        "sky_intensity": 0.08,
        "sky_atmosphere": False,
        "fog": False,
        "exposure_bias": 0.0,
        "bloom": 1.2,
        "bloom_threshold": -1.0,
        "ao": 0.6,
        "building_color": (0.025, 0.027, 0.03),
        "building_roughness": 0.85,
        "rim_glow": 0.04,
        "ground_color": (0.01, 0.01, 0.012),
        "emissive_gain": 4.0,
        "volume_emission_per_cm": 0.002,
        "volume_extinction_per_cm": 0.0001,
        "vignette": 0.35,
    },
    "daylight": {
        "sun_lux": 10.0,
        "sky_intensity": 1.0,
        "sky_atmosphere": True,
        "fog": True,
        "exposure_bias": 3.0,
        "bloom": 0.3,
        "bloom_threshold": 1.0,
        "ao": 0.8,
        "building_color": (0.8, 0.8, 0.78),
        "building_roughness": 0.75,
        "rim_glow": 0.0,
        "ground_color": (0.35, 0.35, 0.35),
        "emissive_gain": 10.0,
        "volume_emission_per_cm": 0.001,
        "volume_extinction_per_cm": 0.0003,
        "vignette": 0.2,
    },
}

LOG_PREFIX = "[LES]"
ACTOR_TAG = "LES"

# =============================================================================
# Pure helpers (no `unreal`): unit tested outside the editor.
# =============================================================================


def sim_to_ue(p, scale=UE_UNITS_PER_METRE):
    """Simulation point (m, right-handed, z-up) -> Unreal location (cm, left-handed)."""
    return (scale * float(p[0]), -scale * float(p[1]), scale * float(p[2]))


def sim_dir_to_ue(v):
    """Direction vector: mirror Y only (no unit scale)."""
    return (float(v[0]), -float(v[1]), float(v[2]))


def unwrap_degrees(angle, reference):
    """Shift ``angle`` by multiples of 360 so it is within 180 of ``reference``."""
    if reference is None:
        return angle
    return angle + 360.0 * round((reference - angle) / 360.0)


def look_at_rotator(location, target, prev_yaw=None):
    """(pitch, yaw, roll) in degrees of an Unreal camera at sim ``location``
    looking at sim ``target``.

    Unreal cameras look down local +X; yaw rotates about +Z from +X toward +Y
    (in Unreal's left-handed frame), pitch is positive looking up, roll is
    always 0 (the bundle contract is world +z up, no roll). A camera looking
    straight up/down keeps ``prev_yaw`` (or 0). Yaw is unwrapped against
    ``prev_yaw`` so keyed rotations never spin through 360.
    """
    a = sim_to_ue(location, 1.0)
    b = sim_to_ue(target, 1.0)
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    horiz = math.hypot(dx, dy)
    if horiz < 1e-9 and abs(dz) < 1e-9:
        return (0.0, prev_yaw if prev_yaw is not None else 0.0, 0.0)
    pitch = math.degrees(math.atan2(dz, horiz))
    if horiz < 1e-9 * max(1.0, abs(dz)):
        yaw = prev_yaw if prev_yaw is not None else 0.0
    else:
        yaw = unwrap_degrees(math.degrees(math.atan2(dy, dx)), prev_yaw)
    return (pitch, yaw, 0.0)


def rotator_forward(pitch, yaw):
    """Unit forward vector (Unreal frame) of a rotator; inverse of look_at_rotator."""
    p, y = math.radians(pitch), math.radians(yaw)
    return (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))


def fps_to_fraction(fps):
    """Frame rate as an integer (numerator, denominator); NTSC rates -> /1001."""
    fps = float(fps)
    if abs(fps - round(fps)) < 1e-6:
        return (int(round(fps)), 1)
    ntsc = fps * 1001.0 / 1000.0
    if abs(ntsc - round(ntsc)) < 1e-3:
        return (int(round(ntsc)) * 1000, 1001)
    return (int(round(fps * 1000)), 1000)


def file_index(frame, frame_step, n_files):
    """Bundle file shown on video ``frame`` (``floor(frame / frame_step)``, clamped)."""
    idx = int(frame) // max(int(frame_step), 1)
    return max(0, min(idx, int(n_files) - 1))


def frame_seconds(frame, fps):
    return float(frame) / float(fps)


def _smoothstep(u):
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _catmull_rom(p0, p1, p2, p3, u):
    u2, u3 = u * u, u * u * u
    return tuple(
        0.5
        * (
            2.0 * b
            + (-a + c) * u
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * u2
            + (-a + 3.0 * b - 3.0 * c + d) * u3
        )
        for a, b, c, d in zip(p0, p1, p2, p3)
    )


def sample_camera(shots, frame):
    """Camera state at video ``frame`` (sim frame).

    Pure-Python port of ``les_render.cameras.sample_camera`` (which the
    Blender preview uses): shot lookup with hard cuts; smoothstep eased
    *once per shot* -- ``u = smoothstep((frame - start) / (end - start))``
    mapped back into key-frame space as ``f' = start + u * (end - start)`` --
    then a uniform Catmull-Rom through keys[i-1..i+2] (end keys duplicated,
    located at ``f'``) for location and target, linear lerp for focal length
    and f-stop on the same key-segment fraction, hold before the first /
    after the last key.
    Returns ``{"location", "target", "focal_length_mm", "fstop", "shot"}``.
    """
    if not shots:
        raise ValueError("no shots to sample")
    frame = int(min(max(int(frame), shots[0]["start"]), shots[-1]["end"]))
    shot = next((s for s in shots if s["start"] <= frame <= s["end"]), shots[-1])
    keys = sorted(shot["keys"], key=lambda k: k["frame"])

    def pack(loc, tgt, focal, fstop):
        return {
            "location": tuple(float(v) for v in loc),
            "target": tuple(float(v) for v in tgt),
            "focal_length_mm": float(focal),
            "fstop": float(fstop),
            "shot": shot["name"],
        }

    def hold(k):
        return pack(
            k["location"], k["target"], k["focal_length_mm"], k.get("fstop", 8.0)
        )

    kf = [k["frame"] for k in keys]
    if len(keys) == 1:
        return hold(keys[0])

    start, end = shot["start"], shot["end"]
    u_global = 0.0 if end <= start else (frame - start) / (end - start)
    f_prime = start + _smoothstep(u_global) * (end - start)

    if f_prime <= kf[0]:
        return hold(keys[0])
    if f_prime >= kf[-1]:
        return hold(keys[-1])
    i = max(j for j in range(len(kf)) if kf[j] <= f_prime)
    i = min(max(i, 0), len(keys) - 2)
    t0, t1 = kf[i], kf[i + 1]
    u = 0.0 if t1 == t0 else (f_prime - t0) / (t1 - t0)
    im1, ip2 = max(i - 1, 0), min(i + 2, len(keys) - 1)
    loc = _catmull_rom(*(keys[j]["location"] for j in (im1, i, i + 1, ip2)), u)
    tgt = _catmull_rom(*(keys[j]["target"] for j in (im1, i, i + 1, ip2)), u)
    a, b = keys[i], keys[i + 1]
    focal = (1.0 - u) * a["focal_length_mm"] + u * b["focal_length_mm"]
    fstop = (1.0 - u) * a.get("fstop", 8.0) + u * b.get("fstop", 8.0)
    return pack(loc, tgt, focal, fstop)


def key_frames(start, end, step):
    """Frames to key in ``[start, end]``: every ``step`` frames plus ``end``."""
    step = max(int(step), 1)
    frames = list(range(int(start), int(end) + 1, step))
    if not frames or frames[-1] != int(end):
        frames.append(int(end))
    return frames


def bake_camera(shots, step=1, baked=None):
    """Per-shot Unreal camera keys.

    Returns ``{shot_name: [{"frame", "location_cm", "rotation": (pitch, yaw,
    roll), "focal_length_mm", "fstop", "focus_distance_cm"}]}``. ``baked`` is
    an optional ``{frame: sample}`` map (``camera_bake.json`` written by
    ``prepare_unreal`` with the reference ``les_render.cameras``); frames it
    lacks fall back to :func:`sample_camera`. Yaw is unwrapped within a shot.
    """
    out = {}
    for shot in shots:
        keys = []
        prev_yaw = None
        for f in key_frames(shot["start"], shot["end"], step):
            s = (baked or {}).get(f) or sample_camera(shots, f)
            pitch, yaw, roll = look_at_rotator(s["location"], s["target"], prev_yaw)
            prev_yaw = yaw
            dist_m = math.dist(s["location"], s["target"])
            keys.append(
                {
                    "frame": f,
                    "location_cm": sim_to_ue(s["location"]),
                    "rotation": (pitch, yaw, roll),
                    "focal_length_mm": float(s["focal_length_mm"]),
                    "fstop": float(s.get("fstop", 8.0)),
                    "focus_distance_cm": dist_m * UE_UNITS_PER_METRE,
                }
            )
        out[shot["name"]] = keys
    return out


def layer_visibility_keys(manifest):
    """``{layer_name: [(frame, visible), ...]}`` from the per-shot ``layers``
    lists (a shot without ``layers`` shows every layer). Consecutive duplicate
    states are collapsed; the first key is always at the first shot's start."""
    shots = sorted(manifest.get("shots") or [], key=lambda s: s["start"])
    out = {}
    for layer in manifest.get("layers") or []:
        name = layer["name"]
        keys = []
        for shot in shots:
            vis = (
                "layers" not in shot or shot["layers"] is None or name in shot["layers"]
            )
            if not keys or keys[-1][1] != vis:
                keys.append((int(shot["start"]), bool(vis)))
        out[name] = keys or [(0, True)]
    return out


def volume_frame_keys(layer, n_frames):
    """``[(video_frame, file_index)]`` at every video frame where the shown VDB
    file changes (key with constant interpolation)."""
    step, n_files = int(layer.get("frame_step", 1)), int(layer["n_files"])
    keys, last = [], None
    for f in range(int(n_frames)):
        idx = file_index(f, step, n_files)
        if idx != last:
            keys.append((f, idx))
            last = idx
    return keys


def _vec3(v):
    if isinstance(v, (int, float)):
        return (float(v),) * 3
    return tuple(float(x) for x in v)


def volume_world_box(layer):
    """Sim-frame (metres) box of all voxels (cell-centred: voxel (0,0,0)'s
    centre sits at ``origin``)."""
    vs, org, shape = (
        _vec3(layer["voxel_size"]),
        _vec3(layer["origin"]),
        _vec3(layer["shape"]),
    )
    lo = tuple(o - 0.5 * d for o, d in zip(org, vs))
    hi = tuple(o + (n - 0.5) * d for o, d, n in zip(org, vs, shape))
    return lo, hi


def _box_inside(inner_lo, inner_hi, outer_lo, outer_hi, tol):
    return all(a >= b - t for a, b, t in zip(inner_lo, outer_lo, tol)) and all(
        a <= b + t for a, b, t in zip(inner_hi, outer_hi, tol)
    )


def parse_engine_version(text):
    """``"5.5.1-37573402+++UE5+Release-5.5"`` -> ``(5, 5)``; None if unparsable."""
    head = str(text).strip().split("-", 1)[0].split("+", 1)[0]
    parts = head.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def volume_transform(
    layer, mode="world", local_bounds=None, engine=None, sign=VDB_AXIS_SIGN
):
    """Actor transform for a Heterogeneous Volume showing ``layer``.

    ``mode``: ``"world"`` (UE >= 5.4: the SVT carries the VDB transform in VDB
    world units == UE units) -> location 0, scale 100 * sign;
    ``"index"`` (UE 5.3: local space is voxel index space) -> scale
    100 * voxel_size * sign, location = origin (cm, Y mirrored);
    ``"auto"``: world on UE >= 5.4 / index on 5.3 (``engine`` = (major,
    minor)); with no engine version, decide from ``local_bounds`` = (min, max)
    of the actor measured at an identity transform. A measurement that
    contradicts the version-based choice wins (and is noted).
    Returns ``{"location", "scale", "mode", "note"}``.
    """
    s = UE_UNITS_PER_METRE
    vs = _vec3(layer["voxel_size"])
    chosen, note = mode, ""
    if mode == "auto":
        in_world = in_index = None
        if local_bounds is not None:
            lo, hi = (tuple(float(x) for x in b) for b in local_bounds)
            w_lo, w_hi = volume_world_box(layer)
            shape = _vec3(layer["shape"])
            in_world = _box_inside(lo, hi, w_lo, w_hi, tuple(2.0 * d for d in vs))
            in_index = _box_inside(
                lo,
                hi,
                (-1.0, -1.0, -1.0),
                tuple(n + 1.0 for n in shape),
                (1.0, 1.0, 1.0),
            )
        if engine is not None:
            chosen = "world" if tuple(engine) >= (5, 4) else "index"
            note = f"UE {engine[0]}.{engine[1]} -> {chosen} placement"
            if chosen == "world" and in_world is False and in_index:
                chosen, note = (
                    "index",
                    note + "; but measured bounds look like voxel index space -> index",
                )
            elif chosen == "index" and in_index is False and in_world:
                chosen, note = (
                    "world",
                    note + "; but measured bounds look like VDB world units -> world",
                )
        elif in_world:
            chosen, note = (
                "world",
                "measured bounds lie inside the manifest domain (VDB world units)",
            )
        elif in_index:
            chosen, note = (
                "index",
                "measured bounds lie inside [0, shape] (voxel index space)",
            )
        else:
            chosen = "world"
            note = (
                "engine version unknown and bounds inconclusive; assuming world placement (UE >= 5.4) -- "
                "set VDB_PLACEMENT / LES_VDB_PLACEMENT if the volume is misplaced"
            )
    if chosen == "index":
        org = _vec3(layer["origin"])
        return {
            "location": sim_to_ue(org),
            "scale": tuple(s * d * g for d, g in zip(vs, sign)),
            "mode": "index",
            "note": note,
        }
    return {
        "location": (0.0, 0.0, 0.0),
        "scale": tuple(s * g for g in sign),
        "mode": "world",
        "note": note,
    }


# Newest OpenVDB file-format version assumed readable by Unreal's bundled
# OpenVDB (224 = OpenVDB 3.3 .. 11.x; OpenVDB >= 12 writes 225).
UE_MAX_VDB_FILE_VERSION = 224


def volume_warnings(layer, file_version=None):
    """Known Sparse-Volume-Texture import blockers for a volume layer."""
    out = []
    vs = _vec3(layer["voxel_size"])
    if max(vs) - min(vs) > 1e-6 * max(vs):
        out.append(
            f"{layer['name']}: non-uniform voxel size {vs} -- Unreal's OpenVDB importer rejects this "
            "('OpenVDB importer cannot handle non uniform voxels'); resample to cubic voxels"
        )
    if file_version is not None and int(file_version) > UE_MAX_VDB_FILE_VERSION:
        out.append(
            f"{layer['name']}: VDB file format {file_version} > {UE_MAX_VDB_FILE_VERSION}; Unreal's bundled "
            "OpenVDB will likely refuse it -- write with OpenVDB <= 11"
        )
    return out


def geometry_fix(expected_lo, expected_hi, actual_lo, actual_hi):
    """Actor transform that moves imported geometry whose bounds (cm, actor at
    identity) are ``actual`` onto the ``expected`` bounds (cm, UE frame).

    Detects the usual importer surprises: a missing x100 (metres read as cm)
    and a wrong horizontal axis mapping (missing Y mirror, x/y swapped by a
    different glTF axis convention). Tries every signed permutation of the
    horizontal axes times a uniform factor in (1, 100, 0.01) and keeps the one
    whose box matches best (identity wins ties). Returns ``{"scale": (sx, sy,
    sz), "yaw": deg, "factor": f, "error": e, "note": str}``; apply as actor
    scale then yaw rotation (Unreal applies scale, then rotation).
    """
    e_lo, e_hi = [float(v) for v in expected_lo], [float(v) for v in expected_hi]
    a_lo, a_hi = [float(v) for v in actual_lo], [float(v) for v in actual_hi]
    e_ctr = [0.5 * (l + h) for l, h in zip(e_lo, e_hi)]
    e_size = [h - l for l, h in zip(e_lo, e_hi)]
    norm = max(max(e_size), 1e-9)
    best = None
    for factor in (1.0, 100.0, 0.01):
        for swap in (False, True):
            for sx in (1.0, -1.0):
                for sy in (1.0, -1.0):
                    # M maps actual (X, Y) -> corrected (X', Y')
                    m = ((0.0, sx), (sy, 0.0)) if swap else ((sx, 0.0), (0.0, sy))
                    xs, ys = [], []
                    for x in (a_lo[0], a_hi[0]):
                        for y in (a_lo[1], a_hi[1]):
                            xs.append(factor * (m[0][0] * x + m[0][1] * y))
                            ys.append(factor * (m[1][0] * x + m[1][1] * y))
                    lo = [min(xs), min(ys), factor * a_lo[2]]
                    hi = [max(xs), max(ys), factor * a_hi[2]]
                    err = (
                        sum(
                            abs(0.5 * (l + h) - c) + abs((h - l) - s)
                            for l, h, c, s in zip(lo, hi, e_ctr, e_size)
                        )
                        / norm
                    )
                    if best is None or err < best[0] - 1e-6:
                        best = (err, factor, m)
    err, factor, m = best
    (a, b), (c, d) = m
    det = a * d - b * c
    sy = 1.0
    if det < 0:  # M = R(yaw) @ diag(1, -1)
        a, b, c, d = a, -b, c, -d
        sy = -1.0
    yaw = math.degrees(math.atan2(c, a))
    yaw = 0.0 if abs(yaw) < 1e-9 else yaw
    ident = factor == 1.0 and sy == 1.0 and yaw == 0.0
    note = (
        "ok"
        if ident
        else f"applied scale x{factor:g}, y-sign {sy:+g}, yaw {yaw:+g} deg"
    )
    return {
        "scale": (factor, factor * sy, factor),
        "yaw": yaw,
        "factor": factor,
        "error": err,
        "note": note,
    }


def apply_fix(point, fix):
    """Apply a :func:`geometry_fix` transform (scale, then yaw) to a UE point."""
    x, y, z = (
        float(point[0]) * fix["scale"][0],
        float(point[1]) * fix["scale"][1],
        float(point[2]) * fix["scale"][2],
    )
    t = math.radians(fix["yaw"])
    return (x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t), z)


def slice_quad(layer, uv_axes=None):
    """Four sim-frame corners and glTF-convention UVs (origin top-left, V down)
    of a slice layer's plane. PNG row 0 is the max-v edge, so the max-v edge
    gets V = 0. Corner order: (u0,v0), (u1,v0), (u1,v1), (u0,v1)."""
    uv_axes = uv_axes or SLICE_UV_AXES
    axis = layer["axis"]
    ua, va = uv_axes[axis]
    (u0, v0), (u1, v1) = layer["extent"]
    idx = {"x": 0, "y": 1, "z": 2}
    corners = []
    for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
        p = [0.0, 0.0, 0.0]
        p[idx[axis]] = float(layer["position"])
        p[idx[ua]] = float(u)
        p[idx[va]] = float(v)
        corners.append(tuple(p))
    uvs = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    return corners, uvs


def asset_name(s):
    """Unreal-safe asset name."""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(s))
    return out if out and not out[0].isdigit() else "_" + out


REQUIRED_KEYS = ("version", "case", "timeline", "render", "layers", "shots")


def parse_manifest(manifest):
    """Validate a manifest dict (version 1) and fill defaults. Raises ValueError."""
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest is missing keys: {missing}")
    if int(manifest["version"]) != 1:
        raise ValueError(
            f"unsupported manifest version {manifest['version']} (expected 1)"
        )
    tl = manifest["timeline"]
    for k in ("fps", "n_frames"):
        if k not in tl:
            raise ValueError(f"manifest timeline lacks {k!r}")
    names = [layer["name"] for layer in manifest["layers"]]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate layer names: {names}")
    for layer in manifest["layers"]:
        for k in ("name", "type", "pattern"):
            if k not in layer:
                raise ValueError(f"layer {layer.get('name')!r} lacks {k!r}")
        layer.setdefault("frame_step", 1)
        layer.setdefault("n_files", int(tl["n_frames"]))
    manifest["render"].setdefault("width", 1920)
    manifest["render"].setdefault("height", 1080)
    manifest["render"].setdefault("look", "dark")
    if not manifest["shots"]:
        raise ValueError("manifest has no shots")
    return manifest


def load_manifest(bundle):
    path = pathlib.Path(bundle) / "manifest.json"
    return parse_manifest(json.loads(path.read_text()))


def build_plan(manifest, bundle, content_root=CONTENT_ROOT):
    """Everything the editor stages need, resolved without ``unreal``:
    content paths, per-layer source files (and whether they exist), LUTs."""
    bundle = pathlib.Path(bundle)
    case = asset_name(manifest["case"]["name"])
    root = f"{content_root}/{case}"
    plan = {
        "case": case,
        "root": root,
        "map": f"{root}/Maps/LES_{case}",
        "sequence": f"{root}/Sequences/LS_{case}",
        "mrq_config": f"{root}/Render/MRQ_{case}",
        "output_dir": str(bundle / "unreal" / "render"),
        "fps": fps_to_fraction(manifest["timeline"]["fps"]),
        "n_frames": int(manifest["timeline"]["n_frames"]),
        "resolution": (
            int(manifest["render"]["width"]),
            int(manifest["render"]["height"]),
        ),
        "look": (
            manifest["render"].get("look", "dark")
            if manifest["render"].get("look") in LOOKS
            else "dark"
        ),
        "geometry": {},
        "layers": [],
    }
    geo = manifest.get("geometry") or {}
    for part in ("buildings", "ground"):
        entry = geo.get(part) or {}
        glb = entry.get("glb")
        plan["geometry"][part] = str(bundle / glb) if glb else None
    for layer in manifest["layers"]:
        name, typ = layer["name"], layer["type"]
        item = {"name": name, "type": typ, "asset": asset_name(name), "layer": layer}
        lut = bundle / "unreal" / "luts" / f"{name}_lut.png"
        item["lut"] = str(lut) if lut.exists() else None
        if typ == "volume":
            item["source"] = str(bundle / layer["pattern"].format(frame=0))
        elif typ in ("particles", "isosurface"):
            item["source"] = str(bundle / "alembic" / f"{name}.abc")
        elif typ == "slice":
            item["source"] = str((bundle / layer["pattern"].format(frame=0)).parent)
            item["plane"] = str(bundle / "unreal" / "meshes" / f"{name}_plane.glb")
        else:
            item["source"] = None
        item["exists"] = bool(item["source"]) and pathlib.Path(item["source"]).exists()
        plan["layers"].append(item)
    return plan


def parse_args(argv, env=None, default_bundle=None):
    """``--bundle DIR``, ``--quit``, ``--renderer``, ``--skip STAGE[,STAGE]``;
    falls back to ``LES_BUNDLE`` then ``default_bundle``."""
    env = os.environ if env is None else env
    opts = {"bundle": None, "quit": False, "renderer": None, "skip": set()}
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--bundle="):
            opts["bundle"] = a.split("=", 1)[1]
        elif a == "--bundle" and i + 1 < len(args):
            opts["bundle"] = args[i + 1]
            i += 1
        elif a == "--quit":
            opts["quit"] = True
        elif a.startswith("--renderer="):
            opts["renderer"] = a.split("=", 1)[1]
        elif a == "--renderer" and i + 1 < len(args):
            opts["renderer"] = args[i + 1]
            i += 1
        elif a.startswith("--skip="):
            opts["skip"] |= set(a.split("=", 1)[1].split(","))
        elif a == "--skip" and i + 1 < len(args):
            opts["skip"] |= set(args[i + 1].split(","))
            i += 1
        i += 1
    opts["bundle"] = opts["bundle"] or env.get("LES_BUNDLE") or default_bundle
    if env.get("LES_QUIT_WHEN_DONE"):
        opts["quit"] = True
    return opts


# =============================================================================
# Unreal side. Nothing below runs at import time.
# =============================================================================

REPORT = []  # (stage, status, message)


def _log(msg):
    if unreal is not None:
        unreal.log(f"{LOG_PREFIX} {msg}")
    else:
        print(f"{LOG_PREFIX} {msg}")


def _warn(msg):
    if unreal is not None:
        unreal.log_warning(f"{LOG_PREFIX} {msg}")
    else:
        print(f"{LOG_PREFIX} WARNING {msg}")


def _stage(title, fallback):
    """Decorator: run a stage, log + record failures with the manual fallback,
    never raise."""

    def deco(fn):
        def wrapper(*a, **kw):
            _log(f"--- {title}")
            try:
                out = fn(*a, **kw)
                REPORT.append((title, "ok", ""))
                return out
            except (
                Exception
            ) as exc:  # noqa: BLE001 -- one failed layer must not abort the scene
                _warn(
                    f"{title} FAILED: {type(exc).__name__}: {exc}. Manual fallback: {fallback}"
                )
                REPORT.append(
                    (
                        title,
                        "failed",
                        f"{type(exc).__name__}: {exc} | fallback: {fallback}",
                    )
                )
                return None

        wrapper.__name__ = fn.__name__
        return wrapper

    return deco


def _enum(enum_name, *names):
    """First existing member of ``unreal.<enum_name>`` among ``names`` (also
    tries the trailing-underscore spelling some enums use)."""
    enum = getattr(unreal, enum_name)
    for n in names:
        for cand in (n, n + "_"):
            if hasattr(enum, cand):
                return getattr(enum, cand)
    raise AttributeError(f"unreal.{enum_name} has none of {names}")


def _set(obj, prop, value, quiet=False):
    try:
        obj.set_editor_property(prop, value)
        return True
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            _warn(f"could not set {type(obj).__name__}.{prop}: {exc}")
        return False


def _vec(v):
    return unreal.Vector(float(v[0]), float(v[1]), float(v[2]))


def _rot(pitch, yaw, roll):
    return unreal.Rotator(roll=float(roll), pitch=float(pitch), yaw=float(yaw))


def _lc(rgb):
    return unreal.LinearColor(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _tools():
    return unreal.AssetToolsHelpers.get_asset_tools()


def _eal():
    return unreal.EditorAssetLibrary


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def _frame(n):
    return unreal.FrameNumber(int(n))


def _ensure_dir(path):
    if not _eal().does_directory_exist(path):
        _eal().make_directory(path)


def _load_or_create(name, folder, cls, factory):
    full = f"{folder}/{name}"
    if _eal().does_asset_exist(full):
        asset = _eal().load_asset(full)
        if asset is not None and isinstance(asset, cls):
            return asset
        _eal().delete_asset(full)
    _ensure_dir(folder)
    return _tools().create_asset(name, folder, cls, factory)


def _import(filename, dest, name=None, options=None, factory=None):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(filename))
    task.set_editor_property("destination_path", dest)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    if name:
        task.set_editor_property("destination_name", name)
    if options is not None:
        task.set_editor_property("options", options)
    if factory is not None:
        task.set_editor_property("factory", factory)
    _ensure_dir(dest)
    _tools().import_asset_tasks([task])
    objs = []
    try:
        objs = list(task.get_objects())
    except Exception:  # noqa: BLE001
        pass
    if not objs:
        for p in task.get_editor_property("imported_object_paths") or []:
            a = _eal().load_asset(p)
            if a is not None:
                objs.append(a)
    if not objs:  # some importers don't report back: scan the folder
        for p in _eal().list_assets(dest, recursive=True, include_folder=False):
            a = _eal().load_asset(p)
            if a is not None:
                objs.append(a)
    return objs


def _of_class(objs, class_name):
    return [o for o in objs if o is not None and type(o).__name__ == class_name]


def _spawn(cls, location=(0, 0, 0), rotation=(0, 0, 0), label=None, folder=None):
    actor = _actors().spawn_actor_from_class(cls, _vec(location), _rot(*rotation))
    _tag(actor, label, folder)
    return actor


def _spawn_object(obj, location=(0, 0, 0), label=None, folder=None):
    actor = _actors().spawn_actor_from_object(obj, _vec(location))
    _tag(actor, label, folder)
    return actor


def _tag(actor, label, folder):
    if actor is None:
        raise RuntimeError("spawn returned None")
    try:
        tags = list(actor.get_editor_property("tags"))
        actor.set_editor_property("tags", tags + [unreal.Name(ACTOR_TAG)])
    except Exception:  # noqa: BLE001
        pass
    if label:
        try:
            actor.set_actor_label(label)
        except Exception:  # noqa: BLE001
            pass
    if folder:
        try:
            actor.set_folder_path(unreal.Name(f"LES/{folder}"))
        except Exception:  # noqa: BLE001
            pass


def _actor_bounds(actor):
    origin, extent = actor.get_actor_bounds(False)
    lo = (origin.x - extent.x, origin.y - extent.y, origin.z - extent.z)
    hi = (origin.x + extent.x, origin.y + extent.y, origin.z + extent.z)
    return lo, hi


# --- material graph helpers ----------------------------------------------------


def _mel():
    return unreal.MaterialEditingLibrary


def _expr(mat, cls_name, x, y, **props):
    cls = getattr(unreal, cls_name)
    e = _mel().create_material_expression(mat, cls, x, y)
    for k, v in props.items():
        _set(e, k, v)
    return e


def _connect(src, src_outs, dst, dst_ins):
    """Try every (output, input) name pair; pin names differ between versions."""
    for o in src_outs:
        for i in dst_ins:
            if _mel().connect_material_expressions(src, o, dst, i):
                return True
    _warn(
        f"could not connect {type(src).__name__}{src_outs} -> {type(dst).__name__}{dst_ins}"
    )
    return False


def _to_prop(src, src_outs, prop_names):
    prop = _enum("MaterialProperty", *prop_names)
    for o in src_outs:
        if _mel().connect_material_property(src, o, prop):
            return True
    _warn(f"could not connect {type(src).__name__}{src_outs} -> {prop_names}")
    return False


def _scalar(mat, name, value, x, y):
    return _expr(
        mat,
        "MaterialExpressionScalarParameter",
        x,
        y,
        parameter_name=name,
        default_value=float(value),
    )


def _vector(mat, name, rgb, x, y):
    return _expr(
        mat,
        "MaterialExpressionVectorParameter",
        x,
        y,
        parameter_name=name,
        default_value=_lc(rgb),
    )


_BINOP_INPUTS = {"Power": (("Base",), ("Exp", "Exponent"))}


def _binop(mat, op, a, b, x, y, a_out=("",), b_out=("",)):
    e = _expr(mat, f"MaterialExpression{op}", x, y)
    ins_a, ins_b = _BINOP_INPUTS.get(op, (("A",), ("B",)))
    _connect(a, a_out, e, ins_a)
    _connect(b, b_out, e, ins_b)
    return e


def _fresh_material(ctx, name):
    mat = _load_or_create(
        name,
        f"{ctx['plan']['root']}/Materials",
        unreal.Material,
        unreal.MaterialFactoryNew(),
    )
    _mel().delete_all_material_expressions(mat)
    return mat


def _finish_material(mat):
    try:
        _mel().layout_material_expressions(mat)
    except Exception:  # noqa: BLE001
        pass
    _mel().recompile_material(mat)
    _eal().save_asset(mat.get_path_name(), only_if_is_dirty=False)
    return mat


def _lut_sample(mat, lut_tex, value, vmin_param, vmax_param, x, y):
    """TextureSample(LUT, (saturate((value - vmin) / (vmax - vmin)), 0.5))."""
    sub = _binop(mat, "Subtract", value, vmin_param, x, y)
    rng = _binop(mat, "Subtract", vmax_param, vmin_param, x, y + 80)
    div = _binop(mat, "Divide", sub, rng, x + 160, y)
    sat = _expr(mat, "MaterialExpressionSaturate", x + 320, y)
    _connect(div, ("",), sat, ("", "Input"))
    half = _expr(mat, "MaterialExpressionConstant", x + 320, y + 80, r=0.5)
    uv = _binop(mat, "AppendVector", sat, half, x + 480, y)
    tex = _expr(mat, "MaterialExpressionTextureSample", x + 640, y)
    if lut_tex is not None:
        _set(tex, "texture", lut_tex)
    _set(
        tex,
        "sampler_type",
        _enum("MaterialSamplerType", "SAMPLERTYPE_COLOR"),
        quiet=True,
    )
    try:
        _set(
            tex,
            "mip_value_mode",
            _enum("TextureMipValueMode", "TMVM_MIP_LEVEL"),
            quiet=True,
        )
    except AttributeError:
        pass
    _connect(uv, ("",), tex, ("UVs", "Coordinates"))
    return tex, sat


# =============================================================================
# Stages
# =============================================================================


@_stage(
    "level",
    "File > New Level (Empty Level), save it as the map path in the log, re-run",
)
def stage_level(ctx):
    plan = ctx["plan"]
    level_sys = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    _ensure_dir(f"{plan['root']}/Maps")
    if _eal().does_asset_exist(plan["map"]):
        level_sys.load_level(plan["map"])
        stale = [
            a
            for a in _actors().get_all_level_actors()
            if ACTOR_TAG in [str(t) for t in (a.get_editor_property("tags") or [])]
        ]
        if stale:
            _actors().destroy_actors(stale)
        _log(f"re-using map {plan['map']} (removed {len(stale)} old LES actors)")
    else:
        if not level_sys.new_level(plan["map"]):
            raise RuntimeError(f"new_level({plan['map']}) returned False")
        _log(f"created map {plan['map']}")
    ctx["actors"] = {}


@_stage(
    "LUT textures",
    "import <bundle>/unreal/luts/*.png by hand: sRGB on, Filter Bilinear, X/Y Tiling Clamp, No Mipmaps",
)
def stage_luts(ctx):
    ctx["luts"] = {}
    for item in ctx["plan"]["layers"]:
        if not item["lut"]:
            continue
        objs = _import(
            item["lut"], f"{ctx['plan']['root']}/LUTs", f"T_{item['asset']}_LUT"
        )
        tex = (_of_class(objs, "Texture2D") or [None])[0]
        if tex is None:
            _warn(f"LUT import for {item['name']} produced no Texture2D")
            continue
        _set(tex, "srgb", True)
        _set(tex, "filter", _enum("TextureFilter", "TF_BILINEAR"))
        _set(tex, "address_x", _enum("TextureAddress", "TA_CLAMP"))
        _set(tex, "address_y", _enum("TextureAddress", "TA_CLAMP"))
        _set(tex, "mip_gen_settings", _enum("TextureMipGenSettings", "TMGS_NO_MIPMAPS"))
        _set(
            tex,
            "compression_settings",
            _enum("TextureCompressionSettings", "TC_EDITOR_ICON", "TC_DEFAULT"),
        )
        _set(tex, "never_stream", True, quiet=True)
        _eal().save_asset(tex.get_path_name(), only_if_is_dirty=False)
        ctx["luts"][item["name"]] = tex


def _surface_material(ctx, name, color, roughness, rim_glow=0.0):
    mat = _fresh_material(ctx, name)
    base = _vector(mat, "BaseColor", color, -600, 0)
    rough = _scalar(mat, "Roughness", roughness, -600, 200)
    _to_prop(base, ("",), ("MP_BASE_COLOR",))
    _to_prop(rough, ("",), ("MP_ROUGHNESS",))
    if rim_glow > 0:
        fres = _expr(mat, "MaterialExpressionFresnel", -600, 350)
        glow = _scalar(mat, "RimGlow", rim_glow, -600, 450)
        em = _binop(mat, "Multiply", fres, glow, -300, 350)
        _to_prop(em, ("",), ("MP_EMISSIVE_COLOR",))
    return _finish_material(mat)


@_stage(
    "geometry",
    "import geometry/buildings.glb and ground.glb (Interchange glTF), place at the origin, check scale x100 and Y mirror",
)
def stage_geometry(ctx):
    plan, look = ctx["plan"], LOOKS[ctx["plan"]["look"]]
    mats = {
        "buildings": _surface_material(
            ctx,
            "M_Buildings",
            look["building_color"],
            look["building_roughness"],
            look["rim_glow"],
        ),
        "ground": _surface_material(ctx, "M_Ground", look["ground_color"], 0.95),
    }
    geo = ctx["manifest"].get("geometry") or {}
    for part, path in plan["geometry"].items():
        if not path or not pathlib.Path(path).exists():
            _warn(f"geometry {part}: {path} missing, skipped")
            continue
        objs = _import(path, f"{plan['root']}/Geometry/{part}")
        meshes = _of_class(objs, "StaticMesh")
        if not meshes:
            raise RuntimeError(
                f"{path}: importer produced no StaticMesh (got {[type(o).__name__ for o in objs]})"
            )
        for i, mesh in enumerate(meshes):
            actor = _spawn_object(mesh, label=f"LES_{part}_{i}", folder="Geometry")
            comp = actor.get_editor_property("static_mesh_component")
            for slot in range(
                max(1, len(mesh.get_editor_property("static_materials") or []))
            ):
                comp.set_material(slot, mats[part])
            ctx["actors"].setdefault(part, []).append(actor)
    if (
        GEOMETRY_AUTOFIX
        and geo.get("buildings_bounds")
        and ctx["actors"].get("buildings")
    ):
        lo, hi = geo["buildings_bounds"]
        e = [sim_to_ue(lo), sim_to_ue(hi)]
        e_lo, e_hi = tuple(map(min, *e)), tuple(map(max, *e))
        bounds = [_actor_bounds(a) for a in ctx["actors"]["buildings"]]
        a_lo = tuple(min(b[0][k] for b in bounds) for k in range(3))
        a_hi = tuple(max(b[1][k] for b in bounds) for k in range(3))
        fix = geometry_fix(e_lo, e_hi, a_lo, a_hi)
        _log(
            f"buildings bounds check: expected {e_lo}..{e_hi}, got {a_lo}..{a_hi}: {fix['note']} (error {fix['error']:.3f})"
        )
        if fix["error"] > 0.05:
            _warn(
                "building bounds do not match the manifest even after the best axis fix; leaving the "
                "geometry as imported -- check the glTF import (expected (x, y, z)_m -> (100x, -100y, 100z)_cm)"
            )
            return
        ctx["geometry_fix"] = fix
        for kind in ("buildings", "ground"):
            for a in ctx["actors"].get(kind, []):
                _apply_fix_to_actor(a, fix)


def _apply_fix_to_actor(actor, fix):
    if fix is None or fix["note"] == "ok":
        return
    actor.set_actor_scale3d(_vec(fix["scale"]))
    actor.set_actor_rotation(_rot(0.0, fix["yaw"], 0.0), False)


@_stage(
    "lighting + post-process",
    "add Directional Light / Sky Light / Sky Atmosphere / Exponential Height Fog and an unbound Post Process Volume with Manual exposure",
)
def stage_look(ctx):
    look_name = ctx["plan"]["look"]
    look = LOOKS[look_name]
    if look["sun_lux"] > 0:
        sun = _spawn(
            unreal.DirectionalLight,
            (0, 0, 10000),
            (-38.0, 135.0, 0.0),
            "LES_Sun",
            "Lighting",
        )
        lc = sun.get_editor_property("light_component")
        lc.set_intensity(look["sun_lux"])
        lc.set_light_color(_lc((1.0, 0.96, 0.9)))
        try:
            lc.set_atmosphere_sun_light(True)
        except Exception:  # noqa: BLE001
            _set(lc, "atmosphere_sun_light", True)
        _set(lc, "light_source_angle", 0.5, quiet=True)
    sky = _spawn(unreal.SkyLight, (0, 0, 10000), (0, 0, 0), "LES_SkyLight", "Lighting")
    sc = sky.get_editor_property("light_component")
    _set(sc, "real_time_capture", bool(look["sky_atmosphere"]))
    _set(sc, "lower_hemisphere_is_black", True, quiet=True)
    sc.set_intensity(look["sky_intensity"])
    if look_name == "dark":
        sc.set_light_color(_lc((0.35, 0.45, 0.7)))
    if look["sky_atmosphere"]:
        _spawn(
            unreal.SkyAtmosphere, (0, 0, 0), (0, 0, 0), "LES_SkyAtmosphere", "Lighting"
        )
    if look["fog"]:
        fog = _spawn(
            unreal.ExponentialHeightFog,
            (0, 0, 0),
            (0, 0, 0),
            "LES_HeightFog",
            "Lighting",
        )
        fc = fog.get_editor_property("component")
        fc.set_fog_density(0.004)
        fc.set_fog_height_falloff(0.05)
        fc.set_volumetric_fog(False)

    ppv = _spawn(
        unreal.PostProcessVolume, (0, 0, 0), (0, 0, 0), "LES_PostProcess", "Lighting"
    )
    _set(ppv, "unbound", True)
    _set(ppv, "priority", 10.0)
    s = ppv.get_editor_property("settings")
    pp = {
        "auto_exposure_method": _enum("AutoExposureMethod", "AEM_MANUAL"),
        "auto_exposure_apply_physical_camera_exposure": False,
        "auto_exposure_bias": look["exposure_bias"],
        "bloom_intensity": look["bloom"],
        "bloom_threshold": look["bloom_threshold"],
        "ambient_occlusion_intensity": look["ao"],
        "motion_blur_amount": 0.4,
        "motion_blur_max": 2.0,
        "vignette_intensity": look["vignette"],
        "path_tracing_max_bounces": 16,
    }
    for k, v in pp.items():
        _set(s, f"override_{k}", True, quiet=True)
        _set(s, k, v)
    ppv.set_editor_property("settings", s)


def _volume_material(ctx, item, svt):
    layer, look = item["layer"], LOOKS[ctx["plan"]["look"]]
    mat = _fresh_material(ctx, f"M_Volume_{item['asset']}")
    _set(mat, "material_domain", _enum("MaterialDomain", "MD_VOLUME"))
    _set(mat, "blend_mode", _enum("BlendMode", "BLEND_ADDITIVE"))
    _set(mat, "used_with_heterogeneous_volumes", True)
    # UVW = (LocalPosition - LocalBounds.Min) / LocalBounds.FullExtents
    lpos = _expr(mat, "MaterialExpressionLocalPosition", -1600, 0)
    bounds = _expr(mat, "MaterialExpressionObjectLocalBounds", -1600, 150)
    sub = _expr(mat, "MaterialExpressionSubtract", -1400, 0)
    _connect(lpos, ("",), sub, ("A",))
    _connect(bounds, ("Min", "Min Bounds"), sub, ("B",))
    div = _expr(mat, "MaterialExpressionDivide", -1250, 0)
    _connect(sub, ("",), div, ("A",))
    _connect(bounds, ("Full Extents", "FullExtents", "Extents", "Size"), div, ("B",))
    samp = _expr(
        mat,
        "MaterialExpressionSparseVolumeTextureSampleParameter",
        -1050,
        0,
        parameter_name="SparseVolumeTexture",
    )
    if svt is not None:
        _set(samp, "sparse_volume_texture", svt)
    _connect(div, ("",), samp, ("UVs", "UV", "UVW", "Coordinates"))
    value = _expr(
        mat,
        "MaterialExpressionComponentMask",
        -800,
        0,
        r=True,
        g=False,
        b=False,
        a=False,
    )
    _connect(
        samp,
        ("Attributes A", "AttributesA", "Attributes", "RGBA"),
        value,
        ("", "Input"),
    )

    d0, d1 = layer.get("density_range", layer.get("range", [0.0, 1.0]))
    c0, c1 = (layer.get("range") or [d0, d1])[:2]
    dmin = _scalar(mat, "DensityMin", d0, -800, 200)
    dmax = _scalar(mat, "DensityMax", d1, -800, 280)
    cmin = _scalar(mat, "ColorMin", c0, -800, 400)
    cmax = _scalar(mat, "ColorMax", c1, -800, 480)
    # density in [0, 1]
    dsub = _binop(mat, "Subtract", value, dmin, -600, 200)
    drng = _binop(mat, "Subtract", dmax, dmin, -600, 280)
    ddiv = _binop(mat, "Divide", dsub, drng, -450, 200)
    dens = _expr(mat, "MaterialExpressionSaturate", -300, 200)
    _connect(ddiv, ("",), dens, ("", "Input"))
    tex, _ = _lut_sample(
        mat, ctx.get("luts", {}).get(item["name"]), value, cmin, cmax, -600, 400
    )

    dscale = _scalar(mat, "DensityScale", layer.get("density_scale", 1.0), -300, 300)
    estr = _scalar(
        mat, "EmissionStrength", layer.get("emission_strength", 1.0), -300, 600
    )
    epcm = _scalar(
        mat,
        "EmissionPerCm",
        look["volume_emission_per_cm"] * look["emissive_gain"],
        -300,
        680,
    )
    xpcm = _scalar(mat, "ExtinctionPerCm", look["volume_extinction_per_cm"], -300, 380)
    d_eff = _binop(mat, "Multiply", dens, dscale, -150, 200)
    ext = _binop(mat, "Multiply", d_eff, xpcm, 0, 250)
    e1 = _binop(mat, "Multiply", tex, d_eff, 0, 450, a_out=("RGB", ""))
    e2 = _binop(mat, "Multiply", e1, estr, 150, 450)
    emis = _binop(mat, "Multiply", e2, epcm, 300, 450)
    _to_prop(tex, ("RGB", ""), ("MP_BASE_COLOR",))  # albedo
    _to_prop(emis, ("",), ("MP_EMISSIVE_COLOR",))
    _to_prop(
        ext, ("",), ("MP_SUBSURFACE_COLOR",)
    )  # shown as "Extinction" in the Volume domain
    return _finish_material(mat)


@_stage(
    "volumes",
    "import volumes/<name>/<name>.0000.vdb via the Content Browser (map grid <name> -> Attributes A.R, 16-bit float), "
    "create a Volume-domain Additive material with 'Used with Heterogeneous Volumes' (or instance /Engine/EngineMaterials/SparseVolumeMaterial), "
    "place a Heterogeneous Volume at the origin with scale (100, -100, 100)",
)
def stage_volumes(ctx):
    for item in ctx["plan"]["layers"]:
        if item["type"] != "volume":
            continue
        _one_volume(ctx, item)


def _one_volume(ctx, item):
    name, layer = item["name"], item["layer"]
    if not item["exists"]:
        _warn(f"volume {name}: {item['source']} missing, skipped")
        return
    for w in volume_warnings(layer):
        _warn(w)
    objs = _import(item["source"], f"{ctx['plan']['root']}/Volumes/{item['asset']}")
    svts = [o for o in objs if "SparseVolumeTexture" in type(o).__name__]
    if not svts:
        raise RuntimeError(
            f"{item['source']}: no SparseVolumeTexture created (got {[type(o).__name__ for o in objs]})"
        )
    svt = svts[0]
    mat = _volume_material(ctx, item, svt)
    actor = _spawn(
        unreal.HeterogeneousVolume, (0, 0, 0), (0, 0, 0), f"LES_{name}", "Volumes"
    )
    comp = actor.get_editor_property("heterogeneous_volume_component")
    comp.set_material(0, mat)
    n_files = int(layer["n_files"])
    fps = float(ctx["manifest"]["timeline"]["fps"])
    _set(comp, "playing", False)
    _set(comp, "looping", False)
    _set(comp, "start_frame", 0.0)
    _set(comp, "end_frame", float(max(n_files - 1, 0)))
    _set(comp, "frame_rate", fps / max(int(layer.get("frame_step", 1)), 1))
    _set(comp, "frame", 0.0)
    _set(comp, "pivot_at_centroid", False, quiet=True)
    local = None
    if VDB_PLACEMENT == "auto":
        try:
            local = _actor_bounds(actor)
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"volume {name}: bounds query failed ({exc}); assuming world placement"
            )
    xf = volume_transform(layer, VDB_PLACEMENT, local, ctx.get("engine"))
    actor.set_actor_location(_vec(xf["location"]), False, False)
    actor.set_actor_scale3d(_vec(xf["scale"]))
    _log(
        f"volume {name}: placement={xf['mode']} location={xf['location']} scale={xf['scale']} {xf['note']}"
    )
    ctx["actors"][name] = [actor]
    ctx.setdefault("volume_components", {})[name] = comp


def _abc_settings(n_frames):
    s = unreal.AbcImportSettings()
    s.set_editor_property("import_type", _enum("AlembicImportType", "GEOMETRY_CACHE"))
    conv = s.get_editor_property("conversion_settings")
    _set(conv, "preset", _enum("AbcConversionPreset", "CUSTOM"), quiet=True)
    conv.set_editor_property("rotation", _vec(ABC_ROTATION))
    conv.set_editor_property("scale", _vec(ABC_SCALE))
    _set(conv, "flip_v", False, quiet=True)
    s.set_editor_property("conversion_settings", conv)
    gc = s.get_editor_property("geometry_cache_settings")
    _set(gc, "flatten_tracks", True, quiet=True)
    _set(gc, "apply_constant_topology_optimizations", False, quiet=True)
    s.set_editor_property("geometry_cache_settings", gc)
    samp = s.get_editor_property("sampling_settings")
    _set(samp, "frame_start", 0, quiet=True)
    _set(samp, "frame_end", int(n_frames) - 1, quiet=True)
    _set(samp, "skip_empty", False, quiet=True)
    s.set_editor_property("sampling_settings", samp)
    return s


def _vertex_color_material(ctx, item, emissive_default):
    look = LOOKS[ctx["plan"]["look"]]
    mat = _fresh_material(ctx, f"M_{item['asset']}")
    _set(mat, "used_with_geometry_cache", True)
    _set(mat, "two_sided", True)
    vc = _expr(mat, "MaterialExpressionVertexColor", -1100, 0)
    # Blender writes "Cd" as sRGB-encoded floats (PLY bytes / 255). Decode with
    # a power curve; set VertexColorGamma to 1 if the importer already did.
    gamma = _scalar(mat, "VertexColorGamma", VERTEX_COLOR_GAMMA, -1100, 150)
    vc_lin = _binop(mat, "Power", vc, gamma, -900, 0, a_out=("RGB", ""))
    fallback = _vector(mat, "FallbackColor", (0.8, 0.8, 0.8), -900, 200)
    use_vc = _scalar(mat, "UseVertexColor", 1.0, -900, 300)
    col = _expr(mat, "MaterialExpressionLinearInterpolate", -600, 0)
    _connect(fallback, ("",), col, ("A",))
    _connect(vc_lin, ("",), col, ("B",))
    _connect(use_vc, ("",), col, ("Alpha",))
    gain = _scalar(
        mat, "EmissiveGain", emissive_default * look["emissive_gain"], -600, 250
    )
    em = _binop(mat, "Multiply", col, gain, -350, 150)
    rough = _scalar(mat, "Roughness", 0.45, -600, 350)
    _to_prop(col, ("",), ("MP_BASE_COLOR",))
    _to_prop(em, ("",), ("MP_EMISSIVE_COLOR",))
    _to_prop(rough, ("",), ("MP_ROUGHNESS",))
    return _finish_material(mat)


def _groom_material(ctx, item):
    look = LOOKS[ctx["plan"]["look"]]
    layer = item["layer"]
    mat = _fresh_material(ctx, f"M_Groom_{item['asset']}")
    _set(mat, "shading_model", _enum("MaterialShadingModel", "MSM_HAIR"))
    _set(mat, "used_with_hair_strands", True)
    # No per-vertex speed on the groom: colour along the strand instead.
    # U = 0 at the root (index 0 = head = newest point), 1 at the tip.
    ha = _expr(mat, "MaterialExpressionHairAttributes", -1300, 0)
    one = _expr(mat, "MaterialExpressionConstant", -1300, 200, r=1.0)
    along = _binop(mat, "Subtract", one, ha, -1100, 0, b_out=("U",))
    zero = _expr(mat, "MaterialExpressionConstant", -1100, 200, r=0.0)
    onep = _expr(mat, "MaterialExpressionConstant", -1100, 280, r=1.0)
    tex, sat = _lut_sample(
        mat, ctx.get("luts", {}).get(item["name"]), along, zero, onep, -900, 0
    )
    fade_pow = _scalar(mat, "TailFadePower", 1.5, -500, 250)
    fade = _binop(mat, "Power", sat, fade_pow, -300, 250)
    gain = _scalar(
        mat,
        "EmissiveGain",
        float(layer.get("emission_strength", 1.0)) * look["emissive_gain"],
        -300,
        350,
    )
    e1 = _binop(mat, "Multiply", tex, fade, -100, 100, a_out=("RGB", ""))
    em = _binop(mat, "Multiply", e1, gain, 100, 100)
    dark = _vector(mat, "BaseColor", (0.02, 0.02, 0.02), -100, 350)
    _to_prop(dark, ("",), ("MP_BASE_COLOR",))
    _to_prop(em, ("",), ("MP_EMISSIVE_COLOR",))
    return _finish_material(mat)


def _import_groom(ctx, item):
    opts = unreal.GroomImportOptions()
    conv = unreal.GroomConversionSettings()
    conv.set_editor_property("rotation", _vec(ABC_ROTATION))
    conv.set_editor_property("scale", _vec(ABC_SCALE))
    opts.set_editor_property("conversion_settings", conv)
    objs = _import(
        item["source"],
        f"{ctx['plan']['root']}/Particles/{item['asset']}",
        options=opts,
        factory=unreal.HairStrandsFactory(),
    )
    grooms = _of_class(objs, "GroomAsset")
    caches = _of_class(objs, "GroomCache")
    if not grooms:
        raise RuntimeError(
            f"no GroomAsset from {item['source']} (got {[type(o).__name__ for o in objs]})"
        )
    if not caches:
        _warn(
            f"{item['name']}: groom imported but no GroomCache was created -- the particles will be static. "
            "Re-import alembic/<layer>.abc from the Content Browser and tick 'Import Groom Cache'."
        )
    actor = _spawn(
        unreal.GroomActor, (0, 0, 0), (0, 0, 0), f"LES_{item['name']}", "Particles"
    )
    comp = actor.get_editor_property("groom_component")
    comp.set_groom_asset(grooms[0])
    if caches:
        comp.set_groom_cache(caches[0])
    try:
        comp.set_enable_simulation(False)
    except Exception:  # noqa: BLE001
        pass
    comp.set_material(0, _groom_material(ctx, item))
    ctx["actors"][item["name"]] = [actor]
    ctx.setdefault("groom_caches", {})[item["name"]] = caches[0] if caches else None
    return actor


def _import_geometry_cache(ctx, item, source, emissive_default):
    objs = _import(
        source,
        f"{ctx['plan']['root']}/Caches/{item['asset']}",
        options=_abc_settings(ctx["plan"]["n_frames"]),
        factory=unreal.AlembicImportFactory(),
    )
    caches = _of_class(objs, "GeometryCache")
    if not caches:
        raise RuntimeError(
            f"no GeometryCache from {source} (got {[type(o).__name__ for o in objs]})"
        )
    actor = _spawn(
        unreal.GeometryCacheActor, (0, 0, 0), (0, 0, 0), f"LES_{item['name']}", "Caches"
    )
    comp = actor.get_geometry_cache_component()
    comp.set_geometry_cache(caches[0])
    comp.set_looping(False)
    comp.set_material(0, _vertex_color_material(ctx, item, emissive_default))
    ctx["actors"][item["name"]] = [actor]
    ctx.setdefault("geometry_caches", {})[item["name"]] = caches[0]
    return actor


@_stage(
    "particles (groom)",
    "import alembic/<layer>.abc as Groom with 'Import Groom Cache' ticked (Rotation 90,0,0 Scale 100,-100,100), "
    "place a Groom Actor at the origin and add a Groom Cache track",
)
def stage_particles(ctx):
    for item in ctx["plan"]["layers"]:
        if item["type"] != "particles" or PARTICLE_MODE == "none":
            continue
        if not item["exists"]:
            _warn(
                f"particles {item['name']}: {item['source']} missing (run the bundle's alembic stage), skipped"
            )
            continue
        try:
            _import_groom(ctx, item)
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"particles {item['name']}: groom import failed ({exc}); not imported. "
                f"Manual fallback: import {item['source']} by hand as Groom with 'Import Groom "
                "Cache' ticked (Rotation 90,0,0 Scale 100,-100,100), place a Groom Actor at the "
                "origin and add a Groom Cache track."
            )


@_stage(
    "isosurfaces (geometry cache)",
    "import alembic/<layer>.abc as Geometry Cache (Rotation 90,0,0 Scale 100,-100,100), "
    "place a Geometry Cache Actor at the origin and add a Geometry Cache track",
)
def stage_isosurfaces(ctx):
    for item in ctx["plan"]["layers"]:
        if item["type"] != "isosurface":
            continue
        if not item["exists"]:
            _warn(
                f"isosurface {item['name']}: {item['source']} missing (run the bundle's alembic stage), skipped"
            )
            continue
        _import_geometry_cache(ctx, item, item["source"], 0.15)


def _slice_material(ctx, item, media_tex):
    look = LOOKS[ctx["plan"]["look"]]
    mat = _fresh_material(ctx, f"M_Slice_{item['asset']}")
    _set(mat, "blend_mode", _enum("BlendMode", "BLEND_TRANSLUCENT"))
    _set(mat, "shading_model", _enum("MaterialShadingModel", "MSM_UNLIT"))
    _set(mat, "two_sided", True)
    tex = _expr(mat, "MaterialExpressionTextureSample", -700, 0)
    if media_tex is not None:
        _set(tex, "texture", media_tex)
    gain = _scalar(mat, "SliceGain", max(1.0, 0.25 * look["emissive_gain"]), -700, 250)
    opac = _scalar(mat, "SliceOpacity", 0.9, -700, 330)
    em = _binop(mat, "Multiply", tex, gain, -400, 0, a_out=("RGB", ""))
    op = _binop(mat, "Multiply", tex, opac, -400, 200, a_out=("A",))
    _to_prop(em, ("",), ("MP_EMISSIVE_COLOR",))
    _to_prop(op, ("",), ("MP_OPACITY",))
    return _finish_material(mat)


@_stage(
    "slices (Img Media)",
    "create an Img Media Source on slices/<name>/ (frame rate = fps / frame_step) + a Media Texture, "
    "put it in an Unlit Translucent two-sided material on unreal/meshes/<name>_plane.glb, and add a Media track",
)
def stage_slices(ctx):
    root = ctx["plan"]["root"]
    fps_num, fps_den = ctx["plan"]["fps"]
    for item in ctx["plan"]["layers"]:
        if item["type"] != "slice":
            continue
        if not item["exists"] or not pathlib.Path(item["plane"]).exists():
            _warn(
                f"slice {item['name']}: {item['source']} or {item['plane']} missing, skipped"
            )
            continue
        folder = f"{root}/Slices/{item['asset']}"
        ims = _load_or_create(
            f"IMS_{item['asset']}",
            folder,
            unreal.ImgMediaSource,
            unreal.ImgMediaSourceFactoryNew(),
        )
        ims.set_sequence_path(str(pathlib.Path(item["source"]).resolve()))
        step = max(int(item["layer"].get("frame_step", 1)), 1)
        _set(ims, "frame_rate_override", unreal.FrameRate(fps_num, fps_den * step))
        mtex = _load_or_create(
            f"MT_{item['asset']}",
            folder,
            unreal.MediaTexture,
            unreal.MediaTextureFactoryNew(),
        )
        _set(mtex, "auto_clear", True, quiet=True)
        _eal().save_asset(ims.get_path_name(), only_if_is_dirty=False)
        _eal().save_asset(mtex.get_path_name(), only_if_is_dirty=False)
        objs = _import(item["plane"], f"{folder}/Mesh")
        meshes = _of_class(objs, "StaticMesh")
        if not meshes:
            raise RuntimeError(f"{item['plane']}: no StaticMesh imported")
        actor = _spawn_object(meshes[0], label=f"LES_{item['name']}", folder="Slices")
        comp = actor.get_editor_property("static_mesh_component")
        comp.set_material(0, _slice_material(ctx, item, mtex))
        _set(comp, "cast_shadow", False)
        _apply_fix_to_actor(actor, ctx.get("geometry_fix"))
        ctx["actors"][item["name"]] = [actor]
        ctx.setdefault("media", {})[item["name"]] = (ims, mtex)


# --- sequence --------------------------------------------------------------------


def _channels(section):
    try:
        return list(section.get_all_channels())
    except Exception:  # noqa: BLE001  (pre-5.1 name)
        return list(section.get_channels())


def _named_channels(section, names):
    chans = _channels(section)
    by_name = {}
    for c in chans:
        try:
            by_name[str(c.get_editor_property("channel_name"))] = c
        except Exception:  # noqa: BLE001
            pass
    if all(n in by_name for n in names):
        return [by_name[n] for n in names]
    return chans[: len(names)]


def _key(channel, frame, value, interp=None):
    if interp is None:
        return channel.add_key(_frame(frame), value)
    return channel.add_key(
        _frame(frame), value, 0.0, _enum("MovieSceneTimeUnit", "DISPLAY_RATE"), interp
    )


def _property_track(binding, track_cls, name, path, start, end):
    track = binding.add_track(track_cls)
    track.set_property_name_and_path(name, path)
    section = track.add_section()
    section.set_range(start, end)
    return track, section


_TRANSFORM_CHANNELS = [
    "Location.X",
    "Location.Y",
    "Location.Z",
    "Rotation.X",
    "Rotation.Y",
    "Rotation.Z",
]


def _camera_for_shot(ctx, seq, shot, keys, sensor_h):
    start, end = int(shot["start"]), int(shot["end"]) + 1
    k0 = keys[0]
    cam = _spawn(
        unreal.CineCameraActor,
        k0["location_cm"],
        k0["rotation"],
        f"LES_Cam_{shot['name']}",
        "Cameras",
    )
    cc = cam.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    _set(fb, "sensor_width", SENSOR_WIDTH_MM)
    _set(fb, "sensor_height", sensor_h)
    cc.set_editor_property("filmback", fb)
    fs = cc.get_editor_property("focus_settings")
    _set(fs, "focus_method", _enum("CameraFocusMethod", "MANUAL"))
    _set(fs, "manual_focus_distance", k0["focus_distance_cm"])
    cc.set_editor_property("focus_settings", fs)
    _set(cc, "current_focal_length", k0["focal_length_mm"])
    _set(cc, "current_aperture", k0["fstop"])

    linear = _enum("MovieSceneKeyInterpolation", "LINEAR")
    binding = seq.add_possessable(cam)
    tr = binding.add_track(unreal.MovieScene3DTransformTrack)
    sec = tr.add_section()
    sec.set_range(start, end)
    lx, ly, lz, rx, ry, rz = _named_channels(sec, _TRANSFORM_CHANNELS)
    for k in keys:
        f = k["frame"]
        (x, y, z), (pitch, yaw, roll) = k["location_cm"], k["rotation"]
        _key(lx, f, x, linear)
        _key(ly, f, y, linear)
        _key(lz, f, z, linear)
        _key(rx, f, roll, linear)
        _key(ry, f, pitch, linear)
        _key(rz, f, yaw, linear)

    cb = seq.add_possessable(cc)
    try:
        cb.set_parent(binding)
    except Exception:  # noqa: BLE001
        pass
    for prop, path, field in (
        ("CurrentFocalLength", "CurrentFocalLength", "focal_length_mm"),
        ("CurrentAperture", "CurrentAperture", "fstop"),
        (
            "ManualFocusDistance",
            "FocusSettings.ManualFocusDistance",
            "focus_distance_cm",
        ),
    ):
        try:
            _, s = _property_track(
                cb, unreal.MovieSceneFloatTrack, prop, path, start, end
            )
            ch = _channels(s)[0]
            for k in keys:
                _key(ch, k["frame"], float(k[field]), linear)
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"camera {shot['name']}: {prop} track failed ({exc}); set it on the camera by hand"
            )
    return cam, binding


def _clear_sequence(seq):
    for b in list(seq.get_bindings()):
        try:
            b.remove()
        except Exception:  # noqa: BLE001
            pass
    for t in list(seq.get_tracks()):
        try:
            seq.remove_track(t)
        except Exception:  # noqa: BLE001
            pass


@_stage(
    "level sequence",
    "create a Level Sequence at the display rate from manifest.timeline.fps, add one CineCamera per shot "
    "(camera_bake.json has every frame), a Camera Cut track, and per-layer visibility / cache / media tracks",
)
def stage_sequence(ctx):
    plan, manifest = ctx["plan"], ctx["manifest"]
    n = plan["n_frames"]
    folder, name = plan["sequence"].rsplit("/", 1)
    seq = _load_or_create(
        name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew()
    )
    _clear_sequence(seq)
    num, den = plan["fps"]
    seq.set_display_rate(unreal.FrameRate(num, den))
    seq.set_playback_start(0)
    seq.set_playback_end(n)
    try:
        seq.set_view_range_start(0.0)
        seq.set_view_range_end(n * den / num)
        seq.set_work_range_start(0.0)
        seq.set_work_range_end(n * den / num)
    except Exception:  # noqa: BLE001
        pass
    ctx["sequence"] = seq

    # cameras + cuts
    cams = bake_camera(manifest["shots"], CAMERA_KEY_STEP, ctx.get("camera_bake"))
    w, h = plan["resolution"]
    sensor_h = SENSOR_WIDTH_MM * h / float(w)
    cut_track = seq.add_track(unreal.MovieSceneCameraCutTrack)
    for shot in sorted(manifest["shots"], key=lambda s: s["start"]):
        try:
            cam, binding = _camera_for_shot(
                ctx, seq, shot, cams[shot["name"]], sensor_h
            )
            cut = cut_track.add_section()
            cut.set_range(int(shot["start"]), int(shot["end"]) + 1)
            cut.set_camera_binding_id(seq.get_binding_id(binding))
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"shot {shot['name']}: camera failed ({exc}); add a CineCamera + camera cut by hand"
            )

    # layer visibility (bHiddenInGame: the property Movie Render Queue honours)
    vis = layer_visibility_keys(manifest)
    constant = _enum("MovieSceneKeyInterpolation", "CONSTANT")
    ctx["bindings"] = {}
    for item in plan["layers"]:
        actors = ctx["actors"].get(item["name"]) or []
        for actor in actors:
            try:
                b = seq.add_possessable(actor)
                ctx["bindings"].setdefault(item["name"], []).append(b)
                _, s = _property_track(
                    b,
                    unreal.MovieSceneVisibilityTrack,
                    "bHiddenInGame",
                    "bHiddenInGame",
                    0,
                    n,
                )
                ch = _channels(s)[0]
                for f, visible in vis.get(item["name"], [(0, True)]):
                    _key(ch, f, not visible)
            except Exception as exc:  # noqa: BLE001
                _warn(
                    f"layer {item['name']}: visibility track failed ({exc}); toggle 'Actor Hidden In Game' per shot by hand"
                )

    # heterogeneous volumes: key the SVT frame (constant steps)
    for name, comp in (ctx.get("volume_components") or {}).items():
        try:
            layer = next(i["layer"] for i in plan["layers"] if i["name"] == name)
            cb = seq.add_possessable(comp)
            parents = ctx["bindings"].get(name) or []
            if parents:
                cb.set_parent(parents[0])
            _, s = _property_track(
                cb, unreal.MovieSceneFloatTrack, "Frame", "Frame", 0, n
            )
            ch = _channels(s)[0]
            for f, idx in volume_frame_keys(layer, n):
                _key(ch, f, float(idx), constant)
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"volume {name}: Frame track failed ({exc}); instead tick Playing on the Heterogeneous Volume "
                f"with Frame Rate = fps / frame_step, or key its 'Frame' property by hand"
            )

    # geometry caches / groom caches
    for kind, track_cls, field in (
        ("geometry_caches", "MovieSceneGeometryCacheTrack", "geometry_cache_asset"),
        ("groom_caches", "MovieSceneGroomCacheTrack", "groom_cache"),
    ):
        for name, asset in (ctx.get(kind) or {}).items():
            if asset is None:
                continue
            try:
                b = (
                    ctx["bindings"].get(name)
                    or [seq.add_possessable(ctx["actors"][name][0])]
                )[0]
                track = b.add_track(getattr(unreal, track_cls))
                sec = track.add_section()
                sec.set_range(0, n)
                params = sec.get_editor_property("params")
                params.set_editor_property(field, asset)
                sec.set_editor_property("params", params)
            except Exception as exc:  # noqa: BLE001
                _warn(
                    f"{name}: {track_cls} failed ({exc}); add it on the actor in Sequencer and pick the cache"
                )

    # slices: media tracks
    for name, (ims, mtex) in (ctx.get("media") or {}).items():
        try:
            track = seq.add_track(unreal.MovieSceneMediaTrack)
            try:
                track.set_display_name(f"Slice {name}")
            except Exception:  # noqa: BLE001
                pass
            sec = track.add_section()
            sec.set_range(0, n)
            sec.set_editor_property("media_source", ims)
            sec.set_editor_property("media_texture", mtex)
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"slice {name}: media track failed ({exc}); add a Media track with IMS_{name} -> MT_{name}"
            )

    _eal().save_asset(seq.get_path_name(), only_if_is_dirty=False)


@_stage("save", "File > Save All")
def stage_save(ctx):
    level_sys = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    level_sys.save_current_level()
    _eal().save_directory(ctx["plan"]["root"], only_if_is_dirty=True, recursive=True)


@_stage(
    "movie render queue",
    "Window > Cinematics > Movie Render Queue: add LS_<case>, set Output (resolution, directory), "
    "Path Tracer (or Deferred) pass, PNG output, Anti-aliasing samples + warm-up, the console variables listed in README, then Save As preset",
)
def stage_mrq(ctx):
    plan = ctx["plan"]
    renderer = ctx["renderer"]
    q_sys = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = q_sys.get_queue()
    for job in list(queue.get_jobs()):
        if job.get_editor_property("job_name") == f"LES_{plan['case']}":
            queue.delete_job(job)
    seq = ctx.get("sequence") or _eal().load_asset(plan["sequence"])
    try:
        job = unreal.MoviePipelineEditorLibrary.create_job_from_sequence(queue, seq)
    except Exception:  # noqa: BLE001
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.set_editor_property("sequence", unreal.SoftObjectPath(seq.get_path_name()))
    job.set_editor_property("map", unreal.SoftObjectPath(_map_object_path(plan["map"])))
    job.set_editor_property("job_name", f"LES_{plan['case']}")
    cfg = job.get_configuration()

    out = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out.set_editor_property(
        "output_directory", unreal.DirectoryPath(plan["output_dir"])
    )
    out.set_editor_property("file_name_format", "{sequence_name}.{frame_number}")
    out.set_editor_property("output_resolution", unreal.IntPoint(*plan["resolution"]))
    out.set_editor_property("zero_pad_frame_numbers", 4)
    _set(out, "override_existing_output", True)
    _set(out, "use_custom_frame_rate", False)

    # render pass
    for cls_name in (
        "MoviePipelineDeferredPassBase",
        "MoviePipelineDeferredPass_PathTracer",
    ):
        for s in cfg.find_settings_by_class(getattr(unreal, cls_name), True, True):
            cfg.remove_setting(s)
    if renderer == "path_tracer":
        rp = cfg.find_or_add_setting_by_class(
            unreal.MoviePipelineDeferredPass_PathTracer
        )
        _set(rp, "reference_motion_blur", True)
    else:
        rp = cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

    # image output
    for cls_name in (
        "MoviePipelineImageSequenceOutput_JPG",
        "MoviePipelineImageSequenceOutput_BMP",
        "MoviePipelineImageSequenceOutput_PNG",
        "MoviePipelineImageSequenceOutput_EXR",
    ):
        cls = getattr(unreal, cls_name, None)
        if cls is not None:
            for s in cfg.find_settings_by_class(cls, True, True):
                cfg.remove_setting(s)
    if OUTPUT_FORMAT == "exr":
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
    else:
        png = cfg.find_or_add_setting_by_class(
            unreal.MoviePipelineImageSequenceOutput_PNG
        )
        _set(png, "write_alpha", False, quiet=True)

    # sampling / warm-up
    q = MRQ[renderer]
    aa = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_property("spatial_sample_count", q["spatial_samples"])
    aa.set_editor_property("temporal_sample_count", q["temporal_samples"])
    aa.set_editor_property("engine_warm_up_count", q["engine_warm_up"])
    aa.set_editor_property("render_warm_up_count", q["render_warm_up"])
    _set(aa, "render_warm_up_frames", q["render_warm_up"] > 0, quiet=True)
    _set(aa, "use_camera_cut_for_warm_up", False, quiet=True)
    if renderer == "path_tracer":
        _set(aa, "override_anti_aliasing", True)
        _set(
            aa,
            "anti_aliasing_method",
            _enum("AntiAliasingMethod", "AAM_NONE"),
            quiet=True,
        )

    # console variables
    cv = cfg.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    for name, value in MRQ_CVARS[renderer].items():
        try:
            cv.add_or_update_console_variable(name, float(value))
        except Exception as exc:  # noqa: BLE001
            _warn(f"cvar {name}: {exc}")

    go = cfg.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    _set(go, "cinematic_quality_settings", True, quiet=True)
    _set(go, "use_lod_zero", True, quiet=True)

    folder, name = plan["mrq_config"].rsplit("/", 1)
    _ensure_dir(folder)
    res = unreal.MoviePipelineEditorLibrary.export_config_to_asset(
        cfg, folder, name, True
    )
    asset = res[0] if isinstance(res, (tuple, list)) else res
    if asset is None:
        raise RuntimeError(f"export_config_to_asset failed: {res}")
    _log(
        f"MRQ config saved as {plan['mrq_config']} (job 'LES_{plan['case']}' is also in the editor's queue)"
    )


def _map_object_path(package_path):
    leaf = package_path.rsplit("/", 1)[1]
    return f"{package_path}.{leaf}"


def _write_scene_json(ctx):
    plan = ctx["plan"]
    info = {
        "case": plan["case"],
        "map": plan["map"],
        "sequence": _map_object_path(plan["sequence"]),
        "mrq_config": _map_object_path(plan["mrq_config"]),
        "output_dir": plan["output_dir"],
        "file_name_format": "{sequence_name}.{frame_number}",
        "resolution": list(plan["resolution"]),
        "fps": list(plan["fps"]),
        "n_frames": plan["n_frames"],
        "renderer": ctx["renderer"],
        "output_format": OUTPUT_FORMAT,
        "look": plan["look"],
        "report": [{"stage": s, "status": st, "message": m} for s, st, m in REPORT],
    }
    path = pathlib.Path(ctx["bundle"]) / "unreal" / "ue_scene.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
    return path


def _read_camera_bake(bundle):
    path = pathlib.Path(bundle) / "unreal" / "camera_bake.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return {int(s["frame"]): s for s in data.get("frames", [])}
    except Exception as exc:  # noqa: BLE001
        _warn(f"camera_bake.json unreadable ({exc}); sampling cameras in-editor")
        return None


STAGES = [
    ("level", stage_level),
    ("luts", stage_luts),
    ("geometry", stage_geometry),
    ("look", stage_look),
    ("volumes", stage_volumes),
    ("particles", stage_particles),
    ("isosurfaces", stage_isosurfaces),
    ("slices", stage_slices),
    ("sequence", stage_sequence),
    ("save", stage_save),
    ("mrq", stage_mrq),
]


def main(argv=None):
    if unreal is None:
        raise SystemExit(
            "build_scene.py must run inside the Unreal Editor (Python Editor Script Plugin)"
        )
    REPORT.clear()
    opts = parse_args(
        sys.argv[1:] if argv is None else argv, default_bundle=DEFAULT_BUNDLE
    )
    if not opts["bundle"]:
        unreal.log_error(
            f"{LOG_PREFIX} no bundle: pass --bundle <dir>, set LES_BUNDLE, or run prepare_unreal()"
        )
        return None
    bundle = pathlib.Path(opts["bundle"]).expanduser().resolve()
    manifest = load_manifest(bundle)
    plan = build_plan(manifest, bundle)
    renderer = opts["renderer"] or RENDERER
    renderer = renderer if renderer in MRQ else "path_tracer"
    try:
        version = unreal.SystemLibrary.get_engine_version()
    except Exception:  # noqa: BLE001
        version = "?"
    _log(
        f"bundle {bundle} case {plan['case']} look {plan['look']} renderer {renderer} engine {version}"
    )
    ctx = {
        "bundle": str(bundle),
        "manifest": manifest,
        "plan": plan,
        "renderer": renderer,
        "actors": {},
        "camera_bake": _read_camera_bake(bundle),
        "engine": parse_engine_version(version),
    }
    for key, fn in STAGES:
        if key in opts["skip"]:
            _log(f"skipping stage {key}")
            continue
        fn(ctx)
    path = _write_scene_json(ctx)
    failed = [r for r in REPORT if r[1] != "ok"]
    _log(
        f"done: {len(REPORT) - len(failed)} stages ok, {len(failed)} failed; summary in {path}"
    )
    for stage, _, msg in failed:
        _warn(f"  {stage}: {msg}")
    if opts["quit"]:
        unreal.SystemLibrary.quit_editor()
    return ctx


if __name__ == "__main__" and unreal is not None:
    main()
