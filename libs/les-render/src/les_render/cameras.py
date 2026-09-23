"""Cinematic camera rigs for a render bundle.

Builds the ``shots`` block of the manifest (see ``docs/les_render.md#shots``):
a short sequence of camera moves, expressed as sparse world-space keys, that
tile the render timeline with no gaps or overlaps. Both renderers (Blender
and Unreal) sample the *same* keys through :func:`sample_camera`, so the
preview and the production render match.

Everything is scaled from the geometry manifest block (``buildings_bounds``,
``max_building_height``, ``footprints``) and the domain block, never from
fixed metre constants, so the same spec produces a sane rig on a 50 m test
case and a 500 m production one.

Coordinate frame: world space, metres, right-handed, z-up (matches the
bundle's simulation frame -- see ``docs/les_render.md``). The flow direction
used to orient shots is ``spec["flow_axis"]`` (default ``"x"``, i.e.
``inflow_angle=0`` blows toward ``+x``; see
``libs/pyudales/src/pyudales/utils/inflow_utils.py:77-97``:
``u = speed*cos(angle_deg), v = speed*sin(angle_deg)``, so angle is measured
from ``+x``, counter-clockwise).

## Spec keys (all optional; ``DEFAULT_SPEC`` documents every default)

``shots``: ``[{"type": "establishing"|"plan"|"street"|"wake", "fraction": f}, ...]``
    Overrides the default 5-shot sequence and/or its fractions of the
    timeline. Types repeat (e.g. two ``"establishing"`` entries bookend the
    sequence for a loopable cut). Fractions need not sum to 1; they are
    renormalised.
``layers``: ``{shot_type: [layer_name, ...]}``
    Visible-layer suggestion per shot type. Falls back to
    ``DEFAULT_LAYERS[type]``; unknown types get no ``"layers"`` key (all
    layers visible, per the manifest contract).
``layer_names``: ``{"streaklines": "...", "trails": "...", "speed_glow": "...",
    "vortices": "...", "ground_speed": "..."}``
    Renames the canonical layer names used in ``layers`` above (e.g. if a
    scene's volume layer is actually called ``"fog"``).
``flow_axis``: ``"x"`` (default) or ``"y"`` -- horizontal axis the inflow
    blows along at ``inflow_angle=0``.
``margin``: float metres (default: ``max(2.0, 0.15 * max_building_height)``)
    -- minimum clearance kept between any camera path and a building.
``n_keys``: ``{"establishing": 3, "plan": 3, "street": 5, "wake": 6}``
``focal_length_mm``: ``{"establishing": 24, "plan": 35, "street": 28, "wake": 35}``
``fstop``: float (default 5.6), or ``{shot_type: fstop}``.
``orbit_span_deg``: float (default 100) -- azimuth swept by the establishing
    and wake orbits.

## Shot types

* ``establishing`` -- high oblique orbit + slow dolly-in over the whole
  building array, looking down and slightly downstream.
* ``plan`` -- near top-down plan view, slow reveal from oblique to near-nadir.
* ``street`` -- low, street-level dolly along a street canyon: a gap between
  building rows found from ``footprints`` and running parallel to
  ``flow_axis``, with a safety margin.
* ``wake`` -- elevated orbit (``z ~ 1.75 H``) looking down-and-forward into
  the recirculation zone behind the tallest building, pulled back enough
  that the building reads as a building rather than a close-up wall.

``make_shots`` assembles the default 5-shot sequence
(``establishing, plan, street, wake, establishing``), tiling ``[0,
timeline.n_frames - 1]`` exactly. The final ``establishing`` shot's keys are
the first establishing shot's keys, reversed, so its last key equals the
first shot's first key -- the render loops cleanly.

``single_shot(kind, ...)`` builds one of the four shot types spanning a given
frame range (the full timeline by default), for previewing or shipping a
single camera.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]


def _vec3(v: Any) -> Vec3:
    x, y, z = v
    return (float(x), float(y), float(z))


# -- defaults ------------------------------------------------------------

DEFAULT_LAYERS: dict[str, list[str]] = {
    "establishing": ["speed_glow", "streaklines", "ground_speed"],
    "plan": ["ground_speed", "trails"],
    "street": ["streaklines", "trails", "ground_speed"],
    "wake": ["vortices", "speed_glow"],
}

DEFAULT_SPEC: dict[str, Any] = {
    "shots": [
        {"type": "establishing", "fraction": 0.20},
        {"type": "plan", "fraction": 0.15},
        {"type": "street", "fraction": 0.25},
        {"type": "wake", "fraction": 0.30},
        {"type": "establishing", "fraction": 0.10},
    ],
    "layers": DEFAULT_LAYERS,
    "layer_names": {
        "streaklines": "streaklines",
        "trails": "trails",
        "speed_glow": "speed_glow",
        "vortices": "vortices",
        "ground_speed": "ground_speed",
    },
    "flow_axis": "x",
    "margin": None,  # None -> derived from max_building_height, see _margin()
    "n_keys": {"establishing": 3, "plan": 3, "street": 5, "wake": 6},
    "focal_length_mm": {
        "establishing": 24.0,
        "plan": 35.0,
        "street": 28.0,
        "wake": 35.0,
    },
    "fstop": 5.6,
    "orbit_span_deg": 80.0,
}

_SHOT_NAMES = {
    "establishing": "establishing",
    "plan": "plan",
    "street": "street",
    "wake": "wake",
}


def _merge_spec(spec: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_SPEC)
    for k, v in (spec or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


# -- small vector/spline helpers ------------------------------------------


def _smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    uc = np.clip(u, 0.0, 1.0)
    return uc * uc * (3.0 - 2.0 * uc)  # type: ignore[no-any-return, unused-ignore]


def _catmull_rom(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float
) -> np.ndarray:
    """Uniform Catmull-Rom (tension 0.5) at ``u`` in [0, 1] between ``p1`` and ``p2``."""
    u2 = u * u
    u3 = u2 * u
    return np.asarray(
        0.5
        * (
            (2.0 * p1)
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
        )
    )


# -- geometry helpers -------------------------------------------------------


def _flow_dir(flow_axis: str) -> np.ndarray:
    """Unit vector the flow blows toward, in the (x, y) plane, at inflow_angle=0."""
    return np.array([1.0, 0.0]) if flow_axis == "x" else np.array([0.0, 1.0])


def _cluster_bbox(geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(geometry["buildings_bounds"], dtype=float)
    return b[0], b[1]


def _margin(geometry: dict, spec: dict) -> float:
    if spec.get("margin") is not None:
        return float(spec["margin"])
    h = float(geometry.get("max_building_height", 10.0)) or 10.0
    return max(2.0, 0.15 * h)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    if not intervals:
        return []
    ivs = sorted(intervals)
    merged = [list(ivs[0])]
    for lo, hi in ivs[1:]:
        if lo <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _find_street_canyon(
    geometry: dict, flow_axis: str, margin: float
) -> Optional[dict]:
    """Widest gap between building rows, running parallel to ``flow_axis``.

    Returns ``{"cross": centre, "axis_range": (lo, hi), "height": local_h,
    "half_width": w}`` in the cross-flow axis, or ``None`` if the footprints
    leave no such corridor.
    """
    footprints = geometry.get("footprints") or []
    if not footprints:
        return None
    axis_i = 0 if flow_axis == "x" else 1
    cross_i = 1 - axis_i

    intervals = [(fp["min"][cross_i], fp["max"][cross_i]) for fp in footprints]
    merged = _merge_intervals(intervals)
    if len(merged) < 2:
        return None

    gaps = []
    for a, b in zip(merged[:-1], merged[1:]):
        gap_lo, gap_hi = a[1], b[0]
        width = gap_hi - gap_lo
        if width > 2.0 * margin:
            gaps.append((gap_lo, gap_hi, width))
    if not gaps:
        return None
    gap_lo, gap_hi, width = max(gaps, key=lambda g: g[2])
    centre = 0.5 * (gap_lo + gap_hi)

    adjacent_h = [
        fp["max"][2]
        for fp in footprints
        if abs(fp["max"][cross_i] - gap_lo) < 1e-3
        or abs(fp["min"][cross_i] - gap_hi) < 1e-3
    ]
    height = (
        float(np.mean(adjacent_h))
        if adjacent_h
        else float(geometry.get("max_building_height", 10.0))
    )

    axis_lo = min(fp["min"][axis_i] for fp in footprints)
    axis_hi = max(fp["max"][axis_i] for fp in footprints)
    return {
        "cross": centre,
        "axis_range": (axis_lo, axis_hi),
        "height": height,
        "half_width": 0.5 * width,
    }


def _tallest_footprint(geometry: dict) -> Optional[dict]:
    footprints: list[dict] = geometry.get("footprints") or []
    if not footprints:
        return None
    return max(footprints, key=lambda fp: fp["max"][2] - fp["min"][2])


def point_in_building(
    point: Sequence[float], footprints: list[dict], margin: float = 0.0
) -> bool:
    """True if ``point`` sits inside (or within ``margin`` of) any footprint's
    horizontal extent AND below its roof -- i.e. would clip through the
    building."""
    x, y, z = point
    for fp in footprints:
        lo, hi = fp["min"], fp["max"]
        if (
            lo[0] - margin <= x <= hi[0] + margin
            and lo[1] - margin <= y <= hi[1] + margin
            and z <= hi[2] + margin
        ):
            return True
    return False


def segment_hits_building(
    p0: Sequence[float],
    p1: Sequence[float],
    footprints: list[dict],
    margin: float = 0.0,
    n: int = 16,
) -> bool:
    """Sample the segment ``p0 -> p1`` and check each sample with :func:`point_in_building`."""
    p0a, p1a = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    for t in np.linspace(0.0, 1.0, n):
        if point_in_building(p0a + t * (p1a - p0a), footprints, margin):
            return True
    return False


# -- per-shot key builders --------------------------------------------------


def _key(
    frame: int, location: np.ndarray, target: np.ndarray, focal: float, fstop: float
) -> dict:
    return {
        "frame": int(frame),
        "location": [round(float(v), 4) for v in location],
        "target": [round(float(v), 4) for v in target],
        "focal_length_mm": float(focal),
        "fstop": float(fstop),
    }


def _establishing_keys(
    geometry: dict,
    domain: dict,
    frames: Sequence[int],
    spec: dict,
) -> list[dict]:
    lo, hi = _cluster_bbox(geometry)
    centre = 0.5 * (lo + hi)
    size = hi - lo
    H = max(float(geometry.get("max_building_height", 10.0)), 1.0)
    flow = _flow_dir(spec["flow_axis"])
    cross = np.array([-flow[1], flow[0]])

    radius0 = 0.6 * float(np.hypot(size[0], size[1])) + _margin(geometry, spec)
    z_cam = min(2.3 * H, float(domain["upper"][2]) * 3.0 + H)
    span = np.deg2rad(spec["orbit_span_deg"])
    n = len(frames)
    thetas = np.linspace(-0.5 * span, 0.5 * span, n)
    radii = np.linspace(1.1 * radius0, 0.75 * radius0, n)
    # target biased slightly downstream of the cluster centre ("looking downstream-ish")
    target_xy = centre[:2] + 0.15 * size[:2] * flow
    focal = spec["focal_length_mm"]["establishing"]
    fstop = (
        spec["fstop"]
        if not isinstance(spec["fstop"], dict)
        else spec["fstop"]["establishing"]
    )

    keys = []
    order = range(n)
    for idx in order:
        theta, r = thetas[idx], radii[idx]
        offset_xy = -flow * r * np.cos(theta) + cross * r * np.sin(theta)
        loc = np.array([centre[0] + offset_xy[0], centre[1] + offset_xy[1], z_cam])
        target = np.array([target_xy[0], target_xy[1], 0.35 * H])
        keys.append(_key(frames[idx], loc, target, focal, fstop))
    return keys


def _plan_keys(
    geometry: dict, domain: dict, frames: Sequence[int], spec: dict
) -> list[dict]:
    lo, hi = _cluster_bbox(geometry)
    centre = 0.5 * (lo + hi)
    size = hi - lo
    H = max(float(geometry.get("max_building_height", 10.0)), 1.0)
    diag = float(np.hypot(size[0], size[1]))
    # A drone-height plan view: high enough to read the ground-level slice
    # over most of the cluster without floating unrealistically far above it.
    z_cam = max(4.5 * H, 0.35 * diag)
    flow = _flow_dir(spec["flow_axis"])

    n = max(len(frames), 2)
    # slow reveal: start slightly oblique (so the audience reads it as a plan
    # view, not a glitch), settle to near-straight-down.
    elev = np.deg2rad(np.linspace(70.0, 87.0, n))
    focal = spec["focal_length_mm"]["plan"]
    fstop = (
        spec["fstop"] if not isinstance(spec["fstop"], dict) else spec["fstop"]["plan"]
    )

    keys = []
    for idx in range(len(frames)):
        e = elev[idx]
        horiz = z_cam / max(np.tan(e), 1e-3)
        loc = np.array(
            [centre[0] - horiz * flow[0], centre[1] - horiz * flow[1], z_cam]
        )
        target = np.array([centre[0], centre[1], 0.0])
        keys.append(_key(frames[idx], loc, target, focal, fstop))
    return keys


def _street_keys(
    geometry: dict, domain: dict, frames: Sequence[int], spec: dict
) -> list[dict]:
    margin = _margin(geometry, spec)
    canyon = _find_street_canyon(geometry, spec["flow_axis"], margin)
    footprints = geometry.get("footprints") or []
    axis_i = 0 if spec["flow_axis"] == "x" else 1
    cross_i = 1 - axis_i
    flow = _flow_dir(spec["flow_axis"])
    focal = spec["focal_length_mm"]["street"]
    fstop = (
        spec["fstop"]
        if not isinstance(spec["fstop"], dict)
        else spec["fstop"]["street"]
    )

    lo, hi = _cluster_bbox(geometry)
    if canyon is None:
        # No canyon found (e.g. a single isolated building, or buildings
        # packed solid across the cross-flow axis so no gap qualifies): fly a
        # low pass just outside the cluster's cross-flow bounding box --
        # beside the array, at pedestrian height -- rather than along its
        # centreline, which can run straight through a packed row of
        # buildings.
        H = max(float(geometry.get("max_building_height", 10.0)), 1.0)
        axis_lo, axis_hi = lo[axis_i] - 0.3 * margin, hi[axis_i] + 0.3 * margin
        z_cam = min(3.0, 0.4 * H)
        cross_pos = float(hi[cross_i]) + margin
    else:
        cross_pos = canyon["cross"]
        axis_lo, axis_hi = canyon["axis_range"]
        pad = 0.2 * (axis_hi - axis_lo)
        axis_lo, axis_hi = axis_lo - pad, axis_hi + pad
        H = max(canyon["height"], 1.0)
        z_cam = float(np.clip(0.4 * H, 1.5, max(1.5, 0.4 * H)))

    axis_lo = max(axis_lo, float(domain["lower"][axis_i]) + margin)
    axis_hi = min(axis_hi, float(domain["upper"][axis_i]) - margin)

    n = max(len(frames), 2)
    axis_vals = np.linspace(axis_lo, axis_hi, n)
    look_ahead = (
        0.12 * (axis_hi - axis_lo) * (1.0 if flow[axis_i] >= 0 or axis_i == 1 else -1.0)
    )

    def _build(cross_pos: float, z_cam: float) -> list[dict]:
        keys = []
        for idx in range(len(frames)):
            pos = np.zeros(3)
            pos[axis_i] = axis_vals[idx]
            pos[cross_i] = cross_pos
            pos[2] = z_cam
            tgt = pos.copy()
            tgt[axis_i] = min(axis_vals[idx] + look_ahead, axis_hi)
            tgt[2] = 0.5 * z_cam
            keys.append(_key(frames[idx], pos, tgt, focal, fstop))
        return keys

    keys = _build(cross_pos, z_cam)

    # Safety net: verify the path between consecutive keys never clips a
    # footprint; widen the cross clearance (by a full margin each time, not a
    # token nudge) if it does.
    for _ in range(6):
        bad = any(
            segment_hits_building(
                keys[i]["location"], keys[i + 1]["location"], footprints, margin=0.0
            )
            for i in range(len(keys) - 1)
        )
        if not bad:
            break
        sign = np.sign(
            cross_pos - np.mean([f["min"][cross_i] for f in footprints] or [0.0])
        )
        cross_pos += (sign or 1.0) * margin
        keys = _build(cross_pos, z_cam)

    # Widening can still fail to clear every footprint (e.g. a very deep
    # cluster hemming in the domain); if so, don't fly through a building --
    # raise the camera above the buildings instead, and say so loudly.
    if any(
        segment_hits_building(keys[i]["location"], keys[i + 1]["location"], footprints)
        for i in range(len(keys) - 1)
    ) or any(point_in_building(k["location"], footprints) for k in keys):
        max_h = float(geometry.get("max_building_height", 10.0))
        safe_z = max_h + margin
        log.warning(
            "street shot: could not clear every footprint by widening clearance; "
            "raising the camera above the buildings (z=%.2f)",
            safe_z,
        )
        keys = _build(cross_pos, safe_z)
    return keys


def _wake_keys(
    geometry: dict, domain: dict, frames: Sequence[int], spec: dict
) -> list[dict]:
    fp = _tallest_footprint(geometry)
    margin = _margin(geometry, spec)
    flow = _flow_dir(spec["flow_axis"])
    focal = spec["focal_length_mm"]["wake"]
    fstop = (
        spec["fstop"] if not isinstance(spec["fstop"], dict) else spec["fstop"]["wake"]
    )

    if fp is None:
        lo, hi = _cluster_bbox(geometry)
        centre3 = 0.5 * (lo + hi)
        H = max(float(geometry.get("max_building_height", 10.0)), 1.0)
        size_xy = (hi - lo)[:2]
    else:
        lo, hi = np.asarray(fp["min"]), np.asarray(fp["max"])
        centre3 = 0.5 * (lo + hi)
        H = max(hi[2] - lo[2], 1.0)
        size_xy = (hi - lo)[:2]

    # Orbit target: the recirculation core in the building's wake, a
    # building-height or so downstream of its actual downstream face (not
    # the footprint's diagonal half-width, which overshoots for anything
    # non-square and put the old target almost on the wall), at a modest
    # height so the camera above it looks down-and-forward into the wake.
    axis_i = 0 if spec["flow_axis"] == "x" else 1
    half_face = 0.5 * float(size_xy[axis_i])  # centre -> downstream face, along flow
    footprint_half = 0.5 * float(
        np.hypot(size_xy[0], size_xy[1])
    )  # for framing distance
    wake_offset = half_face + 0.8 * H
    target = np.array(
        [
            centre3[0] + flow[0] * wake_offset,
            centre3[1] + flow[1] * wake_offset,
            0.45 * H,
        ]
    )

    # Pulled back ~2.3x and raised to ~1.75 H (per render feedback: the
    # previous framing was close enough to read as a close-up of the
    # leeward wall, with isosurfaces right against the lens).
    radius = 2.5 * footprint_half + 2.0 * margin
    z_cam = 1.75 * H
    n = len(frames)
    span = np.deg2rad(min(spec["orbit_span_deg"], 140.0))
    thetas = np.linspace(-0.5 * span, 0.5 * span, n)
    cross = np.array([-flow[1], flow[0]])

    keys = []
    for idx in range(n):
        theta = thetas[idx]
        offset_xy = flow * radius * np.cos(theta) + cross * radius * np.sin(theta)
        loc = np.array([target[0] + offset_xy[0], target[1] + offset_xy[1], z_cam])
        keys.append(_key(frames[idx], loc, target, focal, fstop))
    return keys


_BUILDERS = {
    "plan": _plan_keys,
    "street": _street_keys,
    "wake": _wake_keys,
}


def _build_keys(
    kind: str,
    geometry: dict,
    domain: dict,
    frames: Sequence[int],
    spec: dict,
) -> list[dict]:
    if kind == "establishing":
        return _establishing_keys(geometry, domain, frames, spec)
    if kind not in _BUILDERS:
        raise ValueError(
            f"unknown shot kind {kind!r}; choose from establishing, plan, street, wake"
        )
    return _BUILDERS[kind](geometry, domain, frames, spec)


def _frame_samples(start: int, end: int, n_keys: int) -> list[int]:
    n_keys = max(2, min(n_keys, end - start + 1))
    return [int(round(v)) for v in np.linspace(start, end, n_keys)]


def _layers_for(kind: str, spec: dict) -> Optional[list[str]]:
    layers = spec["layers"].get(kind)
    if layers is None:
        return None
    names = spec["layer_names"]
    return [names.get(n, n) for n in layers]


# -- public API ---------------------------------------------------------


def _tile_frames(fractions: Sequence[float], n_frames: int) -> list[tuple[int, int]]:
    fr = np.asarray(fractions, dtype=float)
    if fr.sum() <= 0:
        raise ValueError("shot fractions must sum to a positive number")
    fr = fr / fr.sum()
    bounds = np.round(np.cumsum(fr) * n_frames).astype(int)
    bounds = np.maximum.accumulate(bounds)
    bounds = np.clip(bounds, 1, n_frames)
    bounds[-1] = n_frames
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = min(bounds[i - 1] + 1, n_frames)
    # re-clamp from the right in case the forward pass ran past n_frames
    for i in range(len(bounds) - 2, -1, -1):
        if bounds[i] >= bounds[i + 1]:
            bounds[i] = bounds[i + 1] - 1
    bounds[0] = max(bounds[0], 1)
    starts = [0] + bounds[:-1].tolist()
    ends = [b - 1 for b in bounds]
    return list(zip(starts, ends))


def make_shots(
    geometry: dict, domain: dict, timeline: "Any", spec: Optional[dict] = None
) -> list[dict]:
    """Build the default cinematic shot sequence tiling ``[0, n_frames - 1]``.

    ``geometry`` and ``domain`` are the manifest blocks of the same name;
    ``timeline`` is a :class:`les_render.timeline.Timeline` (only
    ``n_frames`` is used). See the module docstring for ``spec`` keys.
    """
    spec = _merge_spec(spec)
    n_frames = int(timeline.n_frames)
    shot_specs = spec["shots"]
    fractions = [s.get("fraction", 1.0 / len(shot_specs)) for s in shot_specs]
    ranges = _tile_frames(fractions, n_frames)

    shots: list[dict] = []
    first_establishing_keys: Optional[list[dict]] = None
    for i, (s_spec, (start, end)) in enumerate(zip(shot_specs, ranges)):
        kind = s_spec["type"]
        n_keys = spec["n_keys"].get(kind, 3)
        frames = _frame_samples(start, end, n_keys)
        is_last = i == len(shot_specs) - 1
        if kind == "establishing" and is_last and first_establishing_keys is not None:
            # Loopable end: replay the first establishing shot's keys in
            # reverse, remapped onto this shot's frame range, so this shot's
            # last key exactly equals the first shot's first key.
            src = list(reversed(first_establishing_keys))
            keys = []
            for f, k in zip(
                frames,
                (
                    src[: len(frames)]
                    if len(src) >= len(frames)
                    else src + [src[-1]] * (len(frames) - len(src))
                ),
            ):
                nk = dict(k)
                nk["frame"] = int(f)
                keys.append(nk)
        else:
            keys = _build_keys(kind, geometry, domain, frames, spec)
            if kind == "establishing" and first_establishing_keys is None:
                first_establishing_keys = keys

        shot = {
            "name": (
                f"{_SHOT_NAMES.get(kind, kind)}_{i}"
                if sum(1 for s in shot_specs if s["type"] == kind) > 1
                else _SHOT_NAMES.get(kind, kind)
            ),
            "start": int(start),
            "end": int(end),
            "keys": keys,
        }
        layers = s_spec.get("layers") or _layers_for(kind, spec)
        if layers is not None:
            shot["layers"] = layers
        shots.append(shot)
    return shots


def single_shot(
    kind: str,
    geometry: dict,
    domain: dict,
    timeline: "Any",
    spec: Optional[dict] = None,
    frame_range: Optional[tuple[int, int]] = None,
) -> dict:
    """A single named shot (``"orbit"``, ``"plan"``, ``"street"``, ``"wake"``,
    ``"establishing"``) spanning ``frame_range`` (default: the whole
    timeline). ``"orbit"`` is an alias for ``"establishing"``."""
    spec = _merge_spec(spec)
    n_frames = int(timeline.n_frames)
    start, end = frame_range if frame_range is not None else (0, n_frames - 1)
    resolved = "establishing" if kind == "orbit" else kind
    n_keys = spec["n_keys"].get(resolved, 4)
    frames = _frame_samples(start, end, n_keys)
    keys = _build_keys(resolved, geometry, domain, frames, spec)
    shot = {"name": kind, "start": int(start), "end": int(end), "keys": keys}
    layers = _layers_for(resolved, spec)
    if layers is not None:
        shot["layers"] = layers
    return shot


def sample_camera(shots: list[dict], frame: int) -> tuple[Vec3, Vec3, float]:
    """Sample ``(location, target, focal_length_mm)`` at video ``frame``.

    Renderers use this (or an equivalent implementation) so the Blender
    preview and the Unreal render match exactly. Algorithm:

    1. Find the shot whose ``[start, end]`` contains ``frame`` (clamped to
       the sequence's overall range).
    2. Ease *once per shot*, not once per key segment: ``u = smoothstep((frame
       - start) / (end - start))`` (``u -> 3u^2 - 2u^3``), then map ``u`` back
       into frame-space, ``f' = start + u * (end - start)``. Easing per
       segment would decelerate to a near-stop at every interior key; easing
       once over the whole shot keeps interior keys at speed and only eases
       in/out at the shot's own start and end.
    3. Locate the key segment ``[keys[i], keys[i+1]]`` bracketing ``f'``
       (clamped to the first/last key outside the shot's key range), and the
       local, already-eased fraction of ``f'`` across it.
    4. Evaluate a uniform Catmull-Rom spline (tension 0.5) through
       ``keys[i-1..i+2]`` (end keys duplicated) at that fraction for
       ``location`` and ``target`` independently, component-wise.
    5. ``focal_length_mm`` is linearly interpolated on the same fraction (no
       overshoot wanted for a lens property). This function only returns
       ``(location, target, focal_length_mm)``; renderers (``blender/bundle.py``,
       ``unreal/build_scene.py``) interpolate ``fstop`` the same way from the
       same fraction.
    """
    if not shots:
        raise ValueError("no shots to sample")
    frame = int(np.clip(frame, shots[0]["start"], shots[-1]["end"]))
    shot = next((s for s in shots if s["start"] <= frame <= s["end"]), shots[-1])
    keys = shot["keys"]
    if len(keys) == 1:
        k = keys[0]
        return _vec3(k["location"]), _vec3(k["target"]), float(k["focal_length_mm"])

    kf = [k["frame"] for k in keys]
    start, end = shot["start"], shot["end"]
    u_global = 0.0 if end <= start else (frame - start) / (end - start)
    f_prime = start + float(_smoothstep(u_global)) * (end - start)

    if f_prime <= kf[0]:
        k = keys[0]
        return _vec3(k["location"]), _vec3(k["target"]), float(k["focal_length_mm"])
    if f_prime >= kf[-1]:
        k = keys[-1]
        return _vec3(k["location"]), _vec3(k["target"]), float(k["focal_length_mm"])

    i = int(np.searchsorted(kf, f_prime, side="right") - 1)
    i = min(max(i, 0), len(keys) - 2)
    t0, t1 = kf[i], kf[i + 1]
    u = 0.0 if t1 == t0 else (f_prime - t0) / (t1 - t0)

    p_im1 = (
        np.array(keys[i - 1]["location"])
        if i - 1 >= 0
        else np.array(keys[i]["location"])
    )
    p_i = np.array(keys[i]["location"])
    p_ip1 = np.array(keys[i + 1]["location"])
    p_ip2 = (
        np.array(keys[i + 2]["location"])
        if i + 2 < len(keys)
        else np.array(keys[i + 1]["location"])
    )
    location = _catmull_rom(p_im1, p_i, p_ip1, p_ip2, u)

    t_im1 = (
        np.array(keys[i - 1]["target"]) if i - 1 >= 0 else np.array(keys[i]["target"])
    )
    t_i = np.array(keys[i]["target"])
    t_ip1 = np.array(keys[i + 1]["target"])
    t_ip2 = (
        np.array(keys[i + 2]["target"])
        if i + 2 < len(keys)
        else np.array(keys[i + 1]["target"])
    )
    target = _catmull_rom(t_im1, t_i, t_ip1, t_ip2, u)

    focal = (1.0 - u) * keys[i]["focal_length_mm"] + u * keys[i + 1]["focal_length_mm"]

    return (_vec3(location), _vec3(target), float(focal))


__all__ = [
    "DEFAULT_SPEC",
    "DEFAULT_LAYERS",
    "make_shots",
    "single_shot",
    "sample_camera",
    "point_in_building",
    "segment_hits_building",
]
