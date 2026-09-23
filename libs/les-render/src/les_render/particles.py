"""Particle layers: streaklines (dye rakes) and pathline trails (comets).

The flow is unsteady, so this module never draws steady streamlines. It
integrates massless tracers through the time-interpolated LES velocity and
writes two kinds of polyline caches:

``streaklines``
    Smoke-wire / dye visualisation. Fixed emitters (``seeding``) release a
    tracer every ``release_interval`` sim seconds. The streakline of an emitter
    is the chain of its ``points_per_line`` most recent releases, newest first
    (index 0 = head, at the emitter). Note that point index ``k`` is "the
    ``k``-th newest release", *not* a material particle: between files every
    index moves to a younger particle, so renderers must not derive motion
    blur / velocity from index correspondence for this kind.
``trails``
    Pathline comets. ``counts`` tracers seeded at random in the fluid volume
    (weighted toward the ground by ``z_bias``); each is drawn as its last
    ``points_per_line`` positions, sampled ``samples_per_frame`` times per
    video frame (index 0 = current position). Line ``n`` is one tracer until it
    is recycled (death or ``max_age``), so line identity is stable apart from
    fade-out -> fade-in rebirths.

Output (docs/les_render.md, ``type: particles``):
``particles/<name>/<name>.<FFFF>.npz`` for every video frame (``frame_step``
1), holding ``points (n_lines, points_per_line, 3) float32`` (world metres),
``speed (n_lines, points_per_line) float16`` (m/s) and ``alpha`` float16 in
[0, 1]. Topology is constant over all files.

**Segment rule (renderers must follow it).** Segment ``k -> k+1`` of a line is
drawn with opacity ``min(alpha[k], alpha[k+1])``, i.e. only when *both*
endpoints have ``alpha > 0``. The exporter additionally guarantees that
    * every invalid segment (a dead / unborn endpoint, a segment longer than
      ``max_segment_length``, or one crossing a solid cell) has alpha 0 at
      *both* ends, so even a renderer that linearly interpolates per-point
      alpha (e.g. a Blender curve attribute) never shows a jump;
    * dead and unborn points are collapsed onto the nearest live point of the
      same line, so hidden geometry is zero-length (no stray bounding boxes);
    * no point with ``alpha > 0`` lies in a solid cell or outside the domain
      (trails fading out after leaving the domain are the one exception: they
      keep moving outward with ``alpha`` -> 0 over ``fade_frames``).
Scaling the tube radius by ``alpha`` is a good belt-and-braces choice in
engines that cannot do per-segment opacity.

Physics
-------
* Velocity: trilinear in space on the solid-masked grid (optionally on a
  ``upsample``-x tricubic refinement, :func:`fields.upsample_vector`), linear
  in time between snapshots; the same model as :meth:`FieldSeries.sample_velocity`
  but with a cache-friendly padded-array gather (:class:`VelocitySampler`).
* Integration: explicit RK (``integrator``: ``"rk2"`` midpoint, default, or
  ``"rk4"``). RK2 is the default: the trilinear velocity is only C0 across
  cell faces, so the O(h^2) interpolation error caps accuracy anyway, and RK4
  costs 2x the velocity samples. Measured on pyudales_idealized (4 m cells,
  3.5k tracers, vs. RK4 at cfl 0.05): after 20 s, RK2 at cfl 0.5 is off by
  median 1.5 mm / p99 3 cm (RK4 at cfl 0.5: 0.5 mm / 1.2 cm). Sub-steps are
  CFL-limited: no particle moves more than ``cfl`` native cells per sub-step
  (``substeps`` fixes the count instead).
* Walls: tracers entering a solid cell die (streaklines: hidden; trails:
  freeze and fade out). The ground (z = domain bottom) reflects; periodic
  horizontal boundaries (``periodic``) wrap. Leaving the domain otherwise
  kills a streak tracer; a trail fades out while it drifts away.
  Streak tracers are recycled by the ring buffer after
  ``points_per_line * release_interval`` s, trails after ``max_age`` or after
  ``stagnant_time`` s below ``stagnant_speed`` (near-wall tracers never stick).
* ``preroll`` sim seconds are integrated before the first frame so chains are
  full and trails are in steady state at frame 0. Before the first stored
  snapshot the field is frozen at that snapshot (FieldSeries clamps).

Spec keys (``spec`` dict; defaults from :func:`default_particle_specs`)
----------------------------------------------------------------------
Common:

========================= ====================================================
``name``                  layer / folder name (required)
``kind``                  ``"streaklines"`` | ``"trails"``
``seeding``               streaklines: ``"inlet_rake"`` (default),
                          ``"ground_line"``, ``"building_corners"``,
                          ``"custom"`` or a list of these; trails: ``"volume"``
``counts``                streaklines: dict ``rake_y`` (24), ``rake_z`` (8),
                          ``ground_line`` (48); trails: int or
                          ``{"particles": n}`` (20000)
``points_per_line``       points per polyline (streak 256, trails 24)
``radius``                tube radius in metres (manifest only)
``colormap``              matplotlib name (``"inferno"``)
``color_variable``        ``"speed"`` (the only supported value)
``range``                 ``[vmin, vmax]`` m/s or null = robust auto-range
``emission_strength``     manifest only (renderer emissive multiplier)
``preroll``               sim seconds before ``t_start``; null = domain
                          transit time ``length_x / mean inflow u``
``substeps``              fixed sub-steps per sample interval; null = CFL
``cfl``                   max native cells per sub-step (0.5)
``integrator``            ``"rk2"`` | ``"rk4"``
``upsample``              1 (trilinear on native grid) or 2 (tricubic 2x;
                          smoother paths, ~1.3 s extra per stored snapshot)
``max_segment_length``    m; longer segments are broken (null = 3 cells)
``periodic``              ``[px, py]`` periodic horizontal boundaries (tracers
                          wrap around; the wrap segment is broken); null =
                          auto-detect from the data (:func:`detect_periodic`)
``seed``                  RNG seed (determinism)
``threads``               threads for velocity sampling (1; the generic
                          ``workers`` key is ignored: on a DRAM-bound box
                          extra threads did not help, measure before raising)
========================= ====================================================

Streaklines only: ``line_duration`` (sim s a chain spans; null = transit
time), ``release_interval`` (null = ``line_duration / points_per_line``;
preroll is raised to at least one full chain), ``tail_fade`` (fraction of the
chain whose alpha ramps to 0 at the old end, 0.15), ``rake_x_offset_cells``
(1.5), ``rake_z_min`` (1.5 m), ``rake_z_max`` (null = 1.2 x max building
height), ``rake_z_growth`` (geometric z-spacing ratio, 1.3), ``ground_line_z``
(2 m), ``ground_line_offset`` (m upstream of the first building; null = 2
cells), ``corner_heights`` (fractions of building height, [0.2, 0.5, 0.85]),
``corner_offset`` (m upstream/outward of the corner; null = 0.75 cell),
``footprints`` (list of ``{"min", "max"}`` boxes as in the manifest; null =
connected components of ``blanking``), ``emitters`` (``[[x, y, z], ...]`` for
``"custom"``).

Trails only: ``samples_per_frame`` (2), ``max_age`` (sim s; null = 0.4 x
transit; each tracer draws its life from [0.5, 1] x max_age), ``fade_frames``
(video frames of fade in / out, 6), ``taper`` (alpha exponent along the
trail, 1.0), ``z_bias`` (0 = uniform in z; larger = denser near ground, 1.0),
``z_min`` (1 m), ``z_max`` (null = 1.5 x max building height),
``stagnant_speed`` (m/s; null = 5 % of the inflow speed), ``stagnant_time``
(sim s, 5).
"""

from __future__ import annotations

import concurrent.futures
import copy
import functools
import logging
import math
import pathlib
import time
from typing import Any, Callable, Optional

import numpy as np
from les_render import colormaps
from les_render.fields import (
    FieldSeries,
    Grid,
    robust_range,
    upsample_mask,
    upsample_vector,
)
from les_render.timeline import Timeline
from scipy import ndimage

log = logging.getLogger(__name__)

SEGMENT_RULE = "draw segment k->k+1 only when alpha[k] > 0 and alpha[k+1] > 0"

_COMMON_DEFAULTS: dict[str, Any] = {
    "type": "particles",
    "colormap": "inferno",
    "color_variable": "speed",
    "range": None,
    "preroll": None,
    "substeps": None,
    "cfl": 0.5,
    "integrator": "rk2",
    "upsample": 1,
    "max_segment_length": None,
    "periodic": None,
    "seed": 0,
    "threads": 1,
}

_STREAK_DEFAULTS: dict[str, Any] = {
    **_COMMON_DEFAULTS,
    "name": "streaklines",
    "kind": "streaklines",
    "seeding": "inlet_rake",
    "counts": {"rake_y": 24, "rake_z": 8, "ground_line": 48},
    "points_per_line": 256,
    "radius": 0.25,
    "emission_strength": 3.0,
    "line_duration": None,
    "release_interval": None,
    "tail_fade": 0.15,
    "rake_x_offset_cells": 1.5,
    "rake_z_min": 1.5,
    "rake_z_max": None,
    "rake_z_growth": 1.3,
    "ground_line_z": 2.0,
    "ground_line_offset": None,
    "corner_heights": [0.2, 0.5, 0.85],
    "corner_offset": None,
    "footprints": None,
    "emitters": None,
}

_TRAIL_DEFAULTS: dict[str, Any] = {
    **_COMMON_DEFAULTS,
    "name": "trails",
    "kind": "trails",
    "seeding": "volume",
    "counts": 20000,
    "points_per_line": 24,
    "samples_per_frame": 2,
    "radius": 0.15,
    "emission_strength": 4.0,
    "max_age": None,
    "fade_frames": 6,
    "taper": 1.0,
    "z_bias": 1.0,
    "z_min": 1.0,
    "z_max": None,
    "stagnant_speed": None,
    "stagnant_time": 5.0,
}


def default_particle_specs() -> dict[str, dict[str, Any]]:
    """Sensible default spec dicts for the two kinds (keys ``streaklines``, ``trails``).

    Fresh deep copies, safe to mutate. ``name`` is set to the kind; ``type``
    is ``"particles"``.
    """
    return {
        "streaklines": copy.deepcopy(_STREAK_DEFAULTS),
        "trails": copy.deepcopy(_TRAIL_DEFAULTS),
    }


# -- velocity sampling ---------------------------------------------------------


class VelocitySampler:
    """Trilinear-in-space, linear-in-time velocity at many points, fast.

    Each snapshot is kept as a channels-last ``(nx+1, ny+1, nz+1, 3)`` float32
    array padded by one plane per axis (edge copy, or wrap-around on periodic
    axes), so the 8 interpolation corners of any point are at constant flat
    offsets. For a sample time the two bracketing snapshots are blended once
    (a few MB, reused for a repeated time) and every point then needs 8 small
    ``np.take`` gathers from a cache-resident array. Numerically identical to
    :func:`fields.trilinear` / :meth:`FieldSeries.sample_velocity`.
    """

    def __init__(
        self,
        fields: FieldSeries,
        upsample: int = 1,
        workers: int = 1,
        cache_size: int = 4,
        periodic: tuple[bool, bool, bool] = (False, False, False),
    ):
        self.fields = fields
        self.periodic = np.asarray(periodic, dtype=bool)
        self.upsample = int(upsample)
        self.grid: Grid = (
            fields.grid.refined(self.upsample) if self.upsample > 1 else fields.grid
        )
        self._shape = np.array(self.grid.shape, dtype=np.int64)
        self._origin = self.grid.origin
        self._inv_h = 1.0 / self.grid.spacing
        # clamp (non-periodic) or wrap (periodic) fractional indices
        self._hi = np.where(self.periodic, np.inf, self._shape - 1.0)
        self._lo = np.where(self.periodic, -np.inf, 0.0)
        self._imax = self._shape - 1
        py, pz = self._shape[1] + 1, self._shape[2] + 1
        self._strides = np.array([py * pz, pz, 1], dtype=np.int64)
        self._offsets = [
            dx * py * pz + dy * pz + dz
            for dx in (0, 1)
            for dy in (0, 1)
            for dz in (0, 1)
        ]
        self._umax: dict[int, float] = {}
        self._snapshot = functools.lru_cache(maxsize=cache_size)(self._build)
        self._blend: Optional[np.ndarray] = None
        self._tmp: Optional[np.ndarray] = None
        self._blend_key: Optional[tuple[tuple[int, float], ...]] = None
        self.workers = max(1, int(workers))
        self._pool = (
            concurrent.futures.ThreadPoolExecutor(self.workers)
            if self.workers > 1
            else None
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _build(self, k: int) -> np.ndarray:
        vel = self.fields.velocity_snapshot(k)
        if self.upsample > 1:
            vel = upsample_vector(vel, self.upsample)
            vel[:, upsample_mask(self.fields.solid, self.upsample)] = 0.0
        self._umax[k] = float(
            np.sqrt(np.max(np.sum(vel.astype(np.float32) ** 2, axis=0)))
        )
        f = vel.transpose(1, 2, 3, 0).astype(np.float32)
        for ax in range(
            3
        ):  # "+1" neighbour plane: wrap-around on periodic axes, else edge
            pad = [(0, 0)] * 4
            pad[ax] = (0, 1)
            f = np.pad(f, pad, mode="wrap" if self.periodic[ax] else "edge")
        return np.ascontiguousarray(f).reshape(-1, 3)

    def field_at(self, t: float) -> np.ndarray:
        """Padded, flattened velocity (M, 3) at sim time ``t`` (FieldSeries time weights)."""
        weights = self.fields.time_weights(t)
        if len(weights) == 1 and weights[0][1] == 1.0:
            return self._snapshot(weights[0][0])
        key = tuple((k, round(w, 9)) for k, w in weights)
        if key != self._blend_key:  # RK4's two midpoint stages share one blend
            first = self._snapshot(weights[0][0])
            if self._blend is None:
                self._blend, self._tmp = np.empty_like(first), np.empty_like(first)
            np.multiply(first, np.float32(weights[0][1]), out=self._blend)
            for k, w in weights[1:]:
                np.multiply(self._snapshot(k), np.float32(w), out=self._tmp)
                self._blend += self._tmp
            self._blend_key = key
        assert self._blend is not None
        return self._blend

    def max_speed(self, t0: float, t1: float) -> float:
        """Upper bound of |u| over sim times [t0, t1] (max over bracketing snapshots)."""
        # Cubic time weights also draw on the neighbours of the bracket.
        k0 = max(self.fields.bracket(t0)[0] - 1, 0)
        k1 = min(self.fields.bracket(t1)[1] + 1, self.fields.times.size - 1)
        out = 0.0
        for k in range(k0, k1 + 1):
            if k not in self._umax:
                self._snapshot(k)
            out = max(out, self._umax.get(k, 0.0))
        return out

    def __call__(self, points: np.ndarray, t: float) -> np.ndarray:
        """Velocity (N, 3) float32 at world points (N, 3) and sim time ``t``."""
        n = points.shape[0]
        field = self.field_at(t)
        if self._pool is None or n < 16384 * self.workers:
            return self._sample(points, field)
        bounds = np.linspace(0, n, self.workers + 1).astype(np.int64)
        parts = self._pool.map(
            lambda ij: self._sample(points[ij[0] : ij[1]], field),
            zip(bounds[:-1], bounds[1:]),
        )
        return np.concatenate(list(parts))

    def _sample(self, points: np.ndarray, field: np.ndarray) -> np.ndarray:
        idx = (points - self._origin) * self._inv_h
        if self.periodic.any():
            idx[:, self.periodic] = np.mod(
                idx[:, self.periodic], self._shape[self.periodic]
            )
        np.clip(idx, self._lo, self._hi, out=idx)
        i0 = idx.astype(np.int64)
        np.minimum(i0, self._imax, out=i0)
        w = (idx - i0).astype(np.float32)
        base = i0[:, 0] * self._strides[0] + i0[:, 1] * self._strides[1] + i0[:, 2]
        c = [np.take(field, base + o, axis=0) for o in self._offsets]
        wx, wy, wz = w[:, 0:1], w[:, 1:2], w[:, 2:3]
        # lerp along z, then y, then x (corner order: dx, dy, dz nested)
        for i in range(0, 8, 2):
            c[i] += wz * (c[i + 1] - c[i])
        c[0] += wy * (c[2] - c[0])
        c[4] += wy * (c[6] - c[4])
        c[0] += wx * (c[4] - c[0])
        return np.asarray(c[0])


class _Tracer:
    """Shared integration machinery: RK stepping, walls, CFL sub-stepping."""

    def __init__(self, fields: FieldSeries, spec: dict[str, Any]):
        self.fields = fields
        self.grid = fields.grid
        per = spec.get("periodic")
        per = detect_periodic(fields) if per is None else tuple(bool(v) for v in per)
        self.periodic = np.array((per + (False, False, False))[:3], dtype=bool)
        self.sample = VelocitySampler(
            fields,
            int(spec["upsample"]),
            int(spec.get("threads") or 1),
            periodic=tuple(self.periodic),
        )
        self.lower = self.grid.lower
        self.upper = self.grid.upper
        self.length = self.upper - self.lower
        self.min_h = float(np.min(self.grid.spacing))
        self.cfl = float(spec["cfl"])
        self.substeps = spec["substeps"]
        integ = str(spec["integrator"]).lower()
        if integ not in ("rk2", "rk4"):
            raise ValueError(f"integrator must be 'rk2' or 'rk4', got {integ!r}")
        self.rk4 = integ == "rk4"
        self.n_evals = 0
        # nearest-cell solid lookup (same rule as FieldSeries.is_solid, but flat + wrap-aware)
        self._solid_flat = fields.solid.ravel()
        # cells within one cell (26-neighbourhood) of a solid: segments shorter
        # than a cell whose ends are both outside this set cannot cross a solid
        self._near_flat = ndimage.binary_dilation(
            fields.solid, structure=np.ones((3, 3, 3), bool)
        ).ravel()
        self._shape = np.array(self.grid.shape, dtype=np.int64)
        self._inv_h = 1.0 / self.grid.spacing

    def n_sub(self, t0: float, t1: float) -> int:
        if self.substeps:
            return int(self.substeps)
        umax = self.sample.max_speed(t0, t1)
        return max(1, int(math.ceil(umax * (t1 - t0) / (self.cfl * self.min_h) - 1e-9)))

    def velocity(self, p: np.ndarray, t: float) -> np.ndarray:
        self.n_evals += 1
        return self.sample(p, t)

    def step(self, p: np.ndarray, t: float, h: float) -> tuple[np.ndarray, np.ndarray]:
        """One RK step of size ``h`` from time ``t``; the ground reflects.

        Returns ``(new positions, speed)`` where speed (N,) is the magnitude of
        the step's effective velocity (used for colouring, so no extra
        velocity evaluation is needed at record / export time).
        """
        if p.shape[0] == 0:
            return p, np.zeros(0, dtype=np.float32)
        k1 = self.velocity(p, t)
        if self.rk4:
            k2 = self.velocity(p + 0.5 * h * k1, t + 0.5 * h)
            k3 = self.velocity(p + 0.5 * h * k2, t + 0.5 * h)
            k4 = self.velocity(p + h * k3, t + h)
            vel = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        else:
            vel = self.velocity(p + 0.5 * h * k1, t + 0.5 * h)
        out = p + h * vel
        speed = np.sqrt(np.einsum("nd,nd->n", vel, vel))
        z = out[:, 2]
        below = z < self.lower[2]
        z[below] = 2.0 * self.lower[2] - z[below]
        for ax in np.flatnonzero(self.periodic):
            out[:, ax] = self.lower[ax] + np.mod(
                out[:, ax] - self.lower[ax], self.length[ax]
            )
        return out, speed

    def _cell(self, p: np.ndarray) -> np.ndarray:
        """Flat index of the cell containing each point (clamped / wrapped)."""
        idx = np.floor((p - self.lower) * self._inv_h).astype(np.int64)
        flat = np.zeros(p.shape[0], dtype=np.int64)
        for d in range(3):
            n = self._shape[d]
            col = (
                np.mod(idx[:, d], n)
                if self.periodic[d]
                else np.clip(idx[:, d], 0, n - 1)
            )
            flat *= n
            flat += col
        return flat

    def is_solid(self, p: np.ndarray) -> np.ndarray:
        """True inside a solid cell (same cells as FieldSeries.is_solid; periodic-aware)."""
        return np.take(self._solid_flat, self._cell(p))

    def near_solid(self, p: np.ndarray) -> np.ndarray:
        return np.take(self._near_flat, self._cell(p))

    def in_domain(self, p: np.ndarray) -> np.ndarray:
        return self.fields.in_domain(p)


# -- flow statistics, footprints, emitters -------------------------------------


def detect_periodic(
    fields: FieldSeries, n_snapshots: int = 6
) -> tuple[bool, bool, bool]:
    """Guess periodic horizontal boundaries from the data.

    On a periodic axis the first and last cell planes are neighbours, so their
    velocity fluctuations correlate about as well as planes 0 and 1 do (uDALES
    idealized runs: 0.92 vs 0.90 in y; ~0.3 for planes half a domain apart).
    """
    ks = np.unique(
        np.linspace(fields.times.size // 3, fields.times.size - 1, n_snapshots).astype(
            int
        )
    )
    out = []
    for ax in (0, 1):
        if fields.grid.shape[ax] < 4:
            out.append(False)
            continue
        wrap: list[float] = []
        near: list[float] = []
        for k in ks:
            u = np.moveaxis(fields.velocity_snapshot(int(k))[0], ax, 0)
            fluid = ~np.moveaxis(fields.solid, ax, 0)
            pairs = {"wrap": (0, -1), "near": (0, 1)}
            for key, (a, b) in pairs.items():
                m = fluid[a] & fluid[b]
                x, y = u[a][m], u[b][m]
                if x.size < 8 or x.std() < 1e-6 or y.std() < 1e-6:
                    c = 0.0
                else:
                    c = float(np.corrcoef(x, y)[0, 1])
                (wrap if key == "wrap" else near).append(c)
        cw, cn = float(np.mean(wrap)), float(np.mean(near))
        out.append(cw > 0.6 and cw >= 0.8 * cn)
    return (out[0], out[1], False)


def _flow_stats(fields: FieldSeries, timeline: Timeline) -> dict[str, Any]:
    """Mean inflow u on the inlet planes, transit time, mean horizontal flow direction."""
    ks = np.unique(
        [fields.bracket(t)[0] for t in np.linspace(timeline.t_start, timeline.t_end, 5)]
    )
    fluid = ~fields.solid
    u_in, dirs = [], []
    for k in ks:
        vel = fields.velocity_snapshot(int(k))
        inlet = fluid[:2]
        u_in.append(float(vel[0, :2][inlet].mean()) if inlet.any() else 0.0)
        dirs.append([float(vel[0][fluid].mean()), float(vel[1][fluid].mean())])
    d = np.mean(dirs, axis=0)
    nrm = float(np.hypot(*d))
    direction = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
    u_ref = max(float(np.mean(u_in)), 0.5)
    length_x = float(fields.grid.upper[0] - fields.grid.lower[0])
    return {"u_ref": u_ref, "transit": length_x / u_ref, "direction": direction}


def footprints_from_solid(fields: FieldSeries) -> list[dict[str, Any]]:
    """Axis-aligned boxes of the connected solid regions (same format as the manifest)."""
    labels, _ = ndimage.label(fields.solid)
    lo_face, h = fields.grid.lower, fields.grid.spacing
    out = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        lo = lo_face + h * np.array([s.start for s in sl])
        hi = lo_face + h * np.array([s.stop for s in sl])
        out.append({"min": lo.round(3).tolist(), "max": hi.round(3).tolist()})
    return out


def _max_building_height(
    fields: FieldSeries, footprints: list[dict[str, Any]]
) -> float:
    if footprints:
        return float(max(fp["max"][2] for fp in footprints))
    ks = np.flatnonzero(fields.solid.any(axis=(0, 1)))
    if ks.size == 0:
        return float(0.25 * (fields.grid.upper[2] - fields.grid.lower[2]))
    return float(fields.grid.lower[2] + fields.grid.spacing[2] * (ks[-1] + 1))


def _geometric_levels(z0: float, z1: float, n: int, growth: float) -> np.ndarray:
    if n <= 1:
        return np.array([z0])
    steps = growth ** np.arange(n - 1)
    return z0 + (z1 - z0) * np.concatenate([[0.0], np.cumsum(steps)]) / steps.sum()


def _span(lo: float, hi: float, n: int) -> np.ndarray:
    """``n`` cell-centred positions across [lo, hi]."""
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


def streak_emitters(
    fields: FieldSeries,
    spec: dict[str, Any],
    stats: dict[str, Any],
    footprints: list[dict[str, Any]],
) -> np.ndarray:
    """Emitter positions (E, 3) for the streakline ``seeding`` mode(s); solid ones dropped."""
    g = fields.grid
    h = g.spacing
    counts = spec["counts"] or {}
    hmax = _max_building_height(fields, footprints)
    modes = spec["seeding"]
    modes = [modes] if isinstance(modes, str) else list(modes)
    pts: list[np.ndarray] = []
    for mode in modes:
        if mode == "inlet_rake":
            x = g.lower[0] + float(spec["rake_x_offset_cells"]) * h[0]
            ys = _span(g.lower[1], g.upper[1], int(counts.get("rake_y", 24)))
            z1 = spec["rake_z_max"] or 1.2 * hmax
            z1 = min(float(z1), g.upper[2] - 0.5 * h[2])
            zs = _geometric_levels(
                float(spec["rake_z_min"]),
                z1,
                int(counts.get("rake_z", 8)),
                float(spec["rake_z_growth"]),
            )
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            pts.append(np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], axis=1))
        elif mode == "ground_line":
            off = spec["ground_line_offset"]
            off = 2.0 * h[0] if off is None else float(off)
            x_first = min(
                (fp["min"][0] for fp in footprints),
                default=g.lower[0] + 0.25 * (g.upper[0] - g.lower[0]),
            )
            x = max(x_first - off, g.lower[0] + 1.5 * h[0])
            ys = _span(g.lower[1], g.upper[1], int(counts.get("ground_line", 48)))
            pts.append(
                np.stack(
                    [
                        np.full(ys.size, x),
                        ys,
                        np.full(ys.size, float(spec["ground_line_z"])),
                    ],
                    axis=1,
                )
            )
        elif mode == "building_corners":
            pts.append(_corner_emitters(footprints, spec, stats["direction"], h))
        elif mode == "custom":
            pts.append(np.asarray(spec["emitters"], dtype=np.float64).reshape(-1, 3))
        else:
            raise ValueError(f"unknown streakline seeding {mode!r}")
    emitters = np.concatenate(pts) if pts else np.zeros((0, 3))
    ok = fields.in_domain(emitters) & ~fields.is_solid(emitters)
    if not ok.all():
        log.warning(
            "dropping %d emitters inside solids / outside the domain", int((~ok).sum())
        )
    if not ok.any():
        raise ValueError("no valid streakline emitters")
    return np.asarray(emitters[ok])


def _corner_emitters(
    footprints: list[dict[str, Any]],
    spec: dict[str, Any],
    direction: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Emitters just upstream of the two windward *separation* corners of each building.

    For flow direction f the separation corners are the footprint corners with
    extreme cross-flow coordinate (near-ties -> the more upstream one): for
    flow along +x these are the two upstream corners; for flow toward +x, -y
    the south-west and north-east ones.
    """
    off = spec["corner_offset"]
    off = 0.75 * float(h[0]) if off is None else float(off)
    f = np.asarray(direction, dtype=np.float64)
    n = np.array([-f[1], f[0]])
    out = []
    for fp in footprints:
        lo, hi = np.asarray(fp["min"], float), np.asarray(fp["max"], float)
        corners = np.array(
            [[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]]
        )
        perp, along = corners @ n, corners @ f
        tol = 0.05 * max(hi[0] - lo[0], hi[1] - lo[1], 1.0)  # ~3 deg of flow angle
        centre = 0.5 * (lo[:2] + hi[:2])
        for pick in (perp.min(), perp.max()):
            cand = np.flatnonzero(np.abs(perp - pick) <= tol)
            c = corners[cand[np.argmin(along[cand])]]
            side = np.sign((c - centre) @ n) or 1.0
            xy = c - off * f + side * off * n
            for frac in spec["corner_heights"]:
                out.append([xy[0], xy[1], lo[2] + float(frac) * (hi[2] - lo[2])])
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


# -- polyline finalisation (shared) --------------------------------------------


def finalize_lines(
    points: np.ndarray,
    alive: np.ndarray,
    is_solid: Callable[[np.ndarray], np.ndarray],
    max_segment_length: float,
    check_spacing: float,
    collapse: bool = True,
    near: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the segment rule. Returns ``(points, visible)``.

    A segment is valid when both ends are alive, it is at most
    ``max_segment_length`` long and interior points spaced at most
    ``check_spacing / 4`` apart along it are all fluid (at least the midpoint is
    checked). Both endpoints of every invalid segment are hidden, and (with
    ``collapse``) dead points are moved onto the nearest live point of their
    line, so hidden geometry is zero-length. ``near`` (lines, k) optionally
    flags points close to a solid (:meth:`_Tracer.near_solid`); segments no
    longer than ``check_spacing / 4`` with neither end near a solid skip the
    midpoint test; longer ones are probed every ``check_spacing / 4``.
    """
    n_lines, k = alive.shape
    seg = alive[:, :-1] & alive[:, 1:]
    d = points[:, 1:] - points[:, :-1]
    len2 = np.einsum("lkd,lkd->lk", d, d)
    seg &= len2 <= max_segment_length**2
    # Interior points every quarter ``check_spacing`` (1/8 cell): nearest-cell
    # solids are boxes, and a segment can clip a box corner well away from its
    # midpoint. Diagonal grazes shallower than the probe spacing can remain;
    # at render scale they are invisible.
    step = 0.25 * check_spacing
    long_ = len2 > step**2
    # midpoint test (only near solids when ``near`` is given), extra points on long ones
    test = seg if near is None else seg & (near[:, :-1] | near[:, 1:] | long_)
    li, ki = np.nonzero(test)
    if li.size:
        seg[li, ki] = ~is_solid(points[li, ki] + 0.5 * d[li, ki])
    li, ki = np.nonzero(seg & long_)
    if li.size:
        n_chk = np.ceil(np.sqrt(len2[li, ki]) / step).astype(np.int64) - 1
        bad = np.zeros(li.size, dtype=bool)
        for n in np.unique(n_chk):
            sel = np.flatnonzero(n_chk == n)
            a, b = points[li[sel], ki[sel]], points[li[sel], ki[sel] + 1]
            hit = np.zeros(sel.size, dtype=bool)
            for s in (np.arange(n) + 1.0) / (n + 1.0):
                hit |= is_solid(a + s * (b - a))
            bad[sel] = hit
        seg[li[bad], ki[bad]] = False
    visible = alive.copy()
    visible[:, :-1] &= seg
    visible[:, 1:] &= seg  # => every visible point has only valid segments
    if not collapse:
        return points, visible
    rows = np.flatnonzero(~alive.all(axis=1))
    if rows.size == 0:
        return points, visible
    points = points.copy()
    alive = alive[rows]
    ar = np.arange(k)
    fwd = np.maximum.accumulate(np.where(alive, ar, -1), axis=1)
    bwd = np.minimum.accumulate(np.where(alive, ar, k)[:, ::-1], axis=1)[:, ::-1]
    src = np.where(fwd >= 0, fwd, np.where(bwd < k, bwd, ar))
    points[rows] = np.take_along_axis(points[rows], src[:, :, None], axis=1)
    return points, visible


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.asarray(x * x * (3.0 - 2.0 * x))


# -- simulations ---------------------------------------------------------------


class _Streaklines:
    """Ring buffer of ``K`` releases per emitter; slot ``r % K`` holds release ``r``."""

    def __init__(
        self,
        tracer: _Tracer,
        emitters: np.ndarray,
        k: int,
        release_interval: float,
        t0: float,
    ):
        self.tr = tracer
        self.emitters = emitters
        self.E, self.K = emitters.shape[0], k
        self.tau = float(release_interval)
        self.t0 = t0
        self.pos = np.repeat(emitters[:, None, :], k, axis=1).reshape(-1, 3).copy()
        self.speed = np.zeros(self.E * k, dtype=np.float32)
        self.alive = np.zeros(self.E * k, dtype=bool)
        self.n_released = 0
        self._release(t0, t0)

    def _release(self, t_rel: float, t_now: float) -> None:
        """Release at ``t_rel``, advanced with one Euler step to ``t_now`` (sub-step end)."""
        slot = self.n_released % self.K
        rows = np.arange(self.E) * self.K + slot
        vel = self.tr.velocity(self.emitters, t_rel)
        self.pos[rows] = self.emitters + (t_now - t_rel) * vel
        self.speed[rows] = np.linalg.norm(vel, axis=1)
        self.alive[rows] = True
        self.n_released += 1

    def substep(self, t: float, h: float) -> None:
        idx = np.flatnonzero(self.alive)
        p, speed = self.tr.step(self.pos[idx], t, h)
        ok = self.tr.in_domain(p) & ~self.tr.is_solid(p)
        self.pos[idx] = p
        self.speed[idx] = speed
        self.alive[idx[~ok]] = False
        t_end = t + h
        while self.t0 + self.n_released * self.tau <= t_end + 1e-9:
            self._release(self.t0 + self.n_released * self.tau, t_end)

    def record(self, t: float) -> None:
        pass

    def frame(
        self, max_seg: float, check_spacing: float, tail_fade: float
    ) -> dict[str, np.ndarray]:
        order = (self.n_released - 1 - np.arange(self.K)) % self.K
        pos = self.pos.reshape(self.E, self.K, 3)[:, order]
        speed = self.speed.reshape(self.E, self.K)[:, order]
        alive = self.alive.reshape(self.E, self.K)[:, order]
        alive[:, np.arange(self.K) >= self.n_released] = False
        near = self.tr.near_solid(pos.reshape(-1, 3)).reshape(alive.shape)
        pos, visible = finalize_lines(
            pos, alive, self.tr.is_solid, max_seg, check_spacing, near=near
        )
        ramp = np.ones(self.K)
        n_fade = tail_fade * self.K
        if n_fade > 0:
            ramp = _smoothstep((self.K - 1 - np.arange(self.K)) / n_fade)
        alpha = visible * ramp[None, :]
        return {"points": pos, "speed": np.where(alive, speed, 0.0), "alpha": alpha}


class _Trails:
    """``N`` recycled tracers with a ring-buffered history of ``M`` samples each."""

    def __init__(
        self,
        tracer: _Tracer,
        spec: dict[str, Any],
        n: int,
        m: int,
        record_dt: float,
        frame_dt: float,
        stats: dict[str, Any],
        hmax: float,
        rng: np.random.Generator,
    ):
        self.tr = tracer
        self.rng = rng
        self.N, self.M = n, m
        self.record_dt = record_dt
        g = tracer.grid
        self.z_min = max(float(spec["z_min"]), float(g.lower[2]))
        z_max = spec["z_max"] if spec["z_max"] is not None else 1.5 * hmax
        self.z_max = float(np.clip(z_max, self.z_min + 1e-3, g.upper[2] - 1e-3))
        self.z_bias = float(spec["z_bias"])
        max_age = spec["max_age"]
        self.max_age = float(max_age) if max_age is not None else 0.4 * stats["transit"]
        self.fade = max(float(spec["fade_frames"]) * frame_dt, 1e-6)
        self.max_age = max(self.max_age, 2.5 * self.fade)
        self.taper = float(spec["taper"])
        ss = spec["stagnant_speed"]
        self.stagnant_speed = float(ss) if ss is not None else 0.05 * stats["u_ref"]
        self.stagnant_time = float(spec["stagnant_time"])

        self.pos = self._spawn(n)
        self.speed = np.zeros(n, dtype=np.float32)
        self.life = self._lives(n)
        self.age = rng.uniform(0.0, 1.0, n) * self.life  # staggered from the start
        self.n_rec = np.zeros(n, dtype=np.int64)
        self.frozen = np.zeros(n, dtype=bool)
        self.slow = np.zeros(n)
        self.hist = np.repeat(self.pos[:, None, :].astype(np.float32), m, axis=1)
        self.hist_speed = np.zeros((n, m), dtype=np.float32)
        self.hist_near = np.repeat(tracer.near_solid(self.pos)[:, None], m, axis=1)
        self.head = 0

    def _lives(self, n: int) -> np.ndarray:
        return self.max_age * self.rng.uniform(0.5, 1.0, n)

    def _spawn(self, n: int) -> np.ndarray:
        """Random fluid positions, uniform in x, y; ``z_bias`` densifies low z."""
        g = self.tr.grid
        out = np.empty((0, 3))
        while out.shape[0] < n:
            m = int(1.3 * (n - out.shape[0])) + 16
            r = self.rng.random((m, 3))
            p = np.empty((m, 3))
            p[:, 0] = g.lower[0] + r[:, 0] * (g.upper[0] - g.lower[0])
            p[:, 1] = g.lower[1] + r[:, 1] * (g.upper[1] - g.lower[1])
            p[:, 2] = self.z_min + r[:, 2] ** (1.0 + self.z_bias) * (
                self.z_max - self.z_min
            )
            out = np.concatenate([out, p[~self.tr.is_solid(p)]])
        return out[:n]

    def _die(self, idx: np.ndarray) -> None:
        """Start a fade-out (never a pop): life ends ``fade`` s from now."""
        self.life[idx] = np.minimum(self.life[idx], self.age[idx] + self.fade)

    def substep(self, t: float, h: float) -> None:
        idx = np.flatnonzero(~self.frozen)
        old = self.pos[idx]
        p, speed = self.tr.step(old, t, h)
        inside = self.tr.in_domain(p)
        hit = inside & self.tr.is_solid(p)
        p[hit] = old[hit]  # freeze at the last fluid position, then fade out
        speed[hit] = 0.0
        self.pos[idx] = p
        self.speed[idx] = speed
        self.frozen[idx[hit]] = True
        self._die(idx[hit | ~inside])  # outside: keeps drifting out while fading

    def record(self, t: float) -> None:
        self.age += self.record_dt
        self.n_rec += 1
        self.head = (self.head + 1) % self.M
        self.hist[:, self.head] = self.pos
        self.hist_speed[:, self.head] = self.speed
        self.hist_near[:, self.head] = self.tr.near_solid(self.pos)
        slow = self.speed < self.stagnant_speed
        self.slow = np.where(slow, self.slow + self.record_dt, 0.0)
        stuck = np.flatnonzero(self.slow > self.stagnant_time)
        self._die(stuck)
        self.slow[stuck] = -np.inf  # don't re-trigger while fading
        reborn = np.flatnonzero(self.age >= self.life)
        if reborn.size:
            p = self._spawn(reborn.size)
            speed = np.linalg.norm(self.tr.velocity(p, t), axis=1)
            self.pos[reborn] = p
            self.speed[reborn] = speed
            self.hist[reborn] = p[
                :, None, :
            ]  # fresh history: never connects to the old path
            self.hist_speed[reborn] = speed[:, None]
            self.hist_near[reborn] = self.tr.near_solid(p)[:, None]
            self.age[reborn] = 0.0
            self.n_rec[reborn] = 0
            self.life[reborn] = self._lives(reborn.size)
            self.frozen[reborn] = False
            self.slow[reborn] = 0.0

    def frame(
        self, max_seg: float, check_spacing: float, tail_fade: float
    ) -> dict[str, np.ndarray]:
        order = (self.head - np.arange(self.M)) % self.M
        pos = np.take(self.hist, order, axis=1)
        speed = np.take(self.hist_speed, order, axis=1)
        valid = np.arange(self.M)[None, :] <= self.n_rec[:, None]
        # samples from before (re)birth already sit on the birth point: no collapse needed
        pos, visible = finalize_lines(
            pos,
            valid,
            self.tr.is_solid,
            max_seg,
            check_spacing,
            collapse=False,
            near=np.take(self.hist_near, order, axis=1),
        )
        life = _smoothstep(self.age / self.fade) * _smoothstep(
            (self.life - self.age) / self.fade
        )
        taper = (1.0 - np.arange(self.M) / max(self.M - 1, 1)) ** self.taper
        alpha = np.where(
            visible,
            life.astype(np.float32)[:, None] * taper.astype(np.float32)[None, :],
            np.float32(0),
        )
        return {"points": pos, "speed": speed, "alpha": alpha}


# -- driver ----------------------------------------------------------------------


def _resolve_spec(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec.get("kind", "streaklines")
    if kind not in ("streaklines", "trails"):
        raise ValueError(
            f"particle kind must be 'streaklines' or 'trails', got {kind!r}"
        )
    base = default_particle_specs()[kind]
    # null (Hydra ~) means "use the default"; every auto default is itself null
    out = {**base, **{k: v for k, v in spec.items() if v is not None}}
    if kind == "streaklines" and isinstance(spec.get("counts"), dict):
        out["counts"] = {**base["counts"], **spec["counts"]}
    if "name" not in spec:
        raise KeyError("particle spec needs a 'name'")
    if out["color_variable"] != "speed":
        raise ValueError("particles support color_variable 'speed' only")
    if kind == "trails" and out["seeding"] != "volume":
        raise ValueError(
            f"trails support seeding 'volume' only, got {out['seeding']!r}"
        )
    return out


def _write_npz(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> int:
    np.savez(path, **arrays)  # type: ignore[arg-type, unused-ignore]
    return path.stat().st_size


def _auto_range(fields: FieldSeries, timeline: Timeline) -> tuple[float, float]:
    ks = np.unique(
        [fields.bracket(t)[0] for t in np.linspace(timeline.t_start, timeline.t_end, 5)]
    )
    fluid = ~fields.solid
    vals = []
    for k in ks:
        v = fields.velocity_snapshot(int(k))
        vals.append(np.sqrt(np.sum(v[:, fluid] ** 2, axis=0)))
    return robust_range(np.concatenate(vals), 1.0, 99.5)


def export_particles(
    fields: FieldSeries, timeline: Timeline, spec: dict[str, Any], out_dir: pathlib.Path
) -> dict[str, Any]:
    """Trace a particle layer and write one npz per video frame.

    Returns the manifest layer dict (``type: "particles"``, see the module
    docstring for the spec keys and the segment rule).
    """
    spec = _resolve_spec(spec)
    name, kind = str(spec["name"]), spec["kind"]
    out_dir = pathlib.Path(out_dir)
    folder = out_dir / "particles" / name
    folder.mkdir(parents=True, exist_ok=True)

    stats = _flow_stats(fields, timeline)
    rng = np.random.default_rng(spec["seed"])
    tracer = _Tracer(fields, spec)
    dt = timeline.dt
    k = int(spec["points_per_line"])
    max_seg = spec["max_segment_length"]
    max_seg = (
        3.0 * float(np.max(fields.grid.spacing)) if max_seg is None else float(max_seg)
    )
    check_spacing = 0.5 * float(np.min(fields.grid.spacing))
    preroll = (
        float(spec["preroll"]) if spec["preroll"] is not None else stats["transit"]
    )
    footprints = spec.get("footprints") or footprints_from_solid(fields)
    hmax = _max_building_height(fields, footprints)

    extra: dict[str, Any] = {}
    if kind == "streaklines":
        line_duration = spec["line_duration"] or stats["transit"]
        tau = spec["release_interval"] or line_duration / k
        preroll = max(preroll, k * float(tau))
        n_pre = int(math.ceil(preroll / dt - 1e-9))
        emitters = streak_emitters(fields, spec, stats, footprints)
        sim: Any = _Streaklines(
            tracer, emitters, k, float(tau), timeline.t_start - n_pre * dt
        )
        per_frame = 1
        n_lines = emitters.shape[0]
        tail_fade = float(spec["tail_fade"])
        extra = {"release_interval": float(tau), "emitters": emitters.round(3).tolist()}
    else:
        counts = spec["counts"]
        n = int(counts["particles"] if isinstance(counts, dict) else counts)
        per_frame = max(1, int(spec["samples_per_frame"]))
        n_pre = int(math.ceil(preroll / dt - 1e-9))
        sim = _Trails(tracer, spec, n, k, dt / per_frame, dt, stats, hmax, rng)
        n_lines = n
        tail_fade = 0.0
        extra = {"samples_per_frame": per_frame, "sample_dt": dt / per_frame}

    vrange = spec["range"]
    vmin, vmax = (
        (float(vrange[0]), float(vrange[1]))
        if vrange is not None
        else _auto_range(fields, timeline)
    )

    t_wall = time.perf_counter()
    t_frames = 0.0
    total_bytes = 0
    writer = concurrent.futures.ThreadPoolExecutor(1)
    pending: Optional[concurrent.futures.Future] = None
    t = timeline.t_start - n_pre * dt
    for i in range(-n_pre, timeline.n_frames):
        if i > -n_pre:
            tf = time.perf_counter()
            for r in range(per_frame):
                ta = t + r * dt / per_frame
                tb = ta + dt / per_frame
                ns = tracer.n_sub(ta, tb)
                hs = (tb - ta) / ns
                for s in range(ns):
                    sim.substep(ta + s * hs, hs)
                sim.record(tb)
            t = timeline.t_start + i * dt
            if i >= 0:
                t_frames += time.perf_counter() - tf
        if i >= 0:
            tf = time.perf_counter()
            data = sim.frame(max_seg, check_spacing, tail_fade)
            arrays = {
                "points": data["points"].astype(np.float32),
                "speed": data["speed"].astype(np.float16),
                "alpha": np.clip(data["alpha"], 0.0, 1.0).astype(np.float16),
            }
            # write in the background, overlapping the next frame's integration
            if pending is not None:
                total_bytes += pending.result()
            pending = writer.submit(_write_npz, folder / f"{name}.{i:04d}.npz", arrays)
            t_frames += time.perf_counter() - tf
    if pending is not None:
        total_bytes += pending.result()
    writer.shutdown()
    tracer.sample.close()
    log.info(
        "particles %s (%s): %d lines x %d pts, %d frames (+%d preroll), %.3f s/frame, %.1f s total, %.2f MB/frame",
        name,
        kind,
        n_lines,
        k,
        timeline.n_frames,
        n_pre,
        t_frames / max(timeline.n_frames, 1),
        time.perf_counter() - t_wall,
        total_bytes / max(timeline.n_frames, 1) / 1e6,
    )

    layer: dict[str, Any] = {
        "name": name,
        "type": "particles",
        "kind": kind,
        "pattern": f"particles/{name}/{name}.{{frame:04d}}.npz",
        "frame_step": 1,
        "n_files": timeline.n_frames,
        "n_lines": int(n_lines),
        "points_per_line": k,
        "radius": float(spec["radius"]),
        "emission_strength": float(spec["emission_strength"]),
        "segment_rule": SEGMENT_RULE,
        "preroll": float(n_pre * dt),
        "periodic": [bool(v) for v in tracer.periodic[:2]],
        **extra,
    }
    layer.update(colormaps.layer_color_spec(str(spec["colormap"]), vmin, vmax, "speed"))
    return layer


__all__ = [
    "export_particles",
    "default_particle_specs",
    "finalize_lines",
    "footprints_from_solid",
    "streak_emitters",
    "detect_periodic",
    "VelocitySampler",
    "SEGMENT_RULE",
]
