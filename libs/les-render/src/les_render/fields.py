"""Space-time access to the LES velocity field.

Internal array convention: spatial arrays are indexed ``(x, y, z)`` (the
NetCDF is ``(z, y, x)``), which is also OpenVDB's ``[i, j, k]`` order. Vector
fields carry a leading component axis: ``(3, nx, ny, nz)``.

Solid cells hold solver junk (not zero) in the raw output; every accessor
here returns velocity with solid cells zeroed, which is also the no-slip wall
value, so cubic upsampling produces a physical near-wall ramp instead of
smearing junk into the fluid.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Any, Callable

import numpy as np
import xarray as xr
from scipy import ndimage


@dataclasses.dataclass(frozen=True)
class Grid:
    """Uniform cell-centred grid. ``x, y, z`` are cell-centre coordinates."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.x.size, self.y.size, self.z.size)

    @property
    def spacing(self) -> np.ndarray:
        return np.array(
            [self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0]]
        )

    @property
    def origin(self) -> np.ndarray:
        """Centre of cell (0, 0, 0)."""
        return np.array([self.x[0], self.y[0], self.z[0]])

    @property
    def lower(self) -> np.ndarray:
        """Lower domain corner (cell faces, not centres)."""
        return self.origin - 0.5 * self.spacing

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.x[-1], self.y[-1], self.z[-1]]) + 0.5 * self.spacing

    def to_index(self, points: np.ndarray) -> np.ndarray:
        """World (N, 3) -> fractional cell index (N, 3); cell centres are integers."""
        return np.asarray((points - self.origin) / self.spacing)

    def refined(self, factor: int) -> "Grid":
        """Grid with ``factor``x the cells over the same domain (cell-centred)."""
        axes = []
        for c, h in zip((self.x, self.y, self.z), self.spacing):
            n = c.size * factor
            lo = c[0] - 0.5 * h
            axes.append(lo + (np.arange(n) + 0.5) * h / factor)
        return Grid(*axes)


class FieldSeries:
    """Lazy, cached access to the (solid-masked) velocity snapshots of a case."""

    def __init__(
        self, ds: xr.Dataset, cache_size: int = 8, time_interpolation: str = "cubic"
    ):
        if time_interpolation not in ("cubic", "linear"):
            raise ValueError(
                f"time_interpolation must be 'cubic' or 'linear', got {time_interpolation!r}"
            )
        self.ds = ds
        self.time_interpolation = time_interpolation
        self.grid = Grid(
            ds["xt"].values.astype(np.float64),
            ds["yt"].values.astype(np.float64),
            ds["zt"].values.astype(np.float64),
        )
        self.times = ds["time"].values.astype(np.float64)
        if "blanking" in ds:
            self.solid = ds["blanking"].values.astype(bool).transpose(2, 1, 0)
        else:
            self.solid = np.zeros(self.grid.shape, dtype=bool)
        self.has_pressure = "pres" in ds
        self._velocity = functools.lru_cache(maxsize=cache_size)(self._load_velocity)
        self._pressure = functools.lru_cache(maxsize=cache_size)(self._load_pressure)

    # -- snapshots -----------------------------------------------------------

    def _load_velocity(self, k: int) -> np.ndarray:
        snap = self.ds[["u", "v", "w"]].isel(time=k).load()
        vel = np.stack([snap[c].values.transpose(2, 1, 0) for c in ("u", "v", "w")])
        vel = np.nan_to_num(vel.astype(np.float32))
        vel[:, self.solid] = 0.0
        vel.flags.writeable = False
        return vel

    def _load_pressure(self, k: int) -> np.ndarray:
        p = self.ds["pres"].isel(time=k).values.transpose(2, 1, 0).astype(np.float32)
        p = np.nan_to_num(p)
        fluid = ~self.solid
        p = p - p[fluid].mean()  # gauge: only fluctuations are meaningful
        p[self.solid] = 0.0
        p.flags.writeable = False
        return p

    def velocity_snapshot(self, k: int) -> np.ndarray:
        """Velocity at stored snapshot ``k``: (3, nx, ny, nz), solids zeroed."""
        return self._velocity(int(k))

    def pressure_snapshot(self, k: int) -> np.ndarray:
        return self._pressure(int(k))

    # -- time interpolation --------------------------------------------------

    def bracket(self, t: float) -> tuple[int, int, float]:
        """Snapshots ``(k0, k1)`` around sim time ``t`` and the blend weight of k1."""
        t = float(np.clip(t, self.times[0], self.times[-1]))
        k1 = int(np.searchsorted(self.times, t, side="right"))
        k1 = min(max(k1, 1), self.times.size - 1)
        k0 = k1 - 1
        alpha = (t - self.times[k0]) / (self.times[k1] - self.times[k0])
        return k0, k1, float(np.clip(alpha, 0.0, 1.0))

    def time_weights(self, t: float) -> list[tuple[int, float]]:
        """Snapshot weights ``[(k, w), ...]`` for sim time ``t`` (weights sum to 1).

        ``cubic`` (default) is Catmull-Rom through the four surrounding
        snapshots (end snapshots repeated). Linear blending leaves the time
        derivative of the field discontinuous at every snapshot, which folds
        streaklines into straight segments with sharp kinks spaced
        ``U * dt_snapshot`` apart; Catmull-Rom is C1 and removes them.
        """
        k0, k1, a = self.bracket(t)
        if self.time_interpolation == "linear" or self.times.size < 3:
            return [(k0, 1.0 - a), (k1, a)]
        last = self.times.size - 1
        a2, a3 = a * a, a * a * a
        weights = (
            (max(k0 - 1, 0), 0.5 * (-a3 + 2.0 * a2 - a)),
            (k0, 0.5 * (3.0 * a3 - 5.0 * a2 + 2.0)),
            (k1, 0.5 * (-3.0 * a3 + 4.0 * a2 + a)),
            (min(k1 + 1, last), 0.5 * (a3 - a2)),
        )
        merged: dict[int, float] = {}
        for k, w in weights:
            merged[k] = merged.get(k, 0.0) + w
        return [(k, w) for k, w in merged.items() if w != 0.0]

    def velocity(self, t: float) -> np.ndarray:
        """Velocity at sim time ``t``: (3, nx, ny, nz)."""
        return self._blend(self.velocity_snapshot, t)

    def pressure(self, t: float) -> np.ndarray:
        return self._blend(self.pressure_snapshot, t)

    def _blend(self, snapshot: Callable[[int], np.ndarray], t: float) -> np.ndarray:
        weights = self.time_weights(t)
        out = np.float32(weights[0][1]) * snapshot(weights[0][0])
        for k, w in weights[1:]:
            out += np.float32(w) * snapshot(k)
        return out

    # -- point sampling ------------------------------------------------------

    def sample_velocity(self, points: np.ndarray, t: float) -> np.ndarray:
        """Trilinear-in-space, (by default) cubic-in-time velocity at world points
        (N, 3) -> (N, 3). Points outside the domain are clamped to the boundary cells.
        """
        idx = self.grid.to_index(points)
        out = np.zeros((points.shape[0], 3), dtype=np.float32)
        for k, w in self.time_weights(t):
            out += np.float32(w) * trilinear(self.velocity_snapshot(k), idx)
        return out

    def is_solid(self, points: np.ndarray) -> np.ndarray:
        """True where the nearest cell is solid (points are clamped to the domain)."""
        idx = np.rint(self.grid.to_index(points)).astype(np.int64)
        for d, n in enumerate(self.grid.shape):
            np.clip(idx[:, d], 0, n - 1, out=idx[:, d])
        return self.solid[idx[:, 0], idx[:, 1], idx[:, 2]]

    def in_domain(self, points: np.ndarray) -> np.ndarray:
        return np.all((points >= self.grid.lower) & (points <= self.grid.upper), axis=1)


def trilinear(field: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Trilinear sample of ``field`` ((C, nx, ny, nz) or (nx, ny, nz)) at fractional
    indices ``idx`` (N, 3), clamped to the grid. Returns (N, C) or (N,)."""
    scalar = field.ndim == 3
    f = field[None] if scalar else field
    shape = np.array(f.shape[1:])
    coords = np.clip(np.asarray(idx, dtype=np.float32), 0.0, shape - 1.0).T
    out = np.empty((coords.shape[1], f.shape[0]), dtype=np.float32)
    for c in range(f.shape[0]):
        # order=1 with prefilter off is plain trilinear, evaluated in C.
        ndimage.map_coordinates(
            f[c], coords, output=out[:, c], order=1, mode="nearest", prefilter=False
        )
    return out[:, 0] if scalar else out


# -- derived quantities --------------------------------------------------------


def velocity_gradient(vel: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """``J[i, j] = d u_i / d x_j`` on the grid: (3, 3, nx, ny, nz)."""
    return np.stack(
        [np.stack(np.gradient(vel[i], *spacing, edge_order=1)) for i in range(3)]
    )


def speed(vel: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(vel.astype(np.float32) ** 2, axis=0))


def vorticity(vel: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    J = velocity_gradient(vel, spacing)
    return np.stack([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def q_criterion(vel: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """``Q = 0.5 (|Omega|^2 - |S|^2)``; Q > 0 marks rotation-dominated cores."""
    J = velocity_gradient(vel, spacing)
    S = 0.5 * (J + J.transpose(1, 0, 2, 3, 4))
    W = 0.5 * (J - J.transpose(1, 0, 2, 3, 4))
    return 0.5 * (np.sum(W**2, axis=(0, 1)) - np.sum(S**2, axis=(0, 1)))


SCALARS = ("speed", "vorticity_magnitude", "q_criterion", "u", "v", "w", "pressure")


def scalar_field(
    fields: FieldSeries, name: str, t: float, upsample: int = 1
) -> tuple[np.ndarray, Grid]:
    """A named scalar at sim time ``t`` on the (optionally refined) grid.

    Velocity is upsampled *before* differentiation (tricubic), so derived
    quantities are smooth rather than blocky. Solid cells are zeroed.
    Returns ``(values (nx, ny, nz) float32, grid)``.
    """
    grid = fields.grid
    solid = fields.solid
    if name == "pressure":
        if not fields.has_pressure:
            raise KeyError("state file has no 'pres' variable")
        p = fields.pressure(t)
        if upsample > 1:
            p = ndimage.zoom(p, upsample, order=3, mode="nearest", grid_mode=True)
            solid = upsample_mask(fields.solid, upsample)
            grid = grid.refined(upsample)
        p[solid] = 0.0
        return p.astype(np.float32), grid

    vel = fields.velocity(t)
    if upsample > 1:
        vel = upsample_vector(vel, upsample)
        solid = upsample_mask(fields.solid, upsample)
        grid = grid.refined(upsample)
        vel[:, solid] = 0.0

    if name == "speed":
        out = speed(vel)
    elif name in ("u", "v", "w"):
        out = vel["uvw".index(name)].copy()
    elif name == "vorticity_magnitude":
        out = speed(vorticity(vel, grid.spacing))
    elif name == "q_criterion":
        out = q_criterion(vel, grid.spacing)
    else:
        raise KeyError(f"unknown scalar {name!r}; choose from {SCALARS}")
    out = out.astype(np.float32)
    out[solid] = 0.0
    return out, grid


def upsample_vector(vel: np.ndarray, factor: int) -> np.ndarray:
    return np.stack(
        [ndimage.zoom(c, factor, order=3, mode="nearest", grid_mode=True) for c in vel]
    ).astype(np.float32)


def upsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour refinement: keeps building walls on the original faces."""
    return mask.repeat(factor, 0).repeat(factor, 1).repeat(factor, 2)


def robust_range(
    values: np.ndarray, lo: float = 1.0, hi: float = 99.5
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(finite, [lo, hi])
    return float(a), float(b if b > a else a + 1e-6)


def open_fields(path: str | pathlib.Path, **kwargs: Any) -> FieldSeries:
    return FieldSeries(xr.open_dataset(path), **kwargs)


__all__ = [
    "Grid",
    "FieldSeries",
    "trilinear",
    "scalar_field",
    "robust_range",
    "SCALARS",
    "open_fields",
]
