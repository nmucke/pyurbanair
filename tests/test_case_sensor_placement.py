"""Held-out sensor placement, checked against the voxelized geometry.

The validation sensors in ``conf/case/*.yaml`` only earn their name if they are
both *scoreable* and *worth scoring*: a sensor inside a building samples a masked
cell and its score is meaningless, and a sensor in the same regime as the
assimilated ones turns the held-out score into a near-duplicate of the
assimilated one. Neither property is visible in the coordinates, so both are
checked against the same solid-occupancy mask the solver sees, at the case grid
and at the grids the sweeps use.

Lives apart from ``test_hydra_config.py`` because it voxelizes the STL (seconds,
and it drags in the pylbm submodule), while that module is meant to stay a
cheap config-composition check.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_validation_points,
)

# (nx, ny, nz, x_bounds) — the case's own shape is composed from the config; these
# are the *other* grids the same sensor set has to survive, because `size/` and
# the local/HPC sweep launchers rescale the domain under a fixed sensor layout.
# The x-crop matters as much as the resolution: cropping to x<=40 excludes the
# tallest blocks of the array, so a sensor that clears the canopy only on the
# crop is not an above-canopy sensor.
_SWEEP_GRIDS = [
    (50, 40, 16, (-20.0, 80.0)),
    (60, 80, 16, (-20.0, 80.0)),
    (60, 80, 16, (-20.0, 40.0)),
]


def _case_cfg() -> DictConfig:
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name="run_forward_model", overrides=[])


def _held_out(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The case's held-out points, or fail the test if it defines none."""
    validation = create_validation_points(cfg.obs)
    assert validation is not None, "the default case must define held-out sensors"
    # The helper is untyped, so re-wrap rather than pass its Any straight through.
    val_x, val_y, val_z = validation
    return np.asarray(val_x), np.asarray(val_y), np.asarray(val_z)


def _roof_and_solid(
    cfg: DictConfig, nx: int, ny: int, nz: int, x_bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """``(solid, roof, cell_centers)`` for one grid over the case's geometry."""
    from pylbm.stl_to_lbm import compute_solid_occupancy

    (_, _), (ymin, ymax), (zmin, zmax) = (tuple(axis) for axis in cfg.domain.bounds)
    xmin, xmax = x_bounds
    # stl_path is repo-relative, so the test does not depend on the cwd.
    stl_path = pathlib.Path(__file__).resolve().parents[1] / cfg.geometry.stl_path
    solid = compute_solid_occupancy(
        stl_path=stl_path,
        nx=nx,
        ny=ny,
        nz=nz,
        domain_bounds={
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        },
    )
    dz = (zmax - zmin) / nz
    centers = [
        (np.arange(n) + 0.5) * ((hi - lo) / n) + lo
        for n, lo, hi in ((nx, xmin, xmax), (ny, ymin, ymax), (nz, zmin, zmax))
    ]
    # The vertical mask is filled from the floor up, so a column's solid-cell
    # count is its roof height.
    return solid, solid.sum(axis=2) * dz, centers


def _cell(
    centers: list[np.ndarray], point: tuple[float, float, float]
) -> tuple[int, ...]:
    return tuple(int(np.argmin(np.abs(axis - c))) for axis, c in zip(centers, point))


def test_case_defines_held_out_sensors() -> None:
    cfg = _case_cfg()
    assert (
        len({len(axis) for axis in _held_out(cfg)}) == 1
    ), "ragged validation_*_points"


def test_held_out_sensors_are_not_co_located_with_assimilated_ones() -> None:
    """Held out means genuinely elsewhere, not a shadow of an assimilated sensor."""
    cfg = _case_cfg()
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)
    val_x, val_y, val_z = _held_out(cfg)

    for x, y, z in zip(val_x.tolist(), val_y.tolist(), val_z.tolist()):
        horizontal = float(np.hypot(obs_x - x, obs_y - y).min())
        vertical = float(np.abs(obs_z - z).min())
        # 4 m is a few cells at every grid here. The tightest real pair is the
        # lane sensor at (20, 55), exactly 5 m from the assimilation sensor at
        # (20, 50), so the bound is not tuned to the boundary.
        assert (
            horizontal >= 4.0 or vertical >= 4.0
        ), f"held-out sensor ({x}, {y}, {z}) sits on top of an assimilation sensor"


@pytest.mark.parametrize("grid", _SWEEP_GRIDS)  # type: ignore[misc]
def test_held_out_sensors_are_open_air_on_every_grid(
    grid: tuple[int, int, int, tuple[float, float]],
) -> None:
    """Every held-out sensor is a fluid cell, at the case grid and the sweep grids."""
    cfg = _case_cfg()
    val_x, val_y, val_z = _held_out(cfg)
    grids = [
        (
            int(cfg.domain.nx),
            int(cfg.domain.ny),
            int(cfg.domain.nz),
            tuple(cfg.domain.bounds[0]),
        ),
        grid,
    ]

    for nx, ny, nz, x_bounds in grids:
        solid, roof, centers = _roof_and_solid(cfg, nx, ny, nz, x_bounds)
        for point in zip(val_x.tolist(), val_y.tolist(), val_z.tolist()):
            i, j, k = _cell(centers, point)
            assert not solid[i, j, k], (
                f"held-out sensor {point} is inside a building at "
                f"{nx}x{ny}x{nz} over x={x_bounds}"
            )
            assert roof[i, j] < point[2], (
                f"held-out sensor {point} is below its own roof at "
                f"{nx}x{ny}x{nz} over x={x_bounds}"
            )


def test_one_held_out_sensor_clears_the_whole_canopy() -> None:
    """At least one held-out sensor probes above-canopy flow.

    Checked against the TALLEST roof in the domain, not the sensor's own column:
    a sensor a few metres over a short block is still inside the canopy, and the
    assimilation sensors already cover in-canopy street level. The x-crop is the
    trap here — the array's tallest blocks (17.2 m in the STL) sit beyond x=40,
    so a criterion that passes only on a cropped domain proves nothing.
    """
    cfg = _case_cfg()
    _, _, val_z = _held_out(cfg)
    for nx, ny, nz, x_bounds in _SWEEP_GRIDS:
        _, roof, _ = _roof_and_solid(cfg, nx, ny, nz, x_bounds)
        assert val_z.max() > roof.max(), (
            f"no held-out sensor is above the canopy at {nx}x{ny}x{nz} over "
            f"x={x_bounds} (tallest roof {roof.max():.1f} m, highest sensor "
            f"{val_z.max():.1f} m)"
        )
