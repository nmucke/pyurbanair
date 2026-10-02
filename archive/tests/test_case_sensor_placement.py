"""The frozen smoke sensors must be usable in the frozen smoke geometry."""

import pathlib
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from tests.config_loader import TEST_CONF_DIR


def test_smoke_sensors_are_distinct_in_bounds_and_in_fluid(
    compose_test_cfg: Any,
) -> None:
    from pylbm.stl_to_lbm import compute_solid_occupancy

    cfg = compose_test_cfg()
    bounds = np.asarray(cfg.domain.bounds)
    shape = np.array([cfg.domain.nx, cfg.domain.ny, cfg.domain.nz])
    points = np.column_stack([cfg.obs[f"{axis}_points"] for axis in "xyz"])
    held_out = np.column_stack([cfg.obs[f"validation_{axis}_points"] for axis in "xyz"])
    assert not set(map(tuple, points)) & set(map(tuple, held_out))
    all_points = np.concatenate([points, held_out])
    assert np.all(all_points >= bounds[:, 0])
    assert np.all(all_points < bounds[:, 1])
    solid = compute_solid_occupancy(
        stl_path=pathlib.Path(__file__).resolve().parents[1] / cfg.geometry.stl_path,
        nx=int(shape[0]),
        ny=int(shape[1]),
        nz=int(shape[2]),
        domain_bounds={
            f"{axis}{side}": float(bounds[i, j])
            for i, axis in enumerate("xyz")
            for j, side in enumerate(("min", "max"))
        },
    )
    cells = ((all_points - bounds[:, 0]) / np.diff(bounds).ravel() * shape).astype(int)
    assert not solid[tuple(cells.T)].any()


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


def test_reference_held_out_sensors_survive_resolution_and_domain_crops() -> None:
    """Retain the scientific placement regression without production run tuning."""
    cfg = OmegaConf.load(TEST_CONF_DIR / "sensor_layout.yaml")
    assert isinstance(cfg, DictConfig)
    validation = np.column_stack(
        [cfg.obs[f"validation_{axis}_points"] for axis in "xyz"]
    )
    observed = np.column_stack([cfg.obs[f"{axis}_points"] for axis in "xyz"])
    assert not set(map(tuple, observed)) & set(map(tuple, validation))
    for nx, ny, nz, x_bounds in (
        (50, 40, 16, (-20.0, 80.0)),
        (60, 80, 16, (-20.0, 80.0)),
        (60, 80, 16, (-20.0, 40.0)),
    ):
        solid, roof, centers = _roof_and_solid(cfg, nx, ny, nz, x_bounds)
        assert validation[:, 2].max() > roof.max()
        for point in validation:
            i, j, k = _cell(centers, tuple(point))
            assert not solid[i, j, k], (point, nx, ny, nz)
            assert roof[i, j] < point[2]
