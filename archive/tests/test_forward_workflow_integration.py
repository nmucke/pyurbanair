"""Real backend wiring for complete forward artifacts on the frozen tiny case."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

from pyurbanair.workflows.forward import run

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backend", ["pylbm", "pyudales", "pypalm"])  # type: ignore[misc, unused-ignore]
def test_backend_complete_artifacts(
    backend: str, tmp_path: Path, compose_test_cfg: Any
) -> None:
    overrides = [f"model={backend}", "params=static", "run.skip_viz=true"]
    if backend == "pylbm":
        overrides.append("model.forward_model.cuda=false")
    if backend == "pypalm":
        # PALM 25.10 refuses nz < 14, unlike the other tiny-case backends.
        overrides.append("domain.nz=16")
    cfg = compose_test_cfg(overrides)
    root = run(cfg, complete_artifacts=True, output_dir=tmp_path)
    index = json.loads((root / "artifact_index.json").read_text())
    assert index["status"] == "complete"
    with xr.open_dataset(root / "state.nc") as ds:
        assert {"u", "v", "w"} <= set(ds.data_vars)
        assert ds.sizes["time"] > 0
        assert np.isfinite(ds.u.values).any()
    with xr.open_dataset(root / "params.nc") as ds:
        assert "velocity_magnitude" in ds


def test_surrogate_checkpoint_complete_artifacts(
    tmp_path: Path, compose_test_cfg: Any, surrogate_model_dir_factory: Any
) -> None:
    """A fixture checkpoint exercises deployment, not trained predictive quality."""
    cfg = compose_test_cfg(
        [
            "model=neural_surrogate",
            "params=static",
            "run.skip_viz=true",
            "model.forward_model.device=cpu",
            "time.simulation_time=1.0",
            "run.rollout_steps=2",
        ]
    )
    model_dir = surrogate_model_dir_factory(
        tmp_path / "checkpoint",
        domain=OmegaConf.to_container(cfg.domain, resolve=True),
        time={"output_frequency": 1.0},
        num_history_steps=2,
    )
    cfg.model.forward_model.model_dir = str(model_dir)
    coords: dict[str, Any] = {"time": [0.0, 1.0]}
    for axis, bounds in zip("xyz", cfg.domain.bounds):
        size = cfg.domain[f"n{axis}"]
        coords[axis] = np.linspace(bounds[0], bounds[1], size, endpoint=False) + (
            bounds[1] - bounds[0]
        ) / (2 * size)
    shape = (2, cfg.domain.nz, cfg.domain.ny, cfg.domain.nx)
    initial = xr.Dataset(
        {
            v: (("time", "z", "y", "x"), np.ones(shape, dtype=np.float32))
            for v in ("u", "v", "w")
        },
        coords=coords,
    )
    root = run(
        cfg,
        complete_artifacts=True,
        initial_state=initial,
        output_dir=tmp_path / "results",
    )
    with xr.open_dataset(root / "state.nc") as ds:
        assert ds.sizes["time"] == 3
        np.testing.assert_equal(ds.time.values, [1.0, 2.0, 3.0])
        assert np.isfinite(ds.u.values).all()
    assert (
        json.loads((root / "artifact_index.json").read_text())["status"] == "complete"
    )
