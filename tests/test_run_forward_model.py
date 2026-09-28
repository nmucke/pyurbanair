"""Pairwise smoke coverage of the forward runner's switches on both backends.

Four representative runs cover both backends, static/dynamic parameters,
single/ensemble execution, and state carry. Backend numerical contracts are
tested separately; repeating the full Cartesian product adds solver startup
cost without a distinct assertion. All runs use the independent tests/conf tree.
"""

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.integration


def _overrides(
    model: str,
    params: str,
    rollout_steps: int,
    ensemble: bool,
    tmp_path: pathlib.Path,
) -> Any:
    overrides = [
        f"model={model}",
        f"params={params}",
        f"run.rollout_steps={rollout_steps}",
        f"run.ensemble={str(ensemble).lower()}",
        "run.skip_viz=true",
        # Concrete dirs so the script composes without a live HydraConfig.
        f"paths.experiment_dir={tmp_path / 'experiment'}",
        f"++paths.base_results_dir={tmp_path / 'results'}",
    ]
    if model == "pylbm":
        overrides.append("model.forward_model.cuda=false")
    return overrides


@pytest.mark.parametrize(  # type: ignore[misc]
    "model,params,rollout_steps,ensemble",
    [
        pytest.param("pylbm", "static", 0, False, id="lbm-static-single"),
        pytest.param("pylbm", "dynamic", 1, True, id="lbm-dynamic-ensemble-carry"),
        pytest.param("pyudales", "static", 0, True, id="udales-static-ensemble"),
        pytest.param("pyudales", "dynamic", 1, False, id="udales-dynamic-single-carry"),
    ],
)
def test_run_forward_model(
    model: str,
    params: str,
    rollout_steps: int,
    ensemble: bool,
    tmp_path: pathlib.Path,
    compose_test_cfg: Any,
) -> None:
    """Exercise real solver wiring and check the persisted dynamic run contract."""
    from scripts.run_forward_model import run

    run(compose_test_cfg(_overrides(model, params, rollout_steps, ensemble, tmp_path)))
    if params == "dynamic":
        state_path = next((tmp_path / "results").rglob("state.nc"))
        with xr.open_dataset(state_path) as state:
            assert {"u", "v", "w"} <= set(state.data_vars)
            assert state.sizes["time"] >= 2 * (rollout_steps + 1)
            assert np.all(np.diff(state.time.values) > 0)
        with xr.open_dataset(state_path.with_name("params.nc")) as parameters:
            assert "time" in parameters.dims
            assert "velocity_magnitude" in parameters
