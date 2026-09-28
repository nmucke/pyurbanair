"""Pairwise smoke coverage of the forward runner's switches on both backends.

Each backend exercises every pair of static/dynamic parameters, single/ensemble
execution and cold/rollout runs. One continuation window is enough to exercise
warm starts and concatenation; the full Cartesian product repeats those paths.
"""

import pathlib

import pytest


def _overrides(
    model: str,
    params: str,
    rollout_steps: int,
    ensemble: bool,
    tmp_path: pathlib.Path,
):
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


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("pylbm", id="pylbm"),
        pytest.param("pyudales", id="pyudales"),
    ],
)
@pytest.mark.parametrize(
    "params,rollout_steps,ensemble",
    [
        pytest.param("static", 0, False, id="static-single"),
        pytest.param("static", 1, True, id="static-ensemble-rollout"),
        pytest.param("dynamic", 1, False, id="dynamic-single-rollout"),
        pytest.param("dynamic", 0, True, id="dynamic-ensemble"),
    ],
)
def test_run_forward_model(
    model: str,
    params: str,
    rollout_steps: int,
    ensemble: bool,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Run representative switch combinations through the real solver."""
    from scripts.run_forward_model import run

    run(compose_test_cfg(_overrides(model, params, rollout_steps, ensemble, tmp_path)))
