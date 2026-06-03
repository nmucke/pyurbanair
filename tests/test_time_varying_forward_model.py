import pathlib

import pytest


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("pylbm", id="pylbm"),
        pytest.param("pyudales", id="pyudales"),
    ],
)
def test_run_time_varying_forward_model(
    model: str,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Test run_forward_model.py (run.time_varying=true) runs to completion."""
    from scripts.run_forward_model import run

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    overrides = [
        f"model={model}",
        "run.skip_viz=true",
        "run.time_varying=true",
        f"run.results_dir={results_dir}",
        f"paths.base_results_dir={tmp_path}",
    ]
    if model == "pylbm":
        overrides.append("model.forward_model.cuda=false")

    run(compose_test_cfg(overrides))
