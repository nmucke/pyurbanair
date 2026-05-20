import pathlib

import pytest


@pytest.mark.parametrize(
    "model,use_results_dir",
    [
        pytest.param("pylbm", False, id="pylbm_no_results_dir"),
        pytest.param("pylbm", True, id="pylbm_results_dir"),
        pytest.param("pyudales", False, id="pyudales_no_results_dir"),
        pytest.param("pyudales", True, id="pyudales_results_dir"),
    ],
)
def test_run_rollout_forward_model(
    model: str,
    use_results_dir: bool,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Test run_rollout_forward_model.py with pylbm and pyudales backends."""
    from scripts.run_rollout_forward_model import run

    overrides = [
        f"model={model}",
        "run.skip_viz=true",
        "run.num_steps=2",
        f"paths.base_results_dir={tmp_path}",
    ]
    if model == "pylbm":
        overrides.append("model.forward_model.cuda=false")
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        overrides.append(f"run.results_dir={results_dir}")

    run(compose_test_cfg(overrides))
