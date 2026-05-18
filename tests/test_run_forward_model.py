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
def test_run_forward_model(
    model: str,
    use_results_dir: bool,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Test run_forward_model.py with pylbm and pyudales backends."""
    from scripts.run_forward_model import run

    model_override = f"model={model}"
    overrides = [model_override, "run.skip_viz=true"]
    if model == "pylbm":
        overrides.append("model.forward_model.cuda=false")
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        overrides.append(f"run.results_dir={results_dir}")

    run(compose_test_cfg(overrides))
