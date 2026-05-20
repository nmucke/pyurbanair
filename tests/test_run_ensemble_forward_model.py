import pathlib
import shutil

import pytest


@pytest.mark.parametrize(  # type: ignore[misc]
    "model,use_results_dir,parallel_execution",
    [
        pytest.param("pylbm", False, False, id="pylbm_no_results_dir_sequential"),
        pytest.param("pylbm", True, False, id="pylbm_results_dir_sequential"),
        pytest.param("pylbm", False, True, id="pylbm_no_results_dir_parallel"),
        pytest.param("pylbm", True, True, id="pylbm_results_dir_parallel"),
        pytest.param("pyudales", False, False, id="pyudales_no_results_dir_sequential"),
        pytest.param("pyudales", True, False, id="pyudales_results_dir_sequential"),
        pytest.param("pyudales", False, True, id="pyudales_no_results_dir_parallel"),
        pytest.param("pyudales", True, True, id="pyudales_results_dir_parallel"),
    ],
)
def test_run_ensemble_forward_model(
    model: str,
    use_results_dir: bool,
    parallel_execution: bool,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Test run_ensemble_forward_model.py with pylbm and pyudales backends.

    Covers both sequential (num_parallel_processes=1) and parallel
    (num_parallel_processes>1) execution paths.
    """
    from scripts.run_ensemble_forward_model import run

    overrides = [
        f"model={model}",
        "run.skip_viz=true",
        f"ensemble.num_parallel_processes={2 if parallel_execution else 1}",
    ]
    if model == "pylbm":
        overrides.append("model.forward_model.cuda=false")
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        overrides.append(f"run.results_dir={results_dir}")

    shutil.rmtree(tmp_path)

    run(compose_test_cfg(overrides))
