import pathlib
import shutil
import sys

import pytest

# Patch scripts.config to use tests.config before any script imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config


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
def test_run_ensemble_rollout_forward_model(
    model: str, use_results_dir: bool, parallel_execution: bool, tmp_path: pathlib.Path
) -> None:
    """Test run_ensemble_rollout_forward_model.py with pylbm and pyudales backends.

    Covers both sequential (num_parallel_processes=1) and parallel
    (num_parallel_processes>1) execution paths.
    """
    from scripts.run_ensemble_rollout_forward_model import main

    # Set argv for argparse
    original_argv = sys.argv
    argv = [
        "run_ensemble_rollout_forward_model",
        "--model",
        model,
        "--skip-viz",
        "--num-steps",
        "2",
    ]
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        argv.extend(["--results-dir", str(results_dir)])
    sys.argv = argv

    original_num_parallel = tests_config.ENSEMBLE["num_parallel_processes"]
    try:
        shutil.rmtree(tmp_path, ignore_errors=True)

        tests_config.ENSEMBLE["num_parallel_processes"] = 2 if parallel_execution else 1
        main()

    finally:
        sys.argv = original_argv
        tests_config.ENSEMBLE["num_parallel_processes"] = original_num_parallel
