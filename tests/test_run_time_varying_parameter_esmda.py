import pathlib
import sys

import pytest

# Patch scripts.config to use tests.config before any script imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model,use_results_dir",
    [
        pytest.param("pylbm", "pylbm", False, id="pylbm_pylbm_in_memory"),
        pytest.param("pylbm", "pylbm", True, id="pylbm_pylbm_on_disk"),
        pytest.param("pyudales", "pyudales", False, id="pyudales_pyudales_in_memory"),
        pytest.param("pyudales", "pyudales", True, id="pyudales_pyudales_on_disk"),
        pytest.param("pylbm", "pyudales", False, id="pylbm_pyudales_cross"),
        pytest.param("pyudales", "pylbm", False, id="pyudales_pylbm_cross"),
    ],
)
def test_run_time_varying_parameter_esmda(
    truth_model: str,
    assim_model: str,
    use_results_dir: bool,
    tmp_path: pathlib.Path,
) -> None:
    """Test run_time_varying_parameter_esmda.py with various model and storage combinations."""
    from scripts.run_time_varying_parameter_esmda import main

    original_argv = sys.argv
    original_ensemble = tests_config.ENSEMBLE.copy()
    original_esmda = tests_config.ESMDA.copy()
    original_tv = tests_config.TIME_VARYING_PARAMS.copy()

    argv = [
        "run_time_varying_parameter_esmda",
        "--truth-model",
        truth_model,
        "--assim-model",
        assim_model,
        "--skip-viz",
        "--num-par-time-points",
        "3",
    ]
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        argv.extend(["--results-dir", str(results_dir)])
    sys.argv = argv

    try:
        tests_config.ENSEMBLE["ensemble_size"] = 2
        tests_config.ENSEMBLE["num_parallel_processes"] = 2
        tests_config.ESMDA["num_steps"] = 1
        tests_config.ESMDA["num_assimilation_windows"] = 1

        main()
    finally:
        sys.argv = original_argv
        tests_config.ENSEMBLE.update(original_ensemble)
        tests_config.ESMDA.update(original_esmda)
        tests_config.TIME_VARYING_PARAMS.update(original_tv)
