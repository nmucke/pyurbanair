import pathlib
import sys

import pytest

# Patch scripts.config to use tests.config before any script imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config


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
    model: str, use_results_dir: bool, tmp_path: pathlib.Path
) -> None:
    """Test run_rollout_forward_model.py with pylbm and pyudales backends."""
    from scripts.run_rollout_forward_model import main

    # Set argv for argparse
    original_argv = sys.argv
    argv = ["run_rollout_forward_model", "--model", model, "--skip-viz"]
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        argv.extend(["--results-dir", str(results_dir)])
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = original_argv
